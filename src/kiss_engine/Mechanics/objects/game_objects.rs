//******************************************************************/
//
// GameObjectManager — thin orchestrator for all interactive game objects.
//
// Object structs, parsing, and per-frame logic live in their own modules
// (CBarrel.rs, CDoorSliding.rs, etc.).  This file owns:
//   * The `GameObject` enum wrapping every concrete type.
//   * `GameObjectManager` — the active object list.
//   * Per-frame `update()`, `interact()`, `apply_area_damage()`,
//     `apply_point_damage()`, and `dynamic_lights()`.
//
//******************************************************************/

use crate::abc::PlacedAbcObject;
use crate::dat::{PropertyValue, WorldObject};
use crate::LightObj::Light;
use crate::object_utils::{dist3, hide_draw_group, time_to_fraction};
use crate::types::DrawGroup;
use crate::util::math::*;

use crate::CBarrel::{self, BarrelObject, BarrelState, EXPLOSION_FLASH_DURATION};
use crate::CCrate::{self, CrateObject, CrateState};
use crate::CDoorSliding::{self, DoorObject};
use crate::CLadder::{self, LadderObject};
use crate::CRotatingCeilingFan::{self, FanObject};
use crate::CSwitchRotating::{self, SwitchObject, INTERACT_RADIUS};
use crate::CTorch::{self, SpriteEffectObject, TorchObject};
use crate::CWater::{self, WaterObject};
use crate::CWindow::{self, WindowObject, WindowState};
use crate::DemoSkyWorldModel::{SkyModelInfo, SkyWorldModelObject};
use crate::OutsideDef::OutsideDefObject;
use crate::CPickupItem::{self, PickupItemObject};
use crate::CCreature::{self, CreatureObject};
use crate::scripted_sequence::{BspDoor, BspDoorState, DoorOp, ScriptCommand, ScriptRunner};
use crate::triggers::{TriggerActivation, TriggerDef};

//******************************************************************/
//
// Explosion effect
//
//******************************************************************/

/// Transient explosion light flash managed by the object system.
/// The App merges these into the frame's lighting UBO each frame.
#[derive(Debug, Clone)]
pub struct ExplosionLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub radius: f32,
    pub time_remaining: f32,
}

impl ExplosionLight {
    /// Current intensity, fades linearly to zero over its lifetime.
    pub fn intensity(&self) -> f32 {
        (time_to_fraction(self.time_remaining, EXPLOSION_FLASH_DURATION) * 4.0).min(4.0)
    }

    /// Convert to a `lights::Light` for inclusion in the lighting UBO.
    pub fn to_light(&self) -> Light {
        Light {
            position: self.position,
            radius: self.radius,
            color: self.color,
            intensity: self.intensity(),
        }
    }
}

//******************************************************************/
//
// Game Object enum
//
//******************************************************************/

pub enum GameObject {
    Barrel(BarrelObject),
    Crate(CrateObject),
    Door(DoorObject),
    Switch(SwitchObject),
    Torch(TorchObject),
    SpriteEffect(SpriteEffectObject),
    Fan(FanObject),
    Window(WindowObject),
    Water(WaterObject),
    Ladder(LadderObject),
    SkyModel(SkyWorldModelObject),
    Outside(OutsideDefObject),
    Pickup(PickupItemObject),
    Creature(CreatureObject),
    Headless(crate::headless_enemy::Headless),
}

impl GameObject {
    pub fn position(&self) -> Option<[f32; 3]> {
        match self {
            Self::Barrel(o) => Some(o.position),
            Self::Crate(o) => Some(o.position),
            Self::Door(o) => Some(o.position),
            Self::Switch(o) => Some(o.position),
            Self::Torch(o) => Some(o.position),
            Self::SpriteEffect(o) => Some(o.position),
            Self::Fan(o) => Some(o.position),
            Self::Window(o) => Some(o.position),
            Self::Pickup(o) => Some(o.position),
            Self::Creature(o) => Some(o.position),
            Self::Headless(o) => Some(o.position),
            Self::Water(_) | Self::Ladder(_) | Self::SkyModel(_) | Self::Outside(_) => None,
        }
    }
}

//******************************************************************/
//
// Game Object Manager
//
//******************************************************************/

/// A BSP trigger volume that activates a script when the player presses E inside it.
#[derive(Debug, Clone)]
pub struct ScriptTrigger {
    pub name: String,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub actions: Vec<ScriptCommand>,
    pub activated: bool,
}

#[derive(Debug, Clone)]
pub struct PendingLevelTransition {
    pub next_world: Option<String>,
    pub start_point: String,
}

impl ScriptTrigger {
    pub fn contains(&self, pos: [f32; 3]) -> bool {
        pos[0] >= self.min[0] && pos[0] <= self.max[0]
            && pos[1] >= self.min[1] && pos[1] <= self.max[1]
            && pos[2] >= self.min[2] && pos[2] <= self.max[2]
    }
}

//******************************************************************/
//
// A BSP-based switch (CSwitchSlide / CSwitchRotating) — geometry is a world model, not an ABC.
//
//******************************************************************/

#[derive(Debug, Clone)]
pub struct BspSwitch {
    pub name: String,
    pub center: [f32; 3],
    /// Commands this switch activates, resolved from DAT command properties.
    pub actions: Vec<ScriptCommand>,
    pub activated: bool,
    pub draw_groups: Vec<usize>,
}

pub struct GameObjectManager {
    pub objects: Vec<GameObject>,
    pub explosion_lights: Vec<ExplosionLight>,
    pub player_in_water: bool,
    pub player_on_ladder: bool,
    pub player_outside: bool,
    pub bsp_doors: Vec<BspDoor>,
    pub scripts: Vec<(String, ScriptRunner)>,
    pub triggers: Vec<TriggerDef>,
    pub script_triggers: Vec<ScriptTrigger>,
    pub bsp_switches: Vec<BspSwitch>,
    pub pending_level_transition: Option<PendingLevelTransition>,
    // Reusable temporaries to avoid per-frame heap allocations
    pub tmp_pending: Vec<([f32; 3], f32, f32, [f32; 3])>,
    pub tmp_fired_commands: Vec<ScriptCommand>,
    pub tmp_targets_to_open: Vec<String>,
    pub tmp_bsp_switch_targets: Vec<ScriptCommand>,
    pub tmp_trigger_script_targets: Vec<ScriptCommand>,
    pub tmp_newly_exploding: Vec<usize>,
}

impl GameObjectManager {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            explosion_lights: Vec::new(),
            player_in_water: false,
            player_on_ladder: false,
            player_outside: false,
            bsp_doors: Vec::new(),
            scripts: Vec::new(),
            triggers: Vec::new(),
            script_triggers: Vec::new(),
            bsp_switches: Vec::new(),
            pending_level_transition: None,
            tmp_pending: Vec::new(),
            tmp_fired_commands: Vec::new(),
            tmp_targets_to_open: Vec::new(),
            tmp_bsp_switch_targets: Vec::new(),
            tmp_trigger_script_targets: Vec::new(),
            tmp_newly_exploding: Vec::new(),
        }
    }

    /// Build the manager from DAT objects, placed ABC objects, BSP sub-models,
    /// torch flame entries, and pre-collected sky model infos.
    pub fn extract_from_dat(
        dat_objects: &[WorldObject],
        placed: &[PlacedAbcObject],
        first_draw_group: usize,
        bsp_submodels: &[(String, [f32; 3], Vec<usize>, f32)],
        torch_flames: &[(usize, usize, usize)],
        sky_models: &[SkyModelInfo],
        trigger_volumes: &[(String, [f32; 3], [f32; 3])],
        door_collision_ranges: &[(String, usize, usize)],
        collision_positions: &[Vector3],
        scale: f32,
        creature_anim_data: &[(usize, Vec<u32>, Vec<u32>, u32)],
    ) -> Self {
        let mut mgr = Self::new();

        for (i, abc) in placed.iter().enumerate() {
            let dg = first_draw_group + i;
            let props = dat_objects.get(abc.dat_object_index);
            let pos = abc.position;

            match abc.type_name.as_str() {
                "CBarrel" => {
                    mgr.objects.push(GameObject::Barrel(
                        CBarrel::parse(pos, props, dg, &abc.skin_filename, scale),
                    ));
                }
                "CCrate" | "CModelBreakable" => {
                    mgr.objects.push(GameObject::Crate(CCrate::parse(pos, props, dg)));
                }
                "CDoorSliding" => {
                    let mesh_bounds = if !abc.mesh.vertices.is_empty() {
                        let mut bmin = [f32::MAX; 3];
                        let mut bmax = [f32::MIN; 3];
                        for v in &abc.mesh.vertices {
                            for k in 0..3 {
                                bmin[k] = bmin[k].min(v.pos[k]);
                                bmax[k] = bmax[k].max(v.pos[k]);
                            }
                        }
                        Some((bmin, bmax))
                    } else {
                        None
                    };
                    mgr.objects.push(GameObject::Door(CDoorSliding::parse(pos, props, dg, scale, mesh_bounds)));
                }
                "CSwitchRotating" => {
                    let sw = CSwitchRotating::parse_rotating(pos, props, dg);
                    println!("  Switch at ({:.2},{:.2},{:.2}) → target='{}'", pos[0], pos[1], pos[2], sw.target_name);
                    mgr.objects.push(GameObject::Switch(sw));
                }
                "CSwitchSlide" => {
                    let sw = CSwitchRotating::parse_slide(pos, props, dg);
                    println!("  SwitchSlide at ({:.2},{:.2},{:.2}) → target='{}'", pos[0], pos[1], pos[2], sw.target_name);
                    mgr.objects.push(GameObject::Switch(sw));
                }
                "CTorch" => {
                    let (flame_draw_group, flame_texture_index) = torch_flames
                        .iter()
                        .find(|&&(abc_index, _, _)| abc_index == i)
                        .map(|&(_, draw_group, texture_index)| (draw_group, texture_index))
                        .unwrap_or((0, 0));
                    mgr.objects.push(GameObject::Torch(CTorch::parse(
                        pos,
                        props,
                        dg,
                        flame_draw_group,
                        flame_texture_index,
                    )));
                }
                "CWindow" | "CWindowShattering" => {
                    mgr.objects.push(GameObject::Window(CWindow::parse(pos, props, dg)));
                }
                "CWater" | "CWaterVolume" => {
                    mgr.objects.push(GameObject::Water(CWater::parse(pos, props, scale)));
                }
                "CLadder" => {
                    mgr.objects.push(GameObject::Ladder(CLadder::parse(pos, props, scale)));
                }
                "OutsideDef" => {
                    use crate::dat::PropertyValue;
                    let half = props
                        .and_then(|o| o.get_property("Dims"))
                        .and_then(|v| {
                            if let PropertyValue::Vector(vec) = v {
                                Some([vec.x * scale * 0.5, vec.z * scale * 0.5, vec.y * scale * 0.5])
                            } else {
                                None
                            }
                        })
                        .unwrap_or([5.0, 5.0, 5.0]);
                    mgr.objects.push(GameObject::Outside(OutsideDefObject {
                        min: [pos[0] - half[0], pos[1] - half[1], pos[2] - half[2]],
                        max: [pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]],
                    }));
                }
                _ => {
                    // Pickup items (spinning/bobbing animation)
                    if abc.type_name.ends_with("Item_t") || abc.type_name == "CPickupTrigger" {
                        mgr.objects.push(GameObject::Pickup(
                            CPickupItem::parse(pos, dg),
                        ));
                    }
                    // Creatures (static placement for now)
                    else if abc.type_name.starts_with("CHeadless") {
                        // Create a Headless enemy object that wraps a CreatureObject
                        let mut head = crate::headless_enemy::Headless::new(pos, Some(dg));
                        // Attach creature animation data if available so idle animation plays
                        let mut creature = CCreature::parse(pos, dg, &abc.type_name);
                        if let Some((_, kf_indices, kf_times, duration)) =
                            creature_anim_data.iter().find(|(idx, _, _, _)| *idx == i)
                        {
                            let index_count = abc.mesh.indices.len() as u32;
                            CCreature::set_animation(
                                &mut creature,
                                kf_indices.clone(),
                                index_count,
                                kf_times,
                                *duration,
                            );
                        }
                        head.set_creature(creature);
                        mgr.objects.push(GameObject::Headless(head));
                    }
                    else if abc.type_name.starts_with("CArachniclown")
                        || abc.type_name.starts_with("CBallBuster")
                        || abc.type_name.starts_with("CBatwing")
                        || abc.type_name.starts_with("CBlackwell")
                        || abc.type_name.starts_with("CBladeMaster")
                        || abc.type_name.starts_with("CFatLady")
                        || abc.type_name.starts_with("CGasBag")
                        || abc.type_name.starts_with("CGrinder")
                        || abc.type_name.starts_with("CHellSpore")
                        || abc.type_name.starts_with("CLarva")
                        || abc.type_name.starts_with("CMeanieBeanie")
                        || abc.type_name.starts_with("CPin")
                        || abc.type_name.starts_with("CRotCrawl")
                        || abc.type_name.starts_with("CStrongman")
                        || abc.type_name.starts_with("CStrutter")
                        || abc.type_name.starts_with("CStump")
                        || abc.type_name.starts_with("CTiberius")
                        || abc.type_name.starts_with("CTickler")
                        || abc.type_name.starts_with("CUniPsycho")
                        || abc.type_name.starts_with("CStarGrave")
                        || abc.type_name.starts_with("CFortunado")
                        || abc.type_name.starts_with("CRoly")
                    {
                        let mut creature = CCreature::parse(pos, dg, &abc.type_name);
                        // Set up animation data if available
                        if let Some((_, kf_indices, kf_times, duration)) =
                            creature_anim_data.iter().find(|(idx, _, _, _)| *idx == i)
                        {
                            let index_count = abc.mesh.indices.len() as u32;
                            CCreature::set_animation(
                                &mut creature,
                                kf_indices.clone(),
                                index_count,
                                kf_times,
                                *duration,
                            );
                        }
                        mgr.objects.push(GameObject::Creature(creature));
                    }
                }
            }
        }

        // BSP sub-models: rotating ceiling fans and doors
        for (name, pivot, dgs, z_height) in bsp_submodels {
            if name.to_ascii_uppercase().starts_with("CROTATINGCEILINGFAN") {
                mgr.objects.push(GameObject::Fan(
                    CRotatingCeilingFan::parse_from_bsp(*pivot, dgs.clone()),
                ));
            }
            let nl = name.to_lowercase();
            if nl.starts_with("door") || nl.starts_with("freezedoor") {
                let (slide_dir, slide_distance) = dat_objects
                    .iter()
                    .find(|o| {
                        matches!(o.get_property("Name"), Some(PropertyValue::String(n)) if n.to_lowercase() == nl)
                    })
                    .and_then(|o| {
                        match (o.get_property("Pos"), o.get_property("open_position")) {
                            (
                                Some(PropertyValue::Vector(pos)),
                                Some(PropertyValue::Vector(open_pos)),
                            ) => {
                                let delta = [
                                    (open_pos.x - pos.x) * scale,
                                    (open_pos.z - pos.z) * scale,
                                    (open_pos.y - pos.y) * scale,
                                ];
                                let len = (delta[0] * delta[0]
                                    + delta[1] * delta[1]
                                    + delta[2] * delta[2])
                                    .sqrt();
                                if len > 1e-5 {
                                    Some(([delta[0] / len, delta[1] / len, delta[2] / len], len))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    })
                    .unwrap_or(([0.0, 0.0, 1.0], *z_height));

                // Find the collision vertex range for this door
                let coll_range = door_collision_ranges.iter()
                    .find(|(n, _, _)| n == &nl);
                let (coll_vertex_range, coll_base) = if let Some(&(_, start, end)) = coll_range {
                    (Some((start, end)), collision_positions[start..end].to_vec())
                } else {
                    (None, Vec::new())
                };
                mgr.bsp_doors.push(BspDoor {
                    name: nl,
                    draw_groups: dgs.clone(),
                    slide_distance,
                    slide_dir,
                    slide_offset: 0.0,
                    state: BspDoorState::Closed,
                    collision_vertex_range: coll_vertex_range,
                    collision_base_positions: coll_base,
                });
            }
        }

        // Sky world models (animated)
        for info in sky_models {
            mgr.objects.push(GameObject::SkyModel(
                SkyWorldModelObject::from_info(info, [0.0, 0.0, 0.0]),
            ));
        }

        let _ = trigger_volumes;

        log::info!(
            "GameObjectManager: {} interactive + {} sky objects",
            mgr.objects.iter().filter(|o| !matches!(o, GameObject::SkyModel(_))).count(),
            mgr.objects.iter().filter(|o| matches!(o, GameObject::SkyModel(_))).count(),
        );
        mgr
    }

    /// Collect current AABBs of doors that should block the player.
    pub fn door_aabbs(&self) -> Vec<([f32; 3], [f32; 3])> {
        self.objects.iter().filter_map(|obj| {
            if let GameObject::Door(d) = obj {
                d.current_aabb()
            } else {
                None
            }
        }).collect()
    }

    //******************************************************************/
    //
    // Per-frame update
    //
    //******************************************************************/

    /// Advance all object state machines, apply model-matrix overrides to
    /// `draw_groups`, and determine player zone membership.
    ///
    /// `_time` is the total elapsed time in seconds (reserved for future per-object phase effects).
    pub fn update(
        &mut self,
        dt: f32,
        _time: f32,
        player_pos: [f32; 3],
        draw_groups: &mut Vec<DrawGroup>,
    ) {
        // Reuse temporary pending list to avoid per-frame allocation
        self.tmp_pending.clear();

        for obj in &mut self.objects {
            match obj {
                GameObject::Barrel(barrel) => {
                    if let Some(explosion) = CBarrel::update(barrel, dt) {
                        hide_draw_group(draw_groups, barrel.draw_group);
                        self.tmp_pending.push((
                            explosion.position,
                            explosion.radius,
                            explosion.damage,
                            explosion.color,
                        ));
                    }
                }
                GameObject::Door(d) => CDoorSliding::update(d, dt, player_pos, draw_groups),
                GameObject::Switch(sw) => CSwitchRotating::update(sw, dt, draw_groups),
                GameObject::Fan(fan) => CRotatingCeilingFan::update(fan, dt, draw_groups),
                GameObject::Torch(torch) => CTorch::update(torch, dt, player_pos, draw_groups),
                GameObject::SkyModel(sky) => sky.update(dt, draw_groups),
                GameObject::Pickup(p) => CPickupItem::update(p, dt, _time, draw_groups),
                GameObject::Headless(h) => {
                    // Advance AI and process emitted events
                    let events = h.update(dt, player_pos);
                    for ev in events {
                        match ev {
                            crate::headless_enemy::HeadlessEvent::ExecuteScript(name) => {
                                for (scr_name, runner) in &mut self.scripts {
                                    if scr_name == &name {
                                        runner.start();
                                    }
                                }
                            }
                            crate::headless_enemy::HeadlessEvent::DealDamage(amount) => {
                                log::info!("Headless at {:?} deals {:.1} damage to player (stub)", h.position, amount);
                            }
                            crate::headless_enemy::HeadlessEvent::Moved(m) => {
                                if let Some(dg) = h.draw_group {
                                    crate::object_utils::set_draw_group_matrix(draw_groups, dg, Some(m));
                                }
                            }
                            crate::headless_enemy::HeadlessEvent::AnimationChange(_a) => {
                                // Animation selection currently drives creature_anim playback
                            }
                        }
                    }
                    // Advance attached idle animation (if any)
                    h.update_animation(dt, draw_groups);
                }
                GameObject::Creature(c) => CCreature::update(c, dt, draw_groups),
                _ => {}
            }
        }

        // Take pending items out so we can mutably borrow self inside the loop
        let pending_items = std::mem::take(&mut self.tmp_pending);
        for (pos, radius, damage, color) in pending_items {
            self.explosion_lights.push(ExplosionLight {
                position: pos,
                color,
                radius: radius * 2.5,
                time_remaining: 0.75,
            });
            self.apply_area_damage(pos, radius, damage, draw_groups);
        }

        for el in &mut self.explosion_lights {
            el.time_remaining -= dt;
        }
        self.explosion_lights.retain(|el| el.time_remaining > 0.0);

        // Tick BSP doors
        for door in &mut self.bsp_doors {
            door.update(dt, draw_groups);
        }

        // Tick script runners and collect fired commands (reuse temp vec)
        self.tmp_fired_commands.clear();
        for (_name, runner) in &mut self.scripts {
            self.tmp_fired_commands.extend(runner.update(dt));
        }

        // Dispatch fired script commands
        let fired = std::mem::take(&mut self.tmp_fired_commands);
        for cmd in fired {
            self.dispatch_script_command(cmd, draw_groups);
        }

        self.tmp_trigger_script_targets.clear();
        for trigger in &mut self.triggers {
            if trigger.activation == TriggerActivation::Touch
                && trigger.can_fire()
                && trigger.contains(player_pos)
            {
                println!(
                    "[trigger] touch '{}' ({:?}) -> {:?}",
                    trigger.name, trigger.source, trigger.actions
                );
                trigger.activated = true;
                self.tmp_trigger_script_targets.extend(trigger.actions.clone());
            }
        }
        let touch_commands = std::mem::take(&mut self.tmp_trigger_script_targets);
        for command in touch_commands {
            self.dispatch_script_command(command, draw_groups);
        }

        self.player_in_water = false;
        self.player_on_ladder = false;
        self.player_outside = false;
        for obj in &self.objects {
            match obj {
                GameObject::Water(w) if w.contains(player_pos) => { self.player_in_water = true; }
                GameObject::Ladder(ladder) if ladder.contains(player_pos) => { self.player_on_ladder = true; }
                GameObject::Outside(o) if o.contains(player_pos) => { self.player_outside = true; }
                _ => {}
            }
        }
    }

    /// Update collision mesh vertices for any BSP doors that are sliding open.
    pub fn update_door_collision(&self, collision_positions: &mut [Vector3]) {
        for door in &self.bsp_doors {
            door.update_collision(collision_positions);
        }
    }

    // ── Interaction ──────────────────────────────────────────────────────────

    pub fn interact(&mut self, player_pos: [f32; 3], draw_groups: &mut Vec<DrawGroup>) {
        // Reuse temporary target buffers to avoid allocations during interaction
        self.tmp_targets_to_open.clear();

        for obj in &mut self.objects {
            if let GameObject::Switch(sw) = obj {
                if let Some(target) = CSwitchRotating::try_interact(sw, player_pos, draw_groups) {
                    self.tmp_targets_to_open.push(target);
                }
            }
        }

        for obj in &mut self.objects {
            if let GameObject::Door(d) = obj {
                let in_range = dist3(player_pos, d.position) < INTERACT_RADIUS;
                let switch_linked = !d.trigger_name.is_empty()
                    && self.tmp_targets_to_open.contains(&d.trigger_name);
                if in_range || switch_linked {
                    d.open();
                }
            }
        }

        // Start scripts whose trigger name matches a switch target (case-insensitive)
        let switch_targets = std::mem::take(&mut self.tmp_targets_to_open);
        for target in switch_targets {
            self.dispatch_script_command(
                ScriptCommand::StartScript {
                    script_name: target.to_lowercase(),
                },
                draw_groups,
            );
        }

        // Generic factory-built use triggers: BSP switches, trigger volumes,
        // and any future DAT-driven use targets share this path.
        self.tmp_trigger_script_targets.clear();
        for trigger in &mut self.triggers {
            if trigger.activation == TriggerActivation::Use
                && trigger.can_fire()
                && trigger.can_use_from(player_pos)
            {
                println!(
                    "[trigger] use '{}' ({:?}) -> {:?}",
                    trigger.name, trigger.source, trigger.actions
                );
                trigger.activated = true;
                self.tmp_trigger_script_targets.extend(trigger.actions.clone());
            }
        }
        let trigger_commands = std::mem::take(&mut self.tmp_trigger_script_targets);
        for command in trigger_commands {
            self.dispatch_script_command(command, draw_groups);
        }
    }

    pub fn take_pending_level_transition(&mut self) -> Option<PendingLevelTransition> {
        self.pending_level_transition.take()
    }

    fn dispatch_script_command(&mut self, command: ScriptCommand, draw_groups: &mut Vec<DrawGroup>) {
        match command {
            ScriptCommand::StartScript { script_name } => self.start_script(&script_name),
            ScriptCommand::TriggerDoor { door_name, op } => match op {
                DoorOp::Open => self.open_named_door(&door_name),
                DoorOp::Close => self.close_named_door(&door_name),
            },
            ScriptCommand::TriggerGeneric { target_name } => {
                self.start_script(&target_name);
                self.open_named_door(&target_name);
            }
            ScriptCommand::TransitionLevel { next_world, start_point } => {
                self.pending_level_transition = Some(PendingLevelTransition {
                    next_world: Some(next_world),
                    start_point,
                });
            }
            ScriptCommand::ItemSpawn { target_name } => {
                log::info!("item_spawn '{}' is not implemented yet", target_name);
            }
            ScriptCommand::SetObjectEnabled { target_name, enabled } => {
                log::info!("script/object enable '{}' -> {} is not implemented yet", target_name, enabled);
            }
            ScriptCommand::PlaySound { path, local } => {
                let _ = draw_groups;
                log::info!("play sound '{}' local={} is not implemented yet", path, local);
            }
        }
    }

    fn start_script(&mut self, target: &str) {
        for (name, runner) in &mut self.scripts {
            if name.eq_ignore_ascii_case(target) {
                runner.start();
            }
        }
    }

    fn open_named_door(&mut self, target: &str) {
        for door in &mut self.bsp_doors {
            if door.name.eq_ignore_ascii_case(target) {
                door.open();
            }
        }
        for obj in &mut self.objects {
            if let GameObject::Door(door) = obj {
                if door.trigger_name.eq_ignore_ascii_case(target) {
                    door.open();
                }
            }
        }
    }

    fn close_named_door(&mut self, target: &str) {
        for obj in &mut self.objects {
            if let GameObject::Door(door) = obj {
                if door.trigger_name.eq_ignore_ascii_case(target) {
                    door.close();
                }
            }
        }
        if self.bsp_doors.iter().any(|door| door.name.eq_ignore_ascii_case(target)) {
            log::info!("BSP door close '{}' is not implemented yet", target);
        }
    }

    // ── Damage API ───────────────────────────────────────────────────────────

    pub fn apply_area_damage(
        &mut self,
        origin: [f32; 3],
        radius: f32,
        damage: f32,
        draw_groups: &mut Vec<DrawGroup>,
    ) {
        // Collect which indices are barrels that just exploded so we can chain.
        self.tmp_newly_exploding.clear();

        for (index, obj) in self.objects.iter_mut().enumerate() {
            match obj {
                GameObject::Barrel(barrel) => {
                    if barrel.state == BarrelState::Intact
                        && dist3(origin, barrel.position) < radius
                        && barrel.apply_damage(damage)
                    {
                        self.tmp_newly_exploding.push(index);
                    }
                }
                GameObject::Crate(crate_object) => {
                    if crate_object.state == CrateState::Intact
                        && dist3(origin, crate_object.position) < radius
                        && crate_object.apply_damage(damage)
                    {
                        CCrate::on_destroy(crate_object, draw_groups);
                    }
                }
                GameObject::Window(window) => {
                    if window.state == WindowState::Intact
                        && dist3(origin, window.position) < radius
                        && window.apply_damage(damage)
                    {
                        CWindow::on_break(window, draw_groups);
                    }
                }
                GameObject::Headless(h) => {
                    // Headless can take area damage (explosions, etc.)
                    if dist3(origin, h.position) < radius {
                        if h.apply_damage(damage) {
                            if let Some(dg) = h.draw_group {
                                hide_draw_group(draw_groups, dg);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        for index in self.tmp_newly_exploding.drain(..) {
            if let GameObject::Barrel(barrel) = &mut self.objects[index] {
                hide_draw_group(draw_groups, barrel.draw_group);
                self.explosion_lights.push(ExplosionLight {
                    position: barrel.position,
                    color: barrel.element.flash_color(),
                    radius: barrel.explosion_radius * 2.5,
                    time_remaining: EXPLOSION_FLASH_DURATION,
                });
            }
        }
    }

    /// Directly damage the object closest to `origin` (hit-scan style).
    /// Returns the amount of damage absorbed, or 0 if nothing was hit.
    pub fn apply_point_damage(&mut self, origin: [f32; 3], damage: f32, draw_groups: &mut Vec<DrawGroup>) -> f32 {
        let mut best_dist = 1.5_f32; // max hit-scan reach in Vulkan units
        let mut best_idx: Option<usize> = None;

        for (idx, obj) in self.objects.iter().enumerate() {
            if let Some(pos) = obj.position() {
                let d = dist3(origin, pos);
                if d < best_dist {
                    best_dist = d;
                    best_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = best_idx {
            match &mut self.objects[idx] {
                GameObject::Barrel(b) => {
                    if b.apply_damage(damage) {
                        // Explosion will trigger in next update() via state machine.
                    }
                }
                GameObject::Crate(c) => {
                    if c.apply_damage(damage) {
                        CCrate::on_destroy(c, draw_groups);
                    }
                }
                GameObject::Window(w) => {
                    if w.apply_damage(damage) {
                        CWindow::on_break(w, draw_groups);
                    }
                }
                    GameObject::Headless(h) => {
                        if h.apply_damage(damage) {
                            if let Some(dg) = h.draw_group {
                                hide_draw_group(draw_groups, dg);
                            }
                        }
                    }
                _ => return 0.0,
            }
            damage
        } else {
            0.0
        }
    }

    // ── Shadow caster export ──────────────────────────────────────────────

    /// Returns world-space positions of all creatures for blob shadow casting.
    pub fn shadow_caster_positions(&self) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for obj in &self.objects {
            match obj {
                GameObject::Creature(c) => out.push(c.position),
                GameObject::Headless(h) => out.push(h.position),
                _ => {}
            }
        }
        out
    }

    // ── Dynamic light export ─────────────────────────────────────────────────

    /// Build the list of dynamic lights for this frame: explosion flashes +
    /// torch flicker lights.  The caller merges these with the static world
    /// lights before uploading the lighting UBO.
    /// Returns dynamic point lights this frame: explosion flashes + torch flicker.
    ///
    /// `time` is the total elapsed time in seconds used for torch flicker sine waves.
    pub fn dynamic_lights(&self, time: f32) -> Vec<Light> {
        let mut out: Vec<Light> = Vec::new();

        // Explosion flashes
        for el in &self.explosion_lights {
            if el.time_remaining > 0.0 {
                out.push(el.to_light());
            }
        }

        // Torch flicker
        for obj in &self.objects {
            if let GameObject::Torch(t) = obj {
                out.push(CTorch::dynamic_light(t, time));
            }
        }

        out
    }
}
