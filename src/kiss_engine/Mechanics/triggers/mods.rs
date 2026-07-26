//! Level-trigger definitions and construction.
//!
//! Object-specific trigger behaviour lives in its own module.  `TriggerFactory`
//! is deliberately only the dispatcher: adding a trigger means adding its
//! object module and one type-name arm in `build_object_trigger`.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dat::PropertyValue;
use crate::scripted_sequence::{parse_dat_command, ScriptCommand};
use crate::util::geometry::AABB;

#[path = "TriggerObjects/KExitTrigger.rs"]
mod exit_trigger;
#[path = "TriggerObjects/KTriggerUtil.rs"]
pub(crate) mod util;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerActivation { Touch, Use, Script }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerSource { BspVolume, BspSwitch, DatObject }

#[derive(Debug, Clone)]
pub struct TriggerDef {
    pub name: String,
    pub source: TriggerSource,
    pub activation: TriggerActivation,
    pub once: bool,
    pub enabled: bool,
    pub activated: bool,
    pub volume: Option<AABB>,
    pub center: Option<[f32; 3]>,
    pub use_radius: f32,
    pub actions: Vec<ScriptCommand>,
}

impl TriggerDef {
    pub fn can_fire(&self) -> bool { self.enabled && (!self.once || !self.activated) }

    pub fn contains(&self, pos: [f32; 3]) -> bool {
        self.volume.map(|volume| volume.contains(pos.into())).unwrap_or(false)
    }

    pub fn can_use_from(&self, pos: [f32; 3]) -> bool {
        if let Some(volume) = self.volume { return volume.contains(pos.into()); }
        self.center.map(|center| {
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dz = pos[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt() <= self.use_radius
        }).unwrap_or(false)
    }
}

pub struct TriggerFactory<'a> {
    file_stem: &'a str,
    dat: &'a crate::dat::DatFile,
    trigger_volumes: &'a [(String, [f32; 3], [f32; 3])],
    bsp_submodels: &'a [(String, [f32; 3], Vec<usize>, f32)],
    scale: f32,
    script_object_targets: HashMap<String, Vec<String>>,
}

impl<'a> TriggerFactory<'a> {
    pub fn new(
        file_stem: &'a str,
        dat: &'a crate::dat::DatFile,
        trigger_volumes: &'a [(String, [f32; 3], [f32; 3])],
        bsp_submodels: &'a [(String, [f32; 3], Vec<usize>, f32)],
        scale: f32,
    ) -> Self {
        let mut factory = Self { file_stem, dat, trigger_volumes, bsp_submodels, scale, script_object_targets: HashMap::new() };
        factory.index_script_objects();
        factory
    }

    pub fn build(self) -> Vec<TriggerDef> {
        let mut triggers = Vec::new();
        for object in &self.dat.objects {
            if let Some(trigger) = self.build_object_trigger(object) { triggers.push(trigger); }
        }
        self.build_volume_triggers(&mut triggers);
        self.build_bsp_switch_triggers(&mut triggers);
        triggers
    }

    /// The central type switch.  Object modules construct their own trigger;
    /// this factory only routes a DAT object to the appropriate module.
    fn build_object_trigger(&self, object: &crate::dat::WorldObject) -> Option<TriggerDef> {
        match object.type_name.as_str() {
            "CExitTrigger" => exit_trigger::KExitTrigger::new(self.scale, self.trigger_volumes)
                .build_for_object(object),
            // A few levels use a custom entity class for an exit but retain
            // the standard NextWorld property.  Route those through the same
            // object builder rather than silently dropping the transition.
            _ if matches!(object.get_property("NextWorld"), Some(PropertyValue::String(world)) if !world.trim().is_empty()) => {
                exit_trigger::KExitTrigger::new(self.scale, self.trigger_volumes)
                    .build_for_object(object)
            }
            _ => None,
        }
    }

    fn index_script_objects(&mut self) {
        let empty_targets = HashMap::new();
        for object in &self.dat.objects {
            if object.type_name != "CScriptObject" { continue; }
            let Some(PropertyValue::String(name)) = object.get_property("Name") else { continue; };
            let scripts = command_properties(object, 8)
                .into_iter()
                .flat_map(|command| parse_dat_command(command, self.file_stem, &empty_targets))
                .filter_map(|action| match action { ScriptCommand::StartScript { script_name } => Some(script_name), _ => None })
                .collect::<Vec<_>>();
            if !scripts.is_empty() { self.script_object_targets.insert(name.to_lowercase(), scripts); }
        }
    }

    fn build_volume_triggers(&self, out: &mut Vec<TriggerDef>) {
        for (name, min, max) in self.trigger_volumes {
            let name = name.to_lowercase();
            let dat_object = self.find_dat_object(&name);
            let mut actions = dat_object.map(|object| self.actions_from_object_commands(object, 4)).unwrap_or_default();
            let has_dat_actions = !actions.is_empty();
            if name == "cslime2" { actions = vec![ScriptCommand::StartScript { script_name: "microphone".into() }]; }
            if actions.is_empty() { actions.push(ScriptCommand::StartScript { script_name: name.clone() }); }
            out.push(TriggerDef {
                name, source: TriggerSource::BspVolume,
                activation: if has_dat_actions { TriggerActivation::Touch } else { TriggerActivation::Use },
                once: true, enabled: true, activated: false,
                volume: Some(AABB { min: (*min).into(), max: (*max).into() }), center: None, use_radius: 0.0, actions,
            });
        }
    }

    fn build_bsp_switch_triggers(&self, out: &mut Vec<TriggerDef>) {
        for (name, center, _, _) in self.bsp_submodels {
            let name = name.to_lowercase();
            if !name.starts_with("cswitchslide") && !name.starts_with("cswitchrotating") { continue; }
            let object = self.dat.objects.iter().find(|object| {
                matches!(object.type_name.as_str(), "CSwitchSlide" | "CSwitchRotating")
                    && matches!(object.get_property("Name"), Some(PropertyValue::String(object_name)) if object_name.eq_ignore_ascii_case(&name))
            });
            let actions = object.map(|object| self.actions_from_object_commands(object, 4)).unwrap_or_default();
            if actions.is_empty() { continue; }
            out.push(TriggerDef { name, source: TriggerSource::BspSwitch, activation: TriggerActivation::Use,
                once: true, enabled: true, activated: false, volume: None, center: Some(*center), use_radius: 2.0, actions });
        }
    }

    fn actions_from_object_commands(&self, object: &crate::dat::WorldObject, max_commands: usize) -> Vec<ScriptCommand> {
        command_properties(object, max_commands).into_iter()
            .flat_map(|command| parse_dat_command(command, self.file_stem, &self.script_object_targets))
            .collect()
    }

    fn find_dat_object(&self, name: &str) -> Option<&crate::dat::WorldObject> {
        self.dat.objects.iter().find(|object| matches!(object.get_property("Name"), Some(PropertyValue::String(object_name)) if object_name.eq_ignore_ascii_case(name)))
    }
}

fn command_properties(object: &crate::dat::WorldObject, max_commands: usize) -> Vec<&str> {
    (1..=max_commands).filter_map(|index| match object.get_property(&format!("command{index}")) {
        Some(PropertyValue::String(command)) if !command.trim().is_empty() => Some(command.as_str()),
        _ => None,
    }).collect()
}

#[derive(Serialize, Debug, Clone)]
pub struct TriggerInfo {
    pub source: String, pub name: String, pub position: Option<[f32; 3]>, pub rotation: Option<[f32; 4]>,
    pub aabb_min: Option<[f32; 3]>, pub aabb_max: Option<[f32; 3]>, pub properties: HashMap<String, JsonValue>,
}

fn prop_to_json(property: &PropertyValue) -> JsonValue {
    match property {
        PropertyValue::String(value) => json!(value), PropertyValue::Vector(value) => json!([value.x, value.y, value.z]),
        PropertyValue::Color(value) => json!([value.x, value.y, value.z]), PropertyValue::Float(value) => json!(value),
        PropertyValue::Bool(value) => json!((*value) != 0), PropertyValue::Flags(value) => json!(value),
        PropertyValue::LongInt(value) => json!(value), PropertyValue::Rotation(value) => json!([value.w, value.x, value.y, value.z]),
        PropertyValue::UnknownInt(value) => json!(value),
    }
}

pub fn collect_triggers(dat: &crate::dat::DatFile, trigger_volumes: &[(String, [f32; 3], [f32; 3])], scale: f32) -> Vec<TriggerInfo> {
    let keywords = ["trigger", "volume", "script", "teleport", "portal", "zone", "pickup", "death", "damage", "kill", "lava", "slime"];
    let mut out = dat.objects.iter().filter(|object| {
        let type_name = object.type_name.to_lowercase();
        keywords.iter().any(|keyword| type_name.contains(keyword))
            || object.get_property("Script").is_some() || object.get_property("TargetName").is_some() || object.get_property("Name").is_some()
    }).map(|object| {
        let properties = object.properties.iter().map(|property| (property.name.clone(), prop_to_json(&property.value))).collect();
        TriggerInfo {
            source: "object".into(),
            name: object.get_property("Name").and_then(|value| match value { PropertyValue::String(name) => Some(name.clone()), _ => None }).unwrap_or_else(|| object.type_name.clone()),
            position: object.get_position().map(|position| util::transform_dat_position_to_runtime(position, scale)),
            rotation: object.get_rotation().map(|rotation| [rotation.w, rotation.x, rotation.y, rotation.z]),
            aabb_min: None, aabb_max: None, properties,
        }
    }).collect::<Vec<_>>();
    out.extend(trigger_volumes.iter().map(|(name, min, max)| TriggerInfo { source: "world_model".into(), name: name.clone(), position: None, rotation: None, aabb_min: Some(*min), aabb_max: Some(*max), properties: HashMap::new() }));
    out
}

pub fn export_triggers_json<P: AsRef<Path>>(triggers: &[TriggerInfo], dat_path: P) -> Result<()> {
    let dat_path = dat_path.as_ref();
    let stem = dat_path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("level");
    let out_path = dat_path.with_file_name(format!("{stem}.triggers.json"));
    let json = serde_json::to_string_pretty(triggers).context("serializing triggers to json")?;
    fs::write(&out_path, json).with_context(|| format!("writing triggers to {out_path:?}"))
}
