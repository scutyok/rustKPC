//******************************************************************/
//
//  CBarrel — explosive/breakable barrel for KISS Psycho Circus.
//
//  Barrel types (inferred from skin texture):
//  BARREL_A.DTX  → Air    (large concussive blast)
//  BARREL_F.DTX  → Fire   (burning splash)
//  BARREL_G.DTX  → Earth  (shrapnel)
//  BARREL_W.DTX / BARREL_WATER.DTX → Water (splash)
//  BARREL_E.DTX / BARREL_CIR.DTX  → Electric (chain lightning)
//
//******************************************************************/

use crate::dat::WorldObject;
use crate::object_utils::prop_float;

//******************************************************************/
//
// Constants 
//
//******************************************************************/

pub const BARREL_HEALTH_DEFAULT: f32 = 50.0;
/// How long the explosion light flash lasts (seconds).
pub const EXPLOSION_FLASH_DURATION: f32 = 1.2;

//******************************************************************/
//
// BarrelElement
//
//******************************************************************/

/// Elemental type inferred from the barrel's skin texture filename.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrelElement {
    Generic,
    Water,
    Air,
    Fire,
    Earth,
    Electric,
}

impl BarrelElement {
    pub fn from_skin(skin: &str) -> Self {
        let up = skin.to_ascii_uppercase();
        if up.contains("BARREL_W") || up.contains("BARREL_WATER") {
            Self::Water
        } else if up.contains("BARREL_A") {
            Self::Air
        } else if up.contains("BARREL_F") {
            Self::Fire
        } else if up.contains("BARREL_G") {
            Self::Earth
        } else if up.contains("BARREL_E") || up.contains("BARREL_CIR") {
            Self::Electric
        } else {
            Self::Generic
        }
    }

    pub fn radius_multiplier(self) -> f32 {
        match self {
            Self::Air => 1.6,
            Self::Fire => 1.2,
            _ => 1.0,
        }
    }

    pub fn damage_multiplier(self) -> f32 {
        match self {
            Self::Fire => 1.4,
            Self::Electric => 1.3,
            Self::Earth => 1.2,
            _ => 1.0,
        }
    }

    pub fn flash_color(self) -> [f32; 3] {
        match self {
            Self::Fire => [1.0, 0.4, 0.1],
            Self::Electric => [0.6, 0.8, 1.0],
            Self::Water => [0.3, 0.6, 1.0],
            Self::Earth => [0.8, 0.6, 0.2],
            Self::Air => [0.9, 0.9, 0.8],
            Self::Generic => [1.0, 0.6, 0.2],
        }
    }
}

//******************************************************************/
//
// BarrelState
//
//******************************************************************/

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrelState {
    Intact,
    /// Counting down before the explosion fires (timer in seconds; 0 = immediate).
    Exploding { timer: f32 },
    Destroyed,
}

//******************************************************************/
//
// BarrelObject
//
//******************************************************************/

#[derive(Debug, Clone)]
pub struct BarrelObject {
    pub position: [f32; 3],
    pub element: BarrelElement,
    pub health: f32,
    pub explosion_radius: f32,
    pub explosion_damage: f32,
    pub state: BarrelState,
    pub draw_group: usize,
}

impl BarrelObject {
    pub fn is_alive(&self) -> bool {
        self.state == BarrelState::Intact
    }

    /// Apply damage; returns `true` when the barrel transitions to exploding.
    pub fn apply_damage(&mut self, amount: f32) -> bool {
        if self.state != BarrelState::Intact {
            return false;
        }
        self.health -= amount;
        if self.health <= 0.0 {
            self.state = BarrelState::Exploding { timer: 0.0 };
            return true;
        }
        false
    }
}

//******************************************************************/
//
// Explosion data returned to the manager
//
//******************************************************************/

/// Data emitted when a barrel explodes; the manager turns this into a light
/// flash and triggers area-damage on nearby objects.
#[derive(Debug, Clone)]
pub struct PendingExplosion {
    pub position: [f32; 3],
    pub radius: f32,
    pub damage: f32,
    pub color: [f32; 3],
}

//******************************************************************/
//
// DAT construction
//
//******************************************************************/

pub fn parse(
    pos: [f32; 3],
    props: Option<&WorldObject>,
    draw_group: usize,
    skin_filename: &str,
    scale: f32,
) -> BarrelObject {
    let element = BarrelElement::from_skin(skin_filename);
    let health = prop_float(props, "Health", BARREL_HEALTH_DEFAULT);
    let expl_radius = prop_float(props, "ExplosionRadius", 150.0) * scale * element.radius_multiplier();
    let expl_damage = prop_float(props, "ExplosionDamage", 75.0) * element.damage_multiplier();
    BarrelObject {
        position: pos,
        element,
        health,
        explosion_radius: expl_radius,
        explosion_damage: expl_damage,
        state: BarrelState::Intact,
        draw_group,
    }
}

//******************************************************************/
//
// Per-frame update
//
//******************************************************************/

/// Advance the barrel state machine.
/// Returns `Some(PendingExplosion)` the frame the barrel explodes so the
/// manager can spawn a light flash and trigger area damage.
pub fn update(barrel: &mut BarrelObject, dt: f32) -> Option<PendingExplosion> {
    if let BarrelState::Exploding { ref mut timer } = barrel.state {
        *timer += dt;
        if *timer >= 0.0 {
            let expl = PendingExplosion {
                position: barrel.position,
                radius: barrel.explosion_radius,
                damage: barrel.explosion_damage,
                color: barrel.element.flash_color(),
            };
            barrel.state = BarrelState::Destroyed;
            return Some(expl);
        }
    }
    None
}
