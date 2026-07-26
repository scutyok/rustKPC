//******************************************************************/
//
// CCrate — destructible crate.
//
// Health defaults to 35. On destruction the draw group is hidden.
//
//******************************************************************/

use crate::dat::WorldObject;
use crate::object_utils::{hide_draw_group, prop_float};
use crate::types::DrawGroup;

pub const CRATE_HEALTH_DEFAULT: f32 = 35.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrateState {
    Intact,
    Destroyed,
}

#[derive(Debug, Clone)]
pub struct CrateObject {
    pub position: [f32; 3],
    pub health: f32,
    pub state: CrateState,
    pub draw_group: usize,
}

impl CrateObject {
    /// Apply damage. Returns `true` on destruction.
    pub fn apply_damage(&mut self, amount: f32) -> bool {
        if self.state != CrateState::Intact {
            return false;
        }
        self.health -= amount;
        if self.health <= 0.0 {
            self.state = CrateState::Destroyed;
            return true;
        }
        false
    }
}

//******************************************************************/
//
// DAT construction
//
//******************************************************************/

pub fn parse(pos: [f32; 3], props: Option<&WorldObject>, draw_group: usize) -> CrateObject {
    CrateObject {
        position: pos,
        health: prop_float(props, "Health", CRATE_HEALTH_DEFAULT),
        state: CrateState::Intact,
        draw_group,
    }
}

//******************************************************************/
//
// Per-frame update
//
//******************************************************************/

/// Apply newly-confirmed destruction: hide the draw group.
/// Called by the manager after `apply_damage` returns `true`.
pub fn on_destroy(crate_obj: &CrateObject, draw_groups: &mut Vec<DrawGroup>) {
    hide_draw_group(draw_groups, crate_obj.draw_group);
}
