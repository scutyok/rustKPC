//******************************************************************/
//
// CWindow / CWindowShattering — destructible glass window.
//
// Health defaults to 20. On destruction the draw group is hidden.
// A future extension can spawn glass-shard particles.
//
//******************************************************************/

use crate::dat::WorldObject;
use crate::object_utils::{hide_draw_group, prop_float};
use crate::types::DrawGroup;

pub const WINDOW_HEALTH_DEFAULT: f32 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowState {
    Intact,
    Broken,
}

#[derive(Debug, Clone)]
pub struct WindowObject {
    pub position: [f32; 3],
    pub health: f32,
    pub state: WindowState,
    pub draw_group: usize,
}

impl WindowObject {
    pub fn apply_damage(&mut self, amount: f32) -> bool {
        if self.state != WindowState::Intact {
            return false;
        }
        self.health -= amount;
        if self.health <= 0.0 {
            self.state = WindowState::Broken;
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

pub fn parse(pos: [f32; 3], props: Option<&WorldObject>, draw_group: usize) -> WindowObject {
    WindowObject {
        position: pos,
        health: prop_float(props, "Health", WINDOW_HEALTH_DEFAULT),
        state: WindowState::Intact,
        draw_group,
    }
}

//******************************************************************/
//
// Event
//
//******************************************************************/

pub fn on_break(window: &WindowObject, draw_groups: &mut Vec<DrawGroup>) {
    // Run animation, spawn particles, etc. (not implemented yet)
    hide_draw_group(draw_groups, window.draw_group);
}
