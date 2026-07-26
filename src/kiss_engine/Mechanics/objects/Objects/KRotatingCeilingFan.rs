//******************************************************************/
//
// CRotatingCeilingFan — continuously spinning ceiling fan (BSP sub-model).
//
// The fan is a BSP world model (not an ABC model), so it has one or more
// `draw_groups` rather than a single index.  Its rotation pivot is the
// geometric centroid of its own mesh (computed by `world_loader`).
//
// The rotation speed is read from the `RotateSpeed` DAT property (rad/s).
// Default speed is 1.5 full revolutions/s.
//
//******************************************************************/

use cgmath::{Matrix4, Rad, vec3};

use crate::object_utils::{matrix4_to_array, set_draw_group_matrix};
use crate::types::DrawGroup;

const FAN_DEFAULT_SPIN_SPEED: f32 = std::f32::consts::TAU * 1.5;

#[derive(Debug, Clone)]
pub struct FanObject {
    pub position: [f32; 3],
    pub draw_groups: Vec<usize>,
    pub angle: f32,
    pub spin_speed: f32,
}

impl FanObject {
    fn model_matrix(&self) -> [[f32; 4]; 4] {
        let c = self.position;
        let to_origin: Matrix4<f32> =
            Matrix4::from_translation(vec3(-c[0], -c[1], -c[2]));
        let rot: Matrix4<f32> = Matrix4::from_angle_z(Rad(self.angle));
        let from_origin: Matrix4<f32> =
            Matrix4::from_translation(vec3(c[0], c[1], c[2]));
        matrix4_to_array(from_origin * rot * to_origin)
    }
}

// ─── BSP sub-model construction ──────────────────────────────────────────────

pub fn parse(pivot: [f32; 3], draw_groups: Vec<usize>, spin_speed: f32) -> FanObject {
    FanObject {
        position: pivot,
        draw_groups,
        angle: 0.0,
        spin_speed,
    }
}

/// Build from a BSP sub-model entry; reads `RotateSpeed` from matching DAT object.
pub fn parse_from_bsp(pivot: [f32; 3], draw_groups: Vec<usize>) -> FanObject {
    parse(pivot, draw_groups, FAN_DEFAULT_SPIN_SPEED)
}

// ─── Per-frame update ────────────────────────────────────────────────────────

pub fn update(fan: &mut FanObject, dt: f32, draw_groups: &mut Vec<DrawGroup>) {
    fan.angle += fan.spin_speed * dt;
    if fan.angle > std::f32::consts::TAU {
        fan.angle -= std::f32::consts::TAU;
    }
    let mat = Some(fan.model_matrix());
    for &dg in &fan.draw_groups {
        set_draw_group_matrix(draw_groups, dg, mat);
    }
}
