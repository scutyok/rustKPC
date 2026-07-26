//******************************************************************/
//
// DemoSkyWorldModel — an animated sky BSP world model (dome, clouds, moon, buildings).
//
// In the Lithtech engine, DemoSkyWorldModel is an entity that wraps a BSP world
// model by name and places it in the sky rendering pass (camera-relative, no
// parallax with world movement).
//
// DAT properties read:
//   `Name`          (String)  — BSP world-model name this entity controls.
//   `SkyDims`       (Vector3) — sky sphere extents (cosmetic, not used in our renderer).
//   `InnerPercentZ` (Float)   — inner-camera Z axis fraction (unused here).
//   `Index`         (Float / LongInt) — draw order within the sky pass.
//
// Animation implemented here:
//   * Models containing "cloud" in their name receive slow diagonal panning
//     via a model-matrix translation applied each frame.
//   * Models containing "moon" receive a very slow orbital rotation.
//   * All other sky models (sky dome, ground, buildings) are static.
//
//******************************************************************/

use cgmath::{Matrix4, Rad, vec3};

use crate::object_utils::{matrix4_to_array, set_draw_group_matrix};
use crate::types::DrawGroup;

//******************************************************************/
//
// Pan speeds (UV repeats per second)
//
// The texture sampler repeats, so these continuously scroll each cloud layer
// without moving its sky geometry or opening seams in a bounded skybox.
//
//******************************************************************/

const CLOUD_PAN_SPEED_X: f32 = 0.10;
const CLOUD_PAN_SPEED_Y: f32 = 0.065;
const CLOUD2_PAN_SPEED_X: f32 = -0.075;
const CLOUD2_PAN_SPEED_Y: f32 = 0.095;
const MOON_ROT_SPEED: f32 = 0.002;     // radians/s — slow orbit

//******************************************************************/
//
// SkyModelInfo (returned by world_loader)
//
//******************************************************************/

/// Lightweight descriptor produced by world_loader when it adds a sky world
/// model to the draw-group array.
#[derive(Debug, Clone)]
pub struct SkyModelInfo {
    /// Lower-cased world-model name (matches `DemoSkyWorldModel.Name`).
    pub name: String,
    /// Index of the first DrawGroup owned by this sky model.
    pub draw_group_start: usize,
    /// Number of DrawGroups owned by this sky model.
    pub draw_group_count: usize,
    /// Optional pan X speed (Vulkan units/s) provided by DAT properties.
    pub pan_x: Option<f32>,
    /// Optional pan Y speed (Vulkan units/s) provided by DAT properties.
    pub pan_y: Option<f32>,
    /// Optional rotation speed (radians/s) provided by DAT properties.
    pub rot_speed: Option<f32>,
}
//******************************************************************/
//
// SkyWorldModelObject
//
//******************************************************************/

/// Runtime game-object that animates a sky BSP model each frame.
#[derive(Debug, Clone)]
pub struct SkyWorldModelObject {
    /// Lower-cased world-model name.
    pub name: String,
    /// First owned DrawGroup index in `AppData::draw_groups`.
    pub draw_group_start: usize,
    /// Number of owned DrawGroups.
    pub draw_group_count: usize,
    /// X pan speed (Vulkan units/s).
    pub pan_x: f32,
    /// Y pan speed (Vulkan units/s).
    pub pan_y: f32,
    /// Rotation speed around sky-Z axis (radians/s). 0 = no rotation.
    pub rot_speed: f32,
    /// Current accumulated X offset (sky space).
    pub offset_x: f32,
    /// Current accumulated Y offset (sky space).
    pub offset_y: f32,
    /// Current accumulated rotation angle (radians).
    pub angle: f32,
    /// World-space pivot used as the rotation centre in sky space.
    pub pivot: [f32; 3],
}

impl SkyWorldModelObject {
    /// Build from a loader-produced `SkyModelInfo`, choosing animation params
    /// based on the model name.
    pub fn from_info(info: &SkyModelInfo, pivot: [f32; 3]) -> Self {
        let name_lc = info.name.to_lowercase();
        // DAT-specified values override name-based defaults when present.
        let default = if name_lc.contains("cloud2") || name_lc == "clouds2" {
            // Counter-diagonal movement keeps the overlay distinct.
            (CLOUD2_PAN_SPEED_X, CLOUD2_PAN_SPEED_Y, 0.0)
        } else if name_lc.contains("cloud") {
            // Main cloud layer drifts diagonally across the sky.
            (CLOUD_PAN_SPEED_X, CLOUD_PAN_SPEED_Y, 0.0)
        } else if name_lc.contains("moon") {
            // Moon slowly orbits
            (0.0, 0.0, MOON_ROT_SPEED)
        } else {
            // sky dome, ground, buildings — static
            (0.0, 0.0, 0.0)
        };

        let pan_x = info.pan_x.unwrap_or(default.0);
        let pan_y = info.pan_y.unwrap_or(default.1);
        let rot_speed = info.rot_speed.unwrap_or(default.2);

        Self {
            name: info.name.clone(),
            draw_group_start: info.draw_group_start,
            draw_group_count: info.draw_group_count,
            pan_x,
            pan_y,
            rot_speed,
            offset_x: 0.0,
            offset_y: 0.0,
            angle: 0.0,
            pivot,
        }
    }

    /// Returns true if this sky model needs per-frame updates.
    pub fn is_animated(&self) -> bool {
        self.pan_x.abs() > 1e-6 || self.pan_y.abs() > 1e-6 || self.rot_speed.abs() > 1e-6
    }

    /// Advance animation state and write model matrices to the draw groups.
    pub fn update(&mut self, dt: f32, draw_groups: &mut Vec<DrawGroup>) {
        if !self.is_animated() {
            return;
        }

        // Accumulate offsets
        self.offset_x += self.pan_x * dt;
        self.offset_y += self.pan_y * dt;
        self.angle += self.rot_speed * dt;

        // Loop: prevent float drift after many minutes
        if self.offset_x.abs() > 1000.0 {
            self.offset_x = 0.0;
        }
        if self.offset_y.abs() > 1000.0 {
            self.offset_y = 0.0;
        }
        if self.angle > std::f32::consts::TAU {
            self.angle -= std::f32::consts::TAU;
        }

        let mat = if self.rot_speed.abs() > 1e-6 {
            // Rotation around sky Z (up) axis, pivoting at the model centre
            let p = self.pivot;
            let to_origin: Matrix4<f32> = Matrix4::from_translation(vec3(-p[0], -p[1], -p[2]));
            let rot: Matrix4<f32> = Matrix4::from_angle_z(Rad(self.angle));
            let from_origin: Matrix4<f32> = Matrix4::from_translation(vec3(p[0], p[1], p[2]));
            // Also apply pan offset
            let pan: Matrix4<f32> =
                Matrix4::from_translation(vec3(self.offset_x, self.offset_y, 0.0));
            matrix4_to_array(pan * from_origin * rot * to_origin)
        } else {
            // Pure translation pan
            let pan: Matrix4<f32> =
                Matrix4::from_translation(vec3(self.offset_x, self.offset_y, 0.0));
            matrix4_to_array(pan)
        };

        for dg_idx in self.draw_group_start
            ..self.draw_group_start + self.draw_group_count
        {
            set_draw_group_matrix(draw_groups, dg_idx, Some(mat));
        }
    }
}
