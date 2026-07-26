//******************************************************************/
//
// CTorch — decorative torch that emits a flickering point light and an
// animated billboard flame sprite.
//
// Flame textures: REZ/SPRITETEXTURES/FLAMETEST/TORCH1-6.DTX (6 frames @ ~12 fps).
// Light colour  : orange (1.0, 0.55, 0.1).
// Light radius  : 6.0 Vulkan units (~600 Lithtech units).
//
// DAT properties read:
//   `LightIntensity` (Float) — base light intensity (default 1.5).
//
//******************************************************************/

use cgmath::{Matrix4, Rad, vec3};

use crate::dat::WorldObject;
use crate::LightObj::Light;
use crate::object_utils::{matrix4_to_array, prop_float};
use crate::types::DrawGroup;

const FLAME_FRAME_SECS: f32 = 0.08; // ~12 fps animation
const FLAME_FRAMES: usize = 6;
const FLAME_HEIGHT: f32 = 0.35;
const FLAME_Z_OFFSET: f32 = 0.10;

#[derive(Debug, Clone)]
pub struct SpriteEffectBinding {
    pub dat_object_index: Option<usize>,
    pub position: [f32; 3],
    pub draw_group: usize,
    pub base_tex_index: usize,
    pub frame_count: usize,
    pub frame_secs: f32,
    pub height: f32,
    pub z_offset: f32,
    pub light_color: [f32; 3],
    pub light_radius: f32,
    pub light_intensity: f32,
}

#[derive(Debug, Clone)]
pub struct TorchObject {
    pub position: [f32; 3],
    pub draw_group: usize,
    pub flicker_phase: f32,
    pub base_intensity: f32,
    /// DrawGroup index for the animated flame billboard quad.
    pub flame_draw_group: usize,
    /// Index into `AppData::level_textures` for TORCH1.DTX (frame 0).
    pub flame_base_tex_index: usize,
    pub flame_frame: usize,
    pub flame_frame_timer: f32,
    pub flame_frame_count: usize,
    pub flame_frame_secs: f32,
    pub flame_height: f32,
    pub flame_z_offset: f32,
    pub light_color: [f32; 3],
    pub light_radius: f32,
}

#[derive(Debug, Clone)]
pub struct SpriteEffectObject {
    pub position: [f32; 3],
    pub draw_group: usize,
    pub base_tex_index: usize,
    pub frame: usize,
    pub frame_timer: f32,
    pub frame_count: usize,
    pub frame_secs: f32,
    pub height: f32,
    pub z_offset: f32,
    pub flicker_phase: f32,
    pub light_color: [f32; 3],
    pub light_radius: f32,
    pub light_intensity: f32,
}

// ─── DAT construction ────────────────────────────────────────────────────────

pub fn parse(
    pos: [f32; 3],
    props: Option<&WorldObject>,
    draw_group: usize,
    flame_draw_group: usize,
    flame_base_tex_index: usize,
) -> TorchObject {
    let phase = (pos[0] * 7.3 + pos[1] * 3.7 + pos[2] * 5.1).fract().abs();
    let light_intensity = prop_float(props, "LightIntensity", 1.5);
    TorchObject {
        position: pos,
        draw_group,
        flicker_phase: phase * std::f32::consts::TAU,
        base_intensity: light_intensity,
        flame_draw_group,
        flame_base_tex_index,
        flame_frame: 0,
        flame_frame_timer: 0.0,
        flame_frame_count: FLAME_FRAMES.max(1),
        flame_frame_secs: FLAME_FRAME_SECS,
        flame_height: FLAME_HEIGHT,
        flame_z_offset: FLAME_Z_OFFSET,
        light_color: [1.0, 0.55, 0.1],
        light_radius: 6.0,
    }
}

pub fn parse_sprite_effect(effect: &SpriteEffectBinding) -> SpriteEffectObject {
    let phase = (effect.position[0] * 7.3 + effect.position[1] * 3.7 + effect.position[2] * 5.1)
        .fract()
        .abs();
    SpriteEffectObject {
        position: effect.position,
        draw_group: effect.draw_group,
        base_tex_index: effect.base_tex_index,
        frame: 0,
        frame_timer: 0.0,
        frame_count: effect.frame_count.max(1),
        frame_secs: effect.frame_secs,
        height: effect.height,
        z_offset: effect.z_offset,
        flicker_phase: phase * std::f32::consts::TAU,
        light_color: effect.light_color,
        light_radius: effect.light_radius,
        light_intensity: effect.light_intensity,
    }
}

// ─── Per-frame update ────────────────────────────────────────────────────────

/// Advance flame animation and billboard the quad toward the camera.
pub fn update(
    t: &mut TorchObject,
    dt: f32,
    player_pos: [f32; 3],
    draw_groups: &mut Vec<DrawGroup>,
) {
    // Advance animation frame.
    t.flame_frame_timer += dt;
    if t.flame_frame_timer >= t.flame_frame_secs {
        t.flame_frame_timer -= t.flame_frame_secs;
        t.flame_frame = (t.flame_frame + 1) % t.flame_frame_count;
    }

    // Billboard: rotate the quad around Z (up) to face the camera in XY.
    // Compute angle from +X to camera vector, then rotate so the quad's
    // default +Y-facing orientation matches that direction.
    let dx = player_pos[0] - t.position[0];
    let dy = player_pos[1] - t.position[1];
    let yaw = Rad(dy.atan2(dx) - std::f32::consts::FRAC_PI_2);

    let cx = t.position[0];
    let cy = t.position[1];
    let cz = t.position[2] + t.flame_z_offset + t.flame_height * 0.5;
    let to_origin: Matrix4<f32> = Matrix4::from_translation(vec3(-cx, -cy, -cz));
    let rot: Matrix4<f32> = Matrix4::from_angle_z(yaw);
    let from_origin: Matrix4<f32> = Matrix4::from_translation(vec3(cx, cy, cz));

    if let Some(dg) = draw_groups.get_mut(t.flame_draw_group) {
        dg.texture_index = t.flame_base_tex_index + t.flame_frame;
        dg.model_matrix = Some(matrix4_to_array(from_origin * rot * to_origin));
    }
}

pub fn update_sprite_effect(
    effect: &mut SpriteEffectObject,
    dt: f32,
    player_pos: [f32; 3],
    draw_groups: &mut Vec<DrawGroup>,
) {
    effect.frame_timer += dt;
    if effect.frame_timer >= effect.frame_secs {
        effect.frame_timer -= effect.frame_secs;
        effect.frame = (effect.frame + 1) % effect.frame_count;
    }

    let dx = player_pos[0] - effect.position[0];
    let dy = player_pos[1] - effect.position[1];
    let yaw = Rad(dy.atan2(dx) - std::f32::consts::FRAC_PI_2);

    let cx = effect.position[0];
    let cy = effect.position[1];
    let cz = effect.position[2] + effect.z_offset + effect.height * 0.5;
    let to_origin: Matrix4<f32> = Matrix4::from_translation(vec3(-cx, -cy, -cz));
    let rot: Matrix4<f32> = Matrix4::from_angle_z(yaw);
    let from_origin: Matrix4<f32> = Matrix4::from_translation(vec3(cx, cy, cz));

    if let Some(dg) = draw_groups.get_mut(effect.draw_group) {
        dg.texture_index = effect.base_tex_index + effect.frame;
        dg.model_matrix = Some(matrix4_to_array(from_origin * rot * to_origin));
    }
}

// ─── Dynamic light ──────────────────────────────────────────────────────────

/// Build the per-frame flickering point light for this torch.
/// `time` is total elapsed seconds used for the sine-wave flicker.
pub fn dynamic_light(t: &TorchObject, time: f32) -> Light {
    let flicker = (0.85
        + 0.10 * (time * 7.3 + t.flicker_phase).sin()
        + 0.05 * (time * 17.1 + t.flicker_phase * 2.3).sin())
        .max(0.5);
    Light {
        position: t.position,
        radius: t.light_radius,
        color: t.light_color,
        intensity: t.base_intensity * flicker,
    }
}

pub fn sprite_effect_light(effect: &SpriteEffectObject, time: f32) -> Option<Light> {
    if effect.light_intensity <= 0.0 || effect.light_radius <= 0.0 {
        return None;
    }
    let flicker = (0.85
        + 0.10 * (time * 7.3 + effect.flicker_phase).sin()
        + 0.05 * (time * 17.1 + effect.flicker_phase * 2.3).sin())
        .max(0.5);
    Some(Light {
        position: effect.position,
        radius: effect.light_radius,
        color: effect.light_color,
        intensity: effect.light_intensity * flicker,
    })
}
