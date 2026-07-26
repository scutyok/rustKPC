#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use anyhow::{Result, anyhow};
use cgmath::{Deg, InnerSpace, vec3};
use log::*;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use vulkanalia::window as vk_window;
use winit::window::Window;

use rustKPC::{OcclusionCulling, collision};
use rustKPC::egui_renderer;
use rustKPC::game_objects::GameObjectManager;
use rustKPC::LightObj;
use rustKPC::LightingUbo;
use rustKPC::CPlayerMovement::{self, MovementState};
use rustKPC::CameraObj::{Camera, InputState};
use rustKPC::types::*;
use rustKPC::vulkan;
use rustKPC::world_chooser::{LoadingState, WorldChooser};
use rustKPC::world_loader::load_dat_model;
use rustKPC::util::geometry::*;
use rustKPC::util::math::*;

const SKYBOX_HORIZONTAL_SIZE: f32 = 5000.0;
const SKYBOX_Z_RAISE: f32 = 1.0;
// The bounded-world sky is deliberately scaled up to occupy a volume
// SKYBOX_HORIZONTAL_SIZE units wide, centered on the camera — so a sky
// vertex can end up roughly SKYBOX_HORIZONTAL_SIZE/2 units out. The old
// far plane (1000.0) was smaller than that reach, so the GPU's mandatory
// frustum clipping was slicing the scaled skybox off in a straight line
// right at the far plane (independent of depth testing, which is off for
// the sky anyway — this clip happens earlier, in clip space). Used for
// the render projection, the culling frustum, and the debug FOV overlay
// so all three stay consistent with each other.
const RENDER_FAR_PLANE: f32 = 20_000.0;

//******************************************************************/
//
// Our Vulkan app.
//
//******************************************************************/
pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub models: usize,
    pub camera: Camera,
    pub input: InputState,
    pub move_state: MovementState,
    pub world_chooser: WorldChooser,
    pub current_world: String,
    pub loading_state: LoadingState,
    pub loading_texture_id: Option<egui::TextureId>,
    pub player_mode: collision::PlayerMode,
    pub height_provider: Box<dyn collision::HeightProvider>,
    pub mesh_provider: Option<collision::MeshHeightProvider>,
    // Physics
    pub is_free_cam: bool,
    pub show_fps: bool,
    pub vsync: bool,
    pub freeze_culling: bool,
    /// Enable/disable dynamic point lights and blob shadows
    pub dynamic_lighting: bool,
    /// When true, render sky and world at fixed coordinates (bounded map box)
    pub bounded_world: bool,
    /// World geometry bounds (computed from loaded mesh)
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    pub saved_player_camera: Option<Camera>,
    pub eye_offset_walk: f32,
    pub player_fov: f32,
    pub fps: f32,
    pub occlusion_culler: OcclusionCulling::OcclusionCuller,
    // Lighting
    pub world_lights: Vec<LightObj::Light>,
    /// Cached lighting UBO (rebuilt only when lights change, not every frame)
    pub cached_light_ubo: LightingUBO,
    // Game objects
    pub game_objects: GameObjectManager,
    /// Entity cylinders for solid objects (barrels, creatures) — player can't walk through.
    pub entity_cylinders: Vec<EntityCylinder>,
    /// Total elapsed time in seconds (for torch flicker etc.)
    pub elapsed_time: f32,
    // Fog
    pub fog_enabled: bool,
    pub fog_near: f32,
    pub fog_far: f32,
    pub fog_color: [f32; 3],
    /// Sky fog far distance (from SkyFogFarZ property).
    pub sky_fog_far: f32,
    /// Whether trigger/volume sub-models are visible.
    pub show_triggers: bool,
    /// Draw-group indices + original index counts for trigger volumes.
    pub trigger_draw_groups: Vec<(usize, u32)>,
    /// Pending world load triggered by an exit trigger or other transition.
    pub pending_world_load: Option<String>,
}

fn resolve_world_path(spec: &str) -> Option<String> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return None;
    }

    let normalized = trimmed.replace('\\', "/");
    let without_prefix = normalized
        .strip_prefix("worlds/")
        .unwrap_or(&normalized);
    let without_rez = without_prefix
        .strip_prefix("REZ/WORLDS/")
        .unwrap_or(without_prefix);

    let mut candidates = Vec::new();
    if without_rez.to_ascii_lowercase().starts_with("rez/worlds/") {
        candidates.push(without_rez.to_string());
    } else {
        let mut path = PathBuf::from("REZ/WORLDS");
        for part in without_rez.split('/') {
            if part.is_empty() {
                continue;
            }
            path.push(part);
        }
        let path_str = path.to_string_lossy().replace('\\', "/");
        candidates.push(path_str.clone());
        candidates.push(format!("{}.DAT", path_str));
        candidates.push(format!("{}.dat", path_str));
    }

    if !normalized.to_ascii_lowercase().starts_with("rez/worlds/") {
        let root_candidate = format!("REZ/WORLDS/{}", normalized);
        candidates.push(root_candidate.clone());
        candidates.push(format!("{}.DAT", root_candidate));
        candidates.push(format!("{}.dat", root_candidate));
    }

    for candidate in candidates {
        if Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }

    None
}

impl App {
    
    //******************************************************************/
    //
    // Creates our Vulkan app.
    //
    //******************************************************************/
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = vulkan::create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        vulkan::pick_physical_device(&instance, &mut data)?;
        let device = vulkan::create_logical_device(&entry, &instance, &mut data)?;
        vulkan::create_swapchain(window, &instance, &device, &mut data)?;
        vulkan::create_swapchain_image_views(&device, &mut data)?;
        vulkan::create_render_pass(&instance, &device, &mut data)?;
        vulkan::create_descriptor_set_layout(&device, &mut data)?;
        vulkan::create_pipeline(&device, &mut data)?;
        vulkan::create_command_pools(&instance, &device, &mut data)?;
        vulkan::create_color_objects(&instance, &device, &mut data)?;
        vulkan::create_depth_objects(&instance, &device, &mut data)?;
        vulkan::create_framebuffers(&device, &mut data)?;

        let loaded = load_dat_model(&mut data, "REZ/WORLDS/REALM1/R1M1A.DAT", 0, 0.01)?;
        let initial_lights = loaded.lights;
        let initial_game_objects = loaded.game_objects;
        let initial_entity_cylinders = loaded.entity_cylinders;

        // Compute world bounds from non-sky draw groups. Sky geometry is
        // projected separately and should not expand player/world limits.
        let (world_min, world_max) = Self::compute_non_sky_world_bounds(&data);
        println!(
            "World bounds: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2})",
            world_min[0], world_min[1], world_min[2], world_max[0], world_max[1], world_max[2]
        );

        vulkan::create_texture_image(&instance, &device, &mut data)?;
        vulkan::create_texture_image_view(&device, &mut data)?;
        vulkan::create_texture_sampler(&device, &mut data)?;

        vulkan::create_vertex_buffer(&instance, &device, &mut data)?;
        vulkan::create_index_buffer(&instance, &device, &mut data)?;
        vulkan::create_uniform_buffers(&instance, &device, &mut data)?;
        vulkan::create_descriptor_pool(&device, &mut data)?;
        vulkan::create_descriptor_sets(&device, &mut data)?;
        vulkan::create_command_buffers(&device, &mut data)?;
        vulkan::create_sync_objects(&device, &mut data)?;
        let initial_world = "REZ/WORLDS/REALM1/R1M1A.DAT".to_string();

        let (initial_height_provider, initial_mesh_provider) =
            if !loaded.collision_positions.is_empty() && !loaded.collision_indices.is_empty() {
                let mesh =
                    collision::MeshHeightProvider::new(loaded.collision_positions, loaded.collision_indices);
                (
                    Box::new(mesh.clone()) as Box<dyn collision::HeightProvider>,
                    Some(mesh),
                )
            } else {
                (
                    Box::new(collision::FlatGround) as Box<dyn collision::HeightProvider>,
                    None,
                )
            };

        let initial_occlusion_culler = {
            let mut culler = OcclusionCulling::OcclusionCuller::new();
            let positions: Vec<[f32; 3]> = data
                .vertices
                .iter()
                .map(|v| [v.pos.x, v.pos.y, v.pos.z])
                .collect();
            let groups: Vec<(u32, u32)> = data
                .draw_groups
                .iter()
                .map(|g| (g.first_index, g.index_count))
                .collect();
            culler.build_from_groups(&positions, &data.indices, &groups);
            culler
        };

        let (fog_enabled, fog_near, fog_far, fog_color, sky_fog_far) =
            if let Some(fog) = loaded.fog {
                (fog.enabled, fog.near_z, fog.far_z, fog.color, fog.sky_fog_far)
            } else {
                (true, 5.0_f32, 22.0_f32, [0.05_f32, 0.05, 0.08], 22.0_f32)
            };

        let mut app = Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 1,
            camera: Camera::default(),
            input: InputState::default(),
            move_state: MovementState::default(),
            world_chooser: WorldChooser::new(),
            current_world: initial_world,
            loading_state: LoadingState::Ready,
            loading_texture_id: None,
            player_mode: collision::PlayerMode::Walk,
            height_provider: initial_height_provider,
            mesh_provider: initial_mesh_provider,
            is_free_cam: false,
            show_fps: true,
            vsync: true,
            freeze_culling: true,
            dynamic_lighting: true,
            bounded_world: true,
            world_bounds_min: world_min,
            world_bounds_max: world_max,
            saved_player_camera: None,
            eye_offset_walk: 0.32,
            player_fov: 60.0,
            fps: 0.0,
            occlusion_culler: initial_occlusion_culler,
            world_lights: initial_lights.clone(),
            cached_light_ubo: LightingUbo::build_light_ubo(&initial_lights, &[], fog_color, fog_near, fog_far, fog_enabled, sky_fog_far),
            game_objects: initial_game_objects,
            entity_cylinders: initial_entity_cylinders,
            elapsed_time: 0.0,
            fog_enabled,
            fog_near,
            fog_far,
            fog_color,
            sky_fog_far,
            show_triggers: false,
            trigger_draw_groups: loaded.trigger_draw_groups,
            pending_world_load: None,
        };

        // Hide trigger draw groups by default
        for &(dg_idx, _) in &app.trigger_draw_groups {
            if let Some(dg) = app.data.draw_groups.get_mut(dg_idx) {
                dg.index_count = 0;
            }
        }

        // Upload the full light UBO to all persistently mapped swapchain buffers
        app.upload_light_ubo_to_all();

        Ok(app)
    }

    //******************************************************************/
    //
    // Build a LightingUBO from a list of lights, shadow casters, and current fog settings.
    // Delegates to `lighting_ubo::build_light_ubo`.
    //
    //******************************************************************/
    fn build_light_ubo(
        world_lights: &[LightObj::Light],
        shadow_positions: &[[f32; 3]],
        fog_color: [f32; 3],
        fog_near: f32,
        fog_far: f32,
        fog_enabled: bool,
        sky_fog_far: f32,
    ) -> LightingUBO {
        LightingUbo::build_light_ubo(world_lights, shadow_positions, fog_color, fog_near, fog_far, fog_enabled, sky_fog_far)
    }

    fn compute_non_sky_world_bounds(data: &AppData) -> ([f32; 3], [f32; 3]) {
        let mut world_min = [f32::INFINITY; 3];
        let mut world_max = [f32::NEG_INFINITY; 3];
        let mut any = false;

        for group in data.draw_groups.iter().take(data.sky_draw_group_start) {
            let start = group.first_index as usize;
            let end = (group.first_index + group.index_count) as usize;
            for i in start..end.min(data.indices.len()) {
                let vi = data.indices[i] as usize;
                if let Some(v) = data.vertices.get(vi) {
                    let p = v.pos;
                    world_min[0] = world_min[0].min(p.x);
                    world_min[1] = world_min[1].min(p.y);
                    world_min[2] = world_min[2].min(p.z);
                    world_max[0] = world_max[0].max(p.x);
                    world_max[1] = world_max[1].max(p.y);
                    world_max[2] = world_max[2].max(p.z);
                    any = true;
                }
            }
        }

        if any {
            (world_min, world_max)
        } else {
            ([0.0; 3], [0.0; 3])
        }
    }

    //******************************************************************/
    //
    // Upload the full cached light UBO to all swapchain images' persistently mapped memory.
    // Called once when lights change (map load / swapchain recreate).
    //
    //******************************************************************/
    pub unsafe fn upload_light_ubo_to_all(&self) {
        // When dynamic lighting is disabled, ensure we do not upload any
        // dynamic lights/shadows into GPU memory even if `cached_light_ubo`
        // contains stale values.
        let ubo_to_upload = if self.dynamic_lighting {
            self.cached_light_ubo.clone()
        } else {
            let mut tmp = self.cached_light_ubo.clone();
            tmp.light_count = 0;
            tmp.shadow_count = 0;
            tmp
        };
        for mapped in &self.data.light_uniform_buffers_mapped {
            memcpy(&ubo_to_upload, (*mapped).cast(), 1);
        }
    }

    //******************************************************************/
    //
    // Renders a frame for our Vulkan app.
    //
    //******************************************************************/
    pub unsafe fn render(
        &mut self,
        window: &Window,
        egui_renderer: &mut egui_renderer::EguiRenderer,
        egui_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                return self.recreate_swapchain(window, egui_renderer)
            }
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_command_buffer(image_index, egui_renderer, egui_primitives, pixels_per_point)?;
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window, egui_renderer)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % self.data.swapchain_image_count as usize;

        Ok(())
    }

    //******************************************************************/
    //
    // Updates a command buffer for our Vulkan app.
    //
    //******************************************************************/
    #[rustfmt::skip]
    unsafe fn update_command_buffer(
        &mut self,
        image_index: usize,
        egui_renderer: &mut egui_renderer::EguiRenderer,
        egui_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<()> {
        let command_pool = self.data.command_pools[image_index];
        self.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        // The clear color is what shows through wherever nothing was drawn
        // this frame — gaps between bounded-sky BSP pieces, or simply
        // looking past the edge of the world/sky geometry. Previously this
        // was hardcoded pure black, so it stood out as a stark void even
        // when fog was active and every fogged surface was fading to a
        // lighter fog color instead. Tinting the clear color to match the
        // current fog color makes the void blend seamlessly with the
        // fogged-out sky/world instead of reading as a hole.
        let clear_color = if matches!(self.loading_state, LoadingState::Loading(_)) {
            [0.08, 0.08, 0.12, 1.0]
        } else if self.fog_enabled {
            [self.fog_color[0], self.fog_color[1], self.fog_color[2], 1.0]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear_color,
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);

        //******************************************************************/
        //
        // Run frustum culling
        //
        //******************************************************************/
        {
            let (cull_view, cull_fov) = if self.is_free_cam && self.freeze_culling {
                if let Some(ref saved) = self.saved_player_camera {
                    (saved.view_matrix_with_offset(self.eye_offset_walk), self.player_fov)
                } else {
                    (self.camera.view_matrix_with_offset(0.0), 45.0)
                }
            } else {
                let eye_offset = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
                    self.eye_offset_walk
                } else { 0.0 };
                let fov_deg = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
                    self.player_fov
                } else { 45.0 };
                (self.camera.view_matrix_with_offset(eye_offset), fov_deg)
            };

            let raw_proj = cgmath::perspective(
                    Deg(cull_fov),
                    self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                    0.01,
                    RENDER_FAR_PLANE,
                );
            let vp = raw_proj * cull_view;
            let frustum = OcclusionCulling::Frustum::from_view_proj(&vp);
            self.occlusion_culler.cull(&frustum);
        }

        if self.loading_state == LoadingState::Ready {
            let secondary_command_buffers = (0..self.models)
                .map(|i| self.update_secondary_command_buffer(image_index, i))
                .collect::<Result<Vec<_>, _>>()?;
            self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
        }

        self.device.cmd_end_render_pass(command_buffer);

        egui_renderer.render(
            &self.instance,
            &self.device,
            self.data.physical_device,
            command_buffer,
            image_index,
            egui_primitives,
            pixels_per_point,
        )?;

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    //******************************************************************/
    //
    // Updates a secondary command buffer for our Vulkan app.
    //
    //******************************************************************/
    #[rustfmt::skip]
    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        let world_flag: f32 = 1.0;
        let world_flag_bytes = &world_flag.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(command_buffer, self.data.index_buffer, 0, vk::IndexType::UINT32);

        let sky_start = self.data.sky_draw_group_start;
        let has_sky_groups = sky_start < self.data.draw_groups.len();

        // === Pass 1: Draw sky groups with sky pipeline (no depth writes) ===
        if has_sky_groups {
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.sky_pipeline);

            // Camera eye position, shared by every sky branch below. The sky
            // model is always re-anchored on this every frame — for a
            // bounded world it used to be anchored on a *static* world
            // center instead, which meant the (finite, fixed-size) skybox
            // mesh could be outrun as the camera moved far enough from that
            // fixed point, exposing the raw clear color as a black void
            // that seemed to "cut" through the sky. Re-centering on the
            // camera every frame means the skybox can never be left behind.
            let eye_z = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
                self.eye_offset_walk
            } else {
                0.0
            };
            let cam_eye = self.camera.position + Vector3::new(0.0, 0.0, eye_z);

            // Source-style 3D sky projection: treat the loaded sky BSPs as a
            // small model, scale them into a fixed-size volume, then keep
            // that volume centered on the camera.
            let sky_base_model: Mat4 = if self.bounded_world {
                let sky_min = self.data.sky_bounds_min;
                let sky_max = self.data.sky_bounds_max;
                let sky_extent_x = (sky_max[0] - sky_min[0]).abs();
                let sky_extent_y = (sky_max[1] - sky_min[1]).abs();
                let sky_extent = sky_extent_x.max(sky_extent_y);

                let sky_model = if sky_extent > 0.001 {
                    let sky_scale = SKYBOX_HORIZONTAL_SIZE / sky_extent;
                    // All layers are authored relative to the primary sky
                    // world's origin, not to the combined vertex bounds.  A
                    // combined-bounds centre shifts the individual planes by
                    // different amounts, causing the black seams in bounded
                    // skies.  Anchor every layer to this shared origin.
                    let sky_origin = self.data.sky_translation;
                    Mat4::from_translation((cam_eye + Vector3::new(0.0, 0.0, SKYBOX_Z_RAISE)).into())
                        * Mat4::from_scale(sky_scale)
                        * Mat4::from_translation(-vec3(sky_origin[0], sky_origin[1], sky_origin[2]))
                } else {
                    let sky_t = self.data.sky_translation;
                    Mat4::from_translation((cam_eye + Vector3::new(0.0, 0.0, SKYBOX_Z_RAISE)).into())
                        * Mat4::from_translation(-vec3(sky_t[0], sky_t[1], sky_t[2]))
                };
                sky_model
            } else {
                let sky_t = self.data.sky_translation;
                let sky_scale = 1.0;
                let sky_model = Mat4::from_translation((cam_eye + Vector3::new(0.0, 0.0, SKYBOX_Z_RAISE)).into())
                    * Mat4::from_scale(sky_scale)
                    * Mat4::from_translation(-vec3(sky_t[0], sky_t[1], sky_t[2]));
                sky_model
            };

            // A sky world can list its layers in any order in the DAT, so
            // draw order is never taken from file order or from camera
            // distance. A distance-based sort was tried here, but the AABBs
            // it compared come from each sky mesh's raw, untransformed local
            // coordinates (computed once at load time), while the sky model
            // itself is recentered on the camera every frame — so "distance
            // to centroid" wasn't actually comparable to real camera
            // distance, and could flip layer order the wrong way depending
            // on where the camera was. Instead every sky group is tagged
            // at load time with an explicit sky_draw_layer: 0 = base dome
            // (drawn first, furthest back), 1 = translucent cloud overlay
            // (drawn over the dome), 2 = opaque foreground skybox models
            // such as mountains/buildings (drawn last, occluding the
            // clouds/dome wherever they overlap). Sorting on that layer
            // guarantees the dome can never redraw after — and hide — the
            // clouds, which is what happened when dome and foreground
            // models shared a single "opaque, draw last" bucket.
            let mut sky_group_indices: Vec<usize> =
                (sky_start..self.data.draw_groups.len()).collect();
            sky_group_indices.sort_by_key(|&group_idx| {
                self.data.draw_groups[group_idx].sky_draw_layer
            });

            for group_idx in sky_group_indices {
                let group = &self.data.draw_groups[group_idx];
                if group.index_count == 0 {
                    continue;
                }

                let sky_opacity = group.sky_opacity.unwrap_or(-1.0);
                self.device.cmd_push_constants(
                    command_buffer,
                    self.data.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    64,
                    &sky_opacity.to_ne_bytes(),
                );

                // Apply the per-layer animation matrix as well as the shared
                // bounded/camera-relative sky transform.  The animation was
                // previously updated every frame but never submitted here.
                let layer_model: Mat4 = if let Some(mat) = group.model_matrix {
                    // Sky animation translations represent UV scrolling, not
                    // mesh movement.  Keep rotations (e.g. the moon) but
                    // strip translation so a bounded sky cannot develop gaps
                    // between its BSP layers.
                    Mat4::new(
                        mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                        mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                        mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                        0.0, 0.0, 0.0, mat[3][3],
                    )
                } else {
                    Mat4::from_scale(1.0)
                };
                let model = sky_base_model * layer_model;
                let model_bytes = std::slice::from_raw_parts(
                    &model as *const Mat4 as *const u8,
                    size_of::<Mat4>(),
                );
                self.device.cmd_push_constants(
                    command_buffer,
                    self.data.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    model_bytes,
                );

                // The model animation stores the running cloud offsets in
                // its translation.  Feed those to the fragment shader as UV
                // offsets instead of translating the bounded sky geometry.
                let (sky_u_offset, sky_v_offset) = group.model_matrix
                    .map(|mat| (mat[3][0], mat[3][1]))
                    .unwrap_or((0.0, 0.0));
                self.device.cmd_push_constants(
                    command_buffer,
                    self.data.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    68,
                    &sky_u_offset.to_ne_bytes(),
                );
                self.device.cmd_push_constants(
                    command_buffer,
                    self.data.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    72,
                    &sky_v_offset.to_ne_bytes(),
                );
                let descriptor_set = if group.texture_index < self.data.level_textures.len()
                    && !self.data.level_textures[group.texture_index].descriptor_sets.is_empty()
                {
                    self.data.level_textures[group.texture_index].descriptor_sets[image_index]
                } else {
                    self.data.descriptor_sets[image_index]
                };

                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.data.pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );

                self.device.cmd_draw_indexed(
                    command_buffer,
                    group.index_count,
                    1,
                    group.first_index,
                    0,
                    0
                );
            }
        }

        //******************************************************************/
        //
        // Pass 2: Draw world groups with normal pipeline (depth writes on)
        //
        //******************************************************************/

        {
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);

            // Push world flag (<=1.5) so the fragment shader runs full lighting
            self.device.cmd_push_constants(
                command_buffer,
                self.data.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                64,
                world_flag_bytes,
            );

            // Model matrix is pushed per draw-group inside the loop below.

            if self.data.draw_groups.is_empty() {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.data.pipeline_layout,
                    0,
                    &[self.data.descriptor_sets[image_index]],
                    &[],
                );
                self.device.cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);
            } else {
                for (group_idx, group) in self.data.draw_groups.iter().enumerate().take(sky_start) {
                    if !self.occlusion_culler.is_visible(group_idx) {
                        continue;
                    }
                    // Skip objects that have been destroyed (index_count set to 0)
                    if group.index_count == 0 {
                        continue;
                    }

                    // Push per-group model matrix (identity for static geo, delta for dynamic objects)
                    let model: Mat4 = if let Some(mat) = group.model_matrix {
                        cgmath::Matrix4::new(
                            mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                            mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                            mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                            mat[3][0], mat[3][1], mat[3][2], mat[3][3],
                        )
                    } else {
                        Mat4::from_scale(1.0)
                    };
                    let model_bytes = std::slice::from_raw_parts(
                        &model as *const Mat4 as *const u8, size_of::<Mat4>());
                    self.device.cmd_push_constants(
                        command_buffer,
                        self.data.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        model_bytes,
                    );

                    let descriptor_set = if group.texture_index < self.data.level_textures.len()
                        && !self.data.level_textures[group.texture_index].descriptor_sets.is_empty()
                    {
                        self.data.level_textures[group.texture_index].descriptor_sets[image_index]
                    } else {
                        self.data.descriptor_sets[image_index]
                    };

                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.data.pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );

                    self.device.cmd_draw_indexed(
                        command_buffer,
                        group.index_count,
                        1,
                        group.first_index,
                        0,
                        0
                    );
                }
            }
        }

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    //******************************************************************/
    //
    // Updates camera position based on input state using Quake/Blood2-style
    // acceleration physics (ground friction, air-strafe, etc.).
    //
    //******************************************************************/
    pub fn update_camera(&mut self, dt: f32) {
        let front = self.camera.front();
        let right = self.camera.right();
        let prev_pos = self.camera.position;

        if self.player_mode == collision::PlayerMode::Flying {
            // Noclip / fly mode — direct velocity, no physics
            let delta = CPlayerMovement::fly_tick(
                self.input.forward,
                self.input.backward,
                self.input.left,
                self.input.right,
                self.input.up,
                self.input.down,
                [front.x, front.y, front.z],
                [right.x, right.y, right.z],
                dt,
            );
            self.camera.position.x += delta[0];
            self.camera.position.y += delta[1];
            self.camera.position.z += delta[2];
            self.move_state.velocity = [0.0, 0.0];
            self.move_state.z_vel = 0.0;
            return;
        }

        //******************************************************************/
        //
        // Walk mode: acceleration-based physics
        //
        //******************************************************************/
        let front_flat = {
            let len = (front.x * front.x + front.y * front.y).sqrt();
            if len > 1e-6 {
                [front.x / len, front.y / len]
            } else {
                [1.0, 0.0]
            }
        };

        let input = CPlayerMovement::build_move_input(
            self.input.forward,
            self.input.backward,
            self.input.left,
            self.input.right,
            self.input.up, // Space = jump
            front_flat,
            [right.x, right.y],
        );

        let delta = CPlayerMovement::tick(&mut self.move_state, &input, dt);

        self.camera.position.x += delta[0];
        self.camera.position.y += delta[1];
        self.camera.position.z += delta[2];

        //******************************************************************/
        //
        // Collision resolution
        //
        //******************************************************************/
        const PLAYER_RADIUS: f32 = 0.20;
        const PLAYER_HALF_H: f32 = 0.35;

        if let Some(mesh) = &self.mesh_provider {
            mesh.resolve_player_movement(prev_pos, &mut self.camera.position, PLAYER_RADIUS);
        }

        //******************************************************************/
        //
        // Push player out of entity cylinders (barrels, headless enemies)
        //
        //******************************************************************/
        for cyl in &self.entity_cylinders {
            let player_z_min = self.camera.position.z - PLAYER_HALF_H;
            let player_z_max = self.camera.position.z + PLAYER_HALF_H;
            if player_z_max < cyl.z_min || player_z_min > cyl.z_max {
                continue;
            }
            let dx = self.camera.position.x - cyl.center_x;
            let dy = self.camera.position.y - cyl.center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let min_dist = cyl.radius + PLAYER_RADIUS;
            if dist < min_dist && dist > 1e-6 {
                let penetration = min_dist - dist;
                let nx = dx / dist;
                let ny = dy / dist;
                self.camera.position.x += nx * penetration;
                self.camera.position.y += ny * penetration;
            }
        }

        //******************************************************************/
        //
        // Push player out of door AABBs
        //
        //******************************************************************/
        {
            let pr = PLAYER_RADIUS;
            let pz_min = self.camera.position.z - PLAYER_HALF_H;
            let pz_max = self.camera.position.z + PLAYER_HALF_H;
            for (amin, amax) in self.game_objects.door_aabbs() {
                let ex_min = [amin[0] - pr, amin[1] - pr, amin[2]];
                let ex_max = [amax[0] + pr, amax[1] + pr, amax[2]];
                let px = self.camera.position.x;
                let py = self.camera.position.y;
                if px > ex_min[0] && px < ex_max[0]
                    && py > ex_min[1] && py < ex_max[1]
                    && pz_max > ex_min[2] && pz_min < ex_max[2]
                {
                    let push_xn = px - ex_min[0];
                    let push_xp = ex_max[0] - px;
                    let push_yn = py - ex_min[1];
                    let push_yp = ex_max[1] - py;
                    let min_push = push_xn.min(push_xp).min(push_yn).min(push_yp);
                    if min_push == push_xn {
                        self.camera.position.x = ex_min[0];
                    } else if min_push == push_xp {
                        self.camera.position.x = ex_max[0];
                    } else if min_push == push_yn {
                        self.camera.position.y = ex_min[1];
                    } else {
                        self.camera.position.y = ex_max[1];
                    }
                }
            }
        }

        //******************************************************************/
        //
        // Ground detection & Z resolution
        //
        //******************************************************************/
        let _before_z = self.camera.position.z;
        collision::resolve_player_collision(
            &mut self.camera.position,
            self.height_provider.as_ref(),
            PLAYER_HALF_H,
            PLAYER_HALF_H,
        );
        let ground_z_cur = self.height_provider.ground_height(
            self.camera.position.x,
            self.camera.position.y,
            Some(self.camera.position.z),
        );
        let ground_z_prev = self.height_provider.ground_height(
            self.camera.position.x,
            self.camera.position.y,
            Some(prev_pos.z),
        );
        let ground_z = if prev_pos.z >= ground_z_prev + PLAYER_HALF_H - 0.05
            && self.camera.position.z < ground_z_prev + PLAYER_HALF_H
        {
            ground_z_prev.max(ground_z_cur)
        } else {
            ground_z_cur
        };
        let min_z = ground_z + PLAYER_HALF_H;

        // Reset on_ground for this frame
        self.move_state.on_ground = false;

        if self.camera.position.z < min_z {
            let diff = min_z - self.camera.position.z;
            if diff <= PLAYER_HALF_H {
                self.camera.position.z = min_z;
                if self.move_state.z_vel < 0.0 {
                    self.move_state.z_vel = 0.0;
                }
                self.move_state.on_ground = true;
            }
            // Large diffs are ignored — ground_height found an upper surface,
            // not the floor we're standing on.
        } else if self.camera.position.z < min_z + 0.15 && self.move_state.z_vel <= 0.0 {
            self.camera.position.z = min_z;
            self.move_state.z_vel = 0.0;
            self.move_state.on_ground = true;
        }

        // ── Ceiling check (headroom) — runs AFTER ground so we never ────
        //    push the player below the floor they're standing on.
        let head_z = self.camera.position.z + PLAYER_HALF_H;
        let ceiling_z = self.height_provider.ceiling_height(
            self.camera.position.x,
            self.camera.position.y,
            head_z,
        );
        if ceiling_z < f32::MAX {
            let max_center_z = ceiling_z - PLAYER_HALF_H;
            // Only clamp if ceiling is actually reachable (above ground)
            if max_center_z >= min_z && self.camera.position.z > max_center_z {
                self.camera.position.z = max_center_z;
                if self.move_state.z_vel > 0.0 {
                    self.move_state.z_vel = 0.0;
                }
            }
        }

        // If we hit a wall and XY position was corrected, clamp velocity to
        // prevent sliding along blocked axes (wall-hugging).
        let actual_dx = self.camera.position.x - prev_pos.x;
        let actual_dy = self.camera.position.y - prev_pos.y;
        if dt > 0.0 {
            let intended_dx = delta[0];
            let intended_dy = delta[1];
            // If collision killed most of the motion on an axis, zero velocity there
            if intended_dx.abs() > 1e-6 && (actual_dx / intended_dx) < 0.1 {
                self.move_state.velocity[0] = 0.0;
            }
            if intended_dy.abs() > 1e-6 && (actual_dy / intended_dy) < 0.1 {
                self.move_state.velocity[1] = 0.0;
            }
        }

        // If bounded world mode is enabled, clamp the player's XY position
        // to the loaded world bounding box (with a small margin).
        if self.bounded_world {
            let margin = 0.5_f32;
            self.camera.position.x = self.camera.position.x.clamp(
                self.world_bounds_min[0] + margin,
                self.world_bounds_max[0] - margin,
            );
            self.camera.position.y = self.camera.position.y.clamp(
                self.world_bounds_min[1] + margin,
                self.world_bounds_max[1] - margin,
            );
            // Clamp Z as well to avoid leaving the enclosed box vertically
            self.camera.position.z = self.camera.position.z.clamp(
                self.world_bounds_min[2] + 0.1,
                self.world_bounds_max[2] - 0.1,
            );
        }
    }
    
    //******************************************************************/
    //
    // Tick all game objects, apply physics state-machines, and upload dynamic lights if needed.
    //
    //******************************************************************/
    pub unsafe fn update_objects(&mut self, dt: f32, egui_renderer: &mut egui_renderer::EguiRenderer) {
        self.elapsed_time += dt;
        let player_pos = [
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        ];
        self.game_objects.update(dt, self.elapsed_time, player_pos, &mut self.data.draw_groups);

        if let Some(transition) = self.game_objects.take_pending_level_transition() {
            if let Some(next_world) = transition.next_world.as_deref() {
                if let Some(target_world) = resolve_world_path(next_world) {
                    println!("Transitioning to level: {}", target_world);
                    self.pending_world_load = Some(target_world);
                } else {
                    warn!("Unable to resolve next world '{}' from exit trigger", next_world);
                }
            }
        }

        // Update door collision vertices to follow sliding doors
        if let Some(mesh) = &mut self.mesh_provider {
            self.game_objects.update_door_collision(&mut mesh.positions);
        }

        // Rebuild the UBO every frame. Respect `dynamic_lighting` so disabling
        // dynamic lighting prevents per-frame dynamic lights/shadows from
        // overwriting the user's chosen UBO state.
        let dynamic: Vec<LightObj::Light> = if self.dynamic_lighting {
            self.game_objects.dynamic_lights(self.elapsed_time)
        } else {
            Vec::new()
        };
        let shadow_positions: Vec<[f32; 3]> = if self.dynamic_lighting {
            self.game_objects.shadow_caster_positions()
        } else {
            Vec::new()
        };

        {
            //******************************************************************/
            //
            // Build the combined lighting UBO (world + dynamic lights/shadows)
            // but do NOT write it directly into all persistently mapped
            // buffers here — writing to buffers that the GPU may currently
            // be reading can cause visual corruption. Instead store it in
            // `self.cached_light_ubo` and write only the per-frame mapping
            // for the active swapchain image in `update_uniform_buffer`.
            //
            //******************************************************************/
            let mut all_lights = self.world_lights.clone();
            all_lights.extend(dynamic);
            let combined_ubo = Self::build_light_ubo(
                &all_lights,
                &shadow_positions,
                self.fog_color,
                self.fog_near,
                self.fog_far,
                self.fog_enabled,
                self.sky_fog_far,
            );

            // Cache for upload during `update_uniform_buffer(image_index)`.
            self.cached_light_ubo = combined_ubo;
        }
    }

    //******************************************************************/
    //
    // Handle E-key interaction: open nearby doors, activate switches.
    //
    //******************************************************************/
    pub fn interact(&mut self) {
        let player_pos = [
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        ];
        self.game_objects.interact(player_pos, &mut self.data.draw_groups);
    }

    //******************************************************************/
    //
    // Toggle visibility of trigger / volume sub-models (F2).
    //
    //******************************************************************/
    pub fn toggle_triggers(&mut self) {
        self.show_triggers = !self.show_triggers;
        for &(dg_idx, original_count) in &self.trigger_draw_groups {
            if let Some(dg) = self.data.draw_groups.get_mut(dg_idx) {
                dg.index_count = if self.show_triggers { original_count } else { 0 };
            }
        }
        log::info!("Triggers: {}", if self.show_triggers { "visible" } else { "hidden" });
    }

    //******************************************************************/
    //
    // Run the egui UI
    //
    //******************************************************************/
    pub fn run_ui(&mut self, ctx: &egui::Context, mouse_locked: &mut bool) {
        // Loading screen
        if let LoadingState::Loading(ref map_name) = self.loading_state {
            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::from_rgb(20, 20, 30)))
                .show(ctx, |ui| {
                    if let Some(tex_id) = self.loading_texture_id {
                        let available = ui.available_size();
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(
                            tex_id,
                            available,
                        )));
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.vertical_centered(|ui| {
                                ui.add_space(ui.available_height() / 2.0 - 40.0);
                                ui.heading(
                                    egui::RichText::new("Loading...")
                                        .size(32.0)
                                        .color(egui::Color32::WHITE),
                                );
                                ui.add_space(10.0);
                                ui.label(
                                    egui::RichText::new(map_name)
                                        .size(24.0)
                                        .color(egui::Color32::LIGHT_GRAY),
                                );
                            });
                        });
                    }
                });
            return;
        }

        //******************************************************************/
        //
        // FPS counter
        //
        //******************************************************************/
        if self.show_fps {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("fps_overlay"),
            ));
            let text = format!("{:.0} FPS", self.fps);
            let font_id = egui::FontId::proportional(16.0);
            let galley = painter.layout_no_wrap(text, font_id, egui::Color32::YELLOW);
            let screen = ctx.screen_rect();
            let padding = egui::vec2(8.0, 4.0);
            let text_pos = egui::pos2(
                screen.max.x - galley.size().x - padding.x - 10.0,
                5.0 + padding.y,
            );
            let bg_rect = egui::Rect::from_min_size(text_pos - padding, galley.size() + padding * 2.0);
            painter.rect_filled(
                bg_rect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140),
            );
            painter.galley(text_pos, galley, egui::Color32::YELLOW);
        }

        //******************************************************************/
        //
        // Draw FOV debug rays when freeze culling is active in free cam
        //
        //******************************************************************/
        if self.is_free_cam && self.freeze_culling {
            if let Some(ref saved) = self.saved_player_camera {
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("fov_rays"),
                ));
                let screen = ctx.screen_rect();
                let sw = self.data.swapchain_extent.width as f32;
                let sh = self.data.swapchain_extent.height as f32;
                let aspect = sw / sh;

                let eye_offset = 0.0f32;
                let free_view = self.camera.view_matrix_with_offset(eye_offset);
                #[rustfmt::skip]
                let correction = Mat4::new(
                    1.0,  0.0,       0.0, 0.0,
                    0.0, -1.0,       0.0, 0.0,
                    0.0,  0.0, 1.0 / 2.0, 0.0,
                    0.0,  0.0, 1.0 / 2.0, 1.0,
                );
                let free_proj =
                    correction * cgmath::perspective(Deg(45.0), aspect, 0.01, RENDER_FAR_PLANE);
                let vp = free_proj * free_view;

                let saved_eye = vec3(
                    saved.position.x,
                    saved.position.y,
                    saved.position.z + self.eye_offset_walk,
                );
                let half_fov_rad = (self.player_fov * 0.5f32).to_radians();
                let saved_front = saved.front();
                let saved_right = saved.right();
                let ray_len = 50.0f32;

                let front_flat = vec3(saved_front.x, saved_front.y, 0.0).normalize();
                let right_flat = vec3(saved_right.x, saved_right.y, 0.0).normalize();
                let left_dir = (front_flat * half_fov_rad.cos()
                    - right_flat * half_fov_rad.sin())
                .normalize();
                let right_dir = (front_flat * half_fov_rad.cos()
                    + right_flat * half_fov_rad.sin())
                .normalize();

                let origin = saved_eye;
                let left_end = origin + left_dir * ray_len;
                let right_end = origin + right_dir * ray_len;

                let project = |world_pos: Vec3| -> Option<egui::Pos2> {
                    let p = vp * cgmath::vec4(world_pos.x, world_pos.y, world_pos.z, 1.0);
                    if p.w <= 0.001 {
                        return None;
                    }
                    let ndc_x = p.x / p.w;
                    let ndc_y = p.y / p.w;
                    let sx = (ndc_x + 1.0) * 0.5 * screen.width() + screen.min.x;
                    let sy = (ndc_y + 1.0) * 0.5 * screen.height() + screen.min.y;
                    Some(egui::pos2(sx, sy))
                };

                let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 255, 0));
                if let (Some(p0), Some(p1)) = (project(origin.into()), project(left_end.into())) {
                    painter.line_segment([p0, p1], stroke);
                }
                if let (Some(p0), Some(p1)) = (project(origin.into()), project(right_end.into())) {
                    painter.line_segment([p0, p1], stroke);
                }
                let front_end = origin + front_flat * ray_len;
                let front_stroke =
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(0, 180, 0));
                if let (Some(p0), Some(p1)) = (project(origin.into()), project(front_end.into())) {
                    painter.line_segment([p0, p1], front_stroke);
                }
            }
        }

        //******************************************************************/
        //
        // World chooser panel
        //
        //******************************************************************/
        if self.world_chooser.visible {
            egui::TopBottomPanel::top("world_chooser")
                .frame(
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0))
                        .inner_margin(egui::Margin::same(10.0)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading(
                            egui::RichText::new("World Select").color(egui::Color32::WHITE),
                        );
                        ui.add_space(20.0);
                        ui.label(
                            egui::RichText::new("Select a map to load:")
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                    });

                    //******************************************************************/
                    //
                    // Settings toggles
                    //
                    //******************************************************************/
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Player Mode:")
                                .color(egui::Color32::LIGHT_GRAY),
                        );
                        let prev = self.is_free_cam;
                        if ui
                            .checkbox(&mut self.is_free_cam, "Free Camera")
                            .changed()
                        {
                            if self.is_free_cam != prev {
                                if self.is_free_cam && self.freeze_culling {
                                    self.saved_player_camera = Some(self.camera.clone());
                                } else if !self.is_free_cam && self.freeze_culling {
                                    if let Some(saved) = self.saved_player_camera.take() {
                                        self.camera = saved;
                                    }
                                }
                                self.player_mode = if self.is_free_cam {
                                    collision::PlayerMode::Flying
                                } else {
                                    collision::PlayerMode::Walk
                                };
                                if !self.is_free_cam {
                                    self.move_state.z_vel = 0.0;
                                    self.move_state.velocity = [0.0, 0.0];
                                    let gz = self.height_provider.ground_height(
                                        self.camera.position.x,
                                        self.camera.position.y,
                                        Some(self.camera.position.z),
                                    );
                                    let min_z = gz + 0.35;
                                    if self.camera.position.z < min_z {
                                        self.camera.position.z = min_z;
                                    }
                                    self.move_state.on_ground = true;
                                }
                            }
                        }
                        ui.add_space(20.0);
                        ui.checkbox(&mut self.show_fps, "Show FPS");
                        ui.add_space(20.0);
                        let prev_vsync = self.vsync;
                        ui.checkbox(&mut self.vsync, "VSync");
                        if self.vsync != prev_vsync {
                            self.data.vsync = self.vsync;
                            self.resized = true;
                        }
                        ui.add_space(20.0);
                        let prev_freeze = self.freeze_culling;
                        if ui
                            .checkbox(&mut self.freeze_culling, "Freeze Culling")
                            .changed()
                        {
                            if self.freeze_culling && self.is_free_cam {
                                self.saved_player_camera = Some(self.camera.clone());
                            } else if !self.freeze_culling {
                                self.saved_player_camera = None;
                            }
                        }
                        ui.add_space(20.0);
                        if ui.checkbox(&mut self.dynamic_lighting, "Dynamic Lighting").changed() {
                            if self.dynamic_lighting {
                                // Rebuild the cached UBO (restores lights/shadows)
                                let shadow_pos = self.game_objects.shadow_caster_positions();
                                self.cached_light_ubo = Self::build_light_ubo(
                                    &self.world_lights,
                                    &shadow_pos,
                                    self.fog_color,
                                    self.fog_near,
                                    self.fog_far,
                                    self.fog_enabled,
                                    self.sky_fog_far,
                                );
                            } else {
                                // Disable dynamic lighting by zeroing counts
                                self.cached_light_ubo.light_count = 0;
                                self.cached_light_ubo.shadow_count = 0;
                            }
                            unsafe { self.upload_light_ubo_to_all(); }

                            // Persist the setting so it survives restarts
                            if let Err(e) = (crate::settings::Settings { dynamic_lighting: self.dynamic_lighting }).save() {
                                warn!("Failed to save settings: {}", e);
                            }
                        }
                        ui.add_space(20.0);
                        if ui.checkbox(&mut self.bounded_world, "Bounded World").changed() {
                            if self.bounded_world {
                                // Immediately clamp player into bounds when enabling
                                let margin = 0.5_f32;
                                self.camera.position.x = self.camera.position.x.clamp(
                                    self.world_bounds_min[0] + margin,
                                    self.world_bounds_max[0] - margin,
                                );
                                self.camera.position.y = self.camera.position.y.clamp(
                                    self.world_bounds_min[1] + margin,
                                    self.world_bounds_max[1] - margin,
                                );
                                self.camera.position.z = self.camera.position.z.clamp(
                                    self.world_bounds_min[2] + 0.1,
                                    self.world_bounds_max[2] - 0.1,
                                );
                            }
                        }
                    });

                    ui.add_space(5.0);
                    ui.separator();
                    ui.add_space(5.0);
                    
                    //******************************************************************/
                    //
                    // Fog controls
                    //
                    //******************************************************************/
                    ui.label(egui::RichText::new("Fog").color(egui::Color32::LIGHT_GRAY));
                    let fog_changed = {
                        let prev = self.fog_enabled;
                        ui.checkbox(&mut self.fog_enabled, "Enable Fog");
                        let mut changed = self.fog_enabled != prev;

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Near").color(egui::Color32::GRAY));
                            changed |= ui.add(
                                egui::Slider::new(&mut self.fog_near, 0.0..=50.0)
                                    .fixed_decimals(1)
                                    .text(""),
                            ).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Far ").color(egui::Color32::GRAY));
                            changed |= ui.add(
                                egui::Slider::new(&mut self.fog_far, 1.0..=200.0)
                                    .fixed_decimals(1)
                                    .text(""),
                            ).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Color").color(egui::Color32::GRAY));
                            let mut rgba = egui::Color32::from_rgb(
                                (self.fog_color[0] * 255.0) as u8,
                                (self.fog_color[1] * 255.0) as u8,
                                (self.fog_color[2] * 255.0) as u8,
                            );
                            if ui.color_edit_button_srgba(&mut rgba).changed() {
                                self.fog_color = [
                                    rgba.r() as f32 / 255.0,
                                    rgba.g() as f32 / 255.0,
                                    rgba.b() as f32 / 255.0,
                                ];
                                changed = true;
                            }
                        });
                        changed
                    };
                    if fog_changed {
                        // Rebuild and upload the lighting UBO immediately so the change is
                        // visible next frame without waiting for a dynamic-light tick.
                        let shadow_pos = self.game_objects.shadow_caster_positions();
                        self.cached_light_ubo = Self::build_light_ubo(
                            &self.world_lights, &shadow_pos, self.fog_color, self.fog_near, self.fog_far, self.fog_enabled, self.sky_fog_far);
                        unsafe { self.upload_light_ubo_to_all(); }
                    }

                    ui.add_space(5.0);
                    ui.separator();
                    ui.add_space(5.0);

                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            let worlds = self.world_chooser.worlds.clone();
                            for (i, world_path) in worlds.iter().enumerate() {
                                let display_name =
                                    WorldChooser::get_world_display_name(world_path);
                                let is_selected = i == self.world_chooser.selected_index;
                                let is_current = *world_path == self.current_world;

                                let text = if is_current {
                                    egui::RichText::new(format!(
                                        "{} (current)",
                                        display_name
                                    ))
                                    .color(egui::Color32::LIGHT_GREEN)
                                } else {
                                    egui::RichText::new(&display_name).color(
                                        if is_selected {
                                            egui::Color32::YELLOW
                                        } else {
                                            egui::Color32::WHITE
                                        },
                                    )
                                };

                                let orig_style = ui.style().clone();
                                let mut style = (*orig_style).clone();
                                style.visuals.selection.bg_fill =
                                    egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
                                ui.set_style(std::sync::Arc::new(style));
                                let response = ui.selectable_label(is_selected, text);
                                ui.set_style(orig_style);

                                if response.clicked() {
                                    self.world_chooser.select_index(i);
                                    println!("Selected: {}", display_name);
                                }

                                if response.double_clicked() {
                                    println!("Loading: {}", display_name);
                                    if let Some(path) = self.world_chooser.confirm_selection() {
                                        self.world_chooser.pending_load = Some(path);
                                        *mouse_locked = true;
                                        // Persist settings when closing the world chooser via double-click load
                                        if let Err(e) = (crate::settings::Settings { dynamic_lighting: self.dynamic_lighting }).save() {
                                            warn!("Failed to save settings: {}", e);
                                        }
                                        if !self.dynamic_lighting {
                                            self.cached_light_ubo.light_count = 0;
                                            self.cached_light_ubo.shadow_count = 0;
                                            unsafe { self.upload_light_ubo_to_all(); }
                                        }
                                    }
                                }
                            }
                        });

                    ui.add_space(5.0);
                    ui.separator();
                    ui.add_space(5.0);

                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new("Load Selected")
                                    .fill(egui::Color32::TRANSPARENT),
                            )
                            .clicked()
                        {
                            if let Some(path) = self.world_chooser.confirm_selection() {
                                self.world_chooser.pending_load = Some(path);
                                *mouse_locked = true;
                                // Persist settings when closing the world chooser via Load Selected
                                if let Err(e) = (crate::settings::Settings { dynamic_lighting: self.dynamic_lighting }).save() {
                                    warn!("Failed to save settings: {}", e);
                                }
                                if !self.dynamic_lighting {
                                    self.cached_light_ubo.light_count = 0;
                                    self.cached_light_ubo.shadow_count = 0;
                                    unsafe { self.upload_light_ubo_to_all(); }
                                }
                            }
                        }

                        ui.add_space(10.0);

                        let selected_name = self
                            .world_chooser
                            .worlds
                            .get(self.world_chooser.selected_index)
                            .map(|p| WorldChooser::get_world_display_name(p))
                            .unwrap_or_default();
                        ui.label(
                            egui::RichText::new(format!("Selected: {}", selected_name))
                                .color(egui::Color32::GRAY),
                        );
                    });
                });
        }
    }

    /// Reloads a new world file
    pub unsafe fn reload_world(
        &mut self,
        world_path: &str,
        _egui_renderer: &mut egui_renderer::EguiRenderer,
    ) -> Result<()> {
        info!("Loading world: {}", world_path);

        self.device.device_wait_idle()?;

        //******************************************************************/
        //
        // Destruction
        //
        //******************************************************************/

        // Destroy old level textures
        for texture in &self.data.level_textures {
            self.device.destroy_image_view(texture.view, None);
            self.device.free_memory(texture.memory, None);
            self.device.destroy_image(texture.image, None);
        }
        self.data.level_textures.clear();

        // Destroy old vertex/index buffers
        self.device
            .free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);

        // Destroy old texture resources
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);

        // Clear old data
        self.data.vertices.clear();
        self.data.indices.clear();
        self.data.draw_groups.clear();

        //******************************************************************/
        //
        // Load new world
        //
        //******************************************************************/

        let loaded = load_dat_model(&mut self.data, world_path, 0, 0.01)?;
        self.world_lights = loaded.lights;
        self.game_objects = loaded.game_objects;
        self.entity_cylinders = loaded.entity_cylinders;
        self.trigger_draw_groups = loaded.trigger_draw_groups;
        self.elapsed_time = 0.0;
        let (world_min, world_max) = Self::compute_non_sky_world_bounds(&self.data);
        self.world_bounds_min = world_min;
        self.world_bounds_max = world_max;
        // Hide triggers by default on world load
        if !self.show_triggers {
            for &(dg_idx, _) in &self.trigger_draw_groups {
                if let Some(dg) = self.data.draw_groups.get_mut(dg_idx) {
                    dg.index_count = 0;
                }
            }
        }
        if let Some(fog) = loaded.fog {
            self.fog_enabled = fog.enabled;
            self.fog_near = fog.near_z;
            self.fog_far = fog.far_z;
            self.fog_color = fog.color;
            self.sky_fog_far = fog.sky_fog_far;
        }
        let shadow_pos = self.game_objects.shadow_caster_positions();
        self.cached_light_ubo = Self::build_light_ubo(
            &self.world_lights, &shadow_pos, self.fog_color, self.fog_near, self.fog_far, self.fog_enabled, self.sky_fog_far);
        // Respect user's dynamic lighting preference when applying the freshly
        // built UBO so loading a world does not re-enable blob shadows/lights.
        if !self.dynamic_lighting {
            self.cached_light_ubo.light_count = 0;
            self.cached_light_ubo.shadow_count = 0;
        }
        self.upload_light_ubo_to_all();

        //******************************************************************/
        //
        // Install mesh-backed height provider (uses collision mesh which includes invisible surfaces)
        //
        //******************************************************************/

        if !loaded.collision_positions.is_empty() && !loaded.collision_indices.is_empty() {
            let mesh = collision::MeshHeightProvider::new(loaded.collision_positions, loaded.collision_indices);
            self.height_provider = Box::new(mesh.clone());
            self.mesh_provider = Some(mesh);
        } else {
            self.height_provider = Box::new(collision::FlatGround);
            self.mesh_provider = None;
        }

        //******************************************************************/
        //
        // Recreation
        //
        //******************************************************************/


        // Recreate texture resources

        vulkan::create_texture_image(&self.instance, &self.device, &mut self.data)?;
        vulkan::create_texture_image_view(&self.device, &mut self.data)?;
        vulkan::create_texture_sampler(&self.device, &mut self.data)?;


        // Recreate buffers

        vulkan::create_vertex_buffer(&self.instance, &self.device, &mut self.data)?;
        vulkan::create_index_buffer(&self.instance, &self.device, &mut self.data)?;


        // Recreate descriptor sets

        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        vulkan::create_descriptor_pool(&self.device, &mut self.data)?;
        vulkan::create_descriptor_sets(&self.device, &mut self.data)?;


        // Reset camera

        self.camera = Camera::default();


        // Update current world

        self.current_world = world_path.to_string();


        // Rebuild occlusion culling AABBs

        {
            let positions: Vec<[f32; 3]> = self
                .data
                .vertices
                .iter()
                .map(|v| [v.pos.x, v.pos.y, v.pos.z])
                .collect();
            let groups: Vec<(u32, u32)> = self
                .data
                .draw_groups
                .iter()
                .map(|g| (g.first_index, g.index_count))
                .collect();
            self.occlusion_culler
                .build_from_groups(&positions, &self.data.indices, &groups);
        }

        info!("World loaded successfully: {}", world_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_world_path;

    #[test]
    fn resolves_realm_world_paths() {
        let resolved = resolve_world_path("worlds\\realm1\\R1M1b");
        assert!(resolved.is_some());
        let resolved = resolved.unwrap();
        assert!(resolved.to_ascii_lowercase().ends_with("r1m1b.dat"));
    }
}

impl App {

    //******************************************************************/
    //
    // Updates the uniform buffer object for our Vulkan app.
    //
    //******************************************************************/
    unsafe fn update_uniform_buffer(&mut self, image_index: usize) -> Result<()> {
        let eye_offset = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
            self.eye_offset_walk
        } else {
            0.0
        };
        let view = self.camera.view_matrix_with_offset(eye_offset);

        // Correction matrix is constant — computed once with const values
        #[rustfmt::skip]
        const CORRECTION: [[f32; 4]; 4] = [
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0,  0.5, 0.0],
            [0.0,  0.0,  0.5, 1.0],
        ];
        let correction: Mat4 = CORRECTION.into();

        let fov_deg = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
            self.player_fov
        } else {
            45.0
        };

        let proj = correction
            * cgmath::perspective(
                Deg(fov_deg),
                self.data.swapchain_extent.width as f32
                    / self.data.swapchain_extent.height as f32,
                0.01,
                RENDER_FAR_PLANE,
            );

        let ubo = UniformBufferObject { view, proj };

        // Write directly to persistently mapped memory (no map/unmap overhead)
        memcpy(&ubo, self.data.uniform_buffers_mapped[image_index].cast(), 1);

        //******************************************************************/
        //
        // Upload lighting UBO for the current swapchain image only.
        // Use the cached UBO and set camera_pos here so we avoid writing
        // into buffers that the GPU may be reading for other in-flight
        // frames (this prevents visual corruption / flicker).
        //
        //******************************************************************/
        {
            let mut ubo = self.cached_light_ubo.clone();
            // Enforce the current `dynamic_lighting` state before uploading
            // so a stale `cached_light_ubo` cannot re-enable lights.
            if !self.dynamic_lighting {
                if ubo.light_count != 0 || ubo.shadow_count != 0 {
                    warn!("dynamic_lighting is disabled but UBO contains lights/shadows (lc={}, sc={}) — zeroing before upload", ubo.light_count, ubo.shadow_count);
                }
                ubo.light_count = 0;
                ubo.shadow_count = 0;
            }
            let cam_pos = self.camera.position;
            ubo.camera_pos = [cam_pos.x, cam_pos.y, cam_pos.z, 0.0];
            let dst = self.data.light_uniform_buffers_mapped[image_index] as *mut LightingUBO;
            memcpy(&ubo, dst.cast(), 1);
        }

        Ok(())
    }

    //******************************************************************/
    //
    // Recreates the swapchain for our Vulkan app.
    //
    //******************************************************************/
    #[rustfmt::skip]
    pub unsafe fn recreate_swapchain(&mut self, window: &Window, egui_renderer: &mut egui_renderer::EguiRenderer) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        vulkan::create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        vulkan::create_swapchain_image_views(&self.device, &mut self.data)?;
        vulkan::create_render_pass(&self.instance, &self.device, &mut self.data)?;
        vulkan::create_pipeline(&self.device, &mut self.data)?;
        vulkan::create_color_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan::create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        vulkan::create_framebuffers(&self.device, &mut self.data)?;
        vulkan::create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        // Re-upload light UBO into freshly mapped buffers
        self.upload_light_ubo_to_all();
        vulkan::create_descriptor_pool(&self.device, &mut self.data)?;
        vulkan::create_descriptor_sets(&self.device, &mut self.data)?;
        vulkan::create_command_buffers(&self.device, &mut self.data)?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        
        egui_renderer.resize(
            &self.device,
            &self.data.swapchain_image_views,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
        )?;
        
        Ok(())
    }

    //******************************************************************/
    //
    // Destroys our Vulkan app.
    //
    //******************************************************************/
    #[rustfmt::skip]
    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        for texture in &self.data.level_textures {
            self.device.destroy_image_view(texture.view, None);
            self.device.free_memory(texture.memory, None);
            self.device.destroy_image(texture.image, None);
        }
        self.data.level_textures.clear();

        self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.command_pools.iter().for_each(|p| self.device.destroy_command_pool(*p, None));
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
        self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    //******************************************************************/
    //
    // Destroys the parts of our Vulkan app related to the swapchain.
    //
    //******************************************************************/
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        // Unmap persistently mapped UBO memory before freeing
        for m in &self.data.uniform_buffers_memory { self.device.unmap_memory(*m); }
        self.data.uniform_buffers_mapped.clear();
        self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        for m in &self.data.light_uniform_buffers_memory { self.device.unmap_memory(*m); }
        self.data.light_uniform_buffers_mapped.clear();
        self.data.light_uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.light_uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline(self.data.sky_pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}