#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

// DAT reader modules (import from the library crate)
use rustKPC::dat;
use rustKPC::dat_mesh;
use rustKPC::dtx;
use rustKPC::egui_renderer;
use rustKPC::collision;

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use anyhow::{Result, anyhow};
use cgmath::{Deg, InnerSpace, vec2, vec3};
use log::*;
use thiserror::Error;
use vulkanalia::Version;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder, CursorGrabMode};

use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
/// The Vulkan SDK version that started requiring the portability subset extension for macOS.
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// The maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Movement speeds
const WALK_SPEED: f32 = 6.0; // player walking speed
const FLY_SPEED: f32 = 10.0; // free camera / flying speed (original)
/// Mouse sensitivity
const MOUSE_SENSITIVITY: f32 = 0.1;

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

/// Camera state for FPS-style controls
#[derive(Clone, Debug)]
struct Camera {
    position: Vec3,
    yaw: f32,   // Horizontal rotation (degrees)
    pitch: f32, // Vertical rotation (degrees)
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: vec3(0.0, -50.0, 12.0), // Start position (slightly higher)
            yaw: 90.0,   // Looking forward (+Y)
            pitch: 0.0,  // Level
        }
    }
}

impl Camera {
    fn front(&self) -> Vec3 {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        vec3(
            yaw_rad.cos() * pitch_rad.cos(),
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
        ).normalize()
    }

    fn right(&self) -> Vec3 {
        self.front().cross(vec3(0.0, 0.0, 1.0)).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        self.view_matrix_with_offset(0.0)
    }

    fn view_matrix_with_offset(&self, offset_z: f32) -> Mat4 {
        let eye = vec3(self.position.x, self.position.y, self.position.z + offset_z);
        let front = self.front();
        let target = eye + front;
        Mat4::look_at_rh(
            cgmath::Point3::new(eye.x, eye.y, eye.z),
            cgmath::Point3::new(target.x, target.y, target.z),
            vec3(0.0, 0.0, 1.0),
        )
    }
}

/// Input state tracking
#[derive(Clone, Debug, Default)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

/// Application loading state
#[derive(Clone, Debug, PartialEq)]
enum LoadingState {
    Loading(String),  // Currently loading a map
    Ready,            // Map loaded and ready
}

impl Default for LoadingState {
    fn default() -> Self {
        LoadingState::Ready
    }
}

/// World chooser UI state
#[derive(Clone, Debug)]
struct WorldChooser {
    visible: bool,
    worlds: Vec<String>,  // List of world file paths
    selected_index: usize,
    pending_load: Option<String>,  // World to load on next frame
    scroll_offset: f32,  // For scrolling through long lists
}

impl WorldChooser {
    fn new() -> Self {
        let worlds = Self::scan_worlds();
        Self {
            visible: false,
            worlds,
            selected_index: 0,
            pending_load: None,
            scroll_offset: 0.0,
        }
    }

    fn scan_worlds() -> Vec<String> {
        let mut worlds = Vec::new();
        let base_path = std::path::Path::new("REZ/WORLDS");
        
        if let Ok(entries) = std::fs::read_dir(base_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(files) = std::fs::read_dir(&path) {
                        for file in files.filter_map(|f| f.ok()) {
                            let file_path = file.path();
                            if let Some(ext) = file_path.extension() {
                                if ext.eq_ignore_ascii_case("dat") {
                                    if let Some(path_str) = file_path.to_str() {
                                        worlds.push(path_str.replace('\\', "/"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        worlds.sort();
        worlds
    }

    fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    fn select_index(&mut self, index: usize) {
        if index < self.worlds.len() {
            self.selected_index = index;
        }
    }

    fn confirm_selection(&mut self) -> Option<String> {
        if let Some(world) = self.worlds.get(self.selected_index) {
            self.visible = false;
            Some(world.clone())
        } else {
            None
        }
    }

    fn get_world_display_name(path: &str) -> String {
        std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .to_string()
    }

    fn take_pending_load(&mut self) -> Option<String> {
        self.pending_load.take()
    }
}

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Loading: R1M1A.DAT...")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // Set up egui for UI
    let egui_ctx = egui::Context::default();
    // Set global visuals to fully transparent backgrounds and selection highlights
    let mut visuals = egui::Visuals::dark();
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.faint_bg_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.extreme_bg_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    visuals.window_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
    egui_ctx.set_visuals(visuals);
    let viewport_id = egui_ctx.viewport_id();
    let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None);

    // Run egui once to initialize fonts (required before accessing font texture)
    let _ = egui_ctx.run(egui::RawInput::default(), |_ctx| {});

    // Capture mouse cursor initially
    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
    window.set_cursor_visible(false);

    // Create App
    let mut app = unsafe { App::create(&window)? };
    
    // Create egui renderer (after app is created so we have Vulkan resources)
    let mut egui_renderer = unsafe {
        egui_renderer::EguiRenderer::new(
            &app.instance,
            &app.device,
            app.data.physical_device,
            &app.data.swapchain_image_views,
            app.data.swapchain_format,
            app.data.command_pool,
            app.data.graphics_queue,
            &egui_ctx,
            app.data.swapchain_extent.width,
            app.data.swapchain_extent.height,
        )?
    };
    
    let mut minimized = false;
    let mut last_time = Instant::now();
    let mut mouse_locked = true;
    
    // Set initial title after load
    window.set_title("KISS Psycho Circus: The Nightmare Child [F1: World Select]");
    
    event_loop.run(move |event, elwt| {
        // Let egui handle events when UI is visible
        if app.world_chooser.visible {
            if let Event::WindowEvent { event: ref window_event, .. } = event {
                let response = egui_state.on_window_event(&window, window_event);
                if response.consumed {
                    return;
                }
            }
        }
        
        match event {
            Event::AboutToWait => window.request_redraw(),
            
            // Handle raw mouse motion (only when mouse is locked and UI not visible)
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if mouse_locked && !app.world_chooser.visible && app.loading_state == LoadingState::Ready {
                    app.camera.yaw -= delta.0 as f32 * MOUSE_SENSITIVITY;
                    app.camera.pitch -= delta.1 as f32 * MOUSE_SENSITIVITY;
                    app.camera.pitch = app.camera.pitch.clamp(-89.0, 89.0);
                }
            }
            
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    let now = Instant::now();
                    let dt = (now - last_time).as_secs_f32();
                    last_time = now;
                    let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                    app.fps = fps;
                    
                    // Check for pending world load
                    if let Some(world_path) = app.world_chooser.take_pending_load() {
                        let map_name = WorldChooser::get_world_display_name(&world_path);
                        app.loading_state = LoadingState::Loading(map_name.clone());
                        window.set_title(&format!("Loading: {}...", map_name));
                        
                        // Run egui for loading screen
                        let raw_input = egui_state.take_egui_input(&window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            app.run_ui(ctx, &mut mouse_locked);
                        });
                        egui_state.handle_platform_output(&window, full_output.platform_output);
                        let clipped_primitives = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                        
                        // Render loading screen
                        unsafe { app.render(&window, &mut egui_renderer, &clipped_primitives, full_output.pixels_per_point) }.unwrap();
                        
                        // Load the world
                        if let Err(e) = unsafe { app.reload_world(&world_path, &mut egui_renderer) } {
                            error!("Failed to load world {}: {}", world_path, e);
                        }
                        app.loading_state = LoadingState::Ready;
                        window.set_title("KISS Psycho Circus: The Nightmare Child [F1: World Select]");
                    }
                    
                    // Run egui UI
                    let raw_input = egui_state.take_egui_input(&window);
                    let full_output = egui_ctx.run(raw_input, |ctx| {
                        app.run_ui(ctx, &mut mouse_locked);
                    });
                    egui_state.handle_platform_output(&window, full_output.platform_output);
                    
                    // Tessellate egui shapes into primitives for rendering
                    let clipped_primitives = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                    
                    // Update window title based on UI state
                    if app.world_chooser.visible {
                        let selected_name = app.world_chooser.worlds
                            .get(app.world_chooser.selected_index)
                            .map(|p| WorldChooser::get_world_display_name(p))
                            .unwrap_or_default();
                        window.set_title(&format!(
                            "World Chooser [{}/{}]: {} | Click to select, double-click to load | F1/Esc to close",
                            app.world_chooser.selected_index + 1,
                            app.world_chooser.worlds.len(),
                            selected_name
                        ));
                    }
                    
                    // Update mouse grab based on UI state
                    if app.world_chooser.visible && mouse_locked {
                        mouse_locked = false;
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                    }
                    
                    // Update camera (only when UI is hidden)
                    if !app.world_chooser.visible && app.loading_state == LoadingState::Ready {
                        app.update_camera(dt);
                    }
                    unsafe { app.render(&window, &mut egui_renderer, &clipped_primitives, full_output.pixels_per_point) }.unwrap();
                }
                
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true;
                    }
                }
                
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe { egui_renderer.destroy(&app.device); }
                    unsafe { app.destroy(); }
                }
                
                WindowEvent::KeyboardInput { event, .. } => {
                    if app.loading_state != LoadingState::Ready {
                        return;
                    }
                    
                    let pressed = event.state == ElementState::Pressed;
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::F1) if pressed => {
                            // Open/close the world chooser (F1)
                            app.world_chooser.toggle();
                            if app.world_chooser.visible {
                                mouse_locked = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                                // ensure collisions are active so player doesn't fall through while UI open
                                app.player_mode = collision::PlayerMode::Walk;
                            } else {
                                mouse_locked = true;
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyW) if !app.world_chooser.visible => app.input.forward = pressed,
                        PhysicalKey::Code(KeyCode::KeyS) if !app.world_chooser.visible => app.input.backward = pressed,
                        PhysicalKey::Code(KeyCode::KeyA) if !app.world_chooser.visible => app.input.left = pressed,
                        PhysicalKey::Code(KeyCode::KeyD) if !app.world_chooser.visible => app.input.right = pressed,
                        PhysicalKey::Code(KeyCode::Space) if !app.world_chooser.visible => app.input.up = pressed,
                        PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) if !app.world_chooser.visible => app.input.down = pressed,
                        PhysicalKey::Code(KeyCode::Escape) if pressed => {
                            if app.world_chooser.visible {
                                app.world_chooser.visible = false;
                                mouse_locked = true;
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
                            } else {
                                mouse_locked = !mouse_locked;
                                if mouse_locked {
                                    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                    window.set_cursor_visible(false);
                                } else {
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                            }
                        }
                        _ => { }
                    }
                }
                _ => {}
            }
            _ => {}
        }
    })?;

    Ok(())
}

/// Our Vulkan app.
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
    models: usize,
    camera: Camera,
    input: InputState,
    on_ground: bool,
    world_chooser: WorldChooser,
    current_world: String,
    loading_state: LoadingState,
    player_mode: collision::PlayerMode,
    height_provider: Box<dyn collision::HeightProvider>,
    mesh_provider: Option<collision::MeshHeightProvider>,
    // Physics
    z_vel: f32,
    is_free_cam: bool,
    eye_offset_walk: f32,
    player_fov: f32,
    fps: f32,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_command_pools(&instance, &device, &mut data)?;
        create_color_objects(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        
        // Load DAT model - this populates texture names, vertices, indices, and draw groups
        // Texture dimensions are looked up during loading for UV scaling
        load_dat_model(&mut data, "REZ/WORLDS/REALM1/R1M1A.DAT", 0, 0.01)?;
        
        // Now load textures to GPU (uses level_textures names populated by load_dat_model)
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;
        
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        let initial_world = "REZ/WORLDS/REALM1/R1M1A.DAT".to_string();

        // Prepare mesh-backed provider if possible (clone only position/index arrays)
        let (initial_height_provider, initial_mesh_provider) = if !data.vertices.is_empty() && !data.indices.is_empty() {
            let positions = data.vertices.iter().map(|v| v.pos).collect::<Vec<_>>();
            let indices = data.indices.clone();
            let mesh = collision::MeshHeightProvider::new(positions.clone(), indices.clone());
            (Box::new(mesh.clone()) as Box<dyn collision::HeightProvider>, Some(mesh))
        } else {
            (Box::new(collision::FlatGround) as Box<dyn collision::HeightProvider>, None)
        };

        Ok(Self {
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
            world_chooser: WorldChooser::new(),
            current_world: initial_world,
            loading_state: LoadingState::Ready,
            player_mode: collision::PlayerMode::Walk,
            height_provider: initial_height_provider,
            mesh_provider: initial_mesh_provider,
            z_vel: 0.0,
            is_free_cam: false,
            eye_offset_walk: 0.4,
            player_fov: 60.0,
            fps: 0.0,
            on_ground: false,
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(
        &mut self,
        window: &Window,
        egui_renderer: &mut egui_renderer::EguiRenderer,
        egui_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window, egui_renderer),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device.wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_command_buffer(image_index, egui_renderer, egui_primitives, pixels_per_point)?;
        self.update_uniform_buffer(image_index)?;

        // Wait on the semaphore that was signaled by acquire, signal the render_finished for this frame
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

        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window, egui_renderer)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        // Cycle through frames based on swapchain image count
        self.frame = (self.frame + 1) % self.data.swapchain_image_count as usize;

        Ok(())
    }

    /// Updates a command buffer for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn update_command_buffer(
        &mut self,
        image_index: usize,
        egui_renderer: &mut egui_renderer::EguiRenderer,
        egui_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<()> {
        // Reset

        let command_pool = self.data.command_pools[image_index];
        self.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        // Commands
        let info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        // Use different clear color for loading screen
        let clear_color = if matches!(self.loading_state, LoadingState::Loading(_)) {
            [0.08, 0.08, 0.12, 1.0]  // Dark blue-ish for loading
        } else {
            [0.0, 0.0, 0.0, 1.0]  // Black for normal rendering
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


        // Only render geometry when not loading
        if self.loading_state == LoadingState::Ready {
            let secondary_command_buffers = (0..self.models)
                .map(|i| self.update_secondary_command_buffer(image_index, i))
                .collect::<Result<Vec<_>, _>>()?;
            self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
        }

        self.device.cmd_end_render_pass(command_buffer);

        // Render egui UI (in its own render pass that loads existing content)
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

    /// Updates a secondary command buffer for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        // Allocate

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

        // Model matrix - identity for level geometry
        let model = Mat4::from_scale(1.0);
        let model_bytes = std::slice::from_raw_parts(&model as *const Mat4 as *const u8, size_of::<Mat4>());

        let opacity = 1.0f32;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        // Commands
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);
        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(command_buffer, self.data.index_buffer, 0, vk::IndexType::UINT32);
        
        // Push constants (model matrix and opacity)
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );

        // Draw each texture group with its own texture
        if self.data.draw_groups.is_empty() {
            // Fallback: draw everything with default texture
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
            // Draw each group with its texture
            for group in &self.data.draw_groups {
                // Get the descriptor set for this texture
                let descriptor_set = if group.texture_index < self.data.level_textures.len() 
                    && !self.data.level_textures[group.texture_index].descriptor_sets.is_empty() 
                {
                    self.data.level_textures[group.texture_index].descriptor_sets[image_index]
                } else {
                    // Fallback to default texture
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
                    0,  // vertex_offset - we're using absolute indices
                    0
                );
            }
        }

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    /// Updates camera position based on input state
    fn update_camera(&mut self, dt: f32) {
        let speed = if self.player_mode == collision::PlayerMode::Flying { FLY_SPEED * dt } else { WALK_SPEED * dt };
        let front_flat = {
            let f = self.camera.front();
            vec3(f.x, f.y, 0.0).normalize()
        };
        let right = self.camera.right();
        let prev_pos = self.camera.position;

        self.on_ground = false;
        // ...existing code...
        if self.input.forward {
            self.camera.position = self.camera.position + front_flat * speed;
        }
        if self.input.backward {
            self.camera.position = self.camera.position - front_flat * speed;
        }
        if self.input.left {
            self.camera.position = self.camera.position - right * speed;
        }
        if self.input.right {
            self.camera.position = self.camera.position + right * speed;
        }
        // vertical movement only in Flying mode
        if self.player_mode == collision::PlayerMode::Flying {
            if self.input.up {
                self.camera.position.z += speed;
            }
            if self.input.down {
                self.camera.position.z -= speed;
            }
            self.z_vel = 0.0;
        } else {
            // Walk mode: apply gravity to vertical velocity and integrate
            const GRAVITY: f32 = 9.8 * 3.0;
            self.z_vel -= GRAVITY * dt;
            // Jumping (use on_ground flag)
            if self.input.up && self.on_ground {
                self.z_vel = 7.5; // jump velocity, tweak as needed
                self.input.up = false; // prevent holding space from multi-jumping
                self.on_ground = false;
            }
            self.camera.position.z += self.z_vel * dt;
        }

        // Apply wall collisions (horizontal) then ground collision when in Walk mode
        if self.player_mode == collision::PlayerMode::Walk {
            if let Some(mesh) = &self.mesh_provider {
                let before = self.camera.position;
                // Only pass 3 arguments: prev_pos, &mut self.camera.position, 0.25 (width)
                mesh.resolve_player_movement(prev_pos, &mut self.camera.position, 0.25);
                let horiz_blocked = (before.x != self.camera.position.x || before.y != self.camera.position.y)
                    && (self.camera.position.x == prev_pos.x && self.camera.position.y == prev_pos.y);
                if horiz_blocked {
                    let step_height = 0.9; // Further increased for small ledges
                    let mut try_pos = prev_pos;
                    try_pos.z += step_height;
                    let mut stepped_pos = try_pos;
                    mesh.resolve_player_movement(prev_pos, &mut stepped_pos, 0.25);
                    let ground_z = self.height_provider.ground_height(stepped_pos.x, stepped_pos.y, Some(stepped_pos.z));
                    let min_z = ground_z + 0.5; // keep hitbox tall for vertical offset
                    let dz = min_z - prev_pos.z;
                    if (stepped_pos.x != prev_pos.x || stepped_pos.y != prev_pos.y)
                        && dz > 0.0 && dz <= step_height + 1e-3
                    {
                        self.camera.position = stepped_pos;
                        self.camera.position.z = min_z;
                    }
                }
            }
            let before_z = self.camera.position.z;
            collision::resolve_player_collision(&mut self.camera.position, self.height_provider.as_ref(), 0.25, 0.5);
            let ground_z = self.height_provider.ground_height(self.camera.position.x, self.camera.position.y, Some(self.camera.position.z));
            let min_z = ground_z + 0.5;
            if self.camera.position.z < min_z - 0.05 {
                // Only snap down if clearly below ground
                self.camera.position.z = min_z;
                self.z_vel = 0.0;
                self.on_ground = true;
            } else if (self.camera.position.z - min_z).abs() < 0.1 || self.camera.position.z < min_z + 0.01 {
                // On or very near ground: stabilize position and velocity, no gravity
                self.camera.position.z = min_z;
                self.z_vel = 0.0;
                self.on_ground = true;
            }
        }
    }

    /// Run the egui UI
    fn run_ui(&mut self, ctx: &egui::Context, mouse_locked: &mut bool) {
        // Loading screen
        if let LoadingState::Loading(ref map_name) = self.loading_state {
            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::from_rgb(20, 20, 30)))
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(ui.available_height() / 2.0 - 40.0);
                            ui.heading(egui::RichText::new("Loading...").size(32.0).color(egui::Color32::WHITE));
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new(map_name).size(24.0).color(egui::Color32::LIGHT_GRAY));
                        });
                    });
                });
            return;
        }

        // World chooser panel
        if self.world_chooser.visible {
            egui::TopBottomPanel::top("world_chooser")
                .frame(egui::Frame::none()
                    .fill(egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0)) // fully transparent
                    .inner_margin(egui::Margin::same(10.0)))
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading(egui::RichText::new("World Select").color(egui::Color32::WHITE));
                        ui.add_space(20.0);
                        ui.label(egui::RichText::new("Select a map to load:").color(egui::Color32::LIGHT_GRAY));
                        // FPS label inside the F1 menu
                        ui.add_space(20.0);
                        ui.label(egui::RichText::new(format!("FPS: {:.1}", self.fps)).color(egui::Color32::YELLOW));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Close (F1)").clicked() {
                                self.world_chooser.visible = false;
                                *mouse_locked = true;
                                // Do not reset is_free_cam or player_mode here, preserve toggle state
                            }
                        });
                    });

                    // Player mode controls (Free Camera toggle)
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Player Mode:").color(egui::Color32::LIGHT_GRAY));
                        let prev = self.is_free_cam;
                        if ui.checkbox(&mut self.is_free_cam, "Free Camera").changed() {
                            if self.is_free_cam != prev {
                                self.player_mode = if self.is_free_cam { collision::PlayerMode::Flying } else { collision::PlayerMode::Walk };
                                if !self.is_free_cam { self.z_vel = 0.0; }
                            }
                        }
                    });
                    
                    ui.add_space(5.0);
                    ui.separator();
                    ui.add_space(5.0);
                    
                    // Scrollable list of worlds, with fully transparent background
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            let worlds = self.world_chooser.worlds.clone();
                            for (i, world_path) in worlds.iter().enumerate() {
                                let display_name = WorldChooser::get_world_display_name(world_path);
                                let is_selected = i == self.world_chooser.selected_index;
                                let is_current = *world_path == self.current_world;
                                
                                let text = if is_current {
                                    egui::RichText::new(format!("{} (current)", display_name))
                                        .color(egui::Color32::LIGHT_GREEN)
                                } else {
                                    egui::RichText::new(&display_name)
                                        .color(if is_selected { egui::Color32::YELLOW } else { egui::Color32::WHITE })
                                };
                                
                                // Remove blue selection background for this label
                                let orig_style = ui.style().clone();
                                let mut style = (*orig_style).clone();
                                style.visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
                                ui.set_style(std::sync::Arc::new(style));
                                let response = ui.selectable_label(is_selected, text);
                                // Restore visuals after
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
                                    }
                                }
                            }
                        });
                    
                    ui.add_space(5.0);
                    ui.separator();
                    ui.add_space(5.0);
                    
                    ui.horizontal(|ui| {
                        if ui.add(egui::Button::new("Load Selected").fill(egui::Color32::TRANSPARENT)).clicked() {
                            if let Some(path) = self.world_chooser.confirm_selection() {
                                self.world_chooser.pending_load = Some(path);
                                *mouse_locked = true;
                            }
                        }
                        
                        ui.add_space(10.0);
                        
                        let selected_name = self.world_chooser.worlds
                            .get(self.world_chooser.selected_index)
                            .map(|p| WorldChooser::get_world_display_name(p))
                            .unwrap_or_default();
                        ui.label(egui::RichText::new(format!("Selected: {}", selected_name)).color(egui::Color32::GRAY));
                    });
                });
        }

        // ...overlay removed...
    }

    /// Reloads a new world file
    unsafe fn reload_world(&mut self, world_path: &str, _egui_renderer: &mut egui_renderer::EguiRenderer) -> Result<()> {
        info!("Loading world: {}", world_path);
        
        // Wait for device to be idle before modifying resources
        self.device.device_wait_idle()?;

        // Destroy old level textures
        for texture in &self.data.level_textures {
            self.device.destroy_image_view(texture.view, None);
            self.device.free_memory(texture.memory, None);
            self.device.destroy_image(texture.image, None);
        }
        self.data.level_textures.clear();

        // Destroy old vertex/index buffers
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);

        // Destroy old texture resources
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
        self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);

        // Clear old data
        self.data.vertices.clear();
        self.data.indices.clear();
        self.data.draw_groups.clear();

        // Load new world
        load_dat_model(&mut self.data, world_path, 0, 0.01)?;

        // Install mesh-backed height provider from loaded model
        if !self.data.vertices.is_empty() && !self.data.indices.is_empty() {
            let positions = self.data.vertices.iter().map(|v| v.pos).collect::<Vec<_>>();
            let indices = self.data.indices.clone();
            let mesh = collision::MeshHeightProvider::new(positions.clone(), indices.clone());
            self.height_provider = Box::new(mesh.clone());
            self.mesh_provider = Some(mesh);
        } else {
            self.height_provider = Box::new(collision::FlatGround);
            self.mesh_provider = None;
        }

        // Recreate texture resources
        create_texture_image(&self.instance, &self.device, &mut self.data)?;
        create_texture_image_view(&self.device, &mut self.data)?;
        create_texture_sampler(&self.device, &mut self.data)?;

        // Recreate buffers
        create_vertex_buffer(&self.instance, &self.device, &mut self.data)?;
        create_index_buffer(&self.instance, &self.device, &mut self.data)?;

        // Recreate descriptor sets (they reference the new textures)
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;

        // Reset camera to default position
        self.camera = Camera::default();
        
        // Update current world
        self.current_world = world_path.to_string();

        info!("World loaded successfully: {}", world_path);
        Ok(())
    }

    /// Updates the uniform buffer object for our Vulkan app.
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // Use camera view matrix; raise eye when in player mode (walk) and not free-cam
        let eye_offset = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
            self.eye_offset_walk
        } else { 0.0 };
        let view = self.camera.view_matrix_with_offset(eye_offset);

        #[rustfmt::skip]
        let correction = Mat4::new(
            1.0,  0.0,       0.0, 0.0,
            0.0, -1.0,       0.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );

        let fov_deg = if self.player_mode == collision::PlayerMode::Walk && !self.is_free_cam {
            self.player_fov
        } else { 45.0 };

        let proj = correction
            * cgmath::perspective(
                Deg(fov_deg),
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                0.01,   // Near plane
                1000.0, // Far plane
            );

        let ubo = UniformBufferObject { view, proj };

        // Copy

        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    /// Recreates the swapchain for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, window: &Window, egui_renderer: &mut egui_renderer::EguiRenderer) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        
        // Resize egui renderer framebuffers
        egui_renderer.resize(
            &self.device,
            &self.data.swapchain_image_views,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
        )?;
        
        Ok(())
    }

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

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

    /// Destroys the parts of our Vulkan app related to the swapchain.
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

/// A loaded texture for level geometry
#[derive(Clone, Debug, Default)]
struct LevelTexture {
    name: String,
    width: u32,   // Texture width for UV scaling
    height: u32,  // Texture height for UV scaling
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    // Descriptor set for this texture (one per swapchain image)
    descriptor_sets: Vec<vk::DescriptorSet>,
}

/// A group of indices that share the same texture
#[derive(Clone, Debug, Default)]
struct DrawGroup {
    texture_index: usize,  // Index into level_textures
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    msaa_samples: vk::SampleCountFlags,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_image_count: u32,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    command_pool: vk::CommandPool,
    // Color
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Texture (fallback/default)
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    // Multiple textures for level geometry
    level_textures: Vec<LevelTexture>,
    // Draw groups - each group uses a specific texture
    draw_groups: Vec<DrawGroup>,
    // Model
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    // Debug line rendering
    debug_line_vertex_buffer: vk::Buffer,
    debug_line_vertex_buffer_memory: vk::DeviceMemory,
    debug_line_vertex_capacity: usize,
    debug_line_pipeline: vk::Pipeline,
    debug_line_pipeline_layout: vk::PipelineLayout,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

//
// Instance
//

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial (Rust)\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

//
// Physical Device
//

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions.")))
    }
}

unsafe fn get_max_msaa_samples(instance: &Instance, data: &AppData) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(data.physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts & properties.limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}

//
// Logical Device
//

unsafe fn create_logical_device(entry: &Entry, instance: &Instance, data: &mut AppData) -> Result<Device> {
    // Queue Create Infos

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Layers

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Extensions

    let mut extensions = DEVICE_EXTENSIONS.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    // Features

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .sample_rate_shading(true);

    // Create

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Queues

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

//
// Swapchain
//

unsafe fn create_swapchain(window: &Window, instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Image

    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    // Create

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;

    // Images

    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_image_count = data.swapchain_images.len() as u32;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

#[rustfmt::skip]
fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| create_image_view(device, *i, data.swapchain_format, vk::ImageAspectFlags::COLOR, 1))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

//
// Pipeline
//

unsafe fn create_render_pass(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, data)?)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // Subpasses

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_stencil_attachment_ref)
        .resolve_attachments(resolve_attachments);

    // Dependencies

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    // Create

    let attachments = &[color_attachment, depth_stencil_attachment, color_resolve_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[ubo_binding, sampler_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // Stages

    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    // Vertex Input State

    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Input Assembly State

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Viewport State

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Rasterization State

    // Disable back-face culling to see all geometry regardless of winding order
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)  // Disabled for DAT level geometry
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // Multisample State

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(true)
        .min_sample_shading(0.2)
        .rasterization_samples(data.msaa_samples);

    // Depth Stencil State

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Color Blend State

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Push Constant Ranges

    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64);

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(64)
        .size(4);

    // Layout

    let set_layouts = &[data.descriptor_set_layout];
    let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Create

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    // Cleanup

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());
    Ok(device.create_shader_module(&info, None)?)
}

//
// Framebuffers
//

unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[data.color_image_view, data.depth_image_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

//
// Command Pool
//

unsafe fn create_command_pools(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Global

    data.command_pool = create_command_pool(instance, device, data)?;

    // Per-framebuffer

    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &mut AppData) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

//
// Color Objects
//

unsafe fn create_color_objects(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Image + Image Memory

    let (color_image, color_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        data.msaa_samples,
        data.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;

    // Image View

    data.color_image_view = create_image_view(
        device,
        data.color_image,
        data.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    Ok(())
}

//
// Depth Objects
//

unsafe fn create_depth_objects(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Image + Image Memory

    let format = get_depth_format(instance, data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        data.msaa_samples,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;

    // Image View

    data.depth_image_view = create_image_view(device, data.depth_image, format, vk::ImageAspectFlags::DEPTH, 1)?;

    Ok(())
}

unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

unsafe fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties = instance.get_physical_device_format_properties(data.physical_device, *f);
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

//
// Texture
//

/// Structure to hold texture data before uploading to GPU
struct LoadedTexture {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    name: String,
}

/// Load a DTX texture file
fn load_dtx_texture(path: &std::path::Path) -> Result<LoadedTexture> {
    use crate::dtx::DtxFile;
    
    let dtx = DtxFile::read_from_file(path)
        .map_err(|e| anyhow!("Failed to load DTX file {:?}: {}", path, e))?;
    
    Ok(LoadedTexture {
        pixels: dtx.pixels,
        width: dtx.width as u32,
        height: dtx.height as u32,
        name: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
    })
}

/// Create a solid color texture as fallback
fn create_colored_texture(width: u32, height: u32, r: u8, g: u8, b: u8) -> LoadedTexture {
    let pixel_count = (width * height) as usize;
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for _ in 0..pixel_count {
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
        pixels.push(255);
    }
    LoadedTexture {
        pixels,
        width,
        height,
        name: "fallback".to_string(),
    }
}

/// Search for a DTX file by name in the textures folder
fn find_texture_file(textures_root: &std::path::Path, texture_name: &str) -> Option<std::path::PathBuf> {
    // Clean up texture name - remove path components and convert to uppercase
    let clean_name = texture_name
        .replace(['\\', '/'], "")
        .to_uppercase();
    
    // Add .DTX extension if not present
    let dtx_name = if clean_name.ends_with(".DTX") {
        clean_name
    } else {
        format!("{}.DTX", clean_name)
    };
    
    // Search recursively in all subdirectories
    fn search_recursive(dir: &std::path::Path, target: &str) -> Option<std::path::PathBuf> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(found) = search_recursive(&path, target) {
                        return Some(found);
                    }
                } else if let Some(name) = path.file_name() {
                    if name.to_string_lossy().to_uppercase() == target {
                        return Some(path);
                    }
                }
            }
        }
        None
    }
    
    search_recursive(textures_root, &dtx_name)
}

/// Get texture dimensions without loading full pixel data
fn get_texture_dimensions(texture_name: &str) -> (u32, u32) {
    let textures_path = std::path::Path::new("REZ/TEXTURES");
    
    // Try to find and load the DTX file header
    let variations = [
        texture_name.to_string(),
        texture_name.replace("TEXTURES\\", ""),
        texture_name.replace("textures\\", ""),
        texture_name.split('\\').last().unwrap_or(texture_name).to_string(),
        texture_name.split('/').last().unwrap_or(texture_name).to_string(),
    ];
    
    for var in &variations {
        if let Some(dtx_path) = find_texture_file(textures_path, var) {
            if let Ok(dtx) = crate::dtx::DtxFile::read_from_file(&dtx_path) {
                return (dtx.width as u32, dtx.height as u32);
            }
        }
    }
    
    // Default fallback dimensions
    (256, 256)
}

unsafe fn create_texture_image(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let textures_path = std::path::Path::new("REZ/TEXTURES");
    
    println!("=== LOADING LEVEL TEXTURES ===");
    println!("Textures to load: {}", data.level_textures.len());
    
    // Create a fallback white texture first (for the default single texture)
    let fallback = create_colored_texture(64, 64, 128, 128, 128);
    
    // Load textures for each level texture entry
    let mut loaded_count = 0;
    let mut failed_count = 0;
    
    for i in 0..data.level_textures.len() {
        let texture_name = data.level_textures[i].name.clone();
        
        // Try to find and load the DTX file
        let loaded = if let Some(dtx_path) = find_texture_file(textures_path, &texture_name) {
            match load_dtx_texture(&dtx_path) {
                Ok(tex) => {
                    loaded_count += 1;
                    tex
                }
                Err(e) => {
                    println!("  Failed to load {}: {}", texture_name, e);
                    failed_count += 1;
                    create_colored_texture(64, 64, 255, 0, 255) // Magenta for error
                }
            }
        } else {
            // Try some common path variations
            let variations = [
                texture_name.clone(),
                texture_name.replace("TEXTURES\\", ""),
                texture_name.replace("textures\\", ""),
                texture_name.split('\\').last().unwrap_or(&texture_name).to_string(),
                texture_name.split('/').last().unwrap_or(&texture_name).to_string(),
            ];
            
            let mut found_tex = None;
            for var in &variations {
                if let Some(dtx_path) = find_texture_file(textures_path, var) {
                    if let Ok(tex) = load_dtx_texture(&dtx_path) {
                        found_tex = Some(tex);
                        break;
                    }
                }
            }
            
            if let Some(tex) = found_tex {
                loaded_count += 1;
                tex
            } else {
                failed_count += 1;
                // Use a unique color for missing textures based on hash
                let hash = texture_name.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                let r = ((hash * 17) % 200 + 55) as u8;
                let g = ((hash * 31) % 200 + 55) as u8;  
                let b = ((hash * 47) % 200 + 55) as u8;
                create_colored_texture(64, 64, r, g, b)
            }
        };
        
        // Upload this texture to GPU
        let (image, memory) = upload_texture_to_gpu(instance, device, data, &loaded)?;
        
        // Create image view
        let view = create_image_view(device, image, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR, 1)?;
        
        // Update the level texture entry
        data.level_textures[i].image = image;
        data.level_textures[i].memory = memory;
        data.level_textures[i].view = view;
    }
    
    println!("Loaded: {} textures, Failed: {} textures", loaded_count, failed_count);
    
    // Also create the default/fallback texture (for backwards compatibility)
    let (image, memory) = upload_texture_to_gpu(instance, device, data, &fallback)?;
    data.texture_image = image;
    data.texture_image_memory = memory;
    data.mip_levels = 1;

    Ok(())
}

/// Upload a texture to GPU memory (without mipmaps for simplicity)
unsafe fn upload_texture_to_gpu(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    texture: &LoadedTexture,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let width = texture.width;
    let height = texture.height;
    let size = (width * height * 4) as u64;

    // Create staging buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy to staging
    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(texture.pixels.as_ptr(), memory.cast(), texture.pixels.len());
    device.unmap_memory(staging_buffer_memory);

    // Create image (no mipmaps for level textures - simpler)
    let (image, image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        1,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Transition to transfer dest
    transition_image_layout(
        device,
        data,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        1,
    )?;

    // Copy buffer to image
    copy_buffer_to_image(device, data, staging_buffer, image, width, height)?;

    // Transition to shader read
    transition_image_layout(
        device,
        data,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        1,
    )?;

    // Cleanup staging
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((image, image_memory))
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // Support

    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!("Texture image format does not support linear blitting!"));
    }

    // Mipmaps

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    // Create view for fallback texture
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        1, // No mipmaps for level textures
    )?;

    Ok(())
}

unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(data.mip_levels as f32)
        .mip_lod_bias(0.0);

    data.texture_sampler = device.create_sampler(&info, None)?;

    Ok(())
}

//
// Model
//

fn load_model(data: &mut AppData) -> Result<()> {
    // Model

    let mut reader = BufReader::new(File::open("tutorial/resources/viking_room.obj")?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    // Vertices / Indices

    let mut unique_vertices = HashMap::new();

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = Vertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    Ok(())
}

/// Load a KISS Psycho Circus DAT file (v127) and extract mesh data
/// 
/// # Arguments
/// * `path` - Path to the .dat file
/// * `world_model_index` - Which world model to load (0 = main world)
/// * `scale` - Scale factor for the world geometry (Lithtech units are large)
/// 
/// # Returns
/// Tuple of (vertices, indices) ready for Vulkan rendering
fn load_dat_model<P: AsRef<std::path::Path>>(
    data: &mut AppData,
    path: P,
    world_model_index: usize,
    scale: f32,
) -> Result<()> {
    use crate::dat::DatFile;
    use crate::dat_mesh::MeshExtractor;

    info!("Loading DAT file: {}", path.as_ref().display());

    // Parse the DAT file
    let dat_file = DatFile::read_from_file(&path)
        .map_err(|e| anyhow!("Failed to parse DAT file: {}", e))?;

    info!(
        "DAT file loaded: {} objects, {} world models",
        dat_file.objects.len(),
        dat_file.world_models.len()
    );

    // Print info about the first world model for debugging
    if let Some(world) = dat_file.world_models.get(world_model_index) {
        println!("=== WORLD MODEL DEBUG ===");
        println!("  World name: {}", world.world_name);
        println!("  Points: {}", world.points.len());
        println!("  Polygons: {}", world.polygons.len());
        println!("  Surfaces: {}", world.surfaces.len());
        println!("  Planes: {}", world.planes.len());
        
        // Check a few polygon vertex counts
        for (i, poly) in world.polygons.iter().take(5).enumerate() {
            println!("  Poly[{}]: {} verts, surface_idx={}", i, poly.disk_verts.len(), poly.surface_index);
        }
    }

    // Extract mesh data - try without skip_invisible to see all geometry
    let extractor = MeshExtractor::new(&dat_file)
        .with_scale(scale)
        .with_skip_invisible(false)  // Show all polygons for debugging
        .with_skip_sky(false)
        .with_flip_winding(false);  // Try without flip - Lithtech might use same winding

    let mesh = extractor
        .extract_world_by_index(world_model_index)
        .ok_or_else(|| anyhow!("World model index {} not found", world_model_index))?;

    info!(
        "Extracted mesh '{}': {} vertices, {} indices, {} texture groups",
        mesh.name,
        mesh.vertices.len(),
        mesh.indices.len(),
        mesh.textured_meshes.len()
    );

    // Build draw groups from textured meshes
    // Each textured mesh becomes a draw group with its own texture
    let mut texture_name_to_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut texture_names: Vec<String> = Vec::new();
    let mut texture_dimensions: std::collections::HashMap<String, (u32, u32)> = std::collections::HashMap::new();
    
    let mut current_vertex_offset = 0u32;
    let mut current_index_offset = 0u32;
    
    for textured_mesh in &mesh.textured_meshes {
        // Get or create texture index
        let texture_index = if let Some(&idx) = texture_name_to_index.get(&textured_mesh.texture_name) {
            idx
        } else {
            let idx = texture_names.len();
            texture_names.push(textured_mesh.texture_name.clone());
            texture_name_to_index.insert(textured_mesh.texture_name.clone(), idx);
            idx
        };
        
        // Get texture dimensions for UV scaling
        let (tex_width, tex_height) = if let Some(&dims) = texture_dimensions.get(&textured_mesh.texture_name) {
            dims
        } else {
            let dims = get_texture_dimensions(&textured_mesh.texture_name);
            texture_dimensions.insert(textured_mesh.texture_name.clone(), dims);
            dims
        };
        
        // Add vertices for this group, scaling UVs by texture dimensions
        for dat_vert in &textured_mesh.vertices {
            let vertex = Vertex {
                pos: vec3(dat_vert.pos[0], dat_vert.pos[1], dat_vert.pos[2]),
                color: vec3(dat_vert.color[0], dat_vert.color[1], dat_vert.color[2]),
                // Scale UVs by texture dimensions (Lithtech UV formula)
                tex_coord: vec2(
                    dat_vert.tex_coord[0] / tex_width as f32,
                    dat_vert.tex_coord[1] / tex_height as f32,
                ),
            };
            data.vertices.push(vertex);
        }
        
        // Add indices for this group - offset by VERTEX count, not index count
        // Each textured_mesh has indices relative to its own vertices (0 to vertices.len()-1)
        for &idx in &textured_mesh.indices {
            data.indices.push(current_vertex_offset + idx);
        }
        
        // Create draw group
        data.draw_groups.push(DrawGroup {
            texture_index,
            first_index: current_index_offset,
            index_count: textured_mesh.indices.len() as u32,
            vertex_offset: 0, // Using absolute indices, so no vertex offset needed
        });
        
        current_vertex_offset += textured_mesh.vertices.len() as u32;
        current_index_offset += textured_mesh.indices.len() as u32;
    }
    
    println!("=== TEXTURE GROUPS ===");
    println!("  Total texture groups: {}", data.draw_groups.len());
    println!("  Unique textures: {}", texture_names.len());
    for (i, name) in texture_names.iter().enumerate() {
        println!("    [{}] {}", i, name);
    }
    
    // Store texture names and dimensions for later loading
    for name in &texture_names {
        let (width, height) = texture_dimensions.get(name).copied().unwrap_or((256, 256));
        data.level_textures.push(LevelTexture {
            name: name.clone(),
            width,
            height,
            ..Default::default()
        });
    }

    // Calculate and print bounds
    let mut min_pos = [f32::MAX, f32::MAX, f32::MAX];
    let mut max_pos = [f32::MIN, f32::MIN, f32::MIN];
    for v in &data.vertices {
        min_pos[0] = min_pos[0].min(v.pos.x);
        min_pos[1] = min_pos[1].min(v.pos.y);
        min_pos[2] = min_pos[2].min(v.pos.z);
        max_pos[0] = max_pos[0].max(v.pos.x);
        max_pos[1] = max_pos[1].max(v.pos.y);
        max_pos[2] = max_pos[2].max(v.pos.z);
    }

    // Print bounds for debugging camera position
    println!("=== MESH BOUNDS (after scale {}) ===", scale);
    println!("  Min: ({:.2}, {:.2}, {:.2})", min_pos[0], min_pos[1], min_pos[2]);
    println!("  Max: ({:.2}, {:.2}, {:.2})", max_pos[0], max_pos[1], max_pos[2]);
    let center = [
        (min_pos[0] + max_pos[0]) / 2.0,
        (min_pos[1] + max_pos[1]) / 2.0,
        (min_pos[2] + max_pos[2]) / 2.0,
    ];
    let size = [
        max_pos[0] - min_pos[0],
        max_pos[1] - min_pos[1],
        max_pos[2] - min_pos[2],
    ];
    println!("  Center: ({:.2}, {:.2}, {:.2})", center[0], center[1], center[2]);
    println!("  Size: ({:.2}, {:.2}, {:.2})", size[0], size[1], size[2]);

    info!(
        "Loaded {} vertices, {} indices, {} draw groups from DAT file",
        data.vertices.len(),
        data.indices.len(),
        data.draw_groups.len()
    );

    Ok(())
}

/// Print summary information about a DAT file without loading mesh data
fn print_dat_info<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
    use crate::dat::DatFile;
    use crate::dat_mesh::{MeshExtractor, MeshStats};

    info!("Analyzing DAT file: {}", path.as_ref().display());

    let dat_file = DatFile::read_from_file(&path)
        .map_err(|e| anyhow!("Failed to parse DAT file: {}", e))?;

    println!("\n{}", "=".repeat(60));
    println!("DAT FILE SUMMARY");
    println!("{}", "=".repeat(60));

    println!("\nVersion: {} (KISS Psycho Circus)", dat_file.header.version);
    println!("World Properties: {}", dat_file.world_info.properties);
    println!(
        "Lightmap Grid Size: {}",
        dat_file.world_info.light_map_grid_size
    );

    println!("\n--- WORLD OBJECTS ({}) ---", dat_file.objects.len());
    let mut object_types: HashMap<&str, usize> = HashMap::new();
    for obj in &dat_file.objects {
        *object_types.entry(&obj.type_name).or_insert(0) += 1;
    }
    for (type_name, count) in &object_types {
        println!("  {}: {}", type_name, count);
    }

    println!("\n--- WORLD MODELS ({}) ---", dat_file.world_models.len());
    for (i, model) in dat_file.world_models.iter().enumerate() {
        println!("\n  [{}] {}", i, model.world_name);
        println!("      Points: {}", model.point_count);
        println!("      Polygons: {}", model.poly_count);
        println!("      Surfaces: {}", model.surface_count);
        println!("      Textures: {}", model.texture_count);
        println!(
            "      Bounds: ({:.1}, {:.1}, {:.1}) to ({:.1}, {:.1}, {:.1})",
            model.min_box.x,
            model.min_box.y,
            model.min_box.z,
            model.max_box.x,
            model.max_box.y,
            model.max_box.z
        );
    }

    // Extract and show mesh statistics
    let extractor = MeshExtractor::new(&dat_file);
    let meshes = extractor.extract_all_worlds();
    let stats = MeshStats::from_meshes(&meshes);

    println!("\n--- MESH STATISTICS ---");
    println!("  Total Vertices: {}", stats.total_vertices);
    println!("  Total Indices: {}", stats.total_indices);
    println!("  Total Triangles: {}", stats.total_triangles);
    println!("  Total Texture Groups: {}", stats.texture_count);

    println!("\n{}", "=".repeat(60));

    Ok(())
}

//
// Buffers
//

unsafe fn create_vertex_buffer(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Create (staging)

    let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(data.vertices.as_ptr(), memory.cast(), data.vertices.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (vertex)

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    // Copy (vertex)

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_index_buffer(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Create (staging)

    let size = (size_of::<u32>() * data.indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());

    device.unmap_memory(staging_buffer_memory);

    // Create (index)

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    // Copy (index)

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    // Cleanup

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_uniform_buffers(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

//
// Descriptors
//

unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // We need descriptor sets for:
    // - Main descriptor sets (one per swapchain image)
    // - Per-texture descriptor sets (one per texture per swapchain image)
    let num_textures = data.level_textures.len().max(1);
    let total_sets = data.swapchain_images.len() * (1 + num_textures);
    
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(total_sets as u32);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(total_sets as u32);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(total_sets as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let swapchain_count = data.swapchain_images.len();
    
    // Allocate main descriptor sets (for UBO + fallback texture)
    let layouts = vec![data.descriptor_set_layout; swapchain_count];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    // Update main descriptor sets with UBO and fallback texture
    for i in 0..swapchain_count {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
    }

    // Allocate and update descriptor sets for each level texture
    for tex_idx in 0..data.level_textures.len() {
        // Allocate descriptor sets for this texture (one per swapchain image)
        let layouts = vec![data.descriptor_set_layout; swapchain_count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(data.descriptor_pool)
            .set_layouts(&layouts);

        let tex_descriptor_sets = device.allocate_descriptor_sets(&info)?;
        
        // Update each descriptor set
        for i in 0..swapchain_count {
            let buffer_info_data = vk::DescriptorBufferInfo::builder()
                .buffer(data.uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);

            let buffer_info = &[buffer_info_data];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(tex_descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            let image_info_data = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(data.level_textures[tex_idx].view)
                .sampler(data.texture_sampler);

            let image_info = &[image_info_data];
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(tex_descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
        }
        
        // Store the descriptor sets
        data.level_textures[tex_idx].descriptor_sets = tex_descriptor_sets;
    }

    Ok(())
}

//
// Command Buffers
//

unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let num_images = data.swapchain_images.len();
    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    Ok(())
}

//
// Sync Objects
//

unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    // Create semaphores per swapchain image to avoid reuse conflicts
    for _ in 0..data.swapchain_image_count {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();

    Ok(())
}

//
// Structs
//

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(instance: &Instance, data: &AppData, physical_device: vk::PhysicalDevice) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(physical_device, index as u32, data.surface)? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(instance: &Instance, data: &AppData, physical_device: vk::PhysicalDevice) -> Result<Self> {
        Ok(Self {
            capabilities: instance.get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance.get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    view: Mat4,
    proj: Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self { pos, color, tex_coord }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();
        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

//
// Shared (Buffers)
//

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // Buffer

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    // Memory

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(instance, data, properties, requirements)?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &mut AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

//
// Shared (Images)
//

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    // Image

    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples);

    let image = device.create_image(&info, None)?;

    // Memory

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(instance, data, properties, requirements)?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &mut AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => return Err(anyhow!("Unsupported image layout transition!")),
    };

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &mut AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

//
// Shared (Other)
//

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

unsafe fn begin_single_time_commands(device: &Device, data: &AppData) -> Result<vk::CommandBuffer> {
    // Allocate

    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    // Begin

    let info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_commands(device: &Device, data: &AppData, command_buffer: vk::CommandBuffer) -> Result<()> {
    // End

    device.end_command_buffer(command_buffer)?;

    // Submit

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    // Cleanup

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}