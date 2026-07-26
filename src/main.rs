#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

mod app;
mod settings;
#[path = "kiss_engine/Mechanics/state/KGameState.rs"]
mod k_game_state;

use std::time::Instant;

use anyhow::Result;
use log::*;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowBuilder};

use vulkanalia::prelude::v1_0::*;

use rustKPC::collision;
use rustKPC::egui_renderer;
use rustKPC::pcx;
use rustKPC::types::*;
use rustKPC::world_chooser::{LoadingState, WorldChooser};

use crate::k_game_state::LevelTransition;

use crate::app::App;

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

    // Load saved settings and apply
    match settings::Settings::load() {
        Ok(s) => {
            app.dynamic_lighting = s.dynamic_lighting;
            if !app.dynamic_lighting {
                app.cached_light_ubo.light_count = 0;
                app.cached_light_ubo.shadow_count = 0;
                unsafe { app.upload_light_ubo_to_all(); }
            }
        }
        Err(e) => {
            warn!("Failed to load settings: {}", e);
        }
    }
    
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
    let mut smooth_fps: f32 = 0.0;
    // Track last window title to avoid updating it every frame
    let mut last_window_title = String::from("KISS Psycho Circus: The Nightmare Child [F1: Debug Menu]");
    // Reused buffer for converting egui font atlas to RGBA bytes
    let mut font_rgba: Vec<u8> = Vec::new();
    
    // Set initial title after load
    window.set_title("KISS Psycho Circus: The Nightmare Child [F1: Debug Menu]");
    
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
                    // Exponential moving average for stable FPS display
                    const SMOOTH: f32 = 0.05;
                    smooth_fps = if smooth_fps < 1.0 { fps } else { smooth_fps + (fps - smooth_fps) * SMOOTH };
                    app.fps = smooth_fps;
                    
                    // Check for pending world load (world chooser or exit-trigger transition)
                    if let Some(world_path) = app.world_chooser.take_pending_load().or_else(|| app.pending_world_load.take()) {
                        LevelTransition::load_world_with_loading_screen(&mut app, &window, &egui_ctx, &mut egui_state, &mut egui_renderer, &world_path, &mut mouse_locked);
                        let new_title = String::from("KISS Psycho Circus: The Nightmare Child [F1: World Select]");
                        if new_title != last_window_title {
                            window.set_title(&new_title);
                            last_window_title = new_title;
                        }
                    }
                    
                    // Run egui UI
                    let raw_input = egui_state.take_egui_input(&window);
                    let full_output = egui_ctx.run(raw_input, |ctx| {
                        app.run_ui(ctx, &mut mouse_locked);
                    });
                    egui_state.handle_platform_output(&window, full_output.platform_output);

                    // Update GPU font texture if egui's atlas changed (full or partial)
                    {
                        let needs_update = full_output.textures_delta.set.iter()
                            .any(|(id, _)| *id == egui::TextureId::default());
                        if needs_update {
                            // Get the FULL font atlas from egui context and convert into RGBA bytes
                            let (image_pixels, font_w, font_h) = egui_ctx.fonts(|fonts| {
                                let image = fonts.image();
                                (image.pixels.clone(), image.width() as u32, image.height() as u32)
                            });
                            font_rgba.clear();
                            font_rgba.reserve((font_w * font_h * 4) as usize);
                            for &a in image_pixels.iter() {
                                let alpha = (a * 255.0).round().clamp(0.0, 255.0) as u8;
                                font_rgba.push(255u8);
                                font_rgba.push(255u8);
                                font_rgba.push(255u8);
                                font_rgba.push(alpha);
                            }
                            unsafe {
                                egui_renderer.replace_font_texture(
                                    &app.instance,
                                    &app.device,
                                    app.data.physical_device,
                                    app.data.command_pool,
                                    app.data.graphics_queue,
                                    &font_rgba,
                                    font_w,
                                    font_h,
                                ).unwrap();
                            }
                        }
                    }

                    // Tessellate egui shapes into primitives for rendering
                    let clipped_primitives = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                    
                    // Update window title based on UI state
                    if app.world_chooser.visible {
                        let selected_name = app.world_chooser.worlds
                            .get(app.world_chooser.selected_index)
                            .map(|p| WorldChooser::get_world_display_name(p))
                            .unwrap_or_default();
                        let new_title = format!(
                            "World Chooser [{}/{}]: {} | Click to select, double-click to load | F1/Esc to close",
                            app.world_chooser.selected_index + 1,
                            app.world_chooser.worlds.len(),
                            selected_name
                        );
                        if new_title != last_window_title {
                            window.set_title(&new_title);
                            last_window_title = new_title;
                        }
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
                        unsafe { app.update_objects(dt, &mut egui_renderer); }
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
                    unsafe { app.device.device_wait_idle().unwrap(); }
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
                            } else {
                                mouse_locked = true;
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
                                // Restore player mode from free cam toggle state
                                app.player_mode = if app.is_free_cam { collision::PlayerMode::Flying } else { collision::PlayerMode::Walk };
                                // Persist settings when closing the world chooser so quick interactions
                                // (e.g. toggling a checkbox then immediately pressing F1) are not lost.
                                if let Err(e) = (crate::settings::Settings { dynamic_lighting: app.dynamic_lighting }).save() {
                                    warn!("Failed to save settings: {}", e);
                                }
                                // If dynamic lighting was just disabled, ensure the GPU UBO reflects that immediately.
                                if !app.dynamic_lighting {
                                    app.cached_light_ubo.light_count = 0;
                                    app.cached_light_ubo.shadow_count = 0;
                                    unsafe { app.upload_light_ubo_to_all(); }
                                }
                            }
                        }
                        PhysicalKey::Code(KeyCode::F2) if pressed => {
                            app.toggle_triggers();
                        }
                        PhysicalKey::Code(KeyCode::KeyW) if !app.world_chooser.visible => app.input.forward = pressed,
                        PhysicalKey::Code(KeyCode::KeyS) if !app.world_chooser.visible => app.input.backward = pressed,
                        PhysicalKey::Code(KeyCode::KeyA) if !app.world_chooser.visible => app.input.left = pressed,
                        PhysicalKey::Code(KeyCode::KeyD) if !app.world_chooser.visible => app.input.right = pressed,
                        PhysicalKey::Code(KeyCode::Space) if !app.world_chooser.visible => app.input.up = pressed,
                        PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) if !app.world_chooser.visible => app.input.down = pressed,
                        PhysicalKey::Code(KeyCode::KeyE) if !app.world_chooser.visible => {
                            if pressed { app.interact(); }
                        }
                        PhysicalKey::Code(KeyCode::Escape) if pressed => {
                            if app.world_chooser.visible {
                                app.world_chooser.visible = false;
                                mouse_locked = true;
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
                                // Persist settings when closing the world chooser via Escape as well
                                if let Err(e) = (crate::settings::Settings { dynamic_lighting: app.dynamic_lighting }).save() {
                                    warn!("Failed to save settings: {}", e);
                                }
                                if !app.dynamic_lighting {
                                    app.cached_light_ubo.light_count = 0;
                                    app.cached_light_ubo.shadow_count = 0;
                                    unsafe { app.upload_light_ubo_to_all(); }
                                }
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
