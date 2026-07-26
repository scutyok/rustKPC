use crate::app::App;
use log::*;
use rustKPC::egui_renderer;
use rustKPC::pcx;
use rustKPC::world_chooser::{LoadingState, WorldChooser};
use winit::window::Window;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExitType {
    EndOfWorld,
    EndOfEpisode,
    EndOfSubWorld,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectivesState {
    pub resource_id: u32,
    pub title: String,
    pub text: String,
    pub is_active: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LevelTransition {
    pub next_world: Option<String>,
    pub start_point: String,
    pub exit_type: ExitType,
}

#[derive(Debug, Default)]
pub struct GameState {
    pub current_objective: Option<ObjectivesState>,
    pub pending_level_transition: Option<LevelTransition>, // ExitTrigger writes here
}

impl LevelTransition {
    pub fn next_world(&self) -> Option<&str> {
        self.next_world.as_deref()
    }

    pub fn load_world_with_loading_screen(
        app: &mut App,
        window: &Window,
        egui_ctx: &egui::Context,
        egui_state: &mut egui_winit::State,
        egui_renderer: &mut egui_renderer::EguiRenderer,
        world_path: &str,
        mouse_locked: &mut bool,
    ) {
        let map_name = WorldChooser::get_world_display_name(world_path);
        app.loading_state = LoadingState::Loading(map_name.clone());
        let new_title = format!("Loading: {}...", map_name);
        window.set_title(&new_title);

        {
            let stem = std::path::Path::new(world_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_uppercase();
            let level_key = stem.trim_end_matches(|c: char| c.is_ascii_alphabetic());
            let pcx_path = format!("REZ/SCREENS/LOADINGBACKGROUNDS/{}.PCX", level_key);
            if let Ok(img) = pcx::load_pcx(std::path::Path::new(&pcx_path)) {
                match unsafe {
                    egui_renderer.set_user_texture(
                        &app.instance,
                        &app.device,
                        app.data.physical_device,
                        app.data.command_pool,
                        app.data.graphics_queue,
                        &img.pixels,
                        img.width,
                        img.height,
                    )
                } {
                    Ok(tex_id) => { app.loading_texture_id = Some(tex_id); }
                    Err(e) => { warn!("Failed to set loading texture: {}", e); }
                }
            }
        }

        let raw_input = egui_state.take_egui_input(window);
        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.run_ui(ctx, mouse_locked);
        });
        egui_state.handle_platform_output(window, full_output.platform_output);
        let clipped_primitives = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        unsafe { app.render(window, egui_renderer, &clipped_primitives, full_output.pixels_per_point) }.unwrap();

        if let Err(e) = unsafe { app.reload_world(world_path, egui_renderer) } {
            error!("Failed to load world {}: {}", world_path, e);
        }

        app.loading_state = LoadingState::Ready;
        app.loading_texture_id = None;
        unsafe { egui_renderer.clear_user_texture(&app.device); }
        window.set_title("KISS Psycho Circus: The Nightmare Child");
    }
}