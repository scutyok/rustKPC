//******************************************************************/
//
// SCR script parser and runner for KISS Psycho Circus scripted sequences.
//
// SCR files contain timed commands like:
//   2.8 trigger_door door01 open
//   3.2 trigger_generic Doorless01
//
// This module parses the commands we can act on and runs them at the right
// time offsets when the sequence is triggered.
//
//******************************************************************/


use cgmath::{Matrix4};

use crate::object_utils::matrix4_to_array;
use crate::types::DrawGroup;
use crate::util::math::*;

//******************************************************************/
//
// Script commands
//
//******************************************************************/


#[derive(Debug, Clone)]
pub enum DoorOp {
    Open,
    Close,
}

#[derive(Debug, Clone)]
pub enum ScriptCommand {
    /// Start another SCR script by its normalized trigger name.
    StartScript { script_name: String },
    /// Operate a door or BSP door by name.
    TriggerDoor { door_name: String, op: DoorOp },
    /// Fire a generic target. This is commonly used for spawners, script runners,
    /// exits, and other named DAT objects.
    TriggerGeneric { target_name: String },
    /// Transition to another world/level using the target path and optional start-point name.
    TransitionLevel { next_world: String, start_point: String },
    /// Spawn an item by object name.
    ItemSpawn { target_name: String },
    /// Enable or disable a named script/object.
    SetObjectEnabled { target_name: String, enabled: bool },
    /// Play a level sound. Playback is currently handled by higher-level audio code.
    PlaySound { path: String, local: bool },
}

#[derive(Debug, Clone)]
pub struct TimedCommand {
    pub time: f32,
    pub command: ScriptCommand,
}

//******************************************************************/
//
// SCR parser
//
//******************************************************************/

fn normalize_script_name(path: &str, file_stem: &str) -> String {
    let script_path = path
        .trim_matches('"')
        .replace('\\', "/")
        .to_lowercase();
    let filename = script_path.rsplit('/').next().unwrap_or(&script_path);
    let stem = filename.strip_suffix(".scr").unwrap_or(filename);
    let prefix = format!("{}_", file_stem.to_lowercase());
    if stem.starts_with(&prefix) {
        stem[prefix.len()..].to_string()
    } else {
        stem.to_string()
    }
}

/// Parse a command body after its leading time value.
pub fn parse_command_parts(parts: &[&str], file_stem: &str) -> Option<ScriptCommand> {
    if parts.is_empty() {
        return None;
    }
    match parts[0].to_ascii_lowercase().as_str() {
        "scriptplay" | "cheatscript" | "endingscript" if parts.len() >= 2 => {
            Some(ScriptCommand::StartScript {
                script_name: normalize_script_name(parts[1], file_stem),
            })
        }
        "script_object_enable" if parts.len() >= 2 => Some(ScriptCommand::SetObjectEnabled {
            target_name: parts[1].to_lowercase(),
            enabled: true,
        }),
        "script_object_disable" | "script_object_abort" if parts.len() >= 2 => {
            Some(ScriptCommand::SetObjectEnabled {
                target_name: parts[1].to_lowercase(),
                enabled: false,
            })
        }
        "trigger_door" | "trigger_sliding_door" if parts.len() >= 3 => {
            let op = match parts[2].to_ascii_lowercase().as_str() {
                "close" => DoorOp::Close,
                _ => DoorOp::Open,
            };
            Some(ScriptCommand::TriggerDoor {
                door_name: parts[1].to_lowercase(),
                op,
            })
        }
        "trigger_generic" | "trigger_volume" if parts.len() >= 2 => {
            Some(ScriptCommand::TriggerGeneric {
                target_name: parts[1].to_lowercase(),
            })
        }
        "item_spawn" if parts.len() >= 2 => Some(ScriptCommand::ItemSpawn {
            target_name: parts[1].to_lowercase(),
        }),
        "streamsound" | "sound" | "playsound" if parts.len() >= 2 => {
            let path_index = if parts.len() >= 3 && parts[1].eq_ignore_ascii_case("player") {
                2
            } else {
                1
            };
            Some(ScriptCommand::PlaySound {
                path: parts[path_index].trim_matches('"').to_string(),
                local: parts.iter().any(|p| p.eq_ignore_ascii_case("local")),
            })
        }
        _ => None,
    }
}

/// Parse a full timed command line such as `0.1 trigger_generic boxboy`.
pub fn parse_timed_command_line(line: &str, file_stem: &str) -> Option<TimedCommand> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("//") {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let time: f32 = parts[0].parse().ok()?;
    let command = parse_command_parts(&parts[1..], file_stem)?;
    Some(TimedCommand { time, command })
}

/// Parse a DAT command property such as `0 script_object_play Foo`.
pub fn parse_dat_command(
    command: &str,
    file_stem: &str,
    script_object_targets: &std::collections::HashMap<String, Vec<String>>,
) -> Vec<ScriptCommand> {
    let mut out = Vec::new();
    let line = command.trim();
    if line.is_empty() || line.starts_with("//") {
        return out;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    let body = if parts.first().and_then(|p| p.parse::<f32>().ok()).is_some() {
        &parts[1..]
    } else {
        &parts[..]
    };
    if body.is_empty() {
        return out;
    }

    if body[0].eq_ignore_ascii_case("script_object_play") && body.len() >= 2 {
        let target = body[1].to_lowercase();
        if let Some(script_names) = script_object_targets.get(&target) {
            out.extend(script_names.iter().cloned().map(|script_name| {
                ScriptCommand::StartScript { script_name }
            }));
        } else {
            out.push(ScriptCommand::TriggerGeneric { target_name: target });
        }
        return out;
    }

    if let Some(command) = parse_command_parts(body, file_stem) {
        out.push(command);
    }
    out
}

// Parse an SCR script file into a list of timed commands.
// Only commands we can act on are extracted; the rest are ignored.
pub fn parse_scr_with_prefix(contents: &str, file_stem: &str) -> Vec<TimedCommand> {
    let mut commands = Vec::new();
    for line in contents.lines() {
        if let Some(command) = parse_timed_command_line(line, file_stem) {
            commands.push(command);
        }
    }
    commands.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    commands
}

pub fn parse_scr(contents: &str) -> Vec<TimedCommand> {
    parse_scr_with_prefix(contents, "")
}

//******************************************************************/
//
// BSP Door
//
//******************************************************************/


/// Slide speed in Vulkan units per second.
const BSP_DOOR_SPEED: f32 = 3.0;

#[derive(Debug, Clone, PartialEq)]
pub enum BspDoorState {
    Closed,
    Opening,
    Open,
}

//******************************************************************/
//
// A BSP sub-model door that slides upward when opened.
//
//******************************************************************/

#[derive(Debug, Clone)]
pub struct BspDoor {
    /// Lowercase name matching the BSP world model name (e.g. "door01").
    pub name: String,
    /// Draw group indices for this BSP sub-model.
    pub draw_groups: Vec<usize>,
    /// Slide distance in Vulkan units (computed from mesh height).
    pub slide_distance: f32,
    /// Normalized Vulkan-space slide direction.
    pub slide_dir: [f32; 3],
    /// Current slide offset [0..slide_distance].
    pub slide_offset: f32,
    pub state: BspDoorState,
    /// Range of collision vertex indices [start..end) that belong to this door.
    pub collision_vertex_range: Option<(usize, usize)>,
    /// Original (closed) positions of the collision vertices, so we can re-derive the slid positions.
    pub collision_base_positions: Vec<Vector3>,
}

impl BspDoor {
    pub fn open(&mut self) {
        if self.state == BspDoorState::Closed {
            self.state = BspDoorState::Opening;
        }
    }

    pub fn update(&mut self, dt: f32, draw_groups: &mut Vec<DrawGroup>) {
        if self.state == BspDoorState::Opening {
            self.slide_offset += dt * BSP_DOOR_SPEED;
            if self.slide_offset >= self.slide_distance {
                self.slide_offset = self.slide_distance;
                self.state = BspDoorState::Open;
            }
            let m = matrix4_to_array(
                Matrix4::from_translation(Vector3::new(
                    self.slide_dir[0] * self.slide_offset,
                    self.slide_dir[1] * self.slide_offset,
                    self.slide_dir[2] * self.slide_offset,
                ).into()),
            );
            for &dg in &self.draw_groups {
                if dg < draw_groups.len() {
                    draw_groups[dg].model_matrix = Some(m);
                }
            }
        }
    }

    /// Update the collision mesh positions to follow the door's slide offset.
    pub fn update_collision(&self, collision_positions: &mut [Vector3]) {
        if let Some((start, end)) = self.collision_vertex_range {
            if self.slide_offset > 0.0 && end <= collision_positions.len() {
                for (i, base_pos) in self.collision_base_positions.iter().enumerate() {
                    collision_positions[start + i] = Vector3::new(
                        base_pos.x + self.slide_dir[0] * self.slide_offset,
                        base_pos.y + self.slide_dir[1] * self.slide_offset,
                        base_pos.z + self.slide_dir[2] * self.slide_offset,
                    );
                }
            }
        }
    }
}

//******************************************************************/
//
// Script Runner
//
//******************************************************************/


#[derive(Debug, Clone)]
pub struct ScriptRunner {
    pub commands: Vec<TimedCommand>,
    pub elapsed: f32,
    pub next_index: usize,
    pub running: bool,
}

impl ScriptRunner {
    pub fn new(commands: Vec<TimedCommand>) -> Self {
        Self {
            commands,
            elapsed: 0.0,
            next_index: 0,
            running: false,
        }
    }

    pub fn start(&mut self) {
        if !self.running {
            self.running = true;
            self.elapsed = 0.0;
            self.next_index = 0;
        }
    }

    /// Advance the script clock and return any commands whose time has been reached.
    pub fn update(&mut self, dt: f32) -> Vec<ScriptCommand> {
        if !self.running {
            return Vec::new();
        }
        self.elapsed += dt;
        let mut fired = Vec::new();
        while self.next_index < self.commands.len()
            && self.commands[self.next_index].time <= self.elapsed
        {
            fired.push(self.commands[self.next_index].command.clone());
            self.next_index += 1;
        }
        if self.next_index >= self.commands.len() {
            self.running = false;
        }
        fired
    }
}
