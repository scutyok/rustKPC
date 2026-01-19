#![allow(non_snake_case)]

// Manually mapping the folder structure to modules
#[path = "kiss_engine/dat/dat.rs"]
pub mod dat;

#[path = "kiss_engine/dat/dat_mesh.rs"]
pub mod dat_mesh;

#[path = "kiss_engine/dtx/dtx.rs"]
pub mod dtx;

#[path = "kiss_engine/egui/egui_renderer.rs"]
pub mod egui_renderer;

#[path = "kiss_engine/collision/collision.rs"]
pub mod collision;