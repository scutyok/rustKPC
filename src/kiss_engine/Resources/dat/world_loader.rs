use std::collections::HashMap;

use anyhow::{Result, anyhow};
use crate::util::math::{Vector2, Vector3};
use log::*;

use crate::abc;
use crate::util::geometry;
use crate::dat;
use crate::dat::PropertyValue;
use crate::dat_mesh;
use crate::dtx;
use crate::game_objects::GameObjectManager;
use crate::LightObj;
use crate::types::*;
use crate::vulkan::texture::get_texture_dimensions;
use crate::DemoSkyWorldModel::SkyModelInfo;
use crate::triggers::TriggerInfo;

//******************************************************************/
//
// Fog settings extracted from a level's WorldProperties object.
//
//******************************************************************/
pub struct LevelFogSettings {
    pub enabled: bool,
    pub color: [f32; 3],
    pub near_z: f32,
    pub far_z: f32,
    /// Whether sky-specific fog is enabled (from SkyFog property).
    pub sky_fog_enabled: bool,
    /// Sky fog far distance (from SkyFogFarZ); 0 = use world fog far.
    pub sky_fog_far: f32,
}

//******************************************************************/
//
// Result of loading a DAT world.
//
//******************************************************************/
pub struct LoadedWorld {
    pub lights: Vec<LightObj::Light>,
    /// Collision mesh positions (includes invisible surfaces for blocking).
    pub collision_positions: Vec<Vector3>,
    /// Collision mesh indices.
    pub collision_indices: Vec<u32>,
    /// Interactive game objects parsed from DAT WorldObjects.
    pub game_objects: GameObjectManager,
    /// Fog settings from the level's WorldProperties object, if present.
    pub fog: Option<LevelFogSettings>,
    /// Entity cylinders for solid objects (barrels, headless enemies) in Vulkan coords.
    pub entity_cylinders: Vec<geometry::EntityCylinder>,
    /// Draw-group indices that belong to trigger / volume sub-models (toggleable).
    pub trigger_draw_groups: Vec<(usize, u32)>,
    /// Triggers and volumes parsed from the DAT objects and sub-world models.
    pub triggers: Vec<TriggerInfo>,
}

//******************************************************************/
//
// Load a KISS Psycho Circus DAT file (v127) and extract mesh data
//
// # Arguments
// * `path` - Path to the .dat file
// * `world_model_index` - Which world model to load (0 = main world)
// * `scale` - Scale factor for the world geometry
//
// # Returns
// A `LoadedWorld` containing lights and collision geometry
//
//******************************************************************/

pub fn load_dat_model<P: AsRef<std::path::Path>>(
    data: &mut AppData,
    path: P,
    world_model_index: usize,
    scale: f32,
) -> Result<LoadedWorld> {
    info!("Loading DAT file: {}", path.as_ref().display());

    // Derive preferred texture subfolder from DAT filename (e.g. "R2M1A.DAT" → "R2M1")
    {
        let stem = path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_uppercase();
        let textures_root = std::path::Path::new("REZ/TEXTURES");
        if textures_root.join(&stem).is_dir() {
            data.texture_prefix = stem;
        } else {
            let trimmed = stem.trim_end_matches(|c: char| c.is_ascii_alphabetic());
            if !trimmed.is_empty() && textures_root.join(trimmed).is_dir() {
                data.texture_prefix = trimmed.to_string();
            } else {
                data.texture_prefix.clear();
            }
        }
        println!("Texture prefix: '{}'", data.texture_prefix);
    }

    let dat_file = dat::DatFile::read_from_file(&path)
        .map_err(|e| anyhow!("Failed to parse DAT file: {}", e))?;

    info!(
        "DAT file loaded: {} objects, {} world models",
        dat_file.objects.len(),
        dat_file.world_models.len()
    );

    // Debug: list all world models to identify sky
    for (i, wm) in dat_file.world_models.iter().enumerate() {
        println!("  WorldModel[{}]: name='{}' flags=0x{:X} translation=({:.1},{:.1},{:.1}) polys={}",
            i, wm.world_name, wm.info_flags,
            wm.world_translation.x, wm.world_translation.y, wm.world_translation.z,
            wm.poly_count);
    }

    if let Some(world) = dat_file.world_models.get(world_model_index) {
        // Compute map center from bounding box (Lithtech Y-up → renderer Z-up)
        data.map_center = [
            (world.min_box.x + world.max_box.x) * 0.5 * scale,
            (world.min_box.z + world.max_box.z) * 0.5 * scale,
            (world.min_box.y + world.max_box.y) * 0.5 * scale,
        ];
        println!("  Map center: ({:.2}, {:.2}, {:.2})", data.map_center[0], data.map_center[1], data.map_center[2]);

        println!("=== WORLD MODEL DEBUG ===");
        println!("  World name: {}", world.world_name);
        println!("  Points: {}", world.points.len());
        println!("  Polygons: {}", world.polygons.len());
        println!("  Surfaces: {}", world.surfaces.len());
        println!("  Planes: {}", world.planes.len());

        for (i, poly) in world.polygons.iter().take(5).enumerate() {
            println!(
                "  Poly[{}]: {} verts, surface_idx={}",
                i,
                poly.disk_verts.len(),
                poly.surface_index
            );
        }
    }

    // Extract render mesh (skip Invisible textures and skyportal surfaces)
    let render_extractor = dat_mesh::MeshExtractor::new(&dat_file)
        .with_scale(scale)
        .with_skip_invisible(false)
        .with_skip_sky(false)
        .with_skip_skyportal(true)
        .with_flip_winding(false)
        .with_skip_textures(vec!["invisible".to_string(), "clip".to_string()]);

    let mesh = render_extractor
        .extract_world_by_index(world_model_index)
        .ok_or_else(|| anyhow!("World model index {} not found", world_model_index))?;

    // Extract collision mesh (include invisible surfaces so they still block the player)
    let collision_extractor = dat_mesh::MeshExtractor::new(&dat_file)
        .with_scale(scale)
        .with_skip_invisible(false)
        .with_skip_sky(false)
        .with_flip_winding(false);

    let collision_mesh = collision_extractor
        .extract_world_by_index(world_model_index)
        .ok_or_else(|| anyhow!("World model index {} not found (collision)", world_model_index))?;

    let mut collision_positions: Vec<Vector3> = collision_mesh
        .vertices
        .iter()
        .map(|v| Vector3::new(v.pos[0], v.pos[1], v.pos[2]))
        .collect();
    let mut collision_indices: Vec<u32> = collision_mesh.indices;

    info!(
        "Extracted mesh '{}': {} vertices, {} indices, {} texture groups",
        mesh.name,
        mesh.vertices.len(),
        mesh.indices.len(),
        mesh.textured_meshes.len()
    );

    let mut texture_name_to_index: HashMap<String, usize> = HashMap::new();
    let mut texture_names: Vec<String> = Vec::new();
    let mut texture_dimensions: HashMap<String, (u32, u32)> = HashMap::new();

    let mut current_vertex_offset = 0u32;
    let mut current_index_offset = 0u32;

    for textured_mesh in &mesh.textured_meshes {
        let texture_index =
            if let Some(&idx) = texture_name_to_index.get(&textured_mesh.texture_name) {
                idx
            } else {
                let idx = texture_names.len();
                texture_names.push(textured_mesh.texture_name.clone());
                texture_name_to_index.insert(textured_mesh.texture_name.clone(), idx);
                idx
            };

        let (tex_width, tex_height) =
            if let Some(&dims) = texture_dimensions.get(&textured_mesh.texture_name) {
                dims
            } else {
                let dims = get_texture_dimensions(&textured_mesh.texture_name, 
                    if data.texture_prefix.is_empty() { None } else { Some(&data.texture_prefix) });
                texture_dimensions.insert(textured_mesh.texture_name.clone(), dims);
                dims
            };

        for dat_vert in &textured_mesh.vertices {
            let normal = Vector3::new(
                dat_vert.normal[0],
                dat_vert.normal[2],
                dat_vert.normal[1],
            );
            let vertex = Vertex {
                pos: Vector3::new(dat_vert.pos[0], dat_vert.pos[1], dat_vert.pos[2]),
                color: Vector3::new(dat_vert.color[0], dat_vert.color[1], dat_vert.color[2]),
                tex_coord: Vector2::new(
                    dat_vert.tex_coord[0] / tex_width as f32,
                    dat_vert.tex_coord[1] / tex_height as f32,
                ),
                normal,
            };
            data.vertices.push(vertex);
        }

        for &idx in &textured_mesh.indices {
            data.indices.push(current_vertex_offset + idx);
        }

        data.draw_groups.push(DrawGroup {
            texture_index,
            first_index: current_index_offset,
            index_count: textured_mesh.indices.len() as u32,
            vertex_offset: 0,
            model_matrix: None,
            sky_opacity: None,
            sky_draw_layer: 0,
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

    //******************************************************************/
    //
    // ABC model objects (barrels, decos, pickups, etc.)
    // Extract models first so we can register their skin textures in the atlas before
    // the texture atlas is built. Geometry is added AFTER subdivision (see below).
    //
    // Collect BSP floor-surface triangles for floor-snapping ground objects.
    // In the original Lithtech engine, objects call MoveToFloor() at spawn to
    // drop from their editor position to the nearest surface below.
    // We triangulate upward-facing polygons to allow precise ray-triangle
    // intersection (vertical ray at object XZ position).
    //
    //******************************************************************/

    let floor_tris: Vec<abc::FloorTri> = dat_file.world_models.get(world_model_index)
        .map(|bsp| {
            let mut tris = Vec::new();
            for poly in &bsp.polygons {
                let si = poly.surface_index as usize;
                if si >= bsp.surfaces.len() { continue; }
                let pi = bsp.surfaces[si].plane_index as usize;
                if pi >= bsp.planes.len() { continue; }
                // Floor-like surface: normal Y > 0.5 (upward-facing in Lithtech Y-up)
                if bsp.planes[pi].normal.y <= 0.5 { continue; }

                // Fan-triangulate the polygon (convex)
                let verts: Vec<[f32; 3]> = poly.disk_verts.iter()
                    .filter_map(|dv| {
                        let vi = dv.vertex_index as usize;
                        bsp.points.get(vi).map(|p| [p.x, p.y, p.z])
                    })
                    .collect();

                for i in 1..verts.len().saturating_sub(1) {
                    tris.push(abc::FloorTri {
                        v0: verts[0],
                        v1: verts[i],
                        v2: verts[i + 1],
                    });
                }
            }
            tris
        })
        .unwrap_or_default();

    let abc_objects = abc::extract_abc_objects(&dat_file.objects, "REZ", scale, path.as_ref().to_str().unwrap_or(""), &floor_tris);
    println!("=== ABC OBJECTS: {} found ===", abc_objects.len());
    {
        let mut type_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for obj in &abc_objects {
            *type_counts.entry(obj.type_name.as_str()).or_insert(0) += 1;
        }
        for (tn, count) in &type_counts {
            println!("  {}: {}", tn, count);
        }
    }

    // Register skin textures now so they're included in the atlas build.
    for abc_obj in &abc_objects {
        let skin_name = abc_obj.skin_filename.clone();
        if !texture_name_to_index.contains_key(&skin_name) {
            let idx = texture_names.len();
            texture_names.push(skin_name.clone());
            texture_name_to_index.insert(skin_name.clone(), idx);
        }
        if !texture_dimensions.contains_key(&skin_name) {
            let dims = if std::path::Path::new(&abc_obj.skin_filename).exists() {
                match dtx::DtxFile::read_from_file(&abc_obj.skin_filename) {
                    Ok(dtx) => (dtx.width as u32, dtx.height as u32),
                    Err(_) => (256, 256),
                }
            } else {
                (256, 256)
            };
            texture_dimensions.insert(skin_name, dims);
        }
    }

    // Spatially subdivide WORLD draw groups only (ABC geometry not yet in the buffer).
    subdivide_draw_groups(
        &data.vertices.clone(),
        &mut data.indices,
        &mut data.draw_groups,
        16.0,
    );

    // NOW add ABC geometry – first_abc_draw_group is accurate because world subdivision is done.
    let first_abc_draw_group = data.draw_groups.len();

    // Per-creature animation data: (abc_index, keyframe_first_indices, keyframe_times_ms, duration_ms)
    let mut creature_anim_data: Vec<(usize, Vec<u32>, Vec<u32>, u32)> = Vec::new();

    for (abc_idx, abc_obj) in abc_objects.iter().enumerate() {
        let skin_name = abc_obj.skin_filename.clone();
        // Skin was registered above; look it up directly.
        let tex_index = *texture_name_to_index.get(&skin_name).unwrap();

        let vert_base = data.vertices.len() as u32;
        let idx_base = data.indices.len() as u32;

        for v in &abc_obj.mesh.vertices {
            data.vertices.push(Vertex {
                pos: Vector3::new(v.pos[0], v.pos[1], v.pos[2]),
                color: Vector3::new(1.0, 1.0, 1.0),
                tex_coord: Vector2::new(v.tex_coord[0], v.tex_coord[1]),
                normal: Vector3::new(v.normal[0], v.normal[1], v.normal[2]),
            });
        }

        for &i in &abc_obj.mesh.indices {
            data.indices.push(vert_base + i);
        }

        data.draw_groups.push(DrawGroup {
            texture_index: tex_index,
            first_index: idx_base,
            index_count: abc_obj.mesh.indices.len() as u32,
            vertex_offset: 0,
            model_matrix: None,
            sky_opacity: None,
            sky_draw_layer: 0,
        });

        // Add extra keyframe meshes for animated creatures
        if !abc_obj.anim_frame_meshes.is_empty() {
            let mut kf_first_indices = vec![idx_base]; // keyframe 0

            for kf_mesh in &abc_obj.anim_frame_meshes {
                let kf_vert_base = data.vertices.len() as u32;
                let kf_idx_base = data.indices.len() as u32;

                for v in &kf_mesh.vertices {
                    data.vertices.push(Vertex {
                        pos: Vector3::new(v.pos[0], v.pos[1], v.pos[2]),
                        color: Vector3::new(1.0, 1.0, 1.0),
                        tex_coord: Vector2::new(v.tex_coord[0], v.tex_coord[1]),
                        normal: Vector3::new(v.normal[0], v.normal[1], v.normal[2]),
                    });
                }

                for &i in &kf_mesh.indices {
                    data.indices.push(kf_vert_base + i);
                }

                kf_first_indices.push(kf_idx_base);
            }

            creature_anim_data.push((
                abc_idx,
                kf_first_indices,
                abc_obj.anim_keyframe_times_ms.clone(),
                abc_obj.anim_duration_ms,
            ));

            log::info!(
                "Added {} animation keyframes for creature '{}' ({} extra verts, {} extra indices)",
                abc_obj.anim_frame_meshes.len(),
                abc_obj.type_name,
                abc_obj.anim_frame_meshes.iter().map(|m| m.vertices.len()).sum::<usize>(),
                abc_obj.anim_frame_meshes.iter().map(|m| m.indices.len()).sum::<usize>(),
            );
        }
    }

    let placed_abc_objects = abc_objects;

    // Dynamically gather sky model names from DemoSkyWorldModel and SkyPointer objects
    // Also collect optional pan/rotation properties from DemoSkyWorldModel objects
    let mut sky_model_props: std::collections::HashMap<String, (Option<f32>, Option<f32>, Option<f32>)> = std::collections::HashMap::new();

    let sky_model_names: std::collections::HashSet<String> = {
        let mut set = std::collections::HashSet::new();
        for obj in &dat_file.objects {
            if obj.type_name == "DemoSkyWorldModel" || obj.type_name == "SkyPointer" {
                if let Some(PropertyValue::String(name)) = obj.get_property("Name") {
                    let name_lc = name.to_lowercase();
                    set.insert(name_lc.clone());
                    // If this is an actual DemoSkyWorldModel, try to read pan/rot props
                    if obj.type_name == "DemoSkyWorldModel" {
                        let mut pan_x: Option<f32> = None;
                        let mut pan_y: Option<f32> = None;
                        let mut rot_speed: Option<f32> = None;
                        for prop in &obj.properties {
                            let key = prop.name.to_lowercase();
                            match &prop.value {
                                PropertyValue::Vector(v) => {
                                    // Convert DAT vector (x,y,z) → renderer (x,z,y)
                                    let vx = v.x;
                                    let vy = v.z;
                                    if key == "pan" || key == "panvec" || (key.contains("pan") && (key.contains("vec") || key.contains("vector"))) {
                                        pan_x = Some(vx);
                                        pan_y = Some(vy);
                                    }
                                }
                                PropertyValue::Float(f) => {
                                    if key == "panx" || key == "panspeedx" || key == "pan_x" {
                                        pan_x = Some(*f);
                                    } else if key == "pany" || key == "panspeedy" || key == "pan_y" {
                                        pan_y = Some(*f);
                                    } else if key == "panspeed" {
                                        // If a single scalar is provided, use it as X pan by default
                                        if pan_x.is_none() {
                                            pan_x = Some(*f);
                                        }
                                    } else if (key.contains("rot") && key.contains("speed")) || key == "rotatespeed" || key == "rotationspeed" {
                                        rot_speed = Some(*f);
                                    }
                                }
                                _ => {}
                            }
                        }
                        sky_model_props.insert(name_lc, (pan_x, pan_y, rot_speed));
                    }
                }
                // SkyPointer may also reference via SkyObjectName
                if let Some(PropertyValue::String(name)) = obj.get_property("SkyObjectName") {
                    set.insert(name.to_lowercase());
                }
            }
        }
        // Always include the classic names as fallback
        set.insert("sky".to_string());
        set.insert("clouds".to_string());
        set.insert("clouds2".to_string());
        // These are additional BSP pieces of the authored skybox, not world
        // geometry.  Leaving them in the normal pass makes them intersect the
        // playable map as the large green/black silhouettes.
        set.insert("ground".to_string());
        set.insert("skybuildings".to_string());
        set
    };
    debug!("Sky model names for this level: {:?}", sky_model_names);

    //******************************************************************/
    //
    // Render all sub-world models (doors, windows, crates, ceiling fans, etc.)
    // These are BSP world models with indices 1..end that are not sky/cloud models.
    // Their polygon vertices are in sub-model-local space; we apply world_translation
    // to position them in the world.
    // bsp_submodels tracks (world_name, position_vulkan, draw_group_indices, z_height) for
    // animated objects like ceiling fans and BSP doors.
    //
    //******************************************************************/
    let mut bsp_submodels: Vec<(String, [f32; 3], Vec<usize>, f32)> = Vec::new();
    let mut trigger_draw_groups: Vec<(usize, u32)> = Vec::new();
    // Trigger volumes: (name, AABB min, AABB max) for volumes the player can interact with.
    let mut trigger_volumes: Vec<(String, [f32; 3], [f32; 3])> = Vec::new();
    // Collision vertex ranges for doors: (lowercase_name, start_index, end_index)
    let mut door_collision_ranges: Vec<(String, usize, usize)> = Vec::new();
    {
        let sky_names = &sky_model_names;

        //******************************************************************/
        //
        // Use skip_invisible(false) for all sub-world models: Lithtech marks
        // many valid sub-model surfaces (fences, grates, thin brushes) with the
        // INVISIBLE flag.  The original engine renders them through the object
        // system; we must include them here.  Non-renderable surfaces are still
        // filtered out by texture name ("invisible", "clip").
        //
        //******************************************************************/

        let sub_extractor = dat_mesh::MeshExtractor::new(&dat_file)
            .with_scale(scale)
            .with_skip_invisible(false)
            .with_skip_sky(true)
            .with_skip_skyportal(true)
            .with_flip_winding(false)
            .with_skip_textures(vec!["invisible".to_string(), "clip".to_string()]);

        for wm_idx in 1..dat_file.world_models.len() {
            let wm = &dat_file.world_models[wm_idx];
            let name_lower = wm.world_name.to_lowercase();
            if sky_names.contains(&name_lower) {
                continue;
            }

            let is_fan = name_lower.starts_with("crotatingceilingfan");

            let is_trigger = name_lower.starts_with("volumebrush")
                || name_lower.starts_with("cvolume")
                || name_lower.starts_with("cnodevolume")
                || name_lower.starts_with("cdamagevolume")
                || name_lower.starts_with("ckillvolume")
                || name_lower.starts_with("trigger")
                || name_lower.starts_with("ctrigger")
                // Exit brushes are named CExitTriggerN, which does not match
                // the CTrigger prefix above. They must contribute their BSP
                // AABB so the DAT CExitTrigger can activate across the full
                // end-of-level volume rather than a tiny point fallback.
                || name_lower.starts_with("cexittrigger")
                || name_lower.starts_with("cpickuptrigger")
                || name_lower.starts_with("lava")
                || name_lower.starts_with("clava")
                || name_lower.starts_with("cslime")
                || name_lower.starts_with("zone")
                || name_lower.starts_with("portalbrush")
                || name_lower.starts_with("deathvolume")
                || name_lower.starts_with("cdeathvolume")
                || name_lower.starts_with("teleportvolume")
                || name_lower.starts_with("cteleportvolume");

            let sub_mesh = match sub_extractor
                .extract_world_by_index(wm_idx)
            {
                Some(m) => m,
                None => continue,
            };

            if sub_mesh.vertices.is_empty() {
                continue;
            }

            //******************************************************************/
            //
            // Translation in Vulkan coords (Lithtech Y-up => Vulkan Z-up).
            // NOTE: sub-model BSP vertices are stored in world space already;
            // world_translation is the model pivot (used as rotation centre for
            // animated objects such as fans) — do NOT add it to vertex positions.
            //
            //******************************************************************/

            let tx = wm.world_translation.x * scale;
            let ty = wm.world_translation.z * scale;
            let tz = wm.world_translation.y * scale;

            let mut this_model_dgs: Vec<usize> = Vec::new();
            let is_door = name_lower.starts_with("door") || name_lower.starts_with("freezedoor");
            let collision_start = collision_positions.len();

            for textured_mesh in &sub_mesh.textured_meshes {
                if textured_mesh.indices.is_empty() {
                    continue;
                }

                // Add collision geometry for ladders, chainlink/fence, and door sub-models
                let is_collision_submodel = name_lower.starts_with("cladder")
                    || name_lower.starts_with("midtexture")
                    || name_lower.starts_with("door")
                    || name_lower.starts_with("freezedoor");
                if is_collision_submodel {
                    let base = collision_positions.len() as u32;
                    for v in &textured_mesh.vertices {
                        collision_positions.push(Vector3::new(v.pos[0], v.pos[1], v.pos[2]));
                    }
                    for &i in &textured_mesh.indices {
                        collision_indices.push(base + i);
                    }
                }

                let skin_name = textured_mesh.texture_name.clone();
                let tex_index = if let Some(&idx) = texture_name_to_index.get(&skin_name) {
                    idx
                } else {
                    let idx = texture_names.len();
                    texture_names.push(skin_name.clone());
                    texture_name_to_index.insert(skin_name.clone(), idx);
                    if !texture_dimensions.contains_key(&skin_name) {
                        texture_dimensions.insert(skin_name.clone(), get_texture_dimensions(&skin_name,
                            if data.texture_prefix.is_empty() { None } else { Some(&data.texture_prefix) }));
                    }
                    idx
                };

                let (tex_w, tex_h) = texture_dimensions
                    .get(&skin_name)
                    .copied()
                    .unwrap_or((256, 256));

                let vert_base = data.vertices.len() as u32;
                let idx_base = data.indices.len() as u32;

                for v in &textured_mesh.vertices {
                    data.vertices.push(Vertex {
                        pos: Vector3::new(v.pos[0], v.pos[1], v.pos[2]),
                        color: Vector3::new(v.color[0], v.color[1], v.color[2]),
                        tex_coord: Vector2::new(v.tex_coord[0] / tex_w as f32, v.tex_coord[1] / tex_h as f32),
                        normal: Vector3::new(v.normal[0], v.normal[1], v.normal[2]),
                    });
                }

                for &i in &textured_mesh.indices {
                    data.indices.push(vert_base + i);
                }

                let dg_idx = data.draw_groups.len();
                data.draw_groups.push(DrawGroup {
                    texture_index: tex_index,
                    first_index: idx_base,
                    index_count: textured_mesh.indices.len() as u32,
                    vertex_offset: 0,
                    model_matrix: None,
                    sky_opacity: None,
                    sky_draw_layer: 0,
                });
                this_model_dgs.push(dg_idx);
                if is_trigger {
                    trigger_draw_groups.push((dg_idx, textured_mesh.indices.len() as u32));
                }
            }

            // Record collision vertex range for door sub-models
            let collision_end = collision_positions.len();
            if is_door && collision_end > collision_start {
                door_collision_ranges.push((name_lower.clone(), collision_start, collision_end));
            }

            if !this_model_dgs.is_empty() {
                // For fans, use the geometric centroid of the actual mesh vertices as the
                // rotation pivot. world_translation is often (0,0,0) for BSP sub-models
                // that bake their position into vertex coordinates, so it can't be trusted
                // as the spin center.
                let all_verts: Vec<_> = sub_mesh
                    .textured_meshes
                    .iter()
                    .flat_map(|tm| tm.vertices.iter())
                    .collect();
                let pivot = if is_fan {
                    if all_verts.is_empty() {
                        [tx, ty, tz]
                    } else {
                        let n = all_verts.len() as f32;
                        [
                            all_verts.iter().map(|v| v.pos[0]).sum::<f32>() / n,
                            all_verts.iter().map(|v| v.pos[1]).sum::<f32>() / n,
                            all_verts.iter().map(|v| v.pos[2]).sum::<f32>() / n,
                        ]
                    }
                } else {
                    [tx, ty, tz]
                };
                // Compute Z height (up axis in Vulkan) for BSP door slide distance.
                let z_height = if all_verts.is_empty() {
                    2.0
                } else {
                    let z_min = all_verts.iter().map(|v| v.pos[2]).fold(f32::MAX, f32::min);
                    let z_max = all_verts.iter().map(|v| v.pos[2]).fold(f32::MIN, f32::max);
                    z_max - z_min
                };
                bsp_submodels.push((wm.world_name.clone(), pivot, this_model_dgs, z_height));

                // Track trigger volumes with their AABB for script activation.
                if is_trigger && !all_verts.is_empty() {
                    let mut vmin = [f32::MAX; 3];
                    let mut vmax = [f32::MIN; 3];
                    for v in &all_verts {
                        for k in 0..3 {
                            vmin[k] = vmin[k].min(v.pos[k]);
                            vmax[k] = vmax[k].max(v.pos[k]);
                        }
                    }
                    println!("  Trigger volume '{}': AABB ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2})",
                        wm.world_name, vmin[0], vmin[1], vmin[2], vmax[0], vmax[1], vmax[2]);
                    trigger_volumes.push((wm.world_name.clone(), vmin, vmax));
                }
            }
        }
        println!("=== Sub-world models rendered: {} world models processed ===", dat_file.world_models.len() - 1);
    }

    //******************************************************************/
    //
    // ── Animated torch flame billboard quads ───────────────────────────────
    // For every CTorch ABC object, create a transparent billboard quad that
    // cycles through the 6 TORCH*.DTX frames each frame in game_objects.rs.
    // Each torch gets its own quad geometry placed at its world position so
    // the occlusion culler's AABB (built from vertex positions) is correct.
    //
    //******************************************************************/

    // Prefer the TRCH* frames (they look like the original in-game torch),
    // fall back to TORCH* if TRCH* not present.
    let flame_candidates = [
        [
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH1.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH2.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH3.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH4.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH5.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TRCH6.DTX",
        ],
        [
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH1.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH2.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH3.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH4.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH5.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH6.DTX",
        ],
    ];
    let mut chosen_frames: [&str; 6] = ["", "", "", "", "", ""];
    for cand in &flame_candidates {
        if std::path::Path::new(cand[0]).exists() {
            for i in 0..6 { chosen_frames[i] = cand[i]; }
            break;
        }
    }
    // If neither candidate set found, fall back to TORCH* names (may be missing)
    if chosen_frames[0].is_empty() {
        chosen_frames = [
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH1.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH2.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH3.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH4.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH5.DTX",
            "REZ/SPRITETEXTURES/FLAMETEST/TORCH6.DTX",
        ];
    }
    // Register all 6 frame textures (if not already present) and record the
    // index of the first frame so frames are at base, base+1 … base+5.
    let flame_base_tex_index = texture_names.len();
    for frame_path in &chosen_frames {
        let name = frame_path.to_string();
        if !texture_name_to_index.contains_key(&name) {
            let idx = texture_names.len();
            texture_names.push(name.clone());
            texture_name_to_index.insert(name.clone(), idx);
            if !texture_dimensions.contains_key(&name) {
                let dims = if std::path::Path::new(frame_path).exists() {
                    match dtx::DtxFile::read_from_file(frame_path) {
                        Ok(dtx) => (dtx.width as u32, dtx.height as u32),
                        Err(_) => (64, 64),
                    }
                } else {
                    (64, 64)
                };
                texture_dimensions.insert(name, dims);
            }
        }
    }

    // Quad geometry constants (Vulkan units, scale already applied to pos).
    const FLAME_WIDTH: f32 = 0.28;   // billboard half-width on each side
    const FLAME_HEIGHT: f32 = 0.38;  // billboard total height
    const FLAME_Z_OFFSET: f32 = 0.08; // raise above torch model base

    // (abc_index, flame_draw_group, flame_base_tex_index) for each CTorch.
    let mut torch_flames: Vec<(usize, usize, usize)> = Vec::new();

    for (i, abc) in placed_abc_objects.iter().enumerate() {
        // Support both CTorch and CTorchColored ABC types (maps may use either)
        if abc.type_name != "CTorch" && abc.type_name != "CTorchColored" {
            continue;
        }
        let pos = abc.position; // already in Vulkan/renderer space

        let fw = FLAME_WIDTH * 0.5;
        let fz_bot = pos[2] + FLAME_Z_OFFSET;
        let fz_top = fz_bot + FLAME_HEIGHT;

        // Four vertices of the quad in world space, initially facing +Y.
        // Billboard rotation (applied via model_matrix each frame) will swing
        // the quad to face the camera in the XY plane.
        //   Bottom-left, Bottom-right, Top-right, Top-left
        let quad_verts = [
            ([pos[0] - fw, pos[1], fz_bot], [0.0_f32, 1.0_f32]),
            ([pos[0] + fw, pos[1], fz_bot], [1.0, 1.0]),
            ([pos[0] + fw, pos[1], fz_top], [1.0, 0.0]),
            ([pos[0] - fw, pos[1], fz_top], [0.0, 0.0]),
        ];

        let vert_base = data.vertices.len() as u32;
        let idx_base = data.indices.len() as u32;

        for (p, uv) in &quad_verts {
            data.vertices.push(Vertex {
                pos: Vector3::new(p[0], p[1], p[2]),
                color: Vector3::new(1.0, 1.0, 1.0), // no pre-baked shadow on flame sprite
                tex_coord: Vector2::new(uv[0], uv[1]),
                normal: Vector3::new(0.0, 1.0, 0.0),
            });
        }
        // Two triangles: 0-1-2, 0-2-3
        data.indices.extend_from_slice(&[
            vert_base, vert_base + 1, vert_base + 2,
            vert_base, vert_base + 2, vert_base + 3,
        ]);

        let flame_dg = data.draw_groups.len();
        data.draw_groups.push(DrawGroup {
            texture_index: flame_base_tex_index, // starts at frame 0 (TORCH1)
            first_index: idx_base,
            index_count: 6,
            vertex_offset: 0,
            model_matrix: None,
            sky_opacity: None,
            sky_draw_layer: 0,
        });

        torch_flames.push((i, flame_dg, flame_base_tex_index));
        println!("  CTorch[{}] flame quad: dg={} base_tex={} pos=({:.2},{:.2},{:.2})",
            i, flame_dg, flame_base_tex_index, pos[0], pos[1], pos[2]);
    }
    println!("=== Torch flame quads: {} ===", torch_flames.len());

    // Mark where sky draw groups will start (after all world + ABC + sub-world groups)
    data.sky_draw_group_start = data.draw_groups.len();
    data.sky_bounds_min = [f32::INFINITY; 3];
    data.sky_bounds_max = [f32::NEG_INFINITY; 3];

    // Extract sky world models and collect SkyModelInfo for animation
    let mut sky_model_infos: Vec<SkyModelInfo> = Vec::new();
    {
        let sky_extractor = dat_mesh::MeshExtractor::new(&dat_file)
            .with_scale(scale)
            .with_skip_invisible(false)
            .with_skip_sky(false)
            .with_skip_skyportal(false)
            .with_flip_winding(false)
            .with_skip_textures(vec!["invisible".to_string()]);

        // Find the first sky model to use its translation as the sky origin
        for wm in &dat_file.world_models {
            let name_lower = wm.world_name.to_lowercase();
            if sky_model_names.contains(&name_lower) {
                // Swizzle Y/Z to match renderer coordinate system (Lithtech Y-up → renderer Z-up)
                data.sky_translation = [
                    wm.world_translation.x * scale,
                    wm.world_translation.z * scale,
                    wm.world_translation.y * scale,
                ];
                break;
            }
        }

        for (wm_idx, wm) in dat_file.world_models.iter().enumerate() {
            let name_lower = wm.world_name.to_lowercase();
            if !sky_model_names.contains(&name_lower) {
                continue;
            }

            let dg_start = data.draw_groups.len();

            if let Some(sky_mesh) = sky_extractor.extract_world_by_index(wm_idx) {
                println!("=== SKY MESH '{}': {} verts, {} indices, {} groups, translation=({:.1},{:.1},{:.1}) ===",
                    wm.world_name, sky_mesh.vertices.len(), sky_mesh.indices.len(),
                    sky_mesh.textured_meshes.len(),
                    wm.world_translation.x, wm.world_translation.y, wm.world_translation.z);

                for textured_mesh in &sky_mesh.textured_meshes {
                    let tex_index =
                        if let Some(&idx) = texture_name_to_index.get(&textured_mesh.texture_name) {
                            idx
                        } else {
                            let idx = texture_names.len();
                            texture_names.push(textured_mesh.texture_name.clone());
                            texture_name_to_index.insert(textured_mesh.texture_name.clone(), idx);
                            idx
                        };

                    let (tex_width, tex_height) =
                        if let Some(&dims) = texture_dimensions.get(&textured_mesh.texture_name) {
                            dims
                        } else {
                            let dims = get_texture_dimensions(&textured_mesh.texture_name,
                                if data.texture_prefix.is_empty() { None } else { Some(&data.texture_prefix) });
                            texture_dimensions.insert(textured_mesh.texture_name.clone(), dims);
                            dims
                        };

                    println!("  SKY TEX '{}': {}x{} ({} verts, {} idx)",
                        textured_mesh.texture_name, tex_width, tex_height,
                        textured_mesh.vertices.len(), textured_mesh.indices.len());

                    let group_vert_base = data.vertices.len() as u32;
                    let group_idx_base = data.indices.len() as u32;

                    for dat_vert in &textured_mesh.vertices {
                        let normal = Vector3::new(
                            dat_vert.normal[0],
                            dat_vert.normal[2],
                            dat_vert.normal[1],
                        );
                        let pos = Vector3::new(dat_vert.pos[0], dat_vert.pos[1], dat_vert.pos[2]);
                        data.sky_bounds_min[0] = data.sky_bounds_min[0].min(pos.x);
                        data.sky_bounds_min[1] = data.sky_bounds_min[1].min(pos.y);
                        data.sky_bounds_min[2] = data.sky_bounds_min[2].min(pos.z);
                        data.sky_bounds_max[0] = data.sky_bounds_max[0].max(pos.x);
                        data.sky_bounds_max[1] = data.sky_bounds_max[1].max(pos.y);
                        data.sky_bounds_max[2] = data.sky_bounds_max[2].max(pos.z);

                        data.vertices.push(Vertex {
                            pos,
                            color: Vector3::new(dat_vert.color[0], dat_vert.color[1], dat_vert.color[2]),
                            tex_coord: Vector2::new(
                                dat_vert.tex_coord[0] / tex_width as f32,
                                dat_vert.tex_coord[1] / tex_height as f32,
                            ),
                            normal,
                        });
                    }

                    for &idx in &textured_mesh.indices {
                        data.indices.push(group_vert_base + idx);
                    }

                    // Sky world models fall into three draw layers so they
                    // composite correctly with no depth writes:
                    //   0 = "sky" — the solid background dome. Drawn first
                    //       (furthest back) so every other sky layer paints
                    //       over it.
                    //   1 = cloud layers — translucent overlays (blend at
                    //       30%) drawn over the dome but still beneath the
                    //       foreground skybox models.
                    //   2 = everything else (ground, skybuildings, and any
                    //       other named skybox models such as mountains) —
                    //       opaque foreground silhouettes drawn last, so
                    //       they occlude the clouds/dome where they overlap.
                    // Previously clouds and foreground models were lumped
                    // into one "opaque" bucket that always drew after
                    // clouds; if the dome itself ended up in that bucket
                    // (which it did) it could redraw after the clouds and
                    // paint over them, making the clouds invisible.
                    let (sky_opacity, sky_draw_layer): (f32, u8) =
                        if name_lower == "sky" {
                            (-1.0, 0)
                        } else if name_lower.contains("cloud") {
                            (-0.3, 1)
                        } else {
                            (-1.0, 2)
                        };
                    data.draw_groups.push(DrawGroup {
                        texture_index: tex_index,
                        first_index: group_idx_base,
                        index_count: textured_mesh.indices.len() as u32,
                        vertex_offset: 0,
                        model_matrix: None,
                        sky_opacity: Some(sky_opacity),
                        sky_draw_layer,
                    });
                }

                // Track this sky model's draw group range for animation
                let dg_count = data.draw_groups.len() - dg_start;
                if dg_count > 0 {
                    let name_lc = wm.world_name.to_lowercase();
                    let (raw_pan_x, raw_pan_y, raw_rot_speed) = sky_model_props.get(&name_lc).copied().unwrap_or((None, None, None));
                    // Always use the renderer's animated defaults when a
                    // level omits explicit values.  Several original maps
                    // disable PanSky even though their layered cloud BSPs
                    // still need to scroll in this renderer.
                    let pan_x = raw_pan_x;
                    let pan_y = raw_pan_y;
                    let rot_speed = raw_rot_speed; // rotation not controlled by PanSky
                    sky_model_infos.push(SkyModelInfo {
                        name: name_lc,
                        draw_group_start: dg_start,
                        draw_group_count: dg_count,
                        pan_x,
                        pan_y,
                        rot_speed,
                    });
                }
            }
        }

        println!("=== Sky groups: {} (starting at index {}) ===",
            data.draw_groups.len() - data.sky_draw_group_start,
            data.sky_draw_group_start);
        if data.sky_bounds_min[0].is_infinite() {
            data.sky_bounds_min = [0.0; 3];
            data.sky_bounds_max = [0.0; 3];
        }
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

    println!("=== MESH BOUNDS (after scale {}) ===", scale);
    println!(
        "  Min: ({:.2}, {:.2}, {:.2})",
        min_pos[0], min_pos[1], min_pos[2]
    );
    println!(
        "  Max: ({:.2}, {:.2}, {:.2})",
        max_pos[0], max_pos[1], max_pos[2]
    );
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
    println!(
        "  Center: ({:.2}, {:.2}, {:.2})",
        center[0], center[1], center[2]
    );
    println!(
        "  Size: ({:.2}, {:.2}, {:.2})",
        size[0], size[1], size[2]
    );

    info!(
        "Loaded {} vertices, {} indices, {} draw groups from DAT file",
        data.vertices.len(),
        data.indices.len(),
        data.draw_groups.len()
    );

    let world_lights = LightObj::extract_lights_from_objects(&dat_file.objects, scale);

    {
        let mut types: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for obj in &dat_file.objects {
            types.insert(&obj.type_name);
        }
        let mut sorted: Vec<&str> = types.into_iter().collect();
        sorted.sort();
        println!("=== DAT OBJECT TYPES ({} unique) ===", sorted.len());
        for t in &sorted {
            println!("  {}", t);
        }
        println!("  Lights found: {}", world_lights.len());
    }

    // Extract fog settings from the WorldProperties object.
    let level_fog = dat_file.objects.iter()
        .find(|o| o.type_name == "WorldProperties")
        .map(|wp| {
            // Print ALL WorldProperties for diagnostics
            println!("=== WorldProperties ({} props) ===", wp.properties.len());
            for prop in &wp.properties {
                println!("  {:?} = {:?}", prop.name, prop.value);
            }

            let enabled = match wp.get_property("EnableFog").or_else(|| wp.get_property("FogEnable")) {
                Some(PropertyValue::Bool(b)) => *b != 0,
                _ => false,
            };
            let color = match wp.get_property("FogColor") {
                Some(PropertyValue::Color(c)) | Some(PropertyValue::Vector(c)) =>
                    [c.x / 255.0, c.y / 255.0, c.z / 255.0],
                _ => [0.05, 0.05, 0.08],
            };
            let near_z = match wp.get_property("FogNearZ") {
                Some(PropertyValue::Float(f)) => *f * scale,
                _ => 5.0,
            };
            let far_z = match wp.get_property("FogFarZ") {
                Some(PropertyValue::Float(f)) => *f * scale,
                _ => 22.0,
            };
            let sky_fog_enabled = match wp.get_property("SkyFog") {
                Some(PropertyValue::Bool(b)) => *b != 0,
                _ => enabled,
            };
            let sky_fog_far = match wp.get_property("SkyFogFarZ") {
                Some(PropertyValue::Float(f)) => *f * scale,
                _ => far_z,
            };
            println!("=== Fog: enabled={} color=[{:.3},{:.3},{:.3}] near={:.2} far={:.2} (scale={}) ===",
                enabled, color[0], color[1], color[2], near_z, far_z, scale);
            println!("=== SkyFog: enabled={} far={:.2} ===", sky_fog_enabled, sky_fog_far);
            LevelFogSettings { enabled, color, near_z, far_z, sky_fog_enabled, sky_fog_far }
        });

    // Build the interactive game object manager from placed ABC objects and BSP sub-models.
    let mut game_objects = GameObjectManager::extract_from_dat(
        &dat_file.objects,
        &placed_abc_objects,
        first_abc_draw_group,
        &bsp_submodels,
        &torch_flames,
        &sky_model_infos,
        &trigger_volumes,
        &door_collision_ranges,
        &collision_positions,
        scale,
        &creature_anim_data,
    );

    //******************************************************************/
    //
    // Load SCR scripts from the matching MAPDATA directory.
    // DAT path: REZ/WORLDS/REALM1/R1M1A.DAT
    // SCR dir:  REZ/MAPDATA/REALM1/R1M1/R1M1A/
    //
    //******************************************************************/

    {
        use crate::scripted_sequence;
        let dat_path = path.as_ref();
        let file_stem = dat_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let realm_dir = dat_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
        // Map group is first 4 chars of filename stem, e.g. R1M1A → R1M1
        let map_group = if file_stem.len() >= 4 { &file_stem[..4] } else { file_stem };
        let scr_dir = format!("REZ/MAPDATA/{}/{}/{}", realm_dir, map_group, file_stem);
        println!("=== SCR directory: {} ===", scr_dir);



        if let Ok(entries) = std::fs::read_dir(&scr_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name();
                let fname_str = fname.to_string_lossy();
                if !fname_str.to_uppercase().ends_with(".SCR") {
                    continue;
                }
                // Derive trigger name: strip map prefix and extension.
                // R1M1A_BIGDOOR.SCR → BIGDOOR → "bigdoor"
                let stem = &fname_str[..fname_str.len() - 4]; // strip .SCR
                let prefix = format!("{}_", file_stem);
                let trigger_name = if stem.to_uppercase().starts_with(&prefix.to_uppercase()) {
                    stem[prefix.len()..].to_lowercase()
                } else {
                    stem.to_lowercase()
                };

                if let Ok(contents) = std::fs::read_to_string(entry.path()) {
                    let commands = scripted_sequence::parse_scr_with_prefix(&contents, file_stem);
                    println!("  Loaded SCR '{}' → trigger='{}' ({} commands)",
                        fname_str, trigger_name, commands.len());
                    game_objects.scripts.push((
                        trigger_name,
                        scripted_sequence::ScriptRunner::new(commands),
                    ));
                }
            }
        }
        // Build CScriptObject lookup: name → list of SCR trigger names
        // CScriptObjects have command properties like "0 scriptplay mapdata\...\r1m1a_bigdoor.scr"
        use std::collections::HashMap;
        let mut script_object_scr: HashMap<String, Vec<String>> = HashMap::new();
        for obj in &dat_file.objects {
            if obj.type_name == "CScriptObject" {
                let obj_name = match obj.get_property("Name") {
                    Some(PropertyValue::String(s)) => s.to_lowercase(),
                    _ => continue,
                };
                let mut scr_names = Vec::new();
                for i in 1..=8 {
                    let cmd_key = format!("command{}", i);
                    if let Some(PropertyValue::String(cmd)) = obj.get_property(&cmd_key) {
                        // Parse: "0 scriptplay mapdata\realm1\r1m1\r1m1a\r1m1a_bigdoor.scr"
                        let parts: Vec<&str> = cmd.split_whitespace().collect();
                        if parts.len() >= 3 && parts[1] == "scriptplay" {
                            let scr_path = parts[2].replace('\\', "/").to_lowercase();
                            // Extract trigger name: last component, strip map prefix + .scr
                            if let Some(filename) = scr_path.rsplit('/').next() {
                                let stem = filename.strip_suffix(".scr").unwrap_or(filename);
                                let scr_prefix = format!("{}_", file_stem.to_lowercase());
                                let trigger = if stem.starts_with(&scr_prefix) {
                                    stem[scr_prefix.len()..].to_string()
                                } else {
                                    stem.to_string()
                                };
                                println!("  CScriptObject '{}' → scriptplay → trigger='{}'", obj_name, trigger);
                                scr_names.push(trigger);
                            }
                        }
                    }
                }
                if !scr_names.is_empty() {
                    script_object_scr.insert(obj_name, scr_names);
                }
            }
        }

        // Register BSP switches: parse their commands to find script_object_play targets,
        // then resolve through CScriptObject chain to find ultimate SCR trigger names.
        for (name, pivot, dgs, _zh) in &bsp_submodels {
            let nl = name.to_lowercase();
            if nl.starts_with("cswitchslide") || nl.starts_with("cswitchrotating") {
                // Find matching DAT object by Name property
                let dat_obj = dat_file.objects.iter().find(|o|
                    (o.type_name == "CSwitchSlide" || o.type_name == "CSwitchRotating")
                    && matches!(o.get_property("Name"), Some(PropertyValue::String(n)) if n.to_lowercase() == nl)
                );
                let mut actions = Vec::new();
                if let Some(obj) = dat_obj {
                    for i in 1..=4 {
                        let cmd_key = format!("command{}", i);
                        if let Some(PropertyValue::String(cmd)) = obj.get_property(&cmd_key) {
                            let parsed = scripted_sequence::parse_dat_command(
                                cmd,
                                file_stem,
                                &script_object_scr,
                            );
                            if !parsed.is_empty() {
                                println!("  BSP switch '{}' -> command '{}' -> {:?}", name, cmd, parsed);
                                actions.extend(parsed);
                                continue;
                            }
                            let parts: Vec<&str> = cmd.split_whitespace().collect();
                            if parts.len() >= 3 && parts[1] == "script_object_play" {
                                let target_name = parts[2].to_lowercase();
                                // Resolve through CScriptObject chain
                                if let Some(scr_names) = script_object_scr.get(&target_name) {
                                    actions.extend(scr_names.iter().cloned().map(|script_name| {
                                        scripted_sequence::ScriptCommand::StartScript { script_name }
                                    }));
                                    println!("  BSP switch '{}' → script_object '{}' → scripts: {:?}",
                                        name, target_name, scr_names);
                                }
                            }
                        }
                    }
                }
                if !actions.is_empty() {
                    game_objects.bsp_switches.push(
                        crate::game_objects::BspSwitch {
                            name: nl,
                            center: *pivot,
                            actions,
                            activated: false,
                            draw_groups: dgs.clone(),
                        }
                    );
                }
            }
        }

        // Also connect trigger volumes to CScriptObjects via command matching.
        // Some trigger volumes (CTriggerCommand) have scriptplay commands directly.
        for trigger in &mut game_objects.script_triggers {
            // Check if there's a CTriggerCommand DAT object matching this trigger volume
            let dat_obj = dat_file.objects.iter().find(|o| {
                matches!(o.get_property("Name"), Some(PropertyValue::String(n)) if n.to_lowercase() == trigger.name)
            });
            if let Some(obj) = dat_obj {
                let mut actions = Vec::new();
                for i in 1..=4 {
                        let cmd_key = format!("command{}", i);
                        if let Some(PropertyValue::String(cmd)) = obj.get_property(&cmd_key) {
                            let parsed = scripted_sequence::parse_dat_command(
                                cmd,
                                file_stem,
                                &script_object_scr,
                            );
                            if !parsed.is_empty() {
                                println!("  Trigger volume '{}' -> command '{}' -> {:?}", trigger.name, cmd, parsed);
                                actions.extend(parsed);
                                continue;
                            }
                            let parts: Vec<&str> = cmd.split_whitespace().collect();
                            if parts.len() >= 3 {
                            if parts[1] == "scriptplay" {
                                let scr_path = parts[2].replace('\\', "/").to_lowercase();
                                if let Some(filename) = scr_path.rsplit('/').next() {
                                    let stem = filename.strip_suffix(".scr").unwrap_or(filename);
                                    let scr_prefix = format!("{}_", file_stem.to_lowercase());
                                    let scr_trigger = if stem.starts_with(&scr_prefix) {
                                        stem[scr_prefix.len()..].to_string()
                                    } else {
                                        stem.to_string()
                                    };
                                    // Store the resolved script name in the trigger's name for matching
                                    println!("  Trigger volume '{}' → scriptplay → '{}'", trigger.name, scr_trigger);
                                    actions.push(scripted_sequence::ScriptCommand::StartScript {
                                        script_name: scr_trigger,
                                    });
                                }
                            } else if parts[1] == "script_object_play" {
                                let target = parts[2].to_lowercase();
                                if let Some(scr_names) = script_object_scr.get(&target) {
                                    if let Some(first) = scr_names.first() {
                                        println!("  Trigger volume → script_object '{}' → '{}'", target, first);
                                        actions.push(scripted_sequence::ScriptCommand::StartScript {
                                            script_name: first.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                if !actions.is_empty() {
                    trigger.actions = actions;
                }
            }
        }

        // CSlime2 is a damage volume the original game uses as a physical
        // presence indicator for the microphone stage area.  The user wants
        // it to act as an E-key trigger that starts the microphone script.
        for trigger in &mut game_objects.script_triggers {
            if trigger.name == "cslime2" {
                trigger.actions = vec![scripted_sequence::ScriptCommand::StartScript {
                    script_name: "microphone".to_string(),
                }];
            }
        }

        game_objects.triggers = crate::triggers::TriggerFactory::new(
            file_stem,
            &dat_file,
            &trigger_volumes,
            &bsp_submodels,
            scale,
        )
        .build();

        println!("=== Loaded {} scripts, {} BSP doors, {} triggers ===",
            game_objects.scripts.len(), game_objects.bsp_doors.len(), game_objects.triggers.len());


    }

    // Build entity cylinders for solid objects (barrels, headless enemies).
    // Cylinders match the model shape better than AABBs and produce smooth
    // radial sliding when the player walks into them.
    let entity_cylinders: Vec<geometry::EntityCylinder> = placed_abc_objects.iter()
        .filter(|o| o.type_name == "CBarrel" || o.type_name == "CHeadless")
        .filter_map(|o| {
            if o.mesh.vertices.is_empty() { return None; }
            // Compute XY center and Z bounds from world-space mesh vertices.
            let mut x_min = f32::MAX;
            let mut x_max = f32::MIN;
            let mut y_min = f32::MAX;
            let mut y_max = f32::MIN;
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            for v in &o.mesh.vertices {
                x_min = x_min.min(v.pos[0]);
                x_max = x_max.max(v.pos[0]);
                y_min = y_min.min(v.pos[1]);
                y_max = y_max.max(v.pos[1]);
                z_min = z_min.min(v.pos[2]);
                z_max = z_max.max(v.pos[2]);
            }
            let cx = (x_min + x_max) * 0.5;
            let cy = (y_min + y_max) * 0.5;
            // Radius = half the smaller horizontal extent (tighter fit).
            let half_x = (x_max - x_min) * 0.5;
            let half_y = (y_max - y_min) * 0.5;
            let radius = half_x.min(half_y);
            Some(geometry::EntityCylinder { center_x: cx, center_y: cy, radius, z_min, z_max })
        })
        .collect();
    log::info!("Built {} entity cylinders for collision", entity_cylinders.len());

    let triggers = crate::triggers::collect_triggers(&dat_file, &trigger_volumes, scale);
    if let Err(e) = crate::triggers::export_triggers_json(&triggers, &path) {
        warn!("Failed to export triggers JSON: {}", e);
    }

    Ok(LoadedWorld {
        lights: world_lights,
        collision_positions,
        collision_indices,
        game_objects,
        fog: level_fog,
        entity_cylinders,
        trigger_draw_groups,
        triggers,
    })
}

//******************************************************************/
//
// Print summary information about a DAT file without loading mesh data
//
//******************************************************************/

pub fn print_dat_info<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
    info!("Analyzing DAT file: {}", path.as_ref().display());

    let dat_file = dat::DatFile::read_from_file(&path)
        .map_err(|e| anyhow!("Failed to parse DAT file: {}", e))?;

    println!("\n{}", "=".repeat(60));
    println!("DAT FILE SUMMARY");
    println!("{}", "=".repeat(60));

    println!(
        "\nVersion: {} (KISS Psycho Circus)",
        dat_file.header.version
    );
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

    let extractor = dat_mesh::MeshExtractor::new(&dat_file);
    let meshes = extractor.extract_all_worlds();
    let stats = dat_mesh::MeshStats::from_meshes(&meshes);

    println!("\n--- MESH STATISTICS ---");
    println!("  Total Vertices: {}", stats.total_vertices);
    println!("  Total Indices: {}", stats.total_indices);
    println!("  Total Triangles: {}", stats.total_triangles);
    println!("  Total Texture Groups: {}", stats.texture_count);

    println!("\n{}", "=".repeat(60));

    Ok(())
}

//******************************************************************/
//
// Spatially subdivide draw groups into smaller cells for tighter frustum culling.
//
//******************************************************************/

pub fn subdivide_draw_groups(
    vertices: &[Vertex],
    indices: &mut Vec<u32>,
    draw_groups: &mut Vec<DrawGroup>,
    cell_size: f32,
) {
    use rayon::prelude::*;

    // Each draw group produces a list of sub-groups (texture_index, cell_indices)
    let sub_results: Vec<Vec<(usize, Vec<u32>)>> = draw_groups
        .par_iter()
        .map(|group| {
            let start = group.first_index as usize;
            let end = (start + group.index_count as usize).min(indices.len());
            let group_indices = &indices[start..end];

            if group.index_count <= 36 {
                return vec![(group.texture_index, group_indices.to_vec())];
            }

            let mut cells: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();

            for tri in group_indices.chunks(3) {
                if tri.len() < 3 {
                    continue;
                }
                let v0 = vertices[tri[0] as usize].pos;
                let v1 = vertices[tri[1] as usize].pos;
                let v2 = vertices[tri[2] as usize].pos;
                let cx = (v0.x + v1.x + v2.x) / 3.0;
                let cy = (v0.y + v1.y + v2.y) / 3.0;
                let cz = (v0.z + v1.z + v2.z) / 3.0;

                let cell_x = (cx / cell_size).floor() as i32;
                let cell_y = (cy / cell_size).floor() as i32;
                let cell_z = (cz / cell_size).floor() as i32;

                cells
                    .entry((cell_x, cell_y, cell_z))
                    .or_default()
                    .extend_from_slice(tri);
            }

            cells
                .into_values()
                .map(|cell_indices| (group.texture_index, cell_indices))
                .collect()
        })
        .collect();

    // Flatten results sequentially to build final index/group buffers
    let old_group_count = draw_groups.len();
    let mut new_indices: Vec<u32> = Vec::with_capacity(indices.len());
    let mut new_groups: Vec<DrawGroup> = Vec::new();

    for sub_groups in sub_results {
        for (texture_index, cell_indices) in sub_groups {
            let first_index = new_indices.len() as u32;
            let index_count = cell_indices.len() as u32;
            new_indices.extend_from_slice(&cell_indices);
            new_groups.push(DrawGroup {
                texture_index,
                first_index,
                index_count,
                vertex_offset: 0,
                model_matrix: None,
                sky_opacity: None,
                sky_draw_layer: 0,
            });
        }
    }

    *indices = new_indices;
    *draw_groups = new_groups;
    println!(
        "  Spatial subdivision: {} groups -> {} sub-groups (cell_size={:.0})",
        old_group_count,
        draw_groups.len(),
        cell_size
    );
}