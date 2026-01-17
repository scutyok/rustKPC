//mesh conversion for KISS Psycho Circus DAT files
//
//converts the BSP world data from DAT files into Vulkan-compatible
//vertex and index buffers for rendering

use crate::dat::{DatFile, Vector2, Vector3, WorldBsp, WorldPoly, WorldSurface};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A vertex ready for Vulkan rendering
#[derive(Clone, Copy, Debug)]
pub struct DatVertex {
    /// Position in 3D space
    pub pos: [f32; 3],
    /// Vertex color (from surface)
    pub color: [f32; 3],
    /// Texture coordinates
    pub tex_coord: [f32; 2],
    /// Normal vector
    pub normal: [f32; 3],
}

impl DatVertex {
    pub fn new(pos: Vector3, color: Vector3, tex_coord: Vector2, normal: Vector3) -> Self {
        Self {
            pos: pos.to_array(),
            color: [color.x / 255.0, color.y / 255.0, color.z / 255.0],
            tex_coord: tex_coord.to_array(),
            normal: normal.to_array(),
        }
    }
}

impl PartialEq for DatVertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.color == other.color
            && self.tex_coord == other.tex_coord
            && self.normal == other.normal
    }
}

impl Eq for DatVertex {}

impl Hash for DatVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
    }
}

/// A mesh group sharing the same texture
#[derive(Debug, Default)]
pub struct TexturedMesh {
    /// Texture name (path to .dtx file)
    pub texture_name: String,
    /// Texture width in pixels (for UV scaling, 0 = unknown)
    pub texture_width: u32,
    /// Texture height in pixels (for UV scaling, 0 = unknown)  
    pub texture_height: u32,
    /// Vertices for this texture group
    pub vertices: Vec<DatVertex>,
    /// Indices for this texture group
    pub indices: Vec<u32>,
}

/// Complete mesh data extracted from a world model
#[derive(Debug, Default)]
pub struct WorldMesh {
    /// Name of the world model
    pub name: String,
    /// Bounding box minimum
    pub min_bounds: [f32; 3],
    /// Bounding box maximum
    pub max_bounds: [f32; 3],
    /// All vertices (combined from all texture groups)
    pub vertices: Vec<DatVertex>,
    /// All indices (combined from all texture groups)
    pub indices: Vec<u32>,
    /// Meshes grouped by texture
    pub textured_meshes: Vec<TexturedMesh>,
}

/// Surface flags for filtering polygons
pub mod surface_flags {
    pub const INVISIBLE: u32 = 0x0001;
    pub const NO_SUBDIVIDE: u32 = 0x0002;
    pub const FULLBRIGHT: u32 = 0x0004;
    pub const LIGHTMAP: u32 = 0x0008;
    pub const SKYPORTAL: u32 = 0x0010;
    pub const SKY: u32 = 0x0040;
    pub const PORTAL: u32 = 0x0400;
    pub const TEXTURE_ANIM: u32 = 0x0800;
}

/// Extracts renderable mesh data from a DAT file
pub struct MeshExtractor<'a> {
    dat: &'a DatFile,
    /// Scale factor for the world (Lithtech units to desired units)
    pub scale: f32,
    /// Whether to skip invisible surfaces
    pub skip_invisible: bool,
    /// Whether to skip sky surfaces
    pub skip_sky: bool,
    /// Whether to flip the winding order
    pub flip_winding: bool,
}

impl<'a> MeshExtractor<'a> {
    pub fn new(dat: &'a DatFile) -> Self {
        Self {
            dat,
            scale: 1.0,
            skip_invisible: true,
            skip_sky: false,
            flip_winding: false,
        }
    }

    /// Set the scale factor
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set whether to skip invisible surfaces
    pub fn with_skip_invisible(mut self, skip: bool) -> Self {
        self.skip_invisible = skip;
        self
    }

    /// Set whether to skip sky surfaces
    pub fn with_skip_sky(mut self, skip: bool) -> Self {
        self.skip_sky = skip;
        self
    }

    /// Set whether to flip winding order
    pub fn with_flip_winding(mut self, flip: bool) -> Self {
        self.flip_winding = flip;
        self
    }

    /// Extract mesh from the main world model (index 0)
    pub fn extract_main_world(&self) -> Option<WorldMesh> {
        self.dat.get_main_world().map(|bsp| self.extract_world_bsp(bsp))
    }

    /// Extract meshes from all world models
    pub fn extract_all_worlds(&self) -> Vec<WorldMesh> {
        self.dat
            .world_models
            .iter()
            .map(|bsp| self.extract_world_bsp(bsp))
            .collect()
    }

    /// Extract mesh from a specific world model by index
    pub fn extract_world_by_index(&self, index: usize) -> Option<WorldMesh> {
        self.dat
            .world_models
            .get(index)
            .map(|bsp| self.extract_world_bsp(bsp))
    }

    /// Extract mesh data from a single WorldBsp
    fn extract_world_bsp(&self, bsp: &WorldBsp) -> WorldMesh {
        let mut mesh = WorldMesh {
            name: bsp.world_name.clone(),
            min_bounds: [
                bsp.min_box.x * self.scale,
                bsp.min_box.y * self.scale,
                bsp.min_box.z * self.scale,
            ],
            max_bounds: [
                bsp.max_box.x * self.scale,
                bsp.max_box.y * self.scale,
                bsp.max_box.z * self.scale,
            ],
            ..Default::default()
        };

        // Group polygons by texture
        let mut texture_groups: HashMap<u16, Vec<usize>> = HashMap::new();

        for (poly_idx, polygon) in bsp.polygons.iter().enumerate() {
            if polygon.surface_index as usize >= bsp.surfaces.len() {
                continue;
            }

            let surface = &bsp.surfaces[polygon.surface_index as usize];

            // Filter surfaces
            if self.skip_invisible && (surface.flags & surface_flags::INVISIBLE) != 0 {
                continue;
            }
            if self.skip_sky
                && ((surface.flags & surface_flags::SKY) != 0
                    || (surface.flags & surface_flags::SKYPORTAL) != 0)
            {
                continue;
            }

            texture_groups
                .entry(surface.texture_index)
                .or_default()
                .push(poly_idx);
        }

        // Process each texture group
        for (texture_idx, poly_indices) in texture_groups {
            let texture_name = if (texture_idx as usize) < bsp.textures.len() {
                bsp.textures[texture_idx as usize].name.clone()
            } else {
                format!("texture_{}", texture_idx)
            };

            let mut textured_mesh = TexturedMesh {
                texture_name,
                ..Default::default()
            };

            let mut unique_vertices: HashMap<DatVertex, u32> = HashMap::new();

            for poly_idx in poly_indices {
                let polygon = &bsp.polygons[poly_idx];
                let surface = &bsp.surfaces[polygon.surface_index as usize];

                self.triangulate_polygon(
                    bsp,
                    polygon,
                    surface,
                    &mut textured_mesh.vertices,
                    &mut textured_mesh.indices,
                    &mut unique_vertices,
                );
            }

            if !textured_mesh.vertices.is_empty() {
                mesh.textured_meshes.push(textured_mesh);
            }
        }

        // Combine all vertices and indices into single buffers
        let mut vertex_offset = 0u32;
        for textured_mesh in &mesh.textured_meshes {
            for vertex in &textured_mesh.vertices {
                mesh.vertices.push(*vertex);
            }
            for index in &textured_mesh.indices {
                mesh.indices.push(index + vertex_offset);
            }
            vertex_offset += textured_mesh.vertices.len() as u32;
        }

        log::info!(
            "Extracted world '{}': {} vertices, {} indices, {} texture groups",
            mesh.name,
            mesh.vertices.len(),
            mesh.indices.len(),
            mesh.textured_meshes.len()
        );

        mesh
    }

    /// Triangulate a polygon and add to the mesh buffers
    fn triangulate_polygon(
        &self,
        bsp: &WorldBsp,
        polygon: &WorldPoly,
        surface: &WorldSurface,
        vertices: &mut Vec<DatVertex>,
        indices: &mut Vec<u32>,
        unique_vertices: &mut HashMap<DatVertex, u32>,
    ) {
        if polygon.disk_verts.len() < 3 {
            return;
        }

        // Get plane normal for the polygon
        let normal = if (surface.plane_index as usize) < bsp.planes.len() {
            bsp.planes[surface.plane_index as usize].normal
        } else {
            // Calculate normal from first 3 vertices
            self.calculate_polygon_normal(bsp, polygon)
        };

        // Get vertex positions
        let mut poly_vertices: Vec<(Vector3, Vector2)> = Vec::with_capacity(polygon.disk_verts.len());

        for disk_vert in &polygon.disk_verts {
            let vert_idx = disk_vert.vertex_index as usize;
            if vert_idx >= bsp.points.len() {
                continue;
            }

            let pos = bsp.points[vert_idx];
            // swap Y/Z
            // don't negate X here; the mesh scale will handle mirroring
            let scaled_pos = Vector3::new(
                pos.x * self.scale,   
                pos.z * self.scale,   // Lithtech Z -> Vulkan Y 
                pos.y * self.scale,   // Lithtech Y -> Vulkan Z
            );

            // calculate texture coordinates using surface UV vectors
            let uv = self.calculate_uv(&pos, surface);

            poly_vertices.push((scaled_pos, uv));
        }

        if poly_vertices.len() < 3 {
            return;
        }

        // Fan triangulation from first vertex
        let v0 = &poly_vertices[0];
        // Use white color - let textures provide color, not surface colour
        // Surface colour is often not what we want for textured rendering
        let color = Vector3::new(255.0, 255.0, 255.0);

        for i in 1..poly_vertices.len() - 1 {
            let v1 = &poly_vertices[i];
            let v2 = &poly_vertices[i + 1];

            let (a, b, c) = if self.flip_winding {
                (
                    DatVertex::new(v0.0, color, v0.1, normal),
                    DatVertex::new(v2.0, color, v2.1, normal),
                    DatVertex::new(v1.0, color, v1.1, normal),
                )
            } else {
                (
                    DatVertex::new(v0.0, color, v0.1, normal),
                    DatVertex::new(v1.0, color, v1.1, normal),
                    DatVertex::new(v2.0, color, v2.1, normal),
                )
            };

            // Add vertices with deduplication
            for vertex in [a, b, c] {
                if let Some(&existing_idx) = unique_vertices.get(&vertex) {
                    indices.push(existing_idx);
                } else {
                    let new_idx = vertices.len() as u32;
                    unique_vertices.insert(vertex, new_idx);
                    vertices.push(vertex);
                    indices.push(new_idx);
                }
            }
        }
    }

    /// Calculate UV coordinates for a vertex position
    fn calculate_uv(&self, pos: &Vector3, surface: &WorldSurface) -> Vector2 {
        // Lithtech texture mapping:
        // U = pos dot P + O.x
        // V = pos dot Q + O.y
        let u = pos.dot(&surface.uv_p) + surface.uv_o.x;
        let v = pos.dot(&surface.uv_q) + surface.uv_o.y;
        Vector2::new(u, v)
    }

    /// Calculate polygon normal from vertices
    fn calculate_polygon_normal(&self, bsp: &WorldBsp, polygon: &WorldPoly) -> Vector3 {
        if polygon.disk_verts.len() < 3 {
            return Vector3::new(0.0, 0.0, 1.0);
        }

        let idx0 = polygon.disk_verts[0].vertex_index as usize;
        let idx1 = polygon.disk_verts[1].vertex_index as usize;
        let idx2 = polygon.disk_verts[2].vertex_index as usize;

        if idx0 >= bsp.points.len() || idx1 >= bsp.points.len() || idx2 >= bsp.points.len() {
            return Vector3::new(0.0, 0.0, 1.0);
        }

        let v0 = bsp.points[idx0];
        let v1 = bsp.points[idx1];
        let v2 = bsp.points[idx2];

        let edge1 = v1.sub(&v0);
        let edge2 = v2.sub(&v0);

        edge1.cross(&edge2).normalized()
    }
}

/// Convert a WorldMesh to simple vertex/index arrays for Vulkan
/// Returns (vertices, indices) where each vertex is [pos.x, pos.y, pos.z, color.r, color.g, color.b, u, v]
pub fn to_interleaved_arrays(mesh: &WorldMesh) -> (Vec<f32>, Vec<u32>) {
    let mut vertex_data = Vec::with_capacity(mesh.vertices.len() * 11);

    for v in &mesh.vertices {
        // Position
        vertex_data.push(v.pos[0]);
        vertex_data.push(v.pos[1]);
        vertex_data.push(v.pos[2]);
        // Color
        vertex_data.push(v.color[0]);
        vertex_data.push(v.color[1]);
        vertex_data.push(v.color[2]);
        // Texture coords
        vertex_data.push(v.tex_coord[0]);
        vertex_data.push(v.tex_coord[1]);
        // Normal
        vertex_data.push(v.normal[0]);
        vertex_data.push(v.normal[1]);
        vertex_data.push(v.normal[2]);
    }

    (vertex_data, mesh.indices.clone())
}

/// Statistics about extracted mesh data
#[derive(Debug, Default)]
pub struct MeshStats {
    pub total_vertices: usize,
    pub total_indices: usize,
    pub total_triangles: usize,
    pub texture_count: usize,
    pub world_model_count: usize,
}

impl MeshStats {
    pub fn from_meshes(meshes: &[WorldMesh]) -> Self {
        let mut stats = Self::default();
        stats.world_model_count = meshes.len();

        for mesh in meshes {
            stats.total_vertices += mesh.vertices.len();
            stats.total_indices += mesh.indices.len();
            stats.total_triangles += mesh.indices.len() / 3;
            stats.texture_count += mesh.textured_meshes.len();
        }

        stats
    }
}
