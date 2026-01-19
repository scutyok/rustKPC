//DAT file structures and parsing for KISS Psycho Circus: The Nightmare Child
//
//file format version: 127 (BSP/World version)
//this module handles reading .DAT world files used by the Lithtech 1.5 engine (only kiss)

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

/// DAT version constant for KISS Psycho Circus
pub const DAT_VERSION_PSYCHO: u32 = 127;

#[derive(Error, Debug)]
pub enum DatError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unsupported DAT version: {0} (expected {DAT_VERSION_PSYCHO} for KISS Psycho Circus)")]
    UnsupportedVersion(u32),

    #[error("Invalid string length: {0}")]
    InvalidStringLength(i16),

    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, DatError>;

// 
// Basic Types
//

#[derive(Debug, Clone, Copy, Default)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn dot(&self, other: &Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn normalized(&self) -> Vector3 {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            Vector3 {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            *self
        }
    }

    pub fn sub(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn to_array(&self) -> [f32; 2] {
        [self.x, self.y]
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

// 
// World Header
// 

#[derive(Debug, Default)]
pub struct WorldHeader {
    pub version: u32,
    pub object_data_pos: u32,
    pub render_data_pos: u32,
}

// 
// World Info
// 

#[derive(Debug, Default)]
pub struct WorldInfo {
    pub properties: String,
    pub light_map_grid_size: f32,
    pub padding: [u32; 8],
}

// 
// World Texture
// 

#[derive(Debug, Clone, Default)]
pub struct WorldTexture {
    pub name: String,
}

// 
// World Plane
// 

#[derive(Debug, Clone, Default)]
pub struct WorldPlane {
    pub normal: Vector3,
    pub distance: f32,
}

// 
// World Surface - Contains UV mapping and texture info
// 

#[derive(Debug, Clone, Default)]
pub struct WorldSurface {
    /// O vector for texture mapping (origin)
    pub uv_o: Vector3,
    /// P vector for texture mapping (U direction)
    pub uv_p: Vector3,
    /// Q vector for texture mapping (V direction)
    pub uv_q: Vector3,
    /// Unknown vectors (possibly for lightmapping)
    pub uv4: Vector3,
    pub uv5: Vector3,
    /// Vertex color (RGB)
    pub colour: Vector3,
    /// Index into texture array
    pub texture_index: u16,
    /// Index into plane array
    pub plane_index: u32,
    /// Surface flags (visibility, collision, etc.)
    pub flags: u32,
    pub unknown: [u8; 4],
    /// Whether this surface has shader effects
    pub use_effects: u8,
    pub effect_name: String,
    pub effect_param: String,
    pub texture_flags: u16,
    pub unknown_flag: u16,
}

impl WorldSurface {
    /// Calculate UV coordinates for a vertex position
    pub fn calculate_uv(&self, pos: &Vector3) -> Vector2 {
        let u = pos.dot(&self.uv_p) + self.uv_o.x;
        let v = pos.dot(&self.uv_q) + self.uv_o.y;
        Vector2::new(u, v)
    }
}

// 
// World Leaf - BSP leaf nodes containing visibility data
// 

#[derive(Debug, Clone, Default)]
pub struct LeafData {
    pub portal_id: i16,
    pub size: u16,
    pub contents: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct WorldLeaf {
    pub count: u16,
    pub index: Option<u16>,
    pub data: Vec<LeafData>,
    pub polygon_count: u16,
    pub polygon_data: Vec<u8>,
    pub unknown: f32,
}

// 
// World Polygon - Individual polygon with vertex indices
// 

#[derive(Debug, Clone, Default)]
pub struct DiskVert {
    /// Index into the points array
    pub vertex_index: i16,
    pub dummy: [u8; 3],
}

#[derive(Debug, Clone, Default)]
pub struct WorldPoly {
    pub lightmap_width: i16,
    pub lightmap_height: i16,
    pub unknown1: u32,
    pub unknown2: u32,
    /// Index into surfaces array for texture/UV info
    pub surface_index: u32,
    /// Vertex indices for this polygon
    pub disk_verts: Vec<DiskVert>,
}

// 
// World Node - BSP tree nodes
// 

#[derive(Debug, Clone, Default)]
pub struct WorldNode {
    pub unknown_intro: u32,
    pub poly_index: u32,
    pub leaf_index: i16,
    pub node_index_1: i32,
    pub node_index_2: i32,
    pub unknown_quat: Quaternion,
}

// 
// World User Portal
// 

#[derive(Debug, Clone, Default)]
pub struct WorldUserPortal {
    pub name: String,
    pub unknown1: u32,
    pub unknown_short: u16,
    pub center: Vector3,
    pub dims: Vector3,
}

// 
// PBlock Table - Collision/physics data
// 

#[derive(Debug, Clone, Default)]
pub struct PBlockContents {
    pub data: [u8; 6],
}

#[derive(Debug, Clone, Default)]
pub struct PBlock {
    pub size: i16,
    pub unknown: i16,
    pub contents: Vec<PBlockContents>,
}

#[derive(Debug, Clone, Default)]
pub struct PBlockTable {
    pub dim1: u32,
    pub dim2: u32,
    pub dim3: u32,
    pub vec1: Vector3,
    pub vec2: Vector3,
    pub blocks: Vec<PBlock>,
}

// 
// World BSP
// 

#[derive(Debug, Default)]
pub struct WorldBsp {
    pub info_flags: u32,
    pub world_name: String,
    pub next_position: u32,
    pub point_count: u32,
    pub plane_count: u32,
    pub surface_count: u32,
    pub user_portal_count: u32,
    pub poly_count: u32,
    pub leaf_count: u32,
    pub vert_count: u32,
    pub total_vis_list_size: u32,
    pub leaf_list_count: u32,
    pub node_count: u32,
    pub unknown: u32,
    pub min_box: Vector3,
    pub max_box: Vector3,
    pub world_translation: Vector3,
    pub texture_name_length: u32,
    pub texture_count: u32,

    /// Texture names used by this world model
    pub textures: Vec<WorldTexture>,
    /// Vertex counts per polygon
    pub vertex_counts: Vec<u16>,
    /// BSP leaves
    pub leaves: Vec<WorldLeaf>,
    /// Plane definitions
    pub planes: Vec<WorldPlane>,
    /// Surface definitions (textures, UVs)
    pub surfaces: Vec<WorldSurface>,
    /// Polygon definitions (vertex indices)
    pub polygons: Vec<WorldPoly>,
    /// BSP tree nodes
    pub nodes: Vec<WorldNode>,
    /// User portals (doors, etc.)
    pub user_portals: Vec<WorldUserPortal>,
    /// Vertex positions
    pub points: Vec<Vector3>,
    /// Collision data
    pub block_table: PBlockTable,
    pub root_node_index: u32,
    pub unknown_count: u32,
    /// Polygon center points
    pub polygon_centers: Vec<Vector3>,
    pub lightmap_data_size: u32,
    /// Raw lightmap data (RGB565 packed)
    pub lightmap_data: Vec<u8>,
    pub section_count: u32,
}

// 
// Object Property
// 

#[derive(Debug, Clone)]
pub enum PropertyValue {
    String(String),
    Vector(Vector3),
    Color(Vector3),
    Float(f32),
    Bool(u8),
    Flags(u32),
    LongInt(u32),
    Rotation(Quaternion),
    UnknownInt(u32),
}

#[derive(Debug, Clone)]
pub struct ObjectProperty {
    pub name: String,
    pub code: u8,
    pub flags: u32,
    pub data_length: u16,
    pub value: PropertyValue,
}

// 
// World Object - entities/objects placed in the world
// 

#[derive(Debug, Clone, Default)]
pub struct WorldObject {
    pub data_length: u16,
    pub type_name: String,
    pub properties: Vec<ObjectProperty>,
}

impl WorldObject {
    /// Get a property value by name
    pub fn get_property(&self, name: &str) -> Option<&PropertyValue> {
        self.properties
            .iter()
            .find(|p| p.name == name)
            .map(|p| &p.value)
    }

    /// Get position if this object has a Pos property
    pub fn get_position(&self) -> Option<Vector3> {
        match self.get_property("Pos") {
            Some(PropertyValue::Vector(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get rotation if this object has a Rotation property
    pub fn get_rotation(&self) -> Option<Quaternion> {
        match self.get_property("Rotation") {
            Some(PropertyValue::Rotation(q)) => Some(*q),
            _ => None,
        }
    }
}

// 
// Main DAT File Structure
// 

#[derive(Debug, Default)]
pub struct DatFile {
    pub header: WorldHeader,
    pub world_info: WorldInfo,
    pub objects: Vec<WorldObject>,
    pub world_models: Vec<WorldBsp>,
}

// 
// Reading Implementation
// 

impl DatFile {
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader)
    }

    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let mut dat = DatFile::default();

        // Read header
        dat.header = WorldHeader::read(reader)?;

        // Verify version
        if dat.header.version != DAT_VERSION_PSYCHO {
            return Err(DatError::UnsupportedVersion(dat.header.version));
        }

        log::info!(
            "DAT Version: {} (KISS Psycho Circus)",
            dat.header.version
        );
        log::debug!(
            "Object Data Position: 0x{:08X}",
            dat.header.object_data_pos
        );
        log::debug!(
            "Render Data Position: 0x{:08X}",
            dat.header.render_data_pos
        );

        // Read world info
        dat.world_info = WorldInfo::read(reader)?;
        log::debug!("World Properties: {}", dat.world_info.properties);

        // Read object data
        reader.seek(SeekFrom::Start(dat.header.object_data_pos as u64))?;
        let object_count = reader.read_u32::<LittleEndian>()?;
        log::info!("Object Count: {}", object_count);

        for i in 0..object_count {
            match WorldObject::read(reader) {
                Ok(obj) => dat.objects.push(obj),
                Err(e) => {
                    let pos = reader.stream_position().unwrap_or(0);
                    log::error!(
                        "Error reading object {} at position 0x{:08X}: {}",
                        i,
                        pos,
                        e
                    );
                    return Err(e);
                }
            }
        }

        log::info!("Successfully read {} objects", dat.objects.len());

        // World models come right after object data
        let mut root_model = read_world_model_entry(reader)?;
        root_model.world_name = "Root".to_string();
        dat.world_models.push(root_model);

        // Read additional world models
        let world_model_count = reader.read_u32::<LittleEndian>()?;
        log::info!("World Model Count: {} (+1 root)", world_model_count);

        for _i in 0..world_model_count {
            match read_world_model_entry(reader) {
                Ok(model) => {
                    dat.world_models.push(model);
                }
                Err(e) => {
                    let pos = reader.stream_position().unwrap_or(0);
                    log::error!(
                        "Error reading world model at position 0x{:08X}: {}",
                        pos,
                        e
                    );
                    return Err(e);
                }
            }
        }

        log::info!("Successfully read {} world models", dat.world_models.len());

        Ok(dat)
    }

    /// Get the main world model (index 0)
    pub fn get_main_world(&self) -> Option<&WorldBsp> {
        self.world_models.first()
    }

    /// Get all objects of a specific type
    pub fn get_objects_by_type(&self, type_name: &str) -> Vec<&WorldObject> {
        self.objects
            .iter()
            .filter(|o| o.type_name == type_name)
            .collect()
    }
}

impl WorldHeader {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            version: reader.read_u32::<LittleEndian>()?,
            object_data_pos: reader.read_u32::<LittleEndian>()?,
            render_data_pos: reader.read_u32::<LittleEndian>()?,
        })
    }
}

impl WorldInfo {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let properties = read_string_u32(reader)?;
        let light_map_grid_size = reader.read_f32::<LittleEndian>()?;

        // Read 8 padding ints
        let mut padding = [0u32; 8];
        for p in &mut padding {
            *p = reader.read_u32::<LittleEndian>()?;
        }

        Ok(Self {
            properties,
            light_map_grid_size,
            padding,
        })
    }
}

impl WorldTexture {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        // Read null-terminated string
        let mut name = String::new();
        loop {
            let byte = reader.read_u8()?;
            if byte == 0 {
                break;
            }
            name.push(byte as char);
        }
        Ok(Self { name })
    }
}

impl WorldPlane {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            normal: read_vector3(reader)?,
            distance: reader.read_f32::<LittleEndian>()?,
        })
    }
}

impl LeafData {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let portal_id = reader.read_i16::<LittleEndian>()?;
        let size = reader.read_u16::<LittleEndian>()?;
        let mut contents = vec![0u8; size as usize];
        reader.read_exact(&mut contents)?;

        Ok(Self {
            portal_id,
            size,
            contents,
        })
    }
}

impl WorldLeaf {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let count = reader.read_u16::<LittleEndian>()?;

        let (index, data) = if count == 0xFFFF {
            (Some(reader.read_u16::<LittleEndian>()?), Vec::new())
        } else {
            let mut data = Vec::with_capacity(count as usize);
            for _ in 0..count {
                data.push(LeafData::read(reader)?);
            }
            (None, data)
        };

        let polygon_count = reader.read_u16::<LittleEndian>()?;
        let mut polygon_data = vec![0u8; polygon_count as usize * 4];
        reader.read_exact(&mut polygon_data)?;

        let unknown = reader.read_f32::<LittleEndian>()?;

        Ok(Self {
            count,
            index,
            data,
            polygon_count,
            polygon_data,
            unknown,
        })
    }
}

impl WorldSurface {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let uv_o = read_vector3(reader)?;
        let uv_p = read_vector3(reader)?;
        let uv_q = read_vector3(reader)?;
        let uv4 = read_vector3(reader)?;
        let uv5 = read_vector3(reader)?;
        let colour = read_vector3_clamped(reader)?;

        let texture_index = reader.read_u16::<LittleEndian>()?;
        let plane_index = reader.read_u32::<LittleEndian>()?;
        let flags = reader.read_u32::<LittleEndian>()?;

        let mut unknown = [0u8; 4];
        reader.read_exact(&mut unknown)?;

        let use_effects = reader.read_u8()?;

        let (effect_name, effect_param) = if use_effects > 0 {
            (read_string_u16(reader)?, read_string_u16(reader)?)
        } else {
            (String::new(), String::new())
        };

        let texture_flags = reader.read_u16::<LittleEndian>()?;

        // KISS Psycho Circus specific: extra unknown short
        let unknown_flag = reader.read_u16::<LittleEndian>()?;

        Ok(Self {
            uv_o,
            uv_p,
            uv_q,
            uv4,
            uv5,
            colour,
            texture_index,
            plane_index,
            flags,
            unknown,
            use_effects,
            effect_name,
            effect_param,
            texture_flags,
            unknown_flag,
        })
    }
}

impl DiskVert {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let vertex_index = reader.read_i16::<LittleEndian>()?;
        let mut dummy = [0u8; 3];
        reader.read_exact(&mut dummy)?;

        Ok(Self { vertex_index, dummy })
    }
}

impl WorldPoly {
    fn read<R: Read>(reader: &mut R, vert_count: u16) -> Result<Self> {
        let lightmap_width = reader.read_i16::<LittleEndian>()?;
        let lightmap_height = reader.read_i16::<LittleEndian>()?;
        let unknown1 = reader.read_u32::<LittleEndian>()?;
        let unknown2 = reader.read_u32::<LittleEndian>()?;
        let surface_index = reader.read_u32::<LittleEndian>()?;

        let mut disk_verts = Vec::with_capacity(vert_count as usize);
        for _ in 0..vert_count {
            disk_verts.push(DiskVert::read(reader)?);
        }

        Ok(Self {
            lightmap_width,
            lightmap_height,
            unknown1,
            unknown2,
            surface_index,
            disk_verts,
        })
    }
}

impl WorldNode {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let unknown_intro = reader.read_u32::<LittleEndian>()?;
        let poly_index = reader.read_u32::<LittleEndian>()?;
        let leaf_index = reader.read_i16::<LittleEndian>()?;
        let node_index_1 = reader.read_i32::<LittleEndian>()?;
        let node_index_2 = reader.read_i32::<LittleEndian>()?;
        let unknown_quat = read_quaternion(reader)?;

        Ok(Self {
            unknown_intro,
            poly_index,
            leaf_index,
            node_index_1,
            node_index_2,
            unknown_quat,
        })
    }
}

impl WorldUserPortal {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let name = read_string_u16(reader)?;
        let unknown1 = reader.read_u32::<LittleEndian>()?;
        let unknown_short = reader.read_u16::<LittleEndian>()?;
        let center = read_vector3(reader)?;
        let dims = read_vector3(reader)?;

        Ok(Self {
            name,
            unknown1,
            unknown_short,
            center,
            dims,
        })
    }
}

impl PBlock {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let size = reader.read_i16::<LittleEndian>()?;
        let unknown = reader.read_i16::<LittleEndian>()?;

        let mut contents = Vec::with_capacity(size as usize);
        for _ in 0..size {
            let mut data = [0u8; 6];
            reader.read_exact(&mut data)?;
            contents.push(PBlockContents { data });
        }

        Ok(Self {
            size,
            unknown,
            contents,
        })
    }
}

impl PBlockTable {
    fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let dim1 = reader.read_u32::<LittleEndian>()?;
        let dim2 = reader.read_u32::<LittleEndian>()?;
        let dim3 = reader.read_u32::<LittleEndian>()?;

        let total_size = dim1 * dim2 * dim3;

        let vec1 = read_vector3(reader)?;
        let vec2 = read_vector3(reader)?;

        let mut blocks = Vec::with_capacity(total_size as usize);
        for _ in 0..total_size {
            blocks.push(PBlock::read(reader)?);
        }

        Ok(Self {
            dim1,
            dim2,
            dim3,
            vec1,
            vec2,
            blocks,
        })
    }
}

impl WorldBsp {
    fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let info_flags = reader.read_u32::<LittleEndian>()?;
        let world_name = read_string_u16(reader)?;
        let next_position = reader.read_u32::<LittleEndian>()?;

        let point_count = reader.read_u32::<LittleEndian>()?;
        let plane_count = reader.read_u32::<LittleEndian>()?;
        let surface_count = reader.read_u32::<LittleEndian>()?;
        let user_portal_count = reader.read_u32::<LittleEndian>()?;
        let poly_count = reader.read_u32::<LittleEndian>()?;
        let leaf_count = reader.read_u32::<LittleEndian>()?;
        let vert_count = reader.read_u32::<LittleEndian>()?;
        let total_vis_list_size = reader.read_u32::<LittleEndian>()?;
        let leaf_list_count = reader.read_u32::<LittleEndian>()?;
        let node_count = reader.read_u32::<LittleEndian>()?;
        let unknown = reader.read_u32::<LittleEndian>()?;

        let min_box = read_vector3(reader)?;
        let max_box = read_vector3(reader)?;
        let world_translation = read_vector3(reader)?;

        let texture_name_length = reader.read_u32::<LittleEndian>()?;
        let texture_count = reader.read_u32::<LittleEndian>()?;

        // Read textures (null-terminated strings)
        let mut textures = Vec::with_capacity(texture_count as usize);
        for _ in 0..texture_count {
            textures.push(WorldTexture::read(reader)?);
        }

        // Read vertex counts for polygons (2 bytes per polygon)
        let mut vertex_counts = Vec::with_capacity(poly_count as usize);
        for _ in 0..poly_count {
            let count = reader.read_u8()? as u16;
            let extra = reader.read_u8()? as u16;
            vertex_counts.push(count + extra);
        }

        // Read leaves
        let mut leaves = Vec::with_capacity(leaf_count as usize);
        for _ in 0..leaf_count {
            leaves.push(WorldLeaf::read(reader)?);
        }

        // Read planes
        let mut planes = Vec::with_capacity(plane_count as usize);
        for _ in 0..plane_count {
            planes.push(WorldPlane::read(reader)?);
        }

        // Read surfaces
        let mut surfaces = Vec::with_capacity(surface_count as usize);
        for _ in 0..surface_count {
            surfaces.push(WorldSurface::read(reader)?);
        }

        // Read polygons
        let mut polygons = Vec::with_capacity(poly_count as usize);
        for i in 0..poly_count as usize {
            polygons.push(WorldPoly::read(reader, vertex_counts[i])?);
        }

        // Read nodes
        let mut nodes = Vec::with_capacity(node_count as usize);
        for _ in 0..node_count {
            nodes.push(WorldNode::read(reader)?);
        }

        // Read user portals
        let mut user_portals = Vec::with_capacity(user_portal_count as usize);
        for _ in 0..user_portal_count {
            user_portals.push(WorldUserPortal::read(reader)?);
        }

        // Read points (vertex positions)
        let mut points = Vec::with_capacity(point_count as usize);
        for _ in 0..point_count {
            points.push(read_vector3(reader)?);
        }

        // Read PBlock table
        let block_table = PBlockTable::read(reader)?;

        // Root node index
        let root_node_index = reader.read_u32::<LittleEndian>()?;

        // Unknown count and polygon centers
        let unknown_count = reader.read_u32::<LittleEndian>()?;

        let mut polygon_centers = Vec::with_capacity(poly_count as usize);
        for _ in 0..poly_count {
            polygon_centers.push(read_vector3(reader)?);
        }

        // Lightmap data
        let lightmap_data_size = reader.read_u32::<LittleEndian>()?;
        let mut lightmap_data = vec![0u8; lightmap_data_size as usize];
        if lightmap_data_size > 0 {
            reader.read_exact(&mut lightmap_data)?;
        }

        let section_count = 0;

        Ok(Self {
            info_flags,
            world_name,
            next_position,
            point_count,
            plane_count,
            surface_count,
            user_portal_count,
            poly_count,
            leaf_count,
            vert_count,
            total_vis_list_size,
            leaf_list_count,
            node_count,
            unknown,
            min_box,
            max_box,
            world_translation,
            texture_name_length,
            texture_count,
            textures,
            vertex_counts,
            leaves,
            planes,
            surfaces,
            polygons,
            nodes,
            user_portals,
            points,
            block_table,
            root_node_index,
            unknown_count,
            polygon_centers,
            lightmap_data_size,
            lightmap_data,
            section_count,
        })
    }
}

impl ObjectProperty {
    fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let name = read_string_u16(reader)?;
        let code = reader.read_u8()?;
        let flags = reader.read_u32::<LittleEndian>()?;
        let data_length = reader.read_u16::<LittleEndian>()?;

        let value = match code {
            0 => PropertyValue::String(read_string_u16(reader)?),
            1 => PropertyValue::Vector(read_vector3(reader)?),
            2 => PropertyValue::Color(read_vector3(reader)?),
            3 => PropertyValue::Float(reader.read_f32::<LittleEndian>()?),
            4 => PropertyValue::Flags(reader.read_u32::<LittleEndian>()?),
            5 => PropertyValue::Bool(reader.read_u8()?),
            6 => PropertyValue::LongInt(reader.read_u32::<LittleEndian>()?),
            7 => PropertyValue::Rotation(read_quaternion(reader)?),
            9 => PropertyValue::UnknownInt(reader.read_u32::<LittleEndian>()?),
            _ => {
                // Unknown property type - skip based on data_length
                log::warn!(
                    "Unknown property type {} for '{}', skipping {} bytes",
                    code,
                    name,
                    data_length
                );
                let mut skip_data = vec![0u8; data_length as usize];
                reader.read_exact(&mut skip_data)?;
                PropertyValue::String(format!("<unknown type {}>", code))
            }
        };

        Ok(Self {
            name,
            code,
            flags,
            data_length,
            value,
        })
    }
}

impl WorldObject {
    fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let data_length = reader.read_u16::<LittleEndian>()?;
        let type_name = read_string_u16(reader)?;
        let property_count = reader.read_u32::<LittleEndian>()?;

        let mut properties = Vec::with_capacity(property_count as usize);
        for j in 0..property_count {
            match ObjectProperty::read(reader) {
                Ok(prop) => properties.push(prop),
                Err(e) => {
                    let pos = reader.stream_position().unwrap_or(0);
                    log::error!(
                        "Error reading property {} of object '{}' at 0x{:08X}: {}",
                        j,
                        type_name,
                        pos,
                        e
                    );
                    return Err(e);
                }
            }
        }

        Ok(Self {
            data_length,
            type_name,
            properties,
        })
    }
}

// 
// Helper Functions
// 

/// Read a world model entry (with the next_pos header and padding)
fn read_world_model_entry<R: Read + Seek>(reader: &mut R) -> Result<WorldBsp> {
    let _next_world_model_pos = reader.read_u32::<LittleEndian>()?;

    // Skip 32 bytes of padding/unknown data
    let mut padding = [0u8; 32];
    reader.read_exact(&mut padding)?;

    WorldBsp::read(reader)
}

fn read_string_u16<R: Read>(reader: &mut R) -> Result<String> {
    let length = reader.read_i16::<LittleEndian>()?;
    if length < 0 {
        return Err(DatError::InvalidStringLength(length));
    }

    let mut buffer = vec![0u8; length as usize];
    reader.read_exact(&mut buffer)?;

    Ok(String::from_utf8_lossy(&buffer).to_string())
}

fn read_string_u32<R: Read>(reader: &mut R) -> Result<String> {
    let length = reader.read_u32::<LittleEndian>()?;

    let mut buffer = vec![0u8; length as usize];
    reader.read_exact(&mut buffer)?;

    Ok(String::from_utf8_lossy(&buffer).to_string())
}

fn read_vector3<R: Read>(reader: &mut R) -> Result<Vector3> {
    Ok(Vector3 {
        x: reader.read_f32::<LittleEndian>()?,
        y: reader.read_f32::<LittleEndian>()?,
        z: reader.read_f32::<LittleEndian>()?,
    })
}

fn read_vector3_clamped<R: Read>(reader: &mut R) -> Result<Vector3> {
    let x = reader.read_f32::<LittleEndian>()?.min(255.0);
    let y = reader.read_f32::<LittleEndian>()?.min(255.0);
    let z = reader.read_f32::<LittleEndian>()?.min(255.0);

    Ok(Vector3 { x, y, z })
}

fn read_quaternion<R: Read>(reader: &mut R) -> Result<Quaternion> {
    Ok(Quaternion {
        w: reader.read_f32::<LittleEndian>()?,
        x: reader.read_f32::<LittleEndian>()?,
        y: reader.read_f32::<LittleEndian>()?,
        z: reader.read_f32::<LittleEndian>()?,
    })
}
