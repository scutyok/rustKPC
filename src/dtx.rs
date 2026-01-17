//DTX Texture Reader for KISS
//
//supports reading DTX textures from KISS Psycho Circus (only lithtech 1.5 game)

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Result, Seek};
use std::path::Path;

// DTX version constants (stored as signed but read as unsigned)
pub const DTX_VERSION_LT1: u32 = 0xFFFFFFFE;   // -2 as u32
pub const DTX_VERSION_LT15: u32 = 0xFFFFFFFD;  // -3 as u32
pub const DTX_VERSION_LT2: u32 = 0xFFFFFFFB;   // -5 as u32

// Bytes per pixel types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BppType {
    Bpp8Palette = 0,
    Bpp8 = 1,
    Bpp16 = 2,
    Bpp32 = 3,
    S3tcDxt1 = 4,
    S3tcDxt3 = 5,
    S3tcDxt5 = 6,
    Bpp32Palette = 7,
}

impl BppType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(BppType::Bpp8Palette),
            1 => Some(BppType::Bpp8),
            2 => Some(BppType::Bpp16),
            3 => Some(BppType::Bpp32),
            4 => Some(BppType::S3tcDxt1),
            5 => Some(BppType::S3tcDxt3),
            6 => Some(BppType::S3tcDxt5),
            7 => Some(BppType::Bpp32Palette),
            _ => None,
        }
    }
}

/// DTX texture file
#[derive(Debug)]
pub struct DtxFile {
    /// Resource type (usually 0)
    pub resource_type: u32,
    /// DTX version (-2 for LT1, -3 for LT1.5, -5 for LT2)
    pub version: u32,
    /// Texture width in pixels
    pub width: u16,
    /// Texture height in pixels
    pub height: u16,
    /// Number of mipmaps
    pub mipmap_count: u16,
    /// Number of sections
    pub section_count: u16,
    /// Texture flags
    pub flags: u32,
    /// User-defined flags
    pub user_flags: u32,
    
    // Extra data (version >= LT1.5)
    pub texture_group: u8,
    pub mipmaps_to_use: u8,
    pub bytes_per_pixel: u8,
    pub mipmap_offset: u8,
    pub texture_priority: u8,
    pub detail_scale: f32,
    pub detail_angle: i16,
    
    // Command string for LT1.5/LT2
    pub command_string: String,
    
    /// RGBA pixel data (converted to 8-bit RGBA)
    pub pixels: Vec<u8>,
}

impl Default for DtxFile {
    fn default() -> Self {
        Self {
            resource_type: 0,
            version: DTX_VERSION_LT1,
            width: 0,
            height: 0,
            mipmap_count: 0,
            section_count: 0,
            flags: 0,
            user_flags: 0,
            texture_group: 0,
            mipmaps_to_use: 0,
            bytes_per_pixel: 0,
            mipmap_offset: 0,
            texture_priority: 0,
            detail_scale: 0.0,
            detail_angle: 0,
            command_string: String::new(),
            pixels: Vec::new(),
        }
    }
}

impl DtxFile {
    /// Read a DTX file from disk
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let result = Self::read(&mut reader);
        // Debug: print first texture's version
        if let Ok(ref dtx) = result {
            static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                println!("DTX Debug: version=0x{:08X}, width={}, height={}, bpp={}", 
                    dtx.version, dtx.width, dtx.height, dtx.bytes_per_pixel);
            }
        }
        result
    }

    /// Read DTX from a reader
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let mut dtx = DtxFile::default();

        // Read header (20 bytes for basic header)
        dtx.resource_type = reader.read_u32::<LittleEndian>()?;
        dtx.version = reader.read_u32::<LittleEndian>()?;
        dtx.width = reader.read_u16::<LittleEndian>()?;
        dtx.height = reader.read_u16::<LittleEndian>()?;
        dtx.mipmap_count = reader.read_u16::<LittleEndian>()?;
        dtx.section_count = reader.read_u16::<LittleEndian>()?;
        dtx.flags = reader.read_u32::<LittleEndian>()?;
        dtx.user_flags = reader.read_u32::<LittleEndian>()?;

        // Read extra data for non-LT1 versions
        if dtx.version != DTX_VERSION_LT1 {
            dtx.texture_group = reader.read_u8()?;
            dtx.mipmaps_to_use = reader.read_u8()?;
            dtx.bytes_per_pixel = reader.read_u8()?;
            dtx.mipmap_offset = reader.read_u8()?;
            let _mipmap_tex_coord_offset = reader.read_u8()?; // Missing byte!
            dtx.texture_priority = reader.read_u8()?;
            dtx.detail_scale = reader.read_f32::<LittleEndian>()?;
            dtx.detail_angle = reader.read_i16::<LittleEndian>()?;

            // Read command string (128 bytes, null-terminated)
            let mut cmd_buf = [0u8; 128];
            reader.read_exact(&mut cmd_buf)?;
            dtx.command_string = cmd_buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as char)
                .collect();
        }

        // Determine BPP type
        // LT1 and LT1.5 always use 8-bit palette
        let bpp_type = if dtx.version == DTX_VERSION_LT1 || dtx.version == DTX_VERSION_LT15 {
            BppType::Bpp8Palette
        } else {
            BppType::from_u8(dtx.bytes_per_pixel).unwrap_or(BppType::Bpp8Palette)
        };

        // Read pixel data based on format
        match bpp_type {
            BppType::Bpp8Palette => {
                dtx.pixels = Self::read_8bit_palette(reader, dtx.width, dtx.height)?;
            }
            BppType::Bpp32Palette => {
                dtx.pixels = Self::read_32bit_palette(reader, dtx.width, dtx.height)?;
            }
            BppType::Bpp32 => {
                dtx.pixels = Self::read_32bit_texture(reader, dtx.width, dtx.height)?;
            }
            BppType::S3tcDxt1 => {
                dtx.pixels = Self::read_dxt1(reader, dtx.width, dtx.height)?;
            }
            BppType::S3tcDxt3 => {
                dtx.pixels = Self::read_dxt3(reader, dtx.width, dtx.height)?;
            }
            BppType::S3tcDxt5 => {
                dtx.pixels = Self::read_dxt5(reader, dtx.width, dtx.height)?;
            }
            BppType::Bpp8 | BppType::Bpp16 => {
                // Fallback to palette for unsupported formats
                log::warn!("Unsupported BPP type {:?}, attempting palette fallback", bpp_type);
                dtx.pixels = Self::read_8bit_palette(reader, dtx.width, dtx.height)?;
            }
        }

        Ok(dtx)
    }

    /// Read 8-bit paletted texture (LT1 format)
    fn read_8bit_palette<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        // Read palette header (2 unknown u32s)
        let _palette_header_1 = reader.read_u32::<LittleEndian>()?;
        let _palette_header_2 = reader.read_u32::<LittleEndian>()?;

        // Read 256-color palette as ARGB (per Godot reference), output as RGBA
        let mut palette = Vec::with_capacity(256);
        for _ in 0..256 {
            let a = reader.read_u8()?;
            let r = reader.read_u8()?;
            let g = reader.read_u8()?;
            let b = reader.read_u8()?;
            // Store as RGBA, force alpha to 255 for opaque rendering
            palette.push((r, g, b, if a == 0 { 255 } else { a }));
        }

        // Read indexed pixel data
        let pixel_count = (width as usize) * (height as usize);
        let mut indices = vec![0u8; pixel_count];
        reader.read_exact(&mut indices)?;

        // Convert indexed to RGBA
        let mut rgba = Vec::with_capacity(pixel_count * 4);
        for &idx in &indices {
            let (r, g, b, a) = palette[idx as usize];
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(a);
        }

        Ok(rgba)
    }

    /// Read 32-bit paletted texture (LT1.5+ format)
    fn read_32bit_palette<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        let pixel_count = (width as usize) * (height as usize);
        
        // Read indexed pixel data first
        let mut indices = vec![0u8; pixel_count];
        reader.read_exact(&mut indices)?;

        // Read section header (12 bytes + 4 bytes length)
        let mut section_header = [0u8; 12];
        reader.read_exact(&mut section_header)?;
        let _section_length = reader.read_u32::<LittleEndian>()?;

        // Read 256-color palette (packed 32-bit ARGB)
        let mut palette = Vec::with_capacity(256);
        for _ in 0..256 {
            let packed = reader.read_u32::<LittleEndian>()?;
            let a = ((packed >> 24) & 0xFF) as u8;
            let r = ((packed >> 16) & 0xFF) as u8;
            let g = ((packed >> 8) & 0xFF) as u8;
            let b = (packed & 0xFF) as u8;
            // Store as RGBA, force alpha to 255
            palette.push((r, g, b, if a == 0 { 255 } else { a }));
        }

        // Convert indexed to RGBA
        let mut rgba = Vec::with_capacity(pixel_count * 4);
        for &idx in &indices {
            let (r, g, b, a) = palette[idx as usize];
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(a);
        }

        Ok(rgba)
    }

    /// Read 32-bit uncompressed texture
    fn read_32bit_texture<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        let pixel_count = (width as usize) * (height as usize);
        let mut rgba = Vec::with_capacity(pixel_count * 4);

        for _ in 0..pixel_count {
            let packed = reader.read_u32::<LittleEndian>()?;
            // File stores ARGB, we output RGBA
            let a = ((packed >> 24) & 0xFF) as u8;
            let r = ((packed >> 16) & 0xFF) as u8;
            let g = ((packed >> 8) & 0xFF) as u8;
            let b = (packed & 0xFF) as u8;
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(if a == 0 { 255 } else { a });
        }

        Ok(rgba)
    }

    /// Read DXT1 compressed texture
    fn read_dxt1<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        let block_width = ((width + 3) / 4) as usize;
        let block_height = ((height + 3) / 4) as usize;
        let block_count = block_width * block_height;
        
        // DXT1 is 8 bytes per 4x4 block
        let mut compressed = vec![0u8; block_count * 8];
        reader.read_exact(&mut compressed)?;

        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        Self::decompress_dxt1(&compressed, &mut rgba, width as usize, height as usize);
        Ok(rgba)
    }

    /// Read DXT3 compressed texture
    fn read_dxt3<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        let block_width = ((width + 3) / 4) as usize;
        let block_height = ((height + 3) / 4) as usize;
        let block_count = block_width * block_height;
        
        // DXT3 is 16 bytes per 4x4 block
        let mut compressed = vec![0u8; block_count * 16];
        reader.read_exact(&mut compressed)?;

        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        Self::decompress_dxt3(&compressed, &mut rgba, width as usize, height as usize);
        Ok(rgba)
    }

    /// Read DXT5 compressed texture
    fn read_dxt5<R: Read>(reader: &mut R, width: u16, height: u16) -> Result<Vec<u8>> {
        let block_width = ((width + 3) / 4) as usize;
        let block_height = ((height + 3) / 4) as usize;
        let block_count = block_width * block_height;
        
        // DXT5 is 16 bytes per 4x4 block
        let mut compressed = vec![0u8; block_count * 16];
        reader.read_exact(&mut compressed)?;

        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        Self::decompress_dxt5(&compressed, &mut rgba, width as usize, height as usize);
        Ok(rgba)
    }

    /// Decompress DXT1 block data
    fn decompress_dxt1(data: &[u8], rgba: &mut [u8], width: usize, height: usize) {
        let block_width = (width + 3) / 4;
        let block_height = (height + 3) / 4;

        for by in 0..block_height {
            for bx in 0..block_width {
                let block_idx = (by * block_width + bx) * 8;
                if block_idx + 8 > data.len() {
                    continue;
                }
                
                // Read two 16-bit colors
                let c0 = u16::from_le_bytes([data[block_idx], data[block_idx + 1]]);
                let c1 = u16::from_le_bytes([data[block_idx + 2], data[block_idx + 3]]);

                // Expand to RGBA
                let colors = Self::expand_dxt1_colors(c0, c1);

                // Read 4-byte lookup table (2 bits per pixel)
                let lookup = u32::from_le_bytes([
                    data[block_idx + 4],
                    data[block_idx + 5],
                    data[block_idx + 6],
                    data[block_idx + 7],
                ]);

                // Decode 4x4 block
                for y in 0..4 {
                    for x in 0..4 {
                        let px = bx * 4 + x;
                        let py = by * 4 + y;
                        
                        if px >= width || py >= height {
                            continue;
                        }

                        let idx = (y * 4 + x) * 2;
                        let color_idx = ((lookup >> idx) & 0x3) as usize;
                        let color = &colors[color_idx];

                        let dest_idx = (py * width + px) * 4;
                        rgba[dest_idx] = color[0];
                        rgba[dest_idx + 1] = color[1];
                        rgba[dest_idx + 2] = color[2];
                        rgba[dest_idx + 3] = color[3];
                    }
                }
            }
        }
    }

    /// Expand DXT1 color palette
    fn expand_dxt1_colors(c0: u16, c1: u16) -> [[u8; 4]; 4] {
        let r0 = ((c0 >> 11) & 0x1F) as u8;
        let g0 = ((c0 >> 5) & 0x3F) as u8;
        let b0 = (c0 & 0x1F) as u8;

        let r1 = ((c1 >> 11) & 0x1F) as u8;
        let g1 = ((c1 >> 5) & 0x3F) as u8;
        let b1 = (c1 & 0x1F) as u8;

        // Expand 5/6 bits to 8 bits
        let r0 = (r0 << 3) | (r0 >> 2);
        let g0 = (g0 << 2) | (g0 >> 4);
        let b0 = (b0 << 3) | (b0 >> 2);

        let r1 = (r1 << 3) | (r1 >> 2);
        let g1 = (g1 << 2) | (g1 >> 4);
        let b1 = (b1 << 3) | (b1 >> 2);

        let mut colors = [[0u8; 4]; 4];
        colors[0] = [r0, g0, b0, 255];
        colors[1] = [r1, g1, b1, 255];

        if c0 > c1 {
            // 4-color mode
            colors[2] = [
                ((2 * r0 as u16 + r1 as u16) / 3) as u8,
                ((2 * g0 as u16 + g1 as u16) / 3) as u8,
                ((2 * b0 as u16 + b1 as u16) / 3) as u8,
                255,
            ];
            colors[3] = [
                ((r0 as u16 + 2 * r1 as u16) / 3) as u8,
                ((g0 as u16 + 2 * g1 as u16) / 3) as u8,
                ((b0 as u16 + 2 * b1 as u16) / 3) as u8,
                255,
            ];
        } else {
            // 3-color + transparent mode
            colors[2] = [
                ((r0 as u16 + r1 as u16) / 2) as u8,
                ((g0 as u16 + g1 as u16) / 2) as u8,
                ((b0 as u16 + b1 as u16) / 2) as u8,
                255,
            ];
            colors[3] = [0, 0, 0, 0]; // Transparent
        }

        colors
    }

    /// Decompress DXT3 block data
    fn decompress_dxt3(data: &[u8], rgba: &mut [u8], width: usize, height: usize) {
        let block_width = (width + 3) / 4;
        let block_height = (height + 3) / 4;

        for by in 0..block_height {
            for bx in 0..block_width {
                let block_idx = (by * block_width + bx) * 16;
                if block_idx + 16 > data.len() {
                    continue;
                }

                // First 8 bytes are explicit alpha values (4 bits per pixel)
                let alpha_data = &data[block_idx..block_idx + 8];
                
                // Next 8 bytes are DXT1 color block
                let color_data = &data[block_idx + 8..block_idx + 16];
                
                let c0 = u16::from_le_bytes([color_data[0], color_data[1]]);
                let c1 = u16::from_le_bytes([color_data[2], color_data[3]]);
                let colors = Self::expand_dxt1_colors(c0, c1);

                let lookup = u32::from_le_bytes([
                    color_data[4], color_data[5], color_data[6], color_data[7]
                ]);

                for y in 0..4 {
                    for x in 0..4 {
                        let px = bx * 4 + x;
                        let py = by * 4 + y;
                        
                        if px >= width || py >= height {
                            continue;
                        }

                        let idx = y * 4 + x;
                        let color_idx = ((lookup >> (idx * 2)) & 0x3) as usize;
                        let color = &colors[color_idx];

                        // Get alpha from explicit alpha data
                        let alpha_byte = alpha_data[y * 2 + x / 2];
                        let alpha = if x % 2 == 0 {
                            (alpha_byte & 0x0F) << 4
                        } else {
                            alpha_byte & 0xF0
                        };

                        let dest_idx = (py * width + px) * 4;
                        rgba[dest_idx] = color[0];
                        rgba[dest_idx + 1] = color[1];
                        rgba[dest_idx + 2] = color[2];
                        rgba[dest_idx + 3] = alpha;
                    }
                }
            }
        }
    }

    /// Decompress DXT5 block data
    fn decompress_dxt5(data: &[u8], rgba: &mut [u8], width: usize, height: usize) {
        let block_width = (width + 3) / 4;
        let block_height = (height + 3) / 4;

        for by in 0..block_height {
            for bx in 0..block_width {
                let block_idx = (by * block_width + bx) * 16;
                if block_idx + 16 > data.len() {
                    continue;
                }

                // First 2 bytes are alpha endpoints
                let a0 = data[block_idx];
                let a1 = data[block_idx + 1];
                
                // Next 6 bytes are alpha indices (3 bits per pixel)
                let alpha_indices = &data[block_idx + 2..block_idx + 8];
                
                // Calculate alpha palette
                let alpha_palette = Self::expand_dxt5_alpha(a0, a1);
                
                // Next 8 bytes are DXT1 color block
                let color_data = &data[block_idx + 8..block_idx + 16];
                
                let c0 = u16::from_le_bytes([color_data[0], color_data[1]]);
                let c1 = u16::from_le_bytes([color_data[2], color_data[3]]);
                let colors = Self::expand_dxt1_colors(c0, c1);

                let lookup = u32::from_le_bytes([
                    color_data[4], color_data[5], color_data[6], color_data[7]
                ]);

                // Decode alpha indices from 6 bytes (48 bits for 16 pixels, 3 bits each)
                let alpha_bits = u64::from_le_bytes([
                    alpha_indices[0], alpha_indices[1], alpha_indices[2],
                    alpha_indices[3], alpha_indices[4], alpha_indices[5],
                    0, 0
                ]);

                for y in 0..4 {
                    for x in 0..4 {
                        let px = bx * 4 + x;
                        let py = by * 4 + y;
                        
                        if px >= width || py >= height {
                            continue;
                        }

                        let idx = y * 4 + x;
                        let color_idx = ((lookup >> (idx * 2)) & 0x3) as usize;
                        let color = &colors[color_idx];

                        // Get alpha index (3 bits)
                        let alpha_idx = ((alpha_bits >> (idx * 3)) & 0x7) as usize;
                        let alpha = alpha_palette[alpha_idx];

                        let dest_idx = (py * width + px) * 4;
                        rgba[dest_idx] = color[0];
                        rgba[dest_idx + 1] = color[1];
                        rgba[dest_idx + 2] = color[2];
                        rgba[dest_idx + 3] = alpha;
                    }
                }
            }
        }
    }

    /// Expand DXT5 alpha palette
    fn expand_dxt5_alpha(a0: u8, a1: u8) -> [u8; 8] {
        let mut palette = [0u8; 8];
        palette[0] = a0;
        palette[1] = a1;

        if a0 > a1 {
            // 8-alpha mode
            for i in 2..8 {
                palette[i] = ((((8 - i) as u16 * a0 as u16) + ((i - 1) as u16 * a1 as u16)) / 7) as u8;
            }
        } else {
            // 6-alpha + transparent mode
            for i in 2..6 {
                palette[i] = ((((6 - i) as u16 * a0 as u16) + ((i - 1) as u16 * a1 as u16)) / 5) as u8;
            }
            palette[6] = 0;
            palette[7] = 255;
        }

        palette
    }
}

/// Search for a DTX file in a texture folder by name
pub fn find_dtx_file(textures_root: &Path, texture_name: &str) -> Option<std::path::PathBuf> {
    // Clean up texture name (remove path separators, make uppercase)
    let clean_name = texture_name
        .replace(['\\', '/'], "")
        .to_uppercase();
    
    // Add .DTX extension if not present
    let dtx_name = if clean_name.ends_with(".DTX") {
        clean_name
    } else {
        format!("{}.DTX", clean_name)
    };

    // Search recursively
    search_dtx_recursive(textures_root, &dtx_name)
}

fn search_dtx_recursive(dir: &Path, target_name: &str) -> Option<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = search_dtx_recursive(&path, target_name) {
                    return Some(found);
                }
            } else if let Some(name) = path.file_name() {
                if name.to_string_lossy().to_uppercase() == target_name {
                    return Some(path);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpp_type_from_u8() {
        assert_eq!(BppType::from_u8(0), Some(BppType::Bpp8Palette));
        assert_eq!(BppType::from_u8(4), Some(BppType::S3tcDxt1));
        assert_eq!(BppType::from_u8(99), None);
    }
}
