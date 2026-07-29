//******************************************************************/
//
// Decodes the real per-texel baked lightmap data stored in a .dat
// WorldBsp's `lightmap_data` blob.
//
//
//   For each polygon IN ORDER (not just polygons with a nonzero
//   lightmap_width/height from the polygon header -- those fields are
//   NOT texel counts and are unrelated to this data):
//     - if that polygon's surface does NOT have flag bit 0x80 set
//       (SURFACE_HAS_LIGHTMAP), it contributes NO bytes at all here --
//       skip it entirely and move to the next polygon.
//     - otherwise read:
//         width:  u8
//         height: u8
//         width*height texels, each a little-endian u16 in RGB565:
//           r = (packed & 0xF800) >> 8
//           g = (packed & 0x07E0) >> 3
//           b = (packed & 0x001F) << 3
//
// No RLE, no compression... After many failed tries...
//
//******************************************************************/

use crate::dat::WorldBsp;

/// Surface flag bit that marks a surface as having real baked per-texel
/// lightmap data in the WorldBsp's `lightmap_data` blob.
pub const SURFACE_HAS_LIGHTMAP: u32 = 0x0080;

#[derive(Debug, Clone)]
pub struct DecodedLightmap {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB texels, 0.0..=1.0
    pub texels: Vec<[f32; 3]>,
}

impl DecodedLightmap {
    /// Bilinear sample at normalized coordinates (0.0..=1.0, 0.0..=1.0).
    pub fn sample_normalized(&self, u: f32, v: f32) -> [f32; 3] {
        if self.width == 0 || self.height == 0 || self.texels.is_empty() {
            return [1.0, 1.0, 1.0];
        }
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let fx = u * (self.width.saturating_sub(1)) as f32;
        let fy = v * (self.height.saturating_sub(1)) as f32;

        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let get = |x: usize, y: usize| -> [f32; 3] { self.texels[y * self.width + x] };

        let c00 = get(x0, y0);
        let c10 = get(x1, y0);
        let c01 = get(x0, y1);
        let c11 = get(x1, y1);

        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let top = c00[i] * (1.0 - tx) + c10[i] * tx;
            let bot = c01[i] * (1.0 - tx) + c11[i] * tx;
            out[i] = top * (1.0 - ty) + bot * ty;
        }
        out
    }
}

fn rgb565_to_rgb888(packed: u16) -> [u8; 3] {
    let r = ((packed & 0xF800) >> 8) as u8;
    let g = ((packed & 0x07E0) >> 3) as u8;
    let b = ((packed & 0x001F) << 3) as u8;
    [r, g, b]
}

/// Decode all polygon lightmaps for a WorldBsp. Index i in the returned
/// Vec corresponds to `bsp.polygons[i]`; `None` means that polygon's
/// surface doesn't have the lightmap flag set (no data present for it at
/// all -- callers should fall back to the polygon's own per-vertex
/// `disk_vert.dummy` color in that case, which IS legitimate baked-light
/// data, just coarser/per-vertex instead of per-texel).
pub fn decode_world_lightmaps(wm: &WorldBsp) -> Vec<Option<DecodedLightmap>> {
    let mut result = vec![None; wm.polygons.len()];

    if wm.lightmap_data.is_empty() {
        return result;
    }

    let mut cursor = 0usize;
    let data = &wm.lightmap_data;

    for (poly_idx, poly) in wm.polygons.iter().enumerate() {
        let surface = match wm.surfaces.get(poly.surface_index as usize) {
            Some(s) => s,
            None => continue,
        };

        if surface.flags & SURFACE_HAS_LIGHTMAP == 0 {
            continue;
        }

        if cursor + 2 > data.len() {
            break; // truncated/unexpected end -- stop decoding further polys
        }

        let width = data[cursor] as usize;
        let height = data[cursor + 1] as usize;
        cursor += 2;

        let texel_count = width * height;
        let bytes_needed = texel_count * 2;

        if texel_count == 0 {
            continue;
        }

        if cursor + bytes_needed > data.len() {
            break; // truncated -- stop decoding further polys
        }

        let mut texels = Vec::with_capacity(texel_count);
        for i in 0..texel_count {
            let off = cursor + i * 2;
            let packed = u16::from_le_bytes([data[off], data[off + 1]]);
            let rgb = rgb565_to_rgb888(packed);
            texels.push([
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]);
        }
        cursor += bytes_needed;

        result[poly_idx] = Some(DecodedLightmap { width, height, texels });
    }

    result
}