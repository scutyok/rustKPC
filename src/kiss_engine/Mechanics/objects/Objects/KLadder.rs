//******************************************************************/
//
// CLadder — climbable ladder zone.
//
// While the player is inside the volume:
//   * Gravity is zeroed.
//  * W/S move the player vertically.
//
// No mesh / draw group — purely a physics zone.
// Volume half-extents are read from the `Dims` DAT property.
//
//******************************************************************/

use crate::dat::{PropertyValue, WorldObject};

#[derive(Debug, Clone)]
pub struct LadderObject {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl LadderObject {
    pub fn contains(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }
}

//******************************************************************/
//
// DAT construction 
//
//******************************************************************/

pub fn parse(pos: [f32; 3], props: Option<&WorldObject>, scale: f32) -> LadderObject {
    let ext = props
        .and_then(|o| o.get_property("Dims"))
        .and_then(|v| {
            if let PropertyValue::Vector(vec) = v {
                Some([vec.x * scale * 0.5, vec.z * scale * 0.5, vec.y * scale * 0.5])
            } else {
                None
            }
        })
        .unwrap_or([0.5, 0.5, 1.5]);
    LadderObject {
        min: [pos[0] - ext[0], pos[1] - ext[1], pos[2] - ext[2]],
        max: [pos[0] + ext[0], pos[1] + ext[1], pos[2] + ext[2]],
    }
}
