use crate::dat::WorldObject;
use crate::util::geometry::AABB;
use crate::util::math::Vector3;

pub fn transform_dat_position_to_runtime(position: Vector3, scale: f32) -> [f32; 3] {
    [position.x * scale, position.z * scale, position.y * scale]
}

pub fn build_aabb_from_dat_object(
    object: &WorldObject,
    position: [f32; 3],
    trigger_volumes: &[(String, [f32; 3], [f32; 3])],
) -> AABB {
    // A CExitTrigger's entity `Name` is not necessarily the BSP brush name.
    // For example, R1M1B calls the entity "Gubba" but its volume is named
    // "CExitTrigger0".  Match all stable identifiers before using the small
    // point-volume fallback.
    let mut names = vec![object.type_name.as_str()];
    for property_name in ["Name", "GroupName"] {
        if let Some(crate::dat::PropertyValue::String(name)) = object.get_property(property_name) {
            names.push(name);
        }
    }

    if let Some((_, min, max)) = trigger_volumes.iter().find(|(volume_name, _, _)| {
        names.iter().any(|name| volume_name.eq_ignore_ascii_case(name))
            // Brushes conventionally append an index to the CExitTrigger
            // class name (for example CExitTrigger0).
            || volume_name
                .to_ascii_lowercase()
                .starts_with(&object.type_name.to_ascii_lowercase())
    }) {
        return AABB { min: (*min).into(), max: (*max).into() };
    }

    build_aabb_from_name(&object.type_name, position, trigger_volumes)
}

pub fn build_aabb_from_name(
    name: &str,
    position: [f32; 3],
    trigger_volumes: &[(String, [f32; 3], [f32; 3])],
) -> AABB {
    if let Some((_, min, max)) = trigger_volumes.iter().find(|(volume_name, _, _)| volume_name.eq_ignore_ascii_case(name)) {
        return AABB { min: (*min).into(), max: (*max).into() };
    }
    let half_extent = 0.35;
    AABB {
        min: [position[0] - half_extent, position[1] - half_extent, position[2] - half_extent].into(),
        max: [position[0] + half_extent, position[1] + half_extent, position[2] + half_extent].into(),
    }
}
