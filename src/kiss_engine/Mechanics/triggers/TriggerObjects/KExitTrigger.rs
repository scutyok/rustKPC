use crate::dat::{PropertyValue, WorldObject};
use crate::scripted_sequence::ScriptCommand;
use crate::triggers::{TriggerActivation, TriggerDef, TriggerSource};

use super::util::{build_aabb_from_dat_object, transform_dat_position_to_runtime};

/// Builds the runtime representation of a `CExitTrigger` DAT object.
pub struct KExitTrigger<'a> {
    scale: f32,
    trigger_volumes: &'a [(String, [f32; 3], [f32; 3])],
}

impl<'a> KExitTrigger<'a> {
    pub fn new(scale: f32, trigger_volumes: &'a [(String, [f32; 3], [f32; 3])]) -> Self {
        Self { scale, trigger_volumes }
    }

    pub fn build_for_object(&self, object: &WorldObject) -> Option<TriggerDef> {
        let name = match object.get_property("Name") {
            Some(PropertyValue::String(name)) => name.to_lowercase(),
            _ => object.type_name.to_lowercase(),
        };
        let next_world = match object.get_property("NextWorld") {
            Some(PropertyValue::String(world)) => world.trim().to_owned(),
            _ => return None,
        };
        let start_point = match object.get_property("StartPointName") {
            Some(PropertyValue::String(start)) => start.trim().to_owned(),
            _ => String::new(),
        };
        let position = transform_dat_position_to_runtime(object.get_position()?, self.scale);
        let volume = build_aabb_from_dat_object(object, position, self.trigger_volumes);

        Some(TriggerDef {
            name,
            source: TriggerSource::DatObject,
            activation: TriggerActivation::Touch,
            once: true,
            enabled: true,
            activated: false,
            volume: Some(volume),
            center: None,
            use_radius: 0.0,
            actions: vec![ScriptCommand::TransitionLevel { next_world, start_point }],
        })
    }
}
