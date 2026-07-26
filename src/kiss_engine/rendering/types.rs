#![allow(dead_code)]

use std::hash::{Hash, Hasher};
use std::mem::size_of;

use thiserror::Error;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use crate::util::math::*;

// Type aliases 
pub type Vec2 = Vector2;
pub type Vec3 = Vector3;
pub type Mat4 = cgmath::Matrix4<f32>;

// Constants
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
pub const PORTABILITY_MACOS_VERSION: vulkanalia::Version = vulkanalia::Version::new(1, 3, 216);
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const WALK_SPEED: f32 = 6.0;
pub const FLY_SPEED: f32 = 10.0;
pub const MOUSE_SENSITIVITY: f32 = 0.1;
pub const MAX_LIGHTS: usize = 128;

// Re-export shadow constants and types from the shadows module
pub use crate::ShadowObj::{GpuShadowCaster, MAX_SHADOW_CASTERS};

// Error types
#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

// Vertex
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec3,
    pub tex_coord: Vec2,
    pub normal: Vec3,
}

impl Vertex {
    pub fn new(pos: Vec3, color: Vec3, tex_coord: Vec2, normal: Vec3) -> Self {
        Self { pos, color, tex_coord, normal }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();
        let normal = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>() + size_of::<Vec2>()) as u32)
            .build();
        [pos, color, tex_coord, normal]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord && self.normal == other.normal
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.x.to_bits().hash(state);
        self.pos.y.to_bits().hash(state);
        self.pos.z.to_bits().hash(state);

        self.color.x.to_bits().hash(state);
        self.color.y.to_bits().hash(state);
        self.color.z.to_bits().hash(state);

        self.tex_coord.x.to_bits().hash(state);
        self.tex_coord.y.to_bits().hash(state);

        self.normal.x.to_bits().hash(state);
        self.normal.y.to_bits().hash(state);
        self.normal.z.to_bits().hash(state);
    }
}

// UBO structs
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct GpuLight {
    // xyz = position, w = radius² (for squared-distance early-out)
    pub position_radius_sq: [f32; 4],
    // rgb = color * intensity (pre-multiplied), w = 1/radius (for attenuation)
    pub color_intensity: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone)]
pub struct LightingUBO {
    pub camera_pos: [f32; 4],
    pub ambient: [f32; 4],
    // rgb = fog colour, a = unused
    pub fog_color: [f32; 4],
    // x = fog near distance, y = fog far distance, z = 1.0 if enabled / 0.0 if disabled, w = unused
    pub fog_params: [f32; 4],
    pub light_count: u32,
    pub _pad: [u32; 3],
    pub lights: [GpuLight; MAX_LIGHTS],
    pub shadow_count: u32,
    pub _pad2: [u32; 3],
    pub shadow_casters: [GpuShadowCaster; MAX_SHADOW_CASTERS],
}

impl Default for LightingUBO {
    fn default() -> Self {
        Self {
            camera_pos: [0.0; 4],
            ambient: [0.15, 0.15, 0.15, 0.0],
            fog_color: [0.05, 0.05, 0.08, 0.0],
            fog_params: [5.0, 22.0, 1.0, 0.0],
            light_count: 0,
            _pad: [0; 3],
            lights: [GpuLight {
                position_radius_sq: [0.0; 4],
                color_intensity: [0.0; 4],
            }; MAX_LIGHTS],
            shadow_count: 0,
            _pad2: [0; 3],
            shadow_casters: [GpuShadowCaster {
                position_radius: [0.0; 4],
            }; MAX_SHADOW_CASTERS],
        }
    }
}

// Draw groups and level textures
#[derive(Clone, Debug, Default)]
pub struct DrawGroup {
    pub texture_index: usize,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
    // Per-object model matrix applied as push constant before this draw call.
    // None = identity (world-geometry default). Set by game_objects system for
    // animated objects (doors, fans) and cleared for destroyed objects.
    pub model_matrix: Option<[[f32; 4]; 4]>,
    /// Negative values select the unlit sky shader path.  `None` is regular
    /// world geometry; sky layers use their own per-model opacity.
    pub sky_opacity: Option<f32>,
    /// Draw order among sky groups only (ignored for non-sky geometry).
    /// 0 = base sky dome (drawn first, furthest back), 1 = translucent
    /// cloud overlay (drawn over the dome), 2 = opaque foreground skybox
    /// models such as mountains/buildings (drawn last, on top of clouds).
    pub sky_draw_layer: u8,
}

#[derive(Clone, Debug, Default)]
pub struct LevelTexture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

// AppData - Vulkan handles and associated properties
#[derive(Clone, Debug, Default)]
pub struct AppData {
    // Debug
    pub messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    pub surface: vk::SurfaceKHR,
    // Vsync
    pub vsync: bool,
    // Physical Device / Logical Device
    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    // Swapchain
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_image_count: u32,
    // Pipeline
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    // Sky pipeline (depth write disabled, drawn behind world)
    pub sky_pipeline: vk::Pipeline,
    // Framebuffers
    pub framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    pub command_pool: vk::CommandPool,
    // Color
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
    // Depth
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    // Texture (fallback/default)
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    // Multiple textures for level geometry
    pub level_textures: Vec<LevelTexture>,
    // Draw groups
    pub draw_groups: Vec<DrawGroup>,
    // Sky rendering
    /// Index into draw_groups where sky groups begin (sky groups run from here to end)
    pub sky_draw_group_start: usize,
    /// Sky world model translation (in scaled world coords)
    pub sky_translation: [f32; 3],
    /// Bounds of the loaded sky model vertices, before render-time projection.
    pub sky_bounds_min: [f32; 3],
    pub sky_bounds_max: [f32; 3],
    /// Center of the main world model bounding box (in renderer coords)
    pub map_center: [f32; 3],
    /// Preferred texture subfolder derived from current DAT path (e.g. "R2M1")
    pub texture_prefix: String,
    // Model
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    // Buffers
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    // Debug line rendering
    pub debug_line_vertex_buffer: vk::Buffer,
    pub debug_line_vertex_buffer_memory: vk::DeviceMemory,
    pub debug_line_vertex_capacity: usize,
    pub debug_line_pipeline: vk::Pipeline,
    pub debug_line_pipeline_layout: vk::PipelineLayout,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    /// Persistently mapped pointers for uniform buffers (avoids map/unmap per frame)
    pub uniform_buffers_mapped: Vec<*mut std::ffi::c_void>,
    // Lighting UBO (binding 2)
    pub light_uniform_buffers: Vec<vk::Buffer>,
    pub light_uniform_buffers_memory: Vec<vk::DeviceMemory>,
    /// Persistently mapped pointers for light uniform buffers
    pub light_uniform_buffers_mapped: Vec<*mut std::ffi::c_void>,
    // Descriptors
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Sync Objects
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

// Queue Family Indices
#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, _properties) in properties.iter().enumerate() {
            if instance
                .get_physical_device_surface_support_khr(physical_device, index as u32, data.surface)?
            {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow::anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

// Swapchain Support
#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}