//! egui renderer for Vulkan using vulkanalia
//!
//! This module provides egui integration with the existing Vulkan renderer.

use anyhow::{anyhow, Result};
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;

/// Vertex format for egui rendering (matches shader input)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EguiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

/// egui Vulkan renderer
pub struct EguiRenderer {
    // Own render pass for egui (loads existing content, renders UI on top)
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    
    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,

    // Font texture
    font_image: vk::Image,
    font_memory: vk::DeviceMemory,
    font_view: vk::ImageView,
    font_sampler: vk::Sampler,

    // Descriptor
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // Dynamic buffers (resized as needed)
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_capacity: usize,

    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_capacity: usize,

    // Screen dimensions
    screen_width: f32,
    screen_height: f32,
}

impl EguiRenderer {
    /// Create a new egui renderer
    pub unsafe fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        swapchain_image_views: &[vk::ImageView],
        swapchain_format: vk::Format,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        egui_ctx: &egui::Context,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // Create egui render pass (loads existing content, renders UI on top)
        let render_pass = create_egui_render_pass(device, swapchain_format)?;
        
        // Create framebuffers for egui render pass
        let framebuffers = create_egui_framebuffers(device, render_pass, swapchain_image_views, width, height)?;
        
        // Create descriptor set layout for font texture
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[sampler_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8); // 2 floats for screen size

        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[push_constant_range];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

        // Create pipeline
        let pipeline = create_egui_pipeline(device, render_pass, pipeline_layout)?;

        // Get font texture from egui
        let font_image_data = egui_ctx.fonts(|fonts| {
            let image = fonts.image();
            image
                .srgba_pixels(None)
                .flat_map(|c| [c.r(), c.g(), c.b(), c.a()])
                .collect::<Vec<u8>>()
        });

        let font_dimensions = egui_ctx.fonts(|fonts| {
            let image = fonts.image();
            (image.width() as u32, image.height() as u32)
        });

        // Create font texture
        let (font_image, font_memory, font_view) = create_texture(
            instance,
            device,
            physical_device,
            command_pool,
            graphics_queue,
            &font_image_data,
            font_dimensions.0,
            font_dimensions.1,
        )?;

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let font_sampler = device.create_sampler(&sampler_info, None)?;

        // Create descriptor pool
        let pool_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);

        let pool_sizes = &[pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(1);

        let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

        // Allocate descriptor set
        let layouts = &[descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(layouts);

        let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

        // Update descriptor set with font texture
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(font_view)
            .sampler(font_sampler);

        let image_infos = &[image_info];
        let descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_infos);

        device.update_descriptor_sets(&[descriptor_write], &[] as &[vk::CopyDescriptorSet]);

        // Create initial vertex buffer
        let initial_vertex_capacity = 4096;
        let (vertex_buffer, vertex_memory) = create_buffer(
            instance,
            device,
            physical_device,
            (initial_vertex_capacity * size_of::<EguiVertex>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Create initial index buffer
        let initial_index_capacity = 16384;
        let (index_buffer, index_memory) = create_buffer(
            instance,
            device,
            physical_device,
            (initial_index_capacity * size_of::<u32>()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        Ok(Self {
            render_pass,
            framebuffers,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            font_image,
            font_memory,
            font_view,
            font_sampler,
            descriptor_pool,
            descriptor_set,
            vertex_buffer,
            vertex_memory,
            vertex_capacity: initial_vertex_capacity,
            index_buffer,
            index_memory,
            index_capacity: initial_index_capacity,
            screen_width: width as f32,
            screen_height: height as f32,
        })
    }

    /// Update screen size and recreate framebuffers (call when window resizes)
    pub unsafe fn resize(
        &mut self,
        device: &Device,
        swapchain_image_views: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Destroy old framebuffers
        for fb in self.framebuffers.drain(..) {
            device.destroy_framebuffer(fb, None);
        }
        
        // Create new framebuffers
        self.framebuffers = create_egui_framebuffers(device, self.render_pass, swapchain_image_views, width, height)?;
        self.screen_width = width as f32;
        self.screen_height = height as f32;
        Ok(())
    }

    /// Render egui primitives
    pub unsafe fn render(
        &mut self,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        clipped_primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<()> {
        if clipped_primitives.is_empty() {
            return Ok(());
        }

        // Collect vertices and indices
        let mut vertices: Vec<EguiVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for clipped in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                // Skip if not using font texture
                if mesh.texture_id != egui::TextureId::default() {
                    continue;
                }

                let vertex_offset = vertices.len() as u32;

                for v in &mesh.vertices {
                    vertices.push(EguiVertex {
                        pos: [v.pos.x * pixels_per_point, v.pos.y * pixels_per_point],
                        uv: [v.uv.x, v.uv.y],
                        color: [v.color.r(), v.color.g(), v.color.b(), v.color.a()],
                    });
                }

                for i in &mesh.indices {
                    indices.push(vertex_offset + *i);
                }
            }
        }

        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        // Resize vertex buffer if needed
        if vertices.len() > self.vertex_capacity {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);

            self.vertex_capacity = vertices.len() * 2;
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                (self.vertex_capacity * size_of::<EguiVertex>()) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.vertex_buffer = buf;
            self.vertex_memory = mem;
        }

        // Resize index buffer if needed
        if indices.len() > self.index_capacity {
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_memory, None);

            self.index_capacity = indices.len() * 2;
            let (buf, mem) = create_buffer(
                instance,
                device,
                physical_device,
                (self.index_capacity * size_of::<u32>()) as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.index_buffer = buf;
            self.index_memory = mem;
        }

        // Upload vertex data
        let vertex_size = (vertices.len() * size_of::<EguiVertex>()) as u64;
        let vdata = device.map_memory(self.vertex_memory, 0, vertex_size, vk::MemoryMapFlags::empty())?;
        memcpy(vertices.as_ptr(), vdata.cast(), vertices.len());
        device.unmap_memory(self.vertex_memory);

        // Upload index data
        let index_size = (indices.len() * size_of::<u32>()) as u64;
        let idata = device.map_memory(self.index_memory, 0, index_size, vk::MemoryMapFlags::empty())?;
        memcpy(indices.as_ptr(), idata.cast(), indices.len());
        device.unmap_memory(self.index_memory);

        // Begin egui render pass (loads existing content, renders UI on top)
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(vk::Extent2D {
                width: self.screen_width as u32,
                height: self.screen_height as u32,
            });
        
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(&[]); // No clear - we load existing content

        device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);

        // Bind pipeline
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

        // Set viewport
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.screen_width)
            .height(self.screen_height)
            .min_depth(0.0)
            .max_depth(1.0);
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

        // Push screen size
        let screen_size = [self.screen_width, self.screen_height];
        let screen_size_bytes: &[u8] = std::slice::from_raw_parts(
            screen_size.as_ptr() as *const u8,
            size_of::<[f32; 2]>(),
        );
        device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            screen_size_bytes,
        );

        // Bind buffers
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(command_buffer, self.index_buffer, 0, vk::IndexType::UINT32);

        // Bind descriptor set
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.descriptor_set],
            &[],
        );

        // Draw each primitive with clip rect
        let mut index_offset = 0u32;
        for clipped in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive {
                if mesh.texture_id != egui::TextureId::default() {
                    continue;
                }

                let clip = &clipped.clip_rect;

                // Calculate scissor rect
                let min_x = (clip.min.x * pixels_per_point).round().max(0.0) as i32;
                let min_y = (clip.min.y * pixels_per_point).round().max(0.0) as i32;
                let max_x = (clip.max.x * pixels_per_point).round().min(self.screen_width) as i32;
                let max_y = (clip.max.y * pixels_per_point).round().min(self.screen_height) as i32;

                let width = (max_x - min_x).max(0) as u32;
                let height = (max_y - min_y).max(0) as u32;

                if width == 0 || height == 0 {
                    index_offset += mesh.indices.len() as u32;
                    continue;
                }

                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D { x: min_x, y: min_y })
                    .extent(vk::Extent2D { width, height });

                device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                let index_count = mesh.indices.len() as u32;
                device.cmd_draw_indexed(command_buffer, index_count, 1, index_offset, 0, 0);
                index_offset += index_count;
            }
        }

        // End egui render pass
        device.cmd_end_render_pass(command_buffer);

        Ok(())
    }

    /// Cleanup resources
    pub unsafe fn destroy(&mut self, device: &Device) {
        for fb in &self.framebuffers {
            device.destroy_framebuffer(*fb, None);
        }
        device.destroy_render_pass(self.render_pass, None);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.index_memory, None);
        device.destroy_buffer(self.vertex_buffer, None);
        device.free_memory(self.vertex_memory, None);
        device.destroy_sampler(self.font_sampler, None);
        device.destroy_image_view(self.font_view, None);
        device.free_memory(self.font_memory, None);
        device.destroy_image(self.font_image, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

/// Create the egui rendering pipeline
unsafe fn create_egui_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    // Load shader modules
    let vert_code: &[u8] = include_bytes!("../shaders/egui_vert.spv");
    let frag_code: &[u8] = include_bytes!("../shaders/egui_frag.spv");

    println!("egui shader sizes: vert={}, frag={}", vert_code.len(), frag_code.len());

    let vert_module = create_shader_module(device, vert_code)?;
    let frag_module = create_shader_module(device, frag_code)?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(b"main\0");

    let stages = &[vert_stage, frag_stage];

    // Vertex input
    let binding = vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(size_of::<EguiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);

    let pos_attr = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(0);

    let uv_attr = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(8);

    let color_attr = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(2)
        .format(vk::Format::R8G8B8A8_UNORM)
        .offset(16);

    let bindings = &[binding];
    let attributes = &[pos_attr, uv_attr, color_attr];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Dynamic viewport and scissor
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    // Alpha blending (premultiplied)
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[color_blend_attachment];
    let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(attachments);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // Dynamic states
    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
        .0[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(pipeline)
}

unsafe fn create_shader_module(device: &Device, code: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(code).map_err(|e| anyhow!("Invalid shader bytecode: {}", e))?;
    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());
    Ok(device.create_shader_module(&info, None)?)
}

/// Create a texture for egui (font atlas)
unsafe fn create_texture(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let size = (width * height * 4) as u64;

    // Create staging buffer
    let (staging_buffer, staging_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Copy pixels to staging
    let data = device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(pixels.as_ptr(), data.cast(), pixels.len());
    device.unmap_memory(staging_memory);

    // Create image
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::_1);

    let image = device.create_image(&image_info, None)?;

    let requirements = device.get_image_memory_requirements(image);
    let memory_type = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, memory, 0)?;

    // Transition and copy
    transition_image_layout(
        device,
        command_pool,
        graphics_queue,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )?;

    copy_buffer_to_image(device, command_pool, graphics_queue, staging_buffer, image, width, height)?;

    transition_image_layout(
        device,
        command_pool,
        graphics_queue,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;

    // Cleanup staging
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    // Create image view
    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = device.create_image_view(&view_info, None)?;

    Ok((image, memory, view))
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;
    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_type = find_memory_type(instance, physical_device, requirements.memory_type_bits, properties)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;

    Ok((buffer, memory))
}

unsafe fn find_memory_type(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    for i in 0..memory_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && memory_properties.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }

    Err(anyhow!("Failed to find suitable memory type"))
}

unsafe fn transition_image_layout(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => return Err(anyhow!("Unsupported layout transition")),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, command_pool, queue, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D { width, height, depth: 1 });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(device, command_pool, queue, command_buffer)?;

    Ok(())
}

unsafe fn begin_single_time_commands(device: &Device, command_pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_commands(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;
    device.free_command_buffers(command_pool, command_buffers);

    Ok(())
}
/// Create egui render pass that loads existing content and renders UI on top
unsafe fn create_egui_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
    // Single color attachment that loads existing content
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::LOAD) // Load existing content
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::PRESENT_SRC_KHR) // Already in present layout after main pass
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);  // Stay in present layout

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    // Dependency to ensure main render pass is complete
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    let render_pass = device.create_render_pass(&info, None)?;
    Ok(render_pass)
}

/// Create framebuffers for egui render pass
unsafe fn create_egui_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>> {
    let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());
    
    for &view in swapchain_image_views {
        let attachments = &[view];
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = device.create_framebuffer(&info, None)?;
        framebuffers.push(framebuffer);
    }

    Ok(framebuffers)
}