pub mod gfx;
pub mod math;
pub mod topo;

use std::collections::HashMap;
use std::convert::TryFrom;
use ash::vk;
use ash::extensions as ashex;
use spirq::error::{Error as SpirvError, Result as SpirvResult};
use spirq::SpirvBinary;
use spirq::reflect::Pipeline;
use spirq::sym::{Sym, Symbol};
use crate::gfx::{Context, InterfaceConfig, ShaderModule};

use ash::version::DeviceV1_0;

fn main() {
    use log::{debug, info, warn};
    env_logger::init();
    let render_cfg = InterfaceConfig::new("render")
        .require_transfer()
        .require_graphics();

    let extensions = [
        ash::extensions::khr::Surface::name(),
        ash::extensions::khr::Win32Surface::name(),
    ];

    let ctxt = Context::builder("demo")
        .filter_device(|prop| {
            prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .with_api_extensions(&extensions)
        .with_interface(render_cfg)
        .build()
        .unwrap();
    let shader_mods = collect_spirv_binaries("assets/effects/example")
        .into_iter()
        .filter_map(|(name, spv)| {
            let spv = SpirvBinary::from(spv);
            if let Ok(shader_mod) = ShaderModule::new(&ctxt, &spv) {
                Some((name, shader_mod))
            } else {
                warn!("unable to create shader module for '{}'", name);
                None
            }
        })
        .collect::<HashMap<_, _>>();





    let width: u32 = 1024;
    let height: u32 = 768;

    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            // .set_layouts(/* (...) */)
            // .push_constant_ranges(/* (..) */)
            .build();
        let layout = unsafe { ctxt.dev.create_pipeline_layout(&create_info, None) }.unwrap();
        layout
    };

    let pass = {
        let attms = &[
            vk::AttachmentDescription {
                flags: Default::default(),
                format: vk::Format::R8G8B8A8_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            },
        ];
        let color_attms = &[
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
        ];
        let subpasses = &[
            vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(color_attms)
                .build(),
        ];
        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(attms)
            .subpasses(subpasses)
            // .dependencies(/* (...) */)
            .build();
        let pass = unsafe { ctxt.dev.create_render_pass(&create_info, None) }.unwrap();
        pass
    };

    let entry_point_name = std::ffi::CString::new("main").unwrap();
    let pipe = {
        let psscis = vec![
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mods["example.vert"].handle)
                .name(&entry_point_name)
                // .specialization_info(/* ... */)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mods["example.frag"].handle)
                .name(&entry_point_name)
                .build(),
        ];

        let vert_binds = vec![
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: 2 * std::mem::size_of::<f32>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
        ];
        let vert_attrs = vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
        ];
        let pvisci = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vert_binds)
            .vertex_attribute_descriptions(&vert_attrs)
            .build();
        let piasci = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            // This one is interesting.
            .primitive_restart_enable(false)
            .build();
        let ptsci = vk::PipelineTessellationStateCreateInfo::builder()
            .build();
        let viewports = &[
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        ];
        let scissors = &[
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: width, height: height },
            },
        ];
        let pvsci = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(viewports.len() as u32)
            .viewports(viewports)
            .scissor_count(scissors.len() as u32)
            .scissors(scissors)
            .build();
        let prsci = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .depth_bias_enable(false)
            .line_width(1.0)
            .build();
        let pmsci = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(1.0)
            .build();
        let pdssci = vk::PipelineDepthStencilStateCreateInfo::builder()
            .build();
        let blend_attms = &[
            // Remember to multiply the color with the corresponding alpha.
            vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            },
        ];
        let pcbsci = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(blend_attms)
            .blend_constants([1.0, 1.0, 1.0, 1.0])
            .build();
        let pdsci = vk::PipelineDynamicStateCreateInfo::builder()
            .build();
        let create_infos = &[
            vk::GraphicsPipelineCreateInfo::builder()
                .stages(&psscis)
                .vertex_input_state(&pvisci)
                .input_assembly_state(&piasci)
                .tessellation_state(&ptsci)
                .viewport_state(&pvsci)
                .rasterization_state(&prsci)
                .multisample_state(&pmsci)
                .depth_stencil_state(&pdssci)
                .color_blend_state(&pcbsci)
                .dynamic_state(&pdsci)
                .layout(layout)
                .render_pass(pass)
                .subpass(0)
                .build(),
        ];


        let pipe_cache = vk::PipelineCache::null();
        let pipe = unsafe { ctxt.dev.create_graphics_pipelines(pipe_cache, &*create_infos, None) }.unwrap();
        pipe
    };

    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        dpi::LogicalSize,
    };
    use winit::platform::windows::WindowExtWindows;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_min_inner_size(LogicalSize::new(1024.0, 768.0))
        .with_title("insdraw lab")
        .with_transparent(true)
        .build(&event_loop)
        .unwrap();
    let hinst = window.hinstance();
    let hwnd = window.hwnd();

    let surf = {
        let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(hinst)
            .hwnd(hwnd)
            .build();
        unsafe {
            ashex::khr::Win32Surface::new(&ctxt.entry, &ctxt.inst)
                .create_win32_surface(&create_info, None)
                .unwrap()
        }
        info!("created window surface");
    };

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::EventsCleared => {
                // TODO: Update states.

                window.request_redraw();
            },
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {

            },
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                info!("terminating insdraw...");
                *control_flow = ControlFlow::Exit
            },
            _ => *control_flow = ControlFlow::Poll,
        }
    });
    info!("insdraw terminated");

/*
    let bind1 = VertexBindingPoint::new(pass, stride, input_rate);
    let attr1 = VertexAttribute::new(bind1, attr, offset, fmt);
    let attrs = vec![attr1];
    let vert = Stage::new(, attrs);
    let geom = Stage::new();
    let attms = StageFunctor::new(vert);
    let framebuf = Framebuffer::new(attms);
*/
}

use log::{info, error};
use std::path::Path;

fn collect_spirv_binaries<P: AsRef<Path>>(path: P) -> HashMap<String, SpirvBinary> {
    use std::convert::TryInto;
    use std::ffi::OsStr;
    use std::fs::{read_dir, File};
    use std::io::Read;
    use log::warn;

    read_dir(path).unwrap()
        .filter_map(|x| match x {
            Ok(rv) => Some(rv.path()),
            Err(err) => {
                warn!("cannot access to filesystem item: {}", err);
                None
            },
        })
        .filter_map(|x| {
            let mut buf = Vec::new();
            if !x.is_file() ||
                x.extension() != Some(OsStr::new("spv")) ||
                File::open(&x).and_then(|mut x| x.read_to_end(&mut buf)).is_err() ||
                buf.len() & 3 != 0 {
                return None;
            }
            let spv = buf.into();
            let name = x.file_stem()
                .and_then(OsStr::to_str)
                .map(ToOwned::to_owned)
                .unwrap();
            Some((name, spv))
        })
        .collect::<HashMap<_, _>>()
}
