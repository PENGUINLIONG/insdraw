pub mod gfx;
pub mod math;

use std::collections::HashMap;
use ash::vk;
use spirq::{SpirvBinary};
use crate::gfx::{Context, Device, ShaderModule, Buffer, BufferConfig,
    MemoryUsage, VertexHead, FragmentHead, AttributeBinding, DeviceProc,
    Transaction, BindPoint, AttachmentReference, ShaderArray,
    GraphicsPipeline, RenderPass, GraphicsRasterizationConfig, Image,
    ImageConfig};

fn main() {
    use log::{debug, info, warn};
    env_logger::init();

    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        dpi::LogicalSize,
    };
    use winit::platform::windows::WindowExtWindows;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        //.with_min_inner_size(LogicalSize::new(1024.0, 768.0))
        .with_title("insdraw lab")
        .with_transparent(true)
        .build(&event_loop)
        .unwrap();

    let ctxt = Context::new("demo", Some(&window)).unwrap();
    let physdev = ctxt.physdevs()
        .filter_map(|physdev| physdev.ok())
        .find(|physdev| {
            physdev.prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .unwrap();

    let dev = Device::new(&physdev).unwrap();
    let swapchain_img_cfg = dev.swapchain_img_cfg().unwrap().clone();
    let shader_mods = collect_spirv_binaries("assets/effects/example")
        .into_iter()
        .filter_map(|(name, spv)| {
            let spv = SpirvBinary::from(spv);
            if let Ok(shader_mod) = ShaderModule::new(&dev, &spv) {
                Some((name, shader_mod))
            } else {
                warn!("unable to create shader module for '{}'", name);
                None
            }
        })
        .collect::<HashMap<_, _>>();

    struct Head {
        attr_binds: Vec<AttributeBinding>,
        attm_refs: Vec<AttachmentReference>,
    };
    impl Head {
        fn new(swapchain_img_cfg: &ImageConfig) -> Head {
            Head {
                attr_binds: vec![
                    AttributeBinding {
                        bind: 0,
                        offset: 0,
                        stride: 5 * std::mem::size_of::<f32>(),
                        fmt: vk::Format::R32G32_SFLOAT,
                    },
                    AttributeBinding {
                        bind: 0,
                        offset: 2 * std::mem::size_of::<f32>(),
                        stride: 5 * std::mem::size_of::<f32>(),
                        fmt: vk::Format::R32G32B32_SFLOAT,
                    },
                ],
                attm_refs: vec![
                    AttachmentReference {
                        attm_idx: 0,
                        fmt: swapchain_img_cfg.fmt,
                        load_op: vk::AttachmentLoadOp::DONT_CARE,
                        store_op: vk::AttachmentStoreOp::STORE,
                        blend_state: None,
                        init_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    },
                ],
            }
        }
    }
    impl VertexHead for Head {
        fn attr_bind(&self, location: u32) -> Option<&AttributeBinding> {
            Some(&self.attr_binds[location as usize])
        }
    }
    impl FragmentHead for Head {
        fn color_attm_ref(&self, location: u32) -> Option<&AttachmentReference> {
            Some(&self.attm_refs[location as usize])
        }
        fn depth_attm_ref(&self) -> Option<&AttachmentReference> {
            None
        }
    }
    let vert = shader_mods["example.vert"].entry_points().next().unwrap().unwrap();
    let frag = shader_mods["example.frag"].entry_points().next().unwrap().unwrap();
    let head = Head::new(&swapchain_img_cfg);
    // Note that this is sorted in stage order.
    let shader_arr = ShaderArray::new(&[vert, frag]).unwrap();
    let raster_cfg = GraphicsRasterizationConfig {
        wireframe: false,
        cull_mode: vk::CullModeFlags::NONE,
    };
    /*
    let render_target = {
        let cfg = ImageConfig {
            fmt: vk::Format::R8_UNORM,
            view_ty: vk::ImageViewType::TYPE_2D,
            width: 64,
            height: 64,
            depth: 1,
            nlayer: 1,
            nmip: 1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        };
        Image::new(&dev, cfg, MemoryUsage::Device).unwrap()
    };
    */
    let graph_pipe = GraphicsPipeline::new(
        &shader_arr, &head, &head, raster_cfg, None, None
    ).unwrap();
    let pass = RenderPass::new(&dev, &[graph_pipe], &[]).unwrap();


    let devproc = DeviceProc::new(&dev, |sym| {
        // TODO: Use macro to make this neat?
        let mesh   = sym.buf("mesh");
        let nvert  = sym.count("nvert");
        let ninst  = sym.count("ninst");
        let target = sym.img("target");

        // NOTE: Order is important.
        let read_pass = sym.flow()
            .bind(BindPoint::VertexInput(0), Some(mesh))
            .bind(BindPoint::Attachment(0), Some(target))
            .draw(&pass, nvert, ninst)
            .pause();

        sym.graph(&read_pass)
    });


    let mesh = {
        let verts: Vec<f32> = vec![
            -0.25, 0.433, 1.0, 0.0, 0.0,
             0.0, -0.433, 0.0, 1.0, 0.0,
             0.25, 0.433, 0.0, 0.0, 1.0,
        ];
        Buffer::with_data(
            &dev,
            &verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryUsage::Push,
        ).unwrap()
    };

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::EventsCleared => {
                // TODO: Update states.

                window.request_redraw();
            },
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let swapchain_img = dev.acquire_swapchain_img()
                    .unwrap();
                let var_dict = [
                    ("mesh", mesh.clone().into()),
                    ("nvert", 3.into()),
                    ("ninst", 1.into()),
                    ("target", swapchain_img.img().into()),
                ].iter()
                    .cloned()
                    .collect::<HashMap<_,_>>();
                while swapchain_img.wait(100).is_err() { }
                let mut submitted = Transaction::new(&devproc)
                    .unwrap()
                    .arm(&var_dict)
                    .unwrap()
                    .submit()
                    .unwrap();
                while submitted.wait(100).is_err() { }
                let transact = submitted.reset().unwrap();
            
                let _ = swapchain_img.present();
                window.request_redraw()
            },
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                info!("terminating insdraw...");
                *control_flow = ControlFlow::Exit
            },
            _ => *control_flow = ControlFlow::Poll,
        }
    });
    info!("insdraw terminated");
}

use log::{info, error};
use std::path::Path;

fn collect_spirv_binaries<P: AsRef<Path>>(path: P) -> HashMap<String, SpirvBinary> {
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
