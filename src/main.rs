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
        .with_min_inner_size(LogicalSize::new(1024.0, 768.0))
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
    let buf_cfg = BufferConfig {
        size: 12,
        usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
    };
    let buf = Buffer::new(&dev, buf_cfg, MemoryUsage::Device).unwrap();

    struct Head {
        attr_bind: AttributeBinding,
        attm_ref: AttachmentReference,
    };
    impl Head {
        fn new() -> Head {
            Head {
                attr_bind: AttributeBinding {
                    bind: 0,
                    offset: 0,
                    stride: 2 * std::mem::size_of::<f32>(),
                    fmt: vk::Format::R32G32_SFLOAT,
                },
                attm_ref: AttachmentReference {
                    attm_idx: 0,
                    fmt: vk::Format::R8G8B8A8_UNORM,
                    load_op: vk::AttachmentLoadOp::DONT_CARE,
                    store_op: vk::AttachmentStoreOp::STORE,
                    blend_state: None,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                }
            }
        }
    }
    impl VertexHead for Head {
        fn attr_bind(&self, location: u32) -> Option<&AttributeBinding> {
            if location == 0 { Some(&self.attr_bind) } else { None }
        }
    }
    impl FragmentHead for Head {
        fn attm_ref(&self, location: u32) -> Option<&AttachmentReference> {
            if location == 0 { Some(&self.attm_ref) } else { None }
        }
        fn depth_attm_idx(&self, depth_attm_id: u32) -> Option<u32> {
            None
        }
    }
    let vert = shader_mods["example.vert"].entry_points().next().unwrap().unwrap();
    let frag = shader_mods["example.frag"].entry_points().next().unwrap().unwrap();
    let head = Head::new();
    // Note that this is sorted in stage order.
    let shader_arr = ShaderArray::new(&[vert, frag]).unwrap();
    let raster_cfg = GraphicsRasterizationConfig {
        wireframe: false,
        cull_mode: vk::CullModeFlags::NONE,
    };
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
    let graph_pipe = GraphicsPipeline::new(
        &shader_arr, &head, &head, None, raster_cfg, None, None
    ).unwrap();
    let pass = RenderPass::new(&dev, &[graph_pipe], &[], (64, 64)).unwrap();







    let devproc = DeviceProc::new(&dev, |sym| {
        // TODO: Use macro to make this neat?
        let indices = sym.buf("indices");
        let mesh    = sym.buf("mesh");
        let sampler = sym.sampler("sampler");
        let nvert   = sym.count("nvert");

        // NOTE: Order is important.
        let read_pass = sym.flow()
            .bind(BindPoint::Index, Some(indices))
            .bind(BindPoint::VertexInput(0), Some(mesh))
            .draw(&pass, nvert, 1)
            .pause();

        sym.graph(&read_pass)
    });
    let transact = Transaction::new(&devproc).unwrap();
    //let mut submitted = transact.arm().unwrap()
        //.submit_present(dev.acquire_swapchain_img(100).unwrap()).unwrap();
    //while let Err(t) = submitted.wait(100) {
        //submitted = t;
    //}






/* USAGE CODE

    let pass = RenderPass::new(&dev, ..);
    let img = Image(dev, .., usage);
    let buf = Buffer(dev, .., usage);
    let param = RenderPassParameterPack::new(&pass);
    param.vert_input(buf);
    param.push_const(&[0,0,0,0] as Bytes);
    param.desc_bind(0, 0, buf);
    param.desc_bind(0, 1, img);
    let vert_input = VertexInput::new(&[data]);

    img.push(data, dev_offset); // <- should be equivalent to:
    let dev_offset = 0;
    while let Err(nbyte_txed) = buf.try_push(data, dev_offset) {
        data = &data[nbyte_txed..];
        dev_offset += nbyte_txed;
        dev.sync_data();
    }

    let mut trans = Transaction::new(&dev);
    trans.then(pass, param)
        .then(task, param);
    trans.commit()
        .await!();

*/

/*
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
*/

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
