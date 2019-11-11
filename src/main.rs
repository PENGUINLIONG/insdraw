pub mod gfx;
pub mod math;
pub mod topo;

use std::collections::HashMap;
use crate::gfx::{Context, InterfaceConfig, SpirvBinary, module_lab};
use ash::vk;

fn main() {
    env_logger::init();
    let render_cfg = InterfaceConfig::new("render")
        .require_transfer()
        .require_graphics();

    let ctxt = Context::builder("demo")
        .filter_device(|prop| {
            prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .with_interface(render_cfg)
        .build()
        .unwrap();
    let spvs = collect_spirv_binaries("assets/effects/uniform-pbr");
    info!("collected spirvs: {:?}", spvs.iter().map(|x| x.0.as_ref()).collect::<Vec<&str>>());
    let module = &spvs["uniform-pbr.frag"];



    module_lab(module).unwrap();
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
            let spv = buf.chunks_exact(4)
                .map(|x| x.try_into().unwrap())
                .map(match buf[0] {
                    0x03 => u32::from_le_bytes,
                    0x07 => u32::from_be_bytes,
                    _ => return None,
                })
                .collect::<SpirvBinary>();
            let name = x.file_stem()
                .and_then(OsStr::to_str)
                .map(ToOwned::to_owned)
                .unwrap();
            Some((name, spv))
        })
        .collect::<HashMap<_, _>>()
}
