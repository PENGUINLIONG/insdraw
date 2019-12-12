pub mod gfx;
pub mod math;
pub mod topo;

use std::collections::HashMap;
use std::convert::TryFrom;
use ash::vk;
use spirq::error::{Error as SpirvError, Result as SpirvResult};
use spirq::SpirvBinary;
use spirq::reflect::Pipeline;
use spirq::sym::{Sym, Symbol};
use crate::gfx::{Context, InterfaceConfig, ShaderModule};

fn main() {
    use log::{debug, info, warn};
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
    let spvs = collect_spirv_binaries("assets/effects/uniform-pbr")
        .into_iter()
        .filter_map(|(name, spv)| {
            let spv = SpirvBinary::from(spv);
            if let Ok(shader_mod) = ShaderModule::new(&ctxt, &spv) {
                Some(shader_mod)
            } else {
                warn!("unable to create shader module for '{}'", name);
                None
            }
        })
        .collect::<Vec<_>>();
    graph_pipe = spvs;
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
