pub mod gfx;
pub mod math;
pub mod topo;
pub mod spv;

use std::collections::HashMap;
use std::convert::TryFrom;
use ash::vk;
use crate::gfx::{Context, InterfaceConfig};
use crate::spv::{SpirvBinary, Sym};

fn main() {
    use log::debug;
    env_logger::init();
    /*
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
    */
    let spvs = collect_spirv_binaries("assets/effects/uniform-pbr");
    info!("collected spirvs: {:?}", spvs.iter().map(|x| x.0.as_ref()).collect::<Vec<&str>>());
    let entries = spvs["uniform-pbr.vert"].reflect().unwrap();
    debug!("{:#?}", entries);
    let (offset, var_ty) = entries[0].resolve_desc(Sym::new(".model_view")).unwrap();
    debug!("push_constant[model_view]: offset={:?}, ty={:?}", offset, var_ty);
    let (offset, var_ty) = entries[0].resolve_desc(Sym::new(".view_proj")).unwrap();
    debug!("push_constant[view_proj]: offset={:?}, ty={:?}", offset, var_ty);

    let entries = spvs["uniform-pbr.frag"].reflect().unwrap();
    debug!("{:#?}", entries);
    let (offset, var_ty) = entries[0].resolve_desc(Sym::new("mat.fdsa.1")).unwrap();
    debug!("mat.fdsa.1: offset={:?}, ty={:?}", offset, var_ty);
    let (offset, var_ty) = entries[0].resolve_desc(Sym::new("someImage")).unwrap();
    debug!("someImage: offset={:?}, ty={:?}", offset, var_ty);
    let (offset, var_ty) = entries[0].resolve_desc(Sym::new("imgggg")).unwrap();
    debug!("imgggg: offset={:?}, ty={:?}", offset, var_ty);
    /*
    let spvs = spvs.into_iter()
        .map(|(_, bin)| bin)
        .collect::<Vec<_>>();
    let pipe = spv::PipelineMetadata::new(&spvs);
    debug!("{:#?}", pipe);
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
