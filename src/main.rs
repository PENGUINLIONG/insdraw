pub mod gfx;
pub mod math;
pub mod topo;

use gfx::{Context, InterfaceConfig};
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
    let _ = reflect_spirv(&spvs["uniform-pbr.frag"]).unwrap();
}

// Descriptor Semantics
//
// In InsDraw, 2 or 3 descriptor sets are used for each pipeline stage.
// Descriptor set #1 is used for material data, which can be defined freely by 
// users; #2 is a fixed-size uniform block or a variable size storage buffer
// used to store bone transformation matrices; #3 is a fixed-size uniform block
// or a variable size storage buffer used to store lighting data.
//
// # Array Sizing
//
// # Standard Lighting
//
//
//


enum MaterialValueType {
    /// 32-bit signed integer.
    Int,
    /// 32-bit IEEE 754 floating-point number.
    Float,
    /// 4-component vector.
    Vector,
    /// 4x4 column-major matrix.
    Matrix,
}

enum ImageDimension {
    Image2D
}

/// Represents a binding in descriptor set #1 in a shader module.
enum MaterialBinding {
    BufferedBlock {
        /// Binding point of the buffered block.
        binding: u32,
        /// Absolute offset from the beginning of the buffer.
        offset: u32,
        /// Type of the variable.
        ty: MaterialValueType,
    },
    SampledImage {
        /// Binding point of the sampled image.
        binding: u32,
        /// Dimension configuration of the image.
        dimension: ImageDimension,
        /// Number of layers of the image, when the image object is an array.
        nlayer: Option<u32>,
    },
}

struct UniformBufferAllocationInfo {
    /// Exact size of uniform buffer to be allocated.
    buf_size: u32,
}
struct StorageBufferAllocationInfo {
    // Minimal size of shader storage buffer to be allocated. The actual
    // length of buffer can vary when there is a unsized array at the end.
    min_buf_size: u32,
    /// Whether the binding is decorated as read-only, which can help the
    /// memory allocator to resere better performing buffers.
    readonly: bool,
}
struct Sampler {
    /// Dimension configuration of the image.
    dimension: ImageDimension,
    /// Number of layers of the image, when the image object is an array.
    nlayer: Option<u32>,
}

/// Represents a binding in descriptor set #1 in a shader module.
enum PipelineResourceInfo {
    UniformBuffer(UniformBufferAllocationInfo),
    StorageBuffer(StorageBufferAllocationInfo),
    SampledImage,
}
/// Represents material .
struct MaterialDefinition {
    buf_size: u32,
    //mat_vars: HashMap<String, MaterialVariable>,
    bindings: HashMap<u32, MaterialBinding>,
}
struct MaterialBindingInfo {
    //uniform: 
}

const MATERIAL_SET: u32         = 0;
const LIGHTING_SET: u32         = 1;
const INPUT_ATTACHMENT_SET: u32 = 2;

use std::collections::HashMap;
use std::ffi::CStr;
use log::{info, error};

fn reflect_spirv(spv: &SpirvBinary) -> Result<(), ()> {
    use std::slice;
    use std::collections::HashMap;
    use std::convert::TryInto;
    use std::ffi::{c_void, CStr};
    use spirv_reflect::ffi as refl;
    use spirv_reflect::ffi::SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS as SUCCESS;

    let mut module: refl::SpvReflectShaderModule = unsafe { std::mem::zeroed() };
    let res = unsafe { refl::spvReflectCreateShaderModule(spv.0.len() * std::mem::size_of::<u32>(), spv.0.as_ptr() as *const c_void, &mut module) };
    if res != SUCCESS {
        let msg = spirv_reflect::convert::result_to_string(res);
        error!("cannot load spirv: {}", msg);
        return Err(())
    }
    info!("successfully loaded spirv");

    let ndesc_bind = module.descriptor_binding_count as usize;
    info!("found {} descriptor bindings", ndesc_bind);
    let desc_binds = unsafe { slice::from_raw_parts(module.descriptor_bindings, ndesc_bind) };

    for desc_bind in desc_binds {
        info!("set={}, binding={}: '{:?}'", desc_bind.set, desc_bind.binding, unsafe { CStr::from_ptr(desc_bind.name) });
        let members = unsafe { slice::from_raw_parts(desc_bind.block.members, desc_bind.block.member_count as usize) };
        for member in members {
            let name = unsafe { CStr::from_ptr(member.name) }.to_string_lossy();
            info!("  {} @ {} : {}", name, member.absolute_offset, member.padded_size);
        }
    }

    let ndesc_set = module.descriptor_set_count as usize;
    info!("found {} descriptor sets", ndesc_set);
    let desc_sets = &module.descriptor_sets[..ndesc_set];

    for desc_set in desc_sets {
        let nbinds = desc_set.binding_count as usize;
        let binds = unsafe { slice::from_raw_parts(desc_set.bindings, nbinds) };
        let binds = binds.into_iter()
            .map(|bind| unsafe { **bind }.binding)
            .collect::<Vec<u32>>();
        info!("set={} bindings={:?}", desc_set.set, binds);
    }

    let npush_const = module.push_constant_block_count as usize;
    info!("found {} push constant blocks", npush_const);
    let push_consts = unsafe { slice::from_raw_parts(module.push_constant_blocks, npush_const) };

    for push_const in push_consts {
        info!("offset={}, size={}", push_const.absolute_offset, push_const.padded_size);
        let members = unsafe { slice::from_raw_parts(push_const.members, push_const.member_count as usize) };
        for member in members {
            let name = unsafe { CStr::from_ptr(member.name) }.to_string_lossy();
            info!("  {} @ {} : {}", name, member.absolute_offset, member.padded_size);
        }
    }

    let nentry = module.entry_point_count as usize;
    info!("found {} entry points", nentry);
    let entries = unsafe { slice::from_raw_parts(module.entry_points, nentry) };

    for entry in entries {
        let name = unsafe { CStr::from_ptr(entry.name) }.to_string_lossy();
        let nin_var = entry.input_variable_count as usize;
        let nout_var = entry.output_variable_count as usize;
        info!("{}, #in={}, #out={}", name, nin_var, nout_var);
        let in_vars = unsafe { slice::from_raw_parts(entry.input_variables, nin_var) };
        for in_var in in_vars {
            let name = unsafe { CStr::from_ptr(in_var.name) }.to_string_lossy();
            info!("  in  {} @ {}", name, in_var.location);
        }
        let out_vars = unsafe { slice::from_raw_parts(entry.output_variables, nout_var) };
        for out_var in out_vars {
            let name = unsafe { CStr::from_ptr(out_var.name) }.to_string_lossy();
            info!("  out {} @ {}", name, out_var.location);
        }
    }

    unsafe { refl::spvReflectDestroyShaderModule(&mut module) };
    Ok(())
}

use std::path::Path;
struct SpirvBinary(Vec<u32>);
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
                .collect::<Vec<_>>();
            let name = x.file_stem()
                .and_then(OsStr::to_str)
                .map(ToOwned::to_owned)
                .unwrap();
            Some((name, SpirvBinary(spv)))
        })
        .collect::<HashMap<_, _>>()
}
