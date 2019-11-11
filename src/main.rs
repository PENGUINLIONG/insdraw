pub mod gfx;
pub mod math;
pub mod topo;

use crate::gfx::{Context, InterfaceConfig, SpirvBinary};
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


use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

#[derive(Debug, FromPrimitive)]
enum ExecutionModel {
    Vertex = 0,
    Fragment = 4,
}

fn module_lab(module: &SpirvBinary) -> crate::gfx::Result<()> {
    use log::debug;
    use crate::gfx::Error;

    type TypeId = u32;
    #[derive(Debug, FromPrimitive)]
    enum ImageDim {
        Image1D = 0,
        Image2D = 1,
        Image3D = 2,
        CubeMap = 3,
        SubpassData = 6,
    }
    #[derive(Debug, FromPrimitive)]
    enum ImageContentType {
        Unknown = 2,
        Color = 0,
        Depth = 1,
    }
    #[derive(Debug, FromPrimitive)]
    enum ImageUsage {
        Unknown = 0,
        Sampled = 1,
        Storage = 2,
    }
    #[derive(Debug, FromPrimitive)]
    enum ColorFormat {
        Unknown = 0,
        Rgba32f = 1,
    }
    #[derive(Debug)]
    enum Type<'a> {
        Void,
        Bool,
        Int {
            nbit: u32,
            is_signed: bool,
        },
        Float {
            nbit: u32,
        },
        Vector {
            elem_ty: TypeId,
            nelem: u32,
        },
        Matrix {
            col_ty: TypeId,
            ncol: u32,
        },
        Image {
            prim_ty: TypeId,
            dim: ImageDim,
            content_ty: ImageContentType,
            is_array: bool,
            is_multisampled: bool,
            usage: ImageUsage,
            fmt: ColorFormat,
        },
        Sampler,
        SampledImage {
            img_ty: TypeId,
        },
        Array {
            elem_ty: TypeId,
            nelem: u32,
        },
        RuntimeArray {
            elem_ty: TypeId,
        },
        Struct {
            member_tys: &'a [TypeId]
        },
        Pointer {
            referee_ty: TypeId,
        },
    }

    const OP_ENTRY_POINT: u32 = 15;
    const OP_TYPE_VOID: u32 = 19;
    const OP_TYPE_BOOL: u32 = 20;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_TYPE_MATRIX: u32 = 24;
    const OP_TYPE_IMAGE: u32 = 25;
    const OP_TYPE_SAMPLER: u32 = 26;
    const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
    const OP_TYPE_ARRAY: u32 = 28;
    const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
    const OP_TYPE_STRUCT: u32 = 30;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_ACCESS_CHAIN: u32 = 65;

    macro_rules! spv_ty {
        ($ty_map: ident, $instr: ident, $type: ident) => { spv_ty!($ty_map, $instr, $type, {}) };
        ($ty_map: ident, $instr: ident, $type: ident, { $($id: ident <- $field_ty: ident),* }) => {
            {
                let mut operands = $instr.operands();
                let id = operands.read_u32()?;
                let ty = spv_ty!(_ty operands $type $($id $field_ty)*);
                if $ty_map.insert(id, ty).is_some() {
                    return Err(Error::CorruptedSpirv);
                }
            }
        };
        (_ty $operands: ident $type: ident) => { Type::$type };
        (_ty $operands: ident $type: ident $($id: ident $field_ty: ident)* ) => { Type::$type { $($id: $operands.$field_ty()?,)* } };
    }

    let mut ty_map: HashMap<u32, Type> = HashMap::new();
    for instr in module.instrs() {
        match instr.opcode() {
            OP_ENTRY_POINT => {
                let mut operands = instr.operands();
                let exec_model = ExecutionModel::from_u32(operands.read_u32()?).unwrap();
                let _entry_fn_id = operands.read_u32()?;
                let name = operands.read_str()?;
                let interface_ids = operands.read_list()?;
                debug!("{:?}, {:?}, {:?}", exec_model, name, interface_ids);
            },
            OP_TYPE_VOID => spv_ty!(ty_map, instr, Void),
            OP_TYPE_BOOL => spv_ty!(ty_map, instr, Bool),
            OP_TYPE_INT => spv_ty!(ty_map, instr, Int, { is_signed <- read_bool, nbit <- read_u32 }),
            OP_TYPE_FLOAT => spv_ty!(ty_map, instr, Float, { nbit <- read_u32 }),
            OP_TYPE_VECTOR => spv_ty!(ty_map, instr, Vector, { elem_ty <- read_u32, nelem <- read_u32 }),
            OP_TYPE_MATRIX => spv_ty!(ty_map, instr, Matrix, { col_ty <- read_u32, ncol <- read_u32 }),
            OP_TYPE_IMAGE => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let ty = Type::Image {
                    prim_ty: operands.read_u32()?,
                    dim: operands.read_u32()
                        .map(FromPrimitive::from_u32)?
                        .ok_or(Error::CorruptedSpirv)?,
                    content_ty: operands.read_u32()
                        .map(FromPrimitive::from_u32)?
                        .ok_or(Error::CorruptedSpirv)?,
                    is_array: operands.read_bool()?,
                    is_multisampled: operands.read_bool()?,
                    usage: operands.read_u32()
                        .map(FromPrimitive::from_u32)?
                        .ok_or(Error::CorruptedSpirv)?,
                    fmt: operands.read_u32()
                        .map(FromPrimitive::from_u32)?
                        .ok_or(Error::CorruptedSpirv)?,
                };
                if ty_map.insert(id, ty).is_some() {
                    return Err(Error::CorruptedSpirv);
                }
            },
            OP_TYPE_SAMPLER => spv_ty!(ty_map, instr, Sampler),
            OP_TYPE_SAMPLED_IMAGE => spv_ty!(ty_map, instr, SampledImage, { img_ty <- read_u32 }),
            OP_TYPE_ARRAY => spv_ty!(ty_map, instr, Array, { elem_ty <- read_u32, nelem <- read_u32 }),
            OP_TYPE_RUNTIME_ARRAY => spv_ty!(ty_map, instr, RuntimeArray, { elem_ty <- read_u32 }),
            OP_TYPE_STRUCT => spv_ty!(ty_map, instr, Struct, { member_tys <- read_list }),
            _ => continue,
        }
    }
    debug!("{:?}", ty_map);
    Ok(())
}

use std::collections::HashMap;
use std::ffi::CStr;
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
