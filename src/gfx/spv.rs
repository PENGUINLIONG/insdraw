//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
use crate::gfx::*;
use std::mem::size_of;
use std::marker::PhantomData;
use std::collections::HashMap;
use std::iter::{FromIterator, Peekable};
use std::ffi::CStr;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use log::{debug, info, warn, error};

use std::path::Path;
pub struct SpirvBinary(Vec<u32>);
impl From<Vec<u32>> for SpirvBinary {
    fn from(x: Vec<u32>) -> Self { SpirvBinary(x) }
}
impl FromIterator<u32> for SpirvBinary {
    fn from_iter<I: IntoIterator<Item=u32>>(iter: I) -> Self { SpirvBinary(iter.into_iter().collect::<Vec<u32>>()) }
}

impl SpirvBinary {
    pub fn instrs<'a>(&'a self) -> Instrs<'a> {
        const HEADER_LEN: usize = 5;
        Instrs(&self.0[HEADER_LEN..])
    }
}


pub struct Instrs<'a>(&'a [u32]);
impl<'a> Iterator for Instrs<'a> {
    type Item = Instr<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(head) = self.0.first() {
            let len = ((*head as u32) >> 16) as usize;
            if len <= self.0.len() {
                let opcode = head & 0xFFFF;
                let instr = Instr {
                    opcode: opcode,
                    operands: &self.0[1..len],
                };
                self.0 = &self.0[len..];
                return Some(instr);
            }
        }
        None
    }
}


pub struct Instr<'a> {
    opcode: u32,
    operands: &'a [u32],
}
impl<'a> Instr<'a> {
    /// Get the opcode of the instruction.
    pub fn opcode(&self) -> u32 { self.opcode }
    /// Get the word count of the instruction, including the first word
    /// containing the word count and opcode.
    pub fn word_count(&self) -> usize { self.operands.len() + 1 }
    /// Get an instruction operand reader. The reader does NO boundary checking
    /// so the user code MUST make sure the implementation follows the
    /// specification.
    pub fn operands(&self) -> Operands<'a> {
        Operands(self.operands)
    }
}

pub struct Operands<'a>(&'a [u32]);
impl<'a> Operands<'a> {
    pub fn read_bool(&mut self) -> Result<bool> { self.read_u32().map(|x| x != 0) }
    pub fn read_u32(&mut self) -> Result<u32> {
        if let Some(x) = self.0.first() {
            self.0 = &self.0[1..];
            Ok(*x)
        } else { Err(Error::CorruptedSpirv) }
    }
    pub fn read_str(&mut self) -> Result<&'a str> {
        use std::os::raw::c_char;
        let ptr = self.0.as_ptr() as *const c_char;
        let char_slice = unsafe { std::slice::from_raw_parts(ptr, self.0.len() * 4) };
        if let Some(nul_pos) = char_slice.into_iter().position(|x| *x == 0) {
            let nword = nul_pos / 4 + 1;
            self.0 = &self.0[nword..];
            if let Ok(string) = unsafe { CStr::from_ptr(ptr) }.to_str() {
                return Ok(string);
            }
        }
        Err(Error::CorruptedSpirv)
    }
    pub fn read_list(&mut self) -> Result<&'a [u32]> {
        let rv = self.0;
        self.0 = &[];
        Ok(rv)
    }
}



#[derive(Debug, FromPrimitive)]
enum ExecutionModel {
    Vertex = 0,
    Fragment = 4,
}
#[derive(Debug)]
struct EntryPoint<'a> {
    exec_model: ExecutionModel,
    fn_id: u32,
    name: &'a str,
    interface_ids: &'a [u32],
}



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

#[derive(Debug, FromPrimitive)]
enum BuiltIn {
    Position = 0,
    PointSize = 1,
    ClipDistance = 2,
    CullDistance = 3,
    VertexId = 5,
    InstanceId = 6,
    PrimitiveId = 7,
    InvocationId = 8,
    Layer = 9,
    FragCoord = 15,
    PointCoord = 16,
    FrontFacing = 17,
    SampleMask = 20,
    FragDepth = 22,
    HelperInvocation = 23,
}

#[derive(Debug)]
enum Decoration {
    SpecId(u32),
    Block,
    BufferBlock,
    RowMajor,
    ColMajor,
    ArrayStride(usize),
    MatrixStride(usize),
    BuiltIn(BuiltIn),
    Location(u32),
    Binding(u32),
    DescriptorSet(u32),
    Offset(usize),
    InputAttachmentIndex(u32),
}


fn extract_entry_points<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<Vec<EntryPoint<'a>>> {
    const OP_ENTRY_POINT: u32 = 15;

    let mut entry_points = Vec::with_capacity(1);
    while let Some(instr) = instrs.peek() {
        if instr.opcode() != OP_ENTRY_POINT { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.peek() {
        if instr.opcode() == OP_ENTRY_POINT {
            let mut operands = instr.operands();
            let entry_point = EntryPoint {
                exec_model: ExecutionModel::from_u32(operands.read_u32()?).unwrap(),
                fn_id: operands.read_u32()?,
                name: operands.read_str()?,
                interface_ids: operands.read_list()?,
            };
            entry_points.push(entry_point);
            instrs.next();
        } else { break; }
    }
    Ok(entry_points)
}
fn extract_names<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<HashMap<(u32, Option<u32>), &'a str>> {
    const OP_NAME: u32 = 5;
    const OP_MEMBER_NAME: u32 = 6;
    const RANGE: std::ops::Range<u32> = OP_NAME..(OP_MEMBER_NAME + 1);

    let mut name_map = HashMap::<(u32, Option<u32>), &'a str>::new();
    while let Some(instr) = instrs.next() {
        if !RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.next() {
        let opcode = instr.opcode();
        if RANGE.contains(&opcode) {
            let mut operands = instr.operands();
            let target_id = operands.read_u32()?;
            let member_id = if opcode == OP_MEMBER_NAME { Some(operands.read_u32()?) } else { None};
            let name = operands.read_str()?;
            name_map.insert((target_id, member_id), name);
        } else { break; }
        instrs.next();
    }
    Ok(name_map)
}
fn extract_decos<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<HashMap<(u32, Option<u32>), Decoration>> {
    const OP_DECORATE: u32 = 71;
    const OP_MEMBER_DECORATE: u32 = 72;
    const RANGE: std::ops::Range<u32> = OP_DECORATE..(OP_MEMBER_DECORATE + 1);
    const DECO_SPEC_ID: u32 = 1;
    const DECO_BLOCK: u32 = 2;
    const DECO_BUFFER_BLOCK: u32 = 3;
    const DECO_ROW_MAJOR: u32 = 4;
    const DECO_COL_MAJOR: u32 = 5;
    const DECO_ARRAY_STRIDE: u32 = 6;
    const DECO_MATRIX_STRIDE: u32 = 7;
    const DECO_BUILT_IN: u32 = 11;
    const DECO_LOCATION: u32 = 30;
    const DECO_BINDING: u32 = 33;
    const DECO_DESCRIPTOR_SET: u32 = 34;
    const DECO_OFFSET: u32 = 35;
    const DECO_INPUT_ATTACHMENT_INDEX: u32 = 43;

    let mut deco_map = HashMap::<(u32, Option<u32>), Decoration>::new();
    while let Some(instr) = instrs.next() {
        if !RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.next() {
        let opcode = instr.opcode();
        if RANGE.contains(&opcode) {
            let mut operands = instr.operands();
            let target_id = operands.read_u32()?;
            let member_id = if opcode == OP_MEMBER_DECORATE { Some(operands.read_u32()?) } else { None };
            let deco = operands.read_u32()?;
            let params = operands.read_list()?;
            fn get_first<'a>(params: &'a [u32]) -> Result<u32> { params.first().map(u32::to_owned).ok_or(Error::CorruptedSpirv) }
            let deco = match deco {
                DECO_SPEC_ID => Decoration::SpecId(get_first(params)?),
                DECO_BLOCK => Decoration::Block,
                DECO_BUFFER_BLOCK => Decoration::BufferBlock,
                DECO_ROW_MAJOR => Decoration::RowMajor,
                DECO_COL_MAJOR => Decoration::ColMajor,
                DECO_ARRAY_STRIDE => Decoration::ArrayStride(get_first(params)? as usize),
                DECO_MATRIX_STRIDE => Decoration::MatrixStride(get_first(params)? as usize),
                DECO_BUILT_IN => Decoration::BuiltIn(params.first().to_owned().and_then(|x| BuiltIn::from_u32(*x)).ok_or(Error::CorruptedSpirv)?),
                DECO_LOCATION => Decoration::Location(get_first(params)?),
                DECO_BINDING => Decoration::Binding(get_first(params)?),
                DECO_OFFSET => Decoration::Offset(get_first(params)? as usize),
                DECO_INPUT_ATTACHMENT_INDEX => Decoration::InputAttachmentIndex(get_first(params)?),
                _ => continue, // Ignore unsupported decos.
            };
            deco_map.insert((target_id, member_id), deco);
        } else { break; }
        instrs.next();
    }

    Ok(deco_map)
}
fn extract_types<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<HashMap<u32, Type<'a>>> {
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
    const OP_TYPE_FUNCTION: u32 = 33;
    const OP_ACCESS_CHAIN: u32 = 65;

    macro_rules! spv_ty {
        ($ty_map: ident, $instr: ident, $type: ident) => { spv_ty!($ty_map, $instr, $type, {}) };
        ($ty_map: ident, $instr: ident, $type: ident, { $($id: ident <- $field_ty: ident),* }) => {
            {
                let mut operands = $instr.operands();
                let id = operands.read_u32()?;
                let ty = spv_ty!(_ty operands $type $($id $field_ty)*);
                if $ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            }
        };
        (_ty $operands: ident $type: ident) => { Type::$type };
        (_ty $operands: ident $type: ident $($id: ident $field_ty: ident)* ) => { Type::$type { $($id: $operands.$field_ty()?,)* } };
    }

    let mut in_scope = false;
    let mut ty_map: HashMap<u32, Type<'a>> = HashMap::new();
    while let Some(instr) = instrs.peek() {
        if !(OP_TYPE_VOID..(OP_TYPE_FUNCTION + 1)).contains(&instr.opcode()) { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.peek() {
        match instr.opcode() {
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
                let _access = operands.read_u32();
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_SAMPLER => spv_ty!(ty_map, instr, Sampler),
            OP_TYPE_SAMPLED_IMAGE => spv_ty!(ty_map, instr, SampledImage, { img_ty <- read_u32 }),
            OP_TYPE_ARRAY => spv_ty!(ty_map, instr, Array, { elem_ty <- read_u32, nelem <- read_u32 }),
            OP_TYPE_RUNTIME_ARRAY => spv_ty!(ty_map, instr, RuntimeArray, { elem_ty <- read_u32 }),
            OP_TYPE_STRUCT => spv_ty!(ty_map, instr, Struct, { member_tys <- read_list }),
            OP_TYPE_FUNCTION => { /* Don't need this. */ },
            _ => break,
        }
        instrs.next();
    }
    Ok(ty_map)
}



pub fn module_lab(module: &SpirvBinary) -> crate::gfx::Result<()> {
    use log::debug;
    let mut instrs = module.instrs().peekable();
    let entry_points = extract_entry_points(&mut instrs)?;
    debug!("{:?}", entry_points);
    let name_map = extract_names(&mut instrs)?;
    debug!("{:?}", name_map);
    let deco_map = extract_decos(&mut instrs)?;
    debug!("{:?}", deco_map);
    let ty_map = extract_types(&mut instrs)?;
    debug!("{:?}", ty_map);
    Ok(())
}
