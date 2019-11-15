//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
use crate::gfx::{Result, Error, Symbol};
use std::mem::size_of;
use std::marker::PhantomData;
use std::collections::{HashMap, HashSet};
use std::iter::{FromIterator, Peekable};
use std::ffi::CStr;
use std::ops::{Range, RangeInclusive};
use std::convert::{TryFrom, TryInto};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use log::{debug, info, warn, error};

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


#[derive(Debug)]
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



#[derive(PartialEq, Eq, Debug, FromPrimitive)]
enum ExecutionModel {
    Vertex = 0,
    Fragment = 4,
}
#[derive(Debug)]
struct EntryPoint<'a> {
    exec_model: ExecutionModel,
    func: u32,
    name: &'a str,
}

type TypeId = u32;
#[derive(Debug, Clone)]
enum Type<'a> {
    Numeric(NumericType),
    Image(ImageType),
    Array(TypeId, Option<u32>),
    Struct(&'a [TypeId]),
    Pointer(TypeId),
}
#[derive(Debug, Clone, Default)]
struct NumericType {
    /// Bit-width of this type.
    nbyte: u32,
    /// For integral types the field indicate it's signed ness, true for signed
    /// int and false for unsigned. Floating point number will have this field
    /// `None`.
    is_signed: Option<bool>,
    /// Row number for matrix types and element number for vector types.
    nrow: Option<u32>,
    /// Column number for matrix types.
    ncol: Option<u32>,
}
impl NumericType {
    pub fn i32() -> NumericType {
        NumericType {
            nbyte: 4,
            is_signed: Some(true),
            ..Default::default()
        }
    }
    pub fn u32() -> NumericType {
        NumericType {
            nbyte: 4,
            is_signed: Some(false),
            ..Default::default()
        }
    }
    pub fn f32() -> NumericType {
        NumericType {
            nbyte: 4,
            is_signed: None,
            ..Default::default()
        }
    }
    pub fn vec(elem_ty: &NumericType, nrow: u32) -> NumericType {
        NumericType {
            nbyte: elem_ty.nbyte,
            is_signed: elem_ty.is_signed,
            nrow: Some(nrow),
            ..Default::default()
        }
    }
    pub fn mat(col_ty: &NumericType, ncol: u32) -> NumericType {
        NumericType {
            nbyte: col_ty.nbyte,
            is_signed: col_ty.is_signed,
            nrow: col_ty.nrow,
            ncol: Some(ncol),
        }
    }

    pub fn nbyte(&self) -> u32 { self.nbyte }
    pub fn nrow(&self) -> u32 { self.nrow.unwrap_or(1) }
    pub fn ncol(&self) -> u32 { self.ncol.unwrap_or(1) }

    pub fn is_primitive(&self) -> bool { self.nrow.is_none() && self.ncol.is_none() }
    pub fn is_vec(&self) -> bool { self.nrow.is_some() && self.ncol.is_none() }
    pub fn is_mat(&self) -> bool { self.nrow.is_some() && self.ncol.is_some() }

    pub fn is_sint(&self) -> bool { Some(true) == self.is_signed }
    pub fn is_uint(&self) -> bool { Some(false) == self.is_signed }
    pub fn is_float(&self) -> bool { None == self.is_signed }
}
#[derive(Debug, Clone, Copy)]
pub enum ColorFormat {
    Rgba32f = 1,
    R32f = 3,
    Rgba8 = 4,
}
#[derive(Debug, Clone, Copy)]
pub enum ImageUnitFormat {
    Color(ColorFormat),
    Sampled,
    Depth,
}
#[derive(Debug, Clone)]
pub enum ImageType {
    Image1D {
        /// Format is `None` if the image is a sampled image. Otherwise the
        /// storage image color format is given. This is the only type trait we
        /// should care about at the host, because the SPIR-V might consume
        /// another type converted by the sampler internally.
        fmt: ImageUnitFormat,
        is_array: bool,
    },
    Image2D {
        fmt: ImageUnitFormat,
        is_array: bool,
        is_multisampled: bool,
    },
    Image3D {
        fmt: ImageUnitFormat,
    },
    CubeMap {
        fmt: ImageUnitFormat,
        is_array: bool,
    },
    SubpassData,
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
    BuiltIn,
    Location(u32),
    Binding(u32),
    DescriptorSet(u32),
    Offset(usize),
    InputAttachmentIndex(u32),
}
#[derive(Debug, Clone, Copy, FromPrimitive)]
enum StorageClass {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Function = 7, // Texture calls to sampler object will translate to this.
    PushConstant = 9,
    StorageBuffer = 12,
}

type VariableId = u32;
#[derive(Debug)]
struct Variable {
    ty: TypeId,
    store_cls: StorageClass,
}

type FunctionId = u32;
#[derive(Default, Debug)]
struct Function {
    accessed_vars: HashSet<VariableId>,
    calls: HashSet<FunctionId>,
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
                exec_model: operands.read_u32()
                    .map(FromPrimitive::from_u32)?
                    .ok_or(Error::UnsupportedSpirv)?,
                func: operands.read_u32()?,
                name: operands.read_str()?,
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
    const RANGE: RangeInclusive<u32> = OP_NAME..=OP_MEMBER_NAME;

    let mut name_map = HashMap::<(u32, Option<u32>), &'a str>::new();
    while let Some(instr) = instrs.peek() {
        if !RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.peek() {
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
fn extract_decos<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<HashMap<(u32, Option<u32>), Vec<Decoration>>> {
    const OP_DECORATE: u32 = 71;
    const OP_MEMBER_DECORATE: u32 = 72;
    const RANGE: RangeInclusive<u32> = OP_DECORATE..=OP_MEMBER_DECORATE;
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

    let mut deco_map = HashMap::<(u32, Option<u32>), Vec<Decoration>>::new();
    while let Some(instr) = instrs.peek() {
        if !RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.peek() {
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
                DECO_BUILT_IN => Decoration::BuiltIn,
                DECO_LOCATION => Decoration::Location(get_first(params)?),
                DECO_BINDING => Decoration::Binding(get_first(params)?),
                DECO_OFFSET => Decoration::Offset(get_first(params)? as usize),
                DECO_INPUT_ATTACHMENT_INDEX => Decoration::InputAttachmentIndex(get_first(params)?),
                _ => { instrs.next(); continue; }, // Ignore unsupported decos.
            };
            deco_map.entry((target_id, member_id)).or_default().push(deco);
        } else { break; }
        instrs.next();
    }

    Ok(deco_map)
}
fn extract_types<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<(HashMap<TypeId, Type<'a>>, HashMap<VariableId, Variable>)> {
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
    const OP_CONSTANT_TRUE: u32 = 41;
    const OP_CONSTANT_FALSE: u32 = 42;
    const OP_CONSTANT: u32 = 43;
    const OP_CONSTANT_COMPOSITE: u32 = 44;
    const OP_CONSTANT_SAMPLER: u32 = 45;
    const OP_CONSTANT_NULL: u32 = 46;
    const OP_SPEC_CONSTANT_TRUE: u32 = 48;
    const OP_SPEC_CONSTANT_FALSE: u32 = 49;
    const OP_SPEC_CONSTANT: u32 = 50;
    const OP_SPEC_CONSTANT_COMPOSITE: u32 = 51;
    const OP_SPEC_CONSTANT_OP: u32 = 52;
    const OP_VARIABLE: u32 = 59;
    const TYPE_RANGE: RangeInclusive<u32> = OP_TYPE_VOID..=OP_TYPE_FUNCTION;
    const CONST_RANGE: RangeInclusive<u32> = OP_CONSTANT_TRUE..=OP_SPEC_CONSTANT_OP;

    let mut ty_map: HashMap<TypeId, Type<'a>> = HashMap::new();
    let mut var_map: HashMap<VariableId, Variable> = HashMap::new();
    while let Some(instr) = instrs.peek() {
        let opcode = instr.opcode();
        if !TYPE_RANGE.contains(&opcode) &&
            !CONST_RANGE.contains(&opcode) &&
            opcode != OP_VARIABLE { instrs.next(); } else { break; }
    }
    while let Some(instr) = instrs.peek() {
        match instr.opcode() {
            OP_TYPE_VOID => { /* Never a resource type. */ },
            OP_TYPE_BOOL => { return Err(Error::UnsupportedSpirv) },
            OP_TYPE_INT => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let nbyte = operands.read_u32()? >> 3;
                if nbyte != 4 { return Err(Error::UnsupportedSpirv) }
                let is_signed = operands.read_bool()?;
                let int_ty = if is_signed { NumericType::i32() } else { NumericType::u32() };
                let ty = Type::Numeric(int_ty);
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_FLOAT => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let nbyte = operands.read_u32()? >> 3;
                if nbyte != 4 { return Err(Error::UnsupportedSpirv) }
                let ty = Type::Numeric(NumericType::f32());
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_VECTOR | OP_TYPE_MATRIX => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let elem_ty = operands.read_u32()?;
                let elem_ty = ty_map.get(&elem_ty)
                    .ok_or(Error::CorruptedSpirv)?;
                let elem_ty = if let Type::Numeric(num_ty) = elem_ty {
                    num_ty
                } else { return Err(Error::CorruptedSpirv); };
                let nelem = operands.read_u32()?;
                let ty = if instr.opcode() == OP_TYPE_VECTOR {
                    if !elem_ty.is_primitive() { return Err(Error::CorruptedSpirv); }
                    Type::Numeric(NumericType::vec(elem_ty, nelem))
                } else {
                    if !elem_ty.is_vec() { return Err(Error::CorruptedSpirv); }
                    Type::Numeric(NumericType::mat(elem_ty, nelem))
                };
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_IMAGE => {
                const DIM_IMAGE_1D: u32 = 0;
                const DIM_IMAGE_2D: u32 = 1;
                const DIM_IMAGE_3D: u32 = 2;
                const DIM_IMAGE_CUBE: u32 = 3;
                const DIM_IMAGE_SUBPASS_DATA: u32 = 6;

                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let unit_ty = operands.read_u32()?;
                let dim = operands.read_u32()?;
                let is_depth = match operands.read_u32()? {
                    0 => false, 1 => true,
                    _ => return Err(Error::UnsupportedSpirv),
                };
                let is_array = operands.read_bool()?;
                let is_multisampled = operands.read_bool()?;
                let is_sampled = match operands.read_u32()? {
                    1 => true, 2 => false,
                    _ => return Err(Error::UnsupportedSpirv),
                };
                // Only unit types allowed to be stored in storage images can
                // have given format.
                let fmt = if is_sampled {
                    ImageUnitFormat::Sampled
                } else if is_depth {
                    ImageUnitFormat::Depth
                } else {
                    let color_fmt = match operands.read_u32()? {
                        1 => ColorFormat::Rgba32f,
                        3 => ColorFormat::R32f,
                        4 => ColorFormat::Rgba8,
                        _ => return Err(Error::UnsupportedSpirv),
                    };
                    ImageUnitFormat::Color(color_fmt)
                };

                let img_ty = match dim {
                    DIM_IMAGE_1D => ImageType::Image1D {
                        fmt: fmt,
                        is_array: is_array,
                    },
                    DIM_IMAGE_2D => ImageType::Image2D {
                        fmt: fmt,
                        is_array: is_array,
                        is_multisampled: is_multisampled,
                    },
                    DIM_IMAGE_3D => ImageType::Image3D {
                        fmt: fmt,
                    },
                    DIM_IMAGE_CUBE => ImageType::CubeMap {
                        fmt: fmt,
                        is_array: is_array,
                    },
                    DIM_IMAGE_SUBPASS_DATA => ImageType::SubpassData,
                    _ => return Err(Error::UnsupportedSpirv),
                };
                let ty = Type::Image(img_ty);
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_SAMPLED_IMAGE => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let img_ty_id = operands.read_u32()?;
                let ty = ty_map.get(&img_ty_id)
                    .ok_or(Error::CorruptedSpirv)?;
                if let Type::Image(_) = ty {
                    if ty_map.insert(id, ty.clone()).is_some() { return Err(Error::CorruptedSpirv); }
                } else { return Err(Error::CorruptedSpirv); }
            }
            OP_TYPE_SAMPLER => { /* Not in GLSL. */ },
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let elem_ty_id = operands.read_u32()?;
                let elem_ty = ty_map.get(&elem_ty_id)
                    .ok_or(Error::CorruptedSpirv)?;
                let ty = if let Type::Array(sub_elem_ty_id, sub_nelem) = elem_ty {
                    // Variant-length array can only be the outermost type.
                    let sub_nelem = sub_nelem.ok_or(Error::CorruptedSpirv)?;
                    // Fold nesting sized arrays, sharing the same subtype.
                    let nelem = if instr.opcode() == OP_TYPE_ARRAY {
                        Some(sub_nelem * operands.read_u32()?)
                    } else { None };
                    Type::Array(*sub_elem_ty_id, nelem)
                } else {
                    // Non-nesting sized arrays. Just wrap another layer.
                    let nelem = if instr.opcode() == OP_TYPE_ARRAY {
                        Some(operands.read_u32()?)
                    } else { None };
                    Type::Array(elem_ty_id, nelem)
                };
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_STRUCT => {
                // spv_ty!(ty_map, instr, Struct, { member_tys <- read_list })
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let ty = Type::Struct(operands.read_list()?);
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_POINTER => {
                let mut operands = instr.operands();
                let id = operands.read_u32()?;
                let _store_cls = operands.read_u32()?;
                let target_ty = operands.read_u32()?;
                let ty = Type::Pointer(target_ty);
                if ty_map.insert(id, ty).is_some() { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_FUNCTION => { /* Don't need this. */ },
            OP_VARIABLE => {
                let mut operands = instr.operands();
                let ty = operands.read_u32()?;
                let id = operands.read_u32()?;
                let store_cls = operands.read_u32()
                    .map(FromPrimitive::from_u32)?
                    .ok_or(Error::UnsupportedSpirv)?;
                let var = Variable {
                    ty: ty,
                    store_cls: store_cls,
                };
                if var_map.insert(id, var).is_some() { return Err(Error::CorruptedSpirv); }
            },
            opcode => {
                if !CONST_RANGE.contains(&opcode) { break; }
            },
        }
        instrs.next();
    }
    // Variables' types are always pointer types, which is not very useful for.
    // We dereference the pointer and get the underlying actual types instead.
    for var in var_map.values_mut() {
        if let Some(Type::Pointer(target_ty)) = ty_map.get(&var.ty) {
            var.ty = *target_ty;
        } else {
            return Err(Error::CorruptedSpirv);
        }
    }
    Ok((ty_map, var_map))
}
fn extract_funcs<'a>(instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<HashMap<FunctionId, Function>> {
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_END: u32 = 56;
    const OP_FUNCTION_CALL: u32 = 57;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_LOAD: u32 = 61;
    const OP_STORE: u32 = 62;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;

    let mut func_map: HashMap<FunctionId, Function> = HashMap::new();
    while instrs.peek().is_some() {
        let mut func: &mut Function = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        while let Some(instr) = instrs.next() {
            if instr.opcode() == OP_FUNCTION {
                let mut operands = instr.operands();
                let _rty = operands.read_u32()?;
                let id = operands.read_u32()?;
                func = func_map.entry(id).or_default();
                break;
            }
        }
        while let Some(instr) = instrs.next() {
            match instr.opcode() {
                OP_FUNCTION_CALL => {
                    let mut operands = instr.operands();
                    let _rty = operands.read_u32()?;
                    let _id = operands.read_u32()?;
                    let callee = operands.read_u32()?;
                    func.calls.insert(callee);
                },
                OP_ACCESS_CHAIN | OP_LOAD => {
                    let mut operands = instr.operands();
                    let _rty = operands.read_u32()?;
                    let _id = operands.read_u32()?;
                    let target = operands.read_u32()?;
                    func.accessed_vars.insert(target);
                },
                OP_STORE => {
                    let mut operands = instr.operands();
                    let target = operands.read_u32()?;
                    func.accessed_vars.insert(target);
                },
                OP_FUNCTION_END => break,
                _ => { },
            }
        }
    }
    Ok(func_map)
}

use crate::gfx::contract::{VertexAttributeContract, AttachmentContract, PipelineStageContract, DescriptorContract};

#[derive(Debug)]
struct SpirvMetadata<'a> {
    pub entry_points: Vec<EntryPoint<'a>>,
    pub name_map: HashMap<(u32, Option<u32>), &'a str>,
    pub deco_map: HashMap<(u32, Option<u32>), Vec<Decoration>>,
    pub ty_map: HashMap<TypeId, Type<'a>>,
    pub var_map: HashMap<VariableId, Variable>,
    pub func_map: HashMap<FunctionId, Function>,
}
impl<'a> TryFrom<&'a SpirvBinary> for SpirvMetadata<'a> {
    type Error = Error;
    fn try_from(module: &'a SpirvBinary) -> Result<SpirvMetadata<'a>> {
        // Don't change the order. See _2.4 Logical Layout of a Module_ of the
        // SPIR-V specification for more information.
        let mut instrs = module.instrs().peekable();
        let entry_points = extract_entry_points(&mut instrs)?;
        let name_map = extract_names(&mut instrs)?;
        let deco_map = extract_decos(&mut instrs)?;
        let (ty_map, var_map) = extract_types(&mut instrs)?;
        let func_map = extract_funcs(&mut instrs)?;

        let meta = SpirvMetadata {
            entry_points: entry_points,
            name_map: name_map,
            deco_map: deco_map,
            ty_map: ty_map,
            var_map: var_map,
            func_map: func_map,
        };
        Ok(meta)
    }
}
impl<'a> SpirvMetadata<'a> {
    fn collect_fn_vars_impl(&self, func: FunctionId, vars: &mut HashSet<VariableId>) {
        let func = &self.func_map[&func];
        let it = func.accessed_vars.iter()
            .filter(|x| self.var_map.contains_key(x));
        vars.extend(it);
        for call in func.calls.iter() {
            self.collect_fn_vars_impl(*call, vars);
        }
    }
    pub fn collect_fn_vars(&self, func: FunctionId) -> impl Iterator<Item=VariableId> {
        let mut accessed_vars = HashSet::new();
        self.collect_fn_vars_impl(func, &mut accessed_vars);
        accessed_vars.into_iter()
    }
}

#[derive(Debug)]
struct VertexAttributeContractTemplate {
    bind_point: u32,
    location: u32,
    /// Offset in each set of vertex data.
    offset: usize,
    /// Total byte count at this location.
    nbyte: usize,
}
#[derive(Debug)]
struct AttachmentContractTemplate {
    location: u32,
    /// Total byte count at this location.
    nbyte: usize,
}

/// Resolve the minimum contract for all entry points in the module.
pub fn module_lab(module: &SpirvBinary) -> crate::gfx::Result<()> {
    use std::ops::Deref;
    use log::debug;
    let meta: SpirvMetadata = module.try_into()?;

    let entry_point = &meta.entry_points[0];
    info!("{}", entry_point.name);
    let exec_model = &entry_point.exec_model;

    let mut desc_contracts = Vec::<DescriptorContract>::new();
    let mut attr_offset: usize = 0;
    let mut attr_templates = Vec::<VertexAttributeContractTemplate>::new();
    let mut attm_templates = Vec::<AttachmentContractTemplate>::new();

    for var_id in meta.collect_fn_vars(entry_point.func) {
        let var = &meta.var_map.get(&var_id)
            .ok_or(Error::CorruptedSpirv)?;
        let ty = meta.ty_map.get(&var.ty)
            .ok_or(Error::CorruptedSpirv)?;
        let decos = meta.deco_map.get(&(var_id, None))
            .map_or(&[] as &[Decoration], Deref::deref);
        match var.store_cls {
            StorageClass::Input if *exec_model == ExecutionModel::Vertex => {
                let mut bind_point = 0;
                let mut location = 0;
                for deco in decos.iter() {
                    match deco {
                        Decoration::Location(x) => location = *x,
                        Decoration::Binding(x) => bind_point = *x,
                        _ => {},
                    }
                }
                if let Type::Numeric(num_ty) = ty {
                    let col_nbyte = (num_ty.nbyte() * num_ty.nrow()) as usize;
                    for i in 0..num_ty.ncol() {
                        let template = VertexAttributeContractTemplate {
                            bind_point: bind_point,
                            location: location,
                            offset: attr_offset,
                            nbyte: col_nbyte,
                        };
                        attr_templates.push(template);
                        attr_offset += col_nbyte;
                    }
                }
                // Leak out all inputs that are not attributes.
            },
            StorageClass::Output if *exec_model == ExecutionModel::Fragment => {
                let mut location = 0;
                for deco in decos.iter() {
                    match deco {
                        Decoration::Location(x) => location = *x,
                        _ => {},
                    }
                }
                if let Type::Numeric(num_ty) = ty {
                    // Matrix is not valid attachment type.
                    if num_ty.is_mat() { return Err(Error::CorruptedSpirv); }
                    let col_nbyte = (num_ty.nbyte() * num_ty.nrow()) as usize;
                    let template = AttachmentContractTemplate {
                        location: location,
                        nbyte: col_nbyte,
                    };
                    attm_templates.push(template);
                } else { debug!("123"); }
                // Leak out all outputs that are not attachments.
            },
            _ => {},
        }
    }
    info!("{:?}", attr_templates);
    info!("{:?}", attm_templates);

    Ok(())
}
