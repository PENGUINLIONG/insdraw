//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
use crate::gfx::{Result, Error, Symbol};
use std::mem::size_of;
use std::marker::PhantomData;
use std::collections::{HashMap, HashSet};
use std::iter::{FromIterator, Peekable};
use std::ffi::CStr;
use std::fmt;
use std::ops::{Range, RangeInclusive};
use std::convert::{TryFrom, TryInto};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use log::{debug, info, warn, error};

const OP_ENTRY_POINT: u32 = 15;

const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const NAME_RANGE: RangeInclusive<u32> = OP_NAME..=OP_MEMBER_NAME;

const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const DECO_RANGE: RangeInclusive<u32> = OP_DECORATE..=OP_MEMBER_DECORATE;

// Don't need this: Not a resource type. But kept for the range.
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
// Not in GLSL.
// const OP_TYPE_SAMPLER: u32 = 26;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
// Don't need this: Not a resource type. But kept for the range.
const OP_TYPE_FUNCTION: u32 = 33;
const TYPE_RANGE: RangeInclusive<u32> = OP_TYPE_VOID..=OP_TYPE_FUNCTION;

const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_COMPOSITE: u32 = 44;
const OP_CONSTANT_SAMPLER: u32 = 45;
const OP_CONSTANT_NULL: u32 = 46;
const CONST_RANGE: RangeInclusive<u32> = OP_CONSTANT_TRUE..=OP_CONSTANT_NULL;

const OP_SPEC_CONSTANT_TRUE: u32 = 48;
const OP_SPEC_CONSTANT_FALSE: u32 = 49;
const OP_SPEC_CONSTANT: u32 = 50;
const OP_SPEC_CONSTANT_COMPOSITE: u32 = 51;
const OP_SPEC_CONSTANT_OP: u32 = 52;
const SPEC_CONST_RANGE: RangeInclusive<u32> = OP_SPEC_CONSTANT_TRUE..=OP_SPEC_CONSTANT_OP;

const OP_VARIABLE: u32 = 59;

const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_FUNCTION_CALL: u32 = 57;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;


const EXEC_MODEL_VERTEX: u32 = 0;
const EXEC_MODEL_FRAGMENT: u32 = 4;


const DECO_SPEC_ID: u32 = 1;
const DECO_BLOCK: u32 = 2;
const DECO_BUFFER_BLOCK: u32 = 3;
const DECO_ROW_MAJOR: u32 = 4;
// Don't need this: Column-major matrices are the default.
// const DECO_COL_MAJOR: u32 = 5;
const DECO_ARRAY_STRIDE: u32 = 6;
const DECO_MATRIX_STRIDE: u32 = 7;
// Don't need this: Built-in variables will not be attribute nor attachment.
// const DECO_BUILT_IN: u32 = 11;
const DECO_LOCATION: u32 = 30;
const DECO_BINDING: u32 = 33;
const DECO_DESCRIPTOR_SET: u32 = 34;
const DECO_OFFSET: u32 = 35;
const DECO_INPUT_ATTACHMENT_INDEX: u32 = 43;


const STORE_CLS_UNIFORM_CONSTANT: u32 = 0;
const STORE_CLS_INPUT: u32 = 1;
const STORE_CLS_UNIFORM: u32 = 2;
const STORE_CLS_OUTPUT: u32 = 3;
// Texture calls to sampler object will translate to function class.
const STORE_CLS_FUNCTION: u32 = 7;
const STORE_CLS_PUSH_CONSTANT: u32 = 9;
const STORE_CLS_STORAGE_BUFFER: u32 = 12;


const DIM_IMAGE_1D: u32 = 0;
const DIM_IMAGE_2D: u32 = 1;
const DIM_IMAGE_3D: u32 = 2;
const DIM_IMAGE_CUBE: u32 = 3;
const DIM_IMAGE_SUBPASS_DATA: u32 = 6;


const IMG_UNIT_FMT_UNKNOWN: u32 = 0;
const IMG_UNIT_FMT_RGBA32F: u32 = 1;
const IMG_UNIT_FMT_R32F: u32 = 3;
const IMG_UNIT_FMT_RGBA8: u32 = 4;





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




#[derive(Debug, Clone, Default)]
pub struct NumericType {
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
impl ColorFormat {
    fn from_spv_def(color_fmt: u32) -> Result<ColorFormat> {
        let color_fmt = match color_fmt {
            IMG_UNIT_FMT_RGBA32F => ColorFormat::Rgba32f,
            IMG_UNIT_FMT_R32F => ColorFormat::R32f,
            IMG_UNIT_FMT_RGBA8 => ColorFormat::Rgba8,
            _ => return Err(Error::UnsupportedSpirv),
        };
        Ok(color_fmt)
    }
}
#[derive(Debug, Clone, Copy)]
pub enum ImageUnitFormat {
    Color(ColorFormat),
    Sampled,
    Depth,
}
impl ImageUnitFormat {
    pub fn from_spv_def(is_sampled: bool, is_depth: bool, color_fmt: u32) -> Result<ImageUnitFormat> {
        let img_unit_fmt = match (is_sampled, is_depth, color_fmt) {
            (true, false, _) => ImageUnitFormat::Sampled,
            (true, true, _) => ImageUnitFormat::Depth,
            (false, false, color_fmt) => ImageUnitFormat::Color(ColorFormat::from_spv_def(color_fmt)?),
            _ => return Err(Error::UnsupportedSpirv),
        };
        Ok(img_unit_fmt)
    }
}
#[derive(Debug, Clone, Copy)]
pub enum ImageArrangement {
    Image1D,
    Image2D,
    Image2DMS,
    Image3D,
    CubeMap,
    Image1DArray,
    Image2DArray,
    Image2DMSArray,
    CubeMapArray,
}
impl ImageArrangement {
    /// Do note this dim is not the number of dimensions but a enumeration of
    /// values specified in SPIR-V specification.
    pub fn from_spv_def(dim: u32, is_array: bool, is_multisampled: bool) -> Result<ImageArrangement> {
        let arng = match (dim, is_array, is_multisampled) {
            (DIM_IMAGE_1D, false, false) => ImageArrangement::Image1D,
            (DIM_IMAGE_1D, true, false) => ImageArrangement::Image1DArray,
            (DIM_IMAGE_2D, false, false) => ImageArrangement::Image2D,
            (DIM_IMAGE_2D, false, true) => ImageArrangement::Image2DMS,
            (DIM_IMAGE_2D, true, false) => ImageArrangement::Image2DArray,
            (DIM_IMAGE_3D, false, false) => ImageArrangement::Image3D,
            (DIM_IMAGE_3D, true, false) => ImageArrangement::Image3D,
            (DIM_IMAGE_CUBE, false, false) => ImageArrangement::CubeMap,
            (DIM_IMAGE_CUBE, true, false) => ImageArrangement::CubeMapArray,
            _ => return Err(Error::UnsupportedSpirv),
        };
        Ok(arng)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ImageType {
    fmt: ImageUnitFormat,
    arng: ImageArrangement,
}
#[derive(Debug, Clone, Copy)]
pub struct ArrayType {
    elem_ty: ObjectId,
    nelem: Option<u32>,
}
#[derive(Debug, Clone, Copy)]
pub struct StructMember<'a> {
    ty: ObjectId,
    name: Option<&'a str>,
    offset: Option<usize>,
}
#[derive(Debug, Clone)]
pub struct Variable {
    ty: ObjectId,
    store_cls: StorageClass,
}
#[derive(Debug, Clone)]
pub struct Constant<'a> {
    ty: ObjectId,
    value: &'a [u32],
}
pub type ObjectId = u32;
#[derive(Debug, Clone)]
pub enum SpirvObject<'a> {
    NumericType(NumericType),
    ImageType(Option<ImageType>),
    ArrayType(ArrayType),
    StructType(Vec<StructMember<'a>>),
    PointerType(ObjectId), // Struct ID.
    Variable(Variable),
    Constant(Constant<'a>),
    Function(Function),
}

type Decoration = u32;
type StorageClass = u32;
#[derive(Debug, Default, Clone)]
pub struct ElementMetadata<'a> {
    name: Option<&'a str>,
    decos: HashMap<Decoration, &'a [u32]>,
}
pub struct ElementMetadataMap<'a>(HashMap<(u32, Option<u32>), ElementMetadata<'a>>);

#[derive(Default, Debug, Clone)]
pub struct Function {
    accessed_vars: HashSet<ObjectId>,
    calls: HashSet<ObjectId>,
}

use crate::gfx::contract::{VertexAttributeContract, AttachmentContract, PipelineStageContract, DescriptorContract};

#[derive(Default, Debug)]
struct SpirvMetadata<'a> {
    pub entry_points: Vec<EntryPoint>,
    pub name_map: HashMap<(ObjectId, Option<u32>), &'a str>,
    pub deco_map: HashMap<(ObjectId, Option<u32>, Decoration), &'a [u32]>,
    pub obj_map: HashMap<ObjectId, SpirvObject<'a>>,
}
impl<'a> TryFrom<&'a SpirvBinary> for SpirvMetadata<'a> {
    type Error = Error;
    fn try_from(module: &'a SpirvBinary) -> Result<SpirvMetadata<'a>> {
        // Don't change the order. See _2.4 Logical Layout of a Module_ of the
        // SPIR-V specification for more information.
        let mut instrs = module.instrs().peekable();
        let mut meta = SpirvMetadata::default();
        meta.populate_entry_points(&mut instrs)?;
        meta.populate_names(&mut instrs)?;
        meta.populate_decos(&mut instrs)?;
        meta.populate_defs(&mut instrs)?;
        meta.populate_access(&mut instrs)?;
        for i in 0..meta.entry_points.len() { meta.inflate_entry_point(i)?; }
        Ok(meta)
    }
}
impl<'a> SpirvMetadata<'a> {
    fn insert_obj(&mut self, id: ObjectId, obj: SpirvObject<'a>) -> Result<()> {
        if self.obj_map.insert(id, obj).is_some() { return Err(Error::CorruptedSpirv); }
        Ok(())
    }
    fn get_num_ty(&self, id: ObjectId) -> Result<&NumericType> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::NumericType(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_img_ty(&self, id: ObjectId) -> Result<&Option<ImageType>> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::ImageType(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_arr_ty(&self, id: ObjectId) -> Result<&ArrayType> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::ArrayType(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_struct_ty(&self, id: ObjectId) -> Result<&Vec<StructMember<'a>>> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::StructType(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_ptr_ty(&self, id: ObjectId) -> Result<&ObjectId> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::PointerType(id) => Some(id), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_var(&self, id: ObjectId) -> Result<&Variable> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::Variable(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_const(&self, id: ObjectId) -> Result<&Constant<'a>> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::Constant(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_fn(&self, id: ObjectId) -> Result<&Function> {
        self.obj_map.get(&id)
            .and_then(|x| match x { SpirvObject::Function(ty) => Some(ty), _ => None })
            .ok_or(Error::CorruptedSpirv)
    }
    fn get_fn_or_default_mut(&mut self, id: ObjectId) -> Result<&mut Function> {
        let obj = self.obj_map.entry(id)
            .or_insert_with(|| SpirvObject::Function(Function::default()));
        match obj { SpirvObject::Function(ty) => Ok(ty), _ => Err(Error::CorruptedSpirv) }
    }

    fn populate_entry_points(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()>{
        // Extract entry points.
        while let Some(instr) = instrs.peek() {
            if instr.opcode() != OP_ENTRY_POINT { instrs.next(); } else { break; }
        }
        while let Some(instr) = instrs.peek() {
            if instr.opcode() != OP_ENTRY_POINT { break; }
            let mut operands = instr.operands();
            let entry_point = EntryPoint {
                exec_model: operands.read_u32()?,
                func: operands.read_u32()?,
                name: operands.read_str()?.to_owned(),
                ..Default::default()
            };
            self.entry_points.push(entry_point);
            instrs.next();
        }
        Ok(())
    }
    fn populate_names(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()> {
        // Extract naming. Names are generally produced as debug information by
        // `glslValidator` but it might be in absence.
        while let Some(instr) = instrs.peek() {
            if !NAME_RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
        }
        while let Some(instr) = instrs.peek() {
            let opcode = instr.opcode();
            if !NAME_RANGE.contains(&opcode) { break; }
            let mut operands = instr.operands();
            let target_id = operands.read_u32()?;
            let member_id = if opcode == OP_MEMBER_NAME { Some(operands.read_u32()?) } else { None };
            let name = operands.read_str()?;
            self.name_map.insert((target_id, member_id), name);
            instrs.next();
        }
        Ok(())
    }
    fn populate_decos(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()> {
        while let Some(instr) = instrs.peek() {
            if !DECO_RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
        }
        while let Some(instr) = instrs.peek() {
            let opcode = instr.opcode();
            if !DECO_RANGE.contains(&opcode) { break; }
            let mut operands = instr.operands();
            let target_id = operands.read_u32()?;
            let member_id = if opcode == OP_MEMBER_DECORATE { Some(operands.read_u32()?) } else { None };
            let deco = operands.read_u32()?;
            let params = operands.read_list()?;
            if self.deco_map.insert((target_id, member_id, deco), params).is_some() {
                return Err(Error::CorruptedSpirv);
            }
            instrs.next();
        }
        Ok(())
    }
    fn populate_one_ty(&mut self, instr: &Instr<'a>) -> Result<()> {
        let mut operands = instr.operands();
        let id = operands.read_u32()?;
        let opcode = instr.opcode();
        let ty = match opcode {
            OP_TYPE_VOID | OP_TYPE_FUNCTION => { return Ok(()) },
            OP_TYPE_BOOL => return Err(Error::UnsupportedSpirv),
            OP_TYPE_INT | OP_TYPE_FLOAT => {
                let nbyte = operands.read_u32()? >> 3;
                if nbyte != 4 { return Err(Error::UnsupportedSpirv) }
                let is_signed = if opcode == OP_TYPE_INT { Some(operands.read_bool()?) } else { None };
                let num_ty = match is_signed {
                    Some(true) => NumericType::i32(),
                    Some(false) => NumericType::u32(),
                    None => NumericType::f32(),
                };
                self.insert_obj(id, SpirvObject::NumericType(num_ty))?;
            },
            OP_TYPE_VECTOR | OP_TYPE_MATRIX => {
                let elem_ty = self.get_num_ty(operands.read_u32()?)?;
                let nelem = operands.read_u32()?;
                let num_ty = if opcode == OP_TYPE_VECTOR && elem_ty.is_primitive() {
                    NumericType::vec(&elem_ty, nelem)
                } else if opcode == OP_TYPE_MATRIX && elem_ty.is_vec() {
                    NumericType::mat(&elem_ty, nelem)
                } else { return Err(Error::CorruptedSpirv); };
                self.insert_obj(id, SpirvObject::NumericType(num_ty))?;
            },
            OP_TYPE_IMAGE => {
                let _unit_ty = operands.read_u32()?;
                let dim = operands.read_u32()?;
                if dim == DIM_IMAGE_SUBPASS_DATA {
                    self.insert_obj(id, SpirvObject::ImageType(None))?;
                } else {
                    let is_depth = match operands.read_u32()? {
                        0 => false, 1 => true, _ => return Err(Error::UnsupportedSpirv),
                    };
                    let is_array = operands.read_bool()?;
                    let is_multisampled = operands.read_bool()?;
                    let is_sampled = match operands.read_u32()? {
                        1 => true, 2 => false, _ => return Err(Error::UnsupportedSpirv),
                    };
                    let color_fmt = operands.read_u32()?;
                    
                    // Only unit types allowed to be stored in storage images can
                    // have given format.
                    let img_ty = ImageType {
                        fmt: ImageUnitFormat::from_spv_def(is_sampled, is_depth, color_fmt)?,
                        arng: ImageArrangement::from_spv_def(dim, is_array, is_multisampled)?,
                    };
                    self.insert_obj(id, SpirvObject::ImageType(Some(img_ty)))?;
                }
            },
            OP_TYPE_SAMPLED_IMAGE => {
                let ty = self.get_img_ty(operands.read_u32()?)?;
                self.insert_obj(id, SpirvObject::ImageType(*ty));
            },
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY => {
                let arr_ty = ArrayType {
                    elem_ty: operands.read_u32()?,
                    nelem: if instr.opcode() == OP_TYPE_ARRAY {
                        let nelem = self.get_const(operands.read_u32()?)
                            .and_then(|constant| {
                                let num_ty = self.get_num_ty(constant.ty)?;
                                if num_ty.nbyte == 4 && num_ty.is_uint() && num_ty.is_primitive() {
                                    if let Some(nelem) = constant.value.get(0) {
                                        return Ok(nelem);
                                    }
                                }
                                return Err(Error::CorruptedSpirv);
                            })?;
                        Some(*nelem)
                    } else { None },
                };
                self.insert_obj(id, SpirvObject::ArrayType(arr_ty))?;
            },
            OP_TYPE_STRUCT => {
                let ls = operands.read_list()?;
                let mut members = Vec::with_capacity(ls.len());
                for (i, member) in ls.iter().enumerate() {
                    let name = self.name_map.get(&(id, Some(i as u32)))
                        .map(|x| *x);
                    let offset = self.deco_map.get(&(id, Some(i as u32), DECO_OFFSET))
                        .and_then(|x| x.get(i))
                        .map(|x| *x as usize);
                    let member = StructMember {
                        ty: *member,
                        name: name,
                        offset: offset,
                    };
                    members.push(member);
                }
                self.insert_obj(id, SpirvObject::StructType(members))?;
            },
            OP_TYPE_POINTER => {
                let _store_cls = operands.read_u32()?;
                let target_ty = operands.read_u32()?;
                self.insert_obj(id, SpirvObject::PointerType(target_ty))?;
            },
            _ => return Err(Error::CorruptedSpirv),
        };
        Ok(())
    }
    fn populate_one_var(&mut self, instr: &Instr<'a>) -> Result<()> {
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
        self.insert_obj(id, SpirvObject::Variable(var))?;
        Ok(())
    }
    fn populate_one_const(&mut self, instr: &Instr<'a>) -> Result<()> {
        let mut operands = instr.operands();
        let ty = operands.read_u32()?;
        let id = operands.read_u32()?;
        let value = operands.read_list()?;
        let constant = Constant {
            ty: ty,
            value: value,
        };
        self.insert_obj(id, SpirvObject::Constant(constant))?;
        Ok(())
    }
    fn populate_defs(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()> {
        // type definitions always follow decorations, so we don't skip
        // instructions here.
        while let Some(instr) = instrs.peek() {
            let mut operands = instr.operands();
            let opcode = instr.opcode();
            if (TYPE_RANGE.contains(&opcode)) {
                self.populate_one_ty(instr)?;
            } else if opcode == OP_VARIABLE {
                self.populate_one_var(instr)?;
            } else if CONST_RANGE.contains(&opcode) {
                self.populate_one_const(instr)?;
            } else { break; }
            instrs.next();
        }
        Ok(())
    }
    fn populate_access(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()> {
        while instrs.peek().is_some() {
            let mut access_chain_map = HashMap::new();
            let mut func: &mut Function = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            while let Some(instr) = instrs.next() {
                if instr.opcode() == OP_FUNCTION {
                    let mut operands = instr.operands();
                    let _rty = operands.read_u32()?;
                    let id = operands.read_u32()?;
                    func = self.get_fn_or_default_mut(id)?;
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
                        if !func.calls.insert(callee) {
                            return Err(Error::CorruptedSpirv);
                        }
                    },
                    OP_LOAD => {
                        let mut operands = instr.operands();
                        let _rty = operands.read_u32()?;
                        let _id = operands.read_u32()?;
                        let mut target = operands.read_u32()?;
                        if let Some(&x) = access_chain_map.get(&target) {
                            target = x;
                        }
                        func.accessed_vars.insert(target);
                    },
                    OP_STORE => {
                        let mut operands = instr.operands();
                        let mut target = operands.read_u32()?;
                        if let Some(&x) = access_chain_map.get(&target) {
                            target = x;
                        }
                        func.accessed_vars.insert(target);
                    },
                    OP_ACCESS_CHAIN => {
                        let mut operands = instr.operands();
                        let _rty = operands.read_u32()?;
                        let id = operands.read_u32()?;
                        let target = operands.read_u32()?;
                        if access_chain_map.insert(id, target).is_some() {
                            return Err(Error::CorruptedSpirv);
                        }
                    },
                    OP_FUNCTION_END => break,
                    _ => { },
                }
            }
        }
        Ok(())
    }
    fn collect_fn_vars_impl(&self, func: ObjectId, vars: &mut HashSet<ObjectId>) {
        if let Ok(func) = self.get_fn(func) {
            let it = func.accessed_vars.iter()
                .filter(|x| self.obj_map.contains_key(x));
            vars.extend(it);
            for call in func.calls.iter() {
                self.collect_fn_vars_impl(*call, vars);
            }
        }
    }
    pub fn collect_fn_vars(&self, func: ObjectId) -> impl Iterator<Item=ObjectId> {
        let mut accessed_vars = HashSet::new();
        self.collect_fn_vars_impl(func, &mut accessed_vars);
        accessed_vars.into_iter()
    }
    /// Resolve recurring layers of pointers to the pointer that refer to the
    /// data directly.
    fn resolve_ref(&self, ty_id: ObjectId) -> Result<(ObjectId, &SpirvObject<'a>)> {
        let ty = &self.obj_map.get(&ty_id)
            .ok_or(Error::CorruptedSpirv)?;
        if let SpirvObject::PointerType(ref_ty) = ty {
            self.resolve_ref(*ref_ty)
        } else { Ok((ty_id, ty)) }
    }
    fn get_deco(&self, id: ObjectId, member_idx: Option<u32>, deco: Decoration) -> Option<&[u32]> {
        self.deco_map.get(&(id, member_idx, deco))
            .cloned()
    }
    fn get_deco_u32(&self, id: ObjectId, member_idx: Option<u32>, deco: Decoration) -> Option<u32> {
        self.deco_map.get(&(id, member_idx, deco))
            .and_then(|x| x.get(0))
            .cloned()
    }
    fn get_name(&self, id: ObjectId, member_idx: Option<u32>) -> Option<&'a str> {
        self.name_map.get(&(id, member_idx))
            .map(|x| *x)
    }
    fn ty2node(&self, ty_id: u32, mat_stride: usize, base_offset: usize) -> Result<SymbolNode> {
        let node = match &self.obj_map[&ty_id] {
            SpirvObject::NumericType(num_ty) => {
                if num_ty.is_mat() {
                    let col_nbyte = (num_ty.nrow() * num_ty.nbyte()) as usize;
                    let vec = SymbolNode::Leaf {
                        offset: base_offset,
                        nbyte: col_nbyte,
                    };
                    SymbolNode::Repeat {
                        is_mat: true,
                        offset: base_offset,
                        stride: mat_stride,
                        proto: Box::new(vec),
                        nrepeat: num_ty.ncol.map(|x| x as usize),
                    }
                } else {
                    SymbolNode::Leaf {
                        offset: base_offset,
                        nbyte: num_ty.nbyte() as usize,
                    }
                }
            },
            SpirvObject::StructType(members) => {
                let mut children = Vec::with_capacity(members.len());
                let mut name_map = HashMap::new();
                for (i, member_ty) in members.iter().enumerate() {
                    let offset = self.get_deco_u32(ty_id, Some(i as u32), DECO_OFFSET)
                        .ok_or(Error::UnsupportedSpirv)? as usize;
                    let stride = self.get_deco_u32(ty_id, Some(i as u32), DECO_MATRIX_STRIDE)
                        .map(|x| x as usize)
                        .unwrap_or(mat_stride);
                    children.push(self.ty2node(member_ty.ty, stride, base_offset + offset)?);
                    if let Some(name) = self.get_name(ty_id, Some(i as u32)) {
                        if name_map.insert(name.to_owned(), i).is_some() {
                            return Err(Error::CorruptedSpirv);
                        }
                    }
                }
                SymbolNode::Node {
                    offset: base_offset,
                    children: children,
                    name_map: name_map,
                }
            },
            SpirvObject::ArrayType(arr_ty) => {
                SymbolNode::Repeat {
                    is_mat: false,
                    offset: base_offset,
                    stride: mat_stride,
                    proto: Box::new(self.ty2node(arr_ty.elem_ty, mat_stride, 0)?),
                    nrepeat: arr_ty.nelem.map(|x| x as usize),
                }
            },
            _ => return Err(Error::CorruptedSpirv),
        };
        Ok(node)
    }
    fn inflate_entry_point(&mut self, i: usize) -> Result<()> {
        let entry_point = &self.entry_points[i];
        let exec_model = entry_point.exec_model;

        let mut attr_templates = HashMap::<u32, Vec<VertexAttributeContractTemplate>>::new();
        let mut attm_templates = Vec::new();
        let mut desc_binds = HashMap::new();

        for var_id in self.collect_fn_vars(entry_point.func) {
            let var = &self.get_var(var_id)?;
            let (ty_id, ty) = self.resolve_ref(var.ty)?;
            match var.store_cls {
                STORE_CLS_INPUT if exec_model == EXEC_MODEL_VERTEX => {
                    let bind_point = self.get_deco_u32(var_id, None, DECO_BINDING)
                        .unwrap_or(0);
                    let location = self.get_deco_u32(var_id, None, DECO_LOCATION)
                        .unwrap_or(0);
                    if let SpirvObject::NumericType(num_ty) = ty {
                        let col_nbyte = (num_ty.nbyte() * num_ty.nrow()) as usize;
                        let mut vec = &mut attr_templates.entry(bind_point)
                            .or_default();
                        let base_offset = vec.last()
                            .map(|x| x.offset)
                            .unwrap_or(0);
                        for i in 0..num_ty.ncol() {
                            let template = VertexAttributeContractTemplate {
                                bind_point: bind_point,
                                location: location,
                                offset: base_offset + col_nbyte,
                                nbyte: col_nbyte,
                            };
                            vec.push(template);
                        }
                    } else { return Err(Error::CorruptedSpirv); }
                },
                STORE_CLS_OUTPUT if exec_model == EXEC_MODEL_FRAGMENT => {
                    let mut location = self.get_deco_u32(var_id, None, DECO_LOCATION)
                        .unwrap_or(0);
                    if let SpirvObject::NumericType(num_ty) = ty {
                        // Matrix is not valid attachment type.
                        if num_ty.is_mat() { return Err(Error::CorruptedSpirv); }
                        let col_nbyte = (num_ty.nbyte() * num_ty.nrow()) as usize;
                        let template = AttachmentContractTemplate {
                            location: location,
                            nbyte: col_nbyte,
                        };
                        attm_templates.push(template);
                    } else { return Err(Error::CorruptedSpirv); }
                },
                STORE_CLS_UNIFORM | STORE_CLS_STORAGE_BUFFER => {
                    let desc_set = self.get_deco_u32(var_id, None, DECO_DESCRIPTOR_SET)
                        .unwrap_or(0);
                    let bind_point = self.get_deco_u32(var_id, None, DECO_BINDING)
                        .unwrap_or(0);
                    let sym_node = self.ty2node(ty_id, 0, 0)?;
                    let desc = if var.store_cls == STORE_CLS_STORAGE_BUFFER ||
                        self.get_deco(var_id, None, DECO_BUFFER_BLOCK).is_some() {
                        Descriptor::Storage(sym_node)
                    } else {
                        Descriptor::Uniform(sym_node)
                    };
                    
                    let desc_bind = DescriptorBinding::desc_bind(desc_set, bind_point);
                    if desc_binds.insert(desc_bind, desc).is_some() {
                        return Err(Error::CorruptedSpirv);
                    }
                },
                STORE_CLS_UNIFORM_CONSTANT => {
                    if let SpirvObject::ImageType(img_ty) = ty {
                        let desc_set = self.get_deco_u32(var_id, None, DECO_DESCRIPTOR_SET)
                            .unwrap_or(0);
                        let bind_point = self.get_deco_u32(var_id, None, DECO_BINDING)
                            .unwrap_or(0);
                        let desc = if let Some(img_ty) = img_ty {
                            Descriptor::Image(*img_ty)
                        } else {
                            let input_attm_idx = self.get_deco_u32(var_id, None, DECO_INPUT_ATTACHMENT_INDEX)
                                .ok_or(Error::CorruptedSpirv)?;
                            Descriptor::InputAtatchment(input_attm_idx)
                        };
                        let desc_bind = DescriptorBinding::desc_bind(desc_set, bind_point);
                        if desc_binds.insert(desc_bind, desc).is_some() {
                            return Err(Error::CorruptedSpirv);
                        }
                    }
                    // Leak out unknown types of uniform constants.
                }
                STORE_CLS_PUSH_CONSTANT => {
                    // Push constants have no global offset. Offsets are applied to
                    // members.
                    let sym_node = self.ty2node(ty_id, 0, 0)?;
                    let desc = Descriptor::Uniform(sym_node);

                    let desc_bind = DescriptorBinding::push_const();
                    if desc_binds.insert(desc_bind, desc).is_some() {
                        return Err(Error::CorruptedSpirv);
                    }
                },
                _ => {},
            }
        }
        let mut entry_point = &mut self.entry_points[i];
        entry_point.attr_templates = attr_templates;
        entry_point.attm_templates = attm_templates;
        entry_point.desc_binds = desc_binds;
        Ok(())
    }
    pub fn entry_points(&self) -> &[EntryPoint] { &self.entry_points }
}

#[derive(Debug, Clone)]
struct VertexAttributeContractTemplate {
    bind_point: u32,
    location: u32,
    /// Offset in each set of vertex data.
    offset: usize,
    /// Total byte count at this location.
    nbyte: usize,
}
#[derive(Debug, Clone)]
struct AttachmentContractTemplate {
    location: u32,
    /// Total byte count at this location.
    nbyte: usize,
}

#[derive(Debug, Clone)]
pub enum SymbolNode {
    Node {
        offset: usize,
        children: Vec<SymbolNode>,
        name_map: HashMap<String, usize>,
    },
    Repeat {
        is_mat: bool,
        offset: usize,
        stride: usize,
        proto: Box<SymbolNode>,
        nrepeat: Option<usize>,
    },
    Leaf {
        offset: usize,
        nbyte: usize,
    },
}

#[derive(PartialEq, Eq, Hash, Default, Clone, Copy)]
pub struct DescriptorBinding(Option<(u32, u32)>);
impl DescriptorBinding {
    pub fn push_const() -> Self { DescriptorBinding(None) }
    pub fn desc_bind(desc_set: u32, bind_point: u32) -> Self { DescriptorBinding(Some((desc_set, bind_point))) }

    pub fn is_push_const(&self) -> bool { self.0.is_none() }
    pub fn is_desc_bind(&self) -> bool { self.0.is_some() }
    pub fn into_inner(self) -> Option<(u32, u32)> { self.0 }
}
impl fmt::Debug for DescriptorBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some((set, bind)) = self.0 {
            write!(f, "(Set = {}, Bind = {})", set, bind)
        } else {
            write!(f, "(PushConstant)")
        }
    }
}

#[derive(Debug, Clone)]
pub enum Descriptor {
    Storage(SymbolNode),
    Uniform(SymbolNode),
    Image(ImageType),
    Sampler,
    InputAtatchment(u32),
}

#[derive(Debug, Default, Clone)]
struct EntryPoint {
    func: u32,
    name: String,
    exec_model: u32,
    attr_templates: HashMap<u32, Vec<VertexAttributeContractTemplate>>,
    attm_templates: Vec<AttachmentContractTemplate>,
    desc_binds: HashMap<DescriptorBinding, Descriptor>,
}

pub fn module_lab(module: &SpirvBinary) -> crate::gfx::Result<()> {
    use std::ops::Deref;
    use log::debug;
    let meta: SpirvMetadata = module.try_into()?;
    debug!("{:#?}", meta.entry_points());

    Ok(())
}
