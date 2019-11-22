use std::convert::{TryFrom, TryInto};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::Vacant;
use std::iter::Peekable;
use std::fmt;
use super::consts::*;
use super::instr::*;
use super::parse::{SpirvBinary, Instrs, Instr};
use super::{Error, Result};
use log::debug;

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
    pub fn int(nbyte: u32, is_signed: bool) -> NumericType {
        NumericType {
            nbyte: 4,
            is_signed: Some(is_signed),
            ..Default::default()
        }
    }
    pub fn float(nbyte: u32) -> NumericType {
        NumericType {
            nbyte: 4,
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
#[derive(Debug, Hash, Clone, Copy)]
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
#[derive(Debug, Hash, Clone, Copy)]
pub enum ImageUnitFormat {
    Color(ColorFormat),
    Sampled,
    Depth,
}
impl ImageUnitFormat {
    pub fn from_spv_def(is_sampled: u32, is_depth: u32, color_fmt: u32) -> Result<ImageUnitFormat> {
        let img_unit_fmt = match (is_sampled, is_depth, color_fmt) {
            (1, 0, _) => ImageUnitFormat::Sampled,
            (1, 1, _) => ImageUnitFormat::Depth,
            (2, 0, color_fmt) => ImageUnitFormat::Color(ColorFormat::from_spv_def(color_fmt)?),
            _ => return Err(Error::UnsupportedSpirv),
        };
        Ok(img_unit_fmt)
    }
}
#[derive(Debug, Hash, Clone, Copy)]
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
#[derive(Debug, Hash, Clone)]
pub struct ImageType {
    fmt: ImageUnitFormat,
    arng: ImageArrangement,
}
#[derive(Debug, Clone)]
struct ArrayType {
    elem_ty: InstrId,
    nelem: Option<u32>,
}
#[derive(Debug, Clone)]
struct StructMember<'a> {
    ty: InstrId,
    name: Option<&'a str>,
    offset: Option<usize>,
}
#[derive(Debug, Clone)]
struct Variable {
    ty: InstrId,
    store_cls: StorageClass,
}
#[derive(Debug, Clone)]
struct Constant<'a> {
    ty: InstrId,
    value: &'a [u32],
}
#[derive(Default, Debug, Clone)]
struct Function {
    accessed_vars: HashSet<InstrId>,
    calls: HashSet<InstrId>,
}

#[derive(Debug, Clone)]
enum Type<'a> {
    Numeric(NumericType),
    Image(Option<ImageType>),
    Array(ArrayType),
    Struct(Vec<StructMember<'a>>),
    Pointer(InstrId), // Struct ID.
}


#[derive(Default, Debug)]
pub struct SpirvMetadata<'a> {
    entry_points: Vec<EntryPoint>,
    name_map: HashMap<(InstrId, Option<u32>), &'a str>,
    deco_map: HashMap<(InstrId, Option<u32>, Decoration), &'a [u32]>,
    ty_map: HashMap<TypeId, Type<'a>>,
    rsc_map: HashMap<ResourceId, Variable>,
    const_map: HashMap<ConstantId, Constant<'a>>,
    func_map: HashMap<FunctionId, Function>,
}
impl<'a> TryFrom<&'a SpirvBinary> for SpirvMetadata<'a> {
    type Error = Error;
    fn try_from(module: &'a SpirvBinary) -> Result<SpirvMetadata<'a>> {
        use log::debug;
        // Don't change the order. See _2.4 Logical Layout of a Module_ of the
        // SPIR-V specification for more information.
        let mut instrs = module.instrs().peekable();
        let mut meta = SpirvMetadata::default();
        meta.populate_entry_points(&mut instrs)?;
        meta.populate_names(&mut instrs)?;
        meta.populate_decos(&mut instrs)?;
        meta.populate_defs(&mut instrs)?;
        meta.populate_access(&mut instrs)?;
        for i in 0..meta.entry_points.len() {meta.inflate_entry_point(i)?; }
        Ok(meta)
    }
}
impl<'a> SpirvMetadata<'a> {
    fn populate_entry_points(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()>{
        // Extract entry points.
        while let Some(instr) = instrs.peek() {
            if instr.opcode() != OP_ENTRY_POINT { instrs.next(); } else { break; }
        }
        while let Some(instr) = instrs.peek() {
            if instr.opcode() != OP_ENTRY_POINT { break; }
            let op = OpEntryPoint::try_from(instr)?;
            let entry_point = EntryPoint {
                exec_model: op.exec_model,
                func: op.func_id,
                name: op.name.to_string(),
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
            let (key, value) = match instr.opcode() {
                OP_NAME => {
                    let op = OpName::try_from(instr)?;
                    ((op.target_id, None), op.name)
                },
                OP_MEMBER_NAME => {
                    let op = OpMemberName::try_from(instr)?;
                    ((op.target_id, Some(op.member_idx)), op.name)
                },
                _ => break,
            };
            let collision = self.name_map.insert(key, value);
            if collision.is_some() { return Err(Error::CorruptedSpirv); }
            instrs.next();
        }
        Ok(())
    }
    fn populate_decos(&mut self, instrs: &'_ mut Peekable<Instrs<'a>>) -> Result<()> {
        while let Some(instr) = instrs.peek() {
            if !DECO_RANGE.contains(&instr.opcode()) { instrs.next(); } else { break; }
        }
        while let Some(instr) = instrs.peek() {
            let (key, value) = match instr.opcode() {
                OP_DECORATE => {
                    let op = OpDecorate::try_from(instr)?;
                    ((op.target_id, None, op.deco), op.params)
                }
                OP_MEMBER_DECORATE => {
                    let op = OpMemberDecorate::try_from(instr)?;
                    ((op.target_id, Some(op.member_idx), op.deco), op.params)
                },
                _ => break,
            };
            let collision = self.deco_map.insert(key, value);
            if collision.is_some() { return Err(Error::CorruptedSpirv); }
            instrs.next();
        }
        Ok(())
    }
    fn populate_one_ty(&mut self, instr: &Instr<'a>) -> Result<()> {
        let (key, value) = match instr.opcode() {
            OP_TYPE_VOID | OP_TYPE_FUNCTION => { return Ok(()) },
            OP_TYPE_BOOL => return Err(Error::UnsupportedSpirv),
            OP_TYPE_INT => {
                let op = OpTypeInt::try_from(instr)?;
                let num_ty = NumericType::int(op.nbyte >> 3, op.is_signed);
                (op.ty_id, Type::Numeric(num_ty))
            }
            OP_TYPE_FLOAT => {
                let op = OpTypeFloat::try_from(instr)?;
                let num_ty = NumericType::float(op.nbyte >> 3);
                (op.ty_id, Type::Numeric(num_ty))
            },
            OP_TYPE_VECTOR => {
                let op = OpTypeVector::try_from(instr)?;
                if let Some(Type::Numeric(num_ty)) = self.ty_map.get(&op.num_ty_id) {
                    let vec_ty = NumericType::vec(&num_ty, op.nnum);
                    (op.ty_id, Type::Numeric(vec_ty))
                } else { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_MATRIX => {
                let op = OpTypeMatrix::try_from(instr)?;
                if let Some(Type::Numeric(vec_ty)) = self.ty_map.get(&op.vec_ty_id) {
                    let mat_ty = NumericType::mat(&vec_ty, op.nvec);
                    (op.ty_id, Type::Numeric(mat_ty))
                } else { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_IMAGE => {
                let op = OpTypeImage::try_from(instr)?;
                let img_ty = if op.dim == DIM_IMAGE_SUBPASS_DATA {
                    Type::Image(None)
                } else {
                    // Only unit types allowed to be stored in storage images can
                    // have given format.
                    let fmt = ImageUnitFormat::from_spv_def(op.is_sampled, op.is_depth, op.color_fmt)?;
                    let arng = ImageArrangement::from_spv_def(op.dim, op.is_array, op.is_multisampled)?;
                    let img_ty = ImageType { fmt: fmt, arng: arng };
                    Type::Image(Some(img_ty))
                };
                (op.ty_id, img_ty)
            },
            OP_TYPE_SAMPLED_IMAGE => {
                let op = OpTypeSampledImage::try_from(instr)?;
                if let Some(Type::Image(img_ty)) = self.ty_map.get(&op.img_ty_id) {
                    (op.ty_id, Type::Image(img_ty.clone()))
                } else { return Err(Error::CorruptedSpirv); }
            },
            OP_TYPE_ARRAY => {
                let op = OpTypeArray::try_from(instr)?;
                let nrepeat = self.const_map.get(&op.nrepeat_const_id)
                    .and_then(|constant| {
                        if let Some(Type::Numeric(num_ty)) = self.ty_map.get(&constant.ty) {
                            if num_ty.nbyte == 4 && num_ty.is_uint() && num_ty.is_primitive() {
                                return Some(constant.value[0]);
                            }
                        }
                        None
                    })
                    .ok_or(Error::CorruptedSpirv)?;
                let arr_ty = ArrayType { elem_ty: op.proto_ty_id, nelem: Some(nrepeat) };
                (op.ty_id, Type::Array(arr_ty))
            },
            OP_TYPE_RUNTIME_ARRAY => {
                let op = OpTypeRuntimeArray::try_from(instr)?;
                let arr_ty = ArrayType { elem_ty: op.proto_ty_id, nelem: None };
                (op.ty_id, Type::Array(arr_ty))
            }
            OP_TYPE_STRUCT => {
                let op = OpTypeStruct::try_from(instr)?;
                let mut members = Vec::with_capacity(op.member_ty_ids.len());
                for (i, &member_ty_id) in op.member_ty_ids.iter().enumerate() {
                    let name = self.name_map.get(&(op.ty_id, Some(i as u32)))
                        .map(|x| *x);
                    let offset = self.deco_map.get(&(op.ty_id, Some(i as u32), DECO_OFFSET))
                        .and_then(|x| x.get(i))
                        .map(|x| *x as usize);
                    let member = StructMember {
                        ty: member_ty_id,
                        name: name,
                        offset: offset,
                    };
                    members.push(member);
                }
                (op.ty_id, Type::Struct(members))
            },
            OP_TYPE_POINTER => {
                let op = OpTypePointer::try_from(instr)?;
                (op.ty_id, Type::Pointer(op.target_ty_id))
            },
            _ => return Err(Error::CorruptedSpirv),
        };
        if let Vacant(entry) = self.ty_map.entry(key) {
            entry.insert(value); Ok(())
        } else { Err(Error::CorruptedSpirv) }
    }
    fn populate_one_var(&mut self, instr: &Instr<'a>) -> Result<()> {
        let op = OpVariable::try_from(instr)?;
        let var = Variable { ty: op.ty_id, store_cls: op.store_cls };
        if let Vacant(entry) = self.rsc_map.entry(op.alloc_id) {
            entry.insert(var); Ok(())
        } else { Err(Error::CorruptedSpirv) }
    }
    fn populate_one_const(&mut self, instr: &Instr<'a>) -> Result<()> {
        let op = OpConstant::try_from(instr)?;
        let constant = Constant { ty: op.ty_id, value: op.value };
        if let Vacant(entry) = self.const_map.entry(op.const_id) {
            entry.insert(constant); Ok(())
        } else { Err(Error::CorruptedSpirv) }
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
            while let Some(instr) = instrs.peek() {
                if instr.opcode() == OP_FUNCTION {
                    let op = OpFunction::try_from(instr)?;
                    func = self.func_map.entry(op.func_id).or_default();
                    break;
                }
                instrs.next();
            }
            while let Some(instr) = instrs.peek() {
                match instr.opcode() {
                    OP_FUNCTION_CALL => {
                        let op = OpFunctionCall::try_from(instr)?;
                        if !func.calls.insert(op.func_id) {
                            return Err(Error::CorruptedSpirv);
                        }
                    },
                    OP_LOAD => {
                        let op = OpLoad::try_from(instr)?;
                        let mut rsc_id = op.rsc_id;
                        if let Some(&x) = access_chain_map.get(&rsc_id) { rsc_id = x }
                        func.accessed_vars.insert(rsc_id);
                    },
                    OP_STORE => {
                        let op = OpStore::try_from(instr)?;
                        let mut rsc_id = op.rsc_id;
                        if let Some(&x) = access_chain_map.get(&rsc_id) { rsc_id = x }
                        func.accessed_vars.insert(rsc_id);
                    },
                    OP_ACCESS_CHAIN => {
                        let op = OpAccessChain::try_from(instr)?;
                        if access_chain_map.insert(op.rsc_id, op.accessed_rsc_id).is_some() {
                            return Err(Error::CorruptedSpirv);
                        }
                    },
                    OP_FUNCTION_END => break,
                    _ => { },
                }
                instrs.next();
            }
        }
        Ok(())
    }
    fn collect_fn_vars_impl(&self, func: FunctionId, vars: &mut HashSet<ResourceId>) {
        if let Some(func) = self.func_map.get(&func) {
            let it = func.accessed_vars.iter()
                .filter(|x| self.rsc_map.contains_key(x));
            vars.extend(it);
            for call in func.calls.iter() {
                self.collect_fn_vars_impl(*call, vars);
            }
        }
    }
    pub fn collect_fn_vars(&self, func: FunctionId) -> impl Iterator<Item=ResourceId> {
        let mut accessed_vars = HashSet::new();
        self.collect_fn_vars_impl(func, &mut accessed_vars);
        accessed_vars.into_iter()
    }
    /// Resolve recurring layers of pointers to the pointer that refer to the
    /// data directly.
    fn resolve_ref(&self, ty_id: TypeId) -> Result<(TypeId, &Type<'a>)> {
        let ty = &self.ty_map.get(&ty_id)
            .ok_or(Error::CorruptedSpirv)?;
        if let Type::Pointer(ref_ty) = ty {
            self.resolve_ref(*ref_ty)
        } else { Ok((ty_id, ty)) }
    }
    fn get_deco(&self, id: InstrId, member_idx: Option<u32>, deco: Decoration) -> Option<&[u32]> {
        self.deco_map.get(&(id, member_idx, deco))
            .cloned()
    }
    fn get_deco_u32(&self, id: InstrId, member_idx: Option<u32>, deco: Decoration) -> Option<u32> {
        self.deco_map.get(&(id, member_idx, deco))
            .and_then(|x| x.get(0))
            .cloned()
    }
    fn get_name(&self, id: InstrId, member_idx: Option<u32>) -> Option<&'a str> {
        self.name_map.get(&(id, member_idx))
            .map(|x| *x)
    }
    fn ty2node(&self, ty_id: u32, mat_stride: usize, base_offset: usize) -> Result<SymbolNode> {
        let ty = self.ty_map.get(&ty_id)
            .ok_or(Error::CorruptedSpirv)?;
        let node = match ty {
            Type::Numeric(num_ty) => {
                if num_ty.is_mat() {
                    let col_nbyte = (num_ty.nrow() * num_ty.nbyte()) as usize;
                    let vec = SymbolNode::Leaf {
                        offset: base_offset,
                        nbyte: col_nbyte,
                    };
                    SymbolNode::Repeat {
                        major: Some(MatrixAxisOrder::ColumnMajor),
                        offset: base_offset,
                        stride: Some(mat_stride),
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
            Type::Struct(members) => {
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
            Type::Array(arr_ty) => {
                let stride = self.get_deco_u32(ty_id, None, DECO_ARRAY_STRIDE)
                    .map(|x| x as usize);
                SymbolNode::Repeat {
                    major: None,
                    offset: base_offset,
                    stride: stride,
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

        let mut attr_templates = Vec::new();
        let mut attm_templates = Vec::new();
        let mut desc_binds = HashMap::new();
        let mut desc_name_map = HashMap::new();

        for var_id in self.collect_fn_vars(entry_point.func) {
            let var = &self.rsc_map.get(&var_id)
                .ok_or(Error::CorruptedSpirv)?;
            let (ty_id, ty) = self.resolve_ref(var.ty)?;
            let desc_set = self.get_deco_u32(var_id, None, DECO_DESCRIPTOR_SET)
                .unwrap_or(0);
            let bind_point = self.get_deco_u32(var_id, None, DECO_BINDING)
                .unwrap_or(0);
            let location = self.get_deco_u32(var_id, None, DECO_LOCATION)
                .unwrap_or(0);
            match var.store_cls {
                STORE_CLS_INPUT if exec_model == EXEC_MODEL_VERTEX => {
                    if let Type::Numeric(num_ty) = ty {
                        let col_nbyte = (num_ty.nbyte() * num_ty.nrow()) as usize;
                        let base_offset = attr_templates.last()
                            .map(|x: &VertexAttributeContractTemplate| x.offset)
                            .unwrap_or(0);
                        for i in 0..num_ty.ncol() {
                            let template = VertexAttributeContractTemplate {
                                bind_point: bind_point,
                                location: location,
                                offset: base_offset + col_nbyte,
                                nbyte: col_nbyte,
                            };
                            attr_templates.push(template);
                        }
                    } else { return Err(Error::CorruptedSpirv); }
                },
                STORE_CLS_OUTPUT if exec_model == EXEC_MODEL_FRAGMENT => {
                    if let Type::Numeric(num_ty) = ty {
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
                    if let Type::Image(img_ty) = ty {
                        let desc = if let Some(img_ty) = img_ty {
                            Descriptor::Image(img_ty.clone())
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
            if let Some(name) = self.get_name(var_id, None) {
                let desc_bind = DescriptorBinding::desc_bind(desc_set, bind_point);
                if !name.is_empty() &&
                    desc_name_map.insert(name.to_owned(), desc_bind).is_some() {
                    return Err(Error::CorruptedSpirv);
                }
            }
        }
        let mut entry_point = &mut self.entry_points[i];
        entry_point.attr_templates = attr_templates;
        entry_point.attm_templates = attm_templates;
        entry_point.desc_binds = desc_binds;
        entry_point.desc_name_map = desc_name_map;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VertexAttributeContractTemplate {
    bind_point: u32,
    location: u32,
    /// Offset in each set of vertex data.
    offset: usize,
    /// Total byte count at this location.
    nbyte: usize,
}
#[derive(Debug, Clone)]
pub struct AttachmentContractTemplate {
    location: u32,
    /// Total byte count at this location.
    nbyte: usize,
}

#[derive(Debug, Hash, Clone, Copy)]
pub enum MatrixAxisOrder {
    ColumnMajor,
    RowMajor,
}
impl Default for MatrixAxisOrder {
    fn default() -> MatrixAxisOrder { MatrixAxisOrder::ColumnMajor }
}
#[derive(Debug, Clone)]
pub enum SymbolNode {
    Node {
        offset: usize,
        children: Vec<SymbolNode>,
        name_map: HashMap<String, usize>,
    },
    Repeat {
        /// The axis order of a matrix. If the field is kept `None`, the repeat
        /// represents an array.
        major: Option<MatrixAxisOrder>,
        offset: usize,
        /// This field can be `None` when it represents the number of bindings
        /// at a binding point.
        stride: Option<usize>,
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
            write!(f, "(set={}, bind={})", set, bind)
        } else {
            write!(f, "(push_constant)")
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
    attr_templates: Vec<VertexAttributeContractTemplate>,
    attm_templates: Vec<AttachmentContractTemplate>,
    desc_binds: HashMap<DescriptorBinding, Descriptor>,
    desc_name_map: HashMap<String, DescriptorBinding>,
}

#[derive(Debug, Default)]
pub struct PipelineMetadata {
    attr_templates: Vec<VertexAttributeContractTemplate>,
    attm_templates: Vec<AttachmentContractTemplate>,
    desc_binds: HashMap<DescriptorBinding, Descriptor>,
    desc_name_map: HashMap<String, DescriptorBinding>,
}
impl PipelineMetadata {
    pub fn new(spvs: &[SpirvBinary]) -> Result<PipelineMetadata> {
        use std::convert::TryInto;
        use log::debug;
        let mut found_stages = HashSet::new();
        let mut meta = PipelineMetadata::default();
        for spv in spvs {
            let spv_meta: SpirvMetadata = spv.try_into()?;
            for entry_point in spv_meta.entry_points {
                let EntryPoint {
                    func, name, exec_model,
                    attr_templates, attm_templates,
                    desc_binds, desc_name_map
                } = entry_point;
                if !found_stages.insert(entry_point.exec_model) {
                    // Stage collision.
                    return Err(Error::MalformedPipeline);
                }

                match entry_point.exec_model {
                    EXEC_MODEL_VERTEX => meta.attr_templates = attr_templates,
                    EXEC_MODEL_FRAGMENT => meta.attm_templates = attm_templates,
                    _ => {},
                }
                // TODO: (pengunliong) Resolve structural and naming conflicts.
                for (desc_bind, desc) in desc_binds.into_iter() {
                    meta.desc_binds.entry(desc_bind).or_insert(desc);
                }
                for (name, desc_bind) in desc_name_map.into_iter() {
                    meta.desc_name_map.entry(name).or_insert(desc_bind);
                }
            }
        }
        Ok(meta)
    }
}
