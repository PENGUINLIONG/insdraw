use std::ops::{Range, RangeInclusive};

pub const OP_ENTRY_POINT: u32 = 15;

pub const OP_NAME: u32 = 5;
pub const OP_MEMBER_NAME: u32 = 6;
pub const NAME_RANGE: RangeInclusive<u32> = OP_NAME..=OP_MEMBER_NAME;

pub const OP_DECORATE: u32 = 71;
pub const OP_MEMBER_DECORATE: u32 = 72;
pub const DECO_RANGE: RangeInclusive<u32> = OP_DECORATE..=OP_MEMBER_DECORATE;

// Don't need this: Not a resource type. But kept for the range.
pub const OP_TYPE_VOID: u32 = 19;
pub const OP_TYPE_BOOL: u32 = 20;
pub const OP_TYPE_INT: u32 = 21;
pub const OP_TYPE_FLOAT: u32 = 22;
pub const OP_TYPE_VECTOR: u32 = 23;
pub const OP_TYPE_MATRIX: u32 = 24;
pub const OP_TYPE_IMAGE: u32 = 25;
// Not in GLSL.
// pub const OP_TYPE_SAMPLER: u32 = 26;
pub const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
pub const OP_TYPE_ARRAY: u32 = 28;
pub const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
pub const OP_TYPE_STRUCT: u32 = 30;
pub const OP_TYPE_POINTER: u32 = 32;
// Don't need this: Not a resource type. But kept for the range.
pub const OP_TYPE_FUNCTION: u32 = 33;
pub const TYPE_RANGE: RangeInclusive<u32> = OP_TYPE_VOID..=OP_TYPE_FUNCTION;

pub const OP_CONSTANT_TRUE: u32 = 41;
pub const OP_CONSTANT_FALSE: u32 = 42;
pub const OP_CONSTANT: u32 = 43;
pub const OP_CONSTANT_COMPOSITE: u32 = 44;
pub const OP_CONSTANT_SAMPLER: u32 = 45;
pub const OP_CONSTANT_NULL: u32 = 46;
pub const CONST_RANGE: RangeInclusive<u32> = OP_CONSTANT_TRUE..=OP_CONSTANT_NULL;

pub const OP_SPEC_CONSTANT_TRUE: u32 = 48;
pub const OP_SPEC_CONSTANT_FALSE: u32 = 49;
pub const OP_SPEC_CONSTANT: u32 = 50;
pub const OP_SPEC_CONSTANT_COMPOSITE: u32 = 51;
pub const OP_SPEC_CONSTANT_OP: u32 = 52;
pub const SPEC_CONST_RANGE: RangeInclusive<u32> = OP_SPEC_CONSTANT_TRUE..=OP_SPEC_CONSTANT_OP;

pub const OP_VARIABLE: u32 = 59;

pub const OP_FUNCTION: u32 = 54;
pub const OP_FUNCTION_END: u32 = 56;
pub const OP_FUNCTION_CALL: u32 = 57;
pub const OP_ACCESS_CHAIN: u32 = 65;
pub const OP_LOAD: u32 = 61;
pub const OP_STORE: u32 = 62;
pub const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;


pub const EXEC_MODEL_VERTEX: u32 = 0;
pub const EXEC_MODEL_FRAGMENT: u32 = 4;


pub const DECO_SPEC_ID: u32 = 1;
pub const DECO_BLOCK: u32 = 2;
pub const DECO_BUFFER_BLOCK: u32 = 3;
pub const DECO_ROW_MAJOR: u32 = 4;
pub const DECO_COL_MAJOR: u32 = 5;
pub const DECO_ARRAY_STRIDE: u32 = 6;
pub const DECO_MATRIX_STRIDE: u32 = 7;
// Don't need this: Built-in variables will not be attribute nor attachment.
// pub const DECO_BUILT_IN: u32 = 11;
pub const DECO_LOCATION: u32 = 30;
pub const DECO_BINDING: u32 = 33;
pub const DECO_DESCRIPTOR_SET: u32 = 34;
pub const DECO_OFFSET: u32 = 35;
pub const DECO_INPUT_ATTACHMENT_INDEX: u32 = 43;


pub const STORE_CLS_UNIFORM_CONSTANT: u32 = 0;
pub const STORE_CLS_INPUT: u32 = 1;
pub const STORE_CLS_UNIFORM: u32 = 2;
pub const STORE_CLS_OUTPUT: u32 = 3;
// Texture calls to sampler object will translate to function class.
pub const STORE_CLS_FUNCTION: u32 = 7;
pub const STORE_CLS_PUSH_CONSTANT: u32 = 9;
pub const STORE_CLS_STORAGE_BUFFER: u32 = 12;


pub const DIM_IMAGE_1D: u32 = 0;
pub const DIM_IMAGE_2D: u32 = 1;
pub const DIM_IMAGE_3D: u32 = 2;
pub const DIM_IMAGE_CUBE: u32 = 3;
pub const DIM_IMAGE_SUBPASS_DATA: u32 = 6;


pub const IMG_UNIT_FMT_RGBA32F: u32 = 1;
pub const IMG_UNIT_FMT_R32F: u32 = 3;
pub const IMG_UNIT_FMT_RGBA8: u32 = 4;
