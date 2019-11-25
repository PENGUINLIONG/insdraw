//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
mod consts;
mod parse;
mod instr;
mod reflect;
mod error;
mod sym;

use std::convert::TryInto;
use std::iter::FromIterator;
use consts::*;
use parse::{Instrs, Instr, Operands};
pub use reflect::*;
pub use error::Error;
pub use sym::*;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Default, Clone)]
pub struct SpirvBinary(Vec<u32>);
impl From<Vec<u32>> for SpirvBinary {
    fn from(x: Vec<u32>) -> Self { SpirvBinary(x) }
}
impl FromIterator<u32> for SpirvBinary {
    fn from_iter<I: IntoIterator<Item=u32>>(iter: I) -> Self { SpirvBinary(iter.into_iter().collect::<Vec<u32>>()) }
}
impl From<Vec<u8>> for SpirvBinary {
    fn from(x: Vec<u8>) -> Self {
        if x.len() == 0 { return SpirvBinary::default(); }
        x.chunks_exact(4)
            .map(|x| x.try_into().unwrap())
            .map(match x[0] {
                0x03 => u32::from_le_bytes,
                0x07 => u32::from_be_bytes,
                _ => return SpirvBinary::default(),
            })
            .collect::<SpirvBinary>()
    }
}

impl SpirvBinary {
    pub fn instrs<'a>(&'a self) -> Instrs<'a> { Instrs::new(&self.0) }
    pub fn reflect(&self) -> Result<Box<[EntryPoint]>> {
        reflect::reflect_spirv(&self)
    }
}
