//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
mod consts;
mod parse;
mod instr;
mod reflect;
mod error;
mod sym;

use consts::*;
use parse::{Instrs, Instr, Operands};
pub use reflect::*;
pub use error::Error;
pub use sym::*;
use std::iter::FromIterator;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct SpirvBinary(Vec<u32>);
impl From<Vec<u32>> for SpirvBinary {
    fn from(x: Vec<u32>) -> Self { SpirvBinary(x) }
}
impl FromIterator<u32> for SpirvBinary {
    fn from_iter<I: IntoIterator<Item=u32>>(iter: I) -> Self { SpirvBinary(iter.into_iter().collect::<Vec<u32>>()) }
}

impl SpirvBinary {
    pub fn instrs<'a>(&'a self) -> Instrs<'a> { Instrs::new(&self.0) }
    pub fn reflect(&self) -> Result<Box<[EntryPoint]>> {
        reflect::reflect_spirv(&self)
    }
}
