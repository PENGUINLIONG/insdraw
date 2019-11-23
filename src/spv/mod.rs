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
pub use parse::SpirvBinary;
use parse::{Instrs, Instr, Operands};
//pub use reflect::PipelineMetadata;
pub use reflect::SpirvMetadata;
pub use error::Error;
pub use sym::*;

type Result<T> = std::result::Result<T, Error>;
