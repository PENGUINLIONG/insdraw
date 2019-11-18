//! SPIR-V Reflection
//!
//! Reflect and extract SPIR-V declared materials.
mod consts;
mod parse;
mod reflect;
mod error;

use consts::*;
pub use parse::SpirvBinary;
use parse::{Instrs, Instr, Operands};
pub use reflect::SpirvMetadata;
pub use error::Error;

type Result<T> = std::result::Result<T, Error>;

pub fn module_lab(module: &SpirvBinary) -> crate::spv::Result<()> {
    use std::convert::TryInto;
    use std::ops::Deref;
    use log::debug;
    let meta: SpirvMetadata = module.try_into()?;
    debug!("{:#?}", meta.entry_points());

    Ok(())
}
