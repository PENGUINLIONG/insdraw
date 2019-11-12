use std::result;

mod error;
mod ctxt;
mod spv;
mod sym;
pub mod contract;

pub use error::*;
pub use ctxt::*;
pub use spv::*;
pub use sym::*;

pub type Result<T> = result::Result<T, Error>;
