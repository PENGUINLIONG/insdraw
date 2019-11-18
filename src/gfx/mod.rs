use std::result;

mod error;
mod ctxt;
pub mod contract;

pub use error::*;
pub use ctxt::*;

pub type Result<T> = result::Result<T, Error>;
