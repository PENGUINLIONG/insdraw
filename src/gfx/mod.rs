use std::result;

mod error;
mod ctxt;
mod spv;

pub use error::*;
pub use ctxt::*;
pub use spv::*;

pub type Result<T> = result::Result<T, Error>;

/*
pub struct RenderConfig {
    Pipeline: Enumerate;
}
impl RenderConfig {
    pub fn new() -> RenderConfig {
        RenderConfig {
            instance: instance
        }
    }
}

pub struct PipelineInterface {
    pl: Pipeline;
}
impl PipelineInterface {

}
*/