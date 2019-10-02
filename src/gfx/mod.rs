use std::result;

mod error;
mod ctxt;

pub use error::*;
pub use ctxt::*;

type Result<T> = result::Result<T, Error>;

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