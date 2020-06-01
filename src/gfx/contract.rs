//! Memory sizing and arrangement contract checked against to ensure a basic
//! level of memory usage conformance.
use std::ops::Range;
use ash::vk::{Format, VertexInputRate, AttachmentLoadOp, AttachmentStoreOp};

pub struct RenderPassContract {
    /// Pipeline stages sorted in order, with each subpass started by a
    /// vertex shader and ended by a fragment shader.
    pub stages: Vec<PipelineStageContract>,
}

pub struct PipelineStageContract {
    pub push_const_rngs: Vec<Range<usize>>,
    pub desc_sets: Vec<DescriptorContract>,
    pub stage_appendix: PipelineStageAppendix,
}
pub struct DescriptorContract {
    pub desc_set: u32,
    pub bind_point: u32,
    pub rsc: DescriptorResourceContract,
    pub nbind: Option<usize>,
}
pub enum DescriptorResourceContract {
    Buffer(BufferContract),
    Image(ImageContract),
}
pub enum BufferContract {
    Uniform {
        size: usize,
    },
    Storage {
        size: usize,
        trail_elem_size: Option<usize>,
    },
}
pub enum ImageContract {
    Image1D {
        fmt: Format,
        is_array: bool,
    },
    Image2D {
        fmt: Format,
        is_array: bool,
        is_multisampled: bool,
    },
    Image3D {
        fmt: Format,
    }
}
pub enum PipelineStageAppendix {
    Vertex(VertexAttributeContract),
    Geometry,
    Fragment(AttachmentContract),
    Compute,
}
pub struct VertexAttributeContract {
    pub bind_point: u32,
    pub location: u32,
    pub input_rate: VertexInputRate,
    pub offset: usize,
    pub strides: usize,
    pub format: Format,
}
pub struct AttachmentContract {
    pub location: u32,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub format: Format,
}
