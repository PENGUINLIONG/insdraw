use std::ops::Range;
use ash::vk::{ImageViewType, Format, VertexInputRate, AttachmentLoadOp, AttachmentStoreOp, ImageLayout, DescriptorType};


/// Pipeline resource memory allocation and usage contracts.
pub mod mem {
    use super::*;
    /// Memory sizing and arrangement contract checked against to ensure a basic
    /// level of memory usage conformance.
    pub enum PipelineResourceContract {
        Buffer(BufferContract),
        Image(ImageContract),
    }
    pub enum BufferType {
        Uniform,
        Storage,
    }
    pub struct BufferContract {
        /// Buffer type.
        pub ty: BufferType,
        /// Minimum size of the buffer.
        pub min_size: usize,
        /// Maximum size of the buffer. For UBO equivalents, this will always be the
        /// same as `min_size`. For SSBO equivalents, the value can be `None` when
        /// the buffer ends with a variable-sized array.
        pub max_size: Option<usize>,
    }
    pub struct ImageContract {
        /// Image dimensionality and arrangement type.
        pub ty: ImageViewType,
        /// Pixel data format.
        pub fmt: Format,
    }
}


/// Pipeline creation and usage contracts.
pub mod pipe {
    use super::*;
    pub enum PipelineStage {
        Vertex,
        Geometry,
        Fragment,
    }

    pub struct RenderPassContract {
        pub subpasses: Vec<SubpassContract>,
    }
    pub struct SubpassContract {
        pub attrs: Vec<VertexAttributeContract>,
        pub attms: Vec<AttachmentContract>,
        pub pipe_stages: Vec<PipelineStageContract>,
    }
    pub struct VertexAttributeContract {
        pub binding_point: u32,
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
        pub layout: ImageLayout,
    }
    pub struct PipelineStageContract {
        pub stage: PipelineStage,
        pub push_const_rngs: Vec<Range<usize>>,
        pub desc_sets: Vec<DescriptorContract>,
    }
    pub struct DescriptorContract {
        pub set: u32,
        pub bind_point: u32,
        pub desc_ty: DescriptorType,
        pub nbind: Option<usize>,
    }
}
