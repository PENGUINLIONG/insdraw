//! # Material Information
//!
//! Material allocation and evaluation facilities.
use bytes::{Buf, BufMut};
use std::ffi::c_void;
use super::{Error, Result, BufferType, ImageType, ImageFormat};

/// Memory sizing and arrangement contract checked against to ensure a basic
/// level of memory usage conformance.
pub enum PipelineResourceContract {
    Buffer(BufferManifest),
    Image(ImageManifest),
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
    pub ty: ImageType,
    /// Pixel data format.
    pub fmt: ImageFormat,
}

/// Allocated device memory which can be accessed and modified by host
/// procedures. Different backend might define different behavior to transfer
/// data.
pub trait DeviceMemory {
    /// Get the context this device memory bounds to.
    fn ctxt(&self) -> &Context;
    /// Get the size in bytes of the memory.
    fn size(&self) -> usize;
    fn capacity(&self) -> usize { self.size() }
    fn resize(&mut self, new_size: usize) -> Result<()> {
        Result::Err(Error::OutOfMemory)
    }
    /// Map the device memory to host.
    unsafe fn map(&mut self) -> *mut c_void;
    unsafe fn unmap(&mut self);
}
pub trait BufferMemoryInterface<Mem: DeviceMemory> : AsMut<DeviceMemory> {
}
pub trait ImageMemoryInterface<Mem: DeviceMemory> : AsMut<DeviceMemory> {
}

/// Requirement used to allocate memory on a device.
pub enum PipelineResourceRequirement {
    Buffer(BufferRequirement),
    Image(ImageRequirement),
}
pub struct BufferRequirement {
    pub ty: BufferType;
}
pub struct ImageRequirement {
    pub ty: ImageType,
    pub fmt: ColorFormat,
    pub extent: (usize, usize, usize),
    pub nlayer: Option<usize>,
    pub mip_range: (usize, usize),
}
pub enum PipelineResource<Mem: DeviceMemory> {
    Buffer(Buffer<Mem>),
    Image(Image<Mem>),
}
pub struct Buffer<Mem: DeviceMemory> {
    pub i: Box<dyn BufferMemoryInterface<Mem>>,
    pub req: BufferRequirement,
}
pub struct Image<Mem: DeviceMemory> {
    pub i: Box<dyn ImageMemoryInterface<Mem>>,
    pub req: ImageRequirement,
}

pub trait MemoryAllocator<Mem: DeviceMemory> {
    fn alloc_buf(req: BufferRequirement) -> Buffer<Mem>;
    fn alloc_img(req: ImageRequirement) -> Image<Mem>;
}
