//! # Material Information
//!
//! Material allocation and evaluation facilities.
use bytes::{Buf, BufMut};
use std::ffi::c_void;
use crate::gfx::{Error, Result, BufferType, ImageType, ColorFormat};

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
        Result::Err(Error::InflexibleMemory)
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
    pub ty: BufferType,
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
