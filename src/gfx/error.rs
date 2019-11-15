use std::fmt;
use std::error;
use ash::{LoadingError, InstanceError};
type VkResultCode = ash::vk::Result;

#[derive(Debug)]
pub enum Error {
    LibraryError(LoadingError),
    RuntimeError(InstanceError),
    VulkanError(VkResultCode),
    NoCapablePhysicalDevice,
    CorruptedSpirv,
    UnsupportedSpirv,
    InflexibleMemory,
    InvalidOperation,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;
        match self {
            LibraryError(err) => write!(f, "{}", err),
            RuntimeError(err) => write!(f, "{}", err),
            VulkanError(errcode) => write!(f, "{}", errcode.as_raw()),
            NoCapablePhysicalDevice => write!(f, "no capable device available"),
            CorruptedSpirv => write!(f, "spirv binary is corrupted"),
            UnsupportedSpirv => write!(f, "spirv binary used unsupported feature"),
            InflexibleMemory => write!(f, "memory cannot be resized"),
            InvalidOperation => write!(f, "an invalid operation was invoked internally"),
        }
    }
}
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        use Error::*;
        match self {
            LibraryError(err) => Some(err),
            RuntimeError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<LoadingError> for Error {
    fn from(x: LoadingError) -> Error { Error::LibraryError(x) }
}
impl From<InstanceError> for Error {
    fn from(x: InstanceError) -> Error { Error::RuntimeError(x) }
}
impl From<VkResultCode> for Error {
    fn from(x: VkResultCode) -> Error { Error::VulkanError(x) }
}
