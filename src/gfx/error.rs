use std::fmt;
use std::error;
use ash::{LoadingError, InstanceError};
use spirq::error::Error as SpirqError;
type VkResultCode = ash::vk::Result;

#[derive(Debug)]
pub enum Error {
    LibraryError(LoadingError),
    RuntimeError(InstanceError),
    VulkanError(VkResultCode),
    MissingExtension(&'static std::ffi::CStr),
    UnsupportedPlatform,
    InflexibleMemory,
    InvalidOperation,
    InvalidName(bool), // Whether the name is expected to exist before the op.
    SpirqError(SpirqError),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;
        match self {
            LibraryError(err) => write!(f, "{}", err),
            RuntimeError(err) => write!(f, "{}", err),
            VulkanError(errcode) => write!(f, "{}", errcode.as_raw()),
            MissingExtension(ext_name) => write!(f, "{:?} is required but unsupported", ext_name),
            UnsupportedPlatform => write!(f, "unsupported platform"),
            InflexibleMemory => write!(f, "memory cannot be resized"),
            InvalidOperation => write!(f, "an invalid operation was invoked internally"),
            InvalidName(true) => write!(f, "expect an existing name"),
            InvalidName(false) => write!(f, "expect an absent name"),
            SpirqError(err) => write!(f, "reflection error: {}", err),
        }
    }
}
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        use Error::*;
        match self {
            LibraryError(err) => Some(err),
            RuntimeError(err) => Some(err),
            SpirqError(err) => Some(err),
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
impl From<SpirqError> for Error {
    fn from(x: SpirqError) -> Error { Error::SpirqError(x) }
}

pub type Result<T> = std::result::Result<T, Error>;
