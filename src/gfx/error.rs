use std::fmt;
use std::error::Error;
use ash::{LoadingError, InstanceError};
type VkResultCode = ash::vk::Result;

#[derive(Debug)]
pub enum GraphicsError {
    LibraryError(LoadingError),
    RuntimeError(InstanceError),
    VulkanError(VkResultCode),
    NoCapablePhysicalDevice,
}
impl fmt::Display for GraphicsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use GraphicsError::*;
        match self {
            LibraryError(err) => write!(f, "{}", err),
            RuntimeError(err) => write!(f, "{}", err),
            VulkanError(errcode) => write!(f, "{}", errcode.as_raw()),
            NoCapablePhysicalDevice => write!(f, "no capable device available"),
        }
    }
}
impl Error for GraphicsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use GraphicsError::*;
        match self {
            LibraryError(err) => Some(err),
            RuntimeError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<LoadingError> for GraphicsError {
    fn from(x: LoadingError) -> GraphicsError { GraphicsError::LibraryError(x) }
}
impl From<InstanceError> for GraphicsError {
    fn from(x: InstanceError) -> GraphicsError { GraphicsError::RuntimeError(x) }
}
impl From<VkResultCode> for GraphicsError {
    fn from(x: VkResultCode) -> GraphicsError { GraphicsError::VulkanError(x) }
}

