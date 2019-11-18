use std::fmt;
use std::error;

#[derive(Debug)]
pub enum Error {
    CorruptedSpirv,
    UnsupportedSpirv,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Error::*;
        match self {
            CorruptedSpirv => write!(f, "spirv binary is corrupted"),
            UnsupportedSpirv => write!(f, "spirv binary used unsupported feature"),
        }
    }
}
impl error::Error for Error { }
