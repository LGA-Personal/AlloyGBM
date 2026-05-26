use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidConfig(String),
    Validation(String),
    Io(String),
    Serialization(String),
    NotImplemented(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for CoreError {}

pub type CoreResult<T> = Result<T, CoreError>;
