use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapError {
    InvalidInput(String),
    ContractViolation(String),
    NotSupported(String),
}

impl Display for ShapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::ContractViolation(message) => write!(f, "contract violation: {message}"),
            Self::NotSupported(message) => write!(f, "not supported: {message}"),
        }
    }
}

impl Error for ShapError {}

pub type ShapResult<T> = Result<T, ShapError>;
