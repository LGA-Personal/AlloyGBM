use alloygbm_core::CoreError;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidConfig(String),
    ContractViolation(String),
    BackendUnavailable(String),
    NotImplemented(String),
    Core(CoreError),
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ContractViolation(msg) => write!(f, "contract violation: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for EngineError {}

impl From<CoreError> for EngineError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}
