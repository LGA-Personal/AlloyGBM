use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    pub seed: u64,
    pub deterministic: bool,
    pub learning_rate: f32,
    pub max_depth: u16,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            deterministic: true,
            learning_rate: 0.1,
            max_depth: 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSchema {
    pub feature_count: usize,
    pub has_time_index: bool,
    pub has_group_id: bool,
    pub categorical_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub trained_device: Device,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidConfig(String),
    Validation(String),
    Io(String),
    NotImplemented(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for CoreError {}

pub type CoreResult<T> = Result<T, CoreError>;

pub fn validate_train_params(params: &TrainParams) -> CoreResult<()> {
    if !(0.0..=1.0).contains(&params.learning_rate) || params.learning_rate == 0.0 {
        return Err(CoreError::InvalidConfig(
            "learning_rate must be in (0.0, 1.0]".to_string(),
        ));
    }

    if params.max_depth == 0 {
        return Err(CoreError::InvalidConfig(
            "max_depth must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_default_train_params() {
        let params = TrainParams::default();
        assert!(validate_train_params(&params).is_ok());
    }

    #[test]
    fn rejects_invalid_learning_rate() {
        let params = TrainParams {
            learning_rate: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }
}
