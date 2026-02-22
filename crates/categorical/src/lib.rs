use alloygbm_core::CoreError;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct TargetEncoderConfig {
    pub smoothing: f64,
    pub min_samples_leaf: u32,
    pub time_aware: bool,
}

impl Default for TargetEncoderConfig {
    fn default() -> Self {
        Self {
            smoothing: 20.0,
            min_samples_leaf: 1,
            time_aware: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoricalError {
    InvalidInput(String),
    NotImplemented(String),
    Core(CoreError),
}

impl Display for CategoricalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for CategoricalError {}

impl From<CoreError> for CategoricalError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

pub type CategoricalResult<T> = Result<T, CategoricalError>;

pub fn fit_transform_stub(
    _config: &TargetEncoderConfig,
    values: &[String],
    targets: &[f32],
) -> CategoricalResult<Vec<f32>> {
    if values.is_empty() {
        return Err(CategoricalError::InvalidInput(
            "values cannot be empty".to_string(),
        ));
    }

    if values.len() != targets.len() {
        return Err(CategoricalError::InvalidInput(
            "values and targets must have matching lengths".to_string(),
        ));
    }

    Err(CategoricalError::NotImplemented(
        "fit_transform_stub is a placeholder in v0.0.1".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_transform_rejects_mismatched_lengths() {
        let cfg = TargetEncoderConfig::default();
        let values = vec!["A".to_string(), "B".to_string()];
        let targets = vec![1.0];
        let result = fit_transform_stub(&cfg, &values, &targets);
        assert!(matches!(result, Err(CategoricalError::InvalidInput(_))));
    }

    #[test]
    fn fit_transform_returns_not_implemented_for_valid_input() {
        let cfg = TargetEncoderConfig::default();
        let values = vec!["A".to_string()];
        let targets = vec![1.0];
        let result = fit_transform_stub(&cfg, &values, &targets);
        assert!(matches!(result, Err(CategoricalError::NotImplemented(_))));
    }
}
