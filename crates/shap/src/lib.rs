use alloygbm_core::ModelMetadata;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapError {
    InvalidInput(String),
    NotImplemented(String),
}

impl Display for ShapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for ShapError {}

pub type ShapResult<T> = Result<T, ShapError>;

pub fn shap_values_stub(_metadata: &ModelMetadata, rows: &[Vec<f32>]) -> ShapResult<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Err(ShapError::InvalidInput("rows cannot be empty".to_string()));
    }

    Err(ShapError::NotImplemented(
        "shap_values_stub is a placeholder in v0.0.1".to_string(),
    ))
}

pub fn global_importance_stub(
    _metadata: &ModelMetadata,
    feature_names: &[String],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }

    Err(ShapError::NotImplemented(
        "global_importance_stub is a placeholder in v0.0.1".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{Device, ModelMetadata};

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            format_version: 1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
        }
    }

    #[test]
    fn shap_stub_returns_not_implemented() {
        let result = shap_values_stub(&metadata(), &[vec![1.0]]);
        assert!(matches!(result, Err(ShapError::NotImplemented(_))));
    }

    #[test]
    fn global_importance_rejects_empty_features() {
        let result = global_importance_stub(&metadata(), &[]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }
}
