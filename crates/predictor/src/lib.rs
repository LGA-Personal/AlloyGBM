use alloygbm_core::ModelMetadata;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictorError {
    InvalidInput(String),
    NotImplemented(String),
}

impl Display for PredictorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for PredictorError {}

pub type PredictorResult<T> = Result<T, PredictorError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Predictor {
    pub metadata: ModelMetadata,
}

impl Predictor {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self { metadata }
    }

    pub fn predict_row_stub(&self, features: &[f32]) -> PredictorResult<f32> {
        if features.is_empty() {
            return Err(PredictorError::InvalidInput(
                "features cannot be empty".to_string(),
            ));
        }

        Err(PredictorError::NotImplemented(
            "predict_row_stub is a placeholder in v0.0.1".to_string(),
        ))
    }

    pub fn predict_batch_stub(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }

        Err(PredictorError::NotImplemented(
            "predict_batch_stub is a placeholder in v0.0.1".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{Device, ModelMetadata};

    fn predictor() -> Predictor {
        let metadata = ModelMetadata {
            format_version: 1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
        };
        Predictor::new(metadata)
    }

    #[test]
    fn row_stub_returns_not_implemented() {
        let pred = predictor();
        let result = pred.predict_row_stub(&[1.0]);
        assert!(matches!(result, Err(PredictorError::NotImplemented(_))));
    }

    #[test]
    fn batch_stub_rejects_empty_rows() {
        let pred = predictor();
        let result = pred.predict_batch_stub(&[]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }
}
