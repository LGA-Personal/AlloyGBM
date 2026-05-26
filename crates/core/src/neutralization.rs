use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralizationKind {
    None,
    PreTarget,
    PerRoundGradient,
    SplitPenalty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorNeutralizationConfig {
    pub kind: NeutralizationKind,
    pub ridge_lambda: f32,
    pub split_penalty: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorExposureMatrix {
    pub row_count: usize,
    pub factor_count: usize,
    pub values: Vec<f32>,
}

impl FactorExposureMatrix {
    pub fn new(row_count: usize, factor_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        if row_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures row_count must be greater than 0".to_string(),
            ));
        }
        if factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = row_count.checked_mul(factor_count).ok_or_else(|| {
            CoreError::Validation("factor_exposures row_count * factor_count overflow".to_string())
        })?;
        if values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                values.len(),
                expected_len
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::Validation(
                "factor_exposures must contain only finite values".to_string(),
            ));
        }
        Ok(Self {
            row_count,
            factor_count,
            values,
        })
    }

    pub fn row(&self, row_index: usize) -> CoreResult<&[f32]> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "factor_exposures row_index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        if self.factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = self
            .row_count
            .checked_mul(self.factor_count)
            .ok_or_else(|| {
                CoreError::Validation(
                    "factor_exposures row_count * factor_count overflow".to_string(),
                )
            })?;
        if self.values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                self.values.len(),
                expected_len
            )));
        }
        let start = row_index * self.factor_count;
        Ok(&self.values[start..start + self.factor_count])
    }
}
