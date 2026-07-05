//! Factor neutralization helpers — projection of gradients (or target values)
//! onto a factor-exposure matrix.
//!
//! Used by the single-output trainer in `lib.rs` and the joint multi-output
//! trainer in `joint.rs` (see CLAUDE.md v0.10.6 entry).

use crate::error::{EngineError, EngineResult};
use alloygbm_core::{FactorExposureMatrix, GradientPair, TrainingDataset};
use rayon::prelude::*;

#[allow(dead_code)]
pub(crate) struct FactorProjector<'a> {
    exposures: &'a FactorExposureMatrix,
    weights: Option<&'a [f32]>,
    cholesky_lower: Vec<f64>,
}

#[allow(dead_code)]
impl<'a> FactorProjector<'a> {
    pub(crate) fn new(
        exposures: &'a FactorExposureMatrix,
        weights: Option<&'a [f32]>,
        ridge_lambda: f32,
    ) -> EngineResult<Self> {
        if let Some(w) = weights
            && w.len() != exposures.row_count
        {
            return Err(EngineError::ContractViolation(
                "sample_weight length must match factor_exposures row_count".to_string(),
            ));
        }
        validate_exposure_shape(exposures)?;
        let k = exposures.factor_count;
        let mut gram = exposures
            .values
            .par_chunks_exact(k)
            .enumerate()
            .fold(
                || vec![0.0_f64; k * k],
                |mut local, (row, factors)| {
                    let weight = weights.map_or(1.0_f64, |w| f64::from(w[row]));
                    for a in 0..k {
                        for b in 0..=a {
                            local[a * k + b] +=
                                weight * f64::from(factors[a]) * f64::from(factors[b]);
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0_f64; k * k],
                |mut left, right| {
                    for (left_value, right_value) in left.iter_mut().zip(right) {
                        *left_value += right_value;
                    }
                    left
                },
            );
        for i in 0..k {
            gram[i * k + i] += f64::from(ridge_lambda);
        }
        let cholesky_lower = cholesky_lower(gram, k)?;
        Ok(Self {
            exposures,
            weights,
            cholesky_lower,
        })
    }

    pub(crate) fn project_gradient_pairs_in_place(
        &self,
        gradients: &mut [GradientPair],
    ) -> EngineResult<()> {
        let mut residualized = Vec::new();
        self.project_gradient_pairs_in_place_with_scratch(gradients, &mut residualized)
    }

    pub(crate) fn project_gradient_pairs_in_place_with_scratch(
        &self,
        gradients: &mut [GradientPair],
        residualized: &mut Vec<f32>,
    ) -> EngineResult<()> {
        if gradients.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "gradient length must match factor_exposures row_count".to_string(),
            ));
        }
        let coefficients = self.projection_coefficients_for_gradients(gradients)?;
        residualized.clear();
        residualized.resize(gradients.len(), 0.0);
        let k = self.exposures.factor_count;
        residualized
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, residual)| {
                let factors = &self.exposures.values[row * k..row * k + k];
                *residual = (f64::from(gradients[row].grad)
                    - projected_row_value(factors, &coefficients))
                    as f32;
            });
        if residualized.iter().any(|residual| !residual.is_finite()) {
            return Err(EngineError::ContractViolation(
                "projected gradient must be finite".to_string(),
            ));
        }
        gradients
            .par_iter_mut()
            .zip(residualized.par_iter())
            .for_each(|(gradient, residual)| {
                gradient.grad = *residual;
            });
        Ok(())
    }

    pub(crate) fn residualize_values_in_place(&self, values: &mut [f32]) -> EngineResult<()> {
        let mut residualized = Vec::new();
        self.residualize_values_in_place_with_scratch(values, &mut residualized)
    }

    pub(crate) fn residualize_values_in_place_with_scratch(
        &self,
        values: &mut [f32],
        residualized: &mut Vec<f32>,
    ) -> EngineResult<()> {
        if values.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "value length must match factor_exposures row_count".to_string(),
            ));
        }
        let coefficients = self.projection_coefficients_for_values(values)?;
        residualized.clear();
        residualized.resize(values.len(), 0.0);
        let k = self.exposures.factor_count;
        residualized
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, residual)| {
                let factors = &self.exposures.values[row * k..row * k + k];
                *residual =
                    (f64::from(values[row]) - projected_row_value(factors, &coefficients)) as f32;
            });
        if residualized.iter().any(|residual| !residual.is_finite()) {
            return Err(EngineError::ContractViolation(
                "residualized value must be finite".to_string(),
            ));
        }
        values
            .par_iter_mut()
            .zip(residualized.par_iter())
            .for_each(|(value, residual)| {
                *value = *residual;
            });
        Ok(())
    }

    pub(crate) fn projection_coefficients(
        &self,
        values: impl IntoIterator<Item = f32>,
    ) -> EngineResult<Vec<f64>> {
        let values = values.into_iter().collect::<Vec<_>>();
        self.projection_coefficients_for_values(&values)
    }

    fn projection_coefficients_for_gradients(
        &self,
        gradients: &[GradientPair],
    ) -> EngineResult<Vec<f64>> {
        if gradients.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "value length must match factor_exposures row_count".to_string(),
            ));
        }
        self.projection_coefficients_from_rows(|row| gradients[row].grad)
    }

    fn projection_coefficients_for_values(&self, values: &[f32]) -> EngineResult<Vec<f64>> {
        if values.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "value length must match factor_exposures row_count".to_string(),
            ));
        }
        self.projection_coefficients_from_rows(|row| values[row])
    }

    fn projection_coefficients_from_rows(
        &self,
        value_at: impl Fn(usize) -> f32 + Sync,
    ) -> EngineResult<Vec<f64>> {
        let k = self.exposures.factor_count;
        let rhs = self
            .exposures
            .values
            .par_chunks_exact(k)
            .enumerate()
            .fold(
                || vec![0.0_f64; k],
                |mut local, (row, factors)| {
                    let weight = self.weights.map_or(1.0_f64, |w| f64::from(w[row]));
                    let value = f64::from(value_at(row));
                    for a in 0..k {
                        local[a] += weight * f64::from(factors[a]) * value;
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0_f64; k],
                |mut left, right| {
                    for (left_value, right_value) in left.iter_mut().zip(right) {
                        *left_value += right_value;
                    }
                    left
                },
            );
        self.solve_cholesky(&rhs)
    }

    fn solve_cholesky(&self, rhs: &[f64]) -> EngineResult<Vec<f64>> {
        let k = self.exposures.factor_count;
        if rhs.len() != k {
            return Err(EngineError::ContractViolation(
                "factor projection rhs length must match factor count".to_string(),
            ));
        }

        let mut y = vec![0.0_f64; k];
        for i in 0..k {
            let mut sum = rhs[i];
            for (j, y_j) in y.iter().enumerate().take(i) {
                sum -= self.cholesky_lower[i * k + j] * *y_j;
            }
            y[i] = sum / self.cholesky_lower[i * k + i];
        }

        let mut x = vec![0.0_f64; k];
        for i in (0..k).rev() {
            let mut sum = y[i];
            for (j, x_j) in x.iter().enumerate().take(k).skip(i + 1) {
                sum -= self.cholesky_lower[j * k + i] * *x_j;
            }
            x[i] = sum / self.cholesky_lower[i * k + i];
        }
        Ok(x)
    }
}

#[allow(dead_code)]
fn cholesky_lower(mut matrix: Vec<f64>, k: usize) -> EngineResult<Vec<f64>> {
    for i in 0..k {
        for j in 0..=i {
            let mut sum = matrix[i * k + j];
            for p in 0..j {
                sum -= matrix[i * k + p] * matrix[j * k + p];
            }
            if i == j {
                if sum <= 1e-12 {
                    return Err(EngineError::ContractViolation(
                        "factor exposure Gram matrix is singular; increase factor_neutralization_lambda"
                            .to_string(),
                    ));
                }
                matrix[i * k + j] = sum.sqrt();
            } else {
                matrix[i * k + j] = sum / matrix[j * k + j];
            }
        }
        for j in i + 1..k {
            matrix[i * k + j] = 0.0;
        }
    }
    Ok(matrix)
}

fn validate_exposure_shape(exposures: &FactorExposureMatrix) -> EngineResult<()> {
    if exposures.row_count == 0 {
        return Err(EngineError::ContractViolation(
            "factor_exposures row_count must be greater than 0".to_string(),
        ));
    }
    if exposures.factor_count == 0 {
        return Err(EngineError::ContractViolation(
            "factor_exposures factor_count must be greater than 0".to_string(),
        ));
    }
    let expected_len = exposures
        .row_count
        .checked_mul(exposures.factor_count)
        .ok_or_else(|| {
            EngineError::ContractViolation(
                "factor_exposures row_count * factor_count overflow".to_string(),
            )
        })?;
    if exposures.values.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures values length {} does not match row_count * factor_count {}",
            exposures.values.len(),
            expected_len
        )));
    }
    Ok(())
}

#[allow(dead_code)]
fn projected_row_value(exposures: &[f32], coefficients: &[f64]) -> f64 {
    exposures
        .iter()
        .zip(coefficients.iter())
        .map(|(factor, coefficient)| f64::from(*factor) * coefficient)
        .sum::<f64>()
}

pub(crate) fn apply_pre_target_neutralization(
    dataset: &mut TrainingDataset,
    ridge_lambda: f32,
) -> EngineResult<()> {
    let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
        EngineError::ContractViolation(
            "factor_exposures are required when neutralization is active".to_string(),
        )
    })?;
    FactorProjector::new(exposures, dataset.sample_weights.as_deref(), ridge_lambda)?
        .residualize_values_in_place(&mut dataset.targets)
}
