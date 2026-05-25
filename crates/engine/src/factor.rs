//! Factor neutralization helpers — projection of gradients (or target values)
//! onto a factor-exposure matrix.
//!
//! Used by the single-output trainer in `lib.rs` and the joint multi-output
//! trainer in `joint.rs` (see CLAUDE.md v0.10.6 entry).

use crate::error::{EngineError, EngineResult};
use alloygbm_core::{FactorExposureMatrix, GradientPair, TrainingDataset};

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
        let k = exposures.factor_count;
        let mut gram = vec![0.0_f64; k * k];
        for row in 0..exposures.row_count {
            let weight = weights.map_or(1.0_f64, |w| f64::from(w[row]));
            let f = exposures.row(row)?;
            for a in 0..k {
                for b in 0..=a {
                    gram[a * k + b] += weight * f64::from(f[a]) * f64::from(f[b]);
                }
            }
        }
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
        if gradients.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "gradient length must match factor_exposures row_count".to_string(),
            ));
        }
        let coefficients = self.projection_coefficients(gradients.iter().map(|g| g.grad))?;
        let mut residualized = Vec::with_capacity(gradients.len());
        for (row, gradient) in gradients.iter().enumerate() {
            let residual = f64::from(gradient.grad)
                - projected_row_value(self.exposures.row(row)?, &coefficients);
            let residual = residual as f32;
            if !residual.is_finite() {
                return Err(EngineError::ContractViolation(
                    "projected gradient must be finite".to_string(),
                ));
            }
            residualized.push(residual);
        }
        for (gradient, residual) in gradients.iter_mut().zip(residualized) {
            gradient.grad = residual;
        }
        Ok(())
    }

    pub(crate) fn residualize_values_in_place(&self, values: &mut [f32]) -> EngineResult<()> {
        if values.len() != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "value length must match factor_exposures row_count".to_string(),
            ));
        }
        let coefficients = self.projection_coefficients(values.iter().copied())?;
        let mut residualized = Vec::with_capacity(values.len());
        for (row, value) in values.iter().enumerate() {
            let residual =
                f64::from(*value) - projected_row_value(self.exposures.row(row)?, &coefficients);
            let residual = residual as f32;
            if !residual.is_finite() {
                return Err(EngineError::ContractViolation(
                    "residualized value must be finite".to_string(),
                ));
            }
            residualized.push(residual);
        }
        for (value, residual) in values.iter_mut().zip(residualized) {
            *value = residual;
        }
        Ok(())
    }

    pub(crate) fn projection_coefficients(
        &self,
        values: impl IntoIterator<Item = f32>,
    ) -> EngineResult<Vec<f64>> {
        let k = self.exposures.factor_count;
        let mut rhs = vec![0.0_f64; k];
        let mut value_count = 0;
        for (row, value) in values.into_iter().enumerate() {
            if row >= self.exposures.row_count {
                return Err(EngineError::ContractViolation(
                    "value length must match factor_exposures row_count".to_string(),
                ));
            }
            value_count += 1;
            let weight = self.weights.map_or(1.0_f64, |w| f64::from(w[row]));
            let f = self.exposures.row(row)?;
            for a in 0..k {
                rhs[a] += weight * f64::from(f[a]) * f64::from(value);
            }
        }
        if value_count != self.exposures.row_count {
            return Err(EngineError::ContractViolation(
                "value length must match factor_exposures row_count".to_string(),
            ));
        }
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
