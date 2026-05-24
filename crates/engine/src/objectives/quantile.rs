use std::cmp::Ordering;

use alloygbm_core::GradientPair;

use crate::error::{EngineError, EngineResult};
use crate::traits::ObjectiveOps;

// ── Quantile Regression Objective ────────────────────────────────────────

/// Quantile Regression (pinball loss) objective.
///
/// Since the second derivative (Hessian) of the pinball loss is zero everywhere
/// except at 0 (where it is undefined), standard Newton-Raphson tree boosting
/// cannot compute meaningful split gains directly.
/// To circumvent this, we use a proxy Hessian `h_i = w_i` (sample weight,
/// defaulting to 1.0) during split finding. This acts as a proxy for the sample
/// count in each leaf.
/// Because using a proxy Hessian results in incorrect Newton-Raphson leaf values,
/// a post-growth leaf refinement step (`refine_quantile_leaf_values`) is executed
/// at the end of each round to replace leaf predictions with the actual empirical
/// quantiles of the residuals of rows routed to each leaf.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantileObjective {
    pub alpha: f32,
}

impl ObjectiveOps for QuantileObjective {
    fn objective_name(&self) -> &str {
        "quantile"
    }

    fn quantile_alpha(&self) -> Option<f32> {
        Some(self.alpha)
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        weighted_quantile(targets, sample_weights, self.alpha)
    }

    #[allow(clippy::collapsible_if)]
    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        if predictions.len() != targets.len() {
            return Err(EngineError::ContractViolation(format!(
                "predictions length {} does not match targets length {}",
                predictions.len(),
                targets.len()
            )));
        }
        if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
        }

        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let target = targets[index];
            let prediction = predictions[index];
            let grad = if target > prediction {
                -self.alpha * weight
            } else {
                (1.0 - self.alpha) * weight
            };
            gradients.push(GradientPair::new(grad, weight)?);
        }
        Ok(gradients)
    }

    #[allow(clippy::collapsible_if)]
    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        if predictions.len() != targets.len() {
            return Err(EngineError::ContractViolation(format!(
                "predictions length {} does not match targets length {}",
                predictions.len(),
                targets.len()
            )));
        }
        if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
        }

        buffer.clear();
        if buffer.capacity() < predictions.len() {
            buffer.reserve(predictions.len() - buffer.capacity());
        }
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let target = targets[index];
            let prediction = predictions[index];
            let grad = if target > prediction {
                -self.alpha * weight
            } else {
                (1.0 - self.alpha) * weight
            };
            buffer.push(GradientPair { grad, hess: weight });
        }
        Ok(())
    }

    #[allow(clippy::collapsible_if)]
    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if predictions.len() != targets.len() {
            return Err(EngineError::ContractViolation(format!(
                "predictions length {} does not match targets length {}",
                predictions.len(),
                targets.len()
            )));
        }
        if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
        }

        let n = predictions.len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut total = 0.0_f64;
        for index in 0..n {
            let pred = predictions[index];
            let target = targets[index];
            let weight = sample_weights.map_or(1.0, |w| w[index]);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let diff = target - pred;
            let loss_i = if diff > 0.0 {
                self.alpha * diff
            } else {
                (self.alpha - 1.0) * diff
            };
            total += loss_i as f64 * weight as f64;
        }

        Ok((total / n as f64) as f32)
    }
}

#[allow(clippy::collapsible_if)]
pub(crate) fn weighted_quantile(
    values: &[f32],
    weights: Option<&[f32]>,
    alpha: f32,
) -> EngineResult<f32> {
    if values.is_empty() {
        return Err(EngineError::ContractViolation(
            "values cannot be empty".to_string(),
        ));
    }
    if let Some(w) = weights {
        if w.len() != values.len() {
            return Err(EngineError::ContractViolation(format!(
                "weights length {} does not match values length {}",
                w.len(),
                values.len()
            )));
        }
    }

    if weights.is_none() {
        for &val in values {
            if !val.is_finite() {
                return Err(EngineError::ContractViolation(
                    "values must be finite".to_string(),
                ));
            }
        }
        let mut vals = values.to_vec();
        let k = ((alpha as f64 * vals.len() as f64) - 1.0).max(0.0).ceil() as usize;
        let k = k.min(vals.len() - 1);
        vals.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        return Ok(vals[k]);
    }

    let mut pairs = Vec::with_capacity(values.len());
    let mut total_weight = 0.0_f64;
    for i in 0..values.len() {
        let w = weights.map_or(1.0, |ws| ws[i]);
        if !w.is_finite() || w <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weights must be finite and > 0".to_string(),
            ));
        }
        if !values[i].is_finite() {
            return Err(EngineError::ContractViolation(
                "values must be finite".to_string(),
            ));
        }
        pairs.push((values[i], w));
        total_weight += w as f64;
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let threshold = alpha as f64 * total_weight;
    let mut cum_weight = 0.0_f64;
    for &(val, weight) in &pairs {
        cum_weight += weight as f64;
        if cum_weight >= threshold {
            return Ok(val);
        }
    }

    Ok(pairs.last().unwrap().0)
}

/// Resolves group boundaries for a given data length.
///
/// If `data_len` matches the training boundaries' total, return training
/// boundaries.  If a validation set is present and its boundaries match,
/// return those.  Otherwise fall back to a single-group interpretation.
pub(crate) fn resolve_boundaries_for_len(
    train_boundaries: &[usize],
    validation_boundaries: &Option<Vec<usize>>,
    data_len: usize,
) -> Vec<usize> {
    if let Some(last) = train_boundaries.last()
        && *last == data_len
    {
        return train_boundaries.to_vec();
    }
    if let Some(val_b) = validation_boundaries
        && let Some(last) = val_b.last()
        && *last == data_len
    {
        return val_b.clone();
    }
    // Fallback: treat entire slice as a single group.
    vec![0, data_len]
}
