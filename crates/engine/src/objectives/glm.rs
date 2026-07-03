use alloygbm_core::GradientPair;

use crate::error::{EngineError, EngineResult};
use crate::traits::ObjectiveOps;

/// Clamp `eta` into a safe exponent range before exponentiation so the resulting
/// μ stays in f32-finite range (exp(80) ≈ 5.5e34 fits but exp(89) overflows).
#[inline]
fn glm_clamp_exp(eta: f32) -> f32 {
    eta.clamp(-50.0, 50.0).exp()
}

#[inline]
fn glm_clamp_exp_f64(eta: f32) -> f64 {
    f64::from(eta.clamp(-50.0, 50.0)).exp()
}

const DEFAULT_POISSON_MAX_DELTA_STEP: f32 = 0.7;

/// Weighted-mean-of-targets helper used by every GLM initial prediction.
/// Returns `(sum, total_weight)` or an error on bad weights.
fn glm_weighted_target_sum(
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<(f64, f64)> {
    match sample_weights {
        Some(weights) => {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
            let mut sum = 0.0_f64;
            let mut w_sum = 0.0_f64;
            for (&t, &wi) in targets.iter().zip(weights) {
                if !wi.is_finite() || wi <= 0.0 {
                    return Err(EngineError::ContractViolation(
                        "sample weights must be finite and > 0".to_string(),
                    ));
                }
                sum += f64::from(t) * f64::from(wi);
                w_sum += f64::from(wi);
            }
            Ok((sum, w_sum))
        }
        None => Ok((
            targets.iter().map(|&target| f64::from(target)).sum::<f64>(),
            targets.len() as f64,
        )),
    }
}

/// Poisson regression objective with log-link: `μ = exp(η)`, `y ~ Poisson(μ)`.
/// Targets must be ≥ 0.  Predictions are in log-mean (η) space.
///
/// The Newton hessian is inflated by `exp(max_delta_step)` (LightGBM's
/// `poisson_max_delta_step` stabilizer) to damp updates on sparse or skewed
/// count data.  `max_delta_step` defaults to 0.7.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonObjective {
    max_delta_step: f32,
}

impl Default for PoissonObjective {
    fn default() -> Self {
        Self {
            max_delta_step: DEFAULT_POISSON_MAX_DELTA_STEP,
        }
    }
}

impl PoissonObjective {
    pub fn new(max_delta_step: f32) -> Self {
        Self { max_delta_step }
    }
}

fn poisson_compute_gradients(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    max_delta_step: f32,
) -> EngineResult<Vec<GradientPair>> {
    if predictions.len() != targets.len() {
        return Err(EngineError::ContractViolation(format!(
            "predictions length {} does not match targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    let mut gradients = Vec::with_capacity(predictions.len());
    let hessian_scale = max_delta_step.exp();
    for index in 0..predictions.len() {
        let weight = sample_weights.map_or(1.0, |w| w[index]);
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weights must be finite and > 0".to_string(),
            ));
        }
        let mu = glm_clamp_exp(predictions[index]);
        let grad = (mu - targets[index]) * weight;
        let hess = mu.max(1e-7) * hessian_scale * weight;
        gradients.push(GradientPair::new(grad, hess)?);
    }
    Ok(gradients)
}

impl ObjectiveOps for PoissonObjective {
    fn objective_name(&self) -> &str {
        "poisson"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if targets.is_empty() {
            return Err(EngineError::ContractViolation(
                "targets cannot be empty".to_string(),
            ));
        }
        for &t in targets {
            if !t.is_finite() || t < 0.0 {
                return Err(EngineError::ContractViolation(
                    "Poisson targets must be finite and non-negative".to_string(),
                ));
            }
        }
        let (sum, w_sum) = glm_weighted_target_sum(targets, sample_weights)?;
        if w_sum <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weight sum must be > 0".to_string(),
            ));
        }
        let mean = (sum / w_sum).max(1e-7);
        Ok(mean.ln() as f32)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        poisson_compute_gradients(predictions, targets, sample_weights, self.max_delta_step)
    }

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
        let mut total = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for index in 0..predictions.len() {
            let weight = f64::from(sample_weights.map_or(1.0, |w| w[index]));
            let eta = f64::from(predictions[index].clamp(-50.0, 50.0));
            let mu = eta.exp();
            // Poisson deviance kernel (up to constants): μ − y·η
            total += weight * (mu - f64::from(targets[index]) * eta);
            weight_sum += weight;
        }
        if weight_sum <= 0.0 {
            return Ok(0.0);
        }
        Ok((total / weight_sum) as f32)
    }
}

/// Gamma regression objective with log-link: `μ = exp(η)`, `y ~ Gamma(μ, φ)`.
/// Targets must be strictly positive.  Predictions are in log-mean (η) space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GammaObjective;

impl ObjectiveOps for GammaObjective {
    fn objective_name(&self) -> &str {
        "gamma"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if targets.is_empty() {
            return Err(EngineError::ContractViolation(
                "targets cannot be empty".to_string(),
            ));
        }
        for &t in targets {
            if !t.is_finite() || t <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "Gamma targets must be finite and strictly positive (> 0)".to_string(),
                ));
            }
        }
        let (sum, w_sum) = glm_weighted_target_sum(targets, sample_weights)?;
        if w_sum <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weight sum must be > 0".to_string(),
            ));
        }
        Ok((sum / w_sum).max(1e-7).ln() as f32)
    }

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
        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |w| w[index]);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let mu = glm_clamp_exp(predictions[index]);
            let y_over_mu = targets[index] / mu.max(1e-7);
            let grad = (1.0 - y_over_mu) * weight;
            let hess = y_over_mu.max(1e-7) * weight;
            gradients.push(GradientPair::new(grad, hess)?);
        }
        Ok(gradients)
    }

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
        let mut total = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for index in 0..predictions.len() {
            let weight = f64::from(sample_weights.map_or(1.0, |w| w[index]));
            let mu = glm_clamp_exp_f64(predictions[index]).max(1e-7);
            let r = (f64::from(targets[index]) / mu).max(1e-7);
            total += weight * (r - r.ln() - 1.0);
            weight_sum += weight;
        }
        if weight_sum <= 0.0 {
            return Ok(0.0);
        }
        Ok((total / weight_sum) as f32)
    }
}

/// Tweedie regression objective with log-link for variance power `p ∈ (1, 2)`
/// (compound Poisson-gamma).  Targets must be ≥ 0.  Use [`PoissonObjective`]
/// for `p = 1` and [`GammaObjective`] for `p = 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieObjective {
    pub variance_power: f32,
}

impl TweedieObjective {
    pub fn new(variance_power: f32) -> EngineResult<Self> {
        if !variance_power.is_finite() || variance_power <= 1.0 || variance_power >= 2.0 {
            return Err(EngineError::InvalidConfig(format!(
                "Tweedie variance_power must satisfy 1 < p < 2 (got {variance_power}); \
                 use PoissonObjective for p=1 and GammaObjective for p=2"
            )));
        }
        Ok(Self { variance_power })
    }
}

impl ObjectiveOps for TweedieObjective {
    fn objective_name(&self) -> &str {
        "tweedie"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if targets.is_empty() {
            return Err(EngineError::ContractViolation(
                "targets cannot be empty".to_string(),
            ));
        }
        for &t in targets {
            if !t.is_finite() || t < 0.0 {
                return Err(EngineError::ContractViolation(
                    "Tweedie targets must be finite and non-negative".to_string(),
                ));
            }
        }
        let (sum, w_sum) = glm_weighted_target_sum(targets, sample_weights)?;
        if w_sum <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weight sum must be > 0".to_string(),
            ));
        }
        Ok((sum / w_sum).max(1e-7).ln() as f32)
    }

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
        let p = self.variance_power;
        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |w| w[index]);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let mu = glm_clamp_exp(predictions[index]);
            let mu_2mp = mu.powf(2.0 - p);
            let mu_1mp = mu.powf(1.0 - p);
            let grad = (mu_2mp - targets[index] * mu_1mp) * weight;
            // Simplified Newton hessian (LightGBM/XGBoost convention) — drops the
            // (1-p)·y·μ^(1-p) second-derivative term which would be negative.
            let hess = mu_2mp.max(1e-7) * weight;
            gradients.push(GradientPair::new(grad, hess)?);
        }
        Ok(gradients)
    }

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
        let p = self.variance_power;
        let mut total = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for index in 0..predictions.len() {
            let weight = f64::from(sample_weights.map_or(1.0, |w| w[index]));
            let mu = glm_clamp_exp_f64(predictions[index]);
            let y = f64::from(targets[index]);
            let p = f64::from(p);
            let term1 = if y > 0.0 {
                y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
            } else {
                0.0
            };
            let term2 = y * mu.powf(1.0 - p) / (1.0 - p);
            let term3 = mu.powf(2.0 - p) / (2.0 - p);
            total += 2.0 * weight * (term1 - term2 + term3);
            weight_sum += weight;
        }
        if weight_sum <= 0.0 {
            return Ok(0.0);
        }
        Ok((total / weight_sum) as f32)
    }
}
