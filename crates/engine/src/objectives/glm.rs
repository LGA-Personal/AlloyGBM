use alloygbm_core::{GlmHessian, GradientPair};

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
///
/// With [`GlmHessian::Bounded`] the hessian is `w·(2μ + y)/3` instead and
/// `max_delta_step` is ignored: an intercept-style leaf step becomes
/// `3(Y − M)/(2M + Y)` (`Y = Σy`, `M = Σμ`), which lies in `(−3/2, 3)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonObjective {
    max_delta_step: f32,
    hessian: GlmHessian,
}

impl Default for PoissonObjective {
    fn default() -> Self {
        Self::new(DEFAULT_POISSON_MAX_DELTA_STEP)
    }
}

impl PoissonObjective {
    pub fn new(max_delta_step: f32) -> Self {
        Self {
            max_delta_step,
            hessian: GlmHessian::Exact,
        }
    }

    pub fn with_glm_hessian(mut self, hessian: GlmHessian) -> Self {
        self.hessian = hessian;
        self
    }
}

/// Bounded-step pseudo-Hessian shared by the three log-link objectives
/// (dmlc/xgboost#12538): `(ρ·a + (3 − ρ)·b)/3` with `a = y·μ^(1−ρ)` and
/// `b = μ^(2−ρ)`, for variance power `ρ ∈ [1, 2]`.  `ρ = 1` gives Poisson's
/// `(y + 2μ)/3`; `ρ = 2` gives Gamma's `(2y/μ + 1)/3`.  Unweighted.
#[inline]
fn bounded_log_link_hessian(rho: f32, y: f32, mu_1mp: f32, mu_2mp: f32) -> f32 {
    (rho * y * mu_1mp + (3.0 - rho) * mu_2mp) / 3.0
}

fn poisson_compute_gradients(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    max_delta_step: f32,
    hessian: GlmHessian,
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
        let hess = match hessian {
            GlmHessian::Exact => mu.max(1e-7) * hessian_scale,
            GlmHessian::Bounded => bounded_log_link_hessian(1.0, targets[index], 1.0, mu).max(1e-7),
        } * weight;
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
        poisson_compute_gradients(
            predictions,
            targets,
            sample_weights,
            self.max_delta_step,
            self.hessian,
        )
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
///
/// The exact hessian is `w·y/μ`; with [`GlmHessian::Bounded`] it is
/// `w·(2y/μ + 1)/3`, which keeps an intercept-style leaf step in `(−3, 3/2)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GammaObjective {
    hessian: GlmHessian,
}

impl GammaObjective {
    pub fn with_glm_hessian(mut self, hessian: GlmHessian) -> Self {
        self.hessian = hessian;
        self
    }
}

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
            let hess = match self.hessian {
                GlmHessian::Exact => y_over_mu.max(1e-7),
                GlmHessian::Bounded => bounded_log_link_hessian(2.0, y_over_mu, 1.0, 1.0).max(1e-7),
            } * weight;
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
///
/// The default hessian is the simplified `w·μ^(2−p)`; with
/// [`GlmHessian::Bounded`] it is `w·(p·y·μ^(1−p) + (3−p)·μ^(2−p))/3`, which
/// interpolates the bounded Poisson and Gamma curvatures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieObjective {
    pub variance_power: f32,
    hessian: GlmHessian,
}

impl TweedieObjective {
    pub fn new(variance_power: f32) -> EngineResult<Self> {
        if !variance_power.is_finite() || variance_power <= 1.0 || variance_power >= 2.0 {
            return Err(EngineError::InvalidConfig(format!(
                "Tweedie variance_power must satisfy 1 < p < 2 (got {variance_power}); \
                 use PoissonObjective for p=1 and GammaObjective for p=2"
            )));
        }
        Ok(Self {
            variance_power,
            hessian: GlmHessian::Exact,
        })
    }

    pub fn with_glm_hessian(mut self, hessian: GlmHessian) -> Self {
        self.hessian = hessian;
        self
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
            let hess = match self.hessian {
                GlmHessian::Exact => mu_2mp.max(1e-7),
                GlmHessian::Bounded => {
                    bounded_log_link_hessian(p, targets[index], mu_1mp, mu_2mp).max(1e-7)
                }
            } * weight;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_glm_hessians_match_closed_forms() {
        use alloygbm_core::GlmHessian::Bounded;
        let predictions = vec![-1.0_f32, 0.0, 0.5, 2.0];
        let targets = vec![3.0_f32, 0.5, 1.0, 2.0];
        let weights = vec![1.0_f32, 2.0, 0.5, 1.5];
        let poisson = PoissonObjective::new(0.7)
            .with_glm_hessian(Bounded)
            .compute_gradients(&predictions, &targets, Some(&weights))
            .expect("poisson");
        let gamma = GammaObjective::default()
            .with_glm_hessian(Bounded)
            .compute_gradients(&predictions, &targets, Some(&weights))
            .expect("gamma");
        let p = 1.4_f32;
        let tweedie = TweedieObjective::new(p)
            .expect("tweedie")
            .with_glm_hessian(Bounded)
            .compute_gradients(&predictions, &targets, Some(&weights))
            .expect("tweedie");
        for i in 0..predictions.len() {
            let (mu, y, w) = (predictions[i].exp(), targets[i], weights[i]);
            // max_delta_step is ignored under the bounded Hessian.
            assert!((poisson[i].hess - w * (2.0 * mu + y) / 3.0).abs() < 1e-5);
            assert!((poisson[i].grad - w * (mu - y)).abs() < 1e-5);
            assert!((gamma[i].hess - w * (2.0 * y / mu + 1.0) / 3.0).abs() < 1e-5);
            assert!((gamma[i].grad - w * (1.0 - y / mu)).abs() < 1e-5);
            let a = y * mu.powf(1.0 - p);
            let b = mu.powf(2.0 - p);
            assert!((tweedie[i].hess - w * (p * a + (3.0 - p) * b) / 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn bounded_tweedie_hessian_interpolates_poisson_and_gamma() {
        use alloygbm_core::GlmHessian::Bounded;
        let predictions = vec![-0.5_f32, 0.3, 1.2];
        let targets = vec![2.0_f32, 0.7, 4.0];
        let hess = |grads: Vec<GradientPair>| grads.iter().map(|g| g.hess).collect::<Vec<_>>();
        let poisson = hess(
            PoissonObjective::default()
                .with_glm_hessian(Bounded)
                .compute_gradients(&predictions, &targets, None)
                .unwrap(),
        );
        let gamma = hess(
            GammaObjective::default()
                .with_glm_hessian(Bounded)
                .compute_gradients(&predictions, &targets, None)
                .unwrap(),
        );
        let near = |p: f32| {
            hess(
                TweedieObjective::new(p)
                    .unwrap()
                    .with_glm_hessian(Bounded)
                    .compute_gradients(&predictions, &targets, None)
                    .unwrap(),
            )
        };
        for (t, want) in near(1.0001).iter().zip(&poisson) {
            assert!((t - want).abs() < 1e-3 * want.max(1.0), "{t} vs {want}");
        }
        for (t, want) in near(1.9999).iter().zip(&gamma) {
            assert!((t - want).abs() < 1e-3 * want.max(1.0), "{t} vs {want}");
        }
    }

    /// One Newton step on a constant (intercept) model: `-Σg / Σh`.
    fn intercept_newton_step<O: ObjectiveOps>(objective: &O, eta: f32, targets: &[f32]) -> f32 {
        let predictions = vec![eta; targets.len()];
        let grads = objective
            .compute_gradients(&predictions, targets, None)
            .unwrap();
        let g: f32 = grads.iter().map(|p| p.grad).sum();
        let h: f32 = grads.iter().map(|p| p.hess).sum();
        -g / h
    }

    #[test]
    fn bounded_glm_hessians_cap_the_intercept_step() {
        use alloygbm_core::GlmHessian::{Bounded, Exact};
        // Mean targets far above and far below the current mean, where the exact
        // Hessian's step runs away and the bounded step saturates.
        let high = vec![1.0e4_f32; 8];
        let low = vec![1.0e-4_f32; 8];
        let poisson = PoissonObjective::new(0.0);
        let bounded_poisson = poisson.with_glm_hessian(Bounded);
        assert!(intercept_newton_step(&poisson, 0.0, &high) > 100.0);
        let up = intercept_newton_step(&bounded_poisson, 0.0, &high);
        let down = intercept_newton_step(&bounded_poisson, 0.0, &low);
        assert!(up > 2.9 && up < 3.0, "{up}");
        assert!(down < -1.4 && down > -1.5, "{down}");

        let gamma = GammaObjective::default().with_glm_hessian(Exact);
        let bounded_gamma = GammaObjective::default().with_glm_hessian(Bounded);
        // Exact Gamma Newton step 1 - mu/y explodes when y is tiny.
        assert!(intercept_newton_step(&gamma, 0.0, &low) < -100.0);
        let up = intercept_newton_step(&bounded_gamma, 0.0, &high);
        let down = intercept_newton_step(&bounded_gamma, 0.0, &low);
        assert!(up > 1.49 && up < 1.5, "{up}");
        assert!(down < -2.9 && down > -3.0, "{down}");
    }

    #[test]
    fn glm_hessian_parse_round_trips() {
        use alloygbm_core::{GlmHessian, GrnForm};
        assert_eq!(GlmHessian::parse("exact").unwrap(), GlmHessian::Exact);
        assert_eq!(GlmHessian::parse("bounded").unwrap(), GlmHessian::Bounded);
        assert!(GlmHessian::parse("halley").is_err());
        assert_eq!(GrnForm::parse("per_leaf").unwrap(), GrnForm::PerLeaf);
        assert_eq!(GrnForm::parse("per_row").unwrap(), GrnForm::PerRow);
        assert!(GrnForm::parse("per_tree").is_err());
    }
}
