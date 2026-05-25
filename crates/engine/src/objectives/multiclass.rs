use alloygbm_core::GradientPair;

use crate::error::{EngineError, EngineResult};

/// Softmax cross-entropy objective for K-class classification.
///
/// Does NOT implement [`ObjectiveOps`] because that trait is fundamentally
/// single-output. Multi-class training requires K prediction arrays and K
/// gradient arrays computed jointly (softmax couples all classes).
pub struct MultiClassSoftmaxObjective {
    pub num_classes: usize,
}

impl MultiClassSoftmaxObjective {
    pub fn new(num_classes: usize) -> EngineResult<Self> {
        if num_classes < 2 {
            return Err(EngineError::InvalidConfig(format!(
                "multiclass_softmax requires at least 2 classes, got {num_classes}"
            )));
        }
        Ok(Self { num_classes })
    }

    pub fn objective_name(&self) -> &str {
        "multiclass_softmax"
    }

    /// Returns K initial predictions (all zeros → uniform 1/K under softmax).
    pub fn initial_predictions(&self) -> Vec<f32> {
        vec![0.0; self.num_classes]
    }

    /// Compute gradients for a single class given all K prediction arrays.
    ///
    /// `class_predictions[k][i]` is the raw logit for class k, sample i.
    pub fn compute_gradients_for_class(
        &self,
        class_predictions: &[Vec<f32>],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        class_k: usize,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let k = self.num_classes;
        if class_predictions.len() != k {
            return Err(EngineError::ContractViolation(format!(
                "expected {} class prediction arrays, got {}",
                k,
                class_predictions.len()
            )));
        }
        let n = class_predictions[0].len();
        if targets.len() != n {
            return Err(EngineError::ContractViolation(format!(
                "targets length {} does not match predictions length {n}",
                targets.len()
            )));
        }
        if let Some(w) = sample_weights
            && w.len() != n
        {
            return Err(EngineError::ContractViolation(format!(
                "sample_weights length {} does not match predictions length {n}",
                w.len()
            )));
        }

        buffer.clear();
        buffer.reserve(n.saturating_sub(buffer.capacity()));

        for i in 0..n {
            // Numerically stable softmax: subtract max
            let mut max_logit = f32::NEG_INFINITY;
            for class_preds in class_predictions.iter().take(k) {
                let v = class_preds[i];
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            for class_preds in class_predictions.iter().take(k) {
                sum_exp += (class_preds[i] - max_logit).exp();
            }
            let p_k = (class_predictions[class_k][i] - max_logit).exp() / sum_exp;

            let indicator = if (targets[i] as usize) == class_k {
                1.0
            } else {
                0.0
            };
            let weight = sample_weights.map_or(1.0, |w| w[i]);
            let grad = (p_k - indicator) * weight;
            let hess = (p_k * (1.0 - p_k) * weight).max(1e-7);

            buffer.push(GradientPair { grad, hess });
        }

        Ok(())
    }

    /// Multi-class cross-entropy loss.
    pub fn loss(
        &self,
        class_predictions: &[Vec<f32>],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        let k = self.num_classes;
        if class_predictions.len() != k {
            return Err(EngineError::ContractViolation(format!(
                "expected {} class prediction arrays, got {}",
                k,
                class_predictions.len()
            )));
        }
        let n = class_predictions[0].len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for i in 0..n {
            let target_class = targets[i] as usize;
            let weight = sample_weights.map_or(1.0_f64, |w| w[i] as f64);

            // log-sum-exp trick for numerical stability
            let mut max_logit = f32::NEG_INFINITY;
            for class_preds in class_predictions.iter().take(k) {
                let v = class_preds[i];
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum_exp = 0.0_f64;
            for class_preds in class_predictions.iter().take(k) {
                sum_exp += ((class_preds[i] - max_logit) as f64).exp();
            }
            let log_p = (class_predictions[target_class][i] - max_logit) as f64 - sum_exp.ln();

            total_loss -= log_p * weight;
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return Ok(0.0);
        }
        Ok((total_loss / total_weight) as f32)
    }
}
