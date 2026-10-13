use alloygbm_core::GradientPair;
use rayon::prelude::*;

use crate::error::{EngineError, EngineResult};

/// Smallest row block one thread takes when computing softmax normalizers.
const SOFTMAX_MIN_ROWS_PER_TASK: usize = 2_048;

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
            if !weight.is_finite() || weight < 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and non-negative".to_string(),
                ));
            }
            let grad = (p_k - indicator) * weight;
            let hess = (p_k * (1.0 - p_k)).max(1e-7) * weight;

            buffer.push(GradientPair { grad, hess });
        }

        Ok(())
    }

    /// Gradients for every class at once: `result[k]` equals what
    /// [`Self::compute_gradients_for_class`] returns for class `k`.
    ///
    /// The per-class method recomputes each row's softmax normalizer for
    /// every class, so a round costs K * K exponentials per row. Here the
    /// row's max logit and exp-sum are computed once (in the same order, so
    /// to the same bits) and each class then needs one exponential: the
    /// gradients are bit-identical. Rows and classes are independent, so
    /// both passes run on several threads without changing any value.
    pub fn compute_gradients_all_classes(
        &self,
        class_predictions: &[Vec<f32>],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<Vec<GradientPair>>> {
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
        if let Some(w) = sample_weights {
            if w.len() != n {
                return Err(EngineError::ContractViolation(format!(
                    "sample_weights length {} does not match predictions length {n}",
                    w.len()
                )));
            }
            if w.iter().any(|&weight| !weight.is_finite() || weight < 0.0) {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and non-negative".to_string(),
                ));
            }
        }

        // Per row: (max logit, sum of exp(logit - max)), accumulated over the
        // classes in index order exactly as the per-class method does.
        let normalizers: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .with_min_len(SOFTMAX_MIN_ROWS_PER_TASK)
            .map(|i| {
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
                (max_logit, sum_exp)
            })
            .collect();

        Ok((0..k)
            .into_par_iter()
            .map(|class_k| {
                let class_preds = &class_predictions[class_k];
                (0..n)
                    .map(|i| {
                        let (max_logit, sum_exp) = normalizers[i];
                        let p_k = (class_preds[i] - max_logit).exp() / sum_exp;
                        let indicator = if (targets[i] as usize) == class_k {
                            1.0
                        } else {
                            0.0
                        };
                        let weight = sample_weights.map_or(1.0, |w| w[i]);
                        let grad = (p_k - indicator) * weight;
                        let hess = (p_k * (1.0 - p_k)).max(1e-7) * weight;
                        GradientPair { grad, hess }
                    })
                    .collect()
            })
            .collect())
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

#[cfg(test)]
mod all_classes_tests {
    use super::MultiClassSoftmaxObjective;

    #[test]
    fn all_classes_matches_per_class_bit_for_bit() {
        let k = 5;
        let n = 5_000;
        let objective = MultiClassSoftmaxObjective::new(k).expect("objective");
        let predictions: Vec<Vec<f32>> = (0..k)
            .map(|class| {
                (0..n)
                    .map(|row| ((row * 31 + class * 7) % 97) as f32 * 0.13 - 6.0)
                    .collect()
            })
            .collect();
        let targets: Vec<f32> = (0..n).map(|row| (row % k) as f32).collect();
        let weights: Vec<f32> = (0..n).map(|row| 0.5 + (row % 3) as f32).collect();
        for sample_weights in [None, Some(weights.as_slice())] {
            let all = objective
                .compute_gradients_all_classes(&predictions, &targets, sample_weights)
                .expect("all classes");
            for (class_k, gradients) in all.iter().enumerate() {
                let mut expected = Vec::new();
                objective
                    .compute_gradients_for_class(
                        &predictions,
                        &targets,
                        sample_weights,
                        class_k,
                        &mut expected,
                    )
                    .expect("per class");
                assert_eq!(gradients.len(), expected.len());
                for (got, want) in gradients.iter().zip(&expected) {
                    assert_eq!(got.grad.to_bits(), want.grad.to_bits());
                    assert_eq!(got.hess.to_bits(), want.hess.to_bits());
                }
            }
        }
        let bad_weights = vec![-1.0_f32; n];
        assert!(
            objective
                .compute_gradients_all_classes(&predictions, &targets, Some(&bad_weights))
                .is_err()
        );
    }
}
