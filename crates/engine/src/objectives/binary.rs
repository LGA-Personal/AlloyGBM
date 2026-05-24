use alloygbm_core::GradientPair;

use crate::binary_crossentropy_loss;
use crate::error::{EngineError, EngineResult};
use crate::traits::ObjectiveOps;

/// Binary cross-entropy (log loss) objective for binary classification.
/// Targets must be 0.0 or 1.0. Predictions are in log-odds (logit) space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryCrossEntropyObjective;

/// Numerically stable sigmoid: avoids overflow for large positive/negative inputs.
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

impl ObjectiveOps for BinaryCrossEntropyObjective {
    fn objective_name(&self) -> &str {
        "binary_crossentropy"
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
        // Compute weighted mean of targets, then convert to log-odds.
        let (positive_weight, total_weight) = if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
            let mut pos_w = 0.0_f32;
            let mut tot_w = 0.0_f32;
            for (&target, &weight) in targets.iter().zip(weights) {
                if !weight.is_finite() || weight <= 0.0 {
                    return Err(EngineError::ContractViolation(
                        "sample weights must be finite and > 0".to_string(),
                    ));
                }
                pos_w += target * weight;
                tot_w += weight;
            }
            (pos_w, tot_w)
        } else {
            let pos = targets.iter().sum::<f32>();
            (pos, targets.len() as f32)
        };
        if total_weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weight sum must be greater than 0".to_string(),
            ));
        }
        let p = (positive_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
        // log-odds: log(p / (1 - p))
        Ok((p / (1.0 - p)).ln())
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
        if let Some(weights) = sample_weights
            && weights.len() != targets.len()
        {
            return Err(EngineError::ContractViolation(format!(
                "weights length {} does not match targets length {}",
                weights.len(),
                targets.len()
            )));
        }

        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let p = sigmoid(predictions[index]);
            // grad = (p - y) * w, hess = p * (1 - p) * w
            let grad = (p - targets[index]) * weight;
            let hess = (p * (1.0 - p)).max(1e-7) * weight;
            gradients.push(GradientPair::new(grad, hess)?);
        }
        Ok(gradients)
    }

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
        if let Some(weights) = sample_weights
            && weights.len() != targets.len()
        {
            return Err(EngineError::ContractViolation(format!(
                "weights length {} does not match targets length {}",
                weights.len(),
                targets.len()
            )));
        }
        buffer.clear();
        if buffer.capacity() < predictions.len() {
            buffer.reserve(predictions.len() - buffer.capacity());
        }
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let p = sigmoid(predictions[index]);
            let grad = (p - targets[index]) * weight;
            let hess = (p * (1.0 - p)).max(1e-7) * weight;
            buffer.push(GradientPair { grad, hess });
        }
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        binary_crossentropy_loss(predictions, targets, sample_weights)
    }
}
