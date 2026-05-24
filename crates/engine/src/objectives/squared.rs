use alloygbm_core::GradientPair;

use crate::error::{EngineError, EngineResult};
use crate::squared_error_loss;
use crate::traits::ObjectiveOps;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquaredErrorObjective;

impl ObjectiveOps for SquaredErrorObjective {
    fn objective_name(&self) -> &str {
        "squared_error"
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

        if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for (target, weight) in targets.iter().zip(weights) {
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(EngineError::ContractViolation(
                        "sample weights must be finite and > 0".to_string(),
                    ));
                }
                weighted_sum += target * weight;
                weight_sum += weight;
            }
            if weight_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weight sum must be greater than 0".to_string(),
                ));
            }
            return Ok(weighted_sum / weight_sum);
        }

        let sum = targets.iter().sum::<f32>();
        Ok(sum / targets.len() as f32)
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
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let residual = predictions[index] - targets[index];
            gradients.push(GradientPair::new(residual * weight, weight)?);
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
            let residual = predictions[index] - targets[index];
            buffer.push(GradientPair {
                grad: residual * weight,
                hess: weight,
            });
        }
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        squared_error_loss(predictions, targets, sample_weights)
    }

    fn supports_pre_target_neutralization(&self) -> bool {
        true
    }
}
