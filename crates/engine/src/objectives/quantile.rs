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

    let weights = weights.expect("weighted branch requires weights");
    let mut pairs = Vec::with_capacity(values.len());
    let mut total_weight = 0.0_f64;
    for (value, weight) in values.iter().copied().zip(weights.iter().copied()) {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weights must be finite and > 0".to_string(),
            ));
        }
        if !value.is_finite() {
            return Err(EngineError::ContractViolation(
                "values must be finite".to_string(),
            ));
        }
        pairs.push(WeightedValue { value, weight });
        total_weight += weight as f64;
    }

    let threshold = alpha as f64 * total_weight;
    #[cfg(not(test))]
    let selected = weighted_quantile_select(&mut pairs, threshold);
    #[cfg(test)]
    let selected = weighted_quantile_select(&mut pairs, threshold, &mut 0);
    Ok(selected)
}

#[derive(Clone, Copy)]
struct WeightedValue {
    value: f32,
    weight: f32,
}

/// Select the first value whose cumulative positive weight reaches `threshold`.
///
/// Each loop selects a median-by-count pivot, then partitions the active range
/// into values less than, equal to, and greater than that pivot while summing
/// the first two weight groups. The next active range is at most half as large,
/// making the partition work linear in the leaf size.
fn weighted_quantile_select(
    pairs: &mut [WeightedValue],
    mut threshold: f64,
    #[cfg(test)] partitioned_items: &mut usize,
) -> f32 {
    let mut start = 0;
    let mut end = pairs.len();

    loop {
        let active_len = end - start;
        debug_assert!(active_len > 0);
        let pivot_index = start + active_len / 2;
        pairs[start..end].select_nth_unstable_by(active_len / 2, |left, right| {
            left.value.total_cmp(&right.value)
        });
        let pivot_value = pairs[pivot_index].value;

        #[cfg(test)]
        {
            *partitioned_items += active_len;
        }
        let (less_end, equal_end, less_weight, equal_weight) =
            partition_weighted_values(pairs, start, end, pivot_value);

        if less_weight > 0.0 && threshold <= less_weight {
            end = less_end;
        } else if threshold <= less_weight + equal_weight || equal_end == end {
            return pivot_value;
        } else {
            threshold -= less_weight + equal_weight;
            start = equal_end;
        }
    }
}

fn partition_weighted_values(
    pairs: &mut [WeightedValue],
    start: usize,
    end: usize,
    pivot_value: f32,
) -> (usize, usize, f64, f64) {
    let mut less_end = start;
    let mut scan = start;
    let mut greater_start = end;
    let mut less_weight = 0.0_f64;
    let mut equal_weight = 0.0_f64;

    while scan < greater_start {
        match pairs[scan].value.total_cmp(&pivot_value) {
            Ordering::Less => {
                less_weight += pairs[scan].weight as f64;
                pairs.swap(less_end, scan);
                less_end += 1;
                scan += 1;
            }
            Ordering::Equal => {
                equal_weight += pairs[scan].weight as f64;
                scan += 1;
            }
            Ordering::Greater => {
                greater_start -= 1;
                pairs.swap(scan, greater_start);
            }
        }
    }

    (less_end, greater_start, less_weight, equal_weight)
}

#[cfg(test)]
fn weighted_quantile_with_partition_count_for_test(
    values: &[f32],
    weights: &[f32],
    alpha: f32,
) -> EngineResult<(f32, usize)> {
    let mut pairs = Vec::with_capacity(values.len());
    let mut total_weight = 0.0_f64;
    for (value, weight) in values.iter().copied().zip(weights.iter().copied()) {
        if !value.is_finite() {
            return Err(EngineError::ContractViolation(
                "values must be finite".to_string(),
            ));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weights must be finite and > 0".to_string(),
            ));
        }
        pairs.push(WeightedValue { value, weight });
        total_weight += weight as f64;
    }

    let mut partitioned_items = 0;
    let selected = weighted_quantile_select(
        &mut pairs,
        alpha as f64 * total_weight,
        &mut partitioned_items,
    );
    Ok((selected, partitioned_items))
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

#[cfg(test)]
mod tests {
    use super::{weighted_quantile, weighted_quantile_with_partition_count_for_test};

    fn sorted_weighted_quantile_reference(values: &[f32], weights: &[f32], alpha: f32) -> f32 {
        let mut pairs: Vec<(f32, f32)> = values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        let threshold = alpha as f64 * weights.iter().map(|&weight| weight as f64).sum::<f64>();
        let mut cumulative_weight = 0.0_f64;
        for (value, weight) in pairs {
            cumulative_weight += weight as f64;
            if cumulative_weight >= threshold {
                return value;
            }
        }
        values.iter().copied().max_by(f32::total_cmp).unwrap()
    }

    #[test]
    fn weighted_quantile_matches_sorted_reference_at_ties_and_weight_boundaries() {
        let values = [-4.0, -4.0, -1.0, 0.0, 0.0, 3.0, 8.0];
        let weights = [0.25, 2.75, 1.0, 4.0, 0.5, 3.0, 0.5];
        let total_weight: f32 = weights.iter().sum();
        let alphas = [
            0.0,
            0.25 / total_weight,
            3.0 / total_weight,
            4.0 / total_weight,
            8.5 / total_weight,
            0.99,
            1.0,
            1.25,
            f32::NAN,
        ];

        for alpha in alphas {
            assert_eq!(
                weighted_quantile(&values, Some(&weights), alpha).unwrap(),
                sorted_weighted_quantile_reference(&values, &weights, alpha),
                "alpha={alpha}"
            );
        }
    }

    #[test]
    fn weighted_quantile_partitions_large_leaves_in_linear_work() {
        let values: Vec<f32> = (0..65_537)
            .map(|index| ((index as u64 * 48_271) % 65_537) as f32)
            .collect();
        let weights: Vec<f32> = (0..values.len())
            .map(|index| 1.0 + (index % 11) as f32)
            .collect();
        let alpha = 0.93;

        let (selected, partitioned_items) =
            weighted_quantile_with_partition_count_for_test(&values, &weights, alpha).unwrap();

        assert_eq!(
            selected,
            weighted_quantile(&values, Some(&weights), alpha).unwrap()
        );
        assert!(
            partitioned_items < values.len() * 2,
            "weighted selection partitioned {partitioned_items} items for {} inputs",
            values.len()
        );
    }
}
