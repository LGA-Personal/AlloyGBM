//! Post-growth leaf refinement helpers.
//!
//! Two flavours:
//!
//! * `refine_regression_leaf_values` — recomputes each leaf's value
//!   as the (clamped) weighted mean of the residual-without-this-tree
//!   across the rows that land in it. Used by the regression
//!   trainer's optional leaf-refinement pass.
//!
//! * `refine_quantile_leaf_values` — replaces the Newton-Raphson
//!   leaf prediction with the empirical α-quantile of residuals over
//!   all rows in each leaf, for the quantile objective
//!   (introduced in v0.11.1). Uses the full dataset (not the
//!   subsampled subset) to minimise estimation variance of the empirical
//!   quantile.
//!
//! The two share a small set of recursive walkers
//! (`fill_refined_*`) that map terminal-leaf statistics back up the
//! tree to compute parent absolute outputs, plus per-stump
//! recomputation of the relative left/right leaf deltas at the end.

use std::collections::HashMap;

use alloygbm_core::{BinnedMatrix, LeafValue};

use crate::error::{EngineError, EngineResult};
use crate::objectives::weighted_quantile;
use crate::round::apply_tree_to_binned_predictions;
use crate::tree_node::{decode_tree_node_id, left_child_node_id, right_child_node_id};
use crate::types::TrainedStump;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LeafRefinementStats {
    weighted_sum: f32,
    weight_sum: f32,
}

impl LeafRefinementStats {
    fn push(&mut self, value: f32, weight: f32) {
        self.weighted_sum += value * weight;
        self.weight_sum += weight;
    }
}

pub(crate) fn refine_regression_leaf_values(
    baseline_prediction: f32,
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    binned_matrix: &BinnedMatrix,
    stumps: &mut [TrainedStump],
    stumps_per_completed_round: &[usize],
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if stumps.is_empty() || stumps_per_completed_round.is_empty() {
        return Ok(());
    }
    if targets.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "targets length {} does not match binned row_count {}",
            targets.len(),
            binned_matrix.row_count
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

    let mut ensemble_predictions = vec![0.0_f32; targets.len()];
    for &round_stump_count in stumps_per_completed_round {
        if round_stump_count == 0 {
            continue;
        }
    }

    let mut cursor = 0_usize;
    for &round_stump_count in stumps_per_completed_round {
        let round_end = cursor.checked_add(round_stump_count).ok_or_else(|| {
            EngineError::ContractViolation("round stump count overflow".to_string())
        })?;
        if round_end > stumps.len() {
            return Err(EngineError::ContractViolation(
                "round stump counts exceed trained stump count".to_string(),
            ));
        }
        apply_tree_to_binned_predictions(
            &mut ensemble_predictions,
            binned_matrix,
            &stumps[cursor..round_end],
            None,
        )?;
        cursor = round_end;
    }
    if cursor != stumps.len() {
        return Err(EngineError::ContractViolation(
            "round stump counts do not cover all trained stumps".to_string(),
        ));
    }

    let mut cursor = 0_usize;
    for &round_stump_count in stumps_per_completed_round {
        let round_end = cursor.checked_add(round_stump_count).ok_or_else(|| {
            EngineError::ContractViolation("round stump count overflow".to_string())
        })?;
        if round_stump_count == 0 {
            cursor = round_end;
            continue;
        }

        let round_stumps = &mut stumps[cursor..round_end];
        let old_tree_predictions = tree_predictions_for_binned_rows(binned_matrix, round_stumps)?;
        let residual_without_tree = targets
            .iter()
            .enumerate()
            .map(|(row_index, target)| {
                target
                    - baseline_prediction
                    - (ensemble_predictions[row_index] - old_tree_predictions[row_index])
            })
            .collect::<Vec<_>>();
        let refined_tree = refine_tree_stumps(
            binned_matrix,
            round_stumps,
            &residual_without_tree,
            sample_weights,
            max_abs_leaf_value,
        )?;
        let new_tree_predictions = tree_predictions_for_binned_rows(binned_matrix, &refined_tree)?;
        for row_index in 0..ensemble_predictions.len() {
            ensemble_predictions[row_index] +=
                new_tree_predictions[row_index] - old_tree_predictions[row_index];
        }
        round_stumps.clone_from_slice(&refined_tree);
        cursor = round_end;
    }

    Ok(())
}

pub(crate) fn tree_predictions_for_binned_rows(
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
) -> EngineResult<Vec<f32>> {
    let mut predictions = vec![0.0_f32; binned_matrix.row_count];
    apply_tree_to_binned_predictions(&mut predictions, binned_matrix, stumps, None)?;
    Ok(predictions)
}

fn refine_tree_stumps(
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    residual_without_tree: &[f32],
    sample_weights: Option<&[f32]>,
    max_abs_leaf_value: f32,
) -> EngineResult<Vec<TrainedStump>> {
    if stumps.is_empty() {
        return Ok(Vec::new());
    }
    if residual_without_tree.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "residual length {} does not match binned row_count {}",
            residual_without_tree.len(),
            binned_matrix.row_count
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != residual_without_tree.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match residual length {}",
            weights.len(),
            residual_without_tree.len()
        )));
    }

    let mut stumps_by_local = HashMap::new();
    for stump in stumps {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        stumps_by_local.insert(local_node_id, stump);
    }

    let mut current_absolute_outputs = HashMap::new();
    current_absolute_outputs.insert(0_u32, 0.0_f32);
    populate_child_absolute_outputs(0, &stumps_by_local, &mut current_absolute_outputs)?;

    let mut terminal_stats = HashMap::<u32, LeafRefinementStats>::new();
    for row_index in 0..binned_matrix.row_count {
        let terminal_local_node_id =
            terminal_local_node_id_for_row(row_index, binned_matrix, &stumps_by_local)?;
        let weight = sample_weights.map_or(1.0, |weights| weights[row_index]);
        terminal_stats
            .entry(terminal_local_node_id)
            .or_default()
            .push(residual_without_tree[row_index], weight);
    }

    let mut refined_absolute_outputs = HashMap::new();
    refined_absolute_outputs.insert(0_u32, 0.0_f32);
    fill_refined_child_absolute_outputs(
        0,
        &stumps_by_local,
        &terminal_stats,
        &current_absolute_outputs,
        max_abs_leaf_value,
        &mut refined_absolute_outputs,
    )?;

    let mut refined_stumps = stumps.to_vec();
    for stump in &mut refined_stumps {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let parent_absolute = refined_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0);
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_absolute = refined_absolute_outputs
            .get(&left_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.left_leaf_value.as_scalar());
        let right_absolute = refined_absolute_outputs
            .get(&right_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.right_leaf_value.as_scalar());
        stump.left_leaf_value = LeafValue::Scalar(left_absolute - parent_absolute);
        stump.right_leaf_value = LeafValue::Scalar(right_absolute - parent_absolute);
    }

    Ok(refined_stumps)
}

fn populate_child_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<()> {
    let Some(stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(());
    };
    let parent_absolute = absolute_outputs.get(&local_node_id).copied().unwrap_or(0.0);
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;
    absolute_outputs.insert(
        left_local_node_id,
        parent_absolute + stump.left_leaf_value.as_scalar(),
    );
    absolute_outputs.insert(
        right_local_node_id,
        parent_absolute + stump.right_leaf_value.as_scalar(),
    );
    populate_child_absolute_outputs(left_local_node_id, stumps_by_local, absolute_outputs)?;
    populate_child_absolute_outputs(right_local_node_id, stumps_by_local, absolute_outputs)?;
    Ok(())
}

fn terminal_local_node_id_for_row(
    row_index: usize,
    binned_matrix: &BinnedMatrix,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
) -> EngineResult<u32> {
    let mut local_node_id = 0_u32;
    loop {
        let Some(stump) = stumps_by_local.get(&local_node_id) else {
            return Err(EngineError::ContractViolation(format!(
                "tree is missing split for local node {local_node_id}"
            )));
        };
        let feature_index = stump.split.feature_index as usize;
        if feature_index >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                stump.split.feature_index, binned_matrix.feature_count
            )));
        }
        let cell_index = row_index
            .checked_mul(binned_matrix.feature_count)
            .and_then(|base| base.checked_add(feature_index))
            .ok_or_else(|| {
                EngineError::ContractViolation("binned cell index overflow".to_string())
            })?;
        if cell_index >= binned_matrix.bins_adaptive.len() {
            return Err(EngineError::ContractViolation(format!(
                "binned cell index {cell_index} is out of bounds for bins length {}",
                binned_matrix.bins_adaptive.len()
            )));
        }
        let bin = binned_matrix.row_bin(cell_index);
        let next_local_node_id = if bin <= stump.split.threshold_bin {
            left_child_node_id(local_node_id)?
        } else {
            right_child_node_id(local_node_id)?
        };
        if !stumps_by_local.contains_key(&next_local_node_id) {
            return Ok(next_local_node_id);
        }
        local_node_id = next_local_node_id;
    }
}

fn fill_refined_child_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    terminal_stats: &HashMap<u32, LeafRefinementStats>,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    let Some(_stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(LeafRefinementStats::default());
    };
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;

    let left_stats = fill_refined_subtree_absolute_output(
        left_local_node_id,
        stumps_by_local,
        terminal_stats,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;
    let right_stats = fill_refined_subtree_absolute_output(
        right_local_node_id,
        stumps_by_local,
        terminal_stats,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;

    let mut subtree_stats = left_stats;
    subtree_stats.weighted_sum += right_stats.weighted_sum;
    subtree_stats.weight_sum += right_stats.weight_sum;
    Ok(subtree_stats)
}

fn fill_refined_subtree_absolute_output(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    terminal_stats: &HashMap<u32, LeafRefinementStats>,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    if stumps_by_local.contains_key(&local_node_id) {
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_stats = fill_refined_subtree_absolute_output(
            left_local_node_id,
            stumps_by_local,
            terminal_stats,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let right_stats = fill_refined_subtree_absolute_output(
            right_local_node_id,
            stumps_by_local,
            terminal_stats,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let total_weight = left_stats.weight_sum + right_stats.weight_sum;
        let absolute_output = if total_weight > 0.0 {
            ((left_stats.weighted_sum + right_stats.weighted_sum) / total_weight)
                .clamp(-max_abs_leaf_value, max_abs_leaf_value)
        } else {
            current_absolute_outputs
                .get(&local_node_id)
                .copied()
                .unwrap_or(0.0)
        };
        refined_absolute_outputs.insert(local_node_id, absolute_output);
        return Ok(LeafRefinementStats {
            weighted_sum: absolute_output * total_weight,
            weight_sum: total_weight,
        });
    }

    let stats = terminal_stats
        .get(&local_node_id)
        .copied()
        .unwrap_or_default();
    let absolute_output = if stats.weight_sum > 0.0 {
        (stats.weighted_sum / stats.weight_sum).clamp(-max_abs_leaf_value, max_abs_leaf_value)
    } else {
        current_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0)
    };
    refined_absolute_outputs.insert(local_node_id, absolute_output);
    Ok(LeafRefinementStats {
        weighted_sum: absolute_output * stats.weight_sum,
        weight_sum: stats.weight_sum,
    })
}

struct LeafResiduals {
    residuals: Vec<f32>,
    weights: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_quantile_leaf_values(
    stumps: &mut [TrainedStump],
    binned_matrix: &BinnedMatrix,
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    alpha: f32,
    learning_rate: f32,
    max_abs_leaf_value: f32,
    raw_features: Option<(&[f32], usize)>,
    depth_penalty_base: Option<f32>,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    if targets.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "targets length {} does not match binned row_count {}",
            targets.len(),
            binned_matrix.row_count
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

    let mut stumps_by_local = HashMap::new();
    for stump in stumps.iter() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        stumps_by_local.insert(local_node_id, stump);
    }

    let mut current_absolute_outputs = HashMap::new();
    current_absolute_outputs.insert(0_u32, 0.0_f32);
    populate_child_absolute_outputs(0, &stumps_by_local, &mut current_absolute_outputs)?;

    // We map every row in the full training dataset to its terminal leaf to collect residuals.
    // When row_subsample < 1.0, split-finding is performed on a subsampled subset of rows,
    // but we use the entire training set for the final quantile leaf refinement step
    // to minimize estimation variance of the empirical quantile.
    let mut leaf_residuals: HashMap<u32, LeafResiduals> = HashMap::new();
    for row_index in 0..binned_matrix.row_count {
        let terminal_local_node_id =
            terminal_local_node_id_for_row(row_index, binned_matrix, &stumps_by_local)?;

        // If the routed terminal leaf is Linear, evaluate its linear terms.
        let mut lin_val = 0.0_f32;
        if terminal_local_node_id > 0 {
            let parent_node_id = (terminal_local_node_id - 1) / 2;
            let is_left = (terminal_local_node_id % 2) != 0;
            if let Some(parent_stump) = stumps_by_local.get(&parent_node_id) {
                let leaf_val = if is_left {
                    &parent_stump.left_leaf_value
                } else {
                    &parent_stump.right_leaf_value
                };
                if let LeafValue::Linear(lin) = leaf_val {
                    if let Some((raw, fc)) = raw_features {
                        let row_offset = row_index * fc;
                        lin_val = lin.eval(raw, row_offset) - lin.intercept;
                    }
                }
            }
        }

        let res = targets[row_index] - predictions[row_index] - lin_val;
        let entry = leaf_residuals
            .entry(terminal_local_node_id)
            .or_insert_with(|| LeafResiduals {
                residuals: Vec::new(),
                weights: sample_weights.map(|_| Vec::new()),
            });
        entry.residuals.push(res);
        if let (Some(w_vec), Some(weights)) = (&mut entry.weights, sample_weights) {
            w_vec.push(weights[row_index]);
        }
    }

    let mut refined_absolute_outputs = HashMap::new();
    refined_absolute_outputs.insert(0_u32, 0.0_f32);

    fill_refined_child_quantile_absolute_outputs(
        0,
        &stumps_by_local,
        &leaf_residuals,
        alpha,
        &current_absolute_outputs,
        max_abs_leaf_value,
        &mut refined_absolute_outputs,
    )?;

    for stump in stumps.iter_mut() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let parent_absolute = refined_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0);
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_absolute = refined_absolute_outputs
            .get(&left_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.left_leaf_value.as_scalar());
        let right_absolute = refined_absolute_outputs
            .get(&right_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.right_leaf_value.as_scalar());

        let parent_depth = (32 - (local_node_id + 1).leading_zeros() - 1) as f32;
        let child_depth = parent_depth + 1.0;
        let depth_penalty = depth_penalty_base.map_or(1.0, |base| base.powf(child_depth / 3.0));
        let effective_lr = learning_rate * depth_penalty;

        let dl = (left_absolute - parent_absolute) * effective_lr;
        let dr = (right_absolute - parent_absolute) * effective_lr;

        match &mut stump.left_leaf_value {
            LeafValue::Scalar(_) => {
                stump.left_leaf_value = LeafValue::Scalar(dl);
            }
            LeafValue::Linear(lin) => {
                lin.intercept = dl;
                lin.weights.iter_mut().for_each(|w| *w *= effective_lr);
            }
        }
        match &mut stump.right_leaf_value {
            LeafValue::Scalar(_) => {
                stump.right_leaf_value = LeafValue::Scalar(dr);
            }
            LeafValue::Linear(lin) => {
                lin.intercept = dr;
                lin.weights.iter_mut().for_each(|w| *w *= effective_lr);
            }
        }
    }

    Ok(())
}

fn fill_refined_child_quantile_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    leaf_residuals: &HashMap<u32, LeafResiduals>,
    alpha: f32,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    let Some(_stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(LeafRefinementStats::default());
    };
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;

    let left_stats = fill_refined_subtree_quantile_absolute_output(
        left_local_node_id,
        stumps_by_local,
        leaf_residuals,
        alpha,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;
    let right_stats = fill_refined_subtree_quantile_absolute_output(
        right_local_node_id,
        stumps_by_local,
        leaf_residuals,
        alpha,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;

    let mut subtree_stats = left_stats;
    subtree_stats.weighted_sum += right_stats.weighted_sum;
    subtree_stats.weight_sum += right_stats.weight_sum;
    Ok(subtree_stats)
}

fn fill_refined_subtree_quantile_absolute_output(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    leaf_residuals: &HashMap<u32, LeafResiduals>,
    alpha: f32,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    if stumps_by_local.contains_key(&local_node_id) {
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_stats = fill_refined_subtree_quantile_absolute_output(
            left_local_node_id,
            stumps_by_local,
            leaf_residuals,
            alpha,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let right_stats = fill_refined_subtree_quantile_absolute_output(
            right_local_node_id,
            stumps_by_local,
            leaf_residuals,
            alpha,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let total_weight = left_stats.weight_sum + right_stats.weight_sum;
        let absolute_output = if total_weight > 0.0 {
            ((left_stats.weighted_sum + right_stats.weighted_sum) / total_weight)
                .clamp(-max_abs_leaf_value, max_abs_leaf_value)
        } else {
            current_absolute_outputs
                .get(&local_node_id)
                .copied()
                .unwrap_or(0.0)
        };
        refined_absolute_outputs.insert(local_node_id, absolute_output);
        return Ok(LeafRefinementStats {
            weighted_sum: absolute_output * total_weight,
            weight_sum: total_weight,
        });
    }

    let (q_val, weight_sum) = if let Some(lr) = leaf_residuals.get(&local_node_id) {
        if let Some(ref w_vec) = lr.weights {
            let total_w: f32 = w_vec.iter().sum();
            if total_w > 0.0 {
                let q = weighted_quantile(&lr.residuals, Some(w_vec), alpha)?;
                (q, total_w)
            } else {
                (0.0, 0.0)
            }
        } else {
            let count = lr.residuals.len();
            if count > 0 {
                let q = weighted_quantile(&lr.residuals, None, alpha)?;
                (q, count as f32)
            } else {
                (0.0, 0.0)
            }
        }
    } else {
        (0.0, 0.0)
    };

    let absolute_output = if weight_sum > 0.0 {
        q_val.clamp(-max_abs_leaf_value, max_abs_leaf_value)
    } else {
        current_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0)
    };
    refined_absolute_outputs.insert(local_node_id, absolute_output);
    Ok(LeafRefinementStats {
        weighted_sum: absolute_output * weight_sum,
        weight_sum,
    })
}
