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
use rayon::prelude::*;

use crate::JointObjective;
use crate::MorphState;
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
        if cell_index >= binned_matrix.cell_count() {
            return Err(EngineError::ContractViolation(format!(
                "binned cell index {cell_index} is out of bounds for bins length {}",
                binned_matrix.cell_count()
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
    morph_scale_context: Option<(&MorphState, usize, u32)>,
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

        // If the routed terminal leaf has linear leaves along its path, evaluate their linear terms.
        let mut lin_val = 0.0_f32;
        if let Some((raw, fc)) = raw_features {
            let row_offset = row_index * fc;
            let mut curr_id = terminal_local_node_id;
            while curr_id > 0 {
                let parent_id = (curr_id - 1) / 2;
                let is_left = (curr_id % 2) != 0;
                if let Some(parent_stump) = stumps_by_local.get(&parent_id) {
                    let leaf_val = if is_left {
                        &parent_stump.left_leaf_value
                    } else {
                        &parent_stump.right_leaf_value
                    };
                    if let LeafValue::Linear(lin) = leaf_val {
                        // Candidate-round stumps always carry `tree_weight == 1.0`
                        // here — DART per-tree weights are stamped later, at commit
                        // (post-loss-check). The intercept calibration below is not
                        // tree_weight-scaled, so weighting only the slope would put
                        // slope and intercept on inconsistent scales; accumulate the
                        // unweighted linear contribution to match the build-time leaf.
                        lin_val += lin.eval(raw, row_offset) - lin.intercept;
                    }
                }
                curr_id = parent_id;
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

        let parent_depth = 32 - (local_node_id + 1).leading_zeros() - 1;
        let child_depth = parent_depth + 1;
        let effective_lr =
            morph_scale_context.map_or(learning_rate, |(state, iteration, total)| {
                state
                    .leaf_scale_for_depth(iteration, total, child_depth)
                    .total
            });

        let dl = (left_absolute - parent_absolute) * effective_lr;
        let dr = (right_absolute - parent_absolute) * effective_lr;

        match &mut stump.left_leaf_value {
            LeafValue::Scalar(_) => {
                stump.left_leaf_value = LeafValue::Scalar(dl);
            }
            LeafValue::Linear(lin) => {
                lin.intercept = dl;
            }
        }
        match &mut stump.right_leaf_value {
            LeafValue::Scalar(_) => {
                stump.right_leaf_value = LeafValue::Scalar(dr);
            }
            LeafValue::Linear(lin) => {
                lin.intercept = dr;
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_joint_quantile_leaves(
    stumps: &mut [TrainedStump],
    binned_matrix: &BinnedMatrix,
    predictions_per_output: &[Vec<f32>],
    targets_per_output: &[&[f32]],
    per_output_objective: &[JointObjective],
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    let n_outputs = targets_per_output.len();
    if per_output_objective.len() != n_outputs {
        return Err(EngineError::ContractViolation(format!(
            "per_output_objective length {} != n_outputs {n_outputs}",
            per_output_objective.len()
        )));
    }

    let mut has_quantile = false;
    for &obj in per_output_objective {
        if matches!(obj, JointObjective::Quantile { .. }) {
            has_quantile = true;
            break;
        }
    }
    if !has_quantile {
        return Ok(());
    }

    let mut stumps_by_local = HashMap::new();
    for stump in stumps.iter() {
        let local_node_id = stump.split.node_id;
        stumps_by_local.insert(local_node_id, stump);
    }

    let mut current_absolute_outputs_per_output = vec![HashMap::new(); n_outputs];
    for (k, absolute_outputs) in current_absolute_outputs_per_output.iter_mut().enumerate() {
        absolute_outputs.insert(0_u32, 0.0_f32);
        populate_child_absolute_outputs_joint(0, k, &stumps_by_local, absolute_outputs)?;
    }

    let mut leaf_rows: HashMap<u32, Vec<u32>> = HashMap::new();
    for row_index in 0..binned_matrix.row_count {
        let terminal_local_node_id =
            terminal_local_node_id_for_row(row_index, binned_matrix, &stumps_by_local)?;
        leaf_rows
            .entry(terminal_local_node_id)
            .or_default()
            .push(row_index as u32);
    }

    let mut refined_absolute_outputs_per_output = vec![HashMap::new(); n_outputs];
    for k in 0..n_outputs {
        refined_absolute_outputs_per_output[k].insert(0_u32, 0.0_f32);
        if let JointObjective::Quantile { alpha } = per_output_objective[k] {
            let mut leaf_residuals: HashMap<u32, LeafResiduals> = HashMap::new();
            for (&leaf_id, rows) in &leaf_rows {
                let mut res_vec = Vec::with_capacity(rows.len());
                for &row_index in rows {
                    let r = row_index as usize;
                    let res = targets_per_output[k][r] - predictions_per_output[k][r];
                    res_vec.push(res);
                }
                leaf_residuals.insert(
                    leaf_id,
                    LeafResiduals {
                        residuals: res_vec,
                        weights: None,
                    },
                );
            }
            fill_refined_child_quantile_absolute_outputs(
                0,
                &stumps_by_local,
                &leaf_residuals,
                alpha,
                &current_absolute_outputs_per_output[k],
                f32::INFINITY,
                &mut refined_absolute_outputs_per_output[k],
            )?;
        }
    }

    for stump in stumps.iter_mut() {
        let local_node_id = stump.split.node_id;
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;

        let Some((left_k, right_k)) = stump.multi_output_leaf_values.as_mut() else {
            continue;
        };

        for k in 0..n_outputs {
            if matches!(per_output_objective[k], JointObjective::Quantile { .. }) {
                let parent_absolute = refined_absolute_outputs_per_output[k]
                    .get(&local_node_id)
                    .copied()
                    .unwrap_or(0.0);
                let left_absolute = refined_absolute_outputs_per_output[k]
                    .get(&left_local_node_id)
                    .copied()
                    .unwrap_or(parent_absolute + left_k[k]);
                let right_absolute = refined_absolute_outputs_per_output[k]
                    .get(&right_local_node_id)
                    .copied()
                    .unwrap_or(parent_absolute + right_k[k]);

                left_k[k] = left_absolute - parent_absolute;
                right_k[k] = right_absolute - parent_absolute;
            }
        }
    }

    Ok(())
}

fn populate_child_absolute_outputs_joint(
    local_node_id: u32,
    output_index: usize,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<()> {
    let Some(stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(());
    };
    let parent_absolute = absolute_outputs.get(&local_node_id).copied().unwrap_or(0.0);
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;

    let (left_val, right_val) = if let Some((left_k, right_k)) = &stump.multi_output_leaf_values {
        (left_k[output_index], right_k[output_index])
    } else {
        (
            stump.left_leaf_value.as_scalar(),
            stump.right_leaf_value.as_scalar(),
        )
    };

    absolute_outputs.insert(left_local_node_id, parent_absolute + left_val);
    absolute_outputs.insert(right_local_node_id, parent_absolute + right_val);
    populate_child_absolute_outputs_joint(
        left_local_node_id,
        output_index,
        stumps_by_local,
        absolute_outputs,
    )?;
    populate_child_absolute_outputs_joint(
        right_local_node_id,
        output_index,
        stumps_by_local,
        absolute_outputs,
    )?;
    Ok(())
}

/// Extra Newton steps on each leaf of a just-built binary log-loss tree.
///
/// The tree builder sets each leaf to one Newton step from the current
/// predictions, `-G / (H + l2)`. For log loss that step falls short of the
/// leaf's L2-regularized optimum, most of all in nearly pure leaves, where
/// the Hessian shrinks as the step grows. This repeats the step
/// `extra_iterations` more times on all of the leaf's rows (CatBoost's
/// `leaf_estimation_iterations`), then scales the result by `learning_rate`.
/// Leaves must be scalar with no scale but the learning rate.
///
/// `predictions` are the scores before this tree. On return the stumps hold
/// the refined leaves and `candidate_predictions`, which held `predictions`
/// plus the builder's tree, holds `predictions` plus the refined tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_binary_newton_leaf_values(
    stumps: &mut [TrainedStump],
    binned_matrix: &BinnedMatrix,
    predictions: &[f32],
    candidate_predictions: &mut [f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    l2_lambda: f32,
    learning_rate: f32,
    extra_iterations: usize,
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if stumps.is_empty() || extra_iterations == 0 || learning_rate <= 0.0 {
        return Ok(());
    }
    let row_count = binned_matrix.row_count;
    if predictions.len() != row_count
        || candidate_predictions.len() != row_count
        || targets.len() != row_count
        || sample_weights.is_some_and(|w| w.len() != row_count)
    {
        return Err(EngineError::ContractViolation(
            "binary leaf refinement inputs must match the binned row count".to_string(),
        ));
    }
    let mut stumps_by_local = HashMap::new();
    for stump in stumps.iter() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        stumps_by_local.insert(local_node_id, stump);
    }
    // Node outputs as the builder left them (path sums of scaled deltas).
    let mut absolute_outputs = HashMap::new();
    absolute_outputs.insert(0_u32, 0.0_f32);
    populate_child_absolute_outputs(0, &stumps_by_local, &mut absolute_outputs)?;

    // Dense leaf index per row. Level-wise trees have small local ids, so
    // route through flat tables; very deep leaf-wise trees use the maps.
    let mut leaf_ids: Vec<u32> = absolute_outputs
        .keys()
        .copied()
        .filter(|id| !stumps_by_local.contains_key(id))
        .collect();
    leaf_ids.sort_unstable();
    let leaf_slot: HashMap<u32, usize> = leaf_ids
        .iter()
        .enumerate()
        .map(|(slot, &id)| (id, slot))
        .collect();
    let missing_bin = binned_matrix.missing_bin();
    const FLAT_ROUTING_MAX_ID: u32 = 1 << 14;
    let max_id = leaf_ids.last().copied().unwrap_or(0);
    let flat = (max_id < FLAT_ROUTING_MAX_ID).then(|| {
        let mut flat_stumps: Vec<Option<&TrainedStump>> = vec![None; max_id as usize + 1];
        for (&id, &stump) in &stumps_by_local {
            flat_stumps[id as usize] = Some(stump);
        }
        let mut flat_slot = vec![0_u32; max_id as usize + 1];
        for (slot, &id) in leaf_ids.iter().enumerate() {
            flat_slot[id as usize] = slot as u32;
        }
        (flat_stumps, flat_slot)
    });
    let route = |row: usize| -> u32 {
        match &flat {
            Some((flat_stumps, flat_slot)) => {
                let leaf = crate::round::terminal_local_node_for_row(
                    row,
                    binned_matrix,
                    |id| flat_stumps.get(id as usize).copied().flatten(),
                    missing_bin,
                );
                flat_slot[leaf as usize]
            }
            None => {
                let leaf = crate::round::terminal_local_node_for_row(
                    row,
                    binned_matrix,
                    |id| stumps_by_local.get(&id).copied(),
                    missing_bin,
                );
                leaf_slot[&leaf] as u32
            }
        }
    };

    let leaf_count = leaf_ids.len();
    let mut values: Vec<f32> = leaf_ids
        .iter()
        .map(|id| absolute_outputs[id] / learning_rate)
        .collect();
    // Per-leaf (grad, hess) sums of one chunk of rows at the current values.
    let accumulate = |acc: &mut [f64], row: usize, slot: usize, values: &[f32]| {
        let weight = sample_weights.map_or(1.0, |w| w[row]);
        if weight > 0.0 {
            let p = crate::objectives::sigmoid(predictions[row] + values[slot]);
            acc[2 * slot] += (weight * (p - targets[row])) as f64;
            acc[2 * slot + 1] += (weight * (p * (1.0 - p)).max(1e-7)) as f64;
        }
    };
    let add = |mut a: Vec<f64>, b: Vec<f64>| {
        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        a
    };
    const CHUNK: usize = 4096;
    // The first step also routes each row to its leaf.
    let mut row_leaf = vec![0_u32; row_count];
    let mut sums = row_leaf
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut acc = vec![0.0_f64; 2 * leaf_count];
            for (offset, slot_out) in chunk.iter_mut().enumerate() {
                let row = chunk_index * CHUNK + offset;
                let slot = route(row);
                *slot_out = slot;
                accumulate(&mut acc, row, slot as usize, &values);
            }
            acc
        })
        .reduce(|| vec![0.0_f64; 2 * leaf_count], add);
    let l2 = l2_lambda as f64;
    for iteration in 0..extra_iterations {
        if iteration > 0 {
            let current = &values;
            sums = row_leaf
                .par_chunks(CHUNK)
                .enumerate()
                .map(|(chunk_index, chunk)| {
                    let mut acc = vec![0.0_f64; 2 * leaf_count];
                    for (offset, &slot) in chunk.iter().enumerate() {
                        accumulate(
                            &mut acc,
                            chunk_index * CHUNK + offset,
                            slot as usize,
                            current,
                        );
                    }
                    acc
                })
                .reduce(|| vec![0.0_f64; 2 * leaf_count], add);
        }
        let mut largest_step = 0.0_f64;
        for (slot, value) in values.iter_mut().enumerate() {
            let (grad, hess) = (sums[2 * slot], sums[2 * slot + 1]);
            if hess + l2 <= 0.0 {
                continue;
            }
            let step = -(grad + l2 * *value as f64) / (hess + l2);
            if step.is_finite() {
                *value =
                    ((*value as f64 + step) as f32).clamp(-max_abs_leaf_value, max_abs_leaf_value);
                largest_step = largest_step.max(step.abs());
            }
        }
        if largest_step < 1e-6 {
            break;
        }
    }

    // Move each row by its leaf's change, then rewrite the stumps' deltas so
    // path sums equal the refined leaves. Internal nodes keep their old
    // outputs; path sums telescope, so only leaf outputs reach predictions.
    let leaf_shift: Vec<f32> = leaf_ids
        .iter()
        .zip(&values)
        .map(|(id, value)| value * learning_rate - absolute_outputs[id])
        .collect();
    candidate_predictions
        .par_iter_mut()
        .zip(row_leaf.par_iter())
        .for_each(|(prediction, &slot)| *prediction += leaf_shift[slot as usize]);
    for stump in stumps.iter_mut() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let parent = absolute_outputs.get(&local_node_id).copied().unwrap_or(0.0);
        for (child, leaf) in [
            (
                left_child_node_id(local_node_id)?,
                &mut stump.left_leaf_value,
            ),
            (
                right_child_node_id(local_node_id)?,
                &mut stump.right_leaf_value,
            ),
        ] {
            if let Some(&slot) = leaf_slot.get(&child) {
                *leaf = LeafValue::Scalar(values[slot] * learning_rate - parent);
            }
        }
    }
    Ok(())
}
