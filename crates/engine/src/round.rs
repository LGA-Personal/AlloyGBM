//! Round application helpers — apply built tree stumps to running
//! prediction buffers.
//!
//! Used by the training loop after each round to fold the just-built
//! tree's leaf contributions into `predictions`, and by DART to
//! subtract / re-add dropped trees with arbitrary multiplicative
//! factors.
//!
//! Routing semantics follow the predictor: missing bin routes via
//! `default_left`, native categorical splits consult the bitset,
//! otherwise standard `bin <= threshold_bin` comparison. Leaves
//! evaluate as scalar intercepts unless PL leaves are active and
//! raw feature rows are provided.

use std::collections::HashMap;

use alloygbm_core::{BinnedMatrix, MISSING_BIN_U8, PartitionResult};

use crate::error::{EngineError, EngineResult};
use crate::tree_node::decode_tree_node_id;
use crate::types::TrainedStump;

pub(crate) fn apply_partition_leaf_updates(
    predictions: &mut [f32],
    partition: &PartitionResult,
    left_leaf_value: f32,
    right_leaf_value: f32,
) -> EngineResult<()> {
    let prediction_len = predictions.len();
    for &row_index in &partition.left_row_indices {
        let row_index = row_index as usize;
        if row_index >= prediction_len {
            return Err(EngineError::ContractViolation(format!(
                "left partition row index {row_index} is out of bounds for predictions length {prediction_len}"
            )));
        }
        predictions[row_index] += left_leaf_value;
    }
    for &row_index in &partition.right_row_indices {
        let row_index = row_index as usize;
        if row_index >= prediction_len {
            return Err(EngineError::ContractViolation(format!(
                "right partition row index {row_index} is out of bounds for predictions length {prediction_len}"
            )));
        }
        predictions[row_index] += right_leaf_value;
    }
    Ok(())
}

/// DART helper: apply one tree's stumps to `predictions` with a
/// multiplicative `factor`. `factor = 1.0` reproduces a unit-weight
/// tree walk; `factor = -w` is used to subtract a dropped tree's
/// previous contribution; `factor = new_w` is used to re-add a
/// rescaled tree post-normalization.
///
/// Routing uses the binned-matrix view but with the same split
/// semantics as the predictor: missing bin (`MISSING_BIN_U8`) routes
/// through `default_left`; native categorical splits consult the
/// stump's `categorical_bitset`; otherwise the standard
/// `bin <= threshold_bin` comparison applies.  Using only
/// `bin <= threshold_bin` (the legacy `apply_round_stumps_tree_walk`
/// shortcut) would silently disagree with the predictor on rows with
/// learned-missing-direction or native categorical features, which
/// matters for DART because the dropout subtract / re-add must
/// reproduce the predictor's per-tree contribution exactly.
///
/// `raw_features = Some((raw, fc))` is used only for PL-leaf
/// evaluation (`LeafValue::Linear`).  Constant-leaf models can pass
/// `None` (or an empty raw slice) and the leaf will be evaluated as
/// the scalar intercept.
///
/// All stumps in `stumps` are assumed to belong to the same tree (i.e.,
/// share the same encoded `tree_id` in their `node_id`). The caller is
/// responsible for slicing `stumps` correctly.
pub(crate) fn apply_weighted_round_to_predictions(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
    factor: f32,
) -> EngineResult<()> {
    if stumps.is_empty() || factor == 0.0 {
        return Ok(());
    }
    let mut stump_by_local: HashMap<u32, &TrainedStump> = HashMap::with_capacity(stumps.len());
    for stump in stumps {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        stump_by_local.insert(local_id, stump);
    }
    let feature_count = binned_matrix.feature_count;
    let missing_bin = u16::from(MISSING_BIN_U8);

    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        let row_base = row_index * feature_count;
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break;
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.row_bin(row_base + feature_index);
            let went_left = if bin == missing_bin {
                // Missing-value routing — predictor's `is_nan` short-circuit
                // produces the same `default_left` outcome.
                stump.split.default_left
            } else if stump.split.is_categorical {
                // Native categorical split: consult the bitset (same
                // routing as `predictor_went_left`).
                stump
                    .split
                    .categorical_bitset
                    .as_ref()
                    .map_or(stump.split.default_left, |bs| {
                        let cat_id = bin;
                        let byte_idx = (cat_id / 8) as usize;
                        let bit_idx = (cat_id % 8) as usize;
                        byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
                    })
            } else {
                bin <= stump.split.threshold_bin
            };
            let leaf = if went_left {
                if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.left_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.left_leaf_value.as_scalar()
                }
            } else if let Some((raw, fc)) = raw_features
                && !raw.is_empty()
            {
                let row_offset = row_index * fc;
                stump.right_leaf_value.eval_row(&raw[row_offset..])
            } else {
                stump.right_leaf_value.as_scalar()
            };
            *prediction += factor * leaf;
            local_id = if went_left {
                local_id * 2 + 1
            } else {
                local_id * 2 + 2
            };
        }
    }
    Ok(())
}

pub(crate) fn apply_round_stumps_tree_walk(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    // Build a lookup from local_node_id to stump for tree traversal
    let mut stump_by_local: HashMap<u32, &TrainedStump> = HashMap::with_capacity(stumps.len());
    for stump in stumps {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        stump_by_local.insert(local_id, stump);
    }
    let feature_count = binned_matrix.feature_count;

    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        let row_base = row_index * feature_count;
        // Walk the tree starting from the root (local_node_id = 0)
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break; // reached a leaf — no stump at this node
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.row_bin(row_base + feature_index);
            // v0.10.0 review fix (Comment 1): multiply leaf contribution by
            // `stump.tree_weight` so warm-start prior predictions reflect
            // saved DART weights. For non-DART stumps tree_weight == 1.0,
            // so this is a no-op and preserves byte-identical numerics for
            // every existing caller (Standard/GOSS/Morph/DRO/linear).
            let tree_weight = stump.tree_weight;
            if bin <= stump.split.threshold_bin {
                let leaf_value = if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.left_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.left_leaf_value.as_scalar()
                };
                *prediction += tree_weight * leaf_value;
                local_id = local_id * 2 + 1; // left child
            } else {
                let leaf_value = if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.right_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.right_leaf_value.as_scalar()
                };
                *prediction += tree_weight * leaf_value;
                local_id = local_id * 2 + 2; // right child
            }
        }
    }
    Ok(())
}

pub(crate) fn apply_tree_to_binned_predictions(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    // Split stumps into per-round groups by detecting tree_id changes
    let mut round_start = 0;
    let mut current_tree_id = decode_tree_node_id(stumps[0].split.node_id).0;
    for i in 1..stumps.len() {
        let tree_id = decode_tree_node_id(stumps[i].split.node_id).0;
        if tree_id != current_tree_id {
            apply_round_stumps_tree_walk(
                predictions,
                binned_matrix,
                &stumps[round_start..i],
                raw_features,
            )?;
            round_start = i;
            current_tree_id = tree_id;
        }
    }
    apply_round_stumps_tree_walk(
        predictions,
        binned_matrix,
        &stumps[round_start..],
        raw_features,
    )?;
    Ok(())
}
