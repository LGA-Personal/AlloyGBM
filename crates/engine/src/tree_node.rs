//! Tree-node encoding helpers.
//!
//! Pure helpers that convert between (tree_index, local_node_id) and the
//! flat `u32` representation stored on `SplitCandidate::node_id`. Used by the
//! trainer + predictor paths.

use crate::TrainedStump;
use crate::error::{EngineError, EngineResult};
use alloygbm_core::SplitCandidate;
use std::collections::HashMap;

pub(crate) const TREE_NODE_STRIDE: u32 = 1 << 20;

pub(crate) fn encode_tree_node_id(tree_index: usize, local_node_id: u32) -> EngineResult<u32> {
    // Guard against trees that would need more heap-style node slots than the
    // predictor is willing to allocate at load time. Enforcing the same shared
    // limit here keeps the contract symmetric: a model that trains always
    // loads. Reaching this requires an unusually deep tree (~depth 16+); the
    // fit fails fast with a clear error instead of producing an unloadable
    // artifact.
    if local_node_id as usize >= alloygbm_core::MAX_TREE_NODE_SLOTS {
        return Err(EngineError::ContractViolation(format!(
            "local node_id {local_node_id} exceeds supported tree-node slot limit {}",
            alloygbm_core::MAX_TREE_NODE_SLOTS
        )));
    }
    let tree_index_u32 = u32::try_from(tree_index).map_err(|_| {
        EngineError::ContractViolation(format!("tree index {tree_index} exceeds u32::MAX"))
    })?;
    tree_index_u32
        .checked_mul(TREE_NODE_STRIDE)
        .and_then(|base| base.checked_add(local_node_id))
        .ok_or_else(|| {
            EngineError::ContractViolation("encoded tree node id overflowed u32 range".to_string())
        })
}

pub(crate) fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

pub(crate) fn left_child_node_id(local_node_id: u32) -> EngineResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "left child id overflow for local node {local_node_id}"
            ))
        })
}

pub(crate) fn right_child_node_id(local_node_id: u32) -> EngineResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "right child id overflow for local node {local_node_id}"
            ))
        })
}

pub(crate) fn retained_stump_count_for_rounds(
    stumps_per_completed_round: &[usize],
    round_count: usize,
) -> usize {
    stumps_per_completed_round
        .iter()
        .take(round_count)
        .sum::<usize>()
}

/// Determine if a feature value goes left at a split, handling continuous, categorical, and NaN.
#[inline]
pub(crate) fn split_went_left(split: &SplitCandidate, feature_value: f32) -> bool {
    if feature_value.is_nan() {
        split.default_left
    } else if split.is_categorical {
        split
            .categorical_bitset
            .as_ref()
            .map_or(split.default_left, |bs| {
                let cat_id = feature_value as u16;
                let byte_idx = (cat_id / 8) as usize;
                let bit_idx = (cat_id % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            })
    } else {
        feature_value <= split.threshold_bin as f32
    }
}

pub(crate) fn row_satisfies_stump_path_features(
    features: &[f32],
    stump: &TrainedStump,
    stumps_by_node: &HashMap<u32, &TrainedStump>,
) -> EngineResult<bool> {
    let (tree_id, mut local_node_id) = decode_tree_node_id(stump.split.node_id);
    while local_node_id > 0 {
        let parent_local = (local_node_id - 1) / 2;
        let parent_node_id = encode_tree_node_id(tree_id as usize, parent_local)?;
        let Some(parent_stump) = stumps_by_node.get(&parent_node_id) else {
            return Ok(false);
        };
        let feature_index = parent_stump.split.feature_index as usize;
        if feature_index >= features.len() {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature length {}",
                parent_stump.split.feature_index,
                features.len()
            )));
        }
        let feature_value = features[feature_index];
        let went_left = split_went_left(&parent_stump.split, feature_value);
        let expected_left = local_node_id == parent_local * 2 + 1;
        if went_left != expected_left {
            return Ok(false);
        }
        local_node_id = parent_local;
    }
    Ok(true)
}
