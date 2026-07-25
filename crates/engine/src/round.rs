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
    apply_weighted_round_to_predictions_internal(
        predictions,
        None,
        binned_matrix,
        stumps,
        raw_features,
        factor,
    )
}

pub(crate) fn apply_weighted_round_to_predictions_and_accumulator(
    predictions: &mut [f32],
    accumulator: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
    prediction_factor: f32,
    accumulator_factor: f32,
) -> EngineResult<()> {
    if predictions.len() != accumulator.len() {
        return Err(EngineError::ContractViolation(format!(
            "prediction length {} does not match accumulator length {}",
            predictions.len(),
            accumulator.len(),
        )));
    }
    apply_weighted_round_to_predictions_internal(
        predictions,
        Some((accumulator, accumulator_factor)),
        binned_matrix,
        stumps,
        raw_features,
        prediction_factor,
    )
}

#[inline]
fn binned_split_went_left(stump: &TrainedStump, bin: u16) -> bool {
    if bin == u16::from(MISSING_BIN_U8) {
        stump.split.default_left
    } else if stump.split.is_categorical {
        stump
            .split
            .categorical_bitset
            .as_ref()
            .map_or(stump.split.default_left, |bitset| {
                let byte_index = (bin / 8) as usize;
                let bit_index = (bin % 8) as usize;
                byte_index < bitset.len() && (bitset[byte_index] & (1 << bit_index)) != 0
            })
    } else {
        bin <= stump.split.threshold_bin
    }
}

fn apply_weighted_round_to_predictions_internal(
    predictions: &mut [f32],
    mut accumulator: Option<(&mut [f32], f32)>,
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
    prediction_factor: f32,
) -> EngineResult<()> {
    let accumulator_factor_is_zero = accumulator
        .as_ref()
        .is_none_or(|(_, factor)| *factor == 0.0);
    if stumps.is_empty() || (prediction_factor == 0.0 && accumulator_factor_is_zero) {
        return Ok(());
    }
    let mut stump_by_local: HashMap<u32, &TrainedStump> = HashMap::with_capacity(stumps.len());
    for stump in stumps {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        stump_by_local.insert(local_id, stump);
    }
    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break;
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.col_bin(feature_index * binned_matrix.row_count + row_index);
            let went_left = binned_split_went_left(stump, bin);
            let leaf_contribution = if went_left {
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
            *prediction += prediction_factor * leaf_contribution;
            if let Some((buffer, factor)) = accumulator.as_mut() {
                buffer[row_index] += *factor * leaf_contribution;
            }
            local_id = if went_left {
                local_id * 2 + 1
            } else {
                local_id * 2 + 2
            };
        }
    }
    Ok(())
}

pub(crate) fn apply_scaled_prediction_buffer(
    predictions: &mut [f32],
    contribution: &[f32],
    factor: f32,
) -> EngineResult<()> {
    if predictions.len() != contribution.len() {
        return Err(EngineError::ContractViolation(format!(
            "prediction length {} does not match contribution length {}",
            predictions.len(),
            contribution.len(),
        )));
    }
    for (prediction, value) in predictions.iter_mut().zip(contribution) {
        *prediction += factor * *value;
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
    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        // Walk the tree starting from the root (local_node_id = 0)
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break; // reached a leaf — no stump at this node
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.col_bin(feature_index * binned_matrix.row_count + row_index);
            // v0.10.0 review fix (Comment 1): multiply leaf contribution by
            // `stump.tree_weight` so warm-start prior predictions reflect
            // saved DART weights. For non-DART stumps tree_weight == 1.0,
            // so this is a no-op and preserves byte-identical numerics for
            // every existing caller (Standard/GOSS/Morph/DRO/linear).
            let tree_weight = stump.tree_weight;
            let went_left = binned_split_went_left(stump, bin);
            if went_left {
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{LeafValue, LinearLeaf, NodeStats, SplitCandidate};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    fn scalar_split(
        threshold_bin: u16,
        default_left: bool,
        is_categorical: bool,
        categorical_bitset: Option<Vec<u8>>,
    ) -> SplitCandidate {
        SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin,
            gain: 1.0,
            default_left,
            is_categorical,
            categorical_bitset,
            left_stats: NodeStats {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                row_count: 2,
            },
            right_stats: NodeStats {
                grad_sum: 1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                row_count: 2,
            },
        }
    }

    fn assert_aggregate_matches_reference(
        initial: Vec<f32>,
        binned: BinnedMatrix,
        stumps: Vec<TrainedStump>,
        raw_features: Option<(&[f32], usize)>,
    ) -> EngineResult<()> {
        let old_weight = 0.25;
        let mut predictions = initial.clone();
        let mut aggregate = vec![0.0; initial.len()];
        apply_weighted_round_to_predictions_and_accumulator(
            &mut predictions,
            &mut aggregate,
            &binned,
            &stumps,
            raw_features,
            -old_weight,
            old_weight,
        )?;

        let mut reference = initial.clone();
        apply_weighted_round_to_predictions(
            &mut reference,
            &binned,
            &stumps,
            raw_features,
            -old_weight,
        )?;
        assert_close(&predictions, &reference);

        let mut expected_aggregate = vec![0.0; initial.len()];
        apply_weighted_round_to_predictions(
            &mut expected_aggregate,
            &binned,
            &stumps,
            raw_features,
            old_weight,
        )?;
        assert_close(&aggregate, &expected_aggregate);
        Ok(())
    }

    #[test]
    fn aggregate_scalar_leaf_matches_reference_walk() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(1.5),
            LeafValue::Scalar(-0.5),
        )];
        assert_aggregate_matches_reference(vec![10.0, 20.0, 30.0, 40.0], binned, stumps, None)
    }

    #[test]
    fn aggregate_walk_populates_accumulator_when_prediction_factor_is_zero() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(1.5),
            LeafValue::Scalar(-0.5),
        )];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut predictions = initial.clone();
        let mut aggregate = vec![0.0; initial.len()];
        apply_weighted_round_to_predictions_and_accumulator(
            &mut predictions,
            &mut aggregate,
            &binned,
            &stumps,
            None,
            0.0,
            0.25,
        )?;
        assert_close(&predictions, &initial);

        let mut expected_aggregate = vec![0.0; initial.len()];
        apply_weighted_round_to_predictions(&mut expected_aggregate, &binned, &stumps, None, 0.25)?;
        assert_close(&aggregate, &expected_aggregate);
        Ok(())
    }

    #[test]
    fn aggregate_learned_missing_routing_matches_reference_walk() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, MISSING_BIN_U8, 2, MISSING_BIN_U8])
            .expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, false, false, None),
            LeafValue::Scalar(2.0),
            LeafValue::Scalar(-3.0),
        )];
        assert_aggregate_matches_reference(vec![0.0; 4], binned, stumps, None)
    }

    #[test]
    fn aggregate_native_categorical_routing_matches_reference_walk() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(0, false, true, Some(vec![0b0000_0011])),
            LeafValue::Scalar(0.75),
            LeafValue::Scalar(-1.25),
        )];
        assert_aggregate_matches_reference(vec![1.0; 4], binned, stumps, None)
    }

    #[test]
    fn replay_walk_respects_learned_missing_direction() -> EngineResult<()> {
        let binned =
            BinnedMatrix::new(3, 1, 2, vec![0, MISSING_BIN_U8, 2]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(2.0),
            LeafValue::Scalar(-3.0),
        )];
        let mut predictions = vec![0.0; 3];

        apply_tree_to_binned_predictions(&mut predictions, &binned, &stumps, None)?;

        assert_close(&predictions, &[2.0, 2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn replay_walk_respects_native_categorical_bitset() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(0, false, true, Some(vec![0b0000_0101])),
            LeafValue::Scalar(0.75),
            LeafValue::Scalar(-1.25),
        )];
        let mut predictions = vec![0.0; 4];

        apply_tree_to_binned_predictions(&mut predictions, &binned, &stumps, None)?;

        assert_close(&predictions, &[0.75, -1.25, 0.75, -1.25]);
        Ok(())
    }

    #[test]
    fn aggregate_linear_leaf_evaluation_matches_reference_walk() -> EngineResult<()> {
        let binned =
            BinnedMatrix::new(4, 2, 3, vec![0, 0, 1, 0, 2, 0, 3, 0]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Linear(LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0])),
            LeafValue::Linear(LinearLeaf::identity_scaled(-2.0, vec![0.25], vec![1])),
        )];
        let raw_features = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        assert_aggregate_matches_reference(vec![0.0; 4], binned, stumps, Some((&raw_features, 2)))
    }

    #[test]
    fn aggregate_walk_rejects_prediction_accumulator_length_mismatch() {
        let binned = BinnedMatrix::new(2, 1, 1, vec![0, 1]).expect("valid binned matrix");
        let result = apply_weighted_round_to_predictions_and_accumulator(
            &mut [0.0, 0.0],
            &mut [0.0],
            &binned,
            &[],
            None,
            1.0,
            1.0,
        );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn scaled_prediction_buffer_rejects_prediction_contribution_length_mismatch() {
        let result = apply_scaled_prediction_buffer(&mut [0.0, 0.0], &[1.0], 1.0);
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }
}
