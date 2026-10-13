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

use std::collections::{HashMap, HashSet};

use alloygbm_core::{BinnedMatrix, PartitionResult};
use rayon::prelude::*;

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
/// semantics as the predictor: the matrix's missing bin routes
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

/// Row count below which complement replay stays on the serial path.
///
/// Under this many rows the rayon fork/join costs more than the replay saves.
/// Both paths produce bit-identical predictions, so this threshold -- and the
/// thread count it is compared against -- only affect speed, never results.
const REPLAY_PARALLEL_MIN_ROWS: usize = 32_768;

pub(crate) fn apply_weighted_round_to_rows(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
    row_indices: &[u32],
    factor: f32,
) -> EngineResult<()> {
    validate_restricted_replay_inputs(predictions, binned_matrix, raw_features, row_indices)?;
    let stump_by_local = build_stump_lookup(stumps, binned_matrix.feature_count)?;
    if stumps.is_empty() || factor == 0.0 {
        return Ok(());
    }
    let missing_bin = binned_matrix.missing_bin();

    // Complement replay is a pure scatter: every row writes only its own
    // prediction slot, and the stumps for a given row are replayed in the
    // same order however the work is divided. Splitting it across threads is
    // therefore bit-identical to the serial loop -- there is no cross-row
    // reduction whose associativity could shift, so unlike histogram
    // accumulation this needs no fixed chunk width to stay deterministic.
    //
    // The split works by chunking the *output* slice and binary-searching each
    // chunk's slice of `row_indices`, which requires the indices to be sorted.
    // Every producer in `sampling.rs` sorts them, but the check is O(n) against
    // an O(n * depth) replay, so verify rather than assume and fall back to the
    // serial loop if a future caller passes an unsorted list.
    let parallel = row_indices.len() >= REPLAY_PARALLEL_MIN_ROWS
        && rayon::current_num_threads() > 1
        && row_indices.is_sorted();

    if parallel {
        let threads = rayon::current_num_threads();
        let chunk_rows = predictions.len().div_ceil(threads * 4).max(1);
        predictions.par_chunks_mut(chunk_rows).enumerate().for_each(
            |(chunk_index, chunk_predictions)| {
                let chunk_start = chunk_index * chunk_rows;
                let chunk_end = chunk_start + chunk_predictions.len();
                let lo = row_indices.partition_point(|&row| (row as usize) < chunk_start);
                let hi = row_indices.partition_point(|&row| (row as usize) < chunk_end);
                for &row_index in &row_indices[lo..hi] {
                    let row_index = row_index as usize;
                    let prediction = &mut chunk_predictions[row_index - chunk_start];
                    replay_round_row(
                        row_index,
                        binned_matrix,
                        &stump_by_local,
                        missing_bin,
                        raw_features,
                        |_, leaf_value| *prediction += factor * leaf_value,
                    );
                }
            },
        );
        return Ok(());
    }

    for &row_index in row_indices {
        let row_index = row_index as usize;
        let prediction = &mut predictions[row_index];
        replay_round_row(
            row_index,
            binned_matrix,
            &stump_by_local,
            missing_bin,
            raw_features,
            |_, leaf_value| *prediction += factor * leaf_value,
        );
    }
    Ok(())
}

#[inline]
pub(crate) fn binned_split_went_left(stump: &TrainedStump, bin: u16, missing_bin: u16) -> bool {
    if bin == missing_bin {
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

/// Local node id of the terminal leaf `row_index` lands in, routed like the
/// predictor (learned missing direction, native categorical bitsets).
/// `stump_at` returns the split stored at a local node id, if any.
#[inline]
pub(crate) fn terminal_local_node_for_row<'a>(
    row_index: usize,
    binned_matrix: &BinnedMatrix,
    stump_at: impl Fn(u32) -> Option<&'a TrainedStump>,
    missing_bin: u16,
) -> u32 {
    let mut local_id = 0_u32;
    while let Some(stump) = stump_at(local_id) {
        debug_assert!(
            local_id < u32::MAX / 2,
            "local node id {local_id} overflows"
        );
        let feature_index = stump.split.feature_index as usize;
        let bin = binned_matrix.col_bin(feature_index * binned_matrix.row_count + row_index);
        local_id = if binned_split_went_left(stump, bin, missing_bin) {
            local_id * 2 + 1
        } else {
            local_id * 2 + 2
        };
    }
    local_id
}

/// One round's stumps laid out for replay: each node holds its stump and the
/// array indices of its two children, so walking a row down the tree chases
/// array indices instead of hashing a local node id at every level.
struct ReplayTree<'a> {
    nodes: Vec<ReplayNode<'a>>,
    root: u32,
}

struct ReplayNode<'a> {
    stump: &'a TrainedStump,
    /// `[left, right]` child indices into `nodes`, `NO_REPLAY_NODE` for a leaf.
    children: [u32; 2],
}

const NO_REPLAY_NODE: u32 = u32::MAX;

fn build_stump_lookup(
    stumps: &[TrainedStump],
    feature_count: usize,
) -> EngineResult<ReplayTree<'_>> {
    validate_stump_contracts(stumps, feature_count)?;
    let mut index_by_local: HashMap<u32, u32> = HashMap::with_capacity(stumps.len());
    for (index, stump) in stumps.iter().enumerate() {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        if index_by_local.insert(local_id, index as u32).is_some() {
            return Err(EngineError::ContractViolation(format!(
                "duplicate local node id {local_id} in replay round"
            )));
        }
    }
    let child = |local_id: u32, offset: u32| {
        local_id
            .checked_mul(2)
            .and_then(|id| id.checked_add(offset))
            .and_then(|id| index_by_local.get(&id).copied())
            .unwrap_or(NO_REPLAY_NODE)
    };
    let nodes = stumps
        .iter()
        .map(|stump| {
            let (_, local_id) = decode_tree_node_id(stump.split.node_id);
            ReplayNode {
                stump,
                children: [child(local_id, 1), child(local_id, 2)],
            }
        })
        .collect();
    let root = index_by_local.get(&0).copied().unwrap_or(NO_REPLAY_NODE);
    Ok(ReplayTree { nodes, root })
}

fn validate_stump_contracts(stumps: &[TrainedStump], feature_count: usize) -> EngineResult<()> {
    let mut seen_nodes = HashSet::with_capacity(stumps.len());
    for stump in stumps {
        let feature_index = stump.split.feature_index as usize;
        if feature_index >= feature_count {
            return Err(EngineError::ContractViolation(format!(
                "stump feature index {feature_index} is out of bounds for binned feature count {feature_count}"
            )));
        }
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        if !seen_nodes.insert((tree_id, local_id)) {
            return Err(EngineError::ContractViolation(format!(
                "duplicate local node id {local_id} for tree {tree_id}"
            )));
        }
    }
    Ok(())
}

fn validate_active_raw_features(
    row_count: usize,
    binned_feature_count: usize,
    raw_features: Option<(&[f32], usize)>,
) -> EngineResult<()> {
    let Some((raw, feature_count)) = raw_features else {
        return Ok(());
    };
    if raw.is_empty() {
        return Ok(());
    }
    if feature_count == 0 {
        return Err(EngineError::ContractViolation(
            "raw feature count must be nonzero for nonempty PL input".to_string(),
        ));
    }
    if feature_count != binned_feature_count {
        return Err(EngineError::ContractViolation(format!(
            "raw feature count {feature_count} does not match binned feature count {binned_feature_count}"
        )));
    }
    let required_len = row_count.checked_mul(feature_count).ok_or_else(|| {
        EngineError::ContractViolation(format!(
            "raw feature storage size overflows for row count {row_count} and feature count {feature_count}"
        ))
    })?;
    if raw.len() < required_len {
        return Err(EngineError::ContractViolation(format!(
            "raw feature storage length {} is shorter than required length {required_len}",
            raw.len(),
        )));
    }
    Ok(())
}

fn validate_restricted_replay_inputs(
    predictions: &[f32],
    binned_matrix: &BinnedMatrix,
    raw_features: Option<(&[f32], usize)>,
    row_indices: &[u32],
) -> EngineResult<()> {
    if predictions.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "prediction length {} does not match binned row count {}",
            predictions.len(),
            binned_matrix.row_count,
        )));
    }
    for &row_index in row_indices {
        if row_index as usize >= binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "row index {row_index} is out of bounds for row count {}",
                binned_matrix.row_count,
            )));
        }
    }
    validate_active_raw_features(
        binned_matrix.row_count,
        binned_matrix.feature_count,
        raw_features,
    )
}

fn replay_round_row(
    row_index: usize,
    binned_matrix: &BinnedMatrix,
    tree: &ReplayTree<'_>,
    missing_bin: u16,
    raw_features: Option<(&[f32], usize)>,
    mut apply_stump: impl FnMut(&TrainedStump, f32),
) {
    let mut node_index = tree.root;
    while node_index != NO_REPLAY_NODE {
        let node = &tree.nodes[node_index as usize];
        let stump = node.stump;
        let feature_index = stump.split.feature_index as usize;
        let bin = binned_matrix.col_bin(feature_index * binned_matrix.row_count + row_index);
        let went_left = binned_split_went_left(stump, bin, missing_bin);
        let leaf_value = if went_left {
            evaluate_leaf_for_row(&stump.left_leaf_value, row_index, raw_features)
        } else {
            evaluate_leaf_for_row(&stump.right_leaf_value, row_index, raw_features)
        };
        apply_stump(stump, leaf_value);
        node_index = node.children[usize::from(!went_left)];
    }
}

#[inline]
fn evaluate_leaf_for_row(
    leaf_value: &alloygbm_core::LeafValue,
    row_index: usize,
    raw_features: Option<(&[f32], usize)>,
) -> f32 {
    if let Some((raw, feature_count)) = raw_features
        && !raw.is_empty()
    {
        let row_offset = row_index * feature_count;
        leaf_value.eval_row(&raw[row_offset..])
    } else {
        leaf_value.as_scalar()
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
    validate_active_raw_features(
        binned_matrix.row_count,
        binned_matrix.feature_count,
        raw_features,
    )?;
    let stump_by_local = build_stump_lookup(stumps, binned_matrix.feature_count)?;
    let missing_bin = binned_matrix.missing_bin();
    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        replay_round_row(
            row_index,
            binned_matrix,
            &stump_by_local,
            missing_bin,
            raw_features,
            |_, leaf_value| {
                *prediction += prediction_factor * leaf_value;
                if let Some((buffer, factor)) = accumulator.as_mut() {
                    buffer[row_index] += *factor * leaf_value;
                }
            },
        );
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
    validate_active_raw_features(
        binned_matrix.row_count,
        binned_matrix.feature_count,
        raw_features,
    )?;
    let stump_by_local = build_stump_lookup(stumps, binned_matrix.feature_count)?;
    let missing_bin = binned_matrix.missing_bin();
    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        replay_round_row(
            row_index,
            binned_matrix,
            &stump_by_local,
            missing_bin,
            raw_features,
            |stump, leaf_value| *prediction += stump.tree_weight * leaf_value,
        );
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
    validate_active_raw_features(
        binned_matrix.row_count,
        binned_matrix.feature_count,
        raw_features,
    )?;
    validate_stump_contracts(stumps, binned_matrix.feature_count)?;
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
    use alloygbm_core::{
        LeafValue, LinearLeaf, MISSING_BIN_U8, MISSING_BIN_U16, NodeStats, SplitCandidate,
    };

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

    fn assert_restricted_rows_match_full_replay(
        binned: BinnedMatrix,
        stumps: Vec<TrainedStump>,
        raw_features: Option<(&[f32], usize)>,
    ) -> EngineResult<()> {
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut expected = initial.clone();
        apply_weighted_round_to_predictions(&mut expected, &binned, &stumps, raw_features, 1.0)?;
        let mut actual = initial.clone();
        apply_weighted_round_to_rows(&mut actual, &binned, &stumps, raw_features, &[1, 3], 1.0)?;
        assert_eq!(actual[0].to_bits(), initial[0].to_bits());
        assert_eq!(actual[2].to_bits(), initial[2].to_bits());
        assert_eq!(actual[1].to_bits(), expected[1].to_bits());
        assert_eq!(actual[3].to_bits(), expected[3].to_bits());
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
    fn restricted_numeric_rows_match_full_replay() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(1.5),
            LeafValue::Scalar(-0.5),
        )];
        assert_restricted_rows_match_full_replay(binned, stumps, None)
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
    fn restricted_learned_missing_rows_match_full_replay() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, MISSING_BIN_U8, 2, MISSING_BIN_U8])
            .expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, false, false, None),
            LeafValue::Scalar(2.0),
            LeafValue::Scalar(-3.0),
        )];
        assert_restricted_rows_match_full_replay(binned, stumps, None)
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
    fn restricted_native_categorical_rows_match_full_replay() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(0, false, true, Some(vec![0b0000_0101])),
            LeafValue::Scalar(0.75),
            LeafValue::Scalar(-1.25),
        )];
        assert_restricted_rows_match_full_replay(binned, stumps, None)
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
    fn restricted_linear_leaf_rows_match_full_replay() -> EngineResult<()> {
        let binned =
            BinnedMatrix::new(4, 2, 3, vec![0, 0, 1, 0, 2, 0, 3, 0]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Linear(LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0])),
            LeafValue::Linear(LinearLeaf::identity_scaled(-2.0, vec![0.25], vec![1])),
        )];
        let raw_features = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        assert_restricted_rows_match_full_replay(binned, stumps, Some((&raw_features, 2)))
    }

    fn assert_wide_bin_missing_direction(
        default_left: bool,
        expected_missing_leaf: f32,
    ) -> EngineResult<()> {
        let binned = BinnedMatrix::new_u16(
            4,
            1,
            65_534,
            MISSING_BIN_U16,
            vec![0, MISSING_BIN_U16, 2, MISSING_BIN_U16],
        )
        .expect("valid wide-bin matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, default_left, false, None),
            LeafValue::Scalar(2.0),
            LeafValue::Scalar(-3.0),
        )];

        let mut full = vec![0.0; 4];
        apply_weighted_round_to_predictions(&mut full, &binned, &stumps, None, 1.0)?;
        assert_close(
            &full,
            &[2.0, expected_missing_leaf, -3.0, expected_missing_leaf],
        );

        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut restricted = initial.clone();
        apply_weighted_round_to_rows(&mut restricted, &binned, &stumps, None, &[1, 3], 1.0)?;
        assert_eq!(restricted[0].to_bits(), initial[0].to_bits());
        assert_eq!(restricted[2].to_bits(), initial[2].to_bits());
        assert_eq!(
            restricted[1].to_bits(),
            (20.0 + expected_missing_leaf).to_bits()
        );
        assert_eq!(
            restricted[3].to_bits(),
            (40.0 + expected_missing_leaf).to_bits()
        );
        Ok(())
    }

    #[test]
    fn wide_bin_missing_default_left_true_matches_predictor_routing() -> EngineResult<()> {
        assert_wide_bin_missing_direction(true, 2.0)
    }

    #[test]
    fn wide_bin_missing_default_left_false_matches_predictor_routing() -> EngineResult<()> {
        assert_wide_bin_missing_direction(false, -3.0)
    }

    #[test]
    fn depth_two_replay_preserves_per_stump_f32_sequence() -> EngineResult<()> {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 0, 2, 2]).expect("valid binned matrix");
        let mut root_split = scalar_split(1, true, false, None);
        root_split.node_id = 0;
        let mut child_split = scalar_split(0, true, false, None);
        child_split.node_id = 1;
        let stumps = vec![
            TrainedStump::new_unweighted(
                root_split,
                LeafValue::Scalar(1.0),
                LeafValue::Scalar(0.0),
            ),
            TrainedStump::new_unweighted(
                child_split,
                LeafValue::Scalar(1.0),
                LeafValue::Scalar(0.0),
            ),
        ];
        let initial = vec![10.0, 16_777_216.0, 30.0, 40.0];

        let mut expected = initial.clone();
        expected[1] += 1.0;
        expected[1] += 1.0;

        let mut full = initial.clone();
        apply_weighted_round_to_predictions(&mut full, &binned, &stumps, None, 1.0)?;
        assert_eq!(full[1].to_bits(), expected[1].to_bits());

        let mut restricted = initial.clone();
        apply_weighted_round_to_rows(&mut restricted, &binned, &stumps, None, &[1], 1.0)?;
        assert_eq!(restricted[1].to_bits(), expected[1].to_bits());
        assert_eq!(restricted[0].to_bits(), initial[0].to_bits());
        assert_eq!(restricted[2].to_bits(), initial[2].to_bits());
        assert_eq!(restricted[3].to_bits(), initial[3].to_bits());
        Ok(())
    }

    #[test]
    fn restricted_rows_reject_invalid_descendant_feature_before_mutation() {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 0, 2, 2]).expect("valid binned matrix");
        let mut root_split = scalar_split(1, true, false, None);
        root_split.node_id = 0;
        let mut child_split = scalar_split(0, true, false, None);
        child_split.node_id = 1;
        child_split.feature_index = 1;
        let stumps = vec![
            TrainedStump::new_unweighted(
                root_split,
                LeafValue::Scalar(1.0),
                LeafValue::Scalar(0.0),
            ),
            TrainedStump::new_unweighted(
                child_split,
                LeafValue::Scalar(2.0),
                LeafValue::Scalar(0.0),
            ),
        ];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut actual = initial.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_weighted_round_to_rows(&mut actual, &binned, &stumps, None, &[1], 1.0)
        }));

        assert!(matches!(result, Ok(Err(EngineError::ContractViolation(_)))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_out_of_bounds_before_mutation() {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(1.5),
            LeafValue::Scalar(-0.5),
        )];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut actual = initial.clone();

        let result =
            apply_weighted_round_to_rows(&mut actual, &binned, &stumps, None, &[1, 4], 1.0);

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_undersized_linear_raw_input_before_mutation() {
        let binned =
            BinnedMatrix::new(4, 2, 3, vec![0, 0, 1, 0, 2, 0, 3, 0]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Linear(LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0])),
            LeafValue::Linear(LinearLeaf::identity_scaled(-2.0, vec![0.25], vec![1])),
        )];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let raw_features = vec![1.0, 10.0, 2.0];
        let mut actual = initial.clone();

        let result = apply_weighted_round_to_rows(
            &mut actual,
            &binned,
            &stumps,
            Some((&raw_features, 2)),
            &[1, 3],
            1.0,
        );

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_prediction_length_mismatch_before_mutation() {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(1.5),
            LeafValue::Scalar(-0.5),
        )];
        let initial = vec![10.0, 20.0, 30.0];
        let mut actual = initial.clone();

        let result = apply_weighted_round_to_rows(&mut actual, &binned, &stumps, None, &[1], 1.0);

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_zero_feature_count_before_mutation() {
        let binned =
            BinnedMatrix::new(4, 2, 3, vec![0, 0, 1, 0, 2, 0, 3, 0]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Linear(LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0])),
            LeafValue::Linear(LinearLeaf::identity_scaled(-2.0, vec![0.25], vec![1])),
        )];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let raw_features = vec![1.0];
        let mut actual = initial.clone();

        let result = apply_weighted_round_to_rows(
            &mut actual,
            &binned,
            &stumps,
            Some((&raw_features, 0)),
            &[1, 3],
            1.0,
        );

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_raw_feature_count_mismatch_before_mutation() {
        let binned =
            BinnedMatrix::new(4, 2, 3, vec![0, 0, 1, 0, 2, 0, 3, 0]).expect("valid binned matrix");
        let stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Linear(LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0])),
            LeafValue::Linear(LinearLeaf::identity_scaled(-2.0, vec![0.25], vec![1])),
        )];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let raw_features = vec![1.0, 2.0, 3.0, 4.0];
        let mut actual = initial.clone();

        let result = apply_weighted_round_to_rows(
            &mut actual,
            &binned,
            &stumps,
            Some((&raw_features, 1)),
            &[1, 3],
            1.0,
        );

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn restricted_rows_reject_duplicate_local_ids_before_mutation() {
        let binned = BinnedMatrix::new(4, 1, 3, vec![0, 1, 2, 3]).expect("valid binned matrix");
        let mut duplicate_split = scalar_split(0, true, false, None);
        duplicate_split.node_id = 0;
        let stumps = vec![
            TrainedStump::new_unweighted(
                scalar_split(1, true, false, None),
                LeafValue::Scalar(1.5),
                LeafValue::Scalar(-0.5),
            ),
            TrainedStump::new_unweighted(
                duplicate_split,
                LeafValue::Scalar(9.0),
                LeafValue::Scalar(-9.0),
            ),
        ];
        let initial = vec![10.0, 20.0, 30.0, 40.0];
        let mut actual = initial.clone();

        let result =
            apply_weighted_round_to_rows(&mut actual, &binned, &stumps, None, &[1, 3], 1.0);

        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
        assert_eq!(actual, initial);
    }

    #[test]
    fn weighted_full_no_ops_ignore_malformed_unused_inputs() {
        let binned = BinnedMatrix::new(2, 1, 1, vec![0, 1]).expect("valid binned matrix");
        let malformed_raw = vec![1.0];
        let initial = vec![10.0, 20.0];

        let mut empty_stump_predictions = initial.clone();
        let empty_result = apply_weighted_round_to_predictions(
            &mut empty_stump_predictions,
            &binned,
            &[],
            Some((&malformed_raw, 0)),
            1.0,
        );
        assert_eq!(empty_result, Ok(()));
        assert_eq!(empty_stump_predictions, initial);

        let mut malformed_split = scalar_split(0, true, false, None);
        malformed_split.feature_index = 1;
        let malformed_stumps = vec![TrainedStump::new_unweighted(
            malformed_split,
            LeafValue::Scalar(1.0),
            LeafValue::Scalar(-1.0),
        )];
        let mut zero_factor_predictions = initial.clone();
        let mut zero_factor_accumulator = vec![3.0, 4.0];
        let initial_accumulator = zero_factor_accumulator.clone();
        let zero_result = apply_weighted_round_to_predictions_and_accumulator(
            &mut zero_factor_predictions,
            &mut zero_factor_accumulator,
            &binned,
            &malformed_stumps,
            None,
            0.0,
            0.0,
        );
        assert_eq!(zero_result, Ok(()));
        assert_eq!(zero_factor_predictions, initial);
        assert_eq!(zero_factor_accumulator, initial_accumulator);
    }

    #[test]
    fn restricted_no_ops_still_validate_malformed_inputs() {
        let binned = BinnedMatrix::new(2, 1, 1, vec![0, 1]).expect("valid binned matrix");
        let malformed_raw = vec![1.0];
        let initial = vec![10.0, 20.0];

        let mut empty_stump_predictions = initial.clone();
        let empty_result = apply_weighted_round_to_rows(
            &mut empty_stump_predictions,
            &binned,
            &[],
            Some((&malformed_raw, 0)),
            &[],
            0.0,
        );
        assert!(matches!(
            empty_result,
            Err(EngineError::ContractViolation(_))
        ));
        assert_eq!(empty_stump_predictions, initial);

        let mut malformed_split = scalar_split(0, true, false, None);
        malformed_split.feature_index = 1;
        let malformed_stumps = vec![TrainedStump::new_unweighted(
            malformed_split,
            LeafValue::Scalar(1.0),
            LeafValue::Scalar(-1.0),
        )];
        let mut zero_factor_predictions = initial.clone();
        let zero_result = apply_weighted_round_to_rows(
            &mut zero_factor_predictions,
            &binned,
            &malformed_stumps,
            None,
            &[0],
            0.0,
        );
        assert!(matches!(
            zero_result,
            Err(EngineError::ContractViolation(_))
        ));
        assert_eq!(zero_factor_predictions, initial);
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

    fn binary_leaf_optimum(targets: &[f32], l2: f32) -> f32 {
        // Solve sum(sigmoid(v) - y) + l2 * v = 0 by bisection.
        let (mut lo, mut hi) = (-50.0_f64, 50.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let p = 1.0 / (1.0 + (-mid).exp());
            let g: f64 = targets.iter().map(|&y| p - y as f64).sum::<f64>() + l2 as f64 * mid;
            if g > 0.0 { hi = mid } else { lo = mid }
        }
        (0.5 * (lo + hi)) as f32
    }

    #[test]
    fn binary_newton_refinement_converges_to_the_regularized_leaf_optimum() -> EngineResult<()> {
        // Rows 0,1 (bin 0) and the missing row 2 go left (default_left);
        // row 3 (bin 2) goes right. Left is pure positive, right negative.
        let binned =
            BinnedMatrix::new(4, 1, 2, vec![0, 0, MISSING_BIN_U8, 2]).expect("valid binned matrix");
        let targets = [1.0, 1.0, 1.0, 0.0];
        let predictions = [0.0_f32; 4];
        let (l2, lr) = (1.0_f32, 0.1_f32);
        // Builder's single step from p = 0.5: -G / (H + l2), scaled by lr.
        let left_one_step = 1.5 / (0.75 + l2) * lr;
        let right_one_step = -0.5 / (0.25 + l2) * lr;
        let mut stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(left_one_step),
            LeafValue::Scalar(right_one_step),
        )];

        let mut candidate = predictions.to_vec();
        apply_tree_to_binned_predictions(&mut candidate, &binned, &stumps, None)?;
        crate::leaf_refinement::refine_binary_newton_leaf_values(
            &mut stumps,
            &binned,
            &predictions,
            &mut candidate,
            &targets,
            None,
            l2,
            lr,
            50,
            100.0,
        )?;
        // The candidate scores now carry the refined tree.
        let mut replayed = predictions.to_vec();
        apply_tree_to_binned_predictions(&mut replayed, &binned, &stumps, None)?;
        assert_close(&candidate, &replayed);

        let left = stumps[0].left_leaf_value.as_scalar() / lr;
        let right = stumps[0].right_leaf_value.as_scalar() / lr;
        assert!(
            (left - binary_leaf_optimum(&targets[..3], l2)).abs() < 1e-4,
            "{left}"
        );
        assert!(
            (right - binary_leaf_optimum(&targets[3..], l2)).abs() < 1e-4,
            "{right}"
        );
        // The single step undershoots both leaves.
        assert!(left > left_one_step / lr && right < right_one_step / lr);
        Ok(())
    }

    #[test]
    fn binary_newton_refinement_does_not_depend_on_thread_count() -> EngineResult<()> {
        // Enough rows for many parallel chunks, with gradients of mixed
        // magnitude so a different summation order would change the bits.
        let rows = 50_000;
        let bins: Vec<u8> = (0..rows).map(|row| (row % 3) as u8).collect();
        let binned = BinnedMatrix::new(rows, 1, 3, bins).expect("valid binned matrix");
        let targets: Vec<f32> = (0..rows)
            .map(|row| ((row * 7919) % 5 < 2) as u8 as f32)
            .collect();
        let predictions: Vec<f32> = (0..rows)
            .map(|row| ((row * 104_729) % 1000) as f32 / 97.0 - 5.0)
            .collect();
        let refine = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool builds")
                .install(|| {
                    let mut stumps = vec![TrainedStump::new_unweighted(
                        scalar_split(0, true, false, None),
                        LeafValue::Scalar(0.01),
                        LeafValue::Scalar(-0.01),
                    )];
                    let mut candidate = predictions.clone();
                    apply_tree_to_binned_predictions(&mut candidate, &binned, &stumps, None)?;
                    crate::leaf_refinement::refine_binary_newton_leaf_values(
                        &mut stumps,
                        &binned,
                        &predictions,
                        &mut candidate,
                        &targets,
                        None,
                        3.0,
                        0.1,
                        2,
                        100.0,
                    )?;
                    Ok::<_, EngineError>((stumps, candidate))
                })
        };
        let (one_stumps, one_scores) = refine(1)?;
        for threads in [2, 3, 8] {
            let (stumps, scores) = refine(threads)?;
            assert_eq!(stumps, one_stumps, "{threads} threads");
            assert_eq!(scores, one_scores, "{threads} threads");
        }
        Ok(())
    }

    #[test]
    fn binary_newton_refinement_bounds_the_scaled_leaf_like_the_builder() -> EngineResult<()> {
        // A pure-positive leaf wants a large value; the cap applies to the
        // learning-rate-scaled leaf, as in the tree builder.
        let binned = BinnedMatrix::new(4, 1, 2, vec![0, 0, 0, 2]).expect("valid binned matrix");
        let (lr, cap) = (0.1_f32, 0.2_f32);
        let mut stumps = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(0.1),
            LeafValue::Scalar(-0.1),
        )];
        let predictions = [0.0_f32; 4];
        let mut candidate = predictions.to_vec();
        apply_tree_to_binned_predictions(&mut candidate, &binned, &stumps, None)?;
        crate::leaf_refinement::refine_binary_newton_leaf_values(
            &mut stumps,
            &binned,
            &predictions,
            &mut candidate,
            &[1.0, 1.0, 1.0, 0.0],
            None,
            0.01,
            lr,
            50,
            cap,
        )?;
        assert!((stumps[0].left_leaf_value.as_scalar() - cap).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn binary_newton_refinement_with_zero_steps_is_a_no_op() -> EngineResult<()> {
        let binned = BinnedMatrix::new(2, 1, 2, vec![0, 2]).expect("valid binned matrix");
        let original = vec![TrainedStump::new_unweighted(
            scalar_split(1, true, false, None),
            LeafValue::Scalar(0.3),
            LeafValue::Scalar(-0.2),
        )];
        let mut stumps = original.clone();
        crate::leaf_refinement::refine_binary_newton_leaf_values(
            &mut stumps,
            &binned,
            &[0.0, 0.0],
            &mut [0.3, -0.2],
            &[1.0, 0.0],
            None,
            3.0,
            0.1,
            0,
            100.0,
        )?;
        assert_eq!(stumps, original);
        Ok(())
    }
}
