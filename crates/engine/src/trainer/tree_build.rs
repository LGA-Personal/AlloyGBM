//! Tree construction helpers shared by the single-output trainer.
//!
//! This module hosts the two tree-builder entry points — level-wise and
//! leaf-wise — along with the shared best-split dispatcher, the histogram
//! subtraction trick, the iteration-controls validator, and the
//! single-categorical target encoding pre-pass. Extracted from `lib.rs` to
//! keep the trainer surface area manageable.

use alloygbm_categorical::fit_transform_target_encoder;
use alloygbm_core::{
    BinnedMatrix, DatasetMatrix, FactorExposureMatrix, FeatureTile, GradientPair, HistogramBundle,
    LeafModelKind, LeafValue, LinearLeaf, MAX_PL_REGRESSORS, NodeSlice, PartitionResult,
    SplitCandidate, TrainParams, TrainingDataset, leaf_effective_gradient,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::error::{EngineError, EngineResult};
use crate::morph_state::MorphTreeContext;
use crate::round::apply_partition_leaf_updates;
use crate::split_options::{CategoricalFeatureInfo, SplitSelectionOptions};
use crate::trainer::interaction::{
    InteractionConstraintIndex, filter_histogram_bundle_by_features,
};
use crate::trainer::validate::{factor_split_context_for_node, validate_training_alignment};
use crate::traits::BackendOps;
use crate::tree_node::{encode_tree_node_id, left_child_node_id, right_child_node_id};
use crate::types::{
    CategoricalTargetEncodingSpec, IterationControls, IterationStopReason, TrainedStump,
};

/// Small epsilon added to leaf value denominators to prevent division by zero.
pub(crate) const LEAF_EPSILON: f32 = 1e-6;

/// Type alias for an active node entry in the level-wise tree builder.
/// Fields: (local_node_id, row_indices, histograms, parent_leaf_value, parent_linear_leaf)
type ActiveNodeEntry = (u32, Vec<u32>, HistogramBundle, f32, Option<LinearLeaf>);

/// Type alias for a split linear leaf pair (delta, delta, absolute, absolute).
type LinearLeafQuad = (LinearLeaf, LinearLeaf, LinearLeaf, LinearLeaf);

/// Type alias for a pair of optional linear leaves (delta pair, absolute pair).
type LinearLeafPairSplit = (
    Option<(LinearLeaf, LinearLeaf)>,
    Option<(LinearLeaf, LinearLeaf)>,
);

pub(crate) fn apply_single_categorical_target_encoding(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    spec: &CategoricalTargetEncodingSpec,
) -> EngineResult<(TrainingDataset, BinnedMatrix)> {
    validate_training_alignment(dataset, binned_matrix)?;

    let row_count = dataset.row_count();
    let feature_count = dataset.matrix.feature_count;
    if spec.feature_index >= feature_count {
        return Err(EngineError::ContractViolation(format!(
            "categorical feature index {} is out of bounds for feature_count {}",
            spec.feature_index, feature_count
        )));
    }
    if spec.values.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "categorical values length {} does not match row_count {}",
            spec.values.len(),
            row_count
        )));
    }

    let (_, encoded_values) = fit_transform_target_encoder(
        &spec.config,
        &spec.values,
        &dataset.targets,
        dataset.time_index.as_deref(),
    )
    .map_err(|error| EngineError::ContractViolation(error.to_string()))?;
    let (encoded_bins, encoded_max_bin) = encode_bins_from_encoded_values(&encoded_values)?;

    let mut encoded_dense_values = dataset.matrix.values.clone();
    for (row_index, &encoded_value) in encoded_values.iter().enumerate() {
        let offset = row_index * feature_count + spec.feature_index;
        encoded_dense_values[offset] = encoded_value;
    }

    let encoded_dataset = TrainingDataset {
        matrix: DatasetMatrix::new(row_count, feature_count, encoded_dense_values)?,
        targets: dataset.targets.clone(),
        sample_weights: dataset.sample_weights.clone(),
        time_index: dataset.time_index.clone(),
        group_id: dataset.group_id.clone(),
        factor_exposures: dataset.factor_exposures.clone(),
    };

    let mut encoded_bins_payload = binned_matrix.bins.clone();
    for (row_index, &encoded_bin) in encoded_bins.iter().enumerate() {
        let offset = row_index * feature_count + spec.feature_index;
        encoded_bins_payload[offset] = encoded_bin;
    }
    let encoded_binned_matrix = BinnedMatrix::new(
        row_count,
        feature_count,
        binned_matrix.max_bin.max(encoded_max_bin),
        encoded_bins_payload,
    )?;

    Ok((encoded_dataset, encoded_binned_matrix))
}

pub(crate) fn encode_bins_from_encoded_values(
    encoded_values: &[f32],
) -> EngineResult<(Vec<u8>, u16)> {
    if encoded_values.is_empty() {
        return Err(EngineError::ContractViolation(
            "encoded values cannot be empty".to_string(),
        ));
    }

    for (index, value) in encoded_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::ContractViolation(format!(
                "encoded value at index {index} must be finite"
            )));
        }
    }

    let mut unique_values = encoded_values.to_vec();
    unique_values.sort_by(f32::total_cmp);
    unique_values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    if unique_values.len() > 256 {
        return Err(EngineError::ContractViolation(format!(
            "encoded cardinality {} exceeds supported max 256",
            unique_values.len(),
        )));
    }

    let mut bins = Vec::with_capacity(encoded_values.len());
    for value in encoded_values {
        let position = unique_values
            .binary_search_by(|probe| probe.total_cmp(value))
            .map_err(|_| {
                EngineError::ContractViolation(
                    "encoded value lookup failed during bin mapping".to_string(),
                )
            })?;
        bins.push(position as u8);
    }
    let max_bin = (unique_values.len().saturating_sub(1)) as u16;
    Ok((bins, max_bin))
}

/// Dispatch best-split finding to either the morph variant or the standard
/// variant based on whether a [`MorphTreeContext`] is supplied. Centralizes
/// the choice so all call sites in `build_tree_level_wise` /
/// `build_tree_leaf_wise` stay consistent.
pub(crate) fn find_best_split_dispatch<B: BackendOps>(
    backend: &B,
    histograms: &HistogramBundle,
    options: SplitSelectionOptions,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<&MorphTreeContext<'_>>,
    factor_context: Option<&crate::split_options::FactorSplitContext<'_>>,
) -> EngineResult<Option<SplitCandidate>> {
    if let Some(m) = morph {
        let ctx = m
            .state
            .morph_context(m.iteration, m.total_iterations, m.class_idx);
        backend.best_split_morph_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            &ctx,
            factor_context,
        )
    } else {
        backend.best_split_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
        )
    }
}

/// Build a single tree using level-wise (breadth-first) growth strategy.
///
/// Splits all nodes at depth d before moving to depth d+1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_tree_level_wise<B: BackendOps>(
    backend: &B,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    root_row_indices: Vec<u32>,
    round_index: usize,
    feature_tiles: &[FeatureTile],
    split_options: SplitSelectionOptions,
    params: &TrainParams,
    controls: &IterationControls,
    candidate_predictions: &mut [f32],
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<MorphTreeContext<'_>>,
    raw_feature_values: &[f32],
    factor_exposures: Option<&FactorExposureMatrix>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let mut candidate_round_stumps = Vec::new();
    let mut round_rejection_reason = IterationStopReason::NoSplitCandidate;
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    // Interaction-constraint bookkeeping (no-op when empty).  We track the
    // bitset of still-active groups per node so that the split search can
    // skip features that no surviving group allows on this path.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        binned_matrix.feature_count,
    )?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }
    // Maintain each active node's absolute leaf output so child updates
    // can replace parent contribution via deltas (tree semantics).
    // depth is the current tree level (0-indexed); all nodes at this level share the same depth.
    // The Option<LinearLeaf> carries the parent's absolute linear leaf (for weight delta computation).
    let mut active_nodes: Vec<ActiveNodeEntry> =
        vec![(0_u32, root_node.row_indices, root_histograms, 0.0_f32, None)];

    for depth in 0..(params.max_depth as usize) {
        if active_nodes.is_empty() {
            break;
        }

        let mut next_nodes = Vec::new();
        for (local_node_id, node_rows, histograms, parent_leaf_value, parent_linear_leaf) in
            active_nodes
        {
            let node_id = encode_tree_node_id(round_index, local_node_id)?;
            let node = NodeSlice::new(node_id, node_rows)?;
            let factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &node.row_indices,
            );
            // Filter histogram bundle by interaction constraints (no-op when
            // no constraints are active).  Cloning the bundle here is the
            // simplest way to plug filtering in without changing the
            // `BackendOps` trait surface; the clone is `O(allowed_features
            // × bins)` and only runs on constrained fits.
            let node_active = node_active_groups.get(&local_node_id).copied();
            let filtered_histograms_storage;
            let histograms_for_split = match (constraint_index.as_ref(), node_active) {
                (Some(idx), Some(active_groups)) => {
                    filtered_histograms_storage =
                        filter_histogram_bundle_by_features(&histograms, |f| {
                            idx.feature_allowed(active_groups, f)
                        });
                    &filtered_histograms_storage
                }
                _ => &histograms,
            };
            let Some(mut split) = find_best_split_dispatch(
                backend,
                histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                factor_context.as_ref(),
            )?
            else {
                continue;
            };
            if !split.gain.is_finite() || split.gain <= controls.min_split_gain {
                round_rejection_reason = IterationStopReason::GainBelowThreshold;
                continue;
            }

            let (partition, left_stats, right_stats) =
                backend.apply_split_with_stats(binned_matrix, gradients, &node, &split)?;
            if partition.left_row_indices.len() + partition.right_row_indices.len()
                != node.row_indices.len()
            {
                return Err(EngineError::ContractViolation(
                    "split partition does not cover all node rows".to_string(),
                ));
            }
            if partition.left_row_indices.is_empty()
                || partition.right_row_indices.is_empty()
                || partition.left_row_indices.len() < controls.min_rows_per_leaf
                || partition.right_row_indices.len() < controls.min_rows_per_leaf
            {
                round_rejection_reason = IterationStopReason::LeafRowsBelowThreshold;
                continue;
            }

            if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "backend produced non-positive hessian sums".to_string(),
                ));
            }

            let left_grad = leaf_effective_gradient(
                left_stats.grad_sum,
                left_stats.grad_sq_sum,
                left_stats.row_count,
                split_options.l1_alpha,
                split_options.dro_config.as_ref(),
            );
            let right_grad = leaf_effective_gradient(
                right_stats.grad_sum,
                right_stats.grad_sq_sum,
                right_stats.row_count,
                split_options.l1_alpha,
                split_options.dro_config.as_ref(),
            );
            let lr = morph.map_or(params.learning_rate, |m| m.lr);
            let mut raw_left_leaf_value =
                -lr * left_grad / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
            let mut raw_right_leaf_value =
                -lr * right_grad / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

            // Morph leaf modifications: depth penalty + per-round shrinkage.
            // Children land at depth `depth + 1` in the tree.
            let morph_scale = if let Some(m) = morph.as_ref() {
                let child_depth = (depth + 1) as f32;
                let depth_penalty = m.state.config.depth_penalty_base.powf(child_depth / 3.0);
                let iter_shrinkage = 1.0
                    - m.state.config.morph_rate
                        * (m.iteration as f32 / m.total_iterations.max(1) as f32).min(1.0);
                let scale = depth_penalty * iter_shrinkage;
                raw_left_leaf_value *= scale;
                raw_right_leaf_value *= scale;
                scale
            } else {
                1.0
            };

            let left_leaf_absolute = raw_left_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let right_leaf_absolute = raw_right_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let left_leaf_value = left_leaf_absolute - parent_leaf_value;
            let right_leaf_value = right_leaf_absolute - parent_leaf_value;
            if left_leaf_value.abs() < controls.min_abs_leaf_value
                && right_leaf_value.abs() < controls.min_abs_leaf_value
            {
                round_rejection_reason = IterationStopReason::LeafMagnitudeBelowThreshold;
                continue;
            }

            // Monotone constraint enforcement.
            if !params.monotone_constraints.is_empty() {
                let fi = split.feature_index as usize;
                if fi < params.monotone_constraints.len() {
                    let constraint = params.monotone_constraints[fi];
                    if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                    if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                }
            }

            // max_leaves enforcement.
            if let Some(max_leaves) = controls.max_leaves {
                let leaves_after_split = candidate_round_stumps.len() + 2;
                if leaves_after_split > max_leaves {
                    round_rejection_reason = IterationStopReason::MaxLeavesReached;
                    continue;
                }
            }

            // ── Linear leaf path ───────────────────────────────────────────────
            // If leaf_model == Linear, build a LinearHistogramBundle for this node
            // and solve closed-form ridge leaves. Falls back to scalar on any error.
            let linear_leaf_computation_result: Option<LinearLeafQuad> = if params.leaf_model
                == LeafModelKind::Linear
                && !raw_feature_values.is_empty()
                && !split.is_categorical
            {
                let d = binned_matrix.feature_count.min(MAX_PL_REGRESSORS);
                let regressor_features: Vec<u32> = (0..d as u32).collect();
                backend
                    .build_linear_histograms(
                        binned_matrix,
                        gradients,
                        &node,
                        feature_tiles,
                        &regressor_features,
                        raw_feature_values,
                        binned_matrix.row_count,
                        binned_matrix.feature_count,
                    )
                    .ok()
                    .and_then(|lin_hist| {
                        backend.compute_linear_leaf_pair(
                            &lin_hist,
                            split.feature_index,
                            split.threshold_bin as usize,
                            split.default_left,
                            split_options.missing_bin_index,
                            lr,
                            split_options.l2_lambda,
                        )
                    })
                    .map(|(mut ll_abs, mut rl_abs)| {
                        // Apply morph scaling to weights and intercept.
                        ll_abs.intercept *= morph_scale;
                        rl_abs.intercept *= morph_scale;
                        for w in &mut ll_abs.weights {
                            *w *= morph_scale;
                        }
                        for w in &mut rl_abs.weights {
                            *w *= morph_scale;
                        }
                        // Clamp intercepts (absolute values).
                        ll_abs.intercept = ll_abs
                            .intercept
                            .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                        rl_abs.intercept = rl_abs
                            .intercept
                            .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                        // Compute delta versions (parent-relative).
                        let mut ll_delta = ll_abs.clone();
                        let mut rl_delta = rl_abs.clone();
                        ll_delta.intercept -= parent_leaf_value;
                        rl_delta.intercept -= parent_leaf_value;
                        if let Some(ref p) = parent_linear_leaf {
                            let d = p.weights.len().min(ll_delta.weights.len());
                            for i in 0..d {
                                ll_delta.weights[i] -= p.weights[i];
                            }
                            let d = p.weights.len().min(rl_delta.weights.len());
                            for i in 0..d {
                                rl_delta.weights[i] -= p.weights[i];
                            }
                        }
                        (ll_delta, rl_delta, ll_abs, rl_abs)
                    })
            } else {
                None
            };
            // Split into delta pair (for storage/prediction) and absolute pair (for child tracking).
            let (linear_leaf_pair, linear_leaf_abs_pair): LinearLeafPairSplit =
                match linear_leaf_computation_result {
                    Some((ll_d, rl_d, ll_a, rl_a)) => (Some((ll_d, rl_d)), Some((ll_a, rl_a))),
                    None => (None, None),
                };

            // Apply candidate_predictions update.
            if let Some((ref ll, ref rl)) = linear_leaf_pair {
                let fc = binned_matrix.feature_count;
                for &row in &partition.left_row_indices {
                    let r = row as usize;
                    if r < candidate_predictions.len() {
                        candidate_predictions[r] += ll.eval(raw_feature_values, r * fc);
                    }
                }
                for &row in &partition.right_row_indices {
                    let r = row as usize;
                    if r < candidate_predictions.len() {
                        candidate_predictions[r] += rl.eval(raw_feature_values, r * fc);
                    }
                }
            } else {
                apply_partition_leaf_updates(
                    candidate_predictions,
                    &partition,
                    left_leaf_value,
                    right_leaf_value,
                )?;
            }

            split.left_stats = left_stats;
            split.right_stats = right_stats;

            let PartitionResult {
                left_row_indices,
                right_row_indices,
            } = partition;
            if depth + 1 < params.max_depth as usize {
                let left_local_node_id = left_child_node_id(local_node_id)?;
                let right_local_node_id = right_child_node_id(local_node_id)?;
                let left_node_id = encode_tree_node_id(round_index, left_local_node_id)?;
                let right_node_id = encode_tree_node_id(round_index, right_local_node_id)?;

                // Propagate interaction-constraint active groups to children.
                // Splitting on an unconstrained feature leaves the active
                // set unchanged; a constrained feature narrows it.
                if let (Some(idx), Some(active_groups)) = (constraint_index.as_ref(), node_active) {
                    let child_groups = idx.descend(active_groups, split.feature_index);
                    node_active_groups.insert(left_local_node_id, child_groups);
                    node_active_groups.insert(right_local_node_id, child_groups);
                }

                // Determine the parent-leaf values to track for children.
                // When we have linear leaves, the scalar parent value uses the intercept,
                // and we also pass the full absolute linear leaf for weight delta computation.
                let (left_parent_val, right_parent_val) =
                    if let Some((ref ll_a, ref rl_a)) = linear_leaf_abs_pair {
                        (ll_a.intercept, rl_a.intercept)
                    } else {
                        (left_leaf_absolute, right_leaf_absolute)
                    };
                let left_parent_ll = linear_leaf_abs_pair.as_ref().map(|(ll, _)| ll.clone());
                let right_parent_ll = linear_leaf_abs_pair.as_ref().map(|(_, rl)| rl.clone());

                if left_row_indices.len() <= right_row_indices.len() {
                    let left_node = NodeSlice::new(left_node_id, left_row_indices)?;
                    let left_histograms = backend.build_histograms(
                        binned_matrix,
                        gradients,
                        &left_node,
                        feature_tiles,
                    )?;
                    let right_histograms =
                        subtract_histogram_bundle(&histograms, &left_histograms, right_node_id)?;
                    next_nodes.push((
                        left_local_node_id,
                        left_node.row_indices,
                        left_histograms,
                        left_parent_val,
                        left_parent_ll,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_row_indices,
                        right_histograms,
                        right_parent_val,
                        right_parent_ll,
                    ));
                } else {
                    let right_node = NodeSlice::new(right_node_id, right_row_indices)?;
                    let right_histograms = backend.build_histograms(
                        binned_matrix,
                        gradients,
                        &right_node,
                        feature_tiles,
                    )?;
                    let left_histograms =
                        subtract_histogram_bundle(&histograms, &right_histograms, left_node_id)?;
                    next_nodes.push((
                        left_local_node_id,
                        left_row_indices,
                        left_histograms,
                        left_parent_val,
                        left_parent_ll,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_node.row_indices,
                        right_histograms,
                        right_parent_val,
                        right_parent_ll,
                    ));
                }
            }

            let (final_left_leaf, final_right_leaf) = if let Some((ll, rl)) = linear_leaf_pair {
                (LeafValue::Linear(ll), LeafValue::Linear(rl))
            } else {
                (
                    LeafValue::Scalar(left_leaf_value),
                    LeafValue::Scalar(right_leaf_value),
                )
            };
            candidate_round_stumps.push(TrainedStump {
                split,
                left_leaf_value: final_left_leaf,
                right_leaf_value: final_right_leaf,
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            });
        }
        active_nodes = next_nodes;
    }

    if candidate_round_stumps.is_empty() {
        return Ok((Vec::new(), round_rejection_reason));
    }

    Ok((
        candidate_round_stumps,
        IterationStopReason::CompletedRequestedRounds,
    ))
}

/// A pending leaf split for the leaf-wise priority queue.
/// Ordered by gain (highest gain = highest priority).
pub(crate) struct PendingSplit {
    local_node_id: u32,
    row_indices: Vec<u32>,
    split_candidate: SplitCandidate,
    histograms: HistogramBundle,
    parent_leaf_value: f32,
    /// Absolute linear leaf of the parent (used to compute weight deltas for linear-leaf trees).
    parent_linear_leaf: Option<LinearLeaf>,
    depth: usize,
}

// PartialEq uses exact float comparison for the Eq trait bound required by
// BinaryHeap. NaN gains are filtered before insertion; ordering is handled
// by the Ord impl which falls back to Equal for NaN.
impl PartialEq for PendingSplit {
    fn eq(&self, other: &Self) -> bool {
        self.split_candidate.gain == other.split_candidate.gain
    }
}

impl Eq for PendingSplit {}

impl PartialOrd for PendingSplit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingSplit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.split_candidate
            .gain
            .partial_cmp(&other.split_candidate.gain)
            .unwrap_or(Ordering::Equal)
    }
}

/// Build a single tree using leaf-wise (best-first) growth strategy.
///
/// Instead of splitting all nodes at depth d before moving to depth d+1,
/// this always splits the leaf with the highest gain across the entire tree.
/// Stops when `max_leaves` is reached or no valid splits remain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_tree_leaf_wise<B: BackendOps>(
    backend: &B,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    root_row_indices: Vec<u32>,
    round_index: usize,
    feature_tiles: &[FeatureTile],
    split_options: SplitSelectionOptions,
    params: &TrainParams,
    controls: &IterationControls,
    candidate_predictions: &mut [f32],
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<MorphTreeContext<'_>>,
    raw_feature_values: &[f32],
    factor_exposures: Option<&FactorExposureMatrix>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let max_leaves = controls.max_leaves.unwrap_or(usize::MAX);
    let max_depth = params.max_depth as usize;

    // Build root histograms and find best split.
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    // Interaction-constraint bookkeeping (no-op when empty).  See the
    // matching block in `build_tree_level_wise` for the design rationale —
    // we filter histograms per node at split-search time so constrained
    // features can't appear on a path that already broke into a sibling
    // group.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        binned_matrix.feature_count,
    )?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }
    let root_factor_context = factor_split_context_for_node(
        params,
        binned_matrix,
        factor_exposures,
        &root_node.row_indices,
    );
    let root_filtered_storage;
    let root_histograms_for_split = match (
        constraint_index.as_ref(),
        node_active_groups.get(&0).copied(),
    ) {
        (Some(idx), Some(ag)) => {
            root_filtered_storage = filter_histogram_bundle_by_features(&root_histograms, |f| {
                idx.feature_allowed(ag, f)
            });
            &root_filtered_storage
        }
        _ => &root_histograms,
    };
    let root_split = find_best_split_dispatch(
        backend,
        root_histograms_for_split,
        split_options,
        feature_weights,
        categorical_features,
        morph.as_ref(),
        root_factor_context.as_ref(),
    )?;

    let Some(root_split) = root_split else {
        return Ok((Vec::new(), IterationStopReason::NoSplitCandidate));
    };
    if !root_split.gain.is_finite() || root_split.gain <= controls.min_split_gain {
        return Ok((Vec::new(), IterationStopReason::GainBelowThreshold));
    }

    let mut queue = BinaryHeap::new();
    queue.push(PendingSplit {
        local_node_id: 0,
        row_indices: root_node.row_indices,
        split_candidate: root_split,
        histograms: root_histograms,
        parent_leaf_value: 0.0,
        parent_linear_leaf: None,
        depth: 0,
    });

    // Start with 1 leaf (the root). Each split adds 1 net leaf (splits one into two).
    let mut leaves_used = 1_usize;
    let mut stumps = Vec::new();
    let mut last_rejection = IterationStopReason::NoSplitCandidate;

    while let Some(pending) = queue.pop() {
        // Check max_leaves: splitting adds 1 net leaf.
        if leaves_used + 1 > max_leaves {
            last_rejection = IterationStopReason::MaxLeavesReached;
            break;
        }

        // Check max_depth constraint.
        if pending.depth >= max_depth {
            last_rejection = IterationStopReason::DepthBudgetReached;
            continue;
        }

        let local_node_id = pending.local_node_id;
        let node_id = encode_tree_node_id(round_index, local_node_id)?;
        let node = NodeSlice::new(node_id, pending.row_indices)?;
        let split = pending.split_candidate;

        // Apply the split: partition rows and get stats.
        let (partition, left_stats, right_stats) =
            backend.apply_split_with_stats(binned_matrix, gradients, &node, &split)?;

        if partition.left_row_indices.len() + partition.right_row_indices.len()
            != node.row_indices.len()
        {
            return Err(EngineError::ContractViolation(
                "split partition does not cover all node rows".to_string(),
            ));
        }
        if partition.left_row_indices.is_empty()
            || partition.right_row_indices.is_empty()
            || partition.left_row_indices.len() < controls.min_rows_per_leaf
            || partition.right_row_indices.len() < controls.min_rows_per_leaf
        {
            last_rejection = IterationStopReason::LeafRowsBelowThreshold;
            continue;
        }

        if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
            return Err(EngineError::ContractViolation(
                "backend produced non-positive hessian sums".to_string(),
            ));
        }

        // Compute leaf values.
        let left_grad = leaf_effective_gradient(
            left_stats.grad_sum,
            left_stats.grad_sq_sum,
            left_stats.row_count,
            split_options.l1_alpha,
            split_options.dro_config.as_ref(),
        );
        let right_grad = leaf_effective_gradient(
            right_stats.grad_sum,
            right_stats.grad_sq_sum,
            right_stats.row_count,
            split_options.l1_alpha,
            split_options.dro_config.as_ref(),
        );
        let lr = morph.map_or(params.learning_rate, |m| m.lr);
        let mut raw_left_leaf_value =
            -lr * left_grad / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
        let mut raw_right_leaf_value =
            -lr * right_grad / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

        // Morph leaf modifications: depth penalty + per-round shrinkage.
        // Children of `pending` land at `pending.depth + 1` in the tree.
        let morph_scale = if let Some(m) = morph.as_ref() {
            let child_depth = (pending.depth + 1) as f32;
            let depth_penalty = m.state.config.depth_penalty_base.powf(child_depth / 3.0);
            let iter_shrinkage = 1.0
                - m.state.config.morph_rate
                    * (m.iteration as f32 / m.total_iterations.max(1) as f32).min(1.0);
            let scale = depth_penalty * iter_shrinkage;
            raw_left_leaf_value *= scale;
            raw_right_leaf_value *= scale;
            scale
        } else {
            1.0
        };

        let left_leaf_absolute =
            raw_left_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let right_leaf_absolute =
            raw_right_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let left_leaf_value = left_leaf_absolute - pending.parent_leaf_value;
        let right_leaf_value = right_leaf_absolute - pending.parent_leaf_value;

        if left_leaf_value.abs() < controls.min_abs_leaf_value
            && right_leaf_value.abs() < controls.min_abs_leaf_value
        {
            last_rejection = IterationStopReason::LeafMagnitudeBelowThreshold;
            continue;
        }

        // Monotone constraint enforcement.
        if !params.monotone_constraints.is_empty() {
            let fi = split.feature_index as usize;
            if fi < params.monotone_constraints.len() {
                let constraint = params.monotone_constraints[fi];
                if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
                    last_rejection = IterationStopReason::MonotoneConstraintViolation;
                    continue;
                }
                if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
                    last_rejection = IterationStopReason::MonotoneConstraintViolation;
                    continue;
                }
            }
        }

        // ── Linear leaf path ───────────────────────────────────────────────────
        let linear_leaf_computation_result: Option<LinearLeafQuad> = if params.leaf_model
            == LeafModelKind::Linear
            && !raw_feature_values.is_empty()
            && !split.is_categorical
        {
            let d = binned_matrix.feature_count.min(MAX_PL_REGRESSORS);
            let regressor_features: Vec<u32> = (0..d as u32).collect();
            backend
                .build_linear_histograms(
                    binned_matrix,
                    gradients,
                    &node,
                    feature_tiles,
                    &regressor_features,
                    raw_feature_values,
                    binned_matrix.row_count,
                    binned_matrix.feature_count,
                )
                .ok()
                .and_then(|lin_hist| {
                    backend.compute_linear_leaf_pair(
                        &lin_hist,
                        split.feature_index,
                        split.threshold_bin as usize,
                        split.default_left,
                        split_options.missing_bin_index,
                        lr,
                        split_options.l2_lambda,
                    )
                })
                .map(|(mut ll_abs, mut rl_abs)| {
                    ll_abs.intercept *= morph_scale;
                    rl_abs.intercept *= morph_scale;
                    for w in &mut ll_abs.weights {
                        *w *= morph_scale;
                    }
                    for w in &mut rl_abs.weights {
                        *w *= morph_scale;
                    }
                    ll_abs.intercept = ll_abs
                        .intercept
                        .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                    rl_abs.intercept = rl_abs
                        .intercept
                        .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                    // Compute delta versions (parent-relative).
                    let mut ll_delta = ll_abs.clone();
                    let mut rl_delta = rl_abs.clone();
                    ll_delta.intercept -= pending.parent_leaf_value;
                    rl_delta.intercept -= pending.parent_leaf_value;
                    if let Some(ref p) = pending.parent_linear_leaf {
                        let d = p.weights.len().min(ll_delta.weights.len());
                        for i in 0..d {
                            ll_delta.weights[i] -= p.weights[i];
                        }
                        let d = p.weights.len().min(rl_delta.weights.len());
                        for i in 0..d {
                            rl_delta.weights[i] -= p.weights[i];
                        }
                    }
                    (ll_delta, rl_delta, ll_abs, rl_abs)
                })
        } else {
            None
        };
        // Split into delta pair (for storage/prediction) and absolute pair (for child tracking).
        let (linear_leaf_pair, linear_leaf_abs_pair): LinearLeafPairSplit =
            match linear_leaf_computation_result {
                Some((ll_d, rl_d, ll_a, rl_a)) => (Some((ll_d, rl_d)), Some((ll_a, rl_a))),
                None => (None, None),
            };

        // Commit the split: update predictions and record stump.
        if let Some((ref ll, ref rl)) = linear_leaf_pair {
            let fc = binned_matrix.feature_count;
            for &row in &partition.left_row_indices {
                let r = row as usize;
                if r < candidate_predictions.len() {
                    candidate_predictions[r] += ll.eval(raw_feature_values, r * fc);
                }
            }
            for &row in &partition.right_row_indices {
                let r = row as usize;
                if r < candidate_predictions.len() {
                    candidate_predictions[r] += rl.eval(raw_feature_values, r * fc);
                }
            }
        } else {
            apply_partition_leaf_updates(
                candidate_predictions,
                &partition,
                left_leaf_value,
                right_leaf_value,
            )?;
        }

        let mut committed_split = split;
        committed_split.left_stats = left_stats;
        committed_split.right_stats = right_stats;

        let (final_left_leaf, final_right_leaf) = if let Some((ll, rl)) = linear_leaf_pair {
            (LeafValue::Linear(ll), LeafValue::Linear(rl))
        } else {
            (
                LeafValue::Scalar(left_leaf_value),
                LeafValue::Scalar(right_leaf_value),
            )
        };
        stumps.push(TrainedStump {
            split: committed_split,
            left_leaf_value: final_left_leaf,
            right_leaf_value: final_right_leaf,
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        });
        leaves_used += 1;

        // Enqueue children if within depth budget.
        let child_depth = pending.depth + 1;
        if child_depth < max_depth {
            let left_local = left_child_node_id(local_node_id)?;
            let right_local = right_child_node_id(local_node_id)?;
            let left_node_id = encode_tree_node_id(round_index, left_local)?;
            let right_node_id = encode_tree_node_id(round_index, right_local)?;

            // Subtraction trick: build smaller child, subtract from parent for larger.
            // Determine parent leaf values and linear leaves for each child.
            let (left_parent_val, right_parent_val) =
                if let Some((ref ll_a, ref rl_a)) = linear_leaf_abs_pair {
                    (ll_a.intercept, rl_a.intercept)
                } else {
                    (left_leaf_absolute, right_leaf_absolute)
                };
            let left_parent_ll = linear_leaf_abs_pair.as_ref().map(|(ll, _)| ll.clone());
            let right_parent_ll = linear_leaf_abs_pair.as_ref().map(|(_, rl)| rl.clone());

            let (
                smaller_indices,
                larger_indices,
                smaller_node_id,
                larger_node_id,
                smaller_local,
                larger_local,
                smaller_parent_val,
                larger_parent_val,
                smaller_parent_ll,
                larger_parent_ll,
            ) = if partition.left_row_indices.len() <= partition.right_row_indices.len() {
                (
                    partition.left_row_indices,
                    partition.right_row_indices,
                    left_node_id,
                    right_node_id,
                    left_local,
                    right_local,
                    left_parent_val,
                    right_parent_val,
                    left_parent_ll,
                    right_parent_ll,
                )
            } else {
                (
                    partition.right_row_indices,
                    partition.left_row_indices,
                    right_node_id,
                    left_node_id,
                    right_local,
                    left_local,
                    right_parent_val,
                    left_parent_val,
                    right_parent_ll,
                    left_parent_ll,
                )
            };

            let smaller_node = NodeSlice::new(smaller_node_id, smaller_indices)?;
            let smaller_histograms =
                backend.build_histograms(binned_matrix, gradients, &smaller_node, feature_tiles)?;
            let larger_histograms = subtract_histogram_bundle(
                &pending.histograms,
                &smaller_histograms,
                larger_node_id,
            )?;

            // Propagate interaction-constraint active groups to both
            // children of the just-applied split.  Both children inherit the
            // same descended bitset because the split feature is shared.
            // (`split` itself was moved into `committed_split` above; we
            // read the feature index off the just-pushed stump instead.)
            let split_feature_for_descend =
                stumps.last().map(|s| s.split.feature_index).unwrap_or(0);
            let child_active_groups: Option<u64> = match (
                constraint_index.as_ref(),
                node_active_groups.get(&local_node_id).copied(),
            ) {
                (Some(idx), Some(ag)) => {
                    let descended = idx.descend(ag, split_feature_for_descend);
                    node_active_groups.insert(smaller_local, descended);
                    node_active_groups.insert(larger_local, descended);
                    Some(descended)
                }
                _ => None,
            };

            // Find best split for each child and enqueue if valid.
            let smaller_factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &smaller_node.row_indices,
            );
            let smaller_filtered_storage;
            let smaller_histograms_for_split =
                match (constraint_index.as_ref(), child_active_groups) {
                    (Some(idx), Some(ag)) => {
                        smaller_filtered_storage =
                            filter_histogram_bundle_by_features(&smaller_histograms, |f| {
                                idx.feature_allowed(ag, f)
                            });
                        &smaller_filtered_storage
                    }
                    _ => &smaller_histograms,
                };
            if let Some(child_split) = find_best_split_dispatch(
                backend,
                smaller_histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                smaller_factor_context.as_ref(),
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: smaller_local,
                    row_indices: smaller_node.row_indices,
                    split_candidate: child_split,
                    histograms: smaller_histograms,
                    parent_leaf_value: smaller_parent_val,
                    parent_linear_leaf: smaller_parent_ll,
                    depth: child_depth,
                });
            }

            let larger_factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &larger_indices,
            );
            let larger_filtered_storage;
            let larger_histograms_for_split = match (constraint_index.as_ref(), child_active_groups)
            {
                (Some(idx), Some(ag)) => {
                    larger_filtered_storage =
                        filter_histogram_bundle_by_features(&larger_histograms, |f| {
                            idx.feature_allowed(ag, f)
                        });
                    &larger_filtered_storage
                }
                _ => &larger_histograms,
            };
            if let Some(child_split) = find_best_split_dispatch(
                backend,
                larger_histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                larger_factor_context.as_ref(),
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: larger_local,
                    row_indices: larger_indices,
                    split_candidate: child_split,
                    histograms: larger_histograms,
                    parent_leaf_value: larger_parent_val,
                    parent_linear_leaf: larger_parent_ll,
                    depth: child_depth,
                });
            }
        }
    }

    if stumps.is_empty() {
        return Ok((Vec::new(), last_rejection));
    }

    Ok((stumps, IterationStopReason::CompletedRequestedRounds))
}

/// Subtract child histogram from parent, writing into an existing buffer.
///
/// This avoids allocating a new `HistogramBundle` by reusing `dest`.
/// `dest` must have the same feature count and bin counts as `parent`.
pub(crate) fn subtract_histogram_bundle_into(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
    dest: &mut HistogramBundle,
) -> EngineResult<()> {
    if parent.feature_histograms.len() != child.feature_histograms.len() {
        return Err(EngineError::ContractViolation(format!(
            "parent histogram feature count {} does not match child histogram feature count {}",
            parent.feature_histograms.len(),
            child.feature_histograms.len()
        )));
    }
    dest.node_id = node_id;
    for ((dest_fh, parent_fh), child_fh) in dest
        .feature_histograms
        .iter_mut()
        .zip(&parent.feature_histograms)
        .zip(&child.feature_histograms)
    {
        dest_fh.feature_index = parent_fh.feature_index;
        for ((dest_bin, parent_bin), child_bin) in dest_fh
            .bins
            .iter_mut()
            .zip(&parent_fh.bins)
            .zip(&child_fh.bins)
        {
            dest_bin.grad_sum = parent_bin.grad_sum - child_bin.grad_sum;
            dest_bin.hess_sum = parent_bin.hess_sum - child_bin.hess_sum;
            dest_bin.grad_sq_sum = parent_bin.grad_sq_sum - child_bin.grad_sq_sum;
            dest_bin.count = parent_bin.count - child_bin.count;
        }
    }
    Ok(())
}

pub(crate) fn subtract_histogram_bundle(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
) -> EngineResult<HistogramBundle> {
    // Pre-allocate a dest with the same structure, then delegate to the in-place variant.
    let feature_indices: Vec<u32> = parent
        .feature_histograms
        .iter()
        .map(|fh| fh.feature_index)
        .collect();
    let bin_count = parent
        .feature_histograms
        .first()
        .map_or(0, |fh| fh.bins.len());
    let mut dest = HistogramBundle::new_zeroed(&feature_indices, bin_count);
    subtract_histogram_bundle_into(parent, child, node_id, &mut dest)?;
    Ok(dest)
}

pub(crate) fn validate_iteration_controls(controls: IterationControls) -> EngineResult<()> {
    if controls.rounds == 0 {
        return Err(EngineError::InvalidConfig(
            "rounds must be greater than 0".to_string(),
        ));
    }
    if !controls.min_split_gain.is_finite() || controls.min_split_gain < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_split_gain must be finite and >= 0".to_string(),
        ));
    }
    if controls.min_rows_per_leaf == 0 {
        return Err(EngineError::InvalidConfig(
            "min_rows_per_leaf must be greater than 0".to_string(),
        ));
    }
    if !controls.min_abs_leaf_value.is_finite() || controls.min_abs_leaf_value < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_abs_leaf_value must be finite and >= 0".to_string(),
        ));
    }
    if !controls.max_abs_leaf_value.is_finite() || controls.max_abs_leaf_value <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "max_abs_leaf_value must be finite and > 0".to_string(),
        ));
    }
    if controls.min_abs_leaf_value > controls.max_abs_leaf_value {
        return Err(EngineError::InvalidConfig(
            "min_abs_leaf_value cannot exceed max_abs_leaf_value".to_string(),
        ));
    }
    if !controls.min_loss_improvement.is_finite() || controls.min_loss_improvement < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_loss_improvement must be finite and >= 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&controls.row_subsample) || controls.row_subsample == 0.0 {
        return Err(EngineError::InvalidConfig(
            "row_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&controls.col_subsample) || controls.col_subsample == 0.0 {
        return Err(EngineError::InvalidConfig(
            "col_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }
    if let Some(early_stopping_rounds) = controls.early_stopping_rounds
        && early_stopping_rounds == 0
    {
        return Err(EngineError::InvalidConfig(
            "early_stopping_rounds must be greater than 0".to_string(),
        ));
    }
    if !controls.min_validation_improvement.is_finite() || controls.min_validation_improvement < 0.0
    {
        return Err(EngineError::InvalidConfig(
            "min_validation_improvement must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}
