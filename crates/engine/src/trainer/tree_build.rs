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
    LeafModelKind, LeafValue, LinearFeatureScaler, LinearLeaf, MAX_PL_REGRESSORS, NodeSlice,
    PartitionResult, SplitCandidate, TrainParams, TrainingDataset, leaf_effective_gradient,
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::colsample::{ColsampleBynode, filter_histograms_for_node};
use crate::error::{EngineError, EngineResult};
use crate::morph_state::MorphTreeContext;
use crate::round::apply_partition_leaf_updates;
use crate::split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, PreparedLinearSplit,
    SplitSelectionOptions, feature_weighted_gain,
};
use crate::trainer::interaction::InteractionConstraintIndex;
use crate::trainer::monotone::{
    BoundedChildren, MonotoneBounds, has_active_monotone_constraints,
    monotone_constraint_for_feature, reconstruct_bounded_child,
};
use crate::trainer::validate::{factor_split_context_for_node, validate_training_alignment};
use crate::traits::{BackendOps, HistogramExecution};
use crate::tree_node::{encode_tree_node_id, left_child_node_id, right_child_node_id};
use crate::types::{
    CategoricalTargetEncodingSpec, IterationControls, IterationStopReason, TrainedStump,
};

/// Small epsilon added to leaf value denominators to prevent division by zero.
pub(crate) const LEAF_EPSILON: f32 = 1e-6;

struct ActiveNodeEntry {
    local_node_id: u32,
    row_indices: Vec<u32>,
    histograms: HistogramBundle,
    parent_leaf_value: f32,
    parent_linear_leaf: Option<LinearLeaf>,
    path_features: Vec<u32>,
    monotone_bounds: MonotoneBounds,
}

const MIN_NODE_PARALLEL_WORK: usize = 4_096;

struct LevelNodeProposal {
    local_node_id: u32,
    node_active_groups: Option<u64>,
    split: SplitCandidate,
    partition: PartitionResult,
    left_leaf_value: f32,
    right_leaf_value: f32,
    linear_leaf_pair: Option<(LinearLeaf, LinearLeaf)>,
    children: Option<LevelNodeChildren>,
}

struct LevelNodeChildren {
    left_local_node_id: u32,
    right_local_node_id: u32,
    left_histograms: HistogramBundle,
    right_histograms: HistogramBundle,
    left_parent_value: f32,
    right_parent_value: f32,
    left_parent_linear: Option<LinearLeaf>,
    right_parent_linear: Option<LinearLeaf>,
    path_features: Vec<u32>,
    left_bounds: MonotoneBounds,
    right_bounds: MonotoneBounds,
}

struct LevelNodeOutcome {
    local_node_id: u32,
    rejection_reason: Option<IterationStopReason>,
    proposal: Option<LevelNodeProposal>,
}

impl LevelNodeOutcome {
    fn no_split(local_node_id: u32) -> Self {
        Self {
            local_node_id,
            rejection_reason: None,
            proposal: None,
        }
    }

    fn rejected(local_node_id: u32, rejection_reason: IterationStopReason) -> Self {
        Self {
            local_node_id,
            rejection_reason: Some(rejection_reason),
            proposal: None,
        }
    }

    fn proposed(proposal: LevelNodeProposal) -> Self {
        Self {
            local_node_id: proposal.local_node_id,
            rejection_reason: None,
            proposal: Some(proposal),
        }
    }
}

struct LevelProposalContext<'a, B> {
    backend: &'a B,
    binned_matrix: &'a BinnedMatrix,
    gradients: &'a [GradientPair],
    round_index: usize,
    depth: usize,
    feature_tiles: &'a [FeatureTile],
    split_options: SplitSelectionOptions,
    params: &'a TrainParams,
    controls: &'a IterationControls,
    feature_weights: &'a [f32],
    categorical_features: &'a [CategoricalFeatureInfo],
    morph: Option<MorphTreeContext<'a>>,
    raw_feature_values: &'a [f32],
    feature_scaler: &'a LinearFeatureScaler,
    factor_exposures: Option<&'a FactorExposureMatrix>,
    constraint_index: Option<&'a InteractionConstraintIndex>,
    colsample_bynode: Option<ColsampleBynode>,
    histogram_execution: HistogramExecution,
}

/// Type alias for a split linear leaf pair (delta, delta, absolute, absolute).
type LinearLeafQuad = (LinearLeaf, LinearLeaf, LinearLeaf, LinearLeaf);

/// Type alias for a pair of optional linear leaves (delta pair, absolute pair).
type LinearLeafPairSplit = (
    Option<(LinearLeaf, LinearLeaf)>,
    Option<(LinearLeaf, LinearLeaf)>,
);

struct SelectedNodeSplit {
    split: SplitCandidate,
    prepared_linear_leaf_pair: Option<(LinearLeaf, LinearLeaf)>,
}

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

    let mut encoded_binned_matrix = binned_matrix.clone();
    encoded_binned_matrix.max_bin = encoded_binned_matrix.max_bin.max(encoded_max_bin);
    for (row_index, &encoded_bin) in encoded_bins.iter().enumerate() {
        encoded_binned_matrix.set_bin(row_index, spec.feature_index, u16::from(encoded_bin));
    }

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

fn linear_regressor_path_features(
    path_features: &[u32],
    split_feature: u32,
    split_is_categorical: bool,
    feature_count: usize,
) -> Vec<u32> {
    let split_feature_iter = (!split_is_categorical).then_some(split_feature).into_iter();
    let mut selected = Vec::with_capacity(MAX_PL_REGRESSORS);
    for feature in path_features.iter().copied().chain(split_feature_iter) {
        if (feature as usize) < feature_count
            && !selected.contains(&feature)
            && selected.len() < MAX_PL_REGRESSORS
        {
            selected.push(feature);
        }
    }
    selected
}

#[allow(clippy::too_many_arguments)]
fn select_node_split<B: BackendOps>(
    backend: &B,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    histograms: &HistogramBundle,
    path_features: &[u32],
    options: SplitSelectionOptions,
    params: &TrainParams,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<&MorphTreeContext<'_>>,
    factor_context: Option<&FactorSplitContext<'_>>,
    raw_feature_values: &[f32],
    feature_scaler: &LinearFeatureScaler,
    parent_leaf_value: f32,
    parent_linear_leaf: Option<&LinearLeaf>,
    max_abs_leaf_value: f32,
) -> EngineResult<Option<SelectedNodeSplit>> {
    let legacy_path = params.leaf_model != LeafModelKind::Linear
        || params.pl_split_candidates == 0
        || raw_feature_values.is_empty()
        || morph.is_some()
        || factor_context.is_some();
    if legacy_path {
        return find_best_split_dispatch(
            backend,
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
            factor_context,
        )
        .map(|split| {
            split.map(|split| SelectedNodeSplit {
                split,
                prepared_linear_leaf_pair: None,
            })
        });
    }

    let shortlist = backend.shortlist_standard_splits(
        histograms,
        options,
        feature_weights,
        categorical_features,
        params.pl_split_candidates,
    )?;
    let standard = match shortlist.best_overall {
        Some(split) => split,
        None => return Ok(None),
    };
    if standard.is_categorical {
        return Ok(Some(SelectedNodeSplit {
            split: standard,
            prepared_linear_leaf_pair: None,
        }));
    }

    let evaluated = shortlist
        .numeric_candidates
        .par_iter()
        .map(|candidate| {
            let regressor_features = linear_regressor_path_features(
                path_features,
                candidate.feature_index,
                false,
                binned_matrix.feature_count,
            );
            let linear_context = LinearContext {
                regressor_features,
                l2_lambda: options.l2_lambda,
                max_abs_leaf_value,
            };
            backend.evaluate_shortlisted_linear_feature(
                binned_matrix,
                gradients,
                node,
                candidate.feature_index,
                &linear_context,
                feature_scaler,
                raw_feature_values,
                binned_matrix.row_count,
                binned_matrix.feature_count,
                options,
                params.learning_rate,
                parent_leaf_value,
                parent_linear_leaf,
            )
        })
        .collect::<Vec<_>>();
    let mut best: Option<PreparedLinearSplit> = None;
    for prepared in evaluated {
        let Some(prepared) = prepared? else {
            continue;
        };
        let replace = best.as_ref().is_none_or(|current| {
            let trial = feature_weighted_gain(&prepared.split, feature_weights);
            let retained = feature_weighted_gain(&current.split, feature_weights);
            let tolerance = 1e-6_f32 * trial.abs().max(retained.abs()).max(1.0);
            trial > retained + tolerance
        });
        if replace {
            best = Some(prepared);
        }
    }

    Ok(Some(match best {
        Some(prepared) => SelectedNodeSplit {
            split: prepared.split,
            prepared_linear_leaf_pair: Some((prepared.left_leaf, prepared.right_leaf)),
        },
        None => SelectedNodeSplit {
            split: standard,
            prepared_linear_leaf_pair: None,
        },
    }))
}

fn should_parallelize_level(
    active_nodes: &[ActiveNodeEntry],
    feature_tiles: &[FeatureTile],
) -> bool {
    if active_nodes.len() < 2 || rayon::current_num_threads() < 2 {
        return false;
    }
    let selected_feature_count = feature_tiles
        .iter()
        .map(|tile| (tile.end_feature - tile.start_feature) as usize)
        .sum::<usize>()
        .max(1);
    let row_count = active_nodes
        .iter()
        .map(|node| node.row_indices.len())
        .sum::<usize>();
    row_count.saturating_mul(selected_feature_count) >= MIN_NODE_PARALLEL_WORK
}

fn sort_level_outcomes(outcomes: &mut [LevelNodeOutcome]) {
    outcomes.sort_unstable_by_key(|outcome| outcome.local_node_id);
}

fn propose_level_node<B: BackendOps>(
    context: &LevelProposalContext<'_, B>,
    active_node: ActiveNodeEntry,
    node_active_groups: Option<u64>,
) -> EngineResult<LevelNodeOutcome> {
    let ActiveNodeEntry {
        local_node_id,
        row_indices,
        mut histograms,
        parent_leaf_value,
        parent_linear_leaf,
        path_features,
        monotone_bounds,
    } = active_node;
    let node_id = encode_tree_node_id(context.round_index, local_node_id)?;
    let node = NodeSlice::new(node_id, row_indices)?;
    let factor_context = factor_split_context_for_node(
        context.params,
        context.binned_matrix,
        context.factor_exposures,
        &node.row_indices,
    );
    let parent_row_count = node.row_indices.len();
    let filtered_histograms_storage = filter_histograms_for_node(
        &histograms,
        context.constraint_index,
        node_active_groups,
        context.colsample_bynode,
        node_id as u64,
    );
    let histograms_for_split = filtered_histograms_storage.as_ref().unwrap_or(&histograms);
    let Some(SelectedNodeSplit {
        mut split,
        prepared_linear_leaf_pair,
    }) = select_node_split(
        context.backend,
        context.binned_matrix,
        context.gradients,
        &node,
        histograms_for_split,
        &path_features,
        context.split_options,
        context.params,
        context.feature_weights,
        context.categorical_features,
        context.morph.as_ref(),
        factor_context.as_ref(),
        context.raw_feature_values,
        context.feature_scaler,
        parent_leaf_value,
        parent_linear_leaf.as_ref(),
        context.controls.max_abs_leaf_value,
    )?
    else {
        return Ok(LevelNodeOutcome::no_split(local_node_id));
    };
    if !split.gain.is_finite() || split.gain <= context.controls.min_split_gain {
        return Ok(LevelNodeOutcome::rejected(
            local_node_id,
            IterationStopReason::GainBelowThreshold,
        ));
    }

    let (partition, left_stats, right_stats) = context.backend.apply_split_owned_with_stats(
        context.binned_matrix,
        context.gradients,
        node,
        &split,
    )?;
    if partition.left_row_indices.len() + partition.right_row_indices.len() != parent_row_count {
        return Err(EngineError::ContractViolation(
            "split partition does not cover all node rows".to_string(),
        ));
    }
    if partition.left_row_indices.is_empty()
        || partition.right_row_indices.is_empty()
        || partition.left_row_indices.len() < context.controls.min_rows_per_leaf
        || partition.right_row_indices.len() < context.controls.min_rows_per_leaf
    {
        return Ok(LevelNodeOutcome::rejected(
            local_node_id,
            IterationStopReason::LeafRowsBelowThreshold,
        ));
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
        context.split_options.l1_alpha,
        context.split_options.dro_config.as_ref(),
    );
    let right_grad = leaf_effective_gradient(
        right_stats.grad_sum,
        right_stats.grad_sq_sum,
        right_stats.row_count,
        context.split_options.l1_alpha,
        context.split_options.dro_config.as_ref(),
    );
    let child_depth = (context.depth + 1) as u32;
    let scheduled_lr = context
        .morph
        .map_or(context.params.learning_rate, |morph| morph.scheduled_lr());
    let leaf_scale = context.morph.map_or(context.params.learning_rate, |morph| {
        morph.leaf_scale_for_depth(child_depth).total
    });
    let raw_left_leaf_value = -leaf_scale * left_grad
        / (left_stats.hess_sum + context.split_options.l2_lambda + LEAF_EPSILON);
    let raw_right_leaf_value = -leaf_scale * right_grad
        / (right_stats.hess_sum + context.split_options.l2_lambda + LEAF_EPSILON);
    let morph_scale = context.morph.map_or(1.0, |morph| {
        morph.leaf_scale_for_depth(child_depth).multiplier
    });

    let raw_left_leaf_absolute = raw_left_leaf_value.clamp(
        -context.controls.max_abs_leaf_value,
        context.controls.max_abs_leaf_value,
    );
    let raw_right_leaf_absolute = raw_right_leaf_value.clamp(
        -context.controls.max_abs_leaf_value,
        context.controls.max_abs_leaf_value,
    );
    let monotone_constraints_active =
        has_active_monotone_constraints(&context.params.monotone_constraints);
    if monotone_constraints_active && context.params.leaf_model == LeafModelKind::Linear {
        return Err(EngineError::InvalidConfig(
            "active monotone constraints do not support linear leaves during tree growth"
                .to_string(),
        ));
    }
    let (
        left_leaf_value,
        right_leaf_value,
        left_leaf_absolute,
        right_leaf_absolute,
        left_bounds,
        right_bounds,
    ) = if monotone_constraints_active {
        let BoundedChildren {
            left_output,
            right_output,
            left_bounds,
            right_bounds,
        } = monotone_bounds.bound_children(
            monotone_constraint_for_feature(
                &context.params.monotone_constraints,
                split.feature_index,
            ),
            raw_left_leaf_absolute,
            raw_right_leaf_absolute,
        )?;
        let left = reconstruct_bounded_child(
            parent_leaf_value,
            left_output - parent_leaf_value,
            left_output,
            left_bounds,
            local_node_id,
            "left",
        )?;
        let right = reconstruct_bounded_child(
            parent_leaf_value,
            right_output - parent_leaf_value,
            right_output,
            right_bounds,
            local_node_id,
            "right",
        )?;
        (
            left.delta,
            right.delta,
            left.absolute,
            right.absolute,
            left_bounds,
            right_bounds,
        )
    } else {
        (
            raw_left_leaf_absolute - parent_leaf_value,
            raw_right_leaf_absolute - parent_leaf_value,
            raw_left_leaf_absolute,
            raw_right_leaf_absolute,
            monotone_bounds,
            monotone_bounds,
        )
    };
    if left_leaf_value.abs() < context.controls.min_abs_leaf_value
        && right_leaf_value.abs() < context.controls.min_abs_leaf_value
    {
        return Ok(LevelNodeOutcome::rejected(
            local_node_id,
            IterationStopReason::LeafMagnitudeBelowThreshold,
        ));
    }

    let linear_leaf_computation_result: Option<LinearLeafQuad> = if context.params.leaf_model
        == LeafModelKind::Linear
        && !context.raw_feature_values.is_empty()
        && !split.is_categorical
    {
        let regressor_features = linear_regressor_path_features(
            &path_features,
            split.feature_index,
            split.is_categorical,
            context.binned_matrix.feature_count,
        );
        prepared_linear_leaf_pair
            .or_else(|| {
                context.backend.compute_linear_leaf_pair_from_partitions(
                    context.binned_matrix,
                    context.gradients,
                    context.raw_feature_values,
                    context.binned_matrix.feature_count,
                    split.feature_index,
                    split.threshold_bin,
                    split.default_left,
                    &regressor_features,
                    context.feature_scaler,
                    &partition.left_row_indices,
                    &partition.right_row_indices,
                    scheduled_lr,
                    context.split_options.l2_lambda,
                    context.controls.max_abs_leaf_value,
                )
            })
            .map(|(mut left_absolute, mut right_absolute)| {
                left_absolute.intercept *= morph_scale;
                right_absolute.intercept *= morph_scale;
                for weight in &mut left_absolute.weights {
                    *weight *= morph_scale;
                }
                for weight in &mut right_absolute.weights {
                    *weight *= morph_scale;
                }
                left_absolute.intercept = left_absolute.intercept.clamp(
                    -context.controls.max_abs_leaf_value,
                    context.controls.max_abs_leaf_value,
                );
                right_absolute.intercept = right_absolute.intercept.clamp(
                    -context.controls.max_abs_leaf_value,
                    context.controls.max_abs_leaf_value,
                );
                let mut left_delta = left_absolute.clone();
                let mut right_delta = right_absolute.clone();
                left_delta.intercept -= parent_leaf_value;
                right_delta.intercept -= parent_leaf_value;
                if let Some(parent) = parent_linear_leaf.as_ref() {
                    for (slot, &feature) in left_delta.regressor_features.iter().enumerate() {
                        if let Some(parent_slot) = parent
                            .regressor_features
                            .iter()
                            .position(|&candidate| candidate == feature)
                            && slot < left_delta.weights.len()
                            && parent_slot < parent.weights.len()
                        {
                            left_delta.weights[slot] -= parent.weights[parent_slot];
                        }
                    }
                    for (slot, &feature) in right_delta.regressor_features.iter().enumerate() {
                        if let Some(parent_slot) = parent
                            .regressor_features
                            .iter()
                            .position(|&candidate| candidate == feature)
                            && slot < right_delta.weights.len()
                            && parent_slot < parent.weights.len()
                        {
                            right_delta.weights[slot] -= parent.weights[parent_slot];
                        }
                    }
                }
                (left_delta, right_delta, left_absolute, right_absolute)
            })
    } else {
        None
    };
    let (linear_leaf_pair, linear_leaf_abs_pair): LinearLeafPairSplit =
        match linear_leaf_computation_result {
            Some((left_delta, right_delta, left_absolute, right_absolute)) => (
                Some((left_delta, right_delta)),
                Some((left_absolute, right_absolute)),
            ),
            None => (None, None),
        };

    split.left_stats = left_stats;
    split.right_stats = right_stats;

    let (partition, children) = if context.depth + 1 < context.params.max_depth as usize {
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_node_id = encode_tree_node_id(context.round_index, left_local_node_id)?;
        let right_node_id = encode_tree_node_id(context.round_index, right_local_node_id)?;
        let (left_parent_value, right_parent_value) = linear_leaf_abs_pair.as_ref().map_or(
            (left_leaf_absolute, right_leaf_absolute),
            |(left, right)| (left.intercept, right.intercept),
        );
        let left_parent_linear = linear_leaf_abs_pair.as_ref().map(|(left, _)| left.clone());
        let right_parent_linear = linear_leaf_abs_pair
            .as_ref()
            .map(|(_, right)| right.clone());
        let child_path_features = linear_regressor_path_features(
            &path_features,
            split.feature_index,
            split.is_categorical,
            context.binned_matrix.feature_count,
        );

        let PartitionResult {
            left_row_indices,
            right_row_indices,
        } = partition;
        let (partition, left_histograms, right_histograms) =
            if left_row_indices.len() <= right_row_indices.len() {
                let left_node = NodeSlice::new(left_node_id, left_row_indices)?;
                let left_histograms = context.backend.build_histograms_with_execution(
                    context.binned_matrix,
                    context.gradients,
                    &left_node,
                    context.feature_tiles,
                    context.split_options.requires_grad_sq(),
                    context.histogram_execution,
                )?;
                histograms.subtract_child_in_place(&left_histograms, right_node_id)?;
                let right_histograms = histograms;
                (
                    PartitionResult {
                        left_row_indices: left_node.row_indices,
                        right_row_indices,
                    },
                    left_histograms,
                    right_histograms,
                )
            } else {
                let right_node = NodeSlice::new(right_node_id, right_row_indices)?;
                let right_histograms = context.backend.build_histograms_with_execution(
                    context.binned_matrix,
                    context.gradients,
                    &right_node,
                    context.feature_tiles,
                    context.split_options.requires_grad_sq(),
                    context.histogram_execution,
                )?;
                histograms.subtract_child_in_place(&right_histograms, left_node_id)?;
                let left_histograms = histograms;
                (
                    PartitionResult {
                        left_row_indices,
                        right_row_indices: right_node.row_indices,
                    },
                    left_histograms,
                    right_histograms,
                )
            };
        (
            partition,
            Some(LevelNodeChildren {
                left_local_node_id,
                right_local_node_id,
                left_histograms,
                right_histograms,
                left_parent_value,
                right_parent_value,
                left_parent_linear,
                right_parent_linear,
                path_features: child_path_features,
                left_bounds,
                right_bounds,
            }),
        )
    } else {
        (partition, None)
    };

    Ok(LevelNodeOutcome::proposed(LevelNodeProposal {
        local_node_id,
        node_active_groups,
        split,
        partition,
        left_leaf_value,
        right_leaf_value,
        linear_leaf_pair,
        children,
    }))
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
    colsample_bynode: Option<ColsampleBynode>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let mut candidate_round_stumps = Vec::new();
    let mut round_rejection_reason = IterationStopReason::NoSplitCandidate;
    let feature_scaler = LinearFeatureScaler::from_raw_matrix(
        raw_feature_values,
        binned_matrix.row_count,
        binned_matrix.feature_count,
    );
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms = backend.build_histograms_with_execution(
        binned_matrix,
        gradients,
        &root_node,
        feature_tiles,
        split_options.requires_grad_sq(),
        HistogramExecution::Parallel,
    )?;
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
    let mut active_nodes = vec![ActiveNodeEntry {
        local_node_id: 0,
        row_indices: root_node.row_indices,
        histograms: root_histograms,
        parent_leaf_value: 0.0,
        parent_linear_leaf: None,
        path_features: Vec::new(),
        monotone_bounds: MonotoneBounds::root(controls.max_abs_leaf_value)?,
    }];

    for depth in 0..(params.max_depth as usize) {
        if active_nodes.is_empty() {
            break;
        }

        let parallelize_nodes = should_parallelize_level(&active_nodes, feature_tiles);
        let histogram_execution = if parallelize_nodes {
            HistogramExecution::Sequential
        } else {
            HistogramExecution::Parallel
        };
        let context = LevelProposalContext {
            backend,
            binned_matrix,
            gradients,
            round_index,
            depth,
            feature_tiles,
            split_options,
            params,
            controls,
            feature_weights,
            categorical_features,
            morph,
            raw_feature_values,
            feature_scaler: &feature_scaler,
            factor_exposures,
            constraint_index: constraint_index.as_ref(),
            colsample_bynode,
            histogram_execution,
        };
        let work_items = active_nodes
            .into_iter()
            .map(|active_node| {
                let active_groups = node_active_groups.get(&active_node.local_node_id).copied();
                (active_node, active_groups)
            })
            .collect::<Vec<_>>();
        let outcomes = if parallelize_nodes {
            work_items
                .into_par_iter()
                .map(|(active_node, active_groups)| {
                    propose_level_node(&context, active_node, active_groups)
                })
                .collect::<Vec<_>>()
        } else {
            work_items
                .into_iter()
                .map(|(active_node, active_groups)| {
                    propose_level_node(&context, active_node, active_groups)
                })
                .collect::<Vec<_>>()
        };
        let mut outcomes = outcomes
            .into_iter()
            .collect::<EngineResult<Vec<LevelNodeOutcome>>>()?;
        sort_level_outcomes(&mut outcomes);

        let mut next_nodes = Vec::new();
        for outcome in outcomes {
            if let Some(reason) = outcome.rejection_reason {
                round_rejection_reason = reason;
                continue;
            }
            let Some(mut proposal) = outcome.proposal else {
                continue;
            };

            // Admission and all externally visible mutation remain ordered by
            // local node id, regardless of proposal completion order.
            if let Some(max_leaves) = controls.max_leaves {
                let leaves_after_split = candidate_round_stumps.len() + 2;
                if leaves_after_split > max_leaves {
                    round_rejection_reason = IterationStopReason::MaxLeavesReached;
                    continue;
                }
            }

            if let Some((ref left_leaf, ref right_leaf)) = proposal.linear_leaf_pair {
                let feature_count = binned_matrix.feature_count;
                for &row in &proposal.partition.left_row_indices {
                    let row = row as usize;
                    if row < candidate_predictions.len() {
                        candidate_predictions[row] +=
                            left_leaf.eval(raw_feature_values, row * feature_count);
                    }
                }
                for &row in &proposal.partition.right_row_indices {
                    let row = row as usize;
                    if row < candidate_predictions.len() {
                        candidate_predictions[row] +=
                            right_leaf.eval(raw_feature_values, row * feature_count);
                    }
                }
            } else {
                apply_partition_leaf_updates(
                    candidate_predictions,
                    &proposal.partition,
                    proposal.left_leaf_value,
                    proposal.right_leaf_value,
                )?;
            }

            if let Some(children) = proposal.children.take() {
                let PartitionResult {
                    left_row_indices,
                    right_row_indices,
                } = proposal.partition;
                let left_child = ActiveNodeEntry {
                    local_node_id: children.left_local_node_id,
                    row_indices: left_row_indices,
                    histograms: children.left_histograms,
                    parent_leaf_value: children.left_parent_value,
                    parent_linear_leaf: children.left_parent_linear,
                    path_features: children.path_features.clone(),
                    monotone_bounds: children.left_bounds,
                };
                let right_child = ActiveNodeEntry {
                    local_node_id: children.right_local_node_id,
                    row_indices: right_row_indices,
                    histograms: children.right_histograms,
                    parent_leaf_value: children.right_parent_value,
                    parent_linear_leaf: children.right_parent_linear,
                    path_features: children.path_features,
                    monotone_bounds: children.right_bounds,
                };
                if let (Some(index), Some(active_groups)) =
                    (constraint_index.as_ref(), proposal.node_active_groups)
                {
                    let child_groups = index.descend(active_groups, proposal.split.feature_index);
                    node_active_groups.insert(left_child.local_node_id, child_groups);
                    node_active_groups.insert(right_child.local_node_id, child_groups);
                }
                next_nodes.push(left_child);
                next_nodes.push(right_child);
            }

            let (left_leaf_value, right_leaf_value) =
                if let Some((left_leaf, right_leaf)) = proposal.linear_leaf_pair {
                    (LeafValue::Linear(left_leaf), LeafValue::Linear(right_leaf))
                } else {
                    (
                        LeafValue::Scalar(proposal.left_leaf_value),
                        LeafValue::Scalar(proposal.right_leaf_value),
                    )
                };
            candidate_round_stumps.push(TrainedStump {
                split: proposal.split,
                left_leaf_value,
                right_leaf_value,
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
    path_features: Vec<u32>,
    split_candidate: SplitCandidate,
    prepared_linear_leaf_pair: Option<(LinearLeaf, LinearLeaf)>,
    histograms: HistogramBundle,
    parent_leaf_value: f32,
    /// Absolute linear leaf of the parent (used to compute weight deltas for linear-leaf trees).
    parent_linear_leaf: Option<LinearLeaf>,
    depth: usize,
    monotone_bounds: MonotoneBounds,
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
    colsample_bynode: Option<ColsampleBynode>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let max_leaves = controls.max_leaves.unwrap_or(usize::MAX);
    let max_depth = params.max_depth as usize;
    let feature_scaler = LinearFeatureScaler::from_raw_matrix(
        raw_feature_values,
        binned_matrix.row_count,
        binned_matrix.feature_count,
    );

    // Build root histograms and find best split.
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms = backend.build_histograms_with_grad_sq(
        binned_matrix,
        gradients,
        &root_node,
        feature_tiles,
        split_options.requires_grad_sq(),
    )?;
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
    let root_filtered_storage = filter_histograms_for_node(
        &root_histograms,
        constraint_index.as_ref(),
        node_active_groups.get(&0).copied(),
        colsample_bynode,
        root_node_id as u64,
    );
    let root_histograms_for_split = root_filtered_storage.as_ref().unwrap_or(&root_histograms);
    let root_selection = select_node_split(
        backend,
        binned_matrix,
        gradients,
        &root_node,
        root_histograms_for_split,
        &[],
        split_options,
        params,
        feature_weights,
        categorical_features,
        morph.as_ref(),
        root_factor_context.as_ref(),
        raw_feature_values,
        &feature_scaler,
        0.0,
        None,
        controls.max_abs_leaf_value,
    )?;

    let Some(SelectedNodeSplit {
        split: root_split,
        prepared_linear_leaf_pair: root_prepared_linear_leaf_pair,
    }) = root_selection
    else {
        return Ok((Vec::new(), IterationStopReason::NoSplitCandidate));
    };
    if !root_split.gain.is_finite() || root_split.gain <= controls.min_split_gain {
        return Ok((Vec::new(), IterationStopReason::GainBelowThreshold));
    }

    let mut queue = BinaryHeap::new();
    queue.push(PendingSplit {
        local_node_id: 0,
        row_indices: root_node.row_indices,
        path_features: Vec::new(),
        split_candidate: root_split,
        prepared_linear_leaf_pair: root_prepared_linear_leaf_pair,
        histograms: root_histograms,
        parent_leaf_value: 0.0,
        parent_linear_leaf: None,
        depth: 0,
        monotone_bounds: MonotoneBounds::root(controls.max_abs_leaf_value)?,
    });

    // Start with 1 leaf (the root). Each split adds 1 net leaf (splits one into two).
    let mut leaves_used = 1_usize;
    let mut stumps = Vec::new();
    let mut last_rejection = IterationStopReason::NoSplitCandidate;

    while let Some(pending) = queue.pop() {
        let PendingSplit {
            local_node_id,
            row_indices,
            path_features,
            split_candidate: split,
            prepared_linear_leaf_pair,
            mut histograms,
            parent_leaf_value,
            parent_linear_leaf,
            depth,
            monotone_bounds,
        } = pending;

        // Check max_leaves: splitting adds 1 net leaf.
        if leaves_used + 1 > max_leaves {
            last_rejection = IterationStopReason::MaxLeavesReached;
            break;
        }

        // Check max_depth constraint.
        if depth >= max_depth {
            last_rejection = IterationStopReason::DepthBudgetReached;
            continue;
        }

        let node_id = encode_tree_node_id(round_index, local_node_id)?;
        let node = NodeSlice::new(node_id, row_indices)?;
        let parent_row_count = node.row_indices.len();

        // Apply the split: partition rows and get stats.
        let (partition, left_stats, right_stats) =
            backend.apply_split_owned_with_stats(binned_matrix, gradients, node, &split)?;

        if partition.left_row_indices.len() + partition.right_row_indices.len() != parent_row_count
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
        let child_depth = (depth + 1) as u32;
        let scheduled_lr = morph.map_or(params.learning_rate, |m| m.scheduled_lr());
        let leaf_scale = morph.map_or(params.learning_rate, |m| {
            m.leaf_scale_for_depth(child_depth).total
        });
        let raw_left_leaf_value = -leaf_scale * left_grad
            / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
        let raw_right_leaf_value = -leaf_scale * right_grad
            / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

        // Morph leaf modifications: depth penalty + per-round shrinkage.
        // Children land at `depth + 1` in the tree.
        let morph_scale = if let Some(m) = morph.as_ref() {
            m.leaf_scale_for_depth(child_depth).multiplier
        } else {
            1.0
        };

        let raw_left_leaf_absolute =
            raw_left_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let raw_right_leaf_absolute =
            raw_right_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let monotone_constraints_active =
            has_active_monotone_constraints(&params.monotone_constraints);
        if monotone_constraints_active && params.leaf_model == LeafModelKind::Linear {
            return Err(EngineError::InvalidConfig(
                "active monotone constraints do not support linear leaves during tree growth"
                    .to_string(),
            ));
        }
        let (
            left_leaf_value,
            right_leaf_value,
            left_leaf_absolute,
            right_leaf_absolute,
            left_bounds,
            right_bounds,
        ) = if monotone_constraints_active {
            let BoundedChildren {
                left_output,
                right_output,
                left_bounds,
                right_bounds,
            } = monotone_bounds.bound_children(
                monotone_constraint_for_feature(&params.monotone_constraints, split.feature_index),
                raw_left_leaf_absolute,
                raw_right_leaf_absolute,
            )?;
            let left = reconstruct_bounded_child(
                parent_leaf_value,
                left_output - parent_leaf_value,
                left_output,
                left_bounds,
                local_node_id,
                "left",
            )?;
            let right = reconstruct_bounded_child(
                parent_leaf_value,
                right_output - parent_leaf_value,
                right_output,
                right_bounds,
                local_node_id,
                "right",
            )?;
            (
                left.delta,
                right.delta,
                left.absolute,
                right.absolute,
                left_bounds,
                right_bounds,
            )
        } else {
            (
                raw_left_leaf_absolute - parent_leaf_value,
                raw_right_leaf_absolute - parent_leaf_value,
                raw_left_leaf_absolute,
                raw_right_leaf_absolute,
                monotone_bounds,
                monotone_bounds,
            )
        };

        if left_leaf_value.abs() < controls.min_abs_leaf_value
            && right_leaf_value.abs() < controls.min_abs_leaf_value
        {
            last_rejection = IterationStopReason::LeafMagnitudeBelowThreshold;
            continue;
        }

        // ── Linear leaf path ───────────────────────────────────────────────────
        let linear_leaf_computation_result: Option<LinearLeafQuad> = if params.leaf_model
            == LeafModelKind::Linear
            && !raw_feature_values.is_empty()
            && !split.is_categorical
        {
            let regressor_features = linear_regressor_path_features(
                &path_features,
                split.feature_index,
                split.is_categorical,
                binned_matrix.feature_count,
            );
            prepared_linear_leaf_pair
                .or_else(|| {
                    backend.compute_linear_leaf_pair_from_partitions(
                        binned_matrix,
                        gradients,
                        raw_feature_values,
                        binned_matrix.feature_count,
                        split.feature_index,
                        split.threshold_bin,
                        split.default_left,
                        &regressor_features,
                        &feature_scaler,
                        &partition.left_row_indices,
                        &partition.right_row_indices,
                        scheduled_lr,
                        split_options.l2_lambda,
                        controls.max_abs_leaf_value,
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
                    ll_delta.intercept -= parent_leaf_value;
                    rl_delta.intercept -= parent_leaf_value;
                    if let Some(ref p) = parent_linear_leaf {
                        for (slot, &feature) in ll_delta.regressor_features.iter().enumerate() {
                            if let Some(parent_slot) =
                                p.regressor_features.iter().position(|&f| f == feature)
                                && slot < ll_delta.weights.len()
                                && parent_slot < p.weights.len()
                            {
                                ll_delta.weights[slot] -= p.weights[parent_slot];
                            }
                        }
                        for (slot, &feature) in rl_delta.regressor_features.iter().enumerate() {
                            if let Some(parent_slot) =
                                p.regressor_features.iter().position(|&f| f == feature)
                                && slot < rl_delta.weights.len()
                                && parent_slot < p.weights.len()
                            {
                                rl_delta.weights[slot] -= p.weights[parent_slot];
                            }
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
        let child_depth = depth + 1;
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
                smaller_bounds,
                larger_bounds,
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
                    left_bounds,
                    right_bounds,
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
                    right_bounds,
                    left_bounds,
                )
            };

            let smaller_node = NodeSlice::new(smaller_node_id, smaller_indices)?;
            let smaller_histograms = backend.build_histograms_with_grad_sq(
                binned_matrix,
                gradients,
                &smaller_node,
                feature_tiles,
                split_options.requires_grad_sq(),
            )?;
            histograms.subtract_child_in_place(&smaller_histograms, larger_node_id)?;
            let larger_histograms = histograms;

            // Propagate interaction-constraint active groups to both
            // children of the just-applied split.  Both children inherit the
            // same descended bitset because the split feature is shared.
            // (`split` itself was moved into `committed_split` above; we
            // read the feature index off the just-pushed stump instead.)
            let (split_feature_for_descend, split_is_categorical_for_descend) = stumps
                .last()
                .map(|s| (s.split.feature_index, s.split.is_categorical))
                .unwrap_or((0, false));
            let child_path_features = linear_regressor_path_features(
                &path_features,
                split_feature_for_descend,
                split_is_categorical_for_descend,
                binned_matrix.feature_count,
            );
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
            let smaller_filtered_storage = filter_histograms_for_node(
                &smaller_histograms,
                constraint_index.as_ref(),
                child_active_groups,
                colsample_bynode,
                smaller_node_id as u64,
            );
            let smaller_histograms_for_split = smaller_filtered_storage
                .as_ref()
                .unwrap_or(&smaller_histograms);
            if let Some(SelectedNodeSplit {
                split: child_split,
                prepared_linear_leaf_pair: child_prepared_linear_leaf_pair,
            }) = select_node_split(
                backend,
                binned_matrix,
                gradients,
                &smaller_node,
                smaller_histograms_for_split,
                &child_path_features,
                split_options,
                params,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                smaller_factor_context.as_ref(),
                raw_feature_values,
                &feature_scaler,
                smaller_parent_val,
                smaller_parent_ll.as_ref(),
                controls.max_abs_leaf_value,
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: smaller_local,
                    row_indices: smaller_node.row_indices,
                    path_features: child_path_features.clone(),
                    split_candidate: child_split,
                    prepared_linear_leaf_pair: child_prepared_linear_leaf_pair,
                    histograms: smaller_histograms,
                    parent_leaf_value: smaller_parent_val,
                    parent_linear_leaf: smaller_parent_ll,
                    depth: child_depth,
                    monotone_bounds: smaller_bounds,
                });
            }

            let larger_node = NodeSlice::new(larger_node_id, larger_indices)?;
            let larger_factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &larger_node.row_indices,
            );
            let larger_filtered_storage = filter_histograms_for_node(
                &larger_histograms,
                constraint_index.as_ref(),
                child_active_groups,
                colsample_bynode,
                larger_node_id as u64,
            );
            let larger_histograms_for_split = larger_filtered_storage
                .as_ref()
                .unwrap_or(&larger_histograms);
            if let Some(SelectedNodeSplit {
                split: child_split,
                prepared_linear_leaf_pair: child_prepared_linear_leaf_pair,
            }) = select_node_split(
                backend,
                binned_matrix,
                gradients,
                &larger_node,
                larger_histograms_for_split,
                &child_path_features,
                split_options,
                params,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                larger_factor_context.as_ref(),
                raw_feature_values,
                &feature_scaler,
                larger_parent_val,
                larger_parent_ll.as_ref(),
                controls.max_abs_leaf_value,
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: larger_local,
                    row_indices: larger_node.row_indices,
                    path_features: child_path_features.clone(),
                    split_candidate: child_split,
                    prepared_linear_leaf_pair: child_prepared_linear_leaf_pair,
                    histograms: larger_histograms,
                    parent_leaf_value: larger_parent_val,
                    parent_linear_leaf: larger_parent_ll,
                    depth: child_depth,
                    monotone_bounds: larger_bounds,
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
#[cfg(test)]
pub(crate) fn subtract_histogram_bundle_into(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
    dest: &mut HistogramBundle,
) -> EngineResult<()> {
    if parent.feature_count() != child.feature_count() {
        return Err(EngineError::ContractViolation(format!(
            "parent histogram feature count {} does not match child histogram feature count {}",
            parent.feature_count(),
            child.feature_count()
        )));
    }
    dest.subtract_into(parent, child, node_id)
        .map_err(EngineError::from)
}

#[cfg(test)]
pub(crate) fn subtract_histogram_bundle(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
) -> EngineResult<HistogramBundle> {
    // Pre-allocate a dest with the same structure, then delegate to the in-place variant.
    let mut dest = HistogramBundle::new_zeroed_with_grad_sq(
        parent.feature_indices(),
        parent.bin_count(),
        parent.has_grad_sq_sums(),
    );
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

#[cfg(test)]
mod linear_leaf_path_tests {
    use super::*;
    use crate::split_options::SplitShortlist;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    struct ShortlistRecordingBackend {
        shortlist_calls: AtomicUsize,
        evaluation_calls: AtomicUsize,
        standard_split: SplitCandidate,
        candidates: Vec<SplitCandidate>,
    }

    impl BackendOps for ShortlistRecordingBackend {
        fn build_histograms(
            &self,
            _binned_matrix: &BinnedMatrix,
            _gradients: &[GradientPair],
            _node: &NodeSlice,
            _feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            unreachable!("selector tests provide histograms directly")
        }

        fn best_split(
            &self,
            _histograms: &HistogramBundle,
        ) -> EngineResult<Option<SplitCandidate>> {
            Ok(Some(self.standard_split.clone()))
        }

        fn shortlist_standard_splits(
            &self,
            _histograms: &HistogramBundle,
            _options: SplitSelectionOptions,
            _feature_weights: &[f32],
            _categorical_features: &[CategoricalFeatureInfo],
            max_numeric_features: usize,
        ) -> EngineResult<SplitShortlist> {
            self.shortlist_calls.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(SplitShortlist {
                best_overall: Some(self.standard_split.clone()),
                numeric_candidates: self
                    .candidates
                    .iter()
                    .take(max_numeric_features)
                    .cloned()
                    .collect(),
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn evaluate_shortlisted_linear_feature(
            &self,
            _binned_matrix: &BinnedMatrix,
            _gradients: &[GradientPair],
            _node: &NodeSlice,
            split_feature_index: u32,
            linear_context: &LinearContext,
            _feature_scaler: &LinearFeatureScaler,
            _raw_feature_values: &[f32],
            _row_count: usize,
            _feature_count: usize,
            _options: SplitSelectionOptions,
            _learning_rate: f32,
            _parent_leaf_value: f32,
            _parent_linear_leaf: Option<&LinearLeaf>,
        ) -> EngineResult<Option<PreparedLinearSplit>> {
            self.evaluation_calls.fetch_add(1, AtomicOrdering::Relaxed);
            let candidate = self
                .candidates
                .iter()
                .find(|candidate| candidate.feature_index == split_feature_index)
                .expect("shortlisted feature is recorded")
                .clone();
            let leaf = LinearLeaf::identity_scaled(
                split_feature_index as f32,
                vec![1.0; linear_context.regressor_features.len()],
                linear_context.regressor_features.clone(),
            );
            Ok(Some(PreparedLinearSplit {
                split: candidate,
                left_leaf: leaf.clone(),
                right_leaf: leaf,
            }))
        }

        fn apply_split(
            &self,
            _binned_matrix: &BinnedMatrix,
            _node: &NodeSlice,
            _split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            unreachable!("selector tests do not partition")
        }

        fn reduce_sums(
            &self,
            _gradients: &[GradientPair],
            _row_indices: &[u32],
        ) -> EngineResult<alloygbm_core::NodeStats> {
            unreachable!("selector tests do not reduce")
        }
    }

    fn selector_candidate(feature_index: u32, gain: f32, categorical: bool) -> SplitCandidate {
        SplitCandidate {
            node_id: 0,
            feature_index,
            threshold_bin: 0,
            gain,
            default_left: false,
            is_categorical: categorical,
            categorical_bitset: categorical.then(|| vec![1]),
            left_stats: alloygbm_core::NodeStats {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                row_count: 1,
            },
            right_stats: alloygbm_core::NodeStats {
                grad_sum: 1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                row_count: 1,
            },
        }
    }

    fn run_selector(
        backend: &ShortlistRecordingBackend,
        params: &TrainParams,
        feature_weights: &[f32],
    ) -> SelectedNodeSplit {
        let binned_matrix =
            BinnedMatrix::new(2, 3, 1, vec![0, 0, 0, 1, 1, 1]).expect("selector matrix is valid");
        let gradients = vec![
            GradientPair::new(-1.0, 1.0).expect("gradient is valid"),
            GradientPair::new(1.0, 1.0).expect("gradient is valid"),
        ];
        let node = NodeSlice::new(0, vec![0, 1]).expect("selector node is valid");
        let histograms =
            HistogramBundle::from_soa(0, vec![0], 1, vec![0.0], vec![2.0], None, vec![2])
                .expect("selector histogram is valid");
        let options = SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_rows_per_leaf: 1,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: 255,
        };
        select_node_split(
            backend,
            &binned_matrix,
            &gradients,
            &node,
            &histograms,
            &[],
            options,
            params,
            feature_weights,
            &[],
            None,
            None,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &LinearFeatureScaler::identity(3),
            0.0,
            None,
            f32::INFINITY,
        )
        .expect("selector succeeds")
        .expect("selector returns a split")
    }

    #[test]
    fn linear_regressor_features_follow_split_path_not_first_columns() {
        let selected = linear_regressor_path_features(&[10, 3, 10], 12, false, 20);
        assert_eq!(selected, vec![10, 3, 12]);
    }

    #[test]
    fn linear_regressor_features_cap_at_max_pl_regressors() {
        let path = vec![9, 8, 7, 6, 5, 4, 3, 2];
        let selected = linear_regressor_path_features(&path, 1, false, 20);
        assert_eq!(selected, vec![9, 8, 7, 6, 5, 4, 3, 2]);
    }

    #[test]
    fn linear_regressor_path_skips_categorical_split_features() {
        let selected = linear_regressor_path_features(&[10, 3], 12, true, 20);
        assert_eq!(selected, vec![10, 3]);
    }

    #[test]
    fn linear_selector_rescores_only_the_requested_shortlist_and_reuses_leaf_solves() {
        let backend = ShortlistRecordingBackend {
            shortlist_calls: AtomicUsize::new(0),
            evaluation_calls: AtomicUsize::new(0),
            standard_split: selector_candidate(0, 10.0, false),
            candidates: vec![
                selector_candidate(0, 10.0, false),
                selector_candidate(1, 8.0, false),
                selector_candidate(2, 7.0, false),
            ],
        };
        let params = TrainParams {
            leaf_model: LeafModelKind::Linear,
            pl_split_candidates: 2,
            ..TrainParams::default()
        };

        let selected = run_selector(&backend, &params, &[0.5, 1.0, 1.0]);

        assert_eq!(backend.shortlist_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(backend.evaluation_calls.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(selected.split.feature_index, 1);
        let (left, right) = selected
            .prepared_linear_leaf_pair
            .expect("winning PL solve is retained");
        assert_eq!(left.intercept, 1.0);
        assert_eq!(right.intercept, 1.0);
    }

    #[test]
    fn zero_candidate_limit_uses_the_legacy_dispatcher_exactly() {
        let backend = ShortlistRecordingBackend {
            shortlist_calls: AtomicUsize::new(0),
            evaluation_calls: AtomicUsize::new(0),
            standard_split: selector_candidate(2, 4.0, false),
            candidates: vec![selector_candidate(0, 10.0, false)],
        };
        let params = TrainParams {
            leaf_model: LeafModelKind::Linear,
            pl_split_candidates: 0,
            ..TrainParams::default()
        };

        let selected = run_selector(&backend, &params, &[]);

        assert_eq!(selected.split.feature_index, 2);
        assert!(selected.prepared_linear_leaf_pair.is_none());
        assert_eq!(backend.shortlist_calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(backend.evaluation_calls.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn categorical_standard_winner_bypasses_linear_rescoring() {
        let backend = ShortlistRecordingBackend {
            shortlist_calls: AtomicUsize::new(0),
            evaluation_calls: AtomicUsize::new(0),
            standard_split: selector_candidate(2, 12.0, true),
            candidates: vec![selector_candidate(0, 10.0, false)],
        };
        let params = TrainParams {
            leaf_model: LeafModelKind::Linear,
            pl_split_candidates: 8,
            ..TrainParams::default()
        };

        let selected = run_selector(&backend, &params, &[]);

        assert!(selected.split.is_categorical);
        assert!(selected.prepared_linear_leaf_pair.is_none());
        assert_eq!(backend.shortlist_calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(backend.evaluation_calls.load(AtomicOrdering::Relaxed), 0);
    }

    #[test]
    fn level_node_outcomes_commit_in_local_node_order() {
        let mut outcomes = vec![
            LevelNodeOutcome::no_split(6),
            LevelNodeOutcome::no_split(2),
            LevelNodeOutcome::no_split(5),
            LevelNodeOutcome::no_split(1),
        ];

        sort_level_outcomes(&mut outcomes);

        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.local_node_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 5, 6]
        );
    }
}
