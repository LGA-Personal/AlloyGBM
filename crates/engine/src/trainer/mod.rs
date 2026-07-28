//! Trainer module — gradient-boosting iteration controller.

mod interaction;
#[allow(dead_code)]
mod monotone;
mod policy;
mod tree_build;
mod validate;

pub(crate) use interaction::InteractionConstraintIndex;
use monotone::{
    has_active_monotone_constraints, project_monotone_forest, project_monotone_tree,
    validate_monotone_forest,
};
#[cfg(test)]
pub(crate) use policy::{
    AUTO_SPLIT_L2_NOISY_SMALL_WIDE, should_apply_auto_split_l2,
    split_selection_options_for_training,
};
pub(crate) use policy::{
    resolve_training_policy, split_selection_options_with_resolution_for_training,
};
pub(crate) use tree_build::{
    LEAF_EPSILON, apply_single_categorical_target_encoding, build_tree_leaf_wise,
    build_tree_level_wise, validate_iteration_controls,
};
#[cfg(test)]
pub(crate) use tree_build::{subtract_histogram_bundle, subtract_histogram_bundle_into};
pub(crate) use validate::{
    binned_feature_density, compute_feature_means_from_matrix, factor_split_context_for_node,
    gradient_neutralization_config, prepare_pre_target_training_dataset, target_variance,
    validate_gradient_pair_length, validate_gradient_pairs, validate_neutralization_fit_contract,
    validate_neutralization_fit_contract_for_support, validate_partition_cover,
    validate_training_alignment, validate_warm_start_neutralization_contract,
};

// The Trainer impl uses many crate-level types and pub(crate) helpers; rather than
// enumerate ~50 imports here, we use a glob import. Tightening this is left to a
// future task.
use crate::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Trainer {
    params: TrainParams,
    categorical_features: Vec<CategoricalFeatureInfo>,
}

type MulticlassDartRoundBookkeeping = (Vec<Vec<usize>>, Vec<Vec<usize>>);

pub(crate) fn allocate_dart_contribution_buffer(
    dart_enabled: bool,
    row_count: usize,
) -> Option<Vec<f32>> {
    dart_enabled.then(|| vec![0.0; row_count])
}

pub(crate) fn dense_projection_stump_counts(
    stumps: &[TrainedStump],
    logical_round_count: usize,
) -> EngineResult<Vec<usize>> {
    let mut counts = vec![0_usize; logical_round_count];
    let mut previous_tree_id = None;

    for stump in stumps {
        let tree_id = decode_tree_node_id(stump.split.node_id).0 as usize;
        if let Some(previous_tree_id) = previous_tree_id
            && tree_id < previous_tree_id
        {
            return Err(EngineError::ContractViolation(format!(
                "projection stump order moved backward from tree id {previous_tree_id} to tree id {tree_id}"
            )));
        }
        let count = counts.get_mut(tree_id).ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "projection tree id {tree_id} is outside final logical round count {logical_round_count}"
            ))
        })?;
        *count = count.checked_add(1).ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "projection stump count overflow for tree id {tree_id}"
            ))
        })?;
        previous_tree_id = Some(tree_id);
    }

    let covered_stumps = counts.iter().try_fold(0_usize, |total, &count| {
        total.checked_add(count).ok_or_else(|| {
            EngineError::ContractViolation("projection stump count overflow".to_string())
        })
    })?;
    if covered_stumps != stumps.len() {
        return Err(EngineError::ContractViolation(format!(
            "projection round counts cover {covered_stumps} stumps, expected {}",
            stumps.len()
        )));
    }

    Ok(counts)
}

/// Rebuilds the per-class DART slice map from persisted multiclass stumps.
///
/// Warm-start state can contain skipped rounds, so the map is indexed by the
/// encoded tree id rather than the number of tree groups encountered. A zero
/// count is a phantom logical round and intentionally has no stump slice.
pub(crate) fn reconstruct_multiclass_dart_round_bookkeeping(
    class_stumps: &[Vec<TrainedStump>],
    round_count: usize,
) -> EngineResult<MulticlassDartRoundBookkeeping> {
    let mut round_start_offsets = Vec::with_capacity(class_stumps.len());
    let mut round_counts = Vec::with_capacity(class_stumps.len());

    for (class_k, stumps) in class_stumps.iter().enumerate() {
        let mut starts = vec![stumps.len(); round_count];
        let mut counts = vec![0_usize; round_count];
        let mut start = 0_usize;
        let mut previous_tree_id = None;
        while start < stumps.len() {
            let (tree_id, _) = decode_tree_node_id(stumps[start].split.node_id);
            let tree_id = tree_id as usize;
            if tree_id >= round_count {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start class {class_k} tree_id {tree_id} is outside initial_rounds_completed {round_count}"
                )));
            }
            if previous_tree_id.is_some_and(|previous| tree_id <= previous) {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start class {class_k} tree ids must be strictly increasing; found {tree_id} after {}",
                    previous_tree_id.expect("checked above"),
                )));
            }
            let mut end = start + 1;
            while end < stumps.len() {
                let (next_tree_id, _) = decode_tree_node_id(stumps[end].split.node_id);
                if next_tree_id as usize != tree_id {
                    break;
                }
                end += 1;
            }
            starts[tree_id] = start;
            counts[tree_id] = end - start;
            previous_tree_id = Some(tree_id);
            start = end;
        }
        round_start_offsets.push(starts);
        round_counts.push(counts);
    }

    Ok((round_start_offsets, round_counts))
}

pub(crate) fn append_multiclass_dart_phantom_round(
    dart_state: &mut DartState,
    round_start_offsets: &mut [Vec<usize>],
    round_counts: &mut [Vec<usize>],
    class_stumps: &[Vec<TrainedStump>],
) {
    for ((starts, counts), stumps) in round_start_offsets
        .iter_mut()
        .zip(round_counts.iter_mut())
        .zip(class_stumps)
    {
        starts.push(stumps.len());
        counts.push(0);
    }
    dart_state
        .tree_weights
        .extend(std::iter::repeat_n(1.0, class_stumps.len()));
    dart_state.dropped_per_round.push(Vec::new());
}

pub(crate) fn multiclass_dart_material_classes(
    dropped_tree_indices: &[usize],
    class_count: usize,
    round_counts: &[Vec<usize>],
) -> EngineResult<Vec<usize>> {
    if class_count == 0 || round_counts.len() != class_count {
        return Err(EngineError::ContractViolation(
            "multiclass DART material-class lookup received inconsistent class counts".to_string(),
        ));
    }

    let mut material_classes = Vec::with_capacity(dropped_tree_indices.len().min(class_count));
    for &flat_tree_id in dropped_tree_indices {
        let prior_round = flat_tree_id / class_count;
        let class_k = flat_tree_id % class_count;
        let stump_count = round_counts[class_k].get(prior_round).copied().unwrap_or(0);
        if stump_count > 0 && !material_classes.contains(&class_k) {
            material_classes.push(class_k);
        }
    }
    Ok(material_classes)
}

pub(crate) fn clear_multiclass_dart_contributions(
    contributions: &mut [Vec<f32>],
    material_classes: &[usize],
) -> EngineResult<()> {
    if let Some(&class_k) = material_classes
        .iter()
        .find(|&&class_k| class_k >= contributions.len())
    {
        return Err(EngineError::ContractViolation(format!(
            "multiclass DART material class {class_k} is outside contribution count {}",
            contributions.len(),
        )));
    }
    for &class_k in material_classes {
        contributions[class_k].fill(0.0);
    }
    Ok(())
}

pub(crate) fn apply_multiclass_dart_contributions(
    predictions: &mut [Vec<f32>],
    contributions: &[Vec<f32>],
    material_classes: &[usize],
    factor: f32,
) -> EngineResult<()> {
    if predictions.len() != contributions.len() {
        return Err(EngineError::ContractViolation(format!(
            "multiclass DART prediction count {} does not match contribution count {}",
            predictions.len(),
            contributions.len(),
        )));
    }
    for &class_k in material_classes {
        let Some(prediction) = predictions.get(class_k) else {
            return Err(EngineError::ContractViolation(format!(
                "multiclass DART material class {class_k} is outside prediction count {}",
                predictions.len(),
            )));
        };
        if prediction.len() != contributions[class_k].len() {
            return Err(EngineError::ContractViolation(format!(
                "multiclass DART class {class_k} prediction length {} does not match contribution length {}",
                prediction.len(),
                contributions[class_k].len(),
            )));
        }
    }
    for &class_k in material_classes {
        apply_scaled_prediction_buffer(&mut predictions[class_k], &contributions[class_k], factor)?;
    }
    Ok(())
}

/// Records a skipped multiclass warmup iteration in the same logical-round
/// coordinate space as accepted rounds. Early-stop truncation indexes these
/// vectors by `rounds_completed`, so a phantom slot must be present everywhere
/// rather than only in DART's internal flat-tree state.
#[allow(clippy::too_many_arguments)]
fn append_multiclass_phantom_round_bookkeeping(
    stumps_per_round_per_class: &mut Vec<Vec<usize>>,
    loss_per_completed_round: &mut Vec<f32>,
    validation_loss_per_completed_round: &mut Vec<f32>,
    sampled_rows_per_completed_round: &mut Vec<usize>,
    sampled_features_per_completed_round: &mut Vec<usize>,
    diagnostics_per_round: &mut Vec<IterationDiagnostics>,
    class_count: usize,
    current_loss: f32,
    current_validation_loss: Option<f32>,
    sampled_row_count: usize,
    sampled_feature_count: usize,
    per_class_diagnostics: &[IterationDiagnostics],
) {
    stumps_per_round_per_class.push(vec![0; class_count]);
    loss_per_completed_round.push(current_loss);
    if let Some(validation_loss) = current_validation_loss {
        validation_loss_per_completed_round.push(validation_loss);
    }
    sampled_rows_per_completed_round.push(sampled_row_count);
    sampled_features_per_completed_round.push(sampled_feature_count);
    diagnostics_per_round.push(IterationDiagnostics::aggregate_per_class(
        per_class_diagnostics,
    ));
}

pub(crate) fn replay_multiclass_dart_tree_weights(
    initial_weights: &[f32],
    kept_dropped_per_round: &[Vec<usize>],
    class_count: usize,
    normalize_type: alloygbm_core::DartNormalize,
) -> EngineResult<Vec<f32>> {
    if class_count == 0 || !initial_weights.len().is_multiple_of(class_count) {
        return Err(EngineError::ContractViolation(
            "multiclass DART replay received a non-round-major warm-start weight prefix"
                .to_string(),
        ));
    }
    let initial_round_count = initial_weights.len() / class_count;
    if kept_dropped_per_round.len() < initial_round_count {
        return Err(EngineError::ContractViolation(
            "multiclass DART replay history is shorter than its warm-start prefix".to_string(),
        ));
    }

    let mut weights = initial_weights.to_vec();
    for dropped in &kept_dropped_per_round[initial_round_count..] {
        let dropped_count = dropped.len() as f32;
        let new_weight = 1.0 / (dropped_count + 1.0);
        let dropped_factor = match normalize_type {
            alloygbm_core::DartNormalize::Tree => dropped_count / (dropped_count + 1.0),
            alloygbm_core::DartNormalize::Forest => 1.0 / (dropped_count + 1.0),
        };
        let retained_weight_count = weights.len();
        for &flat_tree_id in dropped {
            let weight = weights.get_mut(flat_tree_id).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "multiclass DART replay dropout index {flat_tree_id} is outside the retained prefix {}",
                    retained_weight_count,
                ))
            })?;
            *weight *= dropped_factor;
        }
        weights.extend(std::iter::repeat_n(new_weight, class_count));
    }
    Ok(weights)
}

impl Trainer {
    pub fn new(params: TrainParams) -> EngineResult<Self> {
        validate_train_params(&params)?;
        Ok(Self {
            params,
            categorical_features: Vec::new(),
        })
    }

    /// Set the categorical feature metadata for native categorical splits.
    pub fn with_categorical_features(mut self, features: Vec<CategoricalFeatureInfo>) -> Self {
        self.categorical_features = features;
        self
    }

    pub fn params(&self) -> &TrainParams {
        &self.params
    }

    pub fn validate_fit_contract<O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        objective: &O,
    ) -> EngineResult<FitContractEvaluation> {
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;

        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);

        self.evaluate_fit_contract_on_active_dataset(active_dataset, objective)
    }

    fn evaluate_fit_contract_on_active_dataset<O: ObjectiveOps>(
        &self,
        active_dataset: &TrainingDataset,
        objective: &O,
    ) -> EngineResult<FitContractEvaluation> {
        let baseline_prediction = objective.initial_prediction(
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        if !baseline_prediction.is_finite() {
            return Err(EngineError::ContractViolation(
                "objective returned non-finite initial prediction".to_string(),
            ));
        }

        let predictions = vec![baseline_prediction; active_dataset.row_count()];
        let mut gradients = objective.compute_gradients(
            &predictions,
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        if let Some(config) = gradient_neutralization_config(&self.params) {
            let exposures = active_dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            FactorProjector::new(
                exposures,
                active_dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?
            .project_gradient_pairs_in_place(&mut gradients)?;
        }
        validate_gradient_pairs(&gradients, active_dataset.row_count())?;

        Ok(FitContractEvaluation {
            baseline_prediction,
            gradients,
        })
    }

    pub fn fit_one_round<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
    ) -> EngineResult<TrainRoundSummary> {
        validate_training_alignment(dataset, binned_matrix)?;

        let fit_contract = self.validate_fit_contract(dataset, objective)?;
        let root_row_indices = (0..dataset.row_count() as u32).collect::<Vec<_>>();
        let root_node = NodeSlice::new(0, root_row_indices)?;
        let feature_tiles = vec![FeatureTile::new(0, binned_matrix.feature_count as u32)?];
        let split_options = split_selection_options_from_env()?;

        let histograms = backend.build_histograms_with_grad_sq(
            binned_matrix,
            &fit_contract.gradients,
            &root_node,
            &feature_tiles,
            split_options.requires_grad_sq(),
        )?;
        let factor_context = factor_split_context_for_node(
            &self.params,
            binned_matrix,
            dataset.factor_exposures.as_ref(),
            &root_node.row_indices,
        );
        let split_candidate = backend.best_split_with_factor_context(
            &histograms,
            split_options,
            &self.params.feature_weights,
            &[],
            factor_context.as_ref(),
        )?;
        let root_stats = backend.reduce_sums(&fit_contract.gradients, &root_node.row_indices)?;

        let partition = if let Some(split) = &split_candidate {
            let partition = backend.apply_split(binned_matrix, &root_node, split)?;
            validate_partition_cover(dataset.row_count(), &partition)?;
            Some(partition)
        } else {
            None
        };

        Ok(TrainRoundSummary {
            baseline_prediction: fit_contract.baseline_prediction,
            root_stats,
            split_candidate,
            partition,
        })
    }

    pub fn fit_iterations<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        rounds: usize,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_policy(
            dataset,
            binned_matrix,
            backend,
            objective,
            rounds,
            TrainingPolicyMode::Manual,
            false,
        )
    }

    pub fn fit_iterations_with_single_target_encoded_feature<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        rounds: usize,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_single_target_encoded_feature_and_policy(
            dataset,
            binned_matrix,
            spec,
            backend,
            objective,
            rounds,
            TrainingPolicyMode::Manual,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_policy<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_policy_request(
            dataset,
            binned_matrix,
            backend,
            objective,
            PolicyFitRequest {
                rounds,
                policy_mode,
                store_node_debug_stats,
            },
        )
    }

    fn fit_iterations_with_policy_request<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        validate_training_alignment(dataset, binned_matrix)?;
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        self.fit_iterations_with_policy_request_active(
            active_dataset,
            binned_matrix,
            backend,
            objective,
            request,
        )
    }

    fn fit_iterations_with_policy_request_active<B: BackendOps, O: ObjectiveOps>(
        &self,
        active_dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        let controls = self.iteration_controls_for_policy_ext(
            active_dataset,
            binned_matrix,
            request.rounds,
            request.policy_mode,
            objective.requires_group_id(),
        )?;
        let summary = self.fit_iterations_with_optional_validation_summary(
            active_dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: Some(request.policy_mode),
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: true,
            },
        )?;
        let model = summary.model;
        if request.store_node_debug_stats {
            model.with_node_debug_stats_from_stumps()
        } else {
            Ok(model)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_single_target_encoded_feature_and_policy<
        B: BackendOps,
        O: ObjectiveOps,
    >(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_single_target_encoded_feature_and_policy_request(
            dataset,
            binned_matrix,
            spec,
            backend,
            objective,
            PolicyFitRequest {
                rounds,
                policy_mode,
                store_node_debug_stats,
            },
        )
    }

    fn fit_iterations_with_single_target_encoded_feature_and_policy_request<
        B: BackendOps,
        O: ObjectiveOps,
    >(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        validate_training_alignment(dataset, binned_matrix)?;
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        let (encoded_dataset, encoded_binned_matrix) =
            apply_single_categorical_target_encoding(active_dataset, binned_matrix, spec)?;
        let categorical_state = CategoricalStatePayloadV1 {
            format_version: alloygbm_core::CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: spec.config.time_aware,
            categorical_feature_indices: vec![spec.feature_index as u32],
        };
        let model = self.fit_iterations_with_policy_request_active(
            &encoded_dataset,
            &encoded_binned_matrix,
            backend,
            objective,
            request,
        )?;
        model.with_categorical_state(Some(categorical_state))
    }

    pub fn iteration_controls_for_policy(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
    ) -> EngineResult<IterationControls> {
        self.iteration_controls_for_policy_ext(dataset, binned_matrix, rounds, policy_mode, false)
    }

    /// Extended variant that accepts an `is_ranking` flag so auto-policy
    /// can skip regularization guards that are too aggressive for ranking
    /// objectives (pairwise/LambdaMART/XeNDCG/YetiRank/QueryRMSE).
    pub fn iteration_controls_for_policy_ext(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        is_ranking: bool,
    ) -> EngineResult<IterationControls> {
        let controls = if experiment_force_manual_policy_enabled() {
            self.default_iteration_controls(rounds)?
        } else {
            match policy_mode {
                TrainingPolicyMode::Manual => self.default_iteration_controls(rounds),
                TrainingPolicyMode::Auto => {
                    self.auto_iteration_controls(dataset, binned_matrix, rounds, is_ranking)
                }
            }?
        };
        Ok(controls.with_policy_request(rounds, policy_mode))
    }

    pub fn fit_iterations_with_controls<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<TrainedModel> {
        let summary =
            self.fit_iterations_with_summary(dataset, binned_matrix, backend, objective, controls)?;
        Ok(summary.model)
    }

    pub fn fit_iterations_with_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    pub fn fit_iterations_with_validation_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Continue training from a previously fitted model (warm-start).
    pub fn fit_iterations_warm_start<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Continue training from a previously fitted model with validation (warm-start).
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_warm_start_with_validation<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    // -- Methods that accept a custom metric callback -------------------------

    /// Fit with validation and an optional custom metric callback for early stopping.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_validation_and_metric<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        custom_metric: Option<&dyn PerRoundMetricCallback>,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: custom_metric,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Fit with warm start, validation, and an optional custom metric callback.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_warm_start_with_validation_and_metric<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
        custom_metric: Option<&dyn PerRoundMetricCallback>,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: custom_metric,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    // -- Multi-class training -------------------------------------------------

    pub fn fit_multiclass_iterations_with_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            None,
            backend,
            objective,
            controls,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_with_validation_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            Some(validation),
            backend,
            objective,
            controls,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_warm_start_with_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: MultiClassWarmStartState,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            None,
            backend,
            objective,
            controls,
            Some(warm_start),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_warm_start_with_validation_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: MultiClassWarmStartState,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            Some(validation),
            backend,
            objective,
            controls,
            Some(warm_start),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass_iterations_impl<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: Option<ValidationDatasetRef<'_>>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: Option<MultiClassWarmStartState>,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        let k = objective.num_classes;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
        // v0.10.1: GOSS supported via
        // `select_row_indices_for_round_multiclass` (per-row score
        // `s_i = sum_k |g_{i,k}|`, LightGBM convention). DART supported
        // via per-round dropout/normalize across a flat
        // `round_index * K + class_k` tree pool — see the DART blocks
        // inside the round loop below.
        let dart_params = match self.params.boosting_mode {
            BoostingMode::Dart {
                drop_rate,
                max_drop,
                normalize_type,
                sample_type,
            } => Some((drop_rate, max_drop, normalize_type, sample_type)),
            _ => None,
        };
        // v0.10.2: leaf-wise multiclass DART is now supported. The per-class
        // `dart_round_start_offsets[k]` / `dart_round_counts[k]` bookkeeping
        // is growth-mode-agnostic because it snapshots `class_stumps[k].len()`
        // around each `build_tree_*` call — under leaf-wise growth each tree
        // has a variable stump count (capped by max_leaves), but the round
        // boundaries are still captured correctly.
        validate_train_params(&self.params)?;
        if has_active_monotone_constraints(&self.params.monotone_constraints) {
            return Err(EngineError::InvalidConfig(
                "multiclass is not supported with active monotone_constraints".to_string(),
            ));
        }
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract_for_support(&self.params, dataset, false)?;
        validate_warm_start_neutralization_contract(&self.params, warm_start.is_some(), dataset)?;
        validate_training_alignment(dataset, binned_matrix)?;
        if let Some(validation_ref) = validation {
            validate_training_alignment(validation_ref.dataset, validation_ref.binned_matrix)?;
            if validation_ref.dataset.matrix.feature_count != dataset.matrix.feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "validation feature_count {} does not match training feature_count {}",
                    validation_ref.dataset.matrix.feature_count, dataset.matrix.feature_count
                )));
            }
        }

        // Validate targets are valid class indices
        for (i, &t) in dataset.targets.iter().enumerate() {
            let class = t as usize;
            if class >= k || t < 0.0 || t != t.floor() {
                return Err(EngineError::ContractViolation(format!(
                    "target at index {i} is {t}, expected integer in [0, {k})"
                )));
            }
        }

        let sampling_seed_base = sampling_seed_base(self.params.seed, self.params.deterministic);
        let split_resolution = split_selection_options_with_resolution_for_training(
            &self.params,
            None,
            dataset,
            binned_matrix,
        )?;
        let resolved_training_policy = resolve_training_policy(controls, &split_resolution);
        let split_options = split_resolution.options;
        let feature_count = binned_matrix.feature_count;
        let gradient_projector = if let Some(config) = gradient_neutralization_config(&self.params)
        {
            let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            Some(FactorProjector::new(
                exposures,
                dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?)
        } else {
            None
        };

        // Initialize K prediction arrays — from warm-start or fresh.
        // `warm_ema_stats` captures the optional MorphBoost EMA
        // snapshot for the v0.7.3 warm-start-equivalence fix; consumed
        // below after `MorphState::new` constructs the fresh EMA.
        // `warm_dart_tree_weights` (v0.10.1+) carries the flat
        // round-major × class-k per-tree weights from the prior DART
        // fit; consumed below where `dart_state` is seeded.
        let (
            baselines,
            mut class_stumps,
            round_index_offset,
            initial_stump_counts,
            warm_ema_stats,
            warm_dart_tree_weights,
        ) = if let Some(ws) = warm_start {
            if ws.baseline_predictions.len() != k {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start baseline count {} != num_classes {k}",
                    ws.baseline_predictions.len()
                )));
            }
            if ws.class_stumps.len() != k {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start class_stumps count {} != num_classes {k}",
                    ws.class_stumps.len()
                )));
            }
            let offset = ws.initial_rounds_completed;
            let initial_counts: Vec<usize> = ws.class_stumps.iter().map(|s| s.len()).collect();
            (
                ws.baseline_predictions,
                ws.class_stumps,
                offset,
                initial_counts,
                ws.initial_ema_stats,
                ws.initial_dart_tree_weights,
            )
        } else {
            let baselines = objective.initial_predictions();
            let class_stumps: Vec<Vec<TrainedStump>> = vec![Vec::new(); k];
            (
                baselines,
                class_stumps,
                0_usize,
                vec![0_usize; k],
                None,
                None,
            )
        };

        let n = dataset.row_count();
        let mut class_predictions: Vec<Vec<f32>> = baselines.iter().map(|&b| vec![b; n]).collect();

        // If warm-starting, apply prior-model stumps to prediction arrays
        if round_index_offset > 0 {
            for class_k in 0..k {
                if !class_stumps[class_k].is_empty() {
                    apply_tree_to_binned_predictions(
                        &mut class_predictions[class_k],
                        binned_matrix,
                        &class_stumps[class_k],
                        Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                    )?;
                }
            }
        }
        let mut class_candidate_predictions: Vec<Vec<f32>> = class_predictions.clone();
        // Track stump counts per class at each round boundary for truncation
        let mut stumps_per_round_per_class: Vec<Vec<usize>> = Vec::new();

        // Validation predictions
        let mut validation_class_predictions: Option<Vec<Vec<f32>>> = validation.map(|v| {
            baselines
                .iter()
                .map(|&b| vec![b; v.dataset.row_count()])
                .collect()
        });
        // Apply warm-start stumps to validation predictions too
        if round_index_offset > 0
            && let Some(validation_ref) = validation
            && let Some(val_preds) = validation_class_predictions.as_mut()
        {
            let val_raw = Some((
                &validation_ref.dataset.matrix.values as &[f32],
                validation_ref.dataset.matrix.feature_count,
            ));
            for class_k in 0..k {
                if !class_stumps[class_k].is_empty() {
                    apply_tree_to_binned_predictions(
                        &mut val_preds[class_k],
                        validation_ref.binned_matrix,
                        &class_stumps[class_k],
                        val_raw,
                    )?;
                }
            }
        }

        let mut dart_train_contribution: Option<Vec<Vec<f32>>> = dart_params.is_some().then(|| {
            class_predictions
                .iter()
                .map(|predictions| vec![0.0; predictions.len()])
                .collect()
        });
        let mut dart_validation_contribution: Option<Vec<Vec<f32>>> = if dart_params.is_some() {
            validation_class_predictions.as_ref().map(|predictions| {
                predictions
                    .iter()
                    .map(|output_predictions| vec![0.0; output_predictions.len()])
                    .collect()
            })
        } else {
            None
        };

        let initial_loss = objective.loss(
            &class_predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
        )?;
        let initial_validation_loss = if let Some(v) = validation {
            let val_preds = validation_class_predictions.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "validation predictions were not initialized".to_string(),
                )
            })?;
            Some(objective.loss(
                val_preds,
                &v.dataset.targets,
                v.dataset.sample_weights.as_deref(),
            )?)
        } else {
            None
        };

        let mut current_loss = initial_loss;
        let mut rounds_completed = 0_usize;
        let mut stop_reason = IterationStopReason::CompletedRequestedRounds;
        let mut loss_per_completed_round = Vec::new();
        let mut validation_loss_per_completed_round = Vec::new();
        let mut sampled_rows_per_completed_round = Vec::new();
        let mut sampled_features_per_completed_round = Vec::new();
        let mut diagnostics_per_round: Vec<IterationDiagnostics> = Vec::new();
        let mut best_validation_loss = initial_validation_loss;
        let mut best_validation_round = initial_validation_loss.map(|_| 0_usize);
        let mut validation_no_improvement_rounds = 0_usize;
        let mut weak_improvement_streak = 0_usize;
        let mut weak_improvement_rounds_committed = 0_usize;
        let mut current_validation_loss = initial_validation_loss;
        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(n);

        let effective_round_cap = controls.rounds;

        // Build MorphState (K classes) for the duration of training when
        // morph_config is set. Total iterations spans warm-start prefix + new rounds.
        let total_iterations = (effective_round_cap + round_index_offset) as u32;
        let mut morph_state: Option<MorphState> = self
            .params
            .morph_config
            .map(|cfg| MorphState::new(cfg, k, total_iterations, self.params.learning_rate));

        // v0.7.3 EMA warm-start: when the multiclass warm-start state
        // carries an EMA snapshot from the previous fit, seed the
        // current `MorphState` with it.  Length mismatch (class count
        // changed across fits) silently falls back to the cold EMA
        // from `MorphState::new`.
        if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), warm_ema_stats.as_ref())
            && ms.ema_stats.len() == snapshot.len()
        {
            ms.ema_stats.copy_from_slice(snapshot);
        }

        // v0.10.1: DART state for multiclass. The flat per-tree weight
        // pool is indexed by `round_index * K + class_k` and committed
        // in lockstep with `class_stumps[class_k]` during the round
        // loop. Warm-start seeds the weights from
        // `warm_dart_tree_weights`; historical RNG-driven dropouts are
        // NOT persisted (same as the binary path).
        //
        // Multiclass-specific bookkeeping (mirrors the binary path's
        // `round_start_offsets` / `dart_round_counts` but tracks each
        // class-tree separately because level-wise trees span multiple
        // stumps per (round, class)):
        //
        // * `dart_round_start_offsets[class_k][r]` — starting index in
        //   `class_stumps[class_k]` for class `class_k`'s tree in round
        //   `r`. Length == `effective_round_index + 1` (with phantom
        //   slots for skipped warmup rounds, matching the binary path).
        // * `dart_round_counts[class_k][r]` — number of stumps in class
        //   `class_k`'s tree at round `r`. `0` means no tree committed
        //   that round (e.g. zero-stump class during warmup).
        //
        // The flat dropout index `flat_idx = r * K + class_k` maps to
        // `&class_stumps[class_k][start..start+count]` via these arrays.
        let mut dart_state = DartState::default();
        let mut dart_round_start_offsets: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut dart_round_counts: Vec<Vec<usize>> = vec![Vec::new(); k];
        if dart_params.is_some() {
            let initial_tree_count = round_index_offset * k;
            if let Some(per_tree) = warm_dart_tree_weights.as_ref() {
                if per_tree.len() != initial_tree_count {
                    return Err(EngineError::ContractViolation(format!(
                        "warm-start initial_dart_tree_weights length {} != \
                         initial_rounds_completed * K = {} * {} = {}",
                        per_tree.len(),
                        round_index_offset,
                        k,
                        initial_tree_count,
                    )));
                }
                dart_state.tree_weights = per_tree.clone();
            } else {
                dart_state.tree_weights = vec![1.0; initial_tree_count];
            }
            for _ in 0..round_index_offset {
                dart_state.dropped_per_round.push(Vec::new());
            }
            (dart_round_start_offsets, dart_round_counts) =
                reconstruct_multiclass_dart_round_bookkeeping(&class_stumps, round_index_offset)?;
        }
        // Historical DART dropout selections are intentionally not persisted
        // across warm starts. Preserve the supplied weight prefix so an
        // early-stop replay only replays the new logical rounds.
        let initial_dart_tree_weights = dart_state.tree_weights.clone();

        for round_index in 0..effective_round_cap {
            let effective_round = round_index + round_index_offset;

            // v0.10.1 DART: drop a random subset of previously-committed
            // class-trees BEFORE computing gradients. The flat pool
            // `dart_state.tree_weights` is indexed by
            // `prior_round * K + class_k`. For each dropped flat index,
            // subtract `w_old * tree_contribution` from the
            // corresponding `class_predictions[class_k]` so the new
            // round's gradients are computed on the dropped-out
            // residual.
            //
            // PR review (C4): a level-wise tree spans MULTIPLE stumps,
            // not one stump per (round, class). Use the per-class
            // `dart_round_start_offsets` / `dart_round_counts` arrays
            // (built from tree_id grouping) to subtract the WHOLE class
            // tree's contribution, mirroring the single-output DART
            // path's `apply_weighted_round_to_predictions(&stumps[start..start+count], ...)`.
            //
            // Backups of `class_predictions` are recorded BEFORE
            // mutation so an early-exit (`!any_tree_produced`, loss
            // regression, etc.) can restore the full pre-dropout
            // ensemble — matching the single-output DART semantics
            // (PR review C1).
            let mut dart_predictions_backup: Option<Vec<Vec<f32>>> = None;
            let (dropped_tree_indices, material_dropped_classes): (Vec<usize>, Vec<usize>) =
                if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                    let dart_contribution = dart_train_contribution.as_mut().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "multiclass DART contribution buffer was not initialized".to_string(),
                        )
                    })?;
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round,
                    );
                    let material_classes =
                        multiclass_dart_material_classes(&drops, k, &dart_round_counts)?;
                    clear_multiclass_dart_contributions(dart_contribution, &material_classes)?;
                    if !drops.is_empty() {
                        dart_predictions_backup = Some(class_predictions.clone());
                    }
                    for &flat_idx in &drops {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let w_old = dart_state.tree_weights[flat_idx];
                        // Snapshot the class-tree slice to a Vec so we can
                        // safely re-borrow `class_predictions[class_k]` as
                        // mutable. (Stumps don't change between subtract
                        // and re-add — the slice can be reused later.)
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        apply_weighted_round_to_predictions_and_accumulator(
                            &mut class_predictions[class_k],
                            &mut dart_contribution[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            -w_old,
                            w_old,
                        )?;
                    }
                    (drops, material_classes)
                } else {
                    (Vec::new(), Vec::new())
                };

            // v0.10.1: pre-compute per-class gradient buffers BEFORE sampling
            // so the multiclass GOSS scorer can see all K gradient channels
            // when ranking rows by `s_i = sum_k |g_{i,k}|`. The original
            // gradient norms (for diagnostics) and the projected buffers are
            // both captured up front.
            let mut class_gradient_buffers: Vec<Vec<GradientPair>> = Vec::with_capacity(k);
            let mut class_original_gradient_norms: Vec<Option<f32>> = Vec::with_capacity(k);
            {
                let mut tmp_buf: Vec<GradientPair> = Vec::with_capacity(n);
                let mut projection_scratch: Vec<f32> = Vec::with_capacity(n);
                for class_k in 0..k {
                    objective.compute_gradients_for_class(
                        &class_predictions,
                        &dataset.targets,
                        dataset.sample_weights.as_deref(),
                        class_k,
                        &mut tmp_buf,
                    )?;
                    let original_norm = if gradient_projector.is_some() {
                        Some(gradient_l2_norm_only(&tmp_buf))
                    } else {
                        None
                    };
                    if let Some(projector) = &gradient_projector {
                        projector.project_gradient_pairs_in_place_with_scratch(
                            &mut tmp_buf,
                            &mut projection_scratch,
                        )?;
                    }
                    class_gradient_buffers.push(tmp_buf.clone());
                    class_original_gradient_norms.push(original_norm);
                }
            }

            // Shared row sampling across all K classes. In GOSS mode this
            // amplifies the sampled-low rows in every class buffer.
            let root_row_indices = select_row_indices_for_round_multiclass(
                self.params.boosting_mode,
                n,
                controls.row_subsample,
                sampling_seed_base,
                effective_round as u64,
                &mut class_gradient_buffers,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                feature_count,
                controls.col_subsample,
                sampling_seed_base,
                effective_round as u64,
            )?;
            let sampled_row_count = root_row_indices.len();

            // Copy current predictions to candidates
            for class_k in 0..k {
                class_candidate_predictions[class_k].copy_from_slice(&class_predictions[class_k]);
            }

            // Record stump counts before this round
            let pre_round_counts: Vec<usize> = class_stumps.iter().map(|s| s.len()).collect();

            // Build K trees
            let mut any_tree_produced = false;
            // Per-class diagnostics for this round; aggregated to a single
            // `IterationDiagnostics` after the class loop completes.
            let mut per_class_diagnostics: Vec<IterationDiagnostics> = Vec::with_capacity(k);
            for class_k in 0..k {
                // Use the pre-computed (and possibly GOSS-amplified) buffer.
                gradient_buffer.clear();
                gradient_buffer.extend_from_slice(&class_gradient_buffers[class_k]);
                let original_gradient_norm = class_original_gradient_norms[class_k];
                per_class_diagnostics.push(IterationDiagnostics::from_gradient_snapshot(
                    &gradient_buffer,
                    original_gradient_norm,
                    sampled_row_count,
                    feature_tiles.len(),
                ));

                // Update per-class EMA stats from this class's gradients.
                if let Some(ms) = morph_state.as_mut() {
                    ms.update_ema_from_gradient_pairs(&gradient_buffer, class_k);
                }

                let morph_tree_ctx: Option<MorphTreeContext<'_>> =
                    morph_state.as_ref().map(|ms| MorphTreeContext {
                        state: ms,
                        iteration: effective_round as u32,
                        total_iterations,
                        class_idx: class_k,
                    });

                let mut round_split_options = split_options;
                round_split_options.min_rows_per_leaf = controls.min_rows_per_leaf;
                let raw_fv = &dataset.matrix.values;
                let (round_stumps, _round_stop) = if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        effective_round,
                        &feature_tiles,
                        round_split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        dataset.factor_exposures.as_ref(),
                    )?
                } else {
                    build_tree_level_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        effective_round,
                        &feature_tiles,
                        round_split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        dataset.factor_exposures.as_ref(),
                    )?
                };

                if !round_stumps.is_empty() {
                    any_tree_produced = true;
                }
                class_stumps[class_k].extend(round_stumps);
            }

            for class_k in 0..k {
                let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                class_candidate_predictions[class_k].copy_from_slice(&class_predictions[class_k]);
                // Tree builders update only the sampled partition rows while constructing
                // split statistics. Rebuild the candidate by walking the accepted tree over
                // every training row so the training state matches inference semantics.
                apply_weighted_round_to_predictions(
                    &mut class_candidate_predictions[class_k],
                    binned_matrix,
                    round_stumps,
                    Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                    1.0,
                )?;
            }

            // v0.10.1 DART post-build: rescale the K new trees to
            // `new_w = 1/(num_dropped + 1)` and re-add each dropped
            // tree's contribution at its post-normalize weight to BOTH
            // `class_predictions` and `class_candidate_predictions`.
            //
            // PR review (C1): `dart_state.tree_weights` mutation and
            // per-stump `tree_weight` stamping are DEFERRED to the
            // round-commit branch below. Rejecting the round
            // (`!any_tree_produced`, loss regression, etc.) restores
            // `class_predictions` from `dart_predictions_backup` so
            // the pre-dropout ensemble is preserved for the next round.
            //
            // PR review (C4, C5): use per-class
            // `dart_round_counts[class_k][prior_round]` to re-add the
            // WHOLE dropped class-tree (not just its root); compute
            // `new_dropped_weights` here but commit them only on
            // round acceptance.
            //
            // `dart_round_finalize` carries the per-round normalization
            // bookkeeping into the commit branch; `None` when DART is
            // off or the round had no dropouts (in which case new
            // trees get `tree_weight = 1.0`).
            let dart_round_finalize: Option<(f32, f32, Vec<f32>)> =
                if let Some((_, _, normalize_type, _)) = dart_params {
                    let n_dropped = dropped_tree_indices.len() as f32;
                    let new_w = 1.0 / (n_dropped + 1.0);
                    let drop_factor = match normalize_type {
                        alloygbm_core::DartNormalize::Tree => n_dropped / (n_dropped + 1.0),
                        alloygbm_core::DartNormalize::Forest => 1.0 / (n_dropped + 1.0),
                    };
                    // Scale each class's new-tree contribution from w=1
                    // (as built into candidate) to w=new_w.
                    // class_candidate[k] = class_predictions[k] + new_w * f_T_k.
                    for class_k in 0..k {
                        let n_rows = class_candidate_predictions[class_k].len();
                        for r in 0..n_rows {
                            let f_t = class_candidate_predictions[class_k][r]
                                - class_predictions[class_k][r];
                            class_candidate_predictions[class_k][r] =
                                class_predictions[class_k][r] + new_w * f_t;
                        }
                    }
                    let new_dropped_weights: Vec<f32> = dropped_tree_indices
                        .iter()
                        .map(|&fi| dart_state.tree_weights[fi] * drop_factor)
                        .collect();
                    let dart_contribution = dart_train_contribution.as_ref().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "multiclass DART contribution buffer was not initialized".to_string(),
                        )
                    })?;
                    apply_multiclass_dart_contributions(
                        &mut class_predictions,
                        dart_contribution,
                        &material_dropped_classes,
                        drop_factor,
                    )?;
                    apply_multiclass_dart_contributions(
                        &mut class_candidate_predictions,
                        dart_contribution,
                        &material_dropped_classes,
                        drop_factor,
                    )?;
                    Some((new_w, drop_factor, new_dropped_weights))
                } else {
                    None
                };

            let in_warmup_phase = morph_state
                .as_ref()
                .is_some_and(|ms| ms.is_in_warmup_phase(effective_round));

            // PR review (C1): rejection paths must restore
            // `class_predictions` from `dart_predictions_backup` so the
            // next round sees the full pre-dropout ensemble.
            // `dart_state.tree_weights` was NOT mutated above, so no
            // weight rollback is needed.
            if !any_tree_produced {
                if in_warmup_phase {
                    // Empty rounds during warmup are expected: tiny LR produces
                    // leaves below `min_abs_leaf_value`, so all splits get
                    // rejected. This is benign — LR will ramp up. Restore
                    // class_predictions and skip this round.
                    if let Some(backup) = dart_predictions_backup.take() {
                        class_predictions = backup;
                    }
                    if dart_params.is_some() {
                        append_multiclass_dart_phantom_round(
                            &mut dart_state,
                            &mut dart_round_start_offsets,
                            &mut dart_round_counts,
                            &class_stumps,
                        );
                    }
                    append_multiclass_phantom_round_bookkeeping(
                        &mut stumps_per_round_per_class,
                        &mut loss_per_completed_round,
                        &mut validation_loss_per_completed_round,
                        &mut sampled_rows_per_completed_round,
                        &mut sampled_features_per_completed_round,
                        &mut diagnostics_per_round,
                        k,
                        current_loss,
                        current_validation_loss,
                        sampled_row_count,
                        sampled_feature_count,
                        &per_class_diagnostics,
                    );
                    rounds_completed += 1;
                    continue;
                }
                // Past warmup: an empty round indicates no useful split exists.
                // Break path: predictions aren't read again after the loop.
                let _ = dart_predictions_backup.take();
                for class_k in 0..k {
                    class_stumps[class_k].truncate(pre_round_counts[class_k]);
                }
                stop_reason = IterationStopReason::NoSplitCandidate;
                break;
            }

            // Check loss improvement
            let candidate_loss = objective.loss(
                &class_candidate_predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            let loss_gate_exempt = dart_params.is_some();
            let loss_gate_active = !loss_gate_exempt
                && (controls.training_loss_gate_enabled
                    || (in_warmup_phase && morph_state.is_some()));
            if loss_gate_active && loss_improvement < 0.0 {
                // Truncate stumps so the model doesn't include this round's contribution.
                // (`class_candidate_predictions` is reset from `class_predictions` at the
                // top of each round, so the candidate state is implicitly rolled back.)
                for class_k in 0..k {
                    class_stumps[class_k].truncate(pre_round_counts[class_k]);
                }
                if in_warmup_phase {
                    // During warmup, slightly-negative loss improvements arise from
                    // numerical noise at tiny LR (e.g., row-subsample variance over
                    // mostly-zero gradient updates). The model is not broken — LR will
                    // ramp up. Restore class_predictions and skip this round.
                    if let Some(backup) = dart_predictions_backup.take() {
                        class_predictions = backup;
                    }
                    if dart_params.is_some() {
                        append_multiclass_dart_phantom_round(
                            &mut dart_state,
                            &mut dart_round_start_offsets,
                            &mut dart_round_counts,
                            &class_stumps,
                        );
                    }
                    append_multiclass_phantom_round_bookkeeping(
                        &mut stumps_per_round_per_class,
                        &mut loss_per_completed_round,
                        &mut validation_loss_per_completed_round,
                        &mut sampled_rows_per_completed_round,
                        &mut sampled_features_per_completed_round,
                        &mut diagnostics_per_round,
                        k,
                        current_loss,
                        current_validation_loss,
                        sampled_row_count,
                        sampled_feature_count,
                        &per_class_diagnostics,
                    );
                    rounds_completed += 1;
                    continue;
                }
                // Break path: predictions aren't read again after the
                // loop, so backup restore is unnecessary.
                let _ = dart_predictions_backup.take();
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if !in_warmup_phase {
                let lr_threshold_scale = morph_state
                    .as_ref()
                    .map_or(1.0, |ms| ms.lr_loss_threshold_scale(effective_round));
                let effective_min_loss_improvement =
                    controls.min_loss_improvement * lr_threshold_scale;
                if controls.training_loss_gate_enabled
                    && !loss_gate_exempt
                    && loss_improvement < effective_min_loss_improvement
                {
                    if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                        for class_k in 0..k {
                            class_stumps[class_k].truncate(pre_round_counts[class_k]);
                        }
                        // Break path: predictions aren't read again
                        // after the loop, so backup restore is
                        // unnecessary; let it drop silently.
                        let _ = dart_predictions_backup.take();
                        stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                        break;
                    }
                    weak_improvement_streak += 1;
                    weak_improvement_rounds_committed += 1;
                } else {
                    weak_improvement_streak = 0;
                }
            }

            // Validation early stopping
            //
            // PR review (C6): when DART is active, mirror the training
            // transition on `validation_class_predictions` so the
            // validation loss is computed against the same full
            // ensemble (post-dropout, scaled new tree, re-added dropped
            // trees). Without this the early-stopping decision is made
            // against an inconsistent ensemble.
            let mut stop_for_validation_plateau = false;
            if let Some(validation_ref) = validation {
                let val_preds = validation_class_predictions.as_mut().unwrap();
                let val_raw = Some((
                    &validation_ref.dataset.matrix.values as &[f32],
                    validation_ref.dataset.matrix.feature_count,
                ));
                if let Some((new_w, drop_factor, _new_dropped_weights)) =
                    dart_round_finalize.as_ref()
                {
                    let dart_contribution =
                        dart_validation_contribution.as_mut().ok_or_else(|| {
                            EngineError::ContractViolation(
                                "validation multiclass DART contribution buffer was not initialized"
                                    .to_string(),
                            )
                        })?;
                    clear_multiclass_dart_contributions(
                        dart_contribution,
                        &material_dropped_classes,
                    )?;
                    // 1. Subtract each dropped class-tree at w_old.
                    for &flat_idx in &dropped_tree_indices {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        let w_old = dart_state.tree_weights[flat_idx];
                        apply_weighted_round_to_predictions_and_accumulator(
                            &mut val_preds[class_k],
                            &mut dart_contribution[class_k],
                            validation_ref.binned_matrix,
                            &stump_slice,
                            val_raw,
                            -w_old,
                            w_old,
                        )?;
                    }
                    // 2. Add the new K class-trees at new_w.
                    for class_k in 0..k {
                        let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                        if round_stumps.is_empty() {
                            continue;
                        }
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            round_stumps,
                            val_raw,
                            *new_w,
                        )?;
                    }
                    // 3. Re-add all dropped class-trees at their shared
                    // post-normalization factor.
                    apply_multiclass_dart_contributions(
                        val_preds,
                        dart_contribution,
                        &material_dropped_classes,
                        *drop_factor,
                    )?;
                } else {
                    // Non-DART (or DART with no dropouts AND new trees
                    // not yet rescaled): plain unit-weight tree walk.
                    for class_k in 0..k {
                        let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                        if !round_stumps.is_empty() {
                            apply_weighted_round_to_predictions(
                                &mut val_preds[class_k],
                                validation_ref.binned_matrix,
                                round_stumps,
                                val_raw,
                                1.0,
                            )?;
                        }
                    }
                }
                let next_validation_loss = objective.loss(
                    val_preds,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?;

                let improved = best_validation_loss
                    .map(|best| best - next_validation_loss > controls.min_validation_improvement)
                    .unwrap_or(true);
                if improved {
                    best_validation_loss = Some(next_validation_loss);
                    best_validation_round = Some(rounds_completed + 1);
                    validation_no_improvement_rounds = 0;
                } else if controls.early_stopping_rounds.is_some() {
                    validation_no_improvement_rounds += 1;
                }
                if let Some(patience) = controls.early_stopping_rounds
                    && validation_no_improvement_rounds >= patience
                {
                    stop_for_validation_plateau = true;
                }

                current_validation_loss = Some(next_validation_loss);
                validation_loss_per_completed_round.push(next_validation_loss);
            }

            // Accept round
            for class_k in 0..k {
                class_predictions[class_k].copy_from_slice(&class_candidate_predictions[class_k]);
            }
            current_loss = candidate_loss;
            loss_per_completed_round.push(candidate_loss);
            sampled_rows_per_completed_round.push(sampled_row_count);
            sampled_features_per_completed_round.push(sampled_feature_count);
            let stump_counts_this_round: Vec<usize> = (0..k)
                .map(|c| class_stumps[c].len() - pre_round_counts[c])
                .collect();
            stumps_per_round_per_class.push(stump_counts_this_round.clone());
            diagnostics_per_round.push(IterationDiagnostics::aggregate_per_class(
                &per_class_diagnostics,
            ));

            // v0.10.1 DART: commit per-tree-weight state for this round
            // ONLY after acceptance (PR review C1). On rejection we
            // never reach this point and `dart_state.tree_weights` /
            // `dart_round_start_offsets` / `dart_round_counts` keep
            // their pre-round shape, so the flat dropout index ↔
            // tree-slice mapping stays consistent.
            //
            // PR review (C5): stamp `stump.tree_weight = new_w` on
            // EVERY stump in `class_stumps[class_k][pre_round_counts[class_k]..]`,
            // not just `last_mut()`. Only push DART slots for class
            // trees that actually produced stumps this round (so
            // zero-stump class trees stay as phantom slots and
            // `dart_round_counts` reflects 0 for them).
            if let Some((new_w, _drop_factor, new_dropped_weights)) = dart_round_finalize.as_ref() {
                // Every skipped warmup round is recorded above in both the
                // DART arrays and the summary bookkeeping. Do not synthesize
                // an internal-only phantom here: that would make early-stop
                // truncation index a sparse stump-count vector again.
                if dart_state.dropped_per_round.len() != effective_round
                    || dart_round_start_offsets
                        .iter()
                        .any(|starts| starts.len() != effective_round)
                    || dart_round_counts
                        .iter()
                        .any(|counts| counts.len() != effective_round)
                {
                    return Err(EngineError::ContractViolation(format!(
                        "multiclass DART logical-round bookkeeping is not dense before round {effective_round}"
                    )));
                }
                // Rescale dropped trees' weights in place.
                for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
                    dart_state.tree_weights[flat_idx] = new_dropped_weights[i];
                }
                // Record this round in `dropped_per_round` (one entry
                // per multiclass round even though K trees are
                // committed).
                dart_state
                    .dropped_per_round
                    .push(dropped_tree_indices.clone());
                // Per-class round bookkeeping + tree_weight stamping.
                for class_k in 0..k {
                    let count = stump_counts_this_round[class_k];
                    let start = pre_round_counts[class_k];
                    dart_round_start_offsets[class_k].push(start);
                    dart_round_counts[class_k].push(count);
                    if count > 0 {
                        for stump in class_stumps[class_k][start..start + count].iter_mut() {
                            stump.tree_weight = *new_w;
                        }
                    }
                    // Push the per-tree weight regardless of count so
                    // the flat layout `r * K + class_k` is preserved.
                    // Phantom (count=0) trees get weight=new_w too;
                    // they contribute nothing to predictions but the
                    // flat indexing stays consistent across rounds.
                    dart_state.tree_weights.push(*new_w);
                }
            }

            rounds_completed += 1;

            if stop_for_validation_plateau {
                stop_reason = IterationStopReason::ValidationLossPlateau;
                break;
            }
        }

        // Truncate to best validation round if early stopping triggered
        if stop_reason == IterationStopReason::ValidationLossPlateau
            && let Some(best_round) = best_validation_round
            && best_round < rounds_completed
        {
            // Compute how many stumps to keep per class (inherited + best new rounds)
            for class_k in 0..k {
                let keep_count: usize = initial_stump_counts[class_k]
                    + stumps_per_round_per_class
                        .iter()
                        .take(best_round)
                        .map(|r| r[class_k])
                        .sum::<usize>();
                class_stumps[class_k].truncate(keep_count);
            }
            stumps_per_round_per_class.truncate(best_round);
            loss_per_completed_round.truncate(best_round);
            validation_loss_per_completed_round.truncate(best_round);
            sampled_rows_per_completed_round.truncate(best_round);
            sampled_features_per_completed_round.truncate(best_round);
            diagnostics_per_round.truncate(best_round);
            // DART normalization from a discarded logical round may have
            // changed retained tree weights. Keep all prefix slots (including
            // warm-start and phantom slots) through the selected logical
            // round, then replay only the new retained rounds from the
            // persisted warm-start weight prefix.
            if let Some((_, _, normalize_type, _)) = dart_params {
                let truncate_at = round_index_offset + best_round;
                for class_k in 0..k {
                    dart_round_start_offsets[class_k].truncate(truncate_at);
                    dart_round_counts[class_k].truncate(truncate_at);
                }
                let kept_dropped_per_round = dart_state
                    .dropped_per_round
                    .iter()
                    .take(truncate_at)
                    .cloned()
                    .collect::<Vec<_>>();
                if kept_dropped_per_round.len() != truncate_at {
                    return Err(EngineError::ContractViolation(format!(
                        "multiclass DART history length {} is shorter than retained logical rounds {truncate_at}",
                        kept_dropped_per_round.len(),
                    )));
                }
                dart_state.tree_weights = replay_multiclass_dart_tree_weights(
                    &initial_dart_tree_weights,
                    &kept_dropped_per_round,
                    k,
                    normalize_type,
                )?;
                dart_state.dropped_per_round = kept_dropped_per_round;
            }
            rounds_completed = best_round;
            current_loss = loss_per_completed_round
                .last()
                .copied()
                .unwrap_or(initial_loss);
            current_validation_loss = best_validation_loss.or(initial_validation_loss);
        }

        if dart_params.is_some() {
            for (class_k, stumps) in class_stumps.iter_mut().enumerate() {
                for stump in stumps {
                    let (tree_id, _) = decode_tree_node_id(stump.split.node_id);
                    let flat_tree_id = tree_id as usize * k + class_k;
                    if let Some(&weight) = dart_state.tree_weights.get(flat_tree_id) {
                        stump.tree_weight = weight;
                    }
                }
            }
        }

        let final_loss = current_loss;
        let final_validation_loss = current_validation_loss;

        let morph_metadata = morph_state.as_ref().map(|ms| MorphMetadataPayload {
            config: ms.config,
            final_iteration: rounds_completed as u32,
            final_total: total_iterations,
            // v0.7.3: persist EMA so warm-start can resume from the
            // exact same EMA state rather than restarting cold.
            ema_stats: ms.ema_stats.clone(),
        });
        let dro_metadata = self
            .params
            .dro_config
            .map(|config| DroMetadataPayload { config });
        Ok(MultiClassIterationRunSummary {
            model: MultiClassTrainedModel {
                num_classes: k,
                baseline_predictions: baselines,
                feature_count,
                class_stumps,
                categorical_state: None,
                objective: objective.objective_name().to_string(),
                morph_metadata,
                dro_metadata,
            },
            rounds_requested: effective_round_cap,
            effective_round_cap,
            resolved_training_policy,
            rounds_completed,
            stop_reason,
            initial_loss,
            initial_validation_loss,
            loss_per_completed_round,
            validation_loss_per_completed_round,
            sampled_rows_per_completed_round,
            sampled_features_per_completed_round,
            best_validation_loss,
            best_validation_round,
            weak_improvement_rounds_committed,
            final_loss,
            final_validation_loss,
            custom_metric_per_round: Vec::new(),
            custom_metric_name: None,
            diagnostics_per_round,
        })
    }

    pub(crate) fn default_iteration_controls(
        &self,
        rounds: usize,
    ) -> EngineResult<IterationControls> {
        let mut controls = IterationControls::new(
            rounds,
            self.params.min_split_gain,
            self.params.min_data_in_leaf as usize,
            0.0,
            1_000_000.0,
            0.0,
            0,
        )?
        .with_subsample_rates(self.params.row_subsample, self.params.col_subsample)?;
        if let Some(early_stopping_rounds) = self.params.early_stopping_rounds {
            controls = controls.with_validation_early_stopping(
                early_stopping_rounds as usize,
                self.params.min_validation_improvement,
            )?;
        }
        controls = controls.with_max_leaves(self.params.max_leaves)?;
        Ok(controls)
    }

    fn auto_iteration_controls(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        is_ranking: bool,
    ) -> EngineResult<IterationControls> {
        validate_training_alignment(dataset, binned_matrix)?;
        let mut controls = self.default_iteration_controls(rounds)?;
        let row_count = dataset.row_count();
        let feature_count = binned_matrix.feature_count;
        let target_variance = target_variance(&dataset.targets, dataset.sample_weights.as_deref())?;
        if row_count < 1_024 {
            let rows_per_feature = row_count as f32 / feature_count.max(1) as f32;
            if feature_count >= 8
                && rounds > 256
                && rows_per_feature < 64.0
                && target_variance > 1.0
            {
                controls.rounds = rounds.min(96);
            }
            return Ok(controls);
        }

        let binned_density = binned_feature_density(binned_matrix);

        let suggested_min_rows = if row_count < 128 {
            1
        } else if row_count < 512 {
            2
        } else if row_count < 2_048 {
            4
        } else if row_count < 8_192 {
            8
        } else {
            16
        };
        let user_min = self.params.min_data_in_leaf as usize;
        controls.min_rows_per_leaf = suggested_min_rows
            .max(user_min)
            .min(row_count.saturating_div(2).max(1));

        // Ranking objectives produce gradients whose gain scale differs from
        // regression/classification. The density-based min_split_gain floor
        // was tuned for regression losses and can stop ranking early, so keep
        // it disabled for ranking. Training-loss stopping is opt-in globally;
        // validation early stopping is the default stopping policy.
        let auto_min_split_gain: f32 = if is_ranking {
            0.0
        } else if binned_density < 0.10 {
            0.001
        } else if row_count.saturating_mul(feature_count) >= 65_536 {
            0.0001
        } else {
            0.0
        };
        controls.min_split_gain = auto_min_split_gain.max(self.params.min_split_gain);
        controls.min_loss_improvement = 0.0;
        controls.max_consecutive_weak_improvements = 0;

        if self.params.row_subsample == 1.0 && row_count >= 2_048 {
            controls.row_subsample = if row_count >= 16_384 { 0.8 } else { 0.9 };
        }
        if self.params.col_subsample == 1.0 && feature_count >= 32 {
            controls.col_subsample = if feature_count >= 256 {
                0.5
            } else if feature_count >= 128 {
                0.65
            } else {
                0.8
            };
        }

        Ok(controls)
    }

    fn fit_iterations_with_optional_validation_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        mut execution: IterationExecutionContext<'_>,
    ) -> EngineResult<IterationRunSummary> {
        let controls = execution.controls;
        let validation = execution.validation;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
        // v0.9.0: DART support is wired through for the single-output
        // trainer. Warm-start + DART is not yet supported — would
        // require persisting `tree_weights` and `dropped_per_round` in
        // `WarmStartState` (tracked as a v0.10.x follow-up).
        let dart_params = match self.params.boosting_mode {
            BoostingMode::Dart {
                drop_rate,
                max_drop,
                normalize_type,
                sample_type,
            } => Some((drop_rate, max_drop, normalize_type, sample_type)),
            _ => None,
        };
        // v0.10.0: DART + warm_start is now supported. See the dart_state
        // seeding logic below — `dart_state.tree_weights` is initialized from
        // `warm_start.initial_dart_tree_weights` when present, falling back
        // to all-1.0s. Historical `dropped_per_round` is not persisted; new
        // rounds start fresh dropout bookkeeping going forward, which is
        // the natural semantics for continuation (RNG-driven dropout
        // history cannot be replayed from the prior fit).
        validate_training_alignment(dataset, binned_matrix)?;
        if objective.requires_group_id() && dataset.group_id.is_none() {
            return Err(EngineError::ContractViolation(
                "this objective requires group_id to be provided on the training dataset"
                    .to_string(),
            ));
        }
        if let Some(validation_ref) = validation {
            validate_training_alignment(validation_ref.dataset, validation_ref.binned_matrix)?;
            if validation_ref.dataset.matrix.feature_count != dataset.matrix.feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "validation feature_count {} does not match training feature_count {}",
                    validation_ref.dataset.matrix.feature_count, dataset.matrix.feature_count
                )));
            }
            if objective.requires_group_id() && validation_ref.dataset.group_id.is_none() {
                return Err(EngineError::ContractViolation(
                    "this objective requires group_id to be provided on the validation dataset"
                        .to_string(),
                ));
            }
        }
        validate_train_params(&self.params)?;
        if let Some(categorical_feature) = self.categorical_features.iter().find(|feature| {
            self.params
                .monotone_constraints
                .get(feature.feature_index)
                .is_some_and(|&constraint| constraint != 0)
        }) {
            return Err(EngineError::InvalidConfig(format!(
                "categorical_features native categorical splitting at feature {} is not supported with active monotone_constraints on that feature",
                categorical_feature.feature_index
            )));
        }
        if let Some(qa) = objective.quantile_alpha()
            && (!qa.is_finite() || qa <= 0.0 || qa >= 1.0)
        {
            return Err(EngineError::InvalidConfig(
                "quantile_alpha must be finite and in (0.0, 1.0)".to_string(),
            ));
        }
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        validate_warm_start_neutralization_contract(
            &self.params,
            execution.warm_start.is_some(),
            dataset,
        )?;
        let owned_dataset = if execution.pre_target_already_applied {
            None
        } else {
            prepare_pre_target_training_dataset(&self.params, dataset)?
        };
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        let fit_contract =
            self.evaluate_fit_contract_on_active_dataset(active_dataset, objective)?;
        let gradient_projector = if let Some(config) = gradient_neutralization_config(&self.params)
        {
            let exposures = active_dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            Some(FactorProjector::new(
                exposures,
                active_dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?)
        } else {
            None
        };
        let sampling_seed_base = sampling_seed_base(self.params.seed, self.params.deterministic);
        let split_resolution = split_selection_options_with_resolution_for_training(
            &self.params,
            execution.policy_mode,
            active_dataset,
            binned_matrix,
        )?;
        let resolved_training_policy = resolve_training_policy(controls, &split_resolution);
        let split_options = split_resolution.options;

        // Warm-start: use existing model's baseline + apply existing
        // trees.  `warm_ema_stats` captures the MorphBoost EMA snapshot
        // for the v0.7.3 warm-start-equivalence fix (consumed below
        // when `MorphState::new` builds the fresh EMA).
        // v0.10.0 review fix (Comment 1): also capture
        // `initial_dart_tree_weights` here BEFORE `take()` consumes the
        // warm_start; the dart_state seeding step below reads this local
        // variable rather than re-querying `execution.warm_start`.
        let (
            baseline_prediction,
            initial_stumps,
            round_index_offset,
            warm_ema_stats,
            initial_dart_tree_weights,
        ) = if let Some(warm_start) = execution.warm_start.take() {
            (
                warm_start.baseline_prediction,
                warm_start.stumps,
                warm_start.initial_rounds_completed,
                warm_start.initial_ema_stats,
                warm_start.initial_dart_tree_weights,
            )
        } else {
            (fit_contract.baseline_prediction, Vec::new(), 0, None, None)
        };
        if has_active_monotone_constraints(&self.params.monotone_constraints)
            && let Some(stump) = initial_stumps.iter().find(|stump| {
                stump.split.is_categorical
                    && self
                        .params
                        .monotone_constraints
                        .get(stump.split.feature_index as usize)
                        .is_some_and(|&direction| direction != 0)
            })
        {
            return Err(EngineError::InvalidConfig(format!(
                "warm_start native categorical split on feature {} is not compatible with active monotone_constraints",
                stump.split.feature_index
            )));
        }
        let mut initial_stumps_per_round = vec![0_usize; round_index_offset];
        for stump in &initial_stumps {
            let tree_id = decode_tree_node_id(stump.split.node_id).0 as usize;
            let initial_round_count = initial_stumps_per_round.len();
            let count = initial_stumps_per_round.get_mut(tree_id).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "warm-start tree_id {tree_id} is outside initial_rounds_completed {initial_round_count}"
                ))
            })?;
            *count = count.checked_add(1).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "warm-start stump count overflow for tree_id {tree_id}"
                ))
            })?;
        }
        validate_monotone_forest(
            &initial_stumps,
            &initial_stumps_per_round,
            &self.params.monotone_constraints,
            controls.max_abs_leaf_value,
        )
        .map_err(|error| {
            EngineError::InvalidConfig(format!(
                "warm_start is not compatible with active monotone_constraints: {error}"
            ))
        })?;
        let raw_features_opt = Some((
            &active_dataset.matrix.values as &[f32],
            active_dataset.matrix.feature_count,
        ));
        let mut predictions = vec![baseline_prediction; active_dataset.row_count()];
        if !initial_stumps.is_empty() {
            apply_tree_to_binned_predictions(
                &mut predictions,
                binned_matrix,
                &initial_stumps,
                raw_features_opt,
            )?;
        }
        let mut candidate_predictions = predictions.clone();
        let mut validation_predictions = if let Some(validation_ref) = validation {
            let mut vp = vec![baseline_prediction; validation_ref.dataset.row_count()];
            if !initial_stumps.is_empty() {
                apply_tree_to_binned_predictions(
                    &mut vp,
                    validation_ref.binned_matrix,
                    &initial_stumps,
                    Some((
                        &validation_ref.dataset.matrix.values as &[f32],
                        validation_ref.dataset.matrix.feature_count,
                    )),
                )?;
            }
            Some(vp)
        } else {
            None
        };
        let mut dart_train_contribution =
            allocate_dart_contribution_buffer(dart_params.is_some(), predictions.len());
        let mut dart_validation_contribution = validation_predictions.as_ref().and_then(|values| {
            allocate_dart_contribution_buffer(dart_params.is_some(), values.len())
        });
        let mut stumps = initial_stumps;
        let initial_stump_count = stumps.len();
        // New material-round counts retain their historical coordinate system.
        // MorphBoost phantom rounds are deliberately absent from this vector.
        let mut stumps_per_completed_round: Vec<usize> = Vec::new();
        let mut rounds_completed = 0_usize;
        let effective_round_cap = controls.rounds;
        let mut stop_reason = IterationStopReason::CompletedRequestedRounds;
        let initial_loss = objective.loss(
            &predictions,
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        let initial_validation_loss = if let Some(validation_ref) = validation {
            let validation_predictions_ref = validation_predictions.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "validation predictions were not initialized".to_string(),
                )
            })?;
            Some(objective.loss(
                validation_predictions_ref,
                &validation_ref.dataset.targets,
                validation_ref.dataset.sample_weights.as_deref(),
            )?)
        } else {
            None
        };
        let mut current_loss = initial_loss;
        let mut current_validation_loss = initial_validation_loss;
        let mut loss_per_completed_round = Vec::new();
        let mut validation_loss_per_completed_round = Vec::new();
        let mut sampled_rows_per_completed_round = Vec::new();
        let mut sampled_features_per_completed_round = Vec::new();
        let mut diagnostics_per_round: Vec<IterationDiagnostics> = Vec::new();
        let mut best_validation_loss = initial_validation_loss;
        let mut best_validation_round = initial_validation_loss.map(|_| 0_usize);
        let mut validation_no_improvement_rounds = 0_usize;
        let mut weak_improvement_streak = 0_usize;
        let mut weak_improvement_rounds_committed = 0_usize;

        // Custom metric tracking
        let custom_metric_callback = execution.custom_metric_callback;
        let mut custom_metric_per_round: Vec<f32> = Vec::new();
        let custom_metric_name = custom_metric_callback.map(|cb| cb.metric_name().to_string());
        let custom_metric_higher_is_better = custom_metric_callback
            .map(|cb| cb.higher_is_better())
            .unwrap_or(false);
        let mut best_custom_metric: Option<f32> = None;
        let mut best_custom_metric_round: Option<usize> = None;
        let mut custom_metric_no_improvement_rounds = 0_usize;

        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(active_dataset.row_count());
        let mut projection_scratch: Vec<f32> = Vec::with_capacity(active_dataset.row_count());

        // DART state: tree_weights[tree_id] tracks the multiplicative
        // weight applied to each previously-trained tree. Populated
        // before the round loop (initial weights = 1.0 for every
        // already-existing tree); per-round `select_dropouts` consults
        // it and `apply_normalization` mutates it. Stays empty for
        // non-DART fits, in which case the stamping step at the bottom
        // of this function is a no-op and stumps keep their default
        // `tree_weight = 1.0`.
        let mut dart_state = DartState::default();
        if dart_params.is_some() {
            // v0.10.0: When continuing a DART fit, seed tree_weights from the
            // warm-start snapshot captured above (BEFORE the take()). Length
            // must equal `stumps.len()` (one weight per warm-start stump).
            // Falls back to all-1.0s when the prior fit did not use DART
            // or no snapshot was provided.
            let initial_tree_count = initial_stumps_per_round.len();
            if let Some(saved_weights) = initial_dart_tree_weights.as_ref() {
                // Caller supplies one weight per stump; we need one weight
                // per tree. Take the first weight of each tree (all stumps
                // in a tree share the same tree_weight after DART
                // normalization, so this is well-defined).
                let mut per_tree = Vec::with_capacity(initial_tree_count);
                let mut stump_offset = 0usize;
                for &count in &initial_stumps_per_round {
                    let weight = saved_weights.get(stump_offset).copied().unwrap_or(1.0);
                    per_tree.push(weight);
                    stump_offset += count;
                }
                dart_state.tree_weights = per_tree;
            } else {
                dart_state.tree_weights = vec![1.0; initial_tree_count];
            }
            // Historical `dropped_per_round` is initialized empty per warm
            // round — RNG-driven dropout history cannot be replayed.
            for _ in 0..initial_tree_count {
                dart_state.dropped_per_round.push(Vec::new());
            }
        }
        // DART-only parallel arrays indexed by `effective_round_index`
        // (= `tree_id` encoded in stump.node_id). Both grow together at
        // commit-time, and skipped warmup rounds get phantom entries
        // (`count = 0`, `start = stumps.len()`) so the indexing stays
        // dense even when MorphBoost skips rounds.
        //
        // `round_start_offsets[t]` is the start index in `stumps` where
        // tree `t`'s stumps begin; `dart_round_counts[t]` is its stump
        // count. Together they slice into `stumps` for the DART dropout
        // subtract/replay step.
        let mut round_start_offsets: Vec<usize> = Vec::new();
        let mut dart_round_counts: Vec<usize> = Vec::new();

        // v0.10.0: DART + warm_start — pre-populate round_start_offsets +
        // dart_round_counts from the warm-start tree shapes so the dropout
        // step can correctly slice into `stumps` for each prior tree.
        if dart_params.is_some() {
            let mut offset = 0usize;
            for &count in &initial_stumps_per_round {
                round_start_offsets.push(offset);
                dart_round_counts.push(count);
                offset += count;
            }
        }

        // Build MorphState for the duration of training when morph_config is set.
        // `total_iterations` corresponds to the round cap (including any warm-start
        // offset already-completed rounds, so the LR schedule lines up).
        let total_iterations = (effective_round_cap + round_index_offset) as u32;
        let mut morph_state: Option<MorphState> = self
            .params
            .morph_config
            .map(|cfg| MorphState::new(cfg, 1, total_iterations, self.params.learning_rate));

        // v0.7.3 EMA warm-start: seed the fresh `MorphState` with the
        // EMA snapshot from the previous fit when the warm-start state
        // carries one.  Single-class MorphState has `ema_stats.len() = 1`.
        if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), warm_ema_stats.as_ref())
            && ms.ema_stats.len() == snapshot.len()
        {
            ms.ema_stats.copy_from_slice(snapshot);
        }

        for round_index in 0..effective_round_cap {
            // Offset round_index for sampling seeds and tree IDs when warm-starting
            let effective_round_index = round_index + round_index_offset;

            // DART: drop a random subset of previously-trained trees
            // before computing gradients. Subtract their (currently
            // weighted) contributions from `predictions` and
            // `validation_predictions` so the new tree fits on
            // residuals of the dropped-out ensemble. The dropped
            // trees are re-added (at rescaled weights) after the new
            // tree is committed; on early-exit (empty round, loss
            // regression, etc.) the buffer backups below are used to
            // restore `predictions`/`validation_predictions` to the
            // full-ensemble state so subsequent rounds aren't poisoned.
            let mut dart_predictions_backup: Option<Vec<f32>> = None;
            let mut dart_validation_backup: Option<Vec<f32>> = None;
            let dropped_tree_ids: Vec<usize> =
                if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                    let train_contribution = dart_train_contribution.as_mut().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "DART contribution buffer was not initialized".to_string(),
                        )
                    })?;
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round_index,
                    );
                    train_contribution.fill(0.0);
                    if let Some(contribution) = dart_validation_contribution.as_mut() {
                        contribution.fill(0.0);
                    }
                    if !drops.is_empty() {
                        dart_predictions_backup = Some(predictions.clone());
                        dart_validation_backup = validation_predictions.clone();
                    }
                    for &tree_id in &drops {
                        let w_old = dart_state.tree_weights[tree_id];
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions_and_accumulator(
                            &mut predictions,
                            train_contribution,
                            binned_matrix,
                            stump_slice,
                            raw_features_opt,
                            -w_old,
                            w_old,
                        )?;
                        if let (Some(vp), Some(validation_ref), Some(contribution)) = (
                            validation_predictions.as_mut(),
                            validation,
                            dart_validation_contribution.as_mut(),
                        ) {
                            let val_raw = Some((
                                &validation_ref.dataset.matrix.values as &[f32],
                                validation_ref.dataset.matrix.feature_count,
                            ));
                            apply_weighted_round_to_predictions_and_accumulator(
                                vp,
                                contribution,
                                validation_ref.binned_matrix,
                                stump_slice,
                                val_raw,
                                -w_old,
                                w_old,
                            )?;
                        }
                    }
                    drops
                } else {
                    Vec::new()
                };

            // Helper to restore `predictions` and `validation_predictions`
            // from the pre-dropout backups. Called on every early-exit
            // path so the full-ensemble state is preserved.
            //
            // Captured as a small struct rather than a closure because
            // we need to call it from multiple branch arms below, and
            // closures over `&mut predictions` get awkward to reuse.
            //
            // NB: we cannot define a helper fn here since `predictions`
            // is borrowed mutably; we inline the restore at each
            // early-exit site instead. The backups are `Option<Vec<f32>>`
            // and `.take()`able so the restore is a single move.

            // v0.8.0: gradient computation moved before row sampling so
            // GOSS can score rows by `|gradient|`.  Standard / DART
            // boosting modes ignore the gradient input and fall back
            // to uniform subsampling.
            objective.compute_gradients_into(
                &predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
                &mut gradient_buffer,
            )?;
            // Capture the pre-projection L2 norm so neutralization
            // effectiveness can be reported alongside the post-projection
            // gradient stats below.  Only allocated when a per-round
            // projection is actually configured for this fit.
            let original_gradient_norm = if gradient_projector.is_some() {
                Some(gradient_l2_norm_only(&gradient_buffer))
            } else {
                None
            };
            if let Some(projector) = &gradient_projector {
                projector.project_gradient_pairs_in_place_with_scratch(
                    &mut gradient_buffer,
                    &mut projection_scratch,
                )?;
            }
            let root_row_indices = select_row_indices_for_round(
                self.params.boosting_mode,
                active_dataset.row_count(),
                controls.row_subsample,
                sampling_seed_base,
                effective_round_index as u64,
                &mut gradient_buffer,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                binned_matrix.feature_count,
                controls.col_subsample,
                sampling_seed_base,
                effective_round_index as u64,
            )?;
            let sampled_row_count = root_row_indices.len();
            let gradients = &gradient_buffer;
            validate_gradient_pair_length(gradients, active_dataset.row_count())?;
            if cfg!(debug_assertions) {
                validate_gradient_pairs(gradients, active_dataset.row_count())?;
            }
            // Capture per-round telemetry from the *post-projection* gradient
            // buffer — i.e., the values the tree-building code actually
            // consumes.  Push happens further below, conditional on the
            // round being committed, so we stay in lockstep with the other
            // per-round vecs (loss_per_completed_round, etc.).  Round-level
            // cost: a single linear pass over the gradient buffer.
            let round_diagnostics = IterationDiagnostics::from_gradient_snapshot(
                gradients,
                original_gradient_norm,
                sampled_row_count,
                feature_tiles.len(),
            );

            // Update EMA stats from this round's gradients before tree-building so
            // morph split selection sees the latest mean/std.
            if let Some(ms) = morph_state.as_mut() {
                ms.update_ema_from_gradient_pairs(gradients, 0);
            }

            candidate_predictions.copy_from_slice(&predictions);

            let morph_tree_ctx: Option<MorphTreeContext<'_>> =
                morph_state.as_ref().map(|ms| MorphTreeContext {
                    state: ms,
                    iteration: effective_round_index as u32,
                    total_iterations,
                    class_idx: 0,
                });

            let mut round_split_options = split_options;
            round_split_options.min_rows_per_leaf = controls.min_rows_per_leaf;
            let raw_fv = &active_dataset.matrix.values;
            let (mut candidate_round_stumps, round_rejection_reason) =
                if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        gradients,
                        root_row_indices,
                        effective_round_index,
                        &feature_tiles,
                        round_split_options,
                        &self.params,
                        &controls,
                        &mut candidate_predictions,
                        &self.params.feature_weights,
                        &execution.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        active_dataset.factor_exposures.as_ref(),
                    )?
                } else {
                    build_tree_level_wise(
                        backend,
                        binned_matrix,
                        gradients,
                        root_row_indices,
                        effective_round_index,
                        &feature_tiles,
                        round_split_options,
                        &self.params,
                        &controls,
                        &mut candidate_predictions,
                        &self.params.feature_weights,
                        &execution.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        active_dataset.factor_exposures.as_ref(),
                    )?
                };

            let in_warmup_phase = morph_state
                .as_ref()
                .is_some_and(|ms| ms.is_in_warmup_phase(effective_round_index));

            if candidate_round_stumps.is_empty() {
                // DART: empty round → no new tree → restore predictions
                // and validation_predictions from backup so subsequent
                // rounds see the full pre-dropout ensemble.
                if let Some(backup) = dart_predictions_backup.take() {
                    predictions = backup;
                }
                if let Some(backup) = dart_validation_backup.take() {
                    validation_predictions = Some(backup);
                }
                if in_warmup_phase {
                    // Empty rounds during warmup are expected: tiny LR produces
                    // leaves below `min_abs_leaf_value`, so all splits get
                    // rejected. This is benign — LR will ramp up. Skip this
                    // round and continue.
                    rounds_completed += 1;
                    continue;
                }
                stop_reason = round_rejection_reason;
                break;
            }

            if let Some(alpha) = objective.quantile_alpha() {
                let morph_scale_context = morph_state
                    .as_ref()
                    .map(|ms| (ms, effective_round_index, total_iterations));
                refine_quantile_leaf_values(
                    &mut candidate_round_stumps,
                    binned_matrix,
                    &predictions,
                    &active_dataset.targets,
                    active_dataset.sample_weights.as_deref(),
                    alpha,
                    self.params.learning_rate,
                    controls.max_abs_leaf_value,
                    raw_features_opt,
                    morph_scale_context,
                )?;
                project_monotone_tree(
                    &mut candidate_round_stumps,
                    &self.params.monotone_constraints,
                    controls.max_abs_leaf_value,
                )?;
            }

            candidate_predictions.copy_from_slice(&predictions);
            // Tree builders update only the sampled partition rows while constructing split
            // statistics. Rebuild the candidate by walking the accepted tree over every
            // training row so the training state matches inference semantics.
            apply_weighted_round_to_predictions(
                &mut candidate_predictions,
                binned_matrix,
                &candidate_round_stumps,
                raw_features_opt,
                1.0,
            )?;

            // DART: rebuild `candidate_predictions` to reflect the
            // post-normalization weights. After `build_tree_*` returned,
            // `candidate_predictions = predictions_dropped_out + 1.0 *
            // f_T(x)`. We want
            // `candidate_predictions = predictions_dropped_out + new_w *
            // f_T(x) + sum_dropped(w_new_i * f_i(x))`. Compute new
            // weights locally; the mutation of `dart_state.tree_weights`
            // only happens on commit (post-loss-check).
            //
            // `dart_round_finalize = Some((new_w, drop_factor, new_dropped_weights))`
            // on a DART round; `None` otherwise. The commit path
            // consumes this to update `dart_state`.
            let dart_round_finalize: Option<(f32, f32, Vec<f32>)> =
                if let Some((_, _, normalize_type, _)) = dart_params {
                    let k = dropped_tree_ids.len() as f32;
                    let new_w = 1.0 / (k + 1.0);
                    let drop_factor = match normalize_type {
                        alloygbm_core::DartNormalize::Tree => k / (k + 1.0),
                        alloygbm_core::DartNormalize::Forest => 1.0 / (k + 1.0),
                    };
                    // Step 1: scale the new tree's f_T contribution from 1.0
                    // to new_w in candidate_predictions.
                    for r in 0..candidate_predictions.len() {
                        let f_t = candidate_predictions[r] - predictions[r];
                        candidate_predictions[r] = predictions[r] + new_w * f_t;
                    }
                    // Step 2: collect the post-normalization weights for
                    // commit, then restore the aggregate dropped-tree
                    // contribution at the common normalization factor.
                    let mut new_dropped_weights = Vec::with_capacity(dropped_tree_ids.len());
                    for &tree_id in &dropped_tree_ids {
                        let w_old = dart_state.tree_weights[tree_id];
                        let w_new = w_old * drop_factor;
                        new_dropped_weights.push(w_new);
                    }
                    let contribution = dart_train_contribution.as_ref().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "DART contribution buffer was not initialized".to_string(),
                        )
                    })?;
                    apply_scaled_prediction_buffer(
                        &mut candidate_predictions,
                        contribution,
                        drop_factor,
                    )?;
                    Some((new_w, drop_factor, new_dropped_weights))
                } else {
                    None
                };

            let candidate_loss = objective.loss(
                &candidate_predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            // Ranking objectives (LambdaMART, pairwise, XeNDCG, YetiRank,
            // QueryRMSE) have bounded, NDCG-weighted losses whose round-to-
            // round training delta is often negative under row_subsample —
            // this does not reflect real ranking quality regression and the
            // boosting loop recovers on subsequent rounds. Skip the hard
            // "loss went up" early-exit for ranking objectives; rely on
            // validation early stopping (if configured) and the round cap.
            // DART is also non-monotone by construction: dropout and
            // normalization can raise training loss for a valid round.
            let loss_gate_exempt = objective.requires_group_id() || dart_params.is_some();
            let loss_gate_active = !loss_gate_exempt
                && (controls.training_loss_gate_enabled
                    || (in_warmup_phase && morph_state.is_some()));
            if loss_gate_active && loss_improvement < 0.0 {
                if in_warmup_phase {
                    // DART: loss regression on warmup continue path → restore
                    // from pre-dropout backup so the next round sees the
                    // full pre-dropout ensemble.
                    if let Some(backup) = dart_predictions_backup.take() {
                        predictions = backup;
                    }
                    if let Some(backup) = dart_validation_backup.take() {
                        validation_predictions = Some(backup);
                    }
                    // During warmup, slightly-negative loss improvements arise from
                    // numerical noise at tiny LR. Skip this round and continue;
                    // candidate predictions reset from current at the top of each round.
                    rounds_completed += 1;
                    continue;
                }
                // Break path: predictions is not read again after the loop,
                // so restoration is unnecessary here.
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if !in_warmup_phase {
                let lr_threshold_scale = morph_state
                    .as_ref()
                    .map_or(1.0, |ms| ms.lr_loss_threshold_scale(effective_round_index));
                let effective_min_loss_improvement =
                    controls.min_loss_improvement * lr_threshold_scale;
                if controls.training_loss_gate_enabled
                    && !loss_gate_exempt
                    && loss_improvement < effective_min_loss_improvement
                {
                    if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                        // Break path: see note above — restoration is
                        // unnecessary since the loop exits.
                        stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                        break;
                    }
                    weak_improvement_streak += 1;
                    weak_improvement_rounds_committed += 1;
                } else {
                    weak_improvement_streak = 0;
                }
            }

            let mut candidate_validation_predictions = None;
            let mut candidate_validation_loss = None;
            let mut stop_for_validation_plateau = false;
            let mut stop_for_custom_metric_plateau = false;
            if let Some(validation_ref) = validation {
                let mut next_validation_predictions =
                    validation_predictions.take().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "validation predictions were not initialized".to_string(),
                        )
                    })?;
                // DART-aware: when dropouts happened this round, add the
                // new tree at `new_w` (not 1.0) and re-add dropped trees
                // at their new weights. Otherwise fall back to the
                // existing unit-weight tree walk.
                if let Some((new_w, drop_factor, _new_dropped_weights)) = &dart_round_finalize {
                    let val_raw = Some((
                        &validation_ref.dataset.matrix.values as &[f32],
                        validation_ref.dataset.matrix.feature_count,
                    ));
                    apply_weighted_round_to_predictions(
                        &mut next_validation_predictions,
                        validation_ref.binned_matrix,
                        &candidate_round_stumps,
                        val_raw,
                        *new_w,
                    )?;
                    let contribution = dart_validation_contribution.as_ref().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "validation DART contribution buffer was not initialized".to_string(),
                        )
                    })?;
                    apply_scaled_prediction_buffer(
                        &mut next_validation_predictions,
                        contribution,
                        *drop_factor,
                    )?;
                } else {
                    apply_weighted_round_to_predictions(
                        &mut next_validation_predictions,
                        validation_ref.binned_matrix,
                        &candidate_round_stumps,
                        Some((
                            &validation_ref.dataset.matrix.values as &[f32],
                            validation_ref.dataset.matrix.feature_count,
                        )),
                        1.0,
                    )?;
                }
                let next_validation_loss = objective.loss(
                    &next_validation_predictions,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?;

                // Custom metric callback: evaluate on validation predictions
                if let Some(cb) = custom_metric_callback {
                    let metric_value = cb.evaluate(
                        &next_validation_predictions,
                        &validation_ref.dataset.targets,
                        validation_ref.dataset.sample_weights.as_deref(),
                    )?;
                    custom_metric_per_round.push(metric_value);

                    // Custom metric drives early stopping when present
                    let metric_improved = match best_custom_metric {
                        Some(best) => {
                            if custom_metric_higher_is_better {
                                metric_value - best > controls.min_validation_improvement
                            } else {
                                best - metric_value > controls.min_validation_improvement
                            }
                        }
                        None => true,
                    };
                    if metric_improved {
                        best_custom_metric = Some(metric_value);
                        best_custom_metric_round = Some(rounds_completed + 1);
                        custom_metric_no_improvement_rounds = 0;
                    } else if controls.early_stopping_rounds.is_some() {
                        custom_metric_no_improvement_rounds += 1;
                    }
                    if let Some(patience) = controls.early_stopping_rounds
                        && custom_metric_no_improvement_rounds >= patience
                    {
                        stop_for_custom_metric_plateau = true;
                    }
                }

                // When custom metric is NOT present, use built-in validation loss for early stopping
                if custom_metric_callback.is_none() {
                    let improved = best_validation_loss
                        .map(|best| {
                            best - next_validation_loss > controls.min_validation_improvement
                        })
                        .unwrap_or(true);
                    if improved {
                        best_validation_loss = Some(next_validation_loss);
                        best_validation_round = Some(rounds_completed + 1);
                        validation_no_improvement_rounds = 0;
                    } else if controls.early_stopping_rounds.is_some() {
                        validation_no_improvement_rounds += 1;
                    }
                    if let Some(patience) = controls.early_stopping_rounds
                        && validation_no_improvement_rounds >= patience
                    {
                        stop_for_validation_plateau = true;
                    }
                } else {
                    // Still track validation loss for reporting, but don't use it for stopping
                    best_validation_loss = best_validation_loss
                        .map(|best| {
                            if next_validation_loss < best {
                                next_validation_loss
                            } else {
                                best
                            }
                        })
                        .or(Some(next_validation_loss));
                    if best_validation_loss == Some(next_validation_loss) {
                        best_validation_round = Some(rounds_completed + 1);
                    }
                }

                candidate_validation_predictions = Some(next_validation_predictions);
                candidate_validation_loss = Some(next_validation_loss);
            }

            std::mem::swap(&mut predictions, &mut candidate_predictions);
            current_loss = candidate_loss;
            loss_per_completed_round.push(candidate_loss);
            sampled_rows_per_completed_round.push(sampled_row_count);
            sampled_features_per_completed_round.push(sampled_feature_count);
            diagnostics_per_round.push(round_diagnostics);
            if let Some(next_validation_predictions) = candidate_validation_predictions {
                validation_predictions = Some(next_validation_predictions);
            }
            if let Some(next_validation_loss) = candidate_validation_loss {
                current_validation_loss = Some(next_validation_loss);
                validation_loss_per_completed_round.push(next_validation_loss);
            }

            // DART: commit the post-normalization weights to dart_state.
            // Backups (`dart_predictions_backup`, `dart_validation_backup`)
            // are loop-scoped, so they get dropped at end-of-iteration
            // automatically — no explicit reset needed.
            //
            // Pad all four DART-indexed parallel arrays
            // (`dart_state.tree_weights`, `dart_state.dropped_per_round`,
            // `round_start_offsets`, `dart_round_counts`) up to
            // `effective_round_index` with phantom entries for any
            // skipped warmup rounds.  Phantoms have weight=1.0 and
            // count=0, so a later `select_dropouts` could pick one but
            // the resulting subtract is a no-op
            // (`apply_weighted_round_to_predictions` early-returns on
            // empty stump slices).  This keeps tree_id (=
            // effective_round_index) and the DART arrays consistent
            // even when MorphBoost skips rounds.
            if dart_params.is_some() {
                while round_start_offsets.len() < effective_round_index {
                    round_start_offsets.push(stumps.len());
                    dart_round_counts.push(0);
                    dart_state.tree_weights.push(1.0);
                    dart_state.dropped_per_round.push(Vec::new());
                }
                if let Some((new_w, _drop_factor, new_dropped_weights)) = dart_round_finalize {
                    for (i, &tree_id) in dropped_tree_ids.iter().enumerate() {
                        dart_state.tree_weights[tree_id] = new_dropped_weights[i];
                    }
                    dart_state.tree_weights.push(new_w);
                    dart_state.dropped_per_round.push(dropped_tree_ids.clone());
                } else {
                    dart_state.tree_weights.push(1.0);
                    dart_state.dropped_per_round.push(Vec::new());
                }
                round_start_offsets.push(stumps.len());
                dart_round_counts.push(candidate_round_stumps.len());
            }

            stumps_per_completed_round.push(candidate_round_stumps.len());
            stumps.extend(candidate_round_stumps);
            rounds_completed += 1;

            if stop_for_custom_metric_plateau {
                stop_reason = IterationStopReason::CustomMetricPlateau;
                break;
            }
            if stop_for_validation_plateau {
                stop_reason = IterationStopReason::ValidationLossPlateau;
                break;
            }
        }

        // Determine the best round for truncation: custom metric takes priority
        let truncation_round = if stop_reason == IterationStopReason::CustomMetricPlateau {
            best_custom_metric_round
        } else if stop_reason == IterationStopReason::ValidationLossPlateau {
            best_validation_round
        } else {
            None
        };

        if let Some(best_round) = truncation_round
            && best_round < rounds_completed
        {
            let kept_stumps =
                retained_stump_count_for_rounds(&stumps_per_completed_round, best_round);
            stumps.truncate(initial_stump_count + kept_stumps);
            stumps_per_completed_round.truncate(best_round);
            loss_per_completed_round.truncate(best_round);
            validation_loss_per_completed_round.truncate(best_round);
            custom_metric_per_round.truncate(best_round);
            sampled_rows_per_completed_round.truncate(best_round);
            sampled_features_per_completed_round.truncate(best_round);
            diagnostics_per_round.truncate(best_round);
            // DART: truncate the parallel DART arrays at the same point
            // as `stumps`, and recompute tree_weights from scratch using
            // only the kept rounds.  Round r's `apply_normalization` may
            // have rescaled weights of trees that themselves get
            // truncated in later rounds, so a naive
            // `dart_state.tree_weights.truncate(best_round)` would leave
            // the kept stumps stamped with weights mutated by trees that
            // no longer exist (the predictor would then return scores
            // that don't match the selected best iteration).  Replaying
            // through `apply_normalization` produces the exact weights
            // for the kept ensemble.
            if dart_params.is_some() {
                // `best_round` is in committed-round space, but the DART
                // arrays are indexed by effective_round_index which
                // includes phantom slots for skipped warmup rounds.
                // Map best_round → corresponding effective_round_index
                // by counting committed rounds in dart_round_counts.
                let mut committed_seen = 0usize;
                let mut truncate_at = dart_round_counts.len();
                for (idx, &count) in dart_round_counts.iter().enumerate() {
                    if count > 0 {
                        committed_seen += 1;
                        if committed_seen == best_round {
                            truncate_at = idx + 1;
                            break;
                        }
                    }
                }
                round_start_offsets.truncate(truncate_at);
                dart_round_counts.truncate(truncate_at);
                let kept_dropped = dart_state
                    .dropped_per_round
                    .iter()
                    .take(truncate_at)
                    .cloned()
                    .collect::<Vec<_>>();
                dart_state.tree_weights = vec![1.0; truncate_at];
                dart_state.dropped_per_round.truncate(truncate_at);
                for (r, dropped) in kept_dropped.iter().enumerate() {
                    if let Some((_, _, normalize_type, _)) = dart_params {
                        apply_normalization(
                            &mut dart_state.tree_weights,
                            dropped,
                            normalize_type,
                            r,
                        );
                    }
                }
            }
            rounds_completed = best_round;
            weak_improvement_rounds_committed =
                weak_improvement_rounds_committed.min(rounds_completed);
            current_loss = if rounds_completed == 0 {
                initial_loss
            } else {
                loss_per_completed_round[rounds_completed - 1]
            };
            current_validation_loss = if rounds_completed == 0 {
                initial_validation_loss
            } else {
                Some(validation_loss_per_completed_round[rounds_completed - 1])
            };
        }

        if experiment_leaf_refinement_enabled()
            && objective.supports_leaf_refinement()
            && gradient_neutralization_config(&self.params).is_none()
        {
            // Leaf refinement re-solves leaves against targets, so skip it for
            // per-round factor-neutralized gradients until refinement can apply
            // the same projection contract.
            let final_logical_round_count = round_index_offset
                .checked_add(rounds_completed)
                .ok_or_else(|| {
                    EngineError::ContractViolation(
                        "final logical round count overflow during leaf refinement".to_string(),
                    )
                })?;
            let projection_stumps_per_round =
                dense_projection_stump_counts(&stumps, final_logical_round_count)?;
            refine_regression_leaf_values(
                baseline_prediction,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
                binned_matrix,
                &mut stumps,
                &projection_stumps_per_round,
                controls.max_abs_leaf_value,
            )?;
            project_monotone_forest(
                &mut stumps,
                &projection_stumps_per_round,
                &self.params.monotone_constraints,
                controls.max_abs_leaf_value,
            )?;

            let mut refined_predictions = vec![baseline_prediction; active_dataset.row_count()];
            apply_tree_to_binned_predictions(
                &mut refined_predictions,
                binned_matrix,
                &stumps,
                raw_features_opt,
            )?;
            current_loss = objective.loss(
                &refined_predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
            )?;
            if let Some(last_loss) = loss_per_completed_round.last_mut() {
                *last_loss = current_loss;
            }
            if let Some(validation_ref) = validation {
                let mut refined_validation_predictions =
                    vec![baseline_prediction; validation_ref.dataset.row_count()];
                apply_tree_to_binned_predictions(
                    &mut refined_validation_predictions,
                    validation_ref.binned_matrix,
                    &stumps,
                    None,
                )?;
                current_validation_loss = Some(objective.loss(
                    &refined_validation_predictions,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?);
                if let (Some(last_validation_loss), Some(refined_validation_loss)) = (
                    validation_loss_per_completed_round.last_mut(),
                    current_validation_loss,
                ) {
                    *last_validation_loss = refined_validation_loss;
                }
            }
        }

        let morph_metadata = morph_state.as_ref().map(|ms| MorphMetadataPayload {
            config: ms.config,
            final_iteration: rounds_completed as u32,
            final_total: total_iterations,
            // v0.7.3: persist EMA so warm-start can resume from the
            // exact same EMA state rather than restarting cold.
            ema_stats: ms.ema_stats.clone(),
        });
        let dro_metadata = self
            .params
            .dro_config
            .map(|config| DroMetadataPayload { config });
        // Record per-feature training-set means only for piecewise-linear
        // artifacts.  SHAP consumes these as the interventional baseline for
        // linear leaves; constant-leaf models have no use for it.
        let feature_baseline = if self.params.leaf_model == LeafModelKind::Linear {
            compute_feature_means_from_matrix(
                &active_dataset.matrix.values,
                active_dataset.matrix.feature_count,
                active_dataset.row_count(),
            )
        } else {
            None
        };
        // DART: stamp per-stump tree_weight from dart_state.tree_weights
        // (indexed by tree_id encoded in stump.node_id). Non-DART fits
        // leave tree_weight at its 1.0 default. This is what the
        // artifact write path (`TrainedModel::to_artifact_bytes`) inspects
        // to decide whether to emit a DartTreeWeights section.
        if dart_params.is_some() {
            for stump in stumps.iter_mut() {
                let (tree_id, _) = decode_tree_node_id(stump.split.node_id);
                let idx = tree_id as usize;
                if let Some(&w) = dart_state.tree_weights.get(idx) {
                    stump.tree_weight = w;
                }
            }
        }

        let model = TrainedModel {
            baseline_prediction,
            feature_count: active_dataset.matrix.feature_count,
            stumps,
            categorical_state: None,
            node_debug_stats: None,
            objective: objective.objective_name().to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata,
            dro_metadata,
            feature_baseline,
            neutralization_metadata: None,
        };
        let final_loss = current_loss;

        Ok(IterationRunSummary {
            model,
            rounds_requested: controls.rounds,
            effective_round_cap,
            resolved_training_policy,
            rounds_completed,
            stop_reason,
            initial_loss,
            initial_validation_loss,
            loss_per_completed_round,
            validation_loss_per_completed_round,
            sampled_rows_per_completed_round,
            sampled_features_per_completed_round,
            best_validation_loss,
            best_validation_round,
            weak_improvement_rounds_committed,
            final_loss,
            final_validation_loss: current_validation_loss,
            custom_metric_per_round,
            custom_metric_name,
            diagnostics_per_round,
        })
    }

    pub fn fit_stub<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
    ) -> EngineResult<TrainRoundSummary> {
        self.fit_one_round(dataset, binned_matrix, backend, objective)
    }
}
