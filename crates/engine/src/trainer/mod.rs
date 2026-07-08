//! Trainer module — gradient-boosting iteration controller.

mod interaction;
mod policy;
mod tree_build;
mod validate;

pub(crate) use interaction::InteractionConstraintIndex;
#[cfg(test)]
pub(crate) use policy::should_apply_auto_split_l2;
pub(crate) use policy::split_selection_options_for_training;
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

        let histograms = backend.build_histograms(
            binned_matrix,
            &fit_contract.gradients,
            &root_node,
            &feature_tiles,
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
        if experiment_force_manual_policy_enabled() {
            return self.default_iteration_controls(rounds);
        }
        match policy_mode {
            TrainingPolicyMode::Manual => self.default_iteration_controls(rounds),
            TrainingPolicyMode::Auto => {
                self.auto_iteration_controls(dataset, binned_matrix, rounds, is_ranking)
            }
        }
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
        let split_options =
            split_selection_options_for_training(&self.params, None, dataset, binned_matrix)?;
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
                    apply_round_stumps_tree_walk(
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
                    apply_round_stumps_tree_walk(
                        &mut val_preds[class_k],
                        validation_ref.binned_matrix,
                        &class_stumps[class_k],
                        val_raw,
                    )?;
                }
            }
        }

        let initial_loss = objective.loss(
            &class_predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
        )?;
        let initial_validation_loss = if let Some(v) = validation {
            let val_preds: Vec<Vec<f32>> = baselines
                .iter()
                .map(|&b| vec![b; v.dataset.row_count()])
                .collect();
            Some(objective.loss(
                &val_preds,
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
            // Reconstruct per-class `dart_round_start_offsets` and
            // `dart_round_counts` for warm-start by grouping
            // `class_stumps[class_k]` by tree_id (decoded from
            // `stump.split.node_id`). All stumps of the same class-tree
            // share a tree_id, and distinct tree_ids appear in
            // round-order, so this gives the contiguous slice
            // boundaries the dropout/normalize code needs.
            for class_k in 0..k {
                let class_total = class_stumps[class_k].len();
                let mut i = 0_usize;
                while i < class_total {
                    let (tid_first, _) =
                        decode_tree_node_id(class_stumps[class_k][i].split.node_id);
                    let start = i;
                    let mut j = i + 1;
                    while j < class_total {
                        let (tid, _) = decode_tree_node_id(class_stumps[class_k][j].split.node_id);
                        if tid != tid_first {
                            break;
                        }
                        j += 1;
                    }
                    dart_round_start_offsets[class_k].push(start);
                    dart_round_counts[class_k].push(j - start);
                    i = j;
                }
                // Pad with phantom (count=0) entries up to
                // round_index_offset so the array length matches the
                // warm-start round count.
                while dart_round_start_offsets[class_k].len() < round_index_offset {
                    dart_round_start_offsets[class_k].push(class_total);
                    dart_round_counts[class_k].push(0);
                }
            }
        }

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
            let dropped_tree_indices: Vec<usize> =
                if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round,
                    );
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
                        apply_weighted_round_to_predictions(
                            &mut class_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            -w_old,
                        )?;
                    }
                    drops
                } else {
                    Vec::new()
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

                let raw_fv = &dataset.matrix.values;
                let (round_stumps, _round_stop) = if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        effective_round,
                        &feature_tiles,
                        split_options,
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
                        split_options,
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
            let dart_round_finalize: Option<(f32, Vec<f32>)> =
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
                    // Re-add each dropped tree's WHOLE slice at the rescaled
                    // weight to BOTH class_predictions (so post-round
                    // commit captures the full ensemble) AND
                    // class_candidate_predictions (so candidate_loss is
                    // computed against the correct full ensemble).
                    for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
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
                        let w_new = new_dropped_weights[i];
                        apply_weighted_round_to_predictions(
                            &mut class_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            w_new,
                        )?;
                        apply_weighted_round_to_predictions(
                            &mut class_candidate_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            w_new,
                        )?;
                    }
                    Some((new_w, new_dropped_weights))
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
                if let Some((new_w, new_dropped_weights)) = dart_round_finalize.as_ref() {
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
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            &stump_slice,
                            val_raw,
                            -w_old,
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
                    // 3. Re-add each dropped class-tree at its new weight.
                    for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
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
                        let w_new = new_dropped_weights[i];
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            &stump_slice,
                            val_raw,
                            w_new,
                        )?;
                    }
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
            if let Some((new_w, new_dropped_weights)) = dart_round_finalize.as_ref() {
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
            loss_per_completed_round.truncate(best_round);
            validation_loss_per_completed_round.truncate(best_round);
            sampled_rows_per_completed_round.truncate(best_round);
            sampled_features_per_completed_round.truncate(best_round);
            diagnostics_per_round.truncate(best_round);
            rounds_completed = best_round;
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
        let split_options = split_selection_options_for_training(
            &self.params,
            execution.policy_mode,
            active_dataset,
            binned_matrix,
        )?;

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
        let mut stumps = initial_stumps;
        let initial_stump_count = stumps.len();
        // `stumps_per_completed_round` stays NEW-ROUND-ONLY (its original
        // semantics): downstream consumers (validation early-stopping
        // truncation via `retained_stump_count_for_rounds`, leaf refinement,
        // DART replay truncation) index into it with a `best_round` value
        // relative to the new fit and assume entry `i` holds the i-th
        // newly-committed round's stump count.
        let mut stumps_per_completed_round: Vec<usize> = Vec::new();
        // Separate vector for warm-start prior-round counts. v0.10.0 review
        // follow-up: this used to live in `stumps_per_completed_round`, but
        // that broke `best_round`-indexed truncation when warm_start combined
        // with eval_set early stopping (kept counts from old rounds rather
        // than new ones). Now kept as its own local consumed only by the
        // DART-state seeding + round_start_offsets/dart_round_counts
        // pre-population blocks below.
        let mut initial_stumps_per_round: Vec<usize> = Vec::new();
        if !stumps.is_empty() {
            let mut current_tree_id = decode_tree_node_id(stumps[0].split.node_id).0;
            let mut current_count = 0usize;
            for stump in &stumps {
                let tree_id = decode_tree_node_id(stump.split.node_id).0;
                if tree_id != current_tree_id {
                    initial_stumps_per_round.push(current_count);
                    current_tree_id = tree_id;
                    current_count = 0;
                }
                current_count += 1;
            }
            initial_stumps_per_round.push(current_count);
        }
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
            // or no snapshot was provided. Uses `initial_stumps_per_round`
            // (warm-start prior-round counts) — distinct from
            // `stumps_per_completed_round` (new-round counts) so downstream
            // best_round-indexed truncation continues to work correctly.
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
        // count.  Together they slice into `stumps` for the DART
        // dropout subtract/replay step.  Stays empty for non-DART
        // fits.  Keep separate from the pre-existing
        // `stumps_per_completed_round` (committed-only) so we don't
        // perturb downstream consumers like
        // `retained_stump_count_for_rounds`.
        let mut round_start_offsets: Vec<usize> = Vec::new();
        let mut dart_round_counts: Vec<usize> = Vec::new();

        // v0.10.0: DART + warm_start — pre-populate round_start_offsets +
        // dart_round_counts from the warm-start tree shapes so the dropout
        // step can correctly slice into `stumps` for each prior tree. Stays
        // a no-op for non-DART or cold fits. Uses `initial_stumps_per_round`
        // (warm-start prior-round counts) so `stumps_per_completed_round`
        // can stay new-round-only for downstream best_round indexing.
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
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round_index,
                    );
                    if !drops.is_empty() {
                        dart_predictions_backup = Some(predictions.clone());
                        dart_validation_backup = validation_predictions.clone();
                    }
                    for &tree_id in &drops {
                        let w_old = dart_state.tree_weights[tree_id];
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut predictions,
                            binned_matrix,
                            stump_slice,
                            raw_features_opt,
                            -w_old,
                        )?;
                        if let (Some(vp), Some(validation_ref)) =
                            (validation_predictions.as_mut(), validation)
                        {
                            let val_raw = Some((
                                &validation_ref.dataset.matrix.values as &[f32],
                                validation_ref.dataset.matrix.feature_count,
                            ));
                            apply_weighted_round_to_predictions(
                                vp,
                                validation_ref.binned_matrix,
                                stump_slice,
                                val_raw,
                                -w_old,
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
                        split_options,
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
                        split_options,
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
            // `dart_round_finalize = Some((new_w, new_dropped_weights))`
            // on a DART round; `None` otherwise. The commit path
            // consumes this to update `dart_state`.
            let dart_round_finalize: Option<(f32, Vec<f32>)> =
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
                    // Step 2: re-add each dropped tree to candidate_predictions
                    // at its new (post-normalize) weight.
                    let mut new_dropped_weights = Vec::with_capacity(dropped_tree_ids.len());
                    for &tree_id in &dropped_tree_ids {
                        let w_old = dart_state.tree_weights[tree_id];
                        let w_new = w_old * drop_factor;
                        new_dropped_weights.push(w_new);
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut candidate_predictions,
                            binned_matrix,
                            stump_slice,
                            raw_features_opt,
                            w_new,
                        )?;
                    }
                    Some((new_w, new_dropped_weights))
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
                if let Some((new_w, new_dropped_weights)) = &dart_round_finalize {
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
                    for (i, &tree_id) in dropped_tree_ids.iter().enumerate() {
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut next_validation_predictions,
                            validation_ref.binned_matrix,
                            stump_slice,
                            val_raw,
                            new_dropped_weights[i],
                        )?;
                    }
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
                if let Some((new_w, new_dropped_weights)) = dart_round_finalize {
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
            refine_regression_leaf_values(
                baseline_prediction,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
                binned_matrix,
                &mut stumps,
                &stumps_per_completed_round,
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
