use alloygbm_categorical::fit_transform_target_encoder;
use alloygbm_core::{
    BinnedMatrix, BoostingMode, CategoricalStatePayloadV1, DatasetMatrix, DroMetadataPayload,
    FactorExposureMatrix, FeatureTile, GradientPair, HistogramBundle, LeafModelKind, LeafValue,
    LinearLeaf, MAX_PL_REGRESSORS, MorphMetadataPayload, NodeSlice, PartitionResult,
    SplitCandidate, TrainParams, TrainingDataset, TreeGrowth, leaf_effective_gradient,
    validate_train_params, validate_training_dataset,
};
#[cfg(test)]
use alloygbm_core::{MODEL_FORMAT_V1, ModelSectionKind, NodeStats};
#[cfg(test)]
use alloygbm_core::{ModelMetadata, deserialize_model_artifact_v1, serialize_model_artifact_v1};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

mod error;
pub use error::{EngineError, EngineResult};

mod env;
mod tree_node;
pub(crate) use tree_node::*;

use env::{
    experiment_force_manual_policy_enabled, experiment_leaf_refinement_enabled,
    split_selection_options_from_env,
};

pub mod dart;
pub use dart::{DartState, apply_normalization, select_dropouts};

pub mod shared_histogram;
pub use shared_histogram::{
    HistComponent, MultiOutputHistogram, build_multi_output_histogram_inplace,
    compute_multi_output_split_gain, subtract_multi_output_histogram,
};

pub mod joint;
pub use joint::{JointObjective, JointRoundResult, JointWarmStartState, build_joint_round};

mod morph_state;
pub use morph_state::{MorphState, resolve_lr_schedule};
pub(crate) use morph_state::MorphTreeContext;

mod factor;
pub(crate) use factor::FactorProjector;

mod split_options;
pub use split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, MorphContext, SplitSelectionOptions,
};

mod traits;
pub use traits::{BackendOps, ObjectiveOps, PerRoundMetricCallback};

mod objectives;
pub use objectives::{
    BinaryCrossEntropyObjective, GammaObjective, LambdaMARTObjective, MultiClassSoftmaxObjective,
    PairwiseRankingObjective, PoissonObjective, QuantileObjective, QueryRMSEObjective,
    SquaredErrorObjective, TweedieObjective, XeNDCGObjective, YetiRankObjective,
    compute_group_boundaries,
};
mod multiclass_model;
pub use multiclass_model::{MultiClassIterationRunSummary, MultiClassTrainedModel};

mod types;
mod warm_start;
pub use types::{
    ArtifactCompatibilityMode, ArtifactCompatibilityReport, CategoricalTargetEncodingSpec,
    FitContractEvaluation, IterationControls, IterationDiagnostics, IterationRunSummary,
    IterationStopReason, NodeDebugStats, TrainRoundSummary, TrainedStump, TrainingPolicyMode,
    ValidationDatasetRef,
};
pub(crate) use types::{IterationExecutionContext, PolicyFitRequest, gradient_l2_norm_only};
pub use warm_start::{MultiClassWarmStartState, WarmStartState};

mod trained_model;
pub use trained_model::TrainedModel;

mod artifact;

mod loss;
pub(crate) use loss::{binary_crossentropy_loss, squared_error_loss};

mod sampling;
pub(crate) use sampling::*;

mod tiling;
pub(crate) use tiling::*;

mod round;
pub(crate) use round::*;

mod leaf_refinement;
pub(crate) use leaf_refinement::*;

mod trainer;
pub(crate) use trainer::*;

/// Small epsilon added to leaf value denominators to prevent division by zero.
const LEAF_EPSILON: f32 = 1e-6;

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
                        projector.project_gradient_pairs_in_place(&mut tmp_buf)?;
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
                        lr: ms.lr_for_iter(effective_round),
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
            if loss_improvement < 0.0 {
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
                if loss_improvement < effective_min_loss_improvement {
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
                            apply_round_stumps_tree_walk(
                                &mut val_preds[class_k],
                                validation_ref.binned_matrix,
                                round_stumps,
                                val_raw,
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

    fn default_iteration_controls(&self, rounds: usize) -> EngineResult<IterationControls> {
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

        // Ranking objectives produce gradients and round-to-round loss deltas
        // that are orders of magnitude smaller than regression/classification,
        // because the loss is bounded by NDCG normalization and pairwise
        // weighting. The density-based min_split_gain floor and the
        // target-variance-scaled min_loss_improvement threshold were tuned for
        // regression losses and cause ranking training to exit after only a
        // handful of rounds with LossImprovementBelowThreshold. Disable both
        // guards for ranking so the user-supplied min_split_gain (default 0.0)
        // and n_estimators are honored.
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
        controls.min_loss_improvement = if is_ranking || row_count < 4_096 {
            0.0
        } else {
            (target_variance.max(1e-6) * 1e-5).min(0.01)
        };
        controls.max_consecutive_weak_improvements = if is_ranking || row_count < 4_096 {
            0
        } else if rounds <= 64 {
            1
        } else {
            3
        };

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
        if let Some(qa) = objective.quantile_alpha() {
            if !qa.is_finite() || qa <= 0.0 || qa >= 1.0 {
                return Err(EngineError::InvalidConfig(
                    "quantile_alpha must be finite and in (0.0, 1.0)".to_string(),
                ));
            }
            if self.params.leaf_model == LeafModelKind::Linear {
                return Err(EngineError::InvalidConfig(
                    "leaf_model='linear' is not supported with objective='quantile'".to_string(),
                ));
            }
            if matches!(self.params.boosting_mode, BoostingMode::Dart { .. }) {
                return Err(EngineError::InvalidConfig(
                    "boosting_mode='dart' is not supported with objective='quantile'".to_string(),
                ));
            }
            if self.params.morph_config.is_some() {
                return Err(EngineError::InvalidConfig(
                    "morph_config is not supported with objective='quantile'".to_string(),
                ));
            }
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
                projector.project_gradient_pairs_in_place(&mut gradient_buffer)?;
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
                    lr: ms.lr_for_iter(effective_round_index),
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
                let lr = morph_state
                    .as_ref()
                    .map(|ms| ms.lr_for_iter(effective_round_index))
                    .unwrap_or(self.params.learning_rate);
                refine_quantile_leaf_values(
                    &mut candidate_round_stumps,
                    binned_matrix,
                    &predictions,
                    &active_dataset.targets,
                    active_dataset.sample_weights.as_deref(),
                    alpha,
                    lr,
                    controls.max_abs_leaf_value,
                )?;
                candidate_predictions.copy_from_slice(&predictions);
                apply_round_stumps_tree_walk(
                    &mut candidate_predictions,
                    binned_matrix,
                    &candidate_round_stumps,
                    raw_features_opt,
                )?;
            }

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
            let objective_is_ranking = objective.requires_group_id();
            if !objective_is_ranking && loss_improvement < 0.0 {
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
                if !objective_is_ranking && loss_improvement < effective_min_loss_improvement {
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
                    apply_round_stumps_tree_walk(
                        &mut next_validation_predictions,
                        validation_ref.binned_matrix,
                        &candidate_round_stumps,
                        Some((
                            &validation_ref.dataset.matrix.values as &[f32],
                            validation_ref.dataset.matrix.feature_count,
                        )),
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


fn apply_single_categorical_target_encoding(
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

fn encode_bins_from_encoded_values(encoded_values: &[f32]) -> EngineResult<(Vec<u8>, u16)> {
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
fn find_best_split_dispatch<B: BackendOps>(
    backend: &B,
    histograms: &HistogramBundle,
    options: SplitSelectionOptions,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<&MorphTreeContext<'_>>,
    factor_context: Option<&FactorSplitContext<'_>>,
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
fn build_tree_level_wise<B: BackendOps>(
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
struct PendingSplit {
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
fn build_tree_leaf_wise<B: BackendOps>(
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
fn subtract_histogram_bundle_into(
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

fn subtract_histogram_bundle(
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

fn validate_iteration_controls(controls: IterationControls) -> EngineResult<()> {
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
mod tests;
