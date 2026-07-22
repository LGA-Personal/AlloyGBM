use crate::callbacks::{CustomPythonMetricCallback, CustomPythonObjective};
use crate::categorical_bridge::{
    apply_bridge_pre_target_neutralization, apply_categorical_encoding_to_training_matrices_multi,
    apply_categorical_encoding_to_validation_matrices_multi, flatten_rows,
    resolve_categorical_spec, resolve_categorical_specs_from_params,
    validate_bridge_pre_target_neutralization_support,
};
use crate::errors::engine_error_to_pyerr;
use crate::params::{
    build_train_params, parse_boosting_mode, parse_dro_config, parse_factor_exposure_matrix,
    parse_leaf_model, parse_leaf_solver, parse_morph_config_from_pydict,
    parse_neutralization_config, parse_training_policy, parse_tree_growth,
    validate_neutralization_leaf_model,
};
use crate::pyclasses::{NativeTrainingResult, NativeTrainingSummary, diagnostics_to_native};
use crate::quantization::{
    ContinuousBinningStrategy, parse_continuous_binning_strategy,
    prepare_training_matrices_from_dense_values, prepare_validation_matrices_from_dense_values,
};
use crate::{DEFAULT_TRAIN_ROUNDS, MAX_SUPPORTED_TRAIN_ROUNDS};

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{
    BinnedLayout, DatasetMatrix, FactorExposureMatrix, NeutralizationKind, TrainParams, TreeGrowth,
};
use alloygbm_engine::{
    BinaryCrossEntropyObjective, CategoricalFeatureInfo, CategoricalTargetEncodingSpec,
    EngineError, GammaObjective, IterationRunSummary, LambdaMARTObjective,
    MultiClassIterationRunSummary, MultiClassSoftmaxObjective, MultiClassTrainedModel,
    MultiClassWarmStartState, ObjectiveOps, PairwiseRankingObjective, PerRoundMetricCallback,
    PoissonObjective, QuantileObjective, QueryRMSEObjective, SquaredErrorObjective, TrainedModel,
    Trainer, TrainingPolicyMode, TweedieObjective, WarmStartState, XeNDCGObjective,
    YetiRankObjective,
};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::time::Instant;

fn build_native_training_summary_from_multiclass(
    summary: &MultiClassIterationRunSummary,
    bridge_prepare_seconds: f64,
    native_train_seconds: f64,
    objective: &str,
) -> NativeTrainingSummary {
    NativeTrainingSummary {
        rounds_requested: summary.rounds_requested,
        rounds_completed: summary.rounds_completed,
        best_validation_round: summary
            .best_validation_round
            .and_then(|round| round.checked_sub(1)),
        best_validation_loss: summary.best_validation_loss,
        train_rmse: summary
            .loss_per_completed_round
            .iter()
            .map(|loss| loss.max(0.0).sqrt())
            .collect(),
        validation_rmse: summary
            .validation_loss_per_completed_round
            .iter()
            .map(|loss| loss.max(0.0).sqrt())
            .collect(),
        train_loss: summary.loss_per_completed_round.clone(),
        validation_loss: summary.validation_loss_per_completed_round.clone(),
        objective: objective.to_string(),
        stop_reason: format!("{:?}", summary.stop_reason),
        bridge_prepare_seconds,
        native_train_seconds,
        custom_metric_values: summary.custom_metric_per_round.clone(),
        custom_metric_name: summary.custom_metric_name.clone(),
        diagnostics_per_round: diagnostics_to_native(&summary.diagnostics_per_round),
    }
}

fn build_native_training_summary(
    summary: &IterationRunSummary,
    bridge_prepare_seconds: f64,
    native_train_seconds: f64,
    objective: &str,
) -> NativeTrainingSummary {
    NativeTrainingSummary {
        rounds_requested: summary.rounds_requested,
        rounds_completed: summary.rounds_completed,
        best_validation_round: summary
            .best_validation_round
            .and_then(|round| round.checked_sub(1)),
        best_validation_loss: summary.best_validation_loss,
        train_rmse: summary
            .loss_per_completed_round
            .iter()
            .map(|loss| loss.max(0.0).sqrt())
            .collect(),
        validation_rmse: summary
            .validation_loss_per_completed_round
            .iter()
            .map(|loss| loss.max(0.0).sqrt())
            .collect(),
        train_loss: summary.loss_per_completed_round.clone(),
        validation_loss: summary.validation_loss_per_completed_round.clone(),
        objective: objective.to_string(),
        stop_reason: format!("{:?}", summary.stop_reason),
        bridge_prepare_seconds,
        native_train_seconds,
        custom_metric_values: summary.custom_metric_per_round.clone(),
        custom_metric_name: summary.custom_metric_name.clone(),
        diagnostics_per_round: diagnostics_to_native(&summary.diagnostics_per_round),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact_with_summary_dense_impl(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    factor_exposures: Option<FactorExposureMatrix>,
    validation_values: Option<&[f32]>,
    validation_row_count: Option<usize>,
    validation_targets: Option<&[f32]>,
    validation_sample_weights: Option<Vec<f32>>,
    validation_group_id: Option<Vec<u32>>,
    mut params: TrainParams,
    rounds: usize,
    time_index: Option<Vec<i64>>,
    validation_time_index: Option<Vec<i64>>,
    categorical_specs: Vec<CategoricalTargetEncodingSpec>,
    validation_categorical_values_list: Vec<Vec<String>>,
    training_policy: TrainingPolicyMode,
    store_node_debug_stats: bool,
    continuous_binning_strategy: ContinuousBinningStrategy,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
    init_artifact_bytes: Option<&[u8]>,
    num_classes: Option<usize>,
    custom_objective_fn: Option<Py<PyAny>>,
    custom_loss_fn: Option<Py<PyAny>>,
    custom_metric_fn: Option<Py<PyAny>>,
    max_cat_threshold: usize,
) -> Result<NativeTrainingResult, EngineError> {
    let bridge_start = Instant::now();
    // Warm-start with neutralization is supported as of v0.7.1.  The Python
    // wrapper enforces that callers supply the same `factor_exposures`
    // matrix used for the initial fit, and the engine re-checks via
    // `validate_warm_start_neutralization_contract` that the dataset carries
    // exposures.  `pre_target` is handled below by residualizing the
    // (already-original) targets again — idempotent against the same
    // exposures — so resumed training sees the same residualized target
    // stream as a fresh fit of `N + M` rounds.
    let is_linear_leaf = params.leaf_model == alloygbm_core::LeafModelKind::Linear;
    // Dense float values are needed for categorical target encoding.  For linear-leaf
    // training we need the raw feature values separately (see post-processing below).
    let need_dense_values = !categorical_specs.is_empty();
    let mut prepared = prepare_training_matrices_from_dense_values(
        values,
        row_count,
        feature_count,
        targets,
        time_index,
        sample_weights,
        group_id,
        continuous_binning_strategy,
        continuous_binning_max_bins,
        quantile_sketch_max_rows,
        need_dense_values,
        BinnedLayout::ColumnMajor,
    )?;
    prepared.dataset.factor_exposures = factor_exposures;

    // For linear-leaf training the engine reads `dataset.matrix.values` as raw (float)
    // feature values inside `build_linear_histograms_cpu`.  The preparation step above
    // stores *bin indices* as f32 when `need_dense_values=true` (for categorical
    // encoding), so we must replace the DatasetMatrix with the original floats here.
    // Categorical encoding runs afterwards and will overwrite its own columns.
    if is_linear_leaf {
        prepared.dataset.matrix = DatasetMatrix::new(row_count, feature_count, values.to_vec())?;
    }

    if let Some(config) = params.neutralization_config
        && config.kind == NeutralizationKind::PreTarget
    {
        validate_bridge_pre_target_neutralization_support(
            &params,
            objective,
            custom_objective_fn.as_ref(),
            validation_targets.is_some(),
        )?;
        apply_bridge_pre_target_neutralization(&mut prepared, config)?;
        params.neutralization_config = None;
    }

    let training_targets_for_validation = prepared.dataset.targets.clone();
    let training_time_index_for_validation = prepared.dataset.time_index.clone();

    // Split categorical features into native (low cardinality) and target-encoded (high cardinality).
    let mut native_cat_mappings: std::collections::HashMap<
        usize,
        std::collections::HashMap<String, u32>,
    > = std::collections::HashMap::new();
    let mut native_cat_infos: Vec<CategoricalFeatureInfo> = Vec::new();
    let mut target_encoding_specs: Vec<CategoricalTargetEncodingSpec> = Vec::new();

    if !categorical_specs.is_empty() && max_cat_threshold > 0 {
        for spec in &categorical_specs {
            // Count unique categories for this feature.
            let mut unique_cats: Vec<String> = spec.values.to_vec();
            unique_cats.sort();
            unique_cats.dedup();
            let num_unique = unique_cats.len();

            if num_unique <= max_cat_threshold && num_unique >= 2 {
                // Native categorical split: map categories to integer IDs.
                let cat_to_id: std::collections::HashMap<String, u32> = unique_cats
                    .iter()
                    .enumerate()
                    .map(|(i, cat)| (cat.clone(), i as u32))
                    .collect();

                // Re-bin the column in the binned matrix: category name → integer bin ID.
                let fi = spec.feature_index;
                let missing_bin = prepared.binned_matrix.missing_bin();
                // Guard: category IDs must not collide with the missing-value sentinel.
                if (num_unique as u16) > missing_bin {
                    return Err(EngineError::ContractViolation(format!(
                        "Native categorical feature {} has {} categories which would collide with the missing-value sentinel (bin {}). Reduce max_cat_threshold or increase continuous_binning_max_bins.",
                        fi, num_unique, missing_bin
                    )));
                }
                for (row_idx, cat_name) in spec.values.iter().enumerate() {
                    let bin_val = cat_to_id
                        .get(cat_name)
                        .map(|&id| id as u16)
                        .unwrap_or(missing_bin);
                    prepared.binned_matrix.set_bin(row_idx, fi, bin_val);
                }
                // Update max_bin if categories exceed current max.
                let cat_max_bin = (num_unique - 1) as u16;
                if cat_max_bin > prepared.binned_matrix.max_bin {
                    prepared.binned_matrix.max_bin = cat_max_bin;
                }

                native_cat_infos.push(CategoricalFeatureInfo {
                    feature_index: fi,
                    num_categories: num_unique,
                });
                native_cat_mappings.insert(fi, cat_to_id);
            } else {
                // Falls back to target encoding.
                target_encoding_specs.push(spec.clone());
            }
        }
    } else {
        target_encoding_specs = categorical_specs.clone();
    }

    let mut categorical_state = None;
    if !target_encoding_specs.is_empty() {
        let (encoded_prepared, state) = apply_categorical_encoding_to_training_matrices_multi(
            prepared,
            &target_encoding_specs,
        )?;
        prepared = encoded_prepared;
        categorical_state = Some(state);
    }

    let validation_prepared = match (validation_values, validation_row_count, validation_targets) {
        (Some(values), Some(row_count), Some(targets)) => {
            if feature_count == 0 {
                return Err(EngineError::ContractViolation(
                    "validation feature_count must be greater than 0".to_string(),
                ));
            }
            let mut prepared_validation = prepare_validation_matrices_from_dense_values(
                values,
                row_count,
                feature_count,
                targets,
                validation_time_index,
                validation_sample_weights,
                validation_group_id,
                continuous_binning_strategy,
                &prepared.metadata,
                need_dense_values,
                continuous_binning_max_bins,
            )?;

            if !categorical_specs.is_empty() {
                if validation_categorical_values_list.len() != categorical_specs.len() {
                    return Err(EngineError::ContractViolation(format!(
                        "validation categorical values list length {} does not match categorical specs count {}",
                        validation_categorical_values_list.len(),
                        categorical_specs.len()
                    )));
                }

                // Apply native categorical binning to validation for native-split features.
                let val_row_count = prepared_validation.dataset.matrix.row_count;
                for (spec, val_cats) in categorical_specs
                    .iter()
                    .zip(&validation_categorical_values_list)
                {
                    if let Some(cat_to_id) = native_cat_mappings.get(&spec.feature_index) {
                        let fi = spec.feature_index;
                        let missing_bin = prepared_validation.binned_matrix.missing_bin();
                        // Update max_bin to match training matrix for this native categorical feature.
                        // (The sentinel collision was already validated on the training path.)
                        let num_unique = cat_to_id.len();
                        let cat_max_bin = (num_unique - 1) as u16;
                        if cat_max_bin > prepared_validation.binned_matrix.max_bin {
                            prepared_validation.binned_matrix.max_bin = cat_max_bin;
                        }
                        for (row_idx, cat_name) in val_cats.iter().enumerate() {
                            if row_idx < val_row_count {
                                let bin_val = cat_to_id
                                    .get(cat_name)
                                    .map(|&id| id as u16)
                                    .unwrap_or(missing_bin);
                                prepared_validation
                                    .binned_matrix
                                    .set_bin(row_idx, fi, bin_val);
                            }
                        }
                    }
                }

                // Apply target encoding only for features that weren't native-split.
                if !target_encoding_specs.is_empty() {
                    let validation_te_specs: Vec<CategoricalTargetEncodingSpec> =
                        target_encoding_specs
                            .iter()
                            .map(|te_spec| {
                                // Find the matching validation categorical values for this feature.
                                let orig_idx = categorical_specs
                                    .iter()
                                    .position(|s| s.feature_index == te_spec.feature_index)
                                    .expect(
                                        "target encoding spec must correspond to an original spec",
                                    );
                                CategoricalTargetEncodingSpec {
                                    feature_index: te_spec.feature_index,
                                    values: validation_categorical_values_list[orig_idx].clone(),
                                    config: te_spec.config.clone(),
                                }
                            })
                            .collect();
                    prepared_validation = apply_categorical_encoding_to_validation_matrices_multi(
                        prepared_validation,
                        &target_encoding_specs,
                        &validation_te_specs,
                        &training_targets_for_validation,
                        training_time_index_for_validation.as_deref(),
                    )?;
                }
            }
            Some(prepared_validation)
        }
        (None, None, None) => None,
        _ => {
            return Err(EngineError::ContractViolation(
                "validation rows, targets, and row_count must be provided together".to_string(),
            ));
        }
    };

    // Warm-start: load existing single-output model if init_artifact_bytes is provided.
    // Multiclass warm-start is handled separately in the multiclass_softmax branch
    // because multiclass artifacts have a different format (MultiClassTrees section).
    let warm_start_state = if objective != "multiclass_softmax" {
        if let Some(init_bytes) = init_artifact_bytes {
            let init_model = TrainedModel::from_artifact_bytes(init_bytes)?;
            if init_model.feature_count != feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "init_model feature_count {} does not match training data feature_count {}",
                    init_model.feature_count, feature_count,
                )));
            }
            let initial_rounds = init_model.rounds_completed();
            // v0.7.3: pull the persisted EMA snapshot (if any) so a
            // MorphBoost warm-start resumes from the previous fit's
            // EMA state rather than restarting it cold.  v1 artifacts
            // (pre-v0.7.3) decode with empty `ema_stats`; we keep
            // `initial_ema_stats = None` for that case so the engine
            // falls back to a cold EMA (preserving prior behaviour).
            let initial_ema_stats = init_model
                .morph_metadata
                .as_ref()
                .filter(|m| !m.ema_stats.is_empty())
                .map(|m| m.ema_stats.clone());
            // v0.10.0+: when the prior fit used DART, capture per-stump
            // tree_weight so the continuation can seed dart_state.tree_weights.
            // Detected by any stump carrying a non-default weight.
            let initial_dart_tree_weights = if init_model
                .stumps
                .iter()
                .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON)
            {
                Some(init_model.stumps.iter().map(|s| s.tree_weight).collect())
            } else {
                None
            };
            Some(WarmStartState {
                baseline_prediction: init_model.baseline_prediction,
                stumps: init_model.stumps,
                initial_rounds_completed: initial_rounds,
                initial_ema_stats,
                initial_dart_tree_weights,
            })
        } else {
            None
        }
    } else {
        None
    };

    let bridge_prepare_seconds = bridge_start.elapsed().as_secs_f64();
    let user_seed = params.seed;
    let tweedie_variance_power = params.tweedie_variance_power;
    let poisson_max_delta_step = params.poisson_max_delta_step;
    let quantile_alpha = params.quantile_alpha;
    let trainer = Trainer::new(params)?.with_categorical_features(native_cat_infos.clone());
    let backend = CpuBackend;
    let native_start = Instant::now();

    // Build the custom metric callback if provided
    let custom_metric_cb = if let Some(metric_fn) = custom_metric_fn {
        Some(CustomPythonMetricCallback::new(metric_fn)?)
    } else {
        None
    };
    let custom_metric_ref: Option<&dyn PerRoundMetricCallback> = custom_metric_cb
        .as_ref()
        .map(|cb| cb as &dyn PerRoundMetricCallback);

    macro_rules! run_training {
        ($obj:expr) => {{
            let controls = trainer.iteration_controls_for_policy_ext(
                &prepared.dataset,
                &prepared.binned_matrix,
                rounds,
                training_policy,
                $obj.requires_group_id(),
            )?;
            if let Some(warm_start) = warm_start_state.clone() {
                if let Some(validation_prepared) = validation_prepared.as_ref() {
                    trainer.fit_iterations_warm_start_with_validation_and_metric(
                        &prepared.dataset,
                        &prepared.binned_matrix,
                        alloygbm_engine::ValidationDatasetRef {
                            dataset: &validation_prepared.dataset,
                            binned_matrix: &validation_prepared.binned_matrix,
                        },
                        &backend,
                        $obj,
                        controls,
                        warm_start,
                        custom_metric_ref,
                    )?
                } else {
                    trainer.fit_iterations_warm_start(
                        &prepared.dataset,
                        &prepared.binned_matrix,
                        &backend,
                        $obj,
                        controls,
                        warm_start,
                    )?
                }
            } else if let Some(validation_prepared) = validation_prepared.as_ref() {
                trainer.fit_iterations_with_validation_and_metric(
                    &prepared.dataset,
                    &prepared.binned_matrix,
                    alloygbm_engine::ValidationDatasetRef {
                        dataset: &validation_prepared.dataset,
                        binned_matrix: &validation_prepared.binned_matrix,
                    },
                    &backend,
                    $obj,
                    controls,
                    custom_metric_ref,
                )?
            } else {
                trainer.fit_iterations_with_summary(
                    &prepared.dataset,
                    &prepared.binned_matrix,
                    &backend,
                    $obj,
                    controls,
                )?
            }
        }};
    }

    let mut summary = match objective {
        "squared_error" => run_training!(&SquaredErrorObjective),
        "binary_crossentropy" => run_training!(&BinaryCrossEntropyObjective),
        "poisson" => {
            let obj = PoissonObjective::new(poisson_max_delta_step);
            run_training!(&obj)
        }
        "gamma" => run_training!(&GammaObjective),
        "quantile" => {
            let obj = QuantileObjective {
                alpha: quantile_alpha,
            };
            run_training!(&obj)
        }
        "tweedie" => {
            let obj = TweedieObjective::new(tweedie_variance_power)?;
            run_training!(&obj)
        }
        "queryrmse" | "rank_pairwise" | "rank_ndcg" | "rank_xendcg" | "yetirank" => {
            let group_id = prepared.dataset.group_id.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "objective '{objective}' requires group_id to be provided"
                ))
            })?;
            let val_group_id = validation_prepared
                .as_ref()
                .and_then(|vp| vp.dataset.group_id.as_ref());
            match objective {
                "queryrmse" => {
                    let mut obj = QueryRMSEObjective::new(group_id);
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                "rank_pairwise" => {
                    let mut obj =
                        PairwiseRankingObjective::new_with_sigma(group_id, ranking_sigma)?;
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                "rank_ndcg" => {
                    let mut obj = LambdaMARTObjective::new_with_options(
                        group_id,
                        ranking_sigma,
                        lambdarank_truncation_level,
                        lambdarank_normalize,
                    )?;
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                "rank_xendcg" => {
                    let mut obj = XeNDCGObjective::new(group_id);
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                "yetirank" => {
                    let mut obj =
                        YetiRankObjective::new_with_sigma(group_id, 10, user_seed, ranking_sigma)?;
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                _ => unreachable!(),
            }
        }
        "multiclass_softmax" => {
            let k = num_classes.ok_or_else(|| {
                EngineError::InvalidConfig(
                    "multiclass_softmax objective requires num_classes to be specified".to_string(),
                )
            })?;
            if k < 2 {
                return Err(EngineError::InvalidConfig(format!(
                    "multiclass_softmax requires num_classes >= 2, got {k}"
                )));
            }
            let mc_obj = MultiClassSoftmaxObjective::new(k)?;
            let controls = trainer.iteration_controls_for_policy(
                &prepared.dataset,
                &prepared.binned_matrix,
                rounds,
                training_policy,
            )?;

            // Build multiclass warm-start state from init_artifact_bytes if available
            let mc_warm_start = if let Some(init_bytes) = init_artifact_bytes {
                let init_mc_model = MultiClassTrainedModel::from_artifact_bytes(init_bytes)?;
                if init_mc_model.feature_count != feature_count {
                    return Err(EngineError::ContractViolation(format!(
                        "init_model feature_count {} does not match training data feature_count {}",
                        init_mc_model.feature_count, feature_count,
                    )));
                }
                if init_mc_model.num_classes != k {
                    return Err(EngineError::ContractViolation(format!(
                        "init_model num_classes {} does not match training num_classes {}",
                        init_mc_model.num_classes, k,
                    )));
                }
                let initial_rounds = init_mc_model.rounds_completed();
                // v0.7.3: pull the persisted EMA snapshot (one entry
                // per class) so MorphBoost warm-start resumes from
                // the previous fit's EMA.  v1 artifacts decode with
                // empty `ema_stats`; treat that as "no snapshot" so
                // the engine falls back to a cold EMA.
                let initial_ema_stats = init_mc_model
                    .morph_metadata
                    .as_ref()
                    .filter(|m| !m.ema_stats.is_empty())
                    .map(|m| m.ema_stats.clone());
                // v0.10.1: capture multiclass DART tree_weights from
                // the prior fit, if any.  Flat layout: round-major ×
                // class-k (matches
                // `MultiClassWarmStartState::initial_dart_tree_weights`).
                //
                // PR review follow-up: each class's stumps are stored
                // FLAT across all rounds (a level-wise tree with
                // depth>=2 contributes multiple stumps per round), so
                // `class_stumps[class_k][r]` is the r-th *stump* —
                // NOT the r-th *tree*. Walk each class's stump list
                // grouping by `tree_id` (decoded from
                // `stump.split.node_id`) to recover one weight per
                // (round, class) tree.  This mirrors the engine-side
                // warm-start reconstruction in
                // `fit_multiclass_iterations_impl` so the flat
                // `r * K + class_k` indexing the engine consumes
                // stays consistent.
                let k_for_warm = init_mc_model.num_classes;
                const TREE_NODE_STRIDE: u32 = 1 << 20;
                let any_dart_weight = init_mc_model
                    .class_stumps
                    .iter()
                    .flat_map(|s| s.iter())
                    .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON);
                let initial_dart_tree_weights = if any_dart_weight {
                    // Per-class list of per-tree weights, in
                    // round-order (which matches the encounter order
                    // of distinct `tree_id`s in `class_stumps[k]`).
                    let mut per_class_tree_weights: Vec<Vec<f32>> = vec![Vec::new(); k_for_warm];
                    for (tree_weights, stumps) in per_class_tree_weights
                        .iter_mut()
                        .take(k_for_warm)
                        .zip(init_mc_model.class_stumps.iter())
                    {
                        let n = stumps.len();
                        let mut i = 0_usize;
                        while i < n {
                            let tid_first = stumps[i].split.node_id / TREE_NODE_STRIDE;
                            // Take the first stump's weight as the
                            // per-tree weight (the engine's
                            // `apply_dart_tree_weights` predictor
                            // helper uses the same convention).
                            tree_weights.push(stumps[i].tree_weight);
                            // Advance past every stump that shares
                            // this tree_id.
                            let mut j = i + 1;
                            while j < n && stumps[j].split.node_id / TREE_NODE_STRIDE == tid_first {
                                j += 1;
                            }
                            i = j;
                        }
                    }
                    // Assemble the flat round-major × class-k array.
                    // Phantom (zero-tree) rounds get a placeholder
                    // weight of 1.0 so the array length equals
                    // `initial_rounds * K` — matches the engine
                    // contract and is consistent with how the
                    // single-output DART path treats phantom
                    // rounds-skipped-during-warmup.
                    let mut flat: Vec<f32> = Vec::with_capacity(initial_rounds * k_for_warm);
                    for r in 0..initial_rounds {
                        for tree_weights in per_class_tree_weights.iter().take(k_for_warm) {
                            let w = tree_weights.get(r).copied().unwrap_or(1.0);
                            flat.push(w);
                        }
                    }
                    Some(flat)
                } else {
                    None
                };
                Some(MultiClassWarmStartState {
                    baseline_predictions: init_mc_model.baseline_predictions,
                    class_stumps: init_mc_model.class_stumps,
                    initial_rounds_completed: initial_rounds,
                    initial_ema_stats,
                    initial_dart_tree_weights,
                })
            } else {
                None
            };

            let mc_summary = if let Some(ws) = mc_warm_start {
                if let Some(validation_prepared) = validation_prepared.as_ref() {
                    trainer.fit_multiclass_iterations_warm_start_with_validation_summary(
                        &prepared.dataset,
                        &prepared.binned_matrix,
                        alloygbm_engine::ValidationDatasetRef {
                            dataset: &validation_prepared.dataset,
                            binned_matrix: &validation_prepared.binned_matrix,
                        },
                        &backend,
                        &mc_obj,
                        controls,
                        ws,
                    )?
                } else {
                    trainer.fit_multiclass_iterations_warm_start_with_summary(
                        &prepared.dataset,
                        &prepared.binned_matrix,
                        &backend,
                        &mc_obj,
                        controls,
                        ws,
                    )?
                }
            } else if let Some(validation_prepared) = validation_prepared.as_ref() {
                trainer.fit_multiclass_iterations_with_validation_summary(
                    &prepared.dataset,
                    &prepared.binned_matrix,
                    alloygbm_engine::ValidationDatasetRef {
                        dataset: &validation_prepared.dataset,
                        binned_matrix: &validation_prepared.binned_matrix,
                    },
                    &backend,
                    &mc_obj,
                    controls,
                )?
            } else {
                trainer.fit_multiclass_iterations_with_summary(
                    &prepared.dataset,
                    &prepared.binned_matrix,
                    &backend,
                    &mc_obj,
                    controls,
                )?
            };
            let native_train_seconds = native_start.elapsed().as_secs_f64();
            let native_summary = build_native_training_summary_from_multiclass(
                &mc_summary,
                bridge_prepare_seconds,
                native_train_seconds,
                objective,
            );
            let mut mc_model = mc_summary.model;
            if let Some(state) = categorical_state {
                mc_model = mc_model.with_categorical_state(Some(state))?;
            }
            let artifact_bytes = mc_model.to_artifact_bytes()?;
            return Ok(NativeTrainingResult {
                artifact_bytes,
                summary: native_summary,
                continuous_binning_metadata: prepared.metadata.into(),
                native_cat_mappings: native_cat_mappings.clone(),
            });
        }
        "custom" => {
            let grad_fn = custom_objective_fn.ok_or_else(|| {
                EngineError::ContractViolation(
                    "objective 'custom' requires a custom_objective_fn to be provided".to_string(),
                )
            })?;
            let obj = CustomPythonObjective::new(grad_fn, custom_loss_fn);
            run_training!(&obj)
        }
        other => {
            return Err(EngineError::InvalidConfig(format!(
                "unknown objective '{other}', expected one of: squared_error, \
                 binary_crossentropy, multiclass_softmax, custom, queryrmse, rank_pairwise, \
                 rank_ndcg, rank_xendcg, yetirank, poisson, gamma, tweedie"
            )));
        }
    };
    let native_train_seconds = native_start.elapsed().as_secs_f64();

    let mut model = summary.model;
    if store_node_debug_stats {
        model = model.with_node_debug_stats_from_stumps()?;
    }
    if let Some(state) = categorical_state {
        model = model.with_categorical_state(Some(state))?;
    }
    // Store native categorical feature indices in the model for artifact serialization.
    model.native_categorical_feature_indices = native_cat_infos
        .iter()
        .map(|c| c.feature_index as u32)
        .collect();
    let artifact_bytes = model.to_artifact_bytes()?;
    summary.model = model;

    Ok(NativeTrainingResult {
        artifact_bytes,
        summary: build_native_training_summary(
            &summary,
            bridge_prepare_seconds,
            native_train_seconds,
            objective,
        ),
        continuous_binning_metadata: prepared.metadata.into(),
        native_cat_mappings,
    })
}

#[pyfunction(signature = (
    rows,
    targets,
    learning_rate,
    max_depth,
    row_subsample,
    col_subsample,
    min_validation_improvement,
    seed,
    deterministic,
    rounds=DEFAULT_TRAIN_ROUNDS,
    early_stopping_rounds=None,
    categorical_feature_index=None,
    categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None,
    continuous_binning_strategy="linear",
    continuous_binning_max_bins=255,
    quantile_sketch_max_rows=None,
    objective="squared_error",
    morph_config=None,
    leaf_model="constant",
    leaf_solver="standard",
    dro_radius=0.05,
    dro_metric="wasserstein",
    neutralization="none",
    factor_neutralization_lambda=1e-6,
    factor_penalty=0.0,
    factor_exposure_values=None,
    factor_exposure_row_count=None,
    factor_exposure_factor_count=None,
    boosting_mode="standard",
    goss_top_rate=None,
    goss_other_rate=None,
    dart_drop_rate=None,
    dart_max_drop=None,
    dart_normalize_type=None,
    dart_sample_type=None,
    tweedie_variance_power=None,
    poisson_max_delta_step=None,
    quantile_alpha=None,
    ranking_sigma=1.0,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact(
    py: Python<'_>,
    rows: Vec<Vec<f32>>,
    targets: Vec<f32>,
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    rounds: usize,
    early_stopping_rounds: Option<u16>,
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    training_policy: &str,
    store_node_stats: bool,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    time_index: Option<Vec<i64>>,
    continuous_binning_strategy: &str,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_model: &str,
    leaf_solver: &str,
    dro_radius: f32,
    dro_metric: &str,
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
) -> PyResult<Vec<u8>> {
    let parsed_morph_config = morph_config
        .map(|d| parse_morph_config_from_pydict(&d))
        .transpose()?;
    let parsed_leaf_model = parse_leaf_model(leaf_model)?;
    let parsed_leaf_solver = parse_leaf_solver(leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, dro_metric)?;
    let parsed_neutralization_config =
        parse_neutralization_config(neutralization, factor_neutralization_lambda, factor_penalty)?;
    validate_neutralization_leaf_model(parsed_neutralization_config, parsed_leaf_model)?;
    let factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode,
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type,
        dart_sample_type,
    )?;
    let params = build_train_params(
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        min_validation_improvement,
        seed,
        deterministic,
        early_stopping_rounds,
        1,
        0.0,
        0.0,
        0.0,
        0.0, // min_split_gain
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        TreeGrowth::Level,
        parsed_morph_config,
        parsed_leaf_model,
        parsed_leaf_solver,
        parsed_dro_config,
        parsed_neutralization_config,
        parsed_boosting_mode,
        tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha.unwrap_or(0.5),
    );

    let categorical_spec = resolve_categorical_spec(
        categorical_feature_index,
        categorical_feature_values,
        categorical_smoothing,
        categorical_min_samples_leaf,
        categorical_time_aware,
        rows.len(),
    )
    .map_err(engine_error_to_pyerr)?;

    let (dense_values, row_count, feature_count) =
        flatten_rows(&rows).map_err(engine_error_to_pyerr)?;
    let objective_name = objective.to_string();
    py.detach(|| {
        train_regression_artifact_with_summary_dense_impl(
            &dense_values,
            row_count,
            feature_count,
            &targets,
            None, // sample_weights
            None, // group_id
            factor_exposures,
            None,
            None,
            None,
            None, // validation_sample_weights
            None, // validation_group_id
            params,
            effective_rounds,
            time_index,
            None,
            categorical_spec.into_iter().collect(),
            Vec::new(),
            training_policy,
            store_node_stats,
            continuous_binning_strategy,
            continuous_binning_max_bins,
            quantile_sketch_max_rows,
            objective_name.as_str(),
            ranking_sigma,
            lambdarank_truncation_level,
            lambdarank_normalize,
            None, // init_artifact_bytes
            None, // num_classes
            None, // custom_objective_fn
            None, // custom_loss_fn
            None, // custom_metric_fn
            0,    // max_cat_threshold (disabled for non-summary paths)
        )
    })
    .map(|result| result.artifact_bytes)
    .map_err(engine_error_to_pyerr)
}

#[pyfunction(signature = (
    values,
    row_count,
    feature_count,
    targets,
    learning_rate,
    max_depth,
    row_subsample,
    col_subsample,
    min_validation_improvement,
    seed,
    deterministic,
    rounds=DEFAULT_TRAIN_ROUNDS,
    early_stopping_rounds=None,
    categorical_feature_index=None,
    categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None,
    continuous_binning_strategy="linear",
    continuous_binning_max_bins=255,
    quantile_sketch_max_rows=None,
    objective="squared_error",
    morph_config=None,
    leaf_model="constant",
    leaf_solver="standard",
    dro_radius=0.05,
    dro_metric="wasserstein",
    neutralization="none",
    factor_neutralization_lambda=1e-6,
    factor_penalty=0.0,
    factor_exposure_values=None,
    factor_exposure_row_count=None,
    factor_exposure_factor_count=None,
    boosting_mode="standard",
    goss_top_rate=None,
    goss_other_rate=None,
    dart_drop_rate=None,
    dart_max_drop=None,
    dart_normalize_type=None,
    dart_sample_type=None,
    tweedie_variance_power=None,
    poisson_max_delta_step=None,
    quantile_alpha=None,
    ranking_sigma=1.0,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact_dense(
    py: Python<'_>,
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    targets: Vec<f32>,
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    rounds: usize,
    early_stopping_rounds: Option<u16>,
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    training_policy: &str,
    store_node_stats: bool,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    time_index: Option<Vec<i64>>,
    continuous_binning_strategy: &str,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_model: &str,
    leaf_solver: &str,
    dro_radius: f32,
    dro_metric: &str,
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
) -> PyResult<Vec<u8>> {
    let parsed_morph_config = morph_config
        .map(|d| parse_morph_config_from_pydict(&d))
        .transpose()?;
    let parsed_leaf_model = parse_leaf_model(leaf_model)?;
    let parsed_leaf_solver = parse_leaf_solver(leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, dro_metric)?;
    let parsed_neutralization_config =
        parse_neutralization_config(neutralization, factor_neutralization_lambda, factor_penalty)?;
    validate_neutralization_leaf_model(parsed_neutralization_config, parsed_leaf_model)?;
    let factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode,
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type,
        dart_sample_type,
    )?;
    let params = build_train_params(
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        min_validation_improvement,
        seed,
        deterministic,
        early_stopping_rounds,
        1,
        0.0,
        0.0,
        0.0,
        0.0, // min_split_gain
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        TreeGrowth::Level,
        parsed_morph_config,
        parsed_leaf_model,
        parsed_leaf_solver,
        parsed_dro_config,
        parsed_neutralization_config,
        parsed_boosting_mode,
        tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha.unwrap_or(0.5),
    );
    let categorical_spec = resolve_categorical_spec(
        categorical_feature_index,
        categorical_feature_values,
        categorical_smoothing,
        categorical_min_samples_leaf,
        categorical_time_aware,
        row_count,
    )
    .map_err(engine_error_to_pyerr)?;
    let objective_name = objective.to_string();
    py.detach(|| {
        train_regression_artifact_with_summary_dense_impl(
            &values,
            row_count,
            feature_count,
            &targets,
            None, // sample_weights
            None, // group_id
            factor_exposures,
            None,
            None,
            None,
            None, // validation_sample_weights
            None, // validation_group_id
            params,
            effective_rounds,
            time_index,
            None,
            categorical_spec.into_iter().collect(),
            Vec::new(),
            training_policy,
            store_node_stats,
            continuous_binning_strategy,
            continuous_binning_max_bins,
            quantile_sketch_max_rows,
            objective_name.as_str(),
            ranking_sigma,
            lambdarank_truncation_level,
            lambdarank_normalize,
            None, // init_artifact_bytes
            None, // num_classes
            None, // custom_objective_fn
            None, // custom_loss_fn
            None, // custom_metric_fn
            0,    // max_cat_threshold (disabled for non-summary paths)
        )
    })
    .map(|result| result.artifact_bytes)
    .map_err(engine_error_to_pyerr)
}

#[pyfunction(signature = (
    rows,
    targets,
    learning_rate,
    max_depth,
    row_subsample,
    col_subsample,
    min_validation_improvement,
    seed,
    deterministic,
    rounds=DEFAULT_TRAIN_ROUNDS,
    early_stopping_rounds=None,
    min_data_in_leaf=1,
    lambda_l1=0.0,
    lambda_l2=0.0,
    min_child_hessian=0.0,
    sample_weights=None,
    group_id=None,
    min_split_gain=0.0,
    validation_rows=None,
    validation_targets=None,
    validation_sample_weights=None,
    validation_group_id=None,
    validation_time_index=None,
    categorical_feature_index=None,
    categorical_feature_values=None,
    validation_categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None,
    continuous_binning_strategy="linear",
    continuous_binning_max_bins=255,
    quantile_sketch_max_rows=None,
    objective="squared_error",
    monotone_constraints=Vec::new(),
    feature_weights=Vec::new(),
    interaction_constraints=Vec::new(),
    max_leaves=None,
    tree_growth="level",
    categorical_feature_indices=None,
    categorical_feature_values_list=None,
    validation_categorical_feature_values_list=None,
    init_artifact_bytes=None,
    num_classes=None,
    custom_objective_fn=None,
    custom_loss_fn=None,
    custom_metric_fn=None,
    max_cat_threshold=0,
    morph_config=None,
    leaf_model="constant",
    leaf_solver="standard",
    dro_radius=0.05,
    dro_metric="wasserstein",
    neutralization="none",
    factor_neutralization_lambda=1e-6,
    factor_penalty=0.0,
    factor_exposure_values=None,
    factor_exposure_row_count=None,
    factor_exposure_factor_count=None,
    boosting_mode="standard",
    goss_top_rate=None,
    goss_other_rate=None,
    dart_drop_rate=None,
    dart_max_drop=None,
    dart_normalize_type=None,
    dart_sample_type=None,
    tweedie_variance_power=None,
    poisson_max_delta_step=None,
    quantile_alpha=None,
    ranking_sigma=1.0,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact_with_summary(
    py: Python<'_>,
    rows: Vec<Vec<f32>>,
    targets: Vec<f32>,
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    rounds: usize,
    early_stopping_rounds: Option<u16>,
    min_data_in_leaf: u32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_child_hessian: f32,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    min_split_gain: f32,
    validation_rows: Option<Vec<Vec<f32>>>,
    validation_targets: Option<Vec<f32>>,
    validation_sample_weights: Option<Vec<f32>>,
    validation_group_id: Option<Vec<u32>>,
    validation_time_index: Option<Vec<i64>>,
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    validation_categorical_feature_values: Option<Vec<String>>,
    training_policy: &str,
    store_node_stats: bool,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    time_index: Option<Vec<i64>>,
    continuous_binning_strategy: &str,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    monotone_constraints: Vec<i8>,
    feature_weights: Vec<f32>,
    interaction_constraints: Vec<Vec<u32>>,
    max_leaves: Option<usize>,
    tree_growth: &str,
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values_list: Option<Vec<Vec<String>>>,
    validation_categorical_feature_values_list: Option<Vec<Vec<String>>>,
    init_artifact_bytes: Option<Vec<u8>>,
    num_classes: Option<usize>,
    custom_objective_fn: Option<Py<PyAny>>,
    custom_loss_fn: Option<Py<PyAny>>,
    custom_metric_fn: Option<Py<PyAny>>,
    max_cat_threshold: usize,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_model: &str,
    leaf_solver: &str,
    dro_radius: f32,
    dro_metric: &str,
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
) -> PyResult<NativeTrainingResult> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
    let tree_growth = parse_tree_growth(tree_growth).map_err(engine_error_to_pyerr)?;
    let parsed_morph_config = morph_config
        .map(|d| parse_morph_config_from_pydict(&d))
        .transpose()?;
    let parsed_leaf_model = parse_leaf_model(leaf_model)?;
    let parsed_leaf_solver = parse_leaf_solver(leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, dro_metric)?;
    let parsed_neutralization_config =
        parse_neutralization_config(neutralization, factor_neutralization_lambda, factor_penalty)?;
    validate_neutralization_leaf_model(parsed_neutralization_config, parsed_leaf_model)?;
    let factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode,
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type,
        dart_sample_type,
    )?;
    let params = build_train_params(
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        min_validation_improvement,
        seed,
        deterministic,
        early_stopping_rounds,
        min_data_in_leaf,
        lambda_l1,
        lambda_l2,
        min_child_hessian,
        min_split_gain,
        monotone_constraints,
        feature_weights,
        interaction_constraints,
        max_leaves,
        tree_growth,
        parsed_morph_config,
        parsed_leaf_model,
        parsed_leaf_solver,
        parsed_dro_config,
        parsed_neutralization_config,
        parsed_boosting_mode,
        tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha.unwrap_or(0.5),
    );
    let (categorical_specs, validation_categorical_values_list) =
        resolve_categorical_specs_from_params(
            categorical_feature_index,
            categorical_feature_values,
            categorical_feature_indices,
            categorical_feature_values_list,
            validation_categorical_feature_values,
            validation_categorical_feature_values_list,
            categorical_smoothing,
            categorical_min_samples_leaf,
            categorical_time_aware,
            rows.len(),
        )
        .map_err(engine_error_to_pyerr)?;
    let (dense_values, row_count, feature_count) =
        flatten_rows(&rows).map_err(engine_error_to_pyerr)?;
    let validation_payload = if let Some(validation_rows) = validation_rows.as_ref() {
        Some(flatten_rows(validation_rows).map_err(engine_error_to_pyerr)?)
    } else {
        None
    };
    let validation_row_count = validation_payload.as_ref().map(|(_, rows, _)| *rows);
    let validation_feature_count = validation_payload
        .as_ref()
        .map(|(_, _, feature_count)| *feature_count);
    if validation_feature_count.is_some_and(|count| count != feature_count) {
        return Err(PyValueError::new_err(
            "validation feature_count must match training feature_count",
        ));
    }

    let objective_name = objective.to_string();
    let result = py.detach(|| {
        train_regression_artifact_with_summary_dense_impl(
            &dense_values,
            row_count,
            feature_count,
            &targets,
            sample_weights,
            group_id,
            factor_exposures,
            validation_payload
                .as_ref()
                .map(|(values, _, _)| values.as_slice()),
            validation_row_count,
            validation_targets.as_deref(),
            validation_sample_weights,
            validation_group_id,
            params,
            effective_rounds,
            time_index,
            validation_time_index,
            categorical_specs,
            validation_categorical_values_list,
            training_policy,
            store_node_stats,
            continuous_binning_strategy,
            continuous_binning_max_bins,
            quantile_sketch_max_rows,
            objective_name.as_str(),
            ranking_sigma,
            lambdarank_truncation_level,
            lambdarank_normalize,
            init_artifact_bytes.as_deref(),
            num_classes,
            custom_objective_fn,
            custom_loss_fn,
            custom_metric_fn,
            max_cat_threshold,
        )
    });

    result.map_err(engine_error_to_pyerr)
}

#[pyfunction(signature = (
    values,
    row_count,
    feature_count,
    targets,
    learning_rate,
    max_depth,
    row_subsample,
    col_subsample,
    min_validation_improvement,
    seed,
    deterministic,
    rounds=DEFAULT_TRAIN_ROUNDS,
    early_stopping_rounds=None,
    min_data_in_leaf=1,
    lambda_l1=0.0,
    lambda_l2=0.0,
    min_child_hessian=0.0,
    sample_weights=None,
    group_id=None,
    min_split_gain=0.0,
    validation_values=None,
    validation_row_count=None,
    validation_targets=None,
    validation_sample_weights=None,
    validation_group_id=None,
    validation_time_index=None,
    categorical_feature_index=None,
    categorical_feature_values=None,
    validation_categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None,
    continuous_binning_strategy="linear",
    continuous_binning_max_bins=255,
    quantile_sketch_max_rows=None,
    objective="squared_error",
    monotone_constraints=Vec::new(),
    feature_weights=Vec::new(),
    interaction_constraints=Vec::new(),
    max_leaves=None,
    tree_growth="level",
    categorical_feature_indices=None,
    categorical_feature_values_list=None,
    validation_categorical_feature_values_list=None,
    init_artifact_bytes=None,
    num_classes=None,
    custom_objective_fn=None,
    custom_loss_fn=None,
    custom_metric_fn=None,
    max_cat_threshold=0,
    morph_config=None,
    leaf_model="constant",
    leaf_solver="standard",
    dro_radius=0.05,
    dro_metric="wasserstein",
    neutralization="none",
    factor_neutralization_lambda=1e-6,
    factor_penalty=0.0,
    factor_exposure_values=None,
    factor_exposure_row_count=None,
    factor_exposure_factor_count=None,
    boosting_mode="standard",
    goss_top_rate=None,
    goss_other_rate=None,
    dart_drop_rate=None,
    dart_max_drop=None,
    dart_normalize_type=None,
    dart_sample_type=None,
    tweedie_variance_power=None,
    poisson_max_delta_step=None,
    quantile_alpha=None,
    ranking_sigma=1.0,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact_dense_with_summary(
    py: Python<'_>,
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    targets: Vec<f32>,
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    rounds: usize,
    early_stopping_rounds: Option<u16>,
    min_data_in_leaf: u32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_child_hessian: f32,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    min_split_gain: f32,
    validation_values: Option<Vec<f32>>,
    validation_row_count: Option<usize>,
    validation_targets: Option<Vec<f32>>,
    validation_sample_weights: Option<Vec<f32>>,
    validation_group_id: Option<Vec<u32>>,
    validation_time_index: Option<Vec<i64>>,
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    validation_categorical_feature_values: Option<Vec<String>>,
    training_policy: &str,
    store_node_stats: bool,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    time_index: Option<Vec<i64>>,
    continuous_binning_strategy: &str,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    monotone_constraints: Vec<i8>,
    feature_weights: Vec<f32>,
    interaction_constraints: Vec<Vec<u32>>,
    max_leaves: Option<usize>,
    tree_growth: &str,
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values_list: Option<Vec<Vec<String>>>,
    validation_categorical_feature_values_list: Option<Vec<Vec<String>>>,
    init_artifact_bytes: Option<Vec<u8>>,
    num_classes: Option<usize>,
    custom_objective_fn: Option<Py<PyAny>>,
    custom_loss_fn: Option<Py<PyAny>>,
    custom_metric_fn: Option<Py<PyAny>>,
    max_cat_threshold: usize,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_model: &str,
    leaf_solver: &str,
    dro_radius: f32,
    dro_metric: &str,
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
) -> PyResult<NativeTrainingResult> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
    let tree_growth = parse_tree_growth(tree_growth).map_err(engine_error_to_pyerr)?;
    let parsed_morph_config = morph_config
        .map(|d| parse_morph_config_from_pydict(&d))
        .transpose()?;
    let parsed_leaf_model = parse_leaf_model(leaf_model)?;
    let parsed_leaf_solver = parse_leaf_solver(leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, dro_metric)?;
    let parsed_neutralization_config =
        parse_neutralization_config(neutralization, factor_neutralization_lambda, factor_penalty)?;
    validate_neutralization_leaf_model(parsed_neutralization_config, parsed_leaf_model)?;
    let factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode,
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type,
        dart_sample_type,
    )?;
    let params = build_train_params(
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        min_validation_improvement,
        seed,
        deterministic,
        early_stopping_rounds,
        min_data_in_leaf,
        lambda_l1,
        lambda_l2,
        min_child_hessian,
        min_split_gain,
        monotone_constraints,
        feature_weights,
        interaction_constraints,
        max_leaves,
        tree_growth,
        parsed_morph_config,
        parsed_leaf_model,
        parsed_leaf_solver,
        parsed_dro_config,
        parsed_neutralization_config,
        parsed_boosting_mode,
        tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha.unwrap_or(0.5),
    );
    let (categorical_specs, validation_categorical_values_list) =
        resolve_categorical_specs_from_params(
            categorical_feature_index,
            categorical_feature_values,
            categorical_feature_indices,
            categorical_feature_values_list,
            validation_categorical_feature_values,
            validation_categorical_feature_values_list,
            categorical_smoothing,
            categorical_min_samples_leaf,
            categorical_time_aware,
            row_count,
        )
        .map_err(engine_error_to_pyerr)?;
    let objective_name = objective.to_string();
    let result = py.detach(|| {
        train_regression_artifact_with_summary_dense_impl(
            &values,
            row_count,
            feature_count,
            &targets,
            sample_weights,
            group_id,
            factor_exposures,
            validation_values.as_deref(),
            validation_row_count,
            validation_targets.as_deref(),
            validation_sample_weights,
            validation_group_id,
            params,
            effective_rounds,
            time_index,
            validation_time_index,
            categorical_specs,
            validation_categorical_values_list,
            training_policy,
            store_node_stats,
            continuous_binning_strategy,
            continuous_binning_max_bins,
            quantile_sketch_max_rows,
            objective_name.as_str(),
            ranking_sigma,
            lambdarank_truncation_level,
            lambdarank_normalize,
            init_artifact_bytes.as_deref(),
            num_classes,
            custom_objective_fn,
            custom_loss_fn,
            custom_metric_fn,
            max_cat_threshold,
        )
    });

    result.map_err(engine_error_to_pyerr)
}

/// Reinterpret raw bytes as f32 slice (safe, no allocation).
fn bytes_to_f32_vec(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values_bytes length must be a multiple of 4 (f32)",
        ));
    }
    let count = bytes.len() / 4;
    let mut result = vec![0.0_f32; count];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        result[i] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(result)
}

fn dense_input_to_f32_vec(input: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(array) = input.cast::<PyArray2<f32>>() {
        let readonly = array.readonly();
        let values = readonly
            .as_slice()
            .map_err(|_| PyValueError::new_err("values array must be C-contiguous float32"))?;
        return Ok(values.to_vec());
    }
    let bytes = input.cast::<PyBytes>().map_err(|_| {
        PyValueError::new_err("values_bytes must be bytes or a C-contiguous float32 array")
    })?;
    bytes_to_f32_vec(bytes.as_bytes())
}

#[pyfunction(signature = (
    values_bytes,
    row_count,
    feature_count,
    targets_bytes,
    learning_rate,
    max_depth,
    row_subsample,
    col_subsample,
    min_validation_improvement,
    seed,
    deterministic,
    rounds=DEFAULT_TRAIN_ROUNDS,
    early_stopping_rounds=None,
    min_data_in_leaf=1,
    lambda_l1=0.0,
    lambda_l2=0.0,
    min_child_hessian=0.0,
    sample_weights=None,
    group_id=None,
    min_split_gain=0.0,
    validation_values_bytes=None,
    validation_row_count=None,
    validation_targets_bytes=None,
    validation_sample_weights=None,
    validation_group_id=None,
    validation_time_index=None,
    categorical_feature_index=None,
    categorical_feature_values=None,
    validation_categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None,
    continuous_binning_strategy="linear",
    continuous_binning_max_bins=255,
    quantile_sketch_max_rows=None,
    objective="squared_error",
    monotone_constraints=Vec::new(),
    feature_weights=Vec::new(),
    interaction_constraints=Vec::new(),
    max_leaves=None,
    tree_growth="level",
    categorical_feature_indices=None,
    categorical_feature_values_list=None,
    validation_categorical_feature_values_list=None,
    init_artifact_bytes=None,
    num_classes=None,
    custom_objective_fn=None,
    custom_loss_fn=None,
    custom_metric_fn=None,
    max_cat_threshold=0,
    morph_config=None,
    leaf_model="constant",
    leaf_solver="standard",
    dro_radius=0.05,
    dro_metric="wasserstein",
    neutralization="none",
    factor_neutralization_lambda=1e-6,
    factor_penalty=0.0,
    factor_exposure_values=None,
    factor_exposure_row_count=None,
    factor_exposure_factor_count=None,
    boosting_mode="standard",
    goss_top_rate=None,
    goss_other_rate=None,
    dart_drop_rate=None,
    dart_max_drop=None,
    dart_normalize_type=None,
    dart_sample_type=None,
    tweedie_variance_power=None,
    poisson_max_delta_step=None,
    quantile_alpha=None,
    ranking_sigma=1.0,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_regression_artifact_dense_with_summary_bytes(
    py: Python<'_>,
    values_bytes: &Bound<'_, PyAny>,
    row_count: usize,
    feature_count: usize,
    targets_bytes: &[u8],
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    rounds: usize,
    early_stopping_rounds: Option<u16>,
    min_data_in_leaf: u32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_child_hessian: f32,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    min_split_gain: f32,
    validation_values_bytes: Option<&Bound<'_, PyAny>>,
    validation_row_count: Option<usize>,
    validation_targets_bytes: Option<&[u8]>,
    validation_sample_weights: Option<Vec<f32>>,
    validation_group_id: Option<Vec<u32>>,
    validation_time_index: Option<Vec<i64>>,
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    validation_categorical_feature_values: Option<Vec<String>>,
    training_policy: &str,
    store_node_stats: bool,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    time_index: Option<Vec<i64>>,
    continuous_binning_strategy: &str,
    continuous_binning_max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    objective: &str,
    monotone_constraints: Vec<i8>,
    feature_weights: Vec<f32>,
    interaction_constraints: Vec<Vec<u32>>,
    max_leaves: Option<usize>,
    tree_growth: &str,
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values_list: Option<Vec<Vec<String>>>,
    validation_categorical_feature_values_list: Option<Vec<Vec<String>>>,
    init_artifact_bytes: Option<Vec<u8>>,
    num_classes: Option<usize>,
    custom_objective_fn: Option<Py<PyAny>>,
    custom_loss_fn: Option<Py<PyAny>>,
    custom_metric_fn: Option<Py<PyAny>>,
    max_cat_threshold: usize,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_model: &str,
    leaf_solver: &str,
    dro_radius: f32,
    dro_metric: &str,
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
) -> PyResult<NativeTrainingResult> {
    let values = dense_input_to_f32_vec(values_bytes)?;
    let targets = bytes_to_f32_vec(targets_bytes)?;
    let validation_values = validation_values_bytes
        .map(dense_input_to_f32_vec)
        .transpose()?;
    let validation_targets = validation_targets_bytes.map(bytes_to_f32_vec).transpose()?;
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
    let tree_growth = parse_tree_growth(tree_growth).map_err(engine_error_to_pyerr)?;
    let parsed_morph_config = morph_config
        .map(|d| parse_morph_config_from_pydict(&d))
        .transpose()?;
    let parsed_leaf_model = parse_leaf_model(leaf_model)?;
    let parsed_leaf_solver = parse_leaf_solver(leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, dro_metric)?;
    let parsed_neutralization_config =
        parse_neutralization_config(neutralization, factor_neutralization_lambda, factor_penalty)?;
    validate_neutralization_leaf_model(parsed_neutralization_config, parsed_leaf_model)?;
    let factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode,
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type,
        dart_sample_type,
    )?;
    let params = build_train_params(
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        min_validation_improvement,
        seed,
        deterministic,
        early_stopping_rounds,
        min_data_in_leaf,
        lambda_l1,
        lambda_l2,
        min_child_hessian,
        min_split_gain,
        monotone_constraints,
        feature_weights,
        interaction_constraints,
        max_leaves,
        tree_growth,
        parsed_morph_config,
        parsed_leaf_model,
        parsed_leaf_solver,
        parsed_dro_config,
        parsed_neutralization_config,
        parsed_boosting_mode,
        tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha.unwrap_or(0.5),
    );
    let (categorical_specs, validation_categorical_values_list) =
        resolve_categorical_specs_from_params(
            categorical_feature_index,
            categorical_feature_values,
            categorical_feature_indices,
            categorical_feature_values_list,
            validation_categorical_feature_values,
            validation_categorical_feature_values_list,
            categorical_smoothing,
            categorical_min_samples_leaf,
            categorical_time_aware,
            row_count,
        )
        .map_err(engine_error_to_pyerr)?;
    let objective_name = objective.to_string();
    let result = py.detach(|| {
        train_regression_artifact_with_summary_dense_impl(
            &values,
            row_count,
            feature_count,
            &targets,
            sample_weights,
            group_id,
            factor_exposures,
            validation_values.as_deref(),
            validation_row_count,
            validation_targets.as_deref(),
            validation_sample_weights,
            validation_group_id,
            params,
            effective_rounds,
            time_index,
            validation_time_index,
            categorical_specs,
            validation_categorical_values_list,
            training_policy,
            store_node_stats,
            continuous_binning_strategy,
            continuous_binning_max_bins,
            quantile_sketch_max_rows,
            objective_name.as_str(),
            ranking_sigma,
            lambdarank_truncation_level,
            lambdarank_normalize,
            init_artifact_bytes.as_deref(),
            num_classes,
            custom_objective_fn,
            custom_loss_fn,
            custom_metric_fn,
            max_cat_threshold,
        )
    });

    result.map_err(engine_error_to_pyerr)
}
