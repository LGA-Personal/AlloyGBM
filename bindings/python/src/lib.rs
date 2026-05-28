#![allow(clippy::too_many_arguments)]

mod callbacks;
mod categorical_bridge;
mod errors;
mod params;
mod predict;
mod pyclasses;
mod quantization;
mod train;
use crate::errors::engine_error_to_pyerr;
use crate::params::{
    parse_boosting_mode, parse_dro_config, parse_factor_exposure_matrix, parse_leaf_solver,
    parse_morph_config_from_pydict, parse_neutralization_config,
};
use crate::predict::{
    NativePredictorHandle, predictor_predict_batch, predictor_predict_batch_canonical,
    predictor_predict_batch_canonical_dense, predictor_predict_batch_dense,
    shap_explain_interactions, shap_explain_interactions_dense,
    shap_explain_interactions_dense_with_binning, shap_explain_interactions_with_binning,
    shap_explain_rows, shap_explain_rows_dense, shap_explain_rows_dense_with_binning,
    shap_explain_rows_with_binning, shap_global_importance, shap_global_importance_dense,
    shap_global_importance_dense_with_binning, shap_global_importance_with_binning,
};
use crate::pyclasses::{
    NativeContinuousBinningMetadata, NativeIterationDiagnostics, NativeRuntimeInfo,
    NativeTrainingResult, NativeTrainingSummary, native_runtime_info,
};
use crate::quantization::{ContinuousBinningStrategy, prepare_training_matrices_from_dense_values};
use crate::train::{
    train_regression_artifact, train_regression_artifact_dense,
    train_regression_artifact_dense_with_summary,
    train_regression_artifact_dense_with_summary_bytes, train_regression_artifact_with_summary,
};

use alloygbm_core::{DenseMatrixView, TrainParams, TreeGrowth};
use alloygbm_engine::CategoricalFeatureInfo;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) const DEFAULT_TRAIN_ROUNDS: usize = 6;
pub(crate) const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
pub(crate) const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U8: u16 = 254;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U16: u16 = 65534;
pub(crate) const MIN_CONTINUOUS_QUANTIZED_BINS: usize = 2;
pub(crate) const LINEAR_TAIL_RANK_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK";
pub(crate) const LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO";
pub(crate) const DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD: f32 = 0.10;

pub(crate) fn dense_rows_from_flat_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Result<Vec<Vec<f32>>, String> {
    DenseMatrixView::new(row_count, feature_count, values).map_err(|error| error.to_string())?;
    Ok(values
        .chunks(feature_count)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>())
}

// ---------------------------------------------------------------------------
// Joint multi-output trainer + predictor handle (v0.10.1)
// ---------------------------------------------------------------------------

/// PyO3 handle that wraps `alloygbm_engine::joint::JointPredictor` for K-output
/// prediction from Python. `MultiLabelGBMRanker(training_mode="joint")` is the
/// primary consumer.
#[pyclass]
#[derive(Debug, Clone)]
struct JointPredictorHandle {
    predictor: alloygbm_engine::joint::JointPredictor,
    feature_count: usize,
}

#[pymethods]
impl JointPredictorHandle {
    #[new]
    fn new(artifact_bytes: &[u8], baselines: Vec<f32>, feature_count: usize) -> PyResult<Self> {
        let predictor =
            alloygbm_engine::joint::JointPredictor::from_artifact_bytes(artifact_bytes, baselines)
                .map_err(PyValueError::new_err)?;
        Ok(Self {
            predictor,
            feature_count,
        })
    }

    /// Predict K outputs for each row. Returns a flat row-major Vec<f32> of
    /// length `n_rows * n_outputs`; the Python wrapper reshapes.
    fn predict_dense(&self, values: Vec<f32>) -> PyResult<Vec<f32>> {
        if self.feature_count == 0 {
            return Err(PyValueError::new_err("feature_count must be > 0"));
        }
        if !values.len().is_multiple_of(self.feature_count) {
            return Err(PyValueError::new_err(format!(
                "values length {} not divisible by feature_count {}",
                values.len(),
                self.feature_count
            )));
        }
        Ok(self.predictor.predict_batch(&values, self.feature_count))
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.predictor.n_outputs
    }

    #[getter]
    fn baselines(&self) -> Vec<f32> {
        self.predictor.baselines.clone()
    }
}

/// Train a joint multi-output model that shares trees across K outputs.
///
/// v0.10.1 minimum-viable wiring: pre-bins dense `x_values` using the existing
/// `prepare_training_matrices_from_dense_values` helper (using y[:, 0] as a
/// throwaway target — binning is target-independent), then dispatches to
/// `alloygbm_engine::joint::fit_joint_multi_output`. Returns the artifact
/// bytes (with `MultiOutputLeafValues` section) along with per-output
/// baselines + the feature count for the `JointPredictorHandle` constructor.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x_values, row_count, feature_count,
    targets_per_output, n_outputs,
    per_output_objective_names,
    group_id,
    n_estimators,
    learning_rate,
    seed,
    max_depth,
    min_data_in_leaf,
    lambda_l2,
    max_bin,
    min_split_gain=0.0_f32,
    row_subsample=1.0_f32,
    col_subsample=1.0_f32,
    interaction_constraints=Vec::<Vec<u32>>::new(),
    tree_growth="level".to_string(),
    max_leaves=None::<usize>,
    categorical_feature_indices=Vec::<usize>::new(),
    max_cat_threshold=0_usize,
    boosting_mode="standard".to_string(),
    goss_top_rate=None::<f32>,
    goss_other_rate=None::<f32>,
    dart_drop_rate=None::<f32>,
    dart_max_drop=None::<usize>,
    dart_normalize_type=None::<String>,
    dart_sample_type=None::<String>,
    init_artifact_bytes=None::<Vec<u8>>,
    init_baselines=None::<Vec<f32>>,
    init_rounds_completed=None::<usize>,
    morph_config=None::<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_solver="standard".to_string(),
    dro_radius=0.05_f32,
    dro_metric="wasserstein".to_string(),
    factor_exposure_values=None::<Vec<f32>>,
    factor_exposure_row_count=None::<usize>,
    factor_exposure_factor_count=None::<usize>,
    neutralization="none".to_string(),
    factor_neutralization_lambda=1e-6_f32,
    factor_penalty=0.0_f32,
))]
fn train_joint_multi_label_ranker(
    x_values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    targets_per_output: Vec<Vec<f32>>,
    n_outputs: usize,
    per_output_objective_names: Vec<String>,
    group_id: Option<Vec<u32>>,
    n_estimators: usize,
    learning_rate: f32,
    seed: u64,
    max_depth: u16,
    min_data_in_leaf: u32,
    lambda_l2: f32,
    max_bin: usize,
    min_split_gain: f32,
    row_subsample: f32,
    col_subsample: f32,
    interaction_constraints: Vec<Vec<u32>>,
    tree_growth: String,
    max_leaves: Option<usize>,
    categorical_feature_indices: Vec<usize>,
    max_cat_threshold: usize,
    boosting_mode: String,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<String>,
    dart_sample_type: Option<String>,
    init_artifact_bytes: Option<Vec<u8>>,
    init_baselines: Option<Vec<f32>>,
    init_rounds_completed: Option<usize>,
    morph_config: Option<pyo3::Bound<'_, pyo3::types::PyDict>>,
    leaf_solver: String,
    dro_radius: f32,
    dro_metric: String,
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
    neutralization: String,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
) -> PyResult<(Vec<u8>, Vec<f32>, usize, usize)> {
    use alloygbm_engine::joint::JointObjective;

    if targets_per_output.len() != n_outputs {
        return Err(PyValueError::new_err(format!(
            "targets_per_output length {} != n_outputs {n_outputs}",
            targets_per_output.len()
        )));
    }
    if per_output_objective_names.len() != n_outputs {
        return Err(PyValueError::new_err(format!(
            "per_output_objective_names length {} != n_outputs {n_outputs}",
            per_output_objective_names.len()
        )));
    }
    if n_outputs == 0 {
        return Err(PyValueError::new_err("n_outputs must be > 0"));
    }
    for (k, tg) in targets_per_output.iter().enumerate() {
        if tg.len() != row_count {
            return Err(PyValueError::new_err(format!(
                "targets[{k}] length {} != row_count {row_count}",
                tg.len()
            )));
        }
    }

    let mut per_output_objective: Vec<JointObjective> = Vec::with_capacity(n_outputs);
    for name in &per_output_objective_names {
        let obj = JointObjective::parse(name)
            .map_err(|e| PyValueError::new_err(format!("invalid objective {name:?}: {e}")))?;
        per_output_objective.push(obj);
    }
    if per_output_objective.iter().any(|o| o.requires_group()) && group_id.is_none() {
        return Err(PyValueError::new_err(
            "at least one ranking objective requires group_id",
        ));
    }

    // Build the binned matrix. The joint trainer needs only `binned_matrix`;
    // use y[:, 0] as a throwaway target since binning is target-independent
    // for the linear and rank strategies.
    let throwaway_targets = targets_per_output[0].clone();
    let mut prepared = prepare_training_matrices_from_dense_values(
        &x_values,
        row_count,
        feature_count,
        &throwaway_targets,
        None,
        None,
        group_id.clone(),
        ContinuousBinningStrategy::Linear,
        max_bin,
        false,
    )
    .map_err(engine_error_to_pyerr)?;

    let tg = match tree_growth.as_str() {
        "level" => TreeGrowth::Level,
        "leaf" => TreeGrowth::Leaf,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown tree_growth {other:?}; expected 'level' or 'leaf'"
            )));
        }
    };
    let parsed_boosting_mode = parse_boosting_mode(
        boosting_mode.as_str(),
        goss_top_rate,
        goss_other_rate,
        dart_drop_rate,
        dart_max_drop,
        dart_normalize_type.as_deref(),
        dart_sample_type.as_deref(),
    )?;
    // v0.10.4: MorphBoost — parse the per-label `morph_config` dict
    // (built by `alloygbm._morph.build_morph_config_dict`) into a
    // `MorphConfig`. `None` means non-morph training. Validation runs in
    // `parse_morph_config_from_pydict` (mirrors single-output path).
    let parsed_morph_config = match morph_config.as_ref() {
        Some(dict) => Some(parse_morph_config_from_pydict(dict)?),
        None => None,
    };
    // v0.10.5: DRO leaf solver — mirrors the single-output path.
    let parsed_leaf_solver = parse_leaf_solver(&leaf_solver)?;
    let parsed_dro_config = parse_dro_config(parsed_leaf_solver, dro_radius, &dro_metric)?;
    // v0.10.6: factor neutralization — accept exposures + neutralization
    // config and cross-validate the two are consistent (active config requires
    // exposures; exposures require an active config).
    let parsed_factor_exposures = parse_factor_exposure_matrix(
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    )?;
    let parsed_neutralization_config = parse_neutralization_config(
        &neutralization,
        factor_neutralization_lambda,
        factor_penalty,
    )?;
    let neutralization_active = parsed_neutralization_config
        .map(|c| c.kind != alloygbm_core::NeutralizationKind::None)
        .unwrap_or(false);
    if neutralization_active && parsed_factor_exposures.is_none() {
        return Err(PyValueError::new_err(
            "factor_exposures are required when neutralization is active",
        ));
    }
    if !neutralization_active && parsed_factor_exposures.is_some() {
        return Err(PyValueError::new_err(
            "factor_exposures were provided but neutralization='none'",
        ));
    }
    if let Some(exposures) = parsed_factor_exposures.as_ref()
        && exposures.row_count != row_count
    {
        return Err(PyValueError::new_err(format!(
            "factor_exposures row_count {} does not match X row_count {row_count}",
            exposures.row_count
        )));
    }
    let params = TrainParams {
        learning_rate,
        seed,
        max_depth,
        min_data_in_leaf,
        lambda_l2,
        min_split_gain,
        row_subsample,
        col_subsample,
        interaction_constraints,
        tree_growth: tg,
        max_leaves,
        boosting_mode: parsed_boosting_mode,
        morph_config: parsed_morph_config,
        leaf_solver: parsed_leaf_solver,
        dro_config: parsed_dro_config,
        neutralization_config: parsed_neutralization_config,
        ..TrainParams::default()
    };

    // v0.10.3: For each requested categorical feature, re-bin the column
    // so `bin_index == category_id`. This is the invariant the joint
    // native-cat trainer requires — `find_best_multi_output_categorical_split`
    // returns a u64 bitset keyed by category ID, and the JointPredictor at
    // predict time reads the raw feature value (cast to integer) as the
    // category ID. If the binning step had reordered or merged categories
    // (which `ContinuousBinningStrategy::Linear` can) the bitset would
    // route rows to the wrong leaf.
    //
    // Strategy: scan the dense float column for unique non-NaN integer
    // values (cast f32 -> i64), sort them, assign category IDs in sort
    // order, and overwrite the binned column. This matches the
    // single-output `apply_categorical_encoding_to_training_matrices_multi`
    // semantics for low-cardinality features.
    let mut cat_features: Vec<CategoricalFeatureInfo> = Vec::new();
    if !categorical_feature_indices.is_empty() && max_cat_threshold > 0 {
        let missing_bin = prepared.binned_matrix.missing_bin();
        for &fi in &categorical_feature_indices {
            if fi >= feature_count {
                return Err(PyValueError::new_err(format!(
                    "categorical_feature_indices contains {fi} which exceeds feature_count {feature_count}"
                )));
            }
            // PR #36 review (C1, C3): the JointPredictor reads the raw
            // feature value as a category ID via `v as i64`
            // (truncation toward zero — see `JointPredictor::predict_row`
            // in `crates/engine/src/joint.rs`). For training and
            // inference to agree, two invariants must hold for every
            // requested categorical column:
            //   (a) values must already be dense zero-based integer
            //       IDs in `{0, 1, ..., K-1}`. A non-dense set like
            //       {10, 20, 30} cannot be remapped to dense
            //       {0, 1, 2} at training time without also remapping
            //       at predict time (the trained bitset is keyed by
            //       the dense ID but predict reads the raw 10/20/30).
            //   (b) values must be exact integer-valued floats. A
            //       value like 0.6 would `round()` to 1 at training
            //       but truncate to 0 at predict.
            // Persisting and reapplying a per-feature `cat_to_id`
            // mapping in the joint artifact path is tracked for
            // v0.10.4 alongside `categorical_state` plumbing on the
            // joint predictor. For v0.10.3 we reject inputs that
            // violate either invariant with a clear actionable error.
            let mut uniq: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
            for row in 0..row_count {
                let v = x_values[row * feature_count + fi];
                if !v.is_finite() {
                    continue;
                }
                let truncated = v as i64;
                if (v - truncated as f32).abs() > f32::EPSILON {
                    return Err(PyValueError::new_err(format!(
                        "Native categorical feature {fi} contains non-integer value {v}; \
                         joint mode requires dense integer category IDs in {{0, 1, ..., K-1}}. \
                         Pre-encode the column (e.g. sklearn LabelEncoder) before fitting."
                    )));
                }
                uniq.insert(truncated);
            }
            let num_categories = uniq.len();
            // Bail out (silently fall back to numeric) if cardinality
            // exceeds `max_cat_threshold` or the 64-category Fisher-sort
            // cap. This mirrors single-output LightGBM semantics.
            if num_categories < 2 || num_categories > max_cat_threshold || num_categories > 64 {
                continue;
            }
            // Invariant (a): values must be exactly {0, 1, ..., K-1}.
            // BTreeSet iterates in sorted order, so the first element
            // must be 0 and the last must be K-1 (with no gaps).
            let min_val = *uniq.iter().next().unwrap();
            let max_val = *uniq.iter().next_back().unwrap();
            if min_val != 0 || max_val != (num_categories as i64) - 1 {
                return Err(PyValueError::new_err(format!(
                    "Native categorical feature {fi} has {num_categories} unique values \
                     ranging from {min_val} to {max_val}; joint mode requires dense \
                     zero-based integer IDs in {{0, 1, ..., K-1}}. Pre-encode the column \
                     (e.g. sklearn LabelEncoder) before fitting. Non-dense category IDs \
                     are tracked for v0.10.4 alongside `categorical_state` plumbing on \
                     the joint predictor."
                )));
            }
            // Guard: category IDs must not collide with the missing-value
            // sentinel.
            if (num_categories as u16) > missing_bin {
                return Err(PyValueError::new_err(format!(
                    "Native categorical feature {fi} has {num_categories} categories which would collide with the missing-value sentinel (bin {missing_bin}). Reduce max_cat_threshold or increase max_bin."
                )));
            }
            // Re-bin the column: `bin_index == category_id` (which
            // equals the raw value here since we've validated dense
            // 0..K-1). Use truncation `v as i64` to match
            // `JointPredictor::predict_row` (PR #36 review C1).
            for row in 0..row_count {
                let v = x_values[row * feature_count + fi];
                let bin_val = if v.is_finite() {
                    v as i64 as u16
                } else {
                    missing_bin
                };
                prepared.binned_matrix.set_bin(row, fi, bin_val);
            }
            let cat_max_bin = (num_categories - 1) as u16;
            if cat_max_bin > prepared.binned_matrix.max_bin {
                prepared.binned_matrix.max_bin = cat_max_bin;
            }
            cat_features.push(CategoricalFeatureInfo {
                feature_index: fi,
                num_categories,
            });
        }
    }

    // v0.10.3: joint warm-start. When `init_artifact_bytes` is
    // provided, rebuild the prior `TrainedModel` from artifact bytes
    // and construct a `JointWarmStartState`. The trainer prepends
    // prior stumps to `all_stumps`, replays their contributions onto
    // `predictions`, and re-encodes new-round `node_id` starting at
    // `initial_rounds_completed` so global tree IDs don't collide.
    let warm_start = if let Some(bytes) = init_artifact_bytes {
        let baselines = init_baselines.ok_or_else(|| {
            PyValueError::new_err("init_baselines is required when init_artifact_bytes is provided")
        })?;
        let rounds = init_rounds_completed.unwrap_or(0);
        let prior_model = alloygbm_engine::TrainedModel::from_artifact_bytes(&bytes)
            .map_err(|e| PyValueError::new_err(format!("init_artifact_bytes decode: {e:?}")))?;
        // PR #40 review (R2): validate that the prior artifact's
        // `neutralization_metadata` matches the current
        // `parsed_neutralization_config` BEFORE constructing the warm-start
        // state. Without this, a caller can resume an unneutralized prior fit
        // under `per_round_gradient` / `split_penalty`, or change
        // `ridge_lambda` / `split_penalty` across resume — the prior trees
        // were trained against one residual / split-penalty geometry and the
        // new gradients would be projected through a different one, producing
        // an artifact whose stumps are inconsistent with its metadata. The
        // single-output path enforces the same contract via
        // `validate_warm_start_neutralization_contract` (lib.rs:5836); the
        // new `NeutralizationMetadata` section gives the bridge enough info
        // to do the same here. Compare the EFFECTIVE config (inert configs
        // — kind=None or SplitPenalty-with-zero-penalty — collapse to None,
        // mirroring `effective_neutralization_config` in joint.rs).
        let prior_effective = prior_model
            .neutralization_metadata
            .as_ref()
            .map(|m| m.config);
        let curr_effective = parsed_neutralization_config.filter(|cfg| {
            cfg.kind != alloygbm_core::NeutralizationKind::None
                && !(cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty
                    && cfg.split_penalty == 0.0)
        });
        match (prior_effective, curr_effective) {
            (None, None) => {}
            (Some(prior), Some(curr)) if prior == curr => {}
            (Some(prior), Some(curr)) => {
                return Err(PyValueError::new_err(format!(
                    "warm-start neutralization contract violated: prior fit used \
                     neutralization (kind={:?}, ridge_lambda={}, split_penalty={}) \
                     but current fit uses (kind={:?}, ridge_lambda={}, split_penalty={}). \
                     The prior trees were trained in a different residual / \
                     split-penalty space; resuming under a different config would \
                     produce inconsistent gradients. Use the same neutralization \
                     parameters for both fits, or fit fresh without warm_start=True.",
                    prior.kind,
                    prior.ridge_lambda,
                    prior.split_penalty,
                    curr.kind,
                    curr.ridge_lambda,
                    curr.split_penalty,
                )));
            }
            (Some(prior), None) => {
                return Err(PyValueError::new_err(format!(
                    "warm-start neutralization contract violated: prior fit used \
                     neutralization (kind={:?}) but current fit has \
                     neutralization='none'. Resuming an in-residual model in raw \
                     gradient space would silently mis-train. Pass the same \
                     neutralization config to the resume fit, or fit fresh \
                     without warm_start=True.",
                    prior.kind
                )));
            }
            (None, Some(curr)) => {
                return Err(PyValueError::new_err(format!(
                    "warm-start neutralization contract violated: prior fit was \
                     unneutralized but current fit requests neutralization \
                     (kind={:?}). Resuming raw-gradient trees in a residual space \
                     would silently mis-train. Either fit the prior model with \
                     the same neutralization, or fit fresh without warm_start=True.",
                    curr.kind
                )));
            }
        }
        // v0.10.4: extract MorphBoost EMA snapshot from the prior artifact's
        // `morph_metadata` section. The engine writes this whenever
        // MorphBoost is active and warm-resume requires it to byte-match a
        // fresh longer fit. When the prior fit didn't use MorphBoost,
        // `morph_metadata` is None and the new fit re-seeds with defaults.
        let initial_ema_stats = prior_model
            .morph_metadata
            .as_ref()
            .map(|m| m.ema_stats.clone())
            .filter(|stats| !stats.is_empty());
        Some(alloygbm_engine::joint::JointWarmStartState {
            baselines,
            stumps: prior_model.stumps,
            initial_rounds_completed: rounds,
            // Pass None — when DART is active, the engine reconstructs
            // per-tree weights from per-stump `tree_weight` (mirrors
            // the multiclass warm-start path).
            initial_dart_tree_weights: None,
            initial_ema_stats,
        })
    } else {
        None
    };

    let summary = alloygbm_engine::joint::fit_joint_multi_output_with_warm_start(
        &params,
        feature_count,
        &prepared.binned_matrix,
        &targets_per_output,
        group_id.as_deref(),
        &per_output_objective,
        n_estimators,
        &cat_features,
        warm_start,
        parsed_factor_exposures.as_ref(),
    )
    .map_err(PyValueError::new_err)?;

    let bytes = summary
        .model
        .to_artifact_bytes()
        .map_err(engine_error_to_pyerr)?;

    Ok((
        bytes,
        summary.baselines,
        feature_count,
        summary.rounds_completed,
    ))
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_class::<NativePredictorHandle>()?;
    m.add_class::<JointPredictorHandle>()?;
    m.add_class::<NativeContinuousBinningMetadata>()?;
    m.add_class::<NativeTrainingSummary>()?;
    m.add_class::<NativeTrainingResult>()?;
    m.add_class::<NativeIterationDiagnostics>()?;
    m.add_function(wrap_pyfunction!(native_runtime_info, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch_dense, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(
        predictor_predict_batch_canonical_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(
        shap_explain_interactions_dense_with_binning,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance, m)?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_dense_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(
        shap_global_importance_dense_with_binning,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact_dense, m)?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact_with_summary, m)?)?;
    m.add_function(wrap_pyfunction!(
        train_regression_artifact_dense_with_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        train_regression_artifact_dense_with_summary_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(train_joint_multi_label_ranker, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_TRAIN_ROUNDS, MAX_CONTINUOUS_QUANTIZED_BIN_U8};
    use crate::categorical_bridge::{flatten_rows, resolve_categorical_spec};
    use crate::predict::{
        predictor_predict_batch_canonical_impl, predictor_predict_batch_impl,
        shap_explain_rows_impl, shap_global_importance_impl,
    };
    use crate::quantization::ContinuousBinningStrategy;
    use crate::train::train_regression_artifact_with_summary_dense_impl;
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_categorical::TargetEncoderConfig;
    use alloygbm_core::{
        BinnedMatrix, DatasetMatrix, FactorExposureMatrix, FactorNeutralizationConfig,
        LeafModelKind, ModelSectionKind, NeutralizationKind, TrainParams, TrainingDataset,
        TreeGrowth, deserialize_model_artifact_v1, serialize_model_artifact_v1,
    };
    use alloygbm_engine::{
        CategoricalTargetEncodingSpec, EngineError, SquaredErrorObjective, Trainer,
        TrainingPolicyMode,
    };

    fn train_regression_artifact_impl(
        rows: &[Vec<f32>],
        targets: &[f32],
        params: TrainParams,
        rounds: usize,
        time_index: Option<Vec<i64>>,
        categorical_spec: Option<CategoricalTargetEncodingSpec>,
        training_policy: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> Result<Vec<u8>, EngineError> {
        let (dense_values, row_count, feature_count) = flatten_rows(rows)?;
        train_regression_artifact_with_summary_dense_impl(
            &dense_values,
            row_count,
            feature_count,
            targets,
            None, // sample_weights
            None, // group_id
            None, // factor_exposures
            None,
            None,
            None,
            None, // validation_sample_weights
            None, // validation_group_id
            params,
            rounds,
            time_index,
            None,
            categorical_spec.into_iter().collect(),
            Vec::new(),
            training_policy,
            store_node_debug_stats,
            ContinuousBinningStrategy::Linear,
            MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1,
            "squared_error",
            None, // init_artifact_bytes
            None, // num_classes
            None, // custom_objective_fn
            None, // custom_loss_fn
            None, // custom_metric_fn
            0,    // max_cat_threshold
        )
        .map(|result| result.artifact_bytes)
    }

    fn quality_fixture_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    4.0, 0.0, //
                    5.0, 0.0, //
                    6.0, 0.0, //
                    7.0, 0.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn quality_fixture_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            8,
            2,
            7,
            vec![
                0, 0, //
                1, 0, //
                2, 0, //
                3, 0, //
                4, 0, //
                5, 0, //
                6, 0, //
                7, 0, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn binned_matrix_from_fixture_dataset(dataset: &TrainingDataset) -> BinnedMatrix {
        let mut bins = Vec::with_capacity(dataset.row_count() * dataset.matrix.feature_count);
        for row in 0..dataset.row_count() {
            for feature in 0..dataset.matrix.feature_count {
                let value = dataset.matrix.values[row * dataset.matrix.feature_count + feature];
                bins.push(value as u8);
            }
        }
        BinnedMatrix::new(dataset.row_count(), dataset.matrix.feature_count, 3, bins)
            .expect("binned matrix is valid")
    }

    fn fixture_rows(dataset: &TrainingDataset) -> Vec<Vec<f32>> {
        dataset
            .matrix
            .values
            .chunks(dataset.matrix.feature_count)
            .map(|row| row.to_vec())
            .collect()
    }

    fn fixture_params() -> TrainParams {
        TrainParams {
            seed: 7,
            deterministic: true,
            learning_rate: 0.3,
            max_depth: 2,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
            monotone_constraints: Vec::new(),
            feature_weights: Vec::new(),
            interaction_constraints: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
            leaf_model: LeafModelKind::Constant,
            leaf_solver: alloygbm_core::LeafSolverKind::Standard,
            dro_config: None,
            neutralization_config: None,
            boosting_mode: alloygbm_core::BoostingMode::Standard,
            tweedie_variance_power: 1.5,
            quantile_alpha: 0.5,
        }
    }

    fn fixture_categorical_values() -> Vec<String> {
        vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ]
    }

    fn train_fixture_model() -> (alloygbm_engine::TrainedModel, TrainingDataset) {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(
                &dataset,
                &binned,
                &backend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("training succeeds");
        (model, dataset)
    }

    fn legacy_trees_only_artifact_bytes() -> (Vec<u8>, Vec<Vec<f32>>) {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let strict_artifact = model.to_artifact_bytes().expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
        let trees_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists");
        let legacy_artifact = serialize_model_artifact_v1(
            &parsed.contract.metadata,
            &[(ModelSectionKind::Trees, trees_payload)],
        )
        .expect("legacy artifact serializes");
        (legacy_artifact, rows)
    }

    #[test]
    fn binding_bridge_predictions_match_engine_predictions() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let rows = fixture_rows(&dataset);
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 2)
            .expect("training succeeds");

        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("bridge predicts");

        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn train_bridge_artifact_predictions_match_engine_predictions() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);

        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("bridge training succeeds");
        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("bridge predicts");

        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn canonical_bridge_predictions_match_engine_for_strict_artifacts() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let strict_artifact = model.to_artifact_bytes().expect("artifact serializes");

        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let canonical_predictions = predictor_predict_batch_canonical_impl(&strict_artifact, &rows)
            .expect("canonical bridge predicts");

        assert_eq!(canonical_predictions, engine_predictions);
    }

    #[test]
    fn canonical_bridge_rejects_legacy_trees_only_artifacts() {
        let (legacy_artifact, rows) = legacy_trees_only_artifact_bytes();
        let result = predictor_predict_batch_canonical_impl(&legacy_artifact, &rows);
        assert!(matches!(
            result,
            Err(alloygbm_predictor::PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn train_bridge_categorical_path_matches_engine_predictions() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let rows = fixture_rows(&dataset);
        let categorical_spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: fixture_categorical_values(),
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };

        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let engine_model = trainer
            .fit_iterations_with_single_target_encoded_feature(
                &dataset,
                &binned,
                &categorical_spec,
                &backend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("categorical engine training succeeds");
        let bridge_artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            Some(categorical_spec.clone()),
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("categorical bridge training succeeds");

        let bridge_model =
            alloygbm_engine::TrainedModel::from_artifact_bytes(&bridge_artifact).expect("parses");
        assert!(bridge_model.categorical_state.is_some());

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&bridge_artifact, &rows).expect("bridge predicts");
        assert_eq!(bridge_predictions, engine_predictions);
    }

    fn target_encoding_factor_loaded_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    2.0, 1.0, //
                    3.0, 1.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: Some(
                FactorExposureMatrix::new(8, 1, vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
                    .expect("factor exposures are valid"),
            ),
        }
    }

    #[test]
    fn train_bridge_pre_target_categorical_encoding_matches_engine_residualized_targets() {
        let dataset = target_encoding_factor_loaded_dataset();
        let binned = binned_matrix_from_fixture_dataset(&dataset);
        let rows = fixture_rows(&dataset);
        let categorical_spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: vec![
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
            ],
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };
        let params = TrainParams {
            neutralization_config: Some(FactorNeutralizationConfig {
                kind: NeutralizationKind::PreTarget,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..fixture_params()
        };

        let engine_model = Trainer::new(params.clone())
            .expect("params are valid")
            .fit_iterations_with_single_target_encoded_feature(
                &dataset,
                &binned,
                &categorical_spec,
                &CpuBackend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("engine training succeeds");
        let bridge_artifact = train_regression_artifact_with_summary_dense_impl(
            &dataset.matrix.values,
            dataset.row_count(),
            dataset.matrix.feature_count,
            &dataset.targets,
            None,
            None,
            dataset.factor_exposures.clone(),
            None,
            None,
            None,
            None,
            None,
            params,
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            vec![categorical_spec],
            Vec::new(),
            TrainingPolicyMode::Manual,
            false,
            ContinuousBinningStrategy::Linear,
            MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1,
            "squared_error",
            None,
            None,
            None,
            None,
            None,
            0,
        )
        .expect("bridge training succeeds")
        .artifact_bytes;

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&bridge_artifact, &rows).expect("bridge predicts");
        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn train_bridge_rejects_partial_categorical_arguments() {
        let rows = fixture_rows(&quality_fixture_dataset());

        let missing_values = resolve_categorical_spec(Some(1), None, 20.0, 1, false, 8);
        assert!(matches!(
            missing_values,
            Err(alloygbm_engine::EngineError::ContractViolation(_))
        ));

        let missing_index = resolve_categorical_spec(
            None,
            Some(fixture_categorical_values()),
            20.0,
            1,
            false,
            rows.len(),
        );
        assert!(matches!(
            missing_index,
            Err(alloygbm_engine::EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn shap_bridge_explain_rows_matches_model_additivity() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        let (expected_value, values) =
            shap_explain_rows_impl(&artifact, &rows).expect("shap bridge explains");
        assert_eq!(values.len(), rows.len());
        assert_eq!(values[0].len(), dataset.matrix.feature_count);

        let predictions = predictor_predict_batch_impl(&artifact, &rows).expect("predicts");
        for (row_values, prediction) in values.iter().zip(predictions.iter()) {
            let reconstructed = expected_value + row_values.iter().sum::<f32>();
            assert!((reconstructed - prediction).abs() <= 1e-5);
        }
    }

    #[test]
    fn shap_bridge_global_importance_is_sorted_descending() {
        let (_model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("bridge training succeeds");

        let global =
            shap_global_importance_impl(&artifact, &rows).expect("global importance computes");
        assert_eq!(global.len(), dataset.matrix.feature_count);
        for (name, value) in &global {
            assert!(name.starts_with('f'));
            assert!(*value >= 0.0);
        }
        for pair in global.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn train_bridge_accepts_continuous_float_rows() {
        let rows = vec![
            vec![-2.7, 0.10],
            vec![0.20, 1.90],
            vec![3.60, 2.20],
            vec![8.40, 5.50],
            vec![15.25, 9.10],
            vec![30.75, 12.80],
        ];
        let targets = vec![-2.0, -0.5, 0.5, 1.5, 3.0, 6.0];

        let artifact = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("continuous rows should train");

        let predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("continuous rows should predict");
        assert_eq!(predictions.len(), rows.len());
    }

    #[test]
    fn train_bridge_quantization_is_deterministic_for_continuous_rows() {
        let rows = vec![
            vec![-1.5, 0.25],
            vec![-0.6, 0.75],
            vec![0.4, 1.20],
            vec![1.4, 1.80],
            vec![2.6, 3.40],
            vec![5.9, 8.10],
        ];
        let targets = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        let artifact_a = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("first deterministic training succeeds");
        let artifact_b = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("second deterministic training succeeds");

        assert_eq!(artifact_a, artifact_b);
    }

    #[test]
    fn train_bridge_pre_binned_path_rejects_u16_overflow() {
        let rows = vec![vec![70000.0, 0.0], vec![1.0, 0.0]];
        let targets = vec![0.0, 1.0];
        let result = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            1,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        );
        assert!(matches!(
            result,
            Err(alloygbm_engine::EngineError::ContractViolation(message))
            if message.contains("exceeds max supported bin")
        ));
    }

    #[test]
    fn train_bridge_large_round_counts_remain_supported_via_round_cap() {
        let dataset = quality_fixture_dataset();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            4096,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("max supported round count should train");
        assert!(!artifact.is_empty());
    }

    #[test]
    fn train_bridge_can_store_node_debug_stats_section() {
        let dataset = quality_fixture_dataset();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            true,
        )
        .expect("bridge training succeeds");
        let parsed = deserialize_model_artifact_v1(&artifact).expect("artifact parses");
        assert!(
            parsed
                .sections
                .iter()
                .any(|section| section.descriptor.kind == ModelSectionKind::NodeDebugStats)
        );
    }
}
