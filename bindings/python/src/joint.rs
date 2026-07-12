use alloygbm_core::{TrainParams, TreeGrowth};
use alloygbm_engine::CategoricalFeatureInfo;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::engine_error_to_pyerr;
use crate::params::{
    parse_boosting_mode, parse_dro_config, parse_factor_exposure_matrix, parse_leaf_solver,
    parse_morph_config_from_pydict, parse_neutralization_config,
};
use crate::quantization::{
    parse_continuous_binning_strategy, prepare_training_matrices_from_dense_values,
};

// ---------------------------------------------------------------------------
// Joint multi-output trainer + predictor handle (v0.10.1)
// ---------------------------------------------------------------------------

/// PyO3 handle that wraps `alloygbm_engine::joint::JointPredictor` for K-output
/// prediction from Python. `MultiLabelGBMRanker(training_mode="joint")` is the
/// primary consumer.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct JointPredictorHandle {
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
    fn predict_dense(&self, py: Python<'_>, values: Vec<f32>) -> PyResult<Vec<f32>> {
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
        Ok(py.detach(|| self.predictor.predict_batch(&values, self.feature_count)))
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
    min_split_gain=0.0_f32,
    row_subsample=1.0_f32,
    col_subsample=1.0_f32,
    interaction_constraints=Vec::<Vec<u32>>::new(),
    tree_growth="level".to_string(),
    max_leaves=None::<usize>,
    categorical_feature_indices=Vec::<usize>::new(),
    max_cat_threshold=0_usize,
    continuous_binning_strategy="quantile".to_string(),
    continuous_binning_max_bins=256_usize,
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
    tweedie_variance_power=None::<f32>,
    poisson_max_delta_step=None::<f32>,
    quantile_alpha=None::<f32>,
    ranking_sigma=1.0_f32,
    lambdarank_truncation_level=None::<usize>,
    lambdarank_normalize=false,
))]
pub(crate) fn train_joint_multi_label_ranker(
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
    min_split_gain: f32,
    row_subsample: f32,
    col_subsample: f32,
    interaction_constraints: Vec<Vec<u32>>,
    tree_growth: String,
    max_leaves: Option<usize>,
    categorical_feature_indices: Vec<usize>,
    max_cat_threshold: usize,
    continuous_binning_strategy: String,
    continuous_binning_max_bins: usize,
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
    tweedie_variance_power: Option<f32>,
    poisson_max_delta_step: Option<f32>,
    quantile_alpha: Option<f32>,
    ranking_sigma: f32,
    lambdarank_truncation_level: Option<usize>,
    lambdarank_normalize: bool,
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
        let obj = match name.as_str() {
            "poisson" => JointObjective::Poisson {
                max_delta_step: poisson_max_delta_step.unwrap_or(0.7),
            },
            "gamma" => JointObjective::Gamma,
            "tweedie" => JointObjective::Tweedie {
                variance_power: tweedie_variance_power.unwrap_or(1.5),
            },
            "quantile" => JointObjective::Quantile {
                alpha: quantile_alpha.unwrap_or(0.5),
            },
            other => JointObjective::parse_with_ranking_options(
                other,
                ranking_sigma,
                lambdarank_truncation_level,
                lambdarank_normalize,
            )
            .map_err(|e| PyValueError::new_err(format!("invalid objective {name:?}: {e}")))?,
        };
        per_output_objective.push(obj);
    }
    if per_output_objective.iter().any(|o| o.requires_group()) && group_id.is_none() {
        return Err(PyValueError::new_err(
            "at least one ranking objective requires group_id",
        ));
    }

    // Build the binned matrix. The joint trainer needs only `binned_matrix`;
    // use y[:, 0] as a throwaway target since binning is target-independent
    // for all supported strategies (linear/rank/quantile derive cuts from the
    // feature distribution, not the target).
    //
    // The strategy and bin count MUST match what the Python
    // `MultiLabelGBMRanker` prediction path quantizes with (see
    // `_uses_continuous_binning` / `_quantize_rows_for_prediction`); otherwise
    // trained split thresholds (in training-bin space) would be applied to
    // differently-binned prediction inputs, silently corrupting predictions.
    let parsed_binning_strategy = parse_continuous_binning_strategy(&continuous_binning_strategy)
        .map_err(engine_error_to_pyerr)?;
    let throwaway_targets = targets_per_output[0].clone();
    let mut prepared = prepare_training_matrices_from_dense_values(
        &x_values,
        row_count,
        feature_count,
        &throwaway_targets,
        None,
        None,
        group_id.clone(),
        parsed_binning_strategy,
        continuous_binning_max_bins,
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
        tweedie_variance_power: tweedie_variance_power.unwrap_or(1.5),
        poisson_max_delta_step: poisson_max_delta_step.unwrap_or(0.7),
        quantile_alpha: quantile_alpha.unwrap_or(0.5),
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
