use alloygbm_core::{
    BoostingMode, DartNormalize, DartSampleType, DroConfig, DroMetric, FactorExposureMatrix,
    FactorNeutralizationConfig, LeafModelKind, LeafSolverKind, LrSchedule, MorphConfig,
    NeutralizationKind, TrainParams, TreeGrowth,
};
use alloygbm_engine::{EngineError, TrainingPolicyMode};
use alloygbm_shap::{BinningContext, ShapError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn parse_training_policy(value: &str) -> Result<TrainingPolicyMode, EngineError> {
    match value {
        "manual" => Ok(TrainingPolicyMode::Manual),
        "auto" => Ok(TrainingPolicyMode::Auto),
        other => Err(EngineError::InvalidConfig(format!(
            "training_policy must be 'auto' or 'manual', received '{other}'"
        ))),
    }
}

pub(crate) fn parse_tree_growth(value: &str) -> Result<TreeGrowth, EngineError> {
    match value {
        "level" => Ok(TreeGrowth::Level),
        "leaf" => Ok(TreeGrowth::Leaf),
        other => Err(EngineError::InvalidConfig(format!(
            "tree_growth must be 'level' or 'leaf', received '{other}'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_binning_context(
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> Result<BinningContext, ShapError> {
    match binning_kind {
        "linear" => {
            let mins = feature_mins.ok_or_else(|| {
                ShapError::InvalidInput("binning_kind='linear' requires feature_mins".to_string())
            })?;
            let maxs = feature_maxs.ok_or_else(|| {
                ShapError::InvalidInput("binning_kind='linear' requires feature_maxs".to_string())
            })?;
            let max_bin = max_data_bin.ok_or_else(|| {
                ShapError::InvalidInput("binning_kind='linear' requires max_data_bin".to_string())
            })?;
            Ok(BinningContext::Linear {
                feature_mins: mins,
                feature_maxs: maxs,
                max_data_bin: max_bin,
            })
        }
        "quantile" => {
            let cuts = feature_cuts.ok_or_else(|| {
                ShapError::InvalidInput("binning_kind='quantile' requires feature_cuts".to_string())
            })?;
            Ok(BinningContext::Quantile { feature_cuts: cuts })
        }
        "prebinned" => Ok(BinningContext::PreBinned),
        "linear_rank" => {
            let per_feature = linear_rank_per_feature.ok_or_else(|| {
                ShapError::InvalidInput(
                    "binning_kind='linear_rank' requires linear_rank_per_feature".to_string(),
                )
            })?;
            let mins = feature_mins.ok_or_else(|| {
                ShapError::InvalidInput(
                    "binning_kind='linear_rank' requires feature_mins".to_string(),
                )
            })?;
            let maxs = feature_maxs.ok_or_else(|| {
                ShapError::InvalidInput(
                    "binning_kind='linear_rank' requires feature_maxs".to_string(),
                )
            })?;
            let max_bin = max_data_bin.ok_or_else(|| {
                ShapError::InvalidInput(
                    "binning_kind='linear_rank' requires max_data_bin".to_string(),
                )
            })?;
            Ok(BinningContext::LinearRank {
                per_feature,
                feature_mins: mins,
                feature_maxs: maxs,
                max_data_bin: max_bin,
            })
        }
        other => Err(ShapError::InvalidInput(format!(
            "unknown binning_kind '{other}' (expected linear|quantile|prebinned|linear_rank)"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_train_params(
    learning_rate: f32,
    max_depth: u16,
    row_subsample: f32,
    col_subsample: f32,
    min_validation_improvement: f32,
    seed: u64,
    deterministic: bool,
    early_stopping_rounds: Option<u16>,
    min_data_in_leaf: u32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_child_hessian: f32,
    min_split_gain: f32,
    monotone_constraints: Vec<i8>,
    feature_weights: Vec<f32>,
    interaction_constraints: Vec<Vec<u32>>,
    max_leaves: Option<usize>,
    tree_growth: TreeGrowth,
    morph_config: Option<MorphConfig>,
    leaf_model: LeafModelKind,
    leaf_solver: LeafSolverKind,
    dro_config: Option<DroConfig>,
    neutralization_config: Option<FactorNeutralizationConfig>,
    boosting_mode: BoostingMode,
    tweedie_variance_power: f32,
    quantile_alpha: f32,
) -> TrainParams {
    TrainParams {
        seed,
        deterministic,
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        early_stopping_rounds,
        min_validation_improvement,
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
        morph_config,
        leaf_model,
        leaf_solver,
        dro_config,
        neutralization_config,
        boosting_mode,
        tweedie_variance_power,
        quantile_alpha,
    }
}

/// Parse Python-side boosting_mode strings + parameters into a
/// [`alloygbm_core::BoostingMode`].  Validation of parameter ranges is
/// deferred to `validate_train_params`, which produces a uniform error
/// message regardless of where the BoostingMode was constructed.
///
/// * `"standard"` — `BoostingMode::Standard`.  Ignores all rate args.
/// * `"goss"` — requires `goss_top_rate` and `goss_other_rate`.
/// * `"dart"` — requires `dart_drop_rate`, `dart_max_drop`,
///   `dart_normalize_type`, `dart_sample_type`.  Fully supported by
///   the single-output trainer as of v0.9.0; multiclass softmax + DART
///   and DART + warm-start are still rejected (v0.10.x follow-ups).
#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_boosting_mode(
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
) -> PyResult<BoostingMode> {
    match boosting_mode {
        "standard" => Ok(alloygbm_core::BoostingMode::Standard),
        "goss" => {
            let top_rate = goss_top_rate.ok_or_else(|| {
                PyValueError::new_err("boosting_mode='goss' requires goss_top_rate")
            })?;
            let other_rate = goss_other_rate.ok_or_else(|| {
                PyValueError::new_err("boosting_mode='goss' requires goss_other_rate")
            })?;
            Ok(alloygbm_core::BoostingMode::Goss {
                top_rate,
                other_rate,
            })
        }
        "dart" => {
            let drop_rate = dart_drop_rate.ok_or_else(|| {
                PyValueError::new_err("boosting_mode='dart' requires dart_drop_rate")
            })?;
            let max_drop = dart_max_drop.ok_or_else(|| {
                PyValueError::new_err("boosting_mode='dart' requires dart_max_drop")
            })?;
            let normalize_type = match dart_normalize_type.unwrap_or("tree") {
                "tree" => DartNormalize::Tree,
                "forest" => DartNormalize::Forest,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "dart_normalize_type must be 'tree' or 'forest', got {other:?}"
                    )));
                }
            };
            let sample_type = match dart_sample_type.unwrap_or("uniform") {
                "uniform" => DartSampleType::Uniform,
                "weighted" => DartSampleType::Weighted,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "dart_sample_type must be 'uniform' or 'weighted', got {other:?}"
                    )));
                }
            };
            Ok(alloygbm_core::BoostingMode::Dart {
                drop_rate,
                max_drop,
                normalize_type,
                sample_type,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "boosting_mode must be 'standard', 'goss', or 'dart', got {other:?}"
        ))),
    }
}

pub(crate) fn parse_neutralization_config(
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
) -> PyResult<Option<FactorNeutralizationConfig>> {
    let kind = match neutralization {
        "none" => NeutralizationKind::None,
        "pre_target" => NeutralizationKind::PreTarget,
        "per_round_gradient" => NeutralizationKind::PerRoundGradient,
        "split_penalty" => NeutralizationKind::SplitPenalty,
        other => {
            return Err(PyValueError::new_err(format!(
                "neutralization must be 'none', 'pre_target', 'per_round_gradient', or 'split_penalty', got '{other}'"
            )));
        }
    };
    if !factor_neutralization_lambda.is_finite() || factor_neutralization_lambda < 0.0 {
        return Err(PyValueError::new_err(
            "factor_neutralization_lambda must be finite and >= 0",
        ));
    }
    if !factor_penalty.is_finite() || factor_penalty < 0.0 {
        return Err(PyValueError::new_err(
            "factor_penalty must be finite and >= 0",
        ));
    }
    if kind != NeutralizationKind::SplitPenalty && factor_penalty != 0.0 {
        return Err(PyValueError::new_err(
            "factor_penalty is only valid with neutralization='split_penalty'",
        ));
    }
    Ok(match kind {
        NeutralizationKind::None => None,
        _ => Some(FactorNeutralizationConfig {
            kind,
            ridge_lambda: factor_neutralization_lambda,
            split_penalty: factor_penalty,
        }),
    })
}

pub(crate) fn validate_neutralization_leaf_model(
    neutralization_config: Option<FactorNeutralizationConfig>,
    leaf_model: LeafModelKind,
) -> PyResult<()> {
    if neutralization_config.is_some_and(|config| config.kind == NeutralizationKind::SplitPenalty)
        && leaf_model == LeafModelKind::Linear
    {
        return Err(PyValueError::new_err(
            "neutralization='split_penalty' requires leaf_model='constant'",
        ));
    }
    Ok(())
}

pub(crate) fn parse_factor_exposure_matrix(
    factor_exposure_values: Option<Vec<f32>>,
    factor_exposure_row_count: Option<usize>,
    factor_exposure_factor_count: Option<usize>,
) -> PyResult<Option<FactorExposureMatrix>> {
    match (
        factor_exposure_values,
        factor_exposure_row_count,
        factor_exposure_factor_count,
    ) {
        (None, None, None) => Ok(None),
        (Some(values), Some(row_count), Some(factor_count)) => {
            FactorExposureMatrix::new(row_count, factor_count, values)
                .map(Some)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        }
        _ => Err(PyValueError::new_err(
            "factor_exposure_values, factor_exposure_row_count, and \
             factor_exposure_factor_count must be provided together",
        )),
    }
}

/// Parse a `leaf_model` string into a [`alloygbm_core::LeafModelKind`].
///
/// Valid values: `"constant"` (default), `"linear"`.
pub(crate) fn parse_leaf_model(leaf_model: &str) -> pyo3::PyResult<LeafModelKind> {
    match leaf_model {
        "constant" => Ok(LeafModelKind::Constant),
        "linear" => Ok(LeafModelKind::Linear),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "leaf_model must be 'constant' or 'linear', got '{other}'"
        ))),
    }
}

pub(crate) fn parse_leaf_solver(leaf_solver: &str) -> pyo3::PyResult<LeafSolverKind> {
    match leaf_solver {
        "standard" => Ok(LeafSolverKind::Standard),
        "dro" => Ok(LeafSolverKind::Dro),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "leaf_solver must be 'standard' or 'dro', got '{other}'"
        ))),
    }
}

pub(crate) fn parse_dro_config(
    leaf_solver: LeafSolverKind,
    dro_radius: f32,
    dro_metric: &str,
) -> pyo3::PyResult<Option<DroConfig>> {
    if !dro_radius.is_finite() || dro_radius < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dro_radius must be finite and >= 0",
        ));
    }
    let metric = match dro_metric {
        "wasserstein" => DroMetric::Wasserstein,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dro_metric must be 'wasserstein', got '{other}'"
            )));
        }
    };
    Ok(match leaf_solver {
        LeafSolverKind::Standard => None,
        LeafSolverKind::Dro => Some(DroConfig {
            radius: dro_radius,
            metric,
        }),
    })
}

/// Parse a Python dict into a `MorphConfig`.
///
/// Expected keys (all optional — missing keys keep the `MorphConfig::default()` value):
///   morph_rate, evolution_pressure, morph_warmup_iters, info_score_weight,
///   depth_penalty_base, balance_penalty, lr_schedule, lr_warmup_frac
/// Parse a Python dict into a `MorphConfig`.
///
/// Expected keys (all optional — missing keys keep `MorphConfig::default()` values):
/// - `morph_rate`, `evolution_pressure`, `morph_warmup_iters`, `info_score_weight`,
///   `depth_penalty_base`, `balance_penalty`
/// - `lr_schedule`: `"constant"` (default) or `"warmup_cosine"`
/// - `lr_warmup_frac`: only meaningful when `lr_schedule="warmup_cosine"`. Providing
///   `lr_warmup_frac` without `lr_schedule="warmup_cosine"` raises `ValueError`.
pub(crate) fn parse_morph_config_from_pydict(
    dict: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> pyo3::PyResult<MorphConfig> {
    let mut cfg = MorphConfig::default();
    if let Some(v) = dict.get_item("morph_rate")? {
        cfg.morph_rate = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("evolution_pressure")? {
        cfg.evolution_pressure = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("morph_warmup_iters")? {
        cfg.morph_warmup_iters = v.extract::<u32>()?;
    }
    if let Some(v) = dict.get_item("info_score_weight")? {
        cfg.info_score_weight = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("depth_penalty_base")? {
        cfg.depth_penalty_base = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("balance_penalty")? {
        cfg.balance_penalty = v.extract::<bool>()?;
    }

    // lr_warmup_frac is only meaningful together with lr_schedule="warmup_cosine".
    let has_warmup_frac = dict.get_item("lr_warmup_frac")?.is_some();

    if let Some(v) = dict.get_item("lr_schedule")? {
        let kind: &str = v.extract()?;
        // Default warmup_frac from the WarmupCosine default (0.1).
        let warmup_frac = dict
            .get_item("lr_warmup_frac")?
            .map(|x| x.extract::<f32>())
            .transpose()?
            .unwrap_or(0.1);
        cfg.lr_schedule = match kind {
            "constant" => {
                if has_warmup_frac {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "lr_warmup_frac has no effect when lr_schedule=\"constant\"",
                    ));
                }
                LrSchedule::Constant
            }
            "warmup_cosine" => LrSchedule::WarmupCosine { warmup_frac },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown lr_schedule: {other:?} (expected \"constant\" or \"warmup_cosine\")"
                )));
            }
        };
    } else if has_warmup_frac {
        // lr_warmup_frac was provided but lr_schedule was not.
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lr_warmup_frac requires lr_schedule=\"warmup_cosine\"",
        ));
    }
    Ok(cfg)
}
