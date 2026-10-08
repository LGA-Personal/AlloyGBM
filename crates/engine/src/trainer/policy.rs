//! Training-policy helpers: split-selection option resolution and auto-mode
//! L2 regularization triggers.

use alloygbm_core::{BinnedMatrix, LeafSolverKind, TrainParams, TrainingDataset};

use crate::env::{split_l2_env_is_configured, split_selection_options_from_env};
use crate::error::EngineResult;
use crate::split_options::SplitSelectionOptions;
use crate::types::{
    IterationControls, ObjectiveFamily, ResolvedTrainingPolicy, TrainingPolicyMode,
};

pub(crate) const AUTO_SPLIT_L2_NOISY_SMALL_WIDE: f32 = 2.0;

/// Auto-policy minimum child Hessian for binary classifiers. Mirrors
/// XGBoost's `min_child_weight=1`: a log-loss row contributes at most 0.25,
/// so a leaf needs at least four maximally uncertain rows, and more as the
/// model grows confident. Without it, a leaf holding a few rows that are
/// already classified confidently has a Hessian sum near zero and gets an
/// enormous value, which wrecks calibration (breast_cancer test log loss
/// 0.53 without it, 0.10 with it; docs/benchmarks/default_quality_v1.md).
pub(crate) const AUTO_MIN_CHILD_HESSIAN_BINARY: f32 = 1.0;
/// Below this many rows the binary Hessian floor shrinks in proportion.
/// A binary row carries at most 0.25 Hessian, so a full floor would leave
/// toy datasets (a handful of rows) with no legal split at all.
pub(crate) const AUTO_MIN_CHILD_HESSIAN_FULL_ROWS: usize = 64;
/// Auto-policy L2 for multiclass softmax. Per-class Hessians are smaller
/// than binary ones, so a Hessian floor blocks too many splits there; leaf
/// L2 gives the same protection and measurably helps every multiclass
/// fixture in the default-quality suite.
pub(crate) const AUTO_LAMBDA_L2_MULTICLASS: f32 = 1.0;

/// `(lambda_l2, min_child_hessian)` the auto policy proposes for `family`.
///
/// Regression keeps neither (outside the small-wide rule): on the
/// default-quality suite L2 helped some regression fixtures and hurt others
/// by similar amounts. Ranking keeps neither: pairwise Hessians are often far
/// below 1, so either would block most splits. Each choice was measured,
/// not assumed; see docs/benchmarks/default_quality_v1.md.
pub(crate) fn auto_leaf_regularization(family: ObjectiveFamily, row_count: usize) -> (f32, f32) {
    match family {
        ObjectiveFamily::BinaryClassification => {
            let taper = (row_count as f32 / AUTO_MIN_CHILD_HESSIAN_FULL_ROWS as f32).min(1.0);
            (0.0, AUTO_MIN_CHILD_HESSIAN_BINARY * taper)
        }
        ObjectiveFamily::MulticlassClassification => (AUTO_LAMBDA_L2_MULTICLASS, 0.0),
        ObjectiveFamily::Regression | ObjectiveFamily::Ranking => (0.0, 0.0),
    }
}

pub(crate) struct ResolvedSplitSelectionOptions {
    pub options: SplitSelectionOptions,
    pub auto_split_l2_applied: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn split_selection_options_for_training(
    params: &TrainParams,
    policy_mode: Option<TrainingPolicyMode>,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<SplitSelectionOptions> {
    Ok(split_selection_options_with_resolution_for_training(
        params,
        policy_mode,
        dataset,
        binned_matrix,
    )?
    .options)
}

pub(crate) fn split_selection_options_with_resolution_for_training(
    params: &TrainParams,
    policy_mode: Option<TrainingPolicyMode>,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<ResolvedSplitSelectionOptions> {
    split_selection_options_with_controls(params, policy_mode, dataset, binned_matrix, None)
}

/// Resolve split options, also honouring the auto policy's regularization
/// choices carried on `controls` (`auto_lambda_l2`, `auto_min_child_hessian`).
///
/// Auto regularization applies only when the caller set none of `lambda_l2`,
/// `lambda_l1` or `min_child_hessian`, and when no experiment environment
/// override is configured: explicit choices always win.
pub(crate) fn split_selection_options_with_controls(
    params: &TrainParams,
    policy_mode: Option<TrainingPolicyMode>,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    controls: Option<&IterationControls>,
) -> EngineResult<ResolvedSplitSelectionOptions> {
    let env_options = split_selection_options_from_env()?;
    let user_set_regularization =
        params.lambda_l2 != 0.0 || params.lambda_l1 != 0.0 || params.min_child_hessian != 0.0;
    let mut options = SplitSelectionOptions {
        l2_lambda: params.lambda_l2,
        l1_alpha: params.lambda_l1,
        min_child_hessian: params.min_child_hessian,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: env_options.min_leaf_magnitude,
        dro_config: params
            .dro_config
            .filter(|config| params.leaf_solver == LeafSolverKind::Dro && config.radius > 0.0),
        missing_bin_index: binned_matrix.missing_bin() as usize,
    };
    let mut auto_split_l2_applied = false;
    if !user_set_regularization {
        options.l2_lambda = env_options.l2_lambda;
        options.l1_alpha = env_options.l1_alpha;
        options.min_child_hessian = env_options.min_child_hessian;
    }
    let auto_requested = matches!(policy_mode, Some(TrainingPolicyMode::Auto))
        || controls.is_some_and(|c| c.requested_policy_mode == TrainingPolicyMode::Auto);
    let auto_regularization_allowed =
        auto_requested && !user_set_regularization && !split_l2_env_is_configured();
    if let Some(controls) = controls.filter(|_| auto_regularization_allowed) {
        if controls.auto_lambda_l2 > 0.0 {
            options.l2_lambda = controls.auto_lambda_l2;
            auto_split_l2_applied = true;
        }
        if controls.auto_min_child_hessian > 0.0 {
            options.min_child_hessian = controls.auto_min_child_hessian;
        }
    }
    if controls.is_none()
        && !split_l2_env_is_configured()
        && auto_requested
        && params.lambda_l2 == 0.0
        && options.l2_lambda < AUTO_SPLIT_L2_NOISY_SMALL_WIDE
        && should_apply_auto_split_l2(dataset, binned_matrix)?
    {
        options.l2_lambda = AUTO_SPLIT_L2_NOISY_SMALL_WIDE;
        auto_split_l2_applied = true;
    }
    Ok(ResolvedSplitSelectionOptions {
        options,
        auto_split_l2_applied,
    })
}

pub(crate) fn resolve_training_policy(
    controls: IterationControls,
    split_resolution: &ResolvedSplitSelectionOptions,
) -> ResolvedTrainingPolicy {
    ResolvedTrainingPolicy {
        requested_mode: controls.requested_policy_mode,
        requested_rounds: controls.requested_rounds,
        effective_round_cap: controls.rounds,
        min_rows_per_leaf: controls.min_rows_per_leaf,
        min_split_gain: controls.min_split_gain,
        row_subsample: controls.row_subsample,
        col_subsample: controls.col_subsample,
        auto_split_l2_applied: split_resolution.auto_split_l2_applied,
        effective_split_l2: split_resolution.options.l2_lambda,
    }
}

/// Whether `dataset` is small and wide enough for the stronger
/// [`AUTO_SPLIT_L2_NOISY_SMALL_WIDE`] leaf regularization.
///
/// The rule is purely about shape. It used to also require a target variance
/// above 4, which made the model depend on the target's units: the same data
/// in dollars and in thousands of dollars got different regularization.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn should_apply_auto_split_l2(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<bool> {
    Ok(is_small_wide(
        dataset.row_count(),
        binned_matrix.feature_count,
    ))
}

/// Fewer than 1,024 rows, at least 8 features and fewer than 64 rows per
/// feature.
pub(crate) fn is_small_wide(row_count: usize, feature_count: usize) -> bool {
    let feature_count = feature_count.max(1);
    row_count < 1_024 && feature_count >= 8 && (row_count as f32 / feature_count as f32) < 64.0
}
