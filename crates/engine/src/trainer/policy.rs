//! Training-policy helpers: split-selection option resolution and auto-mode
//! L2 regularization triggers.

use alloygbm_core::{BinnedMatrix, LeafSolverKind, TrainParams, TrainingDataset};

use crate::env::{split_l2_env_is_configured, split_selection_options_from_env};
use crate::error::EngineResult;
use crate::split_options::SplitSelectionOptions;
use crate::trainer::validate::target_variance;
use crate::types::{IterationControls, ResolvedTrainingPolicy, TrainingPolicyMode};

pub(crate) const AUTO_SPLIT_L2_NOISY_SMALL_WIDE: f32 = 2.0;

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
    if !split_l2_env_is_configured()
        && matches!(policy_mode, Some(TrainingPolicyMode::Auto))
        && params.lambda_l2 == 0.0
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

pub(crate) fn should_apply_auto_split_l2(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<bool> {
    let row_count = dataset.row_count();
    let feature_count = binned_matrix.feature_count.max(1);
    if row_count >= 1_024 || feature_count < 8 {
        return Ok(false);
    }

    let rows_per_feature = row_count as f32 / feature_count as f32;
    if rows_per_feature >= 64.0 {
        return Ok(false);
    }

    let target_variance = target_variance(&dataset.targets, dataset.sample_weights.as_deref())?;
    Ok(target_variance > 4.0)
}
