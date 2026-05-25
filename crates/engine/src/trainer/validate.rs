//! Pre-training and per-round validation helpers, plus small dataset-shape
//! utility functions consumed by the trainer.

use alloygbm_core::{
    BinnedMatrix, FactorExposureMatrix, GradientPair, NeutralizationKind, PartitionResult,
    TrainParams, TrainingDataset, validate_binned_matrix,
};

use crate::error::{EngineError, EngineResult};
use crate::factor::apply_pre_target_neutralization;
use crate::split_options::FactorSplitContext;
use crate::traits::ObjectiveOps;

/// Compute per-feature column means from a row-major raw feature matrix.
///
/// Returns `None` when the matrix has no rows, no features, or its `values`
/// vector is empty (metadata-only datasets).  Non-finite cells are skipped per
/// column so a stray NaN/Inf doesn't poison the entire mean.
pub(crate) fn compute_feature_means_from_matrix(
    values: &[f32],
    feature_count: usize,
    row_count: usize,
) -> Option<Vec<f32>> {
    if feature_count == 0 || row_count == 0 || values.len() < row_count * feature_count {
        return None;
    }
    let mut sums = vec![0.0_f64; feature_count];
    let mut counts = vec![0_u64; feature_count];
    for row in 0..row_count {
        let base = row * feature_count;
        for j in 0..feature_count {
            let v = values[base + j];
            if v.is_finite() {
                sums[j] += v as f64;
                counts[j] += 1;
            }
        }
    }
    let means: Vec<f32> = sums
        .iter()
        .zip(counts.iter())
        .map(|(s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
        .collect();
    Some(means)
}

pub(crate) fn validate_neutralization_fit_contract<O: ObjectiveOps>(
    params: &TrainParams,
    dataset: &TrainingDataset,
    objective: &O,
) -> EngineResult<()> {
    validate_neutralization_fit_contract_for_support(
        params,
        dataset,
        objective.supports_pre_target_neutralization(),
    )
}

pub(crate) fn validate_neutralization_fit_contract_for_support(
    params: &TrainParams,
    dataset: &TrainingDataset,
    supports_pre_target_neutralization: bool,
) -> EngineResult<()> {
    let Some(config) = params.neutralization_config else {
        if dataset.factor_exposures.is_some() {
            return Err(EngineError::ContractViolation(
                "factor_exposures were provided but neutralization='none'".to_string(),
            ));
        }
        return Ok(());
    };
    let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
        EngineError::ContractViolation(
            "factor_exposures are required when neutralization is active".to_string(),
        )
    })?;
    if exposures.row_count != dataset.row_count() {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures row_count {} does not match training row_count {}",
            exposures.row_count,
            dataset.row_count()
        )));
    }
    if config.kind == NeutralizationKind::PreTarget && !supports_pre_target_neutralization {
        return Err(EngineError::ContractViolation(
            "neutralization='pre_target' is only supported for GBMRegressor squared-error training"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_warm_start_neutralization_contract(
    params: &TrainParams,
    has_warm_start: bool,
    dataset: &TrainingDataset,
) -> EngineResult<()> {
    if !has_warm_start {
        return Ok(());
    }
    let Some(config) = params.neutralization_config else {
        return Ok(());
    };
    // Per-round and split-penalty modes project the gradient (or its split
    // contribution) against the factor space on every round.  Continuing
    // training without supplying the same exposures would silently change
    // which directions are projected away — almost certainly not what the
    // caller wants, and not equivalent to fitting `N+M` rounds from scratch.
    // We therefore require an exposures matrix; the caller is responsible
    // for passing the same one used for the initial fit (the Python wrapper
    // surfaces this contract).  `pre_target` is idempotent under repeated
    // residualization against the same exposures so it falls under the same
    // requirement.
    match config.kind {
        NeutralizationKind::None => Ok(()),
        NeutralizationKind::PreTarget
        | NeutralizationKind::PerRoundGradient
        | NeutralizationKind::SplitPenalty => {
            if dataset.factor_exposures.is_none() {
                return Err(EngineError::ContractViolation(
                    "neutralized warm-start training requires factor_exposures to be supplied; pass the same matrix used for the initial fit"
                        .to_string(),
                ));
            }
            Ok(())
        }
    }
}

pub(crate) fn prepare_pre_target_training_dataset(
    params: &TrainParams,
    dataset: &TrainingDataset,
) -> EngineResult<Option<TrainingDataset>> {
    let Some(config) = params.neutralization_config else {
        return Ok(None);
    };
    if config.kind != NeutralizationKind::PreTarget {
        return Ok(None);
    }
    let mut owned_dataset = dataset.clone();
    apply_pre_target_neutralization(&mut owned_dataset, config.ridge_lambda)?;
    Ok(Some(owned_dataset))
}

pub(crate) fn gradient_neutralization_config(
    params: &TrainParams,
) -> Option<alloygbm_core::FactorNeutralizationConfig> {
    params.neutralization_config.filter(|config| {
        matches!(
            config.kind,
            NeutralizationKind::PerRoundGradient | NeutralizationKind::SplitPenalty
        )
    })
}

pub(crate) fn factor_split_context_for_node<'a>(
    params: &TrainParams,
    binned_matrix: &'a BinnedMatrix,
    exposures: Option<&'a FactorExposureMatrix>,
    row_indices: &'a [u32],
) -> Option<FactorSplitContext<'a>> {
    let config = params.neutralization_config?;
    if config.kind != NeutralizationKind::SplitPenalty || config.split_penalty == 0.0 {
        return None;
    }
    Some(FactorSplitContext {
        binned_matrix,
        exposures: exposures?,
        row_indices,
        factor_penalty: config.split_penalty,
    })
}

pub(crate) fn validate_gradient_pairs(
    gradients: &[GradientPair],
    row_count: usize,
) -> EngineResult<()> {
    validate_gradient_pair_length(gradients, row_count)?;
    for gradient in gradients {
        if !gradient.grad.is_finite() || !gradient.hess.is_finite() || gradient.hess <= 0.0 {
            return Err(EngineError::ContractViolation(
                "objective produced invalid gradient/hessian values".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_gradient_pair_length(
    gradients: &[GradientPair],
    row_count: usize,
) -> EngineResult<()> {
    if gradients.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "objective returned {} gradients for row_count {}",
            gradients.len(),
            row_count
        )));
    }
    Ok(())
}

pub(crate) fn validate_training_alignment(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<()> {
    validate_binned_matrix(binned_matrix)?;
    if dataset.row_count() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "dataset row_count {} does not match binned row_count {}",
            dataset.row_count(),
            binned_matrix.row_count
        )));
    }
    if dataset.matrix.feature_count != binned_matrix.feature_count {
        return Err(EngineError::ContractViolation(format!(
            "dataset feature_count {} does not match binned feature_count {}",
            dataset.matrix.feature_count, binned_matrix.feature_count
        )));
    }
    Ok(())
}

pub(crate) fn validate_partition_cover(
    row_count: usize,
    partition: &PartitionResult,
) -> EngineResult<()> {
    if partition.left_row_indices.is_empty() || partition.right_row_indices.is_empty() {
        return Err(EngineError::ContractViolation(
            "split partition produced empty branch".to_string(),
        ));
    }
    if partition.left_row_indices.len() + partition.right_row_indices.len() != row_count {
        return Err(EngineError::ContractViolation(
            "split partition does not cover all rows".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn binned_feature_density(binned_matrix: &BinnedMatrix) -> f32 {
    let bin_count = binned_matrix.max_bin as usize + 1;
    let feature_count = binned_matrix.feature_count;
    let total_slots = feature_count.saturating_mul(bin_count);
    if total_slots == 0 {
        return 0.0;
    }

    let mut seen = vec![false; total_slots];
    for row_index in 0..binned_matrix.row_count {
        let row_base = row_index * feature_count;
        for feature_index in 0..feature_count {
            let bin = binned_matrix.row_bin(row_base + feature_index) as usize;
            seen[feature_index * bin_count + bin] = true;
        }
    }
    let occupied = seen.into_iter().filter(|value| *value).count();
    occupied as f32 / total_slots as f32
}

pub(crate) fn target_variance(
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    if targets.is_empty() {
        return Err(EngineError::ContractViolation(
            "targets cannot be empty".to_string(),
        ));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for index in 0..targets.len() {
        let weight = sample_weights.map_or(1.0, |weights| weights[index]);
        weighted_sum += targets[index] * weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return Err(EngineError::ContractViolation(
            "sample weight sum must be greater than 0".to_string(),
        ));
    }

    let mean = weighted_sum / weight_sum;
    let mut squared_sum = 0.0_f32;
    for index in 0..targets.len() {
        let weight = sample_weights.map_or(1.0, |weights| weights[index]);
        let centered = targets[index] - mean;
        squared_sum += centered * centered * weight;
    }
    Ok(squared_sum / weight_sum)
}
