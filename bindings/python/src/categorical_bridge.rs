use std::collections::HashMap;

use alloygbm_categorical::{TargetEncoderConfig, fit_transform_target_encoder};
use alloygbm_core::{
    CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix, FactorExposureMatrix,
    FactorNeutralizationConfig, NeutralizationKind, TrainParams, TrainingDataset,
};
use alloygbm_engine::{CategoricalTargetEncodingSpec, EngineError};
use pyo3::prelude::*;

use crate::pyclasses::ContinuousBinningMetadataInternal;
use crate::quantization::PreparedTrainingMatrices;

#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_categorical_spec(
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    row_count: usize,
) -> Result<Option<CategoricalTargetEncodingSpec>, EngineError> {
    match (categorical_feature_index, categorical_feature_values) {
        (None, None) => Ok(None),
        (Some(_), None) => Err(EngineError::ContractViolation(
            "categorical_feature_values must be provided when categorical_feature_index is set"
                .to_string(),
        )),
        (None, Some(_)) => Err(EngineError::ContractViolation(
            "categorical_feature_index must be provided when categorical_feature_values is set"
                .to_string(),
        )),
        (Some(feature_index), Some(values)) => {
            if values.len() != row_count {
                return Err(EngineError::ContractViolation(format!(
                    "categorical_feature_values length {} does not match row_count {row_count}",
                    values.len()
                )));
            }
            Ok(Some(CategoricalTargetEncodingSpec {
                feature_index,
                values,
                config: TargetEncoderConfig {
                    smoothing: categorical_smoothing,
                    min_samples_leaf: categorical_min_samples_leaf,
                    time_aware: categorical_time_aware,
                },
            }))
        }
    }
}

/// Resolve categorical specs, preferring plural form over singular.
///
/// When plural params (`categorical_feature_indices` / `categorical_feature_values_list`)
/// are provided, they take precedence. Otherwise, the singular params are converted to
/// a one-element Vec for backward compatibility.
pub(crate) fn resolve_categorical_specs_from_params(
    // singular (backward-compat)
    categorical_feature_index: Option<usize>,
    categorical_feature_values: Option<Vec<String>>,
    // plural (preferred)
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values_list: Option<Vec<Vec<String>>>,
    // validation
    validation_categorical_feature_values: Option<Vec<String>>,
    validation_categorical_feature_values_list: Option<Vec<Vec<String>>>,
    // config
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    row_count: usize,
) -> Result<(Vec<CategoricalTargetEncodingSpec>, Vec<Vec<String>>), EngineError> {
    // Plural form takes precedence
    if categorical_feature_indices.is_some() || categorical_feature_values_list.is_some() {
        let specs = resolve_categorical_specs(
            categorical_feature_indices,
            categorical_feature_values_list,
            categorical_smoothing,
            categorical_min_samples_leaf,
            categorical_time_aware,
            row_count,
        )?;
        let val_list = validation_categorical_feature_values_list.unwrap_or_default();
        return Ok((specs, val_list));
    }
    // Fall back to singular form
    let spec = resolve_categorical_spec(
        categorical_feature_index,
        categorical_feature_values,
        categorical_smoothing,
        categorical_min_samples_leaf,
        categorical_time_aware,
        row_count,
    )?;
    let val_list = validation_categorical_feature_values
        .map(|v| vec![v])
        .unwrap_or_default();
    Ok((spec.into_iter().collect(), val_list))
}

/// Resolve multiple categorical feature specs from parallel vectors.
///
/// `categorical_feature_indices` and `categorical_feature_values_list` must be
/// provided together or both be `None`. Each entry in `values_list` corresponds
/// to the feature index at the same position in `indices`.
fn resolve_categorical_specs(
    categorical_feature_indices: Option<Vec<usize>>,
    categorical_feature_values_list: Option<Vec<Vec<String>>>,
    categorical_smoothing: f64,
    categorical_min_samples_leaf: u32,
    categorical_time_aware: bool,
    row_count: usize,
) -> Result<Vec<CategoricalTargetEncodingSpec>, EngineError> {
    match (categorical_feature_indices, categorical_feature_values_list) {
        (None, None) => Ok(Vec::new()),
        (Some(_), None) => Err(EngineError::ContractViolation(
            "categorical_feature_values_list must be provided when categorical_feature_indices is set"
                .to_string(),
        )),
        (None, Some(_)) => Err(EngineError::ContractViolation(
            "categorical_feature_indices must be provided when categorical_feature_values_list is set"
                .to_string(),
        )),
        (Some(indices), Some(values_list)) => {
            if indices.len() != values_list.len() {
                return Err(EngineError::ContractViolation(format!(
                    "categorical_feature_indices length {} does not match categorical_feature_values_list length {}",
                    indices.len(),
                    values_list.len()
                )));
            }
            // Validate uniqueness
            let mut seen = std::collections::HashSet::new();
            for &idx in &indices {
                if !seen.insert(idx) {
                    return Err(EngineError::ContractViolation(format!(
                        "duplicate categorical_feature_index: {idx}"
                    )));
                }
            }
            let config = TargetEncoderConfig {
                smoothing: categorical_smoothing,
                min_samples_leaf: categorical_min_samples_leaf,
                time_aware: categorical_time_aware,
            };
            let mut specs = Vec::with_capacity(indices.len());
            for (feature_index, values) in indices.into_iter().zip(values_list) {
                if values.len() != row_count {
                    return Err(EngineError::ContractViolation(format!(
                        "categorical_feature_values for feature {feature_index} has length {} but row_count is {row_count}",
                        values.len()
                    )));
                }
                specs.push(CategoricalTargetEncodingSpec {
                    feature_index,
                    values,
                    config: config.clone(),
                });
            }
            // Sort by feature_index for deterministic ordering
            specs.sort_by_key(|s| s.feature_index);
            Ok(specs)
        }
    }
}

pub(crate) fn flatten_rows(rows: &[Vec<f32>]) -> Result<(Vec<f32>, usize, usize), EngineError> {
    if rows.is_empty() {
        return Err(EngineError::ContractViolation(
            "rows cannot be empty".to_string(),
        ));
    }
    let feature_count = rows[0].len();
    if feature_count == 0 {
        return Err(EngineError::ContractViolation(
            "rows must include at least one feature".to_string(),
        ));
    }
    let mut dense_values = Vec::with_capacity(rows.len() * feature_count);
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != feature_count {
            return Err(EngineError::ContractViolation(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row.len()
            )));
        }
        dense_values.extend_from_slice(row);
    }
    Ok((dense_values, rows.len(), feature_count))
}

/// Inference mapping for one target-encoded categorical feature.
///
/// Training stores the feature as bin indices `0..n` (ordered by encoded
/// value), and the dense column holds the same indices as floats. The
/// binning metadata for the feature is rewritten so a float `k` quantizes
/// back to bin `k` under every strategy, so prediction only has to map each
/// category to its bin.
#[derive(Debug, Clone)]
pub(crate) struct TargetEncodedFeatureBins {
    pub(crate) feature_index: usize,
    /// Bin of each training category (its full-training encoding).
    pub(crate) category_bins: HashMap<String, u16>,
    /// Bin of an unseen category (the global target mean).
    pub(crate) unknown_bin: u16,
}

impl TargetEncodedFeatureBins {
    pub(crate) fn bin_for(&self, category: &str) -> u16 {
        self.category_bins
            .get(category)
            .copied()
            .unwrap_or(self.unknown_bin)
    }
}

/// Cut points over encoded values, at most `max_bins - 1` of them. One bin
/// per distinct value when they fit; otherwise adjacent distinct values are
/// grouped into `max_bins` runs of near-equal size. Cuts sit at midpoints
/// between neighbouring distinct values (`bin = #cuts <= value`).
pub(crate) fn target_encoding_cuts(encoded_values: &[f32], max_bins: usize) -> Vec<f32> {
    let mut unique_values = encoded_values.to_vec();
    unique_values.sort_by(f32::total_cmp);
    unique_values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let unique_count = unique_values.len();
    let midpoint = |index: usize| {
        0.5 * (f64::from(unique_values[index - 1]) + f64::from(unique_values[index])) as f32
    };
    if unique_count <= max_bins.max(1) {
        return (1..unique_count).map(midpoint).collect();
    }
    let mut cuts: Vec<f32> = (1..max_bins)
        .map(|j| (j * unique_count) / max_bins)
        .filter(|&index| index > 0)
        .map(midpoint)
        .collect();
    cuts.dedup_by(|left, right| left.to_bits() == right.to_bits());
    cuts
}

pub(crate) fn encoded_value_bin(value: f32, cuts: &[f32]) -> u16 {
    cuts.partition_point(|cut| *cut <= value) as u16
}

/// Make a float `k` quantize to bin `k` for `feature_index` under every
/// continuous binning strategy (see [`TargetEncodedFeatureBins`]).
fn set_index_bin_metadata(
    metadata: &mut ContinuousBinningMetadataInternal,
    feature_index: usize,
    max_data_bin: u16,
) {
    let top = f32::from(max_data_bin);
    if let Some(mins) = metadata.feature_mins.as_mut()
        && let Some(slot) = mins.get_mut(feature_index)
    {
        *slot = 0.0;
    }
    if let Some(maxs) = metadata.feature_maxs.as_mut()
        && let Some(slot) = maxs.get_mut(feature_index)
    {
        *slot = top;
    }
    if let Some(sorted) = metadata.feature_sorted_values.as_mut()
        && let Some(slot) = sorted.get_mut(feature_index)
    {
        *slot = (0..=max_data_bin).map(f32::from).collect();
    }
    if let Some(cuts) = metadata.feature_quantile_cuts.as_mut()
        && let Some(slot) = cuts.get_mut(feature_index)
    {
        *slot = (0..max_data_bin).map(|k| f32::from(k) + 0.5).collect();
    }
    if let Some(flags) = metadata.feature_linear_rank_flags.as_mut()
        && let Some(slot) = flags.get_mut(feature_index)
    {
        *slot = false;
    }
}

fn bridge_cholesky_lower(mut matrix: Vec<f64>, k: usize) -> Result<Vec<f64>, EngineError> {
    for i in 0..k {
        for j in 0..=i {
            let mut sum = matrix[i * k + j];
            for p in 0..j {
                sum -= matrix[i * k + p] * matrix[j * k + p];
            }
            if i == j {
                if sum <= 1e-12 {
                    return Err(EngineError::ContractViolation(
                        "factor exposure Gram matrix is singular; increase factor_neutralization_lambda"
                            .to_string(),
                    ));
                }
                matrix[i * k + j] = sum.sqrt();
            } else {
                matrix[i * k + j] = sum / matrix[j * k + j];
            }
        }
        for j in i + 1..k {
            matrix[i * k + j] = 0.0;
        }
    }
    Ok(matrix)
}

fn bridge_solve_cholesky(lower: &[f64], rhs: &[f64], k: usize) -> Result<Vec<f64>, EngineError> {
    if rhs.len() != k {
        return Err(EngineError::ContractViolation(
            "factor projection rhs length must match factor count".to_string(),
        ));
    }

    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut sum = rhs[i];
        for (j, y_j) in y.iter().enumerate().take(i) {
            sum -= lower[i * k + j] * *y_j;
        }
        y[i] = sum / lower[i * k + i];
    }

    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for (j, x_j) in x.iter().enumerate().take(k).skip(i + 1) {
            sum -= lower[j * k + i] * *x_j;
        }
        x[i] = sum / lower[i * k + i];
    }
    Ok(x)
}

fn bridge_residualize_values_in_place(
    values: &mut [f32],
    exposures: &FactorExposureMatrix,
    weights: Option<&[f32]>,
    ridge_lambda: f32,
) -> Result<(), EngineError> {
    if values.len() != exposures.row_count {
        return Err(EngineError::ContractViolation(
            "value length must match factor_exposures row_count".to_string(),
        ));
    }
    if let Some(weights) = weights
        && weights.len() != exposures.row_count
    {
        return Err(EngineError::ContractViolation(
            "sample_weight length must match factor_exposures row_count".to_string(),
        ));
    }

    let k = exposures.factor_count;
    let mut gram = vec![0.0_f64; k * k];
    let mut rhs = vec![0.0_f64; k];
    for (row, value) in values.iter().enumerate().take(exposures.row_count) {
        let weight = weights.map_or(1.0_f64, |sample_weights| f64::from(sample_weights[row]));
        let factors = exposures.row(row)?;
        for a in 0..k {
            rhs[a] += weight * f64::from(factors[a]) * f64::from(*value);
            for b in 0..=a {
                gram[a * k + b] += weight * f64::from(factors[a]) * f64::from(factors[b]);
            }
        }
    }
    for i in 0..k {
        gram[i * k + i] += f64::from(ridge_lambda);
    }
    let lower = bridge_cholesky_lower(gram, k)?;
    let coefficients = bridge_solve_cholesky(&lower, &rhs, k)?;

    let mut residualized = Vec::with_capacity(values.len());
    for (row, value) in values.iter().enumerate() {
        let projected = exposures
            .row(row)?
            .iter()
            .zip(coefficients.iter())
            .map(|(factor, coefficient)| f64::from(*factor) * coefficient)
            .sum::<f64>();
        let residual = (f64::from(*value) - projected) as f32;
        if !residual.is_finite() {
            return Err(EngineError::ContractViolation(
                "residualized value must be finite".to_string(),
            ));
        }
        residualized.push(residual);
    }
    values.copy_from_slice(&residualized);
    Ok(())
}

pub(crate) fn apply_bridge_pre_target_neutralization(
    prepared: &mut PreparedTrainingMatrices,
    config: FactorNeutralizationConfig,
) -> Result<(), EngineError> {
    if config.kind != NeutralizationKind::PreTarget {
        return Ok(());
    }
    let exposures = prepared.dataset.factor_exposures.as_ref().ok_or_else(|| {
        EngineError::ContractViolation(
            "factor_exposures are required when neutralization is active".to_string(),
        )
    })?;
    bridge_residualize_values_in_place(
        &mut prepared.dataset.targets,
        exposures,
        prepared.dataset.sample_weights.as_deref(),
        config.ridge_lambda,
    )?;
    prepared.dataset.factor_exposures = None;
    Ok(())
}

pub(crate) fn validate_bridge_pre_target_neutralization_support(
    params: &TrainParams,
    objective: &str,
    custom_objective_fn: Option<&Py<PyAny>>,
    has_validation_targets: bool,
) -> Result<(), EngineError> {
    let is_pre_target = params
        .neutralization_config
        .is_some_and(|config| config.kind == NeutralizationKind::PreTarget);
    if is_pre_target && (objective != "squared_error" || custom_objective_fn.is_some()) {
        return Err(EngineError::ContractViolation(
            "neutralization='pre_target' is only supported for GBMRegressor squared-error training"
                .to_string(),
        ));
    }
    if is_pre_target && has_validation_targets {
        return Err(EngineError::ContractViolation(
            "neutralization='pre_target' does not support validation targets in this release because validation factor_exposures are not accepted"
                .to_string(),
        ));
    }
    Ok(())
}

/// Encode multiple categorical features in the training matrices via target encoding.
///
/// Each feature becomes ordered bin indices (see [`TargetEncodedFeatureBins`]);
/// the returned mappings reproduce those bins at validation and prediction.
pub(crate) fn apply_categorical_encoding_to_training_matrices_multi(
    prepared: PreparedTrainingMatrices,
    categorical_specs: &[CategoricalTargetEncodingSpec],
) -> Result<
    (
        PreparedTrainingMatrices,
        CategoricalStatePayloadV1,
        Vec<TargetEncodedFeatureBins>,
    ),
    EngineError,
> {
    if categorical_specs.is_empty() {
        let empty_state = CategoricalStatePayloadV1 {
            format_version: CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: false,
            categorical_feature_indices: Vec::new(),
        };
        return Ok((prepared, empty_state, Vec::new()));
    }

    let row_count = prepared.dataset.row_count();
    let feature_count = prepared.dataset.matrix.feature_count;
    let mut dense_values = prepared.dataset.matrix.values.clone();
    let mut binned_matrix = prepared.binned_matrix;
    let mut metadata = prepared.metadata;
    let mut max_bin = binned_matrix.max_bin;
    let mut any_time_aware = false;
    // Data bins are 0..missing_bin; the missing bin itself stays reserved.
    let missing_bin = binned_matrix.missing_bin();
    let max_data_bin = missing_bin.saturating_sub(1);
    let mut feature_bins = Vec::with_capacity(categorical_specs.len());

    for spec in categorical_specs {
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
        if spec.config.time_aware {
            any_time_aware = true;
        }

        let (encoder_state, encoded_values) = fit_transform_target_encoder(
            &spec.config,
            &spec.values,
            &prepared.dataset.targets,
            prepared.dataset.time_index.as_deref(),
        )
        .map_err(|error| EngineError::ContractViolation(error.to_string()))?;
        if let Some(index) = encoded_values.iter().position(|value| !value.is_finite()) {
            return Err(EngineError::ContractViolation(format!(
                "encoded value at index {index} must be finite"
            )));
        }

        let cuts = target_encoding_cuts(&encoded_values, usize::from(missing_bin));
        for (row_index, &encoded_value) in encoded_values.iter().enumerate() {
            let bin = encoded_value_bin(encoded_value, &cuts);
            max_bin = max_bin.max(bin);
            dense_values[row_index * feature_count + spec.feature_index] = f32::from(bin);
            binned_matrix.set_bin(row_index, spec.feature_index, bin);
        }
        set_index_bin_metadata(&mut metadata, spec.feature_index, max_data_bin);
        feature_bins.push(TargetEncodedFeatureBins {
            feature_index: spec.feature_index,
            category_bins: encoder_state
                .category_stats
                .iter()
                .map(|stats| {
                    (
                        stats.category.clone(),
                        encoded_value_bin(stats.encoded, &cuts),
                    )
                })
                .collect(),
            unknown_bin: encoded_value_bin(encoder_state.global_mean, &cuts),
        });
    }

    let categorical_state = CategoricalStatePayloadV1 {
        format_version: CATEGORICAL_STATE_FORMAT_V1,
        leakage_safe_target_encoding: any_time_aware,
        categorical_feature_indices: categorical_specs
            .iter()
            .map(|s| s.feature_index as u32)
            .collect(),
    };
    Ok((
        PreparedTrainingMatrices {
            dataset: TrainingDataset {
                matrix: DatasetMatrix::new(row_count, feature_count, dense_values)?,
                targets: prepared.dataset.targets,
                sample_weights: prepared.dataset.sample_weights,
                time_index: prepared.dataset.time_index,
                group_id: prepared.dataset.group_id,
                factor_exposures: prepared.dataset.factor_exposures,
            },
            binned_matrix: {
                binned_matrix.max_bin = max_bin;
                binned_matrix
            },
            metadata,
        },
        categorical_state,
        feature_bins,
    ))
}

/// Encode target-encoded categorical features in the validation matrices,
/// using the training bins so validation rows land where prediction will.
pub(crate) fn apply_categorical_encoding_to_validation_matrices_multi(
    prepared: PreparedTrainingMatrices,
    feature_bins: &[TargetEncodedFeatureBins],
    validation_specs: &[CategoricalTargetEncodingSpec],
) -> Result<PreparedTrainingMatrices, EngineError> {
    if feature_bins.is_empty() {
        return Ok(prepared);
    }
    if feature_bins.len() != validation_specs.len() {
        return Err(EngineError::ContractViolation(format!(
            "training specs count {} does not match validation specs count {}",
            feature_bins.len(),
            validation_specs.len()
        )));
    }

    let row_count = prepared.dataset.row_count();
    let feature_count = prepared.dataset.matrix.feature_count;
    let mut dense_values = prepared.dataset.matrix.values.clone();
    let mut binned_matrix = prepared.binned_matrix;
    let mut max_bin = binned_matrix.max_bin;

    for (bins, validation_spec) in feature_bins.iter().zip(validation_specs) {
        if validation_spec.feature_index >= feature_count {
            return Err(EngineError::ContractViolation(format!(
                "categorical feature index {} is out of bounds for feature_count {}",
                validation_spec.feature_index, feature_count
            )));
        }
        if validation_spec.values.len() != row_count {
            return Err(EngineError::ContractViolation(format!(
                "categorical values length {} does not match row_count {}",
                validation_spec.values.len(),
                row_count
            )));
        }
        for (row_index, category) in validation_spec.values.iter().enumerate() {
            let bin = bins.bin_for(category);
            max_bin = max_bin.max(bin);
            dense_values[row_index * feature_count + validation_spec.feature_index] =
                f32::from(bin);
            binned_matrix.set_bin(row_index, validation_spec.feature_index, bin);
        }
    }

    Ok(PreparedTrainingMatrices {
        dataset: TrainingDataset {
            matrix: DatasetMatrix::new(row_count, feature_count, dense_values)?,
            targets: prepared.dataset.targets,
            sample_weights: prepared.dataset.sample_weights,
            time_index: prepared.dataset.time_index,
            group_id: prepared.dataset.group_id,
            factor_exposures: prepared.dataset.factor_exposures,
        },
        binned_matrix: {
            binned_matrix.max_bin = max_bin;
            binned_matrix
        },
        metadata: prepared.metadata,
    })
}
