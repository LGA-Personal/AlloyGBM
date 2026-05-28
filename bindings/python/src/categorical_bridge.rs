use alloygbm_categorical::{
    TargetEncoderConfig, fit_target_encoder, fit_transform_target_encoder, transform_target_encoder,
};
use alloygbm_core::{
    BinnedMatrix, CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix,
    FactorExposureMatrix, FactorNeutralizationConfig, NeutralizationKind, TrainParams,
    TrainingDataset,
};
use alloygbm_engine::{CategoricalTargetEncodingSpec, EngineError};
use pyo3::prelude::*;

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

fn encode_bins_from_encoded_values(encoded_values: &[f32]) -> Result<(Vec<u8>, u16), EngineError> {
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
    Ok((bins, (unique_values.len().saturating_sub(1)) as u16))
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
pub(crate) fn apply_categorical_encoding_to_training_matrices_multi(
    prepared: PreparedTrainingMatrices,
    categorical_specs: &[CategoricalTargetEncodingSpec],
) -> Result<(PreparedTrainingMatrices, CategoricalStatePayloadV1), EngineError> {
    if categorical_specs.is_empty() {
        let empty_state = CategoricalStatePayloadV1 {
            format_version: CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: false,
            categorical_feature_indices: Vec::new(),
        };
        return Ok((prepared, empty_state));
    }

    let row_count = prepared.dataset.row_count();
    let feature_count = prepared.dataset.matrix.feature_count;
    let mut dense_values = prepared.dataset.matrix.values.clone();
    let mut bins = prepared.binned_matrix.bins.clone();
    let mut max_bin = prepared.binned_matrix.max_bin;
    let mut any_time_aware = false;

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

        let (_, encoded_values) = fit_transform_target_encoder(
            &spec.config,
            &spec.values,
            &prepared.dataset.targets,
            prepared.dataset.time_index.as_deref(),
        )
        .map_err(|error| EngineError::ContractViolation(error.to_string()))?;

        let (encoded_bins, encoded_max_bin) = encode_bins_from_encoded_values(&encoded_values)?;
        max_bin = max_bin.max(encoded_max_bin);
        for (row_index, &encoded_value) in encoded_values.iter().enumerate() {
            let offset = row_index * feature_count + spec.feature_index;
            dense_values[offset] = encoded_value;
            bins[offset] = encoded_bins[row_index];
        }
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
            binned_matrix: BinnedMatrix::new(row_count, feature_count, max_bin, bins)?,
            metadata: prepared.metadata,
        },
        categorical_state,
    ))
}

/// Encode multiple categorical features in the validation matrices via target encoding.
///
/// `training_specs` are used to fit the encoder (training values + training targets).
/// `validation_specs` provide the validation values to transform.
pub(crate) fn apply_categorical_encoding_to_validation_matrices_multi(
    prepared: PreparedTrainingMatrices,
    training_specs: &[CategoricalTargetEncodingSpec],
    validation_specs: &[CategoricalTargetEncodingSpec],
    training_targets: &[f32],
    training_time_index: Option<&[i64]>,
) -> Result<PreparedTrainingMatrices, EngineError> {
    if training_specs.is_empty() {
        return Ok(prepared);
    }
    if training_specs.len() != validation_specs.len() {
        return Err(EngineError::ContractViolation(format!(
            "training specs count {} does not match validation specs count {}",
            training_specs.len(),
            validation_specs.len()
        )));
    }

    let row_count = prepared.dataset.row_count();
    let feature_count = prepared.dataset.matrix.feature_count;
    let mut dense_values = prepared.dataset.matrix.values.clone();
    let mut bins = prepared.binned_matrix.bins.clone();
    let mut max_bin = prepared.binned_matrix.max_bin;

    for (training_spec, validation_spec) in training_specs.iter().zip(validation_specs) {
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

        let encoder_state = fit_target_encoder(
            &training_spec.config,
            &training_spec.values,
            training_targets,
            training_time_index,
        )
        .map_err(|error| EngineError::ContractViolation(error.to_string()))?;
        let encoded_values = transform_target_encoder(&encoder_state, &validation_spec.values)
            .map_err(|error| EngineError::ContractViolation(error.to_string()))?;

        let (encoded_bins, encoded_max_bin) = encode_bins_from_encoded_values(&encoded_values)?;
        max_bin = max_bin.max(encoded_max_bin);
        for (row_index, &encoded_value) in encoded_values.iter().enumerate() {
            let offset = row_index * feature_count + validation_spec.feature_index;
            dense_values[offset] = encoded_value;
            bins[offset] = encoded_bins[row_index];
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
        binned_matrix: BinnedMatrix::new(row_count, feature_count, max_bin, bins)?,
        metadata: prepared.metadata,
    })
}
