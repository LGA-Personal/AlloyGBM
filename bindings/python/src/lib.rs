#![allow(clippy::too_many_arguments)]

mod callbacks;
mod errors;
mod pyclasses;
mod quantization;
use crate::callbacks::{CustomPythonMetricCallback, CustomPythonObjective};
use crate::errors::{engine_error_to_pyerr, predictor_error_to_pyerr, shap_error_to_pyerr};
use crate::pyclasses::{
    NativeContinuousBinningMetadata, NativeIterationDiagnostics, NativeRuntimeInfo,
    NativeTrainingResult, NativeTrainingSummary, diagnostics_to_native, native_runtime_info,
};
use crate::quantization::{
    ContinuousBinningStrategy, PreparedTrainingMatrices, parse_continuous_binning_strategy,
    prepare_training_matrices_from_dense_values, prepare_validation_matrices_from_dense_values,
    quantize_dense_values_linear_inplace_wide, quantize_dense_values_linear_rank_inplace_wide,
    quantize_linear_value,
};

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_categorical::{
    TargetEncoderConfig, fit_target_encoder, fit_transform_target_encoder, transform_target_encoder,
};
use alloygbm_core::{
    BinnedMatrix, CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix,
    DenseMatrixView, FactorExposureMatrix, FactorNeutralizationConfig, NeutralizationKind,
    TrainParams, TrainingDataset, TreeGrowth,
};
use alloygbm_engine::{
    ArtifactCompatibilityMode, BinaryCrossEntropyObjective, CategoricalFeatureInfo,
    CategoricalTargetEncodingSpec, EngineError, GammaObjective, IterationRunSummary,
    LambdaMARTObjective, MultiClassIterationRunSummary, MultiClassSoftmaxObjective,
    MultiClassTrainedModel, MultiClassWarmStartState, ObjectiveOps, PairwiseRankingObjective,
    PerRoundMetricCallback, PoissonObjective, QuantileObjective, QueryRMSEObjective,
    SquaredErrorObjective, TrainedModel, Trainer, TrainingPolicyMode, TweedieObjective,
    WarmStartState, XeNDCGObjective, YetiRankObjective,
};
use alloygbm_predictor::{Predictor, PredictorError};
use alloygbm_shap::{
    BinningContext, ShapError, explain_interactions_from_artifact_bytes,
    explain_interactions_from_artifact_bytes_with_binning, explain_rows_from_artifact_bytes,
    explain_rows_from_artifact_bytes_with_binning, global_importance_from_artifact_bytes,
    global_importance_from_artifact_bytes_with_binning,
};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_TRAIN_ROUNDS: usize = 6;
const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
pub(crate) const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U8: u16 = 254;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U16: u16 = 65534;
pub(crate) const MIN_CONTINUOUS_QUANTIZED_BINS: usize = 2;
pub(crate) const LINEAR_TAIL_RANK_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK";
pub(crate) const LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO";
pub(crate) const DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD: f32 = 0.10;

fn parse_training_policy(value: &str) -> Result<TrainingPolicyMode, EngineError> {
    match value {
        "manual" => Ok(TrainingPolicyMode::Manual),
        "auto" => Ok(TrainingPolicyMode::Auto),
        other => Err(EngineError::InvalidConfig(format!(
            "training_policy must be 'auto' or 'manual', received '{other}'"
        ))),
    }
}

fn parse_tree_growth(value: &str) -> Result<TreeGrowth, EngineError> {
    match value {
        "level" => Ok(TreeGrowth::Level),
        "leaf" => Ok(TreeGrowth::Leaf),
        other => Err(EngineError::InvalidConfig(format!(
            "tree_growth must be 'level' or 'leaf', received '{other}'"
        ))),
    }
}

fn dense_rows_from_flat_values(
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

#[pyclass]
#[derive(Debug, Clone)]
struct NativePredictorHandle {
    predictor: Predictor,
}

#[pymethods]
impl NativePredictorHandle {
    #[new]
    #[pyo3(signature = (artifact_bytes, strict=true))]
    fn new(artifact_bytes: &[u8], strict: bool) -> PyResult<Self> {
        let predictor = load_predictor_from_artifact_impl(artifact_bytes, strict)
            .map_err(predictor_error_to_pyerr)?;
        Ok(Self { predictor })
    }

    fn predict_batch(&self, rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch(&rows)
            .map_err(predictor_error_to_pyerr)
    }

    fn predict_dense(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &values,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Predict from a numpy array (zero-copy). Requires float thresholds converted.
    fn predict_numpy(&self, array: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let shape = array.shape();
        let row_count = shape[0];
        let feature_count = shape[1];
        let array_view = array.as_array();
        // Access the underlying contiguous slice (zero-copy)
        let values = array_view.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be C-contiguous")
        })?;
        self.predictor
            .predict_batch_dense(values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Predict from raw f32 bytes — zero Python-to-Rust list overhead.
    /// Requires float thresholds to be converted first (convert_thresholds_to_float).
    fn predict_dense_float_bytes(
        &self,
        values_bytes: &[u8],
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_dense_bytes(values_bytes, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw f32 bytes to bins using linear scaling, then predict.
    /// Single-pass: fuses bytes→f32 conversion with quantization (one allocation).
    fn predict_dense_quantized_linear_bytes(
        &self,
        values_bytes: &[u8],
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        if !values_bytes.len().is_multiple_of(4) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "values_bytes length must be a multiple of 4 (f32)",
            ));
        }
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        // Fused bytes→f32+quantize (single allocation, parallel), then predict.
        let total = row_count * feature_count;
        let mut quantized = vec![0.0_f32; total];
        let chunk_size = 4096.max(row_count / rayon::current_num_threads().max(1));
        quantized
            .par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let rows_in_chunk = out_chunk.len() / feature_count;
                for local_row in 0..rows_in_chunk {
                    let row_index = row_start + local_row;
                    let byte_base = row_index * feature_count * 4;
                    let out_base = local_row * feature_count;
                    for fi in 0..feature_count {
                        let bi = byte_base + fi * 4;
                        let value = f32::from_ne_bytes([
                            values_bytes[bi],
                            values_bytes[bi + 1],
                            values_bytes[bi + 2],
                            values_bytes[bi + 3],
                        ]);
                        // v0.9.0 Limitation 4 fix: preserve NaN through the
                        // f32 cast so the predictor's `is_nan` check fires
                        // and routes through `default_left`.
                        out_chunk[out_base + fi] = if value.is_nan() {
                            f32::NAN
                        } else {
                            quantize_linear_value(value, feature_mins[fi], feature_maxs[fi]) as f32
                        };
                    }
                }
            });
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw float values to bins using linear scaling, then predict.
    /// Avoids the Python-side quantization loop (1.95B iterations for 2.5M×780).
    #[pyo3(signature = (values, row_count, feature_count, feature_mins, feature_maxs, max_data_bin=None))]
    fn predict_dense_quantized_linear(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: Option<u16>,
    ) -> PyResult<Vec<f32>> {
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        let mdb = max_data_bin.unwrap_or(MAX_CONTINUOUS_QUANTIZED_BIN_U8);
        let quantized = quantize_dense_values_linear_inplace_wide(
            &values,
            row_count,
            feature_count,
            &feature_mins,
            &feature_maxs,
            mdb,
        );
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds using per-feature min/max.
    /// After calling this, predict_dense works directly on raw floats — no quantization needed.
    /// `max_data_bin` is the maximum data bin index (e.g. 254 for 256 bins, 510 for 512 bins).
    fn convert_thresholds_to_float(
        &mut self,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: u16,
    ) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float(&feature_mins, &feature_maxs, max_data_bin)
            .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds using per-feature quantile cuts.
    fn convert_thresholds_to_float_quantile(
        &mut self,
        feature_cuts: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float_quantile(&feature_cuts)
            .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds for pre-binned integer data.
    fn convert_thresholds_to_float_prebinned(&mut self) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float_prebinned()
            .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw float values using selective rank (linear + rank fallback), then predict.
    #[pyo3(signature = (values, row_count, feature_count, feature_mins, feature_maxs, rank_flags, feature_sorted_values, max_data_bin=None))]
    fn predict_dense_quantized_linear_rank(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        rank_flags: Vec<bool>,
        feature_sorted_values: Vec<Vec<f32>>,
        max_data_bin: Option<u16>,
    ) -> PyResult<Vec<f32>> {
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        if rank_flags.len() != feature_count || feature_sorted_values.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rank_flags/feature_sorted_values length must match feature_count",
            ));
        }
        let mdb = max_data_bin.unwrap_or(MAX_CONTINUOUS_QUANTIZED_BIN_U8);
        let quantized = quantize_dense_values_linear_rank_inplace_wide(
            &values,
            row_count,
            feature_count,
            &feature_mins,
            &feature_maxs,
            &rank_flags,
            &feature_sorted_values,
            mdb,
        );
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    // -- Multi-class prediction -----------------------------------------------

    fn is_multiclass(&self) -> bool {
        self.predictor.is_multiclass()
    }

    fn num_classes(&self) -> Option<usize> {
        self.predictor.num_classes()
    }

    /// Multi-class prediction returning flat Vec of length n_rows * K.
    fn predict_multiclass(&self, rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_multiclass(&rows)
            .map_err(predictor_error_to_pyerr)
    }

    /// Multi-class prediction from dense flat array.
    fn predict_dense_multiclass(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_dense_multiclass(&values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Multi-class prediction from a numpy array (zero-copy).
    fn predict_numpy_multiclass(&self, array: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let shape = array.shape();
        let row_count = shape[0];
        let feature_count = shape[1];
        let array_view = array.as_array();
        let values = array_view.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be C-contiguous")
        })?;
        self.predictor
            .predict_batch_dense_multiclass(values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_categorical_spec(
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
fn resolve_categorical_specs_from_params(
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

fn flatten_rows(rows: &[Vec<f32>]) -> Result<(Vec<f32>, usize, usize), EngineError> {
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

fn apply_bridge_pre_target_neutralization(
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

fn validate_bridge_pre_target_neutralization_support(
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
fn apply_categorical_encoding_to_training_matrices_multi(
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
fn apply_categorical_encoding_to_validation_matrices_multi(
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
fn train_regression_artifact_with_summary_dense_impl(
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
    objective: &str,
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
        need_dense_values,
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
        "poisson" => run_training!(&PoissonObjective),
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
                    let mut obj = PairwiseRankingObjective::new(group_id);
                    if let Some(vg) = val_group_id {
                        obj = obj.with_validation_group(vg);
                    }
                    run_training!(&obj)
                }
                "rank_ndcg" => {
                    let mut obj = LambdaMARTObjective::new(group_id);
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
                    let mut obj = YetiRankObjective::new(group_id, 10, user_seed);
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

fn load_predictor_from_artifact_impl(
    artifact_bytes: &[u8],
    strict: bool,
) -> Result<Predictor, PredictorError> {
    if strict {
        TrainedModel::from_artifact_bytes_with_mode(
            artifact_bytes,
            ArtifactCompatibilityMode::Strict,
        )
        .map_err(|error| {
            PredictorError::ContractViolation(format!(
                "canonical predictor path requires strict dual-section artifact: {error}"
            ))
        })?;
    }
    Predictor::from_artifact_bytes(artifact_bytes)
}

fn predictor_predict_batch_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, false)?;
    predictor.predict_batch(rows)
}

fn predictor_predict_batch_dense_with_predictor(
    predictor: &Predictor,
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    predictor.predict_batch_dense(values, row_count, feature_count)
}

fn predictor_predict_batch_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, false)?;
    predictor_predict_batch_dense_with_predictor(&predictor, row_count, feature_count, values)
}

fn predictor_predict_batch_canonical_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, true)?;
    predictor.predict_batch(rows)
}

fn predictor_predict_batch_canonical_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, true)?;
    predictor_predict_batch_dense_with_predictor(&predictor, row_count, feature_count, values)
}

fn shap_explain_rows_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<(f32, Vec<Vec<f32>>), ShapError> {
    let explanation = explain_rows_from_artifact_bytes(artifact_bytes, rows)?;
    Ok((explanation.expected_value, explanation.values))
}

fn shap_explain_rows_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<(f32, Vec<Vec<f32>>), ShapError> {
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(ShapError::InvalidInput)?;
    shap_explain_rows_impl(artifact_bytes, &rows)
}

#[allow(clippy::type_complexity)]
fn shap_explain_interactions_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<(f32, Vec<Vec<Vec<f32>>>), ShapError> {
    let batch = explain_interactions_from_artifact_bytes(artifact_bytes, rows)?;
    Ok((batch.expected_value, batch.values))
}

#[allow(clippy::type_complexity)]
fn shap_explain_interactions_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<(f32, Vec<Vec<Vec<f32>>>), ShapError> {
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(ShapError::InvalidInput)?;
    shap_explain_interactions_impl(artifact_bytes, &rows)
}

fn shap_global_importance_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<(String, f32)>, ShapError> {
    global_importance_from_artifact_bytes(artifact_bytes, rows)
}

fn shap_global_importance_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<(String, f32)>, ShapError> {
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(ShapError::InvalidInput)?;
    shap_global_importance_impl(artifact_bytes, &rows)
}

/// Translate Python-side binning kwargs into a `BinningContext` for the
/// SHAP path-walker.
///
/// `binning_kind`:
///
/// - `"linear"`: `BinningContext::Linear`. Needs `feature_mins`,
///   `feature_maxs`, `max_data_bin`.
/// - `"quantile"`: `BinningContext::Quantile`. Needs `feature_cuts`.
/// - `"prebinned"`: `BinningContext::PreBinned`. No aux args.
/// - `"linear_rank"`: `BinningContext::LinearRank`. Needs
///   `feature_mins`, `feature_maxs`, `max_data_bin`, plus
///   `linear_rank_per_feature` (a per-feature list where each entry is
///   either `None` for the linear fallback or `Some(sorted_unique_values)`
///   for rank-based binning).
///
/// On invalid combinations returns a `ShapError::InvalidInput`.
#[allow(clippy::too_many_arguments)]
fn build_binning_context(
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
fn build_train_params(
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
    morph_config: Option<alloygbm_core::MorphConfig>,
    leaf_model: alloygbm_core::LeafModelKind,
    leaf_solver: alloygbm_core::LeafSolverKind,
    dro_config: Option<alloygbm_core::DroConfig>,
    neutralization_config: Option<FactorNeutralizationConfig>,
    boosting_mode: alloygbm_core::BoostingMode,
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
fn parse_boosting_mode(
    boosting_mode: &str,
    goss_top_rate: Option<f32>,
    goss_other_rate: Option<f32>,
    dart_drop_rate: Option<f32>,
    dart_max_drop: Option<usize>,
    dart_normalize_type: Option<&str>,
    dart_sample_type: Option<&str>,
) -> PyResult<alloygbm_core::BoostingMode> {
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
                "tree" => alloygbm_core::DartNormalize::Tree,
                "forest" => alloygbm_core::DartNormalize::Forest,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "dart_normalize_type must be 'tree' or 'forest', got {other:?}"
                    )));
                }
            };
            let sample_type = match dart_sample_type.unwrap_or("uniform") {
                "uniform" => alloygbm_core::DartSampleType::Uniform,
                "weighted" => alloygbm_core::DartSampleType::Weighted,
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

fn parse_neutralization_config(
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

fn validate_neutralization_leaf_model(
    neutralization_config: Option<FactorNeutralizationConfig>,
    leaf_model: alloygbm_core::LeafModelKind,
) -> PyResult<()> {
    if neutralization_config.is_some_and(|config| config.kind == NeutralizationKind::SplitPenalty)
        && leaf_model == alloygbm_core::LeafModelKind::Linear
    {
        return Err(PyValueError::new_err(
            "neutralization='split_penalty' requires leaf_model='constant'",
        ));
    }
    Ok(())
}

fn parse_factor_exposure_matrix(
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
fn parse_leaf_model(leaf_model: &str) -> pyo3::PyResult<alloygbm_core::LeafModelKind> {
    match leaf_model {
        "constant" => Ok(alloygbm_core::LeafModelKind::Constant),
        "linear" => Ok(alloygbm_core::LeafModelKind::Linear),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "leaf_model must be 'constant' or 'linear', got '{other}'"
        ))),
    }
}

fn parse_leaf_solver(leaf_solver: &str) -> pyo3::PyResult<alloygbm_core::LeafSolverKind> {
    match leaf_solver {
        "standard" => Ok(alloygbm_core::LeafSolverKind::Standard),
        "dro" => Ok(alloygbm_core::LeafSolverKind::Dro),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "leaf_solver must be 'standard' or 'dro', got '{other}'"
        ))),
    }
}

fn parse_dro_config(
    leaf_solver: alloygbm_core::LeafSolverKind,
    dro_radius: f32,
    dro_metric: &str,
) -> pyo3::PyResult<Option<alloygbm_core::DroConfig>> {
    if !dro_radius.is_finite() || dro_radius < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dro_radius must be finite and >= 0",
        ));
    }
    let metric = match dro_metric {
        "wasserstein" => alloygbm_core::DroMetric::Wasserstein,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dro_metric must be 'wasserstein', got '{other}'"
            )));
        }
    };
    Ok(match leaf_solver {
        alloygbm_core::LeafSolverKind::Standard => None,
        alloygbm_core::LeafSolverKind::Dro => Some(alloygbm_core::DroConfig {
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
fn parse_morph_config_from_pydict(
    dict: &pyo3::Bound<'_, pyo3::types::PyDict>,
) -> pyo3::PyResult<alloygbm_core::MorphConfig> {
    use alloygbm_core::{LrSchedule, MorphConfig};
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

#[pyfunction]
fn predictor_predict_batch(artifact_bytes: &[u8], rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    predictor_predict_batch_impl(artifact_bytes, &rows).map_err(predictor_error_to_pyerr)
}

#[pyfunction]
fn predictor_predict_batch_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(predictor_error_to_pyerr)
}

#[pyfunction]
fn predictor_predict_batch_canonical(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_canonical_impl(artifact_bytes, &rows).map_err(predictor_error_to_pyerr)
}

#[pyfunction]
fn predictor_predict_batch_canonical_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_canonical_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(predictor_error_to_pyerr)
}

#[pyfunction]
fn shap_explain_rows(artifact_bytes: &[u8], rows: Vec<Vec<f32>>) -> PyResult<(f32, Vec<Vec<f32>>)> {
    shap_explain_rows_impl(artifact_bytes, &rows).map_err(shap_error_to_pyerr)
}

#[pyfunction]
fn shap_explain_rows_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(f32, Vec<Vec<f32>>)> {
    shap_explain_rows_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
fn shap_explain_interactions(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    shap_explain_interactions_impl(artifact_bytes, &rows).map_err(shap_error_to_pyerr)
}

#[pyfunction]
fn shap_explain_interactions_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    shap_explain_interactions_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, rows, binning_kind, feature_mins=None, feature_maxs=None,
    max_data_bin=None, feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_explain_interactions_with_binning(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let batch = explain_interactions_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)?;
    Ok((batch.expected_value, batch.values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, values, row_count, feature_count, binning_kind,
    feature_mins=None, feature_maxs=None, max_data_bin=None,
    feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_explain_interactions_dense_with_binning(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    let rows = dense_rows_from_flat_values(&values, row_count, feature_count)
        .map_err(|msg| shap_error_to_pyerr(ShapError::InvalidInput(msg)))?;
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let batch = explain_interactions_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)?;
    Ok((batch.expected_value, batch.values))
}

#[pyfunction]
fn shap_global_importance(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<Vec<(String, f32)>> {
    shap_global_importance_impl(artifact_bytes, &rows).map_err(shap_error_to_pyerr)
}

#[pyfunction]
fn shap_global_importance_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<(String, f32)>> {
    shap_global_importance_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(shap_error_to_pyerr)
}

// Predictor-aligned SHAP entry points (v0.7.3).  When the caller passes
// the binning info already used by the prediction-time threshold
// conversion (`feature_mins` / `feature_maxs` / `max_data_bin` for
// linear binning, `feature_cuts` for quantile, neither for pre-binned),
// SHAP walks paths the same way the predictor does — strict `<`
// comparison against per-stump float thresholds.  This makes SHAP on
// `leaf_model="linear"` artifacts trained over continuous features
// strictly additive, lifting the legacy best-effort exemption.

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes,
    rows,
    binning_kind,
    feature_mins=None,
    feature_maxs=None,
    max_data_bin=None,
    feature_cuts=None,
    linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_explain_rows_with_binning(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(f32, Vec<Vec<f32>>)> {
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let explanation = explain_rows_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)?;
    Ok((explanation.expected_value, explanation.values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes,
    values,
    row_count,
    feature_count,
    binning_kind,
    feature_mins=None,
    feature_maxs=None,
    max_data_bin=None,
    feature_cuts=None,
    linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_explain_rows_dense_with_binning(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(f32, Vec<Vec<f32>>)> {
    let rows = dense_rows_from_flat_values(&values, row_count, feature_count)
        .map_err(|msg| shap_error_to_pyerr(ShapError::InvalidInput(msg)))?;
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let explanation = explain_rows_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)?;
    Ok((explanation.expected_value, explanation.values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes,
    rows,
    binning_kind,
    feature_mins=None,
    feature_maxs=None,
    max_data_bin=None,
    feature_cuts=None,
    linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_global_importance_with_binning(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<Vec<(String, f32)>> {
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    global_importance_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes,
    values,
    row_count,
    feature_count,
    binning_kind,
    feature_mins=None,
    feature_maxs=None,
    max_data_bin=None,
    feature_cuts=None,
    linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
fn shap_global_importance_dense_with_binning(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<Vec<(String, f32)>> {
    let rows = dense_rows_from_flat_values(&values, row_count, feature_count)
        .map_err(|msg| shap_error_to_pyerr(ShapError::InvalidInput(msg)))?;
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    global_importance_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        .map_err(shap_error_to_pyerr)
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
    quantile_alpha=None
))]
#[allow(clippy::too_many_arguments)]
fn train_regression_artifact(
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
    quantile_alpha: Option<f32>,
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
        objective,
        None, // init_artifact_bytes
        None, // num_classes
        None, // custom_objective_fn
        None, // custom_loss_fn
        None, // custom_metric_fn
        0,    // max_cat_threshold (disabled for non-summary paths)
    )
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
    quantile_alpha=None
))]
#[allow(clippy::too_many_arguments)]
fn train_regression_artifact_dense(
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
    quantile_alpha: Option<f32>,
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
        objective,
        None, // init_artifact_bytes
        None, // num_classes
        None, // custom_objective_fn
        None, // custom_loss_fn
        None, // custom_metric_fn
        0,    // max_cat_threshold (disabled for non-summary paths)
    )
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
    quantile_alpha=None
))]
#[allow(clippy::too_many_arguments)]
fn train_regression_artifact_with_summary(
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
    quantile_alpha: Option<f32>,
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
        objective,
        init_artifact_bytes.as_deref(),
        num_classes,
        custom_objective_fn,
        custom_loss_fn,
        custom_metric_fn,
        max_cat_threshold,
    )
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
    quantile_alpha=None
))]
#[allow(clippy::too_many_arguments)]
fn train_regression_artifact_dense_with_summary(
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
    quantile_alpha: Option<f32>,
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
        objective,
        init_artifact_bytes.as_deref(),
        num_classes,
        custom_objective_fn,
        custom_loss_fn,
        custom_metric_fn,
        max_cat_threshold,
    )
    .map_err(engine_error_to_pyerr)
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
    quantile_alpha=None
))]
#[allow(clippy::too_many_arguments)]
fn train_regression_artifact_dense_with_summary_bytes(
    values_bytes: &[u8],
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
    validation_values_bytes: Option<&[u8]>,
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
    quantile_alpha: Option<f32>,
) -> PyResult<NativeTrainingResult> {
    let values = bytes_to_f32_vec(values_bytes)?;
    let targets = bytes_to_f32_vec(targets_bytes)?;
    let validation_values = validation_values_bytes.map(bytes_to_f32_vec).transpose()?;
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
        objective,
        init_artifact_bytes.as_deref(),
        num_classes,
        custom_objective_fn,
        custom_loss_fn,
        custom_metric_fn,
        max_cat_threshold,
    )
    .map_err(engine_error_to_pyerr)
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
    use super::{
        ContinuousBinningStrategy, DEFAULT_TRAIN_ROUNDS, MAX_CONTINUOUS_QUANTIZED_BIN_U8,
        flatten_rows, predictor_predict_batch_canonical_impl, predictor_predict_batch_impl,
        shap_explain_rows_impl, shap_global_importance_impl,
        train_regression_artifact_with_summary_dense_impl,
    };
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

        let missing_values = super::resolve_categorical_spec(Some(1), None, 20.0, 1, false, 8);
        assert!(matches!(
            missing_values,
            Err(alloygbm_engine::EngineError::ContractViolation(_))
        ));

        let missing_index = super::resolve_categorical_spec(
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
