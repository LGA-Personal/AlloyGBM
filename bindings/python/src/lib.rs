#![allow(clippy::too_many_arguments)]

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_categorical::{
    TargetEncoderConfig, fit_target_encoder, fit_transform_target_encoder, transform_target_encoder,
};
use alloygbm_core::{
    BinnedMatrix, CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix,
    DenseMatrixView, GradientPair, MISSING_BIN_U8, TrainParams, TrainingDataset, TreeGrowth,
};
use alloygbm_engine::{
    ArtifactCompatibilityMode, BinaryCrossEntropyObjective, CategoricalFeatureInfo,
    CategoricalTargetEncodingSpec, EngineError, IterationRunSummary, LambdaMARTObjective,
    MultiClassIterationRunSummary, MultiClassSoftmaxObjective, MultiClassTrainedModel,
    MultiClassWarmStartState, ObjectiveOps, PairwiseRankingObjective, PerRoundMetricCallback,
    QueryRMSEObjective, SquaredErrorObjective, TrainedModel, Trainer, TrainingPolicyMode,
    WarmStartState, XeNDCGObjective, YetiRankObjective,
};
use alloygbm_predictor::{Predictor, PredictorError};
use alloygbm_shap::{
    ShapError, explain_rows_from_artifact_bytes, global_importance_from_artifact_bytes,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_TRAIN_ROUNDS: usize = 6;
const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
const MAX_CONTINUOUS_QUANTIZED_BIN_U8: u16 = 254;
const MAX_CONTINUOUS_QUANTIZED_BIN_U16: u16 = 65534;
const MIN_CONTINUOUS_QUANTIZED_BINS: usize = 2;
const LINEAR_TAIL_RANK_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK";
const LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO";
const DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD: f32 = 0.10;

// ---------------------------------------------------------------------------
// Shared helpers for fast Rust ↔ numpy data transfer
// ---------------------------------------------------------------------------

/// Extract a `Vec<f32>` from a Python object that is expected to be a numpy
/// array.  Tries the zero-overhead path first (`PyReadonlyArray1<f32>` →
/// `as_slice()` → single `memcpy`) and falls back to `PyReadonlyArray1<f64>`
/// with a narrowing copy, then to generic PyO3 extraction (element-wise,
/// slowest) for non-contiguous or exotic dtypes.
fn extract_f32_array(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<Vec<f32>, EngineError> {
    // Fast path: contiguous f32 numpy array — single memcpy via as_slice()
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        if let Ok(slice) = readonly.as_slice() {
            return Ok(slice.to_vec());
        }
        // Non-contiguous — fall through to generic path
    }
    // Medium path: contiguous f64 numpy array — common when user returns
    // plain Python floats which numpy defaults to float64
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        if let Ok(slice) = readonly.as_slice() {
            return Ok(slice.iter().map(|&v| v as f32).collect());
        }
        // Non-contiguous — fall through to generic path
    }
    // Slow fallback: generic extraction (handles lists, non-contiguous arrays, etc.)
    obj.extract::<Vec<f32>>().map_err(|e| {
        EngineError::ContractViolation(format!("failed to extract array as Vec<f32>: {e}"))
    })
}

/// Create a pair of read-only numpy f32 arrays from Rust slices.
/// Uses `PyArray1::from_slice` which does a single memcpy — much faster than
/// going through `Vec → Python list → np.array() → .astype("float32")`.
fn make_numpy_f32_pair<'py>(
    py: Python<'py>,
    a: &[f32],
    b: &[f32],
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
    (PyArray1::from_slice(py, a), PyArray1::from_slice(py, b))
}

// ---------------------------------------------------------------------------
// Custom Python Objective — wraps a Python callable implementing ObjectiveOps
// ---------------------------------------------------------------------------

/// A custom objective backed by a Python callable.
///
/// The callable must have the signature:
///   `(y_true: np.ndarray, y_pred: np.ndarray) -> (grad: np.ndarray, hess: np.ndarray)`
///
/// An optional loss function can be provided with signature:
///   `(y_true: np.ndarray, y_pred: np.ndarray) -> float`
struct CustomPythonObjective {
    grad_hess_fn: Py<PyAny>,
    loss_fn: Option<Py<PyAny>>,
}

impl CustomPythonObjective {
    fn new(grad_hess_fn: Py<PyAny>, loss_fn: Option<Py<PyAny>>) -> Self {
        Self {
            grad_hess_fn,
            loss_fn,
        }
    }

    fn call_python_grad_hess(
        &self,
        predictions: &[f32],
        targets: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        Python::with_gil(|py| {
            // Create numpy arrays directly from Rust slices — avoids creating a
            // Python list and the redundant .astype("float32") call.  PyArray1::from_slice
            // produces a contiguous f32 numpy array backed by a copy of the slice
            // (one memcpy instead of element-by-element Python object creation).
            let y_true = PyArray1::from_slice(py, targets);
            let y_pred = PyArray1::from_slice(py, predictions);

            let result = self.grad_hess_fn.call1(py, (y_true, y_pred)).map_err(|e| {
                EngineError::ContractViolation(format!("custom objective callable failed: {e}"))
            })?;

            let tuple = result
                .downcast_bound::<pyo3::types::PyTuple>(py)
                .map_err(|_| {
                    EngineError::ContractViolation(
                        "custom objective must return a tuple of (gradient, hessian)".to_string(),
                    )
                })?;
            if tuple.len() != 2 {
                return Err(EngineError::ContractViolation(format!(
                    "custom objective must return exactly 2 arrays, got {}",
                    tuple.len()
                )));
            }

            let grad_arr = tuple.get_item(0).map_err(|e| {
                EngineError::ContractViolation(format!("failed to get gradient array: {e}"))
            })?;
            let hess_arr = tuple.get_item(1).map_err(|e| {
                EngineError::ContractViolation(format!("failed to get hessian array: {e}"))
            })?;

            // Fast extraction: try to get a typed numpy array view (single memcpy
            // from contiguous buffer) before falling back to generic element-wise
            // extraction for non-contiguous or non-f32 arrays.
            let grad_list = extract_f32_array(py, &grad_arr)?;
            let hess_list = extract_f32_array(py, &hess_arr)?;

            if grad_list.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "gradient array length {} does not match target length {}",
                    grad_list.len(),
                    targets.len()
                )));
            }
            if hess_list.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "hessian array length {} does not match target length {}",
                    hess_list.len(),
                    targets.len()
                )));
            }

            Ok((grad_list, hess_list))
        })
    }

    fn call_python_loss(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, EngineError> {
        if let Some(ref loss_fn) = self.loss_fn {
            Python::with_gil(|py| {
                let (y_true, y_pred) = make_numpy_f32_pair(py, targets, predictions);
                let result = loss_fn.call1(py, (y_true, y_pred)).map_err(|e| {
                    EngineError::ContractViolation(format!("custom loss callable failed: {e}"))
                })?;
                let value: f64 = result.extract(py).map_err(|e| {
                    EngineError::ContractViolation(format!(
                        "custom loss must return a float, got: {e}"
                    ))
                })?;
                Ok(value as f32)
            })
        } else {
            // Fallback: use MSE as a loss proxy for monitoring/early stopping
            let n = predictions.len();
            if n == 0 {
                return Ok(0.0);
            }
            let mse: f64 = predictions
                .iter()
                .zip(targets.iter())
                .map(|(&p, &t)| {
                    let diff = (p - t) as f64;
                    diff * diff
                })
                .sum::<f64>()
                / n as f64;
            Ok(mse as f32)
        }
    }
}

impl ObjectiveOps for CustomPythonObjective {
    fn objective_name(&self) -> &str {
        "custom"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> alloygbm_engine::EngineResult<f32> {
        // Weighted mean of targets (same as squared error)
        if targets.is_empty() {
            return Ok(0.0);
        }
        if let Some(weights) = sample_weights {
            let total_weight: f64 = weights.iter().map(|&w| w as f64).sum();
            if total_weight <= 0.0 {
                return Ok(0.0);
            }
            let weighted_sum: f64 = targets
                .iter()
                .zip(weights.iter())
                .map(|(&t, &w)| t as f64 * w as f64)
                .sum();
            Ok((weighted_sum / total_weight) as f32)
        } else {
            let sum: f64 = targets.iter().map(|&t| t as f64).sum();
            Ok((sum / targets.len() as f64) as f32)
        }
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> alloygbm_engine::EngineResult<Vec<GradientPair>> {
        let (grads, hessians) = self.call_python_grad_hess(predictions, targets)?;
        let n = grads.len();
        let mut result = Vec::with_capacity(n);
        if let Some(weights) = sample_weights {
            for i in 0..n {
                let w = weights[i];
                result.push(GradientPair {
                    grad: grads[i] * w,
                    hess: hessians[i] * w,
                });
            }
        } else {
            for i in 0..n {
                result.push(GradientPair {
                    grad: grads[i],
                    hess: hessians[i],
                });
            }
        }
        Ok(result)
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> alloygbm_engine::EngineResult<f32> {
        self.call_python_loss(predictions, targets)
    }

    fn requires_group_id(&self) -> bool {
        false
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Custom Python Metric Callback
// ---------------------------------------------------------------------------

/// A custom evaluation metric backed by a Python callable.
///
/// The callable must have the signature:
///   `(y_true: np.ndarray, y_pred: np.ndarray) -> (name: str, value: float, higher_is_better: bool)`
struct CustomPythonMetricCallback {
    metric_fn: Py<PyAny>,
    /// Name discovered during probe, set before training starts
    name: String,
    /// Direction discovered during probe, set before training starts
    higher_is_better_flag: bool,
}

impl CustomPythonMetricCallback {
    /// Create a new callback by probing the Python callable with dummy data
    /// to discover the metric name and direction.
    fn new(metric_fn: Py<PyAny>) -> Result<Self, EngineError> {
        let dummy_targets = vec![0.0_f32; 2];
        let dummy_preds = vec![0.0_f32; 2];

        let (name, _value, higher_is_better) =
            Self::call_python_static(&metric_fn, &dummy_preds, &dummy_targets)?;

        Ok(Self {
            metric_fn,
            name,
            higher_is_better_flag: higher_is_better,
        })
    }

    /// Call the Python metric function and extract (name, value, higher_is_better).
    fn call_python_static(
        metric_fn: &Py<PyAny>,
        predictions: &[f32],
        targets: &[f32],
    ) -> Result<(String, f32, bool), EngineError> {
        Python::with_gil(|py| {
            let (y_true, y_pred) = make_numpy_f32_pair(py, targets, predictions);

            let result = metric_fn.call1(py, (y_true, y_pred)).map_err(|e| {
                EngineError::ContractViolation(format!("custom eval metric callable failed: {e}"))
            })?;

            let tuple = result
                .downcast_bound::<pyo3::types::PyTuple>(py)
                .map_err(|_| {
                    EngineError::ContractViolation(
                        "custom eval metric must return a tuple of (name, value, higher_is_better)"
                            .to_string(),
                    )
                })?;
            if tuple.len() != 3 {
                return Err(EngineError::ContractViolation(format!(
                    "custom eval metric must return exactly 3 values (name, value, higher_is_better), got {}",
                    tuple.len()
                )));
            }

            let name: String = tuple
                .get_item(0)
                .map_err(|e| {
                    EngineError::ContractViolation(format!("failed to get metric name: {e}"))
                })?
                .extract()
                .map_err(|e| {
                    EngineError::ContractViolation(format!("metric name must be a string: {e}"))
                })?;
            let value: f64 = tuple
                .get_item(1)
                .map_err(|e| {
                    EngineError::ContractViolation(format!("failed to get metric value: {e}"))
                })?
                .extract()
                .map_err(|e| {
                    EngineError::ContractViolation(format!("metric value must be a float: {e}"))
                })?;
            let higher_is_better: bool = tuple
                .get_item(2)
                .map_err(|e| {
                    EngineError::ContractViolation(format!("failed to get higher_is_better: {e}"))
                })?
                .extract()
                .map_err(|e| {
                    EngineError::ContractViolation(format!("higher_is_better must be a bool: {e}"))
                })?;

            Ok((name, value as f32, higher_is_better))
        })
    }
}

impl PerRoundMetricCallback for CustomPythonMetricCallback {
    fn evaluate(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> alloygbm_engine::EngineResult<f32> {
        let (_name, value, _higher_is_better) =
            Self::call_python_static(&self.metric_fn, predictions, targets)?;
        Ok(value)
    }

    fn higher_is_better(&self) -> bool {
        self.higher_is_better_flag
    }

    fn metric_name(&self) -> &str {
        &self.name
    }
}

fn is_pre_binned_integer_value(value: f32) -> bool {
    if value < 0.0 {
        return false;
    }
    let rounded = value.round();
    (value - rounded).abs() <= PRE_BINNED_INTEGER_TOLERANCE
}

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
struct NativeRuntimeInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: String,
}

#[pymethods]
impl NativeRuntimeInfo {
    #[new]
    fn new() -> Self {
        Self {
            name: "alloygbm".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[pyfunction]
fn native_runtime_info() -> NativeRuntimeInfo {
    NativeRuntimeInfo::new()
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
                        out_chunk[out_base + fi] =
                            quantize_linear_value(value, feature_mins[fi], feature_maxs[fi]) as f32;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuousBinningStrategy {
    Linear,
    Rank,
    Quantile,
}

#[derive(Debug, Clone)]
struct ContinuousBinningMetadataInternal {
    uses_continuous_binning: bool,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    feature_sorted_values: Option<Vec<Vec<f32>>>,
    feature_quantile_cuts: Option<Vec<Vec<f32>>>,
    feature_linear_rank_flags: Option<Vec<bool>>,
}

impl ContinuousBinningMetadataInternal {
    fn pre_binned() -> Self {
        Self {
            uses_continuous_binning: false,
            feature_mins: None,
            feature_maxs: None,
            feature_sorted_values: None,
            feature_quantile_cuts: None,
            feature_linear_rank_flags: None,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct NativeContinuousBinningMetadata {
    #[pyo3(get)]
    uses_continuous_binning: bool,
    #[pyo3(get)]
    feature_mins: Option<Vec<f32>>,
    #[pyo3(get)]
    feature_maxs: Option<Vec<f32>>,
    #[pyo3(get)]
    feature_sorted_values: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    feature_quantile_cuts: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    feature_linear_rank_flags: Option<Vec<bool>>,
}

impl From<ContinuousBinningMetadataInternal> for NativeContinuousBinningMetadata {
    fn from(value: ContinuousBinningMetadataInternal) -> Self {
        Self {
            uses_continuous_binning: value.uses_continuous_binning,
            feature_mins: value.feature_mins,
            feature_maxs: value.feature_maxs,
            feature_sorted_values: value.feature_sorted_values,
            feature_quantile_cuts: value.feature_quantile_cuts,
            feature_linear_rank_flags: value.feature_linear_rank_flags,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct NativeTrainingSummary {
    #[pyo3(get)]
    rounds_requested: usize,
    #[pyo3(get)]
    rounds_completed: usize,
    #[pyo3(get)]
    best_validation_round: Option<usize>,
    #[pyo3(get)]
    best_validation_loss: Option<f32>,
    #[pyo3(get)]
    train_rmse: Vec<f32>,
    #[pyo3(get)]
    validation_rmse: Vec<f32>,
    /// Raw objective loss per completed round (no sqrt transform).
    #[pyo3(get)]
    train_loss: Vec<f32>,
    /// Raw validation objective loss per completed round (no sqrt transform).
    #[pyo3(get)]
    validation_loss: Vec<f32>,
    /// Objective name (e.g. "squared_error", "binary_crossentropy").
    #[pyo3(get)]
    objective: String,
    #[pyo3(get)]
    stop_reason: String,
    #[pyo3(get)]
    bridge_prepare_seconds: f64,
    #[pyo3(get)]
    native_train_seconds: f64,
    /// Custom eval metric values per completed round (empty when no custom metric).
    #[pyo3(get)]
    custom_metric_values: Vec<f32>,
    /// Custom eval metric name (None when no custom metric).
    #[pyo3(get)]
    custom_metric_name: Option<String>,
}

#[pyclass]
#[derive(Debug, Clone)]
struct NativeTrainingResult {
    #[pyo3(get)]
    artifact_bytes: Vec<u8>,
    #[pyo3(get)]
    summary: NativeTrainingSummary,
    #[pyo3(get)]
    continuous_binning_metadata: NativeContinuousBinningMetadata,
    /// Per-feature category→ID mappings for native categorical splits.
    /// Keys are feature indices, values are dicts {category_name: integer_id}.
    #[pyo3(get)]
    native_cat_mappings: std::collections::HashMap<usize, std::collections::HashMap<String, u32>>,
}

#[derive(Debug, Clone)]
struct PreparedTrainingMatrices {
    dataset: TrainingDataset,
    binned_matrix: BinnedMatrix,
    metadata: ContinuousBinningMetadataInternal,
}

fn parse_continuous_binning_strategy(
    value: &str,
) -> Result<ContinuousBinningStrategy, EngineError> {
    match value {
        "linear" => Ok(ContinuousBinningStrategy::Linear),
        "rank" => Ok(ContinuousBinningStrategy::Rank),
        "quantile" => Ok(ContinuousBinningStrategy::Quantile),
        other => Err(EngineError::InvalidConfig(format!(
            "continuous_binning_strategy must be one of: linear, quantile, rank; received '{other}'"
        ))),
    }
}

fn validate_continuous_binning_max_bins(max_bins: usize) -> Result<(), EngineError> {
    if !(MIN_CONTINUOUS_QUANTIZED_BINS..=(MAX_CONTINUOUS_QUANTIZED_BIN_U16 as usize + 1))
        .contains(&max_bins)
    {
        return Err(EngineError::InvalidConfig(format!(
            "continuous_binning_max_bins must be in [{MIN_CONTINUOUS_QUANTIZED_BINS}, {}]",
            MAX_CONTINUOUS_QUANTIZED_BIN_U16 as usize + 1
        )));
    }
    Ok(())
}

/// Whether the given max_bins requires u16 bin storage.
fn needs_wide_bins(max_bins: usize) -> bool {
    max_bins > (MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1)
}

fn env_toggle_enabled(env_name: &str) -> bool {
    match std::env::var(env_name) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn linear_tail_rank_enabled_from_env() -> bool {
    env_toggle_enabled(LINEAR_TAIL_RANK_ENV_VAR)
}

fn linear_tail_core_span_ratio_threshold_from_env() -> f32 {
    match std::env::var(LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR) {
        Ok(value) => value
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|parsed| parsed.is_finite())
            .map(|parsed| parsed.clamp(0.0, 1.0))
            .unwrap_or(DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD),
        Err(_) => DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD,
    }
}

fn round_half_away_from_zero(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5).floor() as i32
    } else {
        (value - 0.5).ceil() as i32
    }
}

fn quantize_linear_value(value: f32, min_value: f32, max_value: f32) -> u8 {
    quantize_linear_value_wide(value, min_value, max_value, MAX_CONTINUOUS_QUANTIZED_BIN_U8) as u8
}

fn quantize_rank_value(value: f32, sorted_values: &[f32]) -> u8 {
    quantize_rank_value_wide(value, sorted_values, MAX_CONTINUOUS_QUANTIZED_BIN_U8) as u8
}

/// Parameterized linear quantization that supports arbitrary max_data_bin (u16).
fn quantize_linear_value_wide(
    value: f32,
    min_value: f32,
    max_value: f32,
    max_data_bin: u16,
) -> u16 {
    if value <= min_value {
        return 0;
    }
    if value >= max_value {
        return max_data_bin;
    }
    let span = max_value - min_value;
    if span <= PRE_BINNED_INTEGER_TOLERANCE {
        return 0;
    }
    let scaled = ((value - min_value) / span) * max_data_bin as f32;
    round_half_away_from_zero(scaled).clamp(0, max_data_bin as i32) as u16
}

/// Parameterized rank quantization that supports arbitrary max_data_bin (u16).
fn quantize_rank_value_wide(value: f32, sorted_values: &[f32], max_data_bin: u16) -> u16 {
    if sorted_values.len() <= 1 {
        return 0;
    }
    let insertion = sorted_values.partition_point(|probe| *probe <= value);
    let rank = insertion.saturating_sub(1).min(sorted_values.len() - 1);
    let scaled =
        (rank as f32 * max_data_bin as f32) / (sorted_values.len().saturating_sub(1) as f32);
    round_half_away_from_zero(scaled).clamp(0, max_data_bin as i32) as u16
}

/// Parameterized linear quantize for predict-time: supports arbitrary max_data_bin.
fn quantize_dense_values_linear_inplace_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    feature_mins: &[f32],
    feature_maxs: &[f32],
    max_data_bin: u16,
) -> Vec<f32> {
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
                let base = row_index * feature_count;
                let out_base = local_row * feature_count;
                for fi in 0..feature_count {
                    let value = values[base + fi];
                    out_chunk[out_base + fi] = quantize_linear_value_wide(
                        value,
                        feature_mins[fi],
                        feature_maxs[fi],
                        max_data_bin,
                    ) as f32;
                }
            }
        });
    quantized
}

/// Parameterized linear+rank quantize for predict-time: supports arbitrary max_data_bin.
fn quantize_dense_values_linear_rank_inplace_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    feature_mins: &[f32],
    feature_maxs: &[f32],
    rank_flags: &[bool],
    feature_sorted_values: &[Vec<f32>],
    max_data_bin: u16,
) -> Vec<f32> {
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
                let base = row_index * feature_count;
                let out_base = local_row * feature_count;
                for fi in 0..feature_count {
                    let value = values[base + fi];
                    let bin = if rank_flags[fi] {
                        quantize_rank_value_wide(value, &feature_sorted_values[fi], max_data_bin)
                    } else {
                        quantize_linear_value_wide(
                            value,
                            feature_mins[fi],
                            feature_maxs[fi],
                            max_data_bin,
                        )
                    };
                    out_chunk[out_base + fi] = bin as f32;
                }
            }
        });
    quantized
}

fn derive_dense_feature_bounds(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> (Vec<f32>, Vec<f32>) {
    let results: Vec<(f32, f32)> = (0..feature_count)
        .into_par_iter()
        .map(|feature_index| {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for row_index in 0..row_count {
                let value = values[row_index * feature_count + feature_index];
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
            (min_val, max_val)
        })
        .collect();
    let mut mins = Vec::with_capacity(feature_count);
    let mut maxs = Vec::with_capacity(feature_count);
    for (min_val, max_val) in results {
        mins.push(min_val);
        maxs.push(max_val);
    }
    (mins, maxs)
}

fn derive_dense_sorted_feature_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Vec<Vec<f32>> {
    (0..feature_count)
        .into_par_iter()
        .map(|feature_index| {
            let mut column = Vec::with_capacity(row_count);
            for row_index in 0..row_count {
                let value = values[row_index * feature_count + feature_index];
                if !value.is_nan() {
                    column.push(value);
                }
            }
            column.sort_by(f32::total_cmp);
            column
        })
        .collect()
}

fn derive_dense_feature_quantile_cuts(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    max_bins: usize,
) -> Vec<Vec<f32>> {
    let sorted_values = derive_dense_sorted_feature_values(values, row_count, feature_count);
    sorted_values
        .into_par_iter()
        .map(|column| {
            if column.len() <= 1 {
                return Vec::new();
            }
            let bin_count = max_bins.min(column.len());
            let mut cuts = Vec::new();
            for quantile_index in 1..bin_count {
                let mut rank = (quantile_index * column.len()) / bin_count;
                if rank >= column.len() {
                    rank = column.len() - 1;
                }
                let cut_value = column[rank];
                if cuts.last().copied().is_some_and(|last| cut_value <= last) {
                    continue;
                }
                cuts.push(cut_value);
            }
            cuts
        })
        .collect()
}

fn derive_linear_tail_rank_plan(
    sorted_values: &[Vec<f32>],
    core_span_ratio_threshold: f32,
) -> Vec<bool> {
    sorted_values
        .par_iter()
        .map(|values| {
            let value_count = values.len();
            if value_count < 5 {
                return false;
            }
            let full_span = values[value_count - 1] - values[0];
            if full_span <= PRE_BINNED_INTEGER_TOLERANCE {
                return false;
            }
            let trim_count = ((value_count as f32 * 0.1).floor() as usize).max(1);
            if trim_count * 2 >= value_count {
                return false;
            }
            let core_low = values[trim_count];
            let core_high = values[value_count - 1 - trim_count];
            let core_span = core_high - core_low;
            let ratio = core_span / full_span;
            ratio.is_finite() && ratio <= core_span_ratio_threshold
        })
        .collect()
}

fn validate_dense_values_allow_nan(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Result<(), EngineError> {
    let dense_view = DenseMatrixView::new(row_count, feature_count, values)?;
    for row_index in 0..dense_view.row_count {
        let row = dense_view.row(row_index)?;
        for (feature_index, &value) in row.iter().enumerate() {
            if value.is_infinite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must not be infinite"
                )));
            }
        }
    }
    Ok(())
}

fn prepare_validation_matrices_from_dense_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    time_index: Option<Vec<i64>>,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    strategy: ContinuousBinningStrategy,
    training_metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
    max_bins: usize,
) -> Result<PreparedTrainingMatrices, EngineError> {
    if targets.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "rows length {} does not match targets length {}",
            row_count,
            targets.len()
        )));
    }
    validate_dense_values_allow_nan(values, row_count, feature_count)?;

    // When training took the pre-binned path (all-integer features, feature_mins = None),
    // validation must also use the pre-binned path.  Non-integer values (e.g. 1.5, 2.5)
    // are rounded to the nearest integer bin; NaN maps to the missing-bin sentinel.
    if !training_metadata.uses_continuous_binning {
        let use_wide = needs_wide_bins(max_bins);
        if use_wide {
            let max_data_bin = (max_bins - 2) as u16;
            let nan_bin = max_data_bin + 1;
            let mut bins_u16 = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin_seen = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let bin = if value.is_nan() {
                    nan_bin
                } else {
                    let rounded = value.round();
                    if rounded > 65535.0 {
                        return Err(EngineError::ContractViolation(format!(
                            "validation value at index {index} exceeds max supported bin 65535"
                        )));
                    }
                    rounded as u16
                };
                max_bin_seen = max_bin_seen.max(bin);
                bins_u16.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            let dataset = TrainingDataset {
                matrix: if need_dense_values {
                    DatasetMatrix::new(row_count, feature_count, dense_values_out)?
                } else {
                    DatasetMatrix::new_metadata_only(row_count, feature_count)?
                },
                targets: targets.to_vec(),
                sample_weights,
                time_index,
                group_id,
            };
            let binned_matrix = BinnedMatrix::new_u16(
                row_count,
                feature_count,
                if max_bin_seen == 0 { 1 } else { max_bin_seen },
                nan_bin,
                bins_u16,
            )?;
            return Ok(PreparedTrainingMatrices {
                dataset,
                binned_matrix,
                metadata: training_metadata.clone(),
            });
        } else {
            let mut bins = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin_seen = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let bin = if value.is_nan() {
                    MISSING_BIN_U8
                } else {
                    let rounded = value.round();
                    if rounded > 255.0 {
                        return Err(EngineError::ContractViolation(format!(
                            "validation value at index {index} exceeds max supported bin 255"
                        )));
                    }
                    rounded as u8
                };
                max_bin_seen = max_bin_seen.max(u16::from(bin));
                bins.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            let dataset = TrainingDataset {
                matrix: if need_dense_values {
                    DatasetMatrix::new(row_count, feature_count, dense_values_out)?
                } else {
                    DatasetMatrix::new_metadata_only(row_count, feature_count)?
                },
                targets: targets.to_vec(),
                sample_weights,
                time_index,
                group_id,
            };
            let binned_matrix = BinnedMatrix::new(
                row_count,
                feature_count,
                if max_bin_seen == 0 { 1 } else { max_bin_seen },
                bins,
            )?;
            return Ok(PreparedTrainingMatrices {
                dataset,
                binned_matrix,
                metadata: training_metadata.clone(),
            });
        }
    }

    if needs_wide_bins(max_bins) {
        let max_data_bin = (max_bins - 2) as u16;
        let nan_bin = max_data_bin + 1;
        let (dense_values, bins_u16, max_bin) = quantize_dense_values_with_metadata_wide(
            values,
            row_count,
            feature_count,
            strategy,
            training_metadata,
            need_dense_values,
            max_data_bin,
        )?;
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
        };
        let binned_matrix = BinnedMatrix::new_u16(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            nan_bin,
            bins_u16,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata: training_metadata.clone(),
        })
    } else {
        let (dense_values, bins, max_bin) = quantize_dense_values_with_metadata(
            values,
            row_count,
            feature_count,
            strategy,
            training_metadata,
            need_dense_values,
        )?;
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
        };
        let binned_matrix = BinnedMatrix::new(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            bins,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata: training_metadata.clone(),
        })
    }
}

fn quantize_dense_values_with_metadata(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
) -> Result<(Vec<f32>, Vec<u8>, u16), EngineError> {
    // Validate metadata upfront so parallel closures don't need to return Result.
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();

    let total_cells = row_count * feature_count;
    let mut dense_values = if need_dense_values {
        vec![0.0_f32; total_cells]
    } else {
        Vec::new()
    };
    let mut bins = vec![0_u8; total_cells];

    let chunk_size = (row_count / rayon::current_num_threads().max(1)).max(256);

    let max_bin = if need_dense_values {
        dense_values
            .par_chunks_mut(chunk_size * feature_count)
            .zip(bins.par_chunks_mut(chunk_size * feature_count))
            .enumerate()
            .map(|(chunk_idx, (dense_chunk, bin_chunk))| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = dense_chunk.len() / feature_count;
                let mut local_max_bin = 0_u8;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            MISSING_BIN_U8
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value(value, &sv[feature_index])
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value(value, &sv[feature_index])
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    cuts[feature_index].partition_point(|probe| *probe <= value)
                                        as u8
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        dense_chunk[dst_base + feature_index] = bin as f32;
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                u16::from(local_max_bin)
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    } else {
        bins.par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .map(|(chunk_idx, bin_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = bin_chunk.len() / feature_count;
                let mut local_max_bin = 0_u8;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            MISSING_BIN_U8
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value(value, &sv[feature_index])
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value(value, &sv[feature_index])
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    cuts[feature_index].partition_point(|probe| *probe <= value)
                                        as u8
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                u16::from(local_max_bin)
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    };

    Ok((dense_values, bins, max_bin))
}

/// u16 variant of `quantize_dense_values_with_metadata` for max_bins > 256.
/// Data bins scale to 0..max_data_bin; NaN gets max_data_bin + 1.
fn quantize_dense_values_with_metadata_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
    max_data_bin: u16,
) -> Result<(Vec<f32>, Vec<u16>, u16), EngineError> {
    let nan_bin = max_data_bin + 1;
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();

    let total_cells = row_count * feature_count;
    let mut dense_values = if need_dense_values {
        vec![0.0_f32; total_cells]
    } else {
        Vec::new()
    };
    let mut bins = vec![0_u16; total_cells];

    let chunk_size = (row_count / rayon::current_num_threads().max(1)).max(256);

    let max_bin = if need_dense_values {
        dense_values
            .par_chunks_mut(chunk_size * feature_count)
            .zip(bins.par_chunks_mut(chunk_size * feature_count))
            .enumerate()
            .map(|(chunk_idx, (dense_chunk, bin_chunk))| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = dense_chunk.len() / feature_count;
                let mut local_max_bin = 0_u16;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            nan_bin
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value_wide(
                                            value,
                                            &sv[feature_index],
                                            max_data_bin,
                                        )
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value_wide(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                            max_data_bin,
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value_wide(
                                        value,
                                        &sv[feature_index],
                                        max_data_bin,
                                    )
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    cuts[feature_index].partition_point(|probe| *probe <= value)
                                        as u16
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        dense_chunk[dst_base + feature_index] = bin as f32;
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                local_max_bin
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    } else {
        bins.par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .map(|(chunk_idx, bin_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = bin_chunk.len() / feature_count;
                let mut local_max_bin = 0_u16;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            nan_bin
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value_wide(
                                            value,
                                            &sv[feature_index],
                                            max_data_bin,
                                        )
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value_wide(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                            max_data_bin,
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value_wide(
                                        value,
                                        &sv[feature_index],
                                        max_data_bin,
                                    )
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    cuts[feature_index].partition_point(|probe| *probe <= value)
                                        as u16
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                local_max_bin
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    };

    Ok((dense_values, bins, max_bin))
}

fn prepare_training_matrices_from_dense_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    time_index: Option<Vec<i64>>,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    strategy: ContinuousBinningStrategy,
    max_bins: usize,
    need_dense_values: bool,
) -> Result<PreparedTrainingMatrices, EngineError> {
    validate_continuous_binning_max_bins(max_bins)?;
    let dense_view = DenseMatrixView::new(row_count, feature_count, values)?;
    if targets.len() != dense_view.row_count {
        return Err(EngineError::ContractViolation(format!(
            "rows length {} does not match targets length {}",
            dense_view.row_count,
            targets.len()
        )));
    }

    let mut use_pre_binned_path = true;
    for row_index in 0..dense_view.row_count {
        let row = dense_view.row(row_index)?;
        for (feature_index, &value) in row.iter().enumerate() {
            if value.is_infinite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must not be infinite"
                )));
            }
            if use_pre_binned_path && (value.is_nan() || !is_pre_binned_integer_value(value)) {
                use_pre_binned_path = false;
            }
        }
    }

    // Build binning metadata (shared by u8 and u16 paths).
    let wide_bins = needs_wide_bins(max_bins);
    let (metadata, use_wide) = if use_pre_binned_path {
        (ContinuousBinningMetadataInternal::pre_binned(), wide_bins)
    } else {
        let meta = match strategy {
            ContinuousBinningStrategy::Linear => {
                let (feature_mins, feature_maxs) =
                    derive_dense_feature_bounds(values, row_count, feature_count);
                if linear_tail_rank_enabled_from_env() {
                    let sorted_values =
                        derive_dense_sorted_feature_values(values, row_count, feature_count);
                    let rank_flags = derive_linear_tail_rank_plan(
                        &sorted_values,
                        linear_tail_core_span_ratio_threshold_from_env(),
                    );
                    ContinuousBinningMetadataInternal {
                        uses_continuous_binning: true,
                        feature_mins: Some(feature_mins),
                        feature_maxs: Some(feature_maxs),
                        feature_sorted_values: if rank_flags.iter().any(|flag| *flag) {
                            Some(sorted_values)
                        } else {
                            None
                        },
                        feature_quantile_cuts: None,
                        feature_linear_rank_flags: Some(rank_flags),
                    }
                } else {
                    ContinuousBinningMetadataInternal {
                        uses_continuous_binning: true,
                        feature_mins: Some(feature_mins),
                        feature_maxs: Some(feature_maxs),
                        feature_sorted_values: None,
                        feature_quantile_cuts: None,
                        feature_linear_rank_flags: None,
                    }
                }
            }
            ContinuousBinningStrategy::Rank => ContinuousBinningMetadataInternal {
                uses_continuous_binning: true,
                feature_mins: None,
                feature_maxs: None,
                feature_sorted_values: Some(derive_dense_sorted_feature_values(
                    values,
                    row_count,
                    feature_count,
                )),
                feature_quantile_cuts: None,
                feature_linear_rank_flags: None,
            },
            ContinuousBinningStrategy::Quantile => ContinuousBinningMetadataInternal {
                uses_continuous_binning: true,
                feature_mins: None,
                feature_maxs: None,
                feature_sorted_values: None,
                feature_quantile_cuts: Some(derive_dense_feature_quantile_cuts(
                    values,
                    row_count,
                    feature_count,
                    max_bins,
                )),
                feature_linear_rank_flags: None,
            },
        };
        (meta, wide_bins)
    };

    // Encode bins and build BinnedMatrix — u8 fast path or u16 wide path.
    if use_wide {
        let max_data_bin = (max_bins - 2) as u16;
        let nan_bin = max_data_bin + 1;
        let (dense_values, bins_u16, max_bin) = if use_pre_binned_path {
            // Pre-binned u16 path: Python already quantized values as f32 integers.
            let mut bins_u16 = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let rounded = value.round();
                if rounded > 65535.0 {
                    return Err(EngineError::ContractViolation(format!(
                        "value at index {index} exceeds max supported bin 65535"
                    )));
                }
                let bin = rounded as u16;
                max_bin = max_bin.max(bin);
                bins_u16.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            (dense_values_out, bins_u16, max_bin)
        } else {
            quantize_dense_values_with_metadata_wide(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                need_dense_values,
                max_data_bin,
            )?
        };
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
        };
        let binned_matrix = BinnedMatrix::new_u16(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            nan_bin,
            bins_u16,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata,
        })
    } else {
        // u8 path: pre-binned or continuous with max_bins <= 256.
        let (dense_values, bins, max_bin) = if use_pre_binned_path {
            let mut bins = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let rounded = value.round();
                if rounded > 255.0 {
                    return Err(EngineError::ContractViolation(format!(
                        "value at index {index} exceeds max supported bin 255"
                    )));
                }
                let bin = rounded as u8;
                max_bin = max_bin.max(u16::from(bin));
                bins.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            (dense_values_out, bins, max_bin)
        } else {
            quantize_dense_values_with_metadata(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                need_dense_values,
            )?
        };
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
        };
        let binned_matrix = BinnedMatrix::new(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            bins,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata,
        })
    }
}

fn predictor_error_to_pyerr(error: PredictorError) -> PyErr {
    match error {
        PredictorError::InvalidInput(message) => PyValueError::new_err(message),
        PredictorError::ContractViolation(message) => PyRuntimeError::new_err(message),
        PredictorError::Core(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

fn engine_error_to_pyerr(error: EngineError) -> PyErr {
    match error {
        EngineError::InvalidConfig(message) | EngineError::ContractViolation(message) => {
            PyValueError::new_err(message)
        }
        EngineError::BackendUnavailable(message) | EngineError::NotImplemented(message) => {
            PyRuntimeError::new_err(message)
        }
        EngineError::Core(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

fn shap_error_to_pyerr(error: ShapError) -> PyErr {
    match error {
        ShapError::InvalidInput(message) => PyValueError::new_err(message),
        ShapError::ContractViolation(message) => PyRuntimeError::new_err(message),
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
    validation_values: Option<&[f32]>,
    validation_row_count: Option<usize>,
    validation_targets: Option<&[f32]>,
    validation_sample_weights: Option<Vec<f32>>,
    validation_group_id: Option<Vec<u32>>,
    params: TrainParams,
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
            Some(WarmStartState {
                baseline_prediction: init_model.baseline_prediction,
                stumps: init_model.stumps,
                initial_rounds_completed: initial_rounds,
            })
        } else {
            None
        }
    } else {
        None
    };

    let bridge_prepare_seconds = bridge_start.elapsed().as_secs_f64();
    let user_seed = params.seed;
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
                Some(MultiClassWarmStartState {
                    baseline_predictions: init_mc_model.baseline_predictions,
                    class_stumps: init_mc_model.class_stumps,
                    initial_rounds_completed: initial_rounds,
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
                 rank_ndcg, rank_xendcg, yetirank"
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
    max_leaves: Option<usize>,
    tree_growth: TreeGrowth,
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
        max_leaves,
        tree_growth,
        morph_config: None,
    }
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
    objective="squared_error"
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
) -> PyResult<Vec<u8>> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
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
        None,
        TreeGrowth::Level,
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
    objective="squared_error"
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
) -> PyResult<Vec<u8>> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
    let continuous_binning_strategy =
        parse_continuous_binning_strategy(continuous_binning_strategy)
            .map_err(engine_error_to_pyerr)?;
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
        None,
        TreeGrowth::Level,
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
    max_cat_threshold=0
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
        max_leaves,
        tree_growth,
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
    max_cat_threshold=0
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
        max_leaves,
        tree_growth,
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
    max_cat_threshold=0
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
        max_leaves,
        tree_growth,
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

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_class::<NativePredictorHandle>()?;
    m.add_class::<NativeContinuousBinningMetadata>()?;
    m.add_class::<NativeTrainingSummary>()?;
    m.add_class::<NativeTrainingResult>()?;
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
    m.add_function(wrap_pyfunction!(shap_global_importance, m)?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance_dense, m)?)?;
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
        BinnedMatrix, DatasetMatrix, ModelSectionKind, TrainParams, TrainingDataset, TreeGrowth,
        deserialize_model_artifact_v1, serialize_model_artifact_v1,
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
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
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
