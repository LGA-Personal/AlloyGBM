// ---------------------------------------------------------------------------
// Shared helpers for fast Rust ↔ numpy data transfer
// ---------------------------------------------------------------------------

use alloygbm_core::GradientPair;
use alloygbm_engine::{EngineError, ObjectiveOps, PerRoundMetricCallback};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Extract a `Vec<f32>` from a Python object that is expected to be a numpy
/// array.  Tries the zero-overhead path first (`PyReadonlyArray1<f32>` →
/// `as_slice()` → single `memcpy`) and falls back to `PyReadonlyArray1<f64>`
/// with a narrowing copy, then to generic PyO3 extraction (element-wise,
/// slowest) for non-contiguous or exotic dtypes.
fn extract_f32_array(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<Vec<f32>, EngineError> {
    // Fast path: contiguous f32 numpy array — single memcpy via as_slice()
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        if let Ok(slice) = readonly.as_slice() {
            return Ok(slice.to_vec());
        }
        // Non-contiguous — fall through to generic path
    }
    // Medium path: contiguous f64 numpy array — common when user returns
    // plain Python floats which numpy defaults to float64
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
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
pub(crate) struct CustomPythonObjective {
    grad_hess_fn: Py<PyAny>,
    loss_fn: Option<Py<PyAny>>,
}

impl CustomPythonObjective {
    pub(crate) fn new(grad_hess_fn: Py<PyAny>, loss_fn: Option<Py<PyAny>>) -> Self {
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
        Python::attach(|py| {
            // Create numpy arrays directly from Rust slices — avoids creating a
            // Python list and the redundant .astype("float32") call.  PyArray1::from_slice
            // produces a contiguous f32 numpy array backed by a copy of the slice
            // (one memcpy instead of element-by-element Python object creation).
            let y_true = PyArray1::from_slice(py, targets);
            let y_pred = PyArray1::from_slice(py, predictions);

            let result = self.grad_hess_fn.call1(py, (y_true, y_pred)).map_err(|e| {
                EngineError::ContractViolation(format!("custom objective callable failed: {e}"))
            })?;

            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py).map_err(|_| {
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
            Python::attach(|py| {
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
pub(crate) struct CustomPythonMetricCallback {
    metric_fn: Py<PyAny>,
    /// Name discovered during probe, set before training starts
    name: String,
    /// Direction discovered during probe, set before training starts
    higher_is_better_flag: bool,
}

impl CustomPythonMetricCallback {
    /// Create a new callback by probing the Python callable with dummy data
    /// to discover the metric name and direction.
    pub(crate) fn new(metric_fn: Py<PyAny>) -> Result<Self, EngineError> {
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
        Python::attach(|py| {
            let (y_true, y_pred) = make_numpy_f32_pair(py, targets, predictions);

            let result = metric_fn.call1(py, (y_true, y_pred)).map_err(|e| {
                EngineError::ContractViolation(format!("custom eval metric callable failed: {e}"))
            })?;

            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py).map_err(|_| {
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
