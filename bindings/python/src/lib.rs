use alloygbm_predictor::{Predictor, PredictorError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

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

fn predictor_error_to_pyerr(error: PredictorError) -> PyErr {
    match error {
        PredictorError::InvalidInput(message) => PyValueError::new_err(message),
        PredictorError::ContractViolation(message) => PyRuntimeError::new_err(message),
        PredictorError::Core(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
fn predictor_predict_batch(artifact_bytes: &[u8], rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let predictor =
        Predictor::from_artifact_bytes(artifact_bytes).map_err(predictor_error_to_pyerr)?;
    predictor
        .predict_batch(&rows)
        .map_err(predictor_error_to_pyerr)
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_function(wrap_pyfunction!(native_runtime_info, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch, m)?)?;
    Ok(())
}
