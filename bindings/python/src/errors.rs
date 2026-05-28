use alloygbm_engine::EngineError;
use alloygbm_predictor::PredictorError;
use alloygbm_shap::ShapError;
use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};

pub(crate) fn predictor_error_to_pyerr(error: PredictorError) -> PyErr {
    match error {
        PredictorError::InvalidInput(message) => PyValueError::new_err(message),
        PredictorError::ContractViolation(message) => PyRuntimeError::new_err(message),
        PredictorError::Core(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

pub(crate) fn engine_error_to_pyerr(error: EngineError) -> PyErr {
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

pub(crate) fn shap_error_to_pyerr(error: ShapError) -> PyErr {
    match error {
        ShapError::InvalidInput(message) => PyValueError::new_err(message),
        ShapError::ContractViolation(message) => PyRuntimeError::new_err(message),
        ShapError::NotSupported(message) => PyRuntimeError::new_err(message),
    }
}
