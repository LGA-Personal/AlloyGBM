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

fn predictor_predict_batch_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = Predictor::from_artifact_bytes(artifact_bytes)?;
    predictor.predict_batch(rows)
}

#[pyfunction]
fn predictor_predict_batch(artifact_bytes: &[u8], rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    predictor_predict_batch_impl(artifact_bytes, &rows).map_err(predictor_error_to_pyerr)
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_function(wrap_pyfunction!(native_runtime_info, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::predictor_predict_batch_impl;
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_core::{BinnedMatrix, DatasetMatrix, TrainParams, TrainingDataset};
    use alloygbm_engine::{SquaredErrorObjective, Trainer};

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
        }
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
}
