use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, DatasetMatrix, TrainParams, TrainingDataset};
use alloygbm_engine::{EngineError, SquaredErrorObjective, Trainer};
use alloygbm_predictor::{Predictor, PredictorError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const DEFAULT_TRAIN_ROUNDS: usize = 6;

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

fn train_regression_artifact_impl(
    rows: &[Vec<f32>],
    targets: &[f32],
    params: TrainParams,
    rounds: usize,
) -> Result<Vec<u8>, EngineError> {
    if rows.is_empty() {
        return Err(EngineError::ContractViolation(
            "rows cannot be empty".to_string(),
        ));
    }
    if targets.len() != rows.len() {
        return Err(EngineError::ContractViolation(format!(
            "rows length {} does not match targets length {}",
            rows.len(),
            targets.len()
        )));
    }

    let feature_count = rows[0].len();
    if feature_count == 0 {
        return Err(EngineError::ContractViolation(
            "rows must include at least one feature".to_string(),
        ));
    }

    let mut dense_values = Vec::with_capacity(rows.len() * feature_count);
    let mut bins = Vec::with_capacity(rows.len() * feature_count);
    let mut max_bin = 0_u16;
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != feature_count {
            return Err(EngineError::ContractViolation(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row.len()
            )));
        }
        for (feature_index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must be finite"
                )));
            }
            if value < 0.0 {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must be >= 0 for pre-binned training"
                )));
            }

            let rounded = value.round();
            if (value - rounded).abs() > 1e-6 {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must be an integer-valued bin"
                )));
            }
            if rounded > u16::MAX as f32 {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} exceeds max supported bin {}",
                    u16::MAX
                )));
            }

            let bin = rounded as u16;
            max_bin = max_bin.max(bin);
            dense_values.push(value);
            bins.push(bin);
        }
    }

    let matrix = DatasetMatrix::new(rows.len(), feature_count, dense_values)?;
    let dataset = TrainingDataset {
        matrix,
        targets: targets.to_vec(),
        sample_weights: None,
        time_index: None,
        group_id: None,
    };
    let binned = BinnedMatrix::new(
        rows.len(),
        feature_count,
        if max_bin == 0 { 1 } else { max_bin },
        bins,
    )?;

    let trainer = Trainer::new(params)?;
    let backend = CpuBackend;
    let model =
        trainer.fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, rounds)?;
    model.to_artifact_bytes()
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
    early_stopping_rounds=None
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
) -> PyResult<Vec<u8>> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }

    let params = TrainParams {
        seed,
        deterministic,
        learning_rate,
        max_depth,
        row_subsample,
        col_subsample,
        early_stopping_rounds,
        min_validation_improvement,
    };

    train_regression_artifact_impl(&rows, &targets, params, rounds).map_err(engine_error_to_pyerr)
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_function(wrap_pyfunction!(native_runtime_info, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch, m)?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TRAIN_ROUNDS, predictor_predict_batch_impl, train_regression_artifact_impl,
    };
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

    #[test]
    fn train_bridge_artifact_predictions_match_engine_predictions() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let rows = fixture_rows(&dataset);
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

        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
        )
        .expect("bridge training succeeds");
        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("bridge predicts");

        assert_eq!(bridge_predictions, engine_predictions);
    }
}
