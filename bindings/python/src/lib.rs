use alloygbm_backend_cpu::CpuBackend;
use alloygbm_categorical::TargetEncoderConfig;
use alloygbm_core::{BinnedMatrix, DatasetMatrix, DenseMatrixView, TrainParams, TrainingDataset};
use alloygbm_engine::{
    ArtifactCompatibilityMode, CategoricalTargetEncodingSpec, EngineError, SquaredErrorObjective,
    TrainedModel, Trainer, TrainingPolicyMode,
};
use alloygbm_predictor::{Predictor, PredictorError};
use alloygbm_shap::{
    ShapError, explain_rows_from_artifact_bytes, global_importance_from_artifact_bytes,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const DEFAULT_TRAIN_ROUNDS: usize = 6;
const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
const MAX_CONTINUOUS_QUANTIZED_BIN: u16 = 255;

fn is_pre_binned_integer_value(value: f32) -> bool {
    if value < 0.0 {
        return false;
    }
    let rounded = value.round();
    (value - rounded).abs() <= PRE_BINNED_INTEGER_TOLERANCE
}

fn quantize_continuous_value_to_bin(value: f32) -> u16 {
    let rounded = value.round();
    let clamped = rounded.clamp(0.0, MAX_CONTINUOUS_QUANTIZED_BIN as f32);
    clamped as u16
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

#[allow(clippy::too_many_arguments)]
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

    let mut use_pre_binned_path = true;
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
            if use_pre_binned_path && !is_pre_binned_integer_value(value) {
                use_pre_binned_path = false;
            }
        }
    }

    let mut dense_values = Vec::with_capacity(rows.len() * feature_count);
    for row in rows {
        dense_values.extend_from_slice(row);
    }
    train_regression_artifact_dense_impl(
        &dense_values,
        rows.len(),
        feature_count,
        targets,
        params,
        rounds,
        time_index,
        categorical_spec,
        training_policy,
        store_node_debug_stats,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_regression_artifact_dense_impl(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    params: TrainParams,
    rounds: usize,
    time_index: Option<Vec<i64>>,
    categorical_spec: Option<CategoricalTargetEncodingSpec>,
    training_policy: TrainingPolicyMode,
    store_node_debug_stats: bool,
) -> Result<Vec<u8>, EngineError> {
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
            if !value.is_finite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must be finite"
                )));
            }
            if use_pre_binned_path && !is_pre_binned_integer_value(value) {
                use_pre_binned_path = false;
            }
        }
    }

    let mut bins = Vec::with_capacity(dense_view.row_count * feature_count);
    let mut max_bin = 0_u16;
    let mut dense_values = Vec::with_capacity(values.len());
    for row_index in 0..dense_view.row_count {
        let row = dense_view.row(row_index)?;
        for (feature_index, &value) in row.iter().enumerate() {
            let bin = if use_pre_binned_path {
                let rounded = value.round();
                if rounded > u16::MAX as f32 {
                    return Err(EngineError::ContractViolation(format!(
                        "row {row_index} feature {feature_index} exceeds max supported bin {}",
                        u16::MAX
                    )));
                }
                rounded as u16
            } else {
                quantize_continuous_value_to_bin(value)
            };
            max_bin = max_bin.max(bin);
            dense_values.push(bin as f32);
            bins.push(bin);
        }
    }

    let matrix = DatasetMatrix::new(dense_view.row_count, feature_count, dense_values)?;
    let dataset = TrainingDataset {
        matrix,
        targets: targets.to_vec(),
        sample_weights: None,
        time_index,
        group_id: None,
    };
    let binned = BinnedMatrix::new(
        dense_view.row_count,
        feature_count,
        if max_bin == 0 { 1 } else { max_bin },
        bins,
    )?;

    let trainer = Trainer::new(params)?;
    let backend = CpuBackend;
    let model = if let Some(spec) = categorical_spec.as_ref() {
        trainer.fit_iterations_with_single_target_encoded_feature_and_policy(
            &dataset,
            &binned,
            spec,
            &backend,
            &SquaredErrorObjective,
            rounds,
            training_policy,
            store_node_debug_stats,
        )?
    } else {
        trainer.fit_iterations_with_policy(
            &dataset,
            &binned,
            &backend,
            &SquaredErrorObjective,
            rounds,
            training_policy,
            store_node_debug_stats,
        )?
    };
    model.to_artifact_bytes()
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
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(PredictorError::InvalidInput)?;
    predictor.predict_batch(&rows)
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
    time_index=None
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
) -> PyResult<Vec<u8>> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;

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

    let categorical_spec = resolve_categorical_spec(
        categorical_feature_index,
        categorical_feature_values,
        categorical_smoothing,
        categorical_min_samples_leaf,
        categorical_time_aware,
        rows.len(),
    )
    .map_err(engine_error_to_pyerr)?;

    train_regression_artifact_impl(
        &rows,
        &targets,
        params,
        effective_rounds,
        time_index,
        categorical_spec,
        training_policy,
        store_node_stats,
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
    categorical_feature_index=None,
    categorical_feature_values=None,
    training_policy="auto",
    store_node_stats=false,
    categorical_smoothing=20.0,
    categorical_min_samples_leaf=1,
    categorical_time_aware=false,
    time_index=None
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
) -> PyResult<Vec<u8>> {
    if rounds == 0 {
        return Err(PyValueError::new_err("rounds must be greater than 0"));
    }
    let effective_rounds = rounds.min(MAX_SUPPORTED_TRAIN_ROUNDS);
    let training_policy = parse_training_policy(training_policy).map_err(engine_error_to_pyerr)?;
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
    let categorical_spec = resolve_categorical_spec(
        categorical_feature_index,
        categorical_feature_values,
        categorical_smoothing,
        categorical_min_samples_leaf,
        categorical_time_aware,
        row_count,
    )
    .map_err(engine_error_to_pyerr)?;
    train_regression_artifact_dense_impl(
        &values,
        row_count,
        feature_count,
        &targets,
        params,
        effective_rounds,
        time_index,
        categorical_spec,
        training_policy,
        store_node_stats,
    )
    .map_err(engine_error_to_pyerr)
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_class::<NativePredictorHandle>()?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TRAIN_ROUNDS, predictor_predict_batch_canonical_impl, predictor_predict_batch_impl,
        shap_explain_rows_impl, shap_global_importance_impl, train_regression_artifact_impl,
    };
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_categorical::TargetEncoderConfig;
    use alloygbm_core::{
        BinnedMatrix, DatasetMatrix, ModelSectionKind, TrainParams, TrainingDataset,
        deserialize_model_artifact_v1, serialize_model_artifact_v1,
    };
    use alloygbm_engine::{
        CategoricalTargetEncodingSpec, SquaredErrorObjective, Trainer, TrainingPolicyMode,
    };

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
