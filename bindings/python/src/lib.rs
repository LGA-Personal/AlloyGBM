#![allow(clippy::too_many_arguments)]

mod callbacks;
mod categorical_bridge;
mod errors;
mod joint;
mod params;
mod predict;
mod pyclasses;
mod quantization;
mod shap_bridge;
mod train;
use crate::joint::{JointPredictorHandle, train_joint_multi_label_ranker};
use crate::predict::{
    NativePredictorHandle, predictor_predict_batch, predictor_predict_batch_canonical,
    predictor_predict_batch_canonical_dense, predictor_predict_batch_dense,
};
use crate::pyclasses::{
    NativeContinuousBinningMetadata, NativeIterationDiagnostics, NativeRuntimeInfo,
    NativeTrainingResult, NativeTrainingSummary, native_runtime_info,
};
use crate::shap_bridge::{
    shap_explain_interactions, shap_explain_interactions_dense,
    shap_explain_interactions_dense_multi, shap_explain_interactions_dense_with_binning,
    shap_explain_interactions_dense_with_binning_multi, shap_explain_interactions_multi,
    shap_explain_interactions_with_binning, shap_explain_interactions_with_binning_multi,
    shap_explain_rows, shap_explain_rows_dense, shap_explain_rows_dense_multi,
    shap_explain_rows_dense_with_binning, shap_explain_rows_dense_with_binning_multi,
    shap_explain_rows_multi, shap_explain_rows_with_binning, shap_explain_rows_with_binning_multi,
    shap_global_importance, shap_global_importance_dense,
    shap_global_importance_dense_with_binning, shap_global_importance_with_binning,
};
use crate::train::{
    train_regression_artifact, train_regression_artifact_dense,
    train_regression_artifact_dense_with_summary,
    train_regression_artifact_dense_with_summary_bytes, train_regression_artifact_with_summary,
};

use alloygbm_core::DenseMatrixView;
use pyo3::prelude::*;

pub(crate) const DEFAULT_TRAIN_ROUNDS: usize = 6;
pub(crate) const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
pub(crate) const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U8: u16 = 254;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U16: u16 = 65534;
pub(crate) const MIN_CONTINUOUS_QUANTIZED_BINS: usize = 2;
pub(crate) const LINEAR_TAIL_RANK_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK";
pub(crate) const LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO";
pub(crate) const DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD: f32 = 0.10;

pub(crate) fn dense_rows_from_flat_values(
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
    m.add_function(wrap_pyfunction!(shap_explain_rows_multi, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_dense_multi, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_multi, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_dense_multi, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_with_binning_multi, m)?)?;
    m.add_function(wrap_pyfunction!(
        shap_explain_rows_dense_with_binning_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shap_explain_interactions_with_binning_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shap_explain_interactions_dense_with_binning_multi,
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
mod tests;
