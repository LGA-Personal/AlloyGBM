#![allow(clippy::type_complexity)]

use crate::dense_rows_from_flat_values;
use crate::errors::shap_error_to_pyerr;
use crate::params::build_binning_context;
use alloygbm_shap::{
    ShapError, explain_interactions_from_artifact_bytes,
    explain_interactions_from_artifact_bytes_per_output,
    explain_interactions_from_artifact_bytes_with_binning,
    explain_interactions_from_artifact_bytes_with_binning_per_output,
    explain_rows_from_artifact_bytes, explain_rows_from_artifact_bytes_per_output,
    explain_rows_from_artifact_bytes_with_binning,
    explain_rows_from_artifact_bytes_with_binning_per_output,
    global_importance_from_artifact_bytes, global_importance_from_artifact_bytes_with_binning,
};
use pyo3::prelude::*;

pub(crate) fn shap_explain_rows_impl(
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

pub(crate) fn shap_explain_rows_multi_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<Vec<Vec<f32>>>), ShapError> {
    let explanation = explain_rows_from_artifact_bytes_per_output(artifact_bytes, rows)?;
    let expected_values = explanation.iter().map(|b| b.expected_value).collect();
    let values = explanation.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

fn shap_explain_rows_dense_multi_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<(Vec<f32>, Vec<Vec<Vec<f32>>>), ShapError> {
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(ShapError::InvalidInput)?;
    shap_explain_rows_multi_impl(artifact_bytes, &rows)
}

#[allow(clippy::type_complexity)]
fn shap_explain_interactions_multi_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>), ShapError> {
    let batch = explain_interactions_from_artifact_bytes_per_output(artifact_bytes, rows)?;
    let expected_values = batch.iter().map(|b| b.expected_value).collect();
    let values = batch.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

#[allow(clippy::type_complexity)]
fn shap_explain_interactions_dense_multi_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>), ShapError> {
    let rows = dense_rows_from_flat_values(values, row_count, feature_count)
        .map_err(ShapError::InvalidInput)?;
    shap_explain_interactions_multi_impl(artifact_bytes, &rows)
}

pub(crate) fn shap_global_importance_impl(
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
pub(crate) fn shap_explain_rows(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<(f32, Vec<Vec<f32>>)> {
    py.detach(|| shap_explain_rows_impl(artifact_bytes, &rows))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_rows_dense(
    py: Python<'_>,
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(f32, Vec<Vec<f32>>)> {
    py.detach(|| shap_explain_rows_dense_impl(artifact_bytes, row_count, feature_count, &values))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_interactions(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    py.detach(|| shap_explain_interactions_impl(artifact_bytes, &rows))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_interactions_dense(
    py: Python<'_>,
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
    py.detach(|| {
        shap_explain_interactions_dense_impl(artifact_bytes, row_count, feature_count, &values)
    })
    .map_err(shap_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, rows, binning_kind, feature_mins=None, feature_maxs=None,
    max_data_bin=None, feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_interactions_with_binning(
    py: Python<'_>,
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
    let batch = py
        .detach(|| {
            explain_interactions_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        })
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
pub(crate) fn shap_explain_interactions_dense_with_binning(
    py: Python<'_>,
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
    let batch = py
        .detach(|| {
            explain_interactions_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx)
        })
        .map_err(shap_error_to_pyerr)?;
    Ok((batch.expected_value, batch.values))
}

#[pyfunction]
pub(crate) fn shap_explain_rows_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<f32>>>)> {
    py.detach(|| shap_explain_rows_multi_impl(artifact_bytes, &rows))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_rows_dense_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<f32>>>)> {
    py.detach(|| {
        shap_explain_rows_dense_multi_impl(artifact_bytes, row_count, feature_count, &values)
    })
    .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_interactions_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>)> {
    py.detach(|| shap_explain_interactions_multi_impl(artifact_bytes, &rows))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_explain_interactions_dense_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>)> {
    py.detach(|| {
        shap_explain_interactions_dense_multi_impl(
            artifact_bytes,
            row_count,
            feature_count,
            &values,
        )
    })
    .map_err(shap_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, rows, binning_kind, feature_mins=None, feature_maxs=None,
    max_data_bin=None, feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_interactions_with_binning_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>)> {
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let batches = py
        .detach(|| {
            explain_interactions_from_artifact_bytes_with_binning_per_output(
                artifact_bytes,
                &rows,
                &ctx,
            )
        })
        .map_err(shap_error_to_pyerr)?;
    let expected_values = batches.iter().map(|b| b.expected_value).collect();
    let values = batches.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, values, row_count, feature_count, binning_kind,
    feature_mins=None, feature_maxs=None, max_data_bin=None,
    feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_interactions_dense_with_binning_multi(
    py: Python<'_>,
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
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<Vec<f32>>>>)> {
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
    let batches = py
        .detach(|| {
            explain_interactions_from_artifact_bytes_with_binning_per_output(
                artifact_bytes,
                &rows,
                &ctx,
            )
        })
        .map_err(shap_error_to_pyerr)?;
    let expected_values = batches.iter().map(|b| b.expected_value).collect();
    let values = batches.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, rows, binning_kind, feature_mins=None, feature_maxs=None,
    max_data_bin=None, feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_rows_with_binning(
    py: Python<'_>,
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
    let explanation = py
        .detach(|| explain_rows_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx))
        .map_err(shap_error_to_pyerr)?;
    Ok((explanation.expected_value, explanation.values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, values, row_count, feature_count, binning_kind,
    feature_mins=None, feature_maxs=None, max_data_bin=None,
    feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_rows_dense_with_binning(
    py: Python<'_>,
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
    let explanation = py
        .detach(|| explain_rows_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx))
        .map_err(shap_error_to_pyerr)?;
    Ok((explanation.expected_value, explanation.values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, rows, binning_kind, feature_mins=None, feature_maxs=None,
    max_data_bin=None, feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_rows_with_binning_multi(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
    binning_kind: &str,
    feature_mins: Option<Vec<f32>>,
    feature_maxs: Option<Vec<f32>>,
    max_data_bin: Option<u16>,
    feature_cuts: Option<Vec<Vec<f32>>>,
    linear_rank_per_feature: Option<Vec<Option<Vec<f32>>>>,
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<f32>>>)> {
    let ctx = build_binning_context(
        binning_kind,
        feature_mins,
        feature_maxs,
        max_data_bin,
        feature_cuts,
        linear_rank_per_feature,
    )
    .map_err(shap_error_to_pyerr)?;
    let explanations = py
        .detach(|| {
            explain_rows_from_artifact_bytes_with_binning_per_output(artifact_bytes, &rows, &ctx)
        })
        .map_err(shap_error_to_pyerr)?;
    let expected_values = explanations.iter().map(|b| b.expected_value).collect();
    let values = explanations.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

#[pyfunction]
#[pyo3(signature = (
    artifact_bytes, values, row_count, feature_count, binning_kind,
    feature_mins=None, feature_maxs=None, max_data_bin=None,
    feature_cuts=None, linear_rank_per_feature=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shap_explain_rows_dense_with_binning_multi(
    py: Python<'_>,
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
) -> PyResult<(Vec<f32>, Vec<Vec<Vec<f32>>>)> {
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
    let explanations = py
        .detach(|| {
            explain_rows_from_artifact_bytes_with_binning_per_output(artifact_bytes, &rows, &ctx)
        })
        .map_err(shap_error_to_pyerr)?;
    let expected_values = explanations.iter().map(|b| b.expected_value).collect();
    let values = explanations.into_iter().map(|b| b.values).collect();
    Ok((expected_values, values))
}

#[pyfunction]
pub(crate) fn shap_global_importance(
    py: Python<'_>,
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<Vec<(String, f32)>> {
    py.detach(|| shap_global_importance_impl(artifact_bytes, &rows))
        .map_err(shap_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn shap_global_importance_dense(
    py: Python<'_>,
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<(String, f32)>> {
    py.detach(|| {
        shap_global_importance_dense_impl(artifact_bytes, row_count, feature_count, &values)
    })
    .map_err(shap_error_to_pyerr)
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
pub(crate) fn shap_global_importance_with_binning(
    py: Python<'_>,
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
    py.detach(|| global_importance_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx))
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
pub(crate) fn shap_global_importance_dense_with_binning(
    py: Python<'_>,
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
    py.detach(|| global_importance_from_artifact_bytes_with_binning(artifact_bytes, &rows, &ctx))
        .map_err(shap_error_to_pyerr)
}
