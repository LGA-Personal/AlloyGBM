use std::cmp::Ordering;

use alloygbm_core::ModelMetadata;

use crate::binning::BinningContext;
use crate::error::{ShapError, ShapResult};
use crate::types::load_artifact_context;
use crate::{explain_rows_from_model, validate_rows};

pub fn global_importance_from_shap_values(
    feature_names: &[String],
    shap_values: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if shap_values.is_empty() {
        return Err(ShapError::InvalidInput(
            "shap_values cannot be empty".to_string(),
        ));
    }

    let feature_count = feature_names.len();
    let mut contribution_sums = vec![0.0_f32; feature_count];
    for (row_index, row_values) in shap_values.iter().enumerate() {
        if row_values.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row_values.len()
            )));
        }
        for (feature_index, value) in row_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} contribution must be finite"
                )));
            }
            contribution_sums[feature_index] += value.abs();
        }
    }

    let row_count = shap_values.len() as f32;
    let mut global_importance = feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            (
                feature_name.clone(),
                contribution_sums[feature_index] / row_count,
            )
        })
        .collect::<Vec<_>>();

    global_importance.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    Ok(global_importance)
}
fn global_importance_helper(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<Vec<(String, f32)>> {
    let context = load_artifact_context(artifact_bytes)?;
    let mut total_contribution_sums = vec![0.0_f32; context.feature_names.len()];

    for model in &context.models {
        let explanation = explain_rows_from_model(model, rows, binning)?;
        for row_values in &explanation.values {
            for (feature_index, value) in row_values.iter().enumerate() {
                total_contribution_sums[feature_index] += value.abs();
            }
        }
    }

    let n_models = context.models.len() as f32;
    let row_count = rows.len() as f32;
    let divisor = row_count * n_models;
    let mut global_importance = context
        .feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            (
                feature_name.clone(),
                total_contribution_sums[feature_index] / divisor,
            )
        })
        .collect::<Vec<_>>();

    global_importance.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    Ok(global_importance)
}

pub fn global_importance_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    global_importance_helper(artifact_bytes, rows, None)
}

pub fn global_importance_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<Vec<(String, f32)>> {
    global_importance_helper(artifact_bytes, rows, Some(binning))
}
// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `explain_rows_from_artifact_bytes` for artifact-backed explanations.
pub fn shap_values_stub(metadata: &ModelMetadata, rows: &[Vec<f32>]) -> ShapResult<Vec<Vec<f32>>> {
    let feature_count = metadata.feature_names.len();
    validate_rows(rows, feature_count)?;
    Ok(vec![vec![0.0; feature_count]; rows.len()])
}

// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `global_importance_from_shap_values`.
pub fn global_importance_stub(
    metadata: &ModelMetadata,
    feature_names: &[String],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if feature_names.len() != metadata.feature_names.len() {
        return Err(ShapError::InvalidInput(format!(
            "feature_names length {} does not match metadata feature count {}",
            feature_names.len(),
            metadata.feature_names.len()
        )));
    }

    Ok(feature_names
        .iter()
        .map(|name| (name.clone(), 0.0_f32))
        .collect())
}
