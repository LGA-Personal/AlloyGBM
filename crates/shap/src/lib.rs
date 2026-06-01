use alloygbm_engine::TrainedModel;

mod binning;
mod brute_force;
mod error;
mod importance;
mod linear_leaf;
mod tree_shap;
mod types;

use binning::MAX_EXACT_SPLIT_FEATURES;
use brute_force::{explain_rows_brute_force, validate_rows};
use linear_leaf::scale_model_by_tree_weight;
use tree_shap::{explain_interactions_from_model, explain_rows_tree_shap};
use types::load_artifact_context;

pub use binning::BinningContext;
pub use error::{ShapError, ShapResult};
pub use importance::{
    global_importance_from_artifact_bytes, global_importance_from_artifact_bytes_with_binning,
    global_importance_from_shap_values, global_importance_stub, shap_values_stub,
};
pub use types::{ShapExplanationBatch, ShapInteractionBatch};

pub fn explain_rows_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<ShapExplanationBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    if context.models.len() != 1 {
        return Err(ShapError::ContractViolation(
            "Expected a single-output model. For multi-class/multi-output models, use the _per_output variants.".to_string(),
        ));
    }
    explain_rows_from_model(&context.models[0], rows, None)
}

/// Predictor-aligned variant of `explain_rows_from_artifact_bytes`.
///
/// When the caller supplies a `BinningContext`, the SHAP path walker
/// uses the same float-threshold-and-strict-less-than semantics as the
/// predictor's `convert_bin_thresholds_to_float*` family, so per-row
/// attributions reach the same leaf the predictor reaches.  This is
/// required for `leaf_model="linear"` artifacts trained on continuous
/// features — the legacy bin-index path-walker diverges and produces
/// best-effort attributions that fail strict additivity.
///
/// Callers without a `BinningContext` should keep using the legacy
/// entry point above; behavior is unchanged.
pub fn explain_rows_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<ShapExplanationBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    if context.models.len() != 1 {
        return Err(ShapError::ContractViolation(
            "Expected a single-output model. For multi-class/multi-output models, use the _per_output variants.".to_string(),
        ));
    }
    explain_rows_from_model(&context.models[0], rows, Some(binning))
}

/// Compute pairwise SHAP interaction values for the given rows.
///
/// Implements Lundberg et al. (2020) Algorithm 2, "Polynomial-time consistent
/// individualized feature attributions" — extended to pairwise interactions.
/// Cost: `O(T · L · D² · M)` where `M` is the feature count.
///
/// For piecewise-linear leaves (`leaf_model="linear"`), the row-dependent linear
/// deviation is distributed strictly to the diagonal of the interaction matrix
/// (the regressor feature's main effect), preserving both row-marginal and full
/// additivity.
pub fn explain_interactions_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<ShapInteractionBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    if context.models.len() != 1 {
        return Err(ShapError::ContractViolation(
            "Expected a single-output model. For multi-class/multi-output models, use the _per_output variants.".to_string(),
        ));
    }
    explain_interactions_from_model(&context.models[0], rows, None)
}

/// Predictor-aligned variant. See `explain_rows_from_artifact_bytes_with_binning`
/// for the contract.
pub fn explain_interactions_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<ShapInteractionBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    if context.models.len() != 1 {
        return Err(ShapError::ContractViolation(
            "Expected a single-output model. For multi-class/multi-output models, use the _per_output variants.".to_string(),
        ));
    }
    explain_interactions_from_model(&context.models[0], rows, Some(binning))
}

/// Per-output variant of [`explain_rows_from_artifact_bytes`]. Returns one
/// [`ShapExplanationBatch`] per model output:
///
/// - For multiclass classifiers (`num_classes >= 2`): element `k` is the SHAP
///   attribution for class `k`'s logit.
/// - For joint multi-output rankers (`MultiOutputLeafValues` section present):
///   element `k` is the SHAP attribution for output `k`.
/// - For single-output regressors: returns a 1-element `Vec`.
///
/// Additivity per output:
/// `Σⱼ values[k][i][j] + expected_values[k] ≈ raw_prediction_k(rows[i])`
/// within `atol = 1e-5 + rtol = 1e-4 · |raw_prediction_k|`.
pub fn explain_rows_from_artifact_bytes_per_output(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<Vec<ShapExplanationBatch>> {
    let context = load_artifact_context(artifact_bytes)?;
    context
        .models
        .iter()
        .map(|model| explain_rows_from_model(model, rows, None))
        .collect()
}

/// Predictor-aligned per-output variant. Combines the binning-context
/// semantics of [`explain_rows_from_artifact_bytes_with_binning`] with the
/// per-output fan-out of [`explain_rows_from_artifact_bytes_per_output`].
/// Required for `leaf_model="linear"` or `continuous_binning_strategy=
/// "linear"` (LinearRank) joint multi-output artifacts to reach the same
/// leaves as the predictor.
pub fn explain_rows_from_artifact_bytes_with_binning_per_output(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<Vec<ShapExplanationBatch>> {
    let context = load_artifact_context(artifact_bytes)?;
    context
        .models
        .iter()
        .map(|model| explain_rows_from_model(model, rows, Some(binning)))
        .collect()
}

/// Per-output variant of [`explain_interactions_from_artifact_bytes`].
/// Returns one [`ShapInteractionBatch`] per model output. Semantics match
/// [`explain_rows_from_artifact_bytes_per_output`] — see that function's
/// docs for the output-index mapping across multiclass and multi-output
/// artifact types.
///
/// Cost: `O(K · T · L · D² · M)` where `K` is the number of outputs.
pub fn explain_interactions_from_artifact_bytes_per_output(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<Vec<ShapInteractionBatch>> {
    let context = load_artifact_context(artifact_bytes)?;
    context
        .models
        .iter()
        .map(|model| explain_interactions_from_model(model, rows, None))
        .collect()
}

/// Predictor-aligned per-output variant of
/// [`explain_interactions_from_artifact_bytes`]. Combines binning-context
/// semantics with per-output fan-out; see
/// [`explain_rows_from_artifact_bytes_with_binning_per_output`] for the
/// binning contract and [`explain_interactions_from_artifact_bytes_per_output`]
/// for the output-index semantics.
pub fn explain_interactions_from_artifact_bytes_with_binning_per_output(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<Vec<ShapInteractionBatch>> {
    let context = load_artifact_context(artifact_bytes)?;
    context
        .models
        .iter()
        .map(|model| explain_interactions_from_model(model, rows, Some(binning)))
        .collect()
}

pub(crate) fn explain_rows_from_model(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    validate_rows(rows, model.feature_count)?;
    if let Some(ctx) = binning {
        ctx.validate(model.feature_count)?;
    }

    // v0.9.0: DART artifacts carry a per-stump `tree_weight` that the
    // predictor multiplies into the leaf contribution.  All downstream
    // SHAP attribution code (brute-force, TreeSHAP, PL-leaf
    // interventional decomposition, additivity check) operates on
    // unweighted leaf values, so for DART models we fold `tree_weight`
    // into the leaf values up-front and reset weights to 1.0 on a
    // clone.  Folding preserves additivity because the scaling is
    // applied to every leaf and every interventional term
    // uniformly — `predict(x) = Σ tree_weight · leaf` becomes
    // `predict(x) = Σ (tree_weight · leaf)` on the scaled model,
    // which the existing additivity check handles natively.  Non-DART
    // models all have `tree_weight = 1.0`; the clone is bit-identical
    // and adds one allocation but no other overhead.
    let has_non_unit_weights = model
        .stumps
        .iter()
        .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON);
    if has_non_unit_weights {
        let scaled = scale_model_by_tree_weight(model);
        return explain_rows_from_model(&scaled, rows, binning);
    }

    // LinearRank: the predictor evaluates both tree traversal and PL
    // leaves in bin-index space, so quantize rows once at the entry
    // point and dispatch with PreBinned semantics for the remainder.
    // See the `BinningContext::LinearRank` doc comment for the parity
    // rationale.
    if let Some(ctx @ BinningContext::LinearRank { .. }) = binning {
        let quantized: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                ctx.quantize_row_for_linear_rank(row)
                    .expect("LinearRank quantize_row_for_linear_rank returns Some")
            })
            .collect();
        return explain_rows_from_model(model, &quantized, Some(&BinningContext::PreBinned));
    }

    // Count distinct split features to choose algorithm.
    let distinct_split_feature_count = {
        let mut features: Vec<usize> = model
            .stumps
            .iter()
            .map(|s| s.split.feature_index as usize)
            .collect();
        features.sort_unstable();
        features.dedup();
        features.len()
    };

    if distinct_split_feature_count > MAX_EXACT_SPLIT_FEATURES {
        // Too many features for brute-force O(2^N); use TreeSHAP O(TLD^2).
        return explain_rows_tree_shap(model, rows, binning);
    }

    // Brute-force exact Shapley values for models with few split features.
    explain_rows_brute_force(model, rows, binning)
}

#[cfg(test)]
mod tests;
