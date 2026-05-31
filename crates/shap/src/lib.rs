use std::collections::HashMap;

use alloygbm_engine::{TrainedModel, TrainedStump};

mod binning;
mod brute_force;
mod error;
mod importance;
mod linear_leaf;
mod tree_shap;
mod types;

use binning::MAX_EXACT_SPLIT_FEATURES;
use brute_force::{decode_tree_node_id, explain_rows_brute_force, tree_local_key, validate_rows};
use linear_leaf::{model_has_linear_leaves, scale_model_by_tree_weight};
use tree_shap::{build_std_tree, explain_rows_tree_shap, tree_shap_interactions_row};
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
    explain_rows_from_model(&context.model, rows, None)
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
    explain_rows_from_model(&context.model, rows, Some(binning))
}

/// Compute pairwise SHAP interaction values for the given rows.
///
/// Implements Lundberg et al. (2020) Algorithm 2, "Polynomial-time consistent
/// individualized feature attributions" — extended to pairwise interactions.
/// Cost: `O(T · L · D² · M)` where `M` is the feature count.
///
/// Linear-leaf (PL) models are rejected by this entry point in v0.12.4 —
/// see `docs/limitations.md` for the deferred extension.
pub fn explain_interactions_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<ShapInteractionBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    explain_interactions_from_model(&context.model, rows, None)
}

/// Predictor-aligned variant. See `explain_rows_from_artifact_bytes_with_binning`
/// for the contract.
pub fn explain_interactions_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<ShapInteractionBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    explain_interactions_from_model(&context.model, rows, Some(binning))
}

fn explain_interactions_from_model(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapInteractionBatch> {
    validate_rows(rows, model.feature_count)?;

    // Linear leaves are fully supported for SHAP interactions: we run standard
    // TreeSHAP interactions on the constant parts of the leaves, then attribute
    // the row-dependent linear deviation directly to the regressor feature's
    // main effect (the diagonal of the interaction matrix).

    // Pre-scale DART trees by tree_weight so the standard TreeSHAP path
    // produces strictly-additive contributions.  Mirrors `explain_rows_from_model`.
    let has_non_unit_weights = model
        .stumps
        .iter()
        .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON);
    if has_non_unit_weights {
        let scaled = scale_model_by_tree_weight(model);
        return explain_interactions_from_model(&scaled, rows, binning);
    }

    // LinearRank: quantize once and dispatch with PreBinned semantics.
    if let Some(ctx @ BinningContext::LinearRank { .. }) = binning {
        let quantized: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                ctx.quantize_row_for_linear_rank(row)
                    .expect("LinearRank quantize_row_for_linear_rank returns Some")
            })
            .collect();
        return explain_interactions_from_model(
            model,
            &quantized,
            Some(&BinningContext::PreBinned),
        );
    }

    // Build node lookup and standard trees once for all rows.
    let mut nodes_map: HashMap<u64, &TrainedStump> = HashMap::new();
    let mut tree_roots: Vec<u32> = Vec::new();
    for stump in &model.stumps {
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        nodes_map.insert(tree_local_key(tree_id, local_id), stump);
        if local_id == 0 {
            tree_roots.push(tree_id);
        }
    }
    tree_roots.sort_unstable();
    tree_roots.dedup();

    let baseline = model.feature_baseline.as_deref();
    let mut std_trees = Vec::with_capacity(tree_roots.len());
    let mut expected_value_f64 = model.baseline_prediction as f64;

    for &tree_id in &tree_roots {
        let root_key = tree_local_key(tree_id, 0);
        let root_stump = nodes_map.get(&root_key).ok_or_else(|| {
            ShapError::ContractViolation(format!("missing root stump for tree {tree_id}"))
        })?;
        let root_cover = root_stump.split.left_stats.row_count as f64
            + root_stump.split.right_stats.row_count as f64;
        let tree = build_std_tree(tree_id, 0, 0.0, root_cover, &nodes_map, baseline, binning);
        let tree_cover = tree.cover();
        if tree_cover > 0.0 {
            expected_value_f64 += tree.cover_weighted_value_sum() / tree_cover;
        }
        std_trees.push(tree);
    }

    let expected_value = expected_value_f64 as f32;
    let use_float_compare = binning.is_some();

    let mut all_matrices = Vec::with_capacity(rows.len());
    let has_linear = model_has_linear_leaves(model);
    for row in rows {
        let mut matrix_f64 =
            tree_shap_interactions_row(&std_trees, row, model.feature_count, use_float_compare);
            
        if has_linear {
            let mut linear_phi = vec![0.0_f64; model.feature_count];
            crate::linear_leaf::distribute_linear_terms_for_row(
                model,
                row,
                baseline,
                binning,
                &mut linear_phi,
            );
            for i in 0..model.feature_count {
                matrix_f64[i][i] += linear_phi[i];
            }
        }

        let matrix_f32: Vec<Vec<f32>> = matrix_f64
            .into_iter()
            .map(|inner| inner.into_iter().map(|v| v as f32).collect())
            .collect();
        all_matrices.push(matrix_f32);
    }

    Ok(ShapInteractionBatch {
        expected_value,
        values: all_matrices,
    })
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
