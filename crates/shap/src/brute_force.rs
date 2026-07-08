use std::collections::HashMap;

use alloygbm_engine::{TrainedModel, TrainedStump};

use crate::binning::{
    ADDITIVITY_ATOL, ADDITIVITY_RTOL, BinningContext, TREE_NODE_STRIDE, additivity_tolerance,
};
use crate::error::{ShapError, ShapResult};
use crate::linear_leaf::{
    distribute_linear_terms_for_row, leaf_constant_part, model_has_linear_leaves,
};
use crate::types::{ModelStructure, ShapExplanationBatch, build_model_structure};

pub(crate) fn explain_rows_brute_force(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    let model_structure = build_model_structure(model)?;
    let baseline = model.feature_baseline.as_deref();
    let expected_value = expected_prediction_for_subset(
        model,
        rows[0].as_slice(),
        0,
        &model_structure,
        baseline,
        binning,
    )?;

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let values_by_subset =
            compute_subset_expectations(model, row, &model_structure, baseline, binning)?;
        let row_expected_value = values_by_subset[0];

        if (row_expected_value - expected_value).abs() > additivity_tolerance(expected_value) {
            return Err(ShapError::ContractViolation(format!(
                "row {row_index} expected value drift: {row_expected_value} vs baseline {expected_value}"
            )));
        }

        let mut contributions_f64 =
            shapley_values_for_row_f64(model, row, &values_by_subset, &model_structure, row_index)?;

        // Linear-leaf interventional decomposition: the brute-force path
        // attribution above is computed on the "constant part" of each leaf
        // (`intercept + Σ wj * z_j(baseline_raw_j)`). Adding
        // `wj * (z_j(row_raw_j) - z_j(baseline_raw_j))` per regressor at
        // *every visited node along the row's path* restores `predict(x)`
        // exactly (matching how `predict` accumulates `leaf.eval_row(row)`
        // at each visited node) while attributing the row's deviation
        // directly to the relevant features. See
        // `distribute_linear_terms_for_row` for the full path walk.
        if model_has_linear_leaves(model) {
            distribute_linear_terms_for_row(model, row, baseline, binning, &mut contributions_f64);
        }

        let contributions: Vec<f32> = contributions_f64.iter().map(|v| *v as f32).collect();
        verify_additivity(
            model,
            row,
            &contributions,
            row_index,
            expected_value,
            binning,
        )?;
        row_contributions.push(contributions);
    }

    Ok(ShapExplanationBatch {
        expected_value,
        values: row_contributions,
    })
}

pub(crate) fn compute_subset_expectations(
    model: &TrainedModel,
    row: &[f32],
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<Vec<f32>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = 1_usize
        .checked_shl(split_feature_count as u32)
        .ok_or_else(|| ShapError::ContractViolation("subset count overflow".to_string()))?;

    let mut values_by_subset = Vec::with_capacity(subset_count);
    for subset_mask in 0..subset_count {
        let value = expected_prediction_for_subset(
            model,
            row,
            subset_mask as u64,
            model_structure,
            baseline,
            binning,
        )?;
        values_by_subset.push(value);
    }
    Ok(values_by_subset)
}

/// Determine whether a feature value goes to the left child of a split.
/// Uses bitset membership for categorical splits and threshold comparison
/// for numeric splits.
pub(crate) fn stump_goes_left(
    split: &alloygbm_core::SplitCandidate,
    feature_value: f32,
    binning: Option<&BinningContext>,
) -> bool {
    if feature_value.is_nan() {
        return split.default_left;
    }
    if split.is_categorical {
        let cat_id = feature_value as u16;
        return split
            .categorical_bitset
            .as_ref()
            .map_or(split.default_left, |bs| {
                let byte_idx = (cat_id / 8) as usize;
                let bit_idx = (cat_id % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            });
    }
    match binning {
        // Float-threshold path: matches the predictor's strict `<`
        // comparison after `convert_bin_thresholds_to_float*`.  When a
        // binning context is provided, SHAP walks paths the same way
        // the predictor does, so linear-leaf attribution stays
        // additive on continuous features.
        Some(ctx) => {
            let threshold = ctx.float_threshold(split.feature_index as usize, split.threshold_bin);
            feature_value < threshold
        }
        // Legacy bin-index path.  Preserved for callers that don't
        // (or can't) provide a `BinningContext` — categorical-only
        // and pre-binned-integer artifacts predominantly.
        None => feature_value <= split.threshold_bin as f32,
    }
}

fn expected_prediction_for_subset(
    model: &TrainedModel,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<f32> {
    let mut prediction = model.baseline_prediction;
    for tree_id in &model_structure.tree_root_ids {
        prediction += expected_subtree(
            *tree_id,
            0,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;
    }
    Ok(prediction)
}

fn expected_subtree(
    tree_id: u32,
    local_node_id: u32,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<f32> {
    let node_key = tree_local_key(tree_id, local_node_id);
    let Some(stump) = model_structure.nodes_by_tree_local_id.get(&node_key) else {
        return Ok(0.0);
    };

    let split_feature_index = stump.split.feature_index as usize;
    if split_feature_index >= row.len() {
        return Err(ShapError::ContractViolation(format!(
            "split feature_index {} exceeds row feature length {}",
            stump.split.feature_index,
            row.len()
        )));
    }

    let left_child_local = left_child_local_id(local_node_id)?;
    let right_child_local = right_child_local_id(local_node_id)?;

    // Use the leaf "constant part" —
    // `intercept + Σ wj * z_j(baseline_raw_j)` for linear leaves — so the
    // path-based attribution acts on a scalar-valued tree. Linear
    // deviations `wj * (z_j(row_raw_j) - z_j(baseline_raw_j))` are added
    // back to phi after the Shapley computation by
    // `distribute_linear_terms_for_row`.
    let left_const = leaf_constant_part(&stump.left_leaf_value, baseline) as f32;
    let right_const = leaf_constant_part(&stump.right_leaf_value, baseline) as f32;

    if let Some(bit_position) = model_structure.split_feature_bit_positions[split_feature_index] {
        let is_known = (subset_mask & (1_u64 << bit_position)) != 0;
        if is_known {
            let goes_left = stump_goes_left(&stump.split, row[split_feature_index], binning);
            if goes_left {
                return Ok(left_const
                    + expected_subtree(
                        tree_id,
                        left_child_local,
                        row,
                        subset_mask,
                        model_structure,
                        baseline,
                        binning,
                    )?);
            }
            return Ok(right_const
                + expected_subtree(
                    tree_id,
                    right_child_local,
                    row,
                    subset_mask,
                    model_structure,
                    baseline,
                    binning,
                )?);
        }
    }

    let left_count = stump.split.left_stats.row_count as f32;
    let right_count = stump.split.right_stats.row_count as f32;
    let total_count = left_count + right_count;
    let left_probability = if total_count > 0.0 {
        left_count / total_count
    } else {
        0.5
    };
    let right_probability = 1.0 - left_probability;

    let left_expected = left_const
        + expected_subtree(
            tree_id,
            left_child_local,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;
    let right_expected = right_const
        + expected_subtree(
            tree_id,
            right_child_local,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;

    Ok(left_probability * left_expected + right_probability * right_expected)
}

pub(crate) fn shapley_values_for_row_f64(
    model: &TrainedModel,
    _row: &[f32],
    values_by_subset: &[f32],
    model_structure: &ModelStructure<'_>,
    _row_index: usize,
) -> ShapResult<Vec<f64>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = values_by_subset.len();

    let mut contributions = vec![0.0_f64; model.feature_count];
    if split_feature_count == 0 {
        return Ok(contributions);
    }

    let factorials = factorial_table(split_feature_count);
    let total_factorial = factorials[split_feature_count];

    for (feature_bit_position, &feature_index) in model_structure.split_features.iter().enumerate()
    {
        let feature_bit = 1_u64 << feature_bit_position;
        let mut phi = 0.0_f64;

        for subset_mask in 0..subset_count {
            let subset_mask_u64 = subset_mask as u64;
            if (subset_mask_u64 & feature_bit) != 0 {
                continue;
            }

            let with_feature_mask = subset_mask_u64 | feature_bit;
            let subset_size = subset_mask_u64.count_ones() as usize;
            let weight = factorials[subset_size]
                * factorials[split_feature_count - subset_size - 1]
                / total_factorial;

            let marginal =
                values_by_subset[with_feature_mask as usize] - values_by_subset[subset_mask];
            phi += weight * marginal as f64;
        }

        contributions[feature_index] = phi;
    }

    Ok(contributions)
}

pub(crate) fn verify_additivity(
    model: &TrainedModel,
    row: &[f32],
    contributions: &[f32],
    row_index: usize,
    expected_value: f32,
    binning: Option<&BinningContext>,
) -> ShapResult<()> {
    // Compute the prediction by walking each tree once and summing the leaf
    // values along the row's path.  Mirrors `distribute_linear_terms_for_row`.
    //
    // **Tolerance policy.**  Additivity is checked against
    //   atol + rtol * |predicted|
    // rather than a fixed absolute bound.  This matches numpy `allclose`
    // semantics and means accumulated f32 round-off across large
    // explanation batches (e.g. `feature_importances()` over ~1000 rows
    // on California Housing with `n_estimators=200`) does not raise even
    // though the arithmetic is correct.
    //
    // **Linear leaves.**  As of v0.7.4, `leaf_model="linear"` artifacts
    // satisfy strict additivity end-to-end when called with a
    // `BinningContext` (the predictor-aligned path).  The fix combines
    // v0.7.3's float-threshold path walker with crediting
    // `Σⱼ wⱼ·(z_j(row_raw_j) − z_j(baseline_raw_j))` at every visited node
    // along the row's path — matching how `predict` accumulates
    // `leaf.eval_row(row)` at each visited node. See
    // `distribute_linear_terms_for_row` for the path walk and
    // `leaf_constant_part` for the constant-part flow through
    // `expected_subtree` / `build_std_tree`.
    //
    // **v0.8.0:** the `BinningContext::LinearRank` variant joins
    // `Linear`, `Quantile`, and `PreBinned` as a fully strict-additivity
    // context.  When a caller passes any `BinningContext` variant the
    // SHAP path walker matches the predictor's path exactly, so the
    // linear-leaf exemption MUST NOT trigger.
    //
    // When `binning=None`, the SHAP walker uses the legacy `<=` bin-index
    // comparison and may take a different path than the predictor, so
    // strict additivity is not guaranteed for linear leaves on that
    // legacy path.  The exemption is retained only in that case.
    let predicted = local_path_predict(model, row, binning);
    let reconstructed = expected_value + contributions.iter().sum::<f32>();
    if binning.is_none() && model_has_linear_leaves(model) {
        // Legacy path-walker — best-effort interventional explanation.
        // Predictor-aligned (BinningContext) callers (Linear, Quantile,
        // PreBinned, LinearRank) get the strict check.
        return Ok(());
    }
    let mut tolerance = additivity_tolerance(predicted);
    if model_has_linear_leaves(model) {
        let mut nodes_by_key: HashMap<u64, &TrainedStump> =
            HashMap::with_capacity(model.stumps.len());
        let mut tree_roots: Vec<u32> = Vec::new();
        for stump in &model.stumps {
            let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
            nodes_by_key.insert(tree_local_key(tree_id, local_id), stump);
            if local_id == 0 {
                tree_roots.push(tree_id);
            }
        }
        tree_roots.sort_unstable();
        tree_roots.dedup();

        let mut max_linear_deviation = 0.0_f64;
        for tree_id in tree_roots {
            let mut local_id = 0u32;
            while let Some(stump) = nodes_by_key.get(&tree_local_key(tree_id, local_id)) {
                let feat = stump.split.feature_index as usize;
                let feature_value = row.get(feat).copied().unwrap_or(f32::NAN);
                let goes_left = stump_goes_left(&stump.split, feature_value, binning);
                let leaf = if goes_left {
                    &stump.left_leaf_value
                } else {
                    &stump.right_leaf_value
                };
                if let alloygbm_core::LeafValue::Linear(ll) = leaf {
                    for (w, &feat) in ll.weights.iter().zip(ll.regressor_features.iter()) {
                        let feat_idx = feat as usize;
                        let xj = row.get(feat_idx).copied().unwrap_or(0.0) as f64;
                        max_linear_deviation += (w.abs() as f64) * xj.abs();
                    }
                }
                local_id = if goes_left {
                    local_id.saturating_mul(2).saturating_add(1)
                } else {
                    local_id.saturating_mul(2).saturating_add(2)
                };
            }
        }
        tolerance += (max_linear_deviation * (f32::EPSILON as f64)) as f32;
    }
    if (predicted - reconstructed).abs() > tolerance {
        return Err(ShapError::ContractViolation(format!(
            "row {row_index} additivity check failed: predicted={predicted}, reconstructed={reconstructed}, tolerance={tolerance} (atol={ADDITIVITY_ATOL}, rtol={ADDITIVITY_RTOL})"
        )));
    }
    Ok(())
}

/// Compute `predict(row)` by walking each tree along the row's actual path
/// and summing the leaf evaluations at each visited internal node.  Used
/// internally by `verify_additivity`.  This is the same path-walking logic as
/// `distribute_linear_terms_for_row`, but here it accumulates the *full* leaf
/// value (`eval_row`) rather than just the linear deviation.
pub(crate) fn local_path_predict(
    model: &TrainedModel,
    row: &[f32],
    binning: Option<&BinningContext>,
) -> f32 {
    let mut nodes_by_key: HashMap<u64, &TrainedStump> = HashMap::with_capacity(model.stumps.len());
    let mut tree_roots: Vec<u32> = Vec::new();
    for stump in &model.stumps {
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        nodes_by_key.insert(tree_local_key(tree_id, local_id), stump);
        if local_id == 0 {
            tree_roots.push(tree_id);
        }
    }
    tree_roots.sort_unstable();
    tree_roots.dedup();

    let mut prediction = model.baseline_prediction;
    for tree_id in tree_roots {
        let mut local_id = 0u32;
        while let Some(stump) = nodes_by_key.get(&tree_local_key(tree_id, local_id)) {
            let feat = stump.split.feature_index as usize;
            let feature_value = row.get(feat).copied().unwrap_or(f32::NAN);
            let goes_left = stump_goes_left(&stump.split, feature_value, binning);
            let leaf = if goes_left {
                &stump.left_leaf_value
            } else {
                &stump.right_leaf_value
            };
            prediction += leaf.eval_row(row);
            local_id = if goes_left {
                local_id.saturating_mul(2).saturating_add(1)
            } else {
                local_id.saturating_mul(2).saturating_add(2)
            };
        }
    }
    prediction
}

pub(crate) fn factorial_table(max_value: usize) -> Vec<f64> {
    let mut factorials = vec![1.0_f64; max_value + 1];
    for value in 1..=max_value {
        factorials[value] = factorials[value - 1] * value as f64;
    }
    factorials
}

pub(crate) fn tree_local_key(tree_id: u32, local_node_id: u32) -> u64 {
    ((tree_id as u64) << 32) | local_node_id as u64
}

fn left_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ShapError::ContractViolation("left child node id overflow".to_string()))
}

fn right_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| ShapError::ContractViolation("right child node id overflow".to_string()))
}

pub(crate) fn validate_rows(rows: &[Vec<f32>], feature_count: usize) -> ShapResult<()> {
    if feature_count == 0 {
        return Err(ShapError::InvalidInput(
            "model feature_count must be greater than 0".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(ShapError::InvalidInput("rows cannot be empty".to_string()));
    }

    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row.len()
            )));
        }
        for (feature_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} contains NaN/Inf. \
                     SHAP values require finite feature values. If your data \
                     contains missing values, impute them before calling shap_values()."
                )));
            }
        }
    }

    Ok(())
}

pub(crate) fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}
