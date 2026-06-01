use alloygbm_core::{LeafValue, LinearLeaf};
use alloygbm_engine::{TrainedModel, TrainedStump};
use std::collections::HashMap;

use crate::binning::BinningContext;
use crate::brute_force::{decode_tree_node_id, stump_goes_left, tree_local_key};

/// Reduce a leaf value to the "constant part" used by path-based SHAP
/// machinery.
///
/// * `LeafValue::Scalar(v)` reduces to `v`.
/// * `LeafValue::Linear(ll)` reduces to `ll.intercept + Σ wj * μj` when a
///   global feature baseline is available, or to `ll.intercept` otherwise.
///
/// The complementary `linear_leaf_row_terms` returns the row-dependent
/// `wj * (xj - μj)` deviations that must be added back to `phi[j]` for
/// additivity.  Together the two pieces reconstruct
/// `leaf_value.eval_row(row)`.
pub(crate) fn leaf_constant_part(leaf: &LeafValue, baseline: Option<&[f32]>) -> f64 {
    match leaf {
        LeafValue::Scalar(v) => *v as f64,
        LeafValue::Linear(ll) => {
            let mut acc = ll.intercept as f64;
            if let Some(b) = baseline {
                for (w, &feat) in ll.weights.iter().zip(ll.regressor_features.iter()) {
                    if let Some(&mean) = b.get(feat as usize) {
                        acc += (*w as f64) * (mean as f64);
                    }
                }
            }
            acc
        }
    }
}

/// Distribute the row-dependent linear deviations of a leaf onto a `phi`
/// attribution buffer.  Adds `wj * (xj - μj)` to `phi[regressor_j]` for each
/// regressor in a linear leaf.  No-op for scalar leaves.
///
/// When `baseline` is `None`, the deviation degrades to `wj * xj`, which keeps
/// additivity (`Σ phi + expected_value == predict(x)`) but biases the
/// path-attribution baseline.  Callers should prefer running with a baseline
/// recorded at fit time for the cleanest decomposition.
fn linear_leaf_row_terms(leaf: &LeafValue, row: &[f32], baseline: Option<&[f32]>, phi: &mut [f64]) {
    let LeafValue::Linear(ll) = leaf else {
        return;
    };
    accumulate_linear_terms(ll, row, baseline, phi);
}

fn accumulate_linear_terms(
    ll: &LinearLeaf,
    row: &[f32],
    baseline: Option<&[f32]>,
    phi: &mut [f64],
) {
    for (w, &feat) in ll.weights.iter().zip(ll.regressor_features.iter()) {
        let feat_idx = feat as usize;
        if feat_idx >= phi.len() {
            continue;
        }
        let xj = row.get(feat_idx).copied().unwrap_or(0.0) as f64;
        let mean = baseline
            .and_then(|b| b.get(feat_idx).copied())
            .unwrap_or(0.0) as f64;
        phi[feat_idx] += (*w as f64) * (xj - mean);
    }
}

pub(crate) fn model_has_linear_leaves(model: &TrainedModel) -> bool {
    model.stumps.iter().any(|s| {
        matches!(s.left_leaf_value, LeafValue::Linear(_))
            || matches!(s.right_leaf_value, LeafValue::Linear(_))
    })
}

/// Fold `stump.tree_weight` into each stump's leaf values so downstream
/// SHAP attribution code can ignore per-tree weighting. Returns a clone
/// of `model` with leaves pre-scaled and `tree_weight = 1.0` on every
/// stump.
///
/// For `LeafValue::Scalar(v)` the scaled value is `tree_weight · v`.
/// For `LeafValue::Linear { intercept, weights, .. }` both `intercept`
/// and every entry of `weights` are scaled by `tree_weight`, since
/// `tree_weight · (intercept + Σ w · x) = (tree_weight · intercept) +
/// Σ (tree_weight · w) · x`. The regressor feature indices are
/// preserved.
pub(crate) fn scale_model_by_tree_weight(model: &TrainedModel) -> TrainedModel {
    let mut scaled = model.clone();
    for stump in scaled.stumps.iter_mut() {
        let w = stump.tree_weight;
        if (w - 1.0).abs() <= f32::EPSILON {
            continue;
        }
        stump.left_leaf_value = scale_leaf_value(&stump.left_leaf_value, w);
        stump.right_leaf_value = scale_leaf_value(&stump.right_leaf_value, w);
        stump.tree_weight = 1.0;
    }
    scaled
}

fn scale_leaf_value(leaf: &LeafValue, factor: f32) -> LeafValue {
    match leaf {
        LeafValue::Scalar(v) => LeafValue::Scalar(factor * *v),
        LeafValue::Linear(ll) => LeafValue::Linear(alloygbm_core::LinearLeaf {
            intercept: factor * ll.intercept,
            weights: ll.weights.iter().map(|w| factor * *w).collect(),
            regressor_features: ll.regressor_features.clone(),
        }),
    }
}

/// Walk each tree for `row` and credit `wj · (xj − μj)` for every linear leaf
/// the row visits along its path.
///
/// **This must visit every node on the row's path, not just the terminal**
/// — `predict(x)` and `local_path_predict` both accumulate
/// `leaf.eval_row(row)` at every visited node (the predictor loops as long
/// as `nodes_by_local_id.get(child)` returns a stump).  The brute-force
/// SHAP and TreeSHAP polynomial paths already handle the per-visited-node
/// **constant** contribution `intercept + Σⱼ wⱼ·μⱼ` through
/// `leaf_constant_part`.  The per-visited-node **deviation**
/// `Σⱼ wⱼ·(xⱼ − μⱼ)` is uncredited unless we add it here.
///
/// Crediting only the terminal leaf was the pre-v0.7.4 bug: for a row whose
/// path through a tree visits N internal nodes plus a terminal, the SHAP
/// reconstruction was missing N nodes' worth of `Σⱼ wⱼ·(xⱼ − μⱼ)`, scaling
/// with `n_estimators` and `max_depth` and producing additivity drifts on
/// the order of the predictions themselves.
///
/// Trees with only scalar leaves remain no-ops because
/// `linear_leaf_row_terms` does nothing for `LeafValue::Scalar`, so scalar-
/// leaf-only models pay no overhead for the broader walk.
pub(crate) fn distribute_linear_terms_for_row(
    model: &TrainedModel,
    row: &[f32],
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
    phi: &mut [f64],
) {
    // Build a (tree_id, local_id) → stump map once per row is overkill, but
    // SHAP is not on the hot path and rows count is typically modest.  The
    // node-key map is also built inside `build_model_structure` for the
    // brute-force pre-processing; rebuilding here keeps this helper usable
    // from the polynomial TreeSHAP path too.
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

    for tree_id in tree_roots {
        let mut local_id = 0u32;
        while let Some(stump) = nodes_by_key.get(&tree_local_key(tree_id, local_id)) {
            let feat = stump.split.feature_index as usize;
            let feature_value = row.get(feat).copied().unwrap_or(f32::NAN);
            let goes_left = stump_goes_left(&stump.split, feature_value, binning);
            let leaf_value = if goes_left {
                &stump.left_leaf_value
            } else {
                &stump.right_leaf_value
            };
            linear_leaf_row_terms(leaf_value, row, baseline, phi);
            local_id = if goes_left {
                local_id.saturating_mul(2).saturating_add(1)
            } else {
                local_id.saturating_mul(2).saturating_add(2)
            };
        }
    }
}
