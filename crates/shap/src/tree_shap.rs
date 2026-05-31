use std::collections::HashMap;

use alloygbm_engine::{TrainedModel, TrainedStump};

use crate::binning::BinningContext;
use crate::brute_force::{decode_tree_node_id, tree_local_key, validate_rows, verify_additivity};
use crate::error::{ShapError, ShapResult};
use crate::linear_leaf::{
    distribute_linear_terms_for_row, leaf_constant_part, model_has_linear_leaves,
    scale_model_by_tree_weight,
};
use crate::types::{ShapExplanationBatch, ShapInteractionBatch};

/// Standard tree node used by TreeSHAP. Converts from AlloyGBM's stump-based
/// representation where each stump carries left/right leaf values into a
/// conventional tree where leaf values represent total accumulated prediction.
#[derive(Debug, Clone)]
pub(crate) enum StdTreeNode {
    Leaf {
        value: f64,
        cover: f64,
    },
    Internal {
        feature_index: usize,
        threshold: f32,
        default_left: bool,
        is_categorical: bool,
        categorical_bitset: Option<Vec<u8>>,
        left: Box<StdTreeNode>,
        right: Box<StdTreeNode>,
    },
}

impl StdTreeNode {
    pub(crate) fn cover(&self) -> f64 {
        match self {
            Self::Leaf { cover, .. } => *cover,
            Self::Internal { left, right, .. } => left.cover() + right.cover(),
        }
    }

    /// Cover-weighted sum of leaf values. Divide by `cover()` to get E[f_tree(x)].
    pub(crate) fn cover_weighted_value_sum(&self) -> f64 {
        match self {
            Self::Leaf { value, cover } => value * cover,
            Self::Internal { left, right, .. } => {
                left.cover_weighted_value_sum() + right.cover_weighted_value_sum()
            }
        }
    }
}

/// One element in the TreeSHAP path tracking structure.
#[derive(Clone, Copy)]
struct PathElement {
    feature_index: usize,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// Build a standard tree from AlloyGBM's stump representation for a single
/// tree. Accumulated leaf values are pushed down so that each leaf's `value`
/// is the total prediction contribution for samples reaching that leaf.
///
/// For piecewise-linear leaves we accumulate only the "constant part"
/// (`intercept + Σ wj * μj` when `baseline` is `Some`).  The row-dependent
/// `wj * (xj - μj)` terms are credited back to per-feature SHAP values
/// outside the path-based machinery — see `distribute_linear_terms_for_row`.
pub(crate) fn build_std_tree(
    tree_id: u32,
    local_id: u32,
    accumulated_value: f64,
    parent_cover: f64,
    nodes: &HashMap<u64, &TrainedStump>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> StdTreeNode {
    let key = tree_local_key(tree_id, local_id);
    match nodes.get(&key) {
        None => StdTreeNode::Leaf {
            value: accumulated_value,
            cover: parent_cover,
        },
        Some(stump) => {
            let left_cover = stump.split.left_stats.row_count as f64;
            let right_cover = stump.split.right_stats.row_count as f64;
            // When a binning context is provided we bake the predictor-
            // matching float threshold directly into the StdTreeNode; the
            // TreeSHAP recursion (`ts_recurse`) consumes this field as the
            // decision boundary and now compares with `<` instead of `<=`
            // (see `goes_left_with_threshold`).  When `binning` is None
            // we fall back to the legacy bin-index encoding.
            let threshold = match binning {
                Some(ctx) if !stump.split.is_categorical => ctx.float_threshold(
                    stump.split.feature_index as usize,
                    stump.split.threshold_bin,
                ),
                _ => stump.split.threshold_bin as f32,
            };
            StdTreeNode::Internal {
                feature_index: stump.split.feature_index as usize,
                threshold,
                default_left: stump.split.default_left,
                is_categorical: stump.split.is_categorical,
                categorical_bitset: stump.split.categorical_bitset.clone(),
                left: Box::new(build_std_tree(
                    tree_id,
                    2 * local_id + 1,
                    accumulated_value + leaf_constant_part(&stump.left_leaf_value, baseline),
                    left_cover,
                    nodes,
                    baseline,
                    binning,
                )),
                right: Box::new(build_std_tree(
                    tree_id,
                    2 * local_id + 2,
                    accumulated_value + leaf_constant_part(&stump.right_leaf_value, baseline),
                    right_cover,
                    nodes,
                    baseline,
                    binning,
                )),
            }
        }
    }
}

/// Extend the unique path with a new feature (Algorithm 2, Lundberg et al.).
fn ts_extend_path(
    path: &mut [PathElement],
    depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: usize,
) {
    path[depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if depth == 0 { 1.0 } else { 0.0 },
    };
    for i in (0..depth).rev() {
        path[i + 1].pweight += one_fraction * path[i].pweight * (i + 1) as f64 / (depth + 1) as f64;
        path[i].pweight = zero_fraction * path[i].pweight * (depth - i) as f64 / (depth + 1) as f64;
    }
}

/// Remove a feature from the path and shift remaining elements
/// (Algorithm 3, Lundberg et al.).
///
/// **Critical**: the shift at the end moves only `feature_index`,
/// `zero_fraction`, and `one_fraction` — NOT `pweight`.  The unwind
/// loop above has already computed the correct post-unwind pweights
/// in place; shifting them would clobber those values with the
/// pweights of the elements being shifted down (whose pweights were
/// computed when the duplicate was still in the path, not after its
/// removal).
///
/// The reference Python implementation in slundberg/shap uses four
/// parallel arrays (`feature_indexes`, `zero_fractions`,
/// `one_fractions`, `pweights`) and only shifts the first three.
/// The original AlloyGBM port stored all four in a single
/// `PathElement` struct and shifted the entire struct, which broke
/// the TreeSHAP polynomial path for any tree where a feature
/// appeared more than once on a root-to-leaf path (Limitation 5,
/// closed in v0.7.5).
fn ts_unextend_path(path: &mut [PathElement], depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[depth].pweight;

    for i in (0..depth).rev() {
        if one_fraction.abs() > 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight =
                next_one_portion * (depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (depth - i) as f64 / (depth + 1) as f64;
        } else {
            path[i].pweight =
                path[i].pweight * (depth + 1) as f64 / (zero_fraction * (depth - i) as f64);
        }
    }

    // Shift feature_index / zero_fraction / one_fraction only.
    // pweights are NOT shifted — see the function comment above.
    for i in path_index..depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

/// Compute the SHAP weight for unwinding the feature at `path_index`
/// (Algorithm 4, Lundberg et al.).
fn ts_unwound_path_sum(path: &[PathElement], depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[depth].pweight;
    let mut total = 0.0_f64;

    for i in (0..depth).rev() {
        if one_fraction.abs() > 0.0 {
            let tmp = next_one_portion * (depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction * (depth - i) as f64 / (depth + 1) as f64;
        } else if zero_fraction.abs() > 0.0 {
            let ratio = (depth - i) as f64 / (depth + 1) as f64;
            total += path[i].pweight / (zero_fraction * ratio);
        }
    }

    total
}

/// Recursive TreeSHAP walk (Algorithm 1, Lundberg et al.).
///
/// At each node the incoming edge's feature is added to the path. At leaves
/// the path is unwound to attribute contributions. At internal nodes the path
/// is cloned for each child so that modifications are independent.
#[allow(clippy::too_many_arguments)]
fn ts_recurse(
    node: &StdTreeNode,
    row: &[f32],
    path: &mut Vec<PathElement>,
    depth: usize,
    phi: &mut [f64],
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: usize,
    // When true, the tree's `threshold` fields hold float thresholds and
    // the decision uses strict `<` (predictor-aligned).  When false the
    // legacy bin-index encoding is used with `<=`.
    use_float_compare: bool,
) {
    // Ensure the path vector has room for this depth.
    while path.len() <= depth {
        path.push(PathElement {
            feature_index: usize::MAX,
            zero_fraction: 0.0,
            one_fraction: 0.0,
            pweight: 0.0,
        });
    }

    ts_extend_path(path, depth, zero_fraction, one_fraction, feature_index);

    match node {
        StdTreeNode::Leaf { value, .. } => {
            // Unwind each feature to compute its contribution.
            for i in 1..=depth {
                let w = ts_unwound_path_sum(path, depth, i);
                let feat = path[i].feature_index;
                if feat < phi.len() {
                    phi[feat] += w * (path[i].one_fraction - path[i].zero_fraction) * value;
                }
            }
        }
        StdTreeNode::Internal {
            feature_index: node_feature,
            threshold,
            default_left,
            is_categorical,
            categorical_bitset,
            left,
            right,
        } => {
            let goes_left = row
                .get(*node_feature)
                .map(|v| {
                    if *is_categorical {
                        let cat_id = *v as u16;
                        categorical_bitset.as_ref().map_or(*default_left, |bs| {
                            let byte_idx = (cat_id / 8) as usize;
                            let bit_idx = (cat_id % 8) as usize;
                            byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
                        })
                    } else if use_float_compare {
                        // Predictor-aligned strict less-than against the
                        // float threshold baked in by `build_std_tree`.
                        *v < *threshold
                    } else {
                        // Legacy bin-index comparison.
                        *v <= *threshold
                    }
                })
                .unwrap_or(*default_left);
            let (hot, cold) = if goes_left {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            let node_cover = node.cover();
            let hot_zero = if node_cover > 0.0 {
                hot.cover() / node_cover
            } else {
                0.5
            };
            let cold_zero = if node_cover > 0.0 {
                cold.cover() / node_cover
            } else {
                0.5
            };

            // Check whether this split feature already appears in the path.
            let duplicate_index = path[1..=depth]
                .iter()
                .position(|e| e.feature_index == *node_feature)
                .map(|pos| pos + 1);

            // Clone the path for each child so modifications are independent.
            let mut hot_path = path[..=depth].to_vec();
            let mut cold_path = path[..=depth].to_vec();

            if let Some(dup_idx) = duplicate_index {
                // Duplicate feature: combine incoming fractions.
                let incoming_zero = hot_path[dup_idx].zero_fraction;
                let incoming_one = hot_path[dup_idx].one_fraction;
                ts_unextend_path(&mut hot_path, depth, dup_idx);
                ts_unextend_path(&mut cold_path, depth, dup_idx);
                let child_depth = depth - 1;

                ts_recurse(
                    hot,
                    row,
                    &mut hot_path,
                    child_depth + 1,
                    phi,
                    incoming_zero * hot_zero,
                    incoming_one,
                    *node_feature,
                    use_float_compare,
                );
                ts_recurse(
                    cold,
                    row,
                    &mut cold_path,
                    child_depth + 1,
                    phi,
                    incoming_zero * cold_zero,
                    0.0,
                    *node_feature,
                    use_float_compare,
                );
            } else {
                ts_recurse(
                    hot,
                    row,
                    &mut hot_path,
                    depth + 1,
                    phi,
                    hot_zero,
                    1.0,
                    *node_feature,
                    use_float_compare,
                );
                ts_recurse(
                    cold,
                    row,
                    &mut cold_path,
                    depth + 1,
                    phi,
                    cold_zero,
                    0.0,
                    *node_feature,
                    use_float_compare,
                );
            }
        }
    }
}

/// Compute SHAP values for a single row using pre-built standard trees.
fn tree_shap_row(
    trees: &[StdTreeNode],
    row: &[f32],
    feature_count: usize,
    use_float_compare: bool,
) -> Vec<f64> {
    let mut phi = vec![0.0_f64; feature_count];
    for tree in trees {
        let mut path = Vec::with_capacity(32);
        ts_recurse(
            tree,
            row,
            &mut path,
            0,
            &mut phi,
            1.0,
            1.0,
            usize::MAX,
            use_float_compare,
        );
    }
    phi
}

/// Conditioning mode for SHAP-interaction TreeSHAP variant (Lundberg Alg 2).
///
/// Mirrors the canonical reference (slundberg/shap `tree_shap_recursive`,
/// `condition` parameter): `On` ≙ `condition > 0`, `Off` ≙ `condition < 0`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConditioningMode {
    /// Force the conditioning feature to always be present in S.  At any
    /// split on the conditioning feature: walk both children, but the cold
    /// child receives `condition_fraction = 0` (early-returns); the hot
    /// child's condition_fraction is unchanged.
    On,
    /// Force the conditioning feature to never be in S.  At any split on
    /// the conditioning feature: walk both children, each scaled by its
    /// cover ratio (`hot_zero_fraction` / `cold_zero_fraction`).
    Off,
}

/// TreeSHAP recursion with conditioning support (Lundberg et al. 2020
/// Algorithm 2), faithfully ported from the canonical reference
/// (`shap/cext/tree_shap.h::tree_shap_recursive`).
///
/// The path is extended with the parent's edge UNLESS the parent's split
/// was the conditioning feature — in that case the conditioning feature
/// is "factored out" of the path (and the parent's recurse call already
/// decremented `depth` to compensate).
///
/// At a node whose split IS on the conditioning feature, both children are
/// recursed into but with adjusted `condition_fraction`s:
/// - `On` mode: hot child unchanged, cold child gets `condition_fraction = 0`
///   (so it early-returns).
/// - `Off` mode: hot child gets `condition_fraction * hot_zero_fraction`,
///   cold child gets `condition_fraction * cold_zero_fraction`.
#[allow(clippy::too_many_arguments)]
fn ts_recurse_conditioning(
    node: &StdTreeNode,
    row: &[f32],
    path: &mut Vec<PathElement>,
    depth: usize,
    phi: &mut [f64],
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: usize,
    use_float_compare: bool,
    conditioning_feature: usize,
    mode: ConditioningMode,
    condition_fraction: f64,
) {
    // Early return: a zero condition_fraction kills the entire subtree.
    if condition_fraction == 0.0 {
        return;
    }

    while path.len() <= depth {
        path.push(PathElement {
            feature_index: usize::MAX,
            zero_fraction: 0.0,
            one_fraction: 0.0,
            pweight: 0.0,
        });
    }

    // Skip extend_path if the parent's split was the conditioning feature
    // (we're "factoring out" that feature from the path).
    let skip_extend = parent_feature_index == conditioning_feature;
    if !skip_extend {
        ts_extend_path(
            path,
            depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );
    }

    match node {
        StdTreeNode::Leaf { value, .. } => {
            // unique_depth is the index of the last valid path entry.
            // When skip_extend, depth was decremented by the parent so the
            // last entry is at `depth` (NOT `depth - 1`).  When NOT skipped,
            // we just extended to slot `depth`, so the last valid index is
            // also `depth`.  Either way, iterate 1..=depth.
            let effective_depth = depth;
            for i in 1..=effective_depth {
                let w = ts_unwound_path_sum(path, effective_depth, i);
                let feat = path[i].feature_index;
                if feat < phi.len() {
                    phi[feat] += w
                        * (path[i].one_fraction - path[i].zero_fraction)
                        * value
                        * condition_fraction;
                }
            }
        }
        StdTreeNode::Internal {
            feature_index: node_feature,
            threshold,
            default_left,
            is_categorical,
            categorical_bitset,
            left,
            right,
        } => {
            let goes_left = row
                .get(*node_feature)
                .map(|v| {
                    if *is_categorical {
                        let cat_id = *v as u16;
                        categorical_bitset.as_ref().map_or(*default_left, |bs| {
                            let byte_idx = (cat_id / 8) as usize;
                            let bit_idx = (cat_id % 8) as usize;
                            byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
                        })
                    } else if use_float_compare {
                        *v < *threshold
                    } else {
                        *v <= *threshold
                    }
                })
                .unwrap_or(*default_left);
            let (hot, cold) = if goes_left {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            let node_cover = node.cover();
            let hot_zero = if node_cover > 0.0 {
                hot.cover() / node_cover
            } else {
                0.5
            };
            let cold_zero = if node_cover > 0.0 {
                cold.cover() / node_cover
            } else {
                0.5
            };

            // Duplicate-feature handling (BEFORE conditioning logic).
            // We use a signed counter so successive decrements can go below
            // zero — matching slundberg/shap's unsigned-underflow trick,
            // which produces a `child_depth` of 0 (empty leaf-scan) when
            // conditioning fires at the very first split of a tree.
            let mut unique_depth: i32 = depth as i32;
            let duplicate_index = path[1..=depth]
                .iter()
                .position(|e| e.feature_index == *node_feature)
                .map(|pos| pos + 1);
            let mut incoming_zero = 1.0_f64;
            let mut incoming_one = 1.0_f64;
            if let Some(dup_idx) = duplicate_index {
                incoming_zero = path[dup_idx].zero_fraction;
                incoming_one = path[dup_idx].one_fraction;
                ts_unextend_path(path, depth, dup_idx);
                unique_depth -= 1;
            }

            // Conditioning-fraction logic at the OUTGOING split.
            let mut hot_condition_fraction = condition_fraction;
            let mut cold_condition_fraction = condition_fraction;
            if *node_feature == conditioning_feature {
                match mode {
                    ConditioningMode::On => {
                        // ON: only walk hot (cold gets zero → early return).
                        cold_condition_fraction = 0.0;
                    }
                    ConditioningMode::Off => {
                        // OFF: walk both, scale each by its cover ratio.
                        hot_condition_fraction *= hot_zero;
                        cold_condition_fraction *= cold_zero;
                    }
                }
                // Compensate for the skipped extend at the children.
                unique_depth -= 1;
            }

            // Recurse into both children.  Clone the ENTIRE path buffer
            // (not just up to unique_depth) — the canonical reference
            // (slundberg/shap) preserves all filled entries via raw-pointer
            // arithmetic, even after decrements.
            let mut hot_path = path.clone();
            let mut cold_path = path.clone();

            let child_depth = (unique_depth + 1).max(0) as usize;

            ts_recurse_conditioning(
                hot,
                row,
                &mut hot_path,
                child_depth,
                phi,
                incoming_zero * hot_zero,
                incoming_one,
                *node_feature,
                use_float_compare,
                conditioning_feature,
                mode,
                hot_condition_fraction,
            );
            ts_recurse_conditioning(
                cold,
                row,
                &mut cold_path,
                child_depth,
                phi,
                incoming_zero * cold_zero,
                0.0,
                *node_feature,
                use_float_compare,
                conditioning_feature,
                mode,
                cold_condition_fraction,
            );
        }
    }
}

/// Compute pairwise SHAP interactions for a single row using pre-built
/// standard trees (Lundberg Algorithm 2).  For each feature j: run TreeSHAP
/// with j conditioned ON and OFF; the half-difference attributes the
/// off-diagonal `Φ_ij`.  The diagonal is filled from the row-marginal
/// invariant `Σ_j Φ_ij == φ_i`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn tree_shap_interactions_row(
    trees: &[StdTreeNode],
    row: &[f32],
    feature_count: usize,
    use_float_compare: bool,
) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0_f64; feature_count]; feature_count];

    // Standard per-feature SHAP for the diagonal.
    let phi = tree_shap_row(trees, row, feature_count, use_float_compare);

    for j in 0..feature_count {
        let mut phi_on = vec![0.0_f64; feature_count];
        let mut phi_off = vec![0.0_f64; feature_count];
        for tree in trees {
            let mut path_on = Vec::with_capacity(32);
            let mut path_off = Vec::with_capacity(32);
            ts_recurse_conditioning(
                tree,
                row,
                &mut path_on,
                0,
                &mut phi_on,
                1.0,
                1.0,
                usize::MAX,
                use_float_compare,
                j,
                ConditioningMode::On,
                1.0,
            );
            ts_recurse_conditioning(
                tree,
                row,
                &mut path_off,
                0,
                &mut phi_off,
                1.0,
                1.0,
                usize::MAX,
                use_float_compare,
                j,
                ConditioningMode::Off,
                1.0,
            );
        }
        for i in 0..feature_count {
            if i == j {
                continue;
            }
            matrix[i][j] = 0.5 * (phi_on[i] - phi_off[i]);
        }
    }

    // Diagonal from row-marginal invariant.
    for i in 0..feature_count {
        let off_diag: f64 = (0..feature_count)
            .filter(|&k| k != i)
            .map(|k| matrix[i][k])
            .sum();
        matrix[i][i] = phi[i] - off_diag;
    }

    matrix
}

/// Compute SHAP values for multiple rows using TreeSHAP.
pub(crate) fn explain_rows_tree_shap(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    validate_rows(rows, model.feature_count)?;

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
    let has_linear = model_has_linear_leaves(model);

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

        // E[f_tree(x)] = cover-weighted average leaf value (computed on the
        // constant-part tree).  For linear leaves, the row-dependent
        // deviations sum to 0 in expectation (Σ wj · E[Xj - μj] = 0), so the
        // expected_value is the same under either decomposition.
        let tree_cover = tree.cover();
        if tree_cover > 0.0 {
            expected_value_f64 += tree.cover_weighted_value_sum() / tree_cover;
        }

        std_trees.push(tree);
    }

    let expected_value = expected_value_f64 as f32;
    let use_float_compare = binning.is_some();

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let mut phi = tree_shap_row(&std_trees, row, model.feature_count, use_float_compare);
        if has_linear {
            distribute_linear_terms_for_row(model, row, baseline, binning, &mut phi);
        }
        let contributions: Vec<f32> = phi.iter().map(|v| *v as f32).collect();
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

pub(crate) fn explain_interactions_from_model(
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
            distribute_linear_terms_for_row(model, row, baseline, binning, &mut linear_phi);
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
