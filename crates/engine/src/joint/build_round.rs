//! Per-round tree-building primitives for the joint multi-output trainer.
//!
//! Hosts the public single-round adapter [`build_joint_round`] (level-wise,
//! morph/neutralization disabled) and the two private growth-mode builders
//! [`build_joint_round_inner`] (level-wise) and [`build_joint_round_leafwise`]
//! (best-first). The outer training loop in [`super::fit_joint_inner`]
//! dispatches into one of the latter two each round.

use std::collections::HashMap;

use alloygbm_core::{
    BinnedMatrix, FactorExposureMatrix, GradientPair, LeafValue, MISSING_BIN_U8, NodeStats,
    SplitCandidate, TrainParams,
};

use crate::shared_histogram::{
    MultiOutputHistogram, build_multi_output_histogram_inplace, compute_multi_output_split_gain,
};
use crate::{InteractionConstraintIndex, TrainedStump};

use super::helpers::{
    accumulate_factor_sums_for_threshold, effective_dro_config, effective_neutralization_config,
    u64_to_bitset_bytes,
};
use super::types::{JointLeafCandidate, JointLeafNode, JointMorphContext, JointRoundResult};

/// Train a single round of joint multi-output boosting and return the new
/// stumps (already updated to carry K-output leaf values). The caller is
/// responsible for accumulating leaf contributions into per-output prediction
/// vectors using `predictions[k][row] += learning_rate * leaf_value_k`.
///
/// `round_index` is mixed into the `col_subsample` RNG seed so each tree
/// samples a different feature subset (matches LightGBM `feature_fraction`
/// semantics and the single-output trainer's per-round behavior).
///
/// `sampled_root_rows` is the optional row subset to use as the tree's root
/// (used by row_subsample / bagging_fraction). When `None`, the root contains
/// all `n_rows`. Filtering rows at the root means `min_data_in_leaf` operates
/// on the sampled set, matching the single-output trainer (v0.10.2.1 fix).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub fn build_joint_round(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
) -> Result<JointRoundResult, String> {
    build_joint_round_inner(
        params,
        binned_matrix,
        grads_per_output,
        n_outputs,
        categorical_features,
        round_index,
        sampled_root_rows,
        /*morph_ctx=*/ None,
        /*factor_exposures=*/ None,
    )
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_joint_round_inner(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
    morph_ctx: Option<&JointMorphContext>,
    factor_exposures: Option<&FactorExposureMatrix>,
) -> Result<JointRoundResult, String> {
    if grads_per_output.len() != n_outputs {
        return Err(format!(
            "expected {n_outputs} gradient vectors, got {}",
            grads_per_output.len()
        ));
    }
    let n_rows = binned_matrix.row_count;
    for (k, g) in grads_per_output.iter().enumerate() {
        if g.len() != n_rows {
            return Err(format!(
                "gradient vector for output {k} has length {} != {n_rows}",
                g.len()
            ));
        }
    }

    let feature_count = binned_matrix.feature_count;
    // BinnedMatrix exposes a single global `max_bin`; allocate enough bin slots
    // to cover any feature plus the dedicated NaN/missing bin (always 255 for
    // u8 storage in v0.10.0).
    let n_bins = (binned_matrix.max_bin as usize + 1).max(MISSING_BIN_U8 as usize + 1);

    // Pack gradients into row-major K-output flat arrays for histogram build.
    let mut packed_grads = vec![0.0_f32; n_rows * n_outputs];
    let mut packed_hess = vec![0.0_f32; n_rows * n_outputs];
    for k in 0..n_outputs {
        for row in 0..n_rows {
            let gp = grads_per_output[k][row];
            packed_grads[row * n_outputs + k] = gp.grad;
            packed_hess[row * n_outputs + k] = gp.hess;
        }
    }

    let max_depth = params.max_depth.max(1) as usize;
    let min_rows_per_leaf = params.min_data_in_leaf.max(1) as usize;
    let lambda_l2 = params.lambda_l2;

    // v0.10.6: split_penalty neutralization — when active, every candidate's
    // gain is adjusted by a K-output factor-load penalty derived from the
    // candidate's L/R leaf K-vectors and per-side factor sums. Inert configs
    // (kind != SplitPenalty, or split_penalty == 0) leave the gain unchanged.
    // Note: this path is opt-in; the no-op cost (one Option check per
    // candidate) is negligible. The PyO3-only caller (`train_joint_*` from
    // `bindings/python`) doesn't currently provide `factor_exposures`; future
    // wiring in Task 11 will route them through.
    // PR #39 review (R2): when split_penalty is configured but exposures
    // weren't provided, return an explicit error instead of silently treating
    // it as inert. Mirrors the pre_target / per_round_gradient guards in
    // `fit_joint_inner`.
    let split_penalty_ctx: Option<(f32, &alloygbm_core::FactorExposureMatrix)> =
        match (effective_neutralization_config(params), factor_exposures) {
            (Some(cfg), Some(exposures))
                if cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty
                    && cfg.split_penalty > 0.0 =>
            {
                Some((cfg.split_penalty, exposures))
            }
            (Some(cfg), None)
                if cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty
                    && cfg.split_penalty > 0.0 =>
            {
                return Err(
                    "factor_exposures are required when neutralization='split_penalty'".to_string(),
                );
            }
            _ => None,
        };

    // col_subsample (v0.10.2): per-tree feature mask, seeded by
    // `(params.seed, round_index)`. v0.10.2.1 fix: mixing the round index
    // into the seed makes each tree sample a different feature subset, matching
    // LightGBM `feature_fraction` and the single-output trainer's per-round
    // behavior. Without this, every tree saw the same feature subset which
    // defeats the point of column sampling.
    let feature_allowed: Vec<bool> = if params.col_subsample < 1.0 {
        let mut s: u64 = params.seed.wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((round_index as u64).wrapping_mul(0x94D049BB133111EB));
        s ^= s >> 30;
        s = s.wrapping_mul(0x94D049BB133111EB);
        if s == 0 {
            s = 0xDEADBEEFCAFEBABE;
        }
        let rate = params.col_subsample;
        let mut mask: Vec<bool> = (0..feature_count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u01 = ((s >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                u01 < rate
            })
            .collect();
        if !mask.iter().any(|&b| b) {
            // All-zero edge case: fall back to all-allowed.
            for f in mask.iter_mut() {
                *f = true;
            }
        }
        mask
    } else {
        vec![true; feature_count]
    };

    // interaction_constraints (v0.10.2): build the constraint index (returns
    // None if the user didn't configure any constraints). Per-node bookkeeping
    // tracks the active group bitset for each local_node_id; the root starts
    // with all groups active, and descendants narrow the set via `descend`
    // each time their parent splits on a constrained feature.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        feature_count,
    )
    .map_err(|e| format!("interaction_constraints: {e:?}"))?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }

    // Native-categorical lookup (v0.10.2): feature_index → num_categories.
    // Built once per call. `None` means the feature is numeric.
    let cat_num_categories: Vec<Option<usize>> = {
        let mut v = vec![None; feature_count];
        for cf in categorical_features {
            if cf.feature_index < feature_count {
                v[cf.feature_index] = Some(cf.num_categories);
            }
        }
        v
    };

    let mut stumps: Vec<TrainedStump> = Vec::new();
    let root_rows: Vec<u32> = match sampled_root_rows {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows as u32).collect(),
    };
    let mut active: Vec<JointLeafNode> = vec![JointLeafNode {
        local_node_id: 0,
        row_indices: root_rows,
    }];

    for _depth in 0..max_depth {
        if active.is_empty() {
            break;
        }
        let mut next_active: Vec<JointLeafNode> = Vec::new();

        for node in active.drain(..) {
            if node.row_indices.len() < 2 * min_rows_per_leaf {
                // Too few rows to attempt a split — emit a terminal leaf later
                // via parent's leaf assignment; nothing to do here for an
                // already-leaf node.
                continue;
            }

            // Build per-feature K-output histogram from scratch for this node.
            let mut node_hist = MultiOutputHistogram::new(feature_count, n_bins, n_outputs);
            // Slice bin/grad/hess for this node's rows; build kernel reads rows
            // sequentially, so we set up a temporary local view.
            // (For a minimal correct implementation, build histograms directly
            // from the row_indices set.)
            //
            // We accumulate per-feature column.
            for feature in 0..feature_count {
                // col_subsample (v0.10.2): skip histogram build for masked-out features.
                if !feature_allowed[feature] {
                    continue;
                }
                // Subset the bin column for this feature.
                let mut subset_bins: Vec<u8> = Vec::with_capacity(node.row_indices.len());
                for &row in &node.row_indices {
                    // Row-major: bins[row * feature_count + feature].
                    let idx = row as usize * feature_count + feature;
                    subset_bins.push(binned_matrix.bins[idx]);
                }
                // Subset packed_grads/hess for these rows.
                let mut subset_g: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                let mut subset_h: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                for &row in &node.row_indices {
                    for k in 0..n_outputs {
                        subset_g.push(packed_grads[row as usize * n_outputs + k]);
                        subset_h.push(packed_hess[row as usize * n_outputs + k]);
                    }
                }
                build_multi_output_histogram_inplace(
                    &mut node_hist,
                    feature,
                    &subset_bins,
                    &subset_g,
                    &subset_h,
                    n_outputs,
                );
            }

            // Find best split across all (feature, threshold_bin) pairs.
            // For categorical features, the threshold_bin slot is unused
            // (set to 0) and the bitset is carried in the 4th element.
            let mut best: Option<(usize, usize, f32, Option<u64>)> = None;
            // BinnedMatrix exposes max_bin globally; iterate candidate
            // thresholds across the full bin range minus the NaN slot.
            let max_threshold = (binned_matrix.max_bin as usize).min(MISSING_BIN_U8 as usize - 1);
            // interaction_constraints (v0.10.2): look up this node's active
            // group set once outside the feature loop.
            let node_ag = node_active_groups.get(&node.local_node_id).copied();
            for feature in 0..feature_count {
                // col_subsample (v0.10.2): skip masked-out features in split search.
                if !feature_allowed[feature] {
                    continue;
                }
                // interaction_constraints (v0.10.2): skip features outside the
                // active group set for this node.
                if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), node_ag)
                    && !idx.feature_allowed(ag, feature as u32)
                {
                    continue;
                }
                // Native-categorical (v0.10.2): if the feature is in the
                // categorical_features list, use Fisher-sort over the K
                // outputs instead of a threshold sweep. The result carries
                // a `left_bitset: u64` which the partition path consumes.
                if let Some(num_cats) = cat_num_categories[feature] {
                    let cat_factor_penalty =
                        split_penalty_ctx.map(|(factor_penalty, exposures)| {
                            crate::shared_histogram::MultiOutputCategoricalFactorPenaltyContext {
                                binned_matrix,
                                exposures,
                                row_indices: &node.row_indices,
                                factor_penalty,
                                lambda_l1: params.lambda_l1,
                                dro_config: effective_dro_config(params),
                            }
                        });
                    // v0.10.4: route categorical Fisher-sort through the morph
                    // variant when active; falls through to the standard
                    // variant when `morph_ctx` is None.
                    let cat_opt = if let Some(m) = morph_ctx {
                        crate::shared_histogram::find_best_multi_output_categorical_split_morph_with_factor_penalty(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                            cat_factor_penalty,
                        )
                    } else {
                        crate::shared_histogram::find_best_multi_output_categorical_split_with_factor_penalty(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            cat_factor_penalty,
                        )
                    };
                    if let Some(cat_split) = cat_opt
                        && cat_split.gain > best.map(|(_, _, g, _)| g).unwrap_or(0.0)
                    {
                        best = Some((feature, 0, cat_split.gain, Some(cat_split.left_bitset)));
                    }
                    continue; // skip numeric threshold sweep for categorical features
                }
                for threshold_bin in 0..max_threshold {
                    // v0.10.4: route numeric gain through the morph variant when
                    // active; falls through to the standard variant otherwise.
                    let base_gain = if let Some(m) = morph_ctx {
                        crate::shared_histogram::compute_multi_output_split_gain_morph(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                        )
                    } else {
                        compute_multi_output_split_gain(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                        )
                    };
                    // v0.10.6 split_penalty (numeric): subtract per-candidate
                    // K-output factor-load penalty.
                    let gain = if let Some((factor_penalty, exposures)) = split_penalty_ctx {
                        let (leaf_l, leaf_r) =
                            crate::shared_histogram::derive_kvec_leaves_from_threshold_histogram(
                                &node_hist,
                                feature,
                                threshold_bin,
                                lambda_l2,
                                crate::LEAF_EPSILON,
                                params.lambda_l1,
                                effective_dro_config(params),
                            );
                        let (left_sums, right_sums) = accumulate_factor_sums_for_threshold(
                            binned_matrix,
                            exposures,
                            &node.row_indices,
                            feature,
                            threshold_bin as u8,
                            None,
                        );
                        let penalty =
                            crate::shared_histogram::compute_multi_output_factor_split_penalty(
                                &left_sums,
                                &right_sums,
                                &leaf_l,
                                &leaf_r,
                                factor_penalty,
                                node.row_indices.len(),
                            );
                        base_gain - penalty
                    } else {
                        base_gain
                    };
                    if gain <= 0.0 {
                        continue;
                    }
                    if best.map(|(_, _, g, _)| gain > g).unwrap_or(true) {
                        best = Some((feature, threshold_bin, gain, None));
                    }
                }
            }

            let Some((feature, threshold_bin, gain, cat_bitset)) = best else {
                continue; // No positive-gain split — leave node as terminal leaf
            };

            // min_split_gain (v0.10.2): reject splits whose K-output sum-of-gains
            // falls below the user-specified threshold. Mirrors the single-output
            // trainer.
            if gain < params.min_split_gain {
                continue;
            }

            // Partition rows by the chosen split. For categorical splits the
            // bin index is interpreted as a category ID and routed via the
            // bitset; for numeric splits, by threshold_bin. NaN rows
            // (bin == MISSING_BIN_U8) route per default_left below.
            let mut left_rows: Vec<u32> = Vec::new();
            let mut right_rows: Vec<u32> = Vec::new();
            let mut missing_rows: Vec<u32> = Vec::new();
            for &row in &node.row_indices {
                let bin = binned_matrix.bins[row as usize * feature_count + feature];
                if bin == MISSING_BIN_U8 {
                    missing_rows.push(row);
                } else if let Some(bs) = cat_bitset {
                    // Categorical: bit `bin` set → left.
                    if bin < 64 && (bs & (1u64 << bin)) != 0 {
                        left_rows.push(row);
                    } else {
                        right_rows.push(row);
                    }
                } else if (bin as usize) <= threshold_bin {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }
            // v0.10.0 default-direction policy: route missing rows to whichever
            // side currently has more rows (a deterministic, simple heuristic).
            // A smarter learned-direction policy is a v0.10.x follow-up.
            let default_left = left_rows.len() >= right_rows.len();
            if default_left {
                left_rows.append(&mut missing_rows);
            } else {
                right_rows.append(&mut missing_rows);
            }

            if left_rows.len() < min_rows_per_leaf || right_rows.len() < min_rows_per_leaf {
                continue; // Skip this split — would create an under-sized leaf
            }

            // Compute K-output leaf values via Newton-Raphson per output.
            // v0.10.5: route through `leaf_effective_gradient` so L1 and DRO
            // leaf solvers are honored. When `lambda_l1 == 0` AND
            // `dro_config == None`, this returns `g_sum` unchanged — so the
            // v0.10.0–v0.10.4 behavior is preserved byte-for-byte for the
            // default config.
            let leaf_values = |rows: &[u32]| -> Vec<f32> {
                let mut out = vec![0.0_f32; n_outputs];
                let row_count = rows.len() as u32;
                for k in 0..n_outputs {
                    let mut g_sum = 0.0_f32;
                    let mut h_sum = 0.0_f32;
                    let mut g_sq_sum = 0.0_f32;
                    for &row in rows {
                        let gp = grads_per_output[k][row as usize];
                        g_sum += gp.grad;
                        h_sum += gp.hess;
                        g_sq_sum += gp.grad * gp.grad;
                    }
                    let g_eff = alloygbm_core::leaf_effective_gradient(
                        g_sum,
                        g_sq_sum,
                        row_count,
                        params.lambda_l1,
                        effective_dro_config(params),
                    );
                    out[k] = -g_eff / (h_sum + lambda_l2 + crate::LEAF_EPSILON);
                }
                out
            };
            let left_k = leaf_values(&left_rows);
            let right_k = leaf_values(&right_rows);

            // Compute per-side NodeStats summed across outputs (placeholder for
            // SplitCandidate; the joint trainer doesn't consume these stats
            // beyond record-keeping).
            let summarize = |rows: &[u32]| -> NodeStats {
                let mut g = 0.0_f32;
                let mut h = 0.0_f32;
                for grad_vec in grads_per_output.iter().take(n_outputs) {
                    for &row in rows {
                        let gp = grad_vec[row as usize];
                        g += gp.grad;
                        h += gp.hess;
                    }
                }
                NodeStats {
                    grad_sum: g,
                    hess_sum: h,
                    grad_sq_sum: 0.0,
                    row_count: rows.len() as u32,
                }
            };
            let left_stats = summarize(&left_rows);
            let right_stats = summarize(&right_rows);

            let stump = TrainedStump {
                split: SplitCandidate {
                    node_id: node.local_node_id,
                    feature_index: feature as u32,
                    threshold_bin: threshold_bin as u16,
                    gain,
                    default_left,
                    is_categorical: cat_bitset.is_some(),
                    categorical_bitset: cat_bitset.map(u64_to_bitset_bytes),
                    left_stats,
                    right_stats,
                },
                // Placeholder scalar leaves (the K-output values below are
                // authoritative). Using the first output as a representative
                // means single-output prediction paths still return a sensible
                // value if accidentally invoked.
                left_leaf_value: LeafValue::Scalar(left_k[0]),
                right_leaf_value: LeafValue::Scalar(right_k[0]),
                tree_weight: 1.0,
                multi_output_leaf_values: Some((left_k, right_k)),
            };
            stumps.push(stump);

            // interaction_constraints (v0.10.2): propagate the active group
            // set to both children. `descend` returns the intersection of
            // the parent's active groups with the groups containing the
            // split feature (or leaves it unchanged for unconstrained
            // features).
            if let (Some(idx), Some(parent_ag)) = (constraint_index.as_ref(), node_ag) {
                let child_ag = idx.descend(parent_ag, feature as u32);
                node_active_groups.insert(node.local_node_id * 2 + 1, child_ag);
                node_active_groups.insert(node.local_node_id * 2 + 2, child_ag);
            }

            // Schedule child nodes (local_node_id * 2 + 1 and + 2 per predictor
            // traversal convention).
            next_active.push(JointLeafNode {
                local_node_id: node.local_node_id * 2 + 1,
                row_indices: left_rows,
            });
            next_active.push(JointLeafNode {
                local_node_id: node.local_node_id * 2 + 2,
                row_indices: right_rows,
            });
        }

        active = next_active;
    }

    Ok(JointRoundResult { stumps })
}

/// Build one joint round using **leaf-wise (best-first)** tree growth.
///
/// Mirrors `build_joint_round` (level-wise) but pops the best candidate
/// from a max-heap keyed by gain at each step, instead of expanding every
/// node at the current depth. Stops when:
///   - the heap is empty (no positive-gain split available anywhere), OR
///   - the leaf count would exceed `max_leaves`, OR
///   - a candidate's depth (derived from `local_node_id`) would exceed
///     `params.max_depth`.
///
/// Honors `col_subsample`, `interaction_constraints`, and `min_split_gain`
/// using the same logic as the level-wise builder. `row_subsample` is
/// applied at the outer round level via `fit_joint_multi_output` (the
/// gradients passed in are already row-masked when sampling is enabled).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_joint_round_leafwise(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    max_leaves: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
    morph_ctx: Option<&JointMorphContext>,
    factor_exposures: Option<&FactorExposureMatrix>,
) -> Result<JointRoundResult, String> {
    use std::collections::BinaryHeap;

    let n_rows = binned_matrix.row_count;
    let feature_count = binned_matrix.feature_count;
    let n_bins = (binned_matrix.max_bin as usize + 1).max(MISSING_BIN_U8 as usize + 1);

    // Per-output gradient sanity (mirror build_joint_round).
    for (k, g) in grads_per_output.iter().enumerate() {
        if g.len() != n_rows {
            return Err(format!(
                "gradient vector for output {k} has length {} != {n_rows}",
                g.len()
            ));
        }
    }

    // Pack gradients for shared-histogram build (row-major K-output).
    let mut packed_grads = vec![0.0_f32; n_rows * n_outputs];
    let mut packed_hess = vec![0.0_f32; n_rows * n_outputs];
    for k in 0..n_outputs {
        for row in 0..n_rows {
            let gp = grads_per_output[k][row];
            packed_grads[row * n_outputs + k] = gp.grad;
            packed_hess[row * n_outputs + k] = gp.hess;
        }
    }

    let max_depth = params.max_depth.max(1) as usize;
    let min_rows_per_leaf = params.min_data_in_leaf.max(1) as usize;
    let lambda_l2 = params.lambda_l2;

    // col_subsample (same logic as build_joint_round; v0.10.2.1 fix
    // mixes round_index into the seed so each tree samples a different
    // feature subset).
    let feature_allowed: Vec<bool> = if params.col_subsample < 1.0 {
        let mut s: u64 = params.seed.wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((round_index as u64).wrapping_mul(0x94D049BB133111EB));
        s ^= s >> 30;
        s = s.wrapping_mul(0x94D049BB133111EB);
        if s == 0 {
            s = 0xDEADBEEFCAFEBABE;
        }
        let rate = params.col_subsample;
        let mut mask: Vec<bool> = (0..feature_count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u01 = ((s >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                u01 < rate
            })
            .collect();
        if !mask.iter().any(|&b| b) {
            for f in mask.iter_mut() {
                *f = true;
            }
        }
        mask
    } else {
        vec![true; feature_count]
    };

    // interaction_constraints.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        feature_count,
    )
    .map_err(|e| format!("interaction_constraints: {e:?}"))?;
    let root_active_groups = constraint_index
        .as_ref()
        .map(|idx| idx.root_active_groups());

    // Native-categorical lookup (v0.10.2): same as build_joint_round.
    let cat_num_categories: Vec<Option<usize>> = {
        let mut v = vec![None; feature_count];
        for cf in categorical_features {
            if cf.feature_index < feature_count {
                v[cf.feature_index] = Some(cf.num_categories);
            }
        }
        v
    };

    // v0.10.6: split_penalty neutralization — same gate as the level-wise
    // builder (`build_joint_round_inner`). The heap must rank candidates by
    // PENALIZED gain, so the adjustment is applied inside `evaluate_node`
    // before the candidate is pushed.
    //
    // PR #40 review (R1): mirror the level-wise explicit-error gate for
    // `SplitPenalty + missing exposures` so a Rust caller calling the joint
    // entry point directly with `tree_growth=Leaf` can't accidentally train
    // an unpenalized model that still advertises split_penalty in its
    // `NeutralizationMetadata` artifact section.
    let split_penalty_ctx: Option<(f32, &FactorExposureMatrix)> =
        match (effective_neutralization_config(params), factor_exposures) {
            (Some(cfg), Some(exposures))
                if cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty
                    && cfg.split_penalty > 0.0 =>
            {
                Some((cfg.split_penalty, exposures))
            }
            (Some(cfg), None)
                if cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty
                    && cfg.split_penalty > 0.0 =>
            {
                return Err(
                    "factor_exposures are required when neutralization='split_penalty'".to_string(),
                );
            }
            _ => None,
        };

    // Per-node candidate evaluator. Builds the multi-output histogram for
    // `node.row_indices`, sweeps features (respecting `feature_allowed` and
    // the active interaction-constraint group set), picks the best split,
    // partitions rows, computes Newton-Raphson K-output leaf values, and
    // returns a candidate (or None if no positive-gain split survives the
    // constraints + min_data_in_leaf + min_split_gain filters).
    let evaluate_node = |node: JointLeafNode,
                         active_groups: Option<u64>|
     -> Option<JointLeafCandidate> {
        if node.row_indices.len() < 2 * min_rows_per_leaf {
            return None;
        }

        // Build multi-output histogram for this node.
        let mut node_hist = MultiOutputHistogram::new(feature_count, n_bins, n_outputs);
        for feature in 0..feature_count {
            if !feature_allowed[feature] {
                continue;
            }
            if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), active_groups)
                && !idx.feature_allowed(ag, feature as u32)
            {
                continue;
            }
            let mut subset_bins: Vec<u8> = Vec::with_capacity(node.row_indices.len());
            for &row in &node.row_indices {
                let idx = row as usize * feature_count + feature;
                subset_bins.push(binned_matrix.bins[idx]);
            }
            let mut subset_g: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
            let mut subset_h: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
            for &row in &node.row_indices {
                for k in 0..n_outputs {
                    subset_g.push(packed_grads[row as usize * n_outputs + k]);
                    subset_h.push(packed_hess[row as usize * n_outputs + k]);
                }
            }
            build_multi_output_histogram_inplace(
                &mut node_hist,
                feature,
                &subset_bins,
                &subset_g,
                &subset_h,
                n_outputs,
            );
        }

        // Sweep features for the best split. Categorical features dispatch
        // to Fisher-sort and carry a u64 bitset in the 4th slot of `best`.
        let mut best: Option<(usize, usize, f32, Option<u64>)> = None;
        let max_threshold = (binned_matrix.max_bin as usize).min(MISSING_BIN_U8 as usize - 1);
        for feature in 0..feature_count {
            if !feature_allowed[feature] {
                continue;
            }
            if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), active_groups)
                && !idx.feature_allowed(ag, feature as u32)
            {
                continue;
            }
            if let Some(num_cats) = cat_num_categories[feature] {
                let cat_factor_penalty = split_penalty_ctx.map(|(factor_penalty, exposures)| {
                    crate::shared_histogram::MultiOutputCategoricalFactorPenaltyContext {
                        binned_matrix,
                        exposures,
                        row_indices: &node.row_indices,
                        factor_penalty,
                        lambda_l1: params.lambda_l1,
                        dro_config: effective_dro_config(params),
                    }
                });
                // v0.10.4: route through morph variant when active.
                let cat_opt = if let Some(m) = morph_ctx {
                    crate::shared_histogram::find_best_multi_output_categorical_split_morph_with_factor_penalty(
                        &node_hist,
                        feature,
                        num_cats,
                        lambda_l2,
                        crate::LEAF_EPSILON,
                        &m.config,
                        &m.precomputed,
                        m.iteration,
                        m.total_iterations,
                        &m.grad_means,
                        &m.grad_stds,
                        cat_factor_penalty,
                    )
                } else {
                    crate::shared_histogram::find_best_multi_output_categorical_split_with_factor_penalty(
                        &node_hist,
                        feature,
                        num_cats,
                        lambda_l2,
                        crate::LEAF_EPSILON,
                        cat_factor_penalty,
                    )
                };
                if let Some(cat_split) = cat_opt
                    && cat_split.gain > best.map(|(_, _, g, _)| g).unwrap_or(0.0)
                {
                    best = Some((feature, 0, cat_split.gain, Some(cat_split.left_bitset)));
                }
                continue;
            }
            for threshold_bin in 0..max_threshold {
                // v0.10.4: route through morph variant when active.
                let base_gain = if let Some(m) = morph_ctx {
                    crate::shared_histogram::compute_multi_output_split_gain_morph(
                        &node_hist,
                        feature,
                        threshold_bin,
                        lambda_l2,
                        crate::LEAF_EPSILON,
                        &m.config,
                        &m.precomputed,
                        m.iteration,
                        m.total_iterations,
                        &m.grad_means,
                        &m.grad_stds,
                    )
                } else {
                    compute_multi_output_split_gain(
                        &node_hist,
                        feature,
                        threshold_bin,
                        lambda_l2,
                        crate::LEAF_EPSILON,
                    )
                };
                // v0.10.6 split_penalty (numeric, leaf-wise).
                let gain = if let Some((factor_penalty, exposures)) = split_penalty_ctx {
                    let (leaf_l, leaf_r) =
                        crate::shared_histogram::derive_kvec_leaves_from_threshold_histogram(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            params.lambda_l1,
                            effective_dro_config(params),
                        );
                    let (left_sums, right_sums) = accumulate_factor_sums_for_threshold(
                        binned_matrix,
                        exposures,
                        &node.row_indices,
                        feature,
                        threshold_bin as u8,
                        None,
                    );
                    let penalty =
                        crate::shared_histogram::compute_multi_output_factor_split_penalty(
                            &left_sums,
                            &right_sums,
                            &leaf_l,
                            &leaf_r,
                            factor_penalty,
                            node.row_indices.len(),
                        );
                    base_gain - penalty
                } else {
                    base_gain
                };
                if gain <= 0.0 {
                    continue;
                }
                if best.map(|(_, _, g, _)| gain > g).unwrap_or(true) {
                    best = Some((feature, threshold_bin, gain, None));
                }
            }
        }
        let (feature, threshold_bin, gain, cat_bitset) = best?;
        if gain < params.min_split_gain {
            return None;
        }

        // Partition rows.
        let mut left_rows: Vec<u32> = Vec::new();
        let mut right_rows: Vec<u32> = Vec::new();
        let mut missing_rows: Vec<u32> = Vec::new();
        for &row in &node.row_indices {
            let bin = binned_matrix.bins[row as usize * feature_count + feature];
            if bin == MISSING_BIN_U8 {
                missing_rows.push(row);
            } else if let Some(bs) = cat_bitset {
                if bin < 64 && (bs & (1u64 << bin)) != 0 {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            } else if (bin as usize) <= threshold_bin {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        let default_left = left_rows.len() >= right_rows.len();
        if default_left {
            left_rows.append(&mut missing_rows);
        } else {
            right_rows.append(&mut missing_rows);
        }
        if left_rows.len() < min_rows_per_leaf || right_rows.len() < min_rows_per_leaf {
            return None;
        }

        // K-output leaf values via Newton-Raphson per output.
        // v0.10.5: route through `leaf_effective_gradient` so L1 and DRO
        // leaf solvers are honored on the leaf-wise path too.  When
        // `lambda_l1 == 0` AND `dro_config == None`, this returns
        // `g_sum` unchanged — preserving byte-for-byte compatibility
        // with the v0.10.0–v0.10.4 default config.
        let leaf_values = |rows: &[u32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; n_outputs];
            let row_count = rows.len() as u32;
            for k in 0..n_outputs {
                let mut g_sum = 0.0_f32;
                let mut h_sum = 0.0_f32;
                let mut g_sq_sum = 0.0_f32;
                for &row in rows {
                    let gp = grads_per_output[k][row as usize];
                    g_sum += gp.grad;
                    h_sum += gp.hess;
                    g_sq_sum += gp.grad * gp.grad;
                }
                let g_eff = alloygbm_core::leaf_effective_gradient(
                    g_sum,
                    g_sq_sum,
                    row_count,
                    params.lambda_l1,
                    effective_dro_config(params),
                );
                out[k] = -g_eff / (h_sum + lambda_l2 + crate::LEAF_EPSILON);
            }
            out
        };
        let left_k = leaf_values(&left_rows);
        let right_k = leaf_values(&right_rows);

        // NodeStats for record-keeping (joint trainer doesn't consume them
        // beyond carrying them into SplitCandidate for compat).
        let summarize = |rows: &[u32]| -> NodeStats {
            let mut g = 0.0_f32;
            let mut h = 0.0_f32;
            for grad_vec in grads_per_output.iter().take(n_outputs) {
                for &row in rows {
                    let gp = grad_vec[row as usize];
                    g += gp.grad;
                    h += gp.hess;
                }
            }
            NodeStats {
                grad_sum: g,
                hess_sum: h,
                grad_sq_sum: 0.0,
                row_count: rows.len() as u32,
            }
        };
        let left_stats = summarize(&left_rows);
        let right_stats = summarize(&right_rows);

        Some(JointLeafCandidate {
            node,
            feature: feature as u32,
            threshold_bin: threshold_bin as u16,
            default_left,
            gain,
            left_rows,
            right_rows,
            left_k,
            right_k,
            left_stats,
            right_stats,
            parent_active_groups: active_groups,
            cat_bitset,
        })
    };

    // Initialize heap with the root candidate. v0.10.2.1 fix: use the
    // sampled row subset (when row_subsample < 1.0) so min_data_in_leaf
    // operates on the sampled training set, matching single-output.
    let root_rows: Vec<u32> = match sampled_root_rows {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows as u32).collect(),
    };
    let mut heap: BinaryHeap<JointLeafCandidate> = BinaryHeap::new();
    let root_node = JointLeafNode {
        local_node_id: 0,
        row_indices: root_rows,
    };
    if let Some(root_cand) = evaluate_node(root_node, root_active_groups) {
        heap.push(root_cand);
    }

    // Best-first growth: each pop adds one stump (one split → one new leaf).
    let mut stumps: Vec<TrainedStump> = Vec::new();
    let mut leaf_count: usize = 1; // root starts as one leaf

    while let Some(cand) = heap.pop() {
        if leaf_count >= max_leaves {
            break;
        }
        // Depth from local_node_id: depth = floor(log2(node_id + 1)).
        // (0 → 0, 1/2 → 1, 3/4/5/6 → 2, ...)
        let depth = (32 - (cand.node.local_node_id + 1).leading_zeros()) as usize - 1;
        if depth >= max_depth {
            continue;
        }

        let local_node_id = cand.node.local_node_id;
        let left_local = local_node_id * 2 + 1;
        let right_local = local_node_id * 2 + 2;

        // Compute child active groups before consuming cand.feature.
        let child_ag = match (constraint_index.as_ref(), cand.parent_active_groups) {
            (Some(idx), Some(parent_ag)) => Some(idx.descend(parent_ag, cand.feature)),
            _ => None,
        };

        // Commit the split as a TrainedStump.
        let stump = TrainedStump {
            split: SplitCandidate {
                node_id: local_node_id,
                feature_index: cand.feature,
                threshold_bin: cand.threshold_bin,
                gain: cand.gain,
                default_left: cand.default_left,
                is_categorical: cand.cat_bitset.is_some(),
                categorical_bitset: cand.cat_bitset.map(u64_to_bitset_bytes),
                left_stats: cand.left_stats,
                right_stats: cand.right_stats,
            },
            left_leaf_value: LeafValue::Scalar(cand.left_k[0]),
            right_leaf_value: LeafValue::Scalar(cand.right_k[0]),
            tree_weight: 1.0,
            multi_output_leaf_values: Some((cand.left_k.clone(), cand.right_k.clone())),
        };
        stumps.push(stump);
        leaf_count += 1; // splitting a leaf adds 1 net leaf (1 → 2).

        // Evaluate child candidates and push to heap if they have a viable split.
        if depth + 1 < max_depth {
            let left_node = JointLeafNode {
                local_node_id: left_local,
                row_indices: cand.left_rows,
            };
            if let Some(left_cand) = evaluate_node(left_node, child_ag) {
                heap.push(left_cand);
            }
            let right_node = JointLeafNode {
                local_node_id: right_local,
                row_indices: cand.right_rows,
            };
            if let Some(right_cand) = evaluate_node(right_node, child_ag) {
                heap.push(right_cand);
            }
        }
    }

    Ok(JointRoundResult { stumps })
}
