//! Private utility helpers for the joint multi-output trainer.
//!
//! Extracted from `joint/mod.rs` in v0.12.2 task 5.1. All helpers are
//! visible to sibling modules inside the `joint/` subdirectory only
//! (`pub(super)`).

use alloygbm_core::{BinnedMatrix, GradientPair, LeafSolverKind, MISSING_BIN_U8, TrainParams};

use crate::TrainedStump;

use super::TREE_NODE_STRIDE;

/// Convert a u64 categorical bitset (bit `k` = 1 means category `k` goes
/// left) into the byte-packed Vec<u8> format used by the single-output
/// trainer's `SplitCandidate::categorical_bitset`. Bit `K` of byte `K/8`
/// represents category `K`; trailing bytes that contain only zeros are
/// trimmed (single-output convention).
pub(super) fn u64_to_bitset_bytes(bs: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for byte_idx in 0..8 {
        let byte = ((bs >> (byte_idx * 8)) & 0xFF) as u8;
        out.push(byte);
    }
    while out.len() > 1 && *out.last().unwrap() == 0 {
        out.pop();
    }
    out
}

/// Inverse of `u64_to_bitset_bytes`: decodes a Vec<u8> bitset back into a
/// u64. Used by JointPredictor when evaluating categorical stumps.
pub(super) fn bitset_bytes_to_u64(bytes: &[u8]) -> u64 {
    let mut out: u64 = 0;
    for (byte_idx, &byte) in bytes.iter().enumerate().take(8) {
        out |= (byte as u64) << (byte_idx * 8);
    }
    out
}

/// Returns the effective DRO config for joint training. Returns
/// `Some(&cfg)` only when DRO is genuinely active — i.e. the user
/// requested `leaf_solver = Dro` AND the radius is strictly positive.
///
/// This gates against the "raw TrainParams with `dro_config = Some(...)`
/// but `leaf_solver = Standard`" case that the Python bridge already
/// avoids but the public Rust joint API does not (it doesn't call
/// `validate_train_params`). Mirrors the single-output trainer's
/// behavior post-validation, where the same inconsistency is rejected
/// at `validate_train_params` time.
///
/// Note `radius = 0.0` collapses to `None` even when `leaf_solver = Dro`,
/// preserving byte-equivalence between standard fits and DRO-with-radius-0
/// fits (pinned by `joint_dro_radius_zero_matches_standard_byte_for_byte`).
pub(super) fn effective_dro_config(params: &TrainParams) -> Option<&alloygbm_core::DroConfig> {
    if params.leaf_solver != LeafSolverKind::Dro {
        return None;
    }
    let cfg = params.dro_config.as_ref()?;
    if cfg.radius <= 0.0 {
        return None;
    }
    Some(cfg)
}

/// v0.10.6: scan a node's row indices once and accumulate per-side per-factor
/// exposure sums for a candidate split. Used by `split_penalty` mode to compute
/// the factor-load penalty.
///
/// Missing rows (bin == `MISSING_BIN_U8`) are skipped because the
/// `default_left` direction is decided AFTER the best split is selected (the
/// joint trainer's "more rows wins" heuristic). Skipping them under-estimates
/// the penalty by exactly the missing-row contribution to factor load; this is
/// a deterministic conservative choice and matches the documented bias.
pub(super) fn accumulate_factor_sums_for_threshold(
    binned: &BinnedMatrix,
    exposures: &alloygbm_core::FactorExposureMatrix,
    row_indices: &[u32],
    feature: usize,
    threshold_bin: u8,
    cat_bitset: Option<u64>,
) -> (Vec<f32>, Vec<f32>) {
    let factor_count = exposures.factor_count;
    let feature_count = binned.feature_count;
    let mut left = vec![0.0_f32; factor_count];
    let mut right = vec![0.0_f32; factor_count];
    for &row in row_indices {
        let bin = binned
            .row_bin(row as usize * feature_count + feature)
            .min(u16::from(u8::MAX)) as u8;
        if bin == MISSING_BIN_U8 {
            continue;
        }
        let goes_left = if let Some(bs) = cat_bitset {
            bin < 64 && (bs & (1u64 << bin)) != 0
        } else {
            bin <= threshold_bin
        };
        let exposure_start = row as usize * factor_count;
        let exposure_row = &exposures.values[exposure_start..exposure_start + factor_count];
        let target = if goes_left { &mut left } else { &mut right };
        for (s, e) in target.iter_mut().zip(exposure_row) {
            *s += *e;
        }
    }
    (left, right)
}

/// Return `Some(config)` only when factor neutralization is actually active
/// for this fit. Inert configs (kind = None) collapse to `None`.
///
/// Why a helper rather than `validate_train_params`: the joint trainer accepts
/// raw `TrainParams` without invoking the single-output validator (which would
/// also reject joint-unrelated configs like linear leaves). This helper is the
/// SOURCE OF TRUTH for "is neutralization on?" — every behavioral site (target
/// residualization in pre_target mode, gradient projection in
/// per_round_gradient mode, FactorSplitContext construction in split_penalty
/// mode) AND the artifact serializer must call it.
pub(super) fn effective_neutralization_config(
    params: &TrainParams,
) -> Option<&alloygbm_core::FactorNeutralizationConfig> {
    let cfg = params.neutralization_config.as_ref()?;
    if cfg.kind == alloygbm_core::NeutralizationKind::None {
        return None;
    }
    // `SplitPenalty` with `split_penalty == 0` is behaviorally identical to
    // `None` — the per-candidate penalty collapses to 0 — so treat it as inert.
    // This mirrors v0.10.5's `effective_dro_config` collapsing radius=0 to None
    // and preserves byte-equivalence with v0.10.5 fits when the penalty is off.
    // (PreTarget and PerRoundGradient don't have an analogous "off" knob —
    // ridge_lambda is a regularization hyperparameter, not an activation
    // switch — so they only collapse on `kind == None`.)
    if cfg.kind == alloygbm_core::NeutralizationKind::SplitPenalty && cfg.split_penalty == 0.0 {
        return None;
    }
    Some(cfg)
}

/// Walk every row through one tree's stumps and accumulate
/// `sign * scale * leaf_value_k` into `predictions[k][row]`.
///
/// Shared by the round-end prediction update (v0.10.0 fix), DART
/// dropout subtraction (v0.10.3), and warm-start prior-stump replay
/// (v0.10.3). Tree IDs come in two flavors:
///
/// * **pre-encode** — `node_id` is the local node id directly (used
///   inside `fit_joint_inner` between `build_joint_round*` returning
///   and `encode_tree_node_id` rewriting).
/// * **post-encode** — `node_id` carries the global tree id in the
///   high bits (used for prior-round stumps already in `all_stumps`).
///
/// Both forms work because the lookup key is `node_id % TREE_NODE_STRIDE`,
/// which extracts the local id under either encoding (local ids are
/// always `< TREE_NODE_STRIDE`).
#[allow(clippy::too_many_arguments)]
pub(super) fn walk_tree_into_predictions(
    tree_stumps: &[TrainedStump],
    binned_matrix: &BinnedMatrix,
    feature_count: usize,
    n_rows: usize,
    n_outputs: usize,
    predictions: &mut [Vec<f32>],
    sign: f32,
    scale: f32,
) {
    let stumps_by_local: std::collections::HashMap<u32, &TrainedStump> = tree_stumps
        .iter()
        .map(|s| (s.split.node_id % TREE_NODE_STRIDE, s))
        .collect();
    for row in 0..n_rows {
        let mut current_node: u32 = 0;
        let mut last_leaf: Option<&(Vec<f32>, Vec<f32>)> = None;
        let mut last_went_left = false;
        loop {
            let Some(stump) = stumps_by_local.get(&current_node) else {
                break;
            };
            let feature = stump.split.feature_index as usize;
            let threshold_bin = stump.split.threshold_bin as usize;
            let bin = binned_matrix
                .row_bin(row * feature_count + feature)
                .min(u16::from(u8::MAX)) as u8;
            let went_left = if bin == MISSING_BIN_U8 {
                stump.split.default_left
            } else if stump.split.is_categorical {
                let bs = stump
                    .split
                    .categorical_bitset
                    .as_ref()
                    .map(|b| bitset_bytes_to_u64(b))
                    .unwrap_or(0);
                bin < 64 && (bs & (1u64 << bin)) != 0
            } else {
                (bin as usize) <= threshold_bin
            };
            last_leaf = stump.multi_output_leaf_values.as_ref();
            last_went_left = went_left;
            current_node = if went_left {
                current_node * 2 + 1
            } else {
                current_node * 2 + 2
            };
        }
        if let Some((left_k, right_k)) = last_leaf {
            let delta = if last_went_left { left_k } else { right_k };
            for (k, pred_vec) in predictions.iter_mut().enumerate().take(n_outputs) {
                pred_vec[row] += sign * scale * delta[k];
            }
        }
    }
}

/// Joint analogue of [`crate::select_row_indices_for_round_multiclass`].
///
/// For joint multi-output the per-row score is `s_i = Σₖ |g_{i,k}|`
/// across the K per-output gradient buffers (matches LightGBM
/// multiclass GOSS). A single row mask is shared across all K buffers,
/// and the amplification factor is applied identically to every
/// output's gradient and hessian.
///
/// Returns `Some(sampled_rows)` when GOSS is active; `None` when
/// `BoostingMode::Standard` or `BoostingMode::Dart` is in effect (the
/// caller falls back to the existing row_subsample path).
pub(super) fn select_joint_row_indices_for_round(
    boosting_mode: alloygbm_core::BoostingMode,
    n_rows: usize,
    seed_base: u64,
    round_index: u64,
    grads_per_output: &mut [Vec<GradientPair>],
) -> Option<Vec<u32>> {
    use alloygbm_core::BoostingMode as BM;
    match boosting_mode {
        BM::Goss {
            top_rate,
            other_rate,
        } => {
            let n_outputs = grads_per_output.len();
            debug_assert!(n_outputs > 0, "joint GOSS requires K >= 1");
            debug_assert!(
                grads_per_output.iter().all(|buf| buf.len() == n_rows),
                "every per-output gradient buffer must have length n_rows"
            );
            let magnitudes: Vec<f32> = (0..n_rows)
                .map(|i| {
                    grads_per_output
                        .iter()
                        .take(n_outputs)
                        .map(|buf| buf[i].grad.abs())
                        .sum::<f32>()
                })
                .collect();
            let (top, other, amplification) = crate::goss_sample_indices(
                &magnitudes,
                top_rate,
                other_rate,
                seed_base,
                round_index,
            );
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    for buf in grads_per_output.iter_mut().take(n_outputs) {
                        let p = &mut buf[idx];
                        p.grad *= amplification;
                        p.hess *= amplification;
                    }
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            Some(merged)
        }
        BM::Standard | BM::Dart { .. } => None,
    }
}
