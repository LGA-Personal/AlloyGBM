//! DART (Dropouts meet MART) dropout + normalization helpers.
//!
//! Used by the single-output training loop (regression / binary
//! classification / single-label ranking) when [`BoostingMode::Dart`]
//! is active. Multiclass DART is not yet supported.
//!
//! Reference: Rashmi & Gilad-Bachrach, "DART: Dropouts meet Multiple
//! Additive Regression Trees" (AISTATS 2015).
//!
//! Design notes:
//!
//! * Random draws use the same stateless splitmix64-derived hash as the
//!   rest of the engine (see [`mixed_hash`]) — no `rand` crate
//!   dependency, fully deterministic given a `seed_base` + round /
//!   stump index pair.
//! * Per-stump scores are derived independently so the dropout
//!   decision is parallelisable if a future training loop wants to
//!   evaluate it in parallel.
//! * `select_dropouts` always returns at least one stump index when
//!   `n_existing > 0` and `drop_rate > 0`, matching the LightGBM
//!   convention that a DART round must drop something to be a DART
//!   round (otherwise it degrades to standard MART for that round).

use alloygbm_core::{DartNormalize, DartSampleType};

use crate::mixed_hash;

/// Per-fit DART state. `tree_weights` parallels `TrainedModel.stumps`
/// (one entry per stump, in stump order). `dropped_per_round[r]`
/// records which stump indices were dropped before fitting the
/// round-`r` tree (purely diagnostic — not currently persisted).
#[derive(Debug, Clone, Default)]
pub struct DartState {
    /// One entry per stump in stump-order; consumed by the trainer
    /// when stamping `TrainedStump::tree_weight` after the loop.
    pub tree_weights: Vec<f32>,
    /// Per-round dropout record (round index = outer Vec index).
    /// Inner Vec is the stump indices that were dropped before
    /// fitting that round's new tree.
    pub dropped_per_round: Vec<Vec<usize>>,
}

/// Hash a (seed, round, stump) triple to a deterministic float in
/// `[0.0, 1.0)`. Used for uniform per-stump dropout decisions.
fn dropout_score(seed_base: u64, round_index: usize, stump_idx: usize) -> f32 {
    let mixed = mixed_hash(
        seed_base
            ^ (round_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (stump_idx as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D),
    );
    // Top 24 bits → [0, 1).
    ((mixed >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Pick which previously-trained stumps to drop before fitting the new
/// tree at the given `round_index`. Returns sorted stump indices.
///
/// Sampling:
/// - [`DartSampleType::Uniform`]: every stump dropped independently
///   with probability `drop_rate`.
/// - [`DartSampleType::Weighted`]: probability proportional to
///   `|tree_weight|` (heavier trees more likely to drop), normalized
///   so the *expected* drop count matches `drop_rate * n_existing`.
///
/// Results are then truncated to `max_drop`. If `drop_rate > 0` and
/// `n_existing > 0` and the sampling step produced zero candidates,
/// we deterministically pick one stump (the lowest-hash one) to
/// guarantee the LightGBM "at least one drop per DART round"
/// invariant.
pub fn select_dropouts(
    n_existing: usize,
    drop_rate: f32,
    max_drop: usize,
    sample_type: DartSampleType,
    tree_weights: &[f32],
    seed_base: u64,
    round_index: usize,
) -> Vec<usize> {
    if n_existing == 0 || drop_rate <= 0.0 || max_drop == 0 {
        return Vec::new();
    }

    let mut candidates: Vec<usize> = match sample_type {
        DartSampleType::Uniform => (0..n_existing)
            .filter(|&i| dropout_score(seed_base, round_index, i) < drop_rate)
            .collect(),
        DartSampleType::Weighted => {
            let total: f32 = tree_weights.iter().take(n_existing).map(|w| w.abs()).sum();
            if total <= 0.0 {
                Vec::new()
            } else {
                (0..n_existing)
                    .filter(|&i| {
                        let w = tree_weights[i].abs() / total;
                        // p = drop_rate * n_existing * w (expected drop count
                        // = drop_rate * n_existing for uniform; weighted
                        // skews mass toward heavier trees but preserves the
                        // marginal expectation).
                        let p = (drop_rate * n_existing as f32 * w).min(1.0);
                        dropout_score(seed_base, round_index, i) < p
                    })
                    .collect()
            }
        }
    };

    if candidates.is_empty() {
        // LightGBM-style "always drop at least one" guarantee. Pick
        // the stump with the smallest score for stability.
        let (lowest_idx, _) = (0..n_existing)
            .map(|i| (i, dropout_score(seed_base, round_index, i)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("n_existing > 0 ensures at least one candidate");
        candidates.push(lowest_idx);
    }

    if candidates.len() > max_drop {
        // Stable truncation: keep the `max_drop` with the lowest scores
        // (so the truncation respects the same ordering the rest of
        // the function uses).
        candidates.sort_by(|&a, &b| {
            dropout_score(seed_base, round_index, a)
                .partial_cmp(&dropout_score(seed_base, round_index, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(max_drop);
    }

    candidates.sort_unstable();
    candidates
}

/// After fitting the round's new tree, rescale the dropped stumps and
/// the new tree according to `normalize_type`.
///
/// Convention:
///
/// - [`DartNormalize::Tree`]: each dropped tree's weight is multiplied
///   by `K / (K + 1)`; the new tree gets weight `1 / (K + 1)`. The
///   intuition: the new tree's leaf values are fit to the residuals of
///   the *dropped-out* ensemble, so they "absorb" what the dropped
///   trees would have contributed. Splitting the absorbed mass across
///   `K + 1` trees keeps the sum unbiased.
/// - [`DartNormalize::Forest`]: every dropped tree gets weight
///   `1 / (K + 1)`, same as the new tree. A more aggressive rescale.
///
/// `new_tree_index` is the position the brand-new tree will occupy
/// after the trainer appends it. Caller must have already grown
/// `tree_weights` to include the new stump's slot (pushing `1.0` as a
/// placeholder is fine — this function will overwrite it).
pub fn apply_normalization(
    tree_weights: &mut Vec<f32>,
    dropped: &[usize],
    normalize_type: DartNormalize,
    new_tree_index: usize,
) {
    let k = dropped.len() as f32;
    let new_w = 1.0 / (k + 1.0);
    let drop_w = match normalize_type {
        DartNormalize::Tree => k / (k + 1.0),
        DartNormalize::Forest => 1.0 / (k + 1.0),
    };
    for &i in dropped {
        if i < tree_weights.len() {
            tree_weights[i] *= drop_w;
        }
    }
    if new_tree_index < tree_weights.len() {
        tree_weights[new_tree_index] = new_w;
    } else {
        // Caller is expected to size the vec first; push as a defensive
        // fallback so we never panic on out-of-bounds.
        tree_weights.push(new_w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_dropouts_returns_empty_when_no_existing() {
        let got = select_dropouts(0, 0.5, 10, DartSampleType::Uniform, &[], 42, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn select_dropouts_returns_empty_when_drop_rate_zero() {
        let weights = vec![1.0; 100];
        let got = select_dropouts(100, 0.0, 10, DartSampleType::Uniform, &weights, 42, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn select_dropouts_capped_by_max_drop() {
        // drop_rate=1.0 marks every stump; max_drop=3 truncates.
        let weights = vec![1.0; 100];
        let got = select_dropouts(100, 1.0, 3, DartSampleType::Uniform, &weights, 42, 0);
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn select_dropouts_returns_sorted_indices() {
        let weights = vec![1.0; 50];
        let got = select_dropouts(50, 0.5, 50, DartSampleType::Uniform, &weights, 42, 0);
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(got, sorted);
    }

    #[test]
    fn select_dropouts_always_drops_at_least_one_when_drop_rate_positive() {
        // drop_rate=0.0001 means uniform sampling will almost certainly
        // produce zero candidates; the "always drop one" fallback must
        // kick in for every seed.
        let weights = vec![1.0; 100];
        for seed in 0u64..20 {
            let got = select_dropouts(100, 0.0001, 50, DartSampleType::Uniform, &weights, seed, 0);
            assert!(!got.is_empty(), "seed {seed} produced zero drops");
        }
    }

    #[test]
    fn select_dropouts_is_deterministic_given_seed_and_round() {
        let weights = vec![1.0; 50];
        let a = select_dropouts(50, 0.3, 50, DartSampleType::Uniform, &weights, 123, 7);
        let b = select_dropouts(50, 0.3, 50, DartSampleType::Uniform, &weights, 123, 7);
        assert_eq!(a, b, "same (seed, round) must produce same dropouts");
    }

    #[test]
    fn select_dropouts_different_round_produces_different_dropouts() {
        // Sanity check that round_index actually feeds the hash; we
        // expect different sets across rounds (probability of identical
        // output is astronomically low for drop_rate=0.5 on 50 stumps).
        let weights = vec![1.0; 50];
        let r0 = select_dropouts(50, 0.5, 50, DartSampleType::Uniform, &weights, 42, 0);
        let r1 = select_dropouts(50, 0.5, 50, DartSampleType::Uniform, &weights, 42, 1);
        assert_ne!(r0, r1, "different rounds must produce different dropouts");
    }

    #[test]
    fn apply_normalization_tree_mode_preserves_unbiased_sum() {
        // K=2 dropouts, both weight 1.0.
        //  - drop_w = 2/3 → each dropped → 2/3
        //  - new_w  = 1/3
        let mut weights = vec![1.0, 1.0, 1.0]; // last slot = new tree
        apply_normalization(&mut weights, &[0, 1], DartNormalize::Tree, 2);
        assert!((weights[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((weights[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn apply_normalization_forest_mode_uses_single_factor() {
        // K=2: drop_w = 1/3, new_w = 1/3.
        let mut weights = vec![1.0, 1.0, 1.0];
        apply_normalization(&mut weights, &[0, 1], DartNormalize::Forest, 2);
        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((weights[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((weights[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn apply_normalization_with_no_dropouts_sets_new_tree_to_one() {
        // K=0: drop_w doesn't matter (no dropouts), new_w = 1/1 = 1.0.
        let mut weights = vec![1.0, 1.0];
        apply_normalization(&mut weights, &[], DartNormalize::Tree, 1);
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_normalization_grows_vec_if_index_past_end() {
        let mut weights = vec![1.0, 1.0]; // length 2; new_tree_index=5 is past end
        apply_normalization(&mut weights, &[], DartNormalize::Tree, 5);
        // Defensive push: weights gets a 1.0 appended (K=0 → new_w=1.0).
        assert_eq!(weights.len(), 3);
        assert!((weights[2] - 1.0).abs() < 1e-6);
    }
}
