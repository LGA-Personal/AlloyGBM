//! Deterministic row/feature sampling and GOSS helpers.
//!
//! All draws use the same stateless splitmix64-derived hash
//! ([`mixed_hash`]) as the rest of the engine — no `rand` crate
//! dependency, fully deterministic given a `seed_base` + round /
//! index pair.
//!
//! Consumers:
//!
//! * The single-output training loop calls
//!   [`select_row_indices_for_round`] each round to obtain the root
//!   `row_indices` (standard / GOSS / DART dispatch).
//! * The multiclass training loop calls
//!   [`select_row_indices_for_round_multiclass`] which shares a single
//!   mask across the K per-class gradient buffers.
//! * The joint multi-output trainer (`crates/engine/src/joint.rs`)
//!   reuses [`goss_sample_indices`] and [`mixed_hash`] directly.
//! * DART (`crates/engine/src/dart.rs`) reuses [`mixed_hash`].
//! * Feature subsampling uses [`sampled_indices`].

use std::cmp::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use alloygbm_core::{BoostingMode, GradientPair};

pub(crate) fn sampling_seed_base(seed: u64, deterministic: bool) -> u64 {
    if deterministic {
        return seed;
    }
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    seed ^ now_nanos
}

pub(crate) fn mixed_hash(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sampled_count(total_count: usize, subsample: f32) -> usize {
    ((total_count as f32) * subsample)
        .ceil()
        .max(1.0)
        .min(total_count as f32) as usize
}

pub(crate) fn sampled_indices(
    total_count: usize,
    subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> Vec<usize> {
    if total_count == 0 {
        return Vec::new();
    }
    let keep_count = sampled_count(total_count, subsample);
    if keep_count >= total_count {
        return (0..total_count).collect();
    }

    let round_seed = mixed_hash(seed_base ^ round_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut scored = (0..total_count)
        .map(|index| {
            let index_seed = (index as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D);
            let hash = mixed_hash(round_seed ^ index_seed);
            (index, hash)
        })
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(keep_count, |lhs, rhs| {
        lhs.1.cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0))
    });

    let mut selected = scored[..keep_count]
        .iter()
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

pub(crate) fn sampled_row_indices(
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> Vec<u32> {
    sampled_indices(row_count, row_subsample, seed_base, round_index)
        .into_iter()
        .map(|row_index| row_index as u32)
        .collect()
}

/// Per-round row-selection dispatcher.  Dispatches on
/// `TrainParams::boosting_mode`:
///
/// * `BoostingMode::Standard` — uniform subsampling under
///   `row_subsample`.  Byte-identical to v0.7.5.
/// * `BoostingMode::Goss` — gradient-based one-side sampling.
///   `gradients` MUST already be the post-projection gradient buffer
///   for this round; the function mutates it in place to apply the
///   `(n - top_n) / other_n` amplification on the sampled-low rows
///   (top-by-magnitude rows are *not* amplified — they appear with
///   their original gradient/hessian, exactly as in the reference
///   LightGBM implementation).  We use realized counts rather than
///   the configured `(1 - top_rate) / other_rate` symbolic form so
///   that `ceil()` rounding and the `other_n <= n - top_n` cap don't
///   bias the unbiasedness contract at small `n` (see
///   `goss_sample_indices` for details).
/// * `BoostingMode::Dart` — row-selection itself is uniform (same as
///   Standard); the dropout + normalize cycle that makes DART distinct
///   is applied separately in the iteration loop
///   (`fit_iterations_with_optional_validation_summary`) before
///   gradient computation.  See `crates/engine/src/dart.rs`.
///
/// Returns the sorted set of row indices used as `root_row_indices`
/// for tree construction this round.
pub(crate) fn select_row_indices_for_round(
    boosting_mode: BoostingMode,
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
    gradients: &mut [GradientPair],
) -> Vec<u32> {
    match boosting_mode {
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            // Score rows by |gradient|.  Hessian could also be folded
            // in (e.g. `|grad| / sqrt(hess)`) but the LightGBM
            // reference uses |grad| only.
            let magnitudes: Vec<f32> = gradients.iter().map(|g| g.grad.abs()).collect();
            let (top, other, amplification) =
                goss_sample_indices(&magnitudes, top_rate, other_rate, seed_base, round_index);
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    gradients[idx].grad *= amplification;
                    gradients[idx].hess *= amplification;
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            merged
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            sampled_row_indices(row_count, row_subsample, seed_base, round_index)
        }
    }
}

/// Multiclass variant of [`select_row_indices_for_round`].
///
/// For multiclass GOSS the per-row score is the L1 norm of the per-class
/// gradient vector: `s_i = sum_k |g_{i,k}|` (LightGBM convention).  A single
/// row mask is shared across all K class gradient buffers, and the
/// amplification factor is applied identically to every class's gradient and
/// hessian.
///
/// `class_gradient_buffers[k]` is the gradient/hessian buffer for class `k`;
/// every buffer must have length `row_count`.  Mutated in place to apply
/// amplification when GOSS is active.
pub(crate) fn select_row_indices_for_round_multiclass(
    boosting_mode: BoostingMode,
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
    class_gradient_buffers: &mut [Vec<GradientPair>],
) -> Vec<u32> {
    match boosting_mode {
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            let k = class_gradient_buffers.len();
            assert!(
                k > 0,
                "multiclass GOSS requires at least one class gradient buffer"
            );
            debug_assert!(
                class_gradient_buffers
                    .iter()
                    .all(|buf| buf.len() == row_count),
                "every class gradient buffer must have length row_count"
            );
            let magnitudes: Vec<f32> = (0..row_count)
                .map(|i| {
                    class_gradient_buffers
                        .iter()
                        .take(k)
                        .map(|buf| buf[i].grad.abs())
                        .sum::<f32>()
                })
                .collect();
            let (top, other, amplification) =
                goss_sample_indices(&magnitudes, top_rate, other_rate, seed_base, round_index);
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    for class_buf in class_gradient_buffers.iter_mut().take(k) {
                        let pair = &mut class_buf[idx];
                        pair.grad *= amplification;
                        pair.hess *= amplification;
                    }
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            merged
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            sampled_row_indices(row_count, row_subsample, seed_base, round_index)
        }
    }
}

/// Gradient-based One-Side Sampling (GOSS, from LightGBM).
///
/// Strategy: keep the top `top_rate` fraction of rows by
/// `|gradient_magnitude|`, then uniformly sample `other_rate` fraction
/// from the rest.  Sampled-low-gradient rows are *amplified* by
/// `(n - top_n) / other_n` at the gradient-accumulation stage so the
/// histogram statistics remain an unbiased estimator of the full-data
/// gradient sums.  We use realized counts rather than the configured
/// `(1 - top_rate) / other_rate` symbolic form because `ceil()`
/// rounding (and the `other_n <= n - top_n` cap) shifts the realized
/// fractions away from the configured ones at small `n` — the rate
/// form would double the sampled-low contribution in those edge
/// cases.  For large `n` the two forms agree (since `top_n ≈ top_rate
/// · n` and `other_n ≈ other_rate · n`).
///
/// Returns `(sampled_row_indices, amplification, top_kept_count)`:
///
/// * `sampled_row_indices` — sorted ascending, includes both kept-top
///   and sampled-low rows.  Suitable to feed
///   `NodeSlice::row_indices`.
/// * `amplification` — multiplier the caller applies to gradients and
///   hessians on the sampled-low rows (not on the kept-top rows!) to
///   preserve unbiasedness.  Always `>= 1.0`; equals `1.0` when
///   `other_rate == 0`.
/// * `top_kept_count` — number of leading elements in
///   `sampled_row_indices` (after sorting) that are kept-top rows.
///   *Not* used directly — instead, the caller marks each row by
///   checking membership in a separate hash set.  Returned for
///   convenience and unit-test sanity checks.
pub(crate) fn goss_sample_indices(
    gradient_magnitudes: &[f32],
    top_rate: f32,
    other_rate: f32,
    seed_base: u64,
    round_index: u64,
) -> (Vec<u32>, Vec<u32>, f32) {
    let n = gradient_magnitudes.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 1.0);
    }
    let top_n = ((top_rate * n as f32).ceil() as usize).max(1).min(n);
    let other_n = ((other_rate * n as f32).ceil() as usize).min(n - top_n);

    // Rank by |gradient| descending using select_nth_unstable_by.
    let mut indexed: Vec<(u32, f32)> = gradient_magnitudes
        .iter()
        .enumerate()
        .map(|(i, &g)| (i as u32, g.abs()))
        .collect();
    if top_n < n {
        // After this call indexed[..top_n] contains the top_n rows by
        // |gradient| (in arbitrary order); indexed[top_n..] contains
        // the rest.
        indexed.select_nth_unstable_by(top_n - 1, |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
    }
    let mut top_indices: Vec<u32> = indexed[..top_n].iter().map(|(i, _)| *i).collect();

    let mut other_indices: Vec<u32> = if other_n > 0 && top_n < n {
        let mut rest_scored: Vec<(u32, u64)> = indexed[top_n..]
            .iter()
            .map(|(i, _)| {
                let seed = mixed_hash(
                    seed_base
                        ^ round_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (*i as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D),
                );
                (*i, seed)
            })
            .collect();
        rest_scored.select_nth_unstable_by(other_n - 1, |a, b| {
            a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        });
        rest_scored[..other_n].iter().map(|(i, _)| *i).collect()
    } else {
        Vec::new()
    };

    // Amplification uses **realized** counts (`(n - top_n) / other_n`)
    // rather than the configured rates (`(1 - top_rate) / other_rate`).
    // When `ceil()` rounding or the `other_n <= n - top_n` cap shifts the
    // realized fractions away from the configured ones — common at small
    // `n` — the rate-based form double-counts (or under-counts) the
    // sampled-low rows.  Example: `n=5`, `top_rate=0.2`, `other_rate=0.1`
    // gives `top_n=1`, `other_n=1`.  The unbiased multiplier for the
    // remaining pool of 4 rows sampled at size 1 is `4 / 1 = 4`, not
    // `(1 - 0.2) / 0.1 = 8`.  See `goss_amplification_uses_realized_counts`
    // for the contract test.
    let amplification = if other_n > 0 && top_n < n {
        (n - top_n) as f32 / other_n as f32
    } else {
        1.0
    };

    top_indices.sort_unstable();
    other_indices.sort_unstable();
    (top_indices, other_indices, amplification)
}
