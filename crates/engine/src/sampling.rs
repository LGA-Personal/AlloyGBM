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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RoundRowSelection {
    pub(crate) selected: Vec<u32>,
    pub(crate) excluded: Vec<u32>,
}

fn round_seed_for(seed_base: u64, round_index: u64) -> u64 {
    mixed_hash(seed_base ^ round_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Rank key for one index: rows are ordered by a hash of (round seed, index),
/// so which rows are sampled depends only on the seed and the round -- never
/// on the thread count or on iteration order.
///
/// Indices are unique, so the (hash, index) pair is a strict total order and
/// the cut between kept and dropped rows is never ambiguous.
#[inline]
fn selection_key(round_seed: u64, index: usize) -> (u64, u32) {
    let index_seed = (index as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D);
    (mixed_hash(round_seed ^ index_seed), index as u32)
}

/// Split `0..total_count` into a kept sample of `keep_count` and its
/// complement, both in ascending index order.
///
/// The rank order is found with `select_nth_unstable` (O(n)) purely to locate
/// the pivot key; the two output lists then come from a single ascending scan.
/// The obvious alternative -- collect both halves out of the partitioned
/// buffer and sort each -- costs an O(n log n) sort of the whole index space
/// every round to recover an order the scan produces for free. Rehashing each
/// index in the scan is far cheaper than that sort.
fn sampled_index_partition(
    total_count: usize,
    subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> (Vec<u32>, Vec<u32>) {
    if total_count == 0 {
        return (Vec::new(), Vec::new());
    }
    let keep_count = sampled_count(total_count, subsample);
    if keep_count >= total_count {
        return ((0..total_count as u32).collect(), Vec::new());
    }
    let round_seed = round_seed_for(seed_base, round_index);
    let mut scored: Vec<(u64, u32)> = (0..total_count)
        .map(|index| selection_key(round_seed, index))
        .collect();
    scored.select_nth_unstable(keep_count);
    let pivot = scored[keep_count];
    drop(scored);

    let mut selected = Vec::with_capacity(keep_count);
    let mut excluded = Vec::with_capacity(total_count - keep_count);
    for index in 0..total_count {
        if selection_key(round_seed, index) < pivot {
            selected.push(index as u32);
        } else {
            excluded.push(index as u32);
        }
    }
    (selected, excluded)
}

pub(crate) fn sampled_indices(
    total_count: usize,
    subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> Vec<usize> {
    sampled_index_partition(total_count, subsample, seed_base, round_index)
        .0
        .into_iter()
        .map(|index| index as usize)
        .collect()
}

#[allow(dead_code)]
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
/// Returns the sorted selected and excluded row partitions for this round.
pub(crate) fn select_row_indices_for_round(
    boosting_mode: BoostingMode,
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
    gradients: &mut [GradientPair],
) -> RoundRowSelection {
    match boosting_mode {
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            // Score rows by |gradient|.  Hessian could also be folded
            // in (e.g. `|grad| / sqrt(hess)`) but the LightGBM
            // reference uses |grad| only.
            let magnitudes: Vec<f32> = gradients.iter().map(|g| g.grad.abs()).collect();
            let selection =
                goss_index_partition(&magnitudes, top_rate, other_rate, seed_base, round_index);
            let amplification = selection.amplification;
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &selection.other {
                    let idx = row as usize;
                    gradients[idx].grad *= amplification;
                    gradients[idx].hess *= amplification;
                }
            }
            let mut selected = Vec::with_capacity(selection.top.len() + selection.other.len());
            selected.extend_from_slice(&selection.top);
            selected.extend_from_slice(&selection.other);
            selected.sort_unstable();
            RoundRowSelection {
                selected,
                excluded: selection.excluded,
            }
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            // Every row usually has a positive hessian, and then the sample is
            // just a partition of `0..row_count`. Test that with a
            // short-circuiting scan rather than building the index list first:
            // materializing it costs a row_count-sized allocation every round
            // and is thrown away immediately in the common case.
            if gradients.iter().all(|pair| pair.hess > 0.0) {
                let (selected, excluded) =
                    sampled_index_partition(row_count, row_subsample, seed_base, round_index);
                return RoundRowSelection { selected, excluded };
            }
            let active = gradients
                .iter()
                .enumerate()
                .filter_map(|(index, pair)| (pair.hess > 0.0).then_some(index))
                .collect::<Vec<_>>();
            let (selected_positions, _) =
                sampled_index_partition(active.len(), row_subsample, seed_base, round_index);
            let selected = selected_positions
                .into_iter()
                .map(|position| active[position as usize])
                .collect::<Vec<_>>();
            let mut selected_mask = vec![false; row_count];
            for &row_index in &selected {
                selected_mask[row_index] = true;
            }
            let excluded = selected_mask
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| (!selected).then_some(index))
                .collect::<Vec<_>>();
            RoundRowSelection {
                selected: selected
                    .into_iter()
                    .map(|row_index| row_index as u32)
                    .collect(),
                excluded: excluded
                    .into_iter()
                    .map(|row_index| row_index as u32)
                    .collect(),
            }
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
) -> RoundRowSelection {
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
            let selection =
                goss_index_partition(&magnitudes, top_rate, other_rate, seed_base, round_index);
            let amplification = selection.amplification;
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &selection.other {
                    let idx = row as usize;
                    for class_buf in class_gradient_buffers.iter_mut().take(k) {
                        let pair = &mut class_buf[idx];
                        pair.grad *= amplification;
                        pair.hess *= amplification;
                    }
                }
            }
            let mut selected = Vec::with_capacity(selection.top.len() + selection.other.len());
            selected.extend_from_slice(&selection.top);
            selected.extend_from_slice(&selection.other);
            selected.sort_unstable();
            RoundRowSelection {
                selected,
                excluded: selection.excluded,
            }
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            // As in the single-output path: test for a fully active row set
            // with a short-circuiting scan instead of building the index list
            // and comparing its length, which allocates row_count entries per
            // round only to discard them in the common case.
            let all_active = (0..row_count).all(|index| {
                class_gradient_buffers
                    .iter()
                    .any(|gradients| gradients[index].hess > 0.0)
            });
            if all_active {
                let (selected, excluded) =
                    sampled_index_partition(row_count, row_subsample, seed_base, round_index);
                return RoundRowSelection { selected, excluded };
            }
            let active = (0..row_count)
                .filter(|&index| {
                    class_gradient_buffers
                        .iter()
                        .any(|gradients| gradients[index].hess > 0.0)
                })
                .collect::<Vec<_>>();
            let (selected_positions, _) =
                sampled_index_partition(active.len(), row_subsample, seed_base, round_index);
            let selected = selected_positions
                .into_iter()
                .map(|position| active[position as usize])
                .collect::<Vec<_>>();
            let mut selected_mask = vec![false; row_count];
            for &row_index in &selected {
                selected_mask[row_index] = true;
            }
            let excluded = selected_mask
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| (!selected).then_some(index))
                .collect::<Vec<_>>();
            RoundRowSelection {
                selected: selected
                    .into_iter()
                    .map(|row_index| row_index as u32)
                    .collect(),
                excluded: excluded
                    .into_iter()
                    .map(|row_index| row_index as u32)
                    .collect(),
            }
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
/// Returns `(top_indices, other_indices, amplification)`:
///
/// * `top_indices` and `other_indices` — sorted ascending, include both kept-top
///   and sampled-low rows.  Suitable to feed
///   `NodeSlice::row_indices`.
/// * `amplification` — multiplier the caller applies to gradients and
///   hessians on the sampled-low rows (not on the kept-top rows!) to
///   preserve unbiasedness.  Always `>= 1.0`; equals `1.0` when
///   `other_rate == 0`.
///
/// The excluded side is retained internally by the round-selection
/// dispatchers; this compatibility wrapper returns only the historical
/// sampled tuple used by feature sampling, tests, and joint training.
struct GossRowSelection {
    top: Vec<u32>,
    other: Vec<u32>,
    excluded: Vec<u32>,
    amplification: f32,
}

fn goss_index_partition(
    gradient_magnitudes: &[f32],
    top_rate: f32,
    other_rate: f32,
    seed_base: u64,
    round_index: u64,
) -> GossRowSelection {
    let n = gradient_magnitudes.len();
    if n == 0 {
        return GossRowSelection {
            top: Vec::new(),
            other: Vec::new(),
            excluded: Vec::new(),
            amplification: 1.0,
        };
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

    let (mut other_indices, mut excluded_indices): (Vec<u32>, Vec<u32>) = if top_n < n {
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
        if other_n > 0 {
            rest_scored.select_nth_unstable_by(other_n - 1, |a, b| {
                a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
            });
        }
        (
            rest_scored[..other_n].iter().map(|(i, _)| *i).collect(),
            rest_scored[other_n..].iter().map(|(i, _)| *i).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
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
    excluded_indices.sort_unstable();
    GossRowSelection {
        top: top_indices,
        other: other_indices,
        excluded: excluded_indices,
        amplification,
    }
}

pub(crate) fn goss_sample_indices(
    gradient_magnitudes: &[f32],
    top_rate: f32,
    other_rate: f32,
    seed_base: u64,
    round_index: u64,
) -> (Vec<u32>, Vec<u32>, f32) {
    let selection = goss_index_partition(
        gradient_magnitudes,
        top_rate,
        other_rate,
        seed_base,
        round_index,
    );
    (selection.top, selection.other, selection.amplification)
}
