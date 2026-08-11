use alloygbm_core::{HistogramFeatureView, NodeStats, SplitCandidate, leaf_effective_gradient};
use alloygbm_engine::SplitSelectionOptions;
use wide::{CmpGe, CmpGt, f32x4, f64x4};

use crate::split_helpers::gain_materially_exceeds;
use crate::split_scan::with_dro_split_scan_scratch;

const EPSILON: f32 = 1e-6;

/// Apply the DRO radius term to four gradient prefixes in the scalar helper's
/// f64 arithmetic, then cross the same f32 soft-threshold boundary per lane.
#[inline]
pub(crate) fn dro_effective_gradient_f64x4(
    grad_values: [f32; 4],
    grad_sq_values: [f32; 4],
    row_counts: [u32; 4],
    radius: f32,
    l1_alpha: f32,
) -> [f32; 4] {
    let gradients = f64x4::from(grad_values.map(f64::from));
    let gradient_squares = f64x4::from(grad_sq_values.map(f64::from));
    let counts = f64x4::from(row_counts.map(|count| count.max(1) as f64));
    let means = gradients / counts;
    let variance = (gradient_squares / counts - means * means).max(f64x4::ZERO);
    let radius_terms =
        (f64x4::splat(f64::from(radius)) * counts.sqrt() * variance.sqrt()).to_array();
    let l1_alpha = l1_alpha.max(0.0);

    std::array::from_fn(|lane| {
        let threshold = l1_alpha + radius_terms[lane] as f32;
        let gradient = grad_values[lane];
        if gradient > threshold {
            gradient - threshold
        } else if gradient < -threshold {
            gradient + threshold
        } else {
            0.0
        }
    })
}

/// Exhaustively scan a numeric feature with active DRO using safe four-lane
/// SIMD for the f64 variance/radius term and f32 gain evaluation.
pub(crate) fn best_split_dro_numeric_simd(
    feature_histogram: HistogramFeatureView<'_>,
    node_id: u32,
    options: SplitSelectionOptions,
) -> Option<SplitCandidate> {
    let dro_config = options.dro_config.filter(|config| config.radius > 0.0)?;
    if feature_histogram.len() < 2 {
        return None;
    }

    let grad_sums = feature_histogram.grad_sums();
    let hess_sums = feature_histogram.hess_sums();
    let grad_sq_sums = feature_histogram.grad_sq_sums()?;
    let counts = feature_histogram.counts();
    let missing_bin_index = options.missing_bin_index;
    let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
        if missing_bin_index < feature_histogram.len() {
            (
                grad_sums[missing_bin_index],
                hess_sums[missing_bin_index],
                grad_sq_sums[missing_bin_index],
                counts[missing_bin_index],
            )
        } else {
            (0.0, 0.0, 0.0, 0)
        };

    let mut total_grad = 0.0_f32;
    let mut total_hess = 0.0_f32;
    let mut total_grad_sq = 0.0_f32;
    let mut total_count = 0_u32;
    for index in 0..feature_histogram.len() {
        total_grad += grad_sums[index];
        total_hess += hess_sums[index];
        total_grad_sq += grad_sq_sums[index];
        total_count += counts[index];
    }
    if total_hess <= options.min_child_hessian {
        return None;
    }

    let non_missing_grad = total_grad - missing_grad;
    let non_missing_hess = total_hess - missing_hess;
    let non_missing_grad_sq = total_grad_sq - missing_grad_sq;
    let non_missing_count = total_count.saturating_sub(missing_count);
    let parent_effective_gradient = leaf_effective_gradient(
        total_grad,
        total_grad_sq,
        total_count,
        options.l1_alpha,
        options.dro_config.as_ref(),
    );
    let parent_gain_term = 2.0
        * (0.5 * parent_effective_gradient * parent_effective_gradient
            / (total_hess + options.l2_lambda + EPSILON));
    let scan_limit = feature_histogram.len().min(missing_bin_index);
    if scan_limit == 0 {
        return None;
    }

    with_dro_split_scan_scratch(
        scan_limit,
        |cumulative_grad, cumulative_hess, cumulative_grad_sq, cumulative_count| {
            let mut grad = 0.0_f32;
            let mut hess = 0.0_f32;
            let mut grad_sq = 0.0_f32;
            let mut count = 0_u32;
            for index in 0..scan_limit {
                grad += grad_sums[index];
                hess += hess_sums[index];
                grad_sq += grad_sq_sums[index];
                count += counts[index];
                cumulative_grad[index] = grad;
                cumulative_hess[index] = hess;
                cumulative_grad_sq[index] = grad_sq;
                cumulative_count[index] = count;
            }

            let parent_gain_term_v = f32x4::splat(parent_gain_term);
            let lambda_v = f32x4::splat(options.l2_lambda);
            let epsilon_v = f32x4::splat(EPSILON);
            let min_hessian_v = f32x4::splat(options.min_child_hessian);
            let min_rows_v = f32x4::splat(options.min_rows_per_leaf as f32);
            let min_leaf_magnitude_v = f32x4::splat(options.min_leaf_magnitude);
            let negative_infinity_v = f32x4::splat(f32::NEG_INFINITY);

            let mut best_gain = 0.0_f32;
            let mut best_threshold = usize::MAX;
            let mut best_default_left = false;

            let mut chunk_start = 0usize;
            while chunk_start < scan_limit {
                let chunk_end = (chunk_start + 4).min(scan_limit);
                let chunk_len = chunk_end - chunk_start;
                let mut left_grad_values = [0.0_f32; 4];
                let mut left_hess_values = [0.0_f32; 4];
                let mut left_grad_sq_values = [0.0_f32; 4];
                let mut left_count_values = [0_u32; 4];
                for lane in 0..chunk_len {
                    let index = chunk_start + lane;
                    left_grad_values[lane] = cumulative_grad[index];
                    left_hess_values[lane] = cumulative_hess[index];
                    left_grad_sq_values[lane] = cumulative_grad_sq[index];
                    left_count_values[lane] = cumulative_count[index];
                }

                let right_grad_values =
                    std::array::from_fn(|lane| non_missing_grad - left_grad_values[lane]);
                let right_hess_values =
                    std::array::from_fn(|lane| non_missing_hess - left_hess_values[lane]);
                let right_grad_sq_values =
                    std::array::from_fn(|lane| non_missing_grad_sq - left_grad_sq_values[lane]);
                let right_count_values = std::array::from_fn(|lane| {
                    non_missing_count.saturating_sub(left_count_values[lane])
                });

                for &default_left in &[true, false] {
                    let (
                        effective_left_grad_values,
                        effective_left_hess_values,
                        effective_left_grad_sq_values,
                        effective_left_count_values,
                        effective_right_grad_values,
                        effective_right_hess_values,
                        effective_right_grad_sq_values,
                        effective_right_count_values,
                    ) = if default_left {
                        (
                            std::array::from_fn(|lane| left_grad_values[lane] + missing_grad),
                            std::array::from_fn(|lane| left_hess_values[lane] + missing_hess),
                            std::array::from_fn(|lane| left_grad_sq_values[lane] + missing_grad_sq),
                            std::array::from_fn(|lane| left_count_values[lane] + missing_count),
                            right_grad_values,
                            right_hess_values,
                            right_grad_sq_values,
                            right_count_values,
                        )
                    } else {
                        (
                            left_grad_values,
                            left_hess_values,
                            left_grad_sq_values,
                            left_count_values,
                            std::array::from_fn(|lane| right_grad_values[lane] + missing_grad),
                            std::array::from_fn(|lane| right_hess_values[lane] + missing_hess),
                            std::array::from_fn(|lane| {
                                right_grad_sq_values[lane] + missing_grad_sq
                            }),
                            std::array::from_fn(|lane| right_count_values[lane] + missing_count),
                        )
                    };

                    let effective_left = dro_effective_gradient_f64x4(
                        effective_left_grad_values,
                        effective_left_grad_sq_values,
                        effective_left_count_values,
                        dro_config.radius,
                        options.l1_alpha,
                    );
                    let effective_right = dro_effective_gradient_f64x4(
                        effective_right_grad_values,
                        effective_right_grad_sq_values,
                        effective_right_count_values,
                        dro_config.radius,
                        options.l1_alpha,
                    );
                    let effective_left_v = f32x4::from(effective_left);
                    let effective_right_v = f32x4::from(effective_right);
                    let left_hess_v = f32x4::from(effective_left_hess_values);
                    let right_hess_v = f32x4::from(effective_right_hess_values);
                    let left_count_v =
                        f32x4::from(effective_left_count_values.map(|count| count as f32));
                    let right_count_v =
                        f32x4::from(effective_right_count_values.map(|count| count as f32));

                    let left_denom = left_hess_v + lambda_v + epsilon_v;
                    let right_denom = right_hess_v + lambda_v + epsilon_v;
                    let left_gain_term = (f32x4::splat(0.5) * effective_left_v * effective_left_v
                        / left_denom)
                        * f32x4::splat(2.0);
                    let right_gain_term =
                        (f32x4::splat(0.5) * effective_right_v * effective_right_v / right_denom)
                            * f32x4::splat(2.0);
                    let gain_v = left_gain_term + right_gain_term - parent_gain_term_v;

                    let valid_mask = left_count_v.cmp_ge(min_rows_v)
                        & right_count_v.cmp_ge(min_rows_v)
                        & left_hess_v.cmp_gt(min_hessian_v)
                        & right_hess_v.cmp_gt(min_hessian_v);
                    let valid_mask = if options.min_leaf_magnitude > 0.0 {
                        let leaf_magnitude_ok = (effective_left_v.abs() / left_denom)
                            .cmp_ge(min_leaf_magnitude_v)
                            | (effective_right_v.abs() / right_denom).cmp_ge(min_leaf_magnitude_v);
                        valid_mask & leaf_magnitude_ok
                    } else {
                        valid_mask
                    };
                    let mut gains = valid_mask.blend(gain_v, negative_infinity_v).to_array();
                    for gain in gains.iter_mut().skip(chunk_len) {
                        *gain = f32::NEG_INFINITY;
                    }
                    for (lane, gain) in gains.iter_mut().take(chunk_len).enumerate() {
                        let threshold_bin = chunk_start + lane;
                        if threshold_bin + 1 >= scan_limit
                            && non_missing_count == cumulative_count[threshold_bin]
                        {
                            *gain = f32::NEG_INFINITY;
                        }
                        if !gain.is_finite() {
                            continue;
                        }
                        if gain_materially_exceeds(*gain, best_gain) {
                            best_gain = *gain;
                            best_threshold = threshold_bin;
                            best_default_left = default_left;
                        }
                    }
                }
                chunk_start = chunk_end;
            }

            if best_threshold == usize::MAX {
                return None;
            }

            let left_grad = cumulative_grad[best_threshold];
            let left_hess = cumulative_hess[best_threshold];
            let left_grad_sq = cumulative_grad_sq[best_threshold];
            let left_count = cumulative_count[best_threshold];
            let right_grad = non_missing_grad - left_grad;
            let right_hess = non_missing_hess - left_hess;
            let right_grad_sq = non_missing_grad_sq - left_grad_sq;
            let right_count = non_missing_count.saturating_sub(left_count);
            let (left_stats, right_stats) = if best_default_left {
                (
                    NodeStats {
                        grad_sum: left_grad + missing_grad,
                        hess_sum: left_hess + missing_hess,
                        grad_sq_sum: left_grad_sq + missing_grad_sq,
                        row_count: left_count + missing_count,
                    },
                    NodeStats {
                        grad_sum: right_grad,
                        hess_sum: right_hess,
                        grad_sq_sum: right_grad_sq,
                        row_count: right_count,
                    },
                )
            } else {
                (
                    NodeStats {
                        grad_sum: left_grad,
                        hess_sum: left_hess,
                        grad_sq_sum: left_grad_sq,
                        row_count: left_count,
                    },
                    NodeStats {
                        grad_sum: right_grad + missing_grad,
                        hess_sum: right_hess + missing_hess,
                        grad_sq_sum: right_grad_sq + missing_grad_sq,
                        row_count: right_count + missing_count,
                    },
                )
            };

            Some(SplitCandidate {
                node_id,
                feature_index: feature_histogram.feature_index(),
                threshold_bin: best_threshold as u16,
                gain: best_gain,
                default_left: best_default_left,
                is_categorical: false,
                categorical_bitset: None,
                left_stats,
                right_stats,
            })
        },
    )
}
