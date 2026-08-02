use alloygbm_core::{HistogramFeatureView, NodeStats, SplitCandidate};
use alloygbm_engine::{MorphContext, SplitSelectionOptions};
use wide::{CmpGe, CmpGt, CmpLt};

use crate::simd::{f32x8, l1_threshold_f32x8};
use crate::split_helpers::gain_materially_exceeds;
use crate::split_scan::with_split_scan_scratch;

const GAIN_EPSILON: f32 = 1e-6;
const INFO_EPSILON: f32 = 1e-10;

pub(crate) fn best_split_morph_numeric_simd(
    feature_histogram: HistogramFeatureView<'_>,
    node_id: u32,
    options: SplitSelectionOptions,
    morph: &MorphContext,
) -> Option<SplitCandidate> {
    if feature_histogram.len() < 2 {
        return None;
    }

    let grad_sums = feature_histogram.grad_sums();
    let hess_sums = feature_histogram.hess_sums();
    let grad_sq_sums = feature_histogram.grad_sq_sums();
    let counts = feature_histogram.counts();
    let missing_bin_index = options.missing_bin_index;
    let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
        if missing_bin_index < feature_histogram.len() {
            (
                grad_sums[missing_bin_index],
                hess_sums[missing_bin_index],
                grad_sq_sums.map_or(0.0, |values| values[missing_bin_index]),
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
        total_grad_sq += grad_sq_sums.map_or(0.0, |values| values[index]);
        total_count += counts[index];
    }
    if total_hess <= options.min_child_hessian {
        return None;
    }

    let scan_limit = feature_histogram.len().min(missing_bin_index);
    if scan_limit == 0 {
        return None;
    }

    let non_missing_grad = total_grad - missing_grad;
    let non_missing_hess = total_hess - missing_hess;
    let non_missing_count = total_count.saturating_sub(missing_count);
    let parent_gain_gradient =
        crate::split_helpers::l1_threshold_gradient(total_grad, options.l1_alpha);
    let parent_term = parent_gain_gradient * parent_gain_gradient
        / (total_hess + options.l2_lambda + GAIN_EPSILON);
    let parent_curvature = (total_hess + options.l2_lambda).max(GAIN_EPSILON);
    let gradient_scale = morph.grad_std.abs().max(INFO_EPSILON);
    let normalized_gain_denom =
        (parent_curvature * gradient_scale * gradient_scale).max(INFO_EPSILON);
    let smoothing = 1.0
        + morph.config.evolution_pressure
            * (morph.iteration as f32 / morph.total_iterations.max(1) as f32);
    let parent_info = info_side_scalar(
        parent_gain_gradient,
        total_count,
        morph.grad_mean,
        morph.grad_std,
        smoothing,
    );

    with_split_scan_scratch(
        scan_limit,
        |cumulative_grad, cumulative_hess, cumulative_count| {
            let mut grad = 0.0_f32;
            let mut hess = 0.0_f32;
            let mut count = 0_u32;
            for index in 0..scan_limit {
                grad += grad_sums[index];
                hess += hess_sums[index];
                count += counts[index];
                cumulative_grad[index] = grad;
                cumulative_hess[index] = hess;
                cumulative_count[index] = count;
            }

            let zero = f32x8::splat(0.0);
            let one = f32x8::splat(1.0);
            let negative_infinity = f32x8::splat(f32::NEG_INFINITY);
            let non_missing_grad_v = f32x8::splat(non_missing_grad);
            let non_missing_hess_v = f32x8::splat(non_missing_hess);
            let non_missing_count_v = f32x8::splat(non_missing_count as f32);
            let missing_grad_v = f32x8::splat(missing_grad);
            let missing_hess_v = f32x8::splat(missing_hess);
            let missing_count_v = f32x8::splat(missing_count as f32);
            let lambda_v = f32x8::splat(options.l2_lambda);
            let gain_epsilon_v = f32x8::splat(GAIN_EPSILON);
            let min_hessian_v = f32x8::splat(options.min_child_hessian);
            let min_rows_v = f32x8::splat(options.min_rows_per_leaf as f32);
            let min_leaf_magnitude_v = f32x8::splat(options.min_leaf_magnitude);
            let parent_term_v = f32x8::splat(parent_term);
            let normalized_gain_denom_v = f32x8::splat(normalized_gain_denom);
            let grad_mean_v = f32x8::splat(morph.grad_mean);
            let grad_std_v = f32x8::splat(morph.grad_std + INFO_EPSILON);
            let smoothing_v = f32x8::splat(smoothing);
            let parent_info_v = f32x8::splat(parent_info);
            let gradient_coeff_v = f32x8::splat(morph.precomputed.gradient_score_coeff);
            let info_coeff_v = f32x8::splat(morph.precomputed.info_score_coeff);
            let parent_count_v = f32x8::splat(total_count.max(1) as f32);

            let mut best_gain = 0.0_f32;
            let mut best_threshold = usize::MAX;
            let mut best_default_left = false;
            let mut chunk_start = 0usize;
            while chunk_start < scan_limit {
                let chunk_end = (chunk_start + 8).min(scan_limit);
                let chunk_len = chunk_end - chunk_start;
                let mut left_grad_values = [0.0_f32; 8];
                let mut left_hess_values = [0.0_f32; 8];
                let mut left_count_values = [0.0_f32; 8];
                for lane in 0..chunk_len {
                    let index = chunk_start + lane;
                    left_grad_values[lane] = cumulative_grad[index];
                    left_hess_values[lane] = cumulative_hess[index];
                    left_count_values[lane] = cumulative_count[index] as f32;
                }
                let left_grad_v = f32x8::from(left_grad_values);
                let left_hess_v = f32x8::from(left_hess_values);
                let left_count_v = f32x8::from(left_count_values);
                let right_grad_v = non_missing_grad_v - left_grad_v;
                let right_hess_v = non_missing_hess_v - left_hess_v;
                let right_count_v = non_missing_count_v - left_count_v;

                let evaluate_direction = |default_left: bool| -> [f32; 8] {
                    let (
                        raw_left_grad,
                        left_hess,
                        left_count,
                        raw_right_grad,
                        right_hess,
                        right_count,
                    ) = if default_left {
                        (
                            left_grad_v + missing_grad_v,
                            left_hess_v + missing_hess_v,
                            left_count_v + missing_count_v,
                            right_grad_v,
                            right_hess_v,
                            right_count_v,
                        )
                    } else {
                        (
                            left_grad_v,
                            left_hess_v,
                            left_count_v,
                            right_grad_v + missing_grad_v,
                            right_hess_v + missing_hess_v,
                            right_count_v + missing_count_v,
                        )
                    };
                    let left_gain_gradient = l1_threshold_f32x8(raw_left_grad, options.l1_alpha);
                    let right_gain_gradient = l1_threshold_f32x8(raw_right_grad, options.l1_alpha);
                    let left_denom = left_hess + lambda_v + gain_epsilon_v;
                    let right_denom = right_hess + lambda_v + gain_epsilon_v;
                    let gradient_score = left_gain_gradient * left_gain_gradient / left_denom
                        + right_gain_gradient * right_gain_gradient / right_denom
                        - parent_term_v;

                    let mut gain =
                        if morph.precomputed.in_warmup || morph.precomputed.info_score_negligible {
                            gradient_score
                        } else {
                            let left_info = info_side_simd(
                                left_gain_gradient,
                                left_count,
                                grad_mean_v,
                                grad_std_v,
                                smoothing_v,
                                zero,
                                one,
                            );
                            let right_info = info_side_simd(
                                right_gain_gradient,
                                right_count,
                                grad_mean_v,
                                grad_std_v,
                                smoothing_v,
                                zero,
                                one,
                            );
                            gradient_coeff_v * (gradient_score / normalized_gain_denom_v)
                                + info_coeff_v * (left_info + right_info - parent_info_v)
                        };

                    if !morph.precomputed.in_warmup && morph.precomputed.balance_penalty {
                        let left_is_smaller = left_count.cmp_lt(right_count);
                        let min_count = left_is_smaller.blend(left_count, right_count);
                        let ratio = min_count / parent_count_v;
                        let penalized = ratio.cmp_lt(f32x8::splat(0.1));
                        let adjustment =
                            f32x8::splat(-0.5) * (one - (f32x8::splat(-10.0) * ratio).exp());
                        gain += penalized.blend(adjustment, zero);
                    }

                    let valid = left_count.cmp_ge(min_rows_v)
                        & right_count.cmp_ge(min_rows_v)
                        & left_hess.cmp_gt(min_hessian_v)
                        & right_hess.cmp_gt(min_hessian_v);
                    let valid = if options.min_leaf_magnitude > 0.0 {
                        valid
                            & ((left_gain_gradient.abs() / left_denom).cmp_ge(min_leaf_magnitude_v)
                                | (right_gain_gradient.abs() / right_denom)
                                    .cmp_ge(min_leaf_magnitude_v))
                    } else {
                        valid
                    };
                    valid.blend(gain, negative_infinity).to_array()
                };

                let missing_left_gains = evaluate_direction(true);
                let missing_right_gains = evaluate_direction(false);
                for lane in 0..chunk_len {
                    let threshold = chunk_start + lane;
                    if threshold + 1 >= scan_limit
                        && non_missing_count == cumulative_count[threshold]
                    {
                        continue;
                    }
                    for (default_left, gain) in [
                        (true, missing_left_gains[lane]),
                        (false, missing_right_gains[lane]),
                    ] {
                        if gain.is_finite() && gain_materially_exceeds(gain, best_gain) {
                            best_gain = gain;
                            best_threshold = threshold;
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
            let left_count = cumulative_count[best_threshold];
            let right_grad = non_missing_grad - left_grad;
            let right_hess = non_missing_hess - left_hess;
            let right_count = non_missing_count.saturating_sub(left_count);
            let left_grad_sq = grad_sq_sums.map_or(0.0, |values| {
                values[..=best_threshold].iter().copied().sum()
            });
            let right_grad_sq = total_grad_sq - missing_grad_sq - left_grad_sq;
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

fn info_side_scalar(gradient_sum: f32, count: u32, mean: f32, std: f32, smoothing: f32) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let normalized = (gradient_sum / count as f32 - mean) / (std + INFO_EPSILON);
    normalized.abs() * (1.0 + normalized.abs()).ln() / smoothing
}

#[allow(clippy::too_many_arguments)]
fn info_side_simd(
    gradient_sum: f32x8,
    count: f32x8,
    mean: f32x8,
    std: f32x8,
    smoothing: f32x8,
    zero: f32x8,
    one: f32x8,
) -> f32x8 {
    let positive_count = count.cmp_gt(zero);
    let safe_count = positive_count.blend(count, one);
    let normalized = (gradient_sum / safe_count - mean) / std;
    let magnitude = normalized.abs();
    positive_count.blend(magnitude * (one + magnitude).ln() / smoothing, zero)
}
