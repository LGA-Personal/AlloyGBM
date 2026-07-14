//! Morph gain computation for split scoring.
//!
//! Implements the adaptive split criterion from Kriuk (2025), arXiv:2511.13234.
//! Pre-warmup, this returns the standard XGBoost-style gain. Post-warmup, it
//! blends gradient-based scoring with normalized information-theoretic terms.

use alloygbm_core::{MorphConfig, MorphPrecomputed};

/// Statistics for one side of a split (left, right, or parent).
///
/// `gradient_sum` is the gain signal selected by the caller. Standard and
/// MorphBoost-only training pass the L1-thresholded gradient; DRO+Morph passes
/// the DRO effective gradient so MorphBoost blends its information score with
/// the same robust gradient gain used by scalar leaf solving.
#[derive(Debug, Clone, Copy)]
pub struct SplitSideStats {
    pub gradient_sum: f32,
    pub hessian_sum: f32,
    pub count: u32,
}

/// Morph gain inputs.
#[derive(Debug, Clone, Copy)]
pub struct MorphGainInputs {
    pub left: SplitSideStats,
    pub right: SplitSideStats,
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub lambda_l2: f32,
}

const EPS: f32 = 1e-10;

/// Regularisation epsilon added to hessian denominators in `gradient_gain`,
/// matching the `EPSILON = 1e-6` used in the standard `best_split_for_feature`
/// path. Required for warmup byte-equivalence when `l2_lambda > 0`.
const GAIN_EPSILON: f32 = 1e-6;

/// Compute the morph-augmented split gain.
///
/// At `iteration < morph_warmup_iters`, returns the standard XGBoost gain
/// to allow byte-equivalence with the non-morph path during warmup.
///
/// `pre` carries horizon-scaled blend constants and the warmup-branch flag so
/// the inner per-bin loop avoids recomputing them.
pub fn compute_morph_gain(
    inputs: MorphGainInputs,
    config: &MorphConfig,
    pre: &MorphPrecomputed,
) -> f32 {
    let gradient_score = gradient_gain(&inputs);

    let mut gain = if pre.in_warmup || pre.info_score_negligible {
        gradient_score
    } else {
        let info_score = info_gain(&inputs, config);
        let normalized_gradient_score = normalize_gradient_gain(&inputs, gradient_score);
        pre.gradient_score_coeff * normalized_gradient_score + pre.info_score_coeff * info_score
    };

    if !pre.in_warmup && pre.balance_penalty {
        gain += balance_adjustment(&inputs);
    }

    gain
}

/// Expresses XGBoost gain in the same dimensionless units as the information
/// term: parent curvature times the squared EMA gradient scale. The parent
/// curvature grows with node mass and the EMA scale tracks gradient units, so
/// neither node size nor loss scale silently changes the blend weight.
fn normalize_gradient_gain(inputs: &MorphGainInputs, gradient_score: f32) -> f32 {
    let parent_hessian = inputs.left.hessian_sum + inputs.right.hessian_sum;
    let curvature = (parent_hessian + inputs.lambda_l2).max(GAIN_EPSILON);
    let gradient_scale = inputs.grad_std.abs().max(EPS);
    gradient_score / (curvature * gradient_scale * gradient_scale).max(EPS)
}

fn gradient_gain(inputs: &MorphGainInputs) -> f32 {
    let l = &inputs.left;
    let r = &inputs.right;
    let lambda = inputs.lambda_l2;
    let parent_g = l.gradient_sum + r.gradient_sum;
    let parent_h = l.hessian_sum + r.hessian_sum;
    (l.gradient_sum * l.gradient_sum) / (l.hessian_sum + lambda + GAIN_EPSILON)
        + (r.gradient_sum * r.gradient_sum) / (r.hessian_sum + lambda + GAIN_EPSILON)
        - (parent_g * parent_g) / (parent_h + lambda + GAIN_EPSILON)
}

fn info_gain(inputs: &MorphGainInputs, config: &MorphConfig) -> f32 {
    let smoothing = 1.0
        + config.evolution_pressure * (inputs.iteration as f32 / inputs.total_iterations as f32);
    let info_l = info_side(inputs.left, inputs.grad_mean, inputs.grad_std, smoothing);
    let info_r = info_side(inputs.right, inputs.grad_mean, inputs.grad_std, smoothing);
    let info_parent = info_side(
        SplitSideStats {
            gradient_sum: inputs.left.gradient_sum + inputs.right.gradient_sum,
            hessian_sum: inputs.left.hessian_sum + inputs.right.hessian_sum,
            count: inputs.left.count + inputs.right.count,
        },
        inputs.grad_mean,
        inputs.grad_std,
        smoothing,
    );
    info_l + info_r - info_parent
}

fn info_side(stats: SplitSideStats, mean: f32, std: f32, smoothing: f32) -> f32 {
    if stats.count == 0 {
        return 0.0;
    }
    let g_mean = stats.gradient_sum / stats.count as f32;
    let g_norm = (g_mean - mean) / (std + EPS);
    g_norm.abs() * (1.0 + g_norm.abs()).ln() / smoothing
}

fn balance_adjustment(inputs: &MorphGainInputs) -> f32 {
    let total = inputs.left.count + inputs.right.count;
    if total == 0 {
        return 0.0;
    }
    let min_side = inputs.left.count.min(inputs.right.count);
    let balance_ratio = min_side as f32 / total as f32;
    if balance_ratio >= 0.1 {
        return 0.0;
    }
    -0.5 * (1.0 - (-10.0 * balance_ratio).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{LrSchedule, MorphConfig};

    fn config() -> MorphConfig {
        MorphConfig::default()
    }

    fn balanced_inputs(iteration: u32) -> MorphGainInputs {
        MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 5.0,
                hessian_sum: 4.0,
                count: 50,
            },
            right: SplitSideStats {
                gradient_sum: -5.0,
                hessian_sum: 4.0,
                count: 50,
            },
            iteration,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            lambda_l2: 1.0,
        }
    }

    #[test]
    fn warmup_iteration_returns_pure_gradient_gain() {
        let inputs = balanced_inputs(0);
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let pre = MorphPrecomputed::for_iteration(inputs.iteration, inputs.total_iterations, &cfg);
        let gain = compute_morph_gain(inputs, &cfg, &pre);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }

    #[test]
    fn last_warmup_iteration_returns_pure_gradient_gain() {
        let cfg = config();
        let inputs = balanced_inputs(cfg.morph_warmup_iters - 1);
        let cfg_no_balance = MorphConfig {
            balance_penalty: false,
            ..cfg
        };
        let pre = MorphPrecomputed::for_iteration(
            inputs.iteration,
            inputs.total_iterations,
            &cfg_no_balance,
        );
        let gain = compute_morph_gain(inputs, &cfg_no_balance, &pre);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }

    #[test]
    fn first_post_warmup_iteration_blends() {
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let inputs = balanced_inputs(cfg.morph_warmup_iters);
        let pre = MorphPrecomputed::for_iteration(inputs.iteration, inputs.total_iterations, &cfg);
        let gain = compute_morph_gain(inputs, &cfg, &pre);
        let pure_gradient = gradient_gain(&inputs);
        // After warmup, gain should differ from pure gradient.
        assert!((gain - pure_gradient).abs() > 1e-9);
    }

    #[test]
    fn morph_weight_grows_with_iteration() {
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let early = balanced_inputs(cfg.morph_warmup_iters);
        let late = balanced_inputs(50);
        let pre_early =
            MorphPrecomputed::for_iteration(early.iteration, early.total_iterations, &cfg);
        let pre_late = MorphPrecomputed::for_iteration(late.iteration, late.total_iterations, &cfg);
        let gain_early = compute_morph_gain(early, &cfg, &pre_early);
        let gain_late = compute_morph_gain(late, &cfg, &pre_late);
        // Different iterations produce different gains.
        assert!((gain_early - gain_late).abs() > 1e-9);
    }

    #[test]
    fn balance_penalty_reduces_gain_for_unbalanced_split() {
        let unbalanced = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 1.0,
                hessian_sum: 1.0,
                count: 5,
            },
            right: SplitSideStats {
                gradient_sum: -1.0,
                hessian_sum: 1.0,
                count: 95,
            },
            iteration: 5,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            lambda_l2: 1.0,
        };
        let cfg_off = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let cfg_on = MorphConfig {
            balance_penalty: true,
            ..config()
        };
        let pre_off = MorphPrecomputed::for_iteration(
            unbalanced.iteration,
            unbalanced.total_iterations,
            &cfg_off,
        );
        let pre_on = MorphPrecomputed::for_iteration(
            unbalanced.iteration,
            unbalanced.total_iterations,
            &cfg_on,
        );
        let gain_off = compute_morph_gain(unbalanced, &cfg_off, &pre_off);
        let gain_on = compute_morph_gain(unbalanced, &cfg_on, &pre_on);
        assert!(gain_on < gain_off);
    }

    #[test]
    fn balance_penalty_zero_for_balanced_split() {
        let balanced = balanced_inputs(0);
        let cfg = config();
        let cfg_no_balance = MorphConfig {
            balance_penalty: false,
            ..cfg
        };
        let pre =
            MorphPrecomputed::for_iteration(balanced.iteration, balanced.total_iterations, &cfg);
        let pre_nb = MorphPrecomputed::for_iteration(
            balanced.iteration,
            balanced.total_iterations,
            &cfg_no_balance,
        );
        let with_pen = compute_morph_gain(balanced, &cfg, &pre);
        let no_pen = compute_morph_gain(balanced, &cfg_no_balance, &pre_nb);
        assert!((with_pen - no_pen).abs() < 1e-6);
    }

    #[test]
    fn lr_schedule_is_carried_through_unchanged() {
        // Smoke test that LrSchedule on config doesn't alter gain math.
        let cfg = MorphConfig {
            lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.1 },
            balance_penalty: false,
            ..config()
        };
        let inputs = balanced_inputs(0);
        let pre = MorphPrecomputed::for_iteration(inputs.iteration, inputs.total_iterations, &cfg);
        let gain = compute_morph_gain(inputs, &cfg, &pre);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }

    #[test]
    fn precomputed_warmup_branch() {
        let cfg = MorphConfig {
            morph_warmup_iters: 5,
            info_score_weight: 0.3,
            depth_penalty_base: 0.9,
            balance_penalty: false,
            morph_rate: 0.1,
            evolution_pressure: 0.2,
            lr_schedule: LrSchedule::Constant,
        };
        let pre = MorphPrecomputed::for_iteration(2, 100, &cfg);
        assert!(pre.in_warmup);
        assert!(pre.info_score_negligible);
        assert_eq!(pre.gradient_score_coeff, 1.0);
    }

    #[test]
    fn precomputed_post_warmup_branch() {
        let cfg = MorphConfig {
            morph_warmup_iters: 5,
            info_score_weight: 0.3,
            depth_penalty_base: 0.9,
            balance_penalty: true,
            morph_rate: 0.1,
            evolution_pressure: 0.2,
            lr_schedule: LrSchedule::Constant,
        };
        let pre = MorphPrecomputed::for_iteration(20, 100, &cfg);
        assert!(!pre.in_warmup);
        let expected_weight = (3.0_f32 * 20.0 / 100.0).tanh();
        assert!((pre.morph_weight - expected_weight).abs() < 1e-6);
        assert!((pre.gradient_score_coeff - (1.0 - 0.3 * expected_weight)).abs() < 1e-6);
        assert!((pre.info_score_coeff - 0.3 * expected_weight).abs() < 1e-6);
        assert!(pre.balance_penalty);
        assert!(!pre.info_score_negligible);
    }

    #[test]
    fn post_warmup_blend_coefficients_sum_to_one() {
        let cfg = MorphConfig {
            morph_warmup_iters: 5,
            info_score_weight: 0.3,
            ..config()
        };
        let pre = MorphPrecomputed::for_iteration(20, 100, &cfg);

        assert!(
            (pre.gradient_score_coeff + pre.info_score_coeff - 1.0).abs() < 1e-6,
            "post-warmup coefficients must form a convex blend"
        );
    }

    #[test]
    fn morph_ramp_uses_fit_horizon() {
        let cfg = MorphConfig {
            morph_warmup_iters: 0,
            ..config()
        };
        let quarter_of_short_fit = MorphPrecomputed::for_iteration(25, 100, &cfg);
        let quarter_of_long_fit = MorphPrecomputed::for_iteration(250, 1_000, &cfg);
        let same_round_of_long_fit = MorphPrecomputed::for_iteration(25, 1_000, &cfg);

        assert!(
            (quarter_of_short_fit.morph_weight - quarter_of_long_fit.morph_weight).abs() < 1e-6,
            "identical fit progress must receive the same morph ramp"
        );
        assert!(
            quarter_of_short_fit.morph_weight > same_round_of_long_fit.morph_weight,
            "the same absolute round must ramp in more slowly on a longer fit"
        );
    }

    #[test]
    fn warmup_ignores_balance_penalty() {
        let inputs = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 1.0,
                hessian_sum: 1.0,
                count: 5,
            },
            right: SplitSideStats {
                gradient_sum: -1.0,
                hessian_sum: 1.0,
                count: 95,
            },
            iteration: 0,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            lambda_l2: 1.0,
        };
        let cfg = MorphConfig::default();
        let pre = MorphPrecomputed::for_iteration(inputs.iteration, inputs.total_iterations, &cfg);

        assert!((compute_morph_gain(inputs, &cfg, &pre) - gradient_gain(&inputs)).abs() < 1e-6);
    }

    #[test]
    fn post_warmup_gain_is_invariant_to_uniform_node_duplication() {
        let cfg = MorphConfig {
            morph_warmup_iters: 0,
            balance_penalty: false,
            ..config()
        };
        let base = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 8.0,
                hessian_sum: 12.0,
                count: 40,
            },
            right: SplitSideStats {
                gradient_sum: -4.0,
                hessian_sum: 18.0,
                count: 60,
            },
            iteration: 30,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 0.5,
            lambda_l2: 0.0,
        };
        let duplicated = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: base.left.gradient_sum * 2.0,
                hessian_sum: base.left.hessian_sum * 2.0,
                count: base.left.count * 2,
            },
            right: SplitSideStats {
                gradient_sum: base.right.gradient_sum * 2.0,
                hessian_sum: base.right.hessian_sum * 2.0,
                count: base.right.count * 2,
            },
            ..base
        };
        let pre = MorphPrecomputed::for_iteration(base.iteration, base.total_iterations, &cfg);

        let base_gain = compute_morph_gain(base, &cfg, &pre);
        let duplicated_gain = compute_morph_gain(duplicated, &cfg, &pre);
        assert!(
            (base_gain - duplicated_gain).abs() < 1e-6,
            "post-warmup blend must not change just because a node is duplicated: \
             base={base_gain}, duplicated={duplicated_gain}"
        );
    }

    #[test]
    fn post_warmup_gain_is_invariant_to_gradient_scale() {
        let cfg = MorphConfig {
            morph_warmup_iters: 0,
            balance_penalty: false,
            ..config()
        };
        let base = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 3.0,
                hessian_sum: 12.0,
                count: 40,
            },
            right: SplitSideStats {
                gradient_sum: -1.0,
                hessian_sum: 18.0,
                count: 60,
            },
            iteration: 30,
            total_iterations: 100,
            grad_mean: 0.1,
            grad_std: 0.5,
            lambda_l2: 0.0,
        };
        let scaled = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: base.left.gradient_sum * 10.0,
                ..base.left
            },
            right: SplitSideStats {
                gradient_sum: base.right.gradient_sum * 10.0,
                ..base.right
            },
            grad_mean: base.grad_mean * 10.0,
            grad_std: base.grad_std * 10.0,
            ..base
        };
        let pre = MorphPrecomputed::for_iteration(base.iteration, base.total_iterations, &cfg);

        let base_gain = compute_morph_gain(base, &cfg, &pre);
        let scaled_gain = compute_morph_gain(scaled, &cfg, &pre);
        assert!(
            (base_gain - scaled_gain).abs() < 1e-6,
            "post-warmup blend must be independent of gradient units: \
             base={base_gain}, scaled={scaled_gain}"
        );
    }

    #[test]
    fn precomputed_negligible_when_info_weight_zero() {
        let cfg = MorphConfig {
            morph_warmup_iters: 5,
            info_score_weight: 0.0,
            depth_penalty_base: 0.9,
            balance_penalty: false,
            morph_rate: 0.1,
            evolution_pressure: 0.2,
            lr_schedule: LrSchedule::Constant,
        };
        let pre = MorphPrecomputed::for_iteration(20, 100, &cfg);
        assert!(!pre.in_warmup);
        assert!(pre.info_score_negligible);
    }
}
