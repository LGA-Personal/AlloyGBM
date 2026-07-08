use super::*;
use alloygbm_core::{LrSchedule, MorphConfig};

#[test]
fn constant_schedule_produces_uniform_lr() {
    let lrs = resolve_lr_schedule(LrSchedule::Constant, 10, 0.1);
    assert_eq!(lrs.len(), 10);
    for lr in &lrs {
        assert!((*lr - 0.1).abs() < 1e-6, "lr={lr} expected 0.1");
    }
}

#[test]
fn warmup_cosine_starts_low_peaks_at_base_then_decays() {
    let lrs = resolve_lr_schedule(LrSchedule::WarmupCosine { warmup_frac: 0.1 }, 100, 0.1);
    assert_eq!(lrs.len(), 100);
    assert!(lrs[0] < 0.1, "first lr={} should be below base 0.1", lrs[0]);
    let peak_idx = (0.1_f32 * 100.0).round() as usize - 1;
    assert!(
        (lrs[peak_idx] - 0.1).abs() < 1e-3,
        "peak lr={} should be near base_lr 0.1",
        lrs[peak_idx]
    );
    assert!(
        lrs[99] < 0.05,
        "final lr={} should decay toward floor",
        lrs[99]
    );
}

#[test]
fn warmup_cosine_full_warmup_edge_case() {
    // warmup_frac=1.0 means entire schedule is warmup; no cosine decay.
    let lrs = resolve_lr_schedule(LrSchedule::WarmupCosine { warmup_frac: 1.0 }, 5, 0.1);
    assert_eq!(lrs.len(), 5);
    // Last warmup step should hit base_lr.
    assert!((lrs[4] - 0.1).abs() < 1e-5);
}

#[test]
fn resolve_lr_schedule_zero_estimators_returns_empty() {
    let lrs = resolve_lr_schedule(LrSchedule::Constant, 0, 0.1);
    assert!(lrs.is_empty());
}

#[test]
fn morph_state_new_correct_dimensions() {
    let cfg = MorphConfig::default();
    let ms = MorphState::new(cfg, 3, 50, 0.05);
    assert_eq!(ms.ema_stats.len(), 3);
    assert_eq!(ms.lr_per_iter.len(), 50);
}

#[test]
fn morph_state_single_class() {
    let cfg = MorphConfig::default();
    let ms = MorphState::new(cfg, 1, 10, 0.1);
    assert_eq!(ms.ema_stats.len(), 1);
    let ctx = ms.morph_context(0, 10, 0);
    assert_eq!(ctx.iteration, 0);
    assert_eq!(ctx.total_iterations, 10);
}

#[test]
fn morph_state_lr_for_iter_clamps_to_last() {
    let cfg = MorphConfig::default();
    let ms = MorphState::new(cfg, 1, 5, 0.1);
    // Out-of-bounds index returns last value.
    let last = ms.lr_per_iter[4];
    assert!((ms.lr_for_iter(999) - last).abs() < 1e-6);
}

#[test]
fn morph_leaf_scale_matches_schedule_shrinkage_and_depth_penalty() {
    let cfg = MorphConfig {
        morph_warmup_iters: 5,
        info_score_weight: 0.3,
        depth_penalty_base: 0.729,
        balance_penalty: false,
        morph_rate: 0.4,
        evolution_pressure: 0.2,
        lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.25 },
    };
    let state = MorphState::new(cfg, 1, 20, 0.2);
    let scale = state.leaf_scale_for_depth(5, 20, 6);
    let expected_lr = state.lr_for_iter(5);
    let expected_shrinkage = 1.0 - cfg.morph_rate * (5.0_f32 / 20.0);
    let expected_depth_penalty = cfg.depth_penalty_base.powf(6.0 / 3.0);

    assert!((scale.total - expected_lr * expected_shrinkage * expected_depth_penalty).abs() < 1e-6);
    assert!((scale.multiplier - expected_shrinkage * expected_depth_penalty).abs() < 1e-6);
}

#[test]
fn morph_state_lr_threshold_scale_constant_lr() {
    let cfg = MorphConfig {
        morph_warmup_iters: 5,
        info_score_weight: 0.3,
        depth_penalty_base: 0.9,
        balance_penalty: false,
        morph_rate: 0.1,
        evolution_pressure: 0.2,
        lr_schedule: LrSchedule::Constant,
    };
    let state = MorphState::new(cfg, 1, 100, 0.05);
    // Constant schedule: scale must be 1.0 at all iterations.
    for iter in [0, 50, 99] {
        assert!(
            (state.lr_loss_threshold_scale(iter) - 1.0).abs() < 1e-6,
            "constant schedule must yield scale=1.0 at iter {iter}",
        );
    }
}

#[test]
fn morph_state_lr_threshold_scale_warmup_cosine() {
    let cfg = MorphConfig {
        morph_warmup_iters: 5,
        info_score_weight: 0.3,
        depth_penalty_base: 0.9,
        balance_penalty: false,
        morph_rate: 0.1,
        evolution_pressure: 0.2,
        lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.1 },
    };
    let n = 5000;
    let base_lr = 0.01_f32;
    let state = MorphState::new(cfg, 1, n, base_lr);
    // Round 0: lr ≈ 2e-5, max ≈ 1e-2 → scale ≈ 0.002, clamped up to 0.01.
    let scale_warmup = state.lr_loss_threshold_scale(0);
    assert!(
        (0.0099..=0.011).contains(&scale_warmup),
        "warmup scale should be clamped to ~0.01, got {scale_warmup}",
    );
    // Around peak (round 499): scale should be ~1.0.
    let scale_peak = state.lr_loss_threshold_scale(499);
    assert!(
        (scale_peak - 1.0).abs() < 0.01,
        "peak scale should be ~1.0, got {scale_peak}",
    );
    // Final round: scale should be small (decayed cosine).
    let scale_final = state.lr_loss_threshold_scale((n as usize) - 1);
    assert!(
        scale_final < 0.5,
        "final-decay scale should be small, got {scale_final}"
    );
    // Lower clamp guarantees no value below 0.01.
    assert!(
        scale_final >= 0.01,
        "scale must respect lower clamp, got {scale_final}"
    );
}

#[test]
fn morph_state_lr_threshold_scale_handles_degenerate_schedule() {
    let cfg = MorphConfig {
        morph_warmup_iters: 0,
        info_score_weight: 0.0,
        depth_penalty_base: 1.0,
        balance_penalty: false,
        morph_rate: 0.0,
        evolution_pressure: 0.0,
        lr_schedule: LrSchedule::Constant,
    };
    // n=1 schedule, base_lr should still produce scale=1.0 (no division-by-zero).
    let state = MorphState::new(cfg, 1, 1, 0.05);
    assert!((state.lr_loss_threshold_scale(0) - 1.0).abs() < 1e-6);
}

#[test]
fn morph_state_is_in_warmup_phase_constant_schedule() {
    use alloygbm_core::{LrSchedule, MorphConfig};
    let cfg = MorphConfig {
        morph_warmup_iters: 5,
        info_score_weight: 0.3,
        depth_penalty_base: 0.9,
        balance_penalty: false,
        morph_rate: 0.1,
        evolution_pressure: 0.2,
        lr_schedule: LrSchedule::Constant,
    };
    let state = MorphState::new(cfg, 1, 100, 0.05);
    // Constant schedule: never in warmup.
    for iter in [0, 50, 99] {
        assert!(
            !state.is_in_warmup_phase(iter),
            "constant schedule must report not-in-warmup at iter {iter}"
        );
    }
}

#[test]
fn morph_state_is_in_warmup_phase_warmup_cosine_boundary() {
    use alloygbm_core::{LrSchedule, MorphConfig};
    let cfg = MorphConfig {
        morph_warmup_iters: 5,
        info_score_weight: 0.3,
        depth_penalty_base: 0.9,
        balance_penalty: false,
        morph_rate: 0.1,
        evolution_pressure: 0.2,
        lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.1 },
    };
    // n=5000, warmup_frac=0.1 → warmup spans [0, 500).
    let state = MorphState::new(cfg, 1, 5000, 0.01);
    assert!(state.is_in_warmup_phase(0));
    assert!(state.is_in_warmup_phase(499));
    assert!(!state.is_in_warmup_phase(500));
    assert!(!state.is_in_warmup_phase(2500));
    assert!(!state.is_in_warmup_phase(4999));
}

#[test]
fn morph_state_is_in_warmup_phase_handles_degenerate_schedules() {
    use alloygbm_core::{LrSchedule, MorphConfig};
    // n=1 with WarmupCosine → warmup is exactly 1 round (round 0 only).
    let cfg = MorphConfig {
        morph_warmup_iters: 0,
        info_score_weight: 0.0,
        depth_penalty_base: 1.0,
        balance_penalty: false,
        morph_rate: 0.0,
        evolution_pressure: 0.0,
        lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.5 },
    };
    let state = MorphState::new(cfg, 1, 1, 0.01);
    assert!(state.is_in_warmup_phase(0));
    assert!(!state.is_in_warmup_phase(1));
}

// ── IterationDiagnostics tests ──────────────────────────────────────────

#[test]
fn iteration_diagnostics_from_gradient_snapshot_without_projection_omits_effectiveness() {
    let grads = vec![
        GradientPair {
            grad: 1.0,
            hess: 1.0,
        },
        GradientPair {
            grad: -2.0,
            hess: 0.5,
        },
        GradientPair {
            grad: 0.0,
            hess: 1.0,
        },
    ];
    let d = IterationDiagnostics::from_gradient_snapshot(&grads, None, 3, 1);
    // ||g||_2 = sqrt(1 + 4 + 0) = sqrt(5)
    assert!((d.gradient_l2_norm - 5.0_f32.sqrt()).abs() < 1e-5);
    assert!(d.gradient_variance >= 0.0);
    // No projection configured → both pre and post stay None.
    assert!(d.original_gradient_l2_norm.is_none());
    assert!(d.projected_gradient_l2_norm.is_none());
    assert!(d.neutralization_effectiveness.is_none());
    assert_eq!(d.n_active_rows, 3);
    assert_eq!(d.n_active_features, 1);
}

#[test]
fn iteration_diagnostics_effectiveness_in_unit_interval_when_projection_active() {
    // Simulate: pre-projection ||g|| = 4.0, post-projection ||g|| = 1.0.
    // Effectiveness = 1 - 1/4 = 0.75.
    let post = vec![GradientPair {
        grad: 1.0,
        hess: 0.0,
    }];
    let d = IterationDiagnostics::from_gradient_snapshot(&post, Some(4.0), 1, 1);
    assert_eq!(d.original_gradient_l2_norm, Some(4.0));
    assert_eq!(d.projected_gradient_l2_norm, Some(1.0));
    let eff = d
        .neutralization_effectiveness
        .expect("effectiveness present");
    assert!((eff - 0.75).abs() < 1e-5, "expected 0.75, got {eff}");
    assert!((0.0..=1.0).contains(&eff));
}

#[test]
fn iteration_diagnostics_aggregate_per_class_takes_max_effectiveness() {
    // Three classes: two with effectiveness, one without.  Aggregation
    // should report the maximum effectiveness across classes.
    let a = IterationDiagnostics {
        gradient_l2_norm: 1.0,
        gradient_variance: 0.0,
        hessian_l2_norm: 1.0,
        original_gradient_l2_norm: Some(2.0),
        projected_gradient_l2_norm: Some(1.0),
        neutralization_effectiveness: Some(0.5),
        n_active_rows: 10,
        n_active_features: 2,
    };
    let b = IterationDiagnostics {
        gradient_l2_norm: 2.0,
        gradient_variance: 0.0,
        hessian_l2_norm: 1.0,
        original_gradient_l2_norm: Some(4.0),
        projected_gradient_l2_norm: Some(1.0),
        neutralization_effectiveness: Some(0.75),
        n_active_rows: 10,
        n_active_features: 2,
    };
    let c = IterationDiagnostics {
        gradient_l2_norm: 3.0,
        gradient_variance: 0.0,
        hessian_l2_norm: 1.0,
        ..Default::default()
    };
    let agg = IterationDiagnostics::aggregate_per_class(&[a, b, c]);
    assert!((agg.gradient_l2_norm - 2.0).abs() < 1e-5); // mean(1, 2, 3) = 2
    assert_eq!(agg.neutralization_effectiveness, Some(0.75));
    assert_eq!(agg.n_active_rows, 10);
}

// ── GLM objectives (Poisson, Gamma, Tweedie) — v0.11.0 ─────────────

#[test]
fn poisson_initial_prediction_is_log_of_weighted_mean() {
    let targets = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let init = PoissonObjective::default()
        .initial_prediction(&targets, None)
        .expect("init");
    let expected = ((1.0_f32 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0).ln();
    assert!((init - expected).abs() < 1e-6, "got={init} want={expected}");
}

#[test]
fn glm_initial_prediction_accumulates_mean_in_f64() {
    let mut targets = Vec::with_capacity(1_001);
    targets.push(100_000_000.0_f32);
    targets.extend(std::iter::repeat_n(1.0_f32, 1_000));
    let expected_mean = (100_000_000.0_f64 + 1_000.0) / 1_001.0;

    let poisson_init = PoissonObjective::default()
        .initial_prediction(&targets, None)
        .expect("poisson init");
    let gamma_init = GammaObjective
        .initial_prediction(&targets, None)
        .expect("gamma init");
    let tweedie_init = TweedieObjective::new(1.5)
        .expect("tweedie")
        .initial_prediction(&targets, None)
        .expect("tweedie init");

    for init in [poisson_init, gamma_init, tweedie_init] {
        let mean = f64::from(init).exp();
        assert!(
            (mean - expected_mean).abs() < 0.25,
            "mean={mean} expected={expected_mean}"
        );
    }
}

#[test]
fn poisson_gradient_uses_stabilized_hessian() {
    let predictions = vec![0.0_f32, 1.0, 2.0];
    let targets = vec![1.0_f32, 2.0, 3.0];
    let gradients = PoissonObjective::new(0.7)
        .compute_gradients(&predictions, &targets, None)
        .expect("grads");
    for (idx, gp) in gradients.iter().enumerate() {
        let mu = predictions[idx].exp();
        let want_grad = mu - targets[idx];
        let want_hess = mu * 0.7_f32.exp();
        assert!((gp.grad - want_grad).abs() < 1e-5);
        assert!((gp.hess - want_hess).abs() < 1e-5);
    }
}

#[test]
fn poisson_gradient_hessian_stabilizer_is_tunable() {
    let predictions = vec![0.0_f32, 1.0, 2.0];
    let targets = vec![1.0_f32, 2.0, 3.0];
    let gradients = PoissonObjective::new(0.25)
        .compute_gradients(&predictions, &targets, None)
        .expect("grads");
    for (idx, gp) in gradients.iter().enumerate() {
        let mu = predictions[idx].exp();
        let want_hess = mu * 0.25_f32.exp();
        assert!((gp.hess - want_hess).abs() < 1e-5);
    }
}

#[test]
fn glm_losses_accumulate_in_f64() {
    let mut predictions = Vec::with_capacity(1_001);
    predictions.push(100_000_000.0_f32.ln());
    predictions.extend(std::iter::repeat_n(0.0_f32, 1_000));
    let targets = vec![0.0_f32; predictions.len()];

    let poisson_loss = PoissonObjective::default()
        .loss(&predictions, &targets, None)
        .expect("poisson loss");
    let poisson_expected = (100_000_000.0_f64 + 1_000.0) / 1_001.0;
    assert!(
        (f64::from(poisson_loss) - poisson_expected).abs() < 0.05,
        "poisson_loss={poisson_loss} expected={poisson_expected}"
    );

    let mut gamma_targets = Vec::with_capacity(1_001);
    gamma_targets.push(100_000_000.0_f32);
    gamma_targets.extend(std::iter::repeat_n(2.0_f32, 1_000));
    let gamma_predictions = vec![0.0_f32; gamma_targets.len()];
    let gamma_loss = GammaObjective
        .loss(&gamma_predictions, &gamma_targets, None)
        .expect("gamma loss");
    let gamma_big = 100_000_000.0_f64 - 100_000_000.0_f64.ln() - 1.0;
    let gamma_small = 2.0_f64 - 2.0_f64.ln() - 1.0;
    let gamma_expected = (gamma_big + 1_000.0 * gamma_small) / 1_001.0;
    assert!(
        (f64::from(gamma_loss) - gamma_expected).abs() < 0.05,
        "gamma_loss={gamma_loss} expected={gamma_expected}"
    );

    let tweedie = TweedieObjective::new(1.5).expect("tweedie");
    let tweedie_loss = tweedie
        .loss(&gamma_predictions, &gamma_targets, None)
        .expect("tweedie loss");
    let tweedie_term = |y: f64| {
        let p = 1.5_f64;
        let term1 = y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
        let term2 = y / (1.0 - p);
        let term3 = 1.0 / (2.0 - p);
        2.0 * (term1 - term2 + term3)
    };
    let tweedie_expected = (tweedie_term(100_000_000.0) + 1_000.0 * tweedie_term(2.0)) / 1_001.0;
    assert!(
        (f64::from(tweedie_loss) - tweedie_expected).abs() < 0.1,
        "tweedie_loss={tweedie_loss} expected={tweedie_expected}"
    );
}

#[test]
fn poisson_rejects_negative_targets() {
    let targets = vec![1.0_f32, -0.5, 2.0];
    let err = PoissonObjective::default()
        .initial_prediction(&targets, None)
        .unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("non-negative"));
}

#[test]
fn gamma_initial_prediction_is_log_of_weighted_mean() {
    let targets = vec![1.0_f32, 2.0, 4.0, 8.0];
    let init = GammaObjective
        .initial_prediction(&targets, None)
        .expect("init");
    let expected = ((1.0_f32 + 2.0 + 4.0 + 8.0) / 4.0).ln();
    assert!((init - expected).abs() < 1e-6);
}

#[test]
fn gamma_gradient_uses_one_minus_y_over_mu() {
    let predictions = vec![0.0_f32, 1.0];
    let targets = vec![1.0_f32, 4.0];
    let gradients = GammaObjective
        .compute_gradients(&predictions, &targets, None)
        .expect("grads");
    for (idx, gp) in gradients.iter().enumerate() {
        let mu = predictions[idx].exp();
        let want_grad = 1.0 - targets[idx] / mu;
        let want_hess = targets[idx] / mu;
        assert!((gp.grad - want_grad).abs() < 1e-5);
        assert!((gp.hess - want_hess).abs() < 1e-5);
    }
}

#[test]
fn gamma_rejects_zero_or_negative_targets() {
    let err = GammaObjective
        .initial_prediction(&[1.0_f32, 0.0, 3.0], None)
        .unwrap_err();
    let msg = format!("{err:?}").to_lowercase();
    assert!(msg.contains("strictly positive") || msg.contains("> 0"));
}

#[test]
fn tweedie_rejects_invalid_variance_power() {
    assert!(TweedieObjective::new(0.5).is_err());
    assert!(TweedieObjective::new(1.0).is_err());
    assert!(TweedieObjective::new(2.0).is_err());
    assert!(TweedieObjective::new(2.5).is_err());
    assert!(TweedieObjective::new(1.5).is_ok());
}

#[test]
fn tweedie_initial_prediction_log_of_weighted_mean() {
    let targets = vec![0.0_f32, 0.0, 1.0, 2.0, 3.0];
    let init = TweedieObjective::new(1.5)
        .expect("construct")
        .initial_prediction(&targets, None)
        .expect("init");
    let expected = ((0.0_f32 + 0.0 + 1.0 + 2.0 + 3.0) / 5.0).max(1e-7).ln();
    assert!((init - expected).abs() < 1e-6);
}

#[test]
fn tweedie_gradient_uses_power_formula() {
    let p = 1.5_f32;
    let obj = TweedieObjective::new(p).expect("ok");
    let predictions = vec![0.0_f32, 1.0];
    let targets = vec![0.0_f32, 2.0];
    let gradients = obj
        .compute_gradients(&predictions, &targets, None)
        .expect("grads");
    for (idx, gp) in gradients.iter().enumerate() {
        let mu = predictions[idx].exp();
        let want_grad = mu.powf(2.0 - p) - targets[idx] * mu.powf(1.0 - p);
        let want_hess = mu.powf(2.0 - p);
        assert!(
            (gp.grad - want_grad).abs() < 1e-4,
            "idx={idx} grad got={} want={want_grad}",
            gp.grad
        );
        assert!(
            (gp.hess - want_hess).abs() < 1e-4,
            "idx={idx} hess got={} want={want_hess}",
            gp.hess
        );
    }
}

#[test]
fn tweedie_rejects_negative_targets() {
    let err = TweedieObjective::new(1.5)
        .expect("ok")
        .initial_prediction(&[1.0_f32, -0.5], None)
        .unwrap_err();
    assert!(format!("{err:?}").to_lowercase().contains("non-negative"));
}
