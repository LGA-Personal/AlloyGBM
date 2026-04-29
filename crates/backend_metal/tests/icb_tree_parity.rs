//! Parity tests: ICB GPU tree vs CPU baseline.
//!
//! All four tests build the same dataset via MetalBackend (which routes to the
//! ICB path on Metal 4) and via CpuBackend, then compare tree structure and
//! final predictions.  Tests skip silently on non-Metal4 hosts.

#[cfg(target_os = "macos")]
mod tests {
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_backend_metal::MetalBackend;
    use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, TrainParams};
    use alloygbm_engine::{
        BackendOps, IterationControls, IterationStopReason,
        SplitSelectionOptions, TrainedStump,
    };

    /// Build a reproducible (BinnedMatrix, Vec<GradientPair>) pair.
    /// Coprime-stride bin pattern breaks symmetry across features so GPU
    /// tie-breaking is deterministic.
    fn make_fixture(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
    ) -> (BinnedMatrix, Vec<GradientPair>) {
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| {
                let r = i / feature_count;
                let f = i % feature_count;
                let stride = 2 * f + 1; // odd → coprime with any power of 2
                (r.wrapping_mul(stride) % max_bin as usize) as u8
            })
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair {
                grad: (i as i32 - row_count as i32 / 2) as f32 * 0.5,
                hess: 1.0,
            })
            .collect();
        (bm, grads)
    }

    fn default_controls() -> IterationControls {
        IterationControls {
            rounds:                            1,
            min_split_gain:                    0.0,
            min_rows_per_leaf:                 1,
            min_abs_leaf_value:                0.0,
            max_abs_leaf_value:                f32::MAX,
            min_loss_improvement:              0.0,
            max_consecutive_weak_improvements: usize::MAX,
            row_subsample:                     1.0,
            col_subsample:                     1.0,
            early_stopping_rounds:             None,
            min_validation_improvement:        0.0,
            max_leaves:                        None,
        }
    }

    fn default_params(max_depth: u16) -> TrainParams {
        TrainParams {
            max_depth,
            learning_rate: 0.1,
            lambda_l2: 0.1,
            ..TrainParams::default()
        }
    }

    fn split_opts_for(params: &TrainParams) -> SplitSelectionOptions {
        SplitSelectionOptions {
            l2_lambda: params.lambda_l2,
            ..SplitSelectionOptions::default()
        }
    }

    /// Drive the CPU baseline — returns (stumps, preds).
    fn cpu_tree(
        bm: &BinnedMatrix,
        grads: &[GradientPair],
        root_rows: Vec<u32>,
        params: &TrainParams,
        preds: &mut [f32],
    ) -> (Vec<TrainedStump>, IterationStopReason) {
        let controls = default_controls();
        let fc = bm.feature_count as u32;
        let tiles = vec![FeatureTile { start_feature: 0, end_feature: fc }];
        let split_opts = split_opts_for(params);
        alloygbm_engine::build_tree_level_wise_for_test(
            &CpuBackend,
            bm, grads, root_rows, 0, &tiles, split_opts,
            params, &controls, preds, &[], &[],
        )
        .unwrap()
    }

    /// Compare prediction arrays — asserts max abs error < `atol`.
    fn assert_preds_close(metal: &[f32], cpu: &[f32], atol: f32, label: &str) {
        let max_err = metal.iter().zip(cpu.iter())
            .map(|(m, c)| (m - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < atol,
            "{label}: max prediction error {max_err:.6} exceeds atol {atol}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: small dataset, shallow tree (d=4)
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn icb_tree_matches_cpu_small() {
        let Ok(metal) = MetalBackend::new() else { return; };
        if !metal.metal_device.capabilities.metal4 { return; }

        let (bm, grads) = make_fixture(50_000, 20, 64);
        let params      = default_params(4);
        let controls    = default_controls();
        let root_rows: Vec<u32> = (0..50_000u32).collect();
        let tiles = vec![FeatureTile { start_feature: 0, end_feature: 20 }];
        let split_opts = split_opts_for(&params);

        let mut metal_preds = vec![0.0f32; 50_000];
        let mut cpu_preds   = vec![0.0f32; 50_000];

        let Some((metal_stumps, _)) = metal.try_build_tree_level_wise(
            &bm, &grads, &root_rows, 0, &tiles, split_opts,
            &params, &controls, &mut metal_preds, &[], &[],
        ).unwrap() else {
            // ICB path disabled in Stage 5 — skip.
            return;
        };

        let (cpu_stumps, _) = cpu_tree(&bm, &grads, root_rows, &params, &mut cpu_preds);

        assert_eq!(metal_stumps.len(), cpu_stumps.len(), "small: stump count");
        assert_preds_close(&metal_preds, &cpu_preds, 0.05, "small");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: larger dataset, full depth (d=8) — exercises all 8 levels
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn icb_tree_matches_cpu_deep() {
        let Ok(metal) = MetalBackend::new() else { return; };
        if !metal.metal_device.capabilities.metal4 { return; }

        let (bm, grads) = make_fixture(200_000, 50, 127);
        let params      = default_params(8);
        let controls    = default_controls();
        let root_rows: Vec<u32> = (0..200_000u32).collect();
        let tiles = vec![FeatureTile { start_feature: 0, end_feature: 50 }];
        let split_opts = split_opts_for(&params);

        let mut metal_preds = vec![0.0f32; 200_000];
        let mut cpu_preds   = vec![0.0f32; 200_000];

        let Some((metal_stumps, _)) = metal.try_build_tree_level_wise(
            &bm, &grads, &root_rows, 0, &tiles, split_opts,
            &params, &controls, &mut metal_preds, &[], &[],
        ).unwrap() else {
            // ICB path disabled in Stage 5 — skip.
            return;
        };

        let (cpu_stumps, _) = cpu_tree(&bm, &grads, root_rows, &params, &mut cpu_preds);

        assert_eq!(metal_stumps.len(), cpu_stumps.len(), "deep: stump count");
        assert_preds_close(&metal_preds, &cpu_preds, 0.1, "deep");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: high min_split_gain — most nodes pruned to leaves
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn icb_tree_prunes_correctly() {
        let Ok(metal) = MetalBackend::new() else { return; };
        if !metal.metal_device.capabilities.metal4 { return; }

        let (bm, grads) = make_fixture(5_000, 10, 16);
        let mut params = default_params(4);
        params.min_split_gain = 1e4; // forces most nodes to be leaves
        let controls  = default_controls();
        let root_rows: Vec<u32> = (0..5_000u32).collect();
        let tiles = vec![FeatureTile { start_feature: 0, end_feature: 10 }];
        let split_opts = split_opts_for(&params);

        let mut metal_preds = vec![0.0f32; 5_000];
        let mut cpu_preds   = vec![0.0f32; 5_000];

        let Some((metal_stumps, _)) = metal.try_build_tree_level_wise(
            &bm, &grads, &root_rows, 0, &tiles, split_opts,
            &params, &controls, &mut metal_preds, &[], &[],
        ).unwrap() else {
            // ICB path disabled in Stage 5 — skip.
            return;
        };

        let (cpu_stumps, _) = cpu_tree(&bm, &grads, root_rows, &params, &mut cpu_preds);

        assert_eq!(
            metal_stumps.len(), cpu_stumps.len(),
            "prune: stump count metal={} cpu={}",
            metal_stumps.len(), cpu_stumps.len()
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: multi-estimator loop (5 rounds, gradient update between rounds)
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn icb_multi_estimator_predictions_agree() {
        let Ok(metal) = MetalBackend::new() else { return; };
        if !metal.metal_device.capabilities.metal4 { return; }

        let (bm, mut grads) = make_fixture(10_000, 15, 32);
        let params   = default_params(4);
        let controls = default_controls();
        let fc       = bm.feature_count as u32;
        let tiles    = vec![FeatureTile { start_feature: 0, end_feature: fc }];
        let split_opts = split_opts_for(&params);

        let mut metal_preds = vec![0.0f32; 10_000];
        let mut cpu_preds   = vec![0.0f32; 10_000];

        for round in 0..5usize {
            let root_rows: Vec<u32> = (0..10_000u32).collect();

            // ICB path disabled in Stage 5 — skip this test.
            if metal.try_build_tree_level_wise(
                &bm, &grads, &root_rows, round, &tiles, split_opts,
                &params, &controls, &mut metal_preds, &[], &[],
            ).unwrap().is_none() {
                return;
            }

            cpu_tree(&bm, &grads, root_rows, &params, &mut cpu_preds);

            // Re-derive gradients from CPU predictions (pseudo squared-error).
            grads = cpu_preds.iter().enumerate().map(|(i, &p)| GradientPair {
                grad: p - (i as f32 / 10_000.0),
                hess: 1.0,
            }).collect();
        }

        assert_preds_close(&metal_preds, &cpu_preds, 0.05, "multi-estimator");
    }
}
