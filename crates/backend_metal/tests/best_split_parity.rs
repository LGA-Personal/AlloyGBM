//! Bit-exact parity tests: Metal GPU split finding vs CPU baseline.
//! Skipped silently on non-Metal hosts via `MetalBackend::new()` early return.

#[cfg(target_os = "macos")]
mod tests {
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_backend_metal::MetalBackend;
    use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
    use alloygbm_engine::{
        BackendOps, CategoricalFeatureInfo, SplitFindRequest, SplitSelectionOptions,
    };

    fn make_fixture(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
    ) -> (BinnedMatrix, Vec<GradientPair>) {
        // Bins are row-major: index i → row r = i / feature_count, feature f = i % feature_count.
        // Use coprime strides (2f+1) per feature so bin distributions differ across features.
        // Feature 0 (stride 1) gets a monotone gradient-bin correlation → highest gain; features
        // 1, 2, 3, … get scrambled distributions → lower gains.  This breaks the symmetry that
        // caused identical gains (and therefore non-deterministic GPU tie-breaking) when every
        // feature had the same bin pattern.
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| {
                let r = i / feature_count;
                let f = i % feature_count;
                let stride = 2 * f + 1; // 1, 3, 5, 7, … (all odd → coprime with any power of 2)
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

    #[test]
    fn find_best_splits_batch_matches_cpu_all_numeric() {
        let Ok(metal) = MetalBackend::new() else {
            return;
        };
        let cpu = CpuBackend;

        let (bm, grads) = make_fixture(256, 4, 8);
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: 4,
        }];
        let nodes: Vec<NodeSlice> = vec![
            NodeSlice::new(0, (0..128u32).collect()).unwrap(),
            NodeSlice::new(1, (128..256u32).collect()).unwrap(),
            NodeSlice::new(2, (0..96u32).collect()).unwrap(),
        ];

        let cpu_hists: Vec<_> = nodes
            .iter()
            .map(|n| cpu.build_histograms(&bm, &grads, n, &tiles).unwrap())
            .collect();
        let metal_hists: Vec<_> = nodes
            .iter()
            .map(|n| metal.build_histograms(&bm, &grads, n, &tiles).unwrap())
            .collect();

        let options = SplitSelectionOptions::default();
        let weights: Vec<f32> = vec![1.0; 4];
        let cats: Vec<CategoricalFeatureInfo> = vec![];

        let cpu_splits: Vec<_> = cpu_hists
            .iter()
            .map(|h| cpu.best_split_with_options(h, options, &weights, &cats).unwrap())
            .collect();

        let requests: Vec<SplitFindRequest<'_>> = metal_hists
            .iter()
            .map(|h| SplitFindRequest { histograms: h })
            .collect();
        let metal_splits = metal
            .find_best_splits_batch(&requests, options, &weights, &cats)
            .unwrap();

        assert_eq!(metal_splits.len(), cpu_splits.len());
        for (m, c) in metal_splits.iter().zip(cpu_splits.iter()) {
            match (m, c) {
                (None, None) => {}
                (Some(ms), Some(cs)) => {
                    assert_eq!(ms.feature_index, cs.feature_index, "feature_index");
                    assert_eq!(ms.threshold_bin, cs.threshold_bin, "threshold_bin");
                    assert_eq!(ms.default_left, cs.default_left, "default_left");
                    // GPU parallel prefix-scan (simdgroup reductions) accumulates f32
                    // rounding errors differently from the CPU sequential scan.  For
                    // gains ~O(100) the absolute error can reach ~0.01; for gradient
                    // partial sums ~O(50) the error can reach ~0.01.
                    assert!(
                        (ms.gain - cs.gain).abs() < 0.1,
                        "gain: metal={} cpu={}",
                        ms.gain,
                        cs.gain
                    );
                    assert!(
                        (ms.left_stats.grad_sum - cs.left_stats.grad_sum).abs() < 0.05,
                        "left_grad: metal={} cpu={}",
                        ms.left_stats.grad_sum,
                        cs.left_stats.grad_sum
                    );
                    assert!(
                        (ms.left_stats.hess_sum - cs.left_stats.hess_sum).abs() < 0.05,
                        "left_hess: metal={} cpu={}",
                        ms.left_stats.hess_sum,
                        cs.left_stats.hess_sum
                    );
                }
                _ => panic!("split existence mismatch: metal={m:?} cpu={c:?}"),
            }
        }
    }

    #[test]
    fn find_best_splits_batch_empty_is_noop() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let result = backend
            .find_best_splits_batch(&[], SplitSelectionOptions::default(), &[], &[])
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn find_best_splits_batch_falls_back_with_categoricals() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let (bm, grads) = make_fixture(64, 2, 4);
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: 2,
        }];
        let node = NodeSlice::new(0, (0..64u32).collect()).unwrap();
        let h = backend
            .build_histograms(&bm, &grads, &node, &tiles)
            .unwrap();
        let cats = vec![CategoricalFeatureInfo {
            feature_index: 1,
            num_categories: 4,
        }];
        let result = backend
            .find_best_splits_batch(
                &[SplitFindRequest { histograms: &h }],
                SplitSelectionOptions::default(),
                &[1.0; 2],
                &cats,
            )
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    /// Mixed-mode parity: GPU numeric + host categorical merge agrees with the
    /// all-host scalar baseline on a 4-feature bundle where one feature is
    /// categorical.  Feature 0 (stride-1 monotone, highest numeric gain) should
    /// win outright; the merge must not corrupt the result.
    #[test]
    fn find_best_splits_batch_mixed_numeric_categorical_matches_scalar() {
        let Ok(metal) = MetalBackend::new() else {
            return;
        };
        let cpu = CpuBackend;

        let (bm, grads) = make_fixture(256, 4, 8);
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: 4,
        }];
        let node = NodeSlice::new(0, (0..256u32).collect()).unwrap();

        let cpu_h = cpu.build_histograms(&bm, &grads, &node, &tiles).unwrap();
        let metal_h = metal.build_histograms(&bm, &grads, &node, &tiles).unwrap();

        let options = SplitSelectionOptions::default();
        let feature_weights: Vec<f32> = vec![1.0; 4];
        // Feature 2 is categorical; features 0, 1, 3 are numeric.
        let cats = vec![CategoricalFeatureInfo {
            feature_index: 2,
            num_categories: 8,
        }];

        let cpu_split = cpu
            .best_split_with_options(&cpu_h, options, &feature_weights, &cats)
            .unwrap();
        let requests = vec![SplitFindRequest { histograms: &metal_h }];
        let metal_splits = metal
            .find_best_splits_batch(&requests, options, &feature_weights, &cats)
            .unwrap();

        assert_eq!(metal_splits.len(), 1);
        match (&metal_splits[0], &cpu_split) {
            (None, None) => {}
            (Some(m), Some(c)) => {
                assert_eq!(
                    m.feature_index, c.feature_index,
                    "mixed-mode winning feature mismatch: metal={} cpu={}",
                    m.feature_index, c.feature_index,
                );
                assert_eq!(
                    m.is_categorical, c.is_categorical,
                    "is_categorical mismatch",
                );
                // GPU parallel scan accumulates f32 rounding differently from
                // the CPU sequential scan; allow up to 0.1 absolute error.
                assert!(
                    (m.gain - c.gain).abs() < 0.1,
                    "mixed-mode gain mismatch: metal={} cpu={}",
                    m.gain,
                    c.gain,
                );
            }
            _ => panic!(
                "mixed-mode split existence mismatch: metal={:?} cpu={:?}",
                metal_splits[0], cpu_split
            ),
        }
    }
}
