//! Verifies the default `BackendOps::find_best_splits_batch` impl
//! delegates to per-node `best_split_with_options`, exercised through
//! `CpuBackend` (which does not override it).

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, SplitFindRequest, SplitSelectionOptions,
};

#[test]
fn cpu_backend_find_best_splits_batch_default_matches_scalar() {
    let row_count = 64usize;
    let feature_count = 3usize;
    let max_bin: u16 = 7;
    let bins: Vec<u8> = (0..(row_count * feature_count))
        .map(|i| ((i * 11) & 7) as u8)
        .collect();
    let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
    let grads: Vec<GradientPair> = (0..row_count)
        .map(|i| GradientPair {
            grad: i as f32,
            hess: 1.0,
        })
        .collect();
    let tiles = vec![FeatureTile {
        start_feature: 0,
        end_feature: feature_count as u32,
    }];
    let backend = CpuBackend;

    let node_a = NodeSlice::new(0, (0..32u32).collect()).unwrap();
    let node_b = NodeSlice::new(1, (32..64u32).collect()).unwrap();

    let hist_a = backend.build_histograms(&bm, &grads, &node_a, &tiles).unwrap();
    let hist_b = backend.build_histograms(&bm, &grads, &node_b, &tiles).unwrap();

    let options = SplitSelectionOptions::default();
    let feature_weights: Vec<f32> = vec![1.0; feature_count];
    let categorical_features: Vec<CategoricalFeatureInfo> = Vec::new();

    let scalar_a = backend
        .best_split_with_options(&hist_a, options, &feature_weights, &categorical_features)
        .unwrap();
    let scalar_b = backend
        .best_split_with_options(&hist_b, options, &feature_weights, &categorical_features)
        .unwrap();

    let requests = vec![
        SplitFindRequest { histograms: &hist_a },
        SplitFindRequest { histograms: &hist_b },
    ];
    let batched = backend
        .find_best_splits_batch(&requests, options, &feature_weights, &categorical_features)
        .unwrap();
    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0], scalar_a);
    assert_eq!(batched[1], scalar_b);
}

#[test]
fn cpu_backend_find_best_splits_batch_empty_is_noop() {
    let backend = CpuBackend;
    let requests: Vec<SplitFindRequest<'_>> = Vec::new();
    let result = backend
        .find_best_splits_batch(
            &requests,
            SplitSelectionOptions::default(),
            &[],
            &[],
        )
        .unwrap();
    assert!(result.is_empty());
}
