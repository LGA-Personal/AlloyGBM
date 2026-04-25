//! Verifies the default `BackendOps::*_batch` impls forward to the
//! scalar methods correctly, exercised through `CpuBackend` (which
//! does not override them).

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
use alloygbm_engine::{BackendOps, HistogramBuildRequest, SubtractRequest};

#[test]
fn cpu_backend_build_histograms_batch_default_matches_scalar() {
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

    let scalar_a = backend
        .build_histograms(&bm, &grads, &node_a, &tiles)
        .unwrap();
    let scalar_b = backend
        .build_histograms(&bm, &grads, &node_b, &tiles)
        .unwrap();

    let requests = vec![
        HistogramBuildRequest { node: &node_a },
        HistogramBuildRequest { node: &node_b },
    ];
    let batched = backend
        .build_histograms_batch(&bm, &grads, &tiles, &requests)
        .unwrap();
    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0], scalar_a);
    assert_eq!(batched[1], scalar_b);
}

#[test]
fn cpu_backend_subtract_histogram_bundle_batch_default_matches_scalar() {
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

    let parent_node = NodeSlice::new(0, (0..64u32).collect()).unwrap();
    let smaller_node = NodeSlice::new(1, (0..32u32).collect()).unwrap();
    let parent = backend
        .build_histograms(&bm, &grads, &parent_node, &tiles)
        .unwrap();
    let smaller = backend
        .build_histograms(&bm, &grads, &smaller_node, &tiles)
        .unwrap();

    let scalar = backend
        .subtract_histogram_bundle(&parent, &smaller, 2)
        .unwrap();

    let requests = vec![SubtractRequest {
        parent: &parent,
        sibling: &smaller,
        output_node_id: 2,
    }];
    let batched = backend
        .subtract_histogram_bundle_batch(&requests)
        .unwrap();
    assert_eq!(batched.len(), 1);
    assert_eq!(batched[0], scalar);
}
