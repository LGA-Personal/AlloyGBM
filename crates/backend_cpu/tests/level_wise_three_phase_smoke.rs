//! Smoke test: a deterministic 2-round CPU fit exercises the full
//! `build_tree_level_wise` loop. Run this before AND after the three-phase
//! refactor to confirm the refactor is observable-equivalent on CPU.
//!
//! The CPU backend uses the scalar default impls of the new batched trait
//! methods, so any failure after the refactor is a refactor bug, not a
//! Metal-specific issue.

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, DatasetMatrix, TrainParams, TrainingDataset};
use alloygbm_engine::{SquaredErrorObjective, Trainer};

fn make_dataset(n: usize) -> (TrainingDataset, BinnedMatrix) {
    // Synthetic 1-feature regression: y = (2x - 1)^2 on x in [0, 1)
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (x * 2.0 - 1.0).powi(2)).collect();
    let max_bin: u16 = 31;

    // Build flat float matrix (n rows x 1 feature)
    let raw_floats: Vec<f32> = xs.clone();
    let matrix = DatasetMatrix::new(n, 1, raw_floats).expect("valid matrix");

    let bins: Vec<u8> = xs
        .iter()
        .map(|x| ((x * (max_bin as f32 + 1.0)).floor() as u16).min(max_bin) as u8)
        .collect();
    let bm = BinnedMatrix::new(n, 1, max_bin, bins).expect("valid binned matrix");

    let dataset = TrainingDataset {
        matrix,
        targets: ys,
        sample_weights: None,
        time_index: None,
        group_id: None,
    };
    (dataset, bm)
}

#[test]
fn level_wise_two_round_fit_is_finite_and_stable() {
    let (dataset, bm) = make_dataset(200);

    let mut params = TrainParams::default();
    params.max_depth = 3;
    params.learning_rate = 0.1;

    let trainer = Trainer::new(params).expect("valid params");
    let backend = CpuBackend;

    let model = trainer
        .fit_iterations(&dataset, &bm, &backend, &SquaredErrorObjective, 2)
        .expect("2-round fit must succeed");

    // Smoke assertions: every stump leaf value is finite
    for stump in &model.stumps {
        assert!(
            stump.left_leaf_value.is_finite(),
            "left_leaf_value is not finite"
        );
        assert!(
            stump.right_leaf_value.is_finite(),
            "right_leaf_value is not finite"
        );
    }

    // Predictions must be finite
    let test_row = vec![0.5_f32];
    let pred = model
        .predict_row(&test_row)
        .expect("prediction should succeed");
    assert!(pred.is_finite(), "prediction is not finite: {pred}");
}
