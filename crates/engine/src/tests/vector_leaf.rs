//! `MulticlassStrategy::VectorLeaves`: one shared tree per round with a
//! K-vector in each leaf, stored as K per-class trees.

use super::main::MockBackend;
use super::*;
use alloygbm_core::{MISSING_BIN_U8, MulticlassHessian, MulticlassStrategy};

/// The builders' Newton-step denominator epsilon.
const LEAF_EPSILON: f64 = 1e-6;

const CLASSES: usize = 4;
const FEATURES: usize = 3;

/// Rows whose class depends on two of three features, with some missing
/// values in feature 0 and one uninformative feature.
fn vector_leaf_fixture(row_count: usize) -> (TrainingDataset, BinnedMatrix) {
    let mut values = Vec::with_capacity(row_count * FEATURES);
    let mut bins = Vec::with_capacity(row_count * FEATURES);
    let mut targets = Vec::with_capacity(row_count);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for row in 0..row_count {
        let a = (next() % 16) as u8;
        let b = (next() % 16) as u8;
        let noise = (next() % 16) as u8;
        let class = (usize::from(a >= 8) * 2 + usize::from(b >= 6) + usize::from(next() % 7 == 0))
            % CLASSES;
        let a_missing = row % 11 == 3;
        values.push(if a_missing { f32::NAN } else { f32::from(a) });
        values.push(f32::from(b));
        values.push(f32::from(noise));
        bins.push(if a_missing { MISSING_BIN_U8 } else { a });
        bins.push(b);
        bins.push(noise);
        targets.push(class as f32);
    }
    let dataset = TrainingDataset {
        matrix: alloygbm_core::DatasetMatrix::new(row_count, FEATURES, values)
            .expect("vector-leaf fixture matrix"),
        targets,
        sample_weights: None,
        time_index: None,
        group_id: None,
        factor_exposures: None,
    };
    let binned = BinnedMatrix::new(row_count, FEATURES, 15, bins).expect("vector-leaf bins");
    (dataset, binned)
}

fn vector_params(tree_growth: TreeGrowth) -> TrainParams {
    TrainParams {
        seed: 7,
        deterministic: true,
        max_depth: 3,
        max_leaves: (tree_growth == TreeGrowth::Leaf).then_some(5),
        min_data_in_leaf: 1,
        lambda_l2: 1.0,
        learning_rate: 0.3,
        tree_growth,
        multiclass_strategy: MulticlassStrategy::VectorLeaves,
        ..TrainParams::default()
    }
}

fn controls(rounds: usize, tree_growth: TreeGrowth) -> IterationControls {
    IterationControls::new(rounds, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
        .expect("controls")
        .with_max_leaves((tree_growth == TreeGrowth::Leaf).then_some(5))
        .expect("max leaves")
}

fn fit(
    params: TrainParams,
    controls: IterationControls,
    dataset: &TrainingDataset,
    binned: &BinnedMatrix,
) -> EngineResult<MultiClassIterationRunSummary> {
    let objective = MultiClassSoftmaxObjective::new(CLASSES).expect("objective");
    Trainer::new(params)?.fit_multiclass_iterations_with_summary(
        dataset,
        binned,
        &MockBackend,
        &objective,
        controls,
    )
}

/// Node ids per tree, in stump order, for one class.
fn structure(stumps: &[TrainedStump]) -> Vec<(u32, u32, u16, bool)> {
    stumps
        .iter()
        .map(|stump| {
            (
                stump.split.node_id,
                stump.split.feature_index,
                stump.split.threshold_bin,
                stump.split.default_left,
            )
        })
        .collect()
}

/// Raw scores from walking every stored tree, class by class.
fn replay(
    model: &MultiClassTrainedModel,
    dataset: &TrainingDataset,
    binned: &BinnedMatrix,
) -> Vec<Vec<f32>> {
    let mut predictions: Vec<Vec<f32>> = model
        .baseline_predictions
        .iter()
        .map(|&baseline| vec![baseline; binned.row_count])
        .collect();
    for (class_k, stumps) in model.class_stumps.iter().enumerate() {
        let mut start = 0;
        while start < stumps.len() {
            let tree_id = stumps[start].split.node_id / TREE_NODE_STRIDE;
            let mut end = start + 1;
            while end < stumps.len() && stumps[end].split.node_id / TREE_NODE_STRIDE == tree_id {
                end += 1;
            }
            apply_weighted_round_to_predictions(
                &mut predictions[class_k],
                binned,
                &stumps[start..end],
                Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                stumps[start].tree_weight,
            )
            .expect("replay tree");
            start = end;
        }
    }
    predictions
}

fn assert_shared_structure(model: &MultiClassTrainedModel) {
    let first = structure(&model.class_stumps[0]);
    assert!(!first.is_empty(), "vector-leaf fit grew no trees");
    for (class_k, stumps) in model.class_stumps.iter().enumerate().skip(1) {
        assert_eq!(
            structure(stumps),
            first,
            "class {class_k} does not share class 0's tree structure"
        );
    }
}

/// The training loss the trainer tracked must be the loss of the stored
/// model, which catches leaf deltas or candidate predictions that drift from
/// what the predictor replays.
fn assert_tracked_loss_matches_replay(
    summary: &MultiClassIterationRunSummary,
    dataset: &TrainingDataset,
    binned: &BinnedMatrix,
) {
    let objective = MultiClassSoftmaxObjective::new(CLASSES).expect("objective");
    let replayed = replay(&summary.model, dataset, binned);
    let loss = objective
        .loss(&replayed, &dataset.targets, None)
        .expect("replay loss");
    assert!(
        (loss - summary.final_loss).abs() <= 1e-5,
        "tracked loss {} differs from replayed loss {loss}",
        summary.final_loss
    );
    assert!(summary.final_loss < summary.initial_loss);
}

#[test]
fn vector_leaves_share_one_structure_and_replay_level_and_leaf_wise() {
    let (dataset, binned) = vector_leaf_fixture(240);
    for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
        let summary = fit(
            vector_params(tree_growth),
            controls(8, tree_growth),
            &dataset,
            &binned,
        )
        .expect("vector-leaf fit");
        assert_eq!(summary.rounds_completed, 8);
        assert_shared_structure(&summary.model);
        assert_tracked_loss_matches_replay(&summary, &dataset, &binned);
    }
}

#[test]
fn vector_leaves_replay_under_row_subsample_goss_and_dart() {
    let (dataset, binned) = vector_leaf_fixture(240);
    let modes = [
        BoostingMode::Goss {
            top_rate: 0.3,
            other_rate: 0.3,
        },
        BoostingMode::Dart {
            drop_rate: 0.5,
            max_drop: 3,
            normalize_type: alloygbm_core::DartNormalize::Tree,
            sample_type: alloygbm_core::DartSampleType::Uniform,
            skip_drop: 0.0,
        },
    ];
    for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
        let subsampled = controls(6, tree_growth)
            .with_subsample_rates(0.6, 0.7)
            .expect("subsample");
        let summary = fit(vector_params(tree_growth), subsampled, &dataset, &binned)
            .expect("row-subsampled fit");
        assert_shared_structure(&summary.model);
        assert_tracked_loss_matches_replay(&summary, &dataset, &binned);
        for boosting_mode in modes {
            let params = TrainParams {
                boosting_mode,
                ..vector_params(tree_growth)
            };
            let summary =
                fit(params, controls(6, tree_growth), &dataset, &binned).expect("GOSS/DART fit");
            assert_shared_structure(&summary.model);
            assert_tracked_loss_matches_replay(&summary, &dataset, &binned);
        }
    }
}

#[test]
fn vector_leaf_stump_leaves_are_per_class_newton_steps() {
    let (dataset, binned) = vector_leaf_fixture(200);
    let params = TrainParams {
        max_depth: 1,
        ..vector_params(TreeGrowth::Level)
    };
    let summary = fit(params, controls(1, TreeGrowth::Level), &dataset, &binned).expect("fit");
    let objective = MultiClassSoftmaxObjective::new(CLASSES).expect("objective");
    let predictions = vec![vec![0.0_f32; dataset.row_count()]; CLASSES];
    let gradients = objective
        .compute_gradients_all_classes(&predictions, &dataset.targets, None)
        .expect("gradients");
    for (class_k, stumps) in summary.model.class_stumps.iter().enumerate() {
        assert_eq!(stumps.len(), 1, "depth 1 grows one split per class");
        let stump = &stumps[0];
        let feature = stump.split.feature_index as usize;
        let (mut left, mut right) = ((0.0_f64, 0.0_f64), (0.0_f64, 0.0_f64));
        for (row, pair) in gradients[class_k].iter().enumerate() {
            let bin = binned.bin_at(row, feature);
            let goes_left = if bin == binned.missing_bin() {
                stump.split.default_left
            } else {
                bin <= stump.split.threshold_bin
            };
            let side = if goes_left { &mut left } else { &mut right };
            side.0 += f64::from(pair.grad);
            side.1 += f64::from(pair.hess);
        }
        let newton = |(grad, hess): (f64, f64)| (-0.3 * grad / (hess + 1.0 + LEAF_EPSILON)) as f32;
        for (leaf, expected) in [
            (&stump.left_leaf_value, newton(left)),
            (&stump.right_leaf_value, newton(right)),
        ] {
            let LeafValue::Scalar(value) = leaf else {
                panic!("vector leaves store scalar per-class leaves");
            };
            assert!(
                (value - expected).abs() <= 1e-5,
                "class {class_k}: leaf {value}, Newton step {expected}"
            );
        }
    }
}

#[test]
fn vector_leaves_are_identical_across_thread_counts() {
    let (dataset, binned) = vector_leaf_fixture(3_000);
    let fit_with = |threads: usize, tree_growth: TreeGrowth| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                fit(
                    vector_params(tree_growth),
                    controls(5, tree_growth)
                        .with_subsample_rates(0.8, 1.0)
                        .expect("subsample"),
                    &dataset,
                    &binned,
                )
                .expect("fit")
            })
    };
    for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
        let serial = fit_with(1, tree_growth);
        let parallel = fit_with(4, tree_growth);
        assert_eq!(serial.model.class_stumps, parallel.model.class_stumps);
        assert_eq!(serial.final_loss.to_bits(), parallel.final_loss.to_bits());
    }
}

#[test]
fn vector_leaves_respect_max_leaves() {
    let (dataset, binned) = vector_leaf_fixture(400);
    for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
        let params = TrainParams {
            max_depth: 6,
            ..vector_params(tree_growth)
        };
        let capped = controls(4, tree_growth)
            .with_max_leaves(Some(5))
            .expect("max leaves");
        let summary = fit(params, capped, &dataset, &binned).expect("fit");
        for stumps in &summary.model.class_stumps {
            for round in 0..4_u32 {
                let splits = stumps
                    .iter()
                    .filter(|stump| stump.split.node_id / TREE_NODE_STRIDE == round)
                    .count();
                assert!(
                    splits <= 4,
                    "{tree_growth:?} round {round} has {splits} splits, more than 5 leaves"
                );
            }
        }
    }
}

#[test]
fn vector_leaves_reject_unsupported_options() {
    let (dataset, binned) = vector_leaf_fixture(60);
    let rejected = [
        TrainParams {
            interaction_constraints: vec![vec![0, 1]],
            ..vector_params(TreeGrowth::Level)
        },
        TrainParams {
            leaf_model: alloygbm_core::LeafModelKind::Linear,
            ..vector_params(TreeGrowth::Level)
        },
        TrainParams {
            multiclass_strategy: MulticlassStrategy::PerClassTrees,
            multiclass_hessian: MulticlassHessian::Full,
            ..vector_params(TreeGrowth::Level)
        },
        TrainParams {
            multiclass_hessian: MulticlassHessian::Full,
            grn_strength: 1.0,
            grn_form: alloygbm_core::GrnForm::PerRow,
            quantized_grad_bits: None,
            ..vector_params(TreeGrowth::Level)
        },
    ];
    for params in rejected {
        let error = fit(params, controls(2, TreeGrowth::Level), &dataset, &binned)
            .expect_err("unsupported option must be rejected");
        assert!(matches!(error, EngineError::InvalidConfig(_)), "{error:?}");
    }
}

#[test]
fn full_hessian_rejects_more_than_sixteen_classes() {
    let row_count = 40;
    let dataset = TrainingDataset {
        matrix: alloygbm_core::DatasetMatrix::new(
            row_count,
            1,
            (0..row_count).map(|row| row as f32).collect(),
        )
        .expect("matrix"),
        targets: (0..row_count).map(|row| (row % 17) as f32).collect(),
        sample_weights: None,
        time_index: None,
        group_id: None,
        factor_exposures: None,
    };
    let binned = BinnedMatrix::new(row_count, 1, 39, (0..row_count as u8).collect()).expect("bins");
    let params = TrainParams {
        multiclass_hessian: MulticlassHessian::Full,
        ..vector_params(TreeGrowth::Level)
    };
    let objective = MultiClassSoftmaxObjective::new(17).expect("objective");
    let error = Trainer::new(params)
        .expect("trainer")
        .fit_multiclass_iterations_with_summary(
            &dataset,
            &binned,
            &MockBackend,
            &objective,
            controls(2, TreeGrowth::Level),
        )
        .expect_err("17 classes exceed the full-Hessian cap");
    assert!(matches!(error, EngineError::InvalidConfig(_)), "{error:?}");
}

#[test]
fn full_hessian_and_abs_residual_fits_replay() {
    let (dataset, binned) = vector_leaf_fixture(240);
    for (strategy, hessian) in [
        (MulticlassStrategy::VectorLeaves, MulticlassHessian::Full),
        (
            MulticlassStrategy::VectorLeaves,
            MulticlassHessian::AbsResidual,
        ),
        (
            MulticlassStrategy::PerClassTrees,
            MulticlassHessian::AbsResidual,
        ),
    ] {
        let params = TrainParams {
            multiclass_strategy: strategy,
            multiclass_hessian: hessian,
            ..vector_params(TreeGrowth::Level)
        };
        let summary = fit(
            params,
            controls(6, TreeGrowth::Level)
                .with_subsample_rates(0.7, 1.0)
                .expect("subsample"),
            &dataset,
            &binned,
        )
        .expect("fit");
        if strategy == MulticlassStrategy::VectorLeaves {
            assert_shared_structure(&summary.model);
        }
        assert_tracked_loss_matches_replay(&summary, &dataset, &binned);
    }
}

#[test]
fn full_hessian_leaves_are_centred_across_classes() {
    let (dataset, binned) = vector_leaf_fixture(240);
    let params = TrainParams {
        multiclass_hessian: MulticlassHessian::Full,
        ..vector_params(TreeGrowth::Level)
    };
    let summary = fit(params, controls(3, TreeGrowth::Level), &dataset, &binned).expect("fit");
    let stump_count = summary.model.class_stumps[0].len();
    for index in 0..stump_count {
        for side in [true, false] {
            let total: f32 = summary
                .model
                .class_stumps
                .iter()
                .map(|stumps| {
                    let leaf = if side {
                        &stumps[index].left_leaf_value
                    } else {
                        &stumps[index].right_leaf_value
                    };
                    match leaf {
                        LeafValue::Scalar(value) => *value,
                        LeafValue::Linear(_) => panic!("scalar leaves expected"),
                    }
                })
                .sum();
            assert!(
                total.abs() <= 1e-5,
                "stump {index}: class deltas sum to {total}"
            );
        }
    }
}
