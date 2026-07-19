use crate::factor_split::factor_split_penalty;
use crate::*;
use alloygbm_core::{
    DatasetMatrix, FactorExposureMatrix, FeatureHistogram, FeatureTile, HistogramBin,
    LeafModelKind, TrainParams, TrainingDataset, TreeGrowth,
};
use alloygbm_engine::{
    BackendOps, FactorSplitContext, HistogramExecution, SquaredErrorObjective, Trainer,
};

fn sample_binned_matrix() -> BinnedMatrix {
    BinnedMatrix::new(
        4,
        2,
        3,
        vec![
            0, 0, //
            1, 0, //
            2, 1, //
            3, 1, //
        ],
    )
    .expect("binned matrix is valid")
}

fn quality_fixture_dataset() -> TrainingDataset {
    TrainingDataset {
        matrix: DatasetMatrix::new(
            8,
            2,
            vec![
                0.0, 0.0, //
                1.0, 0.0, //
                2.0, 0.0, //
                3.0, 0.0, //
                4.0, 0.0, //
                5.0, 0.0, //
                6.0, 0.0, //
                7.0, 0.0, //
            ],
        )
        .expect("matrix is valid"),
        targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        sample_weights: None,
        time_index: None,
        group_id: None,
        factor_exposures: None,
    }
}

fn node_parallelism_fixture() -> (TrainingDataset, BinnedMatrix) {
    const ROW_COUNT: usize = 8_192;
    const FEATURE_COUNT: usize = 8;
    let mut values = Vec::with_capacity(ROW_COUNT * FEATURE_COUNT);
    let mut bins = Vec::with_capacity(ROW_COUNT * FEATURE_COUNT);
    let mut targets = Vec::with_capacity(ROW_COUNT);
    for row in 0..ROW_COUNT {
        let bin = (row % 256) as u8;
        for _ in 0..FEATURE_COUNT {
            values.push(bin as f32);
            bins.push(bin);
        }
        let centered = bin as f32 - 127.5;
        targets.push(centered.signum() * centered.abs().sqrt());
    }
    (
        TrainingDataset {
            matrix: DatasetMatrix::new(ROW_COUNT, FEATURE_COUNT, values)
                .expect("parallel fixture matrix is valid"),
            targets,
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        },
        BinnedMatrix::new(ROW_COUNT, FEATURE_COUNT, 255, bins)
            .expect("parallel fixture bins are valid"),
    )
}

fn train_node_parallelism_fixture(thread_count: usize) -> alloygbm_engine::TrainedModel {
    let (dataset, binned) = node_parallelism_fixture();
    let mut params = fixture_params();
    params.max_depth = 8;
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .expect("test pool should build")
        .install(|| {
            Trainer::new(params)
                .expect("parallel fixture params are valid")
                .fit_iterations(&dataset, &binned, &CpuBackend, &SquaredErrorObjective, 1)
                .expect("parallel fixture should train")
        })
}

fn quality_fixture_binned_matrix() -> BinnedMatrix {
    BinnedMatrix::new(
        8,
        2,
        7,
        vec![
            0, 0, //
            1, 0, //
            2, 0, //
            3, 0, //
            4, 0, //
            5, 0, //
            6, 0, //
            7, 0, //
        ],
    )
    .expect("binned matrix is valid")
}

fn fixture_rows(dataset: &TrainingDataset) -> Vec<Vec<f32>> {
    dataset
        .matrix
        .values
        .chunks(dataset.matrix.feature_count)
        .map(|row| row.to_vec())
        .collect()
}

fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    let error_sum = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| {
            let error = prediction - target;
            error * error
        })
        .sum::<f32>();
    error_sum / predictions.len() as f32
}

fn fixture_params() -> TrainParams {
    TrainParams {
        seed: 7,
        deterministic: true,
        learning_rate: 0.3,
        max_depth: 6,
        row_subsample: 1.0,
        col_subsample: 1.0,
        early_stopping_rounds: None,
        min_validation_improvement: 0.0,
        min_data_in_leaf: 1,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_child_hessian: 0.0,
        min_split_gain: 0.0,
        monotone_constraints: Vec::new(),
        feature_weights: Vec::new(),
        interaction_constraints: Vec::new(),
        max_leaves: None,
        tree_growth: TreeGrowth::Level,
        morph_config: None,
        leaf_model: LeafModelKind::Constant,
        leaf_solver: alloygbm_core::LeafSolverKind::Standard,
        dro_config: None,
        neutralization_config: None,
        boosting_mode: alloygbm_core::BoostingMode::Standard,
        tweedie_variance_power: 1.5,
        poisson_max_delta_step: 0.7,
        quantile_alpha: 0.5,
    }
}

fn sample_gradients() -> Vec<GradientPair> {
    vec![
        GradientPair {
            grad: 2.0,
            hess: 1.0,
        },
        GradientPair {
            grad: 1.0,
            hess: 1.0,
        },
        GradientPair {
            grad: -1.0,
            hess: 1.0,
        },
        GradientPair {
            grad: -2.0,
            hess: 1.0,
        },
    ]
}

fn sample_node() -> NodeSlice {
    NodeSlice::new(0, vec![0, 1, 2, 3]).expect("node is valid")
}

fn with_histogram_feature<R>(
    feature: &FeatureHistogram,
    f: impl FnOnce(HistogramFeatureView<'_>) -> R,
) -> R {
    let bundle = HistogramBundle::from_feature_histograms(0, vec![feature.clone()], true)
        .expect("valid histogram fixture");
    f(bundle.feature(0).expect("fixture feature"))
}

#[test]
fn build_histograms_aggregates_bins() {
    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    assert_eq!(histograms.feature_count(), 2);
    assert!(!histograms.has_grad_sq_sums());
    let feature0 = histograms.feature(0).expect("first feature");
    assert_eq!(feature0.feature_index(), 0);
    assert_eq!(feature0.len(), 4);
    assert_eq!(feature0.bin(0).expect("bin").count, 1);
    assert_eq!(feature0.bin(1).expect("bin").count, 1);
    assert_eq!(feature0.bin(2).expect("bin").count, 1);
    assert_eq!(feature0.bin(3).expect("bin").count, 1);
    assert!((feature0.bin(0).expect("bin").grad_sum - 2.0).abs() < 1e-6);
    assert!((feature0.bin(3).expect("bin").grad_sum + 2.0).abs() < 1e-6);
}

#[test]
fn squared_gradient_column_is_allocated_only_when_requested() {
    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let gradients = sample_gradients();
    let node = sample_node();
    let tiles = [FeatureTile::new(0, 2).expect("feature tile is valid")];

    let standard = backend
        .build_histograms(&matrix, &gradients, &node, &tiles)
        .expect("standard histograms should build");
    let dro = backend
        .build_histograms_with_grad_sq(&matrix, &gradients, &node, &tiles, true)
        .expect("DRO histograms should build");

    assert!(!standard.has_grad_sq_sums());
    assert!(dro.has_grad_sq_sums());
    assert_eq!(standard.feature(0).expect("feature").grad_sq_sums(), None);
    assert_eq!(
        dro.feature(0).expect("feature").grad_sq_sums(),
        Some(&[4.0, 1.0, 1.0, 4.0][..])
    );
}

#[test]
fn build_histograms_is_tile_partition_invariant() {
    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let gradients = sample_gradients();
    let node = sample_node();

    let single_tile = backend
        .build_histograms(
            &matrix,
            &gradients,
            &node,
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("single-tile histograms should build");
    let split_tiles = backend
        .build_histograms(
            &matrix,
            &gradients,
            &node,
            &[
                FeatureTile::new(0, 1).expect("feature tile is valid"),
                FeatureTile::new(1, 2).expect("feature tile is valid"),
            ],
        )
        .expect("split-tile histograms should build");

    assert_eq!(single_tile, split_tiles);
    assert_eq!(
        backend
            .best_split(&single_tile)
            .expect("single-tile split should succeed"),
        backend
            .best_split(&split_tiles)
            .expect("split-tile split should succeed")
    );
}

#[test]
fn histogram_tile_strategies_are_equivalent() {
    let matrix = sample_binned_matrix();
    let gradients = sample_gradients();
    let node = sample_node();
    let bin_count = matrix.max_bin as usize + 1;

    let mut per_feature_arena = HistogramArena::new(2, bin_count, true);
    CpuBackend::build_tile_histograms_per_feature::<true>(
        &matrix,
        &gradients,
        &node,
        0,
        2,
        &mut per_feature_arena,
    );
    let per_feature = per_feature_arena
        .to_bundle(0, 0)
        .expect("per-feature histogram bundle");

    let mut arena = HistogramArena::new(2, bin_count, true);
    CpuBackend::build_tile_histograms_row_first(&matrix, &gradients, &node, 0, 2, &mut arena);
    let row_first = arena.to_bundle(0, 0).expect("row-first histogram bundle");

    assert_eq!(per_feature, row_first);
}

#[test]
fn histogram_kernel_path_prefers_tiny_node_scalar_for_small_nodes() {
    let path = CpuBackend::select_histogram_kernel_path(8, SMALL_TILE_WORKLOAD_THRESHOLD, 16);
    assert_eq!(path, HistogramKernelPath::TinyNodeScalar);
}

#[test]
fn histogram_kernel_path_prefers_unrolled_for_large_tiles() {
    let path = CpuBackend::select_histogram_kernel_path(256, SMALL_TILE_WORKLOAD_THRESHOLD + 1, 64);
    assert_eq!(path, HistogramKernelPath::ArenaRowFirstUnrolled);
}

#[test]
fn histogram_kernel_path_prefers_bin_heavy_fallback_for_wide_bins() {
    let path = CpuBackend::select_histogram_kernel_path(
        512,
        SMALL_TILE_WORKLOAD_THRESHOLD + 1,
        BIN_HEAVY_THRESHOLD,
    );
    assert_eq!(path, HistogramKernelPath::BinHeavyPerFeatureScalar);
}

#[test]
fn tile_parallelization_policy_requires_sufficient_workload() {
    assert!(!CpuBackend::should_parallelize_tiles(1, 4096, 128));
    assert!(!CpuBackend::should_parallelize_tiles(4, 128, 8));

    let expected = rayon::current_num_threads() > 1;
    assert_eq!(CpuBackend::should_parallelize_tiles(4, 4096, 128), expected);
}

#[test]
fn build_histograms_parallel_tiles_match_sequential() {
    let backend = CpuBackend;
    let matrix = quality_fixture_binned_matrix();
    let gradients = (0..matrix.row_count)
        .map(|row_index| {
            let grad = (row_index as f32 % 23.0) - 11.0;
            let hess = 1.0 + (row_index as f32 % 5.0) * 0.1;
            GradientPair::new(grad, hess).expect("gradient pair should be valid")
        })
        .collect::<Vec<_>>();
    let node =
        NodeSlice::new(0, (0..matrix.row_count as u32).collect()).expect("node should be valid");
    let feature_tiles = vec![
        FeatureTile::new(0, 1).expect("feature tile should be valid"),
        FeatureTile::new(1, 2).expect("feature tile should be valid"),
    ];

    let sequential = CpuBackend::build_histograms_internal(
        &matrix,
        &gradients,
        &node,
        &feature_tiles,
        false,
        false,
    )
    .expect("sequential histograms should build");
    let parallel = CpuBackend::build_histograms_internal(
        &matrix,
        &gradients,
        &node,
        &feature_tiles,
        true,
        false,
    )
    .expect("parallel histograms should build");

    assert_eq!(sequential, parallel);
    assert_eq!(
        backend
            .best_split(&sequential)
            .expect("sequential split should succeed"),
        backend
            .best_split(&parallel)
            .expect("parallel split should succeed")
    );
}

#[test]
fn explicit_histogram_execution_policies_are_equivalent() {
    let backend = CpuBackend;
    let matrix = quality_fixture_binned_matrix();
    let gradients = (0..matrix.row_count)
        .map(|row_index| {
            GradientPair::new((row_index as f32 - 3.5) * 0.5, 1.0 + row_index as f32 * 0.1)
                .expect("gradient pair is finite")
        })
        .collect::<Vec<_>>();
    let node =
        NodeSlice::new(0, (0..matrix.row_count as u32).collect()).expect("node indices are valid");
    let tiles = [FeatureTile::new(0, matrix.feature_count as u32).expect("valid feature tile")];

    let sequential = backend
        .build_histograms_with_execution(
            &matrix,
            &gradients,
            &node,
            &tiles,
            false,
            HistogramExecution::Sequential,
        )
        .expect("sequential histograms should build");
    let parallel = backend
        .build_histograms_with_execution(
            &matrix,
            &gradients,
            &node,
            &tiles,
            false,
            HistogramExecution::Parallel,
        )
        .expect("parallel histograms should build");

    assert_eq!(sequential, parallel);
}

#[test]
fn unrolled_row_first_histograms_match_per_feature() {
    let matrix = quality_fixture_binned_matrix();
    let gradients = (0..matrix.row_count)
        .map(|row_index| {
            GradientPair::new((row_index as f32 - 3.5) * 0.5, 1.0 + row_index as f32 * 0.1)
                .expect("gradient pair is finite")
        })
        .collect::<Vec<_>>();
    let node =
        NodeSlice::new(0, (0..matrix.row_count as u32).collect()).expect("node indices are valid");
    let bin_count = matrix.max_bin as usize + 1;

    let mut per_feature_arena = HistogramArena::new(matrix.feature_count, bin_count, true);
    CpuBackend::build_tile_histograms_per_feature::<true>(
        &matrix,
        &gradients,
        &node,
        0,
        matrix.feature_count,
        &mut per_feature_arena,
    );
    let per_feature = per_feature_arena
        .to_bundle(0, 0)
        .expect("per-feature histogram bundle");

    let mut unrolled_arena = HistogramArena::new(matrix.feature_count, bin_count, true);
    CpuBackend::build_tile_histograms_row_first_unrolled::<true>(
        &matrix,
        &gradients,
        &node,
        0,
        matrix.feature_count,
        &mut unrolled_arena,
    );
    let unrolled = unrolled_arena
        .to_bundle(0, 0)
        .expect("unrolled histogram bundle");

    assert_eq!(per_feature, unrolled);
}

#[test]
fn best_split_returns_high_gain_candidate() {
    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");
    let split = backend
        .best_split(&histograms)
        .expect("split search should succeed")
        .expect("split should exist");

    assert_eq!(split.feature_index, 0);
    assert_eq!(split.threshold_bin, 1);
    assert!(split.gain > 0.0);
    assert_eq!(split.left_stats.row_count, 2);
    assert_eq!(split.right_stats.row_count, 2);
}

#[test]
fn best_split_with_l2_regularization_reduces_gain_magnitude() {
    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let unregularized = backend
        .best_split(&histograms)
        .expect("unregularized split search should succeed")
        .expect("unregularized split should exist");
    let regularized = backend
        .best_split_with_options(
            &histograms,
            SplitSelectionOptions {
                l2_lambda: 1.0,
                l1_alpha: 0.0,
                min_child_hessian: 0.0,
                min_rows_per_leaf: 1,
                min_leaf_magnitude: 0.0,
                dro_config: None,
                missing_bin_index: 255,
            },
            &[],
            &[],
        )
        .expect("regularized split search should succeed")
        .expect("regularized split should exist");

    assert_eq!(unregularized.feature_index, regularized.feature_index);
    assert_eq!(unregularized.threshold_bin, regularized.threshold_bin);
    assert!(regularized.gain < unregularized.gain);
}

#[test]
fn best_split_with_l1_regularization_reduces_gain_magnitude() {
    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let unregularized = backend
        .best_split(&histograms)
        .expect("unregularized split search should succeed")
        .expect("unregularized split should exist");
    let regularized = backend
        .best_split_with_options(
            &histograms,
            SplitSelectionOptions {
                l2_lambda: 0.0,
                l1_alpha: 0.5,
                min_child_hessian: 0.0,
                min_rows_per_leaf: 1,
                min_leaf_magnitude: 0.0,
                dro_config: None,
                missing_bin_index: 255,
            },
            &[],
            &[],
        )
        .expect("regularized split search should succeed")
        .expect("regularized split should exist");

    assert_eq!(unregularized.feature_index, regularized.feature_index);
    assert_eq!(unregularized.threshold_bin, regularized.threshold_bin);
    assert!(regularized.gain < unregularized.gain);
}

#[test]
fn factor_split_penalty_reduces_factor_loaded_gain() {
    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let node = sample_node();
    let histograms = backend
        .build_histograms(
            &matrix,
            &sample_gradients(),
            &node,
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");
    let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0])
        .expect("factor exposures are valid");
    let no_penalty = backend
        .best_split_with_options(&histograms, SplitSelectionOptions::default(), &[], &[])
        .expect("split search should succeed")
        .expect("split should exist");
    let factor_context = FactorSplitContext {
        binned_matrix: &matrix,
        exposures: &exposures,
        row_indices: &node.row_indices,
        factor_penalty: 0.1,
    };
    let penalized = backend
        .best_split_with_factor_context(
            &histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
            Some(&factor_context),
        )
        .expect("split search should succeed")
        .expect("split should exist");
    assert!(penalized.gain <= no_penalty.gain);
}

#[test]
fn morph_neutralization_split_penalty_reduces_factor_loaded_gain() {
    use alloygbm_core::{MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let node = sample_node();
    let histograms = backend
        .build_histograms(
            &matrix,
            &sample_gradients(),
            &node,
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");
    let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0])
        .expect("factor exposures are valid");
    let cfg = MorphConfig {
        morph_warmup_iters: 0,
        balance_penalty: false,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 10,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(10, 100, &cfg),
    };
    let no_penalty = backend
        .best_split_morph(
            &histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
            &morph,
        )
        .expect("morph split search should succeed")
        .expect("split should exist");
    let factor_context = FactorSplitContext {
        binned_matrix: &matrix,
        exposures: &exposures,
        row_indices: &node.row_indices,
        factor_penalty: 0.1,
    };
    let penalized = backend
        .best_split_morph_with_factor_context(
            &histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
            &morph,
            Some(&factor_context),
        )
        .expect("morph split search should succeed")
        .expect("split should exist");

    assert_eq!(penalized.feature_index, no_penalty.feature_index);
    assert_eq!(penalized.threshold_bin, no_penalty.threshold_bin);
    let expected_penalty =
        factor_split_penalty(&[2.0], &[-2.0], -1.5, 1.5, 0.1, node.row_indices.len());
    let observed_penalty = no_penalty.gain - penalized.gain;
    assert!(
        (observed_penalty - expected_penalty).abs() < 1e-6,
        "expected Morph factor penalty {expected_penalty}, observed {observed_penalty}"
    );
    assert!(
        observed_penalty > 0.5,
        "factor context should strictly reduce Morph gain, observed {observed_penalty}"
    );
}

#[test]
fn factor_split_penalty_formula_matches_expected() {
    let left_factor_sums = [3.0_f32, -1.0];
    let right_factor_sums = [-2.0_f32, 4.0];
    let penalty = factor_split_penalty(&left_factor_sums, &right_factor_sums, 0.5, -0.25, 2.0, 5);

    let load0 = 3.0 * 0.5 + -2.0 * -0.25;
    let load1 = -0.5 + 4.0 * -0.25;
    let expected = 2.0 * (load0 * load0 + load1 * load1) / 5.0;
    assert!((penalty - expected).abs() < 1e-6);
}

#[test]
fn factor_split_penalty_rejects_malformed_factor_context() {
    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let node = sample_node();
    let histograms = backend
        .build_histograms(
            &matrix,
            &sample_gradients(),
            &node,
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");
    let cases = [
        (
            FactorExposureMatrix {
                row_count: 4,
                factor_count: 0,
                values: Vec::new(),
            },
            "factor_exposures factor_count must be greater than 0",
        ),
        (
            FactorExposureMatrix {
                row_count: 4,
                factor_count: 1,
                values: vec![1.0, 1.0, -1.0],
            },
            "factor_exposures values length 3 does not match row_count * factor_count 4",
        ),
        (
            FactorExposureMatrix {
                row_count: 4,
                factor_count: 1,
                values: vec![1.0, f32::NAN, -1.0, -1.0],
            },
            "factor_exposures must contain only finite values",
        ),
    ];

    for (malformed, expected_message) in cases {
        let factor_context = FactorSplitContext {
            binned_matrix: &matrix,
            exposures: &malformed,
            row_indices: &node.row_indices,
            factor_penalty: 0.1,
        };

        let err = backend
            .best_split_with_factor_context(
                &histograms,
                SplitSelectionOptions::default(),
                &[],
                &[],
                Some(&factor_context),
            )
            .expect_err("malformed factor context should be rejected");
        assert!(matches!(err, EngineError::ContractViolation(_)));
        assert!(
            err.to_string().contains(expected_message),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn best_split_with_min_child_hessian_can_prune_all_splits() {
    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let split = backend
        .best_split_with_options(
            &histograms,
            SplitSelectionOptions {
                l2_lambda: 0.0,
                l1_alpha: 0.0,
                min_child_hessian: 10.0,
                min_rows_per_leaf: 1,
                min_leaf_magnitude: 0.0,
                dro_config: None,
                missing_bin_index: 255,
            },
            &[],
            &[],
        )
        .expect("split search should succeed");

    assert!(split.is_none());
}

#[test]
fn best_split_with_min_leaf_magnitude_skips_weak_leaf_updates() {
    let backend = CpuBackend;
    let histograms = HistogramBundle::from_feature_histograms(
        0,
        vec![
            FeatureHistogram {
                feature_index: 0,
                bins: vec![
                    HistogramBin {
                        grad_sum: 1.0,
                        hess_sum: 20.0,
                        grad_sq_sum: 0.0,
                        count: 5,
                    },
                    HistogramBin {
                        grad_sum: -1.0,
                        hess_sum: 20.0,
                        grad_sq_sum: 0.0,
                        count: 5,
                    },
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        grad_sq_sum: 0.0,
                        count: 0,
                    },
                ],
            },
            FeatureHistogram {
                feature_index: 1,
                bins: vec![
                    HistogramBin {
                        grad_sum: 0.5,
                        hess_sum: 5.0,
                        grad_sq_sum: 0.0,
                        count: 5,
                    },
                    HistogramBin {
                        grad_sum: -0.5,
                        hess_sum: 5.0,
                        grad_sq_sum: 0.0,
                        count: 5,
                    },
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        grad_sq_sum: 0.0,
                        count: 0,
                    },
                ],
            },
        ],
        true,
    )
    .expect("valid histogram bundle");

    let unfiltered = backend
        .best_split(&histograms)
        .expect("default split search should succeed")
        .expect("default split should exist");
    let filtered = backend
        .best_split_with_options(
            &histograms,
            SplitSelectionOptions {
                l2_lambda: 0.0,
                l1_alpha: 0.0,
                min_child_hessian: 0.0,
                min_rows_per_leaf: 1,
                min_leaf_magnitude: 0.06,
                dro_config: None,
                missing_bin_index: 255,
            },
            &[],
            &[],
        )
        .expect("magnitude-filtered split search should succeed")
        .expect("magnitude-filtered split should exist");

    assert_eq!(unfiltered.feature_index, 0);
    assert_eq!(filtered.feature_index, 1);
    assert!(filtered.gain > 0.0);
}

#[test]
fn apply_split_partitions_rows() {
    let backend = CpuBackend;
    let split = SplitCandidate {
        node_id: 0,
        feature_index: 0,
        threshold_bin: 1,
        gain: 1.0,
        default_left: false,
        is_categorical: false,
        categorical_bitset: None,
        left_stats: NodeStats {
            grad_sum: 3.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            row_count: 2,
        },
        right_stats: NodeStats {
            grad_sum: -3.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            row_count: 2,
        },
    };
    let partition = backend
        .apply_split(&sample_binned_matrix(), &sample_node(), &split)
        .expect("partition should succeed");

    assert_eq!(partition.left_row_indices, vec![0, 1]);
    assert_eq!(partition.right_row_indices, vec![2, 3]);
}

#[test]
fn apply_split_with_stats_matches_partition_and_reduction_reference() {
    let backend = CpuBackend;
    let matrix = sample_binned_matrix();
    let node = sample_node();
    let gradients = sample_gradients();
    let split = SplitCandidate {
        node_id: 0,
        feature_index: 0,
        threshold_bin: 1,
        gain: 1.0,
        default_left: false,
        is_categorical: false,
        categorical_bitset: None,
        left_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            row_count: 0,
        },
        right_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            row_count: 0,
        },
    };

    let (partition, left_stats, right_stats) = backend
        .apply_split_with_stats(&matrix, &gradients, &node, &split)
        .expect("fused split should succeed");
    let reference_partition = backend
        .apply_split(&matrix, &node, &split)
        .expect("reference split should succeed");
    let reference_left = backend
        .reduce_sums(&gradients, &reference_partition.left_row_indices)
        .expect("reference left reduction should succeed");
    let reference_right = backend
        .reduce_sums(&gradients, &reference_partition.right_row_indices)
        .expect("reference right reduction should succeed");

    assert_eq!(partition, reference_partition);
    assert_eq!(left_stats, reference_left);
    assert_eq!(right_stats, reference_right);
}

#[test]
fn reduce_sums_aggregates_requested_rows() {
    let backend = CpuBackend;
    let stats = backend
        .reduce_sums(&sample_gradients(), &[0, 3])
        .expect("reductions should succeed");
    assert_eq!(stats.row_count, 2);
    assert!(stats.grad_sum.abs() < 1e-6);
    assert!((stats.hess_sum - 2.0).abs() < 1e-6);
}

#[test]
fn backend_reports_cpu_device() {
    assert_eq!(CpuBackend.device(), Device::Cpu);
}

#[test]
fn cpu_backend_training_beats_naive_baseline_mse() {
    let dataset = quality_fixture_dataset();
    let binned = quality_fixture_binned_matrix();
    let trainer = Trainer::new(fixture_params()).expect("params are valid");
    let backend = CpuBackend;
    let model = trainer
        .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
        .expect("training succeeds");

    assert!(!model.stumps.is_empty());

    let rows = fixture_rows(&dataset);
    let model_predictions = model.predict_batch(&rows).expect("predictions succeed");
    let baseline_prediction = dataset.targets.iter().sum::<f32>() / dataset.targets.len() as f32;
    let baseline_predictions = vec![baseline_prediction; dataset.targets.len()];

    let model_mse = mean_squared_error(&model_predictions, &dataset.targets);
    let baseline_mse = mean_squared_error(&baseline_predictions, &dataset.targets);
    assert!(model_mse < baseline_mse);
}

#[test]
fn cpu_backend_deterministic_training_has_stable_artifact_bytes() {
    let dataset = quality_fixture_dataset();
    let binned = quality_fixture_binned_matrix();
    let trainer = Trainer::new(fixture_params()).expect("params are valid");
    let backend = CpuBackend;
    let model_a = trainer
        .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
        .expect("first training succeeds");
    let model_b = trainer
        .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
        .expect("second training succeeds");

    let bytes_a = model_a.to_artifact_bytes().expect("artifact serializes");
    let bytes_b = model_b.to_artifact_bytes().expect("artifact serializes");
    assert_eq!(bytes_a, bytes_b);
}

#[test]
fn node_parallel_training_has_stable_artifacts_at_eight_threads() {
    let model_a = train_node_parallelism_fixture(8);
    let model_b = train_node_parallelism_fixture(8);

    assert_eq!(
        model_a.to_artifact_bytes().expect("artifact serializes"),
        model_b.to_artifact_bytes().expect("artifact serializes")
    );
}

#[test]
fn node_parallel_training_matches_single_thread_predictions() {
    let (dataset, _) = node_parallelism_fixture();
    let rows = fixture_rows(&dataset);
    let single_thread = train_node_parallelism_fixture(1)
        .predict_batch(&rows)
        .expect("single-thread predictions succeed");
    let eight_threads = train_node_parallelism_fixture(8)
        .predict_batch(&rows)
        .expect("eight-thread predictions succeed");

    assert_eq!(single_thread.len(), eight_threads.len());
    for (row, (single, parallel)) in single_thread.iter().zip(&eight_threads).enumerate() {
        assert!(
            (single - parallel).abs() <= 1e-6,
            "prediction drift at row {row}: single={single}, parallel={parallel}"
        );
    }
}

// ── Native categorical split tests ──────────────────────────────────

#[test]
fn test_best_split_categorical_basic() {
    // 3-category feature (bins 0,1,2) + NaN bin (bin 255)
    // Category 0: positive grad, category 1: positive grad, category 2: negative grad
    // Optimal split: categories 0,1 go left, category 2 goes right (or vice versa)
    let num_cats = 3;
    let nan_bin = 255usize;
    let num_bins = nan_bin + 1;
    let mut bins = vec![
        HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        num_bins
    ];
    // Category 0: grad=-2.0, hess=2.0 (score = -2/2 = -1.0)
    bins[0] = HistogramBin {
        grad_sum: -2.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 10,
    };
    // Category 1: grad=-1.5, hess=2.0 (score = -1.5/2 = -0.75)
    bins[1] = HistogramBin {
        grad_sum: -1.5,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 10,
    };
    // Category 2: grad=3.5, hess=2.0 (score = 3.5/2 = 1.75)
    bins[2] = HistogramBin {
        grad_sum: 3.5,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 10,
    };
    // NaN bin: no data
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };

    let options = SplitSelectionOptions {
        l2_lambda: 0.0,
        l1_alpha: 0.0,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.0,
        dro_config: None,
        missing_bin_index: nan_bin,
    };

    let result = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_categorical_feature(view, 0, options, num_cats, None)
    });
    assert!(result.is_some(), "should find a split");
    let split = result.unwrap();
    assert!(split.is_categorical);
    assert!(split.categorical_bitset.is_some());
    assert!(split.gain > 0.0, "gain should be positive");

    // Verify bitset: categories 0 and 1 should be on one side, category 2 on the other
    let bitset = split.categorical_bitset.as_ref().unwrap();
    let cat0_left = bitset[0] & (1 << 0) != 0;
    let cat1_left = bitset[0] & (1 << 1) != 0;
    let cat2_left = bitset[0] & (1 << 2) != 0;
    // Categories 0,1 have similar scores and should be grouped together
    assert_eq!(cat0_left, cat1_left, "cats 0 and 1 should be on same side");
    assert_ne!(cat0_left, cat2_left, "cat 2 should be on opposite side");
}

#[test]
fn dro_categorical_split_stats_match_direct_scan() {
    let num_cats = 4usize;
    let nan_bin = 15usize;
    let mut bins = vec![
        HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        nan_bin + 1
    ];
    bins[0] = HistogramBin {
        grad_sum: -4.0,
        hess_sum: 3.0,
        grad_sq_sum: 7.0,
        count: 6,
    };
    bins[1] = HistogramBin {
        grad_sum: -2.0,
        hess_sum: 2.0,
        grad_sq_sum: 3.0,
        count: 4,
    };
    bins[2] = HistogramBin {
        grad_sum: 3.0,
        hess_sum: 2.5,
        grad_sq_sum: 5.0,
        count: 5,
    };
    bins[3] = HistogramBin {
        grad_sum: 4.0,
        hess_sum: 3.5,
        grad_sq_sum: 8.0,
        count: 7,
    };
    bins[nan_bin] = HistogramBin {
        grad_sum: 0.75,
        hess_sum: 1.0,
        grad_sq_sum: 0.75,
        count: 2,
    };
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = SplitSelectionOptions {
        dro_config: Some(alloygbm_core::DroConfig {
            radius: 0.05,
            metric: alloygbm_core::DroMetric::Wasserstein,
        }),
        missing_bin_index: nan_bin,
        ..SplitSelectionOptions::default()
    };

    let split = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_categorical_feature(view, 0, options, num_cats, None)
    })
    .expect("dro categorical split should exist");
    let bitset = split
        .categorical_bitset
        .as_ref()
        .expect("categorical split has bitset");
    let mut expected_left = HistogramBin {
        grad_sum: 0.0,
        hess_sum: 0.0,
        grad_sq_sum: 0.0,
        count: 0,
    };
    let mut expected_right = HistogramBin {
        grad_sum: 0.0,
        hess_sum: 0.0,
        grad_sq_sum: 0.0,
        count: 0,
    };
    for (bin_id, bin) in fh.bins.iter().enumerate() {
        if bin.count == 0 {
            continue;
        }
        let goes_left = if bin_id == nan_bin {
            split.default_left
        } else if bin_id < num_cats {
            bitset[bin_id / 8] & (1 << (bin_id % 8)) != 0
        } else {
            continue;
        };
        let target = if goes_left {
            &mut expected_left
        } else {
            &mut expected_right
        };
        target.grad_sum += bin.grad_sum;
        target.hess_sum += bin.hess_sum;
        target.grad_sq_sum += bin.grad_sq_sum;
        target.count += bin.count;
    }

    assert!((split.left_stats.grad_sum - expected_left.grad_sum).abs() < 1e-6);
    assert!((split.left_stats.hess_sum - expected_left.hess_sum).abs() < 1e-6);
    assert!((split.left_stats.grad_sq_sum - expected_left.grad_sq_sum).abs() < 1e-6);
    assert_eq!(split.left_stats.row_count, expected_left.count);
    assert!((split.right_stats.grad_sum - expected_right.grad_sum).abs() < 1e-6);
    assert!((split.right_stats.hess_sum - expected_right.hess_sum).abs() < 1e-6);
    assert!((split.right_stats.grad_sq_sum - expected_right.grad_sq_sum).abs() < 1e-6);
    assert_eq!(split.right_stats.row_count, expected_right.count);
}

#[test]
fn test_best_split_categorical_single_populated() {
    // Only 1 category has data -> no valid split possible
    let num_cats = 3;
    let nan_bin = 255usize;
    let num_bins = nan_bin + 1;
    let mut bins = vec![
        HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        num_bins
    ];
    bins[1] = HistogramBin {
        grad_sum: 2.0,
        hess_sum: 5.0,
        grad_sq_sum: 0.0,
        count: 20,
    };
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };

    let options = SplitSelectionOptions::default();
    let result = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_categorical_feature(view, 0, options, num_cats, None)
    });
    assert!(
        result.is_none(),
        "single populated category should not split"
    );
}

#[test]
fn test_apply_split_categorical_bitset() {
    // Create a BinnedMatrix with 6 rows, 1 feature.
    // Category bin values: [0, 1, 2, 0, 1, 2]
    // Bitset: category 0 and 1 go left (bits 0,1 set = 0b0000_0011 = 3)
    let binned = BinnedMatrix::new(
        6,
        1,
        2, // max_bin = 2
        vec![0, 1, 2, 0, 1, 2],
    )
    .expect("valid matrix");

    let split = SplitCandidate {
        node_id: 0,
        feature_index: 0,
        threshold_bin: 0, // unused for categorical
        gain: 1.0,
        default_left: true,
        is_categorical: true,
        categorical_bitset: Some(vec![0b0000_0011]), // cats 0,1 go left
        left_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            row_count: 0,
        },
        right_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            row_count: 0,
        },
    };

    let node_slice = NodeSlice {
        node_id: 0,
        row_indices: (0..6).collect(),
    };

    let backend = CpuBackend;
    let partition = backend
        .apply_split(&binned, &node_slice, &split)
        .expect("partition should succeed");
    let left = &partition.left_row_indices;
    let right = &partition.right_row_indices;
    // Rows with bin 0 or 1 go left, rows with bin 2 go right
    assert_eq!(left.len(), 4, "categories 0,1 should go left");
    assert_eq!(right.len(), 2, "category 2 should go right");
    // Verify specific rows
    assert!(left.contains(&0)); // bin 0
    assert!(left.contains(&1)); // bin 1
    assert!(left.contains(&3)); // bin 0
    assert!(left.contains(&4)); // bin 1
    assert!(right.contains(&2)); // bin 2
    assert!(right.contains(&5)); // bin 2
}

#[test]
fn best_split_morph_at_warmup_matches_best_split_with_options() {
    use alloygbm_core::{MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let options = SplitSelectionOptions {
        l2_lambda: 0.0,
        l1_alpha: 0.0,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.0,
        dro_config: None,
        missing_bin_index: 255,
    };

    let cfg = MorphConfig {
        balance_penalty: false,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 0,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(0, 100, &cfg),
    };

    let standard = backend
        .best_split_with_options(&histograms, options, &[], &[])
        .expect("standard split search should succeed");
    let morph_result = backend
        .best_split_morph(&histograms, options, &[], &[], &morph)
        .expect("morph split search should succeed");

    // At iteration < warmup with balance penalty off, compute_morph_gain returns
    // exactly the standard XGBoost gain, so both paths must select the same split.
    assert!(
        standard.is_some(),
        "test fixture must produce a non-trivial split (standard path returned None)"
    );
    match (standard, morph_result) {
        (Some(a), Some(b)) => {
            assert_eq!(a.feature_index, b.feature_index, "feature_index disagreed");
            assert_eq!(a.threshold_bin, b.threshold_bin, "threshold_bin disagreed");
        }
        (None, None) => {}
        (a, b) => panic!(
            "split selection presence disagreed: standard={:?}, morph={:?}",
            a, b
        ),
    }
}

/// Regression test: warmup byte-equivalence must hold even with non-zero L1
/// and L2 regularisation. This specifically guards against the bugs where:
/// - EPSILON was missing from `gradient_gain` denominators (Issue 1)
/// - L1 thresholding was not applied in the morph path (Issue 2)
/// - `min_leaf_magnitude` was not checked in the morph path (Issue 3)
#[test]
fn best_split_morph_at_warmup_matches_with_l1_l2_regularization() {
    use alloygbm_core::{MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let options = SplitSelectionOptions {
        l2_lambda: 1.0,
        l1_alpha: 0.5,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.0,
        dro_config: None,
        missing_bin_index: 255,
    };

    let cfg = MorphConfig {
        balance_penalty: false,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 0,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(0, 100, &cfg),
    };

    let standard = backend
        .best_split_with_options(&histograms, options, &[], &[])
        .expect("standard split search should succeed");
    let morph_result = backend
        .best_split_morph(&histograms, options, &[], &[], &morph)
        .expect("morph split search should succeed");

    assert!(
        standard.is_some(),
        "test fixture must produce a non-trivial split (standard path returned None)"
    );
    let a = standard.unwrap();
    let b = morph_result.unwrap();
    assert_eq!(
        a.feature_index, b.feature_index,
        "feature_index disagreed under L1/L2 regularization"
    );
    assert_eq!(
        a.threshold_bin, b.threshold_bin,
        "threshold_bin disagreed under L1/L2 regularization"
    );
}

#[test]
fn best_split_morph_with_dro_uses_robust_gradient_gain_signal() {
    use alloygbm_core::{DroConfig, DroMetric, MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let backend = CpuBackend;
    let histograms = backend
        .build_histograms(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
        )
        .expect("histograms should build");

    let options = SplitSelectionOptions {
        l2_lambda: 0.1,
        l1_alpha: 0.0,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.0,
        dro_config: Some(DroConfig {
            radius: 0.05,
            metric: DroMetric::Wasserstein,
        }),
        missing_bin_index: 255,
    };

    let cfg = MorphConfig {
        morph_warmup_iters: 0,
        balance_penalty: false,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 10,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(10, 100, &cfg),
    };

    let split = backend
        .best_split_morph(&histograms, options, &[], &[], &morph)
        .expect("morph split search should succeed")
        .expect("test fixture should produce a split");

    let left_gradient_sum = leaf_effective_gradient(
        split.left_stats.grad_sum,
        split.left_stats.grad_sq_sum,
        split.left_stats.row_count,
        options.l1_alpha,
        options.dro_config.as_ref(),
    );
    let right_gradient_sum = leaf_effective_gradient(
        split.right_stats.grad_sum,
        split.right_stats.grad_sq_sum,
        split.right_stats.row_count,
        options.l1_alpha,
        options.dro_config.as_ref(),
    );
    let expected = compute_morph_gain(
        MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: left_gradient_sum,
                hessian_sum: split.left_stats.hess_sum,
                count: split.left_stats.row_count,
            },
            right: SplitSideStats {
                gradient_sum: right_gradient_sum,
                hessian_sum: split.right_stats.hess_sum,
                count: split.right_stats.row_count,
            },
            iteration: morph.iteration,
            total_iterations: morph.total_iterations,
            grad_mean: morph.grad_mean,
            grad_std: morph.grad_std,
            lambda_l2: options.l2_lambda,
        },
        &morph.config,
        &morph.precomputed,
    );

    assert!((split.gain - expected).abs() < 1e-6);
}

/// Regression test: at `iteration < morph_warmup_iters` with `balance_penalty=false`,
/// the morph categorical path must select the same partition as the standard path.
///
/// Uses a 4-category bundle where categories 0,1 have strongly negative gradients
/// and categories 2,3 have strongly positive gradients, making the best split
/// unambiguous regardless of the gain formula used.
#[test]
fn best_split_morph_at_warmup_matches_categorical_split() {
    use alloygbm_core::{MorphConfig, MorphPrecomputed};
    use alloygbm_engine::{CategoricalFeatureInfo, MorphContext};

    // Build a HistogramBundle with one categorical feature (4 categories).
    // Categories 0,1: negative gradient (score < 0)
    // Categories 2,3: positive gradient (score > 0)
    // Fisher-sort will place cats 0,1 on the left side, cats 2,3 on the right.
    let num_cats = 4usize;
    let nan_bin = 255usize;
    let num_bins = nan_bin + 1;
    let mut bins = vec![
        HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        num_bins
    ];
    // Category 0: strongly negative gradient
    bins[0] = HistogramBin {
        grad_sum: -4.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 20,
    };
    // Category 1: negative gradient
    bins[1] = HistogramBin {
        grad_sum: -3.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 20,
    };
    // Category 2: positive gradient
    bins[2] = HistogramBin {
        grad_sum: 3.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 20,
    };
    // Category 3: strongly positive gradient
    bins[3] = HistogramBin {
        grad_sum: 4.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 20,
    };

    let feature_histogram = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let histograms = HistogramBundle::from_feature_histograms(0, vec![feature_histogram], true)
        .expect("valid histogram bundle");

    let options = SplitSelectionOptions {
        l2_lambda: 0.0,
        l1_alpha: 0.0,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.0,
        dro_config: None,
        missing_bin_index: nan_bin,
    };

    let cat_features = vec![CategoricalFeatureInfo {
        feature_index: 0,
        num_categories: num_cats,
    }];

    let cfg = MorphConfig {
        balance_penalty: false,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 0,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(0, 100, &cfg),
    };

    let backend = CpuBackend;
    let standard = backend
        .best_split_with_options(&histograms, options, &[], &cat_features)
        .expect("standard split search should succeed");
    let morph_result = backend
        .best_split_morph(&histograms, options, &[], &cat_features, &morph)
        .expect("morph split search should succeed");

    assert!(
        standard.is_some(),
        "test fixture must produce a non-trivial split (standard path returned None)"
    );
    let a = standard.unwrap();
    let b = morph_result.unwrap();

    assert!(a.is_categorical, "standard split should be categorical");
    assert!(b.is_categorical, "morph split should be categorical");
    assert_eq!(
        a.feature_index, b.feature_index,
        "feature_index disagreed for categorical morph at warmup"
    );
    // Both paths must select the same bitset partition.
    assert_eq!(
        a.categorical_bitset, b.categorical_bitset,
        "categorical_bitset disagreed for morph at warmup"
    );
    assert_eq!(
        a.default_left, b.default_left,
        "default_left (NaN direction) disagreed for morph at warmup"
    );
    assert!(
        (a.gain - b.gain).abs() < 1e-5,
        "gain diverged at warmup: standard={}, morph={}",
        a.gain,
        b.gain
    );
}

fn make_options(
    l1_alpha: f32,
    l2_lambda: f32,
    min_child_hessian: f32,
    min_leaf_magnitude: f32,
    missing_bin_index: usize,
) -> SplitSelectionOptions {
    SplitSelectionOptions {
        l1_alpha,
        l2_lambda,
        min_child_hessian,
        min_rows_per_leaf: 1,
        min_leaf_magnitude,
        dro_config: None,
        missing_bin_index,
    }
}

#[test]
fn simd_standard_bin_scan_matches_scalar() {
    let bins: Vec<HistogramBin> = (0..32)
        .map(|i| HistogramBin {
            grad_sum: ((i as f32 - 15.5) * 0.1).sin(),
            hess_sum: 0.5 + (i as f32 * 0.05).cos().abs(),
            grad_sq_sum: 0.0,
            count: 10 + (i as u32 % 7),
        })
        .collect();
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = make_options(0.05, 0.1, 1.0, 0.0, 31);
    let scalar = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    });
    let simd = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 0, options)
    });
    match (scalar, simd) {
        (Some(s), Some(v)) => {
            assert_eq!(s.threshold_bin, v.threshold_bin, "threshold_bin mismatch");
            assert!(
                (s.gain - v.gain).abs() < 1e-4,
                "gain drift: scalar={} simd={}",
                s.gain,
                v.gain
            );
            assert_eq!(s.default_left, v.default_left);
        }
        (None, None) => {}
        (a, b) => panic!(
            "scalar/simd disagree on Some-ness: scalar={}, simd={}",
            a.is_some(),
            b.is_some()
        ),
    }
}

#[test]
fn simd_standard_bin_scan_matches_scalar_with_l1() {
    let bins: Vec<HistogramBin> = (0..16)
        .map(|i| HistogramBin {
            grad_sum: (i as f32 - 7.5) * 0.02,
            hess_sum: 1.0,
            grad_sq_sum: 0.0,
            count: 20,
        })
        .collect();
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = make_options(0.10, 0.1, 0.5, 0.0, 15);
    let scalar = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    });
    let simd = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 0, options)
    });
    match (scalar, simd) {
        (Some(s), Some(v)) => {
            assert_eq!(s.threshold_bin, v.threshold_bin);
            assert!((s.gain - v.gain).abs() < 1e-4);
        }
        (None, None) => {}
        _ => panic!("scalar/simd disagreement"),
    }
}

#[test]
fn simd_standard_bin_scan_matches_scalar_with_min_leaf_magnitude() {
    // Exercise the min_leaf_magnitude rejection branch.
    let bins: Vec<HistogramBin> = (0..16)
        .map(|i| HistogramBin {
            grad_sum: ((i as f32 - 7.5) * 0.05).sin(),
            hess_sum: 1.0 + (i as f32 * 0.1).cos().abs(),
            grad_sq_sum: 0.0,
            count: 12 + (i as u32 % 5),
        })
        .collect();
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = make_options(0.0, 0.1, 0.0, 0.05, 15);
    let scalar = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    });
    let simd = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 0, options)
    });
    match (scalar, simd) {
        (Some(s), Some(v)) => {
            assert_eq!(s.threshold_bin, v.threshold_bin);
            assert!((s.gain - v.gain).abs() < 1e-4);
            assert_eq!(s.default_left, v.default_left);
        }
        (None, None) => {}
        _ => panic!("scalar/simd disagreement on min_leaf_magnitude path"),
    }
}

#[test]
fn simd_standard_bin_scan_matches_scalar_with_missing_bin() {
    // Real missing-bin contribution exercises the NaN-direction routing.
    let mut bins: Vec<HistogramBin> = (0..16)
        .map(|i| HistogramBin {
            grad_sum: ((i as f32 - 7.5) * 0.1).sin(),
            hess_sum: 1.0 + (i as f32 * 0.05).cos().abs(),
            grad_sq_sum: 0.0,
            count: 8 + (i as u32 % 4),
        })
        .collect();
    // Simulate non-trivial missing bin at index 15.
    bins[15] = HistogramBin {
        grad_sum: 0.4,
        hess_sum: 1.5,
        grad_sq_sum: 0.0,
        count: 7,
    };
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = make_options(0.0, 0.1, 0.5, 0.0, 15);
    let scalar = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    });
    let simd = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 0, options)
    });
    match (scalar, simd) {
        (Some(s), Some(v)) => {
            assert_eq!(s.threshold_bin, v.threshold_bin);
            assert!((s.gain - v.gain).abs() < 1e-4);
            assert_eq!(s.default_left, v.default_left);
            assert_eq!(s.left_stats.row_count, v.left_stats.row_count);
            assert_eq!(s.right_stats.row_count, v.right_stats.row_count);
        }
        (None, None) => {}
        _ => panic!("scalar/simd disagreement on missing-bin path"),
    }
}

#[test]
fn dro_missing_bin_split_stats_match_direct_scan() {
    let missing_bin = 7usize;
    let mut bins: Vec<HistogramBin> = (0..=missing_bin)
        .map(|i| HistogramBin {
            grad_sum: (i as f32 - 3.0) * 0.7,
            hess_sum: 1.0 + i as f32 * 0.2,
            grad_sq_sum: 0.5 + i as f32 * 0.4,
            count: 3 + i as u32,
        })
        .collect();
    bins[missing_bin] = HistogramBin {
        grad_sum: -0.8,
        hess_sum: 1.4,
        grad_sq_sum: 1.2,
        count: 5,
    };
    let fh = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = SplitSelectionOptions {
        dro_config: Some(alloygbm_core::DroConfig {
            radius: 0.05,
            metric: alloygbm_core::DroMetric::Wasserstein,
        }),
        missing_bin_index: missing_bin,
        ..SplitSelectionOptions::default()
    };

    let split = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    })
    .expect("dro split with missing bin should exist");
    let mut expected_left = HistogramBin {
        grad_sum: 0.0,
        hess_sum: 0.0,
        grad_sq_sum: 0.0,
        count: 0,
    };
    let mut expected_right = HistogramBin {
        grad_sum: 0.0,
        hess_sum: 0.0,
        grad_sq_sum: 0.0,
        count: 0,
    };
    for (bin_id, bin) in fh.bins.iter().enumerate() {
        let goes_left = if bin_id == missing_bin {
            split.default_left
        } else {
            bin_id <= split.threshold_bin as usize
        };
        let target = if goes_left {
            &mut expected_left
        } else {
            &mut expected_right
        };
        target.grad_sum += bin.grad_sum;
        target.hess_sum += bin.hess_sum;
        target.grad_sq_sum += bin.grad_sq_sum;
        target.count += bin.count;
    }

    assert!((split.left_stats.grad_sum - expected_left.grad_sum).abs() < 1e-6);
    assert!((split.left_stats.hess_sum - expected_left.hess_sum).abs() < 1e-6);
    assert!((split.left_stats.grad_sq_sum - expected_left.grad_sq_sum).abs() < 1e-6);
    assert_eq!(split.left_stats.row_count, expected_left.count);
    assert!((split.right_stats.grad_sum - expected_right.grad_sum).abs() < 1e-6);
    assert!((split.right_stats.hess_sum - expected_right.hess_sum).abs() < 1e-6);
    assert!((split.right_stats.grad_sq_sum - expected_right.grad_sq_sum).abs() < 1e-6);
    assert_eq!(split.right_stats.row_count, expected_right.count);
}

#[test]
fn numeric_split_scanner_skips_candidates_below_min_rows_per_leaf() {
    let fh = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: 20.0,
                hess_sum: 1.0,
                grad_sq_sum: 400.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: 3.0,
                hess_sum: 3.0,
                grad_sq_sum: 9.0,
                count: 3,
            },
            HistogramBin {
                grad_sum: -23.0,
                hess_sum: 5.0,
                grad_sq_sum: 529.0,
                count: 5,
            },
        ],
    };
    let split = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            0,
            SplitSelectionOptions {
                min_rows_per_leaf: 4,
                missing_bin_index: 255,
                ..SplitSelectionOptions::default()
            },
            GainStrategy::Standard,
            None,
        )
    })
    .expect("expected feasible fallback split");

    assert_eq!(split.threshold_bin, 1);
    assert!(split.left_stats.row_count >= 4);
    assert!(split.right_stats.row_count >= 4);
}

#[test]
fn categorical_split_scanner_skips_candidates_below_min_rows_per_leaf() {
    let num_cats = 3;
    let fh = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: -20.0,
                hess_sum: 1.0,
                grad_sq_sum: 400.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -3.0,
                hess_sum: 3.0,
                grad_sq_sum: 9.0,
                count: 3,
            },
            HistogramBin {
                grad_sum: 23.0,
                hess_sum: 5.0,
                grad_sq_sum: 529.0,
                count: 5,
            },
        ],
    };
    let split = with_histogram_feature(&fh, |view| {
        CpuBackend::best_split_for_categorical_feature(
            view,
            0,
            SplitSelectionOptions {
                min_rows_per_leaf: 4,
                missing_bin_index: 255,
                ..SplitSelectionOptions::default()
            },
            num_cats,
            None,
        )
    })
    .expect("expected feasible categorical fallback split");

    assert!(split.left_stats.row_count >= 4);
    assert!(split.right_stats.row_count >= 4);
}
