use crate::factor_split::factor_split_penalty;
use crate::split_helpers::goes_left_for_split;
use crate::*;
use alloygbm_core::{
    DatasetMatrix, FactorExposureMatrix, FeatureHistogram, FeatureTile, HistogramBin,
    LeafModelKind, LinearFeatureScaler, MISSING_BIN_U8, MorphConfig, MorphPrecomputed, TrainParams,
    TrainingDataset, TreeGrowth, discover_exact_feature_bundles,
};
use alloygbm_engine::{
    BackendOps, FactorSplitContext, HistogramExecution, LinearContext, MorphContext,
    SquaredErrorObjective, Trainer,
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
        pl_split_candidates: 8,
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

fn split_candidate(
    feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
    is_categorical: bool,
    categorical_bitset: Option<Vec<u8>>,
) -> SplitCandidate {
    SplitCandidate {
        node_id: 0,
        feature_index,
        threshold_bin,
        gain: 0.0,
        default_left,
        is_categorical,
        categorical_bitset,
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
    }
}

fn shortlist_standard_fixture(feature_count: usize) -> HistogramBundle {
    let features = (0..feature_count)
        .map(|feature_index| {
            let scale = (feature_index + 1) as f32;
            FeatureHistogram {
                feature_index: feature_index as u32,
                bins: vec![
                    HistogramBin {
                        grad_sum: 2.0 * scale,
                        hess_sum: 2.0,
                        grad_sq_sum: 0.0,
                        count: 2,
                    },
                    HistogramBin {
                        grad_sum: -2.0 * scale,
                        hess_sum: 2.0,
                        grad_sq_sum: 0.0,
                        count: 2,
                    },
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        grad_sq_sum: 0.0,
                        count: 0,
                    },
                ],
            }
        })
        .collect();
    HistogramBundle::from_feature_histograms(17, features, false)
        .expect("shortlist fixture is valid")
}

#[test]
fn shortlist_standard_respects_k_weighting_and_production_winner() {
    let backend = CpuBackend;
    let histograms = shortlist_standard_fixture(5);
    let options = SplitSelectionOptions {
        missing_bin_index: 2,
        ..SplitSelectionOptions::default()
    };
    let weights = [1.0, 100.0, 1.0, 1.0, 1.0];

    let shortlist = backend
        .shortlist_standard_splits(&histograms, options, &weights, &[], 2)
        .expect("shortlist succeeds");
    let production = backend
        .best_split_with_options(&histograms, options, &weights, &[])
        .expect("production selection succeeds");

    assert_eq!(shortlist.best_overall, production);
    assert_eq!(
        shortlist
            .numeric_candidates
            .iter()
            .map(|candidate| candidate.feature_index)
            .collect::<Vec<_>>(),
        vec![1, 4]
    );
}

#[test]
fn shortlist_standard_handles_zero_exhaustive_ties_and_parallel_features() {
    let backend = CpuBackend;
    for feature_count in [4, CpuBackend::PARALLEL_SPLIT_FEATURE_THRESHOLD] {
        let histograms = shortlist_standard_fixture(feature_count);
        let options = SplitSelectionOptions {
            missing_bin_index: 2,
            ..SplitSelectionOptions::default()
        };
        let empty = backend
            .shortlist_standard_splits(&histograms, options, &[], &[], 0)
            .expect("zero shortlist succeeds");
        assert!(empty.numeric_candidates.is_empty());

        let exhaustive = backend
            .shortlist_standard_splits(&histograms, options, &[], &[], usize::MAX)
            .expect("exhaustive shortlist succeeds");
        assert_eq!(exhaustive.numeric_candidates.len(), feature_count);
        assert_eq!(
            exhaustive.numeric_candidates[0].feature_index,
            (feature_count - 1) as u32
        );
        assert_eq!(
            exhaustive.numeric_candidates.last().unwrap().feature_index,
            0
        );
        for _ in 0..8 {
            assert_eq!(
                backend
                    .shortlist_standard_splits(
                        &histograms,
                        options,
                        &vec![0.0; feature_count],
                        &[],
                        feature_count,
                    )
                    .expect("tied shortlist succeeds")
                    .numeric_candidates
                    .iter()
                    .map(|candidate| candidate.feature_index)
                    .collect::<Vec<_>>(),
                (0..feature_count as u32).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn shortlist_standard_keeps_categorical_overall_out_of_numeric_candidates() {
    let backend = CpuBackend;
    let mut features = shortlist_standard_fixture(2)
        .features()
        .map(|view| {
            let mut bins = view
                .grad_sums()
                .iter()
                .zip(view.hess_sums())
                .zip(view.counts())
                .map(|((&grad_sum, &hess_sum), &count)| HistogramBin {
                    grad_sum,
                    hess_sum,
                    grad_sq_sum: 0.0,
                    count,
                })
                .collect::<Vec<_>>();
            bins.push(HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            });
            FeatureHistogram {
                feature_index: view.feature_index(),
                bins,
            }
        })
        .collect::<Vec<_>>();
    let mut categorical_bins = vec![
        HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        4
    ];
    categorical_bins[0] = HistogramBin {
        grad_sum: 50.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 2,
    };
    categorical_bins[1] = HistogramBin {
        grad_sum: -50.0,
        hess_sum: 2.0,
        grad_sq_sum: 0.0,
        count: 2,
    };
    features.push(FeatureHistogram {
        feature_index: 2,
        bins: categorical_bins,
    });
    let histograms = HistogramBundle::from_feature_histograms(17, features, false)
        .expect("mixed shortlist fixture is valid");
    let options = SplitSelectionOptions {
        missing_bin_index: 3,
        ..SplitSelectionOptions::default()
    };
    let categorical = [CategoricalFeatureInfo {
        feature_index: 2,
        num_categories: 2,
    }];

    let shortlist = backend
        .shortlist_standard_splits(&histograms, options, &[], &categorical, 8)
        .expect("mixed shortlist succeeds");

    assert!(shortlist.best_overall.as_ref().unwrap().is_categorical);
    assert_eq!(shortlist.numeric_candidates.len(), 2);
    assert!(
        shortlist
            .numeric_candidates
            .iter()
            .all(|candidate| !candidate.is_categorical)
    );
}

#[test]
fn shortlisted_linear_feature_matches_owned_histogram_and_leaf_oracle() {
    let backend = CpuBackend;
    let binned =
        BinnedMatrix::new(4, 1, 1, vec![0_u8, 0, 1, 1]).expect("shortlisted PL matrix is valid");
    let gradients = vec![
        GradientPair {
            grad: 2.0,
            hess: 1.0,
        },
        GradientPair {
            grad: 2.0,
            hess: 1.0,
        },
        GradientPair {
            grad: -2.0,
            hess: 1.0,
        },
        GradientPair {
            grad: -2.0,
            hess: 1.0,
        },
    ];
    let node = sample_node();
    let raw = vec![1.0; 4];
    let scaler = LinearFeatureScaler::identity(1);
    let regressors = vec![0];
    let context = LinearContext {
        regressor_features: regressors.clone(),
        l2_lambda: 1.0,
    };
    let options = SplitSelectionOptions {
        l2_lambda: 1.0,
        missing_bin_index: binned.missing_bin() as usize,
        ..SplitSelectionOptions::default()
    };
    let owned = backend
        .build_linear_histograms(
            &binned,
            &gradients,
            &node,
            &[FeatureTile::new(0, 1).expect("single feature tile")],
            &regressors,
            &scaler,
            &raw,
            4,
            1,
        )
        .expect("owned linear histogram builds");
    let oracle_split = backend
        .best_split_linear(&owned, options, &[], &[], &context)
        .expect("owned split search succeeds")
        .expect("owned split exists");
    let oracle_leaves = backend
        .compute_linear_leaf_pair(
            &owned,
            oracle_split.feature_index,
            oracle_split.threshold_bin as usize,
            oracle_split.default_left,
            options.missing_bin_index,
            0.1,
            context.l2_lambda,
            &scaler,
        )
        .expect("owned leaves solve");

    let prepared = backend
        .evaluate_shortlisted_linear_feature(
            &binned, &gradients, &node, 0, &context, &scaler, &raw, 4, 1, options, 0.1,
        )
        .expect("shortlisted evaluation succeeds")
        .expect("shortlisted split exists");

    assert_eq!(prepared.split, oracle_split);
    assert_eq!(prepared.split.threshold_bin, 0);
    assert!(prepared.split.default_left);
    assert!((prepared.split.gain - (16.0 / 3.0)).abs() < 1e-5);
    assert_eq!(prepared.left_leaf.regressor_features, regressors);
    assert_eq!((prepared.left_leaf, prepared.right_leaf), oracle_leaves);
}

fn legacy_parallel_partition_with_stats(
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    split: &SplitCandidate,
) -> (PartitionResult, NodeStats, NodeStats) {
    type ChunkResult = (Vec<u32>, Vec<u32>, f32, f32, f32, f32, f32, f32);

    let chunk_size = (node.row_indices.len() / rayon::current_num_threads().max(1)).max(4096);
    let chunk_results: Vec<ChunkResult> = node
        .row_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut left_grad = 0.0_f32;
            let mut left_hess = 0.0_f32;
            let mut left_grad_sq = 0.0_f32;
            let mut right_grad = 0.0_f32;
            let mut right_hess = 0.0_f32;
            let mut right_grad_sq = 0.0_f32;
            let feature_index = split.feature_index as usize;
            let use_col_major = binned_matrix.has_col_major();
            let column_base = feature_index * binned_matrix.row_count;
            let missing_bin = binned_matrix.missing_bin();

            for &row in chunk {
                let row_index = row as usize;
                let bin = if use_col_major {
                    binned_matrix.col_bin(column_base + row_index)
                } else {
                    binned_matrix.row_bin(row_index * binned_matrix.feature_count + feature_index)
                };
                let gradient = gradients[row_index];
                if goes_left_for_split(bin, missing_bin, split) {
                    left.push(row);
                    left_grad += gradient.grad;
                    left_hess += gradient.hess;
                    left_grad_sq += gradient.grad * gradient.grad;
                } else {
                    right.push(row);
                    right_grad += gradient.grad;
                    right_hess += gradient.hess;
                    right_grad_sq += gradient.grad * gradient.grad;
                }
            }

            (
                left,
                right,
                left_grad,
                left_hess,
                left_grad_sq,
                right_grad,
                right_hess,
                right_grad_sq,
            )
        })
        .collect();

    let mut left_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
    let mut right_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
    let mut left_grad = 0.0_f32;
    let mut left_hess = 0.0_f32;
    let mut left_grad_sq = 0.0_f32;
    let mut right_grad = 0.0_f32;
    let mut right_hess = 0.0_f32;
    let mut right_grad_sq = 0.0_f32;

    for (left, right, chunk_lg, chunk_lh, chunk_lq, chunk_rg, chunk_rh, chunk_rq) in chunk_results {
        left_row_indices.extend(left);
        right_row_indices.extend(right);
        left_grad += chunk_lg;
        left_hess += chunk_lh;
        left_grad_sq += chunk_lq;
        right_grad += chunk_rg;
        right_hess += chunk_rh;
        right_grad_sq += chunk_rq;
    }

    let left_count = left_row_indices.len() as u32;
    let right_count = right_row_indices.len() as u32;
    (
        PartitionResult {
            left_row_indices,
            right_row_indices,
        },
        NodeStats {
            grad_sum: left_grad,
            hess_sum: left_hess,
            grad_sq_sum: left_grad_sq,
            row_count: left_count,
        },
        NodeStats {
            grad_sum: right_grad,
            hess_sum: right_hess,
            grad_sq_sum: right_grad_sq,
            row_count: right_count,
        },
    )
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
fn bundled_histogram_kernel_matches_unbundled_feature_histograms() {
    let matrix = BinnedMatrix::new(
        6,
        4,
        1,
        vec![
            1, 0, 0, 0, //
            0, 1, 0, 0, //
            0, 0, 1, 0, //
            0, 0, 0, 1, //
            1, 0, 0, 0, //
            0, 1, 0, 0,
        ],
    )
    .expect("fixture");
    let map = discover_exact_feature_bundles(&matrix, &[false; 4]).expect("bundle map");
    let bundled = matrix
        .clone()
        .with_exact_feature_bundles(map)
        .expect("bundled matrix");
    let gradients = vec![
        GradientPair::new(1.0, 1.0).expect("gradient"),
        GradientPair::new(-0.5, 1.5).expect("gradient"),
        GradientPair::new(2.0, 0.75).expect("gradient"),
        GradientPair::new(-1.0, 2.0).expect("gradient"),
        GradientPair::new(0.25, 1.25).expect("gradient"),
        GradientPair::new(-2.0, 0.5).expect("gradient"),
    ];
    let node = NodeSlice::new(7, (0..6).collect()).expect("node");
    let tile = FeatureTile::new(0, 4).expect("tile");

    let expected =
        CpuBackend::build_feature_histograms_for_tile(&matrix, &gradients, &node, &tile, 2, true)
            .expect("unbundled histograms");
    let actual = CpuBackend::build_feature_histograms_for_bundled_tile(
        &bundled, &gradients, &node, &tile, 2, true,
    )
    .expect("bundled histograms");

    assert_eq!(actual, expected);
}

#[test]
fn bundled_histograms_preserve_logical_missing_bins_on_unbundled_features() {
    let matrix = BinnedMatrix::new(
        8,
        3,
        u16::from(MISSING_BIN_U8),
        vec![
            200,
            0,
            MISSING_BIN_U8,
            0,
            200,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    .expect("fixture");
    let map = discover_exact_feature_bundles(&matrix, &[false; 3]).expect("bundle map");
    let bundled = matrix
        .clone()
        .with_exact_feature_bundles(map)
        .expect("bundled matrix");
    let gradients = vec![GradientPair::new(1.0, 1.0).expect("gradient"); 8];
    let node = NodeSlice::new(8, (0..8).collect()).expect("node");
    let tile = FeatureTile::new(0, 3).expect("tile");

    let expected = CpuBackend::build_feature_histograms_for_tile(
        &matrix, &gradients, &node, &tile, 256, false,
    )
    .expect("unbundled histograms");
    let actual = CpuBackend::build_feature_histograms_for_bundled_tile(
        &bundled, &gradients, &node, &tile, 256, false,
    )
    .expect("bundled histograms");

    assert_eq!(actual, expected);
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
fn split_gain_comparison_treats_f32_noise_as_a_tie() {
    assert!(!gain_materially_exceeds(1.0 + 5e-7, 1.0));
    assert!(gain_materially_exceeds(1.0 + 2e-6, 1.0));
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
fn owned_partition_reuses_parent_storage_for_numeric_thresholds() {
    let backend = CpuBackend;
    let matrix = BinnedMatrix::new(5, 1, 4, vec![0, 1, 2, 3, 4]).expect("valid matrix");
    let gradients = vec![GradientPair::new(1.0, 1.0).expect("gradient"); 5];
    let mut rows = Vec::with_capacity(8);
    rows.extend(0..5);
    let node = NodeSlice::new(0, rows).expect("node");
    let split = split_candidate(0, 2, false, false, None);

    let expected = backend
        .apply_split_with_stats(&matrix, &gradients, &node, &split)
        .expect("borrowed partition");
    let parent_ptr = node.row_indices.as_ptr();
    let actual = backend
        .apply_split_owned_with_stats(&matrix, &gradients, node, &split)
        .expect("owned partition");

    assert_eq!(actual, expected);
    assert_eq!(actual.0.right_row_indices.as_ptr(), parent_ptr);
}

#[test]
fn owned_partition_compaction_is_stable_and_preserves_parent_allocation() {
    let mut rows = Vec::with_capacity(12);
    rows.extend(0..8);
    let parent_ptr = rows.as_ptr();
    let parent_capacity = rows.capacity();

    let (left, right) = CpuBackend::stable_partition_owned_rows(rows, 4, |row| row % 2 == 1);

    assert_eq!(left, vec![1, 3, 5, 7]);
    assert_eq!(right, vec![0, 2, 4, 6]);
    assert_eq!(right.as_ptr(), parent_ptr);
    assert_eq!(right.capacity(), parent_capacity);

    let (left, right) = CpuBackend::stable_partition_owned_rows(vec![2, 1, 0], 3, |_| true);
    assert_eq!(left, vec![2, 1, 0]);
    assert!(right.is_empty());

    let (left, right) = CpuBackend::stable_partition_owned_rows(vec![2, 1, 0], 0, |_| false);
    assert!(left.is_empty());
    assert_eq!(right, vec![2, 1, 0]);
}

#[test]
fn owned_partition_reuses_parent_storage_for_missing_values_in_both_directions() {
    let backend = CpuBackend;
    let missing = u16::from(MISSING_BIN_U8);
    let matrix =
        BinnedMatrix::new(4, 1, missing, vec![0, MISSING_BIN_U8, 2, 3]).expect("valid matrix");
    let gradients = vec![GradientPair::new(1.0, 1.0).expect("gradient"); 4];

    for default_left in [false, true] {
        let mut rows = Vec::with_capacity(8);
        rows.extend(0..4);
        let node = NodeSlice::new(0, rows).expect("node");
        let split = split_candidate(0, 1, default_left, false, None);

        let expected = backend
            .apply_split_with_stats(&matrix, &gradients, &node, &split)
            .expect("borrowed partition");
        let parent_ptr = node.row_indices.as_ptr();
        let actual = backend
            .apply_split_owned_with_stats(&matrix, &gradients, node, &split)
            .expect("owned partition");

        assert_eq!(actual, expected);
        assert_eq!(actual.0.right_row_indices.as_ptr(), parent_ptr);
    }
}

#[test]
fn owned_partition_reuses_parent_storage_for_categorical_bitsets() {
    let backend = CpuBackend;
    let matrix = BinnedMatrix::new(6, 1, 2, vec![0, 1, 2, 0, 1, 2]).expect("valid matrix");
    let gradients = vec![GradientPair::new(1.0, 1.0).expect("gradient"); 6];
    let mut rows = Vec::with_capacity(8);
    rows.extend(0..6);
    let node = NodeSlice::new(0, rows).expect("node");
    let split = split_candidate(0, 0, true, true, Some(vec![0b0000_0011]));

    let expected = backend
        .apply_split_with_stats(&matrix, &gradients, &node, &split)
        .expect("borrowed partition");
    let parent_ptr = node.row_indices.as_ptr();
    let actual = backend
        .apply_split_owned_with_stats(&matrix, &gradients, node, &split)
        .expect("owned partition");

    assert_eq!(actual, expected);
    assert_eq!(actual.0.right_row_indices.as_ptr(), parent_ptr);
}

#[test]
fn owned_partition_matches_parallel_chunk_statistic_order() {
    const ROWS: usize = 50_000;

    let backend = CpuBackend;
    let bins = (0..ROWS).map(|row| (row % 2) as u8).collect();
    let matrix = BinnedMatrix::new(ROWS, 1, 1, bins).expect("valid matrix");
    let gradients = (0..ROWS)
        .map(|row| {
            let magnitude = if row % 2 == 0 { 1.0e20 } else { 1.0 };
            let sign = match (row / 2) % 3 {
                1 => -1.0,
                _ => 1.0,
            };
            GradientPair::new(sign * magnitude, 1.0 + (row % 2) as f32).expect("gradient")
        })
        .collect::<Vec<_>>();
    let node = NodeSlice::new(0, (0..ROWS as u32).collect()).expect("node");
    let split = split_candidate(0, 0, false, false, None);

    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool")
        .install(|| {
            assert_eq!(rayon::current_num_threads(), 4);
            let expected = legacy_parallel_partition_with_stats(&matrix, &gradients, &node, &split);
            let actual = backend
                .apply_split_owned_with_stats(&matrix, &gradients, node, &split)
                .expect("owned partition");

            assert_eq!(actual, expected);
        });
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
        (Some(a), Some(b)) => assert_split_candidates_match(Some(&a), Some(&b)),
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
        min_leaf_magnitude: 0.1,
        dro_config: None,
        missing_bin_index: 3,
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
    assert_split_candidates_match(standard.as_ref(), morph_result.as_ref());
}

#[test]
fn morph_standard_scanner_shortcut_excludes_dro_and_factor_penalties() {
    use alloygbm_core::{DroConfig, DroMetric, MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let cfg = MorphConfig::default();
    let morph = MorphContext {
        iteration: 0,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: cfg,
        precomputed: MorphPrecomputed::for_iteration(0, 100, &cfg),
    };
    let ordinary = SplitSelectionOptions::default();
    assert!(crate::backend_ops::morph_can_use_standard_scanner(
        &morph, &ordinary, false
    ));

    let with_dro = SplitSelectionOptions {
        dro_config: Some(DroConfig {
            radius: 0.05,
            metric: DroMetric::Wasserstein,
        }),
        ..ordinary
    };
    assert!(!crate::backend_ops::morph_can_use_standard_scanner(
        &morph, &with_dro, false
    ));
    assert!(!crate::backend_ops::morph_can_use_standard_scanner(
        &morph, &ordinary, true
    ));
}

#[test]
fn best_split_morph_with_dro_uses_robust_gradient_gain_signal() {
    use alloygbm_core::{DroConfig, DroMetric, MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let backend = CpuBackend;
    let histograms = backend
        .build_histograms_with_grad_sq(
            &sample_binned_matrix(),
            &sample_gradients(),
            &sample_node(),
            &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            true,
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
    let parent_gradient_sum = leaf_effective_gradient(
        split.left_stats.grad_sum + split.right_stats.grad_sum,
        split.left_stats.grad_sq_sum + split.right_stats.grad_sq_sum,
        split.left_stats.row_count + split.right_stats.row_count,
        options.l1_alpha,
        options.dro_config.as_ref(),
    );
    let expected = compute_morph_gain(
        MorphGainInputs {
            parent: SplitSideStats {
                gain_gradient_sum: parent_gradient_sum,
                info_gradient_sum: parent_gradient_sum,
                hessian_sum: split.left_stats.hess_sum + split.right_stats.hess_sum,
                count: split.left_stats.row_count + split.right_stats.row_count,
            },
            left: SplitSideStats {
                gain_gradient_sum: left_gradient_sum,
                info_gradient_sum: left_gradient_sum,
                hessian_sum: split.left_stats.hess_sum,
                count: split.left_stats.row_count,
            },
            right: SplitSideStats {
                gain_gradient_sum: right_gradient_sum,
                info_gradient_sum: right_gradient_sum,
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
    bins[nan_bin] = HistogramBin {
        grad_sum: 0.75,
        hess_sum: 1.0,
        grad_sq_sum: 0.75 * 0.75,
        count: 3,
    };

    let feature_histogram = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let histograms = HistogramBundle::from_feature_histograms(0, vec![feature_histogram], true)
        .expect("valid histogram bundle");

    let options = SplitSelectionOptions {
        l2_lambda: 0.7,
        l1_alpha: 0.25,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.05,
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
    assert_split_candidates_match(Some(&a), Some(&b));
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

fn assert_split_candidates_match(scalar: Option<&SplitCandidate>, simd: Option<&SplitCandidate>) {
    match (scalar, simd) {
        (Some(scalar), Some(simd)) => {
            assert_eq!(scalar.feature_index, simd.feature_index);
            assert_eq!(scalar.threshold_bin, simd.threshold_bin);
            assert_eq!(scalar.default_left, simd.default_left);
            assert!((scalar.gain - simd.gain).abs() < 1e-4);
            assert_eq!(scalar.left_stats, simd.left_stats);
            assert_eq!(scalar.right_stats, simd.right_stats);
        }
        (None, None) => {}
        (scalar, simd) => panic!(
            "scalar/simd disagree on Some-ness: scalar={}, simd={}",
            scalar.is_some(),
            simd.is_some()
        ),
    }
}

fn standard_simd_and_scalar_candidates(
    feature: &FeatureHistogram,
    options: SplitSelectionOptions,
) -> (Option<SplitCandidate>, Option<SplitCandidate>) {
    let scalar = with_histogram_feature(feature, |view| {
        CpuBackend::best_split_for_feature_inner(view, 0, options, GainStrategy::Standard, None)
    });
    let simd = with_histogram_feature(feature, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 0, options)
    });

    assert_split_candidates_match(scalar.as_ref(), simd.as_ref());
    (scalar, simd)
}

#[test]
fn morph_simd_matches_scalar_on_fixed_histogram() {
    use alloygbm_core::{MorphConfig, MorphPrecomputed};
    use alloygbm_engine::MorphContext;

    let feature = FeatureHistogram {
        feature_index: 3,
        bins: (0..17)
            .map(|index| HistogramBin {
                grad_sum: ((index as f32 - 8.0) * 0.37).sin(),
                hess_sum: 0.5 + index as f32 * 0.1,
                grad_sq_sum: 0.0,
                count: 2 + index as u32 % 5,
            })
            .collect(),
    };
    let config = MorphConfig {
        morph_warmup_iters: 0,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 40,
        total_iterations: 100,
        grad_mean: 0.05,
        grad_std: 0.8,
        config,
        precomputed: MorphPrecomputed::for_iteration(40, 100, &config),
    };
    let options = make_options(0.1, 0.2, 0.1, 0.01, 16);
    let scalar = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            7,
            options,
            GainStrategy::Morph(&morph),
            None,
        )
    });
    let simd = with_histogram_feature(&feature, |view| {
        crate::morph_scan::best_split_morph_numeric_simd(view, 7, options, &morph)
    });

    assert_split_candidates_match(scalar.as_ref(), simd.as_ref());
}

fn morph_test_context(iteration: u32, balance_penalty: bool) -> MorphContext {
    let config = MorphConfig {
        morph_warmup_iters: 5,
        balance_penalty,
        ..MorphConfig::default()
    };
    MorphContext {
        iteration,
        total_iterations: 100,
        grad_mean: -0.03,
        grad_std: 0.7,
        config,
        precomputed: MorphPrecomputed::for_iteration(iteration, 100, &config),
    }
}

fn assert_morph_split_parity(scalar: Option<&SplitCandidate>, simd: Option<&SplitCandidate>) {
    match (scalar, simd) {
        (Some(scalar), Some(simd)) => {
            assert_eq!(scalar.feature_index, simd.feature_index);
            assert_eq!(scalar.threshold_bin, simd.threshold_bin);
            assert_eq!(scalar.default_left, simd.default_left);
            let tolerance = 1e-5_f32.max(1e-5 * scalar.gain.abs());
            assert!(
                (scalar.gain - simd.gain).abs() <= tolerance,
                "Morph gain drift: scalar={} simd={} tolerance={tolerance}",
                scalar.gain,
                simd.gain
            );
            assert_eq!(scalar.left_stats, simd.left_stats);
            assert_eq!(scalar.right_stats, simd.right_stats);
        }
        (None, None) => {}
        (scalar, simd) => panic!(
            "Morph scalar/SIMD disagree on Some-ness: scalar={}, simd={}",
            scalar.is_some(),
            simd.is_some()
        ),
    }
}

fn morph_simd_and_scalar_candidates(
    feature: &FeatureHistogram,
    options: SplitSelectionOptions,
    morph: &MorphContext,
) -> (Option<SplitCandidate>, Option<SplitCandidate>) {
    let scalar = with_histogram_feature(feature, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            11,
            options,
            GainStrategy::Morph(morph),
            None,
        )
    });
    let simd = with_histogram_feature(feature, |view| {
        crate::morph_scan::best_split_morph_numeric_simd(view, 11, options, morph)
    });
    assert_morph_split_parity(scalar.as_ref(), simd.as_ref());
    (scalar, simd)
}

#[test]
fn morph_simd_matches_scalar_across_randomized_histograms() {
    for &bin_count in &[16_usize, 64, 255] {
        for seed in 0..8_u32 {
            let mut state = 0xA341_316C_u32 ^ seed ^ bin_count as u32;
            let mut bins = Vec::with_capacity(bin_count);
            for _ in 0..bin_count {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let gradient = ((state >> 8) as i16 as f32) / 8192.0;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bins.push(HistogramBin {
                    grad_sum: gradient,
                    hess_sum: 0.05 + (state % 2_000) as f32 / 1_000.0,
                    grad_sq_sum: gradient * gradient,
                    count: 1 + state % 13,
                });
            }
            let feature = FeatureHistogram {
                feature_index: seed,
                bins,
            };
            for &iteration in &[0_u32, 25, 99] {
                let morph = morph_test_context(iteration, seed % 2 == 0);
                let missing_bin = if seed % 2 == 0 {
                    bin_count - 1
                } else {
                    bin_count
                };
                let options = make_options(
                    if seed % 3 == 0 { 0.15 } else { 0.0 },
                    0.2,
                    if seed % 4 == 0 { 0.1 } else { 0.0 },
                    if seed % 5 == 0 { 0.01 } else { 0.0 },
                    missing_bin,
                );
                morph_simd_and_scalar_candidates(&feature, options, &morph);
            }
        }
    }
}

#[test]
fn morph_simd_masks_tail_invalid_and_non_finite_candidates() {
    let all_invalid = FeatureHistogram {
        feature_index: 0,
        bins: (0..11)
            .map(|index| HistogramBin {
                grad_sum: index as f32 - 5.0,
                hess_sum: 0.1,
                grad_sq_sum: 0.0,
                count: 1,
            })
            .collect(),
    };
    let morph = morph_test_context(40, true);
    let options = SplitSelectionOptions {
        min_rows_per_leaf: 20,
        missing_bin_index: 10,
        ..SplitSelectionOptions::default()
    };
    let (scalar, simd) = morph_simd_and_scalar_candidates(&all_invalid, options, &morph);
    assert!(scalar.is_none() && simd.is_none());

    let non_finite = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: 1.0e30,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 4,
            },
            HistogramBin {
                grad_sum: -1.0e30,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 4,
            },
        ],
    };
    let (scalar, simd) =
        morph_simd_and_scalar_candidates(&non_finite, make_options(0.0, 0.0, 0.0, 0.0, 2), &morph);
    assert!(scalar.is_none() && simd.is_none());
}

#[test]
fn morph_simd_preserves_balance_boundary_behavior() {
    let feature = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: 3.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 9,
            },
            HistogramBin {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -2.0,
                hess_sum: 8.0,
                grad_sq_sum: 0.0,
                count: 90,
            },
        ],
    };
    let morph = morph_test_context(50, true);
    morph_simd_and_scalar_candidates(&feature, make_options(0.0, 0.1, 0.0, 0.0, 3), &morph);
}

#[test]
#[ignore = "release-mode MorphBoost profiling"]
fn benchmark_morph_categorical_scan() {
    use std::hint::black_box;
    use std::time::Instant;

    let num_categories = 64;
    let missing_bin = num_categories;
    let mut bins: Vec<HistogramBin> = (0..num_categories)
        .map(|category| {
            let gradient = ((category as f32) * 0.37).sin() * 4.0;
            HistogramBin {
                grad_sum: gradient,
                hess_sum: 0.5 + category as f32 * 0.03,
                grad_sq_sum: 0.0,
                count: 2 + category as u32 % 11,
            }
        })
        .collect();
    bins.push(HistogramBin {
        grad_sum: 0.75,
        hess_sum: 1.5,
        grad_sq_sum: 0.0,
        count: 7,
    });
    let feature = FeatureHistogram {
        feature_index: 0,
        bins,
    };
    let options = make_options(0.05, 0.1, 0.0, 0.0, missing_bin);
    let morph = morph_test_context(50, true);
    with_histogram_feature(&feature, |view| {
        for _ in 0..3 {
            black_box(CpuBackend::best_split_morph_categorical_feature(
                view,
                0,
                &options,
                num_categories,
                &morph,
                None,
            ));
        }
        for _ in 0..7 {
            let started = Instant::now();
            for _ in 0..2_048 {
                black_box(CpuBackend::best_split_morph_categorical_feature(
                    view,
                    0,
                    &options,
                    num_categories,
                    &morph,
                    None,
                ));
            }
            let ns_per_iter = started.elapsed().as_nanos() as f64 / 2_048.0;
            println!("benchmark_morph_categorical_scan: ns_per_iter={ns_per_iter:.2}");
        }
    });
}

#[test]
fn split_scan_simd_matches_scalar_across_bin_counts_missing_directions_and_ties() {
    let mut missing_left_fixture = None;
    let mut missing_right_fixture = None;
    for &bin_count in &[2_usize, 7, 8, 9, 63, 255, 65_535] {
        let mut state = 0x9E37_79B9_u32 ^ bin_count as u32;
        let mut bins = Vec::with_capacity(bin_count);
        for _ in 0..bin_count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let grad_sum = ((state >> 8) as i16 as f32) / 4096.0;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let hess_sum = 0.5 + (state % 10_000) as f32 / 10_000.0;
            bins.push(HistogramBin {
                grad_sum,
                hess_sum,
                grad_sq_sum: 0.0,
                count: 1 + state % 11,
            });
        }
        let feature = FeatureHistogram {
            feature_index: 0,
            bins,
        };

        for missing_bin_index in [bin_count, bin_count - 1] {
            let (scalar, _) = standard_simd_and_scalar_candidates(
                &feature,
                make_options(0.05, 0.1, 0.0, 0.0, missing_bin_index),
            );
            match (bin_count, missing_bin_index) {
                (8, 7) => missing_left_fixture = scalar,
                (7, 6) => missing_right_fixture = scalar,
                _ => {}
            }
        }
    }

    let missing_left_fixture =
        missing_left_fixture.expect("seeded missing-left fixture should produce a split");
    assert_eq!(missing_left_fixture.threshold_bin, 3);
    assert!(missing_left_fixture.default_left);

    let missing_right_fixture =
        missing_right_fixture.expect("seeded missing-right fixture should produce a split");
    assert_eq!(missing_right_fixture.threshold_bin, 0);
    assert!(!missing_right_fixture.default_left);

    let tied = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: 1.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: 1.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
        ],
    };
    let (scalar_tie, simd_tie) =
        standard_simd_and_scalar_candidates(&tied, make_options(0.0, 0.1, 0.0, 0.0, 4));
    let scalar_tie = scalar_tie.expect("symmetric fixture should produce a split");
    let simd_tie = simd_tie.expect("symmetric fixture should produce a split");
    assert_eq!(scalar_tie.threshold_bin, 0);
    assert!(scalar_tie.default_left);
    assert_eq!(simd_tie.threshold_bin, 0);
    assert!(simd_tie.default_left);
}

#[test]
fn split_scan_nested_rayon_multi_worker_selection_matches_sequential_scalar_oracle() {
    const FEATURE_COUNT: usize = 16;
    let gradients = [3.0_f32, 2.0, 1.0, -1.0, -2.0, -3.0, 1.0, -1.0, 0.5];
    let features = (0..FEATURE_COUNT)
        .map(|feature_index| {
            let scale = feature_index as f32 + 1.0;
            FeatureHistogram {
                feature_index: feature_index as u32,
                bins: gradients
                    .iter()
                    .enumerate()
                    .map(|(bin_index, &gradient)| HistogramBin {
                        grad_sum: gradient * scale,
                        hess_sum: 1.0 + bin_index as f32 * 0.125,
                        grad_sq_sum: 0.0,
                        count: 3 + bin_index as u32,
                    })
                    .collect(),
            }
        })
        .collect();
    let histograms = HistogramBundle::from_feature_histograms(7, features, true)
        .expect("parallel split fixture");
    assert!(histograms.feature_count() >= CpuBackend::PARALLEL_SPLIT_FEATURE_THRESHOLD);
    let options = make_options(0.05, 0.1, 0.0, 0.0, 8);
    let expected = histograms
        .features()
        .filter_map(|feature| {
            CpuBackend::best_split_for_feature_inner(
                feature,
                histograms.node_id,
                options,
                GainStrategy::Standard,
                None,
            )
        })
        .reduce(|left, right| {
            if apply_feature_weight(&right, &[]) > apply_feature_weight(&left, &[]) {
                right
            } else {
                left
            }
        });

    let nested_histograms = histograms.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("nested split test pool");
    pool.spawn(move || {
        let actual = CpuBackend::best_split_with_options_internal(
            &nested_histograms,
            options,
            &[],
            &[],
            None,
        );
        sender
            .send((
                rayon::current_num_threads(),
                rayon::current_thread_index(),
                actual,
            ))
            .expect("test receiver remains live");
    });
    let (thread_count, worker_index, actual) = receiver.recv().expect("nested split result");

    assert_eq!(thread_count, 4);
    assert!(worker_index.is_some());
    assert_split_candidates_match(expected.as_ref(), actual.as_ref());
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

fn dro_scan_options(
    radius: f32,
    l1_alpha: f32,
    l2_lambda: f32,
    min_rows_per_leaf: usize,
    min_child_hessian: f32,
    min_leaf_magnitude: f32,
    missing_bin_index: usize,
) -> SplitSelectionOptions {
    SplitSelectionOptions {
        l1_alpha,
        l2_lambda,
        min_child_hessian,
        min_rows_per_leaf,
        min_leaf_magnitude,
        dro_config: Some(alloygbm_core::DroConfig {
            radius,
            metric: alloygbm_core::DroMetric::Wasserstein,
        }),
        missing_bin_index,
    }
}

fn dro_random_feature(bin_count: usize, seed: u32, missing_bin_index: usize) -> FeatureHistogram {
    let mut state = 0xA341_316C_u32 ^ seed ^ bin_count as u32;
    let bins = (0..bin_count)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let gradient = ((state >> 8) as i16 as f32) / 4096.0;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let count = 1 + state % 13;
            let hess_sum = 0.05 + (state % 2_000) as f32 / 1_000.0;
            let variance_per_row = 0.01 + (state % 17) as f32 / 100.0;
            let grad_sq_sum = gradient * gradient / count as f32 + variance_per_row;
            let mut bin = HistogramBin {
                grad_sum: gradient,
                hess_sum,
                grad_sq_sum: grad_sq_sum * count as f32,
                count,
            };
            if index == missing_bin_index {
                bin.grad_sum = -gradient * 1.7;
                bin.hess_sum += 0.4;
                bin.grad_sq_sum += 0.8 * count as f32;
                bin.count += 3;
            }
            bin
        })
        .collect();
    FeatureHistogram {
        feature_index: seed + 3,
        bins,
    }
}

fn assert_dro_split_candidates_match(
    scalar: Option<&SplitCandidate>,
    simd: Option<&SplitCandidate>,
) {
    match (scalar, simd) {
        (Some(scalar), Some(simd)) => {
            assert_eq!(scalar.node_id, simd.node_id);
            assert_eq!(scalar.feature_index, simd.feature_index);
            assert_eq!(scalar.threshold_bin, simd.threshold_bin);
            assert_eq!(scalar.default_left, simd.default_left);
            let tolerance = 1e-5_f32.max(1e-5 * scalar.gain.abs());
            assert!(
                (scalar.gain - simd.gain).abs() <= tolerance,
                "DRO gain drift: scalar={} simd={} tolerance={tolerance}",
                scalar.gain,
                simd.gain
            );
            assert_eq!(scalar.left_stats, simd.left_stats);
            assert_eq!(scalar.right_stats, simd.right_stats);
        }
        (None, None) => {}
        (scalar, simd) => panic!(
            "DRO scalar/SIMD disagree on Some-ness: scalar={}, simd={}",
            scalar.is_some(),
            simd.is_some()
        ),
    }
}

fn dro_simd_and_scalar_candidates(
    feature: &FeatureHistogram,
    node_id: u32,
    options: SplitSelectionOptions,
) -> (Option<SplitCandidate>, Option<SplitCandidate>) {
    let scalar = with_histogram_feature(feature, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            node_id,
            options,
            GainStrategy::Standard,
            None,
        )
    });
    let simd = with_histogram_feature(feature, |view| {
        crate::dro_scan::best_split_dro_numeric_simd(view, node_id, options)
    });
    assert_dro_split_candidates_match(scalar.as_ref(), simd.as_ref());
    (scalar, simd)
}

#[test]
fn dro_simd_matches_scalar_across_bin_counts_radii_regularization_and_missing_directions() {
    for &bin_count in &[16_usize, 64, 255] {
        for seed in 0..4_u32 {
            let missing_bin_index = if seed % 2 == 0 {
                bin_count - 1
            } else {
                bin_count
            };
            let feature = dro_random_feature(bin_count, seed, missing_bin_index);
            for &radius in &[0.001_f32, 0.05, 0.5] {
                for &l1_alpha in &[0.0_f32, 0.1, 1.0] {
                    let options =
                        dro_scan_options(radius, l1_alpha, 0.7, 1, 0.0, 0.0, missing_bin_index);
                    dro_simd_and_scalar_candidates(&feature, 17, options);
                }
            }
        }
    }
}

#[test]
fn dro_simd_routes_missing_mass_left_and_right_with_exact_child_statistics() {
    let make_feature = |missing_gradient: f32| FeatureHistogram {
        feature_index: 4,
        bins: vec![
            HistogramBin {
                grad_sum: 10.0,
                hess_sum: 10.0,
                grad_sq_sum: 10.0,
                count: 10,
            },
            HistogramBin {
                grad_sum: -10.0,
                hess_sum: 10.0,
                grad_sq_sum: 10.0,
                count: 10,
            },
            HistogramBin {
                grad_sum: missing_gradient,
                hess_sum: 10.0,
                grad_sq_sum: 10.0,
                count: 10,
            },
        ],
    };
    let options = dro_scan_options(0.05, 0.0, 1.0, 1, 0.0, 0.0, 2);

    let (missing_left, _) = dro_simd_and_scalar_candidates(&make_feature(10.0), 3, options);
    assert!(
        missing_left
            .as_ref()
            .expect("missing-left split")
            .default_left
    );

    let (missing_right, _) = dro_simd_and_scalar_candidates(&make_feature(-10.0), 3, options);
    assert!(
        !missing_right
            .as_ref()
            .expect("missing-right split")
            .default_left
    );
}

// Golden-reference parent-baseline invariant for the DRO scanner.
//
// The scalar/SIMD parity tests cross-check the two scanners against each other,
// but that shares a blind spot: a parent-term regression applied to *both*
// paths would still pass.  This test instead pins the DRO SIMD gain to an
// independent `alloygbm_core::leaf_gain_term` reference whose parent term is
// computed from the node's OWN aggregates — never `eff(left) + eff(right)` —
// and proves that the sum-of-children baseline is measurably different.  Under
// DRO's non-linear shrinkage the two diverge (they coincide only for a linear
// leaf transform), so this guards the invariant for the DRO path and the
// upcoming PL work.  Mirrors `morph_parent_baseline_matches_standard_leaf_gain_*`.
#[test]
fn dro_simd_parent_baseline_matches_core_leaf_gain_not_sum_of_children() {
    use alloygbm_core::{DroConfig, DroMetric, leaf_effective_gradient, leaf_gain_term};

    // Two data bins whose per-side variances differ enough that DRO shrinkage
    // makes eff(parent) != eff(l) + eff(r) while leaving all three effective
    // gradients clearly nonzero — so the test is sensitive to *any* parent
    // miscalculation, not only the exact sum-of-children value.
    let feature = FeatureHistogram {
        feature_index: 7,
        bins: vec![
            HistogramBin {
                grad_sum: 8.0,
                hess_sum: 4.0,
                grad_sq_sum: 21.6,
                count: 40,
            },
            HistogramBin {
                grad_sum: 1.0,
                hess_sum: 4.0,
                grad_sq_sum: 4.025,
                count: 40,
            },
        ],
    };
    let (radius, l1, l2) = (0.1_f32, 0.0_f32, 0.7_f32);
    let dro = DroConfig {
        radius,
        metric: DroMetric::Wasserstein,
    };
    // `missing_bin_index` past the two bins => no missing mass, single interior
    // split at threshold 0 (left = bin 0, right = bin 1).
    let options = dro_scan_options(radius, l1, l2, 1, 0.0, 0.0, 5);
    let candidate = with_histogram_feature(&feature, |view| {
        crate::dro_scan::best_split_dro_numeric_simd(view, 17, options)
    })
    .expect("dro split exists");
    assert_eq!(candidate.threshold_bin, 0);

    // Independent reference: 2*leaf_gain_term per side, parent from its own
    // aggregates. `leaf_gain_term` is `0.5 * eff^2/(h+l2+eps)`, so `2*` matches
    // the scanner's `eff^2/(h+l2+eps)` gain term exactly.
    let lgt = |g, h, gsq, c| leaf_gain_term(g, h, gsq, c, l1, l2, Some(&dro));
    let (gp, hp, gsp, cp) = (8.0_f32 + 1.0, 4.0_f32 + 4.0, 21.6_f32 + 4.025, 80_u32);
    let reference =
        2.0 * lgt(8.0, 4.0, 21.6, 40) + 2.0 * lgt(1.0, 4.0, 4.025, 40) - 2.0 * lgt(gp, hp, gsp, cp);
    assert!(
        (candidate.gain - reference).abs() <= 1e-5 * reference.abs().max(1.0),
        "DRO SIMD gain must equal core::leaf_gain_term with the parent from totals: \
         got {} expected {}",
        candidate.gain,
        reference,
    );

    // The buggy sum-of-children parent baseline must differ, so this actually
    // locks out the regression rather than tautologically passing.
    let eff = |g, gsq, c| leaf_effective_gradient(g, gsq, c, l1, Some(&dro));
    let buggy_parent = eff(8.0, 21.6, 40) + eff(1.0, 4.025, 40);
    let buggy = 2.0 * lgt(8.0, 4.0, 21.6, 40) + 2.0 * lgt(1.0, 4.0, 4.025, 40)
        - buggy_parent * buggy_parent / (hp + l2 + 1e-6);
    assert!(
        (reference - buggy).abs() > 1e-4,
        "fixture must distinguish the sum-of-children parent regression",
    );
}

#[test]
fn dro_simd_handles_tail_lengths_zero_variance_and_cancellation() {
    for tail in 1..=3_usize {
        let bin_count = 4 + tail;
        let feature = dro_random_feature(bin_count, tail as u32 + 40, bin_count);
        dro_simd_and_scalar_candidates(
            &feature,
            19,
            dro_scan_options(0.05, 0.1, 1.0, 1, 0.0, 0.0, bin_count),
        );
    }

    let cancellation = FeatureHistogram {
        feature_index: 5,
        bins: vec![
            HistogramBin {
                grad_sum: 1_000.0,
                hess_sum: 4.0,
                grad_sq_sum: 250_000.0,
                count: 4,
            },
            HistogramBin {
                grad_sum: -800.0,
                hess_sum: 4.0,
                grad_sq_sum: 160_000.0,
                count: 4,
            },
            HistogramBin {
                grad_sum: 200.0,
                hess_sum: 4.0,
                grad_sq_sum: 10_000.0,
                count: 4,
            },
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 4.0,
                grad_sq_sum: 0.0,
                count: 4,
            },
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            },
        ],
    };
    dro_simd_and_scalar_candidates(
        &cancellation,
        23,
        dro_scan_options(0.5, 1.0, 2.0, 2, 0.5, 0.001, 5),
    );
}

#[test]
fn dro_simd_applies_minimum_constraints_and_rejects_invalid_edges() {
    let feature = FeatureHistogram {
        feature_index: 6,
        bins: vec![
            HistogramBin {
                grad_sum: 3.0,
                hess_sum: 1.0,
                grad_sq_sum: 9.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -2.0,
                hess_sum: 1.0,
                grad_sq_sum: 4.0,
                count: 10,
            },
        ],
    };
    let (_, simd) = dro_simd_and_scalar_candidates(
        &feature,
        29,
        dro_scan_options(0.05, 0.0, 1.0, 2, 0.0, 0.0, 3),
    );
    let split = simd.expect("the middle split meets the row minimum");
    assert!(
        split.threshold_bin < 2,
        "the final edge threshold must be rejected"
    );

    let (_, no_rows) = dro_simd_and_scalar_candidates(
        &feature,
        29,
        dro_scan_options(0.05, 0.0, 1.0, 20, 0.0, 0.0, 3),
    );
    assert!(no_rows.is_none());

    let (_, no_hessian) = dro_simd_and_scalar_candidates(
        &feature,
        29,
        dro_scan_options(0.05, 0.0, 1.0, 1, 1.1, 0.0, 3),
    );
    assert!(no_hessian.is_none());

    let (_, no_leaf_magnitude) = dro_simd_and_scalar_candidates(
        &feature,
        29,
        dro_scan_options(0.05, 0.0, 1.0, 1, 0.0, 100.0, 3),
    );
    assert!(no_leaf_magnitude.is_none());
}

#[test]
fn dro_simd_keeps_row_minimum_exact_above_f32_precision_limit() {
    let feature = FeatureHistogram {
        feature_index: 21,
        bins: vec![
            HistogramBin {
                grad_sum: 100.0,
                hess_sum: 16_777_216.0,
                grad_sq_sum: 10_000.0,
                count: 16_777_216,
            },
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -100.0,
                hess_sum: 16_777_217.0,
                grad_sq_sum: 10_000.0,
                count: 16_777_217,
            },
        ],
    };
    let options = dro_scan_options(0.05, 0.0, 1.0, 16_777_217, 0.0, 0.0, 3);

    let (scalar, simd) = dro_simd_and_scalar_candidates(&feature, 23, options);
    let scalar = scalar.expect("exact row minimum should leave one valid threshold");
    assert_eq!(scalar.threshold_bin, 1);
    assert_dro_split_candidates_match(Some(&scalar), simd.as_ref());
}

#[test]
fn dro_simd_masks_non_finite_gain_and_requires_gradient_square_plane() {
    let non_finite = FeatureHistogram {
        feature_index: 7,
        bins: vec![
            HistogramBin {
                grad_sum: 1.0,
                hess_sum: -1e-6,
                grad_sq_sum: 1.0,
                count: 1,
            },
            HistogramBin {
                grad_sum: -1.0,
                hess_sum: 1.0,
                grad_sq_sum: 1.0,
                count: 5,
            },
            HistogramBin {
                grad_sum: 0.2,
                hess_sum: 1.0,
                grad_sq_sum: 0.2,
                count: 5,
            },
        ],
    };
    dro_simd_and_scalar_candidates(
        &non_finite,
        31,
        dro_scan_options(0.05, 0.0, 0.0, 1, -1.0, 0.0, 3),
    );

    let bundle = HistogramBundle::from_feature_histograms(0, vec![non_finite], false)
        .expect("histogram without gradient-square plane");
    let view = bundle
        .feature(0)
        .expect("feature without gradient-square plane");
    assert!(
        crate::dro_scan::best_split_dro_numeric_simd(
            view,
            31,
            dro_scan_options(0.05, 0.0, 0.0, 1, 0.0, 0.0, 3),
        )
        .is_none()
    );
}

#[test]
fn dro_simd_nested_rayon_calls_keep_thread_local_scratch_isolated() {
    let features = (0..16_u32)
        .map(|feature_index| dro_random_feature(17, feature_index + 100, 17))
        .map(|mut feature| {
            feature.feature_index %= 16;
            feature
        })
        .collect::<Vec<_>>();
    let histograms = HistogramBundle::from_feature_histograms(7, features, true)
        .expect("DRO nested parallel histogram fixture");
    let options = dro_scan_options(0.05, 0.1, 0.5, 1, 0.0, 0.0, 17);
    let expected = histograms
        .features()
        .filter_map(|view| {
            CpuBackend::best_split_for_feature_inner(
                view,
                histograms.node_id,
                options,
                GainStrategy::Standard,
                None,
            )
        })
        .reduce(|left, right| {
            if gain_materially_exceeds(right.gain, left.gain) {
                right
            } else {
                left
            }
        });

    let nested_histograms = histograms.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("DRO nested split test pool");
    let actual = pool.install(|| {
        CpuBackend::best_split_with_options_internal(&nested_histograms, options, &[], &[], None)
    });
    assert_dro_split_candidates_match(expected.as_ref(), actual.as_ref());
}

#[test]
fn dro_routing_uses_dedicated_numeric_scanner_and_radius_zero_uses_standard_simd() {
    let feature = dro_random_feature(32, 91, 31);
    let active = dro_scan_options(0.05, 0.1, 0.5, 1, 0.0, 0.0, 31);
    let active_production = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature(view, 11, active, None)
    });
    let active_simd = with_histogram_feature(&feature, |view| {
        crate::dro_scan::best_split_dro_numeric_simd(view, 11, active)
    });
    assert_dro_split_candidates_match(active_simd.as_ref(), active_production.as_ref());

    let radius_zero = dro_scan_options(0.0, 0.1, 0.5, 1, 0.0, 0.0, 31);
    let routed = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature(view, 11, radius_zero, None)
    });
    let standard_simd = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature_standard_simd(view, 11, radius_zero)
    });
    assert_split_candidates_match(routed.as_ref(), standard_simd.as_ref());
}

#[test]
fn dro_factor_and_morph_combinations_retain_scalar_fallbacks() {
    let mut feature = dro_random_feature(8, 113, 7);
    feature.feature_index = 0;
    let options = dro_scan_options(0.05, 0.1, 0.5, 1, 0.0, 0.0, 7);
    let matrix = sample_binned_matrix();
    let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0])
        .expect("factor exposure fixture");
    let node = sample_node();
    let factor_context = FactorSplitContext {
        binned_matrix: &matrix,
        exposures: &exposures,
        row_indices: &node.row_indices,
        factor_penalty: 0.1,
    };
    let factor_routed = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature(view, 13, options, Some(&factor_context))
    });
    let factor_scalar = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            13,
            options,
            GainStrategy::Standard,
            Some(&factor_context),
        )
    });
    assert_dro_split_candidates_match(factor_scalar.as_ref(), factor_routed.as_ref());

    let config = alloygbm_core::MorphConfig {
        morph_warmup_iters: 0,
        ..alloygbm_core::MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 20,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config,
        precomputed: alloygbm_core::MorphPrecomputed::for_iteration(20, 100, &config),
    };
    let morph_routed = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_morph_numeric_feature(view, 13, &options, &morph, None)
    });
    let morph_scalar = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_feature_inner(
            view,
            13,
            options,
            GainStrategy::Morph(&morph),
            None,
        )
    });
    assert_dro_split_candidates_match(morph_scalar.as_ref(), morph_routed.as_ref());
}

#[test]
fn dro_categorical_production_dispatch_retains_scalar_fallback() {
    use alloygbm_engine::CategoricalFeatureInfo;

    let feature = FeatureHistogram {
        feature_index: 0,
        bins: vec![
            HistogramBin {
                grad_sum: -4.0,
                hess_sum: 2.0,
                grad_sq_sum: 8.0,
                count: 20,
            },
            HistogramBin {
                grad_sum: -3.0,
                hess_sum: 2.0,
                grad_sq_sum: 4.5,
                count: 20,
            },
            HistogramBin {
                grad_sum: 3.0,
                hess_sum: 2.0,
                grad_sq_sum: 4.5,
                count: 20,
            },
            HistogramBin {
                grad_sum: 4.0,
                hess_sum: 2.0,
                grad_sq_sum: 8.0,
                count: 20,
            },
            HistogramBin {
                grad_sum: 0.75,
                hess_sum: 1.0,
                grad_sq_sum: 0.75 * 0.75,
                count: 3,
            },
        ],
    };
    let histograms = HistogramBundle::from_feature_histograms(0, vec![feature.clone()], true)
        .expect("valid categorical DRO histogram bundle");
    let options = SplitSelectionOptions {
        l2_lambda: 0.7,
        l1_alpha: 0.25,
        min_child_hessian: 0.0,
        min_rows_per_leaf: 1,
        min_leaf_magnitude: 0.05,
        dro_config: Some(alloygbm_core::DroConfig {
            radius: 0.05,
            metric: alloygbm_core::DroMetric::Wasserstein,
        }),
        missing_bin_index: 4,
    };
    let categorical_features = [CategoricalFeatureInfo {
        feature_index: 0,
        num_categories: 4,
    }];

    let scalar = with_histogram_feature(&feature, |view| {
        CpuBackend::best_split_for_categorical_feature(view, 0, options, 4, None)
    });
    let production = CpuBackend::best_split_with_options_internal(
        &histograms,
        options,
        &[],
        &categorical_features,
        None,
    );

    assert_split_candidates_match(scalar.as_ref(), production.as_ref());
    let scalar = scalar.expect("categorical DRO oracle should produce a split");
    let production = production.expect("categorical DRO production dispatch should split");
    assert!(production.is_categorical);
    assert_eq!(scalar.categorical_bitset, production.categorical_bitset);
}
