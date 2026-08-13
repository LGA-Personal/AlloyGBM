use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{
    BinnedMatrix, DroConfig, DroMetric, FeatureHistogram, FeatureTile, GradientPair, HistogramBin,
    HistogramBundle, LinearFeatureScaler, LinearHistogramBin, MorphConfig, MorphPrecomputed,
    NodeSlice,
};
use alloygbm_engine::{BackendOps, MorphContext, SplitSelectionOptions};
use std::hint::black_box;
use std::time::Instant;

const DISABLE_AVX2_ENV_VAR: &str = "ALLOYGBM_DISABLE_AVX2";

struct BenchmarkFixture {
    binned_matrix: BinnedMatrix,
    gradients: Vec<GradientPair>,
    node: NodeSlice,
    feature_tiles: Vec<FeatureTile>,
}

fn build_fixture(
    row_count: usize,
    feature_count: usize,
    max_bin: u16,
    tile_span: usize,
) -> BenchmarkFixture {
    let mut bins = Vec::with_capacity(row_count * feature_count);
    for row_index in 0..row_count {
        for feature_index in 0..feature_count {
            let value = ((row_index.wrapping_mul(31) + feature_index.wrapping_mul(17))
                % (max_bin as usize + 1)) as u8;
            bins.push(value);
        }
    }

    let mut gradients = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        let grad = ((row_index % 29) as f32 - 14.0) / 7.0;
        let hess = 1.0 + ((row_index % 11) as f32 * 0.05);
        gradients.push(
            GradientPair::new(grad, hess)
                .expect("benchmark fixture must construct finite gradient pair"),
        );
    }

    let node = NodeSlice::new(0, (0..row_count as u32).collect())
        .expect("benchmark fixture node indices must be valid");
    let mut feature_tiles = Vec::new();
    let step = tile_span.max(1);
    let mut start_feature = 0usize;
    while start_feature < feature_count {
        let end_feature = (start_feature + step).min(feature_count);
        feature_tiles.push(
            FeatureTile::new(start_feature as u32, end_feature as u32)
                .expect("feature tile must be valid"),
        );
        start_feature = end_feature;
    }
    let binned_matrix =
        BinnedMatrix::new(row_count, feature_count, max_bin, bins).expect("fixture matrix valid");

    BenchmarkFixture {
        binned_matrix,
        gradients,
        node,
        feature_tiles,
    }
}

fn run_case<F>(name: &str, warmup_iters: usize, measure_iters: usize, mut f: F)
where
    F: FnMut(),
{
    for _ in 0..warmup_iters {
        f();
    }

    let start = Instant::now();
    for _ in 0..measure_iters {
        f();
    }
    let elapsed = start.elapsed();
    let nanos_per_iter = elapsed.as_nanos() as f64 / measure_iters as f64;
    println!(
        "{name}: total_ms={:.3} iterations={measure_iters} ns_per_iter={nanos_per_iter:.2}",
        elapsed.as_secs_f64() * 1_000.0
    );
}

fn build_histograms_baseline_reference(
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    feature_tiles: &[FeatureTile],
) -> HistogramBundle {
    if gradients.len() != binned_matrix.row_count {
        panic!("baseline reference requires gradients length to match row_count");
    }
    if feature_tiles.is_empty() {
        panic!("baseline reference requires non-empty feature_tiles");
    }
    node.validate_bounds(binned_matrix.row_count)
        .expect("baseline reference node bounds must be valid");

    let mut feature_histograms = Vec::new();
    for tile in feature_tiles {
        if tile.end_feature as usize > binned_matrix.feature_count {
            panic!("baseline reference feature tile end must not exceed feature_count");
        }
        for feature_index in tile.start_feature..tile.end_feature {
            let mut bins = vec![
                HistogramBin {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    count: 0,
                };
                binned_matrix.max_bin as usize + 1
            ];

            for &row_index in &node.row_indices {
                let row_index = row_index as usize;
                let cell_index = row_index * binned_matrix.feature_count + feature_index as usize;
                let bin_index = binned_matrix.row_bin(cell_index) as usize;
                let gradient = gradients[row_index];
                let target_bin = &mut bins[bin_index];
                target_bin.grad_sum += gradient.grad;
                target_bin.hess_sum += gradient.hess;
                target_bin.count += 1;
            }

            feature_histograms.push(FeatureHistogram {
                feature_index,
                bins,
            });
        }
    }

    HistogramBundle::from_feature_histograms(node.node_id, feature_histograms, false)
        .expect("reference histograms have a valid layout")
}

fn run_pl_histogram_storage_case(backend: &CpuBackend, feature_count: usize) {
    let fixture = build_fixture(1_024, feature_count, 63, 8);
    let raw_feature_values: Vec<f32> = (0..fixture.binned_matrix.row_count * feature_count)
        .map(|index| ((index % 97) as f32 - 48.0) / 13.0)
        .collect();
    let regressor_features: Vec<u32> = (0..feature_count.min(8) as u32).collect();
    let bundle = backend
        .build_linear_histograms(
            &fixture.binned_matrix,
            &fixture.gradients,
            &fixture.node,
            &fixture.feature_tiles,
            &regressor_features,
            &LinearFeatureScaler::identity(feature_count),
            &raw_feature_values,
            fixture.binned_matrix.row_count,
            feature_count,
        )
        .expect("PL histogram storage benchmark should succeed");
    let retained_bin_capacity: usize = bundle
        .feature_histograms
        .iter()
        .map(|histogram| histogram.bins.capacity())
        .sum();
    let retained_bytes = retained_bin_capacity * std::mem::size_of::<LinearHistogramBin>();
    println!(
        "pl_histogram_storage: production_base=ea4df36 features={feature_count} bins_per_feature={} bin_size_bytes={} retained_bundle_bytes={retained_bytes}",
        fixture.binned_matrix.max_bin as usize + 2,
        std::mem::size_of::<LinearHistogramBin>(),
    );
    black_box(bundle);
}

fn main() {
    let backend = CpuBackend;
    println!("production_base: ea4df36");
    println!("runtime_target_arch: {}", std::env::consts::ARCH);
    println!("runtime_avx2_enabled: {}", runtime_avx2_enabled());
    println!(
        "runtime_avx2_override: {}",
        std::env::var(DISABLE_AVX2_ENV_VAR).unwrap_or_else(|_| "unset".to_string())
    );

    for feature_count in [8, 32, 128] {
        run_pl_histogram_storage_case(&backend, feature_count);
    }

    let tiny_fixture = build_fixture(256, 8, 31, 4);
    run_case("histogram_build_tiny_baseline_ref", 10, 220, || {
        let histograms = build_histograms_baseline_reference(
            &tiny_fixture.binned_matrix,
            &tiny_fixture.gradients,
            &tiny_fixture.node,
            &tiny_fixture.feature_tiles,
        );
        black_box(histograms);
    });
    run_case("histogram_build_tiny_backend", 10, 220, || {
        let histograms = backend
            .build_histograms(
                &tiny_fixture.binned_matrix,
                &tiny_fixture.gradients,
                &tiny_fixture.node,
                &tiny_fixture.feature_tiles,
            )
            .expect("histogram benchmark should succeed");
        black_box(histograms);
    });

    let small_fixture = build_fixture(1_024, 16, 63, 4);
    let morph_16_fixture = build_fixture(1_024, 16, 15, 4);
    run_case("histogram_build_small_baseline_ref", 8, 140, || {
        let histograms = build_histograms_baseline_reference(
            &small_fixture.binned_matrix,
            &small_fixture.gradients,
            &small_fixture.node,
            &small_fixture.feature_tiles,
        );
        black_box(histograms);
    });
    run_case("histogram_build_small_backend", 8, 140, || {
        let histograms = backend
            .build_histograms(
                &small_fixture.binned_matrix,
                &small_fixture.gradients,
                &small_fixture.node,
                &small_fixture.feature_tiles,
            )
            .expect("histogram benchmark should succeed");
        black_box(histograms);
    });

    let medium_fixture = build_fixture(4_096, 128, 255, 8);
    run_case("histogram_build_medium_baseline_ref", 6, 80, || {
        let histograms = build_histograms_baseline_reference(
            &medium_fixture.binned_matrix,
            &medium_fixture.gradients,
            &medium_fixture.node,
            &medium_fixture.feature_tiles,
        );
        black_box(histograms);
    });
    run_case("histogram_build_medium_backend", 6, 80, || {
        let histograms = backend
            .build_histograms(
                &medium_fixture.binned_matrix,
                &medium_fixture.gradients,
                &medium_fixture.node,
                &medium_fixture.feature_tiles,
            )
            .expect("histogram benchmark should succeed");
        black_box(histograms);
    });

    let split_histograms_small = backend
        .build_histograms(
            &small_fixture.binned_matrix,
            &small_fixture.gradients,
            &small_fixture.node,
            &small_fixture.feature_tiles,
        )
        .expect("small split benchmark histogram precompute should succeed");
    let split_histograms_16 = backend
        .build_histograms(
            &morph_16_fixture.binned_matrix,
            &morph_16_fixture.gradients,
            &morph_16_fixture.node,
            &morph_16_fixture.feature_tiles,
        )
        .expect("16-bin split benchmark histogram precompute should succeed");
    let split_histograms_medium = backend
        .build_histograms(
            &medium_fixture.binned_matrix,
            &medium_fixture.gradients,
            &medium_fixture.node,
            &medium_fixture.feature_tiles,
        )
        .expect("medium split benchmark histogram precompute should succeed");
    let dro_histograms_16 = backend
        .build_histograms_with_grad_sq(
            &morph_16_fixture.binned_matrix,
            &morph_16_fixture.gradients,
            &morph_16_fixture.node,
            &morph_16_fixture.feature_tiles,
            true,
        )
        .expect("16-bin DRO split benchmark histogram precompute should succeed");
    let dro_histograms_64 = backend
        .build_histograms_with_grad_sq(
            &small_fixture.binned_matrix,
            &small_fixture.gradients,
            &small_fixture.node,
            &small_fixture.feature_tiles,
            true,
        )
        .expect("64-bin DRO split benchmark histogram precompute should succeed");
    let dro_histograms_255 = backend
        .build_histograms_with_grad_sq(
            &medium_fixture.binned_matrix,
            &medium_fixture.gradients,
            &medium_fixture.node,
            &medium_fixture.feature_tiles,
            true,
        )
        .expect("255-bin DRO split benchmark histogram precompute should succeed");
    run_case("best_split_small", 12, 500, || {
        let split = backend
            .best_split(&split_histograms_small)
            .expect("best split benchmark should succeed");
        black_box(split);
    });
    run_case("best_split_medium", 12, 500, || {
        let split = backend
            .best_split(&split_histograms_medium)
            .expect("best split benchmark should succeed");
        black_box(split);
    });

    let dro_options = SplitSelectionOptions {
        l1_alpha: 0.1,
        l2_lambda: 1.0,
        dro_config: Some(DroConfig {
            radius: 0.05,
            metric: DroMetric::Wasserstein,
        }),
        ..SplitSelectionOptions::default()
    };
    run_case("best_split_dro_16", 12, 500, || {
        let split = backend
            .best_split_with_options(&dro_histograms_16, dro_options, &[], &[])
            .expect("16-bin DRO split benchmark should succeed");
        black_box(split);
    });
    run_case("best_split_dro_64", 12, 500, || {
        let split = backend
            .best_split_with_options(&dro_histograms_64, dro_options, &[], &[])
            .expect("64-bin DRO split benchmark should succeed");
        black_box(split);
    });
    run_case("best_split_dro_255", 12, 500, || {
        let split = backend
            .best_split_with_options(&dro_histograms_255, dro_options, &[], &[])
            .expect("255-bin DRO split benchmark should succeed");
        black_box(split);
    });

    let morph_config = MorphConfig {
        morph_warmup_iters: 0,
        ..MorphConfig::default()
    };
    let morph = MorphContext {
        iteration: 50,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: morph_config,
        precomputed: MorphPrecomputed::for_iteration(50, 100, &morph_config),
    };
    let split_options = SplitSelectionOptions::default();
    run_case("best_split_morph_16", 12, 500, || {
        let split = backend
            .best_split_morph(&split_histograms_16, split_options, &[], &[], &morph)
            .expect("16-bin Morph split benchmark should succeed");
        black_box(split);
    });
    run_case("best_split_morph_64", 12, 500, || {
        let split = backend
            .best_split_morph(&split_histograms_small, split_options, &[], &[], &morph)
            .expect("64-bin Morph split benchmark should succeed");
        black_box(split);
    });
    run_case("best_split_morph_255", 12, 500, || {
        let split = backend
            .best_split_morph(&split_histograms_medium, split_options, &[], &[], &morph)
            .expect("255-bin Morph split benchmark should succeed");
        black_box(split);
    });
}

fn avx2_disabled_by_env() -> bool {
    match std::env::var(DISABLE_AVX2_ENV_VAR) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !(normalized.is_empty()
                || normalized == "0"
                || normalized == "false"
                || normalized == "off")
        }
        Err(_) => false,
    }
}

fn runtime_avx2_enabled() -> bool {
    if avx2_disabled_by_env() {
        return false;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}
