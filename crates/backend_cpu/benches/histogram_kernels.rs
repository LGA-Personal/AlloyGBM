use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{
    BinnedMatrix, FeatureHistogram, FeatureTile, GradientPair, HistogramBin, HistogramBundle,
    NodeSlice,
};
use alloygbm_engine::BackendOps;
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
                    count: 0,
                };
                binned_matrix.max_bin as usize + 1
            ];

            for &row_index in node.row_indices() {
                let row_index = row_index as usize;
                let cell_index = row_index * binned_matrix.feature_count + feature_index as usize;
                let bin_index = binned_matrix.bins[cell_index] as usize;
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

    HistogramBundle::from_cpu(node.node_id, feature_histograms)
}

fn main() {
    let backend = CpuBackend;
    println!("runtime_target_arch: {}", std::env::consts::ARCH);
    println!("runtime_avx2_enabled: {}", runtime_avx2_enabled());
    println!(
        "runtime_avx2_override: {}",
        std::env::var(DISABLE_AVX2_ENV_VAR).unwrap_or_else(|_| "unset".to_string())
    );

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
    let split_histograms_medium = backend
        .build_histograms(
            &medium_fixture.binned_matrix,
            &medium_fixture.gradients,
            &medium_fixture.node,
            &medium_fixture.feature_tiles,
        )
        .expect("medium split benchmark histogram precompute should succeed");
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
