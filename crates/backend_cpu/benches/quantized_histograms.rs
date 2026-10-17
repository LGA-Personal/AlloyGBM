//! Float vs quantized-gradient histogram builds on one thread.
//!
//! `cargo bench -p alloygbm-backend-cpu --bench quantized_histograms`

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
use alloygbm_engine::{BackendOps, HistogramExecution, QuantizedGradients, SplitSelectionOptions};
use std::hint::black_box;
use std::time::Instant;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn main() {
    let rows = 60_000;
    let features = 18;
    let reps = 60;
    for &bins in &[4_u16, 32, 255] {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut column_major = Vec::with_capacity(rows * features);
        for _ in 0..rows * features {
            column_major.push((xorshift(&mut state) % u64::from(bins)) as u8);
        }
        let matrix =
            BinnedMatrix::new_from_column_major(rows, features, bins - 1, column_major).unwrap();
        let gradients = (0..rows)
            .map(|_| GradientPair {
                grad: (xorshift(&mut state) % 2000) as f32 / 1000.0 - 1.0,
                hess: 0.25 + (xorshift(&mut state) % 100) as f32 / 400.0,
            })
            .collect::<Vec<_>>();
        let quantized = QuantizedGradients::quantize(&gradients, 4, 1).unwrap();
        let start = Instant::now();
        for seed in 0..reps as u64 {
            black_box(QuantizedGradients::quantize(&gradients, 4, seed).unwrap());
        }
        println!(
            "quantize {rows} rows: {:.3} ms",
            start.elapsed().as_secs_f64() / reps as f64 * 1e3
        );
        let tiles = [FeatureTile::new(0, features as u32).unwrap()];
        for &fraction in &[1_usize, 8, 32, 128] {
            let node_rows = (0..rows as u32)
                .filter(|_| xorshift(&mut state).is_multiple_of(fraction as u64))
                .collect::<Vec<_>>();
            let node = NodeSlice::new(0, node_rows).unwrap();
            let backend = CpuBackend;
            let time = |quantize: bool| {
                let start = Instant::now();
                for _ in 0..reps {
                    let bundle = if quantize {
                        backend.build_histograms_quantized(
                            &matrix,
                            &quantized,
                            &node,
                            &tiles,
                            HistogramExecution::Sequential,
                        )
                    } else {
                        backend.build_histograms_with_execution(
                            &matrix,
                            &gradients,
                            &node,
                            &tiles,
                            false,
                            HistogramExecution::Sequential,
                        )
                    };
                    black_box(bundle.unwrap());
                }
                start.elapsed().as_secs_f64() / reps as f64 * 1e3
            };
            let mut float_ms = f64::MAX;
            let mut quantized_ms = f64::MAX;
            for _ in 0..5 {
                float_ms = float_ms.min(time(false));
                quantized_ms = quantized_ms.min(time(true));
            }
            let float_bundle = backend
                .build_histograms_with_execution(
                    &matrix,
                    &gradients,
                    &node,
                    &tiles,
                    false,
                    HistogramExecution::Sequential,
                )
                .unwrap();
            let quantized_bundle = backend
                .build_histograms_quantized(
                    &matrix,
                    &quantized,
                    &node,
                    &tiles,
                    HistogramExecution::Sequential,
                )
                .unwrap();
            let options = SplitSelectionOptions {
                missing_bin_index: usize::from(bins),
                min_rows_per_leaf: 16,
                l2_lambda: 1.0,
                ..Default::default()
            };
            let scan = |bundle: &alloygbm_core::HistogramBundle| {
                let start = Instant::now();
                for _ in 0..reps * 10 {
                    black_box(
                        backend
                            .best_split_with_options(bundle, options, &[], &[])
                            .unwrap(),
                    );
                }
                start.elapsed().as_secs_f64() / (reps * 10) as f64 * 1e3
            };
            let (mut scan_float, mut scan_quantized) = (f64::MAX, f64::MAX);
            for _ in 0..5 {
                scan_float = scan_float.min(scan(&float_bundle));
                scan_quantized = scan_quantized.min(scan(&quantized_bundle));
            }
            println!("  split scan float {scan_float:.4} ms quantized {scan_quantized:.4} ms");
            println!(
                "bins {bins:>3} rows {:>6}: float {float_ms:.3} ms  quantized {quantized_ms:.3} ms  ({:.2}x)",
                node.row_indices.len(),
                float_ms / quantized_ms
            );
        }
    }
}
