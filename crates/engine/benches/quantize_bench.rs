use alloygbm_core::GradientPair;
use alloygbm_engine::QuantizedGradients;
use std::hint::black_box;
use std::time::Instant;
fn main() {
    for &n in &[60_000usize, 600_000] {
        for constant in [true, false] {
            let g: Vec<GradientPair> = (0..n)
                .map(|i| GradientPair {
                    grad: ((i * 7919) % 2000) as f32 / 1000.0 - 1.0,
                    hess: if constant {
                        1.0
                    } else {
                        0.25 + ((i * 31) % 100) as f32 / 400.0
                    },
                })
                .collect();
            let mut best = f64::MAX;
            let mut q = QuantizedGradients::quantize(&g, 4, 0).unwrap();
            for r in 0..50u64 {
                let t = Instant::now();
                q.requantize(&g, 4, r).unwrap();
                black_box(&q);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "n {n} constant {constant}: {:.3} ms ({:.2} ns/row)",
                best * 1e3,
                best * 1e9 / n as f64
            );
        }
    }
}
