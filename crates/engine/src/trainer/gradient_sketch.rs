//! Gradient sketches for vector-leaf multiclass split scoring (SketchBoost,
//! Iosipoi & Vakhrushev, NeurIPS 2022).
//!
//! A vector-leaf tree scores every split on all K class gradients, so
//! histogram building and split scanning cost grows with K. A sketch replaces
//! the K columns with `d < K` derived columns for split scoring only; leaves
//! still solve on all K classes (see `VectorLeafGradients`).

use alloygbm_core::{GradientPair, MulticlassSketch};

use crate::sampling::mixed_hash;

/// Split-scoring columns for one round, or `None` when the sketch is off or
/// at least as wide as the class count (then all K columns score splits).
///
/// Norms and sampling probabilities are measured over `rows`, the rows this
/// round's tree is grown on; the sketch itself covers every row so it lines
/// up with the leaf columns.
pub(crate) fn sketch_gradients(
    kind: MulticlassSketch,
    dim: usize,
    gradients: &[Vec<GradientPair>],
    rows: &[u32],
    seed: u64,
) -> Option<Vec<Vec<GradientPair>>> {
    let classes = gradients.len();
    if kind == MulticlassSketch::None || dim == 0 || dim >= classes {
        return None;
    }
    Some(match kind {
        MulticlassSketch::None => unreachable!("handled above"),
        MulticlassSketch::TopOutputs => top_outputs(gradients, rows, dim),
        MulticlassSketch::RandomSampling => random_sampling(gradients, rows, dim, seed),
        MulticlassSketch::RandomProjection => random_projection(gradients, dim, seed),
    })
}

/// Sum of squared gradients per class over `rows`.
fn squared_norms(gradients: &[Vec<GradientPair>], rows: &[u32]) -> Vec<f64> {
    gradients
        .iter()
        .map(|column| {
            rows.iter()
                .map(|&row| {
                    let grad = f64::from(column[row as usize].grad);
                    grad * grad
                })
                .sum()
        })
        .collect()
}

/// Per-row mean Hessian over classes: a projected or rescaled column has no
/// Hessian of its own, so every sketched column shares this one.
fn mean_hessians(gradients: &[Vec<GradientPair>]) -> Vec<f32> {
    let row_count = gradients[0].len();
    let mut sums = vec![0.0_f64; row_count];
    for column in gradients {
        for (sum, pair) in sums.iter_mut().zip(column) {
            *sum += f64::from(pair.hess);
        }
    }
    let classes = gradients.len() as f64;
    sums.into_iter().map(|sum| (sum / classes) as f32).collect()
}

/// The `dim` classes with the largest gradient norms, with their own
/// Hessians. Ties go to the lower class index.
fn top_outputs(
    gradients: &[Vec<GradientPair>],
    rows: &[u32],
    dim: usize,
) -> Vec<Vec<GradientPair>> {
    let norms = squared_norms(gradients, rows);
    let mut order: Vec<usize> = (0..gradients.len()).collect();
    order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]).then(a.cmp(&b)));
    order[..dim]
        .iter()
        .map(|&class| gradients[class].clone())
        .collect()
}

/// `dim` classes drawn with replacement, with probability `p_c` proportional
/// to the squared gradient norm, each scaled by `1 / sqrt(dim * p_c)` so the
/// sketched squared norms are unbiased for the full ones.
fn random_sampling(
    gradients: &[Vec<GradientPair>],
    rows: &[u32],
    dim: usize,
    seed: u64,
) -> Vec<Vec<GradientPair>> {
    let norms = squared_norms(gradients, rows);
    let total: f64 = norms.iter().sum();
    let classes = gradients.len();
    // All-zero gradients: sample uniformly (the columns are zero anyway).
    let probabilities: Vec<f64> = if total > 0.0 && total.is_finite() {
        norms.iter().map(|&norm| norm / total).collect()
    } else {
        vec![1.0 / classes as f64; classes]
    };
    let hessians = mean_hessians(gradients);
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = mixed_hash(state);
            let draw = unit_interval(state);
            let mut class = classes - 1;
            let mut cumulative = 0.0;
            for (c, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if draw < cumulative && p > 0.0 {
                    class = c;
                    break;
                }
            }
            let p = probabilities[class].max(f64::MIN_POSITIVE);
            let scale = (1.0 / (dim as f64 * p).sqrt()) as f32;
            gradients[class]
                .iter()
                .zip(&hessians)
                .map(|(pair, &hess)| GradientPair {
                    grad: pair.grad * scale,
                    hess,
                })
                .collect()
        })
        .collect()
}

/// The gradient matrix times a `K x dim` standard Gaussian matrix scaled by
/// `1 / sqrt(dim)`, so squared row norms are preserved in expectation.
fn random_projection(
    gradients: &[Vec<GradientPair>],
    dim: usize,
    seed: u64,
) -> Vec<Vec<GradientPair>> {
    let classes = gradients.len();
    let row_count = gradients[0].len();
    let scale = 1.0 / (dim as f64).sqrt();
    let mut state = seed;
    // projection[c * dim + j]
    let projection: Vec<f32> = (0..classes * dim)
        .map(|_| {
            state = mixed_hash(state);
            let u1 = unit_interval(state).max(f64::MIN_POSITIVE);
            state = mixed_hash(state);
            let u2 = unit_interval(state);
            let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            (normal * scale) as f32
        })
        .collect();
    let hessians = mean_hessians(gradients);
    let mut out: Vec<Vec<GradientPair>> = (0..dim)
        .map(|_| {
            hessians
                .iter()
                .map(|&hess| GradientPair { grad: 0.0, hess })
                .collect()
        })
        .collect();
    for (c, column) in gradients.iter().enumerate() {
        let weights = &projection[c * dim..(c + 1) * dim];
        for (j, &weight) in weights.iter().enumerate() {
            for (target, pair) in out[j].iter_mut().zip(column) {
                target.grad += pair.grad * weight;
            }
        }
    }
    debug_assert!(out.iter().all(|column| column.len() == row_count));
    out
}

/// Uniform in `[0, 1)` from the top 53 bits of a hash.
#[inline]
fn unit_interval(hash: u64) -> f64 {
    (hash >> 11) as f64 / (1_u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(classes: usize, rows: usize) -> Vec<Vec<GradientPair>> {
        (0..classes)
            .map(|c| {
                (0..rows)
                    .map(|r| GradientPair {
                        // Class c has gradients of magnitude about c + 1.
                        grad: (c as f32 + 1.0) * if (r + c) % 3 == 0 { -1.0 } else { 0.5 },
                        hess: 0.1 * (c as f32 + 1.0),
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn off_or_wide_sketches_score_on_all_classes() {
        let g = columns(4, 10);
        let rows: Vec<u32> = (0..10).collect();
        assert!(sketch_gradients(MulticlassSketch::None, 2, &g, &rows, 1).is_none());
        for kind in [
            MulticlassSketch::TopOutputs,
            MulticlassSketch::RandomSampling,
            MulticlassSketch::RandomProjection,
        ] {
            assert!(sketch_gradients(kind, 4, &g, &rows, 1).is_none());
            assert!(sketch_gradients(kind, 9, &g, &rows, 1).is_none());
            let sketch = sketch_gradients(kind, 2, &g, &rows, 1).expect("sketch");
            assert_eq!(sketch.len(), 2);
            assert!(sketch.iter().all(|column| column.len() == 10));
        }
    }

    #[test]
    fn top_outputs_keeps_the_largest_columns_with_their_hessians() {
        let g = columns(5, 12);
        let rows: Vec<u32> = (0..12).collect();
        let sketch = sketch_gradients(MulticlassSketch::TopOutputs, 2, &g, &rows, 7).unwrap();
        assert_eq!(sketch[0], g[4]);
        assert_eq!(sketch[1], g[3]);
    }

    #[test]
    fn random_sketches_use_the_mean_hessian_and_are_seeded() {
        let g = columns(6, 20);
        let rows: Vec<u32> = (0..20).collect();
        let mean_hess = 0.1 * (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6.0;
        for kind in [
            MulticlassSketch::RandomSampling,
            MulticlassSketch::RandomProjection,
        ] {
            let a = sketch_gradients(kind, 3, &g, &rows, 11).unwrap();
            let b = sketch_gradients(kind, 3, &g, &rows, 11).unwrap();
            let c = sketch_gradients(kind, 3, &g, &rows, 12).unwrap();
            assert_eq!(a, b);
            assert_ne!(a, c);
            for column in &a {
                for pair in column {
                    assert!((pair.hess - mean_hess).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn random_projection_preserves_squared_norm_on_average() {
        let g = columns(8, 30);
        let rows: Vec<u32> = (0..30).collect();
        let full: f64 = squared_norms(&g, &rows).iter().sum();
        let trials = 400;
        let mut mean = 0.0;
        for seed in 0..trials {
            let sketch =
                sketch_gradients(MulticlassSketch::RandomProjection, 4, &g, &rows, seed).unwrap();
            mean += squared_norms(&sketch, &rows).iter().sum::<f64>() / trials as f64;
        }
        assert!((mean / full - 1.0).abs() < 0.1, "ratio {}", mean / full);
    }

    #[test]
    fn random_sampling_preserves_squared_norm_exactly_per_draw() {
        // Each draw of class c contributes ||g_c||^2 / (dim * p_c), and with
        // p_c proportional to ||g_c||^2 that is ||G||^2 / dim for every c.
        let g = columns(5, 15);
        let rows: Vec<u32> = (0..15).collect();
        let full: f64 = squared_norms(&g, &rows).iter().sum();
        let sketch = sketch_gradients(MulticlassSketch::RandomSampling, 3, &g, &rows, 5).unwrap();
        let sketched: f64 = squared_norms(&sketch, &rows).iter().sum();
        assert!(
            (sketched / full - 1.0).abs() < 1e-4,
            "ratio {}",
            sketched / full
        );
    }
}
