//! SIMD-accelerated kernels for histogram operations and EMA stats.
//!
//! Built on the `wide` crate, which compiles to AVX2 (x86_64) and NEON (arm64)
//! intrinsics behind a safe API. On hardware without SIMD support, falls back
//! to scalar code automatically.

pub use wide::f32x8;

/// SIMD L1 soft-threshold across a lane vector: `sign(g) * max(|g| - alpha, 0)`.
///
/// Mirrors the scalar `l1_threshold_gradient` helper used during split selection.
/// When `alpha <= 0`, returns the input unchanged.
#[inline]
pub fn l1_threshold_f32x8(g: f32x8, alpha: f32) -> f32x8 {
    if alpha <= 0.0 {
        return g;
    }
    let alpha_v = f32x8::splat(alpha);
    let abs_g = g.abs();
    let thresholded = (abs_g - alpha_v).max(f32x8::ZERO);
    // Re-attach sign of the original input. `copysign(magnitude, sign_source)`
    // returns `magnitude` with the sign bit of `sign_source`.
    thresholded.copysign(g)
}

/// Sum of an `f32` slice, vectorized 8-wide.
pub fn sum_f32(values: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        let v = f32x8::from(<[f32; 8]>::try_from(chunk).unwrap());
        acc += v;
    }
    let mut total = acc.reduce_add();
    for &x in chunks.remainder() {
        total += x;
    }
    total
}

/// Sum-of-squares of an `f32` slice, vectorized 8-wide.
pub fn sum_squares_f32(values: &[f32]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut chunks = values.chunks_exact(8);
    for chunk in &mut chunks {
        let v = f32x8::from(<[f32; 8]>::try_from(chunk).unwrap());
        acc += v * v;
    }
    let mut total = acc.reduce_add();
    for &x in chunks.remainder() {
        total += x * x;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_f32_matches_scalar() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.017).sin()).collect();
        let scalar: f32 = v.iter().sum();
        let vec_sum = sum_f32(&v);
        assert!(
            (scalar - vec_sum).abs() < 1e-3,
            "scalar={scalar} simd={vec_sum}"
        );
    }

    #[test]
    fn l1_threshold_f32x8_matches_scalar() {
        fn scalar(g: f32, a: f32) -> f32 {
            if a <= 0.0 {
                g
            } else if g > a {
                g - a
            } else if g < -a {
                g + a
            } else {
                0.0
            }
        }
        let alphas = [0.0_f32, 0.05, 0.5];
        let inputs: [f32; 8] = [1.0, -1.0, 0.0, 0.04, -0.04, 5.0, -5.0, 0.5];
        for &alpha in &alphas {
            let v = f32x8::from(inputs);
            let out = l1_threshold_f32x8(v, alpha).to_array();
            for i in 0..8 {
                let expected = scalar(inputs[i], alpha);
                assert!(
                    (expected - out[i]).abs() < 1e-6,
                    "lane {i} alpha={alpha}: scalar={expected} simd={}",
                    out[i]
                );
            }
        }
    }

    #[test]
    fn sum_squares_f32_matches_scalar() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.017).sin()).collect();
        let scalar: f32 = v.iter().map(|x| x * x).sum();
        let vec_sumsq = sum_squares_f32(&v);
        assert!(
            (scalar - vec_sumsq).abs() < 1e-3,
            "scalar={scalar} simd={vec_sumsq}"
        );
    }
}
