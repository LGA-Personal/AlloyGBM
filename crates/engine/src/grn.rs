//! Gradient-regularized Newton (GRN) leaf damping.
//!
//! Mishchenko (2023) regularizes each Newton step with
//! `λ_k = sqrt(M · ‖∇f(x_k)‖)`, where `M` is the Lipschitz constant of the
//! loss's Hessian. A boosting round is one Newton step in function space, so
//! each round we compute `λ_k = grn_strength · sqrt(‖g_k‖)` with
//! `grn_strength = C·sqrt(M)` and `‖g_k‖ = sqrt(mean_i ‖g_i‖²)` over that
//! round's row gradients (which already carry sample weights), then add
//! `λ_k` to the round's leaf solve as `alloygbm_core::GrnForm` says. As the fit
//! converges the gradients shrink and GRN fades back to plain Newton.

use alloygbm_core::GradientPair;

/// `‖g‖ = sqrt(Σ_rows Σ_outputs g² / rows)`. For multiclass each row's
/// gradient is the K-vector across classes, per the paper's vector form.
pub(crate) fn grn_gradient_norm<'a>(
    buffers: impl IntoIterator<Item = &'a [GradientPair]>,
    row_count: usize,
) -> f64 {
    if row_count == 0 {
        return 0.0;
    }
    let sum_sq: f64 = buffers
        .into_iter()
        .flat_map(|buffer| buffer.iter())
        .map(|pair| f64::from(pair.grad) * f64::from(pair.grad))
        .sum();
    (sum_sq / row_count as f64).sqrt()
}

/// This round's GRN regularizer, or `None` when GRN is off.
pub(crate) fn grn_round_lambda<'a>(
    strength: f32,
    buffers: impl IntoIterator<Item = &'a [GradientPair]>,
    row_count: usize,
) -> Option<f32> {
    if strength <= 0.0 {
        return None;
    }
    let lambda = f64::from(strength) * grn_gradient_norm(buffers, row_count).sqrt();
    Some(lambda as f32)
}

/// Add `lambda` to every row's Hessian, scaled by the row's sample weight
/// (the per-row form). Call before row sampling so GOSS amplification sees
/// the regularized Hessians.
pub(crate) fn add_grn_to_row_hessians(
    gradients: &mut [GradientPair],
    sample_weights: Option<&[f32]>,
    lambda: f32,
) {
    match sample_weights {
        Some(weights) => {
            for (pair, &weight) in gradients.iter_mut().zip(weights) {
                pair.hess += lambda * weight;
            }
        }
        None => {
            for pair in gradients.iter_mut() {
                pair.hess += lambda;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(grad: f32, hess: f32) -> GradientPair {
        GradientPair { grad, hess }
    }

    #[test]
    fn norm_is_root_mean_square_over_rows_and_outputs() {
        let a = [pair(3.0, 1.0), pair(0.0, 1.0)];
        let b = [pair(4.0, 1.0), pair(0.0, 1.0)];
        // rows: (3,4) and (0,0) -> mean squared row norm = 25/2
        let norm = grn_gradient_norm([&a[..], &b[..]], 2);
        assert!((norm - (12.5_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn zero_strength_is_off() {
        let a = [pair(3.0, 1.0)];
        assert_eq!(grn_round_lambda(0.0, [&a[..]], 1), None);
        let lambda = grn_round_lambda(2.0, [&a[..]], 1).unwrap();
        assert!((lambda - 2.0 * 3.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn per_row_form_scales_by_weight() {
        let mut buffer = vec![pair(1.0, 1.0), pair(1.0, 2.0)];
        add_grn_to_row_hessians(&mut buffer, Some(&[1.0, 3.0]), 0.5);
        assert_eq!(buffer[0].hess, 1.5);
        assert_eq!(buffer[1].hess, 3.5);
    }
}
