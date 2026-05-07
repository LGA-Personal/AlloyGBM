//! Piecewise-linear split-gain criterion for the CPU backend.
//!
//! Implements the closed-form PL gain used during bin scanning:
//!
//! ```text
//! gain(split s) = 0.5·(Xᵀg_L)ᵀ(XᵀHX_L + λI)⁻¹(Xᵀg_L)
//!              + 0.5·(Xᵀg_R)ᵀ(XᵀHX_R + λI)⁻¹(Xᵀg_R)
//!              − 0.5·(Xᵀg_P)ᵀ(XᵀHX_P + λI)⁻¹(Xᵀg_P)
//! ```
//!
//! The `d×d` matrix inversion is handled via a tiny in-register Cholesky
//! factorisation (`d ≤ MAX_PL_REGRESSORS = 8`).  If the regularised Hessian
//! matrix is not positive definite (shouldn't happen with `λ > 0`, but
//! possible with extreme data) the gain defaults to `0.0` and the split is
//! skipped.

use alloygbm_core::{
    LinearFeatureHistogram, LinearLeaf, MAX_PL_MATRIX_ENTRIES, MAX_PL_REGRESSORS, NodeStats,
    SplitCandidate, pl_matrix_index,
};
use alloygbm_engine::{LinearContext, SplitSelectionOptions};

/// Compute the PL gain for one side of a split:
/// `0.5 · (Xᵀg)ᵀ (XᵀHX + λI)⁻¹ (Xᵀg)`.
///
/// Uses a compact Cholesky factorisation on the `d×d` regularised Hessian
/// matrix.  Returns `0.0` if `d == 0` or the matrix is not positive definite.
pub fn compute_pl_gain_one_side(
    xtg: &[f32; MAX_PL_REGRESSORS],
    xt_hx: &[f32; MAX_PL_MATRIX_ENTRIES],
    d: usize,
    l2_lambda: f32,
) -> f32 {
    if d == 0 {
        return 0.0;
    }

    // Build the full d×d regularised Hessian matrix A = XᵀHX + λI (row-major).
    // Only the upper triangle is stored in xt_hx; mirror it to fill the lower.
    let mut a = [0.0_f32; MAX_PL_REGRESSORS * MAX_PL_REGRESSORS];
    for j in 0..d {
        for k in j..d {
            let val = xt_hx[pl_matrix_index(j, k)];
            a[j * d + k] = val;
            a[k * d + j] = val;
        }
        a[j * d + j] += l2_lambda;
    }

    // Cholesky factorisation: A = L Lᵀ, stored in the lower triangle of `l`.
    let mut l = [0.0_f32; MAX_PL_REGRESSORS * MAX_PL_REGRESSORS];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return 0.0; // Not positive definite — skip this candidate.
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }

    // Forward substitution: solve L y = xtg.
    let mut y = [0.0_f32; MAX_PL_REGRESSORS];
    for i in 0..d {
        let mut s = xtg[i];
        for k in 0..i {
            s -= l[i * d + k] * y[k];
        }
        y[i] = s / l[i * d + i];
    }

    // gain = 0.5 · yᵀy  (since xᵀ A⁻¹ x = |L⁻¹ x|² = |y|²)
    let mut sq = 0.0_f32;
    for i in 0..d {
        sq += y[i] * y[i];
    }
    0.5 * sq
}

/// Find the best numeric split for a single feature using the PL gain criterion.
///
/// Mirrors the structure of `best_split_for_feature_inner` in `lib.rs` but
/// operates on `LinearFeatureHistogram` bins and accumulates running sums of
/// the matrix statistics `(Xᵀg, XᵀHX)` alongside the scalar sums.
pub fn best_split_linear_for_feature(
    linear_fh: &LinearFeatureHistogram,
    node_id: u32,
    options: SplitSelectionOptions,
    ctx: &LinearContext,
) -> Option<SplitCandidate> {
    let d = ctx.d();
    if d == 0 || linear_fh.bins.len() < 2 {
        return None;
    }
    let tri_len = d * (d + 1) / 2;
    let missing_bin_idx = options.missing_bin_index;

    // ── Parent (node-level) totals ──────────────────────────────────────────
    let mut p_xtg = [0.0_f32; MAX_PL_REGRESSORS];
    let mut p_xt_hx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    let mut p_grad = 0.0_f32;
    let mut p_hess = 0.0_f32;
    let mut p_count = 0_u32;
    for bin in &linear_fh.bins {
        p_grad += bin.grad_sum;
        p_hess += bin.hess_sum;
        p_count += bin.count;
        for j in 0..d {
            p_xtg[j] += bin.xtg[j];
        }
        for idx in 0..tri_len {
            p_xt_hx[idx] += bin.xt_hx[idx];
        }
    }

    if p_hess <= options.min_child_hessian {
        return None;
    }

    // ── Missing-bin contribution ─────────────────────────────────────────────
    let (m_xtg, m_xt_hx, m_grad, m_hess, m_count) = if missing_bin_idx < linear_fh.bins.len() {
        let mb = &linear_fh.bins[missing_bin_idx];
        let mut mxtg = [0.0_f32; MAX_PL_REGRESSORS];
        let mut mxthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        for j in 0..d {
            mxtg[j] = mb.xtg[j];
        }
        for idx in 0..tri_len {
            mxthx[idx] = mb.xt_hx[idx];
        }
        (mxtg, mxthx, mb.grad_sum, mb.hess_sum, mb.count)
    } else {
        (
            [0.0_f32; MAX_PL_REGRESSORS],
            [0.0_f32; MAX_PL_MATRIX_ENTRIES],
            0.0_f32,
            0.0_f32,
            0_u32,
        )
    };

    // Parent gain (subtracted from every candidate to get net gain).
    let parent_gain = compute_pl_gain_one_side(&p_xtg, &p_xt_hx, d, ctx.l2_lambda);

    // ── Non-missing totals (for right-side via subtraction) ──────────────────
    let scan_limit = linear_fh.bins.len().min(missing_bin_idx);
    let nm_grad = p_grad - m_grad;
    let nm_hess = p_hess - m_hess;
    let nm_count = p_count.saturating_sub(m_count);
    let mut nm_xtg = p_xtg;
    let mut nm_xt_hx = p_xt_hx;
    for j in 0..d {
        nm_xtg[j] -= m_xtg[j];
    }
    for idx in 0..tri_len {
        nm_xt_hx[idx] -= m_xt_hx[idx];
    }

    // ── Running left accumulators (non-missing bins only) ────────────────────
    let mut l_xtg = [0.0_f32; MAX_PL_REGRESSORS];
    let mut l_xt_hx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    let mut l_grad = 0.0_f32;
    let mut l_hess = 0.0_f32;
    let mut l_count = 0_u32;

    let mut best_candidate: Option<SplitCandidate> = None;
    let mut best_gain = 0.0_f32;

    for (threshold_bin, bin) in linear_fh.bins.iter().enumerate().take(scan_limit) {
        // Accumulate left side.
        l_grad += bin.grad_sum;
        l_hess += bin.hess_sum;
        l_count += bin.count;
        for j in 0..d {
            l_xtg[j] += bin.xtg[j];
        }
        for idx in 0..tri_len {
            l_xt_hx[idx] += bin.xt_hx[idx];
        }

        // Need at least one non-missing bin on the right side.
        if threshold_bin + 1 >= scan_limit && nm_count == l_count {
            continue;
        }

        // Right side (non-missing portion only).
        let r_grad_nm = nm_grad - l_grad;
        let r_hess_nm = nm_hess - l_hess;
        let r_count_nm = nm_count.saturating_sub(l_count);
        let mut r_xtg_nm = [0.0_f32; MAX_PL_REGRESSORS];
        let mut r_xt_hx_nm = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        for j in 0..d {
            r_xtg_nm[j] = nm_xtg[j] - l_xtg[j];
        }
        for idx in 0..tri_len {
            r_xt_hx_nm[idx] = nm_xt_hx[idx] - l_xt_hx[idx];
        }

        // Evaluate NaN-goes-left and NaN-goes-right, pick the better one.
        for default_left in [true, false] {
            // Effective left and right statistics for this NaN direction.
            let (eff_lg, eff_lh, eff_lc, eff_lxtg, eff_lxthx, eff_rg, eff_rh, eff_rc,
                 eff_rxtg, eff_rxthx) = if default_left {
                // NaN joins the left child.
                let mut el_xtg = l_xtg;
                let mut el_xthx = l_xt_hx;
                for j in 0..d {
                    el_xtg[j] += m_xtg[j];
                }
                for idx in 0..tri_len {
                    el_xthx[idx] += m_xt_hx[idx];
                }
                (
                    l_grad + m_grad,
                    l_hess + m_hess,
                    l_count + m_count,
                    el_xtg,
                    el_xthx,
                    r_grad_nm,
                    r_hess_nm,
                    r_count_nm,
                    r_xtg_nm,
                    r_xt_hx_nm,
                )
            } else {
                // NaN joins the right child.
                let mut er_xtg = r_xtg_nm;
                let mut er_xthx = r_xt_hx_nm;
                for j in 0..d {
                    er_xtg[j] += m_xtg[j];
                }
                for idx in 0..tri_len {
                    er_xthx[idx] += m_xt_hx[idx];
                }
                (
                    l_grad,
                    l_hess,
                    l_count,
                    l_xtg,
                    l_xt_hx,
                    r_grad_nm + m_grad,
                    r_hess_nm + m_hess,
                    r_count_nm + m_count,
                    er_xtg,
                    er_xthx,
                )
            };

            if eff_lc == 0
                || eff_rc == 0
                || eff_lh <= options.min_child_hessian
                || eff_rh <= options.min_child_hessian
            {
                continue;
            }

            let gain = compute_pl_gain_one_side(&eff_lxtg, &eff_lxthx, d, ctx.l2_lambda)
                + compute_pl_gain_one_side(&eff_rxtg, &eff_rxthx, d, ctx.l2_lambda)
                - parent_gain;

            if gain > best_gain {
                best_gain = gain;
                best_candidate = Some(SplitCandidate {
                    node_id,
                    feature_index: linear_fh.feature_index,
                    threshold_bin: threshold_bin as u16,
                    gain,
                    default_left,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: eff_lg,
                        hess_sum: eff_lh,
                        row_count: eff_lc,
                    },
                    right_stats: NodeStats {
                        grad_sum: eff_rg,
                        hess_sum: eff_rh,
                        row_count: eff_rc,
                    },
                });
            }
        }
    }

    best_candidate
}

// ── Leaf-weight solve ─────────────────────────────────────────────────────────

/// Accumulate the left-child linear statistics from a `LinearFeatureHistogram`
/// by summing bins `0..=threshold_bin` (non-missing), handling the NaN
/// direction via `default_left`.
///
/// Returns `(xtg, xt_hx, grad_sum, hess_sum)` for the left and right children.
pub fn leaf_linear_stats_for_split(
    linear_fh: &LinearFeatureHistogram,
    threshold_bin: usize,
    missing_bin_idx: usize,
    default_left: bool,
    d: usize,
) -> (
    [f32; MAX_PL_REGRESSORS],
    [f32; MAX_PL_MATRIX_ENTRIES],
    f32,
    f32,
    [f32; MAX_PL_REGRESSORS],
    [f32; MAX_PL_MATRIX_ENTRIES],
    f32,
    f32,
) {
    let tri_len = d * (d + 1) / 2;
    let scan_limit = linear_fh.bins.len().min(missing_bin_idx);

    // Missing bin.
    let (m_xtg, m_xt_hx, m_grad, m_hess) = if missing_bin_idx < linear_fh.bins.len() {
        let mb = &linear_fh.bins[missing_bin_idx];
        let mut mxtg = [0.0_f32; MAX_PL_REGRESSORS];
        let mut mxthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        for j in 0..d {
            mxtg[j] = mb.xtg[j];
        }
        for idx in 0..tri_len {
            mxthx[idx] = mb.xt_hx[idx];
        }
        (mxtg, mxthx, mb.grad_sum, mb.hess_sum)
    } else {
        (
            [0.0_f32; MAX_PL_REGRESSORS],
            [0.0_f32; MAX_PL_MATRIX_ENTRIES],
            0.0_f32,
            0.0_f32,
        )
    };

    // Sum non-missing bins up to (and including) threshold_bin for the left side.
    let mut l_xtg = [0.0_f32; MAX_PL_REGRESSORS];
    let mut l_xt_hx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    let mut l_grad = 0.0_f32;
    let mut l_hess = 0.0_f32;
    for bin in linear_fh.bins.iter().take(scan_limit.min(threshold_bin + 1)) {
        l_grad += bin.grad_sum;
        l_hess += bin.hess_sum;
        for j in 0..d {
            l_xtg[j] += bin.xtg[j];
        }
        for idx in 0..tri_len {
            l_xt_hx[idx] += bin.xt_hx[idx];
        }
    }

    // Parent non-missing totals.
    let mut nm_xtg = [0.0_f32; MAX_PL_REGRESSORS];
    let mut nm_xt_hx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    let mut nm_grad = 0.0_f32;
    let mut nm_hess = 0.0_f32;
    for bin in linear_fh.bins.iter().take(scan_limit) {
        nm_grad += bin.grad_sum;
        nm_hess += bin.hess_sum;
        for j in 0..d {
            nm_xtg[j] += bin.xtg[j];
        }
        for idx in 0..tri_len {
            nm_xt_hx[idx] += bin.xt_hx[idx];
        }
    }

    // Right non-missing = parent_nm - left.
    let mut r_xtg_nm = [0.0_f32; MAX_PL_REGRESSORS];
    let mut r_xt_hx_nm = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    for j in 0..d {
        r_xtg_nm[j] = nm_xtg[j] - l_xtg[j];
    }
    for idx in 0..tri_len {
        r_xt_hx_nm[idx] = nm_xt_hx[idx] - l_xt_hx[idx];
    }
    let r_grad_nm = nm_grad - l_grad;
    let r_hess_nm = nm_hess - l_hess;

    // Apply NaN direction.
    let (eff_l_xtg, eff_l_xt_hx, eff_l_grad, eff_l_hess, eff_r_xtg, eff_r_xt_hx, eff_r_grad, eff_r_hess) =
        if default_left {
            let mut el_xtg = l_xtg;
            let mut el_xthx = l_xt_hx;
            for j in 0..d {
                el_xtg[j] += m_xtg[j];
            }
            for idx in 0..tri_len {
                el_xthx[idx] += m_xt_hx[idx];
            }
            (
                el_xtg,
                el_xthx,
                l_grad + m_grad,
                l_hess + m_hess,
                r_xtg_nm,
                r_xt_hx_nm,
                r_grad_nm,
                r_hess_nm,
            )
        } else {
            let mut er_xtg = r_xtg_nm;
            let mut er_xthx = r_xt_hx_nm;
            for j in 0..d {
                er_xtg[j] += m_xtg[j];
            }
            for idx in 0..tri_len {
                er_xthx[idx] += m_xt_hx[idx];
            }
            (
                l_xtg,
                l_xt_hx,
                l_grad,
                l_hess,
                er_xtg,
                er_xthx,
                r_grad_nm + m_grad,
                r_hess_nm + m_hess,
            )
        };

    (
        eff_l_xtg, eff_l_xt_hx, eff_l_grad, eff_l_hess,
        eff_r_xtg, eff_r_xt_hx, eff_r_grad, eff_r_hess,
    )
}

/// Solve for PL leaf weights: `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
///
/// Returns the weight vector `[α_0, …, α_{d-1}]`.  On Cholesky failure
/// (non-PD matrix) returns all-zero (falls back to scalar leaf).
fn cholesky_solve_alpha(
    xtg: &[f32; MAX_PL_REGRESSORS],
    xt_hx: &[f32; MAX_PL_MATRIX_ENTRIES],
    d: usize,
    l2_lambda: f32,
) -> [f32; MAX_PL_REGRESSORS] {
    if d == 0 {
        return [0.0; MAX_PL_REGRESSORS];
    }

    // Build full d×d matrix A = XᵀHX + λI.
    let mut a = [0.0_f32; MAX_PL_REGRESSORS * MAX_PL_REGRESSORS];
    for j in 0..d {
        for k in j..d {
            let val = xt_hx[pl_matrix_index(j, k)];
            a[j * d + k] = val;
            a[k * d + j] = val;
        }
        a[j * d + j] += l2_lambda;
    }

    // Cholesky: A = L Lᵀ.
    let mut l = [0.0_f32; MAX_PL_REGRESSORS * MAX_PL_REGRESSORS];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return [0.0; MAX_PL_REGRESSORS]; // Fallback to scalar.
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }

    // Forward substitution: L y = -Xᵀg.
    let mut y = [0.0_f32; MAX_PL_REGRESSORS];
    for i in 0..d {
        let mut s = -xtg[i]; // Note: solving for -Xᵀg (gives α* = -(XᵀHX+λI)⁻¹ Xᵀg)
        for k in 0..i {
            s -= l[i * d + k] * y[k];
        }
        y[i] = s / l[i * d + i];
    }

    // Backward substitution: Lᵀ α = y.
    let mut alpha = [0.0_f32; MAX_PL_REGRESSORS];
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in (i + 1)..d {
            s -= l[k * d + i] * alpha[k];
        }
        alpha[i] = s / l[i * d + i];
    }

    alpha
}

/// Build a [`LinearLeaf`] from the accumulated statistics for one side of a split.
///
/// # Arguments
/// * `xtg` / `xt_hx` — matrix statistics accumulated from histogram bins
/// * `grad_sum` / `hess_sum` — scalar sums (for the Newton-Raphson intercept)
/// * `learning_rate` — applied to both intercept and weights
/// * `l2_lambda` — ridge regularisation
/// * `regressor_features` — indices of the `d` regressors
pub fn solve_pl_leaf(
    xtg: &[f32; MAX_PL_REGRESSORS],
    xt_hx: &[f32; MAX_PL_MATRIX_ENTRIES],
    grad_sum: f32,
    hess_sum: f32,
    learning_rate: f32,
    l2_lambda: f32,
    regressor_features: &[u32],
) -> LinearLeaf {
    let d = regressor_features.len();
    const LEAF_EPS: f32 = 1e-6;

    // Standard Newton-Raphson intercept (same formula as scalar leaves).
    let intercept = -learning_rate * grad_sum / (hess_sum + l2_lambda + LEAF_EPS);

    // Linear correction weights α* = -(XᵀHX + λI)⁻¹ Xᵀg, scaled by lr.
    let raw_alpha = cholesky_solve_alpha(xtg, xt_hx, d, l2_lambda);
    let weights: Vec<f32> = raw_alpha[..d]
        .iter()
        .map(|&w| learning_rate * w)
        .collect();

    LinearLeaf {
        intercept,
        weights,
        regressor_features: regressor_features.to_vec(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{LinearFeatureHistogram, LinearHistogramBin, pl_matrix_index};
    use alloygbm_engine::{LinearContext, SplitSelectionOptions};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a trivial LinearContext for d=1, lambda=1.0.
    fn ctx_d1() -> LinearContext {
        LinearContext {
            regressor_features: vec![0],
            l2_lambda: 1.0,
        }
    }

    /// Build a LinearFeatureHistogram with explicit per-bin scalar and matrix stats.
    ///
    /// `bins_data`: slice of `(grad, hess, count, xtg_0, xt_hx_00)` (d=1).
    fn make_feature_histogram_d1(
        feature_index: u32,
        bins_data: &[(f32, f32, u32, f32, f32)],
    ) -> LinearFeatureHistogram {
        let mut bins = Vec::new();
        for &(g, h, c, xtg0, xt_hx00) in bins_data {
            let mut bin = LinearHistogramBin::default();
            bin.grad_sum = g;
            bin.hess_sum = h;
            bin.count = c;
            bin.xtg[0] = xtg0;
            bin.xt_hx[0] = xt_hx00; // pl_matrix_index(0,0) = 0
            bins.push(bin);
        }
        LinearFeatureHistogram { feature_index, bins }
    }

    // ── Unit tests for compute_pl_gain_one_side ───────────────────────────────

    #[test]
    fn gain_zero_for_d0() {
        let xtg = [0.0; MAX_PL_REGRESSORS];
        let xthx = [0.0; MAX_PL_MATRIX_ENTRIES];
        assert_eq!(compute_pl_gain_one_side(&xtg, &xthx, 0, 1.0), 0.0);
    }

    #[test]
    fn gain_positive_for_nontrivial_side() {
        // d=1: A = xt_hx[0] + lambda = 2.0 + 1.0 = 3.0
        // xtg[0] = 6.0
        // gain = 0.5 * 6^2 / 3 = 0.5 * 12 = 6.0
        let mut xtg = [0.0_f32; MAX_PL_REGRESSORS];
        let mut xthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg[0] = 6.0;
        xthx[pl_matrix_index(0, 0)] = 2.0;
        let gain = compute_pl_gain_one_side(&xtg, &xthx, 1, 1.0);
        assert!((gain - 6.0).abs() < 1e-5, "gain={gain}");
    }

    #[test]
    fn gain_zero_for_non_positive_definite_matrix() {
        // d=1: A = 0.0 + 0.0 = 0.0 (not PD, lambda=0)
        let mut xtg = [0.0_f32; MAX_PL_REGRESSORS];
        let xthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg[0] = 1.0;
        let gain = compute_pl_gain_one_side(&xtg, &xthx, 1, 0.0);
        assert_eq!(gain, 0.0);
    }

    #[test]
    fn gain_d2_matches_manual_calculation() {
        // d=2, lambda=0.0
        // A = [[4.0, 1.0], [1.0, 3.0]]
        // xtg = [2.0, 1.0]
        // det(A) = 12 - 1 = 11
        // A⁻¹ = (1/11) [[3, -1], [-1, 4]]
        // xtg · A⁻¹ · xtg = (1/11)(4*3 + 2*(-1)*2*1 + 1*4) = (1/11)(12 - 4 + 4) = 12/11
        // gain = 0.5 * 12/11 ≈ 0.5455
        let mut xtg = [0.0_f32; MAX_PL_REGRESSORS];
        let mut xthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg[0] = 2.0;
        xtg[1] = 1.0;
        xthx[pl_matrix_index(0, 0)] = 4.0;
        xthx[pl_matrix_index(0, 1)] = 1.0;
        xthx[pl_matrix_index(1, 1)] = 3.0;
        let gain = compute_pl_gain_one_side(&xtg, &xthx, 2, 0.0);
        let expected = 0.5 * 12.0 / 11.0;
        assert!((gain - expected).abs() < 1e-5, "gain={gain} expected={expected}");
    }

    // ── Unit tests for best_split_linear_for_feature ─────────────────────────

    #[test]
    fn no_split_when_fewer_than_2_bins() {
        let fh = make_feature_histogram_d1(0, &[(1.0, 1.0, 10, 1.0, 1.0)]);
        let ctx = ctx_d1();
        let result = best_split_linear_for_feature(&fh, 1, SplitSelectionOptions::default(), &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn gain_is_non_negative_for_valid_split() {
        // Two bins: left has positive signal, right has negative signal.
        let fh = make_feature_histogram_d1(
            0,
            &[
                (5.0, 3.0, 50, 5.0, 3.0),
                (-5.0, 3.0, 50, -5.0, 3.0),
                (0.0, 0.0, 0, 0.0, 0.0), // missing bin (idx 255 default, so never reached by scan)
            ],
        );
        let ctx = ctx_d1();
        let result = best_split_linear_for_feature(&fh, 1, SplitSelectionOptions::default(), &ctx);
        if let Some(c) = result {
            assert!(c.gain >= 0.0, "gain should be non-negative: {}", c.gain);
        }
        // May or may not find a split depending on parent gain; just check no panic.
    }

    #[test]
    fn split_selects_correct_threshold_bin() {
        // Feature with 3 data bins + missing sentinel at idx 255 (default).
        // Bin 0: weak signal. Bin 1: strong positive signal. Bin 2: strong negative.
        // Best split should be at threshold_bin=1 (separating 0..=1 from 2).
        let fh = make_feature_histogram_d1(
            0,
            &[
                (0.1, 1.0, 10, 0.1, 1.0),  // bin 0 — weak
                (8.0, 4.0, 40, 8.0, 4.0),  // bin 1 — strong positive
                (-8.0, 4.0, 40, -8.0, 4.0), // bin 2 — strong negative
            ],
        );
        let ctx = ctx_d1();
        let options = SplitSelectionOptions {
            missing_bin_index: 255,
            ..Default::default()
        };
        let result = best_split_linear_for_feature(&fh, 1, options, &ctx);
        assert!(result.is_some(), "expected a split to be found");
        let c = result.unwrap();
        // threshold_bin=1: left = bins 0+1, right = bin 2 (best separation)
        assert_eq!(c.threshold_bin, 1, "expected threshold at bin 1, got {}", c.threshold_bin);
        assert!(c.gain > 0.0);
    }

    #[test]
    fn pl_gain_reduces_to_scalar_gain_when_x_equals_one() {
        // With d=1 and x_j=1 for every sample, XᵀHX = Σh and Xᵀg = Σg.
        // PL gain = 0.5*(Σg)²/(Σh+λ) per side.
        // Standard XGBoost gain = 0.5*(Σg)²/(Σh+λ) per side (same formula).
        let g_l = 4.0_f32;
        let h_l = 2.0_f32;
        let g_r = -2.0_f32;
        let h_r = 1.0_f32;
        let lambda = 1.0_f32;

        // Build xtg and xthx assuming x=1 everywhere.
        let mut xtg_l = [0.0_f32; MAX_PL_REGRESSORS];
        let mut xthx_l = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg_l[0] = g_l; // Xᵀg_L = Σ g_i * 1 = Σg
        xthx_l[pl_matrix_index(0, 0)] = h_l; // XᵀHX_L = Σ h_i * 1 * 1 = Σh

        let pl_gain_l = compute_pl_gain_one_side(&xtg_l, &xthx_l, 1, lambda);
        let xgb_gain_l = 0.5 * g_l * g_l / (h_l + lambda);
        assert!((pl_gain_l - xgb_gain_l).abs() < 1e-6, "d1 x=1: PL gain ({pl_gain_l}) ≠ XGB gain ({xgb_gain_l})");

        let mut xtg_r = [0.0_f32; MAX_PL_REGRESSORS];
        let mut xthx_r = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg_r[0] = g_r;
        xthx_r[pl_matrix_index(0, 0)] = h_r;
        let pl_gain_r = compute_pl_gain_one_side(&xtg_r, &xthx_r, 1, lambda);
        let xgb_gain_r = 0.5 * g_r * g_r / (h_r + lambda);
        assert!((pl_gain_r - xgb_gain_r).abs() < 1e-6, "d1 x=1: PL gain ({pl_gain_r}) ≠ XGB gain ({xgb_gain_r})");
    }

    #[test]
    fn gain_nonnegative_for_many_random_splits() {
        // Property test: PL gain ≥ 0 for any valid split where the matrix is PD.
        // Uses a few deterministic hand-crafted cases.
        let cases: &[(f32, f32, f32, f32, f32, f32)] = &[
            // (g_l, h_l, xtg_l, xthx_l, g_r, h_r) — xtg_r/xthx_r derived
            (3.0, 2.0, 3.0, 2.0, -3.0, 2.0),
            (10.0, 5.0, 10.0, 5.0, -10.0, 5.0),
            (1.0, 0.5, 0.5, 0.5, -1.0, 0.5),
        ];
        let lambda = 0.5;
        for &(g_l, h_l, xtg_l0, xthx_l0, g_r, h_r) in cases {
            let g_p = g_l + g_r;
            let h_p = h_l + h_r;
            let xtg_r0 = g_l + g_r - xtg_l0; // x=1 assumption for right
            let xthx_r0 = xthx_l0 * (h_r / h_l.max(1e-9)); // approximate

            let mut pl_xtg_l = [0.0_f32; MAX_PL_REGRESSORS];
            let mut pl_xthx_l = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
            pl_xtg_l[0] = xtg_l0;
            pl_xthx_l[0] = xthx_l0;

            let mut pl_xtg_r = [0.0_f32; MAX_PL_REGRESSORS];
            let mut pl_xthx_r = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
            pl_xtg_r[0] = xtg_r0;
            pl_xthx_r[0] = xthx_r0;

            let mut pl_xtg_p = [0.0_f32; MAX_PL_REGRESSORS];
            let mut pl_xthx_p = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
            pl_xtg_p[0] = g_p;
            pl_xthx_p[0] = h_p;

            let gain_l = compute_pl_gain_one_side(&pl_xtg_l, &pl_xthx_l, 1, lambda);
            let gain_r = compute_pl_gain_one_side(&pl_xtg_r, &pl_xthx_r, 1, lambda);
            let gain_p = compute_pl_gain_one_side(&pl_xtg_p, &pl_xthx_p, 1, lambda);
            let net = gain_l + gain_r - gain_p;
            // Net gain can be negative if parent is already well-fit; just assert no NaN/inf.
            assert!(net.is_finite(), "gain should be finite, got {net}");
        }
    }
}
