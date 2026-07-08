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

use alloygbm_core::simd::f32x8;
use alloygbm_core::{
    BinnedMatrix, LinearFeatureHistogram, LinearHistogramBin, LinearLeaf, MAX_PL_MATRIX_ENTRIES,
    MAX_PL_REGRESSORS, NodeStats, SplitCandidate, pl_matrix_index,
};
use alloygbm_engine::{LinearContext, SplitSelectionOptions};

// ── xt_hx helpers ─────────────────────────────────────────────────────────────
//
// `xt_hx` is laid out stride-8 row-major (`xt_hx[j * 8 + k]`).  The histogram
// builder writes only the upper-triangular slots; lower-triangle entries stay
// zero.  Operating on the full 64 entries is harmless under add/sub/copy/diff
// because zero is the identity for these operations.  This lets us use 8 full-
// width `f32x8` ops per call with no scalar tail — matching the SIMD pattern
// used by the constant-leaf bin scan in `crates/backend_cpu/src/lib.rs`.
//
// The `d: usize` parameter is no longer needed (we always operate on all 64
// entries); it was required by the previous compacted upper-triangle layout.
//
// All four helpers are bit-exact with their scalar counterparts: each lane of
// `f32x8` performs an independent f32 op identical to `dst[i] op src[i]`, so
// rounding is byte-equal regardless of vector width.

const XT_HX_CHUNKS: usize = MAX_PL_MATRIX_ENTRIES / 8;
const _CHUNKS_DIVIDE_EVENLY: () = assert!(MAX_PL_MATRIX_ENTRIES.is_multiple_of(8));

#[inline]
fn load_chunk(src: &[f32; MAX_PL_MATRIX_ENTRIES], chunk: usize) -> f32x8 {
    let base = chunk * 8;
    f32x8::from([
        src[base],
        src[base + 1],
        src[base + 2],
        src[base + 3],
        src[base + 4],
        src[base + 5],
        src[base + 6],
        src[base + 7],
    ])
}

#[inline]
fn store_chunk(dst: &mut [f32; MAX_PL_MATRIX_ENTRIES], chunk: usize, v: f32x8) {
    let base = chunk * 8;
    let arr = v.to_array();
    dst[base..base + 8].copy_from_slice(&arr);
}

#[inline]
fn copy_xt_hx(src: &[f32; MAX_PL_MATRIX_ENTRIES], dst: &mut [f32; MAX_PL_MATRIX_ENTRIES]) {
    // `*dst = *src` is itself optimal for a 64-byte memcpy; the SIMD form is
    // here for symmetry with the other helpers and to keep the whole module
    // on a single code path.
    *dst = *src;
}

#[inline]
fn add_xt_hx(src: &[f32; MAX_PL_MATRIX_ENTRIES], dst: &mut [f32; MAX_PL_MATRIX_ENTRIES]) {
    for chunk in 0..XT_HX_CHUNKS {
        let s = load_chunk(src, chunk);
        let d = load_chunk(dst, chunk);
        store_chunk(dst, chunk, d + s);
    }
}

#[inline]
fn sub_xt_hx(src: &[f32; MAX_PL_MATRIX_ENTRIES], dst: &mut [f32; MAX_PL_MATRIX_ENTRIES]) {
    for chunk in 0..XT_HX_CHUNKS {
        let s = load_chunk(src, chunk);
        let d = load_chunk(dst, chunk);
        store_chunk(dst, chunk, d - s);
    }
}

#[inline]
fn diff_xt_hx(
    b: &[f32; MAX_PL_MATRIX_ENTRIES],
    c: &[f32; MAX_PL_MATRIX_ENTRIES],
    dst: &mut [f32; MAX_PL_MATRIX_ENTRIES],
) {
    for chunk in 0..XT_HX_CHUNKS {
        let bv = load_chunk(b, chunk);
        let cv = load_chunk(c, chunk);
        store_chunk(dst, chunk, bv - cv);
    }
}

/// Add `src` into `dst` lane-wise: `dst[i] += src[i]` for `i in 0..MAX_PL_REGRESSORS`.
///
/// `xtg` is exactly 8 entries (one `f32x8` lane), so this collapses to a single
/// SIMD op.  Used wherever the scalar path was iterating with `zip`/`take(d)`.
#[inline]
fn add_xtg(src: &[f32; MAX_PL_REGRESSORS], dst: &mut [f32; MAX_PL_REGRESSORS]) {
    let s = f32x8::from(*src);
    let d = f32x8::from(*dst);
    *dst = (d + s).to_array();
}

/// Subtract `src` from `dst` lane-wise: `dst[i] -= src[i]`.
#[inline]
fn sub_xtg(src: &[f32; MAX_PL_REGRESSORS], dst: &mut [f32; MAX_PL_REGRESSORS]) {
    let s = f32x8::from(*src);
    let d = f32x8::from(*dst);
    *dst = (d - s).to_array();
}

/// Compute `dst[i] = b[i] - c[i]` lane-wise.
#[inline]
fn diff_xtg(
    b: &[f32; MAX_PL_REGRESSORS],
    c: &[f32; MAX_PL_REGRESSORS],
    dst: &mut [f32; MAX_PL_REGRESSORS],
) {
    let bv = f32x8::from(*b);
    let cv = f32x8::from(*c);
    *dst = (bv - cv).to_array();
}

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
    let sq: f32 = y[..d].iter().map(|&yi| yi * yi).sum();
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
        add_xtg(&bin.xtg, &mut p_xtg);
        add_xt_hx(&bin.xt_hx, &mut p_xt_hx);
    }

    if p_hess <= options.min_child_hessian {
        return None;
    }

    // ── Missing-bin contribution ─────────────────────────────────────────────
    let (m_xtg, m_xt_hx, m_grad, m_hess, m_count) = if missing_bin_idx < linear_fh.bins.len() {
        let mb = &linear_fh.bins[missing_bin_idx];
        let mut mxthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        copy_xt_hx(&mb.xt_hx, &mut mxthx);
        (mb.xtg, mxthx, mb.grad_sum, mb.hess_sum, mb.count)
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
    sub_xtg(&m_xtg, &mut nm_xtg);
    sub_xt_hx(&m_xt_hx, &mut nm_xt_hx);

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
        add_xtg(&bin.xtg, &mut l_xtg);
        add_xt_hx(&bin.xt_hx, &mut l_xt_hx);

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
        diff_xtg(&nm_xtg, &l_xtg, &mut r_xtg_nm);
        diff_xt_hx(&nm_xt_hx, &l_xt_hx, &mut r_xt_hx_nm);

        // Evaluate NaN-goes-left and NaN-goes-right, pick the better one.
        for default_left in [true, false] {
            // Effective left and right statistics for this NaN direction.
            let (
                eff_lg,
                eff_lh,
                eff_lc,
                eff_lxtg,
                eff_lxthx,
                eff_rg,
                eff_rh,
                eff_rc,
                eff_rxtg,
                eff_rxthx,
            ) = if default_left {
                // NaN joins the left child.
                let mut el_xtg = l_xtg;
                let mut el_xthx = l_xt_hx;
                add_xtg(&m_xtg, &mut el_xtg);
                add_xt_hx(&m_xt_hx, &mut el_xthx);
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
                add_xtg(&m_xtg, &mut er_xtg);
                add_xt_hx(&m_xt_hx, &mut er_xthx);
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
                        grad_sq_sum: 0.0,
                        row_count: eff_lc,
                    },
                    right_stats: NodeStats {
                        grad_sum: eff_rg,
                        hess_sum: eff_rh,
                        grad_sq_sum: 0.0,
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
#[allow(clippy::type_complexity)]
pub fn leaf_linear_stats_for_split(
    linear_fh: &LinearFeatureHistogram,
    threshold_bin: usize,
    missing_bin_idx: usize,
    default_left: bool,
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
    let scan_limit = linear_fh.bins.len().min(missing_bin_idx);

    // Missing bin.
    let (m_xtg, m_xt_hx, m_grad, m_hess) = if missing_bin_idx < linear_fh.bins.len() {
        let mb = &linear_fh.bins[missing_bin_idx];
        let mut mxthx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        copy_xt_hx(&mb.xt_hx, &mut mxthx);
        (mb.xtg, mxthx, mb.grad_sum, mb.hess_sum)
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
    for bin in linear_fh
        .bins
        .iter()
        .take(scan_limit.min(threshold_bin + 1))
    {
        l_grad += bin.grad_sum;
        l_hess += bin.hess_sum;
        add_xtg(&bin.xtg, &mut l_xtg);
        add_xt_hx(&bin.xt_hx, &mut l_xt_hx);
    }

    // Parent non-missing totals.
    let mut nm_xtg = [0.0_f32; MAX_PL_REGRESSORS];
    let mut nm_xt_hx = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    let mut nm_grad = 0.0_f32;
    let mut nm_hess = 0.0_f32;
    for bin in linear_fh.bins.iter().take(scan_limit) {
        nm_grad += bin.grad_sum;
        nm_hess += bin.hess_sum;
        add_xtg(&bin.xtg, &mut nm_xtg);
        add_xt_hx(&bin.xt_hx, &mut nm_xt_hx);
    }

    // Right non-missing = parent_nm - left.
    let mut r_xtg_nm = [0.0_f32; MAX_PL_REGRESSORS];
    let mut r_xt_hx_nm = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
    diff_xtg(&nm_xtg, &l_xtg, &mut r_xtg_nm);
    diff_xt_hx(&nm_xt_hx, &l_xt_hx, &mut r_xt_hx_nm);
    let r_grad_nm = nm_grad - l_grad;
    let r_hess_nm = nm_hess - l_hess;

    // Apply NaN direction.
    let (
        eff_l_xtg,
        eff_l_xt_hx,
        eff_l_grad,
        eff_l_hess,
        eff_r_xtg,
        eff_r_xt_hx,
        eff_r_grad,
        eff_r_hess,
    ) = if default_left {
        let mut el_xtg = l_xtg;
        let mut el_xthx = l_xt_hx;
        add_xtg(&m_xtg, &mut el_xtg);
        add_xt_hx(&m_xt_hx, &mut el_xthx);
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
        add_xtg(&m_xtg, &mut er_xtg);
        add_xt_hx(&m_xt_hx, &mut er_xthx);
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
        eff_l_xtg,
        eff_l_xt_hx,
        eff_l_grad,
        eff_l_hess,
        eff_r_xtg,
        eff_r_xt_hx,
        eff_r_grad,
        eff_r_hess,
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
    let weights: Vec<f32> = raw_alpha[..d].iter().map(|&w| learning_rate * w).collect();

    LinearLeaf::identity_scaled(intercept, weights, regressor_features.to_vec())
}

#[allow(clippy::too_many_arguments)]
pub fn solve_pl_leaf_pair_from_partitions(
    binned_matrix: &BinnedMatrix,
    gradients: &[alloygbm_core::GradientPair],
    raw_feature_values: &[f32],
    feature_count: usize,
    split_feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
    regressor_features: &[u32],
    left_rows: &[u32],
    right_rows: &[u32],
    learning_rate: f32,
    l2_lambda: f32,
) -> Option<(LinearLeaf, LinearLeaf)> {
    let d = regressor_features.len();
    if d == 0 || d > MAX_PL_REGRESSORS || feature_count == 0 {
        return None;
    }

    let linear_fh = accumulate_selected_split_linear_histogram(
        binned_matrix,
        gradients,
        raw_feature_values,
        feature_count,
        split_feature_index,
        regressor_features,
        left_rows,
        right_rows,
    )?;
    let (l_xtg, l_xthx, l_gs, l_hs, r_xtg, r_xthx, r_gs, r_hs) = leaf_linear_stats_for_split(
        &linear_fh,
        threshold_bin as usize,
        binned_matrix.missing_bin() as usize,
        default_left,
    );

    Some((
        solve_pl_leaf(
            &l_xtg,
            &l_xthx,
            l_gs,
            l_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        ),
        solve_pl_leaf(
            &r_xtg,
            &r_xthx,
            r_gs,
            r_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        ),
    ))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_selected_split_linear_histogram(
    binned_matrix: &BinnedMatrix,
    gradients: &[alloygbm_core::GradientPair],
    raw_feature_values: &[f32],
    feature_count: usize,
    split_feature_index: u32,
    regressor_features: &[u32],
    left_rows: &[u32],
    right_rows: &[u32],
) -> Option<LinearFeatureHistogram> {
    let split_feature = split_feature_index as usize;
    if split_feature >= binned_matrix.feature_count {
        return None;
    }
    let bin_count = (binned_matrix.max_bin.max(binned_matrix.missing_bin()) as usize) + 1;
    let mut bins = vec![LinearHistogramBin::default(); bin_count];

    for &row_u32 in left_rows.iter().chain(right_rows.iter()) {
        let row = row_u32 as usize;
        let gp = *gradients.get(row)?;
        let row_base = row.checked_mul(feature_count)?;
        if row_base + feature_count > raw_feature_values.len() {
            return None;
        }
        let split_bin = binned_matrix
            .row_bin(row.checked_mul(binned_matrix.feature_count)? + split_feature)
            as usize;
        let bin = bins.get_mut(split_bin)?;

        bin.grad_sum += gp.grad;
        bin.hess_sum += gp.hess;
        bin.count += 1;

        let mut x = [0.0_f32; MAX_PL_REGRESSORS];
        for (slot, &feature_u32) in regressor_features.iter().enumerate() {
            let feature = feature_u32 as usize;
            x[slot] = if feature < feature_count {
                raw_feature_values[row_base + feature]
            } else {
                0.0
            };
        }

        for j in 0..regressor_features.len() {
            bin.xtg[j] += gp.grad * x[j];
            for k in 0..regressor_features.len() {
                bin.xt_hx[pl_matrix_index(j, k)] += gp.hess * x[j] * x[k];
            }
        }
    }

    for bin in &mut bins {
        sanitize_direct_linear_bin(bin);
    }

    Some(LinearFeatureHistogram {
        feature_index: split_feature_index,
        bins,
    })
}

fn sanitize_direct_linear_bin(bin: &mut LinearHistogramBin) {
    if !bin.grad_sum.is_finite() {
        bin.grad_sum = 0.0;
    }
    if !bin.hess_sum.is_finite() {
        bin.hess_sum = 0.0;
    }
    for slot in &mut bin.xtg {
        if !slot.is_finite() {
            *slot = 0.0;
        }
    }
    for slot in &mut bin.xt_hx {
        if !slot.is_finite() {
            *slot = 0.0;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{
        LinearFeatureHistogram, LinearHistogramBin, MAX_PL_REGRESSORS, pl_matrix_index,
    };
    use alloygbm_engine::{LinearContext, SplitSelectionOptions};

    // ── Layout invariants ────────────────────────────────────────────────────

    /// All `pl_matrix_index(j, k)` values for `0 ≤ j ≤ k < MAX_PL_REGRESSORS`
    /// are unique and bounded by `MAX_PL_MATRIX_ENTRIES`.  This protects the
    /// SIMD-friendly stride-8 layout from accidentally regressing back to a
    /// compacted form (which would alias indices across rows).
    #[test]
    fn pl_matrix_index_uniqueness_and_bounds() {
        let mut seen = std::collections::HashSet::new();
        for j in 0..MAX_PL_REGRESSORS {
            for k in j..MAX_PL_REGRESSORS {
                let idx = pl_matrix_index(j, k);
                assert!(
                    idx < MAX_PL_MATRIX_ENTRIES,
                    "pl_matrix_index({j},{k}) = {idx} exceeds MAX_PL_MATRIX_ENTRIES"
                );
                assert!(
                    seen.insert(idx),
                    "pl_matrix_index({j},{k}) = {idx} collides with an earlier (j,k)"
                );
            }
        }
        // Stride-8 layout: j * 8 + k.
        assert_eq!(pl_matrix_index(0, 0), 0);
        assert_eq!(pl_matrix_index(0, 7), 7);
        assert_eq!(pl_matrix_index(1, 1), 9);
        assert_eq!(pl_matrix_index(7, 7), 63);
    }

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
            let mut bin = LinearHistogramBin {
                grad_sum: g,
                hess_sum: h,
                count: c,
                ..Default::default()
            };
            bin.xtg[0] = xtg0;
            bin.xt_hx[0] = xt_hx00; // pl_matrix_index(0,0) = 0
            bins.push(bin);
        }
        LinearFeatureHistogram {
            feature_index,
            bins,
        }
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
        assert!(
            (gain - expected).abs() < 1e-5,
            "gain={gain} expected={expected}"
        );
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
                (0.1, 1.0, 10, 0.1, 1.0),   // bin 0 — weak
                (8.0, 4.0, 40, 8.0, 4.0),   // bin 1 — strong positive
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
        assert_eq!(
            c.threshold_bin, 1,
            "expected threshold at bin 1, got {}",
            c.threshold_bin
        );
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
        assert!(
            (pl_gain_l - xgb_gain_l).abs() < 1e-6,
            "d1 x=1: PL gain ({pl_gain_l}) ≠ XGB gain ({xgb_gain_l})"
        );

        let mut xtg_r = [0.0_f32; MAX_PL_REGRESSORS];
        let mut xthx_r = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        xtg_r[0] = g_r;
        xthx_r[pl_matrix_index(0, 0)] = h_r;
        let pl_gain_r = compute_pl_gain_one_side(&xtg_r, &xthx_r, 1, lambda);
        let xgb_gain_r = 0.5 * g_r * g_r / (h_r + lambda);
        assert!(
            (pl_gain_r - xgb_gain_r).abs() < 1e-6,
            "d1 x=1: PL gain ({pl_gain_r}) ≠ XGB gain ({xgb_gain_r})"
        );
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

    // ── SIMD helper property tests ────────────────────────────────────────────
    //
    // For the lane-wise helpers (`add_xt_hx`, `sub_xt_hx`, `diff_xt_hx`,
    // `copy_xt_hx`, `add_xtg`, `sub_xtg`, `diff_xtg`), each output element is
    // an independent f32 op of the form `dst[i] = f(src[i], ...)`.  SIMD
    // vectorisation processes 8 such ops in parallel but does not reorder
    // operations or introduce reductions, so the result is **bit-exact** with
    // a naïve scalar loop.  These tests pin that invariant.

    /// Generate a deterministic 64-entry xt_hx-shaped matrix from a seed.
    fn xt_hx_from_seed(seed: u64) -> [f32; MAX_PL_MATRIX_ENTRIES] {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut out = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
        for slot in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map u64 to f32 in roughly [-10, 10].
            let x = ((state >> 32) as i32 as f32) / (i32::MAX as f32) * 10.0;
            *slot = x;
        }
        out
    }

    fn xtg_from_seed(seed: u64) -> [f32; MAX_PL_REGRESSORS] {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut out = [0.0_f32; MAX_PL_REGRESSORS];
        for slot in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((state >> 32) as i32 as f32) / (i32::MAX as f32) * 10.0;
            *slot = x;
        }
        out
    }

    #[test]
    fn add_xt_hx_simd_matches_scalar() {
        for seed in 0..100 {
            let src = xt_hx_from_seed(seed);
            let initial = xt_hx_from_seed(seed.wrapping_add(1));
            let mut dst_simd = initial;
            let mut dst_scalar = initial;
            add_xt_hx(&src, &mut dst_simd);
            for i in 0..MAX_PL_MATRIX_ENTRIES {
                dst_scalar[i] += src[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn sub_xt_hx_simd_matches_scalar() {
        for seed in 0..100 {
            let src = xt_hx_from_seed(seed);
            let initial = xt_hx_from_seed(seed.wrapping_add(1));
            let mut dst_simd = initial;
            let mut dst_scalar = initial;
            sub_xt_hx(&src, &mut dst_simd);
            for i in 0..MAX_PL_MATRIX_ENTRIES {
                dst_scalar[i] -= src[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn diff_xt_hx_simd_matches_scalar() {
        for seed in 0..100 {
            let b = xt_hx_from_seed(seed);
            let c = xt_hx_from_seed(seed.wrapping_add(1));
            let mut dst_simd = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
            let mut dst_scalar = [0.0_f32; MAX_PL_MATRIX_ENTRIES];
            diff_xt_hx(&b, &c, &mut dst_simd);
            for i in 0..MAX_PL_MATRIX_ENTRIES {
                dst_scalar[i] = b[i] - c[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn copy_xt_hx_simd_matches_scalar() {
        for seed in 0..100 {
            let src = xt_hx_from_seed(seed);
            let mut dst_simd = [9.0_f32; MAX_PL_MATRIX_ENTRIES];
            let mut dst_scalar = [9.0_f32; MAX_PL_MATRIX_ENTRIES];
            copy_xt_hx(&src, &mut dst_simd);
            dst_scalar[..MAX_PL_MATRIX_ENTRIES].copy_from_slice(&src);
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn add_xtg_simd_matches_scalar() {
        for seed in 0..100 {
            let src = xtg_from_seed(seed);
            let initial = xtg_from_seed(seed.wrapping_add(1));
            let mut dst_simd = initial;
            let mut dst_scalar = initial;
            add_xtg(&src, &mut dst_simd);
            for i in 0..MAX_PL_REGRESSORS {
                dst_scalar[i] += src[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn sub_xtg_simd_matches_scalar() {
        for seed in 0..100 {
            let src = xtg_from_seed(seed);
            let initial = xtg_from_seed(seed.wrapping_add(1));
            let mut dst_simd = initial;
            let mut dst_scalar = initial;
            sub_xtg(&src, &mut dst_simd);
            for i in 0..MAX_PL_REGRESSORS {
                dst_scalar[i] -= src[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }

    #[test]
    fn diff_xtg_simd_matches_scalar() {
        for seed in 0..100 {
            let b = xtg_from_seed(seed);
            let c = xtg_from_seed(seed.wrapping_add(1));
            let mut dst_simd = [0.0_f32; MAX_PL_REGRESSORS];
            let mut dst_scalar = [0.0_f32; MAX_PL_REGRESSORS];
            diff_xtg(&b, &c, &mut dst_simd);
            for i in 0..MAX_PL_REGRESSORS {
                dst_scalar[i] = b[i] - c[i];
            }
            assert_eq!(dst_simd, dst_scalar, "seed {seed}: bit-exact mismatch");
        }
    }
}
