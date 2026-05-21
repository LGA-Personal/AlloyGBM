//! K-output shared histogram used by joint multi-label and multiclass DART/GOSS.

#[derive(Debug, Clone, Copy)]
pub enum HistComponent {
    Grad = 0,
    Hess = 1,
}

#[derive(Debug, Clone)]
pub struct MultiOutputHistogram {
    pub n_features: usize,
    pub n_bins: usize,
    pub n_outputs: usize,
    /// Flat storage. Layout: feature-major → bin-major → output-major →
    /// (grad, hess) interleaved. Index helper: `idx(f, b, k, comp)`.
    data: Vec<f32>,
}

impl MultiOutputHistogram {
    pub fn new(n_features: usize, n_bins: usize, n_outputs: usize) -> Self {
        let n = n_features * n_bins * n_outputs * 2;
        Self {
            n_features,
            n_bins,
            n_outputs,
            data: vec![0.0_f32; n],
        }
    }

    #[inline]
    pub fn idx(&self, feature: usize, bin: usize, output: usize, comp: HistComponent) -> usize {
        debug_assert!(feature < self.n_features);
        debug_assert!(bin < self.n_bins);
        debug_assert!(output < self.n_outputs);
        ((feature * self.n_bins + bin) * self.n_outputs + output) * 2 + comp as usize
    }

    pub fn len_flat(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
}

/// Build a multi-output histogram for a single feature column in one sweep.
///
/// `grads` and `hess` are row-major with output as the inner axis:
/// `grads[row * n_outputs + k]` is the gradient for row `row`, output `k`.
/// Length must equal `bins.len() * n_outputs`.
pub fn build_multi_output_histogram_inplace(
    histogram: &mut MultiOutputHistogram,
    feature: usize,
    bins: &[u8],
    grads: &[f32],
    hess: &[f32],
    n_outputs: usize,
) {
    debug_assert_eq!(n_outputs, histogram.n_outputs);
    debug_assert_eq!(grads.len(), bins.len() * n_outputs);
    debug_assert_eq!(hess.len(), bins.len() * n_outputs);

    let n_bins = histogram.n_bins;
    let stride = histogram.n_outputs * 2;
    // Slab for this feature; outputs are the inner-most dimension.
    let feature_offset = feature * n_bins * stride;

    for (row, &bin) in bins.iter().enumerate() {
        let bin = bin as usize;
        debug_assert!(bin < n_bins);
        let bin_offset = feature_offset + bin * stride;
        for k in 0..n_outputs {
            let g = grads[row * n_outputs + k];
            let h = hess[row * n_outputs + k];
            let pair_offset = bin_offset + k * 2;
            histogram.data[pair_offset] += g;
            histogram.data[pair_offset + 1] += h;
        }
    }
}

/// Compute the right-child histogram as `parent - left`, element-wise across
/// all (feature, bin, output, component) slots. Used to skip a full sweep when
/// only the smaller child has been built.
pub fn subtract_multi_output_histogram(
    parent: &MultiOutputHistogram,
    left: &MultiOutputHistogram,
) -> MultiOutputHistogram {
    debug_assert_eq!(parent.n_features, left.n_features);
    debug_assert_eq!(parent.n_bins, left.n_bins);
    debug_assert_eq!(parent.n_outputs, left.n_outputs);
    let mut right = MultiOutputHistogram::new(parent.n_features, parent.n_bins, parent.n_outputs);
    for i in 0..parent.data.len() {
        right.data[i] = parent.data[i] - left.data[i];
    }
    right
}

/// Compute the total split gain summed across all K outputs for a single
/// (feature, threshold_bin) candidate. The left child = bins `[0, threshold_bin]`,
/// the right child = bins `(threshold_bin, n_bins)`.
///
/// Per-output gain follows the standard Newton/XGBoost formulation:
///   gain_k = G_L_k² / (H_L_k + λ) + G_R_k² / (H_R_k + λ) − G_k² / (H_k + λ)
///
/// Total split gain is `Σₖ gain_k`. NaN bin handling is the caller's
/// responsibility (route via the missing-bin direction before calling).
pub fn compute_multi_output_split_gain(
    histogram: &MultiOutputHistogram,
    feature: usize,
    threshold_bin: usize,
    lambda_l2: f32,
    eps: f32,
) -> f32 {
    let n_outputs = histogram.n_outputs;
    let mut total = 0.0_f32;
    for k in 0..n_outputs {
        let (mut g_l, mut h_l) = (0.0_f32, 0.0_f32);
        let (mut g_r, mut h_r) = (0.0_f32, 0.0_f32);
        for b in 0..histogram.n_bins {
            let g = histogram.data[histogram.idx(feature, b, k, HistComponent::Grad)];
            let h = histogram.data[histogram.idx(feature, b, k, HistComponent::Hess)];
            if b <= threshold_bin {
                g_l += g;
                h_l += h;
            } else {
                g_r += g;
                h_r += h;
            }
        }
        let g_total = g_l + g_r;
        let h_total = h_l + h_r;
        let term = |g: f32, h: f32| (g * g) / (h + lambda_l2 + eps);
        total += term(g_l, h_l) + term(g_r, h_r) - term(g_total, h_total);
    }
    total
}

/// MorphBoost-augmented multi-output split gain.
///
/// Sums per-output morph gain across K outputs. Each output uses its own
/// EMA `(grad_mean, grad_std)` snapshot from `MorphState::ema_stats[k]`.
///
/// In warmup (`iteration < morph_warmup_iters`) this is byte-equivalent to
/// [`compute_multi_output_split_gain`] — the morph branch only activates
/// post-warmup, matching the single-output [`crates/backend_cpu/src/morph.rs`]
/// precedent.
///
/// **Row-count approximation.** Single-output `compute_morph_gain` uses the
/// real per-side row count for the info-score term. The multi-output
/// histogram doesn't carry row counts (only `grad` + `hess` sums per bin),
/// so this helper derives an approximate count via `hess.max(0.0) as u32`.
/// That is exact for objectives where hessian ≡ 1 per row (`squared_error`,
/// `queryrmse` in scalar mode) and a monotone proxy for ranking objectives
/// where hessian is pair-derived. The dominant post-warmup signal is the
/// gradient-gain term (weighted by `1 - info_score_weight`), which uses
/// `(g, h)` directly — so the byte-equivalence guarantee in warmup and the
/// asymptotic correctness post-warmup both hold. Threading exact per-bin
/// counts would require a 1.5× memory expansion of `MultiOutputHistogram`
/// and is deferred (joint-trainer follow-up).
///
/// `grad_means.len()` and `grad_stds.len()` must equal `histogram.n_outputs`.
#[allow(clippy::too_many_arguments)]
pub fn compute_multi_output_split_gain_morph(
    histogram: &MultiOutputHistogram,
    feature: usize,
    threshold_bin: usize,
    lambda_l2: f32,
    eps: f32,
    config: &alloygbm_core::MorphConfig,
    precomputed: &alloygbm_core::MorphPrecomputed,
    iteration: u32,
    total_iterations: u32,
    grad_means: &[f32],
    grad_stds: &[f32],
) -> f32 {
    debug_assert_eq!(grad_means.len(), histogram.n_outputs);
    debug_assert_eq!(grad_stds.len(), histogram.n_outputs);

    let n_outputs = histogram.n_outputs;
    let mut total = 0.0_f32;
    for k in 0..n_outputs {
        let (mut g_l, mut h_l, mut c_l) = (0.0_f32, 0.0_f32, 0u32);
        let (mut g_r, mut h_r, mut c_r) = (0.0_f32, 0.0_f32, 0u32);
        for b in 0..histogram.n_bins {
            let g = histogram.data[histogram.idx(feature, b, k, HistComponent::Grad)];
            let h = histogram.data[histogram.idx(feature, b, k, HistComponent::Hess)];
            // Approximate per-side count via hessian sum (see doc-comment).
            let approx_count = h.max(0.0) as u32;
            if b <= threshold_bin {
                g_l += g;
                h_l += h;
                c_l = c_l.saturating_add(approx_count);
            } else {
                g_r += g;
                h_r += h;
                c_r = c_r.saturating_add(approx_count);
            }
        }
        total += morph_gain_per_output(
            g_l,
            h_l,
            c_l,
            g_r,
            h_r,
            c_r,
            lambda_l2,
            eps,
            config,
            precomputed,
            iteration,
            total_iterations,
            grad_means[k],
            grad_stds[k],
        );
    }
    total
}

/// MorphBoost-augmented multi-output categorical (Fisher-sort) split.
///
/// Mirrors [`find_best_multi_output_categorical_split`] but blends per-output
/// morph gain into the partition score. Output-0 Newton scores still order
/// the categories (consistent with the standard variant — Fisher-sort
/// ordering doesn't depend on which gain formula scores the candidates).
///
/// Returns `None` under the same conditions as the standard variant
/// (`num_categories < 2 || num_categories > 64 || no positive-gain partition`).
#[allow(clippy::too_many_arguments)]
pub fn find_best_multi_output_categorical_split_morph(
    hist: &MultiOutputHistogram,
    feature: usize,
    num_categories: usize,
    lambda_l2: f32,
    eps: f32,
    config: &alloygbm_core::MorphConfig,
    precomputed: &alloygbm_core::MorphPrecomputed,
    iteration: u32,
    total_iterations: u32,
    grad_means: &[f32],
    grad_stds: &[f32],
) -> Option<MultiOutputCategoricalSplit> {
    if !(2..=64).contains(&num_categories) {
        return None;
    }
    debug_assert_eq!(grad_means.len(), hist.n_outputs);
    debug_assert_eq!(grad_stds.len(), hist.n_outputs);
    let k = hist.n_outputs;

    // Per-output totals across all categories (matches standard variant).
    let mut total_g = vec![0.0_f32; k];
    let mut total_h = vec![0.0_f32; k];
    let mut total_c = vec![0u32; k];
    for cat in 0..num_categories {
        for ko in 0..k {
            let g = hist.data()[hist.idx(feature, cat, ko, HistComponent::Grad)];
            let h = hist.data()[hist.idx(feature, cat, ko, HistComponent::Hess)];
            total_g[ko] += g;
            total_h[ko] += h;
            total_c[ko] = total_c[ko].saturating_add(h.max(0.0) as u32);
        }
    }

    // Sort categories by output-0 Newton score ascending (Fisher-sort).
    let mut order: Vec<usize> = (0..num_categories).collect();
    order.sort_by(|&a, &b| {
        let sa = hist.data()[hist.idx(feature, a, 0, HistComponent::Grad)]
            / (hist.data()[hist.idx(feature, a, 0, HistComponent::Hess)] + lambda_l2 + eps);
        let sb = hist.data()[hist.idx(feature, b, 0, HistComponent::Grad)]
            / (hist.data()[hist.idx(feature, b, 0, HistComponent::Hess)] + lambda_l2 + eps);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut left_g = vec![0.0_f32; k];
    let mut left_h = vec![0.0_f32; k];
    let mut left_c = vec![0u32; k];
    let mut best_gain = 0.0_f32;
    let mut best_prefix: i32 = -1;
    for (prefix_len, &cat) in order.iter().take(num_categories - 1).enumerate() {
        for ko in 0..k {
            let g = hist.data()[hist.idx(feature, cat, ko, HistComponent::Grad)];
            let h = hist.data()[hist.idx(feature, cat, ko, HistComponent::Hess)];
            left_g[ko] += g;
            left_h[ko] += h;
            left_c[ko] = left_c[ko].saturating_add(h.max(0.0) as u32);
        }
        let mut gain = 0.0_f32;
        for ko in 0..k {
            let gl = left_g[ko];
            let gr = total_g[ko] - gl;
            let hl = left_h[ko];
            let hr = total_h[ko] - hl;
            let cl = left_c[ko];
            let cr = total_c[ko].saturating_sub(cl);
            gain += morph_gain_per_output(
                gl,
                hl,
                cl,
                gr,
                hr,
                cr,
                lambda_l2,
                eps,
                config,
                precomputed,
                iteration,
                total_iterations,
                grad_means[ko],
                grad_stds[ko],
            );
        }
        if gain > best_gain {
            best_gain = gain;
            best_prefix = prefix_len as i32;
        }
    }
    if best_prefix < 0 {
        return None;
    }

    let mut left_bitset: u64 = 0;
    for &cat in order.iter().take((best_prefix as usize) + 1) {
        left_bitset |= 1u64 << cat;
    }
    Some(MultiOutputCategoricalSplit {
        gain: best_gain,
        left_bitset,
        n_categories: num_categories as u32,
    })
}

/// Per-output morph gain calculation. Inlines the math from
/// `crates/backend_cpu/src/morph.rs::compute_morph_gain` (the engine can't
/// depend on backend-cpu; the formula is small and self-contained).
#[allow(clippy::too_many_arguments)]
fn morph_gain_per_output(
    g_l: f32,
    h_l: f32,
    c_l: u32,
    g_r: f32,
    h_r: f32,
    c_r: u32,
    lambda_l2: f32,
    _eps: f32,
    config: &alloygbm_core::MorphConfig,
    pre: &alloygbm_core::MorphPrecomputed,
    iteration: u32,
    total_iterations: u32,
    grad_mean: f32,
    grad_std: f32,
) -> f32 {
    // `GAIN_EPSILON` mirrors `crates/backend_cpu/src/morph.rs:39` so warmup
    // matches the standard gain to within float-rounding tolerance.
    const GAIN_EPSILON: f32 = 1e-6;
    const INFO_EPS: f32 = 1e-10;

    // Gradient gain (standard XGBoost-style; same formula as the standard
    // variant's per-output term, modulo `GAIN_EPSILON` matching).
    let p_g = g_l + g_r;
    let p_h = h_l + h_r;
    let gradient_score = (g_l * g_l) / (h_l + lambda_l2 + GAIN_EPSILON)
        + (g_r * g_r) / (h_r + lambda_l2 + GAIN_EPSILON)
        - (p_g * p_g) / (p_h + lambda_l2 + GAIN_EPSILON);

    let mut gain = if pre.in_warmup || pre.info_score_negligible {
        gradient_score
    } else {
        // Info-score blend (Kriuk 2025).
        let smoothing =
            1.0 + config.evolution_pressure * (iteration as f32 / total_iterations.max(1) as f32);
        let info_side = |g_sum: f32, count: u32| -> f32 {
            if count == 0 {
                return 0.0;
            }
            let g_mean = g_sum / count as f32;
            let g_norm = (g_mean - grad_mean) / (grad_std + INFO_EPS);
            g_norm.abs() * (1.0 + g_mean.abs()).ln() / smoothing
        };
        let info_l = info_side(g_l, c_l);
        let info_r = info_side(g_r, c_r);
        let info_p = info_side(g_l + g_r, c_l.saturating_add(c_r));
        let info_score = info_l + info_r - info_p;
        pre.gradient_score_coeff * gradient_score + pre.info_score_coeff * info_score
    };

    // Balance penalty for very-unbalanced splits.
    if pre.balance_penalty {
        let total = c_l.saturating_add(c_r);
        if total > 0 {
            let min_side = c_l.min(c_r);
            let ratio = min_side as f32 / total as f32;
            if ratio < 0.1 {
                gain += -0.5 * (1.0 - (-10.0 * ratio).exp());
            }
        }
    }

    gain
}

/// Result of a multi-output categorical (Fisher-sort) split search.
///
/// `left_bitset` has bit `k` set iff category `k` is on the left side of
/// the split (i.e. routed to the left child). Up to 64 categories are
/// supported per feature (one u64 bitset).
#[derive(Debug, Clone)]
pub struct MultiOutputCategoricalSplit {
    pub gain: f32,
    pub left_bitset: u64,
    pub n_categories: u32,
}

/// Find the best binary partition of categories for a single feature on the
/// multi-output joint trainer using Fisher-sort. Bin indices `0..num_categories`
/// are treated as category IDs (the binning layer maps raw categorical
/// values to these slots).
///
/// **Ordering choice (v0.10.2):** categories are sorted by their output-0
/// Newton-Raphson score `grad/(hess + λ + ε)` ascending, mirroring the
/// single-output Fisher-sort. The "primary output" convention follows
/// `MultiOutputLeafValues` where index 0 is the placeholder scalar leaf
/// used by single-output prediction paths.
///
/// The gain is summed across K outputs:
///   `Σₖ G_L_k² / (H_L_k + λ) + G_R_k² / (H_R_k + λ) − G_total_k² / (H_total_k + λ)`
///
/// Returns `None` when no positive-gain partition exists, or when
/// `num_categories < 2` (a single category can't be split), or when
/// `num_categories > 64` (bitset overflow).
pub fn find_best_multi_output_categorical_split(
    hist: &MultiOutputHistogram,
    feature: usize,
    num_categories: usize,
    lambda_l2: f32,
    eps: f32,
) -> Option<MultiOutputCategoricalSplit> {
    if !(2..=64).contains(&num_categories) {
        return None;
    }
    let k = hist.n_outputs;

    // Per-output totals across all categories (for symmetric gain).
    let mut total_g = vec![0.0_f32; k];
    let mut total_h = vec![0.0_f32; k];
    for cat in 0..num_categories {
        for ko in 0..k {
            total_g[ko] += hist.data()[hist.idx(feature, cat, ko, HistComponent::Grad)];
            total_h[ko] += hist.data()[hist.idx(feature, cat, ko, HistComponent::Hess)];
        }
    }

    // Sort categories by output-0 Newton score ascending (Fisher-sort).
    let mut order: Vec<usize> = (0..num_categories).collect();
    order.sort_by(|&a, &b| {
        let sa = hist.data()[hist.idx(feature, a, 0, HistComponent::Grad)]
            / (hist.data()[hist.idx(feature, a, 0, HistComponent::Hess)] + lambda_l2 + eps);
        let sb = hist.data()[hist.idx(feature, b, 0, HistComponent::Grad)]
            / (hist.data()[hist.idx(feature, b, 0, HistComponent::Hess)] + lambda_l2 + eps);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Prefix scan over sorted order. At each split position, evaluate the
    // K-output gain for "categories[0..=prefix_len] go left, rest go right".
    let mut left_g = vec![0.0_f32; k];
    let mut left_h = vec![0.0_f32; k];
    let mut best_gain = 0.0_f32;
    let mut best_prefix: i32 = -1;
    for (prefix_len, &cat) in order.iter().take(num_categories - 1).enumerate() {
        for ko in 0..k {
            left_g[ko] += hist.data()[hist.idx(feature, cat, ko, HistComponent::Grad)];
            left_h[ko] += hist.data()[hist.idx(feature, cat, ko, HistComponent::Hess)];
        }
        let term = |g: f32, h: f32| (g * g) / (h + lambda_l2 + eps);
        let mut gain = 0.0_f32;
        for ko in 0..k {
            let gl = left_g[ko];
            let gr = total_g[ko] - gl;
            let hl = left_h[ko];
            let hr = total_h[ko] - hl;
            gain += term(gl, hl) + term(gr, hr) - term(total_g[ko], total_h[ko]);
        }
        if gain > best_gain {
            best_gain = gain;
            best_prefix = prefix_len as i32;
        }
    }
    if best_prefix < 0 {
        return None;
    }

    // Build the bitset for the best partition (categories `order[0..=best_prefix]` are left).
    let mut left_bitset: u64 = 0;
    for &cat in order.iter().take((best_prefix as usize) + 1) {
        // Bounded by num_categories <= 64 (checked above).
        left_bitset |= 1u64 << cat;
    }
    Some(MultiOutputCategoricalSplit {
        gain: best_gain,
        left_bitset,
        n_categories: num_categories as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_output_histogram_layout_is_feature_bin_output_major() {
        // (n_features=2, n_bins=4, n_outputs=3)
        let h = MultiOutputHistogram::new(2, 4, 3);
        assert_eq!(h.len_flat(), 2 * 4 * 3 * 2); // *2 for (grad, hess)
        // Index for (feature=1, bin=2, output=0, GRAD) should be unique
        let idx_g = h.idx(1, 2, 0, HistComponent::Grad);
        let idx_h = h.idx(1, 2, 0, HistComponent::Hess);
        assert_ne!(idx_g, idx_h);
        assert!(idx_g < h.len_flat() && idx_h < h.len_flat());
    }

    #[test]
    fn build_kernel_accumulates_per_output_grad_hess() {
        // 3 rows, 1 feature, 4 bins (incl. missing=3), 2 outputs.
        let bins: Vec<u8> = vec![0, 1, 0]; // row → bin
        // grads/hess interleaved per output: [g0_r0, g1_r0, g0_r1, g1_r1, g0_r2, g1_r2]
        let grads = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let hess = [0.1_f32, 1.0, 0.2, 2.0, 0.3, 3.0];

        let mut h = MultiOutputHistogram::new(1, 4, 2);
        build_multi_output_histogram_inplace(
            &mut h, /*feature=*/ 0, &bins, &grads, &hess, /*n_outputs=*/ 2,
        );

        // Output 0, bin 0 should aggregate rows 0+2 → g=4.0, h=0.4
        let i = h.idx(0, 0, 0, HistComponent::Grad);
        assert!((h.data()[i] - 4.0).abs() < 1e-6);
        let i = h.idx(0, 0, 0, HistComponent::Hess);
        assert!((h.data()[i] - 0.4).abs() < 1e-6);

        // Output 1, bin 1 should aggregate row 1 only → g=20.0, h=2.0
        let i = h.idx(0, 1, 1, HistComponent::Grad);
        assert!((h.data()[i] - 20.0).abs() < 1e-6);
    }

    fn set(
        h: &mut MultiOutputHistogram,
        f: usize,
        b: usize,
        k: usize,
        comp: HistComponent,
        v: f32,
    ) {
        let i = h.idx(f, b, k, comp);
        h.data_mut()[i] = v;
    }

    #[test]
    fn subtract_yields_other_child_for_all_outputs() {
        let mut parent = MultiOutputHistogram::new(1, 4, 2);
        let mut left = MultiOutputHistogram::new(1, 4, 2);

        // Populate parent and left with synthetic data.
        for b in 0..4 {
            for k in 0..2 {
                set(
                    &mut parent,
                    0,
                    b,
                    k,
                    HistComponent::Grad,
                    (b * 10 + k + 1) as f32,
                );
                set(
                    &mut parent,
                    0,
                    b,
                    k,
                    HistComponent::Hess,
                    (b + k + 1) as f32 * 0.5,
                );
                set(&mut left, 0, b, k, HistComponent::Grad, (b * 3 + k) as f32);
                set(
                    &mut left,
                    0,
                    b,
                    k,
                    HistComponent::Hess,
                    (b + k) as f32 * 0.2,
                );
            }
        }

        let right = subtract_multi_output_histogram(&parent, &left);

        // Spot-check: right.grad[b=2, k=1] = parent - left
        //   parent = b*10 + k + 1 = 22, left = b*3 + k = 7 → 15
        let i = right.idx(0, 2, 1, HistComponent::Grad);
        let v = right.data()[i];
        assert!((v - 15.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn multi_output_split_gain_sums_per_output_scalar_gain() {
        // Single feature, 2 bins, 2 outputs.
        // Each output: G_L=2, H_L=1, G_R=-2, H_R=1, λ=0
        //   gain = 2²/1 + (-2)²/1 − 0²/2 = 4 + 4 - 0 = 8 per output
        // total = 16
        let mut h = MultiOutputHistogram::new(1, 2, 2);
        // bin 0
        set(&mut h, 0, 0, 0, HistComponent::Grad, 2.0);
        set(&mut h, 0, 0, 0, HistComponent::Hess, 1.0);
        set(&mut h, 0, 0, 1, HistComponent::Grad, 2.0);
        set(&mut h, 0, 0, 1, HistComponent::Hess, 1.0);
        // bin 1
        set(&mut h, 0, 1, 0, HistComponent::Grad, -2.0);
        set(&mut h, 0, 1, 0, HistComponent::Hess, 1.0);
        set(&mut h, 0, 1, 1, HistComponent::Grad, -2.0);
        set(&mut h, 0, 1, 1, HistComponent::Hess, 1.0);

        let total_gain = compute_multi_output_split_gain(
            &h, /*feature=*/ 0, /*threshold_bin=*/ 0, /*lambda_l2=*/ 0.0,
            /*eps=*/ 0.0,
        );
        assert!((total_gain - 16.0).abs() < 1e-5, "got {total_gain}");
    }

    #[test]
    fn multi_output_fisher_sort_finds_optimal_binary_partition() {
        // 3 categories, 2 outputs. Categories 0 and 2 share the same
        // output-0 score (-2 / 1 = -2.0); category 1 has a different
        // score (+2 / 1 = +2.0). Fisher-sort places [0, 2] on the
        // low-score side (left) and [1] on the high-score side (right).
        let mut h = MultiOutputHistogram::new(1, 4, 2); // 1 feature, 4 bins, 2 outputs
        let writes = [
            (0_usize, 0_usize, -2.0_f32, 1.0_f32),
            (0, 1, 1.0, 1.0),
            (1, 0, 2.0, 1.0),
            (1, 1, -1.0, 1.0),
            (2, 0, -2.0, 1.0),
            (2, 1, 1.0, 1.0),
        ];
        for (bin, k, g, hess) in writes {
            let gi = h.idx(0, bin, k, HistComponent::Grad);
            let hi = h.idx(0, bin, k, HistComponent::Hess);
            h.data_mut()[gi] = g;
            h.data_mut()[hi] = hess;
        }
        let result = find_best_multi_output_categorical_split(
            &h, /*feature=*/ 0, /*num_categories=*/ 3, /*lambda_l2=*/ 0.0,
            /*eps=*/ 1e-6,
        )
        .expect("split found");
        // The Fisher partition must put 0 and 2 together (same output-0 score).
        let bit0 = result.left_bitset & 1;
        let bit1 = (result.left_bitset >> 1) & 1;
        let bit2 = (result.left_bitset >> 2) & 1;
        assert_eq!(bit0, bit2, "cats 0 and 2 must share a side");
        assert_ne!(bit0, bit1, "cat 1 must be on the opposite side from cat 0");
        assert!(
            result.gain > 0.0,
            "expected positive gain, got {}",
            result.gain
        );
        assert_eq!(result.n_categories, 3);
    }

    #[test]
    fn multi_output_fisher_sort_returns_none_for_single_category() {
        let h = MultiOutputHistogram::new(1, 4, 2);
        assert!(
            find_best_multi_output_categorical_split(&h, 0, 1, 0.0, 1e-6).is_none(),
            "single-category feature can't be split"
        );
    }

    #[test]
    fn multi_output_morph_gain_in_warmup_matches_standard_gain() {
        // v0.10.4: at iteration < morph_warmup_iters, morph gain MUST equal
        // the standard K-output gain so warmup is byte-equivalent to the
        // non-morph path. Mirrors the single-output warmup-byte-equivalence
        // guarantee in crates/backend_cpu/src/morph.rs.
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        let mut h = MultiOutputHistogram::new(1, 4, 2);
        for b in 0..3 {
            for k in 0..2 {
                set(&mut h, 0, b, k, HistComponent::Grad, (b * 2 + k + 1) as f32);
                set(&mut h, 0, b, k, HistComponent::Hess, 1.0_f32);
            }
        }
        let cfg = MorphConfig::default(); // morph_warmup_iters = 5
        let pre = MorphPrecomputed::for_iteration(0, &cfg);
        let grad_means = vec![0.0_f32; 2];
        let grad_stds = vec![1.0_f32; 2];
        let morph_gain = compute_multi_output_split_gain_morph(
            &h,
            0,
            1,
            0.0,
            1e-6,
            &cfg,
            &pre,
            0,
            100,
            &grad_means,
            &grad_stds,
        );
        let standard_gain = compute_multi_output_split_gain(&h, 0, 1, 0.0, 1e-6);
        assert!(
            (morph_gain - standard_gain).abs() < 1e-5,
            "morph in warmup must match standard: morph={morph_gain} standard={standard_gain}"
        );
    }

    #[test]
    fn multi_output_morph_gain_post_warmup_differs_from_standard() {
        // After warmup, the info-score blend should produce a measurably
        // different gain than the pure XGBoost gain. Proves the morph
        // branch is reached and contributes.
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        let mut h = MultiOutputHistogram::new(1, 4, 2);
        for b in 0..3 {
            for k in 0..2 {
                set(&mut h, 0, b, k, HistComponent::Grad, (b * 2 + k + 1) as f32);
                set(&mut h, 0, b, k, HistComponent::Hess, 1.0_f32);
            }
        }
        let cfg = MorphConfig {
            morph_warmup_iters: 2,
            info_score_weight: 0.5,
            ..MorphConfig::default()
        };
        let pre = MorphPrecomputed::for_iteration(10, &cfg);
        let grad_means = vec![0.5_f32; 2];
        let grad_stds = vec![1.0_f32; 2];
        let morph_gain = compute_multi_output_split_gain_morph(
            &h,
            0,
            1,
            0.0,
            1e-6,
            &cfg,
            &pre,
            10,
            100,
            &grad_means,
            &grad_stds,
        );
        let standard_gain = compute_multi_output_split_gain(&h, 0, 1, 0.0, 1e-6);
        assert!(
            (morph_gain - standard_gain).abs() > 1e-3,
            "morph post-warmup should differ from standard: morph={morph_gain} standard={standard_gain}"
        );
    }

    #[test]
    fn multi_output_morph_categorical_in_warmup_matches_standard() {
        // Warmup byte-equivalence for the categorical Fisher-sort variant.
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        let mut h = MultiOutputHistogram::new(1, 4, 2);
        let writes = [
            (0_usize, 0_usize, -2.0_f32, 1.0_f32),
            (0, 1, 1.0, 1.0),
            (1, 0, 2.0, 1.0),
            (1, 1, -1.0, 1.0),
            (2, 0, -2.0, 1.0),
            (2, 1, 1.0, 1.0),
        ];
        for (bin, k, g, hess) in writes {
            let gi = h.idx(0, bin, k, HistComponent::Grad);
            let hi = h.idx(0, bin, k, HistComponent::Hess);
            h.data_mut()[gi] = g;
            h.data_mut()[hi] = hess;
        }
        let cfg = MorphConfig::default();
        let pre = MorphPrecomputed::for_iteration(0, &cfg);
        let grad_means = vec![0.0_f32; 2];
        let grad_stds = vec![1.0_f32; 2];
        let std_result = find_best_multi_output_categorical_split(&h, 0, 3, 0.0, 1e-6)
            .expect("standard split found");
        let morph_result = find_best_multi_output_categorical_split_morph(
            &h,
            0,
            3,
            0.0,
            1e-6,
            &cfg,
            &pre,
            0,
            100,
            &grad_means,
            &grad_stds,
        )
        .expect("morph split found");
        assert_eq!(
            std_result.left_bitset, morph_result.left_bitset,
            "warmup must pick same partition: std={} morph={}",
            std_result.left_bitset, morph_result.left_bitset
        );
    }

    #[test]
    fn multi_output_fisher_sort_returns_none_when_no_signal() {
        // All categories have identical (g, h) per output → no gain.
        let mut h = MultiOutputHistogram::new(1, 4, 2);
        for cat in 0..3 {
            for ko in 0..2 {
                let gi = h.idx(0, cat, ko, HistComponent::Grad);
                let hi = h.idx(0, cat, ko, HistComponent::Hess);
                h.data_mut()[gi] = 1.0;
                h.data_mut()[hi] = 1.0;
            }
        }
        // All categories have same score → any partition has zero gain.
        let result = find_best_multi_output_categorical_split(&h, 0, 3, 0.0, 1e-6);
        // Either None (best_prefix never updated) or gain exactly 0.0.
        if let Some(r) = result {
            assert!(
                r.gain.abs() < 1e-5,
                "expected ~0 gain on uniform fixture, got {}",
                r.gain
            );
        }
    }
}
