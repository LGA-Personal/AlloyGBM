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
/// so this helper derives an approximate count via `morph_count_proxy`
/// (see below): for `h > 0` it returns `max(1, ceil(h))` so any bin with
/// positive hessian mass contributes at least one row to the post-warmup
/// `info_side` and balance-penalty computations. The single-output exact
/// path uses true integer counts; on the joint trainer the per-bin
/// hessian is the closest available proxy without a 1.5× expansion of
/// `MultiOutputHistogram` (deferred).
///
/// PR #37 review (C2): a previous draft used `hess.max(0.0) as u32`,
/// which floors fractional hessians (common in ranking objectives where
/// per-row hessians are well below 1) to zero — disabling `info_side`
/// and the balance penalty for ranking. The `morph_count_proxy` rounds
/// up so any positive-hessian bin gets count ≥ 1, restoring both signals.
/// Warmup byte-equivalence with `compute_multi_output_split_gain` still
/// holds because the morph branch is only taken when
/// `!pre.in_warmup && !pre.info_score_negligible`.
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
            let approx_count = morph_count_proxy(h);
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
    find_best_multi_output_categorical_split_morph_with_factor_penalty(
        hist,
        feature,
        num_categories,
        lambda_l2,
        eps,
        config,
        precomputed,
        iteration,
        total_iterations,
        grad_means,
        grad_stds,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn find_best_multi_output_categorical_split_morph_with_factor_penalty(
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
    factor_penalty: Option<MultiOutputCategoricalFactorPenaltyContext<'_>>,
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
            total_c[ko] = total_c[ko].saturating_add(morph_count_proxy(h));
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

    let mut factor_prefix = factor_penalty
        .as_ref()
        .map(|ctx| MultiOutputCategoricalFactorPrefix::new(ctx, feature, &order, num_categories));

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
            left_c[ko] = left_c[ko].saturating_add(morph_count_proxy(h));
        }
        if let Some(prefix) = factor_prefix.as_mut() {
            prefix.add_category_order_index(prefix_len);
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
        if let (Some(ctx), Some(prefix)) = (factor_penalty.as_ref(), factor_prefix.as_ref()) {
            let (leaf_left, leaf_right) = derive_kvec_leaves_from_side_sums(
                &left_g,
                &left_h,
                &total_g,
                &total_h,
                lambda_l2,
                eps,
                ctx.lambda_l1,
                ctx.dro_config,
            );
            gain -= prefix.penalty(
                &leaf_left,
                &leaf_right,
                ctx.factor_penalty,
                ctx.row_indices.len(),
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

/// Bin-level row-count proxy for MorphBoost gain on the multi-output
/// histogram. Returns `0` when `h <= 0` (no row mass in this bin),
/// otherwise `max(1, ceil(h))` — guarantees any positive-hessian bin
/// contributes at least one count so `info_side` and the balance
/// penalty actually fire for ranking objectives where per-row hessians
/// can be well below 1.
///
/// See the doc-comment on `compute_multi_output_split_gain_morph` for
/// why this proxy lives in the morph-gain helpers rather than as a
/// general count field on `MultiOutputHistogram`.
#[inline]
fn morph_count_proxy(h: f32) -> u32 {
    if h <= 0.0 || !h.is_finite() {
        return 0;
    }
    // ceil() returns f32 in [1.0, ∞) for h in (0, ∞); cast to u32
    // saturates on overflow (well-defined per the Rust reference).
    (h.ceil() as u32).max(1)
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

pub struct MultiOutputCategoricalFactorPenaltyContext<'a> {
    pub binned_matrix: &'a alloygbm_core::BinnedMatrix,
    pub exposures: &'a alloygbm_core::FactorExposureMatrix,
    pub row_indices: &'a [u32],
    pub factor_penalty: f32,
    pub lambda_l1: f32,
    pub dro_config: Option<&'a alloygbm_core::DroConfig>,
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
    find_best_multi_output_categorical_split_with_factor_penalty(
        hist,
        feature,
        num_categories,
        lambda_l2,
        eps,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn find_best_multi_output_categorical_split_with_factor_penalty(
    hist: &MultiOutputHistogram,
    feature: usize,
    num_categories: usize,
    lambda_l2: f32,
    eps: f32,
    factor_penalty: Option<MultiOutputCategoricalFactorPenaltyContext<'_>>,
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

    let mut factor_prefix = factor_penalty
        .as_ref()
        .map(|ctx| MultiOutputCategoricalFactorPrefix::new(ctx, feature, &order, num_categories));

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
        if let Some(prefix) = factor_prefix.as_mut() {
            prefix.add_category_order_index(prefix_len);
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
        if let (Some(ctx), Some(prefix)) = (factor_penalty.as_ref(), factor_prefix.as_ref()) {
            let (leaf_left, leaf_right) = derive_kvec_leaves_from_side_sums(
                &left_g,
                &left_h,
                &total_g,
                &total_h,
                lambda_l2,
                eps,
                ctx.lambda_l1,
                ctx.dro_config,
            );
            gain -= prefix.penalty(
                &leaf_left,
                &leaf_right,
                ctx.factor_penalty,
                ctx.row_indices.len(),
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

struct MultiOutputCategoricalFactorPrefix {
    factor_count: usize,
    category_factor_sums: Vec<f32>,
    total_factor_sums: Vec<f32>,
    left_factor_sums: Vec<f32>,
}

impl MultiOutputCategoricalFactorPrefix {
    fn new(
        ctx: &MultiOutputCategoricalFactorPenaltyContext<'_>,
        feature: usize,
        order: &[usize],
        num_categories: usize,
    ) -> Self {
        let factor_count = ctx.exposures.factor_count;
        let mut category_order = vec![usize::MAX; num_categories.min(64)];
        for (order_index, &category) in order.iter().enumerate() {
            if category < category_order.len() {
                category_order[category] = order_index;
            }
        }
        let mut category_factor_sums = vec![0.0_f32; order.len().saturating_mul(factor_count)];
        let mut total_factor_sums = vec![0.0_f32; factor_count];
        let feature_count = ctx.binned_matrix.feature_count;
        let missing_bin = ctx.binned_matrix.missing_bin();
        for &row_u32 in ctx.row_indices {
            let row = row_u32 as usize;
            let bin = ctx.binned_matrix.row_bin(row * feature_count + feature) as usize;
            if bin as u16 == missing_bin || bin >= category_order.len() {
                continue;
            }
            let order_index = category_order[bin];
            if order_index == usize::MAX {
                continue;
            }
            let exposure_start = row * factor_count;
            let exposure_row = &ctx.exposures.values[exposure_start..exposure_start + factor_count];
            let category_base = order_index * factor_count;
            for factor_index in 0..factor_count {
                let exposure = exposure_row[factor_index];
                category_factor_sums[category_base + factor_index] += exposure;
                total_factor_sums[factor_index] += exposure;
            }
        }
        Self {
            factor_count,
            category_factor_sums,
            total_factor_sums,
            left_factor_sums: vec![0.0; factor_count],
        }
    }

    fn add_category_order_index(&mut self, order_index: usize) {
        let base = order_index * self.factor_count;
        for factor_index in 0..self.factor_count {
            self.left_factor_sums[factor_index] += self.category_factor_sums[base + factor_index];
        }
    }

    fn penalty(
        &self,
        leaf_left: &[f32],
        leaf_right: &[f32],
        factor_penalty: f32,
        row_count: usize,
    ) -> f32 {
        if factor_penalty == 0.0 || row_count == 0 {
            return 0.0;
        }
        let mut penalty_sum = 0.0_f32;
        for output_index in 0..leaf_left.len() {
            let left_leaf = leaf_left[output_index];
            let right_leaf = leaf_right[output_index];
            let mut norm_sq = 0.0_f32;
            for factor_index in 0..self.factor_count {
                let left_sum = self.left_factor_sums[factor_index];
                let right_sum = self.total_factor_sums[factor_index] - left_sum;
                let load = left_sum * left_leaf + right_sum * right_leaf;
                norm_sq += load * load;
            }
            penalty_sum += norm_sq;
        }
        factor_penalty * penalty_sum / row_count as f32
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_kvec_leaves_from_side_sums(
    left_g: &[f32],
    left_h: &[f32],
    total_g: &[f32],
    total_h: &[f32],
    lambda_l2: f32,
    eps: f32,
    lambda_l1: f32,
    dro_config: Option<&alloygbm_core::DroConfig>,
) -> (Vec<f32>, Vec<f32>) {
    let n_outputs = left_g.len();
    let mut left = vec![0.0_f32; n_outputs];
    let mut right = vec![0.0_f32; n_outputs];
    for k in 0..n_outputs {
        let gl = left_g[k];
        let hl = left_h[k];
        let gr = total_g[k] - gl;
        let hr = total_h[k] - hl;
        let g_eff_l = alloygbm_core::leaf_effective_gradient(gl, 0.0, 1, lambda_l1, dro_config);
        let g_eff_r = alloygbm_core::leaf_effective_gradient(gr, 0.0, 1, lambda_l1, dro_config);
        left[k] = -g_eff_l / (hl + lambda_l2 + eps);
        right[k] = -g_eff_r / (hr + lambda_l2 + eps);
    }
    (left, right)
}

/// v0.10.6: derive the K-output Newton-Raphson left/right leaf K-vectors for
/// a numeric (threshold-based) split from the multi-output histogram. Used by
/// the joint trainer's `split_penalty` mode at gain-evaluation time to compute
/// the factor-load penalty per candidate.
///
/// Returns `(leaf_left, leaf_right)` each of length `n_outputs`. Mirrors the
/// closed-form leaf step used inside the joint trainer's `leaf_values` closure
/// (sum bins 0..=threshold_bin into left, rest into right), including the
/// L1 / DRO shrinkage path via `alloygbm_core::leaf_effective_gradient`.
///
/// **PR #39 review (R1):** previously the helper used the bare
/// `-g_sum/(h_sum+λ2)` Newton step regardless of `lambda_l1` / `dro_config`.
/// That mis-ranked candidates whenever split_penalty was combined with L1
/// or DRO because the penalty was computed from leaf magnitudes that
/// differed from what `build_joint_round_*` would actually write at leaf
/// finalization. L1 routing is now exact (the soft-threshold uses only
/// `grad_sum` and `l1_alpha`). DRO routing is conservative: the multi-output
/// histogram doesn't carry per-bin `grad_sq_sum`, so we pass `g_sq_sum=0`
/// which collapses the DRO variance term to 0. The resulting leaf magnitudes
/// are an upper bound on the actual DRO-shrunk leaves, which means the
/// penalty is an upper bound too — splits with high factor load are slightly
/// MORE penalized under DRO than a fully-accurate per-bin g_sq accumulation
/// would penalize them. This is the same conservative tradeoff documented
/// for v0.10.5's leaf-only DRO (split-time DRO would require a 1.5× memory
/// inflation on the multi-output histogram).
pub fn derive_kvec_leaves_from_threshold_histogram(
    histogram: &MultiOutputHistogram,
    feature: usize,
    threshold_bin: usize,
    lambda_l2: f32,
    eps: f32,
    lambda_l1: f32,
    dro_config: Option<&alloygbm_core::DroConfig>,
) -> (Vec<f32>, Vec<f32>) {
    let n_outputs = histogram.n_outputs;
    let mut left = vec![0.0_f32; n_outputs];
    let mut right = vec![0.0_f32; n_outputs];
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
        let g_eff_l = alloygbm_core::leaf_effective_gradient(g_l, 0.0, 1, lambda_l1, dro_config);
        let g_eff_r = alloygbm_core::leaf_effective_gradient(g_r, 0.0, 1, lambda_l1, dro_config);
        left[k] = -g_eff_l / (h_l + lambda_l2 + eps);
        right[k] = -g_eff_r / (h_r + lambda_l2 + eps);
    }
    (left, right)
}

/// v0.10.6: same as `derive_kvec_leaves_from_threshold_histogram` but for the
/// native-categorical split path — bin `cat` goes left iff bit `cat` of
/// `left_bitset` is set, mirroring the partition rule in
/// `find_best_multi_output_categorical_split`. Routes through
/// `leaf_effective_gradient` with the same L1-exact / DRO-conservative caveats
/// documented on the threshold variant above.
#[allow(clippy::too_many_arguments)]
pub fn derive_kvec_leaves_from_categorical_histogram(
    histogram: &MultiOutputHistogram,
    feature: usize,
    left_bitset: u64,
    num_categories: usize,
    lambda_l2: f32,
    eps: f32,
    lambda_l1: f32,
    dro_config: Option<&alloygbm_core::DroConfig>,
) -> (Vec<f32>, Vec<f32>) {
    let n_outputs = histogram.n_outputs;
    let mut left = vec![0.0_f32; n_outputs];
    let mut right = vec![0.0_f32; n_outputs];
    for k in 0..n_outputs {
        let (mut g_l, mut h_l) = (0.0_f32, 0.0_f32);
        let (mut g_r, mut h_r) = (0.0_f32, 0.0_f32);
        for cat in 0..num_categories.min(64) {
            let g = histogram.data[histogram.idx(feature, cat, k, HistComponent::Grad)];
            let h = histogram.data[histogram.idx(feature, cat, k, HistComponent::Hess)];
            if (left_bitset >> cat) & 1 == 1 {
                g_l += g;
                h_l += h;
            } else {
                g_r += g;
                h_r += h;
            }
        }
        let g_eff_l = alloygbm_core::leaf_effective_gradient(g_l, 0.0, 1, lambda_l1, dro_config);
        let g_eff_r = alloygbm_core::leaf_effective_gradient(g_r, 0.0, 1, lambda_l1, dro_config);
        left[k] = -g_eff_l / (h_l + lambda_l2 + eps);
        right[k] = -g_eff_r / (h_r + lambda_l2 + eps);
    }
    (left, right)
}

/// v0.10.6: K-output factor-load penalty for a candidate split on the joint
/// multi-output trainer. Generalizes the single-output
/// `factor_split_penalty_for_candidate` (in `crates/backend_cpu/src/lib.rs`)
/// to K outputs by summing the per-output factor-load squared-norm:
///
/// ```text
/// load_{i,k} = left_factor_sums[i] * leaf_left[k]
///            + right_factor_sums[i] * leaf_right[k]
/// penalty    = factor_penalty * Σₖ Σᵢ load_{i,k}^2 / row_count
/// ```
///
/// The caller performs the per-row scan that fills `left_factor_sums` /
/// `right_factor_sums` once per candidate, because the goes-left decision is
/// identical for all K outputs (it depends only on the candidate threshold and
/// row bin, not on output identity). This helper only consumes the
/// pre-computed sums.
pub fn compute_multi_output_factor_split_penalty(
    left_factor_sums: &[f32],
    right_factor_sums: &[f32],
    leaf_left: &[f32],
    leaf_right: &[f32],
    factor_penalty: f32,
    row_count: usize,
) -> f32 {
    if factor_penalty == 0.0 || row_count == 0 {
        return 0.0;
    }
    debug_assert_eq!(left_factor_sums.len(), right_factor_sums.len());
    debug_assert_eq!(leaf_left.len(), leaf_right.len());
    let mut penalty_sum = 0.0_f32;
    for k in 0..leaf_left.len() {
        let lv = leaf_left[k];
        let rv = leaf_right[k];
        let mut norm_sq = 0.0_f32;
        for i in 0..left_factor_sums.len() {
            let load = left_factor_sums[i] * lv + right_factor_sums[i] * rv;
            norm_sq += load * load;
        }
        penalty_sum += norm_sq;
    }
    factor_penalty * penalty_sum / row_count as f32
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

    #[test]
    fn multi_output_categorical_split_penalty_scores_each_prefix() {
        let bins = vec![0_u8, 1, 2];
        let grads = vec![-3.0_f32, -1.0, 3.0];
        let hess = vec![1.0_f32, 1.0, 1.0];
        let mut hist = MultiOutputHistogram::new(1, 4, 1);
        build_multi_output_histogram_inplace(&mut hist, 0, &bins, &grads, &hess, 1);

        let raw =
            find_best_multi_output_categorical_split(&hist, 0, 3, 0.0, 1e-6).expect("raw split");
        assert_eq!(raw.left_bitset, 0b011);

        let binned = alloygbm_core::BinnedMatrix::new(3, 1, 2, bins).expect("binned matrix");
        let exposures =
            alloygbm_core::FactorExposureMatrix::new(3, 1, vec![1.0, 3.0, 0.0]).unwrap();
        let rows = vec![0_u32, 1, 2];
        let penalized = find_best_multi_output_categorical_split_with_factor_penalty(
            &hist,
            0,
            3,
            0.0,
            1e-6,
            Some(MultiOutputCategoricalFactorPenaltyContext {
                binned_matrix: &binned,
                exposures: &exposures,
                row_indices: &rows,
                factor_penalty: 1.0,
                lambda_l1: 0.0,
                dro_config: None,
            }),
        )
        .expect("penalized split");

        assert_eq!(penalized.left_bitset, 0b001);
        assert!(penalized.gain < raw.gain);
    }

    #[test]
    fn factor_split_penalty_zero_when_penalty_zero() {
        let p = compute_multi_output_factor_split_penalty(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[0.5, 0.25],
            &[0.1, 0.2],
            0.0,
            10,
        );
        assert_eq!(p, 0.0);
    }

    #[test]
    fn factor_split_penalty_zero_when_row_count_zero() {
        let p = compute_multi_output_factor_split_penalty(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[0.5, 0.25],
            &[0.1, 0.2],
            0.5,
            0,
        );
        assert_eq!(p, 0.0);
    }

    #[test]
    fn factor_split_penalty_matches_hand_computation() {
        // 1 factor, 2 outputs, row_count=4.
        // left_sum = 2.0, right_sum = -1.0.
        // leaf_left  = [0.5, 0.25]
        // leaf_right = [0.1, 0.2]
        // Output 0: load = 2.0 * 0.5 + (-1.0) * 0.1 = 0.9 → norm² = 0.81
        // Output 1: load = 2.0 * 0.25 + (-1.0) * 0.2 = 0.3 → norm² = 0.09
        // total norm² = 0.90; penalty = 0.5 * 0.90 / 4 = 0.1125
        let p = compute_multi_output_factor_split_penalty(
            &[2.0],
            &[-1.0],
            &[0.5, 0.25],
            &[0.1, 0.2],
            0.5,
            4,
        );
        assert!((p - 0.1125_f32).abs() < 1e-6, "got {p}");
    }
}
