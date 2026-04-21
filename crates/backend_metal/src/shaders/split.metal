// Best-split kernel — per-feature gain sweep with deterministic order.
//
// Contract (locked by DECISIONS for S2):
//   * No float atomics anywhere.
//   * Structural equivalence with CpuBackend: for well-conditioned
//     inputs, every split's (feature_index, threshold_bin, default_left)
//     must match CPU output exactly. Per-bin gain values may differ by
//     a few ulps from the CPU's serial sweep because the GPU folds in
//     a SIMD-width block-scan rather than a single-threaded running
//     accumulator, but the argmax result is stable.
//   * NaN handling per-threshold mirrors CPU: try NaN-left and
//     NaN-right at every candidate threshold and pick the higher-gain
//     option.
//   * L1/L2 regularization mirrors CPU formulas exactly.
//
// Design:
//
//   Kernel: best_split_per_feature
//   -----------------------------
//   Grid:          (n_features, 1, 1)
//   Threads/TG:    32 (exactly one SIMD group)
//
//   Each threadgroup owns exactly one feature. The 32-lane SIMD
//   sweeps the feature's B bins in chunks of 32:
//
//   Phase 1 (totals): stride-32 serial-per-lane reduction over bins
//   to compute total_grad / total_hess / total_count (all bins).
//   Missing-bin stats are read directly from bin index
//   `options.missing_bin_index`.
//
//   Phase 2 (sweep + argmax): outer loop over chunk bases
//   `base = 0, 32, 64, ...` up to `scan_limit = min(BIN_COUNT,
//   missing_bin_index)`. For each chunk:
//     - Each lane L reads (g, h, c) at bin `base + L`.
//     - `simd_prefix_inclusive_sum` produces a prefix inside the chunk.
//     - left_cumulative = running_prefix + simd_prefix  (running base
//       carried between chunks via the chunk-sum of the prior iter).
//     - Active lanes (bin < scan_limit) compute:
//         * NaN-left candidate  : effective L/R stats include missing in L
//         * NaN-right candidate : effective L/R stats include missing in R
//       for each, apply the guard conditions
//       (eff_lc != 0, eff_rc != 0, eff_lh > min_child_hessian,
//        eff_rh > min_child_hessian, min_leaf_magnitude threshold).
//       If both NaN candidates pass, pick the higher-gain one.
//     - Each lane updates its private (best_gain, best_threshold,
//       best_default_left, ...) if this threshold beats the current
//       running best.
//
//   Phase 3 (lane reduction): butterfly reduce across the 32 lanes
//   to pick the threadgroup winner. Tie-break rules (identical to
//   the CPU's left-to-right iteration order):
//     1. "has_split" bit wins over no-split.
//     2. Higher gain wins.
//     3. On exact-tie gain, lower threshold_bin wins.
//     4. On exact-tie threshold, NaN-left (default_left=1) wins.
//
//   The final per-feature candidate (Phase 3 winner, including the
//   has_split bit and full L/R stats) is written to
//   `out_candidates[feature]` by lane 0.
//
//   Cross-feature argmax happens on the CPU side. Reading back
//   n_features FeatureSplitCandidate structs is cheap
//   (~44 bytes / feature) and keeps the kernel simple.
//
// Function constants (bound at pipeline-create time):
//   0: BIN_COUNT  — number of bins per feature (1..=MAX_BIN_COUNT)
//   1: L1_ENABLED — true when l1_alpha > 0; dead-code-eliminates the
//                   soft-threshold branch on the hot path otherwise.
//
// Buffer layout:
//   buffer(0) const float*  grad_sums        ([n_features × BIN_COUNT])
//   buffer(1) const float*  hess_sums        ([n_features × BIN_COUNT])
//   buffer(2) const uint*   counts           ([n_features × BIN_COUNT])
//   buffer(3) const uchar*  continuous_mask  ([n_features]; 1=continuous, 0=categorical)
//   buffer(4) constant SplitOptionsPOD&      options
//   buffer(5) device FeatureSplitCandidate*  out_candidates ([n_features])

#include <metal_stdlib>

using namespace metal;

constant uint BIN_COUNT     [[function_constant(0)]];
constant bool L1_ENABLED    [[function_constant(1)]];

constant uint EFFECTIVE_BIN_COUNT = is_function_constant_defined(BIN_COUNT)
    ? BIN_COUNT
    : 256u;
constant bool EFFECTIVE_L1 = is_function_constant_defined(L1_ENABLED)
    ? L1_ENABLED
    : false;

constant uint  SIMD_WIDTH = 32u;
constant float SPLIT_EPSILON = 1e-6f;

struct SplitOptionsPOD {
    uint  missing_bin_index;
    float l1_alpha;
    float l2_lambda;
    float min_child_hessian;
    float min_leaf_magnitude;
};

// Packed per-feature result. 40 bytes; 4-byte aligned.
struct FeatureSplitCandidate {
    float gain;           // unweighted gain for this feature's best split
    uint  threshold_bin;  // u32 to avoid struct padding surprises
    uint  default_left;   // 0 = NaN goes right; 1 = NaN goes left
    uint  has_split;      // 0 = no valid split; 1 = candidate valid
    float left_grad;
    float left_hess;
    uint  left_count;
    float right_grad;
    float right_hess;
    uint  right_count;
};

static inline float l1_threshold(float g, float alpha) {
    if (!EFFECTIVE_L1) {
        return g;
    }
    if (g >  alpha) return g - alpha;
    if (g < -alpha) return g + alpha;
    return 0.0f;
}

// Order-stable tiebreak: returns true if `cand_b` should replace `cand_a`
// as the running best. Mirrors the CPU's left-to-right sweep in which
// the first threshold seen with strictly greater gain wins, and ties
// (which the CPU implicitly breaks by "earlier iteration wins") are
// resolved here by (lower threshold_bin, then default_left=true) so
// that any GPU permutation of the lanes still converges on the same
// candidate as the CPU's serial order.
static inline bool candidate_replaces(
    uint  a_has,  float a_gain, uint a_thr, uint a_dl,
    uint  b_has,  float b_gain, uint b_thr, uint b_dl
) {
    if (b_has == 0u) {
        return false;
    }
    if (a_has == 0u) {
        return true;
    }
    if (b_gain > a_gain) return true;
    if (b_gain < a_gain) return false;
    // Exact-equal gains — CPU picks the earlier iteration; earlier
    // iteration has the smaller threshold_bin (and, within a single
    // threshold, NaN-left is evaluated before NaN-right).
    if (b_thr < a_thr) return true;
    if (b_thr > a_thr) return false;
    return (b_dl > a_dl);
}

kernel void best_split_per_feature(
    device const float*   grad_sums       [[buffer(0)]],
    device const float*   hess_sums       [[buffer(1)]],
    device const uint*    counts          [[buffer(2)]],
    device const uchar*   continuous_mask [[buffer(3)]],
    constant SplitOptionsPOD&       options [[buffer(4)]],
    device FeatureSplitCandidate*   out_candidates [[buffer(5)]],
    uint3 thread_in_tg3 [[thread_position_in_threadgroup]],
    uint3 tg_id3        [[threadgroup_position_in_grid]]
) {
    const uint tid      = thread_in_tg3.x;
    const uint feature  = tg_id3.x;
    const uint feat_base = feature * EFFECTIVE_BIN_COUNT;

    // Categorical features are handled on the CPU. Emit an empty
    // candidate here so the CPU-side post-processor can skip the slot
    // without reading stale device memory.
    if (continuous_mask[feature] == 0u) {
        if (tid == 0u) {
            FeatureSplitCandidate blank;
            blank.gain = 0.0f;
            blank.threshold_bin = 0u;
            blank.default_left = 0u;
            blank.has_split = 0u;
            blank.left_grad = 0.0f;
            blank.left_hess = 0.0f;
            blank.left_count = 0u;
            blank.right_grad = 0.0f;
            blank.right_hess = 0.0f;
            blank.right_count = 0u;
            out_candidates[feature] = blank;
        }
        return;
    }

    // ---- Phase 1: total sums + missing-bin stats ----
    //
    // Each lane walks every 32nd bin and keeps a local sum; then a
    // single SIMD reduction yields the totals. Deterministic: the
    // lane partition is fixed and every lane accumulates in the same
    // bin-ascending order.
    float my_grad = 0.0f;
    float my_hess = 0.0f;
    uint  my_count = 0u;
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += SIMD_WIDTH) {
        my_grad  += grad_sums[feat_base + b];
        my_hess  += hess_sums[feat_base + b];
        my_count += counts   [feat_base + b];
    }
    const float total_grad  = simd_sum(my_grad);
    const float total_hess  = simd_sum(my_hess);
    const uint  total_count = simd_sum(my_count);

    float missing_grad  = 0.0f;
    float missing_hess  = 0.0f;
    uint  missing_count = 0u;
    if (options.missing_bin_index < EFFECTIVE_BIN_COUNT) {
        const uint idx = feat_base + options.missing_bin_index;
        missing_grad  = grad_sums[idx];
        missing_hess  = hess_sums[idx];
        missing_count = counts   [idx];
    }

    // Early-out mirrors the CPU: abort if even the parent is too thin.
    if (total_hess <= options.min_child_hessian) {
        if (tid == 0u) {
            FeatureSplitCandidate blank;
            blank.gain = 0.0f;
            blank.threshold_bin = 0u;
            blank.default_left = 0u;
            blank.has_split = 0u;
            blank.left_grad = 0.0f;
            blank.left_hess = 0.0f;
            blank.left_count = 0u;
            blank.right_grad = 0.0f;
            blank.right_hess = 0.0f;
            blank.right_count = 0u;
            out_candidates[feature] = blank;
        }
        return;
    }

    const float nm_total_grad  = total_grad  - missing_grad;
    const float nm_total_hess  = total_hess  - missing_hess;
    const uint  nm_total_count = (total_count >= missing_count)
        ? (total_count - missing_count) : 0u;

    const float parent_denom      = total_hess + options.l2_lambda + SPLIT_EPSILON;
    const float parent_grad_l1    = l1_threshold(total_grad, options.l1_alpha);
    const float parent_gain_term  = (parent_grad_l1 * parent_grad_l1) / parent_denom;

    // ---- Phase 2: block-scan sweep ----
    const uint scan_limit = min(EFFECTIVE_BIN_COUNT, options.missing_bin_index);

    float running_grad  = 0.0f;
    float running_hess  = 0.0f;
    uint  running_count = 0u;

    // Per-lane running best. Initialised to "no split".
    float best_gain = 0.0f;
    uint  best_thr = 0u;
    uint  best_dl  = 0u;
    uint  best_has = 0u;
    float best_lg = 0.0f, best_lh = 0.0f;
    uint  best_lc = 0u;
    float best_rg = 0.0f, best_rh = 0.0f;
    uint  best_rc = 0u;

    for (uint base = 0u; base < scan_limit; base += SIMD_WIDTH) {
        const uint b     = base + tid;
        const bool active = (b < scan_limit);

        float g = 0.0f, h = 0.0f;
        uint  c = 0u;
        if (active) {
            g = grad_sums[feat_base + b];
            h = hess_sums[feat_base + b];
            c = counts   [feat_base + b];
        }

        const float pref_g = simd_prefix_inclusive_sum(g);
        const float pref_h = simd_prefix_inclusive_sum(h);
        const uint  pref_c = simd_prefix_inclusive_sum(c);

        const float left_grad  = running_grad  + pref_g;
        const float left_hess  = running_hess  + pref_h;
        const uint  left_count = running_count + pref_c;

        if (active) {
            const uint threshold_bin = b;

            // Skip when this is the last non-missing bin and the right
            // side has zero non-missing rows. Matches CPU line 509-511.
            const bool is_last_nm = (threshold_bin + 1u >= scan_limit);
            const bool right_is_empty = (nm_total_count == left_count);
            if (!(is_last_nm && right_is_empty)) {
                const float right_grad  = nm_total_grad  - left_grad;
                const float right_hess  = nm_total_hess  - left_hess;
                const uint  right_count = (nm_total_count >= left_count)
                    ? (nm_total_count - left_count) : 0u;

                // Evaluate NaN-left then NaN-right. CPU order matters
                // for the tie-break — see `candidate_replaces`.
                for (uint nan_dir = 0u; nan_dir < 2u; ++nan_dir) {
                    const bool nan_left = (nan_dir == 0u);
                    const float eff_lg = nan_left ? (left_grad  + missing_grad)
                                                  : left_grad;
                    const float eff_lh = nan_left ? (left_hess  + missing_hess)
                                                  : left_hess;
                    const uint  eff_lc = nan_left ? (left_count + missing_count)
                                                  : left_count;
                    const float eff_rg = nan_left ? right_grad
                                                  : (right_grad  + missing_grad);
                    const float eff_rh = nan_left ? right_hess
                                                  : (right_hess  + missing_hess);
                    const uint  eff_rc = nan_left ? right_count
                                                  : (right_count + missing_count);

                    if (eff_lc == 0u || eff_rc == 0u) continue;
                    if (eff_lh <= options.min_child_hessian) continue;
                    if (eff_rh <= options.min_child_hessian) continue;

                    const float lg_l1       = l1_threshold(eff_lg, options.l1_alpha);
                    const float rg_l1       = l1_threshold(eff_rg, options.l1_alpha);
                    const float left_denom  = eff_lh + options.l2_lambda + SPLIT_EPSILON;
                    const float right_denom = eff_rh + options.l2_lambda + SPLIT_EPSILON;

                    if (options.min_leaf_magnitude > 0.0f) {
                        const float lm = fabs(lg_l1) / left_denom;
                        const float rm = fabs(rg_l1) / right_denom;
                        if (lm < options.min_leaf_magnitude &&
                            rm < options.min_leaf_magnitude) {
                            continue;
                        }
                    }

                    const float gain = (lg_l1 * lg_l1) / left_denom
                                     + (rg_l1 * rg_l1) / right_denom
                                     - parent_gain_term;

                    // CPU uses `gain > best_gain` with initial best=0.
                    // Respect that exactly.
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_thr  = threshold_bin;
                        best_dl   = nan_left ? 1u : 0u;
                        best_has  = 1u;
                        best_lg = eff_lg; best_lh = eff_lh; best_lc = eff_lc;
                        best_rg = eff_rg; best_rh = eff_rh; best_rc = eff_rc;
                    }
                }
            }
        }

        // Carry the chunk total forward. Last lane (31) holds the full
        // chunk sum from simd_prefix_inclusive_sum.
        const float chunk_sum_g = simd_shuffle(pref_g, SIMD_WIDTH - 1u);
        const float chunk_sum_h = simd_shuffle(pref_h, SIMD_WIDTH - 1u);
        const uint  chunk_sum_c = simd_shuffle(pref_c, SIMD_WIDTH - 1u);
        running_grad  += chunk_sum_g;
        running_hess  += chunk_sum_h;
        running_count += chunk_sum_c;
    }

    // ---- Phase 3: SIMD reduction across lanes ----
    //
    // Butterfly reduction: at each stride, lane L compares its best
    // with the best from lane (L ^ stride). Both lanes converge on
    // the same winner by stride=16.
    for (uint stride = 1u; stride < SIMD_WIDTH; stride <<= 1u) {
        const float other_gain = simd_shuffle_xor(best_gain, stride);
        const uint  other_thr  = simd_shuffle_xor(best_thr,  stride);
        const uint  other_dl   = simd_shuffle_xor(best_dl,   stride);
        const uint  other_has  = simd_shuffle_xor(best_has,  stride);
        const float other_lg   = simd_shuffle_xor(best_lg,   stride);
        const float other_lh   = simd_shuffle_xor(best_lh,   stride);
        const uint  other_lc   = simd_shuffle_xor(best_lc,   stride);
        const float other_rg   = simd_shuffle_xor(best_rg,   stride);
        const float other_rh   = simd_shuffle_xor(best_rh,   stride);
        const uint  other_rc   = simd_shuffle_xor(best_rc,   stride);

        if (candidate_replaces(
                best_has,  best_gain, best_thr, best_dl,
                other_has, other_gain, other_thr, other_dl))
        {
            best_gain = other_gain;
            best_thr  = other_thr;
            best_dl   = other_dl;
            best_has  = other_has;
            best_lg = other_lg; best_lh = other_lh; best_lc = other_lc;
            best_rg = other_rg; best_rh = other_rh; best_rc = other_rc;
        }
    }

    if (tid == 0u) {
        FeatureSplitCandidate out;
        out.gain = best_gain;
        out.threshold_bin = best_thr;
        out.default_left = best_dl;
        out.has_split = best_has;
        out.left_grad = best_lg;
        out.left_hess = best_lh;
        out.left_count = best_lc;
        out.right_grad = best_rg;
        out.right_hess = best_rh;
        out.right_count = best_rc;
        out_candidates[feature] = out;
    }
}
