// Stage 4a — GPU best-split kernel for numeric features.
//
// Mirrors `crates/backend_cpu/src/lib.rs::best_split_for_feature`
// byte-for-byte (in floating-point arithmetic order). Categorical
// Fisher-sort stays on CPU; mixed models do per-node host merge.
//
// Two kernel entry points:
//   1. `best_split_per_feature` — one threadgroup per (node, numeric
//      feature). Threads inside a threadgroup own one bin each.
//      Writes a per-(node, feature) candidate to a scratch buffer.
//   2. `best_split_reduce_features` — one threadgroup per node.
//      Reduces the per-feature scratch to a single SplitDecisionGpu.
//
// The wide histogram path stores `(grad, hess, count)` planes
// in row-major (feature, bin) order at the residency-pool entry.
// We read those planes directly in shared storage mode.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr float EPSILON = 1e-6f;
constant constexpr uint MAX_BINS = 1024u;

// Mirror of Rust `SplitDecisionGpu` (24 bytes).
struct SplitDecisionGpu {
    uint feature_idx;
    uint bin_threshold;
    float gain;
    float grad_left;
    float hess_left;
    uint flags;            // bit 0: missing-goes-right; bit 1: invalid
};

// Per-feature scratch entry (one per (node, feature) pair).
struct PerFeatureCandidate {
    float gain;            // unweighted, for downstream weighting
    float weighted_gain;   // gain * feature_weights[fi]
    float grad_left;
    float hess_left;
    uint feature_idx;      // the global feature index this slot was built for
    uint bin_threshold;
    uint flags;            // matches SplitDecisionGpu.flags layout
    uint _pad;             // align to 32 bytes for nicer simdgroup loads
};

// L1 thresholding mirrors `l1_threshold_gradient` in CPU code.
inline float l1_threshold(float grad_sum, float l1_alpha) {
    if (l1_alpha <= 0.0f) return grad_sum;
    if (grad_sum > l1_alpha) return grad_sum - l1_alpha;
    if (grad_sum < -l1_alpha) return grad_sum + l1_alpha;
    return 0.0f;
}

// Compute per-side denom + l1-thresholded grad as a tiny helper.
inline void leaf_terms(float g, float h, float l1, float l2, thread float& out_grad, thread float& out_denom) {
    out_grad = l1_threshold(g, l1);
    out_denom = h + l2 + EPSILON;
}

inline float gain_term(float grad_thresholded, float denom) {
    return (grad_thresholded * grad_thresholded) / denom;
}

struct BestSplitParams {
    float l2_lambda;
    float l1_alpha;
    float min_child_hessian;
    float min_leaf_magnitude;
    uint missing_bin_index; // u8 = 255, u16 = max_data_bin + 1
    uint bin_count;
    uint numeric_feature_count;
    uint node_count;
};

// Kernel 1: best_split_per_feature.
//
// Threadgroup grid: (numeric_feature_count, node_count, 1).
// Threadgroup size:  (bin_count_padded_to_32, 1, 1).
// Each thread loads one bin's stats; reductions are simdgroup + threadgroup.
//
// Inputs:
//   - grads:               [node x feature x bin] f32, row-major (node major)
//   - hesses:              [node x feature x bin] f32, row-major
//   - counts:              [node x feature x bin] u32, row-major
//   - feature_indices:     [numeric_feature_count] u32 - global feature index
//                          for each numeric slot
//   - feature_weights_buf: [numeric_feature_count] f32
//   - params:              BestSplitParams
// Output:
//   - per_feature_scratch: [node x numeric_feature_count] PerFeatureCandidate
[[kernel]]
void best_split_per_feature(
    device const float* grads             [[buffer(0)]],
    device const float* hesses            [[buffer(1)]],
    device const uint* counts             [[buffer(2)]],
    device const uint* feature_indices    [[buffer(3)]],
    device const float* feature_weights   [[buffer(4)]],
    constant BestSplitParams& params      [[buffer(5)]],
    device PerFeatureCandidate* scratch   [[buffer(6)]],
    uint3 tg_id                           [[threadgroup_position_in_grid]],
    uint3 thread_id                       [[thread_position_in_threadgroup]],
    uint3 tg_size                         [[threads_per_threadgroup]]
) {
    const uint feature_slot = tg_id.x;     // 0..numeric_feature_count
    const uint node_idx = tg_id.y;         // 0..node_count
    const uint bin = thread_id.x;          // 0..bin_count_padded
    const uint bin_count = params.bin_count;

    threadgroup float tg_left_grad[MAX_BINS];
    threadgroup float tg_left_hess[MAX_BINS];
    threadgroup uint tg_left_count[MAX_BINS];

    // ---- Load bin stats ----
    // Per-bin grad/hess/count for this (node, feature). Threads beyond
    // bin_count zero-fill (so they don't disturb prefix sums or argmax).
    float my_grad = 0.0f;
    float my_hess = 0.0f;
    uint my_count = 0u;
    if (bin < bin_count) {
        const uint plane_stride = params.numeric_feature_count * bin_count;
        const uint base = node_idx * plane_stride + feature_slot * bin_count + bin;
        my_grad = grads[base];
        my_hess = hesses[base];
        my_count = counts[base];
    }

    // ---- Per-feature totals (reduce across all bins) ----
    // Use simdgroup_sum + threadgroup-broadcast pattern. Two-pass
    // reduction matches the histogram kernel's pattern (D-003).
    float total_grad = simd_sum(my_grad);
    float total_hess = simd_sum(my_hess);
    uint total_count = simd_sum(my_count);

    threadgroup float tg_partial_grad[32];
    threadgroup float tg_partial_hess[32];
    threadgroup uint tg_partial_count[32];

    const uint simd_idx = bin / 32u;
    const uint lane_idx = bin % 32u;
    if (lane_idx == 0) {
        tg_partial_grad[simd_idx] = total_grad;
        tg_partial_hess[simd_idx] = total_hess;
        tg_partial_count[simd_idx] = total_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float gp = (lane_idx < num_simds) ? tg_partial_grad[lane_idx] : 0.0f;
        float hp = (lane_idx < num_simds) ? tg_partial_hess[lane_idx] : 0.0f;
        uint cp = (lane_idx < num_simds) ? tg_partial_count[lane_idx] : 0u;
        gp = simd_sum(gp);
        hp = simd_sum(hp);
        cp = simd_sum(cp);
        if (lane_idx == 0u) {
            tg_partial_grad[0] = gp;
            tg_partial_hess[0] = hp;
            tg_partial_count[0] = cp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_grad = tg_partial_grad[0];
    total_hess = tg_partial_hess[0];
    total_count = tg_partial_count[0];

    // ---- Missing-bin extraction ----
    float missing_grad = 0.0f;
    float missing_hess = 0.0f;
    uint missing_count = 0u;
    if (params.missing_bin_index < bin_count) {
        const uint plane_stride = params.numeric_feature_count * bin_count;
        const uint base = node_idx * plane_stride
            + feature_slot * bin_count
            + params.missing_bin_index;
        missing_grad = grads[base];
        missing_hess = hesses[base];
        missing_count = counts[base];
    }

    // ---- Early-out: parent constraint check ----
    if (total_hess <= params.min_child_hessian) {
        if (bin == 0u) {
            const uint out_idx = node_idx * params.numeric_feature_count + feature_slot;
            scratch[out_idx].gain = -INFINITY;
            scratch[out_idx].weighted_gain = -INFINITY;
            scratch[out_idx].feature_idx = feature_indices[feature_slot];
            scratch[out_idx].bin_threshold = 0u;
            scratch[out_idx].grad_left = 0.0f;
            scratch[out_idx].hess_left = 0.0f;
            scratch[out_idx].flags = 0x2u;  // INVALID
        }
        return;
    }

    const float nm_total_grad = total_grad - missing_grad;
    const float nm_total_hess = total_hess - missing_hess;
    const uint nm_total_count = (total_count > missing_count)
        ? (total_count - missing_count) : 0u;

    const float parent_grad = l1_threshold(total_grad, params.l1_alpha);
    const float parent_denom = total_hess + params.l2_lambda + EPSILON;
    const float parent_gain_term = (parent_grad * parent_grad) / parent_denom;

    // ---- Inclusive prefix scan over non-missing bins ----
    // We exclude the missing bin from the scan: writing zero into
    // tg_left_* at that index makes it inert relative to the
    // prefix-sum totals (because we subtract `missing_*` from totals
    // before computing right-side stats).
    const bool is_missing_slot = (bin == params.missing_bin_index);
    float scan_grad = (bin < bin_count && !is_missing_slot) ? my_grad : 0.0f;
    float scan_hess = (bin < bin_count && !is_missing_slot) ? my_hess : 0.0f;
    uint scan_count = (bin < bin_count && !is_missing_slot) ? my_count : 0u;

    // Two-pass exclusive prefix: simd_prefix_inclusive_sum + simd-block scan.
    // We want INCLUSIVE prefix sum (left side includes bin K when threshold = K).
    float prefix_grad = simd_prefix_inclusive_sum(scan_grad);
    float prefix_hess = simd_prefix_inclusive_sum(scan_hess);
    uint prefix_count = simd_prefix_inclusive_sum(scan_count);

    threadgroup float tg_simd_grad_total[32];
    threadgroup float tg_simd_hess_total[32];
    threadgroup uint tg_simd_count_total[32];
    if (lane_idx == 31u) {
        tg_simd_grad_total[simd_idx] = prefix_grad;
        tg_simd_hess_total[simd_idx] = prefix_hess;
        tg_simd_count_total[simd_idx] = prefix_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate per-simd offsets serially in thread 0, then broadcast.
    if (bin == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float gacc = 0.0f;
        float hacc = 0.0f;
        uint cacc = 0u;
        for (uint i = 0u; i < num_simds; ++i) {
            float g = tg_simd_grad_total[i];
            float h = tg_simd_hess_total[i];
            uint c = tg_simd_count_total[i];
            tg_simd_grad_total[i] = gacc;
            tg_simd_hess_total[i] = hacc;
            tg_simd_count_total[i] = cacc;
            gacc += g;
            hacc += h;
            cacc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix_grad += tg_simd_grad_total[simd_idx];
    prefix_hess += tg_simd_hess_total[simd_idx];
    prefix_count += tg_simd_count_total[simd_idx];

    if (bin < bin_count) {
        tg_left_grad[bin] = prefix_grad;
        tg_left_hess[bin] = prefix_hess;
        tg_left_count[bin] = prefix_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Per-bin gain candidates ----
    // Each thread evaluates threshold = its bin index, comparing
    // tg_left_*[bin] against (nm_total - left + missing_*).
    //
    // CPU mirror: scan_limit = bin_count.min(missing_bin_idx). We
    // skip bins where threshold + 1 >= scan_limit AND nm_total_count
    // == left_count (matches the CPU early-continue for the final
    // bin).
    const uint scan_limit = min(bin_count, params.missing_bin_index);

    float best_gain = 0.0f;
    uint best_threshold = 0u;
    float best_grad_left = 0.0f;
    float best_hess_left = 0.0f;
    uint best_flags = 0x2u;       // INVALID until we find one

    if (bin < scan_limit) {
        const float left_grad = tg_left_grad[bin];
        const float left_hess = tg_left_hess[bin];
        const uint left_count = tg_left_count[bin];

        const bool last_threshold = (bin + 1u >= scan_limit);
        const bool exhausts_right = (nm_total_count == left_count);
        if (!(last_threshold && exhausts_right)) {
            const float right_grad = nm_total_grad - left_grad;
            const float right_hess = nm_total_hess - left_hess;
            const uint right_count = (nm_total_count > left_count)
                ? (nm_total_count - left_count) : 0u;

            // Try NaN-left and NaN-right; keep the better.
            for (uint dir = 0u; dir < 2u; ++dir) {
                const bool nan_left = (dir == 0u);
                const float eff_lg = nan_left ? (left_grad + missing_grad) : left_grad;
                const float eff_lh = nan_left ? (left_hess + missing_hess) : left_hess;
                const uint eff_lc = nan_left ? (left_count + missing_count) : left_count;
                const float eff_rg = nan_left ? right_grad : (right_grad + missing_grad);
                const float eff_rh = nan_left ? right_hess : (right_hess + missing_hess);
                const uint eff_rc = nan_left ? right_count : (right_count + missing_count);

                if (eff_lc == 0u || eff_rc == 0u
                    || eff_lh <= params.min_child_hessian
                    || eff_rh <= params.min_child_hessian) continue;

                float left_grad_for_gain;
                float left_denom;
                float right_grad_for_gain;
                float right_denom;
                leaf_terms(eff_lg, eff_lh, params.l1_alpha, params.l2_lambda,
                           left_grad_for_gain, left_denom);
                leaf_terms(eff_rg, eff_rh, params.l1_alpha, params.l2_lambda,
                           right_grad_for_gain, right_denom);

                if (params.min_leaf_magnitude > 0.0f) {
                    const float lm = fabs(left_grad_for_gain) / left_denom;
                    const float rm = fabs(right_grad_for_gain) / right_denom;
                    if (lm < params.min_leaf_magnitude && rm < params.min_leaf_magnitude) continue;
                }

                const float gain = gain_term(left_grad_for_gain, left_denom)
                    + gain_term(right_grad_for_gain, right_denom)
                    - parent_gain_term;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_threshold = bin;
                    best_grad_left = eff_lg;
                    best_hess_left = eff_lh;
                    best_flags = nan_left ? 0u : 0x1u;
                }
            }
        }
    }

    // ---- Threadgroup argmax across bins ----
    // Two-pass deterministic reduction: each lane writes its (gain,
    // bin) into shared memory, simdgroup max picks the per-simd
    // winner, thread 0 reduces simd winners. Ties: lower bin wins
    // (matches CPU `gain > best_gain` strict comparison).
    threadgroup float tg_best_gain[32];
    threadgroup uint tg_best_threshold[32];
    threadgroup float tg_best_grad_left[32];
    threadgroup float tg_best_hess_left[32];
    threadgroup uint tg_best_flags[32];

    // Per-simdgroup reduction first.
    float lane_gain = best_gain;
    uint lane_threshold = best_threshold;
    float lane_grad_left = best_grad_left;
    float lane_hess_left = best_hess_left;
    uint lane_flags = best_flags;

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        const float other_gain = simd_shuffle_xor(lane_gain, offset);
        const uint other_thresh = simd_shuffle_xor(lane_threshold, offset);
        const float other_gl = simd_shuffle_xor(lane_grad_left, offset);
        const float other_hl = simd_shuffle_xor(lane_hess_left, offset);
        const uint other_flags = simd_shuffle_xor(lane_flags, offset);
        // Tie-break: strictly greater wins; equal keeps lower bin.
        const bool take_other = (other_gain > lane_gain)
            || (other_gain == lane_gain && other_thresh < lane_threshold);
        if (take_other) {
            lane_gain = other_gain;
            lane_threshold = other_thresh;
            lane_grad_left = other_gl;
            lane_hess_left = other_hl;
            lane_flags = other_flags;
        }
    }
    if (lane_idx == 0u) {
        tg_best_gain[simd_idx] = lane_gain;
        tg_best_threshold[simd_idx] = lane_threshold;
        tg_best_grad_left[simd_idx] = lane_grad_left;
        tg_best_hess_left[simd_idx] = lane_hess_left;
        tg_best_flags[simd_idx] = lane_flags;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (bin == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float final_gain = tg_best_gain[0];
        uint final_thresh = tg_best_threshold[0];
        float final_gl = tg_best_grad_left[0];
        float final_hl = tg_best_hess_left[0];
        uint final_flags = tg_best_flags[0];
        for (uint i = 1u; i < num_simds; ++i) {
            const float g = tg_best_gain[i];
            const uint t = tg_best_threshold[i];
            const bool take = (g > final_gain)
                || (g == final_gain && t < final_thresh);
            if (take) {
                final_gain = g;
                final_thresh = t;
                final_gl = tg_best_grad_left[i];
                final_hl = tg_best_hess_left[i];
                final_flags = tg_best_flags[i];
            }
        }

        const uint out_idx = node_idx * params.numeric_feature_count + feature_slot;
        const uint feat_idx = feature_indices[feature_slot];
        const float weighted = final_gain
            * (feature_slot < params.numeric_feature_count
                ? feature_weights[feature_slot] : 1.0f);
        scratch[out_idx].gain = final_gain;
        scratch[out_idx].weighted_gain = weighted;
        scratch[out_idx].grad_left = final_gl;
        scratch[out_idx].hess_left = final_hl;
        scratch[out_idx].feature_idx = feat_idx;
        scratch[out_idx].bin_threshold = final_thresh;
        scratch[out_idx].flags = final_flags;
    }
}

// Kernel 2: best_split_reduce_features.
//
// Threadgroup grid: (node_count, 1, 1).
// Threadgroup size:  (numeric_feature_count_padded_to_32, 1, 1).
[[kernel]]
void best_split_reduce_features(
    device const PerFeatureCandidate* scratch [[buffer(0)]],
    constant BestSplitParams& params         [[buffer(1)]],
    device SplitDecisionGpu* out             [[buffer(2)]],
    uint3 tg_id                              [[threadgroup_position_in_grid]],
    uint3 thread_id                          [[thread_position_in_threadgroup]]
) {
    const uint node_idx = tg_id.x;
    const uint feat_slot = thread_id.x;
    const uint nf = params.numeric_feature_count;

    PerFeatureCandidate cand;
    if (feat_slot < nf) {
        cand = scratch[node_idx * nf + feat_slot];
    } else {
        cand.weighted_gain = -INFINITY;
        cand.gain = -INFINITY;
        cand.feature_idx = 0xFFFFFFFFu;
        cand.bin_threshold = 0u;
        cand.grad_left = 0.0f;
        cand.hess_left = 0.0f;
        cand.flags = 0x2u;  // INVALID
    }

    // Reduce across feature slots; tie-break: strictly greater
    // weighted gain wins; equal keeps lower feature_idx.
    threadgroup PerFeatureCandidate tg_cands[32];
    const uint simd_idx = feat_slot / 32u;
    const uint lane_idx = feat_slot % 32u;

    PerFeatureCandidate lane_cand = cand;
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        PerFeatureCandidate other;
        other.weighted_gain = simd_shuffle_xor(lane_cand.weighted_gain, offset);
        other.gain = simd_shuffle_xor(lane_cand.gain, offset);
        other.feature_idx = simd_shuffle_xor(lane_cand.feature_idx, offset);
        other.bin_threshold = simd_shuffle_xor(lane_cand.bin_threshold, offset);
        other.grad_left = simd_shuffle_xor(lane_cand.grad_left, offset);
        other.hess_left = simd_shuffle_xor(lane_cand.hess_left, offset);
        other.flags = simd_shuffle_xor(lane_cand.flags, offset);
        const bool take = (other.weighted_gain > lane_cand.weighted_gain)
            || (other.weighted_gain == lane_cand.weighted_gain
                && other.feature_idx < lane_cand.feature_idx);
        if (take) lane_cand = other;
    }

    if (lane_idx == 0u) {
        tg_cands[simd_idx] = lane_cand;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (feat_slot == 0u) {
        const uint num_simds = (nf + 31u) / 32u;
        PerFeatureCandidate winner = tg_cands[0];
        for (uint i = 1u; i < num_simds; ++i) {
            const PerFeatureCandidate other = tg_cands[i];
            const bool take = (other.weighted_gain > winner.weighted_gain)
                || (other.weighted_gain == winner.weighted_gain
                    && other.feature_idx < winner.feature_idx);
            if (take) winner = other;
        }

        device SplitDecisionGpu& dest = out[node_idx];
        if (winner.gain > 0.0f && (winner.flags & 0x2u) == 0u) {
            dest.feature_idx = winner.feature_idx;
            dest.bin_threshold = winner.bin_threshold;
            dest.gain = winner.gain;
            dest.grad_left = winner.grad_left;
            dest.hess_left = winner.hess_left;
            dest.flags = winner.flags & ~0x2u;
        } else {
            dest.feature_idx = 0xFFFFFFFFu;
            dest.bin_threshold = 0u;
            dest.gain = 0.0f;
            dest.grad_left = 0.0f;
            dest.hess_left = 0.0f;
            dest.flags = 0x2u;
        }
    }
}
