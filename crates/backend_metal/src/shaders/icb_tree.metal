//! Stage 4b ICB chaining shaders.
//!
//! Three kernels share `IcbConstants` (48 bytes, bind to highest buffer slot
//! for each kernel). All buffers allocated from one MTLHeap; inherited
//! residency (Metal 4) means no per-buffer useResource calls inside ICB.
//
// Node numbering: root = 0, left(n) = 2n+1, right(n) = 2n+2.
// Level L: first node = (2^L) - 1, count = 2^L.

#include <metal_stdlib>
using namespace metal;

// ── Shared constants struct ─────────────────────────────────────────────────
// Must match IcbConstantsGpu in Rust (48 bytes, 4-byte aligned).
struct IcbConstants {
    uint32_t row_count;
    uint32_t feature_count;
    uint32_t bin_count;
    uint32_t level_node_offset;   // first node index at this level (= 2^L - 1)
    uint32_t level_node_end;      // level_node_offset + 2^L
    uint32_t level_node_count;    // 2^L
    uint32_t min_rows_per_leaf;
    float    min_split_gain;
    float    lambda;
    float    learning_rate;
    uint32_t _pad0;
    uint32_t _pad1;               // pad to 48 bytes
};

// ── Per-node split decision ─────────────────────────────────────────────────
// Must match IcbSplitDecisionGpu in Rust (40 bytes).
// feature_idx == 0xFFFFFFFF means "no split found" (sentinel).
struct SplitDecision {
    uint32_t feature_idx;
    uint32_t threshold_bin;
    uint32_t flags;          // bit 0 = nan_goes_right
    uint32_t _pad;
    float    gain;
    float    grad_left;
    float    hess_left;
    float    grad_total;
    float    hess_total;
    float    _pad2;
};

// ── Kernel 1: icb_histogram ─────────────────────────────────────────────────
// One thread per row. Scatter-accumulates (grad, hess) into the histogram
// for the current level's nodes. Rows belonging to inactive or out-of-level
// nodes are skipped. Histogram layout: [level_node_count × F × B × 2] f32,
// where the two f32 per bin are (grad_sum, hess_sum).

kernel void icb_histogram(
    device const uint16_t*  row_node_id  [[ buffer(0) ]],
    device const uint8_t*   node_active  [[ buffer(1) ]],
    device const float*     gradients    [[ buffer(2) ]],
    device const float*     hessians     [[ buffer(3) ]],
    device const uint8_t*   bin_data     [[ buffer(4) ]],
    device atomic_float*    histograms   [[ buffer(5) ]],
    constant IcbConstants&  c            [[ buffer(6) ]],
    uint                    gid          [[ thread_position_in_grid ]]
) {
    if (gid >= c.row_count) return;

    uint node = row_node_id[gid];
    if (node < c.level_node_offset || node >= c.level_node_end) return;
    if (!node_active[node]) return;

    uint local_node = node - c.level_node_offset;
    float g = gradients[gid];
    float h = hessians[gid];

    for (uint f = 0; f < c.feature_count; f++) {
        uint bin = bin_data[gid * c.feature_count + f];
        uint base = (local_node * c.feature_count + f) * c.bin_count * 2;
        atomic_fetch_add_explicit(
            &histograms[base + bin * 2],     g, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &histograms[base + bin * 2 + 1], h, memory_order_relaxed);
    }
}

// ── Kernel 2: icb_split_find ────────────────────────────────────────────────
// One thread per node at this level. Prefix-scans the histogram to find the
// best (feature, bin) split by Newton gain. Writes decision and activates
// children, or writes leaf_values and leaves children inactive.

kernel void icb_split_find(
    device const atomic_float*  histograms   [[ buffer(0) ]],
    device SplitDecision*       decisions    [[ buffer(1) ]],
    device uint8_t*             node_active  [[ buffer(2) ]],
    device float*               leaf_values  [[ buffer(3) ]],
    constant IcbConstants&      c            [[ buffer(4) ]],
    uint                        node         [[ thread_position_in_grid ]]
) {
    if (node >= c.level_node_count) return;
    uint global_node = c.level_node_offset + node;
    if (!node_active[global_node]) return;

    float best_gain      = c.min_split_gain;
    uint  best_feature   = 0;
    uint  best_bin       = 0;
    float best_grad_left = 0.0f;
    float best_hess_left = 0.0f;
    bool  best_nan_right = false;

    float grad_total = 0.0f;
    float hess_total = 0.0f;

    for (uint f = 0; f < c.feature_count; f++) {
        uint base = (node * c.feature_count + f) * c.bin_count * 2;

        // Compute totals for this feature.
        float feat_grad_total = 0.0f;
        float feat_hess_total = 0.0f;
        for (uint b = 0; b < c.bin_count; b++) {
            feat_grad_total += atomic_load_explicit(
                &histograms[base + b * 2],     memory_order_relaxed);
            feat_hess_total += atomic_load_explicit(
                &histograms[base + b * 2 + 1], memory_order_relaxed);
        }
        if (f == 0) {
            grad_total = feat_grad_total;
            hess_total = feat_hess_total;
        }

        // Missing-value bin is the last bin (bin_count - 1).
        float nan_g = atomic_load_explicit(
            &histograms[base + (c.bin_count - 1) * 2],     memory_order_relaxed);
        float nan_h = atomic_load_explicit(
            &histograms[base + (c.bin_count - 1) * 2 + 1], memory_order_relaxed);

        float g_left = 0.0f;
        float h_left = 0.0f;

        for (uint b = 0; b < c.bin_count - 1; b++) {
            g_left += atomic_load_explicit(
                &histograms[base + b * 2],     memory_order_relaxed);
            h_left += atomic_load_explicit(
                &histograms[base + b * 2 + 1], memory_order_relaxed);

            float g_right = feat_grad_total - g_left;
            float h_right = feat_hess_total - h_left;

            // NaN-right: NaN rows go right; left = g_left (no NaN).
            float h_left_nr  = h_left;
            float g_left_nr  = g_left;
            float h_right_nr = h_right;  // includes NaN

            // NaN-left: NaN rows go left; left = g_left + nan_g.
            float h_left_nl  = h_left  + nan_h;
            float g_left_nl  = g_left  + nan_g;
            float h_right_nl = h_right - nan_h;

            // Try NaN-right.
            if (h_left_nr >= (float)c.min_rows_per_leaf &&
                h_right_nr >= (float)c.min_rows_per_leaf) {
                float g_right_nr = feat_grad_total - g_left_nr;
                float gain_nr = 0.5f * (
                    (g_left_nr  * g_left_nr)  / (h_left_nr  + c.lambda)
                  + (g_right_nr * g_right_nr) / (h_right_nr + c.lambda)
                  - feat_grad_total * feat_grad_total / (feat_hess_total + c.lambda)
                );
                if (gain_nr > best_gain) {
                    best_gain      = gain_nr;
                    best_feature   = f;
                    best_bin       = b;
                    best_grad_left = g_left_nr;
                    best_hess_left = h_left_nr;
                    best_nan_right = true;
                }
            }
            // Try NaN-left.
            if (h_left_nl >= (float)c.min_rows_per_leaf &&
                h_right_nl >= (float)c.min_rows_per_leaf) {
                float g_right_nl = feat_grad_total - g_left_nl;
                float gain_nl = 0.5f * (
                    (g_left_nl  * g_left_nl)  / (h_left_nl  + c.lambda)
                  + (g_right_nl * g_right_nl) / (h_right_nl + c.lambda)
                  - feat_grad_total * feat_grad_total / (feat_hess_total + c.lambda)
                );
                if (gain_nl > best_gain) {
                    best_gain      = gain_nl;
                    best_feature   = f;
                    best_bin       = b;
                    best_grad_left = g_left_nl;
                    best_hess_left = h_left_nl;
                    best_nan_right = false;
                }
            }
        }
    }

    if (best_gain > c.min_split_gain) {
        SplitDecision d;
        d.feature_idx    = best_feature;
        d.threshold_bin  = best_bin;
        d.flags          = best_nan_right ? 1u : 0u;
        d._pad           = 0u;
        d.gain           = best_gain;
        d.grad_left      = best_grad_left;
        d.hess_left      = best_hess_left;
        d.grad_total     = grad_total;
        d.hess_total     = hess_total;
        d._pad2          = 0.0f;
        decisions[global_node] = d;
        // Activate children for the next level.
        node_active[2 * global_node + 1] = 1;
        node_active[2 * global_node + 2] = 1;
    } else {
        // This node is a leaf: compute leaf value and leave children inactive.
        leaf_values[global_node] =
            -c.learning_rate * grad_total / (hess_total + c.lambda);
    }
}

// ── Kernel 3: icb_partition ─────────────────────────────────────────────────
// One thread per row. Moves each row from its current level-L node to its
// level-(L+1) child based on the split decision. Rows in inactive nodes or
// out-of-range nodes are skipped. Rows whose node has the "no split" sentinel
// (feature_idx == 0xFFFFFFFF) are also skipped — they stay in their leaf node.

kernel void icb_partition(
    device uint16_t*              row_node_id  [[ buffer(0) ]],
    device const uint8_t*         node_active  [[ buffer(1) ]],
    device const SplitDecision*   decisions    [[ buffer(2) ]],
    device const uint8_t*         bin_data     [[ buffer(3) ]],
    constant IcbConstants&        c            [[ buffer(4) ]],
    uint                          gid          [[ thread_position_in_grid ]]
) {
    if (gid >= c.row_count) return;

    uint node = row_node_id[gid];
    if (node < c.level_node_offset || node >= c.level_node_end) return;
    if (!node_active[node]) return;

    SplitDecision d = decisions[node];
    // Sentinel: this node has no valid split, row stays here (it is a leaf).
    if (d.feature_idx == 0xFFFFFFFF) return;

    uint bin = bin_data[gid * c.feature_count + d.feature_idx];
    bool nan_goes_right = (d.flags & 1u) != 0u;
    bool is_missing     = (bin == (c.bin_count - 1));
    bool goes_left;
    if (is_missing) {
        goes_left = !nan_goes_right;
    } else {
        goes_left = (bin <= d.threshold_bin);
    }
    row_node_id[gid] = (uint16_t)(goes_left ? (2 * node + 1) : (2 * node + 2));
}
