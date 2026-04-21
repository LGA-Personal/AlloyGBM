// Histogram-subtract kernel — GPU elementwise subtract for the
// level-wise/leaf-wise "parent minus smaller sibling" trick.
//
// Contract:
//   * Deterministic: elementwise f32/f32/u32 subtraction produces bit-
//     identical results to the CPU backend's `subtract_histogram_bundle`,
//     since there is no reduction or reordering involved.
//   * No atomics.
//   * Three flat SoA buffers (`grad_sum`, `hess_sum`, `counts`) of length
//     `F * B` each; thread `i` computes `out[i] = parent[i] - child[i]`
//     for each of the three channels.
//
// Grid:         (ceil(total_elems / BLOCK_SIZE), 1, 1)
// Threads/TG:   BLOCK_SIZE  (1024 typical — 32 SIMD groups of 32 lanes)
//
// `total_elems = feature_count * bin_count`. A single 1-D dispatch
// handles all channels at the same index because their layouts match.
// Each thread guards against its tail index via the `total_elems`
// uniform.
//
// Function constants:
//   * BLOCK_SIZE  (index 0) — threads per threadgroup. Specialization
//                 allows the compiler to unroll and pick optimal register
//                 allocation for the hot path.

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_SIZE [[function_constant(0)]];

struct SubtractUniform {
    uint total_elems;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// --- Kernel ---------------------------------------------------------

kernel void subtract_elementwise(
    device const float*          parent_grad   [[buffer(0)]],
    device const float*          parent_hess   [[buffer(1)]],
    device const uint*           parent_counts [[buffer(2)]],
    device const float*          child_grad    [[buffer(3)]],
    device const float*          child_hess    [[buffer(4)]],
    device const uint*           child_counts  [[buffer(5)]],
    device float*                out_grad      [[buffer(6)]],
    device float*                out_hess      [[buffer(7)]],
    device uint*                 out_counts    [[buffer(8)]],
    constant SubtractUniform&    uniform_      [[buffer(9)]],
    uint                         gid           [[thread_position_in_grid]]
) {
    if (gid >= uniform_.total_elems) {
        return;
    }
    out_grad[gid]   = parent_grad[gid]   - child_grad[gid];
    out_hess[gid]   = parent_hess[gid]   - child_hess[gid];
    // Counts are u32; parent >= child by construction (child is a proper
    // subset of parent), so the subtraction never underflows. Matches
    // the CPU contract in `subtract_histogram_bundle_into`.
    out_counts[gid] = parent_counts[gid] - child_counts[gid];
}
