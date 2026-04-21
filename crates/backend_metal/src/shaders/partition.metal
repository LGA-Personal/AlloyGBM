// Row-partition kernel — GPU stream compaction for `apply_split`.
//
// Contract (DECISIONS S3):
//   * Deterministic: output `left`/`right` vectors preserve the original
//     order of `node.row_indices`, bit-identical to the CPU backend's
//     stable partition. Stream compaction is implemented via prefix-sum
//     scatter, NOT atomic claim — so reordering of lane execution does
//     not perturb the output.
//   * No float atomics; no device-memory integer atomics. Block totals
//     are fanned through a 3-pass scan.
//   * Supports both continuous-threshold and categorical-bitset splits.
//     Chosen via the `SPLIT_KIND` function constant to allow dead-code
//     elimination.
//   * NaN-bin rows go to `split.default_left` in both kinds. Matches
//     `goes_left_for_split` in backend_cpu/src/lib.rs.
//
// Three-pass design:
//
//   Pass 1 — `partition_flag_and_count`
//   ------------------------------------
//   Grid:          (num_blocks, 1, 1)
//   Threads / TG:  BLOCK_SIZE  (power of two <= 1024; typically 1024,
//                   so 32 SIMD groups of 32 lanes each)
//
//   Each thread reads one row index from `row_indices[block*BLOCK+tid]`,
//   looks up its column-major bin, and computes the direction via the
//   same rules as `goes_left_for_split` (bin == missing → default_left;
//   categorical bit test; threshold compare). The per-row flag is
//   written to `direction_flags[i]` (0 = left, 1 = right).
//
//   Threadgroup reduction counts the number of "left" rows in this
//   block. Lane 0 writes `block_left_totals[block_id]`.
//
//   Pass 2 — `partition_scan_blocks`
//   --------------------------------
//   Grid:          (1, 1, 1)
//   Threads / TG:  BLOCK_SIZE  (up to 1024 blocks scanned in a single
//                   threadgroup)
//
//   Single threadgroup performs an exclusive prefix-sum over
//   `block_left_totals`, writing `block_left_bases[block_id]`. The
//   sentinel slot `block_left_bases[num_blocks]` receives the grand
//   total. Bounded by single-threadgroup capacity; beyond BLOCK_SIZE
//   blocks a future hierarchical scan is required — the Rust
//   orchestrator must fall back to CPU above that cap (documented
//   alongside `MAX_BLOCKS_SINGLE_SCAN`).
//
//   Pass 3 — `partition_scatter`
//   ----------------------------
//   Grid:          (num_blocks, 1, 1)
//   Threads / TG:  BLOCK_SIZE
//
//   Each thread re-reads `direction_flags[i]` (bandwidth-cheap) and
//   recomputes the intra-block exclusive prefix sum of "is-left"
//   locally using `simd_prefix_inclusive_sum` + a cross-SIMD-group
//   shared-memory fan-in. Combined with `block_left_bases[block_id]`,
//   this yields the final absolute destination index in either the
//   left or right output buffer:
//
//     left_dst  = block_left_base + local_left_offset
//     right_dst = block_right_base + (tid - local_left_offset)
//     block_right_base = block_start - block_left_base
//
//   `row_indices[i]` is scattered accordingly.
//
// Function constants:
//   0: BLOCK_SIZE    (uint)  threads per threadgroup. Power of two
//                            <= 1024.
//   1: SPLIT_KIND    (uint)  0 = continuous-threshold; 1 = categorical
//                            bitset.
//   2: BIN_IS_U16    (bool)  column-major bin width. When true, the
//                            u8 buffer is bound to a dummy and vice
//                            versa — the kernel reads the selected
//                            buffer via a compile-time branch.
//
// Buffer layout (per pass):
//
//   partition_flag_and_count:
//     buffer(0)  device const uchar*  bins_col_u8
//     buffer(1)  device const ushort* bins_col_u16
//     buffer(2)  device const uint*   row_indices
//     buffer(3)  constant PartitionUniform&  uniform
//     buffer(4)  device const uchar*  categorical_bitset (dummy if
//                                                         SPLIT_KIND==0)
//     buffer(5)  device uchar*        direction_flags
//     buffer(6)  device uint*         block_left_totals
//
//   partition_scan_blocks:
//     buffer(0)  device const uint*   block_left_totals
//     buffer(1)  device uint*         block_left_bases  (len = num_blocks + 1)
//     buffer(2)  constant uint&       num_blocks
//
//   partition_scatter:
//     buffer(0)  device const uint*   row_indices
//     buffer(1)  device const uchar*  direction_flags
//     buffer(2)  device const uint*   block_left_bases
//     buffer(3)  constant uint&       num_rows_in_node
//     buffer(4)  device uint*         out_left
//     buffer(5)  device uint*         out_right

#include <metal_stdlib>

using namespace metal;

constant uint BLOCK_SIZE  [[function_constant(0)]];
constant uint SPLIT_KIND  [[function_constant(1)]];
constant bool BIN_IS_U16  [[function_constant(2)]];

constant uint EFFECTIVE_BLOCK_SIZE = is_function_constant_defined(BLOCK_SIZE)
    ? BLOCK_SIZE
    : 1024u;
constant uint EFFECTIVE_SPLIT_KIND = is_function_constant_defined(SPLIT_KIND)
    ? SPLIT_KIND
    : 0u;
constant bool EFFECTIVE_U16 = is_function_constant_defined(BIN_IS_U16)
    ? BIN_IS_U16
    : false;

constant uint SIMD_WIDTH = 32u;
// BLOCK_SIZE is capped at 1024 and a multiple of 32, so we need at
// most 32 partial-sum slots.
constant uint MAX_SIMD_GROUPS = 32u;

/// Uniform packed for pass-1 kernel. Keep 32-byte aligned for Metal's
/// constant-buffer layout rules.
struct PartitionUniform {
    uint feature_col_base;   // feature_index * row_count
    uint row_count;          // full dataset row_count (stride; unused
                             // after feature_col_base is precomputed,
                             // but kept for clarity)
    uint missing_bin;
    uint num_rows_in_node;
    uint threshold_bin;      // used when SPLIT_KIND == 0
    uint default_left;       // 1 = NaN goes left, 0 = NaN goes right
    uint bitset_byte_len;    // used when SPLIT_KIND == 1
    uint _pad;
};

/// Returns 0 = left, 1 = right. Matches `goes_left_for_split` semantics.
static inline uint direction_for_bin(
    uint bin_val,
    constant PartitionUniform& u,
    device const uchar* bitset
) {
    if (bin_val == u.missing_bin) {
        return (u.default_left != 0u) ? 0u : 1u;
    }
    if (EFFECTIVE_SPLIT_KIND == 1u) {
        // Categorical bitset: bit set → left.
        const uint byte_idx = bin_val / 8u;
        const uint bit_idx  = bin_val % 8u;
        if (byte_idx >= u.bitset_byte_len) {
            // Bin outside encoded bitset extent → right. Matches the
            // CPU's `byte_idx < bs.len()` guard which returns false.
            return 1u;
        }
        const uint set_bit = (uint)(bitset[byte_idx] >> bit_idx) & 1u;
        return (set_bit != 0u) ? 0u : 1u;
    }
    // Continuous threshold: bin <= threshold → left.
    return (bin_val <= u.threshold_bin) ? 0u : 1u;
}

// Threadgroup-shared buffers; separate instances per kernel because
// Metal shares only within a single dispatch.
threadgroup uint tg_pass1_partials[MAX_SIMD_GROUPS];
threadgroup uint tg_pass2_scan[1024];
threadgroup uint tg_pass3_partials[MAX_SIMD_GROUPS];
threadgroup uint tg_pass3_bases[MAX_SIMD_GROUPS];

/// Pass 1: per-row direction flag + per-block left count.
kernel void partition_flag_and_count(
    device const uchar*           bins_u8            [[buffer(0)]],
    device const ushort*          bins_u16           [[buffer(1)]],
    device const uint*            row_indices        [[buffer(2)]],
    constant PartitionUniform&    uniform            [[buffer(3)]],
    device const uchar*           categorical_bitset [[buffer(4)]],
    device uchar*                 direction_flags    [[buffer(5)]],
    device uint*                  block_left_totals  [[buffer(6)]],
    uint3 thread_in_tg3 [[thread_position_in_threadgroup]],
    uint3 tg_id3        [[threadgroup_position_in_grid]],
    uint  simd_lane     [[thread_index_in_simdgroup]],
    uint  simd_group    [[simdgroup_index_in_threadgroup]]
) {
    const uint tid         = thread_in_tg3.x;
    const uint block_id    = tg_id3.x;
    const uint block_start = block_id * EFFECTIVE_BLOCK_SIZE;
    const uint i           = block_start + tid;

    uint is_left_flag = 0u;

    if (i < uniform.num_rows_in_node) {
        const uint row = row_indices[i];
        uint bin = 0u;
        if (EFFECTIVE_U16) {
            bin = (uint)bins_u16[uniform.feature_col_base + row];
        } else {
            bin = (uint)bins_u8[uniform.feature_col_base + row];
        }
        const uint direction = direction_for_bin(bin, uniform, categorical_bitset);
        is_left_flag = (direction == 0u) ? 1u : 0u;
        direction_flags[i] = (uchar)direction;
    }

    // Intra-SIMD-group reduction.
    const uint simd_sum_val = simd_sum(is_left_flag);

    // Stash one partial per SIMD group in shared memory.
    const uint num_simd_groups = EFFECTIVE_BLOCK_SIZE / SIMD_WIDTH;
    if (simd_lane == 0u) {
        tg_pass1_partials[simd_group] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction within SIMD group 0 (up to 32 partials).
    if (simd_group == 0u) {
        const uint partial = (simd_lane < num_simd_groups)
            ? tg_pass1_partials[simd_lane]
            : 0u;
        const uint block_total = simd_sum(partial);
        if (simd_lane == 0u) {
            block_left_totals[block_id] = block_total;
        }
    }
}

/// Pass 2: exclusive prefix-sum over `block_left_totals` into
/// `block_left_bases`. Sentinel slot `[num_blocks]` holds the grand
/// total. Single threadgroup — caller must fall back to CPU above
/// `BLOCK_SIZE` blocks.
kernel void partition_scan_blocks(
    device const uint*  block_left_totals [[buffer(0)]],
    device uint*        block_left_bases  [[buffer(1)]],
    constant uint&      num_blocks        [[buffer(2)]],
    uint3 thread_in_tg3 [[thread_position_in_threadgroup]]
) {
    const uint tid = thread_in_tg3.x;

    uint v = 0u;
    if (tid < num_blocks) {
        v = block_left_totals[tid];
    }
    tg_pass2_scan[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis–Steele inclusive scan; log2(BLOCK_SIZE) rounds.
    for (uint offset = 1u; offset < EFFECTIVE_BLOCK_SIZE; offset <<= 1u) {
        const uint me = tg_pass2_scan[tid];
        uint left = 0u;
        if (tid >= offset) {
            left = tg_pass2_scan[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_pass2_scan[tid] = me + left;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert inclusive → exclusive.
    if (tid < num_blocks) {
        const uint inclusive = tg_pass2_scan[tid];
        block_left_bases[tid] = inclusive - v;
    }
    // Sentinel: grand total.
    if (tid == num_blocks) {
        const uint grand_total = (num_blocks == 0u)
            ? 0u
            : tg_pass2_scan[num_blocks - 1u];
        block_left_bases[num_blocks] = grand_total;
    }
}

/// Pass 3: recompute intra-block prefix and scatter.
kernel void partition_scatter(
    device const uint*   row_indices      [[buffer(0)]],
    device const uchar*  direction_flags  [[buffer(1)]],
    device const uint*   block_left_bases [[buffer(2)]],
    constant uint&       num_rows_in_node [[buffer(3)]],
    device uint*         out_left         [[buffer(4)]],
    device uint*         out_right        [[buffer(5)]],
    uint3 thread_in_tg3  [[thread_position_in_threadgroup]],
    uint3 tg_id3         [[threadgroup_position_in_grid]],
    uint  simd_lane      [[thread_index_in_simdgroup]],
    uint  simd_group     [[simdgroup_index_in_threadgroup]]
) {
    const uint tid         = thread_in_tg3.x;
    const uint block_id    = tg_id3.x;
    const uint block_start = block_id * EFFECTIVE_BLOCK_SIZE;
    const uint i           = block_start + tid;
    const bool active      = (i < num_rows_in_node);

    const uint block_left_base  = block_left_bases[block_id];
    const uint block_right_base = block_start - block_left_base;

    uint is_left_flag = 0u;
    uint direction    = 1u;
    if (active) {
        direction    = (uint)direction_flags[i];
        is_left_flag = (direction == 0u) ? 1u : 0u;
    }

    // Intra-SIMD-group inclusive → exclusive prefix.
    const uint simd_inclusive = simd_prefix_inclusive_sum(is_left_flag);
    const uint simd_exclusive = simd_inclusive - is_left_flag;

    // Cross-SIMD-group fan-in: each SIMD group's total flows through
    // shared memory, then a single SIMD group exclusive-scans those
    // totals into per-SIMD-group bases.
    const uint simd_group_total = simd_shuffle(simd_inclusive, SIMD_WIDTH - 1u);
    if (simd_lane == 0u) {
        tg_pass3_partials[simd_group] = simd_group_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_simd_groups = EFFECTIVE_BLOCK_SIZE / SIMD_WIDTH;
    if (simd_group == 0u) {
        const uint v = (simd_lane < num_simd_groups)
            ? tg_pass3_partials[simd_lane]
            : 0u;
        const uint inclusive_outer = simd_prefix_inclusive_sum(v);
        const uint exclusive_outer = inclusive_outer - v;
        if (simd_lane < num_simd_groups) {
            tg_pass3_bases[simd_lane] = exclusive_outer;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simd_group_base  = tg_pass3_bases[simd_group];
    const uint local_left_offset = simd_group_base + simd_exclusive;

    if (active) {
        const uint row_val = row_indices[i];
        if (direction == 0u) {
            out_left[block_left_base + local_left_offset] = row_val;
        } else {
            // Every earlier lane in the block that was not-left was
            // right, so local_right_offset = tid - local_left_offset.
            const uint local_right_offset = tid - local_left_offset;
            out_right[block_right_base + local_right_offset] = row_val;
        }
    }
}
