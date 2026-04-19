// Histogram build kernel — deterministic, no float atomics.
//
// Contract (locked by DECISIONS D-003):
//   * Bit-identical results across runs.
//   * No atomic_fetch_add_explicit on floats at any memory level.
//
// Design:
//
//   Pass 1 (histogram_build_scatter) — per-threadgroup scatter
//   ---------------------------------------------------------
//   Grid: (n_features, n_chunks, 1). Each threadgroup owns exactly one
//   (feature, chunk) pair. A chunk is a contiguous slice of rows in
//   `row_indices`. We run one SIMD group per threadgroup (32 threads).
//
//   Threadgroup memory holds a single float2 histogram of BIN_COUNT bins,
//   indexed by bin. To avoid float atomics while still parallelising
//   writes across 32 lanes, we *partition the histogram by bin mod 32*:
//   lane `k` is the sole writer for every bin `b` with `b % 32 == k`.
//
//   Per iteration (stride 32 over rows):
//     1. Every lane reads its assigned row and computes (bin, grad, hess).
//     2. Serialise over source lanes 0..32: broadcast that lane's
//        (bin, grad, hess) to every lane via `simd_shuffle`. The one
//        lane where `shuffled_bin % 32 == thread_in_tg` adds into
//        `hist[shuffled_bin]`. Other lanes idle for that inner step.
//     3. After the outer row loop, publish the threadgroup histogram to
//        a device-memory scratch region indexed by (chunk, feature, bin).
//
//   Order is fully deterministic: row stride is deterministic, inner
//   serialisation order is 0..32, and the writer for each bin is fixed
//   by the bin index. No lane ever contends for the same bin within a
//   threadgroup.
//
//   Pass 2 (histogram_reduce) — cross-threadgroup reduction
//   -------------------------------------------------------
//   Grid: (n_features, ceil(BIN_COUNT / 32), 1). Each thread computes one
//   (feature, bin) pair's final float2 by walking every chunk's scratch
//   entry in strict ascending chunk order. Single-threaded per output
//   slot — deterministic by construction.
//
// Function constants (bound at pipeline create in S1.5):
//   0: BIN_COUNT    — number of bins per feature (1..=MAX_BIN_COUNT)
//   1: USE_U16_BINS — false => binned matrix is u8; true => u16
//
// Buffer layout:
//   Pass 1:
//     buffer(0) const uint8_t*  binned_u8  (column-major, [n_rows × n_features])
//     buffer(1) const uint16_t* binned_u16 (column-major, [n_rows × n_features])
//     buffer(2) const float2*   gradients  ([n_rows] — (grad, hess) interleaved)
//     buffer(3) const uint*     row_indices ([node_row_count] — rows in this node)
//     buffer(4) float2*         scratch     ([n_chunks × n_features × BIN_COUNT])
//     buffer(5) const uint&     n_rows_total
//     buffer(6) const uint&     node_row_count
//     buffer(7) const uint&     rows_per_chunk
//     buffer(8) const uint&     n_features
//   Pass 2:
//     buffer(0) const float2*   scratch     (same as pass 1)
//     buffer(1) float2*         output      ([n_features × BIN_COUNT])
//     buffer(2) const uint&     n_chunks
//     buffer(3) const uint&     n_features

#include <metal_stdlib>

using namespace metal;

// Fallbacks let `newLibraryWithSource` succeed without function-constant
// values — the real values are injected at pipeline create time. We cap
// the compile-time histogram array at 4096 bins (32 KiB of tgmem for
// float2), which dominates Apple Silicon's per-threadgroup allocation.
constant uint BIN_COUNT     [[function_constant(0)]];
constant bool USE_U16_BINS  [[function_constant(1)]];

constant uint EFFECTIVE_BIN_COUNT = is_function_constant_defined(BIN_COUNT)
    ? BIN_COUNT
    : 256u;
constant bool EFFECTIVE_USE_U16 = is_function_constant_defined(USE_U16_BINS)
    ? USE_U16_BINS
    : false;

constant uint MAX_BIN_COUNT = 4096u;
constant uint SIMD_WIDTH    = 32u;

static inline uint load_bin(
    device const uint8_t*  binned_u8,
    device const uint16_t* binned_u16,
    uint row,
    uint feature,
    uint n_rows_total
) {
    const uint index = feature * n_rows_total + row;
    return EFFECTIVE_USE_U16 ? uint(binned_u16[index]) : uint(binned_u8[index]);
}

kernel void histogram_build_scatter(
    device const uint8_t*  binned_u8      [[buffer(0)]],
    device const uint16_t* binned_u16     [[buffer(1)]],
    device const float2*   gradients      [[buffer(2)]],
    device const uint*     row_indices    [[buffer(3)]],
    device float2*         scratch        [[buffer(4)]],
    constant uint&         n_rows_total   [[buffer(5)]],
    constant uint&         node_row_count [[buffer(6)]],
    constant uint&         rows_per_chunk [[buffer(7)]],
    constant uint&         n_features     [[buffer(8)]],
    uint3                  thread_in_tg3  [[thread_position_in_threadgroup]],
    uint3                  tg_id3         [[threadgroup_position_in_grid]]
) {
    const uint thread_in_tg = thread_in_tg3.x;
    const uint feature      = tg_id3.x;
    const uint chunk        = tg_id3.y;

    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // Shared histogram. Per-bin single-writer discipline (see file header)
    // makes this race-free without any atomic.
    threadgroup float2 local_hist[MAX_BIN_COUNT];
    for (uint i = thread_in_tg; i < EFFECTIVE_BIN_COUNT; i += SIMD_WIDTH) {
        local_hist[i] = float2(0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row_cursor = chunk_start; row_cursor < chunk_end; row_cursor += SIMD_WIDTH) {
        const uint my_offset = row_cursor + thread_in_tg;
        const bool my_active = my_offset < chunk_end;

        uint   my_bin = 0u;
        float2 my_gh  = float2(0.0f, 0.0f);
        if (my_active) {
            const uint row = row_indices[my_offset];
            my_bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
            my_gh  = gradients[row];
        }

        // Serialise over source lanes — every lane observes the same
        // src_lane order (0..32), so every (bin, lane) destination is
        // deterministic.
        for (uint src_lane = 0u; src_lane < SIMD_WIDTH; ++src_lane) {
            const uint   src_bin    = simd_shuffle(my_bin,  src_lane);
            const float  src_grad   = simd_shuffle(my_gh.x, src_lane);
            const float  src_hess   = simd_shuffle(my_gh.y, src_lane);
            const uint   src_active = simd_shuffle(uint(my_active ? 1u : 0u), src_lane);

            const bool owns_bin = ((src_bin % SIMD_WIDTH) == thread_in_tg);
            if (src_active != 0u && owns_bin && src_bin < EFFECTIVE_BIN_COUNT) {
                local_hist[src_bin] += float2(src_grad, src_hess);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Publish to device scratch: layout is (chunk, feature, bin).
    const uint scratch_base = (chunk * n_features + feature) * EFFECTIVE_BIN_COUNT;
    for (uint i = thread_in_tg; i < EFFECTIVE_BIN_COUNT; i += SIMD_WIDTH) {
        scratch[scratch_base + i] = local_hist[i];
    }
}

kernel void histogram_reduce(
    device const float2* scratch      [[buffer(0)]],
    device float2*       output       [[buffer(1)]],
    constant uint&       n_chunks     [[buffer(2)]],
    constant uint&       n_features   [[buffer(3)]],
    uint3                thread_in_tg3 [[thread_position_in_threadgroup]],
    uint3                tg_id3        [[threadgroup_position_in_grid]]
) {
    const uint thread_in_tg = thread_in_tg3.x;
    const uint feature      = tg_id3.x;
    const uint bin          = tg_id3.y * SIMD_WIDTH + thread_in_tg;
    if (bin >= EFFECTIVE_BIN_COUNT) {
        return;
    }

    float2 accum = float2(0.0f, 0.0f);
    for (uint c = 0u; c < n_chunks; ++c) {
        accum += scratch[(c * n_features + feature) * EFFECTIVE_BIN_COUNT + bin];
    }

    output[feature * EFFECTIVE_BIN_COUNT + bin] = accum;
}
