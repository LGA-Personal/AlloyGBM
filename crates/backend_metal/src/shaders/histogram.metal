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
//   entry in strict ascending chunk order, then scatters the accumulated
//   values into two SoA output planes (grad_out, hess_out).
//   Single-threaded per output slot — deterministic by construction.
//
//   The SoA output format (D-019) matches `HistogramResidencyPool`'s
//   three-buffer storage layout and `split.metal`'s three-buffer input
//   contract, letting the reduce output land directly in pool-owned
//   buffers that the split kernel reads without any reshape. The
//   scatter pass (pass 1) still uses an internal `float2 local_hist`
//   because the per-bin single-writer discipline benefits from keeping
//   `(grad, hess)` coresident in threadgroup memory; only the final
//   device-memory write splits the planes.
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
//     buffer(1) float*          grad_out    ([n_features × BIN_COUNT])
//     buffer(2) float*          hess_out    ([n_features × BIN_COUNT])
//     buffer(3) const uint&     n_chunks
//     buffer(4) const uint&     n_features

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

// -----------------------------------------------------------------
// Wide scatter — D-021 follow-up.
//
// Same deterministic discipline as `histogram_build_scatter` (no
// float atomics, per-bin single-writer within a simdgroup), but
// widens the threadgroup from 1 simdgroup (32 threads) to 4
// simdgroups (128 threads) by giving each simdgroup its own
// private histogram in threadgroup memory. A final tree reduction
// (simdgroup 0 sums in fixed order 0 += 1 += 2 += 3) merges them
// before publishing to `scratch`. Valid only when
// EFFECTIVE_BIN_COUNT <= MAX_BIN_COUNT_WIDE (caller enforces);
// threadgroup allocation is always the 32 KB maximum.
//
// Rationale: the narrow kernel pays ~30x arithmetic overhead per
// row because 31/32 lanes idle during the `src_lane` serialisation.
// Four simdgroups process 4x disjoint rows in parallel while
// keeping the same per-simdgroup discipline, and the end-of-kernel
// tree reduction over 4 histograms is cheap compared to the row
// loop.
// -----------------------------------------------------------------

constant uint SIMDGROUPS_WIDE     = 4u;
constant uint THREADS_WIDE        = SIMD_WIDTH * SIMDGROUPS_WIDE; // 128
constant uint MAX_BIN_COUNT_WIDE  = 1024u;

kernel void histogram_build_scatter_wide(
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
    const uint thread_in_tg  = thread_in_tg3.x;
    const uint simdgroup_id  = thread_in_tg / SIMD_WIDTH;
    const uint lane_in_sg    = thread_in_tg % SIMD_WIDTH;
    const uint feature       = tg_id3.x;
    const uint chunk         = tg_id3.y;

    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // SIMDGROUPS_WIDE private histograms. All four are zero-initialised
    // collaboratively across the 128 threads.
    threadgroup float2 local_hist[SIMDGROUPS_WIDE][MAX_BIN_COUNT_WIDE];
    for (uint sg = 0u; sg < SIMDGROUPS_WIDE; ++sg) {
        for (uint i = thread_in_tg; i < EFFECTIVE_BIN_COUNT; i += THREADS_WIDE) {
            local_hist[sg][i] = float2(0.0f, 0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles a disjoint stride of rows. Stride =
    // THREADS_WIDE = SIMDGROUPS_WIDE * SIMD_WIDTH. Simdgroup `s`
    // starts at `chunk_start + s * SIMD_WIDTH`.
    for (uint row_cursor = chunk_start + simdgroup_id * SIMD_WIDTH;
         row_cursor < chunk_end;
         row_cursor += THREADS_WIDE) {
        const uint my_offset = row_cursor + lane_in_sg;
        const bool my_active = my_offset < chunk_end;

        uint   my_bin = 0u;
        float2 my_gh  = float2(0.0f, 0.0f);
        if (my_active) {
            const uint row = row_indices[my_offset];
            my_bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
            my_gh  = gradients[row];
        }

        // Same intra-simdgroup shuffle + per-bin ownership as the
        // narrow path, applied independently per simdgroup into that
        // simdgroup's private histogram. simd_shuffle is
        // intra-simdgroup, so cross-simdgroup state never mixes here.
        for (uint src_lane = 0u; src_lane < SIMD_WIDTH; ++src_lane) {
            const uint   src_bin    = simd_shuffle(my_bin,  src_lane);
            const float  src_grad   = simd_shuffle(my_gh.x, src_lane);
            const float  src_hess   = simd_shuffle(my_gh.y, src_lane);
            const uint   src_active = simd_shuffle(uint(my_active ? 1u : 0u), src_lane);

            const bool owns_bin = ((src_bin % SIMD_WIDTH) == lane_in_sg);
            if (src_active != 0u && owns_bin && src_bin < EFFECTIVE_BIN_COUNT) {
                local_hist[simdgroup_id][src_bin] += float2(src_grad, src_hess);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree-reduce the four private histograms into local_hist[0].
    // Fixed order `(((local_hist[0] + local_hist[1]) + local_hist[2]) + local_hist[3])`
    // makes the floating-point result bit-reproducible across runs.
    // Simdgroup 0's 32 lanes share the reduction work across bin slots.
    if (simdgroup_id == 0u) {
        for (uint i = lane_in_sg; i < EFFECTIVE_BIN_COUNT; i += SIMD_WIDTH) {
            float2 sum = local_hist[0][i];
            for (uint sg = 1u; sg < SIMDGROUPS_WIDE; ++sg) {
                sum += local_hist[sg][i];
            }
            local_hist[0][i] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Publish to device scratch — same layout as the narrow kernel,
    // so `histogram_reduce` is unchanged.
    const uint scratch_base = (chunk * n_features + feature) * EFFECTIVE_BIN_COUNT;
    for (uint i = thread_in_tg; i < EFFECTIVE_BIN_COUNT; i += THREADS_WIDE) {
        scratch[scratch_base + i] = local_hist[0][i];
    }
}

// -----------------------------------------------------------------
// Dynamic-threadgroup wide scatter — Stage 5 occupancy fix +
// gradient pre-gather (Stage 5 bandwidth fix).
//
// Buffer(2) is `gathered_grads`: the caller pre-scatters gradient
// data via `histogram_gather_grads` so that:
//   gathered_grads[i] == gradients[row_indices[i]]  for i in 0..node_row_count
//
// This converts the dominant hot-path memory access from random
// (gradients[row]) to sequential (gathered_grads[chunk_start+tid]),
// reducing gradient DRAM reads from 100× per level to 1× per level:
//
//   Before gather: every one of the 100 features re-reads the 8 MB
//     gradient buffer by random row index → 100× random-access cost.
//   After gather:  the 100 feature TGs all read sequentially from
//     an 8 MB gathered buffer that stays warm in GPU L2 cache → 1×
//     random-access cost (paid by the gather kernel) + near-peak
//     sequential bandwidth for all 100 feature passes.
//
// The bin lookup still uses row_indices → random access into the
// column-major bins buffer, but uint8 elements have 8× better cache-
// line utilization than float2, and the bins buffer column for a
// given feature is 1 MB (fits in L2 after a few chunks).
//
// Occupancy:
//   SIMDGROUPS_WIDE × EFFECTIVE_BIN_COUNT × 8 bytes threadgroup memory
//   = 4 × 256 × 8 = 8 KB for 256 bins → 4 concurrent TGs per CU.
//
// Caller sets threadgroup(0) = SIMDGROUPS_WIDE * bin_count * sizeof(float2).
// Valid when bin_count <= MAX_BIN_COUNT_WIDE.
// -----------------------------------------------------------------

kernel void histogram_build_wide_dyn(
    device const uint8_t*  binned_u8      [[buffer(0)]],
    device const uint16_t* binned_u16     [[buffer(1)]],
    device const float2*   gathered_grads [[buffer(2)]],  // pre-gathered, position-indexed
    device const uint*     row_indices    [[buffer(3)]],  // for bin lookup only
    device float2*         scratch        [[buffer(4)]],
    constant uint&         n_rows_total   [[buffer(5)]],
    constant uint&         node_row_count [[buffer(6)]],
    constant uint&         rows_per_chunk [[buffer(7)]],
    constant uint&         n_features     [[buffer(8)]],
    threadgroup float2*    local_hist     [[threadgroup(0)]],
    uint3                  thread_in_tg3  [[thread_position_in_threadgroup]],
    uint3                  tg_id3         [[threadgroup_position_in_grid]]
) {
    const uint thread_in_tg  = thread_in_tg3.x;
    const uint simdgroup_id  = thread_in_tg / SIMD_WIDTH;
    const uint lane_in_sg    = thread_in_tg % SIMD_WIDTH;
    const uint feature       = tg_id3.x;
    const uint chunk         = tg_id3.y;

    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // Zero-initialise only this simdgroup's private histogram slice.
    // All THREADS_WIDE threads work in parallel across their respective slices.
    const uint sg_base = simdgroup_id * EFFECTIVE_BIN_COUNT;
    for (uint i = lane_in_sg; i < EFFECTIVE_BIN_COUNT; i += SIMD_WIDTH) {
        local_hist[sg_base + i] = float2(0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles a disjoint stride of rows.
    // Simdgroup `s` starts at chunk_start + s * SIMD_WIDTH, stride = THREADS_WIDE.
    for (uint row_cursor = chunk_start + simdgroup_id * SIMD_WIDTH;
         row_cursor < chunk_end;
         row_cursor += THREADS_WIDE) {
        const uint my_offset = row_cursor + lane_in_sg;
        const bool my_active = my_offset < chunk_end;

        uint   my_bin = 0u;
        float2 my_gh  = float2(0.0f, 0.0f);
        if (my_active) {
            // Bin lookup: still uses row_indices → random access into bins buffer.
            // uint8 elements have 64× better cache-line utilization than float2,
            // so this remains much cheaper than the old gradient random access.
            const uint row = row_indices[my_offset];
            my_bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
            // Gradient read: sequential — gathered_grads[my_offset] was written
            // by histogram_gather_grads in node-position order.
            my_gh  = gathered_grads[my_offset];
        }

        // Intra-simdgroup shuffle: lane `k` is sole writer for bins where
        // `bin % SIMD_WIDTH == k` within this simdgroup's private histogram.
        for (uint src_lane = 0u; src_lane < SIMD_WIDTH; ++src_lane) {
            const uint   src_bin    = simd_shuffle(my_bin,  src_lane);
            const float  src_grad   = simd_shuffle(my_gh.x, src_lane);
            const float  src_hess   = simd_shuffle(my_gh.y, src_lane);
            const uint   src_active = simd_shuffle(uint(my_active ? 1u : 0u), src_lane);

            const bool owns_bin = ((src_bin % SIMD_WIDTH) == lane_in_sg);
            if (src_active != 0u && owns_bin && src_bin < EFFECTIVE_BIN_COUNT) {
                local_hist[sg_base + src_bin] += float2(src_grad, src_hess);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree-reduce the SIMDGROUPS_WIDE private histograms into
    // local_hist[0..EFFECTIVE_BIN_COUNT).
    // Fixed accumulation order (0 += 1 += 2 += 3) guarantees
    // bit-identical results across runs.
    if (simdgroup_id == 0u) {
        for (uint i = lane_in_sg; i < EFFECTIVE_BIN_COUNT; i += SIMD_WIDTH) {
            float2 sum = local_hist[i];  // sg 0's contribution
            for (uint sg = 1u; sg < SIMDGROUPS_WIDE; ++sg) {
                sum += local_hist[sg * EFFECTIVE_BIN_COUNT + i];
            }
            local_hist[i] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Publish to device scratch — same (chunk, feature, bin) layout as all
    // other scatter kernels, so `histogram_reduce` is unchanged.
    const uint scratch_base = (chunk * n_features + feature) * EFFECTIVE_BIN_COUNT;
    for (uint i = thread_in_tg; i < EFFECTIVE_BIN_COUNT; i += THREADS_WIDE) {
        scratch[scratch_base + i] = local_hist[i];
    }
}

// -----------------------------------------------------------------
// Gradient pre-gather — Stage 5 bandwidth fix.
//
// Scatters gradient data from the original row-indexed buffer into
// a compact node-position-indexed buffer:
//   gathered_out[i] = gradients[row_indices[i]]  for i in 0..node_row_count
//
// This is a simple 1:1 gather — one thread per row, no reduction.
// Grid: (node_row_count, 1, 1) — dispatched as 1-D.
//
// After this pass, `histogram_build_wide_dyn` reads `gathered_out[i]`
// (sequential, hardware-prefetchable) instead of `gradients[row_indices[i]]`
// (random, cache-unfriendly), reducing gradient DRAM reads from
// n_features × node_row_count to 1 × node_row_count.
//
// Determinism: the gather is a pure permutation — no floating-point
// arithmetic — so results are bit-identical across runs.
// -----------------------------------------------------------------

kernel void histogram_gather_grads(
    device const float2* gradients   [[buffer(0)]],
    device const uint*   row_indices [[buffer(1)]],
    device float2*       gathered    [[buffer(2)]],
    uint                 gid         [[thread_position_in_grid]]
) {
    gathered[gid] = gradients[row_indices[gid]];
}

// -----------------------------------------------------------------
// Tiled private-register scatter — Stage 5.
//
// Replaces the simd_shuffle serialisation (32× inner loop, 31/32
// lanes idle per iteration) with per-thread private-register
// accumulation over TILE_BIN_COUNT bins at a time, followed by a
// two-level tree reduction (simd_sum + inter-simdgroup merge via
// threadgroup memory).  No float atomics anywhere — compiles on
// all MSL versions.
//
// Grid: (n_features, n_chunks, 1) — same as scatter/wide kernels.
// Scratch output: identical layout — `histogram_reduce` unchanged.
//
// Algorithm (one (feature, chunk) threadgroup, iterating tiles):
//   For each 32-bin tile t:
//     1. Each thread strides through its rows, accumulating only
//        those whose bin falls in [tile_start, tile_start+32) into
//        private float registers grad_tile[32] / hess_tile[32].
//        No inter-thread communication, no atomics.
//     2. simd_sum collapses per-lane partials to a per-simdgroup
//        value (identical in all lanes on Apple Silicon).
//     3. Lane 0 of each simdgroup writes 32 floats to threadgroup
//        staging: tg_grad[N_SIMDGROUPS_T][32] + tg_hess[...].
//     4. Simdgroup 0's 32 lanes each own one bin, sum N_SIMDGROUPS_T
//        contributions, and write float2 to scratch.
//
// Memory:
//   Threadgroup: 4 × 32 × 2 × 4 bytes = 1 KB  (32× smaller than
//   the wide kernel's 32 KB — up to 32× more concurrent threadgroups
//   per shader core, hiding memory latency).
//   Registers:   32 floats × 2 planes = 64 floats per thread.
//
// Tile passes: 8 for 256 bins, 32 for 1024 bins.  Chunk data
// (~96 KB for 8192 rows) fits in Apple M-series L1 cache (384 KB),
// so tile passes 2+ are mostly cache hits.
//
// Valid when bin_count ≤ MAX_BIN_COUNT_TILED; the pipeline-selection
// logic in Rust enforces this.
// -----------------------------------------------------------------

constant uint MAX_BIN_COUNT_TILED = 1024u;
constant uint TILE_BIN_COUNT      = 32u;   // equals SIMD_WIDTH
constant uint N_SIMDGROUPS_T      = 4u;
constant uint THREADS_TILED       = N_SIMDGROUPS_T * SIMD_WIDTH; // 128

kernel void histogram_build_tiled(
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
    const uint tid         = thread_in_tg3.x;
    const uint sg_id       = tid / SIMD_WIDTH;  // simdgroup [0..N_SIMDGROUPS_T)
    const uint lane        = tid % SIMD_WIDTH;  // lane [0..32)
    const uint feature     = tg_id3.x;
    const uint chunk       = tg_id3.y;
    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // Threadgroup staging: 4 × 32 × 8 bytes = 1 KB total.
    threadgroup float tg_grad[N_SIMDGROUPS_T][TILE_BIN_COUNT];
    threadgroup float tg_hess[N_SIMDGROUPS_T][TILE_BIN_COUNT];

    const uint scratch_base = (chunk * n_features + feature) * EFFECTIVE_BIN_COUNT;
    const uint n_tiles = (EFFECTIVE_BIN_COUNT + TILE_BIN_COUNT - 1u) / TILE_BIN_COUNT;

    for (uint t = 0u; t < n_tiles; ++t) {
        const uint tile_start = t * TILE_BIN_COUNT;
        const uint tile_end   = min(tile_start + TILE_BIN_COUNT, EFFECTIVE_BIN_COUNT);
        const uint tile_width = tile_end - tile_start;

        // Private accumulators for this tile. Constant array size
        // lets the compiler allocate these in ALU registers.
        float grad_tile[TILE_BIN_COUNT];
        float hess_tile[TILE_BIN_COUNT];
        for (uint b = 0u; b < TILE_BIN_COUNT; ++b) {
            grad_tile[b] = 0.0f;
            hess_tile[b] = 0.0f;
        }

        // Thread `tid` owns rows: chunk_start+tid, +THREADS_TILED, …
        // Accumulate only rows whose bin lands in this tile.
        for (uint i = chunk_start + tid; i < chunk_end; i += THREADS_TILED) {
            const uint   row = row_indices[i];
            const uint   bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
            const float2 gh  = gradients[row];
            if (bin >= tile_start && bin < tile_end) {
                const uint local_b = bin - tile_start;
                grad_tile[local_b] += gh.x;
                hess_tile[local_b] += gh.y;
            }
        }

        // Phase 1: simd_sum collapses per-lane partials to a
        // per-simdgroup sum.  Lane 0 of each simdgroup writes
        // to threadgroup staging.
        for (uint b = 0u; b < tile_width; ++b) {
            const float sg_g = simd_sum(grad_tile[b]);
            const float sg_h = simd_sum(hess_tile[b]);
            if (lane == 0u) {
                tg_grad[sg_id][b] = sg_g;
                tg_hess[sg_id][b] = sg_h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: simdgroup 0's lanes 0..tile_width-1 each own one
        // bin.  Sum the N_SIMDGROUPS_T contributions and write to scratch.
        if (sg_id == 0u && lane < tile_width) {
            float tot_g = 0.0f;
            float tot_h = 0.0f;
            for (uint sg = 0u; sg < N_SIMDGROUPS_T; ++sg) {
                tot_g += tg_grad[sg][lane];
                tot_h += tg_hess[sg][lane];
            }
            scratch[scratch_base + tile_start + lane] = float2(tot_g, tot_h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// -----------------------------------------------------------------
// GPU count accumulation — Stage 5.
//
// Replaces the sequential CPU `accumulate_counts` post-step
// (~1137 ms / 640 calls for the large benchmark).  One threadgroup
// per (feature, chunk); uses threadgroup atomic_uint to count
// per-bin row frequencies, then atomically adds chunk contributions
// to the global count buffer.
//
// Global atomic pressure: n_chunks × bin_count × n_features atomics
// ≈ 122 × 256 × 100 = 3.1 M for the large benchmark (vs 100 M for a
// naive per-row global approach).
//
// Buffer layout:
//   buffer(0) binned_u8   — column-major u8 bins
//   buffer(1) binned_u16  — column-major u16 bins (dummy when u8)
//   buffer(2) row_indices — [node_row_count]
//   buffer(3) counts_out  — device atomic_uint [n_features × BIN_COUNT]
//   buffer(4) n_rows_total
//   buffer(5) node_row_count
//   buffer(6) rows_per_chunk
//   buffer(7) n_features
// -----------------------------------------------------------------

kernel void histogram_count_accumulate(
    device const uint8_t*  binned_u8      [[buffer(0)]],
    device const uint16_t* binned_u16     [[buffer(1)]],
    device const uint*     row_indices    [[buffer(2)]],
    device atomic_uint*    counts_out     [[buffer(3)]],
    constant uint&         n_rows_total   [[buffer(4)]],
    constant uint&         node_row_count [[buffer(5)]],
    constant uint&         rows_per_chunk [[buffer(6)]],
    constant uint&         n_features     [[buffer(7)]],
    uint3                  thread_in_tg3  [[thread_position_in_threadgroup]],
    uint3                  tg_id3         [[threadgroup_position_in_grid]]
) {
    const uint tid     = thread_in_tg3.x;
    const uint feature = tg_id3.x;
    const uint chunk   = tg_id3.y;

    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // Threadgroup-local uint count (no float precision issues).
    // Array size matches MAX_BIN_COUNT_TILED (same 1024-entry cap as the
    // tiled scatter kernel); caller enforces bin_count ≤ MAX_BIN_COUNT_TILED.
    threadgroup atomic_uint local_counts[MAX_BIN_COUNT_TILED];
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_TILED) {
        atomic_store_explicit(&local_counts[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = chunk_start + tid; i < chunk_end; i += THREADS_TILED) {
        const uint row = row_indices[i];
        const uint bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
        atomic_fetch_add_explicit(&local_counts[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One global atomic_uint per (feature, bin) per chunk — low contention.
    const uint out_base = feature * EFFECTIVE_BIN_COUNT;
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_TILED) {
        const uint c = atomic_load_explicit(&local_counts[b], memory_order_relaxed);
        if (c > 0u) {
            atomic_fetch_add_explicit(&counts_out[out_base + b], c, memory_order_relaxed);
        }
    }
}

kernel void histogram_reduce(
    device const float2* scratch      [[buffer(0)]],
    device float*        grad_out     [[buffer(1)]],
    device float*        hess_out     [[buffer(2)]],
    constant uint&       n_chunks     [[buffer(3)]],
    constant uint&       n_features   [[buffer(4)]],
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

    const uint out_index = feature * EFFECTIVE_BIN_COUNT + bin;
    grad_out[out_index] = accum.x;
    hess_out[out_index] = accum.y;
}
