# Stage 5 — GPU Histogram Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 32× simd_shuffle serialization in the Stage 1 scatter kernel and the GPU count_accumulate CPU post-step with GPU-native equivalents, cutting GPU+CPU histogram time from ~5800ms to ≤300ms for the `metal_friendly_large_icb` benchmark (1M×100, d=8, 5 rounds).

**Architecture:**
- Drop the Stage 4b ICB override for histogram build — it uses a naive 1-atomic-per-row kernel with no subtraction trick. Revert `try_build_tree_level_wise` to return `Ok(None)`, falling back to Stage 4a's per-level `build_histograms_batch` path.
- Replace `histogram_build_scatter_wide` (32× simd_shuffle serialization, ~4626ms across 40 dispatches) with `histogram_tg_atomic_scatter`, which uses `threadgroup atomic_float` accumulation. Metal GPU Family 7+ (all M-series from M1) supports `atomic<float>` in threadgroup address space. Apple M4 is Family 9, fully supported.
- Add `histogram_count_accumulate` GPU kernel that atomically counts per-(feature,bin) row frequencies, replacing the sequential CPU `accumulate_counts` loop (1137ms across 640 calls).
- Stage 4a's histogram subtraction trick is preserved unchanged — the non-ICB path already builds only smaller-child histograms, halving work vs ICB.

**Tech Stack:**
- MSL (Metal Shading Language 3.x): `threadgroup atomic_float`, `atomic_uint`
- Rust + objc2/objc2-metal: pipeline cache, dispatch encoding, buffer management
- Existing scratch buffer + reduce pass unchanged (tg_atomic output matches existing scratch layout)

**Kill-criterion target:** `metal_friendly_large_icb` benchmark (1M×100, d=8, bins=255, 5 rounds, Apple M4) → GPU ratio ≥1.0× vs CPU.

---

## Root Cause Analysis

Stage 4a profiling (STATUS.md, 2026-04-28):

| Site | calls | total_ms | bottleneck |
|---|---|---|---|
| `build_histograms_batch.commit_wait` | 40 | 4626 | **32× shuffle overhead** |
| `count_accumulate` (CPU post-step) | 640 | 1137 | **sequential CPU bin-count** |
| `find_best_splits_batch.commit_wait` | 40 | 45 | fine |
| `subtract_histogram_bundle_batch` | 40 | 69 | fine |
| `apply_split` (CPU) | 1275 | 580 | secondary (fix after kill-criterion) |

**Why `histogram_build_scatter_wide` is slow:**
The wide scatter kernel uses `simd_shuffle` to broadcast each row's (bin, grad, hess) to all 32 lanes in the simdgroup, then only 1/32 lanes writes (the one that "owns" that bin by modulo). Each outer row iteration runs 32 inner shuffle iterations, with 31/32 idle per iteration — a 32× throughput penalty. For 1M rows × 100 features across 5 rounds with subtraction, this accumulates to ~3.2B simd_shuffle ops.

**Fix:** `threadgroup atomic_float` accumulation. Each of the 128 threads independently reads its rows and atomically adds to the threadgroup histogram. No inner loop, no shuffles. Threadgroup atomics are L1-speed (no chip-wide contention). With 128 threads × 8192 rows/chunk / 256 bins ≈ 32-way contention per bin — fast on Apple Silicon's hardware atomic units.

**Why `count_accumulate` is slow:**
After each GPU histogram dispatch, the CPU serially counts bin frequencies for all (node, feature) pairs. This is 640 calls × 1.78ms = 1137ms of sequential memory-bound CPU work, all blocking the next GPU submission.

**Fix:** `histogram_count_accumulate` GPU kernel, encoded directly into the same command buffer as the scatter pass. Grid = `(n_features, n_chunks, 1)`. Each (feature, chunk) threadgroup uses `threadgroup atomic_uint` to count bins for its row chunk, then atomically adds to a global count buffer. Total global atomics: n_chunks × bin_count × n_features ≈ 122 × 256 × 100 = 3.1M uints vs. the current CPU's sequential loop.

---

## File Structure

| File | Change |
|---|---|
| `crates/backend_metal/src/shaders/histogram.metal` | Add `histogram_tg_atomic_scatter` + `histogram_count_accumulate` kernels |
| `crates/backend_metal/src/kernels/histogram.rs` | Add `KERNEL_NAME_TG_ATOMIC_SCATTER`, `KERNEL_NAME_COUNT_ACCUMULATE` constants; update dispatch to use new kernels and GPU counts |
| `crates/backend_metal/src/pipelines.rs` | Add `tg_atomic_scatter: Option<Retained<…>>` + `count_accumulate: Retained<…>` to `HistogramPipelineEntry`; compile in `get_or_build` |
| `crates/backend_metal/src/lib.rs` | `try_build_tree_level_wise` → return `Ok(None)` (disable ICB, fall through to Stage 4a path with new kernels) |
| `benchmarks/metal_histogram.py` | Add `scenario_stage5` for the Stage 5 benchmark |
| `docs/metal-backend/STATUS.md` | Update stage/checklist |
| `docs/metal-backend/SESSIONS.md` | Append session entry |

---

## Task 1: New MSL kernels in `histogram.metal`

**Files:**
- Modify: `crates/backend_metal/src/shaders/histogram.metal`

- [ ] **Step 1: Add `histogram_tg_atomic_scatter` kernel after the existing `histogram_build_scatter_wide` kernel**

Append to `histogram.metal`:

```metal
// -----------------------------------------------------------------
// Threadgroup-atomic scatter — Stage 5.
//
// Replaces the simd_shuffle serialisation (32× inner loop) with
// threadgroup-local atomic_float accumulation. Each threadgroup
// handles one (feature, chunk) pair, exactly as the wide kernel,
// and writes to the SAME scratch layout — the existing
// `histogram_reduce` pass is unchanged.
//
// Requires Metal GPU Family 7+ (A15/M1 and later) for
// `atomic<float>` in threadgroup address space. MetalBackend::new()
// already gates on `apple7`, so no additional check is needed here.
//
// Non-determinism note: `memory_order_relaxed` threadgroup float
// atomics do not guarantee accumulation order, so results may
// differ by a few ULP from the simd_shuffle kernel. This is
// acceptable — parity tests already use atol=0.05–0.1.
// -----------------------------------------------------------------

constant uint THREADS_PER_TG_TGA = 128u;

kernel void histogram_tg_atomic_scatter(
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
    const uint tid     = thread_in_tg3.x;
    const uint feature = tg_id3.x;
    const uint chunk   = tg_id3.y;

    const uint chunk_start = chunk * rows_per_chunk;
    const uint chunk_end   = min(chunk_start + rows_per_chunk, node_row_count);

    // Threadgroup-local histogram using atomic<float>.
    // Physical size is MAX_BIN_COUNT; only [:EFFECTIVE_BIN_COUNT] is used.
    threadgroup atomic<float> local_grad[MAX_BIN_COUNT];
    threadgroup atomic<float> local_hess[MAX_BIN_COUNT];

    // Zero-initialise collaboratively across all threads.
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_PER_TG_TGA) {
        atomic_store_explicit(&local_grad[b], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&local_hess[b], 0.0f, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread strides over its assigned rows and accumulates.
    // No inner serialisation loop — each thread atomically adds
    // to whichever bin its row lands in.
    for (uint i = chunk_start + tid; i < chunk_end; i += THREADS_PER_TG_TGA) {
        const uint row = row_indices[i];
        const uint bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
        const float2 gh = gradients[row];
        atomic_fetch_add_explicit(&local_grad[bin], gh.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hess[bin], gh.y, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Publish to device scratch — identical layout to histogram_build_scatter,
    // so `histogram_reduce` is unchanged.
    const uint scratch_base = (chunk * n_features + feature) * EFFECTIVE_BIN_COUNT;
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_PER_TG_TGA) {
        scratch[scratch_base + b] = float2(
            atomic_load_explicit(&local_grad[b], memory_order_relaxed),
            atomic_load_explicit(&local_hess[b], memory_order_relaxed)
        );
    }
}

// -----------------------------------------------------------------
// GPU count accumulation — Stage 5.
//
// Replaces the CPU-side `accumulate_counts` loop (1137ms across
// 640 calls for the large benchmark). One threadgroup per
// (feature, chunk); uses threadgroup atomic_uint to count
// per-bin row frequencies without float precision issues, then
// adds to global count buffer. Total global atomics:
// n_chunks × bin_count × n_features ≈ 3.1M for the large benchmark.
//
// Buffer layout:
//   buffer(0) binned_u8  — same as scatter
//   buffer(1) binned_u16 — same as scatter
//   buffer(2) row_indices — same as scatter
//   buffer(3) atomic_uint* counts_out — [n_features × BIN_COUNT]
//   buffer(4) n_rows_total — same
//   buffer(5) node_row_count — same
//   buffer(6) rows_per_chunk — same
//   buffer(7) n_features — same
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

    // Threadgroup-local count histogram (uint — no float precision issues).
    threadgroup atomic_uint local_counts[MAX_BIN_COUNT];
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_PER_TG_TGA) {
        atomic_store_explicit(&local_counts[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = chunk_start + tid; i < chunk_end; i += THREADS_PER_TG_TGA) {
        const uint row = row_indices[i];
        const uint bin = load_bin(binned_u8, binned_u16, row, feature, n_rows_total);
        atomic_fetch_add_explicit(&local_counts[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One global atomic_uint add per (feature, bin) per chunk.
    // Global count buffer layout: [feature * BIN_COUNT + bin]
    const uint out_base = feature * EFFECTIVE_BIN_COUNT;
    for (uint b = tid; b < EFFECTIVE_BIN_COUNT; b += THREADS_PER_TG_TGA) {
        uint c = atomic_load_explicit(&local_counts[b], memory_order_relaxed);
        if (c > 0u) {
            atomic_fetch_add_explicit(&counts_out[out_base + b], c, memory_order_relaxed);
        }
    }
}
```

- [ ] **Step 2: Verify the file compiles mentally — all functions use `EFFECTIVE_BIN_COUNT`, `MAX_BIN_COUNT`, `load_bin` which are already defined at the top of `histogram.metal`. `THREADS_PER_TG_TGA = 128` is new and must not conflict with `THREADS_WIDE = 128` (it's a different constant, fine). Check that `atomic<float>` alias works — in MSL the type is spelled `atomic<float>` or the typedef `atomic_float`; use `atomic<float>` for clarity.**

---

## Task 2: Constants in `histogram.rs`

**Files:**
- Modify: `crates/backend_metal/src/kernels/histogram.rs`

- [ ] **Step 1: Add kernel name constants after existing ones**

After line `pub const KERNEL_NAME_REDUCE: &str = "histogram_reduce";`, add:

```rust
/// Entry-point name for the threadgroup-atomic scatter kernel (Stage 5).
/// Requires Metal GPU Family 7+ (apple7 capability flag).
pub const KERNEL_NAME_TG_ATOMIC_SCATTER: &str = "histogram_tg_atomic_scatter";

/// Entry-point name for the GPU count accumulation kernel (Stage 5).
/// Requires Metal GPU Family 7+ (apple7 capability flag).
pub const KERNEL_NAME_COUNT_ACCUMULATE: &str = "histogram_count_accumulate";

/// Threads per threadgroup for the threadgroup-atomic scatter and
/// count kernels (Stage 5). Must match `THREADS_PER_TG_TGA` in
/// `shaders/histogram.metal`.
pub const THREADS_PER_TG_TGA: usize = 128;
```

- [ ] **Step 2: Run `cargo check --workspace`** — expects no errors at this step (new constants don't break anything yet).

---

## Task 3: Update `HistogramPipelineEntry` and `HistogramPipelineCache`

**Files:**
- Modify: `crates/backend_metal/src/pipelines.rs`

- [ ] **Step 1: Add fields to `HistogramPipelineEntry`**

The struct currently has `scatter`, `scatter_wide: Option<…>`, and `reduce`. Add:

```rust
/// Threadgroup-atomic scatter kernel (Stage 5; apple7+).
/// `None` when the device is below Metal GPU Family 7.
pub tg_atomic_scatter: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
/// GPU count accumulation kernel (Stage 5; apple7+).
/// `None` when the device is below Metal GPU Family 7.
pub count_accumulate: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
```

- [ ] **Step 2: In `HistogramPipelineCache::get_or_build`, compile the two new pipelines after the existing `scatter_wide` block**

Import new kernel name constants at the top of the function:

```rust
use crate::kernels::histogram::{KERNEL_NAME_TG_ATOMIC_SCATTER, KERNEL_NAME_COUNT_ACCUMULATE};
```

Then after the `scatter_wide` block, add:

```rust
// Stage 5: threadgroup-atomic scatter + count kernels.
// Compile for apple7+ (all M-series); on older devices leave None.
let (tg_atomic_scatter, count_accumulate) = if capabilities.apple7 {
    let tga_name = NSString::from_str(KERNEL_NAME_TG_ATOMIC_SCATTER);
    let cnt_name = NSString::from_str(KERNEL_NAME_COUNT_ACCUMULATE);

    let tga_fn = library
        .newFunctionWithName_constantValues_error(&tga_name, &constants)
        .map_err(|e| format!("could not specialize `{KERNEL_NAME_TG_ATOMIC_SCATTER}`: {e}"))?;
    let cnt_fn = library
        .newFunctionWithName_constantValues_error(&cnt_name, &constants)
        .map_err(|e| format!("could not specialize `{KERNEL_NAME_COUNT_ACCUMULATE}`: {e}"))?;

    let make_pso = |f: Retained<ProtocolObject<dyn MTLFunction>>, label: &str| -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&f));
        let pso = self.device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|e| format!("{label} pipeline creation failed: {e}"))?;
        Ok(pso)
    };

    let tga_pso = make_pso(tga_fn, "tg_atomic_scatter")?;
    let cnt_pso = make_pso(cnt_fn, "count_accumulate")?;
    (Some(tga_pso), Some(cnt_pso))
} else {
    (None, None)
};
```

- [ ] **Step 3: Add fields to the `HistogramPipelineEntry` literal at the end of `get_or_build`**

In the `Ok(HistogramPipelineEntry { scatter, scatter_wide, reduce })` return, add:

```rust
Ok(HistogramPipelineEntry {
    scatter,
    scatter_wide,
    reduce,
    tg_atomic_scatter,
    count_accumulate,
})
```

- [ ] **Step 4: Run `cargo check --workspace`** to confirm struct fields and `get_or_build` compile cleanly.

---

## Task 4: Update `histogram.rs` dispatch to use new kernels

**Files:**
- Modify: `crates/backend_metal/src/kernels/histogram.rs`

- [ ] **Step 1: Update scatter kernel selection in `encode_one_histogram_request`**

The current selection logic:
```rust
let (scatter_pipeline, scatter_threads_per_tg): (_, usize) =
    if let Some(wide) = &pipelines.scatter_wide {
        (wide, THREADS_PER_TG_WIDE)
    } else {
        (&pipelines.scatter, 32usize)
    };
```

Replace with (prefer tg_atomic > scatter_wide > scatter_narrow):
```rust
let (scatter_pipeline, scatter_threads_per_tg): (
    &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    usize,
) = if let Some(tga) = &pipelines.tg_atomic_scatter {
    (tga, THREADS_PER_TG_TGA)
} else if let Some(wide) = &pipelines.scatter_wide {
    (wide, THREADS_PER_TG_WIDE)
} else {
    (&pipelines.scatter, 32usize)
};
```

Make sure to import `THREADS_PER_TG_TGA` at the top of the file (it's already in the same module, so just reference it as `super::THREADS_PER_TG_TGA` or add `use crate::kernels::histogram::THREADS_PER_TG_TGA`… actually since it's in the same file, just use the constant directly).

- [ ] **Step 2: Add GPU count accumulation encoding in `encode_one_histogram_request`**

After the Pass 2 reduce encoder block (ending with `encoder.endEncoding()`), add a Pass 3 count encoding block:

```rust
// --------- Pass 3: GPU count accumulation (Stage 5) ---------
// Replaces the CPU `accumulate_counts` post-step. Same grid as
// the scatter pass; counts are written into a freshly-allocated
// zeroed MTLBuffer that is stored on the `EncodedHistogramRequest`
// and read back after `waitUntilCompleted`.
if let Some(count_pso) = &pipelines.count_accumulate {
    let count_buf_bytes = (total_selected as usize) * (bin_count as usize) * std::mem::size_of::<u32>();
    let count_buffer = device
        .newBufferWithLength_options(count_buf_bytes.max(1), options)
        .ok_or_else(|| EngineError::BackendUnavailable("could not allocate count buffer".to_string()))?;
    // Zero-init (StorageModeShared buffers are not zero-init by Metal).
    unsafe {
        std::ptr::write_bytes(count_buffer.contents().as_ptr() as *mut u8, 0, count_buf_bytes);
    }

    let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
        EngineError::BackendUnavailable("no compute command encoder for count pass".to_string())
    })?;
    encoder.setComputePipelineState(count_pso);

    // cumulative_features tracks how far into the output slice we are
    // (reset it for count pass — iterate tiles again).
    // NOTE: We use a fresh tile loop here to avoid borrow complexity.
    let mut count_feature_offset: u32 = 0;
    for tile in feature_tiles {
        let tile_n_features = tile.end_feature - tile.start_feature;
        let binned_offset = (tile.start_feature as usize) * (n_rows_total as usize) * bin_sz;
        let count_byte_offset = (count_feature_offset as usize) * (bin_count as usize) * std::mem::size_of::<u32>();

        unsafe {
            if use_u16 {
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 1);
            } else {
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 1);
            }
            encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), count_byte_offset, 3);

            let n_rows_total_cell: u32 = n_rows_total;
            let node_row_count_cell: u32 = node_row_count;
            let rows_per_chunk_cell: u32 = rows_per_chunk;
            let tile_n_features_cell: u32 = tile_n_features;
            set_u32_bytes(&encoder, &n_rows_total_cell, 4);
            set_u32_bytes(&encoder, &node_row_count_cell, 5);
            set_u32_bytes(&encoder, &rows_per_chunk_cell, 6);
            set_u32_bytes(&encoder, &tile_n_features_cell, 7);

            let threadgroups = MTLSize { width: tile_n_features as usize, height: n_chunks as usize, depth: 1 };
            let threads_per_tg = MTLSize { width: THREADS_PER_TG_TGA, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        }
        count_feature_offset += tile_n_features;
    }
    encoder.endEncoding();
    // Store count_buffer in the encoded request for readback after commit.
    // (Added to EncodedHistogramRequest below.)
}
```

- [ ] **Step 3: Add `count_buffer` to `EncodedHistogramRequest`**

```rust
struct EncodedHistogramRequest {
    pool_handle: alloygbm_core::GpuHistogramHandle,
    bin_count: u32,
    total_selected: u32,
    selected_features: Vec<u32>,
    scratch_keepalive: Vec<…>,
    /// GPU count buffer (Stage 5): `[total_selected × bin_count]` u32 values.
    /// `None` when the count kernel is unavailable (pre-apple7 fallback).
    gpu_count_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}
```

- [ ] **Step 4: Update `finalize_one_histogram_request` to use GPU counts when available**

Before the `accumulate_counts` call:
```rust
// Stage 5: if GPU count buffer is available, read it back instead of
// running the CPU `accumulate_counts` loop.
if let Some(count_buf) = &encoded.gpu_count_buffer {
    use objc2_metal::MTLBuffer;
    let ptr = count_buf.contents().as_ptr() as *const u32;
    let gpu_counts = unsafe {
        std::slice::from_raw_parts(ptr, encoded.total_selected as usize * encoded.bin_count as usize)
    };
    // Copy GPU counts directly into `counts_flat`.
    counts_flat.copy_from_slice(gpu_counts);
    histogram_residency.write_counts(encoded.pool_handle, &counts_flat)?;
} else {
    // CPU fallback (pre-apple7 devices or when count kernel unavailable).
    let _p = profile::ScopedProbe::new(&profile::BH_COUNT_ACCUMULATE);
    for (local_f, &feature_index) in encoded.selected_features.iter().enumerate() {
        let base = local_f * encoded.bin_count as usize;
        accumulate_counts(binned_matrix, row_indices_slice, feature_index, &mut counts_flat[base..]);
    }
    histogram_residency.write_counts(encoded.pool_handle, &counts_flat)?;
}
```

- [ ] **Step 5: Run `cargo check --workspace` and `cargo test --workspace -- --test-threads=1`** to catch any compilation issues.

---

## Task 5: Disable ICB override in `lib.rs`

**Files:**
- Modify: `crates/backend_metal/src/lib.rs`

- [ ] **Step 1: At the top of `try_build_tree_level_wise`, add an early return**

```rust
fn try_build_tree_level_wise(&self, ...) -> EngineResult<Option<...>> {
    // Stage 5: ICB path disabled — Stage 4b's icb_histogram uses 1 global
    // atomic per row (100M/dispatch), with no subtraction trick. Stage 5
    // uses the new histogram_tg_atomic_scatter kernel via Stage 4a's
    // per-level batch path, which is both faster and benefits from
    // histogram subtraction. Re-enable when ICB's histogram kernel is
    // ported to the tg-atomic approach.
    return Ok(None);

    // … existing code below …
}
```

- [ ] **Step 2: Run `cargo test --workspace -- --test-threads=1`** to confirm all 244 tests still pass (ICB parity tests skip on the early return since `None` → CPU fallback; but actually the tests call `try_build_tree_level_wise` directly, not the engine path. Let me check — the parity tests call `metal.try_build_tree_level_wise(...)` directly and `unwrap().expect("ICB path should be eligible on Metal4")`. With `return Ok(None)`, `unwrap()` succeeds but `.expect()` on `None` panics. Need to handle: change the early return to `return Ok(None)` and update the parity tests to check for `Some` or `None`.)

Actually the cleaner fix: keep `try_build_tree_level_wise` returning `Ok(None)` but also update the parity tests to gracefully handle `None` (skip the test or check CPU parity differently). The ICB parity tests are already soft (they only run on Metal4), so making them also return early when ICB is disabled is correct.

Updated parity test check:
```rust
let Some((metal_stumps, _)) = metal.try_build_tree_level_wise(...).unwrap() else {
    eprintln!("ICB path disabled in Stage 5 — skipping ICB parity test");
    return;
};
```

- [ ] **Step 3: Update all 4 parity tests in `tests/icb_tree_parity.rs`** to use the `let Some(...) else { return; }` pattern.

- [ ] **Step 4: Run `cargo test --test icb_tree_parity -- --test-threads=1`** — all 4 tests should now pass (by skipping gracefully).

---

## Task 6: Benchmark + documentation

**Files:**
- Modify: `benchmarks/metal_histogram.py`
- Modify: `docs/metal-backend/STATUS.md`
- Modify: `docs/metal-backend/SESSIONS.md`

- [ ] **Step 1: Add `scenario_stage5` to `metal_histogram.py`**

```python
def scenario_stage5(args):
    """Stage 5: tg-atomic histogram + GPU counts, 1M×100, d=8, 5 rounds."""
    X, y = make_regression_dataset(n_samples=1_000_000, n_features=100)
    results = {}
    for device in ["cpu", "metal"]:
        model = GBMRegressor(
            n_estimators=5, max_depth=8, max_bins=255,
            device=device, verbose=0
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        results[device] = time.perf_counter() - t0
    ratio = results["cpu"] / results["metal"]
    print(f"stage5  cpu={results['cpu']:.2f}s  metal={results['metal']:.2f}s  ratio={ratio:.2f}x")
    return ratio
```

Add `"stage5"` to `SCENARIO_CHOICES` and call it in `main`.

- [ ] **Step 2: Run benchmark: `.venv/bin/python benchmarks/metal_histogram.py stage5`**

- [ ] **Step 3: Update `docs/metal-backend/STATUS.md`** with Stage 5 checklist and result.

- [ ] **Step 4: Append session entry to `docs/metal-backend/SESSIONS.md`.**

- [ ] **Step 5: Commit all changes with message:**
```
feat: Stage 5 — threadgroup-atomic histogram + GPU count accumulation

Replace histogram_build_scatter_wide (32× simd_shuffle serialization,
4626ms across 40 dispatches) with histogram_tg_atomic_scatter using
threadgroup atomic<float> accumulation on Metal GPU Family 7+.

Add histogram_count_accumulate GPU kernel encoded in the same command
buffer as the scatter/reduce passes, eliminating the 1137ms sequential
CPU accumulate_counts post-step.

Disable ICB override (try_build_tree_level_wise returns Ok(None)),
falling back to Stage 4a's level-wise batch path which uses histogram
subtraction and the new faster kernel.

Kill-criterion result: [fill in after benchmark]
```

---

## Expected Outcome

| Metric | Stage 4b (before) | Stage 5 (expected) |
|---|---|---|
| `build_histograms_batch.commit_wait` | 4626ms | ~200–500ms |
| `count_accumulate` (CPU) | 1137ms | ~10ms (GPU) |
| `find_best_splits_batch` | 45ms | 45ms |
| `subtract_histogram_bundle_batch` | 69ms | 69ms |
| `apply_split` (CPU) | 580ms | 580ms |
| **Total** | **~6500ms** | **~900–1200ms** |
| **vs CPU (1530ms)** | **0.24×** | **≥1.0× target** |

If `apply_split` becomes the new dominant bottleneck (580ms = ~50% of GPU+CPU time), a Stage 6 GPU partition kernel can address it.
