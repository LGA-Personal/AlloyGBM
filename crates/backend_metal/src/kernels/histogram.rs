//! Histogram kernel — Rust-side orchestration.
//!
//! `HISTOGRAM_SHADER_SOURCE` embeds the MSL source; `dispatch_histograms`
//! wraps buffer allocation, command-buffer encoding, commit + wait, and
//! readback into a CPU-owned `HistogramBundle`.
//!
//! Pipeline compilation (per-call in S1.4; cached via `MTLBinaryArchive`
//! in S1.5) is delegated to `crate::pipelines::build_histogram_pipelines`.

use alloygbm_core::{
    BinStorage, BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, NodeSlice,
};
use alloygbm_engine::{EngineError, EngineResult};

#[cfg(target_os = "macos")]
use crate::buffers::BufferCache;
#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::histogram_residency::HistogramResidencyPool;
#[cfg(target_os = "macos")]
use crate::pipelines::HistogramPipelineCache;
#[cfg(target_os = "macos")]
use crate::profile;
#[cfg(target_os = "macos")]
use crate::residency::ResidencyPool;
#[cfg(target_os = "macos")]
use crate::row_index_residency::RowIndexResidencyPool;

/// Embedded MSL source for the two-pass histogram build.
///
/// Exposes two entry points: `histogram_build_scatter` (per-threadgroup
/// scatter into a chunked device-memory scratch buffer) and
/// `histogram_reduce` (cross-chunk ascending reduce into the final
/// `HistogramBundle`). See `shaders/histogram.metal` for the full
/// specification.
pub const HISTOGRAM_SHADER_SOURCE: &str = include_str!("../shaders/histogram.metal");

/// Entry-point name for pass 1 (narrow path, 1 simdgroup / 32 threads).
pub const KERNEL_NAME_SCATTER: &str = "histogram_build_scatter";

/// Entry-point name for the wide pass 1 (4 simdgroups / 128 threads,
/// per-simdgroup private histograms; D-021). Valid only when
/// `bin_count <= MAX_BIN_COUNT_WIDE`.
pub const KERNEL_NAME_SCATTER_WIDE: &str = "histogram_build_scatter_wide";

/// Entry-point name for pass 2.
pub const KERNEL_NAME_REDUCE: &str = "histogram_reduce";

/// Upper bound on the number of bins that fit inside the narrow
/// kernel's threadgroup-memory `local_hist[MAX_BIN_COUNT]` array. Must
/// match the `MAX_BIN_COUNT` constant in `shaders/histogram.metal`.
pub const MAX_BIN_COUNT: u32 = 4096;

/// Upper bound on bin count for the wide kernel. Four private
/// histograms of this size fit in 32 KB of threadgroup memory
/// (4 * MAX_BIN_COUNT_WIDE * 8 bytes). Must match
/// `MAX_BIN_COUNT_WIDE` in `shaders/histogram.metal`.
pub const MAX_BIN_COUNT_WIDE: u32 = 1024;

/// Threads per threadgroup for the wide kernel (4 simdgroups * 32).
pub const THREADS_PER_TG_WIDE: usize = 128;

/// Default chunk size (rows per threadgroup) for pass 1. Small enough
/// that the per-chunk scratch stays modest for large nodes; large
/// enough that per-chunk fixed overhead amortizes. Tuning is deferred
/// to the benchmark phase (S1.14) — this value was chosen to match the
/// CPU backend's `SMALL_TILE_WORKLOAD_THRESHOLD` order of magnitude.
pub const ROWS_PER_CHUNK_DEFAULT: u32 = 8_192;

// -------- macOS-only dispatch path ---------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn dispatch_histograms(
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
    histogram_residency: &HistogramResidencyPool,
    row_index_pool: &RowIndexResidencyPool,
    residency: &ResidencyPool,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    feature_tiles: &[FeatureTile],
) -> EngineResult<HistogramBundle> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    // --- Contract checks (mirror CpuBackend) ---
    if gradients.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "gradients length {} does not match row_count {}",
            gradients.len(),
            binned_matrix.row_count
        )));
    }
    if feature_tiles.is_empty() {
        return Err(EngineError::ContractViolation(
            "feature_tiles cannot be empty".to_string(),
        ));
    }
    node.validate_bounds(binned_matrix.row_count)?;

    let p_setup = profile::ScopedProbe::new(&profile::BH_BUFFER_SETUP);

    let n_rows_total = binned_matrix.row_count as u32;
    let node_row_count = node.row_count() as u32;
    if node_row_count == 0 {
        return Err(EngineError::ContractViolation(
            "node row_indices cannot be empty".to_string(),
        ));
    }
    let bin_count = binned_matrix.max_bin as u32 + 1;
    if bin_count == 0 || bin_count > MAX_BIN_COUNT {
        return Err(EngineError::BackendUnavailable(format!(
            "Metal backend requires bin_count in 1..={MAX_BIN_COUNT}, got {bin_count}"
        )));
    }
    for tile in feature_tiles {
        if (tile.end_feature as usize) > binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature tile end {} exceeds feature_count {}",
                tile.end_feature, binned_matrix.feature_count
            )));
        }
    }

    let use_u16 = binned_matrix.is_wide_bins();
    let rows_per_chunk = ROWS_PER_CHUNK_DEFAULT;
    let n_chunks = node_row_count.div_ceil(rows_per_chunk);

    let selected_features: Vec<u32> = feature_tiles
        .iter()
        .flat_map(|t| t.start_feature..t.end_feature)
        .collect();
    let total_selected = selected_features.len() as u32;

    // --- Pipelines (cached across dispatches per S1.5) ---
    let pipelines = pipeline_cache
        .get_or_build(bin_count, use_u16)
        .map_err(EngineError::BackendUnavailable)?;

    let device = &metal_device.device;
    let options = MTLResourceOptions::StorageModeShared;

    // --- Buffer allocation ---
    //
    // The column-major binned matrix is bound once and re-offset per
    // tile (via `setBuffer:offset:atIndex:`). The kernel's
    // `USE_U16_BINS` function constant dead-code-eliminates one of the
    // two pointer branches; we still bind a 1-byte dummy into the
    // unused slot because Metal requires a valid buffer for every
    // argument slot referenced in the kernel signature.
    //
    // `buffer_cache` persists buffers across dispatches within a fit:
    //   * The binned matrix is keyed by `(ptr, len_bytes, is_wide)` and
    //     re-used zero-copy across the ~63 `build_histograms` calls a
    //     depth-6 tree makes (and across all trees within a fit).
    //   * The gradients slot is allocation-reused; contents are copied
    //     fresh every call because the engine rewrites the gradient
    //     buffer once per boosting round via
    //     `objective.compute_gradients_into`.
    //   * The row-indices slot is allocation-reused; contents change
    //     per node (every call passes a different row subset).
    let binned_buffer = match &binned_matrix.bins_col_adaptive {
        BinStorage::U8(v) => buffer_cache.get_or_upload_binned(device, v, false)?,
        BinStorage::U16(v) => buffer_cache.get_or_upload_binned(device, v, true)?,
    };
    let dummy_buffer = device
        .newBufferWithLength_options(1, options)
        .ok_or_else(|| {
            EngineError::BackendUnavailable("could not allocate dummy buffer".to_string())
        })?;

    // `GradientPair` is not `#[repr(C)]`, so explicitly repack to an
    // `[f32; 2]` layout that matches MSL `float2` (8 bytes, 4-byte
    // aligned per component). The repack itself is unavoidable; the
    // buffer allocation behind it is cache-backed.
    let gradients_raw: Vec<[f32; 2]> = gradients.iter().map(|g| [g.grad, g.hess]).collect();
    let gradients_buffer = buffer_cache.write_gradients(device, &gradients_raw)?;
    // S3.7e.2b — accept both row-index storage variants. Cpu goes
    // through the buffer_cache upload path (as before); Gpu clones the
    // already-resident pool buffer (zero-copy). The buffer length may
    // be larger than `node_row_count * 4` for Gpu variants (the
    // partition kernel's output buffers are sized to the parent's row
    // count); only the `node_row_count` prefix is read by the kernel,
    // which is exactly what `validate_bounds` + `node_row_count` above
    // pins down.
    let row_indices_buffer = match &node.rows {
        alloygbm_core::RowIndexStorage::Cpu(v) => {
            buffer_cache.write_row_indices(device, v.as_slice())?
        }
        alloygbm_core::RowIndexStorage::Gpu { handle, .. } => {
            row_index_pool
                .get(*handle)
                .ok_or_else(|| {
                    EngineError::BackendUnavailable(format!(
                        "build_histograms: row-index handle {} not in residency pool",
                        handle.0
                    ))
                })?
                .buffer
        }
    };

    let pair_bytes = std::mem::size_of::<[f32; 2]>();
    let f32_bytes = std::mem::size_of::<f32>();
    // SoA output buffers (D-019): two parallel planes sized
    // [n_features × bin_count]. The scatter pass's internal scratch
    // stays AoS (`float2`) because the per-bin single-writer
    // discipline benefits from `(grad, hess)` coresidency in
    // threadgroup memory; only the reduce pass's final write splits
    // the planes.
    //
    // S3.7c.2: mint a pool entry for this bundle so the kernel's
    // reduce pass writes directly into GPU-resident pool buffers.
    // The pool also owns the counts buffer (written on the CPU via
    // `write_counts` below — counts are computed CPU-side per D-008
    // and memcpied into pool storage so downstream GPU consumers —
    // subtract, residency-warm split — can read them without a
    // round-trip).
    let pool_handle = histogram_residency.mint(device, residency, &selected_features, bin_count)?;
    let pool_entry = histogram_residency.get(pool_handle).ok_or_else(|| {
        EngineError::BackendUnavailable(
            "histogram residency pool: freshly minted handle lost".to_string(),
        )
    })?;
    let grad_out_buffer = pool_entry.grad.clone();
    let hess_out_buffer = pool_entry.hess.clone();

    drop(p_setup);
    let p_dispatch = profile::ScopedProbe::new(&profile::BH_GPU_DISPATCH);

    // --- Encode the command buffer ---
    let p_encode_cb = profile::ScopedProbe::new(&profile::BH_ENCODE);
    let command_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| EngineError::BackendUnavailable("no command buffer".to_string()))?;
    drop(p_encode_cb);

    let bin_sz = if use_u16 { 2usize } else { 1usize };
    let mut cumulative_features: u32 = 0;

    // Keep scratch buffers alive until after `waitUntilCompleted` below.
    let mut scratch_keepalive: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
        Vec::with_capacity(feature_tiles.len());

    for tile in feature_tiles {
        let tile_n_features = tile.end_feature - tile.start_feature;
        let tile_scratch_elems = n_chunks as usize * tile_n_features as usize * bin_count as usize;
        let tile_scratch_bytes = tile_scratch_elems * pair_bytes;
        let scratch_buffer = {
            let _p = profile::ScopedProbe::new(&profile::BH_SCRATCH_ALLOC);
            device
                .newBufferWithLength_options(tile_scratch_bytes.max(1), options)
                .ok_or_else(|| {
                    EngineError::BackendUnavailable("could not allocate scratch buffer".to_string())
                })?
        };

        let p_encode = profile::ScopedProbe::new(&profile::BH_ENCODE);
        let binned_offset = (tile.start_feature as usize) * (n_rows_total as usize) * bin_sz;
        let output_offset = (cumulative_features as usize) * (bin_count as usize) * f32_bytes;

        // --------- Pass 1: scatter ---------
        //
        // D-021 wide path: when `bin_count <= MAX_BIN_COUNT_WIDE` the
        // pipeline cache has compiled `histogram_build_scatter_wide`
        // (4 simdgroups / 128 threads) and we dispatch it instead of
        // the narrow kernel. Grid dimensions are unchanged — only the
        // `threads_per_tg.width` differs.
        let (scatter_pipeline, scatter_threads_per_tg): (
            &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            usize,
        ) = if let Some(wide) = &pipelines.scatter_wide {
            (wide, THREADS_PER_TG_WIDE)
        } else {
            (&pipelines.scatter, 32usize)
        };
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("no compute command encoder".to_string())
        })?;
        encoder.setComputePipelineState(scatter_pipeline);
        // SAFETY: every pointer below either refers to a live stack slot
        // whose bytes `setBytes` copies synchronously, or a
        // `MTLBuffer` that outlives the encoder through `Retained`.
        // Buffer-argument indices (0..=8) match the MSL kernel header
        // in `shaders/histogram.metal`.
        unsafe {
            if use_u16 {
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 1);
            } else {
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 1);
            }
            encoder.setBuffer_offset_atIndex(Some(&gradients_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&scratch_buffer), 0, 4);

            let n_rows_total_cell: u32 = n_rows_total;
            let node_row_count_cell: u32 = node_row_count;
            let rows_per_chunk_cell: u32 = rows_per_chunk;
            let tile_n_features_cell: u32 = tile_n_features;
            set_u32_bytes(&encoder, &n_rows_total_cell, 5);
            set_u32_bytes(&encoder, &node_row_count_cell, 6);
            set_u32_bytes(&encoder, &rows_per_chunk_cell, 7);
            set_u32_bytes(&encoder, &tile_n_features_cell, 8);

            let threadgroups = MTLSize {
                width: tile_n_features as usize,
                height: n_chunks as usize,
                depth: 1,
            };
            let threads_per_tg = MTLSize {
                width: scatter_threads_per_tg,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        }
        encoder.endEncoding();

        // --------- Pass 2: reduce ---------
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("no compute command encoder".to_string())
        })?;
        encoder.setComputePipelineState(&pipelines.reduce);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&scratch_buffer), 0, 0);
            // SoA outputs (D-019): grad plane at buffer(1),
            // hess plane at buffer(2). Each tile writes into its own
            // `[tile_n_features × bin_count]` slice via a shared
            // `output_offset` (measured in f32s) across both planes.
            encoder.setBuffer_offset_atIndex(Some(&grad_out_buffer), output_offset, 1);
            encoder.setBuffer_offset_atIndex(Some(&hess_out_buffer), output_offset, 2);

            let n_chunks_cell: u32 = n_chunks;
            let tile_n_features_cell: u32 = tile_n_features;
            set_u32_bytes(&encoder, &n_chunks_cell, 3);
            set_u32_bytes(&encoder, &tile_n_features_cell, 4);

            let bin_tg_count = bin_count.div_ceil(32) as usize;
            let threadgroups = MTLSize {
                width: tile_n_features as usize,
                height: bin_tg_count,
                depth: 1,
            };
            let threads_per_tg = MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        }
        encoder.endEncoding();

        scratch_keepalive.push(scratch_buffer);
        cumulative_features += tile_n_features;
        drop(p_encode);
    }

    {
        let _p = profile::ScopedProbe::new(&profile::BH_COMMIT_WAIT);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }
    drop(p_dispatch);

    // --- CPU-computed counts → GPU-resident pool buffer (D-008) ---
    //
    // The GPU kernel emits only `(grad_sum, hess_sum)` via the reduce
    // pass; counts are computed CPU-side (single u8/u16 bin-read +
    // uint increment per row — trivially deterministic) and memcpied
    // into the pool's counts buffer via `write_counts`. Downstream
    // GPU consumers (`subtract`, any future residency-warm reducer)
    // read counts directly from the pool buffer without a host
    // round-trip.
    let counts_total = (total_selected as usize) * (bin_count as usize);
    let mut counts_flat = vec![0u32; counts_total];
    // CPU-side count accumulation wants a &[u32]. Cpu variants loan
    // their existing slice; Gpu variants read the row-index prefix
    // out of the pool buffer as `[u32; node_row_count]`. The readback
    // is a single unavoidable host-visible copy per call — a future
    // GPU counts kernel (deferred) would remove it.
    let gpu_rows_cache: Vec<u32>;
    let row_indices_slice: &[u32] = match &node.rows {
        alloygbm_core::RowIndexStorage::Cpu(v) => v.as_slice(),
        alloygbm_core::RowIndexStorage::Gpu { handle, row_count } => {
            let _p = profile::ScopedProbe::new(&profile::BH_ROW_READBACK);
            let entry = row_index_pool.get(*handle).ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "build_histograms: row-index handle {} not in residency pool (count path)",
                    handle.0
                ))
            })?;
            let ptr = entry.buffer.contents().as_ptr() as *const u32;
            // SAFETY: shared-mode MTLBuffer contents() yields a live
            // CPU-visible pointer; the first `row_count * 4` bytes are
            // the valid prefix (see RowIndexEntry::row_count docs).
            let slice = unsafe { std::slice::from_raw_parts(ptr, *row_count as usize) };
            gpu_rows_cache = slice.to_vec();
            gpu_rows_cache.as_slice()
        }
    };
    {
        let _p = profile::ScopedProbe::new(&profile::BH_COUNT_ACCUMULATE);
        for (local_f, &feature_index) in selected_features.iter().enumerate() {
            let base = local_f * bin_count as usize;
            accumulate_counts(
                binned_matrix,
                row_indices_slice,
                feature_index,
                &mut counts_flat[base..base + bin_count as usize],
            );
        }
        histogram_residency.write_counts(pool_handle, &counts_flat)?;
    }

    // Keep scratch alive until after readback — see macro tricks below.
    drop(scratch_keepalive);
    // Keep `pool_entry`'s cloned buffer handles alive until here; the
    // real buffers stay live via the pool itself (`pool_handle` is
    // still registered). Drop locally to avoid a redundant refcount.
    drop(pool_entry);
    drop(grad_out_buffer);
    drop(hess_out_buffer);

    Ok(HistogramBundle::from_gpu(
        node.node_id,
        pool_handle,
        total_selected,
        bin_count,
    ))
}

// -------- Helpers (macOS only) -------------------------------------

/// SAFETY: caller holds `value` live for the duration of the
/// `setBytes` call; Metal copies the bytes synchronously.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
unsafe fn set_u32_bytes(
    encoder: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    value: &u32,
    index: usize,
) {
    use objc2_metal::MTLComputeCommandEncoder;
    use std::ffi::c_void;
    use std::ptr::NonNull;
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(value as *const u32 as *mut c_void),
            std::mem::size_of::<u32>(),
            index,
        );
    }
}

/// CPU-side count accumulation — see D-008. Counts are inherently
/// deterministic by integer arithmetic, so this never affects the
/// bit-exactness contract with `CpuBackend`.
#[cfg(target_os = "macos")]
fn accumulate_counts(
    binned_matrix: &BinnedMatrix,
    row_indices: &[u32],
    feature: u32,
    counts: &mut [u32],
) {
    let row_count = binned_matrix.row_count;
    let feat_base = (feature as usize) * row_count;
    let bin_count = counts.len();
    for &row in row_indices {
        let bin = binned_matrix.col_bin(feat_base + row as usize) as usize;
        if bin < bin_count {
            counts[bin] += 1;
        }
    }
}
