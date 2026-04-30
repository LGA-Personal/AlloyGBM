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

/// Entry-point name for the dynamic-threadgroup wide scatter kernel (Stage 5 occupancy fix).
/// Same algorithm as `histogram_build_scatter_wide` but with runtime-sized
/// threadgroup memory via `[[threadgroup(0)]]`.  For 256 bins this reduces
/// threadgroup allocation from 32 KB to 8 KB, enabling 4× more concurrent
/// threadgroups per shader core and better gradient-read latency hiding.
/// Caller must call `setThreadgroupMemoryLength_atIndex` before dispatch.
pub const KERNEL_NAME_SCATTER_WIDE_DYN: &str = "histogram_build_wide_dyn";

/// Entry-point name for the tiled private-register scatter kernel (Stage 5).
/// No float atomics — works on all MSL compiler versions.
/// Same buffer layout and scratch output as the wide kernel — reuses
/// `histogram_reduce` unchanged.
pub const KERNEL_NAME_TG_ATOMIC_SCATTER: &str = "histogram_build_tiled";

/// Entry-point name for the GPU count accumulation kernel (Stage 5).
/// Uses `threadgroup atomic_uint` (supported on all Metal versions).
/// Replaces the sequential CPU `accumulate_counts` post-step.
pub const KERNEL_NAME_COUNT_ACCUMULATE: &str = "histogram_count_accumulate";

/// Entry-point name for the gradient pre-gather kernel (Stage 5 bandwidth fix).
/// Writes `gathered[i] = gradients[row_indices[i]]` for i in 0..node_row_count.
/// One thread per node row; dispatched as a 1-D grid before the scatter pass.
/// Converts random gradient reads in the scatter kernel to sequential reads.
pub const KERNEL_NAME_GATHER_GRADS: &str = "histogram_gather_grads";

/// Threads per threadgroup for the tiled scatter and count kernels (Stage 5).
/// Must match `THREADS_TILED` in `shaders/histogram.metal` (= N_SIMDGROUPS_T × 32).
pub const THREADS_PER_TG_TGA: usize = 128;

/// Maximum bin count supported by the tiled scatter and count kernels.
/// Must match `MAX_BIN_COUNT_TILED` in `shaders/histogram.metal`.
/// The count kernel declares `threadgroup atomic_uint[1024]` = 4 KB;
/// the tiled scatter kernel uses only 1 KB threadgroup memory for any
/// bin count, but we cap both at 1024 for uniformity.
pub const MAX_BIN_COUNT_TGA: u32 = 1024;

/// Upper bound on the number of bins that fit inside the narrow
/// kernel's threadgroup-memory `local_hist[MAX_BIN_COUNT]` array. Must
/// match the `MAX_BIN_COUNT` constant in `shaders/histogram.metal`.
pub const MAX_BIN_COUNT: u32 = 4096;

/// Upper bound on bin count for the wide kernel. Four private
/// histograms of this size fit in 32 KB of threadgroup memory
/// (4 * MAX_BIN_COUNT_WIDE * 8 bytes). Must match
/// `MAX_BIN_COUNT_WIDE` in `shaders/histogram.metal`.
pub const MAX_BIN_COUNT_WIDE: u32 = 1024;

/// Number of simdgroups per threadgroup for the wide and wide-dyn scatter kernels.
/// Must match `SIMDGROUPS_WIDE` in `shaders/histogram.metal`.
/// Used by the dynamic kernel to compute threadgroup memory size at dispatch time:
///   `SIMDGROUPS_WIDE * bin_count * size_of::<[f32; 2]>()`
pub const SIMDGROUPS_WIDE: usize = 4;

/// Threads per threadgroup for the wide kernel (SIMDGROUPS_WIDE * 32).
pub const THREADS_PER_TG_WIDE: usize = 128;

/// Default chunk size (rows per threadgroup) for pass 1.
///
/// Smaller chunks → more threadgroups → better GPU latency-hiding (more TGs
/// to switch to while one awaits memory).  On Apple M4 the tiled kernel
/// processes `ROWS_PER_CHUNK_DEFAULT / THREADS_TILED` rows per thread, so
/// keeping chunks moderate prevents long sequential dependency chains that
/// prevent the GPU from hiding memory latency.
///
/// 8,192 rows × 8 B/row = 64 KB gradient data per TG — fits comfortably in
/// the GPU L2 cache for tile-pass reuse (Apple M4 L2 ≈ 4 MB / 10 CUs).
pub const ROWS_PER_CHUNK_DEFAULT: u32 = 8_192;

// -------- macOS-only dispatch path ---------------------------------

/// Per-request output state captured after `encode_one_histogram_request`
/// and consumed by `finalize_one_histogram_request` + the caller's
/// `HistogramBundle::from_gpu` call. The `scratch_keepalive` vec holds
/// all per-tile scratch buffers alive until after `waitUntilCompleted`.
#[cfg(target_os = "macos")]
struct EncodedHistogramRequest {
    pool_handle: alloygbm_core::GpuHistogramHandle,
    bin_count: u32,
    total_selected: u32,
    selected_features: Vec<u32>,
    /// Owned scratch buffers must outlive `waitUntilCompleted`.
    scratch_keepalive:
        Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    /// Stage 5: GPU count buffer (`[total_selected × bin_count]` u32 values).
    /// Populated by `histogram_count_accumulate` during the GPU dispatch;
    /// read back in `finalize_one_histogram_request` to replace the CPU
    /// `accumulate_counts` loop.  `None` when the count kernel is
    /// unavailable (bin_count > MAX_BIN_COUNT_TGA or pre-apple7 device).
    gpu_count_buffer: Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    >,
}

/// Encodes scatter+reduce passes for one request into `command_buffer`
/// without committing. The caller is responsible for
/// `command_buffer.commit()` + `command_buffer.waitUntilCompleted()`.
///
/// After the wait, call `finalize_one_histogram_request` to populate
/// the CPU-side counts buffer, then drop `EncodedHistogramRequest`.
#[cfg(target_os = "macos")]
#[allow(unsafe_code, clippy::too_many_arguments)]
fn encode_one_histogram_request(
    command_buffer: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
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
) -> EngineResult<EncodedHistogramRequest> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice,
        MTLResourceOptions, MTLSize,
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
    let _p_dispatch = profile::ScopedProbe::new(&profile::BH_GPU_DISPATCH);

    // Stage 5: allocate one count buffer for all tiles when the GPU count
    // kernel is available.  Metal shared-mode buffers are zero-initialised
    // by the driver, so no explicit memset is needed.
    let u32_bytes = std::mem::size_of::<u32>();
    let gpu_count_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>> =
        if pipelines.count_accumulate.is_some() {
            let count_bytes =
                (total_selected as usize) * (bin_count as usize) * u32_bytes;
            Some(
                device
                    .newBufferWithLength_options(count_bytes.max(1), options)
                    .ok_or_else(|| {
                        EngineError::BackendUnavailable(
                            "could not allocate GPU count buffer".to_string(),
                        )
                    })?,
            )
        } else {
            None
        };

    // --- Encode into the provided command buffer ---
    let bin_sz = if use_u16 { 2usize } else { 1usize };
    let mut cumulative_features: u32 = 0;

    // Keep scratch buffers alive until after `waitUntilCompleted` below.
    let mut scratch_keepalive: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
        Vec::with_capacity(feature_tiles.len());

    // --------- Pre-pass: gradient gather (Stage 5 bandwidth fix) ---------
    //
    // When `scatter_wide_dyn` is active (and tiled kernel is unavailable),
    // a one-time gather kernel converts random gradient reads (100× per level)
    // to sequential reads (1× amortised).
    // `gathered_grads[i] = gradients[row_indices[i]]` for i in 0..node_row_count.
    // The gather buffer is shared across all feature tiles; each tile's scatter
    // pass binds it in place of the original gradients buffer.
    //
    // Gather is NOT used with the tiled kernel: the tiled kernel reads
    // `gradients[row_indices[i]]` directly (random access), but its tile passes
    // reuse gradient data from GPU L2 cache, so gradient bandwidth is amortised
    // without an explicit gather pre-pass.
    let use_gather = pipelines.gather_grads.is_some()
        && pipelines.scatter_wide_dyn.is_some()
        && pipelines.tg_atomic_scatter.is_none(); // tiled takes priority; no gather needed
    let gathered_grads_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>> = if use_gather {
        let gathered_bytes = node_row_count as usize * pair_bytes;
        let buf = device
            .newBufferWithLength_options(gathered_bytes.max(1), options)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "could not allocate gathered_grads buffer".to_string(),
                )
            })?;
        // Encode gather pass: grid = (node_row_count, 1, 1), 1 thread per row.
        let gather_pso = pipelines.gather_grads.as_ref().unwrap();
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("no compute encoder for gather pass".to_string())
        })?;
        encoder.setComputePipelineState(gather_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&gradients_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&buf), 0, 2);
            // Use dispatchThreads (non-threadgroup-aligned) to avoid a
            // tail-padding guard in the kernel — every gid is a valid row.
            let threads = MTLSize {
                width: node_row_count as usize,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(threads, tg_size);
        }
        encoder.endEncoding();
        Some(buf)
    } else {
        None
    };

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
        // Priority (best → fallback):
        //   1. tg_atomic_scatter (tiled) — 1 KB threadgroup, 8 tile passes,
        //      NO simd_shuffle serialisation overhead (32× less wasted compute
        //      vs scatter/wide kernels).  Up to 32× better TG occupancy →
        //      superior GPU latency hiding.  Enabled ≤ MAX_BIN_COUNT_TGA.
        //   2. scatter_wide_dyn — 8 KB threadgroup (runtime-sized), simd_shuffle
        //      serialisation, 4 concurrent TGs/CU.  Enabled ≤ MAX_BIN_COUNT_WIDE.
        //   3. scatter_wide (static 32 KB) — single-pass, 1 TG/CU occupancy.
        //   4. scatter_narrow — 1-simdgroup fallback for any bin count.
        let (scatter_pipeline, scatter_threads_per_tg, scatter_tg_mem_bytes): (
            &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            usize,
            Option<usize>,
        ) = if let Some(tga) = &pipelines.tg_atomic_scatter {
            (tga, THREADS_PER_TG_TGA, None)
        } else if let Some(dyn_wide) = &pipelines.scatter_wide_dyn {
            // Threadgroup(0) memory = SIMDGROUPS_WIDE × bin_count × sizeof(float2).
            let tg_mem = SIMDGROUPS_WIDE * bin_count as usize * std::mem::size_of::<[f32; 2]>();
            (dyn_wide, THREADS_PER_TG_WIDE, Some(tg_mem))
        } else if let Some(wide) = &pipelines.scatter_wide {
            (wide, THREADS_PER_TG_WIDE, None)
        } else {
            (&pipelines.scatter, 32usize, None)
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
        //
        // For `histogram_build_wide_dyn` (use_gather=true): buffer(2) is the
        // pre-gathered gradient buffer (position-indexed, sequential access).
        // For all other kernels: buffer(2) is the original gradients buffer
        // (row-indexed, random access — legacy behaviour).
        let grad_buf_for_scatter = if use_gather {
            gathered_grads_buffer.as_ref().unwrap()
        } else {
            &gradients_buffer
        };
        unsafe {
            if use_u16 {
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 1);
            } else {
                encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 1);
            }
            encoder.setBuffer_offset_atIndex(Some(grad_buf_for_scatter), 0, 2);
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

            // For the dynamic-threadgroup kernel, set runtime threadgroup
            // memory length before dispatch.  Metal requires this to be
            // called before dispatchThreadgroups — passing None skips the
            // call for the static-memory kernels.
            if let Some(tg_mem) = scatter_tg_mem_bytes {
                encoder.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
            }

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

        // --------- Pass 3: GPU count accumulation (Stage 5) ---------
        //
        // Encodes the count kernel into the same command buffer as scatter
        // and reduce, so all three passes complete in a single GPU dispatch.
        // The count buffer is bound at a tile-specific byte offset so each
        // tile writes to its disjoint slice without any cross-tile atomics.
        if let (Some(count_pso), Some(count_buf)) =
            (&pipelines.count_accumulate, &gpu_count_buffer)
        {
            let count_byte_offset =
                (cumulative_features as usize) * (bin_count as usize) * u32_bytes;
            let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "no compute command encoder for count pass".to_string(),
                )
            })?;
            encoder.setComputePipelineState(count_pso);
            unsafe {
                // buffer(0/1): binned data (same u8/u16 selection as scatter)
                if use_u16 {
                    encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 1);
                } else {
                    encoder.setBuffer_offset_atIndex(Some(&binned_buffer), binned_offset, 0);
                    encoder.setBuffer_offset_atIndex(Some(&dummy_buffer), 0, 1);
                }
                // buffer(2): row_indices
                encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 2);
                // buffer(3): counts_out at tile offset
                encoder.setBuffer_offset_atIndex(Some(count_buf), count_byte_offset, 3);

                let n_rows_total_cell: u32 = n_rows_total;
                let node_row_count_cell: u32 = node_row_count;
                let rows_per_chunk_cell: u32 = rows_per_chunk;
                let tile_n_features_cell: u32 = tile_n_features;
                set_u32_bytes(&encoder, &n_rows_total_cell, 4);
                set_u32_bytes(&encoder, &node_row_count_cell, 5);
                set_u32_bytes(&encoder, &rows_per_chunk_cell, 6);
                set_u32_bytes(&encoder, &tile_n_features_cell, 7);

                let threadgroups = MTLSize {
                    width: tile_n_features as usize,
                    height: n_chunks as usize,
                    depth: 1,
                };
                let threads_per_tg = MTLSize {
                    width: THREADS_PER_TG_TGA,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            }
            encoder.endEncoding();
        }

        scratch_keepalive.push(scratch_buffer);
        cumulative_features += tile_n_features;
        drop(p_encode);
    }

    // Drop locally-cloned buffer handles; the real buffers stay live
    // via the pool itself (`pool_handle` is still registered).
    drop(pool_entry);
    drop(grad_out_buffer);
    drop(hess_out_buffer);

    Ok(EncodedHistogramRequest {
        pool_handle,
        bin_count,
        total_selected,
        selected_features,
        scratch_keepalive,
        gpu_count_buffer,
    })
}

/// CPU-side count accumulation and pool write after
/// `waitUntilCompleted`. This is the "finalize" half that was
/// previously inlined at lines 376–424 of `dispatch_histograms`.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn finalize_one_histogram_request(
    binned_matrix: &BinnedMatrix,
    row_index_pool: &RowIndexResidencyPool,
    histogram_residency: &HistogramResidencyPool,
    node: &NodeSlice,
    encoded: &EncodedHistogramRequest,
) -> EngineResult<()> {
    use objc2_metal::MTLBuffer;

    let counts_total = (encoded.total_selected as usize) * (encoded.bin_count as usize);
    let mut counts_flat = vec![0u32; counts_total];

    // Stage 5: if the GPU count kernel ran, read back its output directly
    // instead of running the serial CPU `accumulate_counts` loop.
    if let Some(count_buf) = &encoded.gpu_count_buffer {
        // SAFETY: shared-mode MTLBuffer contents() is a CPU-visible pointer
        // valid for the buffer's entire lifetime; GPU writes are complete
        // after `waitUntilCompleted`.
        let ptr = count_buf.contents().as_ptr() as *const u32;
        let gpu_counts = unsafe { std::slice::from_raw_parts(ptr, counts_total) };
        counts_flat.copy_from_slice(gpu_counts);
        histogram_residency.write_counts(encoded.pool_handle, &counts_flat)?;
        return Ok(());
    }

    // CPU fallback: sequential bin-count accumulation (pre-apple7 devices
    // or when bin_count > MAX_BIN_COUNT_TGA).
    let gpu_rows_cache: Vec<u32>;
    let row_indices_slice: &[u32] = match &node.rows {
        alloygbm_core::RowIndexStorage::Cpu(v) => v.as_slice(),
        alloygbm_core::RowIndexStorage::Gpu { handle, row_count } => {
            let _p = profile::ScopedProbe::new(&profile::BH_ROW_READBACK);
            let entry = row_index_pool.get(*handle).ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "histograms finalize: row-index handle {} not in residency pool",
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
        for (local_f, &feature_index) in encoded.selected_features.iter().enumerate() {
            let base = local_f * encoded.bin_count as usize;
            accumulate_counts(
                binned_matrix,
                row_indices_slice,
                feature_index,
                &mut counts_flat[base..base + encoded.bin_count as usize],
            );
        }
        histogram_residency.write_counts(encoded.pool_handle, &counts_flat)?;
    }
    Ok(())
}

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
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    let command_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| EngineError::BackendUnavailable("no command buffer".to_string()))?;
    let encoded = encode_one_histogram_request(
        &command_buffer,
        metal_device,
        pipeline_cache,
        buffer_cache,
        histogram_residency,
        row_index_pool,
        residency,
        binned_matrix,
        gradients,
        node,
        feature_tiles,
    )?;
    {
        let _p = profile::ScopedProbe::new(&profile::BH_COMMIT_WAIT);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }
    finalize_one_histogram_request(
        binned_matrix,
        row_index_pool,
        histogram_residency,
        node,
        &encoded,
    )?;
    drop(encoded.scratch_keepalive);
    Ok(HistogramBundle::from_gpu(
        node.node_id,
        encoded.pool_handle,
        encoded.total_selected,
        encoded.bin_count,
    ))
}

/// Batched histogram build: encodes scatter+reduce passes for every
/// request into a single Metal command buffer, commits and waits
/// once, then runs CPU-side count finalisation per request.
///
/// Determinism is preserved by construction: the wide and narrow
/// scatter kernels are unchanged, each request writes into a
/// disjoint freshly-minted pool entry, and within one command buffer
/// Metal does not reorder writes that target the same buffer.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn dispatch_histograms_batch(
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
    histogram_residency: &HistogramResidencyPool,
    row_index_pool: &RowIndexResidencyPool,
    residency: &ResidencyPool,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    requests: &[(&NodeSlice, &[FeatureTile])],
) -> EngineResult<Vec<HistogramBundle>> {
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let command_buffer = metal_device.queue.commandBuffer().ok_or_else(|| {
        EngineError::BackendUnavailable("histogram batch: no command buffer".to_string())
    })?;

    let mut encoded: Vec<(EncodedHistogramRequest, &NodeSlice)> =
        Vec::with_capacity(requests.len());
    for (node, feature_tiles) in requests.iter().copied() {
        let one = encode_one_histogram_request(
            &command_buffer,
            metal_device,
            pipeline_cache,
            buffer_cache,
            histogram_residency,
            row_index_pool,
            residency,
            binned_matrix,
            gradients,
            node,
            feature_tiles,
        )?;
        encoded.push((one, node));
    }

    {
        let _p = profile::ScopedProbe::new(&profile::BH_COMMIT_WAIT);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }

    let mut bundles: Vec<HistogramBundle> = Vec::with_capacity(encoded.len());
    for (one, node) in encoded {
        finalize_one_histogram_request(
            binned_matrix,
            row_index_pool,
            histogram_residency,
            node,
            &one,
        )?;
        bundles.push(HistogramBundle::from_gpu(
            node.node_id,
            one.pool_handle,
            one.total_selected,
            one.bin_count,
        ));
        drop(one.scratch_keepalive);
    }

    Ok(bundles)
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
