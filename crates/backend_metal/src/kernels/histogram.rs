//! Histogram kernel — Rust-side orchestration.
//!
//! `HISTOGRAM_SHADER_SOURCE` embeds the MSL source; `dispatch_histograms`
//! wraps buffer allocation, command-buffer encoding, commit + wait, and
//! readback into a CPU-owned `HistogramBundle`.
//!
//! Pipeline compilation (per-call in S1.4; cached via `MTLBinaryArchive`
//! in S1.5) is delegated to `crate::pipelines::build_histogram_pipelines`.

use alloygbm_core::{
    BinStorage, BinnedMatrix, FeatureHistogram, FeatureTile, GradientPair, HistogramBin,
    HistogramBundle, NodeSlice,
};
use alloygbm_engine::{EngineError, EngineResult};

#[cfg(target_os = "macos")]
use crate::buffers::BufferCache;
#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::pipelines::HistogramPipelineCache;

/// Embedded MSL source for the two-pass histogram build.
///
/// Exposes two entry points: `histogram_build_scatter` (per-threadgroup
/// scatter into a chunked device-memory scratch buffer) and
/// `histogram_reduce` (cross-chunk ascending reduce into the final
/// `HistogramBundle`). See `shaders/histogram.metal` for the full
/// specification.
pub const HISTOGRAM_SHADER_SOURCE: &str = include_str!("../shaders/histogram.metal");

/// Entry-point name for pass 1. Matched against `newFunctionWithName:`
/// when building the pipeline state in S1.5.
pub const KERNEL_NAME_SCATTER: &str = "histogram_build_scatter";

/// Entry-point name for pass 2.
pub const KERNEL_NAME_REDUCE: &str = "histogram_reduce";

/// Upper bound on the number of bins that fit inside the kernel's
/// threadgroup-memory `local_hist[MAX_BIN_COUNT]` array. Must match the
/// `MAX_BIN_COUNT` constant in `shaders/histogram.metal`.
pub const MAX_BIN_COUNT: u32 = 4096;

/// Default chunk size (rows per threadgroup) for pass 1. Small enough
/// that the per-chunk scratch stays modest for large nodes; large
/// enough that per-chunk fixed overhead amortizes. Tuning is deferred
/// to the benchmark phase (S1.14) — this value was chosen to match the
/// CPU backend's `SMALL_TILE_WORKLOAD_THRESHOLD` order of magnitude.
pub const ROWS_PER_CHUNK_DEFAULT: u32 = 8_192;

// -------- macOS-only dispatch path ---------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn dispatch_histograms(
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
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
    let row_indices_buffer = buffer_cache.write_row_indices(device, node.row_indices())?;

    let pair_bytes = std::mem::size_of::<[f32; 2]>();
    let output_elems = total_selected as usize * bin_count as usize;
    let output_bytes = output_elems * pair_bytes;
    let output_buffer = device
        .newBufferWithLength_options(output_bytes, options)
        .ok_or_else(|| {
            EngineError::BackendUnavailable("could not allocate output buffer".to_string())
        })?;

    // --- Encode the command buffer ---
    let command_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| EngineError::BackendUnavailable("no command buffer".to_string()))?;

    let bin_sz = if use_u16 { 2usize } else { 1usize };
    let mut cumulative_features: u32 = 0;

    // Keep scratch buffers alive until after `waitUntilCompleted` below.
    let mut scratch_keepalive: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
        Vec::with_capacity(feature_tiles.len());

    for tile in feature_tiles {
        let tile_n_features = tile.end_feature - tile.start_feature;
        let tile_scratch_elems = n_chunks as usize * tile_n_features as usize * bin_count as usize;
        let tile_scratch_bytes = tile_scratch_elems * pair_bytes;
        let scratch_buffer = device
            .newBufferWithLength_options(tile_scratch_bytes.max(1), options)
            .ok_or_else(|| {
                EngineError::BackendUnavailable("could not allocate scratch buffer".to_string())
            })?;

        let binned_offset = (tile.start_feature as usize) * (n_rows_total as usize) * bin_sz;
        let output_offset = (cumulative_features as usize) * (bin_count as usize) * pair_bytes;

        // --------- Pass 1: scatter ---------
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("no compute command encoder".to_string())
        })?;
        encoder.setComputePipelineState(&pipelines.scatter);
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
                width: 32,
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
            encoder.setBuffer_offset_atIndex(Some(&output_buffer), output_offset, 1);

            let n_chunks_cell: u32 = n_chunks;
            let tile_n_features_cell: u32 = tile_n_features;
            set_u32_bytes(&encoder, &n_chunks_cell, 2);
            set_u32_bytes(&encoder, &tile_n_features_cell, 3);

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
    }

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // --- Readback: materialize HistogramBundle ---
    //
    // Counts-per-bin are computed on CPU (see D-008). The GPU kernel
    // emits only `(grad_sum, hess_sum)` float2 pairs; reconstructing
    // counts is a single u8/u16 bin-read + uint increment per row and
    // is trivially deterministic.
    let mut feature_histograms = Vec::with_capacity(total_selected as usize);

    // SAFETY: `output_buffer` is `StorageModeShared`, fully written by
    // the reduce pass above, and we've waited for completion. The
    // pointer is valid for `output_elems` consecutive `[f32; 2]`
    // values.
    let output_ptr: *const [f32; 2] = output_buffer.contents().as_ptr() as *const [f32; 2];

    for (local_f, &feature_index) in selected_features.iter().enumerate() {
        let mut counts = vec![0u32; bin_count as usize];
        accumulate_counts(
            binned_matrix,
            node.row_indices(),
            feature_index,
            &mut counts,
        );

        let base = local_f * bin_count as usize;
        let mut bins = Vec::with_capacity(bin_count as usize);
        for (bin_idx, &count) in counts.iter().enumerate() {
            // SAFETY: `base + bin_idx < output_elems` by construction
            // (`local_f < total_selected` and `bin_idx < bin_count`).
            let gh = unsafe { *output_ptr.add(base + bin_idx) };
            bins.push(HistogramBin {
                grad_sum: gh[0],
                hess_sum: gh[1],
                count,
            });
        }
        feature_histograms.push(FeatureHistogram {
            feature_index,
            bins,
        });
    }

    // Keep scratch alive until after readback — see macro tricks below.
    drop(scratch_keepalive);

    Ok(HistogramBundle::from_cpu(node.node_id, feature_histograms))
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
