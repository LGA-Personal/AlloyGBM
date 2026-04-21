//! Row-partition kernel — Rust-side orchestration.
//!
//! `PARTITION_SHADER_SOURCE` embeds the MSL source; `dispatch_partition`
//! runs the three-pass stream-compaction kernel and returns CPU-resident
//! `left_row_indices` and `right_row_indices` vectors that are
//! bit-identical to the CPU backend's stable partition.
//!
//! Stage 3 scope (DECISIONS D-017): the GPU partition handles both
//! continuous-threshold splits and categorical-bitset splits. Only
//! single-threadgroup block scans are supported at this stage — node
//! sizes that exceed `BLOCK_SIZE × MAX_BLOCKS_SINGLE_SCAN` fall back
//! to CPU. At `BLOCK_SIZE = 1024` and a single-threadgroup scan of
//! 1024 block totals, the GPU path covers nodes up to ~1M rows; the
//! fallback is documented so Stage 3's hot training shapes go
//! through the GPU while pathological cases are still correct.
//!
//! Pipeline compilation is delegated to
//! [`crate::pipelines::PartitionPipelineCache`].

use alloygbm_core::{BinnedMatrix, NodeSlice, PartitionResult, SplitCandidate};
use alloygbm_engine::{EngineError, EngineResult};

#[cfg(target_os = "macos")]
use crate::buffers::BufferCache;
#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::pipelines::PartitionPipelineCache;

/// Embedded MSL source for the partition kernel.
///
/// Exposes three entry points:
///  - `partition_flag_and_count` — per-row direction + per-block left count
///  - `partition_scan_blocks`    — single-TG exclusive scan of block totals
///  - `partition_scatter`        — stable stream compaction into two buffers
pub const PARTITION_SHADER_SOURCE: &str = include_str!("../shaders/partition.metal");

/// Kernel entry-point names.
pub const KERNEL_NAME_PARTITION_FLAG_AND_COUNT: &str = "partition_flag_and_count";
pub const KERNEL_NAME_PARTITION_SCAN_BLOCKS: &str = "partition_scan_blocks";
pub const KERNEL_NAME_PARTITION_SCATTER: &str = "partition_scatter";

/// Threads per threadgroup for the partition passes. Must match the
/// `BLOCK_SIZE` function constant baked into the pipeline.
///
/// Chosen at 1024 so 32 SIMD groups of 32 lanes cover one block
/// (aligned with the Metal 4 expert's guidance that SIMD width is
/// always 32 on M-series). 1024 also matches the single-threadgroup
/// scan cap in pass 2 — raise this and pass 2 must grow a
/// hierarchical scan.
pub const BLOCK_SIZE: u32 = 1024;

/// Single-threadgroup pass 2 scan cannot exceed this many block
/// totals. With `BLOCK_SIZE = 1024`, this corresponds to ~1M rows in
/// a single node — the fallback gate ships the overflow case to CPU
/// with a clear comment pointing at a future hierarchical scan.
pub const MAX_BLOCKS_SINGLE_SCAN: u32 = BLOCK_SIZE;

/// Wire-format uniform for the pass-1 kernel. Must match the MSL
/// `PartitionUniform` layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct PartitionUniformPod {
    feature_col_base: u32,
    row_count: u32,
    missing_bin: u32,
    num_rows_in_node: u32,
    threshold_bin: u32,
    default_left: u32,
    bitset_byte_len: u32,
    _pad: u32,
}

/// Function-constant pair driving kernel specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionSpecKey {
    pub block_size: u32,
    pub split_kind: u32,
    pub bin_is_u16: bool,
}

// -------- macOS-only dispatch path ---------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn dispatch_partition(
    metal_device: &MetalDevice,
    pipeline_cache: &PartitionPipelineCache,
    buffer_cache: &BufferCache,
    binned_matrix: &BinnedMatrix,
    node: &NodeSlice,
    split: &SplitCandidate,
) -> EngineResult<PartitionResult> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    node.validate_bounds(binned_matrix.row_count)?;
    if split.feature_index as usize >= binned_matrix.feature_count {
        return Err(EngineError::ContractViolation(format!(
            "split feature_index {} exceeds feature_count {}",
            split.feature_index, binned_matrix.feature_count
        )));
    }

    let num_rows = node.row_count() as u32;
    if num_rows == 0 {
        return Ok(PartitionResult::from_cpu(Vec::new(), Vec::new()));
    }

    let num_blocks = num_rows.div_ceil(BLOCK_SIZE);
    if num_blocks > MAX_BLOCKS_SINGLE_SCAN {
        // Pass 2's single-threadgroup scan cannot cover this many
        // blocks. Signal to caller by returning a sentinel error that
        // callers translate into a CPU fallback. Using
        // BackendUnavailable keeps the engine's existing error
        // handling contract simple.
        return Err(EngineError::BackendUnavailable(format!(
            "partition kernel: node has {num_rows} rows (>{limit} blocks); \
             falling back to CPU pending a hierarchical scan",
            limit = MAX_BLOCKS_SINGLE_SCAN
        )));
    }

    // --- Binned-matrix column buffer (reused across calls via binned cache) ---
    let (bins_buffer, bin_is_u16) =
        upload_binned_column(buffer_cache, metal_device, binned_matrix)?;

    // --- Row indices (reused slot; content fresh per call) ---
    let row_indices_buffer =
        buffer_cache.write_row_indices(&metal_device.device, node.row_indices())?;

    // --- Uniform ---
    let feature_col_base = split.feature_index as usize * binned_matrix.row_count;
    let missing = binned_matrix.missing_bin() as u32;
    let split_kind = if split.is_categorical { 1u32 } else { 0u32 };
    let (bitset_bytes, bitset_len_u32): (Vec<u8>, u32) = match &split.categorical_bitset {
        Some(bs) => (bs.clone(), bs.len() as u32),
        None => (Vec::new(), 0u32),
    };
    let uniform = PartitionUniformPod {
        feature_col_base: feature_col_base as u32,
        row_count: binned_matrix.row_count as u32,
        missing_bin: missing,
        num_rows_in_node: num_rows,
        threshold_bin: split.threshold_bin as u32,
        default_left: if split.default_left { 1 } else { 0 },
        bitset_byte_len: bitset_len_u32,
        _pad: 0,
    };

    // --- Pipelines (specialized on BLOCK_SIZE, SPLIT_KIND, BIN_IS_U16) ---
    let spec = PartitionSpecKey {
        block_size: BLOCK_SIZE,
        split_kind,
        bin_is_u16,
    };
    let pipelines = pipeline_cache
        .get_or_build(spec)
        .map_err(EngineError::BackendUnavailable)?;

    let device = &metal_device.device;
    let res_opts = MTLResourceOptions::StorageModeShared;

    // --- Per-call scratch / output buffers (sized to this node) ---
    let direction_flags_buf = device
        .newBufferWithLength_options((num_rows as usize).max(1), res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("direction_flags alloc".to_string()))?;
    let block_totals_buf = device
        .newBufferWithLength_options(
            (num_blocks as usize) * std::mem::size_of::<u32>().max(1),
            res_opts,
        )
        .ok_or_else(|| EngineError::BackendUnavailable("block_left_totals alloc".to_string()))?;
    let block_bases_buf = device
        .newBufferWithLength_options(
            (num_blocks as usize + 1) * std::mem::size_of::<u32>(),
            res_opts,
        )
        .ok_or_else(|| EngineError::BackendUnavailable("block_left_bases alloc".to_string()))?;
    // Output buffers sized to the worst case (everything lands on one
    // side). We shrink on readback once the grand total is known.
    let out_left_buf = device
        .newBufferWithLength_options(
            (num_rows as usize).max(1) * std::mem::size_of::<u32>(),
            res_opts,
        )
        .ok_or_else(|| EngineError::BackendUnavailable("out_left alloc".to_string()))?;
    let out_right_buf = device
        .newBufferWithLength_options(
            (num_rows as usize).max(1) * std::mem::size_of::<u32>(),
            res_opts,
        )
        .ok_or_else(|| EngineError::BackendUnavailable("out_right alloc".to_string()))?;

    // Dummy bitset buffer for continuous splits (at least 1 byte so
    // Metal never sees a zero-length binding).
    let bitset_bytes_padded = if bitset_bytes.is_empty() {
        vec![0u8]
    } else {
        bitset_bytes.clone()
    };
    let bitset_buf = device
        .newBufferWithLength_options(bitset_bytes_padded.len().max(1), res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("bitset alloc".to_string()))?;
    if !bitset_bytes.is_empty() {
        // SAFETY: `bitset_buf` is StorageModeShared so contents() is
        // valid for `bitset_bytes.len()` bytes. We just allocated it
        // so no one else is reading.
        unsafe {
            let dst = bitset_buf.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(bitset_bytes.as_ptr(), dst, bitset_bytes.len());
        }
    }

    // Dummy u8 / u16 buffer for whichever one isn't used by this
    // pipeline variant. Metal requires a valid (non-null) binding at
    // every used buffer index.
    let dummy_buf = device
        .newBufferWithLength_options(1, res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("dummy alloc".to_string()))?;

    let (bind_u8, bind_u16) = if bin_is_u16 {
        (dummy_buf.clone(), bins_buffer.clone())
    } else {
        (bins_buffer.clone(), dummy_buf.clone())
    };

    // --- Encode all three passes on one command buffer ---
    let command_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| EngineError::BackendUnavailable("no command buffer".to_string()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| EngineError::BackendUnavailable("no compute encoder".to_string()))?;

    // Pass 1 — flag + count.
    encoder.setComputePipelineState(&pipelines.flag_and_count);
    // SAFETY: all Metal buffers outlive the encoder through `Retained`;
    // `uniform` is a stack slot whose bytes are copied synchronously by
    // `setBytes`. Buffer indices match the MSL attribute declarations.
    unsafe {
        use std::ffi::c_void;
        use std::ptr::NonNull;
        encoder.setBuffer_offset_atIndex(Some(&bind_u8), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bind_u16), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked((&raw const uniform) as *mut c_void),
            std::mem::size_of::<PartitionUniformPod>(),
            3,
        );
        encoder.setBuffer_offset_atIndex(Some(&bitset_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&direction_flags_buf), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&block_totals_buf), 0, 6);

        let threadgroups = MTLSize {
            width: num_blocks as usize,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: BLOCK_SIZE as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    }

    // Pass 2 — scan block totals (single threadgroup).
    encoder.setComputePipelineState(&pipelines.scan_blocks);
    // SAFETY: `num_blocks_val` is a stack slot copied synchronously by
    // setBytes.
    let num_blocks_val: u32 = num_blocks;
    unsafe {
        use std::ffi::c_void;
        use std::ptr::NonNull;
        encoder.setBuffer_offset_atIndex(Some(&block_totals_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&block_bases_buf), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked((&raw const num_blocks_val) as *mut c_void),
            std::mem::size_of::<u32>(),
            2,
        );

        let threadgroups = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: BLOCK_SIZE as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    }

    // Pass 3 — scatter into out_left / out_right.
    encoder.setComputePipelineState(&pipelines.scatter);
    let num_rows_val: u32 = num_rows;
    // SAFETY: see pass 2.
    unsafe {
        use std::ffi::c_void;
        use std::ptr::NonNull;
        encoder.setBuffer_offset_atIndex(Some(&row_indices_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&direction_flags_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&block_bases_buf), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked((&raw const num_rows_val) as *mut c_void),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBuffer_offset_atIndex(Some(&out_left_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&out_right_buf), 0, 5);

        let threadgroups = MTLSize {
            width: num_blocks as usize,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: BLOCK_SIZE as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    }

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // --- Readback ---
    //
    // Grand total of left rows lives in the sentinel slot
    // `block_left_bases[num_blocks]`.
    //
    // SAFETY: both buffers are StorageModeShared, fully written by
    // kernels we just awaited, and hold the expected number of u32
    // elements.
    let total_left = unsafe {
        let ptr = block_bases_buf.contents().as_ptr() as *const u32;
        *ptr.add(num_blocks as usize)
    };
    if total_left > num_rows {
        return Err(EngineError::ContractViolation(format!(
            "partition kernel produced total_left={total_left} > num_rows={num_rows}"
        )));
    }
    let total_right = num_rows - total_left;

    let mut left_vec = vec![0u32; total_left as usize];
    let mut right_vec = vec![0u32; total_right as usize];

    // SAFETY: the kernel populated `out_left[0..total_left]` and
    // `out_right[0..total_right]` with u32 row indices.
    unsafe {
        if total_left > 0 {
            let src = out_left_buf.contents().as_ptr() as *const u32;
            std::ptr::copy_nonoverlapping(src, left_vec.as_mut_ptr(), total_left as usize);
        }
        if total_right > 0 {
            let src = out_right_buf.contents().as_ptr() as *const u32;
            std::ptr::copy_nonoverlapping(src, right_vec.as_mut_ptr(), total_right as usize);
        }
    }

    // Keep temporaries alive until after readback.
    drop(out_left_buf);
    drop(out_right_buf);
    drop(direction_flags_buf);
    drop(block_totals_buf);
    drop(block_bases_buf);
    drop(bitset_buf);
    drop(dummy_buf);
    drop(bind_u8);
    drop(bind_u16);

    Ok(PartitionResult::from_cpu(left_vec, right_vec))
}

/// Upload the column-major bin column for the kernel's feature. Reuses
/// `BufferCache`'s binned slot so repeated calls with the same matrix
/// don't recopy the (potentially large) column-major table.
///
/// Returns `(buffer, is_u16)`.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn upload_binned_column(
    buffer_cache: &BufferCache,
    metal_device: &MetalDevice,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<(
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    bool,
)> {
    use alloygbm_core::BinStorage;

    let device = &metal_device.device;
    match &binned_matrix.bins_col_adaptive {
        BinStorage::U8(bytes) => {
            let buf = buffer_cache.get_or_upload_binned(device, bytes.as_slice(), false)?;
            Ok((buf, false))
        }
        BinStorage::U16(words) => {
            let buf = buffer_cache.get_or_upload_binned(device, words.as_slice(), true)?;
            Ok((buf, true))
        }
    }
}

// -------- Non-macOS stub --------------------------------------------

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub fn dispatch_partition() -> EngineResult<PartitionResult> {
    Err(EngineError::BackendUnavailable(
        "Metal partition kernel is only available on macOS".to_string(),
    ))
}
