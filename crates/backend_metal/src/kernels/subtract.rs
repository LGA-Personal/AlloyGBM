//! Histogram-subtract kernel — Rust-side orchestration.
//!
//! `SUBTRACT_SHADER_SOURCE` embeds the MSL source; `dispatch_subtract`
//! flattens two CPU-resident `HistogramBundle`s into three SoA buffers
//! each (grad_sum, hess_sum, counts), dispatches the elementwise
//! kernel, and reconstructs a CPU-resident `HistogramBundle` from the
//! three output buffers.
//!
//! Stage 3 scope (S3.6): this is infrastructural. The CPU round-trip
//! (flatten → upload → dispatch → readback → repack) has no chance of
//! beating the CPU elementwise loop — the win lands in S3.7 once both
//! `parent` and `child` are already GPU-resident, at which point the
//! upload/readback steps drop and only the 1-dispatch kernel cost
//! remains. The contract here (bit-exact with
//! `subtract_histogram_bundle`) holds in both forms.
//!
//! Pipeline compilation is delegated to
//! [`crate::pipelines::SubtractPipelineCache`].

use alloygbm_core::{FeatureHistogram, HistogramBin, HistogramBundle};
use alloygbm_engine::{EngineError, EngineResult};

#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::pipelines::SubtractPipelineCache;

/// Embedded MSL source for the elementwise subtract kernel.
pub const SUBTRACT_SHADER_SOURCE: &str = include_str!("../shaders/subtract.metal");

/// Kernel entry-point name for `subtract.metal`.
pub const KERNEL_NAME_SUBTRACT: &str = "subtract_elementwise";

/// Threads per threadgroup for the subtract kernel. Matches the
/// `BLOCK_SIZE` function constant baked into the pipeline.
pub const BLOCK_SIZE: u32 = 1024;

/// Wire-format uniform. Must match the MSL `SubtractUniform` layout.
///
/// `dead_code` allow: the consumer is S3.7 (see module doc). Until
/// then this type + the `dispatch_subtract` function below are
/// exercised by unit tests but not by the release-library surface.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SubtractUniformPod {
    total_elems: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Function-constant key driving kernel specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubtractSpecKey {
    pub block_size: u32,
}

// -------- macOS-only dispatch path ---------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code, dead_code)]
pub(crate) fn dispatch_subtract(
    metal_device: &MetalDevice,
    pipeline_cache: &SubtractPipelineCache,
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
) -> EngineResult<HistogramBundle> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    // --- Contract checks (mirror engine::subtract_histogram_bundle_into) ---
    let parent_fhs = parent.feature_histograms();
    let child_fhs = child.feature_histograms();
    if parent_fhs.len() != child_fhs.len() {
        return Err(EngineError::ContractViolation(format!(
            "parent histogram feature count {} does not match child histogram feature count {}",
            parent_fhs.len(),
            child_fhs.len()
        )));
    }
    let feature_count = parent_fhs.len();
    if feature_count == 0 {
        return Ok(HistogramBundle::new_zeroed(&[], 0));
    }
    let bin_count = parent_fhs[0].bins.len();
    for (i, (p, c)) in parent_fhs.iter().zip(child_fhs).enumerate() {
        if p.bins.len() != bin_count || c.bins.len() != bin_count {
            return Err(EngineError::ContractViolation(format!(
                "feature {i} histogram bin counts differ: parent={}, child={}, expected={bin_count}",
                p.bins.len(),
                c.bins.len(),
            )));
        }
        if p.feature_index != c.feature_index {
            return Err(EngineError::ContractViolation(format!(
                "feature_index mismatch at position {i}: parent={}, child={}",
                p.feature_index, c.feature_index
            )));
        }
    }

    let total_elems: u32 = u32::try_from(feature_count * bin_count).map_err(|_| {
        EngineError::BackendUnavailable(
            "subtract kernel: F*B exceeds u32 range (unsupported at this stage)".to_string(),
        )
    })?;

    // --- Flatten to SoA on the CPU side. This is the upload path that
    //     S3.7 will bypass by passing GPU-resident buffers directly. ---
    let mut parent_grad = Vec::with_capacity(total_elems as usize);
    let mut parent_hess = Vec::with_capacity(total_elems as usize);
    let mut parent_counts = Vec::with_capacity(total_elems as usize);
    let mut child_grad = Vec::with_capacity(total_elems as usize);
    let mut child_hess = Vec::with_capacity(total_elems as usize);
    let mut child_counts = Vec::with_capacity(total_elems as usize);
    for (pfh, cfh) in parent_fhs.iter().zip(child_fhs) {
        for bin in &pfh.bins {
            parent_grad.push(bin.grad_sum);
            parent_hess.push(bin.hess_sum);
            parent_counts.push(bin.count);
        }
        for bin in &cfh.bins {
            child_grad.push(bin.grad_sum);
            child_hess.push(bin.hess_sum);
            child_counts.push(bin.count);
        }
    }

    // --- Pipeline (single function-constant: BLOCK_SIZE) ---
    let spec = SubtractSpecKey {
        block_size: BLOCK_SIZE,
    };
    let pipelines = pipeline_cache
        .get_or_build(spec)
        .map_err(EngineError::BackendUnavailable)?;
    let pipeline = &pipelines.subtract;

    let device = &metal_device.device;
    let res_opts = MTLResourceOptions::StorageModeShared;

    // Helper: upload a `&[T]` slice to a fresh shared buffer.
    let upload_bytes = |bytes: &[u8]| -> EngineResult<_> {
        let len = bytes.len().max(1);
        let buf = device
            .newBufferWithLength_options(len, res_opts)
            .ok_or_else(|| EngineError::BackendUnavailable("subtract buffer alloc".to_string()))?;
        if !bytes.is_empty() {
            let ptr = buf.contents().as_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            }
        }
        Ok(buf)
    };
    let upload_slice_f32 = |data: &[f32]| -> EngineResult<_> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        upload_bytes(bytes)
    };
    let upload_slice_u32 = |data: &[u32]| -> EngineResult<_> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        upload_bytes(bytes)
    };

    let parent_grad_buf = upload_slice_f32(&parent_grad)?;
    let parent_hess_buf = upload_slice_f32(&parent_hess)?;
    let parent_counts_buf = upload_slice_u32(&parent_counts)?;
    let child_grad_buf = upload_slice_f32(&child_grad)?;
    let child_hess_buf = upload_slice_f32(&child_hess)?;
    let child_counts_buf = upload_slice_u32(&child_counts)?;

    let float_out_bytes = (total_elems as usize).max(1) * std::mem::size_of::<f32>();
    let uint_out_bytes = (total_elems as usize).max(1) * std::mem::size_of::<u32>();
    let out_grad_buf = device
        .newBufferWithLength_options(float_out_bytes, res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("subtract out_grad alloc".to_string()))?;
    let out_hess_buf = device
        .newBufferWithLength_options(float_out_bytes, res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("subtract out_hess alloc".to_string()))?;
    let out_counts_buf = device
        .newBufferWithLength_options(uint_out_bytes, res_opts)
        .ok_or_else(|| EngineError::BackendUnavailable("subtract out_counts alloc".to_string()))?;

    let uniform_pod = SubtractUniformPod {
        total_elems,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uniform_bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&uniform_pod).cast::<u8>(),
            std::mem::size_of::<SubtractUniformPod>(),
        )
    };
    let uniform_buf = upload_bytes(uniform_bytes)?;

    // --- Dispatch ---
    let command_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| EngineError::BackendUnavailable("subtract command buffer".to_string()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| EngineError::BackendUnavailable("subtract compute encoder".to_string()))?;
    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&parent_grad_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&parent_hess_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&parent_counts_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&child_grad_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&child_hess_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&child_counts_buf), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&out_grad_buf), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&out_hess_buf), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&out_counts_buf), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&uniform_buf), 0, 9);
    }
    let num_blocks = total_elems.div_ceil(BLOCK_SIZE);
    let grid = MTLSize {
        width: num_blocks as usize,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: BLOCK_SIZE as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // --- Readback into a fresh HistogramBundle ---
    let out_grad_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            out_grad_buf.contents().as_ptr().cast::<f32>(),
            total_elems as usize,
        )
    };
    let out_hess_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            out_hess_buf.contents().as_ptr().cast::<f32>(),
            total_elems as usize,
        )
    };
    let out_counts_slice: &[u32] = unsafe {
        std::slice::from_raw_parts(
            out_counts_buf.contents().as_ptr().cast::<u32>(),
            total_elems as usize,
        )
    };

    let mut feature_histograms = Vec::with_capacity(feature_count);
    for (f, pfh) in parent_fhs.iter().enumerate() {
        let base = f * bin_count;
        let mut bins = Vec::with_capacity(bin_count);
        for b in 0..bin_count {
            bins.push(HistogramBin {
                grad_sum: out_grad_slice[base + b],
                hess_sum: out_hess_slice[base + b],
                count: out_counts_slice[base + b],
            });
        }
        feature_histograms.push(FeatureHistogram {
            feature_index: pfh.feature_index,
            bins,
        });
    }

    Ok(HistogramBundle::from_cpu(node_id, feature_histograms))
}

// -------- Non-macOS stub -------------------------------------------

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub(crate) fn dispatch_subtract(
    _parent: &HistogramBundle,
    _child: &HistogramBundle,
    _node_id: u32,
) -> EngineResult<HistogramBundle> {
    Err(EngineError::BackendUnavailable(
        "Metal backend is macOS-only".to_string(),
    ))
}
