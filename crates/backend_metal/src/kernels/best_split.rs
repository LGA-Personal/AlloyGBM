//! GPU dispatch wrapper for the Stage 4a `best_split.metal` kernel.
//!
//! Mirrors the shape of `kernels/subtract.rs` (batched pool-direct
//! path): one command buffer per call, two kernel dispatches (per-
//! feature scan then cross-feature reduce), one commit +
//! waitUntilCompleted, then host reads back the small per-node
//! `SplitDecisionGpu` array.

#![cfg(target_os = "macos")]
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use alloygbm_core::{GpuHistogramHandle, HistogramStorage};
use alloygbm_engine::{EngineError, EngineResult, SplitFindRequest, SplitSelectionOptions};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::device::MetalDevice;
use crate::histogram_residency::HistogramResidencyPool;
use crate::pipelines::BestSplitPipelineCache;
use crate::profile;
use crate::residency::ResidencyPool;
use crate::split_decision_residency::{
    SplitDecisionGpu, SplitDecisionPool, SplitDecisionReleaseGuard,
};

/// Mirror of MSL `BestSplitParams` (32 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BestSplitParamsGpu {
    pub l2_lambda: f32,
    pub l1_alpha: f32,
    pub min_child_hessian: f32,
    pub min_leaf_magnitude: f32,
    pub missing_bin_index: u32,
    pub bin_count: u32,
    pub numeric_feature_count: u32,
    pub node_count: u32,
}

const _: () = assert!(size_of::<BestSplitParamsGpu>() == 32);

/// Per-feature scratch entry — must match MSL `PerFeatureCandidate` (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PerFeatureCandidateGpu {
    gain: f32,
    weighted_gain: f32,
    grad_left: f32,
    hess_left: f32,
    feature_idx: u32,
    bin_threshold: u32,
    flags: u32,
    _pad: u32,
}

const _: () = assert!(size_of::<PerFeatureCandidateGpu>() == 32);

/// Encode + commit + wait + read back N nodes' best-split decisions
/// in a single command buffer.
///
/// Each request carries its own GPU-resident histogram (separate pool entry).
/// The per-feature kernel is dispatched once per node, using buffer offsets into
/// the shared scratch buffer so each node's candidates land at `node_idx × nf`.
/// A single reduce-features dispatch then collapses each node's nf candidates
/// into one `SplitDecisionGpu`. This avoids any memcpy concatenation of
/// histogram planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_find_best_splits_batch(
    metal_device: &MetalDevice,
    pipeline_cache: &BestSplitPipelineCache,
    histogram_residency: &HistogramResidencyPool,
    split_decision_pool: &SplitDecisionPool,
    residency: &ResidencyPool,
    requests: &[SplitFindRequest<'_>],
    options: SplitSelectionOptions,
    feature_weights: &[f32],
) -> EngineResult<Vec<SplitDecisionGpu>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let node_count = requests.len() as u32;

    // Collect histogram pool entries for every node upfront so their
    // Retained<> handles stay alive for the full command-buffer lifetime.
    let hist_entries: Vec<_> = requests
        .iter()
        .map(|r| {
            let handle: GpuHistogramHandle = match &r.histograms.storage {
                HistogramStorage::Gpu { handle, .. } => *handle,
                HistogramStorage::Cpu(_) => {
                    return Err(EngineError::BackendUnavailable(
                        "dispatch_find_best_splits_batch: histogram not GPU-resident".to_string(),
                    ));
                }
            };
            histogram_residency.get(handle).ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "dispatch_find_best_splits_batch: histogram pool entry not found".to_string(),
                )
            })
        })
        .collect::<EngineResult<Vec<_>>>()?;

    // All nodes share feature count and bin count (engine level-wise invariant).
    let numeric_feature_count = hist_entries[0].feature_indices.len() as u32;
    let bin_count = hist_entries[0].shape.bin_count;

    // Allocate the output split-decision buffer via the pool (RAII guard
    // releases it when the function returns, success or error).
    let device = &metal_device.device;
    let out_handle = split_decision_pool.mint(device, residency, node_count)?;
    let _guard = SplitDecisionReleaseGuard::new(split_decision_pool, residency, out_handle);
    let out_buffer = split_decision_pool.buffer_for(out_handle)?;

    // Flat scratch buffer: [node_count × numeric_feature_count × PerFeatureCandidateGpu].
    // Each per-node dispatch writes into its node's slice via buffer offset.
    let scratch_per_node = (numeric_feature_count as usize) * size_of::<PerFeatureCandidateGpu>();
    let scratch_bytes = (node_count as usize)
        .saturating_mul(scratch_per_node)
        .max(16);
    let scratch_buf = device
        .newBufferWithLength_options(scratch_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "dispatch_find_best_splits_batch: scratch alloc failed ({scratch_bytes} bytes)"
            ))
        })?;

    // Feature-indices buffer [numeric_feature_count × u32] — shared across nodes.
    let feat_idx_bytes = (numeric_feature_count as usize * 4).max(16);
    let feat_idx_buf = device
        .newBufferWithLength_options(feat_idx_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: feature_indices alloc failed".to_string(),
            )
        })?;
    // SAFETY: feat_idx_buf is StorageModeShared with at least `feat_idx_bytes` valid memory;
    // we write exactly `numeric_feature_count` u32 values from a Vec<u32>.
    // Metal page-aligned allocations are 4-byte aligned. Buffer outlives the copy.
    unsafe {
        std::ptr::copy_nonoverlapping(
            hist_entries[0].feature_indices.as_ptr(),
            feat_idx_buf.contents().as_ptr() as *mut u32,
            numeric_feature_count as usize,
        );
    }

    // Feature-weights buffer [numeric_feature_count × f32] — shared across nodes.
    let fw_bytes = (numeric_feature_count as usize * 4).max(16);
    let fw_buf = device
        .newBufferWithLength_options(fw_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: feature_weights alloc failed".to_string(),
            )
        })?;
    let fw_src: Vec<f32> = hist_entries[0]
        .feature_indices
        .iter()
        .map(|&fi| feature_weights.get(fi as usize).copied().unwrap_or(1.0))
        .collect();
    // SAFETY: same rationale as feat_idx_buf above.
    unsafe {
        std::ptr::copy_nonoverlapping(
            fw_src.as_ptr(),
            fw_buf.contents().as_ptr() as *mut f32,
            numeric_feature_count as usize,
        );
    }

    // Params for per-node dispatches (node_count=1 per dispatch; the kernel
    // uses tg_id.y as node_idx which is always 0 in a height-1 grid — the
    // scratch buffer offset places the result in the correct node slot).
    let per_node_params = BestSplitParamsGpu {
        l2_lambda: options.l2_lambda,
        l1_alpha: options.l1_alpha,
        min_child_hessian: options.min_child_hessian,
        min_leaf_magnitude: options.min_leaf_magnitude,
        missing_bin_index: options.missing_bin_index as u32,
        bin_count,
        numeric_feature_count,
        node_count: 1,
    };
    // Params for the cross-node reduce dispatch.
    let reduce_params = BestSplitParamsGpu {
        node_count,
        ..per_node_params
    };

    let _p_total = profile::ScopedProbe::new(&profile::FIND_BEST_SPLITS_BATCH);

    let cmd_buf = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: commandBuffer() returned nil".to_string(),
            )
        })?;

    let tg_size = ((bin_count + 31) / 32 * 32).min(1024) as usize;
    let reduce_tg = ((numeric_feature_count + 31) / 32 * 32).min(1024) as usize;

    {
        let _p_dispatch = profile::ScopedProbe::new(&profile::BS_DISPATCH);

        // ---- Pass 1: per-feature scan, one dispatch per node ----
        // Each dispatch uses that node's own histogram buffers and writes to
        // its slice of the scratch buffer (via buffer offset). Sequential
        // dispatches within a single encoder are ordered; no barrier needed
        // since each node writes to a distinct scratch region.
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "dispatch_find_best_splits_batch: computeCommandEncoder() returned nil (pass 1)"
                        .to_string(),
                )
            })?;
        encoder.setComputePipelineState(&pipeline_cache.per_feature);

        for (node_idx, entry) in hist_entries.iter().enumerate() {
            let scratch_offset = node_idx * scratch_per_node;
            // SAFETY: all Retained buffers outlive the encoder; per_node_params is a
            // stack-local POD struct valid for the synchronous setBytes call.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&entry.grad), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&entry.hess), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&entry.counts), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&feat_idx_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&fw_buf), 0, 4);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked((&raw const per_node_params) as *mut c_void),
                    size_of::<BestSplitParamsGpu>(),
                    5,
                );
                encoder.setBuffer_offset_atIndex(Some(&scratch_buf), scratch_offset, 6);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: numeric_feature_count as usize, height: 1, depth: 1 },
                MTLSize { width: tg_size, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();

        // ---- Pass 2: cross-feature reduce, one dispatch for all nodes ----
        let encoder2 = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "dispatch_find_best_splits_batch: computeCommandEncoder() returned nil (pass 2)"
                        .to_string(),
                )
            })?;
        encoder2.setComputePipelineState(&pipeline_cache.reduce_features);
        // SAFETY: same as pass 1.
        unsafe {
            encoder2.setBuffer_offset_atIndex(Some(&scratch_buf), 0, 0);
            encoder2.setBytes_length_atIndex(
                NonNull::new_unchecked((&raw const reduce_params) as *mut c_void),
                size_of::<BestSplitParamsGpu>(),
                1,
            );
            encoder2.setBuffer_offset_atIndex(Some(&out_buffer), 0, 2);
        }
        encoder2.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: node_count as usize, height: 1, depth: 1 },
            MTLSize { width: reduce_tg, height: 1, depth: 1 },
        );
        encoder2.endEncoding();
    }

    {
        let _p_wait = profile::ScopedProbe::new(&profile::BS_COMMIT_WAIT);
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    let decisions = {
        let _p_read = profile::ScopedProbe::new(&profile::BS_DECISION_READBACK);
        split_decision_pool.read_decisions(out_handle)?
    };
    // _guard drops here → split_decision_pool.release(residency, out_handle)

    Ok(decisions)
}
