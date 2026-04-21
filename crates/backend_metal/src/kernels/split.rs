//! Best-split kernel — Rust-side orchestration.
//!
//! `SPLIT_SHADER_SOURCE` embeds the MSL source; `dispatch_best_split`
//! partitions the histogram bundle into continuous / categorical
//! features, runs the continuous half on the GPU, runs the categorical
//! half on the embedded `CpuBackend`, and combines the two candidate
//! pools applying `feature_weights` on the CPU side.
//!
//! Stage 2 scope (DECISIONS D-010): categorical features stay on CPU
//! because the Fisher-sort optimal-binary-partition is a separate
//! research problem on GPU. The GPU path handles every continuous
//! feature with a single dispatch.
//!
//! Pipeline compilation is delegated to
//! [`crate::pipelines::SplitPipelineCache`].

use alloygbm_core::{FeatureHistogram, HistogramBundle, NodeStats, SplitCandidate};
use alloygbm_engine::{CategoricalFeatureInfo, EngineError, EngineResult, SplitSelectionOptions};

#[cfg(target_os = "macos")]
use alloygbm_backend_cpu::CpuBackend;
#[cfg(target_os = "macos")]
use alloygbm_engine::BackendOps;

#[cfg(target_os = "macos")]
use crate::buffers::BufferCache;
#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::pipelines::SplitPipelineCache;

/// Embedded MSL source for the best-split kernel.
///
/// Exposes one entry point, `best_split_per_feature`, that takes a
/// flattened `HistogramBundle` plus options and emits one
/// `FeatureSplitCandidate` per feature. Cross-feature argmax is done
/// on the CPU after readback.
pub const SPLIT_SHADER_SOURCE: &str = include_str!("../shaders/split.metal");

/// Entry-point name for the per-feature split kernel.
pub const KERNEL_NAME_BEST_SPLIT_PER_FEATURE: &str = "best_split_per_feature";

/// Must match the kernel's `MAX_BIN_COUNT` equivalent — the histogram
/// kernel caps at 4096 so the split kernel mirrors that. Dispatched
/// pipelines reject bin counts above this.
pub const MAX_BIN_COUNT: u32 = 4096;

/// Wire-format struct matching the MSL `SplitOptionsPOD` layout.
#[repr(C)]
#[derive(Clone, Copy)]
struct SplitOptionsPod {
    missing_bin_index: u32,
    l1_alpha: f32,
    l2_lambda: f32,
    min_child_hessian: f32,
    min_leaf_magnitude: f32,
}

/// Wire-format struct matching the MSL `FeatureSplitCandidate` layout.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FeatureSplitCandidatePod {
    gain: f32,
    threshold_bin: u32,
    default_left: u32,
    has_split: u32,
    left_grad: f32,
    left_hess: f32,
    left_count: u32,
    right_grad: f32,
    right_hess: f32,
    right_count: u32,
}

// -------- macOS-only dispatch path ---------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn dispatch_best_split(
    metal_device: &MetalDevice,
    pipeline_cache: &SplitPipelineCache,
    buffer_cache: &BufferCache,
    cpu: &CpuBackend,
    histograms: &HistogramBundle,
    options: SplitSelectionOptions,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
) -> EngineResult<Option<SplitCandidate>> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    if histograms.feature_histograms.is_empty() {
        return Ok(None);
    }

    // Uniform bin-count check: every FeatureHistogram in a bundle must
    // have the same number of bins by construction, but assert it
    // defensively — the kernel assumes a flat `[n_features × BIN_COUNT]`
    // layout.
    let bin_count = histograms.feature_histograms[0].bins.len();
    if bin_count == 0 {
        return Ok(None);
    }
    if (bin_count as u32) > MAX_BIN_COUNT {
        // Out-of-range bin count — fall back to CPU.
        return cpu.best_split_with_options(
            histograms,
            options,
            feature_weights,
            categorical_features,
        );
    }
    for fh in &histograms.feature_histograms {
        if fh.bins.len() != bin_count {
            return Err(EngineError::ContractViolation(format!(
                "Metal best_split requires uniform bin count across features; \
                 expected {}, got {} for feature {}",
                bin_count,
                fh.bins.len(),
                fh.feature_index
            )));
        }
    }

    // --- Partition features into continuous vs categorical ---
    let is_categorical = |feature_index: u32| -> bool {
        categorical_features
            .iter()
            .any(|c| c.feature_index == feature_index as usize)
    };

    let (cont_histograms, cat_histograms): (Vec<FeatureHistogram>, Vec<FeatureHistogram>) =
        histograms
            .feature_histograms
            .iter()
            .cloned()
            .partition(|fh| !is_categorical(fh.feature_index));

    // --- Categorical features (if any) go through the CPU backend ---
    let cat_candidate: Option<SplitCandidate> = if cat_histograms.is_empty() {
        None
    } else {
        let cat_bundle = HistogramBundle {
            node_id: histograms.node_id,
            feature_histograms: cat_histograms,
        };
        cpu.best_split_with_options(&cat_bundle, options, feature_weights, categorical_features)?
    };

    // --- Continuous features (if any) go through the GPU kernel ---
    let cont_candidate: Option<SplitCandidate> = if cont_histograms.is_empty() {
        None
    } else {
        // Flatten histograms to SoA layout expected by the kernel:
        //   grad[n_features × bin_count], hess[...], counts[...].
        let n_features = cont_histograms.len();
        let mut grad_sums = Vec::with_capacity(n_features * bin_count);
        let mut hess_sums = Vec::with_capacity(n_features * bin_count);
        let mut counts = Vec::with_capacity(n_features * bin_count);
        // All continuous after the partition — but still build the mask
        // for kernel robustness.
        let mut continuous_mask = Vec::with_capacity(n_features);
        for fh in &cont_histograms {
            for bin in &fh.bins {
                grad_sums.push(bin.grad_sum);
                hess_sums.push(bin.hess_sum);
                counts.push(bin.count);
            }
            continuous_mask.push(1u8);
        }

        // --- Options uniform ---
        let opts_pod = SplitOptionsPod {
            missing_bin_index: options.missing_bin_index as u32,
            l1_alpha: options.l1_alpha,
            l2_lambda: options.l2_lambda,
            min_child_hessian: options.min_child_hessian,
            min_leaf_magnitude: options.min_leaf_magnitude,
        };

        // --- Pipeline lookup (keyed on bin_count + L1-enabled branch) ---
        let l1_enabled = options.l1_alpha > 0.0;
        let pipeline = pipeline_cache
            .get_or_build(bin_count as u32, l1_enabled)
            .map_err(EngineError::BackendUnavailable)?;

        let device = &metal_device.device;
        let res_options = MTLResourceOptions::StorageModeShared;

        // --- Inputs: reuse slots in BufferCache ---
        let grad_buffer = buffer_cache.write_split_grad(device, &grad_sums)?;
        let hess_buffer = buffer_cache.write_split_hess(device, &hess_sums)?;
        let counts_buffer = buffer_cache.write_split_counts(device, &counts)?;
        let mask_buffer = buffer_cache.write_continuous_mask(device, &continuous_mask)?;

        // --- Output: fresh per-call since n_features can change ---
        let out_bytes = n_features * std::mem::size_of::<FeatureSplitCandidatePod>();
        let out_buffer = device
            .newBufferWithLength_options(out_bytes.max(1), res_options)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "could not allocate split output buffer".to_string(),
                )
            })?;

        // --- Encode, commit, wait ---
        let command_buffer = metal_device
            .queue
            .commandBuffer()
            .ok_or_else(|| EngineError::BackendUnavailable("no command buffer".to_string()))?;
        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("no compute command encoder".to_string())
        })?;
        encoder.setComputePipelineState(&pipeline.per_feature);
        // SAFETY: all buffers outlive the encoder through `Retained`
        // clones; the `opts_pod` stack slot outlives the `setBytes` call
        // because Metal copies synchronously.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&grad_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&hess_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&counts_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&mask_buffer), 0, 3);
            use std::ffi::c_void;
            use std::ptr::NonNull;
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked((&raw const opts_pod) as *mut c_void),
                std::mem::size_of::<SplitOptionsPod>(),
                4,
            );
            encoder.setBuffer_offset_atIndex(Some(&out_buffer), 0, 5);

            let threadgroups = MTLSize {
                width: n_features,
                height: 1,
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
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // --- Readback + cross-feature argmax on CPU ---
        //
        // `feature_weights` may be empty — match CPU semantics (unit
        // weight when out of range).
        // SAFETY: `out_buffer` is StorageModeShared, fully written by
        // the dispatched kernel (which we waited on), and holds
        // `n_features` consecutive `FeatureSplitCandidatePod` values.
        let out_ptr: *const FeatureSplitCandidatePod =
            out_buffer.contents().as_ptr() as *const FeatureSplitCandidatePod;

        let weighted_gain = |feature_index: usize, gain: f32| -> f32 {
            if feature_index < feature_weights.len() {
                gain * feature_weights[feature_index]
            } else {
                gain
            }
        };

        let mut best: Option<SplitCandidate> = None;
        let mut best_weighted: f32 = 0.0;
        for (local_f, fh) in cont_histograms.iter().enumerate() {
            // SAFETY: `local_f < n_features` by construction.
            let pod = unsafe { *out_ptr.add(local_f) };
            if pod.has_split == 0 {
                continue;
            }
            let feature_index = fh.feature_index as usize;
            let this_weighted = weighted_gain(feature_index, pod.gain);

            let take = match best {
                None => true,
                Some(_) => this_weighted > best_weighted,
            };
            if take {
                best_weighted = this_weighted;
                best = Some(SplitCandidate {
                    node_id: histograms.node_id,
                    feature_index: fh.feature_index,
                    threshold_bin: pod.threshold_bin as u16,
                    gain: pod.gain,
                    default_left: pod.default_left != 0,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: pod.left_grad,
                        hess_sum: pod.left_hess,
                        row_count: pod.left_count,
                    },
                    right_stats: NodeStats {
                        grad_sum: pod.right_grad,
                        hess_sum: pod.right_hess,
                        row_count: pod.right_count,
                    },
                });
            }
        }
        // Keep out_buffer alive until readback completed above.
        drop(out_buffer);
        best
    };

    // --- Combine continuous + categorical winners via feature_weights ---
    let weighted_gain = |candidate: &SplitCandidate| -> f32 {
        let fi = candidate.feature_index as usize;
        if fi < feature_weights.len() {
            candidate.gain * feature_weights[fi]
        } else {
            candidate.gain
        }
    };

    let winner = match (cont_candidate, cat_candidate) {
        (None, None) => None,
        (Some(c), None) => Some(c),
        (None, Some(c)) => Some(c),
        (Some(a), Some(b)) => {
            if weighted_gain(&b) > weighted_gain(&a) {
                Some(b)
            } else {
                Some(a)
            }
        }
    };
    Ok(winner)
}

// -------- Non-macOS stub (never called — gated off at the BackendOps layer) -------

#[cfg(not(target_os = "macos"))]
pub fn dispatch_best_split() -> EngineResult<Option<SplitCandidate>> {
    // The non-macOS build of `MetalBackend` never constructs successfully
    // (see `lib.rs`), so this function is unreachable on non-macOS.
    Err(EngineError::BackendUnavailable(
        "Metal split kernel is only available on macOS".to_string(),
    ))
}
