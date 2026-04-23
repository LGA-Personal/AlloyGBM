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

use alloygbm_core::{FeatureHistogram, HistogramBin, HistogramBundle, NodeStats, SplitCandidate};
use alloygbm_engine::{CategoricalFeatureInfo, EngineError, EngineResult, SplitSelectionOptions};

#[cfg(target_os = "macos")]
use alloygbm_backend_cpu::CpuBackend;
#[cfg(target_os = "macos")]
use alloygbm_core::HistogramStorage;
#[cfg(target_os = "macos")]
use alloygbm_engine::BackendOps;

#[cfg(target_os = "macos")]
use crate::buffers::BufferCache;
#[cfg(target_os = "macos")]
use crate::device::MetalDevice;
#[cfg(target_os = "macos")]
use crate::histogram_residency::HistogramResidencyPool;
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
    histogram_residency: &HistogramResidencyPool,
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

    // Split dispatch must handle both Cpu and Gpu storage (D-015).
    // Rather than forking the whole kernel dispatch path, we first
    // canonicalise the bundle to a `(feature_indices, bin_count,
    // input_source)` triple. The kernel path then dispatches over
    // the FULL feature set (continuous + categorical) with a mask
    // that zeroes categorical slots — the MSL `split.metal` kernel
    // honours `continuous_mask[feature] == 0` by emitting a blank
    // candidate, so binding pool buffers that hold categorical rows
    // too is harmless. This keeps the Gpu arm pool-direct without a
    // CPU round-trip on the hot path.
    let (feature_indices, bin_count): (Vec<u32>, usize) = match &histograms.storage {
        HistogramStorage::Cpu(fhs) => {
            if fhs.is_empty() {
                return Ok(None);
            }
            let bc = fhs[0].bins.len();
            if bc == 0 {
                return Ok(None);
            }
            for fh in fhs {
                if fh.bins.len() != bc {
                    return Err(EngineError::ContractViolation(format!(
                        "Metal best_split requires uniform bin count across features; \
                         expected {}, got {} for feature {}",
                        bc,
                        fh.bins.len(),
                        fh.feature_index
                    )));
                }
            }
            (fhs.iter().map(|fh| fh.feature_index).collect(), bc)
        }
        HistogramStorage::Gpu {
            handle,
            feature_count: _,
            bin_count,
        } => {
            let entry = histogram_residency.get(*handle).ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "Metal best_split: histogram handle {:?} missing from residency pool",
                    handle.0
                ))
            })?;
            if entry.feature_indices.is_empty() || *bin_count == 0 {
                return Ok(None);
            }
            (entry.feature_indices.clone(), *bin_count as usize)
        }
    };
    let n_features = feature_indices.len();
    if (bin_count as u32) > MAX_BIN_COUNT {
        // Out-of-range bin count — fall back to CPU. Only reachable on
        // CPU-origin bundles because the histogram pipeline caps
        // pool-owned bin counts at MAX_BIN_COUNT. The Gpu arm cannot
        // reach this branch, so the unwrap below is safe.
        return cpu.best_split_with_options(
            histograms,
            options,
            feature_weights,
            categorical_features,
        );
    }

    // --- Build continuous-mask over the FULL feature list ---
    let is_categorical_at_pos = |local_f: usize| -> bool {
        let fi = feature_indices[local_f] as usize;
        categorical_features.iter().any(|c| c.feature_index == fi)
    };
    let continuous_mask: Vec<u8> = (0..n_features)
        .map(|i| if is_categorical_at_pos(i) { 0u8 } else { 1u8 })
        .collect();
    let any_continuous = continuous_mask.contains(&1);
    let any_categorical = continuous_mask.contains(&0);

    // --- Categorical features (if any) delegate to CPU per D-012 ---
    let cat_candidate: Option<SplitCandidate> = if any_categorical {
        // Build a CPU sub-bundle containing only the categorical rows.
        let cat_indices: Vec<usize> = (0..n_features)
            .filter(|&i| is_categorical_at_pos(i))
            .collect();
        let cat_fhs: Vec<FeatureHistogram> = match &histograms.storage {
            HistogramStorage::Cpu(fhs) => cat_indices.iter().map(|&i| fhs[i].clone()).collect(),
            HistogramStorage::Gpu { handle, .. } => {
                let planes = histogram_residency.read_planes(*handle)?;
                let bc = planes.bin_count as usize;
                cat_indices
                    .iter()
                    .map(|&i| {
                        let base = i * bc;
                        let bins = (0..bc)
                            .map(|b| HistogramBin {
                                grad_sum: planes.grad[base + b],
                                hess_sum: planes.hess[base + b],
                                count: planes.counts[base + b],
                            })
                            .collect();
                        FeatureHistogram {
                            feature_index: feature_indices[i],
                            bins,
                        }
                    })
                    .collect()
            }
        };
        let cat_bundle = HistogramBundle::from_cpu(histograms.node_id, cat_fhs);
        cpu.best_split_with_options(&cat_bundle, options, feature_weights, categorical_features)?
    } else {
        None
    };

    // --- Continuous features → GPU kernel, full-width dispatch ---
    let cont_candidate: Option<SplitCandidate> = if any_continuous {
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

        // --- Inputs: either pool-direct (Gpu) or flatten-and-upload (Cpu) ---
        //
        // The `_keepalive` Retained<> clones below pin pool buffers
        // to the lifetime of the dispatch. On the Cpu arm they're
        // the buffer-cache slots; on the Gpu arm they're the pool
        // entry's grad/hess/counts buffers, which stay live through
        // the pool as long as the handle isn't released.
        let (grad_buffer, hess_buffer, counts_buffer) = match &histograms.storage {
            HistogramStorage::Cpu(fhs) => {
                let mut grad_sums = Vec::with_capacity(n_features * bin_count);
                let mut hess_sums = Vec::with_capacity(n_features * bin_count);
                let mut counts_vec = Vec::with_capacity(n_features * bin_count);
                for fh in fhs.iter() {
                    for bin in &fh.bins {
                        grad_sums.push(bin.grad_sum);
                        hess_sums.push(bin.hess_sum);
                        counts_vec.push(bin.count);
                    }
                }
                (
                    buffer_cache.write_split_grad(device, &grad_sums)?,
                    buffer_cache.write_split_hess(device, &hess_sums)?,
                    buffer_cache.write_split_counts(device, &counts_vec)?,
                )
            }
            HistogramStorage::Gpu { handle, .. } => {
                let entry = histogram_residency.get(*handle).ok_or_else(|| {
                    EngineError::BackendUnavailable(format!(
                        "Metal best_split: histogram handle {:?} missing from residency pool",
                        handle.0
                    ))
                })?;
                (entry.grad.clone(), entry.hess.clone(), entry.counts.clone())
            }
        };
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
        for local_f in 0..n_features {
            // Categorical slots emit blank candidates per the kernel
            // contract (`has_split=0` when `continuous_mask=0`). The
            // has_split check below also covers them, but skip
            // explicitly to keep intent clear.
            if continuous_mask[local_f] == 0 {
                continue;
            }
            // SAFETY: `local_f < n_features` by construction.
            let pod = unsafe { *out_ptr.add(local_f) };
            if pod.has_split == 0 {
                continue;
            }
            let feature_index = feature_indices[local_f];
            let this_weighted = weighted_gain(feature_index as usize, pod.gain);

            let take = match best {
                None => true,
                Some(_) => this_weighted > best_weighted,
            };
            if take {
                best_weighted = this_weighted;
                best = Some(SplitCandidate {
                    node_id: histograms.node_id,
                    feature_index,
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
    } else {
        None
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
