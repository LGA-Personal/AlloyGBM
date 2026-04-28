//! Stage 4b: ICB-based full-tree GPU encoding and reconstruction.
//!
//! `IcbTreeEncoder::encode_and_run` encodes depth×3 ICB commands, commits
//! the outer command buffer, waits once, then returns reconstructed
//! `Vec<TrainedStump>` + the candidate_predictions update.

#![cfg(target_os = "macos")]
#![allow(unsafe_code)]

use std::mem::size_of;

use alloygbm_core::{BinnedMatrix, SplitCandidate, NodeStats, TrainParams};
use alloygbm_engine::{
    EngineError, EngineResult, IterationStopReason, SplitSelectionOptions, TrainedStump,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use crate::icb_buffer_pool::{IcbBufferPool, IcbConstantsGpu, IcbSplitDecisionGpu};
use crate::pipelines::IcbPipelineCache;
use crate::profile;

/// Parameters extracted from `TrainParams` for ICB encoding.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IcbTreeParams {
    pub depth:             u8,
    pub feature_count:     u32,
    pub bin_count:         u32,
    pub row_count:         u32,
    pub min_split_gain:    f32,
    pub lambda:            f32,
    pub learning_rate:     f32,
    pub min_rows_per_leaf: u32,
}

impl IcbTreeParams {
    pub(crate) fn from_train_params(p: &TrainParams, bm: &BinnedMatrix) -> Self {
        Self {
            depth:             p.max_depth as u8,
            feature_count:     bm.feature_count as u32,
            bin_count:         bm.max_bin as u32 + 1,
            row_count:         bm.row_count as u32,
            min_split_gain:    p.min_split_gain,
            lambda:            p.lambda_l2,
            learning_rate:     p.learning_rate,
            min_rows_per_leaf: p.min_data_in_leaf,
        }
    }
}

/// Encodes and submits one full tree as a single ICB-backed command buffer.
pub(crate) struct IcbTreeEncoder {
    pipeline_cache: IcbPipelineCache,
    icb:            Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    queue:          Retained<ProtocolObject<dyn MTLCommandQueue>>,
    depth_max:      u8,
}

// SAFETY: all Retained<> Metal handles are thread-safe per Apple docs.
unsafe impl Send for IcbTreeEncoder {}
unsafe impl Sync for IcbTreeEncoder {}

impl IcbTreeEncoder {
    /// Create an encoder. Compiles three PSOs and allocates the ICB for
    /// `depth_max × 3` commands.
    pub(crate) fn new(
        device:    &ProtocolObject<dyn MTLDevice>,
        queue:     Retained<ProtocolObject<dyn MTLCommandQueue>>,
        depth_max: u8,
    ) -> EngineResult<Self> {
        let pipeline_cache = IcbPipelineCache::new(device)
            .map_err(EngineError::BackendUnavailable)?;

        let cmd_count = (depth_max as usize) * 3;
        let icb_desc = MTLIndirectCommandBufferDescriptor::new();
        icb_desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        icb_desc.setInheritBuffers(false);
        icb_desc.setInheritPipelineState(false);
        icb_desc.setMaxKernelBufferBindCount(8);

        // SAFETY: newIndirectCommandBufferWithDescriptor_maxCommandCount_options is
        // documented safe with a valid descriptor and non-zero count.
        let icb = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &icb_desc,
                cmd_count,
                MTLResourceOptions::empty(),
            )
        }
        .ok_or_else(|| EngineError::BackendUnavailable(
            "IcbTreeEncoder: MTLIndirectCommandBuffer alloc failed".to_string()))?;

        Ok(Self { pipeline_cache, icb, queue, depth_max })
    }

    /// Encode `depth` levels of (histogram, split_find, partition) ICB commands,
    /// submit as a single outer MTLCommandBuffer, wait, then return stumps +
    /// prediction updates.
    ///
    /// The caller is responsible for calling `pool.reset_for_tree(root_row_indices)`
    /// and `pool.upload_gradients(grads, hess)` before calling this function.
    pub(crate) fn encode_and_run(
        &self,
        pool:                  &IcbBufferPool,
        params:                &IcbTreeParams,
        bin_data_buf:          &ProtocolObject<dyn MTLBuffer>,
        root_row_indices:      &[u32],
        candidate_predictions: &mut [f32],
        split_options:         SplitSelectionOptions,
    ) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
        let _p_total = profile::ScopedProbe::new(&profile::ICB_TREE);

        let depth = params.depth as usize;
        let tg_rows  = 256usize;
        let tg_nodes = 32usize;

        {
            let _p_enc = profile::ScopedProbe::new(&profile::ICB_ENCODE);

            // ── 0. Upload per-level constants to shared buffer ────────────────
            let mut all_consts: Vec<IcbConstantsGpu> = Vec::with_capacity(depth);
            for level in 0..depth {
                let node_offset = (1u32 << level) - 1;
                let node_count  = 1u32 << level;
                let node_end    = node_offset + node_count;
                all_consts.push(IcbConstantsGpu {
                    row_count:         params.row_count,
                    feature_count:     params.feature_count,
                    bin_count:         params.bin_count,
                    level_node_offset: node_offset,
                    level_node_end:    node_end,
                    level_node_count:  node_count,
                    min_rows_per_leaf: params.min_rows_per_leaf,
                    min_split_gain:    split_options.l2_lambda.max(params.min_split_gain),
                    lambda:            split_options.l2_lambda,
                    learning_rate:     params.learning_rate,
                    _pad0:             0,
                    _pad1:             0,
                });
            }
            pool.upload_constants(&all_consts);

            // ── 1. Encode ICB commands ────────────────────────────────────────
            for level in 0..depth {
                let node_count    = 1u32 << level;
                let consts_offset = level * size_of::<IcbConstantsGpu>();

                // Zero histogram buffer for this level before encoding.
                pool.zero_histograms(node_count as usize);

                let base_cmd = level * 3;

                // Command 0: icb_histogram
                // SAFETY: indirectComputeCommandAtIndex is documented safe for
                // indices within the ICB's command count.
                let hist_cmd = unsafe { self.icb.indirectComputeCommandAtIndex(base_cmd) };
                // SAFETY: All setComputePipelineState / setKernelBuffer_offset_atIndex
                // calls on ICB commands are documented safe for pre-encoded Metal ICBs.
                unsafe {
                    hist_cmd.setComputePipelineState(&self.pipeline_cache.histogram);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.row_node_id,  0,             0);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.node_active,  0,             1);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.gradients,    0,             2);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.hessians,     0,             3);
                    hist_cmd.setKernelBuffer_offset_atIndex(bin_data_buf,       0,             4);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.histograms,   0,             5);
                    hist_cmd.setKernelBuffer_offset_atIndex(&pool.constants,    consts_offset, 6);
                }
                let rows_tg_count = (params.row_count as usize + tg_rows - 1) / tg_rows;
                hist_cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: rows_tg_count, height: 1, depth: 1 },
                    MTLSize { width: tg_rows, height: 1, depth: 1 },
                );

                // Command 1: icb_split_find
                let sf_cmd = unsafe { self.icb.indirectComputeCommandAtIndex(base_cmd + 1) };
                unsafe {
                    sf_cmd.setComputePipelineState(&self.pipeline_cache.split_find);
                    sf_cmd.setKernelBuffer_offset_atIndex(&pool.histograms,      0,             0);
                    sf_cmd.setKernelBuffer_offset_atIndex(&pool.split_decisions, 0,             1);
                    sf_cmd.setKernelBuffer_offset_atIndex(&pool.node_active,     0,             2);
                    sf_cmd.setKernelBuffer_offset_atIndex(&pool.leaf_values,     0,             3);
                    sf_cmd.setKernelBuffer_offset_atIndex(&pool.constants,       consts_offset, 4);
                }
                let nodes_tg_count = (node_count as usize + tg_nodes - 1) / tg_nodes;
                sf_cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nodes_tg_count, height: 1, depth: 1 },
                    MTLSize { width: tg_nodes, height: 1, depth: 1 },
                );

                // Command 2: icb_partition (last level = no-op zero-width dispatch)
                let part_cmd = unsafe { self.icb.indirectComputeCommandAtIndex(base_cmd + 2) };
                unsafe {
                    part_cmd.setComputePipelineState(&self.pipeline_cache.partition);
                    part_cmd.setKernelBuffer_offset_atIndex(&pool.row_node_id,     0,             0);
                    part_cmd.setKernelBuffer_offset_atIndex(&pool.node_active,     0,             1);
                    part_cmd.setKernelBuffer_offset_atIndex(&pool.split_decisions, 0,             2);
                    part_cmd.setKernelBuffer_offset_atIndex(bin_data_buf,          0,             3);
                    part_cmd.setKernelBuffer_offset_atIndex(&pool.constants,       consts_offset, 4);
                }
                let part_width = if level < depth - 1 { rows_tg_count } else { 0 };
                part_cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: part_width, height: 1, depth: 1 },
                    MTLSize { width: tg_rows, height: 1, depth: 1 },
                );
            }

            // ── 2. Encode outer MTLCommandBuffer ─────────────────────────────
            let cmd_buf = self.queue.commandBuffer().ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "IcbTreeEncoder: commandBuffer() returned nil".to_string())
            })?;
            let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "IcbTreeEncoder: computeCommandEncoder() returned nil".to_string())
            })?;

            // Declare heap residency: all pool buffers inherit from the heap.
            encoder.useHeap(&pool.heap);

            // Declare bin_data_buf residency: it is not in the pool heap.
            // SAFETY: MTLBuffer is a subprotocol of MTLResource in ObjC; both
            // traits share the same vtable pointer layout for the ObjC runtime,
            // so the pointer cast is valid. The cast is only used to satisfy the
            // Rust type system — the underlying ObjC method receives the same
            // object pointer either way.
            unsafe {
                let resource_ptr = bin_data_buf as *const ProtocolObject<dyn MTLBuffer>
                    as *const ProtocolObject<dyn MTLResource>;
                encoder.useResource_usage(&*resource_ptr, MTLResourceUsage::Read);
            }

            use objc2_foundation::NSRange;
            for level in 0..depth {
                let base_cmd = level * 3;

                unsafe {
                    // Histogram
                    encoder.executeCommandsInBuffer_withRange(
                        &self.icb,
                        NSRange { location: base_cmd, length: 1 },
                    );
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);

                    // Split-find
                    encoder.executeCommandsInBuffer_withRange(
                        &self.icb,
                        NSRange { location: base_cmd + 1, length: 1 },
                    );
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);

                    // Partition (may be zero-dispatch on last level)
                    encoder.executeCommandsInBuffer_withRange(
                        &self.icb,
                        NSRange { location: base_cmd + 2, length: 1 },
                    );
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                }
            }

            encoder.endEncoding();

            {
                let _p_sub = profile::ScopedProbe::new(&profile::ICB_SUBMIT);
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            }
        }

        // ── 3. Reconstruct tree and update predictions ────────────────────────
        let _p_rb = profile::ScopedProbe::new(&profile::ICB_READBACK);

        let decisions    = pool.read_decisions();
        let leaf_values  = pool.read_leaf_values();
        let row_node_ids = pool.read_row_node_ids();

        let (stumps, stop_reason) = reconstruct_tree_from_icb(
            &decisions,
            &leaf_values,
            params,
            split_options,
        );

        update_candidate_predictions(
            candidate_predictions,
            root_row_indices,
            &row_node_ids,
            &decisions,
            &leaf_values,
            params,
            split_options,
        );

        Ok((stumps, stop_reason))
    }
}

/// Walk the level-wise node tree and build `Vec<TrainedStump>`.
fn reconstruct_tree_from_icb(
    decisions:   &[IcbSplitDecisionGpu],
    leaf_values: &[f32],
    params:      &IcbTreeParams,
    options:     SplitSelectionOptions,
) -> (Vec<TrainedStump>, IterationStopReason) {
    let depth = params.depth as usize;
    let lambda = options.l2_lambda;
    let lr     = params.learning_rate;
    let mut stumps = Vec::new();
    let mut found_any = false;

    for level in 0..depth {
        let node_start = (1usize << level) - 1;
        let node_end   = node_start + (1usize << level);
        for n in node_start..node_end.min(decisions.len()) {
            let d = &decisions[n];
            if d.feature_idx == 0xFFFF_FFFFu32 {
                continue;
            }
            found_any = true;

            let left_leaf_value  = -lr * d.grad_left / (d.hess_left + lambda + 1e-9);
            let grad_right       = d.grad_total - d.grad_left;
            let hess_right       = d.hess_total - d.hess_left;
            let right_leaf_value = -lr * grad_right / (hess_right + lambda + 1e-9);

            let parent_leaf = leaf_from_parent(n, decisions, lr, lambda);

            let split = SplitCandidate {
                node_id:            n as u32,
                feature_index:      d.feature_idx,
                threshold_bin:      d.threshold_bin as u16,
                gain:               d.gain,
                default_left:       (d.flags & 1u32) == 0,
                is_categorical:     false,
                categorical_bitset: None,
                left_stats:  NodeStats {
                    grad_sum:  d.grad_left,
                    hess_sum:  d.hess_left,
                    row_count: 0,
                },
                right_stats: NodeStats {
                    grad_sum:  grad_right,
                    hess_sum:  hess_right,
                    row_count: 0,
                },
            };
            stumps.push(TrainedStump {
                split,
                left_leaf_value:  left_leaf_value  - parent_leaf,
                right_leaf_value: right_leaf_value - parent_leaf,
            });
        }
    }

    let _ = leaf_values;

    let stop_reason = if found_any {
        IterationStopReason::DepthBudgetReached
    } else {
        IterationStopReason::NoSplitCandidate
    };
    (stumps, stop_reason)
}

/// Recurse up the parent chain to compute the absolute leaf value at node `n`.
fn leaf_from_parent(
    n:         usize,
    decisions: &[IcbSplitDecisionGpu],
    lr:        f32,
    lambda:    f32,
) -> f32 {
    if n == 0 { return 0.0; }
    let p       = (n - 1) / 2;
    let is_left = n % 2 == 1;
    let dp      = &decisions[p];
    if dp.feature_idx == 0xFFFF_FFFFu32 { return 0.0; }
    let val = if is_left {
        -lr * dp.grad_left / (dp.hess_left + lambda + 1e-9)
    } else {
        let g = dp.grad_total - dp.grad_left;
        let h = dp.hess_total - dp.hess_left;
        -lr * g / (h + lambda + 1e-9)
    };
    val + leaf_from_parent(p, decisions, lr, lambda)
}

/// Apply the ICB tree's leaf deltas to `candidate_predictions`.
fn update_candidate_predictions(
    candidate_predictions: &mut [f32],
    root_row_indices:      &[u32],
    row_node_ids:          &[u16],
    decisions:             &[IcbSplitDecisionGpu],
    leaf_values:           &[f32],
    params:                &IcbTreeParams,
    options:               SplitSelectionOptions,
) {
    let lambda = options.l2_lambda;
    let lr     = params.learning_rate;

    for &r in root_row_indices {
        let n = row_node_ids[r as usize] as usize;
        if n >= decisions.len() { continue; }
        let d = &decisions[n];
        let delta = if d.feature_idx == 0xFFFF_FFFFu32 {
            leaf_values[n]
        } else {
            let g = d.grad_total;
            let h = d.hess_total;
            -lr * g / (h + lambda + 1e-9)
        };
        candidate_predictions[r as usize] += delta;
    }
}
