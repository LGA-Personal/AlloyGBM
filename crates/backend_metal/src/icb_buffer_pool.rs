//! Pre-allocated Metal heap buffer pool for Stage 4b ICB tree execution.
//!
//! All buffers share one MTLHeap so that `encoder.use_heap(&pool.heap)`
//! in the outer command buffer covers them for ICB inherited residency
//! (Metal 4). The pool is constructed once at `MetalBackend::new` (when
//! Metal 4 is present) and reused across all trees and all estimators.

#![cfg(target_os = "macos")]
#![allow(unsafe_code)]

use std::mem::size_of;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLResourceOptions, MTLStorageMode,
};

use alloygbm_engine::{EngineError, EngineResult};

/// 40-byte GPU-side split decision. Must match MSL `SplitDecision` exactly.
/// `feature_idx == 0xFFFF_FFFF` is the "no split" sentinel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IcbSplitDecisionGpu {
    pub feature_idx:    u32,
    pub threshold_bin:  u32,
    pub flags:          u32,   // bit 0 = nan_goes_right
    pub _pad:           u32,
    pub gain:           f32,
    pub grad_left:      f32,
    pub hess_left:      f32,
    pub grad_total:     f32,
    pub hess_total:     f32,
    pub _pad2:          f32,
}

const _: () = assert!(size_of::<IcbSplitDecisionGpu>() == 40);

/// 48-byte push-constant block for ICB kernels. Must match MSL `IcbConstants`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IcbConstantsGpu {
    pub row_count:           u32,
    pub feature_count:       u32,
    pub bin_count:           u32,
    pub level_node_offset:   u32,
    pub level_node_end:      u32,
    pub level_node_count:    u32,
    pub min_rows_per_leaf:   u32,
    pub min_split_gain:      f32,
    pub lambda:              f32,
    pub learning_rate:       f32,
    pub _pad0:               u32,
    pub _pad1:               u32,
}

const _: () = assert!(size_of::<IcbConstantsGpu>() == 48);

/// Pre-allocated heap buffer pool shared across all ICB tree executions.
pub(crate) struct IcbBufferPool {
    /// Current node assignment for every training row. u16; `0xFFFF` = inactive.
    pub row_node_id:      Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-node active flag. u8; 1 = active. Indexed by global node id.
    pub node_active:      Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-node best split. IcbSplitDecisionGpu (40 B). Sentinel = feature_idx 0xFFFF_FFFF.
    pub split_decisions:  Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-node leaf value (f32). Written by icb_split_find when no valid split.
    pub leaf_values:      Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Histogram buffer, reused per level. [level_node_count × F × B × 2] f32.
    /// Sized for the worst case level (half of max_nodes nodes at depth_max-1).
    pub histograms:       Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-row gradients (f32). Uploaded once per tree.
    pub gradients:        Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Per-row hessians (f32). Uploaded once per tree.
    pub hessians:         Retained<ProtocolObject<dyn MTLBuffer>>,
    /// The heap from which all buffers above are sub-allocated.
    pub heap:             Retained<ProtocolObject<dyn MTLHeap>>,
    /// Snapshot of construction parameters.
    pub row_count:        usize,
    pub feature_count:    usize,
    pub bin_count:        usize,
    pub max_nodes:        usize,  // 2^depth_max total nodes (root to leaves)
}

// SAFETY: MTLHeap and MTLBuffer protocol objects are thread-safe per Apple docs.
unsafe impl Send for IcbBufferPool {}
unsafe impl Sync for IcbBufferPool {}

impl IcbBufferPool {
    /// Allocate all buffers for the given training shape.
    ///
    /// `max_depth` is `TrainParams::max_depth` (e.g. 8). The worst-case
    /// histogram level needs `2^(max_depth-1)` nodes × F × B × 2 f32 slots.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        row_count: usize,
        feature_count: usize,
        bin_count: usize,
        max_depth: usize,
    ) -> EngineResult<Self> {
        let max_nodes     = 1usize << max_depth;          // 2^depth total addressable nodes
        let max_level_nodes = max_nodes / 2;              // largest level = 2^(depth-1) nodes

        let row_node_id_bytes    = row_count        * size_of::<u16>();
        let node_active_bytes    = max_nodes        * size_of::<u8>();
        let split_decisions_bytes= max_nodes        * size_of::<IcbSplitDecisionGpu>();
        let leaf_values_bytes    = max_nodes        * size_of::<f32>();
        let histograms_bytes     = max_level_nodes  * feature_count * bin_count * 2 * size_of::<f32>();
        let gradients_bytes      = row_count        * size_of::<f32>();
        let hessians_bytes       = row_count        * size_of::<f32>();

        // Align each allocation to 256 bytes (Metal heap alignment requirement).
        fn align256(n: usize) -> usize { (n + 255) & !255 }
        let total = align256(row_node_id_bytes)
            + align256(node_active_bytes)
            + align256(split_decisions_bytes)
            + align256(leaf_values_bytes)
            + align256(histograms_bytes)
            + align256(gradients_bytes)
            + align256(hessians_bytes)
            + 4096; // guard headroom

        let heap_desc = MTLHeapDescriptor::new();
        heap_desc.setSize(total);
        heap_desc.setStorageMode(MTLStorageMode::Shared);
        let heap = device
            .newHeapWithDescriptor(&heap_desc)
            .ok_or_else(|| EngineError::BackendUnavailable(
                format!("IcbBufferPool: MTLHeap alloc failed ({total} bytes)")))?;

        // Helper: sub-allocate from heap with StorageModeShared.
        let alloc = |bytes: usize, label: &str| -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
            let aligned = align256(bytes).max(256);
            heap.newBufferWithLength_options(aligned, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| EngineError::BackendUnavailable(
                    format!("IcbBufferPool: sub-alloc failed for '{label}' ({aligned} bytes)")))
        };

        let row_node_id     = alloc(row_node_id_bytes,     "row_node_id")?;
        let node_active     = alloc(node_active_bytes,     "node_active")?;
        let split_decisions = alloc(split_decisions_bytes, "split_decisions")?;
        let leaf_values     = alloc(leaf_values_bytes,     "leaf_values")?;
        let histograms      = alloc(histograms_bytes.max(256), "histograms")?;
        let gradients       = alloc(gradients_bytes,       "gradients")?;
        let hessians        = alloc(hessians_bytes,        "hessians")?;

        Ok(Self {
            row_node_id,
            node_active,
            split_decisions,
            leaf_values,
            histograms,
            gradients,
            hessians,
            heap,
            row_count,
            feature_count,
            bin_count,
            max_nodes,
        })
    }

    /// Reset pool buffers for a new tree.
    ///
    /// - `node_active`: zero entire buffer, then set byte 0 (root) = 1.
    /// - `row_node_id`: for every index in `active_rows`, write 0 (root).
    ///   All other entries keep their previous values (irrelevant: kernels
    ///   check `active_rows` bounds or level range).
    /// - `split_decisions`: fill all `feature_idx` fields with 0xFFFF_FFFF sentinel.
    /// - `histograms`: zeroed at the start of each level in the encoder (not here).
    pub(crate) fn reset_for_tree(&self, active_rows: &[u32]) {
        // SAFETY: all Shared-mode buffers have CPU-visible contents() pointers.
        unsafe {
            // Zero node_active, then set root.
            let na_ptr = self.node_active.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(na_ptr, 0u8, self.max_nodes);
            *na_ptr = 1u8;  // root node is active

            // Set row_node_id = 0 for all active rows.
            let rni_ptr = self.row_node_id.contents().as_ptr() as *mut u16;
            for &r in active_rows {
                *rni_ptr.add(r as usize) = 0u16;
            }

            // Fill split_decisions with sentinel (feature_idx = 0xFFFF_FFFF).
            // Each IcbSplitDecisionGpu is 40 bytes; the sentinel is just
            // the first u32. We zero the whole buffer then set feature_idx fields.
            let sd_ptr = self.split_decisions.contents().as_ptr() as *mut IcbSplitDecisionGpu;
            std::ptr::write_bytes(sd_ptr, 0u8, self.max_nodes);  // zero everything
            for i in 0..self.max_nodes {
                (*sd_ptr.add(i)).feature_idx = 0xFFFF_FFFFu32;
            }
        }
    }

    /// Upload gradients and hessians from `GradientPair` slices into the
    /// pre-allocated GPU buffers.
    pub(crate) fn upload_gradients(
        &self,
        grads: &[f32],
        hess: &[f32],
    ) {
        // SAFETY: StorageModeShared buffers have CPU-writable contents().
        unsafe {
            std::ptr::copy_nonoverlapping(
                grads.as_ptr(),
                self.gradients.contents().as_ptr() as *mut f32,
                grads.len(),
            );
            std::ptr::copy_nonoverlapping(
                hess.as_ptr(),
                self.hessians.contents().as_ptr() as *mut f32,
                hess.len(),
            );
        }
    }

    /// Zero the histogram buffer for the current level (called by the
    /// encoder before binding the histogram buffer to a new level's commands).
    pub(crate) fn zero_histograms(&self, level_node_count: usize) {
        let bytes = level_node_count * self.feature_count * self.bin_count * 2 * size_of::<f32>();
        // SAFETY: StorageModeShared, buffer was allocated for this size.
        unsafe {
            std::ptr::write_bytes(
                self.histograms.contents().as_ptr() as *mut u8,
                0u8,
                bytes,
            );
        }
    }

    /// Read back `split_decisions` and `leaf_values` from the GPU buffer.
    pub(crate) fn read_decisions(&self) -> Vec<IcbSplitDecisionGpu> {
        // SAFETY: StorageModeShared; GPU write has completed before this call.
        unsafe {
            let ptr = self.split_decisions.contents().as_ptr() as *const IcbSplitDecisionGpu;
            std::slice::from_raw_parts(ptr, self.max_nodes).to_vec()
        }
    }

    pub(crate) fn read_leaf_values(&self) -> Vec<f32> {
        // SAFETY: same as above.
        unsafe {
            let ptr = self.leaf_values.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.max_nodes).to_vec()
        }
    }

    pub(crate) fn read_row_node_ids(&self) -> Vec<u16> {
        // SAFETY: same as above.
        unsafe {
            let ptr = self.row_node_id.contents().as_ptr() as *const u16;
            std::slice::from_raw_parts(ptr, self.row_count).to_vec()
        }
    }
}
