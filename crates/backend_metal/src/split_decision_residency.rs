//! Pool for device-side `SplitDecision` output buffers used by the
//! GPU split-finding kernel (Stage 4a).
//!
//! Each `find_best_splits_batch` call mints a single buffer sized
//! `node_count × size_of::<SplitDecisionGpu>()` (24 bytes per entry).
//! The pool owns the buffer for the call's lifetime; the
//! `SplitDecisionReleaseGuard` RAII helper releases it on Drop, mirroring
//! `HistogramReleaseGuard` from Stage 3.
//!
//! Sibling to `HistogramResidencyPool` rather than shared because the
//! lifecycle differs: histograms persist across multiple levels (parent
//! of subtract), split decisions live exactly one level.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use alloygbm_engine::{EngineError, EngineResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::residency::ResidencyPool;

/// Opaque handle to a minted split-decision buffer. Stored as a
/// monotonically-increasing token; 0 reserved as the "no handle"
/// sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitDecisionHandle(pub(crate) u64);

/// Fixed-size on-device representation of one node's split decision.
/// Must match the MSL `struct SplitDecisionGpu` declared in
/// `shaders/best_split.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SplitDecisionGpu {
    pub feature_idx: u32,   // 0xFFFFFFFF if no valid split found
    pub bin_threshold: u32,
    pub gain: f32,
    pub grad_left: f32,
    pub hess_left: f32,
    pub flags: u32,         // bit 0: missing-goes-right; bit 1: invalid
}

const SPLIT_DECISION_BYTES: usize = std::mem::size_of::<SplitDecisionGpu>();

const _: () = assert!(SPLIT_DECISION_BYTES == 24);

#[allow(dead_code)]
pub(crate) const SPLIT_FLAG_MISSING_GOES_RIGHT: u32 = 1 << 0;
#[allow(dead_code)]
pub(crate) const SPLIT_FLAG_INVALID: u32 = 1 << 1;

struct SplitDecisionEntry {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    node_count: u32,
}

struct PoolState {
    /// Monotonic token counter. `u64` gives > 500 years of fit lifetime
    /// at 1 million mints / second, so wrap-through-zero is not a
    /// practical concern; if it did wrap, the next mint would emit the
    /// reserved-zero sentinel and the next `read_decisions` would fail
    /// with "unknown handle".
    next_token: u64,
    live: HashMap<u64, SplitDecisionEntry>,
}

pub(crate) struct SplitDecisionPool {
    state: Mutex<PoolState>,
}

#[allow(dead_code)]
impl SplitDecisionPool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                next_token: 1,
                live: HashMap::new(),
            }),
        }
    }

    pub(crate) fn mint(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &ResidencyPool,
        node_count: u32,
    ) -> EngineResult<SplitDecisionHandle> {
        let bytes = (node_count as usize)
            .checked_mul(SPLIT_DECISION_BYTES)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "split decision pool: node_count ({node_count}) × 24 overflowed usize"
                ))
            })?;

        // Pad to at least 16 bytes (Metal's minimum allocation alignment).
        let alloc_bytes = bytes.max(16);

        let buffer = device
            .newBufferWithLength_options(alloc_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "split decision pool: device returned None for {alloc_bytes}-byte buffer"
                ))
            })?;

        // SAFETY: buffer is StorageModeShared with `alloc_bytes` of
        // valid memory; we write exactly that many zero bytes so
        // unused slots read as feature_idx = 0, flags = 0 (the
        // kernel must explicitly set flags |= INVALID for "no split").
        // `buffer.contents()` returns `NonNull<c_void>` (non-null by
        // type); Metal allocates `StorageModeShared` buffers at page
        // granularity so the pointer is well-aligned for byte writes.
        // Explicit zero-init is retained for portability; Apple-Silicon
        // Metal already zero-fills new allocations but the explicit
        // `write_bytes` makes the contract independent of platform
        // behaviour.
        unsafe {
            std::ptr::write_bytes(
                buffer.contents().as_ptr() as *mut u8,
                0,
                alloc_bytes,
            );
        }

        residency.add_buffer(&buffer);
        residency.commit();

        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let token = state.next_token;
        state.next_token = state.next_token.wrapping_add(1);
        state.live.insert(
            token,
            SplitDecisionEntry {
                buffer,
                node_count,
            },
        );
        Ok(SplitDecisionHandle(token))
    }

    pub(crate) fn buffer_for(
        &self,
        handle: SplitDecisionHandle,
    ) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let entry = state.live.get(&handle.0).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "split decision pool: buffer_for called on unknown handle {:?}",
                handle.0
            ))
        })?;
        Ok(entry.buffer.clone())
    }

    pub(crate) fn read_decisions(
        &self,
        handle: SplitDecisionHandle,
    ) -> EngineResult<Vec<SplitDecisionGpu>> {
        let state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let entry = state.live.get(&handle.0).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "split decision pool: read_decisions called on unknown handle {:?}",
                handle.0
            ))
        })?;
        let n = entry.node_count as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: buffer is StorageModeShared with `n × 24` valid
        // bytes; caller must have waited on the producing CB.
        // `buffer.contents()` returns `NonNull<c_void>` (non-null by
        // type); Metal's page-aligned allocation guarantees alignment
        // ≥ `align_of::<SplitDecisionGpu>()` (4 bytes — all fields are
        // 4-byte primitives).
        let ptr = entry.buffer.contents().as_ptr() as *const SplitDecisionGpu;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(slice.to_vec())
    }

    pub(crate) fn release(
        &self,
        residency: &ResidencyPool,
        handle: SplitDecisionHandle,
    ) -> EngineResult<()> {
        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        if let Some(entry) = state.live.remove(&handle.0) {
            residency.remove_buffer(&entry.buffer);
            residency.commit();
            // `entry` drops here — the Retained<> refcount decrements
            // and the backing allocation returns to Metal's pool.
        }
        Ok(())
    }
}

// SAFETY: all mutable state is behind a Mutex. The Retained<...>
// Metal handles stored in entries are thread-safe per Apple's
// `MTLBuffer` contract.
unsafe impl Send for SplitDecisionPool {}
// SAFETY: see Send impl.
unsafe impl Sync for SplitDecisionPool {}

pub(crate) struct SplitDecisionReleaseGuard<'a> {
    pool: &'a SplitDecisionPool,
    residency: &'a ResidencyPool,
    handle: SplitDecisionHandle,
}

#[allow(dead_code)]
impl<'a> SplitDecisionReleaseGuard<'a> {
    pub(crate) fn new(
        pool: &'a SplitDecisionPool,
        residency: &'a ResidencyPool,
        handle: SplitDecisionHandle,
    ) -> Self {
        Self { pool, residency, handle }
    }
}

impl Drop for SplitDecisionReleaseGuard<'_> {
    fn drop(&mut self) {
        let _ = self.pool.release(self.residency, self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn try_probe_device() -> Option<MetalDevice> {
        MetalDevice::probe().ok()
    }

    #[test]
    fn mint_release_round_trip() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let residency = ResidencyPool::new(
            &metal_device.device,
            metal_device.queue.clone(),
            "alloygbm::split-decision-residency::test",
        );
        let pool = SplitDecisionPool::new();
        let handle = pool.mint(&metal_device.device, &residency, 8).unwrap();
        let decisions = pool.read_decisions(handle).unwrap();
        assert_eq!(decisions.len(), 8);
        for d in &decisions {
            assert_eq!(d.feature_idx, 0);
            assert_eq!(d.flags, 0);
        }
        pool.release(&residency, handle).unwrap();
        // double-release is a no-op
        pool.release(&residency, handle).unwrap();
    }

    #[test]
    fn mint_zero_node_count_is_safe() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let residency = ResidencyPool::new(
            &metal_device.device,
            metal_device.queue.clone(),
            "alloygbm::split-decision-residency::test-zero",
        );
        let pool = SplitDecisionPool::new();
        let handle = pool.mint(&metal_device.device, &residency, 0).unwrap();
        let decisions = pool.read_decisions(handle).unwrap();
        assert!(decisions.is_empty());
        pool.release(&residency, handle).unwrap();
    }
}
