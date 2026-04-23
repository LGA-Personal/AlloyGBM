//! GPU-resident row-index pool (S3.7e.2a).
//!
//! # What this is
//!
//! Stage 1 + Stage 2's `apply_split` produced CPU-resident
//! `PartitionResult` values: the partition kernel wrote row indices
//! into shared-mode MTLBuffers and immediately copied them back into
//! `Vec<u32>`s for the trainer. The next call — `build_histograms` on
//! a child node — then re-uploaded those same u32s into a shared-mode
//! buffer via `BufferCache::write_row_indices`. On a depth-8 level-
//! wise tree that round-trip runs ~128 times per fit-level; on
//! leaf-wise with `max_leaves=256` it runs ~256 times.
//!
//! Stage 3's fix keeps the partition output **GPU-resident** in a
//! buffer owned by this pool. `HistogramResidencyPool` handed out
//! opaque `GpuHistogramHandle(u64)` tokens against a triple of
//! `(grad, hess, counts)` buffers; the row-index pool follows the
//! same pattern with a single buffer per handle — row indices are
//! just `[u32; row_count]`.
//!
//! # Pool lifecycle
//!
//! `mint(device, residency, buffer) -> GpuRowIndexHandle`:
//!   * accepts an *already-allocated* `MTLBuffer` (the partition
//!     kernel sizes its scratch buffers to the worst case and we
//!     don't want to re-allocate for residency — we just take
//!     ownership of the existing buffer);
//!   * registers the buffer with the working-set `ResidencyPool`;
//!   * returns an opaque handle the engine can thread through
//!     `RowIndexStorage::Gpu { handle, row_count }`.
//!
//! `get(handle) -> Option<RowIndexEntry>` returns a cloned
//! `Retained` handle to the buffer plus the declared row count —
//! callers that need to bind the buffer to a kernel dispatch or
//! read back on the host go through this path.
//!
//! `release(handle, residency)` detaches the entry from residency
//! and drops the `Retained` handle. On a use-after-release pattern
//! the semantics match `HashMap::remove` — a no-op rather than a
//! panic. The engine's `RowIndexReleaseGuard` (S3.7e.1) calls this
//! at every "row indices no longer needed" site.
//!
//! # Why this is a separate pool from `HistogramResidencyPool`
//!
//! Histogram handles and row-index handles have different
//! lifetimes: a histogram lives one level; a child's row indices
//! live until the child is split (one level), but the resulting
//! grandchild row indices start their life on the *same fit* with
//! no histogram counterpart yet. Keeping the two pools separate
//! gives each a clean "free-on-consume" discipline and matches the
//! opaque `GpuRowIndexHandle` / `GpuHistogramHandle` distinction
//! that `core` already models.
//!
//! # Concurrency
//!
//! Mutex-guarded HashMap, same as the histogram pool. Mint + release
//! are rare (once per partition, once per node-retire) so the lock
//! cost is trivial against a dispatch-bound workload.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use alloygbm_core::GpuRowIndexHandle;
use alloygbm_engine::{EngineError, EngineResult};

use crate::residency::ResidencyPool;

/// Cloned view of one live pool entry. `Retained` handles remain
/// live until the cloned entry drops, so callers can hand the
/// buffer to a kernel dispatch without holding the pool mutex.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct RowIndexEntry {
    /// Shared-mode `MTLBuffer` holding exactly `row_count * 4`
    /// useful bytes at the front of the allocation. The buffer's
    /// physical `length` may be larger (sized to the partition
    /// kernel's worst case, which is the parent node's row count);
    /// the pool tracks the logical row count separately via the
    /// `row_count` field on the entry, mirroring `GpuRowIndexHandle`'s
    /// payload in `core::RowIndexStorage::Gpu`.
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Logical row count — the valid prefix of `buffer`.
    pub row_count: u32,
}

struct PoolState {
    /// Monotonic token counter, matches the histogram pool.
    next_token: u64,
    live: HashMap<u64, RowIndexEntry>,
}

/// GPU-resident row-index pool. One per [`crate::MetalBackend`]
/// instance; clones cheaply via `Arc`.
#[allow(dead_code)]
pub(crate) struct RowIndexResidencyPool {
    state: Mutex<PoolState>,
}

#[allow(dead_code)]
impl RowIndexResidencyPool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                next_token: 1, // reserve 0 as the "no handle" sentinel
                live: HashMap::new(),
            }),
        }
    }

    /// Take ownership of `buffer`, register it with `residency`, and
    /// return an opaque handle. `row_count` is the logical length —
    /// the valid prefix of `buffer` that downstream kernels and
    /// readbacks should consider.
    ///
    /// The partition kernel in S3.7e.2b is the first caller: after
    /// the scatter pass, it hands its `out_left_buf` / `out_right_buf`
    /// shared-mode `MTLBuffer`s to this pool along with the known
    /// `total_left` / `total_right` from the `block_left_bases`
    /// sentinel readback. The pool becomes the owner; the partition
    /// dispatch returns `RowIndexStorage::Gpu { handle, row_count }`.
    pub(crate) fn mint(
        &self,
        residency: &ResidencyPool,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        row_count: u32,
    ) -> EngineResult<GpuRowIndexHandle> {
        // Register before minting: the engine must never see a handle
        // whose buffer isn't pinned. Mirror of the histogram pool.
        residency.add_buffer(&buffer);
        residency.commit();

        let entry = RowIndexEntry { buffer, row_count };

        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("row-index residency pool poisoned: {e}"))
        })?;
        let token = state.next_token;
        state.next_token = state.next_token.wrapping_add(1);
        state.live.insert(token, entry);
        Ok(GpuRowIndexHandle(token))
    }

    /// Look up the entry for `handle`. Returns a cloned `RowIndexEntry`
    /// whose `Retained` buffer handle outlives the mutex release —
    /// callers can bind the buffer to a kernel dispatch or read it
    /// back on the host without holding the pool lock.
    ///
    /// Returns `None` if the handle was already released or never
    /// belonged to this pool.
    pub(crate) fn get(&self, handle: GpuRowIndexHandle) -> Option<RowIndexEntry> {
        let state = self.state.lock().ok()?;
        state.live.get(&handle.0).cloned()
    }

    /// Release `handle` from the pool, detaching its buffer from
    /// `residency` and dropping the `Retained` handle. No-op on an
    /// unknown handle — the semantics match a `HashMap::remove`.
    ///
    /// Called by `MetalBackend::release_row_indices` (trait override)
    /// when the engine's `RowIndexReleaseGuard` fires. Per D-016
    /// free-on-consume discipline, every minted handle has exactly
    /// one matching release call on the success path; the trainer's
    /// RAII guards cover continue/break/? escape paths.
    pub(crate) fn release(&self, handle: GpuRowIndexHandle, residency: &ResidencyPool) {
        let Ok(mut state) = self.state.lock() else {
            // See histogram_residency::release — a poisoned pool
            // means another thread panicked holding the lock.
            // Refuse to leak silently, but also can't do useful
            // work from here. Dead under normal operation.
            return;
        };
        if let Some(entry) = state.live.remove(&handle.0) {
            residency.remove_buffer(&entry.buffer);
            residency.commit();
            // `entry` drops here — the Retained<> refcount
            // decrements and the backing allocation returns to
            // Metal's pool.
        }
    }

    /// Diagnostic helper: live-entry count. Unit tests use it to
    /// confirm that `release` actually removes the entry.
    #[cfg(test)]
    pub(crate) fn live_count(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.live.len())
            .unwrap_or(usize::MAX)
    }
}

// SAFETY: all mutable state is behind a Mutex. The Retained<...>
// Metal handles stored in entries are thread-safe per Apple's
// `MTLBuffer` contract.
unsafe impl Send for RowIndexResidencyPool {}
// SAFETY: see Send impl.
unsafe impl Sync for RowIndexResidencyPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;
    use objc2_metal::{MTLDevice, MTLResourceOptions};

    fn try_probe_device() -> Option<MetalDevice> {
        MetalDevice::probe().ok()
    }

    fn alloc_shared(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .expect("alloc shared")
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
            "alloygbm::row-index-residency::test",
        );
        let pool = RowIndexResidencyPool::new();

        let buf = alloc_shared(&metal_device.device, 1024);
        let handle = pool
            .mint(&residency, buf, 100)
            .expect("mint should succeed");
        assert_eq!(pool.live_count(), 1);

        let entry = pool.get(handle).expect("handle should resolve");
        assert_eq!(entry.row_count, 100);

        pool.release(handle, &residency);
        assert_eq!(pool.live_count(), 0);
        assert!(
            pool.get(handle).is_none(),
            "released handle must no longer resolve"
        );
    }

    #[test]
    fn mint_issues_distinct_handles() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let residency = ResidencyPool::disabled(metal_device.queue.clone());
        let pool = RowIndexResidencyPool::new();

        let buf_a = alloc_shared(&metal_device.device, 64);
        let buf_b = alloc_shared(&metal_device.device, 64);
        let h1 = pool.mint(&residency, buf_a, 16).expect("first mint");
        let h2 = pool.mint(&residency, buf_b, 16).expect("second mint");
        assert_ne!(h1, h2, "two mints must produce different handles");
        assert_eq!(pool.live_count(), 2);

        pool.release(h1, &residency);
        pool.release(h2, &residency);
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn release_unknown_handle_is_no_op() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let residency = ResidencyPool::disabled(metal_device.queue.clone());
        let pool = RowIndexResidencyPool::new();

        // Handle that was never minted — release should quietly drop it.
        pool.release(GpuRowIndexHandle(9999), &residency);
        assert_eq!(pool.live_count(), 0);
    }
}
