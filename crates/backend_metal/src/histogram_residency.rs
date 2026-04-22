//! GPU-resident histogram pool (S3.7b skeleton).
//!
//! # What this is
//!
//! Stage 1 produced `HistogramBundle`s whose `grad_sum` / `hess_sum` /
//! `counts` arrays lived on the CPU, even though `build_histograms` ran
//! on the GPU. The kernel wrote into a shared-mode scratch buffer, a
//! memcpy produced a `Vec<HistogramBin>`, and the very next
//! `best_split` call uploaded those bytes back to a shared-mode buffer
//! on the GPU. That round-trip — ~F × B × 12 bytes per `best_split` —
//! was the dominant Stage-2 dispatch cost identified in the
//! post-Stage-2 benchmark postmortem.
//!
//! Stage 3's fix is to keep the histogram bytes GPU-resident across
//! the level. `HistogramResidencyPool` owns the GPU buffers backing
//! live `GpuHistogramHandle` tokens: [`mint`] allocates a triple of
//! `(grad, hess, counts)` buffers, registers them with the backend's
//! [`ResidencyPool`] so the working set keeps them warm, and hands
//! back an opaque `GpuHistogramHandle(u64)` for the engine to thread
//! through the storage-enum variants. [`release`] detaches the entry
//! from residency and drops the buffers back to the allocator (or,
//! in a future iteration, returns them to a same-shape free-list).
//!
//! # Scope of this skeleton (S3.7b)
//!
//! This file ships the lifecycle scaffolding only. `build_histograms`
//! still produces CPU-side `HistogramStorage::Cpu(..)` bundles today;
//! the kernel-writes-into-pool-entry plumbing lands in S3.7c. Shipping
//! the pool in isolation keeps the change reviewable and lets the
//! engine-side trainer refactor (S3.3) land against a stable pool API.
//!
//! # Free-on-consume discipline (D-016)
//!
//! Memory policy is "free-on-consume". The trainer calls [`release`]
//! on an entry the moment the histogram is no longer needed —
//! level-wise that means after both children's best-splits have been
//! computed; leaf-wise that means when the matching `PendingSplit`
//! pops. The pool never implicitly frees entries; a missing
//! [`release`] call is a leak, not a bug that self-corrects. The
//! budget guard in `budget.rs` projects peak residency under this
//! discipline at fit start and refuses pathological shapes.
//!
//! # Concurrency
//!
//! The internal map is behind a `Mutex`. `mint` and `release` are
//! rare (once per histogram bundle, once per level-transition) so the
//! lock cost is trivial against a dispatch-bound workload. The
//! `Retained<ProtocolObject<dyn MTLBuffer>>` handles themselves are
//! documented thread-safe, so cloned handles handed to a kernel
//! dispatch outlive the mutex-guarded lookup path without
//! ceremony.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use alloygbm_core::GpuHistogramHandle;
use alloygbm_engine::{EngineError, EngineResult};

use crate::residency::ResidencyPool;

/// Per-buffer size multiplier — each of `grad`, `hess`, `counts` is
/// 4 bytes per cell (f32 / f32 / u32). The combined 12 bytes/cell
/// figure is owned by `budget::HISTOGRAM_CELL_BYTES` so the budget
/// projection and the allocator stay consistent; this constant is the
/// per-plane view the pool needs when sizing one of three parallel
/// buffers.
const BYTES_PER_CELL: usize = 4;

/// Shape snapshot of one live entry. Tracked so lookups can return
/// both the buffers and the extent the kernel should bind to.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct HistogramShape {
    pub feature_count: u32,
    pub bin_count: u32,
}

/// Borrowed view of one live pool entry. Cloned `Retained` handles
/// remain live until the `HistogramEntry` is dropped; the caller can
/// hand them to a kernel dispatch without holding the pool mutex.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct HistogramEntry {
    pub grad: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub hess: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub counts: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub shape: HistogramShape,
}

struct PoolState {
    /// Monotonic token counter. `u64` gives > 500 years of fit
    /// lifetime at 1 million mints / second, so overflow is not a
    /// practical concern.
    next_token: u64,
    live: HashMap<u64, HistogramEntry>,
}

/// GPU-resident histogram pool. Owned by [`crate::MetalBackend`];
/// handed mutable references to `build_histograms` once S3.7c wires
/// the kernel output into pool entries.
///
/// `dead_code` allow at the struct + inherent-impl level: the public
/// consumer is S3.7c (kernel output goes straight into `mint`'d
/// buffers) + S3.3 (trainer threads `HistogramStorage::Gpu(..)`
/// through). The skeleton ships with unit-test coverage so S3.7c can
/// land against a stable API.
#[allow(dead_code)]
pub(crate) struct HistogramResidencyPool {
    state: Mutex<PoolState>,
}

#[allow(dead_code)]
impl HistogramResidencyPool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                next_token: 1, // reserve 0 as the "no handle" sentinel
                live: HashMap::new(),
            }),
        }
    }

    /// Allocate a fresh `(grad, hess, counts)` triple sized for
    /// `feature_count × bin_count`, register it with `residency`,
    /// and return the opaque handle.
    ///
    /// All three buffers are allocated with
    /// `StorageModeShared` so the `build_histograms` kernel can
    /// write into them and subsequent kernels (`best_split`,
    /// `subtract`) can read from them without a CPU round-trip.
    /// On Apple-Silicon UMA this is the canonical zero-copy
    /// layout.
    ///
    /// Returns `EngineError::BackendUnavailable` on any allocation
    /// failure, consistent with the rest of the Metal backend's
    /// error surface.
    pub(crate) fn mint(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &ResidencyPool,
        feature_count: u32,
        bin_count: u32,
    ) -> EngineResult<GpuHistogramHandle> {
        // Overflow-check the whole byte-budget chain: each plane is
        // `feature_count × bin_count × 4`. Checking only the cell
        // count isn't enough — `cells * 4` can still wrap after a
        // clean `feature × bin` product. `checked_mul` through the
        // whole chain catches both.
        let plane_bytes = (feature_count as usize)
            .checked_mul(bin_count as usize)
            .and_then(|c| c.checked_mul(BYTES_PER_CELL))
            .ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "histogram residency pool: feature_count ({feature_count}) × \
                     bin_count ({bin_count}) × 4 bytes overflowed `usize`"
                ))
            })?;

        let grad = allocate_zero_shared(device, plane_bytes, "grad")?;
        let hess = allocate_zero_shared(device, plane_bytes, "hess")?;
        let counts = allocate_zero_shared(device, plane_bytes, "counts")?;

        // Register with residency before exposing the handle — the
        // engine must never see a handle whose buffers aren't pinned.
        residency.add_buffer(&grad);
        residency.add_buffer(&hess);
        residency.add_buffer(&counts);
        residency.commit();

        let entry = HistogramEntry {
            grad,
            hess,
            counts,
            shape: HistogramShape {
                feature_count,
                bin_count,
            },
        };

        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("histogram residency pool poisoned: {e}"))
        })?;
        let token = state.next_token;
        state.next_token = state.next_token.wrapping_add(1);
        state.live.insert(token, entry);
        Ok(GpuHistogramHandle(token))
    }

    /// Look up the entry for `handle`. Returns a cloned `HistogramEntry`
    /// whose `Retained` buffer handles outlive the mutex release —
    /// callers can hand the buffers to a kernel dispatch without
    /// holding the pool lock.
    ///
    /// Returns `None` if the handle was already released or never
    /// belonged to this pool. A calling-site that relies on the
    /// entry being live should escalate to a hard error: a
    /// use-after-release means the trainer's handle-lifetime
    /// discipline has drifted from the free-on-consume contract.
    pub(crate) fn get(&self, handle: GpuHistogramHandle) -> Option<HistogramEntry> {
        let state = self.state.lock().ok()?;
        state.live.get(&handle.0).cloned()
    }

    /// Release `handle` from the pool, detaching its buffers from
    /// `residency` and dropping the `Retained` handles. No-op on
    /// an unknown handle — the semantics match a `HashMap::remove`.
    ///
    /// Future iterations can swap the drop for a same-shape
    /// free-list push; the public contract (handle becomes invalid
    /// after release) does not change.
    pub(crate) fn release(&self, handle: GpuHistogramHandle, residency: &ResidencyPool) {
        let Ok(mut state) = self.state.lock() else {
            // A poisoned pool means some other thread panicked
            // while holding the lock. We refuse to silently leak,
            // but we also can't do anything useful from here —
            // this branch is effectively dead under normal
            // operation.
            return;
        };
        if let Some(entry) = state.live.remove(&handle.0) {
            residency.remove_buffer(&entry.grad);
            residency.remove_buffer(&entry.hess);
            residency.remove_buffer(&entry.counts);
            residency.commit();
            // `entry` drops here — the Retained<> refcount decrements
            // and the backing allocations return to Metal's pool.
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
unsafe impl Send for HistogramResidencyPool {}
// SAFETY: see Send impl.
unsafe impl Sync for HistogramResidencyPool {}

/// Allocate a zero-initialized shared-mode buffer. Metal's
/// `newBufferWithLength_options` zero-fills new allocations on
/// Apple Silicon; we rely on that so the histogram kernel can do
/// additive atomics into fresh bins without an extra clear pass.
///
/// The `what` argument is only used to format a human-readable error
/// message on allocation failure.
fn allocate_zero_shared(
    device: &ProtocolObject<dyn MTLDevice>,
    len_bytes: usize,
    what: &str,
) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    // Metal refuses a zero-length allocation; round up to 1 so
    // degenerate-shape fits (feature_count=0 edge case) still
    // produce a valid handle for the engine to drop.
    let len = len_bytes.max(1);
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "histogram residency pool: failed to allocate {what} buffer of {len} bytes"
            ))
        })
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
            "alloygbm::histogram-residency::test",
        );
        let pool = HistogramResidencyPool::new();

        let handle = pool
            .mint(&metal_device.device, &residency, 16, 255)
            .expect("mint should succeed on a realistic shape");
        assert_eq!(pool.live_count(), 1);

        let entry = pool.get(handle).expect("handle should resolve");
        assert_eq!(entry.shape.feature_count, 16);
        assert_eq!(entry.shape.bin_count, 255);

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
        let pool = HistogramResidencyPool::new();

        let h1 = pool
            .mint(&metal_device.device, &residency, 4, 16)
            .expect("first mint");
        let h2 = pool
            .mint(&metal_device.device, &residency, 4, 16)
            .expect("second mint");
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
        let pool = HistogramResidencyPool::new();

        // Handle that was never minted — release should quietly drop it.
        pool.release(GpuHistogramHandle(9999), &residency);
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn mint_rejects_overflow_shape() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let residency = ResidencyPool::disabled(metal_device.queue.clone());
        let pool = HistogramResidencyPool::new();

        // u32::MAX × u32::MAX overflows a usize even on 64-bit —
        // the guard should catch it and return an error rather
        // than silently allocating a wrapped-around size.
        let err = pool
            .mint(&metal_device.device, &residency, u32::MAX, u32::MAX)
            .expect_err("shape overflow must be rejected");
        assert!(
            matches!(err, EngineError::BackendUnavailable(ref msg) if msg.contains("overflowed")),
            "expected overflow-diagnostic, got {err:?}"
        );
        assert_eq!(pool.live_count(), 0);
    }
}
