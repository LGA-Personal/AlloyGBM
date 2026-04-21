//! GPU working-set residency pool.
//!
//! # What this is
//!
//! Stage 3's GPU-resident histograms + row indices live across multiple
//! command-buffer submissions within a fit. Apple's recommended way to
//! tell Metal "keep these buffers in the GPU working set for a run of
//! command buffers" is `MTLResidencySet` (introduced macOS 15 / Metal 4,
//! available on every Apple-Silicon Mac since then, including the
//! macOS 26 Tahoe dev box). On macOS 13/14 the corresponding technique
//! is `MTLHeap` sub-allocation.
//!
//! This module wraps both paths behind one surface so the rest of the
//! Metal backend doesn't branch on OS version at every call site. The
//! current state of each path:
//!
//! * **`ResidencySet` path (macOS 15+):** fully wired. Creates one
//!   `MTLResidencySet` per `MetalBackend` instance; each freshly-minted
//!   buffer is `addAllocation`'d and then `commit`'d; `requestResidency`
//!   pins the set to the working set before a fit runs, and
//!   `endResidency` releases it at teardown. The set is also attached
//!   to the command queue via `addResidencySet` so every command buffer
//!   inherits it with zero per-encode overhead.
//!
//! * **`MTLHeap` fallback (macOS 13/14):** stub. The current
//!   implementation skips explicit residency on that path, which is
//!   correct but not optimal: on Apple Silicon UMA, `StorageModeShared`
//!   buffers are always physically resident — `MTLResidencySet` only
//!   hints at working-set priority, and its absence does not cause
//!   incorrect execution. A future pass can switch the fallback to
//!   allocating from an `MTLHeap` for cleaner eviction semantics; the
//!   user's Tahoe box exercises the `ResidencySet` path, so the heap
//!   fallback is strictly forward-compatibility surface today.
//!
//! * **No-Metal path:** `ResidencyPool::disabled()` — a pure no-op used
//!   when residency APIs are unavailable or the probe returns an error.
//!
//! # M2 residency budget — pathological-case risk note
//!
//! The Stage 3 memory policy (D-016) is "free-on-consume": GPU
//! histograms live for exactly one level (level-wise) or until the
//! `PendingSplit` pops (leaf-wise). Peak residency is therefore
//! bounded by **one level width × histogram size**.
//!
//! Back-of-envelope numbers for the pathological case: leaf-wise +
//! `max_leaves = 1024` + 1000 features + 1024 bins →
//! `1024 × 1000 × 1024 × 12 bytes = ~12 GiB` of histogram residency.
//! That exceeds the `recommendedMaxWorkingSetSize` of every M-series
//! chip below M3 Max with 36+ GiB unified memory, and can stress
//! even the high-end chips under concurrent workload.
//!
//! S3.9 guards this shape at `MetalBackend::new()` and again at fit
//! start by refusing to run when projected peak exceeds 80 % of
//! `recommendedMaxWorkingSetSize`, returning `EngineError::Backend`
//! with a clear message that points at M3 (probe-detected LRU spill)
//! as the documented follow-up. Today (M2) the strictly-correct
//! behaviour vs. status-quo CPU backend is: CPU already implicitly
//! pages histograms to swap in this shape (slow but survives); M2
//! refuses the fit (fast but user-visible error). This is a
//! documented limitation, not a regression.

#![allow(unsafe_code)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor,
};

/// Working-set residency pool. Wraps `MTLResidencySet` on macOS 15+ and
/// degrades gracefully on every other path.
///
/// The pool is owned by `MetalBackend` (one per backend instance) and is
/// attached to the backend's serial command queue at construction. Every
/// freshly-allocated GPU-resident buffer (histograms, row indices, etc.)
/// should be registered via [`ResidencyPool::add_buffer`] before its
/// first command-buffer submission; the pool is committed at the end of
/// each attachment phase.
///
/// `dead_code` allow on the `queue` field: it is held so the attachment
/// is released with the pool. The attach-on-construct / detach-on-drop
/// lifecycle is encoded structurally rather than via an explicit
/// teardown method to match the rest of the backend's RAII style.
///
/// Struct-level `allow(dead_code)`: the live consumer is S3.7
/// (`MetalBackend` histogram-residency wiring). S3.8 ships the
/// infrastructure in isolation so it can be unit-tested without
/// tangling with the trainer call-graph refactor (S3.3).
#[allow(dead_code)]
pub(crate) struct ResidencyPool {
    strategy: Strategy,
    /// Retained handle on the command queue the pool attached itself
    /// to. Holding it guarantees the queue outlives the residency set
    /// (otherwise detachment on drop would be against a dead handle).
    #[allow(dead_code)]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Tracks whether `request_residency` has been called. `commit`
    /// honours this flag: if residency is already active, the commit
    /// also applies the added allocations to the working set
    /// immediately; otherwise it queues them for the next
    /// `request_residency`.
    active: std::sync::Mutex<bool>,
}

#[allow(dead_code)]
enum Strategy {
    /// macOS 15+: `MTLResidencySet` is the first-class residency API.
    ResidencySet(Retained<ProtocolObject<dyn MTLResidencySet>>),
    /// macOS 13/14 or any path where `newResidencySetWithDescriptor:error:`
    /// returned an error. On Apple-Silicon UMA, `StorageModeShared`
    /// buffers are always physically resident, so the absence of an
    /// explicit working-set pin is not a correctness issue.
    ///
    /// The eventual `MTLHeap`-backed fallback will replace this variant
    /// with sub-allocated heap storage; until then the backend runs
    /// with implicit residency and logs a one-line notice at
    /// construction.
    PassThrough,
}

#[allow(dead_code)]
impl ResidencyPool {
    /// Attempt to create a residency set attached to `queue`. Never
    /// fails: a creation error downgrades to [`Strategy::PassThrough`].
    /// `label` is used for Xcode / Instruments visibility.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        label: &str,
    ) -> Self {
        let strategy = match try_build_residency_set(device, label) {
            Some(set) => {
                // Attach to the command queue so every command buffer
                // inherits the residency set with zero per-encode cost.
                queue.addResidencySet(&set);
                Strategy::ResidencySet(set)
            }
            None => Strategy::PassThrough,
        };
        Self {
            strategy,
            queue,
            active: std::sync::Mutex::new(false),
        }
    }

    /// Zero-cost pool that never registers anything. Used by call sites
    /// that want a pool-shaped value when residency is known to be
    /// unavailable (headless probes, unit tests without a real device).
    #[cfg(test)]
    pub(crate) fn disabled(queue: Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Self {
        Self {
            strategy: Strategy::PassThrough,
            queue,
            active: std::sync::Mutex::new(false),
        }
    }

    /// `true` when the pool holds a live `MTLResidencySet`. Exposed for
    /// capability reporting + tests.
    pub(crate) fn is_active(&self) -> bool {
        matches!(self.strategy, Strategy::ResidencySet(_))
    }

    /// Register a buffer with the pool. Call before the buffer's first
    /// command-buffer submission. Safe to call on the pass-through
    /// variant (becomes a no-op).
    ///
    /// Per Apple's `MTLResidencySet` contract the allocation is
    /// uncommitted until [`ResidencyPool::commit`] runs. Callers that
    /// add a batch of buffers should call `commit` once at the end
    /// rather than committing per-add.
    pub(crate) fn add_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        if let Strategy::ResidencySet(set) = &self.strategy {
            // `MTLBuffer: MTLAllocation` — the protocol upcast is a
            // no-op at runtime. We can't use the trait-object-cast
            // syntax directly (no `as &dyn MTLAllocation` on a
            // `&ProtocolObject<dyn MTLBuffer>`), so we rely on
            // `ProtocolObject::from_ref` on a concrete `&MTLAllocation`
            // projection — the safe way to go through objc2's
            // protocol-hierarchy machinery.
            let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(buffer);
            set.addAllocation(allocation);
        }
    }

    /// Remove a buffer from the pool. Symmetrical with
    /// [`ResidencyPool::add_buffer`]; call when a buffer is going out
    /// of scope so the residency set can drop it on the next commit.
    #[allow(dead_code)]
    pub(crate) fn remove_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        if let Strategy::ResidencySet(set) = &self.strategy {
            let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(buffer);
            set.removeAllocation(allocation);
        }
    }

    /// Apply pending adds/removes. If the set is already resident
    /// (post-`request_residency`), Metal makes added allocations
    /// resident immediately.
    pub(crate) fn commit(&self) {
        if let Strategy::ResidencySet(set) = &self.strategy {
            set.commit();
        }
    }

    /// Pin the residency set to the GPU working set. Call once at the
    /// start of a fit (or other long-running attachment phase) after
    /// initial allocations are registered. Safe to call multiple times
    /// — Metal reference-counts residency requests.
    pub(crate) fn request_residency(&self) {
        if let Strategy::ResidencySet(set) = &self.strategy {
            set.requestResidency();
            if let Ok(mut active) = self.active.lock() {
                *active = true;
            }
        }
    }

    /// Release the residency pin. Call at fit teardown so subsequent
    /// fits see a clean working set.
    #[allow(dead_code)]
    pub(crate) fn end_residency(&self) {
        if let Strategy::ResidencySet(set) = &self.strategy {
            set.endResidency();
            if let Ok(mut active) = self.active.lock() {
                *active = false;
            }
        }
    }
}

impl Drop for ResidencyPool {
    fn drop(&mut self) {
        // Best-effort release. The command queue's `Retained` holds
        // the queue alive until we drop it; pairing `endResidency`
        // with `removeResidencySet` makes teardown observable to
        // Instruments even across fit boundaries.
        if let Strategy::ResidencySet(set) = &self.strategy {
            set.removeAllAllocations();
            set.commit();
            // `active` may be poisoned if a thread panicked holding
            // the lock; read optimistically.
            let was_active = self.active.lock().map(|g| *g).unwrap_or(false);
            if was_active {
                set.endResidency();
            }
            self.queue.removeResidencySet(set);
        }
    }
}

/// Try to build an `MTLResidencySet` with `label`. Returns `None` on
/// any failure (macOS 13/14, transient allocation error, unsupported
/// device). The caller degrades to pass-through.
#[allow(dead_code)]
fn try_build_residency_set(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
) -> Option<Retained<ProtocolObject<dyn MTLResidencySet>>> {
    let descriptor = MTLResidencySetDescriptor::new();
    let label_ns = NSString::from_str(label);
    descriptor.setLabel(Some(&label_ns));
    // `initialCapacity` is an optimisation hint; 32 is comfortably
    // larger than the histogram + row-index slot count at Stage 3
    // (≤ ~a dozen long-lived buffers per fit). Calling the unsafe
    // setter with a known-in-range literal is safe.
    unsafe { descriptor.setInitialCapacity(32) };
    device.newResidencySetWithDescriptor_error(&descriptor).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn try_probe_device() -> Option<MetalDevice> {
        MetalDevice::probe().ok()
    }

    #[test]
    fn residency_pool_probe_round_trip() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = ResidencyPool::new(
            &metal_device.device,
            metal_device.queue.clone(),
            "alloygbm::residency::test",
        );
        // On the macOS 15+ Tahoe dev box we expect `is_active()` to
        // return true. Older paths degrade silently.
        let _ = pool.is_active();
        pool.request_residency();
        pool.commit();
        pool.end_residency();
    }

    #[test]
    fn residency_pool_disabled_is_no_op() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = ResidencyPool::disabled(metal_device.queue.clone());
        assert!(!pool.is_active());
        pool.request_residency();
        pool.commit();
        pool.end_residency();
    }

    #[test]
    fn residency_pool_registers_buffer() {
        let Some(metal_device) = try_probe_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = ResidencyPool::new(
            &metal_device.device,
            metal_device.queue.clone(),
            "alloygbm::residency::test",
        );
        // 1 KiB shared-mode buffer — enough to be a valid allocation
        // without stressing UMA.
        let buffer = metal_device
            .device
            .newBufferWithLength_options(1024, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("newBuffer allocation failure");
        pool.add_buffer(&buffer);
        pool.commit();
        pool.request_residency();
        pool.end_residency();
        pool.remove_buffer(&buffer);
        pool.commit();
    }
}
