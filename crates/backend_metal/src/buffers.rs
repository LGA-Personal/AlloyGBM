//! Persistent Metal buffer pool for the histogram dispatch path.
//!
//! `dispatch_histograms` runs ~63 times per tree at depth 6 and the
//! engine calls it with the **same** `BinnedMatrix` every time within
//! a fit. Allocating a fresh `MTLBuffer` and copying the column-major
//! binned matrix on every call can cost tens of GiB of needless
//! memcpy per fit at realistic scales. `BufferCache` holds persistent
//! slots for the buffers that a single dispatch needs so that a warm
//! second-call path only re-encodes the pipeline — no reallocation
//! and, for the binned matrix, no copy at all.
//!
//! Cache keys are `(ptr_as_usize, len_bytes)`. That's safe only when
//! the underlying memory is immutable for the lifetime of the cached
//! buffer:
//!
//! * **Binned matrix** — immutable for the whole fit (built once in
//!   the core crate, read-only thereafter). Cached across calls.
//! * **Gradients** — overwritten in place once per boosting round by
//!   `objective.compute_gradients_into`. Same pointer + length can
//!   legitimately hold new content, so we only reuse the *buffer
//!   allocation* and rewrite its contents every call. No key check.
//! * **Row indices** — same as gradients: reuse the allocation, copy
//!   fresh bytes every call.
//!
//! The two reusable slots grow monotonically — if the next call asks
//! for more bytes than the slot holds, the slot is replaced; smaller
//! requests reuse the existing allocation.

#![allow(unsafe_code)]

use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use alloygbm_engine::{EngineError, EngineResult};

/// Cache key for the binned-matrix buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BinnedKey {
    ptr: usize,
    len_bytes: usize,
    is_wide: bool,
}

struct CachedBinned {
    key: BinnedKey,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Reusable buffer slot — held across calls and resized on demand.
struct ReusableSlot {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity_bytes: usize,
}

pub(crate) struct BufferCache {
    binned: Mutex<Option<CachedBinned>>,
    gradients: Mutex<Option<ReusableSlot>>,
    row_indices: Mutex<Option<ReusableSlot>>,
}

// SAFETY: all mutable state is behind Mutex. The Retained<ProtocolObject>
// Metal handles are documented thread-safe.
unsafe impl Send for BufferCache {}
// SAFETY: see Send impl.
unsafe impl Sync for BufferCache {}

impl BufferCache {
    pub(crate) fn new() -> Self {
        Self {
            binned: Mutex::new(None),
            gradients: Mutex::new(None),
            row_indices: Mutex::new(None),
        }
    }

    /// Return a buffer holding the same bytes as `data`. If the last
    /// call passed the same `(ptr, len, is_wide)` key, the cached
    /// buffer is cloned (zero-copy `Retained` clone); otherwise a
    /// fresh shared buffer is allocated and copied into.
    ///
    /// Caller must guarantee the memory at `data` is immutable for
    /// the lifetime of the returned buffer (i.e. between this call
    /// and `dispatch_histograms` returning). The engine upholds this
    /// for the binned matrix because it is constructed once per fit
    /// and never mutated thereafter.
    #[allow(unsafe_code)]
    pub(crate) fn get_or_upload_binned<T: Copy>(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
        is_wide: bool,
    ) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let len_bytes = std::mem::size_of_val(data);
        let ptr = data.as_ptr() as usize;
        let key = BinnedKey {
            ptr,
            len_bytes,
            is_wide,
        };

        let mut guard = self.binned.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("binned-matrix cache poisoned: {e}"))
        })?;
        if let Some(cached) = guard.as_ref()
            && cached.key == key
        {
            return Ok(cached.buffer.clone());
        }

        let buffer = allocate_and_copy(device, data)?;
        *guard = Some(CachedBinned {
            key,
            buffer: buffer.clone(),
        });
        Ok(buffer)
    }

    /// Ensure the gradients slot has capacity for `data.len()` bytes,
    /// then copy `data` into the slot and return a handle to it. The
    /// slot is replaced (not resized in place) if capacity is
    /// insufficient.
    #[allow(unsafe_code)]
    pub(crate) fn write_gradients<T: Copy>(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let mut guard = self.gradients.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("gradients slot cache poisoned: {e}"))
        })?;
        write_into_slot(device, &mut guard, data)
    }

    /// As `write_gradients` but for the row-indices buffer.
    #[allow(unsafe_code)]
    pub(crate) fn write_row_indices<T: Copy>(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let mut guard = self.row_indices.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("row-indices slot cache poisoned: {e}"))
        })?;
        write_into_slot(device, &mut guard, data)
    }
}

#[allow(unsafe_code)]
fn write_into_slot<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    slot: &mut Option<ReusableSlot>,
    data: &[T],
) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let needed_bytes = std::mem::size_of_val(data).max(1);

    let buffer_ref: &Retained<ProtocolObject<dyn MTLBuffer>> = match slot.as_mut() {
        Some(current) if current.capacity_bytes >= needed_bytes => &current.buffer,
        _ => {
            let buffer = device
                .newBufferWithLength_options(needed_bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    EngineError::BackendUnavailable(
                        "could not allocate reusable shared buffer".to_string(),
                    )
                })?;
            *slot = Some(ReusableSlot {
                buffer,
                capacity_bytes: needed_bytes,
            });
            &slot.as_ref().expect("just set").buffer
        }
    };

    if std::mem::size_of_val(data) > 0 {
        // SAFETY: `buffer_ref` is StorageModeShared so `.contents()`
        // returns a valid CPU-visible pointer for `capacity_bytes`.
        // `data` is a valid Rust slice; we memcpy its bytes into the
        // buffer. No one else is reading the buffer at this point —
        // the previous dispatch's `waitUntilCompleted` guarantees the
        // GPU is done with it.
        unsafe {
            let dst = buffer_ref.contents().as_ptr() as *mut u8;
            let src = data.as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src, dst, std::mem::size_of_val(data));
        }
    }

    Ok(buffer_ref.clone())
}

#[allow(unsafe_code)]
fn allocate_and_copy<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let len_bytes = std::mem::size_of_val(data);
    if len_bytes == 0 {
        return device
            .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "could not allocate empty placeholder buffer".to_string(),
                )
            });
    }
    let ptr = data.as_ptr() as *mut std::ffi::c_void;
    // SAFETY: `ptr` is non-null (len_bytes > 0) and valid for `len_bytes`
    // bytes. Metal copies the bytes into a fresh `StorageModeShared`
    // buffer synchronously; the original slice is not aliased past this
    // call.
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new_unchecked(ptr),
            len_bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| {
        EngineError::BackendUnavailable("could not allocate Metal buffer".to_string())
    })?;
    Ok(buffer)
}
