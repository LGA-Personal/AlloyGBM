//! Compile the histogram MSL kernel and build the pair of specialized
//! `MTLComputePipelineState`s that `dispatch_histograms` executes.
//!
//! # S1.5 — Pipeline caching
//!
//! The pre-S1.5 code compiled the MSL library and built both pipeline
//! states fresh on every node-level dispatch — dozens of milliseconds
//! per node. S1.5 introduces [`HistogramPipelineCache`] which owns:
//!
//! 1. A **compiled [`MTLLibrary`] held for the lifetime of the backend**.
//!    The MSL source is compiled exactly once per process.
//! 2. An **in-process `Mutex<HashMap<(bin_count, use_u16_bins),
//!    Arc<HistogramPipelines>>>`** keyed on the two function-constant
//!    dimensions. In typical training runs there is a single key, so
//!    every dispatch after the first is an `Arc::clone`.
//! 3. A best-effort **[`MTLBinaryArchive`] persisted at
//!    `~/Library/Caches/com.alloygbm/pipelines-<family>-<device>.metalarchive`**.
//!    The archive is handed to Metal via
//!    `MTLComputePipelineDescriptor::setBinaryArchives` so that pipeline
//!    build can source already-compiled code from disk on second and
//!    subsequent runs. When a fresh build happens the compiled pipeline
//!    functions are added back to the archive with
//!    `addComputePipelineFunctionsWithDescriptor:error:`, and the
//!    archive is serialized to a temp path + `rename`d into place when
//!    the cache drops. Archive failures are non-fatal — Metal falls
//!    back to source compilation on every step.
//!
//! Correctness invariants:
//!
//! * Every `Arc<HistogramPipelines>` returned from `get_or_build` has
//!   been successfully built on the current device with the matching
//!   function-constant values; a binary-archive hit or source compile
//!   is indistinguishable from the caller's perspective.
//! * Failure to create or load the on-disk archive never prevents
//!   pipeline construction — we build without archive assistance and
//!   log at `eprintln!` when best-effort paths fail.
//!
//! The cache is thread-safe through a plain `Mutex`; concurrent
//! callers with the same key contend once on the slow path and then
//! hit the cached `Arc` forever after.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLPipelineOption,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::device::MetalCapabilities;
use crate::kernels::histogram::{
    HISTOGRAM_SHADER_SOURCE, KERNEL_NAME_REDUCE, KERNEL_NAME_SCATTER, KERNEL_NAME_SCATTER_WIDE,
    MAX_BIN_COUNT_WIDE,
};
use crate::kernels::partition::{
    KERNEL_NAME_PARTITION_FLAG_AND_COUNT, KERNEL_NAME_PARTITION_SCAN_BLOCKS,
    KERNEL_NAME_PARTITION_SCATTER, PARTITION_SHADER_SOURCE, PartitionSpecKey,
};
use crate::kernels::split::{KERNEL_NAME_BEST_SPLIT_PER_FEATURE, SPLIT_SHADER_SOURCE};
use crate::kernels::subtract::{KERNEL_NAME_SUBTRACT, SUBTRACT_SHADER_SOURCE, SubtractSpecKey};

/// A pair of compute pipelines specialized for a given `(bin_count,
/// use_u16_bins)` pair. Both pipelines share the same underlying
/// `MTLLibrary`; the function constants baked at pipeline-create time
/// drive the kernel's internal branches and buffer layout.
pub struct HistogramPipelines {
    pub scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Wide-path scatter (4 simdgroups / 128 threads, per-simdgroup
    /// private histograms; D-021). `None` when `bin_count >
    /// MAX_BIN_COUNT_WIDE` — the wide kernel's threadgroup memory
    /// array is sized for the wide cap, so it can't be instantiated
    /// for wider bin counts.
    pub scatter_wide: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub reduce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub bin_count: u32,
    pub use_u16_bins: bool,
}

/// Process-lifetime cache of histogram pipelines, with optional
/// on-disk persistence via `MTLBinaryArchive`.
///
/// Hold one instance per `MetalBackend`.
///
/// # Thread-safety
///
/// Apple documents `MTLDevice`, `MTLLibrary`, and `MTLComputePipelineState`
/// as thread-safe for concurrent use — multiple threads may submit
/// commands or query metadata against a shared device or library
/// without external synchronization. `MTLBinaryArchive` mutation
/// methods (`addComputePipelineFunctions...`, `serializeToURL:`) are
/// additionally guarded by the cache's own `dirty` mutex, so
/// concurrent `get_or_build` callers cannot race on the archive.
/// These `unsafe impl`s assert thread-safety for the overall struct
/// given those invariants.
// SAFETY: Metal protocol objects held inside this struct are
// thread-safe per Apple's documentation, and all mutable state
// (`entries`, `dirty`) is guarded by `Mutex`. See doc comment above.
unsafe impl Send for HistogramPipelineCache {}
// SAFETY: See `Send` impl.
unsafe impl Sync for HistogramPipelineCache {}

pub struct HistogramPipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Best-effort on-disk archive. `None` when the OS returned an
    /// error creating it (permissions, read-only volume, etc.) — the
    /// cache still functions, we just never persist across runs.
    archive: Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    /// Location we would serialize `archive` to at drop time.
    archive_path: Option<PathBuf>,
    /// Set whenever a fresh pipeline build successfully adds into
    /// `archive`. If never set by drop time, serialization is skipped
    /// (the on-disk file already reflects the current content).
    dirty: Mutex<bool>,
    entries: Mutex<HashMap<(u32, bool), Arc<HistogramPipelines>>>,
}

impl HistogramPipelineCache {
    /// Compile the MSL library once, attempt to open/create the on-disk
    /// binary archive, and return a ready cache. Failure to set up the
    /// archive is logged and swallowed — pipeline builds still work.
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        capabilities: &MetalCapabilities,
    ) -> Result<Self, String> {
        let source = NSString::from_str(HISTOGRAM_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|err| {
                format!(
                    "histogram.metal library compile failed: {}",
                    err.localizedDescription()
                )
            })?;

        let (archive, archive_path) = open_or_create_archive(&device, capabilities);

        Ok(Self {
            device,
            library,
            archive,
            archive_path,
            dirty: Mutex::new(false),
            entries: Mutex::new(HashMap::new()),
        })
    }

    /// Return an `Arc` to the pipelines specialized for the given
    /// `(bin_count, use_u16)` pair, building and caching on first use.
    pub fn get_or_build(
        &self,
        bin_count: u32,
        use_u16: bool,
    ) -> Result<Arc<HistogramPipelines>, String> {
        // Cheap fast path — read lock.
        {
            let entries = self
                .entries
                .lock()
                .map_err(|_| "histogram pipeline cache mutex poisoned".to_string())?;
            if let Some(cached) = entries.get(&(bin_count, use_u16)) {
                return Ok(Arc::clone(cached));
            }
        }

        // Slow path — compile + insert. We release the entries lock
        // above so that an unrelated key's build in another thread
        // is not blocked on ours, then re-lock to insert. A
        // double-build under contention is still cheaper than holding
        // the lock across the GPU driver call.
        let pipelines = self.build(bin_count, use_u16)?;
        let arc = Arc::new(pipelines);

        let mut entries = self
            .entries
            .lock()
            .map_err(|_| "histogram pipeline cache mutex poisoned".to_string())?;
        let stored = entries
            .entry((bin_count, use_u16))
            .or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }

    fn build(&self, bin_count: u32, use_u16_bins: bool) -> Result<HistogramPipelines, String> {
        // Specialize both MSL entry points via function constants.
        let constants = MTLFunctionConstantValues::new();
        let bin_count_cell: u32 = bin_count;
        let use_u16_cell: u8 = u8::from(use_u16_bins);

        // SAFETY: both pointers are to live stack slots with the
        // matching `MTLDataType` size; indices 0 and 1 match the
        // `[[function_constant(N)]]` declarations in
        // `shaders/histogram.metal`.
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const bin_count_cell) as *mut c_void),
                MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const use_u16_cell) as *mut c_void),
                MTLDataType::Bool,
                1,
            );
        }

        let scatter_name = NSString::from_str(KERNEL_NAME_SCATTER);
        let scatter_wide_name = NSString::from_str(KERNEL_NAME_SCATTER_WIDE);
        let reduce_name = NSString::from_str(KERNEL_NAME_REDUCE);

        let scatter_fn = self
            .library
            .newFunctionWithName_constantValues_error(&scatter_name, &constants)
            .map_err(|err| {
                format!(
                    "could not specialize `{KERNEL_NAME_SCATTER}`: {}",
                    err.localizedDescription()
                )
            })?;
        let reduce_fn = self
            .library
            .newFunctionWithName_constantValues_error(&reduce_name, &constants)
            .map_err(|err| {
                format!(
                    "could not specialize `{KERNEL_NAME_REDUCE}`: {}",
                    err.localizedDescription()
                )
            })?;

        // Build via descriptors so we can pass our on-disk archive to
        // the driver. If the archive contains a hit the driver will
        // skip source compilation; otherwise it compiles as before.
        let scatter_desc = MTLComputePipelineDescriptor::new();
        scatter_desc.setComputeFunction(Some(&scatter_fn));
        let reduce_desc = MTLComputePipelineDescriptor::new();
        reduce_desc.setComputeFunction(Some(&reduce_fn));

        if let Some(archive) = &self.archive {
            let archive_obj: &ProtocolObject<dyn MTLBinaryArchive> = archive;
            let archive_array = NSArray::from_slice(&[archive_obj]);
            scatter_desc.setBinaryArchives(Some(&archive_array));
            reduce_desc.setBinaryArchives(Some(&archive_array));
        }

        let scatter = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &scatter_desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|err| {
                format!(
                    "scatter pipeline creation failed: {}",
                    err.localizedDescription()
                )
            })?;
        let reduce = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &reduce_desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|err| {
                format!(
                    "reduce pipeline creation failed: {}",
                    err.localizedDescription()
                )
            })?;

        // Wide scatter is only valid when the per-simdgroup private
        // histograms fit in threadgroup memory (D-021). For wider bin
        // counts we dispatch the narrow path.
        let (scatter_wide, scatter_wide_desc) = if bin_count <= MAX_BIN_COUNT_WIDE {
            let scatter_wide_fn = self
                .library
                .newFunctionWithName_constantValues_error(&scatter_wide_name, &constants)
                .map_err(|err| {
                    format!(
                        "could not specialize `{KERNEL_NAME_SCATTER_WIDE}`: {}",
                        err.localizedDescription()
                    )
                })?;
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(Some(&scatter_wide_fn));
            if let Some(archive) = &self.archive {
                let archive_obj: &ProtocolObject<dyn MTLBinaryArchive> = archive;
                let archive_array = NSArray::from_slice(&[archive_obj]);
                desc.setBinaryArchives(Some(&archive_array));
            }
            let pipeline = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|err| {
                    format!(
                        "scatter_wide pipeline creation failed: {}",
                        err.localizedDescription()
                    )
                })?;
            (Some(pipeline), Some(desc))
        } else {
            (None, None)
        };

        // If we have an archive, opportunistically persist the
        // freshly-compiled pipeline functions so the next run can
        // skip compilation. This is strictly best-effort: a failure
        // here only means we'll compile again next run.
        if let Some(archive) = &self.archive {
            let mut added_any = false;
            if let Err(err) = archive.addComputePipelineFunctionsWithDescriptor_error(&scatter_desc)
            {
                eprintln!(
                    "[alloygbm metal] warning: could not add scatter pipeline to archive: {}",
                    err.localizedDescription()
                );
            } else {
                added_any = true;
            }
            if let Some(desc) = &scatter_wide_desc {
                if let Err(err) = archive.addComputePipelineFunctionsWithDescriptor_error(desc) {
                    eprintln!(
                        "[alloygbm metal] warning: could not add scatter_wide pipeline to archive: {}",
                        err.localizedDescription()
                    );
                } else {
                    added_any = true;
                }
            }
            if let Err(err) = archive.addComputePipelineFunctionsWithDescriptor_error(&reduce_desc)
            {
                eprintln!(
                    "[alloygbm metal] warning: could not add reduce pipeline to archive: {}",
                    err.localizedDescription()
                );
            } else {
                added_any = true;
            }
            if added_any && let Ok(mut dirty) = self.dirty.lock() {
                *dirty = true;
            }
        }

        Ok(HistogramPipelines {
            scatter,
            scatter_wide,
            reduce,
            bin_count,
            use_u16_bins,
        })
    }
}

impl Drop for HistogramPipelineCache {
    /// Flush the archive to disk when the backend shuts down. We
    /// write to `<path>.tmp` first and `rename` into place so a crash
    /// mid-write leaves the previous archive intact — per Apple's
    /// guidance that runtime archive updates require corruption
    /// resiliency.
    fn drop(&mut self) {
        let (archive, path) = match (&self.archive, &self.archive_path) {
            (Some(a), Some(p)) => (a, p),
            _ => return,
        };

        let dirty = self.dirty.lock().map(|g| *g).unwrap_or(false);
        if !dirty {
            return;
        }

        let tmp_path = path.with_extension("metalarchive.tmp");
        let tmp_nsstring = NSString::from_str(&tmp_path.to_string_lossy());
        let tmp_url = NSURL::fileURLWithPath(&tmp_nsstring);

        if let Err(err) = archive.serializeToURL_error(&tmp_url) {
            eprintln!(
                "[alloygbm metal] warning: could not serialize binary archive to {}: {}",
                tmp_path.display(),
                err.localizedDescription()
            );
            return;
        }

        if let Err(err) = std::fs::rename(&tmp_path, path) {
            eprintln!(
                "[alloygbm metal] warning: could not rename archive {} -> {}: {err}",
                tmp_path.display(),
                path.display()
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

/// Attempt to locate (or create) the on-disk pipeline archive.
/// Returns `(None, None)` on any failure so the caller can proceed
/// without persistence.
fn open_or_create_archive(
    device: &ProtocolObject<dyn MTLDevice>,
    capabilities: &MetalCapabilities,
) -> (
    Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    Option<PathBuf>,
) {
    let Some(cache_dir) = user_cache_dir() else {
        return (None, None);
    };
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        eprintln!(
            "[alloygbm metal] warning: could not create cache dir {}: {err}",
            cache_dir.display()
        );
        return (None, None);
    }

    let filename = archive_filename(capabilities);
    let archive_path = cache_dir.join(filename);

    let descriptor = MTLBinaryArchiveDescriptor::new();
    // If the file exists, point the descriptor at it so the driver
    // loads existing contents. If it doesn't, leave `url` nil for a
    // fresh empty archive.
    if archive_path.exists() {
        let ns = NSString::from_str(&archive_path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&ns);
        descriptor.setUrl(Some(&url));
    }

    match device.newBinaryArchiveWithDescriptor_error(&descriptor) {
        Ok(archive) => (Some(archive), Some(archive_path)),
        Err(err) => {
            eprintln!(
                "[alloygbm metal] warning: could not open binary archive at {}: {}. \
                 Continuing without pipeline persistence.",
                archive_path.display(),
                err.localizedDescription()
            );
            // Delete a corrupt file so next run has a clean slate.
            if archive_path.exists() {
                let _ = std::fs::remove_file(&archive_path);
            }
            // Try one more time with an empty descriptor; if that
            // also fails, give up.
            let fresh_descriptor = MTLBinaryArchiveDescriptor::new();
            match device.newBinaryArchiveWithDescriptor_error(&fresh_descriptor) {
                Ok(archive) => (Some(archive), Some(archive_path)),
                Err(err2) => {
                    eprintln!(
                        "[alloygbm metal] warning: could not create empty binary archive: {}",
                        err2.localizedDescription()
                    );
                    (None, None)
                }
            }
        }
    }
}

fn user_cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = Path::new(&home).join("Library/Caches/com.alloygbm");
    Some(path)
}

fn archive_filename(capabilities: &MetalCapabilities) -> String {
    let family = if capabilities.metal4 {
        "metal4"
    } else if capabilities.apple7 {
        "apple7"
    } else {
        "generic"
    };
    let device_slug = slugify_device_name(&capabilities.device_name);
    format!("pipelines-{family}-{device_slug}.metalarchive")
}

fn split_archive_filename(capabilities: &MetalCapabilities) -> String {
    let family = if capabilities.metal4 {
        "metal4"
    } else if capabilities.apple7 {
        "apple7"
    } else {
        "generic"
    };
    let device_slug = slugify_device_name(&capabilities.device_name);
    format!("split-pipelines-{family}-{device_slug}.metalarchive")
}

// ---------------------------------------------------------------------
// S2.3 — SplitPipelineCache
// ---------------------------------------------------------------------
//
// Mirrors `HistogramPipelineCache`'s structure but compiles
// `shaders/split.metal` and specializes on `(bin_count, l1_enabled)`.
// The on-disk archive lives in a distinct file so the two caches never
// contend on the same `MTLBinaryArchive` object; both are best-effort
// persistence and independent.

pub struct SplitPipelines {
    pub per_feature: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub bin_count: u32,
    pub l1_enabled: bool,
}

// SAFETY: Metal protocol objects held inside this struct are
// thread-safe per Apple's documentation, and all mutable state
// (`entries`, `dirty`) is guarded by `Mutex`.
unsafe impl Send for SplitPipelineCache {}
// SAFETY: See `Send` impl.
unsafe impl Sync for SplitPipelineCache {}

pub struct SplitPipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    archive: Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    archive_path: Option<PathBuf>,
    dirty: Mutex<bool>,
    entries: Mutex<HashMap<(u32, bool), Arc<SplitPipelines>>>,
}

impl SplitPipelineCache {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        capabilities: &MetalCapabilities,
    ) -> Result<Self, String> {
        let source = NSString::from_str(SPLIT_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|err| {
                format!(
                    "split.metal library compile failed: {}",
                    err.localizedDescription()
                )
            })?;

        let (archive, archive_path) = open_or_create_split_archive(&device, capabilities);

        Ok(Self {
            device,
            library,
            archive,
            archive_path,
            dirty: Mutex::new(false),
            entries: Mutex::new(HashMap::new()),
        })
    }

    /// Return an `Arc` to the pipeline specialized for the given
    /// `(bin_count, l1_enabled)` pair, building and caching on first use.
    pub fn get_or_build(
        &self,
        bin_count: u32,
        l1_enabled: bool,
    ) -> Result<Arc<SplitPipelines>, String> {
        {
            let entries = self
                .entries
                .lock()
                .map_err(|_| "split pipeline cache mutex poisoned".to_string())?;
            if let Some(cached) = entries.get(&(bin_count, l1_enabled)) {
                return Ok(Arc::clone(cached));
            }
        }

        let pipeline = self.build(bin_count, l1_enabled)?;
        let arc = Arc::new(pipeline);

        let mut entries = self
            .entries
            .lock()
            .map_err(|_| "split pipeline cache mutex poisoned".to_string())?;
        let stored = entries
            .entry((bin_count, l1_enabled))
            .or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }

    fn build(&self, bin_count: u32, l1_enabled: bool) -> Result<SplitPipelines, String> {
        let constants = MTLFunctionConstantValues::new();
        let bin_count_cell: u32 = bin_count;
        let l1_cell: u8 = u8::from(l1_enabled);

        // SAFETY: both pointers are to live stack slots with the
        // matching `MTLDataType` size; indices 0 and 1 match the
        // `[[function_constant(N)]]` declarations in
        // `shaders/split.metal`.
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const bin_count_cell) as *mut c_void),
                MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const l1_cell) as *mut c_void),
                MTLDataType::Bool,
                1,
            );
        }

        let fn_name = NSString::from_str(KERNEL_NAME_BEST_SPLIT_PER_FEATURE);
        let per_feature_fn = self
            .library
            .newFunctionWithName_constantValues_error(&fn_name, &constants)
            .map_err(|err| {
                format!(
                    "could not specialize `{KERNEL_NAME_BEST_SPLIT_PER_FEATURE}`: {}",
                    err.localizedDescription()
                )
            })?;

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&per_feature_fn));

        if let Some(archive) = &self.archive {
            let archive_obj: &ProtocolObject<dyn MTLBinaryArchive> = archive;
            let archive_array = NSArray::from_slice(&[archive_obj]);
            desc.setBinaryArchives(Some(&archive_array));
        }

        let per_feature = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|err| {
                format!(
                    "split per-feature pipeline creation failed: {}",
                    err.localizedDescription()
                )
            })?;

        if let Some(archive) = &self.archive {
            if let Err(err) = archive.addComputePipelineFunctionsWithDescriptor_error(&desc) {
                eprintln!(
                    "[alloygbm metal] warning: could not add split pipeline to archive: {}",
                    err.localizedDescription()
                );
            } else if let Ok(mut dirty) = self.dirty.lock() {
                *dirty = true;
            }
        }

        Ok(SplitPipelines {
            per_feature,
            bin_count,
            l1_enabled,
        })
    }
}

impl Drop for SplitPipelineCache {
    fn drop(&mut self) {
        let (archive, path) = match (&self.archive, &self.archive_path) {
            (Some(a), Some(p)) => (a, p),
            _ => return,
        };

        let dirty = self.dirty.lock().map(|g| *g).unwrap_or(false);
        if !dirty {
            return;
        }

        let tmp_path = path.with_extension("metalarchive.tmp");
        let tmp_nsstring = NSString::from_str(&tmp_path.to_string_lossy());
        let tmp_url = NSURL::fileURLWithPath(&tmp_nsstring);

        if let Err(err) = archive.serializeToURL_error(&tmp_url) {
            eprintln!(
                "[alloygbm metal] warning: could not serialize split binary archive to {}: {}",
                tmp_path.display(),
                err.localizedDescription()
            );
            return;
        }

        if let Err(err) = std::fs::rename(&tmp_path, path) {
            eprintln!(
                "[alloygbm metal] warning: could not rename split archive {} -> {}: {err}",
                tmp_path.display(),
                path.display()
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

fn open_or_create_split_archive(
    device: &ProtocolObject<dyn MTLDevice>,
    capabilities: &MetalCapabilities,
) -> (
    Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    Option<PathBuf>,
) {
    let Some(cache_dir) = user_cache_dir() else {
        return (None, None);
    };
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        eprintln!(
            "[alloygbm metal] warning: could not create cache dir {}: {err}",
            cache_dir.display()
        );
        return (None, None);
    }

    let filename = split_archive_filename(capabilities);
    let archive_path = cache_dir.join(filename);

    let descriptor = MTLBinaryArchiveDescriptor::new();
    if archive_path.exists() {
        let ns = NSString::from_str(&archive_path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&ns);
        descriptor.setUrl(Some(&url));
    }

    match device.newBinaryArchiveWithDescriptor_error(&descriptor) {
        Ok(archive) => (Some(archive), Some(archive_path)),
        Err(err) => {
            eprintln!(
                "[alloygbm metal] warning: could not open split binary archive at {}: {}. \
                 Continuing without pipeline persistence.",
                archive_path.display(),
                err.localizedDescription()
            );
            if archive_path.exists() {
                let _ = std::fs::remove_file(&archive_path);
            }
            let fresh_descriptor = MTLBinaryArchiveDescriptor::new();
            match device.newBinaryArchiveWithDescriptor_error(&fresh_descriptor) {
                Ok(archive) => (Some(archive), Some(archive_path)),
                Err(err2) => {
                    eprintln!(
                        "[alloygbm metal] warning: could not create empty split binary archive: {}",
                        err2.localizedDescription()
                    );
                    (None, None)
                }
            }
        }
    }
}

/// Lowercase + ascii-alphanumeric-or-hyphen slug of a device name.
/// Characters outside `[a-z0-9-]` collapse to a single `-`.
fn slugify_device_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_sep = true;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if !prev_sep {
            out.push('-');
            prev_sep = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

// ---------------------------------------------------------------------
// S3.5 — PartitionPipelineCache
// ---------------------------------------------------------------------
//
// Mirrors `HistogramPipelineCache` / `SplitPipelineCache` but compiles
// `shaders/partition.metal` and specializes on
// `(block_size, split_kind, bin_is_u16)`. Binary archive lives in its
// own file. Three pipeline-state objects per key — one per pass — all
// bundled in `PartitionPipelines` so a single `Arc` clone returns the
// whole set.

pub struct PartitionPipelines {
    pub flag_and_count: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub scan_blocks: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub spec: PartitionSpecKey,
}

// SAFETY: Metal protocol objects held here are thread-safe per Apple's
// documentation; mutable cache state is `Mutex`-guarded.
unsafe impl Send for PartitionPipelineCache {}
// SAFETY: See Send impl.
unsafe impl Sync for PartitionPipelineCache {}

pub struct PartitionPipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    archive: Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    archive_path: Option<PathBuf>,
    dirty: Mutex<bool>,
    entries: Mutex<HashMap<PartitionSpecKey, Arc<PartitionPipelines>>>,
}

impl PartitionPipelineCache {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        capabilities: &MetalCapabilities,
    ) -> Result<Self, String> {
        let source = NSString::from_str(PARTITION_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|err| {
                format!(
                    "partition.metal library compile failed: {}",
                    err.localizedDescription()
                )
            })?;

        let (archive, archive_path) = open_or_create_partition_archive(&device, capabilities);

        Ok(Self {
            device,
            library,
            archive,
            archive_path,
            dirty: Mutex::new(false),
            entries: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_or_build(&self, spec: PartitionSpecKey) -> Result<Arc<PartitionPipelines>, String> {
        {
            let entries = self
                .entries
                .lock()
                .map_err(|_| "partition pipeline cache mutex poisoned".to_string())?;
            if let Some(cached) = entries.get(&spec) {
                return Ok(Arc::clone(cached));
            }
        }

        let pipelines = self.build(spec)?;
        let arc = Arc::new(pipelines);

        let mut entries = self
            .entries
            .lock()
            .map_err(|_| "partition pipeline cache mutex poisoned".to_string())?;
        let stored = entries.entry(spec).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }

    fn build(&self, spec: PartitionSpecKey) -> Result<PartitionPipelines, String> {
        let constants = MTLFunctionConstantValues::new();
        let block_size_cell: u32 = spec.block_size;
        let split_kind_cell: u32 = spec.split_kind;
        let bin_is_u16_cell: u8 = u8::from(spec.bin_is_u16);

        // SAFETY: pointers are to live stack slots; indices 0/1/2
        // match the `[[function_constant(N)]]` declarations in
        // `shaders/partition.metal`.
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const block_size_cell) as *mut c_void),
                MTLDataType::UInt,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const split_kind_cell) as *mut c_void),
                MTLDataType::UInt,
                1,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const bin_is_u16_cell) as *mut c_void),
                MTLDataType::Bool,
                2,
            );
        }

        let make_fn = |name: &str| -> Result<_, String> {
            let ns_name = NSString::from_str(name);
            self.library
                .newFunctionWithName_constantValues_error(&ns_name, &constants)
                .map_err(|err| {
                    format!(
                        "could not specialize `{name}`: {}",
                        err.localizedDescription()
                    )
                })
        };

        let flag_fn = make_fn(KERNEL_NAME_PARTITION_FLAG_AND_COUNT)?;
        let scan_fn = make_fn(KERNEL_NAME_PARTITION_SCAN_BLOCKS)?;
        let scatter_fn = make_fn(KERNEL_NAME_PARTITION_SCATTER)?;

        let mut descs: Vec<Retained<MTLComputePipelineDescriptor>> = Vec::new();
        for f in [&flag_fn, &scan_fn, &scatter_fn] {
            let d = MTLComputePipelineDescriptor::new();
            d.setComputeFunction(Some(f));
            if let Some(archive) = &self.archive {
                let archive_obj: &ProtocolObject<dyn MTLBinaryArchive> = archive;
                let archive_array = NSArray::from_slice(&[archive_obj]);
                d.setBinaryArchives(Some(&archive_array));
            }
            descs.push(d);
        }

        let make_state = |idx: usize, label: &str| -> Result<_, String> {
            self.device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &descs[idx],
                    MTLPipelineOption::empty(),
                    None,
                )
                .map_err(|err| {
                    format!(
                        "partition {label} pipeline creation failed: {}",
                        err.localizedDescription()
                    )
                })
        };

        let flag_and_count = make_state(0, "flag_and_count")?;
        let scan_blocks = make_state(1, "scan_blocks")?;
        let scatter = make_state(2, "scatter")?;

        if let Some(archive) = &self.archive {
            let mut added_any = false;
            for (idx, label) in [(0, "flag_and_count"), (1, "scan_blocks"), (2, "scatter")] {
                match archive.addComputePipelineFunctionsWithDescriptor_error(&descs[idx]) {
                    Ok(_) => added_any = true,
                    Err(err) => eprintln!(
                        "[alloygbm metal] warning: could not add partition {label} pipeline to archive: {}",
                        err.localizedDescription()
                    ),
                }
            }
            if added_any && let Ok(mut dirty) = self.dirty.lock() {
                *dirty = true;
            }
        }

        Ok(PartitionPipelines {
            flag_and_count,
            scan_blocks,
            scatter,
            spec,
        })
    }
}

impl Drop for PartitionPipelineCache {
    fn drop(&mut self) {
        let (archive, path) = match (&self.archive, &self.archive_path) {
            (Some(a), Some(p)) => (a, p),
            _ => return,
        };

        let dirty = self.dirty.lock().map(|g| *g).unwrap_or(false);
        if !dirty {
            return;
        }

        let tmp_path = path.with_extension("metalarchive.tmp");
        let tmp_nsstring = NSString::from_str(&tmp_path.to_string_lossy());
        let tmp_url = NSURL::fileURLWithPath(&tmp_nsstring);

        if let Err(err) = archive.serializeToURL_error(&tmp_url) {
            eprintln!(
                "[alloygbm metal] warning: could not serialize partition binary archive to {}: {}",
                tmp_path.display(),
                err.localizedDescription()
            );
            return;
        }

        if let Err(err) = std::fs::rename(&tmp_path, path) {
            eprintln!(
                "[alloygbm metal] warning: could not rename partition archive {} -> {}: {err}",
                tmp_path.display(),
                path.display()
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

fn partition_archive_filename(capabilities: &MetalCapabilities) -> String {
    let family = if capabilities.metal4 {
        "metal4"
    } else if capabilities.apple7 {
        "apple7"
    } else {
        "generic"
    };
    let device_slug = slugify_device_name(&capabilities.device_name);
    format!("partition-pipelines-{family}-{device_slug}.metalarchive")
}

fn open_or_create_partition_archive(
    device: &ProtocolObject<dyn MTLDevice>,
    capabilities: &MetalCapabilities,
) -> (
    Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    Option<PathBuf>,
) {
    let Some(cache_dir) = user_cache_dir() else {
        return (None, None);
    };
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        eprintln!(
            "[alloygbm metal] warning: could not create cache dir {}: {err}",
            cache_dir.display()
        );
        return (None, None);
    }

    let filename = partition_archive_filename(capabilities);
    let archive_path = cache_dir.join(filename);

    let descriptor = MTLBinaryArchiveDescriptor::new();
    if archive_path.exists() {
        let ns = NSString::from_str(&archive_path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&ns);
        descriptor.setUrl(Some(&url));
    }

    match device.newBinaryArchiveWithDescriptor_error(&descriptor) {
        Ok(archive) => (Some(archive), Some(archive_path)),
        Err(err) => {
            eprintln!(
                "[alloygbm metal] warning: could not open partition binary archive at {}: {}. \
                 Continuing without pipeline persistence.",
                archive_path.display(),
                err.localizedDescription()
            );
            if archive_path.exists() {
                let _ = std::fs::remove_file(&archive_path);
            }
            let fresh_descriptor = MTLBinaryArchiveDescriptor::new();
            match device.newBinaryArchiveWithDescriptor_error(&fresh_descriptor) {
                Ok(archive) => (Some(archive), Some(archive_path)),
                Err(err2) => {
                    eprintln!(
                        "[alloygbm metal] warning: could not create empty partition binary archive: {}",
                        err2.localizedDescription()
                    );
                    (None, None)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// S3.6 — SubtractPipelineCache
// ---------------------------------------------------------------------
//
// Mirrors `HistogramPipelineCache` / `SplitPipelineCache` /
// `PartitionPipelineCache` but compiles `shaders/subtract.metal` and
// specializes on a single function constant (`BLOCK_SIZE`). Separate
// on-disk archive file so the four caches never contend on the same
// `MTLBinaryArchive` object.

pub struct SubtractPipelines {
    pub subtract: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub spec: SubtractSpecKey,
}

// SAFETY: Metal protocol objects held here are thread-safe per Apple's
// documentation; mutable cache state is `Mutex`-guarded.
unsafe impl Send for SubtractPipelineCache {}
// SAFETY: See Send impl.
unsafe impl Sync for SubtractPipelineCache {}

pub struct SubtractPipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    archive: Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    archive_path: Option<PathBuf>,
    dirty: Mutex<bool>,
    entries: Mutex<HashMap<SubtractSpecKey, Arc<SubtractPipelines>>>,
}

impl SubtractPipelineCache {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        capabilities: &MetalCapabilities,
    ) -> Result<Self, String> {
        let source = NSString::from_str(SUBTRACT_SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|err| {
                format!(
                    "subtract.metal library compile failed: {}",
                    err.localizedDescription()
                )
            })?;

        let (archive, archive_path) = open_or_create_subtract_archive(&device, capabilities);

        Ok(Self {
            device,
            library,
            archive,
            archive_path,
            dirty: Mutex::new(false),
            entries: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_or_build(&self, spec: SubtractSpecKey) -> Result<Arc<SubtractPipelines>, String> {
        {
            let entries = self
                .entries
                .lock()
                .map_err(|_| "subtract pipeline cache mutex poisoned".to_string())?;
            if let Some(cached) = entries.get(&spec) {
                return Ok(Arc::clone(cached));
            }
        }

        let pipelines = self.build(spec)?;
        let arc = Arc::new(pipelines);

        let mut entries = self
            .entries
            .lock()
            .map_err(|_| "subtract pipeline cache mutex poisoned".to_string())?;
        let stored = entries.entry(spec).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }

    fn build(&self, spec: SubtractSpecKey) -> Result<SubtractPipelines, String> {
        let constants = MTLFunctionConstantValues::new();
        let block_size_cell: u32 = spec.block_size;

        // SAFETY: `block_size_cell` is a live stack slot whose size and
        // alignment match `MTLDataType::UInt`; index 0 matches the
        // `[[function_constant(0)]]` declaration in
        // `shaders/subtract.metal`.
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::new_unchecked((&raw const block_size_cell) as *mut c_void),
                MTLDataType::UInt,
                0,
            );
        }

        let fn_name = NSString::from_str(KERNEL_NAME_SUBTRACT);
        let subtract_fn = self
            .library
            .newFunctionWithName_constantValues_error(&fn_name, &constants)
            .map_err(|err| {
                format!(
                    "could not specialize `{KERNEL_NAME_SUBTRACT}`: {}",
                    err.localizedDescription()
                )
            })?;

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(&subtract_fn));

        if let Some(archive) = &self.archive {
            let archive_obj: &ProtocolObject<dyn MTLBinaryArchive> = archive;
            let archive_array = NSArray::from_slice(&[archive_obj]);
            desc.setBinaryArchives(Some(&archive_array));
        }

        let subtract = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|err| {
                format!(
                    "subtract pipeline creation failed: {}",
                    err.localizedDescription()
                )
            })?;

        if let Some(archive) = &self.archive {
            if let Err(err) = archive.addComputePipelineFunctionsWithDescriptor_error(&desc) {
                eprintln!(
                    "[alloygbm metal] warning: could not add subtract pipeline to archive: {}",
                    err.localizedDescription()
                );
            } else if let Ok(mut dirty) = self.dirty.lock() {
                *dirty = true;
            }
        }

        Ok(SubtractPipelines { subtract, spec })
    }
}

impl Drop for SubtractPipelineCache {
    fn drop(&mut self) {
        let (archive, path) = match (&self.archive, &self.archive_path) {
            (Some(a), Some(p)) => (a, p),
            _ => return,
        };

        let dirty = self.dirty.lock().map(|g| *g).unwrap_or(false);
        if !dirty {
            return;
        }

        let tmp_path = path.with_extension("metalarchive.tmp");
        let tmp_nsstring = NSString::from_str(&tmp_path.to_string_lossy());
        let tmp_url = NSURL::fileURLWithPath(&tmp_nsstring);

        if let Err(err) = archive.serializeToURL_error(&tmp_url) {
            eprintln!(
                "[alloygbm metal] warning: could not serialize subtract binary archive to {}: {}",
                tmp_path.display(),
                err.localizedDescription()
            );
            return;
        }

        if let Err(err) = std::fs::rename(&tmp_path, path) {
            eprintln!(
                "[alloygbm metal] warning: could not rename subtract archive {} -> {}: {err}",
                tmp_path.display(),
                path.display()
            );
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

fn subtract_archive_filename(capabilities: &MetalCapabilities) -> String {
    let family = if capabilities.metal4 {
        "metal4"
    } else if capabilities.apple7 {
        "apple7"
    } else {
        "generic"
    };
    let device_slug = slugify_device_name(&capabilities.device_name);
    format!("subtract-pipelines-{family}-{device_slug}.metalarchive")
}

fn open_or_create_subtract_archive(
    device: &ProtocolObject<dyn MTLDevice>,
    capabilities: &MetalCapabilities,
) -> (
    Option<Retained<ProtocolObject<dyn MTLBinaryArchive>>>,
    Option<PathBuf>,
) {
    let Some(cache_dir) = user_cache_dir() else {
        return (None, None);
    };
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        eprintln!(
            "[alloygbm metal] warning: could not create cache dir {}: {err}",
            cache_dir.display()
        );
        return (None, None);
    }

    let filename = subtract_archive_filename(capabilities);
    let archive_path = cache_dir.join(filename);

    let descriptor = MTLBinaryArchiveDescriptor::new();
    if archive_path.exists() {
        let ns = NSString::from_str(&archive_path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&ns);
        descriptor.setUrl(Some(&url));
    }

    match device.newBinaryArchiveWithDescriptor_error(&descriptor) {
        Ok(archive) => (Some(archive), Some(archive_path)),
        Err(err) => {
            eprintln!(
                "[alloygbm metal] warning: could not open subtract binary archive at {}: {}. \
                 Continuing without pipeline persistence.",
                archive_path.display(),
                err.localizedDescription()
            );
            if archive_path.exists() {
                let _ = std::fs::remove_file(&archive_path);
            }
            let fresh_descriptor = MTLBinaryArchiveDescriptor::new();
            match device.newBinaryArchiveWithDescriptor_error(&fresh_descriptor) {
                Ok(archive) => (Some(archive), Some(archive_path)),
                Err(err2) => {
                    eprintln!(
                        "[alloygbm metal] warning: could not create empty subtract binary archive: {}",
                        err2.localizedDescription()
                    );
                    (None, None)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Stage 4a — BestSplitPipelineCache
// ---------------------------------------------------------------------
//
// Unlike the histogram/split/partition/subtract caches this is NOT
// parameterized on function constants. `best_split.metal` accepts all
// tunable parameters via a `BestSplitParams` constant buffer at
// dispatch time, so both entry points compile exactly once and live as
// plain `Retained` handles — no inner `HashMap`, no `Arc` wrapping.

/// Compile-once pipeline cache for `best_split.metal`.
/// Holds both entry points: `best_split_per_feature` (sweeps bins
/// per feature per node) and `best_split_reduce_features` (picks the
/// single best feature per node).
pub(crate) struct BestSplitPipelineCache {
    pub(crate) per_feature: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) reduce_features: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// SAFETY: Metal protocol objects are thread-safe per Apple's docs.
unsafe impl Send for BestSplitPipelineCache {}
// SAFETY: see Send impl.
unsafe impl Sync for BestSplitPipelineCache {}

impl BestSplitPipelineCache {
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Self, String> {
        let source = include_str!("shaders/best_split.metal");
        let nss = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&nss, None)
            .map_err(|e| format!("best_split.metal compile failed: {}", e.localizedDescription()))?;

        let per_feature_fn = library
            .newFunctionWithName(&NSString::from_str("best_split_per_feature"))
            .ok_or_else(|| "best_split.metal: missing best_split_per_feature".to_string())?;
        let reduce_fn = library
            .newFunctionWithName(&NSString::from_str("best_split_reduce_features"))
            .ok_or_else(|| "best_split.metal: missing best_split_reduce_features".to_string())?;

        let per_feature = device
            .newComputePipelineStateWithFunction_error(&per_feature_fn)
            .map_err(|e| format!("best_split_per_feature pipeline failed: {}", e.localizedDescription()))?;
        let reduce_features = device
            .newComputePipelineStateWithFunction_error(&reduce_fn)
            .map_err(|e| format!("best_split_reduce_features pipeline failed: {}", e.localizedDescription()))?;

        Ok(Self { per_feature, reduce_features })
    }
}

// ---------------------------------------------------------------------
// Stage 4b — IcbPipelineCache
// ---------------------------------------------------------------------
//
// Compiles `shaders/icb_tree.metal` once and builds three pipeline
// states: `icb_histogram`, `icb_split_find`, `icb_partition`.  Unlike
// the earlier parameterized caches these kernels take all tunable
// parameters via a constant buffer bound at encode time, so no
// function constants and no per-key HashMap — just three plain
// `Retained` handles compiled once at construction.

/// Compiled-once compute pipeline states for Stage 4b ICB tree kernels.
pub(crate) struct IcbPipelineCache {
    pub(crate) histogram:   Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) split_find:  Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) partition:   Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// SAFETY: Metal protocol objects are thread-safe per Apple's docs.
unsafe impl Send for IcbPipelineCache {}
// SAFETY: see Send impl.
unsafe impl Sync for IcbPipelineCache {}

impl IcbPipelineCache {
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Self, String> {
        let source = include_str!("shaders/icb_tree.metal");
        let nss    = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&nss, None)
            .map_err(|e| format!("icb_tree.metal compile failed: {}", e.localizedDescription()))?;

        let histogram_fn = library
            .newFunctionWithName(&NSString::from_str("icb_histogram"))
            .ok_or_else(|| "icb_tree.metal: missing icb_histogram".to_string())?;
        let split_find_fn = library
            .newFunctionWithName(&NSString::from_str("icb_split_find"))
            .ok_or_else(|| "icb_tree.metal: missing icb_split_find".to_string())?;
        let partition_fn = library
            .newFunctionWithName(&NSString::from_str("icb_partition"))
            .ok_or_else(|| "icb_tree.metal: missing icb_partition".to_string())?;

        let histogram  = device
            .newComputePipelineStateWithFunction_error(&histogram_fn)
            .map_err(|e| format!("icb_histogram pipeline failed: {}", e.localizedDescription()))?;
        let split_find = device
            .newComputePipelineStateWithFunction_error(&split_find_fn)
            .map_err(|e| format!("icb_split_find pipeline failed: {}", e.localizedDescription()))?;
        let partition  = device
            .newComputePipelineStateWithFunction_error(&partition_fn)
            .map_err(|e| format!("icb_partition pipeline failed: {}", e.localizedDescription()))?;

        Ok(Self { histogram, split_find, partition })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_handles_common_device_names() {
        assert_eq!(slugify_device_name("Apple M2 Pro"), "apple-m2-pro");
        assert_eq!(slugify_device_name("Apple M1"), "apple-m1");
        assert_eq!(slugify_device_name("   "), "unknown");
        assert_eq!(slugify_device_name("Apple_M4/Max"), "apple-m4-max");
    }

    #[test]
    fn archive_filename_encodes_family_and_device() {
        let caps = MetalCapabilities {
            apple7: true,
            metal4: false,
            device_name: "Apple M2 Pro".to_string(),
        };
        assert_eq!(
            archive_filename(&caps),
            "pipelines-apple7-apple-m2-pro.metalarchive"
        );

        let caps_m4 = MetalCapabilities {
            apple7: true,
            metal4: true,
            device_name: "Apple M4 Max".to_string(),
        };
        assert_eq!(
            archive_filename(&caps_m4),
            "pipelines-metal4-apple-m4-max.metalarchive"
        );
    }
}
