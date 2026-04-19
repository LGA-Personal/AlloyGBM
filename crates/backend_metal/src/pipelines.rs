//! Compile the histogram MSL kernel and build the pair of specialized
//! `MTLComputePipelineState`s that S1.4 dispatches against.
//!
//! S1.4 builds a fresh pair on every call — good enough for correctness.
//! Pipeline caching via `MTLBinaryArchive` arrives in S1.5.

#![allow(unsafe_code)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::kernels::histogram::{HISTOGRAM_SHADER_SOURCE, KERNEL_NAME_REDUCE, KERNEL_NAME_SCATTER};

/// A pair of compute pipelines specialized for a given `(bin_count,
/// use_u16_bins)` pair. Both pipelines share the same underlying
/// `MTLLibrary`; the function constants baked at pipeline create time
/// drive the kernel's internal branches and buffer layout.
pub struct HistogramPipelines {
    pub scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub reduce: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub bin_count: u32,
    pub use_u16_bins: bool,
}

/// Compile the histogram library, inject function constants for
/// `BIN_COUNT` (index 0) and `USE_U16_BINS` (index 1), and build the
/// scatter + reduce pipeline states.
pub fn build_histogram_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    bin_count: u32,
    use_u16_bins: bool,
) -> Result<HistogramPipelines, String> {
    let source = NSString::from_str(HISTOGRAM_SHADER_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|err| {
            format!(
                "histogram.metal library compile failed: {}",
                err.localizedDescription()
            )
        })?;

    // Inject function constants. `BIN_COUNT` is a `uint`; `USE_U16_BINS`
    // is a `bool`. Metal expects a bool to be a single byte (0/1). The
    // pointer must outlive the `setConstantValue_type_atIndex` call; we
    // bind the temporaries to stack slots and hold them for the scope of
    // this function.
    let constants = MTLFunctionConstantValues::new();
    let bin_count_cell: u32 = bin_count;
    let use_u16_cell: u8 = u8::from(use_u16_bins);

    // SAFETY: both pointers are to live stack slots with the matching
    // `MTLDataType` size; indices 0 and 1 match the `[[function_constant(N)]]`
    // declarations in `shaders/histogram.metal`.
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
    let reduce_name = NSString::from_str(KERNEL_NAME_REDUCE);

    let scatter_fn = library
        .newFunctionWithName_constantValues_error(&scatter_name, &constants)
        .map_err(|err| {
            format!(
                "could not specialize `{KERNEL_NAME_SCATTER}`: {}",
                err.localizedDescription()
            )
        })?;
    let reduce_fn = library
        .newFunctionWithName_constantValues_error(&reduce_name, &constants)
        .map_err(|err| {
            format!(
                "could not specialize `{KERNEL_NAME_REDUCE}`: {}",
                err.localizedDescription()
            )
        })?;

    let scatter = device
        .newComputePipelineStateWithFunction_error(&scatter_fn)
        .map_err(|err| {
            format!(
                "scatter pipeline creation failed: {}",
                err.localizedDescription()
            )
        })?;
    let reduce = device
        .newComputePipelineStateWithFunction_error(&reduce_fn)
        .map_err(|err| {
            format!(
                "reduce pipeline creation failed: {}",
                err.localizedDescription()
            )
        })?;

    Ok(HistogramPipelines {
        scatter,
        reduce,
        bin_count,
        use_u16_bins,
    })
}
