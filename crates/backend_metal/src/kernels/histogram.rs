//! Histogram kernel — Rust-side holder for the MSL source.
//!
//! Dispatch orchestration (buffer wrapping, encoding, submit, readback)
//! lands in S1.4 alongside the `BackendOps::build_histograms` impl.
//! Pipeline compilation + caching lands in S1.5.

/// Embedded MSL source for the two-pass histogram build.
///
/// Exposes two entry points: `histogram_build_scatter` (per-threadgroup
/// scatter into a chunked device-memory scratch buffer) and
/// `histogram_reduce` (cross-chunk ascending reduce into the final
/// `HistogramBundle`). See `shaders/histogram.metal` for the full
/// specification.
pub const HISTOGRAM_SHADER_SOURCE: &str = include_str!("../shaders/histogram.metal");

/// Entry-point name for pass 1. Matched against `newFunctionWithName:`
/// when building the pipeline state in S1.5.
pub const KERNEL_NAME_SCATTER: &str = "histogram_build_scatter";

/// Entry-point name for pass 2.
pub const KERNEL_NAME_REDUCE: &str = "histogram_reduce";
