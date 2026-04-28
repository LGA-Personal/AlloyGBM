//! Rust-side holders for MSL kernels.
//!
//! Each submodule exposes the raw shader source as a `&'static str`. The
//! compilation + pipeline-caching plumbing lives in `pipelines.rs`; the
//! dispatch orchestration lives alongside each source module.

pub mod best_split;
pub mod histogram;
pub mod partition;
pub mod split;
pub mod subtract;

#[cfg(target_os = "macos")]
pub mod icb_tree;
