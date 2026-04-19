//! Rust-side holders for MSL kernels.
//!
//! Each submodule exposes the raw shader source as a `&'static str`. The
//! compilation + pipeline-caching plumbing lives in `pipelines.rs`
//! (arriving in S1.5); the dispatch orchestration lives alongside each
//! source module (arriving in S1.4+).

pub mod histogram;
