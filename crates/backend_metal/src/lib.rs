//! Metal GPU backend for AlloyGBM on Apple Silicon.
//!
//! Scaffolded in S1.1. Real implementation arrives incrementally across
//! Stage 1 sub-tasks — see `docs/metal-backend/STATUS.md`.

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MetalBackend;

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }
}
