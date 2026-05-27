//! Joint shared-tree multi-output trainer (v0.10.0).
//!
//! Grows one shared tree per round whose splits minimize the sum of per-output
//! split gain (across K outputs) and whose leaves carry K independent
//! Newton-Raphson values. Each output may use its own scalar objective (one of
//! the supported objectives listed in [`JointObjective`]).
//!
//! ## Scope
//!
//! v0.10.0 ships a minimal implementation:
//! - Level-wise tree growth (no leaf-wise / best-first)
//! - Standard boosting only (no DART / GOSS)
//! - No MorphBoost, no DRO, no neutralization, no leaf-wise
//! - No warm-start
//! - No native categorical splits (categorical features are honored at the
//!   binning layer but split semantics use the standard threshold path)
//! - No interaction constraints
//!
//! Richer feature coverage on the joint path will land in v0.10.x point
//! releases. See `docs/limitations.md` for the active follow-up list.

mod build_round;
mod fit;
mod helpers;
mod types;

pub use build_round::build_joint_round;
pub use fit::{
    build_joint_metadata, fit_joint_multi_output, fit_joint_multi_output_with_categorical,
    fit_joint_multi_output_with_warm_start,
};
pub use types::{
    JointObjective, JointPredictor, JointRoundResult, JointTrainingSummary, JointWarmStartState,
};

/// Tree-node-id stride used by the engine (1 << 20). Must match
/// `crate::TREE_NODE_STRIDE`; duplicated here as a `pub(super)` constant
/// because the engine's copy is private.
pub(super) const TREE_NODE_STRIDE: u32 = 1 << 20;

#[cfg(test)]
mod tests;
