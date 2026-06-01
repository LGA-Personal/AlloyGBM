//! Joint shared-tree multi-output trainer.
//!
//! Grows one shared tree per round whose splits minimize the sum of per-output
//! split gain (across K outputs) and whose leaves carry K independent
//! Newton-Raphson values. Each output may use its own scalar objective (one
//! of the supported objectives listed in [`JointObjective`]). The K-output
//! histogram primitive is `crate::shared_histogram::MultiOutputHistogram`;
//! single-output training (`crate::trainer::Trainer`) uses a different
//! per-feature histogram type and remains the path for K = 1.
//!
//! ## Capabilities (current, as of v0.12.6)
//!
//! The joint path reached feature parity with the single-output trainer
//! over the v0.10.x line; the original v0.10.0 minimal scope is long
//! superseded. Currently supported:
//!
//! - **Tree growth**: level-wise (default) and leaf-wise / best-first via
//!   `tree_growth = "leaf"` + `max_leaves` (v0.10.2).
//! - **Boosting modes**: standard, GOSS (per-round mask via
//!   `select_joint_row_indices_for_round` in `helpers`, v0.10.3), and DART
//!   (full subtract / replay cycle with per-tree weights persisted to the
//!   artifact, v0.10.3).
//! - **Leaf solvers**: Newton-Raphson and DRO (`leaf_solver = "dro"`, the
//!   K-output Newton step routes through
//!   `alloygbm_core::leaf_effective_gradient`, v0.10.5).
//! - **MorphBoost**: per-iteration LR schedule + per-leaf depth penalty +
//!   per-iteration leaf shrinkage; gradient EMA persisted in the artifact
//!   (v0.10.4).
//! - **Factor neutralization**: `pre_target`, `per_round_gradient`, and
//!   `split_penalty` modes (v0.10.6).
//! - **Native categorical splits**: Fisher-sort over K outputs with `u64`
//!   bitset encoding (Rust v0.10.2, Python wiring v0.10.3).
//! - **Interaction constraints**: per-node bitset narrowing, shared with
//!   the single-output trainer's `InteractionConstraintIndex` (v0.10.2).
//! - **Row / column subsampling**: seeded per-round masks (`global_round`
//!   is mixed into the seed for warm-start determinism, v0.10.2).
//! - **Warm-start**: [`JointWarmStartState`] replays prior stumps,
//!   reconstructs DART per-tree weights, and re-seeds MorphBoost EMA
//!   continuity across the resume boundary (v0.10.3 / v0.10.4).
//!
//! See `docs/limitations.md` for the active follow-up list and `CLAUDE.md`
//! for the per-feature implementation references with their release tags.
//!
//! ## Module layout (post-v0.12.2 decomposition)
//!
//! This `mod.rs` is the scaffolding / re-export layer added in v0.12.2
//! (PR #46). The actual implementation lives in sibling modules:
//!
//! - `helpers` — private RNG, row-sampling, factor-sums, iteration helpers
//! - `types` — [`JointObjective`], [`JointPredictor`],
//!   [`JointWarmStartState`], and the per-round / per-leaf data structures
//! - `build_round` — level-wise and leaf-wise round builders
//! - `fit` — `fit_joint_multi_output*` drivers and metadata serialization
//! - `tests` — unit tests (gated by `#[cfg(test)]`)

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
