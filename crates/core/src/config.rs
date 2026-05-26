use crate::dro::DroConfig;
use crate::error::{CoreError, CoreResult};
use crate::neutralization::FactorNeutralizationConfig;
use crate::training_mode::MorphConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeGrowth {
    /// Level-wise (breadth-first): split all nodes at depth d before depth d+1.
    #[default]
    Level,
    /// Leaf-wise (best-first): always split the leaf with the highest gain.
    Leaf,
}

/// Leaf representation strategy for tree models.
///
/// `Constant` (default) is identical to current behavior: each leaf stores a
/// single scalar output `f_s = -lr * Σg / (Σh + λ)`.
///
/// `Linear` replaces the scalar with a small ridge-regression model
/// `f_s(x) = b_s + Σ_j α_j x_{k_j}` fit analytically via the closed-form
/// Newton step `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.  Feature regressors are chosen
/// incrementally as the tree grows (inheriting the parent's set plus the
/// current split feature, capped at `min(8, max_depth)`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeafModelKind {
    /// Single scalar leaf value (current default behavior).
    #[default]
    Constant,
    /// Piecewise-linear leaf model fit by ridge regression.
    Linear,
}

/// Scalar leaf solver strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeafSolverKind {
    /// Standard Newton leaf solve using empirical gradient/hessian sums.
    #[default]
    Standard,
    /// Distributionally robust leaf solve over within-leaf gradient dispersion.
    Dro,
}

/// DART tree-weight normalization policy.  Mirrors LightGBM/MART
/// terminology — see `crates/engine/src/dart.rs` for how the policy
/// affects per-round leaf-weight rescaling.
///
/// * `Tree`: each new tree is scaled by `1 / (K + 1)` and each of the
///   K dropped trees by `K / (K + 1)`.  Keeps the cumulative ensemble
///   prediction unbiased after the per-round dropout swap.
/// * `Forest`: applies a forest-wide rescaling factor; commonly used
///   when the user expects DART to behave more like a random-forest
///   ensemble than a boosted ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DartNormalize {
    Tree,
    Forest,
}

/// DART dropout sampling strategy.
///
/// * `Uniform`: each of the existing K trees is dropped with
///   probability `drop_rate`, capped at `max_drop`.
/// * `Weighted`: drop probability is weighted by per-tree contribution
///   magnitude so larger-impact trees are more likely to be dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DartSampleType {
    Uniform,
    Weighted,
}

/// Per-round boosting strategy.
///
/// * `Standard` (default): byte-identical to v0.7.5 behaviour —
///   uniform row subsampling under `row_subsample`, no per-round
///   tree dropout.
/// * `Goss { top_rate, other_rate }`: gradient-based one-side
///   sampling — keep the top `top_rate` rows by `|gradient|`, sample
///   `other_rate` from the rest, and amplify the small-gradient rows
///   by `(1 - top_rate) / other_rate` to maintain unbiased gradient
///   sums.  See `crates/engine/src/sampling.rs`.
/// * `Dart { drop_rate, max_drop, normalize_type, sample_type }`:
///   per-round tree dropout — drop K existing trees before computing
///   gradients, fit a new tree, then rescale per `normalize_type`.
///   Requires per-stump `tree_weight: f32` in the artifact (back-compat:
///   missing field defaults to 1.0).  See `crates/engine/src/dart.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoostingMode {
    Standard,
    Goss {
        top_rate: f32,
        other_rate: f32,
    },
    Dart {
        drop_rate: f32,
        max_drop: usize,
        normalize_type: DartNormalize,
        sample_type: DartSampleType,
    },
}

impl BoostingMode {
    /// Stable string label for artifact metadata and Python-side error
    /// messages.  Matches the `boosting_mode=` ctor strings on the
    /// estimators.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Goss { .. } => "goss",
            Self::Dart { .. } => "dart",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn as_metadata_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }

    pub fn parse_metadata_label(value: &str) -> CoreResult<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            other => Err(CoreError::Validation(format!(
                "unsupported trained_device '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    pub seed: u64,
    pub deterministic: bool,
    pub learning_rate: f32,
    pub max_depth: u16,
    pub row_subsample: f32,
    pub col_subsample: f32,
    pub early_stopping_rounds: Option<u16>,
    pub min_validation_improvement: f32,
    pub min_data_in_leaf: u32,
    pub lambda_l1: f32,
    pub lambda_l2: f32,
    pub min_child_hessian: f32,
    pub min_split_gain: f32,
    /// Per-feature monotone constraints: +1 non-decreasing, -1 non-increasing, 0 unconstrained.
    /// Empty means no constraints.
    pub monotone_constraints: Vec<i8>,
    /// Per-feature importance weights for split selection (gain is multiplied by weight).
    /// Empty means uniform weighting.
    pub feature_weights: Vec<f32>,
    /// Interaction constraints (LightGBM-compatible semantics).  Each inner
    /// `Vec` is a group of feature indices that are allowed to co-occur on
    /// any root-to-leaf path.  Features that don't appear in any group are
    /// unconstrained and may be used freely alongside any group.  Empty
    /// outer `Vec` means no constraints — equivalent to the v0.7.0
    /// behaviour.  Limit: up to 64 groups per fit (a u64 bitset tracks the
    /// active set per node).
    pub interaction_constraints: Vec<Vec<u32>>,
    /// Maximum number of leaves per tree. None means depth-limited only.
    pub max_leaves: Option<usize>,
    /// Tree growth strategy: level-wise (default) or leaf-wise (best-first).
    pub tree_growth: TreeGrowth,
    /// MorphBoost-inspired training profile config. `None` = non-morph (current behavior).
    pub morph_config: Option<MorphConfig>,
    /// Leaf representation strategy.  `Constant` (default) preserves all existing
    /// behaviour.  `Linear` enables piecewise-linear leaves fitted by ridge regression.
    pub leaf_model: LeafModelKind,
    /// Scalar leaf solver strategy. `Standard` preserves existing behavior.
    pub leaf_solver: LeafSolverKind,
    /// Configuration for `leaf_solver == Dro`.
    pub dro_config: Option<DroConfig>,
    pub neutralization_config: Option<FactorNeutralizationConfig>,
    /// v0.8.0: per-round boosting strategy.  Default `Standard`
    /// preserves v0.7.5 behaviour exactly.  See [`BoostingMode`] for
    /// the GOSS / DART semantics.
    pub boosting_mode: BoostingMode,
    /// v0.11.0: Tweedie variance power `p ∈ (1, 2)` for
    /// `objective="tweedie"`.  Ignored for all other objectives.
    /// Defaults to 1.5 (a common starting point for insurance/claims data).
    pub tweedie_variance_power: f32,
    /// Quantile alpha `alpha ∈ (0.0, 1.0)` for `objective="quantile"`.
    /// Ignored for all other objectives.
    /// Defaults to 0.5 (median).
    pub quantile_alpha: f32,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            deterministic: true,
            learning_rate: 0.1,
            max_depth: 6,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
            monotone_constraints: Vec::new(),
            feature_weights: Vec::new(),
            interaction_constraints: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
            leaf_model: LeafModelKind::Constant,
            leaf_solver: LeafSolverKind::Standard,
            dro_config: None,
            neutralization_config: None,
            boosting_mode: BoostingMode::Standard,
            tweedie_variance_power: 1.5,
            quantile_alpha: 0.5,
        }
    }
}
