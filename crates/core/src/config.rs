use crate::dro::DroConfig;
use crate::error::{CoreError, CoreResult};
use crate::neutralization::FactorNeutralizationConfig;
use crate::training_mode::MorphConfig;

/// Upper bound on the number of heap-style local node slots a single tree may
/// occupy, shared by the trainer and the predictor so the contract is
/// symmetric: any model the trainer emits (local node id `< MAX_TREE_NODE_SLOTS`)
/// is guaranteed to load, and any artifact the predictor accepts could have been
/// trained.  The trainer enforces this via `encode_tree_node_id` (a split whose
/// child would land at slot `>= MAX_TREE_NODE_SLOTS` fails fit); the predictor
/// enforces it before allocating the per-tree `nodes_by_local_id` array, so a
/// crafted/corrupt artifact cannot force an oversized allocation.
///
/// Node ids are heap-indexed (`left = 2i+1`, `right = 2i+2`), so this caps
/// effective tree depth at ~16 while still permitting far more leaves than any
/// realistic GBDT tree.  It is deliberately tighter than the tree-node encoding
/// stride (`1 << 20`), which remains the multiplier separating per-tree id
/// ranges.
pub const MAX_TREE_NODE_SLOTS: usize = 1 << 16;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeGrowth {
    /// Level-wise (breadth-first): split all nodes at depth d before depth d+1.
    #[default]
    Level,
    /// Leaf-wise (best-first): always split the leaf with the highest gain.
    Leaf,
}

/// How a multiclass softmax fit grows its trees each round.
///
/// `PerClassTrees` (default) grows K independent trees per round, one per
/// class, each scored on that class's gradients alone. `VectorLeaves` grows
/// one tree per round whose splits are scored on all K classes at once and
/// whose leaves hold a K-vector (one Newton step per class). The vector tree
/// is stored as K per-class trees that share its structure, so artifacts,
/// prediction and SHAP are unchanged. Ignored by every other objective.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MulticlassStrategy {
    #[default]
    PerClassTrees,
    VectorLeaves,
}

/// Which Hessian a multiclass softmax fit uses for its Newton steps.
///
/// `Diagonal` (default) uses `p_k (1 - p_k)` per class, the softmax
/// Hessian's diagonal. `Full` (vector leaves only) solves each leaf with the
/// full softmax Hessian `diag(p) - p p^T`, summed over the leaf's rows;
/// splits are still scored on the diagonal. `AbsResidual` replaces the
/// Hessian with `|p_k - y_k|`, a pseudo-Hessian that bounds each row's step.
/// Ignored by every other objective.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MulticlassHessian {
    #[default]
    Diagonal,
    Full,
    AbsResidual,
}

/// How a vector-leaf multiclass fit scores its splits (SketchBoost,
/// Iosipoi & Vakhrushev, NeurIPS 2022).
///
/// `None` (default) scores splits on all K class gradients. The other modes
/// score them on a sketch of [`TrainParams::multiclass_sketch_dim`] columns
/// derived from the K gradients, which cuts histogram building and split
/// scanning from K columns to the sketch's width. Leaves still solve on all K
/// classes. Only applies with [`MulticlassStrategy::VectorLeaves`].
///
/// - `TopOutputs`: the columns with the largest gradient norms, each with
///   its own Hessian.
/// - `RandomSampling`: columns drawn with probability proportional to their
///   gradient norm, rescaled so the sketched gain is unbiased; Hessians are
///   the per-row mean over classes.
/// - `RandomProjection`: the gradient matrix times a Gaussian `K x d`
///   matrix scaled by `1/sqrt(d)`; Hessians are the per-row mean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MulticlassSketch {
    #[default]
    None,
    TopOutputs,
    RandomSampling,
    RandomProjection,
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

/// Curvature used by the log-link objectives (Poisson, Gamma, Tweedie).
///
/// `Exact` (default) keeps today's Hessians: Poisson `μ·exp(max_delta_step)`,
/// Gamma `y/μ`, Tweedie `μ^(2-p)`. `Bounded` uses the bounded-step
/// pseudo-Hessians from XGBoost (dmlc/xgboost#12538): a convex blend of the
/// exact curvature and its `y`-free counterpart that caps each Newton leaf
/// step (Poisson steps lie in `(-3/2, 3)`, Gamma steps in `(-3, 3/2)`) and
/// keeps the quadratic gain from overstating the real loss reduction.
/// `Bounded` ignores `poisson_max_delta_step`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmHessian {
    #[default]
    Exact,
    Bounded,
}

impl GlmHessian {
    pub fn parse(value: &str) -> CoreResult<Self> {
        match value {
            "exact" => Ok(Self::Exact),
            "bounded" => Ok(Self::Bounded),
            other => Err(CoreError::InvalidConfig(format!(
                "glm_hessian must be 'exact' or 'bounded' (got '{other}')"
            ))),
        }
    }
}

/// How the gradient-regularized Newton (GRN) term enters the leaf solve.
///
/// Each round GRN computes `λ_k = grn_strength · sqrt(‖g_k‖)`, with
/// `‖g_k‖ = sqrt(mean_i g_i²)` over that round's gradients (Mishchenko 2023).
/// `PerLeaf` adds `λ_k` to the split and leaf L2 denominator
/// (`-G / (H + λ + λ_k)`); `PerRow` adds `λ_k·w_i` to every row's Hessian,
/// which is the paper's per-row regularizer (`-G / (H + λ_k·Σw)` in a leaf).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GrnForm {
    #[default]
    PerLeaf,
    PerRow,
}

impl GrnForm {
    pub fn parse(value: &str) -> CoreResult<Self> {
        match value {
            "per_leaf" => Ok(Self::PerLeaf),
            "per_row" => Ok(Self::PerRow),
            other => Err(CoreError::InvalidConfig(format!(
                "grn_form must be 'per_leaf' or 'per_row' (got '{other}')"
            ))),
        }
    }
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
/// * `Dart { drop_rate, max_drop, normalize_type, sample_type, skip_drop }`:
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
        /// Probability that a round performs no dropout at all. Rolled once
        /// per round, independently of the per-tree dropout draws. LightGBM
        /// calls this `skip_drop` and defaults it to 0.5: skipping dropout
        /// on some rounds lets the ensemble consolidate, which both speeds
        /// training and typically improves accuracy on clean data. `0.0`
        /// reproduces always-drop behaviour.
        skip_drop: f32,
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
    /// Poisson max-delta-step stabilizer. The hessian is scaled by
    /// `exp(poisson_max_delta_step)` to damp Newton updates for sparse or
    /// skewed count data. Defaults to 0.7.
    pub poisson_max_delta_step: f32,
    /// Quantile alpha `alpha ∈ (0.0, 1.0)` for `objective="quantile"`.
    /// Ignored for all other objectives.
    /// Defaults to 0.5 (median).
    pub quantile_alpha: f32,
    /// Number of numeric features shortlisted by scalar gain for piecewise-linear
    /// split rescoring. Zero preserves scalar split selection; ignored for constant leaves.
    pub pl_split_candidates: usize,
    /// Per-node feature subsampling fraction (`colsample_bynode`). At each
    /// split-search node an independent deterministic subset of this fraction
    /// of features is drawn as candidates, composing with `col_subsample`
    /// (per-tree) and interaction constraints. `1.0` disables it (inert).
    pub colsample_bynode: f32,
    /// Opt-in: make joint multi-output DRO split *selection* robust by carrying
    /// per-bin `grad_sq` and scoring splits through the DRO effective gradient
    /// (numeric threshold splits only). `false` keeps the standard Newton split
    /// gain (DRO applies to leaf values only), which is byte-identical to prior
    /// behavior. Costs ~1.5× joint-histogram memory when enabled. Only takes
    /// effect on the joint multi-output trainer when `leaf_solver == Dro` and
    /// `dro_config` has a nonzero radius (see `joint::helpers::effective_dro_config`);
    /// otherwise it is a no-op. Native-categorical and MorphBoost joint splits
    /// are unaffected even when this flag is on (deferred scope).
    pub joint_dro_robust_split_gain: bool,
    /// Relative validation tolerance for early stopping. When positive (and
    /// `early_stopping_rounds` is set), training also stops once
    /// `early_stopping_rounds` consecutive rounds have failed to beat the best
    /// validation loss so far by this fraction of it. The model is still
    /// truncated to the best round under `min_validation_improvement`; this
    /// only ends a long, nearly flat tail sooner. `0.0` (default) disables it.
    /// `f64` so a caller replaying the same rule on the reported curve in
    /// double precision makes identical decisions.
    pub early_stopping_relative_tolerance: f64,
    /// Split-score noise strength (CatBoost's `random_strength`). When a node
    /// picks its split, each feature's best candidate competes on its gain
    /// times `max(0, 1 + strength * z)` with `z` a deterministic unit normal;
    /// the chosen split keeps its true gain. `None` (default) lets the
    /// training policy decide (auto: 1.0 for multiclass softmax, else 0);
    /// `Some(s)` uses `s` under either policy, and `Some(0.0)` turns it off.
    /// Applies to the single-output and multiclass trainers on the standard
    /// split path, not to MorphBoost, piecewise-linear shortlists or the joint
    /// multi-output trainer.
    pub split_noise_strength: Option<f32>,
    /// Tree layout for multiclass softmax; see [`MulticlassStrategy`].
    /// `PerClassTrees` (default) keeps the existing K-trees-per-round fit.
    pub multiclass_strategy: MulticlassStrategy,
    /// Newton Hessian for multiclass softmax; see [`MulticlassHessian`].
    /// `Diagonal` (default) keeps the existing leaf and split math.
    pub multiclass_hessian: MulticlassHessian,
    /// Curvature for the log-link objectives. `Exact` (default) is unchanged
    /// behavior; ignored by every other objective. See [`GlmHessian`].
    pub glm_hessian: GlmHessian,
    /// Gradient-regularized Newton strength (`C·sqrt(M)` in the paper). `0.0`
    /// (default) disables GRN and leaves training byte-identical. Applies to
    /// the single-output and multiclass trainers, not the joint multi-output
    /// trainer.
    pub grn_strength: f32,
    /// Where the GRN term enters the leaf solve. Ignored when
    /// `grn_strength == 0`. See [`GrnForm`].
    pub grn_form: GrnForm,
    /// Quantized-gradient histograms (LightGBM's `use_quantized_grad`):
    /// `Some(bits)` stochastically rounds each round's gradients and
    /// Hessians to `bits`-bit integers (2..=8) for histogram building and
    /// split scoring; leaf values still use the float gradients. `None`
    /// (default) leaves training byte-identical. Supported on the standard
    /// split path of the single-output and multiclass trainers; rejected
    /// with DRO leaves, linear leaves, MorphBoost and split-penalty
    /// neutralization.
    pub quantized_grad_bits: Option<u8>,
    /// Split-scoring sketch for vector-leaf multiclass; see
    /// [`MulticlassSketch`]. `None` (default) leaves training unchanged.
    pub multiclass_sketch: MulticlassSketch,
    /// Sketch width (columns that score splits). Must be at least 1 when a
    /// sketch is set; a width of K or more scores on all K classes.
    pub multiclass_sketch_dim: usize,
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
            poisson_max_delta_step: 0.7,
            quantile_alpha: 0.5,
            pl_split_candidates: 0,
            colsample_bynode: 1.0,
            joint_dro_robust_split_gain: false,
            early_stopping_relative_tolerance: 0.0,
            split_noise_strength: None,
            multiclass_strategy: MulticlassStrategy::PerClassTrees,
            multiclass_hessian: MulticlassHessian::Diagonal,
            glm_hessian: GlmHessian::Exact,
            grn_strength: 0.0,
            grn_form: GrnForm::PerLeaf,
            quantized_grad_bits: None,
            multiclass_sketch: MulticlassSketch::None,
            multiclass_sketch_dim: 5,
        }
    }
}
