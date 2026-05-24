use alloygbm_categorical::{TargetEncoderConfig, fit_transform_target_encoder};
use alloygbm_core::{
    BinnedMatrix, BoostingMode, CategoricalStatePayloadV1, DartTreeWeightsPayload,
    DatasetMatrix, Device, DroMetadataPayload, FactorExposureMatrix,
    FeatureBaselinePayload, FeatureTile, GradientEmaStats, GradientPair, HistogramBundle,
    LeafModelKind, LeafSolverKind, LeafValue, LinearLeaf,
    LinearLeafCoefficientsPayload, LinearLeafEntry, MAX_PL_REGRESSORS, MISSING_BIN_U8,
    MODEL_FORMAT_V1, ModelArtifactSection, ModelMetadata, ModelSectionKind,
    MorphMetadataPayload, NativeCategoricalSplitsPayload, NeutralizationKind,
    NeutralizationMetadataPayload, NodeSlice, NodeStats, PartitionResult, SplitCandidate,
    TrainParams, TrainingDataset, TreeGrowth, decode_optional_categorical_state_section_v1,
    decode_optional_dart_tree_weights_section, decode_optional_dro_metadata_artifact_section,
    decode_optional_feature_baseline_section, decode_optional_linear_leaf_coefficients_section,
    decode_optional_morph_metadata_artifact_section,
    decode_optional_native_categorical_splits_section,
    decode_optional_neutralization_metadata_artifact_section, deserialize_model_artifact_v1,
    encode_categorical_state_payload_v1, encode_dart_tree_weights_payload,
    encode_dro_metadata_payload, encode_feature_baseline_payload,
    encode_linear_leaf_coefficients_payload, encode_morph_metadata_payload,
    encode_native_categorical_splits_payload, encode_neutralization_metadata_payload,
    format_required_section_auto_mode_error, format_required_section_mode_error,
    leaf_effective_gradient, required_section_compatibility_report, serialize_model_artifact_v1,
    validate_binned_matrix, validate_categorical_state_payload_v1, validate_train_params,
    validate_training_dataset,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

mod error;
pub use error::{EngineError, EngineResult};

mod env;
mod tree_node;
pub(crate) use tree_node::*;

use env::{
    experiment_force_manual_policy_enabled, experiment_leaf_refinement_enabled,
    split_l2_env_is_configured, split_selection_options_from_env,
};

pub mod dart;
pub use dart::{DartState, apply_normalization, select_dropouts};

pub mod shared_histogram;
pub use shared_histogram::{
    HistComponent, MultiOutputHistogram, build_multi_output_histogram_inplace,
    compute_multi_output_split_gain, subtract_multi_output_histogram,
};

pub mod joint;
pub use joint::{JointObjective, JointRoundResult, JointWarmStartState, build_joint_round};

mod morph_state;
pub use morph_state::{MorphState, resolve_lr_schedule};
pub(crate) use morph_state::MorphTreeContext;

mod factor;
pub(crate) use factor::{FactorProjector, apply_pre_target_neutralization};

mod split_options;
pub use split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, MorphContext, SplitSelectionOptions,
};

mod traits;
pub use traits::{BackendOps, ObjectiveOps, PerRoundMetricCallback};

mod objectives;
pub use objectives::{
    BinaryCrossEntropyObjective, GammaObjective, LambdaMARTObjective, MultiClassSoftmaxObjective,
    PairwiseRankingObjective, PoissonObjective, QuantileObjective, QueryRMSEObjective,
    SquaredErrorObjective, TweedieObjective, XeNDCGObjective, YetiRankObjective,
    compute_group_boundaries,
};
pub(crate) use objectives::weighted_quantile;

mod multiclass_model;
pub use multiclass_model::{MultiClassIterationRunSummary, MultiClassTrainedModel};

/// Small epsilon added to leaf value denominators to prevent division by zero.
const LEAF_EPSILON: f32 = 1e-6;

/// Type alias for an active node entry in the level-wise tree builder.
/// Fields: (local_node_id, row_indices, histograms, parent_leaf_value, parent_linear_leaf)
type ActiveNodeEntry = (u32, Vec<u32>, HistogramBundle, f32, Option<LinearLeaf>);

/// Type alias for a split linear leaf pair (delta, delta, absolute, absolute).
type LinearLeafQuad = (LinearLeaf, LinearLeaf, LinearLeaf, LinearLeaf);

/// Type alias for a pair of optional linear leaves (delta pair, absolute pair).
type LinearLeafPairSplit = (
    Option<(LinearLeaf, LinearLeaf)>,
    Option<(LinearLeaf, LinearLeaf)>,
);


#[derive(Debug, Clone, PartialEq)]
pub struct FitContractEvaluation {
    pub baseline_prediction: f32,
    pub gradients: Vec<GradientPair>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainRoundSummary {
    pub baseline_prediction: f32,
    pub root_stats: NodeStats,
    pub split_candidate: Option<SplitCandidate>,
    pub partition: Option<PartitionResult>,
}

#[derive(Debug, Clone, Copy)]
pub struct ValidationDatasetRef<'a> {
    pub dataset: &'a TrainingDataset,
    pub binned_matrix: &'a BinnedMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainedStump {
    pub split: SplitCandidate,
    pub left_leaf_value: LeafValue,
    pub right_leaf_value: LeafValue,
    /// Multiplicative weight applied to this stump's leaf contributions at
    /// predict time. `1.0` for every stump trained under
    /// `BoostingMode::Standard` or `BoostingMode::Goss` (this preserves v0.8.0
    /// numerics bit-for-bit). DART rounds emit stumps with `tree_weight`
    /// determined by the dropout-normalization step — see [`crate::dart`].
    pub tree_weight: f32,
    /// v0.10.0+: K-output leaf values for the joint multi-label trainer.
    /// `Some((left_k_values, right_k_values))` where both Vec<f32> have
    /// length `n_outputs`. `None` for scalar / linear-leaf models — in
    /// that case `left_leaf_value` / `right_leaf_value` are authoritative.
    /// When `Some`, `left_leaf_value` / `right_leaf_value` still carry a
    /// `LeafValue::Scalar(_)` placeholder (typically `0.0`) so the
    /// existing scalar code paths remain well-typed.
    pub multi_output_leaf_values: Option<(Vec<f32>, Vec<f32>)>,
}

impl TrainedStump {
    /// Default constructor that sets `tree_weight = 1.0` and
    /// `multi_output_leaf_values = None`. Use this anywhere the boosting mode
    /// is known to be Standard or GOSS, or for tests that don't exercise
    /// DART or joint multi-output semantics.
    pub fn new_unweighted(
        split: SplitCandidate,
        left_leaf_value: LeafValue,
        right_leaf_value: LeafValue,
    ) -> Self {
        Self {
            split,
            left_leaf_value,
            right_leaf_value,
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeDebugStats {
    pub node_id: u32,
    pub feature_index: u32,
    pub threshold_bin: u16,
    pub gain: f32,
    pub default_left: bool,
    pub left_stats: NodeStats,
    pub right_stats: NodeStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainedModel {
    pub baseline_prediction: f32,
    pub feature_count: usize,
    pub stumps: Vec<TrainedStump>,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    pub node_debug_stats: Option<Vec<NodeDebugStats>>,
    /// Objective name recorded in the model artifact metadata.
    pub objective: String,
    /// Feature indices that use native categorical splits (empty if none).
    pub native_categorical_feature_indices: Vec<u32>,
    /// Morph training metadata (None for non-morph artifacts).
    pub morph_metadata: Option<MorphMetadataPayload>,
    /// DRO leaf-solver metadata (None for standard leaf solving).
    pub dro_metadata: Option<DroMetadataPayload>,
    /// Global per-feature training-set means.  `Some(_)` only when the model
    /// uses piecewise-linear leaves and the feature baseline was recorded at
    /// fit time.  Length equals `feature_count`.  Consumed by SHAP for
    /// interventional decomposition of linear-leaf contributions.
    pub feature_baseline: Option<Vec<f32>>,
    /// v0.10.6+: Optional factor-neutralization configuration that was active
    /// during training. `Some(...)` only when the joint trainer's
    /// `effective_neutralization_config` returned a non-inert config. Mirrors
    /// `dro_metadata` — metadata only, prediction never reads it.
    pub neutralization_metadata: Option<NeutralizationMetadataPayload>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalTargetEncodingSpec {
    pub feature_index: usize,
    pub values: Vec<String>,
    pub config: TargetEncoderConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationControls {
    pub rounds: usize,
    pub min_split_gain: f32,
    pub min_rows_per_leaf: usize,
    pub min_abs_leaf_value: f32,
    pub max_abs_leaf_value: f32,
    pub min_loss_improvement: f32,
    pub max_consecutive_weak_improvements: usize,
    pub row_subsample: f32,
    pub col_subsample: f32,
    pub early_stopping_rounds: Option<usize>,
    pub min_validation_improvement: f32,
    /// Maximum number of leaves per tree. None means depth-limited only.
    pub max_leaves: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationStopReason {
    CompletedRequestedRounds,
    DepthBudgetReached,
    NoSplitCandidate,
    GainBelowThreshold,
    LeafRowsBelowThreshold,
    LeafMagnitudeBelowThreshold,
    LossImprovementBelowThreshold,
    MonotoneConstraintViolation,
    MaxLeavesReached,
    ValidationLossPlateau,
    CustomMetricPlateau,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IterationRunSummary {
    pub model: TrainedModel,
    pub rounds_requested: usize,
    pub effective_round_cap: usize,
    pub rounds_completed: usize,
    pub stop_reason: IterationStopReason,
    pub initial_loss: f32,
    pub initial_validation_loss: Option<f32>,
    pub loss_per_completed_round: Vec<f32>,
    pub validation_loss_per_completed_round: Vec<f32>,
    pub sampled_rows_per_completed_round: Vec<usize>,
    pub sampled_features_per_completed_round: Vec<usize>,
    pub best_validation_loss: Option<f32>,
    pub best_validation_round: Option<usize>,
    pub weak_improvement_rounds_committed: usize,
    pub final_loss: f32,
    pub final_validation_loss: Option<f32>,
    /// Per-round custom metric values (empty when no custom metric callback is used).
    pub custom_metric_per_round: Vec<f32>,
    /// Name of the custom metric (None when no custom metric callback is used).
    pub custom_metric_name: Option<String>,
    /// Per-round diagnostic snapshot: gradient stats, hessian magnitude, and —
    /// when factor neutralization runs per round — a "neutralization
    /// effectiveness" score `1 - ||g_proj|| / ||g_orig||` bounded in [0, 1].
    /// Length equals `rounds_completed` after a successful fit.
    pub diagnostics_per_round: Vec<IterationDiagnostics>,
}

/// Per-round training telemetry recorded by the fit loop.
///
/// Capturing this is intentionally cheap: each value is one or two reductions
/// over the gradient/hessian buffer that the trainer already owns.  The data
/// is exposed on every estimator (regressor / classifier / ranker) so callers
/// can inspect gradient trajectories, confirm convergence, and — for
/// neutralized fits — verify how much signal the factor projection removed.
///
/// For multiclass training, the per-round entry is an aggregate across the
/// K class buffers: mean-of-class for gradient/hessian norms and variance,
/// max-of-class for `neutralization_effectiveness` (the worst-projected
/// class is the most informative statistic).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IterationDiagnostics {
    /// L2 norm of the per-row gradient buffer the trees consumed for this
    /// round (post-projection when factor neutralization is active).
    pub gradient_l2_norm: f32,
    /// Sample variance of the per-row gradient values for this round.
    pub gradient_variance: f32,
    /// L2 norm of the per-row hessian buffer for this round.
    pub hessian_l2_norm: f32,
    /// Pre-projection L2 norm.  `Some(_)` only when per-round factor
    /// neutralization (`per_round_gradient` or `split_penalty`) is enabled
    /// — `pre_target` mode residualizes targets once and never projects
    /// gradients, so this stays `None` there.
    pub original_gradient_l2_norm: Option<f32>,
    /// Post-projection L2 norm.  Mirrors `original_gradient_l2_norm`'s
    /// availability rules.
    pub projected_gradient_l2_norm: Option<f32>,
    /// `1 - projected_l2 / original_l2`, clamped to `[0, 1]`.  Higher means
    /// more gradient signal was removed by the factor projection.  `None`
    /// when projection isn't active (constant-leaf, `pre_target`, or no
    /// neutralization configured) or when `original_l2 == 0`.
    pub neutralization_effectiveness: Option<f32>,
    /// Number of training rows sampled for this round (after row_subsample).
    pub n_active_rows: usize,
    /// Number of split-feature tiles available for this round (after
    /// col_subsample).
    pub n_active_features: usize,
}

impl IterationDiagnostics {
    /// Construct a diagnostics record from a gradient buffer pre- and
    /// post-projection.  `original` is `Some(&original_buffer)` only when a
    /// per-round projection was applied; otherwise pass `None` and the
    /// projection-related fields stay `None`.
    pub fn from_gradient_snapshot(
        post_projection_gradients: &[GradientPair],
        original_gradient_norm: Option<f32>,
        n_active_rows: usize,
        n_active_features: usize,
    ) -> Self {
        let (g_norm, g_var, h_norm) = gradient_buffer_stats(post_projection_gradients);
        let projected = original_gradient_norm.map(|_| g_norm);
        let effectiveness = match (original_gradient_norm, projected) {
            (Some(orig), Some(proj)) if orig > 0.0 => {
                let raw = 1.0 - proj / orig;
                Some(raw.clamp(0.0, 1.0))
            }
            _ => None,
        };
        Self {
            gradient_l2_norm: g_norm,
            gradient_variance: g_var,
            hessian_l2_norm: h_norm,
            original_gradient_l2_norm: original_gradient_norm,
            projected_gradient_l2_norm: projected,
            neutralization_effectiveness: effectiveness,
            n_active_rows,
            n_active_features,
        }
    }

    /// Aggregate per-class diagnostics into a single per-round record for
    /// multiclass training.  Norms / variance are mean-of-class; effectiveness
    /// is max-of-class (so users see the worst-projected class).
    pub fn aggregate_per_class(class_entries: &[IterationDiagnostics]) -> Self {
        if class_entries.is_empty() {
            return Self::default();
        }
        let k = class_entries.len() as f32;
        let mean = |f: fn(&IterationDiagnostics) -> f32| -> f32 {
            class_entries.iter().map(f).sum::<f32>() / k
        };
        let max_opt = |f: fn(&IterationDiagnostics) -> Option<f32>| -> Option<f32> {
            class_entries
                .iter()
                .filter_map(f)
                .fold(None, |acc, v| Some(acc.map_or(v, |a: f32| a.max(v))))
        };
        Self {
            gradient_l2_norm: mean(|d| d.gradient_l2_norm),
            gradient_variance: mean(|d| d.gradient_variance),
            hessian_l2_norm: mean(|d| d.hessian_l2_norm),
            original_gradient_l2_norm: max_opt(|d| d.original_gradient_l2_norm),
            projected_gradient_l2_norm: max_opt(|d| d.projected_gradient_l2_norm),
            neutralization_effectiveness: max_opt(|d| d.neutralization_effectiveness),
            n_active_rows: class_entries[0].n_active_rows,
            n_active_features: class_entries[0].n_active_features,
        }
    }
}

/// Compute `(L2_norm_of_grads, variance_of_grads, L2_norm_of_hessians)` over
/// a gradient/hessian buffer in a single pass.  Skips non-finite entries so a
/// stray NaN doesn't poison the telemetry.
fn gradient_buffer_stats(gradients: &[GradientPair]) -> (f32, f32, f32) {
    if gradients.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut g_sq_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut g_finite_count = 0u64;
    let mut h_sq_sum = 0.0f64;
    for gp in gradients {
        if gp.grad.is_finite() {
            let g = gp.grad as f64;
            g_sq_sum += g * g;
            g_sum += g;
            g_finite_count += 1;
        }
        if gp.hess.is_finite() {
            let h = gp.hess as f64;
            h_sq_sum += h * h;
        }
    }
    let g_norm = g_sq_sum.sqrt() as f32;
    let h_norm = h_sq_sum.sqrt() as f32;
    let g_var = if g_finite_count > 1 {
        let mean = g_sum / g_finite_count as f64;
        let variance = (g_sq_sum / g_finite_count as f64) - mean * mean;
        variance.max(0.0) as f32
    } else {
        0.0
    };
    (g_norm, g_var, h_norm)
}

/// L2 norm of the gradient channel of a `GradientPair` buffer.  Used as a
/// "pre-projection snapshot" before `FactorProjector::project_gradient_pairs_in_place`
/// mutates the buffer in-place.
fn gradient_l2_norm_only(gradients: &[GradientPair]) -> f32 {
    let mut sq = 0.0f64;
    for gp in gradients {
        if gp.grad.is_finite() {
            let g = gp.grad as f64;
            sq += g * g;
        }
    }
    (sq.sqrt()) as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactCompatibilityMode {
    Strict,
    AllowLegacyTreesOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPolicyMode {
    Manual,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PolicyFitRequest {
    rounds: usize,
    policy_mode: TrainingPolicyMode,
    store_node_debug_stats: bool,
}

/// Initial model state for warm-starting (continuing training from a previous model).
#[derive(Debug, Clone)]
pub struct WarmStartState {
    /// Baseline prediction (initial bias) from the original model.
    pub baseline_prediction: f32,
    /// Previously trained tree stumps.
    pub stumps: Vec<TrainedStump>,
    /// Number of rounds already completed in the initial model.
    pub initial_rounds_completed: usize,
    /// MorphBoost EMA snapshot from the previous fit (v0.7.3+).  When
    /// `Some` and the current fit also uses `training_mode="morph"`,
    /// the engine seeds `MorphState::ema_stats` from this snapshot so a
    /// resumed `N + M`-round model matches a fresh `N + M`-round fit.
    /// Empty / missing → EMA starts cold (legacy v0.7.1/v0.7.2 behaviour).
    pub initial_ema_stats: Option<Vec<GradientEmaStats>>,
    /// v0.10.0+: When the prior fit used DART, the per-stump `tree_weight`
    /// array (length = `stumps.len()`). `None` for non-DART warm-starts.
    /// On a DART warm-start continuation, the engine seeds
    /// `dart_state.tree_weights` from this snapshot so prior-tree dropouts
    /// during new rounds use the correct accumulated weights. Historical
    /// `dropped_per_round` arrays do *not* round-trip — new rounds start
    /// fresh dropout bookkeeping going forward.
    pub initial_dart_tree_weights: Option<Vec<f32>>,
}

/// State needed to continue multiclass training from a prior model.
pub struct MultiClassWarmStartState {
    pub baseline_predictions: Vec<f32>,
    pub class_stumps: Vec<Vec<TrainedStump>>,
    pub initial_rounds_completed: usize,
    /// MorphBoost EMA snapshot from the previous fit (v0.7.3+).  See
    /// `WarmStartState::initial_ema_stats`.
    pub initial_ema_stats: Option<Vec<GradientEmaStats>>,
    /// v0.10.1+: per-tree weights for multiclass DART warm-start.  Flat
    /// layout `[round 0 class 0, round 0 class 1, ..., round 0 class
    /// K-1, round 1 class 0, ...]` — round-major × class-k.  Length
    /// must equal `initial_rounds_completed * K`.  `None` means the
    /// prior fit was not multiclass DART; the engine falls back to a
    /// fresh DART state in that case.
    pub initial_dart_tree_weights: Option<Vec<f32>>,
}

struct IterationExecutionContext<'a> {
    controls: IterationControls,
    validation: Option<ValidationDatasetRef<'a>>,
    policy_mode: Option<TrainingPolicyMode>,
    warm_start: Option<WarmStartState>,
    custom_metric_callback: Option<&'a dyn PerRoundMetricCallback>,
    /// Features that use native categorical splits (empty = all continuous).
    categorical_features: Vec<CategoricalFeatureInfo>,
    pre_target_already_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactCompatibilityReport {
    pub trees_section_count: usize,
    pub predictor_layout_section_count: usize,
    pub strict_compatible: bool,
    pub legacy_trees_only_compatible: bool,
    pub legacy_compatible: bool,
    pub recommended_mode: Option<ArtifactCompatibilityMode>,
}

impl ArtifactCompatibilityReport {
    fn required_section_report(self) -> alloygbm_core::RequiredSectionCompatibilityReport {
        alloygbm_core::RequiredSectionCompatibilityReport {
            trees_section_count: self.trees_section_count,
            predictor_layout_section_count: self.predictor_layout_section_count,
            strict_compatible: self.strict_compatible,
            legacy_trees_only_compatible: self.legacy_trees_only_compatible,
            legacy_compatible: self.legacy_compatible,
        }
    }
}

impl IterationControls {
    pub fn new(
        rounds: usize,
        min_split_gain: f32,
        min_rows_per_leaf: usize,
        min_abs_leaf_value: f32,
        max_abs_leaf_value: f32,
        min_loss_improvement: f32,
        max_consecutive_weak_improvements: usize,
    ) -> EngineResult<Self> {
        if rounds == 0 {
            return Err(EngineError::InvalidConfig(
                "rounds must be greater than 0".to_string(),
            ));
        }
        if !min_split_gain.is_finite() || min_split_gain < 0.0 {
            return Err(EngineError::InvalidConfig(
                "min_split_gain must be finite and >= 0".to_string(),
            ));
        }
        if min_rows_per_leaf == 0 {
            return Err(EngineError::InvalidConfig(
                "min_rows_per_leaf must be greater than 0".to_string(),
            ));
        }
        if !min_abs_leaf_value.is_finite() || min_abs_leaf_value < 0.0 {
            return Err(EngineError::InvalidConfig(
                "min_abs_leaf_value must be finite and >= 0".to_string(),
            ));
        }
        if !max_abs_leaf_value.is_finite() || max_abs_leaf_value <= 0.0 {
            return Err(EngineError::InvalidConfig(
                "max_abs_leaf_value must be finite and > 0".to_string(),
            ));
        }
        if min_abs_leaf_value > max_abs_leaf_value {
            return Err(EngineError::InvalidConfig(
                "min_abs_leaf_value cannot exceed max_abs_leaf_value".to_string(),
            ));
        }
        if !min_loss_improvement.is_finite() || min_loss_improvement < 0.0 {
            return Err(EngineError::InvalidConfig(
                "min_loss_improvement must be finite and >= 0".to_string(),
            ));
        }

        Ok(Self {
            rounds,
            min_split_gain,
            min_rows_per_leaf,
            min_abs_leaf_value,
            max_abs_leaf_value,
            min_loss_improvement,
            max_consecutive_weak_improvements,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            max_leaves: None,
        })
    }

    pub fn with_max_leaves(mut self, max_leaves: Option<usize>) -> EngineResult<Self> {
        if let Some(n) = max_leaves
            && n < 2
        {
            return Err(EngineError::InvalidConfig(
                "max_leaves must be >= 2 when set".to_string(),
            ));
        }
        self.max_leaves = max_leaves;
        Ok(self)
    }

    pub fn with_subsample_rates(
        mut self,
        row_subsample: f32,
        col_subsample: f32,
    ) -> EngineResult<Self> {
        if !(0.0..=1.0).contains(&row_subsample) || row_subsample == 0.0 {
            return Err(EngineError::InvalidConfig(
                "row_subsample must be in (0.0, 1.0]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&col_subsample) || col_subsample == 0.0 {
            return Err(EngineError::InvalidConfig(
                "col_subsample must be in (0.0, 1.0]".to_string(),
            ));
        }
        self.row_subsample = row_subsample;
        self.col_subsample = col_subsample;
        Ok(self)
    }

    pub fn with_validation_early_stopping(
        mut self,
        early_stopping_rounds: usize,
        min_validation_improvement: f32,
    ) -> EngineResult<Self> {
        if early_stopping_rounds == 0 {
            return Err(EngineError::InvalidConfig(
                "early_stopping_rounds must be greater than 0".to_string(),
            ));
        }
        if !min_validation_improvement.is_finite() || min_validation_improvement < 0.0 {
            return Err(EngineError::InvalidConfig(
                "min_validation_improvement must be finite and >= 0".to_string(),
            ));
        }
        self.early_stopping_rounds = Some(early_stopping_rounds);
        self.min_validation_improvement = min_validation_improvement;
        Ok(self)
    }
}

impl TrainedModel {
    /// Count the number of distinct tree rounds in this model.
    pub fn rounds_completed(&self) -> usize {
        if self.stumps.is_empty() {
            return 0;
        }
        let max_tree_id = self
            .stumps
            .iter()
            .map(|s| decode_tree_node_id(s.split.node_id).0 as usize)
            .max()
            .unwrap_or(0);
        max_tree_id + 1
    }

    pub fn with_categorical_state(
        mut self,
        categorical_state: Option<CategoricalStatePayloadV1>,
    ) -> EngineResult<Self> {
        if let Some(state) = categorical_state.as_ref() {
            validate_categorical_state_payload_v1(state, Some(self.feature_count))?;
        }
        self.categorical_state = categorical_state;
        Ok(self)
    }

    pub fn with_node_debug_stats(
        mut self,
        node_debug_stats: Option<Vec<NodeDebugStats>>,
    ) -> EngineResult<Self> {
        if let Some(stats) = node_debug_stats.as_ref() {
            for stat in stats {
                if stat.feature_index as usize >= self.feature_count {
                    return Err(EngineError::ContractViolation(format!(
                        "node debug stats feature_index {} exceeds feature_count {}",
                        stat.feature_index, self.feature_count
                    )));
                }
            }
        }
        self.node_debug_stats = node_debug_stats;
        Ok(self)
    }

    pub fn with_node_debug_stats_from_stumps(self) -> EngineResult<Self> {
        let stats = self
            .stumps
            .iter()
            .map(|stump| NodeDebugStats {
                node_id: stump.split.node_id,
                feature_index: stump.split.feature_index,
                threshold_bin: stump.split.threshold_bin,
                gain: stump.split.gain,
                default_left: stump.split.default_left,
                left_stats: stump.split.left_stats.clone(),
                right_stats: stump.split.right_stats.clone(),
            })
            .collect();
        self.with_node_debug_stats(Some(stats))
    }

    pub fn predict_row(&self, features: &[f32]) -> EngineResult<f32> {
        if features.len() != self.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                self.feature_count
            )));
        }

        let stumps_by_node = self
            .stumps
            .iter()
            .map(|stump| (stump.split.node_id, stump))
            .collect::<HashMap<_, _>>();
        let mut prediction = self.baseline_prediction;
        for stump in &self.stumps {
            if !row_satisfies_stump_path_features(features, stump, &stumps_by_node)? {
                continue;
            }
            let feature_index = stump.split.feature_index as usize;
            let feature_value = features[feature_index];
            let leaf = if split_went_left(&stump.split, feature_value) {
                stump.left_leaf_value.eval_row(features)
            } else {
                stump.right_leaf_value.eval_row(features)
            };
            // v0.9.0: DART artifacts carry a per-stump `tree_weight` that
            // scales the leaf contribution at predict time. Non-DART
            // models have `tree_weight = 1.0` and this multiplication is
            // a no-op (bit-identical to v0.8.0).
            prediction += stump.tree_weight * leaf;
        }

        Ok(prediction)
    }

    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> EngineResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(EngineError::ContractViolation(
                "rows cannot be empty".to_string(),
            ));
        }
        rows.iter().map(|row| self.predict_row(row)).collect()
    }

    pub fn to_artifact_bytes(&self) -> EngineResult<Vec<u8>> {
        let trees_payload = encode_trained_model_payload(self)?;
        let predictor_layout_payload = encode_predictor_layout_payload(self)?;
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..self.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: self.objective.clone(),
            num_classes: None,
        };

        let mut sections = vec![
            (ModelSectionKind::Trees, trees_payload),
            (ModelSectionKind::PredictorLayout, predictor_layout_payload),
        ];
        if let Some(categorical_state) = self.categorical_state.as_ref() {
            let categorical_payload = encode_categorical_state_payload_v1(categorical_state)?;
            sections.push((ModelSectionKind::CategoricalState, categorical_payload));
        }
        if let Some(node_debug_stats) = self.node_debug_stats.as_ref() {
            let node_stats_payload = encode_node_debug_stats_payload(node_debug_stats)?;
            sections.push((ModelSectionKind::NodeDebugStats, node_stats_payload));
        }
        // Serialize native categorical splits if any stumps are categorical.
        if self.stumps.iter().any(|s| s.split.is_categorical) {
            let stump_bitsets: Vec<(u32, Vec<u8>)> = self
                .stumps
                .iter()
                .enumerate()
                .filter(|(_, s)| s.split.is_categorical)
                .map(|(i, s)| {
                    (
                        i as u32,
                        s.split.categorical_bitset.clone().unwrap_or_default(),
                    )
                })
                .collect();
            let payload = NativeCategoricalSplitsPayload {
                native_categorical_feature_indices: self.native_categorical_feature_indices.clone(),
                stump_bitsets,
            };
            let cat_bytes = encode_native_categorical_splits_payload(&payload)?;
            sections.push((ModelSectionKind::NativeCategoricalSplits, cat_bytes));
        }
        // Morph metadata section (optional — only for morph-trained artifacts)
        if let Some(morph) = self.morph_metadata.as_ref() {
            sections.push((
                ModelSectionKind::MorphMetadata,
                encode_morph_metadata_payload(morph),
            ));
        }
        // DRO metadata section (optional — only for DRO leaf-solver artifacts)
        if let Some(dro) = self.dro_metadata.as_ref() {
            sections.push((
                ModelSectionKind::DroMetadata,
                encode_dro_metadata_payload(dro),
            ));
        }
        // Neutralization metadata section (optional — only for joint artifacts with
        // factor neutralization active at training time).
        if let Some(neut) = self.neutralization_metadata.as_ref() {
            sections.push((
                ModelSectionKind::NeutralizationMetadata,
                encode_neutralization_metadata_payload(neut),
            ));
        }
        // Linear leaf coefficients section (optional — only for pl-tree artifacts)
        {
            let linear_entries: Vec<LinearLeafEntry> = self
                .stumps
                .iter()
                .enumerate()
                .filter_map(|(idx, stump)| {
                    let left = match &stump.left_leaf_value {
                        LeafValue::Linear(ll) => Some(ll.clone()),
                        _ => None,
                    };
                    let right = match &stump.right_leaf_value {
                        LeafValue::Linear(rl) => Some(rl.clone()),
                        _ => None,
                    };
                    if left.is_some() || right.is_some() {
                        Some(LinearLeafEntry {
                            stump_idx: idx as u32,
                            left_leaf: left,
                            right_leaf: right,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            if !linear_entries.is_empty() {
                sections.push((
                    ModelSectionKind::LinearLeafCoefficients,
                    encode_linear_leaf_coefficients_payload(&LinearLeafCoefficientsPayload {
                        entries: linear_entries,
                    }),
                ));
            }
        }
        // FeatureBaseline section (optional — written only when linear leaves
        // are present and the baseline was captured at fit time).  Provides
        // global per-feature means so SHAP can decompose linear leaves
        // interventionally without needing the original training data.
        if let Some(baseline) = self.feature_baseline.as_ref()
            && baseline.len() == self.feature_count
            && self.stumps.iter().any(|s| {
                matches!(s.left_leaf_value, LeafValue::Linear(_))
                    || matches!(s.right_leaf_value, LeafValue::Linear(_))
            })
        {
            sections.push((
                ModelSectionKind::FeatureBaseline,
                encode_feature_baseline_payload(&FeatureBaselinePayload {
                    feature_means: baseline.clone(),
                }),
            ));
        }

        // DART per-stump tree weights (optional). Emitted only when at least
        // one stump has a non-default weight, which keeps Standard/GOSS
        // artifacts byte-identical to v0.8.0.
        if self
            .stumps
            .iter()
            .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON)
        {
            sections.push((
                ModelSectionKind::DartTreeWeights,
                encode_dart_tree_weights_payload(&DartTreeWeightsPayload {
                    weights: self.stumps.iter().map(|s| s.tree_weight).collect(),
                }),
            ));
        }

        // Multi-output leaf values section (v0.10.0+). Emitted only when at
        // least one stump carries K-output leaves (joint multi-label
        // trainer). One Vec<f32> per stump: [left_K_values..., right_K_values...].
        if self
            .stumps
            .iter()
            .any(|s| s.multi_output_leaf_values.is_some())
        {
            let n_outputs = self
                .stumps
                .iter()
                .find_map(|s| s.multi_output_leaf_values.as_ref().map(|v| v.0.len()))
                .unwrap_or(0) as u32;
            let per_stump_leaf_values: Vec<Vec<f32>> = self
                .stumps
                .iter()
                .map(|s| match s.multi_output_leaf_values.as_ref() {
                    Some((left, right)) => {
                        let mut packed = Vec::with_capacity(left.len() + right.len());
                        packed.extend_from_slice(left);
                        packed.extend_from_slice(right);
                        packed
                    }
                    None => Vec::new(),
                })
                .collect();
            sections.push((
                ModelSectionKind::MultiOutputLeafValues,
                alloygbm_core::encode_multi_output_leaf_values_payload(
                    &alloygbm_core::MultiOutputLeafValuesPayload {
                        n_outputs,
                        per_stump_leaf_values,
                    },
                ),
            ));
        }

        serialize_model_artifact_v1(&metadata, &sections).map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        Self::from_artifact_bytes_with_mode(bytes, ArtifactCompatibilityMode::AllowLegacyTreesOnly)
    }

    pub fn artifact_compatibility_report(
        bytes: &[u8],
    ) -> EngineResult<ArtifactCompatibilityReport> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;
        Ok(artifact_compatibility_report_from_sections(
            &parsed.sections,
        ))
    }

    pub fn from_artifact_bytes_auto(
        bytes: &[u8],
    ) -> EngineResult<(Self, ArtifactCompatibilityMode)> {
        let report = Self::artifact_compatibility_report(bytes)?;
        let mode = report.recommended_mode.ok_or_else(|| {
            EngineError::ContractViolation(format_required_section_auto_mode_error(
                report.required_section_report(),
            ))
        })?;
        let model = Self::from_artifact_bytes_with_mode(bytes, mode)?;
        Ok((model, mode))
    }

    pub fn from_artifact_bytes_with_mode(
        bytes: &[u8],
        compatibility_mode: ArtifactCompatibilityMode,
    ) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;
        let compatibility_report = artifact_compatibility_report_from_sections(&parsed.sections);

        match compatibility_mode {
            ArtifactCompatibilityMode::Strict if !compatibility_report.strict_compatible => {
                return Err(EngineError::ContractViolation(
                    format_required_section_mode_error(
                        compatibility_report.required_section_report(),
                        false,
                    ),
                ));
            }
            ArtifactCompatibilityMode::AllowLegacyTreesOnly
                if !compatibility_report.legacy_compatible =>
            {
                return Err(EngineError::ContractViolation(
                    format_required_section_mode_error(
                        compatibility_report.required_section_report(),
                        true,
                    ),
                ));
            }
            _ => {}
        }

        let trees_section = required_single_section(&parsed.sections, ModelSectionKind::Trees)?;
        let metadata_feature_count = parsed.contract.metadata.feature_names.len();
        let predictor_layout =
            resolve_predictor_layout(&parsed.sections, metadata_feature_count, compatibility_mode)?;

        let mut model = decode_trained_model_payload(&trees_section.payload)?;

        if predictor_layout.feature_count != metadata_feature_count {
            return Err(EngineError::ContractViolation(format!(
                "predictor layout feature_count {} does not match metadata feature count {}",
                predictor_layout.feature_count, metadata_feature_count
            )));
        }
        if model.feature_count != predictor_layout.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "decoded trees feature_count {} does not match predictor layout feature_count {}",
                model.feature_count, predictor_layout.feature_count
            )));
        }
        if model.feature_count != metadata_feature_count {
            return Err(EngineError::ContractViolation(format!(
                "decoded trees feature_count {} does not match metadata feature count {}",
                model.feature_count, metadata_feature_count
            )));
        }

        model.categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, metadata_feature_count)?;
        model.node_debug_stats = decode_optional_node_debug_stats_section(&parsed.sections)?;

        // Decode optional native categorical splits section and populate stump bitsets.
        if let Some(cat_payload) =
            decode_optional_native_categorical_splits_section(&parsed.sections)?
        {
            model.native_categorical_feature_indices =
                cat_payload.native_categorical_feature_indices;
            for (stump_index, bitset) in cat_payload.stump_bitsets {
                let idx = stump_index as usize;
                if idx < model.stumps.len() {
                    model.stumps[idx].split.categorical_bitset = Some(bitset);
                }
            }
        }

        // Decode optional morph metadata section.
        model.morph_metadata = decode_optional_morph_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;
        model.dro_metadata = decode_optional_dro_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;
        model.neutralization_metadata =
            decode_optional_neutralization_metadata_artifact_section(&parsed.sections)
                .map_err(EngineError::from)?;

        // Decode optional linear leaf coefficients section and backfill LeafValue::Linear on stumps.
        if let Some(ll_payload) = decode_optional_linear_leaf_coefficients_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            for entry in ll_payload.entries {
                let idx = entry.stump_idx as usize;
                if idx < model.stumps.len() {
                    if let Some(ll) = entry.left_leaf {
                        model.stumps[idx].left_leaf_value = LeafValue::Linear(ll);
                    }
                    if let Some(rl) = entry.right_leaf {
                        model.stumps[idx].right_leaf_value = LeafValue::Linear(rl);
                    }
                }
            }
        }

        // Decode optional FeatureBaseline section.  Only retain when the
        // length matches feature_count to defend against artifact corruption
        // or schema drift; mismatches silently fall back to `None`, which
        // SHAP treats as "no linear-leaf support recorded for this artifact".
        model.feature_baseline = decode_optional_feature_baseline_section(&parsed.sections)
            .map_err(EngineError::from)?
            .map(|payload| payload.feature_means)
            .filter(|means| means.len() == metadata_feature_count);

        // Decode optional DartTreeWeights section and apply per-stump weights.
        // Pre-v0.9.0 artifacts have no section; stumps keep their default 1.0.
        if let Some(dart_payload) = decode_optional_dart_tree_weights_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            if dart_payload.weights.len() != model.stumps.len() {
                return Err(EngineError::ContractViolation(format!(
                    "DartTreeWeights length {} != stump count {}",
                    dart_payload.weights.len(),
                    model.stumps.len()
                )));
            }
            for (stump, w) in model.stumps.iter_mut().zip(dart_payload.weights.iter()) {
                stump.tree_weight = *w;
            }
        }

        // Decode optional MultiOutputLeafValues section (v0.10.0+) and attach
        // K-output leaf values to stumps. Pre-v0.10.0 artifacts have no section.
        if let Some(mo_payload) =
            alloygbm_core::decode_optional_multi_output_leaf_values_section(&parsed.sections)
                .map_err(EngineError::from)?
        {
            if mo_payload.per_stump_leaf_values.len() != model.stumps.len() {
                return Err(EngineError::ContractViolation(format!(
                    "MultiOutputLeafValues length {} != stump count {}",
                    mo_payload.per_stump_leaf_values.len(),
                    model.stumps.len()
                )));
            }
            let k = mo_payload.n_outputs as usize;
            for (stump, packed) in model
                .stumps
                .iter_mut()
                .zip(mo_payload.per_stump_leaf_values.into_iter())
            {
                if packed.is_empty() {
                    continue;
                }
                if packed.len() != 2 * k {
                    return Err(EngineError::ContractViolation(format!(
                        "MultiOutputLeafValues stump entry has {} values, expected 2 × n_outputs = {}",
                        packed.len(),
                        2 * k
                    )));
                }
                let (left, right) = packed.split_at(k);
                stump.multi_output_leaf_values = Some((left.to_vec(), right.to_vec()));
            }
        }

        model.feature_count = metadata_feature_count;
        model.objective = parsed.contract.metadata.objective.clone();
        Ok(model)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trainer {
    params: TrainParams,
    categorical_features: Vec<CategoricalFeatureInfo>,
}

impl Trainer {
    pub fn new(params: TrainParams) -> EngineResult<Self> {
        validate_train_params(&params)?;
        Ok(Self {
            params,
            categorical_features: Vec::new(),
        })
    }

    /// Set the categorical feature metadata for native categorical splits.
    pub fn with_categorical_features(mut self, features: Vec<CategoricalFeatureInfo>) -> Self {
        self.categorical_features = features;
        self
    }

    pub fn params(&self) -> &TrainParams {
        &self.params
    }

    pub fn validate_fit_contract<O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        objective: &O,
    ) -> EngineResult<FitContractEvaluation> {
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;

        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);

        self.evaluate_fit_contract_on_active_dataset(active_dataset, objective)
    }

    fn evaluate_fit_contract_on_active_dataset<O: ObjectiveOps>(
        &self,
        active_dataset: &TrainingDataset,
        objective: &O,
    ) -> EngineResult<FitContractEvaluation> {
        let baseline_prediction = objective.initial_prediction(
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        if !baseline_prediction.is_finite() {
            return Err(EngineError::ContractViolation(
                "objective returned non-finite initial prediction".to_string(),
            ));
        }

        let predictions = vec![baseline_prediction; active_dataset.row_count()];
        let mut gradients = objective.compute_gradients(
            &predictions,
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        if let Some(config) = gradient_neutralization_config(&self.params) {
            let exposures = active_dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            FactorProjector::new(
                exposures,
                active_dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?
            .project_gradient_pairs_in_place(&mut gradients)?;
        }
        validate_gradient_pairs(&gradients, active_dataset.row_count())?;

        Ok(FitContractEvaluation {
            baseline_prediction,
            gradients,
        })
    }

    pub fn fit_one_round<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
    ) -> EngineResult<TrainRoundSummary> {
        validate_training_alignment(dataset, binned_matrix)?;

        let fit_contract = self.validate_fit_contract(dataset, objective)?;
        let root_row_indices = (0..dataset.row_count() as u32).collect::<Vec<_>>();
        let root_node = NodeSlice::new(0, root_row_indices)?;
        let feature_tiles = vec![FeatureTile::new(0, binned_matrix.feature_count as u32)?];
        let split_options = split_selection_options_from_env()?;

        let histograms = backend.build_histograms(
            binned_matrix,
            &fit_contract.gradients,
            &root_node,
            &feature_tiles,
        )?;
        let factor_context = factor_split_context_for_node(
            &self.params,
            binned_matrix,
            dataset.factor_exposures.as_ref(),
            &root_node.row_indices,
        );
        let split_candidate = backend.best_split_with_factor_context(
            &histograms,
            split_options,
            &self.params.feature_weights,
            &[],
            factor_context.as_ref(),
        )?;
        let root_stats = backend.reduce_sums(&fit_contract.gradients, &root_node.row_indices)?;

        let partition = if let Some(split) = &split_candidate {
            let partition = backend.apply_split(binned_matrix, &root_node, split)?;
            validate_partition_cover(dataset.row_count(), &partition)?;
            Some(partition)
        } else {
            None
        };

        Ok(TrainRoundSummary {
            baseline_prediction: fit_contract.baseline_prediction,
            root_stats,
            split_candidate,
            partition,
        })
    }

    pub fn fit_iterations<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        rounds: usize,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_policy(
            dataset,
            binned_matrix,
            backend,
            objective,
            rounds,
            TrainingPolicyMode::Manual,
            false,
        )
    }

    pub fn fit_iterations_with_single_target_encoded_feature<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        rounds: usize,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_single_target_encoded_feature_and_policy(
            dataset,
            binned_matrix,
            spec,
            backend,
            objective,
            rounds,
            TrainingPolicyMode::Manual,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_policy<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_policy_request(
            dataset,
            binned_matrix,
            backend,
            objective,
            PolicyFitRequest {
                rounds,
                policy_mode,
                store_node_debug_stats,
            },
        )
    }

    fn fit_iterations_with_policy_request<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        validate_training_alignment(dataset, binned_matrix)?;
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        self.fit_iterations_with_policy_request_active(
            active_dataset,
            binned_matrix,
            backend,
            objective,
            request,
        )
    }

    fn fit_iterations_with_policy_request_active<B: BackendOps, O: ObjectiveOps>(
        &self,
        active_dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        let controls = self.iteration_controls_for_policy_ext(
            active_dataset,
            binned_matrix,
            request.rounds,
            request.policy_mode,
            objective.requires_group_id(),
        )?;
        let summary = self.fit_iterations_with_optional_validation_summary(
            active_dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: Some(request.policy_mode),
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: true,
            },
        )?;
        let model = summary.model;
        if request.store_node_debug_stats {
            model.with_node_debug_stats_from_stumps()
        } else {
            Ok(model)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_single_target_encoded_feature_and_policy<
        B: BackendOps,
        O: ObjectiveOps,
    >(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> EngineResult<TrainedModel> {
        self.fit_iterations_with_single_target_encoded_feature_and_policy_request(
            dataset,
            binned_matrix,
            spec,
            backend,
            objective,
            PolicyFitRequest {
                rounds,
                policy_mode,
                store_node_debug_stats,
            },
        )
    }

    fn fit_iterations_with_single_target_encoded_feature_and_policy_request<
        B: BackendOps,
        O: ObjectiveOps,
    >(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        spec: &CategoricalTargetEncodingSpec,
        backend: &B,
        objective: &O,
        request: PolicyFitRequest,
    ) -> EngineResult<TrainedModel> {
        validate_training_alignment(dataset, binned_matrix)?;
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        let owned_dataset = prepare_pre_target_training_dataset(&self.params, dataset)?;
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        let (encoded_dataset, encoded_binned_matrix) =
            apply_single_categorical_target_encoding(active_dataset, binned_matrix, spec)?;
        let categorical_state = CategoricalStatePayloadV1 {
            format_version: alloygbm_core::CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: spec.config.time_aware,
            categorical_feature_indices: vec![spec.feature_index as u32],
        };
        let model = self.fit_iterations_with_policy_request_active(
            &encoded_dataset,
            &encoded_binned_matrix,
            backend,
            objective,
            request,
        )?;
        model.with_categorical_state(Some(categorical_state))
    }

    pub fn iteration_controls_for_policy(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
    ) -> EngineResult<IterationControls> {
        self.iteration_controls_for_policy_ext(dataset, binned_matrix, rounds, policy_mode, false)
    }

    /// Extended variant that accepts an `is_ranking` flag so auto-policy
    /// can skip regularization guards that are too aggressive for ranking
    /// objectives (pairwise/LambdaMART/XeNDCG/YetiRank/QueryRMSE).
    pub fn iteration_controls_for_policy_ext(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        policy_mode: TrainingPolicyMode,
        is_ranking: bool,
    ) -> EngineResult<IterationControls> {
        if experiment_force_manual_policy_enabled() {
            return self.default_iteration_controls(rounds);
        }
        match policy_mode {
            TrainingPolicyMode::Manual => self.default_iteration_controls(rounds),
            TrainingPolicyMode::Auto => {
                self.auto_iteration_controls(dataset, binned_matrix, rounds, is_ranking)
            }
        }
    }

    pub fn fit_iterations_with_controls<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<TrainedModel> {
        let summary =
            self.fit_iterations_with_summary(dataset, binned_matrix, backend, objective, controls)?;
        Ok(summary.model)
    }

    pub fn fit_iterations_with_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    pub fn fit_iterations_with_validation_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Continue training from a previously fitted model (warm-start).
    pub fn fit_iterations_warm_start<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: None,
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Continue training from a previously fitted model with validation (warm-start).
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_warm_start_with_validation<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: None,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    // -- Methods that accept a custom metric callback -------------------------

    /// Fit with validation and an optional custom metric callback for early stopping.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_with_validation_and_metric<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        custom_metric: Option<&dyn PerRoundMetricCallback>,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: None,
                custom_metric_callback: custom_metric,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    /// Fit with warm start, validation, and an optional custom metric callback.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_iterations_warm_start_with_validation_and_metric<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &O,
        controls: IterationControls,
        warm_start: WarmStartState,
        custom_metric: Option<&dyn PerRoundMetricCallback>,
    ) -> EngineResult<IterationRunSummary> {
        self.fit_iterations_with_optional_validation_summary(
            dataset,
            binned_matrix,
            backend,
            objective,
            IterationExecutionContext {
                controls,
                validation: Some(validation),
                policy_mode: None,
                warm_start: Some(warm_start),
                custom_metric_callback: custom_metric,
                categorical_features: self.categorical_features.clone(),
                pre_target_already_applied: false,
            },
        )
    }

    // -- Multi-class training -------------------------------------------------

    pub fn fit_multiclass_iterations_with_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            None,
            backend,
            objective,
            controls,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_with_validation_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            Some(validation),
            backend,
            objective,
            controls,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_warm_start_with_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: MultiClassWarmStartState,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            None,
            backend,
            objective,
            controls,
            Some(warm_start),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fit_multiclass_iterations_warm_start_with_validation_summary<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: ValidationDatasetRef<'_>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: MultiClassWarmStartState,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        self.fit_multiclass_iterations_impl(
            dataset,
            binned_matrix,
            Some(validation),
            backend,
            objective,
            controls,
            Some(warm_start),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass_iterations_impl<B: BackendOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        validation: Option<ValidationDatasetRef<'_>>,
        backend: &B,
        objective: &MultiClassSoftmaxObjective,
        controls: IterationControls,
        warm_start: Option<MultiClassWarmStartState>,
    ) -> EngineResult<MultiClassIterationRunSummary> {
        let k = objective.num_classes;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
        // v0.10.1: GOSS supported via
        // `select_row_indices_for_round_multiclass` (per-row score
        // `s_i = sum_k |g_{i,k}|`, LightGBM convention). DART supported
        // via per-round dropout/normalize across a flat
        // `round_index * K + class_k` tree pool — see the DART blocks
        // inside the round loop below.
        let dart_params = match self.params.boosting_mode {
            BoostingMode::Dart {
                drop_rate,
                max_drop,
                normalize_type,
                sample_type,
            } => Some((drop_rate, max_drop, normalize_type, sample_type)),
            _ => None,
        };
        // v0.10.2: leaf-wise multiclass DART is now supported. The per-class
        // `dart_round_start_offsets[k]` / `dart_round_counts[k]` bookkeeping
        // is growth-mode-agnostic because it snapshots `class_stumps[k].len()`
        // around each `build_tree_*` call — under leaf-wise growth each tree
        // has a variable stump count (capped by max_leaves), but the round
        // boundaries are still captured correctly.
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract_for_support(&self.params, dataset, false)?;
        validate_warm_start_neutralization_contract(&self.params, warm_start.is_some(), dataset)?;
        validate_training_alignment(dataset, binned_matrix)?;
        if let Some(validation_ref) = validation {
            validate_training_alignment(validation_ref.dataset, validation_ref.binned_matrix)?;
            if validation_ref.dataset.matrix.feature_count != dataset.matrix.feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "validation feature_count {} does not match training feature_count {}",
                    validation_ref.dataset.matrix.feature_count, dataset.matrix.feature_count
                )));
            }
        }

        // Validate targets are valid class indices
        for (i, &t) in dataset.targets.iter().enumerate() {
            let class = t as usize;
            if class >= k || t < 0.0 || t != t.floor() {
                return Err(EngineError::ContractViolation(format!(
                    "target at index {i} is {t}, expected integer in [0, {k})"
                )));
            }
        }

        let sampling_seed_base = sampling_seed_base(self.params.seed, self.params.deterministic);
        let split_options =
            split_selection_options_for_training(&self.params, None, dataset, binned_matrix)?;
        let feature_count = binned_matrix.feature_count;
        let gradient_projector = if let Some(config) = gradient_neutralization_config(&self.params)
        {
            let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            Some(FactorProjector::new(
                exposures,
                dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?)
        } else {
            None
        };

        // Initialize K prediction arrays — from warm-start or fresh.
        // `warm_ema_stats` captures the optional MorphBoost EMA
        // snapshot for the v0.7.3 warm-start-equivalence fix; consumed
        // below after `MorphState::new` constructs the fresh EMA.
        // `warm_dart_tree_weights` (v0.10.1+) carries the flat
        // round-major × class-k per-tree weights from the prior DART
        // fit; consumed below where `dart_state` is seeded.
        let (
            baselines,
            mut class_stumps,
            round_index_offset,
            initial_stump_counts,
            warm_ema_stats,
            warm_dart_tree_weights,
        ) = if let Some(ws) = warm_start {
            if ws.baseline_predictions.len() != k {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start baseline count {} != num_classes {k}",
                    ws.baseline_predictions.len()
                )));
            }
            if ws.class_stumps.len() != k {
                return Err(EngineError::ContractViolation(format!(
                    "warm-start class_stumps count {} != num_classes {k}",
                    ws.class_stumps.len()
                )));
            }
            let offset = ws.initial_rounds_completed;
            let initial_counts: Vec<usize> = ws.class_stumps.iter().map(|s| s.len()).collect();
            (
                ws.baseline_predictions,
                ws.class_stumps,
                offset,
                initial_counts,
                ws.initial_ema_stats,
                ws.initial_dart_tree_weights,
            )
        } else {
            let baselines = objective.initial_predictions();
            let class_stumps: Vec<Vec<TrainedStump>> = vec![Vec::new(); k];
            (
                baselines,
                class_stumps,
                0_usize,
                vec![0_usize; k],
                None,
                None,
            )
        };

        let n = dataset.row_count();
        let mut class_predictions: Vec<Vec<f32>> = baselines.iter().map(|&b| vec![b; n]).collect();

        // If warm-starting, apply prior-model stumps to prediction arrays
        if round_index_offset > 0 {
            for class_k in 0..k {
                if !class_stumps[class_k].is_empty() {
                    apply_round_stumps_tree_walk(
                        &mut class_predictions[class_k],
                        binned_matrix,
                        &class_stumps[class_k],
                        Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                    )?;
                }
            }
        }
        let mut class_candidate_predictions: Vec<Vec<f32>> = class_predictions.clone();
        // Track stump counts per class at each round boundary for truncation
        let mut stumps_per_round_per_class: Vec<Vec<usize>> = Vec::new();

        // Validation predictions
        let mut validation_class_predictions: Option<Vec<Vec<f32>>> = validation.map(|v| {
            baselines
                .iter()
                .map(|&b| vec![b; v.dataset.row_count()])
                .collect()
        });
        // Apply warm-start stumps to validation predictions too
        if round_index_offset > 0
            && let Some(validation_ref) = validation
            && let Some(val_preds) = validation_class_predictions.as_mut()
        {
            let val_raw = Some((
                &validation_ref.dataset.matrix.values as &[f32],
                validation_ref.dataset.matrix.feature_count,
            ));
            for class_k in 0..k {
                if !class_stumps[class_k].is_empty() {
                    apply_round_stumps_tree_walk(
                        &mut val_preds[class_k],
                        validation_ref.binned_matrix,
                        &class_stumps[class_k],
                        val_raw,
                    )?;
                }
            }
        }

        let initial_loss = objective.loss(
            &class_predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
        )?;
        let initial_validation_loss = if let Some(v) = validation {
            let val_preds: Vec<Vec<f32>> = baselines
                .iter()
                .map(|&b| vec![b; v.dataset.row_count()])
                .collect();
            Some(objective.loss(
                &val_preds,
                &v.dataset.targets,
                v.dataset.sample_weights.as_deref(),
            )?)
        } else {
            None
        };

        let mut current_loss = initial_loss;
        let mut rounds_completed = 0_usize;
        let mut stop_reason = IterationStopReason::CompletedRequestedRounds;
        let mut loss_per_completed_round = Vec::new();
        let mut validation_loss_per_completed_round = Vec::new();
        let mut sampled_rows_per_completed_round = Vec::new();
        let mut sampled_features_per_completed_round = Vec::new();
        let mut diagnostics_per_round: Vec<IterationDiagnostics> = Vec::new();
        let mut best_validation_loss = initial_validation_loss;
        let mut best_validation_round = initial_validation_loss.map(|_| 0_usize);
        let mut validation_no_improvement_rounds = 0_usize;
        let mut weak_improvement_streak = 0_usize;
        let mut weak_improvement_rounds_committed = 0_usize;
        let mut current_validation_loss = initial_validation_loss;
        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(n);

        let effective_round_cap = controls.rounds;

        // Build MorphState (K classes) for the duration of training when
        // morph_config is set. Total iterations spans warm-start prefix + new rounds.
        let total_iterations = (effective_round_cap + round_index_offset) as u32;
        let mut morph_state: Option<MorphState> = self
            .params
            .morph_config
            .map(|cfg| MorphState::new(cfg, k, total_iterations, self.params.learning_rate));

        // v0.7.3 EMA warm-start: when the multiclass warm-start state
        // carries an EMA snapshot from the previous fit, seed the
        // current `MorphState` with it.  Length mismatch (class count
        // changed across fits) silently falls back to the cold EMA
        // from `MorphState::new`.
        if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), warm_ema_stats.as_ref())
            && ms.ema_stats.len() == snapshot.len()
        {
            ms.ema_stats.copy_from_slice(snapshot);
        }

        // v0.10.1: DART state for multiclass. The flat per-tree weight
        // pool is indexed by `round_index * K + class_k` and committed
        // in lockstep with `class_stumps[class_k]` during the round
        // loop. Warm-start seeds the weights from
        // `warm_dart_tree_weights`; historical RNG-driven dropouts are
        // NOT persisted (same as the binary path).
        //
        // Multiclass-specific bookkeeping (mirrors the binary path's
        // `round_start_offsets` / `dart_round_counts` but tracks each
        // class-tree separately because level-wise trees span multiple
        // stumps per (round, class)):
        //
        // * `dart_round_start_offsets[class_k][r]` — starting index in
        //   `class_stumps[class_k]` for class `class_k`'s tree in round
        //   `r`. Length == `effective_round_index + 1` (with phantom
        //   slots for skipped warmup rounds, matching the binary path).
        // * `dart_round_counts[class_k][r]` — number of stumps in class
        //   `class_k`'s tree at round `r`. `0` means no tree committed
        //   that round (e.g. zero-stump class during warmup).
        //
        // The flat dropout index `flat_idx = r * K + class_k` maps to
        // `&class_stumps[class_k][start..start+count]` via these arrays.
        let mut dart_state = DartState::default();
        let mut dart_round_start_offsets: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut dart_round_counts: Vec<Vec<usize>> = vec![Vec::new(); k];
        if dart_params.is_some() {
            let initial_tree_count = round_index_offset * k;
            if let Some(per_tree) = warm_dart_tree_weights.as_ref() {
                if per_tree.len() != initial_tree_count {
                    return Err(EngineError::ContractViolation(format!(
                        "warm-start initial_dart_tree_weights length {} != \
                         initial_rounds_completed * K = {} * {} = {}",
                        per_tree.len(),
                        round_index_offset,
                        k,
                        initial_tree_count,
                    )));
                }
                dart_state.tree_weights = per_tree.clone();
            } else {
                dart_state.tree_weights = vec![1.0; initial_tree_count];
            }
            for _ in 0..round_index_offset {
                dart_state.dropped_per_round.push(Vec::new());
            }
            // Reconstruct per-class `dart_round_start_offsets` and
            // `dart_round_counts` for warm-start by grouping
            // `class_stumps[class_k]` by tree_id (decoded from
            // `stump.split.node_id`). All stumps of the same class-tree
            // share a tree_id, and distinct tree_ids appear in
            // round-order, so this gives the contiguous slice
            // boundaries the dropout/normalize code needs.
            for class_k in 0..k {
                let class_total = class_stumps[class_k].len();
                let mut i = 0_usize;
                while i < class_total {
                    let (tid_first, _) =
                        decode_tree_node_id(class_stumps[class_k][i].split.node_id);
                    let start = i;
                    let mut j = i + 1;
                    while j < class_total {
                        let (tid, _) = decode_tree_node_id(class_stumps[class_k][j].split.node_id);
                        if tid != tid_first {
                            break;
                        }
                        j += 1;
                    }
                    dart_round_start_offsets[class_k].push(start);
                    dart_round_counts[class_k].push(j - start);
                    i = j;
                }
                // Pad with phantom (count=0) entries up to
                // round_index_offset so the array length matches the
                // warm-start round count.
                while dart_round_start_offsets[class_k].len() < round_index_offset {
                    dart_round_start_offsets[class_k].push(class_total);
                    dart_round_counts[class_k].push(0);
                }
            }
        }

        for round_index in 0..effective_round_cap {
            let effective_round = round_index + round_index_offset;

            // v0.10.1 DART: drop a random subset of previously-committed
            // class-trees BEFORE computing gradients. The flat pool
            // `dart_state.tree_weights` is indexed by
            // `prior_round * K + class_k`. For each dropped flat index,
            // subtract `w_old * tree_contribution` from the
            // corresponding `class_predictions[class_k]` so the new
            // round's gradients are computed on the dropped-out
            // residual.
            //
            // PR review (C4): a level-wise tree spans MULTIPLE stumps,
            // not one stump per (round, class). Use the per-class
            // `dart_round_start_offsets` / `dart_round_counts` arrays
            // (built from tree_id grouping) to subtract the WHOLE class
            // tree's contribution, mirroring the single-output DART
            // path's `apply_weighted_round_to_predictions(&stumps[start..start+count], ...)`.
            //
            // Backups of `class_predictions` are recorded BEFORE
            // mutation so an early-exit (`!any_tree_produced`, loss
            // regression, etc.) can restore the full pre-dropout
            // ensemble — matching the single-output DART semantics
            // (PR review C1).
            let mut dart_predictions_backup: Option<Vec<Vec<f32>>> = None;
            let dropped_tree_indices: Vec<usize> =
                if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round,
                    );
                    if !drops.is_empty() {
                        dart_predictions_backup = Some(class_predictions.clone());
                    }
                    for &flat_idx in &drops {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let w_old = dart_state.tree_weights[flat_idx];
                        // Snapshot the class-tree slice to a Vec so we can
                        // safely re-borrow `class_predictions[class_k]` as
                        // mutable. (Stumps don't change between subtract
                        // and re-add — the slice can be reused later.)
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        apply_weighted_round_to_predictions(
                            &mut class_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            -w_old,
                        )?;
                    }
                    drops
                } else {
                    Vec::new()
                };

            // v0.10.1: pre-compute per-class gradient buffers BEFORE sampling
            // so the multiclass GOSS scorer can see all K gradient channels
            // when ranking rows by `s_i = sum_k |g_{i,k}|`. The original
            // gradient norms (for diagnostics) and the projected buffers are
            // both captured up front.
            let mut class_gradient_buffers: Vec<Vec<GradientPair>> = Vec::with_capacity(k);
            let mut class_original_gradient_norms: Vec<Option<f32>> = Vec::with_capacity(k);
            {
                let mut tmp_buf: Vec<GradientPair> = Vec::with_capacity(n);
                for class_k in 0..k {
                    objective.compute_gradients_for_class(
                        &class_predictions,
                        &dataset.targets,
                        dataset.sample_weights.as_deref(),
                        class_k,
                        &mut tmp_buf,
                    )?;
                    let original_norm = if gradient_projector.is_some() {
                        Some(gradient_l2_norm_only(&tmp_buf))
                    } else {
                        None
                    };
                    if let Some(projector) = &gradient_projector {
                        projector.project_gradient_pairs_in_place(&mut tmp_buf)?;
                    }
                    class_gradient_buffers.push(tmp_buf.clone());
                    class_original_gradient_norms.push(original_norm);
                }
            }

            // Shared row sampling across all K classes. In GOSS mode this
            // amplifies the sampled-low rows in every class buffer.
            let root_row_indices = select_row_indices_for_round_multiclass(
                self.params.boosting_mode,
                n,
                controls.row_subsample,
                sampling_seed_base,
                effective_round as u64,
                &mut class_gradient_buffers,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                feature_count,
                controls.col_subsample,
                sampling_seed_base,
                effective_round as u64,
            )?;
            let sampled_row_count = root_row_indices.len();

            // Copy current predictions to candidates
            for class_k in 0..k {
                class_candidate_predictions[class_k].copy_from_slice(&class_predictions[class_k]);
            }

            // Record stump counts before this round
            let pre_round_counts: Vec<usize> = class_stumps.iter().map(|s| s.len()).collect();

            // Build K trees
            let mut any_tree_produced = false;
            // Per-class diagnostics for this round; aggregated to a single
            // `IterationDiagnostics` after the class loop completes.
            let mut per_class_diagnostics: Vec<IterationDiagnostics> = Vec::with_capacity(k);
            for class_k in 0..k {
                // Use the pre-computed (and possibly GOSS-amplified) buffer.
                gradient_buffer.clear();
                gradient_buffer.extend_from_slice(&class_gradient_buffers[class_k]);
                let original_gradient_norm = class_original_gradient_norms[class_k];
                per_class_diagnostics.push(IterationDiagnostics::from_gradient_snapshot(
                    &gradient_buffer,
                    original_gradient_norm,
                    sampled_row_count,
                    feature_tiles.len(),
                ));

                // Update per-class EMA stats from this class's gradients.
                if let Some(ms) = morph_state.as_mut() {
                    ms.update_ema_from_gradient_pairs(&gradient_buffer, class_k);
                }

                let morph_tree_ctx: Option<MorphTreeContext<'_>> =
                    morph_state.as_ref().map(|ms| MorphTreeContext {
                        state: ms,
                        iteration: effective_round as u32,
                        total_iterations,
                        class_idx: class_k,
                        lr: ms.lr_for_iter(effective_round),
                    });

                let raw_fv = &dataset.matrix.values;
                let (round_stumps, _round_stop) = if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        effective_round,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        dataset.factor_exposures.as_ref(),
                    )?
                } else {
                    build_tree_level_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        effective_round,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        dataset.factor_exposures.as_ref(),
                    )?
                };

                if !round_stumps.is_empty() {
                    any_tree_produced = true;
                }
                class_stumps[class_k].extend(round_stumps);
            }

            // v0.10.1 DART post-build: rescale the K new trees to
            // `new_w = 1/(num_dropped + 1)` and re-add each dropped
            // tree's contribution at its post-normalize weight to BOTH
            // `class_predictions` and `class_candidate_predictions`.
            //
            // PR review (C1): `dart_state.tree_weights` mutation and
            // per-stump `tree_weight` stamping are DEFERRED to the
            // round-commit branch below. Rejecting the round
            // (`!any_tree_produced`, loss regression, etc.) restores
            // `class_predictions` from `dart_predictions_backup` so
            // the pre-dropout ensemble is preserved for the next round.
            //
            // PR review (C4, C5): use per-class
            // `dart_round_counts[class_k][prior_round]` to re-add the
            // WHOLE dropped class-tree (not just its root); compute
            // `new_dropped_weights` here but commit them only on
            // round acceptance.
            //
            // `dart_round_finalize` carries the per-round normalization
            // bookkeeping into the commit branch; `None` when DART is
            // off or the round had no dropouts (in which case new
            // trees get `tree_weight = 1.0`).
            let dart_round_finalize: Option<(f32, Vec<f32>)> =
                if let Some((_, _, normalize_type, _)) = dart_params {
                    let n_dropped = dropped_tree_indices.len() as f32;
                    let new_w = 1.0 / (n_dropped + 1.0);
                    let drop_factor = match normalize_type {
                        alloygbm_core::DartNormalize::Tree => n_dropped / (n_dropped + 1.0),
                        alloygbm_core::DartNormalize::Forest => 1.0 / (n_dropped + 1.0),
                    };
                    // Scale each class's new-tree contribution from w=1
                    // (as built into candidate) to w=new_w.
                    // class_candidate[k] = class_predictions[k] + new_w * f_T_k.
                    for class_k in 0..k {
                        let n_rows = class_candidate_predictions[class_k].len();
                        for r in 0..n_rows {
                            let f_t = class_candidate_predictions[class_k][r]
                                - class_predictions[class_k][r];
                            class_candidate_predictions[class_k][r] =
                                class_predictions[class_k][r] + new_w * f_t;
                        }
                    }
                    let new_dropped_weights: Vec<f32> = dropped_tree_indices
                        .iter()
                        .map(|&fi| dart_state.tree_weights[fi] * drop_factor)
                        .collect();
                    // Re-add each dropped tree's WHOLE slice at the rescaled
                    // weight to BOTH class_predictions (so post-round
                    // commit captures the full ensemble) AND
                    // class_candidate_predictions (so candidate_loss is
                    // computed against the correct full ensemble).
                    for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        let w_new = new_dropped_weights[i];
                        apply_weighted_round_to_predictions(
                            &mut class_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            w_new,
                        )?;
                        apply_weighted_round_to_predictions(
                            &mut class_candidate_predictions[class_k],
                            binned_matrix,
                            &stump_slice,
                            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
                            w_new,
                        )?;
                    }
                    Some((new_w, new_dropped_weights))
                } else {
                    None
                };

            let in_warmup_phase = morph_state
                .as_ref()
                .is_some_and(|ms| ms.is_in_warmup_phase(effective_round));

            // PR review (C1): rejection paths must restore
            // `class_predictions` from `dart_predictions_backup` so the
            // next round sees the full pre-dropout ensemble.
            // `dart_state.tree_weights` was NOT mutated above, so no
            // weight rollback is needed.
            if !any_tree_produced {
                if in_warmup_phase {
                    // Empty rounds during warmup are expected: tiny LR produces
                    // leaves below `min_abs_leaf_value`, so all splits get
                    // rejected. This is benign — LR will ramp up. Restore
                    // class_predictions and skip this round.
                    if let Some(backup) = dart_predictions_backup.take() {
                        class_predictions = backup;
                    }
                    rounds_completed += 1;
                    continue;
                }
                // Past warmup: an empty round indicates no useful split exists.
                // Break path: predictions aren't read again after the loop.
                let _ = dart_predictions_backup.take();
                for class_k in 0..k {
                    class_stumps[class_k].truncate(pre_round_counts[class_k]);
                }
                stop_reason = IterationStopReason::NoSplitCandidate;
                break;
            }

            // Check loss improvement
            let candidate_loss = objective.loss(
                &class_candidate_predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            if loss_improvement < 0.0 {
                // Truncate stumps so the model doesn't include this round's contribution.
                // (`class_candidate_predictions` is reset from `class_predictions` at the
                // top of each round, so the candidate state is implicitly rolled back.)
                for class_k in 0..k {
                    class_stumps[class_k].truncate(pre_round_counts[class_k]);
                }
                if in_warmup_phase {
                    // During warmup, slightly-negative loss improvements arise from
                    // numerical noise at tiny LR (e.g., row-subsample variance over
                    // mostly-zero gradient updates). The model is not broken — LR will
                    // ramp up. Restore class_predictions and skip this round.
                    if let Some(backup) = dart_predictions_backup.take() {
                        class_predictions = backup;
                    }
                    rounds_completed += 1;
                    continue;
                }
                // Break path: predictions aren't read again after the
                // loop, so backup restore is unnecessary.
                let _ = dart_predictions_backup.take();
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if !in_warmup_phase {
                let lr_threshold_scale = morph_state
                    .as_ref()
                    .map_or(1.0, |ms| ms.lr_loss_threshold_scale(effective_round));
                let effective_min_loss_improvement =
                    controls.min_loss_improvement * lr_threshold_scale;
                if loss_improvement < effective_min_loss_improvement {
                    if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                        for class_k in 0..k {
                            class_stumps[class_k].truncate(pre_round_counts[class_k]);
                        }
                        // Break path: predictions aren't read again
                        // after the loop, so backup restore is
                        // unnecessary; let it drop silently.
                        let _ = dart_predictions_backup.take();
                        stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                        break;
                    }
                    weak_improvement_streak += 1;
                    weak_improvement_rounds_committed += 1;
                } else {
                    weak_improvement_streak = 0;
                }
            }

            // Validation early stopping
            //
            // PR review (C6): when DART is active, mirror the training
            // transition on `validation_class_predictions` so the
            // validation loss is computed against the same full
            // ensemble (post-dropout, scaled new tree, re-added dropped
            // trees). Without this the early-stopping decision is made
            // against an inconsistent ensemble.
            let mut stop_for_validation_plateau = false;
            if let Some(validation_ref) = validation {
                let val_preds = validation_class_predictions.as_mut().unwrap();
                let val_raw = Some((
                    &validation_ref.dataset.matrix.values as &[f32],
                    validation_ref.dataset.matrix.feature_count,
                ));
                if let Some((new_w, new_dropped_weights)) = dart_round_finalize.as_ref() {
                    // 1. Subtract each dropped class-tree at w_old.
                    for &flat_idx in &dropped_tree_indices {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        let w_old = dart_state.tree_weights[flat_idx];
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            &stump_slice,
                            val_raw,
                            -w_old,
                        )?;
                    }
                    // 2. Add the new K class-trees at new_w.
                    for class_k in 0..k {
                        let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                        if round_stumps.is_empty() {
                            continue;
                        }
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            round_stumps,
                            val_raw,
                            *new_w,
                        )?;
                    }
                    // 3. Re-add each dropped class-tree at its new weight.
                    for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
                        let prior_round = flat_idx / k;
                        let class_k = flat_idx % k;
                        let count = dart_round_counts[class_k]
                            .get(prior_round)
                            .copied()
                            .unwrap_or(0);
                        if count == 0 {
                            continue;
                        }
                        let start = dart_round_start_offsets[class_k][prior_round];
                        let stump_slice = class_stumps[class_k][start..start + count].to_vec();
                        let w_new = new_dropped_weights[i];
                        apply_weighted_round_to_predictions(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            &stump_slice,
                            val_raw,
                            w_new,
                        )?;
                    }
                } else {
                    // Non-DART (or DART with no dropouts AND new trees
                    // not yet rescaled): plain unit-weight tree walk.
                    for class_k in 0..k {
                        let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                        if !round_stumps.is_empty() {
                            apply_round_stumps_tree_walk(
                                &mut val_preds[class_k],
                                validation_ref.binned_matrix,
                                round_stumps,
                                val_raw,
                            )?;
                        }
                    }
                }
                let next_validation_loss = objective.loss(
                    val_preds,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?;

                let improved = best_validation_loss
                    .map(|best| best - next_validation_loss > controls.min_validation_improvement)
                    .unwrap_or(true);
                if improved {
                    best_validation_loss = Some(next_validation_loss);
                    best_validation_round = Some(rounds_completed + 1);
                    validation_no_improvement_rounds = 0;
                } else if controls.early_stopping_rounds.is_some() {
                    validation_no_improvement_rounds += 1;
                }
                if let Some(patience) = controls.early_stopping_rounds
                    && validation_no_improvement_rounds >= patience
                {
                    stop_for_validation_plateau = true;
                }

                current_validation_loss = Some(next_validation_loss);
                validation_loss_per_completed_round.push(next_validation_loss);
            }

            // Accept round
            for class_k in 0..k {
                class_predictions[class_k].copy_from_slice(&class_candidate_predictions[class_k]);
            }
            current_loss = candidate_loss;
            loss_per_completed_round.push(candidate_loss);
            sampled_rows_per_completed_round.push(sampled_row_count);
            sampled_features_per_completed_round.push(sampled_feature_count);
            let stump_counts_this_round: Vec<usize> = (0..k)
                .map(|c| class_stumps[c].len() - pre_round_counts[c])
                .collect();
            stumps_per_round_per_class.push(stump_counts_this_round.clone());
            diagnostics_per_round.push(IterationDiagnostics::aggregate_per_class(
                &per_class_diagnostics,
            ));

            // v0.10.1 DART: commit per-tree-weight state for this round
            // ONLY after acceptance (PR review C1). On rejection we
            // never reach this point and `dart_state.tree_weights` /
            // `dart_round_start_offsets` / `dart_round_counts` keep
            // their pre-round shape, so the flat dropout index ↔
            // tree-slice mapping stays consistent.
            //
            // PR review (C5): stamp `stump.tree_weight = new_w` on
            // EVERY stump in `class_stumps[class_k][pre_round_counts[class_k]..]`,
            // not just `last_mut()`. Only push DART slots for class
            // trees that actually produced stumps this round (so
            // zero-stump class trees stay as phantom slots and
            // `dart_round_counts` reflects 0 for them).
            if let Some((new_w, new_dropped_weights)) = dart_round_finalize.as_ref() {
                // Rescale dropped trees' weights in place.
                for (i, &flat_idx) in dropped_tree_indices.iter().enumerate() {
                    dart_state.tree_weights[flat_idx] = new_dropped_weights[i];
                }
                // Record this round in `dropped_per_round` (one entry
                // per multiclass round even though K trees are
                // committed).
                dart_state
                    .dropped_per_round
                    .push(dropped_tree_indices.clone());
                // Per-class round bookkeeping + tree_weight stamping.
                for class_k in 0..k {
                    let count = stump_counts_this_round[class_k];
                    let start = pre_round_counts[class_k];
                    dart_round_start_offsets[class_k].push(start);
                    dart_round_counts[class_k].push(count);
                    if count > 0 {
                        for stump in class_stumps[class_k][start..start + count].iter_mut() {
                            stump.tree_weight = *new_w;
                        }
                    }
                    // Push the per-tree weight regardless of count so
                    // the flat layout `r * K + class_k` is preserved.
                    // Phantom (count=0) trees get weight=new_w too;
                    // they contribute nothing to predictions but the
                    // flat indexing stays consistent across rounds.
                    dart_state.tree_weights.push(*new_w);
                }
            }

            rounds_completed += 1;

            if stop_for_validation_plateau {
                stop_reason = IterationStopReason::ValidationLossPlateau;
                break;
            }
        }

        // Truncate to best validation round if early stopping triggered
        if stop_reason == IterationStopReason::ValidationLossPlateau
            && let Some(best_round) = best_validation_round
            && best_round < rounds_completed
        {
            // Compute how many stumps to keep per class (inherited + best new rounds)
            for class_k in 0..k {
                let keep_count: usize = initial_stump_counts[class_k]
                    + stumps_per_round_per_class
                        .iter()
                        .take(best_round)
                        .map(|r| r[class_k])
                        .sum::<usize>();
                class_stumps[class_k].truncate(keep_count);
            }
            loss_per_completed_round.truncate(best_round);
            validation_loss_per_completed_round.truncate(best_round);
            sampled_rows_per_completed_round.truncate(best_round);
            sampled_features_per_completed_round.truncate(best_round);
            diagnostics_per_round.truncate(best_round);
            rounds_completed = best_round;
        }

        let final_loss = current_loss;
        let final_validation_loss = current_validation_loss;

        let morph_metadata = morph_state.as_ref().map(|ms| MorphMetadataPayload {
            config: ms.config,
            final_iteration: rounds_completed as u32,
            final_total: total_iterations,
            // v0.7.3: persist EMA so warm-start can resume from the
            // exact same EMA state rather than restarting cold.
            ema_stats: ms.ema_stats.clone(),
        });
        let dro_metadata = self
            .params
            .dro_config
            .map(|config| DroMetadataPayload { config });
        Ok(MultiClassIterationRunSummary {
            model: MultiClassTrainedModel {
                num_classes: k,
                baseline_predictions: baselines,
                feature_count,
                class_stumps,
                categorical_state: None,
                objective: objective.objective_name().to_string(),
                morph_metadata,
                dro_metadata,
            },
            rounds_requested: effective_round_cap,
            effective_round_cap,
            rounds_completed,
            stop_reason,
            initial_loss,
            initial_validation_loss,
            loss_per_completed_round,
            validation_loss_per_completed_round,
            sampled_rows_per_completed_round,
            sampled_features_per_completed_round,
            best_validation_loss,
            best_validation_round,
            weak_improvement_rounds_committed,
            final_loss,
            final_validation_loss,
            custom_metric_per_round: Vec::new(),
            custom_metric_name: None,
            diagnostics_per_round,
        })
    }

    fn default_iteration_controls(&self, rounds: usize) -> EngineResult<IterationControls> {
        let mut controls = IterationControls::new(
            rounds,
            self.params.min_split_gain,
            self.params.min_data_in_leaf as usize,
            0.0,
            1_000_000.0,
            0.0,
            0,
        )?
        .with_subsample_rates(self.params.row_subsample, self.params.col_subsample)?;
        if let Some(early_stopping_rounds) = self.params.early_stopping_rounds {
            controls = controls.with_validation_early_stopping(
                early_stopping_rounds as usize,
                self.params.min_validation_improvement,
            )?;
        }
        controls = controls.with_max_leaves(self.params.max_leaves)?;
        Ok(controls)
    }

    fn auto_iteration_controls(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        rounds: usize,
        is_ranking: bool,
    ) -> EngineResult<IterationControls> {
        validate_training_alignment(dataset, binned_matrix)?;
        let mut controls = self.default_iteration_controls(rounds)?;
        let row_count = dataset.row_count();
        let feature_count = binned_matrix.feature_count;
        let target_variance = target_variance(&dataset.targets, dataset.sample_weights.as_deref())?;
        if row_count < 1_024 {
            let rows_per_feature = row_count as f32 / feature_count.max(1) as f32;
            if feature_count >= 8
                && rounds > 256
                && rows_per_feature < 64.0
                && target_variance > 1.0
            {
                controls.rounds = rounds.min(96);
            }
            return Ok(controls);
        }

        let binned_density = binned_feature_density(binned_matrix);

        let suggested_min_rows = if row_count < 128 {
            1
        } else if row_count < 512 {
            2
        } else if row_count < 2_048 {
            4
        } else if row_count < 8_192 {
            8
        } else {
            16
        };
        let user_min = self.params.min_data_in_leaf as usize;
        controls.min_rows_per_leaf = suggested_min_rows
            .max(user_min)
            .min(row_count.saturating_div(2).max(1));

        // Ranking objectives produce gradients and round-to-round loss deltas
        // that are orders of magnitude smaller than regression/classification,
        // because the loss is bounded by NDCG normalization and pairwise
        // weighting. The density-based min_split_gain floor and the
        // target-variance-scaled min_loss_improvement threshold were tuned for
        // regression losses and cause ranking training to exit after only a
        // handful of rounds with LossImprovementBelowThreshold. Disable both
        // guards for ranking so the user-supplied min_split_gain (default 0.0)
        // and n_estimators are honored.
        let auto_min_split_gain: f32 = if is_ranking {
            0.0
        } else if binned_density < 0.10 {
            0.001
        } else if row_count.saturating_mul(feature_count) >= 65_536 {
            0.0001
        } else {
            0.0
        };
        controls.min_split_gain = auto_min_split_gain.max(self.params.min_split_gain);
        controls.min_loss_improvement = if is_ranking || row_count < 4_096 {
            0.0
        } else {
            (target_variance.max(1e-6) * 1e-5).min(0.01)
        };
        controls.max_consecutive_weak_improvements = if is_ranking || row_count < 4_096 {
            0
        } else if rounds <= 64 {
            1
        } else {
            3
        };

        if self.params.row_subsample == 1.0 && row_count >= 2_048 {
            controls.row_subsample = if row_count >= 16_384 { 0.8 } else { 0.9 };
        }
        if self.params.col_subsample == 1.0 && feature_count >= 32 {
            controls.col_subsample = if feature_count >= 256 {
                0.5
            } else if feature_count >= 128 {
                0.65
            } else {
                0.8
            };
        }

        Ok(controls)
    }

    fn fit_iterations_with_optional_validation_summary<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
        mut execution: IterationExecutionContext<'_>,
    ) -> EngineResult<IterationRunSummary> {
        let controls = execution.controls;
        let validation = execution.validation;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
        // v0.9.0: DART support is wired through for the single-output
        // trainer. Warm-start + DART is not yet supported — would
        // require persisting `tree_weights` and `dropped_per_round` in
        // `WarmStartState` (tracked as a v0.10.x follow-up).
        let dart_params = match self.params.boosting_mode {
            BoostingMode::Dart {
                drop_rate,
                max_drop,
                normalize_type,
                sample_type,
            } => Some((drop_rate, max_drop, normalize_type, sample_type)),
            _ => None,
        };
        // v0.10.0: DART + warm_start is now supported. See the dart_state
        // seeding logic below — `dart_state.tree_weights` is initialized from
        // `warm_start.initial_dart_tree_weights` when present, falling back
        // to all-1.0s. Historical `dropped_per_round` is not persisted; new
        // rounds start fresh dropout bookkeeping going forward, which is
        // the natural semantics for continuation (RNG-driven dropout
        // history cannot be replayed from the prior fit).
        validate_training_alignment(dataset, binned_matrix)?;
        if objective.requires_group_id() && dataset.group_id.is_none() {
            return Err(EngineError::ContractViolation(
                "this objective requires group_id to be provided on the training dataset"
                    .to_string(),
            ));
        }
        if let Some(validation_ref) = validation {
            validate_training_alignment(validation_ref.dataset, validation_ref.binned_matrix)?;
            if validation_ref.dataset.matrix.feature_count != dataset.matrix.feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "validation feature_count {} does not match training feature_count {}",
                    validation_ref.dataset.matrix.feature_count, dataset.matrix.feature_count
                )));
            }
            if objective.requires_group_id() && validation_ref.dataset.group_id.is_none() {
                return Err(EngineError::ContractViolation(
                    "this objective requires group_id to be provided on the validation dataset"
                        .to_string(),
                ));
            }
        }
        validate_train_params(&self.params)?;
        if let Some(qa) = objective.quantile_alpha() {
            if !qa.is_finite() || qa <= 0.0 || qa >= 1.0 {
                return Err(EngineError::InvalidConfig(
                    "quantile_alpha must be finite and in (0.0, 1.0)".to_string(),
                ));
            }
            if self.params.leaf_model == LeafModelKind::Linear {
                return Err(EngineError::InvalidConfig(
                    "leaf_model='linear' is not supported with objective='quantile'".to_string(),
                ));
            }
            if matches!(self.params.boosting_mode, BoostingMode::Dart { .. }) {
                return Err(EngineError::InvalidConfig(
                    "boosting_mode='dart' is not supported with objective='quantile'".to_string(),
                ));
            }
            if self.params.morph_config.is_some() {
                return Err(EngineError::InvalidConfig(
                    "morph_config is not supported with objective='quantile'".to_string(),
                ));
            }
        }
        validate_training_dataset(dataset)?;
        validate_neutralization_fit_contract(&self.params, dataset, objective)?;
        validate_warm_start_neutralization_contract(
            &self.params,
            execution.warm_start.is_some(),
            dataset,
        )?;
        let owned_dataset = if execution.pre_target_already_applied {
            None
        } else {
            prepare_pre_target_training_dataset(&self.params, dataset)?
        };
        let active_dataset = owned_dataset.as_ref().unwrap_or(dataset);
        let fit_contract =
            self.evaluate_fit_contract_on_active_dataset(active_dataset, objective)?;
        let gradient_projector = if let Some(config) = gradient_neutralization_config(&self.params)
        {
            let exposures = active_dataset.factor_exposures.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "factor_exposures are required when neutralization is active".to_string(),
                )
            })?;
            Some(FactorProjector::new(
                exposures,
                active_dataset.sample_weights.as_deref(),
                config.ridge_lambda,
            )?)
        } else {
            None
        };
        let sampling_seed_base = sampling_seed_base(self.params.seed, self.params.deterministic);
        let split_options = split_selection_options_for_training(
            &self.params,
            execution.policy_mode,
            active_dataset,
            binned_matrix,
        )?;

        // Warm-start: use existing model's baseline + apply existing
        // trees.  `warm_ema_stats` captures the MorphBoost EMA snapshot
        // for the v0.7.3 warm-start-equivalence fix (consumed below
        // when `MorphState::new` builds the fresh EMA).
        // v0.10.0 review fix (Comment 1): also capture
        // `initial_dart_tree_weights` here BEFORE `take()` consumes the
        // warm_start; the dart_state seeding step below reads this local
        // variable rather than re-querying `execution.warm_start`.
        let (
            baseline_prediction,
            initial_stumps,
            round_index_offset,
            warm_ema_stats,
            initial_dart_tree_weights,
        ) = if let Some(warm_start) = execution.warm_start.take() {
            (
                warm_start.baseline_prediction,
                warm_start.stumps,
                warm_start.initial_rounds_completed,
                warm_start.initial_ema_stats,
                warm_start.initial_dart_tree_weights,
            )
        } else {
            (fit_contract.baseline_prediction, Vec::new(), 0, None, None)
        };
        let raw_features_opt = Some((
            &active_dataset.matrix.values as &[f32],
            active_dataset.matrix.feature_count,
        ));
        let mut predictions = vec![baseline_prediction; active_dataset.row_count()];
        if !initial_stumps.is_empty() {
            apply_tree_to_binned_predictions(
                &mut predictions,
                binned_matrix,
                &initial_stumps,
                raw_features_opt,
            )?;
        }
        let mut candidate_predictions = predictions.clone();
        let mut validation_predictions = if let Some(validation_ref) = validation {
            let mut vp = vec![baseline_prediction; validation_ref.dataset.row_count()];
            if !initial_stumps.is_empty() {
                apply_tree_to_binned_predictions(
                    &mut vp,
                    validation_ref.binned_matrix,
                    &initial_stumps,
                    Some((
                        &validation_ref.dataset.matrix.values as &[f32],
                        validation_ref.dataset.matrix.feature_count,
                    )),
                )?;
            }
            Some(vp)
        } else {
            None
        };
        let mut stumps = initial_stumps;
        let initial_stump_count = stumps.len();
        // `stumps_per_completed_round` stays NEW-ROUND-ONLY (its original
        // semantics): downstream consumers (validation early-stopping
        // truncation via `retained_stump_count_for_rounds`, leaf refinement,
        // DART replay truncation) index into it with a `best_round` value
        // relative to the new fit and assume entry `i` holds the i-th
        // newly-committed round's stump count.
        let mut stumps_per_completed_round: Vec<usize> = Vec::new();
        // Separate vector for warm-start prior-round counts. v0.10.0 review
        // follow-up: this used to live in `stumps_per_completed_round`, but
        // that broke `best_round`-indexed truncation when warm_start combined
        // with eval_set early stopping (kept counts from old rounds rather
        // than new ones). Now kept as its own local consumed only by the
        // DART-state seeding + round_start_offsets/dart_round_counts
        // pre-population blocks below.
        let mut initial_stumps_per_round: Vec<usize> = Vec::new();
        if !stumps.is_empty() {
            let mut current_tree_id = decode_tree_node_id(stumps[0].split.node_id).0;
            let mut current_count = 0usize;
            for stump in &stumps {
                let tree_id = decode_tree_node_id(stump.split.node_id).0;
                if tree_id != current_tree_id {
                    initial_stumps_per_round.push(current_count);
                    current_tree_id = tree_id;
                    current_count = 0;
                }
                current_count += 1;
            }
            initial_stumps_per_round.push(current_count);
        }
        let mut rounds_completed = 0_usize;
        let effective_round_cap = controls.rounds;
        let mut stop_reason = IterationStopReason::CompletedRequestedRounds;
        let initial_loss = objective.loss(
            &predictions,
            &active_dataset.targets,
            active_dataset.sample_weights.as_deref(),
        )?;
        let initial_validation_loss = if let Some(validation_ref) = validation {
            let validation_predictions_ref = validation_predictions.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "validation predictions were not initialized".to_string(),
                )
            })?;
            Some(objective.loss(
                validation_predictions_ref,
                &validation_ref.dataset.targets,
                validation_ref.dataset.sample_weights.as_deref(),
            )?)
        } else {
            None
        };
        let mut current_loss = initial_loss;
        let mut current_validation_loss = initial_validation_loss;
        let mut loss_per_completed_round = Vec::new();
        let mut validation_loss_per_completed_round = Vec::new();
        let mut sampled_rows_per_completed_round = Vec::new();
        let mut sampled_features_per_completed_round = Vec::new();
        let mut diagnostics_per_round: Vec<IterationDiagnostics> = Vec::new();
        let mut best_validation_loss = initial_validation_loss;
        let mut best_validation_round = initial_validation_loss.map(|_| 0_usize);
        let mut validation_no_improvement_rounds = 0_usize;
        let mut weak_improvement_streak = 0_usize;
        let mut weak_improvement_rounds_committed = 0_usize;

        // Custom metric tracking
        let custom_metric_callback = execution.custom_metric_callback;
        let mut custom_metric_per_round: Vec<f32> = Vec::new();
        let custom_metric_name = custom_metric_callback.map(|cb| cb.metric_name().to_string());
        let custom_metric_higher_is_better = custom_metric_callback
            .map(|cb| cb.higher_is_better())
            .unwrap_or(false);
        let mut best_custom_metric: Option<f32> = None;
        let mut best_custom_metric_round: Option<usize> = None;
        let mut custom_metric_no_improvement_rounds = 0_usize;

        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(active_dataset.row_count());

        // DART state: tree_weights[tree_id] tracks the multiplicative
        // weight applied to each previously-trained tree. Populated
        // before the round loop (initial weights = 1.0 for every
        // already-existing tree); per-round `select_dropouts` consults
        // it and `apply_normalization` mutates it. Stays empty for
        // non-DART fits, in which case the stamping step at the bottom
        // of this function is a no-op and stumps keep their default
        // `tree_weight = 1.0`.
        let mut dart_state = DartState::default();
        if dart_params.is_some() {
            // v0.10.0: When continuing a DART fit, seed tree_weights from the
            // warm-start snapshot captured above (BEFORE the take()). Length
            // must equal `stumps.len()` (one weight per warm-start stump).
            // Falls back to all-1.0s when the prior fit did not use DART
            // or no snapshot was provided. Uses `initial_stumps_per_round`
            // (warm-start prior-round counts) — distinct from
            // `stumps_per_completed_round` (new-round counts) so downstream
            // best_round-indexed truncation continues to work correctly.
            let initial_tree_count = initial_stumps_per_round.len();
            if let Some(saved_weights) = initial_dart_tree_weights.as_ref() {
                // Caller supplies one weight per stump; we need one weight
                // per tree. Take the first weight of each tree (all stumps
                // in a tree share the same tree_weight after DART
                // normalization, so this is well-defined).
                let mut per_tree = Vec::with_capacity(initial_tree_count);
                let mut stump_offset = 0usize;
                for &count in &initial_stumps_per_round {
                    let weight = saved_weights.get(stump_offset).copied().unwrap_or(1.0);
                    per_tree.push(weight);
                    stump_offset += count;
                }
                dart_state.tree_weights = per_tree;
            } else {
                dart_state.tree_weights = vec![1.0; initial_tree_count];
            }
            // Historical `dropped_per_round` is initialized empty per warm
            // round — RNG-driven dropout history cannot be replayed.
            for _ in 0..initial_tree_count {
                dart_state.dropped_per_round.push(Vec::new());
            }
        }
        // DART-only parallel arrays indexed by `effective_round_index`
        // (= `tree_id` encoded in stump.node_id). Both grow together at
        // commit-time, and skipped warmup rounds get phantom entries
        // (`count = 0`, `start = stumps.len()`) so the indexing stays
        // dense even when MorphBoost skips rounds.
        //
        // `round_start_offsets[t]` is the start index in `stumps` where
        // tree `t`'s stumps begin; `dart_round_counts[t]` is its stump
        // count.  Together they slice into `stumps` for the DART
        // dropout subtract/replay step.  Stays empty for non-DART
        // fits.  Keep separate from the pre-existing
        // `stumps_per_completed_round` (committed-only) so we don't
        // perturb downstream consumers like
        // `retained_stump_count_for_rounds`.
        let mut round_start_offsets: Vec<usize> = Vec::new();
        let mut dart_round_counts: Vec<usize> = Vec::new();

        // v0.10.0: DART + warm_start — pre-populate round_start_offsets +
        // dart_round_counts from the warm-start tree shapes so the dropout
        // step can correctly slice into `stumps` for each prior tree. Stays
        // a no-op for non-DART or cold fits. Uses `initial_stumps_per_round`
        // (warm-start prior-round counts) so `stumps_per_completed_round`
        // can stay new-round-only for downstream best_round indexing.
        if dart_params.is_some() {
            let mut offset = 0usize;
            for &count in &initial_stumps_per_round {
                round_start_offsets.push(offset);
                dart_round_counts.push(count);
                offset += count;
            }
        }

        // Build MorphState for the duration of training when morph_config is set.
        // `total_iterations` corresponds to the round cap (including any warm-start
        // offset already-completed rounds, so the LR schedule lines up).
        let total_iterations = (effective_round_cap + round_index_offset) as u32;
        let mut morph_state: Option<MorphState> = self
            .params
            .morph_config
            .map(|cfg| MorphState::new(cfg, 1, total_iterations, self.params.learning_rate));

        // v0.7.3 EMA warm-start: seed the fresh `MorphState` with the
        // EMA snapshot from the previous fit when the warm-start state
        // carries one.  Single-class MorphState has `ema_stats.len() = 1`.
        if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), warm_ema_stats.as_ref())
            && ms.ema_stats.len() == snapshot.len()
        {
            ms.ema_stats.copy_from_slice(snapshot);
        }

        for round_index in 0..effective_round_cap {
            // Offset round_index for sampling seeds and tree IDs when warm-starting
            let effective_round_index = round_index + round_index_offset;

            // DART: drop a random subset of previously-trained trees
            // before computing gradients. Subtract their (currently
            // weighted) contributions from `predictions` and
            // `validation_predictions` so the new tree fits on
            // residuals of the dropped-out ensemble. The dropped
            // trees are re-added (at rescaled weights) after the new
            // tree is committed; on early-exit (empty round, loss
            // regression, etc.) the buffer backups below are used to
            // restore `predictions`/`validation_predictions` to the
            // full-ensemble state so subsequent rounds aren't poisoned.
            let mut dart_predictions_backup: Option<Vec<f32>> = None;
            let mut dart_validation_backup: Option<Vec<f32>> = None;
            let dropped_tree_ids: Vec<usize> =
                if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                    let drops = select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        sampling_seed_base,
                        effective_round_index,
                    );
                    if !drops.is_empty() {
                        dart_predictions_backup = Some(predictions.clone());
                        dart_validation_backup = validation_predictions.clone();
                    }
                    for &tree_id in &drops {
                        let w_old = dart_state.tree_weights[tree_id];
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut predictions,
                            binned_matrix,
                            stump_slice,
                            raw_features_opt,
                            -w_old,
                        )?;
                        if let (Some(vp), Some(validation_ref)) =
                            (validation_predictions.as_mut(), validation)
                        {
                            let val_raw = Some((
                                &validation_ref.dataset.matrix.values as &[f32],
                                validation_ref.dataset.matrix.feature_count,
                            ));
                            apply_weighted_round_to_predictions(
                                vp,
                                validation_ref.binned_matrix,
                                stump_slice,
                                val_raw,
                                -w_old,
                            )?;
                        }
                    }
                    drops
                } else {
                    Vec::new()
                };

            // Helper to restore `predictions` and `validation_predictions`
            // from the pre-dropout backups. Called on every early-exit
            // path so the full-ensemble state is preserved.
            //
            // Captured as a small struct rather than a closure because
            // we need to call it from multiple branch arms below, and
            // closures over `&mut predictions` get awkward to reuse.
            //
            // NB: we cannot define a helper fn here since `predictions`
            // is borrowed mutably; we inline the restore at each
            // early-exit site instead. The backups are `Option<Vec<f32>>`
            // and `.take()`able so the restore is a single move.

            // v0.8.0: gradient computation moved before row sampling so
            // GOSS can score rows by `|gradient|`.  Standard / DART
            // boosting modes ignore the gradient input and fall back
            // to uniform subsampling.
            objective.compute_gradients_into(
                &predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
                &mut gradient_buffer,
            )?;
            // Capture the pre-projection L2 norm so neutralization
            // effectiveness can be reported alongside the post-projection
            // gradient stats below.  Only allocated when a per-round
            // projection is actually configured for this fit.
            let original_gradient_norm = if gradient_projector.is_some() {
                Some(gradient_l2_norm_only(&gradient_buffer))
            } else {
                None
            };
            if let Some(projector) = &gradient_projector {
                projector.project_gradient_pairs_in_place(&mut gradient_buffer)?;
            }
            let root_row_indices = select_row_indices_for_round(
                self.params.boosting_mode,
                active_dataset.row_count(),
                controls.row_subsample,
                sampling_seed_base,
                effective_round_index as u64,
                &mut gradient_buffer,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                binned_matrix.feature_count,
                controls.col_subsample,
                sampling_seed_base,
                effective_round_index as u64,
            )?;
            let sampled_row_count = root_row_indices.len();
            let gradients = &gradient_buffer;
            validate_gradient_pair_length(gradients, active_dataset.row_count())?;
            if cfg!(debug_assertions) {
                validate_gradient_pairs(gradients, active_dataset.row_count())?;
            }
            // Capture per-round telemetry from the *post-projection* gradient
            // buffer — i.e., the values the tree-building code actually
            // consumes.  Push happens further below, conditional on the
            // round being committed, so we stay in lockstep with the other
            // per-round vecs (loss_per_completed_round, etc.).  Round-level
            // cost: a single linear pass over the gradient buffer.
            let round_diagnostics = IterationDiagnostics::from_gradient_snapshot(
                gradients,
                original_gradient_norm,
                sampled_row_count,
                feature_tiles.len(),
            );

            // Update EMA stats from this round's gradients before tree-building so
            // morph split selection sees the latest mean/std.
            if let Some(ms) = morph_state.as_mut() {
                ms.update_ema_from_gradient_pairs(gradients, 0);
            }

            candidate_predictions.copy_from_slice(&predictions);

            let morph_tree_ctx: Option<MorphTreeContext<'_>> =
                morph_state.as_ref().map(|ms| MorphTreeContext {
                    state: ms,
                    iteration: effective_round_index as u32,
                    total_iterations,
                    class_idx: 0,
                    lr: ms.lr_for_iter(effective_round_index),
                });

            let raw_fv = &active_dataset.matrix.values;
            let (mut candidate_round_stumps, round_rejection_reason) =
                if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        gradients,
                        root_row_indices,
                        effective_round_index,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut candidate_predictions,
                        &self.params.feature_weights,
                        &execution.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        active_dataset.factor_exposures.as_ref(),
                    )?
                } else {
                    build_tree_level_wise(
                        backend,
                        binned_matrix,
                        gradients,
                        root_row_indices,
                        effective_round_index,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut candidate_predictions,
                        &self.params.feature_weights,
                        &execution.categorical_features,
                        morph_tree_ctx,
                        raw_fv,
                        active_dataset.factor_exposures.as_ref(),
                    )?
                };

            let in_warmup_phase = morph_state
                .as_ref()
                .is_some_and(|ms| ms.is_in_warmup_phase(effective_round_index));

            if candidate_round_stumps.is_empty() {
                // DART: empty round → no new tree → restore predictions
                // and validation_predictions from backup so subsequent
                // rounds see the full pre-dropout ensemble.
                if let Some(backup) = dart_predictions_backup.take() {
                    predictions = backup;
                }
                if let Some(backup) = dart_validation_backup.take() {
                    validation_predictions = Some(backup);
                }
                if in_warmup_phase {
                    // Empty rounds during warmup are expected: tiny LR produces
                    // leaves below `min_abs_leaf_value`, so all splits get
                    // rejected. This is benign — LR will ramp up. Skip this
                    // round and continue.
                    rounds_completed += 1;
                    continue;
                }
                stop_reason = round_rejection_reason;
                break;
            }

            if let Some(alpha) = objective.quantile_alpha() {
                let lr = morph_state
                    .as_ref()
                    .map(|ms| ms.lr_for_iter(effective_round_index))
                    .unwrap_or(self.params.learning_rate);
                refine_quantile_leaf_values(
                    &mut candidate_round_stumps,
                    binned_matrix,
                    &predictions,
                    &active_dataset.targets,
                    active_dataset.sample_weights.as_deref(),
                    alpha,
                    lr,
                    controls.max_abs_leaf_value,
                )?;
                candidate_predictions.copy_from_slice(&predictions);
                apply_round_stumps_tree_walk(
                    &mut candidate_predictions,
                    binned_matrix,
                    &candidate_round_stumps,
                    raw_features_opt,
                )?;
            }

            // DART: rebuild `candidate_predictions` to reflect the
            // post-normalization weights. After `build_tree_*` returned,
            // `candidate_predictions = predictions_dropped_out + 1.0 *
            // f_T(x)`. We want
            // `candidate_predictions = predictions_dropped_out + new_w *
            // f_T(x) + sum_dropped(w_new_i * f_i(x))`. Compute new
            // weights locally; the mutation of `dart_state.tree_weights`
            // only happens on commit (post-loss-check).
            //
            // `dart_round_finalize = Some((new_w, new_dropped_weights))`
            // on a DART round; `None` otherwise. The commit path
            // consumes this to update `dart_state`.
            let dart_round_finalize: Option<(f32, Vec<f32>)> =
                if let Some((_, _, normalize_type, _)) = dart_params {
                    let k = dropped_tree_ids.len() as f32;
                    let new_w = 1.0 / (k + 1.0);
                    let drop_factor = match normalize_type {
                        alloygbm_core::DartNormalize::Tree => k / (k + 1.0),
                        alloygbm_core::DartNormalize::Forest => 1.0 / (k + 1.0),
                    };
                    // Step 1: scale the new tree's f_T contribution from 1.0
                    // to new_w in candidate_predictions.
                    for r in 0..candidate_predictions.len() {
                        let f_t = candidate_predictions[r] - predictions[r];
                        candidate_predictions[r] = predictions[r] + new_w * f_t;
                    }
                    // Step 2: re-add each dropped tree to candidate_predictions
                    // at its new (post-normalize) weight.
                    let mut new_dropped_weights = Vec::with_capacity(dropped_tree_ids.len());
                    for &tree_id in &dropped_tree_ids {
                        let w_old = dart_state.tree_weights[tree_id];
                        let w_new = w_old * drop_factor;
                        new_dropped_weights.push(w_new);
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut candidate_predictions,
                            binned_matrix,
                            stump_slice,
                            raw_features_opt,
                            w_new,
                        )?;
                    }
                    Some((new_w, new_dropped_weights))
                } else {
                    None
                };

            let candidate_loss = objective.loss(
                &candidate_predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            // Ranking objectives (LambdaMART, pairwise, XeNDCG, YetiRank,
            // QueryRMSE) have bounded, NDCG-weighted losses whose round-to-
            // round training delta is often negative under row_subsample —
            // this does not reflect real ranking quality regression and the
            // boosting loop recovers on subsequent rounds. Skip the hard
            // "loss went up" early-exit for ranking objectives; rely on
            // validation early stopping (if configured) and the round cap.
            let objective_is_ranking = objective.requires_group_id();
            if !objective_is_ranking && loss_improvement < 0.0 {
                if in_warmup_phase {
                    // DART: loss regression on warmup continue path → restore
                    // from pre-dropout backup so the next round sees the
                    // full pre-dropout ensemble.
                    if let Some(backup) = dart_predictions_backup.take() {
                        predictions = backup;
                    }
                    if let Some(backup) = dart_validation_backup.take() {
                        validation_predictions = Some(backup);
                    }
                    // During warmup, slightly-negative loss improvements arise from
                    // numerical noise at tiny LR. Skip this round and continue;
                    // candidate predictions reset from current at the top of each round.
                    rounds_completed += 1;
                    continue;
                }
                // Break path: predictions is not read again after the loop,
                // so restoration is unnecessary here.
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if !in_warmup_phase {
                let lr_threshold_scale = morph_state
                    .as_ref()
                    .map_or(1.0, |ms| ms.lr_loss_threshold_scale(effective_round_index));
                let effective_min_loss_improvement =
                    controls.min_loss_improvement * lr_threshold_scale;
                if !objective_is_ranking && loss_improvement < effective_min_loss_improvement {
                    if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                        // Break path: see note above — restoration is
                        // unnecessary since the loop exits.
                        stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                        break;
                    }
                    weak_improvement_streak += 1;
                    weak_improvement_rounds_committed += 1;
                } else {
                    weak_improvement_streak = 0;
                }
            }

            let mut candidate_validation_predictions = None;
            let mut candidate_validation_loss = None;
            let mut stop_for_validation_plateau = false;
            let mut stop_for_custom_metric_plateau = false;
            if let Some(validation_ref) = validation {
                let mut next_validation_predictions =
                    validation_predictions.take().ok_or_else(|| {
                        EngineError::ContractViolation(
                            "validation predictions were not initialized".to_string(),
                        )
                    })?;
                // DART-aware: when dropouts happened this round, add the
                // new tree at `new_w` (not 1.0) and re-add dropped trees
                // at their new weights. Otherwise fall back to the
                // existing unit-weight tree walk.
                if let Some((new_w, new_dropped_weights)) = &dart_round_finalize {
                    let val_raw = Some((
                        &validation_ref.dataset.matrix.values as &[f32],
                        validation_ref.dataset.matrix.feature_count,
                    ));
                    apply_weighted_round_to_predictions(
                        &mut next_validation_predictions,
                        validation_ref.binned_matrix,
                        &candidate_round_stumps,
                        val_raw,
                        *new_w,
                    )?;
                    for (i, &tree_id) in dropped_tree_ids.iter().enumerate() {
                        let start = round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        let stump_slice = &stumps[start..start + count];
                        apply_weighted_round_to_predictions(
                            &mut next_validation_predictions,
                            validation_ref.binned_matrix,
                            stump_slice,
                            val_raw,
                            new_dropped_weights[i],
                        )?;
                    }
                } else {
                    apply_round_stumps_tree_walk(
                        &mut next_validation_predictions,
                        validation_ref.binned_matrix,
                        &candidate_round_stumps,
                        Some((
                            &validation_ref.dataset.matrix.values as &[f32],
                            validation_ref.dataset.matrix.feature_count,
                        )),
                    )?;
                }
                let next_validation_loss = objective.loss(
                    &next_validation_predictions,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?;

                // Custom metric callback: evaluate on validation predictions
                if let Some(cb) = custom_metric_callback {
                    let metric_value = cb.evaluate(
                        &next_validation_predictions,
                        &validation_ref.dataset.targets,
                        validation_ref.dataset.sample_weights.as_deref(),
                    )?;
                    custom_metric_per_round.push(metric_value);

                    // Custom metric drives early stopping when present
                    let metric_improved = match best_custom_metric {
                        Some(best) => {
                            if custom_metric_higher_is_better {
                                metric_value - best > controls.min_validation_improvement
                            } else {
                                best - metric_value > controls.min_validation_improvement
                            }
                        }
                        None => true,
                    };
                    if metric_improved {
                        best_custom_metric = Some(metric_value);
                        best_custom_metric_round = Some(rounds_completed + 1);
                        custom_metric_no_improvement_rounds = 0;
                    } else if controls.early_stopping_rounds.is_some() {
                        custom_metric_no_improvement_rounds += 1;
                    }
                    if let Some(patience) = controls.early_stopping_rounds
                        && custom_metric_no_improvement_rounds >= patience
                    {
                        stop_for_custom_metric_plateau = true;
                    }
                }

                // When custom metric is NOT present, use built-in validation loss for early stopping
                if custom_metric_callback.is_none() {
                    let improved = best_validation_loss
                        .map(|best| {
                            best - next_validation_loss > controls.min_validation_improvement
                        })
                        .unwrap_or(true);
                    if improved {
                        best_validation_loss = Some(next_validation_loss);
                        best_validation_round = Some(rounds_completed + 1);
                        validation_no_improvement_rounds = 0;
                    } else if controls.early_stopping_rounds.is_some() {
                        validation_no_improvement_rounds += 1;
                    }
                    if let Some(patience) = controls.early_stopping_rounds
                        && validation_no_improvement_rounds >= patience
                    {
                        stop_for_validation_plateau = true;
                    }
                } else {
                    // Still track validation loss for reporting, but don't use it for stopping
                    best_validation_loss = best_validation_loss
                        .map(|best| {
                            if next_validation_loss < best {
                                next_validation_loss
                            } else {
                                best
                            }
                        })
                        .or(Some(next_validation_loss));
                    if best_validation_loss == Some(next_validation_loss) {
                        best_validation_round = Some(rounds_completed + 1);
                    }
                }

                candidate_validation_predictions = Some(next_validation_predictions);
                candidate_validation_loss = Some(next_validation_loss);
            }

            std::mem::swap(&mut predictions, &mut candidate_predictions);
            current_loss = candidate_loss;
            loss_per_completed_round.push(candidate_loss);
            sampled_rows_per_completed_round.push(sampled_row_count);
            sampled_features_per_completed_round.push(sampled_feature_count);
            diagnostics_per_round.push(round_diagnostics);
            if let Some(next_validation_predictions) = candidate_validation_predictions {
                validation_predictions = Some(next_validation_predictions);
            }
            if let Some(next_validation_loss) = candidate_validation_loss {
                current_validation_loss = Some(next_validation_loss);
                validation_loss_per_completed_round.push(next_validation_loss);
            }

            // DART: commit the post-normalization weights to dart_state.
            // Backups (`dart_predictions_backup`, `dart_validation_backup`)
            // are loop-scoped, so they get dropped at end-of-iteration
            // automatically — no explicit reset needed.
            //
            // Pad all four DART-indexed parallel arrays
            // (`dart_state.tree_weights`, `dart_state.dropped_per_round`,
            // `round_start_offsets`, `dart_round_counts`) up to
            // `effective_round_index` with phantom entries for any
            // skipped warmup rounds.  Phantoms have weight=1.0 and
            // count=0, so a later `select_dropouts` could pick one but
            // the resulting subtract is a no-op
            // (`apply_weighted_round_to_predictions` early-returns on
            // empty stump slices).  This keeps tree_id (=
            // effective_round_index) and the DART arrays consistent
            // even when MorphBoost skips rounds.
            if dart_params.is_some() {
                while round_start_offsets.len() < effective_round_index {
                    round_start_offsets.push(stumps.len());
                    dart_round_counts.push(0);
                    dart_state.tree_weights.push(1.0);
                    dart_state.dropped_per_round.push(Vec::new());
                }
                if let Some((new_w, new_dropped_weights)) = dart_round_finalize {
                    for (i, &tree_id) in dropped_tree_ids.iter().enumerate() {
                        dart_state.tree_weights[tree_id] = new_dropped_weights[i];
                    }
                    dart_state.tree_weights.push(new_w);
                    dart_state.dropped_per_round.push(dropped_tree_ids.clone());
                } else {
                    dart_state.tree_weights.push(1.0);
                    dart_state.dropped_per_round.push(Vec::new());
                }
                round_start_offsets.push(stumps.len());
                dart_round_counts.push(candidate_round_stumps.len());
            }

            stumps_per_completed_round.push(candidate_round_stumps.len());
            stumps.extend(candidate_round_stumps);
            rounds_completed += 1;

            if stop_for_custom_metric_plateau {
                stop_reason = IterationStopReason::CustomMetricPlateau;
                break;
            }
            if stop_for_validation_plateau {
                stop_reason = IterationStopReason::ValidationLossPlateau;
                break;
            }
        }

        // Determine the best round for truncation: custom metric takes priority
        let truncation_round = if stop_reason == IterationStopReason::CustomMetricPlateau {
            best_custom_metric_round
        } else if stop_reason == IterationStopReason::ValidationLossPlateau {
            best_validation_round
        } else {
            None
        };

        if let Some(best_round) = truncation_round
            && best_round < rounds_completed
        {
            let kept_stumps =
                retained_stump_count_for_rounds(&stumps_per_completed_round, best_round);
            stumps.truncate(initial_stump_count + kept_stumps);
            stumps_per_completed_round.truncate(best_round);
            loss_per_completed_round.truncate(best_round);
            validation_loss_per_completed_round.truncate(best_round);
            custom_metric_per_round.truncate(best_round);
            sampled_rows_per_completed_round.truncate(best_round);
            sampled_features_per_completed_round.truncate(best_round);
            diagnostics_per_round.truncate(best_round);
            // DART: truncate the parallel DART arrays at the same point
            // as `stumps`, and recompute tree_weights from scratch using
            // only the kept rounds.  Round r's `apply_normalization` may
            // have rescaled weights of trees that themselves get
            // truncated in later rounds, so a naive
            // `dart_state.tree_weights.truncate(best_round)` would leave
            // the kept stumps stamped with weights mutated by trees that
            // no longer exist (the predictor would then return scores
            // that don't match the selected best iteration).  Replaying
            // through `apply_normalization` produces the exact weights
            // for the kept ensemble.
            if dart_params.is_some() {
                // `best_round` is in committed-round space, but the DART
                // arrays are indexed by effective_round_index which
                // includes phantom slots for skipped warmup rounds.
                // Map best_round → corresponding effective_round_index
                // by counting committed rounds in dart_round_counts.
                let mut committed_seen = 0usize;
                let mut truncate_at = dart_round_counts.len();
                for (idx, &count) in dart_round_counts.iter().enumerate() {
                    if count > 0 {
                        committed_seen += 1;
                        if committed_seen == best_round {
                            truncate_at = idx + 1;
                            break;
                        }
                    }
                }
                round_start_offsets.truncate(truncate_at);
                dart_round_counts.truncate(truncate_at);
                let kept_dropped = dart_state
                    .dropped_per_round
                    .iter()
                    .take(truncate_at)
                    .cloned()
                    .collect::<Vec<_>>();
                dart_state.tree_weights = vec![1.0; truncate_at];
                dart_state.dropped_per_round.truncate(truncate_at);
                for (r, dropped) in kept_dropped.iter().enumerate() {
                    if let Some((_, _, normalize_type, _)) = dart_params {
                        apply_normalization(
                            &mut dart_state.tree_weights,
                            dropped,
                            normalize_type,
                            r,
                        );
                    }
                }
            }
            rounds_completed = best_round;
            weak_improvement_rounds_committed =
                weak_improvement_rounds_committed.min(rounds_completed);
            current_loss = if rounds_completed == 0 {
                initial_loss
            } else {
                loss_per_completed_round[rounds_completed - 1]
            };
            current_validation_loss = if rounds_completed == 0 {
                initial_validation_loss
            } else {
                Some(validation_loss_per_completed_round[rounds_completed - 1])
            };
        }

        if experiment_leaf_refinement_enabled()
            && objective.supports_leaf_refinement()
            && gradient_neutralization_config(&self.params).is_none()
        {
            // Leaf refinement re-solves leaves against targets, so skip it for
            // per-round factor-neutralized gradients until refinement can apply
            // the same projection contract.
            refine_regression_leaf_values(
                baseline_prediction,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
                binned_matrix,
                &mut stumps,
                &stumps_per_completed_round,
                controls.max_abs_leaf_value,
            )?;

            let mut refined_predictions = vec![baseline_prediction; active_dataset.row_count()];
            apply_tree_to_binned_predictions(
                &mut refined_predictions,
                binned_matrix,
                &stumps,
                raw_features_opt,
            )?;
            current_loss = objective.loss(
                &refined_predictions,
                &active_dataset.targets,
                active_dataset.sample_weights.as_deref(),
            )?;
            if let Some(last_loss) = loss_per_completed_round.last_mut() {
                *last_loss = current_loss;
            }
            if let Some(validation_ref) = validation {
                let mut refined_validation_predictions =
                    vec![baseline_prediction; validation_ref.dataset.row_count()];
                apply_tree_to_binned_predictions(
                    &mut refined_validation_predictions,
                    validation_ref.binned_matrix,
                    &stumps,
                    None,
                )?;
                current_validation_loss = Some(objective.loss(
                    &refined_validation_predictions,
                    &validation_ref.dataset.targets,
                    validation_ref.dataset.sample_weights.as_deref(),
                )?);
                if let (Some(last_validation_loss), Some(refined_validation_loss)) = (
                    validation_loss_per_completed_round.last_mut(),
                    current_validation_loss,
                ) {
                    *last_validation_loss = refined_validation_loss;
                }
            }
        }

        let morph_metadata = morph_state.as_ref().map(|ms| MorphMetadataPayload {
            config: ms.config,
            final_iteration: rounds_completed as u32,
            final_total: total_iterations,
            // v0.7.3: persist EMA so warm-start can resume from the
            // exact same EMA state rather than restarting cold.
            ema_stats: ms.ema_stats.clone(),
        });
        let dro_metadata = self
            .params
            .dro_config
            .map(|config| DroMetadataPayload { config });
        // Record per-feature training-set means only for piecewise-linear
        // artifacts.  SHAP consumes these as the interventional baseline for
        // linear leaves; constant-leaf models have no use for it.
        let feature_baseline = if self.params.leaf_model == LeafModelKind::Linear {
            compute_feature_means_from_matrix(
                &active_dataset.matrix.values,
                active_dataset.matrix.feature_count,
                active_dataset.row_count(),
            )
        } else {
            None
        };
        // DART: stamp per-stump tree_weight from dart_state.tree_weights
        // (indexed by tree_id encoded in stump.node_id). Non-DART fits
        // leave tree_weight at its 1.0 default. This is what the
        // artifact write path (`TrainedModel::to_artifact_bytes`) inspects
        // to decide whether to emit a DartTreeWeights section.
        if dart_params.is_some() {
            for stump in stumps.iter_mut() {
                let (tree_id, _) = decode_tree_node_id(stump.split.node_id);
                let idx = tree_id as usize;
                if let Some(&w) = dart_state.tree_weights.get(idx) {
                    stump.tree_weight = w;
                }
            }
        }

        let model = TrainedModel {
            baseline_prediction,
            feature_count: active_dataset.matrix.feature_count,
            stumps,
            categorical_state: None,
            node_debug_stats: None,
            objective: objective.objective_name().to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata,
            dro_metadata,
            feature_baseline,
            neutralization_metadata: None,
        };
        let final_loss = current_loss;

        Ok(IterationRunSummary {
            model,
            rounds_requested: controls.rounds,
            effective_round_cap,
            rounds_completed,
            stop_reason,
            initial_loss,
            initial_validation_loss,
            loss_per_completed_round,
            validation_loss_per_completed_round,
            sampled_rows_per_completed_round,
            sampled_features_per_completed_round,
            best_validation_loss,
            best_validation_round,
            weak_improvement_rounds_committed,
            final_loss,
            final_validation_loss: current_validation_loss,
            custom_metric_per_round,
            custom_metric_name,
            diagnostics_per_round,
        })
    }

    pub fn fit_stub<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        binned_matrix: &BinnedMatrix,
        backend: &B,
        objective: &O,
    ) -> EngineResult<TrainRoundSummary> {
        self.fit_one_round(dataset, binned_matrix, backend, objective)
    }
}

/// Precomputed lookup driving interaction-constraint enforcement during
/// tree growth.  Built once per fit from `TrainParams.interaction_constraints`.
///
/// Each feature carries a `u64` bitset of the constraint groups it belongs
/// to.  Features outside every group ("unconstrained") have a `0` bitset and
/// are always allowed at any node.  Constrained features are allowed only
/// when at least one of their containing groups is still in the per-node
/// `active_groups` bitset; once a path commits to a group, the active set
/// narrows to the intersection of that feature's groups.
#[derive(Debug, Clone)]
pub(crate) struct InteractionConstraintIndex {
    /// `feature_groups[f]` — bitset of constraint-group indices that contain
    /// feature `f`.  `0` means the feature is unconstrained.
    feature_groups: Vec<u64>,
    /// Number of declared constraint groups (`≤ 64`).
    group_count: u32,
}

impl InteractionConstraintIndex {
    /// Build the lookup from `TrainParams.interaction_constraints`.  Returns
    /// `None` when no constraints are configured so callers can skip the
    /// per-node bookkeeping entirely.  Validation of group count / indices
    /// happens earlier in `core::validate_train_params`; this routine just
    /// re-checks bounds defensively (the engine doesn't always know the
    /// feature count at param-validation time).
    pub(crate) fn from_constraints(
        constraints: &[Vec<u32>],
        feature_count: usize,
    ) -> EngineResult<Option<Self>> {
        if constraints.is_empty() {
            return Ok(None);
        }
        if constraints.len() > 64 {
            return Err(EngineError::InvalidConfig(format!(
                "interaction_constraints supports at most 64 groups (got {})",
                constraints.len()
            )));
        }
        let mut feature_groups = vec![0u64; feature_count];
        for (gi, group) in constraints.iter().enumerate() {
            let bit = 1u64 << gi;
            for &f in group {
                let fi = f as usize;
                if fi >= feature_count {
                    return Err(EngineError::InvalidConfig(format!(
                        "interaction_constraints group {gi} references feature {f} which exceeds feature_count {feature_count}"
                    )));
                }
                feature_groups[fi] |= bit;
            }
        }
        Ok(Some(Self {
            feature_groups,
            group_count: constraints.len() as u32,
        }))
    }

    /// Bitset of all groups marked active at the tree root.  All groups
    /// start active and are intersected as a path descends through
    /// constrained features.
    pub(crate) fn root_active_groups(&self) -> u64 {
        if self.group_count == 0 {
            0
        } else if self.group_count >= 64 {
            u64::MAX
        } else {
            (1u64 << self.group_count) - 1
        }
    }

    /// Compute the `active_groups` bitset for a child node when the parent
    /// splits on `split_feature`.  Splitting on an unconstrained feature
    /// leaves the active set unchanged; splitting on a constrained feature
    /// narrows the set to groups that *also* contain that feature.
    #[inline]
    pub(crate) fn descend(&self, active_groups: u64, split_feature: u32) -> u64 {
        let f = split_feature as usize;
        if f >= self.feature_groups.len() {
            return active_groups;
        }
        let fg = self.feature_groups[f];
        if fg == 0 {
            active_groups
        } else {
            active_groups & fg
        }
    }

    /// Whether `feature` is allowed at a node whose ancestors imply
    /// `active_groups`.  Unconstrained features are always allowed; a
    /// constrained feature is allowed iff some group containing it is still
    /// active.
    #[inline]
    pub(crate) fn feature_allowed(&self, active_groups: u64, feature: u32) -> bool {
        let f = feature as usize;
        if f >= self.feature_groups.len() {
            return true;
        }
        let fg = self.feature_groups[f];
        fg == 0 || (active_groups & fg) != 0
    }
}

/// Clone a [`HistogramBundle`] keeping only the per-feature histograms whose
/// feature index satisfies `is_allowed`.  Used as a per-node filter for
/// interaction constraints — child histograms are still built with the
/// parent's tiles (so the subtraction trick keeps working), but the split
/// search at a constrained node ignores feature columns that aren't allowed
/// on this path.
pub(crate) fn filter_histogram_bundle_by_features(
    bundle: &HistogramBundle,
    is_allowed: impl Fn(u32) -> bool,
) -> HistogramBundle {
    HistogramBundle {
        node_id: bundle.node_id,
        feature_histograms: bundle
            .feature_histograms
            .iter()
            .filter(|fh| is_allowed(fh.feature_index))
            .cloned()
            .collect(),
    }
}

/// Compute per-feature column means from a row-major raw feature matrix.
///
/// Returns `None` when the matrix has no rows, no features, or its `values`
/// vector is empty (metadata-only datasets).  Non-finite cells are skipped per
/// column so a stray NaN/Inf doesn't poison the entire mean.
fn compute_feature_means_from_matrix(
    values: &[f32],
    feature_count: usize,
    row_count: usize,
) -> Option<Vec<f32>> {
    if feature_count == 0 || row_count == 0 || values.len() < row_count * feature_count {
        return None;
    }
    let mut sums = vec![0.0_f64; feature_count];
    let mut counts = vec![0_u64; feature_count];
    for row in 0..row_count {
        let base = row * feature_count;
        for j in 0..feature_count {
            let v = values[base + j];
            if v.is_finite() {
                sums[j] += v as f64;
                counts[j] += 1;
            }
        }
    }
    let means: Vec<f32> = sums
        .iter()
        .zip(counts.iter())
        .map(|(s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
        .collect();
    Some(means)
}

fn validate_neutralization_fit_contract<O: ObjectiveOps>(
    params: &TrainParams,
    dataset: &TrainingDataset,
    objective: &O,
) -> EngineResult<()> {
    validate_neutralization_fit_contract_for_support(
        params,
        dataset,
        objective.supports_pre_target_neutralization(),
    )
}

fn validate_neutralization_fit_contract_for_support(
    params: &TrainParams,
    dataset: &TrainingDataset,
    supports_pre_target_neutralization: bool,
) -> EngineResult<()> {
    let Some(config) = params.neutralization_config else {
        if dataset.factor_exposures.is_some() {
            return Err(EngineError::ContractViolation(
                "factor_exposures were provided but neutralization='none'".to_string(),
            ));
        }
        return Ok(());
    };
    let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
        EngineError::ContractViolation(
            "factor_exposures are required when neutralization is active".to_string(),
        )
    })?;
    if exposures.row_count != dataset.row_count() {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures row_count {} does not match training row_count {}",
            exposures.row_count,
            dataset.row_count()
        )));
    }
    if config.kind == NeutralizationKind::PreTarget && !supports_pre_target_neutralization {
        return Err(EngineError::ContractViolation(
            "neutralization='pre_target' is only supported for GBMRegressor squared-error training"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_warm_start_neutralization_contract(
    params: &TrainParams,
    has_warm_start: bool,
    dataset: &TrainingDataset,
) -> EngineResult<()> {
    if !has_warm_start {
        return Ok(());
    }
    let Some(config) = params.neutralization_config else {
        return Ok(());
    };
    // Per-round and split-penalty modes project the gradient (or its split
    // contribution) against the factor space on every round.  Continuing
    // training without supplying the same exposures would silently change
    // which directions are projected away — almost certainly not what the
    // caller wants, and not equivalent to fitting `N+M` rounds from scratch.
    // We therefore require an exposures matrix; the caller is responsible
    // for passing the same one used for the initial fit (the Python wrapper
    // surfaces this contract).  `pre_target` is idempotent under repeated
    // residualization against the same exposures so it falls under the same
    // requirement.
    match config.kind {
        NeutralizationKind::None => Ok(()),
        NeutralizationKind::PreTarget
        | NeutralizationKind::PerRoundGradient
        | NeutralizationKind::SplitPenalty => {
            if dataset.factor_exposures.is_none() {
                return Err(EngineError::ContractViolation(
                    "neutralized warm-start training requires factor_exposures to be supplied; pass the same matrix used for the initial fit"
                        .to_string(),
                ));
            }
            Ok(())
        }
    }
}

fn prepare_pre_target_training_dataset(
    params: &TrainParams,
    dataset: &TrainingDataset,
) -> EngineResult<Option<TrainingDataset>> {
    let Some(config) = params.neutralization_config else {
        return Ok(None);
    };
    if config.kind != NeutralizationKind::PreTarget {
        return Ok(None);
    }
    let mut owned_dataset = dataset.clone();
    apply_pre_target_neutralization(&mut owned_dataset, config.ridge_lambda)?;
    Ok(Some(owned_dataset))
}

fn gradient_neutralization_config(
    params: &TrainParams,
) -> Option<alloygbm_core::FactorNeutralizationConfig> {
    params.neutralization_config.filter(|config| {
        matches!(
            config.kind,
            NeutralizationKind::PerRoundGradient | NeutralizationKind::SplitPenalty
        )
    })
}

fn factor_split_context_for_node<'a>(
    params: &TrainParams,
    binned_matrix: &'a BinnedMatrix,
    exposures: Option<&'a FactorExposureMatrix>,
    row_indices: &'a [u32],
) -> Option<FactorSplitContext<'a>> {
    let config = params.neutralization_config?;
    if config.kind != NeutralizationKind::SplitPenalty || config.split_penalty == 0.0 {
        return None;
    }
    Some(FactorSplitContext {
        binned_matrix,
        exposures: exposures?,
        row_indices,
        factor_penalty: config.split_penalty,
    })
}

fn validate_gradient_pairs(gradients: &[GradientPair], row_count: usize) -> EngineResult<()> {
    validate_gradient_pair_length(gradients, row_count)?;
    for gradient in gradients {
        if !gradient.grad.is_finite() || !gradient.hess.is_finite() || gradient.hess <= 0.0 {
            return Err(EngineError::ContractViolation(
                "objective produced invalid gradient/hessian values".to_string(),
            ));
        }
    }
    Ok(())
}

const AUTO_SPLIT_L2_NOISY_SMALL_WIDE: f32 = 2.0;

fn split_selection_options_for_training(
    params: &TrainParams,
    policy_mode: Option<TrainingPolicyMode>,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<SplitSelectionOptions> {
    let env_options = split_selection_options_from_env()?;
    let user_set_regularization =
        params.lambda_l2 != 0.0 || params.lambda_l1 != 0.0 || params.min_child_hessian != 0.0;
    let mut options = SplitSelectionOptions {
        l2_lambda: params.lambda_l2,
        l1_alpha: params.lambda_l1,
        min_child_hessian: params.min_child_hessian,
        min_leaf_magnitude: env_options.min_leaf_magnitude,
        dro_config: params
            .dro_config
            .filter(|config| params.leaf_solver == LeafSolverKind::Dro && config.radius > 0.0),
        missing_bin_index: binned_matrix.nan_bin_index as usize,
    };
    if !user_set_regularization {
        options.l2_lambda = env_options.l2_lambda;
        options.l1_alpha = env_options.l1_alpha;
        options.min_child_hessian = env_options.min_child_hessian;
    }
    if !split_l2_env_is_configured()
        && matches!(policy_mode, Some(TrainingPolicyMode::Auto))
        && params.lambda_l2 == 0.0
        && should_apply_auto_split_l2(dataset, binned_matrix)?
    {
        options.l2_lambda = AUTO_SPLIT_L2_NOISY_SMALL_WIDE;
    }
    Ok(options)
}

fn should_apply_auto_split_l2(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<bool> {
    let row_count = dataset.row_count();
    let feature_count = binned_matrix.feature_count.max(1);
    if row_count >= 1_024 || feature_count < 8 {
        return Ok(false);
    }

    let rows_per_feature = row_count as f32 / feature_count as f32;
    if rows_per_feature >= 64.0 {
        return Ok(false);
    }

    let target_variance = target_variance(&dataset.targets, dataset.sample_weights.as_deref())?;
    Ok(target_variance > 4.0)
}

fn validate_gradient_pair_length(gradients: &[GradientPair], row_count: usize) -> EngineResult<()> {
    if gradients.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "objective returned {} gradients for row_count {}",
            gradients.len(),
            row_count
        )));
    }
    Ok(())
}

fn apply_single_categorical_target_encoding(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    spec: &CategoricalTargetEncodingSpec,
) -> EngineResult<(TrainingDataset, BinnedMatrix)> {
    validate_training_alignment(dataset, binned_matrix)?;

    let row_count = dataset.row_count();
    let feature_count = dataset.matrix.feature_count;
    if spec.feature_index >= feature_count {
        return Err(EngineError::ContractViolation(format!(
            "categorical feature index {} is out of bounds for feature_count {}",
            spec.feature_index, feature_count
        )));
    }
    if spec.values.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "categorical values length {} does not match row_count {}",
            spec.values.len(),
            row_count
        )));
    }

    let (_, encoded_values) = fit_transform_target_encoder(
        &spec.config,
        &spec.values,
        &dataset.targets,
        dataset.time_index.as_deref(),
    )
    .map_err(|error| EngineError::ContractViolation(error.to_string()))?;
    let (encoded_bins, encoded_max_bin) = encode_bins_from_encoded_values(&encoded_values)?;

    let mut encoded_dense_values = dataset.matrix.values.clone();
    for (row_index, &encoded_value) in encoded_values.iter().enumerate() {
        let offset = row_index * feature_count + spec.feature_index;
        encoded_dense_values[offset] = encoded_value;
    }

    let encoded_dataset = TrainingDataset {
        matrix: DatasetMatrix::new(row_count, feature_count, encoded_dense_values)?,
        targets: dataset.targets.clone(),
        sample_weights: dataset.sample_weights.clone(),
        time_index: dataset.time_index.clone(),
        group_id: dataset.group_id.clone(),
        factor_exposures: dataset.factor_exposures.clone(),
    };

    let mut encoded_bins_payload = binned_matrix.bins.clone();
    for (row_index, &encoded_bin) in encoded_bins.iter().enumerate() {
        let offset = row_index * feature_count + spec.feature_index;
        encoded_bins_payload[offset] = encoded_bin;
    }
    let encoded_binned_matrix = BinnedMatrix::new(
        row_count,
        feature_count,
        binned_matrix.max_bin.max(encoded_max_bin),
        encoded_bins_payload,
    )?;

    Ok((encoded_dataset, encoded_binned_matrix))
}

fn encode_bins_from_encoded_values(encoded_values: &[f32]) -> EngineResult<(Vec<u8>, u16)> {
    if encoded_values.is_empty() {
        return Err(EngineError::ContractViolation(
            "encoded values cannot be empty".to_string(),
        ));
    }

    for (index, value) in encoded_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::ContractViolation(format!(
                "encoded value at index {index} must be finite"
            )));
        }
    }

    let mut unique_values = encoded_values.to_vec();
    unique_values.sort_by(f32::total_cmp);
    unique_values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    if unique_values.len() > 256 {
        return Err(EngineError::ContractViolation(format!(
            "encoded cardinality {} exceeds supported max 256",
            unique_values.len(),
        )));
    }

    let mut bins = Vec::with_capacity(encoded_values.len());
    for value in encoded_values {
        let position = unique_values
            .binary_search_by(|probe| probe.total_cmp(value))
            .map_err(|_| {
                EngineError::ContractViolation(
                    "encoded value lookup failed during bin mapping".to_string(),
                )
            })?;
        bins.push(position as u8);
    }
    let max_bin = (unique_values.len().saturating_sub(1)) as u16;
    Ok((bins, max_bin))
}

fn validate_training_alignment(
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
) -> EngineResult<()> {
    validate_binned_matrix(binned_matrix)?;
    if dataset.row_count() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "dataset row_count {} does not match binned row_count {}",
            dataset.row_count(),
            binned_matrix.row_count
        )));
    }
    if dataset.matrix.feature_count != binned_matrix.feature_count {
        return Err(EngineError::ContractViolation(format!(
            "dataset feature_count {} does not match binned feature_count {}",
            dataset.matrix.feature_count, binned_matrix.feature_count
        )));
    }
    Ok(())
}

fn validate_partition_cover(row_count: usize, partition: &PartitionResult) -> EngineResult<()> {
    if partition.left_row_indices.is_empty() || partition.right_row_indices.is_empty() {
        return Err(EngineError::ContractViolation(
            "split partition produced empty branch".to_string(),
        ));
    }
    if partition.left_row_indices.len() + partition.right_row_indices.len() != row_count {
        return Err(EngineError::ContractViolation(
            "split partition does not cover all rows".to_string(),
        ));
    }
    Ok(())
}

fn binned_feature_density(binned_matrix: &BinnedMatrix) -> f32 {
    let bin_count = binned_matrix.max_bin as usize + 1;
    let feature_count = binned_matrix.feature_count;
    let total_slots = feature_count.saturating_mul(bin_count);
    if total_slots == 0 {
        return 0.0;
    }

    let mut seen = vec![false; total_slots];
    for row_index in 0..binned_matrix.row_count {
        let row_base = row_index * feature_count;
        for feature_index in 0..feature_count {
            let bin = binned_matrix.row_bin(row_base + feature_index) as usize;
            seen[feature_index * bin_count + bin] = true;
        }
    }
    let occupied = seen.into_iter().filter(|value| *value).count();
    occupied as f32 / total_slots as f32
}

fn target_variance(targets: &[f32], sample_weights: Option<&[f32]>) -> EngineResult<f32> {
    if targets.is_empty() {
        return Err(EngineError::ContractViolation(
            "targets cannot be empty".to_string(),
        ));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    let mut weighted_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for index in 0..targets.len() {
        let weight = sample_weights.map_or(1.0, |weights| weights[index]);
        weighted_sum += targets[index] * weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return Err(EngineError::ContractViolation(
            "sample weight sum must be greater than 0".to_string(),
        ));
    }

    let mean = weighted_sum / weight_sum;
    let mut squared_sum = 0.0_f32;
    for index in 0..targets.len() {
        let weight = sample_weights.map_or(1.0, |weights| weights[index]);
        let centered = targets[index] - mean;
        squared_sum += centered * centered * weight;
    }
    Ok(squared_sum / weight_sum)
}

/// Dispatch best-split finding to either the morph variant or the standard
/// variant based on whether a [`MorphTreeContext`] is supplied. Centralizes
/// the choice so all call sites in `build_tree_level_wise` /
/// `build_tree_leaf_wise` stay consistent.
fn find_best_split_dispatch<B: BackendOps>(
    backend: &B,
    histograms: &HistogramBundle,
    options: SplitSelectionOptions,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<&MorphTreeContext<'_>>,
    factor_context: Option<&FactorSplitContext<'_>>,
) -> EngineResult<Option<SplitCandidate>> {
    if let Some(m) = morph {
        let ctx = m
            .state
            .morph_context(m.iteration, m.total_iterations, m.class_idx);
        backend.best_split_morph_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            &ctx,
            factor_context,
        )
    } else {
        backend.best_split_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
        )
    }
}

/// Build a single tree using level-wise (breadth-first) growth strategy.
///
/// Splits all nodes at depth d before moving to depth d+1.
#[allow(clippy::too_many_arguments)]
fn build_tree_level_wise<B: BackendOps>(
    backend: &B,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    root_row_indices: Vec<u32>,
    round_index: usize,
    feature_tiles: &[FeatureTile],
    split_options: SplitSelectionOptions,
    params: &TrainParams,
    controls: &IterationControls,
    candidate_predictions: &mut [f32],
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<MorphTreeContext<'_>>,
    raw_feature_values: &[f32],
    factor_exposures: Option<&FactorExposureMatrix>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let mut candidate_round_stumps = Vec::new();
    let mut round_rejection_reason = IterationStopReason::NoSplitCandidate;
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    // Interaction-constraint bookkeeping (no-op when empty).  We track the
    // bitset of still-active groups per node so that the split search can
    // skip features that no surviving group allows on this path.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        binned_matrix.feature_count,
    )?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }
    // Maintain each active node's absolute leaf output so child updates
    // can replace parent contribution via deltas (tree semantics).
    // depth is the current tree level (0-indexed); all nodes at this level share the same depth.
    // The Option<LinearLeaf> carries the parent's absolute linear leaf (for weight delta computation).
    let mut active_nodes: Vec<ActiveNodeEntry> =
        vec![(0_u32, root_node.row_indices, root_histograms, 0.0_f32, None)];

    for depth in 0..(params.max_depth as usize) {
        if active_nodes.is_empty() {
            break;
        }

        let mut next_nodes = Vec::new();
        for (local_node_id, node_rows, histograms, parent_leaf_value, parent_linear_leaf) in
            active_nodes
        {
            let node_id = encode_tree_node_id(round_index, local_node_id)?;
            let node = NodeSlice::new(node_id, node_rows)?;
            let factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &node.row_indices,
            );
            // Filter histogram bundle by interaction constraints (no-op when
            // no constraints are active).  Cloning the bundle here is the
            // simplest way to plug filtering in without changing the
            // `BackendOps` trait surface; the clone is `O(allowed_features
            // × bins)` and only runs on constrained fits.
            let node_active = node_active_groups.get(&local_node_id).copied();
            let filtered_histograms_storage;
            let histograms_for_split = match (constraint_index.as_ref(), node_active) {
                (Some(idx), Some(active_groups)) => {
                    filtered_histograms_storage =
                        filter_histogram_bundle_by_features(&histograms, |f| {
                            idx.feature_allowed(active_groups, f)
                        });
                    &filtered_histograms_storage
                }
                _ => &histograms,
            };
            let Some(mut split) = find_best_split_dispatch(
                backend,
                histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                factor_context.as_ref(),
            )?
            else {
                continue;
            };
            if !split.gain.is_finite() || split.gain <= controls.min_split_gain {
                round_rejection_reason = IterationStopReason::GainBelowThreshold;
                continue;
            }

            let (partition, left_stats, right_stats) =
                backend.apply_split_with_stats(binned_matrix, gradients, &node, &split)?;
            if partition.left_row_indices.len() + partition.right_row_indices.len()
                != node.row_indices.len()
            {
                return Err(EngineError::ContractViolation(
                    "split partition does not cover all node rows".to_string(),
                ));
            }
            if partition.left_row_indices.is_empty()
                || partition.right_row_indices.is_empty()
                || partition.left_row_indices.len() < controls.min_rows_per_leaf
                || partition.right_row_indices.len() < controls.min_rows_per_leaf
            {
                round_rejection_reason = IterationStopReason::LeafRowsBelowThreshold;
                continue;
            }

            if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "backend produced non-positive hessian sums".to_string(),
                ));
            }

            let left_grad = leaf_effective_gradient(
                left_stats.grad_sum,
                left_stats.grad_sq_sum,
                left_stats.row_count,
                split_options.l1_alpha,
                split_options.dro_config.as_ref(),
            );
            let right_grad = leaf_effective_gradient(
                right_stats.grad_sum,
                right_stats.grad_sq_sum,
                right_stats.row_count,
                split_options.l1_alpha,
                split_options.dro_config.as_ref(),
            );
            let lr = morph.map_or(params.learning_rate, |m| m.lr);
            let mut raw_left_leaf_value =
                -lr * left_grad / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
            let mut raw_right_leaf_value =
                -lr * right_grad / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

            // Morph leaf modifications: depth penalty + per-round shrinkage.
            // Children land at depth `depth + 1` in the tree.
            let morph_scale = if let Some(m) = morph.as_ref() {
                let child_depth = (depth + 1) as f32;
                let depth_penalty = m.state.config.depth_penalty_base.powf(child_depth / 3.0);
                let iter_shrinkage = 1.0
                    - m.state.config.morph_rate
                        * (m.iteration as f32 / m.total_iterations.max(1) as f32).min(1.0);
                let scale = depth_penalty * iter_shrinkage;
                raw_left_leaf_value *= scale;
                raw_right_leaf_value *= scale;
                scale
            } else {
                1.0
            };

            let left_leaf_absolute = raw_left_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let right_leaf_absolute = raw_right_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let left_leaf_value = left_leaf_absolute - parent_leaf_value;
            let right_leaf_value = right_leaf_absolute - parent_leaf_value;
            if left_leaf_value.abs() < controls.min_abs_leaf_value
                && right_leaf_value.abs() < controls.min_abs_leaf_value
            {
                round_rejection_reason = IterationStopReason::LeafMagnitudeBelowThreshold;
                continue;
            }

            // Monotone constraint enforcement.
            if !params.monotone_constraints.is_empty() {
                let fi = split.feature_index as usize;
                if fi < params.monotone_constraints.len() {
                    let constraint = params.monotone_constraints[fi];
                    if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                    if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                }
            }

            // max_leaves enforcement.
            if let Some(max_leaves) = controls.max_leaves {
                let leaves_after_split = candidate_round_stumps.len() + 2;
                if leaves_after_split > max_leaves {
                    round_rejection_reason = IterationStopReason::MaxLeavesReached;
                    continue;
                }
            }

            // ── Linear leaf path ───────────────────────────────────────────────
            // If leaf_model == Linear, build a LinearHistogramBundle for this node
            // and solve closed-form ridge leaves. Falls back to scalar on any error.
            let linear_leaf_computation_result: Option<LinearLeafQuad> = if params.leaf_model
                == LeafModelKind::Linear
                && !raw_feature_values.is_empty()
                && !split.is_categorical
            {
                let d = binned_matrix.feature_count.min(MAX_PL_REGRESSORS);
                let regressor_features: Vec<u32> = (0..d as u32).collect();
                backend
                    .build_linear_histograms(
                        binned_matrix,
                        gradients,
                        &node,
                        feature_tiles,
                        &regressor_features,
                        raw_feature_values,
                        binned_matrix.row_count,
                        binned_matrix.feature_count,
                    )
                    .ok()
                    .and_then(|lin_hist| {
                        backend.compute_linear_leaf_pair(
                            &lin_hist,
                            split.feature_index,
                            split.threshold_bin as usize,
                            split.default_left,
                            split_options.missing_bin_index,
                            lr,
                            split_options.l2_lambda,
                        )
                    })
                    .map(|(mut ll_abs, mut rl_abs)| {
                        // Apply morph scaling to weights and intercept.
                        ll_abs.intercept *= morph_scale;
                        rl_abs.intercept *= morph_scale;
                        for w in &mut ll_abs.weights {
                            *w *= morph_scale;
                        }
                        for w in &mut rl_abs.weights {
                            *w *= morph_scale;
                        }
                        // Clamp intercepts (absolute values).
                        ll_abs.intercept = ll_abs
                            .intercept
                            .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                        rl_abs.intercept = rl_abs
                            .intercept
                            .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                        // Compute delta versions (parent-relative).
                        let mut ll_delta = ll_abs.clone();
                        let mut rl_delta = rl_abs.clone();
                        ll_delta.intercept -= parent_leaf_value;
                        rl_delta.intercept -= parent_leaf_value;
                        if let Some(ref p) = parent_linear_leaf {
                            let d = p.weights.len().min(ll_delta.weights.len());
                            for i in 0..d {
                                ll_delta.weights[i] -= p.weights[i];
                            }
                            let d = p.weights.len().min(rl_delta.weights.len());
                            for i in 0..d {
                                rl_delta.weights[i] -= p.weights[i];
                            }
                        }
                        (ll_delta, rl_delta, ll_abs, rl_abs)
                    })
            } else {
                None
            };
            // Split into delta pair (for storage/prediction) and absolute pair (for child tracking).
            let (linear_leaf_pair, linear_leaf_abs_pair): LinearLeafPairSplit =
                match linear_leaf_computation_result {
                    Some((ll_d, rl_d, ll_a, rl_a)) => (Some((ll_d, rl_d)), Some((ll_a, rl_a))),
                    None => (None, None),
                };

            // Apply candidate_predictions update.
            if let Some((ref ll, ref rl)) = linear_leaf_pair {
                let fc = binned_matrix.feature_count;
                for &row in &partition.left_row_indices {
                    let r = row as usize;
                    if r < candidate_predictions.len() {
                        candidate_predictions[r] += ll.eval(raw_feature_values, r * fc);
                    }
                }
                for &row in &partition.right_row_indices {
                    let r = row as usize;
                    if r < candidate_predictions.len() {
                        candidate_predictions[r] += rl.eval(raw_feature_values, r * fc);
                    }
                }
            } else {
                apply_partition_leaf_updates(
                    candidate_predictions,
                    &partition,
                    left_leaf_value,
                    right_leaf_value,
                )?;
            }

            split.left_stats = left_stats;
            split.right_stats = right_stats;

            let PartitionResult {
                left_row_indices,
                right_row_indices,
            } = partition;
            if depth + 1 < params.max_depth as usize {
                let left_local_node_id = left_child_node_id(local_node_id)?;
                let right_local_node_id = right_child_node_id(local_node_id)?;
                let left_node_id = encode_tree_node_id(round_index, left_local_node_id)?;
                let right_node_id = encode_tree_node_id(round_index, right_local_node_id)?;

                // Propagate interaction-constraint active groups to children.
                // Splitting on an unconstrained feature leaves the active
                // set unchanged; a constrained feature narrows it.
                if let (Some(idx), Some(active_groups)) = (constraint_index.as_ref(), node_active) {
                    let child_groups = idx.descend(active_groups, split.feature_index);
                    node_active_groups.insert(left_local_node_id, child_groups);
                    node_active_groups.insert(right_local_node_id, child_groups);
                }

                // Determine the parent-leaf values to track for children.
                // When we have linear leaves, the scalar parent value uses the intercept,
                // and we also pass the full absolute linear leaf for weight delta computation.
                let (left_parent_val, right_parent_val) =
                    if let Some((ref ll_a, ref rl_a)) = linear_leaf_abs_pair {
                        (ll_a.intercept, rl_a.intercept)
                    } else {
                        (left_leaf_absolute, right_leaf_absolute)
                    };
                let left_parent_ll = linear_leaf_abs_pair.as_ref().map(|(ll, _)| ll.clone());
                let right_parent_ll = linear_leaf_abs_pair.as_ref().map(|(_, rl)| rl.clone());

                if left_row_indices.len() <= right_row_indices.len() {
                    let left_node = NodeSlice::new(left_node_id, left_row_indices)?;
                    let left_histograms = backend.build_histograms(
                        binned_matrix,
                        gradients,
                        &left_node,
                        feature_tiles,
                    )?;
                    let right_histograms =
                        subtract_histogram_bundle(&histograms, &left_histograms, right_node_id)?;
                    next_nodes.push((
                        left_local_node_id,
                        left_node.row_indices,
                        left_histograms,
                        left_parent_val,
                        left_parent_ll,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_row_indices,
                        right_histograms,
                        right_parent_val,
                        right_parent_ll,
                    ));
                } else {
                    let right_node = NodeSlice::new(right_node_id, right_row_indices)?;
                    let right_histograms = backend.build_histograms(
                        binned_matrix,
                        gradients,
                        &right_node,
                        feature_tiles,
                    )?;
                    let left_histograms =
                        subtract_histogram_bundle(&histograms, &right_histograms, left_node_id)?;
                    next_nodes.push((
                        left_local_node_id,
                        left_row_indices,
                        left_histograms,
                        left_parent_val,
                        left_parent_ll,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_node.row_indices,
                        right_histograms,
                        right_parent_val,
                        right_parent_ll,
                    ));
                }
            }

            let (final_left_leaf, final_right_leaf) = if let Some((ll, rl)) = linear_leaf_pair {
                (LeafValue::Linear(ll), LeafValue::Linear(rl))
            } else {
                (
                    LeafValue::Scalar(left_leaf_value),
                    LeafValue::Scalar(right_leaf_value),
                )
            };
            candidate_round_stumps.push(TrainedStump {
                split,
                left_leaf_value: final_left_leaf,
                right_leaf_value: final_right_leaf,
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            });
        }
        active_nodes = next_nodes;
    }

    if candidate_round_stumps.is_empty() {
        return Ok((Vec::new(), round_rejection_reason));
    }

    Ok((
        candidate_round_stumps,
        IterationStopReason::CompletedRequestedRounds,
    ))
}

/// A pending leaf split for the leaf-wise priority queue.
/// Ordered by gain (highest gain = highest priority).
struct PendingSplit {
    local_node_id: u32,
    row_indices: Vec<u32>,
    split_candidate: SplitCandidate,
    histograms: HistogramBundle,
    parent_leaf_value: f32,
    /// Absolute linear leaf of the parent (used to compute weight deltas for linear-leaf trees).
    parent_linear_leaf: Option<LinearLeaf>,
    depth: usize,
}

// PartialEq uses exact float comparison for the Eq trait bound required by
// BinaryHeap. NaN gains are filtered before insertion; ordering is handled
// by the Ord impl which falls back to Equal for NaN.
impl PartialEq for PendingSplit {
    fn eq(&self, other: &Self) -> bool {
        self.split_candidate.gain == other.split_candidate.gain
    }
}

impl Eq for PendingSplit {}

impl PartialOrd for PendingSplit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingSplit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.split_candidate
            .gain
            .partial_cmp(&other.split_candidate.gain)
            .unwrap_or(Ordering::Equal)
    }
}

/// Build a single tree using leaf-wise (best-first) growth strategy.
///
/// Instead of splitting all nodes at depth d before moving to depth d+1,
/// this always splits the leaf with the highest gain across the entire tree.
/// Stops when `max_leaves` is reached or no valid splits remain.
#[allow(clippy::too_many_arguments)]
fn build_tree_leaf_wise<B: BackendOps>(
    backend: &B,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    root_row_indices: Vec<u32>,
    round_index: usize,
    feature_tiles: &[FeatureTile],
    split_options: SplitSelectionOptions,
    params: &TrainParams,
    controls: &IterationControls,
    candidate_predictions: &mut [f32],
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    morph: Option<MorphTreeContext<'_>>,
    raw_feature_values: &[f32],
    factor_exposures: Option<&FactorExposureMatrix>,
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let max_leaves = controls.max_leaves.unwrap_or(usize::MAX);
    let max_depth = params.max_depth as usize;

    // Build root histograms and find best split.
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    // Interaction-constraint bookkeeping (no-op when empty).  See the
    // matching block in `build_tree_level_wise` for the design rationale —
    // we filter histograms per node at split-search time so constrained
    // features can't appear on a path that already broke into a sibling
    // group.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        binned_matrix.feature_count,
    )?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }
    let root_factor_context = factor_split_context_for_node(
        params,
        binned_matrix,
        factor_exposures,
        &root_node.row_indices,
    );
    let root_filtered_storage;
    let root_histograms_for_split = match (
        constraint_index.as_ref(),
        node_active_groups.get(&0).copied(),
    ) {
        (Some(idx), Some(ag)) => {
            root_filtered_storage = filter_histogram_bundle_by_features(&root_histograms, |f| {
                idx.feature_allowed(ag, f)
            });
            &root_filtered_storage
        }
        _ => &root_histograms,
    };
    let root_split = find_best_split_dispatch(
        backend,
        root_histograms_for_split,
        split_options,
        feature_weights,
        categorical_features,
        morph.as_ref(),
        root_factor_context.as_ref(),
    )?;

    let Some(root_split) = root_split else {
        return Ok((Vec::new(), IterationStopReason::NoSplitCandidate));
    };
    if !root_split.gain.is_finite() || root_split.gain <= controls.min_split_gain {
        return Ok((Vec::new(), IterationStopReason::GainBelowThreshold));
    }

    let mut queue = BinaryHeap::new();
    queue.push(PendingSplit {
        local_node_id: 0,
        row_indices: root_node.row_indices,
        split_candidate: root_split,
        histograms: root_histograms,
        parent_leaf_value: 0.0,
        parent_linear_leaf: None,
        depth: 0,
    });

    // Start with 1 leaf (the root). Each split adds 1 net leaf (splits one into two).
    let mut leaves_used = 1_usize;
    let mut stumps = Vec::new();
    let mut last_rejection = IterationStopReason::NoSplitCandidate;

    while let Some(pending) = queue.pop() {
        // Check max_leaves: splitting adds 1 net leaf.
        if leaves_used + 1 > max_leaves {
            last_rejection = IterationStopReason::MaxLeavesReached;
            break;
        }

        // Check max_depth constraint.
        if pending.depth >= max_depth {
            last_rejection = IterationStopReason::DepthBudgetReached;
            continue;
        }

        let local_node_id = pending.local_node_id;
        let node_id = encode_tree_node_id(round_index, local_node_id)?;
        let node = NodeSlice::new(node_id, pending.row_indices)?;
        let split = pending.split_candidate;

        // Apply the split: partition rows and get stats.
        let (partition, left_stats, right_stats) =
            backend.apply_split_with_stats(binned_matrix, gradients, &node, &split)?;

        if partition.left_row_indices.len() + partition.right_row_indices.len()
            != node.row_indices.len()
        {
            return Err(EngineError::ContractViolation(
                "split partition does not cover all node rows".to_string(),
            ));
        }
        if partition.left_row_indices.is_empty()
            || partition.right_row_indices.is_empty()
            || partition.left_row_indices.len() < controls.min_rows_per_leaf
            || partition.right_row_indices.len() < controls.min_rows_per_leaf
        {
            last_rejection = IterationStopReason::LeafRowsBelowThreshold;
            continue;
        }

        if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
            return Err(EngineError::ContractViolation(
                "backend produced non-positive hessian sums".to_string(),
            ));
        }

        // Compute leaf values.
        let left_grad = leaf_effective_gradient(
            left_stats.grad_sum,
            left_stats.grad_sq_sum,
            left_stats.row_count,
            split_options.l1_alpha,
            split_options.dro_config.as_ref(),
        );
        let right_grad = leaf_effective_gradient(
            right_stats.grad_sum,
            right_stats.grad_sq_sum,
            right_stats.row_count,
            split_options.l1_alpha,
            split_options.dro_config.as_ref(),
        );
        let lr = morph.map_or(params.learning_rate, |m| m.lr);
        let mut raw_left_leaf_value =
            -lr * left_grad / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
        let mut raw_right_leaf_value =
            -lr * right_grad / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

        // Morph leaf modifications: depth penalty + per-round shrinkage.
        // Children of `pending` land at `pending.depth + 1` in the tree.
        let morph_scale = if let Some(m) = morph.as_ref() {
            let child_depth = (pending.depth + 1) as f32;
            let depth_penalty = m.state.config.depth_penalty_base.powf(child_depth / 3.0);
            let iter_shrinkage = 1.0
                - m.state.config.morph_rate
                    * (m.iteration as f32 / m.total_iterations.max(1) as f32).min(1.0);
            let scale = depth_penalty * iter_shrinkage;
            raw_left_leaf_value *= scale;
            raw_right_leaf_value *= scale;
            scale
        } else {
            1.0
        };

        let left_leaf_absolute =
            raw_left_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let right_leaf_absolute =
            raw_right_leaf_value.clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
        let left_leaf_value = left_leaf_absolute - pending.parent_leaf_value;
        let right_leaf_value = right_leaf_absolute - pending.parent_leaf_value;

        if left_leaf_value.abs() < controls.min_abs_leaf_value
            && right_leaf_value.abs() < controls.min_abs_leaf_value
        {
            last_rejection = IterationStopReason::LeafMagnitudeBelowThreshold;
            continue;
        }

        // Monotone constraint enforcement.
        if !params.monotone_constraints.is_empty() {
            let fi = split.feature_index as usize;
            if fi < params.monotone_constraints.len() {
                let constraint = params.monotone_constraints[fi];
                if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
                    last_rejection = IterationStopReason::MonotoneConstraintViolation;
                    continue;
                }
                if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
                    last_rejection = IterationStopReason::MonotoneConstraintViolation;
                    continue;
                }
            }
        }

        // ── Linear leaf path ───────────────────────────────────────────────────
        let linear_leaf_computation_result: Option<LinearLeafQuad> = if params.leaf_model
            == LeafModelKind::Linear
            && !raw_feature_values.is_empty()
            && !split.is_categorical
        {
            let d = binned_matrix.feature_count.min(MAX_PL_REGRESSORS);
            let regressor_features: Vec<u32> = (0..d as u32).collect();
            backend
                .build_linear_histograms(
                    binned_matrix,
                    gradients,
                    &node,
                    feature_tiles,
                    &regressor_features,
                    raw_feature_values,
                    binned_matrix.row_count,
                    binned_matrix.feature_count,
                )
                .ok()
                .and_then(|lin_hist| {
                    backend.compute_linear_leaf_pair(
                        &lin_hist,
                        split.feature_index,
                        split.threshold_bin as usize,
                        split.default_left,
                        split_options.missing_bin_index,
                        lr,
                        split_options.l2_lambda,
                    )
                })
                .map(|(mut ll_abs, mut rl_abs)| {
                    ll_abs.intercept *= morph_scale;
                    rl_abs.intercept *= morph_scale;
                    for w in &mut ll_abs.weights {
                        *w *= morph_scale;
                    }
                    for w in &mut rl_abs.weights {
                        *w *= morph_scale;
                    }
                    ll_abs.intercept = ll_abs
                        .intercept
                        .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                    rl_abs.intercept = rl_abs
                        .intercept
                        .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
                    // Compute delta versions (parent-relative).
                    let mut ll_delta = ll_abs.clone();
                    let mut rl_delta = rl_abs.clone();
                    ll_delta.intercept -= pending.parent_leaf_value;
                    rl_delta.intercept -= pending.parent_leaf_value;
                    if let Some(ref p) = pending.parent_linear_leaf {
                        let d = p.weights.len().min(ll_delta.weights.len());
                        for i in 0..d {
                            ll_delta.weights[i] -= p.weights[i];
                        }
                        let d = p.weights.len().min(rl_delta.weights.len());
                        for i in 0..d {
                            rl_delta.weights[i] -= p.weights[i];
                        }
                    }
                    (ll_delta, rl_delta, ll_abs, rl_abs)
                })
        } else {
            None
        };
        // Split into delta pair (for storage/prediction) and absolute pair (for child tracking).
        let (linear_leaf_pair, linear_leaf_abs_pair): LinearLeafPairSplit =
            match linear_leaf_computation_result {
                Some((ll_d, rl_d, ll_a, rl_a)) => (Some((ll_d, rl_d)), Some((ll_a, rl_a))),
                None => (None, None),
            };

        // Commit the split: update predictions and record stump.
        if let Some((ref ll, ref rl)) = linear_leaf_pair {
            let fc = binned_matrix.feature_count;
            for &row in &partition.left_row_indices {
                let r = row as usize;
                if r < candidate_predictions.len() {
                    candidate_predictions[r] += ll.eval(raw_feature_values, r * fc);
                }
            }
            for &row in &partition.right_row_indices {
                let r = row as usize;
                if r < candidate_predictions.len() {
                    candidate_predictions[r] += rl.eval(raw_feature_values, r * fc);
                }
            }
        } else {
            apply_partition_leaf_updates(
                candidate_predictions,
                &partition,
                left_leaf_value,
                right_leaf_value,
            )?;
        }

        let mut committed_split = split;
        committed_split.left_stats = left_stats;
        committed_split.right_stats = right_stats;

        let (final_left_leaf, final_right_leaf) = if let Some((ll, rl)) = linear_leaf_pair {
            (LeafValue::Linear(ll), LeafValue::Linear(rl))
        } else {
            (
                LeafValue::Scalar(left_leaf_value),
                LeafValue::Scalar(right_leaf_value),
            )
        };
        stumps.push(TrainedStump {
            split: committed_split,
            left_leaf_value: final_left_leaf,
            right_leaf_value: final_right_leaf,
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        });
        leaves_used += 1;

        // Enqueue children if within depth budget.
        let child_depth = pending.depth + 1;
        if child_depth < max_depth {
            let left_local = left_child_node_id(local_node_id)?;
            let right_local = right_child_node_id(local_node_id)?;
            let left_node_id = encode_tree_node_id(round_index, left_local)?;
            let right_node_id = encode_tree_node_id(round_index, right_local)?;

            // Subtraction trick: build smaller child, subtract from parent for larger.
            // Determine parent leaf values and linear leaves for each child.
            let (left_parent_val, right_parent_val) =
                if let Some((ref ll_a, ref rl_a)) = linear_leaf_abs_pair {
                    (ll_a.intercept, rl_a.intercept)
                } else {
                    (left_leaf_absolute, right_leaf_absolute)
                };
            let left_parent_ll = linear_leaf_abs_pair.as_ref().map(|(ll, _)| ll.clone());
            let right_parent_ll = linear_leaf_abs_pair.as_ref().map(|(_, rl)| rl.clone());

            let (
                smaller_indices,
                larger_indices,
                smaller_node_id,
                larger_node_id,
                smaller_local,
                larger_local,
                smaller_parent_val,
                larger_parent_val,
                smaller_parent_ll,
                larger_parent_ll,
            ) = if partition.left_row_indices.len() <= partition.right_row_indices.len() {
                (
                    partition.left_row_indices,
                    partition.right_row_indices,
                    left_node_id,
                    right_node_id,
                    left_local,
                    right_local,
                    left_parent_val,
                    right_parent_val,
                    left_parent_ll,
                    right_parent_ll,
                )
            } else {
                (
                    partition.right_row_indices,
                    partition.left_row_indices,
                    right_node_id,
                    left_node_id,
                    right_local,
                    left_local,
                    right_parent_val,
                    left_parent_val,
                    right_parent_ll,
                    left_parent_ll,
                )
            };

            let smaller_node = NodeSlice::new(smaller_node_id, smaller_indices)?;
            let smaller_histograms =
                backend.build_histograms(binned_matrix, gradients, &smaller_node, feature_tiles)?;
            let larger_histograms = subtract_histogram_bundle(
                &pending.histograms,
                &smaller_histograms,
                larger_node_id,
            )?;

            // Propagate interaction-constraint active groups to both
            // children of the just-applied split.  Both children inherit the
            // same descended bitset because the split feature is shared.
            // (`split` itself was moved into `committed_split` above; we
            // read the feature index off the just-pushed stump instead.)
            let split_feature_for_descend =
                stumps.last().map(|s| s.split.feature_index).unwrap_or(0);
            let child_active_groups: Option<u64> = match (
                constraint_index.as_ref(),
                node_active_groups.get(&local_node_id).copied(),
            ) {
                (Some(idx), Some(ag)) => {
                    let descended = idx.descend(ag, split_feature_for_descend);
                    node_active_groups.insert(smaller_local, descended);
                    node_active_groups.insert(larger_local, descended);
                    Some(descended)
                }
                _ => None,
            };

            // Find best split for each child and enqueue if valid.
            let smaller_factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &smaller_node.row_indices,
            );
            let smaller_filtered_storage;
            let smaller_histograms_for_split =
                match (constraint_index.as_ref(), child_active_groups) {
                    (Some(idx), Some(ag)) => {
                        smaller_filtered_storage =
                            filter_histogram_bundle_by_features(&smaller_histograms, |f| {
                                idx.feature_allowed(ag, f)
                            });
                        &smaller_filtered_storage
                    }
                    _ => &smaller_histograms,
                };
            if let Some(child_split) = find_best_split_dispatch(
                backend,
                smaller_histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                smaller_factor_context.as_ref(),
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: smaller_local,
                    row_indices: smaller_node.row_indices,
                    split_candidate: child_split,
                    histograms: smaller_histograms,
                    parent_leaf_value: smaller_parent_val,
                    parent_linear_leaf: smaller_parent_ll,
                    depth: child_depth,
                });
            }

            let larger_factor_context = factor_split_context_for_node(
                params,
                binned_matrix,
                factor_exposures,
                &larger_indices,
            );
            let larger_filtered_storage;
            let larger_histograms_for_split = match (constraint_index.as_ref(), child_active_groups)
            {
                (Some(idx), Some(ag)) => {
                    larger_filtered_storage =
                        filter_histogram_bundle_by_features(&larger_histograms, |f| {
                            idx.feature_allowed(ag, f)
                        });
                    &larger_filtered_storage
                }
                _ => &larger_histograms,
            };
            if let Some(child_split) = find_best_split_dispatch(
                backend,
                larger_histograms_for_split,
                split_options,
                feature_weights,
                categorical_features,
                morph.as_ref(),
                larger_factor_context.as_ref(),
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: larger_local,
                    row_indices: larger_indices,
                    split_candidate: child_split,
                    histograms: larger_histograms,
                    parent_leaf_value: larger_parent_val,
                    parent_linear_leaf: larger_parent_ll,
                    depth: child_depth,
                });
            }
        }
    }

    if stumps.is_empty() {
        return Ok((Vec::new(), last_rejection));
    }

    Ok((stumps, IterationStopReason::CompletedRequestedRounds))
}

/// Subtract child histogram from parent, writing into an existing buffer.
///
/// This avoids allocating a new `HistogramBundle` by reusing `dest`.
/// `dest` must have the same feature count and bin counts as `parent`.
fn subtract_histogram_bundle_into(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
    dest: &mut HistogramBundle,
) -> EngineResult<()> {
    if parent.feature_histograms.len() != child.feature_histograms.len() {
        return Err(EngineError::ContractViolation(format!(
            "parent histogram feature count {} does not match child histogram feature count {}",
            parent.feature_histograms.len(),
            child.feature_histograms.len()
        )));
    }
    dest.node_id = node_id;
    for ((dest_fh, parent_fh), child_fh) in dest
        .feature_histograms
        .iter_mut()
        .zip(&parent.feature_histograms)
        .zip(&child.feature_histograms)
    {
        dest_fh.feature_index = parent_fh.feature_index;
        for ((dest_bin, parent_bin), child_bin) in dest_fh
            .bins
            .iter_mut()
            .zip(&parent_fh.bins)
            .zip(&child_fh.bins)
        {
            dest_bin.grad_sum = parent_bin.grad_sum - child_bin.grad_sum;
            dest_bin.hess_sum = parent_bin.hess_sum - child_bin.hess_sum;
            dest_bin.grad_sq_sum = parent_bin.grad_sq_sum - child_bin.grad_sq_sum;
            dest_bin.count = parent_bin.count - child_bin.count;
        }
    }
    Ok(())
}

fn subtract_histogram_bundle(
    parent: &HistogramBundle,
    child: &HistogramBundle,
    node_id: u32,
) -> EngineResult<HistogramBundle> {
    // Pre-allocate a dest with the same structure, then delegate to the in-place variant.
    let feature_indices: Vec<u32> = parent
        .feature_histograms
        .iter()
        .map(|fh| fh.feature_index)
        .collect();
    let bin_count = parent
        .feature_histograms
        .first()
        .map_or(0, |fh| fh.bins.len());
    let mut dest = HistogramBundle::new_zeroed(&feature_indices, bin_count);
    subtract_histogram_bundle_into(parent, child, node_id, &mut dest)?;
    Ok(dest)
}

fn validate_iteration_controls(controls: IterationControls) -> EngineResult<()> {
    if controls.rounds == 0 {
        return Err(EngineError::InvalidConfig(
            "rounds must be greater than 0".to_string(),
        ));
    }
    if !controls.min_split_gain.is_finite() || controls.min_split_gain < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_split_gain must be finite and >= 0".to_string(),
        ));
    }
    if controls.min_rows_per_leaf == 0 {
        return Err(EngineError::InvalidConfig(
            "min_rows_per_leaf must be greater than 0".to_string(),
        ));
    }
    if !controls.min_abs_leaf_value.is_finite() || controls.min_abs_leaf_value < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_abs_leaf_value must be finite and >= 0".to_string(),
        ));
    }
    if !controls.max_abs_leaf_value.is_finite() || controls.max_abs_leaf_value <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "max_abs_leaf_value must be finite and > 0".to_string(),
        ));
    }
    if controls.min_abs_leaf_value > controls.max_abs_leaf_value {
        return Err(EngineError::InvalidConfig(
            "min_abs_leaf_value cannot exceed max_abs_leaf_value".to_string(),
        ));
    }
    if !controls.min_loss_improvement.is_finite() || controls.min_loss_improvement < 0.0 {
        return Err(EngineError::InvalidConfig(
            "min_loss_improvement must be finite and >= 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&controls.row_subsample) || controls.row_subsample == 0.0 {
        return Err(EngineError::InvalidConfig(
            "row_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&controls.col_subsample) || controls.col_subsample == 0.0 {
        return Err(EngineError::InvalidConfig(
            "col_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }
    if let Some(early_stopping_rounds) = controls.early_stopping_rounds
        && early_stopping_rounds == 0
    {
        return Err(EngineError::InvalidConfig(
            "early_stopping_rounds must be greater than 0".to_string(),
        ));
    }
    if !controls.min_validation_improvement.is_finite() || controls.min_validation_improvement < 0.0
    {
        return Err(EngineError::InvalidConfig(
            "min_validation_improvement must be finite and >= 0".to_string(),
        ));
    }
    Ok(())
}

fn sampling_seed_base(seed: u64, deterministic: bool) -> u64 {
    if deterministic {
        return seed;
    }
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    seed ^ now_nanos
}

pub(crate) fn mixed_hash(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn sampled_count(total_count: usize, subsample: f32) -> usize {
    ((total_count as f32) * subsample)
        .ceil()
        .max(1.0)
        .min(total_count as f32) as usize
}

fn sampled_indices(
    total_count: usize,
    subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> Vec<usize> {
    if total_count == 0 {
        return Vec::new();
    }
    let keep_count = sampled_count(total_count, subsample);
    if keep_count >= total_count {
        return (0..total_count).collect();
    }

    let round_seed = mixed_hash(seed_base ^ round_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut scored = (0..total_count)
        .map(|index| {
            let index_seed = (index as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D);
            let hash = mixed_hash(round_seed ^ index_seed);
            (index, hash)
        })
        .collect::<Vec<_>>();
    scored.select_nth_unstable_by(keep_count, |lhs, rhs| {
        lhs.1.cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0))
    });

    let mut selected = scored[..keep_count]
        .iter()
        .map(|(index, _)| *index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

fn sampled_row_indices(
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> Vec<u32> {
    sampled_indices(row_count, row_subsample, seed_base, round_index)
        .into_iter()
        .map(|row_index| row_index as u32)
        .collect()
}

/// Per-round row-selection dispatcher.  Dispatches on
/// `TrainParams::boosting_mode`:
///
/// * `BoostingMode::Standard` — uniform subsampling under
///   `row_subsample`.  Byte-identical to v0.7.5.
/// * `BoostingMode::Goss` — gradient-based one-side sampling.
///   `gradients` MUST already be the post-projection gradient buffer
///   for this round; the function mutates it in place to apply the
///   `(n - top_n) / other_n` amplification on the sampled-low rows
///   (top-by-magnitude rows are *not* amplified — they appear with
///   their original gradient/hessian, exactly as in the reference
///   LightGBM implementation).  We use realized counts rather than
///   the configured `(1 - top_rate) / other_rate` symbolic form so
///   that `ceil()` rounding and the `other_n <= n - top_n` cap don't
///   bias the unbiasedness contract at small `n` (see
///   `goss_sample_indices` for details).
/// * `BoostingMode::Dart` — row-selection itself is uniform (same as
///   Standard); the dropout + normalize cycle that makes DART distinct
///   is applied separately in the iteration loop
///   (`fit_iterations_with_optional_validation_summary`) before
///   gradient computation.  See `crates/engine/src/dart.rs`.
///
/// Returns the sorted set of row indices used as `root_row_indices`
/// for tree construction this round.
fn select_row_indices_for_round(
    boosting_mode: BoostingMode,
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
    gradients: &mut [GradientPair],
) -> Vec<u32> {
    match boosting_mode {
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            // Score rows by |gradient|.  Hessian could also be folded
            // in (e.g. `|grad| / sqrt(hess)`) but the LightGBM
            // reference uses |grad| only.
            let magnitudes: Vec<f32> = gradients.iter().map(|g| g.grad.abs()).collect();
            let (top, other, amplification) =
                goss_sample_indices(&magnitudes, top_rate, other_rate, seed_base, round_index);
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    gradients[idx].grad *= amplification;
                    gradients[idx].hess *= amplification;
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            merged
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            sampled_row_indices(row_count, row_subsample, seed_base, round_index)
        }
    }
}

/// Multiclass variant of [`select_row_indices_for_round`].
///
/// For multiclass GOSS the per-row score is the L1 norm of the per-class
/// gradient vector: `s_i = sum_k |g_{i,k}|` (LightGBM convention).  A single
/// row mask is shared across all K class gradient buffers, and the
/// amplification factor is applied identically to every class's gradient and
/// hessian.
///
/// `class_gradient_buffers[k]` is the gradient/hessian buffer for class `k`;
/// every buffer must have length `row_count`.  Mutated in place to apply
/// amplification when GOSS is active.
fn select_row_indices_for_round_multiclass(
    boosting_mode: BoostingMode,
    row_count: usize,
    row_subsample: f32,
    seed_base: u64,
    round_index: u64,
    class_gradient_buffers: &mut [Vec<GradientPair>],
) -> Vec<u32> {
    match boosting_mode {
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            let k = class_gradient_buffers.len();
            assert!(
                k > 0,
                "multiclass GOSS requires at least one class gradient buffer"
            );
            debug_assert!(
                class_gradient_buffers
                    .iter()
                    .all(|buf| buf.len() == row_count),
                "every class gradient buffer must have length row_count"
            );
            let magnitudes: Vec<f32> = (0..row_count)
                .map(|i| {
                    class_gradient_buffers
                        .iter()
                        .take(k)
                        .map(|buf| buf[i].grad.abs())
                        .sum::<f32>()
                })
                .collect();
            let (top, other, amplification) =
                goss_sample_indices(&magnitudes, top_rate, other_rate, seed_base, round_index);
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    for class_buf in class_gradient_buffers.iter_mut().take(k) {
                        let pair = &mut class_buf[idx];
                        pair.grad *= amplification;
                        pair.hess *= amplification;
                    }
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            merged
        }
        BoostingMode::Standard | BoostingMode::Dart { .. } => {
            sampled_row_indices(row_count, row_subsample, seed_base, round_index)
        }
    }
}

/// Gradient-based One-Side Sampling (GOSS, from LightGBM).
///
/// Strategy: keep the top `top_rate` fraction of rows by
/// `|gradient_magnitude|`, then uniformly sample `other_rate` fraction
/// from the rest.  Sampled-low-gradient rows are *amplified* by
/// `(n - top_n) / other_n` at the gradient-accumulation stage so the
/// histogram statistics remain an unbiased estimator of the full-data
/// gradient sums.  We use realized counts rather than the configured
/// `(1 - top_rate) / other_rate` symbolic form because `ceil()`
/// rounding (and the `other_n <= n - top_n` cap) shifts the realized
/// fractions away from the configured ones at small `n` — the rate
/// form would double the sampled-low contribution in those edge
/// cases.  For large `n` the two forms agree (since `top_n ≈ top_rate
/// · n` and `other_n ≈ other_rate · n`).
///
/// Returns `(sampled_row_indices, amplification, top_kept_count)`:
///
/// * `sampled_row_indices` — sorted ascending, includes both kept-top
///   and sampled-low rows.  Suitable to feed
///   `NodeSlice::row_indices`.
/// * `amplification` — multiplier the caller applies to gradients and
///   hessians on the sampled-low rows (not on the kept-top rows!) to
///   preserve unbiasedness.  Always `>= 1.0`; equals `1.0` when
///   `other_rate == 0`.
/// * `top_kept_count` — number of leading elements in
///   `sampled_row_indices` (after sorting) that are kept-top rows.
///   *Not* used directly — instead, the caller marks each row by
///   checking membership in a separate hash set.  Returned for
///   convenience and unit-test sanity checks.
pub(crate) fn goss_sample_indices(
    gradient_magnitudes: &[f32],
    top_rate: f32,
    other_rate: f32,
    seed_base: u64,
    round_index: u64,
) -> (Vec<u32>, Vec<u32>, f32) {
    let n = gradient_magnitudes.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 1.0);
    }
    let top_n = ((top_rate * n as f32).ceil() as usize).max(1).min(n);
    let other_n = ((other_rate * n as f32).ceil() as usize).min(n - top_n);

    // Rank by |gradient| descending using select_nth_unstable_by.
    let mut indexed: Vec<(u32, f32)> = gradient_magnitudes
        .iter()
        .enumerate()
        .map(|(i, &g)| (i as u32, g.abs()))
        .collect();
    if top_n < n {
        // After this call indexed[..top_n] contains the top_n rows by
        // |gradient| (in arbitrary order); indexed[top_n..] contains
        // the rest.
        indexed.select_nth_unstable_by(top_n - 1, |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
    }
    let mut top_indices: Vec<u32> = indexed[..top_n].iter().map(|(i, _)| *i).collect();

    let mut other_indices: Vec<u32> = if other_n > 0 && top_n < n {
        let mut rest_scored: Vec<(u32, u64)> = indexed[top_n..]
            .iter()
            .map(|(i, _)| {
                let seed = mixed_hash(
                    seed_base
                        ^ round_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (*i as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D),
                );
                (*i, seed)
            })
            .collect();
        rest_scored.select_nth_unstable_by(other_n - 1, |a, b| {
            a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        });
        rest_scored[..other_n].iter().map(|(i, _)| *i).collect()
    } else {
        Vec::new()
    };

    // Amplification uses **realized** counts (`(n - top_n) / other_n`)
    // rather than the configured rates (`(1 - top_rate) / other_rate`).
    // When `ceil()` rounding or the `other_n <= n - top_n` cap shifts the
    // realized fractions away from the configured ones — common at small
    // `n` — the rate-based form double-counts (or under-counts) the
    // sampled-low rows.  Example: `n=5`, `top_rate=0.2`, `other_rate=0.1`
    // gives `top_n=1`, `other_n=1`.  The unbiased multiplier for the
    // remaining pool of 4 rows sampled at size 1 is `4 / 1 = 4`, not
    // `(1 - 0.2) / 0.1 = 8`.  See `goss_amplification_uses_realized_counts`
    // for the contract test.
    let amplification = if other_n > 0 && top_n < n {
        (n - top_n) as f32 / other_n as f32
    } else {
        1.0
    };

    top_indices.sort_unstable();
    other_indices.sort_unstable();
    (top_indices, other_indices, amplification)
}

/// Maximum features per tile. Keeps the histogram arena small enough to fit in
/// L2 cache (64 features × 256 bins × 12 bytes ≈ 192 KB) and creates enough
/// tiles for rayon to parallelize across cores.
const MAX_TILE_FEATURE_WIDTH: usize = 64;

/// Compute a tile size that keeps each thread busy with enough work but
/// produces enough tiles to amortize parallelism overhead. Aim for roughly
/// 2 tiles per thread so straggling threads can steal work. Falls back to
/// `MAX_TILE_FEATURE_WIDTH` for low-feature workloads.
fn compute_optimal_tile_size(feature_count: usize, n_threads: usize) -> usize {
    if n_threads <= 1 || feature_count <= 16 {
        return feature_count.clamp(1, MAX_TILE_FEATURE_WIDTH);
    }
    let target_tiles = n_threads.saturating_mul(2);
    let raw_tile = feature_count.div_ceil(target_tiles);
    raw_tile.clamp(16, MAX_TILE_FEATURE_WIDTH)
}

fn feature_tiles_from_sorted_indices(indices: &[usize]) -> EngineResult<Vec<FeatureTile>> {
    if indices.is_empty() {
        return Err(EngineError::ContractViolation(
            "feature subsampling produced no feature indices".to_string(),
        ));
    }

    let n_threads = rayon::current_num_threads();
    let tile_width = compute_optimal_tile_size(indices.len(), n_threads);

    let mut tiles = Vec::new();
    let mut run_start = indices[0];
    let mut previous = indices[0];
    for &current in indices.iter().skip(1) {
        if current == previous + 1 && (current - run_start) < tile_width {
            previous = current;
            continue;
        }
        tiles.push(FeatureTile::new(run_start as u32, (previous + 1) as u32)?);
        run_start = current;
        previous = current;
    }
    tiles.push(FeatureTile::new(run_start as u32, (previous + 1) as u32)?);
    Ok(tiles)
}

fn sampled_feature_tiles(
    feature_count: usize,
    col_subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> EngineResult<(Vec<FeatureTile>, usize)> {
    let selected = sampled_indices(feature_count, col_subsample, seed_base, round_index);
    let coverage_count = selected.len();
    let tiles = feature_tiles_from_sorted_indices(&selected)?;
    Ok((tiles, coverage_count))
}

fn apply_partition_leaf_updates(
    predictions: &mut [f32],
    partition: &PartitionResult,
    left_leaf_value: f32,
    right_leaf_value: f32,
) -> EngineResult<()> {
    let prediction_len = predictions.len();
    for &row_index in &partition.left_row_indices {
        let row_index = row_index as usize;
        if row_index >= prediction_len {
            return Err(EngineError::ContractViolation(format!(
                "left partition row index {row_index} is out of bounds for predictions length {prediction_len}"
            )));
        }
        predictions[row_index] += left_leaf_value;
    }
    for &row_index in &partition.right_row_indices {
        let row_index = row_index as usize;
        if row_index >= prediction_len {
            return Err(EngineError::ContractViolation(format!(
                "right partition row index {row_index} is out of bounds for predictions length {prediction_len}"
            )));
        }
        predictions[row_index] += right_leaf_value;
    }
    Ok(())
}

/// DART helper: apply one tree's stumps to `predictions` with a
/// multiplicative `factor`. `factor = 1.0` reproduces a unit-weight
/// tree walk; `factor = -w` is used to subtract a dropped tree's
/// previous contribution; `factor = new_w` is used to re-add a
/// rescaled tree post-normalization.
///
/// Routing uses the binned-matrix view but with the same split
/// semantics as the predictor: missing bin (`MISSING_BIN_U8`) routes
/// through `default_left`; native categorical splits consult the
/// stump's `categorical_bitset`; otherwise the standard
/// `bin <= threshold_bin` comparison applies.  Using only
/// `bin <= threshold_bin` (the legacy `apply_round_stumps_tree_walk`
/// shortcut) would silently disagree with the predictor on rows with
/// learned-missing-direction or native categorical features, which
/// matters for DART because the dropout subtract / re-add must
/// reproduce the predictor's per-tree contribution exactly.
///
/// `raw_features = Some((raw, fc))` is used only for PL-leaf
/// evaluation (`LeafValue::Linear`).  Constant-leaf models can pass
/// `None` (or an empty raw slice) and the leaf will be evaluated as
/// the scalar intercept.
///
/// All stumps in `stumps` are assumed to belong to the same tree (i.e.,
/// share the same encoded `tree_id` in their `node_id`). The caller is
/// responsible for slicing `stumps` correctly.
fn apply_weighted_round_to_predictions(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
    factor: f32,
) -> EngineResult<()> {
    if stumps.is_empty() || factor == 0.0 {
        return Ok(());
    }
    let mut stump_by_local: HashMap<u32, &TrainedStump> = HashMap::with_capacity(stumps.len());
    for stump in stumps {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        stump_by_local.insert(local_id, stump);
    }
    let feature_count = binned_matrix.feature_count;
    let missing_bin = u16::from(MISSING_BIN_U8);

    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        let row_base = row_index * feature_count;
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break;
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.row_bin(row_base + feature_index);
            let went_left = if bin == missing_bin {
                // Missing-value routing — predictor's `is_nan` short-circuit
                // produces the same `default_left` outcome.
                stump.split.default_left
            } else if stump.split.is_categorical {
                // Native categorical split: consult the bitset (same
                // routing as `predictor_went_left`).
                stump
                    .split
                    .categorical_bitset
                    .as_ref()
                    .map_or(stump.split.default_left, |bs| {
                        let cat_id = bin;
                        let byte_idx = (cat_id / 8) as usize;
                        let bit_idx = (cat_id % 8) as usize;
                        byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
                    })
            } else {
                bin <= stump.split.threshold_bin
            };
            let leaf = if went_left {
                if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.left_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.left_leaf_value.as_scalar()
                }
            } else if let Some((raw, fc)) = raw_features
                && !raw.is_empty()
            {
                let row_offset = row_index * fc;
                stump.right_leaf_value.eval_row(&raw[row_offset..])
            } else {
                stump.right_leaf_value.as_scalar()
            };
            *prediction += factor * leaf;
            local_id = if went_left {
                local_id * 2 + 1
            } else {
                local_id * 2 + 2
            };
        }
    }
    Ok(())
}

fn apply_round_stumps_tree_walk(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    // Build a lookup from local_node_id to stump for tree traversal
    let mut stump_by_local: HashMap<u32, &TrainedStump> = HashMap::with_capacity(stumps.len());
    for stump in stumps {
        let (_, local_id) = decode_tree_node_id(stump.split.node_id);
        stump_by_local.insert(local_id, stump);
    }
    let feature_count = binned_matrix.feature_count;

    for (row_index, prediction) in predictions.iter_mut().enumerate() {
        let row_base = row_index * feature_count;
        // Walk the tree starting from the root (local_node_id = 0)
        let mut local_id = 0_u32;
        loop {
            let Some(stump) = stump_by_local.get(&local_id) else {
                break; // reached a leaf — no stump at this node
            };
            let feature_index = stump.split.feature_index as usize;
            let bin = binned_matrix.row_bin(row_base + feature_index);
            // v0.10.0 review fix (Comment 1): multiply leaf contribution by
            // `stump.tree_weight` so warm-start prior predictions reflect
            // saved DART weights. For non-DART stumps tree_weight == 1.0,
            // so this is a no-op and preserves byte-identical numerics for
            // every existing caller (Standard/GOSS/Morph/DRO/linear).
            let tree_weight = stump.tree_weight;
            if bin <= stump.split.threshold_bin {
                let leaf_value = if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.left_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.left_leaf_value.as_scalar()
                };
                *prediction += tree_weight * leaf_value;
                local_id = local_id * 2 + 1; // left child
            } else {
                let leaf_value = if let Some((raw, fc)) = raw_features
                    && !raw.is_empty()
                {
                    let row_offset = row_index * fc;
                    stump.right_leaf_value.eval_row(&raw[row_offset..])
                } else {
                    stump.right_leaf_value.as_scalar()
                };
                *prediction += tree_weight * leaf_value;
                local_id = local_id * 2 + 2; // right child
            }
        }
    }
    Ok(())
}

fn apply_tree_to_binned_predictions(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    raw_features: Option<(&[f32], usize)>,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    // Split stumps into per-round groups by detecting tree_id changes
    let mut round_start = 0;
    let mut current_tree_id = decode_tree_node_id(stumps[0].split.node_id).0;
    for i in 1..stumps.len() {
        let tree_id = decode_tree_node_id(stumps[i].split.node_id).0;
        if tree_id != current_tree_id {
            apply_round_stumps_tree_walk(
                predictions,
                binned_matrix,
                &stumps[round_start..i],
                raw_features,
            )?;
            round_start = i;
            current_tree_id = tree_id;
        }
    }
    apply_round_stumps_tree_walk(
        predictions,
        binned_matrix,
        &stumps[round_start..],
        raw_features,
    )?;
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct LeafRefinementStats {
    weighted_sum: f32,
    weight_sum: f32,
}

impl LeafRefinementStats {
    fn push(&mut self, value: f32, weight: f32) {
        self.weighted_sum += value * weight;
        self.weight_sum += weight;
    }
}

fn refine_regression_leaf_values(
    baseline_prediction: f32,
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    binned_matrix: &BinnedMatrix,
    stumps: &mut [TrainedStump],
    stumps_per_completed_round: &[usize],
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if stumps.is_empty() || stumps_per_completed_round.is_empty() {
        return Ok(());
    }
    if targets.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "targets length {} does not match binned row_count {}",
            targets.len(),
            binned_matrix.row_count
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    let mut ensemble_predictions = vec![0.0_f32; targets.len()];
    for &round_stump_count in stumps_per_completed_round {
        if round_stump_count == 0 {
            continue;
        }
    }

    let mut cursor = 0_usize;
    for &round_stump_count in stumps_per_completed_round {
        let round_end = cursor.checked_add(round_stump_count).ok_or_else(|| {
            EngineError::ContractViolation("round stump count overflow".to_string())
        })?;
        if round_end > stumps.len() {
            return Err(EngineError::ContractViolation(
                "round stump counts exceed trained stump count".to_string(),
            ));
        }
        apply_tree_to_binned_predictions(
            &mut ensemble_predictions,
            binned_matrix,
            &stumps[cursor..round_end],
            None,
        )?;
        cursor = round_end;
    }
    if cursor != stumps.len() {
        return Err(EngineError::ContractViolation(
            "round stump counts do not cover all trained stumps".to_string(),
        ));
    }

    let mut cursor = 0_usize;
    for &round_stump_count in stumps_per_completed_round {
        let round_end = cursor.checked_add(round_stump_count).ok_or_else(|| {
            EngineError::ContractViolation("round stump count overflow".to_string())
        })?;
        if round_stump_count == 0 {
            cursor = round_end;
            continue;
        }

        let round_stumps = &mut stumps[cursor..round_end];
        let old_tree_predictions = tree_predictions_for_binned_rows(binned_matrix, round_stumps)?;
        let residual_without_tree = targets
            .iter()
            .enumerate()
            .map(|(row_index, target)| {
                target
                    - baseline_prediction
                    - (ensemble_predictions[row_index] - old_tree_predictions[row_index])
            })
            .collect::<Vec<_>>();
        let refined_tree = refine_tree_stumps(
            binned_matrix,
            round_stumps,
            &residual_without_tree,
            sample_weights,
            max_abs_leaf_value,
        )?;
        let new_tree_predictions = tree_predictions_for_binned_rows(binned_matrix, &refined_tree)?;
        for row_index in 0..ensemble_predictions.len() {
            ensemble_predictions[row_index] +=
                new_tree_predictions[row_index] - old_tree_predictions[row_index];
        }
        round_stumps.clone_from_slice(&refined_tree);
        cursor = round_end;
    }

    Ok(())
}

fn tree_predictions_for_binned_rows(
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
) -> EngineResult<Vec<f32>> {
    let mut predictions = vec![0.0_f32; binned_matrix.row_count];
    apply_tree_to_binned_predictions(&mut predictions, binned_matrix, stumps, None)?;
    Ok(predictions)
}

fn refine_tree_stumps(
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
    residual_without_tree: &[f32],
    sample_weights: Option<&[f32]>,
    max_abs_leaf_value: f32,
) -> EngineResult<Vec<TrainedStump>> {
    if stumps.is_empty() {
        return Ok(Vec::new());
    }
    if residual_without_tree.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "residual length {} does not match binned row_count {}",
            residual_without_tree.len(),
            binned_matrix.row_count
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != residual_without_tree.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match residual length {}",
            weights.len(),
            residual_without_tree.len()
        )));
    }

    let mut stumps_by_local = HashMap::new();
    for stump in stumps {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        stumps_by_local.insert(local_node_id, stump);
    }

    let mut current_absolute_outputs = HashMap::new();
    current_absolute_outputs.insert(0_u32, 0.0_f32);
    populate_child_absolute_outputs(0, &stumps_by_local, &mut current_absolute_outputs)?;

    let mut terminal_stats = HashMap::<u32, LeafRefinementStats>::new();
    for row_index in 0..binned_matrix.row_count {
        let terminal_local_node_id =
            terminal_local_node_id_for_row(row_index, binned_matrix, &stumps_by_local)?;
        let weight = sample_weights.map_or(1.0, |weights| weights[row_index]);
        terminal_stats
            .entry(terminal_local_node_id)
            .or_default()
            .push(residual_without_tree[row_index], weight);
    }

    let mut refined_absolute_outputs = HashMap::new();
    refined_absolute_outputs.insert(0_u32, 0.0_f32);
    fill_refined_child_absolute_outputs(
        0,
        &stumps_by_local,
        &terminal_stats,
        &current_absolute_outputs,
        max_abs_leaf_value,
        &mut refined_absolute_outputs,
    )?;

    let mut refined_stumps = stumps.to_vec();
    for stump in &mut refined_stumps {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let parent_absolute = refined_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0);
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_absolute = refined_absolute_outputs
            .get(&left_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.left_leaf_value.as_scalar());
        let right_absolute = refined_absolute_outputs
            .get(&right_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.right_leaf_value.as_scalar());
        stump.left_leaf_value = LeafValue::Scalar(left_absolute - parent_absolute);
        stump.right_leaf_value = LeafValue::Scalar(right_absolute - parent_absolute);
    }

    Ok(refined_stumps)
}

fn populate_child_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<()> {
    let Some(stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(());
    };
    let parent_absolute = absolute_outputs.get(&local_node_id).copied().unwrap_or(0.0);
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;
    absolute_outputs.insert(
        left_local_node_id,
        parent_absolute + stump.left_leaf_value.as_scalar(),
    );
    absolute_outputs.insert(
        right_local_node_id,
        parent_absolute + stump.right_leaf_value.as_scalar(),
    );
    populate_child_absolute_outputs(left_local_node_id, stumps_by_local, absolute_outputs)?;
    populate_child_absolute_outputs(right_local_node_id, stumps_by_local, absolute_outputs)?;
    Ok(())
}

fn terminal_local_node_id_for_row(
    row_index: usize,
    binned_matrix: &BinnedMatrix,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
) -> EngineResult<u32> {
    let mut local_node_id = 0_u32;
    loop {
        let Some(stump) = stumps_by_local.get(&local_node_id) else {
            return Err(EngineError::ContractViolation(format!(
                "tree is missing split for local node {local_node_id}"
            )));
        };
        let feature_index = stump.split.feature_index as usize;
        if feature_index >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                stump.split.feature_index, binned_matrix.feature_count
            )));
        }
        let cell_index = row_index
            .checked_mul(binned_matrix.feature_count)
            .and_then(|base| base.checked_add(feature_index))
            .ok_or_else(|| {
                EngineError::ContractViolation("binned cell index overflow".to_string())
            })?;
        if cell_index >= binned_matrix.bins_adaptive.len() {
            return Err(EngineError::ContractViolation(format!(
                "binned cell index {cell_index} is out of bounds for bins length {}",
                binned_matrix.bins_adaptive.len()
            )));
        }
        let bin = binned_matrix.row_bin(cell_index);
        let next_local_node_id = if bin <= stump.split.threshold_bin {
            left_child_node_id(local_node_id)?
        } else {
            right_child_node_id(local_node_id)?
        };
        if !stumps_by_local.contains_key(&next_local_node_id) {
            return Ok(next_local_node_id);
        }
        local_node_id = next_local_node_id;
    }
}

fn fill_refined_child_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    terminal_stats: &HashMap<u32, LeafRefinementStats>,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    let Some(_stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(LeafRefinementStats::default());
    };
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;

    let left_stats = fill_refined_subtree_absolute_output(
        left_local_node_id,
        stumps_by_local,
        terminal_stats,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;
    let right_stats = fill_refined_subtree_absolute_output(
        right_local_node_id,
        stumps_by_local,
        terminal_stats,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;

    let mut subtree_stats = left_stats;
    subtree_stats.weighted_sum += right_stats.weighted_sum;
    subtree_stats.weight_sum += right_stats.weight_sum;
    Ok(subtree_stats)
}

fn fill_refined_subtree_absolute_output(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    terminal_stats: &HashMap<u32, LeafRefinementStats>,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    if stumps_by_local.contains_key(&local_node_id) {
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_stats = fill_refined_subtree_absolute_output(
            left_local_node_id,
            stumps_by_local,
            terminal_stats,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let right_stats = fill_refined_subtree_absolute_output(
            right_local_node_id,
            stumps_by_local,
            terminal_stats,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let total_weight = left_stats.weight_sum + right_stats.weight_sum;
        let absolute_output = if total_weight > 0.0 {
            ((left_stats.weighted_sum + right_stats.weighted_sum) / total_weight)
                .clamp(-max_abs_leaf_value, max_abs_leaf_value)
        } else {
            current_absolute_outputs
                .get(&local_node_id)
                .copied()
                .unwrap_or(0.0)
        };
        refined_absolute_outputs.insert(local_node_id, absolute_output);
        return Ok(LeafRefinementStats {
            weighted_sum: absolute_output * total_weight,
            weight_sum: total_weight,
        });
    }

    let stats = terminal_stats
        .get(&local_node_id)
        .copied()
        .unwrap_or_default();
    let absolute_output = if stats.weight_sum > 0.0 {
        (stats.weighted_sum / stats.weight_sum).clamp(-max_abs_leaf_value, max_abs_leaf_value)
    } else {
        current_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0)
    };
    refined_absolute_outputs.insert(local_node_id, absolute_output);
    Ok(LeafRefinementStats {
        weighted_sum: absolute_output * stats.weight_sum,
        weight_sum: stats.weight_sum,
    })
}

struct LeafResiduals {
    residuals: Vec<f32>,
    weights: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn refine_quantile_leaf_values(
    stumps: &mut [TrainedStump],
    binned_matrix: &BinnedMatrix,
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
    alpha: f32,
    learning_rate: f32,
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if stumps.is_empty() {
        return Ok(());
    }
    if targets.len() != binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "targets length {} does not match binned row_count {}",
            targets.len(),
            binned_matrix.row_count
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    let mut stumps_by_local = HashMap::new();
    for stump in stumps.iter() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        stumps_by_local.insert(local_node_id, stump);
    }

    let mut current_absolute_outputs = HashMap::new();
    current_absolute_outputs.insert(0_u32, 0.0_f32);
    populate_child_absolute_outputs(0, &stumps_by_local, &mut current_absolute_outputs)?;

    // We map every row in the full training dataset to its terminal leaf to collect residuals.
    // When row_subsample < 1.0, split-finding is performed on a subsampled subset of rows,
    // but we use the entire training set for the final quantile leaf refinement step
    // to minimize estimation variance of the empirical quantile.
    let mut leaf_residuals: HashMap<u32, LeafResiduals> = HashMap::new();
    for row_index in 0..binned_matrix.row_count {
        let terminal_local_node_id =
            terminal_local_node_id_for_row(row_index, binned_matrix, &stumps_by_local)?;
        let res = targets[row_index] - predictions[row_index];
        let entry = leaf_residuals
            .entry(terminal_local_node_id)
            .or_insert_with(|| LeafResiduals {
                residuals: Vec::new(),
                weights: sample_weights.map(|_| Vec::new()),
            });
        entry.residuals.push(res);
        if let (Some(w_vec), Some(weights)) = (&mut entry.weights, sample_weights) {
            w_vec.push(weights[row_index]);
        }
    }

    let mut refined_absolute_outputs = HashMap::new();
    refined_absolute_outputs.insert(0_u32, 0.0_f32);

    fill_refined_child_quantile_absolute_outputs(
        0,
        &stumps_by_local,
        &leaf_residuals,
        alpha,
        &current_absolute_outputs,
        max_abs_leaf_value,
        &mut refined_absolute_outputs,
    )?;

    for stump in stumps.iter_mut() {
        let (_, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let parent_absolute = refined_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0);
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_absolute = refined_absolute_outputs
            .get(&left_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.left_leaf_value.as_scalar());
        let right_absolute = refined_absolute_outputs
            .get(&right_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.right_leaf_value.as_scalar());

        let dl = (left_absolute - parent_absolute) * learning_rate;
        let dr = (right_absolute - parent_absolute) * learning_rate;
        stump.left_leaf_value = LeafValue::Scalar(dl);
        stump.right_leaf_value = LeafValue::Scalar(dr);
    }

    Ok(())
}

fn fill_refined_child_quantile_absolute_outputs(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    leaf_residuals: &HashMap<u32, LeafResiduals>,
    alpha: f32,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    let Some(_stump) = stumps_by_local.get(&local_node_id) else {
        return Ok(LeafRefinementStats::default());
    };
    let left_local_node_id = left_child_node_id(local_node_id)?;
    let right_local_node_id = right_child_node_id(local_node_id)?;

    let left_stats = fill_refined_subtree_quantile_absolute_output(
        left_local_node_id,
        stumps_by_local,
        leaf_residuals,
        alpha,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;
    let right_stats = fill_refined_subtree_quantile_absolute_output(
        right_local_node_id,
        stumps_by_local,
        leaf_residuals,
        alpha,
        current_absolute_outputs,
        max_abs_leaf_value,
        refined_absolute_outputs,
    )?;

    let mut subtree_stats = left_stats;
    subtree_stats.weighted_sum += right_stats.weighted_sum;
    subtree_stats.weight_sum += right_stats.weight_sum;
    Ok(subtree_stats)
}

fn fill_refined_subtree_quantile_absolute_output(
    local_node_id: u32,
    stumps_by_local: &HashMap<u32, &TrainedStump>,
    leaf_residuals: &HashMap<u32, LeafResiduals>,
    alpha: f32,
    current_absolute_outputs: &HashMap<u32, f32>,
    max_abs_leaf_value: f32,
    refined_absolute_outputs: &mut HashMap<u32, f32>,
) -> EngineResult<LeafRefinementStats> {
    if stumps_by_local.contains_key(&local_node_id) {
        let left_local_node_id = left_child_node_id(local_node_id)?;
        let right_local_node_id = right_child_node_id(local_node_id)?;
        let left_stats = fill_refined_subtree_quantile_absolute_output(
            left_local_node_id,
            stumps_by_local,
            leaf_residuals,
            alpha,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let right_stats = fill_refined_subtree_quantile_absolute_output(
            right_local_node_id,
            stumps_by_local,
            leaf_residuals,
            alpha,
            current_absolute_outputs,
            max_abs_leaf_value,
            refined_absolute_outputs,
        )?;
        let total_weight = left_stats.weight_sum + right_stats.weight_sum;
        let absolute_output = if total_weight > 0.0 {
            ((left_stats.weighted_sum + right_stats.weighted_sum) / total_weight)
                .clamp(-max_abs_leaf_value, max_abs_leaf_value)
        } else {
            current_absolute_outputs
                .get(&local_node_id)
                .copied()
                .unwrap_or(0.0)
        };
        refined_absolute_outputs.insert(local_node_id, absolute_output);
        return Ok(LeafRefinementStats {
            weighted_sum: absolute_output * total_weight,
            weight_sum: total_weight,
        });
    }

    let (q_val, weight_sum) = if let Some(lr) = leaf_residuals.get(&local_node_id) {
        if let Some(ref w_vec) = lr.weights {
            let total_w: f32 = w_vec.iter().sum();
            if total_w > 0.0 {
                let q = weighted_quantile(&lr.residuals, Some(w_vec), alpha)?;
                (q, total_w)
            } else {
                (0.0, 0.0)
            }
        } else {
            let count = lr.residuals.len();
            if count > 0 {
                let q = weighted_quantile(&lr.residuals, None, alpha)?;
                (q, count as f32)
            } else {
                (0.0, 0.0)
            }
        }
    } else {
        (0.0, 0.0)
    };

    let absolute_output = if weight_sum > 0.0 {
        q_val.clamp(-max_abs_leaf_value, max_abs_leaf_value)
    } else {
        current_absolute_outputs
            .get(&local_node_id)
            .copied()
            .unwrap_or(0.0)
    };
    refined_absolute_outputs.insert(local_node_id, absolute_output);
    Ok(LeafRefinementStats {
        weighted_sum: absolute_output * weight_sum,
        weight_sum,
    })
}

pub(crate) fn squared_error_loss(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    if predictions.len() != targets.len() {
        return Err(EngineError::ContractViolation(format!(
            "predictions length {} does not match targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    Ok(squared_error_loss_unchecked(
        predictions,
        targets,
        sample_weights,
    ))
}

fn squared_error_loss_unchecked(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> f32 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let sum = if let Some(weights) = sample_weights {
        let mut total = 0.0_f32;
        for index in 0..n {
            let residual = predictions[index] - targets[index];
            total += residual * residual * weights[index];
        }
        total
    } else {
        // Unrolled 4-wide accumulation for auto-vectorization
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut sum2 = 0.0_f32;
        let mut sum3 = 0.0_f32;
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let r0 = predictions[base] - targets[base];
            let r1 = predictions[base + 1] - targets[base + 1];
            let r2 = predictions[base + 2] - targets[base + 2];
            let r3 = predictions[base + 3] - targets[base + 3];
            sum0 += r0 * r0;
            sum1 += r1 * r1;
            sum2 += r2 * r2;
            sum3 += r3 * r3;
        }
        let mut total = sum0 + sum1 + sum2 + sum3;
        for index in (chunks * 4)..n {
            let residual = predictions[index] - targets[index];
            total += residual * residual;
        }
        total
    };
    // Return mean squared error (not sum) for scale-independent loss values.
    sum / n as f32
}

pub(crate) fn binary_crossentropy_loss(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    if predictions.len() != targets.len() {
        return Err(EngineError::ContractViolation(format!(
            "predictions length {} does not match targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }
    // Numerically stable log-loss: -[y*log(p) + (1-y)*log(1-p)]
    // where p = sigmoid(prediction) and prediction is in logit space.
    // Stable formulation: max(pred,0) - pred*y + log(1 + exp(-|pred|))
    let n = predictions.len();
    if n == 0 {
        return Ok(0.0);
    }
    let mut total = 0.0_f32;
    for index in 0..n {
        let pred = predictions[index];
        let y = targets[index];
        let weight = sample_weights.map_or(1.0, |w| w[index]);
        let loss = pred.max(0.0) - pred * y + (1.0 + (-pred.abs()).exp()).ln();
        total += loss * weight;
    }
    // Return mean log-loss (not sum) for scale-independent loss values.
    Ok(total / n as f32)
}

pub(crate) fn required_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> EngineResult<&ModelArtifactSection> {
    optional_single_section(sections, kind)?.ok_or_else(|| {
        EngineError::ContractViolation(format!(
            "model artifact missing required {:?} section",
            kind
        ))
    })
}

fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> EngineResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(EngineError::ContractViolation(format!(
                "model artifact contains duplicate required {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

fn artifact_compatibility_report_from_sections(
    sections: &[ModelArtifactSection],
) -> ArtifactCompatibilityReport {
    let report = required_section_compatibility_report(sections);
    let recommended_mode = if report.strict_compatible {
        Some(ArtifactCompatibilityMode::Strict)
    } else if report.legacy_trees_only_compatible {
        Some(ArtifactCompatibilityMode::AllowLegacyTreesOnly)
    } else {
        None
    };

    ArtifactCompatibilityReport {
        trees_section_count: report.trees_section_count,
        predictor_layout_section_count: report.predictor_layout_section_count,
        strict_compatible: report.strict_compatible,
        legacy_trees_only_compatible: report.legacy_trees_only_compatible,
        legacy_compatible: report.legacy_compatible,
        recommended_mode,
    }
}

fn resolve_predictor_layout(
    sections: &[ModelArtifactSection],
    metadata_feature_count: usize,
    compatibility_mode: ArtifactCompatibilityMode,
) -> EngineResult<PredictorLayoutPayload> {
    if let Some(section) = optional_single_section(sections, ModelSectionKind::PredictorLayout)? {
        return decode_predictor_layout_payload(&section.payload);
    }

    if compatibility_mode == ArtifactCompatibilityMode::AllowLegacyTreesOnly
        && sections.len() == 1
        && sections[0].descriptor.kind == ModelSectionKind::Trees
    {
        // Compatibility path for v0.0.4 legacy payloads that only carried Trees.
        return Ok(PredictorLayoutPayload {
            feature_count: metadata_feature_count,
        });
    }

    Err(EngineError::ContractViolation(
        "model artifact missing required PredictorLayout section".to_string(),
    ))
}

fn encode_predictor_layout_payload(model: &TrainedModel) -> EngineResult<Vec<u8>> {
    let feature_count = u32::try_from(model.feature_count).map_err(|_| {
        EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
    })?;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&THRESHOLD_MODE_BIN_INDEX.to_le_bytes());
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PredictorLayoutPayload {
    feature_count: usize,
}

fn decode_predictor_layout_payload(bytes: &[u8]) -> EngineResult<PredictorLayoutPayload> {
    const LAYOUT_LEN: usize = 12;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
    if bytes.len() != LAYOUT_LEN {
        return Err(EngineError::ContractViolation(format!(
            "predictor layout payload length {} does not match expected {LAYOUT_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported predictor layout format version {format_version}"
        )));
    }

    let feature_count = read_u32_le(bytes, 4)? as usize;
    let threshold_mode = read_u32_le(bytes, 8)?;
    if threshold_mode != THRESHOLD_MODE_BIN_INDEX {
        return Err(EngineError::ContractViolation(format!(
            "unsupported predictor layout threshold mode {threshold_mode}"
        )));
    }

    Ok(PredictorLayoutPayload { feature_count })
}

fn encode_node_debug_stats_payload(node_debug_stats: &[NodeDebugStats]) -> EngineResult<Vec<u8>> {
    let record_count = u32::try_from(node_debug_stats.len()).map_err(|_| {
        EngineError::ContractViolation("node debug stats count exceeds u32::MAX".to_string())
    })?;

    let mut bytes = Vec::with_capacity(8 + node_debug_stats.len() * 40);
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    for record in node_debug_stats {
        bytes.extend_from_slice(&record.node_id.to_le_bytes());
        bytes.extend_from_slice(&record.feature_index.to_le_bytes());
        bytes.extend_from_slice(&record.threshold_bin.to_le_bytes());
        let flags: u16 = if record.default_left { 1 } else { 0 };
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&record.gain.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.grad_sum.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.hess_sum.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.row_count.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.grad_sum.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.hess_sum.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.row_count.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_node_debug_stats_payload(bytes: &[u8]) -> EngineResult<Vec<NodeDebugStats>> {
    const HEADER_SIZE: usize = 8;
    const RECORD_SIZE: usize = 40;
    if bytes.len() < HEADER_SIZE {
        return Err(EngineError::ContractViolation(
            "node debug stats payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported node debug stats format version {format_version}"
        )));
    }
    let record_count = read_u32_le(bytes, 4)? as usize;
    let expected_len = HEADER_SIZE
        .checked_add(record_count.checked_mul(RECORD_SIZE).ok_or_else(|| {
            EngineError::ContractViolation("node debug stats payload length overflow".to_string())
        })?)
        .ok_or_else(|| {
            EngineError::ContractViolation("node debug stats payload length overflow".to_string())
        })?;
    if bytes.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
            "node debug stats payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut records = Vec::with_capacity(record_count);
    for record_index in 0..record_count {
        let base = HEADER_SIZE + record_index * RECORD_SIZE;
        let nds_flags = read_u16_le(bytes, base + 10)?;
        records.push(NodeDebugStats {
            node_id: read_u32_le(bytes, base)?,
            feature_index: read_u32_le(bytes, base + 4)?,
            threshold_bin: read_u16_le(bytes, base + 8)?,
            gain: read_f32_le(bytes, base + 12)?,
            default_left: (nds_flags & 1) != 0,
            left_stats: NodeStats {
                grad_sum: read_f32_le(bytes, base + 16)?,
                hess_sum: read_f32_le(bytes, base + 20)?,
                grad_sq_sum: 0.0,
                row_count: read_u32_le(bytes, base + 24)?,
            },
            right_stats: NodeStats {
                grad_sum: read_f32_le(bytes, base + 28)?,
                hess_sum: read_f32_le(bytes, base + 32)?,
                grad_sq_sum: 0.0,
                row_count: read_u32_le(bytes, base + 36)?,
            },
        });
    }
    Ok(records)
}

fn decode_optional_node_debug_stats_section(
    sections: &[ModelArtifactSection],
) -> EngineResult<Option<Vec<NodeDebugStats>>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::NodeDebugStats)? else {
        return Ok(None);
    };
    Ok(Some(decode_node_debug_stats_payload(&section.payload)?))
}

fn encode_trained_model_payload(model: &TrainedModel) -> EngineResult<Vec<u8>> {
    let feature_count = u32::try_from(model.feature_count).map_err(|_| {
        EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
    })?;
    let stump_count = u32::try_from(model.stumps.len())
        .map_err(|_| EngineError::ContractViolation("stump count exceeds u32::MAX".to_string()))?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&stump_count.to_le_bytes());
    bytes.extend_from_slice(&model.baseline_prediction.to_le_bytes());

    for stump in &model.stumps {
        bytes.extend_from_slice(&stump.split.node_id.to_le_bytes());
        bytes.extend_from_slice(&stump.split.feature_index.to_le_bytes());
        bytes.extend_from_slice(&stump.split.threshold_bin.to_le_bytes());
        let mut stump_flags: u16 = if stump.split.default_left { 1 } else { 0 };
        if stump.split.is_categorical {
            stump_flags |= 2; // bit 1 = is_categorical
        }
        bytes.extend_from_slice(&stump_flags.to_le_bytes());
        bytes.extend_from_slice(&stump.split.gain.to_le_bytes());
        bytes.extend_from_slice(&stump.left_leaf_value.as_scalar().to_le_bytes());
        bytes.extend_from_slice(&stump.right_leaf_value.as_scalar().to_le_bytes());
        bytes.extend_from_slice(&stump.split.left_stats.row_count.to_le_bytes());
        bytes.extend_from_slice(&stump.split.right_stats.row_count.to_le_bytes());
    }

    Ok(bytes)
}

fn decode_trained_model_payload(bytes: &[u8]) -> EngineResult<TrainedModel> {
    const HEADER_SIZE: usize = 16;
    const STUMP_SIZE: usize = 32;
    if bytes.len() < HEADER_SIZE {
        return Err(EngineError::ContractViolation(
            "model payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported model payload format version {format_version}"
        )));
    }
    let feature_count = read_u32_le(bytes, 4)? as usize;
    let stump_count = read_u32_le(bytes, 8)? as usize;
    let baseline_prediction = read_f32_le(bytes, 12)?;

    let expected_len = HEADER_SIZE
        .checked_add(stump_count.checked_mul(STUMP_SIZE).ok_or_else(|| {
            EngineError::ContractViolation("stump payload length overflow".to_string())
        })?)
        .ok_or_else(|| EngineError::ContractViolation("payload length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
            "model payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut stumps = Vec::with_capacity(stump_count);
    for stump_index in 0..stump_count {
        let base = HEADER_SIZE + stump_index * STUMP_SIZE;
        let node_id = read_u32_le(bytes, base)?;
        let feature_index = read_u32_le(bytes, base + 4)?;
        let threshold_bin = read_u16_le(bytes, base + 8)?;
        let flags = read_u16_le(bytes, base + 10)?;
        let default_left = (flags & 1) != 0;
        let is_categorical = (flags & 2) != 0;
        let gain = read_f32_le(bytes, base + 12)?;
        let left_leaf_value = read_f32_le(bytes, base + 16)?;
        let right_leaf_value = read_f32_le(bytes, base + 20)?;
        let left_count = read_u32_le(bytes, base + 24)?;
        let right_count = read_u32_le(bytes, base + 28)?;

        stumps.push(TrainedStump {
            split: SplitCandidate {
                node_id,
                feature_index,
                threshold_bin,
                gain,
                default_left,
                is_categorical,
                categorical_bitset: None, // populated from NativeCategoricalSplits section
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: left_count as f32,
                    grad_sq_sum: 0.0,
                    row_count: left_count,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: right_count as f32,
                    grad_sq_sum: 0.0,
                    row_count: right_count,
                },
            },
            left_leaf_value: LeafValue::Scalar(left_leaf_value),
            right_leaf_value: LeafValue::Scalar(right_leaf_value),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        });
    }

    Ok(TrainedModel {
        baseline_prediction,
        feature_count,
        stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    })
}

pub(crate) fn read_u32_le(bytes: &[u8], start: usize) -> EngineResult<u32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
            "unexpected end of payload when reading u32".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}

pub(crate) fn read_u16_le(bytes: &[u8], start: usize) -> EngineResult<u16> {
    let end = start + 2;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
            "unexpected end of payload when reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

pub(crate) fn read_f32_le(bytes: &[u8], start: usize) -> EngineResult<f32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
            "unexpected end of payload when reading f32".to_string(),
        ));
    }
    Ok(f32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}


#[cfg(test)]
mod tests;
