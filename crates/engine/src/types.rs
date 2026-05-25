use alloygbm_categorical::TargetEncoderConfig;
use alloygbm_core::{
    BinnedMatrix, GradientPair, LeafValue, NodeStats, PartitionResult, SplitCandidate,
    TrainingDataset,
};

use crate::error::{EngineError, EngineResult};
use crate::split_options::CategoricalFeatureInfo;
use crate::traits::PerRoundMetricCallback;
use crate::warm_start::WarmStartState;

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
    pub model: crate::TrainedModel,
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
pub(crate) fn gradient_buffer_stats(gradients: &[GradientPair]) -> (f32, f32, f32) {
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
pub(crate) fn gradient_l2_norm_only(gradients: &[GradientPair]) -> f32 {
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
pub(crate) struct PolicyFitRequest {
    pub(crate) rounds: usize,
    pub(crate) policy_mode: TrainingPolicyMode,
    pub(crate) store_node_debug_stats: bool,
}

pub(crate) struct IterationExecutionContext<'a> {
    pub(crate) controls: IterationControls,
    pub(crate) validation: Option<ValidationDatasetRef<'a>>,
    pub(crate) policy_mode: Option<TrainingPolicyMode>,
    pub(crate) warm_start: Option<WarmStartState>,
    pub(crate) custom_metric_callback: Option<&'a dyn PerRoundMetricCallback>,
    /// Features that use native categorical splits (empty = all continuous).
    pub(crate) categorical_features: Vec<CategoricalFeatureInfo>,
    pub(crate) pre_target_already_applied: bool,
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
    pub(crate) fn required_section_report(
        self,
    ) -> alloygbm_core::RequiredSectionCompatibilityReport {
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
