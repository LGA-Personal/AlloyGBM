//! Data-shape types for the joint multi-output trainer.
//!
//! Extracted from `joint/mod.rs` in v0.12.2 task 5.2. Public items
//! (`JointObjective`, `JointTrainingSummary`, `JointRoundResult`,
//! `JointWarmStartState`, `JointPredictor`) are re-exported by the
//! parent `joint` module. The remaining helpers (`JointLeafNode`,
//! `JointLeafCandidate`, `JointMorphContext`, `JointPredictorStump`)
//! are `pub(super)` so sibling modules inside `joint/` can consume
//! them.

use alloygbm_core::{GradientPair, NodeStats};

use crate::{
    GammaObjective, LambdaMARTObjective, ObjectiveOps, PairwiseRankingObjective, PoissonObjective,
    QuantileObjective, QueryRMSEObjective, SquaredErrorObjective, TrainedModel, TrainedStump,
    TweedieObjective, XeNDCGObjective,
};

use super::TREE_NODE_STRIDE;
use super::helpers::bitset_bytes_to_u64;

/// Runtime selector for per-output objective on the joint trainer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointObjective {
    SquaredError,
    QueryRmse,
    RankPairwise,
    RankPairwiseWithSigma { sigma: f32 },
    RankNdcg,
    RankNdcgWithSigma { sigma: f32 },
    RankXendcg,
    Poisson { max_delta_step: f32 },
    Gamma,
    Tweedie { variance_power: f32 },
    Quantile { alpha: f32 },
}

impl JointObjective {
    pub fn parse(name: &str) -> Result<Self, String> {
        Self::parse_with_ranking_sigma(name, 1.0)
    }

    pub fn parse_with_ranking_sigma(name: &str, ranking_sigma: f32) -> Result<Self, String> {
        let sigma = validate_joint_ranking_sigma(ranking_sigma)?;
        match name {
            "squared_error" => Ok(Self::SquaredError),
            "queryrmse" => Ok(Self::QueryRmse),
            "rank:pairwise" if sigma == 1.0 => Ok(Self::RankPairwise),
            "rank:pairwise" => Ok(Self::RankPairwiseWithSigma { sigma }),
            "rank:ndcg" if sigma == 1.0 => Ok(Self::RankNdcg),
            "rank:ndcg" => Ok(Self::RankNdcgWithSigma { sigma }),
            "rank:xendcg" => Ok(Self::RankXendcg),
            "poisson" => Ok(Self::Poisson {
                max_delta_step: 0.7,
            }),
            "gamma" => Ok(Self::Gamma),
            "tweedie" => Ok(Self::Tweedie {
                variance_power: 1.5,
            }),
            "quantile" => Ok(Self::Quantile { alpha: 0.5 }),
            other => Err(format!(
                "joint multi-output trainer does not support objective {other:?}; \
                 supported: squared_error, queryrmse, rank:pairwise, rank:ndcg, rank:xendcg, poisson, gamma, tweedie, quantile"
            )),
        }
    }

    pub fn requires_group(&self) -> bool {
        matches!(
            self,
            Self::QueryRmse
                | Self::RankPairwise
                | Self::RankPairwiseWithSigma { .. }
                | Self::RankNdcg
                | Self::RankNdcgWithSigma { .. }
                | Self::RankXendcg
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::SquaredError => "squared_error",
            Self::QueryRmse => "queryrmse",
            Self::RankPairwise | Self::RankPairwiseWithSigma { .. } => "rank:pairwise",
            Self::RankNdcg | Self::RankNdcgWithSigma { .. } => "rank:ndcg",
            Self::RankXendcg => "rank:xendcg",
            Self::Poisson { .. } => "poisson",
            Self::Gamma => "gamma",
            Self::Tweedie { .. } => "tweedie",
            Self::Quantile { .. } => "quantile",
        }
    }

    /// Compute initial prediction baseline for this output.
    pub fn initial_prediction(&self, targets: &[f32]) -> f32 {
        match self {
            Self::SquaredError => {
                if targets.is_empty() {
                    0.0
                } else {
                    targets.iter().sum::<f32>() / targets.len() as f32
                }
            }
            Self::Poisson { .. } => {
                let obj = PoissonObjective::default();
                obj.initial_prediction(targets, None).unwrap_or(0.0)
            }
            Self::Gamma => {
                let obj = GammaObjective;
                obj.initial_prediction(targets, None).unwrap_or(0.0)
            }
            Self::Tweedie { variance_power } => {
                let obj = TweedieObjective::new(*variance_power).unwrap();
                obj.initial_prediction(targets, None).unwrap_or(0.0)
            }
            Self::Quantile { alpha } => {
                let obj = QuantileObjective { alpha: *alpha };
                obj.initial_prediction(targets, None).unwrap_or(0.0)
            }
            // Ranking objectives use 0.0 as the conventional initial prediction
            // (gradient depends on relative score within group).
            _ => 0.0,
        }
    }

    /// Compute (grad, hess) per row for this output's predictions vs targets,
    /// optionally with per-row group identifiers (required for ranking
    /// objectives). The returned Vec is row-major with the same length as
    /// `predictions`.
    pub fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        group: Option<&[u32]>,
    ) -> Result<Vec<GradientPair>, String> {
        match self {
            Self::SquaredError => {
                let obj = SquaredErrorObjective;
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::QueryRmse => {
                let group_ids = group
                    .ok_or_else(|| "queryrmse objective requires group identifiers".to_string())?;
                let obj = QueryRMSEObjective::new(group_ids);
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::RankPairwise => {
                let group_ids = group.ok_or_else(|| {
                    "rank:pairwise objective requires group identifiers".to_string()
                })?;
                let obj = PairwiseRankingObjective::new(group_ids);
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::RankPairwiseWithSigma { sigma } => {
                let group_ids = group.ok_or_else(|| {
                    "rank:pairwise objective requires group identifiers".to_string()
                })?;
                let obj = PairwiseRankingObjective::new_with_sigma(group_ids, *sigma)
                    .map_err(|e| e.to_string())?;
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::RankNdcg => {
                let group_ids = group
                    .ok_or_else(|| "rank:ndcg objective requires group identifiers".to_string())?;
                let obj = LambdaMARTObjective::new(group_ids);
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::RankNdcgWithSigma { sigma } => {
                let group_ids = group
                    .ok_or_else(|| "rank:ndcg objective requires group identifiers".to_string())?;
                let obj = LambdaMARTObjective::new_with_sigma(group_ids, *sigma)
                    .map_err(|e| e.to_string())?;
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::RankXendcg => {
                let group_ids = group.ok_or_else(|| {
                    "rank:xendcg objective requires group identifiers".to_string()
                })?;
                let obj = XeNDCGObjective::new(group_ids);
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::Poisson { max_delta_step } => {
                let obj = PoissonObjective::new(*max_delta_step);
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::Gamma => {
                let obj = GammaObjective;
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::Tweedie { variance_power } => {
                let obj = TweedieObjective::new(*variance_power).map_err(|e| e.to_string())?;
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
            Self::Quantile { alpha } => {
                let obj = QuantileObjective { alpha: *alpha };
                obj.compute_gradients(predictions, targets, None)
                    .map_err(|e| e.to_string())
            }
        }
    }
}

fn validate_joint_ranking_sigma(sigma: f32) -> Result<f32, String> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err("ranking_sigma must be finite and > 0".to_string());
    }
    Ok(sigma)
}

/// Summary returned by [`fit_joint_multi_output`].
#[derive(Debug, Clone)]
pub struct JointTrainingSummary {
    /// Per-output baseline predictions (initial residual zero-point). Length = K.
    pub baselines: Vec<f32>,
    /// The trained model. Stumps carry `multi_output_leaf_values` and the
    /// artifact serializer will emit a `MultiOutputLeafValues` section
    /// alongside the standard `Trees` payload.
    pub model: TrainedModel,
    /// Per-output objective names that were used (for metadata round-trip).
    pub per_output_objective_names: Vec<String>,
    /// Number of boosting rounds actually completed. In v0.10.x the joint
    /// trainer always runs to `n_estimators` (no early stopping yet), so
    /// `rounds_completed == n_estimators` for every successful fit.
    pub rounds_completed: usize,
}

/// Per-leaf bookkeeping during level-wise tree growth on the joint trainer.
#[derive(Debug)]
pub(super) struct JointLeafNode {
    /// 0-indexed local node id within this tree (matches predictor traversal).
    pub(super) local_node_id: u32,
    /// Row indices currently routed to this node.
    pub(super) row_indices: Vec<u32>,
}

/// A candidate split for leaf-wise (best-first) growth on the joint trainer.
/// Held in a `BinaryHeap` keyed by `gain` (max-heap). The candidate carries
/// the resolved split decision (feature, threshold_bin, row partition, K-output
/// leaf values) so popping the best candidate immediately commits a stump
/// without re-running the histogram sweep.
///
/// `parent_active_groups` carries the parent node's interaction-constraint
/// active group bitset so descendants of a split node propagate the
/// constraint set correctly.
#[derive(Debug)]
pub(super) struct JointLeafCandidate {
    pub(super) node: JointLeafNode,
    pub(super) feature: u32,
    pub(super) threshold_bin: u16,
    pub(super) default_left: bool,
    pub(super) gain: f32,
    pub(super) left_rows: Vec<u32>,
    pub(super) right_rows: Vec<u32>,
    pub(super) left_k: Vec<f32>,
    pub(super) right_k: Vec<f32>,
    pub(super) left_stats: NodeStats,
    pub(super) right_stats: NodeStats,
    pub(super) parent_active_groups: Option<u64>,
    /// Categorical bitset (Some when this is a Fisher-sort categorical
    /// split, None for numeric threshold splits). Bit `k` = 1 means
    /// category `k` is routed to the left child.
    pub(super) cat_bitset: Option<u64>,
}

impl PartialEq for JointLeafCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain == other.gain
    }
}
impl Eq for JointLeafCandidate {}
impl Ord for JointLeafCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // f32 max-heap: NaN treated as least via `unwrap_or(Equal)`.
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for JointLeafCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Output of a single joint training round: one shared tree expressed as a
/// sequence of stumps with K-output leaf values populated.
#[derive(Debug, Clone)]
pub struct JointRoundResult {
    pub stumps: Vec<TrainedStump>,
}

/// Per-round MorphBoost context passed to `build_joint_round*`.
///
/// Carries the snapshot of `MorphState` needed for split-gain dispatch:
/// the morph config + precomputed per-iteration constants, plus per-output
/// `(grad_mean, grad_std)` pulled from `MorphState::ema_stats[k]`.
///
/// Separate from the single-output `crate::MorphTreeContext` (which is
/// pub(crate) and tied to single-output `MorphState`) — joint mode tracks
/// K independent EMA snapshots and routes them through the multi-output
/// gain helpers in `shared_histogram.rs`.
#[derive(Debug, Clone)]
pub(super) struct JointMorphContext {
    pub(super) config: alloygbm_core::MorphConfig,
    pub(super) precomputed: alloygbm_core::MorphPrecomputed,
    pub(super) iteration: u32,
    pub(super) total_iterations: u32,
    pub(super) grad_means: Vec<f32>,
    pub(super) grad_stds: Vec<f32>,
}

impl JointMorphContext {
    pub(super) fn from_state(
        state: &crate::MorphState,
        iteration: u32,
        total_iterations: u32,
    ) -> Self {
        let grad_means: Vec<f32> = state.ema_stats.iter().map(|s| s.mean).collect();
        let grad_stds: Vec<f32> = state.ema_stats.iter().map(|s| s.std).collect();
        Self {
            config: state.config,
            precomputed: alloygbm_core::MorphPrecomputed::for_iteration(iteration, &state.config),
            iteration,
            total_iterations,
            grad_means,
            grad_stds,
        }
    }
}

/// Run the full joint multi-output training loop and return a `TrainedModel`
/// plus per-output baselines. The model's stumps carry
/// `multi_output_leaf_values` and serialize into both the `Trees` and
/// `MultiOutputLeafValues` sections.
///
/// Arguments:
///   `targets_per_output[k]` is the target vector for output `k` (length n_rows).
///   `per_output_objective[k]` selects the objective used for output `k`.
///   `group_id` is required when any objective in `per_output_objective` is a
///     ranking objective.
/// State carried over from a prior joint fit to enable warm-start
/// continuation. Mirrors `WarmStartState` (single-output) and
/// `MultiClassWarmStartState` (multiclass).
#[derive(Debug, Clone, Default)]
pub struct JointWarmStartState {
    /// Per-output baseline predictions from the prior fit (length K).
    pub baselines: Vec<f32>,
    /// Trained stumps from the prior fit. The new fit prepends these
    /// to `all_stumps` and applies their contributions to
    /// `predictions` before the new round loop begins (so the new
    /// trees see the correct residual).
    pub stumps: Vec<TrainedStump>,
    /// How many rounds the prior fit completed. New rounds re-encode
    /// `node_id` starting at `initial_rounds_completed` so the global
    /// `tree_id = node_id / TREE_NODE_STRIDE` mapping stays
    /// non-overlapping.
    pub initial_rounds_completed: usize,
    /// When the prior fit used DART, the per-tree weights (length
    /// `initial_rounds_completed`). `None` means the prior fit was
    /// Standard / GOSS, or the caller wants the engine to reconstruct
    /// weights from per-stump `tree_weight` automatically.
    pub initial_dart_tree_weights: Option<Vec<f32>>,
    /// v0.10.4: EMA snapshot from the prior fit's `MorphState`. `Some(snap)`
    /// seeds the fresh `MorphState::ema_stats` on warm-resume so the
    /// gradient-statistics smoothing is continuous across the resume
    /// boundary — new rounds see the same per-output `(mean, std)` they
    /// would have seen had training never been interrupted.
    ///
    /// **Not byte-equivalent to a fresh longer fit (PR #37 review C3).**
    /// MorphBoost's per-iteration leaf shrinkage
    /// (`1 − morph_rate * round/total`) and LR schedule are resolved
    /// against the `total_iterations` horizon at training time. A prior
    /// fit with `n_estimators=6` baked its first six trees against a
    /// 6-round horizon; resuming with `n_estimators=4` runs the new four
    /// rounds against a 10-round horizon, but the prior six trees keep
    /// their original shrinkage. So a `6+4` warm-resumed MorphBoost fit
    /// does not match a fresh `n_estimators=10` MorphBoost fit
    /// byte-for-byte; the prior trees can't be retroactively re-scaled.
    /// This mirrors the single-output MorphBoost warm-start behavior.
    /// The EMA continuity is the practical guarantee; byte-level
    /// reproducibility across a horizon change is intentionally out of
    /// scope.
    ///
    /// `None` when the prior fit didn't use MorphBoost (the fresh
    /// `MorphState::new` defaults are used instead). Length must equal
    /// `n_outputs` on the new fit, or the warm-start branch returns an
    /// error.
    pub initial_ema_stats: Option<Vec<alloygbm_core::GradientEmaStats>>,
}

/// Compact joint-mode predictor. Builds itself from artifact bytes that were
/// produced by [`fit_joint_multi_output`] (i.e. carry both a `Trees` section
/// and a `MultiOutputLeafValues` section). Each prediction returns a
/// `Vec<f32>` of length `n_outputs`.
#[derive(Debug, Clone)]
pub struct JointPredictor {
    pub n_outputs: usize,
    pub baselines: Vec<f32>,
    /// One entry per stump (in tree order): (split, left_K_values, right_K_values).
    stumps: Vec<JointPredictorStump>,
    /// For each round, the list of stump indices that belong to that round.
    /// Used to walk one tree at a time.
    rounds: Vec<Vec<usize>>,
    /// v0.10.3: per-tree (per-round) weight, read from each tree's first
    /// stump's `tree_weight` field. For non-DART artifacts every entry is
    /// 1.0 and this collapses to the v0.10.2 behavior. For DART artifacts
    /// the weights come from the persisted `DartTreeWeights` section
    /// (kind=11) which `TrainedModel::from_artifact_bytes` applies onto
    /// stumps via `apply_dart_tree_weights`.
    tree_weights: Vec<f32>,
}

#[derive(Debug, Clone)]
struct JointPredictorStump {
    /// Local node id within the tree (0 = root, 1 = left child of root, 2 = right child, etc.).
    local_node_id: u32,
    feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
    /// True if this stump uses a categorical bitset (Fisher-sort) instead of
    /// a numeric threshold compare.
    is_categorical: bool,
    /// Categorical left-bitset (bit `k` = 1 means category `k` is routed
    /// left). Populated only when `is_categorical` is true. Single u64
    /// supports up to 64 categories per feature (the joint trainer's
    /// per-feature cap from the Fisher-sort helper).
    cat_bitset: u64,
    left_k: Vec<f32>,
    right_k: Vec<f32>,
}

impl JointPredictor {
    /// Build a JointPredictor from artifact bytes. The artifact must have
    /// been produced by the joint trainer (i.e. stumps carry
    /// `multi_output_leaf_values`). The per-output `baselines` are passed
    /// separately because the v0.10.0 joint trainer doesn't yet emit a
    /// dedicated baseline-per-output artifact section.
    pub fn from_artifact_bytes(bytes: &[u8], baselines: Vec<f32>) -> Result<Self, String> {
        // Reuse TrainedModel::from_artifact_bytes for full decoding — it
        // already loads MultiOutputLeafValues onto the stumps (v0.10.0+).
        let model = TrainedModel::from_artifact_bytes(bytes).map_err(|e| format!("{e:?}"))?;
        if model
            .stumps
            .iter()
            .all(|s| s.multi_output_leaf_values.is_none())
        {
            return Err("artifact does not carry MultiOutputLeafValues".to_string());
        }
        let n_outputs = model
            .stumps
            .iter()
            .find_map(|s| s.multi_output_leaf_values.as_ref().map(|v| v.0.len()))
            .unwrap_or(0);
        if baselines.len() != n_outputs {
            return Err(format!(
                "baselines length {} != n_outputs {n_outputs}",
                baselines.len()
            ));
        }

        let mut stumps: Vec<JointPredictorStump> = Vec::new();
        let mut rounds: Vec<Vec<usize>> = Vec::new();
        let mut tree_weights: Vec<f32> = Vec::new();
        for stump in model.stumps.iter() {
            let Some(mo) = stump.multi_output_leaf_values.as_ref() else {
                continue;
            };
            let tree_id = (stump.split.node_id / TREE_NODE_STRIDE) as usize;
            let local_node_id = stump.split.node_id % TREE_NODE_STRIDE;
            let new_idx = stumps.len();
            let cat_bitset = stump
                .split
                .categorical_bitset
                .as_ref()
                .map(|b| bitset_bytes_to_u64(b))
                .unwrap_or(0);
            stumps.push(JointPredictorStump {
                local_node_id,
                feature_index: stump.split.feature_index,
                threshold_bin: stump.split.threshold_bin,
                default_left: stump.split.default_left,
                is_categorical: stump.split.is_categorical,
                cat_bitset,
                left_k: mo.0.clone(),
                right_k: mo.1.clone(),
            });
            while rounds.len() <= tree_id {
                rounds.push(Vec::new());
                tree_weights.push(1.0);
            }
            // Record this tree's weight when we see its FIRST stump
            // (matches `apply_dart_tree_weights` in
            // crates/predictor/src/lib.rs:1214 — all stumps in a tree
            // share the same `tree_weight`).
            if rounds[tree_id].is_empty() {
                tree_weights[tree_id] = stump.tree_weight;
            }
            rounds[tree_id].push(new_idx);
        }

        Ok(JointPredictor {
            n_outputs,
            baselines,
            stumps,
            rounds,
            tree_weights,
        })
    }

    /// Predict K outputs for a single row. The returned vector has length
    /// `n_outputs`. Each output is the sum of per-tree leaf contributions
    /// plus the per-output baseline.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut out = self.baselines.clone();
        for (tree_idx, tree_stump_indices) in self.rounds.iter().enumerate() {
            if tree_stump_indices.is_empty() {
                continue;
            }
            // v0.10.3: per-tree DART weight (1.0 for non-DART artifacts).
            let tree_w = *self.tree_weights.get(tree_idx).unwrap_or(&1.0);
            // Build a lookup from local_node_id to stump_index for this tree.
            let mut stumps_by_node: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::with_capacity(tree_stump_indices.len());
            for &si in tree_stump_indices {
                stumps_by_node.insert(self.stumps[si].local_node_id, si);
            }

            // Walk from root (local_node_id = 0) until we fall off the tree at
            // a terminal leaf. At each step we look up the current node's
            // stump; if absent, this node is a terminal leaf and we already
            // accumulated its parent's leaf value at the prior iteration.
            let mut current_node: u32 = 0;
            let mut last_leaf_value: Option<&Vec<f32>> = None;
            loop {
                let Some(&si) = stumps_by_node.get(&current_node) else {
                    break;
                };
                let stump = &self.stumps[si];
                let f = stump.feature_index as usize;
                let v = features.get(f).copied().unwrap_or(f32::NAN);
                let went_left = if v.is_nan() {
                    stump.default_left
                } else if stump.is_categorical {
                    // Categorical stump: route by bitset. The raw feature
                    // value is interpreted as the category ID; bit `cat_id`
                    // of `cat_bitset` decides the direction.
                    let cat_id = v as i64;
                    if !(0..64).contains(&cat_id) {
                        stump.default_left
                    } else {
                        (stump.cat_bitset & (1u64 << cat_id)) != 0
                    }
                } else {
                    (v as i32) <= stump.threshold_bin as i32
                };
                last_leaf_value = Some(if went_left {
                    &stump.left_k
                } else {
                    &stump.right_k
                });
                current_node = if went_left {
                    current_node * 2 + 1
                } else {
                    current_node * 2 + 2
                };
            }
            if let Some(leaf) = last_leaf_value {
                for k in 0..self.n_outputs {
                    out[k] += tree_w * leaf[k];
                }
            }
        }
        out
    }

    /// Predict K outputs for a batch of rows. Returns shape `(n_rows, n_outputs)`
    /// as a flat row-major Vec<f32>.
    pub fn predict_batch(&self, features_flat: &[f32], feature_count: usize) -> Vec<f32> {
        let n_rows = features_flat.len() / feature_count;
        let mut out = Vec::with_capacity(n_rows * self.n_outputs);
        for row in 0..n_rows {
            let row_slice = &features_flat[row * feature_count..(row + 1) * feature_count];
            let preds = self.predict_row(row_slice);
            out.extend_from_slice(&preds);
        }
        out
    }
}
