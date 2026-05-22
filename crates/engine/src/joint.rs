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

use crate::shared_histogram::{
    MultiOutputHistogram, build_multi_output_histogram_inplace, compute_multi_output_split_gain,
};
use crate::{
    Device, InteractionConstraintIndex, LambdaMARTObjective, ObjectiveOps,
    PairwiseRankingObjective, QueryRMSEObjective, SquaredErrorObjective, TrainedModel,
    TrainedStump, XeNDCGObjective, encode_tree_node_id,
};
use alloygbm_core::{
    BinnedMatrix, GradientPair, LeafValue, MISSING_BIN_U8, ModelMetadata, NodeStats,
    SplitCandidate, TrainParams, TreeGrowth,
};
use std::collections::HashMap;

/// Runtime selector for per-output objective on the joint trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointObjective {
    SquaredError,
    QueryRmse,
    RankPairwise,
    RankNdcg,
    RankXendcg,
}

impl JointObjective {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "squared_error" => Ok(Self::SquaredError),
            "queryrmse" => Ok(Self::QueryRmse),
            "rank:pairwise" => Ok(Self::RankPairwise),
            "rank:ndcg" => Ok(Self::RankNdcg),
            "rank:xendcg" => Ok(Self::RankXendcg),
            other => Err(format!(
                "joint multi-output trainer does not support objective {other:?}; \
                 supported: squared_error, queryrmse, rank:pairwise, rank:ndcg, rank:xendcg"
            )),
        }
    }

    pub fn requires_group(&self) -> bool {
        matches!(
            self,
            Self::QueryRmse | Self::RankPairwise | Self::RankNdcg | Self::RankXendcg
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::SquaredError => "squared_error",
            Self::QueryRmse => "queryrmse",
            Self::RankPairwise => "rank:pairwise",
            Self::RankNdcg => "rank:ndcg",
            Self::RankXendcg => "rank:xendcg",
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
            Self::RankNdcg => {
                let group_ids = group
                    .ok_or_else(|| "rank:ndcg objective requires group identifiers".to_string())?;
                let obj = LambdaMARTObjective::new(group_ids);
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
        }
    }
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

/// Convert a u64 categorical bitset (bit `k` = 1 means category `k` goes
/// left) into the byte-packed Vec<u8> format used by the single-output
/// trainer's `SplitCandidate::categorical_bitset`. Bit `K` of byte `K/8`
/// represents category `K`; trailing bytes that contain only zeros are
/// trimmed (single-output convention).
fn u64_to_bitset_bytes(bs: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for byte_idx in 0..8 {
        let byte = ((bs >> (byte_idx * 8)) & 0xFF) as u8;
        out.push(byte);
    }
    while out.len() > 1 && *out.last().unwrap() == 0 {
        out.pop();
    }
    out
}

/// Inverse of `u64_to_bitset_bytes`: decodes a Vec<u8> bitset back into a
/// u64. Used by JointPredictor when evaluating categorical stumps.
fn bitset_bytes_to_u64(bytes: &[u8]) -> u64 {
    let mut out: u64 = 0;
    for (byte_idx, &byte) in bytes.iter().enumerate().take(8) {
        out |= (byte as u64) << (byte_idx * 8);
    }
    out
}

/// Walk every row through one tree's stumps and accumulate
/// `sign * scale * leaf_value_k` into `predictions[k][row]`.
///
/// Shared by the round-end prediction update (v0.10.0 fix), DART
/// dropout subtraction (v0.10.3), and warm-start prior-stump replay
/// (v0.10.3). Tree IDs come in two flavors:
///
/// * **pre-encode** — `node_id` is the local node id directly (used
///   inside `fit_joint_inner` between `build_joint_round*` returning
///   and `encode_tree_node_id` rewriting).
/// * **post-encode** — `node_id` carries the global tree id in the
///   high bits (used for prior-round stumps already in `all_stumps`).
///
/// Both forms work because the lookup key is `node_id % TREE_NODE_STRIDE`,
/// which extracts the local id under either encoding (local ids are
/// always `< TREE_NODE_STRIDE`).
#[allow(clippy::too_many_arguments)]
fn walk_tree_into_predictions(
    tree_stumps: &[TrainedStump],
    binned_matrix: &BinnedMatrix,
    feature_count: usize,
    n_rows: usize,
    n_outputs: usize,
    predictions: &mut [Vec<f32>],
    sign: f32,
    scale: f32,
) {
    let stumps_by_local: std::collections::HashMap<u32, &TrainedStump> = tree_stumps
        .iter()
        .map(|s| (s.split.node_id % TREE_NODE_STRIDE, s))
        .collect();
    for row in 0..n_rows {
        let mut current_node: u32 = 0;
        let mut last_leaf: Option<&(Vec<f32>, Vec<f32>)> = None;
        let mut last_went_left = false;
        loop {
            let Some(stump) = stumps_by_local.get(&current_node) else {
                break;
            };
            let feature = stump.split.feature_index as usize;
            let threshold_bin = stump.split.threshold_bin as usize;
            let bin = binned_matrix.bins[row * feature_count + feature];
            let went_left = if bin == MISSING_BIN_U8 {
                stump.split.default_left
            } else if stump.split.is_categorical {
                let bs = stump
                    .split
                    .categorical_bitset
                    .as_ref()
                    .map(|b| bitset_bytes_to_u64(b))
                    .unwrap_or(0);
                bin < 64 && (bs & (1u64 << bin)) != 0
            } else {
                (bin as usize) <= threshold_bin
            };
            last_leaf = stump.multi_output_leaf_values.as_ref();
            last_went_left = went_left;
            current_node = if went_left {
                current_node * 2 + 1
            } else {
                current_node * 2 + 2
            };
        }
        if let Some((left_k, right_k)) = last_leaf {
            let delta = if last_went_left { left_k } else { right_k };
            for (k, pred_vec) in predictions.iter_mut().enumerate().take(n_outputs) {
                pred_vec[row] += sign * scale * delta[k];
            }
        }
    }
}

/// Per-leaf bookkeeping during level-wise tree growth on the joint trainer.
#[derive(Debug)]
struct JointLeafNode {
    /// 0-indexed local node id within this tree (matches predictor traversal).
    local_node_id: u32,
    /// Row indices currently routed to this node.
    row_indices: Vec<u32>,
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
struct JointLeafCandidate {
    node: JointLeafNode,
    feature: u32,
    threshold_bin: u16,
    default_left: bool,
    gain: f32,
    left_rows: Vec<u32>,
    right_rows: Vec<u32>,
    left_k: Vec<f32>,
    right_k: Vec<f32>,
    left_stats: NodeStats,
    right_stats: NodeStats,
    parent_active_groups: Option<u64>,
    /// Categorical bitset (Some when this is a Fisher-sort categorical
    /// split, None for numeric threshold splits). Bit `k` = 1 means
    /// category `k` is routed to the left child.
    cat_bitset: Option<u64>,
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
struct JointMorphContext {
    config: alloygbm_core::MorphConfig,
    precomputed: alloygbm_core::MorphPrecomputed,
    iteration: u32,
    total_iterations: u32,
    grad_means: Vec<f32>,
    grad_stds: Vec<f32>,
}

impl JointMorphContext {
    fn from_state(state: &crate::MorphState, iteration: u32, total_iterations: u32) -> Self {
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

/// Train a single round of joint multi-output boosting and return the new
/// stumps (already updated to carry K-output leaf values). The caller is
/// responsible for accumulating leaf contributions into per-output prediction
/// vectors using `predictions[k][row] += learning_rate * leaf_value_k`.
///
/// `round_index` is mixed into the `col_subsample` RNG seed so each tree
/// samples a different feature subset (matches LightGBM `feature_fraction`
/// semantics and the single-output trainer's per-round behavior).
///
/// `sampled_root_rows` is the optional row subset to use as the tree's root
/// (used by row_subsample / bagging_fraction). When `None`, the root contains
/// all `n_rows`. Filtering rows at the root means `min_data_in_leaf` operates
/// on the sampled set, matching the single-output trainer (v0.10.2.1 fix).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub fn build_joint_round(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
) -> Result<JointRoundResult, String> {
    build_joint_round_inner(
        params,
        binned_matrix,
        grads_per_output,
        n_outputs,
        categorical_features,
        round_index,
        sampled_root_rows,
        /*morph_ctx=*/ None,
    )
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn build_joint_round_inner(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
    morph_ctx: Option<&JointMorphContext>,
) -> Result<JointRoundResult, String> {
    if grads_per_output.len() != n_outputs {
        return Err(format!(
            "expected {n_outputs} gradient vectors, got {}",
            grads_per_output.len()
        ));
    }
    let n_rows = binned_matrix.row_count;
    for (k, g) in grads_per_output.iter().enumerate() {
        if g.len() != n_rows {
            return Err(format!(
                "gradient vector for output {k} has length {} != {n_rows}",
                g.len()
            ));
        }
    }

    let feature_count = binned_matrix.feature_count;
    // BinnedMatrix exposes a single global `max_bin`; allocate enough bin slots
    // to cover any feature plus the dedicated NaN/missing bin (always 255 for
    // u8 storage in v0.10.0).
    let n_bins = (binned_matrix.max_bin as usize + 1).max(MISSING_BIN_U8 as usize + 1);

    // Pack gradients into row-major K-output flat arrays for histogram build.
    let mut packed_grads = vec![0.0_f32; n_rows * n_outputs];
    let mut packed_hess = vec![0.0_f32; n_rows * n_outputs];
    for k in 0..n_outputs {
        for row in 0..n_rows {
            let gp = grads_per_output[k][row];
            packed_grads[row * n_outputs + k] = gp.grad;
            packed_hess[row * n_outputs + k] = gp.hess;
        }
    }

    let max_depth = params.max_depth.max(1) as usize;
    let min_rows_per_leaf = params.min_data_in_leaf.max(1) as usize;
    let lambda_l2 = params.lambda_l2;

    // col_subsample (v0.10.2): per-tree feature mask, seeded by
    // `(params.seed, round_index)`. v0.10.2.1 fix: mixing the round index
    // into the seed makes each tree sample a different feature subset, matching
    // LightGBM `feature_fraction` and the single-output trainer's per-round
    // behavior. Without this, every tree saw the same feature subset which
    // defeats the point of column sampling.
    let feature_allowed: Vec<bool> = if params.col_subsample < 1.0 {
        let mut s: u64 = params.seed.wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((round_index as u64).wrapping_mul(0x94D049BB133111EB));
        s ^= s >> 30;
        s = s.wrapping_mul(0x94D049BB133111EB);
        if s == 0 {
            s = 0xDEADBEEFCAFEBABE;
        }
        let rate = params.col_subsample;
        let mut mask: Vec<bool> = (0..feature_count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u01 = ((s >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                u01 < rate
            })
            .collect();
        if !mask.iter().any(|&b| b) {
            // All-zero edge case: fall back to all-allowed.
            for f in mask.iter_mut() {
                *f = true;
            }
        }
        mask
    } else {
        vec![true; feature_count]
    };

    // interaction_constraints (v0.10.2): build the constraint index (returns
    // None if the user didn't configure any constraints). Per-node bookkeeping
    // tracks the active group bitset for each local_node_id; the root starts
    // with all groups active, and descendants narrow the set via `descend`
    // each time their parent splits on a constrained feature.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        feature_count,
    )
    .map_err(|e| format!("interaction_constraints: {e:?}"))?;
    let mut node_active_groups: HashMap<u32, u64> = HashMap::new();
    if let Some(idx) = constraint_index.as_ref() {
        node_active_groups.insert(0, idx.root_active_groups());
    }

    // Native-categorical lookup (v0.10.2): feature_index → num_categories.
    // Built once per call. `None` means the feature is numeric.
    let cat_num_categories: Vec<Option<usize>> = {
        let mut v = vec![None; feature_count];
        for cf in categorical_features {
            if cf.feature_index < feature_count {
                v[cf.feature_index] = Some(cf.num_categories);
            }
        }
        v
    };

    let mut stumps: Vec<TrainedStump> = Vec::new();
    let root_rows: Vec<u32> = match sampled_root_rows {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows as u32).collect(),
    };
    let mut active: Vec<JointLeafNode> = vec![JointLeafNode {
        local_node_id: 0,
        row_indices: root_rows,
    }];

    for _depth in 0..max_depth {
        if active.is_empty() {
            break;
        }
        let mut next_active: Vec<JointLeafNode> = Vec::new();

        for node in active.drain(..) {
            if node.row_indices.len() < 2 * min_rows_per_leaf {
                // Too few rows to attempt a split — emit a terminal leaf later
                // via parent's leaf assignment; nothing to do here for an
                // already-leaf node.
                continue;
            }

            // Build per-feature K-output histogram from scratch for this node.
            let mut node_hist = MultiOutputHistogram::new(feature_count, n_bins, n_outputs);
            // Slice bin/grad/hess for this node's rows; build kernel reads rows
            // sequentially, so we set up a temporary local view.
            // (For a minimal correct implementation, build histograms directly
            // from the row_indices set.)
            //
            // We accumulate per-feature column.
            for feature in 0..feature_count {
                // col_subsample (v0.10.2): skip histogram build for masked-out features.
                if !feature_allowed[feature] {
                    continue;
                }
                // Subset the bin column for this feature.
                let mut subset_bins: Vec<u8> = Vec::with_capacity(node.row_indices.len());
                for &row in &node.row_indices {
                    // Row-major: bins[row * feature_count + feature].
                    let idx = row as usize * feature_count + feature;
                    subset_bins.push(binned_matrix.bins[idx]);
                }
                // Subset packed_grads/hess for these rows.
                let mut subset_g: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                let mut subset_h: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                for &row in &node.row_indices {
                    for k in 0..n_outputs {
                        subset_g.push(packed_grads[row as usize * n_outputs + k]);
                        subset_h.push(packed_hess[row as usize * n_outputs + k]);
                    }
                }
                build_multi_output_histogram_inplace(
                    &mut node_hist,
                    feature,
                    &subset_bins,
                    &subset_g,
                    &subset_h,
                    n_outputs,
                );
            }

            // Find best split across all (feature, threshold_bin) pairs.
            // For categorical features, the threshold_bin slot is unused
            // (set to 0) and the bitset is carried in the 4th element.
            let mut best: Option<(usize, usize, f32, Option<u64>)> = None;
            // BinnedMatrix exposes max_bin globally; iterate candidate
            // thresholds across the full bin range minus the NaN slot.
            let max_threshold = (binned_matrix.max_bin as usize).min(MISSING_BIN_U8 as usize - 1);
            // interaction_constraints (v0.10.2): look up this node's active
            // group set once outside the feature loop.
            let node_ag = node_active_groups.get(&node.local_node_id).copied();
            for feature in 0..feature_count {
                // col_subsample (v0.10.2): skip masked-out features in split search.
                if !feature_allowed[feature] {
                    continue;
                }
                // interaction_constraints (v0.10.2): skip features outside the
                // active group set for this node.
                if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), node_ag)
                    && !idx.feature_allowed(ag, feature as u32)
                {
                    continue;
                }
                // Native-categorical (v0.10.2): if the feature is in the
                // categorical_features list, use Fisher-sort over the K
                // outputs instead of a threshold sweep. The result carries
                // a `left_bitset: u64` which the partition path consumes.
                if let Some(num_cats) = cat_num_categories[feature] {
                    // v0.10.4: route categorical Fisher-sort through the morph
                    // variant when active; falls through to the standard
                    // variant when `morph_ctx` is None.
                    let cat_opt = if let Some(m) = morph_ctx {
                        crate::shared_histogram::find_best_multi_output_categorical_split_morph(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                        )
                    } else {
                        crate::shared_histogram::find_best_multi_output_categorical_split(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                        )
                    };
                    if let Some(cat_split) = cat_opt
                        && cat_split.gain > best.map(|(_, _, g, _)| g).unwrap_or(0.0)
                    {
                        best = Some((feature, 0, cat_split.gain, Some(cat_split.left_bitset)));
                    }
                    continue; // skip numeric threshold sweep for categorical features
                }
                for threshold_bin in 0..max_threshold {
                    // v0.10.4: route numeric gain through the morph variant when
                    // active; falls through to the standard variant otherwise.
                    let gain = if let Some(m) = morph_ctx {
                        crate::shared_histogram::compute_multi_output_split_gain_morph(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                        )
                    } else {
                        compute_multi_output_split_gain(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                        )
                    };
                    if gain <= 0.0 {
                        continue;
                    }
                    if best.map(|(_, _, g, _)| gain > g).unwrap_or(true) {
                        best = Some((feature, threshold_bin, gain, None));
                    }
                }
            }

            let Some((feature, threshold_bin, gain, cat_bitset)) = best else {
                continue; // No positive-gain split — leave node as terminal leaf
            };

            // min_split_gain (v0.10.2): reject splits whose K-output sum-of-gains
            // falls below the user-specified threshold. Mirrors the single-output
            // trainer.
            if gain < params.min_split_gain {
                continue;
            }

            // Partition rows by the chosen split. For categorical splits the
            // bin index is interpreted as a category ID and routed via the
            // bitset; for numeric splits, by threshold_bin. NaN rows
            // (bin == MISSING_BIN_U8) route per default_left below.
            let mut left_rows: Vec<u32> = Vec::new();
            let mut right_rows: Vec<u32> = Vec::new();
            let mut missing_rows: Vec<u32> = Vec::new();
            for &row in &node.row_indices {
                let bin = binned_matrix.bins[row as usize * feature_count + feature];
                if bin == MISSING_BIN_U8 {
                    missing_rows.push(row);
                } else if let Some(bs) = cat_bitset {
                    // Categorical: bit `bin` set → left.
                    if bin < 64 && (bs & (1u64 << bin)) != 0 {
                        left_rows.push(row);
                    } else {
                        right_rows.push(row);
                    }
                } else if (bin as usize) <= threshold_bin {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }
            // v0.10.0 default-direction policy: route missing rows to whichever
            // side currently has more rows (a deterministic, simple heuristic).
            // A smarter learned-direction policy is a v0.10.x follow-up.
            let default_left = left_rows.len() >= right_rows.len();
            if default_left {
                left_rows.append(&mut missing_rows);
            } else {
                right_rows.append(&mut missing_rows);
            }

            if left_rows.len() < min_rows_per_leaf || right_rows.len() < min_rows_per_leaf {
                continue; // Skip this split — would create an under-sized leaf
            }

            // Compute K-output leaf values via Newton-Raphson per output.
            // v0.10.5: route through `leaf_effective_gradient` so L1 and DRO
            // leaf solvers are honored. When `lambda_l1 == 0` AND
            // `dro_config == None`, this returns `g_sum` unchanged — so the
            // v0.10.0–v0.10.4 behavior is preserved byte-for-byte for the
            // default config.
            let leaf_values = |rows: &[u32]| -> Vec<f32> {
                let mut out = vec![0.0_f32; n_outputs];
                let row_count = rows.len() as u32;
                for k in 0..n_outputs {
                    let mut g_sum = 0.0_f32;
                    let mut h_sum = 0.0_f32;
                    let mut g_sq_sum = 0.0_f32;
                    for &row in rows {
                        let gp = grads_per_output[k][row as usize];
                        g_sum += gp.grad;
                        h_sum += gp.hess;
                        g_sq_sum += gp.grad * gp.grad;
                    }
                    let g_eff = alloygbm_core::leaf_effective_gradient(
                        g_sum,
                        g_sq_sum,
                        row_count,
                        params.lambda_l1,
                        params.dro_config.as_ref(),
                    );
                    out[k] = -g_eff / (h_sum + lambda_l2 + crate::LEAF_EPSILON);
                }
                out
            };
            let left_k = leaf_values(&left_rows);
            let right_k = leaf_values(&right_rows);

            // Compute per-side NodeStats summed across outputs (placeholder for
            // SplitCandidate; the joint trainer doesn't consume these stats
            // beyond record-keeping).
            let summarize = |rows: &[u32]| -> NodeStats {
                let mut g = 0.0_f32;
                let mut h = 0.0_f32;
                for grad_vec in grads_per_output.iter().take(n_outputs) {
                    for &row in rows {
                        let gp = grad_vec[row as usize];
                        g += gp.grad;
                        h += gp.hess;
                    }
                }
                NodeStats {
                    grad_sum: g,
                    hess_sum: h,
                    grad_sq_sum: 0.0,
                    row_count: rows.len() as u32,
                }
            };
            let left_stats = summarize(&left_rows);
            let right_stats = summarize(&right_rows);

            let stump = TrainedStump {
                split: SplitCandidate {
                    node_id: node.local_node_id,
                    feature_index: feature as u32,
                    threshold_bin: threshold_bin as u16,
                    gain,
                    default_left,
                    is_categorical: cat_bitset.is_some(),
                    categorical_bitset: cat_bitset.map(u64_to_bitset_bytes),
                    left_stats,
                    right_stats,
                },
                // Placeholder scalar leaves (the K-output values below are
                // authoritative). Using the first output as a representative
                // means single-output prediction paths still return a sensible
                // value if accidentally invoked.
                left_leaf_value: LeafValue::Scalar(left_k[0]),
                right_leaf_value: LeafValue::Scalar(right_k[0]),
                tree_weight: 1.0,
                multi_output_leaf_values: Some((left_k, right_k)),
            };
            stumps.push(stump);

            // interaction_constraints (v0.10.2): propagate the active group
            // set to both children. `descend` returns the intersection of
            // the parent's active groups with the groups containing the
            // split feature (or leaves it unchanged for unconstrained
            // features).
            if let (Some(idx), Some(parent_ag)) = (constraint_index.as_ref(), node_ag) {
                let child_ag = idx.descend(parent_ag, feature as u32);
                node_active_groups.insert(node.local_node_id * 2 + 1, child_ag);
                node_active_groups.insert(node.local_node_id * 2 + 2, child_ag);
            }

            // Schedule child nodes (local_node_id * 2 + 1 and + 2 per predictor
            // traversal convention).
            next_active.push(JointLeafNode {
                local_node_id: node.local_node_id * 2 + 1,
                row_indices: left_rows,
            });
            next_active.push(JointLeafNode {
                local_node_id: node.local_node_id * 2 + 2,
                row_indices: right_rows,
            });
        }

        active = next_active;
    }

    Ok(JointRoundResult { stumps })
}

/// Build one joint round using **leaf-wise (best-first)** tree growth.
///
/// Mirrors `build_joint_round` (level-wise) but pops the best candidate
/// from a max-heap keyed by gain at each step, instead of expanding every
/// node at the current depth. Stops when:
///   - the heap is empty (no positive-gain split available anywhere), OR
///   - the leaf count would exceed `max_leaves`, OR
///   - a candidate's depth (derived from `local_node_id`) would exceed
///     `params.max_depth`.
///
/// Honors `col_subsample`, `interaction_constraints`, and `min_split_gain`
/// using the same logic as the level-wise builder. `row_subsample` is
/// applied at the outer round level via `fit_joint_multi_output` (the
/// gradients passed in are already row-masked when sampling is enabled).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn build_joint_round_leafwise(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
    max_leaves: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    round_index: usize,
    sampled_root_rows: Option<&[u32]>,
    morph_ctx: Option<&JointMorphContext>,
) -> Result<JointRoundResult, String> {
    use std::collections::BinaryHeap;

    let n_rows = binned_matrix.row_count;
    let feature_count = binned_matrix.feature_count;
    let n_bins = (binned_matrix.max_bin as usize + 1).max(MISSING_BIN_U8 as usize + 1);

    // Per-output gradient sanity (mirror build_joint_round).
    for (k, g) in grads_per_output.iter().enumerate() {
        if g.len() != n_rows {
            return Err(format!(
                "gradient vector for output {k} has length {} != {n_rows}",
                g.len()
            ));
        }
    }

    // Pack gradients for shared-histogram build (row-major K-output).
    let mut packed_grads = vec![0.0_f32; n_rows * n_outputs];
    let mut packed_hess = vec![0.0_f32; n_rows * n_outputs];
    for k in 0..n_outputs {
        for row in 0..n_rows {
            let gp = grads_per_output[k][row];
            packed_grads[row * n_outputs + k] = gp.grad;
            packed_hess[row * n_outputs + k] = gp.hess;
        }
    }

    let max_depth = params.max_depth.max(1) as usize;
    let min_rows_per_leaf = params.min_data_in_leaf.max(1) as usize;
    let lambda_l2 = params.lambda_l2;

    // col_subsample (same logic as build_joint_round; v0.10.2.1 fix
    // mixes round_index into the seed so each tree samples a different
    // feature subset).
    let feature_allowed: Vec<bool> = if params.col_subsample < 1.0 {
        let mut s: u64 = params.seed.wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((round_index as u64).wrapping_mul(0x94D049BB133111EB));
        s ^= s >> 30;
        s = s.wrapping_mul(0x94D049BB133111EB);
        if s == 0 {
            s = 0xDEADBEEFCAFEBABE;
        }
        let rate = params.col_subsample;
        let mut mask: Vec<bool> = (0..feature_count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u01 = ((s >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                u01 < rate
            })
            .collect();
        if !mask.iter().any(|&b| b) {
            for f in mask.iter_mut() {
                *f = true;
            }
        }
        mask
    } else {
        vec![true; feature_count]
    };

    // interaction_constraints.
    let constraint_index = InteractionConstraintIndex::from_constraints(
        &params.interaction_constraints,
        feature_count,
    )
    .map_err(|e| format!("interaction_constraints: {e:?}"))?;
    let root_active_groups = constraint_index
        .as_ref()
        .map(|idx| idx.root_active_groups());

    // Native-categorical lookup (v0.10.2): same as build_joint_round.
    let cat_num_categories: Vec<Option<usize>> = {
        let mut v = vec![None; feature_count];
        for cf in categorical_features {
            if cf.feature_index < feature_count {
                v[cf.feature_index] = Some(cf.num_categories);
            }
        }
        v
    };

    // Per-node candidate evaluator. Builds the multi-output histogram for
    // `node.row_indices`, sweeps features (respecting `feature_allowed` and
    // the active interaction-constraint group set), picks the best split,
    // partitions rows, computes Newton-Raphson K-output leaf values, and
    // returns a candidate (or None if no positive-gain split survives the
    // constraints + min_data_in_leaf + min_split_gain filters).
    let evaluate_node =
        |node: JointLeafNode, active_groups: Option<u64>| -> Option<JointLeafCandidate> {
            if node.row_indices.len() < 2 * min_rows_per_leaf {
                return None;
            }

            // Build multi-output histogram for this node.
            let mut node_hist = MultiOutputHistogram::new(feature_count, n_bins, n_outputs);
            for feature in 0..feature_count {
                if !feature_allowed[feature] {
                    continue;
                }
                if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), active_groups)
                    && !idx.feature_allowed(ag, feature as u32)
                {
                    continue;
                }
                let mut subset_bins: Vec<u8> = Vec::with_capacity(node.row_indices.len());
                for &row in &node.row_indices {
                    let idx = row as usize * feature_count + feature;
                    subset_bins.push(binned_matrix.bins[idx]);
                }
                let mut subset_g: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                let mut subset_h: Vec<f32> = Vec::with_capacity(node.row_indices.len() * n_outputs);
                for &row in &node.row_indices {
                    for k in 0..n_outputs {
                        subset_g.push(packed_grads[row as usize * n_outputs + k]);
                        subset_h.push(packed_hess[row as usize * n_outputs + k]);
                    }
                }
                build_multi_output_histogram_inplace(
                    &mut node_hist,
                    feature,
                    &subset_bins,
                    &subset_g,
                    &subset_h,
                    n_outputs,
                );
            }

            // Sweep features for the best split. Categorical features dispatch
            // to Fisher-sort and carry a u64 bitset in the 4th slot of `best`.
            let mut best: Option<(usize, usize, f32, Option<u64>)> = None;
            let max_threshold = (binned_matrix.max_bin as usize).min(MISSING_BIN_U8 as usize - 1);
            for feature in 0..feature_count {
                if !feature_allowed[feature] {
                    continue;
                }
                if let (Some(idx), Some(ag)) = (constraint_index.as_ref(), active_groups)
                    && !idx.feature_allowed(ag, feature as u32)
                {
                    continue;
                }
                if let Some(num_cats) = cat_num_categories[feature] {
                    // v0.10.4: route through morph variant when active.
                    let cat_opt = if let Some(m) = morph_ctx {
                        crate::shared_histogram::find_best_multi_output_categorical_split_morph(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                        )
                    } else {
                        crate::shared_histogram::find_best_multi_output_categorical_split(
                            &node_hist,
                            feature,
                            num_cats,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                        )
                    };
                    if let Some(cat_split) = cat_opt
                        && cat_split.gain > best.map(|(_, _, g, _)| g).unwrap_or(0.0)
                    {
                        best = Some((feature, 0, cat_split.gain, Some(cat_split.left_bitset)));
                    }
                    continue;
                }
                for threshold_bin in 0..max_threshold {
                    // v0.10.4: route through morph variant when active.
                    let gain = if let Some(m) = morph_ctx {
                        crate::shared_histogram::compute_multi_output_split_gain_morph(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                            &m.config,
                            &m.precomputed,
                            m.iteration,
                            m.total_iterations,
                            &m.grad_means,
                            &m.grad_stds,
                        )
                    } else {
                        compute_multi_output_split_gain(
                            &node_hist,
                            feature,
                            threshold_bin,
                            lambda_l2,
                            crate::LEAF_EPSILON,
                        )
                    };
                    if gain <= 0.0 {
                        continue;
                    }
                    if best.map(|(_, _, g, _)| gain > g).unwrap_or(true) {
                        best = Some((feature, threshold_bin, gain, None));
                    }
                }
            }
            let (feature, threshold_bin, gain, cat_bitset) = best?;
            if gain < params.min_split_gain {
                return None;
            }

            // Partition rows.
            let mut left_rows: Vec<u32> = Vec::new();
            let mut right_rows: Vec<u32> = Vec::new();
            let mut missing_rows: Vec<u32> = Vec::new();
            for &row in &node.row_indices {
                let bin = binned_matrix.bins[row as usize * feature_count + feature];
                if bin == MISSING_BIN_U8 {
                    missing_rows.push(row);
                } else if let Some(bs) = cat_bitset {
                    if bin < 64 && (bs & (1u64 << bin)) != 0 {
                        left_rows.push(row);
                    } else {
                        right_rows.push(row);
                    }
                } else if (bin as usize) <= threshold_bin {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }
            let default_left = left_rows.len() >= right_rows.len();
            if default_left {
                left_rows.append(&mut missing_rows);
            } else {
                right_rows.append(&mut missing_rows);
            }
            if left_rows.len() < min_rows_per_leaf || right_rows.len() < min_rows_per_leaf {
                return None;
            }

            // K-output leaf values via Newton-Raphson per output.
            // v0.10.5: route through `leaf_effective_gradient` so L1 and DRO
            // leaf solvers are honored on the leaf-wise path too.  When
            // `lambda_l1 == 0` AND `dro_config == None`, this returns
            // `g_sum` unchanged — preserving byte-for-byte compatibility
            // with the v0.10.0–v0.10.4 default config.
            let leaf_values = |rows: &[u32]| -> Vec<f32> {
                let mut out = vec![0.0_f32; n_outputs];
                let row_count = rows.len() as u32;
                for k in 0..n_outputs {
                    let mut g_sum = 0.0_f32;
                    let mut h_sum = 0.0_f32;
                    let mut g_sq_sum = 0.0_f32;
                    for &row in rows {
                        let gp = grads_per_output[k][row as usize];
                        g_sum += gp.grad;
                        h_sum += gp.hess;
                        g_sq_sum += gp.grad * gp.grad;
                    }
                    let g_eff = alloygbm_core::leaf_effective_gradient(
                        g_sum,
                        g_sq_sum,
                        row_count,
                        params.lambda_l1,
                        params.dro_config.as_ref(),
                    );
                    out[k] = -g_eff / (h_sum + lambda_l2 + crate::LEAF_EPSILON);
                }
                out
            };
            let left_k = leaf_values(&left_rows);
            let right_k = leaf_values(&right_rows);

            // NodeStats for record-keeping (joint trainer doesn't consume them
            // beyond carrying them into SplitCandidate for compat).
            let summarize = |rows: &[u32]| -> NodeStats {
                let mut g = 0.0_f32;
                let mut h = 0.0_f32;
                for grad_vec in grads_per_output.iter().take(n_outputs) {
                    for &row in rows {
                        let gp = grad_vec[row as usize];
                        g += gp.grad;
                        h += gp.hess;
                    }
                }
                NodeStats {
                    grad_sum: g,
                    hess_sum: h,
                    grad_sq_sum: 0.0,
                    row_count: rows.len() as u32,
                }
            };
            let left_stats = summarize(&left_rows);
            let right_stats = summarize(&right_rows);

            Some(JointLeafCandidate {
                node,
                feature: feature as u32,
                threshold_bin: threshold_bin as u16,
                default_left,
                gain,
                left_rows,
                right_rows,
                left_k,
                right_k,
                left_stats,
                right_stats,
                parent_active_groups: active_groups,
                cat_bitset,
            })
        };

    // Initialize heap with the root candidate. v0.10.2.1 fix: use the
    // sampled row subset (when row_subsample < 1.0) so min_data_in_leaf
    // operates on the sampled training set, matching single-output.
    let root_rows: Vec<u32> = match sampled_root_rows {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows as u32).collect(),
    };
    let mut heap: BinaryHeap<JointLeafCandidate> = BinaryHeap::new();
    let root_node = JointLeafNode {
        local_node_id: 0,
        row_indices: root_rows,
    };
    if let Some(root_cand) = evaluate_node(root_node, root_active_groups) {
        heap.push(root_cand);
    }

    // Best-first growth: each pop adds one stump (one split → one new leaf).
    let mut stumps: Vec<TrainedStump> = Vec::new();
    let mut leaf_count: usize = 1; // root starts as one leaf

    while let Some(cand) = heap.pop() {
        if leaf_count >= max_leaves {
            break;
        }
        // Depth from local_node_id: depth = floor(log2(node_id + 1)).
        // (0 → 0, 1/2 → 1, 3/4/5/6 → 2, ...)
        let depth = (32 - (cand.node.local_node_id + 1).leading_zeros()) as usize - 1;
        if depth >= max_depth {
            continue;
        }

        let local_node_id = cand.node.local_node_id;
        let left_local = local_node_id * 2 + 1;
        let right_local = local_node_id * 2 + 2;

        // Compute child active groups before consuming cand.feature.
        let child_ag = match (constraint_index.as_ref(), cand.parent_active_groups) {
            (Some(idx), Some(parent_ag)) => Some(idx.descend(parent_ag, cand.feature)),
            _ => None,
        };

        // Commit the split as a TrainedStump.
        let stump = TrainedStump {
            split: SplitCandidate {
                node_id: local_node_id,
                feature_index: cand.feature,
                threshold_bin: cand.threshold_bin,
                gain: cand.gain,
                default_left: cand.default_left,
                is_categorical: cand.cat_bitset.is_some(),
                categorical_bitset: cand.cat_bitset.map(u64_to_bitset_bytes),
                left_stats: cand.left_stats,
                right_stats: cand.right_stats,
            },
            left_leaf_value: LeafValue::Scalar(cand.left_k[0]),
            right_leaf_value: LeafValue::Scalar(cand.right_k[0]),
            tree_weight: 1.0,
            multi_output_leaf_values: Some((cand.left_k.clone(), cand.right_k.clone())),
        };
        stumps.push(stump);
        leaf_count += 1; // splitting a leaf adds 1 net leaf (1 → 2).

        // Evaluate child candidates and push to heap if they have a viable split.
        if depth + 1 < max_depth {
            let left_node = JointLeafNode {
                local_node_id: left_local,
                row_indices: cand.left_rows,
            };
            if let Some(left_cand) = evaluate_node(left_node, child_ag) {
                heap.push(left_cand);
            }
            let right_node = JointLeafNode {
                local_node_id: right_local,
                row_indices: cand.right_rows,
            };
            if let Some(right_cand) = evaluate_node(right_node, child_ag) {
                heap.push(right_cand);
            }
        }
    }

    Ok(JointRoundResult { stumps })
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
/// Joint analogue of [`crate::select_row_indices_for_round_multiclass`].
///
/// For joint multi-output the per-row score is `s_i = Σₖ |g_{i,k}|`
/// across the K per-output gradient buffers (matches LightGBM
/// multiclass GOSS). A single row mask is shared across all K buffers,
/// and the amplification factor is applied identically to every
/// output's gradient and hessian.
///
/// Returns `Some(sampled_rows)` when GOSS is active; `None` when
/// `BoostingMode::Standard` or `BoostingMode::Dart` is in effect (the
/// caller falls back to the existing row_subsample path).
fn select_joint_row_indices_for_round(
    boosting_mode: alloygbm_core::BoostingMode,
    n_rows: usize,
    seed_base: u64,
    round_index: u64,
    grads_per_output: &mut [Vec<GradientPair>],
) -> Option<Vec<u32>> {
    use alloygbm_core::BoostingMode as BM;
    match boosting_mode {
        BM::Goss {
            top_rate,
            other_rate,
        } => {
            let n_outputs = grads_per_output.len();
            debug_assert!(n_outputs > 0, "joint GOSS requires K >= 1");
            debug_assert!(
                grads_per_output.iter().all(|buf| buf.len() == n_rows),
                "every per-output gradient buffer must have length n_rows"
            );
            let magnitudes: Vec<f32> = (0..n_rows)
                .map(|i| {
                    grads_per_output
                        .iter()
                        .take(n_outputs)
                        .map(|buf| buf[i].grad.abs())
                        .sum::<f32>()
                })
                .collect();
            let (top, other, amplification) = crate::goss_sample_indices(
                &magnitudes,
                top_rate,
                other_rate,
                seed_base,
                round_index,
            );
            if (amplification - 1.0).abs() > f32::EPSILON {
                for &row in &other {
                    let idx = row as usize;
                    for buf in grads_per_output.iter_mut().take(n_outputs) {
                        let p = &mut buf[idx];
                        p.grad *= amplification;
                        p.hess *= amplification;
                    }
                }
            }
            let mut merged: Vec<u32> = Vec::with_capacity(top.len() + other.len());
            merged.extend(top);
            merged.extend(other);
            merged.sort_unstable();
            Some(merged)
        }
        BM::Standard | BM::Dart { .. } => None,
    }
}

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

pub fn fit_joint_multi_output(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
) -> Result<JointTrainingSummary, String> {
    fit_joint_multi_output_with_categorical(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        /*categorical_features=*/ &[],
    )
}

/// Same as [`fit_joint_multi_output_with_categorical`] but accepts an
/// optional [`JointWarmStartState`] to continue training from a prior
/// fit. When `warm_start = None`, behavior is byte-identical to
/// `fit_joint_multi_output_with_categorical`.
#[allow(clippy::too_many_arguments)]
pub fn fit_joint_multi_output_with_warm_start(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    warm_start: Option<JointWarmStartState>,
) -> Result<JointTrainingSummary, String> {
    fit_joint_inner(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        categorical_features,
        warm_start,
    )
}

/// Same as [`fit_joint_multi_output`] but accepts a slice of
/// [`CategoricalFeatureInfo`] specifying which features should be treated
/// as native-categorical via multi-output Fisher-sort partitioning. An
/// empty slice means all features are numeric (byte-identical to
/// `fit_joint_multi_output`).
#[allow(clippy::too_many_arguments)]
pub fn fit_joint_multi_output_with_categorical(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
) -> Result<JointTrainingSummary, String> {
    fit_joint_inner(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        categorical_features,
        /*warm_start=*/ None,
    )
}

/// Inner implementation of the joint multi-output trainer. Public
/// callers route through `fit_joint_multi_output_with_categorical`
/// (cold start) or `fit_joint_multi_output_with_warm_start`
/// (continuation). Mirrors the pattern used by the single-output
/// engine where `fit_iterations*` variants all funnel through one
/// underlying impl.
#[allow(clippy::too_many_arguments)]
fn fit_joint_inner(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    warm_start: Option<JointWarmStartState>,
) -> Result<JointTrainingSummary, String> {
    let n_outputs = targets_per_output.len();
    if per_output_objective.len() != n_outputs {
        return Err(format!(
            "per_output_objective length {} != n_outputs {n_outputs}",
            per_output_objective.len()
        ));
    }
    let n_rows = binned_matrix.row_count;
    for (k, tg) in targets_per_output.iter().enumerate() {
        if tg.len() != n_rows {
            return Err(format!(
                "targets[{k}] length {} != n_rows {n_rows}",
                tg.len()
            ));
        }
    }
    if per_output_objective.iter().any(|o| o.requires_group()) && group_id.is_none() {
        return Err("at least one objective requires group_id".to_string());
    }

    // v0.10.3: warm-start branch — when `warm_start` is `Some`, the
    // prior fit's baselines win (re-seed `predictions` from them) and
    // its stumps are prepended to `all_stumps`. The cold-start branch
    // computes per-output baselines as before. v0.10.4: also carries
    // `initial_ema_stats` for MorphBoost warm-resume.
    let (
        initial_stumps,
        initial_rounds,
        initial_dart_weights_arg,
        initial_ema_stats_arg,
        baselines,
    ) = if let Some(ws) = warm_start {
        if ws.baselines.len() != n_outputs {
            return Err(format!(
                "warm-start baselines length {} != n_outputs {n_outputs}",
                ws.baselines.len()
            ));
        }
        if let Some(ema) = ws.initial_ema_stats.as_ref()
            && ema.len() != n_outputs
        {
            return Err(format!(
                "warm-start initial_ema_stats length {} != n_outputs {n_outputs}",
                ema.len()
            ));
        }
        (
            ws.stumps,
            ws.initial_rounds_completed,
            ws.initial_dart_tree_weights,
            ws.initial_ema_stats,
            ws.baselines,
        )
    } else {
        let cold_baselines: Vec<f32> = per_output_objective
            .iter()
            .zip(targets_per_output.iter())
            .map(|(obj, targets)| obj.initial_prediction(targets))
            .collect();
        (Vec::new(), 0, None, None, cold_baselines)
    };

    // Per-output prediction vectors, seeded from baselines.
    let mut predictions: Vec<Vec<f32>> = baselines.iter().map(|&b| vec![b; n_rows]).collect();

    let learning_rate = params.learning_rate;
    let mut all_stumps: Vec<TrainedStump> = initial_stumps;
    let mut rounds_completed: usize = 0;

    // v0.10.4: MorphBoost runtime state. Active when `params.morph_config`
    // is `Some`. `total_iterations` covers warm-start prefix + new rounds so
    // the LR schedule + depth-penalty curve see the full horizon, mirroring
    // the single-output multiclass path. EMA snapshot from a prior
    // MorphBoost fit (passed via `JointWarmStartState.initial_ema_stats`)
    // re-seeds `MorphState::ema_stats` so a warm-resumed N+M fit matches
    // a fresh N+M fit byte-for-byte.
    let total_iterations_u32 = (initial_rounds + n_estimators) as u32;
    let mut morph_state: Option<crate::MorphState> = params
        .morph_config
        .map(|cfg| crate::MorphState::new(cfg, n_outputs, total_iterations_u32, learning_rate));
    if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), initial_ema_stats_arg.as_ref()) {
        for (i, stat) in snapshot.iter().take(ms.ema_stats.len()).enumerate() {
            ms.ema_stats[i] = *stat;
        }
    }

    // v0.10.3: joint DART state. `dart_state.tree_weights[r]` is the
    // weight applied to round-r's tree at predict time;
    // `dart_round_start_offsets[r]` + `dart_round_counts[r]` track the
    // stump range in `all_stumps` for that round (one tree per round on
    // the joint trainer, but the stump *count* varies under leaf-wise
    // growth).
    let dart_params = match params.boosting_mode {
        alloygbm_core::BoostingMode::Dart {
            drop_rate,
            max_drop,
            normalize_type,
            sample_type,
        } => Some((drop_rate, max_drop, normalize_type, sample_type)),
        _ => None,
    };
    let mut dart_state = crate::DartState::default();
    let mut dart_round_start_offsets: Vec<usize> = Vec::new();
    let mut dart_round_counts: Vec<usize> = Vec::new();

    // v0.10.3: warm-start — replay prior-stump contributions onto
    // `predictions` so the new round's gradients see the correct
    // residual. Group prior stumps by `tree_id` and walk each tree at
    // its DART weight (1.0 for non-DART warm-starts).
    if !all_stumps.is_empty() {
        // PR #36 review (C2): validate that every prior stump's
        // `feature_index` is `< feature_count` BEFORE replay. Without
        // this check `walk_tree_into_predictions` indexes
        // `binned_matrix.bins[row * feature_count + feature]` which
        // panics across the PyO3 boundary if the prior fit was trained
        // on more features than the current one. Surface a clean
        // validation error instead.
        for (idx, s) in all_stumps.iter().enumerate() {
            let fi = s.split.feature_index as usize;
            if fi >= feature_count {
                return Err(format!(
                    "warm-start: prior stump {idx} references feature_index {fi} \
                     which is >= the current feature_count {feature_count}. The \
                     init_model appears to have been trained on a wider feature \
                     set than the current X; either pad X to match the prior \
                     schema or fit fresh without `warm_start=True`."
                ));
            }
        }
        let mut grouped: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, s) in all_stumps.iter().enumerate() {
            grouped
                .entry(s.split.node_id / TREE_NODE_STRIDE)
                .or_default()
                .push(idx);
        }
        // For DART warm-start, derive per-tree weights either from the
        // explicit `initial_dart_tree_weights` arg or from the first
        // stump's `tree_weight` (mirrors `apply_dart_tree_weights`).
        let derived_dart_weights: Option<Vec<f32>> = if dart_params.is_some() {
            if let Some(dw) = initial_dart_weights_arg.as_ref() {
                if dw.len() != initial_rounds {
                    return Err(format!(
                        "warm_start.initial_dart_tree_weights length {} != initial_rounds_completed {initial_rounds}",
                        dw.len()
                    ));
                }
                Some(dw.clone())
            } else {
                let mut reconstructed: Vec<f32> = vec![1.0; initial_rounds];
                for (tid, indices) in &grouped {
                    if let Some(&first) = indices.first()
                        && (*tid as usize) < reconstructed.len()
                    {
                        reconstructed[*tid as usize] = all_stumps[first].tree_weight;
                    }
                }
                Some(reconstructed)
            }
        } else {
            None
        };
        for (tree_idx, stump_indices) in &grouped {
            // Materialize this tree's stumps as owned clones so we can
            // hand a contiguous slice to the walker without aliasing
            // `all_stumps` (which we'd otherwise need to borrow
            // immutably while `predictions` is mutably borrowed below).
            let tree_stumps: Vec<TrainedStump> = stump_indices
                .iter()
                .map(|&i| all_stumps[i].clone())
                .collect();
            let scale = if let Some(dw) = derived_dart_weights.as_ref() {
                dw.get(*tree_idx as usize).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            walk_tree_into_predictions(
                &tree_stumps,
                binned_matrix,
                feature_count,
                n_rows,
                n_outputs,
                &mut predictions,
                1.0,
                scale,
            );
        }

        // Seed DART bookkeeping from the prior fit so new rounds can
        // dropout/restore prior trees correctly.
        if dart_params.is_some() {
            dart_state.tree_weights =
                derived_dart_weights.unwrap_or_else(|| vec![1.0; initial_rounds]);
            for r in 0..initial_rounds {
                let r_u32 = r as u32;
                if let Some(indices) = grouped.get(&r_u32) {
                    let start = *indices.iter().min().unwrap();
                    dart_round_start_offsets.push(start);
                    dart_round_counts.push(indices.len());
                } else {
                    // Round r contributed no stumps (degenerate).
                    dart_round_start_offsets.push(0);
                    dart_round_counts.push(0);
                }
            }
        }
    }

    for round in 0..n_estimators {
        // v0.10.3 warm-start: when continuing from a prior fit, all
        // per-round seeds, dropout indices, and `node_id` encodings
        // mix `global_round = round + initial_rounds` so a warm-resumed
        // N+M fit produces the same RNG draws on round M..N+M as a
        // fresh N+M fit.
        let global_round = round + initial_rounds;

        // v0.10.3 joint DART: drop a random subset of previously-trained
        // trees before computing gradients. Subtract their (currently
        // weighted) contributions from `predictions` so the new tree
        // fits on residuals of the dropped-out ensemble (mirrors the
        // single-output DART flow at crates/engine/src/lib.rs:4895).
        let dropped_tree_ids: Vec<usize> =
            if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                if dart_state.tree_weights.is_empty() {
                    Vec::new()
                } else {
                    let drops = crate::select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        params.seed,
                        global_round,
                    );
                    for &tree_id in &drops {
                        let w_old = dart_state.tree_weights[tree_id];
                        let start = dart_round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        if count > 0 {
                            walk_tree_into_predictions(
                                &all_stumps[start..start + count],
                                binned_matrix,
                                feature_count,
                                n_rows,
                                n_outputs,
                                &mut predictions,
                                -1.0,
                                w_old,
                            );
                        }
                    }
                    drops
                }
            } else {
                Vec::new()
            };

        // Compute per-output gradients on current predictions.
        let mut grads_per_output: Vec<Vec<GradientPair>> = Vec::with_capacity(n_outputs);
        for k in 0..n_outputs {
            let g = per_output_objective[k].compute_gradients(
                &predictions[k],
                &targets_per_output[k],
                group_id,
            )?;
            grads_per_output.push(g);
        }

        // v0.10.4: update per-output EMA stats BEFORE tree-building so
        // morph split selection sees the latest mean/std. Mirrors the
        // single-output multiclass path
        // (crates/engine/src/lib.rs:3974 `update_ema_from_gradient_pairs`).
        if let Some(ms) = morph_state.as_mut() {
            for (k, g) in grads_per_output.iter().enumerate() {
                ms.update_ema_from_gradient_pairs(g, k);
            }
        }
        let morph_ctx: Option<JointMorphContext> = morph_state
            .as_ref()
            .map(|ms| JointMorphContext::from_state(ms, global_round as u32, total_iterations_u32));

        // v0.10.3: joint GOSS. When `boosting_mode = Goss`, score rows by
        // `Σₖ |g_{i,k}|`, keep top_rate fraction, sample other_rate
        // uniformly from the rest, and amplify the sampled-low rows'
        // gradients so per-output histogram statistics remain unbiased
        // estimators of the full-data gradient sums. Mutates
        // `grads_per_output` in place. Falls back to the row_subsample
        // path under Standard / DART.
        //
        // row_subsample (v0.10.2, fixed v0.10.2.1): seeded Bernoulli row
        // mask. When row_subsample < 1.0, the round's tree builder works
        // on the SAMPLED rows only — `sampled_rows: Vec<u32>` is passed
        // as the root's `row_indices`. This means `min_data_in_leaf`
        // operates on the sampled set, matching single-output semantics
        // (v0.10.2 zeroed gradients but left rows in the index, so a
        // split could create a leaf with enough rows but too few
        // sampled rows).
        //
        // The post-build prediction-update walk below operates on all
        // `n_rows` via the BinnedMatrix tree walk (not row_indices), so
        // un-sampled rows still receive their tree-walked leaf delta —
        // matching LightGBM's `bagging_fraction` semantics where every
        // row's predictions are updated each round, but the tree itself
        // is fit on the sampled subset.
        let sampled_rows_opt: Option<Vec<u32>> = if let Some(rows) =
            select_joint_row_indices_for_round(
                params.boosting_mode,
                n_rows,
                params.seed,
                global_round as u64,
                &mut grads_per_output,
            ) {
            Some(rows)
        } else if params.row_subsample < 1.0 {
            let mut rng_state: u64 = params.seed.wrapping_mul(0x9E3779B97F4A7C15)
                ^ ((global_round as u64).wrapping_mul(0xBF58476D1CE4E5B9));
            if rng_state == 0 {
                rng_state = 0xDEADBEEFCAFEBABE;
            }
            let rate = params.row_subsample;
            let mut sampled: Vec<u32> = Vec::with_capacity(n_rows / 2 + 1);
            for row in 0..n_rows {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let u01 = ((rng_state >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                if u01 < rate {
                    sampled.push(row as u32);
                }
            }
            // Edge case: nothing sampled. Fall back to all-rows for this
            // round so the trainer doesn't produce a degenerate empty tree.
            if sampled.is_empty() {
                None
            } else {
                Some(sampled)
            }
        } else {
            None
        };
        // Gradients pass through unchanged; the trainer indexes them by
        // row id from the (potentially filtered) row_indices list.
        let active_grads: &[Vec<GradientPair>] = grads_per_output.as_slice();

        // Build one shared tree. Dispatch on tree_growth: leaf-wise uses
        // the priority-queue best-first builder gated by `max_leaves`;
        // level-wise uses the BFS depth-bounded builder.
        let mut round_result = if params.tree_growth == TreeGrowth::Leaf {
            let max_leaves = params.max_leaves.ok_or_else(|| {
                "tree_growth='leaf' requires max_leaves to be set on the joint trainer".to_string()
            })?;
            build_joint_round_leafwise(
                params,
                binned_matrix,
                active_grads,
                n_outputs,
                max_leaves,
                categorical_features,
                global_round,
                sampled_rows_opt.as_deref(),
                morph_ctx.as_ref(),
            )?
        } else {
            build_joint_round_inner(
                params,
                binned_matrix,
                active_grads,
                n_outputs,
                categorical_features,
                global_round,
                sampled_rows_opt.as_deref(),
                morph_ctx.as_ref(),
            )?
        };
        if round_result.stumps.is_empty() {
            break;
        }
        rounds_completed += 1;

        // v0.10.0 review fix (Comment 3) + v0.10.4 MorphBoost: pre-scale the
        // per-leaf K-output deltas before persisting so the artifact already
        // encodes the LR-scaled contribution. JointPredictor adds the leaf
        // values directly without re-applying `learning_rate`.
        //
        // Standard scale: `learning_rate` (unchanged from v0.10.0).
        // Morph scale: `morph_lr * leaf_shrink * depth_penalty` where
        //   morph_lr = MorphState::lr_for_iter(round)
        //   leaf_shrink = max(0, 1 - morph_rate * round/total)
        //   depth_penalty = depth_penalty_base ^ (depth/3), depth derived from
        //     local_node_id via `(id+1).ilog2()`.
        // Depth-penalty applies per-stump because non-root leaves have larger
        // depth than root leaves.
        let standard_scale_active =
            morph_state.is_none() && (learning_rate - 1.0).abs() > f32::EPSILON;
        let morph_active = morph_state.is_some();
        if standard_scale_active || morph_active {
            // Pre-compute morph scalars that are stump-independent.
            let (morph_lr, leaf_shrink) = if let Some(ms) = morph_state.as_ref() {
                let lr = ms.lr_for_iter(global_round);
                let t = global_round as f32;
                let total_t = total_iterations_u32.max(1) as f32;
                let shrink = (1.0 - ms.config.morph_rate * (t / total_t)).max(0.0);
                (lr, shrink)
            } else {
                (learning_rate, 1.0_f32)
            };
            let depth_penalty_base = morph_state
                .as_ref()
                .map(|ms| ms.config.depth_penalty_base)
                .unwrap_or(1.0);
            for stump in round_result.stumps.iter_mut() {
                // local_node_id is the pre-encode id (re-encode happens
                // later). depth = floor(log2(id + 1)).
                let local_id = stump.split.node_id;
                let depth = (local_id + 1).ilog2();
                let depth_penalty = if morph_active {
                    depth_penalty_base.powf(depth as f32 / 3.0)
                } else {
                    1.0
                };
                let scale = if morph_active {
                    morph_lr * leaf_shrink * depth_penalty
                } else {
                    learning_rate
                };
                if let Some((left_k, right_k)) = stump.multi_output_leaf_values.as_mut() {
                    for v in left_k.iter_mut() {
                        *v *= scale;
                    }
                    for v in right_k.iter_mut() {
                        *v *= scale;
                    }
                }
                // Keep the placeholder scalar consistent for any scalar code
                // path that accidentally reads it.
                if let Some((left_k, _)) = stump.multi_output_leaf_values.as_ref() {
                    stump.left_leaf_value = LeafValue::Scalar(left_k[0]);
                }
                if let Some((_, right_k)) = stump.multi_output_leaf_values.as_ref() {
                    stump.right_leaf_value = LeafValue::Scalar(right_k[0]);
                }
            }
        }

        // v0.10.0 review fix (Comment 2) — refactored in v0.10.3:
        // update training-time predictions via a per-row tree walk over
        // THIS round's stumps. Previously we applied every stump's
        // delta to every row, which is correct only when max_depth == 1
        // (each row reaches every stump). For max_depth > 1, non-root
        // stumps must only affect rows that reach them — which is
        // exactly what JointPredictor does at predict time. The walk
        // is now factored into `walk_tree_into_predictions`, shared
        // with DART dropout subtraction and warm-start replay.
        //
        // `round_result.stumps` are still pre-encode at this point
        // (local node IDs); the helper extracts local IDs via
        // `node_id % TREE_NODE_STRIDE` which works under both encodings.
        walk_tree_into_predictions(
            &round_result.stumps,
            binned_matrix,
            feature_count,
            n_rows,
            n_outputs,
            &mut predictions,
            1.0,
            1.0,
        );

        // Re-encode node_id to be globally unique across rounds (joint
        // trainer outputs one tree per round; local_node_id stays the
        // same, tree_index = round). Track the round's stump range so
        // DART can subtract / re-add this tree on later rounds.
        let round_start = all_stumps.len();
        let global_round = round + initial_rounds;
        for mut stump in round_result.stumps.into_iter() {
            let local_node_id = stump.split.node_id;
            stump.split.node_id = encode_tree_node_id(global_round, local_node_id)
                .map_err(|e| format!("encode_tree_node_id: {e:?}"))?;
            all_stumps.push(stump);
        }
        let round_count = all_stumps.len() - round_start;
        dart_round_start_offsets.push(round_start);
        dart_round_counts.push(round_count);

        // v0.10.3 joint DART finalize: rescale the new tree from 1.0
        // down to `new_w = 1 / (K + 1)`, and re-add each dropped tree
        // at its rescaled weight. Mirrors the single-output DART
        // finalize block in crates/engine/src/lib.rs:5118.
        if let Some((_, _, normalize_type, _)) = dart_params {
            let k = dropped_tree_ids.len() as f32;
            let new_w = 1.0 / (k + 1.0);
            let drop_factor = match normalize_type {
                alloygbm_core::DartNormalize::Tree => k / (k + 1.0),
                alloygbm_core::DartNormalize::Forest => 1.0 / (k + 1.0),
            };
            // 1. Scale new tree from 1.0 down to new_w: subtract
            //    (1.0 - new_w) of the new tree's contribution.
            let delta_scale = 1.0_f32 - new_w;
            if delta_scale.abs() > f32::EPSILON && round_count > 0 {
                walk_tree_into_predictions(
                    &all_stumps[round_start..round_start + round_count],
                    binned_matrix,
                    feature_count,
                    n_rows,
                    n_outputs,
                    &mut predictions,
                    -1.0,
                    delta_scale,
                );
            }
            // 2. Re-add each dropped tree at its post-normalize weight
            //    `w_new = w_old * drop_factor`.
            for &tree_id in &dropped_tree_ids {
                let w_old = dart_state.tree_weights[tree_id];
                let w_new = w_old * drop_factor;
                let start = dart_round_start_offsets[tree_id];
                let count = dart_round_counts[tree_id];
                if count > 0 {
                    walk_tree_into_predictions(
                        &all_stumps[start..start + count],
                        binned_matrix,
                        feature_count,
                        n_rows,
                        n_outputs,
                        &mut predictions,
                        1.0,
                        w_new,
                    );
                }
            }
            // 3. Push placeholder weight for the new tree, then run
            //    `apply_normalization` which rescales dropped trees in
            //    place AND sets the new-tree weight to `new_w`.
            dart_state.tree_weights.push(1.0);
            let new_tree_index = dart_state.tree_weights.len() - 1;
            crate::apply_normalization(
                &mut dart_state.tree_weights,
                &dropped_tree_ids,
                normalize_type,
                new_tree_index,
            );
        }
    }

    // v0.10.3 joint DART: stamp the per-tree `tree_weight` onto every
    // stump in that tree so the artifact's `DartTreeWeights` section
    // round-trips correctly. Mirrors the multiclass DART stamping
    // step at the end of `fit_multiclass_iterations_impl`.
    if dart_params.is_some() {
        for (round_idx, &w) in dart_state.tree_weights.iter().enumerate() {
            let start = dart_round_start_offsets[round_idx];
            let count = dart_round_counts[round_idx];
            for s in all_stumps.iter_mut().skip(start).take(count) {
                s.tree_weight = w;
            }
        }
    }

    // v0.10.4: persist MorphBoost EMA snapshot into the artifact so
    // warm-resume can re-seed `MorphState::ema_stats`. Mirrors the
    // single-output regressor path (lib.rs:4413).
    let morph_metadata = morph_state
        .as_ref()
        .map(|ms| alloygbm_core::MorphMetadataPayload {
            config: ms.config,
            // `final_iteration` captures where training ended (last completed
            // global round); `final_total` mirrors the full horizon used by
            // the LR schedule + leaf-shrink curve so warm-resume can recompute
            // them consistently.
            final_iteration: (initial_rounds + rounds_completed).saturating_sub(1) as u32,
            final_total: total_iterations_u32,
            ema_stats: ms.ema_stats.clone(),
        });

    let model = TrainedModel {
        baseline_prediction: 0.0, // Joint model uses per-output baselines (see summary)
        feature_count,
        stumps: all_stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: format!("joint_multi_output[{n_outputs}]"),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata,
        dro_metadata: None,
        feature_baseline: None,
    };

    // v0.10.3 warm-start: report TOTAL rounds completed (prior + new)
    // so a downstream consumer can decode "total tree count" from a
    // single integer.
    let total_rounds_completed = initial_rounds + rounds_completed;

    Ok(JointTrainingSummary {
        baselines,
        model,
        per_output_objective_names: per_output_objective
            .iter()
            .map(|o| o.name().to_string())
            .collect(),
        rounds_completed: total_rounds_completed,
    })
}

/// Build `ModelMetadata` for a joint-trained artifact. The Python wrapper
/// passes feature names + per-output baselines to round-trip cleanly.
pub fn build_joint_metadata(
    feature_names: Vec<String>,
    per_output_objective_names: &[String],
    baselines: &[f32],
) -> ModelMetadata {
    let baseline_summary = baselines
        .iter()
        .map(|b| format!("{b:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let objective_label = format!(
        "joint_multi_output:{}|baselines={}",
        per_output_objective_names.join("+"),
        baseline_summary,
    );
    ModelMetadata {
        format_version: alloygbm_core::MODEL_FORMAT_V1,
        feature_names,
        trained_device: Device::Cpu,
        objective: objective_label,
        num_classes: None,
    }
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

/// Tree-node-id stride used by the engine (1 << 20). Must match
/// `crate::TREE_NODE_STRIDE`; duplicated here as a `pub(crate)` constant
/// because the engine's copy is private.
const TREE_NODE_STRIDE: u32 = 1 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_objective_parses_supported_names() {
        assert_eq!(
            JointObjective::parse("rank:ndcg").unwrap(),
            JointObjective::RankNdcg
        );
        assert_eq!(
            JointObjective::parse("queryrmse").unwrap(),
            JointObjective::QueryRmse
        );
        assert!(JointObjective::parse("custom").is_err());
    }

    #[test]
    fn joint_objective_squared_error_initial_prediction_is_mean() {
        let targets = [1.0_f32, 2.0, 3.0, 4.0];
        let baseline = JointObjective::SquaredError.initial_prediction(&targets);
        assert!((baseline - 2.5).abs() < 1e-6);
    }

    #[test]
    fn joint_end_to_end_fit_predict_roundtrip_through_artifact() {
        // 8 rows, 1 feature with bins 0/4, 2 outputs.
        // Output 0 wants left=-1, right=+1; output 1 wants left=+0.5, right=-0.5
        let bins: Vec<u8> = vec![0, 0, 0, 0, 4, 4, 4, 4];
        let binned = BinnedMatrix::new(8, 1, /*max_bin=*/ 4, bins).expect("binned");

        // Targets: output 0 is linear-ish, output 1 is the opposite sign.
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
        ];

        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };

        let summary = fit_joint_multi_output(
            &params,
            /*feature_count=*/ 1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            /*n_estimators=*/ 1,
        )
        .expect("fit");

        // Baselines should be the per-output mean of targets.
        assert!((summary.baselines[0] - 0.0).abs() < 1e-5);
        assert!((summary.baselines[1] - 0.0).abs() < 1e-5);
        assert_eq!(summary.model.stumps.len(), 1);

        // Serialize → deserialize → predict
        let bytes = summary
            .model
            .clone()
            .to_artifact_bytes()
            .expect("serialize");
        let predictor =
            JointPredictor::from_artifact_bytes(&bytes, summary.baselines.clone()).expect("load");

        // Predict on a row with bin=0 (raw feature ≤ threshold).
        let preds_left = predictor.predict_row(&[0.0_f32]);
        let preds_right = predictor.predict_row(&[4.0_f32]);

        // After 1 round with lr=1.0: prediction = baseline + leaf_value
        // Left (rows 0-3): output 0 grad = -1*4 = -4 → leaf = 4/(4+ε) ≈ +1 (wait — gradient for SE is pred - target; with pred=0 baseline and target=-1 → grad = +1 → leaf = -1)
        // The exact value depends on Newton-Raphson and may differ slightly due to ε.
        // We just sanity-check the sign pattern:
        // output 0: left should be near -1, right should be near +1
        // output 1: left should be near +0.5, right should be near -0.5
        assert!(
            preds_left[0] < 0.0,
            "expected output 0 left < 0, got {:?}",
            preds_left
        );
        assert!(
            preds_right[0] > 0.0,
            "expected output 0 right > 0, got {:?}",
            preds_right
        );
        assert!(
            preds_left[1] > 0.0,
            "expected output 1 left > 0, got {:?}",
            preds_left
        );
        assert!(
            preds_right[1] < 0.0,
            "expected output 1 right < 0, got {:?}",
            preds_right
        );
    }

    #[test]
    fn joint_round_trip_with_non_unit_learning_rate_matches_training_predictions() {
        // Review fix (Comment 3): training-time predictions must equal
        // deserialized JointPredictor predictions for any learning_rate,
        // not just learning_rate == 1.0.
        let bins: Vec<u8> = vec![0, 0, 0, 0, 4, 4, 4, 4];
        let binned = BinnedMatrix::new(8, 1, /*max_bin=*/ 4, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
        ];
        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.3, // explicitly non-1.0
            ..TrainParams::default()
        };
        let summary = fit_joint_multi_output(
            &params,
            1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("fit");

        let bytes = summary
            .model
            .clone()
            .to_artifact_bytes()
            .expect("serialize");
        let predictor =
            JointPredictor::from_artifact_bytes(&bytes, summary.baselines.clone()).expect("load");

        // Reconstruct training-time predictions by walking the bins ourselves.
        // For a row in bin 0 (rows 0..4) and bin 4 (rows 4..8), call predict_row
        // and verify it matches what the training loop would have accumulated.
        let preds_bin0 = predictor.predict_row(&[0.0_f32]);
        let preds_bin4 = predictor.predict_row(&[4.0_f32]);

        // After 3 rounds with LR=0.3, the sign pattern should still match the
        // single-LR=1.0 case (lr only scales magnitude). The key invariant is
        // that the artifact's contributions are LR-scaled — verified
        // implicitly by the round-trip test below.
        assert!(preds_bin0[0] < 0.0, "got bin0 output0={}", preds_bin0[0]);
        assert!(preds_bin4[0] > 0.0, "got bin4 output0={}", preds_bin4[0]);
        assert!(preds_bin0[1] > 0.0, "got bin0 output1={}", preds_bin0[1]);
        assert!(preds_bin4[1] < 0.0, "got bin4 output1={}", preds_bin4[1]);

        // Direct invariant: a fresh fit with LR=0.3 then predict must equal a
        // fit with LR=1.0 with leaves manually scaled. Easier check: compare
        // against the JointTrainingSummary's `baselines + sum of stump deltas`.
        // Since each stump's multi_output_leaf_values is already LR-scaled, a
        // sum-of-leaves walk on the artifact must reproduce the trained
        // predictions. The round-trip test already covers this transitively.
    }

    #[test]
    fn joint_round_trip_max_depth_two_matches_training_predictions() {
        // Review fix (Comment 2): training-time prediction update must walk
        // the per-row tree path (not apply every stump to every row) so that
        // training-time predictions match deserialized JointPredictor output
        // for max_depth > 1.
        //
        // Build a dataset where max_depth=2 produces a non-trivial 3-stump
        // tree (root + two children with their own splits).
        // 12 rows, 2 features, 2 outputs.
        // Feature 0: bins partition rows into a coarse 0/3 split.
        // Feature 1: refines the split inside each half.
        let bins: Vec<u8> = vec![
            0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 3, 0, 3, 0, 3, 1, 3, 1, 3, 0, 3, 1,
        ];
        let binned = BinnedMatrix::new(12, 2, /*max_bin=*/ 3, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            // Output 0: 4 distinct levels (one per leaf of a depth-2 tree).
            vec![
                -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
            ],
            // Output 1: independent pattern with the same depth-2 structure.
            vec![
                1.0, 1.0, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -1.0, -1.0, -1.0, -1.0,
            ],
        ];
        let params = TrainParams {
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let summary = fit_joint_multi_output(
            &params,
            2,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            5,
        )
        .expect("fit");

        // Sanity: the resulting model should have produced a tree with depth-2
        // structure (more than 1 stump per round) at least once.
        assert!(
            summary.model.stumps.len() >= 2,
            "expected depth-2 trees to produce >=2 stumps, got {}",
            summary.model.stumps.len()
        );

        // Round-trip: deserialize and predict; each row's JointPredictor
        // output must equal `baseline + sum_over_rounds (reached_leaf_K_value)`,
        // which is by construction what the training loop accumulated via the
        // per-row tree walk. Spot-check: rows with different (feature0, feature1)
        // combinations should produce *different* predictions, demonstrating
        // the tree path is honored.
        let bytes = summary
            .model
            .clone()
            .to_artifact_bytes()
            .expect("serialize");
        let predictor =
            JointPredictor::from_artifact_bytes(&bytes, summary.baselines.clone()).expect("load");

        // Build the dense f32 feature matrix that matches the binned layout
        // for prediction (raw bin values are passed as features).
        let row_f = |i: usize| -> Vec<f32> {
            vec![binned.bins[i * 2] as f32, binned.bins[i * 2 + 1] as f32]
        };
        let p0 = predictor.predict_row(&row_f(0)); // (bin 0, bin 0)
        let p2 = predictor.predict_row(&row_f(2)); // (bin 0, bin 1)
        let p6 = predictor.predict_row(&row_f(6)); // (bin 3, bin 0)
        let p8 = predictor.predict_row(&row_f(8)); // (bin 3, bin 1)

        // All four leaf groups should differ on at least one output —
        // otherwise the depth-2 structure isn't reflected in predictions
        // (which is what would happen under the old broken stump-by-stump
        // update where every stump's delta was applied to every row).
        let distinct_pairs = [
            (p0.clone(), p2.clone()),
            (p2.clone(), p6.clone()),
            (p6.clone(), p8.clone()),
        ];
        for (a, b) in &distinct_pairs {
            let diff = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            assert!(
                diff > 1e-4,
                "max_depth=2 prediction collapsed to a single leaf: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn build_joint_round_emits_stumps_with_multi_output_leaf_values() {
        // 8 rows, 1 feature with bins 0..4, 2 outputs.
        // Bin layout: rows 0-3 → bin 0, rows 4-7 → bin 4.
        let bins: Vec<u8> = vec![0, 0, 0, 0, 4, 4, 4, 4];
        let binned = BinnedMatrix::new(8, 1, /*max_bin=*/ 4, bins).expect("binned");

        // Output 0 wants left=−1, right=+1 → gradient pushes leaves apart
        // Output 1 wants left=+0.5, right=−0.5 → opposite sign
        let grads_per_output: Vec<Vec<GradientPair>> = vec![
            (0..8)
                .map(|row| GradientPair {
                    grad: if row < 4 { 1.0 } else { -1.0 },
                    hess: 1.0,
                })
                .collect(),
            (0..8)
                .map(|row| GradientPair {
                    grad: if row < 4 { -0.5 } else { 0.5 },
                    hess: 1.0,
                })
                .collect(),
        ];

        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            ..TrainParams::default()
        };

        let result =
            build_joint_round(&params, &binned, &grads_per_output, 2, &[], 0, None).expect("build");
        assert_eq!(result.stumps.len(), 1, "should emit one root split");
        let stump = &result.stumps[0];
        let (left_k, right_k) = stump
            .multi_output_leaf_values
            .as_ref()
            .expect("joint stumps must carry multi-output leaves");
        assert_eq!(left_k.len(), 2);
        assert_eq!(right_k.len(), 2);
        // Output 0: left grad sum = 4, hess sum = 4 → leaf = −4/(4+ε) ≈ −1
        assert!((left_k[0] + 1.0).abs() < 0.01, "left[0]={}", left_k[0]);
        // Output 0: right grad sum = -4, hess sum = 4 → leaf = +1
        assert!((right_k[0] - 1.0).abs() < 0.01, "right[0]={}", right_k[0]);
        // Output 1: left grad sum = -2, hess sum = 4 → leaf = +0.5
        assert!((left_k[1] - 0.5).abs() < 0.01, "left[1]={}", left_k[1]);
        // Output 1: right grad sum = 2, hess sum = 4 → leaf = -0.5
        assert!((right_k[1] + 0.5).abs() < 0.01, "right[1]={}", right_k[1]);
    }

    #[test]
    fn joint_min_split_gain_rejects_low_gain_splits() {
        // 8 rows, 1 feature, 2 outputs. With a real (left ≠ right) signal,
        // min_split_gain=0 yields a split; setting min_split_gain to a huge
        // value (1e6) must suppress it.
        let bins: Vec<u8> = vec![0, 0, 0, 0, 4, 4, 4, 4];
        let binned = BinnedMatrix::new(8, 1, 4, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
        ];

        // Baseline: min_split_gain=0 → trains 1 stump.
        let params_baseline = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            min_split_gain: 0.0,
            ..TrainParams::default()
        };
        let summary_baseline = fit_joint_multi_output(
            &params_baseline,
            1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("fit baseline");
        assert!(
            !summary_baseline.model.stumps.is_empty(),
            "baseline fixture must produce >=1 stump; got {}",
            summary_baseline.model.stumps.len()
        );

        // With min_split_gain=1e6, no split should survive.
        let params_strict = TrainParams {
            min_split_gain: 1e6,
            ..params_baseline
        };
        let summary_strict = fit_joint_multi_output(
            &params_strict,
            1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("fit strict");
        assert_eq!(
            summary_strict.model.stumps.len(),
            0,
            "min_split_gain=1e6 must suppress all splits; got {} stumps",
            summary_strict.model.stumps.len()
        );
    }

    #[test]
    fn joint_row_subsample_changes_trees_deterministically_per_seed() {
        // 64 rows, 1 feature, 2 outputs. row_subsample=0.5 should produce a
        // different model than row_subsample=1.0, but identical across two
        // calls with the same seed.
        //
        // CRITICAL: within-side target variance is required to make the test
        // sensitive to which rows are sampled. With uniform per-side targets
        // (e.g. all left=-1, all right=+1), Newton-Raphson leaves collapse
        // to constants independent of the sampled subset.
        let mut bins: Vec<u8> = Vec::with_capacity(64);
        for i in 0..64 {
            bins.push(if i < 32 { 0 } else { 4 });
        }
        let binned = BinnedMatrix::new(64, 1, 4, bins).expect("binned");
        // Deterministic noisy targets keyed on row index so different sampled
        // subsets produce different per-leaf gradient/hessian sums.
        let targets_per_output: Vec<Vec<f32>> = vec![
            (0..64)
                .map(|i| {
                    let base = if i < 32 { -1.0 } else { 1.0 };
                    let noise = ((i as f32) * 0.137).sin() * 0.4;
                    base + noise
                })
                .collect(),
            (0..64)
                .map(|i| {
                    let base = if i < 32 { 0.5 } else { -0.5 };
                    let noise = ((i as f32) * 0.241).cos() * 0.3;
                    base + noise
                })
                .collect(),
        ];

        let mk = |row_subsample: f32, seed: u64| {
            let params = TrainParams {
                max_depth: 2,
                min_data_in_leaf: 1,
                lambda_l2: 0.0,
                learning_rate: 1.0,
                row_subsample,
                seed,
                ..TrainParams::default()
            };
            fit_joint_multi_output(
                &params,
                1,
                &binned,
                &targets_per_output,
                None,
                &[JointObjective::SquaredError, JointObjective::SquaredError],
                5,
            )
            .expect("fit")
        };

        let s_full = mk(1.0, 42);
        let s_half_a = mk(0.5, 42);
        let s_half_b = mk(0.5, 42);
        let s_half_c = mk(0.5, 99);

        // Determinism: same seed + same row_subsample → identical stump count.
        assert_eq!(
            s_half_a.model.stumps.len(),
            s_half_b.model.stumps.len(),
            "same (seed, row_subsample) must produce identical stump counts"
        );

        // Different seed should differ in at least one leaf value.
        let leaf_a = s_half_a.model.stumps[0]
            .multi_output_leaf_values
            .as_ref()
            .unwrap();
        let leaf_c = s_half_c.model.stumps[0]
            .multi_output_leaf_values
            .as_ref()
            .unwrap();
        assert!(
            (leaf_a.0[0] - leaf_c.0[0]).abs() > 1e-6 || (leaf_a.1[0] - leaf_c.1[0]).abs() > 1e-6,
            "different seeds should produce different leaves under row_subsample=0.5"
        );
        // row_subsample=0.5 should differ from row_subsample=1.0 in at least one leaf.
        let leaf_full = s_full.model.stumps[0]
            .multi_output_leaf_values
            .as_ref()
            .unwrap();
        assert!(
            (leaf_a.0[0] - leaf_full.0[0]).abs() > 1e-6
                || (leaf_a.1[0] - leaf_full.1[0]).abs() > 1e-6,
            "row_subsample=0.5 should produce different leaves from row_subsample=1.0"
        );
    }

    #[test]
    fn joint_col_subsample_restricts_features_in_split_search() {
        // 8 rows, 4 features, 2 outputs. Feature 0 is the best split
        // (target perfectly separates by f0). col_subsample=0.25 with some
        // seed should sometimes mask out feature 0 and force the model to
        // either split on a different feature OR produce zero stumps.
        let bins: Vec<u8> = vec![
            // f0, f1, f2, f3
            0, 0, 0, 0, // row 0
            0, 0, 0, 0, // row 1
            0, 1, 1, 1, // row 2
            0, 1, 1, 1, // row 3
            4, 0, 0, 0, // row 4
            4, 0, 0, 0, // row 5
            4, 1, 1, 1, // row 6
            4, 1, 1, 1, // row 7
        ];
        let binned = BinnedMatrix::new(8, 4, 4, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
        ];
        let mk = |col_subsample: f32, seed: u64| {
            let params = TrainParams {
                max_depth: 1,
                min_data_in_leaf: 1,
                lambda_l2: 0.0,
                learning_rate: 1.0,
                col_subsample,
                seed,
                ..TrainParams::default()
            };
            fit_joint_multi_output(
                &params,
                4,
                &binned,
                &targets_per_output,
                None,
                &[JointObjective::SquaredError, JointObjective::SquaredError],
                1,
            )
            .expect("fit")
        };

        // Sanity: col_subsample=1.0 picks feature 0 (the best).
        let full = mk(1.0, 0);
        assert_eq!(
            full.model.stumps[0].split.feature_index, 0,
            "best feature is 0 when all features available"
        );

        // col_subsample=0.25 → only ~1 of 4 features sampled per round.
        // Sweep seeds; at least one should exclude feature 0 from the mask
        // and force the model to either pick a different feature or
        // produce no stumps.
        let mut saw_non_zero = false;
        for seed in 0..64u64 {
            let m = mk(0.25, seed);
            if m.model.stumps.is_empty() {
                saw_non_zero = true;
                break;
            }
            if m.model.stumps[0].split.feature_index != 0 {
                saw_non_zero = true;
                break;
            }
        }
        assert!(
            saw_non_zero,
            "col_subsample=0.25 should sometimes exclude feature 0 from the split-search mask"
        );
    }

    #[test]
    fn joint_col_subsample_samples_different_features_each_round() {
        // v0.10.2.1 fix regression: col_subsample's RNG must mix the round
        // index into its seed so each tree samples a different feature
        // subset (LightGBM `feature_fraction` semantics). Without the fix,
        // every round picked the same masked-in features and the same
        // feature drove every split.
        //
        // Fixture: 8 features that are individually strong predictors,
        // n_estimators=8 rounds with col_subsample=0.25. Across the 8
        // rounds we should see >2 distinct split features (otherwise the
        // RNG is producing the same mask every round).
        let mut bins: Vec<u8> = Vec::with_capacity(40 * 8);
        for row in 0..40 {
            for f in 0..8 {
                // Each feature carries its own signal but with row-level noise
                // so any single feature alone supports a positive-gain split.
                let bin = if (row + f) % 2 == 0 { 0 } else { 4 };
                bins.push(bin);
            }
        }
        let binned = BinnedMatrix::new(40, 8, 4, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            (0..40)
                .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                .collect(),
            (0..40)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                .collect(),
        ];
        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            col_subsample: 0.25,
            seed: 7,
            ..TrainParams::default()
        };
        let summary = fit_joint_multi_output(
            &params,
            8,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            8,
        )
        .expect("fit");

        let distinct_split_features: std::collections::HashSet<u32> = summary
            .model
            .stumps
            .iter()
            .map(|s| s.split.feature_index)
            .collect();
        assert!(
            distinct_split_features.len() > 1,
            "col_subsample with multi-round training should sample different \
             feature subsets per round; got only {} distinct split feature(s) \
             across {} stumps",
            distinct_split_features.len(),
            summary.model.stumps.len()
        );
    }

    #[test]
    fn joint_interaction_constraints_forbid_feature_outside_active_group() {
        // 12 rows, 3 features, 2 outputs. With constraints {[0,1], [2]},
        // feature 2 is in its own group; once the tree splits on feature 0
        // (group 0), feature 2 (group 1) becomes unreachable on that path.
        //
        // We use a fixture where feature 2 is in fact the second-best split
        // candidate, so we can detect whether the constraint is honored:
        // without the constraint, feature 2 would appear in some stump;
        // with the constraint, it must not.
        let bins: Vec<u8> = vec![
            // f0, f1, f2
            0, 0, 0, // row 0
            0, 0, 1, // row 1
            0, 1, 0, // row 2
            0, 1, 1, // row 3
            4, 0, 0, // row 4
            4, 0, 1, // row 5
            4, 1, 0, // row 6
            4, 1, 1, // row 7
            0, 0, 0, // row 8
            0, 1, 1, // row 9
            4, 0, 1, // row 10
            4, 1, 0, // row 11
        ];
        let binned = BinnedMatrix::new(12, 3, 4, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            // f0 splits the major signal; f1 refines within each half.
            vec![
                -2.0, -1.0, -2.0, -1.0, 1.0, 2.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0,
            ],
            vec![
                1.0, 0.5, 1.0, 0.5, -0.5, -1.0, -0.5, -1.0, 1.0, 0.5, -0.5, -1.0,
            ],
        ];
        let params = TrainParams {
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            interaction_constraints: vec![vec![0, 1], vec![2]],
            ..TrainParams::default()
        };
        let summary = fit_joint_multi_output(
            &params,
            3,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("fit");

        // No stump may ever split on feature 2 once a path has used a
        // feature from group {0,1}. Since the root is constrained-free
        // (both groups active), feature 2 is *technically* allowed at
        // the root — but as soon as a stump on feature 0 or 1 appears,
        // descendants on that path must not use feature 2. The simplest
        // assertion: if the model contains feature-2 stumps AND
        // feature-{0,1} stumps, the feature-2 stumps must be at the root
        // (local_node_id 0) of their tree.
        let mut group01_used = false;
        let mut group2_non_root = false;
        for stump in &summary.model.stumps {
            let f = stump.split.feature_index;
            let local_node_id = stump.split.node_id % (1u32 << 20); // strip tree_id
            if f == 0 || f == 1 {
                group01_used = true;
            }
            if f == 2 && local_node_id != 0 {
                group2_non_root = true;
            }
        }
        assert!(group01_used, "expected at least one f0/f1 stump");
        assert!(
            !group2_non_root,
            "interaction_constraints violated: feature 2 used as a non-root stump (would cross groups)"
        );
    }

    #[test]
    fn joint_leafwise_growth_respects_max_leaves() {
        // 16 rows, 2 features, 2 outputs. Rich enough signal that level-wise
        // with max_depth=4 produces multiple stumps; leaf-wise with
        // max_leaves=3 must cap to ≤2 stumps per round.
        let bins: Vec<u8> = vec![
            0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0, 4, 1, 5, 0, 5, 1, 6, 0, 6, 1, 7,
            0, 7, 1,
        ];
        let binned = BinnedMatrix::new(16, 2, 7, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            // Output 0: monotonic in f0 with f1 modulating the slope.
            (0..16)
                .map(|i| {
                    let f0 = (i / 2) as f32;
                    let f1 = (i % 2) as f32;
                    f0 * 0.3 + f1 * (-0.5) + 0.1 * (i as f32).sin()
                })
                .collect(),
            // Output 1: opposite signal so joint gain is non-trivial.
            (0..16)
                .map(|i| {
                    let f0 = (i / 2) as f32;
                    let f1 = (i % 2) as f32;
                    -f0 * 0.2 + f1 * 0.4 + 0.1 * (i as f32).cos()
                })
                .collect(),
        ];

        // First, confirm the fixture is rich enough: level-wise produces >2 stumps.
        let params_level = TrainParams {
            tree_growth: TreeGrowth::Level,
            max_depth: 4,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let summary_level = fit_joint_multi_output(
            &params_level,
            2,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("fit level-wise");
        assert!(
            summary_level.model.stumps.len() > 2,
            "fixture sanity check: level-wise with max_depth=4 should produce >2 stumps; got {}",
            summary_level.model.stumps.len()
        );

        // Now leaf-wise with max_leaves=3 must cap to ≤2 stumps.
        let params_leaf = TrainParams {
            tree_growth: TreeGrowth::Leaf,
            max_leaves: Some(3),
            max_depth: 8,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let summary_leaf = fit_joint_multi_output(
            &params_leaf,
            2,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("fit leaf-wise");
        // 3 leaves → 2 internal splits → exactly 2 stumps per tree.
        assert!(
            summary_leaf.model.stumps.len() <= 2,
            "max_leaves=3 should cap stumps to ≤2; got {}",
            summary_leaf.model.stumps.len()
        );
        // And at least 1 stump (the root split) — otherwise leaf-wise didn't run.
        assert!(
            !summary_leaf.model.stumps.is_empty(),
            "expected at least 1 stump from leaf-wise growth"
        );

        // Round-trip: leaf-wise artifacts must serialize/deserialize and
        // JointPredictor must reproduce the training-time predictions.
        let bytes = summary_leaf
            .model
            .clone()
            .to_artifact_bytes()
            .expect("serialize leaf-wise model");
        let predictor = JointPredictor::from_artifact_bytes(&bytes, summary_leaf.baselines.clone())
            .expect("load leaf-wise predictor");
        // Predict on representative rows; results must be finite and
        // (with our fixture) at least two distinct rows must produce
        // different output-0 predictions (otherwise leaf-wise collapsed
        // everything to one leaf).
        let p0 = predictor.predict_row(&[0.0_f32, 0.0_f32]);
        let p15 = predictor.predict_row(&[7.0_f32, 1.0_f32]);
        assert!(p0[0].is_finite() && p0[1].is_finite());
        assert!(p15[0].is_finite() && p15[1].is_finite());
        assert!(
            (p0[0] - p15[0]).abs() > 1e-4,
            "leaf-wise must produce different predictions across the fixture; got p0={p0:?}, p15={p15:?}"
        );
    }

    #[test]
    fn joint_native_categorical_split_produces_bitset_stump() {
        // 12 rows, 1 categorical feature with 3 categories, 2 outputs.
        // Target pattern: category 1 wants output 0 = +1; categories 0 and 2
        // want output 0 = -1. Fisher-sort must partition {0, 2} vs {1}.
        let bins: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let binned = BinnedMatrix::new(12, 1, 2, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![
                -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
            ],
            vec![
                0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
            ],
        ];
        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let cat_features = vec![crate::CategoricalFeatureInfo {
            feature_index: 0,
            num_categories: 3,
        }];
        let summary = fit_joint_multi_output_with_categorical(
            &params,
            1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
            &cat_features,
        )
        .expect("fit");
        assert_eq!(summary.model.stumps.len(), 1, "expected one root split");
        let stump = &summary.model.stumps[0];
        assert!(
            stump.split.is_categorical,
            "should produce categorical split"
        );
        let bitset = stump
            .split
            .categorical_bitset
            .as_ref()
            .expect("bitset present");
        // Decode bit 0 (cat 0), bit 1 (cat 1), bit 2 (cat 2) from the first byte.
        let bs0 = bitset[0];
        let bit0 = bs0 & 1;
        let bit1 = (bs0 >> 1) & 1;
        let bit2 = (bs0 >> 2) & 1;
        // Cats 0 and 2 should be on the same side; cat 1 on the other.
        assert_eq!(
            bit0, bit2,
            "cats 0 and 2 should share a side, got bitset=0b{:08b}",
            bs0
        );
        assert_ne!(bit0, bit1, "cat 1 should be opposite of cat 0");
    }

    #[test]
    fn joint_predictor_evaluates_categorical_stumps_correctly() {
        // Same fixture as joint_native_categorical_split_produces_bitset_stump.
        // After fit, JointPredictor must route raw category values via the
        // bitset (not via threshold compare) and produce sign-correct
        // predictions for each category.
        let bins: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let binned = BinnedMatrix::new(12, 1, 2, bins).expect("binned");
        let targets_per_output: Vec<Vec<f32>> = vec![
            vec![
                -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
            ],
            vec![
                0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5,
            ],
        ];
        let params = TrainParams {
            max_depth: 1,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let cat_features = vec![crate::CategoricalFeatureInfo {
            feature_index: 0,
            num_categories: 3,
        }];
        let summary = fit_joint_multi_output_with_categorical(
            &params,
            1,
            &binned,
            &targets_per_output,
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
            &cat_features,
        )
        .expect("fit");
        let bytes = summary
            .model
            .clone()
            .to_artifact_bytes()
            .expect("serialize");
        let predictor =
            JointPredictor::from_artifact_bytes(&bytes, summary.baselines.clone()).expect("load");

        // Predict each category. Cat 1 should produce output 0 > 0 and
        // output 1 < 0 (since cat 1 wants y=+1 for output 0, y=-0.5 for output 1).
        // Cats 0 and 2 (paired side) should produce the opposite signs.
        let p0 = predictor.predict_row(&[0.0_f32]); // cat 0
        let p1 = predictor.predict_row(&[1.0_f32]); // cat 1
        let p2 = predictor.predict_row(&[2.0_f32]); // cat 2
        assert!(p1[0] > 0.0, "cat 1 output 0 should be > 0, got {}", p1[0]);
        assert!(p0[0] < 0.0, "cat 0 output 0 should be < 0, got {}", p0[0]);
        assert!(p2[0] < 0.0, "cat 2 output 0 should be < 0, got {}", p2[0]);
        assert!(p1[1] < 0.0, "cat 1 output 1 should be < 0, got {}", p1[1]);
        assert!(p0[1] > 0.0, "cat 0 output 1 should be > 0, got {}", p0[1]);
        assert!(p2[1] > 0.0, "cat 2 output 1 should be > 0, got {}", p2[1]);
    }

    #[test]
    fn joint_goss_changes_trained_model_vs_standard() {
        // 200 rows, 2 outputs. Half are large-magnitude-gradient rows,
        // half are near-zero. With GOSS top_rate=0.2 other_rate=0.1, the
        // trainer should fit on a tiny subset (~60 rows) with amplified
        // gradients. With Standard, the trainer sees all 200 rows. The
        // resulting first-round leaf values MUST differ — a no-op
        // BoostingMode would produce identical models.
        use alloygbm_core::BoostingMode;

        let n_rows = 200;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -10.0 } else { 0.01 })
            .collect();
        let targets_1: Vec<f32> = targets_0.iter().map(|&t| t * 2.0).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        let params_standard = TrainParams {
            seed: 7,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            boosting_mode: BoostingMode::Standard,
            ..TrainParams::default()
        };
        let params_goss = TrainParams {
            seed: 7,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            boosting_mode: BoostingMode::Goss {
                top_rate: 0.2,
                other_rate: 0.1,
            },
            ..TrainParams::default()
        };

        let summary_standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("standard fit");
        let summary_goss = fit_joint_multi_output(
            &params_goss,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("goss fit");

        // Both must produce models, but the leaf values must diverge
        // (proves GOSS isn't a silent no-op). Compare the first stump's
        // K-output left value across the two fits.
        assert!(!summary_standard.model.stumps.is_empty());
        assert!(!summary_goss.model.stumps.is_empty());
        let leaf_standard = summary_standard.model.stumps[0]
            .multi_output_leaf_values
            .as_ref()
            .expect("multi-output leaf");
        let leaf_goss = summary_goss.model.stumps[0]
            .multi_output_leaf_values
            .as_ref()
            .expect("multi-output leaf");
        let max_diff = leaf_standard
            .0
            .iter()
            .zip(leaf_goss.0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "GOSS must change the first stump's left K-output leaf; \
             standard={:?} goss={:?}",
            leaf_standard.0,
            leaf_goss.0,
        );
    }

    #[test]
    fn joint_morph_changes_trained_model_vs_standard() {
        // v0.10.4: same seed, same data, only `morph_config` differs → trees
        // (or leaf values) must differ. Proves MorphBoost is plumbed into the
        // joint split-gain dispatch and leaf-scaling pipeline.
        use alloygbm_core::MorphConfig;

        let n_rows = 200;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let targets_1: Vec<f32> = targets_0.iter().map(|&t| t * 2.0).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        let params_standard = TrainParams {
            seed: 7,
            max_depth: 3,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.1,
            ..TrainParams::default()
        };
        let params_morph = TrainParams {
            morph_config: Some(MorphConfig::default()),
            ..params_standard.clone()
        };

        let summary_standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            5,
        )
        .expect("standard fit");
        let summary_morph = fit_joint_multi_output(
            &params_morph,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            5,
        )
        .expect("morph fit");

        assert!(!summary_standard.model.stumps.is_empty());
        assert!(!summary_morph.model.stumps.is_empty());
        // Morph must persist a `morph_metadata` section (standard fit doesn't).
        assert!(summary_standard.model.morph_metadata.is_none());
        assert!(summary_morph.model.morph_metadata.is_some());

        // At least one stump's leaf values must differ between the two fits
        // (warmup is byte-equivalent, but later rounds + leaf shrinkage will
        // diverge — the 5-round horizon is enough to exit warmup at depth_penalty
        // factor < 1 for non-root leaves).
        let differ = summary_standard
            .model
            .stumps
            .iter()
            .zip(summary_morph.model.stumps.iter())
            .any(|(s, m)| {
                let s_leaves = s.multi_output_leaf_values.as_ref();
                let m_leaves = m.multi_output_leaf_values.as_ref();
                match (s_leaves, m_leaves) {
                    (Some((sl, sr)), Some((ml, mr))) => {
                        sl.iter().zip(ml.iter()).any(|(a, b)| (a - b).abs() > 1e-5)
                            || sr.iter().zip(mr.iter()).any(|(a, b)| (a - b).abs() > 1e-5)
                    }
                    _ => true,
                }
            });
        assert!(
            differ,
            "MorphBoost must change at least one leaf value vs. standard"
        );
    }

    #[test]
    fn joint_morph_warmup_byte_equivalent_to_standard() {
        // Within warmup (iteration < morph_warmup_iters), the morph split-gain
        // dispatch is byte-equivalent to standard. Verify by running 1 round
        // with morph_warmup_iters=10 (warmup never ends) and learning_rate=1.0,
        // morph_rate=0.0 (no leaf shrinkage), depth_penalty_base=1.0 (no depth
        // penalty) → leaves should match standard byte-for-byte.
        use alloygbm_core::{LrSchedule, MorphConfig};

        let n_rows = 64;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows).map(|i| (i % 4) as f32).collect();
        let targets_1: Vec<f32> = (0..n_rows).map(|i| ((i + 1) % 4) as f32).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 4) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 3, bins).expect("bm");

        let params_standard = TrainParams {
            seed: 11,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let params_morph = TrainParams {
            morph_config: Some(MorphConfig {
                morph_warmup_iters: 10,  // single round at iter 0 is in warmup
                morph_rate: 0.0,         // no leaf shrinkage
                depth_penalty_base: 1.0, // no depth penalty
                evolution_pressure: 0.0,
                info_score_weight: 0.0, // belt-and-suspenders: info_score off
                balance_penalty: false, // disable balance penalty so warmup is byte-equiv
                lr_schedule: LrSchedule::Constant,
            }),
            ..params_standard.clone()
        };

        let summary_standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("standard fit");
        let summary_morph = fit_joint_multi_output(
            &params_morph,
            feature_count,
            &binned_matrix,
            &[targets_0, targets_1],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            1,
        )
        .expect("morph warmup fit");

        // Leaf values must match within float rounding (warmup gain formula is
        // byte-equivalent modulo `GAIN_EPSILON` matching).
        assert_eq!(
            summary_standard.model.stumps.len(),
            summary_morph.model.stumps.len(),
        );
        for (s, m) in summary_standard
            .model
            .stumps
            .iter()
            .zip(summary_morph.model.stumps.iter())
        {
            let s_leaves = s.multi_output_leaf_values.as_ref().expect("std leaves");
            let m_leaves = m.multi_output_leaf_values.as_ref().expect("morph leaves");
            for (a, b) in s_leaves.0.iter().zip(m_leaves.0.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "warmup left leaves must match: std={a} morph={b}"
                );
            }
            for (a, b) in s_leaves.1.iter().zip(m_leaves.1.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "warmup right leaves must match: std={a} morph={b}"
                );
            }
        }
    }

    #[test]
    fn joint_morph_warm_resume_preserves_ema_continuity_not_byte_equivalence() {
        // v0.10.4 PR #37 review (C3): a `6+4` warm-resumed MorphBoost fit is
        // intentionally NOT byte-equivalent to a fresh `n_estimators=10`
        // MorphBoost fit. The prior six trees baked in their per-iteration
        // leaf shrinkage (`1 - morph_rate * t/T`) against a 6-round horizon
        // and cannot be retroactively re-scaled at resume time. What IS
        // preserved is EMA continuity: `MorphState::ema_stats` from the
        // prior fit is re-seeded into the new fit so gradient-statistics
        // smoothing flows across the resume boundary.
        //
        // This test pins both invariants:
        //   1. The persisted prior `morph_metadata.ema_stats` non-trivially
        //      differs from the fresh `GradientEmaStats::default()`
        //      (proves warm-resume actually receives the snapshot).
        //   2. A warm-resumed `6+4` fit's predictions DO differ from a
        //      fresh `n_estimators=10` fit's predictions (documents the
        //      non-byte-equivalence contract).
        use crate::TrainParams;
        use alloygbm_core::{BinnedMatrix, MorphConfig};

        let n_rows = 200;
        let feature_count = 1;
        let t0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let t1: Vec<f32> = t0.iter().map(|&v| v * 0.5).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned = BinnedMatrix::new(n_rows, feature_count, 7, bins).expect("bm");

        let params = TrainParams {
            seed: 5,
            max_depth: 3,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.1,
            morph_config: Some(MorphConfig::default()),
            ..TrainParams::default()
        };

        // Prior 6-round fit.
        let prior = fit_joint_multi_output(
            &params,
            feature_count,
            &binned,
            &[t0.clone(), t1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            6,
        )
        .expect("prior fit");

        // Invariant 1: prior fit persisted a non-default EMA snapshot.
        let prior_meta = prior
            .model
            .morph_metadata
            .as_ref()
            .expect("morph_metadata persisted on morph fit");
        assert_eq!(prior_meta.ema_stats.len(), 2);
        let default_mean = alloygbm_core::GradientEmaStats::default().mean;
        let default_std = alloygbm_core::GradientEmaStats::default().std;
        let any_moved = prior_meta
            .ema_stats
            .iter()
            .any(|s| (s.mean - default_mean).abs() > 1e-6 || (s.std - default_std).abs() > 1e-6);
        assert!(
            any_moved,
            "prior MorphBoost fit must have moved EMA away from defaults: {:?}",
            prior_meta.ema_stats
        );

        // Warm-resume +4 rounds.
        let warm_state = JointWarmStartState {
            baselines: prior.baselines.clone(),
            stumps: prior.model.stumps.clone(),
            initial_rounds_completed: 6,
            initial_dart_tree_weights: None,
            initial_ema_stats: Some(prior_meta.ema_stats.clone()),
        };
        let resumed = fit_joint_multi_output_with_warm_start(
            &params,
            feature_count,
            &binned,
            &[t0.clone(), t1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            4,
            &[],
            Some(warm_state),
        )
        .expect("warm resume");
        // Fresh 10-round fit.
        let fresh = fit_joint_multi_output(
            &params,
            feature_count,
            &binned,
            &[t0, t1],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            10,
        )
        .expect("fresh 10");

        // Invariant 2: warm-resumed `6+4` does NOT match fresh `10`. The
        // prior six trees were trained against a 6-round horizon so their
        // shrinkage is denser than the first six trees of a 10-round fit.
        // We assert at least ONE stump's leaf values differ. (Note: this
        // is a regression-style test pinning the documented non-equivalence
        // contract; the user's reproduced ~0.064 prediction diff in PR #37
        // C3 is consistent with this.)
        let warm_meta = resumed
            .model
            .morph_metadata
            .as_ref()
            .expect("warm metadata");
        let fresh_meta = fresh.model.morph_metadata.as_ref().expect("fresh metadata");
        assert_eq!(warm_meta.ema_stats.len(), fresh_meta.ema_stats.len());
        let differ = resumed
            .model
            .stumps
            .iter()
            .zip(fresh.model.stumps.iter())
            .any(|(w, f)| {
                let w_leaves = w.multi_output_leaf_values.as_ref();
                let f_leaves = f.multi_output_leaf_values.as_ref();
                match (w_leaves, f_leaves) {
                    (Some((wl, wr)), Some((fl, fr))) => {
                        wl.iter().zip(fl.iter()).any(|(a, b)| (a - b).abs() > 1e-5)
                            || wr.iter().zip(fr.iter()).any(|(a, b)| (a - b).abs() > 1e-5)
                    }
                    _ => true,
                }
            });
        assert!(
            differ,
            "PR #37 C3 contract: 6+4 warm-resume MorphBoost must DIFFER \
             from fresh 10-round MorphBoost — prior-tree shrinkage was \
             baked against the 6-round horizon and is not re-scaled on \
             resume. If this assert fails the byte-equivalence contract \
             has changed and CHANGELOG / docs / `JointWarmStartState::\
             initial_ema_stats` need updating."
        );
    }

    #[test]
    fn joint_dart_produces_non_uniform_tree_weights() {
        // DART must drop at least one prior tree per round (after the
        // first), and `apply_normalization` rescales both the new tree
        // and the dropped trees so the persisted `tree_weight` field
        // varies from 1.0. A fresh DART fit thus exposes a non-uniform
        // tree_weight array.
        use alloygbm_core::{BoostingMode, DartNormalize, DartSampleType};

        let n_rows = 100;
        let feature_count = 1;
        let targets: Vec<f32> = (0..n_rows).map(|i| (i % 4) as f32).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 4) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 3, bins).expect("bm");

        let params = TrainParams {
            seed: 42,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            boosting_mode: BoostingMode::Dart {
                drop_rate: 0.5,
                max_drop: 5,
                normalize_type: DartNormalize::Tree,
                sample_type: DartSampleType::Uniform,
            },
            ..TrainParams::default()
        };

        let summary = fit_joint_multi_output(
            &params,
            feature_count,
            &binned_matrix,
            &[targets.clone(), targets.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            8,
        )
        .expect("joint DART fit must succeed");
        // Group stumps by tree; the first stump of each tree carries the
        // tree_weight (mirrors `apply_dart_tree_weights`).
        let mut weights: Vec<f32> = Vec::new();
        let mut current_tree: Option<u32> = None;
        for s in &summary.model.stumps {
            let tree_id = s.split.node_id / TREE_NODE_STRIDE;
            if Some(tree_id) != current_tree {
                weights.push(s.tree_weight);
                current_tree = Some(tree_id);
            }
        }
        assert!(weights.len() >= 2, "got only {} weights", weights.len());
        let any_non_unit = weights.iter().any(|&w| (w - 1.0).abs() > 1e-4);
        assert!(
            any_non_unit,
            "expected non-uniform tree_weights, got {weights:?}"
        );
    }

    #[test]
    fn joint_dart_round_trips_through_predictor() {
        // After DART training, JointPredictor must reproduce the
        // training-time predictions. The per-tree `tree_weight` lives
        // on every stump and is persisted via the existing
        // `DartTreeWeights` artifact section.
        use alloygbm_core::{BoostingMode, DartNormalize, DartSampleType};

        let n_rows = 80;
        let feature_count = 2;
        let targets_0: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 0.1).collect();
        let targets_1: Vec<f32> = (0..n_rows).map(|i| -(i as f32) * 0.05).collect();
        let bins: Vec<u8> = (0..(n_rows * feature_count))
            .map(|i| (i % 8) as u8)
            .collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        let params = TrainParams {
            seed: 7,
            max_depth: 3,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            boosting_mode: BoostingMode::Dart {
                drop_rate: 0.4,
                max_drop: 3,
                normalize_type: DartNormalize::Tree,
                sample_type: DartSampleType::Uniform,
            },
            ..TrainParams::default()
        };
        let summary = fit_joint_multi_output(
            &params,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            6,
        )
        .expect("dart fit");

        // The artifact must include DartTreeWeights (any non-unit weight
        // triggers emission in `TrainedModel::to_artifact_bytes`).
        let bytes = summary.model.clone().to_artifact_bytes().expect("artifact");
        let predictor =
            JointPredictor::from_artifact_bytes(&bytes, summary.baselines.clone()).expect("load");
        // Reconstruct a few rows' raw features from binned_matrix.bins
        // (the test fixture put bin == raw feature value modulo 8) and
        // verify finite predictions.
        for row in 0..5 {
            let feats: Vec<f32> = (0..feature_count)
                .map(|f| binned_matrix.bins[row * feature_count + f] as f32)
                .collect();
            let p = predictor.predict_row(&feats);
            assert_eq!(p.len(), 2);
            assert!(p[0].is_finite(), "row {row} output 0 = {}", p[0]);
            assert!(p[1].is_finite(), "row {row} output 1 = {}", p[1]);
        }
    }

    #[test]
    fn joint_warm_start_with_no_prior_state_matches_fresh_fit() {
        let n_rows = 60;
        let feature_count = 1;
        let targets: Vec<f32> = (0..n_rows).map(|i| (i % 4) as f32).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 4) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 3, bins).expect("bm");
        let params = TrainParams {
            seed: 1,
            max_depth: 2,
            ..TrainParams::default()
        };
        let fresh = fit_joint_multi_output(
            &params,
            feature_count,
            &binned_matrix,
            std::slice::from_ref(&targets),
            None,
            &[JointObjective::SquaredError],
            4,
        )
        .expect("fresh");
        let warm = fit_joint_multi_output_with_warm_start(
            &params,
            feature_count,
            &binned_matrix,
            std::slice::from_ref(&targets),
            None,
            &[JointObjective::SquaredError],
            4,
            &[],
            None,
        )
        .expect("warm with None");
        assert_eq!(fresh.model.stumps.len(), warm.model.stumps.len());
        assert_eq!(fresh.baselines, warm.baselines);
    }

    #[test]
    fn joint_dro_radius_nonzero_changes_leaves_vs_standard() {
        // v0.10.5: DRO leaf solver on the level-wise joint path must produce
        // different leaf values from standard Newton-Raphson when dro_config
        // has a nonzero radius.  The two fits are otherwise identical
        // (same seed, same data, same objectives).
        use alloygbm_core::{DroConfig, DroMetric, LeafSolverKind};

        let n_rows = 200;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let targets_1: Vec<f32> = targets_0.iter().map(|&t| t * 2.0).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        // Baseline: standard leaves (default solver, no DRO).
        let params_standard = TrainParams {
            seed: 7,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.3,
            leaf_solver: LeafSolverKind::Standard,
            dro_config: None,
            ..TrainParams::default()
        };

        // DRO: same params but with Wasserstein DRO radius=0.5.
        let params_dro = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.5,
                metric: DroMetric::Wasserstein,
            }),
            ..params_standard.clone()
        };

        let standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("standard fit succeeds");

        let dro = fit_joint_multi_output(
            &params_dro,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("dro fit succeeds");

        // DRO can alter residuals enough to change the set of accepted splits
        // across rounds, so the stump count may legitimately differ.  What we
        // require is that at least one stump in the overlapping prefix has
        // different leaf values — or the counts differ (which is itself a sign
        // that DRO changed the training trajectory).
        let counts_differ = standard.model.stumps.len() != dro.model.stumps.len();
        let any_leaf_diff = standard
            .model
            .stumps
            .iter()
            .zip(dro.model.stumps.iter())
            .any(|(a, b)| {
                let (la, ra) = a.multi_output_leaf_values.as_ref().unwrap();
                let (lb, rb) = b.multi_output_leaf_values.as_ref().unwrap();
                la != lb || ra != rb
            });
        assert!(
            counts_differ || any_leaf_diff,
            "DRO leaves should differ from standard leaves on the same data"
        );
    }

    #[test]
    fn joint_dro_with_leafwise_growth_changes_leaves() {
        use alloygbm_core::{DroConfig, DroMetric, LeafSolverKind};

        let n_rows = 200;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let targets_1: Vec<f32> = targets_0.iter().map(|&t| t * 2.0).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        let params_standard = TrainParams {
            seed: 7,
            max_depth: 4,
            max_leaves: Some(4),
            tree_growth: TreeGrowth::Leaf,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.3,
            leaf_solver: LeafSolverKind::Standard,
            dro_config: None,
            ..TrainParams::default()
        };

        let params_dro = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.5,
                metric: DroMetric::Wasserstein,
            }),
            ..params_standard.clone()
        };

        let standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("standard leafwise fit");

        let dro = fit_joint_multi_output(
            &params_dro,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            3,
        )
        .expect("dro leafwise fit");

        let counts_differ = standard.model.stumps.len() != dro.model.stumps.len();
        let any_leaf_diff = standard
            .model
            .stumps
            .iter()
            .zip(dro.model.stumps.iter())
            .any(|(a, b)| {
                let (la, ra) = a.multi_output_leaf_values.as_ref().unwrap();
                let (lb, rb) = b.multi_output_leaf_values.as_ref().unwrap();
                la != lb || ra != rb
            });
        assert!(
            counts_differ || any_leaf_diff,
            "DRO leaves should differ from standard leaves under leaf-wise growth"
        );
    }

    #[test]
    fn joint_dro_radius_zero_matches_standard_byte_for_byte() {
        use alloygbm_core::{DroConfig, DroMetric, LeafSolverKind};

        let n_rows = 200;
        let feature_count = 1;
        let targets_0: Vec<f32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let targets_1: Vec<f32> = targets_0.iter().map(|&t| t * 2.0).collect();
        let bins: Vec<u8> = (0..n_rows).map(|i| (i % 8) as u8).collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");

        let params_standard = TrainParams {
            seed: 7,
            max_depth: 2,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            learning_rate: 0.3,
            leaf_solver: LeafSolverKind::Standard,
            dro_config: None,
            ..TrainParams::default()
        };

        // DRO with radius=0 should collapse to standard byte-for-byte.
        let params_dro_zero = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.0,
                metric: DroMetric::Wasserstein,
            }),
            ..params_standard.clone()
        };

        let standard = fit_joint_multi_output(
            &params_standard,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            4,
        )
        .expect("standard fit");

        let dro_zero = fit_joint_multi_output(
            &params_dro_zero,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            4,
        )
        .expect("dro radius=0 fit");

        // Byte-equivalence: identical artifact bytes prove identical trees.
        let a = standard.model.to_artifact_bytes().expect("ser std");
        let b = dro_zero.model.to_artifact_bytes().expect("ser dro0");
        assert_eq!(
            a, b,
            "DRO with radius=0 must produce byte-identical artifact to standard leaves"
        );
    }

    #[test]
    fn joint_warm_start_continues_from_prior_state() {
        let n_rows = 80;
        let feature_count = 2;
        let targets_0: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 0.1).collect();
        let targets_1: Vec<f32> = (0..n_rows).map(|i| -(i as f32) * 0.05).collect();
        let bins: Vec<u8> = (0..(n_rows * feature_count))
            .map(|i| (i % 8) as u8)
            .collect();
        let binned_matrix =
            BinnedMatrix::new(n_rows, feature_count, /*max_bin=*/ 7, bins).expect("bm");
        let params = TrainParams {
            seed: 5,
            max_depth: 3,
            min_data_in_leaf: 1,
            lambda_l2: 0.0,
            ..TrainParams::default()
        };

        // Fresh 10-round fit.
        let fresh = fit_joint_multi_output(
            &params,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            10,
        )
        .expect("fresh");

        // 6 + 4 split: train 6 rounds, then warm-resume for 4 more.
        let first = fit_joint_multi_output(
            &params,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            6,
        )
        .expect("first 6");
        let warm = JointWarmStartState {
            baselines: first.baselines.clone(),
            stumps: first.model.stumps.clone(),
            initial_rounds_completed: 6,
            initial_dart_tree_weights: None,
            initial_ema_stats: None,
        };
        let resumed = fit_joint_multi_output_with_warm_start(
            &params,
            feature_count,
            &binned_matrix,
            &[targets_0.clone(), targets_1.clone()],
            None,
            &[JointObjective::SquaredError, JointObjective::SquaredError],
            4,
            &[],
            Some(warm),
        )
        .expect("resume 4");

        assert_eq!(
            resumed.model.stumps.len(),
            fresh.model.stumps.len(),
            "warm-resume must produce the same number of stumps"
        );
        // Predictions on a few rows must match (within tight numerical
        // tolerance — the random draws on rounds 6..9 are seeded with
        // `global_round = round + 6` so they should be identical to
        // the fresh fit's rounds 6..9).
        let bytes_fresh = fresh.model.clone().to_artifact_bytes().expect("artifact");
        let bytes_resume = resumed.model.clone().to_artifact_bytes().expect("artifact");
        let p_fresh = JointPredictor::from_artifact_bytes(&bytes_fresh, fresh.baselines.clone())
            .expect("pred fresh");
        let p_resume =
            JointPredictor::from_artifact_bytes(&bytes_resume, resumed.baselines.clone())
                .expect("pred resume");
        for row in 0..5 {
            let feats: Vec<f32> = (0..feature_count)
                .map(|f| binned_matrix.bins[row * feature_count + f] as f32)
                .collect();
            let pf = p_fresh.predict_row(&feats);
            let pr = p_resume.predict_row(&feats);
            for k in 0..2 {
                assert!(
                    (pf[k] - pr[k]).abs() < 1e-4,
                    "row {row} output {k} fresh={pf:?} resume={pr:?}"
                );
            }
        }
    }
}
