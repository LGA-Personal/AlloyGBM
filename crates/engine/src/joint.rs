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
    Device, LambdaMARTObjective, ObjectiveOps, PairwiseRankingObjective, QueryRMSEObjective,
    SquaredErrorObjective, TrainedModel, TrainedStump, XeNDCGObjective, encode_tree_node_id,
};
use alloygbm_core::{
    BinnedMatrix, GradientPair, LeafValue, MISSING_BIN_U8, ModelMetadata, NodeStats,
    SplitCandidate, TrainParams,
};

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
                let group_ids = group.ok_or_else(|| {
                    "queryrmse objective requires group identifiers".to_string()
                })?;
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
                let group_ids = group.ok_or_else(|| {
                    "rank:ndcg objective requires group identifiers".to_string()
                })?;
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
}

/// Per-leaf bookkeeping during level-wise tree growth on the joint trainer.
#[derive(Debug)]
struct JointLeafNode {
    /// 0-indexed local node id within this tree (matches predictor traversal).
    local_node_id: u32,
    /// Row indices currently routed to this node.
    row_indices: Vec<u32>,
}

/// Output of a single joint training round: one shared tree expressed as a
/// sequence of stumps with K-output leaf values populated.
#[derive(Debug, Clone)]
pub struct JointRoundResult {
    pub stumps: Vec<TrainedStump>,
}

/// Train a single round of joint multi-output boosting and return the new
/// stumps (already updated to carry K-output leaf values). The caller is
/// responsible for accumulating leaf contributions into per-output prediction
/// vectors using `predictions[k][row] += learning_rate * leaf_value_k`.
pub fn build_joint_round(
    params: &TrainParams,
    binned_matrix: &BinnedMatrix,
    grads_per_output: &[Vec<GradientPair>],
    n_outputs: usize,
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
    let lambda_l2 = params.lambda_l2 as f32;

    let mut stumps: Vec<TrainedStump> = Vec::new();
    let mut active: Vec<JointLeafNode> = vec![JointLeafNode {
        local_node_id: 0,
        row_indices: (0..n_rows as u32).collect(),
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
                // Subset the bin column for this feature.
                let mut subset_bins: Vec<u8> = Vec::with_capacity(node.row_indices.len());
                for &row in &node.row_indices {
                    // Row-major: bins[row * feature_count + feature].
                    let idx = row as usize * feature_count + feature;
                    subset_bins.push(binned_matrix.bins[idx]);
                }
                // Subset packed_grads/hess for these rows.
                let mut subset_g: Vec<f32> =
                    Vec::with_capacity(node.row_indices.len() * n_outputs);
                let mut subset_h: Vec<f32> =
                    Vec::with_capacity(node.row_indices.len() * n_outputs);
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
            let mut best: Option<(usize, usize, f32)> = None; // (feature, threshold_bin, gain)
            // BinnedMatrix exposes max_bin globally; iterate candidate
            // thresholds across the full bin range minus the NaN slot.
            let max_threshold = (binned_matrix.max_bin as usize).min(MISSING_BIN_U8 as usize - 1);
            for feature in 0..feature_count {
                for threshold_bin in 0..max_threshold {
                    let gain = compute_multi_output_split_gain(
                        &node_hist,
                        feature,
                        threshold_bin,
                        lambda_l2,
                        crate::LEAF_EPSILON,
                    );
                    if gain <= 0.0 {
                        continue;
                    }
                    if best.map(|(_, _, g)| gain > g).unwrap_or(true) {
                        best = Some((feature, threshold_bin, gain));
                    }
                }
            }

            let Some((feature, threshold_bin, gain)) = best else {
                continue; // No positive-gain split — leave node as terminal leaf
            };

            // Partition rows by the chosen split. NaN rows (bin == MISSING_BIN_U8)
            // route per default_left below; we pick the direction yielding more
            // rows on either side (simple v0.10.0 default).
            let mut left_rows: Vec<u32> = Vec::new();
            let mut right_rows: Vec<u32> = Vec::new();
            let mut missing_rows: Vec<u32> = Vec::new();
            for &row in &node.row_indices {
                let bin = binned_matrix.bins[row as usize * feature_count + feature];
                if bin == MISSING_BIN_U8 {
                    missing_rows.push(row);
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
                left_rows.extend(missing_rows.drain(..));
            } else {
                right_rows.extend(missing_rows.drain(..));
            }

            if left_rows.len() < min_rows_per_leaf || right_rows.len() < min_rows_per_leaf {
                continue; // Skip this split — would create an under-sized leaf
            }

            // Compute K-output leaf values via Newton-Raphson per output.
            let leaf_values =
                |rows: &[u32]| -> Vec<f32> {
                    let mut out = vec![0.0_f32; n_outputs];
                    for k in 0..n_outputs {
                        let mut g_sum = 0.0_f32;
                        let mut h_sum = 0.0_f32;
                        for &row in rows {
                            let gp = grads_per_output[k][row as usize];
                            g_sum += gp.grad;
                            h_sum += gp.hess;
                        }
                        out[k] = -g_sum / (h_sum + lambda_l2 + crate::LEAF_EPSILON);
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
                for k in 0..n_outputs {
                    for &row in rows {
                        let gp = grads_per_output[k][row as usize];
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
                    is_categorical: false,
                    categorical_bitset: None,
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

/// Run the full joint multi-output training loop and return a `TrainedModel`
/// + per-output baselines. The model's stumps carry `multi_output_leaf_values`
/// and serialize into both the `Trees` and `MultiOutputLeafValues` sections.
///
/// `targets_per_output[k]` is the target vector for output `k` (length n_rows).
/// `per_output_objective[k]` selects the objective used for output `k`. The
/// `group_id` argument is required when any objective in
/// `per_output_objective` is a ranking objective.
pub fn fit_joint_multi_output(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
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

    let baselines: Vec<f32> = per_output_objective
        .iter()
        .zip(targets_per_output.iter())
        .map(|(obj, targets)| obj.initial_prediction(targets))
        .collect();

    // Per-output prediction vectors, seeded from baselines.
    let mut predictions: Vec<Vec<f32>> = baselines
        .iter()
        .map(|&b| vec![b; n_rows])
        .collect();

    let learning_rate = params.learning_rate;
    let mut all_stumps: Vec<TrainedStump> = Vec::new();
    let mut rounds_completed: usize = 0;

    for round in 0..n_estimators {
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

        // Build one shared tree.
        let result = build_joint_round(params, binned_matrix, &grads_per_output, n_outputs)?;
        if result.stumps.is_empty() {
            break;
        }
        rounds_completed += 1;

        // Apply leaves to predictions + re-encode local node_ids as global
        // tree_id-encoded node_ids so the artifact predictor can group stumps
        // by round.
        for mut stump in result.stumps.into_iter() {
            let (left_k, right_k) = match stump.multi_output_leaf_values.as_ref() {
                Some(v) => (v.0.clone(), v.1.clone()),
                None => continue,
            };

            // Build the row → leaf assignment for this stump's split.
            let feature = stump.split.feature_index as usize;
            let threshold_bin = stump.split.threshold_bin as usize;
            let default_left = stump.split.default_left;
            for row in 0..n_rows {
                let bin = binned_matrix.bins[row * feature_count + feature];
                let went_left = if bin == MISSING_BIN_U8 {
                    default_left
                } else {
                    (bin as usize) <= threshold_bin
                };
                let delta = if went_left { &left_k } else { &right_k };
                for k in 0..n_outputs {
                    predictions[k][row] += learning_rate * delta[k];
                }
            }

            // Re-encode node_id to be globally unique across rounds (joint
            // trainer outputs one tree per round; local_node_id stays the
            // same, tree_index = round).
            let local_node_id = stump.split.node_id;
            stump.split.node_id = encode_tree_node_id(round, local_node_id)
                .map_err(|e| format!("encode_tree_node_id: {e:?}"))?;
            all_stumps.push(stump);
        }
    }

    let model = TrainedModel {
        baseline_prediction: 0.0, // Joint model uses per-output baselines (see summary)
        feature_count,
        stumps: all_stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: format!("joint_multi_output[{n_outputs}]"),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
    };

    let _ = rounds_completed; // for future use (loss history etc.)

    Ok(JointTrainingSummary {
        baselines,
        model,
        per_output_objective_names: per_output_objective
            .iter()
            .map(|o| o.name().to_string())
            .collect(),
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
}

#[derive(Debug, Clone)]
struct JointPredictorStump {
    /// Local node id within the tree (0 = root, 1 = left child of root, 2 = right child, etc.).
    local_node_id: u32,
    feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
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
        for stump in model.stumps.iter() {
            let Some(mo) = stump.multi_output_leaf_values.as_ref() else {
                continue;
            };
            let tree_id = (stump.split.node_id / TREE_NODE_STRIDE) as usize;
            let local_node_id = stump.split.node_id % TREE_NODE_STRIDE;
            let new_idx = stumps.len();
            stumps.push(JointPredictorStump {
                local_node_id,
                feature_index: stump.split.feature_index,
                threshold_bin: stump.split.threshold_bin,
                default_left: stump.split.default_left,
                left_k: mo.0.clone(),
                right_k: mo.1.clone(),
            });
            while rounds.len() <= tree_id {
                rounds.push(Vec::new());
            }
            rounds[tree_id].push(new_idx);
        }

        Ok(JointPredictor {
            n_outputs,
            baselines,
            stumps,
            rounds,
        })
    }

    /// Predict K outputs for a single row. The returned vector has length
    /// `n_outputs`. Each output is the sum of per-tree leaf contributions
    /// plus the per-output baseline.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32> {
        let mut out = self.baselines.clone();
        for tree_stump_indices in &self.rounds {
            if tree_stump_indices.is_empty() {
                continue;
            }
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
                    out[k] += leaf[k];
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

        let mut params = TrainParams::default();
        params.max_depth = 1;
        params.min_data_in_leaf = 1;
        params.lambda_l2 = 0.0;
        params.learning_rate = 1.0;

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
        let bytes = summary.model.clone().to_artifact_bytes().expect("serialize");
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
        assert!(preds_left[0] < 0.0, "expected output 0 left < 0, got {:?}", preds_left);
        assert!(preds_right[0] > 0.0, "expected output 0 right > 0, got {:?}", preds_right);
        assert!(preds_left[1] > 0.0, "expected output 1 left > 0, got {:?}", preds_left);
        assert!(preds_right[1] < 0.0, "expected output 1 right < 0, got {:?}", preds_right);
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

        let mut params = TrainParams::default();
        params.max_depth = 1;
        params.min_data_in_leaf = 1;
        params.lambda_l2 = 0.0;

        let result = build_joint_round(&params, &binned, &grads_per_output, 2).expect("build");
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
}
