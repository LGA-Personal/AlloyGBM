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
    LambdaMARTObjective, ObjectiveOps, PairwiseRankingObjective, QueryRMSEObjective,
    SquaredErrorObjective, TrainedStump, XeNDCGObjective,
};
use alloygbm_core::{
    BinnedMatrix, GradientPair, LeafValue, MISSING_BIN_U8, NodeStats, SplitCandidate, TrainParams,
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
