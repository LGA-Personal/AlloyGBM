//! Ranking objectives and shared NDCG helpers.
//!
//! This module bundles all learning-to-rank objectives (`queryrmse`,
//! `rank:pairwise`, `rank:ndcg`, `rank_xendcg`, `yetirank`) plus the DCG/NDCG
//! computation helpers and `compute_group_boundaries`. The objectives share
//! these helpers so they live together.

use alloygbm_core::GradientPair;

use super::{SquaredErrorObjective, resolve_boundaries_for_len, sigmoid};
use crate::error::EngineResult;
use crate::traits::ObjectiveOps;

// ── Ranking helpers ───────────────────────────────────────────────────────

/// Compute contiguous group boundaries from a sorted `group_id` array.
///
/// Returns `[0, len_group_0, len_group_0 + len_group_1, ..., row_count]`.
/// The input **must** be sorted such that all rows with the same `group_id`
/// are adjacent.
pub fn compute_group_boundaries(group_id: &[u32]) -> Vec<usize> {
    let mut boundaries = vec![0_usize];
    for i in 1..group_id.len() {
        if group_id[i] != group_id[i - 1] {
            boundaries.push(i);
        }
    }
    boundaries.push(group_id.len());
    boundaries
}

/// DCG (Discounted Cumulative Gain) for a ranking ordered by `scores` desc.
///
/// `labels[i]` is the relevance of document i, `scores[i]` is the predicted
/// score. Documents are ranked by descending `scores`, then DCG is computed
/// as the sum of `(2^label - 1) / log2(rank + 1)` for the top-k positions.
fn dcg_by_scores(labels: &[f32], scores: &[f32], k: usize) -> f32 {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cutoff = k.min(labels.len());
    order
        .iter()
        .take(cutoff)
        .enumerate()
        .map(|(rank, &idx)| {
            let gain = (2.0_f32).powf(labels[idx]) - 1.0;
            let discount = 1.0 / ((rank as f32 + 2.0).log2());
            gain * discount
        })
        .sum()
}

/// Ideal DCG for a set of labels (labels sorted descending).
fn ideal_dcg(labels: &[f32], k: usize) -> f32 {
    let mut sorted = labels.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = k.min(sorted.len());
    sorted
        .iter()
        .take(cutoff)
        .enumerate()
        .map(|(rank, &label)| {
            let gain = (2.0_f32).powf(label) - 1.0;
            let discount = 1.0 / ((rank as f32 + 2.0).log2());
            gain * discount
        })
        .sum()
}

/// NDCG for a single query group.
fn ndcg_for_group(labels: &[f32], scores: &[f32]) -> f32 {
    let k = labels.len();
    let idcg = ideal_dcg(labels, k);
    if idcg <= 0.0 {
        return 1.0; // degenerate: all labels identical or zero
    }
    dcg_by_scores(labels, scores, k) / idcg
}

/// Numerically stable log-sum-exp for softmax computation.
pub(crate) fn log_sum_exp(values: &[f32]) -> f32 {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return 0.0;
    }
    let sum_exp: f32 = values.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

// ── QueryRMSE Objective ──────────────────────────────────────────────────

/// Query-grouped RMSE objective. Gradients are standard MSE per-document,
/// but the loss is reported as the mean of per-group RMSE values, giving
/// equal weight to every query regardless of size.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryRMSEObjective {
    pub group_boundaries: Vec<usize>,
    pub validation_group_boundaries: Option<Vec<usize>>,
}

impl QueryRMSEObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
        }
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }
}

impl ObjectiveOps for QueryRMSEObjective {
    fn objective_name(&self) -> &str {
        "queryrmse"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        // Global weighted mean, same as SquaredError.
        SquaredErrorObjective.initial_prediction(targets, sample_weights)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        // Per-document MSE gradients, same as SquaredError.
        SquaredErrorObjective.compute_gradients(predictions, targets, sample_weights)
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        // Mean of per-group RMSE.
        let boundaries = resolve_boundaries_for_len(
            &self.group_boundaries,
            &self.validation_group_boundaries,
            predictions.len(),
        );
        let num_groups = boundaries.len() - 1;
        if num_groups == 0 {
            return SquaredErrorObjective.loss(predictions, targets, sample_weights);
        }
        let mut group_rmse_sum = 0.0_f64;
        for g in 0..num_groups {
            let start = boundaries[g];
            let end = boundaries[g + 1];
            let group_len = (end - start) as f64;
            let mut mse_sum = 0.0_f64;
            for i in start..end {
                let w = sample_weights.map_or(1.0_f64, |ws| ws[i] as f64);
                let diff = (predictions[i] - targets[i]) as f64;
                mse_sum += w * diff * diff;
            }
            group_rmse_sum += (mse_sum / group_len).sqrt();
        }
        Ok((group_rmse_sum / num_groups as f64) as f32)
    }

    fn requires_group_id(&self) -> bool {
        true
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}

// ── Pairwise Logistic (RankNet) Objective ────────────────────────────────

/// Pairwise logistic ranking objective (RankNet / `rank:pairwise`).
///
/// For each pair (i, j) within a query where `label[i] > label[j]`, computes
/// logistic gradients that push document i's score above document j's score.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseRankingObjective {
    pub group_boundaries: Vec<usize>,
    pub validation_group_boundaries: Option<Vec<usize>>,
}

impl PairwiseRankingObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
        }
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }

    /// Compute per-document gradients/hessians from pairwise logistic loss.
    fn pairwise_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<(Vec<f32>, Vec<f32>)> {
        let n = predictions.len();
        let mut grads = vec![0.0_f32; n];
        let mut hesses = vec![0.0_f32; n];
        let num_groups = self.group_boundaries.len() - 1;

        for g in 0..num_groups {
            let start = self.group_boundaries[g];
            let end = self.group_boundaries[g + 1];
            for i in start..end {
                for j in (i + 1)..end {
                    if targets[i] == targets[j] {
                        continue;
                    }
                    let (hi, lo) = if targets[i] > targets[j] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    // score difference: higher-labeled doc minus lower
                    let s = predictions[hi] - predictions[lo];
                    // rho = sigma(-s) = 1 / (1 + exp(s))
                    let rho = sigmoid(-s);
                    let lambda = -rho;
                    let hess_pair = rho * (1.0 - rho);

                    grads[hi] += lambda;
                    grads[lo] -= lambda;
                    hesses[hi] += hess_pair;
                    hesses[lo] += hess_pair;
                }
            }
        }
        Ok((grads, hesses))
    }
}

impl ObjectiveOps for PairwiseRankingObjective {
    fn objective_name(&self) -> &str {
        "rank_pairwise"
    }

    fn initial_prediction(
        &self,
        _targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        Ok(0.0)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        let (grads, hesses) = self.pairwise_gradients(predictions, targets, sample_weights)?;
        let mut pairs = Vec::with_capacity(grads.len());
        for i in 0..grads.len() {
            let hess = hesses[i].max(1e-7);
            pairs.push(GradientPair::new(grads[i], hess)?);
        }
        Ok(pairs)
    }

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let (grads, hesses) = self.pairwise_gradients(predictions, targets, sample_weights)?;
        buffer.clear();
        if buffer.capacity() < grads.len() {
            buffer.reserve(grads.len() - buffer.capacity());
        }
        for i in 0..grads.len() {
            let hess = hesses[i].max(1e-7);
            buffer.push(GradientPair {
                grad: grads[i],
                hess,
            });
        }
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        let boundaries = resolve_boundaries_for_len(
            &self.group_boundaries,
            &self.validation_group_boundaries,
            predictions.len(),
        );
        let num_groups = boundaries.len() - 1;
        let mut total_loss = 0.0_f64;
        let mut total_pairs = 0_u64;
        for g in 0..num_groups {
            let start = boundaries[g];
            let end = boundaries[g + 1];
            for i in start..end {
                for j in (i + 1)..end {
                    if targets[i] == targets[j] {
                        continue;
                    }
                    let (hi, lo) = if targets[i] > targets[j] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    let s = (predictions[hi] - predictions[lo]) as f64;
                    let pair_loss = if s >= 0.0 {
                        (-s).exp().ln_1p()
                    } else {
                        -s + s.exp().ln_1p()
                    };
                    total_loss += pair_loss;
                    total_pairs += 1;
                }
            }
        }
        if total_pairs == 0 {
            return Ok(0.0);
        }
        Ok((total_loss / total_pairs as f64) as f32)
    }

    fn requires_group_id(&self) -> bool {
        true
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}

// ── LambdaMART (rank:ndcg) Objective ─────────────────────────────────────

/// LambdaMART ranking objective (`rank:ndcg`).
///
/// Like RankNet but weights each pair's gradient by the absolute change in
/// NDCG that would result from swapping the two documents' positions.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaMARTObjective {
    pub group_boundaries: Vec<usize>,
    pub validation_group_boundaries: Option<Vec<usize>>,
}

impl LambdaMARTObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
        }
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }
}

impl ObjectiveOps for LambdaMARTObjective {
    fn objective_name(&self) -> &str {
        "rank_ndcg"
    }

    fn initial_prediction(
        &self,
        _targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        Ok(0.0)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        let n = predictions.len();
        let mut grads = vec![0.0_f32; n];
        let mut hesses = vec![0.0_f32; n];
        let num_groups = self.group_boundaries.len() - 1;

        for g in 0..num_groups {
            let start = self.group_boundaries[g];
            let end = self.group_boundaries[g + 1];
            let group_labels = &targets[start..end];
            let group_scores = &predictions[start..end];
            let group_len = end - start;

            // Compute ideal DCG for normalization.
            let idcg = ideal_dcg(group_labels, group_len);
            if idcg <= 0.0 {
                continue; // all labels identical, no useful pairs
            }
            let inv_idcg = 1.0 / idcg;

            // Sort by predictions descending to get ranks.
            let mut order: Vec<usize> = (0..group_len).collect();
            order.sort_by(|&a, &b| {
                group_scores[b]
                    .partial_cmp(&group_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut ranks = vec![0_usize; group_len];
            for (rank, &idx) in order.iter().enumerate() {
                ranks[idx] = rank;
            }

            for i in 0..group_len {
                for j in (i + 1)..group_len {
                    if group_labels[i] == group_labels[j] {
                        continue;
                    }
                    let (hi, lo) = if group_labels[i] > group_labels[j] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    let s = group_scores[hi] - group_scores[lo];
                    let rho = sigmoid(-s);

                    // ΔNDCG if positions of hi and lo were swapped.
                    let gain_hi = (2.0_f32).powf(group_labels[hi]) - 1.0;
                    let gain_lo = (2.0_f32).powf(group_labels[lo]) - 1.0;
                    let discount_hi = 1.0 / ((ranks[hi] as f32 + 2.0).log2());
                    let discount_lo = 1.0 / ((ranks[lo] as f32 + 2.0).log2());
                    let delta_ndcg =
                        ((gain_hi - gain_lo) * (discount_hi - discount_lo)).abs() * inv_idcg;

                    let lambda = -rho * delta_ndcg;
                    let hess_pair = rho * (1.0 - rho) * delta_ndcg;

                    grads[start + hi] += lambda;
                    grads[start + lo] -= lambda;
                    hesses[start + hi] += hess_pair;
                    hesses[start + lo] += hess_pair;
                }
            }
        }

        let mut pairs = Vec::with_capacity(n);
        for i in 0..n {
            let hess = hesses[i].max(1e-7);
            pairs.push(GradientPair::new(grads[i], hess)?);
        }
        Ok(pairs)
    }

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let pairs = self.compute_gradients(predictions, targets, sample_weights)?;
        buffer.clear();
        buffer.extend(pairs);
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        // 1 - mean NDCG across all query groups.
        let boundaries = resolve_boundaries_for_len(
            &self.group_boundaries,
            &self.validation_group_boundaries,
            predictions.len(),
        );
        let num_groups = boundaries.len() - 1;
        if num_groups == 0 {
            return Ok(0.0);
        }
        let mut ndcg_sum = 0.0_f64;
        for g in 0..num_groups {
            let start = boundaries[g];
            let end = boundaries[g + 1];
            ndcg_sum += ndcg_for_group(&targets[start..end], &predictions[start..end]) as f64;
        }
        Ok((1.0 - ndcg_sum / num_groups as f64) as f32)
    }

    fn requires_group_id(&self) -> bool {
        true
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}

// ── XE_NDCG Objective ────────────────────────────────────────────────────

/// Cross-entropy approximation to NDCG (`rank_xendcg`).
///
/// Treats ranking as a classification problem: the "ideal" distribution is
/// a softmax over relevance labels, and the "predicted" distribution is a
/// softmax over predicted scores. The loss is the cross-entropy between them.
/// Gradients are O(n) per group, unlike the O(n²) pairwise objectives.
#[derive(Debug, Clone, PartialEq)]
pub struct XeNDCGObjective {
    pub group_boundaries: Vec<usize>,
    pub validation_group_boundaries: Option<Vec<usize>>,
}

impl XeNDCGObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
        }
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }
}

impl ObjectiveOps for XeNDCGObjective {
    fn objective_name(&self) -> &str {
        "rank_xendcg"
    }

    fn initial_prediction(
        &self,
        _targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        Ok(0.0)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        let n = predictions.len();
        let mut grads = vec![0.0_f32; n];
        let mut hesses = vec![0.0_f32; n];
        let num_groups = self.group_boundaries.len() - 1;

        for g in 0..num_groups {
            let start = self.group_boundaries[g];
            let end = self.group_boundaries[g + 1];
            let group_len = end - start;
            if group_len <= 1 {
                continue;
            }

            // Ideal distribution: softmax of relevance labels.
            let label_slice: Vec<f32> = targets[start..end].to_vec();
            let label_lse = log_sum_exp(&label_slice);
            // Predicted distribution: softmax of scores.
            let score_slice: Vec<f32> = predictions[start..end].to_vec();
            let score_lse = log_sum_exp(&score_slice);

            for i in 0..group_len {
                let ideal_prob = (label_slice[i] - label_lse).exp();
                let pred_prob = (score_slice[i] - score_lse).exp();
                // Gradient of cross-entropy w.r.t. scores.
                grads[start + i] = pred_prob - ideal_prob;
                // Hessian for Newton step.
                hesses[start + i] = (pred_prob * (1.0 - pred_prob)).max(1e-7);
            }
        }

        let mut pairs = Vec::with_capacity(n);
        for i in 0..n {
            pairs.push(GradientPair::new(grads[i], hesses[i].max(1e-7))?);
        }
        Ok(pairs)
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        let boundaries = resolve_boundaries_for_len(
            &self.group_boundaries,
            &self.validation_group_boundaries,
            predictions.len(),
        );
        let num_groups = boundaries.len() - 1;
        if num_groups == 0 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0_f64;
        for g in 0..num_groups {
            let start = boundaries[g];
            let end = boundaries[g + 1];
            if end - start <= 1 {
                continue;
            }
            let label_slice: Vec<f32> = targets[start..end].to_vec();
            let label_lse = log_sum_exp(&label_slice);
            let score_slice: Vec<f32> = predictions[start..end].to_vec();
            let score_lse = log_sum_exp(&score_slice);

            let mut group_loss = 0.0_f64;
            for i in 0..(end - start) {
                let ideal_prob = ((label_slice[i] - label_lse) as f64).exp();
                let log_pred_prob = (score_slice[i] - score_lse) as f64;
                group_loss -= ideal_prob * log_pred_prob;
            }
            total_loss += group_loss;
        }
        Ok((total_loss / num_groups as f64) as f32)
    }

    fn requires_group_id(&self) -> bool {
        true
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}

// ── YetiRank Objective ───────────────────────────────────────────────────

/// YetiRank / YetiRankPairwise ranking objective.
///
/// Uses stochastic pairwise comparisons with NDCG-weighted gradients.
/// Each gradient computation samples random permutations to estimate
/// position-dependent importance weights, providing a smoother gradient
/// signal than deterministic LambdaMART.
#[derive(Debug, Clone, PartialEq)]
pub struct YetiRankObjective {
    pub group_boundaries: Vec<usize>,
    pub validation_group_boundaries: Option<Vec<usize>>,
    /// Number of random permutations to sample per query per gradient call.
    pub num_permutations: usize,
    /// Base seed for reproducible sampling.
    pub seed: u64,
}

impl YetiRankObjective {
    pub fn new(group_id: &[u32], num_permutations: usize, seed: u64) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            num_permutations,
            seed,
        }
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }

    /// Simple deterministic hash-based PRNG for permutation generation.
    /// Returns a pseudo-random u64 from a state, advancing the state.
    fn next_random(state: &mut u64) -> u64 {
        // SplitMix64
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl ObjectiveOps for YetiRankObjective {
    fn objective_name(&self) -> &str {
        "yetirank"
    }

    fn initial_prediction(
        &self,
        _targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        Ok(0.0)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
        let n = predictions.len();
        let mut grads = vec![0.0_f32; n];
        let mut hesses = vec![0.0_f32; n];
        let num_groups = self.group_boundaries.len() - 1;

        for g in 0..num_groups {
            let start = self.group_boundaries[g];
            let end = self.group_boundaries[g + 1];
            let group_len = end - start;
            if group_len <= 1 {
                continue;
            }

            let group_labels = &targets[start..end];
            let group_scores = &predictions[start..end];

            // Ideal DCG for normalization.
            let idcg = ideal_dcg(group_labels, group_len);
            if idcg <= 0.0 {
                continue;
            }
            let inv_idcg = 1.0 / idcg;

            // For each permutation, compute position-based weights.
            let inv_perms = 1.0 / self.num_permutations as f32;

            for perm_idx in 0..self.num_permutations {
                // Seed deterministically from group + permutation index.
                let mut rng_state = self
                    .seed
                    .wrapping_add(g as u64 * 1_000_003)
                    .wrapping_add(perm_idx as u64 * 7);

                // Create a permuted ordering biased by current scores.
                // Add noise to scores to sample different orderings.
                let mut noisy_order: Vec<usize> = (0..group_len).collect();
                let mut noisy_scores = group_scores.to_vec();
                for s in &mut noisy_scores {
                    let r = Self::next_random(&mut rng_state);
                    let noise = ((r as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
                    *s += noise * 0.5;
                }
                noisy_order.sort_by(|&a, &b| {
                    noisy_scores[b]
                        .partial_cmp(&noisy_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Compute ranks in this permutation.
                let mut perm_ranks = vec![0_usize; group_len];
                for (rank, &idx) in noisy_order.iter().enumerate() {
                    perm_ranks[idx] = rank;
                }

                // Pairwise gradients weighted by delta-NDCG in this permutation.
                for i in 0..group_len {
                    for j in (i + 1)..group_len {
                        if group_labels[i] == group_labels[j] {
                            continue;
                        }
                        let (hi, lo) = if group_labels[i] > group_labels[j] {
                            (i, j)
                        } else {
                            (j, i)
                        };
                        let s = group_scores[hi] - group_scores[lo];
                        let rho = sigmoid(-s);

                        let gain_hi = (2.0_f32).powf(group_labels[hi]) - 1.0;
                        let gain_lo = (2.0_f32).powf(group_labels[lo]) - 1.0;
                        let discount_hi = 1.0 / ((perm_ranks[hi] as f32 + 2.0).log2());
                        let discount_lo = 1.0 / ((perm_ranks[lo] as f32 + 2.0).log2());
                        let delta_ndcg =
                            ((gain_hi - gain_lo) * (discount_hi - discount_lo)).abs() * inv_idcg;

                        let lambda = -rho * delta_ndcg * inv_perms;
                        let hess_pair = rho * (1.0 - rho) * delta_ndcg * inv_perms;

                        grads[start + hi] += lambda;
                        grads[start + lo] -= lambda;
                        hesses[start + hi] += hess_pair;
                        hesses[start + lo] += hess_pair;
                    }
                }
            }
        }

        let mut pairs = Vec::with_capacity(n);
        for i in 0..n {
            let hess = hesses[i].max(1e-7);
            pairs.push(GradientPair::new(grads[i], hess)?);
        }
        Ok(pairs)
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        _sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        let boundaries = resolve_boundaries_for_len(
            &self.group_boundaries,
            &self.validation_group_boundaries,
            predictions.len(),
        );
        let num_groups = boundaries.len() - 1;
        if num_groups == 0 {
            return Ok(0.0);
        }
        let mut ndcg_sum = 0.0_f64;
        for g in 0..num_groups {
            let start = boundaries[g];
            let end = boundaries[g + 1];
            ndcg_sum += ndcg_for_group(&targets[start..end], &predictions[start..end]) as f64;
        }
        Ok((1.0 - ndcg_sum / num_groups as f64) as f32)
    }

    fn requires_group_id(&self) -> bool {
        true
    }

    fn supports_leaf_refinement(&self) -> bool {
        false
    }
}
