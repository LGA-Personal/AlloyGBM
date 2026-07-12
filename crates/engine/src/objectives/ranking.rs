//! Ranking objectives and shared NDCG helpers.
//!
//! This module bundles all learning-to-rank objectives (`queryrmse`,
//! `rank:pairwise`, `rank:ndcg`, `rank_xendcg`, `yetirank`) plus the DCG/NDCG
//! computation helpers and `compute_group_boundaries`. The objectives share
//! these helpers so they live together.

use alloygbm_core::GradientPair;
use rayon::prelude::*;

use super::{SquaredErrorObjective, resolve_boundaries_for_len, sigmoid};
use crate::error::{EngineError, EngineResult};
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
fn log_sum_exp(values: &[f32]) -> f32 {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return 0.0;
    }
    let sum_exp: f32 = values.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

#[derive(Debug)]
struct GroupGradientChunk {
    start: usize,
    grads: Vec<f32>,
    hesses: Vec<f32>,
}

fn gradient_pairs_from_parts(grads: &[f32], hesses: &[f32]) -> EngineResult<Vec<GradientPair>> {
    let mut pairs = Vec::with_capacity(grads.len());
    for i in 0..grads.len() {
        pairs.push(GradientPair::new(grads[i], hesses[i].max(1e-7))?);
    }
    Ok(pairs)
}

fn fill_gradient_pair_buffer(
    grads: &[f32],
    hesses: &[f32],
    buffer: &mut Vec<GradientPair>,
) -> EngineResult<()> {
    for i in 0..grads.len() {
        let _ = GradientPair::new(grads[i], hesses[i].max(1e-7))?;
    }
    buffer.clear();
    if buffer.capacity() < grads.len() {
        buffer.reserve(grads.len() - buffer.capacity());
    }
    for i in 0..grads.len() {
        buffer.push(GradientPair {
            grad: grads[i],
            hess: hesses[i].max(1e-7),
        });
    }
    Ok(())
}

fn merge_group_chunks(chunks: Vec<GroupGradientChunk>, grads: &mut [f32], hesses: &mut [f32]) {
    for chunk in chunks {
        let end = chunk.start + chunk.grads.len();
        grads[chunk.start..end].copy_from_slice(&chunk.grads);
        hesses[chunk.start..end].copy_from_slice(&chunk.hesses);
    }
}

fn gains_for_labels(labels: &[f32]) -> Vec<f32> {
    labels
        .iter()
        .map(|&label| (2.0_f32).powf(label) - 1.0)
        .collect()
}

fn discounts_for_ranks(ranks: &[usize]) -> Vec<f32> {
    ranks
        .iter()
        .map(|&rank| 1.0 / ((rank as f32 + 2.0).log2()))
        .collect()
}

fn validate_ranking_sigma(sigma: f32) -> EngineResult<f32> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "ranking_sigma must be finite and > 0".to_string(),
        ));
    }
    Ok(sigma)
}

fn validate_lambdarank_truncation_level(level: Option<usize>) -> EngineResult<Option<usize>> {
    if matches!(level, Some(0)) {
        return Err(EngineError::InvalidConfig(
            "lambdarank_truncation_level must be None or >= 1".to_string(),
        ));
    }
    Ok(level)
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
    pub sigma: f32,
}

impl PairwiseRankingObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            sigma: 1.0,
        }
    }

    pub fn new_with_sigma(group_id: &[u32], sigma: f32) -> EngineResult<Self> {
        Ok(Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            sigma: validate_ranking_sigma(sigma)?,
        })
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
        let chunks: Vec<GroupGradientChunk> = self
            .group_boundaries
            .par_windows(2)
            .map(|group| {
                let start = group[0];
                let end = group[1];
                let group_len = end - start;
                let mut group_grads = vec![0.0_f32; group_len];
                let mut group_hesses = vec![0.0_f32; group_len];
                for i in 0..group_len {
                    for j in (i + 1)..group_len {
                        if targets[start + i] == targets[start + j] {
                            continue;
                        }
                        let (hi, lo) = if targets[start + i] > targets[start + j] {
                            (i, j)
                        } else {
                            (j, i)
                        };
                        let s = predictions[start + hi] - predictions[start + lo];
                        let rho = sigmoid(-self.sigma * s);
                        let lambda = -self.sigma * rho;
                        let hess_pair = self.sigma * self.sigma * rho * (1.0 - rho);

                        group_grads[hi] += lambda;
                        group_grads[lo] -= lambda;
                        group_hesses[hi] += hess_pair;
                        group_hesses[lo] += hess_pair;
                    }
                }
                GroupGradientChunk {
                    start,
                    grads: group_grads,
                    hesses: group_hesses,
                }
            })
            .collect();

        merge_group_chunks(chunks, &mut grads, &mut hesses);
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
        gradient_pairs_from_parts(&grads, &hesses)
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
                    let s = (self.sigma * (predictions[hi] - predictions[lo])) as f64;
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
    pub sigma: f32,
    pub truncation_level: Option<usize>,
}

impl LambdaMARTObjective {
    pub fn new(group_id: &[u32]) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            sigma: 1.0,
            truncation_level: None,
        }
    }

    pub fn new_with_sigma(group_id: &[u32], sigma: f32) -> EngineResult<Self> {
        Self::new_with_sigma_and_truncation(group_id, sigma, None)
    }

    pub fn new_with_sigma_and_truncation(
        group_id: &[u32],
        sigma: f32,
        truncation_level: Option<usize>,
    ) -> EngineResult<Self> {
        Ok(Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            sigma: validate_ranking_sigma(sigma)?,
            truncation_level: validate_lambdarank_truncation_level(truncation_level)?,
        })
    }

    pub fn with_validation_group(mut self, validation_group_id: &[u32]) -> Self {
        self.validation_group_boundaries = Some(compute_group_boundaries(validation_group_id));
        self
    }

    fn lambdamart_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
    ) -> EngineResult<(Vec<f32>, Vec<f32>)> {
        let n = predictions.len();
        let mut grads = vec![0.0_f32; n];
        let mut hesses = vec![0.0_f32; n];
        let chunks: Vec<GroupGradientChunk> = self
            .group_boundaries
            .par_windows(2)
            .map(|group| {
                let start = group[0];
                let end = group[1];
                let group_labels = &targets[start..end];
                let group_scores = &predictions[start..end];
                let group_len = end - start;
                let idcg = ideal_dcg(group_labels, group_len);
                let mut group_grads = vec![0.0_f32; group_len];
                let mut group_hesses = vec![0.0_f32; group_len];
                if idcg <= 0.0 {
                    return GroupGradientChunk {
                        start,
                        grads: group_grads,
                        hesses: group_hesses,
                    };
                }
                let inv_idcg = 1.0 / idcg;
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
                let gains = gains_for_labels(group_labels);
                let discounts = discounts_for_ranks(&ranks);
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
                        if let Some(k) = self.truncation_level
                            && ranks[hi] >= k
                            && ranks[lo] >= k
                        {
                            continue;
                        }
                        let s = group_scores[hi] - group_scores[lo];
                        let rho = sigmoid(-self.sigma * s);
                        let delta_ndcg =
                            ((gains[hi] - gains[lo]) * (discounts[hi] - discounts[lo])).abs()
                                * inv_idcg;
                        let lambda = -self.sigma * rho * delta_ndcg;
                        let hess_pair = self.sigma * self.sigma * rho * (1.0 - rho) * delta_ndcg;
                        group_grads[hi] += lambda;
                        group_grads[lo] -= lambda;
                        group_hesses[hi] += hess_pair;
                        group_hesses[lo] += hess_pair;
                    }
                }
                GroupGradientChunk {
                    start,
                    grads: group_grads,
                    hesses: group_hesses,
                }
            })
            .collect();
        merge_group_chunks(chunks, &mut grads, &mut hesses);
        Ok((grads, hesses))
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
        let (grads, hesses) = self.lambdamart_gradients(predictions, targets)?;
        gradient_pairs_from_parts(&grads, &hesses)
    }

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let _ = sample_weights;
        let (grads, hesses) = self.lambdamart_gradients(predictions, targets)?;
        fill_gradient_pair_buffer(&grads, &hesses, buffer)
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
        let chunks: Vec<GroupGradientChunk> = self
            .group_boundaries
            .par_windows(2)
            .map(|group| {
                let start = group[0];
                let end = group[1];
                let group_len = end - start;
                let mut group_grads = vec![0.0_f32; group_len];
                let mut group_hesses = vec![0.0_f32; group_len];
                if group_len <= 1 {
                    return GroupGradientChunk {
                        start,
                        grads: group_grads,
                        hesses: group_hesses,
                    };
                }
                let label_slice = &targets[start..end];
                let score_slice = &predictions[start..end];
                let label_lse = log_sum_exp(label_slice);
                let score_lse = log_sum_exp(score_slice);

                for i in 0..group_len {
                    let ideal_prob = (label_slice[i] - label_lse).exp();
                    let pred_prob = (score_slice[i] - score_lse).exp();
                    group_grads[i] = pred_prob - ideal_prob;
                    group_hesses[i] = (pred_prob * (1.0 - pred_prob)).max(1e-7);
                }
                GroupGradientChunk {
                    start,
                    grads: group_grads,
                    hesses: group_hesses,
                }
            })
            .collect();
        merge_group_chunks(chunks, &mut grads, &mut hesses);
        gradient_pairs_from_parts(&grads, &hesses)
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
    pub sigma: f32,
}

impl YetiRankObjective {
    pub fn new(group_id: &[u32], num_permutations: usize, seed: u64) -> Self {
        Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            num_permutations,
            seed,
            sigma: 1.0,
        }
    }

    pub fn new_with_sigma(
        group_id: &[u32],
        num_permutations: usize,
        seed: u64,
        sigma: f32,
    ) -> EngineResult<Self> {
        Ok(Self {
            group_boundaries: compute_group_boundaries(group_id),
            validation_group_boundaries: None,
            num_permutations,
            seed,
            sigma: validate_ranking_sigma(sigma)?,
        })
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
        let chunks: Vec<GroupGradientChunk> = self
            .group_boundaries
            .par_windows(2)
            .enumerate()
            .map(|(group_idx, group)| {
                let start = group[0];
                let end = group[1];
                let group_len = end - start;
                let mut group_grads = vec![0.0_f32; group_len];
                let mut group_hesses = vec![0.0_f32; group_len];
                if group_len <= 1 {
                    return GroupGradientChunk {
                        start,
                        grads: group_grads,
                        hesses: group_hesses,
                    };
                }
                let group_labels = &targets[start..end];
                let group_scores = &predictions[start..end];
                let idcg = ideal_dcg(group_labels, group_len);
                if idcg <= 0.0 {
                    return GroupGradientChunk {
                        start,
                        grads: group_grads,
                        hesses: group_hesses,
                    };
                }
                let gains = gains_for_labels(group_labels);
                let inv_idcg = 1.0 / idcg;
                let inv_perms = 1.0 / self.num_permutations as f32;

                for perm_idx in 0..self.num_permutations {
                    let mut rng_state = self
                        .seed
                        .wrapping_add(group_idx as u64 * 1_000_003)
                        .wrapping_add(perm_idx as u64 * 7);
                    let mut noisy_order: Vec<usize> = (0..group_len).collect();
                    let mut noisy_scores = group_scores.to_vec();
                    for score in &mut noisy_scores {
                        let r = Self::next_random(&mut rng_state);
                        let noise = ((r as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
                        *score += noise * 0.5;
                    }
                    noisy_order.sort_by(|&a, &b| {
                        noisy_scores[b]
                            .partial_cmp(&noisy_scores[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let mut perm_ranks = vec![0_usize; group_len];
                    for (rank, &idx) in noisy_order.iter().enumerate() {
                        perm_ranks[idx] = rank;
                    }
                    let discounts = discounts_for_ranks(&perm_ranks);

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
                            let rho = sigmoid(-self.sigma * s);
                            let delta_ndcg =
                                ((gains[hi] - gains[lo]) * (discounts[hi] - discounts[lo])).abs()
                                    * inv_idcg;
                            let lambda = -self.sigma * rho * delta_ndcg * inv_perms;
                            let hess_pair = self.sigma
                                * self.sigma
                                * rho
                                * (1.0 - rho)
                                * delta_ndcg
                                * inv_perms;
                            group_grads[hi] += lambda;
                            group_grads[lo] -= lambda;
                            group_hesses[hi] += hess_pair;
                            group_hesses[lo] += hess_pair;
                        }
                    }
                }

                GroupGradientChunk {
                    start,
                    grads: group_grads,
                    hesses: group_hesses,
                }
            })
            .collect();
        merge_group_chunks(chunks, &mut grads, &mut hesses);
        gradient_pairs_from_parts(&grads, &hesses)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_gradient_pairs_close(
        actual: &[GradientPair],
        expected: &[GradientPair],
        tolerance: f32,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual_pair, expected_pair)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual_pair.grad - expected_pair.grad).abs() <= tolerance,
                "grad mismatch at {idx}: actual={}, expected={}",
                actual_pair.grad,
                expected_pair.grad
            );
            assert!(
                (actual_pair.hess - expected_pair.hess).abs() <= tolerance,
                "hess mismatch at {idx}: actual={}, expected={}",
                actual_pair.hess,
                expected_pair.hess
            );
        }
    }

    fn pairs_from_parts(grads: &[f32], hesses: &[f32]) -> Vec<GradientPair> {
        grads
            .iter()
            .zip(hesses.iter())
            .map(|(&grad, &hess)| GradientPair::new(grad, hess.max(1e-7)).unwrap())
            .collect()
    }

    fn serial_pairwise_reference(
        boundaries: &[usize],
        predictions: &[f32],
        targets: &[f32],
    ) -> Vec<GradientPair> {
        let mut grads = vec![0.0_f32; predictions.len()];
        let mut hesses = vec![0.0_f32; predictions.len()];

        for group in boundaries.windows(2) {
            let start = group[0];
            let end = group[1];
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
                    let rho = sigmoid(-(predictions[hi] - predictions[lo]));
                    let lambda = -rho;
                    let hess_pair = rho * (1.0 - rho);
                    grads[hi] += lambda;
                    grads[lo] -= lambda;
                    hesses[hi] += hess_pair;
                    hesses[lo] += hess_pair;
                }
            }
        }

        pairs_from_parts(&grads, &hesses)
    }

    fn serial_lambdamart_reference(
        boundaries: &[usize],
        predictions: &[f32],
        targets: &[f32],
    ) -> Vec<GradientPair> {
        let mut grads = vec![0.0_f32; predictions.len()];
        let mut hesses = vec![0.0_f32; predictions.len()];

        for group in boundaries.windows(2) {
            let start = group[0];
            let end = group[1];
            let group_labels = &targets[start..end];
            let group_scores = &predictions[start..end];
            let group_len = end - start;
            let idcg = ideal_dcg(group_labels, group_len);
            if idcg <= 0.0 {
                continue;
            }
            let inv_idcg = 1.0 / idcg;
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
                    let rho = sigmoid(-(group_scores[hi] - group_scores[lo]));
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

        pairs_from_parts(&grads, &hesses)
    }

    fn serial_xendcg_reference(
        boundaries: &[usize],
        predictions: &[f32],
        targets: &[f32],
    ) -> Vec<GradientPair> {
        let mut grads = vec![0.0_f32; predictions.len()];
        let mut hesses = vec![0.0_f32; predictions.len()];

        for group in boundaries.windows(2) {
            let start = group[0];
            let end = group[1];
            let group_len = end - start;
            if group_len <= 1 {
                continue;
            }
            let label_slice = &targets[start..end];
            let score_slice = &predictions[start..end];
            let label_lse = log_sum_exp(label_slice);
            let score_lse = log_sum_exp(score_slice);

            for idx in 0..group_len {
                let ideal_prob = (label_slice[idx] - label_lse).exp();
                let pred_prob = (score_slice[idx] - score_lse).exp();
                grads[start + idx] = pred_prob - ideal_prob;
                hesses[start + idx] = (pred_prob * (1.0 - pred_prob)).max(1e-7);
            }
        }

        pairs_from_parts(&grads, &hesses)
    }

    fn serial_yetirank_reference(
        boundaries: &[usize],
        predictions: &[f32],
        targets: &[f32],
        num_permutations: usize,
        seed: u64,
    ) -> Vec<GradientPair> {
        let mut grads = vec![0.0_f32; predictions.len()];
        let mut hesses = vec![0.0_f32; predictions.len()];

        for (group_idx, group) in boundaries.windows(2).enumerate() {
            let start = group[0];
            let end = group[1];
            let group_len = end - start;
            if group_len <= 1 {
                continue;
            }

            let group_labels = &targets[start..end];
            let group_scores = &predictions[start..end];
            let idcg = ideal_dcg(group_labels, group_len);
            if idcg <= 0.0 {
                continue;
            }
            let inv_idcg = 1.0 / idcg;
            let inv_perms = 1.0 / num_permutations as f32;
            for perm_idx in 0..num_permutations {
                let mut rng_state = seed
                    .wrapping_add(group_idx as u64 * 1_000_003)
                    .wrapping_add(perm_idx as u64 * 7);
                let mut noisy_order: Vec<usize> = (0..group_len).collect();
                let mut noisy_scores = group_scores.to_vec();
                for score in &mut noisy_scores {
                    let r = YetiRankObjective::next_random(&mut rng_state);
                    let noise = ((r as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
                    *score += noise * 0.5;
                }
                noisy_order.sort_by(|&a, &b| {
                    noisy_scores[b]
                        .partial_cmp(&noisy_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut perm_ranks = vec![0_usize; group_len];
                for (rank, &idx) in noisy_order.iter().enumerate() {
                    perm_ranks[idx] = rank;
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
                        let rho = sigmoid(-(group_scores[hi] - group_scores[lo]));
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

        pairs_from_parts(&grads, &hesses)
    }

    #[test]
    fn pairwise_parallel_gradients_match_serial_reference() {
        let group_id = [0, 0, 0, 1, 1, 1, 1];
        let predictions = [0.3, -0.2, 0.9, 1.1, -0.4, 0.0, 0.7];
        let targets = [2.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0];
        let objective = PairwiseRankingObjective::new(&group_id);
        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();
        let expected =
            serial_pairwise_reference(&compute_group_boundaries(&group_id), &predictions, &targets);
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn pairwise_sigma_scales_logistic_margin_gradient_and_hessian() {
        let group_id = [0, 0];
        let predictions = [0.8, -0.2];
        let targets = [1.0, 0.0];
        let objective = PairwiseRankingObjective::new_with_sigma(&group_id, 2.0).unwrap();

        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();

        let rho = sigmoid(-2.0 * (predictions[0] - predictions[1]));
        let expected_grad = -2.0 * rho;
        let expected_hess = 4.0 * rho * (1.0 - rho);
        let expected = vec![
            GradientPair::new(expected_grad, expected_hess).unwrap(),
            GradientPair::new(-expected_grad, expected_hess).unwrap(),
        ];
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn lambdamart_parallel_gradients_match_serial_reference() {
        let group_id = [0, 0, 0, 0, 1, 1, 1];
        let predictions = [0.8, -0.1, 0.4, 0.2, 1.2, -0.5, 0.3];
        let targets = [3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0];
        let objective = LambdaMARTObjective::new(&group_id);
        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();
        let expected = serial_lambdamart_reference(
            &compute_group_boundaries(&group_id),
            &predictions,
            &targets,
        );
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn lambdamart_sigma_scales_pairwise_component_without_changing_ndcg_weight() {
        let group_id = [0, 0];
        let predictions = [0.8, -0.2];
        let targets = [1.0, 0.0];
        let objective = LambdaMARTObjective::new_with_sigma(&group_id, 2.0).unwrap();

        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();

        let rho = sigmoid(-2.0 * (predictions[0] - predictions[1]));
        let delta_ndcg = 1.0 - 1.0 / 3.0_f32.log2();
        let expected_grad = -2.0 * rho * delta_ndcg;
        let expected_hess = 4.0 * rho * (1.0 - rho) * delta_ndcg;
        let expected = vec![
            GradientPair::new(expected_grad, expected_hess).unwrap(),
            GradientPair::new(-expected_grad, expected_hess).unwrap(),
        ];
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn lambdamart_truncation_level_limits_pairs_to_current_top_k() {
        let group_id = [0, 0, 0];
        let predictions = [3.0, 2.0, 1.0];
        let targets = [0.0, 2.0, 1.0];
        let objective =
            LambdaMARTObjective::new_with_sigma_and_truncation(&group_id, 1.0, Some(1)).unwrap();

        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();

        let idcg = ideal_dcg(&targets, targets.len());
        let inv_idcg = 1.0 / idcg;
        let gains = gains_for_labels(&targets);
        let discounts = discounts_for_ranks(&[0, 1, 2]);
        let mut expected_grads = [0.0_f32; 3];
        let mut expected_hesses = [0.0_f32; 3];
        for (hi, lo) in [(1_usize, 0_usize), (2, 0)] {
            let rho = sigmoid(-(predictions[hi] - predictions[lo]));
            let delta_ndcg =
                ((gains[hi] - gains[lo]) * (discounts[hi] - discounts[lo])).abs() * inv_idcg;
            let lambda = -rho * delta_ndcg;
            let hess_pair = rho * (1.0 - rho) * delta_ndcg;
            expected_grads[hi] += lambda;
            expected_grads[lo] -= lambda;
            expected_hesses[hi] += hess_pair;
            expected_hesses[lo] += hess_pair;
        }
        let expected = expected_grads
            .iter()
            .zip(expected_hesses.iter())
            .map(|(&grad, &hess)| GradientPair::new(grad, hess).unwrap())
            .collect::<Vec<_>>();

        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn ranking_sigma_must_be_positive_and_finite() {
        let group_id = [0, 0];

        assert!(PairwiseRankingObjective::new_with_sigma(&group_id, 0.0).is_err());
        assert!(LambdaMARTObjective::new_with_sigma(&group_id, -1.0).is_err());
        assert!(YetiRankObjective::new_with_sigma(&group_id, 3, 17, f32::INFINITY).is_err());
    }

    #[test]
    fn xendcg_parallel_gradients_match_serial_reference() {
        let group_id = [0, 0, 1, 1, 1, 2, 2];
        let predictions = [0.2, -0.3, 1.0, 0.5, -0.8, 0.1, 0.4];
        let targets = [1.0, 0.0, 3.0, 1.0, 0.0, 2.0, 2.0];
        let objective = XeNDCGObjective::new(&group_id);
        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();
        let expected =
            serial_xendcg_reference(&compute_group_boundaries(&group_id), &predictions, &targets);
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn yetirank_parallel_gradients_match_serial_reference() {
        let group_id = [0, 0, 0, 1, 1, 1];
        let predictions = [0.6, -0.2, 0.1, 0.9, 0.0, -0.5];
        let targets = [2.0, 1.0, 0.0, 3.0, 1.0, 0.0];
        let objective = YetiRankObjective::new(&group_id, 3, 17);
        let actual = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();
        let expected = serial_yetirank_reference(
            &compute_group_boundaries(&group_id),
            &predictions,
            &targets,
            3,
            17,
        );
        assert_gradient_pairs_close(&actual, &expected, 1e-6);
    }

    #[test]
    fn lambdamart_compute_gradients_into_reuses_buffer() {
        let group_id = [0, 0, 0, 1, 1];
        let predictions = [0.4, -0.1, 0.2, 0.0, 0.5];
        let targets = [2.0, 0.0, 1.0, 1.0, 0.0];
        let objective = LambdaMARTObjective::new(&group_id);
        let mut buffer = Vec::with_capacity(16);
        let original_capacity = buffer.capacity();
        objective
            .compute_gradients_into(&predictions, &targets, None, &mut buffer)
            .unwrap();
        assert_eq!(buffer.len(), predictions.len());
        assert_eq!(buffer.capacity(), original_capacity);
        let direct = objective
            .compute_gradients(&predictions, &targets, None)
            .unwrap();
        assert_gradient_pairs_close(&buffer, &direct, 1e-6);
    }

    #[test]
    fn lambdamart_compute_gradients_into_rejects_non_finite_pairs() {
        let group_id = [0, 0];
        let predictions = [f32::NAN, 0.5];
        let targets = [1.0, 0.0];
        let objective = LambdaMARTObjective::new(&group_id);
        let sentinel = GradientPair::new(1.0, 1.0).unwrap();
        let mut buffer = vec![sentinel];

        let err = objective
            .compute_gradients_into(&predictions, &targets, None, &mut buffer)
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("gradient and hessian must be finite"),
            "unexpected error: {err}"
        );
        assert_eq!(buffer, vec![sentinel]);
    }
}
