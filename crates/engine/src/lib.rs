use alloygbm_categorical::{TargetEncoderConfig, fit_transform_target_encoder};
use alloygbm_core::{
    BinnedMatrix, CategoricalStatePayloadV1, CoreError, DatasetMatrix, Device, FeatureTile,
    GradientPair, HistogramBundle, MISSING_BIN_U8, MODEL_FORMAT_V1, ModelArtifactSection,
    ModelMetadata, ModelSectionKind, NativeCategoricalSplitsPayload, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate, TrainParams, TrainingDataset, TreeGrowth,
    decode_optional_categorical_state_section_v1,
    decode_optional_native_categorical_splits_section, deserialize_model_artifact_v1,
    encode_categorical_state_payload_v1, encode_native_categorical_splits_payload,
    format_required_section_auto_mode_error, format_required_section_mode_error,
    required_section_compatibility_report, serialize_model_artifact_v1, validate_binned_matrix,
    validate_categorical_state_payload_v1, validate_train_params, validate_training_dataset,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::{SystemTime, UNIX_EPOCH};

/// Small epsilon added to leaf value denominators to prevent division by zero.
const LEAF_EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidConfig(String),
    ContractViolation(String),
    BackendUnavailable(String),
    NotImplemented(String),
    Core(CoreError),
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ContractViolation(msg) => write!(f, "contract violation: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for EngineError {}

impl From<CoreError> for EngineError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitSelectionOptions {
    pub l2_lambda: f32,
    pub l1_alpha: f32,
    pub min_child_hessian: f32,
    pub min_leaf_magnitude: f32,
    /// Histogram index for the NaN/missing bin.
    /// For u8 bins: 255. For u16 bins: max_data_bin + 1 (dynamic).
    pub missing_bin_index: usize,
}

/// Metadata about a feature that uses native categorical splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalFeatureInfo {
    /// The feature index in the BinnedMatrix.
    pub feature_index: usize,
    /// Number of categories (valid bin IDs are 0..num_categories).
    pub num_categories: usize,
}

impl Default for SplitSelectionOptions {
    fn default() -> Self {
        Self {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: MISSING_BIN_U8 as usize,
        }
    }
}

pub trait BackendOps {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle>;
    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>>;
    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        _options: SplitSelectionOptions,
        _feature_weights: &[f32],
        _categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        self.best_split(histograms)
    }
    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult>;
    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        let partition = self.apply_split(binned_matrix, node, split)?;
        let left_stats = self.reduce_sums(gradients, &partition.left_row_indices)?;
        let right_stats = self.reduce_sums(gradients, &partition.right_row_indices)?;
        Ok((partition, left_stats, right_stats))
    }
    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats>;
}

/// Callback invoked after each boosting round to evaluate a custom metric.
///
/// When provided alongside `early_stopping_rounds`, the custom metric value
/// drives early stopping *instead of* the built-in objective loss.
pub trait PerRoundMetricCallback {
    /// Evaluate the metric on `predictions` vs `targets`.
    ///
    /// For single-output models, `predictions` contains the raw model outputs
    /// (pre-sigmoid for binary, raw scores for regression/ranking).
    fn evaluate(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32>;

    /// Whether higher metric values are better (`true`) or lower is better (`false`).
    fn higher_is_better(&self) -> bool;

    /// The name of this metric (e.g. `"custom_rmse"`).
    fn metric_name(&self) -> &str;
}

pub trait ObjectiveOps {
    /// Canonical name for this objective (e.g. "squared_error", "binary_crossentropy").
    fn objective_name(&self) -> &str;

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32>;

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>>;

    /// Compute the objective loss for a set of predictions.
    /// This is used for monitoring convergence and early stopping.
    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32>;

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let gradients = self.compute_gradients(predictions, targets, sample_weights)?;
        buffer.clear();
        buffer.extend(gradients);
        Ok(())
    }

    /// Whether this objective requires `group_id` on the training dataset.
    fn requires_group_id(&self) -> bool {
        false
    }

    /// Whether MSE-based leaf refinement is supported for this objective.
    fn supports_leaf_refinement(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquaredErrorObjective;

impl ObjectiveOps for SquaredErrorObjective {
    fn objective_name(&self) -> &str {
        "squared_error"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if targets.is_empty() {
            return Err(EngineError::ContractViolation(
                "targets cannot be empty".to_string(),
            ));
        }

        if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for (target, weight) in targets.iter().zip(weights) {
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(EngineError::ContractViolation(
                        "sample weights must be finite and > 0".to_string(),
                    ));
                }
                weighted_sum += target * weight;
                weight_sum += weight;
            }
            if weight_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weight sum must be greater than 0".to_string(),
                ));
            }
            return Ok(weighted_sum / weight_sum);
        }

        let sum = targets.iter().sum::<f32>();
        Ok(sum / targets.len() as f32)
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
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

        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "sample weights must be finite and > 0".to_string(),
                ));
            }
            let residual = predictions[index] - targets[index];
            gradients.push(GradientPair::new(residual * weight, weight)?);
        }
        Ok(gradients)
    }

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
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
        buffer.clear();
        if buffer.capacity() < predictions.len() {
            buffer.reserve(predictions.len() - buffer.capacity());
        }
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let residual = predictions[index] - targets[index];
            buffer.push(GradientPair {
                grad: residual * weight,
                hess: weight,
            });
        }
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        squared_error_loss(predictions, targets, sample_weights)
    }
}

/// Binary cross-entropy (log loss) objective for binary classification.
/// Targets must be 0.0 or 1.0. Predictions are in log-odds (logit) space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryCrossEntropyObjective;

/// Numerically stable sigmoid: avoids overflow for large positive/negative inputs.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

impl ObjectiveOps for BinaryCrossEntropyObjective {
    fn objective_name(&self) -> &str {
        "binary_crossentropy"
    }

    fn initial_prediction(
        &self,
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        if targets.is_empty() {
            return Err(EngineError::ContractViolation(
                "targets cannot be empty".to_string(),
            ));
        }
        // Compute weighted mean of targets, then convert to log-odds.
        let (positive_weight, total_weight) = if let Some(weights) = sample_weights {
            if weights.len() != targets.len() {
                return Err(EngineError::ContractViolation(format!(
                    "weights length {} does not match targets length {}",
                    weights.len(),
                    targets.len()
                )));
            }
            let mut pos_w = 0.0_f32;
            let mut tot_w = 0.0_f32;
            for (&target, &weight) in targets.iter().zip(weights) {
                if !weight.is_finite() || weight <= 0.0 {
                    return Err(EngineError::ContractViolation(
                        "sample weights must be finite and > 0".to_string(),
                    ));
                }
                pos_w += target * weight;
                tot_w += weight;
            }
            (pos_w, tot_w)
        } else {
            let pos = targets.iter().sum::<f32>();
            (pos, targets.len() as f32)
        };
        if total_weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weight sum must be greater than 0".to_string(),
            ));
        }
        let p = (positive_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
        // log-odds: log(p / (1 - p))
        Ok((p / (1.0 - p)).ln())
    }

    fn compute_gradients(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<Vec<GradientPair>> {
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

        let mut gradients = Vec::with_capacity(predictions.len());
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let p = sigmoid(predictions[index]);
            // grad = (p - y) * w, hess = p * (1 - p) * w
            let grad = (p - targets[index]) * weight;
            let hess = (p * (1.0 - p)).max(1e-7) * weight;
            gradients.push(GradientPair::new(grad, hess)?);
        }
        Ok(gradients)
    }

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
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
        buffer.clear();
        if buffer.capacity() < predictions.len() {
            buffer.reserve(predictions.len() - buffer.capacity());
        }
        for index in 0..predictions.len() {
            let weight = sample_weights.map_or(1.0, |weights| weights[index]);
            let p = sigmoid(predictions[index]);
            let grad = (p - targets[index]) * weight;
            let hess = (p * (1.0 - p)).max(1e-7) * weight;
            buffer.push(GradientPair { grad, hess });
        }
        Ok(())
    }

    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        binary_crossentropy_loss(predictions, targets, sample_weights)
    }
}

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

// ── QueryRMSE Objective ──────────────────────────────────────────────────

/// Resolves group boundaries for a given data length.
///
/// If `data_len` matches the training boundaries' total, return training
/// boundaries.  If a validation set is present and its boundaries match,
/// return those.  Otherwise fall back to a single-group interpretation.
fn resolve_boundaries_for_len(
    train_boundaries: &[usize],
    validation_boundaries: &Option<Vec<usize>>,
    data_len: usize,
) -> Vec<usize> {
    if let Some(last) = train_boundaries.last()
        && *last == data_len
    {
        return train_boundaries.to_vec();
    }
    if let Some(val_b) = validation_boundaries
        && let Some(last) = val_b.last()
        && *last == data_len
    {
        return val_b.clone();
    }
    // Fallback: treat entire slice as a single group.
    vec![0, data_len]
}

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
    pub left_leaf_value: f32,
    pub right_leaf_value: f32,
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
}

struct IterationExecutionContext<'a> {
    controls: IterationControls,
    validation: Option<ValidationDatasetRef<'a>>,
    policy_mode: Option<TrainingPolicyMode>,
    warm_start: Option<WarmStartState>,
    custom_metric_callback: Option<&'a dyn PerRoundMetricCallback>,
    /// Features that use native categorical splits (empty = all continuous).
    categorical_features: Vec<CategoricalFeatureInfo>,
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
            prediction += if split_went_left(&stump.split, feature_value) {
                stump.left_leaf_value
            } else {
                stump.right_leaf_value
            };
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
                        s.split
                            .categorical_bitset
                            .clone()
                            .unwrap_or_default(),
                    )
                })
                .collect();
            let payload = NativeCategoricalSplitsPayload {
                native_categorical_feature_indices: self
                    .native_categorical_feature_indices
                    .clone(),
                stump_bitsets,
            };
            let cat_bytes = encode_native_categorical_splits_payload(&payload)?;
            sections.push((
                ModelSectionKind::NativeCategoricalSplits,
                cat_bytes,
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
    pub fn with_categorical_features(
        mut self,
        features: Vec<CategoricalFeatureInfo>,
    ) -> Self {
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

        let baseline_prediction =
            objective.initial_prediction(&dataset.targets, dataset.sample_weights.as_deref())?;
        if !baseline_prediction.is_finite() {
            return Err(EngineError::ContractViolation(
                "objective returned non-finite initial prediction".to_string(),
            ));
        }

        let predictions = vec![baseline_prediction; dataset.row_count()];
        let gradients = objective.compute_gradients(
            &predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
        )?;
        validate_gradient_pairs(&gradients, dataset.row_count())?;

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
        let split_candidate = backend.best_split_with_options(
            &histograms,
            split_options,
            &self.params.feature_weights,
            &[],
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
        let controls = self.iteration_controls_for_policy(
            dataset,
            binned_matrix,
            request.rounds,
            request.policy_mode,
        )?;
        let summary = self.fit_iterations_with_optional_validation_summary(
            dataset,
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
        let (encoded_dataset, encoded_binned_matrix) =
            apply_single_categorical_target_encoding(dataset, binned_matrix, spec)?;
        let categorical_state = CategoricalStatePayloadV1 {
            format_version: alloygbm_core::CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: spec.config.time_aware,
            categorical_feature_indices: vec![spec.feature_index as u32],
        };
        let model = self.fit_iterations_with_policy_request(
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
        if experiment_force_manual_policy_enabled() {
            return self.default_iteration_controls(rounds);
        }
        match policy_mode {
            TrainingPolicyMode::Manual => self.default_iteration_controls(rounds),
            TrainingPolicyMode::Auto => {
                self.auto_iteration_controls(dataset, binned_matrix, rounds)
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
    ) -> EngineResult<MultiClassIterationRunSummary> {
        let k = objective.num_classes;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
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

        // Initialize K prediction arrays
        let baselines = objective.initial_predictions();
        let n = dataset.row_count();
        let mut class_predictions: Vec<Vec<f32>> = baselines.iter().map(|&b| vec![b; n]).collect();
        let mut class_candidate_predictions: Vec<Vec<f32>> = class_predictions.clone();
        let mut class_stumps: Vec<Vec<TrainedStump>> = vec![Vec::new(); k];
        // Track stump counts per class at each round boundary for truncation
        let mut stumps_per_round_per_class: Vec<Vec<usize>> = Vec::new();

        // Validation predictions
        let mut validation_class_predictions: Option<Vec<Vec<f32>>> = validation.map(|v| {
            baselines
                .iter()
                .map(|&b| vec![b; v.dataset.row_count()])
                .collect()
        });

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
        let mut best_validation_loss = initial_validation_loss;
        let mut best_validation_round = initial_validation_loss.map(|_| 0_usize);
        let mut validation_no_improvement_rounds = 0_usize;
        let mut weak_improvement_streak = 0_usize;
        let mut weak_improvement_rounds_committed = 0_usize;
        let mut current_validation_loss = initial_validation_loss;
        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(n);

        let effective_round_cap = controls.rounds;

        for round_index in 0..effective_round_cap {
            // Shared sampling for all K classes
            let root_row_indices = sampled_row_indices(
                n,
                controls.row_subsample,
                sampling_seed_base,
                round_index as u64,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                feature_count,
                controls.col_subsample,
                sampling_seed_base,
                round_index as u64,
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
            for class_k in 0..k {
                objective.compute_gradients_for_class(
                    &class_predictions,
                    &dataset.targets,
                    dataset.sample_weights.as_deref(),
                    class_k,
                    &mut gradient_buffer,
                )?;

                let (round_stumps, _round_stop) = if self.params.tree_growth == TreeGrowth::Leaf {
                    build_tree_leaf_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        round_index,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                    )?
                } else {
                    build_tree_level_wise(
                        backend,
                        binned_matrix,
                        &gradient_buffer,
                        root_row_indices.clone(),
                        round_index,
                        &feature_tiles,
                        split_options,
                        &self.params,
                        &controls,
                        &mut class_candidate_predictions[class_k],
                        &self.params.feature_weights,
                        &self.categorical_features,
                    )?
                };

                if !round_stumps.is_empty() {
                    any_tree_produced = true;
                }
                class_stumps[class_k].extend(round_stumps);
            }

            if !any_tree_produced {
                // Revert stump counts
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
                // Revert stump counts
                for class_k in 0..k {
                    class_stumps[class_k].truncate(pre_round_counts[class_k]);
                }
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if loss_improvement < controls.min_loss_improvement {
                if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                    for class_k in 0..k {
                        class_stumps[class_k].truncate(pre_round_counts[class_k]);
                    }
                    stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                    break;
                }
                weak_improvement_streak += 1;
                weak_improvement_rounds_committed += 1;
            } else {
                weak_improvement_streak = 0;
            }

            // Validation early stopping
            let mut stop_for_validation_plateau = false;
            if let Some(validation_ref) = validation {
                let val_preds = validation_class_predictions.as_mut().unwrap();
                for class_k in 0..k {
                    let round_stumps = &class_stumps[class_k][pre_round_counts[class_k]..];
                    if !round_stumps.is_empty() {
                        apply_round_stumps_tree_walk(
                            &mut val_preds[class_k],
                            validation_ref.binned_matrix,
                            round_stumps,
                        )?;
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
            stumps_per_round_per_class.push(
                (0..k)
                    .map(|c| class_stumps[c].len() - pre_round_counts[c])
                    .collect(),
            );
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
            // Compute how many stumps to keep per class
            for class_k in 0..k {
                let keep_count: usize = stumps_per_round_per_class
                    .iter()
                    .take(best_round)
                    .map(|r| r[class_k])
                    .sum();
                class_stumps[class_k].truncate(keep_count);
            }
            rounds_completed = best_round;
        }

        let final_loss = current_loss;
        let final_validation_loss = current_validation_loss;

        Ok(MultiClassIterationRunSummary {
            model: MultiClassTrainedModel {
                num_classes: k,
                baseline_predictions: baselines,
                feature_count,
                class_stumps,
                categorical_state: None,
                objective: objective.objective_name().to_string(),
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
        let auto_min_split_gain: f32 = if binned_density < 0.10 {
            0.001
        } else if row_count.saturating_mul(feature_count) >= 65_536 {
            0.0001
        } else {
            0.0
        };
        controls.min_split_gain = auto_min_split_gain.max(self.params.min_split_gain);
        controls.min_loss_improvement = if row_count < 4_096 {
            0.0
        } else {
            (target_variance.max(1e-6) * 1e-5).min(0.01)
        };
        controls.max_consecutive_weak_improvements = if row_count < 4_096 {
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
        execution: IterationExecutionContext<'_>,
    ) -> EngineResult<IterationRunSummary> {
        let controls = execution.controls;
        let validation = execution.validation;
        validate_iteration_controls(controls)?;
        if controls.early_stopping_rounds.is_some() && validation.is_none() {
            return Err(EngineError::InvalidConfig(
                "validation early stopping requires a validation dataset".to_string(),
            ));
        }
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
        let fit_contract = self.validate_fit_contract(dataset, objective)?;
        let sampling_seed_base = sampling_seed_base(self.params.seed, self.params.deterministic);
        let split_options = split_selection_options_for_training(
            &self.params,
            execution.policy_mode,
            dataset,
            binned_matrix,
        )?;

        // Warm-start: use existing model's baseline + apply existing trees
        let (baseline_prediction, initial_stumps, round_index_offset) =
            if let Some(warm_start) = execution.warm_start {
                (
                    warm_start.baseline_prediction,
                    warm_start.stumps,
                    warm_start.initial_rounds_completed,
                )
            } else {
                (fit_contract.baseline_prediction, Vec::new(), 0)
            };
        let mut predictions = vec![baseline_prediction; dataset.row_count()];
        if !initial_stumps.is_empty() {
            apply_tree_to_binned_predictions(&mut predictions, binned_matrix, &initial_stumps)?;
        }
        let mut candidate_predictions = predictions.clone();
        let mut validation_predictions = if let Some(validation_ref) = validation {
            let mut vp = vec![baseline_prediction; validation_ref.dataset.row_count()];
            if !initial_stumps.is_empty() {
                apply_tree_to_binned_predictions(
                    &mut vp,
                    validation_ref.binned_matrix,
                    &initial_stumps,
                )?;
            }
            Some(vp)
        } else {
            None
        };
        let mut stumps = initial_stumps;
        let initial_stump_count = stumps.len();
        let mut stumps_per_completed_round = Vec::new();
        let mut rounds_completed = 0_usize;
        let effective_round_cap = controls.rounds;
        let mut stop_reason = IterationStopReason::CompletedRequestedRounds;
        let initial_loss = objective.loss(
            &predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
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

        let mut gradient_buffer: Vec<GradientPair> = Vec::with_capacity(dataset.row_count());

        for round_index in 0..effective_round_cap {
            // Offset round_index for sampling seeds and tree IDs when warm-starting
            let effective_round_index = round_index + round_index_offset;
            let root_row_indices = sampled_row_indices(
                dataset.row_count(),
                controls.row_subsample,
                sampling_seed_base,
                effective_round_index as u64,
            );
            let (feature_tiles, sampled_feature_count) = sampled_feature_tiles(
                binned_matrix.feature_count,
                controls.col_subsample,
                sampling_seed_base,
                effective_round_index as u64,
            )?;
            let sampled_row_count = root_row_indices.len();
            objective.compute_gradients_into(
                &predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
                &mut gradient_buffer,
            )?;
            let gradients = &gradient_buffer;
            validate_gradient_pair_length(gradients, dataset.row_count())?;
            if cfg!(debug_assertions) {
                validate_gradient_pairs(gradients, dataset.row_count())?;
            }

            candidate_predictions.copy_from_slice(&predictions);

            let (candidate_round_stumps, round_rejection_reason) =
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
                    )?
                };

            if candidate_round_stumps.is_empty() {
                stop_reason = round_rejection_reason;
                break;
            }

            let candidate_loss = objective.loss(
                &candidate_predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            if loss_improvement < 0.0 {
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            if loss_improvement < controls.min_loss_improvement {
                if weak_improvement_streak >= controls.max_consecutive_weak_improvements {
                    stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                    break;
                }
                weak_improvement_streak += 1;
                weak_improvement_rounds_committed += 1;
            } else {
                weak_improvement_streak = 0;
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
                apply_round_stumps_tree_walk(
                    &mut next_validation_predictions,
                    validation_ref.binned_matrix,
                    &candidate_round_stumps,
                )?;
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
            if let Some(next_validation_predictions) = candidate_validation_predictions {
                validation_predictions = Some(next_validation_predictions);
            }
            if let Some(next_validation_loss) = candidate_validation_loss {
                current_validation_loss = Some(next_validation_loss);
                validation_loss_per_completed_round.push(next_validation_loss);
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

        if experiment_leaf_refinement_enabled() && objective.supports_leaf_refinement() {
            refine_regression_leaf_values(
                baseline_prediction,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
                binned_matrix,
                &mut stumps,
                &stumps_per_completed_round,
                controls.max_abs_leaf_value,
            )?;

            let mut refined_predictions = vec![baseline_prediction; dataset.row_count()];
            apply_tree_to_binned_predictions(&mut refined_predictions, binned_matrix, &stumps)?;
            current_loss = objective.loss(
                &refined_predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
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

        let model = TrainedModel {
            baseline_prediction,
            feature_count: dataset.matrix.feature_count,
            stumps,
            categorical_state: None,
            node_debug_stats: None,
            objective: objective.objective_name().to_string(),
            native_categorical_feature_indices: Vec::new(),
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

const SPLIT_L2_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_SPLIT_L2";
const SPLIT_L1_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_SPLIT_L1";
const MIN_CHILD_HESS_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS";
const SPLIT_MIN_LEAF_MAGNITUDE_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_SPLIT_MIN_LEAF_MAGNITUDE";
const FORCE_MANUAL_POLICY_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_FORCE_MANUAL_POLICY";
const ENABLE_LEAF_REFINEMENT_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_ENABLE_LEAF_REFINEMENT";
const AUTO_SPLIT_L2_NOISY_SMALL_WIDE: f32 = 2.0;

fn split_selection_options_from_env() -> EngineResult<SplitSelectionOptions> {
    Ok(SplitSelectionOptions {
        l2_lambda: parse_nonnegative_env_f32(SPLIT_L2_ENV_VAR)?,
        l1_alpha: parse_nonnegative_env_f32(SPLIT_L1_ENV_VAR)?,
        min_child_hessian: parse_nonnegative_env_f32(MIN_CHILD_HESS_ENV_VAR)?,
        min_leaf_magnitude: parse_nonnegative_env_f32(SPLIT_MIN_LEAF_MAGNITUDE_ENV_VAR)?,
        missing_bin_index: MISSING_BIN_U8 as usize,
    })
}

fn experiment_force_manual_policy_enabled() -> bool {
    env_toggle_enabled(FORCE_MANUAL_POLICY_ENV_VAR)
}

fn experiment_leaf_refinement_enabled() -> bool {
    env_toggle_enabled(ENABLE_LEAF_REFINEMENT_ENV_VAR)
}

fn env_toggle_enabled(env_name: &str) -> bool {
    match std::env::var(env_name) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn parse_nonnegative_env_f32(env_name: &str) -> EngineResult<f32> {
    match std::env::var(env_name) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Ok(0.0);
            }
            let parsed = trimmed.parse::<f32>().map_err(|_| {
                EngineError::InvalidConfig(format!(
                    "{env_name} must be a finite, non-negative f32 value"
                ))
            })?;
            if !parsed.is_finite() || parsed < 0.0 {
                return Err(EngineError::InvalidConfig(format!(
                    "{env_name} must be finite and >= 0"
                )));
            }
            Ok(parsed)
        }
        Err(_) => Ok(0.0),
    }
}

fn l1_threshold_gradient(grad_sum: f32, l1_alpha: f32) -> f32 {
    if l1_alpha <= 0.0 {
        return grad_sum;
    }
    if grad_sum > l1_alpha {
        grad_sum - l1_alpha
    } else if grad_sum < -l1_alpha {
        grad_sum + l1_alpha
    } else {
        0.0
    }
}

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

fn split_l2_env_is_configured() -> bool {
    std::env::var_os(SPLIT_L2_ENV_VAR).is_some()
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
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let mut candidate_round_stumps = Vec::new();
    let mut round_rejection_reason = IterationStopReason::NoSplitCandidate;
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    // Maintain each active node's absolute leaf output so child updates
    // can replace parent contribution via deltas (tree semantics).
    let mut active_nodes = vec![(0_u32, root_node.row_indices, root_histograms, 0.0_f32)];

    for depth in 0..(params.max_depth as usize) {
        if active_nodes.is_empty() {
            break;
        }

        let mut next_nodes = Vec::new();
        for (local_node_id, node_rows, histograms, parent_leaf_value) in active_nodes {
            let node_id = encode_tree_node_id(round_index, local_node_id)?;
            let node = NodeSlice::new(node_id, node_rows)?;
            let Some(mut split) =
                backend.best_split_with_options(&histograms, split_options, feature_weights, categorical_features)?
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

            let left_grad = l1_threshold_gradient(left_stats.grad_sum, split_options.l1_alpha);
            let right_grad = l1_threshold_gradient(right_stats.grad_sum, split_options.l1_alpha);
            let raw_left_leaf_value = -params.learning_rate * left_grad
                / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
            let raw_right_leaf_value = -params.learning_rate * right_grad
                / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

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

            apply_partition_leaf_updates(
                candidate_predictions,
                &partition,
                left_leaf_value,
                right_leaf_value,
            )?;

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
                        left_leaf_absolute,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_row_indices,
                        right_histograms,
                        right_leaf_absolute,
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
                        left_leaf_absolute,
                    ));
                    next_nodes.push((
                        right_local_node_id,
                        right_node.row_indices,
                        right_histograms,
                        right_leaf_absolute,
                    ));
                }
            }

            candidate_round_stumps.push(TrainedStump {
                split,
                left_leaf_value,
                right_leaf_value,
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
) -> EngineResult<(Vec<TrainedStump>, IterationStopReason)> {
    let max_leaves = controls.max_leaves.unwrap_or(usize::MAX);
    let max_depth = params.max_depth as usize;

    // Build root histograms and find best split.
    let root_node_id = encode_tree_node_id(round_index, 0)?;
    let root_node = NodeSlice::new(root_node_id, root_row_indices)?;
    let root_histograms =
        backend.build_histograms(binned_matrix, gradients, &root_node, feature_tiles)?;
    let root_split =
        backend.best_split_with_options(&root_histograms, split_options, feature_weights, categorical_features)?;

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
        let left_grad = l1_threshold_gradient(left_stats.grad_sum, split_options.l1_alpha);
        let right_grad = l1_threshold_gradient(right_stats.grad_sum, split_options.l1_alpha);
        let raw_left_leaf_value = -params.learning_rate * left_grad
            / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
        let raw_right_leaf_value = -params.learning_rate * right_grad
            / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

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

        // Commit the split: update predictions and record stump.
        apply_partition_leaf_updates(
            candidate_predictions,
            &partition,
            left_leaf_value,
            right_leaf_value,
        )?;

        let mut committed_split = split;
        committed_split.left_stats = left_stats;
        committed_split.right_stats = right_stats;

        stumps.push(TrainedStump {
            split: committed_split,
            left_leaf_value,
            right_leaf_value,
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
            let (
                smaller_indices,
                larger_indices,
                smaller_node_id,
                larger_node_id,
                smaller_local,
                larger_local,
                smaller_leaf_abs,
                larger_leaf_abs,
            ) = if partition.left_row_indices.len() <= partition.right_row_indices.len() {
                (
                    partition.left_row_indices,
                    partition.right_row_indices,
                    left_node_id,
                    right_node_id,
                    left_local,
                    right_local,
                    left_leaf_absolute,
                    right_leaf_absolute,
                )
            } else {
                (
                    partition.right_row_indices,
                    partition.left_row_indices,
                    right_node_id,
                    left_node_id,
                    right_local,
                    left_local,
                    right_leaf_absolute,
                    left_leaf_absolute,
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

            // Find best split for each child and enqueue if valid.
            if let Some(child_split) = backend.best_split_with_options(
                &smaller_histograms,
                split_options,
                feature_weights,
                categorical_features,
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: smaller_local,
                    row_indices: smaller_node.row_indices,
                    split_candidate: child_split,
                    histograms: smaller_histograms,
                    parent_leaf_value: smaller_leaf_abs,
                    depth: child_depth,
                });
            }

            if let Some(child_split) = backend.best_split_with_options(
                &larger_histograms,
                split_options,
                feature_weights,
                categorical_features,
            )? && child_split.gain.is_finite()
                && child_split.gain > controls.min_split_gain
            {
                queue.push(PendingSplit {
                    local_node_id: larger_local,
                    row_indices: larger_indices,
                    split_candidate: child_split,
                    histograms: larger_histograms,
                    parent_leaf_value: larger_leaf_abs,
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

const TREE_NODE_STRIDE: u32 = 1 << 20;

fn encode_tree_node_id(tree_index: usize, local_node_id: u32) -> EngineResult<u32> {
    if local_node_id >= TREE_NODE_STRIDE {
        return Err(EngineError::ContractViolation(format!(
            "local node_id {local_node_id} exceeds supported tree-node stride {TREE_NODE_STRIDE}"
        )));
    }
    let tree_index_u32 = u32::try_from(tree_index).map_err(|_| {
        EngineError::ContractViolation(format!("tree index {tree_index} exceeds u32::MAX"))
    })?;
    tree_index_u32
        .checked_mul(TREE_NODE_STRIDE)
        .and_then(|base| base.checked_add(local_node_id))
        .ok_or_else(|| {
            EngineError::ContractViolation("encoded tree node id overflowed u32 range".to_string())
        })
}

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

fn left_child_node_id(local_node_id: u32) -> EngineResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "left child id overflow for local node {local_node_id}"
            ))
        })
}

fn right_child_node_id(local_node_id: u32) -> EngineResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| {
            EngineError::ContractViolation(format!(
                "right child id overflow for local node {local_node_id}"
            ))
        })
}

fn retained_stump_count_for_rounds(
    stumps_per_completed_round: &[usize],
    round_count: usize,
) -> usize {
    stumps_per_completed_round
        .iter()
        .take(round_count)
        .sum::<usize>()
}

/// Determine if a feature value goes left at a split, handling continuous, categorical, and NaN.
#[inline]
fn split_went_left(split: &SplitCandidate, feature_value: f32) -> bool {
    if feature_value.is_nan() {
        split.default_left
    } else if split.is_categorical {
        split
            .categorical_bitset
            .as_ref()
            .map_or(split.default_left, |bs| {
                let cat_id = feature_value as u16;
                let byte_idx = (cat_id / 8) as usize;
                let bit_idx = (cat_id % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            })
    } else {
        feature_value <= split.threshold_bin as f32
    }
}

fn row_satisfies_stump_path_features(
    features: &[f32],
    stump: &TrainedStump,
    stumps_by_node: &HashMap<u32, &TrainedStump>,
) -> EngineResult<bool> {
    let (tree_id, mut local_node_id) = decode_tree_node_id(stump.split.node_id);
    while local_node_id > 0 {
        let parent_local = (local_node_id - 1) / 2;
        let parent_node_id = encode_tree_node_id(tree_id as usize, parent_local)?;
        let Some(parent_stump) = stumps_by_node.get(&parent_node_id) else {
            return Ok(false);
        };
        let feature_index = parent_stump.split.feature_index as usize;
        if feature_index >= features.len() {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature length {}",
                parent_stump.split.feature_index,
                features.len()
            )));
        }
        let feature_value = features[feature_index];
        let went_left = split_went_left(&parent_stump.split, feature_value);
        let expected_left = local_node_id == parent_local * 2 + 1;
        if went_left != expected_left {
            return Ok(false);
        }
        local_node_id = parent_local;
    }
    Ok(true)
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

fn mixed_hash(mut value: u64) -> u64 {
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

/// Maximum features per tile. Keeps the histogram arena small enough to fit in
/// L2 cache (64 features × 256 bins × 12 bytes ≈ 192 KB) and creates enough
/// tiles for rayon to parallelize across cores.
const MAX_TILE_FEATURE_WIDTH: usize = 64;

fn feature_tiles_from_sorted_indices(indices: &[usize]) -> EngineResult<Vec<FeatureTile>> {
    if indices.is_empty() {
        return Err(EngineError::ContractViolation(
            "feature subsampling produced no feature indices".to_string(),
        ));
    }

    let mut tiles = Vec::new();
    let mut run_start = indices[0];
    let mut previous = indices[0];
    for &current in indices.iter().skip(1) {
        if current == previous + 1 && (current - run_start) < MAX_TILE_FEATURE_WIDTH {
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

fn apply_round_stumps_tree_walk(
    predictions: &mut [f32],
    binned_matrix: &BinnedMatrix,
    stumps: &[TrainedStump],
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
            if bin <= stump.split.threshold_bin {
                *prediction += stump.left_leaf_value;
                local_id = local_id * 2 + 1; // left child
            } else {
                *prediction += stump.right_leaf_value;
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
            apply_round_stumps_tree_walk(predictions, binned_matrix, &stumps[round_start..i])?;
            round_start = i;
            current_tree_id = tree_id;
        }
    }
    apply_round_stumps_tree_walk(predictions, binned_matrix, &stumps[round_start..])?;
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
    apply_tree_to_binned_predictions(&mut predictions, binned_matrix, stumps)?;
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
            .unwrap_or(parent_absolute + stump.left_leaf_value);
        let right_absolute = refined_absolute_outputs
            .get(&right_local_node_id)
            .copied()
            .unwrap_or(parent_absolute + stump.right_leaf_value);
        stump.left_leaf_value = left_absolute - parent_absolute;
        stump.right_leaf_value = right_absolute - parent_absolute;
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
    absolute_outputs.insert(left_local_node_id, parent_absolute + stump.left_leaf_value);
    absolute_outputs.insert(
        right_local_node_id,
        parent_absolute + stump.right_leaf_value,
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

fn squared_error_loss(
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

fn binary_crossentropy_loss(
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

fn required_single_section(
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
                row_count: read_u32_le(bytes, base + 24)?,
            },
            right_stats: NodeStats {
                grad_sum: read_f32_le(bytes, base + 28)?,
                hess_sum: read_f32_le(bytes, base + 32)?,
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
        bytes.extend_from_slice(&stump.left_leaf_value.to_le_bytes());
        bytes.extend_from_slice(&stump.right_leaf_value.to_le_bytes());
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
                    row_count: left_count,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: right_count as f32,
                    row_count: right_count,
                },
            },
            left_leaf_value,
            right_leaf_value,
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
    })
}

fn read_u32_le(bytes: &[u8], start: usize) -> EngineResult<u32> {
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

fn read_u16_le(bytes: &[u8], start: usize) -> EngineResult<u16> {
    let end = start + 2;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
            "unexpected end of payload when reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

fn read_f32_le(bytes: &[u8], start: usize) -> EngineResult<f32> {
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

// ---------------------------------------------------------------------------
// Multi-class softmax classification
// ---------------------------------------------------------------------------

/// Softmax cross-entropy objective for K-class classification.
///
/// Does NOT implement [`ObjectiveOps`] because that trait is fundamentally
/// single-output. Multi-class training requires K prediction arrays and K
/// gradient arrays computed jointly (softmax couples all classes).
pub struct MultiClassSoftmaxObjective {
    pub num_classes: usize,
}

impl MultiClassSoftmaxObjective {
    pub fn new(num_classes: usize) -> EngineResult<Self> {
        if num_classes < 2 {
            return Err(EngineError::InvalidConfig(format!(
                "multiclass_softmax requires at least 2 classes, got {num_classes}"
            )));
        }
        Ok(Self { num_classes })
    }

    pub fn objective_name(&self) -> &str {
        "multiclass_softmax"
    }

    /// Returns K initial predictions (all zeros → uniform 1/K under softmax).
    pub fn initial_predictions(&self) -> Vec<f32> {
        vec![0.0; self.num_classes]
    }

    /// Compute gradients for a single class given all K prediction arrays.
    ///
    /// `class_predictions[k][i]` is the raw logit for class k, sample i.
    pub fn compute_gradients_for_class(
        &self,
        class_predictions: &[Vec<f32>],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        class_k: usize,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let k = self.num_classes;
        if class_predictions.len() != k {
            return Err(EngineError::ContractViolation(format!(
                "expected {} class prediction arrays, got {}",
                k,
                class_predictions.len()
            )));
        }
        let n = class_predictions[0].len();
        if targets.len() != n {
            return Err(EngineError::ContractViolation(format!(
                "targets length {} does not match predictions length {n}",
                targets.len()
            )));
        }
        if let Some(w) = sample_weights
            && w.len() != n
        {
            return Err(EngineError::ContractViolation(format!(
                "sample_weights length {} does not match predictions length {n}",
                w.len()
            )));
        }

        buffer.clear();
        buffer.reserve(n.saturating_sub(buffer.capacity()));

        for i in 0..n {
            // Numerically stable softmax: subtract max
            let mut max_logit = f32::NEG_INFINITY;
            for class_preds in class_predictions.iter().take(k) {
                let v = class_preds[i];
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            for class_preds in class_predictions.iter().take(k) {
                sum_exp += (class_preds[i] - max_logit).exp();
            }
            let p_k = (class_predictions[class_k][i] - max_logit).exp() / sum_exp;

            let indicator = if (targets[i] as usize) == class_k {
                1.0
            } else {
                0.0
            };
            let weight = sample_weights.map_or(1.0, |w| w[i]);
            let grad = (p_k - indicator) * weight;
            let hess = (p_k * (1.0 - p_k) * weight).max(1e-7);

            buffer.push(GradientPair { grad, hess });
        }

        Ok(())
    }

    /// Multi-class cross-entropy loss.
    pub fn loss(
        &self,
        class_predictions: &[Vec<f32>],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32> {
        let k = self.num_classes;
        if class_predictions.len() != k {
            return Err(EngineError::ContractViolation(format!(
                "expected {} class prediction arrays, got {}",
                k,
                class_predictions.len()
            )));
        }
        let n = class_predictions[0].len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut total_loss = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for i in 0..n {
            let target_class = targets[i] as usize;
            let weight = sample_weights.map_or(1.0_f64, |w| w[i] as f64);

            // log-sum-exp trick for numerical stability
            let mut max_logit = f32::NEG_INFINITY;
            for class_preds in class_predictions.iter().take(k) {
                let v = class_preds[i];
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum_exp = 0.0_f64;
            for class_preds in class_predictions.iter().take(k) {
                sum_exp += ((class_preds[i] - max_logit) as f64).exp();
            }
            let log_p = (class_predictions[target_class][i] - max_logit) as f64 - sum_exp.ln();

            total_loss -= log_p * weight;
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return Ok(0.0);
        }
        Ok((total_loss / total_weight) as f32)
    }
}

/// Trained multi-class model: K tree sequences (one per class).
#[derive(Debug, Clone, PartialEq)]
pub struct MultiClassTrainedModel {
    pub num_classes: usize,
    pub baseline_predictions: Vec<f32>,
    pub feature_count: usize,
    pub class_stumps: Vec<Vec<TrainedStump>>,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    pub objective: String,
}

impl MultiClassTrainedModel {
    pub fn rounds_completed(&self) -> usize {
        if self.class_stumps.is_empty() || self.class_stumps[0].is_empty() {
            return 0;
        }
        // Count unique tree IDs in class 0's stumps
        let mut max_tree_id = 0_u32;
        for stump in &self.class_stumps[0] {
            let tree_id = stump.split.node_id / TREE_NODE_STRIDE;
            if tree_id > max_tree_id {
                max_tree_id = tree_id;
            }
        }
        max_tree_id as usize + 1
    }

    pub fn with_categorical_state(
        mut self,
        state: Option<CategoricalStatePayloadV1>,
    ) -> EngineResult<Self> {
        if let Some(ref state) = state {
            validate_categorical_state_payload_v1(state, Some(self.feature_count))?;
        }
        self.categorical_state = state;
        Ok(self)
    }

    pub fn to_artifact_bytes(&self) -> EngineResult<Vec<u8>> {
        let feature_count_u32 = u32::try_from(self.feature_count).map_err(|_| {
            EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
        })?;
        let num_classes_u32 = u32::try_from(self.num_classes).map_err(|_| {
            EngineError::ContractViolation("num_classes exceeds u32::MAX".to_string())
        })?;

        // Build MultiClassTrees section payload
        let mut mc_payload = Vec::new();
        mc_payload.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
        mc_payload.extend_from_slice(&num_classes_u32.to_le_bytes());
        mc_payload.extend_from_slice(&feature_count_u32.to_le_bytes());

        for baseline in &self.baseline_predictions {
            mc_payload.extend_from_slice(&baseline.to_le_bytes());
        }

        for class_stumps in &self.class_stumps {
            let count = u32::try_from(class_stumps.len()).map_err(|_| {
                EngineError::ContractViolation("stump count exceeds u32::MAX".to_string())
            })?;
            mc_payload.extend_from_slice(&count.to_le_bytes());
        }

        for class_stumps in &self.class_stumps {
            for stump in class_stumps {
                mc_payload.extend_from_slice(&stump.split.node_id.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.feature_index.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.threshold_bin.to_le_bytes());
                let mut flags: u16 = if stump.split.default_left { 1 } else { 0 };
                if stump.split.is_categorical {
                    flags |= 2;
                }
                mc_payload.extend_from_slice(&flags.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.gain.to_le_bytes());
                mc_payload.extend_from_slice(&stump.left_leaf_value.to_le_bytes());
                mc_payload.extend_from_slice(&stump.right_leaf_value.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.left_stats.row_count.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.right_stats.row_count.to_le_bytes());
            }
        }

        // Build PredictorLayout payload
        let mut layout_payload = Vec::new();
        const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
        layout_payload.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
        layout_payload.extend_from_slice(&feature_count_u32.to_le_bytes());
        layout_payload.extend_from_slice(&THRESHOLD_MODE_BIN_INDEX.to_le_bytes());

        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..self.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: self.objective.clone(),
            num_classes: Some(num_classes_u32),
        };

        let mut sections = vec![
            (ModelSectionKind::MultiClassTrees, mc_payload),
            (ModelSectionKind::PredictorLayout, layout_payload),
        ];
        if let Some(categorical_state) = self.categorical_state.as_ref() {
            let categorical_payload = encode_categorical_state_payload_v1(categorical_state)?;
            sections.push((ModelSectionKind::CategoricalState, categorical_payload));
        }

        serialize_model_artifact_v1(&metadata, &sections).map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;

        let mc_section =
            required_single_section(&parsed.sections, ModelSectionKind::MultiClassTrees)?;

        let payload = &mc_section.payload;
        const MC_HEADER_SIZE: usize = 12; // format_version + num_classes + feature_count
        if payload.len() < MC_HEADER_SIZE {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small".to_string(),
            ));
        }

        let format_version = read_u32_le(payload, 0)?;
        if format_version != MODEL_FORMAT_V1 {
            return Err(EngineError::ContractViolation(format!(
                "unsupported multiclass trees format version {format_version}"
            )));
        }
        let num_classes = read_u32_le(payload, 4)? as usize;
        let feature_count = read_u32_le(payload, 8)? as usize;

        let baselines_start = MC_HEADER_SIZE;
        let baselines_end = baselines_start + num_classes * 4;
        if payload.len() < baselines_end {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small for baselines".to_string(),
            ));
        }
        let mut baseline_predictions = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            baseline_predictions.push(read_f32_le(payload, baselines_start + k * 4)?);
        }

        let counts_start = baselines_end;
        let counts_end = counts_start + num_classes * 4;
        if payload.len() < counts_end {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small for stump counts".to_string(),
            ));
        }
        let mut stump_counts = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            stump_counts.push(read_u32_le(payload, counts_start + k * 4)? as usize);
        }

        const STUMP_SIZE: usize = 32;
        let total_stumps: usize = stump_counts.iter().sum();
        let stumps_start = counts_end;
        let expected_len = stumps_start + total_stumps * STUMP_SIZE;
        if payload.len() != expected_len {
            return Err(EngineError::ContractViolation(format!(
                "multiclass trees payload length {} does not match expected {expected_len}",
                payload.len()
            )));
        }

        let mut class_stumps = Vec::with_capacity(num_classes);
        let mut offset = stumps_start;
        for &count in stump_counts.iter().take(num_classes) {
            let mut stumps = Vec::with_capacity(count);
            for _ in 0..count {
                let node_id = read_u32_le(payload, offset)?;
                let feature_index = read_u32_le(payload, offset + 4)?;
                let threshold_bin = read_u16_le(payload, offset + 8)?;
                let flags = read_u16_le(payload, offset + 10)?;
                let default_left = (flags & 1) != 0;
                let is_categorical = (flags & 2) != 0;
                let gain = read_f32_le(payload, offset + 12)?;
                let left_leaf_value = read_f32_le(payload, offset + 16)?;
                let right_leaf_value = read_f32_le(payload, offset + 20)?;
                let left_count = read_u32_le(payload, offset + 24)?;
                let right_count = read_u32_le(payload, offset + 28)?;

                stumps.push(TrainedStump {
                    split: SplitCandidate {
                        node_id,
                        feature_index,
                        threshold_bin,
                        gain,
                        default_left,
                        is_categorical,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: left_count as f32,
                            row_count: left_count,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: right_count as f32,
                            row_count: right_count,
                        },
                    },
                    left_leaf_value,
                    right_leaf_value,
                });
                offset += STUMP_SIZE;
            }
            class_stumps.push(stumps);
        }

        let categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, feature_count)?;

        Ok(Self {
            num_classes,
            baseline_predictions,
            feature_count,
            class_stumps,
            categorical_state,
            objective: parsed.contract.metadata.objective.clone(),
        })
    }
}

/// Summary from a multi-class training run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiClassIterationRunSummary {
    pub model: MultiClassTrainedModel,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;
    struct BadObjective;

    fn sample_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 1.0, //
                    3.0, 1.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![2.0, 1.0, -1.0, -2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        }
    }

    fn sample_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            4,
            2,
            3,
            vec![
                0, 0, //
                1, 0, //
                2, 1, //
                3, 1, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn sample_wide_small_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                8,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, //
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![2.0, 1.0, -1.0, -2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        }
    }

    fn sample_wide_small_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            4,
            8,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 1, 1, 1, 1, //
                2, 2, 2, 2, 2, 2, 2, 2, //
                3, 3, 3, 3, 3, 3, 3, 3, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn sample_noisy_wide_small_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                8,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, //
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![10.0, 5.0, -5.0, -10.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        }
    }

    fn sample_trained_model() -> TrainedModel {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("iterative training succeeds")
    }

    impl BackendOps for MockBackend {
        fn build_histograms(
            &self,
            _binned_matrix: &BinnedMatrix,
            _gradients: &[GradientPair],
            node: &NodeSlice,
            _feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            Ok(HistogramBundle {
                node_id: node.node_id,
                feature_histograms: Vec::new(),
            })
        }

        fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
            let (_, local_node_id) = decode_tree_node_id(histograms.node_id);
            let threshold_bin = match local_node_id {
                0 => 1,
                1 => 0,
                2 => 2,
                _ => 1,
            };
            Ok(Some(SplitCandidate {
                node_id: histograms.node_id,
                feature_index: 0,
                threshold_bin,
                gain: 3.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    row_count: 0,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    row_count: 0,
                },
            }))
        }

        fn apply_split(
            &self,
            binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            node.validate_bounds(binned_matrix.row_count)?;

            let mut left_row_indices = Vec::new();
            let mut right_row_indices = Vec::new();
            for &row_index in &node.row_indices {
                let row_index = row_index as usize;
                let cell_index =
                    row_index * binned_matrix.feature_count + split.feature_index as usize;
                let bin = binned_matrix.row_bin(cell_index);
                if bin <= split.threshold_bin {
                    left_row_indices.push(row_index as u32);
                } else {
                    right_row_indices.push(row_index as u32);
                }
            }
            Ok(PartitionResult {
                left_row_indices,
                right_row_indices,
            })
        }

        fn reduce_sums(
            &self,
            gradients: &[GradientPair],
            row_indices: &[u32],
        ) -> EngineResult<NodeStats> {
            let mut grad_sum = 0.0_f32;
            let mut hess_sum = 0.0_f32;
            for &row_index in row_indices {
                let gp = gradients.get(row_index as usize).ok_or_else(|| {
                    EngineError::ContractViolation(
                        "row index out of bounds in mock reduction".to_string(),
                    )
                })?;
                grad_sum += gp.grad;
                hess_sum += gp.hess;
            }
            Ok(NodeStats {
                grad_sum,
                hess_sum,
                row_count: row_indices.len() as u32,
            })
        }
    }

    impl ObjectiveOps for BadObjective {
        fn objective_name(&self) -> &str {
            "bad"
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
            _predictions: &[f32],
            _targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<Vec<GradientPair>> {
            Ok(vec![GradientPair {
                grad: 0.1,
                hess: 1.0,
            }])
        }

        fn loss(
            &self,
            predictions: &[f32],
            targets: &[f32],
            sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            squared_error_loss(predictions, targets, sample_weights)
        }
    }

    #[test]
    fn squared_error_objective_produces_expected_baseline() {
        let objective = SquaredErrorObjective;
        let baseline = objective
            .initial_prediction(&[2.0, 0.0, -2.0], None)
            .expect("baseline should compute");
        assert!(baseline.abs() < 1e-6);
    }

    #[test]
    fn trainer_validates_fit_contract() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer
            .validate_fit_contract(&sample_dataset(), &SquaredErrorObjective)
            .expect("contract validation succeeds");
        assert_eq!(result.gradients.len(), 4);
    }

    #[test]
    fn trainer_rejects_gradient_length_mismatch() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer.validate_fit_contract(&sample_dataset(), &BadObjective);
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn fit_one_round_returns_coherent_summary() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let summary = trainer
            .fit_one_round(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
            )
            .expect("fit one round should succeed");

        assert_eq!(summary.root_stats.row_count, 4);
        assert!(summary.split_candidate.is_some());
        assert!(summary.partition.is_some());
    }

    #[test]
    fn fit_one_round_rejects_row_mismatch() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let bad_binned = BinnedMatrix::new(3, 2, 3, vec![0, 0, 1, 0, 2, 1]).expect("valid matrix");
        let result = trainer.fit_one_round(
            &sample_dataset(),
            &bad_binned,
            &MockBackend,
            &SquaredErrorObjective,
        );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn fit_iterations_builds_model_and_changes_predictions() {
        let params = TrainParams {
            learning_rate: 0.5,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let model = trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                3,
            )
            .expect("iterative training succeeds");

        assert!(!model.stumps.is_empty());
        let left_pred = model.predict_row(&[0.0, 0.0]).expect("prediction succeeds");
        let right_pred = model.predict_row(&[3.0, 1.0]).expect("prediction succeeds");
        assert!(left_pred > right_pred);
    }

    #[test]
    fn refine_regression_leaf_values_reduces_loss_for_fixed_structure() {
        let node_id = encode_tree_node_id(0, 0).expect("node id encodes");
        let mut stumps = vec![TrainedStump {
            split: SplitCandidate {
                node_id,
                feature_index: 0,
                threshold_bin: 1,
                gain: 1.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    row_count: 2,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    row_count: 2,
                },
            },
            left_leaf_value: 0.0,
            right_leaf_value: 0.0,
        }];
        let matrix = sample_binned_matrix();
        let targets = sample_dataset().targets;

        let before = tree_predictions_for_binned_rows(&matrix, &stumps)
            .expect("tree predictions should compute");
        let before_loss = squared_error_loss(&before, &targets, None).expect("loss should compute");

        refine_regression_leaf_values(0.0, &targets, None, &matrix, &mut stumps, &[1], 1_000_000.0)
            .expect("refinement should succeed");

        let after = tree_predictions_for_binned_rows(&matrix, &stumps)
            .expect("tree predictions should compute");
        let after_loss = squared_error_loss(&after, &targets, None).expect("loss should compute");

        assert!(after_loss < before_loss);
        assert!(stumps[0].left_leaf_value > 0.0);
        assert!(stumps[0].right_leaf_value < 0.0);
    }

    #[test]
    fn fit_iterations_rejects_zero_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let result = trainer.fit_iterations(
            &sample_dataset(),
            &sample_binned_matrix(),
            &MockBackend,
            &SquaredErrorObjective,
            0,
        );
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
    }

    #[test]
    fn trainer_rejects_invalid_subsample_params() {
        let params = TrainParams {
            row_subsample: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            Trainer::new(params),
            Err(EngineError::Core(CoreError::InvalidConfig(_)))
        ));
    }

    #[test]
    fn auto_policy_preserves_default_controls_on_small_datasets() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let manual = trainer
            .default_iteration_controls(8)
            .expect("manual controls should build");
        let auto = trainer
            .iteration_controls_for_policy(
                &sample_dataset(),
                &sample_binned_matrix(),
                8,
                TrainingPolicyMode::Auto,
            )
            .expect("auto controls should build");

        assert_eq!(auto.min_rows_per_leaf, manual.min_rows_per_leaf);
        assert_eq!(auto.min_split_gain, manual.min_split_gain);
        assert_eq!(auto.min_loss_improvement, manual.min_loss_improvement);
        assert_eq!(
            auto.max_consecutive_weak_improvements,
            manual.max_consecutive_weak_improvements
        );
        assert_eq!(auto.row_subsample, manual.row_subsample);
        assert_eq!(auto.col_subsample, manual.col_subsample);
    }

    #[test]
    fn auto_policy_caps_rounds_for_small_wide_datasets() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let controls = trainer
            .iteration_controls_for_policy(
                &sample_wide_small_dataset(),
                &sample_wide_small_binned_matrix(),
                1_200,
                TrainingPolicyMode::Auto,
            )
            .expect("auto controls should build");

        assert_eq!(controls.rounds, 96);
    }

    #[test]
    fn auto_split_l2_targets_noisy_small_wide_datasets() {
        assert!(
            should_apply_auto_split_l2(
                &sample_noisy_wide_small_dataset(),
                &sample_wide_small_binned_matrix()
            )
            .expect("heuristic should evaluate")
        );
    }

    #[test]
    fn auto_split_l2_skips_dense_numeric_style_datasets() {
        assert!(
            !should_apply_auto_split_l2(&sample_dataset(), &sample_binned_matrix())
                .expect("heuristic should evaluate")
        );
    }

    #[test]
    fn fit_iterations_controls_enforce_min_split_gain() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn fit_iterations_summary_reports_gain_threshold_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.effective_round_cap, 3);
        assert_eq!(summary.rounds_completed, 0);
        assert_eq!(summary.stop_reason, IterationStopReason::GainBelowThreshold);
        assert!(summary.model.stumps.is_empty());
        assert!(summary.loss_per_completed_round.is_empty());
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_reports_completed_requested_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 1);
        assert_eq!(summary.effective_round_cap, 1);
        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CompletedRequestedRounds
        );
        assert!(!summary.model.stumps.is_empty());
        assert_eq!(summary.loss_per_completed_round.len(), 1);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(
            summary.final_loss,
            summary.loss_per_completed_round[summary.loss_per_completed_round.len() - 1]
        );
    }

    #[test]
    fn fit_iterations_summary_uses_round_count_as_round_cap() {
        let params = TrainParams {
            max_depth: 1,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.effective_round_cap, 3);
        assert_eq!(summary.rounds_completed, 3);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CompletedRequestedRounds
        );
        assert_eq!(summary.model.stumps.len(), 3);
        assert_eq!(summary.loss_per_completed_round.len(), 3);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
    }

    #[test]
    fn fit_iterations_grows_multiple_nodes_per_round_when_depth_allows() {
        let params = TrainParams {
            max_depth: 2,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(summary.model.stumps.len(), 3);
        let node_ids = summary
            .model
            .stumps
            .iter()
            .map(|stump| stump.split.node_id)
            .collect::<Vec<_>>();
        assert!(node_ids.contains(&0));
        assert!(node_ids.contains(&1));
        assert!(node_ids.contains(&2));
    }

    #[test]
    fn fit_iterations_controls_enforce_min_rows_per_leaf() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 3, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn iteration_controls_reject_invalid_values() {
        assert!(matches!(
            IterationControls::new(0, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, -0.1, 1, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 0, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, -0.1, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 0.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 2.0, 1.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, -0.1, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, 0.0, 0)
                .and_then(|controls| controls.with_subsample_rates(0.0, 1.0)),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, 0.0, 0)
                .and_then(|controls| controls.with_validation_early_stopping(0, 0.0)),
            Err(EngineError::InvalidConfig(_))
        ));
    }

    #[test]
    fn sampled_row_indices_are_seeded_and_non_prefix() {
        let selected = sampled_row_indices(8, 0.5, 17, 0);
        let selected_repeat = sampled_row_indices(8, 0.5, 17, 0);
        assert_eq!(selected, selected_repeat);
        assert_eq!(selected.len(), 4);
        assert_ne!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn sampled_feature_tiles_cover_expected_feature_count() {
        let (tiles, coverage_count) =
            sampled_feature_tiles(10, 0.3, 23, 0).expect("feature tiles should sample");
        assert_eq!(coverage_count, 3);
        let tile_coverage = tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum::<usize>();
        assert_eq!(tile_coverage, coverage_count);
    }

    #[test]
    fn sampled_feature_tiles_are_seeded_and_non_prefix() {
        let expand = |tiles: &[FeatureTile]| {
            tiles
                .iter()
                .flat_map(|tile| tile.start_feature..tile.end_feature)
                .map(|index| index as usize)
                .collect::<Vec<_>>()
        };

        let (tiles, coverage_count) =
            sampled_feature_tiles(12, 0.4, 17, 0).expect("feature tiles should sample");
        let (tiles_repeat, coverage_count_repeat) =
            sampled_feature_tiles(12, 0.4, 17, 0).expect("feature tiles should sample");
        assert_eq!(coverage_count, 5);
        assert_eq!(coverage_count, coverage_count_repeat);
        let selected = expand(&tiles);
        let selected_repeat = expand(&tiles_repeat);
        assert_eq!(selected, selected_repeat);
        assert_ne!(selected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sampled_indices_respect_ceil_minimum_and_upper_bound_rules() {
        let one_row = sampled_row_indices(5, 0.01, 5, 0);
        assert_eq!(one_row.len(), 1);

        let half_rows = sampled_row_indices(5, 0.5, 5, 0);
        assert_eq!(half_rows.len(), 3);

        let all_rows = sampled_row_indices(5, 1.0, 5, 0);
        assert_eq!(all_rows.len(), 5);

        let (_, one_feature) =
            sampled_feature_tiles(7, 0.01, 5, 0).expect("feature tiles should sample");
        assert_eq!(one_feature, 1);

        let (_, all_features) =
            sampled_feature_tiles(7, 1.0, 5, 0).expect("feature tiles should sample");
        assert_eq!(all_features, 7);
    }

    #[test]
    fn fit_iterations_controls_enforce_min_abs_leaf_value() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 10.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn fit_iterations_controls_clamp_leaf_values() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(1, 0.0, 1, 0.0, 0.1, 0.0, 0).expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(!model.stumps.is_empty());
        for stump in &model.stumps {
            assert!(stump.left_leaf_value.abs() <= 0.1);
            assert!(stump.right_leaf_value.abs() <= 0.1);
        }
    }

    #[test]
    fn fit_iterations_summary_reports_loss_improvement_threshold_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 100.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.rounds_completed, 0);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::LossImprovementBelowThreshold
        );
        assert!(summary.model.stumps.is_empty());
        assert!(summary.loss_per_completed_round.is_empty());
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_tracks_loss_trace_for_completed_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(2, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_completed, 2);
        assert_eq!(summary.loss_per_completed_round.len(), 2);
        assert!(summary.loss_per_completed_round[0] < summary.initial_loss);
        assert!(summary.loss_per_completed_round[1] <= summary.loss_per_completed_round[0]);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.final_loss, summary.loss_per_completed_round[1]);
    }

    #[test]
    fn fit_iterations_summary_allows_bounded_weak_improvement_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 100.0, 1)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::LossImprovementBelowThreshold
        );
        assert_eq!(summary.weak_improvement_rounds_committed, 1);
        assert_eq!(summary.loss_per_completed_round.len(), 1);
        assert!(!summary.model.stumps.is_empty());
    }

    #[test]
    fn predict_row_applies_non_root_nodes_only_when_path_matches() {
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 1,
            stumps: vec![
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 0,
                        gain: 1.0,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 1,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 1,
                        },
                    },
                    left_leaf_value: 0.0,
                    right_leaf_value: 1.0,
                },
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 2,
                        feature_index: 0,
                        threshold_bin: 0,
                        gain: 1.0,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 1,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 1,
                        },
                    },
                    left_leaf_value: 10.0,
                    right_leaf_value: 20.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
        };

        let left = model.predict_row(&[0.0]).expect("left prediction succeeds");
        let right = model
            .predict_row(&[1.0])
            .expect("right prediction succeeds");

        assert_eq!(left, 0.0);
        assert_eq!(right, 21.0);
    }

    #[test]
    fn retained_stump_count_for_rounds_handles_multi_stump_rounds() {
        let stumps_per_round = vec![3, 2, 4];
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 0), 0);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 1), 3);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 2), 5);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 3), 9);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 10), 9);
    }

    #[test]
    fn validation_early_stopping_requires_validation_dataset() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 0.0)
            .expect("validation controls are valid");
        let result = trainer.fit_iterations_with_summary(
            &sample_dataset(),
            &sample_binned_matrix(),
            &MockBackend,
            &SquaredErrorObjective,
            controls,
        );
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
    }

    #[test]
    fn fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 100.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let summary = trainer
            .fit_iterations_with_validation_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training with validation succeeds");

        assert_eq!(
            summary.stop_reason,
            IterationStopReason::ValidationLossPlateau
        );
        assert_eq!(summary.rounds_completed, 0);
        assert!(summary.model.stumps.is_empty());
        assert!(summary.initial_validation_loss.is_some());
        assert!(summary.validation_loss_per_completed_round.is_empty());
        assert_eq!(
            summary.best_validation_loss,
            summary.initial_validation_loss
        );
        assert_eq!(summary.best_validation_round, Some(0));
        assert!(summary.final_validation_loss.is_some());
        assert_eq!(
            summary.final_validation_loss,
            summary.initial_validation_loss
        );
        assert!(summary.sampled_rows_per_completed_round.is_empty());
        assert!(summary.sampled_features_per_completed_round.is_empty());
        assert_eq!(summary.final_loss, summary.initial_loss);
    }

    #[test]
    fn trained_model_artifact_roundtrip_preserves_predictions() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact parses");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(model.feature_count, restored.feature_count);
        assert_eq!(model.stumps.len(), restored.stumps.len());
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].descriptor.kind, ModelSectionKind::Trees);
        assert_eq!(
            parsed.sections[1].descriptor.kind,
            ModelSectionKind::PredictorLayout
        );

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn trained_model_artifact_roundtrip_preserves_optional_categorical_state() {
        let model = sample_trained_model()
            .with_categorical_state(Some(CategoricalStatePayloadV1 {
                format_version: alloygbm_core::CATEGORICAL_STATE_FORMAT_V1,
                leakage_safe_target_encoding: true,
                categorical_feature_indices: vec![1],
            }))
            .expect("categorical state is valid");
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact parses");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(parsed.sections.len(), 3);
        assert_eq!(
            parsed.sections[2].descriptor.kind,
            ModelSectionKind::CategoricalState
        );
        assert_eq!(model.categorical_state, restored.categorical_state);
    }

    #[test]
    fn fit_iterations_with_single_target_encoded_feature_attaches_categorical_state() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: vec![
                "A".to_string(),
                "A".to_string(),
                "B".to_string(),
                "B".to_string(),
            ],
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };
        let model = trainer
            .fit_iterations_with_single_target_encoded_feature(
                &sample_dataset(),
                &sample_binned_matrix(),
                &spec,
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("training succeeds");

        let state = model
            .categorical_state
            .as_ref()
            .expect("categorical state is attached");
        assert_eq!(
            state.format_version,
            alloygbm_core::CATEGORICAL_STATE_FORMAT_V1
        );
        assert!(!state.leakage_safe_target_encoding);
        assert_eq!(state.categorical_feature_indices, vec![1]);
    }

    #[test]
    fn trained_model_artifact_accepts_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");

        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");
        let restored =
            TrainedModel::from_artifact_bytes(&legacy_trees_only).expect("legacy artifact parses");

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn strict_mode_rejects_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        assert!(matches!(
            TrainedModel::from_artifact_bytes_with_mode(
                &legacy_trees_only,
                ArtifactCompatibilityMode::Strict
            ),
            Err(EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn strict_mode_accepts_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored =
            TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::Strict)
                .expect("strict artifact parse succeeds");

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn artifact_compatibility_report_classifies_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let report =
            TrainedModel::artifact_compatibility_report(&bytes).expect("report should parse");

        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 1);
        assert!(report.strict_compatible);
        assert!(!report.legacy_trees_only_compatible);
        assert!(report.legacy_compatible);
        assert_eq!(
            report.recommended_mode,
            Some(ArtifactCompatibilityMode::Strict)
        );
    }

    #[test]
    fn artifact_compatibility_report_classifies_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        let report = TrainedModel::artifact_compatibility_report(&legacy_trees_only)
            .expect("report should parse");
        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 0);
        assert!(!report.strict_compatible);
        assert!(report.legacy_trees_only_compatible);
        assert!(report.legacy_compatible);
        assert_eq!(
            report.recommended_mode,
            Some(ArtifactCompatibilityMode::AllowLegacyTreesOnly)
        );
    }

    #[test]
    fn subtract_histogram_bundle_derives_complementary_child() {
        let parent = HistogramBundle {
            node_id: 7,
            feature_histograms: vec![
                alloygbm_core::FeatureHistogram {
                    feature_index: 0,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 3.0,
                            hess_sum: 5.0,
                            count: 4,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            count: 2,
                        },
                    ],
                },
                alloygbm_core::FeatureHistogram {
                    feature_index: 1,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 1.5,
                            hess_sum: 4.0,
                            count: 3,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            count: 1,
                        },
                    ],
                },
            ],
        };
        let child = HistogramBundle {
            node_id: 15,
            feature_histograms: vec![
                alloygbm_core::FeatureHistogram {
                    feature_index: 0,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 2.0,
                            hess_sum: 3.0,
                            count: 2,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.25,
                            hess_sum: 0.5,
                            count: 1,
                        },
                    ],
                },
                alloygbm_core::FeatureHistogram {
                    feature_index: 1,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 1.0,
                            hess_sum: 2.5,
                            count: 2,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.25,
                            hess_sum: 0.25,
                            count: 1,
                        },
                    ],
                },
            ],
        };

        let complement =
            subtract_histogram_bundle(&parent, &child, 16).expect("subtraction should succeed");
        assert_eq!(complement.node_id, 16);
        assert_eq!(complement.feature_histograms.len(), 2);
        assert_eq!(complement.feature_histograms[0].bins[0].count, 2);
        assert_eq!(complement.feature_histograms[0].bins[1].count, 1);
        assert!((complement.feature_histograms[0].bins[0].grad_sum - 1.0).abs() < 1e-6);
        assert!((complement.feature_histograms[0].bins[1].grad_sum + 0.75).abs() < 1e-6);
        assert!((complement.feature_histograms[1].bins[0].hess_sum - 1.5).abs() < 1e-6);
        assert!((complement.feature_histograms[1].bins[1].hess_sum - 0.75).abs() < 1e-6);
    }

    #[test]
    fn subtract_histogram_bundle_into_matches_allocating_variant() {
        let parent = HistogramBundle {
            node_id: 7,
            feature_histograms: vec![alloygbm_core::FeatureHistogram {
                feature_index: 0,
                bins: vec![
                    alloygbm_core::HistogramBin {
                        grad_sum: 3.0,
                        hess_sum: 5.0,
                        count: 4,
                    },
                    alloygbm_core::HistogramBin {
                        grad_sum: -1.0,
                        hess_sum: 2.0,
                        count: 2,
                    },
                ],
            }],
        };
        let child = HistogramBundle {
            node_id: 15,
            feature_histograms: vec![alloygbm_core::FeatureHistogram {
                feature_index: 0,
                bins: vec![
                    alloygbm_core::HistogramBin {
                        grad_sum: 2.0,
                        hess_sum: 3.0,
                        count: 2,
                    },
                    alloygbm_core::HistogramBin {
                        grad_sum: -0.25,
                        hess_sum: 0.5,
                        count: 1,
                    },
                ],
            }],
        };

        // Allocating variant
        let allocated =
            subtract_histogram_bundle(&parent, &child, 16).expect("subtraction should succeed");

        // In-place variant
        let mut dest = HistogramBundle::new_zeroed(&[0], 2);
        subtract_histogram_bundle_into(&parent, &child, 16, &mut dest)
            .expect("in-place subtraction should succeed");

        assert_eq!(allocated.node_id, dest.node_id);
        assert_eq!(
            allocated.feature_histograms.len(),
            dest.feature_histograms.len()
        );
        for (a, d) in allocated
            .feature_histograms
            .iter()
            .zip(&dest.feature_histograms)
        {
            assert_eq!(a.feature_index, d.feature_index);
            for (ab, db) in a.bins.iter().zip(&d.bins) {
                assert!((ab.grad_sum - db.grad_sum).abs() < 1e-6);
                assert!((ab.hess_sum - db.hess_sum).abs() < 1e-6);
                assert_eq!(ab.count, db.count);
            }
        }
    }

    #[test]
    fn histogram_bundle_reset_zeros_all_bins() {
        let mut bundle = HistogramBundle::new_zeroed(&[0, 1], 3);
        // Set some values
        bundle.feature_histograms[0].bins[0].grad_sum = 5.0;
        bundle.feature_histograms[0].bins[0].hess_sum = 3.0;
        bundle.feature_histograms[0].bins[0].count = 10;
        bundle.feature_histograms[1].bins[2].grad_sum = -2.5;
        bundle.feature_histograms[1].bins[2].count = 7;

        bundle.reset(42);
        assert_eq!(bundle.node_id, 42);
        for fh in &bundle.feature_histograms {
            for bin in &fh.bins {
                assert_eq!(bin.grad_sum, 0.0);
                assert_eq!(bin.hess_sum, 0.0);
                assert_eq!(bin.count, 0);
            }
        }
    }

    #[test]
    fn histogram_bundle_new_zeroed_creates_correct_structure() {
        let features = [0, 3, 7];
        let bundle = HistogramBundle::new_zeroed(&features, 5);
        assert_eq!(bundle.feature_histograms.len(), 3);
        assert_eq!(bundle.feature_histograms[0].feature_index, 0);
        assert_eq!(bundle.feature_histograms[1].feature_index, 3);
        assert_eq!(bundle.feature_histograms[2].feature_index, 7);
        for fh in &bundle.feature_histograms {
            assert_eq!(fh.bins.len(), 5);
        }
    }

    #[test]
    fn artifact_compatibility_report_marks_malformed_required_sections_incompatible() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");
        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
            .expect("report should parse");
        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 2);
        assert!(!report.strict_compatible);
        assert!(!report.legacy_trees_only_compatible);
        assert!(!report.legacy_compatible);
        assert_eq!(report.recommended_mode, None);
    }

    #[test]
    fn from_artifact_bytes_auto_selects_strict_for_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let (restored, selected_mode) =
            TrainedModel::from_artifact_bytes_auto(&bytes).expect("auto import succeeds");

        assert_eq!(selected_mode, ArtifactCompatibilityMode::Strict);
        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn from_artifact_bytes_auto_selects_legacy_for_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        let (restored, selected_mode) = TrainedModel::from_artifact_bytes_auto(&legacy_trees_only)
            .expect("auto import succeeds");
        assert_eq!(
            selected_mode,
            ArtifactCompatibilityMode::AllowLegacyTreesOnly
        );

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn from_artifact_bytes_auto_rejects_malformed_required_section_layouts() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");
        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let result = TrainedModel::from_artifact_bytes_auto(&duplicate_predictor);
        match result {
            Err(EngineError::ContractViolation(message)) => {
                let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
                    .expect("report should parse");
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_auto_mode_error(
                        report.required_section_report()
                    )
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    #[test]
    fn trained_model_artifact_rejects_missing_required_sections() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");

        let non_legacy_missing_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload.clone()),
                (ModelSectionKind::ShapAux, vec![9_u8]),
            ],
        )
        .expect("artifact serializes");
        let missing_trees = serialize_model_artifact_v1(
            &metadata,
            &[(ModelSectionKind::PredictorLayout, layout_payload.clone())],
        )
        .expect("artifact serializes");

        assert!(matches!(
            TrainedModel::from_artifact_bytes(&non_legacy_missing_predictor),
            Err(EngineError::ContractViolation(_))
        ));
        assert!(matches!(
            TrainedModel::from_artifact_bytes(&missing_trees),
            Err(EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn trained_model_artifact_rejects_duplicate_required_sections() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");

        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let result = TrainedModel::from_artifact_bytes(&duplicate_predictor);
        match result {
            Err(EngineError::ContractViolation(message)) => {
                let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
                    .expect("report should parse");
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(
                        report.required_section_report(),
                        true
                    )
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    // -- Multi-class tests ---------------------------------------------------

    #[test]
    fn test_multiclass_softmax_rejects_single_class() {
        let result = MultiClassSoftmaxObjective::new(1);
        assert!(result.is_err());
        if let Err(EngineError::InvalidConfig(msg)) = result {
            assert!(msg.contains("at least 2"), "unexpected error: {msg}");
        } else {
            panic!("expected InvalidConfig error");
        }
    }

    #[test]
    fn test_multiclass_softmax_rejects_zero_classes() {
        let result = MultiClassSoftmaxObjective::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiclass_softmax_creates_with_valid_k() {
        let obj = MultiClassSoftmaxObjective::new(3).expect("k=3 should work");
        assert_eq!(obj.num_classes, 3);
        assert_eq!(obj.objective_name(), "multiclass_softmax");
    }

    #[test]
    fn test_multiclass_softmax_initial_predictions() {
        let obj = MultiClassSoftmaxObjective::new(4).unwrap();
        let preds = obj.initial_predictions();
        assert_eq!(preds.len(), 4);
        for &p in &preds {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn test_multiclass_softmax_gradients_basic() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        // 3 samples with targets [0, 1, 2]
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Initial predictions: all zeros -> uniform softmax: 1/3 each
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let mut buffer = Vec::new();

        // Gradients for class 0:
        // Sample 0 (target=0): grad = p_0 - 1 = 1/3 - 1 = -2/3
        // Sample 1 (target=1): grad = p_0 - 0 = 1/3
        // Sample 2 (target=2): grad = p_0 - 0 = 1/3
        let _ = obj.compute_gradients_for_class(&predictions, &targets, None, 0, &mut buffer);
        assert_eq!(buffer.len(), 3);
        // Sample 0: grad should be negative (correct class)
        assert!(
            buffer[0].grad < 0.0,
            "grad for correct class should be negative"
        );
        // Sample 1: grad should be positive (wrong class)
        assert!(
            buffer[1].grad > 0.0,
            "grad for wrong class should be positive"
        );
        // Sample 2: grad should be positive (wrong class)
        assert!(
            buffer[2].grad > 0.0,
            "grad for wrong class should be positive"
        );

        // Hessians should all be positive
        for gp in &buffer {
            assert!(gp.hess > 0.0, "hessian must be positive");
        }

        // Verify approximate values: grad ≈ -2/3 for correct, 1/3 for wrong
        assert!((buffer[0].grad - (-2.0 / 3.0)).abs() < 0.01);
        assert!((buffer[1].grad - (1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_multiclass_softmax_gradients_with_weights() {
        let obj = MultiClassSoftmaxObjective::new(2).unwrap();
        let targets = vec![0.0_f32, 1.0];
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 2]; 2];
        let weights = vec![2.0_f32, 0.5];
        let mut buffer = Vec::new();

        let _ =
            obj.compute_gradients_for_class(&predictions, &targets, Some(&weights), 0, &mut buffer);
        // With uniform softmax (p=0.5): grad_0 = (0.5 - 1) * 2.0 = -1.0
        assert!((buffer[0].grad - (-1.0)).abs() < 0.01);
        // grad_1 = (0.5 - 0) * 0.5 = 0.25
        assert!((buffer[1].grad - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_multiclass_softmax_loss() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Uniform predictions: loss = -log(1/3) = log(3) ≈ 1.0986
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let loss = obj.loss(&predictions, &targets, None).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
        let expected = (3.0_f32).ln();
        assert!((loss - expected).abs() < 0.01, "loss {loss} ≈ {expected}");
    }

    #[test]
    fn test_multiclass_softmax_loss_perfect_predictions() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Strong predictions toward correct classes
        let predictions: Vec<Vec<f32>> = vec![
            vec![10.0, -10.0, -10.0], // strongly class 0
            vec![-10.0, 10.0, -10.0], // strongly class 1
            vec![-10.0, -10.0, 10.0], // strongly class 2
        ];
        let loss = obj.loss(&predictions, &targets, None).unwrap();
        assert!(
            loss < 0.01,
            "loss should be near zero for perfect predictions, got {loss}"
        );
    }

    #[test]
    fn test_multiclass_trained_model_artifact_roundtrip() {
        // Create a minimal model manually
        let model = MultiClassTrainedModel {
            num_classes: 3,
            baseline_predictions: vec![0.0, 0.0, 0.0],
            feature_count: 2,
            class_stumps: vec![
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 1,
                        gain: 2.5,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            row_count: 3,
                        },
                        right_stats: NodeStats {
                            grad_sum: 1.0,
                            hess_sum: 2.0,
                            row_count: 3,
                        },
                    },
                    left_leaf_value: -0.1,
                    right_leaf_value: 0.1,
                }],
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 1,
                        threshold_bin: 2,
                        gain: 1.5,
                        default_left: true,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.5,
                            hess_sum: 1.0,
                            row_count: 2,
                        },
                        right_stats: NodeStats {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            row_count: 4,
                        },
                    },
                    left_leaf_value: 0.2,
                    right_leaf_value: -0.05,
                }],
                vec![], // class 2 has no stumps
            ],
            categorical_state: None,
            objective: "multiclass_softmax".to_string(),
        };

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        // Deserialize
        let restored = MultiClassTrainedModel::from_artifact_bytes(&bytes)
            .expect("deserialize should succeed");
        assert_eq!(restored.num_classes, 3);
        assert_eq!(restored.feature_count, 2);
        assert_eq!(restored.baseline_predictions, vec![0.0, 0.0, 0.0]);
        assert_eq!(restored.class_stumps.len(), 3);
        assert_eq!(restored.class_stumps[0].len(), 1);
        assert_eq!(restored.class_stumps[1].len(), 1);
        assert_eq!(restored.class_stumps[2].len(), 0);
        assert_eq!(restored.objective, "multiclass_softmax");
    }

    #[test]
    fn test_multiclass_trained_model_rounds_completed() {
        let model = MultiClassTrainedModel {
            num_classes: 2,
            baseline_predictions: vec![0.0, 0.0],
            feature_count: 1,
            class_stumps: vec![
                // Class 0: 2 trees (round 0 has 3 stumps, round 1 has 1 stump)
                vec![
                    TrainedStump {
                        split: SplitCandidate {
                            node_id: 0, // tree 0, node 0
                            feature_index: 0,
                            threshold_bin: 1,
                            gain: 1.0,
                            default_left: false,
                            is_categorical: false,
                            categorical_bitset: None,
                            left_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                row_count: 2,
                            },
                            right_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                row_count: 2,
                            },
                        },
                        left_leaf_value: -0.1,
                        right_leaf_value: 0.1,
                    },
                    TrainedStump {
                        split: SplitCandidate {
                            node_id: TREE_NODE_STRIDE, // tree 1, node 0
                            feature_index: 0,
                            threshold_bin: 2,
                            gain: 0.5,
                            default_left: false,
                            is_categorical: false,
                            categorical_bitset: None,
                            left_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                row_count: 2,
                            },
                            right_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                row_count: 2,
                            },
                        },
                        left_leaf_value: -0.05,
                        right_leaf_value: 0.05,
                    },
                ],
                // Class 1: same structure
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 1,
                        gain: 1.0,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 2,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            row_count: 2,
                        },
                    },
                    left_leaf_value: 0.1,
                    right_leaf_value: -0.1,
                }],
            ],
            categorical_state: None,
            objective: "multiclass_softmax".to_string(),
        };
        assert_eq!(model.rounds_completed(), 2);
    }

    // ── Per-round metric callback tests ─────────────────────────────────

    /// A simple test callback that returns MSE as the metric value.
    struct MseMetricCallback;

    impl PerRoundMetricCallback for MseMetricCallback {
        fn evaluate(
            &self,
            predictions: &[f32],
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            if predictions.len() != targets.len() {
                return Err(EngineError::ContractViolation(
                    "predictions and targets length mismatch".into(),
                ));
            }
            let n = predictions.len() as f32;
            let mse: f32 = predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / n;
            Ok(mse)
        }
        fn higher_is_better(&self) -> bool {
            false
        }
        fn metric_name(&self) -> &str {
            "test_mse"
        }
    }

    /// A metric callback where higher values are better (e.g. R²-like).
    struct HigherIsBetterCallback;

    impl PerRoundMetricCallback for HigherIsBetterCallback {
        fn evaluate(
            &self,
            predictions: &[f32],
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            // Return negative MSE so higher = better
            let n = predictions.len() as f32;
            let neg_mse: f32 = -(predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / n);
            Ok(neg_mse)
        }
        fn higher_is_better(&self) -> bool {
            true
        }
        fn metric_name(&self) -> &str {
            "neg_mse"
        }
    }

    #[test]
    fn test_per_round_callback_basic() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(5, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = MseMetricCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with metric callback succeeds");

        // Callback should have been invoked each round.
        assert_eq!(
            summary.custom_metric_per_round.len(),
            summary.rounds_completed
        );
        assert_eq!(summary.custom_metric_name.as_deref(), Some("test_mse"));
        // Metric values should be non-negative (MSE).
        for v in &summary.custom_metric_per_round {
            assert!(*v >= 0.0, "MSE metric should be non-negative");
        }
    }

    #[test]
    fn test_per_round_callback_early_stopping() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        // Set early_stopping_rounds=1 with very high min_improvement so it
        // stops almost immediately when the custom metric plateaus.
        let controls = IterationControls::new(100, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 1000.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = MseMetricCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with callback early stopping succeeds");

        // Should have stopped early due to the custom metric plateau.
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
            "expected custom metric plateau stop reason, got {:?}",
            summary.stop_reason,
        );
        assert!(
            summary.rounds_completed < 100,
            "should have stopped before all 100 rounds"
        );
    }

    #[test]
    fn test_per_round_callback_higher_is_better() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(100, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 1000.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = HigherIsBetterCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with higher-is-better callback succeeds");

        // Should also stop early.
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
        );
        assert_eq!(summary.custom_metric_name.as_deref(), Some("neg_mse"));
    }

    #[test]
    fn test_per_round_callback_none_no_effect() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        // Pass None — should behave identically to the non-metric path.
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                None,
            )
            .expect("training without callback succeeds");

        assert!(summary.custom_metric_per_round.is_empty());
        assert!(summary.custom_metric_name.is_none());
        assert_ne!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
        );
    }

    #[test]
    fn test_custom_metric_plateau_stop_reason() {
        // Verify the CustomMetricPlateau variant is distinct from ValidationLossPlateau.
        assert_ne!(
            IterationStopReason::CustomMetricPlateau,
            IterationStopReason::ValidationLossPlateau,
        );
        assert_ne!(
            IterationStopReason::CustomMetricPlateau,
            IterationStopReason::CompletedRequestedRounds,
        );
    }

    // ── Native categorical split tests ──────────────────────────────────

    #[test]
    fn test_trained_model_categorical_roundtrip() {
        // Build a TrainedModel with one categorical stump and one continuous stump.
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 0,
                        gain: 2.0,
                        default_left: true,
                        is_categorical: true,
                        categorical_bitset: Some(vec![0b0000_0011]), // cats 0,1 left
                        left_stats: NodeStats {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            row_count: 10,
                        },
                        right_stats: NodeStats {
                            grad_sum: 1.0,
                            hess_sum: 2.0,
                            row_count: 10,
                        },
                    },
                    left_leaf_value: -0.1,
                    right_leaf_value: 0.1,
                },
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 1,
                        threshold_bin: 3,
                        gain: 1.5,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.5,
                            hess_sum: 1.0,
                            row_count: 5,
                        },
                        right_stats: NodeStats {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            row_count: 5,
                        },
                    },
                    left_leaf_value: 0.05,
                    right_leaf_value: -0.05,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
        };

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        let restored = TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::AllowLegacyTreesOnly)
            .expect("deserialize should succeed");

        // Verify basic fields
        assert_eq!(restored.feature_count, 2);
        assert_eq!(restored.stumps.len(), 2);
        assert_eq!(restored.native_categorical_feature_indices, vec![0]);

        // Verify categorical stump
        let stump0 = &restored.stumps[0];
        assert!(stump0.split.is_categorical);
        assert_eq!(stump0.split.categorical_bitset, Some(vec![0b0000_0011]));
        assert_eq!(stump0.left_leaf_value, -0.1);
        assert_eq!(stump0.right_leaf_value, 0.1);

        // Verify continuous stump
        let stump1 = &restored.stumps[1];
        assert!(!stump1.split.is_categorical);
        assert!(stump1.split.categorical_bitset.is_none());
        assert_eq!(stump1.split.threshold_bin, 3);
    }

    #[test]
    fn test_trained_model_categorical_backward_compat() {
        // Build a model WITHOUT any categorical stumps (old-style).
        let model = TrainedModel {
            baseline_prediction: 1.0,
            feature_count: 1,
            stumps: vec![TrainedStump {
                split: SplitCandidate {
                    node_id: 0,
                    feature_index: 0,
                    threshold_bin: 2,
                    gain: 1.0,
                    default_left: false,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: -0.5,
                        hess_sum: 1.0,
                        row_count: 3,
                    },
                    right_stats: NodeStats {
                        grad_sum: 0.5,
                        hess_sum: 1.0,
                        row_count: 3,
                    },
                },
                left_leaf_value: -0.2,
                right_leaf_value: 0.2,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
        };

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");

        // Deserialize — should work fine with no categorical section
        let restored = TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::AllowLegacyTreesOnly)
            .expect("deserialize should succeed");
        assert_eq!(restored.stumps.len(), 1);
        assert!(!restored.stumps[0].split.is_categorical);
        assert!(restored.native_categorical_feature_indices.is_empty());
    }
}
