use alloygbm_core::{
    BinnedMatrix, CoreError, Device, FeatureTile, GradientPair, HistogramBundle, MODEL_FORMAT_V1,
    ModelArtifactSection, ModelMetadata, ModelSectionKind, NodeSlice, NodeStats, PartitionResult,
    SplitCandidate, TrainParams, TrainingDataset, deserialize_model_artifact_v1,
    serialize_model_artifact_v1, validate_binned_matrix, validate_train_params,
    validate_training_dataset,
};
use std::error::Error;
use std::fmt::{Display, Formatter};

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

pub trait BackendOps {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle>;
    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>>;
    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult>;
    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats>;
}

pub trait ObjectiveOps {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquaredErrorObjective;

impl ObjectiveOps for SquaredErrorObjective {
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

#[derive(Debug, Clone, PartialEq)]
pub struct TrainedStump {
    pub split: SplitCandidate,
    pub left_leaf_value: f32,
    pub right_leaf_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainedModel {
    pub baseline_prediction: f32,
    pub feature_count: usize,
    pub stumps: Vec<TrainedStump>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationControls {
    pub rounds: usize,
    pub min_split_gain: f32,
    pub min_rows_per_leaf: usize,
    pub min_abs_leaf_value: f32,
    pub max_abs_leaf_value: f32,
    pub min_loss_improvement: f32,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct IterationRunSummary {
    pub model: TrainedModel,
    pub rounds_requested: usize,
    pub effective_round_cap: usize,
    pub rounds_completed: usize,
    pub stop_reason: IterationStopReason,
    pub initial_loss: f32,
    pub loss_per_completed_round: Vec<f32>,
    pub final_loss: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactCompatibilityMode {
    Strict,
    AllowLegacyTreesOnly,
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

impl IterationControls {
    pub fn new(
        rounds: usize,
        min_split_gain: f32,
        min_rows_per_leaf: usize,
        min_abs_leaf_value: f32,
        max_abs_leaf_value: f32,
        min_loss_improvement: f32,
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
        })
    }
}

impl TrainedModel {
    pub fn predict_row(&self, features: &[f32]) -> EngineResult<f32> {
        if features.len() != self.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                self.feature_count
            )));
        }

        let mut prediction = self.baseline_prediction;
        for stump in &self.stumps {
            let feature_index = stump.split.feature_index as usize;
            let feature_value = features[feature_index];
            let threshold = stump.split.threshold_bin as f32;
            prediction += if feature_value <= threshold {
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
        };

        serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, predictor_layout_payload),
            ],
        )
        .map_err(EngineError::from)
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
            EngineError::ContractViolation(format!(
                "unable to determine artifact compatibility mode (Trees sections: {}, PredictorLayout sections: {})",
                report.trees_section_count, report.predictor_layout_section_count
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
                return Err(EngineError::ContractViolation(format!(
                    "strict compatibility mode requires exactly one Trees and one PredictorLayout section (found Trees={}, PredictorLayout={})",
                    compatibility_report.trees_section_count,
                    compatibility_report.predictor_layout_section_count
                )));
            }
            ArtifactCompatibilityMode::AllowLegacyTreesOnly
                if !compatibility_report.legacy_compatible =>
            {
                return Err(EngineError::ContractViolation(format!(
                    "legacy-compatible mode only supports strict dual-section artifacts or legacy Trees-only artifacts (found Trees={}, PredictorLayout={})",
                    compatibility_report.trees_section_count,
                    compatibility_report.predictor_layout_section_count
                )));
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

        model.feature_count = metadata_feature_count;
        Ok(model)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trainer {
    params: TrainParams,
}

impl Trainer {
    pub fn new(params: TrainParams) -> EngineResult<Self> {
        validate_train_params(&params)?;
        Ok(Self { params })
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

        let histograms = backend.build_histograms(
            binned_matrix,
            &fit_contract.gradients,
            &root_node,
            &feature_tiles,
        )?;
        let split_candidate = backend.best_split(&histograms)?;
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
        let controls = IterationControls::new(rounds, 0.0, 1, 0.0, 1_000_000.0, 0.0)?;
        self.fit_iterations_with_controls(dataset, binned_matrix, backend, objective, controls)
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
        validate_iteration_controls(controls)?;
        validate_training_alignment(dataset, binned_matrix)?;
        let fit_contract = self.validate_fit_contract(dataset, objective)?;

        let mut predictions = vec![fit_contract.baseline_prediction; dataset.row_count()];
        let root_row_indices = (0..dataset.row_count() as u32).collect::<Vec<_>>();
        let root_node = NodeSlice::new(0, root_row_indices)?;
        let feature_tiles = vec![FeatureTile::new(0, binned_matrix.feature_count as u32)?];
        let mut stumps = Vec::new();
        let mut rounds_completed = 0_usize;
        let effective_round_cap = controls.rounds.min(self.params.max_depth as usize);
        let mut stop_reason = if controls.rounds > effective_round_cap {
            IterationStopReason::DepthBudgetReached
        } else {
            IterationStopReason::CompletedRequestedRounds
        };
        let initial_loss = squared_error_loss(
            &predictions,
            &dataset.targets,
            dataset.sample_weights.as_deref(),
        )?;
        let mut current_loss = initial_loss;
        let mut loss_per_completed_round = Vec::new();

        const LEAF_EPSILON: f32 = 1e-6;

        for _round in 0..effective_round_cap {
            let gradients = objective.compute_gradients(
                &predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            validate_gradient_pairs(&gradients, dataset.row_count())?;

            let histograms =
                backend.build_histograms(binned_matrix, &gradients, &root_node, &feature_tiles)?;
            let Some(mut split) = backend.best_split(&histograms)? else {
                stop_reason = IterationStopReason::NoSplitCandidate;
                break;
            };
            if !split.gain.is_finite() || split.gain <= controls.min_split_gain {
                stop_reason = IterationStopReason::GainBelowThreshold;
                break;
            }

            let partition = backend.apply_split(binned_matrix, &root_node, &split)?;
            validate_partition_cover(dataset.row_count(), &partition)?;
            if partition.left_row_indices.len() < controls.min_rows_per_leaf
                || partition.right_row_indices.len() < controls.min_rows_per_leaf
            {
                stop_reason = IterationStopReason::LeafRowsBelowThreshold;
                break;
            }

            let left_stats = backend.reduce_sums(&gradients, &partition.left_row_indices)?;
            let right_stats = backend.reduce_sums(&gradients, &partition.right_row_indices)?;
            if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "backend produced non-positive hessian sums".to_string(),
                ));
            }

            let raw_left_leaf_value = -self.params.learning_rate * left_stats.grad_sum
                / (left_stats.hess_sum + LEAF_EPSILON);
            let raw_right_leaf_value = -self.params.learning_rate * right_stats.grad_sum
                / (right_stats.hess_sum + LEAF_EPSILON);

            let left_leaf_value = raw_left_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let right_leaf_value = raw_right_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            if left_leaf_value.abs() < controls.min_abs_leaf_value
                && right_leaf_value.abs() < controls.min_abs_leaf_value
            {
                stop_reason = IterationStopReason::LeafMagnitudeBelowThreshold;
                break;
            }

            let mut candidate_predictions = predictions.clone();
            for &row_index in &partition.left_row_indices {
                candidate_predictions[row_index as usize] += left_leaf_value;
            }
            for &row_index in &partition.right_row_indices {
                candidate_predictions[row_index as usize] += right_leaf_value;
            }
            let candidate_loss = squared_error_loss(
                &candidate_predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            let loss_improvement = current_loss - candidate_loss;
            if loss_improvement < controls.min_loss_improvement {
                stop_reason = IterationStopReason::LossImprovementBelowThreshold;
                break;
            }
            predictions = candidate_predictions;
            current_loss = candidate_loss;
            loss_per_completed_round.push(candidate_loss);

            split.left_stats = left_stats;
            split.right_stats = right_stats;

            stumps.push(TrainedStump {
                split,
                left_leaf_value,
                right_leaf_value,
            });
            rounds_completed += 1;
        }

        let model = TrainedModel {
            baseline_prediction: fit_contract.baseline_prediction,
            feature_count: dataset.matrix.feature_count,
            stumps,
        };
        let final_loss = current_loss;

        Ok(IterationRunSummary {
            model,
            rounds_requested: controls.rounds,
            effective_round_cap,
            rounds_completed,
            stop_reason,
            initial_loss,
            loss_per_completed_round,
            final_loss,
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
    if gradients.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "objective returned {} gradients for row_count {}",
            gradients.len(),
            row_count
        )));
    }
    for gradient in gradients {
        if !gradient.grad.is_finite() || !gradient.hess.is_finite() || gradient.hess <= 0.0 {
            return Err(EngineError::ContractViolation(
                "objective produced invalid gradient/hessian values".to_string(),
            ));
        }
    }
    Ok(())
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
    Ok(())
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

    let mut total = 0.0_f32;
    for index in 0..predictions.len() {
        let weight = sample_weights.map_or(1.0, |weights| weights[index]);
        if !weight.is_finite() || weight <= 0.0 {
            return Err(EngineError::ContractViolation(
                "sample weights must be finite and > 0".to_string(),
            ));
        }
        let residual = predictions[index] - targets[index];
        total += residual * residual * weight;
    }

    Ok(total)
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
    let trees_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::Trees)
        .count();
    let predictor_layout_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
        .count();

    let strict_compatible = trees_section_count == 1 && predictor_layout_section_count == 1;
    let legacy_trees_only_compatible = trees_section_count == 1
        && predictor_layout_section_count == 0
        && sections.len() == 1
        && sections[0].descriptor.kind == ModelSectionKind::Trees;
    let legacy_compatible = strict_compatible || legacy_trees_only_compatible;
    let recommended_mode = if strict_compatible {
        Some(ArtifactCompatibilityMode::Strict)
    } else if legacy_trees_only_compatible {
        Some(ArtifactCompatibilityMode::AllowLegacyTreesOnly)
    } else {
        None
    };

    ArtifactCompatibilityReport {
        trees_section_count,
        predictor_layout_section_count,
        strict_compatible,
        legacy_trees_only_compatible,
        legacy_compatible,
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
        bytes.extend_from_slice(&0_u16.to_le_bytes());
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
            Ok(Some(SplitCandidate {
                node_id: histograms.node_id,
                feature_index: 0,
                threshold_bin: 1,
                gain: 3.0,
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
            _binned_matrix: &BinnedMatrix,
            _node: &NodeSlice,
            _split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            Ok(PartitionResult {
                left_row_indices: vec![0, 1],
                right_row_indices: vec![2, 3],
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
    fn fit_iterations_controls_enforce_min_split_gain() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
        let controls =
            IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_reports_completed_requested_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
        assert_eq!(
            summary.final_loss,
            summary.loss_per_completed_round[summary.loss_per_completed_round.len() - 1]
        );
    }

    #[test]
    fn fit_iterations_summary_reports_depth_budget_stop_reason() {
        let params = TrainParams {
            max_depth: 1,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let controls =
            IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
        assert_eq!(summary.effective_round_cap, 1);
        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(summary.stop_reason, IterationStopReason::DepthBudgetReached);
        assert_eq!(summary.model.stumps.len(), 1);
        assert_eq!(summary.loss_per_completed_round.len(), 1);
    }

    #[test]
    fn fit_iterations_controls_enforce_min_rows_per_leaf() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(3, 0.0, 3, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
            IterationControls::new(0, 0.0, 1, 0.0, 1_000_000.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, -0.1, 1, 0.0, 1_000_000.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 0, 0.0, 1_000_000.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, -0.1, 1_000_000.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 0.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 2.0, 1.0, 0.0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, -0.1),
            Err(EngineError::InvalidConfig(_))
        ));
    }

    #[test]
    fn fit_iterations_controls_enforce_min_abs_leaf_value() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(3, 0.0, 1, 10.0, 1_000_000.0, 0.0).expect("controls are valid");
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
            IterationControls::new(1, 0.0, 1, 0.0, 0.1, 0.0).expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(model.stumps.len(), 1);
        let stump = &model.stumps[0];
        assert!(stump.left_leaf_value.abs() <= 0.1);
        assert!(stump.right_leaf_value.abs() <= 0.1);
    }

    #[test]
    fn fit_iterations_summary_reports_loss_improvement_threshold_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 100.0).expect("controls are valid");
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
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_tracks_loss_trace_for_completed_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(2, 0.0, 1, 0.0, 1_000_000.0, 0.0).expect("controls are valid");
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
        assert_eq!(summary.final_loss, summary.loss_per_completed_round[1]);
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
    fn trained_model_artifact_accepts_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
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
    fn artifact_compatibility_report_marks_malformed_required_sections_incompatible() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
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

        assert!(matches!(
            TrainedModel::from_artifact_bytes_auto(&duplicate_predictor),
            Err(EngineError::ContractViolation(_))
        ));
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

        assert!(matches!(
            TrainedModel::from_artifact_bytes(&duplicate_predictor),
            Err(EngineError::ContractViolation(_))
        ));
    }
}
