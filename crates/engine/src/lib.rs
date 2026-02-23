use alloygbm_core::{
    BinnedMatrix, CoreError, Device, FeatureTile, GradientPair, HistogramBundle, MODEL_FORMAT_V1,
    ModelMetadata, ModelSectionKind, NodeSlice, NodeStats, PartitionResult, SplitCandidate,
    TrainParams, TrainingDataset, deserialize_model_artifact_v1, serialize_model_artifact_v1,
    validate_binned_matrix, validate_train_params, validate_training_dataset,
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
        let payload = encode_trained_model_payload(self)?;
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..self.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
        };

        serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, payload)])
            .map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;

        let tree_section = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .ok_or_else(|| {
                EngineError::ContractViolation(
                    "model artifact missing required Trees section".to_string(),
                )
            })?;

        let mut model = decode_trained_model_payload(&tree_section.payload)?;
        if model.feature_count != parsed.contract.metadata.feature_names.len() {
            return Err(EngineError::ContractViolation(format!(
                "decoded feature_count {} does not match metadata feature count {}",
                model.feature_count,
                parsed.contract.metadata.feature_names.len()
            )));
        }

        model.feature_count = parsed.contract.metadata.feature_names.len();
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
        if rounds == 0 {
            return Err(EngineError::InvalidConfig(
                "rounds must be greater than 0".to_string(),
            ));
        }

        validate_training_alignment(dataset, binned_matrix)?;
        let fit_contract = self.validate_fit_contract(dataset, objective)?;

        let mut predictions = vec![fit_contract.baseline_prediction; dataset.row_count()];
        let root_row_indices = (0..dataset.row_count() as u32).collect::<Vec<_>>();
        let root_node = NodeSlice::new(0, root_row_indices)?;
        let feature_tiles = vec![FeatureTile::new(0, binned_matrix.feature_count as u32)?];
        let mut stumps = Vec::new();

        const LEAF_EPSILON: f32 = 1e-6;

        for _round in 0..rounds {
            let gradients = objective.compute_gradients(
                &predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )?;
            validate_gradient_pairs(&gradients, dataset.row_count())?;

            let histograms =
                backend.build_histograms(binned_matrix, &gradients, &root_node, &feature_tiles)?;
            let Some(mut split) = backend.best_split(&histograms)? else {
                break;
            };
            if !split.gain.is_finite() || split.gain <= 0.0 {
                break;
            }

            let partition = backend.apply_split(binned_matrix, &root_node, &split)?;
            validate_partition_cover(dataset.row_count(), &partition)?;

            let left_stats = backend.reduce_sums(&gradients, &partition.left_row_indices)?;
            let right_stats = backend.reduce_sums(&gradients, &partition.right_row_indices)?;
            if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "backend produced non-positive hessian sums".to_string(),
                ));
            }

            let left_leaf_value = -self.params.learning_rate * left_stats.grad_sum
                / (left_stats.hess_sum + LEAF_EPSILON);
            let right_leaf_value = -self.params.learning_rate * right_stats.grad_sum
                / (right_stats.hess_sum + LEAF_EPSILON);

            for &row_index in &partition.left_row_indices {
                predictions[row_index as usize] += left_leaf_value;
            }
            for &row_index in &partition.right_row_indices {
                predictions[row_index as usize] += right_leaf_value;
            }

            split.left_stats = left_stats;
            split.right_stats = right_stats;

            stumps.push(TrainedStump {
                split,
                left_leaf_value,
                right_leaf_value,
            });
        }

        Ok(TrainedModel {
            baseline_prediction: fit_contract.baseline_prediction,
            feature_count: dataset.matrix.feature_count,
            stumps,
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
    fn trained_model_artifact_roundtrip_preserves_predictions() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let model = trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("iterative training succeeds");
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact parses");

        assert_eq!(model.feature_count, restored.feature_count);
        assert_eq!(model.stumps.len(), restored.stumps.len());

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }
}
