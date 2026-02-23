use alloygbm_core::{
    BinnedMatrix, CoreError, FeatureTile, GradientPair, HistogramBundle, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate, TrainParams, TrainingDataset, validate_binned_matrix,
    validate_train_params, validate_training_dataset,
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

        if gradients.len() != dataset.row_count() {
            return Err(EngineError::ContractViolation(format!(
                "objective returned {} gradients for row_count {}",
                gradients.len(),
                dataset.row_count()
            )));
        }
        for gradient in &gradients {
            if !gradient.grad.is_finite() || !gradient.hess.is_finite() || gradient.hess <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "objective produced invalid gradient/hessian values".to_string(),
                ));
            }
        }

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
            if partition.left_row_indices.is_empty() || partition.right_row_indices.is_empty() {
                return Err(EngineError::ContractViolation(
                    "split partition produced empty branch".to_string(),
                ));
            }
            if partition.left_row_indices.len() + partition.right_row_indices.len()
                != dataset.row_count()
            {
                return Err(EngineError::ContractViolation(
                    "split partition does not cover all rows".to_string(),
                ));
            }
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
                    0.1, 0.0, //
                    0.2, 0.0, //
                    0.8, 1.0, //
                    0.9, 1.0, //
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
                    grad_sum: 3.0,
                    hess_sum: 2.0,
                    row_count: 2,
                },
                right_stats: NodeStats {
                    grad_sum: -3.0,
                    hess_sum: 2.0,
                    row_count: 2,
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
}
