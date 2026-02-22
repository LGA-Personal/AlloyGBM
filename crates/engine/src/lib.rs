use alloygbm_core::{
    BinnedMatrix, CoreError, FeatureTile, GradientPair, HistogramBundle, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate, TrainParams, TrainingDataset, validate_train_params,
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
    ) -> EngineResult<Vec<GradientPair>> {
        validate_train_params(&self.params)?;
        validate_training_dataset(dataset)?;

        let baseline =
            objective.initial_prediction(&dataset.targets, dataset.sample_weights.as_deref())?;
        if !baseline.is_finite() {
            return Err(EngineError::ContractViolation(
                "objective returned non-finite initial prediction".to_string(),
            ));
        }

        let predictions = vec![baseline; dataset.row_count()];
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

        Ok(gradients)
    }

    pub fn fit_stub<B: BackendOps, O: ObjectiveOps>(
        &self,
        dataset: &TrainingDataset,
        _backend: &B,
        objective: &O,
    ) -> EngineResult<()> {
        let _ = self.validate_fit_contract(dataset, objective)?;
        Err(EngineError::NotImplemented(
            "fit_stub is a placeholder in v0.0.2".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;
    struct MockObjective;
    struct BadObjective;

    fn sample_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(2, 2, vec![0.2, 0.3, 0.5, 0.7])
                .expect("matrix is valid"),
            targets: vec![1.0, 0.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        }
    }

    impl BackendOps for MockBackend {
        fn build_histograms(
            &self,
            _binned_matrix: &BinnedMatrix,
            _gradients: &[GradientPair],
            _node: &NodeSlice,
            _feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            Ok(HistogramBundle {
                node_id: 0,
                feature_histograms: Vec::new(),
            })
        }

        fn best_split(
            &self,
            _histograms: &HistogramBundle,
        ) -> EngineResult<Option<SplitCandidate>> {
            Ok(None)
        }

        fn apply_split(
            &self,
            _binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            _split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            Ok(PartitionResult {
                left_row_indices: node.row_indices.clone(),
                right_row_indices: Vec::new(),
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

    impl ObjectiveOps for MockObjective {
        fn initial_prediction(
            &self,
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            let sum = targets.iter().sum::<f32>();
            Ok(sum / targets.len() as f32)
        }

        fn compute_gradients(
            &self,
            predictions: &[f32],
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<Vec<GradientPair>> {
            Ok(predictions
                .iter()
                .zip(targets)
                .map(|(prediction, target)| GradientPair {
                    grad: prediction - target,
                    hess: 1.0,
                })
                .collect())
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
    fn trainer_validates_fit_contract() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let gradients = trainer
            .validate_fit_contract(&sample_dataset(), &MockObjective)
            .expect("contract validation succeeds");
        assert_eq!(gradients.len(), 2);
    }

    #[test]
    fn trainer_rejects_gradient_length_mismatch() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer.validate_fit_contract(&sample_dataset(), &BadObjective);
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn trainer_fit_stub_returns_not_implemented_after_contract_checks() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer.fit_stub(&sample_dataset(), &MockBackend, &MockObjective);
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
    }
}
