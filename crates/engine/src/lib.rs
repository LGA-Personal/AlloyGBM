use alloygbm_core::{CoreError, TrainParams, validate_train_params};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidConfig(String),
    BackendUnavailable(String),
    NotImplemented(String),
    Core(CoreError),
}

impl Display for EngineError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
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
    fn build_histograms(&self) -> EngineResult<()>;
    fn best_split(&self) -> EngineResult<()>;
    fn apply_split(&self) -> EngineResult<()>;
    fn reduce_sums(&self) -> EngineResult<()>;
}

pub trait ObjectiveOps {
    fn compute_grad_hess(&self) -> EngineResult<()>;
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

    pub fn fit_stub<B: BackendOps, O: ObjectiveOps>(
        &self,
        _backend: &B,
        _objective: &O,
    ) -> EngineResult<()> {
        Err(EngineError::NotImplemented(
            "fit_stub is a placeholder in v0.0.1".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;
    struct MockObjective;

    impl BackendOps for MockBackend {
        fn build_histograms(&self) -> EngineResult<()> {
            Ok(())
        }

        fn best_split(&self) -> EngineResult<()> {
            Ok(())
        }

        fn apply_split(&self) -> EngineResult<()> {
            Ok(())
        }

        fn reduce_sums(&self) -> EngineResult<()> {
            Ok(())
        }
    }

    impl ObjectiveOps for MockObjective {
        fn compute_grad_hess(&self) -> EngineResult<()> {
            Ok(())
        }
    }

    #[test]
    fn trainer_fit_stub_returns_not_implemented() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer.fit_stub(&MockBackend, &MockObjective);
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
    }
}
