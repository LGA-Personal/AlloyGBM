use alloygbm_core::Device;
use alloygbm_engine::{BackendOps, EngineError, EngineResult};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn device(&self) -> Device {
        Device::Cpu
    }
}

impl BackendOps for CpuBackend {
    fn build_histograms(&self) -> EngineResult<()> {
        Err(EngineError::NotImplemented(
            "CPU histogram kernel is not implemented in v0.0.1".to_string(),
        ))
    }

    fn best_split(&self) -> EngineResult<()> {
        Err(EngineError::NotImplemented(
            "CPU best split search is not implemented in v0.0.1".to_string(),
        ))
    }

    fn apply_split(&self) -> EngineResult<()> {
        Err(EngineError::NotImplemented(
            "CPU split application is not implemented in v0.0.1".to_string(),
        ))
    }

    fn reduce_sums(&self) -> EngineResult<()> {
        Err(EngineError::NotImplemented(
            "CPU reductions are not implemented in v0.0.1".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_ops_return_stub_errors() {
        let backend = CpuBackend;
        assert!(matches!(
            backend.build_histograms(),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.best_split(),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.apply_split(),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.reduce_sums(),
            Err(EngineError::NotImplemented(_))
        ));
    }

    #[test]
    fn backend_reports_cpu_device() {
        assert_eq!(CpuBackend.device(), Device::Cpu);
    }
}
