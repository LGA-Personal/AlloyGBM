use alloygbm_core::{
    BinnedMatrix, Device, FeatureTile, GradientPair, HistogramBundle, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate,
};
use alloygbm_engine::{BackendOps, EngineError, EngineResult};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn device(&self) -> Device {
        Device::Cpu
    }
}

impl BackendOps for CpuBackend {
    fn build_histograms(
        &self,
        _binned_matrix: &BinnedMatrix,
        _gradients: &[GradientPair],
        _node: &NodeSlice,
        _feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        Err(EngineError::NotImplemented(
            "CPU histogram kernel is not implemented in v0.0.2".to_string(),
        ))
    }

    fn best_split(&self, _histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        Err(EngineError::NotImplemented(
            "CPU best split search is not implemented in v0.0.2".to_string(),
        ))
    }

    fn apply_split(
        &self,
        _binned_matrix: &BinnedMatrix,
        _node: &NodeSlice,
        _split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        Err(EngineError::NotImplemented(
            "CPU split application is not implemented in v0.0.2".to_string(),
        ))
    }

    fn reduce_sums(
        &self,
        _gradients: &[GradientPair],
        _row_indices: &[u32],
    ) -> EngineResult<NodeStats> {
        Err(EngineError::NotImplemented(
            "CPU reductions are not implemented in v0.0.2".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{FeatureHistogram, HistogramBin};

    fn sample_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(2, 2, 15, vec![1, 2, 3, 4]).expect("binned matrix is valid")
    }

    fn sample_gradients() -> Vec<GradientPair> {
        vec![
            GradientPair {
                grad: 0.1,
                hess: 1.0,
            },
            GradientPair {
                grad: -0.2,
                hess: 1.0,
            },
        ]
    }

    fn sample_node() -> NodeSlice {
        NodeSlice::new(0, vec![0, 1]).expect("node is valid")
    }

    #[test]
    fn backend_ops_return_stub_errors() {
        let backend = CpuBackend;
        assert!(matches!(
            backend.build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            ),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.best_split(&HistogramBundle {
                node_id: 0,
                feature_histograms: vec![FeatureHistogram {
                    feature_index: 0,
                    bins: vec![HistogramBin {
                        grad_sum: 0.1,
                        hess_sum: 1.0,
                        count: 1,
                    }],
                }],
            }),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.apply_split(
                &sample_binned_matrix(),
                &sample_node(),
                &SplitCandidate {
                    node_id: 0,
                    feature_index: 0,
                    threshold_bin: 1,
                    gain: 0.2,
                    left_stats: NodeStats {
                        grad_sum: 0.1,
                        hess_sum: 1.0,
                        row_count: 1,
                    },
                    right_stats: NodeStats {
                        grad_sum: -0.1,
                        hess_sum: 1.0,
                        row_count: 1,
                    },
                },
            ),
            Err(EngineError::NotImplemented(_))
        ));
        assert!(matches!(
            backend.reduce_sums(&sample_gradients(), &[0, 1]),
            Err(EngineError::NotImplemented(_))
        ));
    }

    #[test]
    fn backend_reports_cpu_device() {
        assert_eq!(CpuBackend.device(), Device::Cpu);
    }
}
