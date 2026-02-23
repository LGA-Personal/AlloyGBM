use alloygbm_core::{
    BinnedMatrix, Device, FeatureHistogram, FeatureTile, GradientPair, HistogramBin,
    HistogramBundle, NodeSlice, NodeStats, PartitionResult, SplitCandidate,
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
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        if gradients.len() != binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "gradients length {} does not match row_count {}",
                gradients.len(),
                binned_matrix.row_count
            )));
        }
        if feature_tiles.is_empty() {
            return Err(EngineError::ContractViolation(
                "feature_tiles cannot be empty".to_string(),
            ));
        }
        node.validate_bounds(binned_matrix.row_count)?;

        let mut feature_histograms = Vec::new();
        for tile in feature_tiles {
            if tile.end_feature as usize > binned_matrix.feature_count {
                return Err(EngineError::ContractViolation(format!(
                    "feature tile end {} exceeds feature_count {}",
                    tile.end_feature, binned_matrix.feature_count
                )));
            }

            for feature_index in tile.start_feature..tile.end_feature {
                let mut bins = vec![
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        count: 0,
                    };
                    binned_matrix.max_bin as usize + 1
                ];

                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let cell_index =
                        row_index * binned_matrix.feature_count + feature_index as usize;
                    let bin_index = binned_matrix.bins[cell_index] as usize;
                    let gradient = gradients[row_index];
                    let target_bin = &mut bins[bin_index];
                    target_bin.grad_sum += gradient.grad;
                    target_bin.hess_sum += gradient.hess;
                    target_bin.count += 1;
                }

                feature_histograms.push(FeatureHistogram {
                    feature_index,
                    bins,
                });
            }
        }

        Ok(HistogramBundle {
            node_id: node.node_id,
            feature_histograms,
        })
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        const EPSILON: f32 = 1e-6;

        for feature_histogram in &histograms.feature_histograms {
            if feature_histogram.bins.len() < 2 {
                continue;
            }

            let mut total_grad = 0.0_f32;
            let mut total_hess = 0.0_f32;
            let mut total_count = 0_u32;
            for bin in &feature_histogram.bins {
                total_grad += bin.grad_sum;
                total_hess += bin.hess_sum;
                total_count += bin.count;
            }

            let mut left_grad = 0.0_f32;
            let mut left_hess = 0.0_f32;
            let mut left_count = 0_u32;

            for (threshold_bin, bin) in feature_histogram
                .bins
                .iter()
                .enumerate()
                .take(feature_histogram.bins.len() - 1)
            {
                left_grad += bin.grad_sum;
                left_hess += bin.hess_sum;
                left_count += bin.count;

                let right_grad = total_grad - left_grad;
                let right_hess = total_hess - left_hess;
                let right_count = total_count.saturating_sub(left_count);

                if left_count == 0 || right_count == 0 || left_hess <= 0.0 || right_hess <= 0.0 {
                    continue;
                }

                let gain = (left_grad * left_grad) / (left_hess + EPSILON)
                    + (right_grad * right_grad) / (right_hess + EPSILON)
                    - (total_grad * total_grad) / (total_hess + EPSILON);

                if gain > best_gain {
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id: histograms.node_id,
                        feature_index: feature_histogram.feature_index,
                        threshold_bin: threshold_bin as u16,
                        gain,
                        left_stats: NodeStats {
                            grad_sum: left_grad,
                            hess_sum: left_hess,
                            row_count: left_count,
                        },
                        right_stats: NodeStats {
                            grad_sum: right_grad,
                            hess_sum: right_hess,
                            row_count: right_count,
                        },
                    });
                }
            }
        }

        Ok(best_candidate)
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        node.validate_bounds(binned_matrix.row_count)?;
        if split.feature_index as usize >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                split.feature_index, binned_matrix.feature_count
            )));
        }

        let mut left_row_indices = Vec::new();
        let mut right_row_indices = Vec::new();
        for &row_index in &node.row_indices {
            let row_index = row_index as usize;
            let cell_index = row_index * binned_matrix.feature_count + split.feature_index as usize;
            let bin = binned_matrix.bins[cell_index];
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
        if row_indices.is_empty() {
            return Err(EngineError::ContractViolation(
                "row_indices cannot be empty".to_string(),
            ));
        }

        let mut grad_sum = 0.0_f32;
        let mut hess_sum = 0.0_f32;
        for &row_index in row_indices {
            let gradient = gradients.get(row_index as usize).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "row index {row_index} is out of bounds for gradients length {}",
                    gradients.len()
                ))
            })?;
            grad_sum += gradient.grad;
            hess_sum += gradient.hess;
        }

        Ok(NodeStats {
            grad_sum,
            hess_sum,
            row_count: row_indices.len() as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::FeatureTile;

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

    fn sample_gradients() -> Vec<GradientPair> {
        vec![
            GradientPair {
                grad: 2.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -2.0,
                hess: 1.0,
            },
        ]
    }

    fn sample_node() -> NodeSlice {
        NodeSlice::new(0, vec![0, 1, 2, 3]).expect("node is valid")
    }

    #[test]
    fn build_histograms_aggregates_bins() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        assert_eq!(histograms.feature_histograms.len(), 2);
        let feature0 = &histograms.feature_histograms[0];
        assert_eq!(feature0.feature_index, 0);
        assert_eq!(feature0.bins.len(), 4);
        assert_eq!(feature0.bins[0].count, 1);
        assert_eq!(feature0.bins[1].count, 1);
        assert_eq!(feature0.bins[2].count, 1);
        assert_eq!(feature0.bins[3].count, 1);
        assert!((feature0.bins[0].grad_sum - 2.0).abs() < 1e-6);
        assert!((feature0.bins[3].grad_sum + 2.0).abs() < 1e-6);
    }

    #[test]
    fn best_split_returns_high_gain_candidate() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");
        let split = backend
            .best_split(&histograms)
            .expect("split search should succeed")
            .expect("split should exist");

        assert_eq!(split.feature_index, 0);
        assert_eq!(split.threshold_bin, 1);
        assert!(split.gain > 0.0);
        assert_eq!(split.left_stats.row_count, 2);
        assert_eq!(split.right_stats.row_count, 2);
    }

    #[test]
    fn apply_split_partitions_rows() {
        let backend = CpuBackend;
        let split = SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin: 1,
            gain: 1.0,
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
        };
        let partition = backend
            .apply_split(&sample_binned_matrix(), &sample_node(), &split)
            .expect("partition should succeed");

        assert_eq!(partition.left_row_indices, vec![0, 1]);
        assert_eq!(partition.right_row_indices, vec![2, 3]);
    }

    #[test]
    fn reduce_sums_aggregates_requested_rows() {
        let backend = CpuBackend;
        let stats = backend
            .reduce_sums(&sample_gradients(), &[0, 3])
            .expect("reductions should succeed");
        assert_eq!(stats.row_count, 2);
        assert!(stats.grad_sum.abs() < 1e-6);
        assert!((stats.hess_sum - 2.0).abs() < 1e-6);
    }

    #[test]
    fn backend_reports_cpu_device() {
        assert_eq!(CpuBackend.device(), Device::Cpu);
    }
}
