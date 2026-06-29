use alloygbm_core::{
    BinnedMatrix, FeatureHistogram, FeatureTile, GradientPair, HistogramBundle,
    LinearFeatureHistogram, LinearHistogramBundle, LinearLeaf, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate,
};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, EngineError, EngineResult, FactorSplitContext,
    LinearContext, MorphContext, SplitSelectionOptions,
};
use rayon::prelude::*;

use crate::CpuBackend;
use crate::factor_split::validate_factor_split_context;
use crate::split_helpers::{apply_feature_weight, goes_left_for_split};
use crate::{pl, pl_histogram};

impl BackendOps for CpuBackend {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        let selected_feature_count = feature_tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum();
        let parallel_tiles = Self::should_parallelize_tiles(
            feature_tiles.len(),
            node.row_indices.len(),
            selected_feature_count,
        );
        Self::build_histograms_internal(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            parallel_tiles,
        )
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        Ok(Self::best_split_with_options_internal(
            histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
            None,
        ))
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        Ok(Self::best_split_with_options_internal(
            histograms,
            options,
            feature_weights,
            categorical_features,
            None,
        ))
    }

    fn best_split_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        if let Some(ctx) = factor_context {
            validate_factor_split_context(ctx)?;
        }
        Ok(Self::best_split_with_options_internal(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
        ))
    }

    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.best_split_morph_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
            None,
        )
    }

    fn best_split_morph_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        if let Some(ctx) = factor_context {
            validate_factor_split_context(ctx)?;
        }
        let find_best = |fh: &FeatureHistogram| -> Option<SplitCandidate> {
            let fi = fh.feature_index as usize;
            if let Some(cat_info) = categorical_features.iter().find(|c| c.feature_index == fi) {
                Self::best_split_morph_categorical_feature(
                    fh,
                    histograms.node_id,
                    &options,
                    cat_info.num_categories,
                    morph,
                    factor_context,
                )
            } else {
                Self::best_split_morph_numeric_feature(
                    fh,
                    histograms.node_id,
                    &options,
                    morph,
                    factor_context,
                )
            }
        };

        let result =
            if histograms.feature_histograms.len() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
                histograms
                    .feature_histograms
                    .par_iter()
                    .filter_map(find_best)
                    .reduce_with(|a, b| {
                        if apply_feature_weight(&b, feature_weights)
                            > apply_feature_weight(&a, feature_weights)
                        {
                            b
                        } else {
                            a
                        }
                    })
            } else {
                histograms
                    .feature_histograms
                    .iter()
                    .filter_map(find_best)
                    .reduce(|a, b| {
                        if apply_feature_weight(&b, feature_weights)
                            > apply_feature_weight(&a, feature_weights)
                        {
                            b
                        } else {
                            a
                        }
                    })
            };

        Ok(result)
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
        let feature_index = split.feature_index as usize;
        let use_col_major = binned_matrix.has_col_major();
        let col_base = feature_index * binned_matrix.row_count;
        let missing = binned_matrix.missing_bin();
        for &row_index in &node.row_indices {
            let row_index = row_index as usize;
            let bin_val = if use_col_major {
                binned_matrix.col_bin(col_base + row_index)
            } else {
                let cell_index = row_index * binned_matrix.feature_count + feature_index;
                binned_matrix.row_bin(cell_index)
            };
            if goes_left_for_split(bin_val, missing, split) {
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

    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        node.validate_bounds(binned_matrix.row_count)?;
        if gradients.len() != binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "gradients length {} does not match row_count {}",
                gradients.len(),
                binned_matrix.row_count
            )));
        }
        if split.feature_index as usize >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                split.feature_index, binned_matrix.feature_count
            )));
        }

        const PARALLEL_PARTITION_THRESHOLD: usize = 50_000;

        if node.row_indices.len() >= PARALLEL_PARTITION_THRESHOLD {
            return Self::apply_split_with_stats_parallel(binned_matrix, gradients, node, split);
        }

        let mut left_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut right_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut left_grad_sum = 0.0_f32;
        let mut left_hess_sum = 0.0_f32;
        let mut left_grad_sq_sum = 0.0_f32;
        let mut right_grad_sum = 0.0_f32;
        let mut right_hess_sum = 0.0_f32;
        let mut right_grad_sq_sum = 0.0_f32;

        let feature_index = split.feature_index as usize;
        let use_col_major = binned_matrix.has_col_major();
        let col_base = feature_index * binned_matrix.row_count;
        let missing = binned_matrix.missing_bin();
        for &row_index_u32 in &node.row_indices {
            let row_index = row_index_u32 as usize;
            let bin_val = if use_col_major {
                binned_matrix.col_bin(col_base + row_index)
            } else {
                let cell_index = row_index * binned_matrix.feature_count + feature_index;
                binned_matrix.row_bin(cell_index)
            };
            let gradient = gradients[row_index];
            if goes_left_for_split(bin_val, missing, split) {
                left_row_indices.push(row_index_u32);
                left_grad_sum += gradient.grad;
                left_hess_sum += gradient.hess;
                left_grad_sq_sum += gradient.grad * gradient.grad;
            } else {
                right_row_indices.push(row_index_u32);
                right_grad_sum += gradient.grad;
                right_hess_sum += gradient.hess;
                right_grad_sq_sum += gradient.grad * gradient.grad;
            }
        }

        let partition = PartitionResult {
            left_row_indices,
            right_row_indices,
        };
        let left_stats = NodeStats {
            grad_sum: left_grad_sum,
            hess_sum: left_hess_sum,
            grad_sq_sum: left_grad_sq_sum,
            row_count: partition.left_row_indices.len() as u32,
        };
        let right_stats = NodeStats {
            grad_sum: right_grad_sum,
            hess_sum: right_hess_sum,
            grad_sq_sum: right_grad_sq_sum,
            row_count: partition.right_row_indices.len() as u32,
        };

        Ok((partition, left_stats, right_stats))
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
        let mut grad_sq_sum = 0.0_f32;
        for &row_index in row_indices {
            let gradient = gradients.get(row_index as usize).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "row index {row_index} is out of bounds for gradients length {}",
                    gradients.len()
                ))
            })?;
            grad_sum += gradient.grad;
            hess_sum += gradient.hess;
            grad_sq_sum += gradient.grad * gradient.grad;
        }

        Ok(NodeStats {
            grad_sum,
            hess_sum,
            grad_sq_sum,
            row_count: row_indices.len() as u32,
        })
    }

    fn build_linear_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        regressor_features: &[u32],
        raw_feature_values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> EngineResult<LinearHistogramBundle> {
        pl_histogram::build_linear_histograms_cpu(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            regressor_features,
            raw_feature_values,
            row_count,
            feature_count,
        )
    }

    fn best_split_linear(
        &self,
        linear_histograms: &LinearHistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        _categorical_features: &[CategoricalFeatureInfo],
        ctx: &LinearContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        let node_id = linear_histograms.node_id;
        let find_best = |linear_fh: &LinearFeatureHistogram| -> Option<SplitCandidate> {
            pl::best_split_linear_for_feature(linear_fh, node_id, options, ctx)
        };

        let result = if linear_histograms.feature_histograms.len()
            >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD
        {
            linear_histograms
                .feature_histograms
                .par_iter()
                .filter_map(find_best)
                .reduce_with(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        } else {
            linear_histograms
                .feature_histograms
                .iter()
                .filter_map(find_best)
                .reduce(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        };

        Ok(result)
    }

    fn compute_linear_leaf_pair(
        &self,
        linear_histograms: &LinearHistogramBundle,
        feature_index: u32,
        threshold_bin: usize,
        default_left: bool,
        missing_bin_index: usize,
        learning_rate: f32,
        l2_lambda: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        let d = linear_histograms.num_regressors;
        if d == 0 {
            return None;
        }
        let lin_fh = linear_histograms
            .feature_histograms
            .iter()
            .find(|fh| fh.feature_index == feature_index)?;

        let (l_xtg, l_xthx, l_gs, l_hs, r_xtg, r_xthx, r_gs, r_hs) =
            pl::leaf_linear_stats_for_split(lin_fh, threshold_bin, missing_bin_index, default_left);

        let regressor_features = &linear_histograms.regressor_features;
        let left_leaf = pl::solve_pl_leaf(
            &l_xtg,
            &l_xthx,
            l_gs,
            l_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        );
        let right_leaf = pl::solve_pl_leaf(
            &r_xtg,
            &r_xthx,
            r_gs,
            r_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        );

        Some((left_leaf, right_leaf))
    }

    fn compute_linear_leaf_pair_from_partitions(
        &self,
        gradients: &[GradientPair],
        raw_feature_values: &[f32],
        feature_count: usize,
        regressor_features: &[u32],
        left_rows: &[u32],
        right_rows: &[u32],
        learning_rate: f32,
        l2_lambda: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        pl::solve_pl_leaf_pair_from_partitions(
            gradients,
            raw_feature_values,
            feature_count,
            regressor_features,
            left_rows,
            right_rows,
            learning_rate,
            l2_lambda,
        )
    }
}
