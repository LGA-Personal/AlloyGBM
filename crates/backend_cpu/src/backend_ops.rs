use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, HistogramFeatureView,
    LinearFeatureHistogram, LinearFeatureScaler, LinearHistogramBundle, LinearLeaf, NodeSlice,
    NodeStats, PartitionResult, SplitCandidate,
};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, EngineError, EngineResult, FactorSplitContext,
    HistogramExecution, LinearContext, MorphContext, SplitSelectionOptions, SplitShortlist,
};
use rayon::prelude::*;

use crate::CpuBackend;
use crate::factor_split::validate_factor_split_context;
use crate::split_helpers::{apply_feature_weight, gain_materially_exceeds};
use crate::{NodeStatsAccumulator, pl, pl_histogram};

pub(crate) fn morph_uses_standard_gain_only(morph: &MorphContext) -> bool {
    morph.precomputed.in_warmup
        || (morph.precomputed.info_score_negligible && !morph.precomputed.balance_penalty)
}

pub(crate) fn morph_can_use_standard_scanner(
    morph: &MorphContext,
    options: &SplitSelectionOptions,
    has_factor_context: bool,
) -> bool {
    !has_factor_context && options.dro_config.is_none() && morph_uses_standard_gain_only(morph)
}

impl BackendOps for CpuBackend {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        self.build_histograms_with_execution(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            false,
            HistogramExecution::Parallel,
        )
    }

    fn build_histograms_with_grad_sq(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        include_grad_sq: bool,
    ) -> EngineResult<HistogramBundle> {
        self.build_histograms_with_execution(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            include_grad_sq,
            HistogramExecution::Parallel,
        )
    }

    fn build_histograms_with_execution(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        include_grad_sq: bool,
        execution: HistogramExecution,
    ) -> EngineResult<HistogramBundle> {
        let selected_feature_count = feature_tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum();
        let parallel_tiles = execution == HistogramExecution::Parallel
            && Self::should_parallelize_tiles(
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
            include_grad_sq,
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

    fn shortlist_standard_splits(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        max_numeric_features: usize,
    ) -> EngineResult<SplitShortlist> {
        let find_best = |fh: HistogramFeatureView<'_>| -> Option<SplitCandidate> {
            let feature_index = fh.feature_index() as usize;
            if let Some(info) = categorical_features
                .iter()
                .find(|info| info.feature_index == feature_index)
            {
                Self::best_split_for_categorical_feature(
                    fh,
                    histograms.node_id,
                    options,
                    info.num_categories,
                    None,
                )
            } else {
                Self::best_split_for_feature(fh, histograms.node_id, options, None)
            }
        };

        let per_feature: Vec<Option<SplitCandidate>> =
            if histograms.feature_count() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
                (0..histograms.feature_count())
                    .into_par_iter()
                    .map(|index| find_best(histograms.feature(index).expect("bounded feature")))
                    .collect()
            } else {
                histograms.features().map(find_best).collect()
            };

        let best_overall = per_feature
            .iter()
            .flatten()
            .cloned()
            .reduce(|current, candidate| {
                if gain_materially_exceeds(
                    apply_feature_weight(&candidate, feature_weights),
                    apply_feature_weight(&current, feature_weights),
                ) {
                    candidate
                } else {
                    current
                }
            });

        let mut remaining: Vec<SplitCandidate> = per_feature
            .into_iter()
            .flatten()
            .filter(|candidate| !candidate.is_categorical)
            .collect();
        let target = max_numeric_features.min(remaining.len());
        let mut numeric_candidates = Vec::with_capacity(target);
        for _ in 0..target {
            let mut best_index = 0;
            for index in 1..remaining.len() {
                if gain_materially_exceeds(
                    apply_feature_weight(&remaining[index], feature_weights),
                    apply_feature_weight(&remaining[best_index], feature_weights),
                ) {
                    best_index = index;
                }
            }
            numeric_candidates.push(remaining.remove(best_index));
        }

        Ok(SplitShortlist {
            best_overall,
            numeric_candidates,
        })
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
        if morph_can_use_standard_scanner(morph, &options, factor_context.is_some()) {
            return Ok(Self::best_split_with_options_internal(
                histograms,
                options,
                feature_weights,
                categorical_features,
                None,
            ));
        }
        let find_best = |fh: HistogramFeatureView<'_>| -> Option<SplitCandidate> {
            let fi = fh.feature_index() as usize;
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

        let result = if histograms.feature_count() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
            (0..histograms.feature_count())
                .into_par_iter()
                .filter_map(|index| find_best(histograms.feature(index).expect("bounded feature")))
                .reduce_with(|a, b| {
                    if gain_materially_exceeds(
                        apply_feature_weight(&b, feature_weights),
                        apply_feature_weight(&a, feature_weights),
                    ) {
                        b
                    } else {
                        a
                    }
                })
        } else {
            histograms.features().filter_map(find_best).reduce(|a, b| {
                if gain_materially_exceeds(
                    apply_feature_weight(&b, feature_weights),
                    apply_feature_weight(&a, feature_weights),
                ) {
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
        let lookup = Self::split_row_lookup(binned_matrix, None, node, split)?;

        let mut left_row_indices = Vec::new();
        let mut right_row_indices = Vec::new();
        for &row_index in &node.row_indices {
            if lookup.goes_left(binned_matrix, row_index, split) {
                left_row_indices.push(row_index);
            } else {
                right_row_indices.push(row_index);
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
        let lookup = Self::split_row_lookup(binned_matrix, Some(gradients), node, split)?;

        const PARALLEL_PARTITION_THRESHOLD: usize = 50_000;

        if node.row_indices.len() >= PARALLEL_PARTITION_THRESHOLD {
            return Self::apply_split_with_stats_parallel_with_lookup(
                binned_matrix,
                gradients,
                node,
                split,
                lookup,
            );
        }

        let mut left_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut right_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut left_stats = NodeStatsAccumulator::default();
        let mut right_stats = NodeStatsAccumulator::default();

        for &row_index_u32 in &node.row_indices {
            let gradient = gradients[row_index_u32 as usize];
            if lookup.goes_left(binned_matrix, row_index_u32, split) {
                left_row_indices.push(row_index_u32);
                left_stats.add(gradient);
            } else {
                right_row_indices.push(row_index_u32);
                right_stats.add(gradient);
            }
        }

        let partition = PartitionResult {
            left_row_indices,
            right_row_indices,
        };
        let left_count = partition.left_row_indices.len();
        let right_count = partition.right_row_indices.len();
        Ok((
            partition,
            left_stats.into_node_stats(left_count),
            right_stats.into_node_stats(right_count),
        ))
    }

    fn apply_split_owned_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        let lookup = Self::split_row_lookup(binned_matrix, Some(gradients), &node, split)?;

        const PARALLEL_PARTITION_THRESHOLD: usize = 50_000;

        if node.row_indices.len() >= PARALLEL_PARTITION_THRESHOLD {
            return Self::apply_split_owned_with_stats_parallel(
                binned_matrix,
                gradients,
                node,
                split,
                lookup,
            );
        }

        let mut left_stats = NodeStatsAccumulator::default();
        let mut right_stats = NodeStatsAccumulator::default();
        let left_capacity = node.row_indices.len() / 2;
        let (left_row_indices, right_row_indices) =
            Self::stable_partition_owned_rows(node.row_indices, left_capacity, |row| {
                let gradient = gradients[row as usize];
                if lookup.goes_left(binned_matrix, row, split) {
                    left_stats.add(gradient);
                    true
                } else {
                    right_stats.add(gradient);
                    false
                }
            });

        let partition = PartitionResult {
            left_row_indices,
            right_row_indices,
        };
        let left_count = partition.left_row_indices.len();
        let right_count = partition.right_row_indices.len();
        Ok((
            partition,
            left_stats.into_node_stats(left_count),
            right_stats.into_node_stats(right_count),
        ))
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
        feature_scaler: &LinearFeatureScaler,
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
            feature_scaler,
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
                    if gain_materially_exceeds(
                        apply_feature_weight(&b, feature_weights),
                        apply_feature_weight(&a, feature_weights),
                    ) {
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
                    if gain_materially_exceeds(
                        apply_feature_weight(&b, feature_weights),
                        apply_feature_weight(&a, feature_weights),
                    ) {
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
        feature_scaler: &LinearFeatureScaler,
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
            pl::LinearLeafSolveParams {
                grad_sum: l_gs,
                hess_sum: l_hs,
                learning_rate,
                l2_lambda,
            },
            regressor_features,
            feature_scaler,
        );
        let right_leaf = pl::solve_pl_leaf(
            &r_xtg,
            &r_xthx,
            pl::LinearLeafSolveParams {
                grad_sum: r_gs,
                hess_sum: r_hs,
                learning_rate,
                l2_lambda,
            },
            regressor_features,
            feature_scaler,
        );

        Some((left_leaf, right_leaf))
    }

    fn compute_linear_leaf_pair_from_partitions(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        raw_feature_values: &[f32],
        feature_count: usize,
        split_feature_index: u32,
        threshold_bin: u16,
        default_left: bool,
        regressor_features: &[u32],
        feature_scaler: &LinearFeatureScaler,
        left_rows: &[u32],
        right_rows: &[u32],
        learning_rate: f32,
        l2_lambda: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        pl::solve_pl_leaf_pair_from_partitions(
            binned_matrix,
            gradients,
            raw_feature_values,
            feature_count,
            split_feature_index,
            threshold_bin,
            default_left,
            regressor_features,
            feature_scaler,
            left_rows,
            right_rows,
            learning_rate,
            l2_lambda,
        )
    }
}
