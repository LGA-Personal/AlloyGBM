use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, LinearHistogramBundle, LinearLeaf,
    NodeSlice, NodeStats, PartitionResult, SplitCandidate,
};

use crate::error::{EngineError, EngineResult};
use crate::split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, MorphContext, SplitSelectionOptions,
};

pub trait BackendOps {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle>;
    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>>;
    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        _options: SplitSelectionOptions,
        _feature_weights: &[f32],
        _categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        self.best_split(histograms)
    }
    fn best_split_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        if factor_context.is_some() {
            return Err(EngineError::ContractViolation(
                "factor split context is not supported by this backend".to_string(),
            ));
        }
        self.best_split_with_options(histograms, options, feature_weights, categorical_features)
    }
    /// Morph-mode split selection. Default implementation delegates to
    /// `best_split_with_options` (i.e. ignores morph context), so backends that
    /// don't implement morph fall back gracefully.
    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        _morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.best_split_with_options(histograms, options, feature_weights, categorical_features)
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
        if factor_context.is_some() {
            return Err(EngineError::ContractViolation(
                "factor split context is not supported by this backend".to_string(),
            ));
        }
        self.best_split_morph(
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
        )
    }
    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult>;
    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        let partition = self.apply_split(binned_matrix, node, split)?;
        let left_stats = self.reduce_sums(gradients, &partition.left_row_indices)?;
        let right_stats = self.reduce_sums(gradients, &partition.right_row_indices)?;
        Ok((partition, left_stats, right_stats))
    }
    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats>;

    /// Build piecewise-linear histogram statistics for a node.
    ///
    /// `regressor_features` lists the feature indices whose raw float values
    /// are used as regressors.  The returned bundle contains `(Xᵀg, XᵀHX)`
    /// statistics per bin, per split feature, needed for the PL gain criterion
    /// and leaf-weight solve.
    ///
    /// Default implementation returns `EngineError::NotImplemented`; backends
    /// that support PL Trees override this.
    #[allow(clippy::too_many_arguments)]
    fn build_linear_histograms(
        &self,
        _binned_matrix: &BinnedMatrix,
        _gradients: &[GradientPair],
        _node: &NodeSlice,
        _feature_tiles: &[FeatureTile],
        _regressor_features: &[u32],
        _raw_feature_values: &[f32],
        _row_count: usize,
        _feature_count: usize,
    ) -> EngineResult<LinearHistogramBundle> {
        Err(EngineError::NotImplemented(
            "build_linear_histograms not implemented for this backend".to_string(),
        ))
    }

    /// Find the best split point using the PL ridge-regression gain criterion.
    ///
    /// Scans the `LinearHistogramBundle` (one `LinearFeatureHistogram` per split
    /// feature) and returns the `SplitCandidate` with the highest PL gain:
    /// `gain = 0.5·(Xᵀg_L)ᵀ(XᵀHX_L + λI)⁻¹(Xᵀg_L)
    ///       + 0.5·(Xᵀg_R)ᵀ(XᵀHX_R + λI)⁻¹(Xᵀg_R)
    ///       − 0.5·(Xᵀg_P)ᵀ(XᵀHX_P + λI)⁻¹(Xᵀg_P)`
    ///
    /// Default implementation returns `EngineError::NotImplemented`.
    fn best_split_linear(
        &self,
        _linear_histograms: &LinearHistogramBundle,
        _options: SplitSelectionOptions,
        _feature_weights: &[f32],
        _categorical_features: &[CategoricalFeatureInfo],
        _ctx: &LinearContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        Err(EngineError::NotImplemented(
            "best_split_linear not implemented for this backend".to_string(),
        ))
    }

    /// Given a `LinearHistogramBundle` for a node and the winning split parameters,
    /// solve for the optimal `LinearLeaf` for each child.
    ///
    /// Returns `Some((left_leaf, right_leaf))` on success, or `None` if the feature
    /// is not found in the bundle or the matrix is singular (causing a fallback to
    /// scalar leaves).
    ///
    /// Default implementation returns `None`; backends override this.
    #[allow(clippy::too_many_arguments)]
    fn compute_linear_leaf_pair(
        &self,
        _linear_histograms: &LinearHistogramBundle,
        _feature_index: u32,
        _threshold_bin: usize,
        _default_left: bool,
        _missing_bin_index: usize,
        _learning_rate: f32,
        _l2_lambda: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        None
    }

    /// Solve linear leaves directly from the already-materialized child
    /// partitions for a chosen split. This avoids rebuilding a full
    /// `LinearHistogramBundle` when the standard scalar split criterion has
    /// already selected the split.
    #[allow(clippy::too_many_arguments)]
    fn compute_linear_leaf_pair_from_partitions(
        &self,
        _binned_matrix: &BinnedMatrix,
        _gradients: &[GradientPair],
        _raw_feature_values: &[f32],
        _feature_count: usize,
        _split_feature_index: u32,
        _threshold_bin: u16,
        _default_left: bool,
        _regressor_features: &[u32],
        _left_rows: &[u32],
        _right_rows: &[u32],
        _learning_rate: f32,
        _l2_lambda: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        None
    }
}

/// Callback invoked after each boosting round to evaluate a custom metric.
///
/// When provided alongside `early_stopping_rounds`, the custom metric value
/// drives early stopping *instead of* the built-in objective loss.
pub trait PerRoundMetricCallback {
    /// Evaluate the metric on `predictions` vs `targets`.
    ///
    /// For single-output models, `predictions` contains the raw model outputs
    /// (pre-sigmoid for binary, raw scores for regression/ranking).
    fn evaluate(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32>;

    /// Whether higher metric values are better (`true`) or lower is better (`false`).
    fn higher_is_better(&self) -> bool;

    /// The name of this metric (e.g. `"custom_rmse"`).
    fn metric_name(&self) -> &str;
}

pub trait ObjectiveOps {
    /// Canonical name for this objective (e.g. "squared_error", "binary_crossentropy").
    fn objective_name(&self) -> &str;

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

    /// Compute the objective loss for a set of predictions.
    /// This is used for monitoring convergence and early stopping.
    fn loss(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
    ) -> EngineResult<f32>;

    fn compute_gradients_into(
        &self,
        predictions: &[f32],
        targets: &[f32],
        sample_weights: Option<&[f32]>,
        buffer: &mut Vec<GradientPair>,
    ) -> EngineResult<()> {
        let gradients = self.compute_gradients(predictions, targets, sample_weights)?;
        buffer.clear();
        buffer.extend(gradients);
        Ok(())
    }

    /// Whether this objective requires `group_id` on the training dataset.
    fn requires_group_id(&self) -> bool {
        false
    }

    /// Return the quantile alpha if this is a quantile objective.
    fn quantile_alpha(&self) -> Option<f32> {
        None
    }

    /// Whether MSE-based leaf refinement is supported for this objective.
    fn supports_leaf_refinement(&self) -> bool {
        self.quantile_alpha().is_none()
    }

    /// Whether pre-target factor neutralization is valid for this objective.
    fn supports_pre_target_neutralization(&self) -> bool {
        false
    }
}
