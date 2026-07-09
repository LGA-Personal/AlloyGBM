use alloygbm_core::{
    BinnedMatrix, DroConfig, FactorExposureMatrix, MISSING_BIN_U8, MorphConfig, MorphPrecomputed,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitSelectionOptions {
    pub l2_lambda: f32,
    pub l1_alpha: f32,
    pub min_child_hessian: f32,
    pub min_rows_per_leaf: usize,
    pub min_leaf_magnitude: f32,
    pub dro_config: Option<DroConfig>,
    /// Histogram index for the NaN/missing bin.
    /// For u8 bins: 255. For u16 bins: max_data_bin + 1 (dynamic).
    pub missing_bin_index: usize,
}

/// Metadata about a feature that uses native categorical splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalFeatureInfo {
    /// The feature index in the BinnedMatrix.
    pub feature_index: usize,
    /// Number of categories (valid bin IDs are 0..num_categories).
    pub num_categories: usize,
}

impl Default for SplitSelectionOptions {
    fn default() -> Self {
        Self {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_rows_per_leaf: 1,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: MISSING_BIN_U8 as usize,
        }
    }
}

/// Per-node context for split exposure penalties over scalar leaves.
///
/// This is passed separately from [`SplitSelectionOptions`] so the hot no-penalty
/// path can keep using small `Copy` options.
#[derive(Debug, Clone, Copy)]
pub struct FactorSplitContext<'a> {
    pub binned_matrix: &'a BinnedMatrix,
    pub exposures: &'a FactorExposureMatrix,
    pub row_indices: &'a [u32],
    pub factor_penalty: f32,
}

/// Per-round context for morph-gain split selection.
/// Passed to `BackendOps::best_split_morph` in addition to the standard options.
#[derive(Debug, Clone, Copy)]
pub struct MorphContext {
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub config: MorphConfig,
    /// Per-round constants (`tanh(iter/20)`, blend coefficients, warmup branch)
    /// hoisted out of the per-bin gain inner loop. Computed once via
    /// [`MorphPrecomputed::for_iteration`] when the context is built.
    pub precomputed: MorphPrecomputed,
}

/// Per-node context for piecewise-linear split-gain selection.
///
/// Passed to `BackendOps::best_split_linear` alongside `SplitSelectionOptions`.
/// Carries the regressor feature set and the PL-specific ridge regularisation.
#[derive(Debug, Clone)]
pub struct LinearContext {
    /// Indices of features used as linear regressors in this node's leaf model
    /// (length `d`, capped at `MAX_PL_REGRESSORS`).
    pub regressor_features: Vec<u32>,
    /// L2 regularisation added to the diagonal of `XᵀHX` before inversion.
    pub l2_lambda: f32,
}

impl LinearContext {
    /// Number of regressors (`d`).
    pub fn d(&self) -> usize {
        self.regressor_features.len()
    }
}
