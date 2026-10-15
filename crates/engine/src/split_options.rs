use alloygbm_core::{
    BinnedMatrix, DroConfig, FactorExposureMatrix, LinearLeaf, MISSING_BIN_U8, MorphConfig,
    MorphPrecomputed, SplitCandidate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct SplitShortlist {
    pub best_overall: Option<SplitCandidate>,
    pub numeric_candidates: Vec<SplitCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedLinearSplit {
    pub split: SplitCandidate,
    pub left_leaf: LinearLeaf,
    pub right_leaf: LinearLeaf,
}

/// Random noise on split selection (CatBoost's `random_strength`).
///
/// When choosing a node's split, each feature's best candidate competes on
/// `weighted_gain * (1 + strength * z)`, where `z` is a deterministic,
/// roughly standard-normal draw keyed by the seed, the node and the feature.
/// Features whose true gains are close then win in turn across trees instead
/// of one feature always edging out the others, which decorrelates the
/// trees much like row and column sampling. The chosen candidate keeps its
/// true gain, so `min_split_gain` and leaf values are unaffected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitNoise {
    pub strength: f32,
    pub seed: u64,
}

impl SplitNoise {
    /// `Some` only for a positive, finite strength.
    pub fn new(strength: f32, seed: u64) -> Option<Self> {
        (strength.is_finite() && strength > 0.0).then_some(Self { strength, seed })
    }

    /// Non-negative multiplier for `feature_index`'s gain at `node_id`.
    pub fn gain_multiplier(&self, node_id: u32, feature_index: u32) -> f32 {
        let mut state = self.seed
            ^ u64::from(node_id).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(feature_index).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        // Irwin-Hall: the sum of four U(-0.5, 0.5) draws has variance 1/3.
        let mut sum = 0.0_f32;
        for _ in 0..4 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            sum += (z >> 40) as f32 / (1_u64 << 24) as f32 - 0.5;
        }
        (1.0 + self.strength * sum * 3.0_f32.sqrt()).max(0.0)
    }
}

/// Return the gain used to compare candidates across features without
/// changing the unweighted gain persisted in the candidate.
pub fn feature_weighted_gain(candidate: &SplitCandidate, feature_weights: &[f32]) -> f32 {
    feature_weights
        .get(candidate.feature_index as usize)
        .map_or(candidate.gain, |weight| candidate.gain * weight)
}

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

impl SplitSelectionOptions {
    pub fn requires_grad_sq(self) -> bool {
        self.dro_config.is_some_and(|config| config.radius > 0.0)
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
    /// Per-round horizon-scaled blend coefficients and warmup branch
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
    /// Magnitude ceiling for a PL leaf's output, mirroring the clamp applied
    /// to scalar leaves. A solved leaf whose bounded output would exceed this
    /// is rejected and the split falls back to scalar leaves. Use
    /// `f32::INFINITY` to disable.
    pub max_abs_leaf_value: f32,
}

impl LinearContext {
    /// Number of regressors (`d`).
    pub fn d(&self) -> usize {
        self.regressor_features.len()
    }
}
