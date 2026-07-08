use crate::histogram::NodeStats;

/// A linear leaf model:
/// `f_s(x) = intercept + Σ_j weights[j] * z_j(x[regressor_features[j]])`.
///
/// `intercept` holds the standard Newton-Raphson scalar (so constant-only behaviour
/// degrades gracefully when `weights` is all-zero).  `weights` contains the `d` linear
/// correction coefficients solved via `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeaf {
    /// Constant offset (standard scalar leaf value, learning-rate scaled).
    pub intercept: f32,
    /// Linear correction weights `α_j` for each regressor (length `d`).
    pub weights: Vec<f32>,
    /// Feature indices of the regressors (length `d`, matches `weights`).
    pub regressor_features: Vec<u32>,
    /// Per-regressor training-set means used to standardize raw feature values.
    pub feature_means: Vec<f32>,
    /// Per-regressor inverse standard deviations used to standardize raw feature values.
    pub feature_inv_stds: Vec<f32>,
}

impl LinearLeaf {
    pub fn identity_scaled(
        intercept: f32,
        weights: Vec<f32>,
        regressor_features: Vec<u32>,
    ) -> Self {
        let d = weights.len();
        Self {
            intercept,
            weights,
            regressor_features,
            feature_means: vec![0.0; d],
            feature_inv_stds: vec![1.0; d],
        }
    }

    pub fn scaled(
        intercept: f32,
        weights: Vec<f32>,
        regressor_features: Vec<u32>,
        feature_means: Vec<f32>,
        feature_inv_stds: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(weights.len(), regressor_features.len());
        debug_assert_eq!(weights.len(), feature_means.len());
        debug_assert_eq!(weights.len(), feature_inv_stds.len());
        Self {
            intercept,
            weights,
            regressor_features,
            feature_means,
            feature_inv_stds,
        }
    }

    #[inline]
    pub fn slot_value(&self, slot: usize, raw_value: f32) -> f32 {
        if !raw_value.is_finite() {
            return 0.0;
        }
        let mean = self.feature_means.get(slot).copied().unwrap_or(0.0);
        let inv_std = self.feature_inv_stds.get(slot).copied().unwrap_or(1.0);
        (raw_value - mean) * inv_std
    }

    /// Evaluate the leaf model for one row of raw (float) feature data.
    ///
    /// `raw_features` is the full flat row-major feature matrix; `row_offset` is
    /// `row_index * feature_count`.
    ///
    /// **NaN policy (v0.9.0):** NaN feature values contribute 0.0 to the
    /// linear sum rather than propagating through `w · NaN = NaN`. This
    /// prevents NaN-poisoning of the final prediction when the predictor
    /// routes through `default_left` on a missing feature — the
    /// constant-leaf path treats missing values as "absent contribution"
    /// (the path's leaf eval is independent of the feature), and the
    /// PL-leaf path now does the same for any regressor feature that
    /// happens to be NaN on that row. See Limitation 4 in
    /// `docs/limitations.md` (resolved in v0.9.0).
    #[inline]
    pub fn eval(&self, raw_features: &[f32], row_offset: usize) -> f32 {
        let mut val = self.intercept;
        for (slot, (w, &feat)) in self
            .weights
            .iter()
            .zip(self.regressor_features.iter())
            .enumerate()
        {
            let idx = row_offset + feat as usize;
            if idx < raw_features.len() {
                val += w * self.slot_value(slot, raw_features[idx]);
            }
        }
        val
    }
}

/// The value stored at a trained leaf node.
///
/// * `Scalar(f32)` — constant leaf; identical to the pre-PL-Trees behaviour.
/// * `Linear(LinearLeaf)` — piecewise-linear leaf with `d` regressor features.
#[derive(Debug, Clone, PartialEq)]
pub enum LeafValue {
    Scalar(f32),
    Linear(LinearLeaf),
}

impl LeafValue {
    /// Extract the scalar representation of this leaf value.
    ///
    /// For `Scalar(v)` this is exact.  For `Linear`, this returns the intercept
    /// (the best constant approximation), used in places that do not yet support
    /// full row-level linear evaluation (Phase-4 code will handle those properly).
    #[inline]
    pub fn as_scalar(&self) -> f32 {
        match self {
            Self::Scalar(v) => *v,
            Self::Linear(leaf) => leaf.intercept,
        }
    }

    /// Evaluate this leaf for a single row passed as a flat feature slice.
    ///
    /// For [`LeafValue::Scalar`], returns the scalar directly.
    /// For [`LeafValue::Linear`], computes `intercept + Σ w_j * features[regressor_j]`.
    #[inline]
    pub fn eval_row(&self, features: &[f32]) -> f32 {
        match self {
            Self::Scalar(v) => *v,
            Self::Linear(leaf) => leaf.eval(features, 0),
        }
    }

    /// Return `true` when this is a constant (scalar) leaf.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitCandidate {
    pub node_id: u32,
    pub feature_index: u32,
    pub threshold_bin: u16,
    pub gain: f32,
    pub default_left: bool,
    /// When true, this split uses a categorical bitset instead of `threshold_bin`.
    pub is_categorical: bool,
    /// Bitset of category IDs that go to the left child. Bit K = 1 means category K goes left.
    /// Only populated when `is_categorical` is true.
    pub categorical_bitset: Option<Vec<u8>>,
    pub left_stats: NodeStats,
    pub right_stats: NodeStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionResult {
    pub left_row_indices: Vec<u32>,
    pub right_row_indices: Vec<u32>,
}
