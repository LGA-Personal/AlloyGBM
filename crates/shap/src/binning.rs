use crate::error::{ShapError, ShapResult};

pub(crate) const TREE_NODE_STRIDE: u32 = 1 << 20;
// SHAP additivity tolerance is computed as
//   atol + rtol * |predicted|
// rather than a fixed absolute bound, so accumulated f32 round-off in
// large-sample explanations (e.g. `feature_importances()` over ~1000
// rows on California Housing with `n_estimators=200`) does not raise
// even though the arithmetic is correct.  Values follow numpy's
// `allclose` convention (atol=1e-5, rtol=1e-4).
pub(crate) const ADDITIVITY_ATOL: f32 = 1e-5;
pub(crate) const ADDITIVITY_RTOL: f32 = 1e-4;

/// Per-feature binning state needed to translate a stump's
/// `threshold_bin: u16` (a bin index in the artifact) to the float
/// threshold the predictor uses at inference time.  Mirrors the three
/// conversion modes implemented by `crates/predictor/src/lib.rs`
/// (`convert_bin_thresholds_to_float`,
/// `convert_bin_thresholds_to_float_quantile`, and
/// `convert_bin_thresholds_to_float_prebinned`).
///
/// When a `BinningContext` is threaded through SHAP, the path walker
/// compares `feature_value < float_threshold` instead of the legacy
/// `feature_value <= split.threshold_bin as f32`.  For
/// `leaf_model="constant"` artifacts the two paths usually reach the
/// same leaf so the legacy comparison sums to a consistent value; for
/// `leaf_model="linear"` artifacts the leaf value depends on `x_j`
/// directly, so disagreement between SHAP's path and the predictor's
/// path produces measurable additivity drift.  This context aligns the
/// two paths.
#[derive(Debug, Clone, PartialEq)]
pub enum BinningContext {
    /// Linear-spaced bins between per-feature `[min, max]`.
    /// Float threshold = `min + ((bin + 0.5) / max_data_bin) * (max - min)`.
    Linear {
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: u16,
    },
    /// Quantile bins. Float threshold = `cuts[bin]` (or `f32::MAX` past
    /// the last cut).
    Quantile { feature_cuts: Vec<Vec<f32>> },
    /// Pre-binned integer features. Float threshold = `bin + 0.5`.
    PreBinned,
    /// Mixed linear / rank-based linear bins.  Features whose
    /// `per_feature` entry is `Some(sorted_values)` were quantized by
    /// rank (sorted unique values → bin = round(rank * max_data_bin /
    /// (N - 1))).  Features whose entry is `None` fall back to standard
    /// linear binning using the global `feature_mins`/`feature_maxs`.
    ///
    /// **Predictor parity.** For mixed linear-rank artifacts the
    /// predictor evaluates *both* tree traversal and piecewise-linear
    /// leaves in bin-index space (see
    /// `predict_dense_quantized_linear_rank` in
    /// `bindings/python/src/lib.rs` — raw floats are quantized once,
    /// then bin indices feed splits and `LinearLeaf::eval_row` alike).
    /// SHAP matches this by quantizing rows internally at the
    /// `explain_rows_from_model` entry point and then dispatching the
    /// rest of the path-walker on `BinningContext::PreBinned`
    /// semantics (`bin_value < bin + 0.5` ⟺ `bin_value ≤ bin`).
    /// `BinningContext::LinearRank` therefore acts as a carrier for the
    /// quantization parameters; its `float_threshold` is not invoked at
    /// runtime — the transformation happens earlier.  Tests against
    /// `float_threshold` document the boundary math for completeness.
    LinearRank {
        per_feature: Vec<Option<Vec<f32>>>,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: u16,
    },
}

/// Quantize a single value with the predictor's rank-quantize rule,
/// matching `quantize_rank_value_wide` in `bindings/python/src/lib.rs`.
fn quantize_rank_value(value: f32, sorted_values: &[f32], max_data_bin: u16) -> f32 {
    if sorted_values.len() <= 1 {
        return 0.0;
    }
    let insertion = sorted_values.partition_point(|probe| *probe <= value);
    let rank = insertion.saturating_sub(1).min(sorted_values.len() - 1);
    let scaled =
        (rank as f32 * max_data_bin as f32) / (sorted_values.len().saturating_sub(1) as f32);
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    };
    rounded.clamp(0.0, max_data_bin as f32)
}

/// Quantize a single value with the predictor's linear-quantize rule,
/// matching `quantize_linear_value_wide` in `bindings/python/src/lib.rs`.
fn quantize_linear_value(value: f32, min_val: f32, max_val: f32, max_data_bin: u16) -> f32 {
    let span = max_val - min_val;
    if span <= f32::EPSILON {
        return 0.0;
    }
    let scaled = ((value - min_val) / span) * max_data_bin as f32;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    };
    rounded.clamp(0.0, max_data_bin as f32)
}

impl BinningContext {
    /// Return the float threshold for a split, matching the predictor's
    /// conversion math exactly.  Panics if the feature index is out of
    /// range — callers must validate before calling.
    #[inline]
    pub(crate) fn float_threshold(&self, feature_index: usize, bin: u16) -> f32 {
        match self {
            BinningContext::Linear {
                feature_mins,
                feature_maxs,
                max_data_bin,
            } => {
                let min_val = feature_mins[feature_index];
                let max_val = feature_maxs[feature_index];
                let span = max_val - min_val;
                if span <= f32::EPSILON {
                    min_val + f32::EPSILON
                } else {
                    min_val + ((bin as f32 + 0.5) / *max_data_bin as f32) * span
                }
            }
            BinningContext::Quantile { feature_cuts } => {
                let cuts = &feature_cuts[feature_index];
                let idx = bin as usize;
                if idx < cuts.len() {
                    cuts[idx]
                } else {
                    f32::MAX
                }
            }
            BinningContext::PreBinned => bin as f32 + 0.5,
            BinningContext::LinearRank {
                per_feature,
                feature_mins,
                feature_maxs,
                max_data_bin,
            } => match &per_feature[feature_index] {
                Some(sorted_values) => {
                    let n = sorted_values.len();
                    if n <= 1 {
                        return f32::MAX;
                    }
                    let n_minus_1 = (n - 1) as f32;
                    let denom = *max_data_bin as f32;
                    let r_crit = (bin as f32 + 0.5) * n_minus_1 / denom;
                    let r_star = (r_crit.ceil() as usize).min(n - 1);
                    sorted_values[r_star]
                }
                None => {
                    let min_val = feature_mins[feature_index];
                    let max_val = feature_maxs[feature_index];
                    let span = max_val - min_val;
                    if span <= f32::EPSILON {
                        min_val + f32::EPSILON
                    } else {
                        min_val + ((bin as f32 + 0.5) / *max_data_bin as f32) * span
                    }
                }
            },
        }
    }

    /// Validate against an expected feature count; returns a
    /// human-readable error otherwise.
    pub(crate) fn validate(&self, feature_count: usize) -> ShapResult<()> {
        match self {
            BinningContext::Linear {
                feature_mins,
                feature_maxs,
                ..
            } => {
                if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
                    return Err(ShapError::InvalidInput(format!(
                        "BinningContext::Linear: feature_mins/feature_maxs length ({}/{}) must match feature_count {feature_count}",
                        feature_mins.len(),
                        feature_maxs.len(),
                    )));
                }
            }
            BinningContext::Quantile { feature_cuts } => {
                if feature_cuts.len() != feature_count {
                    return Err(ShapError::InvalidInput(format!(
                        "BinningContext::Quantile: feature_cuts length {} must match feature_count {feature_count}",
                        feature_cuts.len(),
                    )));
                }
            }
            BinningContext::PreBinned => {}
            BinningContext::LinearRank {
                per_feature,
                feature_mins,
                feature_maxs,
                ..
            } => {
                if per_feature.len() != feature_count
                    || feature_mins.len() != feature_count
                    || feature_maxs.len() != feature_count
                {
                    return Err(ShapError::InvalidInput(format!(
                        "BinningContext::LinearRank: per_feature/feature_mins/feature_maxs lengths ({}/{}/{}) must all match feature_count {feature_count}",
                        per_feature.len(),
                        feature_mins.len(),
                        feature_maxs.len(),
                    )));
                }
            }
        }
        Ok(())
    }

    /// Apply `BinningContext::LinearRank` quantization to a single row,
    /// returning the bin-index representation that the predictor uses
    /// at inference (linear quantize for unflagged features, rank
    /// quantize for `Some(sorted)` features).  Returns `None` for any
    /// other variant — only `LinearRank` triggers internal
    /// quantization.
    pub(crate) fn quantize_row_for_linear_rank(&self, row: &[f32]) -> Option<Vec<f32>> {
        match self {
            BinningContext::LinearRank {
                per_feature,
                feature_mins,
                feature_maxs,
                max_data_bin,
            } => {
                let mdb = *max_data_bin;
                let mut out = Vec::with_capacity(row.len());
                for (fi, &value) in row.iter().enumerate() {
                    let bin = match per_feature.get(fi).and_then(|opt| opt.as_ref()) {
                        Some(sorted) => quantize_rank_value(value, sorted, mdb),
                        None => {
                            quantize_linear_value(value, feature_mins[fi], feature_maxs[fi], mdb)
                        }
                    };
                    out.push(bin);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[inline]
pub(crate) fn additivity_tolerance(predicted: f32) -> f32 {
    ADDITIVITY_ATOL + ADDITIVITY_RTOL * predicted.abs()
}

pub(crate) const MAX_EXACT_SPLIT_FEATURES: usize = 25;
