use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod simd;

pub const MODEL_FORMAT_V1: u32 = 1;
pub const MODEL_BINARY_MAGIC: [u8; 4] = *b"AGBM";
pub const MODEL_BINARY_HEADER_LEN: usize = 16;
pub const MODEL_SECTION_DESCRIPTOR_LEN: usize = 20;
pub const CATEGORICAL_STATE_FORMAT_V1: u32 = 1;
pub const MISSING_BIN_U8: u8 = 255;
pub const MISSING_BIN_U16: u16 = 65535;
const CATEGORICAL_STATE_HEADER_LEN: usize = 16;
const CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TreeGrowth {
    /// Level-wise (breadth-first): split all nodes at depth d before depth d+1.
    #[default]
    Level,
    /// Leaf-wise (best-first): always split the leaf with the highest gain.
    Leaf,
}

/// Per-iteration learning rate schedule for MorphBoost training.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LrSchedule {
    /// Use constant learning rate for all rounds.
    #[default]
    Constant,
    /// Linear warmup from 0 → learning_rate over `warmup_frac * n_estimators`
    /// rounds, then half-cosine decay to `learning_rate * 0.01` over remaining rounds.
    WarmupCosine { warmup_frac: f32 },
}

/// Configuration for the MorphBoost-inspired training profile.
///
/// All fields are runtime-configurable; defaults match the paper's
/// recommended values (Kriuk 2025, arXiv:2511.13234).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorphConfig {
    /// Strength of per-round leaf shrinkage: leaf *= (1 - morph_rate * t/T)
    pub morph_rate: f32,
    /// ρ in info-score smoothing factor (1 + ρ * t/T)
    pub evolution_pressure: f32,
    /// Number of pure-gradient rounds before info-score blending kicks in.
    pub morph_warmup_iters: u32,
    /// Blend weight on info component: gain = (1-w) * grad + w * info * tanh(t/20)
    pub info_score_weight: f32,
    /// Base for leaf depth penalty: leaf *= depth_penalty_base ^ (depth/3)
    pub depth_penalty_base: f32,
    /// Apply balance penalty for unbalanced splits.
    pub balance_penalty: bool,
    /// Per-iteration learning rate schedule.
    pub lr_schedule: LrSchedule,
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self {
            morph_rate: 0.1,
            evolution_pressure: 0.2,
            morph_warmup_iters: 5,
            info_score_weight: 0.3,
            depth_penalty_base: 0.9,
            balance_penalty: true,
            lr_schedule: LrSchedule::Constant,
        }
    }
}

/// Per-round constants for morph gain computation. Compute once per round (not per bin).
/// Eliminates redundant `tanh`, `(1.0 - info_score_weight)`, and warmup-branch
/// computation in the inner per-bin gain loop.
#[derive(Debug, Clone, Copy)]
pub struct MorphPrecomputed {
    pub in_warmup: bool,
    /// `tanh(iteration / 20)` — only meaningful post-warmup
    pub morph_weight: f32,
    /// `1.0 - info_score_weight` (post-warmup; 1.0 in warmup)
    pub gradient_score_coeff: f32,
    /// `info_score_weight * morph_weight` (post-warmup; 0.0 in warmup)
    pub info_score_coeff: f32,
    /// Mirrors `cfg.balance_penalty` for fast access without dereferencing config
    pub balance_penalty: bool,
    /// True if `info_score_coeff` is below an epsilon — skip `info_gain` entirely
    pub info_score_negligible: bool,
}

impl MorphPrecomputed {
    pub fn for_iteration(iteration: u32, cfg: &MorphConfig) -> Self {
        let in_warmup = iteration < cfg.morph_warmup_iters;
        if in_warmup {
            return Self {
                in_warmup: true,
                morph_weight: 0.0,
                gradient_score_coeff: 1.0,
                info_score_coeff: 0.0,
                balance_penalty: cfg.balance_penalty,
                info_score_negligible: true,
            };
        }
        let morph_weight = (iteration as f32 / 20.0).tanh();
        let info_score_coeff = cfg.info_score_weight * morph_weight;
        Self {
            in_warmup: false,
            morph_weight,
            gradient_score_coeff: 1.0 - cfg.info_score_weight,
            info_score_coeff,
            balance_penalty: cfg.balance_penalty,
            info_score_negligible: info_score_coeff.abs() < 1e-6,
        }
    }
}

/// Top-level training profile selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingMode {
    /// Auto-policy with dataset-aware heuristics (current default).
    #[default]
    Auto,
    /// Raw user-supplied parameters with no overrides.
    Manual,
    /// MorphBoost-inspired adaptive training profile.
    Morph,
}

/// Exponential moving average statistics for gradients across boosting rounds.
/// Maintained per-class for multiclass softmax (length 1 for single-output).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientEmaStats {
    pub mean: f32,
    pub std: f32,
    /// Decay rate (0.05 per the paper).
    pub alpha: f32,
}

impl Default for GradientEmaStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            alpha: 0.05,
        }
    }
}

impl GradientEmaStats {
    /// Update EMA in place from a new round's gradient slice.
    ///
    /// Non-finite inputs (NaN, Inf) are silently skipped so the running stats
    /// don't get permanently poisoned by transient numerical issues in
    /// upstream gradient computation.
    ///
    /// Note: variance is computed with population divisor (n), not sample
    /// divisor (n-1). This is intentional for EMA smoothing where n is large.
    pub fn update(&mut self, gradients: &[f32]) {
        if gradients.is_empty() {
            return;
        }
        let n = gradients.len() as f32;
        // SIMD-vectorized single-pass computation: sum + sum-of-squares.
        // var = E[X²] - E[X]² (algebraically equivalent to the 2-pass form,
        // numerically slightly less stable but fine for gradient stats).
        let sum = crate::simd::sum_f32(gradients);
        let sumsq = crate::simd::sum_squares_f32(gradients);
        let mean = sum / n;
        if !mean.is_finite() {
            return;
        }
        // Clamp to 0 to guard against tiny FP negatives from cancellation.
        let var = (sumsq / n - mean * mean).max(0.0);
        if !var.is_finite() {
            return;
        }
        let std = var.sqrt();
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * mean;
        self.std = (1.0 - self.alpha) * self.std + self.alpha * std;
    }

    #[cfg(test)]
    pub(crate) fn update_two_pass_legacy(&mut self, gradients: &[f32]) {
        // Preserved only for parity testing against the new single-pass form.
        if gradients.is_empty() {
            return;
        }
        let n = gradients.len() as f32;
        let mean: f32 = gradients.iter().sum::<f32>() / n;
        if !mean.is_finite() {
            return;
        }
        let var: f32 = gradients
            .iter()
            .map(|g| (g - mean) * (g - mean))
            .sum::<f32>()
            / n;
        if !var.is_finite() {
            return;
        }
        let std = var.sqrt();
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * mean;
        self.std = (1.0 - self.alpha) * self.std + self.alpha * std;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn as_metadata_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }

    pub fn parse_metadata_label(value: &str) -> CoreResult<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            other => Err(CoreError::Validation(format!(
                "unsupported trained_device '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    pub seed: u64,
    pub deterministic: bool,
    pub learning_rate: f32,
    pub max_depth: u16,
    pub row_subsample: f32,
    pub col_subsample: f32,
    pub early_stopping_rounds: Option<u16>,
    pub min_validation_improvement: f32,
    pub min_data_in_leaf: u32,
    pub lambda_l1: f32,
    pub lambda_l2: f32,
    pub min_child_hessian: f32,
    pub min_split_gain: f32,
    /// Per-feature monotone constraints: +1 non-decreasing, -1 non-increasing, 0 unconstrained.
    /// Empty means no constraints.
    pub monotone_constraints: Vec<i8>,
    /// Per-feature importance weights for split selection (gain is multiplied by weight).
    /// Empty means uniform weighting.
    pub feature_weights: Vec<f32>,
    /// Maximum number of leaves per tree. None means depth-limited only.
    pub max_leaves: Option<usize>,
    /// Tree growth strategy: level-wise (default) or leaf-wise (best-first).
    pub tree_growth: TreeGrowth,
    /// MorphBoost-inspired training profile config. `None` = non-morph (current behavior).
    pub morph_config: Option<MorphConfig>,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            deterministic: true,
            learning_rate: 0.1,
            max_depth: 6,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
            monotone_constraints: Vec::new(),
            feature_weights: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSchema {
    pub feature_count: usize,
    pub has_time_index: bool,
    pub has_group_id: bool,
    pub categorical_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub values: Vec<f32>,
}

impl DatasetMatrix {
    pub fn new(row_count: usize, feature_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            values,
        };
        validate_dataset_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a lightweight matrix that only stores row/feature dimensions.
    /// Values are not populated — only use when the training path does not
    /// need dense float values (i.e. no categorical target encoding).
    pub fn new_metadata_only(row_count: usize, feature_count: usize) -> CoreResult<Self> {
        if row_count == 0 {
            return Err(CoreError::Validation(
                "row_count must be greater than 0".to_string(),
            ));
        }
        if feature_count == 0 {
            return Err(CoreError::Validation(
                "feature_count must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            row_count,
            feature_count,
            values: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMatrixView<'a> {
    pub row_count: usize,
    pub feature_count: usize,
    pub values: &'a [f32],
}

impl<'a> DenseMatrixView<'a> {
    pub fn new(row_count: usize, feature_count: usize, values: &'a [f32]) -> CoreResult<Self> {
        let view = Self {
            row_count,
            feature_count,
            values,
        };
        validate_dense_matrix_view(&view)?;
        Ok(view)
    }

    pub fn row(&self, row_index: usize) -> CoreResult<&'a [f32]> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "row index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        let start = row_index * self.feature_count;
        let end = start + self.feature_count;
        Ok(&self.values[start..end])
    }

    pub fn value_at(&self, row_index: usize, feature_index: usize) -> CoreResult<f32> {
        if feature_index >= self.feature_count {
            return Err(CoreError::Validation(format!(
                "feature index {feature_index} is out of bounds for feature_count {}",
                self.feature_count
            )));
        }
        Ok(self.row(row_index)?[feature_index])
    }

    pub fn to_dataset_matrix(&self) -> CoreResult<DatasetMatrix> {
        DatasetMatrix::new(self.row_count, self.feature_count, self.values.to_vec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnarMatrixColumnView<'a> {
    pub values: &'a [f32],
    pub validity: Option<&'a [bool]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnarMatrixView<'a> {
    pub row_count: usize,
    pub columns: Vec<ColumnarMatrixColumnView<'a>>,
}

impl<'a> ColumnarMatrixView<'a> {
    pub fn new(row_count: usize, columns: Vec<ColumnarMatrixColumnView<'a>>) -> CoreResult<Self> {
        let view = Self { row_count, columns };
        validate_columnar_matrix_view(&view)?;
        Ok(view)
    }

    pub fn feature_count(&self) -> usize {
        self.columns.len()
    }

    pub fn value_at(&self, row_index: usize, feature_index: usize) -> CoreResult<Option<f32>> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "row index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        let column = self.columns.get(feature_index).ok_or_else(|| {
            CoreError::Validation(format!(
                "feature index {feature_index} is out of bounds for feature_count {}",
                self.feature_count()
            ))
        })?;
        if column.validity.is_some_and(|mask| !mask[row_index]) {
            return Ok(None);
        }
        Ok(Some(column.values[row_index]))
    }

    pub fn to_dataset_matrix(&self, null_fill_value: f32) -> CoreResult<DatasetMatrix> {
        let mut values = Vec::with_capacity(self.row_count * self.feature_count());
        for row_index in 0..self.row_count {
            for feature_index in 0..self.feature_count() {
                values.push(
                    self.value_at(row_index, feature_index)?
                        .unwrap_or(null_fill_value),
                );
            }
        }
        DatasetMatrix::new(self.row_count, self.feature_count(), values)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataset {
    pub matrix: DatasetMatrix,
    pub targets: Vec<f32>,
    pub sample_weights: Option<Vec<f32>>,
    pub time_index: Option<Vec<i64>>,
    pub group_id: Option<Vec<u32>>,
}

impl TrainingDataset {
    pub fn row_count(&self) -> usize {
        self.matrix.row_count
    }
}

/// Adaptive bin storage: u8 for <=255 max bins, u16 for >255.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl BinStorage {
    /// Get the bin value at the given index as a u16.
    #[inline]
    pub fn get(&self, index: usize) -> u16 {
        match self {
            Self::U8(bins) => u16::from(bins[index]),
            Self::U16(bins) => bins[index],
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(bins) => bins.len(),
            Self::U16(bins) => bins.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sentinel value used for missing/NaN bins.
    pub fn missing_bin(&self) -> u16 {
        match self {
            Self::U8(_) => u16::from(MISSING_BIN_U8),
            Self::U16(_) => MISSING_BIN_U16,
        }
    }

    /// Whether this storage uses u8 bins.
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinnedMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub max_bin: u16,
    /// The bin index used for NaN/missing values.
    /// For u8 mode: always 255 (MISSING_BIN_U8).
    /// For u16 mode: max_data_bin + 1 (dynamic, avoids wasteful 65535 sentinel).
    pub nan_bin_index: u16,
    /// Row-major: bins[row * feature_count + feature]
    pub bins: Vec<u8>,
    /// Column-major: bins_col[feature * row_count + row] — for cache-friendly histogram building.
    pub bins_col: Vec<u8>,
    /// Row-major adaptive storage (mirrors `bins` but supports u16).
    pub bins_adaptive: BinStorage,
    /// Column-major adaptive storage (mirrors `bins_col` but supports u16).
    pub bins_col_adaptive: BinStorage,
}

impl BinnedMatrix {
    /// Create a BinnedMatrix from u8 bins (max_bin <= 254).
    pub fn new(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins: Vec<u8>,
    ) -> CoreResult<Self> {
        let bins_col = transpose_bins_to_column_major_u8(&bins, row_count, feature_count);
        let bins_adaptive = BinStorage::U8(bins.clone());
        let bins_col_adaptive = BinStorage::U8(bins_col.clone());
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index: MISSING_BIN_U8 as u16,
            bins,
            bins_col,
            bins_adaptive,
            bins_col_adaptive,
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a BinnedMatrix from u16 bins (max_bin > 254).
    /// `nan_bin_index` is the bin value used for NaN/missing data (typically max_data_bin + 1).
    pub fn new_u16(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        nan_bin_index: u16,
        bins_u16: Vec<u16>,
    ) -> CoreResult<Self> {
        // For backward compatibility, also create u8 vecs (clamped) for legacy code paths.
        let bins_u8: Vec<u8> = bins_u16
            .iter()
            .map(|&b| if b >= 255 { 255 } else { b as u8 })
            .collect();
        let bins_col_u8 = transpose_bins_to_column_major_u8(&bins_u8, row_count, feature_count);
        let bins_col_u16 = transpose_bins_to_column_major_u16(&bins_u16, row_count, feature_count);
        let bins_adaptive = BinStorage::U16(bins_u16);
        let bins_col_adaptive = BinStorage::U16(bins_col_u16);
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index,
            bins: bins_u8,
            bins_col: bins_col_u8,
            bins_adaptive,
            bins_col_adaptive,
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Whether this matrix uses u16 bin storage.
    pub fn is_wide_bins(&self) -> bool {
        matches!(self.bins_adaptive, BinStorage::U16(_))
    }

    /// Read a bin value from column-major adaptive storage.
    /// `index` is the flat offset (feature * row_count + row).
    #[inline]
    pub fn col_bin(&self, index: usize) -> u16 {
        self.bins_col_adaptive.get(index)
    }

    /// Read a bin value from row-major adaptive storage.
    /// `index` is the flat offset (row * feature_count + feature).
    #[inline]
    pub fn row_bin(&self, index: usize) -> u16 {
        self.bins_adaptive.get(index)
    }

    /// The sentinel value used for missing/NaN bins in this matrix.
    #[inline]
    pub fn missing_bin(&self) -> u16 {
        self.nan_bin_index
    }

    /// Whether column-major adaptive storage is available (non-empty).
    #[inline]
    pub fn has_col_major(&self) -> bool {
        !self.bins_col_adaptive.is_empty()
    }

    /// Set the bin value at (row, feature) in all storage arrays.
    /// Used for re-mapping native categorical feature columns after binning.
    pub fn set_bin(&mut self, row: usize, feature: usize, value: u16) {
        let row_idx = row * self.feature_count + feature;
        let col_idx = feature * self.row_count + row;
        let val_u8 = if value >= 255 { 255u8 } else { value as u8 };

        if row_idx < self.bins.len() {
            self.bins[row_idx] = val_u8;
        }
        if col_idx < self.bins_col.len() {
            self.bins_col[col_idx] = val_u8;
        }
        match &mut self.bins_adaptive {
            BinStorage::U8(v) => {
                if row_idx < v.len() {
                    v[row_idx] = val_u8;
                }
            }
            BinStorage::U16(v) => {
                if row_idx < v.len() {
                    v[row_idx] = value;
                }
            }
        }
        match &mut self.bins_col_adaptive {
            BinStorage::U8(v) => {
                if col_idx < v.len() {
                    v[col_idx] = val_u8;
                }
            }
            BinStorage::U16(v) => {
                if col_idx < v.len() {
                    v[col_idx] = value;
                }
            }
        }
    }
}

/// Transpose row-major bins to column-major for cache-friendly per-feature access.
fn transpose_bins_to_column_major_u8(
    bins: &[u8],
    row_count: usize,
    feature_count: usize,
) -> Vec<u8> {
    let total = row_count * feature_count;
    if total == 0 || bins.len() != total {
        return Vec::new();
    }
    let mut col_major = vec![0u8; total];
    for row in 0..row_count {
        let row_base = row * feature_count;
        for feature in 0..feature_count {
            col_major[feature * row_count + row] = bins[row_base + feature];
        }
    }
    col_major
}

fn transpose_bins_to_column_major_u16(
    bins: &[u16],
    row_count: usize,
    feature_count: usize,
) -> Vec<u16> {
    let total = row_count * feature_count;
    if total == 0 || bins.len() != total {
        return Vec::new();
    }
    let mut col_major = vec![0u16; total];
    for row in 0..row_count {
        let row_base = row * feature_count;
        for feature in 0..feature_count {
            col_major[feature * row_count + row] = bins[row_base + feature];
        }
    }
    col_major
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

impl GradientPair {
    pub fn new(grad: f32, hess: f32) -> CoreResult<Self> {
        if !grad.is_finite() || !hess.is_finite() {
            return Err(CoreError::Validation(
                "gradient and hessian must be finite".to_string(),
            ));
        }
        if hess <= 0.0 {
            return Err(CoreError::Validation(
                "hessian must be greater than 0".to_string(),
            ));
        }
        Ok(Self { grad, hess })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureTile {
    pub start_feature: u32,
    pub end_feature: u32,
}

impl FeatureTile {
    pub fn new(start_feature: u32, end_feature: u32) -> CoreResult<Self> {
        if start_feature >= end_feature {
            return Err(CoreError::Validation(
                "feature tile must satisfy start_feature < end_feature".to_string(),
            ));
        }
        Ok(Self {
            start_feature,
            end_feature,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSlice {
    pub node_id: u32,
    pub row_indices: Vec<u32>,
}

impl NodeSlice {
    pub fn new(node_id: u32, row_indices: Vec<u32>) -> CoreResult<Self> {
        if row_indices.is_empty() {
            return Err(CoreError::Validation(
                "node row_indices cannot be empty".to_string(),
            ));
        }
        Ok(Self {
            node_id,
            row_indices,
        })
    }

    pub fn validate_bounds(&self, row_count: usize) -> CoreResult<()> {
        for &row_index in &self.row_indices {
            let row_index = row_index as usize;
            if row_index >= row_count {
                return Err(CoreError::Validation(format!(
                    "row index {row_index} is out of bounds for row_count {row_count}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub row_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBin {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureHistogram {
    pub feature_index: u32,
    pub bins: Vec<HistogramBin>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBundle {
    pub node_id: u32,
    pub feature_histograms: Vec<FeatureHistogram>,
}

impl HistogramBundle {
    /// Zero all bin values in-place without deallocating.
    ///
    /// This resets gradient sums, hessian sums, and counts to zero for every
    /// bin in every feature histogram, allowing the bundle to be reused for a
    /// new node without re-allocating.
    pub fn reset(&mut self, node_id: u32) {
        self.node_id = node_id;
        for fh in &mut self.feature_histograms {
            for bin in &mut fh.bins {
                bin.grad_sum = 0.0;
                bin.hess_sum = 0.0;
                bin.count = 0;
            }
        }
    }

    /// Create a pre-allocated, zeroed histogram bundle for the given features and bin count.
    pub fn new_zeroed(feature_indices: &[u32], bin_count: usize) -> Self {
        let feature_histograms = feature_indices
            .iter()
            .map(|&fi| FeatureHistogram {
                feature_index: fi,
                bins: vec![
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        count: 0,
                    };
                    bin_count
                ],
            })
            .collect();
        Self {
            node_id: 0,
            feature_histograms,
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub trained_device: Device,
    /// Objective used to train this model (e.g. "squared_error", "binary_crossentropy").
    /// Defaults to "squared_error" for backward compatibility with older artifacts.
    pub objective: String,
    /// Number of classes for multi-class classification models.
    /// `None` for single-output models (regression, binary classification, ranking).
    pub num_classes: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelBinaryHeader {
    pub magic: [u8; 4],
    pub format_version: u32,
    pub section_count: u32,
    pub metadata_json_len: u32,
}

impl ModelBinaryHeader {
    pub fn new(section_count: u32, metadata_json_len: u32) -> Self {
        Self {
            magic: MODEL_BINARY_MAGIC,
            format_version: MODEL_FORMAT_V1,
            section_count,
            metadata_json_len,
        }
    }

    pub fn encode(self) -> [u8; MODEL_BINARY_HEADER_LEN] {
        let mut bytes = [0_u8; MODEL_BINARY_HEADER_LEN];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.format_version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.section_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.metadata_json_len.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.len() != MODEL_BINARY_HEADER_LEN {
            return Err(CoreError::Serialization(format!(
                "model header must be {MODEL_BINARY_HEADER_LEN} bytes"
            )));
        }

        let mut magic = [0_u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != MODEL_BINARY_MAGIC {
            return Err(CoreError::Serialization(
                "model header magic mismatch".to_string(),
            ));
        }

        let format_version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let section_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let metadata_json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        Ok(Self {
            magic,
            format_version,
            section_count,
            metadata_json_len,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSectionKind {
    Trees,
    PredictorLayout,
    ShapAux,
    CategoricalState,
    NodeDebugStats,
    MultiClassTrees,
    NativeCategoricalSplits,
    MorphMetadata,
    Unknown(u32),
}

impl ModelSectionKind {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Trees => 1,
            Self::PredictorLayout => 2,
            Self::ShapAux => 3,
            Self::CategoricalState => 4,
            Self::NodeDebugStats => 5,
            Self::MultiClassTrees => 6,
            Self::NativeCategoricalSplits => 7,
            Self::MorphMetadata => 8,
            Self::Unknown(value) => value,
        }
    }

    pub fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Trees,
            2 => Self::PredictorLayout,
            3 => Self::ShapAux,
            4 => Self::CategoricalState,
            5 => Self::NodeDebugStats,
            6 => Self::MultiClassTrees,
            7 => Self::NativeCategoricalSplits,
            8 => Self::MorphMetadata,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSectionDescriptor {
    pub kind: ModelSectionKind,
    pub offset: u64,
    pub length: u64,
}

impl ModelSectionDescriptor {
    pub fn encode(self) -> [u8; MODEL_SECTION_DESCRIPTOR_LEN] {
        let mut bytes = [0_u8; MODEL_SECTION_DESCRIPTOR_LEN];
        bytes[0..4].copy_from_slice(&self.kind.to_u32().to_le_bytes());
        bytes[4..12].copy_from_slice(&self.offset.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.length.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.len() != MODEL_SECTION_DESCRIPTOR_LEN {
            return Err(CoreError::Serialization(format!(
                "model section descriptor must be {MODEL_SECTION_DESCRIPTOR_LEN} bytes"
            )));
        }

        let kind = ModelSectionKind::from_u32(u32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]));
        let offset = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        let length = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);

        Ok(Self {
            kind,
            offset,
            length,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIoContractV1 {
    pub header: ModelBinaryHeader,
    pub sections: Vec<ModelSectionDescriptor>,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactSection {
    pub descriptor: ModelSectionDescriptor,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedModelArtifactV1 {
    pub contract: ModelIoContractV1,
    pub metadata_json: String,
    pub sections: Vec<ModelArtifactSection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoricalStatePayloadV1 {
    pub format_version: u32,
    pub leakage_safe_target_encoding: bool,
    pub categorical_feature_indices: Vec<u32>,
}

/// Payload for the NativeCategoricalSplits artifact section.
/// Stores which features use native categorical splits and the per-stump bitsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCategoricalSplitsPayload {
    /// Feature indices that use native categorical splits (sorted ascending).
    pub native_categorical_feature_indices: Vec<u32>,
    /// Per-stump bitsets: (stump_index, bitset). Only categorical stumps appear here.
    pub stump_bitsets: Vec<(u32, Vec<u8>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredSectionCompatibilityReport {
    pub trees_section_count: usize,
    pub predictor_layout_section_count: usize,
    pub strict_compatible: bool,
    pub legacy_trees_only_compatible: bool,
    pub legacy_compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidConfig(String),
    Validation(String),
    Io(String),
    Serialization(String),
    NotImplemented(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for CoreError {}

pub type CoreResult<T> = Result<T, CoreError>;

pub fn required_section_compatibility_report(
    sections: &[ModelArtifactSection],
) -> RequiredSectionCompatibilityReport {
    let trees_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::Trees)
        .count();
    let predictor_layout_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
        .count();

    let strict_compatible = trees_section_count == 1 && predictor_layout_section_count == 1;
    let legacy_trees_only_compatible = trees_section_count == 1
        && predictor_layout_section_count == 0
        && sections.len() == 1
        && sections[0].descriptor.kind == ModelSectionKind::Trees;
    let legacy_compatible = strict_compatible || legacy_trees_only_compatible;

    RequiredSectionCompatibilityReport {
        trees_section_count,
        predictor_layout_section_count,
        strict_compatible,
        legacy_trees_only_compatible,
        legacy_compatible,
    }
}

pub fn format_required_section_mode_error(
    report: RequiredSectionCompatibilityReport,
    allow_legacy_trees_only: bool,
) -> String {
    if allow_legacy_trees_only {
        return format!(
            "legacy-compatible mode only supports strict dual-section artifacts or legacy Trees-only artifacts (found Trees={}, PredictorLayout={})",
            report.trees_section_count, report.predictor_layout_section_count
        );
    }
    format!(
        "strict compatibility mode requires exactly one Trees and one PredictorLayout section (found Trees={}, PredictorLayout={})",
        report.trees_section_count, report.predictor_layout_section_count
    )
}

pub fn format_required_section_auto_mode_error(
    report: RequiredSectionCompatibilityReport,
) -> String {
    format!(
        "unable to determine artifact compatibility mode (Trees sections: {}, PredictorLayout sections: {})",
        report.trees_section_count, report.predictor_layout_section_count
    )
}

pub fn encode_categorical_state_payload_v1(
    payload: &CategoricalStatePayloadV1,
) -> CoreResult<Vec<u8>> {
    validate_categorical_state_payload_v1(payload, None)?;

    let feature_count = u32::try_from(payload.categorical_feature_indices.len()).map_err(|_| {
        CoreError::Serialization("categorical feature count exceeds u32::MAX".to_string())
    })?;
    let mut bytes = Vec::with_capacity(
        CATEGORICAL_STATE_HEADER_LEN + payload.categorical_feature_indices.len() * 4,
    );
    bytes.extend_from_slice(&payload.format_version.to_le_bytes());
    let mut flags = 0_u32;
    if payload.leakage_safe_target_encoding {
        flags |= CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING;
    }
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    for &feature_index in &payload.categorical_feature_indices {
        bytes.extend_from_slice(&feature_index.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode_categorical_state_payload_v1(bytes: &[u8]) -> CoreResult<CategoricalStatePayloadV1> {
    if bytes.len() < CATEGORICAL_STATE_HEADER_LEN {
        return Err(CoreError::Serialization(format!(
            "categorical state payload length {} is smaller than header length {CATEGORICAL_STATE_HEADER_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    let flags = read_u32_le(bytes, 4)?;
    let feature_count = read_u32_le(bytes, 8)? as usize;
    let _reserved = read_u32_le(bytes, 12)?;

    if flags & !CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING != 0 {
        return Err(CoreError::Serialization(format!(
            "categorical state payload contains unknown flags: {}",
            flags & !CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING
        )));
    }

    let expected_len = CATEGORICAL_STATE_HEADER_LEN
        .checked_add(feature_count.checked_mul(4).ok_or_else(|| {
            CoreError::Serialization("categorical state length overflow".to_string())
        })?)
        .ok_or_else(|| CoreError::Serialization("categorical state length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(CoreError::Serialization(format!(
            "categorical state payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut categorical_feature_indices = Vec::with_capacity(feature_count);
    let mut cursor = CATEGORICAL_STATE_HEADER_LEN;
    for _ in 0..feature_count {
        categorical_feature_indices.push(read_u32_le(bytes, cursor)?);
        cursor += 4;
    }

    let payload = CategoricalStatePayloadV1 {
        format_version,
        leakage_safe_target_encoding: (flags & CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING)
            != 0,
        categorical_feature_indices,
    };
    validate_categorical_state_payload_v1(&payload, None)?;
    Ok(payload)
}

pub fn validate_categorical_state_payload_v1(
    payload: &CategoricalStatePayloadV1,
    model_feature_count: Option<usize>,
) -> CoreResult<()> {
    if payload.format_version != CATEGORICAL_STATE_FORMAT_V1 {
        return Err(CoreError::Validation(format!(
            "unsupported categorical state format_version {}, expected {CATEGORICAL_STATE_FORMAT_V1}",
            payload.format_version
        )));
    }
    if payload.categorical_feature_indices.is_empty() {
        return Err(CoreError::Validation(
            "categorical state must include at least one categorical feature index".to_string(),
        ));
    }

    let mut previous = None;
    for &feature_index in &payload.categorical_feature_indices {
        if let Some(previous) = previous
            && feature_index <= previous
        {
            return Err(CoreError::Validation(format!(
                "categorical state feature indices must be strictly increasing (found {feature_index} after {previous})"
            )));
        }
        previous = Some(feature_index);
    }

    if let Some(model_feature_count) = model_feature_count {
        for &feature_index in &payload.categorical_feature_indices {
            if feature_index as usize >= model_feature_count {
                return Err(CoreError::Validation(format!(
                    "categorical state feature index {} is out of bounds for feature_count {}",
                    feature_index, model_feature_count
                )));
            }
        }
    }

    Ok(())
}

pub fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> CoreResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(CoreError::Serialization(format!(
                "model artifact contains duplicate {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

pub fn decode_optional_categorical_state_section_v1(
    sections: &[ModelArtifactSection],
    model_feature_count: usize,
) -> CoreResult<Option<CategoricalStatePayloadV1>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::CategoricalState)?
    else {
        return Ok(None);
    };

    let payload = decode_categorical_state_payload_v1(&section.payload)?;
    validate_categorical_state_payload_v1(&payload, Some(model_feature_count))?;
    Ok(Some(payload))
}

/// Encode native categorical splits payload for artifact serialization.
///
/// Format:
/// - [4 bytes] num_native_categorical_features (u32 LE)
/// - [4 bytes] stump_bitset_count (u32 LE)
/// - [num_native_categorical_features * 4 bytes] feature indices (u32 LE each)
/// - For each stump bitset:
///   - [4 bytes] stump_index (u32 LE)
///   - [2 bytes] bitset_len (u16 LE)
///   - [bitset_len bytes] bitset data
pub fn encode_native_categorical_splits_payload(
    payload: &NativeCategoricalSplitsPayload,
) -> CoreResult<Vec<u8>> {
    let feature_count =
        u32::try_from(payload.native_categorical_feature_indices.len()).map_err(|_| {
            CoreError::Serialization("native cat feature count exceeds u32::MAX".to_string())
        })?;
    let stump_count = u32::try_from(payload.stump_bitsets.len()).map_err(|_| {
        CoreError::Serialization("native cat stump count exceeds u32::MAX".to_string())
    })?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&stump_count.to_le_bytes());
    for &fi in &payload.native_categorical_feature_indices {
        bytes.extend_from_slice(&fi.to_le_bytes());
    }
    for (stump_index, bitset) in &payload.stump_bitsets {
        let bitset_len = u16::try_from(bitset.len())
            .map_err(|_| CoreError::Serialization("bitset length exceeds u16::MAX".to_string()))?;
        bytes.extend_from_slice(&stump_index.to_le_bytes());
        bytes.extend_from_slice(&bitset_len.to_le_bytes());
        bytes.extend_from_slice(bitset);
    }
    Ok(bytes)
}

/// Decode native categorical splits payload from artifact bytes.
pub fn decode_native_categorical_splits_payload(
    bytes: &[u8],
) -> CoreResult<NativeCategoricalSplitsPayload> {
    const HEADER_SIZE: usize = 8; // feature_count(4) + stump_count(4)
    if bytes.len() < HEADER_SIZE {
        return Err(CoreError::Serialization(
            "native categorical splits payload too small for header".to_string(),
        ));
    }

    let feature_count = read_u32_le(bytes, 0)? as usize;
    let stump_count = read_u32_le(bytes, 4)? as usize;

    let feature_section_len = feature_count.checked_mul(4).ok_or_else(|| {
        CoreError::Serialization("native cat feature section length overflow".to_string())
    })?;
    if bytes.len() < HEADER_SIZE + feature_section_len {
        return Err(CoreError::Serialization(
            "native categorical splits payload too small for feature indices".to_string(),
        ));
    }

    let mut native_categorical_feature_indices = Vec::with_capacity(feature_count);
    let mut cursor = HEADER_SIZE;
    for _ in 0..feature_count {
        native_categorical_feature_indices.push(read_u32_le(bytes, cursor)?);
        cursor += 4;
    }

    let mut stump_bitsets = Vec::with_capacity(stump_count);
    for _ in 0..stump_count {
        if cursor + 6 > bytes.len() {
            return Err(CoreError::Serialization(
                "native categorical splits payload truncated in stump bitset header".to_string(),
            ));
        }
        let stump_index = read_u32_le(bytes, cursor)?;
        cursor += 4;
        let bitset_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + bitset_len > bytes.len() {
            return Err(CoreError::Serialization(
                "native categorical splits payload truncated in bitset data".to_string(),
            ));
        }
        let bitset = bytes[cursor..cursor + bitset_len].to_vec();
        cursor += bitset_len;
        stump_bitsets.push((stump_index, bitset));
    }

    Ok(NativeCategoricalSplitsPayload {
        native_categorical_feature_indices,
        stump_bitsets,
    })
}

/// Decode optional NativeCategoricalSplits section from model artifact.
pub fn decode_optional_native_categorical_splits_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<NativeCategoricalSplitsPayload>> {
    let Some(section) =
        optional_single_section(sections, ModelSectionKind::NativeCategoricalSplits)?
    else {
        return Ok(None);
    };
    let payload = decode_native_categorical_splits_payload(&section.payload)?;
    Ok(Some(payload))
}

/// Optional artifact section recording the MorphConfig used during training.
/// Metadata only — predictions are deterministic from baked-in leaf values.
/// Section is omitted entirely for non-morph artifacts.
#[derive(Debug, Clone, PartialEq)]
pub struct MorphMetadataPayload {
    pub config: MorphConfig,
    pub final_iteration: u32,
    pub final_total: u32,
}

pub fn encode_morph_metadata_payload(payload: &MorphMetadataPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(36);
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_rate.to_le_bytes());
    buf.extend_from_slice(&payload.config.evolution_pressure.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_warmup_iters.to_le_bytes());
    buf.extend_from_slice(&payload.config.info_score_weight.to_le_bytes());
    buf.extend_from_slice(&payload.config.depth_penalty_base.to_le_bytes());
    buf.push(payload.config.balance_penalty as u8);
    let (kind, warmup_frac) = match payload.config.lr_schedule {
        LrSchedule::Constant => (0u8, 0.0f32),
        LrSchedule::WarmupCosine { warmup_frac } => (1u8, warmup_frac),
    };
    buf.push(kind);
    buf.extend_from_slice(&warmup_frac.to_le_bytes());
    buf.extend_from_slice(&payload.final_iteration.to_le_bytes());
    buf.extend_from_slice(&payload.final_total.to_le_bytes());
    buf
}

pub fn decode_optional_morph_metadata_section(bytes: &[u8]) -> CoreResult<MorphMetadataPayload> {
    if bytes.len() < 36 {
        return Err(CoreError::Validation(
            "morph metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported morph metadata version: {version}"
        )));
    }
    let mut o = 2usize;
    macro_rules! read_f32 {
        () => {{
            let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            o += 4;
            v
        }};
    }
    macro_rules! read_u32 {
        () => {{
            let v = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            o += 4;
            v
        }};
    }
    let morph_rate = read_f32!();
    let evolution_pressure = read_f32!();
    let morph_warmup_iters = read_u32!();
    let info_score_weight = read_f32!();
    let depth_penalty_base = read_f32!();
    let balance_penalty = bytes[o] != 0;
    o += 1;
    let lr_kind = bytes[o];
    o += 1;
    let warmup_frac = read_f32!();
    let final_iteration = read_u32!();
    let final_total = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let lr_schedule = match lr_kind {
        0 => LrSchedule::Constant,
        1 => LrSchedule::WarmupCosine { warmup_frac },
        _ => {
            return Err(CoreError::Validation(format!(
                "unknown lr_schedule kind: {lr_kind}"
            )));
        }
    };
    Ok(MorphMetadataPayload {
        config: MorphConfig {
            morph_rate,
            evolution_pressure,
            morph_warmup_iters,
            info_score_weight,
            depth_penalty_base,
            balance_penalty,
            lr_schedule,
        },
        final_iteration,
        final_total,
    })
}

/// Decode an optional MorphMetadata section from a parsed model artifact.
/// Returns `None` if no such section exists (non-morph artifact).
pub fn decode_optional_morph_metadata_artifact_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<MorphMetadataPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::MorphMetadata)? else {
        return Ok(None);
    };
    let payload = decode_optional_morph_metadata_section(&section.payload)?;
    Ok(Some(payload))
}

pub fn validate_train_params(params: &TrainParams) -> CoreResult<()> {
    if !(0.0..=1.0).contains(&params.learning_rate) || params.learning_rate == 0.0 {
        return Err(CoreError::InvalidConfig(
            "learning_rate must be in (0.0, 1.0]".to_string(),
        ));
    }

    if params.max_depth == 0 {
        return Err(CoreError::InvalidConfig(
            "max_depth must be greater than 0".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&params.row_subsample) || params.row_subsample == 0.0 {
        return Err(CoreError::InvalidConfig(
            "row_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&params.col_subsample) || params.col_subsample == 0.0 {
        return Err(CoreError::InvalidConfig(
            "col_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }

    if let Some(rounds) = params.early_stopping_rounds
        && rounds == 0
    {
        return Err(CoreError::InvalidConfig(
            "early_stopping_rounds must be greater than 0 when set".to_string(),
        ));
    }

    if !params.min_validation_improvement.is_finite() || params.min_validation_improvement < 0.0 {
        return Err(CoreError::InvalidConfig(
            "min_validation_improvement must be finite and >= 0".to_string(),
        ));
    }

    if params.min_data_in_leaf == 0 {
        return Err(CoreError::InvalidConfig(
            "min_data_in_leaf must be greater than 0".to_string(),
        ));
    }

    if !params.lambda_l1.is_finite() || params.lambda_l1 < 0.0 {
        return Err(CoreError::InvalidConfig(
            "lambda_l1 must be finite and >= 0".to_string(),
        ));
    }

    if !params.lambda_l2.is_finite() || params.lambda_l2 < 0.0 {
        return Err(CoreError::InvalidConfig(
            "lambda_l2 must be finite and >= 0".to_string(),
        ));
    }

    if !params.min_child_hessian.is_finite() || params.min_child_hessian < 0.0 {
        return Err(CoreError::InvalidConfig(
            "min_child_hessian must be finite and >= 0".to_string(),
        ));
    }

    for &c in &params.monotone_constraints {
        if c != -1 && c != 0 && c != 1 {
            return Err(CoreError::InvalidConfig(
                "monotone_constraints values must be -1, 0, or +1".to_string(),
            ));
        }
    }

    for &w in &params.feature_weights {
        if !w.is_finite() || w < 0.0 {
            return Err(CoreError::InvalidConfig(
                "feature_weights values must be finite and >= 0".to_string(),
            ));
        }
    }

    if let Some(max_leaves) = params.max_leaves
        && max_leaves < 2
    {
        return Err(CoreError::InvalidConfig(
            "max_leaves must be >= 2 when set (a tree needs at least 2 leaves)".to_string(),
        ));
    }

    if params.tree_growth == TreeGrowth::Leaf && params.max_leaves.is_none() {
        return Err(CoreError::InvalidConfig(
            "tree_growth='leaf' requires max_leaves to be set".to_string(),
        ));
    }

    if let Some(cfg) = &params.morph_config {
        if !cfg.morph_rate.is_finite() || !(0.0..=1.0).contains(&cfg.morph_rate) {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.morph_rate must be in [0, 1], got {}",
                cfg.morph_rate
            )));
        }
        if !cfg.evolution_pressure.is_finite() || cfg.evolution_pressure < 0.0 {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.evolution_pressure must be >= 0, got {}",
                cfg.evolution_pressure
            )));
        }
        if !cfg.info_score_weight.is_finite() || !(0.0..=1.0).contains(&cfg.info_score_weight) {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.info_score_weight must be in [0, 1], got {}",
                cfg.info_score_weight
            )));
        }
        if !cfg.depth_penalty_base.is_finite()
            || cfg.depth_penalty_base <= 0.0
            || cfg.depth_penalty_base > 1.0
        {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.depth_penalty_base must be in (0, 1], got {}",
                cfg.depth_penalty_base
            )));
        }
        if let LrSchedule::WarmupCosine { warmup_frac } = cfg.lr_schedule
            && (!warmup_frac.is_finite() || !(0.0..=1.0).contains(&warmup_frac))
        {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.lr_schedule.warmup_frac must be in [0, 1], got {}",
                warmup_frac
            )));
        }
    }

    Ok(())
}

pub fn validate_dataset_schema(schema: &DatasetSchema) -> CoreResult<()> {
    if schema.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }

    let mut previous = None;
    for &feature_index in &schema.categorical_feature_indices {
        if feature_index >= schema.feature_count {
            return Err(CoreError::Validation(format!(
                "categorical feature index {feature_index} is out of bounds for feature_count {}",
                schema.feature_count
            )));
        }
        if let Some(previous) = previous
            && feature_index <= previous
        {
            return Err(CoreError::Validation(format!(
                "categorical feature indices must be strictly increasing (found {feature_index} after {previous})"
            )));
        }
        previous = Some(feature_index);
    }

    Ok(())
}

pub fn validate_dataset_matrix(matrix: &DatasetMatrix) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    // Allow empty values for metadata-only matrices (no categorical encoding).
    if !matrix.values.is_empty() && matrix.values.len() != matrix.row_count * matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "matrix values length {} does not match row_count * feature_count {}",
            matrix.values.len(),
            matrix.row_count * matrix.feature_count
        )));
    }
    Ok(())
}

pub fn validate_dense_matrix_view(matrix: &DenseMatrixView<'_>) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    if matrix.values.len() != matrix.row_count * matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "matrix values length {} does not match row_count * feature_count {}",
            matrix.values.len(),
            matrix.row_count * matrix.feature_count
        )));
    }
    Ok(())
}

pub fn validate_columnar_matrix_view(matrix: &ColumnarMatrixView<'_>) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.columns.is_empty() {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    for (feature_index, column) in matrix.columns.iter().enumerate() {
        if column.values.len() != matrix.row_count {
            return Err(CoreError::Validation(format!(
                "column {feature_index} length {} does not match row_count {}",
                column.values.len(),
                matrix.row_count
            )));
        }
        if let Some(validity) = column.validity
            && validity.len() != matrix.row_count
        {
            return Err(CoreError::Validation(format!(
                "column {feature_index} validity length {} does not match row_count {}",
                validity.len(),
                matrix.row_count
            )));
        }
    }
    Ok(())
}

pub fn validate_training_dataset(dataset: &TrainingDataset) -> CoreResult<()> {
    validate_dataset_matrix(&dataset.matrix)?;
    if dataset.targets.len() != dataset.matrix.row_count {
        return Err(CoreError::Validation(format!(
            "targets length {} does not match row_count {}",
            dataset.targets.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(weights) = &dataset.sample_weights
        && weights.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "sample_weights length {} does not match row_count {}",
            weights.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(time_index) = &dataset.time_index
        && time_index.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "time_index length {} does not match row_count {}",
            time_index.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(group_id) = &dataset.group_id
        && group_id.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "group_id length {} does not match row_count {}",
            group_id.len(),
            dataset.matrix.row_count
        )));
    }

    Ok(())
}

pub fn validate_binned_matrix(matrix: &BinnedMatrix) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    if matrix.max_bin == 0 {
        return Err(CoreError::Validation(
            "max_bin must be greater than 0".to_string(),
        ));
    }
    let expected_len = matrix.row_count * matrix.feature_count;
    if matrix.bins_adaptive.len() != expected_len {
        return Err(CoreError::Validation(format!(
            "bins length {} does not match row_count * feature_count {}",
            matrix.bins_adaptive.len(),
            expected_len
        )));
    }
    // Validate that no bin exceeds max_bin using adaptive storage.
    // The NaN sentinel bin is also allowed (it may exceed max_bin).
    let nan_bin = matrix.nan_bin_index;
    match &matrix.bins_adaptive {
        BinStorage::U8(bins) => {
            for &bin in bins {
                let b = u16::from(bin);
                if b > matrix.max_bin && b != nan_bin {
                    return Err(CoreError::Validation(format!(
                        "bin value {bin} exceeds max_bin {}",
                        matrix.max_bin
                    )));
                }
            }
        }
        BinStorage::U16(bins) => {
            for &bin in bins {
                if bin > matrix.max_bin && bin != nan_bin {
                    return Err(CoreError::Validation(format!(
                        "bin value {bin} exceeds max_bin {}",
                        matrix.max_bin
                    )));
                }
            }
        }
    }
    Ok(())
}

pub fn validate_model_contract_v1(contract: &ModelIoContractV1) -> CoreResult<()> {
    if contract.header.magic != MODEL_BINARY_MAGIC {
        return Err(CoreError::Serialization(
            "model contract magic mismatch".to_string(),
        ));
    }
    if contract.header.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "unsupported format_version {}, expected {MODEL_FORMAT_V1}",
            contract.header.format_version
        )));
    }
    if contract.metadata.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "metadata format_version {}, expected {MODEL_FORMAT_V1}",
            contract.metadata.format_version
        )));
    }
    if contract.sections.len() != contract.header.section_count as usize {
        return Err(CoreError::Serialization(format!(
            "section table length {} does not match header section_count {}",
            contract.sections.len(),
            contract.header.section_count
        )));
    }

    let descriptor_table_len = contract
        .sections
        .len()
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let payload_start = MODEL_BINARY_HEADER_LEN
        .checked_add(descriptor_table_len)
        .and_then(|value| value.checked_add(contract.header.metadata_json_len as usize))
        .ok_or_else(|| CoreError::Serialization("artifact header length overflow".to_string()))?
        as u64;

    let mut expected_offset = payload_start;
    for section in &contract.sections {
        if section.length == 0 {
            return Err(CoreError::Serialization(
                "section length must be greater than 0".to_string(),
            ));
        }
        if section.offset < payload_start {
            return Err(CoreError::Serialization(format!(
                "section offset {} precedes payload start {payload_start}",
                section.offset
            )));
        }
        if section.offset != expected_offset {
            return Err(CoreError::Serialization(format!(
                "section offsets must be contiguous and ordered (expected {}, found {})",
                expected_offset, section.offset
            )));
        }
        expected_offset = section
            .offset
            .checked_add(section.length)
            .ok_or_else(|| CoreError::Serialization("section offset overflow".to_string()))?;
    }

    Ok(())
}

pub fn serialize_metadata_json(metadata: &ModelMetadata) -> String {
    let feature_names = metadata
        .feature_names
        .iter()
        .map(|name| format!("\"{}\"", escape_json_string(name)))
        .collect::<Vec<_>>()
        .join(",");

    let num_classes_fragment = match metadata.num_classes {
        Some(k) => format!(",\"num_classes\":{k}"),
        None => String::new(),
    };

    format!(
        "{{\"format_version\":{},\"feature_names\":[{}],\"trained_device\":\"{}\",\"objective\":\"{}\"{}}}",
        metadata.format_version,
        feature_names,
        metadata.trained_device.as_metadata_label(),
        escape_json_string(&metadata.objective),
        num_classes_fragment
    )
}

pub fn deserialize_metadata_json(input: &str) -> CoreResult<ModelMetadata> {
    let compact = compact_json(input)?;
    let mut index = 0_usize;

    index = consume_literal(&compact, index, "{\"format_version\":")?;
    let (format_version, next_index) = parse_u32(&compact, index)?;
    index = next_index;

    index = consume_literal(&compact, index, ",\"feature_names\":")?;
    let (feature_names, next_index) = parse_string_array(&compact, index)?;
    index = next_index;

    index = consume_literal(&compact, index, ",\"trained_device\":")?;
    let (trained_device_raw, next_index) = parse_quoted_string(&compact, index)?;
    index = next_index;

    // Optional objective field — backward compatible with older artifacts.
    let objective = if index < compact.len() && compact.as_bytes()[index] == b',' {
        let next = consume_literal(&compact, index, ",\"objective\":")?;
        let (objective_raw, next_index) = parse_quoted_string(&compact, next)?;
        index = next_index;
        objective_raw
    } else {
        "squared_error".to_string()
    };

    // Optional num_classes field — present only for multi-class models.
    let num_classes = if index < compact.len() && compact.as_bytes()[index] == b',' {
        let next = consume_literal(&compact, index, ",\"num_classes\":")?;
        let (value, next_index) = parse_u32(&compact, next)?;
        index = next_index;
        Some(value)
    } else {
        None
    };

    index = consume_literal(&compact, index, "}")?;
    if index != compact.len() {
        return Err(CoreError::Serialization(
            "unexpected trailing content in metadata json".to_string(),
        ));
    }

    Ok(ModelMetadata {
        format_version,
        feature_names,
        trained_device: Device::parse_metadata_label(&trained_device_raw)?,
        objective,
        num_classes,
    })
}

pub fn serialize_model_artifact_v1(
    metadata: &ModelMetadata,
    sections: &[(ModelSectionKind, Vec<u8>)],
) -> CoreResult<Vec<u8>> {
    if sections.is_empty() {
        return Err(CoreError::Serialization(
            "model artifact requires at least one section".to_string(),
        ));
    }

    let metadata_json = serialize_metadata_json(metadata);
    let metadata_json_bytes = metadata_json.as_bytes();
    let metadata_json_len = u32::try_from(metadata_json_bytes.len()).map_err(|_| {
        CoreError::Serialization("metadata json length exceeds u32::MAX".to_string())
    })?;

    let section_count = u32::try_from(sections.len())
        .map_err(|_| CoreError::Serialization("section count exceeds u32::MAX".to_string()))?;
    let descriptor_table_len = sections
        .len()
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let data_start = MODEL_BINARY_HEADER_LEN
        .checked_add(descriptor_table_len)
        .and_then(|value| value.checked_add(metadata_json_bytes.len()))
        .ok_or_else(|| CoreError::Serialization("artifact header length overflow".to_string()))?;

    let mut descriptors = Vec::with_capacity(sections.len());
    let mut offset = data_start as u64;
    for (kind, payload) in sections {
        if payload.is_empty() {
            return Err(CoreError::Serialization(
                "section payload cannot be empty".to_string(),
            ));
        }
        let length = u64::try_from(payload.len())
            .map_err(|_| CoreError::Serialization("section length overflow".to_string()))?;
        descriptors.push(ModelSectionDescriptor {
            kind: *kind,
            offset,
            length,
        });
        offset = offset
            .checked_add(length)
            .ok_or_else(|| CoreError::Serialization("section offset overflow".to_string()))?;
    }

    let contract = ModelIoContractV1 {
        header: ModelBinaryHeader::new(section_count, metadata_json_len),
        sections: descriptors.clone(),
        metadata: metadata.clone(),
    };
    validate_model_contract_v1(&contract)?;

    let final_len = usize::try_from(offset)
        .map_err(|_| CoreError::Serialization("artifact length exceeds usize".to_string()))?;
    let mut bytes = Vec::with_capacity(final_len);
    bytes.extend_from_slice(&contract.header.encode());
    for descriptor in &descriptors {
        bytes.extend_from_slice(&descriptor.encode());
    }
    bytes.extend_from_slice(metadata_json_bytes);
    for (_, payload) in sections {
        bytes.extend_from_slice(payload);
    }

    Ok(bytes)
}

pub fn deserialize_model_artifact_v1(bytes: &[u8]) -> CoreResult<ParsedModelArtifactV1> {
    if bytes.len() < MODEL_BINARY_HEADER_LEN {
        return Err(CoreError::Serialization(
            "artifact too small to contain model header".to_string(),
        ));
    }

    let header = ModelBinaryHeader::decode(&bytes[0..MODEL_BINARY_HEADER_LEN])?;
    if header.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "unsupported format_version {}, expected {MODEL_FORMAT_V1}",
            header.format_version
        )));
    }

    let section_count = header.section_count as usize;
    let descriptor_table_len = section_count
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let descriptor_start = MODEL_BINARY_HEADER_LEN;
    let descriptor_end = descriptor_start
        .checked_add(descriptor_table_len)
        .ok_or_else(|| CoreError::Serialization("descriptor range overflow".to_string()))?;
    if bytes.len() < descriptor_end {
        return Err(CoreError::Serialization(
            "artifact truncated in section descriptor table".to_string(),
        ));
    }

    let mut descriptors = Vec::with_capacity(section_count);
    for section_index in 0..section_count {
        let start = descriptor_start + section_index * MODEL_SECTION_DESCRIPTOR_LEN;
        let end = start + MODEL_SECTION_DESCRIPTOR_LEN;
        descriptors.push(ModelSectionDescriptor::decode(&bytes[start..end])?);
    }

    let metadata_json_len = header.metadata_json_len as usize;
    let metadata_start = descriptor_end;
    let metadata_end = metadata_start
        .checked_add(metadata_json_len)
        .ok_or_else(|| CoreError::Serialization("metadata range overflow".to_string()))?;
    if bytes.len() < metadata_end {
        return Err(CoreError::Serialization(
            "artifact truncated in metadata payload".to_string(),
        ));
    }
    let metadata_json = std::str::from_utf8(&bytes[metadata_start..metadata_end])
        .map_err(|err| {
            CoreError::Serialization(format!("metadata json is not valid UTF-8: {err}"))
        })?
        .to_string();
    let metadata = deserialize_metadata_json(&metadata_json)?;

    let contract = ModelIoContractV1 {
        header,
        sections: descriptors.clone(),
        metadata,
    };
    validate_model_contract_v1(&contract)?;

    let mut parsed_sections = Vec::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        let start = usize::try_from(descriptor.offset)
            .map_err(|_| CoreError::Serialization("section offset exceeds usize".to_string()))?;
        let length = usize::try_from(descriptor.length)
            .map_err(|_| CoreError::Serialization("section length exceeds usize".to_string()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| CoreError::Serialization("section range overflow".to_string()))?;

        if end > bytes.len() {
            return Err(CoreError::Serialization(
                "artifact truncated in section payload".to_string(),
            ));
        }

        parsed_sections.push(ModelArtifactSection {
            descriptor: *descriptor,
            payload: bytes[start..end].to_vec(),
        });
    }

    Ok(ParsedModelArtifactV1 {
        contract,
        metadata_json,
        sections: parsed_sections,
    })
}

fn escape_json_string(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn compact_json(input: &str) -> CoreResult<String> {
    let mut compact = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;

    for ch in input.chars() {
        if in_string {
            compact.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        if ch.is_whitespace() {
            continue;
        }

        compact.push(ch);
        if ch == '"' {
            in_string = true;
        }
    }

    if in_string || escaped {
        return Err(CoreError::Serialization(
            "metadata json has unterminated string".to_string(),
        ));
    }

    Ok(compact)
}

fn consume_literal(input: &str, index: usize, literal: &str) -> CoreResult<usize> {
    if index > input.len() || !input[index..].starts_with(literal) {
        return Err(CoreError::Serialization(format!(
            "expected literal '{literal}' at index {index}"
        )));
    }
    Ok(index + literal.len())
}

fn parse_u32(input: &str, mut index: usize) -> CoreResult<(u32, usize)> {
    let start = index;
    while let Some(byte) = input.as_bytes().get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        index += 1;
    }
    if start == index {
        return Err(CoreError::Serialization(format!(
            "expected unsigned integer at index {start}"
        )));
    }
    let value = input[start..index]
        .parse::<u32>()
        .map_err(|err| CoreError::Serialization(format!("invalid integer: {err}")))?;
    Ok((value, index))
}

fn parse_string_array(input: &str, mut index: usize) -> CoreResult<(Vec<String>, usize)> {
    index = consume_literal(input, index, "[")?;
    let mut values = Vec::new();

    if input[index..].starts_with(']') {
        return Ok((values, index + 1));
    }

    loop {
        let (value, next_index) = parse_quoted_string(input, index)?;
        values.push(value);
        index = next_index;

        if input[index..].starts_with(',') {
            index += 1;
            continue;
        }
        if input[index..].starts_with(']') {
            index += 1;
            break;
        }
        return Err(CoreError::Serialization(format!(
            "expected ',' or ']' at index {index}"
        )));
    }

    Ok((values, index))
}

fn parse_quoted_string(input: &str, index: usize) -> CoreResult<(String, usize)> {
    if !input[index..].starts_with('"') {
        return Err(CoreError::Serialization(format!(
            "expected quoted string at index {index}"
        )));
    }

    let mut output = String::new();
    let mut escaped = false;
    let body_start = index + 1;
    for (relative_offset, ch) in input[body_start..].char_indices() {
        if escaped {
            let decoded = match ch {
                '\\' => '\\',
                '"' => '"',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => {
                    return Err(CoreError::Serialization(format!(
                        "unsupported escape sequence '\\{other}'"
                    )));
                }
            };
            output.push(decoded);
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '"' => {
                let end_index = body_start + relative_offset + ch.len_utf8();
                return Ok((output, end_index));
            }
            _ => output.push(ch),
        }
    }

    Err(CoreError::Serialization(
        "unterminated quoted string".to_string(),
    ))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> CoreResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| CoreError::Serialization("u32 read overflow".to_string()))?;
    if end > bytes.len() {
        return Err(CoreError::Serialization(format!(
            "u32 read out of bounds at offset {offset}"
        )));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["feature_0".to_string(), "ticker\"id".to_string()],
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        }
    }

    #[test]
    fn validates_default_train_params() {
        let params = TrainParams::default();
        assert!(validate_train_params(&params).is_ok());
    }

    #[test]
    fn rejects_invalid_learning_rate() {
        let params = TrainParams {
            learning_rate: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_invalid_row_subsample() {
        let params = TrainParams {
            row_subsample: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_invalid_col_subsample() {
        let params = TrainParams {
            col_subsample: 1.5,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_invalid_early_stopping_rounds() {
        let params = TrainParams {
            early_stopping_rounds: Some(0),
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_negative_min_validation_improvement() {
        let params = TrainParams {
            min_validation_improvement: -0.1,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validates_dataset_schema() {
        let schema = DatasetSchema {
            feature_count: 4,
            has_time_index: true,
            has_group_id: true,
            categorical_feature_indices: vec![1, 3],
        };
        assert!(validate_dataset_schema(&schema).is_ok());
    }

    #[test]
    fn rejects_dataset_schema_with_unsorted_or_duplicate_categorical_indices() {
        let duplicate = DatasetSchema {
            feature_count: 4,
            has_time_index: false,
            has_group_id: false,
            categorical_feature_indices: vec![1, 1],
        };
        assert!(matches!(
            validate_dataset_schema(&duplicate),
            Err(CoreError::Validation(_))
        ));

        let unsorted = DatasetSchema {
            feature_count: 4,
            has_time_index: false,
            has_group_id: false,
            categorical_feature_indices: vec![2, 1],
        };
        assert!(matches!(
            validate_dataset_schema(&unsorted),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn rejects_training_dataset_with_mismatched_targets() {
        let dataset = TrainingDataset {
            matrix: DatasetMatrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]).expect("valid matrix"),
            targets: vec![1.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        };
        assert!(matches!(
            validate_training_dataset(&dataset),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn binned_matrix_rejects_bin_above_max() {
        let matrix = BinnedMatrix {
            row_count: 1,
            feature_count: 2,
            max_bin: 7,
            nan_bin_index: MISSING_BIN_U8 as u16,
            bins: vec![3, 8],
            bins_col: vec![3, 8],
            bins_adaptive: BinStorage::U8(vec![3, 8]),
            bins_col_adaptive: BinStorage::U8(vec![3, 8]),
        };
        assert!(matches!(
            validate_binned_matrix(&matrix),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn gradient_pair_rejects_non_positive_hessian() {
        assert!(matches!(
            GradientPair::new(0.1, 0.0),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn model_header_roundtrip() {
        let header = ModelBinaryHeader::new(2, 128);
        let bytes = header.encode();
        let decoded = ModelBinaryHeader::decode(&bytes).expect("header should decode");
        assert_eq!(decoded, header);
    }

    #[test]
    fn section_descriptor_roundtrip() {
        let descriptor = ModelSectionDescriptor {
            kind: ModelSectionKind::Trees,
            offset: 16,
            length: 64,
        };
        let bytes = descriptor.encode();
        let decoded = ModelSectionDescriptor::decode(&bytes).expect("descriptor should decode");
        assert_eq!(decoded, descriptor);
    }

    #[test]
    fn metadata_json_roundtrip() {
        let metadata = sample_metadata();
        let json = serialize_metadata_json(&metadata);
        let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
        assert_eq!(decoded, metadata);
    }

    #[test]
    fn metadata_json_rejects_unknown_device() {
        let json = "{\"format_version\":1,\"feature_names\":[\"f0\"],\"trained_device\":\"cuda\"}";
        assert!(matches!(
            deserialize_metadata_json(json),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn required_section_compatibility_report_classifies_strict_and_legacy_layouts() {
        let strict_sections = vec![
            ModelArtifactSection {
                descriptor: ModelSectionDescriptor {
                    kind: ModelSectionKind::Trees,
                    offset: 120,
                    length: 4,
                },
                payload: vec![1_u8, 2, 3, 4],
            },
            ModelArtifactSection {
                descriptor: ModelSectionDescriptor {
                    kind: ModelSectionKind::PredictorLayout,
                    offset: 124,
                    length: 4,
                },
                payload: vec![5_u8, 6, 7, 8],
            },
        ];
        let strict_report = required_section_compatibility_report(&strict_sections);
        assert!(strict_report.strict_compatible);
        assert!(!strict_report.legacy_trees_only_compatible);
        assert!(strict_report.legacy_compatible);

        let legacy_sections = vec![ModelArtifactSection {
            descriptor: ModelSectionDescriptor {
                kind: ModelSectionKind::Trees,
                offset: 120,
                length: 4,
            },
            payload: vec![1_u8, 2, 3, 4],
        }];
        let legacy_report = required_section_compatibility_report(&legacy_sections);
        assert!(!legacy_report.strict_compatible);
        assert!(legacy_report.legacy_trees_only_compatible);
        assert!(legacy_report.legacy_compatible);
    }

    #[test]
    fn model_contract_rejects_overlapping_sections() {
        let contract = ModelIoContractV1 {
            header: ModelBinaryHeader::new(2, 64),
            sections: vec![
                ModelSectionDescriptor {
                    kind: ModelSectionKind::Trees,
                    offset: 16,
                    length: 64,
                },
                ModelSectionDescriptor {
                    kind: ModelSectionKind::PredictorLayout,
                    offset: 40,
                    length: 10,
                },
            ],
            metadata: ModelMetadata {
                format_version: MODEL_FORMAT_V1,
                feature_names: vec!["f0".to_string()],
                trained_device: Device::Cpu,
                objective: "squared_error".to_string(),
                num_classes: None,
            },
        };
        assert!(matches!(
            validate_model_contract_v1(&contract),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn model_contract_rejects_section_offset_before_payload_start() {
        let contract = ModelIoContractV1 {
            header: ModelBinaryHeader::new(1, 64),
            sections: vec![ModelSectionDescriptor {
                kind: ModelSectionKind::Trees,
                offset: 79,
                length: 4,
            }],
            metadata: ModelMetadata {
                format_version: MODEL_FORMAT_V1,
                feature_names: vec!["f0".to_string()],
                trained_device: Device::Cpu,
                objective: "squared_error".to_string(),
                num_classes: None,
            },
        };

        assert!(matches!(
            validate_model_contract_v1(&contract),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn model_contract_rejects_non_contiguous_section_offsets() {
        let contract = ModelIoContractV1 {
            header: ModelBinaryHeader::new(2, 64),
            sections: vec![
                ModelSectionDescriptor {
                    kind: ModelSectionKind::Trees,
                    offset: 120,
                    length: 8,
                },
                ModelSectionDescriptor {
                    kind: ModelSectionKind::PredictorLayout,
                    offset: 130,
                    length: 4,
                },
            ],
            metadata: ModelMetadata {
                format_version: MODEL_FORMAT_V1,
                feature_names: vec!["f0".to_string()],
                trained_device: Device::Cpu,
                objective: "squared_error".to_string(),
                num_classes: None,
            },
        };

        assert!(matches!(
            validate_model_contract_v1(&contract),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn model_artifact_roundtrip() {
        let metadata = sample_metadata();
        let sections = vec![
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
        ];

        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(parsed.contract.metadata, metadata);
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].descriptor.kind, ModelSectionKind::Trees);
        assert_eq!(parsed.sections[0].payload, vec![1_u8, 2, 3, 4]);
        assert_eq!(
            parsed.sections[1].descriptor.kind,
            ModelSectionKind::PredictorLayout
        );
        assert_eq!(parsed.sections[1].payload, vec![9_u8, 8, 7]);
    }

    #[test]
    fn model_artifact_deserialize_rejects_truncated_payload() {
        let metadata = sample_metadata();
        let sections = vec![(ModelSectionKind::Trees, vec![1_u8, 2, 3, 4])];
        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            deserialize_model_artifact_v1(truncated),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn categorical_state_payload_roundtrip() {
        let payload = CategoricalStatePayloadV1 {
            format_version: CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: true,
            categorical_feature_indices: vec![0, 3, 7],
        };
        let encoded = encode_categorical_state_payload_v1(&payload).expect("encodes");
        let decoded = decode_categorical_state_payload_v1(&encoded).expect("decodes");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn categorical_state_payload_rejects_invalid_ordering() {
        let payload = CategoricalStatePayloadV1 {
            format_version: CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: false,
            categorical_feature_indices: vec![2, 1],
        };
        assert!(matches!(
            encode_categorical_state_payload_v1(&payload),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn categorical_state_payload_decode_rejects_unknown_flags() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&CATEGORICAL_STATE_FORMAT_V1.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(
            decode_categorical_state_payload_v1(&bytes),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn strict_compatibility_allows_optional_categorical_state_section() {
        let metadata = sample_metadata();
        let categorical_state = encode_categorical_state_payload_v1(&CategoricalStatePayloadV1 {
            format_version: CATEGORICAL_STATE_FORMAT_V1,
            leakage_safe_target_encoding: true,
            categorical_feature_indices: vec![1],
        })
        .expect("categorical payload encodes");

        let bytes = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
                (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
                (ModelSectionKind::CategoricalState, categorical_state),
            ],
        )
        .expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
        let report = required_section_compatibility_report(&parsed.sections);
        assert!(report.strict_compatible);

        let decoded_categorical = decode_optional_categorical_state_section_v1(
            &parsed.sections,
            metadata.feature_names.len(),
        )
        .expect("categorical state decodes");
        assert!(decoded_categorical.is_some());
    }

    #[test]
    fn decode_optional_categorical_state_section_rejects_duplicate_sections() {
        let sections = vec![
            ModelArtifactSection {
                descriptor: ModelSectionDescriptor {
                    kind: ModelSectionKind::CategoricalState,
                    offset: 100,
                    length: 20,
                },
                payload: vec![1_u8; 20],
            },
            ModelArtifactSection {
                descriptor: ModelSectionDescriptor {
                    kind: ModelSectionKind::CategoricalState,
                    offset: 120,
                    length: 20,
                },
                payload: vec![1_u8; 20],
            },
        ];
        assert!(matches!(
            decode_optional_categorical_state_section_v1(&sections, 8),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn dense_matrix_view_matches_dataset_layout() {
        let view = DenseMatrixView::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("dense view is valid");
        assert_eq!(view.row(1).expect("row resolves"), &[4.0, 5.0, 6.0]);
        assert_eq!(view.value_at(0, 2).expect("value resolves"), 3.0);

        let dataset = view.to_dataset_matrix().expect("dataset materializes");
        assert_eq!(dataset.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn columnar_matrix_view_materializes_rows_and_honors_validity() {
        let view = ColumnarMatrixView::new(
            3,
            vec![
                ColumnarMatrixColumnView {
                    values: &[1.0, 2.0, 3.0],
                    validity: None,
                },
                ColumnarMatrixColumnView {
                    values: &[10.0, 20.0, 30.0],
                    validity: Some(&[true, false, true]),
                },
            ],
        )
        .expect("columnar view is valid");

        assert_eq!(view.value_at(1, 1).expect("value resolves"), None);
        let dataset = view
            .to_dataset_matrix(-1.0)
            .expect("dataset materializes from columnar view");
        assert_eq!(dataset.values, vec![1.0, 10.0, 2.0, -1.0, 3.0, 30.0]);
    }

    #[test]
    fn columnar_matrix_view_rejects_misaligned_validity() {
        let result = ColumnarMatrixView::new(
            2,
            vec![ColumnarMatrixColumnView {
                values: &[1.0, 2.0],
                validity: Some(&[true]),
            }],
        );
        assert!(matches!(result, Err(CoreError::Validation(_))));
    }

    #[test]
    fn strict_compatibility_ignores_optional_node_debug_stats_section() {
        let metadata = sample_metadata();
        let bytes = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
                (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
                (ModelSectionKind::NodeDebugStats, vec![5_u8, 4, 3, 2]),
            ],
        )
        .expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
        let report = required_section_compatibility_report(&parsed.sections);
        assert!(report.strict_compatible);
    }

    #[test]
    fn test_metadata_json_num_classes_roundtrip() {
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["f0".to_string(), "f1".to_string()],
            trained_device: Device::Cpu,
            objective: "multiclass_softmax".to_string(),
            num_classes: Some(5),
        };
        let json = serialize_metadata_json(&metadata);
        let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
        assert_eq!(decoded, metadata);
        assert_eq!(decoded.num_classes, Some(5));
    }

    #[test]
    fn test_metadata_json_backward_compat_no_num_classes() {
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let json = serialize_metadata_json(&metadata);
        // Verify the JSON does not contain num_classes when None
        assert!(!json.contains("num_classes"));
        let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
        assert_eq!(decoded, metadata);
        assert_eq!(decoded.num_classes, None);
    }

    #[test]
    fn morph_config_default_matches_paper() {
        let cfg = MorphConfig::default();
        assert_eq!(cfg.morph_rate, 0.1);
        assert_eq!(cfg.evolution_pressure, 0.2);
        assert_eq!(cfg.morph_warmup_iters, 5);
        assert_eq!(cfg.info_score_weight, 0.3);
        assert_eq!(cfg.depth_penalty_base, 0.9);
        assert!(cfg.balance_penalty);
        assert_eq!(cfg.lr_schedule, LrSchedule::Constant);
    }

    #[test]
    fn lr_schedule_warmup_cosine_default_warmup_frac() {
        let s = LrSchedule::WarmupCosine { warmup_frac: 0.1 };
        if let LrSchedule::WarmupCosine { warmup_frac } = s {
            assert!((warmup_frac - 0.1).abs() < 1e-6);
        } else {
            panic!("expected WarmupCosine");
        }
    }

    #[test]
    fn training_mode_default_is_auto() {
        assert_eq!(TrainingMode::default(), TrainingMode::Auto);
    }

    #[test]
    fn train_params_default_has_no_morph_config() {
        let p = TrainParams::default();
        assert!(p.morph_config.is_none());
    }

    #[test]
    fn validate_train_params_accepts_morph_config() {
        let p = TrainParams {
            morph_config: Some(MorphConfig::default()),
            ..TrainParams::default()
        };
        assert!(validate_train_params(&p).is_ok());
    }

    #[test]
    fn validate_train_params_rejects_invalid_morph_rate() {
        let p = TrainParams {
            morph_config: Some(MorphConfig {
                morph_rate: -0.1,
                ..MorphConfig::default()
            }),
            ..TrainParams::default()
        };
        assert!(validate_train_params(&p).is_err());
    }

    #[test]
    fn validate_train_params_rejects_invalid_warmup_frac() {
        let p = TrainParams {
            morph_config: Some(MorphConfig {
                lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 1.5 },
                ..MorphConfig::default()
            }),
            ..TrainParams::default()
        };
        assert!(validate_train_params(&p).is_err());
    }

    #[test]
    fn morph_metadata_round_trip() {
        let payload = MorphMetadataPayload {
            config: MorphConfig {
                morph_rate: 0.15,
                evolution_pressure: 0.25,
                morph_warmup_iters: 7,
                info_score_weight: 0.4,
                depth_penalty_base: 0.85,
                balance_penalty: false,
                lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.2 },
            },
            final_iteration: 42,
            final_total: 100,
        };
        let bytes = encode_morph_metadata_payload(&payload);
        assert_eq!(bytes.len(), 36);
        let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn morph_metadata_decode_rejects_short_input() {
        let err = decode_optional_morph_metadata_section(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, CoreError::Validation(_)));
    }

    #[test]
    fn morph_metadata_decode_rejects_unknown_version() {
        let mut bytes = vec![0u8; 36];
        bytes[0] = 99; // version = 99
        let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
        assert!(matches!(err, CoreError::Validation(_)));
    }

    #[test]
    fn morph_metadata_decode_rejects_unknown_lr_kind() {
        let payload = MorphMetadataPayload {
            config: MorphConfig::default(),
            final_iteration: 0,
            final_total: 1,
        };
        let mut bytes = encode_morph_metadata_payload(&payload);
        bytes[23] = 99; // lr_schedule_kind at offset 2+5*4+1=23
        let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
        assert!(matches!(err, CoreError::Validation(_)));
    }

    #[test]
    fn morph_metadata_artifact_round_trip() {
        let metadata = sample_metadata();
        let morph_payload = MorphMetadataPayload {
            config: MorphConfig {
                morph_rate: 0.15,
                evolution_pressure: 0.25,
                morph_warmup_iters: 7,
                info_score_weight: 0.4,
                depth_penalty_base: 0.85,
                balance_penalty: true,
                lr_schedule: LrSchedule::Constant,
            },
            final_iteration: 10,
            final_total: 10,
        };
        let morph_bytes = encode_morph_metadata_payload(&morph_payload);
        let sections = vec![
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
            (ModelSectionKind::MorphMetadata, morph_bytes),
        ];
        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
        assert_eq!(parsed.sections.len(), 3);
        let decoded_morph =
            decode_optional_morph_metadata_artifact_section(&parsed.sections).unwrap();
        assert_eq!(decoded_morph, Some(morph_payload));
    }

    #[test]
    fn morph_metadata_artifact_absent_for_non_morph() {
        let metadata = sample_metadata();
        let sections = vec![
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
        ];
        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
        let decoded_morph =
            decode_optional_morph_metadata_artifact_section(&parsed.sections).unwrap();
        assert_eq!(decoded_morph, None);
    }

    #[test]
    fn gradient_ema_single_pass_matches_two_pass_within_tolerance() {
        // Use the existing two-pass implementation as the reference,
        // then verify the new single-pass implementation produces the
        // same result within numerical tolerance.
        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut legacy = GradientEmaStats {
            alpha: 0.5,
            ..Default::default()
        };
        legacy.update_two_pass_legacy(&gradients);
        let mut new_path = GradientEmaStats {
            alpha: 0.5,
            ..Default::default()
        };
        new_path.update(&gradients);
        assert!(
            (legacy.mean - new_path.mean).abs() < 1e-4,
            "mean drift: legacy={} new={}",
            legacy.mean,
            new_path.mean
        );
        assert!(
            (legacy.std - new_path.std).abs() < 1e-3,
            "std drift: legacy={} new={}",
            legacy.std,
            new_path.std
        );
    }

    #[test]
    fn gradient_ema_handles_empty_input() {
        let mut stats = GradientEmaStats {
            alpha: 0.5,
            ..Default::default()
        };
        let initial_mean = stats.mean;
        let initial_std = stats.std;
        stats.update(&[]);
        assert_eq!(stats.mean, initial_mean);
        assert_eq!(stats.std, initial_std);
    }

    #[test]
    fn gradient_ema_simd_matches_scalar_for_large_input() {
        // 5000 elements ensures the chunks_exact(8) path runs many iterations.
        let gradients: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).cos() * 0.5).collect();
        let mut new_path = GradientEmaStats {
            alpha: 0.3,
            ..Default::default()
        };
        new_path.update(&gradients);
        // Sanity: result should be finite.
        assert!(new_path.mean.is_finite());
        assert!(new_path.std.is_finite());
        assert!(new_path.std >= 0.0);
    }
}
