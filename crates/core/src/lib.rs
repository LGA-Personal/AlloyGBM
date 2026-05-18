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

/// Leaf representation strategy for tree models.
///
/// `Constant` (default) is identical to current behavior: each leaf stores a
/// single scalar output `f_s = -lr * Σg / (Σh + λ)`.
///
/// `Linear` replaces the scalar with a small ridge-regression model
/// `f_s(x) = b_s + Σ_j α_j x_{k_j}` fit analytically via the closed-form
/// Newton step `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.  Feature regressors are chosen
/// incrementally as the tree grows (inheriting the parent's set plus the
/// current split feature, capped at `min(8, max_depth)`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeafModelKind {
    /// Single scalar leaf value (current default behavior).
    #[default]
    Constant,
    /// Piecewise-linear leaf model fit by ridge regression.
    Linear,
}

/// Scalar leaf solver strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeafSolverKind {
    /// Standard Newton leaf solve using empirical gradient/hessian sums.
    #[default]
    Standard,
    /// Distributionally robust leaf solve over within-leaf gradient dispersion.
    Dro,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralizationKind {
    None,
    PreTarget,
    PerRoundGradient,
    SplitPenalty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorNeutralizationConfig {
    pub kind: NeutralizationKind,
    pub ridge_lambda: f32,
    pub split_penalty: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorExposureMatrix {
    pub row_count: usize,
    pub factor_count: usize,
    pub values: Vec<f32>,
}

impl FactorExposureMatrix {
    pub fn new(row_count: usize, factor_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        if row_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures row_count must be greater than 0".to_string(),
            ));
        }
        if factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = row_count.checked_mul(factor_count).ok_or_else(|| {
            CoreError::Validation("factor_exposures row_count * factor_count overflow".to_string())
        })?;
        if values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                values.len(),
                expected_len
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::Validation(
                "factor_exposures must contain only finite values".to_string(),
            ));
        }
        Ok(Self {
            row_count,
            factor_count,
            values,
        })
    }

    pub fn row(&self, row_index: usize) -> CoreResult<&[f32]> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "factor_exposures row_index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        if self.factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = self
            .row_count
            .checked_mul(self.factor_count)
            .ok_or_else(|| {
                CoreError::Validation(
                    "factor_exposures row_count * factor_count overflow".to_string(),
                )
            })?;
        if self.values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                self.values.len(),
                expected_len
            )));
        }
        let start = row_index * self.factor_count;
        Ok(&self.values[start..start + self.factor_count])
    }
}

/// Uncertainty metric used by the DRO leaf solver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DroMetric {
    /// Wasserstein-inspired uncertainty radius over leaf gradient dispersion.
    #[default]
    Wasserstein,
}

/// Configuration for the fast DRO-style scalar leaf solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroConfig {
    /// Non-negative robustness radius. `0.0` is exactly standard leaf behavior.
    pub radius: f32,
    /// Uncertainty metric for interpreting the radius.
    pub metric: DroMetric,
}

impl Default for DroConfig {
    fn default() -> Self {
        Self {
            radius: 0.05,
            metric: DroMetric::Wasserstein,
        }
    }
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

/// DART tree-weight normalization policy.  Mirrors LightGBM/MART
/// terminology — see `crates/engine/src/dart.rs` for how the policy
/// affects per-round leaf-weight rescaling.
///
/// * `Tree`: each new tree is scaled by `1 / (K + 1)` and each of the
///   K dropped trees by `K / (K + 1)`.  Keeps the cumulative ensemble
///   prediction unbiased after the per-round dropout swap.
/// * `Forest`: applies a forest-wide rescaling factor; commonly used
///   when the user expects DART to behave more like a random-forest
///   ensemble than a boosted ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DartNormalize {
    Tree,
    Forest,
}

/// DART dropout sampling strategy.
///
/// * `Uniform`: each of the existing K trees is dropped with
///   probability `drop_rate`, capped at `max_drop`.
/// * `Weighted`: drop probability is weighted by per-tree contribution
///   magnitude so larger-impact trees are more likely to be dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DartSampleType {
    Uniform,
    Weighted,
}

/// Per-round boosting strategy.
///
/// * `Standard` (default): byte-identical to v0.7.5 behaviour —
///   uniform row subsampling under `row_subsample`, no per-round
///   tree dropout.
/// * `Goss { top_rate, other_rate }`: gradient-based one-side
///   sampling — keep the top `top_rate` rows by `|gradient|`, sample
///   `other_rate` from the rest, and amplify the small-gradient rows
///   by `(1 - top_rate) / other_rate` to maintain unbiased gradient
///   sums.  See `crates/engine/src/sampling.rs`.
/// * `Dart { drop_rate, max_drop, normalize_type, sample_type }`:
///   per-round tree dropout — drop K existing trees before computing
///   gradients, fit a new tree, then rescale per `normalize_type`.
///   Requires per-stump `tree_weight: f32` in the artifact (back-compat:
///   missing field defaults to 1.0).  See `crates/engine/src/dart.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoostingMode {
    Standard,
    Goss {
        top_rate: f32,
        other_rate: f32,
    },
    Dart {
        drop_rate: f32,
        max_drop: usize,
        normalize_type: DartNormalize,
        sample_type: DartSampleType,
    },
}

impl BoostingMode {
    /// Stable string label for artifact metadata and Python-side error
    /// messages.  Matches the `boosting_mode=` ctor strings on the
    /// estimators.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Goss { .. } => "goss",
            Self::Dart { .. } => "dart",
        }
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
    /// Interaction constraints (LightGBM-compatible semantics).  Each inner
    /// `Vec` is a group of feature indices that are allowed to co-occur on
    /// any root-to-leaf path.  Features that don't appear in any group are
    /// unconstrained and may be used freely alongside any group.  Empty
    /// outer `Vec` means no constraints — equivalent to the v0.7.0
    /// behaviour.  Limit: up to 64 groups per fit (a u64 bitset tracks the
    /// active set per node).
    pub interaction_constraints: Vec<Vec<u32>>,
    /// Maximum number of leaves per tree. None means depth-limited only.
    pub max_leaves: Option<usize>,
    /// Tree growth strategy: level-wise (default) or leaf-wise (best-first).
    pub tree_growth: TreeGrowth,
    /// MorphBoost-inspired training profile config. `None` = non-morph (current behavior).
    pub morph_config: Option<MorphConfig>,
    /// Leaf representation strategy.  `Constant` (default) preserves all existing
    /// behaviour.  `Linear` enables piecewise-linear leaves fitted by ridge regression.
    pub leaf_model: LeafModelKind,
    /// Scalar leaf solver strategy. `Standard` preserves existing behavior.
    pub leaf_solver: LeafSolverKind,
    /// Configuration for `leaf_solver == Dro`.
    pub dro_config: Option<DroConfig>,
    pub neutralization_config: Option<FactorNeutralizationConfig>,
    /// v0.8.0: per-round boosting strategy.  Default `Standard`
    /// preserves v0.7.5 behaviour exactly.  See [`BoostingMode`] for
    /// the GOSS / DART semantics.
    pub boosting_mode: BoostingMode,
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
            interaction_constraints: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
            leaf_model: LeafModelKind::Constant,
            leaf_solver: LeafSolverKind::Standard,
            dro_config: None,
            neutralization_config: None,
            boosting_mode: BoostingMode::Standard,
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
    pub factor_exposures: Option<FactorExposureMatrix>,
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

pub fn leaf_effective_gradient(
    grad_sum: f32,
    grad_sq_sum: f32,
    row_count: u32,
    l1_alpha: f32,
    dro_config: Option<&DroConfig>,
) -> f32 {
    let mut threshold = l1_alpha.max(0.0);
    if let Some(cfg) = dro_config
        && cfg.radius > 0.0
    {
        let n = row_count.max(1) as f64;
        let mean = f64::from(grad_sum) / n;
        let variance = (f64::from(grad_sq_sum) / n - mean * mean).max(0.0);
        threshold += (f64::from(cfg.radius) * n.sqrt() * variance.sqrt()) as f32;
    }
    if grad_sum > threshold {
        grad_sum - threshold
    } else if grad_sum < -threshold {
        grad_sum + threshold
    } else {
        0.0
    }
}

pub fn leaf_gain_term(
    grad_sum: f32,
    hess_sum: f32,
    grad_sq_sum: f32,
    row_count: u32,
    l1_alpha: f32,
    l2_lambda: f32,
    dro_config: Option<&DroConfig>,
) -> f32 {
    const EPSILON: f32 = 1e-6;
    let effective = leaf_effective_gradient(grad_sum, grad_sq_sum, row_count, l1_alpha, dro_config);
    0.5 * effective * effective / (hess_sum + l2_lambda + EPSILON)
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
    pub grad_sq_sum: f32,
    pub row_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBin {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub grad_sq_sum: f32,
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
                bin.grad_sq_sum = 0.0;
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
                        grad_sq_sum: 0.0,
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

// ── Piecewise-linear leaf histogram types ────────────────────────────────────

/// Maximum number of regressor features per leaf for piecewise-linear trees.
/// Matches `min(8, max_depth)` at run time, but we always pad to this constant
/// so that bin layouts are fixed-size and cache-friendly.
pub const MAX_PL_REGRESSORS: usize = 8;

/// Number of `XᵀHX` matrix entries stored per histogram bin: a padded 8×8 = 64
/// stride-8 layout (changed from a 36-entry compacted upper-triangle in v0.5.0
/// so that all matrix operations map cleanly to `wide::f32x8` lanes with no
/// scalar tail). The lower-triangle entries stay zero (`XᵀHX` is symmetric and
/// only the upper triangle is populated by `pl_histogram`); SIMD operations on
/// those zero slots are harmless.
pub const MAX_PL_MATRIX_ENTRIES: usize = MAX_PL_REGRESSORS * MAX_PL_REGRESSORS;

/// A single histogram bin for a piecewise-linear (PL) leaf model.
///
/// Stores the `(Xᵀg, XᵀHX)` statistics needed for the closed-form ridge-
/// regression leaf-weight solve `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
///
/// Only the first `d` entries of `xtg` and only the upper-triangle of the
/// `d × d` block of `xt_hx` are written by the histogram builder; the rest of
/// each array is zero padding (so SIMD operations covering the full storage
/// produce mathematically correct results).  `d` is recorded in the parent
/// [`LinearHistogramBundle`].
///
/// `xt_hx` uses a stride-8 row-major layout: `xt_hx[j * MAX_PL_REGRESSORS + k]`
/// holds `Σ h_i x_{i,j} x_{i,k}` for `j ≤ k < d`.  See [`pl_matrix_index`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearHistogramBin {
    /// Sum of gradients for samples in this bin.
    pub grad_sum: f32,
    /// Sum of hessians for samples in this bin.
    pub hess_sum: f32,
    /// Number of samples in this bin.
    pub count: u32,
    /// `Xᵀg` vector: `xtg[j] = Σ_{i in bin} g_i * x_{i, regressor_j}` for j = 0..d.
    pub xtg: [f32; MAX_PL_REGRESSORS],
    /// `XᵀHX` stride-8 row-major: `xt_hx[j * MAX_PL_REGRESSORS + k]` for j ≤ k < d.
    pub xt_hx: [f32; MAX_PL_MATRIX_ENTRIES],
}

impl Default for LinearHistogramBin {
    fn default() -> Self {
        Self {
            grad_sum: 0.0,
            hess_sum: 0.0,
            count: 0,
            xtg: [0.0; MAX_PL_REGRESSORS],
            xt_hx: [0.0; MAX_PL_MATRIX_ENTRIES],
        }
    }
}

/// Return the flat index into the stride-8 row-major `xt_hx` array for element
/// `(j, k)`.
///
/// The histogram builder writes only the upper triangle (`j ≤ k < d`); lower-
/// triangle entries stay zero.  Callers may also reference `(j, k)` pairs in
/// the lower triangle for symmetric reads — those slots are zero and a SIMD
/// reduction over the full row produces the correct result.
///
/// Panics in debug builds if either index is out of range.
#[inline]
pub fn pl_matrix_index(j: usize, k: usize) -> usize {
    debug_assert!(j < MAX_PL_REGRESSORS, "j ({j}) must be < MAX_PL_REGRESSORS");
    debug_assert!(k < MAX_PL_REGRESSORS, "k ({k}) must be < MAX_PL_REGRESSORS");
    j * MAX_PL_REGRESSORS + k
}

/// Per-feature histogram for piecewise-linear leaf statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFeatureHistogram {
    pub feature_index: u32,
    pub bins: Vec<LinearHistogramBin>,
}

/// Bundle of per-feature PL histograms for a single tree node.
///
/// `num_regressors` tells how many entries in each bin's `xtg` / `xt_hx`
/// fields are valid; the rest are zero padding.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearHistogramBundle {
    pub node_id: u32,
    /// Number of active regressors for this node (1 ≤ d ≤ MAX_PL_REGRESSORS).
    pub num_regressors: usize,
    /// Which feature indices are the regressors for this node, length = num_regressors.
    pub regressor_features: Vec<u32>,
    pub feature_histograms: Vec<LinearFeatureHistogram>,
}

impl LinearHistogramBundle {
    /// Zero all bin values in-place without deallocating.
    pub fn reset(&mut self, node_id: u32) {
        self.node_id = node_id;
        for fh in &mut self.feature_histograms {
            for bin in &mut fh.bins {
                *bin = LinearHistogramBin::default();
            }
        }
    }

    /// Create a pre-allocated, zeroed linear histogram bundle.
    pub fn new_zeroed(
        node_id: u32,
        feature_indices: &[u32],
        bin_count: usize,
        num_regressors: usize,
        regressor_features: Vec<u32>,
    ) -> Self {
        debug_assert!(
            num_regressors <= MAX_PL_REGRESSORS,
            "num_regressors ({num_regressors}) exceeds MAX_PL_REGRESSORS"
        );
        let feature_histograms = feature_indices
            .iter()
            .map(|&fi| LinearFeatureHistogram {
                feature_index: fi,
                bins: vec![LinearHistogramBin::default(); bin_count],
            })
            .collect();
        Self {
            node_id,
            num_regressors,
            regressor_features,
            feature_histograms,
        }
    }
}

/// Compute the larger-child linear histogram bundle using the subtraction trick.
///
/// Given `parent` and the `smaller` child's bundle (both with the same node
/// layout), returns the `larger` child bundle via element-wise subtraction:
/// `larger[f][b] = parent[f][b] - smaller[f][b]` for all fields.
///
/// Both bundles must have the same number of features (in the same order) and
/// the same number of bins per feature.
pub fn subtract_linear_histogram_bundle(
    parent: &LinearHistogramBundle,
    smaller: &LinearHistogramBundle,
) -> LinearHistogramBundle {
    debug_assert_eq!(
        parent.feature_histograms.len(),
        smaller.feature_histograms.len(),
        "feature count mismatch in linear histogram subtraction"
    );
    debug_assert_eq!(
        parent.num_regressors, smaller.num_regressors,
        "num_regressors mismatch in linear histogram subtraction"
    );
    let feature_histograms = parent
        .feature_histograms
        .iter()
        .zip(smaller.feature_histograms.iter())
        .map(|(pfh, sfh)| {
            debug_assert_eq!(pfh.bins.len(), sfh.bins.len());
            let bins = pfh
                .bins
                .iter()
                .zip(sfh.bins.iter())
                .map(|(pb, sb)| {
                    // Operate on all `MAX_PL_REGRESSORS` / `MAX_PL_MATRIX_ENTRIES`
                    // slots — the histogram builder may populate both triangles
                    // of `xt_hx` (full 8×8 outer product) and unused entries
                    // are zero in both `pb` and `sb`, so subtracting all
                    // entries is correct and matches the SIMD-friendly
                    // stride-8 layout.
                    let mut xtg = [0.0f32; MAX_PL_REGRESSORS];
                    for (j, xtg_j) in xtg.iter_mut().enumerate() {
                        *xtg_j = pb.xtg[j] - sb.xtg[j];
                    }
                    let mut xt_hx = [0.0f32; MAX_PL_MATRIX_ENTRIES];
                    for (i, slot) in xt_hx.iter_mut().enumerate() {
                        *slot = pb.xt_hx[i] - sb.xt_hx[i];
                    }
                    LinearHistogramBin {
                        grad_sum: pb.grad_sum - sb.grad_sum,
                        hess_sum: pb.hess_sum - sb.hess_sum,
                        count: pb.count - sb.count,
                        xtg,
                        xt_hx,
                    }
                })
                .collect();
            LinearFeatureHistogram {
                feature_index: pfh.feature_index,
                bins,
            }
        })
        .collect();
    LinearHistogramBundle {
        node_id: smaller.node_id, // larger child gets a different node_id assigned by caller
        num_regressors: parent.num_regressors,
        regressor_features: parent.regressor_features.clone(),
        feature_histograms,
    }
}

/// A linear leaf model: `f_s(x) = intercept + Σ_j weights[j] * x[regressor_features[j]]`.
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
}

impl LinearLeaf {
    /// Evaluate the leaf model for one row of raw (float) feature data.
    ///
    /// `raw_features` is the full flat row-major feature matrix; `row_offset` is
    /// `row_index * feature_count`.
    #[inline]
    pub fn eval(&self, raw_features: &[f32], row_offset: usize) -> f32 {
        let mut val = self.intercept;
        for (w, &feat) in self.weights.iter().zip(self.regressor_features.iter()) {
            let idx = row_offset + feat as usize;
            if idx < raw_features.len() {
                val += w * raw_features[idx];
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
    /// Per-stump linear leaf coefficients (intercept + weights + regressor features).
    /// Written alongside the Trees section for `leaf_model="linear"` models.
    LinearLeafCoefficients,
    /// Metadata for DRO-style scalar leaf solving.
    DroMetadata,
    /// Global per-feature training-set means.  Optional section written by
    /// piecewise-linear (`leaf_model="linear"`) artifacts so that SHAP can
    /// compute interventional attributions for linear leaves without needing
    /// the original training data.  Length matches `metadata.feature_names`.
    FeatureBaseline,
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
            Self::LinearLeafCoefficients => 9,
            Self::DroMetadata => 10,
            Self::FeatureBaseline => 11,
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
            9 => Self::LinearLeafCoefficients,
            10 => Self::DroMetadata,
            11 => Self::FeatureBaseline,
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
///
/// **Version history.**
///
/// * v1 (v0.4.0+): `config` + `final_iteration` + `final_total`.  Fixed
///   36-byte payload.
/// * v2 (v0.7.3+): appends a length-prefixed `ema_stats: Vec<GradientEmaStats>`
///   so MorphBoost warm-starts can resume with the EMA state from the
///   previous fit rather than restarting it cold.  Legacy v1 artifacts
///   decode with `ema_stats = Vec::new()` and the warm-start path
///   falls back to a cold EMA (legacy v0.7.1/v0.7.2 behavior).
#[derive(Debug, Clone, PartialEq)]
pub struct MorphMetadataPayload {
    pub config: MorphConfig,
    pub final_iteration: u32,
    pub final_total: u32,
    /// EMA snapshot captured at training-finalize time.  Empty when the
    /// payload was decoded from a pre-v0.7.3 (version 1) artifact, in
    /// which case warm-start initializes the EMA cold.  Indexed by
    /// class for multiclass models (length 1 for single-output).
    pub ema_stats: Vec<GradientEmaStats>,
}

pub fn encode_morph_metadata_payload(payload: &MorphMetadataPayload) -> Vec<u8> {
    // v2 layout: 36 bytes header (same as v1) + 4 bytes count +
    // 12 bytes per GradientEmaStats (mean, std, alpha as little-endian f32).
    let ema_section_len = 4 + payload.ema_stats.len() * 12;
    let mut buf = Vec::with_capacity(36 + ema_section_len);
    buf.extend_from_slice(&2u16.to_le_bytes());
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
    // v2 EMA tail.
    let ema_count = payload.ema_stats.len() as u32;
    buf.extend_from_slice(&ema_count.to_le_bytes());
    for stats in &payload.ema_stats {
        buf.extend_from_slice(&stats.mean.to_le_bytes());
        buf.extend_from_slice(&stats.std.to_le_bytes());
        buf.extend_from_slice(&stats.alpha.to_le_bytes());
    }
    buf
}

pub fn decode_optional_morph_metadata_section(bytes: &[u8]) -> CoreResult<MorphMetadataPayload> {
    if bytes.len() < 36 {
        return Err(CoreError::Validation(
            "morph metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 && version != 2 {
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
    o += 4;
    let lr_schedule = match lr_kind {
        0 => LrSchedule::Constant,
        1 => LrSchedule::WarmupCosine { warmup_frac },
        _ => {
            return Err(CoreError::Validation(format!(
                "unknown lr_schedule kind: {lr_kind}"
            )));
        }
    };
    // v2 tail: optional EMA stats.  v1 artifacts have no tail and
    // decode with `ema_stats = Vec::new()`.
    let ema_stats = if version >= 2 && o + 4 <= bytes.len() {
        let count =
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        o += 4;
        let expected_tail = count.checked_mul(12).ok_or_else(|| {
            CoreError::Validation("morph metadata ema count overflow".to_string())
        })?;
        if o + expected_tail > bytes.len() {
            return Err(CoreError::Validation(format!(
                "morph metadata ema tail truncated: expected {} bytes after header, got {}",
                expected_tail,
                bytes.len() - o
            )));
        }
        let mut stats = Vec::with_capacity(count);
        for _ in 0..count {
            let mean = read_f32!();
            let std = read_f32!();
            let alpha = read_f32!();
            stats.push(GradientEmaStats { mean, std, alpha });
        }
        stats
    } else {
        Vec::new()
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
        ema_stats,
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

// ── DRO leaf-solver metadata section ────────────────────────────────────────

/// Optional artifact section recording the DRO leaf solver configuration.
/// Metadata only — prediction uses baked scalar leaf values.
#[derive(Debug, Clone, PartialEq)]
pub struct DroMetadataPayload {
    pub config: DroConfig,
}

pub fn encode_dro_metadata_payload(payload: &DroMetadataPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(7);
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&payload.config.radius.to_le_bytes());
    buf.push(match payload.config.metric {
        DroMetric::Wasserstein => 0,
    });
    buf
}

pub fn decode_dro_metadata_payload(bytes: &[u8]) -> CoreResult<DroMetadataPayload> {
    if bytes.len() < 7 {
        return Err(CoreError::Validation(
            "dro metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported dro metadata version: {version}"
        )));
    }
    let radius = f32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    if !radius.is_finite() || radius < 0.0 {
        return Err(CoreError::Validation(
            "dro metadata radius must be finite and >= 0".to_string(),
        ));
    }
    let metric = match bytes[6] {
        0 => DroMetric::Wasserstein,
        other => {
            return Err(CoreError::Validation(format!(
                "unsupported dro metadata metric kind: {other}"
            )));
        }
    };
    Ok(DroMetadataPayload {
        config: DroConfig { radius, metric },
    })
}

pub fn decode_optional_dro_metadata_artifact_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<DroMetadataPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::DroMetadata)? else {
        return Ok(None);
    };
    Ok(Some(decode_dro_metadata_payload(&section.payload)?))
}

// ── Linear-leaf coefficients section ─────────────────────────────────────────

/// One stump's linear-leaf entries inside the `LinearLeafCoefficients` section.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeafEntry {
    pub stump_idx: u32,
    pub left_leaf: Option<LinearLeaf>,
    pub right_leaf: Option<LinearLeaf>,
}

/// Payload for `ModelSectionKind::LinearLeafCoefficients`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeafCoefficientsPayload {
    pub entries: Vec<LinearLeafEntry>,
}

/// Encode a `LinearLeafCoefficientsPayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=1] [u32 entry_count]
/// For each entry:
///   [u32 stump_idx] [u8 flags]
///   if flags & 1: [u8 d] [f32 intercept] [d × f32 weights] [d × u32 regressor_features]
///   if flags & 2: same for right leaf
/// ```
pub fn encode_linear_leaf_coefficients_payload(payload: &LinearLeafCoefficientsPayload) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(payload.entries.len() as u32).to_le_bytes());
    for entry in &payload.entries {
        buf.extend_from_slice(&entry.stump_idx.to_le_bytes());
        let flags: u8 =
            (entry.left_leaf.is_some() as u8) | ((entry.right_leaf.is_some() as u8) << 1);
        buf.push(flags);

        let write_leaf = |buf: &mut Vec<u8>, leaf: &LinearLeaf| {
            let d = leaf.weights.len().min(MAX_PL_REGRESSORS);
            buf.push(d as u8);
            buf.extend_from_slice(&leaf.intercept.to_le_bytes());
            for i in 0..d {
                buf.extend_from_slice(&leaf.weights[i].to_le_bytes());
            }
            for i in 0..d {
                let feat = *leaf.regressor_features.get(i).unwrap_or(&0);
                buf.extend_from_slice(&feat.to_le_bytes());
            }
        };
        if let Some(ref ll) = entry.left_leaf {
            write_leaf(&mut buf, ll);
        }
        if let Some(ref rl) = entry.right_leaf {
            write_leaf(&mut buf, rl);
        }
    }
    buf
}

/// Decode a `LinearLeafCoefficientsPayload` from raw section bytes.
pub fn decode_linear_leaf_coefficients_payload(
    bytes: &[u8],
) -> CoreResult<LinearLeafCoefficientsPayload> {
    if bytes.len() < 8 {
        return Err(CoreError::Validation(
            "linear leaf coefficients section too short".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported linear leaf coefficients version: {version}"
        )));
    }
    let entry_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut o = 8usize;
    let mut entries = Vec::with_capacity(entry_count);

    let read_u32 = |bytes: &[u8], o: &mut usize| -> CoreResult<u32> {
        if *o + 4 > bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let v = u32::from_le_bytes([bytes[*o], bytes[*o + 1], bytes[*o + 2], bytes[*o + 3]]);
        *o += 4;
        Ok(v)
    };
    let read_f32 = |bytes: &[u8], o: &mut usize| -> CoreResult<f32> {
        if *o + 4 > bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let v = f32::from_le_bytes([bytes[*o], bytes[*o + 1], bytes[*o + 2], bytes[*o + 3]]);
        *o += 4;
        Ok(v)
    };

    for _ in 0..entry_count {
        let stump_idx = read_u32(bytes, &mut o)?;
        if o >= bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let flags = bytes[o];
        o += 1;

        let read_leaf = |bytes: &[u8], o: &mut usize| -> CoreResult<LinearLeaf> {
            if *o >= bytes.len() {
                return Err(CoreError::Validation(
                    "unexpected end reading linear leaf".to_string(),
                ));
            }
            let d = bytes[*o] as usize;
            *o += 1;
            let intercept = read_f32(bytes, o)?;
            let mut weights = Vec::with_capacity(d);
            for _ in 0..d {
                weights.push(read_f32(bytes, o)?);
            }
            let mut regressor_features = Vec::with_capacity(d);
            for _ in 0..d {
                regressor_features.push(read_u32(bytes, o)?);
            }
            Ok(LinearLeaf {
                intercept,
                weights,
                regressor_features,
            })
        };

        let left_leaf = if flags & 1 != 0 {
            Some(read_leaf(bytes, &mut o)?)
        } else {
            None
        };
        let right_leaf = if flags & 2 != 0 {
            Some(read_leaf(bytes, &mut o)?)
        } else {
            None
        };
        entries.push(LinearLeafEntry {
            stump_idx,
            left_leaf,
            right_leaf,
        });
    }

    Ok(LinearLeafCoefficientsPayload { entries })
}

/// Decode an optional `LinearLeafCoefficients` section from parsed artifact sections.
/// Returns `None` if no such section exists (constant-leaf artifact).
pub fn decode_optional_linear_leaf_coefficients_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<LinearLeafCoefficientsPayload>> {
    let Some(section) =
        optional_single_section(sections, ModelSectionKind::LinearLeafCoefficients)?
    else {
        return Ok(None);
    };
    let payload = decode_linear_leaf_coefficients_payload(&section.payload)?;
    Ok(Some(payload))
}

// ── Feature baseline section ─────────────────────────────────────────────────

/// Payload for `ModelSectionKind::FeatureBaseline`.
///
/// Stores the global (training-set marginal) mean for each feature.  Length
/// matches `ModelMetadata::feature_names`.  Used by SHAP for piecewise-linear
/// leaves so that linear-leaf contributions can be decomposed into a
/// path-attributed expected value plus per-feature deviations
/// `wj * (xj - feature_means[j])`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBaselinePayload {
    pub feature_means: Vec<f32>,
}

/// Encode a `FeatureBaselinePayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=1] [u32 feature_count] [feature_count × f32 means]
/// ```
pub fn encode_feature_baseline_payload(payload: &FeatureBaselinePayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + payload.feature_means.len() * 4);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(payload.feature_means.len() as u32).to_le_bytes());
    for m in &payload.feature_means {
        buf.extend_from_slice(&m.to_le_bytes());
    }
    buf
}

/// Decode a `FeatureBaselinePayload` from raw section bytes.
pub fn decode_feature_baseline_payload(bytes: &[u8]) -> CoreResult<FeatureBaselinePayload> {
    if bytes.len() < 8 {
        return Err(CoreError::Validation(
            "feature baseline section too short".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported feature baseline version: {version}"
        )));
    }
    let feature_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + feature_count * 4;
    if bytes.len() < expected {
        return Err(CoreError::Validation(format!(
            "feature baseline section too short: need {expected} bytes, got {}",
            bytes.len()
        )));
    }
    let mut feature_means = Vec::with_capacity(feature_count);
    let mut o = 8usize;
    for _ in 0..feature_count {
        let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        feature_means.push(v);
        o += 4;
    }
    Ok(FeatureBaselinePayload { feature_means })
}

/// Decode an optional `FeatureBaseline` section from parsed artifact sections.
/// Returns `None` if no such section exists (legacy artifact without baseline).
pub fn decode_optional_feature_baseline_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<FeatureBaselinePayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::FeatureBaseline)?
    else {
        return Ok(None);
    };
    let payload = decode_feature_baseline_payload(&section.payload)?;
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

    if params.interaction_constraints.len() > 64 {
        return Err(CoreError::InvalidConfig(format!(
            "interaction_constraints supports at most 64 groups (got {})",
            params.interaction_constraints.len()
        )));
    }
    for (gi, group) in params.interaction_constraints.iter().enumerate() {
        if group.is_empty() {
            return Err(CoreError::InvalidConfig(format!(
                "interaction_constraints group {gi} is empty; groups must contain at least one feature index"
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for &f in group {
            if !seen.insert(f) {
                return Err(CoreError::InvalidConfig(format!(
                    "interaction_constraints group {gi} contains duplicate feature index {f}"
                )));
            }
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

    if let Some(config) = params.neutralization_config {
        if config.kind == NeutralizationKind::None {
            return Err(CoreError::InvalidConfig(
                "neutralization_config must be None when neutralization kind is None".to_string(),
            ));
        }
        if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_neutralization_lambda must be finite and >= 0".to_string(),
            ));
        }
        if !config.split_penalty.is_finite() || config.split_penalty < 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_penalty must be finite and >= 0".to_string(),
            ));
        }
        if config.kind != NeutralizationKind::SplitPenalty && config.split_penalty != 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_penalty is only valid with neutralization='split_penalty'".to_string(),
            ));
        }
        if config.kind == NeutralizationKind::SplitPenalty
            && params.leaf_model == LeafModelKind::Linear
        {
            return Err(CoreError::InvalidConfig(
                "neutralization='split_penalty' requires leaf_model='constant'".to_string(),
            ));
        }
    }

    if params.leaf_solver == LeafSolverKind::Dro {
        if params.leaf_model != LeafModelKind::Constant {
            return Err(CoreError::InvalidConfig(
                "leaf_solver='dro' requires leaf_model='constant' in v0.7.4".to_string(),
            ));
        }
        let Some(cfg) = params.dro_config else {
            return Err(CoreError::InvalidConfig(
                "leaf_solver='dro' requires dro_config".to_string(),
            ));
        };
        if !cfg.radius.is_finite() || cfg.radius < 0.0 {
            return Err(CoreError::InvalidConfig(
                "dro_config.radius must be finite and >= 0".to_string(),
            ));
        }
    } else if params.dro_config.is_some() {
        return Err(CoreError::InvalidConfig(
            "dro_config is only valid with leaf_solver='dro'".to_string(),
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

    match params.boosting_mode {
        BoostingMode::Standard => {}
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            if !top_rate.is_finite() || !(0.0..1.0).contains(&top_rate) || top_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_top_rate in (0, 1), got {top_rate}"
                )));
            }
            if !other_rate.is_finite() || !(0.0..1.0).contains(&other_rate) || other_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_other_rate in (0, 1), got {other_rate}"
                )));
            }
            if top_rate + other_rate > 1.0 + f32::EPSILON {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_top_rate + goss_other_rate <= 1.0 (got {} + {} = {})",
                    top_rate,
                    other_rate,
                    top_rate + other_rate
                )));
            }
        }
        BoostingMode::Dart {
            drop_rate,
            max_drop,
            ..
        } => {
            if !drop_rate.is_finite() || !(0.0..1.0).contains(&drop_rate) || drop_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='dart' requires dart_drop_rate in (0, 1), got {drop_rate}"
                )));
            }
            if max_drop == 0 {
                return Err(CoreError::InvalidConfig(
                    "boosting_mode='dart' requires dart_max_drop >= 1".to_string(),
                ));
            }
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

    if let Some(factor_exposures) = &dataset.factor_exposures {
        if factor_exposures.row_count != dataset.matrix.row_count {
            return Err(CoreError::Validation(format!(
                "factor_exposures row_count {} does not match row_count {}",
                factor_exposures.row_count, dataset.matrix.row_count
            )));
        }
        if factor_exposures.factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = factor_exposures
            .row_count
            .checked_mul(factor_exposures.factor_count)
            .ok_or_else(|| {
                CoreError::Validation(
                    "factor_exposures row_count * factor_count overflow".to_string(),
                )
            })?;
        if factor_exposures.values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                factor_exposures.values.len(),
                expected_len
            )));
        }
        if factor_exposures.values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::Validation(
                "factor_exposures must contain only finite values".to_string(),
            ));
        }
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
    fn train_params_default_has_no_neutralization_config() {
        let params = TrainParams::default();
        assert!(params.neutralization_config.is_none());
    }

    #[test]
    fn validates_factor_exposure_matrix_shape_and_finiteness() {
        assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).is_ok());
        assert!(FactorExposureMatrix::new(0, 2, vec![]).is_err());
        assert!(FactorExposureMatrix::new(2, 0, vec![]).is_err());
        assert!(FactorExposureMatrix::new(2, 2, vec![1.0, f32::NAN, 0.0, 1.0]).is_err());
        assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn factor_exposure_matrix_row_rejects_out_of_bounds_index() {
        let exposures =
            FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid exposures");
        assert_eq!(exposures.row(1).expect("row exists"), &[0.0, 1.0]);
        assert!(exposures.row(2).is_err());

        let malformed = FactorExposureMatrix {
            row_count: 2,
            factor_count: 2,
            values: vec![1.0],
        };
        assert!(malformed.row(0).is_err());
    }

    #[test]
    fn validates_neutralization_config_contract() {
        let mut params = TrainParams {
            neutralization_config: Some(FactorNeutralizationConfig {
                kind: NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        assert!(validate_train_params(&params).is_ok());

        params.neutralization_config = Some(FactorNeutralizationConfig {
            kind: NeutralizationKind::SplitPenalty,
            ridge_lambda: 1e-6,
            split_penalty: 0.1,
        });
        assert!(validate_train_params(&params).is_ok());

        params.neutralization_config = Some(FactorNeutralizationConfig {
            kind: NeutralizationKind::SplitPenalty,
            ridge_lambda: 1e-6,
            split_penalty: -0.1,
        });
        assert!(validate_train_params(&params).is_err());
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
            factor_exposures: None,
        };
        assert!(matches!(
            validate_training_dataset(&dataset),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn validate_training_dataset_rejects_invalid_public_factor_exposures() {
        let dataset = TrainingDataset {
            matrix: DatasetMatrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]).expect("valid matrix"),
            targets: vec![1.0, 2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: Some(FactorExposureMatrix {
                row_count: 1,
                factor_count: 2,
                values: vec![1.0, f32::NAN],
            }),
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
    fn dro_effective_gradient_zero_radius_matches_l1_threshold() {
        let cfg = DroConfig {
            radius: 0.0,
            metric: DroMetric::Wasserstein,
        };
        assert_eq!(leaf_effective_gradient(4.0, 20.0, 5, 0.5, Some(&cfg)), 3.5);
        assert_eq!(
            leaf_effective_gradient(-4.0, 20.0, 5, 0.5, Some(&cfg)),
            -3.5
        );
    }

    #[test]
    fn dro_effective_gradient_shrinks_uncertain_leaf_signal() {
        let cfg = DroConfig {
            radius: 1.0,
            metric: DroMetric::Wasserstein,
        };
        let robust = leaf_effective_gradient(2.0, 20.0, 5, 0.0, Some(&cfg));
        assert!(robust.abs() < 2.0);
        assert_eq!(robust, 0.0);
    }

    #[test]
    fn dro_leaf_gain_uses_same_effective_gradient_as_leaf_solve() {
        let cfg = DroConfig {
            radius: 0.25,
            metric: DroMetric::Wasserstein,
        };
        let effective = leaf_effective_gradient(10.0, 120.0, 8, 0.1, Some(&cfg));
        let gain = leaf_gain_term(10.0, 6.0, 120.0, 8, 0.1, 0.3, Some(&cfg));
        let expected = 0.5 * effective * effective / (6.0 + 0.3 + 1e-6);
        assert!((gain - expected).abs() < 1e-6);
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
            ema_stats: Vec::new(),
        };
        let bytes = encode_morph_metadata_payload(&payload);
        // v2 envelope: 36 byte header + 4 byte EMA count (= 0) = 40 bytes.
        assert_eq!(bytes.len(), 40);
        let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn morph_metadata_round_trip_with_ema_state() {
        // v0.7.3 EMA persistence: payload encodes/decodes multi-class
        // EMA stats round-trip-clean so warm-start can resume the
        // exact EMA state from the previous fit.
        let payload = MorphMetadataPayload {
            config: MorphConfig::default(),
            final_iteration: 50,
            final_total: 50,
            ema_stats: vec![
                GradientEmaStats {
                    mean: 0.012,
                    std: 0.85,
                    alpha: 0.05,
                },
                GradientEmaStats {
                    mean: -0.003,
                    std: 1.12,
                    alpha: 0.05,
                },
                GradientEmaStats {
                    mean: 0.007,
                    std: 0.93,
                    alpha: 0.05,
                },
            ],
        };
        let bytes = encode_morph_metadata_payload(&payload);
        // 36 header + 4 count + 3 * 12 stats = 76 bytes.
        assert_eq!(bytes.len(), 76);
        let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn morph_metadata_v1_backcompat_decodes_with_empty_ema() {
        // Pre-v0.7.3 (version 1) MorphMetadata payloads have no EMA
        // tail.  Decoder must accept them and return an empty EMA Vec
        // so the warm-start path falls back to a cold EMA (matches
        // pre-v0.7.3 behaviour).
        let v1_bytes = {
            let mut buf = Vec::with_capacity(36);
            buf.extend_from_slice(&1u16.to_le_bytes()); // version = 1
            buf.extend_from_slice(&0.1f32.to_le_bytes()); // morph_rate
            buf.extend_from_slice(&0.2f32.to_le_bytes()); // evolution_pressure
            buf.extend_from_slice(&5u32.to_le_bytes()); // morph_warmup_iters
            buf.extend_from_slice(&0.3f32.to_le_bytes()); // info_score_weight
            buf.extend_from_slice(&0.9f32.to_le_bytes()); // depth_penalty_base
            buf.push(1); // balance_penalty
            buf.push(0); // lr_kind = Constant
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // warmup_frac
            buf.extend_from_slice(&10u32.to_le_bytes()); // final_iteration
            buf.extend_from_slice(&10u32.to_le_bytes()); // final_total
            buf
        };
        assert_eq!(v1_bytes.len(), 36);
        let decoded = decode_optional_morph_metadata_section(&v1_bytes).unwrap();
        assert!(
            decoded.ema_stats.is_empty(),
            "v1 payload must decode with empty ema_stats"
        );
        assert_eq!(decoded.config.morph_rate, 0.1);
        assert_eq!(decoded.final_iteration, 10);
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
            ema_stats: Vec::new(),
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
            ema_stats: Vec::new(),
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

    #[test]
    fn feature_baseline_payload_roundtrip() {
        let payload = FeatureBaselinePayload {
            feature_means: vec![0.1, -1.5, 2.0, 0.0],
        };
        let bytes = encode_feature_baseline_payload(&payload);
        let decoded = decode_feature_baseline_payload(&bytes).expect("decode succeeds");
        assert_eq!(decoded.feature_means.len(), 4);
        for (a, b) in payload
            .feature_means
            .iter()
            .zip(decoded.feature_means.iter())
        {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn feature_baseline_payload_empty_decodes_cleanly() {
        let payload = FeatureBaselinePayload {
            feature_means: vec![],
        };
        let bytes = encode_feature_baseline_payload(&payload);
        let decoded = decode_feature_baseline_payload(&bytes).expect("decode succeeds");
        assert!(decoded.feature_means.is_empty());
    }

    #[test]
    fn feature_baseline_payload_rejects_short_buffer() {
        // Header-only — claims 5 features but no body.  Decode must fail
        // cleanly rather than read past the end.
        let mut bytes = 1u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&5u32.to_le_bytes());
        let result = decode_feature_baseline_payload(&bytes);
        assert!(matches!(result, Err(CoreError::Validation(_))));
    }

    #[test]
    fn feature_baseline_section_kind_round_trips() {
        // Variant-id stability is part of the on-disk contract; if this test
        // fails, an existing artifact's section was renumbered.
        assert_eq!(ModelSectionKind::FeatureBaseline.to_u32(), 11);
        assert!(matches!(
            ModelSectionKind::from_u32(11),
            ModelSectionKind::FeatureBaseline
        ));
    }
}
