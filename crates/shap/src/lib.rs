use alloygbm_core::{
    LeafValue, LinearLeaf, ModelMetadata, deserialize_model_artifact_v1,
    format_required_section_mode_error, required_section_compatibility_report,
};
use alloygbm_engine::{ArtifactCompatibilityMode, TrainedModel, TrainedStump};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Reduce a leaf value to the "constant part" used by path-based SHAP
/// machinery.
///
/// * `LeafValue::Scalar(v)` reduces to `v`.
/// * `LeafValue::Linear(ll)` reduces to `ll.intercept + Σ wj * μj` when a
///   global feature baseline is available, or to `ll.intercept` otherwise.
///
/// The complementary `linear_leaf_row_terms` returns the row-dependent
/// `wj * (xj - μj)` deviations that must be added back to `phi[j]` for
/// additivity.  Together the two pieces reconstruct
/// `leaf_value.eval_row(row)`.
fn leaf_constant_part(leaf: &LeafValue, baseline: Option<&[f32]>) -> f64 {
    match leaf {
        LeafValue::Scalar(v) => *v as f64,
        LeafValue::Linear(ll) => {
            let mut acc = ll.intercept as f64;
            if let Some(b) = baseline {
                for (w, &feat) in ll.weights.iter().zip(ll.regressor_features.iter()) {
                    if let Some(&mean) = b.get(feat as usize) {
                        acc += (*w as f64) * (mean as f64);
                    }
                }
            }
            acc
        }
    }
}

/// Distribute the row-dependent linear deviations of a leaf onto a `phi`
/// attribution buffer.  Adds `wj * (xj - μj)` to `phi[regressor_j]` for each
/// regressor in a linear leaf.  No-op for scalar leaves.
///
/// When `baseline` is `None`, the deviation degrades to `wj * xj`, which keeps
/// additivity (`Σ phi + expected_value == predict(x)`) but biases the
/// path-attribution baseline.  Callers should prefer running with a baseline
/// recorded at fit time for the cleanest decomposition.
fn linear_leaf_row_terms(leaf: &LeafValue, row: &[f32], baseline: Option<&[f32]>, phi: &mut [f64]) {
    let LeafValue::Linear(ll) = leaf else {
        return;
    };
    accumulate_linear_terms(ll, row, baseline, phi);
}

fn accumulate_linear_terms(
    ll: &LinearLeaf,
    row: &[f32],
    baseline: Option<&[f32]>,
    phi: &mut [f64],
) {
    for (w, &feat) in ll.weights.iter().zip(ll.regressor_features.iter()) {
        let feat_idx = feat as usize;
        if feat_idx >= phi.len() {
            continue;
        }
        let xj = row.get(feat_idx).copied().unwrap_or(0.0) as f64;
        let mean = baseline
            .and_then(|b| b.get(feat_idx).copied())
            .unwrap_or(0.0) as f64;
        phi[feat_idx] += (*w as f64) * (xj - mean);
    }
}

const TREE_NODE_STRIDE: u32 = 1 << 20;
// SHAP additivity tolerance is computed as
//   atol + rtol * |predicted|
// rather than a fixed absolute bound, so accumulated f32 round-off in
// large-sample explanations (e.g. `feature_importances()` over ~1000
// rows on California Housing with `n_estimators=200`) does not raise
// even though the arithmetic is correct.  Values follow numpy's
// `allclose` convention (atol=1e-5, rtol=1e-4).
const ADDITIVITY_ATOL: f32 = 1e-5;
const ADDITIVITY_RTOL: f32 = 1e-4;

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
    fn float_threshold(&self, feature_index: usize, bin: u16) -> f32 {
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
    fn validate(&self, feature_count: usize) -> ShapResult<()> {
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
    fn quantize_row_for_linear_rank(&self, row: &[f32]) -> Option<Vec<f32>> {
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
fn additivity_tolerance(predicted: f32) -> f32 {
    ADDITIVITY_ATOL + ADDITIVITY_RTOL * predicted.abs()
}

const MAX_EXACT_SPLIT_FEATURES: usize = 25;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapError {
    InvalidInput(String),
    ContractViolation(String),
    NotSupported(String),
}

impl Display for ShapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::ContractViolation(message) => write!(f, "contract violation: {message}"),
            Self::NotSupported(message) => write!(f, "not supported: {message}"),
        }
    }
}

impl Error for ShapError {}

pub type ShapResult<T> = Result<T, ShapError>;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapExplanationBatch {
    pub expected_value: f32,
    pub values: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct ArtifactShapContext {
    feature_names: Vec<String>,
    model: TrainedModel,
}

#[derive(Debug)]
struct ModelStructure<'a> {
    tree_root_ids: Vec<u32>,
    nodes_by_tree_local_id: HashMap<u64, &'a TrainedStump>,
    split_features: Vec<usize>,
    split_feature_bit_positions: Vec<Option<u8>>,
}

pub fn explain_rows_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<ShapExplanationBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    explain_rows_from_model(&context.model, rows, None)
}

/// Predictor-aligned variant of `explain_rows_from_artifact_bytes`.
///
/// When the caller supplies a `BinningContext`, the SHAP path walker
/// uses the same float-threshold-and-strict-less-than semantics as the
/// predictor's `convert_bin_thresholds_to_float*` family, so per-row
/// attributions reach the same leaf the predictor reaches.  This is
/// required for `leaf_model="linear"` artifacts trained on continuous
/// features — the legacy bin-index path-walker diverges and produces
/// best-effort attributions that fail strict additivity.
///
/// Callers without a `BinningContext` should keep using the legacy
/// entry point above; behavior is unchanged.
pub fn explain_rows_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<ShapExplanationBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    explain_rows_from_model(&context.model, rows, Some(binning))
}

pub fn global_importance_from_shap_values(
    feature_names: &[String],
    shap_values: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if shap_values.is_empty() {
        return Err(ShapError::InvalidInput(
            "shap_values cannot be empty".to_string(),
        ));
    }

    let feature_count = feature_names.len();
    let mut contribution_sums = vec![0.0_f32; feature_count];
    for (row_index, row_values) in shap_values.iter().enumerate() {
        if row_values.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row_values.len()
            )));
        }
        for (feature_index, value) in row_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} contribution must be finite"
                )));
            }
            contribution_sums[feature_index] += value.abs();
        }
    }

    let row_count = shap_values.len() as f32;
    let mut global_importance = feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            (
                feature_name.clone(),
                contribution_sums[feature_index] / row_count,
            )
        })
        .collect::<Vec<_>>();

    global_importance.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    Ok(global_importance)
}

pub fn global_importance_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    let context = load_artifact_context(artifact_bytes)?;
    let explanation = explain_rows_from_model(&context.model, rows, None)?;
    global_importance_from_shap_values(&context.feature_names, &explanation.values)
}

/// Predictor-aligned variant of `global_importance_from_artifact_bytes`.
/// See `explain_rows_from_artifact_bytes_with_binning` for the contract.
pub fn global_importance_from_artifact_bytes_with_binning(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
    binning: &BinningContext,
) -> ShapResult<Vec<(String, f32)>> {
    let context = load_artifact_context(artifact_bytes)?;
    let explanation = explain_rows_from_model(&context.model, rows, Some(binning))?;
    global_importance_from_shap_values(&context.feature_names, &explanation.values)
}

// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `explain_rows_from_artifact_bytes` for artifact-backed explanations.
pub fn shap_values_stub(metadata: &ModelMetadata, rows: &[Vec<f32>]) -> ShapResult<Vec<Vec<f32>>> {
    let feature_count = metadata.feature_names.len();
    validate_rows(rows, feature_count)?;
    Ok(vec![vec![0.0; feature_count]; rows.len()])
}

// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `global_importance_from_shap_values`.
pub fn global_importance_stub(
    metadata: &ModelMetadata,
    feature_names: &[String],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if feature_names.len() != metadata.feature_names.len() {
        return Err(ShapError::InvalidInput(format!(
            "feature_names length {} does not match metadata feature count {}",
            feature_names.len(),
            metadata.feature_names.len()
        )));
    }

    Ok(feature_names
        .iter()
        .map(|name| (name.clone(), 0.0_f32))
        .collect())
}

fn load_artifact_context(artifact_bytes: &[u8]) -> ShapResult<ArtifactShapContext> {
    let parsed = deserialize_model_artifact_v1(artifact_bytes)
        .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

    if parsed.contract.metadata.num_classes.is_some() {
        return Err(ShapError::ContractViolation(
            "SHAP values are not yet supported for multi-class models".to_string(),
        ));
    }

    let compatibility_report = required_section_compatibility_report(&parsed.sections);
    if !compatibility_report.legacy_compatible {
        return Err(ShapError::ContractViolation(
            format_required_section_mode_error(compatibility_report, true),
        ));
    }

    let model = TrainedModel::from_artifact_bytes_with_mode(
        artifact_bytes,
        ArtifactCompatibilityMode::AllowLegacyTreesOnly,
    )
    .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

    if parsed.contract.metadata.feature_names.len() != model.feature_count {
        return Err(ShapError::ContractViolation(format!(
            "metadata feature count {} does not match model feature count {}",
            parsed.contract.metadata.feature_names.len(),
            model.feature_count
        )));
    }

    Ok(ArtifactShapContext {
        feature_names: parsed.contract.metadata.feature_names,
        model,
    })
}

fn explain_rows_from_model(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    validate_rows(rows, model.feature_count)?;
    if let Some(ctx) = binning {
        ctx.validate(model.feature_count)?;
    }

    // LinearRank: the predictor evaluates both tree traversal and PL
    // leaves in bin-index space, so quantize rows once at the entry
    // point and dispatch with PreBinned semantics for the remainder.
    // See the `BinningContext::LinearRank` doc comment for the parity
    // rationale.
    if let Some(ctx @ BinningContext::LinearRank { .. }) = binning {
        let quantized: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                ctx.quantize_row_for_linear_rank(row)
                    .expect("LinearRank quantize_row_for_linear_rank returns Some")
            })
            .collect();
        return explain_rows_from_model(model, &quantized, Some(&BinningContext::PreBinned));
    }

    // Count distinct split features to choose algorithm.
    let distinct_split_feature_count = {
        let mut features: Vec<usize> = model
            .stumps
            .iter()
            .map(|s| s.split.feature_index as usize)
            .collect();
        features.sort_unstable();
        features.dedup();
        features.len()
    };

    if distinct_split_feature_count > MAX_EXACT_SPLIT_FEATURES {
        // Too many features for brute-force O(2^N); use TreeSHAP O(TLD^2).
        return explain_rows_tree_shap(model, rows, binning);
    }

    // Brute-force exact Shapley values for models with few split features.
    explain_rows_brute_force(model, rows, binning)
}

fn explain_rows_brute_force(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    let model_structure = build_model_structure(model)?;
    let baseline = model.feature_baseline.as_deref();
    let expected_value = expected_prediction_for_subset(
        model,
        rows[0].as_slice(),
        0,
        &model_structure,
        baseline,
        binning,
    )?;

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let values_by_subset =
            compute_subset_expectations(model, row, &model_structure, baseline, binning)?;
        let row_expected_value = values_by_subset[0];

        if (row_expected_value - expected_value).abs() > additivity_tolerance(expected_value) {
            return Err(ShapError::ContractViolation(format!(
                "row {row_index} expected value drift: {row_expected_value} vs baseline {expected_value}"
            )));
        }

        let mut contributions_f64 =
            shapley_values_for_row_f64(model, row, &values_by_subset, &model_structure, row_index)?;

        // Linear-leaf interventional decomposition: the brute-force path
        // attribution above is computed on the "constant part" of each leaf
        // (`intercept + Σ wj * μj`).  Adding `wj * (xj - μj)` per regressor
        // at *every visited node along the row's path* restores `predict(x)`
        // exactly (matching how `predict` accumulates `leaf.eval_row(row)`
        // at each visited node) while attributing the row's deviation
        // directly to the relevant features.  See
        // `distribute_linear_terms_for_row` for the full path walk.
        if model_has_linear_leaves(model) {
            distribute_linear_terms_for_row(model, row, baseline, binning, &mut contributions_f64);
        }

        let contributions: Vec<f32> = contributions_f64.iter().map(|v| *v as f32).collect();
        verify_additivity(
            model,
            row,
            &contributions,
            row_index,
            expected_value,
            binning,
        )?;
        row_contributions.push(contributions);
    }

    Ok(ShapExplanationBatch {
        expected_value,
        values: row_contributions,
    })
}

fn model_has_linear_leaves(model: &TrainedModel) -> bool {
    model.stumps.iter().any(|s| {
        matches!(s.left_leaf_value, LeafValue::Linear(_))
            || matches!(s.right_leaf_value, LeafValue::Linear(_))
    })
}

/// Walk each tree for `row` and credit `wj · (xj − μj)` for every linear leaf
/// the row visits along its path.
///
/// **This must visit every node on the row's path, not just the terminal**
/// — `predict(x)` and `local_path_predict` both accumulate
/// `leaf.eval_row(row)` at every visited node (the predictor loops as long
/// as `nodes_by_local_id.get(child)` returns a stump).  The brute-force
/// SHAP and TreeSHAP polynomial paths already handle the per-visited-node
/// **constant** contribution `intercept + Σⱼ wⱼ·μⱼ` through
/// `leaf_constant_part`.  The per-visited-node **deviation**
/// `Σⱼ wⱼ·(xⱼ − μⱼ)` is uncredited unless we add it here.
///
/// Crediting only the terminal leaf was the pre-v0.7.4 bug: for a row whose
/// path through a tree visits N internal nodes plus a terminal, the SHAP
/// reconstruction was missing N nodes' worth of `Σⱼ wⱼ·(xⱼ − μⱼ)`, scaling
/// with `n_estimators` and `max_depth` and producing additivity drifts on
/// the order of the predictions themselves.
///
/// Trees with only scalar leaves remain no-ops because
/// `linear_leaf_row_terms` does nothing for `LeafValue::Scalar`, so scalar-
/// leaf-only models pay no overhead for the broader walk.
fn distribute_linear_terms_for_row(
    model: &TrainedModel,
    row: &[f32],
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
    phi: &mut [f64],
) {
    // Build a (tree_id, local_id) → stump map once per row is overkill, but
    // SHAP is not on the hot path and rows count is typically modest.  The
    // node-key map is also built inside `build_model_structure` for the
    // brute-force pre-processing; rebuilding here keeps this helper usable
    // from the polynomial TreeSHAP path too.
    let mut nodes_by_key: HashMap<u64, &TrainedStump> = HashMap::with_capacity(model.stumps.len());
    let mut tree_roots: Vec<u32> = Vec::new();
    for stump in &model.stumps {
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        nodes_by_key.insert(tree_local_key(tree_id, local_id), stump);
        if local_id == 0 {
            tree_roots.push(tree_id);
        }
    }
    tree_roots.sort_unstable();
    tree_roots.dedup();

    for tree_id in tree_roots {
        let mut local_id = 0u32;
        while let Some(stump) = nodes_by_key.get(&tree_local_key(tree_id, local_id)) {
            let feat = stump.split.feature_index as usize;
            let feature_value = row.get(feat).copied().unwrap_or(f32::NAN);
            let goes_left = stump_goes_left(&stump.split, feature_value, binning);
            let leaf_value = if goes_left {
                &stump.left_leaf_value
            } else {
                &stump.right_leaf_value
            };
            // Credit the visited leaf's linear deviation, whether it's an
            // internal-node side or the terminal.  No-op for scalar leaves.
            linear_leaf_row_terms(leaf_value, row, baseline, phi);
            local_id = if goes_left {
                local_id.saturating_mul(2).saturating_add(1)
            } else {
                local_id.saturating_mul(2).saturating_add(2)
            };
        }
    }
}

fn build_model_structure(model: &TrainedModel) -> ShapResult<ModelStructure<'_>> {
    let mut nodes_by_tree_local_id = HashMap::new();
    let mut tree_root_ids = Vec::new();
    let mut split_features = Vec::new();

    for stump in &model.stumps {
        let (tree_id, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let node_key = tree_local_key(tree_id, local_node_id);
        nodes_by_tree_local_id.insert(node_key, stump);
        if local_node_id == 0 {
            tree_root_ids.push(tree_id);
        }

        let feature_index = stump.split.feature_index as usize;
        if feature_index >= model.feature_count {
            return Err(ShapError::ContractViolation(format!(
                "stump feature_index {} exceeds model feature_count {}",
                stump.split.feature_index, model.feature_count
            )));
        }
        split_features.push(feature_index);
    }

    tree_root_ids.sort_unstable();
    tree_root_ids.dedup();
    split_features.sort_unstable();
    split_features.dedup();

    if split_features.len() > MAX_EXACT_SPLIT_FEATURES {
        return Err(ShapError::ContractViolation(format!(
            "exact SHAP supports at most {MAX_EXACT_SPLIT_FEATURES} distinct split features per model (found {})",
            split_features.len()
        )));
    }

    let mut split_feature_bit_positions = vec![None; model.feature_count];
    for (bit_position, feature_index) in split_features.iter().enumerate() {
        split_feature_bit_positions[*feature_index] = Some(bit_position as u8);
    }

    Ok(ModelStructure {
        tree_root_ids,
        nodes_by_tree_local_id,
        split_features,
        split_feature_bit_positions,
    })
}

fn compute_subset_expectations(
    model: &TrainedModel,
    row: &[f32],
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<Vec<f32>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = 1_usize
        .checked_shl(split_feature_count as u32)
        .ok_or_else(|| ShapError::ContractViolation("subset count overflow".to_string()))?;

    let mut values_by_subset = Vec::with_capacity(subset_count);
    for subset_mask in 0..subset_count {
        let value = expected_prediction_for_subset(
            model,
            row,
            subset_mask as u64,
            model_structure,
            baseline,
            binning,
        )?;
        values_by_subset.push(value);
    }
    Ok(values_by_subset)
}

/// Determine whether a feature value goes to the left child of a split.
/// Uses bitset membership for categorical splits and threshold comparison
/// for numeric splits.
fn stump_goes_left(
    split: &alloygbm_core::SplitCandidate,
    feature_value: f32,
    binning: Option<&BinningContext>,
) -> bool {
    if feature_value.is_nan() {
        return split.default_left;
    }
    if split.is_categorical {
        let cat_id = feature_value as u16;
        return split
            .categorical_bitset
            .as_ref()
            .map_or(split.default_left, |bs| {
                let byte_idx = (cat_id / 8) as usize;
                let bit_idx = (cat_id % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            });
    }
    match binning {
        // Float-threshold path: matches the predictor's strict `<`
        // comparison after `convert_bin_thresholds_to_float*`.  When a
        // binning context is provided, SHAP walks paths the same way
        // the predictor does, so linear-leaf attribution stays
        // additive on continuous features.
        Some(ctx) => {
            let threshold = ctx.float_threshold(split.feature_index as usize, split.threshold_bin);
            feature_value < threshold
        }
        // Legacy bin-index path.  Preserved for callers that don't
        // (or can't) provide a `BinningContext` — categorical-only
        // and pre-binned-integer artifacts predominantly.
        None => feature_value <= split.threshold_bin as f32,
    }
}

fn expected_prediction_for_subset(
    model: &TrainedModel,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<f32> {
    let mut prediction = model.baseline_prediction;
    for tree_id in &model_structure.tree_root_ids {
        prediction += expected_subtree(
            *tree_id,
            0,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;
    }
    Ok(prediction)
}

fn expected_subtree(
    tree_id: u32,
    local_node_id: u32,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> ShapResult<f32> {
    let node_key = tree_local_key(tree_id, local_node_id);
    let Some(stump) = model_structure.nodes_by_tree_local_id.get(&node_key) else {
        return Ok(0.0);
    };

    let split_feature_index = stump.split.feature_index as usize;
    if split_feature_index >= row.len() {
        return Err(ShapError::ContractViolation(format!(
            "split feature_index {} exceeds row feature length {}",
            stump.split.feature_index,
            row.len()
        )));
    }

    let left_child_local = left_child_local_id(local_node_id)?;
    let right_child_local = right_child_local_id(local_node_id)?;

    // Use the leaf "constant part" — `intercept + Σ wj * μj` for linear
    // leaves — so the path-based attribution acts on a scalar-valued tree.
    // Linear deviations `wj * (xj - μj)` are added back to phi after the
    // Shapley computation by `distribute_linear_terms_for_row`.
    let left_const = leaf_constant_part(&stump.left_leaf_value, baseline) as f32;
    let right_const = leaf_constant_part(&stump.right_leaf_value, baseline) as f32;

    if let Some(bit_position) = model_structure.split_feature_bit_positions[split_feature_index] {
        let is_known = (subset_mask & (1_u64 << bit_position)) != 0;
        if is_known {
            let goes_left = stump_goes_left(&stump.split, row[split_feature_index], binning);
            if goes_left {
                return Ok(left_const
                    + expected_subtree(
                        tree_id,
                        left_child_local,
                        row,
                        subset_mask,
                        model_structure,
                        baseline,
                        binning,
                    )?);
            }
            return Ok(right_const
                + expected_subtree(
                    tree_id,
                    right_child_local,
                    row,
                    subset_mask,
                    model_structure,
                    baseline,
                    binning,
                )?);
        }
    }

    let left_count = stump.split.left_stats.row_count as f32;
    let right_count = stump.split.right_stats.row_count as f32;
    let total_count = left_count + right_count;
    let left_probability = if total_count > 0.0 {
        left_count / total_count
    } else {
        0.5
    };
    let right_probability = 1.0 - left_probability;

    let left_expected = left_const
        + expected_subtree(
            tree_id,
            left_child_local,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;
    let right_expected = right_const
        + expected_subtree(
            tree_id,
            right_child_local,
            row,
            subset_mask,
            model_structure,
            baseline,
            binning,
        )?;

    Ok(left_probability * left_expected + right_probability * right_expected)
}

fn shapley_values_for_row_f64(
    model: &TrainedModel,
    _row: &[f32],
    values_by_subset: &[f32],
    model_structure: &ModelStructure<'_>,
    _row_index: usize,
) -> ShapResult<Vec<f64>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = values_by_subset.len();

    let mut contributions = vec![0.0_f64; model.feature_count];
    if split_feature_count == 0 {
        return Ok(contributions);
    }

    let factorials = factorial_table(split_feature_count);
    let total_factorial = factorials[split_feature_count];

    for (feature_bit_position, &feature_index) in model_structure.split_features.iter().enumerate()
    {
        let feature_bit = 1_u64 << feature_bit_position;
        let mut phi = 0.0_f64;

        for subset_mask in 0..subset_count {
            let subset_mask_u64 = subset_mask as u64;
            if (subset_mask_u64 & feature_bit) != 0 {
                continue;
            }

            let with_feature_mask = subset_mask_u64 | feature_bit;
            let subset_size = subset_mask_u64.count_ones() as usize;
            let weight = factorials[subset_size]
                * factorials[split_feature_count - subset_size - 1]
                / total_factorial;

            let marginal =
                values_by_subset[with_feature_mask as usize] - values_by_subset[subset_mask];
            phi += weight * marginal as f64;
        }

        contributions[feature_index] = phi;
    }

    Ok(contributions)
}

fn verify_additivity(
    model: &TrainedModel,
    row: &[f32],
    contributions: &[f32],
    row_index: usize,
    expected_value: f32,
    binning: Option<&BinningContext>,
) -> ShapResult<()> {
    // Compute the prediction by walking each tree once and summing the leaf
    // values along the row's path.  Mirrors `distribute_linear_terms_for_row`.
    //
    // **Tolerance policy.**  Additivity is checked against
    //   atol + rtol * |predicted|
    // rather than a fixed absolute bound.  This matches numpy `allclose`
    // semantics and means accumulated f32 round-off across large
    // explanation batches (e.g. `feature_importances()` over ~1000 rows
    // on California Housing with `n_estimators=200`) does not raise even
    // though the arithmetic is correct.
    //
    // **Linear leaves.**  As of v0.7.4, `leaf_model="linear"` artifacts
    // satisfy strict additivity end-to-end when called with a
    // `BinningContext` (the predictor-aligned path).  The fix combines
    // v0.7.3's float-threshold path walker with crediting
    // `Σⱼ wⱼ·(xⱼ − μⱼ)` at every visited node along the row's path —
    // matching how `predict` accumulates `leaf.eval_row(row)` at each
    // visited node.  See `distribute_linear_terms_for_row` for the path
    // walk and `leaf_constant_part` for the constant-part flow through
    // `expected_subtree` / `build_std_tree`.
    //
    // **v0.8.0:** the `BinningContext::LinearRank` variant joins
    // `Linear`, `Quantile`, and `PreBinned` as a fully strict-additivity
    // context.  When a caller passes any `BinningContext` variant the
    // SHAP path walker matches the predictor's path exactly, so the
    // linear-leaf exemption MUST NOT trigger.
    //
    // When `binning=None`, the SHAP walker uses the legacy `<=` bin-index
    // comparison and may take a different path than the predictor, so
    // strict additivity is not guaranteed for linear leaves on that
    // legacy path.  The exemption is retained only in that case.
    let predicted = local_path_predict(model, row, binning);
    let reconstructed = expected_value + contributions.iter().sum::<f32>();
    if binning.is_none() && model_has_linear_leaves(model) {
        // Legacy path-walker — best-effort interventional explanation.
        // Predictor-aligned (BinningContext) callers (Linear, Quantile,
        // PreBinned, LinearRank) get the strict check.
        return Ok(());
    }
    let tolerance = additivity_tolerance(predicted);
    if (predicted - reconstructed).abs() > tolerance {
        return Err(ShapError::ContractViolation(format!(
            "row {row_index} additivity check failed: predicted={predicted}, reconstructed={reconstructed}, tolerance={tolerance} (atol={ADDITIVITY_ATOL}, rtol={ADDITIVITY_RTOL})"
        )));
    }
    Ok(())
}

/// Compute `predict(row)` by walking each tree along the row's actual path
/// and summing the leaf evaluations at each visited internal node.  Used
/// internally by `verify_additivity`.  This is the same path-walking logic as
/// `distribute_linear_terms_for_row`, but here it accumulates the *full* leaf
/// value (`eval_row`) rather than just the linear deviation.
fn local_path_predict(model: &TrainedModel, row: &[f32], binning: Option<&BinningContext>) -> f32 {
    let mut nodes_by_key: HashMap<u64, &TrainedStump> = HashMap::with_capacity(model.stumps.len());
    let mut tree_roots: Vec<u32> = Vec::new();
    for stump in &model.stumps {
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        nodes_by_key.insert(tree_local_key(tree_id, local_id), stump);
        if local_id == 0 {
            tree_roots.push(tree_id);
        }
    }
    tree_roots.sort_unstable();
    tree_roots.dedup();

    let mut prediction = model.baseline_prediction;
    for tree_id in tree_roots {
        let mut local_id = 0u32;
        while let Some(stump) = nodes_by_key.get(&tree_local_key(tree_id, local_id)) {
            let feat = stump.split.feature_index as usize;
            let feature_value = row.get(feat).copied().unwrap_or(f32::NAN);
            let goes_left = stump_goes_left(&stump.split, feature_value, binning);
            let leaf = if goes_left {
                &stump.left_leaf_value
            } else {
                &stump.right_leaf_value
            };
            prediction += leaf.eval_row(row);
            local_id = if goes_left {
                local_id.saturating_mul(2).saturating_add(1)
            } else {
                local_id.saturating_mul(2).saturating_add(2)
            };
        }
    }
    prediction
}

fn factorial_table(max_value: usize) -> Vec<f64> {
    let mut factorials = vec![1.0_f64; max_value + 1];
    for value in 1..=max_value {
        factorials[value] = factorials[value - 1] * value as f64;
    }
    factorials
}

fn tree_local_key(tree_id: u32, local_node_id: u32) -> u64 {
    ((tree_id as u64) << 32) | local_node_id as u64
}

fn left_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ShapError::ContractViolation("left child node id overflow".to_string()))
}

fn right_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| ShapError::ContractViolation("right child node id overflow".to_string()))
}

fn validate_rows(rows: &[Vec<f32>], feature_count: usize) -> ShapResult<()> {
    if feature_count == 0 {
        return Err(ShapError::InvalidInput(
            "model feature_count must be greater than 0".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(ShapError::InvalidInput("rows cannot be empty".to_string()));
    }

    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row.len()
            )));
        }
        for (feature_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} contains NaN/Inf. \
                     SHAP values require finite feature values. If your data \
                     contains missing values, impute them before calling shap_values()."
                )));
            }
        }
    }

    Ok(())
}

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

// ---------------------------------------------------------------------------
// TreeSHAP: Polynomial-time O(TLD^2) exact Shapley values
// Based on Lundberg et al. (2020), "From local explanations to global
// understanding with explainable AI for trees"
// ---------------------------------------------------------------------------

/// Standard tree node used by TreeSHAP. Converts from AlloyGBM's stump-based
/// representation where each stump carries left/right leaf values into a
/// conventional tree where leaf values represent total accumulated prediction.
#[derive(Debug, Clone)]
enum StdTreeNode {
    Leaf {
        value: f64,
        cover: f64,
    },
    Internal {
        feature_index: usize,
        threshold: f32,
        default_left: bool,
        is_categorical: bool,
        categorical_bitset: Option<Vec<u8>>,
        left: Box<StdTreeNode>,
        right: Box<StdTreeNode>,
    },
}

impl StdTreeNode {
    fn cover(&self) -> f64 {
        match self {
            Self::Leaf { cover, .. } => *cover,
            Self::Internal { left, right, .. } => left.cover() + right.cover(),
        }
    }

    /// Cover-weighted sum of leaf values. Divide by `cover()` to get E[f_tree(x)].
    fn cover_weighted_value_sum(&self) -> f64 {
        match self {
            Self::Leaf { value, cover } => value * cover,
            Self::Internal { left, right, .. } => {
                left.cover_weighted_value_sum() + right.cover_weighted_value_sum()
            }
        }
    }
}

/// One element in the TreeSHAP path tracking structure.
#[derive(Clone, Copy)]
struct PathElement {
    feature_index: usize,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// Build a standard tree from AlloyGBM's stump representation for a single
/// tree. Accumulated leaf values are pushed down so that each leaf's `value`
/// is the total prediction contribution for samples reaching that leaf.
///
/// For piecewise-linear leaves we accumulate only the "constant part"
/// (`intercept + Σ wj * μj` when `baseline` is `Some`).  The row-dependent
/// `wj * (xj - μj)` terms are credited back to per-feature SHAP values
/// outside the path-based machinery — see `distribute_linear_terms_for_row`.
fn build_std_tree(
    tree_id: u32,
    local_id: u32,
    accumulated_value: f64,
    parent_cover: f64,
    nodes: &HashMap<u64, &TrainedStump>,
    baseline: Option<&[f32]>,
    binning: Option<&BinningContext>,
) -> StdTreeNode {
    let key = tree_local_key(tree_id, local_id);
    match nodes.get(&key) {
        None => StdTreeNode::Leaf {
            value: accumulated_value,
            cover: parent_cover,
        },
        Some(stump) => {
            let left_cover = stump.split.left_stats.row_count as f64;
            let right_cover = stump.split.right_stats.row_count as f64;
            // When a binning context is provided we bake the predictor-
            // matching float threshold directly into the StdTreeNode; the
            // TreeSHAP recursion (`ts_recurse`) consumes this field as the
            // decision boundary and now compares with `<` instead of `<=`
            // (see `goes_left_with_threshold`).  When `binning` is None
            // we fall back to the legacy bin-index encoding.
            let threshold = match binning {
                Some(ctx) if !stump.split.is_categorical => ctx.float_threshold(
                    stump.split.feature_index as usize,
                    stump.split.threshold_bin,
                ),
                _ => stump.split.threshold_bin as f32,
            };
            StdTreeNode::Internal {
                feature_index: stump.split.feature_index as usize,
                threshold,
                default_left: stump.split.default_left,
                is_categorical: stump.split.is_categorical,
                categorical_bitset: stump.split.categorical_bitset.clone(),
                left: Box::new(build_std_tree(
                    tree_id,
                    2 * local_id + 1,
                    accumulated_value + leaf_constant_part(&stump.left_leaf_value, baseline),
                    left_cover,
                    nodes,
                    baseline,
                    binning,
                )),
                right: Box::new(build_std_tree(
                    tree_id,
                    2 * local_id + 2,
                    accumulated_value + leaf_constant_part(&stump.right_leaf_value, baseline),
                    right_cover,
                    nodes,
                    baseline,
                    binning,
                )),
            }
        }
    }
}

/// Extend the unique path with a new feature (Algorithm 2, Lundberg et al.).
fn ts_extend_path(
    path: &mut [PathElement],
    depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: usize,
) {
    path[depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if depth == 0 { 1.0 } else { 0.0 },
    };
    for i in (0..depth).rev() {
        path[i + 1].pweight += one_fraction * path[i].pweight * (i + 1) as f64 / (depth + 1) as f64;
        path[i].pweight = zero_fraction * path[i].pweight * (depth - i) as f64 / (depth + 1) as f64;
    }
}

/// Remove a feature from the path and shift remaining elements
/// (Algorithm 3, Lundberg et al.).
///
/// **Critical**: the shift at the end moves only `feature_index`,
/// `zero_fraction`, and `one_fraction` — NOT `pweight`.  The unwind
/// loop above has already computed the correct post-unwind pweights
/// in place; shifting them would clobber those values with the
/// pweights of the elements being shifted down (whose pweights were
/// computed when the duplicate was still in the path, not after its
/// removal).
///
/// The reference Python implementation in slundberg/shap uses four
/// parallel arrays (`feature_indexes`, `zero_fractions`,
/// `one_fractions`, `pweights`) and only shifts the first three.
/// The original AlloyGBM port stored all four in a single
/// `PathElement` struct and shifted the entire struct, which broke
/// the TreeSHAP polynomial path for any tree where a feature
/// appeared more than once on a root-to-leaf path (Limitation 5,
/// closed in v0.7.5).
fn ts_unextend_path(path: &mut [PathElement], depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[depth].pweight;

    for i in (0..depth).rev() {
        if one_fraction.abs() > 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight =
                next_one_portion * (depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (depth - i) as f64 / (depth + 1) as f64;
        } else {
            path[i].pweight =
                path[i].pweight * (depth + 1) as f64 / (zero_fraction * (depth - i) as f64);
        }
    }

    // Shift feature_index / zero_fraction / one_fraction only.
    // pweights are NOT shifted — see the function comment above.
    for i in path_index..depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

/// Compute the SHAP weight for unwinding the feature at `path_index`
/// (Algorithm 4, Lundberg et al.).
fn ts_unwound_path_sum(path: &[PathElement], depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[depth].pweight;
    let mut total = 0.0_f64;

    for i in (0..depth).rev() {
        if one_fraction.abs() > 0.0 {
            let tmp = next_one_portion * (depth + 1) as f64 / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction * (depth - i) as f64 / (depth + 1) as f64;
        } else if zero_fraction.abs() > 0.0 {
            let ratio = (depth - i) as f64 / (depth + 1) as f64;
            total += path[i].pweight / (zero_fraction * ratio);
        }
    }

    total
}

/// Recursive TreeSHAP walk (Algorithm 1, Lundberg et al.).
///
/// At each node the incoming edge's feature is added to the path. At leaves
/// the path is unwound to attribute contributions. At internal nodes the path
/// is cloned for each child so that modifications are independent.
#[allow(clippy::too_many_arguments)]
fn ts_recurse(
    node: &StdTreeNode,
    row: &[f32],
    path: &mut Vec<PathElement>,
    depth: usize,
    phi: &mut [f64],
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: usize,
    // When true, the tree's `threshold` fields hold float thresholds and
    // the decision uses strict `<` (predictor-aligned).  When false the
    // legacy bin-index encoding is used with `<=`.
    use_float_compare: bool,
) {
    // Ensure the path vector has room for this depth.
    while path.len() <= depth {
        path.push(PathElement {
            feature_index: usize::MAX,
            zero_fraction: 0.0,
            one_fraction: 0.0,
            pweight: 0.0,
        });
    }

    ts_extend_path(path, depth, zero_fraction, one_fraction, feature_index);

    match node {
        StdTreeNode::Leaf { value, .. } => {
            // Unwind each feature to compute its contribution.
            for i in 1..=depth {
                let w = ts_unwound_path_sum(path, depth, i);
                let feat = path[i].feature_index;
                if feat < phi.len() {
                    phi[feat] += w * (path[i].one_fraction - path[i].zero_fraction) * value;
                }
            }
        }
        StdTreeNode::Internal {
            feature_index: node_feature,
            threshold,
            default_left,
            is_categorical,
            categorical_bitset,
            left,
            right,
        } => {
            let goes_left = row
                .get(*node_feature)
                .map(|v| {
                    if *is_categorical {
                        let cat_id = *v as u16;
                        categorical_bitset.as_ref().map_or(*default_left, |bs| {
                            let byte_idx = (cat_id / 8) as usize;
                            let bit_idx = (cat_id % 8) as usize;
                            byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
                        })
                    } else if use_float_compare {
                        // Predictor-aligned strict less-than against the
                        // float threshold baked in by `build_std_tree`.
                        *v < *threshold
                    } else {
                        // Legacy bin-index comparison.
                        *v <= *threshold
                    }
                })
                .unwrap_or(*default_left);
            let (hot, cold) = if goes_left {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            let node_cover = node.cover();
            let hot_zero = if node_cover > 0.0 {
                hot.cover() / node_cover
            } else {
                0.5
            };
            let cold_zero = if node_cover > 0.0 {
                cold.cover() / node_cover
            } else {
                0.5
            };

            // Check whether this split feature already appears in the path.
            let duplicate_index = path[1..=depth]
                .iter()
                .position(|e| e.feature_index == *node_feature)
                .map(|pos| pos + 1);

            // Clone the path for each child so modifications are independent.
            let mut hot_path = path[..=depth].to_vec();
            let mut cold_path = path[..=depth].to_vec();

            if let Some(dup_idx) = duplicate_index {
                // Duplicate feature: combine incoming fractions.
                let incoming_zero = hot_path[dup_idx].zero_fraction;
                let incoming_one = hot_path[dup_idx].one_fraction;
                ts_unextend_path(&mut hot_path, depth, dup_idx);
                ts_unextend_path(&mut cold_path, depth, dup_idx);
                let child_depth = depth - 1;

                ts_recurse(
                    hot,
                    row,
                    &mut hot_path,
                    child_depth + 1,
                    phi,
                    incoming_zero * hot_zero,
                    incoming_one,
                    *node_feature,
                    use_float_compare,
                );
                ts_recurse(
                    cold,
                    row,
                    &mut cold_path,
                    child_depth + 1,
                    phi,
                    incoming_zero * cold_zero,
                    0.0,
                    *node_feature,
                    use_float_compare,
                );
            } else {
                ts_recurse(
                    hot,
                    row,
                    &mut hot_path,
                    depth + 1,
                    phi,
                    hot_zero,
                    1.0,
                    *node_feature,
                    use_float_compare,
                );
                ts_recurse(
                    cold,
                    row,
                    &mut cold_path,
                    depth + 1,
                    phi,
                    cold_zero,
                    0.0,
                    *node_feature,
                    use_float_compare,
                );
            }
        }
    }
}

/// Compute SHAP values for a single row using pre-built standard trees.
fn tree_shap_row(
    trees: &[StdTreeNode],
    row: &[f32],
    feature_count: usize,
    use_float_compare: bool,
) -> Vec<f64> {
    let mut phi = vec![0.0_f64; feature_count];
    for tree in trees {
        let mut path = Vec::with_capacity(32);
        ts_recurse(
            tree,
            row,
            &mut path,
            0,
            &mut phi,
            1.0,
            1.0,
            usize::MAX,
            use_float_compare,
        );
    }
    phi
}

/// Compute SHAP values for multiple rows using TreeSHAP.
fn explain_rows_tree_shap(
    model: &TrainedModel,
    rows: &[Vec<f32>],
    binning: Option<&BinningContext>,
) -> ShapResult<ShapExplanationBatch> {
    validate_rows(rows, model.feature_count)?;

    // Build node lookup and standard trees once for all rows.
    let mut nodes_map: HashMap<u64, &TrainedStump> = HashMap::new();
    let mut tree_roots: Vec<u32> = Vec::new();
    for stump in &model.stumps {
        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
        nodes_map.insert(tree_local_key(tree_id, local_id), stump);
        if local_id == 0 {
            tree_roots.push(tree_id);
        }
    }
    tree_roots.sort_unstable();
    tree_roots.dedup();

    let baseline = model.feature_baseline.as_deref();
    let has_linear = model_has_linear_leaves(model);

    let mut std_trees = Vec::with_capacity(tree_roots.len());
    let mut expected_value_f64 = model.baseline_prediction as f64;

    for &tree_id in &tree_roots {
        let root_key = tree_local_key(tree_id, 0);
        let root_stump = nodes_map.get(&root_key).ok_or_else(|| {
            ShapError::ContractViolation(format!("missing root stump for tree {tree_id}"))
        })?;
        let root_cover = root_stump.split.left_stats.row_count as f64
            + root_stump.split.right_stats.row_count as f64;

        let tree = build_std_tree(tree_id, 0, 0.0, root_cover, &nodes_map, baseline, binning);

        // E[f_tree(x)] = cover-weighted average leaf value (computed on the
        // constant-part tree).  For linear leaves, the row-dependent
        // deviations sum to 0 in expectation (Σ wj · E[Xj - μj] = 0), so the
        // expected_value is the same under either decomposition.
        let tree_cover = tree.cover();
        if tree_cover > 0.0 {
            expected_value_f64 += tree.cover_weighted_value_sum() / tree_cover;
        }

        std_trees.push(tree);
    }

    let expected_value = expected_value_f64 as f32;
    let use_float_compare = binning.is_some();

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let mut phi = tree_shap_row(&std_trees, row, model.feature_count, use_float_compare);
        if has_linear {
            distribute_linear_terms_for_row(model, row, baseline, binning, &mut phi);
        }
        let contributions: Vec<f32> = phi.iter().map(|v| *v as f32).collect();
        verify_additivity(
            model,
            row,
            &contributions,
            row_index,
            expected_value,
            binning,
        )?;
        row_contributions.push(contributions);
    }

    Ok(ShapExplanationBatch {
        expected_value,
        values: row_contributions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{
        Device, LeafValue, ModelMetadata, ModelSectionKind, NodeStats, SplitCandidate,
        serialize_model_artifact_v1,
    };
    use alloygbm_engine::TrainedModel;
    use alloygbm_predictor::Predictor;

    fn sample_metadata(feature_names: &[&str]) -> ModelMetadata {
        ModelMetadata {
            format_version: 1,
            feature_names: feature_names
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        }
    }

    fn split(node_id: u32, feature_index: u32, threshold_bin: u16) -> SplitCandidate {
        SplitCandidate {
            node_id,
            feature_index,
            threshold_bin,
            gain: 1.0,
            default_left: false,
            is_categorical: false,
            categorical_bitset: None,
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                row_count: 1,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                row_count: 1,
            },
        }
    }

    fn split_with_counts(
        node_id: u32,
        feature_index: u32,
        threshold_bin: u16,
        left_count: u32,
        right_count: u32,
    ) -> SplitCandidate {
        SplitCandidate {
            node_id,
            feature_index,
            threshold_bin,
            gain: 1.0,
            default_left: false,
            is_categorical: false,
            categorical_bitset: None,
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: left_count as f32,
                grad_sq_sum: 0.0,
                row_count: left_count,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: right_count as f32,
                grad_sq_sum: 0.0,
                row_count: right_count,
            },
        }
    }

    fn fixture_model() -> TrainedModel {
        TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: split(0, 0, 1),
                    left_leaf_value: LeafValue::Scalar(1.0),
                    right_leaf_value: LeafValue::Scalar(2.0),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split(1, 1, 0),
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(0.2),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split(2, 1, 1),
                    left_leaf_value: LeafValue::Scalar(0.3),
                    right_leaf_value: LeafValue::Scalar(0.4),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        }
    }

    fn fixture_model_with_unused_feature() -> TrainedModel {
        TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 3,
            stumps: fixture_model().stumps,
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        }
    }

    fn fixture_rows() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0],
            vec![0.0, 2.0],
            vec![3.0, 0.0],
            vec![3.0, 2.0],
        ]
    }

    fn fixture_trees_payload() -> Vec<u8> {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&artifact).expect("artifact parses");
        parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists")
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= ADDITIVITY_ATOL,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn explain_rows_from_artifact_rejects_empty_rows() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_feature_count_mismatch() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[vec![0.0]]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_non_finite_features() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[vec![f32::NAN, 0.0]]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_incompatible_required_sections() {
        let layout_payload = {
            let strict_artifact = fixture_model()
                .to_artifact_bytes()
                .expect("artifact serializes");
            let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
            parsed
                .sections
                .iter()
                .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
                .map(|section| section.payload.clone())
                .expect("predictor layout payload exists")
        };

        let incompatible_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[(ModelSectionKind::PredictorLayout, layout_payload)],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&incompatible_artifact, &[vec![0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_accepts_legacy_trees_only_artifact() {
        let legacy_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[(ModelSectionKind::Trees, fixture_trees_payload())],
        )
        .expect("artifact serializes");

        let explanation = explain_rows_from_artifact_bytes(&legacy_artifact, &fixture_rows())
            .expect("legacy artifact explains");
        assert_close(explanation.expected_value, 2.25);
        assert_eq!(explanation.values.len(), 4);
        assert_eq!(explanation.values[0].len(), 2);
    }

    #[test]
    fn explain_rows_from_artifact_rejects_duplicate_trees_sections() {
        let trees_payload = fixture_trees_payload();
        let duplicate_trees_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[
                (ModelSectionKind::Trees, trees_payload.clone()),
                (ModelSectionKind::Trees, trees_payload),
            ],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&duplicate_trees_artifact, &[vec![0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_metadata_feature_count_mismatch() {
        let mismatched_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1", "f2"]),
            &[(ModelSectionKind::Trees, fixture_trees_payload())],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&mismatched_artifact, &[vec![0.0, 0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_computes_exact_expected_value_and_contributions() {
        let model = fixture_model();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = fixture_rows();

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        assert_close(explanation.expected_value, 2.25);
        assert_eq!(explanation.values.len(), rows.len());
        for row_values in &explanation.values {
            assert_eq!(row_values.len(), model.feature_count);
        }

        let expected_values = [
            vec![-0.6, -0.05],
            vec![-0.6, 0.05],
            vec![0.6, -0.05],
            vec![0.6, 0.05],
        ];

        for (actual_row, expected_row) in explanation.values.iter().zip(expected_values.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_close(*actual, *expected);
            }
        }

        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn explain_rows_from_artifact_matches_predictor_predictions() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor loads");
        let rows = fixture_rows();

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        for (row_index, row) in rows.iter().enumerate() {
            let predicted = predictor.predict_row(row).expect("predicts");
            let reconstructed =
                explanation.expected_value + explanation.values[row_index].iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn binning_context_linear_matches_predictor_conversion() {
        // The float threshold SHAP computes must equal the float
        // threshold the predictor would compute via
        // `convert_bin_thresholds_to_float`.  Spot-check a few bins.
        let ctx = BinningContext::Linear {
            feature_mins: vec![-2.0, 0.0],
            feature_maxs: vec![3.0, 10.0],
            max_data_bin: 254,
        };
        // Predictor formula: min + ((bin + 0.5) / 254) * (max - min).
        for &bin in &[0u16, 1, 64, 127, 254] {
            let shap_thr_f0 = ctx.float_threshold(0, bin);
            let expected_f0 = -2.0 + ((bin as f32 + 0.5) / 254.0) * 5.0;
            assert!((shap_thr_f0 - expected_f0).abs() < 1e-6);
            let shap_thr_f1 = ctx.float_threshold(1, bin);
            let expected_f1 = 0.0 + ((bin as f32 + 0.5) / 254.0) * 10.0;
            assert!((shap_thr_f1 - expected_f1).abs() < 1e-6);
        }
    }

    #[test]
    fn binning_context_prebinned_matches_predictor_conversion() {
        let ctx = BinningContext::PreBinned;
        for &bin in &[0u16, 1, 64, 127, 254] {
            // Predictor pre-binned: float threshold = bin + 0.5.
            assert!((ctx.float_threshold(0, bin) - (bin as f32 + 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn binning_context_quantile_matches_predictor_conversion() {
        let ctx = BinningContext::Quantile {
            feature_cuts: vec![vec![0.1, 0.5, 0.9], vec![1.0, 2.0, 3.0, 4.0]],
        };
        assert!((ctx.float_threshold(0, 0) - 0.1).abs() < 1e-6);
        assert!((ctx.float_threshold(0, 2) - 0.9).abs() < 1e-6);
        // Past the last cut → f32::MAX.
        assert_eq!(ctx.float_threshold(0, 3), f32::MAX);
        assert_eq!(ctx.float_threshold(1, 4), f32::MAX);
    }

    #[test]
    fn binning_context_linear_rank_inverts_rank_mapping() {
        // For a rank-flagged feature with 5 unique sorted values and
        // max_data_bin = 4, the rank-to-bin formula is bin = round(rank).
        // Threshold conversion: float_threshold(bin) = sorted[r*] where
        // r* = ceil((bin + 0.5) * (N-1) / max_data_bin).
        let sorted = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let ctx = BinningContext::LinearRank {
            per_feature: vec![Some(sorted.clone())],
            feature_mins: vec![1.0],
            feature_maxs: vec![5.0],
            max_data_bin: 4,
        };
        // r* = ceil(0.5 * 4 / 4) = ceil(0.5) = 1 → sorted[1] = 2.0
        assert!((ctx.float_threshold(0, 0) - 2.0).abs() < 1e-6);
        // r* = ceil(1.5 * 4 / 4) = ceil(1.5) = 2 → sorted[2] = 3.0
        assert!((ctx.float_threshold(0, 1) - 3.0).abs() < 1e-6);
        // r* = ceil(2.5 * 4 / 4) = ceil(2.5) = 3 → sorted[3] = 4.0
        assert!((ctx.float_threshold(0, 2) - 4.0).abs() < 1e-6);
        // r* = ceil(3.5 * 4 / 4) = ceil(3.5) = 4 → sorted[4] = 5.0
        assert!((ctx.float_threshold(0, 3) - 5.0).abs() < 1e-6);
        // Bin past the data range clamps to the last sorted value.
        assert!((ctx.float_threshold(0, 4) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn binning_context_linear_rank_falls_back_to_linear_for_non_rank_features() {
        // Feature 0 uses rank binning, feature 1 falls back to standard
        // linear (per_feature[1] is None).
        let sorted = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let ctx = BinningContext::LinearRank {
            per_feature: vec![Some(sorted), None],
            feature_mins: vec![0.0, -2.0],
            feature_maxs: vec![4.0, 3.0],
            max_data_bin: 254,
        };
        // Feature 1 (None) must match the existing Linear formula
        // exactly.
        for &bin in &[0u16, 1, 64, 127, 254] {
            let got = ctx.float_threshold(1, bin);
            let expected = -2.0 + ((bin as f32 + 0.5) / 254.0) * 5.0;
            assert!(
                (got - expected).abs() < 1e-6,
                "bin {bin}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn binning_context_linear_rank_matches_predictor_on_round_trip() {
        // Generate a small sorted-values fixture; for each unique value
        // compute the bin via the predictor's rank-quantize formula,
        // then convert the bin back to a float via float_threshold,
        // and assert the predictor's `value < float_threshold` decision
        // matches the integer-bin comparison `quantized_bin <= bin - 1`.
        let sorted: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();
        let max_data_bin: u16 = 8;
        let ctx = BinningContext::LinearRank {
            per_feature: vec![Some(sorted.clone())],
            feature_mins: vec![sorted[0]],
            feature_maxs: vec![*sorted.last().unwrap()],
            max_data_bin,
        };
        let n = sorted.len();
        // Mimic quantize_rank_value_wide(value, sorted, max_data_bin).
        let quantize = |value: f32| -> u16 {
            let insertion = sorted.partition_point(|probe| *probe <= value);
            let rank = insertion.saturating_sub(1).min(n - 1);
            let scaled = (rank as f32 * max_data_bin as f32) / (n - 1) as f32;
            let rounded = if scaled >= 0.0 {
                (scaled + 0.5).floor() as i32
            } else {
                (scaled - 0.5).ceil() as i32
            };
            rounded.clamp(0, max_data_bin as i32) as u16
        };
        for &threshold_bin in &[0u16, 1, 3, 4, 6, 7] {
            let float_threshold = ctx.float_threshold(0, threshold_bin);
            // Every sorted value should agree on side: the predictor's
            // bin comparison `quantize(v) <= threshold_bin` must equal
            // SHAP's `v < float_threshold`.
            for &value in &sorted {
                let predictor_left = quantize(value) <= threshold_bin;
                let shap_left = value < float_threshold;
                assert_eq!(
                    predictor_left,
                    shap_left,
                    "threshold_bin={threshold_bin}, value={value}, float_threshold={float_threshold}, quantize={}",
                    quantize(value),
                );
            }
        }
    }

    #[test]
    fn binning_context_explanation_matches_predictor_on_constant_leaves() {
        // Build a simple two-tree constant-leaf model with bin-index
        // thresholds.  SHAP without binning would compare raw values
        // against bin indices and reach a different leaf than the
        // predictor (which uses float thresholds).  With binning, SHAP
        // must reach the same leaf and produce strict additivity.
        let model = fixture_model();
        let artifact = model.to_artifact_bytes().expect("serializes");

        // Feature mins/maxs that put raw input values within the
        // float-threshold-converted decision region.  fixture_model
        // splits on feature 0 at bin 2 and feature 1 at bin 1.
        let binning = BinningContext::Linear {
            feature_mins: vec![0.0, 0.0],
            feature_maxs: vec![10.0, 10.0],
            max_data_bin: 254,
        };
        let rows = vec![
            vec![0.05_f32, 0.05_f32], // both below thresholds
            vec![5.0_f32, 5.0_f32],   // both above
        ];

        let explanation = explain_rows_from_artifact_bytes_with_binning(&artifact, &rows, &binning)
            .expect("with-binning explains");
        // Additivity check inside explain enforces strict tolerance
        // when binning is provided and leaves are scalar — if this
        // returns Ok the path walker matched the predictor's path.
        for (row_index, row) in rows.iter().enumerate() {
            let reconstructed =
                explanation.expected_value + explanation.values[row_index].iter().sum::<f32>();
            // Validate against a hand walk via local_path_predict —
            // same code path the verify_additivity call uses internally.
            let predicted = local_path_predict(&model, row, Some(&binning));
            assert!((reconstructed - predicted).abs() < 1e-4);
        }
    }

    #[test]
    fn explain_rows_from_artifact_assigns_zero_to_unused_features() {
        let model = fixture_model_with_unused_feature();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = vec![vec![0.0, 0.0, 5.0], vec![3.0, 2.0, 9.0]];

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        assert_eq!(explanation.values[0].len(), 3);
        assert_close(explanation.values[0][2], 0.0);
        assert_close(explanation.values[1][2], 0.0);
    }

    #[test]
    fn global_importance_aggregates_mean_absolute_contribution() {
        let feature_names = vec!["f0".to_string(), "f1".to_string()];
        let shap_values = vec![
            vec![-0.6, -0.05],
            vec![-0.6, 0.05],
            vec![0.6, -0.05],
            vec![0.6, 0.05],
        ];

        let global = global_importance_from_shap_values(&feature_names, &shap_values)
            .expect("global importance computes");
        assert_close(global[0].1, 0.6);
        assert_close(global[1].1, 0.05);
        assert_eq!(global[0].0, "f0");
        assert_eq!(global[1].0, "f1");
    }

    #[test]
    fn global_importance_from_artifact_uses_metadata_feature_names() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let global = global_importance_from_artifact_bytes(&artifact, &fixture_rows())
            .expect("global computes");

        assert_eq!(global.len(), 2);
        assert_eq!(global[0].0, "f0");
        assert_eq!(global[1].0, "f1");
    }

    #[test]
    fn global_importance_breaks_ties_by_feature_name() {
        let feature_names = vec!["zeta".to_string(), "alpha".to_string(), "beta".to_string()];
        let shap_values = vec![vec![1.0, -1.0, 0.0], vec![-1.0, 1.0, 0.0]];

        let global = global_importance_from_shap_values(&feature_names, &shap_values)
            .expect("global importance computes");

        assert_eq!(global.len(), 3);
        assert_eq!(global[0].0, "alpha");
        assert_eq!(global[1].0, "zeta");
        assert_eq!(global[2].0, "beta");
        assert_close(global[0].1, 1.0);
        assert_close(global[1].1, 1.0);
        assert_close(global[2].1, 0.0);
    }

    #[test]
    fn legacy_stub_helpers_return_deterministic_outputs() {
        let metadata = sample_metadata(&["f0", "f1"]);
        let rows = fixture_rows();
        let shap_values = shap_values_stub(&metadata, &rows).expect("stub values compute");
        assert_eq!(shap_values.len(), rows.len());
        assert_eq!(shap_values[0], vec![0.0, 0.0]);

        let global = global_importance_stub(&metadata, &metadata.feature_names)
            .expect("stub global computes");
        assert_eq!(
            global,
            vec![("f0".to_string(), 0.0), ("f1".to_string(), 0.0)]
        );
    }

    // -------------------------------------------------------------------
    // TreeSHAP tests
    // -------------------------------------------------------------------

    #[test]
    fn tree_shap_matches_brute_force_on_fixture_model() {
        let model = fixture_model();
        let rows = fixture_rows();

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        assert_close(brute_force.expected_value, tree_shap.expected_value);
        assert_eq!(brute_force.values.len(), tree_shap.values.len());

        for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
            assert_eq!(bf_row.len(), ts_row.len());
            for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }
    }

    #[test]
    fn tree_shap_matches_brute_force_on_unused_feature_model() {
        let model = fixture_model_with_unused_feature();
        let rows = vec![vec![0.0, 0.0, 5.0], vec![3.0, 2.0, 9.0]];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
            for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }
        // Feature 2 should be zero in both.
        assert_close(tree_shap.values[0][2], 0.0);
        assert_close(tree_shap.values[1][2], 0.0);
    }

    #[test]
    fn tree_shap_additivity_holds_for_all_rows() {
        let model = fixture_model();
        let rows = fixture_rows();
        let explanation = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn tree_shap_single_stump_model() {
        // A single-tree, single-node (depth-1) model splitting on feature 0.
        let model = TrainedModel {
            baseline_prediction: 1.0,
            feature_count: 2,
            stumps: vec![TrainedStump {
                split: split_with_counts(0, 0, 5, 3, 7),
                left_leaf_value: LeafValue::Scalar(-0.5),
                right_leaf_value: LeafValue::Scalar(0.3),
                tree_weight: 1.0,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![vec![3.0, 0.0], vec![8.0, 0.0]];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
            for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }
    }

    #[test]
    fn tree_shap_multi_tree_model() {
        // Two trees, each with depth 1, splitting on different features.
        let stride = 1u32 << 20;
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 3,
            stumps: vec![
                TrainedStump {
                    split: split_with_counts(0, 0, 5, 4, 6),
                    left_leaf_value: LeafValue::Scalar(1.0),
                    right_leaf_value: LeafValue::Scalar(-1.0),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(stride, 1, 3, 5, 5),
                    left_leaf_value: LeafValue::Scalar(0.5),
                    right_leaf_value: LeafValue::Scalar(-0.5),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![3.0, 1.0, 0.0],
            vec![8.0, 5.0, 0.0],
            vec![3.0, 5.0, 0.0],
        ];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (row_idx, (bf_row, ts_row)) in brute_force
            .values
            .iter()
            .zip(tree_shap.values.iter())
            .enumerate()
        {
            for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }
    }

    /// Regression test for asymmetric-depth TreeSHAP attribution
    /// (added while reviewing PR #27).
    ///
    /// `explain_rows_tree_shap` must match the brute-force exact Shapley
    /// path even when leaves are at varying depths (the common case for
    /// any real model with `min_data_in_leaf`, `min_split_gain`, or
    /// early-stop).  This test holds today on the minimal asymmetric
    /// topology but the polynomial path has a separate, pre-existing
    /// additivity drift on much larger / deeper variable-depth trees
    /// (see Limitation 5 in `docs/limitations.md`).
    #[test]
    fn tree_shap_asymmetric_depth_tree_matches_brute_force_and_predict() {
        // Stumps:
        //   id 0 (root):         feat 0, threshold 1, leaves {1.0, 2.0}, counts l=80 r=20
        //   id 1 (left child):   feat 1, threshold 2, leaves {3.0, 4.0}, counts l=50 r=30 (sum=80)
        //   id 2 (right child):  DOES NOT EXIST — depth-1 early-stop on the right
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: split_with_counts(0, 0, 1, 80, 20),
                    left_leaf_value: LeafValue::Scalar(1.0),
                    right_leaf_value: LeafValue::Scalar(2.0),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(1, 1, 2, 50, 30),
                    left_leaf_value: LeafValue::Scalar(3.0),
                    right_leaf_value: LeafValue::Scalar(4.0),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![0.0, 0.0], // L at root → L at stump 1 → adds 1.0 + 3.0 → predict = 4.5
            vec![0.0, 5.0], // L at root → R at stump 1 → adds 1.0 + 4.0 → predict = 5.5
            vec![5.0, 0.0], // R at root → stump 2 missing → adds 2.0 → predict = 2.5
            vec![5.0, 5.0], // R at root → stump 2 missing → adds 2.0 → predict = 2.5
        ];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        // Brute-force gives the reference (exact 2^N Shapley).
        for (row_idx, (bf_row, ts_row)) in brute_force
            .values
            .iter()
            .zip(tree_shap.values.iter())
            .enumerate()
        {
            for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
                assert!(
                    (bf_val - ts_val).abs() <= 1e-5,
                    "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val} \
                     (this asymmetric-depth tree is what TreeSHAP gets wrong without the v0.7.4 fix)"
                );
            }
        }

        // Independent additivity check against TrainedModel::predict_row.
        for (row_idx, (row, ts_values)) in rows.iter().zip(tree_shap.values.iter()).enumerate() {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = tree_shap.expected_value + ts_values.iter().sum::<f32>();
            assert!(
                (predicted - reconstructed).abs() <= 1e-5,
                "row {row_idx}: predict_row={predicted}, expected_value+Σphi={reconstructed}, \
                 gap={}",
                (predicted - reconstructed).abs()
            );
        }
    }

    /// Spine-tree reproducer: every level only goes deeper on the left
    /// (stumps at 0, 1, 3, 7), missing all right-side and inner descendant
    /// stumps.  Rows reach leaves at depths 1, 2, 3, 4 depending on where
    /// they branch off the spine.  This is the topology most real models
    /// produce when one branch is dominant and others early-stop.
    #[test]
    fn tree_shap_spine_tree_matches_brute_force() {
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 4,
            stumps: vec![
                TrainedStump {
                    split: split_with_counts(0, 0, 1, 70, 30),
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(0.2),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(1, 1, 1, 50, 20),
                    left_leaf_value: LeafValue::Scalar(0.3),
                    right_leaf_value: LeafValue::Scalar(0.4),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(3, 2, 1, 30, 20),
                    left_leaf_value: LeafValue::Scalar(0.5),
                    right_leaf_value: LeafValue::Scalar(0.6),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(7, 3, 1, 20, 10),
                    left_leaf_value: LeafValue::Scalar(0.7),
                    right_leaf_value: LeafValue::Scalar(0.8),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![0.0, 0.0, 0.0, 0.0], // LLLL → visits 0,1,3,7 → pred=baseline+0.1+0.3+0.5+0.7
            vec![0.0, 0.0, 0.0, 5.0], // LLLR → visits 0,1,3,7 → pred=baseline+0.1+0.3+0.5+0.8
            vec![0.0, 0.0, 5.0, 0.0], // LLR_ → visits 0,1,3 → pred=baseline+0.1+0.3+0.6
            vec![0.0, 5.0, 0.0, 0.0], // LR__ → visits 0,1 → pred=baseline+0.1+0.4
            vec![5.0, 0.0, 0.0, 0.0], // R___ → visits 0 → pred=baseline+0.2
        ];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (row_idx, (bf_row, ts_row)) in brute_force
            .values
            .iter()
            .zip(tree_shap.values.iter())
            .enumerate()
        {
            for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
                assert!(
                    (bf_val - ts_val).abs() <= 1e-5,
                    "row {row_idx} feature {feat_idx}: bf={bf_val:.6} ts={ts_val:.6} \
                     gap={:.3e} — TreeSHAP must match brute force on asymmetric spine trees",
                    (bf_val - ts_val).abs()
                );
            }
        }

        // Additivity vs predict_row for each row.
        for (row_idx, (row, ts_values)) in rows.iter().zip(tree_shap.values.iter()).enumerate() {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = tree_shap.expected_value + ts_values.iter().sum::<f32>();
            assert!(
                (predicted - reconstructed).abs() <= 1e-5,
                "row {row_idx}: predict_row={predicted:.6}, expected_value+Σphi={reconstructed:.6}, \
                 gap={:.3e}",
                (predicted - reconstructed).abs()
            );
        }
    }

    /// Build a SplitCandidate with a categorical bitset.
    fn categorical_split_with_counts(
        node_id: u32,
        feature_index: u32,
        bitset: Vec<u8>,
        left_count: u32,
        right_count: u32,
    ) -> SplitCandidate {
        SplitCandidate {
            node_id,
            feature_index,
            threshold_bin: 0,
            gain: 1.0,
            default_left: true,
            is_categorical: true,
            categorical_bitset: Some(bitset),
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: left_count as f32,
                grad_sq_sum: 0.0,
                row_count: left_count,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: right_count as f32,
                grad_sq_sum: 0.0,
                row_count: right_count,
            },
        }
    }

    #[test]
    fn brute_force_categorical_split_additivity() {
        // Single tree with one categorical split on feature 0.
        // Bitset 0b0000_0101 = categories {0, 2} go left; {1, 3} go right.
        let model = TrainedModel {
            baseline_prediction: 1.0,
            feature_count: 2,
            stumps: vec![TrainedStump {
                split: categorical_split_with_counts(0, 0, vec![0b0000_0101], 4, 6),
                left_leaf_value: LeafValue::Scalar(-0.3),
                right_leaf_value: LeafValue::Scalar(0.2),
                tree_weight: 1.0,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        // Feature 0 values: 0.0 (cat 0, left), 1.0 (cat 1, right),
        //                    2.0 (cat 2, left), 3.0 (cat 3, right)
        let rows = vec![
            vec![0.0, 5.0], // cat 0 -> left
            vec![1.0, 5.0], // cat 1 -> right
            vec![2.0, 5.0], // cat 2 -> left
            vec![3.0, 5.0], // cat 3 -> right
        ];

        let explanation = explain_rows_brute_force(&model, &rows, None).expect("brute force works");

        // Verify additivity: sum of SHAP values + expected_value == prediction
        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn tree_shap_categorical_split_additivity() {
        // Same model as brute_force_categorical_split_additivity.
        let model = TrainedModel {
            baseline_prediction: 1.0,
            feature_count: 2,
            stumps: vec![TrainedStump {
                split: categorical_split_with_counts(0, 0, vec![0b0000_0101], 4, 6),
                left_leaf_value: LeafValue::Scalar(-0.3),
                right_leaf_value: LeafValue::Scalar(0.2),
                tree_weight: 1.0,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![0.0, 5.0],
            vec![1.0, 5.0],
            vec![2.0, 5.0],
            vec![3.0, 5.0],
        ];

        let explanation = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn tree_shap_matches_brute_force_on_categorical_model() {
        // Two trees: first uses a categorical split on feature 0, second
        // uses a numeric split on feature 1. This exercises both split types
        // in the same model and verifies the algorithms agree.
        let stride = 1u32 << 20;
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    // Tree 0: categorical split on feature 0
                    // Bitset 0b0000_0011 = categories {0, 1} go left
                    split: categorical_split_with_counts(0, 0, vec![0b0000_0011], 5, 5),
                    left_leaf_value: LeafValue::Scalar(-0.2),
                    right_leaf_value: LeafValue::Scalar(0.3),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    // Tree 1: numeric split on feature 1 at threshold 3
                    split: split_with_counts(stride, 1, 3, 4, 6),
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(-0.1),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![0.0, 1.0], // cat 0 left, numeric left
            vec![1.0, 5.0], // cat 1 left, numeric right
            vec![2.0, 1.0], // cat 2 right, numeric left
            vec![3.0, 5.0], // cat 3 right, numeric right
        ];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        assert_close(brute_force.expected_value, tree_shap.expected_value);

        for (row_idx, (bf_row, ts_row)) in brute_force
            .values
            .iter()
            .zip(tree_shap.values.iter())
            .enumerate()
        {
            for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }

        // Also verify additivity for both algorithms.
        for (row, values) in rows.iter().zip(brute_force.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = brute_force.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn tree_shap_deep_tree_with_repeated_feature() {
        // A single tree of depth 2 that splits on feature 0 at both levels.
        // Root (node 0): split on f0 at 5
        //   Left (node 1): split on f0 at 2
        //   Right (node 2): split on f1 at 3
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: split_with_counts(0, 0, 5, 6, 4),
                    left_leaf_value: LeafValue::Scalar(0.2),
                    right_leaf_value: LeafValue::Scalar(-0.3),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(1, 0, 2, 3, 3),
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(-0.1),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(2, 1, 3, 2, 2),
                    left_leaf_value: LeafValue::Scalar(0.15),
                    right_leaf_value: LeafValue::Scalar(-0.15),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        };

        let rows = vec![
            vec![1.0, 1.0],
            vec![1.0, 5.0],
            vec![4.0, 1.0],
            vec![4.0, 5.0],
            vec![8.0, 1.0],
            vec![8.0, 5.0],
        ];

        let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
        let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

        for (row_idx, (bf_row, ts_row)) in brute_force
            .values
            .iter()
            .zip(tree_shap.values.iter())
            .enumerate()
        {
            for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
                assert!(
                    (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                    "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
                );
            }
        }
    }

    // ── Linear-leaf (piecewise-linear / leaf_model='linear') SHAP tests ─────

    /// Build a 2-feature, 1-stump model whose left/right leaves are linear in
    /// feature 1 with regressor mean 0.5.  Row layout: feature 0 is the split
    /// feature, feature 1 is the linear regressor.
    fn linear_fixture_model(feature_baseline: Option<Vec<f32>>) -> TrainedModel {
        // Tree:  split on feature 0 at bin 1
        //   left  leaf:  intercept=0.4, w=0.7 on feature 1
        //   right leaf:  intercept=-0.2, w=-0.3 on feature 1
        TrainedModel {
            baseline_prediction: 0.1,
            feature_count: 2,
            stumps: vec![TrainedStump {
                split: split_with_counts(0, 0, 1, 6, 4),
                left_leaf_value: LeafValue::Linear(LinearLeaf {
                    intercept: 0.4,
                    weights: vec![0.7],
                    regressor_features: vec![1],
                }),
                right_leaf_value: LeafValue::Linear(LinearLeaf {
                    intercept: -0.2,
                    weights: vec![-0.3],
                    regressor_features: vec![1],
                }),
                tree_weight: 1.0,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline,
        }
    }

    #[test]
    fn shap_linear_leaves_does_not_reject_artifact() {
        // Regression guard: TreeSHAP used to error with `NotSupported` when any
        // leaf was linear; v0.7.1 lifts that and decomposes the leaf instead.
        let model = linear_fixture_model(Some(vec![0.0, 0.5]));
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = vec![vec![0.0, 0.5], vec![3.0, 0.5]];
        let result = explain_rows_from_artifact_bytes(&artifact, &rows);
        assert!(result.is_ok(), "expected ok, got {result:?}");
    }

    #[test]
    fn shap_linear_leaves_additivity_with_baseline_brute_force() {
        // Brute-force exact path (≤ 25 split features).  With a baseline
        // recorded for feature 1, `Σ phi[i] + expected_value == predict(x)`
        // and the path-attribution-vs-linear-deviation split is well-defined.
        let baseline = vec![0.0_f32, 0.5_f32];
        let model = linear_fixture_model(Some(baseline));
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        let rows = vec![
            vec![0.0_f32, 1.0_f32], // goes left
            vec![3.0_f32, 1.0_f32], // goes right
            vec![0.0_f32, -1.0_f32],
            vec![3.0_f32, -1.0_f32],
        ];
        let explanation =
            explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

        let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");
        for (row, phi) in rows.iter().zip(explanation.values.iter()) {
            let predicted = predictor.predict_row(row).expect("predict succeeds");
            let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn shap_linear_leaves_additivity_without_baseline_brute_force() {
        // Back-compat: artifact produced before v0.7.1 will not carry a
        // FeatureBaseline section.  SHAP must still satisfy additivity in
        // that case (treating the global baseline as 0 — degraded
        // interventional decomposition but still exact in aggregate).
        let model = linear_fixture_model(None);
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        let rows = vec![vec![0.0_f32, 1.0_f32], vec![3.0_f32, -0.5_f32]];
        let explanation =
            explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

        let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");
        for (row, phi) in rows.iter().zip(explanation.values.iter()) {
            let predicted = predictor.predict_row(row).expect("predict succeeds");
            let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn shap_linear_leaves_attribute_deviation_to_regressor_feature() {
        // For a row sitting exactly at the baseline of the regressor, all
        // linear-deviation terms vanish and SHAP[regressor] == 0 (any
        // attribution must come purely from path effects).  Conversely, when
        // the regressor sits off-baseline, that feature picks up the
        // deviation w_j * (x_j - μ_j) on top of any path contribution.
        let baseline = vec![0.0_f32, 0.5_f32];
        let model = linear_fixture_model(Some(baseline.clone()));
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        // Row 0: feature 0 = 0 → goes left, feature 1 = 0.5 (= μ_1).
        // Linear deviation w_left * (0.5 - 0.5) = 0.
        let on_baseline_row = vec![0.0_f32, 0.5_f32];
        let off_baseline_row = vec![0.0_f32, 1.5_f32];
        let explanation = explain_rows_from_artifact_bytes(
            &artifact,
            &[on_baseline_row.clone(), off_baseline_row.clone()],
        )
        .expect("explanation succeeds");

        // The two rows take the same path (feature 0 = 0 → left); they only
        // differ in feature 1.  Therefore SHAP[feature 1] must differ by
        // exactly w_left * (1.5 - 0.5) = 0.7 * 1.0 = 0.7.
        let delta_phi_feat1 = explanation.values[1][1] - explanation.values[0][1];
        assert!(
            (delta_phi_feat1 - 0.7).abs() <= ADDITIVITY_ATOL,
            "expected ΔSHAP[feature 1] = 0.7, got {delta_phi_feat1}"
        );
    }

    /// Build a 3-feature 2-stump model that mixes a scalar leaf with a linear
    /// leaf, so we exercise the codepath that has to handle both leaf flavours
    /// within a single tree.
    fn mixed_leaf_fixture_model() -> TrainedModel {
        TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 3,
            stumps: vec![
                TrainedStump {
                    split: split_with_counts(0, 0, 1, 5, 5),
                    // Left leaf: scalar
                    left_leaf_value: LeafValue::Scalar(0.3),
                    // Right child has another split, so the right leaf value
                    // here is the partial contribution along that branch.
                    right_leaf_value: LeafValue::Scalar(-0.1),
                    tree_weight: 1.0,
                },
                TrainedStump {
                    split: split_with_counts(2, 2, 0, 3, 2),
                    left_leaf_value: LeafValue::Linear(LinearLeaf {
                        intercept: 0.1,
                        weights: vec![0.4],
                        regressor_features: vec![1],
                    }),
                    right_leaf_value: LeafValue::Linear(LinearLeaf {
                        intercept: -0.2,
                        weights: vec![0.6],
                        regressor_features: vec![1],
                    }),
                    tree_weight: 1.0,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: Some(vec![0.0, 0.5, 0.0]),
        }
    }

    #[test]
    fn shap_linear_leaves_mixed_with_scalar_leaves_satisfies_additivity() {
        let model = mixed_leaf_fixture_model();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");

        let rows = vec![
            vec![0.0_f32, 0.5_f32, 0.0_f32],  // left scalar leaf
            vec![3.0_f32, 1.0_f32, 0.0_f32],  // right→left linear leaf
            vec![3.0_f32, -0.5_f32, 2.0_f32], // right→right linear leaf
        ];
        let explanation =
            explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

        for (row_idx, (row, phi)) in rows.iter().zip(explanation.values.iter()).enumerate() {
            let predicted = predictor.predict_row(row).expect("predict succeeds");
            let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
            assert!(
                (reconstructed - predicted).abs() <= ADDITIVITY_ATOL,
                "row {row_idx}: reconstructed {reconstructed} vs predicted {predicted}"
            );
        }
    }

    // ── TreeSHAP polynomial-path diagnostic: synthetic deep trees ────────────
    //
    // Used to localize Limitation 5 (TreeSHAP polynomial-path additivity
    // drift on deep trees with many distinct splits).  The strategy:
    // build a synthetic depth-D tree using only F (≤25) distinct features
    // so the brute-force exact path remains tractable, then call BOTH
    // `explain_rows_brute_force` and `explain_rows_tree_shap` directly
    // and require they agree per-feature.  Brute-force is the ground
    // truth (it enumerates all 2^F subsets).
    //
    // We sweep over depth and feature-pattern strategies to find the
    // minimal topology that triggers the polynomial-path bug.

    /// Build a full binary tree of depth `depth` with stumps at every
    /// internal node.  Each stump's feature is chosen via `feature_for`.
    /// Each stump's leaves are scalar values: `leaf_value_for(node_id,
    /// goes_left)`.  Per-stump cover is `cover_for(node_id)`.  Threshold
    /// is `node_id as u16 % 4` (arbitrary; rows below choose splits that
    /// always go a deterministic direction).
    fn build_full_tree(
        feature_count: usize,
        depth: usize,
        feature_for: impl Fn(u32) -> u32,
        leaf_value_for: impl Fn(u32, bool) -> f32,
        cover_for: impl Fn(u32) -> u32,
    ) -> Vec<TrainedStump> {
        // Pre-compute per-leaf covers, then propagate up so each parent's
        // left_stats/right_stats == sum of its descendant leaf covers.
        // Without this consistency, `node.cover()` recursion and the
        // per-stump left_stats.row_count would disagree.
        // node_id convention: children of n are 2n+1 and 2n+2.  For a
        // full tree of depth D: internal nodes have ids [0, 2^D - 1),
        // leaves have ids [2^D - 1, 2^(D+1) - 1).
        let n_leaves = 1u32 << depth; // count
        let leaf_id_start = n_leaves - 1; // first leaf node_id
        let total_nodes = (1u32 << (depth + 1)) - 1; // = internal + leaves
        let mut subtree_cover = vec![0u32; total_nodes as usize];
        for leaf_node_id in leaf_id_start..total_nodes {
            subtree_cover[leaf_node_id as usize] = cover_for(leaf_node_id).max(1);
        }
        // Bottom-up propagation: internal node count = leaf_id_start.
        for node_id in (0..leaf_id_start).rev() {
            let l = subtree_cover[(2 * node_id + 1) as usize];
            let r = subtree_cover[(2 * node_id + 2) as usize];
            subtree_cover[node_id as usize] = l + r;
        }

        let mut stumps = Vec::new();
        for node_id in 0..leaf_id_start {
            let feat = feature_for(node_id);
            let left_count = subtree_cover[(2 * node_id + 1) as usize];
            let right_count = subtree_cover[(2 * node_id + 2) as usize];
            let _ = feature_count; // sanity argument; not used directly
            stumps.push(TrainedStump {
                split: split_with_counts(
                    node_id,
                    feat,
                    (node_id as u16) & 0x3,
                    left_count,
                    right_count,
                ),
                left_leaf_value: LeafValue::Scalar(leaf_value_for(node_id, true)),
                right_leaf_value: LeafValue::Scalar(leaf_value_for(node_id, false)),
                tree_weight: 1.0,
            });
        }
        stumps
    }

    /// Compute the depth of a node_id in a full binary tree
    /// (root = 0, children of n = 2n+1, 2n+2).
    fn node_depth(mut node_id: u32) -> u32 {
        let mut depth = 0;
        while node_id > 0 {
            node_id = (node_id - 1) / 2;
            depth += 1;
        }
        depth
    }

    fn synthetic_deep_model(depth: usize, n_features: usize, _seed: u64) -> TrainedModel {
        let stumps = build_full_tree(
            n_features,
            depth,
            // Feature pattern: feature index = node's depth in the tree.
            // Guarantees every root-to-leaf path uses DISTINCT features
            // (no duplicates), as long as n_features > depth.  This
            // isolates the duplicate-handling code path from other
            // potential bugs.  When n_features <= depth, paths cycle
            // through features (forces duplicates).
            |node_id| node_depth(node_id) % n_features as u32,
            // Leaf value: deterministic pseudo-random in [-1, 1].
            |node_id, goes_left| {
                let h = ((node_id as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D)
                    ^ if goes_left { 0xAAAA } else { 0x5555 }) as u32;
                ((h as f32) / (u32::MAX as f32) - 0.5) * 2.0
            },
            // Cover: weight by node depth so deep nodes have small cover.
            |node_id| {
                // Approximate row count: total / (2^subtree_depth)
                // for node at depth d in a full tree of depth `depth`.
                // For diagnosis we don't need realism, just non-zero.
                let h = ((node_id as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) as u32;
                (h % 100) + 1
            },
        );
        TrainedModel {
            baseline_prediction: 0.0,
            feature_count: n_features,
            stumps,
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
        }
    }

    fn deterministic_rows(feature_count: usize, n_rows: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(n_rows);
        let mut state = seed;
        for _ in 0..n_rows {
            let row = (0..feature_count)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // Choose values in [0, 5) so any threshold <= 3 splits.
                    (((state >> 32) as f32) / (u32::MAX as f32 + 1.0)) * 5.0
                })
                .collect::<Vec<_>>();
            out.push(row);
        }
        out
    }

    /// Regression test for the TreeSHAP polynomial-path additivity drift
    /// closed in v0.7.5 (formerly Limitation 5).
    ///
    /// The bug was in `ts_unextend_path`: when removing a duplicate feature
    /// entry from the path, the function shifted the entire `PathElement`
    /// struct (including `pweight`), clobbering the pweights that the
    /// unwind loop had just carefully computed in place.  The reference
    /// implementation in slundberg/shap stores the four path fields as
    /// four parallel arrays and only shifts the first three (feature_index,
    /// zero_fraction, one_fraction), preserving pweights.
    ///
    /// This sweep builds synthetic full binary trees of varying depth and
    /// distinct-feature count, then asserts that the polynomial TreeSHAP
    /// path agrees with the brute-force exact path per-feature within
    /// floating-point tolerance.  Both `n_features < depth` (forced
    /// path-duplicates) and `n_features >= depth` (no duplicates) are
    /// covered, so the unwind path is exercised across the full matrix.
    ///
    /// Brute-force is the ground truth (it enumerates 2^N subsets).
    /// Capped at depth 7 to keep brute-force tractable.
    #[test]
    fn tree_shap_polynomial_path_matches_brute_force_on_full_trees() {
        for &depth in &[2_usize, 3, 4, 5, 6, 7] {
            for &n_features in &[2_usize, 3, 5, 8, 12] {
                let model = synthetic_deep_model(depth, n_features, 0xABCD_EF01);
                let rows = deterministic_rows(n_features, 4, 0x1234_5678);
                let bf = explain_rows_brute_force(&model, &rows, None)
                    .expect("brute-force exact path succeeds");
                let poly = explain_rows_tree_shap(&model, &rows, None)
                    .expect("polynomial path succeeds (no additivity drift)");

                for (row_idx, (bf_row, poly_row)) in
                    bf.values.iter().zip(poly.values.iter()).enumerate()
                {
                    for (feat_idx, (a, b)) in bf_row.iter().zip(poly_row.iter()).enumerate() {
                        assert!(
                            (a - b).abs() <= 1e-5,
                            "depth={depth} n_features={n_features} row={row_idx} \
                             feat={feat_idx}: brute_force={a}, polynomial={b}, \
                             |diff|={}",
                            (a - b).abs(),
                        );
                    }
                }
            }
        }
    }
}
