use alloygbm_engine::IterationDiagnostics;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct NativeRuntimeInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: String,
}

#[pymethods]
impl NativeRuntimeInfo {
    #[new]
    fn new() -> Self {
        Self {
            name: "alloygbm".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[pyfunction]
pub(crate) fn native_runtime_info() -> NativeRuntimeInfo {
    NativeRuntimeInfo::new()
}

#[derive(Debug, Clone)]
pub(crate) struct ContinuousBinningMetadataInternal {
    pub(crate) uses_continuous_binning: bool,
    pub(crate) feature_mins: Option<Vec<f32>>,
    pub(crate) feature_maxs: Option<Vec<f32>>,
    pub(crate) feature_sorted_values: Option<Vec<Vec<f32>>>,
    pub(crate) feature_quantile_cuts: Option<Vec<Vec<f32>>>,
    pub(crate) feature_linear_rank_flags: Option<Vec<bool>>,
}

impl ContinuousBinningMetadataInternal {
    pub(crate) fn pre_binned() -> Self {
        Self {
            uses_continuous_binning: false,
            feature_mins: None,
            feature_maxs: None,
            feature_sorted_values: None,
            feature_quantile_cuts: None,
            feature_linear_rank_flags: None,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct NativeContinuousBinningMetadata {
    #[pyo3(get)]
    uses_continuous_binning: bool,
    #[pyo3(get)]
    feature_mins: Option<Vec<f32>>,
    #[pyo3(get)]
    feature_maxs: Option<Vec<f32>>,
    #[pyo3(get)]
    feature_sorted_values: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    feature_quantile_cuts: Option<Vec<Vec<f32>>>,
    #[pyo3(get)]
    feature_linear_rank_flags: Option<Vec<bool>>,
}

impl From<ContinuousBinningMetadataInternal> for NativeContinuousBinningMetadata {
    fn from(value: ContinuousBinningMetadataInternal) -> Self {
        Self {
            uses_continuous_binning: value.uses_continuous_binning,
            feature_mins: value.feature_mins,
            feature_maxs: value.feature_maxs,
            feature_sorted_values: value.feature_sorted_values,
            feature_quantile_cuts: value.feature_quantile_cuts,
            feature_linear_rank_flags: value.feature_linear_rank_flags,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct NativeTrainingSummary {
    #[pyo3(get)]
    pub(crate) rounds_requested: usize,
    #[pyo3(get)]
    pub(crate) rounds_completed: usize,
    #[pyo3(get)]
    pub(crate) best_validation_round: Option<usize>,
    #[pyo3(get)]
    pub(crate) best_validation_loss: Option<f32>,
    #[pyo3(get)]
    pub(crate) train_rmse: Vec<f32>,
    #[pyo3(get)]
    pub(crate) validation_rmse: Vec<f32>,
    /// Raw objective loss per completed round (no sqrt transform).
    #[pyo3(get)]
    pub(crate) train_loss: Vec<f32>,
    /// Raw validation objective loss per completed round (no sqrt transform).
    #[pyo3(get)]
    pub(crate) validation_loss: Vec<f32>,
    /// Objective name (e.g. "squared_error", "binary_crossentropy").
    #[pyo3(get)]
    pub(crate) objective: String,
    #[pyo3(get)]
    pub(crate) stop_reason: String,
    #[pyo3(get)]
    pub(crate) bridge_prepare_seconds: f64,
    #[pyo3(get)]
    pub(crate) native_train_seconds: f64,
    /// Custom eval metric values per completed round (empty when no custom metric).
    #[pyo3(get)]
    pub(crate) custom_metric_values: Vec<f32>,
    /// Custom eval metric name (None when no custom metric).
    #[pyo3(get)]
    pub(crate) custom_metric_name: Option<String>,
    /// Per-round diagnostic snapshot.  Each entry is a `NativeIterationDiagnostics`
    /// pyclass exposing the fields of `engine::IterationDiagnostics`.  Length
    /// equals `rounds_completed` after a successful fit.
    #[pyo3(get)]
    pub(crate) diagnostics_per_round: Vec<NativeIterationDiagnostics>,
}

/// Python-visible view of an `engine::IterationDiagnostics` snapshot.  Field
/// names mirror the Rust struct one-to-one.  Projection-related fields are
/// `None` when factor neutralization isn't active for the fit.
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct NativeIterationDiagnostics {
    #[pyo3(get)]
    gradient_l2_norm: f32,
    #[pyo3(get)]
    gradient_variance: f32,
    #[pyo3(get)]
    hessian_l2_norm: f32,
    #[pyo3(get)]
    original_gradient_l2_norm: Option<f32>,
    #[pyo3(get)]
    projected_gradient_l2_norm: Option<f32>,
    /// `1 - projected_l2 / original_l2`, clamped to `[0, 1]`.  Higher means
    /// more gradient signal was projected away.
    #[pyo3(get)]
    neutralization_effectiveness: Option<f32>,
    #[pyo3(get)]
    n_active_rows: usize,
    #[pyo3(get)]
    n_active_features: usize,
}

impl From<&IterationDiagnostics> for NativeIterationDiagnostics {
    fn from(value: &IterationDiagnostics) -> Self {
        Self {
            gradient_l2_norm: value.gradient_l2_norm,
            gradient_variance: value.gradient_variance,
            hessian_l2_norm: value.hessian_l2_norm,
            original_gradient_l2_norm: value.original_gradient_l2_norm,
            projected_gradient_l2_norm: value.projected_gradient_l2_norm,
            neutralization_effectiveness: value.neutralization_effectiveness,
            n_active_rows: value.n_active_rows,
            n_active_features: value.n_active_features,
        }
    }
}

/// Convert a slice of engine diagnostics into the Python-visible pyclass
/// vector.  Used by both `build_native_training_summary` variants.
pub(crate) fn diagnostics_to_native(
    entries: &[IterationDiagnostics],
) -> Vec<NativeIterationDiagnostics> {
    entries
        .iter()
        .map(NativeIterationDiagnostics::from)
        .collect()
}

#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct NativeTrainingResult {
    #[pyo3(get)]
    pub(crate) artifact_bytes: Vec<u8>,
    #[pyo3(get)]
    pub(crate) summary: NativeTrainingSummary,
    #[pyo3(get)]
    pub(crate) continuous_binning_metadata: NativeContinuousBinningMetadata,
    /// Per-feature category→ID mappings for native categorical splits.
    /// Keys are feature indices, values are dicts {category_name: integer_id}.
    #[pyo3(get)]
    pub(crate) native_cat_mappings:
        std::collections::HashMap<usize, std::collections::HashMap<String, u32>>,
}
