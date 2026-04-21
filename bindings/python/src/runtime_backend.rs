//! Runtime backend selection for the PyO3 training entry points.
//!
//! The engine's `Trainer::fit_iterations<B: BackendOps, O: ObjectiveOps>`
//! is generic in the backend, but the Python layer needs to pick one
//! at call time. Rather than rewrite the engine to use
//! `Box<dyn BackendOps>` — which would forfeit monomorphization for
//! every objective × backend combo — we introduce a single concrete
//! [`RuntimeBackend`] enum here that *itself* implements
//! [`BackendOps`] by forwarding each method to the inner variant.
//!
//! Callers resolve a user-provided `device: &str` (one of `"cpu"`,
//! `"metal"`, `"auto"`) via [`resolve_runtime_backend`] and then pass
//! `&backend` into the generic `fit_iterations*` family exactly like
//! the old `CpuBackend` did. The discriminant-check branch cost at
//! each forwarded call is negligible next to the compute inside
//! `build_histograms` / `apply_split` / etc.
//!
//! See DECISION D-004 for the full rationale.

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{
    BinnedMatrix, Device, FeatureTile, GradientPair, HistogramBundle, NodeSlice, NodeStats,
    PartitionResult, RowIndexStorage, SplitCandidate,
};
use alloygbm_engine::{BackendOps, CategoricalFeatureInfo, EngineResult, SplitSelectionOptions};
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
use alloygbm_backend_metal::MetalBackend;

/// Single concrete backend type handed to the training loop.
///
/// On non-macOS builds and on builds with `--no-default-features`
/// (i.e. the `metal` feature disabled) the enum collapses to the
/// `Cpu` variant only; `device="metal"` requests are rejected at
/// [`resolve_runtime_backend`] time with a clear error message.
pub enum RuntimeBackend {
    Cpu(CpuBackend),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(MetalBackend),
}

impl RuntimeBackend {
    /// Canonical lowercase name of the active backend. Used for
    /// logging and artifact-metadata recording.
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeBackend::Cpu(_) => "cpu",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(_) => "metal",
        }
    }

    /// Matching [`Device`] for the active backend. Callers stash this
    /// into [`alloygbm_engine::TrainedModel::trained_device`] before
    /// serializing the artifact so the `trained_device` field in the
    /// artifact's metadata JSON reflects the resolved backend (per
    /// S1.9 handoff).
    pub fn device(&self) -> Device {
        match self {
            RuntimeBackend::Cpu(_) => Device::Cpu,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(_) => Device::Metal,
        }
    }
}

impl std::fmt::Debug for RuntimeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `MetalBackend` does not derive Debug (its internals hold
        // Metal protocol objects that aren't Debug-printable), so we
        // emit a stable variant name only — enough for
        // `unwrap_err()`-style test diagnostics without forcing the
        // backend crates to derive Debug themselves.
        f.debug_tuple("RuntimeBackend").field(&self.name()).finish()
    }
}

impl BackendOps for RuntimeBackend {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        match self {
            RuntimeBackend::Cpu(b) => {
                b.build_histograms(binned_matrix, gradients, node, feature_tiles)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => {
                b.build_histograms(binned_matrix, gradients, node, feature_tiles)
            }
        }
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        match self {
            RuntimeBackend::Cpu(b) => b.best_split(histograms),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => b.best_split(histograms),
        }
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        match self {
            RuntimeBackend::Cpu(b) => b.best_split_with_options(
                histograms,
                options,
                feature_weights,
                categorical_features,
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => b.best_split_with_options(
                histograms,
                options,
                feature_weights,
                categorical_features,
            ),
        }
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        match self {
            RuntimeBackend::Cpu(b) => b.apply_split(binned_matrix, node, split),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => b.apply_split(binned_matrix, node, split),
        }
    }

    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        match self {
            RuntimeBackend::Cpu(b) => {
                b.apply_split_with_stats(binned_matrix, gradients, node, split)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => {
                b.apply_split_with_stats(binned_matrix, gradients, node, split)
            }
        }
    }

    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        rows: &RowIndexStorage,
    ) -> EngineResult<NodeStats> {
        match self {
            RuntimeBackend::Cpu(b) => b.reduce_sums(gradients, rows),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => b.reduce_sums(gradients, rows),
        }
    }
}

/// Validate a user-supplied `device` string and build the matching
/// [`RuntimeBackend`].
///
/// Accepted values (case-insensitive, whitespace-trimmed):
/// - `"cpu"` — always returns [`RuntimeBackend::Cpu`].
/// - `"metal"` — returns [`RuntimeBackend::Metal`] on macOS with the
///   `metal` feature compiled in; otherwise returns `Err`. On
///   macOS/metal builds, Metal initialisation failures also surface
///   as `Err` here. Warn-and-fallback to CPU is the job of S1.9 at
///   the PyO3 entry point.
/// - `"auto"` — in S1.7 this is an alias for `"cpu"`. The Stage 2+
///   heuristic (select Metal when rows × features × bin-count crosses
///   the break-even shape) is intentionally deferred so we can ship
///   the plumbing without locking in a heuristic we haven't measured.
///
/// Any other value returns `Err` with the list of accepted options.
/// The error string is plain so callers can wrap it into either
/// `EngineError::InvalidConfig` (for Rust-level failures) or
/// `PyValueError` (for PyO3-level).
#[allow(dead_code)] // retained as a pure (no-warn) companion to `resolve_runtime_backend_with_fallback`; used by unit tests.
pub fn resolve_runtime_backend(device: &str) -> Result<RuntimeBackend, String> {
    let normalized = device.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "cpu" | "auto" => Ok(RuntimeBackend::Cpu(CpuBackend)),
        "metal" => build_metal_backend(),
        other => Err(format!(
            "device must be one of 'cpu', 'metal', or 'auto'; got '{other}'"
        )),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn build_metal_backend() -> Result<RuntimeBackend, String> {
    // Test/ops escape hatch: setting `ALLOYGBM_METAL_DISABLE=1` forces
    // the Metal constructor to report a synthetic failure so the
    // warn-and-fallback path is exercisable from Python tests and from
    // users reproducing issues on machines that *do* have Metal. The
    // message is intentionally unique so tests can assert against it.
    if std::env::var_os("ALLOYGBM_METAL_DISABLE")
        .map(|value| value == "1")
        .unwrap_or(false)
    {
        return Err(
            "Metal backend disabled via ALLOYGBM_METAL_DISABLE=1 (test escape hatch)".to_string(),
        );
    }
    MetalBackend::new()
        .map(RuntimeBackend::Metal)
        .map_err(|msg| format!("could not initialise Metal backend: {msg}"))
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn build_metal_backend() -> Result<RuntimeBackend, String> {
    Err(
        "device='metal' requires macOS with the 'metal' feature enabled; \
         this build does not include the Metal backend"
            .to_string(),
    )
}

/// PyO3-layer companion to [`resolve_runtime_backend`] that applies
/// the S1.9 warn-and-fallback policy:
///
/// - `"cpu"` → always returns [`RuntimeBackend::Cpu`], no warning.
/// - `"auto"` → aliases to `"cpu"` in Stage 1, no warning. The real
///   shape-based heuristic (pick Metal when rows × features ×
///   bin-count crosses the break-even shape) is deferred to Stage 2+.
/// - `"metal"` → try [`MetalBackend::new()`]; on success returns
///   [`RuntimeBackend::Metal`]. On failure (headless CI, Intel Mac,
///   MetalBackend constructor error, or non-macOS build) emits a
///   Python [`RuntimeWarning`] explaining the fallback and returns
///   [`RuntimeBackend::Cpu`]. Unlike [`resolve_runtime_backend`],
///   this never propagates the Metal-init error as a hard failure —
///   per the plan, the user experience of `device="metal"` on a
///   machine without Metal is "fall back transparently, warn once."
/// - Anything else → `PyValueError` listing the accepted options.
///
/// `warn_source` is prepended to the warning text so the message
/// makes sense in context (e.g. `"GBMRegressor.fit"` or
/// `"GBMClassifier.fit"`); pass `"train"` if unsure.
pub fn resolve_runtime_backend_with_fallback(
    py: Python<'_>,
    device: &str,
    warn_source: &str,
) -> Result<RuntimeBackend, String> {
    let normalized = device.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "cpu" | "auto" => Ok(RuntimeBackend::Cpu(CpuBackend)),
        "metal" => match build_metal_backend() {
            Ok(backend) => Ok(backend),
            Err(err) => {
                emit_metal_fallback_warning(py, warn_source, &err)?;
                Ok(RuntimeBackend::Cpu(CpuBackend))
            }
        },
        other => Err(format!(
            "device must be one of 'cpu', 'metal', or 'auto'; got '{other}'"
        )),
    }
}

/// Emit `RuntimeWarning` for the Metal-init fallback path. Swallowed
/// errors would be a UX regression (the user would see a silent
/// fallback), so any failure to construct or raise the warning is
/// propagated up as a plain String — callers wrap that into
/// `EngineError::InvalidConfig` which surfaces as `PyValueError` at
/// the Python boundary.
fn emit_metal_fallback_warning(
    py: Python<'_>,
    warn_source: &str,
    reason: &str,
) -> Result<(), String> {
    let warning_type = py.get_type::<PyRuntimeWarning>();
    let message = std::ffi::CString::new(format!(
        "{warn_source}: device='metal' requested but Metal backend is unavailable \
         ({reason}); falling back to CPU. Set device='cpu' to silence this warning."
    ))
    .map_err(|err| format!("internal: Metal fallback warning contained a null byte: {err}"))?;
    PyErr::warn(py, &warning_type, &message, 1)
        .map_err(|err| format!("failed to emit Metal fallback RuntimeWarning: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cpu_variant() {
        let backend = resolve_runtime_backend("cpu").expect("cpu");
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn resolve_auto_aliases_to_cpu_in_stage_1() {
        let backend = resolve_runtime_backend("auto").expect("auto");
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn resolve_is_case_insensitive_and_trims() {
        assert_eq!(resolve_runtime_backend("  CPU  ").unwrap().name(), "cpu");
        assert_eq!(resolve_runtime_backend("Auto").unwrap().name(), "cpu");
    }

    #[test]
    fn resolve_rejects_unknown_device() {
        let err = resolve_runtime_backend("tpu").unwrap_err();
        assert!(err.contains("'cpu'"));
        assert!(err.contains("'metal'"));
        assert!(err.contains("'auto'"));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn resolve_metal_on_macos_with_feature() {
        match resolve_runtime_backend("metal") {
            Ok(backend) => assert_eq!(backend.name(), "metal"),
            Err(_) => {
                // Headless CI may not have a Metal device; that's
                // a platform/availability error rather than a parse
                // error, and here we just accept it.
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn resolve_metal_off_macos_rejects() {
        let err = resolve_runtime_backend("metal").unwrap_err();
        assert!(err.contains("macOS"));
    }
}
