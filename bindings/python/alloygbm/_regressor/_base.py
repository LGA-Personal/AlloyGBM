"""Module-level constants, native loader stubs, env-toggle helpers, and base class for GBMRegressor."""

from __future__ import annotations

import math
import os

_PRE_BINNED_INTEGER_TOLERANCE = 1e-6
_MAX_CONTINUOUS_QUANTIZED_BIN_U8 = 254
_MISSING_BIN_U8 = 255
_MAX_CONTINUOUS_QUANTIZED_BIN = 65534


def _max_data_bin_for_max_bins(max_bins):
    """Return the highest data bin index for a given max_bins setting.

    For max_bins=256 (default): max_data_bin=254, nan_bin=255 (u8 path).
    For max_bins=512: max_data_bin=510, nan_bin=511 (u16 path).
    """
    return max_bins - 2


def _nan_bin_for_max_bins(max_bins):
    """Return the NaN sentinel bin index for a given max_bins setting."""
    return max_bins - 1
_MIN_CONTINUOUS_QUANTIZED_BINS = 2
_VALID_CONTINUOUS_BINNING_STRATEGIES = {"linear", "rank", "quantile"}
# Per-round boosting strategies.  "standard" is the default v0.7.5
# behaviour, "goss" (v0.8.0+) is LightGBM-style gradient-based one-side
# sampling, "dart" (v0.9.0+) is Dropouts-meet-MART.  See
# crates/core/src/lib.rs::BoostingMode.
_VALID_BOOSTING_MODES = {"standard", "goss", "dart"}
_LINEAR_TAIL_RANK_ENV_VAR = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK"
_LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO"
_DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD = 0.10


def _load_native_predictor_predict_batch():
    try:
        from alloygbm._alloygbm import predictor_predict_batch
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch


def _load_native_predictor_predict_batch_dense():
    try:
        from alloygbm._alloygbm import predictor_predict_batch_dense
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch_dense


def _load_native_predictor_predict_batch_canonical():
    try:
        from alloygbm._alloygbm import predictor_predict_batch_canonical
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native canonical predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch_canonical


def _load_native_predictor_predict_batch_canonical_dense():
    try:
        from alloygbm._alloygbm import predictor_predict_batch_canonical_dense
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense canonical predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch_canonical_dense


def _load_native_predictor_handle_class():
    try:
        from alloygbm._alloygbm import NativePredictorHandle
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native predictor handle binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return NativePredictorHandle


def _load_native_train_regression_artifact():
    try:
        from alloygbm._alloygbm import train_regression_artifact
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native training binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return train_regression_artifact


def _load_native_train_regression_artifact_dense():
    try:
        from alloygbm._alloygbm import train_regression_artifact_dense
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense training binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return train_regression_artifact_dense


def _load_native_train_regression_artifact_with_summary():
    try:
        from alloygbm._alloygbm import train_regression_artifact_with_summary
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native training summary binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return train_regression_artifact_with_summary


def _load_native_train_regression_artifact_dense_with_summary():
    try:
        from alloygbm._alloygbm import train_regression_artifact_dense_with_summary
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense training summary binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return train_regression_artifact_dense_with_summary


def _load_native_shap_explain_rows():
    try:
        from alloygbm._alloygbm import shap_explain_rows
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native SHAP explain binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_explain_rows


def _load_native_shap_explain_rows_dense():
    try:
        from alloygbm._alloygbm import shap_explain_rows_dense
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense SHAP explain binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_explain_rows_dense


def _load_native_shap_global_importance():
    try:
        from alloygbm._alloygbm import shap_global_importance
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native SHAP global-importance binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_global_importance


def _load_native_shap_global_importance_dense():
    try:
        from alloygbm._alloygbm import shap_global_importance_dense
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native dense SHAP global-importance binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_global_importance_dense


# v0.7.3 predictor-aligned SHAP variants.  These accept binning kwargs
# (`feature_mins`, `feature_maxs`, `max_data_bin`, or `feature_cuts`)
# so SHAP's path walker uses the same float thresholds the predictor
# uses, lifting the legacy "best-effort" exemption for
# `leaf_model="linear"` artifacts on continuous features.
def _load_native_shap_explain_rows_with_binning():
    try:
        from alloygbm._alloygbm import shap_explain_rows_with_binning
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "native predictor-aligned SHAP binding is unavailable; rebuild the alloygbm extension module"
        ) from exc
    return shap_explain_rows_with_binning


def _load_native_shap_explain_rows_dense_with_binning():
    try:
        from alloygbm._alloygbm import shap_explain_rows_dense_with_binning
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "native predictor-aligned dense SHAP binding is unavailable; rebuild the alloygbm extension module"
        ) from exc
    return shap_explain_rows_dense_with_binning


def _load_native_shap_global_importance_with_binning():
    try:
        from alloygbm._alloygbm import shap_global_importance_with_binning
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "native predictor-aligned SHAP global-importance binding is unavailable; rebuild the alloygbm extension module"
        ) from exc
    return shap_global_importance_with_binning


def _load_native_shap_global_importance_dense_with_binning():
    try:
        from alloygbm._alloygbm import shap_global_importance_dense_with_binning
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "native predictor-aligned dense SHAP global-importance binding is unavailable; rebuild the alloygbm extension module"
        ) from exc
    return shap_global_importance_dense_with_binning


def _parse_env_toggle(env_name: str) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _linear_tail_rank_enabled_from_env() -> bool:
    return _parse_env_toggle(_LINEAR_TAIL_RANK_ENV_VAR)


def _linear_tail_core_span_ratio_threshold_from_env() -> float:
    value = os.environ.get(_LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR)
    if value is None:
        return _DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD
    normalized = value.strip()
    if normalized == "":
        return _DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD
    try:
        parsed = float(normalized)
    except ValueError:
        return _DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD
    if not math.isfinite(parsed):
        return _DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD
    return min(1.0, max(0.0, parsed))


try:
    from sklearn.base import BaseEstimator, RegressorMixin

    class _GBMRegressorBase(BaseEstimator, RegressorMixin):
        pass

    _SKLEARN_AVAILABLE = True
except ImportError:

    class _GBMRegressorBase:  # type: ignore[no-redef]
        pass

    _SKLEARN_AVAILABLE = False


def _diagnostics_to_dicts(diagnostics):
    """Convert a list of native ``IterationDiagnostics`` objects into a list
    of plain Python dicts.

    The dict keys mirror the Rust struct field names one-to-one. Returns
    ``None`` when ``diagnostics`` is missing/empty so unfitted models surface
    ``diagnostics_per_round_ is None`` cleanly.
    """
    if not diagnostics:
        return None

    def _opt(value):
        return float(value) if value is not None else None

    return [
        {
            "gradient_l2_norm": float(d.gradient_l2_norm),
            "gradient_variance": float(d.gradient_variance),
            "hessian_l2_norm": float(d.hessian_l2_norm),
            "original_gradient_l2_norm": _opt(d.original_gradient_l2_norm),
            "projected_gradient_l2_norm": _opt(d.projected_gradient_l2_norm),
            "neutralization_effectiveness": _opt(d.neutralization_effectiveness),
            "n_active_rows": int(d.n_active_rows),
            "n_active_features": int(d.n_active_features),
        }
        for d in diagnostics
    ]

def _validate_quantile_alpha(quantile_alpha: float) -> None:
    if not (0.0 < quantile_alpha < 1.0):
        raise ValueError("quantile_alpha must be in (0.0, 1.0)")
