"""Python-facing estimator baseline scaffold for AlloyGBM."""

from __future__ import annotations

import bisect
import math
import os
import struct
import time
from collections.abc import Sequence

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
_VALID_DEVICES = {"cpu", "metal", "auto"}
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


class GBMRegressor(_GBMRegressorBase):
    """Gradient Boosted Decision Tree regressor with sklearn-compatible API."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 6,
        row_subsample: float = 1.0,
        col_subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        min_validation_improvement: float = 0.0,
        min_data_in_leaf: int = 1,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_child_hessian: float = 0.0,
        min_split_gain: float = 0.0,
        seed: int = 0,
        deterministic: bool = True,
        continuous_binning_strategy: str = "linear",
        continuous_binning_max_bins: int = 256,
        categorical_feature_index: int | None = None,
        categorical_feature_indices: list[int] | None = None,
        training_policy: str = "auto",
        store_node_stats: bool = False,
        categorical_smoothing: float = 20.0,
        categorical_min_samples_leaf: int = 1,
        categorical_time_aware: bool = False,
        monotone_constraints: list[int] | dict[int, int] | None = None,
        feature_weights: list[float] | dict[int, float] | None = None,
        max_leaves: int | None = None,
        tree_growth: str = "level",
        warm_start: bool = False,
        objective: "str | None | object" = None,
        max_cat_threshold: int = 0,
        device: str = "cpu",
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if not (0.0 < row_subsample <= 1.0):
            raise ValueError("row_subsample must be in (0.0, 1.0]")
        if not (0.0 < col_subsample <= 1.0):
            raise ValueError("col_subsample must be in (0.0, 1.0]")
        if early_stopping_rounds is not None and int(early_stopping_rounds) <= 0:
            raise ValueError("early_stopping_rounds must be greater than 0 when set")
        if min_validation_improvement < 0.0:
            raise ValueError("min_validation_improvement must be >= 0")
        if int(min_data_in_leaf) <= 0:
            raise ValueError("min_data_in_leaf must be greater than 0")
        if lambda_l1 < 0.0:
            raise ValueError("lambda_l1 must be >= 0")
        if lambda_l2 < 0.0:
            raise ValueError("lambda_l2 must be >= 0")
        if min_child_hessian < 0.0:
            raise ValueError("min_child_hessian must be >= 0")
        if not math.isfinite(min_split_gain) or min_split_gain < 0.0:
            raise ValueError("min_split_gain must be a finite value >= 0")
        if categorical_feature_index is not None and int(categorical_feature_index) < 0:
            raise ValueError("categorical_feature_index must be >= 0 when set")
        if categorical_feature_indices is not None:
            if not isinstance(categorical_feature_indices, (list, tuple)):
                raise TypeError("categorical_feature_indices must be a list of ints")
            for idx in categorical_feature_indices:
                if int(idx) < 0:
                    raise ValueError(
                        "all values in categorical_feature_indices must be >= 0"
                    )
            if len(set(int(i) for i in categorical_feature_indices)) != len(
                categorical_feature_indices
            ):
                raise ValueError(
                    "categorical_feature_indices must not contain duplicates"
                )
        if categorical_feature_index is not None and categorical_feature_indices is not None:
            raise ValueError(
                "categorical_feature_index and categorical_feature_indices are mutually exclusive; use categorical_feature_indices for multiple columns"
            )
        if categorical_smoothing < 0.0:
            raise ValueError("categorical_smoothing must be >= 0")
        if int(categorical_min_samples_leaf) <= 0:
            raise ValueError("categorical_min_samples_leaf must be greater than 0")
        if training_policy not in {"auto", "manual"}:
            raise ValueError("training_policy must be 'auto' or 'manual'")
        if continuous_binning_strategy not in _VALID_CONTINUOUS_BINNING_STRATEGIES:
            raise ValueError(
                "continuous_binning_strategy must be one of: "
                + ", ".join(sorted(_VALID_CONTINUOUS_BINNING_STRATEGIES))
            )
        max_bins = int(continuous_binning_max_bins)
        if not (
            _MIN_CONTINUOUS_QUANTIZED_BINS
            <= max_bins
            <= (_MAX_CONTINUOUS_QUANTIZED_BIN + 1)
        ):
            raise ValueError(
                "continuous_binning_max_bins must be in "
                f"[{_MIN_CONTINUOUS_QUANTIZED_BINS}, {_MAX_CONTINUOUS_QUANTIZED_BIN + 1}]"
            )
        if monotone_constraints is not None:
            if isinstance(monotone_constraints, dict):
                for v in monotone_constraints.values():
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
            else:
                for v in monotone_constraints:
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
        if feature_weights is not None:
            vals = (
                feature_weights.values()
                if isinstance(feature_weights, dict)
                else feature_weights
            )
            for w in vals:
                if not math.isfinite(float(w)) or float(w) < 0.0:
                    raise ValueError(
                        "feature_weights values must be finite and >= 0"
                    )
        if max_leaves is not None:
            if int(max_leaves) < 2:
                raise ValueError(
                    "max_leaves must be >= 2 when set"
                )
        _VALID_TREE_GROWTH = {"level", "leaf"}
        if tree_growth not in _VALID_TREE_GROWTH:
            raise ValueError(
                "tree_growth must be one of: " + ", ".join(sorted(_VALID_TREE_GROWTH))
            )
        if tree_growth == "leaf" and max_leaves is None:
            raise ValueError(
                "max_leaves must be set when tree_growth='leaf'"
            )
        if objective is not None and not callable(objective) and not isinstance(objective, str):
            raise TypeError(
                "objective must be a string, a callable, or None"
            )
        if int(max_cat_threshold) < 0:
            raise ValueError("max_cat_threshold must be >= 0")
        if device not in _VALID_DEVICES:
            raise ValueError(
                "device must be one of: " + ", ".join(sorted(_VALID_DEVICES))
            )

        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.n_estimators = int(n_estimators)
        self.row_subsample = float(row_subsample)
        self.col_subsample = float(col_subsample)
        self.early_stopping_rounds = (
            int(early_stopping_rounds)
            if early_stopping_rounds is not None
            else None
        )
        self.min_validation_improvement = float(min_validation_improvement)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.min_child_hessian = float(min_child_hessian)
        self.min_split_gain = float(min_split_gain)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.continuous_binning_strategy = str(continuous_binning_strategy)
        self.continuous_binning_max_bins = max_bins
        self.categorical_feature_index = (
            int(categorical_feature_index)
            if categorical_feature_index is not None
            else None
        )
        self.categorical_feature_indices = (
            [int(i) for i in categorical_feature_indices]
            if categorical_feature_indices is not None
            else None
        )
        self.training_policy = str(training_policy)
        self.store_node_stats = bool(store_node_stats)
        self.categorical_smoothing = float(categorical_smoothing)
        self.categorical_min_samples_leaf = int(categorical_min_samples_leaf)
        self.categorical_time_aware = bool(categorical_time_aware)
        self.monotone_constraints = (
            monotone_constraints if monotone_constraints is not None else None
        )
        self.feature_weights = (
            feature_weights if feature_weights is not None else None
        )
        self.max_leaves = int(max_leaves) if max_leaves is not None else None
        self.tree_growth = str(tree_growth)
        self.warm_start = bool(warm_start)
        self.objective = objective
        self.max_cat_threshold = int(max_cat_threshold)
        self.device = str(device)
        self._is_fitted = False
        self._artifact_bytes: bytes | None = None
        self._native_predictor_handle: object | None = None
        self._float_thresholds_converted: bool = False
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins: list[float] | None = None
        self._continuous_feature_maxs: list[float] | None = None
        self._continuous_feature_sorted_values: list[list[float]] | None = None
        self._continuous_feature_quantile_cuts: list[list[float]] | None = None
        self._continuous_feature_linear_rank_flags: list[bool] | None = None
        self.feature_names_in_: list[str] | None = None
        self._native_cat_mappings_: dict[int, dict[str, int]] | None = None
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.n_estimators_: int | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] | None = None
        self.fit_timing_: dict[str, float] | None = None

    def __repr__(self) -> str:
        return (
            "GBMRegressor("
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"n_estimators={self.n_estimators}, "
            f"row_subsample={self.row_subsample}, "
            f"col_subsample={self.col_subsample}, "
            f"early_stopping_rounds={self.early_stopping_rounds}, "
            f"min_validation_improvement={self.min_validation_improvement}, "
            f"min_data_in_leaf={self.min_data_in_leaf}, "
            f"lambda_l1={self.lambda_l1}, "
            f"lambda_l2={self.lambda_l2}, "
            f"min_child_hessian={self.min_child_hessian}, "
            f"min_split_gain={self.min_split_gain}, "
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"continuous_binning_strategy='{self.continuous_binning_strategy}', "
            f"continuous_binning_max_bins={self.continuous_binning_max_bins}, "
            f"categorical_feature_index={self.categorical_feature_index}, "
            f"categorical_feature_indices={self.categorical_feature_indices}, "
            f"training_policy='{self.training_policy}', "
            f"store_node_stats={self.store_node_stats}, "
            f"categorical_smoothing={self.categorical_smoothing}, "
            f"categorical_min_samples_leaf={self.categorical_min_samples_leaf}, "
            f"categorical_time_aware={self.categorical_time_aware}, "
            f"monotone_constraints={self.monotone_constraints}, "
            f"feature_weights={self.feature_weights}, "
            f"max_leaves={self.max_leaves}, "
            f"tree_growth='{self.tree_growth}', "
            f"warm_start={self.warm_start}, "
            f"objective={self.objective!r}, "
            f"max_cat_threshold={self.max_cat_threshold}, "
            f"device='{self.device}'"
            ")"
        )

    def get_params(self, deep: bool = True) -> dict[str, float | int | bool | None]:
        """Return estimator parameters in sklearn-compatible shape."""
        del deep  # Not used until nested estimators exist.
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "row_subsample": self.row_subsample,
            "col_subsample": self.col_subsample,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_validation_improvement": self.min_validation_improvement,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_child_hessian": self.min_child_hessian,
            "min_split_gain": self.min_split_gain,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "continuous_binning_strategy": self.continuous_binning_strategy,
            "continuous_binning_max_bins": self.continuous_binning_max_bins,
            "categorical_feature_index": self.categorical_feature_index,
            "categorical_feature_indices": self.categorical_feature_indices,
            "training_policy": self.training_policy,
            "store_node_stats": self.store_node_stats,
            "categorical_smoothing": self.categorical_smoothing,
            "categorical_min_samples_leaf": self.categorical_min_samples_leaf,
            "categorical_time_aware": self.categorical_time_aware,
            "monotone_constraints": self.monotone_constraints,
            "feature_weights": self.feature_weights,
            "max_leaves": self.max_leaves,
            "tree_growth": self.tree_growth,
            "warm_start": self.warm_start,
            "objective": self.objective,
            "max_cat_threshold": self.max_cat_threshold,
            "device": self.device,
        }

    def set_params(self, **params: float | int | bool | str | None) -> "GBMRegressor":
        """Set estimator parameters with constructor-equivalent validation."""
        allowed = {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "row_subsample",
            "col_subsample",
            "early_stopping_rounds",
            "min_validation_improvement",
            "min_data_in_leaf",
            "lambda_l1",
            "lambda_l2",
            "min_child_hessian",
            "min_split_gain",
            "seed",
            "deterministic",
            "continuous_binning_strategy",
            "continuous_binning_max_bins",
            "categorical_feature_index",
            "categorical_feature_indices",
            "training_policy",
            "store_node_stats",
            "categorical_smoothing",
            "categorical_min_samples_leaf",
            "categorical_time_aware",
            "monotone_constraints",
            "feature_weights",
            "max_leaves",
            "tree_growth",
            "warm_start",
            "objective",
            "max_cat_threshold",
            "device",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")

        if "learning_rate" in params:
            learning_rate = float(params["learning_rate"])
            if not (0.0 < learning_rate <= 1.0):
                raise ValueError("learning_rate must be in (0.0, 1.0]")
            self.learning_rate = learning_rate

        if "max_depth" in params:
            max_depth = int(params["max_depth"])
            if max_depth <= 0:
                raise ValueError("max_depth must be greater than 0")
            self.max_depth = max_depth

        if "n_estimators" in params:
            n_estimators = int(params["n_estimators"])
            if n_estimators <= 0:
                raise ValueError("n_estimators must be greater than 0")
            self.n_estimators = n_estimators

        if "row_subsample" in params:
            row_subsample = float(params["row_subsample"])
            if not (0.0 < row_subsample <= 1.0):
                raise ValueError("row_subsample must be in (0.0, 1.0]")
            self.row_subsample = row_subsample

        if "col_subsample" in params:
            col_subsample = float(params["col_subsample"])
            if not (0.0 < col_subsample <= 1.0):
                raise ValueError("col_subsample must be in (0.0, 1.0]")
            self.col_subsample = col_subsample

        if "early_stopping_rounds" in params:
            if params["early_stopping_rounds"] is None:
                self.early_stopping_rounds = None
            else:
                early_stopping_rounds = int(params["early_stopping_rounds"])
                if early_stopping_rounds <= 0:
                    raise ValueError(
                        "early_stopping_rounds must be greater than 0 when set"
                    )
                self.early_stopping_rounds = early_stopping_rounds

        if "min_validation_improvement" in params:
            min_validation_improvement = float(params["min_validation_improvement"])
            if min_validation_improvement < 0.0:
                raise ValueError("min_validation_improvement must be >= 0")
            self.min_validation_improvement = min_validation_improvement

        if "min_data_in_leaf" in params:
            min_data_in_leaf = int(params["min_data_in_leaf"])
            if min_data_in_leaf <= 0:
                raise ValueError("min_data_in_leaf must be greater than 0")
            self.min_data_in_leaf = min_data_in_leaf

        if "lambda_l1" in params:
            lambda_l1 = float(params["lambda_l1"])
            if lambda_l1 < 0.0:
                raise ValueError("lambda_l1 must be >= 0")
            self.lambda_l1 = lambda_l1

        if "lambda_l2" in params:
            lambda_l2 = float(params["lambda_l2"])
            if lambda_l2 < 0.0:
                raise ValueError("lambda_l2 must be >= 0")
            self.lambda_l2 = lambda_l2

        if "min_child_hessian" in params:
            min_child_hessian = float(params["min_child_hessian"])
            if min_child_hessian < 0.0:
                raise ValueError("min_child_hessian must be >= 0")
            self.min_child_hessian = min_child_hessian

        if "min_split_gain" in params:
            min_split_gain = float(params["min_split_gain"])
            if not math.isfinite(min_split_gain) or min_split_gain < 0.0:
                raise ValueError("min_split_gain must be a finite value >= 0")
            self.min_split_gain = min_split_gain

        if "seed" in params:
            self.seed = int(params["seed"])

        if "deterministic" in params:
            self.deterministic = bool(params["deterministic"])

        if "continuous_binning_strategy" in params:
            strategy = str(params["continuous_binning_strategy"])
            if strategy not in _VALID_CONTINUOUS_BINNING_STRATEGIES:
                raise ValueError(
                    "continuous_binning_strategy must be one of: "
                    + ", ".join(sorted(_VALID_CONTINUOUS_BINNING_STRATEGIES))
                )
            if strategy != self.continuous_binning_strategy and self._is_fitted:
                self._reset_fitted_state()
            self.continuous_binning_strategy = strategy

        if "continuous_binning_max_bins" in params:
            max_bins = int(params["continuous_binning_max_bins"])
            if not (
                _MIN_CONTINUOUS_QUANTIZED_BINS
                <= max_bins
                <= (_MAX_CONTINUOUS_QUANTIZED_BIN + 1)
            ):
                raise ValueError(
                    "continuous_binning_max_bins must be in "
                    f"[{_MIN_CONTINUOUS_QUANTIZED_BINS}, {_MAX_CONTINUOUS_QUANTIZED_BIN + 1}]"
                )
            if max_bins != self.continuous_binning_max_bins and self._is_fitted:
                self._reset_fitted_state()
            self.continuous_binning_max_bins = max_bins

        if "categorical_feature_index" in params:
            if params["categorical_feature_index"] is None:
                self.categorical_feature_index = None
            else:
                categorical_feature_index = int(params["categorical_feature_index"])
                if categorical_feature_index < 0:
                    raise ValueError("categorical_feature_index must be >= 0 when set")
                self.categorical_feature_index = categorical_feature_index

        if "categorical_feature_indices" in params:
            if params["categorical_feature_indices"] is None:
                self.categorical_feature_indices = None
            else:
                indices = params["categorical_feature_indices"]
                if not isinstance(indices, (list, tuple)):
                    raise TypeError("categorical_feature_indices must be a list of ints")
                int_indices = [int(i) for i in indices]
                for idx in int_indices:
                    if idx < 0:
                        raise ValueError(
                            "all values in categorical_feature_indices must be >= 0"
                        )
                if len(set(int_indices)) != len(int_indices):
                    raise ValueError(
                        "categorical_feature_indices must not contain duplicates"
                    )
                self.categorical_feature_indices = int_indices

        # Mutual exclusion check after both may have been set
        if self.categorical_feature_index is not None and self.categorical_feature_indices is not None:
            raise ValueError(
                "categorical_feature_index and categorical_feature_indices are mutually exclusive; use categorical_feature_indices for multiple columns"
            )

        if "training_policy" in params:
            training_policy = str(params["training_policy"])
            if training_policy not in {"auto", "manual"}:
                raise ValueError("training_policy must be 'auto' or 'manual'")
            self.training_policy = training_policy

        if "store_node_stats" in params:
            self.store_node_stats = bool(params["store_node_stats"])

        if "categorical_smoothing" in params:
            categorical_smoothing = float(params["categorical_smoothing"])
            if categorical_smoothing < 0.0:
                raise ValueError("categorical_smoothing must be >= 0")
            self.categorical_smoothing = categorical_smoothing

        if "categorical_min_samples_leaf" in params:
            categorical_min_samples_leaf = int(params["categorical_min_samples_leaf"])
            if categorical_min_samples_leaf <= 0:
                raise ValueError("categorical_min_samples_leaf must be greater than 0")
            self.categorical_min_samples_leaf = categorical_min_samples_leaf

        if "categorical_time_aware" in params:
            self.categorical_time_aware = bool(params["categorical_time_aware"])

        if "monotone_constraints" in params:
            mc = params["monotone_constraints"]
            if mc is not None:
                vals = mc.values() if isinstance(mc, dict) else mc
                for v in vals:
                    if int(v) not in (-1, 0, 1):
                        raise ValueError(
                            "monotone_constraints values must be -1, 0, or +1"
                        )
            self.monotone_constraints = mc

        if "feature_weights" in params:
            fw = params["feature_weights"]
            if fw is not None:
                vals = fw.values() if isinstance(fw, dict) else fw
                for w in vals:
                    if not math.isfinite(float(w)) or float(w) < 0.0:
                        raise ValueError(
                            "feature_weights values must be finite and >= 0"
                        )
            self.feature_weights = fw

        if "max_leaves" in params:
            if params["max_leaves"] is None:
                self.max_leaves = None
            else:
                ml = int(params["max_leaves"])
                if ml < 2:
                    raise ValueError("max_leaves must be >= 2 when set")
                self.max_leaves = ml

        if "tree_growth" in params:
            tg = str(params["tree_growth"])
            if tg not in {"level", "leaf"}:
                raise ValueError("tree_growth must be 'level' or 'leaf'")
            self.tree_growth = tg

        if "warm_start" in params:
            self.warm_start = bool(params["warm_start"])

        if "objective" in params:
            obj = params["objective"]
            if obj is not None and not callable(obj) and not isinstance(obj, str):
                raise TypeError("objective must be a string, a callable, or None")
            self.objective = obj

        if "max_cat_threshold" in params:
            mct = int(params["max_cat_threshold"])
            if mct < 0:
                raise ValueError("max_cat_threshold must be >= 0")
            self.max_cat_threshold = mct

        if "device" in params:
            dev = str(params["device"])
            if dev not in _VALID_DEVICES:
                raise ValueError(
                    "device must be one of: " + ", ".join(sorted(_VALID_DEVICES))
                )
            self.device = dev

        # Cross-field validation: leaf growth requires max_leaves
        if self.tree_growth == "leaf" and self.max_leaves is None:
            raise ValueError("max_leaves must be set when tree_growth='leaf'")

        return self

    def _resolve_monotone_constraints(self, feature_count: int) -> list[int]:
        """Resolve monotone_constraints to a dense list[int] for the bridge."""
        if self.monotone_constraints is None:
            return []
        if isinstance(self.monotone_constraints, dict):
            dense = [0] * feature_count
            for idx, val in self.monotone_constraints.items():
                if 0 <= int(idx) < feature_count:
                    dense[int(idx)] = int(val)
            return dense
        return [int(v) for v in self.monotone_constraints]

    def _resolve_feature_weights(self, feature_count: int) -> list[float]:
        """Resolve feature_weights to a dense list[float] for the bridge."""
        if self.feature_weights is None:
            return []
        if isinstance(self.feature_weights, dict):
            dense = [1.0] * feature_count
            for idx, val in self.feature_weights.items():
                if 0 <= int(idx) < feature_count:
                    dense[int(idx)] = float(val)
            return dense
        return [float(w) for w in self.feature_weights]

    def _objective_name(self) -> str:
        """Return the objective function name passed to the native training bridge."""
        if self.objective is not None:
            if callable(self.objective):
                return "custom"
            return str(self.objective)
        return "squared_error"

    @staticmethod
    def _loss_metric_name_for(objective: str) -> str:
        """Map an objective name to its natural loss metric name."""
        if objective == "binary_crossentropy":
            return "logloss"
        if objective in ("rank_pairwise", "rank_ndcg", "rank_xendcg", "yetirank"):
            return "ndcg"
        if objective == "queryrmse":
            return "queryrmse"
        if objective == "custom":
            return "loss"
        return "mse"

    def _loss_metric_name(self) -> str:
        """Return the natural loss metric name for this estimator's objective."""
        return self._loss_metric_name_for(self._objective_name())

    @staticmethod
    def _build_evals_result(summary: object) -> dict:
        """Build ``evals_result_`` from a ``NativeTrainingSummary``.

        The result always contains a backward-compatible ``"rmse"`` key plus
        the objective-native loss metric (``"mse"`` for squared_error,
        ``"logloss"`` for binary_crossentropy).  When a custom eval metric
        was used during training, its per-round values are included under
        the ``"validation"`` key with the metric's own name.
        """
        loss_metric = GBMRegressor._loss_metric_name_for(summary.objective)
        result: dict[str, dict[str, list[float]]] = {
            "train": {
                "rmse": [float(v) for v in summary.train_rmse],
                loss_metric: [float(v) for v in summary.train_loss],
            }
        }
        if summary.validation_rmse:
            result["validation"] = {
                "rmse": [float(v) for v in summary.validation_rmse],
                loss_metric: [float(v) for v in summary.validation_loss],
            }
        # Include custom metric values when present.
        custom_metric_values = getattr(summary, "custom_metric_values", None)
        custom_metric_name = getattr(summary, "custom_metric_name", None)
        if custom_metric_values and custom_metric_name:
            if "validation" not in result:
                result["validation"] = {}
            result["validation"][custom_metric_name] = [
                float(v) for v in custom_metric_values
            ]
        return result

    def fit(
        self,
        X: object,
        y: object,
        *,
        sample_weight: object | None = None,
        eval_set: tuple[object, object] | None = None,
        eval_sample_weight: object | None = None,
        group: object | None = None,
        eval_group: object | None = None,
        eval_time_index: object | None = None,
        categorical_feature_values: object | None = None,
        categorical_feature_values_list: object | None = None,
        time_index: object | None = None,
        init_model: "GBMRegressor | None" = None,
        eval_metric: object | None = None,
    ) -> "GBMRegressor":
        """Fit native-backed regression model artifact state.

        Parameters
        ----------
        init_model : GBMRegressor or None, optional
            A previously fitted model to continue training from (warm-start).
            When provided, training resumes from this model's trees and
            ``n_estimators`` additional rounds are trained. If ``warm_start=True``
            is set on the estimator, the model's own previous state is used
            instead. ``init_model`` takes priority over ``warm_start``.
        eval_metric : callable or None, optional
            Custom evaluation metric callable. The function signature must be
            ``(y_true: np.ndarray, y_pred: np.ndarray) -> (name, value, higher_is_better)``
            where *name* is a string, *value* is a float, and *higher_is_better*
            is a bool.  When provided together with ``eval_set``, the metric is
            evaluated after each boosting round and can drive early stopping
            (instead of the built-in loss) when ``early_stopping_rounds`` is set.
        """
        fit_start = time.perf_counter()
        self._fit_start_time = fit_start
        targets = self._validate_targets(y)
        if self.early_stopping_rounds is not None and eval_set is None:
            raise ValueError("early_stopping_rounds requires eval_set to be provided")
        if eval_time_index is not None and eval_set is None:
            raise ValueError("eval_time_index requires eval_set to be provided")
        if eval_metric is not None and not callable(eval_metric):
            raise TypeError("eval_metric must be a callable or None")
        if eval_metric is not None and eval_set is None:
            raise ValueError("eval_metric requires eval_set to be provided")
        if categorical_feature_values is not None and categorical_feature_values_list is not None:
            raise ValueError(
                "categorical_feature_values and categorical_feature_values_list are "
                "mutually exclusive"
            )

        # ── Resolve warm-start artifact bytes ──────────────────────────
        init_artifact_bytes: bytes | None = None
        if init_model is not None:
            if not hasattr(init_model, "_artifact_bytes") or init_model._artifact_bytes is None:
                raise ValueError("init_model must be a fitted GBMRegressor with artifact bytes")
            if hasattr(init_model, "_objective_name"):
                init_objective = init_model._objective_name()
                current_objective = self._objective_name()
                if init_objective != current_objective:
                    raise ValueError(
                        f"init_model objective '{init_objective}' does not match "
                        f"current objective '{current_objective}'"
                    )
            init_artifact_bytes = init_model._artifact_bytes
        elif self.warm_start and self._is_fitted and self._artifact_bytes is not None:
            init_artifact_bytes = self._artifact_bytes

        # ── Normalize categorical configuration to plural form ──────────
        # effective_categorical_indices: list of column indices (or None if no categoricals)
        # categorical_values_list: list of per-column string values (or None)
        effective_categorical_indices: list[int] | None = None
        categorical_values_list: list[list[str]] | None = None

        if self.categorical_feature_indices is not None:
            effective_categorical_indices = list(self.categorical_feature_indices)
        elif self.categorical_feature_index is not None:
            effective_categorical_indices = [self.categorical_feature_index]

        if categorical_feature_values_list is not None:
            categorical_values_list = self._validate_categorical_values_list(
                categorical_feature_values_list, len(targets)
            )
        elif categorical_feature_values is not None:
            categorical_values_list = [
                self._validate_categorical_values(
                    categorical_feature_values, len(targets)
                )
            ]
        elif effective_categorical_indices is None:
            # Auto-infer from DataFrame dtypes
            inferred = self._infer_explicit_categorical_features(X)
            if inferred is not None:
                effective_categorical_indices, categorical_values_list = inferred

        has_categorical = (
            effective_categorical_indices is not None
            and len(effective_categorical_indices) > 0
        )

        # Backward-compat aliases used by some downstream code paths
        effective_categorical_feature_index: int | None = (
            effective_categorical_indices[0]
            if has_categorical and len(effective_categorical_indices) == 1
            else None
        )
        categorical_values: list[str] | None = (
            categorical_values_list[0]
            if categorical_values_list is not None and len(categorical_values_list) == 1
            else None
        )

        # Prefer bytes payload (zero-copy numpy→Rust) over list[float] payload
        dense_training_bytes_payload = (
            self._native_matrix_bytes_payload(X)
            if not has_categorical
            else None
        )
        dense_training_payload = (
            self._native_matrix_flat_payload(X)
            if not has_categorical and dense_training_bytes_payload is None
            else None
        )
        training_rows: list[list[float]] | None = None
        if dense_training_bytes_payload is not None:
            _, row_count, feature_count = dense_training_bytes_payload
        elif dense_training_payload is not None:
            _, row_count, feature_count = dense_training_payload
        else:
            training_rows = self._validate_rows(
                X, categorical_feature_indices=effective_categorical_indices
            )
            row_count = len(training_rows)
            feature_count = len(training_rows[0])
        if row_count != len(targets):
            raise ValueError("X and y must contain the same number of rows")

        if init_model is not None and hasattr(init_model, "_n_features_in"):
            if (
                init_model._n_features_in is not None
                and init_model._n_features_in != feature_count
            ):
                raise ValueError(
                    f"init_model was fitted with {init_model._n_features_in} features, "
                    f"but X has {feature_count} features"
                )

        # Validate sample_weight if provided.
        validated_sample_weights: list[float] | None = None
        if sample_weight is not None:
            validated_sample_weights = self._validate_sample_weight(sample_weight, row_count)
            obj = self._objective_name()
            if obj in ("rank_pairwise", "rank_ndcg", "rank_xendcg", "yetirank"):
                import warnings

                warnings.warn(
                    f"sample_weight is ignored by ranking objective '{obj}'",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate group if provided.
        validated_group_id: list[int] | None = None
        if group is not None:
            validated_group_id = self._validate_group(group, row_count)

        if eval_sample_weight is not None and eval_set is None:
            raise ValueError("eval_sample_weight requires eval_set to be provided")
        if eval_group is not None and eval_set is None:
            raise ValueError("eval_group requires eval_set to be provided")

        # Capture feature names from DataFrame columns if available.
        columns = getattr(X, "columns", None)
        if columns is not None:
            names = [str(c) for c in columns]
            if len(names) == feature_count:
                self.feature_names_in_ = names
            else:
                self.feature_names_in_ = None
        else:
            self.feature_names_in_ = None

        if has_categorical:
            assert effective_categorical_indices is not None
            for idx in effective_categorical_indices:
                if idx >= feature_count:
                    raise ValueError(
                        f"categorical feature index {idx} must be within fitted feature bounds (feature_count={feature_count})"
                    )

        if not has_categorical and categorical_values_list is not None:
            raise ValueError(
                "categorical_feature_values requires categorical_feature_index or categorical_feature_indices to be set"
            )
        if has_categorical and categorical_values_list is None:
            raise ValueError(
                "categorical_feature_values must be provided when categorical feature indices are set"
            )
        if (
            has_categorical
            and categorical_values_list is not None
            and len(categorical_values_list) != len(effective_categorical_indices)  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"categorical_feature_values_list length ({len(categorical_values_list)}) "
                f"does not match categorical_feature_indices length ({len(effective_categorical_indices)})"  # type: ignore[arg-type]
            )

        validated_time_index = (
            self._validate_time_index(time_index, row_count) if time_index is not None else None
        )

        validation_X: object | None = None
        validation_targets: list[float] | None = None
        validation_dense_payload: tuple[list[float], int, int] | None = None
        validation_dense_bytes_payload: tuple[bytes, int, int] | None = None
        validation_rows: list[list[float]] | None = None
        validation_categorical_values: list[str] | None = None
        validation_categorical_values_list: list[list[str]] | None = None
        validated_eval_time_index: list[int] | None = None
        validated_eval_sample_weights: list[float] | None = None
        validated_eval_group_id: list[int] | None = None
        if eval_set is not None:
            validation_X, validation_y = eval_set
            validation_targets = self._validate_targets(validation_y)
            validation_dense_bytes_payload = (
                self._native_matrix_bytes_payload(validation_X)
                if not has_categorical
                else None
            )
            validation_dense_payload = (
                self._native_matrix_flat_payload(validation_X)
                if not has_categorical and validation_dense_bytes_payload is None
                else None
            )
            if validation_dense_bytes_payload is not None:
                _, validation_row_count, validation_feature_count = validation_dense_bytes_payload
            elif validation_dense_payload is not None:
                _, validation_row_count, validation_feature_count = validation_dense_payload
            else:
                validation_rows = self._validate_rows(
                    validation_X,
                    categorical_feature_indices=effective_categorical_indices,
                )
                validation_row_count = len(validation_rows)
                validation_feature_count = len(validation_rows[0])
            if validation_row_count != len(validation_targets):
                raise ValueError("eval_set X and y must contain the same number of rows")
            if validation_feature_count != feature_count:
                raise ValueError(
                    "eval_set feature count must match training feature count"
                )
            if eval_time_index is not None:
                validated_eval_time_index = self._validate_time_index(
                    eval_time_index, validation_row_count
                )
            if has_categorical:
                validation_categorical_values_list = (
                    self._extract_categorical_values_for_indices(
                        validation_X,
                        effective_categorical_indices,  # type: ignore[arg-type]
                        validation_row_count,
                    )
                )
                # Keep backward-compat alias for singular case
                if len(effective_categorical_indices) == 1:  # type: ignore[arg-type]
                    validation_categorical_values = validation_categorical_values_list[0]
            if eval_sample_weight is not None:
                validated_eval_sample_weights = self._validate_sample_weight(
                    eval_sample_weight, validation_row_count
                )
            if eval_group is not None:
                validated_eval_group_id = self._validate_group(
                    eval_group, validation_row_count
                )

        if (
            has_categorical
            and self.categorical_time_aware
            and validated_time_index is None
        ):
            raise ValueError(
                "time_index must be provided when categorical_time_aware=True and categorical features are set"
            )
        if (
            has_categorical
            and self.categorical_time_aware
            and eval_set is not None
            and validated_eval_time_index is None
        ):
            raise ValueError(
                "eval_time_index must be provided when categorical_time_aware=True and eval_set is used"
            )

        import numpy as np
        targets_bytes = np.asarray(targets, dtype=np.float32).tobytes() if dense_training_bytes_payload is not None else None
        validation_targets_bytes = (
            np.asarray(validation_targets, dtype=np.float32).tobytes()
            if validation_targets is not None and dense_training_bytes_payload is not None
            else None
        )

        input_adaptation_seconds = time.perf_counter() - fit_start

        # Resolve custom objective / metric callables for the native bridge.
        _custom_objective_fn = self.objective if callable(self.objective) else None
        _custom_loss_fn = None  # reserved for future extension
        _custom_metric_fn = eval_metric if callable(eval_metric) else None

        # Try bytes path first (avoids Python list→Vec<f32> conversion overhead)
        if dense_training_bytes_payload is not None:
            try:
                from alloygbm._alloygbm import train_regression_artifact_dense_with_summary_bytes
                native_result = train_regression_artifact_dense_with_summary_bytes(
                    values_bytes=dense_training_bytes_payload[0],
                    row_count=dense_training_bytes_payload[1],
                    feature_count=dense_training_bytes_payload[2],
                    targets_bytes=targets_bytes,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    row_subsample=self.row_subsample,
                    col_subsample=self.col_subsample,
                    min_validation_improvement=self.min_validation_improvement,
                    seed=self.seed,
                    deterministic=self.deterministic,
                    rounds=self.n_estimators,
                    early_stopping_rounds=self.early_stopping_rounds,
                    min_data_in_leaf=self.min_data_in_leaf,
                    lambda_l1=self.lambda_l1,
                    lambda_l2=self.lambda_l2,
                    min_child_hessian=self.min_child_hessian,
                    sample_weights=validated_sample_weights,
                    group_id=validated_group_id,
                    min_split_gain=self.min_split_gain,
                    validation_values_bytes=(
                        validation_dense_bytes_payload[0]
                        if validation_dense_bytes_payload is not None
                        else None
                    ),
                    validation_row_count=(
                        validation_dense_bytes_payload[1]
                        if validation_dense_bytes_payload is not None
                        else None
                    ),
                    validation_targets_bytes=validation_targets_bytes,
                    validation_sample_weights=validated_eval_sample_weights,
                    validation_group_id=validated_eval_group_id,
                    validation_time_index=validated_eval_time_index,
                    categorical_feature_index=effective_categorical_feature_index,
                    categorical_feature_values=categorical_values,
                    validation_categorical_feature_values=validation_categorical_values,
                    training_policy=self.training_policy,
                    store_node_stats=self.store_node_stats,
                    categorical_smoothing=self.categorical_smoothing,
                    categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                    categorical_time_aware=self.categorical_time_aware,
                    time_index=validated_time_index,
                    continuous_binning_strategy=self.continuous_binning_strategy,
                    continuous_binning_max_bins=self.continuous_binning_max_bins,
                    objective=self._objective_name(),
                    monotone_constraints=self._resolve_monotone_constraints(feature_count),
                    feature_weights=self._resolve_feature_weights(feature_count),
                    max_leaves=self.max_leaves,
                    tree_growth=self.tree_growth,
                    categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                    categorical_feature_values_list=categorical_values_list if has_categorical else None,
                    validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                    init_artifact_bytes=init_artifact_bytes,
                    num_classes=getattr(self, '_num_classes_for_training', None),
                    custom_objective_fn=_custom_objective_fn,
                    custom_loss_fn=_custom_loss_fn,
                    custom_metric_fn=_custom_metric_fn,
                    max_cat_threshold=self.max_cat_threshold,
                    device=self.device,
                )
                return self._finalize_training_result(native_result, input_adaptation_seconds, feature_count=feature_count)
            except (ImportError, AttributeError):
                pass  # Fall through to list-based path
            except Exception:
                pass  # Bytes path not available or failed; fall through

        # Compute dense_training_payload lazily if only bytes payload was prepared
        if dense_training_payload is None and dense_training_bytes_payload is not None:
            dense_training_payload = self._native_matrix_flat_payload(X)

        try:
            if dense_training_payload is not None:
                train_with_summary = _load_native_train_regression_artifact_dense_with_summary()
            else:
                train_with_summary = _load_native_train_regression_artifact_with_summary()
        except RuntimeError:
            return self._fit_with_legacy_native_bridge(
                X=X,
                targets=targets,
                dense_training_payload=dense_training_payload,
                training_rows=training_rows if dense_training_payload is None else None,
                feature_count=feature_count,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_values=categorical_values,
                time_index=validated_time_index,
                eval_set=eval_set,
                input_adaptation_seconds=input_adaptation_seconds,
            )

        if dense_training_payload is not None:
            native_result = train_with_summary(
                values=dense_training_payload[0],
                row_count=dense_training_payload[1],
                feature_count=dense_training_payload[2],
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                min_child_hessian=self.min_child_hessian,
                sample_weights=validated_sample_weights,
                group_id=validated_group_id,
                min_split_gain=self.min_split_gain,
                validation_values=(
                    validation_dense_payload[0]
                    if validation_dense_payload is not None
                    else None
                ),
                validation_row_count=(
                    validation_dense_payload[1]
                    if validation_dense_payload is not None
                    else None
                ),
                validation_targets=validation_targets,
                validation_sample_weights=validated_eval_sample_weights,
                validation_group_id=validated_eval_group_id,
                validation_time_index=validated_eval_time_index,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_feature_values=categorical_values,
                validation_categorical_feature_values=validation_categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=validated_time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                monotone_constraints=self._resolve_monotone_constraints(feature_count),
                feature_weights=self._resolve_feature_weights(feature_count),
                max_leaves=self.max_leaves,
                tree_growth=self.tree_growth,
                categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                categorical_feature_values_list=categorical_values_list if has_categorical else None,
                validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                init_artifact_bytes=init_artifact_bytes,
                num_classes=getattr(self, '_num_classes_for_training', None),
                custom_objective_fn=_custom_objective_fn,
                custom_loss_fn=_custom_loss_fn,
                custom_metric_fn=_custom_metric_fn,
                max_cat_threshold=self.max_cat_threshold,
                device=self.device,
            )
        else:
            assert training_rows is not None
            native_result = train_with_summary(
                rows=training_rows,
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                min_child_hessian=self.min_child_hessian,
                sample_weights=validated_sample_weights,
                group_id=validated_group_id,
                min_split_gain=self.min_split_gain,
                validation_rows=validation_rows,
                validation_targets=validation_targets,
                validation_sample_weights=validated_eval_sample_weights,
                validation_group_id=validated_eval_group_id,
                validation_time_index=validated_eval_time_index,
                categorical_feature_index=effective_categorical_feature_index,
                categorical_feature_values=categorical_values,
                validation_categorical_feature_values=validation_categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=validated_time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                monotone_constraints=self._resolve_monotone_constraints(feature_count),
                feature_weights=self._resolve_feature_weights(feature_count),
                max_leaves=self.max_leaves,
                tree_growth=self.tree_growth,
                categorical_feature_indices=effective_categorical_indices if has_categorical else None,
                categorical_feature_values_list=categorical_values_list if has_categorical else None,
                validation_categorical_feature_values_list=validation_categorical_values_list if has_categorical else None,
                init_artifact_bytes=init_artifact_bytes,
                num_classes=getattr(self, '_num_classes_for_training', None),
                custom_objective_fn=_custom_objective_fn,
                custom_loss_fn=_custom_loss_fn,
                custom_metric_fn=_custom_metric_fn,
                max_cat_threshold=self.max_cat_threshold,
                device=self.device,
            )

        self._apply_continuous_binning_metadata(native_result.continuous_binning_metadata)
        self._n_features_in = feature_count
        self._artifact_bytes = bytes(native_result.artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        raw_mappings = getattr(native_result, "native_cat_mappings", None)
        if raw_mappings:
            self._native_cat_mappings_ = {
                int(k): {str(ck): int(cv) for ck, cv in v.items()}
                for k, v in raw_mappings.items()
            }
        else:
            self._native_cat_mappings_ = None
        summary = native_result.summary
        self.best_iteration_ = summary.best_validation_round
        self.best_score_ = (
            float(summary.best_validation_loss)
            if summary.best_validation_loss is not None
            else None
        )
        self.n_estimators_ = int(summary.rounds_completed)
        self.rounds_completed_ = int(summary.rounds_completed)
        self.stop_reason_ = (
            str(summary.stop_reason) if summary.stop_reason is not None else None
        )
        self.evals_result_ = self._build_evals_result(summary)
        total_fit_seconds = time.perf_counter() - fit_start
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": float(summary.bridge_prepare_seconds),
            "native_train_seconds": float(summary.native_train_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._is_fitted = True
        return self

    def _finalize_training_result(
        self,
        native_result: object,
        input_adaptation_seconds: float,
        feature_count: int | None = None,
    ) -> "GBMRegressor":
        self._apply_continuous_binning_metadata(native_result.continuous_binning_metadata)
        if feature_count is not None:
            self._n_features_in = feature_count
        self._artifact_bytes = bytes(native_result.artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        raw_mappings = getattr(native_result, "native_cat_mappings", None)
        if raw_mappings:
            self._native_cat_mappings_ = {
                int(k): {str(ck): int(cv) for ck, cv in v.items()}
                for k, v in raw_mappings.items()
            }
        else:
            self._native_cat_mappings_ = None
        summary = native_result.summary
        self.best_iteration_ = summary.best_validation_round
        self.best_score_ = (
            float(summary.best_validation_loss)
            if summary.best_validation_loss is not None
            else None
        )
        self.n_estimators_ = int(summary.rounds_completed)
        self.rounds_completed_ = int(summary.rounds_completed)
        self.stop_reason_ = (
            str(summary.stop_reason) if summary.stop_reason is not None else None
        )
        self.evals_result_ = self._build_evals_result(summary)
        total_fit_seconds = time.perf_counter() - self._fit_start_time
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": float(summary.bridge_prepare_seconds),
            "native_train_seconds": float(summary.native_train_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._is_fitted = True
        return self

    def _fit_with_legacy_native_bridge(
        self,
        *,
        X: object,
        targets: list[float],
        dense_training_payload: tuple[list[float], int, int] | None,
        training_rows: list[list[float]] | None,
        feature_count: int,
        categorical_feature_index: int | None,
        categorical_values: list[str] | None,
        time_index: list[int] | None,
        eval_set: tuple[object, object] | None,
        input_adaptation_seconds: float,
    ) -> "GBMRegressor":
        if eval_set is not None:
            raise RuntimeError(
                "eval_set requires a native alloygbm build with training summary support"
            )
        if (
            self.min_data_in_leaf != 1
            or self.lambda_l1 != 0.0
            or self.lambda_l2 != 0.0
            or self.min_child_hessian != 0.0
        ):
            raise RuntimeError(
                "explicit regularization and min_data_in_leaf require a native alloygbm build with training summary support"
            )

        fit_start = time.perf_counter()
        native_training_rows: object | None = None
        active_dense_training_payload = dense_training_payload
        rows = training_rows
        if active_dense_training_payload is None:
            assert rows is not None
            self._uses_continuous_binning = not self._rows_are_pre_binned(rows)
            if self._uses_continuous_binning:
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._derive_continuous_feature_bounds(rows)
                    self._continuous_feature_mins = mins
                    self._continuous_feature_maxs = maxs
                    if _linear_tail_rank_enabled_from_env():
                        core_span_ratio_threshold = (
                            _linear_tail_core_span_ratio_threshold_from_env()
                        )
                        rank_flags, sorted_values = (
                            self._derive_continuous_feature_tail_rank_plan(
                                rows, core_span_ratio_threshold
                            )
                        )
                        self._continuous_feature_linear_rank_flags = rank_flags
                        if any(rank_flags):
                            self._continuous_feature_sorted_values = sorted_values
                            native_training_rows = (
                                self._quantize_rows_linear_with_selective_rank(
                                    rows, mins, maxs, rank_flags, sorted_values,
                                    max_bins=self.continuous_binning_max_bins,
                                )
                            )
                        else:
                            self._continuous_feature_sorted_values = None
                            native_training_rows = self._quantize_rows_linear(
                                rows, mins, maxs,
                                max_bins=self.continuous_binning_max_bins,
                            )
                    else:
                        self._continuous_feature_sorted_values = None
                        self._continuous_feature_linear_rank_flags = None
                        native_training_rows = self._quantize_rows_linear(
                            rows, mins, maxs,
                            max_bins=self.continuous_binning_max_bins,
                        )
                    self._continuous_feature_quantile_cuts = None
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._derive_continuous_feature_sorted_values(rows)
                    self._continuous_feature_sorted_values = sorted_values
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_quantile_cuts = None
                    self._continuous_feature_linear_rank_flags = None
                    native_training_rows = self._quantize_rows_rank(
                        rows, sorted_values,
                        max_bins=self.continuous_binning_max_bins,
                    )
                else:
                    quantile_cuts = self._derive_continuous_feature_quantile_cuts(
                        rows, self.continuous_binning_max_bins
                    )
                    self._continuous_feature_quantile_cuts = quantile_cuts
                    self._continuous_feature_sorted_values = None
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_linear_rank_flags = None
                    native_training_rows = self._quantize_rows_quantile(
                        rows, quantile_cuts,
                        max_bins=self.continuous_binning_max_bins,
                    )
            else:
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                self._continuous_feature_sorted_values = None
                self._continuous_feature_quantile_cuts = None
                self._continuous_feature_linear_rank_flags = None
                native_training_rows = rows
        else:
            flat_values, row_count, dense_feature_count = active_dense_training_payload
            if self._check_pre_binned_integers(flat_values):
                self._uses_continuous_binning = False
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                self._continuous_feature_sorted_values = None
                self._continuous_feature_quantile_cuts = None
                self._continuous_feature_linear_rank_flags = None
            else:
                self._uses_continuous_binning = True
                rows = self._validate_rows(X)
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._derive_dense_feature_bounds(
                        flat_values, row_count, dense_feature_count
                    )
                    self._continuous_feature_mins = mins
                    self._continuous_feature_maxs = maxs
                    if _linear_tail_rank_enabled_from_env():
                        core_span_ratio_threshold = (
                            _linear_tail_core_span_ratio_threshold_from_env()
                        )
                        rank_flags, sorted_values = (
                            self._derive_continuous_feature_tail_rank_plan(
                                rows, core_span_ratio_threshold
                            )
                        )
                        self._continuous_feature_linear_rank_flags = rank_flags
                        if any(rank_flags):
                            self._continuous_feature_sorted_values = sorted_values
                            active_dense_training_payload = (
                                self._quantize_dense_values_linear_with_selective_rank(
                                    flat_values,
                                    row_count,
                                    dense_feature_count,
                                    mins,
                                    maxs,
                                    rank_flags,
                                    sorted_values,
                                    max_bins=self.continuous_binning_max_bins,
                                ),
                                row_count,
                                dense_feature_count,
                            )
                        else:
                            self._continuous_feature_sorted_values = None
                            active_dense_training_payload = (
                                self._quantize_dense_values_linear(
                                    flat_values, row_count, dense_feature_count, mins, maxs,
                                    max_bins=self.continuous_binning_max_bins,
                                ),
                                row_count,
                                dense_feature_count,
                            )
                    else:
                        self._continuous_feature_sorted_values = None
                        self._continuous_feature_linear_rank_flags = None
                        active_dense_training_payload = (
                            self._quantize_dense_values_linear(
                                flat_values, row_count, dense_feature_count, mins, maxs,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            dense_feature_count,
                        )
                    self._continuous_feature_quantile_cuts = None
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._derive_dense_sorted_feature_values(
                        flat_values, row_count, dense_feature_count
                    )
                    self._continuous_feature_sorted_values = sorted_values
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_quantile_cuts = None
                    self._continuous_feature_linear_rank_flags = None
                    active_dense_training_payload = (
                        self._quantize_dense_values_rank(
                            flat_values, row_count, dense_feature_count, sorted_values,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        dense_feature_count,
                    )
                else:
                    quantile_cuts = self._derive_dense_feature_quantile_cuts(
                        flat_values,
                        row_count,
                        dense_feature_count,
                        self.continuous_binning_max_bins,
                    )
                    self._continuous_feature_quantile_cuts = quantile_cuts
                    self._continuous_feature_sorted_values = None
                    self._continuous_feature_mins = None
                    self._continuous_feature_maxs = None
                    self._continuous_feature_linear_rank_flags = None
                    active_dense_training_payload = (
                        self._quantize_dense_values_quantile(
                            flat_values, row_count, dense_feature_count, quantile_cuts,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        dense_feature_count,
                    )

        if active_dense_training_payload is not None:
            train_regression_artifact_dense = _load_native_train_regression_artifact_dense()
            artifact_bytes = train_regression_artifact_dense(
                values=active_dense_training_payload[0],
                row_count=active_dense_training_payload[1],
                feature_count=active_dense_training_payload[2],
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                categorical_feature_index=categorical_feature_index,
                categorical_feature_values=categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                device=self.device,
            )
        else:
            train_regression_artifact = _load_native_train_regression_artifact()
            artifact_bytes = train_regression_artifact(
                rows=native_training_rows,
                targets=targets,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                row_subsample=self.row_subsample,
                col_subsample=self.col_subsample,
                min_validation_improvement=self.min_validation_improvement,
                seed=self.seed,
                deterministic=self.deterministic,
                rounds=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                categorical_feature_index=categorical_feature_index,
                categorical_feature_values=categorical_values,
                training_policy=self.training_policy,
                store_node_stats=self.store_node_stats,
                categorical_smoothing=self.categorical_smoothing,
                categorical_min_samples_leaf=self.categorical_min_samples_leaf,
                categorical_time_aware=self.categorical_time_aware,
                time_index=time_index,
                continuous_binning_strategy=self.continuous_binning_strategy,
                continuous_binning_max_bins=self.continuous_binning_max_bins,
                objective=self._objective_name(),
                device=self.device,
            )

        self._n_features_in = feature_count
        self._artifact_bytes = bytes(artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._convert_predictor_thresholds_to_float()
        self.best_iteration_ = None
        self.best_score_ = None
        self.n_estimators_ = self.n_estimators
        loss_metric = self._loss_metric_name()
        self.evals_result_ = {"train": {"rmse": [], loss_metric: []}}
        total_fit_seconds = time.perf_counter() - fit_start
        self.fit_timing_ = {
            "input_adaptation_seconds": float(input_adaptation_seconds),
            "native_bridge_prepare_seconds": 0.0,
            "native_train_seconds": float(total_fit_seconds),
            "total_fit_seconds": float(total_fit_seconds),
        }
        self._is_fitted = True
        return self

    def predict(self, X: object) -> list[float]:
        """Predict using the fitted native artifact."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before predict")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")
        # Lazily reconstruct the native predictor after pickle roundtrip.
        if self._native_predictor_handle is None and getattr(
            self, "_predictor_needs_rebuild", False
        ):
            self._predictor_needs_rebuild = False
            self._native_predictor_handle = self._build_native_predictor_handle(
                self._artifact_bytes
            )
            self._convert_predictor_thresholds_to_float()
        # Convert native categorical columns to integer IDs before prediction.
        X = self._apply_native_cat_mappings_for_predict(X)
        rows: object
        # Fast path: float thresholds + zero-copy numpy — no data copying
        if self._float_thresholds_converted:
            try:
                import numpy as _np

                candidate = self._native_matrix_fast_path_candidate(X)
                if candidate is not None:
                    arr = _np.ascontiguousarray(candidate, dtype=_np.float32)
                    if arr.shape[1] != self._n_features_in:
                        raise ValueError(
                            f"X feature count {arr.shape[1]} does not match fitted "
                            f"feature count {self._n_features_in}"
                        )
                    predict_numpy = getattr(
                        self._native_predictor_handle, "predict_numpy", None
                    )
                    if callable(predict_numpy):
                        return list(predict_numpy(arr))
            except ImportError:
                pass

        if self._uses_continuous_binning:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                flat_values, row_count, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                # Float threshold fallback (when bytes path unavailable)
                if self._float_thresholds_converted:
                    predict_dense = getattr(
                        self._native_predictor_handle, "predict_dense", None
                    )
                    if callable(predict_dense):
                        return list(predict_dense(flat_values, row_count, feature_count))

                # Use Rust-side fused quantize+predict when native handle is available
                if self.continuous_binning_strategy == "linear":
                    mins, maxs = self._require_continuous_feature_bounds()
                    rank_flags = self._continuous_feature_linear_rank_flags
                    if rank_flags is not None and any(rank_flags):
                        sorted_values = self._require_continuous_feature_sorted_values()
                        predict_fn = getattr(
                            self._native_predictor_handle,
                            "predict_dense_quantized_linear_rank",
                            None,
                        )
                        if callable(predict_fn):
                            return list(
                                predict_fn(
                                    flat_values,
                                    row_count,
                                    feature_count,
                                    list(mins),
                                    list(maxs),
                                    list(rank_flags),
                                    [list(sv) for sv in sorted_values],
                                    _max_data_bin_for_max_bins(
                                        self.continuous_binning_max_bins
                                    ),
                                )
                            )
                        # Fallback to Python quantization
                        dense_payload = (
                            self._quantize_dense_values_linear_with_selective_rank(
                                flat_values,
                                row_count,
                                feature_count,
                                mins,
                                maxs,
                                rank_flags,
                                sorted_values,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            feature_count,
                        )
                    else:
                        predict_fn = getattr(
                            self._native_predictor_handle,
                            "predict_dense_quantized_linear",
                            None,
                        )
                        if callable(predict_fn):
                            return list(
                                predict_fn(
                                    flat_values,
                                    row_count,
                                    feature_count,
                                    list(mins),
                                    list(maxs),
                                    _max_data_bin_for_max_bins(
                                        self.continuous_binning_max_bins
                                    ),
                                )
                            )
                        # Fallback to Python quantization
                        dense_payload = (
                            self._quantize_dense_values_linear(
                                flat_values,
                                row_count,
                                feature_count,
                                mins,
                                maxs,
                                max_bins=self.continuous_binning_max_bins,
                            ),
                            row_count,
                            feature_count,
                        )
                elif self.continuous_binning_strategy == "rank":
                    sorted_values = self._require_continuous_feature_sorted_values()
                    dense_payload = (
                        self._quantize_dense_values_rank(
                            flat_values,
                            row_count,
                            feature_count,
                            sorted_values,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        feature_count,
                    )
                else:
                    quantile_cuts = self._require_continuous_feature_quantile_cuts()
                    dense_payload = (
                        self._quantize_dense_values_quantile(
                            flat_values,
                            row_count,
                            feature_count,
                            quantile_cuts,
                            max_bins=self.continuous_binning_max_bins,
                        ),
                        row_count,
                        feature_count,
                    )
                flat_values, row_count, feature_count = dense_payload
                predict_dense = getattr(self._native_predictor_handle, "predict_dense", None)
                if callable(predict_dense):
                    return list(predict_dense(flat_values, row_count, feature_count))
                predictor_predict_batch_dense = _load_native_predictor_predict_batch_dense()
                return list(
                    predictor_predict_batch_dense(
                        self._artifact_bytes, flat_values, row_count, feature_count
                    )
                )
            if self._float_thresholds_converted:
                # Float thresholds: send raw (unquantized) rows directly
                validated_rows = self._validate_rows(X)
                if len(validated_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(validated_rows[0])} does not match fitted "
                        f"feature count {self._n_features_in}"
                    )
                rows = validated_rows
            else:
                quantized_rows = self._quantize_rows_for_prediction(self._validate_rows(X))
                if len(quantized_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(quantized_rows[0])} does not match fitted "
                        f"feature count {self._n_features_in}"
                    )
                rows = quantized_rows
        else:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                _, _, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = dense_payload
            else:
                validated_rows = self._validate_rows(X)
                if len(validated_rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(validated_rows[0])} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = validated_rows
        if self._native_predictor_handle is not None:
            if isinstance(rows, tuple):
                predict_dense = getattr(self._native_predictor_handle, "predict_dense", None)
                if callable(predict_dense):
                    try:
                        flat_values, row_count, feature_count = rows
                        return list(predict_dense(flat_values, row_count, feature_count))
                    except RuntimeError:
                        self._native_predictor_handle = None
            predict_batch = getattr(self._native_predictor_handle, "predict_batch", None)
            if callable(predict_batch) and not isinstance(rows, tuple):
                try:
                    return list(predict_batch(rows))
                except RuntimeError:
                    self._native_predictor_handle = None
        if isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            predictor_predict_batch_canonical_dense = (
                _load_native_predictor_predict_batch_canonical_dense()
            )
            return list(
                predictor_predict_batch_canonical_dense(
                    self._artifact_bytes,
                    flat_values,
                    row_count=row_count,
                    feature_count=feature_count,
                )
            )
        predictor_predict_batch_canonical = _load_native_predictor_predict_batch_canonical()
        return list(predictor_predict_batch_canonical(self._artifact_bytes, rows))

    def shap_values(
        self, X: object, *, include_expected_value: bool = False
    ) -> list[list[float]] | tuple[float, list[list[float]]]:
        """Return SHAP values for the provided rows using the fitted artifact."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before shap_values")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")

        if self._uses_continuous_binning:
            rows: object = self._quantize_rows_for_prediction(self._validate_rows(X))
            if len(rows[0]) != self._n_features_in:
                raise ValueError(
                    f"X feature count {len(rows[0])} does not match fitted feature count "
                    f"{self._n_features_in}"
                )
        else:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                _, _, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = dense_payload
            else:
                rows = self._validate_rows(X)
                if len(rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(rows[0])} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )

        if isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            shap_explain_rows_dense = _load_native_shap_explain_rows_dense()
            expected_value, values = shap_explain_rows_dense(
                self._artifact_bytes,
                flat_values,
                row_count=row_count,
                feature_count=feature_count,
            )
        else:
            shap_explain_rows = _load_native_shap_explain_rows()
            expected_value, values = shap_explain_rows(self._artifact_bytes, rows)
        shap_matrix = [list(row) for row in values]
        if include_expected_value:
            return float(expected_value), shap_matrix
        return shap_matrix

    def feature_importances(
        self, X: object, *, method: str = "shap"
    ) -> list[tuple[str, float]]:
        """Return feature importances for the provided rows."""
        if method != "shap":
            raise ValueError("unsupported feature importance method; expected 'shap'")
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before feature_importances")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")

        if self._uses_continuous_binning:
            rows: object = self._quantize_rows_for_prediction(self._validate_rows(X))
            if len(rows[0]) != self._n_features_in:
                raise ValueError(
                    f"X feature count {len(rows[0])} does not match fitted feature count "
                    f"{self._n_features_in}"
                )
        else:
            dense_payload = self._native_matrix_flat_payload(X)
            if dense_payload is not None:
                _, _, feature_count = dense_payload
                if feature_count != self._n_features_in:
                    raise ValueError(
                        f"X feature count {feature_count} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )
                rows = dense_payload
            else:
                rows = self._validate_rows(X)
                if len(rows[0]) != self._n_features_in:
                    raise ValueError(
                        f"X feature count {len(rows[0])} does not match fitted feature count "
                        f"{self._n_features_in}"
                    )

        if isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            shap_global_importance_dense = _load_native_shap_global_importance_dense()
            importance = shap_global_importance_dense(
                self._artifact_bytes,
                flat_values,
                row_count=row_count,
                feature_count=feature_count,
            )
        else:
            shap_global_importance = _load_native_shap_global_importance()
            importance = shap_global_importance(self._artifact_bytes, rows)
        result = [(str(name), float(value)) for name, value in importance]
        if self.feature_names_in_ is not None and len(result) == len(
            self.feature_names_in_
        ):
            result = [
                (self.feature_names_in_[i], score)
                for i, (_name, score) in enumerate(result)
            ]
        return result

    def score(self, X: object, y: object, sample_weight: object = None) -> float:
        """Return R² score for the given test data and labels."""
        from alloygbm.evaluation import r2_score

        predictions = self.predict(X)
        targets = self._validate_targets(y)
        return float(r2_score(targets, predictions))

    def __sklearn_tags__(self):
        if not hasattr(super(), "__sklearn_tags__"):
            return {
                "non_deterministic": not self.deterministic,
                "requires_y": True,
                "allow_nan": True,
                "X_types": ["2darray"],
            }
        tags = super().__sklearn_tags__()
        # sklearn >= 1.6 returns a Tags dataclass
        if hasattr(tags, "non_deterministic"):
            tags.non_deterministic = not self.deterministic
        if hasattr(tags, "input_tags") and hasattr(tags.input_tags, "allow_nan"):
            tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    @staticmethod
    def predict_from_artifact(
        artifact_bytes: bytes | bytearray | memoryview, X: object
    ) -> list[float]:
        """Run predictor-backed inference from serialized model artifact bytes."""
        if not isinstance(artifact_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("artifact_bytes must be bytes-like")
        dense_payload = GBMRegressor._native_matrix_flat_payload(X)
        if dense_payload is not None:
            flat_values, row_count, feature_count = dense_payload
            predictor_predict_batch_dense = _load_native_predictor_predict_batch_dense()
            return list(
                predictor_predict_batch_dense(
                    bytes(artifact_bytes),
                    flat_values,
                    row_count=row_count,
                    feature_count=feature_count,
                )
            )
        rows = GBMRegressor._validate_rows(X)
        predictor_predict_batch = _load_native_predictor_predict_batch()
        return list(predictor_predict_batch(bytes(artifact_bytes), rows))

    @staticmethod
    def _validate_rows(
        X: object,
        *,
        categorical_feature_index: int | None = None,
        categorical_feature_indices: list[int] | None = None,
    ) -> list[list[float]]:
        rows_like = GBMRegressor._coerce_sequence_like(X, "X")
        if not isinstance(rows_like, Sequence) or isinstance(rows_like, (str, bytes)):
            raise TypeError("X must be a sequence of feature rows")
        if len(rows_like) == 0:
            raise ValueError("X must contain at least one row")

        # Build a set of all categorical column indices for fast lookup
        cat_indices_set: set[int] = set()
        if categorical_feature_indices is not None:
            cat_indices_set.update(categorical_feature_indices)
        elif categorical_feature_index is not None:
            cat_indices_set.add(categorical_feature_index)

        normalized: list[list[float]] = []
        expected_width: int | None = None
        for row in rows_like:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("each X row must be a sequence of numeric values")
            if len(row) == 0:
                raise ValueError("each X row must contain at least one feature value")
            row_values: list[float] = []
            for feature_index, value in enumerate(row):
                if feature_index in cat_indices_set:
                    try:
                        row_values.append(float(value))
                    except (TypeError, ValueError):
                        row_values.append(0.0)
                else:
                    row_values.append(float(value))
            if expected_width is None:
                expected_width = len(row_values)
            elif len(row_values) != expected_width:
                raise ValueError("all X rows must have the same feature count")
            normalized.append(row_values)

        return normalized

    @staticmethod
    def _adapt_native_array_candidate(value: object) -> object | None:
        current = value
        for _ in range(2):
            try:
                view = memoryview(current)
            except TypeError:
                view = None
            if view is not None and getattr(view, "ndim", 0) == 2:
                return current
            if hasattr(current, "to_numpy"):
                next_value = current.to_numpy()  # type: ignore[call-arg]
                if next_value is current:
                    break
                current = next_value
                continue
            break
        return None

    @staticmethod
    def _native_matrix_shape(value: object) -> tuple[int, int]:
        try:
            view = memoryview(value)
        except TypeError as exc:
            raise TypeError("X is not a native dense matrix candidate") from exc
        shape = getattr(view, "shape", None)
        if view.ndim != 2 or shape is None or len(shape) != 2:
            raise TypeError("X is not a 2D native dense matrix candidate")
        row_count = int(shape[0])
        feature_count = int(shape[1])
        if row_count <= 0 or feature_count <= 0:
            raise ValueError("X must contain at least one row and one feature")
        return row_count, feature_count

    @staticmethod
    def _buffer_format_is_integer(value: object) -> bool:
        try:
            view = memoryview(value)
        except TypeError:
            return False
        format_code = getattr(view, "format", "") or ""
        normalized = str(format_code).lower()
        return normalized not in {"f", "d", "e"}

    @staticmethod
    def _native_matrix_fast_path_candidate(
        X: object, *, require_integer: bool = False
    ) -> object | None:
        candidate = GBMRegressor._adapt_native_array_candidate(X)
        if candidate is None:
            return None
        GBMRegressor._native_matrix_shape(candidate)
        if require_integer and not GBMRegressor._buffer_format_is_integer(candidate):
            return None
        return candidate

    @staticmethod
    def _native_matrix_flat_payload(
        X: object, *, require_integer: bool = False
    ) -> tuple[list[float], int, int] | None:
        candidate = GBMRegressor._native_matrix_fast_path_candidate(
            X, require_integer=require_integer
        )
        if candidate is None:
            return None
        row_count, feature_count = GBMRegressor._native_matrix_shape(candidate)
        return (
            GBMRegressor._flatten_native_matrix_candidate(candidate),
            row_count,
            feature_count,
        )

    @staticmethod
    def _native_matrix_bytes_payload(
        X: object,
    ) -> tuple[bytes, int, int] | None:
        """Return raw f32 bytes of the matrix for zero-copy transfer to Rust."""
        try:
            import numpy as np
            candidate = GBMRegressor._native_matrix_fast_path_candidate(X)
            if candidate is None:
                return None
            row_count, feature_count = GBMRegressor._native_matrix_shape(candidate)
            arr = np.ascontiguousarray(candidate, dtype=np.float32)
            return (arr.tobytes(), row_count, feature_count)
        except ImportError:
            return None

    @staticmethod
    def _flatten_native_matrix_candidate(candidate: object) -> list[float]:
        # Fast path: numpy arrays can use .astype(float32).ravel().tolist()
        # which is 10-100× faster than struct.iter_unpack for large arrays
        try:
            import numpy as np
            if isinstance(candidate, np.ndarray):
                flat = np.ascontiguousarray(candidate, dtype=np.float32).ravel()
                return flat.tolist()
        except ImportError:
            pass

        view = memoryview(candidate)
        format_code = getattr(view, "format", "") or ""
        normalized = str(format_code).strip()
        type_code = next(
            (character for character in reversed(normalized) if character.isalpha()),
            "",
        )
        if type_code == "":
            raise TypeError("native dense matrix format is not supported")
        if type_code == "?":
            unpack_code = "?"
        else:
            unpack_code = type_code
        if unpack_code not in {"b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d"}:
            raise TypeError(
                f"native dense matrix format '{normalized}' is not supported"
            )
        raw_bytes = view.tobytes()
        return [float(value[0]) for value in struct.iter_unpack("@" + unpack_code, raw_bytes)]

    @staticmethod
    def _column_values_from_flat_payload(
        flat_values: Sequence[float], row_count: int, feature_count: int, feature_index: int
    ) -> list[float]:
        return [
            float(flat_values[row_index * feature_count + feature_index])
            for row_index in range(row_count)
        ]

    @staticmethod
    def _derive_dense_feature_bounds(
        flat_values: Sequence[float], row_count: int, feature_count: int
    ) -> tuple[list[float], list[float]]:
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32).reshape(row_count, feature_count)
            return np.nanmin(arr, axis=0).tolist(), np.nanmax(arr, axis=0).tolist()
        except (ImportError, ValueError):
            pass
        mins = [float("inf")] * feature_count
        maxs = [float("-inf")] * feature_count
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    continue
                if value < mins[feature_index]:
                    mins[feature_index] = value
                if value > maxs[feature_index]:
                    maxs[feature_index] = value
        return mins, maxs

    @staticmethod
    def _derive_dense_sorted_feature_values(
        flat_values: Sequence[float], row_count: int, feature_count: int
    ) -> list[list[float]]:
        sorted_values: list[list[float]] = []
        for feature_index in range(feature_count):
            values = GBMRegressor._column_values_from_flat_payload(
                flat_values, row_count, feature_count, feature_index
            )
            values = [v for v in values if not math.isnan(v)]
            values.sort()
            sorted_values.append(values)
        return sorted_values

    @staticmethod
    def _derive_dense_feature_quantile_cuts(
        flat_values: Sequence[float], row_count: int, feature_count: int, max_bins: int
    ) -> list[list[float]]:
        feature_cuts: list[list[float]] = []
        for feature_index in range(feature_count):
            values = GBMRegressor._column_values_from_flat_payload(
                flat_values, row_count, feature_count, feature_index
            )
            values = [v for v in values if not math.isnan(v)]
            values.sort()
            if len(values) <= 1:
                feature_cuts.append([])
                continue

            bin_count = min(max_bins, len(values))
            cuts: list[float] = []
            for quantile_index in range(1, bin_count):
                rank = (quantile_index * len(values)) // bin_count
                if rank >= len(values):
                    rank = len(values) - 1
                cut_value = values[rank]
                if cuts and cut_value <= cuts[-1]:
                    continue
                cuts.append(cut_value)
            feature_cuts.append(cuts)
        return feature_cuts

    @staticmethod
    def _quantize_dense_values_linear(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        max_bins: int = 256,
    ) -> list[float]:
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32).reshape(row_count, feature_count)
            mins = np.asarray(feature_mins, dtype=np.float32)
            maxs = np.asarray(feature_maxs, dtype=np.float32)
            span = maxs - mins
            span_ok = span > _PRE_BINNED_INTEGER_TOLERANCE
            nan_mask = np.isnan(arr)
            # Vectorized quantization
            result = np.zeros_like(arr)
            for fi in range(feature_count):
                if not span_ok[fi]:
                    result[:, fi] = 0.0
                else:
                    scaled = ((arr[:, fi] - mins[fi]) / span[fi]) * max_bin
                    rounded = np.floor(scaled + 0.5)
                    result[:, fi] = np.clip(rounded, 0, max_bin)
            # Clamp min/max boundaries
            result = np.where(arr <= mins, 0.0, result)
            result = np.where(arr >= maxs, float(max_bin), result)
            result = np.where(nan_mask, float(nan_bin), result)
            return result.ravel().tolist()
        except (ImportError, ValueError):
            pass
        quantized: list[float] = []
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                min_value = feature_mins[feature_index]
                max_value = feature_maxs[feature_index]
                if value <= min_value:
                    clamped = 0
                elif value >= max_value:
                    clamped = max_bin
                else:
                    s = max_value - min_value
                    if s <= _PRE_BINNED_INTEGER_TOLERANCE:
                        clamped = 0
                    else:
                        scaled = ((value - min_value) / s) * max_bin
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_linear_with_selective_rank(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        rank_flags: Sequence[bool],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                if rank_flags[feature_index]:
                    sorted_values = feature_sorted_values[feature_index]
                    if len(sorted_values) <= 1:
                        clamped = 0
                    else:
                        rank = bisect.bisect_right(sorted_values, value) - 1
                        if rank < 0:
                            rank = 0
                        elif rank >= len(sorted_values):
                            rank = len(sorted_values) - 1
                        scaled = (rank * max_bin) / (len(sorted_values) - 1)
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                else:
                    min_value = feature_mins[feature_index]
                    max_value = feature_maxs[feature_index]
                    if value <= min_value:
                        clamped = 0
                    elif value >= max_value:
                        clamped = max_bin
                    else:
                        span = max_value - min_value
                        if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                            clamped = 0
                        else:
                            scaled = ((value - min_value) / span) * max_bin
                            rounded = GBMRegressor._round_half_away_from_zero(scaled)
                            clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_rank(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                sorted_values = feature_sorted_values[feature_index]
                if len(sorted_values) <= 1:
                    clamped = 0
                else:
                    rank = bisect.bisect_right(sorted_values, value) - 1
                    if rank < 0:
                        rank = 0
                    elif rank >= len(sorted_values):
                        rank = len(sorted_values) - 1
                    scaled = (rank * max_bin) / (len(sorted_values) - 1)
                    rounded = GBMRegressor._round_half_away_from_zero(scaled)
                    clamped = min(max_bin, max(0, rounded))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _quantize_dense_values_quantile(
        flat_values: Sequence[float],
        row_count: int,
        feature_count: int,
        feature_quantile_cuts: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[float]:
        quantized: list[float] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row_index in range(row_count):
            row_base = row_index * feature_count
            for feature_index in range(feature_count):
                value = float(flat_values[row_base + feature_index])
                if math.isnan(value):
                    quantized.append(float(nan_bin))
                    continue
                cuts = feature_quantile_cuts[feature_index]
                bucket = bisect.bisect_right(cuts, value)
                clamped = min(max_bin, max(0, bucket))
                quantized.append(float(clamped))
        return quantized

    @staticmethod
    def _dtype_is_explicit_categorical(dtype: object) -> bool:
        dtype_name = getattr(dtype, "name", None)
        normalized = str(dtype_name if dtype_name is not None else dtype).strip().lower()
        return normalized in {"category", "categorical", "enum"} or "categor" in normalized

    @staticmethod
    def _extract_column_values(
        X: object, column_label: object, column_index: int
    ) -> list[str]:
        candidate = None
        try:
            candidate = X[column_label]  # type: ignore[index]
        except Exception:
            if hasattr(X, "__getitem__"):
                candidate = X[column_index]  # type: ignore[index]
        if candidate is None:
            raise TypeError(
                "X exposes categorical dtypes but column values could not be retrieved"
            )
        values_like = GBMRegressor._coerce_sequence_like(candidate, "categorical_feature_values")
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("categorical_feature_values must be a sequence of strings")
        return [str(value) for value in values_like]

    def _apply_native_cat_mappings_for_predict(self, X: object) -> object:
        """Replace native-categorical columns with integer IDs for prediction.

        If the model has native categorical mappings (from native categorical
        splits), this converts the categorical column values in X to the
        corresponding integer IDs. For DataFrame inputs, string columns are
        looked up in the mapping. Unknown categories become NaN (will follow
        the default_left direction in the tree).
        """
        mappings = getattr(self, "_native_cat_mappings_", None)
        if not mappings:
            return X
        # DataFrame path: replace string columns with float IDs
        if hasattr(X, "columns") and hasattr(X, "iloc"):
            import numpy as np
            X = X.copy()
            for feat_idx, cat_map in mappings.items():
                if feat_idx < len(X.columns):
                    col = X.iloc[:, feat_idx]
                    # Build a string-keyed float-valued dict for vectorized .map()
                    float_map = {str(k): np.float32(v) for k, v in cat_map.items()}
                    mapped = col.astype(str).map(float_map).astype(np.float32)
                    X[X.columns[feat_idx]] = mapped
            return X
        # numpy array path: values are already float, return as-is
        if hasattr(X, "dtype") and hasattr(X, "shape"):
            return X
        # List-of-lists path: convert string values to float IDs
        if isinstance(X, (list, tuple)) and len(X) > 0:
            first = X[0]
            if isinstance(first, (list, tuple)):
                result = []
                for row in X:
                    new_row = list(row)
                    for feat_idx, cat_map in mappings.items():
                        if feat_idx < len(new_row):
                            val = str(new_row[feat_idx])
                            new_row[feat_idx] = float(cat_map[val]) if val in cat_map else float("nan")
                    result.append(new_row)
                return result
        return X

    @staticmethod
    def _extract_categorical_values_for_index(
        X: object, categorical_feature_index: int, row_count: int
    ) -> list[str]:
        if hasattr(X, "columns") and hasattr(X, "dtypes"):
            columns = GBMRegressor._coerce_sequence_like(getattr(X, "columns"), "X.columns")
            if isinstance(columns, Sequence) and categorical_feature_index < len(columns):
                return GBMRegressor._extract_column_values(
                    X, columns[categorical_feature_index], categorical_feature_index
                )

        rows_like = GBMRegressor._coerce_sequence_like(X, "X")
        if not isinstance(rows_like, Sequence) or isinstance(rows_like, (str, bytes)):
            raise TypeError(
                "categorical eval_set values could not be extracted from X"
            )
        if len(rows_like) != row_count:
            raise ValueError(
                "categorical eval_set values must match the number of validation rows"
            )

        values: list[str] = []
        for row in rows_like:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("each X row must be a sequence when extracting categories")
            if categorical_feature_index >= len(row):
                raise ValueError(
                    "categorical_feature_index must be within validation feature bounds"
                )
            values.append(str(row[categorical_feature_index]))
        return values

    @staticmethod
    def _infer_explicit_categorical_feature(
        X: object,
    ) -> tuple[int, list[str]] | None:
        """Infer a single categorical column from X (backward-compat).

        Raises ValueError if multiple categorical columns are found.
        """
        result = GBMRegressor._infer_explicit_categorical_features(X)
        if result is None:
            return None
        indices, values_list = result
        if len(indices) > 1:
            raise ValueError(
                "X contains multiple explicit categorical columns; set categorical_feature_index explicitly"
            )
        return indices[0], values_list[0]

    @staticmethod
    def _infer_explicit_categorical_features(
        X: object,
    ) -> tuple[list[int], list[list[str]]] | None:
        """Infer all categorical columns from a DataFrame-like X.

        Returns (indices, values_lists) where each element in values_lists
        is the string values for the corresponding categorical column.
        Returns None if no categorical columns are detected.
        """
        if not hasattr(X, "dtypes") or not hasattr(X, "columns"):
            return None
        dtypes = GBMRegressor._coerce_sequence_like(getattr(X, "dtypes"), "X.dtypes")
        columns = GBMRegressor._coerce_sequence_like(getattr(X, "columns"), "X.columns")
        if not isinstance(dtypes, Sequence) or not isinstance(columns, Sequence):
            return None
        if len(dtypes) != len(columns) or len(columns) == 0:
            return None
        categorical_indices = [
            index
            for index, dtype in enumerate(dtypes)
            if GBMRegressor._dtype_is_explicit_categorical(dtype)
        ]
        if not categorical_indices:
            return None
        values_list: list[list[str]] = []
        for idx in categorical_indices:
            column_values = GBMRegressor._extract_column_values(
                X, columns[idx], idx
            )
            values_list.append(column_values)
        return categorical_indices, values_list

    @staticmethod
    def _check_pre_binned_integers(flat_values: Sequence[float]) -> bool:
        """Check if flat values are pre-binned non-negative integers. Uses numpy fast path."""
        try:
            import numpy as np
            arr = np.asarray(flat_values, dtype=np.float32)
            if np.any(np.isnan(arr)):
                return False
            if np.any(arr < 0.0):
                return False
            rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
            return bool(np.all(np.abs(arr - rounded) <= _PRE_BINNED_INTEGER_TOLERANCE))
        except ImportError:
            pass
        for value in flat_values:
            if math.isnan(value):
                return False
            if value < 0.0:
                return False
            rounded = GBMRegressor._round_half_away_from_zero(value)
            if abs(value - float(rounded)) > _PRE_BINNED_INTEGER_TOLERANCE:
                return False
        return True

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        if value >= 0.0:
            return int(math.floor(value + 0.5))
        return int(math.ceil(value - 0.5))

    @staticmethod
    def _rows_are_pre_binned(rows: Sequence[Sequence[float]]) -> bool:
        for row in rows:
            for value in row:
                if math.isnan(value):
                    return False
                if value < 0.0:
                    return False
                rounded = float(GBMRegressor._round_half_away_from_zero(value))
                if abs(value - rounded) > _PRE_BINNED_INTEGER_TOLERANCE:
                    return False
        return True

    @staticmethod
    def _derive_continuous_feature_bounds(
        rows: Sequence[Sequence[float]],
    ) -> tuple[list[float], list[float]]:
        feature_count = len(rows[0])
        mins = [float("inf")] * feature_count
        maxs = [float("-inf")] * feature_count
        for row in rows:
            for feature_index, value in enumerate(row):
                if value < mins[feature_index]:
                    mins[feature_index] = value
                if value > maxs[feature_index]:
                    maxs[feature_index] = value
        return mins, maxs

    @staticmethod
    def _derive_continuous_feature_sorted_values(
        rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        feature_count = len(rows[0])
        columns: list[list[float]] = [[] for _ in range(feature_count)]
        for row in rows:
            for feature_index, value in enumerate(row):
                if not math.isnan(value):
                    columns[feature_index].append(value)
        for feature_index in range(feature_count):
            columns[feature_index].sort()
        return columns

    @staticmethod
    def _derive_continuous_feature_tail_rank_plan(
        rows: Sequence[Sequence[float]],
        core_span_ratio_threshold: float,
    ) -> tuple[list[bool], list[list[float]]]:
        columns = GBMRegressor._derive_continuous_feature_sorted_values(rows)
        flags: list[bool] = []
        for values in columns:
            value_count = len(values)
            if value_count < 5:
                flags.append(False)
                continue
            full_span = values[-1] - values[0]
            if full_span <= _PRE_BINNED_INTEGER_TOLERANCE:
                flags.append(False)
                continue
            trim_count = max(1, int(math.floor(value_count * 0.1)))
            if trim_count * 2 >= value_count:
                flags.append(False)
                continue
            core_low = values[trim_count]
            core_high = values[value_count - 1 - trim_count]
            core_span = core_high - core_low
            ratio = core_span / full_span
            flags.append(
                math.isfinite(ratio) and ratio <= core_span_ratio_threshold
            )
        return flags, columns

    @staticmethod
    def _quantize_rows_linear(
        rows: Sequence[Sequence[float]],
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                min_value = feature_mins[feature_index]
                max_value = feature_maxs[feature_index]
                if value <= min_value:
                    clamped = 0
                elif value >= max_value:
                    clamped = max_bin
                else:
                    span = max_value - min_value
                    if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                        clamped = 0
                    else:
                        scaled = ((value - min_value) / span) * max_bin
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _quantize_rows_linear_with_selective_rank(
        rows: Sequence[Sequence[float]],
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
        rank_flags: Sequence[bool],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                if rank_flags[feature_index]:
                    sorted_values = feature_sorted_values[feature_index]
                    if len(sorted_values) <= 1:
                        clamped = 0
                    else:
                        rank = bisect.bisect_right(sorted_values, value) - 1
                        if rank < 0:
                            rank = 0
                        elif rank >= len(sorted_values):
                            rank = len(sorted_values) - 1
                        scaled = (rank * max_bin) / (len(sorted_values) - 1)
                        rounded = GBMRegressor._round_half_away_from_zero(scaled)
                        clamped = min(max_bin, max(0, rounded))
                else:
                    min_value = feature_mins[feature_index]
                    max_value = feature_maxs[feature_index]
                    if value <= min_value:
                        clamped = 0
                    elif value >= max_value:
                        clamped = max_bin
                    else:
                        span = max_value - min_value
                        if span <= _PRE_BINNED_INTEGER_TOLERANCE:
                            clamped = 0
                        else:
                            scaled = ((value - min_value) / span) * max_bin
                            rounded = GBMRegressor._round_half_away_from_zero(scaled)
                            clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _quantize_rows_rank(
        rows: Sequence[Sequence[float]],
        feature_sorted_values: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                sorted_values = feature_sorted_values[feature_index]
                if len(sorted_values) <= 1:
                    clamped = 0
                else:
                    rank = bisect.bisect_right(sorted_values, value) - 1
                    if rank < 0:
                        rank = 0
                    elif rank >= len(sorted_values):
                        rank = len(sorted_values) - 1
                    scaled = (rank * max_bin) / (len(sorted_values) - 1)
                    rounded = GBMRegressor._round_half_away_from_zero(scaled)
                    clamped = min(max_bin, max(0, rounded))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    @staticmethod
    def _derive_continuous_feature_quantile_cuts(
        rows: Sequence[Sequence[float]],
        max_bins: int,
    ) -> list[list[float]]:
        feature_count = len(rows[0])
        columns: list[list[float]] = [[] for _ in range(feature_count)]
        for row in rows:
            for feature_index, value in enumerate(row):
                if not math.isnan(value):
                    columns[feature_index].append(value)

        feature_cuts: list[list[float]] = []
        for feature_index in range(feature_count):
            values = columns[feature_index]
            values.sort()
            if len(values) <= 1:
                feature_cuts.append([])
                continue

            bin_count = min(max_bins, len(values))
            cuts: list[float] = []
            for quantile_index in range(1, bin_count):
                rank = (quantile_index * len(values)) // bin_count
                if rank >= len(values):
                    rank = len(values) - 1
                cut_value = values[rank]
                if cuts and cut_value <= cuts[-1]:
                    continue
                cuts.append(cut_value)
            feature_cuts.append(cuts)
        return feature_cuts

    @staticmethod
    def _quantize_rows_quantile(
        rows: Sequence[Sequence[float]],
        feature_quantile_cuts: Sequence[Sequence[float]],
        max_bins: int = 256,
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _max_data_bin_for_max_bins(max_bins)
        nan_bin = _nan_bin_for_max_bins(max_bins)
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                if math.isnan(value):
                    quantized_row.append(float(nan_bin))
                    continue
                cuts = feature_quantile_cuts[feature_index]
                bucket = bisect.bisect_right(cuts, value)
                clamped = min(max_bin, max(0, bucket))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    def _quantize_rows_for_prediction(
        self, rows: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        if self.continuous_binning_strategy == "linear":
            mins, maxs = self._require_continuous_feature_bounds()
            rank_flags = self._continuous_feature_linear_rank_flags
            if rank_flags is not None and any(rank_flags):
                sorted_values = self._require_continuous_feature_sorted_values()
                return self._quantize_rows_linear_with_selective_rank(
                    rows, mins, maxs, rank_flags, sorted_values,
                    max_bins=self.continuous_binning_max_bins,
                )
            return self._quantize_rows_linear(
                rows, mins, maxs,
                max_bins=self.continuous_binning_max_bins,
            )
        if self.continuous_binning_strategy == "rank":
            sorted_values = self._require_continuous_feature_sorted_values()
            return self._quantize_rows_rank(
                rows, sorted_values,
                max_bins=self.continuous_binning_max_bins,
            )
        quantile_cuts = self._require_continuous_feature_quantile_cuts()
        return self._quantize_rows_quantile(
            rows, quantile_cuts,
            max_bins=self.continuous_binning_max_bins,
        )

    def _require_continuous_feature_bounds(self) -> tuple[list[float], list[float]]:
        if self._continuous_feature_mins is None or self._continuous_feature_maxs is None:
            raise RuntimeError(
                "continuous-feature quantization bounds are missing; refit the model"
            )
        return self._continuous_feature_mins, self._continuous_feature_maxs

    def _require_continuous_feature_sorted_values(self) -> list[list[float]]:
        if self._continuous_feature_sorted_values is None:
            raise RuntimeError(
                "continuous-feature quantization bounds are missing; refit the model"
            )
        return self._continuous_feature_sorted_values

    def _require_continuous_feature_quantile_cuts(self) -> list[list[float]]:
        if self._continuous_feature_quantile_cuts is None:
            raise RuntimeError(
                "continuous-feature quantization cuts are missing; refit the model"
            )
        return self._continuous_feature_quantile_cuts

    def _apply_continuous_binning_metadata(self, metadata: object) -> None:
        self._uses_continuous_binning = bool(
            getattr(metadata, "uses_continuous_binning", False)
        )
        self._continuous_feature_mins = getattr(metadata, "feature_mins", None)
        self._continuous_feature_maxs = getattr(metadata, "feature_maxs", None)
        self._continuous_feature_sorted_values = getattr(
            metadata, "feature_sorted_values", None
        )
        self._continuous_feature_quantile_cuts = getattr(
            metadata, "feature_quantile_cuts", None
        )
        self._continuous_feature_linear_rank_flags = getattr(
            metadata, "feature_linear_rank_flags", None
        )

    def _reset_fitted_state(self) -> None:
        self._is_fitted = False
        self._artifact_bytes = None
        self._native_predictor_handle = None
        self._float_thresholds_converted = False
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins = None
        self._continuous_feature_maxs = None
        self._continuous_feature_sorted_values = None
        self._continuous_feature_quantile_cuts = None
        self._continuous_feature_linear_rank_flags = None
        self.feature_names_in_ = None
        self.best_iteration_ = None
        self.best_score_ = None
        self.n_estimators_ = None
        self.rounds_completed_ = None
        self.stop_reason_ = None
        self.evals_result_ = None
        self.fit_timing_ = None

    # ── Serialization / persistence ──────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # _native_predictor_handle is a PyO3 object and cannot be pickled.
        # It will be lazily reconstructed from _artifact_bytes on first predict().
        state.pop("_native_predictor_handle", None)
        # Custom objective callables are not serializable.  Store None and let
        # the user re-provide the callable if they need to re-train.
        if callable(state.get("objective")):
            import warnings
            warnings.warn(
                "Custom objective callable cannot be pickled. "
                "The model artifact is preserved for prediction, but "
                "re-training will require re-providing the objective callable.",
                UserWarning,
                stacklevel=2,
            )
            state["objective"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._native_predictor_handle = None
        self._float_thresholds_converted = False
        self._predictor_needs_rebuild = True

    def save_model(self, path: str) -> None:
        """Save the fitted model to a file.

        The file contains a JSON metadata header (constructor params, binning
        metadata, training history) followed by the raw binary artifact.  Use
        :meth:`load_model` to restore.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        import json

        saved_params = self.get_params()
        # Callable objectives are not JSON-serializable; store "custom" string.
        if callable(saved_params.get("objective")):
            saved_params["objective"] = None
        metadata = {
            "params": saved_params,
            "n_features_in": self._n_features_in,
            "uses_continuous_binning": self._uses_continuous_binning,
            "continuous_feature_mins": self._continuous_feature_mins,
            "continuous_feature_maxs": self._continuous_feature_maxs,
            "continuous_feature_sorted_values": self._continuous_feature_sorted_values,
            "continuous_feature_quantile_cuts": self._continuous_feature_quantile_cuts,
            "continuous_feature_linear_rank_flags": self._continuous_feature_linear_rank_flags,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "n_estimators_actual": self.n_estimators_,
            "evals_result": self.evals_result_,
            "feature_names_in": self.feature_names_in_,
            "native_cat_mappings": (
                {str(k): v for k, v in self._native_cat_mappings_.items()}
                if self._native_cat_mappings_
                else None
            ),
        }
        # Classifier-specific metadata
        from .classifier import GBMClassifier
        if isinstance(self, GBMClassifier):
            metadata["classifier_classes"] = getattr(self, "classes_", None)
            metadata["classifier_n_classes"] = getattr(self, "n_classes_", None)
            encoder = getattr(self, "_label_encoder", None)
            metadata["classifier_label_encoder"] = (
                {str(k): v for k, v in encoder.items()} if encoder is not None else None
            )
            metadata["classifier_num_classes_for_training"] = getattr(
                self, "_num_classes_for_training", None
            )
        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_len = len(metadata_json)

        with open(path, "wb") as f:
            f.write(b"AGBP")  # magic: AlloyGBM Python model
            f.write(metadata_len.to_bytes(4, "little"))
            f.write(metadata_json)
            f.write(self._artifact_bytes)

    @classmethod
    def load_model(cls, path: str) -> "GBMRegressor":
        """Load a model previously saved with :meth:`save_model`.

        Returns a fitted ``GBMRegressor`` ready for prediction.
        """
        import json

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"AGBP":
                raise ValueError(
                    f"Not a valid AlloyGBM model file (expected magic b'AGBP', got {magic!r})"
                )
            metadata_len = int.from_bytes(f.read(4), "little")
            metadata_json = f.read(metadata_len)
            artifact_bytes = f.read()

        metadata = json.loads(metadata_json)
        params = metadata["params"]
        # Filter to known params for forward compatibility.
        # Use get_params() keys from a default instance to correctly handle
        # subclasses that use **kwargs (e.g. GBMRanker, GBMClassifier).
        try:
            # Build a temporary default instance to discover valid param names.
            # This works even for subclasses with **kwargs forwarding.
            _probe = cls.__new__(cls)
            cls.__init__(_probe)
            known = set(_probe.get_params().keys())
        except Exception:
            # Fallback: accept all saved params.
            known = set(params.keys())
        model = cls(**{k: v for k, v in params.items() if k in known})
        model._artifact_bytes = artifact_bytes
        model._n_features_in = metadata["n_features_in"]
        model._uses_continuous_binning = metadata["uses_continuous_binning"]
        model._continuous_feature_mins = metadata.get("continuous_feature_mins")
        model._continuous_feature_maxs = metadata.get("continuous_feature_maxs")
        model._continuous_feature_sorted_values = metadata.get(
            "continuous_feature_sorted_values"
        )
        model._continuous_feature_quantile_cuts = metadata.get(
            "continuous_feature_quantile_cuts"
        )
        model._continuous_feature_linear_rank_flags = metadata.get(
            "continuous_feature_linear_rank_flags"
        )
        model.best_iteration_ = metadata.get("best_iteration")
        model.best_score_ = metadata.get("best_score")
        model.n_estimators_ = metadata.get("n_estimators_actual")
        model.evals_result_ = metadata.get("evals_result")
        model.feature_names_in_ = metadata.get("feature_names_in")
        saved_cat_mappings = metadata.get("native_cat_mappings")
        if saved_cat_mappings:
            model._native_cat_mappings_ = {
                int(k): v for k, v in saved_cat_mappings.items()
            }
        else:
            model._native_cat_mappings_ = None
        model._is_fitted = True
        model._native_predictor_handle = None
        model._float_thresholds_converted = False
        # Eagerly reconstruct the native predictor so predict() uses the fast path.
        model._native_predictor_handle = cls._build_native_predictor_handle(
            artifact_bytes
        )
        model._convert_predictor_thresholds_to_float()

        # Restore subclass-specific fitted attributes.
        from .classifier import GBMClassifier

        if isinstance(model, GBMClassifier):
            saved_classes = metadata.get("classifier_classes")
            if saved_classes is not None:
                model.classes_ = saved_classes
                model.n_classes_ = metadata.get("classifier_n_classes", len(saved_classes))
            else:
                model.classes_ = [0, 1]
                model.n_classes_ = 2
            saved_encoder = metadata.get("classifier_label_encoder")
            if saved_encoder is not None:
                model._label_encoder = {int(k): v for k, v in saved_encoder.items()}
                model._label_decoder = {v: int(k) for k, v in saved_encoder.items()}
            else:
                model._label_encoder = None
                model._label_decoder = None
            model._num_classes_for_training = metadata.get(
                "classifier_num_classes_for_training"
            )

        return model

    def save_artifact(self, path: str) -> None:
        """Save only the raw model artifact bytes to a file.

        The resulting file can be loaded with :meth:`predict_from_artifact` for
        lightweight deployment scenarios where retraining is not needed.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving artifact")
        with open(path, "wb") as f:
            f.write(self._artifact_bytes)

    @property
    def artifact_bytes(self) -> bytes:
        """The raw binary model artifact.

        Can be stored externally (database, object store) and used with
        :meth:`predict_from_artifact` for serving without the full model.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to access artifact bytes")
        return self._artifact_bytes

    @staticmethod
    def _build_native_predictor_handle(artifact_bytes: bytes) -> object | None:
        try:
            native_predictor_handle_class = _load_native_predictor_handle_class()
        except RuntimeError:
            return None
        try:
            return native_predictor_handle_class(artifact_bytes, strict=True)
        except Exception:
            pass
        # Fallback: non-strict loading (required for multi-class models which
        # use MultiClassTrees section instead of the dual Trees+PredictorLayout
        # format that strict mode requires).
        try:
            return native_predictor_handle_class(artifact_bytes, strict=False)
        except Exception:
            return None

    def _convert_predictor_thresholds_to_float(self) -> None:
        """Convert bin-index thresholds to float thresholds on the native predictor.

        After conversion, predict_dense works directly on raw floats — no quantization needed.
        Supports linear binning, quantile binning, and pre-binned integer data.
        """
        if self._native_predictor_handle is None:
            return
        try:
            if not self._uses_continuous_binning:
                # Pre-binned integer data: threshold_float = bin + 0.5
                convert_fn = getattr(
                    self._native_predictor_handle,
                    "convert_thresholds_to_float_prebinned",
                    None,
                )
                if callable(convert_fn):
                    result = convert_fn()
                    if result is None:
                        self._float_thresholds_converted = True
                return

            strategy = self.continuous_binning_strategy
            if strategy == "linear":
                rank_flags = self._continuous_feature_linear_rank_flags
                if rank_flags is not None and any(rank_flags):
                    return  # rank features need bin-based prediction
                convert_fn = getattr(
                    self._native_predictor_handle, "convert_thresholds_to_float", None
                )
                if not callable(convert_fn):
                    return
                mins, maxs = self._require_continuous_feature_bounds()
                max_data_bin = _max_data_bin_for_max_bins(
                    self.continuous_binning_max_bins
                )
                result = convert_fn(list(mins), list(maxs), max_data_bin)
                # Rust PyO3 method returns None on success; mock objects return Mock.
                if result is None:
                    self._float_thresholds_converted = True
            elif strategy == "quantile":
                convert_fn = getattr(
                    self._native_predictor_handle,
                    "convert_thresholds_to_float_quantile",
                    None,
                )
                if not callable(convert_fn):
                    return
                cuts = self._continuous_feature_quantile_cuts
                if cuts is None:
                    return
                # Convert list[list[float]] → list[list[f32]] for Rust
                result = convert_fn(
                    [[float(v) for v in c] for c in cuts]
                )
                if result is None:
                    self._float_thresholds_converted = True
        except Exception:
            self._float_thresholds_converted = False

    @staticmethod
    def _validate_targets(y: object) -> list[float]:
        targets_like = GBMRegressor._coerce_sequence_like(y, "y")
        if not isinstance(targets_like, Sequence) or isinstance(targets_like, (str, bytes)):
            raise TypeError("y must be a sequence of numeric values")
        if len(targets_like) == 0:
            raise ValueError("y must contain at least one value")
        return [float(value) for value in targets_like]

    @staticmethod
    def _validate_categorical_values(
        categorical_feature_values: object, row_count: int
    ) -> list[str]:
        values_like = GBMRegressor._coerce_sequence_like(
            categorical_feature_values, "categorical_feature_values"
        )
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("categorical_feature_values must be a sequence of strings")
        if len(values_like) != row_count:
            raise ValueError(
                "categorical_feature_values must have the same number of rows as X"
            )
        return [str(value) for value in values_like]

    @staticmethod
    def _validate_categorical_values_list(
        categorical_feature_values_list: object, row_count: int
    ) -> list[list[str]]:
        """Validate a list of per-column categorical value sequences."""
        outer = GBMRegressor._coerce_sequence_like(
            categorical_feature_values_list, "categorical_feature_values_list"
        )
        if not isinstance(outer, Sequence) or isinstance(outer, (str, bytes)):
            raise TypeError(
                "categorical_feature_values_list must be a sequence of value sequences"
            )
        result: list[list[str]] = []
        for i, inner in enumerate(outer):
            validated = GBMRegressor._validate_categorical_values(inner, row_count)
            result.append(validated)
        return result

    @staticmethod
    def _extract_categorical_values_for_indices(
        X: object, categorical_feature_indices: list[int], row_count: int
    ) -> list[list[str]]:
        """Extract categorical string values for multiple feature indices."""
        return [
            GBMRegressor._extract_categorical_values_for_index(
                X, idx, row_count
            )
            for idx in categorical_feature_indices
        ]

    @staticmethod
    def _validate_time_index(time_index: object, row_count: int) -> list[int]:
        values_like = GBMRegressor._coerce_sequence_like(time_index, "time_index")
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("time_index must be a sequence of integer-like values")
        if len(values_like) != row_count:
            raise ValueError("time_index must have the same number of rows as X")
        return [int(value) for value in values_like]

    @staticmethod
    def _validate_sample_weight(
        sample_weight: object, row_count: int
    ) -> list[float]:
        """Validate and convert sample weights.

        Weights must be finite and non-negative. Zero weights are allowed
        and produce zero-gradient pairs (effectively excluding the row from
        that training round).
        """
        values_like = GBMRegressor._coerce_sequence_like(
            sample_weight, "sample_weight"
        )
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("sample_weight must be a sequence of numeric values")
        if len(values_like) != row_count:
            raise ValueError(
                "sample_weight must have the same number of elements as rows in X"
            )
        weights: list[float] = []
        for i, w in enumerate(values_like):
            fw = float(w)
            if not math.isfinite(fw):
                raise ValueError(f"sample_weight[{i}] must be finite")
            if fw < 0.0:
                raise ValueError(f"sample_weight[{i}] must be non-negative")
            weights.append(fw)
        return weights

    @staticmethod
    def _validate_group(group: object, row_count: int) -> list[int]:
        values_like = GBMRegressor._coerce_sequence_like(group, "group")
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("group must be a sequence of integer-like values")
        if len(values_like) != row_count:
            raise ValueError(
                "group must have the same number of elements as rows in X"
            )
        group_ids: list[int] = []
        for i, g in enumerate(values_like):
            gi = int(g)
            if gi < 0:
                raise ValueError(f"group[{i}] must be non-negative")
            group_ids.append(gi)
        return group_ids

    @staticmethod
    def _coerce_sequence_like(value: object, argument_name: str) -> object:
        current = value
        for _ in range(4):
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                return current

            next_value: object | None = None
            if hasattr(current, "to_numpy"):
                next_value = current.to_numpy()  # type: ignore[call-arg]
            elif hasattr(current, "to_list"):
                next_value = current.to_list()  # type: ignore[call-arg]
            elif hasattr(current, "tolist"):
                next_value = current.tolist()  # type: ignore[call-arg]

            if next_value is None or next_value is current:
                break
            current = next_value

        raise TypeError(
            f"{argument_name} must be a sequence or provide to_numpy/to_list/tolist conversion"
        )
