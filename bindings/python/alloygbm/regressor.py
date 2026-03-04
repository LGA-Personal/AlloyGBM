"""Python-facing estimator baseline scaffold for AlloyGBM."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

_PRE_BINNED_INTEGER_TOLERANCE = 1e-6
_MAX_CONTINUOUS_QUANTIZED_BIN = 255
_MIN_CONTINUOUS_QUANTIZED_BINS = 2
_VALID_CONTINUOUS_BINNING_STRATEGIES = {"linear", "rank", "quantile"}


def _load_native_predictor_predict_batch():
    try:
        from alloygbm._alloygbm import predictor_predict_batch
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch


def _load_native_predictor_predict_batch_canonical():
    try:
        from alloygbm._alloygbm import predictor_predict_batch_canonical
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native canonical predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch_canonical


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


def _load_native_shap_explain_rows():
    try:
        from alloygbm._alloygbm import shap_explain_rows
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native SHAP explain binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_explain_rows


def _load_native_shap_global_importance():
    try:
        from alloygbm._alloygbm import shap_global_importance
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native SHAP global-importance binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return shap_global_importance


class GBMRegressor:
    """Sklearn-style contract stub for the future native estimator."""

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
        seed: int = 0,
        deterministic: bool = True,
        continuous_binning_strategy: str = "linear",
        continuous_binning_max_bins: int = 256,
        categorical_feature_index: int | None = None,
        categorical_smoothing: float = 20.0,
        categorical_min_samples_leaf: int = 1,
        categorical_time_aware: bool = False,
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
        if categorical_feature_index is not None and int(categorical_feature_index) < 0:
            raise ValueError("categorical_feature_index must be >= 0 when set")
        if categorical_smoothing < 0.0:
            raise ValueError("categorical_smoothing must be >= 0")
        if int(categorical_min_samples_leaf) <= 0:
            raise ValueError("categorical_min_samples_leaf must be greater than 0")
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
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.continuous_binning_strategy = str(continuous_binning_strategy)
        self.continuous_binning_max_bins = max_bins
        self.categorical_feature_index = (
            int(categorical_feature_index)
            if categorical_feature_index is not None
            else None
        )
        self.categorical_smoothing = float(categorical_smoothing)
        self.categorical_min_samples_leaf = int(categorical_min_samples_leaf)
        self.categorical_time_aware = bool(categorical_time_aware)
        self._is_fitted = False
        self._artifact_bytes: bytes | None = None
        self._native_predictor_handle: object | None = None
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins: list[float] | None = None
        self._continuous_feature_maxs: list[float] | None = None
        self._continuous_feature_sorted_values: list[list[float]] | None = None
        self._continuous_feature_quantile_cuts: list[list[float]] | None = None

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
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"continuous_binning_strategy='{self.continuous_binning_strategy}', "
            f"continuous_binning_max_bins={self.continuous_binning_max_bins}, "
            f"categorical_feature_index={self.categorical_feature_index}, "
            f"categorical_smoothing={self.categorical_smoothing}, "
            f"categorical_min_samples_leaf={self.categorical_min_samples_leaf}, "
            f"categorical_time_aware={self.categorical_time_aware}"
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
            "seed": self.seed,
            "deterministic": self.deterministic,
            "continuous_binning_strategy": self.continuous_binning_strategy,
            "continuous_binning_max_bins": self.continuous_binning_max_bins,
            "categorical_feature_index": self.categorical_feature_index,
            "categorical_smoothing": self.categorical_smoothing,
            "categorical_min_samples_leaf": self.categorical_min_samples_leaf,
            "categorical_time_aware": self.categorical_time_aware,
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
            "seed",
            "deterministic",
            "continuous_binning_strategy",
            "continuous_binning_max_bins",
            "categorical_feature_index",
            "categorical_smoothing",
            "categorical_min_samples_leaf",
            "categorical_time_aware",
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

        return self

    def fit(
        self,
        X: object,
        y: object,
        *,
        categorical_feature_values: object | None = None,
        time_index: object | None = None,
    ) -> "GBMRegressor":
        """Fit native-backed regression model artifact state."""
        rows = self._validate_rows(X)
        targets = self._validate_targets(y)
        if len(rows) != len(targets):
            raise ValueError("X and y must contain the same number of rows")
        if (
            self.categorical_feature_index is not None
            and self.categorical_feature_index >= len(rows[0])
        ):
            raise ValueError(
                "categorical_feature_index must be within fitted feature bounds"
            )

        categorical_values = None
        if categorical_feature_values is not None:
            categorical_values = self._validate_categorical_values(
                categorical_feature_values, len(rows)
            )

        if self.categorical_feature_index is None and categorical_values is not None:
            raise ValueError(
                "categorical_feature_values requires categorical_feature_index to be set"
            )
        if self.categorical_feature_index is not None and categorical_values is None:
            raise ValueError(
                "categorical_feature_values must be provided when categorical_feature_index is set"
            )

        validated_time_index = None
        if time_index is not None:
            validated_time_index = self._validate_time_index(time_index, len(rows))
        if (
            self.categorical_feature_index is not None
            and self.categorical_time_aware
            and validated_time_index is None
        ):
            raise ValueError(
                "time_index must be provided when categorical_time_aware=True and categorical_feature_index is set"
            )

        self._uses_continuous_binning = not self._rows_are_pre_binned(rows)
        if self._uses_continuous_binning:
            if self.continuous_binning_strategy == "linear":
                mins, maxs = self._derive_continuous_feature_bounds(rows)
                self._continuous_feature_mins = mins
                self._continuous_feature_maxs = maxs
                self._continuous_feature_sorted_values = None
                self._continuous_feature_quantile_cuts = None
                training_rows = self._quantize_rows_linear(rows, mins, maxs)
            elif self.continuous_binning_strategy == "rank":
                sorted_values = self._derive_continuous_feature_sorted_values(rows)
                self._continuous_feature_sorted_values = sorted_values
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                self._continuous_feature_quantile_cuts = None
                training_rows = self._quantize_rows_rank(rows, sorted_values)
            else:
                quantile_cuts = self._derive_continuous_feature_quantile_cuts(
                    rows, self.continuous_binning_max_bins
                )
                self._continuous_feature_quantile_cuts = quantile_cuts
                self._continuous_feature_sorted_values = None
                self._continuous_feature_mins = None
                self._continuous_feature_maxs = None
                training_rows = self._quantize_rows_quantile(rows, quantile_cuts)
        else:
            self._continuous_feature_mins = None
            self._continuous_feature_maxs = None
            self._continuous_feature_sorted_values = None
            self._continuous_feature_quantile_cuts = None
            training_rows = rows

        train_regression_artifact = _load_native_train_regression_artifact()
        artifact_bytes = train_regression_artifact(
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
            categorical_feature_index=self.categorical_feature_index,
            categorical_feature_values=categorical_values,
            categorical_smoothing=self.categorical_smoothing,
            categorical_min_samples_leaf=self.categorical_min_samples_leaf,
            categorical_time_aware=self.categorical_time_aware,
            time_index=validated_time_index,
        )

        self._n_features_in = len(rows[0])
        self._artifact_bytes = bytes(artifact_bytes)
        self._native_predictor_handle = self._build_native_predictor_handle(
            self._artifact_bytes
        )
        self._is_fitted = True
        return self

    def predict(self, X: object) -> list[float]:
        """Predict using the fitted native artifact."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before predict")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")
        rows = self._validate_rows(X)
        if len(rows[0]) != self._n_features_in:
            raise ValueError(
                f"X feature count {len(rows[0])} does not match fitted feature count "
                f"{self._n_features_in}"
            )
        if self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(rows)
        if self._native_predictor_handle is not None:
            predict_batch = getattr(self._native_predictor_handle, "predict_batch", None)
            if callable(predict_batch):
                try:
                    return list(predict_batch(rows))
                except RuntimeError:
                    self._native_predictor_handle = None
        predictor_predict_batch_canonical = (
            _load_native_predictor_predict_batch_canonical()
        )
        return list(predictor_predict_batch_canonical(self._artifact_bytes, rows))

    def shap_values(
        self, X: object, *, include_expected_value: bool = False
    ) -> list[list[float]] | tuple[float, list[list[float]]]:
        """Return SHAP values for the provided rows using the fitted artifact."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before shap_values")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")

        rows = self._validate_rows(X)
        if len(rows[0]) != self._n_features_in:
            raise ValueError(
                f"X feature count {len(rows[0])} does not match fitted feature count "
                f"{self._n_features_in}"
            )
        if self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(rows)

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

        rows = self._validate_rows(X)
        if len(rows[0]) != self._n_features_in:
            raise ValueError(
                f"X feature count {len(rows[0])} does not match fitted feature count "
                f"{self._n_features_in}"
            )
        if self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(rows)

        shap_global_importance = _load_native_shap_global_importance()
        importance = shap_global_importance(self._artifact_bytes, rows)
        return [(str(name), float(value)) for name, value in importance]

    @staticmethod
    def predict_from_artifact(
        artifact_bytes: bytes | bytearray | memoryview, X: object
    ) -> list[float]:
        """Run predictor-backed inference from serialized model artifact bytes."""
        if not isinstance(artifact_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("artifact_bytes must be bytes-like")
        rows = GBMRegressor._validate_rows(X)
        predictor_predict_batch = _load_native_predictor_predict_batch()
        return list(predictor_predict_batch(bytes(artifact_bytes), rows))

    @staticmethod
    def _validate_rows(X: object) -> list[list[float]]:
        rows_like = GBMRegressor._coerce_sequence_like(X, "X")
        if not isinstance(rows_like, Sequence) or isinstance(rows_like, (str, bytes)):
            raise TypeError("X must be a sequence of feature rows")
        if len(rows_like) == 0:
            raise ValueError("X must contain at least one row")

        normalized: list[list[float]] = []
        expected_width: int | None = None
        for row in rows_like:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("each X row must be a sequence of numeric values")
            if len(row) == 0:
                raise ValueError("each X row must contain at least one feature value")
            row_values = [float(value) for value in row]
            if expected_width is None:
                expected_width = len(row_values)
            elif len(row_values) != expected_width:
                raise ValueError("all X rows must have the same feature count")
            normalized.append(row_values)

        return normalized

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        if value >= 0.0:
            return int(math.floor(value + 0.5))
        return int(math.ceil(value - 0.5))

    @staticmethod
    def _rows_are_pre_binned(rows: Sequence[Sequence[float]]) -> bool:
        for row in rows:
            for value in row:
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
                columns[feature_index].append(value)
        for feature_index in range(feature_count):
            columns[feature_index].sort()
        return columns

    @staticmethod
    def _quantize_rows_linear(
        rows: Sequence[Sequence[float]],
        feature_mins: Sequence[float],
        feature_maxs: Sequence[float],
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _MAX_CONTINUOUS_QUANTIZED_BIN
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
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
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        max_bin = _MAX_CONTINUOUS_QUANTIZED_BIN
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
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
    ) -> list[list[float]]:
        quantized: list[list[float]] = []
        for row in rows:
            quantized_row: list[float] = []
            for feature_index, value in enumerate(row):
                cuts = feature_quantile_cuts[feature_index]
                bucket = bisect.bisect_right(cuts, value)
                clamped = min(_MAX_CONTINUOUS_QUANTIZED_BIN, max(0, bucket))
                quantized_row.append(float(clamped))
            quantized.append(quantized_row)
        return quantized

    def _quantize_rows_for_prediction(
        self, rows: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        if self.continuous_binning_strategy == "linear":
            mins, maxs = self._require_continuous_feature_bounds()
            return self._quantize_rows_linear(rows, mins, maxs)
        if self.continuous_binning_strategy == "rank":
            sorted_values = self._require_continuous_feature_sorted_values()
            return self._quantize_rows_rank(rows, sorted_values)
        quantile_cuts = self._require_continuous_feature_quantile_cuts()
        return self._quantize_rows_quantile(rows, quantile_cuts)

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

    def _reset_fitted_state(self) -> None:
        self._is_fitted = False
        self._artifact_bytes = None
        self._native_predictor_handle = None
        self._n_features_in = 0
        self._uses_continuous_binning = False
        self._continuous_feature_mins = None
        self._continuous_feature_maxs = None
        self._continuous_feature_sorted_values = None
        self._continuous_feature_quantile_cuts = None

    @staticmethod
    def _build_native_predictor_handle(artifact_bytes: bytes) -> object | None:
        try:
            native_predictor_handle_class = _load_native_predictor_handle_class()
        except RuntimeError:
            return None
        try:
            return native_predictor_handle_class(artifact_bytes, strict=True)
        except Exception:
            return None

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
    def _validate_time_index(time_index: object, row_count: int) -> list[int]:
        values_like = GBMRegressor._coerce_sequence_like(time_index, "time_index")
        if not isinstance(values_like, Sequence) or isinstance(values_like, (str, bytes)):
            raise TypeError("time_index must be a sequence of integer-like values")
        if len(values_like) != row_count:
            raise ValueError("time_index must have the same number of rows as X")
        return [int(value) for value in values_like]

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
