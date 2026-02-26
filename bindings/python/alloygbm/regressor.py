"""Python-facing estimator baseline scaffold for AlloyGBM."""

from __future__ import annotations

from collections.abc import Sequence


def _load_native_predictor_predict_batch():
    try:
        from alloygbm._alloygbm import predictor_predict_batch
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch


class GBMRegressor:
    """Sklearn-style contract stub for the future native estimator."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        row_subsample: float = 1.0,
        col_subsample: float = 1.0,
        early_stopping_rounds: int | None = None,
        min_validation_improvement: float = 0.0,
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        if not (0.0 < row_subsample <= 1.0):
            raise ValueError("row_subsample must be in (0.0, 1.0]")
        if not (0.0 < col_subsample <= 1.0):
            raise ValueError("col_subsample must be in (0.0, 1.0]")
        if early_stopping_rounds is not None and int(early_stopping_rounds) <= 0:
            raise ValueError("early_stopping_rounds must be greater than 0 when set")
        if min_validation_improvement < 0.0:
            raise ValueError("min_validation_improvement must be >= 0")

        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
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
        self._is_fitted = False
        self._baseline_prediction = 0.0
        self._n_features_in = 0

    def __repr__(self) -> str:
        return (
            "GBMRegressor("
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"row_subsample={self.row_subsample}, "
            f"col_subsample={self.col_subsample}, "
            f"early_stopping_rounds={self.early_stopping_rounds}, "
            f"min_validation_improvement={self.min_validation_improvement}, "
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}"
            ")"
        )

    def get_params(self, deep: bool = True) -> dict[str, float | int | bool | None]:
        """Return estimator parameters in sklearn-compatible shape."""
        del deep  # Not used until nested estimators exist.
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "row_subsample": self.row_subsample,
            "col_subsample": self.col_subsample,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_validation_improvement": self.min_validation_improvement,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    def set_params(self, **params: float | int | bool | None) -> "GBMRegressor":
        """Set estimator parameters with constructor-equivalent validation."""
        allowed = {
            "learning_rate",
            "max_depth",
            "row_subsample",
            "col_subsample",
            "early_stopping_rounds",
            "min_validation_improvement",
            "seed",
            "deterministic",
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

        return self

    def fit(self, X: object, y: object) -> "GBMRegressor":
        """Fit a deterministic constant baseline predictor."""
        rows = self._validate_rows(X)
        targets = self._validate_targets(y)
        if len(rows) != len(targets):
            raise ValueError("X and y must contain the same number of rows")

        self._n_features_in = len(rows[0])
        self._baseline_prediction = sum(targets) / len(targets)
        self._is_fitted = True
        return self

    def predict(self, X: object) -> list[float]:
        """Predict constant baseline values for each input row."""
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before predict")
        rows = self._validate_rows(X)
        if len(rows[0]) != self._n_features_in:
            raise ValueError(
                f"X feature count {len(rows[0])} does not match fitted feature count "
                f"{self._n_features_in}"
            )
        return [self._baseline_prediction] * len(rows)

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
        if not isinstance(X, Sequence) or isinstance(X, (str, bytes)):
            raise TypeError("X must be a sequence of feature rows")
        if len(X) == 0:
            raise ValueError("X must contain at least one row")

        normalized: list[list[float]] = []
        expected_width: int | None = None
        for row in X:
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
    def _validate_targets(y: object) -> list[float]:
        if not isinstance(y, Sequence) or isinstance(y, (str, bytes)):
            raise TypeError("y must be a sequence of numeric values")
        if len(y) == 0:
            raise ValueError("y must contain at least one value")
        return [float(value) for value in y]
