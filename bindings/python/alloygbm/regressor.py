"""Python-facing estimator baseline for v0.0.3."""

from __future__ import annotations

from collections.abc import Sequence

class GBMRegressor:
    """Sklearn-style contract stub for the future native estimator."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        seed: int = 0,
        deterministic: bool = True,
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
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
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}"
            ")"
        )

    def get_params(self, deep: bool = True) -> dict[str, float | int | bool]:
        """Return estimator parameters in sklearn-compatible shape."""
        del deep  # Not used until nested estimators exist.
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    def set_params(self, **params: float | int | bool) -> "GBMRegressor":
        """Set estimator parameters with constructor-equivalent validation."""
        allowed = {"learning_rate", "max_depth", "seed", "deterministic"}
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
