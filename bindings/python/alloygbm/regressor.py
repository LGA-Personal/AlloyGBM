"""Python-facing estimator stubs for v0.0.2."""

from __future__ import annotations


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
        """Validate fit contract shape, then fail explicitly until v0.0.3+."""
        del X, y
        raise NotImplementedError("fit is not implemented in v0.0.2")

    def predict(self, X: object) -> list[float]:
        """Require fitted-state before prediction; kernel remains unimplemented."""
        del X
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before predict")
        raise NotImplementedError("predict is not implemented in v0.0.2")
