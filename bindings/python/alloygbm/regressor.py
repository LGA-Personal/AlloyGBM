"""Python-facing estimator stubs for v0.0.1."""

from __future__ import annotations


class GBMRegressor:
    """Stub estimator with constructor-time parameter validation only."""

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

    def __repr__(self) -> str:
        return (
            "GBMRegressor("
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}"
            ")"
        )
