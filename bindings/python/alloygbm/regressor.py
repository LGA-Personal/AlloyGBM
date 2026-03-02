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


def _load_native_predictor_predict_batch_canonical():
    try:
        from alloygbm._alloygbm import predictor_predict_batch_canonical
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native canonical predictor binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return predictor_predict_batch_canonical


def _load_native_train_regression_artifact():
    try:
        from alloygbm._alloygbm import train_regression_artifact
    except Exception as exc:  # pragma: no cover - exercised via contract tests.
        raise RuntimeError(
            "native training binding is unavailable; build/install the alloygbm extension module"
        ) from exc
    return train_regression_artifact


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
        self._is_fitted = False
        self._artifact_bytes: bytes | None = None
        self._n_features_in = 0

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
            f"deterministic={self.deterministic}"
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
        }

    def set_params(self, **params: float | int | bool | None) -> "GBMRegressor":
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

        return self

    def fit(self, X: object, y: object) -> "GBMRegressor":
        """Fit native-backed regression model artifact state."""
        rows = self._validate_rows(X)
        targets = self._validate_targets(y)
        if len(rows) != len(targets):
            raise ValueError("X and y must contain the same number of rows")

        train_regression_artifact = _load_native_train_regression_artifact()
        artifact_bytes = train_regression_artifact(
            rows=rows,
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
        )

        self._n_features_in = len(rows[0])
        self._artifact_bytes = bytes(artifact_bytes)
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
        predictor_predict_batch_canonical = (
            _load_native_predictor_predict_batch_canonical()
        )
        return list(predictor_predict_batch_canonical(self._artifact_bytes, rows))

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
    def _validate_targets(y: object) -> list[float]:
        targets_like = GBMRegressor._coerce_sequence_like(y, "y")
        if not isinstance(targets_like, Sequence) or isinstance(targets_like, (str, bytes)):
            raise TypeError("y must be a sequence of numeric values")
        if len(targets_like) == 0:
            raise ValueError("y must contain at least one value")
        return [float(value) for value in targets_like]

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
