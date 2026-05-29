"""Validation and resolution mixin for GBMRegressor."""

from __future__ import annotations

import math
from collections.abc import Sequence


class _ValidationMixin:
    """Mixin carrying validation/resolution methods for GBMRegressor.

    All 27 methods are moved verbatim from GBMRegressor in _core.py.
    ``GBMRegressor`` references inside static method bodies resolve at
    call-time from this module's globals: after defining the class, _core.py
    injects ``_validation.GBMRegressor = GBMRegressor`` (a top-level
    ``from ._core import GBMRegressor`` would create a circular import).
    """

    def _prepare_factor_exposures(self, factor_exposures, n_rows: int):
        if self.neutralization == "none":
            if factor_exposures is not None:
                raise ValueError("factor_exposures were provided but neutralization='none'")
            return None, 0, 0
        if factor_exposures is None:
            raise ValueError("factor_exposures are required when neutralization is active")
        import numpy as np

        arr = np.asarray(factor_exposures, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("factor_exposures must be a 2D array")
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"factor_exposures row count {arr.shape[0]} does not match X row count {n_rows}"
            )
        if arr.shape[1] == 0:
            raise ValueError("factor_exposures must contain at least one factor")
        if not np.all(np.isfinite(arr)):
            raise ValueError("factor_exposures must contain only finite values")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr.ravel().tolist(), int(arr.shape[0]), int(arr.shape[1])

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

    def _resolve_interaction_constraints(self, feature_count: int) -> list[list[int]]:
        """Resolve ``interaction_constraints`` to a ``list[list[int]]`` for the
        native bridge.  Any feature index that exceeds ``feature_count`` is
        skipped (defensive — input is already validated).  Empty list when
        unset.
        """
        del feature_count  # bridge validates against the dataset shape itself
        if self.interaction_constraints is None:
            return []
        return [[int(f) for f in group] for group in self.interaction_constraints]

    def _objective_name(self) -> str:
        """Return the objective function name passed to the native training bridge."""
        if self.objective is not None:
            if callable(self.objective):
                return "custom"
            return str(self.objective)
        return "squared_error"

    def _validate_glm_target_domain(self, y: object, *, role: str) -> None:
        """Reject targets that violate the active GLM objective's domain.

        Called on both training y and validation y from `eval_set` so that
        early-stopping / validation-loss reporting don't operate on values
        that the loss function can't evaluate (e.g. `log(y/μ)` for y=0 in
        Gamma).  No-op for non-GLM objectives.

        `role` is a free-form label ("training" / "validation") used only
        to make the error message specific.
        """
        obj_name = self._objective_name()
        if obj_name not in ("poisson", "tweedie", "gamma"):
            return
        import numpy as _np

        y_arr = _np.asarray(y, dtype=_np.float64)
        if obj_name == "gamma":
            if _np.any(y_arr <= 0):
                raise ValueError(
                    f"objective='gamma' requires strictly positive {role} targets, "
                    f"got min(y)={float(y_arr.min())}"
                )
        else:
            if _np.any(y_arr < 0):
                raise ValueError(
                    f"objective={obj_name!r} requires non-negative {role} targets, "
                    f"got min(y)={float(y_arr.min())}"
                )

    @staticmethod
    def _loss_metric_name_for(objective: str) -> str:
        """Map an objective name to its natural loss metric name."""
        if objective == "binary_crossentropy":
            return "logloss"
        if objective in ("rank_pairwise", "rank_ndcg", "rank_xendcg", "yetirank"):
            return "ndcg"
        if objective == "queryrmse":
            return "queryrmse"
        if objective == "poisson":
            return "poisson_deviance"
        if objective == "gamma":
            return "gamma_deviance"
        if objective == "tweedie":
            return "tweedie_deviance"
        if objective == "quantile":
            return "quantile"
        if objective == "custom":
            return "loss"
        return "mse"

    def _loss_metric_name(self) -> str:
        """Return the natural loss metric name for this estimator's objective."""
        return self._loss_metric_name_for(self._objective_name())

    def _record_fit_neutralization_contract(self) -> None:
        self._fit_neutralization = self.neutralization
        self._fit_factor_neutralization_lambda = self.factor_neutralization_lambda
        self._fit_factor_penalty = self.factor_penalty

    def _fitted_neutralization_contract(self) -> tuple[str, float, float]:
        fit_neutralization = getattr(self, "_fit_neutralization", None)
        fit_lambda = getattr(self, "_fit_factor_neutralization_lambda", None)
        fit_penalty = getattr(self, "_fit_factor_penalty", None)
        if fit_neutralization is None:
            fit_neutralization = getattr(self, "neutralization", "none")
        if fit_lambda is None:
            fit_lambda = getattr(self, "factor_neutralization_lambda", 1e-6)
        if fit_penalty is None:
            fit_penalty = getattr(self, "factor_penalty", 0.0)
        return str(fit_neutralization), float(fit_lambda), float(fit_penalty)

    @staticmethod
    def _raise_if_neutralized_warm_start_contract(
        fit_neutralization: str,
        factor_exposures: object,
    ) -> None:
        """Validate the warm-start contract for neutralized models.

        As of v0.7.1, warm-starting a neutralized model is supported, but the
        caller must supply the same ``factor_exposures`` matrix used for the
        initial fit so the projection has a consistent column space.  We
        cannot persist the exposures matrix on the artifact (would balloon
        the model size and surface a sensitive dataset), so the contract is
        positional — passing a different matrix changes which directions are
        projected away and breaks numerical equivalence to fresh training.
        """
        if fit_neutralization == "none":
            return
        if factor_exposures is None:
            raise ValueError(
                "warm-start training of a neutralized model requires "
                "factor_exposures to be supplied; pass the same matrix used "
                "for the initial fit"
            )

    def _raise_if_neutralization_settings_mismatch(
        self,
        init_neutralization: str,
        init_lambda: float | None,
        init_penalty: float | None,
        *,
        origin: str,
    ) -> None:
        """Reject warm-start when the init model's neutralization mode (or
        its ``factor_neutralization_lambda`` / ``factor_penalty``) does not
        match the current estimator.

        Resuming training under a different mode silently changes the
        boosting path (e.g. ``per_round_gradient`` projects gradients each
        round while ``pre_target`` residualizes the targets once), so the
        resumed model is not equivalent to a fresh ``N+M``-round fit under
        either configuration.  We therefore require an exact match for
        ``neutralization``, ``factor_neutralization_lambda``, and
        ``factor_penalty``.

        ``origin`` is ``"init_model"`` or ``"warm_start"`` and is used to
        produce a user-facing error message that names the offending
        source.
        """
        if init_neutralization != self.neutralization:
            raise ValueError(
                f"{origin} neutralization '{init_neutralization}' does not "
                f"match current estimator neutralization '{self.neutralization}'; "
                "warm-start requires the same mode so resumed training stays "
                "equivalent to a fresh fit"
            )
        if init_neutralization == "none":
            return
        # Lambda comparison — None matches None; otherwise require numerical
        # equality (these are user-supplied floats, so an exact match is the
        # correct contract).
        if init_lambda != self.factor_neutralization_lambda:
            raise ValueError(
                f"{origin} factor_neutralization_lambda={init_lambda!r} does not "
                f"match current estimator factor_neutralization_lambda="
                f"{self.factor_neutralization_lambda!r}"
            )
        # split_penalty only consumes factor_penalty; other modes ignore it,
        # so the mismatch only matters when both sides are split_penalty.
        if init_neutralization == "split_penalty" and init_penalty != self.factor_penalty:
            raise ValueError(
                f"{origin} factor_penalty={init_penalty!r} does not match "
                f"current estimator factor_penalty={self.factor_penalty!r}"
            )

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


# GBMRegressor is injected into this module's namespace by _core.py after the
# class is defined (see the bottom of _core.py).  Static methods in
# _ValidationMixin reference GBMRegressor by name; those names are resolved at
# call-time against this module's globals, which by then contain the real class.
