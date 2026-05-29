"""SHAP explanation methods mixin for GBMRegressor."""

from __future__ import annotations

from . import _base
from ._base import _max_data_bin_for_max_bins


class _ShapMixin:
    """Mixin carrying SHAP explanation and feature importance methods for GBMRegressor.

    All 4 methods are moved verbatim from GBMRegressor in _core.py.
    No ``GBMRegressor`` class-name references exist in these method bodies,
    so no post-definition injection is required.
    """

    def _shap_binning_kwargs(self) -> dict | None:
        """Return kwargs for the predictor-aligned SHAP entry points,
        or `None` when SHAP should fall back to the legacy
        bin-index-quantized path.

        Returned when continuous binning was applied with a strategy
        whose bin → float-threshold conversion has a SHAP equivalent:

        - `linear` (no per-feature rank flags): use feature mins/maxs +
          max_data_bin.
        - `linear` with at least one per-feature rank flag set: use the
          mixed-mode `linear_rank` context (per-feature sorted unique
          values for rank-flagged columns, fall back to linear binning
          for the rest).  v0.8.0 closed Limitation 4 by wiring this
          through.
        - `quantile`: use feature quantile cuts.

        Pre-binned artifacts use the legacy path because the bin-index
        comparison is already correct for integer data.
        """
        if not self._is_fitted or not self._uses_continuous_binning:
            return None
        strategy = self.continuous_binning_strategy
        if strategy == "linear":
            mins = self._continuous_feature_mins
            maxs = self._continuous_feature_maxs
            if mins is None or maxs is None:
                return None
            rank_flags = self._continuous_feature_linear_rank_flags
            if rank_flags is not None and any(rank_flags):
                sorted_values = self._continuous_feature_sorted_values
                if sorted_values is None:
                    # Defensive: rank flags fired but sorted values were
                    # not persisted.  Fall back to the legacy path
                    # rather than mis-route SHAP to an inconsistent
                    # context.
                    return None
                per_feature: list[list[float] | None] = []
                for flag, column in zip(rank_flags, sorted_values):
                    if flag and column is not None:
                        per_feature.append(list(column))
                    else:
                        per_feature.append(None)
                return {
                    "binning_kind": "linear_rank",
                    "feature_mins": list(mins),
                    "feature_maxs": list(maxs),
                    "max_data_bin": _max_data_bin_for_max_bins(
                        self.continuous_binning_max_bins
                    ),
                    "linear_rank_per_feature": per_feature,
                }
            return {
                "binning_kind": "linear",
                "feature_mins": list(mins),
                "feature_maxs": list(maxs),
                "max_data_bin": _max_data_bin_for_max_bins(
                    self.continuous_binning_max_bins
                ),
            }
        if strategy == "quantile":
            cuts = self._continuous_feature_quantile_cuts
            if cuts is None:
                return None
            return {
                "binning_kind": "quantile",
                "feature_cuts": [list(c) for c in cuts],
            }
        return None

    def shap_values(
        self, X: object, *, include_expected_value: bool = False
    ) -> list[list[float]] | tuple[float, list[list[float]]]:
        """Return SHAP values for the provided rows using the fitted artifact.

        For ``leaf_model="linear"`` (piecewise-linear) artifacts this is an
        *interventional* decomposition: the path-based attribution acts on
        each leaf's "constant part" ``intercept + Σ wj·μj_global`` and
        per-feature deviations ``wj · (xj − μj_global)`` are credited
        directly to each regressor at every visited node along the row's
        path through each tree.

        As of v0.7.4, ``Σ shap_values + expected_value == predict(x)``
        holds within ``atol + rtol · |predict(x)|`` (default
        ``atol=1e-5, rtol=1e-4``) for every leaf model on the default
        predictor-aligned binning path.  v0.8.0 extends this guarantee
        to the mixed linear-rank binning path
        (``continuous_binning_strategy="linear"`` combined with
        per-feature rank-based binning on at least one column) via the
        new ``BinningContext::LinearRank`` variant — see
        ``docs/limitations.md`` "Resolved" entry for v0.8.0.
        The legacy non-binning path retains the best-effort exemption
        for linear leaves only.
        """
        if not self._is_fitted:
            raise RuntimeError("GBMRegressor must be fit before shap_values")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")

        # v0.7.3+: when we have a `BinningContext` available (continuous
        # features under linear, quantile, or — as of v0.8.0 — mixed
        # linear-rank binning), pass raw rows + binning kwargs so the
        # native SHAP path-walker matches the predictor.  Pre-binned
        # integer data and unbinned float artifacts fall back to the
        # legacy quantize-then-walk path.
        binning_kwargs = self._shap_binning_kwargs()
        if binning_kwargs is not None:
            rows = self._native_matrix_flat_payload(X) or self._validate_rows(X)
            row_feature_count = (
                rows[2] if isinstance(rows, tuple) else len(rows[0])
            )
            if row_feature_count != self._n_features_in:
                raise ValueError(
                    f"X feature count {row_feature_count} does not match fitted feature count "
                    f"{self._n_features_in}"
                )
        elif self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(self._validate_rows(X))
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

        if binning_kwargs is not None:
            if isinstance(rows, tuple):
                flat_values, row_count, feature_count = rows
                shap_fn = _base._load_native_shap_explain_rows_dense_with_binning()
                expected_value, values = shap_fn(
                    self._artifact_bytes,
                    flat_values,
                    row_count,
                    feature_count,
                    **binning_kwargs,
                )
            else:
                shap_fn = _base._load_native_shap_explain_rows_with_binning()
                expected_value, values = shap_fn(
                    self._artifact_bytes, rows, **binning_kwargs
                )
        elif isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            shap_explain_rows_dense = _base._load_native_shap_explain_rows_dense()
            expected_value, values = shap_explain_rows_dense(
                self._artifact_bytes,
                flat_values,
                row_count=row_count,
                feature_count=feature_count,
            )
        else:
            shap_explain_rows = _base._load_native_shap_explain_rows()
            expected_value, values = shap_explain_rows(self._artifact_bytes, rows)
        shap_matrix = [list(row) for row in values]
        if include_expected_value:
            return float(expected_value), shap_matrix
        return shap_matrix

    def shap_interaction_values(
        self, X: object, *, include_expected_value: bool = False
    ):
        """Return pairwise SHAP interaction values for the provided rows.

        Implements Lundberg et al. (2020) Algorithm 2 in polynomial time
        (``O(T · L · D² · M)``).  Returns a 3-D structure
        ``values[row][i][j]`` such that:

        - ``values[row][i][j] == values[row][j][i]`` (symmetric).
        - ``Σ_j values[row][i][j] == shap_values(X)[row][i]`` (row marginal).
        - ``Σ_i Σ_j values[row][i][j] + expected_value == predict(x)``
          (full additivity, mod f32 round-off).

        Linear-leaf (``leaf_model="linear"``) artifacts are rejected in
        v0.12.3 — pairwise interactions on PL leaves require a different
        decomposition that is not yet implemented.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "GBMRegressor must be fit before shap_interaction_values"
            )
        if self._artifact_bytes is None:
            raise RuntimeError("GBMRegressor native artifact is not available")

        binning_kwargs = self._shap_binning_kwargs()
        if binning_kwargs is not None:
            rows = self._native_matrix_flat_payload(X) or self._validate_rows(X)
            row_feature_count = (
                rows[2] if isinstance(rows, tuple) else len(rows[0])
            )
            if row_feature_count != self._n_features_in:
                raise ValueError(
                    f"X feature count {row_feature_count} does not match fitted feature count "
                    f"{self._n_features_in}"
                )
        elif self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(self._validate_rows(X))
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

        if binning_kwargs is not None:
            from alloygbm._alloygbm import (
                shap_explain_interactions_dense_with_binning,
                shap_explain_interactions_with_binning,
            )

            if isinstance(rows, tuple):
                flat, row_count, feature_count = rows
                expected_value, values = shap_explain_interactions_dense_with_binning(
                    self._artifact_bytes,
                    flat,
                    row_count,
                    feature_count,
                    **binning_kwargs,
                )
            else:
                expected_value, values = shap_explain_interactions_with_binning(
                    self._artifact_bytes, rows, **binning_kwargs
                )
        else:
            from alloygbm._alloygbm import (
                shap_explain_interactions,
                shap_explain_interactions_dense,
            )

            if isinstance(rows, tuple):
                flat, row_count, feature_count = rows
                expected_value, values = shap_explain_interactions_dense(
                    self._artifact_bytes, flat, row_count, feature_count
                )
            else:
                expected_value, values = shap_explain_interactions(
                    self._artifact_bytes, rows
                )

        matrix = [[list(col) for col in row] for row in values]
        if include_expected_value:
            return float(expected_value), matrix
        return matrix

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

        binning_kwargs = self._shap_binning_kwargs()
        if binning_kwargs is not None:
            rows = self._native_matrix_flat_payload(X) or self._validate_rows(X)
            row_feature_count = (
                rows[2] if isinstance(rows, tuple) else len(rows[0])
            )
            if row_feature_count != self._n_features_in:
                raise ValueError(
                    f"X feature count {row_feature_count} does not match fitted feature count "
                    f"{self._n_features_in}"
                )
        elif self._uses_continuous_binning:
            rows = self._quantize_rows_for_prediction(self._validate_rows(X))
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

        if binning_kwargs is not None:
            if isinstance(rows, tuple):
                flat_values, row_count, feature_count = rows
                shap_fn = _base._load_native_shap_global_importance_dense_with_binning()
                importance = shap_fn(
                    self._artifact_bytes,
                    flat_values,
                    row_count,
                    feature_count,
                    **binning_kwargs,
                )
            else:
                shap_fn = _base._load_native_shap_global_importance_with_binning()
                importance = shap_fn(self._artifact_bytes, rows, **binning_kwargs)
        elif isinstance(rows, tuple):
            flat_values, row_count, feature_count = rows
            shap_global_importance_dense = _base._load_native_shap_global_importance_dense()
            importance = shap_global_importance_dense(
                self._artifact_bytes,
                flat_values,
                row_count=row_count,
                feature_count=feature_count,
            )
        else:
            shap_global_importance = _base._load_native_shap_global_importance()
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
