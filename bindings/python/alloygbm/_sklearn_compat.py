"""Optional scikit-learn integration used by AlloyGBM estimators."""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.exceptions import NotFittedError as _NotFittedError
    from sklearn.utils.validation import check_array as _check_array
    from sklearn.utils.validation import check_is_fitted as _check_is_fitted
    from sklearn.utils.validation import column_or_1d as _column_or_1d
    from sklearn.utils.multiclass import type_of_target as _type_of_target

    _SKLEARN_AVAILABLE = True
except ImportError:

    class _BaseEstimator:  # type: ignore[no-redef]
        """Dependency-free base used when scikit-learn is unavailable."""

    class _RegressorMixin:  # type: ignore[no-redef]
        """Dependency-free regressor marker."""

    class _ClassifierMixin:  # type: ignore[no-redef]
        """Dependency-free classifier marker."""

    class _NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        """Fallback matching sklearn's unfitted-estimator exception shape."""

    def _check_is_fitted(estimator: object) -> None:  # type: ignore[no-redef]
        fitted = getattr(estimator, "__sklearn_is_fitted__", None)
        if callable(fitted) and fitted():
            return
        raise _NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. Call "
            "'fit' with appropriate arguments before using this estimator."
        )

    def _check_array(  # type: ignore[no-redef]
        value: object,
        *,
        accept_sparse: bool = False,
        dtype: object = "numeric",
        ensure_all_finite: bool | str = True,
        ensure_2d: bool = True,
        ensure_min_samples: int = 1,
        ensure_min_features: int = 1,
        **_: object,
    ):
        import numpy as np

        if hasattr(value, "tocsr") or "sparse" in type(value).__module__.lower():
            if accept_sparse:
                return value
            raise TypeError("Sparse input is not supported; provide a dense array")
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError("Complex data is not supported")
        if ensure_2d and array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")
        if array.ndim >= 1 and array.shape[0] < ensure_min_samples:
            raise ValueError(f"Found array with shape={array.shape}; a minimum of 1 is required")
        if ensure_2d and array.shape[1] < ensure_min_features:
            raise ValueError(f"Found array with shape={array.shape}; a minimum of 1 is required")
        numeric = np.asarray(array, dtype=np.float64 if dtype == "numeric" else dtype)
        if ensure_all_finite == "allow-nan":
            if np.isinf(numeric).any():
                raise ValueError("Input contains infinity")
        elif ensure_all_finite and not np.isfinite(numeric).all():
            raise ValueError("Input contains NaN or infinity")
        return numeric

    def _column_or_1d(value: object, *, warn: bool = False):  # type: ignore[no-redef]
        import warnings
        import numpy as np

        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] == 1:
            if warn:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was expected.",
                    UserWarning,
                    stacklevel=2,
                )
            return array.reshape(-1)
        if array.ndim != 1:
            raise ValueError(f"y should be a 1d array, got an array of shape {array.shape} instead")
        return array

    def _type_of_target(  # type: ignore[no-redef]
        value: object,
        *,
        input_name: str = "",
        raise_unknown: bool = False,
    ) -> str:
        import numpy as np

        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError("Complex data not supported")
        if array.ndim != 1:
            if raise_unknown:
                raise ValueError(f"Unknown label type for {input_name or 'target'}")
            return "unknown"
        if np.issubdtype(array.dtype, np.number):
            numeric = np.asarray(array, dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("Input contains NaN or infinity")
            if np.issubdtype(array.dtype, np.floating) and not np.equal(
                numeric, np.floor(numeric)
            ).all():
                return "continuous"
        return "binary" if len(set(array.tolist())) <= 2 else "multiclass"

    _SKLEARN_AVAILABLE = False


__all__ = [
    "_BaseEstimator",
    "_check_array",
    "_check_is_fitted",
    "_ClassifierMixin",
    "_column_or_1d",
    "_NotFittedError",
    "_RegressorMixin",
    "_SKLEARN_AVAILABLE",
    "_type_of_target",
]
