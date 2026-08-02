"""Optional scikit-learn integration used by AlloyGBM estimators."""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.exceptions import NotFittedError as _NotFittedError
    from sklearn.utils.validation import check_is_fitted as _check_is_fitted

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

    _SKLEARN_AVAILABLE = False


__all__ = [
    "_BaseEstimator",
    "_ClassifierMixin",
    "_NotFittedError",
    "_RegressorMixin",
    "_SKLEARN_AVAILABLE",
    "_check_is_fitted",
]
