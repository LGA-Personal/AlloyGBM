"""Optional scikit-learn integration used by AlloyGBM estimators."""

from __future__ import annotations

try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin

    _SKLEARN_AVAILABLE = True
except ImportError:

    class _BaseEstimator:  # type: ignore[no-redef]
        """Dependency-free base used when scikit-learn is unavailable."""

    class _RegressorMixin:  # type: ignore[no-redef]
        """Dependency-free regressor marker."""

    class _ClassifierMixin:  # type: ignore[no-redef]
        """Dependency-free classifier marker."""

    _SKLEARN_AVAILABLE = False


__all__ = [
    "_BaseEstimator",
    "_ClassifierMixin",
    "_RegressorMixin",
    "_SKLEARN_AVAILABLE",
]
