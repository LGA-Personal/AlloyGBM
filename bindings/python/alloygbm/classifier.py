"""Binary classification estimator for AlloyGBM."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .regressor import GBMRegressor

if TYPE_CHECKING:
    pass

try:
    from sklearn.base import ClassifierMixin

    _SKLEARN_CLASSIFIER_MIXIN = ClassifierMixin
except ImportError:
    _SKLEARN_CLASSIFIER_MIXIN = object  # type: ignore[assignment,misc]


class GBMClassifier(GBMRegressor, _SKLEARN_CLASSIFIER_MIXIN):
    """Gradient Boosted Decision Tree binary classifier with sklearn-compatible API.

    This classifier trains a gradient-boosted model using the binary cross-entropy
    (log-loss) objective. Predictions are probabilities in [0, 1] obtained by
    applying a sigmoid transform to the raw model output.

    Parameters are identical to :class:`GBMRegressor`. The objective is always
    ``binary_crossentropy`` and is not configurable.
    """

    # -- Fitted attributes ---------------------------------------------------
    classes_: list[int]
    """Unique class labels, always ``[0, 1]`` after fit."""

    n_classes_: int
    """Number of classes, always ``2`` after fit."""

    def _objective_name(self) -> str:
        return "binary_crossentropy"

    # -- fit ------------------------------------------------------------------

    def fit(
        self,
        X: object,
        y: object,
        *,
        sample_weight: object | None = None,
        eval_set: tuple[object, object] | None = None,
        eval_sample_weight: object | None = None,
        group: object | None = None,
        eval_group: object | None = None,
        eval_time_index: object | None = None,
        categorical_feature_values: object | None = None,
        time_index: object | None = None,
    ) -> "GBMClassifier":
        """Fit the binary classifier.

        Parameters are identical to :meth:`GBMRegressor.fit`. ``y`` must contain
        only values in {0, 1} (or {0.0, 1.0}).

        Returns
        -------
        self
        """
        # Validate targets are binary {0, 1}
        targets = self._validate_binary_targets(y)

        # Validate eval_set targets if provided
        if eval_set is not None:
            _eval_X, eval_y = eval_set
            self._validate_binary_targets(eval_y)

        # Delegate to GBMRegressor.fit which calls self._objective_name()
        super().fit(
            X,
            targets,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=eval_sample_weight,
            group=group,
            eval_group=eval_group,
            eval_time_index=eval_time_index,
            categorical_feature_values=categorical_feature_values,
            time_index=time_index,
        )
        self.classes_ = [0, 1]
        self.n_classes_ = 2
        return self

    # -- predict (class labels) -----------------------------------------------

    def predict(self, X: object) -> list[int]:
        """Predict class labels for samples in X.

        Returns a list of integers (0 or 1) by thresholding predicted
        probabilities at 0.5.
        """
        p1 = super().predict(X)
        return [1 if p >= 0.5 else 0 for p in p1]

    # -- predict_proba --------------------------------------------------------

    def predict_proba(self, X: object) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Returns an ndarray of shape ``(n_samples, 2)`` with columns
        ``[P(y=0), P(y=1)]``, compatible with the sklearn classifier API.
        """
        # GBMRegressor.predict returns sigmoid-transformed probabilities
        # because the predictor crate applies the post-transform based on the
        # objective stored in the model artifact.
        p1 = np.asarray(super().predict(X), dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])

    # -- predict_log_proba ----------------------------------------------------

    def predict_log_proba(self, X: object) -> np.ndarray:
        """Predict log-probabilities for samples in X.

        Returns an ndarray of shape ``(n_samples, 2)`` with columns
        ``[log(P(y=0)), log(P(y=1))]``.
        """
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-15, None))

    # -- sklearn overrides ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            "GBMClassifier("
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"n_estimators={self.n_estimators}, "
            f"row_subsample={self.row_subsample}, "
            f"col_subsample={self.col_subsample}, "
            f"early_stopping_rounds={self.early_stopping_rounds}, "
            f"min_validation_improvement={self.min_validation_improvement}, "
            f"min_data_in_leaf={self.min_data_in_leaf}, "
            f"lambda_l1={self.lambda_l1}, "
            f"lambda_l2={self.lambda_l2}, "
            f"min_child_hessian={self.min_child_hessian}, "
            f"min_split_gain={self.min_split_gain}, "
            f"seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"continuous_binning_strategy='{self.continuous_binning_strategy}', "
            f"continuous_binning_max_bins={self.continuous_binning_max_bins}, "
            f"categorical_feature_index={self.categorical_feature_index}, "
            f"categorical_feature_indices={self.categorical_feature_indices}, "
            f"training_policy='{self.training_policy}', "
            f"store_node_stats={self.store_node_stats}, "
            f"categorical_smoothing={self.categorical_smoothing}, "
            f"categorical_min_samples_leaf={self.categorical_min_samples_leaf}, "
            f"categorical_time_aware={self.categorical_time_aware}, "
            f"monotone_constraints={self.monotone_constraints}, "
            f"feature_weights={self.feature_weights}, "
            f"max_leaves={self.max_leaves}, "
            f"tree_growth='{self.tree_growth}', "
            f"warm_start={self.warm_start}"
            ")"
        )

    def __sklearn_tags__(self):
        if not hasattr(super(GBMRegressor, self), "__sklearn_tags__"):
            return {
                "non_deterministic": not self.deterministic,
                "requires_y": True,
                "allow_nan": True,
                "X_types": ["2darray"],
                "binary_only": True,
            }
        tags = super().__sklearn_tags__()
        if hasattr(tags, "non_deterministic"):
            tags.non_deterministic = not self.deterministic
        if hasattr(tags, "input_tags") and hasattr(tags.input_tags, "allow_nan"):
            tags.input_tags.allow_nan = True
        if hasattr(tags, "classifier_tags"):
            tags.classifier_tags.multi_output = False
        return tags

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True, "binary_only": True}

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _validate_binary_targets(y: object) -> list[float]:
        """Validate and convert targets to float list with values in {0.0, 1.0}."""
        try:
            import numpy as _np

            if isinstance(y, _np.ndarray):
                targets = y.ravel().tolist()
            elif hasattr(y, "__iter__"):
                targets = list(y)
            else:
                raise TypeError(
                    f"y must be array-like, got {type(y).__name__}"
                )
        except ImportError:
            if hasattr(y, "__iter__"):
                targets = list(y)
            else:
                raise TypeError(
                    f"y must be array-like, got {type(y).__name__}"
                )

        if len(targets) == 0:
            raise ValueError("y must not be empty")

        float_targets: list[float] = []
        for i, val in enumerate(targets):
            fval = float(val)
            if fval != 0.0 and fval != 1.0:
                raise ValueError(
                    f"GBMClassifier requires binary targets in {{0, 1}}, "
                    f"but found value {val!r} at index {i}"
                )
            float_targets.append(fval)

        unique = set(float_targets)
        if unique == {0.0} or unique == {1.0}:
            raise ValueError(
                f"GBMClassifier requires both classes present in y, "
                f"but found only class {int(next(iter(unique)))}"
            )

        return float_targets
