"""Binary and multi-class classification estimator for AlloyGBM."""

from __future__ import annotations

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
    """Gradient Boosted Decision Tree classifier with sklearn-compatible API.

    Supports binary classification (log-loss) and multi-class classification
    (softmax cross-entropy). The number of classes is auto-detected from the
    training labels:

    - **2 classes with labels {0, 1}**: binary cross-entropy (single tree per round)
    - **K > 2 classes**: softmax / multinomial cross-entropy (K trees per round)

    Predictions are probabilities obtained by applying sigmoid (binary) or
    softmax (multi-class) transforms to the raw model outputs.

    Parameters are identical to :class:`GBMRegressor`. The objective is
    auto-detected and is not configurable.
    """

    # -- Fitted attributes ---------------------------------------------------
    classes_: list[int]
    """Unique class labels (sorted) after fit."""

    n_classes_: int
    """Number of classes after fit."""

    # -- Private multi-class state -------------------------------------------
    _label_encoder: dict[int, int] | None
    """Maps original label -> 0..K-1 index. None for native {0,1} binary."""

    _label_decoder: dict[int, int] | None
    """Maps 0..K-1 index -> original label. None for native {0,1} binary."""

    _num_classes_for_training: int | None
    """Passed to Rust bridge for multiclass_softmax. None for binary."""

    def _objective_name(self) -> str:
        # Custom callable objective takes priority over auto-detection.
        if getattr(self, 'objective', None) is not None:
            if callable(self.objective):
                return "custom"
            return str(self.objective)
        num_classes = getattr(self, '_num_classes_for_training', None)
        if num_classes is not None and num_classes > 2:
            return "multiclass_softmax"
        return "binary_crossentropy"

    @property
    def _is_multiclass(self) -> bool:
        return getattr(self, '_num_classes_for_training', None) is not None and self._num_classes_for_training > 2

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
        categorical_feature_values_list: object | None = None,
        time_index: object | None = None,
        init_model: "GBMRegressor | None" = None,
        eval_metric: object | None = None,
        factor_exposures: object | None = None,
    ) -> "GBMClassifier":
        """Fit the classifier.

        Parameters are identical to :meth:`GBMRegressor.fit`. ``y`` must contain
        integer class labels. For binary classification, labels must be in {0, 1}.
        For multi-class, labels can be any set of integers with K >= 3 unique values.

        Parameters
        ----------
        eval_metric : callable or None, optional
            Custom evaluation metric. See :meth:`GBMRegressor.fit` for details.

        Returns
        -------
        self
        """
        if self.neutralization == "pre_target":
            raise ValueError(
                "neutralization='pre_target' is only supported for GBMRegressor "
                "squared-error training"
            )

        # Encode targets
        float_targets, sorted_classes, n_classes, label_map = (
            self._encode_classification_targets(y)
        )

        # Store label encoding state
        if n_classes == 2 and sorted_classes == [0, 1]:
            # Native binary path — no encoding needed
            self._label_encoder = None
            self._label_decoder = None
            self._num_classes_for_training = None
        else:
            self._label_encoder = label_map
            self._label_decoder = {v: k for k, v in label_map.items()}
            self._num_classes_for_training = n_classes

        # Reject custom callable objectives with multiclass labels — training produces
        # a single-output model that is incompatible with multiclass prediction routing.
        if self._is_multiclass and callable(getattr(self, 'objective', None)):
            raise ValueError(
                "GBMClassifier does not support custom callable objectives with "
                f"multiclass labels (detected {self._num_classes_for_training} classes). "
                "Custom objectives produce single-output models incompatible with "
                "multiclass prediction. Use binary classification or a built-in objective."
            )

        # Validate eval_set targets if provided
        if eval_set is not None:
            _eval_X, eval_y = eval_set
            eval_float_targets, _, eval_n, _ = self._encode_classification_targets(
                eval_y, label_map=label_map
            )
            eval_set = (_eval_X, eval_float_targets)

        # Delegate to GBMRegressor.fit which calls self._objective_name()
        super().fit(
            X,
            float_targets,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_sample_weight=eval_sample_weight,
            group=group,
            eval_group=eval_group,
            eval_time_index=eval_time_index,
            categorical_feature_values=categorical_feature_values,
            categorical_feature_values_list=categorical_feature_values_list,
            time_index=time_index,
            init_model=init_model,
            eval_metric=eval_metric,
            factor_exposures=factor_exposures,
        )
        self.classes_ = sorted_classes
        self.n_classes_ = n_classes
        return self

    # -- predict (class labels) -----------------------------------------------

    def predict(self, X: object) -> list[int]:
        """Predict class labels for samples in X.

        For binary: thresholds predicted probabilities at 0.5.
        For multi-class: returns argmax of predicted probabilities.
        """
        if self._is_multiclass:
            proba = self.predict_proba(X)
            indices = np.argmax(proba, axis=1)
            decoder = self._label_decoder
            if decoder is not None:
                return [decoder[int(i)] for i in indices]
            return [int(i) for i in indices]
        else:
            p1 = super().predict(X)
            raw = [1 if p >= 0.5 else 0 for p in p1]
            if self._label_decoder is not None:
                return [self._label_decoder[v] for v in raw]
            return raw

    # -- predict_proba --------------------------------------------------------

    def predict_proba(self, X: object) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Returns an ndarray of shape ``(n_samples, n_classes)`` with columns
        ordered by ``self.classes_``.
        """
        if self._is_multiclass:
            return self._predict_proba_multiclass(X)
        # Binary path
        p1 = np.asarray(super().predict(X), dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])

    def _predict_proba_multiclass(self, X: object) -> np.ndarray:
        """Multi-class prediction using native multi-class predictor."""
        if not self._is_fitted:
            raise RuntimeError("GBMClassifier must be fit before predict")
        if self._artifact_bytes is None:
            raise RuntimeError("GBMClassifier native artifact is not available")

        # Apply native categorical mappings (string -> float ID) before prediction
        X = self._apply_native_cat_mappings_for_predict(X)

        # Lazily reconstruct the native predictor after pickle roundtrip
        if self._native_predictor_handle is None and getattr(
            self, "_predictor_needs_rebuild", False
        ):
            self._predictor_needs_rebuild = False
            self._native_predictor_handle = self._build_native_predictor_handle(
                self._artifact_bytes
            )
            self._convert_predictor_thresholds_to_float()

        k = self.n_classes_
        handle = self._native_predictor_handle

        # Try numpy fast path first
        try:
            candidate = self._native_matrix_fast_path_candidate(X)
            if candidate is not None:
                arr = np.ascontiguousarray(candidate, dtype=np.float32)
                if arr.shape[1] != self._n_features_in:
                    raise ValueError(
                        f"X feature count {arr.shape[1]} does not match fitted "
                        f"feature count {self._n_features_in}"
                    )
                predict_fn = getattr(handle, "predict_numpy_multiclass", None)
                if callable(predict_fn):
                    flat = predict_fn(arr)
                    return np.asarray(flat, dtype=np.float64).reshape(-1, k)
        except (ImportError, AttributeError):
            pass

        # Dense path
        dense_payload = self._native_matrix_flat_payload(X)
        if dense_payload is not None:
            flat_values, row_count, feature_count = dense_payload
            if feature_count != self._n_features_in:
                raise ValueError(
                    f"X feature count {feature_count} does not match fitted "
                    f"feature count {self._n_features_in}"
                )
            predict_fn = getattr(handle, "predict_dense_multiclass", None)
            if callable(predict_fn):
                flat = predict_fn(flat_values, row_count, feature_count)
                return np.asarray(flat, dtype=np.float64).reshape(-1, k)

        # Rows path (fallback)
        rows = self._validate_rows(X)
        predict_fn = getattr(handle, "predict_multiclass", None)
        if callable(predict_fn):
            flat = predict_fn(rows)
            return np.asarray(flat, dtype=np.float64).reshape(-1, k)

        raise RuntimeError(
            "Native predictor does not support multi-class prediction"
        )

    # -- predict_log_proba ----------------------------------------------------

    def predict_log_proba(self, X: object) -> np.ndarray:
        """Predict log-probabilities for samples in X.

        Returns an ndarray of shape ``(n_samples, n_classes)`` with columns
        ``[log(P(y=class_0)), ..., log(P(y=class_K-1))]``.
        """
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-15, None))

    # -- score (accuracy, not R^2) ---------------------------------------------

    def score(self, X: object, y: object, sample_weight: object = None) -> float:
        """Return accuracy score for the given test data and labels.

        Overrides ``GBMRegressor.score()`` to use accuracy (the standard
        sklearn classifier convention) instead of R-squared.
        """
        from .evaluation import accuracy

        predictions = self.predict(X)
        _, _, _, label_map = self._encode_classification_targets(y, label_map=self._label_encoder)
        # Convert y to encoded labels for comparison
        if self._label_encoder is not None:
            y_encoded = [self._label_encoder[int(v)] for v in self._to_list(y)]
            pred_encoded = [self._label_encoder.get(p, p) for p in predictions]
        else:
            y_encoded = [int(v) for v in self._to_list(y)]
            pred_encoded = predictions
        return float(accuracy(y_encoded, pred_encoded))

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
            f"warm_start={self.warm_start}, "
            f"max_cat_threshold={self.max_cat_threshold}, "
            f"training_mode='{self.training_mode}', "
            f"morph_rate={self.morph_rate}, "
            f"evolution_pressure={self.evolution_pressure}, "
            f"morph_warmup_iters={self.morph_warmup_iters}, "
            f"lr_schedule='{self.lr_schedule}', "
            f"lr_warmup_frac={self.lr_warmup_frac}, "
            f"leaf_model='{self.leaf_model}', "
            f"leaf_solver='{self.leaf_solver}', "
            f"dro_radius={self.dro_radius}, "
            f"dro_metric='{self.dro_metric}'"
            ")"
        )

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            return {
                "non_deterministic": not self.deterministic,
                "requires_y": True,
                "allow_nan": True,
                "X_types": ["2darray"],
            }
        if hasattr(tags, "non_deterministic"):
            tags.non_deterministic = not self.deterministic
        if hasattr(tags, "input_tags") and hasattr(tags.input_tags, "allow_nan"):
            tags.input_tags.allow_nan = True
        if hasattr(tags, "classifier_tags"):
            tags.classifier_tags.multi_output = False
        return tags

    def _more_tags(self):
        return {"allow_nan": True, "requires_y": True}

    # -- serialization --------------------------------------------------------

    def __getstate__(self):
        state = super().__getstate__()
        state["_classifier_classes"] = getattr(self, "classes_", None)
        state["_classifier_n_classes"] = getattr(self, "n_classes_", None)
        state["_classifier_label_encoder"] = getattr(self, "_label_encoder", None)
        state["_classifier_label_decoder"] = getattr(self, "_label_decoder", None)
        state["_classifier_num_classes_for_training"] = getattr(
            self, "_num_classes_for_training", None
        )
        return state

    def __setstate__(self, state):
        classes = state.pop("_classifier_classes", None)
        n_classes = state.pop("_classifier_n_classes", None)
        label_encoder = state.pop("_classifier_label_encoder", None)
        label_decoder = state.pop("_classifier_label_decoder", None)
        num_classes_for_training = state.pop("_classifier_num_classes_for_training", None)
        super().__setstate__(state)
        if classes is not None:
            self.classes_ = classes
        if n_classes is not None:
            self.n_classes_ = n_classes
        self._label_encoder = label_encoder
        self._label_decoder = label_decoder
        self._num_classes_for_training = num_classes_for_training

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _to_list(y: object) -> list:
        """Convert array-like to a flat Python list."""
        if isinstance(y, np.ndarray):
            return y.ravel().tolist()
        if hasattr(y, "__iter__"):
            return list(y)
        raise TypeError(f"y must be array-like, got {type(y).__name__}")

    @staticmethod
    def _encode_classification_targets(
        y: object,
        *,
        label_map: dict[int, int] | None = None,
    ) -> tuple[list[float], list[int], int, dict[int, int]]:
        """Encode classification targets to contiguous 0..K-1 float values.

        Parameters
        ----------
        y : array-like
            Target labels (integers).
        label_map : dict or None
            If provided, use this existing mapping. Otherwise, build one from y.

        Returns
        -------
        float_targets : list[float]
            Encoded targets as floats in 0..K-1.
        sorted_classes : list[int]
            Sorted unique class labels from the original data.
        n_classes : int
            Number of unique classes.
        label_map : dict[int, int]
            Mapping from original label to 0..K-1 index.
        """
        try:
            if isinstance(y, np.ndarray):
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

        # Convert to integers
        int_targets: list[int] = []
        for i, val in enumerate(targets):
            fval = float(val)
            rounded = round(fval)
            if abs(fval - rounded) > 1e-6:
                raise ValueError(
                    f"GBMClassifier requires integer class labels, "
                    f"but found value {val!r} at index {i}"
                )
            int_targets.append(int(rounded))

        if label_map is None:
            # Build encoding from data
            sorted_classes = sorted(set(int_targets))
            n_classes = len(sorted_classes)
            if n_classes < 2:
                raise ValueError(
                    f"GBMClassifier requires at least 2 classes in y, "
                    f"but found only {n_classes} class(es)"
                )
            label_map = {cls: idx for idx, cls in enumerate(sorted_classes)}
        else:
            sorted_classes = sorted(label_map.keys())
            n_classes = len(sorted_classes)

        # Encode targets
        float_targets: list[float] = []
        for i, val in enumerate(int_targets):
            if val not in label_map:
                raise ValueError(
                    f"Label {val} at index {i} not in known classes {sorted_classes}"
                )
            float_targets.append(float(label_map[val]))

        return float_targets, sorted_classes, n_classes, label_map
