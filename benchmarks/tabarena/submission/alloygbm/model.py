"""AlloyGBM: deterministic Rust gradient boosting with a sklearn-style API.

Codebase: https://github.com/LGA-Personal/AlloyGBM
License: MIT. No associated research paper is claimed.
"""

from __future__ import annotations

import time
from functools import partial
from typing import ClassVar

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded


def _evaluate_margin(y_true, margins, *, metric, problem_type):
    # AlloyGBM callbacks receive raw margins, whereas AutoGluon scorers expect
    # transformed predictions. AUC can use margins directly without sigmoid ties.
    if metric.name == "roc_auc" and np.unique(y_true).size < 2:
        # The native callback constructor probes with two all-zero dummy rows.
        # Actual AUC validation partitions are checked before training below.
        return metric.name, 0.5, False
    predictions = margins
    if problem_type == "binary" and metric.name != "roc_auc":
        from scipy.special import expit

        predictions = expit(np.asarray(margins, dtype=np.float64))
    return metric.name, float(metric.error(y_true, predictions)), False


def stopping_callback(problem_type, metric):
    """Use the task scorer; multiclass already stops on native softmax log loss."""
    supported = {
        "binary": {"roc_auc", "log_loss"},
        "regression": {"root_mean_squared_error", "rmse"},
        "multiclass": {"log_loss"},
    }
    if metric.name not in supported.get(problem_type, set()):
        raise ValueError(f"unsupported stopping metric: {problem_type}/{metric.name}")
    if problem_type == "multiclass":
        return None
    return partial(_evaluate_margin, metric=metric, problem_type=problem_type)


class AlloyGBMModel(AbstractModel):
    """One model fitted by TabArena's supplied validation and resource protocol."""

    ag_key = "TA-ALLOYGBM"
    ag_name = "TA-AlloyGBM"
    ag_priority = 65
    seed_name = "seed"
    _supported_problem_types: ClassVar[list[str]] = [
        "binary",
        "multiclass",
        "regression",
    ]
    _default_auxiliary_params_extra: ClassVar[dict] = {
        "valid_raw_types": ["int", "float", "category"]
    }
    warmup_modules: ClassVar[tuple[str, ...]] = ("alloygbm",)
    cheap_hyperparameters: ClassVar[dict] = {
        "n_estimators": 2,
        "early_stopping_rounds": None,
        "learning_rate": 0.1,
        "max_depth": 2,
    }

    def _set_default_params(self):
        # The author's requested public-default arm: do not replace auto policy/rounds.
        pass

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs).copy()
        for column in X.select_dtypes(include="number").columns:
            X[column] = X[column].astype(np.float32)
        return X

    def _fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        time_limit=None,
        num_cpus=1,
        num_gpus=0,
        **kwargs,
    ):
        started = time.monotonic()
        from alloygbm import GBMClassifier, GBMRegressor

        if num_gpus:
            raise ValueError("AlloyGBM is CPU only")
        params = self._get_model_params()
        params["n_jobs"] = num_cpus
        automatic = params.get("n_estimators", "auto") == "auto"
        if (automatic or params.get("early_stopping_rounds")) and (
            X_val is None or y_val is None
        ):
            raise ValueError(
                "AlloyGBM requires the supplied validation set; internal holdout is forbidden"
            )
        X = self.preprocess(X, is_train=True)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (self.preprocess(X_val), y_val)
        metric = stopping_callback(self.problem_type, self.stopping_metric)
        if (
            self.stopping_metric.name == "roc_auc"
            and y_val is not None
            and np.unique(y_val).size < 2
        ):
            raise ValueError(
                "ROC AUC requires both classes in the validation partition"
            )
        remaining = None
        if time_limit is not None:
            remaining = (time_limit - (time.monotonic() - started)) * 0.95
            if remaining <= 0:
                raise TimeLimitExceeded
        model_cls = GBMRegressor if self.problem_type == "regression" else GBMClassifier
        self.model = model_cls(**params)
        try:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                time_limit=remaining,
                eval_metric=metric if eval_set is not None else None,
            )
        except TimeoutError as exc:
            self.model = None
            raise TimeLimitExceeded from exc
        if (
            self.problem_type != "regression"
            and len(self.model.classes_) != self.num_classes
        ):
            raise ValueError("training partition is missing an expected target class")
        # Refit is deliberately not advertised; no second automatic holdout is allowed.
        self.params_trained["n_estimators"] = self.model.n_estimators_
        self._alloy_fit_metadata = {
            "parameters": self.model.get_params(),
            "rounds": self.model.n_estimators_,
            "best_iteration": self.model.best_iteration_,
            "stop_reason": self.model.stop_reason_,
            "auto_rounds": getattr(self.model, "auto_rounds_", None),
            "policy": getattr(self.model, "resolved_training_policy_", None),
            "stopping_metric": self.stopping_metric.name,
        }

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            values = np.asarray(self.model.predict(X))
        else:
            values = np.asarray(self.model.predict_proba(X))
            if values.shape != (len(X), self.num_classes):
                raise ValueError(
                    "prediction class dimension differs from the fitted task"
                )
            if not np.all((values >= 0) & (values <= 1)) or not np.allclose(
                values.sum(axis=1), 1, atol=1e-5
            ):
                raise ValueError("invalid class probabilities")
            if self.problem_type == "binary":
                values = values[:, 1]
        if not np.isfinite(values).all():
            raise ValueError("non-finite predictions")
        return values

    def _more_tags(self):
        return {"can_refit_full": False}

    def get_info(self, include_feature_metadata=True):
        info = super().get_info(include_feature_metadata=include_feature_metadata)
        if hasattr(self, "_alloy_fit_metadata"):
            info["alloygbm"] = self._alloy_fit_metadata
        return info
