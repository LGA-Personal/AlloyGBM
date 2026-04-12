"""Learning-to-rank estimator for AlloyGBM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .regressor import GBMRegressor

if TYPE_CHECKING:
    pass

_RANKING_OBJECTIVES = frozenset({
    "rank:pairwise",
    "rank:ndcg",
    "rank:xendcg",
    "queryrmse",
    "yetirank",
})

_OBJECTIVE_NAME_MAP = {
    "rank:pairwise": "rank_pairwise",
    "rank:ndcg": "rank_ndcg",
    "rank:xendcg": "rank_xendcg",
    "queryrmse": "queryrmse",
    "yetirank": "yetirank",
}


class GBMRanker(GBMRegressor):
    """Gradient Boosted Decision Tree learning-to-rank estimator.

    Trains a ranking model using one of several learning-to-rank objectives.
    All ranking objectives require ``group`` to be provided in :meth:`fit`.

    Parameters
    ----------
    ranking_objective : str, default ``"rank:ndcg"``
        The ranking objective function. Supported values:

        * ``"rank:pairwise"`` -- Pairwise logistic (RankNet)
        * ``"rank:ndcg"`` -- LambdaMART with NDCG weighting
        * ``"rank:xendcg"`` -- Cross-entropy approximation to NDCG
        * ``"queryrmse"`` -- Query-grouped RMSE
        * ``"yetirank"`` -- YetiRank (stochastic NDCG-weighted pairwise)

    Other parameters are identical to :class:`GBMRegressor`.
    """

    def __init__(self, *, ranking_objective: str = "rank:ndcg", **kwargs: object) -> None:
        if ranking_objective not in _RANKING_OBJECTIVES:
            raise ValueError(
                f"ranking_objective must be one of {sorted(_RANKING_OBJECTIVES)}, "
                f"got {ranking_objective!r}"
            )
        super().__init__(**kwargs)
        self.ranking_objective = ranking_objective

    def _objective_name(self) -> str:
        # Custom callable objective takes priority.
        if getattr(self, 'objective', None) is not None:
            if callable(self.objective):
                return "custom"
            return str(self.objective)
        return _OBJECTIVE_NAME_MAP[self.ranking_objective]

    # -- fit ------------------------------------------------------------------

    def fit(
        self,
        X: object,
        y: object,
        *,
        group: object,
        sample_weight: object | None = None,
        eval_set: tuple[object, object] | None = None,
        eval_sample_weight: object | None = None,
        eval_group: object | None = None,
        eval_time_index: object | None = None,
        categorical_feature_values: object | None = None,
        categorical_feature_values_list: object | None = None,
        time_index: object | None = None,
        init_model: "GBMRegressor | None" = None,
        eval_metric: object | None = None,
    ) -> "GBMRanker":
        """Fit the ranker.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Relevance labels (higher = more relevant). Can be graded
            (e.g. 0, 1, 2, 3, 4) or binary.
        group : array-like of shape (n_samples,)
            Query group identifier for each row. All rows with the same
            group ID belong to the same query. Data is sorted by group
            internally.
        eval_set : tuple of (X_val, y_val) or None
            Validation data for early stopping.
        eval_group : array-like or None
            Query group IDs for the validation set. Required when
            ``eval_set`` is provided.
        eval_metric : callable or None, optional
            Custom evaluation metric. See :meth:`GBMRegressor.fit` for details.
        """
        if group is None:
            raise ValueError("GBMRanker requires 'group' to be provided in fit()")

        # Sort training data by group, preserving DataFrame metadata.
        X_sorted, y_sorted, group_sorted, sort_idx = self._sort_by_group(X, y, group)

        # Sort sample_weight to match if present.
        sorted_sample_weight = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=np.float32).ravel()
            sorted_sample_weight = sw_arr[sort_idx]

        # Sort time_index to match if present.
        sorted_time_index = None
        if time_index is not None:
            ti_arr = np.asarray(time_index).ravel()
            sorted_time_index = ti_arr[sort_idx]

        # Sort categorical_feature_values to match if present.
        sorted_categorical_feature_values = categorical_feature_values
        if categorical_feature_values is not None:
            cfv_arr = np.asarray(categorical_feature_values)
            sorted_categorical_feature_values = cfv_arr[sort_idx].tolist()

        # Sort categorical_feature_values_list to match if present.
        sorted_categorical_feature_values_list = categorical_feature_values_list
        if categorical_feature_values_list is not None:
            sorted_cfv_list = []
            for cfv in categorical_feature_values_list:
                cfv_arr = np.asarray(cfv)
                sorted_cfv_list.append(cfv_arr[sort_idx].tolist())
            sorted_categorical_feature_values_list = sorted_cfv_list

        # Sort eval data by group if provided.
        sorted_eval_set = eval_set
        sorted_eval_group = eval_group
        sorted_eval_sample_weight = eval_sample_weight
        sorted_eval_time_index = eval_time_index

        if eval_set is not None:
            if eval_group is None:
                raise ValueError(
                    "eval_group must be provided when eval_set is used with GBMRanker"
                )
            eval_X, eval_y = eval_set
            eval_X_sorted, eval_y_sorted, eval_group_sorted, eval_sort_idx = (
                self._sort_by_group(eval_X, eval_y, eval_group)
            )
            sorted_eval_set = (eval_X_sorted, eval_y_sorted)
            sorted_eval_group = eval_group_sorted

            if eval_sample_weight is not None:
                esw_arr = np.asarray(eval_sample_weight, dtype=np.float32).ravel()
                sorted_eval_sample_weight = esw_arr[eval_sort_idx]

            if eval_time_index is not None:
                eti_arr = np.asarray(eval_time_index).ravel()
                sorted_eval_time_index = eti_arr[eval_sort_idx]

        # Delegate to GBMRegressor.fit which uses self._objective_name().
        super().fit(
            X_sorted,
            y_sorted,
            sample_weight=sorted_sample_weight,
            eval_set=sorted_eval_set,
            eval_sample_weight=sorted_eval_sample_weight,
            group=group_sorted,
            eval_group=sorted_eval_group,
            eval_time_index=sorted_eval_time_index,
            categorical_feature_values=sorted_categorical_feature_values,
            categorical_feature_values_list=sorted_categorical_feature_values_list,
            time_index=sorted_time_index,
            init_model=init_model,
            eval_metric=eval_metric,
        )
        return self

    # -- predict (relevance scores) -------------------------------------------

    def predict(self, X: object) -> list[float]:
        """Predict relevance scores for samples in X.

        Returns raw model scores (higher = more relevant). No post-transform
        is applied for ranking objectives.
        """
        return super().predict(X)

    # -- sklearn overrides ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            "GBMRanker("
            f"ranking_objective='{self.ranking_objective}', "
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
            f"objective={self.objective!r}"
            ")"
        )

    def get_params(self, deep: bool = True) -> dict:
        params = super().get_params(deep=deep)
        params["ranking_objective"] = self.ranking_objective
        return params

    def set_params(self, **params: object) -> "GBMRanker":
        if "ranking_objective" in params:
            val = params.pop("ranking_objective")
            if val not in _RANKING_OBJECTIVES:
                raise ValueError(
                    f"ranking_objective must be one of {sorted(_RANKING_OBJECTIVES)}, "
                    f"got {val!r}"
                )
            self.ranking_objective = val
        super().set_params(**params)
        return self

    # -- internal helpers ------------------------------------------------------

    @staticmethod
    def _sort_by_group(
        X: object, y: object, group: object
    ) -> tuple:
        """Sort X, y, and group by group ID. Returns (X, y, group, sort_indices).

        Preserves DataFrame type for X when possible so that downstream code
        (e.g. feature name capture, categorical auto-inference) works correctly.
        """
        group_arr = np.asarray(group, dtype=np.uint32).ravel()
        sort_idx = np.argsort(group_arr, kind="stable")

        # Preserve DataFrame metadata if X is a DataFrame.
        try:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                y_arr = np.asarray(y, dtype=np.float32).ravel()
                return X.iloc[sort_idx], y_arr[sort_idx], group_arr[sort_idx], sort_idx
        except ImportError:
            pass

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        return X_arr[sort_idx], y_arr[sort_idx], group_arr[sort_idx], sort_idx
