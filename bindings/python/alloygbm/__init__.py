"""Public Python API for the AlloyGBM gradient-boosted decision tree library."""

from ._alloygbm import native_runtime_info
from .classifier import GBMClassifier
from .evaluation import (
    accuracy,
    hit_rate,
    icir,
    log_loss,
    mae,
    ndcg,
    pearson_correlation,
    r2_score,
    rank_ic,
    rmse,
)
from .ranker import GBMRanker
from .regressor import GBMRegressor
from .validation import purged_panel_splits, purged_time_series_splits

__all__ = [
    "GBMClassifier",
    "GBMRanker",
    "GBMRegressor",
    "accuracy",
    "hit_rate",
    "icir",
    "log_loss",
    "mae",
    "native_runtime_info",
    "ndcg",
    "pearson_correlation",
    "purged_panel_splits",
    "purged_time_series_splits",
    "r2_score",
    "rank_ic",
    "rmse",
]
