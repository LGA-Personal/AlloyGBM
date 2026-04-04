"""Public Python API for the AlloyGBM gradient-boosted decision tree library."""

from ._alloygbm import native_runtime_info
from .classifier import GBMClassifier
from .evaluation import (
    accuracy,
    hit_rate,
    icir,
    log_loss,
    mae,
    pearson_correlation,
    r2_score,
    rank_ic,
    rmse,
)
from .regressor import GBMRegressor
from .validation import purged_panel_splits, purged_time_series_splits

__all__ = [
    "GBMClassifier",
    "GBMRegressor",
    "accuracy",
    "hit_rate",
    "icir",
    "log_loss",
    "mae",
    "native_runtime_info",
    "pearson_correlation",
    "purged_panel_splits",
    "purged_time_series_splits",
    "r2_score",
    "rank_ic",
    "rmse",
]
