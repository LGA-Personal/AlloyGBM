"""Public Python API for the AlloyGBM baseline regressor scaffold."""

from ._alloygbm import native_runtime_info
from .evaluation import (
    hit_rate,
    icir,
    mae,
    pearson_correlation,
    r2_score,
    rank_ic,
    rmse,
)
from .regressor import GBMRegressor
from .validation import purged_panel_splits, purged_time_series_splits

__all__ = [
    "GBMRegressor",
    "hit_rate",
    "icir",
    "mae",
    "native_runtime_info",
    "pearson_correlation",
    "purged_panel_splits",
    "purged_time_series_splits",
    "r2_score",
    "rank_ic",
    "rmse",
]
