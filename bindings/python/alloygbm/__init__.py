"""Public Python API for the AlloyGBM gradient-boosted decision tree library."""

__version__ = "0.12.10"

from ._alloygbm import native_runtime_info
from .classifier import GBMClassifier
from .evaluation import (
    accuracy,
    gamma_deviance,
    hit_rate,
    icir,
    log_loss,
    mae,
    multiclass_log_loss,
    ndcg,
    pearson_correlation,
    poisson_deviance,
    r2_score,
    rank_ic,
    rmse,
    tweedie_deviance,
)
from .multi_label_ranker import MultiLabelGBMRanker
from .ranker import GBMRanker
from .regressor import GBMRegressor
from .validation import purged_panel_splits, purged_time_series_splits

__all__ = [
    "GBMClassifier",
    "GBMRanker",
    "GBMRegressor",
    "MultiLabelGBMRanker",
    "__version__",
    "accuracy",
    "gamma_deviance",
    "hit_rate",
    "icir",
    "log_loss",
    "mae",
    "multiclass_log_loss",
    "native_runtime_info",
    "ndcg",
    "pearson_correlation",
    "poisson_deviance",
    "purged_panel_splits",
    "purged_time_series_splits",
    "r2_score",
    "rank_ic",
    "rmse",
    "tweedie_deviance",
]
