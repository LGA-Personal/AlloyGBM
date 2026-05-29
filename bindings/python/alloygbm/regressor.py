"""Backwards-compatible shim. GBMRegressor now lives in alloygbm._regressor._core."""

from alloygbm._regressor._core import *  # noqa: F401,F403  re-export public API
from alloygbm._regressor._core import GBMRegressor  # noqa: F401  explicit (survives __all__)
