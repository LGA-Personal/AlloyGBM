"""Conformance coverage for AlloyGBM's sklearn estimator contracts."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    is_classifier,
    is_regressor,
)

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def test_regressor_mixin_precedes_base_estimator() -> None:
    mro = GBMRegressor.__mro__

    assert RegressorMixin in mro
    assert mro.index(RegressorMixin) < mro.index(BaseEstimator)
    assert is_regressor(GBMRegressor())
    assert not is_classifier(GBMRegressor())


def test_classifier_has_only_classifier_semantics() -> None:
    mro = GBMClassifier.__mro__

    assert ClassifierMixin in mro
    assert RegressorMixin not in mro
    assert mro.index(ClassifierMixin) < mro.index(BaseEstimator)
    assert is_classifier(GBMClassifier())
    assert not is_regressor(GBMClassifier())


def test_ranker_has_no_regression_or_classification_mixin() -> None:
    mro = GBMRanker.__mro__

    assert RegressorMixin not in mro
    assert ClassifierMixin not in mro
    assert BaseEstimator in mro
    assert not is_regressor(GBMRanker())
    assert not is_classifier(GBMRanker())


def test_public_estimator_paths_and_parameter_signatures_are_stable() -> None:
    assert GBMRegressor.__module__ == "alloygbm.regressor"
    assert GBMClassifier.__module__ == "alloygbm.classifier"
    assert GBMRanker.__module__ == "alloygbm.ranker"

    regressor_parameters = inspect.signature(GBMRegressor.__init__).parameters
    classifier_parameters = inspect.signature(GBMClassifier.__init__).parameters
    ranker_parameters = inspect.signature(GBMRanker.__init__).parameters

    assert classifier_parameters == regressor_parameters
    assert set(regressor_parameters) <= set(ranker_parameters)
    assert {
        "ranking_objective",
        "ranking_sigma",
        "lambdarank_truncation_level",
        "lambdarank_normalize",
    } <= set(ranker_parameters)


def test_estimators_remain_importable_without_sklearn() -> None:
    package_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ImportError("sklearn blocked for optional-dependency test")
        return None

sys.meta_path.insert(0, BlockSklearn())

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

assert GBMRegressor().get_params()["n_estimators"] == 6
assert GBMClassifier().get_params()["n_estimators"] == 6
assert GBMRanker().get_params()["ranking_objective"] == "rank:ndcg"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
