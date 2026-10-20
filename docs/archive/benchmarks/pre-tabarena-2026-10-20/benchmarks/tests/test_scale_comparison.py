"""Tests for the dependency-light scale comparison factory contract."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scale_comparison_module", REPO_ROOT / "benchmarks" / "scale_comparison.py"
)
assert SPEC is not None and SPEC.loader is not None
SCALE_COMPARISON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCALE_COMPARISON
SPEC.loader.exec_module(SCALE_COMPARISON)


class _FakeEstimator:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "task",
    [
        "regression",
        "classification",
    ],
)
def test_catboost_factory_matches_sampling_and_common_controls(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    alloygbm = types.ModuleType("alloygbm")
    alloygbm.GBMRegressor = _FakeEstimator
    alloygbm.GBMClassifier = _FakeEstimator
    lightgbm = types.ModuleType("lightgbm")
    lightgbm.LGBMRegressor = _FakeEstimator
    lightgbm.LGBMClassifier = _FakeEstimator
    xgboost = types.ModuleType("xgboost")
    xgboost.XGBRegressor = _FakeEstimator
    xgboost.XGBClassifier = _FakeEstimator
    catboost = types.ModuleType("catboost")
    catboost.CatBoostRegressor = _FakeEstimator
    catboost.CatBoostClassifier = _FakeEstimator
    monkeypatch.setitem(sys.modules, "alloygbm", alloygbm)
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "xgboost", xgboost)
    monkeypatch.setitem(sys.modules, "catboost", catboost)

    model = SCALE_COMPARISON._factories(
        task=task, threads=3, rounds=17, depth=5, lr=0.07, seed=19
    )["catboost"]()

    assert isinstance(model, _FakeEstimator)
    assert model.kwargs == {
        "iterations": 17,
        "learning_rate": 0.07,
        "depth": 5,
        "random_seed": 19,
        "thread_count": 3,
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.8,
    }
