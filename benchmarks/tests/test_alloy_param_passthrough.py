"""Tests for benchmark AlloyGBM parameter overrides and provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = REPO_ROOT / "benchmarks" / "run_model_comparison.py"
    spec = importlib.util.spec_from_file_location("alloy_param_passthrough_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def test_parses_float_int_bool_and_str():
    got = RUNNER.parse_alloy_param_overrides(
        [
            "min_child_hessian=1.0",
            "min_data_in_leaf=32",
            "deterministic=true",
            "tree_growth=leaf",
        ]
    )
    assert got == {
        "min_child_hessian": 1.0,
        "min_data_in_leaf": 32,
        "deterministic": True,
        "tree_growth": "leaf",
    }
    assert isinstance(got["min_data_in_leaf"], int)
    assert isinstance(got["min_child_hessian"], float)


def test_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        RUNNER.parse_alloy_param_overrides(["min_child_hessian"])


def test_rejects_empty_key():
    with pytest.raises(ValueError, match="empty parameter name"):
        RUNNER.parse_alloy_param_overrides(["=1.0"])


def test_rejects_nonfinite_numeric_values():
    with pytest.raises(ValueError, match="finite"):
        RUNNER.parse_alloy_param_overrides(["min_child_hessian=nan"])


def test_rejects_duplicate_parameter_names():
    with pytest.raises(ValueError, match="more than once"):
        RUNNER.parse_alloy_param_overrides(["max_leaves=8", "max_leaves=12"])


def test_empty_list_is_empty_dict():
    assert RUNNER.parse_alloy_param_overrides([]) == {}


def test_overrides_reach_real_regression_binary_multiclass_and_ranking_factories():
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

    overrides = {
        "min_child_hessian": 4.0,
        "min_data_in_leaf": 8,
        "colsample_bynode": 0.6,
        "tree_growth": "leaf",
        "max_leaves": 10,
        "training_mode": "auto",
    }
    common = {
        "seed": 7,
        "learning_rate": 0.1,
        "max_depth": 6,
        "rounds": 12,
        "alloy_continuous_binning_strategy": "linear",
        "alloy_continuous_binning_max_bins": 256,
        "alloy_param_overrides": overrides,
    }
    factories = [
        RUNNER._model_factories(
            gbm_regressor_cls=GBMRegressor,
            catboost_regressor_cls=None,
            **common,
        ),
        RUNNER._classifier_factories(
            gbm_classifier_cls=GBMClassifier,
            catboost_classifier_cls=None,
            **common,
        ),
        RUNNER._multiclass_classifier_factories(
            gbm_classifier_cls=GBMClassifier,
            catboost_classifier_cls=None,
            n_classes=3,
            **common,
        ),
        RUNNER._ranker_factories(
            gbm_ranker_cls=GBMRanker,
            catboost_available=False,
            **common,
        ),
    ]

    for task_factories in factories:
        params = task_factories["alloygbm"]().get_params()
        for name, value in overrides.items():
            assert params[name] == value

    regression_factories = factories[0]
    assert regression_factories["alloygbm_morph"]().get_params()["training_mode"] == "auto"
    assert regression_factories["alloygbm_dro"]().get_params()["tree_growth"] == "leaf"


def test_linear_variant_applies_lambda_override_without_duplicate_keyword():
    from alloygbm import GBMRegressor

    factories = RUNNER._model_factories(
        gbm_regressor_cls=GBMRegressor,
        catboost_regressor_cls=None,
        seed=7,
        learning_rate=0.1,
        max_depth=6,
        rounds=12,
        alloy_continuous_binning_strategy="linear",
        alloy_continuous_binning_max_bins=256,
        alloy_param_overrides={"lambda_l2": 0.4},
    )

    assert factories["alloygbm_linear"]().get_params()["lambda_l2"] == 0.4


def test_unknown_parameter_fails_in_real_factory():
    from alloygbm import GBMRegressor

    with pytest.raises(ValueError, match="not a parameter"):
        RUNNER._model_factories(
            gbm_regressor_cls=GBMRegressor,
            catboost_regressor_cls=None,
            seed=7,
            learning_rate=0.1,
            max_depth=6,
            rounds=12,
            alloy_continuous_binning_strategy="linear",
            alloy_continuous_binning_max_bins=256,
            alloy_param_overrides={"nonsuch_parameter": 1},
        )


def test_conflicting_fairness_override_is_rejected():
    from alloygbm import GBMRegressor

    with pytest.raises(ValueError, match="shared CLI control"):
        RUNNER._model_factories(
            gbm_regressor_cls=GBMRegressor,
            catboost_regressor_cls=None,
            seed=7,
            learning_rate=0.1,
            max_depth=6,
            rounds=12,
            alloy_continuous_binning_strategy="linear",
            alloy_continuous_binning_max_bins=256,
            threads=2,
            alloy_param_overrides={"n_jobs": 1},
        )


def test_learning_rate_and_round_overrides_are_marked_as_asymmetric():
    profiles = [
        RUNNER.BenchmarkProfile("a", learning_rate=0.1, max_depth=6, rounds=120),
        RUNNER.BenchmarkProfile("b", learning_rate=0.05, max_depth=8, rounds=400),
    ]
    assert RUNNER._alloy_param_asymmetries(
        {"learning_rate": 0.025, "n_estimators": 800, "min_child_hessian": 2.0},
        profiles,
    ) == {
        "learning_rate": {
            "alloy_value": 0.025,
            "peer_parameter": "profile.learning_rate",
            "peer_values": [0.1, 0.05],
        },
        "n_estimators": {
            "alloy_value": 800,
            "peer_parameter": "profile.rounds",
            "peer_values": [120, 400],
        },
    }


class _InspectableRegressionModel:
    def __init__(self) -> None:
        self.fit_timing_ = {}

    def get_params(self) -> dict[str, object]:
        return {"max_depth": 12, "tree_growth": "leaf"}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0.0] * len(X)


def test_benchmark_record_preserves_requested_and_effective_parameters():
    import numpy as np

    requested = {"max_leaves": 10}
    record = RUNNER._run_model(
        model_name="alloygbm",
        factory=_InspectableRegressionModel,
        x_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0.0, 1.0]),
        x_test=np.array([[0.5], [0.75]]),
        y_test=np.array([0.0, 1.0]),
        scenario="dense_numeric",
        profile=RUNNER.DEFAULT_PROFILES[0],
        profile_index=1,
        run_index=1,
        seed=7,
        requested_alloy_param_overrides=requested,
    )

    assert record.requested_alloy_param_overrides == requested
    assert record.effective_estimator_params == {
        "max_depth": 12,
        "tree_growth": "leaf",
    }
    assert record.status == "PASS"


def test_output_json_keeps_parameter_provenance(tmp_path):
    import numpy as np

    record = RUNNER._run_model(
        model_name="alloygbm",
        factory=_InspectableRegressionModel,
        x_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0.0, 1.0]),
        x_test=np.array([[0.5], [0.75]]),
        y_test=np.array([0.0, 1.0]),
        scenario="dense_numeric",
        profile=RUNNER.DEFAULT_PROFILES[0],
        profile_index=1,
        run_index=1,
        seed=7,
        requested_alloy_param_overrides={"max_leaves": 10},
    )
    paths = RUNNER._write_outputs(
        tmp_path,
        "test-run",
        [record],
        {
            "profile_mode": "single",
            "profile_grid": "none",
            "profiles": [],
            "profile_seeds": [7],
            "seed": 7,
            "learning_rate": 0.1,
            "max_depth": 6,
            "rounds": 120,
            "alloy_continuous_binning_strategy": "linear",
            "alloy_continuous_binning_max_bins": 256,
            "test_size": 0.2,
            "scenarios": ["dense_numeric"],
            "requested_alloy_param_overrides": {"max_leaves": 10},
        },
    )

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["params"]["requested_alloy_param_overrides"] == {"max_leaves": 10}
    assert payload["records"][0]["requested_alloy_param_overrides"] == {"max_leaves": 10}
    assert payload["records"][0]["effective_estimator_params"] == {
        "max_depth": 12,
        "tree_growth": "leaf",
    }
