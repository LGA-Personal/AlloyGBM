import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module(
    "benchmark_runner_model_factories_module",
    REPO_ROOT / "benchmarks" / "run_model_comparison.py",
)


class _CompatibleRegressor:
    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 120,
        row_subsample: float = 0.8,
        col_subsample: float = 0.8,
        seed: int = 7,
        deterministic: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.row_subsample = row_subsample
        self.col_subsample = col_subsample
        self.seed = seed
        self.deterministic = deterministic


class _FakeCatBoostRegressor:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class RunModelComparisonTests(unittest.TestCase):
    def test_model_factories_exclude_catboost_when_unavailable(self) -> None:
        factories = RUNNER._model_factories(
            gbm_regressor_cls=_CompatibleRegressor,
            catboost_regressor_cls=None,
            seed=7,
            learning_rate=0.1,
            max_depth=6,
            rounds=120,
            alloy_continuous_binning_strategy="linear",
            alloy_continuous_binning_max_bins=256,
        )

        self.assertEqual(
            sorted(factories.keys()), ["alloygbm", "lightgbm", "xgboost"]
        )

    def test_model_factories_include_catboost_when_available(self) -> None:
        factories = RUNNER._model_factories(
            gbm_regressor_cls=_CompatibleRegressor,
            catboost_regressor_cls=_FakeCatBoostRegressor,
            seed=7,
            learning_rate=0.1,
            max_depth=6,
            rounds=120,
            alloy_continuous_binning_strategy="linear",
            alloy_continuous_binning_max_bins=256,
        )

        self.assertIn("catboost", factories)
        model = factories["catboost"]()
        self.assertIsInstance(model, _FakeCatBoostRegressor)
        self.assertEqual(model.kwargs["loss_function"], "RMSE")
        self.assertEqual(model.kwargs["iterations"], 120)


if __name__ == "__main__":
    unittest.main()
