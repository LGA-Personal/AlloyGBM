import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np


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


class _TrackingAlloyRegressor:
    fit_input: object | None = None
    predict_input: object | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(self, X: object, y: object) -> "_TrackingAlloyRegressor":
        type(self).fit_input = X
        return self

    def predict(self, X: object) -> list[float]:
        type(self).predict_input = X
        row_count = int(getattr(X, "shape", [len(X)])[0])  # type: ignore[arg-type]
        return [0.0] * row_count


class RunModelComparisonTests(unittest.TestCase):
    def test_available_scenarios_include_real_application_additions(self) -> None:
        self.assertIn("california_housing", RUNNER.AVAILABLE_SCENARIOS)
        self.assertIn("bike_sharing", RUNNER.AVAILABLE_SCENARIOS)

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

    def test_run_model_passes_numpy_arrays_to_alloy(self) -> None:
        x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        y_train = np.array([0.0, 1.0], dtype=float)
        x_test = np.array([[0.5, 0.5]], dtype=float)
        y_test = np.array([0.0], dtype=float)

        RUNNER._run_model(
            model_name="alloygbm",
            factory=_TrackingAlloyRegressor,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scenario="dense_numeric",
            profile=RUNNER.DEFAULT_PROFILES[0],
            profile_index=1,
            run_index=1,
            seed=7,
        )

        self.assertIs(_TrackingAlloyRegressor.fit_input, x_train)
        self.assertIs(_TrackingAlloyRegressor.predict_input, x_test)


if __name__ == "__main__":
    unittest.main()
