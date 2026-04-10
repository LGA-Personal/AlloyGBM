import importlib.util
import math
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
        self.fit_timing_ = {
            "input_adaptation_seconds": 0.01,
            "native_bridge_prepare_seconds": 0.02,
            "native_train_seconds": 0.03,
        }

    def fit(self, X: object, y: object) -> "_TrackingAlloyRegressor":
        type(self).fit_input = X
        return self

    def predict(self, X: object) -> list[float]:
        type(self).predict_input = X
        row_count = int(getattr(X, "shape", [len(X)])[0])  # type: ignore[arg-type]
        return [0.0] * row_count


class _CompatibleClassifier(_CompatibleRegressor):
    pass


class _TrackingAlloyClassifier:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_timing_ = {
            "input_adaptation_seconds": 0.01,
            "native_bridge_prepare_seconds": 0.02,
            "native_train_seconds": 0.03,
        }

    def fit(self, X: object, y: object) -> "_TrackingAlloyClassifier":
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        row_count = int(getattr(X, "shape", [len(X)])[0])
        p1 = np.full(row_count, 0.8)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: object) -> list[int]:
        row_count = int(getattr(X, "shape", [len(X)])[0])
        return [1] * row_count


class _TrackingAlloyRanker:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_timing_ = {
            "input_adaptation_seconds": 0.01,
            "native_bridge_prepare_seconds": 0.02,
            "native_train_seconds": 0.03,
        }

    def fit(self, X: object, y: object, group: object = None) -> "_TrackingAlloyRanker":
        return self

    def predict(self, X: object) -> list[float]:
        row_count = int(getattr(X, "shape", [len(X)])[0])
        return [float(i) for i in range(row_count)]


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

        record = RUNNER._run_model(
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
        self.assertEqual(record.input_adaptation_seconds, 0.01)
        self.assertEqual(record.native_bridge_prepare_seconds, 0.02)
        self.assertEqual(record.native_train_seconds, 0.03)


    def test_available_scenarios_include_classification_and_ranking(self) -> None:
        self.assertIn("breast_cancer", RUNNER.AVAILABLE_SCENARIOS)
        self.assertIn("synthetic_classification", RUNNER.AVAILABLE_SCENARIOS)
        self.assertIn("synthetic_ranking", RUNNER.AVAILABLE_SCENARIOS)

    def test_classifier_factories_produces_expected_models(self) -> None:
        factories = RUNNER._classifier_factories(
            gbm_classifier_cls=_CompatibleClassifier,
            catboost_classifier_cls=None,
            seed=7,
            learning_rate=0.1,
            max_depth=6,
            rounds=120,
            alloy_continuous_binning_strategy="linear",
            alloy_continuous_binning_max_bins=256,
        )
        self.assertEqual(sorted(factories.keys()), ["alloygbm", "lightgbm", "xgboost"])

    def test_benchmark_record_includes_task_type(self) -> None:
        self.assertTrue(hasattr(RUNNER.BenchmarkRecord, "__dataclass_fields__"))
        self.assertIn("task_type", RUNNER.BenchmarkRecord.__dataclass_fields__)

    def test_run_model_classification_computes_metrics(self) -> None:
        x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        y_train = np.array([0.0, 1.0], dtype=float)
        x_test = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=float)
        y_test = np.array([0.0, 1.0], dtype=float)

        record = RUNNER._run_model(
            model_name="alloygbm",
            factory=_TrackingAlloyClassifier,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scenario="breast_cancer",
            profile=RUNNER.DEFAULT_PROFILES[0],
            profile_index=1,
            run_index=1,
            seed=7,
            task_type="classification",
        )

        self.assertEqual(record.status, "PASS")
        self.assertEqual(record.task_type, "classification")
        self.assertFalse(math.isnan(record.accuracy))
        self.assertFalse(math.isnan(record.log_loss_val))
        self.assertFalse(math.isnan(record.auc))
        # Regression and ranking metrics should be NaN.
        self.assertTrue(math.isnan(record.rmse))
        self.assertTrue(math.isnan(record.ndcg_full))

    def test_run_model_ranking_computes_metrics(self) -> None:
        # 6 rows, 3 queries of 2 docs each.
        x_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=float)
        y_train = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0], dtype=float)
        x_test = np.array([[1.5], [2.5], [3.5], [4.5]], dtype=float)
        y_test = np.array([1.0, 0.0, 2.0, 1.0], dtype=float)
        g_train = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32)
        g_test = np.array([10, 10, 11, 11], dtype=np.uint32)

        record = RUNNER._run_model(
            model_name="alloygbm",
            factory=_TrackingAlloyRanker,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scenario="synthetic_ranking",
            profile=RUNNER.DEFAULT_PROFILES[0],
            profile_index=1,
            run_index=1,
            seed=7,
            task_type="ranking",
            group_train=g_train,
            group_test=g_test,
        )

        self.assertEqual(record.status, "PASS")
        self.assertEqual(record.task_type, "ranking")
        self.assertFalse(math.isnan(record.ndcg_5))
        self.assertFalse(math.isnan(record.ndcg_10))
        self.assertFalse(math.isnan(record.ndcg_full))
        # Regression and classification metrics should be NaN.
        self.assertTrue(math.isnan(record.rmse))
        self.assertTrue(math.isnan(record.accuracy))


if __name__ == "__main__":
    unittest.main()
