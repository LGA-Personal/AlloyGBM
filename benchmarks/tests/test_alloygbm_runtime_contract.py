import importlib.util
import sys
import types
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
    "benchmark_runner_runtime_contract_module",
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


class _MissingRoundsRegressor:
    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        seed: int = 7,
        deterministic: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed
        self.deterministic = deterministic


class AlloyRuntimeContractTests(unittest.TestCase):
    def test_runtime_contract_accepts_compatible_regressor(self) -> None:
        native_module = types.SimpleNamespace(train_regression_artifact=lambda *args, **kwargs: b"")
        parameters = RUNNER._verify_alloygbm_runtime_contract(
            _CompatibleRegressor, native_module
        )

        self.assertIn("n_estimators", parameters)
        self.assertIn("row_subsample", parameters)
        self.assertIn("col_subsample", parameters)

    def test_runtime_contract_rejects_missing_training_parameters(self) -> None:
        native_module = types.SimpleNamespace(train_regression_artifact=lambda *args, **kwargs: b"")

        with self.assertRaisesRegex(RuntimeError, "missing required __init__ parameters"):
            RUNNER._verify_alloygbm_runtime_contract(
                _MissingRoundsRegressor, native_module
            )

    def test_runtime_contract_rejects_missing_native_training_symbol(self) -> None:
        native_module = types.SimpleNamespace()

        with self.assertRaisesRegex(RuntimeError, "train_regression_artifact"):
            RUNNER._verify_alloygbm_runtime_contract(
                _CompatibleRegressor, native_module
            )


if __name__ == "__main__":
    unittest.main()
