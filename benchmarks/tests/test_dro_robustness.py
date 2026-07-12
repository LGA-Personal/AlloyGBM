"""Contract tests for the deterministic DRO robustness benchmark."""

from __future__ import annotations

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


BENCHMARK = _load_module(
    "dro_robustness_benchmark_module",
    REPO_ROOT / "benchmarks" / "dro_robustness.py",
)


class DroRobustnessBenchmarkTests(unittest.TestCase):
    def test_fixture_is_deterministic_and_contaminates_only_training_targets(self) -> None:
        first = BENCHMARK.make_noisy_regression(seed=17, n_train=48, n_test=24)
        second = BENCHMARK.make_noisy_regression(seed=17, n_train=48, n_test=24)

        for first_value, second_value in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_value, second_value)

        _, y_clean_train, y_corrupted_train, _, y_clean_test = first
        self.assertGreater(float(np.mean(np.abs(y_corrupted_train - y_clean_train))), 0.0)
        self.assertTrue(np.isfinite(y_clean_test).all())

    def test_small_real_model_run_returns_finite_scalar_and_joint_scores(self) -> None:
        scalar_rows, joint_rows = BENCHMARK.run_benchmark(
            seeds=(7,),
            n_train=80,
            n_test=40,
        )

        self.assertEqual([row.solver for row in scalar_rows], ["standard", "dro"])
        self.assertEqual([row.solver for row in joint_rows], ["standard", "dro"])
        for row in [*scalar_rows, *joint_rows]:
            self.assertTrue(np.isfinite(row.clean_train_rmse))
            self.assertTrue(np.isfinite(row.corrupted_train_rmse))

    def test_report_explains_clean_holdout_and_joint_leaf_only_limitation(self) -> None:
        report = BENCHMARK.render_report(
            scalar_rows=[
                BENCHMARK.BenchmarkRow(
                    seed=7,
                    solver="standard",
                    clean_train_rmse=0.1,
                    corrupted_train_rmse=0.2,
                ),
                BENCHMARK.BenchmarkRow(
                    seed=7,
                    solver="dro",
                    clean_train_rmse=0.11,
                    corrupted_train_rmse=0.18,
                ),
            ],
            joint_rows=[
                BENCHMARK.BenchmarkRow(
                    seed=7,
                    solver="standard",
                    clean_train_rmse=0.3,
                    corrupted_train_rmse=0.4,
                ),
                BENCHMARK.BenchmarkRow(
                    seed=7,
                    solver="dro",
                    clean_train_rmse=0.31,
                    corrupted_train_rmse=0.35,
                ),
            ],
            radius=0.05,
            contamination_fraction=0.12,
        )

        self.assertIn("clean held-out targets", report)
        self.assertIn("joint path applies DRO only to final leaf values", report)
        self.assertIn("0.050", report)
        self.assertIn("standard", report)
        self.assertIn("dro", report)


if __name__ == "__main__":
    unittest.main()
