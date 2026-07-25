"""Contract tests for July-review benchmark evidence guardrails."""

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
    "review_guardrails_benchmark_module",
    REPO_ROOT / "benchmarks" / "review_guardrails.py",
)


class ReviewGuardrailTests(unittest.TestCase):
    def test_quantile_fixture_and_weighted_quantile_are_deterministic(self) -> None:
        first = BENCHMARK.make_quantile_split_data(seed=17, n_train=96, n_test=48)
        second = BENCHMARK.make_quantile_split_data(seed=17, n_train=96, n_test=48)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)

        value = BENCHMARK.weighted_quantile(
            np.asarray([1.0, 5.0, 9.0]),
            np.asarray([1.0, 3.0, 1.0]),
            0.5,
        )
        self.assertEqual(value, 5.0)

    def test_smoothed_pinball_gradient_is_continuous_with_nonnegative_hessian(self) -> None:
        alpha = 0.2
        width = 0.5
        left_width = width * (1.0 - alpha)
        right_width = width * alpha
        residual = np.asarray(
            [
                -left_width - 1e-9,
                -left_width + 1e-9,
                0.0,
                right_width - 1e-9,
                right_width + 1e-9,
            ]
        )
        gradient, hessian = BENCHMARK.smoothed_pinball_grad_hess(
            residual, np.ones_like(residual), alpha=alpha, width=width
        )
        self.assertLess(np.max(np.abs(np.diff(gradient)[[0, 3]])), 1e-6)
        self.assertTrue(np.all(hessian >= 0.0))
        self.assertLess(abs(gradient[2]), 1e-12)

    def test_quantile_split_experiment_returns_valid_finite_children(self) -> None:
        rows = BENCHMARK.run_quantile_experiment(
            seeds=(7,), alphas=(0.1, 0.5, 0.9), n_train=160, n_test=96
        )
        self.assertEqual({row.arm for row in rows}, {"proxy", "smooth_0.05", "smooth_0.10"})
        self.assertTrue(
            all(np.isfinite(row.gain) and np.isfinite(row.pinball_loss) for row in rows)
        )
        self.assertTrue(all(row.left_count >= 8 and row.right_count >= 8 for row in rows))

    def test_boosting_fixture_and_dropout_pressure_are_deterministic(self) -> None:
        first = BENCHMARK.make_boosting_data(seed=19, n_train=128, n_test=64)
        second = BENCHMARK.make_boosting_data(seed=19, n_train=128, n_test=64)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)

        low_cap = BENCHMARK.configured_dropout_pressure(
            n_estimators=100, drop_rate=0.2, max_drop=5
        )
        high_cap = BENCHMARK.configured_dropout_pressure(
            n_estimators=100, drop_rate=0.2, max_drop=50
        )
        self.assertLess(0.0, low_cap)
        self.assertLess(low_cap, high_cap)

    def test_small_goss_and_dart_runs_are_finite_and_complete(self) -> None:
        goss = BENCHMARK.run_goss_experiment(
            seeds=(7,), n_train=192, n_test=96, n_estimators=8, rates=((0.2, 0.1),)
        )
        dart = BENCHMARK.run_dart_experiment(
            seeds=(7,), n_train=192, n_test=96, configs=((8, 0.1, 5),)
        )
        for row in [*goss, *dart]:
            self.assertTrue(np.isfinite(row.rmse))
            self.assertGreater(row.fit_seconds, 0.0)
            self.assertEqual(row.completed_rounds, row.requested_rounds)


if __name__ == "__main__":
    unittest.main()
