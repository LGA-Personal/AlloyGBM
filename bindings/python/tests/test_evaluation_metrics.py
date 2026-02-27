"""Deterministic contract tests for AlloyGBM evaluation metrics."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


def load_evaluation_module():
    evaluation_path = (
        Path(__file__).resolve().parents[1] / "alloygbm" / "evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("alloygbm_evaluation", evaluation_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load alloygbm evaluation module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation_module = load_evaluation_module()
mae = evaluation_module.mae
pearson_correlation = evaluation_module.pearson_correlation
r2_score = evaluation_module.r2_score
rmse = evaluation_module.rmse


class _FakeNumpyLike:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _FakePandasLikeSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_numpy(self) -> _FakeNumpyLike:
        return _FakeNumpyLike(self._values)


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_predictions_metrics(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]

        self.assertEqual(rmse(y_true, y_pred), 0.0)
        self.assertEqual(mae(y_true, y_pred), 0.0)
        self.assertEqual(r2_score(y_true, y_pred), 1.0)
        self.assertEqual(pearson_correlation(y_true, y_pred), 1.0)

    def test_inverse_order_metrics(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [3.0, 2.0, 1.0]

        self.assertAlmostEqual(rmse(y_true, y_pred), math.sqrt(8.0 / 3.0), places=12)
        self.assertAlmostEqual(mae(y_true, y_pred), 4.0 / 3.0, places=12)
        self.assertAlmostEqual(r2_score(y_true, y_pred), -3.0, places=12)
        self.assertAlmostEqual(pearson_correlation(y_true, y_pred), -1.0, places=12)

    def test_non_trivial_fixture_metrics(self) -> None:
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]

        self.assertAlmostEqual(rmse(y_true, y_pred), math.sqrt(0.375), places=12)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.5, places=12)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.9486081370449679, places=12)
        self.assertAlmostEqual(
            pearson_correlation(y_true, y_pred), 0.9848696184482703, places=12
        )

    def test_metrics_are_deterministic_for_repeated_calls(self) -> None:
        y_true = [2.0, 4.0, 6.0, 8.0]
        y_pred = [2.5, 3.5, 6.5, 7.5]

        first = (
            rmse(y_true, y_pred),
            mae(y_true, y_pred),
            r2_score(y_true, y_pred),
            pearson_correlation(y_true, y_pred),
        )
        second = (
            rmse(y_true, y_pred),
            mae(y_true, y_pred),
            r2_score(y_true, y_pred),
            pearson_correlation(y_true, y_pred),
        )
        self.assertEqual(first, second)

    def test_metrics_accept_sequence_like_adapters(self) -> None:
        y_true = _FakePandasLikeSeries([1.0, 2.0, 3.0, 4.0])
        y_pred = _FakeNumpyLike([1.2, 1.8, 3.2, 3.8])

        self.assertGreater(rmse(y_true, y_pred), 0.0)
        self.assertGreater(mae(y_true, y_pred), 0.0)
        self.assertLessEqual(r2_score(y_true, y_pred), 1.0)
        self.assertLessEqual(abs(pearson_correlation(y_true, y_pred)), 1.0)

    def test_mismatched_lengths_raise_value_error(self) -> None:
        y_true = [1.0, 2.0]
        y_pred = [1.0]

        for metric in (rmse, mae, r2_score, pearson_correlation):
            with self.assertRaisesRegex(ValueError, "same number of values"):
                metric(y_true, y_pred)

    def test_empty_inputs_raise_value_error(self) -> None:
        y_true: list[float] = []
        y_pred: list[float] = []

        for metric in (rmse, mae, r2_score, pearson_correlation):
            with self.assertRaisesRegex(ValueError, "at least one value"):
                metric(y_true, y_pred)

    def test_non_finite_values_raise_value_error(self) -> None:
        y_true = [1.0, 2.0, float("nan")]
        y_pred = [1.0, 2.0, 3.0]
        for metric in (rmse, mae, r2_score, pearson_correlation):
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                metric(y_true, y_pred)

        y_true_inf = [1.0, float("inf"), 3.0]
        y_pred_inf = [1.0, 2.0, 3.0]
        for metric in (rmse, mae, r2_score, pearson_correlation):
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                metric(y_true_inf, y_pred_inf)

    def test_r2_constant_target_fallback_behavior(self) -> None:
        y_true = [2.0, 2.0, 2.0]
        self.assertEqual(r2_score(y_true, [2.0, 2.0, 2.0]), 1.0)
        self.assertEqual(r2_score(y_true, [2.0, 3.0, 2.0]), 0.0)

    def test_pearson_returns_zero_for_zero_variance_series(self) -> None:
        self.assertEqual(pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)
        self.assertEqual(pearson_correlation([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
