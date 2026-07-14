"""Contract tests for the deterministic ranking and GLM benchmark pack."""

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
    "objective_benchmark_module",
    REPO_ROOT / "benchmarks" / "objective_benchmark.py",
)


class ObjectiveBenchmarkTests(unittest.TestCase):
    def test_fixtures_are_deterministic_and_respect_glm_domains(self) -> None:
        first = BENCHMARK.make_large_query_ranking(seed=17, query_size=32)
        second = BENCHMARK.make_large_query_ranking(seed=17, query_size=32)
        for first_value, second_value in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_value, second_value)

        glm = BENCHMARK.make_skewed_glm_data(seed=23, n_train=48, n_test=24)
        for values in glm.targets_train.values():
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue((glm.targets_train["poisson"] >= 0.0).all())
        self.assertTrue((glm.targets_train["gamma"] > 0.0).all())
        self.assertTrue((glm.targets_train["tweedie"] >= 0.0).all())

    def test_small_real_model_run_has_finite_ranking_and_glm_metrics(self) -> None:
        ranking_rows, glm_rows = BENCHMARK.run_benchmark(
            seeds=(7,),
            query_size=32,
            n_train=96,
            n_test=48,
            n_estimators=8,
        )

        self.assertEqual([row.arm for row in ranking_rows], ["full", "top_10"])
        self.assertEqual(
            [row.objective for row in glm_rows],
            ["poisson_default", "poisson_no_stabilizer", "gamma", "tweedie"],
        )
        for row in [*ranking_rows, *glm_rows]:
            self.assertTrue(np.isfinite(row.metric))
            self.assertTrue(np.isfinite(row.fit_seconds))

    def test_report_states_the_quality_and_stability_contract(self) -> None:
        report = BENCHMARK.render_report(
            ranking_rows=[
                BENCHMARK.RankingRow(seed=7, arm="full", ndcg_at_10=0.71, fit_seconds=1.2),
                BENCHMARK.RankingRow(seed=7, arm="top_10", ndcg_at_10=0.68, fit_seconds=0.8),
            ],
            glm_rows=[
                BENCHMARK.GlmRow(seed=7, objective="poisson_default", deviance=1.0, baseline_deviance=1.2, fit_seconds=0.2),
                BENCHMARK.GlmRow(seed=7, objective="poisson_no_stabilizer", deviance=1.1, baseline_deviance=1.2, fit_seconds=0.2),
                BENCHMARK.GlmRow(seed=7, objective="gamma", deviance=0.9, baseline_deviance=1.1, fit_seconds=0.2),
                BENCHMARK.GlmRow(seed=7, objective="tweedie", deviance=0.8, baseline_deviance=1.0, fit_seconds=0.2),
            ],
            query_size=512,
            n_estimators=50,
        )

        self.assertIn("large query groups", report)
        self.assertIn("Poisson stabilizer", report)
        self.assertIn("held-out", report)
        self.assertIn("top_10", report)

    def test_gate_rejects_material_ranking_quality_loss(self) -> None:
        gates = BENCHMARK.evaluate_gates(
            ranking_rows=[
                BENCHMARK.RankingRow(seed=7, arm="full", ndcg_at_10=0.80, fit_seconds=1.0),
                BENCHMARK.RankingRow(seed=7, arm="top_10", ndcg_at_10=0.69, fit_seconds=0.5),
            ],
            glm_rows=[
                BENCHMARK.GlmRow(seed=7, objective="poisson_default", deviance=1.0, baseline_deviance=1.2, fit_seconds=0.1),
                BENCHMARK.GlmRow(seed=7, objective="poisson_no_stabilizer", deviance=1.0, baseline_deviance=1.2, fit_seconds=0.1),
                BENCHMARK.GlmRow(seed=7, objective="gamma", deviance=1.0, baseline_deviance=1.2, fit_seconds=0.1),
                BENCHMARK.GlmRow(seed=7, objective="tweedie", deviance=1.0, baseline_deviance=1.2, fit_seconds=0.1),
            ],
        )

        self.assertFalse(gates[0].passed)


if __name__ == "__main__":
    unittest.main()
