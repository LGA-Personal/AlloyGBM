"""Contract tests for the deferred-architecture benchmark pack."""

from __future__ import annotations

import importlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module(name: str):
    return importlib.import_module(f"benchmarks.architectural_backlog.{name}")


class ArchitecturalBacklogBenchmarkTests(unittest.TestCase):
    def test_package_entrypoint_exists(self) -> None:
        self.assertTrue(
            (REPO_ROOT / "benchmarks" / "architectural_backlog" / "run.py").is_file()
        )

    def test_fixtures_are_deterministic_and_cover_all_scenarios(self) -> None:
        fixtures = _module("fixtures")
        scenarios = _module("scenarios")

        first = fixtures.make_fixture("efb", "exclusive_one_hot", "quick", seed=17)
        second = fixtures.make_fixture("efb", "exclusive_one_hot", "quick", seed=17)
        self.assertEqual(first.dimensions, second.dimensions)
        self.assertEqual(first.X.tobytes(), second.X.tobytes())
        self.assertEqual(first.y.tobytes(), second.y.tobytes())
        self.assertEqual(
            set(scenarios.SCENARIO_CASES),
            {
                "soa_histograms",
                "node_parallelism",
                "duplicate_bins",
                "compact_nodes",
                "efb",
                "quantile_sketches",
            },
        )

    def test_report_roundtrip_and_duplicate_key_validation(self) -> None:
        common = _module("common")
        report = common.select_scenarios(
            common.synthetic_report_for_tests(mode="baseline", profile="quick"),
            ["duplicate_bins"],
        )
        result = report.results[0]

        decoded = common.BenchmarkReport.from_json(report.to_json())
        self.assertEqual(decoded, report)
        with self.assertRaisesRegex(ValueError, "duplicate result key"):
            common.validate_report(
                common.BenchmarkReport(
                    schema_version=1,
                    profile="quick",
                    mode="baseline",
                    environment=report.environment,
                    results=(result, result),
                )
            )

    def test_environment_compatibility_is_explicit(self) -> None:
        common = _module("common")
        baseline = {
            "platform": "darwin",
            "machine": "arm64",
            "logical_cpus": 10,
            "python_major_minor": "3.13",
            "rayon_num_threads": None,
        }
        self.assertEqual(common.environment_mismatches(baseline, dict(baseline)), [])
        candidate = dict(baseline, logical_cpus=8)
        self.assertEqual(
            common.environment_mismatches(baseline, candidate),
            ["logical_cpus: baseline=10 candidate=8"],
        )
        threaded = dict(baseline, rayon_num_threads="8")
        self.assertEqual(
            common.environment_mismatches(baseline, threaded),
            ["rayon_num_threads: baseline=None candidate=8"],
        )

    def test_rss_normalization_handles_macos_and_linux_units(self) -> None:
        common = _module("common")
        self.assertAlmostEqual(common.normalize_max_rss_mb(10 * 1024 * 1024, "darwin"), 10.0)
        self.assertAlmostEqual(common.normalize_max_rss_mb(10 * 1024, "linux"), 10.0)
        self.assertIsNone(common.normalize_max_rss_mb(1234, "win32"))
        current = common.current_rss_mb()
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            self.assertGreater(current, 0.0)
        else:
            self.assertIsNone(current)

    def test_comparator_separates_quality_and_performance_gates(self) -> None:
        common = _module("common")
        baseline = common.synthetic_report_for_tests(mode="baseline", profile="full")
        candidate = common.synthetic_report_for_tests(mode="candidate", profile="full")

        gates = common.compare_reports(baseline, candidate)
        self.assertTrue(gates)
        self.assertTrue(all(gate.passed for gate in gates), gates)
        slower = common.replace_metric(
            candidate,
            scenario="soa_histograms",
            case="standard_wide",
            metric="native_train_seconds",
            value=2.0,
        )
        failed = common.compare_reports(baseline, slower)
        self.assertTrue(
            any(gate.name == "soa_histograms: standard speed" and not gate.passed for gate in failed)
        )

        failure_metrics = {
            "node_parallelism": ("threads_8", "native_train_seconds", 2.0),
            "duplicate_bins": ("wide_shallow_u8", "fit_peak_rss_mb", 100.0),
            "compact_nodes": ("sparse_spines", "predict_seconds_per_row", 2.0),
            "efb": ("exclusive_one_hot", "candidate_active", False),
            "quantile_sketches": ("large_skewed", "max_rank_error", 0.02),
        }
        for scenario, (case, metric, value) in failure_metrics.items():
            with self.subTest(scenario=scenario):
                report = common.replace_metric(
                    candidate,
                    scenario=scenario,
                    case=case,
                    metric=metric,
                    value=value,
                )
                self.assertTrue(
                    any(not gate.passed for gate in common.compare_reports(baseline, report))
                )

    def test_comparator_checks_each_repetition_and_efb_artifact(self) -> None:
        common = _module("common")
        baseline = common.synthetic_report_for_tests(mode="baseline", profile="full")
        candidate = common.synthetic_report_for_tests(mode="candidate", profile="full")

        mismatched_prediction = common.replace_metric(
            candidate,
            scenario="soa_histograms",
            case="standard_wide",
            repetition=0,
            metric="prediction_digest",
            value="x" * 64,
        )
        gates = common.compare_reports(baseline, mismatched_prediction)
        self.assertTrue(
            any("repetition 0 parity" in gate.name and not gate.passed for gate in gates)
        )

        mismatched_artifact = common.replace_metric(
            candidate,
            scenario="efb",
            case="exclusive_one_hot",
            repetition=1,
            metric="artifact_digest",
            value="y" * 64,
        )
        gates = common.compare_reports(baseline, mismatched_artifact)
        self.assertTrue(
            any("repetition 1 artifact parity" in gate.name and not gate.passed for gate in gates)
        )

    def test_scenario_selection_and_report_validation_are_strict(self) -> None:
        common = _module("common")
        baseline = common.synthetic_report_for_tests(mode="baseline", profile="full")
        selected = common.select_scenarios(baseline, ["compact_nodes"])
        self.assertEqual({result.scenario for result in selected.results}, {"compact_nodes"})
        self.assertEqual(len(selected.results), 3)
        candidate = common.select_scenarios(
            common.synthetic_report_for_tests(mode="candidate", profile="full"),
            ["compact_nodes"],
        )
        self.assertTrue(
            all(gate.passed for gate in common.compare_reports(selected, candidate))
        )

        negative = common.replace_metric(
            selected,
            scenario="compact_nodes",
            case="sparse_spines",
            metric="load_seconds",
            value=-1.0,
        )
        with self.assertRaisesRegex(ValueError, "negative"):
            common.validate_report(negative)
        wrong_type = common.replace_metric(
            selected,
            scenario="compact_nodes",
            case="sparse_spines",
            metric="load_seconds",
            value="fast",
        )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            common.validate_report(wrong_type)
        unknown_case = replace(
            selected,
            results=(
                replace(selected.results[0], case="unknown_case"),
                *selected.results[1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown case"):
            common.validate_report(unknown_case)

    def test_quantile_rank_errors_require_cut_metadata(self) -> None:
        scenarios = _module("scenarios")
        import numpy as np

        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            scenarios._rank_errors(np.ones((4, 1), dtype=np.float32), None)
        with self.assertRaisesRegex(RuntimeError, "feature count"):
            scenarios._rank_errors(
                np.ones((4, 2), dtype=np.float32), [[0.5]]
            )

    def test_quick_baseline_case_runs_a_real_model(self) -> None:
        scenarios = _module("scenarios")
        result = scenarios.run_case(
            scenario="duplicate_bins",
            case="wide_shallow_u8",
            profile="quick",
            mode="baseline",
            repetition=0,
            seed=7,
        )
        self.assertGreater(result.metrics["fit_seconds"], 0.0)
        self.assertTrue(math.isfinite(result.metrics["rmse"]))
        self.assertEqual(len(result.metrics["prediction_digest"]), 64)

    def test_cli_writes_json_and_markdown_for_selected_scenario(self) -> None:
        runner = _module("run")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            exit_code = runner.main(
                [
                    "--profile",
                    "quick",
                    "--mode",
                    "baseline",
                    "--scenario",
                    "duplicate_bins",
                    "--output",
                    str(output),
                    "--gate",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["profile"], "quick")
            report = output.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("Duplicate Bin Storage", report)
            self.assertIn("quality", report.lower())


if __name__ == "__main__":
    unittest.main()
