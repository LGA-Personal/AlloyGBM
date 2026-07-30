"""Contract tests for the multiclass parallelism acceptance benchmark."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "multiclass_parallelism_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "multiclass_parallelism_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def _record(scenario, n_jobs):
    suffix = "serial" if n_jobs == 1 else "parallel"
    return {
        "scenario": scenario.name,
        "shape": scenario.shape,
        "n_rows": scenario.n_rows,
        "n_features": scenario.n_features,
        "n_classes": scenario.n_classes,
        "tree_growth": scenario.tree_growth,
        "seed": scenario.seed,
        "requested_workers": n_jobs,
        "resolved_workers": n_jobs,
        "class_parallel_eligible": n_jobs > 1
        and scenario.n_classes * scenario.n_rows * scenario.n_features
        >= BENCHMARK.MIN_MULTICLASS_PARALLEL_WORK,
        "fit_seconds": 1.0 if n_jobs == 1 else 0.8,
        "requested_rounds": BENCHMARK.QUICK_ROUNDS,
        "completed_rounds": BENCHMARK.QUICK_ROUNDS,
        "stop_reason": "completed",
        "artifact_sha256": f"artifact-{scenario.name}",
        "prediction_sha256": f"prediction-{scenario.name}",
        "predictions_finite": True,
        "multiclass_log_loss": 0.7,
        "class_prior_log_loss": 1.2,
        "peak_incremental_rss_bytes": 0,
        "arm": suffix,
    }


def _report():
    scenarios = BENCHMARK.quick_scenarios()
    records = []
    for scenario in scenarios:
        records.extend((_record(scenario, 1), _record(scenario, 2)))
    return {
        "quick": True,
        "rounds": BENCHMARK.QUICK_ROUNDS,
        "parallel_workers": 2,
        "scenarios": [BENCHMARK.asdict(scenario) for scenario in scenarios],
        "records": records,
        "environment": {"source_commit": "a" * 40},
    }


def test_scenario_matrices_cover_required_shapes_classes_growth_and_seeds():
    quick = BENCHMARK.quick_scenarios()
    full = BENCHMARK.full_scenarios()

    assert len(quick) == 3 * 2 * 2
    assert len(full) == 3 * 2 * 2 * 3
    assert {(scenario.n_rows, scenario.n_features) for scenario in full} == {
        (32_768, 8),
        (4_096, 128),
        (512, 8),
    }
    assert {scenario.n_classes for scenario in full} == {3, 12}
    assert {scenario.tree_growth for scenario in full} == {"level", "leaf"}
    assert {scenario.seed for scenario in quick} == {0}
    assert {scenario.seed for scenario in full} == {0, 1, 2}


def test_fixture_is_deterministic_finite_and_multiclass():
    scenario = BENCHMARK.quick_scenarios()[0]
    first = BENCHMARK.make_fixture(scenario)
    second = BENCHMARK.make_fixture(scenario)

    for left, right in zip(first.arrays(), second.arrays(), strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()
    assert first.X_train.shape == (scenario.n_rows, scenario.n_features)
    assert set(np.unique(first.y_train)) == set(range(scenario.n_classes))
    assert set(np.unique(first.y_holdout)) == set(range(scenario.n_classes))


def test_valid_quick_report_passes_gate():
    assert BENCHMARK.evaluate_gate(_report()) == []


def test_gate_rejects_missing_and_duplicate_record_identities():
    missing = _report()
    missing["records"].pop()
    assert any("record identities" in error for error in BENCHMARK.evaluate_gate(missing))

    duplicate = _report()
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    assert any(
        "record identities" in error for error in BENCHMARK.evaluate_gate(duplicate)
    )


def test_gate_rejects_thread_count_artifact_and_prediction_mismatch():
    report = _report()
    report["records"][1]["artifact_sha256"] = "different"
    report["records"][1]["prediction_sha256"] = "different"

    failures = BENCHMARK.evaluate_gate(report)
    assert any("artifact" in failure for failure in failures)
    assert any("prediction" in failure for failure in failures)


def test_gate_rejects_quality_round_worker_and_finite_failures():
    report = _report()
    parallel = report["records"][1]
    parallel["multiclass_log_loss"] = 1.3
    parallel["completed_rounds"] = 2
    parallel["resolved_workers"] = 3
    parallel["predictions_finite"] = False

    failures = BENCHMARK.evaluate_gate(report)
    assert any("quality" in failure or "baseline" in failure for failure in failures)
    assert any("round" in failure for failure in failures)
    assert any("worker" in failure for failure in failures)
    assert any("finite" in failure for failure in failures)


def test_gate_requires_an_eligible_high_class_parallel_record():
    report = _report()
    for record in report["records"]:
        record["class_parallel_eligible"] = False

    assert any(
        "eligible high-class" in failure
        for failure in BENCHMARK.evaluate_gate(report)
    )


def test_render_and_cli_write_json_and_markdown(monkeypatch, tmp_path):
    report = _report()
    monkeypatch.setattr(BENCHMARK, "run_benchmark", lambda quick: report)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    exit_code = BENCHMARK.main(
        [
            "--quick",
            "--gate",
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Multiclass Parallelism Benchmark" in markdown
    assert "Artifact SHA-256" in markdown


def test_ci_runs_contract_and_quick_gate():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "python -m pytest benchmarks/tests/test_multiclass_parallelism_benchmark.py -q"
        in workflow
    )
    assert (
        "python benchmarks/multiclass_parallelism_benchmark.py --quick --gate"
        in workflow
    )
