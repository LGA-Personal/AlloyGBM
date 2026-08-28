"""Contract tests for the performance-regression benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "perf_regression_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "perf_regression_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def _healthy_record(name="reg-medium", **overrides):
    record = {
        "scenario": name,
        "task": "regression",
        "n_base": 3000,
        "fit_seconds_n": 0.10,
        "fit_seconds_2n": 0.21,
        "fit_scaling_ratio": 2.1,
        "predict_seconds_2n": 0.01,
        "metric": "rmse",
        "metric_value": 0.30,
        "metric_baseline": 1.00,
        "quality_beats_baseline": True,
        "predictions_finite": True,
    }
    record.update(overrides)
    return record


def test_quick_scenarios_are_fixed_and_nonempty():
    scenarios = BENCHMARK.quick_scenarios()
    assert scenarios, "quick scenarios must not be empty"
    # deterministic: same specs on every call
    assert [s.name for s in scenarios] == [s.name for s in BENCHMARK.quick_scenarios()]
    for s in scenarios:
        assert s.task in {"regression", "binary"}
        assert s.n_base > 0 and s.n_estimators > 0


def test_gate_passes_on_healthy_report():
    report = [_healthy_record(), _healthy_record(name="cls-medium", task="binary",
                                                 metric="accuracy", metric_value=0.9,
                                                 metric_baseline=0.5)]
    assert BENCHMARK.evaluate_gate(report) == []


def test_gate_flags_timing_scaling_regression():
    report = [_healthy_record(fit_scaling_ratio=BENCHMARK.SCALING_CEILING + 0.5)]
    failures = BENCHMARK.evaluate_gate(report)
    assert any("scaling ratio" in f for f in failures)


def test_gate_flags_quality_regression():
    report = [_healthy_record(quality_beats_baseline=False)]
    failures = BENCHMARK.evaluate_gate(report)
    assert any("beat baseline" in f or "baseline" in f for f in failures)


def test_gate_flags_non_finite_predictions():
    report = [_healthy_record(predictions_finite=False)]
    failures = BENCHMARK.evaluate_gate(report)
    assert any("finite" in f for f in failures)


def test_render_markdown_contains_every_scenario():
    report = [_healthy_record(), _healthy_record(name="cls-medium")]
    rendered = BENCHMARK.render_markdown(report)
    assert "reg-medium" in rendered and "cls-medium" in rendered


@pytest.mark.slow
def test_quick_run_end_to_end_passes_gate():
    report = BENCHMARK.run_benchmark(quick=True)
    assert report, "quick run produced no records"
    for r in report:
        assert r["predictions_finite"]
        assert np.isfinite(r["fit_scaling_ratio"])
    assert BENCHMARK.evaluate_gate(report) == []


def test_main_quick_gate_returns_zero():
    assert BENCHMARK.main(["--quick", "--gate"]) == 0
