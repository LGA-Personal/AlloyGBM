"""Contract tests for the PR #135 DRO performance benchmark harness."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "dro_performance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dro_performance_module", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_full_specs_cover_shape_and_task_matrix():
    specs = MODULE.full_specs()
    assert {(s.shape, s.task_family) for s in specs} >= {
        ("small-narrow", "regression"),
        ("small-wide", "regression"),
        ("tall-narrow", "regression"),
        ("tall-wide", "regression"),
        ("medium", "binary"),
        ("small-wide", "multiclass"),
        ("tall-narrow", "ranking"),
    }
    assert {s.variant for s in specs if s.task_family == "ranking"} == {
        "small-query",
        "large-query",
    }


def test_quick_specs_cap_rows_features_and_rounds_without_dropping_families():
    specs = MODULE.quick_specs()
    assert {s.task_family for s in specs} == {
        "regression",
        "binary",
        "multiclass",
        "ranking",
    }
    assert all(s.rows <= 768 and s.features <= 48 and s.rounds <= 12 for s in specs)


def test_compare_results_rejects_quality_drift():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=1e-3, time_ratio=0.7)
    with pytest.raises(ValueError, match="quality equivalence"):
        MODULE.compare_results(baseline, candidate)


def test_compare_results_enforces_shape_regression_limit():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.051)
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("shape regression" in reason for reason in summary.reasons)


def test_compare_results_rejects_duplicate_keys():
    records = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.compare_results(records, [*records, records[0]])


def test_compare_results_rejects_missing_keys():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    with pytest.raises(ValueError, match="keys"):
        MODULE.compare_results(baseline, candidate[:-1])


def test_compare_results_rejects_non_finite_values():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate[0] = replace(candidate[0], fit_seconds=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        MODULE.compare_results(baseline, candidate)


def test_compare_results_rejects_metric_direction_changes():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=0.7)
    flipped = [replace(record, higher_is_better=not record.higher_is_better) for record in candidate]
    with pytest.raises(ValueError, match="metric direction"):
        MODULE.compare_results(baseline, flipped)


def test_write_results_is_deterministic_and_excludes_no_timing_identity_fields(tmp_path):
    records = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = {"seeds": [0, 1], "arms": ["standard", "dro"]}
    MODULE.write_results(first, records, arguments, git_head="candidate")
    MODULE.write_results(second, records, arguments, git_head="candidate")
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["production_base"] == "2b2e3ef"
    assert payload["git_head"] == "candidate"
    assert payload["records"] == sorted(
        payload["records"],
        key=lambda record: (
            record["arm"],
            record["dataset"],
            record["task_family"],
            record["shape"],
            record["seed"],
            record["primary_metric"],
        ),
    )


def test_compare_results_accepts_fifteen_percent_median_fit_time_fallback():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=0.84)
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert summary.dro_fit_improvement >= 0.15


def test_compare_results_rejects_standard_arm_sentinel_regression():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(
        metric_delta=0.0,
        time_ratio=0.7,
        standard_time_ratio=1.031,
    )
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("standard-arm" in reason for reason in summary.reasons)


def test_standard_sentinel_gates_on_case_median_and_exposes_record_outlier():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=0.7)
    candidate = [
        replace(
            record,
            fit_seconds=1.2,
        )
        if record.arm == "standard"
        and record.dataset == "synthetic-small-narrow"
        and record.seed == 0
        else record
        for record in candidate
    ]

    summary = MODULE.compare_results(baseline, candidate)

    assert summary.passed
    assert summary.standard_case_time_ratios["synthetic-small-narrow"] == 1.0
    assert summary.worst_standard_case_ratio == 1.0
    assert summary.worst_standard_record_ratio == 1.2


def test_timed_fit_seconds_excludes_prediction_latency(monkeypatch):
    def measure(prediction_duration):
        current = 0.0

        def clock():
            return current

        def fit():
            nonlocal current
            current += 1.0

        def predict():
            nonlocal current
            current += prediction_duration
            return "prediction"

        monkeypatch.setattr(MODULE.time, "perf_counter", clock)
        return MODULE._timed_fit_and_predict(fit, predict)

    fast_fit, fast_predict, fast_value = measure(0.1)
    slow_fit, slow_predict, slow_value = measure(10.0)

    assert fast_fit == slow_fit == 1.0
    assert fast_predict == pytest.approx(0.1)
    assert slow_predict == pytest.approx(10.0)
    assert fast_value == slow_value == "prediction"


def test_rejected_prototype_trials_remain_machine_readable():
    baseline = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=1.0)
    candidate = MODULE.synthetic_paired_records(metric_delta=0.0, time_ratio=0.7)
    summary = MODULE.compare_results(
        baseline,
        candidate,
        rejected_trials=MODULE.REJECTED_TRIALS,
    )

    assert [trial["label"] for trial in summary.rejected_trials] == [
        "initial four-lane scanner",
        "invariant-hoisting scanner",
    ]
