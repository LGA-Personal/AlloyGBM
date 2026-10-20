"""Contract tests for the PR #136 top-k PL benchmark harness."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "pl_topk_performance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pl_topk_performance_module", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_full_specs_cover_required_shapes_and_tasks():
    coverage = {(spec.shape, spec.task_family) for spec in MODULE.full_specs()}
    assert {
        ("small-narrow", "regression"),
        ("small-wide", "regression"),
        ("tall-narrow", "regression"),
        ("tall-wide", "regression"),
        ("medium", "binary"),
        ("small-wide", "multiclass"),
        ("tall-narrow", "ranking"),
    } <= coverage


def test_comparison_requires_exact_k0_production_digests():
    baseline, candidate = MODULE.synthetic_result_pair()
    k0_index = next(index for index, record in enumerate(candidate) if record.arm == "k0")
    candidate[k0_index] = replace(candidate[k0_index], artifact_sha256="different")
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("k0 artifact parity" in reason for reason in summary.reasons)


def test_comparison_requires_exact_default_production_digests():
    baseline, candidate = MODULE.synthetic_result_pair()
    default_index = next(index for index, record in enumerate(candidate) if record.arm == "default")
    candidate[default_index] = replace(candidate[default_index], artifact_sha256="different")
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("default artifact parity" in reason for reason in summary.reasons)


def test_opt_in_quality_regression_is_reported_without_rejecting_safe_default():
    baseline, candidate = MODULE.synthetic_result_pair(k8_quality_ratio=1.011)
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert not summary.opt_in_quality_within_one_percent
    assert any("quality" in observation for observation in summary.observations)


def test_opt_in_quality_report_respects_higher_is_better_metrics():
    baseline, candidate = MODULE.synthetic_result_pair(
        task_family="ranking",
        primary_metric="ndcg",
        higher_is_better=True,
        k8_quality_ratio=0.989,
    )
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert not summary.opt_in_quality_within_one_percent
    assert any("quality" in observation for observation in summary.observations)


def test_comparison_records_no_pl_friendly_improvement_without_failing():
    baseline, candidate = MODULE.synthetic_result_pair(k8_quality_ratio=1.0)
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert summary.pl_friendly_improvement == 0.0


def test_comparison_reports_opt_in_fixed_round_cost_without_failing():
    baseline, candidate = MODULE.synthetic_result_pair(features=16, k8_time_ratio=3.01)
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert summary.median_opt_in_time_ratio == pytest.approx(3.01)


def test_comparison_reports_worst_opt_in_case_cost():
    baseline, candidate = MODULE.synthetic_result_pair(features=16, k8_time_ratio=1.0)
    candidate = [
        replace(record, fit_seconds=5.01)
        if record.arm == "k8" and record.seed == 0
        else record
        for record in candidate
    ]
    summary = MODULE.compare_results(baseline, candidate)
    assert summary.passed
    assert summary.worst_opt_in_time_ratio == pytest.approx(5.01)


def test_comparison_rejects_wide_shortlist_without_half_cost_benefit():
    baseline, candidate = MODULE.synthetic_result_pair(
        features=64,
        k8_time_ratio=1.0,
        all_time_ratio=1.9,
    )
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("exhaustive" in reason for reason in summary.reasons)


def test_comparison_rejects_excess_rss():
    baseline, candidate = MODULE.synthetic_result_pair(
        k0_peak_rss_bytes=100 * 1024 * 1024,
        k8_peak_rss_bytes=134 * 1024 * 1024,
    )
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("RSS" in reason for reason in summary.reasons)


def test_comparison_rejects_duplicate_or_missing_keys():
    baseline, candidate = MODULE.synthetic_result_pair()
    duplicate = MODULE.compare_results(baseline + baseline[:1], candidate)
    missing = MODULE.compare_results(baseline, candidate[:-1])
    assert not duplicate.passed
    assert any("duplicate" in reason for reason in duplicate.reasons)
    assert not missing.passed
    assert any("coverage" in reason for reason in missing.reasons)


def test_comparison_rejects_non_finite_values():
    baseline, candidate = MODULE.synthetic_result_pair()
    candidate[0] = replace(candidate[0], primary_value=float("nan"))
    summary = MODULE.compare_results(baseline, candidate)
    assert not summary.passed
    assert any("non-finite" in reason for reason in summary.reasons)


def test_results_round_trip_in_deterministic_order(tmp_path):
    baseline, candidate = MODULE.synthetic_result_pair()
    records = list(reversed(candidate + baseline))
    path = tmp_path / "results.json"
    MODULE.write_results(path, records, argv=["run", "--quick"])
    first = path.read_text()
    loaded = MODULE.read_results(path)
    MODULE.write_results(path, loaded, argv=["run", "--quick"])
    second = path.read_text()
    assert first == second
    parsed = json.loads(first)
    assert parsed["schema_version"] == MODULE.RESULT_SCHEMA_VERSION
    assert loaded == sorted(loaded, key=MODULE.record_sort_key)


def test_subprocess_rss_normalization_uses_bytes():
    assert MODULE.normalize_peak_rss_bytes(1024, platform_name="linux") == 1024 * 1024
    assert MODULE.normalize_peak_rss_bytes(1024, platform_name="darwin") == 1024


def test_rejected_trial_schema_is_machine_readable():
    trial = MODULE.validate_rejected_trial(
        {
            "label": "prototype",
            "reason": "failed cost gate",
            "commit": "deadbeef",
            "metrics": {"fit_ratio": 3.2},
        }
    )
    assert trial["label"] == "prototype"
    with pytest.raises(ValueError, match="metrics"):
        MODULE.validate_rejected_trial({"label": "bad", "reason": "bad", "metrics": {"x": float("nan")}})
