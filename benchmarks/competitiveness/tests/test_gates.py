from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.competitiveness.gates import (
    GateResult,
    evaluate_default_policy,
    evaluate_quality,
    evaluate_speed,
    normalized_ranks,
    catastrophic_regressions,
)
from benchmarks.competitiveness.schema import BenchmarkRecordV1, SCHEMA_VERSION
from benchmarks.competitiveness.summarize import aggregate_records, render_markdown, summarize_file


def record(**changes: object) -> BenchmarkRecordV1:
    value: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "run_id": "run-current",
        "repetition": 0,
        "dataset_sha256": "a" * 64,
        "scenario": "dense_regression",
        "task": "regression",
        "library": "alloygbm",
        "library_version": "1.0.0",
        "git_sha": "abc",
        "seed": 20260904,
        "threads": 1,
        "effective_params": {"depth": 6},
        "input_representation": "dense",
        "preprocessing_seconds": 0.1,
        "fit_seconds": 1.0,
        "predict_seconds": 0.1,
        "peak_rss_bytes": 100,
        "metric_name": "rmse",
        "metric_value": 1.0,
        "rounds_completed": 10,
        "machine": {"hostname": "host-a", "platform": "linux"},
        "profile": None,
    }
    value.update(changes)
    return BenchmarkRecordV1(**value)  # type: ignore[arg-type]


def records(run_id: str, library: str, *, fit: float, metric: float, scenario: str = "dense_regression", n: int = 5, **changes: object) -> list[BenchmarkRecordV1]:
    return [
        record(run_id=run_id, library=library, fit_seconds=fit, metric_value=metric, scenario=scenario, repetition=i, **changes)
        for i in range(n)
    ]


def test_aggregate_uses_median_unscaled_mad_and_raw_provenance() -> None:
    raw = [replace(item, fit_seconds=value, metric_value=value) for item, value in zip(records("run-current", "alloygbm", fit=1.0, metric=1.0), [1.0, 2.0, 3.0, 4.0, 100.0])]
    summary = aggregate_records(raw)[0]
    assert summary.fit_median_seconds == 3.0
    assert summary.fit_mad_seconds == 1.0
    assert summary.metric_median == 3.0
    assert summary.raw_repetition_ids == (0, 1, 2, 3, 4)
    markdown = render_markdown([summary], raw_path="raw.jsonl")
    assert "raw.jsonl#L" not in markdown


def test_file_summary_links_global_jsonl_lines_across_groups(tmp_path: Path) -> None:
    raw = records("run-current", "alloygbm", fit=1.0, metric=1.0, n=3)
    raw += records("run-current", "alloygbm", fit=1.0, metric=1.0, scenario="binary", n=3)
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(item.to_json() for item in raw) + "\n")
    summaries = summarize_file(path)
    markdown = render_markdown(summaries, raw_path=path)
    assert "raw.jsonl#L1" in markdown
    assert "raw.jsonl#L4" in markdown and "raw.jsonl#L6" in markdown


def test_in_memory_summary_without_line_numbers_does_not_invent_anchors() -> None:
    summary = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    markdown = render_markdown([summary], raw_path="raw.jsonl")
    assert "raw.jsonl#L" not in markdown


def test_aggregate_enforces_minimum_distinct_repetitions() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0, n=3), minimum_repetitions=5)


@pytest.mark.parametrize(
    ("metric_name", "reference", "candidate", "expected"),
    [("rmse", 0.0, 0.1, 100_000_000_000.0), ("r2", 0.5, 0.6, -0.2)],
)
def test_direction_aware_metric_regression_handles_zero_reference(metric_name: str, reference: float, candidate: float, expected: float) -> None:
    from benchmarks.competitiveness.gates import relative_metric_regression
    assert relative_metric_regression(metric_name, candidate, reference) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("candidate_fit", "reference_fit", "candidate_metric", "status"),
    [(0.90, 1.0, 1.0, "pass"), (0.899, 1.0, 1.011, "reject"), (0.95, 1.0, 1.0, "defer")],
)
def test_speed_gate_has_exact_ten_percent_and_one_percent_boundaries(candidate_fit: float, reference_fit: float, candidate_metric: float, status: str) -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=candidate_fit, metric=candidate_metric))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=reference_fit, metric=1.0))[0]
    result = evaluate_speed([current], [baseline])
    assert result.status == status


def test_quality_requires_two_noise_clearing_scenarios_and_allows_quality_first_fit_cost() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=1.2, metric=0.9, scenario=scenario))[0]
        for scenario in ("a", "b")
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario=scenario))[0]
        for scenario in ("a", "b")
    ]
    assert evaluate_quality(current, baseline).status == "reject"
    assert evaluate_quality(current, baseline, quality_first=True).status == "pass"
    assert evaluate_quality(current[:1], baseline[:1]).status == "insufficient-data"


def test_quality_requires_two_distinct_scenarios_not_two_thread_slices() -> None:
    current = [aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=0.9, threads=threads))[0] for threads in (1, 4)]
    baseline = [aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, threads=threads))[0] for threads in (1, 4)]
    assert evaluate_quality(current, baseline).status == "insufficient-data"


def test_requested_target_scenario_missing_from_one_side_is_insufficient() -> None:
    current = [aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0))[0]]
    baseline = [aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]]
    result = evaluate_speed(current, baseline, target_scenarios=["dense_regression", "missing"])
    assert result.status == "insufficient-data"
    assert any("missing" in reason for reason in result.reasons)


def test_normalized_ranks_average_ties_and_sole_library_is_insufficient() -> None:
    summaries = [
        aggregate_records(records("run", "alloygbm", fit=1, metric=1.0))[0],
        aggregate_records(records("run", "lightgbm", fit=1, metric=1.0))[0],
        aggregate_records(records("run", "xgboost", fit=1, metric=2.0))[0],
    ]
    ranks = normalized_ranks(summaries)
    assert ranks["alloygbm"] == pytest.approx(0.25)
    assert normalized_ranks(summaries[:1]) == {}


def test_duplicate_summary_slice_is_insufficient_for_gate_and_rank() -> None:
    one = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    reference = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    result = evaluate_speed([one, one], [reference])
    assert result.status == "insufficient-data"
    assert "duplicate" in " ".join(result.reasons)
    assert normalized_ranks([one, one]) == {}


@pytest.mark.parametrize(("metric", "fit", "status"), [(1.05, 2.0, "pass"), (1.0501, 2.0, "reject"), (1.0, 2.0001, "reject")])
def test_catastrophic_regression_uses_strict_boundaries(metric: float, fit: float, status: str) -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=fit, metric=metric))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert catastrophic_regressions([current], [baseline])[0].status == status


def test_default_policy_passes_only_on_strictly_better_candidate_rank() -> None:
    current = [aggregate_records(records("run-current", "alloygbm", fit=1, metric=0.5))[0], aggregate_records(records("run-current", "lightgbm", fit=1, metric=0.7))[0]]
    baseline = [aggregate_records(records("run-base", "alloygbm", fit=1, metric=0.8))[0], aggregate_records(records("run-base", "lightgbm", fit=1, metric=0.7))[0]]
    assert evaluate_default_policy(current, baseline).status == "pass"


def test_serialized_summary_retains_provenance_for_mismatch_rejection() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    current = type(current).from_json(current.to_json())
    baseline = type(baseline).from_json(replace(baseline, machine={"hostname": "other"}).to_json())
    assert current.machine == {"hostname": "host-a", "platform": "linux"}
    assert current.effective_params == {"depth": 6}
    assert evaluate_speed([current], [baseline]).status == "insufficient-data"


def test_candidate_allowlisted_top_level_parameter_can_differ_but_other_keys_cannot() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0, effective_params={"depth": 7, "mechanism": "new"}))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, effective_params={"depth": 6, "mechanism": "new"}))[0]
    assert evaluate_speed([current], [baseline], allowed_param_differences=["depth"]).status == "pass"
    assert evaluate_speed([current], [baseline], allowed_param_differences=["mechanism"]).status == "insufficient-data"


def test_competitor_parameter_difference_cannot_be_allowlisted() -> None:
    current = [aggregate_records(records("run-current", library, fit=1.0, metric=value, effective_params={"depth": 7}))[0] for library, value in (("alloygbm", 1.0), ("lightgbm", 0.8))]
    baseline = [aggregate_records(records("run-base", library, fit=1.0, metric=value, effective_params={"depth": 6}))[0] for library, value in (("alloygbm", 1.0), ("lightgbm", 0.8))]
    assert evaluate_default_policy(current, baseline, allowed_param_differences=["depth"]).status == "insufficient-data"


def test_gate_result_is_serializable() -> None:
    result = GateResult("speed", "defer", ("missed threshold",), {"slice": {"improvement": 0.05}})
    assert json.loads(result.to_json())["status"] == "defer"


def test_deep_scaling_manifest_has_exact_27_scenarios() -> None:
    import yaml
    path = Path(__file__).parents[1] / "manifests" / "deep_scaling.yaml"
    manifest = yaml.safe_load(path.read_text())
    assert manifest["timed_repetitions"] == 3
    assert len(manifest["scenarios"]) == 27
    expected = {
        (rows, features, depth)
        for rows in (100000, 500000, 1000000)
        for features in (20, 100, 500)
        for depth in (4, 8, 12)
    }
    actual = {(item["rows"], item["features"], item["depth"]) for item in manifest["scenarios"]}
    assert actual == expected
    assert len({item["name"] for item in manifest["scenarios"]}) == 27
    for item in manifest["scenarios"]:
        assert item["name"] == f"dense_r{item['rows']}_f{item['features']}_d{item['depth']}"
        assert item == {
            "name": item["name"], "task": "regression", "rows": item["rows"],
            "features": item["features"], "rounds": 200, "depth": item["depth"],
            "metric": "rmse", "input_representation": "dense",
        }
