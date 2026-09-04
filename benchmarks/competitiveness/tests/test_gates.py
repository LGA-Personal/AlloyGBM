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
    evaluate_catastrophic_regression,
    evaluate_claim,
)
from benchmarks.competitiveness.schema import BenchmarkRecordV1, SCHEMA_VERSION
from benchmarks.competitiveness.schema import BenchmarkSummaryV1
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


def test_speed_exact_one_percent_metric_guardrail_is_not_rejected() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.9, metric=1.01))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert evaluate_speed([current], [baseline]).status == "pass"


def test_quality_exact_ten_percent_fit_guardrail_is_allowed() -> None:
    current = [aggregate_records(records("run-current", "alloygbm", fit=1.1, metric=0.9, scenario=name))[0] for name in ("a", "b")]
    baseline = [aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario=name))[0] for name in ("a", "b")]
    assert evaluate_quality(current, baseline).status == "pass"


def test_quality_exact_relative_floor_and_noise_do_not_count_but_just_above_does() -> None:
    baseline = [aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario=name))[0] for name in ("a", "b")]
    exact_floor = [aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=0.995, scenario=name))[0] for name in ("a", "b")]
    assert evaluate_quality(exact_floor, baseline).status == "defer"
    just_above = [aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=0.9949, scenario=name))[0] for name in ("a", "b")]
    assert evaluate_quality(just_above, baseline).status == "pass"
    exact_noise = [replace(item, metric_median=0.99, metric_mad=0.01) for item in just_above]
    assert evaluate_quality(exact_noise, baseline).status == "defer"
    just_above_noise = [replace(item, metric_median=0.9899, metric_mad=0.01) for item in exact_noise]
    assert evaluate_quality(just_above_noise, baseline).status == "pass"


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


def test_quality_requires_two_distinct_meaningful_scenarios_not_two_wins_in_one() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=0.9, threads=threads, scenario="a"))[0]
        for threads in (1, 4)
    ] + [aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0, scenario="b"))[0]]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, threads=threads, scenario="a"))[0]
        for threads in (1, 4)
    ] + [aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="b"))[0]]
    result = evaluate_quality(current, baseline)
    assert result.status == "defer"


@pytest.mark.parametrize("gate", [evaluate_speed, evaluate_quality, catastrophic_regressions, evaluate_default_policy])
def test_public_gates_reject_mixed_current_run_ids_even_for_disjoint_scenarios(gate) -> None:
    current = [
        aggregate_records(records("run-a", "alloygbm", fit=0.8, metric=0.9, scenario="a"))[0],
        aggregate_records(records("run-b", "alloygbm", fit=0.8, metric=0.9, scenario="b"))[0],
    ]
    baseline = [
        aggregate_records(records("base", "alloygbm", fit=1.0, metric=1.0, scenario="a"))[0],
        aggregate_records(records("base", "alloygbm", fit=1.0, metric=1.0, scenario="b"))[0],
    ]
    result = gate(current, baseline)
    if isinstance(result, list):
        assert result[0].status == "insufficient-data"
    else:
        assert result.status == "insufficient-data"


def test_default_policy_requires_one_machine_for_all_libraries_in_each_ranked_slice() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=0.5, machine={"hostname": "a"}))[0],
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=0.7, machine={"hostname": "b"}))[0],
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=0.8, machine={"hostname": "a"}))[0],
        aggregate_records(records("run-base", "lightgbm", fit=1.0, metric=0.7, machine={"hostname": "a"}))[0],
    ]
    assert evaluate_default_policy(current, baseline).status == "insufficient-data"


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


def test_gates_validate_duplicate_summary_repetition_provenance_and_ranks_return_empty() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    malformed = replace(current, raw_repetition_ids=(0, 0, 1, 2, 3))
    assert evaluate_speed([malformed], [baseline]).status == "insufficient-data"
    assert normalized_ranks([malformed]) == {}


def test_duplicate_raw_source_lines_are_invalid_for_gates_and_ranks() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    malformed = replace(current, raw_line_numbers=(1, 1, 2, 3, 4))
    assert evaluate_speed([malformed], [baseline]).status == "insufficient-data"
    assert catastrophic_regressions([malformed], [baseline])[0].status == "insufficient-data"
    assert normalized_ranks([malformed]) == {}


def test_normalized_ranks_requires_durable_provenance_and_shared_machine() -> None:
    alloy = aggregate_records(records("run", "alloygbm", fit=1.0, metric=1.0))[0]
    light = aggregate_records(records("run", "lightgbm", fit=1.0, metric=2.0))[0]
    assert normalized_ranks([replace(alloy, machine=None), light]) == {}
    assert normalized_ranks([replace(alloy, effective_params=None), light]) == {}
    assert normalized_ranks([alloy, replace(light, machine={"hostname": "other"})]) == {}


def test_cli_baseline_without_claim_reports_insufficient_data(tmp_path: Path) -> None:
    from benchmarks.competitiveness.summarize import _cli
    raw = tmp_path / "raw.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    output = tmp_path / "summary.json"
    rows = records("run-current", "alloygbm", fit=1.0, metric=1.0)
    raw.write_text("\n".join(item.to_json() for item in rows) + "\n")
    baseline.write_text("\n".join(item.to_json() for item in rows) + "\n")
    assert _cli([str(raw), "--baseline", str(baseline), "--json-output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "insufficient-data"


def test_hand_built_summary_missing_provenance_is_insufficient() -> None:
    current = BenchmarkSummaryV1.from_dict(aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0))[0].to_dict())
    baseline = replace(current, run_id="run-base", fit_median_seconds=1.0, machine=None)
    assert evaluate_speed([current], [baseline]).status == "insufficient-data"


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


def test_parameter_presence_differs_from_explicit_null() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0, effective_params={"mechanism": None}))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, effective_params={}))[0]
    assert evaluate_speed([current], [baseline]).status == "insufficient-data"
    assert evaluate_speed([current], [baseline], allowed_param_differences=["mechanism"]).status == "pass"


def test_competitor_parameter_difference_cannot_be_allowlisted() -> None:
    current = [aggregate_records(records("run-current", library, fit=1.0, metric=value, effective_params={"depth": 7}))[0] for library, value in (("alloygbm", 1.0), ("lightgbm", 0.8))]
    baseline = [aggregate_records(records("run-base", library, fit=1.0, metric=value, effective_params={"depth": 6}))[0] for library, value in (("alloygbm", 1.0), ("lightgbm", 0.8))]
    assert evaluate_default_policy(current, baseline, allowed_param_differences=["depth"]).status == "insufficient-data"


@pytest.mark.parametrize("gate", [evaluate_speed, evaluate_quality, evaluate_default_policy])
def test_public_gate_rejects_nonpositive_minimum_repetitions(gate) -> None:
    with pytest.raises(ValueError, match="minimum_repetitions"):
        gate([], [], minimum_repetitions=0)


def test_bare_string_and_blank_allow_list_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="allowed_param_differences"):
        evaluate_speed([], [], allowed_param_differences="depth")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_speed([], [], allowed_param_differences=["  "])


def test_catastrophic_public_result_records_allowed_keys_at_top_level() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    result = evaluate_catastrophic_regression([current], [baseline], allowed_param_differences=["depth"])
    assert result.to_dict()["evidence"]["allowed_param_differences"] == ["depth"]


@pytest.mark.parametrize("bad_summary", [None, {}, object()])
@pytest.mark.parametrize("claim", ["speed", "quality", "default-policy"])
def test_gate_claims_return_insufficient_for_malformed_summary_inputs(bad_summary: object, claim: str) -> None:
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    result = evaluate_claim(claim, [bad_summary], [baseline])  # type: ignore[list-item]
    assert result.status == "insufficient-data"


def test_public_catastrophic_gate_wrappers_return_insufficient_for_malformed_summary() -> None:
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    for bad_summary in (None, {}, replace(baseline, run_id=123), replace(baseline, run_id=["unhashable"])):
        results = catastrophic_regressions([bad_summary], [baseline])  # type: ignore[list-item]
        assert results[0].status == "insufficient-data"
        collapsed = evaluate_catastrophic_regression([bad_summary], [baseline])  # type: ignore[list-item]
        assert collapsed.status == "insufficient-data"


@pytest.mark.parametrize("gate", [evaluate_speed, evaluate_quality, evaluate_default_policy])
def test_direct_public_gates_return_insufficient_for_invalid_summary_fields(gate) -> None:
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    malformed = replace(baseline, run_id=["unhashable"])
    assert gate([malformed], [baseline]).status == "insufficient-data"


def test_default_policy_rejects_catastrophe_when_competitor_context_is_missing_or_mismatched() -> None:
    current_alloy = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    baseline_alloy = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    current_light = aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0]
    mismatched_baseline_light = aggregate_records(
        records("run-base", "lightgbm", fit=1.0, metric=1.0, machine={"hostname": "other"})
    )[0]
    assert evaluate_default_policy([current_alloy], [baseline_alloy]).status == "reject"
    assert evaluate_default_policy(
        [current_alloy, current_light], [baseline_alloy, mismatched_baseline_light]
    ).status == "reject"


def test_default_policy_preserves_catastrophe_with_another_missing_or_invalid_candidate_slice() -> None:
    current_catastrophe = aggregate_records(
        records("run-current", "alloygbm", fit=2.1, metric=1.1, scenario="catastrophe")
    )[0]
    baseline_catastrophe = aggregate_records(
        records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="catastrophe")
    )[0]
    current_invalid = replace(
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0, scenario="invalid"))[0],
        machine=None,
    )
    baseline_invalid = aggregate_records(
        records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="invalid")
    )[0]
    result = evaluate_default_policy(
        [current_catastrophe, current_invalid], [baseline_catastrophe, baseline_invalid]
    )
    assert result.status == "reject"


def test_default_policy_isolates_catastrophe_from_malformed_competitor() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert evaluate_default_policy([current, None], [baseline, {}]).status == "reject"


def test_default_policy_isolates_catastrophe_from_competitor_run_id() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    current_competitor = aggregate_records(records("other-run", "lightgbm", fit=1.0, metric=1.0))[0]
    baseline_competitor = aggregate_records(records("other-base", "lightgbm", fit=1.0, metric=1.0))[0]
    assert evaluate_default_policy(
        [current, current_competitor], [baseline, baseline_competitor]
    ).status == "reject"


def test_default_policy_isolates_catastrophe_from_malformed_competitor_params() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    bad_current = replace(
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
        effective_params={"bad": {1, 2}},
    )
    assert evaluate_default_policy([current, bad_current], [baseline]).status == "reject"


@pytest.mark.parametrize("outer", [None, 1, {}, "summary"])
def test_gate_apis_return_insufficient_for_invalid_outer_summary_inputs(outer: object) -> None:
    valid = aggregate_records(records("run", "alloygbm", fit=1.0, metric=1.0))[0]
    for gate in (evaluate_speed, evaluate_quality, evaluate_default_policy):
        assert gate(outer, [valid]).status == "insufficient-data"  # type: ignore[arg-type]
        assert gate([valid], outer).status == "insufficient-data"  # type: ignore[arg-type]
    assert catastrophic_regressions(outer, [valid])[0].status == "insufficient-data"  # type: ignore[arg-type]
    assert evaluate_catastrophic_regression(outer, [valid]).status == "insufficient-data"  # type: ignore[arg-type]
    assert normalized_ranks(outer) == {}  # type: ignore[arg-type]


def test_gate_apis_return_insufficient_for_single_summary_outer_input() -> None:
    valid = aggregate_records(records("run", "alloygbm", fit=1.0, metric=1.0))[0]
    assert evaluate_speed(valid, [valid]).status == "insufficient-data"
    assert catastrophic_regressions([valid], valid)[0].status == "insufficient-data"
    assert normalized_ranks(valid) == {}


@pytest.mark.parametrize("target", ["dense_regression", 1, {}, ["dense_regression", "  "], [None]])
def test_speed_and_quality_reject_invalid_target_scenarios(target: object) -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=0.8, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert evaluate_speed([current], [baseline], target_scenarios=target).status == "insufficient-data"  # type: ignore[arg-type]
    assert evaluate_quality([current], [baseline], target_scenarios=target).status == "insufficient-data"  # type: ignore[arg-type]


def test_json_like_parameter_validation_rejects_set_values_in_records_and_summaries() -> None:
    with pytest.raises(ValueError, match="JSON-like|effective_params"):
        from benchmarks.competitiveness.schema import validate_record
        validate_record(record(effective_params={"bad": {1, 2}}))
    summary = aggregate_records(records("run", "alloygbm", fit=1.0, metric=1.0))[0]
    malformed = replace(summary, effective_params={"bad": {1, 2}})
    with pytest.raises(ValueError, match="JSON-like|effective_params"):
        from benchmarks.competitiveness.schema import validate_summary
        validate_summary(malformed)
    baseline = aggregate_records(records("base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert evaluate_speed([malformed], [baseline]).status == "insufficient-data"


@pytest.mark.parametrize("gate", [evaluate_speed, evaluate_quality, evaluate_default_policy])
def test_whole_cohort_gates_reject_unpaired_missing_provenance_competitor(gate) -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    competitor = replace(
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
        machine=None,
        effective_params=None,
    )
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert gate([current, competitor], [baseline]).status == "insufficient-data"


def test_catastrophic_wrappers_require_missing_competitor_provenance_without_catastrophe() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    competitor = replace(
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
        machine=None,
        effective_params=None,
    )
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    results = catastrophic_regressions([current, competitor], [baseline])
    assert results[0].status == "insufficient-data"
    assert evaluate_catastrophic_regression([current, competitor], [baseline]).status == "insufficient-data"


def test_catastrophic_wrappers_preserve_reject_with_missing_competitor_provenance() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    competitor = replace(
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
        machine=None,
        effective_params=None,
    )
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    results = catastrophic_regressions([current, competitor], [baseline])
    assert results[0].status == "reject"
    assert any(result.status == "insufficient-data" for result in results[1:])
    assert evaluate_catastrophic_regression([current, competitor], [baseline]).status == "reject"
    assert evaluate_default_policy([current, competitor], [baseline]).status == "reject"


def test_catastrophic_rejects_unpaired_competitor_from_other_run_without_catastrophe() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    competitor = aggregate_records(records("other-run", "lightgbm", fit=1.0, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert catastrophic_regressions([current, competitor], [baseline])[0].status == "insufficient-data"
    assert evaluate_catastrophic_regression([current, competitor], [baseline]).status == "insufficient-data"


def test_catastrophic_rejects_unpaired_competitor_from_other_run_with_catastrophe() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1))[0]
    competitor = aggregate_records(records("other-run", "lightgbm", fit=1.0, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    results = catastrophic_regressions([current, competitor], [baseline])
    assert results[0].status == "reject"
    assert any(item.status == "insufficient-data" for item in results[1:])
    assert evaluate_catastrophic_regression([current, competitor], [baseline]).status == "reject"


def test_catastrophic_rejects_same_run_unpaired_competitor_without_catastrophe() -> None:
    current = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0]
    competitor = aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0]
    baseline = aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0]
    assert catastrophic_regressions([current, competitor], [baseline])[0].status == "insufficient-data"


def test_catastrophic_rejects_paired_competitor_mismatch_without_catastrophe() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0],
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0],
        aggregate_records(records("run-base", "lightgbm", fit=1.0, metric=1.0, machine={"hostname": "other"}))[0],
    ]
    assert catastrophic_regressions(current, baseline)[0].status == "insufficient-data"


def test_catastrophic_passes_clean_paired_competitors_without_catastrophe() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0))[0],
        aggregate_records(records("run-current", "lightgbm", fit=1.0, metric=1.0))[0],
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0))[0],
        aggregate_records(records("run-base", "lightgbm", fit=1.0, metric=1.0))[0],
    ]
    assert catastrophic_regressions(current, baseline)[0].status == "pass"


def test_catastrophic_mixed_current_runs_keep_valid_catastrophe_and_reject_noncat_is_insufficient() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1, scenario="cat"))[0],
        aggregate_records(records("run-other", "alloygbm", fit=1.0, metric=1.0, scenario="plain"))[0],
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="cat"))[0],
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="plain"))[0],
    ]
    results = catastrophic_regressions(current, baseline)
    assert results[0].status == "reject"
    assert any("run_id" in reason for result in results for reason in result.reasons)
    noncat = catastrophic_regressions(
        [replace(item, fit_median_seconds=1.0, metric_median=1.0) for item in current], baseline
    )
    assert noncat[0].status == "insufficient-data"


def test_catastrophic_mixed_baseline_runs_keep_valid_catastrophe_and_reject_noncat_is_insufficient() -> None:
    current = [
        aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1, scenario="cat"))[0],
        aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0, scenario="plain"))[0],
    ]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="cat"))[0],
        aggregate_records(records("run-other-base", "alloygbm", fit=1.0, metric=1.0, scenario="plain"))[0],
    ]
    results = catastrophic_regressions(current, baseline)
    assert results[0].status == "reject"
    assert any("run_id" in reason for result in results for reason in result.reasons)
    noncat = catastrophic_regressions(
        [replace(item, fit_median_seconds=1.0, metric_median=1.0) for item in current], baseline
    )
    assert noncat[0].status == "insufficient-data"


def test_catastrophic_duplicate_slice_is_unusable_regardless_order_but_other_slice_rejects() -> None:
    duplicate_first = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1, scenario="dup"))[0]
    duplicate_second = aggregate_records(records("run-current", "alloygbm", fit=1.0, metric=1.0, scenario="dup"))[0]
    catastrophe = aggregate_records(records("run-current", "alloygbm", fit=2.1, metric=1.1, scenario="cat"))[0]
    baseline = [
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="dup"))[0],
        aggregate_records(records("run-base", "alloygbm", fit=1.0, metric=1.0, scenario="cat"))[0],
    ]
    for ordered in ([duplicate_first, duplicate_second, catastrophe], [duplicate_second, duplicate_first, catastrophe]):
        results = catastrophic_regressions(ordered, baseline)
        assert results[0].status == "reject"
        assert all("dup|threads=1" not in result.evidence for result in results)
        assert any("duplicate" in reason for result in results[1:] for reason in result.reasons)


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
