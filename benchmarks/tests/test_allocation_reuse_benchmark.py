"""Contract tests for the allocation-reuse A/B benchmark harness."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "allocation_reuse_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "allocation_reuse_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def _result(
    *,
    case: str = "tall_deep-level",
    repetition: int = 0,
    native_seconds: float = 1.0,
    rss_mib: float = 10.0,
    artifact_sha256: str = "a",
    prediction_sha256: str = "b",
    rmse: float = 0.25,
    source_commit: str = "1" * 40,
    extension_sha256: str = "e" * 64,
):
    return BENCHMARK.CaseResult(
        artifact_sha256=artifact_sha256,
        prediction_sha256=prediction_sha256,
        native_seconds=native_seconds,
        rss_mib=rss_mib,
        rmse=rmse,
        case=case,
        shape=case.rsplit("-", 1)[0],
        tree_growth=case.rsplit("-", 1)[1],
        repetition=repetition,
        source_commit=source_commit,
        extension_sha256=extension_sha256,
    )


def _pairs(*, candidate_time: float = 0.98, candidate_rss: float = 9.8):
    pairs = []
    for case in ("tall_deep-level", "wide_deep-leaf", "short_wide-level"):
        for repetition in range(3):
            baseline = _result(case=case, repetition=repetition)
            candidate = replace(
                baseline,
                native_seconds=candidate_time,
                rss_mib=candidate_rss,
                source_commit="2" * 40,
                extension_sha256="f" * 64,
            )
            pairs.append(BENCHMARK.PairedResult(baseline, candidate))
    return pairs


def test_fixture_generation_is_deterministic_and_contains_missing_values():
    case = BENCHMARK.quick_cases()[0]

    first = BENCHMARK.make_fixture(case)
    second = BENCHMARK.make_fixture(case)

    for left, right in zip(first.arrays(), second.arrays(), strict=True):
        np.testing.assert_array_equal(left, right)
    assert first.X_train.shape == (case.n_rows, case.n_features)
    assert first.X_eval.shape == (case.n_eval_rows, case.n_features)
    assert np.isnan(first.X_train).any()
    assert np.isnan(first.X_eval).any()
    assert np.isfinite(first.y_train).all()
    assert np.isfinite(first.y_eval).all()


def test_profiles_cover_required_shapes_and_growth_modes():
    quick = BENCHMARK.quick_cases()
    full = BENCHMARK.full_cases()

    assert {case.shape for case in quick} == {
        "tall_deep",
        "wide_deep",
        "short_wide",
        "shallow_tall",
    }
    assert {case.tree_growth for case in quick} == {"level", "leaf"}
    assert {(case.shape, case.tree_growth) for case in full} == {
        (shape, growth)
        for shape in ("tall_deep", "wide_deep", "short_wide", "shallow_tall")
        for growth in ("level", "leaf")
    }
    assert max(case.n_rows for case in full) > max(case.n_rows for case in quick)
    assert max(case.n_features for case in full) > max(
        case.n_features for case in quick
    )
    assert BENCHMARK.profile_repetitions("quick") == 1
    assert BENCHMARK.profile_repetitions("full") >= 5


def test_worker_command_is_isolated_and_bound_to_runtime_worktree(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    workdir = tmp_path / "source"
    runtime = BENCHMARK.RuntimeSpec(
        name="baseline",
        python=python,
        workdir=workdir,
        source_commit="1" * 40,
    )
    case = BENCHMARK.quick_cases()[0]

    invocation = BENCHMARK.build_worker_invocation(
        runtime,
        case,
        profile="quick",
        repetition=0,
        n_jobs=2,
    )

    assert invocation.command[:2] == (str(python), "-I")
    assert str(BENCHMARK_PATH) in invocation.command
    assert invocation.cwd == workdir
    assert invocation.value_after("--runtime-workdir") == str(workdir)
    assert invocation.value_after("--expected-source-commit") == "1" * 40
    assert "PYTHONPATH" not in invocation.env
    assert "PYTHONHOME" not in invocation.env


def test_worker_output_parser_accepts_one_complete_json_record():
    expected = _result()

    parsed = BENCHMARK.parse_worker_output(
        json.dumps(expected.to_dict()),
        runtime_name="baseline",
        expected_case=expected.case,
        expected_repetition=0,
        expected_source_commit="1" * 40,
    )

    assert parsed == expected


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("noise\n{}", "single JSON object"),
        (json.dumps(_result(source_commit="2" * 40).to_dict()), "source commit"),
    ],
)
def test_worker_output_parser_rejects_noise_and_wrong_source(stdout, message):
    with pytest.raises(ValueError, match=message):
        BENCHMARK.parse_worker_output(
            stdout,
            runtime_name="baseline",
            expected_case="tall_deep-level",
            expected_repetition=0,
            expected_source_commit="1" * 40,
        )


def test_pair_requires_exact_artifact_prediction_and_rmse_equivalence():
    baseline = _result()
    candidate = replace(
        baseline,
        native_seconds=0.9,
        rss_mib=8.0,
        source_commit="2" * 40,
        extension_sha256="f" * 64,
    )
    evaluation = BENCHMARK.evaluate_pair(baseline, candidate)

    assert evaluation.equivalent
    assert evaluation.timing_ratio == pytest.approx(0.9)
    assert evaluation.rss_ratio == pytest.approx(0.8)

    for changed in (
        replace(candidate, artifact_sha256="different"),
        replace(candidate, prediction_sha256="different"),
        replace(candidate, rmse=np.nextafter(candidate.rmse, np.inf)),
    ):
        mismatch = BENCHMARK.evaluate_pair(baseline, changed)
        assert not mismatch.equivalent
        assert mismatch.failures


def test_full_gates_use_case_medians_and_aggregate_budgets():
    passing = BENCHMARK.evaluate_gates(_pairs(), profile="full")
    assert passing.failures == ()
    assert passing.aggregate_timing_ratio == pytest.approx(0.98)
    assert passing.aggregate_rss_ratio == pytest.approx(0.98)

    timing_failure = BENCHMARK.evaluate_gates(
        _pairs(candidate_time=1.031, candidate_rss=9.9), profile="full"
    )
    assert any("aggregate timing" in failure for failure in timing_failure.failures)

    rss_failure = BENCHMARK.evaluate_gates(
        _pairs(candidate_time=0.99, candidate_rss=10.51), profile="full"
    )
    assert any("aggregate RSS" in failure for failure in rss_failure.failures)

    no_deep_improvement = BENCHMARK.evaluate_gates(
        _pairs(candidate_time=1.0, candidate_rss=10.0), profile="full"
    )
    assert any("deep pressure" in failure for failure in no_deep_improvement.failures)


def test_quick_gate_checks_equivalence_without_noisy_performance_claims():
    pairs = _pairs(candidate_time=4.0, candidate_rss=40.0)

    evaluation = BENCHMARK.evaluate_gates(pairs, profile="quick")

    assert evaluation.failures == ()
    assert evaluation.performance_gated is False


def test_runtime_validation_rejects_shared_native_binary_for_distinct_commits():
    baseline = BENCHMARK.RuntimeSpec(
        "baseline", Path("/baseline/python"), Path("/baseline"), "1" * 40
    )
    candidate = BENCHMARK.RuntimeSpec(
        "candidate", Path("/candidate/python"), Path("/candidate"), "2" * 40
    )
    pair = BENCHMARK.PairedResult(
        _result(extension_sha256="e" * 64),
        _result(source_commit="2" * 40, extension_sha256="e" * 64),
    )

    with pytest.raises(ValueError, match="same native extension"):
        BENCHMARK.validate_runtime_pair(baseline, candidate, [pair])


def test_markdown_renders_runtime_identity_aggregate_gates_and_digests():
    report = BENCHMARK.build_report(
        profile="full",
        baseline=BENCHMARK.RuntimeSpec(
            "baseline", Path("/baseline/python"), Path("/baseline"), "1" * 40
        ),
        candidate=BENCHMARK.RuntimeSpec(
            "candidate", Path("/candidate/python"), Path("/candidate"), "2" * 40
        ),
        pairs=_pairs(),
        n_jobs=4,
    )

    markdown = BENCHMARK.render_markdown(report)

    assert "# Allocation Reuse Benchmark" in markdown
    assert "`1111111111111111111111111111111111111111`" in markdown
    assert "Aggregate timing ratio" in markdown
    assert "Aggregate RSS ratio" in markdown
    assert "Artifact SHA-256" in markdown
    assert "Prediction SHA-256" in markdown
    assert "per-case values are descriptive" in markdown


def test_ci_runs_allocation_contract_and_quick_self_consistency_gate():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "python -m pytest benchmarks/tests/test_allocation_reuse_benchmark.py -q"
        in workflow
    )
    assert (
        "python benchmarks/allocation_reuse_benchmark.py --quick --gate"
        in workflow
    )
