"""Contract tests for the allocation-reuse A/B benchmark harness."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
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
    runtime_name: str = "baseline",
    python_executable: str = "",
    package_path: str = "",
    extension_path: str = "",
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
        runtime_name=runtime_name,
        python_executable=python_executable,
        package_path=package_path,
        extension_path=extension_path,
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
                runtime_name="candidate",
            )
            pairs.append(BENCHMARK.PairedResult(baseline, candidate))
    return pairs


def _runtime(tmp_path: Path, name: str, commit: str):
    workdir = tmp_path / name
    python = workdir / ".venv" / "bin" / "python"
    package = workdir / "bindings" / "python" / "alloygbm"
    python.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    python.touch()
    (package / "__init__.py").touch()
    (package / "_alloygbm.abi3.so").touch()
    return BENCHMARK.RuntimeSpec(name, python, workdir, commit)


def _bound_result(runtime, case, repetition, **changes):
    package = runtime.workdir / "bindings" / "python" / "alloygbm"
    result = _result(
        case=case.name,
        repetition=repetition,
        source_commit=runtime.source_commit,
        extension_sha256=(
            "e" if runtime.source_commit == "1" * 40 else "f"
        )
        * 64,
        runtime_name=runtime.name,
        python_executable=str(runtime.python.absolute()),
        package_path=str(package / "__init__.py"),
        extension_path=str(package / "_alloygbm.abi3.so"),
    )
    return replace(result, **changes)


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
    assert invocation.value_after("--expected-python") == str(python)
    assert invocation.value_after("--expected-source-commit") == "1" * 40
    assert "PYTHONPATH" not in invocation.env
    assert "PYTHONHOME" not in invocation.env


def test_worker_output_parser_accepts_one_complete_bound_json_record(tmp_path):
    runtime = _runtime(tmp_path, "baseline", "1" * 40)
    case = BENCHMARK.quick_cases()[0]
    expected = _bound_result(runtime, case, 0)

    parsed = BENCHMARK.parse_worker_output(
        json.dumps(expected.to_dict()),
        runtime=runtime,
        expected_case=expected.case,
        expected_repetition=0,
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
    runtime = BENCHMARK.RuntimeSpec(
        "baseline", Path("/baseline/python"), Path("/baseline"), "1" * 40
    )
    with pytest.raises(ValueError, match=message):
        BENCHMARK.parse_worker_output(
            stdout,
            runtime=runtime,
            expected_case="tall_deep-level",
            expected_repetition=0,
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


def test_full_configuration_requires_distinct_runtime_identity(tmp_path):
    baseline = _runtime(tmp_path, "baseline", "1" * 40)
    candidate = _runtime(tmp_path, "candidate", "2" * 40)
    BENCHMARK.validate_run_configuration(
        profile="full", baseline=baseline, candidate=candidate, repetitions=3
    )

    invalid_pairs = (
        (baseline, replace(candidate, python=baseline.python)),
        (baseline, replace(candidate, workdir=baseline.workdir)),
        (baseline, replace(candidate, source_commit=baseline.source_commit)),
    )
    for left, right in invalid_pairs:
        with pytest.raises(ValueError, match="distinct"):
            BENCHMARK.validate_run_configuration(
                profile="full", baseline=left, candidate=right, repetitions=3
            )


def test_quick_configuration_is_the_only_self_comparison_mode(tmp_path):
    candidate = _runtime(tmp_path, "candidate", "2" * 40)
    baseline = replace(candidate, name="baseline")

    BENCHMARK.validate_run_configuration(
        profile="quick", baseline=baseline, candidate=candidate, repetitions=1
    )
    with pytest.raises(ValueError, match="self-comparison"):
        BENCHMARK.validate_run_configuration(
            profile="full", baseline=baseline, candidate=candidate, repetitions=3
        )


def test_worker_result_must_bind_to_declared_interpreter_and_worktree(tmp_path):
    runtime = _runtime(tmp_path, "baseline", "1" * 40)
    case = BENCHMARK.quick_cases()[0]
    result = _bound_result(runtime, case, 0)

    BENCHMARK.validate_result_binding(runtime, result)

    foreign = tmp_path / "foreign" / "alloygbm"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").touch()
    (foreign / "_alloygbm.so").touch()
    invalid = (
        replace(result, python_executable=str(tmp_path / "other-python")),
        replace(result, package_path=str(foreign / "__init__.py")),
        replace(result, extension_path=str(foreign / "_alloygbm.so")),
        replace(result, package_path=""),
    )
    for record in invalid:
        with pytest.raises(ValueError, match="runtime binding"):
            BENCHMARK.validate_result_binding(runtime, record)


@pytest.mark.parametrize(
    "timing",
    [
        {},
        {"native_train_seconds": None},
        {"native_train_seconds": float("nan")},
        {"native_train_seconds": float("inf")},
    ],
)
def test_native_train_timing_is_required_and_finite(timing):
    with pytest.raises(ValueError, match="native_train_seconds"):
        BENCHMARK.require_native_train_seconds(timing)


def test_zero_rss_is_unavailable_without_erasing_other_case_ratios():
    zero = BENCHMARK.PairedResult(
        _result(case="tall_deep-level", rss_mib=10.0),
        _result(
            case="tall_deep-level",
            rss_mib=0.0,
            source_commit="2" * 40,
            runtime_name="candidate",
        ),
    )
    measurable = BENCHMARK.PairedResult(
        _result(case="wide_deep-leaf", rss_mib=10.0),
        _result(
            case="wide_deep-leaf",
            rss_mib=11.0,
            source_commit="2" * 40,
            runtime_name="candidate",
        ),
    )

    evaluation = BENCHMARK.evaluate_gates([zero, measurable], profile="full")

    assert evaluation.aggregate_rss_ratio == pytest.approx(1.1)
    assert evaluation.rss_cases_available == 1
    assert evaluation.case_summaries[0].rss_ratio is None
    encoded = BENCHMARK.strict_json_dumps(
        {"ratio": evaluation.case_summaries[0].rss_ratio}
    )
    assert encoded == '{"ratio": null}'


def test_all_zero_rss_is_reported_unavailable_and_fails_full_gate():
    pairs = _pairs(candidate_rss=0.0)
    pairs = [
        BENCHMARK.PairedResult(replace(pair.baseline, rss_mib=0.0), pair.candidate)
        for pair in pairs
    ]

    evaluation = BENCHMARK.evaluate_gates(pairs, profile="full")

    assert evaluation.aggregate_rss_ratio is None
    assert evaluation.rss_cases_available == 0
    assert any("RSS unavailable" in failure for failure in evaluation.failures)


def test_rss_ratio_overflow_is_unavailable():
    assert BENCHMARK._rss_ratio(1e308, 1e-308) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_strict_json_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        BENCHMARK.strict_json_dumps({"value": value})


def test_orchestrator_excludes_warmups_and_alternates_arm_order(monkeypatch, tmp_path):
    candidate = _runtime(tmp_path, "candidate", "1" * 40)
    baseline = replace(candidate, name="baseline")
    calls = []

    def fake_run_worker(runtime, case, **kwargs):
        calls.append((case.name, runtime.name, kwargs["repetition"]))
        return _bound_result(runtime, case, kwargs["repetition"])

    monkeypatch.setattr(BENCHMARK, "run_worker", fake_run_worker)

    report = BENCHMARK.run_benchmark(
        profile="quick",
        baseline=baseline,
        candidate=candidate,
        repetitions=2,
        n_jobs=2,
        timeout_seconds=10.0,
    )

    assert len(report["pairs"]) == len(BENCHMARK.quick_cases()) * 2
    assert all(pair["baseline"]["repetition"] >= 0 for pair in report["pairs"])
    first_case = BENCHMARK.quick_cases()[0].name
    assert [call[1:] for call in calls if call[0] == first_case] == [
        ("baseline", -1),
        ("candidate", -1),
        ("baseline", 0),
        ("candidate", 0),
        ("candidate", 1),
        ("baseline", 1),
    ]


def test_worker_parse_errors_include_runtime_and_case_context(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, "baseline", "1" * 40)
    case = BENCHMARK.quick_cases()[0]
    completed = subprocess.CompletedProcess([], 0, stdout="not-json", stderr="")
    monkeypatch.setattr(BENCHMARK.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match=f"baseline.*{case.name}.*single JSON"):
        BENCHMARK.run_worker(
            runtime,
            case,
            profile="quick",
            repetition=0,
            n_jobs=2,
            timeout_seconds=10.0,
        )


def test_cli_requires_profile_and_full_runtime_arguments():
    parser = BENCHMARK._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    baseline_only = parser.parse_args(
        ["--full", "--baseline-python", "/baseline/python"]
    )
    with pytest.raises(ValueError, match="explicit"):
        BENCHMARK.validate_cli_args(baseline_only)


def test_full_cli_supports_repetitions_and_exact_output_flags():
    args = BENCHMARK._parser().parse_args(
        [
            "--full",
            "--baseline-python",
            "/baseline/python",
            "--baseline-workdir",
            "/baseline",
            "--candidate-python",
            "/candidate/python",
            "--candidate-workdir",
            "/candidate",
            "--repetitions",
            "3",
            "--output-json",
            "/tmp/report.json",
            "--output-markdown",
            "/tmp/report.md",
        ]
    )

    BENCHMARK.validate_cli_args(args)
    assert BENCHMARK.profile_from_args(args) == "full"
    assert args.repetitions == 3
    assert args.output_json == Path("/tmp/report.json")
    assert args.output_markdown == Path("/tmp/report.md")


def test_markdown_renders_runtime_identity_aggregate_gates_and_digests():
    pairs = _pairs()
    pairs[0] = BENCHMARK.PairedResult(
        pairs[0].baseline,
        replace(
            pairs[0].candidate,
            artifact_sha256="candidate-artifact",
            prediction_sha256="candidate-prediction",
        ),
    )
    report = BENCHMARK.build_report(
        profile="full",
        baseline=BENCHMARK.RuntimeSpec(
            "baseline", Path("/baseline/python"), Path("/baseline"), "1" * 40
        ),
        candidate=BENCHMARK.RuntimeSpec(
            "candidate", Path("/candidate/python"), Path("/candidate"), "2" * 40
        ),
        pairs=pairs,
        repetitions=3,
        n_jobs=4,
    )

    markdown = BENCHMARK.render_markdown(report)

    assert "# Allocation Reuse Benchmark" in markdown
    assert "`1111111111111111111111111111111111111111`" in markdown
    assert "Aggregate timing ratio" in markdown
    assert "Aggregate RSS ratio" in markdown
    assert "Artifact SHA-256" in markdown
    assert "Prediction SHA-256" in markdown
    assert "Baseline extension SHA-256" in markdown
    assert "Candidate extension SHA-256" in markdown
    assert "`a`" in markdown
    assert "`candidate-artifact`" in markdown
    assert "`b`" in markdown
    assert "`candidate-prediction`" in markdown
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
