"""Contracts for the sampled-prediction-delta A/B benchmark harness."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "sampled_prediction_delta_benchmark.py"

QUICK_CASE_NAMES = {
    "scalar_tall_narrow_level_subsample_050",
    "multiclass_tall_narrow_level_subsample_050",
    "fallback_scalar_dart_subsample_050",
    "fallback_scalar_quantile_subsample_050",
}
FULL_CASE_NAMES = {
    "scalar_tall_narrow_level_full",
    "scalar_tall_narrow_level_subsample_080",
    "scalar_tall_narrow_level_subsample_050",
    "scalar_shallow_tall_leaf_subsample_050",
    "scalar_medium_wide_level_goss",
    "scalar_small_wide_leaf_subsample_050",
    "multiclass_tall_narrow_level_subsample_050",
    "multiclass_medium_wide_leaf_goss",
    "fallback_scalar_dart_subsample_050",
    "fallback_scalar_quantile_subsample_050",
}


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "sampled_prediction_delta_benchmark_module", BENCHMARK_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


def _runtime(tmp_path: Path, name: str, commit: str):
    workdir = tmp_path / name
    python = workdir / ".venv" / "bin" / "python"
    package = workdir / "bindings" / "python" / "alloygbm"
    python.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    python.touch()
    package_path = package / "__init__.py"
    extension_path = package / "_alloygbm.abi3.so"
    package_path.touch()
    extension_path.write_bytes(commit.encode("ascii"))
    manifest = BENCHMARK.RuntimeManifest(
        schema_version=BENCHMARK.RUNTIME_MANIFEST_SCHEMA_VERSION,
        runtime_name=name,
        source_commit=commit,
        python_executable=str(python.absolute()),
        package_path=str(package_path.resolve()),
        extension_path=str(extension_path.resolve()),
        extension_sha256=hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        extension_source_commit=commit,
    )
    manifest_path = workdir / f"{name}-runtime.json"
    BENCHMARK.write_runtime_manifest(manifest_path, manifest)
    return BENCHMARK.RuntimeSpec(
        name=name,
        python=python,
        workdir=workdir,
        source_commit=commit,
        manifest_path=manifest_path,
        attestation=manifest,
    )


def _result(
    case=None,
    *,
    repetition: int = 0,
    runtime_name: str = "baseline",
    source_commit: str = "1" * 40,
    native_seconds: float = 1.0,
    rss_mib: float | None = 10.0,
    quality_value: float = 0.25,
):
    case = case or BENCHMARK.full_cases()[1]
    return BENCHMARK.CaseResult(
        case=case.name,
        shape=case.shape,
        task=case.task,
        growth=case.growth,
        sampling=case.sampling,
        repetition=repetition,
        runtime_name=runtime_name,
        source_commit=source_commit,
        python_executable=f"/{runtime_name}/python",
        package_path=f"/{runtime_name}/alloygbm/__init__.py",
        extension_path=f"/{runtime_name}/alloygbm/_alloygbm.so",
        extension_sha256=("e" if runtime_name == "baseline" else "f") * 64,
        artifact_sha256="a" * 64,
        prediction_sha256="b" * 64,
        native_seconds=native_seconds,
        rss_mib=rss_mib,
        completed_rounds=24,
        stop_reason="CompletedRequestedRounds",
        quality_metric="rmse" if case.task == "scalar" else "log_loss",
        quality_value=quality_value,
        fallback_sentinel=case.fallback_sentinel,
        dimensions={
            "n_rows": case.n_rows,
            "n_features": case.n_features,
            "n_eval_rows": case.n_eval_rows,
        },
        parameters={"seed": case.seed},
    )


def _bound_result(runtime, case, repetition, **changes):
    result = _result(
        case,
        repetition=repetition,
        runtime_name=runtime.name,
        source_commit=runtime.source_commit,
    )
    result = replace(
        result,
        python_executable=str(runtime.python.absolute()),
        package_path=runtime.attestation.package_path,
        extension_path=runtime.attestation.extension_path,
        extension_sha256=runtime.attestation.extension_sha256,
    )
    return replace(result, **changes)


def _full_pairs(
    *,
    eligible_ratio: float = 1.0,
    delta_ratio: float | None = None,
    candidate_rss: float = 10.0,
):
    pairs = []
    for case in BENCHMARK.full_cases():
        ratio = eligible_ratio
        if delta_ratio is not None and case.delta_sensitive:
            ratio = delta_ratio
        if case.fallback_sentinel is not None:
            ratio = 9.0
        for repetition in range(5):
            baseline = _result(case, repetition=repetition)
            candidate = replace(
                baseline,
                runtime_name="candidate",
                source_commit="2" * 40,
                extension_sha256="f" * 64,
                native_seconds=ratio,
                rss_mib=candidate_rss,
            )
            pairs.append(BENCHMARK.PairedResult(baseline, candidate))
    return pairs


def test_fixtures_are_deterministic_finite_and_contain_missing_values():
    for task in ("scalar", "multiclass"):
        case = next(case for case in BENCHMARK.full_cases() if case.task == task)
        first = BENCHMARK.make_fixture(case)
        second = BENCHMARK.make_fixture(case)

        for left, right in zip(first.arrays(), second.arrays(), strict=True):
            np.testing.assert_array_equal(left, right)
        assert first.X_train.shape == (case.n_rows, case.n_features)
        assert first.X_eval.shape == (case.n_eval_rows, case.n_features)
        assert first.X_train.dtype == np.float32
        assert np.isnan(first.X_train).any()
        assert np.isnan(first.X_eval).any()
        assert np.isfinite(first.y_train).all()
        assert np.isfinite(first.y_eval).all()
        if task == "multiclass":
            assert set(np.unique(first.y_train)) == {0, 1, 2}


def test_profiles_have_exact_predeclared_names_and_fallbacks():
    quick = BENCHMARK.quick_cases()
    full = BENCHMARK.full_cases()

    assert {case.name for case in quick} == QUICK_CASE_NAMES
    assert {case.name for case in full} == FULL_CASE_NAMES
    assert BENCHMARK.QUICK_CASE_NAMES == QUICK_CASE_NAMES
    assert BENCHMARK.FULL_CASE_NAMES == FULL_CASE_NAMES
    assert {case.shape for case in full} == {
        "tall_narrow",
        "shallow_tall",
        "medium_wide",
        "small_wide",
    }
    assert {case.task for case in full} == {"scalar", "multiclass"}
    assert {case.growth for case in full} == {"level", "leaf"}
    assert {
        (case.name, case.fallback_sentinel)
        for case in full
        if case.fallback_sentinel is not None
    } == {
        ("fallback_scalar_dart_subsample_050", "dart_full_replay"),
        ("fallback_scalar_quantile_subsample_050", "quantile_full_replay"),
    }
    assert BENCHMARK.profile_repetitions("quick") == 1
    assert BENCHMARK.profile_repetitions("full") == 5


def test_worker_invocation_is_isolated_and_manifest_bound(tmp_path):
    runtime = _runtime(tmp_path, "candidate", "2" * 40)
    case = BENCHMARK.quick_cases()[0]

    invocation = BENCHMARK.build_worker_invocation(
        runtime, case, profile="quick", repetition=0, n_jobs=2
    )

    assert invocation.command[:2] == (str(runtime.python), "-I")
    assert str(BENCHMARK_PATH) in invocation.command
    assert invocation.cwd == runtime.workdir
    assert invocation.value_after("--runtime-manifest") == str(
        runtime.manifest_path
    )
    assert invocation.value_after("--expected-source-commit") == "2" * 40
    assert "PYTHONPATH" not in invocation.env
    assert "PYTHONHOME" not in invocation.env


def test_worker_validates_manifest_before_importing_alloygbm(monkeypatch, tmp_path):
    args = BENCHMARK._parser().parse_args(
        [
            "--quick",
            "--worker",
            "--runtime-name",
            "candidate",
            "--runtime-workdir",
            str(tmp_path),
            "--runtime-manifest",
            str(tmp_path / "invalid.json"),
            "--expected-python",
            sys.executable,
            "--expected-source-commit",
            "2" * 40,
            "--case",
            BENCHMARK.quick_cases()[0].name,
            "--repetition",
            "0",
        ]
    )
    imported = False
    real_import = __import__

    def tracking_import(name, *import_args, **import_kwargs):
        nonlocal imported
        if name == "alloygbm" or name.startswith("alloygbm."):
            imported = True
        return real_import(name, *import_args, **import_kwargs)

    monkeypatch.setattr(BENCHMARK, "_require_clean_worktree", lambda _: None)
    monkeypatch.setattr(BENCHMARK, "_git_commit", lambda _: "2" * 40)
    monkeypatch.setattr("builtins.__import__", tracking_import)

    with pytest.raises(ValueError, match="manifest"):
        BENCHMARK._worker_result(args)
    assert imported is False


def test_worker_parser_accepts_one_strict_complete_identity_bound_record(tmp_path):
    runtime = _runtime(tmp_path, "candidate", "2" * 40)
    case = BENCHMARK.quick_cases()[0]
    expected = _bound_result(runtime, case, 0)

    parsed = BENCHMARK.parse_worker_output(
        BENCHMARK.strict_json_dumps(expected.to_dict()),
        runtime=runtime,
        expected_case=case.name,
        expected_repetition=0,
    )

    assert parsed == expected


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("noise\n{}", "single JSON object"),
        ('{"native_seconds": NaN}', "single JSON object"),
        ('{"case": "a", "case": "b"}', "single JSON object"),
    ],
)
def test_worker_parser_rejects_noise_nonfinite_json_and_duplicate_keys(
    tmp_path, payload, message
):
    runtime = _runtime(tmp_path, "candidate", "2" * 40)
    with pytest.raises(ValueError, match=message):
        BENCHMARK.parse_worker_output(
            payload,
            runtime=runtime,
            expected_case=BENCHMARK.quick_cases()[0].name,
            expected_repetition=0,
        )


def test_worker_parser_rejects_nonfinite_metrics_and_wrong_identity(tmp_path):
    runtime = _runtime(tmp_path, "candidate", "2" * 40)
    case = BENCHMARK.quick_cases()[0]
    result = _bound_result(runtime, case, 0)

    for changed, message in (
        (replace(result, quality_value=float("inf")), "single JSON object"),
        (replace(result, native_seconds=float("nan")), "single JSON object"),
        (replace(result, source_commit="3" * 40), "source commit"),
        (replace(result, extension_sha256="0" * 64), "manifest"),
    ):
        with pytest.raises(ValueError, match=message):
            BENCHMARK.parse_worker_output(
                json.dumps(changed.to_dict()),
                runtime=runtime,
                expected_case=case.name,
                expected_repetition=0,
            )


def test_worker_parser_binds_named_case_to_catalog_metadata(tmp_path):
    runtime = _runtime(tmp_path, "candidate", "2" * 40)
    case = next(
        case
        for case in BENCHMARK.quick_cases()
        if case.fallback_sentinel == "dart_full_replay"
    )
    result = _bound_result(runtime, case, 0)

    for changed in (
        replace(result, shape="wrong"),
        replace(result, task="multiclass"),
        replace(result, fallback_sentinel=None),
        replace(result, quality_metric="log_loss"),
    ):
        with pytest.raises(ValueError, match="case metadata"):
            BENCHMARK.parse_worker_output(
                json.dumps(changed.to_dict()),
                runtime=runtime,
                expected_case=case.name,
                expected_repetition=0,
            )


def test_pairing_requires_exact_digests_rounds_stop_reason_and_quality():
    baseline = _result()
    candidate = replace(
        baseline,
        runtime_name="candidate",
        source_commit="2" * 40,
        extension_sha256="f" * 64,
        native_seconds=0.9,
        rss_mib=8.0,
    )

    evaluation = BENCHMARK.evaluate_pair(baseline, candidate)
    assert evaluation.equivalent
    assert evaluation.time_ratio == pytest.approx(0.9)
    assert evaluation.rss_ratio == pytest.approx(0.8)

    changes = (
        replace(candidate, artifact_sha256="c" * 64),
        replace(candidate, prediction_sha256="d" * 64),
        replace(candidate, completed_rounds=23),
        replace(candidate, stop_reason="NoValidSplit"),
        replace(candidate, quality_value=np.nextafter(0.25, np.inf)),
        replace(candidate, quality_metric="different"),
    )
    for changed in changes:
        mismatch = BENCHMARK.evaluate_pair(baseline, changed)
        assert not mismatch.equivalent
        assert mismatch.failures


def test_worker_requires_native_completion_diagnostics():
    class Complete:
        rounds_completed_ = 24
        stop_reason_ = "CompletedRequestedRounds"

    assert BENCHMARK.require_completion_diagnostics(Complete()) == (
        24,
        "CompletedRequestedRounds",
    )

    for incomplete in (object(), type("NoStop", (), {"rounds_completed_": 24})()):
        with pytest.raises(ValueError, match="completion diagnostics"):
            BENCHMARK.require_completion_diagnostics(incomplete)


def test_fallback_timing_is_excluded_but_equivalence_is_not():
    pairs = _full_pairs(eligible_ratio=1.0)

    evaluation = BENCHMARK.evaluate_gates(pairs, profile="full")
    assert evaluation.delta_sensitive_time_ratio == pytest.approx(1.0)
    assert evaluation.all_eligible_time_ratio == pytest.approx(1.0)
    assert not any("fallback" in failure for failure in evaluation.failures)

    fallback_index = next(
        index
        for index, pair in enumerate(pairs)
        if pair.baseline.fallback_sentinel is not None
    )
    pair = pairs[fallback_index]
    pairs[fallback_index] = BENCHMARK.PairedResult(
        pair.baseline,
        replace(pair.candidate, prediction_sha256="0" * 64),
    )
    failed = BENCHMARK.evaluate_gates(pairs, profile="full")
    assert any("prediction" in failure for failure in failed.failures)


def test_time_gate_boundaries_are_predeclared_and_inclusive():
    passing = BENCHMARK.evaluate_gates(
        _full_pairs(eligible_ratio=1.0, delta_ratio=0.98), profile="full"
    )
    assert not any("delta-sensitive" in failure for failure in passing.failures)

    delta_failure = BENCHMARK.evaluate_gates(
        _full_pairs(eligible_ratio=1.0, delta_ratio=0.981), profile="full"
    )
    assert any("delta-sensitive" in failure for failure in delta_failure.failures)

    all_failure = BENCHMARK.evaluate_gates(
        _full_pairs(eligible_ratio=1.031), profile="full"
    )
    assert any("all-eligible" in failure for failure in all_failure.failures)

    boundary = BENCHMARK.evaluate_gates(
        _full_pairs(eligible_ratio=1.08, delta_ratio=0.98), profile="full"
    )
    assert not any("per-case" in failure for failure in boundary.failures)

    pairs = _full_pairs(eligible_ratio=1.0, delta_ratio=0.98)
    eligible = next(
        case for case in BENCHMARK.full_cases() if not case.delta_sensitive and case.performance_eligible
    )
    pairs = [
        BENCHMARK.PairedResult(pair.baseline, replace(pair.candidate, native_seconds=1.081))
        if pair.baseline.case == eligible.name
        else pair
        for pair in pairs
    ]
    per_case_failure = BENCHMARK.evaluate_gates(pairs, profile="full")
    assert any("per-case" in failure for failure in per_case_failure.failures)


def test_per_case_gate_waives_only_serialized_noise_floor_cases(tmp_path):
    pairs = _full_pairs(eligible_ratio=1.0, delta_ratio=0.98)
    control = next(
        case for case in BENCHMARK.full_cases() if case.performance_eligible and not case.delta_sensitive
    )
    below_floor = BENCHMARK.NATIVE_TIME_NOISE_FLOOR_SECONDS / 2.0
    pairs = [
        BENCHMARK.PairedResult(
            replace(pair.baseline, native_seconds=below_floor),
            replace(pair.candidate, native_seconds=below_floor * 2.0),
        )
        if pair.baseline.case == control.name
        else pair
        for pair in pairs
    ]

    evaluation = BENCHMARK.evaluate_gates(pairs, profile="full")
    assert not any("per-case" in failure for failure in evaluation.failures)

    baseline = _runtime(tmp_path, "baseline", "1" * 40)
    candidate = _runtime(tmp_path, "candidate", "2" * 40)
    report = BENCHMARK.build_report(
        profile="full",
        baseline=baseline,
        candidate=candidate,
        pairs=pairs,
        repetitions=5,
        n_jobs=2,
    )
    assert report["gate"]["limits"]["native_time_noise_floor_seconds"] == (
        BENCHMARK.NATIVE_TIME_NOISE_FLOOR_SECONDS
    )
    BENCHMARK.strict_json_dumps(report)


def test_rss_gate_boundary_and_every_full_pair_must_be_measurable():
    boundary = BENCHMARK.evaluate_gates(
        _full_pairs(delta_ratio=0.98, candidate_rss=10.5), profile="full"
    )
    assert boundary.aggregate_rss_ratio == pytest.approx(1.05)
    assert not any("aggregate RSS" in failure for failure in boundary.failures)

    over = BENCHMARK.evaluate_gates(
        _full_pairs(delta_ratio=0.98, candidate_rss=10.51), profile="full"
    )
    assert any("aggregate RSS" in failure for failure in over.failures)

    pairs = _full_pairs(delta_ratio=0.98)
    first = pairs[0]
    pairs[0] = BENCHMARK.PairedResult(
        replace(first.baseline, rss_mib=0.0), first.candidate
    )
    unavailable = BENCHMARK.evaluate_gates(pairs, profile="full")
    assert any("measurable positive RSS" in failure for failure in unavailable.failures)


def test_quick_gate_checks_consistency_without_performance_claims():
    cases = {case.name: case for case in BENCHMARK.quick_cases()}
    pairs = []
    for case_name in sorted(QUICK_CASE_NAMES):
        baseline = _result(cases[case_name])
        candidate = replace(
            baseline,
            runtime_name="candidate",
            native_seconds=20.0,
            rss_mib=100.0,
        )
        pairs.append(BENCHMARK.PairedResult(baseline, candidate))

    evaluation = BENCHMARK.evaluate_gates(pairs, profile="quick")
    assert evaluation.failures == ()
    assert evaluation.performance_gated is False


def test_configuration_requires_one_quick_manifest_and_distinct_full_manifests(
    tmp_path,
):
    candidate = _runtime(tmp_path, "candidate", "2" * 40)
    baseline_alias = BENCHMARK.quick_baseline_runtime(candidate)
    BENCHMARK.validate_run_configuration(
        profile="quick",
        baseline=baseline_alias,
        candidate=candidate,
        repetitions=1,
    )

    baseline = _runtime(tmp_path, "baseline", "1" * 40)
    BENCHMARK.validate_run_configuration(
        profile="full", baseline=baseline, candidate=candidate, repetitions=5
    )
    with pytest.raises(ValueError, match="distinct.*manifest"):
        BENCHMARK.validate_run_configuration(
            profile="full",
            baseline=baseline,
            candidate=replace(
                candidate,
                manifest_path=baseline.manifest_path,
                attestation=baseline.attestation,
            ),
            repetitions=5,
        )
    with pytest.raises(ValueError, match="at least 5"):
        BENCHMARK.validate_run_configuration(
            profile="full", baseline=baseline, candidate=candidate, repetitions=4
        )


def test_orchestrator_excludes_one_warmup_and_alternates_arms(monkeypatch, tmp_path):
    candidate = _runtime(tmp_path, "candidate", "2" * 40)
    baseline = BENCHMARK.quick_baseline_runtime(candidate)
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

    assert len(report["pairs"]) == len(QUICK_CASE_NAMES) * 2
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


def test_cli_has_manifest_quick_and_five_repetition_full_modes():
    parser = BENCHMARK._parser()

    quick = parser.parse_args(
        ["--quick", "--candidate-manifest", "/tmp/candidate.json", "--gate"]
    )
    BENCHMARK.validate_cli_args(quick)
    assert BENCHMARK.profile_from_args(quick) == "quick"

    full = parser.parse_args(
        [
            "--baseline-manifest",
            "/tmp/baseline.json",
            "--candidate-manifest",
            "/tmp/candidate.json",
            "--repetitions",
            "5",
            "--output-json",
            "/tmp/report.json",
            "--output-markdown",
            "/tmp/report.md",
        ]
    )
    BENCHMARK.validate_cli_args(full)
    assert BENCHMARK.profile_from_args(full) == "full"
    assert BENCHMARK.repetitions_from_args(full) == 5

    manifest = parser.parse_args(
        [
            "--write-runtime-manifest",
            "/tmp/candidate.json",
            "--runtime-name",
            "candidate",
            "--runtime-python",
            "/candidate/python",
            "--runtime-workdir",
            "/candidate",
        ]
    )
    BENCHMARK.validate_cli_args(manifest)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_strict_json_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        BENCHMARK.strict_json_dumps({"value": value})
