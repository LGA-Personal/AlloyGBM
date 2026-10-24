import pytest

from benchmarks.tabarena.acceptance import validate_scope_and_outcomes
from benchmarks.tabarena.provenance import fingerprint


def test_smoke_is_never_lite_acceptance():
    with pytest.raises(ValueError, match="51-dataset"):
        validate_scope_and_outcomes({"spec": {"profile": "smoke"}}, [], {}, {})


def test_all_jobs_and_failure_explanations_are_required():
    manifest = {
        "spec": {
            "profile": "lite-hpo",
            "models": ["AlloyGBM"],
            "execution": "upstream",
            "num_cpus": None,
        },
        "tasks": list(range(51)),
        "runtime": {
            "platform": "Linux-test",
            "machine": "x86_64",
            "hardware": {"effective_cpus": 16, "memory_total_bytes": 31 * 1024**3},
            "thread_environment": {},
        },
    }
    jobs = [{"job": i} for i in range(1326)]
    rows = {fingerprint(j): {"status": "completed"} for j in jobs}
    assert not validate_scope_and_outcomes(manifest, jobs, rows, {})
    key = fingerprint(jobs[0])
    rows[key]["status"] = "started"
    with pytest.raises(ValueError, match="terminal"):
        validate_scope_and_outcomes(manifest, jobs, rows, {})
    rows[key]["status"] = "failed"
    with pytest.raises(ValueError, match="explanation"):
        validate_scope_and_outcomes(manifest, jobs, rows, {})
    assert validate_scope_and_outcomes(
        manifest, jobs, rows, {key: "resource exhaustion; examined error"}
    ) == {key}


@pytest.mark.parametrize(
    "override",
    [
        {"execution": "local"},
        {"num_cpus": 1},
        {"num_cpus": 8},
    ],
)
def test_acceptance_rejects_development_execution(override):
    from benchmarks.tabarena.resources import validate_acceptance_resources

    spec = {"execution": "upstream", "num_cpus": None, **override}
    runtime = {
        "platform": "Linux-test",
        "machine": "x86_64",
        "hardware": {"effective_cpus": 16, "memory_total_bytes": 31 * 1024**3},
        "thread_environment": {},
    }
    with pytest.raises(ValueError, match="acceptance resources"):
        validate_acceptance_resources(spec, runtime)


def test_acceptance_checks_actual_host_and_thread_caps():
    from benchmarks.tabarena.resources import validate_acceptance_resources

    spec = {"execution": "upstream", "num_cpus": None}
    runtime = {
        "platform": "Linux-test",
        "machine": "x86_64",
        "hardware": {"effective_cpus": 16, "memory_total_bytes": 64 * 1024**3},
        "thread_environment": {},
    }
    validate_acceptance_resources(spec, runtime)
    for change in [
        {"platform": "macOS"},
        {"machine": "arm64"},
        {"hardware": {"effective_cpus": 8, "memory_total_bytes": 64 * 1024**3}},
        {"hardware": {"effective_cpus": 16, "memory_total_bytes": 24 * 1024**3}},
        {"thread_environment": {"OMP_NUM_THREADS": "1"}},
    ]:
        with pytest.raises(ValueError, match="acceptance resources"):
            validate_acceptance_resources(spec, {**runtime, **change})


def test_acceptance_requires_standard_context_bound_to_same_ledger():
    from benchmarks.tabarena.acceptance import validate_report_evidence

    manifest = {"fingerprint": "fit"}
    summary = {
        "published_context": True,
        "run_fingerprint": "fit",
        "ledger_sha256": "ledger",
        "report_code_sha256": "processor",
        "local_pool_calibration": None,
    }
    sources = {
        **summary,
        "local_entrants": ["[Local] AlloyGBM"],
        "calibration_method": "RF (default)",
        "fillna_method": "RF (default)",
    }
    validate_report_evidence(manifest, summary, sources, "ledger")
    with pytest.raises(ValueError, match="standard calibration"):
        validate_report_evidence(
            manifest, {**summary, "published_context": False}, sources, "ledger"
        )
    with pytest.raises(ValueError, match="terminal ledger"):
        validate_report_evidence(
            manifest, summary, {**sources, "ledger_sha256": "stale"}, "ledger"
        )
    with pytest.raises(ValueError, match="processor"):
        validate_report_evidence(
            manifest, summary, {**sources, "report_code_sha256": "old"}, "ledger"
        )
    with pytest.raises(ValueError, match="only AlloyGBM"):
        validate_report_evidence(
            manifest,
            summary,
            {**sources, "local_entrants": ["[Local] AlloyGBM", "[Local] LightGBM"]},
            "ledger",
        )
    with pytest.raises(ValueError, match="published RF"):
        validate_report_evidence(
            manifest, summary, {**sources, "fillna_method": None}, "ledger"
        )
