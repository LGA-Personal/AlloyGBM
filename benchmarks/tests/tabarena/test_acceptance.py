import pytest

from benchmarks.tabarena.acceptance import validate_scope_and_outcomes
from benchmarks.tabarena.provenance import fingerprint


def test_smoke_is_never_lite_acceptance():
    with pytest.raises(ValueError, match="full four-method"):
        validate_scope_and_outcomes({"spec": {"profile": "smoke"}}, [], {}, {})


def test_all_jobs_and_failure_explanations_are_required():
    manifest = {
        "spec": {
            "profile": "lite-hpo",
            "models": ["AlloyGBM", "LightGBM", "XGBoost", "CatBoost"],
        },
        "tasks": list(range(51)),
    }
    jobs = [{"job": i} for i in range(5304)]
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


def test_acceptance_requires_standard_context_bound_to_same_ledger():
    from benchmarks.tabarena.acceptance import validate_report_evidence

    manifest = {"fingerprint": "fit"}
    summary = {
        "published_context": True,
        "run_fingerprint": "fit",
        "ledger_sha256": "ledger",
        "report_code_sha256": "processor",
    }
    sources = dict(summary)
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
