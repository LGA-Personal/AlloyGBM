import json

import pytest

from benchmarks.tabarena.provenance import fingerprint
from benchmarks.tabarena.run import load_latest, validate_resume, write_event


def test_resume_rejects_changed_identity(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint({"build": "a"})})
    )
    validate_resume(tmp_path, fingerprint({"build": "a"}))
    with pytest.raises(ValueError, match="identity"):
        validate_resume(tmp_path, fingerprint({"build": "b"}))


def test_interrupted_job_ledger_preserves_failures(tmp_path):
    write_event(tmp_path, "a", "started")
    write_event(tmp_path, "a", "failed", error="budget exhausted")
    write_event(tmp_path, "b", "completed")
    rows = load_latest(tmp_path)
    assert rows["a"]["status"] == "failed"
    assert rows["a"]["error"] == "budget exhausted"
    assert rows["b"]["status"] == "completed"


def test_partial_final_ledger_write_is_recoverable(tmp_path):
    write_event(tmp_path, "a", "completed")
    with (tmp_path / "jobs.jsonl").open("a") as stream:
        stream.write('{"key": "unfinished')
    assert load_latest(tmp_path)["a"]["status"] == "completed"
    write_event(tmp_path, "b", "failed", error="timeout")
    assert load_latest(tmp_path)["b"]["status"] == "failed"


def test_duplicate_driver_cannot_own_run(tmp_path):
    from benchmarks.tabarena.run import exclusive_run

    with (
        exclusive_run(tmp_path),
        pytest.raises(ValueError, match="another driver"),
        exclusive_run(tmp_path),
    ):
        pass
    with exclusive_run(tmp_path):
        pass


def test_worker_failure_retains_actual_traceback(tmp_path):
    import subprocess

    from benchmarks.tabarena.run import execute_worker

    job = tmp_path / "invalid.json"
    job.write_text("{broken")
    with pytest.raises(subprocess.CalledProcessError):
        execute_worker(job, tmp_path, "local")
    log = (tmp_path / "job_logs/invalid.log").read_text()
    assert "JSONDecodeError" in log
    assert "attempt_started" in log
