import json
import shutil
from pathlib import Path

import pytest

from benchmarks.tabarena.provenance import file_hash
from benchmarks.tabarena.run import load_latest, write_event


def fixture_run(root):
    run = root / "run"
    run.mkdir()
    for name in ("configs.json", "tasks.json", "job_plan.json"):
        (run / name).write_text("[]\n")
    (run / "run_manifest.json").write_text(json.dumps({"fingerprint": "fixture"}))
    return run


def complete(run, key, content=b"result"):
    cached = run / "data" / key / "results.pkl"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    write_event(
        run,
        key,
        "completed",
        cache_path=str(cached.relative_to(run)),
        result_sha256=file_hash(cached),
    )
    (run / "job_logs").mkdir(exist_ok=True)
    (run / "job_logs" / f"{key}.log").write_text("8 workers x2 cpus\n")
    return cached


def test_incremental_checkpoint_restore_after_server_loss(tmp_path):
    from benchmarks.tabarena.checkpoints import (
        create_checkpoint,
        restore_checkpoints,
        verify_checkpoint,
    )

    run = fixture_run(tmp_path)
    complete(run, "first")
    a = create_checkpoint(run)
    complete(run, "second", b"prediction bytes")
    b = create_checkpoint(run)
    assert a != b
    second = verify_checkpoint(b)
    assert second["completed_keys"] == ["second"]
    assert second["previous_sha256"] == file_hash(a)
    backup = tmp_path / "offserver"
    shutil.copytree(run / "checkpoints", backup)
    shutil.rmtree(run)
    restored = tmp_path / "restored"
    restore_checkpoints(backup, restored)
    assert set(load_latest(restored)) == {"first", "second"}
    assert (restored / "data/second/results.pkl").read_bytes() == b"prediction bytes"
    assert (restored / "job_logs/first.log").exists()
    assert create_checkpoint(restored) is None


def test_checkpoint_rejects_corruption_missing_chain_and_changed_results(tmp_path):
    from benchmarks.tabarena.checkpoints import (
        create_checkpoint,
        restore_checkpoints,
        verify_checkpoint,
    )

    run = fixture_run(tmp_path)
    cached = complete(run, "first")
    a = create_checkpoint(run)
    complete(run, "second")
    b = create_checkpoint(run)
    only_second = tmp_path / "missing"
    only_second.mkdir()
    shutil.copy(b, only_second / b.name)
    shutil.copy(Path(str(b) + ".json"), only_second / (b.name + ".json"))
    with pytest.raises(ValueError, match="chain"):
        restore_checkpoints(only_second, tmp_path / "restored")
    a.write_bytes(a.read_bytes()[:-8])
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(a)
    cached = complete(run, "third")
    cached.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        create_checkpoint(run)


def test_export_failure_does_not_change_durable_jobs(tmp_path, monkeypatch):
    import tarfile

    from benchmarks.tabarena.checkpoints import create_checkpoint

    run = fixture_run(tmp_path)
    cache = complete(run, "saved")

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(tarfile, "open", fail)
    with pytest.raises(OSError):
        create_checkpoint(run)
    assert load_latest(run)["saved"]["status"] == "completed"
    assert cache.read_bytes() == b"result"
    assert not list((run / "checkpoints").glob("*.tar.gz"))


def test_checkpoint_recovers_published_archive_after_index_write_failure(
    tmp_path, monkeypatch
):
    from benchmarks.tabarena import checkpoints

    run = fixture_run(tmp_path)
    complete(run, "one")
    original = checkpoints.atomic_json

    def interrupted(path, payload):
        if path.name == "state.json":
            raise OSError("simulated power loss after receipt publication")
        return original(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "atomic_json", interrupted)
        with pytest.raises(OSError):
            checkpoints.create_checkpoint(run)
    assert checkpoints.create_checkpoint(run) is None
    checkpoints.restore_checkpoints(run / "checkpoints", tmp_path / "restored")
    assert load_latest(tmp_path / "restored")["one"]["status"] == "completed"


def test_failed_attempt_traceback_survives_checkpoint_restore(tmp_path):
    from benchmarks.tabarena.checkpoints import create_checkpoint, restore_checkpoints

    run = fixture_run(tmp_path)
    write_event(run, "failed", "failed", error="model failure")
    (run / "job_logs").mkdir()
    (run / "job_logs/failed.log").write_text("Traceback: actual native failure")
    create_checkpoint(run)
    restore_checkpoints(run / "checkpoints", tmp_path / "restored")
    assert (
        tmp_path / "restored/job_logs/failed.log"
    ).read_text() == "Traceback: actual native failure"
    assert load_latest(tmp_path / "restored")["failed"]["error"] == "model failure"
