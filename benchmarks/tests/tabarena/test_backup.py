import json
import shutil

import pytest

from benchmarks.tests.tabarena.test_checkpoints import complete, fixture_run


def test_verified_backup_incremental_chain_and_stale_guard(tmp_path):
    from benchmarks.tabarena.backup import require_recent_backup, verify_local_chain
    from benchmarks.tabarena.checkpoints import create_checkpoint

    run = fixture_run(tmp_path)
    complete(run, "one")
    first = create_checkpoint(run)
    local = tmp_path / "backup"
    shutil.copytree(run / "checkpoints", local)
    receipt = verify_local_chain(local, now=100)
    (run / "backup-ack.json").write_text(json.dumps(receipt))
    require_recent_backup(run, maximum_age=60, now=150)
    with pytest.raises(ValueError, match="backup"):
        require_recent_backup(run, maximum_age=60, now=161)
    complete(run, "two")
    second = create_checkpoint(run)
    shutil.copy(second, local / second.name)
    shutil.copy(str(second) + ".json", str(local / second.name) + ".json")
    next_receipt = verify_local_chain(local, now=170)
    assert next_receipt["sequence"] == 2
    assert next_receipt["run_fingerprint"] == "fixture"
    assert (local / first.name).exists()
    (local / second.name).write_bytes(b"broken")
    # Independent restore/reverification can still detect corruption of old local bytes.
    from benchmarks.tabarena.checkpoints import restore_checkpoints

    with pytest.raises(ValueError, match="checksum"):
        restore_checkpoints(local, tmp_path / "restored")


def test_corrupt_backup_never_advances_receipt(tmp_path):
    from benchmarks.tabarena.backup import verify_local_chain
    from benchmarks.tabarena.checkpoints import create_checkpoint

    run = fixture_run(tmp_path)
    complete(run, "one")
    shard = create_checkpoint(run)
    shard.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checksum"):
        verify_local_chain(run / "checkpoints")
    assert not (run / "checkpoints/verified-index.json").exists()


def test_previously_verified_backup_cannot_acknowledge_a_missing_old_segment(tmp_path):
    from benchmarks.tabarena.backup import verify_local_chain
    from benchmarks.tabarena.checkpoints import create_checkpoint

    run = fixture_run(tmp_path)
    complete(run, "first")
    first = create_checkpoint(run)
    complete(run, "second")
    create_checkpoint(run)
    verify_local_chain(run / "checkpoints", now=100)
    first.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_local_chain(run / "checkpoints", now=200)
