import json
from copy import deepcopy

import pytest

from benchmarks.tabarena.provenance import file_hash, fingerprint
from benchmarks.tabarena.run import load_latest, write_event


def prepare(tmp_path):
    from benchmarks.tabarena.artifacts import result_cacher
    from benchmarks.tabarena.protocol import RunSpec, resolve_run
    from benchmarks.tests.tabarena.test_artifacts import valid_result

    resolved = resolve_run(RunSpec(num_cpus=16))
    job = resolved.jobs[0]
    meta = resolved.context.metadata_for_jobs([job])[0]
    source, dest = tmp_path / "old", tmp_path / "new"
    manifest = {
        "fingerprint": "origin",
        "source_hash": "source",
        "lock": "lock",
        "upstream": "upstream",
        "registration": {},
        "integration": {"benchmarks/tabarena/submission/alloygbm/model.py": "adapter"},
        "runtime": {
            "python": "3.12.12",
            "machine": "x86_64",
            "build_flags": {},
            "platform": "Linux-old",
        },
        "versions": {"numpy": "2.0"},
        "provenance": {"profile": "release", "rustc": "1.92"},
        "spec": {"num_cpus": 16, "execution": "upstream"},
        "extension": "old-binary",
    }
    for directory in (source, dest):
        directory.mkdir()
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        (directory / "job_plan.json").write_text(json.dumps([job.to_dict()]))
    newer = deepcopy(manifest)
    newer.update(fingerprint="destination", extension="new-binary")
    newer["runtime"]["platform"] = "Linux-new"
    (dest / "run_manifest.json").write_text(json.dumps(newer))
    cacher = result_cacher(source, job, meta)
    cacher.save_cache(valid_result())
    cacher.save_aux("validation_protocol", {"key": "8x1"})
    (source / "job_logs").mkdir()
    (source / "job_logs" / f"{fingerprint(job.to_dict())}.log").write_text(
        "original log"
    )
    write_event(
        source, fingerprint(job.to_dict()), "failed", error="historical network error"
    )
    write_event(
        source,
        fingerprint(job.to_dict()),
        "completed",
        result_sha256=file_hash(cacher.cache_file),
        elapsed_seconds=2,
    )
    return source, dest, resolved, cacher


def test_import_reuses_verified_exact_jobs_and_keeps_origin(tmp_path):
    from benchmarks.tabarena.import_fits import import_fits

    source, dest, resolved, cacher = prepare(tmp_path)
    assert import_fits(source, dest, resolved) == 1
    row = next(iter(load_latest(dest).values()))
    assert row["origin_fingerprint"] == "origin"
    assert row["result_sha256"] == file_hash(cacher.cache_file)
    origin = dest / "origins/origin"
    assert "historical network error" in (origin / "jobs.jsonl").read_text()
    assert (
        json.loads((origin / "run_manifest.json").read_text())["extension"]
        == "old-binary"
    )
    assert import_fits(source, dest, resolved) == 0


@pytest.mark.parametrize(
    "changed", ["source_hash", "lock", "upstream", "registration", "versions"]
)
def test_import_rejects_changed_model_or_dependencies(tmp_path, changed):
    from benchmarks.tabarena.import_fits import import_fits

    source, dest, resolved, _ = prepare(tmp_path)
    p = dest / "run_manifest.json"
    m = json.loads(p.read_text())
    m[changed] = "changed"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="incompatible"):
        import_fits(source, dest, resolved)
    assert not load_latest(dest)


def test_import_rejects_changed_bytes_and_does_not_overwrite_completed(tmp_path):
    from benchmarks.tabarena.import_fits import import_fits

    source, dest, resolved, cacher = prepare(tmp_path)
    from pathlib import Path

    Path(cacher.cache_file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        import_fits(source, dest, resolved)
    assert not load_latest(dest)


def test_import_requires_protocol_side_record(tmp_path):
    from pathlib import Path

    from benchmarks.tabarena.import_fits import import_fits

    source, dest, resolved, cache = prepare(tmp_path)
    (Path(cache.cache_file).parent / "validation_protocol.json").unlink()
    with pytest.raises(ValueError, match="8x1"):
        import_fits(source, dest, resolved)
