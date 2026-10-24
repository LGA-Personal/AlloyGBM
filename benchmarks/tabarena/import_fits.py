"""Import trusted, verified fits into a new scope without rewriting their origin.

A replacement host has a new run identity. Imported timings retain the old host
manifest and extension hash. Only byte-identical job specifications may match.
Never use this on untrusted pickle caches; verify the evidence archive first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmarks.tabarena.artifacts import result_cacher, validate_result
from benchmarks.tabarena.checkpoints import durable_file
from benchmarks.tabarena.provenance import file_hash, fingerprint
from benchmarks.tabarena.run import exclusive_run, load_latest, write_event


def validate_compatibility(old, new):
    for name in ("source_hash", "lock", "upstream", "registration", "versions"):
        if old[name] != new[name]:
            raise ValueError("incompatible fit origin: " + name)
    adapter = "benchmarks/tabarena/submission/alloygbm/model.py"
    if old["integration"][adapter] != new["integration"][adapter]:
        raise ValueError("incompatible fit origin: model adapter")
    for name in ("machine", "build_flags"):
        if old["runtime"][name] != new["runtime"][name]:
            raise ValueError("incompatible fit origin: " + name)
    if old["runtime"]["python"].split()[0] != new["runtime"]["python"].split()[0]:
        raise ValueError("incompatible fit origin: Python")
    for name in ("profile", "rustc"):
        if old["provenance"][name] != new["provenance"][name]:
            raise ValueError("incompatible fit origin: " + name)
    if any(
        not m["runtime"]["platform"].startswith("Linux")
        or m["spec"]["execution"] != "upstream"
        for m in (old, new)
    ):
        raise ValueError("incompatible fit origin: requires Linux upstream evidence")


def import_fits(source, destination, resolved):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("origin and destination must be separate runs")
    with exclusive_run(source), exclusive_run(destination):
        old = json.loads((source / "run_manifest.json").read_text())
        new = json.loads((destination / "run_manifest.json").read_text())
        validate_compatibility(old, new)
        recorded = json.loads((source / "job_plan.json").read_text())
        old_jobs = {fingerprint(j): j for j in recorded}
        latest, existing = load_latest(source), load_latest(destination)
        # Metadata only for matching jobs: no need to expand all 164,016 tasks.
        matches = [
            j
            for j in resolved.jobs
            if fingerprint(j.to_dict()) in old_jobs
            and latest.get(fingerprint(j.to_dict()), {}).get("status") == "completed"
        ]
        pending = []
        for job, meta in zip(matches, resolved.context.metadata_for_jobs(matches)):
            key = fingerprint(job.to_dict())
            if job.to_dict() != old_jobs[key]:
                raise ValueError("job specification differs")
            if key in existing:
                if existing[key]["status"] != "completed":
                    raise ValueError(
                        "destination already attempted matching job: " + key
                    )
                cache = result_cacher(destination, job, meta)
                if file_hash(cache.cache_file) != existing[key]["result_sha256"]:
                    raise ValueError("destination checksum differs")
                continue
            cache = result_cacher(source, job, meta)
            if file_hash(cache.cache_file) != latest[key]["result_sha256"]:
                raise ValueError("origin result checksum differs: " + key)
            protocol = cache.load_aux("validation_protocol")
            if not protocol or protocol.get("key") != "8x1":
                raise ValueError("origin cache lacks the standard 8x1 side record")
            validate_result(cache.load_cache())
            pending.append((job, meta, key, cache))
        origin = destination / "origins" / old["fingerprint"]
        origin.mkdir(parents=True, exist_ok=True)
        for name in ("run_manifest.json", "job_plan.json", "jobs.jsonl"):
            target = origin / name
            if target.exists() and file_hash(target) != file_hash(source / name):
                raise ValueError("origin receipt conflicts")
            shutil.copyfile(source / name, target)
            durable_file(target)
        copied = 0
        for job, meta, key, cache in pending:
            target = result_cacher(destination, job, meta)
            cachedir = Path(cache.cache_file).parent
            if cachedir.is_symlink() or any(
                p.is_symlink() for p in cachedir.rglob("*")
            ):
                raise ValueError("symlinked origin evidence")
            targetdir = Path(target.cache_file).parent
            if targetdir.exists():
                # An interrupted import can leave bytes before the ledger event.
                # Reconcile identical bytes only; never overwrite conflicting data.
                for p in cachedir.rglob("*"):
                    if p.is_file():
                        q = targetdir / p.relative_to(cachedir)
                        if not q.exists() or file_hash(p) != file_hash(q):
                            raise ValueError("destination cache conflicts")
            else:
                shutil.copytree(cachedir, targetdir)
            for p in targetdir.rglob("*"):
                if p.is_file():
                    durable_file(p)
            for folder, suffix in (("job_logs", ".log"), ("job_specs", ".json")):
                p = source / folder / (key + suffix)
                if p.exists():
                    q = destination / folder / p.name
                    q.parent.mkdir(exist_ok=True)
                    shutil.copyfile(p, q)
                    durable_file(q)
            row = {
                k: v
                for k, v in latest[key].items()
                if k not in ("key", "status", "timestamp")
            }
            row.update(
                cache_path=str(Path(target.cache_file).relative_to(destination)),
                origin_fingerprint=old["fingerprint"],
                origin_manifest_sha256=file_hash(origin / "run_manifest.json"),
                origin_ledger_sha256=file_hash(origin / "jobs.jsonl"),
                imported=True,
                cache_hit=True,
            )
            write_event(destination, key, "completed", **row)
            copied += 1
        return copied


def main():
    from benchmarks.tabarena.protocol import RunSpec, resolve_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.destination / "run_manifest.json").read_text())
    print(
        import_fits(
            args.source, args.destination, resolve_run(RunSpec(**manifest["spec"]))
        )
    )


if __name__ == "__main__":
    main()
