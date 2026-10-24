"""Run standard TabArena jobs with immutable scope/build identity and a ledger."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def select_stage_jobs(resolved, *, dataset_names=None, candidate_limit=None):
    """Execute a subset of an immutable full plan so later stages reuse its caches."""
    if candidate_limit is not None and (
        isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit < 0
    ):
        raise ValueError("stage candidate limit must be nonnegative")
    available = {j.task.dataset for j in resolved.jobs}
    if dataset_names and set(dataset_names) - available:
        raise ValueError("stage dataset is outside the recorded plan")
    jobs = []
    for job in resolved.jobs:
        if dataset_names and job.task.dataset not in dataset_names:
            continue
        if candidate_limit is not None:
            match = re.search(r"_r(\d+)_", job.experiment.name)
            if match and int(match[1]) > candidate_limit:
                continue
        jobs.append(job)
    return jobs


def execute_upstream_job(context, job, run_dir):
    """Reuse Linux's Ray runtime; capture Python/native diagnostics per job."""
    from benchmarks.tabarena.provenance import fingerprint

    log = run_dir / "job_logs" / (fingerprint(job.to_dict()) + ".log")
    log.parent.mkdir(exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    try:
        with log.open("a") as stream:
            stream.write("\n" + json.dumps({"attempt_started": time.time()}) + "\n")
            stream.flush()
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            try:
                results = context.run_jobs(
                    [job],
                    expname=run_dir,
                    register=False,
                    debug_mode=False,
                )
                if not results:
                    raise RuntimeError("upstream returned no result; see job log")
            except BaseException:
                traceback.print_exc(file=stream)
                raise
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)
    return log


def apply_thread_limits(num_cpus):
    if num_cpus is not None:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = str(num_cpus)


def validate_resume(run_dir, identity):
    old = json.loads((run_dir / "run_manifest.json").read_text())
    if old["fingerprint"] != identity:
        raise ValueError(
            "resume identity differs: build, dependencies, protocol, resources, scope or configurations changed"
        )


def write_event(run_dir, key, status, **details):
    path = run_dir / "jobs.jsonl"
    # Recover a torn final write before appending; otherwise its JSON prefix would
    # corrupt the next complete event. Complete malformed lines still fail loudly.
    if path.exists():
        with path.open("r+b") as recovery:
            recovery.seek(0, os.SEEK_END)
            size = recovery.tell()
            if size:
                recovery.seek(size - 1)
            if size and recovery.read(1) != b"\n":
                # Appends do not rescan a growing 164k-job ledger. Only its torn
                # tail needs repair; load_latest validates full lines on startup.
                position = size
                newline = -1
                while position and newline < 0:
                    start = max(0, position - 65536)
                    recovery.seek(start)
                    tail = recovery.read(position - start)
                    found = tail.rfind(b"\n")
                    if found >= 0:
                        newline = start + found
                    position = start
                recovery.truncate(newline + 1)
                recovery.flush()
                os.fsync(recovery.fileno())
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {"key": key, "status": status, "timestamp": time.time(), **details},
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def load_latest(run_dir):
    out = {}
    path = run_dir / "jobs.jsonl"
    if path.exists():
        # A process can die during a final append; only the incomplete final line is ignored.
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                out[row["key"]] = row
    return out


@contextmanager
def exclusive_run(run_dir):
    """Fail rather than let two drivers race on the same immutable run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another driver owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute_worker(job_file, run_dir, execution):
    """Keep complete worker diagnostics with the immutable run, including crashes."""
    log = run_dir / "job_logs" / (job_file.stem + ".log")
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as stream:
        stream.write("\n" + json.dumps({"attempt_started": time.time()}) + "\n")
        stream.flush()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.tabarena.run",
                "--worker-job-file",
                str(job_file),
                "--worker-run-dir",
                str(run_dir),
                "--execution",
                execution,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return log


def run_experiments(
    resolved,
    *,
    output_root,
    receipt_path,
    resume=None,
    retry_failed=False,
    plan_only=False,
    stage_datasets=None,
    stage_candidates=None,
    checkpoint_jobs=50,
    checkpoint_seconds=900,
    min_free_gib=20,
    backup_max_age_seconds=0,
):
    from benchmarks.tabarena.prepare_upstream import UPSTREAM_REV, verify
    from benchmarks.tabarena.provenance import (
        collect_provenance,
        file_hash,
        fingerprint,
    )
    from tabarena.benchmark.experiment.experiment_runner_api import job_cache_exists
    from tabarena.caching import CacheConfig

    selected = select_stage_jobs(
        resolved, dataset_names=stage_datasets, candidate_limit=stage_candidates
    )
    if checkpoint_jobs < 1 or checkpoint_seconds < 1 or min_free_gib < 1:
        raise ValueError("checkpoint intervals and disk reserve must be positive")
    selected_keys = {fingerprint(j.to_dict()) for j in selected}

    provenance = collect_provenance(ROOT, receipt_path)
    registration = verify(ROOT / ".cache/tabarena/upstream")
    payload = {
        "spec": asdict(resolved.spec),
        "configs": resolved.configs,
        "tasks": resolved.tasks,
        "source_hash": provenance["source_hash"],
        "extension": provenance["extension_sha256"],
        "versions": provenance["versions"],
        "runtime": {
            k: provenance[k]
            for k in (
                "python",
                "platform",
                "machine",
                "logical_cpus",
                "hardware",
                "thread_environment",
                "build_flags",
            )
        },
        "lock": provenance["dependency_lock_sha256"],
        "upstream": UPSTREAM_REV,
        "registration": registration,
        "integration": {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in sorted((ROOT / "benchmarks/tabarena").rglob("*.py"))
        },
    }
    identity = fingerprint(payload)
    run_dir = Path(resume) if resume else Path(output_root) / identity[:16]
    with exclusive_run(run_dir):
        if run_dir.exists() and (run_dir / "run_manifest.json").exists():
            validate_resume(run_dir, identity)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "fingerprint": identity,
                **payload,
                "provenance": provenance,
                "expected_jobs": len(resolved.jobs),
                "command": sys.argv,
                "created_at": time.time(),
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            (run_dir / "configs.json").write_text(
                json.dumps(resolved.configs, indent=2) + "\n"
            )
            (run_dir / "tasks.json").write_text(
                json.dumps(resolved.tasks, indent=2) + "\n"
            )
            (run_dir / "job_plan.json").write_text(
                json.dumps([j.to_dict() for j in resolved.jobs], indent=2) + "\n"
            )
        if plan_only:
            print(
                f"Plan: {len(resolved.jobs)} jobs, official 8x1 protocol; {run_dir}",
                flush=True,
            )
            return run_dir
        cache = CacheConfig.from_root(ROOT / ".cache/tabarena/cache")
        resolved.context.cache_config = cache
        latest = load_latest(run_dir)
        checkpoint_at = time.monotonic()
        since_checkpoint = 0
        metadata = resolved.context.metadata_for_jobs(resolved.jobs)
        for index, (job, meta) in enumerate(zip(resolved.jobs, metadata), 1):
            key = fingerprint(job.to_dict())
            if key not in selected_keys:
                continue
            previous = latest.get(key, {})
            if previous.get("status") == "failed" and not retry_failed:
                continue
            task_id = meta.task_id_str
            exists = job_cache_exists(
                output_dir=str(run_dir),
                method_name=job.experiment.name,
                task_id_str=task_id,
                fold=job.task.fold,
                repeat=job.task.repeat,
            )
            if previous.get("status") == "completed" and not exists:
                raise ValueError(
                    "completed job has lost its upstream result cache: " + key
                )
            if previous.get("status") == "completed":
                from benchmarks.tabarena.artifacts import result_cacher, validate_result

                cacher = result_cacher(run_dir, job, meta)
                if file_hash(cacher.cache_file) != previous.get("result_sha256"):
                    raise ValueError("completed result checksum differs: " + key)
                validate_result(cacher.load_cache())
                continue
            import shutil

            from benchmarks.tabarena.backup import require_recent_backup

            require_recent_backup(run_dir, maximum_age=backup_max_age_seconds)

            if shutil.disk_usage(run_dir).free < min_free_gib * 1024**3:
                raise RuntimeError(
                    "disk reserve reached; stop and preserve evidence before continuing"
                )
            if exists:
                from benchmarks.tabarena.artifacts import result_cacher

                # A killed cache writer can leave a partial pickle. Preserve it
                # for diagnosis, but never let upstream treat it as a cache hit.
                cache_dir = Path(result_cacher(run_dir, job, meta).cache_file).parent
                quarantine = run_dir / "interrupted_caches" / key / str(time.time_ns())
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                cache_dir.rename(quarantine)
                exists = False
            print(
                f"[{index}/{len(resolved.jobs)}] {job.task.dataset} {job.experiment.name}",
                flush=True,
            )
            started = time.monotonic()
            write_event(
                run_dir,
                key,
                "started",
                log_path="job_logs/" + key + ".log",
                experiment=job.experiment.name,
                task=job.task.as_triple(),
            )
            try:
                job_file = run_dir / "job_specs" / (key + ".json")
                job_file.parent.mkdir(exist_ok=True)
                job_file.write_text(json.dumps(job.to_dict(), indent=2))
                if resolved.spec.execution == "upstream" and sys.platform.startswith(
                    "linux"
                ):
                    execute_upstream_job(resolved.context, job, run_dir)
                else:
                    execute_worker(job_file, run_dir, resolved.spec.execution)
                from benchmarks.tabarena.artifacts import result_cacher, validate_result

                cacher = result_cacher(run_dir, job, meta)
                validate_result(cacher.load_cache())
            except Exception as exc:  # noqa: BLE001 -- benchmark failures must be recorded; interrupts propagate
                write_event(
                    run_dir,
                    key,
                    "failed",
                    experiment=job.experiment.name,
                    task=job.task.as_triple(),
                    elapsed_seconds=time.monotonic() - started,
                    error=f"{type(exc).__name__}: {exc}",
                    log_path="job_logs/" + key + ".log",
                )
                print(f"Failed: {exc}", file=sys.stderr, flush=True)
            else:
                from benchmarks.tabarena.checkpoints import durable_file

                for path in Path(cacher.cache_file).parent.rglob("*"):
                    if path.is_file():
                        durable_file(path)
                write_event(
                    run_dir,
                    key,
                    "completed",
                    experiment=job.experiment.name,
                    task=job.task.as_triple(),
                    elapsed_seconds=time.monotonic() - started,
                    cache_hit=exists,
                    result_sha256=file_hash(cacher.cache_file),
                    cache_path=str(Path(cacher.cache_file).relative_to(run_dir)),
                )
            since_checkpoint += 1
            if (
                since_checkpoint >= checkpoint_jobs
                or time.monotonic() - checkpoint_at >= checkpoint_seconds
            ):
                from benchmarks.tabarena.checkpoints import create_checkpoint

                checkpoint = create_checkpoint(run_dir)
                print(f"Durable checkpoint: {checkpoint}", flush=True)
                since_checkpoint, checkpoint_at = 0, time.monotonic()
        from benchmarks.tabarena.checkpoints import create_checkpoint

        if (run_dir / "jobs.jsonl").exists():
            create_checkpoint(run_dir)
        return run_dir


def main():
    from benchmarks.tabarena.protocol import DEFAULT_MODELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "lite-defaults", "lite-hpo", "lite-200", "full"],
        default="smoke",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--search-space", choices=["v1", "modes-v2"], default="v1")
    parser.add_argument("--checkpoint-jobs", type=int, default=50)
    parser.add_argument("--checkpoint-seconds", type=int, default=900)
    parser.add_argument("--min-free-gib", type=int, default=20)
    parser.add_argument(
        "--backup-max-age-seconds",
        type=int,
        default=0,
        help="Stop before another fit if verified off-host backup is stale; 0 disables the guard",
    )
    parser.add_argument(
        "--execution",
        choices=["local", "upstream"],
        default="upstream",
        help="local is the upstream quickstart native/debug runner; upstream uses Ray defaults",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "benchmarks/results/tabarena"
    )
    parser.add_argument(
        "--build-receipt",
        type=Path,
        default=ROOT / ".cache/tabarena/build-receipt.json",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stage-datasets",
        nargs="+",
        help="Execute only these planned datasets; preserve full run identity",
    )
    parser.add_argument(
        "--stage-candidates",
        type=int,
        help="Execute default plus up to N planned samples; 0 executes defaults only",
    )
    parser.add_argument("--worker-job-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job_file:
        from tabarena.benchmark.experiment.job import Job
        from tabarena.caching import CacheConfig
        from tabarena.contexts import TabArenaContext

        context = TabArenaContext(
            backend="native" if args.execution == "local" else "ray",
            cache_config=CacheConfig.from_root(ROOT / ".cache/tabarena/cache"),
        )
        job = Job.from_dict(json.loads(args.worker_job_file.read_text()))
        result = context.run_jobs(
            [job],
            expname=args.worker_run_dir,
            register=False,
            debug_mode=args.execution == "local",
        )
        return 0 if result else 1
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    apply_thread_limits(args.threads)
    from benchmarks.tabarena.protocol import RunSpec, resolve_run

    spec = RunSpec(
        args.profile,
        args.threads,
        tuple(args.datasets) if args.datasets else None,
        tuple(args.models),
        args.execution,
        args.search_space,
    )
    directory = run_experiments(
        resolve_run(spec),
        output_root=args.output_root,
        receipt_path=args.build_receipt,
        resume=args.resume,
        retry_failed=args.retry_failed,
        plan_only=args.plan_only,
        stage_datasets=args.stage_datasets,
        stage_candidates=args.stage_candidates,
        checkpoint_jobs=args.checkpoint_jobs,
        checkpoint_seconds=args.checkpoint_seconds,
        min_free_gib=args.min_free_gib,
        backup_max_age_seconds=args.backup_max_age_seconds,
    )
    print(directory, flush=True)
    if any(r["status"] == "failed" for r in load_latest(directory).values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
