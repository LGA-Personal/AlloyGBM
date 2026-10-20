"""Run standard TabArena jobs with immutable scope/build identity and a ledger."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    load_latest(run_dir)
    if path.exists():
        with path.open("r+b") as recovery:
            data = recovery.read()
            if data and not data.endswith(b"\n"):
                recovery.truncate(data.rfind(b"\n") + 1)
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
        lines = path.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if i == len(lines) - 1 and not line.endswith("\n"):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not line.endswith("\n"):
                    break
                raise
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
):
    from benchmarks.tabarena.prepare_upstream import UPSTREAM_REV, verify
    from benchmarks.tabarena.provenance import (
        collect_provenance,
        file_hash,
        fingerprint,
    )
    from tabarena.benchmark.experiment.experiment_runner_api import job_cache_exists
    from tabarena.caching import CacheConfig

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
        metadata = resolved.context.metadata_for_jobs(resolved.jobs)
        for index, (job, meta) in enumerate(zip(resolved.jobs, metadata), 1):
            key = fingerprint(job.to_dict())
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
                write_event(
                    run_dir,
                    key,
                    "completed",
                    experiment=job.experiment.name,
                    task=job.task.as_triple(),
                    elapsed_seconds=time.monotonic() - started,
                    cache_hit=exists,
                    result_sha256=file_hash(cacher.cache_file),
                )
        return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "lite-defaults", "lite-hpo", "full"],
        default="smoke",
    )
    parser.add_argument("--threads", type=int)
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
        default=["AlloyGBM", "LightGBM", "XGBoost", "CatBoost", "RandomForest"],
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
    )
    directory = run_experiments(
        resolve_run(spec),
        output_root=args.output_root,
        receipt_path=args.build_receipt,
        resume=args.resume,
        retry_failed=args.retry_failed,
        plan_only=args.plan_only,
    )
    print(directory, flush=True)
    if any(r["status"] == "failed" for r in load_latest(directory).values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
