"""Sweep auto-policy rules on the dev suite with a pool of single-threaded jobs.

Each arm is an AlloyGBM configuration at its defaults plus constructor
overrides and/or ``ALLOYGBM_EXPERIMENT_AUTO_*`` environment overrides (the
engine reads those per fit, so an arm that needs one runs in its own
process). Every (arm, dev task) pair is one single-threaded
``frontier_matrix.py run`` job; ``--workers`` jobs run at once, largest
first, and finished jobs are kept, so an interrupted sweep resumes. Each
arm's jobs are merged into ``OUT/<arm>.json`` for ``special_modes.py``::

    python benchmarks/auto_defaults_sweep.py --cache DEV_DIR --out-dir OUT \\
        --arm alloy --arm "rs07=@AUTO_ROW_SUBSAMPLE=0.7" --arm "d8=max_depth=8"
    python benchmarks/special_modes.py OUT/*.json --baseline alloy

An arm is ``name=item,item,...`` where an item is ``key=value`` (a
constructor parameter, JSON-parsed) or ``@NAME=value`` (the environment
variable ``ALLOYGBM_EXPERIMENT_NAME``). A bare ``alloy`` is the defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(spec: str) -> tuple[str, str, dict[str, str]]:
    """``(arm name, frontier_matrix arm spec, extra environment)``."""
    if "=" not in spec:
        return spec, spec, {}
    name, _, rest = spec.partition("=")
    params, env = [], {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        if key.startswith("@"):
            env[f"ALLOYGBM_EXPERIMENT_{key[1:]}"] = value
        else:
            params.append(f"{key}={value}")
    return name, f"{name}=alloygbm:{','.join(params)}", env


def cost(task: dict) -> float:
    """Rough fit cost of a dev task: capped rows x features x sqrt(classes)."""
    return min(task["rows"], 20_000) * max(task["features"], 1) * max(task.get("classes", 1), 1) ** 0.5


def merge(out_dir: Path, name: str, task_ids: list[int], env: dict) -> Path | None:
    parts = [out_dir / name / f"{tid}.json" for tid in task_ids]
    if not all(p.exists() for p in parts):
        return None
    merged = None
    for p in parts:
        part = json.loads(p.read_text())
        if merged is None:
            merged = part
        else:
            merged["records"] += part["records"]
    merged["arm_env"] = env
    final = out_dir / f"{name}.json"
    final.write_text(json.dumps(merged, indent=1))
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tasks", type=int, nargs="+", help="only these dev task ids")
    p.add_argument("--skip-tasks", type=int, nargs="+", default=[], help="leave these dev task ids out")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--train-cap", type=int)
    args = p.parse_args(argv)

    manifest = json.loads((HERE / "dev_suite_manifest.json").read_text())
    tasks = [t for t in manifest["tasks"] if (args.cache / f"{t['task_id']}.npz").exists()
             and (not args.tasks or t["task_id"] in args.tasks)
             and t["task_id"] not in args.skip_tasks]
    task_ids = [t["task_id"] for t in tasks]
    arms = [parse(spec) for spec in args.arm]
    jobs = []  # (cost, arm index, task)
    for a, (name, _, _) in enumerate(arms):
        (args.out_dir / name).mkdir(parents=True, exist_ok=True)
        for t in tasks:
            if not (args.out_dir / name / f"{t['task_id']}.json").exists():
                jobs.append((cost(t), a, t))
    # Arms in the order given; within an arm, largest tasks first so the
    # pool's tail is short tasks.
    jobs.sort(key=lambda j: (j[1], -j[0]))
    running: list[tuple[subprocess.Popen, int, dict, Path]] = []
    pending_arms = {name for name, _, _ in arms}

    def reap(block: bool) -> None:
        while running:
            for item in list(running):
                proc, a, t, tmp = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                name = arms[a][0]
                if proc.returncode == 0:
                    tmp.rename(args.out_dir / name / f"{t['task_id']}.json")
                else:
                    print(f"{name} {t['task_id']}: failed ({proc.returncode})", flush=True)
            if not block or len(running) < args.workers:
                return
            time.sleep(1)

    def report_finished() -> None:
        for name, _, env in arms:
            if name in pending_arms:
                final = merge(args.out_dir, name, task_ids, env)
                if final is not None:
                    pending_arms.discard(name)
                    recs = json.loads(final.read_text())["records"]
                    errors = sum(r.get("error") is not None for r in recs)
                    fit = sum(r.get("fit_seconds") or 0 for r in recs)
                    print(f"{name}: {len(recs)} records, {errors} errors, fit {fit:.0f}s", flush=True)

    for _, a, t in jobs:
        reap(block=True)
        report_finished()
        name, arm_spec, env = arms[a]
        tmp = args.out_dir / name / f".{t['task_id']}.json"
        cmd = [sys.executable, str(HERE / "frontier_matrix.py"), "run", "--dev-suite",
               str(args.cache), "--regimes", "defaults", "--arms", arm_spec, "--threads", "1",
               "--seeds", str(args.seeds), "--dev-tasks", str(t["task_id"]), "--output", str(tmp)]
        if args.train_cap:
            cmd += ["--dev-train-cap", str(args.train_cap)]
        running.append((subprocess.Popen(cmd, env=os.environ | env, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL), a, t, tmp))
    while running:
        reap(block=False)
        time.sleep(1)
    report_finished()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
