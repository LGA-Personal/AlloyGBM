"""Measure seed-to-seed variance of the curated comparison suite.

`run_model_comparison.py` reports one seed per scenario. That is enough to
rank libraries on a stable scenario, but not to judge whether a single
scenario's movement between two builds is real: a change that shifts split
boundaries reshuffles which splits are available, and on a high-variance
scenario that alone can move a metric by double digits.

This runs the curated suite across several seeds and reports the spread. The
spread is the noise floor: a before/after delta smaller than it is not
evidence of anything, whichever direction it points.

`--seed` drives both the train/test split and every library's own seed, so the
spread covers data partition and model randomness together. Scenarios that
split chronologically (`panel_time_series`, `dow_jones_financial`) do not
repartition with the seed, so their spread reflects model randomness only and
will read narrower for a structural reason rather than because they are more
stable.

Usage:

    python benchmarks/curated_seed_variance.py --seeds 5 --threads 1

Results land in `--output-dir` as one subdirectory per seed plus a
`seed_variance_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# Metric per task type, and whether smaller is better.
METRIC_FOR_TASK = {
    "regression": "rmse",
    "classification": "log_loss_val",
    "multiclass_classification": "log_loss_val",
    "ranking": "ndcg_10",
}
LOWER_IS_BETTER = {"rmse", "mae", "log_loss_val"}

# Chronological splits ignore the seed, so their spread is model-randomness
# only. Flagged in the report rather than excluded.
TIME_SPLIT_SCENARIOS = {"panel_time_series", "dow_jones_financial"}


def run_one_seed(seed: int, threads: int, output_dir: Path, extra: list[str]) -> Path:
    target = output_dir / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "benchmarks/run_model_comparison.py",
        "--threads", str(threads),
        "--seed", str(seed),
        "--output-dir", str(target),
        *extra,
    ]
    print(f"  seed {seed}: running...", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-5:]
        raise SystemExit(f"seed {seed} failed:\n" + "\n".join(tail))
    return target / "model_comparison_latest.json"


def load_records(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    return value["records"] if isinstance(value, dict) and "records" in value else value


def summarize(per_seed: dict[int, list[dict]]) -> dict:
    """Collect each (scenario, model) metric across seeds."""
    collected: dict[tuple[str, str], dict] = {}
    for seed, records in per_seed.items():
        for record in records:
            task_type = record.get("task_type")
            if task_type not in METRIC_FOR_TASK:
                # Silently dropping a scenario is how a variance report ends up
                # describing fewer scenarios than it claims. Fail instead.
                raise SystemExit(
                    f"unknown task_type {task_type!r} for scenario "
                    f"{record.get('scenario')!r}; add it to METRIC_FOR_TASK"
                )
            metric_name = METRIC_FOR_TASK[task_type]
            value = record.get(metric_name)
            if not isinstance(value, (int, float)) or value != value:
                continue
            key = (record["scenario"], record["model"])
            entry = collected.setdefault(
                key, {"scenario": key[0], "model": key[1],
                      "metric": metric_name, "by_seed": {}}
            )
            entry["by_seed"][str(seed)] = float(value)

    for entry in collected.values():
        values = list(entry["by_seed"].values())
        median = statistics.median(values)
        entry["n_seeds"] = len(values)
        entry["median"] = median
        entry["min"] = min(values)
        entry["max"] = max(values)
        # Spread as a percentage of the median, in the same "improvement"
        # units the before/after comparisons use, so the two are comparable.
        entry["spread_pct"] = (
            (max(values) - min(values)) / abs(median) * 100.0 if median else float("nan")
        )
        entry["time_split"] = entry["scenario"] in TIME_SPLIT_SCENARIOS
    return {f"{k[0]}|{k[1]}": v for k, v in collected.items()}


def report(summary: dict, focus_model: str) -> None:
    rows = [v for v in summary.values() if v["model"] == focus_model]
    rows.sort(key=lambda r: r["spread_pct"], reverse=True)
    print(f"\n=== seed-to-seed spread for `{focus_model}` (noise floor) ===")
    # Absolute min/max are printed alongside the percentage because relative
    # spread explodes when the metric itself is near zero -- a log-loss of
    # 0.08 swinging to 0.20 is 154% but may matter less than a 2% swing on a
    # large RMSE. Both numbers are needed to judge a delta.
    print(
        f"{'scenario':26} {'metric':12} {'median':>10} {'min':>10} {'max':>10} "
        f"{'spread %':>9}  note"
    )
    for row in rows:
        note = "chronological split (model randomness only)" if row["time_split"] else ""
        print(
            f"{row['scenario']:26} {row['metric']:12} {row['median']:10.5f} "
            f"{row['min']:10.5f} {row['max']:10.5f} {row['spread_pct']:8.2f}%  {note}"
        )
    if rows:
        spreads = [r["spread_pct"] for r in rows if r["spread_pct"] == r["spread_pct"]]
        print(
            f"\n  median scenario spread: {statistics.median(spreads):.2f}%   "
            f"max: {max(spreads):.2f}% ({max(rows, key=lambda r: r['spread_pct'])['scenario']})"
        )
        print(
            "\n  Read a before/after delta as signal only when it clearly exceeds "
            "this scenario's spread."
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--base-seed", type=int, default=20260906)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--focus-model", default="alloygbm")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/seed_variance")
    )
    parser.add_argument(
        "--runner-arg", action="append", default=[],
        help="extra argument forwarded to run_model_comparison.py (repeatable)",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + offset for offset in range(args.seeds)]
    print(f"curated suite across {len(seeds)} seeds at {args.threads} thread(s): {seeds}")

    per_seed: dict[int, list[dict]] = {}
    for seed in seeds:
        path = run_one_seed(seed, args.threads, args.output_dir, args.runner_arg)
        per_seed[seed] = load_records(path)
        print(f"  seed {seed}: {len(per_seed[seed])} records", flush=True)

    summary = summarize(per_seed)
    out = args.output_dir / "seed_variance_summary.json"
    out.write_text(json.dumps({"seeds": seeds, "threads": args.threads,
                               "entries": summary}, indent=2))
    report(summary, args.focus_model)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
