#!/usr/bin/env python3
"""Descriptive multiclass DART scratch-scaling benchmark.

This deliberately combines a high class count with a low DART drop cap. It
measures the regime where only a few class slices contain material dropped
trees while an all-class scratch clear/finalize would still touch every row.
Timing is descriptive only; this harness has no CI performance gate.

Usage:
    .venv/bin/python benchmarks/multiclass_dart_scratch_benchmark.py
    .venv/bin/python benchmarks/multiclass_dart_scratch_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TimingRow:
    arm: str
    seconds: tuple[float, ...]

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.seconds)


def make_fixture(
    *,
    rows: int,
    features: int,
    classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rows < classes:
        raise ValueError("rows must be >= classes")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    projection = X[:, : min(features, 4)].sum(axis=1)
    order = np.argsort(projection, kind="mergesort")
    y = np.empty(rows, dtype=np.int64)
    y[order] = np.arange(rows, dtype=np.int64) % classes
    return X, y


def _fit_once(
    X: np.ndarray,
    y: np.ndarray,
    *,
    arm: str,
    rounds: int,
    max_drop: int,
    seed: int,
) -> float:
    from alloygbm import GBMClassifier

    params: dict[str, object] = {
        "n_estimators": rounds,
        "max_depth": 2,
        "min_data_in_leaf": 2,
        "learning_rate": 0.06,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "deterministic": True,
        "seed": seed,
    }
    if arm == "dart":
        params.update(
            {
                "boosting_mode": "dart",
                "dart_drop_rate": 0.75,
                "dart_max_drop": max_drop,
            }
        )
    elif arm != "standard":
        raise ValueError(f"unsupported arm: {arm}")

    model = GBMClassifier(**params)
    started = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba(X[:8])
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{arm} produced non-finite probabilities")
    return elapsed


def run_benchmark(
    *,
    rows: int,
    features: int,
    classes: int,
    rounds: int,
    max_drop: int,
    repeats: int,
    seed: int,
) -> list[TimingRow]:
    X, y = make_fixture(rows=rows, features=features, classes=classes, seed=seed)
    # Warm native code and allocation paths outside measured repetitions.
    _fit_once(
        X,
        y,
        arm="standard",
        rounds=max(2, rounds // 4),
        max_drop=max_drop,
        seed=seed,
    )
    rows_out = []
    for arm in ("standard", "dart"):
        timings = tuple(
            _fit_once(
                X,
                y,
                arm=arm,
                rounds=rounds,
                max_drop=max_drop,
                seed=seed,
            )
            for _ in range(repeats)
        )
        rows_out.append(TimingRow(arm=arm, seconds=timings))
    return rows_out


def render_report(
    timing_rows: Sequence[TimingRow],
    *,
    rows: int,
    features: int,
    classes: int,
    rounds: int,
    max_drop: int,
    repeats: int,
    seed: int,
    source: str,
) -> str:
    by_arm = {row.arm: row for row in timing_rows}
    standard = by_arm["standard"].median_seconds
    dart = by_arm["dart"].median_seconds
    ratio = dart / standard
    lines = [
        "# Multiclass DART Scratch Benchmark",
        "",
        f"Source: `{source}`",
        "",
        f"Captured: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        (
            f"Environment: {platform.platform()}; Python {platform.python_version()}; "
            f"{platform.machine()}."
        ),
        "",
        (
            f"Profile: {rows} rows, {features} features, {classes} classes, "
            f"{rounds} rounds, `dart_max_drop={max_drop}`, {repeats} repetitions, "
            f"seed {seed}."
        ),
        "",
        "| Arm | Timings (s) | Median (s) |",
        "|---|---:|---:|",
    ]
    for row in timing_rows:
        rendered = ", ".join(f"{value:.6f}" for value in row.seconds)
        lines.append(f"| `{row.arm}` | {rendered} | {row.median_seconds:.6f} |")
    lines.extend(
        [
            "",
            f"Descriptive DART/standard median ratio: `{ratio:.3f}x`.",
            "",
            "No timing threshold is enforced by this harness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", default="working tree")
    args = parser.parse_args()

    profile = {
        "rows": 512 if args.quick else 2_048,
        "features": 8 if args.quick else 16,
        "classes": 32 if args.quick else 64,
        "rounds": 6 if args.quick else 16,
        "max_drop": 2,
        "repeats": 3 if args.quick else 5,
        "seed": 29,
    }
    timing_rows = run_benchmark(**profile)
    report = render_report(timing_rows, source=args.source, **profile)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
