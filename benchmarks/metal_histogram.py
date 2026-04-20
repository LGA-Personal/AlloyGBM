#!/usr/bin/env python3
"""CPU vs Metal training-throughput harness for AlloyGBM's Stage 1
Metal backend.

Stage 1 accelerates the histogram-build phase only; split finding,
partitioning, and prediction still run on the CPU path inside
`MetalBackend`. Because AlloyGBM doesn't expose a histogram-only
entry point to Python, this harness measures the whole fit() wall
clock — which is what a user sees when they flip `device="cpu"`
to `device="metal"`.

The harness exposes named **scenarios**, each of which isolates
one parameter likely to shift the CPU/Metal balance:

  shape_grid      — (rows × features) matrix, regression task
  depth_sweep     — fixed shape, varying `max_depth`
  bins_sweep      — fixed shape, varying `continuous_binning_max_bins`
  estimator_sweep — fixed shape, varying `n_estimators`
  task_mix        — fixed shape, varying estimator type
                    (regression / binary / multiclass / ranking)
  metal_friendly  — the single config we expect to be best for Metal
                    (big rows, many features, deep trees, many bins)
  all             — runs every scenario in sequence

Output: a markdown table per scenario on stdout, optionally mirrored
to a JSON file via `--json-out`. Large shapes are memory-gated via
`--memory-budget-gb` (default 8 GB).

Examples
--------
  # Default: the shape grid on regression (S1.14 baseline)
  .venv/bin/python benchmarks/metal_histogram.py

  # Characterisation sweep for deciding whether Stage 1 Metal ever wins
  .venv/bin/python benchmarks/metal_histogram.py --scenario all \\
      --json-out /tmp/metal_bench.json

  # Isolate the single variable
  .venv/bin/python benchmarks/metal_histogram.py --scenario depth_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from alloygbm import (
    GBMClassifier,
    GBMRanker,
    GBMRegressor,
    native_runtime_info,
)


# ---------------------------------------------------------------
# Shape grids for `shape_grid` scenario
# ---------------------------------------------------------------
DEFAULT_ROWS = (10_000, 100_000, 1_000_000)
FULL_ROWS = (10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_FEATURES = (10, 100, 1000)

TASK_CHOICES = ("regression", "binary", "multiclass_3", "multiclass_10", "ranking")


# ---------------------------------------------------------------
# Result dataclasses — one per table row. `params` carries the
# scenario-specific axis labels (e.g. depth, bins, task name) so the
# renderer can dynamically size columns.
# ---------------------------------------------------------------
@dataclass
class Cell:
    params: dict[str, Any]
    bytes_input: int
    cpu_seconds: Optional[float]
    metal_seconds: Optional[float]
    speedup: Optional[float] = None
    note: str = ""


@dataclass
class ScenarioResult:
    name: str
    description: str
    column_order: list[str]
    cells: list[Cell] = field(default_factory=list)


# ---------------------------------------------------------------
# Dataset generators per task. Everything is float32 dense — same
# path users hit with numpy input.
# ---------------------------------------------------------------
def _make_regression(rows: int, features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(rows, features), dtype=np.float32)
    noise = rng.standard_normal(size=rows, dtype=np.float32) * 0.1
    y = (
        2.0 * X[:, 0]
        - X[:, 1 % features]
        + 0.5 * X[:, 2 % features]
        + noise
    ).astype(np.float32)
    return X, y, None


def _make_binary(rows: int, features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(rows, features), dtype=np.float32)
    logits = 1.5 * X[:, 0] - X[:, 1 % features] + 0.3 * X[:, 2 % features]
    y = (logits > 0).astype(np.int64)
    return X, y, None


def _make_multiclass(rows: int, features: int, seed: int, n_classes: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(rows, features), dtype=np.float32)
    # Assign class from a noisy linear combo of the first few features
    # mapped through argmax of `n_classes` linear projections.
    projections = rng.standard_normal(size=(features, n_classes), dtype=np.float32)
    scores = X @ projections
    y = np.argmax(scores, axis=1).astype(np.int64)
    return X, y, None


def _make_ranking(rows: int, features: int, seed: int, n_groups: int = 200):
    """Group-sorted ranking fixture. `n_groups` evenly-sized groups
    unless `rows` doesn't divide; the last group absorbs any remainder.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(rows, features), dtype=np.float32)
    group_size = max(1, rows // n_groups)
    sizes: list[int] = [group_size] * (rows // group_size)
    if sum(sizes) < rows:
        sizes.append(rows - sum(sizes))
    # Stable-sorted group IDs for the ranker contract
    group = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    # Labels: 5-level graded relevance drawn uniformly; noisy but
    # gives the tree real signal to split on.
    y = rng.integers(low=0, high=5, size=rows).astype(np.float32)
    return X, y, group


def _dataset_for(task: str, rows: int, features: int, seed: int):
    if task == "regression":
        return _make_regression(rows, features, seed)
    if task == "binary":
        return _make_binary(rows, features, seed)
    if task == "multiclass_3":
        return _make_multiclass(rows, features, seed, n_classes=3)
    if task == "multiclass_10":
        return _make_multiclass(rows, features, seed, n_classes=10)
    if task == "ranking":
        return _make_ranking(rows, features, seed)
    raise ValueError(f"unknown task: {task!r}")


# ---------------------------------------------------------------
# Fit dispatcher — routes to the right estimator per task and
# returns wall-clock seconds.
# ---------------------------------------------------------------
def _fit_seconds(
    task: str,
    device: str,
    X: np.ndarray,
    y: np.ndarray,
    group: Optional[np.ndarray],
    *,
    estimators: int,
    max_depth: int,
    bins: int,
    seed: int,
) -> float:
    kwargs = dict(
        learning_rate=0.1,
        n_estimators=estimators,
        max_depth=max_depth,
        seed=seed,
        deterministic=True,
        continuous_binning_max_bins=bins,
        device=device,
    )
    if task == "ranking":
        # `ranking_objective` belongs to Ranker only. `deterministic`
        # also isn't an accepted param on Ranker (per signature check
        # in the main block), so pop it.
        rkwargs = {k: v for k, v in kwargs.items() if k != "deterministic"}
        rkwargs["ranking_objective"] = "rank:ndcg"
        model = GBMRanker(**rkwargs)
        t0 = time.perf_counter()
        model.fit(X, y, group=group)
        return time.perf_counter() - t0
    if task == "regression":
        model = GBMRegressor(**kwargs)
    elif task in ("binary", "multiclass_3", "multiclass_10"):
        model = GBMClassifier(**kwargs)
    else:
        raise ValueError(f"unknown task: {task!r}")
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0


def _warmup_metal(*, seed: int) -> None:
    """Tiny (1024 × 4) regression fit so the first real cell doesn't
    absorb Metal's one-time pipeline-compilation cost.
    """
    X, y, _ = _make_regression(rows=1024, features=4, seed=seed)
    try:
        _fit_seconds(
            "regression", "metal", X, y, None,
            estimators=3, max_depth=4, bins=255, seed=seed,
        )
    except Exception as exc:  # pragma: no cover - best effort
        warnings.warn(f"Metal warmup failed: {exc}", RuntimeWarning, stacklevel=2)


# ---------------------------------------------------------------
# Cell runner — one row in the result table.
# ---------------------------------------------------------------
def _run_cell(
    *,
    task: str,
    rows: int,
    features: int,
    estimators: int,
    max_depth: int,
    bins: int,
    seed: int,
    memory_budget_bytes: int,
    metal_available: bool,
    skip_metal: bool,
    extra_params: Optional[dict[str, Any]] = None,
) -> Cell:
    bytes_input = rows * features * 4
    params: dict[str, Any] = {
        "task": task,
        "rows": rows,
        "features": features,
        "estimators": estimators,
        "max_depth": max_depth,
        "bins": bins,
    }
    if extra_params:
        params.update(extra_params)

    if bytes_input > memory_budget_bytes:
        gb = bytes_input / 1024**3
        note = f"skipped: {gb:.1f} GB > budget"
        print(
            f"[skip] task={task} rows={rows:>9} features={features:>5} "
            f"depth={max_depth} bins={bins} est={estimators}  ({gb:.1f} GB)",
            file=sys.stderr,
            flush=True,
        )
        return Cell(params=params, bytes_input=bytes_input,
                    cpu_seconds=None, metal_seconds=None, note=note)

    print(
        f"[run ] task={task} rows={rows:>9} features={features:>5} "
        f"depth={max_depth} bins={bins} est={estimators}  "
        f"(input={bytes_input / 1024**2:.0f} MiB)",
        file=sys.stderr,
        flush=True,
    )
    X, y, group = _dataset_for(task, rows, features, seed)

    cell = Cell(params=params, bytes_input=bytes_input,
                cpu_seconds=None, metal_seconds=None)
    try:
        cell.cpu_seconds = _fit_seconds(
            task, "cpu", X, y, group,
            estimators=estimators, max_depth=max_depth, bins=bins, seed=seed,
        )
    except Exception as exc:
        cell.note = f"cpu fit failed: {exc}"
        return cell

    if metal_available and not skip_metal:
        try:
            cell.metal_seconds = _fit_seconds(
                task, "metal", X, y, group,
                estimators=estimators, max_depth=max_depth, bins=bins, seed=seed,
            )
        except Exception as exc:
            cell.note = f"metal fit failed: {exc}"

    if cell.cpu_seconds and cell.metal_seconds:
        cell.speedup = cell.cpu_seconds / cell.metal_seconds
    return cell


# ---------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------
def scenario_shape_grid(
    *, rows_grid: tuple[int, ...], features_grid: tuple[int, ...],
    task: str, estimators: int, max_depth: int, bins: int, seed: int,
    memory_budget_bytes: int, metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    result = ScenarioResult(
        name="shape_grid",
        description=(
            f"task={task} depth={max_depth} est={estimators} bins={bins} "
            f"across rows × features"
        ),
        column_order=["rows", "features"],
    )
    for rows in rows_grid:
        for features in features_grid:
            result.cells.append(_run_cell(
                task=task, rows=rows, features=features,
                estimators=estimators, max_depth=max_depth, bins=bins,
                seed=seed, memory_budget_bytes=memory_budget_bytes,
                metal_available=metal_available, skip_metal=skip_metal,
            ))
    return result


def scenario_depth_sweep(
    *, rows: int, features: int, depths: tuple[int, ...],
    task: str, estimators: int, bins: int, seed: int,
    memory_budget_bytes: int, metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    result = ScenarioResult(
        name="depth_sweep",
        description=(
            f"task={task} rows={rows:,} features={features} est={estimators} "
            f"bins={bins}, sweeping max_depth"
        ),
        column_order=["max_depth"],
    )
    for depth in depths:
        result.cells.append(_run_cell(
            task=task, rows=rows, features=features,
            estimators=estimators, max_depth=depth, bins=bins,
            seed=seed, memory_budget_bytes=memory_budget_bytes,
            metal_available=metal_available, skip_metal=skip_metal,
        ))
    return result


def scenario_bins_sweep(
    *, rows: int, features: int, bins_grid: tuple[int, ...],
    task: str, estimators: int, max_depth: int, seed: int,
    memory_budget_bytes: int, metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    result = ScenarioResult(
        name="bins_sweep",
        description=(
            f"task={task} rows={rows:,} features={features} est={estimators} "
            f"depth={max_depth}, sweeping continuous_binning_max_bins"
        ),
        column_order=["bins"],
    )
    for bins in bins_grid:
        result.cells.append(_run_cell(
            task=task, rows=rows, features=features,
            estimators=estimators, max_depth=max_depth, bins=bins,
            seed=seed, memory_budget_bytes=memory_budget_bytes,
            metal_available=metal_available, skip_metal=skip_metal,
        ))
    return result


def scenario_estimator_sweep(
    *, rows: int, features: int, estimator_grid: tuple[int, ...],
    task: str, max_depth: int, bins: int, seed: int,
    memory_budget_bytes: int, metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    result = ScenarioResult(
        name="estimator_sweep",
        description=(
            f"task={task} rows={rows:,} features={features} depth={max_depth} "
            f"bins={bins}, sweeping n_estimators"
        ),
        column_order=["estimators"],
    )
    for est in estimator_grid:
        result.cells.append(_run_cell(
            task=task, rows=rows, features=features,
            estimators=est, max_depth=max_depth, bins=bins,
            seed=seed, memory_budget_bytes=memory_budget_bytes,
            metal_available=metal_available, skip_metal=skip_metal,
        ))
    return result


def scenario_task_mix(
    *, rows: int, features: int, tasks: tuple[str, ...],
    estimators: int, max_depth: int, bins: int, seed: int,
    memory_budget_bytes: int, metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    result = ScenarioResult(
        name="task_mix",
        description=(
            f"rows={rows:,} features={features} est={estimators} "
            f"depth={max_depth} bins={bins}, sweeping task type"
        ),
        column_order=["task"],
    )
    for task in tasks:
        result.cells.append(_run_cell(
            task=task, rows=rows, features=features,
            estimators=estimators, max_depth=max_depth, bins=bins,
            seed=seed, memory_budget_bytes=memory_budget_bytes,
            metal_available=metal_available, skip_metal=skip_metal,
        ))
    return result


def scenario_metal_friendly(
    *, seed: int, memory_budget_bytes: int,
    metal_available: bool, skip_metal: bool,
) -> ScenarioResult:
    """The single configuration we expect to be the BEST case for
    Stage 1 Metal: the histogram phase should dominate because of
    (many rows, many features, deep trees, many bins, multiclass).
    If Metal doesn't win here, Stage 1 Metal likely doesn't win
    anywhere — we'd need Stage 2 to move the needle.
    """
    result = ScenarioResult(
        name="metal_friendly",
        description=(
            "configurations theoretically most favourable to Stage 1 Metal: "
            "many rows × many features × deep trees × many bins × multiclass"
        ),
        column_order=["task", "rows", "features", "max_depth", "bins"],
    )
    candidates = [
        # (task, rows, features, estimators, depth, bins)
        # Kept modest on purpose: Stage 1 per-call buffer-copy overhead
        # scales with `rows × features × build_histograms-calls`, so
        # very deep trees at big shapes blow the budget. These are the
        # shapes where Metal should be *most* favourable — if it loses
        # here, it loses everywhere under Stage 1's current framing.
        ("regression",    200_000, 200, 5,  8,  255),
        ("regression",    200_000, 200, 5, 10,  255),
        ("regression",    200_000, 200, 5,  6, 1024),
        ("multiclass_3",  100_000, 100, 5,  8,  255),
        ("multiclass_10", 100_000, 100, 5,  8,  255),
    ]
    for task, rows, features, estimators, depth, bins in candidates:
        result.cells.append(_run_cell(
            task=task, rows=rows, features=features,
            estimators=estimators, max_depth=depth, bins=bins,
            seed=seed, memory_budget_bytes=memory_budget_bytes,
            metal_available=metal_available, skip_metal=skip_metal,
        ))
    return result


# ---------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------
def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= 10.0:
        return f"{value:.1f}s"
    if value >= 1.0:
        return f"{value:.2f}s"
    return f"{value * 1000:.0f}ms"


def _format_speedup(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}x"


def render_scenario(result: ScenarioResult) -> str:
    axis_cols = list(result.column_order)
    headers = axis_cols + ["input MiB", "cpu", "metal", "speedup", "note"]
    align = [":---" if c == "task" else "---:" for c in axis_cols]
    align += ["---:", "---:", "---:", "---:", ":---"]
    lines = [
        f"### Scenario: `{result.name}`",
        f"_{result.description}_",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for cell in result.cells:
        axis_vals = [_axis_value(cell.params.get(c), c) for c in axis_cols]
        mib = f"{cell.bytes_input / 1024**2:,.0f}"
        row = axis_vals + [
            mib,
            _format_seconds(cell.cpu_seconds),
            _format_seconds(cell.metal_seconds),
            _format_speedup(cell.speedup),
            cell.note,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _axis_value(value: Any, column: str) -> str:
    if value is None:
        return ""
    if column in ("rows", "features", "estimators", "bins", "max_depth"):
        return f"{int(value):,}"
    return str(value)


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
SCENARIO_CHOICES = (
    "shape_grid",
    "depth_sweep",
    "bins_sweep",
    "estimator_sweep",
    "task_mix",
    "metal_friendly",
    "all",
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="shape_grid")
    parser.add_argument("--task", choices=TASK_CHOICES, default="regression",
                        help="task for shape_grid / depth_sweep / bins_sweep / estimator_sweep")
    parser.add_argument("--rows", type=int, nargs="+", default=None,
                        help=f"shape_grid row sizes (default {DEFAULT_ROWS})")
    parser.add_argument("--features", type=int, nargs="+", default=None,
                        help=f"shape_grid feature counts (default {DEFAULT_FEATURES})")
    parser.add_argument("--full", action="store_true",
                        help="shape_grid: include 10M rows")
    parser.add_argument("--estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--bins", type=int, default=255)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--memory-budget-gb", type=float, default=8.0)
    parser.add_argument("--skip-metal", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)

    # Scenario-specific knobs
    parser.add_argument("--depth-sweep-rows", type=int, default=200_000)
    parser.add_argument("--depth-sweep-features", type=int, default=100)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--bins-sweep-rows", type=int, default=200_000)
    parser.add_argument("--bins-sweep-features", type=int, default=100)
    parser.add_argument("--bins-grid", type=int, nargs="+", default=[32, 64, 255, 1024])
    parser.add_argument("--estimator-sweep-rows", type=int, default=200_000)
    parser.add_argument("--estimator-sweep-features", type=int, default=100)
    parser.add_argument("--estimator-grid", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument("--task-mix-rows", type=int, default=200_000)
    parser.add_argument("--task-mix-features", type=int, default=100)

    args = parser.parse_args(argv)

    rows_grid = tuple(args.rows) if args.rows else tuple(
        FULL_ROWS if args.full else DEFAULT_ROWS
    )
    features_grid = tuple(args.features) if args.features else DEFAULT_FEATURES

    info = native_runtime_info()
    metal_available = bool(info.metal_available)

    print(
        f"# AlloyGBM Metal characterisation  "
        f"({datetime.now(timezone.utc).isoformat(timespec='seconds')})",
        file=sys.stderr,
    )
    print(
        f"# gpu_family={info.gpu_family!r}  metal_available={metal_available}  "
        f"metal4={info.metal4_available}",
        file=sys.stderr,
    )
    print(
        f"# scenario={args.scenario}  task={args.task}  "
        f"est={args.estimators}  depth={args.max_depth}  bins={args.bins}  "
        f"budget={args.memory_budget_gb} GB",
        file=sys.stderr,
    )

    if metal_available and not args.skip_metal and not args.no_warmup:
        print("# warming up Metal pipeline cache ...", file=sys.stderr, flush=True)
        _warmup_metal(seed=args.seed)

    budget_bytes = int(args.memory_budget_gb * 1024**3)
    scenarios_to_run: list[str]
    if args.scenario == "all":
        scenarios_to_run = [
            "shape_grid", "depth_sweep", "bins_sweep",
            "estimator_sweep", "task_mix", "metal_friendly",
        ]
    else:
        scenarios_to_run = [args.scenario]

    all_results: list[ScenarioResult] = []

    for name in scenarios_to_run:
        print(f"\n# --- scenario: {name} ---", file=sys.stderr, flush=True)
        if name == "shape_grid":
            result = scenario_shape_grid(
                rows_grid=rows_grid, features_grid=features_grid,
                task=args.task, estimators=args.estimators,
                max_depth=args.max_depth, bins=args.bins, seed=args.seed,
                memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        elif name == "depth_sweep":
            result = scenario_depth_sweep(
                rows=args.depth_sweep_rows, features=args.depth_sweep_features,
                depths=tuple(args.depths),
                task=args.task, estimators=args.estimators, bins=args.bins,
                seed=args.seed, memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        elif name == "bins_sweep":
            result = scenario_bins_sweep(
                rows=args.bins_sweep_rows, features=args.bins_sweep_features,
                bins_grid=tuple(args.bins_grid),
                task=args.task, estimators=args.estimators,
                max_depth=args.max_depth, seed=args.seed,
                memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        elif name == "estimator_sweep":
            result = scenario_estimator_sweep(
                rows=args.estimator_sweep_rows, features=args.estimator_sweep_features,
                estimator_grid=tuple(args.estimator_grid),
                task=args.task, max_depth=args.max_depth, bins=args.bins,
                seed=args.seed, memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        elif name == "task_mix":
            result = scenario_task_mix(
                rows=args.task_mix_rows, features=args.task_mix_features,
                tasks=TASK_CHOICES, estimators=args.estimators,
                max_depth=args.max_depth, bins=args.bins, seed=args.seed,
                memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        elif name == "metal_friendly":
            result = scenario_metal_friendly(
                seed=args.seed, memory_budget_bytes=budget_bytes,
                metal_available=metal_available, skip_metal=args.skip_metal,
            )
        else:
            raise RuntimeError(f"unhandled scenario {name!r}")

        print(render_scenario(result))
        all_results.append(result)

    if args.json_out is not None:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_family": info.gpu_family,
            "metal_available": metal_available,
            "metal4_available": bool(info.metal4_available),
            "seed": args.seed,
            "memory_budget_gb": args.memory_budget_gb,
            "scenarios": [
                {
                    "name": r.name,
                    "description": r.description,
                    "column_order": r.column_order,
                    "cells": [asdict(c) for c in r.cells],
                }
                for r in all_results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n# wrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
