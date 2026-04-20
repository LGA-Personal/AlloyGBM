#!/usr/bin/env python3
"""CPU vs Metal training-throughput harness for AlloyGBM's Stage 1
Metal backend.

Stage 1 accelerates the histogram-build phase only; split finding,
partitioning, and prediction still run on the CPU path inside
`MetalBackend`. Because AlloyGBM doesn't expose a histogram-only
entry point to Python, this harness measures the whole fit() wall
clock — which is what a user sees when they flip `device="cpu"`
to `device="metal"`.

Shape grid follows the S1.14 plan: (rows) × (features) across
  rows:     10_000, 100_000, 1_000_000, 10_000_000
  features: 10, 100, 1000
plus a small fixed `--estimators` count (default 5). Large corners
are memory-gated with `--memory-budget-gb` (default 8 GB) so the
harness is runnable on a laptop without OOM'ing on the 10M × 1000
cell (~40 GB of float32 storage).

Output: a markdown table to stdout with columns
  rows  features  cpu_s  metal_s  speedup  metal_vs_cpu
optionally mirrored to a JSON file via `--json-out`.

Example
-------
  # quick (default grid, default 5 estimators)
  .venv/bin/python benchmarks/metal_histogram.py

  # full plan-spec grid, 20 estimators, JSON artifact
  .venv/bin/python benchmarks/metal_histogram.py \\
      --full --estimators 20 --json-out /tmp/metal_hist.json

  # explicit custom grid
  .venv/bin/python benchmarks/metal_histogram.py \\
      --rows 50000 500000 --features 32 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from alloygbm import GBMRegressor, native_runtime_info


DEFAULT_ROWS = (10_000, 100_000, 1_000_000)
FULL_ROWS = (10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_FEATURES = (10, 100, 1000)


@dataclass
class CellResult:
    rows: int
    features: int
    estimators: int
    bytes_input: int
    cpu_seconds: Optional[float]
    metal_seconds: Optional[float]
    speedup: Optional[float]  # cpu_seconds / metal_seconds
    note: str = ""


def _make_dataset(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic dense regression fixture.

    `float32` matches the wheel's default input dtype and halves
    memory vs `float64`. The target is a noisy linear combination
    of the first three columns — enough structure for the tree
    to find non-trivial splits and thus exercise the full fit
    pipeline, not just a degenerate single-node case.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(rows, features), dtype=np.float32)
    noise = rng.standard_normal(size=rows, dtype=np.float32) * 0.1
    y = (
        2.0 * X[:, 0]
        - X[:, 1 % features]
        + 0.5 * X[:, 2 % features]
        + noise
    ).astype(np.float32)
    return X, y


def _fit_seconds(device: str, X: np.ndarray, y: np.ndarray, *, estimators: int,
                  bins: int, seed: int) -> float:
    """Fit a single regressor and return wall-clock seconds."""
    model = GBMRegressor(
        n_estimators=estimators,
        seed=seed,
        deterministic=True,
        continuous_binning_max_bins=bins,
        device=device,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0


def _warmup_metal(bins: int, estimators: int, seed: int) -> None:
    """First Metal fit pays one-time pipeline-compilation cost.

    Warming up with a tiny dataset amortises the compile across
    the benchmark grid rather than poisoning the first real cell's
    reading.
    """
    X, y = _make_dataset(rows=1024, features=4, seed=seed)
    try:
        _fit_seconds("metal", X, y, estimators=estimators, bins=bins, seed=seed)
    except Exception as exc:  # pragma: no cover - best effort
        warnings.warn(f"Metal warmup failed: {exc}", RuntimeWarning, stacklevel=2)


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
    # Speedup = cpu / metal. Values < 1.0 mean Metal is slower
    # than CPU (expected below the crossover); values > 1.0 mean
    # Metal is faster.
    return f"{value:.2f}x"


def run_grid(
    rows_grid: tuple[int, ...],
    features_grid: tuple[int, ...],
    *,
    estimators: int,
    bins: int,
    seed: int,
    memory_budget_bytes: int,
    metal_available: bool,
    skip_metal: bool,
) -> list[CellResult]:
    results: list[CellResult] = []
    for rows in rows_grid:
        for features in features_grid:
            bytes_input = rows * features * 4  # float32
            cell = CellResult(
                rows=rows,
                features=features,
                estimators=estimators,
                bytes_input=bytes_input,
                cpu_seconds=None,
                metal_seconds=None,
                speedup=None,
            )
            if bytes_input > memory_budget_bytes:
                gb = bytes_input / 1024**3
                cell.note = f"skipped: {gb:.1f} GB exceeds --memory-budget-gb"
                results.append(cell)
                print(
                    f"[skip] rows={rows:>9} features={features:>5}  "
                    f"({gb:.1f} GB > budget)",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            print(
                f"[run ] rows={rows:>9} features={features:>5}  "
                f"(input={bytes_input / 1024**2:.0f} MiB)",
                file=sys.stderr,
                flush=True,
            )
            X, y = _make_dataset(rows=rows, features=features, seed=seed)

            try:
                cell.cpu_seconds = _fit_seconds(
                    "cpu", X, y, estimators=estimators, bins=bins, seed=seed
                )
            except Exception as exc:
                cell.note = f"cpu fit failed: {exc}"
                results.append(cell)
                continue

            if metal_available and not skip_metal:
                try:
                    cell.metal_seconds = _fit_seconds(
                        "metal", X, y, estimators=estimators, bins=bins, seed=seed
                    )
                except Exception as exc:
                    cell.note = f"metal fit failed: {exc}"

            if cell.cpu_seconds and cell.metal_seconds:
                cell.speedup = cell.cpu_seconds / cell.metal_seconds

            results.append(cell)
    return results


def render_markdown(results: list[CellResult]) -> str:
    lines = [
        "| rows | features | input MiB | cpu | metal | speedup | note |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for r in results:
        mib = r.bytes_input / 1024**2
        lines.append(
            f"| {r.rows:,} | {r.features:,} | {mib:,.0f} | "
            f"{_format_seconds(r.cpu_seconds)} | "
            f"{_format_seconds(r.metal_seconds)} | "
            f"{_format_speedup(r.speedup)} | {r.note} |"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help=f"row sizes (default {DEFAULT_ROWS}; --full adds 10M)",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help=f"feature counts (default {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the plan-spec grid including 10M rows (very memory-heavy)",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=5,
        help="number of boosting rounds per fit (default 5)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=255,
        help="continuous_binning_max_bins (default 255; triggers u8 path)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=8.0,
        help="skip shapes whose float32 input exceeds this budget (default 8 GB)",
    )
    parser.add_argument(
        "--skip-metal",
        action="store_true",
        help="only run the CPU leg (useful for debugging the harness)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the Metal pipeline warmup fit",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="also write the results as JSON to this path",
    )
    args = parser.parse_args(argv)

    if args.rows is None:
        rows_grid = tuple(FULL_ROWS if args.full else DEFAULT_ROWS)
    else:
        rows_grid = tuple(args.rows)
    features_grid = tuple(args.features) if args.features is not None else DEFAULT_FEATURES

    info = native_runtime_info()
    metal_available = bool(info.metal_available)

    print(
        f"# AlloyGBM Metal-histogram throughput  "
        f"({datetime.now(timezone.utc).isoformat(timespec='seconds')})",
        file=sys.stderr,
    )
    print(
        f"# gpu_family={info.gpu_family!r}  metal_available={metal_available}  "
        f"metal4={info.metal4_available}",
        file=sys.stderr,
    )
    print(
        f"# estimators={args.estimators}  bins={args.bins}  "
        f"rows={rows_grid}  features={features_grid}  "
        f"budget={args.memory_budget_gb} GB",
        file=sys.stderr,
    )

    if metal_available and not args.skip_metal and not args.no_warmup:
        print("# warming up Metal pipeline cache ...", file=sys.stderr, flush=True)
        _warmup_metal(bins=args.bins, estimators=args.estimators, seed=args.seed)

    results = run_grid(
        rows_grid=rows_grid,
        features_grid=features_grid,
        estimators=args.estimators,
        bins=args.bins,
        seed=args.seed,
        memory_budget_bytes=int(args.memory_budget_gb * 1024**3),
        metal_available=metal_available,
        skip_metal=args.skip_metal,
    )

    print(render_markdown(results))

    if args.json_out is not None:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_family": info.gpu_family,
            "metal_available": metal_available,
            "metal4_available": bool(info.metal4_available),
            "estimators": args.estimators,
            "bins": args.bins,
            "seed": args.seed,
            "memory_budget_gb": args.memory_budget_gb,
            "results": [asdict(r) for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"# wrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
