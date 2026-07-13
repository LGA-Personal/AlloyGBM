#!/usr/bin/env python3
"""Deterministic clean-holdout robustness benchmark for DRO leaf solvers.

The benchmark compares standard and ``leaf_solver="dro"`` models after fitting
on clean and contaminated labels. It evaluates every model against the clean
held-out target, so the reported corruption penalty measures sensitivity to
training-label outliers rather than fit to the contaminated observations.

Usage:
    .venv/bin/python benchmarks/dro_robustness.py
    .venv/bin/python benchmarks/dro_robustness.py --seeds 7,13 --quick
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEFAULT_SEEDS = (7, 13, 29, 47, 61)
DEFAULT_RADIUS = 0.05
DEFAULT_CONTAMINATION_FRACTION = 0.12


@dataclass(frozen=True)
class BenchmarkRow:
    """Clean-holdout scores for one solver and deterministic data seed."""

    seed: int
    solver: str
    clean_train_rmse: float
    corrupted_train_rmse: float

    @property
    def corruption_penalty(self) -> float:
        return self.corrupted_train_rmse - self.clean_train_rmse


def _signal(X: np.ndarray) -> np.ndarray:
    return (
        1.5 * np.sin(X[:, 0])
        + 0.8 * X[:, 1] * X[:, 2]
        + 0.45 * X[:, 3] ** 2
        - 0.35 * X[:, 4]
    ).astype(np.float32)


def _contaminate(
    rng: np.random.Generator,
    y_clean: np.ndarray,
    contamination_fraction: float,
) -> np.ndarray:
    if not 0.0 < contamination_fraction < 1.0:
        raise ValueError("contamination_fraction must be in (0, 1)")
    contaminated = y_clean.copy()
    count = max(1, int(round(contamination_fraction * len(contaminated))))
    rows = rng.choice(len(contaminated), size=count, replace=False)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=count)
    magnitude = 4.0 + np.abs(rng.normal(loc=0.0, scale=2.0, size=count))
    contaminated[rows] += (signs * magnitude).astype(np.float32)
    return contaminated


def make_noisy_regression(
    *,
    seed: int,
    n_train: int = 600,
    n_test: int = 400,
    contamination_fraction: float = DEFAULT_CONTAMINATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return clean and corrupted training labels plus a clean holdout target."""
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, 6)).astype(np.float32)
    X_test = rng.normal(size=(n_test, 6)).astype(np.float32)
    y_clean_train = (_signal(X_train) + rng.normal(scale=0.12, size=n_train)).astype(np.float32)
    y_clean_test = (_signal(X_test) + rng.normal(scale=0.12, size=n_test)).astype(np.float32)
    y_corrupted_train = _contaminate(rng, y_clean_train, contamination_fraction)
    return X_train, y_clean_train, y_corrupted_train, X_test, y_clean_test


def make_noisy_joint_regression(
    *,
    seed: int,
    n_train: int = 600,
    n_test: int = 400,
    contamination_fraction: float = DEFAULT_CONTAMINATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a two-output shared-tree fixture with training-only outliers."""
    rng = np.random.default_rng(seed + 10_000)
    X_train = rng.normal(size=(n_train, 6)).astype(np.float32)
    X_test = rng.normal(size=(n_test, 6)).astype(np.float32)
    y_clean_train = np.column_stack(
        (
            _signal(X_train) + rng.normal(scale=0.12, size=n_train),
            0.9 * np.cos(X_train[:, 0]) - 0.7 * X_train[:, 1] * X_train[:, 4]
            + 0.25 * X_train[:, 5] ** 2
            + rng.normal(scale=0.12, size=n_train),
        )
    ).astype(np.float32)
    y_clean_test = np.column_stack(
        (
            _signal(X_test) + rng.normal(scale=0.12, size=n_test),
            0.9 * np.cos(X_test[:, 0]) - 0.7 * X_test[:, 1] * X_test[:, 4]
            + 0.25 * X_test[:, 5] ** 2
            + rng.normal(scale=0.12, size=n_test),
        )
    ).astype(np.float32)
    y_corrupted_train = np.column_stack(
        tuple(
            _contaminate(rng, y_clean_train[:, output], contamination_fraction)
            for output in range(y_clean_train.shape[1])
        )
    ).astype(np.float32)
    return X_train, y_clean_train, y_corrupted_train, X_test, y_clean_test


def _rmse(y_true: np.ndarray, prediction: object) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(residual * residual)))


def _fit_scalar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    solver: str,
    radius: float,
) -> np.ndarray:
    from alloygbm import GBMRegressor

    params: dict[str, object] = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.06,
        "lambda_l2": 1.0,
        "seed": seed,
    }
    if solver == "dro":
        params.update(
            leaf_solver="dro", dro_radius=radius, dro_metric="wasserstein"
        )
    return np.asarray(GBMRegressor(**params).fit(X_train, y_train).predict(X_test))


def _fit_joint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    solver: str,
    radius: float,
) -> np.ndarray:
    from alloygbm import MultiLabelGBMRanker

    params: dict[str, object] = {
        "ranking_objective": "squared_error",
        "multi_label_mode": "joint",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.06,
        "lambda_l2": 1.0,
        "seed": seed,
    }
    if solver == "dro":
        params.update(
            leaf_solver="dro", dro_radius=radius, dro_metric="wasserstein"
        )
    return np.asarray(MultiLabelGBMRanker(**params).fit(X_train, y_train).predict(X_test))


def _run_case(
    fixture: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    fit: Callable[..., np.ndarray],
    *,
    seed: int,
    radius: float,
    contamination_fraction: float,
    n_train: int,
    n_test: int,
) -> list[BenchmarkRow]:
    X_train, y_clean_train, y_corrupted_train, X_test, y_clean_test = fixture(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        contamination_fraction=contamination_fraction,
    )
    rows = []
    for solver in ("standard", "dro"):
        clean_prediction = fit(
            X_train, y_clean_train, X_test, seed=seed, solver=solver, radius=radius
        )
        corrupted_prediction = fit(
            X_train, y_corrupted_train, X_test, seed=seed, solver=solver, radius=radius
        )
        rows.append(
            BenchmarkRow(
                seed=seed,
                solver=solver,
                clean_train_rmse=_rmse(y_clean_test, clean_prediction),
                corrupted_train_rmse=_rmse(y_clean_test, corrupted_prediction),
            )
        )
    return rows


def run_benchmark(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    radius: float = DEFAULT_RADIUS,
    contamination_fraction: float = DEFAULT_CONTAMINATION_FRACTION,
    n_train: int = 600,
    n_test: int = 400,
) -> tuple[list[BenchmarkRow], list[BenchmarkRow]]:
    """Run scalar and joint clean-holdout comparisons for fixed seeds."""
    scalar_rows: list[BenchmarkRow] = []
    joint_rows: list[BenchmarkRow] = []
    for seed in seeds:
        scalar_rows.extend(
            _run_case(
                make_noisy_regression,
                _fit_scalar,
                seed=seed,
                radius=radius,
                contamination_fraction=contamination_fraction,
                n_train=n_train,
                n_test=n_test,
            )
        )
        joint_rows.extend(
            _run_case(
                make_noisy_joint_regression,
                _fit_joint,
                seed=seed,
                radius=radius,
                contamination_fraction=contamination_fraction,
                n_train=n_train,
                n_test=n_test,
            )
        )
    return scalar_rows, joint_rows


def _median_table(rows: Sequence[BenchmarkRow]) -> list[tuple[str, float, float, float]]:
    return [
        (
            solver,
            float(np.median([row.clean_train_rmse for row in rows if row.solver == solver])),
            float(np.median([row.corrupted_train_rmse for row in rows if row.solver == solver])),
            float(np.median([row.corruption_penalty for row in rows if row.solver == solver])),
        )
        for solver in ("standard", "dro")
    ]


def _render_table(rows: Sequence[BenchmarkRow]) -> list[str]:
    lines = [
        "| Solver | Clean-label-fit RMSE | Contaminated-label-fit RMSE | Corruption penalty |",
        "| --- | ---: | ---: | ---: |",
    ]
    for solver, clean, corrupted, penalty in _median_table(rows):
        lines.append(
            f"| `{solver}` | {clean:.5f} | {corrupted:.5f} | {penalty:+.5f} |"
        )
    return lines


def render_report(
    *,
    scalar_rows: Sequence[BenchmarkRow],
    joint_rows: Sequence[BenchmarkRow],
    radius: float,
    contamination_fraction: float,
) -> str:
    """Render a compact Markdown result with its interpretation contract."""
    lines = [
        "# DRO Robustness Benchmark",
        "",
        "This deterministic synthetic benchmark fits on clean and contaminated training labels,",
        "then measures RMSE against clean held-out targets. The corruption penalty is",
        "`corrupted-train RMSE - clean-train RMSE`; smaller is more robust to the injected outliers.",
        "",
        f"- Seeds: {', '.join(str(row.seed) for row in scalar_rows if row.solver == 'standard')}",
        f"- Contaminated training labels: {contamination_fraction:.0%} per output",
        f"- `dro_radius`: {radius:.3f}",
        "- All models: 100 trees, depth 4, learning rate 0.06, `lambda_l2=1.0`.",
        "",
        "## Scalar Regressor",
        "",
        *_render_table(scalar_rows),
        "",
        "## Joint Shared-Tree Multi-Label Regressor",
        "",
        *_render_table(joint_rows),
        "",
        "The joint path applies DRO only to final leaf values; its shared-histogram split",
        "selection does not retain gradient-square statistics and therefore remains standard.",
        "This benchmark is evidence for deciding whether the extra joint histogram memory required",
        "for robust split selection is justified; it is not a claim that DRO wins every dataset.",
        "",
    ]
    return "\n".join(lines)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument(
        "--contamination-fraction",
        type=float,
        default=DEFAULT_CONTAMINATION_FRACTION,
    )
    parser.add_argument("--quick", action="store_true", help="Use two seeds and smaller datasets.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    if args.radius < 0.0:
        parser.error("--radius must be non-negative")
    if not 0.0 < args.contamination_fraction < 1.0:
        parser.error("--contamination-fraction must be in (0, 1)")
    seeds = args.seeds[:2] if args.quick else args.seeds
    n_train, n_test = (240, 160) if args.quick else (600, 400)
    scalar_rows, joint_rows = run_benchmark(
        seeds=seeds,
        radius=args.radius,
        contamination_fraction=args.contamination_fraction,
        n_train=n_train,
        n_test=n_test,
    )
    report = render_report(
        scalar_rows=scalar_rows,
        joint_rows=joint_rows,
        radius=args.radius,
        contamination_fraction=args.contamination_fraction,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
