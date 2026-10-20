#!/usr/bin/env python3
"""Fixed-seed performance-regression benchmark for AlloyGBM.

Guards against *performance* regressions (the other benchmark gates guard
accuracy/quality). For each scenario the harness trains + predicts at two
dataset sizes N and 2N and enforces:

  * Quality (hard, deterministic, platform-independent): the trained model must
    beat a naive constant predictor by a healthy margin.
  * Timing (loose, machine-independent): the fit-time scaling ratio
    fit(2N)/fit(N) must stay below SCALING_CEILING. A healthy histogram GBDT
    scales ~linearly (~2x); an algorithmic regression to O(N^2) shows ~4x.
    Absolute wall-clock times are reported as descriptive deltas but NOT gated
    (they are not comparable across CI runners).

Usage:
  python benchmarks/perf_regression_benchmark.py            # full report
  python benchmarks/perf_regression_benchmark.py --quick    # compact CI matrix
  python benchmarks/perf_regression_benchmark.py --quick --gate  # nonzero on failure
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor

SCALING_CEILING = 3.0
REGRESSION_QUALITY_RATIO = 0.6
CLASSIFICATION_QUALITY_MARGIN = 0.15
TIMING_REPEATS = 3


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str  # "regression" | "binary"
    n_base: int
    n_features: int
    n_estimators: int
    max_depth: int
    seed: int
    leaf_model: str | None = None


def quick_scenarios() -> list[Scenario]:
    return [
        Scenario("reg-medium", "regression", 3000, 12, 40, 5, 1),
        Scenario("cls-medium", "binary", 3000, 12, 40, 5, 2),
        # A wide, 50k-row scenario so the fit(2N)/fit(N) scaling check
        # exercises the regime where histogram construction (rather than
        # per-round fixed costs) dominates. Small datasets can hide an
        # algorithmic regression behind constant overheads.
        Scenario("reg-large-wide", "regression", 50_000, 40, 25, 6, 4),
    ]


def full_scenarios() -> list[Scenario]:
    return [
        Scenario("reg-medium", "regression", 12000, 16, 120, 6, 1),
        Scenario("reg-linear", "regression", 12000, 16, 120, 6, 3, leaf_model="linear"),
        Scenario("cls-medium", "binary", 12000, 16, 120, 6, 2),
        Scenario("reg-large-wide", "regression", 125_000, 50, 40, 6, 4),
    ]


def _make_data(scenario: Scenario, n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, scenario.n_features)).astype(np.float64)
    signal = 1.3 * X[:, 0] - X[:, 2] ** 2 + 0.7 * X[:, 1] * X[:, 3]
    if scenario.task == "regression":
        y = (signal + rng.normal(scale=0.3, size=n_rows)).astype(np.float64)
    else:
        prob = 1.0 / (1.0 + np.exp(-signal))
        y = (rng.random(n_rows) < prob).astype(int)
    return X, y


def _estimator(scenario: Scenario):
    common = dict(
        n_estimators=scenario.n_estimators,
        max_depth=scenario.max_depth,
        seed=scenario.seed,
        n_jobs=1,
    )
    if scenario.task == "regression":
        if scenario.leaf_model is not None:
            return GBMRegressor(leaf_model=scenario.leaf_model, **common)
        return GBMRegressor(**common)
    return GBMClassifier(**common)


def _best_fit_seconds(scenario: Scenario, X, y):
    best = float("inf")
    model = None
    for _ in range(TIMING_REPEATS):
        candidate = _estimator(scenario)
        start = time.perf_counter()
        candidate.fit(X, y)
        best = min(best, time.perf_counter() - start)
        model = candidate
    return best, model


def _quality(scenario: Scenario, model, X, y):
    predictions = np.asarray(model.predict(X))
    finite = bool(np.all(np.isfinite(predictions.astype(np.float64))))
    if scenario.task == "regression":
        rmse = float(np.sqrt(np.mean((predictions.astype(np.float64) - y) ** 2)))
        constant_rmse = float(np.sqrt(np.mean((y - float(np.mean(y))) ** 2)))
        beats = rmse <= REGRESSION_QUALITY_RATIO * constant_rmse
        return "rmse", rmse, constant_rmse, beats, finite
    accuracy = float(np.mean(predictions == y))
    beats = accuracy >= 0.5 + CLASSIFICATION_QUALITY_MARGIN
    return "accuracy", accuracy, 0.5, beats, finite


def _best_predict_seconds(model, X):
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        model.predict(X)
        best = min(best, time.perf_counter() - start)
    return best


def run_scenario(scenario: Scenario) -> dict:
    n1, n2 = scenario.n_base, 2 * scenario.n_base
    x1, y1 = _make_data(scenario, n1, scenario.seed)
    x2, y2 = _make_data(scenario, n2, scenario.seed + 101)
    fit1, _ = _best_fit_seconds(scenario, x1, y1)
    fit2, model2 = _best_fit_seconds(scenario, x2, y2)
    ratio = fit2 / max(fit1, 1e-9)
    metric, value, baseline, beats, finite = _quality(scenario, model2, x2, y2)
    predict_seconds = _best_predict_seconds(model2, x2)
    return {
        "scenario": scenario.name,
        "task": scenario.task,
        "n_base": scenario.n_base,
        "fit_seconds_n": fit1,
        "fit_seconds_2n": fit2,
        "fit_scaling_ratio": ratio,
        "predict_seconds_2n": predict_seconds,
        "metric": metric,
        "metric_value": value,
        "metric_baseline": baseline,
        "quality_beats_baseline": beats,
        "predictions_finite": finite,
    }


def run_benchmark(quick: bool) -> list[dict]:
    scenarios = quick_scenarios() if quick else full_scenarios()
    return [run_scenario(scenario) for scenario in scenarios]


def evaluate_gate(report: list[dict]) -> list[str]:
    failures: list[str] = []
    for record in report:
        name = record["scenario"]
        if not record["predictions_finite"]:
            failures.append(f"{name}: non-finite predictions")
        if not record["quality_beats_baseline"]:
            failures.append(
                f"{name}: {record['metric']} {record['metric_value']:.4f} "
                f"fails to beat baseline {record['metric_baseline']:.4f}"
            )
        if record["fit_scaling_ratio"] > SCALING_CEILING:
            failures.append(
                f"{name}: fit scaling ratio {record['fit_scaling_ratio']:.2f} "
                f"exceeds ceiling {SCALING_CEILING:.2f}"
            )
    return failures


def render_markdown(report: list[dict]) -> str:
    lines = [
        "# AlloyGBM Performance-Regression Benchmark",
        "",
        f"- Timing gate: fit(2N)/fit(N) scaling ratio <= `{SCALING_CEILING:.2f}` "
        "(machine-independent; absolute times below are descriptive only).",
        f"- Quality gate: regression RMSE <= `{REGRESSION_QUALITY_RATIO:.2f}` x constant-predictor RMSE; "
        f"binary accuracy >= `{0.5 + CLASSIFICATION_QUALITY_MARGIN:.2f}`.",
        "",
        "| Scenario | Task | N | fit(N) s | fit(2N) s | scaling | predict(2N) s | Metric | Value | Baseline | Beats | Finite |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|:---:|:---:|",
    ]
    for r in report:
        lines.append(
            f"| {r['scenario']} | {r['task']} | {r['n_base']} | "
            f"{r['fit_seconds_n']:.4f} | {r['fit_seconds_2n']:.4f} | "
            f"{r['fit_scaling_ratio']:.3f} | {r['predict_seconds_2n']:.4f} | "
            f"{r['metric']} | {r['metric_value']:.4f} | {r['metric_baseline']:.4f} | "
            f"{'yes' if r['quality_beats_baseline'] else 'NO'} | "
            f"{'yes' if r['predictions_finite'] else 'NO'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run the compact CI matrix")
    parser.add_argument("--gate", action="store_true", help="return nonzero on acceptance failures")
    parser.add_argument("--output", type=str, help="write the Markdown report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_benchmark(quick=args.quick)
    rendered = render_markdown(report)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    failures = evaluate_gate(report)
    if args.gate and failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
