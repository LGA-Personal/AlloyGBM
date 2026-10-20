#!/usr/bin/env python3
"""Deterministic large-query ranking and skewed-count GLM benchmark pack.

The ranking fixture compares full LambdaMART pair enumeration with top-10
truncation on held-out large query groups. The GLM fixture evaluates Poisson,
Gamma, and Tweedie models on separate skewed, log-link-compatible targets and
includes the Poisson max-delta-step stabilizer A/B.

Usage:
    .venv/bin/python benchmarks/objective_benchmark.py
    .venv/bin/python benchmarks/objective_benchmark.py --quick --gate
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_SEEDS = (7, 13, 29)
DEFAULT_QUERY_SIZE = 512
DEFAULT_N_ESTIMATORS = 50
TOP_K = 10


@dataclass(frozen=True)
class SkewedGlmData:
    X_train: np.ndarray
    X_test: np.ndarray
    targets_train: dict[str, np.ndarray]
    targets_test: dict[str, np.ndarray]


@dataclass(frozen=True)
class RankingRow:
    seed: int
    arm: str
    ndcg_at_10: float
    fit_seconds: float

    @property
    def metric(self) -> float:
        return self.ndcg_at_10


@dataclass(frozen=True)
class GlmRow:
    seed: int
    objective: str
    deviance: float
    baseline_deviance: float
    fit_seconds: float

    @property
    def metric(self) -> float:
        return self.deviance


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def _ranking_partition(
    rng: np.random.Generator,
    *,
    n_queries: int,
    query_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_queries * query_size, 8)).astype(np.float32)
    latent = (
        1.2 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.5 * X[:, 2] * X[:, 3]
        + 0.3 * np.sin(X[:, 4])
        + rng.normal(scale=0.6, size=len(X))
    )
    labels = np.empty(len(X), dtype=np.float32)
    for query in range(n_queries):
        start = query * query_size
        end = start + query_size
        ranks = np.empty(query_size, dtype=np.intp)
        ranks[np.argsort(latent[start:end], kind="mergesort")] = np.arange(query_size)
        labels[start:end] = np.minimum(4, (ranks * 5) // query_size)
    group = np.repeat(np.arange(n_queries, dtype=np.int32), query_size)
    return X, labels, group


def make_large_query_ranking(
    *,
    seed: int,
    query_size: int = DEFAULT_QUERY_SIZE,
    n_train_queries: int = 4,
    n_test_queries: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic train/test ranking data with contiguous large queries."""
    if query_size < TOP_K:
        raise ValueError(f"query_size must be >= {TOP_K}")
    rng = np.random.default_rng(seed)
    X_train, y_train, group_train = _ranking_partition(
        rng, n_queries=n_train_queries, query_size=query_size
    )
    X_test, y_test, group_test = _ranking_partition(
        rng, n_queries=n_test_queries, query_size=query_size
    )
    return X_train, y_train, group_train, X_test, y_test, group_test


def _log_mean(X: np.ndarray) -> np.ndarray:
    return np.clip(
        0.45 + 0.7 * X[:, 0] - 0.45 * X[:, 1] + 0.3 * X[:, 2] * X[:, 3],
        -2.5,
        3.0,
    )


def _glm_targets(rng: np.random.Generator, X: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.exp(_log_mean(X))
    poisson = rng.poisson(mean).astype(np.float32)
    gamma = rng.gamma(shape=1.25, scale=mean / 1.25).astype(np.float32)
    nonzero = rng.random(len(X)) < (1.0 - np.exp(-mean))
    tweedie = np.where(
        nonzero,
        rng.gamma(shape=1.5, scale=mean / 1.5),
        0.0,
    ).astype(np.float32)
    return {"poisson": poisson, "gamma": gamma, "tweedie": tweedie}


def make_skewed_glm_data(
    *,
    seed: int,
    n_train: int = 1_200,
    n_test: int = 800,
) -> SkewedGlmData:
    """Build held-out skewed targets in each GLM objective's valid domain."""
    rng = np.random.default_rng(seed + 100_000)
    X_train = rng.normal(size=(n_train, 6)).astype(np.float32)
    X_test = rng.normal(size=(n_test, 6)).astype(np.float32)
    return SkewedGlmData(
        X_train=X_train,
        X_test=X_test,
        targets_train=_glm_targets(rng, X_train),
        targets_test=_glm_targets(rng, X_test),
    )


def _fit_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    truncation_level: int | None,
) -> tuple[np.ndarray, float]:
    from alloygbm import GBMRanker

    model = GBMRanker(
        ranking_objective="rank:ndcg",
        lambdarank_truncation_level=truncation_level,
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.06,
        lambda_l2=1.0,
        training_policy="manual",
        deterministic=True,
        seed=seed,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, group=group_train)
    return np.asarray(model.predict(X_test)), time.perf_counter() - start


def _fit_glm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    objective: str,
    seed: int,
    n_estimators: int,
    poisson_max_delta_step: float | None = None,
) -> tuple[np.ndarray, float]:
    from alloygbm import GBMRegressor

    params: dict[str, object] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.06,
        "lambda_l2": 1.0,
        "training_policy": "manual",
        "deterministic": True,
        "seed": seed,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5
    if poisson_max_delta_step is not None:
        params["poisson_max_delta_step"] = poisson_max_delta_step
    model = GBMRegressor(**params)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test)), time.perf_counter() - start


def _glm_deviance(objective: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    from alloygbm.evaluation import gamma_deviance, poisson_deviance, tweedie_deviance

    if objective == "poisson":
        return poisson_deviance(y_true, prediction)
    if objective == "gamma":
        return gamma_deviance(y_true, prediction)
    if objective == "tweedie":
        return tweedie_deviance(y_true, prediction, variance_power=1.5)
    raise ValueError(f"unsupported GLM objective: {objective}")


def run_benchmark(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    query_size: int = DEFAULT_QUERY_SIZE,
    n_train: int = 1_200,
    n_test: int = 800,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> tuple[list[RankingRow], list[GlmRow]]:
    """Run deterministic held-out ranking and GLM benchmark arms."""
    from alloygbm.evaluation import ndcg

    ranking_rows: list[RankingRow] = []
    glm_rows: list[GlmRow] = []
    for seed in seeds:
        X_train, y_train, group_train, X_test, y_test, group_test = make_large_query_ranking(
            seed=seed, query_size=query_size
        )
        for arm, truncation_level in (("full", None), (f"top_{TOP_K}", TOP_K)):
            prediction, elapsed = _fit_ranking(
                X_train,
                y_train,
                group_train,
                X_test,
                seed=seed,
                n_estimators=n_estimators,
                truncation_level=truncation_level,
            )
            ranking_rows.append(
                RankingRow(
                    seed=seed,
                    arm=arm,
                    ndcg_at_10=ndcg(y_test, prediction, group=group_test, k=TOP_K),
                    fit_seconds=elapsed,
                )
            )

        glm_data = make_skewed_glm_data(seed=seed, n_train=n_train, n_test=n_test)
        for name, objective, max_delta_step in (
            ("poisson_default", "poisson", None),
            ("poisson_no_stabilizer", "poisson", 0.0),
            ("gamma", "gamma", None),
            ("tweedie", "tweedie", None),
        ):
            y_train_glm = glm_data.targets_train[objective]
            y_test_glm = glm_data.targets_test[objective]
            prediction, elapsed = _fit_glm(
                glm_data.X_train,
                y_train_glm,
                glm_data.X_test,
                objective=objective,
                seed=seed,
                n_estimators=n_estimators,
                poisson_max_delta_step=max_delta_step,
            )
            baseline = np.full(
                len(y_test_glm),
                max(float(np.mean(y_train_glm)), 1e-6),
                dtype=np.float32,
            )
            glm_rows.append(
                GlmRow(
                    seed=seed,
                    objective=name,
                    deviance=_glm_deviance(objective, y_test_glm, prediction),
                    baseline_deviance=_glm_deviance(objective, y_test_glm, baseline),
                    fit_seconds=elapsed,
                )
            )
    return ranking_rows, glm_rows


def _median_ranking(rows: Sequence[RankingRow], arm: str) -> tuple[float, float]:
    selected = [row for row in rows if row.arm == arm]
    return (
        float(np.median([row.ndcg_at_10 for row in selected])),
        float(np.median([row.fit_seconds for row in selected])),
    )


def _median_glm(rows: Sequence[GlmRow], objective: str) -> tuple[float, float, float]:
    selected = [row for row in rows if row.objective == objective]
    return (
        float(np.median([row.deviance for row in selected])),
        float(np.median([row.baseline_deviance for row in selected])),
        float(np.median([row.fit_seconds for row in selected])),
    )


def evaluate_gates(
    ranking_rows: Sequence[RankingRow], glm_rows: Sequence[GlmRow]
) -> list[GateResult]:
    """Check held-out quality and finite metrics without asserting timing."""
    full_ndcg, _ = _median_ranking(ranking_rows, "full")
    top_ndcg, _ = _median_ranking(ranking_rows, f"top_{TOP_K}")
    gates = [
        GateResult(
            "LambdaMART top-10 truncation",
            bool(np.isfinite(full_ndcg) and np.isfinite(top_ndcg) and top_ndcg >= full_ndcg - 0.10),
            f"full={full_ndcg:.5f}, top_{TOP_K}={top_ndcg:.5f}, limit=-0.10000",
        )
    ]
    for name in ("poisson_default", "poisson_no_stabilizer", "gamma", "tweedie"):
        deviance, baseline, _ = _median_glm(glm_rows, name)
        gates.append(
            GateResult(
                name,
                bool(np.isfinite(deviance) and np.isfinite(baseline) and deviance < baseline),
                f"deviance={deviance:.5f}, baseline={baseline:.5f}",
            )
        )
    return gates


def render_report(
    *,
    ranking_rows: Sequence[RankingRow],
    glm_rows: Sequence[GlmRow],
    query_size: int,
    n_estimators: int,
) -> str:
    """Render the benchmark result with its non-performance-claim contract."""
    full_ndcg, full_time = _median_ranking(ranking_rows, "full")
    top_ndcg, top_time = _median_ranking(ranking_rows, f"top_{TOP_K}")
    lines = [
        "# Objective Benchmark Pack",
        "",
        "This deterministic, offline benchmark validates the reviewed ranking and GLM paths on",
        "held-out synthetic data. It is a regression and calibration check, not a cross-library claim.",
        "",
        f"- Seeds: {', '.join(str(row.seed) for row in ranking_rows if row.arm == 'full')}",
        f"- Ranking: 4 training and 2 held-out large query groups of {query_size} rows.",
        f"- All models: {n_estimators} trees, depth 4, learning rate 0.06, `lambda_l2=1.0`.",
        "",
        "## LambdaMART Large-Query A/B",
        "",
        "`full` evaluates all pairwise candidates. `top_10` uses the public",
        "`lambdarank_truncation_level=10` path; both are measured against held-out NDCG@10.",
        "Timing is descriptive because host load and Rayon scheduling make it unsuitable as a hard gate.",
        "",
        "| Arm | Held-out NDCG@10 | Median fit (s) |",
        "| --- | ---: | ---: |",
        f"| `full` | {full_ndcg:.5f} | {full_time:.3f} |",
        f"| `top_{TOP_K}` | {top_ndcg:.5f} | {top_time:.3f} |",
        "",
        "## Skewed-Count GLM Validation",
        "",
        "Each objective uses its valid target domain and reports held-out mean deviance against",
        "a train-mean baseline. Lower is better. The two Poisson rows isolate the default",
        "Poisson stabilizer (`poisson_max_delta_step=0.7`) from the legacy zero-step setting.",
        "",
        "| Objective | Held-out deviance | Train-mean baseline | Median fit (s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for objective in ("poisson_default", "poisson_no_stabilizer", "gamma", "tweedie"):
        deviance, baseline, elapsed = _median_glm(glm_rows, objective)
        lines.append(f"| `{objective}` | {deviance:.5f} | {baseline:.5f} | {elapsed:.3f} |")
    lines.extend([
        "",
        "The Poisson stabilizer row is evidence that the stabilized implementation remains finite",
        "and learns the held-out skewed-count fixture. It does not assert that a nonzero",
        "`max_delta_step` dominates every distribution or hyperparameter setting.",
        "",
    ])
    return "\n".join(lines)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--query-size", type=int, default=DEFAULT_QUERY_SIZE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--quick", action="store_true", help="Use one seed and smaller fixtures.")
    parser.add_argument("--gate", action="store_true", help="Fail on held-out quality regressions.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    if args.query_size < TOP_K:
        parser.error(f"--query-size must be >= {TOP_K}")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be >= 1")
    seeds = args.seeds[:1] if args.quick else args.seeds
    query_size = 128 if args.quick else args.query_size
    n_train, n_test = (240, 160) if args.quick else (1_200, 800)
    n_estimators = 15 if args.quick else args.n_estimators
    ranking_rows, glm_rows = run_benchmark(
        seeds=seeds,
        query_size=query_size,
        n_train=n_train,
        n_test=n_test,
        n_estimators=n_estimators,
    )
    report = render_report(
        ranking_rows=ranking_rows,
        glm_rows=glm_rows,
        query_size=query_size,
        n_estimators=n_estimators,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    if args.gate:
        gates = evaluate_gates(ranking_rows, glm_rows)
        print("## Gates")
        for gate in gates:
            print(f"- {'pass' if gate.passed else 'FAIL'}: {gate.name} ({gate.detail})")
        if not all(gate.passed for gate in gates):
            raise SystemExit("objective benchmark gate failed")


if __name__ == "__main__":
    main()
