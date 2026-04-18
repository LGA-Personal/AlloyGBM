#!/usr/bin/env python3
"""Diagnostic for GBMRanker zero-tree / bit-identical-NDCG bug.

Builds synthetic ranking data matching benchmarks/synthetic_ranking/prepare.py
(200 queries × 25 docs = 5000 rows, 16 features) and fits two GBMRanker models:

  1. Auto policy (as used by the benchmark).
  2. Manual override with min_split_gain=0.0 explicitly set.

The auto-policy path is expected to fail with NoSplitImprovement on round 0,
leaving n_estimators_=0 and all predictions at the objective's initial value
of 0.0. The manual override should complete all rounds and produce non-zero,
varying predictions.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from alloygbm import GBMRanker


QUERIES = 200
DOCS_PER_QUERY = 25
FEATURES = 16
SEED = 7


def _generate_feature(rng: random.Random, feature_index: int) -> float:
    if feature_index == 0:
        return round(rng.random() * 8.0) / 8.0
    if feature_index == 1:
        return math.exp(rng.gauss(0.0, 0.5))
    if feature_index % 4 == 0:
        return round(rng.uniform(-2.0, 2.0), 2)
    return rng.uniform(-1.0, 1.0)


def _generate_relevance(features: list[float], rng: random.Random) -> int:
    weighted = 0.0
    for index, value in enumerate(features[:6]):
        weighted += value * (0.5 - (index * 0.06))
    nonlinear = math.sin(features[0] * 2.0) + 0.5 * math.log1p(abs(features[1]))
    noise = rng.gauss(0.0, 0.4)
    score = weighted + nonlinear + noise
    if score < -0.8:
        return 0
    if score < -0.2:
        return 1
    if score < 0.4:
        return 2
    if score < 1.0:
        return 3
    return 4


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for query_index in range(QUERIES):
        for _ in range(DOCS_PER_QUERY):
            feats = [_generate_feature(rng, i) for i in range(FEATURES)]
            rows.append(feats)
            labels.append(_generate_relevance(feats, rng))
            groups.append(query_index)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(groups, dtype=np.int32)


def report(label: str, model: GBMRanker, fit_secs: float, preds: np.ndarray) -> None:
    print(f"=== {label} ===")
    print(f"  stop_reason_       = {getattr(model, 'stop_reason_', '<missing>')}")
    print(f"  rounds_completed_  = {getattr(model, 'rounds_completed_', '<missing>')}")
    print(f"  n_estimators_      = {model.n_estimators_}")
    print(f"  fit_seconds        = {fit_secs:.4f}")
    print(f"  preds[:10]         = {np.round(preds[:10], 6).tolist()}")
    print(f"  preds.std          = {float(preds.std()):.6f}")
    print(f"  preds.unique_count = {int(np.unique(preds).size)}")
    print()


def main() -> int:
    X, y, group = build_dataset()
    print(f"dataset: rows={X.shape[0]}, features={X.shape[1]}, queries={QUERIES}\n")

    # Replicate the benchmark exactly: default rank:ndcg + 0.8 subsamples,
    # mid_balanced profile (lr=0.05, depth=6, n_estimators=1200).
    bench = GBMRanker(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        row_subsample=0.8,
        col_subsample=0.8,
        seed=SEED,
    )
    t0 = time.perf_counter()
    bench.fit(X, y, group=group)
    bench_secs = time.perf_counter() - t0
    report("BENCHMARK (rank:ndcg, auto, 1200)", bench, bench_secs, np.asarray(bench.predict(X)))

    # Same as above but with rank:pairwise to isolate objective.
    pairwise = GBMRanker(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        row_subsample=0.8,
        col_subsample=0.8,
        ranking_objective="rank:pairwise",
        seed=SEED,
    )
    t0 = time.perf_counter()
    pairwise.fit(X, y, group=group)
    pairwise_secs = time.perf_counter() - t0
    report("rank:pairwise (auto, 1200)", pairwise, pairwise_secs, np.asarray(pairwise.predict(X)))

    # Manual override — bypass the density floor by explicitly setting min_split_gain.
    manual = GBMRanker(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        row_subsample=0.8,
        col_subsample=0.8,
        seed=SEED,
        training_policy="manual",
        min_split_gain=0.0,
    )
    t0 = time.perf_counter()
    manual.fit(X, y, group=group)
    manual_secs = time.perf_counter() - t0
    report("MANUAL min_split_gain=0.0", manual, manual_secs, np.asarray(manual.predict(X)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
