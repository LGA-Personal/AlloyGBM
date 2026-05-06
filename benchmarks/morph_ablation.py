#!/usr/bin/env python3
"""Morph ablation harness.

Toggles each morph component independently on 3 representative datasets and
prints a markdown summary table.

Usage::
    python benchmarks/morph_ablation.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- Dataset generators ---

def _regression_dataset(n=2000, n_features=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    coefs = rng.standard_normal(n_features).astype(np.float32)
    y = X @ coefs + 0.2 * rng.standard_normal(n).astype(np.float32)
    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


def _binary_dataset(n=2000, n_features=20, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal(n_features).astype(np.float32)
    y = (logits > 0).astype(np.int32)
    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


def _ranking_dataset(n=2000, n_features=20, n_groups=50, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.float32)
    sizes = [n // n_groups] * n_groups
    sizes[-1] += n - sum(sizes)
    group = np.repeat(np.arange(n_groups), sizes).astype(np.int32)
    order = np.argsort(group)
    X, y, group = X[order], y[order], group[order]
    split_g = n_groups * 4 // 5
    split_n = sum(sizes[:split_g])
    return X[:split_n], y[:split_n], group[:split_g], X[split_n:], y[split_n:], group[split_g:]


# --- Ablation configs ---

ABLATION_CONFIGS = {
    "baseline_auto": {},  # no morph
    "morph_full": {"training_mode": "morph"},
    "morph_no_balance": {"training_mode": "morph", "balance_penalty": False},  # note: not exposed yet — skip
    "morph_cosine": {"training_mode": "morph", "lr_schedule": "warmup_cosine", "lr_warmup_frac": 0.1},
    "morph_no_warmup": {"training_mode": "morph", "morph_warmup_iters": 0},
}

# balance_penalty is not yet a top-level Python param (it's in MorphConfig internals) — skip that variant


# --- Metrics ---

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# --- Run ---

@dataclass
class Result:
    config: str
    dataset: str
    metric: float
    metric_name: str
    train_sec: float


def run_ablation(quick: bool = False) -> list[Result]:
    from alloygbm import GBMClassifier, GBMRanker, GBMRegressor

    n = 500 if quick else 2000
    n_est = 30 if quick else 100
    results = []

    # Regression
    X_tr, y_tr, X_te, y_te = _regression_dataset(n=n)
    for name, kwargs in ABLATION_CONFIGS.items():
        if "balance_penalty" in kwargs:
            continue
        tm = kwargs.get("training_mode", "auto")
        extra = {k: v for k, v in kwargs.items() if k != "training_mode"}
        try:
            m = GBMRegressor(n_estimators=n_est, max_depth=5, training_mode=tm,
                             seed=0, **extra)
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            rmse = _rmse(y_te, np.asarray(m.predict(X_te)))
            results.append(Result(name, "regression", rmse, "RMSE", elapsed))
        except Exception as exc:
            results.append(Result(name, "regression", float("nan"), f"ERROR: {exc}", 0.0))

    # Binary classification
    X_tr, y_tr, X_te, y_te = _binary_dataset(n=n)
    for name, kwargs in ABLATION_CONFIGS.items():
        if "balance_penalty" in kwargs:
            continue
        tm = kwargs.get("training_mode", "auto")
        extra = {k: v for k, v in kwargs.items() if k != "training_mode"}
        try:
            m = GBMClassifier(n_estimators=n_est, max_depth=5, training_mode=tm,
                              seed=0, **extra)
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            acc = _accuracy(y_te, np.asarray(m.predict(X_te)))
            results.append(Result(name, "binary_classification", acc, "Accuracy", elapsed))
        except Exception as exc:
            results.append(Result(name, "binary_classification", float("nan"), f"ERROR: {exc}", 0.0))

    # Ranking (simplified — skip in quick mode)
    if not quick:
        X_tr, y_tr, g_tr, X_te, y_te, g_te = _ranking_dataset(n=n)
        for name, kwargs in ABLATION_CONFIGS.items():
            if "balance_penalty" in kwargs:
                continue
            tm = kwargs.get("training_mode", "auto")
            extra = {k: v for k, v in kwargs.items() if k != "training_mode"}
            try:
                m = GBMRanker(n_estimators=n_est, max_depth=5, training_mode=tm,
                               seed=0, **extra)
                t0 = time.perf_counter()
                m.fit(X_tr, y_tr, group=g_tr)
                elapsed = time.perf_counter() - t0
                # Use RMSE as proxy for ranking score (NDCG would need group info)
                rmse = _rmse(y_te, np.asarray(m.predict(X_te)))
                results.append(Result(name, "ranking", rmse, "RMSE(proxy)", elapsed))
            except Exception as exc:
                results.append(Result(name, "ranking", float("nan"), f"ERROR: {exc}", 0.0))

    return results


def print_markdown_table(results: list[Result]) -> None:
    datasets = sorted({r.dataset for r in results})
    configs = list(ABLATION_CONFIGS.keys())

    for ds in datasets:
        ds_results = {r.config: r for r in results if r.dataset == ds}
        if not ds_results:
            continue
        metric_name = next(iter(ds_results.values())).metric_name
        print(f"\n### {ds} ({metric_name})\n")
        print(f"| Config | {metric_name} | Train (s) |")
        print("|---|---|---|")
        for cfg in configs:
            if cfg in ds_results:
                r = ds_results[cfg]
                print(f"| {cfg} | {r.metric:.4f} | {r.train_sec:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller datasets")
    args = parser.parse_args()
    results = run_ablation(quick=args.quick)
    print_markdown_table(results)


if __name__ == "__main__":
    main()
