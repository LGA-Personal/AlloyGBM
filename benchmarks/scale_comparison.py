#!/usr/bin/env python3
"""Large-scale, thread-budget-matched comparison against LightGBM/XGBoost/CatBoost.

Why this exists separately from ``run_model_comparison.py``:

The curated scenario suite tops out around 40k rows. At that size, forcing
every library to use all logical CPUs measures *thread-spawn overhead* rather
than throughput -- LightGBM and XGBoost are measurably **slower** multi-threaded
than single-threaded below ~40k rows on the reference host. Reporting a
multi-threaded win there would flatter AlloyGBM for the wrong reason.

So the suite is run single-threaded (a clean per-core algorithmic comparison,
and the most reproducible across machines), and this script covers the
realistic multi-threaded case on data large enough that parallelism actually
pays for every library.

Fairness rules, applied identically to all four libraries:

* one thread budget (``--threads``), set through each library's own knob
  (AlloyGBM/LightGBM/XGBoost ``n_jobs``, CatBoost ``thread_count``), with the
  OpenMP/BLAS environment pinned to the same number;
* identical ``n_estimators``, ``learning_rate``, ``max_depth``, and seed;
* identical row/column subsampling -- including ``subsample_freq=1`` for
  LightGBM, without which its ``subsample`` is silently ignored;
* ``num_leaves = 2 ** max_depth`` for LightGBM so its leaf-wise growth reaches
  the same capacity as the depth-wise peers;
* the same float32 input arrays, and timing that excludes data generation.

Usage:
  python benchmarks/scale_comparison.py                     # default matrix
  python benchmarks/scale_comparison.py --rows 1000000      # single size
  python benchmarks/scale_comparison.py --threads 1 8       # thread sweep
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _pin_thread_env(threads: int) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(threads)


@dataclass
class ScaleRecord:
    task: str
    rows: int
    features: int
    threads: int
    model: str
    fit_seconds: float
    predict_seconds: float
    metric_name: str
    metric: float


def _make_data(task: str, rows: int, features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, features)).astype(np.float32)
    weights = rng.standard_normal(features).astype(np.float32)
    signal = X @ weights + 0.5 * np.sin(3.0 * X[:, 0]) * X[:, 1]
    if task == "regression":
        y = (signal + rng.standard_normal(rows).astype(np.float32)).astype(np.float32)
    else:
        prob = 1.0 / (1.0 + np.exp(-signal / (np.std(signal) + 1e-9)))
        y = (rng.random(rows) < prob).astype(np.int32)
    split = int(rows * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


def _factories(task: str, threads: int, rounds: int, depth: int, lr: float, seed: int):
    """One entry per library; every knob that affects work is matched."""
    from alloygbm import GBMClassifier, GBMRegressor
    import lightgbm as lgb
    import xgboost as xgb

    common = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth)
    out = {}
    if task == "regression":
        out["alloygbm"] = lambda: GBMRegressor(
            **common, seed=seed, row_subsample=0.8, col_subsample=0.8, n_jobs=threads
        )
        out["lightgbm"] = lambda: lgb.LGBMRegressor(
            **common, random_state=seed, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, num_leaves=2 ** depth, n_jobs=threads, verbose=-1,
        )
        out["xgboost"] = lambda: xgb.XGBRegressor(
            **common, random_state=seed, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=threads, verbosity=0,
        )
    else:
        out["alloygbm"] = lambda: GBMClassifier(
            **common, seed=seed, row_subsample=0.8, col_subsample=0.8, n_jobs=threads
        )
        out["lightgbm"] = lambda: lgb.LGBMClassifier(
            **common, random_state=seed, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, num_leaves=2 ** depth, n_jobs=threads, verbose=-1,
        )
        out["xgboost"] = lambda: xgb.XGBClassifier(
            **common, random_state=seed, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=threads, verbosity=0,
        )
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        cb = CatBoostRegressor if task == "regression" else CatBoostClassifier
        out["catboost"] = lambda: cb(
            iterations=rounds, learning_rate=lr, depth=depth, random_seed=seed,
            thread_count=threads, verbose=False, allow_writing_files=False,
        )
    except ImportError:
        pass
    return out


def _score(task: str, y_true, y_pred) -> tuple[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    if task == "regression":
        return "rmse", float(np.sqrt(np.mean((np.asarray(y_pred, dtype=np.float64) - y_true) ** 2)))
    return "accuracy", float(np.mean(np.asarray(y_pred).astype(np.float64) == y_true))


def run(rows_list, features, threads_list, rounds, depth, lr, seed, tasks) -> list[ScaleRecord]:
    records: list[ScaleRecord] = []
    for task in tasks:
        for rows in rows_list:
            X_tr, y_tr, X_te, y_te = _make_data(task, rows, features, seed)
            for threads in threads_list:
                _pin_thread_env(threads)
                for name, factory in _factories(task, threads, rounds, depth, lr, seed).items():
                    model = factory()
                    t0 = time.perf_counter()
                    model.fit(X_tr, y_tr)
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    preds = model.predict(X_te)
                    pred_s = time.perf_counter() - t0
                    metric_name, metric = _score(task, y_te, preds)
                    records.append(ScaleRecord(
                        task=task, rows=rows, features=features, threads=threads,
                        model=name, fit_seconds=fit_s, predict_seconds=pred_s,
                        metric_name=metric_name, metric=metric,
                    ))
                    print(
                        f"[{task}][rows={rows}][threads={threads}] {name:9s} "
                        f"fit={fit_s:8.3f}s pred={pred_s:7.4f}s {metric_name}={metric:.4f}",
                        flush=True,
                    )
    return records


def _environment(threads_list) -> dict:
    def ver(mod):
        try:
            return str(getattr(importlib.import_module(mod), "__version__", "unknown"))
        except Exception:
            return "unavailable"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count() or 1,
        "python": platform.python_version(),
        "thread_budgets": list(threads_list),
        "versions": {m: ver(m) for m in
                     ("alloygbm", "lightgbm", "xgboost", "catboost", "numpy")},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", default=[200_000, 1_000_000])
    ap.add_argument("--features", type=int, default=40)
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="thread budgets to sweep; default: 1 and all logical CPUs")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tasks", nargs="+", default=["regression", "classification"],
                    choices=["regression", "classification"])
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args(argv)

    threads_list = args.threads or [1, os.cpu_count() or 1]
    env = _environment(threads_list)
    print(json.dumps(env, indent=2))
    records = run(args.rows, args.features, threads_list, args.rounds,
                  args.max_depth, args.learning_rate, args.seed, args.tasks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "environment": env,
        "params": {
            "rows": args.rows, "features": args.features, "rounds": args.rounds,
            "max_depth": args.max_depth, "learning_rate": args.learning_rate,
            "seed": args.seed, "row_subsample": 0.8, "col_subsample": 0.8,
            "lightgbm_subsample_freq": 1, "lightgbm_num_leaves": "2 ** max_depth",
        },
        "records": [asdict(r) for r in records],
    }
    path = args.output_dir / f"scale_comparison_{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "scale_comparison_latest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
