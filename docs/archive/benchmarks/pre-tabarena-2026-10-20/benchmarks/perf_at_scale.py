#!/usr/bin/env python3
"""Performance regression harness for AlloyGBM at scale.

Trains AlloyGBM on synthetic regression data at three scales and reports
fit_seconds plus the internal fit_timing_ breakdown. Designed to be run
before and after a perf change to quantify wall-time impact.

Usage:
    .venv/bin/python benchmarks/perf_at_scale.py
    .venv/bin/python benchmarks/perf_at_scale.py --scale large
"""
from __future__ import annotations
import argparse
import gc
import time
import numpy as np

SCALES = {
    "small":  {"n_rows":  50_000, "n_features": 100, "n_estimators": 200},
    "medium": {"n_rows": 200_000, "n_features": 400, "n_estimators": 500},
    "large":  {"n_rows": 500_000, "n_features": 780, "n_estimators": 1200},
}

def make_dataset(n_rows, n_features, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features), dtype=np.float32)
    coefs = rng.standard_normal(n_features).astype(np.float32) * 0.1
    y = X @ coefs + 0.1 * rng.standard_normal(n_rows).astype(np.float32)
    return X, y

def time_fit(arm: str, X, y, n_estimators):
    from alloygbm import GBMRegressor
    kwargs = {"n_estimators": n_estimators, "max_depth": 6, "learning_rate": 0.05,
              "row_subsample": 0.8, "col_subsample": 0.3, "min_data_in_leaf": 5000,
              "lambda_l2": 1.0, "min_child_hessian": 5000.0, "seed": 42, "deterministic": True}
    if arm == "morph":
        kwargs["training_mode"] = "morph"
    elif arm == "morph_cosine":
        kwargs["training_mode"] = "morph"
        kwargs["lr_schedule"] = "warmup_cosine"
        kwargs["lr_warmup_frac"] = 0.1
    m = GBMRegressor(**kwargs)
    t0 = time.perf_counter()
    m.fit(X, y)
    elapsed = time.perf_counter() - t0
    timing = getattr(m, "fit_timing_", {}) or {}
    return elapsed, timing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALES), default="medium")
    parser.add_argument("--arms", nargs="+", default=["auto", "morph", "morph_cosine"])
    args = parser.parse_args()
    cfg = SCALES[args.scale]
    print(f"Scale: {args.scale} ({cfg})")
    X, y = make_dataset(cfg["n_rows"], cfg["n_features"])
    print(f"Dataset: {X.shape} float32 ({X.nbytes/1e6:.0f} MB)\n")
    for arm in args.arms:
        gc.collect()
        elapsed, timing = time_fit(arm, X, y, cfg["n_estimators"])
        native = timing.get("native_train_seconds", float("nan"))
        adapt = timing.get("input_adaptation_seconds", float("nan"))
        bridge = timing.get("native_bridge_prepare_seconds", float("nan"))
        print(f"  {arm:>14}: fit={elapsed:7.2f}s  native={native:7.2f}s  adapt={adapt:6.3f}s  bridge={bridge:6.3f}s")

if __name__ == "__main__":
    main()
