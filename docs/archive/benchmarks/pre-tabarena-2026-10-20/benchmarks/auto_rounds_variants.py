"""Auto-rounds design sweep on the default-quality suite.

Each arm emulates one candidate design for n_estimators="auto" with
explicit AlloyGBM parameters (holdout fraction, learning rate, round cap,
patience, relative tolerance, refit or keep the holdout model), so it runs on
any build. It produced the variant table in docs/benchmarks/auto_rounds_v1.md.

Usage::

    python benchmarks/auto_rounds_variants.py default,es_refit_lr0.05 5 out.json [--quick]

The output uses the default_quality result schema, so default_quality.py
compare and render_summary work on it.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import default_quality as dq  # noqa: E402


def catboost_lr(n, task, iters=1000):
    """CatBoost's automatic learning rate (CPU, no best-model selection).

    Coefficients from catboost/libs/train_lib/options_helper.cpp.
    """
    coeffs = {
        dq.REGRESSION: (0.158, -4.287, -0.813, 2.571),
        dq.BINARY: (0.408, -7.299, -0.928, 2.701),
        dq.MULTICLASS: (0.051, -2.889, -0.845, 2.928),
    }[task]
    a, b, c, d = coeffs
    base = np.exp(a * np.log(n) + b)
    return float(min(base * np.exp(c * np.log(iters) + d) / np.exp(c * np.log(1000) + d), 0.5))


def holdout(X, y, task, frac, seed):
    from sklearn.model_selection import train_test_split

    strat = y if task != dq.REGRESSION else None
    return train_test_split(X, y, test_size=frac, random_state=seed + 7919, stratify=strat)


def make(task, seed, threads, **params):
    from alloygbm import GBMClassifier, GBMRegressor

    cls = GBMClassifier if task != dq.REGRESSION else GBMRegressor
    return cls(seed=seed, n_jobs=threads, **params)


def fit_arm(arm, task, X, y, seed, threads):
    kind = arm["kind"]
    p = dict(arm.get("params", {}))
    info = {}
    if kind == "default":
        m = make(task, seed, threads, **p)
        m.fit(X, y)
        return m, info
    if kind == "cb_lr":
        lr = catboost_lr(len(y), task, arm["rounds"])
        m = make(task, seed, threads, learning_rate=lr, n_estimators=arm["rounds"], **p)
        m.fit(X, y)
        info["lr"] = lr
        return m, info
    lr = arm["lr"] if arm["lr"] != "cb" else catboost_lr(len(y), task, arm["cap"])
    Xt, Xv, yt, yv = holdout(X, y, task, arm["frac"], seed)
    m = make(task, seed, threads, learning_rate=lr, n_estimators=arm["cap"],
             early_stopping_rounds=arm["patience"], **p)
    _t = time.perf_counter()
    m.fit(Xt, yt, eval_set=(Xv, yv))
    info["es_seconds"] = time.perf_counter() - _t
    # best_iteration_ is 0-based, so arms without rel_tol refit one round
    # short of the holdout optimum; kept as run for the published table.
    best = m.best_iteration_ or m.rounds_completed_
    info.update(lr=lr, best=int(best))
    tol = arm.get("rel_tol")
    if tol:
        curve = list(m.evals_result_["validation"].values())[0]
        b_val, b_r, stop = curve[0], 0, len(curve) - 1
        for r_, v in enumerate(curve):
            if v < b_val * (1.0 - tol):
                b_val, b_r = v, r_
            if r_ - b_r >= arm["patience"]:
                stop = r_
                break
        info["emu_best"], info["emu_stop"] = b_r, stop
        info["es_time_frac"] = stop / max(1, len(curve) - 1 + (arm["patience"] if m.stop_reason_ == "ValidationLossPlateau" else 0))
        best = max(1, b_r)
    if kind == "es_keep":
        return m, info
    scale = arm.get("refit_scale", 1.0 / (1.0 - arm["frac"]))
    n_refit = max(1, int(round(best * scale)))
    m2 = make(task, seed, threads, learning_rate=lr, n_estimators=n_refit, **p)
    m2.fit(X, y)
    info["refit_rounds"] = n_refit
    return m2, info


ARMS = {
    # The pre-auto-rounds defaults (100 rounds at 0.1).
    "default": {"kind": "default", "params": {"n_estimators": 100, "learning_rate": 0.1}},
    "es_keep_lr0.1": {"kind": "es_keep", "lr": 0.1, "cap": 2000, "frac": 0.1, "patience": 50},
    "es_refit_lr0.1": {"kind": "es_refit", "lr": 0.1, "cap": 2000, "frac": 0.1, "patience": 50},
    "es_refit_lr0.05": {"kind": "es_refit", "lr": 0.05, "cap": 3000, "frac": 0.1, "patience": 100},
    "es_keep_lr0.05": {"kind": "es_keep", "lr": 0.05, "cap": 3000, "frac": 0.1, "patience": 100},
    "es_refit_cb": {"kind": "es_refit", "lr": "cb", "cap": 3000, "frac": 0.1, "patience": 100},
    "es_refit_lr0.05_f20": {"kind": "es_refit", "lr": 0.05, "cap": 3000, "frac": 0.2, "patience": 100},
    "cb_lr_1000": {"kind": "cb_lr", "rounds": 1000},
    "r05_c1000_p50_refit": {"kind": "es_refit", "lr": 0.05, "cap": 1000, "frac": 0.1, "patience": 50},
    "r05_c1000_p50_keep": {"kind": "es_keep", "lr": 0.05, "cap": 1000, "frac": 0.1, "patience": 50},
    "r05_p100_tol1e-3_refit": {"kind": "es_refit", "lr": 0.05, "cap": 3000, "frac": 0.1, "patience": 100, "rel_tol": 1e-3},
    "r05_p100_tol1e-4_refit": {"kind": "es_refit", "lr": 0.05, "cap": 3000, "frac": 0.1, "patience": 100, "rel_tol": 1e-4},
    "r10_c1000_p50_tol1e-3_refit": {"kind": "es_refit", "lr": 0.1, "cap": 1000, "frac": 0.1, "patience": 50, "rel_tol": 1e-3},
}


def main():
    warnings.filterwarnings("ignore")
    names = sys.argv[1].split(",")
    seeds = list(range(int(sys.argv[2])))
    out = Path(sys.argv[3])
    threads = 4
    suite = dq.build_suite(quick="--quick" in sys.argv)
    records = []
    for ds in suite:
        for seed in seeds:
            X_train, X_test, y_train, y_test = dq.split(ds, seed)
            ytr = y_train * ds.target_scale if ds.task == dq.REGRESSION else y_train
            for name in names:
                t0 = time.perf_counter()
                m, info = fit_arm(ARMS[name], ds.task, X_train, ytr, seed, threads)
                secs = time.perf_counter() - t0
                if ds.task == dq.REGRESSION:
                    pred = np.asarray(m.predict(X_test), dtype=float) / ds.target_scale
                else:
                    pred = m.predict_proba(X_test)
                loss = dq.score(ds.task, y_test, pred, np.unique(ds.y))
                records.append({"dataset": ds.name, "task": ds.task, "stratum": ds.stratum,
                                "seed": seed, "arm": name, "loss": loss, "fit_seconds": secs,
                                "error": None, **info})
        print(ds.name, "done", flush=True)
    result = {"arms": [{"name": n, "library": "alloygbm", "version": "proto"} for n in names],
              "seeds": seeds,
              "datasets": [{"name": d.name, "task": d.task, "stratum": d.stratum} for d in suite],
              "records": records}
    out.write_text(json.dumps(result, indent=1))
    print(dq.render_summary(result))


if __name__ == "__main__":
    main()
