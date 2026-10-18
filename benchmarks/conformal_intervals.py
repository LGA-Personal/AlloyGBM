"""Prediction-interval benchmark: LoBoost vs split conformal vs conformalized quantile regression.

For every regression task in the dev suite (``dev_suite.py export``) and every
seeded 75/25 train/test fold, the training rows are split again 75/25 into
fit and calibration rows. One ``GBMRegressor`` is fitted on the fit rows and
wrapped by:

- ``split``: :class:`alloygbm.conformal.SplitConformal` (one global quantile);
- ``loboost``: :class:`alloygbm.conformal.LoBoost` (leaf-prefix local quantiles);

and two quantile regressors (``objective="quantile"`` at alpha/2 and
1 - alpha/2) are fitted on the same fit rows for:

- ``qr``: the raw quantile band, uncalibrated;
- ``cqr``: conformalized quantile regression (Romano et al. 2019), calibrated
  on the same calibration rows.

Reported per method on the test fold: marginal coverage, mean width and mean
interval score (Gneiting & Raftery 2007; the LoBoost paper's SMIS), both
divided by the test-target standard deviation so tasks can be pooled,
worst-slab coverage (Cauchois et al. 2021; slab chosen on one half of the test
rows, coverage measured on the other half) and calibration / predict seconds.

Usage::

    python benchmarks/conformal_intervals.py --dev-suite DIR --seeds 3 --output out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

METHODS = ("split", "loboost", "qr", "cqr")


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    return (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


def worst_slab_coverage(
    X: np.ndarray, covered: np.ndarray, rng: np.random.Generator,
    delta: float = 0.2, directions: int = 200, grid: int = 21,
) -> float:
    """Worst coverage over slabs {a <= v.x <= b} holding at least ``delta`` of the rows.

    The slab is chosen on one random half of the rows and its coverage is
    measured on the other half, so the minimum is not biased by selection.
    """
    n = X.shape[0]
    if n < 100:
        return float("nan")
    Z = X.copy()
    medians = np.nanmedian(Z, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    Z = np.where(np.isfinite(Z), Z, medians)
    scale = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    order = rng.permutation(n)
    pick, check = order[: n // 2], order[n // 2:]
    best = (math.inf, None, 0.0, 0.0)
    quantiles = np.linspace(0.0, 1.0, grid)
    for _ in range(directions):
        v = rng.normal(size=Z.shape[1])
        proj = Z[pick] @ v
        cuts = np.quantile(proj, quantiles)
        for i in range(grid):
            for j in range(i + 1, grid):
                if quantiles[j] - quantiles[i] < delta:
                    continue
                inside = (proj >= cuts[i]) & (proj <= cuts[j])
                if inside.sum() == 0:
                    continue
                cov = covered[pick][inside].mean()
                if cov < best[0]:
                    best = (cov, v, cuts[i], cuts[j])
    _, v, a, b = best
    if v is None:
        return float("nan")
    proj = Z[check] @ v
    inside = (proj >= a) & (proj <= b)
    return float(covered[check][inside].mean()) if inside.any() else float("nan")


def run_fold(ds, fold, seed: int, alpha: float, threads: int, check_apply: bool) -> dict:
    import frontier_matrix as fm
    from alloygbm import GBMRegressor
    from alloygbm.conformal import LoBoost, SplitConformal

    train, test = fold
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train)
    cut = int(round(0.75 * len(shuffled)))
    fit_rows, cal_rows = np.sort(shuffled[:cut]), np.sort(shuffled[cut:])
    X_all = fm._frame(ds.X, ds.categorical, "alloygbm")
    X_fit, X_cal, X_te = (fm._take(X_all, r) for r in (fit_rows, cal_rows, test))
    y_fit, y_cal, y_te = ds.y[fit_rows], ds.y[cal_rows], ds.y[test]

    out: dict = {"fit_rows": len(fit_rows), "cal_rows": len(cal_rows), "test_rows": len(test)}
    started = time.perf_counter()
    model = GBMRegressor(seed=seed, n_jobs=threads).fit(X_fit, y_fit)
    out["fit_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    point = np.asarray(model.predict(X_te), dtype=float)
    out["predict_seconds"] = time.perf_counter() - started
    out["rmse"] = float(np.sqrt(np.mean((point - y_te) ** 2)))
    if check_apply:
        _, outputs = model._leaf_assignments(X_te)
        offset = point - outputs.sum(axis=1)
        out["apply_offset_spread"] = float(offset.max() - offset.min())

    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, cls in (("split", SplitConformal), ("loboost", LoBoost)):
        calibrator = cls(model)
        started = time.perf_counter()
        calibrator.calibrate(X_cal, y_cal, alpha=alpha)
        calibrate_seconds = time.perf_counter() - started
        started = time.perf_counter()
        intervals[name] = calibrator.predict_interval(X_te)
        out[f"{name}_calibrate_seconds"] = calibrate_seconds
        out[f"{name}_interval_seconds"] = time.perf_counter() - started
        if name == "loboost":
            out["loboost_regions"] = calibrator.n_regions_
            out["loboost_depth"] = calibrator.depth_
            out["loboost_min_group"] = calibrator.min_group_size_

    started = time.perf_counter()
    bands = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        qm = GBMRegressor(seed=seed, n_jobs=threads, objective="quantile", quantile_alpha=q)
        qm.fit(X_fit, y_fit)
        bands.append(qm)
    out["qr_fit_seconds"] = time.perf_counter() - started
    lo_cal, hi_cal = (np.asarray(m.predict(X_cal), dtype=float) for m in bands)
    lo_te, hi_te = (np.asarray(m.predict(X_te), dtype=float) for m in bands)
    lo_cal, hi_cal = np.minimum(lo_cal, hi_cal), np.maximum(lo_cal, hi_cal)
    lo_te, hi_te = np.minimum(lo_te, hi_te), np.maximum(lo_te, hi_te)
    intervals["qr"] = (lo_te, hi_te)
    scores = np.sort(np.maximum(lo_cal - y_cal, y_cal - hi_cal))
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    q_hat = scores[rank - 1] if rank <= len(scores) else math.inf
    intervals["cqr"] = (lo_te - q_hat, hi_te + q_hat)

    scale = float(np.std(y_te)) or 1.0
    X_num = np.asarray(ds.X[test], dtype=float)
    for name, (lo, hi) in intervals.items():
        covered = (y_te >= lo) & (y_te <= hi)
        out[f"{name}_coverage"] = float(covered.mean())
        out[f"{name}_width"] = float(np.mean(hi - lo)) / scale
        out[f"{name}_mis"] = float(np.mean(interval_score(y_te, lo, hi, alpha))) / scale
        out[f"{name}_wsc"] = worst_slab_coverage(X_num, covered, np.random.default_rng(seed + 7))
    return out


def summarize(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    summary: dict = {"tasks": len(by_task), "folds": len(rows), "methods": {}}
    for m in METHODS:
        summary["methods"][m] = {
            k: float(np.nanmean([r[f"{m}_{k}"] for r in rows]))
            for k in ("coverage", "width", "mis", "wsc")
        }
    for base in ("split", "cqr"):
        ratios, wins = [], 0
        for task_rows in by_task.values():
            lob = np.mean([r["loboost_mis"] for r in task_rows])
            ref = np.mean([r[f"{base}_mis"] for r in task_rows])
            ratios.append(lob / ref)
            wins += lob < ref
        summary[f"loboost_vs_{base}"] = {
            "mis_ratio_geomean": float(np.exp(np.mean(np.log(ratios)))),
            "tasks_loboost_lower_mis": int(wins),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    import dev_suite

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev-suite", type=Path, required=True, help="dev_suite.py export cache")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--tasks", type=int, nargs="*", default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    datasets = [d for d in dev_suite.load(args.dev_suite, splits=args.seeds, tasks=args.tasks)
                if d.task == "regression"]
    rows: list[dict] = []
    for ds in datasets:
        for seed, fold in enumerate(ds.folds):
            started = time.perf_counter()
            try:
                row = run_fold(ds, fold, seed, args.alpha, args.threads, check_apply=seed == 0)
            except Exception as error:  # keep going; record the failure
                print(f"{ds.spec.name} seed {seed}: FAILED {error!r}", flush=True)
                continue
            row.update(task=ds.spec.name, seed=seed)
            rows.append(row)
            print(
                f"{ds.spec.name:50s} s{seed} "
                + " ".join(f"{m}={row[f'{m}_coverage']:.3f}/{row[f'{m}_mis']:.3f}" for m in METHODS)
                + f" regions={row['loboost_regions']} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
            args.output.write_text(json.dumps({"alpha": args.alpha, "rows": rows}, indent=1))
    summary = summarize(rows)
    args.output.write_text(json.dumps({"alpha": args.alpha, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
