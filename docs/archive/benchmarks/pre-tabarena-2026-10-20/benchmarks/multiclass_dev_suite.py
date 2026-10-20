"""Multiclass classification dev suite (PMLB, disjoint from OpenML-CC18).

The matched-capacity OpenML run (depth 6, lr 0.05, early stopping on a 15%
validation split) put Alloy about 12% behind CatBoost on CC18 multiclass,
3/13/17 win/tie/loss. CC18 is the held-out test set, so multiclass defaults
are developed here instead, on PMLB multiclass datasets that do not duplicate
a CC18 task. Excluded as duplicates: balance-scale, vehicle, vowel, cmc /
contraceptive, car / car_evaluation, the six mfeat sets, segmentation, dna,
splice, texture, optdigits, satimage, pendigits, letter, connect-4,
analcatdata_dmft / authorship.

Larger datasets are capped at 10k rows (stratified) and a few are also scored
after a stratified subsample to 1,000 rows (suffix ``@1k``). Classes with
fewer than 10 rows are dropped so every fold and validation split sees them.

Two regimes:

* ``defaults``: every library at its defaults (Alloy: auto policy).
* ``matched``: the OpenML ``early_stopping`` regime: depth 6, lr 0.05, up to
  5,000 rounds, stop after 50 rounds without improvement on a stratified 15%
  validation split of the training fold.

Usage::

    python benchmarks/multiclass_dev_suite.py download --cache DIR
    python benchmarks/multiclass_dev_suite.py run --cache DIR --regime matched \\
        --config base --config 'l2=lambda_l2=3' --peers catboost --output out.json
    python benchmarks/multiclass_dev_suite.py report out.json [--baseline base]

A ``--config`` is ``name`` (Alloy defaults) or ``name=key=value,key=value``
(GBMClassifier keyword arguments; values parsed as Python literals). A
``library=catboost`` (or ``lightgbm`` / ``xgboost``) item runs that peer with
the other items as its keyword arguments, for ablating a peer's defaults.
Config keyword arguments override the regime's; ``ALLOYGBM_*`` items are set
as experiment environment variables for that fit only.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"

DATASETS = [
    "lymphography", "iris", "tae", "hayes_roth", "flags", "wine_recognition", "_deprecated_glass",
    "auto_insurance_symboling", "new_thyroid", "ecoli", "schizo", "movement_libras", "dermatology",
    "horse_colic_lesion_type", "cars", "calendarDOW", "collins", "soybean",
    "_deprecated_solar_flare_2", "yeast", "wine_quality_red", "led24", "led7", "allhypo", "allrep",
    "wine_quality_white", "waveform_21", "waveform_40", "page_blocks", "ann_thyroid", "nursery",
    "krkopt", "shuttle", "mnist", "fars", "sleep",
]
SUBSAMPLE_1K = ["waveform_40", "wine_quality_white", "nursery", "krkopt", "mnist", "led24", "fars"]
MAX_ROWS = {"mnist": 3_000}
DEFAULT_MAX_ROWS = 10_000
MIN_CLASS_ROWS = 10
SHIFT = 0.01  # added to both losses before a ratio (shifted geometric mean)
PEERS = ("catboost", "lightgbm", "xgboost")


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        path = cache / f"{name}.tsv.gz"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
            print("downloaded", name)


def _stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        take = max(MIN_CLASS_ROWS, int(round(n * len(idx) / len(y))))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def _relabel(X: np.ndarray, y_raw: np.ndarray):
    labels, counts = np.unique(y_raw, return_counts=True)
    kept = labels[counts >= MIN_CLASS_ROWS]
    mask = np.isin(y_raw, kept)
    X, y_raw = X[mask], y_raw[mask]
    y = np.searchsorted(kept, y_raw).astype(np.float64)
    return X, y


def load(cache: Path, names: list[str] | None = None):
    import pandas as pd

    out = []
    for name in DATASETS:
        if names and name not in names and f"{name}@1k" not in names:
            continue
        frame = pd.read_csv(cache / f"{name}.tsv.gz", sep="\t")
        y_raw = frame.pop("target").to_numpy()
        X = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        X, y = _relabel(X, y_raw)
        if len(np.unique(y)) < 3:
            continue
        cap = MAX_ROWS.get(name, DEFAULT_MAX_ROWS)
        if len(y) > cap:
            keep = _stratified_subsample(y, cap, 0)
            X, y = _relabel(X[keep], y[keep])
        if not names or name in names:
            out.append((name, X, y))
        if name in SUBSAMPLE_1K and (not names or f"{name}@1k" in names):
            keep = _stratified_subsample(y, 1000, 1)
            Xs, ys = _relabel(X[keep], y[keep])
            out.append((f"{name}@1k", Xs, ys))
    return out


# Kept out of the dev suite by its manifest (CC18 near-duplicates), plus
# PMLB duplicates and tasks too large for a quick loop.
NPZ_EXCLUDED = {"yeast", "shuttle", "helena"}


def load_npz(directory: Path):
    """Multiclass tasks of the shared dev-suite cache (``dev_suite.py export``)."""
    out = []
    for file in sorted(directory.glob("*.npz")):
        if file.stat().st_size > 50e6:
            continue
        z = np.load(file, allow_pickle=False)
        if str(z["kind"]) == "regression" or str(z["name"]) in NPZ_EXCLUDED:
            continue
        y = z["y"].astype(float)
        keep = np.isfinite(y)
        if len(np.unique(y[keep])) < 3:
            continue
        X, y = _relabel(z["X"][keep].astype(np.float64), y[keep])
        out.append((f"dev_{int(z['task_id'])}_{str(z['name'])}", X, y))
    return out


def splits(y: np.ndarray, repeats: int):
    from sklearn.model_selection import StratifiedKFold

    folds = 5
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1000 + r)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield r * folds + k, tr, te


def parse_config(text: str) -> tuple[str, dict]:
    if "=" not in text:
        return text, {}
    name, _, rest = text.partition("=")
    params = {}
    for item in rest.split(","):
        key, _, value = item.partition("=")
        try:
            params[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            params[key.strip()] = value.strip()
    return name, params


def regime_params(library: str, regime: str) -> dict:
    if regime == "defaults":
        return {}
    if library == "catboost":
        return {"iterations": 5000, "learning_rate": 0.05, "depth": 6, "early_stopping_rounds": 50}
    if library == "lightgbm":
        return {"n_estimators": 5000, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 64}
    if library == "xgboost":
        return {"n_estimators": 5000, "learning_rate": 0.05, "max_depth": 6, "early_stopping_rounds": 50}
    return {"n_estimators": 5000, "learning_rate": 0.05, "max_depth": 6, "early_stopping_rounds": 50}


def fit_predict(library: str, params: dict, regime: str, X, y, Xt, seed: int, threads: int):
    import os

    # ``ALLOYGBM_*`` config items are experiment environment variables, set
    # for this fit only.
    env = {key: str(value) for key, value in params.items() if key.startswith("ALLOYGBM_")}
    params = {key: value for key, value in params.items() if key not in env}
    os.environ.update(env)
    try:
        return _fit_predict(library, params, regime, X, y, Xt, seed, threads)
    finally:
        for key in env:
            os.environ.pop(key, None)


def _fit_predict(library: str, params: dict, regime: str, X, y, Xt, seed: int, threads: int):
    params = {**regime_params(library, regime), **params}
    fit_kwargs: dict = {}
    if regime == "matched":
        from sklearn.model_selection import train_test_split

        tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=seed, stratify=y)
        X, y, Xv, yv = X[tr], y[tr], X[va], y[va]
    if library == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(verbose=0, random_seed=seed, thread_count=threads,
                                   allow_writing_files=False, **params)
        if regime == "matched":
            fit_kwargs["eval_set"] = (Xv, yv)
    elif library == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(verbose=-1, random_state=seed, n_jobs=threads, **params)
        if regime == "matched":
            fit_kwargs.update(eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif library == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(random_state=seed, n_jobs=threads, tree_method="hist", **params)
        if regime == "matched":
            fit_kwargs.update(eval_set=[(Xv, yv)], verbose=False)
    else:
        from alloygbm import GBMClassifier

        model = GBMClassifier(seed=seed, n_jobs=threads, **params)
        if regime == "matched":
            fit_kwargs["eval_set"] = (Xv, yv)
    start = time.perf_counter()
    model.fit(X, y, **fit_kwargs)
    fit_seconds = time.perf_counter() - start
    proba = np.asarray(model.predict_proba(Xt), dtype=np.float64)
    extra = {}
    if library == "alloygbm":
        best = getattr(model, "best_iteration_", None)
        done = getattr(model, "rounds_completed_", None)
        extra["rounds"] = int(best) + 1 if best is not None else int(done or 0)
    elif library == "catboost":
        extra["rounds"] = int(model.tree_count_)
    return proba, fit_seconds, extra


def score(y, proba):
    p = np.clip(proba, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    yi = y.astype(int)
    ll = float(-np.mean(np.log(p[np.arange(len(yi)), yi])))
    err = float(np.mean(np.argmax(p, axis=1) != yi))
    return ll, err


def run(args) -> None:
    arms = []
    for text in args.config or ["base"]:
        name, params = parse_config(text)
        # ``library=catboost`` (or lightgbm / xgboost) ablates a peer instead.
        arms.append((params.pop("library", "alloygbm"), name, params))
    for peer in args.peers or []:
        arms.append((peer, peer, {}))
    datasets = load_npz(Path(args.npz)) if args.npz else load(Path(args.cache), args.datasets)
    if args.npz and args.datasets:
        datasets = [d for d in datasets if d[0] in args.datasets]
    records = []
    output = Path(args.output)
    if output.exists() and args.resume:
        records = json.loads(output.read_text())["records"]
    done = {(r["dataset"], r["arm"], r["split"]) for r in records}
    for name, X, y in datasets:
        repeats = args.repeats if len(y) < 1000 else 1
        for split, tr, te in splits(y, repeats):
            for library, arm, params in arms:
                if (name, arm, split) in done:
                    continue
                try:
                    p, fit_s, extra = fit_predict(library, params, args.regime, X[tr], y[tr], X[te],
                                                  split, args.threads)
                    ll, err = score(y[te], p)
                    error = None
                except Exception as exc:  # noqa: BLE001 - record and continue
                    ll = err = fit_s = float("nan")
                    extra, error = {}, f"{type(exc).__name__}: {exc}"
                records.append({"dataset": name, "rows": int(len(y)), "features": int(X.shape[1]),
                                "classes": int(len(np.unique(y))), "arm": arm, "split": split,
                                "log_loss": ll, "error_rate": err, "fit_seconds": fit_s,
                                "error": error, **extra})
        print(f"{name}: done ({len(y)} rows)", file=sys.stderr, flush=True)
        output.write_text(json.dumps({"regime": args.regime, "records": records}))
    output.write_text(json.dumps({"regime": args.regime, "records": records}))


def report(args) -> None:
    records = []
    for path in args.inputs:
        records += json.loads(Path(path).read_text())["records"]
    cell = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["error"] is None:
            cell[r["dataset"]][r["arm"]].append(r)
        elif args.verbose:
            print("error", r["dataset"], r["arm"], r["error"], file=sys.stderr)
    arms = sorted({r["arm"] for r in records}, key=lambda a: (a in PEERS, a))
    base = args.baseline or arms[0]
    ref = args.reference
    datasets = sorted(cell, key=lambda d: cell[d][base][0]["rows"] if cell[d].get(base) else 0)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))

    def common(d, a, b):
        sa = {r["split"]: r for r in cell[d].get(a, [])}
        sb = {r["split"]: r for r in cell[d].get(b, [])}
        keys = sorted(set(sa) & set(sb))
        return [sa[k] for k in keys], [sb[k] for k in keys]

    if not args.summary_only:
        print(f"{'dataset':<30}{'n':>6}{'K':>4} " + " ".join(f"{a[:10]:>10}" for a in arms))
        for d in datasets:
            row = [f"{mean(cell[d][a], 'log_loss'):10.4f}" if cell[d].get(a) else f"{'-':>10}" for a in arms]
            first = next(iter(cell[d].values()))[0]
            print(f"{d[:30]:<30}{first['rows']:>6}{first['classes']:>4} " + " ".join(row))
        print()
    print(f"{'arm':<22}{'LL vs ' + base[:8]:>15}{'LL vs ' + ref[:8]:>15}{'W/L vs ' + base[:7]:>15}"
          f"{'W/L vs ' + ref[:7]:>15}{'err vs ' + ref[:5]:>13}{'fit s':>8}{'rounds':>8}")
    for a in arms:
        ratios, rratios, errr, fits, rounds = [], [], [], [], []
        wl = {"base": [0, 0], "ref": [0, 0]}
        for d in datasets:
            for key, other, sink in (("base", base, ratios), ("ref", ref, rratios)):
                ra, rb = common(d, a, other)
                if not ra:
                    continue
                la, lb = mean(ra, "log_loss"), mean(rb, "log_loss")
                sink.append(math.log((la + SHIFT) / (lb + SHIFT)))
                diffs = np.array([x["log_loss"] - y["log_loss"] for x, y in zip(ra, rb)])
                se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0
                if abs(diffs.mean()) > 2 * se and abs(diffs.mean()) > 0.005 * lb:
                    wl[key][0] += diffs.mean() < 0
                    wl[key][1] += diffs.mean() > 0
                if key == "ref":
                    ea, eb = mean(ra, "error_rate"), mean(rb, "error_rate")
                    errr.append(math.log((ea + SHIFT) / (eb + SHIFT)))
            if cell[d].get(a):
                fits.append(mean(cell[d][a], "fit_seconds"))
                if "rounds" in cell[d][a][0]:
                    rounds.append(mean(cell[d][a], "rounds"))
        g = lambda xs: f"{math.exp(np.mean(xs)):.3f}" if xs else "-"
        print(f"{a[:22]:<22}{g(ratios):>15}{g(rratios):>15}{'%d/%d' % tuple(wl['base']):>15}"
              f"{'%d/%d' % tuple(wl['ref']):>15}{g(errr):>13}{np.mean(fits):8.2f}"
              f"{(np.mean(rounds) if rounds else float('nan')):8.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download"); p.add_argument("--cache", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cache")
    p.add_argument("--npz", help="score the multiclass tasks of a dev_suite.py .npz cache instead")
    p.add_argument("--regime", choices=("defaults", "matched"), default="matched")
    p.add_argument("--config", action="append")
    p.add_argument("--peers", nargs="*")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("report"); p.add_argument("inputs", nargs="+"); p.add_argument("--baseline")
    p.add_argument("--reference", default="catboost")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    {"download": lambda: download(Path(args.cache)), "run": lambda: run(args), "report": lambda: report(args)}[args.cmd]()


if __name__ == "__main__":
    main()
