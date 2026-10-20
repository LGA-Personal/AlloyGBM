"""Small-data binary classification dev suite (PMLB, disjoint from OpenML-CC18).

The OpenML pilot (docs/benchmarks/data/frontier_matrix_v1/openml_pilot.md)
showed Alloy's binary gap is a small-data gap: every CC18 binary task where
Alloy trails the best peer by more than 10% has fewer than 5,000 rows. CC18
is the held-out test set, so binary defaults are developed here instead, on
PMLB datasets that do not duplicate a CC18 or CTR23 task. Excluded as
duplicates (same data under another name or the same shape): wdbc / breast-w,
pima / diabetes, german / credit-g, australian / crx / credit-approval,
tic-tac-toe, kr-vs-kp / chess, spambase, churn, phoneme, dis (= sick).

Larger datasets are also scored after a stratified subsample to 1,000 rows
(suffix ``@1k``), because the small-n regime is where the gap is.

Usage::

    python benchmarks/binary_dev_suite.py download --cache DIR
    python benchmarks/binary_dev_suite.py run --cache DIR --config base \\
        --config 'l2=lambda_l2=3' --peers catboost lightgbm --output out.json
    python benchmarks/binary_dev_suite.py report out.json [--baseline base]

A ``--config`` is ``name`` (Alloy defaults) or ``name=key=value,key=value``
(GBMClassifier keyword arguments; values parsed as Python literals). A
``library=catboost`` (or ``lightgbm`` / ``xgboost``) item runs that peer with
the other items as its keyword arguments, for ablating a peer's defaults.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"

# Real (or classic benchmark) binary datasets, 150 to 20k rows.
REAL = [
    "hepatitis", "backache", "sonar", "biomed", "spect", "_deprecated_heart_statlog",
    "breast_cancer", "heart_disease_cleveland", "heart_disease_hungarian", "haberman",
    "bupa", "spectf", "ionosphere", "horse_colic_surgery", "congressional_voting_records",
    "saheart", "clean1", "irish", "profb", "tokyo1", "_deprecated_flare",
    "Hill_Valley_with_noise", "titanic", "hypothyroid", "banana", "ring", "twonorm",
    "clean2", "coil2000", "magic", "analcatdata_lawsuit", "prnn_synth",
]
# Synthetic logic / epistasis tasks (the madelon failure mode: signal only in
# feature interactions, invisible to the first few greedy splits).
SYNTHETIC = [
    "monk1", "monk2", "monk3", "parity5+5", "mofn_3_7_10", "xd6", "threeOf9", "corral",
    "GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1",
    "GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1",
    "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001",
]
SUBSAMPLE_1K = ["magic", "coil2000", "clean2", "ring", "twonorm", "banana", "hypothyroid", "titanic"]
MAX_ROWS = 10_000
SHIFT = 0.01  # added to both losses before a ratio (shifted geometric mean)


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in REAL + SYNTHETIC:
        path = cache / f"{name}.tsv.gz"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
            print("downloaded", name)


def _stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        take = max(2, int(round(n * len(idx) / len(y))))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def load(cache: Path, names: list[str] | None = None):
    import pandas as pd

    out = []
    for name in REAL + SYNTHETIC:
        if names and name not in names and f"{name}@1k" not in names:
            continue
        frame = pd.read_csv(cache / f"{name}.tsv.gz", sep="\t")
        y_raw = frame.pop("target").to_numpy()
        classes = np.unique(y_raw)
        if len(classes) != 2:
            continue
        y = (y_raw == classes[1]).astype(np.float64)
        X = frame.to_numpy(dtype=np.float64)
        if len(y) > MAX_ROWS:
            keep = _stratified_subsample(y, MAX_ROWS, 0)
            X, y = X[keep], y[keep]
        if not names or name in names:
            out.append((name, X, y))
        if name in SUBSAMPLE_1K and (not names or f"{name}@1k" in names):
            keep = _stratified_subsample(y, 1000, 1)
            out.append((f"{name}@1k", X[keep], y[keep]))
    return out


# Kept out of the dev suite by its manifest (CC18 near-duplicates).
NPZ_EXCLUDED = {"ada", "gina", "madeline", "sylvine"}


def load_npz(directory: Path):
    """Binary tasks of the shared dev-suite cache (``dev_suite.py export``)."""
    out = []
    for file in sorted(directory.glob("*.npz")):
        z = np.load(file, allow_pickle=False)
        if str(z["kind"]) == "regression" or str(z["name"]) in NPZ_EXCLUDED:
            continue
        y = z["y"].astype(float)
        keep = np.isfinite(y)
        labels = np.unique(y[keep])
        if len(labels) != 2:
            continue
        X, y = z["X"][keep], (y[keep] == labels[1]).astype(np.float64)
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


def fit_predict(arm: str, params: dict, X, y, Xt, seed: int, threads: int):
    if arm == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(verbose=0, random_seed=seed, thread_count=threads, **params)
    elif arm == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(verbose=-1, random_state=seed, n_jobs=threads, **params)
    elif arm == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(random_state=seed, n_jobs=threads, **params)
    else:
        from alloygbm import GBMClassifier

        model = GBMClassifier(seed=seed, n_jobs=threads, **params)
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    proba = np.asarray(model.predict_proba(Xt))[:, 1]
    extra = {}
    if arm not in ("catboost", "lightgbm", "xgboost"):
        extra["rounds"] = int(getattr(model, "rounds_completed_", 0) or 0)
    return proba, fit_seconds, extra


def score(y, p):
    from sklearn.metrics import roc_auc_score

    p = np.clip(p, 1e-15, 1 - 1e-15)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    return ll, auc


def run(args) -> None:
    arms = []
    for text in args.config or ["base"]:
        name, params = parse_config(text)
        # ``library=catboost`` (or lightgbm / xgboost) ablates a peer instead.
        arms.append((params.pop("library", "alloy"), name, params))
    for peer in args.peers or []:
        arms.append((peer, peer, {}))
    datasets = load_npz(Path(args.npz)) if args.npz else load(Path(args.cache), args.datasets)
    records = []
    output = Path(args.output)
    if output.exists() and args.resume:
        records = json.loads(output.read_text())["records"]
    done = {(r["dataset"], r["arm"], r["split"]) for r in records}
    for name, X, y in datasets:
        repeats = args.repeats if len(y) < 3000 else 1
        for split, tr, te in splits(y, repeats):
            for library, arm, params in arms:
                if (name, arm, split) in done:
                    continue
                try:
                    p, fit_s, extra = fit_predict(library, params, X[tr], y[tr], X[te], split, args.threads)
                    ll, auc = score(y[te], p)
                    err = None
                except Exception as exc:  # noqa: BLE001 - record and continue
                    ll = auc = fit_s = float("nan")
                    extra, err = {}, f"{type(exc).__name__}: {exc}"
                records.append({"dataset": name, "rows": int(len(y)), "features": int(X.shape[1]),
                                "arm": arm, "split": split, "log_loss": ll, "auc": auc,
                                "fit_seconds": fit_s, "error": err, **extra})
        print(f"{name}: done ({len(y)} rows)", file=sys.stderr, flush=True)
        output.write_text(json.dumps({"records": records}))
    output.write_text(json.dumps({"records": records}))


def report(args) -> None:
    from collections import defaultdict

    records = []
    for path in args.inputs:
        text = gzip.decompress(Path(path).read_bytes()) if path.endswith(".gz") else Path(path).read_text()
        records += json.loads(text)["records"]
    cell = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["error"] is None:
            cell[r["dataset"]][r["arm"]].append(r)
    arms = sorted({r["arm"] for r in records}, key=lambda a: (a in ("catboost", "lightgbm", "xgboost"), a))
    base = args.baseline or arms[0]
    datasets = sorted(cell, key=lambda d: cell[d][base][0]["rows"] if cell[d].get(base) else 0)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))

    def common(d, a, b):
        sa = {r["split"]: r for r in cell[d].get(a, [])}
        sb = {r["split"]: r for r in cell[d].get(b, [])}
        keys = sorted(set(sa) & set(sb))
        return [sa[k] for k in keys], [sb[k] for k in keys]

    print(f"{'dataset':<34}{'n':>6} " + " ".join(f"{a[:11]:>11}" for a in arms))
    for d in datasets:
        row = [f"{mean(cell[d][a], 'log_loss'):11.4f}" if cell[d].get(a) else f"{'-':>11}" for a in arms]
        n = next(iter(cell[d].values()))[0]["rows"]
        print(f"{d[:34]:<34}{n:>6} " + " ".join(row))
    print()
    print(f"{'arm':<16}{'sgeo LL vs ' + base[:7]:>18}{'sgeo LL vs catb':>16}{'W/L vs ' + base[:8]:>16}{'geo AUCerr':>12}{'fit s':>9}{'rounds':>8}")
    for a in arms:
        ratios, cratios, aucr, wins, losses, fits, rounds = [], [], [], 0, 0, [], []
        for d in datasets:
            ra, rb = common(d, a, base)
            if not ra:
                continue
            la, lb = mean(ra, "log_loss"), mean(rb, "log_loss")
            # Shifted ratio: near-perfect datasets (loss ~1e-3) would otherwise
            # dominate the geomean with ratios driven by noise.
            ratios.append(math.log((la + SHIFT) / (lb + SHIFT)))
            diffs = np.array([x["log_loss"] - y["log_loss"] for x, y in zip(ra, rb)])
            se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0
            if abs(diffs.mean()) > 2 * se and abs(diffs.mean()) > 0.005 * lb:
                wins += diffs.mean() < 0
                losses += diffs.mean() > 0
            ea = 1 - mean(ra, "auc"); eb = 1 - mean(rb, "auc")
            if ea > 0 and eb > 0:
                aucr.append(math.log(ea / eb))
            fits.append(mean(ra, "fit_seconds"))
            if "rounds" in ra[0]:
                rounds.append(mean(ra, "rounds"))
            rc, cb = common(d, a, "catboost")
            if rc:
                cratios.append(math.log((mean(rc, "log_loss") + SHIFT) / (mean(cb, "log_loss") + SHIFT)))
        g = lambda xs: f"{math.exp(np.mean(xs)):.3f}" if xs else "-"
        print(f"{a[:16]:<16}{g(ratios):>18}{g(cratios):>16}{f'{wins}/{losses}':>16}{g(aucr):>12}"
              f"{np.mean(fits):9.3f}{(np.mean(rounds) if rounds else float('nan')):8.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download"); p.add_argument("--cache", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cache")
    p.add_argument("--npz", help="score the binary tasks of a dev_suite.py .npz cache instead")
    p.add_argument("--config", action="append")
    p.add_argument("--peers", nargs="*")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("report"); p.add_argument("inputs", nargs="+"); p.add_argument("--baseline")
    args = parser.parse_args()
    {"download": lambda: download(Path(args.cache)), "run": lambda: run(args), "report": lambda: report(args)}[args.cmd]()


if __name__ == "__main__":
    main()
