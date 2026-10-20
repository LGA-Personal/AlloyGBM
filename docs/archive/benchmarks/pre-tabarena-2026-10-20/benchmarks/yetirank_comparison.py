"""Compare GBMRanker's yetirank with rank:ndcg_smoothed (the objective
named yetirank in v1.0.0 and earlier), rank:ndcg, CatBoost YetiRank and
LightGBM lambdarank. Every arm uses learning rate 0.1, depth 6 and the same
round count; the score is test NDCG@10, paired by seed.

The data directory holds learning-to-rank fixtures that ship with the
reference libraries' sources (no download needed):

- mslr_web1k/{train,test}: unzipped from CatBoost's
  catboost/pytest/data/mslr_web1k.zip (a 600 / 200 query MSLR-WEB10K slice)
- rank.{train,test}[.query]: LightGBM's examples/lambdarank
- train / test: CatBoost's catboost/pytest/data/querywise
- synthetic_ranking.csv: benchmarks/synthetic_ranking/prepare.py output

Usage::

    python benchmarks/yetirank_comparison.py DATA_DIR results.json [SEEDS] [DATASETS]
    python benchmarks/yetirank_comparison.py --summarize results.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

def summarize(path):
    rows = json.loads(Path(path).read_text())
    for name in sorted({r["dataset"] for r in rows}):
        by = {}
        for r in rows:
            if r["dataset"] == name:
                by.setdefault(r["arm"], {})[r["seed"]] = r["ndcg10"]
        base = by.get("alloy:yetirank", {})
        print(f"\n{name}")
        for arm, vals in by.items():
            seeds = sorted(set(vals) & set(base))
            diff = np.array([base[s] - vals[s] for s in seeds])
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
            print(f"  {arm:28s} {np.mean(list(vals.values())):.4f}  yetirank minus this: {diff.mean():+.4f} (se {se:.4f}, n={len(diff)})")

if sys.argv[1] == "--summarize":
    summarize(sys.argv[2]); sys.exit(0)
DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2])
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else None

def load_tsv(path, label_col, group_col, drop):
    a = np.loadtxt(path, delimiter="\t", dtype=object)
    keep = [c for c in range(a.shape[1]) if c not in drop + [label_col, group_col]]
    X = a[:, keep].astype(np.float32)
    y = a[:, label_col].astype(np.float32)
    _, g = np.unique(a[:, group_col], return_inverse=True)
    return X, y, g.astype(np.int64)

def load_svm(path, qpath, n_features=None):
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(str(path), n_features=n_features)
    sizes = np.loadtxt(qpath, dtype=int)
    g = np.repeat(np.arange(len(sizes)), sizes)
    return np.asarray(X.todense(), dtype=np.float32), y.astype(np.float32), g

def ndcg_at(y, p, g, k=10):
    out = []
    for q in np.unique(g):
        m = g == q; yy = y[m]; pp = p[m]
        order = np.argsort(-pp, kind="stable")[:k]
        disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
        dcg = ((2.0 ** yy[order] - 1) * disc).sum()
        ideal = np.sort(yy)[::-1][:k]
        idcg = ((2.0 ** ideal - 1) * disc[: len(ideal)]).sum()
        out.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(out))

def datasets():
    ds = {}
    Xtr, ytr, gtr = load_tsv(DATA / "mslr_web1k/train", 0, 1, [])
    Xte, yte, gte = load_tsv(DATA / "mslr_web1k/test", 0, 1, [])
    ds["mslr_web1k_subset"] = (Xtr, ytr, gtr, Xte, yte, gte, 300)
    Xtr, ytr, gtr = load_svm(DATA / "rank.train", DATA / "rank.train.query")
    Xte, yte, gte = load_svm(DATA / "rank.test", DATA / "rank.test.query", n_features=Xtr.shape[1])
    ds["lightgbm_rank_example"] = (Xtr, ytr, gtr, Xte, yte, gte, 200)
    Xtr, ytr, gtr = load_tsv(DATA / "train", 2, 1, [0, 3, 4])
    Xte, yte, gte = load_tsv(DATA / "test", 2, 1, [0, 3, 4])
    ds["catboost_querywise"] = (Xtr, ytr, gtr, Xte, yte, gte, 200)
    import csv
    rows = list(csv.DictReader(open(DATA / "synthetic_ranking.csv")))
    q = np.array([int(r["query_id"]) for r in rows]); y = np.array([float(r["relevance"]) for r in rows], np.float32)
    fcols = [c for c in rows[0] if c not in ("query_id", "relevance")]
    X = np.array([[float(r[c]) for c in fcols] for r in rows], np.float32)
    tr = q % 5 != 0
    ds["synthetic_ranking"] = (X[tr], y[tr], q[tr], X[~tr], y[~tr], q[~tr], 200)
    return ds

def fit_predict(arm, Xtr, ytr, gtr, Xte, rounds, seed):
    if arm.startswith("alloy:"):
        from alloygbm import GBMRanker
        m = GBMRanker(ranking_objective=arm[6:], n_estimators=rounds, learning_rate=0.1, max_depth=6, seed=seed)
        m.fit(Xtr, ytr, group=gtr)
        return m.predict(Xte)
    if arm == "catboost:YetiRank":
        from catboost import CatBoostRanker, Pool
        m = CatBoostRanker(loss_function="YetiRank", iterations=rounds, learning_rate=0.1, depth=6,
                           random_seed=seed, verbose=False, thread_count=-1)
        m.fit(Pool(Xtr, ytr, group_id=gtr))
        return m.predict(Xte)
    if arm == "lightgbm:lambdarank":
        import lightgbm as lgb
        sizes = np.unique(gtr, return_counts=True)[1]
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=rounds, learning_rate=0.1, max_depth=6,
                           num_leaves=63, random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, group=sizes)
        return m.predict(Xte)
    raise ValueError(arm)

ARMS = ["alloy:yetirank", "alloy:rank:ndcg_smoothed", "alloy:rank:ndcg", "catboost:YetiRank", "lightgbm:lambdarank"]
results = json.loads(OUT.read_text()) if OUT.exists() else []
done = {(r["dataset"], r["arm"], r["seed"]) for r in results}
for name, (Xtr, ytr, gtr, Xte, yte, gte, rounds) in datasets().items():
    if ONLY and name not in ONLY: continue
    for seed in SEEDS:
        for arm in ARMS:
            if (name, arm, seed) in done: continue
            if arm.startswith("lightgbm") and not np.all(ytr == np.round(ytr)): continue  # needs integer labels
            t = time.perf_counter(); p = fit_predict(arm, Xtr, ytr, gtr, Xte, rounds, seed); dt = time.perf_counter() - t
            r = dict(dataset=name, arm=arm, seed=seed, ndcg10=ndcg_at(yte, np.asarray(p), gte, 10), fit_s=dt)
            results.append(r); OUT.write_text(json.dumps(results, indent=1)); print(r, flush=True)
