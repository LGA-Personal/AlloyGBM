"""Fixed-round fit time on larger data (dense bins, where split-scan pruning helps least).

Usage: python benchmarks/fit_speed_scale.py out.json
"""
import json, sys, time, numpy as np
from alloygbm import GBMRegressor, GBMClassifier
rng = np.random.default_rng(7)
def reg(n, f):
    X = rng.normal(size=(n, f)); y = X[:, :5] @ np.linspace(1, .2, 5) + X[:, 0] * X[:, 1] + rng.normal(0, .3, n); return X, y
def mc(n, f, k):
    X = rng.normal(size=(n, f)); y = np.digitize(X[:, :3].sum(1) + .3 * rng.normal(size=n), np.quantile(X[:, :3].sum(1), np.linspace(0, 1, k + 1)[1:-1])); return X, y
def lowcard(n, f):
    X = rng.integers(0, 12, size=(n, f)).astype(float); y = X[:, 0] - X[:, 1] + (X[:, 2] > 6) * 3 + rng.normal(0, .5, n); return X, y
cases = {
  "reg_200k_x30_d6": (GBMRegressor, reg(200_000, 30), 6),
  "reg_200k_x30_d10": (GBMRegressor, reg(200_000, 30), 10),
  "lowcard_100k_x40_d8": (GBMRegressor, lowcard(100_000, 40), 8),
  "multiclass5_50k_x20_d6": (GBMClassifier, mc(50_000, 20, 5), 6),
}
out = {}
for name, (E, (X, y), depth) in cases.items():
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); E(n_estimators=100, learning_rate=0.1, max_depth=depth, n_jobs=4).fit(X, y); best = min(best, time.perf_counter() - t)
    out[name] = best; print(name, round(best, 3), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
