"""Targeted feature-shape fixtures for comparing quantile border methods.

Each fixture has one feature whose shape separates border methods (a mass
point inside continuous data, rounded values, a zero mass with a heavy tail,
a flat Zipf) plus three Gaussian features, one of which adds a linear term. Trains GBMRegressor at its
defaults over 10 seeds and writes per-fixture test RMSE to a JSON file. Pick
the border method with the ALLOYGBM_EXPERIMENT_* env vars, e.g.::

    python benchmarks/greedy_log_sum_shapes.py greedy.json
    ALLOYGBM_EXPERIMENT_GREEDY_LOG_SUM_BINS=1 \\
        python benchmarks/greedy_log_sum_shapes.py greedy_log_sum.json
"""

import json
import sys

import numpy as np
from alloygbm import GBMRegressor
from sklearn.model_selection import train_test_split


def fixtures(seed):
    rng = np.random.default_rng(seed)
    n = 20000
    out = {}
    # spike: 5% mass at 1.0 inside a continuous feature, signal kinks near it
    x = np.where(rng.random(n) < 0.05, 1.0, rng.normal(size=n))
    z = rng.normal(size=(n, 3))
    out["spike"] = (
        np.column_stack([x, z]),
        np.sin(3 * x) + (x == 1.0) * 1.5 + 0.5 * z[:, 0] + rng.normal(0, 0.3, n),
    )
    # rounded: ~600 distinct values, step signal
    x = np.round(rng.normal(size=n) * 100) / 100
    out["rounded_600"] = (
        np.column_stack([x, z]),
        np.floor(4 * x) / 2 + 0.5 * z[:, 1] + rng.normal(0, 0.3, n),
    )
    # heavy tail with mass point at 0
    x = np.where(rng.random(n) < 0.3, 0.0, rng.lognormal(0, 1.5, n))
    out["mass0_lognormal"] = (
        np.column_stack([x, z]),
        np.log1p(x) + 0.5 * z[:, 2] + rng.normal(0, 0.3, n),
    )
    # many mid-weight values: zipf exponent 1.1, 5k distinct
    x = np.minimum(rng.zipf(1.1, n), 5000).astype(float)
    out["zipf_flat"] = (
        np.column_stack([x, z]),
        np.sin(np.log(x) * 2) + 0.5 * z[:, 0] + rng.normal(0, 0.3, n),
    )
    return out


res = {}
for seed in range(10):
    for k, (X, y) in fixtures(seed).items():
        Xtr, Xte, ytr, yte = train_test_split(
            X.astype(np.float32),
            y.astype(np.float32),
            test_size=0.25,
            random_state=seed,
        )
        p = GBMRegressor(seed=seed).fit(Xtr, ytr).predict(Xte)
        res.setdefault(k, []).append(float(np.sqrt(np.mean((p - yte) ** 2))))
with open(sys.argv[1], "w") as output:
    json.dump(res, output)
