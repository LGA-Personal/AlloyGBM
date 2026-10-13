"""Fit a battery of configs; write sha256 of artifact bytes + predictions to argv[1].

GOLDEN_N sets the row count (default 1500), GOLDEN_JOBS the thread count (default 2).
"""
import sys, json, hashlib, numpy as np
from alloygbm import GBMRegressor, GBMClassifier, GBMRanker, MultiLabelGBMRanker
rng = np.random.default_rng(0)
import os
n = int(os.environ.get("GOLDEN_N", "1500"))
Xc = rng.normal(size=(n, 10))
Xc[rng.random((n,10)) < 0.05] = np.nan          # missing values
Xlow = np.floor(np.abs(rng.normal(size=(n, 6))) * 2)  # low-card ints
Xbin = (rng.random((n, 20)) < 0.1).astype(float)      # sparse binary
Xcat = rng.integers(0, 7, size=(n, 1)).astype(float)
X = np.hstack([Xc, Xlow, Xbin, Xcat])
f = np.nan_to_num(Xc[:, 0]) + Xlow[:, 0] * 0.5 - Xbin[:, 0] + (Xcat[:, 0] % 3) * 0.3 + np.nan_to_num(Xc[:,1]*Xc[:,2])
yreg = f + rng.normal(0, 0.3, n)
ybin = (f + rng.normal(0, 0.5, n) > np.median(f)).astype(int)
ymc = np.digitize(f + rng.normal(0, 0.5, n), np.quantile(f, [0.25, 0.5, 0.75]))
ypos = np.exp(f / 3)
group = np.repeat(np.arange(n // 15), 15)
yrank = np.clip(np.round(f - f.min()), 0, 4).astype(int)
# wide u16-bin variant
Xwide = np.hstack([rng.normal(size=(n, 3)), rng.integers(0, 2000, size=(n, 2)).astype(float)])
ywide = Xwide[:, 0] + Xwide[:, 3] / 1000 + rng.normal(0, .2, n)
cat_idx = [X.shape[1] - 1]
base = dict(n_estimators=25, n_jobs=int(os.environ.get("GOLDEN_JOBS", "2")))
configs = [
    ("reg", GBMRegressor, {}, yreg, X),
    ("reg_auto", GBMRegressor, {"n_estimators": "auto"}, yreg, X),
    ("reg_manual", GBMRegressor, {"training_policy": "manual"}, yreg, X),
    ("reg_leaf", GBMRegressor, {"tree_growth": "leaf", "max_leaves": 20}, yreg, X),
    ("reg_deep", GBMRegressor, {"max_depth": 10}, yreg, X),
    ("reg_l1l2", GBMRegressor, {"lambda_l1": 0.5, "lambda_l2": 2.0, "min_child_hessian": 1.0, "training_policy": "manual"}, yreg, X),
    ("reg_mono", GBMRegressor, {"monotone_constraints": {0: 1}}, yreg, X),
    ("reg_inter", GBMRegressor, {"interaction_constraints": [[0, 1, 2], [10, 11, 36]]}, yreg, X),
    ("reg_fw", GBMRegressor, {"feature_weights": {0: 0.5, 3: 2.0}}, yreg, X),
    ("reg_cat_native", GBMRegressor, {"categorical_feature_indices": cat_idx, "max_cat_threshold": 16}, yreg, X),
    ("reg_cat_te", GBMRegressor, {"categorical_feature_indices": cat_idx}, yreg, X),
    ("reg_pl", GBMRegressor, {"leaf_model": "linear"}, yreg, X),
    ("reg_dro", GBMRegressor, {"leaf_solver": "dro", "dro_radius": 0.1}, yreg, X),
    ("reg_morph", GBMRegressor, {"training_mode": "morph"}, yreg, X),
    ("reg_goss", GBMRegressor, {"boosting_mode": "goss"}, yreg, X),
    ("reg_dart", GBMRegressor, {"boosting_mode": "dart"}, yreg, X),
    ("reg_colnode", GBMRegressor, {"colsample_bynode": 0.5, "col_subsample": 0.8}, yreg, X),
    ("reg_quantile", GBMRegressor, {"objective": "quantile", "quantile_alpha": 0.3}, yreg, X),
    ("reg_poisson", GBMRegressor, {"objective": "poisson"}, ypos, X),
    ("reg_bundle", GBMRegressor, {"feature_bundling": "exact"}, yreg, X),
    ("reg_wide", GBMRegressor, {"continuous_binning_max_bins": 1024}, ywide, Xwide),
    ("bin", GBMClassifier, {}, ybin, X),
    ("bin_auto", GBMClassifier, {"n_estimators": "auto"}, ybin, X),
    ("bin_pl", GBMClassifier, {"leaf_model": "linear"}, ybin, X),
    ("mc", GBMClassifier, {}, ymc, X),
    ("mc_dart", GBMClassifier, {"boosting_mode": "dart"}, ymc, X),
    ("mc_leaf", GBMClassifier, {"tree_growth": "leaf", "max_leaves": 16}, ymc, X),
    ("rank", GBMRanker, {}, yrank, X),
]
out = {}
for name, E, kw, y, XX in configs:
    params = dict(base); params.update(kw)
    try:
        m = E(**params)
        if E is GBMRanker:
            m.fit(XX, y, group=group)
        elif "cat" in name:
            m.fit(XX, y, categorical_feature_values_list=[XX[:, cat_idx[0]].astype(int).tolist()])
        else:
            m.fit(XX, y)
        h = hashlib.sha256(m.artifact_bytes).hexdigest()[:16]
        p = np.asarray(m.predict_proba(XX) if E is GBMClassifier else m.predict(XX), dtype=np.float64)
        ph = hashlib.sha256(p.tobytes()).hexdigest()[:16]
        out[name] = [h, ph]
    except Exception as e:
        out[name] = ["ERR", repr(e)[:120]]
# joint multilabel
try:
    Y = np.column_stack([yrank, (yrank + 1) % 5])
    m = MultiLabelGBMRanker(n_estimators=15, multi_label_mode="joint").fit(X, Y, group=group)
    out["joint"] = ["-", hashlib.sha256(np.asarray(m.predict(X), dtype=np.float64).tobytes()).hexdigest()[:16]]
except Exception as e:
    out["joint"] = ["ERR", repr(e)[:120]]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
