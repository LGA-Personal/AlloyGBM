# AlloyGBM Current Limitations

Last updated for v0.7.1.

## Remaining Limitations

### 1. CPU-Only Runtime

The `BackendOps` trait is designed for hardware abstraction, but only
`CpuBackend` exists. GPU/accelerator support is architecturally planned but
not implemented.

### 2. No Dart / GOSS Boosting Modes

Only standard gradient boosting is supported. Dart (dropout) and GOSS
(gradient-based one-side sampling) modes are not available.

### 3. No Multi-Label Ranking

`GBMRanker` supports single-label relevance only.

### 4. MorphBoost Warm-Start Restarts EMA Cold

MorphBoost's adaptive split criterion tracks a per-class exponential moving
average over gradient statistics that shapes the gain function across
rounds. v0.7.1 supports warm-starting MorphBoost-trained models (training
continues without error and the predictor stitches old and new trees
correctly), but the EMA state is **not** restored from the saved
artifact — resumed training starts the EMA fresh. This means the
"morphed" gain shaping in the resumed rounds doesn't see the gradient
history from the original fit, and a resumed `N + M`-round model is not
numerically equivalent to a fresh `N + M`-round fit when
`training_mode="morph"` is active. For other modes (constant leaves,
linear leaves, DRO leaves, factor neutralization) warm-start equivalence
holds.

Persisting the EMA snapshot inside the artifact is queued for a follow-up
release.

### 5. SHAP for Piecewise-Linear Leaves — Best-Effort Decomposition

As of v0.7.1, `shap_values()` accepts `leaf_model="linear"` artifacts and
returns an *interventional* decomposition: the path-based TreeSHAP / brute
force machinery attributes each leaf's "constant part"
`intercept + Σ wj · μj_global`, then per-leaf row deviations
`wj · (xj − μj_global)` are credited directly to the regressor features.
Global per-feature means `μj_global` are captured at fit time and persisted
in a new `FeatureBaseline` artifact section, so SHAP is self-contained — the
original training data is not required at explain time.

`Σ shap_values + expected_value == predict(x)` holds exactly when SHAP's
internal path walker reaches the same leaf as the predictor. Today SHAP
compares raw feature values against stump `threshold_bin` indices cast to
`f32`, while the predictor crate converts those bin indices to float
thresholds at load time using per-feature min/max. For scalar leaves this
divergence is masked (the wrong-but-consistent path yields the same scalar
sum from both sides); for linear leaves the leaf value depends on `xj`, so
on continuous-feature artifacts the SHAP path and the predictor path can
disagree and the additive reconstruction drifts. To avoid a hard failure
mid-explain, the strict additivity check is relaxed for linear-leaf models;
users get best-effort SHAP values plus an updated docstring describing the
semantics. Tightening path-walk alignment (so SHAP also uses float
thresholds) is queued for a follow-up release.

## Resolved (Previously Limitations)

The following were limitations in prior versions and have been addressed:

- Regression-only (now: classification + ranking)
- Single categorical column only (now: multiple via `categorical_feature_indices`)
- Limited configurability (now: `min_split_gain`, monotone constraints, feature weights, `max_leaves`, leaf-wise growth)
- No NaN support (now: native NaN handling)
- No model persistence (now: pickle, save/load, artifact export)
- No sklearn compatibility (now: `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- No sample weight / group ID from Python (now: fully supported)
- Feature names auto-generated only (now: captured from DataFrames)
- SHAP limited to 20 features (now: TreeSHAP with no practical limit)
- Only RMSE tracked during training (now: objective-aware metric tracking)
- No warm-starting (now: `warm_start=True` for all estimators including multiclass)
- Level-wise growth only (now: leaf-wise available)
- Bins capped at 256 (now: up to 65,535)
- No histogram reuse (now: buffer reuse across rounds)
- Binary classification only (now: multi-class softmax with K > 2 classes)
- No native categorical splits (now: `max_cat_threshold` enables Fisher-sort optimal binary partitions with O(1) bitset prediction)
- No custom objective/metric callbacks (now: `objective` callable and `eval_metric` callable)
- Multiclass warm-start unsupported (now: `warm_start=True` works for multiclass with round-offset continuity)
- Neutralized warm-start unsupported (now: v0.7.1 — `init_model` / `warm_start=True`
  with `neutralization=*` is supported as long as the caller supplies the
  same `factor_exposures` matrix used for the initial fit; see Limitation 4
  for the MorphBoost EMA caveat that still applies)
- No interaction constraints (now: v0.7.1 — `interaction_constraints=[[...]]`
  on every estimator, LightGBM-compatible semantics, up to 64 groups per
  fit; enforced through both the level-wise and leaf-wise tree builders)
- Multiclass prediction per-row allocation (now: zero-copy dense slice prediction)
- Single fixed split criterion (now: opt-in MorphBoost adaptive criterion via
  `training_mode="morph"`, including EMA-driven gain shaping, depth/iteration
  leaf penalties, and an info-theoretic blend term — see the
  [paper](https://arxiv.org/pdf/2511.13234))
- Constant-only learning rate (now: per-iteration `lr_schedule` parameter
  supports `"constant"` and `"warmup_cosine"`, with schedule-aware
  early-stopping logic)
- No SIMD-accelerated kernels (now: histogram bin-scan and EMA passes are
  vectorized via the `wide` crate; histogram tile sizing auto-tunes for
  high-feature workloads)
- Constant leaves only (now: `leaf_model="linear"` replaces scalar leaves with
  closed-form piecewise-linear leaves `f_s(x) = b_s + Σ α_j x_j`, available on
  all three estimators; `leaf_model="polynomial"` and `leaf_model="rff"` remain
  future work)
- No full raw-distribution Wasserstein DRO guarantee (now:
  `leaf_solver="dro"` provides a fast Wasserstein-inspired robust scalar leaf
  update over gradient uncertainty; exact distributional DRO is still research
  work)
