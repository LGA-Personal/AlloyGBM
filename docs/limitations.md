# AlloyGBM Current Limitations

Last updated for v0.7.4.

## Remaining Limitations

### 1. CPU-Only Runtime

The `BackendOps` trait is designed for hardware abstraction, but only
`CpuBackend` exists. GPU/accelerator support is architecturally planned but
not implemented.

### 2. No Dart / GOSS Boosting Modes

Only standard gradient boosting is supported. Dart (dropout) and GOSS
(gradient-based one-side sampling) modes are not available.

### 3. Multi-Label Ranking — Independent Per-Label Trees

As of v0.7.1, ``MultiLabelGBMRanker`` exposes a unified multi-output
ranking API: ``y`` is shaped ``(n_rows, n_labels)`` and ``predict``
returns scores with the same column layout.  Internally, the wrapper
trains one independent :class:`GBMRanker` per label, sharing the same
``group`` (and optional ``factor_exposures``) so the per-label fits
remain comparable.  This makes the implementation a thin orchestration
layer that reuses every existing :class:`GBMRanker` feature
(warm-start, neutralization, MorphBoost, PL leaves, DRO, interaction
constraints).

Numerically the wrapper is equivalent to training each label
separately.  Joint shared-tree multi-label boosting — where a single
ensemble updates all label predictions simultaneously via shared splits
— would let correlated labels share split information across trees and
typically reduces total model size for related tasks.  Deferred to
v0.8.0, paired with the shared-histogram speedup where the
``MulticlassSoftmaxObjective``-style K-tree-per-round engine plumbing
for ranking objectives has a real performance story.

### 4. SHAP On The Mixed Linear-Rank Binning Path (Narrow)

When `continuous_binning_strategy="linear"` is combined with
per-feature *rank-based* linear binning enabled on at least one feature
(`_continuous_feature_linear_rank_flags` has any `True` entry), the
Python `shap_values()` path falls back to the legacy
quantize-then-walk SHAP entry point (`shap_explain_rows` /
`shap_explain_rows_dense`) instead of the predictor-aligned
`shap_explain_rows_with_binning` variants.  This is because the
rank-aware `BinningContext` conversion is not yet wired through.

For `leaf_model="constant"` artifacts the legacy path is still correct
under the usual round-off tolerance, so this path is largely benign.
For `leaf_model="linear"` artifacts it triggers the
`binning.is_none() && model_has_linear_leaves(model)` exemption in
`crates/shap/src/lib.rs::verify_additivity` and you get best-effort
interventional explanations rather than strict additivity.

Deferred to v0.8.0.  Users who need strict PL-leaf additivity in the
meantime can pick `continuous_binning_strategy="quantile"` or use the
pure-`linear` mode (no per-feature rank flags set).

### 5. TreeSHAP Polynomial Path Additivity Drift (Pre-existing)

For models with `distinct_split_feature_count > MAX_EXACT_SPLIT_FEATURES`
(=25 — see `crates/shap/src/lib.rs`), SHAP's dispatcher switches from
the brute-force exact Shapley path to the polynomial-time TreeSHAP
implementation (`explain_rows_tree_shap`).  On large gradient-trained
trees of depth ≥ 6 with ≥ 30 distinct split features the TreeSHAP
polynomial path drifts from strict additivity by ~0.5–1% of
`|predict(x)|`.  This affects both `leaf_model="constant"` and
`leaf_model="linear"` artifacts (it is independent of the v0.7.4
linear-leaf fix).

The bug is **pre-existing in v0.7.3** and earlier; it was uncovered by
the v0.7.4 PR #27 review.  Minimal Rust reproductions (asymmetric
2-stump and 4-stump spine trees) do not trigger it, so it requires
specific topological conditions met only by full gradient-boosting
output.

The internal Rust `verify_additivity` is the ground truth: when a
model triggers the polynomial path on these conditions, `shap_values()`
will raise `RuntimeError: row N additivity check failed`.  Users
encountering this can:

- Pin `n_estimators` or `max_depth` smaller so the tree uses ≤ 25
  distinct split features (brute-force path is correct).
- Use the legacy SHAP entry points without a `BinningContext` if
  best-effort interventional explanations are acceptable.
- Wait for the v0.7.x / v0.8.0 follow-up — coverage is pinned by an
  `@xfail(strict=True)` regression test in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` so the fix
  will land with that test flipping to a regular pass.

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
  same `factor_exposures` matrix used for the initial fit)
- No interaction constraints (now: v0.7.1 — `interaction_constraints=[[...]]`
  on every estimator, LightGBM-compatible semantics, up to 64 groups per
  fit; enforced through both the level-wise and leaf-wise tree builders)
- Single-label ranking only (partially resolved: v0.7.1 — `MultiLabelGBMRanker`
  exposes a unified `fit`/`predict` for K ranking labels per item, trained
  as K independent per-label `GBMRanker` instances sharing `group` and
  `factor_exposures`.  Joint shared-tree training is a follow-up; see
  Limitation 3)
- MorphBoost warm-start restarts EMA cold (now: v0.7.3 — MorphMetadata
  artifact section bumped to v2 with appended `Vec<GradientEmaStats>`;
  `WarmStartState` / `MultiClassWarmStartState` carry the snapshot and
  the engine seeds `MorphState.ema_stats` from it on resumed fits)
- SHAP additivity tolerance was f32-tight at `1e-5` absolute (now:
  v0.7.3 — `atol + rtol * |predicted|`, numpy `allclose` semantics)
- SHAP path-walker used bin-index thresholds instead of float
  thresholds (now: v0.7.3 — `shap::BinningContext` + new PyO3 entry
  points pass per-feature mins / maxs / cuts so the walker compares
  against the same float thresholds the predictor uses; resolved for
  scalar-leaf artifacts on continuous features.  The PL-leaf piece
  was finished in v0.7.4 — see the dedicated entry below)
- RUSTSEC-2025-0020 in `pyo3 < 0.24.1` (now: v0.7.3 — pyo3 0.23.5 →
  0.24, `deny.toml` and the cargo-audit CI step no longer ignore the
  advisory)
- Strict SHAP additivity for `leaf_model="linear"` on continuous
  features (now: v0.7.4 — `distribute_linear_terms_for_row` credits
  `wⱼ · (xⱼ − μⱼ)` at every visited node along the row's path,
  matching how `predict` accumulates `leaf.eval_row(row)` at each
  visited node; the linear-leaf exemption in `verify_additivity` was
  removed when a `BinningContext` is supplied — i.e. for the default
  Python path on continuous features.  Strict additivity now holds
  for `GBMRegressor`, `GBMClassifier`, and `GBMRanker` under
  `leaf_model="linear"`, with `training_mode="manual"` or
  `"morph"`, on both `quantile` and `linear` binning, with or without
  `interaction_constraints`, across `lambda_l2`, `max_depth`,
  `n_estimators` and skewed-scale features.  The legacy non-binning
  path retains the exemption.)
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
