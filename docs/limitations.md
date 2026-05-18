# AlloyGBM Current Limitations

Last updated for v0.8.0 (Commit 1 — Mixed linear-rank SHAP).

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
- SHAP on the mixed linear-rank binning path (now: v0.8.0 — new
  `BinningContext::LinearRank` variant carries per-feature sorted
  unique values + global mins/maxs + `max_data_bin`.  At the
  `explain_rows_from_model` entry point SHAP internally quantizes
  rows to bin indices (linear-quantize unflagged features,
  rank-quantize flagged features — exactly matching
  `predict_dense_quantized_linear_rank`) and dispatches the
  remainder with `BinningContext::PreBinned` semantics, so tree
  traversal and PL-leaf evaluation share the same bin-index space
  as the predictor.  Strict additivity now holds for
  `leaf_model="linear"` on the mixed linear-rank path; the
  predictor-aligned binning kwargs `_shap_binning_kwargs()`
  returns include `binning_kind="linear_rank"` whenever any
  per-feature rank flag is set.  Closes Limitation 4.)
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
- TreeSHAP polynomial-path additivity drift on trees with a feature
  appearing more than once on a root-to-leaf path (now: v0.7.5 —
  `ts_unextend_path` in `crates/shap/src/lib.rs` now shifts only
  the `feature_index`, `zero_fraction`, and `one_fraction` fields
  when removing a duplicate from the path, preserving the `pweight`
  values that the unwind loop has just computed in place.  The
  previous implementation shifted the entire `PathElement` struct,
  clobbering the post-unwind pweights with values from the elements
  being shifted down.  The reference implementation in
  `slundberg/shap` uses four parallel arrays and only shifts the
  first three.  Closes the pre-existing Limitation 5 from v0.7.3 /
  v0.7.4.  The Python regression
  `test_strict_additivity_via_tree_shap_polynomial_path` in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` is no
  longer `@xfail(strict=True)`.)
