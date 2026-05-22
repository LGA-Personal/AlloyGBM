# AlloyGBM

[![CI](https://github.com/LGA-Personal/AlloyGBM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LGA-Personal/AlloyGBM/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/alloygbm.svg)](https://pypi.org/project/alloygbm/)
[![Python versions](https://img.shields.io/pypi/pyversions/alloygbm.svg)](https://pypi.org/project/alloygbm/)
[![Documentation Status](https://readthedocs.org/projects/alloygbm/badge/?version=latest)](https://alloygbm.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.92+](https://img.shields.io/badge/rust-1.92%2B-orange.svg)](https://www.rust-lang.org/)

AlloyGBM is a Rust-first gradient boosting library with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is built for fast native execution, deterministic training, and time-aware tabular workflows.

AlloyGBM is strongest on panel and finance-style problems where leakage-aware validation and practical iteration speed matter. It also performs competitively on general tabular benchmarks and includes native artifact prediction, TreeSHAP explanations, and purged time-series split helpers.

## When To Use AlloyGBM

AlloyGBM is a good fit when you want:

- a native Rust-backed gradient boosting library with regression, classification, and ranking
- deterministic CPU training and inference
- sklearn-compatible estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`)
- time-aware validation helpers for forecasting or panel-style workflows
- native prediction from serialized artifacts
- TreeSHAP explanations and global feature importances
- NaN/missing value support out of the box
- model persistence via pickle, save/load, or artifact export

## Installation

PyPI:

```bash
pip install alloygbm
```

From source:

```bash
python -m pip install --upgrade maturin
maturin develop --manifest-path bindings/python/Cargo.toml --release
```

AlloyGBM targets Python `3.11+` and uses a native Rust extension module.

Wheel targets for `0.10.2`:

- macOS `arm64`
- Linux `x86_64` (manylinux)
- source distribution for other platforms

## Quick Examples

### Regression

```python
from alloygbm import GBMRegressor, rmse

model = GBMRegressor(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1200,
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
print(rmse(y_test, model.predict(X_test)))
```

### Binary Classification

```python
from alloygbm import GBMClassifier, accuracy, log_loss

model = GBMClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)            # [0, 1, 1, 0, ...]
probas = model.predict_proba(X_test)      # [[P(0), P(1)], ...]

print("accuracy:", accuracy(y_test, labels))
print("log_loss:", log_loss(y_test, probas[:, 1]))
```

### Learning-to-Rank

```python
from alloygbm import GBMRanker, ndcg

model = GBMRanker(
    ranking_objective="rank:ndcg",
    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train, group=query_ids_train)

scores = model.predict(X_test)
print("NDCG@10:", ndcg(y_test, scores, group=query_ids_test, k=10))
```

### MorphBoost (Adaptive Split Criterion)

MorphBoost is an opt-in training mode that blends the standard gradient gain
with a normalized information-theoretic term. Across rounds, the blend ramps
in via a `tanh(iter/20)` warmup, an EMA over per-class gradient statistics
shapes split selection, and leaf magnitudes are scaled by a depth penalty
and per-iteration shrinkage. See the
[MorphBoost paper](https://arxiv.org/pdf/2511.13234) for the formulation.

```python
from alloygbm import GBMRegressor

# Constant LR (default) with morph adaptive split criterion
model = GBMRegressor(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.05,
    training_mode="morph",      # opt in
    morph_rate=0.1,             # per-round leaf shrinkage
    info_score_weight=0.3,      # blend weight for info-theoretic term
    depth_penalty_base=0.9,     # multiplier per depth level
    balance_penalty=True,       # penalize highly imbalanced splits
    seed=7,
)
model.fit(X_train, y_train)

# With warmup-cosine LR schedule (good fit for very-low-LR runs)
model = GBMRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    training_mode="morph",
    lr_schedule="warmup_cosine",
    lr_warmup_frac=0.1,         # fraction of n_estimators spent in warmup
    seed=7,
)
```

`training_mode="morph"` works with `GBMClassifier` and `GBMRanker` too, with
identical parameter semantics.

### DRO Leaf Solver (Robust Scalar Leaves)

Set `leaf_solver="dro"` to use a fast Wasserstein-inspired robust Newton update
for scalar leaves. The solver penalizes each candidate leaf by within-leaf
gradient dispersion, reducing sensitivity to noisy or weak leaf signals while
keeping prediction speed identical to standard constant leaves.

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    leaf_solver="dro",
    dro_radius=0.05,
    dro_metric="wasserstein",
    seed=7,
)
model.fit(X_train, y_train)
```

`leaf_solver="dro"` works with `GBMRegressor`, `GBMClassifier`, and
`GBMRanker`, and composes with `training_mode="morph"`. It requires
`leaf_model="constant"`; piecewise-linear leaves still use the standard PL
solver. `dro_radius=0.0` preserves standard-leaf predictions while retaining
DRO metadata in the artifact.

### Factor-Neutral Boosting

Use `neutralization="per_round_gradient"` with `fit(..., factor_exposures=F)` to project each boosting round's pseudo-residuals away from user-supplied nuisance factors. This is useful when common factors explain high-variance signal that you do not want the model to spend tree capacity learning.

This is a training-time regularization tool. It does not guarantee prediction-time zero exposure unless predictions are neutralized against evaluation-time factors outside the model.

Constructor parameters:

```python
GBMRegressor(
    neutralization="none",                 # "none" | "pre_target" | "per_round_gradient" | "split_penalty"
    factor_neutralization_lambda=1e-6,      # finite, >= 0 ridge added to F^T W F
    factor_penalty=0.0,                     # finite, >= 0; only active for neutralization="split_penalty"
)
```

`factor_exposures` is dense, row-major, finite, and shaped
`(n_rows, n_factors)`. It is fit data, not constructor state, so sklearn
cloning remains clean and large matrices are not embedded in estimator params.

Mode semantics:

`neutralization="none"` preserves current behavior and ignores
`factor_exposures` unless a non-`None` matrix is provided with an inactive mode,
in which case Python raises a clear validation error to prevent silent user
mistakes.

`neutralization="pre_target"` residualizes the regression target once before
training:

```text
y_perp = y - F (F^T W F + lambda I)^-1 F^T W y
```

This mode is supported for `GBMRegressor` only. It is rejected for
classification and ranking because target residualization is not well-defined
for class labels or ranking relevance. `eval_set` is also rejected for
`pre_target` in this release because the public API does not yet accept
validation-set factor exposures to residualize validation targets consistently.

`neutralization="per_round_gradient"` projects objective gradients before each
boosting round:

```text
g_perp = g - F (F^T W F + lambda I)^-1 F^T W g
```

Hessians are unchanged. This mode is supported for regression, binary
classification, multiclass, and ranking. For multiclass, each class-gradient
column is projected independently against the same factor projector.

`neutralization="split_penalty"` includes per-round gradient projection and
subtracts a factor-load penalty from split gain:

```text
penalty = factor_penalty * || F_L^T update_L + F_R^T update_R ||^2 / max(row_count, 1)
gain_final = gain_after_existing_modes - penalty
```

For scalar leaves, `update_L` and `update_R` are the candidate scalar leaf
values before any final MorphBoost depth/iteration leaf scaling. For DRO
leaves, the scalar values use the DRO effective gradients. For MorphBoost, the
order is: project gradients, compute standard/DRO gradient gain, blend
MorphBoost information score, subtract factor penalty, then apply MorphBoost
leaf scaling when storing leaves. `split_penalty` performs additional
factor-exposure work during split search and should be treated as the slowest
neutralization mode until production-scale benchmarks justify stronger claims.

Compatibility:

| Feature | pre_target | per_round_gradient | split_penalty |
| --- | --- | --- | --- |
| `GBMRegressor` | supported | supported | supported |
| `GBMClassifier` | rejected | supported | supported |
| `GBMRanker` | rejected | supported | supported |
| `training_mode="morph"` | supported | supported | supported |
| `leaf_solver="dro"` | supported | supported | supported |
| `leaf_model="linear"` | supported | supported | rejected |
| warm start | supported | supported | supported |

Exposure matrices are not persisted in the estimator or artifact. As of
v0.7.1, neutralized warm-start and `init_model` continuation are supported
across all three modes provided the caller supplies the same
`factor_exposures` matrix used for the initial fit; `neutralization`,
`factor_neutralization_lambda`, and (for `split_penalty`) `factor_penalty`
must match the persisted contract — mismatches raise a clear "does not
match" error.

### Piecewise-Linear Leaves

Set `leaf_model="linear"` on any estimator to replace scalar leaves with small
closed-form linear models (`f_s(x) = b_s + Σ α_j x_j`). Weights are solved via
ridge regression `α* = -(XᵀHX + λI)⁻¹ Xᵀg` regularised by `lambda_l2`. This
typically converges in fewer rounds on data with linear within-node residual
structure (e.g. California Housing), at a 2–8× per-round training overhead.

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    leaf_model="linear",
    lambda_l2=0.01,    # recommended >= 0.01 with linear leaves
    seed=7,
)
model.fit(X_train, y_train)
```

`leaf_model="linear"` works with `GBMClassifier` and `GBMRanker`, and composes
with `training_mode="morph"`. As of v0.7.1, SHAP works on `leaf_model="linear"`
artifacts as a best-effort interventional decomposition (exact additivity is
relaxed for continuous-feature PL artifacts; see
[docs/limitations.md](docs/limitations.md)).

### Time-Aware Validation

```python
from alloygbm import GBMRegressor, purged_time_series_splits, rmse

splits = purged_time_series_splits(time_index, n_splits=5, purge_gap=1, embargo=0)

for train_idx, test_idx in splits:
    model = GBMRegressor(deterministic=True, seed=7)
    model.fit(
        [rows[i] for i in train_idx],
        [targets[i] for i in train_idx],
    )
    score = rmse(
        [targets[i] for i in test_idx],
        model.predict([rows[i] for i in test_idx]),
    )
```

For panel data, use `purged_panel_splits(...)`.

### Model Persistence

```python
import pickle

# Pickle round-trip
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Native save/load
model.save_model("model.agbm")
loaded = GBMRegressor.load_model("model.agbm")

# Artifact export for deployment
artifact_bytes = model.artifact_bytes
```

## Feature Summary

### Estimators

- **`GBMRegressor`** -- squared-error regression with dataset-aware `training_policy`
- **`GBMClassifier`** -- binary classification with log-loss objective, `predict_proba`, sklearn `ClassifierMixin`
- **`GBMRanker`** -- learning-to-rank with 5 objectives: `rank:pairwise`, `rank:ndcg`, `rank:xendcg`, `queryrmse`, `yetirank`
- **`MultiLabelGBMRanker`** -- multi-output ranking: `y` shaped `(n_rows, n_labels)`, `predict` returns the same shape, per-label `ranking_objective` lists supported. As of v0.10.1 also supports `multi_label_mode="joint"` for shared-tree training across all K labels via `engine::joint::fit_joint_multi_output` (default `"independent"` preserves the K-per-label `GBMRanker` fallback). v0.10.2 expanded joint-mode kwargs to include `tree_growth="leaf"` + `max_leaves`, `interaction_constraints`, `min_split_gain`, `row_subsample`, and `col_subsample`. v0.10.3 wires native-categorical splits (`categorical_feature_indices` + `max_cat_threshold`) through the joint Python bridge, adds `boosting_mode="goss"` and `boosting_mode="dart"` to the joint trainer, and supports `warm_start=True` + `init_model=...` on the joint path. v0.10.4 adds MorphBoost to the joint trainer (`training_mode="morph"` + the full `morph_*` / `lr_schedule` surface, with EMA warm-resume via the `MorphMetadata` artifact section). v0.10.5 adds joint DRO leaves (`leaf_solver="dro"` + `dro_radius` / `dro_metric`)
- All estimators are sklearn-compatible (`get_params`, `set_params`, `score`, pipeline integration)

### Training Features

- NaN/missing value support with learned split direction
- Sample weights via `fit(..., sample_weight=...)`
- Monotone constraints via `monotone_constraints`
- Feature importance weighting via `feature_weights`
- Leaf-wise (best-first) tree growth via `tree_growth="leaf"`
- Warm-starting / incremental training via `warm_start=True`
- Up to 65,535 bins per feature (`continuous_binning_max_bins`)
- Multiple categorical column support via `categorical_feature_indices`
- Early stopping with `best_iteration_`, `best_score_`, `evals_result_`
- Objective-aware training metric tracking (RMSE, log-loss, accuracy, NDCG)
- Adaptive split criterion via `training_mode="morph"` ([MorphBoost](https://arxiv.org/pdf/2511.13234))
- Per-iteration learning-rate schedules: `lr_schedule="constant"` (default) or `"warmup_cosine"`
- DRO-style robust scalar leaves via `leaf_solver="dro"` (closed-form gradient-uncertainty penalty)
- GOSS (gradient-based one-side sampling, LightGBM-style) via `boosting_mode="goss"` + `goss_top_rate` / `goss_other_rate` on regression, binary classification, and ranking. As of v0.10.1, GOSS is also supported on **multiclass classification** (K ≥ 3 classes) — per-row score `s_i = Σₖ |g_{i,k}|` (LightGBM convention) drives a shared sampling mask across all K class gradient buffers. Default `boosting_mode="standard"` is byte-identical to v0.7.5.
- DART (Dropouts meet MART) via `boosting_mode="dart"` + `dart_drop_rate` / `dart_max_drop` / `dart_normalize_type` (`"tree"` or `"forest"`) / `dart_sample_type` (`"uniform"` or `"weighted"`) on regression, binary classification, and ranking. Per-stump weights ride in a new `DartTreeWeights` artifact section emitted only when at least one stump diverges from `tree_weight = 1.0`, so Standard / GOSS artifacts stay byte-identical to v0.8.0. **DART + `warm_start` continuation** is supported (v0.10.0+) — pass a fitted DART model via `fit(..., init_model=prior_model)` to add more rounds on top. As of v0.10.1, DART is also supported on **multiclass classification** (K ≥ 3 classes) including warm-start. v0.10.2 lifts the `tree_growth="level"` restriction — multiclass DART now also works with `tree_growth="leaf"` + `max_leaves`.
- Piecewise-linear leaves via `leaf_model="linear"` (closed-form ridge solve, faster convergence on linear-trend data)
- Factor-neutral boosting via `neutralization` + fit-time `factor_exposures` (`pre_target`, `per_round_gradient`, `split_penalty`)
- LightGBM-compatible feature interaction constraints via `interaction_constraints=[[...]]` (up to 64 groups, level-wise and leaf-wise enforcement)
- Neutralized warm-start / `init_model` continuation with matching-exposures contract
- Per-round training diagnostics via `diagnostics_per_round_` (gradient stats, sampling counts, `neutralization_effectiveness`)

### Inference and Explanations

- Zero-copy numpy prediction from native artifacts
- TreeSHAP explanations via `shap_values(...)` (polynomial-time, no feature limit, also supports `leaf_model="linear"` as a best-effort interventional decomposition)
- Global feature importance via `feature_importances(...)`
- Artifact-backed prediction via `predict_from_artifact(...)`

### Validation Helpers

- `purged_time_series_splits(...)` -- leakage-aware time-series cross-validation
- `purged_panel_splits(...)` -- panel-data cross-validation

### Metrics

- Regression: `rmse`, `mae`, `r2_score`
- Classification: `accuracy`, `log_loss`
- Ranking: `ndcg`
- Finance: `pearson_correlation`, `rank_ic`, `hit_rate`, `icir`

## Benchmark Snapshot

The benchmark suite compares AlloyGBM against XGBoost, LightGBM, and CatBoost across regression, classification, and ranking tasks.

**Regression:**

- AlloyGBM is strongest on `panel_time_series`
- AlloyGBM is strong on `dow_jones_financial`
- AlloyGBM is competitive on `dense_numeric`, trails on `california_housing` and `bike_sharing`

**Classification:**

- AlloyGBM is competitive with established libraries on `breast_cancer` and `synthetic_classification`

**Ranking:**

- AlloyGBM competes on `synthetic_ranking` using its native LambdaMART implementation

Benchmark tooling and methodology live in [benchmarks/README.md](benchmarks/README.md).

## Current Limitations

- CPU-only runtime (GPU backend is architecturally planned but not implemented)
- `MultiLabelGBMRanker(multi_label_mode="joint")` supports built-in `squared_error` / `queryrmse` / `rank:*` objectives. v0.10.2 added leaf-wise growth + `max_leaves`, `interaction_constraints`, `min_split_gain`, `row_subsample`, and `col_subsample`. v0.10.3 added native-categorical Python wiring, joint GOSS, joint DART, and joint warm-start. v0.10.4 added joint MorphBoost (`training_mode="morph"` + the full morph kwargs). v0.10.5 added joint DRO leaves (`leaf_solver="dro"` + `dro_radius` / `dro_metric`). Still deferred: **v0.10.6** ships joint factor neutralization.
- `leaf_solver="dro"` is a robust scalar leaf update, not a full raw-distribution Wasserstein DRO guarantee

See [docs/limitations.md](docs/limitations.md) for the full list.

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Hosted Sphinx docs: [alloygbm.readthedocs.io](https://alloygbm.readthedocs.io/en/latest/)
- Runnable examples: [examples/](examples/) (8 end-to-end scripts)
- Benchmark guide: [benchmarks/README.md](benchmarks/README.md)
- Current roadmap: [docs/roadmap/current.md](docs/roadmap/current.md)
- Current limitations: [docs/limitations.md](docs/limitations.md)
- Archive: [docs/archive/README.md](docs/archive/README.md)

## Contributing

- [Contributing guide](CONTRIBUTING.md) (dev setup, coding standards, test commands)
- [Security policy](SECURITY.md) (private vulnerability reporting)
- [Release operating manual](docs/reference/release_checklist.md)

## License

MIT. See [LICENSE](LICENSE).
