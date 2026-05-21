# GBMRanker

`GBMRanker` is the learning-to-rank estimator in AlloyGBM.

## Overview

`GBMRanker` extends `GBMRegressor` with ranking-specific objectives. All ranking
objectives require query group identifiers to be passed in `fit()`. Data is
sorted by group internally.

## Quick Example

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

## Ranking Objectives

- `"rank:pairwise"` -- Pairwise logistic loss (RankNet)
- `"rank:ndcg"` -- LambdaMART with NDCG weighting (default)
- `"rank:xendcg"` -- Cross-entropy approximation to NDCG
- `"queryrmse"` -- Query-grouped RMSE
- `"yetirank"` -- YetiRank (stochastic NDCG-weighted pairwise)

## Parameters

### Ranker-Specific

- `ranking_objective: str = "rank:ndcg"`
  - The ranking loss function. Must be one of the supported objectives above.

### Inherited

All other parameters are inherited from `GBMRegressor` (learning rate, depth,
regularization, etc.). This includes:
- `leaf_solver="dro"` for robust scalar leaves (see
  [GBMRegressor — DRO Leaf Solver](gbmregressor.md#dro-leaf-solver)). It
  requires `leaf_model="constant"`.
- `neutralization="per_round_gradient"` or `neutralization="split_penalty"` with
  `fit(..., factor_exposures=F)` for training-time factor/gradient
  neutralization. `neutralization="pre_target"` is rejected for rankers because
  target residualization is not well-defined for ranking relevance. See
  [GBMRegressor — Factor-Neutral Boosting](gbmregressor.md#factor-neutral-boosting).
- `leaf_model="linear"` for piecewise-linear leaves (see
  [GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves)).
  Pair with `lambda_l2 >= 0.01` for weight stability.
- `training_mode="morph"` and the MorphBoost / LR-schedule parameters
  (`morph_rate`, `evolution_pressure`, `morph_warmup_iters`, `info_score_weight`,
  `depth_penalty_base`, `balance_penalty`, `lr_schedule`, `lr_warmup_frac`).
  `leaf_model="linear"` and `training_mode="morph"` can be combined.
  See [MorphBoost](morphboost.md) for the full reference.
- `interaction_constraints=[[...]]` for LightGBM-compatible interaction
  constraints across both level-wise and leaf-wise tree builders (see
  [GBMRegressor — Constraints](gbmregressor.md#constraints)).
- `warm_start=True` / `init_model` for incremental training. Neutralized
  warm-start is supported when the caller resupplies the same
  `factor_exposures` matrix used for the initial fit.
- `boosting_mode="goss"` with `goss_top_rate` / `goss_other_rate` for
  LightGBM-style gradient-based one-side sampling, or
  `boosting_mode="dart"` with `dart_drop_rate` / `dart_max_drop` /
  `dart_normalize_type` / `dart_sample_type` for Dropouts-meet-MART
  (see [GBMRegressor — Boosting Mode](gbmregressor.md#boosting-mode)).

```python
# MorphBoost on ranking
model = GBMRanker(
    ranking_objective="rank:ndcg",
    n_estimators=300,
    learning_rate=0.05,
    training_mode="morph",
    seed=7,
)
model.fit(X_train, y_train, group=query_ids_train)
```

## Methods

### `fit(X, y, *, group, eval_set=None, eval_group=None, factor_exposures=None, ...)`

Trains the ranker.

- `X` -- feature matrix (n_samples, n_features)
- `y` -- relevance labels (higher = more relevant). Can be graded (e.g. 0-4) or
  binary.
- `group` -- per-row query group identifiers. **Required.** All rows with the
  same group ID belong to the same query.
- `factor_exposures` -- optional row-aligned factor matrix required when
  neutralization is active. The ranker applies the same internal group sorting
  to factor rows as it applies to `X`, `y`, and `group`.
- `eval_set` -- optional validation data `(X_val, y_val)`
- `eval_group` -- query group IDs for the validation set. Required when
  `eval_set` is provided.

### `predict(X) -> list[float]`

Returns raw relevance scores (higher = more relevant). No post-transform is
applied for ranking objectives. Use these scores to rank documents within a
query.

## Evaluation

Use `ndcg(...)` for ranking evaluation:

```python
from alloygbm import ndcg

# Full NDCG
score = ndcg(y_test, predictions, group=query_ids_test)

# NDCG@k (top-k positions only)
score_at_5 = ndcg(y_test, predictions, group=query_ids_test, k=5)
score_at_10 = ndcg(y_test, predictions, group=query_ids_test, k=10)
```

## Early Stopping

Early stopping monitors NDCG on the validation set when `eval_set` is provided:

```python
model = GBMRanker(
    ranking_objective="rank:ndcg",
    n_estimators=2000,
    early_stopping_rounds=50,
    deterministic=True,
    seed=7,
)
model.fit(
    X_train, y_train,
    group=query_ids_train,
    eval_set=(X_valid, y_valid),
    eval_group=query_ids_valid,
)
print(model.best_iteration_)
```

## Group Format

The `group` parameter accepts per-row group identifiers (e.g. query IDs). This
is different from LightGBM's group-size format. AlloyGBM sorts by group
internally, so rows do not need to be pre-sorted.

```python
# Per-row group IDs (AlloyGBM format)
group = [0, 0, 0, 1, 1, 2, 2, 2, 2]
```

## Current Scope

- 5 ranking objectives implemented natively in Rust
- Single-label per `GBMRanker`. For multi-output ranking
  (`y` shaped `(n_rows, n_labels)`, `predict` returns the same shape) use
  `MultiLabelGBMRanker`, which trains one independent `GBMRanker` per label
  by default (sharing `group` / `factor_exposures` / kwargs and supporting
  per-label `ranking_objective` lists), and can opt into joint shared-tree
  training with `multi_label_mode="joint"` (v0.10.1+). See the
  *Multi-Output Ranking* section below for both modes.
- Group identifiers must be unsigned integers

## Multi-Output Ranking — `MultiLabelGBMRanker`

```python
from alloygbm import MultiLabelGBMRanker, ndcg
import numpy as np

# y shaped (n_rows, n_labels), one column per label
y_train = np.column_stack([clicks_train, conversions_train])

model = MultiLabelGBMRanker(
    ranking_objective=["rank:ndcg", "rank:pairwise"],
    learning_rate=0.05,
    n_estimators=300,
    seed=7,
)
model.fit(X_train, y_train, group=query_ids_train)

scores = model.predict(X_test)  # shape (n_rows, n_labels)
```

`ranking_objective` may be a single string (broadcast to every label) or a
list with one objective per label. `save_model` / `load_model` round-trip
the wrapper, and `eval_set` y-columns are sliced per label so early
stopping and custom eval metrics work end-to-end.

### Training modes

The `multi_label_mode` constructor argument selects the shared-tree
strategy:

- **`"independent"`** (default, ≥ v0.7.1) — K separate `GBMRanker`
  instances share `group` and `factor_exposures`. Every per-label
  feature (warm-start, neutralization, MorphBoost, PL leaves, DRO,
  interaction constraints, custom eval metrics) flows through unchanged.
- **`"joint"`** (≥ v0.10.1) — single shared tree ensemble with per-leaf
  K-output values, trained by `engine::joint::fit_joint_multi_output`.
  Splits are chosen using the per-output sum-of-gains
  `Σₖ (G_L_k²/(H_L_k+λ) + G_R_k²/(H_R_k+λ) − G_k²/(H_k+λ))`. Joint
  mode is more efficient when labels share signal. As of v0.10.3,
  joint mode supports `tree_growth="leaf"` + `max_leaves`,
  `interaction_constraints`, `min_split_gain`, `row_subsample`,
  `col_subsample`, native-categorical splits
  (`categorical_feature_indices` + `max_cat_threshold`),
  `boosting_mode="goss"` / `boosting_mode="dart"`, `warm_start=True`
  + `init_model=...`, and the built-in `squared_error` /
  `queryrmse` / `rank:pairwise` / `rank:ndcg` / `rank:xendcg`
  objectives. **Still deferred to v0.10.4** (joint MorphBoost,
  neutralization, DRO) — see [../limitations.md](../limitations.md).

```python
model = MultiLabelGBMRanker(
    multi_label_mode="joint",
    ranking_objective="rank:ndcg",
    n_estimators=300,
    learning_rate=0.05,
    seed=7,
)
model.fit(X_train, y_train, group=query_ids_train)
scores = model.predict(X_test)  # shape (n_rows, n_labels)
```
