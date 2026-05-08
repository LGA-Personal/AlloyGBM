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
- `leaf_model="linear"` for piecewise-linear leaves (see
  [GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves)).
  Pair with `lambda_l2 >= 0.01` for weight stability.
- `training_mode="morph"` and the MorphBoost / LR-schedule parameters
  (`morph_rate`, `evolution_pressure`, `morph_warmup_iters`, `info_score_weight`,
  `depth_penalty_base`, `balance_penalty`, `lr_schedule`, `lr_warmup_frac`).
  `leaf_model="linear"` and `training_mode="morph"` can be combined.
  See [MorphBoost](morphboost.md) for the full reference.

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

### `fit(X, y, *, group, eval_set=None, eval_group=None, ...)`

Trains the ranker.

- `X` -- feature matrix (n_samples, n_features)
- `y` -- relevance labels (higher = more relevant). Can be graded (e.g. 0-4) or
  binary.
- `group` -- per-row query group identifiers. **Required.** All rows with the
  same group ID belong to the same query.
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
- Single-label relevance only (no multi-label)
- Group identifiers must be unsigned integers
