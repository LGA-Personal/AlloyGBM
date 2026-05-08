# Quickstart

## Regression

```python
from alloygbm import GBMRegressor, rmse

X_train = [
    [0.0, 1.0],
    [1.0, 0.0],
    [2.0, 1.0],
    [3.0, 0.0],
]
y_train = [0.2, 0.9, 1.8, 2.7]

X_test = [
    [1.5, 1.0],
    [2.5, 0.0],
]
y_test = [1.3, 2.3]

model = GBMRegressor(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1200,
    training_policy="auto",
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
print("rmse:", rmse(y_test, predictions))
```

## Binary Classification

```python
from alloygbm import GBMClassifier, accuracy, log_loss

model = GBMClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train)  # y must be {0, 1}

labels = model.predict(X_test)            # [0, 1, 1, 0, ...]
probas = model.predict_proba(X_test)      # shape (n_samples, 2)

print("accuracy:", accuracy(y_test, labels))
print("log_loss:", log_loss(y_test, probas[:, 1]))
```

`GBMClassifier` uses binary cross-entropy loss internally and applies a sigmoid
transform to produce probabilities. It inherits sklearn's `ClassifierMixin` when
sklearn is available.

## Learning-to-Rank

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

`GBMRanker` requires `group` (query IDs) to be passed in `fit()`. The `group`
parameter accepts per-row group identifiers; data is sorted by group internally.

Supported ranking objectives: `rank:pairwise`, `rank:ndcg`, `rank:xendcg`,
`queryrmse`, `yetirank`.

## MorphBoost (Optional Adaptive Mode)

Any of the three estimators supports an opt-in MorphBoost training mode
that augments the standard gradient gain with an information-theoretic
term and EMA-driven gain shaping. See
[Kriuk (2025), *MorphBoost*](https://arxiv.org/pdf/2511.13234) for the
formulation.

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1200,
    training_mode="morph",   # opt in
    seed=7,
)
model.fit(X_train, y_train)
```

A learning-rate schedule (`lr_schedule="warmup_cosine"`) can also be
applied independently of `training_mode`, useful for very low-LR
high-`n_estimators` runs:

```python
model = GBMRegressor(
    learning_rate=0.01,
    n_estimators=5000,
    training_mode="morph",
    lr_schedule="warmup_cosine",
    lr_warmup_frac=0.1,
)
```

Full parameter reference: [MorphBoost](morphboost.md).

## Piecewise-Linear Leaves (Optional)

`leaf_model="linear"` replaces scalar leaves with closed-form linear models
solved per leaf via `α* = -(XᵀHX + λI)⁻¹ Xᵀg`. On data with linear
within-node residual structure this typically reaches the same loss in
fewer rounds.

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

`leaf_model="linear"` works on `GBMClassifier` and `GBMRanker` too, and
composes with `training_mode="morph"`. SHAP currently requires
`leaf_model="constant"`. Full reference:
[GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves).

## Validation And Early Stopping

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1200,
    early_stopping_rounds=50,
    min_validation_improvement=1e-4,
    min_data_in_leaf=32,
    lambda_l2=1.0,
    deterministic=True,
    seed=7,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
)

print("best_iteration_:", model.best_iteration_)
print("best_score_:", model.best_score_)
print("n_estimators_:", model.n_estimators_)
print("evals_result_ keys:", model.evals_result_.keys())
print("fit_timing_:", model.fit_timing_)
```

Use `eval_set` whenever you enable `early_stopping_rounds`. Early stopping
monitors the objective-appropriate metric (RMSE for regression, log-loss for
classification, NDCG for ranking).

## Model Persistence

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

# Artifact bytes for deployment
artifact_bytes = model.artifact_bytes
```

All three estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) support
pickle, `save_model` / `load_model`, and artifact byte export.

## What The Model Stores

After `fit(...)`, the estimator keeps a serialized native model artifact and a
native predictor handle. That means you can:

- call `predict(...)`
- call `shap_values(...)`
- call `feature_importances(...)`
- use `predict_from_artifact(...)` with serialized artifact bytes
- inspect fitted summaries through:
  - `best_iteration_`
  - `best_score_`
  - `n_estimators_`
  - `evals_result_`
  - `fit_timing_`

## NaN / Missing Values

AlloyGBM handles NaN values natively. You do not need to impute missing values
before training or prediction. The engine learns the optimal split direction for
missing values at each node.

## Dense Array-Like Inputs

The Python bridge has optimized paths for array-like inputs that expose
`to_numpy`, `to_list`, or `tolist`. You do not need to manually convert every
input to nested Python lists.

## Next Step

If your data is time-indexed or panel-like, continue to
[Time-Aware Validation](validation.md) before you benchmark results.
