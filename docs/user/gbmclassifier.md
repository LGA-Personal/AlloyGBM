# GBMClassifier

`GBMClassifier` is the binary classification estimator in AlloyGBM.

## Overview

`GBMClassifier` extends `GBMRegressor` with a binary cross-entropy (log-loss)
objective. Predictions are probabilities obtained via sigmoid transform. When
sklearn is available, `GBMClassifier` inherits `ClassifierMixin` for full
pipeline compatibility.

## Quick Example

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

labels = model.predict(X_test)
probas = model.predict_proba(X_test)

print("accuracy:", accuracy(y_test, labels))
print("log_loss:", log_loss(y_test, probas[:, 1]))
```

## Parameters

All parameters from `GBMRegressor` are accepted, including:
- `leaf_solver="dro"` for robust scalar leaves (see
  [GBMRegressor — DRO Leaf Solver](gbmregressor.md#dro-leaf-solver)). It works
  for binary and multi-class classification and requires `leaf_model="constant"`.
- `leaf_model="linear"` for piecewise-linear leaves (see
  [GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves)).
  Multi-class softmax fits each per-class tree sequence with linear leaves
  independently. Pair with `lambda_l2 >= 0.01` for weight stability.
- `training_mode="morph"` and the rest of the MorphBoost / LR-schedule parameters
  (`morph_rate`, `evolution_pressure`, `morph_warmup_iters`, `info_score_weight`,
  `depth_penalty_base`, `balance_penalty`, `lr_schedule`, `lr_warmup_frac`).
  `leaf_model="linear"` and `training_mode="morph"` can be combined.
  See [MorphBoost](morphboost.md) for the full reference.

The objective is always binary cross-entropy and is not configurable.

```python
# MorphBoost on binary classification
model = GBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    training_mode="morph",
    seed=7,
)
```

## Target Requirements

- `y` must contain only values in `{0, 1}` (or `{0.0, 1.0}`)
- Both classes must be present in the training targets

## Methods

### `fit(X, y, *, sample_weight=None, eval_set=None, ...)`

Trains the classifier. Accepts the same keyword arguments as
`GBMRegressor.fit()`. Returns `self`.

### `predict(X) -> list[int]`

Returns class labels (0 or 1) by thresholding probabilities at 0.5.

### `predict_proba(X) -> np.ndarray`

Returns an array of shape `(n_samples, 2)` with columns `[P(y=0), P(y=1)]`.
This is the standard sklearn classifier probability interface.

### `predict_log_proba(X) -> np.ndarray`

Returns log-probabilities of shape `(n_samples, 2)`.

## Post-Fit Attributes

In addition to the standard `GBMRegressor` post-fit attributes:

- `classes_: list[int]` -- always `[0, 1]`
- `n_classes_: int` -- always `2`

## sklearn Compatibility

When sklearn is installed, `GBMClassifier`:

- inherits from `ClassifierMixin`
- works with `cross_val_score`, `GridSearchCV`, `Pipeline`
- implements `__sklearn_tags__` and `_more_tags`
- `score(X, y)` returns accuracy (the sklearn classifier convention)

## Early Stopping

Early stopping monitors log-loss on the validation set when `eval_set` is
provided:

```python
model = GBMClassifier(
    n_estimators=2000,
    early_stopping_rounds=50,
    deterministic=True,
    seed=7,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
print(model.best_iteration_)
print(model.best_score_)
```

## Current Scope

- Binary cross-entropy and multi-class softmax objectives are supported.
- No `scale_pos_weight` parameter (use `sample_weight` for class imbalance).
