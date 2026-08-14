# GBMClassifier

`GBMClassifier` is the binary classification estimator in AlloyGBM.

## Overview

`GBMClassifier` is a classification estimator shell over AlloyGBM's shared
native training core. Binary and multiclass targets are supported. When
scikit-learn is available, `ClassifierMixin` precedes the shared estimator core
in the MRO, so classifier tags, scoring, cloning, and pipeline behavior do not
inherit regressor semantics.

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
- `neutralization="per_round_gradient"` or `neutralization="split_penalty"` with
  `fit(..., factor_exposures=F)` for training-time factor/gradient
  neutralization. `neutralization="pre_target"` is rejected for classifiers
  because target residualization is not well-defined for class labels.
  `factor_exposure_transform="center"` / `"standardize"` may be used to
  preprocess exposure columns before projection; active `split_penalty`
  defaults to effective `"standardize"` preprocessing. See
  [GBMRegressor — Factor-Neutral Boosting](gbmregressor.md#factor-neutral-boosting).
- `leaf_model="linear"` for piecewise-linear leaves (see
  [GBMRegressor — Piecewise-Linear Leaves](gbmregressor.md#piecewise-linear-leaves)).
  Multi-class softmax fits each per-class tree sequence with linear leaves
  independently. Pair with `lambda_l2 >= 0.01` for weight stability.
- `training_mode="morph"` and the rest of the MorphBoost / LR-schedule parameters
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
- `diagnostics_per_round_` is populated after `fit()` with per-round gradient
  statistics, sampling counts, and (when factor neutralization is active)
  the `neutralization_effectiveness` score.
- `boosting_mode="goss"` with `goss_top_rate` / `goss_other_rate` for
  LightGBM-style gradient-based one-side sampling, and
  `boosting_mode="dart"` with the DART parameters, are supported on binary
  and multi-class targets (see [GBMRegressor — Boosting Mode](gbmregressor.md#boosting-mode)).
  For multi-class DART, `dart_max_drop` applies to the actual class-tree pool,
  so a four-class round advances the dropout pressure by four trees.

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

### `fit(X, y, *, sample_weight=None, eval_set=None, factor_exposures=None, ...)`

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

In addition to the shared estimator post-fit attributes:

- `classes_: np.ndarray` -- sorted labels in their original numeric, boolean,
  or string dtype
- `n_classes_: int` -- number of fitted classes

## sklearn Compatibility

When scikit-learn is installed, `GBMClassifier`:

- is certified against every applicable estimator check in scikit-learn 1.8
  and 1.9
- works with `cross_val_score`, `GridSearchCV`, `Pipeline`
- implements `__sklearn_tags__` and `_more_tags`
- `score(X, y)` returns accuracy (the sklearn classifier convention)
- returns one-dimensional NumPy label arrays from `predict(...)`

Sparse matrices are not a supported training format and raise an explicit
error. Known constructor and `set_params(...)` values are validated at the
start of `fit(...)`, matching scikit-learn cloning and model-selection rules.

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
