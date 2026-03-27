# Quickstart

## Basic Regression

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

Use `eval_set` whenever you enable `early_stopping_rounds`.

## What The Model Stores

After `fit(...)`, the regressor keeps a serialized native model artifact and a
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

## Continuous Features

If your inputs are continuous floats, AlloyGBM will quantize them inside the
native Rust training path. Supported binning strategies are:

- `linear`
- `rank`
- `quantile`

The default strategy is `linear`.

## Dense Array-Like Inputs

The Python bridge has optimized paths for array-like inputs that expose
`to_numpy`, `to_list`, or `tolist`. You do not need to manually convert every
input to nested Python lists.

## Next Step

If your data is time-indexed or panel-like, continue to
[Time-Aware Validation](validation.md) before you benchmark results.
