# Time-Aware Validation

AlloyGBM includes leakage-aware split helpers for time series and panel data.

## Why Use Them

Random train/test splits are often wrong for:

- forecasting
- panel regression
- finance datasets with ordered timestamps

If the same time bucket appears in both training and test data, benchmark scores
can be misleading.

## Time Series Splits

Use `purged_time_series_splits(...)` when you have a single time axis:

```python
from alloygbm import purged_time_series_splits

time_index = [0, 0, 1, 1, 2, 2, 3, 3]
splits = purged_time_series_splits(
    time_index,
    n_splits=4,
    purge_gap=0,
    embargo=0,
)
```

Parameters:

- `n_splits`
  - Number of contiguous folds.
- `purge_gap`
  - Number of periods removed immediately before each test window.
- `embargo`
  - Number of periods removed immediately after each test window.

## Panel Splits

Use `purged_panel_splits(...)` when you have both a time index and a group id:

```python
from alloygbm import purged_panel_splits

time_index = [0, 0, 1, 1, 2, 2]
group_index = ["A", "B", "A", "B", "A", "B"]

splits = purged_panel_splits(
    time_index,
    group_index,
    n_splits=3,
    purge_gap=0,
    embargo=0,
)
```

Panel behavior is still time-bucketed across all groups, which is usually the
right default when leakage is primarily temporal.

## Using Splits With AlloyGBM

```python
from alloygbm import GBMRegressor, purged_time_series_splits, rmse

rows = [[float(i), float(i % 2)] for i in range(20)]
targets = [float(i) * 0.1 for i in range(20)]
time_index = [i // 2 for i in range(20)]

scores = []
for train_idx, test_idx in purged_time_series_splits(time_index, n_splits=5):
    model = GBMRegressor(deterministic=True, seed=7)
    model.fit([rows[i] for i in train_idx], [targets[i] for i in train_idx])
    preds = model.predict([rows[i] for i in test_idx])
    scores.append(rmse([targets[i] for i in test_idx], preds))
```

## Explicit Validation With `fit(...)`

Use `eval_set` when you want AlloyGBM to track validation metrics during
training or to enable early stopping:

```python
from alloygbm import GBMRegressor

model = GBMRegressor(
    n_estimators=1200,
    early_stopping_rounds=50,
    min_validation_improvement=1e-4,
    deterministic=True,
    seed=7,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
)

print(model.best_iteration_)
print(model.best_score_)
print(model.n_estimators_)
```

Rules:

- `early_stopping_rounds` requires `eval_set`.
- `eval_time_index` requires `eval_set`.
- Validation early stopping monitors the objective-appropriate metric:
  - RMSE for `GBMRegressor`
  - Log-loss for `GBMClassifier`
  - NDCG for `GBMRanker`

## When To Pass `time_index` Into `fit(...)`

Pass `time_index=` to `fit(...)` when you are using:

- `categorical_feature_index` or `categorical_feature_indices`
- `categorical_time_aware=True`

That combination enables time-aware categorical handling during training.

If you also pass `eval_set` with `categorical_time_aware=True`, you must pass
`eval_time_index=` for the validation rows as well.
