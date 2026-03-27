# AlloyGBM

AlloyGBM is a Rust-first gradient boosting library for structured regression, with a Python API focused on fast native execution, deterministic training, and time-aware tabular workflows.

It is currently strongest on panel and finance-style regression problems where leakage-aware validation and practical iteration speed matter. It also includes native artifact prediction, SHAP explanations, and purged time-series split helpers in the Python package.

## When To Use AlloyGBM

AlloyGBM is a good fit when you want:

- a native-backed gradient boosting regressor with a small Python API surface
- deterministic CPU training and inference
- time-aware validation helpers for forecasting or panel-style workflows
- native prediction from serialized artifacts
- SHAP-based local explanations and global feature importances

If you need the broadest possible objective support, classification, ranking, multiple categorical columns, or the strongest out-of-the-box results on generic tabular benchmarks, you should still expect XGBoost, LightGBM, or CatBoost to be stronger today.

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

AlloyGBM currently targets Python `3.10+` and uses a native Rust extension module.

Initial `0.1.0` packaging policy:

- tested directly on macOS Apple Silicon
- planned wheel targets: macOS `arm64` and Linux `x86_64`
- Windows support is deferred until after `0.1.0`
- source distribution remains the fallback for unsupported environments

## Minimal Example

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
print(rmse(y_test, predictions))
```

## Time-Aware Validation Example

```python
from alloygbm import GBMRegressor, purged_time_series_splits, rmse

rows = [
    [0.1, 1.0],
    [0.2, 1.1],
    [0.4, 0.9],
    [0.6, 1.2],
    [0.8, 1.3],
    [1.0, 1.4],
]
targets = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
time_index = [0, 0, 1, 1, 2, 2]

splits = purged_time_series_splits(
    time_index,
    n_splits=3,
    purge_gap=0,
    embargo=0,
)

fold_scores = []
for train_idx, test_idx in splits:
    model = GBMRegressor(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=400,
        deterministic=True,
        seed=7,
    )
    X_train = [rows[i] for i in train_idx]
    y_train = [targets[i] for i in train_idx]
    X_test = [rows[i] for i in test_idx]
    y_test = [targets[i] for i in test_idx]

    model.fit(X_train, y_train)
    fold_scores.append(rmse(y_test, model.predict(X_test)))

print(fold_scores)
```

For panel data, use `purged_panel_splits(...)`.

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

print(model.best_iteration_)
print(model.best_score_)
print(model.n_estimators_)
print(model.evals_result_)
print(model.fit_timing_)
```

`early_stopping_rounds` is explicit-only: pass `eval_set=(X_valid, y_valid)` when
you enable it.

## Feature Summary

- Native Rust-backed training and prediction from Python
- `GBMRegressor` with deterministic training controls and dataset-aware `training_policy`
- Explicit validation support via `fit(..., eval_set=..., eval_time_index=...)`
- Early stopping with fitted summaries: `best_iteration_`, `best_score_`, `n_estimators_`, `evals_result_`
- Leaf and split controls: `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `min_child_hessian`
- Continuous-feature binning strategies: `linear`, `rank`, `quantile`
- Optional single-column categorical encoding path
- Artifact-backed prediction via `predict_from_artifact(...)`
- SHAP row explanations via `shap_values(...)`
- SHAP global feature importance via `feature_importances(...)`
- Time-aware validation helpers:
  - `purged_time_series_splits(...)`
  - `purged_panel_splits(...)`
- Metric helpers:
  - `rmse`, `mae`, `r2_score`
  - `pearson_correlation`, `rank_ic`, `hit_rate`, `icir`

## Benchmark Snapshot

The current public benchmark suite compares AlloyGBM against XGBoost, LightGBM, and CatBoost on synthetic and real regression datasets.

Current headline results from the expanded suite:

- AlloyGBM is best on the `panel_time_series` benchmark across the tested profiles.
- AlloyGBM is strong on `dow_jones_financial`, with its best showing under the deeper low-learning-rate profile.
- AlloyGBM is competitive on `dense_numeric`, but still trails XGBoost and CatBoost on RMSE.
- AlloyGBM currently lags all three libraries on `california_housing` and `bike_sharing`.
- In the latest recorded public-suite refresh, AlloyGBM was also the fastest trainer on most scenario/profile rows.

The latest recorded benchmark refresh after moving dense continuous-feature
preprocessing into Rust did not show an RMSE collapse for AlloyGBM, and fit time
on the public suite improved materially versus the previous stored comparison.
The benchmark runner now also reports stage timings for:

- Python input adaptation
- native bridge preparation
- native training
- total fit time
- predict time

The honest short version is:

- strong on `panel_time_series`
- strong on `dow_jones_financial`
- weaker on `california_housing` and `bike_sharing`

Benchmark tooling and methodology live in [benchmarks/README.md](benchmarks/README.md).

## Current Limitations

- Regression-only. Classification and ranking are not implemented yet.
- CPU-only runtime today.
- Single categorical feature support only.
- Best performance is still concentrated in time-aware and finance-style structured regression, not broad tabular dominance.
- The API is intentionally small and still evolving toward a more complete `0.x` user-facing surface.

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Benchmark guide: [benchmarks/README.md](benchmarks/README.md)
- Current roadmap: [docs/roadmap/current.md](docs/roadmap/current.md)
- Archive: [docs/archive/README.md](docs/archive/README.md)

## License

MIT. See [LICENSE](LICENSE).
