# AlloyGBM

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

Wheel targets for `0.3.1`:

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

### Inference and Explanations

- Zero-copy numpy prediction from native artifacts
- TreeSHAP explanations via `shap_values(...)` (polynomial-time, no feature limit)
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

- Binary classification only (no multi-class yet)
- CPU-only runtime (GPU backend is architecturally planned but not implemented)
- No custom objective / custom metric callbacks from Python
- No interaction constraints
- No dart/goss boosting modes

## Documentation

- Docs index: [docs/README.md](docs/README.md)
- Benchmark guide: [benchmarks/README.md](benchmarks/README.md)
- Current roadmap: [docs/roadmap/current.md](docs/roadmap/current.md)
- Archive: [docs/archive/README.md](docs/archive/README.md)

## License

MIT. See [LICENSE](LICENSE).
