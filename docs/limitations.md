# AlloyGBM Current Limitations

Last updated for v0.2.0.

## Remaining Limitations

### 1. Binary Classification Only

`GBMClassifier` supports binary classification (2 classes). Multi-class
classification (softmax, one-vs-rest) is not yet implemented.

### 2. CPU-Only Runtime

The `BackendOps` trait is designed for hardware abstraction, but only
`CpuBackend` exists. GPU/accelerator support is architecturally planned but
not implemented.

### 3. No Custom Objective / Custom Metric Callbacks

The `ObjectiveOps` trait is not exposed to Python. Users cannot define custom
loss functions or custom evaluation metrics from Python.

### 4. No Interaction Constraints

There is no way to constrain which features can interact within the same tree.

### 5. No Dart / GOSS Boosting Modes

Only standard gradient boosting is supported. Dart (dropout) and GOSS
(gradient-based one-side sampling) modes are not available.

### 6. No Native Categorical Splits

Categorical features are handled via target encoding (with optional time-aware
leakage prevention). Native histogram-based categorical splits (like LightGBM's
optimal split) are not implemented.

### 7. No Multi-Label Ranking

`GBMRanker` supports single-label relevance only.

## Resolved In 0.2.0 (Previously Limitations)

The following were limitations in `0.1.x` and have been addressed:

- Regression-only (now: classification + ranking)
- Single categorical column only (now: multiple via `categorical_feature_indices`)
- Limited configurability (now: `min_split_gain`, monotone constraints, feature weights, `max_leaves`, leaf-wise growth)
- No NaN support (now: native NaN handling)
- No model persistence (now: pickle, save/load, artifact export)
- No sklearn compatibility (now: `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- No sample weight / group ID from Python (now: fully supported)
- Feature names auto-generated only (now: captured from DataFrames)
- SHAP limited to 20 features (now: TreeSHAP with no practical limit)
- Only RMSE tracked during training (now: objective-aware metric tracking)
- No warm-starting (now: `warm_start=True`)
- Level-wise growth only (now: leaf-wise available)
- Bins capped at 256 (now: up to 65,535)
- No histogram reuse (now: buffer reuse across rounds)
