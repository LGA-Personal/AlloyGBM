# AlloyGBM Comprehensive Limitation Analysis

## User-Identified Limitations (Confirmed & Detailed)

### 1. Regression Only -- No Classification or Ranking

This is deeply structural. The entire system is hardcoded to `SquaredErrorObjective`:
- `engine/src/lib.rs` defines an `ObjectiveOps` trait, but the *only implementation* is `SquaredErrorObjective` (MSE loss, gradient = residual, hessian = 1.0/weight)
- The Python binding functions are literally named `train_regression_artifact`, `train_regression_artifact_dense`, etc.
- The Python class is `GBMRegressor` -- there is no `GBMClassifier` or `GBMRanker`
- All evaluation metrics in `evaluation.py` are regression-oriented (RMSE, MAE, R2) or financial (rank_ic, hit_rate, ICIR) -- no log-loss, AUC, accuracy, NDCG, MAP
- The model artifact format has no concept of `num_class`, output transformation (sigmoid/softmax), or label encoding
- Predictions are raw floats with no post-transformation -- no probability calibration, no class label output

The `ObjectiveOps` trait *does* exist as an abstraction, so the engine is at least *designed* for extensibility here, but nothing beyond squared error has been built.

### 2. Limited to 1 Categorical Column

Confirmed exactly. The pipeline supports at most a single categorical feature via target encoding:
- `GBMRegressor.__init__` accepts `categorical_feature_index: int | None` -- singular
- `CategoricalTargetEncodingSpec` takes one `feature_index: usize` and one `values: Vec<String>`
- `fit_iterations_with_single_target_encoded_feature` in the engine literally has "single" in its name
- `CategoricalStatePayloadV1` does store `categorical_feature_indices: Vec<u32>` (a vector), suggesting multi-categorical was *envisioned*, but the training path only ever populates it with a single entry
- The categorical encoding is via target encoding only (with optional time-aware leakage prevention) -- no native categorical split support (like LightGBM's optimal histogram split)

### 3. Only Somewhat Configurable

The exposed parameters on `GBMRegressor` are:
- `learning_rate`, `max_depth`, `n_estimators`, `row_subsample`, `col_subsample`
- `early_stopping_rounds`, `min_validation_improvement`
- `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `min_child_hessian`
- `seed`, `deterministic`
- `continuous_binning_strategy` (linear/rank/quantile), `continuous_binning_max_bins`
- `categorical_*` params, `training_policy` (auto/manual)

What's missing vs. XGBoost/LightGBM:
- **No `max_leaves` / leaf-wise growth** -- tree growth is purely depth-first level-wise with `max_depth` only
- **No `min_split_gain` exposed** -- it exists internally in `IterationControls` and the auto policy sets it, but it's not a user-facing parameter
- **No monotone constraints** -- no way to constrain features to be monotonically increasing/decreasing
- **No interaction constraints** -- no way to limit which features can interact in the same tree
- **No feature importance weighting at split time**
- **No custom objective / custom metric callbacks** -- the `ObjectiveOps` trait isn't exposed to Python
- **No `max_bin` per feature** -- all features share the same 256-bin scheme
- **No dart/goss/other boosting modes** -- pure gradient boosting only
- **No `scale_pos_weight`** (moot without classification, but worth noting)

## Additional Identified Limitations

### 4. No Missing Value (NaN) Support

All features must be finite. Both training and prediction explicitly reject NaN/Inf values:
- `validate_dense_values_finite()` in `lib.rs` checks every cell
- `_validate_rows` in `regressor.py` casts to `float()` with no NaN handling
- The `BinnedMatrix` has no reserved "missing" bin
- The `ColumnarMatrixColumnView` has a `validity: Option<&[bool]>` bitmap suggesting missing-value support was *considered*, but it's never wired into the training or prediction paths

This is a significant practical limitation -- real-world tabular data almost always has missing values.

### 5. No Model Save/Load (Persistence)

The `GBMRegressor` has no `save_model()` / `load_model()` methods. The fitted state lives in memory as `_artifact_bytes` plus quantization metadata (`_continuous_feature_mins`, `_continuous_feature_maxs`, `_continuous_feature_sorted_values`, etc.). There's:
- No pickle/joblib support (no `__getstate__`/`__setstate__`)
- No `to_file()` / `from_file()` convenience
- `predict_from_artifact` exists as a static method accepting raw bytes, but there's no way to *get* the artifact bytes out programmatically nor save/restore the full estimator state including the quantization metadata needed for prediction on continuous features

### 6. No GPU/Accelerator Backend

The `Device` enum only has `Device::Cpu`. The crate structure (`backend_cpu`) clearly anticipates a future `backend_gpu`, but there's no implementation. The `BackendOps` trait is properly abstracted for this, but histogram building, split finding, and partitioning are CPU-only (with rayon parallelism).

### 7. Bins Capped at 256 (u8)

The `BinnedMatrix` uses `Vec<u8>` for bins, capping at 256 bins per feature. While adequate for many use cases, LightGBM defaults to 255 bins but supports up to 65535 for high-cardinality features. The `threshold_bin` is stored as `u16` in the split, but the actual bin storage is `u8`, so features with more than 256 distinct values lose resolution.

### 8. SHAP Exact Method Limited to 20 Split Features

`MAX_EXACT_SPLIT_FEATURES` is 20. The SHAP implementation uses exact Shapley values (exponential in the number of split features: 2^N subsets). If a model uses more than 20 distinct features in its splits, SHAP computation errors out. Real models on wide datasets will commonly exceed this. TreeSHAP's polynomial-time algorithm isn't implemented.

### 9. No Native sklearn Compatibility

While `GBMRegressor` has `get_params()` / `set_params()` / `fit()` / `predict()`, it doesn't inherit from `sklearn.base.BaseEstimator` or `RegressorMixin`. This means:
- No `sklearn.model_selection.cross_val_score` compatibility
- No pipeline integration
- `score()` method is missing
- `__sklearn_tags__` not implemented

### 10. No Sample Weight Support from Python

The engine's `TrainingDataset` has `sample_weights: Option<Vec<f32>>` and the `SquaredErrorObjective` handles them correctly. But the Python bridge *always* sets `sample_weights: None` -- the `fit()` method has no `sample_weight` parameter, so this Rust-side capability is completely inaccessible from Python.

### 11. No Group ID Support from Python (Ranking Prerequisite)

Similarly, `TrainingDataset` has `group_id: Option<Vec<u32>>`, but it's always `None` in the Python bridge. This would be needed for LambdaMART/ranking objectives.

### 12. Feature Names Are Auto-Generated

When the model artifact is serialized, feature names are auto-generated as `f0`, `f1`, ... (`to_artifact_bytes()` in engine). The user's real feature names from a DataFrame are not preserved through training. This affects SHAP explanations and feature importance output.

### 13. Only RMSE Tracked During Training

The `evals_result_` only tracks RMSE per round. There's no callback system, no custom evaluation metric during training, no way to track MAE or other metrics alongside.

### 14. No Warm-Starting / Incremental Training

There's no way to continue training from a previously fitted model. Each call to `fit()` starts fresh. No `init_model` / `keep_training_booster` equivalent.

### 15. Level-Wise Tree Growth Only

The engine grows trees level-by-level (depth-first BFS through `active_nodes`). There's no leaf-wise (best-first) growth option. Leaf-wise growth is often faster and more accurate for the same number of leaves (this is LightGBM's key innovation).

### 16. No Histogram Subtraction Trick for Root

The engine *does* implement the histogram subtraction trick for child nodes (`subtract_histogram_bundle`), which is good. But there's no global histogram caching between rounds -- histograms are rebuilt from scratch each boosting iteration.
