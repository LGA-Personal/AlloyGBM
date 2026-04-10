# Changelog

## 0.2.0

Major capability expansion from the regression-only `0.1.x` series.

### New Estimators

- **`GBMClassifier`** -- binary classification with binary cross-entropy (log-loss) objective, `predict_proba`, `predict_log_proba`, sklearn `ClassifierMixin` integration
- **`GBMRanker`** -- learning-to-rank with 5 objectives:
  - `rank:pairwise` (RankNet)
  - `rank:ndcg` (LambdaMART)
  - `rank:xendcg` (cross-entropy NDCG approximation)
  - `queryrmse` (query-grouped RMSE)
  - `yetirank` (stochastic NDCG-weighted pairwise)

### Core Improvements

- **NaN / missing value support** across all crates -- training and prediction handle NaN natively with learned split directions
- **Sample weight support** via `fit(..., sample_weight=...)`
- **Group ID support** via `fit(..., group=...)` for ranking objectives
- **Model persistence** -- pickle round-trip, `save_model(path)` / `load_model(path)`, and `artifact_bytes` property for artifact export
- **Feature name capture** from pandas DataFrames and other named inputs
- **sklearn compatibility** -- `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`, `get_params`, `set_params`, `score`, pipeline/cross-validation support
- **`min_split_gain` exposed** as a user-facing parameter

### Training Enhancements

- **Leaf-wise (best-first) tree growth** via `tree_growth="leaf"` -- similar to LightGBM's growth strategy
- **Monotone constraints** via `monotone_constraints` parameter
- **Feature importance weighting** via `feature_weights` parameter
- **`max_leaves` parameter** for leaf-budget-oriented training
- **Warm-starting / incremental training** via `warm_start=True`
- **Up to 65,535 bins per feature** (up from 256) with adaptive u8/u16 storage
- **Multiple categorical column support** via `categorical_feature_indices`
- **Histogram buffer reuse** to reduce allocation pressure
- **Objective-aware training metric tracking** -- `evals_result_` now tracks the appropriate metric per objective (RMSE, log-loss, accuracy, NDCG)

### Explanations

- **TreeSHAP** -- polynomial-time exact Shapley values (replaces the previous brute-force method limited to 20-25 features)
- SHAP explanations work with all three estimators

### New Metrics

- `accuracy` -- classification accuracy
- `log_loss` -- binary cross-entropy loss
- `ndcg` -- normalized discounted cumulative gain (with optional `k` parameter)

### Benchmarks

- **Classification scenarios**: `breast_cancer`, `synthetic_classification`
- **Ranking scenario**: `synthetic_ranking`
- Task-type-aware benchmark runner with per-type metrics, factories, and markdown rendering
- Library adapter classes for cross-library ranking comparison (LightGBM, XGBoost, CatBoost)

### Polish

- Codebase-wide hardening pass (Tier 6)
- Integration tests for warm-start, TreeSHAP, multi-categorical, wide bins, configurability, and native runtime

## 0.1.2

- Zero-copy numpy prediction (75-105x prediction speedup)
- Dense native preprocessing path
- Stage timing output in benchmarks

## 0.1.1

- Expanded benchmark suite (5 regression scenarios)
- Dataset-aware training policy improvements

## 0.1.0

- Initial public release
- `GBMRegressor` with squared-error objective
- Deterministic CPU training with Rayon parallelism
- SHAP explanations (brute-force, 20-feature limit)
- Purged time-series and panel cross-validation splits
- Native artifact prediction
- macOS arm64 and Linux x86_64 wheels
