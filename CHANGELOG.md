# Changelog

## 0.4.0

### New Features

- **MorphBoost adaptive training mode** -- opt-in via `training_mode="morph"` on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`. Augments the standard gradient gain with a normalized information-theoretic term (with `tanh(iter/20)` warmup ramp), per-class EMA-driven gain shaping, depth-based leaf penalty, per-iteration leaf shrinkage, and an optional balance penalty. Implementation follows the formulation in [Kriuk (2025), *MorphBoost*](https://arxiv.org/pdf/2511.13234) with deliberate corrections vs the paper's reference code.
- **Per-iteration learning-rate schedules** -- new `lr_schedule` parameter (`"constant"` default or `"warmup_cosine"`), independent of `training_mode`. The `warmup_cosine` schedule does linear warmup over `lr_warmup_frac * n_estimators` rounds then half-cosine decay to a `0.01 * learning_rate` floor.
- **Schedule-aware auto early-stopping** -- when an LR schedule is active, the auto-tuned `min_loss_improvement` threshold is scaled by `current_lr / max_lr`, and warmup-phase rounds (empty trees, slightly-negative loss improvements) do not terminate training. Outside warmup, behaviour is bit-identical to the previous policy.
- **MorphBoost configuration in artifacts** -- the `MorphConfig` and `final_iteration` are persisted as an optional artifact section so loaded models predict consistently.

### Performance

- **SIMD-accelerated kernels** via the `wide` crate (safe API; AVX2 / NEON intrinsics under the hood, scalar fallback otherwise). Standard-path histogram bin-scan and `GradientEmaStats` mean+variance pass are vectorized.
- **Tile-size auto-tuning** for histogram parallelism. The hard-coded `MAX_TILE_FEATURE_WIDTH=64` is replaced by a thread-count-aware helper that targets ~2 tiles per thread, clamped to `[16, 64]`. Improves utilization at high feature counts (Numerai 780-feature `medium` set, etc.).
- **Hoisted morph per-round constants** -- `tanh(iter/20)`, gradient/info-score blend coefficients, and warmup-branch flags are precomputed once per round into `MorphPrecomputed` rather than evaluated per bin candidate.

### Benchmarks

- **`alloygbm_morph` and `alloygbm_morph_cosine` arms** added to `benchmarks/run_model_comparison.py` for all four task types. New `--models` flag filters which arms run.
- **`benchmarks/morph_report.py`** -- focused MorphBoost-vs-peers comparison on a curated set of sklearn datasets (~2 minutes with `--quick`).
- **`benchmarks/morph_ablation.py`** -- toggles MorphBoost components individually on synthetic regression/binary/ranking datasets to attribute per-component impact.
- **`benchmarks/numerai_benchmark.py`** -- adds `alloygbm_morph` and `alloygbm_morph_cosine` arms, plus a startup build-freshness check that logs the loaded extension's mtime, the worktree's git HEAD, and a `STALE BUILD` warning if the extension predates HEAD.

### Documentation

- New dedicated MorphBoost guides at `docs/user/morphboost.md` and `docs/site/source/morphboost.rst`, with the formulation, full parameter reference, LR-schedule behaviour, and tuning notes.
- Cross-references and parameter notes added across the user guide, Sphinx site, benchmark READMEs, and the limitations doc.

## 0.3.2

### Bug Fixes

- **GBMRanker silent zero-tree training** -- the auto training policy's density-based `min_split_gain` and `min_loss_improvement` floors were being applied to ranking objectives, whose gradient magnitudes are an order of magnitude smaller than regression/classification gradients; on datasets with `row_count * feature_count >= 65_536` no split cleared the floor and training exited after round 1. The auto policy is now objective-aware and skips those floors for all ranking objectives (`rank:pairwise`, `rank:ndcg`, `rank:xendcg`, `queryrmse`, `yetirank`).
- **Training loop loss-regression break for ranking** -- the main training loop's unconditional `loss_improvement < 0` early-exit was firing on ranking objectives where round-to-round loss oscillation is expected; that guard is now skipped for objectives that require group IDs.
- **`GBMRanker` signature introspection** -- `inspect.signature(GBMRanker.__init__)` previously returned only `(self, ranking_objective, **kwargs)`, causing tools that build parameters via signature inspection (sklearn clone, benchmark runners, IDEs) to silently use `n_estimators=6` default; `__init__.__signature__` is now set to the combined `GBMRegressor + ranking_objective` parameter list.

### New Features

- **`stop_reason_` and `rounds_completed_` attributes** on all estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) -- set after `fit()` to surface the engine's early-stop reason and actual round count for diagnostics and debugging.

### Benchmarks

- **`california_ranking` scenario** -- California Housing dataset reframed as learning-to-rank: 1-degree lat/lon grid cells act as query groups (~44 queries × 468 docs = ~20,595 rows), and `median_house_value` is bucketed into 5 quantile-based relevance levels; provides a real-data complement to `synthetic_ranking`.

## 0.3.1

### Bug Fixes

- **Multiclass predictor threshold conversion** -- `convert_bin_thresholds_to_float*` functions in `crates/predictor` now correctly convert `class_trees` in addition to `trees`; previously, multiclass models with continuous float features produced near-random predictions because `class_trees` bin-ID thresholds were never converted to float values
- **Multiclass argmax label mapping** -- benchmark runner maps `np.argmax` column indices through `model.classes_` so accuracy is computed correctly when class labels are not exactly `0..K-1`

### Benchmarks

- **Real-dataset benchmark scenarios** -- added `wine_multiclass` (sklearn Wine, 3-class, 178 rows), `digits_multiclass` (sklearn Digits, 10-class, 1797 rows), `adult_income` (UCI Adult, binary classification, ~30K rows, mixed features), `abalone_regression` (UCI Abalone, regression, 4177 rows), and `news_ranking` (placeholder with setup instructions)
- **Multiclass classification support** in `run_model_comparison.py` -- stratified split, argmax predictions with label mapping, multiclass log-loss, separate factory functions with correct per-library multiclass objectives
- **Activated dormant scenarios** -- `synthetic_multiclass` and `synthetic_categorical` are now included in `AVAILABLE_SCENARIOS`
- **Rewritten `benchmarks/README.md`** -- comprehensive scenario table, task-type split strategies, feature coverage table, per-record timing reference, recently-shipped feature coverage matrix

## 0.3.0

### Native Categorical Splits

- **Fisher-sort categorical split-finding** -- optimal binary partition of categories in O(K log K) time via gradient-ordered category sorting with O(K) prefix-scan split evaluation
- **Bitset-based O(1) prediction** -- compact `Vec<u8>` bitset encoding where bit K=1 means category K goes left; prediction is a single bit-test per tree node
- **`max_cat_threshold` parameter** -- controls the maximum number of categories for native splits (default 0 = disabled, opt-in); features exceeding the threshold fall back to target encoding
- **Backward-compatible artifact format** -- new `NativeCategoricalSplits` section (ID=7) with `stump_flags` bit 1 encoding; old artifacts load without changes
- **Category-to-ID mapping** -- string categories are mapped to integer IDs at the Python layer; mappings are preserved through pickle, save/load, and get/set params
- **Full estimator support** -- works with `GBMRegressor`, `GBMClassifier`, and `GBMRanker` (via inheritance)

### Multi-Class Classification

- **`GBMClassifier` multi-class support** -- softmax (multinomial cross-entropy) objective for K > 2 classes, auto-detected from training labels
- **`predict_proba`** returns (n_samples, K) probability matrix with softmax normalization
- **Label encoding** -- arbitrary integer labels are mapped to 0..K-1 internally

### Custom Objectives and Metrics

- **Custom objective functions** via `objective=callable` -- user-defined gradient/hessian computation with fast numpy I/O
- **Custom evaluation metrics** via `eval_metric=callable` -- user-defined metric callbacks for early stopping and `evals_result_` tracking
- **`higher_is_better` protocol** -- custom metrics declare optimization direction

### Benchmarks

- **`synthetic_categorical`** benchmark scenario for evaluating native categorical split performance
- **`synthetic_custom_objective`** and **`synthetic_multiclass`** benchmark scenarios

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
