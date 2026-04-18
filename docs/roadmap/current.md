# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is aimed at strong practical performance on structured tabular workloads, with particular strength on financial and time-aware problems.

The `0.3.2` release fixes GBMRanker training (silent zero-tree training on larger datasets) and adds a real-data ranking benchmark. The `0.3.1` release fixed multiclass prediction and expanded the benchmark suite. The `0.3.0` release added native categorical splits, multi-class classification, and custom objective/metric support.

## What Shipped In 0.3.2

- Fixed GBMRanker silent zero-tree training: the auto training policy's density-based `min_split_gain` floor and `min_loss_improvement` floor were being applied to ranking objectives, which have gradient magnitudes an order of magnitude smaller than regression/classification — no split cleared the floor and training exited on round 1. The auto policy is now objective-aware and skips those floors for all ranking objectives.
- Fixed training loop loss-regression early break firing on ranking objectives where round-to-round loss oscillation is expected and benign
- Fixed `inspect.signature(GBMRanker.__init__)` returning only 3 parameters (`self`, `ranking_objective`, `**kwargs`) — parameter-building tools (sklearn clone, benchmarks, IDEs) using signature introspection silently trained with `n_estimators=6` default; now exposes the full parameter set
- Added `stop_reason_` and `rounds_completed_` attributes on all estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) for training diagnostics
- Added `california_ranking` benchmark scenario: California Housing reframed as learning-to-rank with geographic grid cells as queries and median house value bucketed into 5 graded relevance levels (~44 queries × 468 docs)

## What Shipped In 0.3.1

- Fixed multiclass predictor threshold conversion: `class_trees` are now converted in all three threshold-conversion paths (linear, quantile, pre-binned); continuous-feature multiclass models now produce correct predictions
- Fixed multiclass benchmark argmax label mapping: `model.classes_` is now used so accuracy is correct for non-zero-indexed labels
- Added real-dataset benchmark scenarios: `wine_multiclass`, `digits_multiclass`, `adult_income`, `abalone_regression`
- Added `news_ranking` placeholder scenario with dataset selection instructions
- Activated `synthetic_multiclass` and `synthetic_categorical` benchmark scenarios
- Rewrote `benchmarks/README.md` with scenario table, feature coverage matrix, timing reference, and usage examples

## What Shipped In 0.3.0

- Native categorical splits with Fisher-sort algorithm and bitset-based O(1) prediction (`max_cat_threshold`)
- Multi-class classification (`GBMClassifier` with softmax/multinomial for K > 2 classes)
- Custom objective functions (`objective=callable`) with fast numpy I/O
- Custom evaluation metric callbacks (`eval_metric=callable`) with early stopping support
- Synthetic categorical and custom objective benchmark scenarios

## What Shipped In 0.2.0

- Binary classification (`GBMClassifier`) with log-loss objective
- Learning-to-rank (`GBMRanker`) with 5 objectives (RankNet, LambdaMART, XE-NDCG, QueryRMSE, YetiRank)
- NaN / missing value support across all crates
- Sample weight and group ID support from Python
- Model persistence (pickle, save/load, artifact export)
- Feature name capture and sklearn compatibility (`BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- TreeSHAP (polynomial-time, replaces the old 25-feature-capped brute-force method)
- Monotone constraints and feature importance weighting
- Leaf-wise (best-first) tree growth strategy
- Warm-starting / incremental training
- Up to 65,535 bins per feature (up from 256)
- Multiple categorical column support
- Histogram buffer reuse
- Objective-aware training metric tracking
- Expanded benchmark suite (regression + classification + ranking)

## Current Priorities

1. Close remaining performance gaps on broad tabular datasets.
2. Explore GPU/accelerator backend after the CPU baseline is solid enough to serve as reference.
3. Continue expanding the benchmark suite with real-world classification and ranking datasets.

## Longer-Term Themes

- Multi-label ranking.
- Interaction constraints.
- Dart / GOSS boosting modes.
- GPU backend.
- Better operational diagnostics and model introspection.

## Planning Style

The project no longer uses the old version-layer planning hierarchy as the active documentation model.

Going forward:

- current intent lives in `docs/roadmap/`
- research notes live in `docs/ideas/`
- benchmark framing lives in `docs/benchmarks/` and `benchmarks/`
- implementation plans from the 0.1.x cycle are archived in `docs/archive/v0.1_plans/`
