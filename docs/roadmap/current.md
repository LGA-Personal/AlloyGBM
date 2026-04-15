# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is aimed at strong practical performance on structured tabular workloads, with particular strength on financial and time-aware problems.

The `0.3.0` release adds native categorical splits, multi-class classification, and custom objective/metric support.

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
