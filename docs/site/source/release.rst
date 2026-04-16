Release and platform policy
===========================

AlloyGBM ``0.3.0`` release notes and platform policy.

What's new in 0.3.0
--------------------

``0.3.0`` adds native categorical splits, multi-class classification, and
custom objective/metric support:

**Native categorical splits:**

- Fisher-sort categorical split-finding with O(K log K) optimal binary
  partitions and O(1) bitset prediction
- ``max_cat_threshold`` parameter controls the maximum category cardinality
  for native splits (default 0 = disabled, opt-in)
- Category-to-ID mappings preserved through pickle, save/load, and params
- Full support across ``GBMRegressor``, ``GBMClassifier``, and ``GBMRanker``

**Multi-class classification:**

- ``GBMClassifier`` auto-detects K > 2 classes and uses softmax
  (multinomial cross-entropy) objective with K trees per round
- ``predict_proba`` returns (n_samples, K) probability matrix

**Custom objectives and metrics:**

- ``objective=callable`` for user-defined gradient/hessian computation
- ``eval_metric=callable`` for custom evaluation metrics with early stopping
- ``higher_is_better`` protocol for metric direction

What was new in 0.2.0
---------------------

``0.2.0`` was a major capability expansion from the regression-only ``0.1.x``
series:

**New estimators:**

- ``GBMClassifier`` -- binary classification with log-loss objective,
  ``predict_proba``, sklearn ``ClassifierMixin``
- ``GBMRanker`` -- learning-to-rank with 5 objectives (RankNet, LambdaMART,
  XE-NDCG, QueryRMSE, YetiRank)

**Core improvements:**

- NaN / missing value support across training and prediction
- Sample weight support via ``fit(..., sample_weight=...)``
- Group ID support via ``fit(..., group=...)``
- Model persistence: pickle, ``save_model``/``load_model``, artifact export
- Feature name capture from pandas DataFrames and other named inputs
- sklearn compatibility (``BaseEstimator``, ``RegressorMixin``,
  ``ClassifierMixin``, ``get_params``, ``set_params``, ``score``)
- ``min_split_gain`` exposed as a user parameter

**Training enhancements:**

- Leaf-wise (best-first) tree growth via ``tree_growth="leaf"``
- Monotone constraints via ``monotone_constraints``
- Feature importance weighting via ``feature_weights``
- ``max_leaves`` parameter for leaf-budget-oriented training
- Warm-starting / incremental training via ``warm_start=True``
- Up to 65,535 bins per feature (adaptive u8/u16 storage)
- Multiple categorical column support via ``categorical_feature_indices``
- Histogram buffer reuse to reduce allocation pressure
- Objective-aware training metric tracking (RMSE, log-loss, accuracy, NDCG)

**Explanations:**

- TreeSHAP (polynomial-time exact Shapley values, replaces the 25-feature
  brute-force method)
- SHAP limit raised from 20 to 25 features (for legacy brute-force path),
  then replaced entirely by TreeSHAP

**Metrics:**

- ``accuracy`` -- classification accuracy
- ``log_loss`` -- binary cross-entropy
- ``ndcg`` -- normalized discounted cumulative gain (with optional k)

**Benchmarks:**

- Classification scenarios: ``breast_cancer``, ``synthetic_classification``
- Ranking scenario: ``synthetic_ranking``
- Task-type-aware benchmark runner with per-type metrics and rendering

Validated release surface
-------------------------

For ``0.3.0``, the intended release surface is:

- macOS ``arm64`` wheel
- Linux ``x86_64`` manylinux wheel
- source distribution

Deferred targets
----------------

These are intentionally deferred:

- Windows wheels
- macOS Intel wheels

Release checklist summary
-------------------------

Before a public release:

- confirm package metadata and version
- confirm user docs are up to date
- confirm CI is green
- confirm the built wheel installs in a fresh environment
- confirm the publish workflow smoke-tests its wheel artifacts before upload
- confirm benchmark messaging stays narrow and defensible
