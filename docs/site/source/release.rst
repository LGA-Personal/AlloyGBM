Release and platform policy
===========================

AlloyGBM ``0.5.0`` release notes and platform policy.

What's new in 0.5.0
-------------------

**Piecewise-linear (PL) tree leaves:**

- New opt-in ``leaf_model="linear"`` parameter on
  :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`, and
  :class:`~alloygbm.GBMRanker`. Each leaf stores a small linear model
  ``f_s(x) = b_s + Σ α_j x_j`` (up to 8 regressors per leaf, inherited from
  the split path's feature indices; the cap is internal and not user-tunable
  in v0.5.0). Optimal weights are solved in closed form via the ridge
  regression ``α* = -(XᵀHX + λI)⁻¹ Xᵀg``, regularised by ``lambda_l2``.
- Default ``leaf_model="constant"`` preserves all prior behaviour exactly.
- New artifact section ``ModelSectionKind::LinearLeafCoefficients`` stores
  per-stump linear leaf data; backward-compatible with v0.4.0 artifacts.
- Native-bitset categorical splits (``max_cat_threshold > 0``) fall back to
  constant leaves at the categorical split node; descendant numeric leaves
  use linear leaves normally.
- Multi-class softmax fits each per-class tree sequence with linear leaves
  independently.
- ``leaf_model="linear"`` composes with ``training_mode="morph"``.
- SHAP (``shap_values``, ``feature_importances``) currently raises an error
  for ``leaf_model="linear"`` artifacts; use ``leaf_model="constant"`` if you
  need SHAP.

**Performance:**

- ~10× faster convergence on linearly-structured datasets (fewer rounds to
  reach the same RMSE).
- +3.5% RMSE on California Housing and +1.75pp accuracy on Breast Cancer vs
  constant leaves.
- 2–8× per-round training overhead from the closed-form Cholesky solve.
  Recommended ``lambda_l2 >= 0.01`` for weight stability.

**Benchmarks:**

- ``alloygbm_linear`` and ``alloygbm_morph_linear`` arms added to
  ``benchmarks/run_model_comparison.py`` for all four task types.
- New ``benchmarks/pl_trees_benchmark.py`` script with convergence-curve and
  λ-sweep analysis.
- Benchmark report committed to ``docs/benchmarks/pl_trees_v1.md``.

What's new in 0.4.0
-------------------

**MorphBoost mode and SIMD acceleration:**

- New opt-in adaptive training mode via ``training_mode="morph"``,
  implementing the criterion from
  `Kriuk (2025) <https://arxiv.org/pdf/2511.13234>`_. Available on
  :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`, and
  :class:`~alloygbm.GBMRanker`. See :doc:`morphboost`.
- New per-iteration learning-rate schedule parameter ``lr_schedule``
  (``"constant"`` default, ``"warmup_cosine"`` available). Independent of
  ``training_mode`` — usable on its own.
- Schedule-aware auto early-stopping: when an LR schedule is active, the
  auto-tuned ``min_loss_improvement`` threshold is scaled by
  ``current_lr / max_lr``, and warmup-phase rounds are tolerated without
  termination.
- Backend SIMD acceleration via the ``wide`` crate (safe API; AVX2 / NEON
  intrinsics underneath, scalar fallback otherwise). Histogram bin-scan
  and EMA passes are now vectorized; histogram tile sizing is auto-tuned
  for high-feature workloads.
- New benchmark harnesses: ``benchmarks/morph_report.py``,
  ``benchmarks/morph_ablation.py``, and an enhanced
  ``benchmarks/numerai_benchmark.py`` with MorphBoost arms and a startup
  build-freshness check.
- ``benchmarks/run_model_comparison.py`` registers two new arms by default
  per task type: ``alloygbm_morph`` and ``alloygbm_morph_cosine``. New
  ``--models`` flag filters which arms run.

What's new in 0.3.2
--------------------

``0.3.2`` fixes silent zero-tree training in ``GBMRanker``, corrects signature
introspection, and adds a real-data ranking benchmark:

**GBMRanker training fixes:**

- The auto training policy's density-based ``min_split_gain`` and
  ``min_loss_improvement`` floors are no longer applied to ranking objectives.
  Ranking gradients are an order of magnitude smaller than
  regression/classification gradients; on datasets where
  ``row_count * feature_count >= 65 536`` these floors were causing training to
  exit after round 1 with zero trees committed.
- The main training loop's unconditional ``loss_improvement < 0`` early-exit no
  longer fires for ranking objectives, where round-to-round loss oscillation is
  expected behaviour.
- ``inspect.signature(GBMRanker.__init__)`` now returns the full parameter set
  (``ranking_objective`` plus all ``GBMRegressor`` parameters). Previously only
  three parameters were visible, causing tools that build kwargs via signature
  introspection to silently train with ``n_estimators=6``.

**Diagnostics:**

- ``stop_reason_`` and ``rounds_completed_`` attributes are now set on all
  estimators after ``fit()`` to surface the engine's early-stop reason and
  actual committed round count.

**Benchmarks:**

- Added ``california_ranking``: California Housing reframed as learning-to-rank
  with geographic grid cells as queries and ``median_house_value`` bucketed into
  5 graded relevance levels (~44 queries × 468 docs = ~20 595 rows).

What was new in 0.3.1
----------------------

``0.3.1`` fixed multiclass prediction and expanded the benchmark suite:

- Fixed ``class_trees`` threshold conversion so multiclass models predict
  correctly with continuous float features
- Fixed multiclass benchmark argmax label mapping with ``model.classes_``
- Added ``wine_multiclass``, ``digits_multiclass``, ``adult_income``,
  ``abalone_regression`` benchmark scenarios
- Activated ``synthetic_multiclass`` and ``synthetic_categorical`` scenarios
- Rewrote ``benchmarks/README.md``

What was new in 0.3.0
----------------------

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

For ``0.5.0``, the intended release surface is:

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
