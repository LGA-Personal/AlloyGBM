Benchmarks
==========

This page summarizes how AlloyGBM is benchmarked and what the current public
results say.

Methodology
-----------

The benchmark runner lives in ``benchmarks/run_model_comparison.py`` and
compares AlloyGBM against:

- XGBoost
- LightGBM
- CatBoost

It also includes additional AlloyGBM variants as separate arms by default
per task type:

- ``alloygbm_morph`` -- ``training_mode="morph"`` with constant LR
- ``alloygbm_morph_cosine`` -- ``training_mode="morph"`` with
  ``lr_schedule="warmup_cosine"``
- ``alloygbm_linear`` -- ``leaf_model="linear"`` (piecewise-linear leaves)
  with auto training mode
- ``alloygbm_morph_linear`` -- ``leaf_model="linear"`` combined with
  ``training_mode="morph"``

Use the runner's ``--models`` flag to filter which arms run. Focused
harnesses are also provided:

- ``benchmarks/morph_report.py`` -- quick MorphBoost-vs-peers comparison
- ``docs/benchmarks/morphboost_pr132.md`` -- exhaustive SIMD scanner,
  quality-matrix, calibration, and secondary-cost evidence for PR #132
- ``benchmarks/numerai_benchmark.py`` -- Numerai tournament benchmark with
  walk-forward CV, residualized targets, and Numerai-specific scoring
- ``benchmarks/pl_trees_benchmark.py`` -- piecewise-linear-leaf
  convergence-curve and λ-sweep analysis. Report at
  ``docs/benchmarks/pl_trees_v1.md``.
- ``benchmarks/objective_benchmark.py`` -- deterministic large-query
  LambdaMART and skewed-count GLM validation. Report at
  ``docs/benchmarks/objective_benchmark_v1.md``.
- ``benchmarks/review_guardrails.py`` -- deterministic July-review evidence
  for smoothed-pinball split selection, GOSS rates, and DART dropout profiles.
  The report is ``docs/benchmarks/review_guardrails_v1.md``. DART timing is
  descriptive; the 1.50x RMSE gate applies only to explicit default-like
  profiles with ``drop_rate <= 0.10``, while the stress profile remains
  reported and contract-checked.
- ``benchmarks/monotone_constraints_benchmark.py`` -- deterministic scalar
  monotone-constraint acceptance evidence for regression and binary models.
  The report is ``docs/benchmarks/monotone_constraints_v1.md``; finite numeric
  sweeps and held-out quality are gated, while timing is descriptive.
- ``benchmarks/multiclass_parallelism_benchmark.py`` -- deterministic
  ``n_jobs`` and multiclass class-tree scaling evidence across matrix shapes,
  class counts, and both growth strategies. The report is
  ``docs/benchmarks/multiclass_parallelism_v1.md``; serial and parallel
  artifacts and predictions must match exactly.
- ``benchmarks/allocation_reuse_benchmark.py`` -- isolated, manifest-attested
  baseline/candidate evidence for histogram and row-partition allocation reuse
  across four matrix shapes and both growth strategies. The report is
  ``docs/benchmarks/allocation_reuse_v1.md``; artifacts and predictions must
  match exactly before aggregate native-time and RSS gates are applied.

The suite spans three task types with the following scenarios:

**Regression:** ``dense_numeric``, ``california_housing``, ``bike_sharing``,
``panel_time_series``, ``dow_jones_financial``

**Classification:** ``breast_cancer``, ``synthetic_classification``

**Ranking:** ``synthetic_ranking``, ``california_ranking``

Profiles are evaluated across shallow, mid, and deep configurations so the
comparison is not tied to a single parameter shape.

Current results
---------------

**Regression:**

- AlloyGBM is strongest on ``panel_time_series``
- AlloyGBM is strong on ``dow_jones_financial``
- AlloyGBM is competitive but not leading on ``dense_numeric``
- AlloyGBM trails on ``california_housing`` and ``bike_sharing``
- AlloyGBM is typically the fastest trainer on most scenario/profile rows

**Classification:**

- AlloyGBM is competitive with established libraries on accuracy, log-loss, and
  AUC across ``breast_cancer`` and ``synthetic_classification``

**Ranking:**

- AlloyGBM competes on ``synthetic_ranking`` and ``california_ranking`` using
  native LambdaMART, evaluated via NDCG@5, NDCG@10, and full NDCG

**MorphBoost variants:**

- On Numerai-style residualized regression at scale (~2.7M rows × 42 features
  × 5000 rounds), AlloyGBM's MorphBoost variants lead all peer libraries on
  validation MMC (Meta-Model Contribution) and Sharpe; numerai_corr trails by
  a small margin (~0.0006-0.0009).
- ``alloygbm_morph`` is typically the fastest of the three AlloyGBM variants
  on this workload due to faster convergence under the EMA-shaped gain.

**Piecewise-linear leaf variants:**

- ``leaf_model="linear"`` shows ~10× faster convergence on linearly-structured
  data, +3.5% RMSE on California Housing, and +1.75pp accuracy on Breast
  Cancer vs constant-leaf baselines, at a 2–8× per-round training overhead.
- See ``docs/benchmarks/pl_trees_v1.md`` for the full report.

Metrics by task type
--------------------

.. list-table::
   :header-rows: 1

   * - Task type
     - Metrics
   * - Regression
     - RMSE, MAE, R2
   * - Classification
     - Accuracy, Log-Loss, AUC
   * - Ranking
     - NDCG@5, NDCG@10, NDCG

How to run the suite
--------------------

.. code-block:: console

   python3 benchmarks/run_model_comparison.py --force-prepare

Focused regression comparison:

.. code-block:: console

   python3 benchmarks/run_model_comparison.py \
     --force-prepare \
     --scenarios california_housing bike_sharing dense_numeric panel_time_series dow_jones_financial

Review-evidence capture:

.. code-block:: console

   python3 benchmarks/review_guardrails.py --gate \
     --output docs/benchmarks/review_guardrails_v1.md

Scalar monotone-constraint acceptance:

.. code-block:: console

   python3 benchmarks/monotone_constraints_benchmark.py --quick --gate
   python3 benchmarks/monotone_constraints_benchmark.py \
     --gate \
     --output docs/benchmarks/monotone_constraints_v1.md

Classification only:

.. code-block:: console

   python3 benchmarks/run_model_comparison.py \
     --force-prepare \
     --scenarios breast_cancer synthetic_classification

Ranking only:

.. code-block:: console

   python3 benchmarks/run_model_comparison.py \
     --force-prepare \
     --scenarios synthetic_ranking california_ranking

Stage timing output
-------------------

Per-record benchmark output includes:

- ``input_adaptation_seconds``
- ``native_bridge_prepare_seconds``
- ``native_train_seconds``
- ``fit_seconds``
- ``predict_seconds``

Use those timing columns to tell apart Python-side adaptation cost and native
training cost.

Interpretation
--------------

The benchmark suite is designed to answer both of these questions:

- Where is AlloyGBM already strong?
- Where does it still lag established libraries?

The second question matters. These docs intentionally preserve that honesty.
