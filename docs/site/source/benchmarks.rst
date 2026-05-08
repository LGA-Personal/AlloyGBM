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
- ``benchmarks/numerai_benchmark.py`` -- Numerai tournament benchmark with
  walk-forward CV, residualized targets, and Numerai-specific scoring
- ``benchmarks/pl_trees_benchmark.py`` -- piecewise-linear-leaf
  convergence-curve and λ-sweep analysis. Report at
  ``docs/benchmarks/pl_trees_v1.md``.

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
