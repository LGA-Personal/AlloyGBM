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

The suite spans three task types with the following scenarios:

**Regression:** ``dense_numeric``, ``california_housing``, ``bike_sharing``,
``panel_time_series``, ``dow_jones_financial``

**Classification:** ``breast_cancer``, ``synthetic_classification``

**Ranking:** ``synthetic_ranking``

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

- AlloyGBM competes on ``synthetic_ranking`` using native LambdaMART,
  evaluated via NDCG@5, NDCG@10, and full NDCG

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
     --scenarios synthetic_ranking

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
