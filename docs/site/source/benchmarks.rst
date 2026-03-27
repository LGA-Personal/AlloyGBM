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

The expanded public regression set currently includes:

- ``dense_numeric``
- ``california_housing``
- ``bike_sharing``
- ``panel_time_series``
- ``dow_jones_financial``

Profiles are evaluated across shallow, mid, and deep configurations so the
comparison is not tied to a single parameter shape.

Current public benchmark statement
----------------------------------

The narrow, defensible release claim is:

- strong on ``panel_time_series``
- strong on ``dow_jones_financial``
- weaker on ``california_housing`` and ``bike_sharing``

Additional nuance:

- AlloyGBM is competitive but not leading on ``dense_numeric``
- in the latest recorded benchmark refresh, AlloyGBM was also the fastest
  trainer on most scenario/profile rows
- the latest benchmark refresh did not show a broad AlloyGBM RMSE regression
  after the training-contract and native dense-preprocessing changes

Representative results
----------------------

.. list-table::
   :header-rows: 1

   * - Scenario
     - Best model
     - Profile
     - Interpretation
   * - ``panel_time_series``
     - AlloyGBM
     - ``shallow_high_lr``
     - Clear AlloyGBM strength
   * - ``dow_jones_financial``
     - AlloyGBM
     - ``deep_low_lr``
     - Strong finance-style showing
   * - ``dense_numeric``
     - CatBoost / XGBoost
     - ``deep_low_lr``
     - AlloyGBM remains competitive but behind
   * - ``california_housing``
     - XGBoost
     - ``deep_low_lr``
     - Visible general-tabular performance gap
   * - ``bike_sharing``
     - CatBoost
     - ``mid_balanced``
     - AlloyGBM improves with depth but does not lead

How to run the suite
--------------------

.. code-block:: console

   python3 benchmarks/run_model_comparison.py --force-prepare

Focused public regression comparison:

.. code-block:: console

   python3 benchmarks/run_model_comparison.py \
     --force-prepare \
     --scenarios california_housing bike_sharing dense_numeric panel_time_series dow_jones_financial \
     --profile-grid default \
     --profile-seeds 7

Stage timing output
-------------------

Per-record benchmark output now includes:

- ``input_adaptation_seconds``
- ``native_bridge_prepare_seconds``
- ``native_train_seconds``
- ``fit_seconds``
- ``predict_seconds``

Use those timing columns to tell apart Python-side adaptation cost and native
training cost when AlloyGBM performance changes.

Interpretation
--------------

The benchmark suite is designed to answer both of these questions:

- Where is AlloyGBM already strong?
- Where does it still lag established libraries?

The second question matters. These docs intentionally preserve that honesty.
