Time-aware validation
=====================

AlloyGBM includes leakage-aware split helpers for time series and panel data.

Why they matter
---------------

Random splits are often misleading for:

- forecasting
- panel regression
- finance datasets with ordered timestamps

If the same time bucket appears in both train and test data, measured
performance can be inflated by leakage.

Time series splits
------------------

Use ``purged_time_series_splits(...)`` when you have a single time axis:

.. code-block:: python

   from alloygbm import purged_time_series_splits

   time_index = [0, 0, 1, 1, 2, 2, 3, 3]
   splits = purged_time_series_splits(
       time_index,
       n_splits=4,
       purge_gap=0,
       embargo=0,
   )

Key parameters:

- ``n_splits``
- ``purge_gap``
- ``embargo``

Panel splits
------------

Use ``purged_panel_splits(...)`` when you have both a time index and a group
index:

.. code-block:: python

   from alloygbm import purged_panel_splits

   time_index = [0, 0, 1, 1, 2, 2]
   group_index = ["A", "B", "A", "B", "A", "B"]

   splits = purged_panel_splits(
       time_index,
       group_index,
       n_splits=3,
       purge_gap=0,
       embargo=0,
   )

Panel behavior is still time-bucketed across all groups, which is usually the
correct default when leakage is primarily temporal.

Using the splits with AlloyGBM
------------------------------

.. code-block:: python

   from alloygbm import GBMRegressor, purged_time_series_splits, rmse

   rows = [[float(i), float(i % 2)] for i in range(20)]
   targets = [float(i) * 0.1 for i in range(20)]
   time_index = [i // 2 for i in range(20)]

   scores = []
   for train_idx, test_idx in purged_time_series_splits(time_index, n_splits=5):
       model = GBMRegressor(deterministic=True, seed=7)
       model.fit([rows[i] for i in train_idx], [targets[i] for i in train_idx])
       preds = model.predict([rows[i] for i in test_idx])
       scores.append(rmse([targets[i] for i in test_idx], preds))

Passing ``time_index`` into ``fit(...)``
----------------------------------------

Pass ``time_index=`` to ``GBMRegressor.fit(...)`` when you are using:

- ``categorical_feature_index``
- ``categorical_time_aware=True``

That enables time-aware categorical handling during training.
