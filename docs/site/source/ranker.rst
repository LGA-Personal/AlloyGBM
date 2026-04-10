GBMRanker
=========

``GBMRanker`` is the learning-to-rank estimator in AlloyGBM.

Overview
--------

``GBMRanker`` extends ``GBMRegressor`` with ranking-specific objectives. All
ranking objectives require query group identifiers to be passed in ``fit()``.
Data is sorted by group internally.

Quick example
-------------

.. code-block:: python

   from alloygbm import GBMRanker, ndcg

   model = GBMRanker(
       ranking_objective="rank:ndcg",
       learning_rate=0.05,
       max_depth=6,
       n_estimators=300,
       deterministic=True,
       seed=7,
   )
   model.fit(X_train, y_train, group=query_ids_train)

   scores = model.predict(X_test)
   print("NDCG@10:", ndcg(y_test, scores, group=query_ids_test, k=10))

Ranking objectives
------------------

- ``"rank:pairwise"`` -- Pairwise logistic loss (RankNet)
- ``"rank:ndcg"`` -- LambdaMART with NDCG weighting (default)
- ``"rank:xendcg"`` -- Cross-entropy approximation to NDCG
- ``"queryrmse"`` -- Query-grouped RMSE
- ``"yetirank"`` -- YetiRank (stochastic NDCG-weighted pairwise)

Parameters
----------

- ``ranking_objective: str = "rank:ndcg"`` -- the ranking loss function

All other parameters are inherited from ``GBMRegressor``.

Methods
-------

- ``fit(X, y, *, group, eval_set=None, eval_group=None, ...)`` -- trains the
  ranker. ``group`` is required and provides per-row query identifiers.
- ``predict(X)`` -- returns raw relevance scores (higher = more relevant)

Evaluation
----------

.. code-block:: python

   from alloygbm import ndcg

   score = ndcg(y_test, predictions, group=query_ids_test)
   score_at_10 = ndcg(y_test, predictions, group=query_ids_test, k=10)

Group format
------------

The ``group`` parameter accepts per-row group identifiers (e.g. query IDs).
AlloyGBM sorts by group internally, so rows do not need to be pre-sorted.

.. code-block:: python

   # Per-row group IDs (AlloyGBM format)
   group = [0, 0, 0, 1, 1, 2, 2, 2, 2]

Early stopping
--------------

.. code-block:: python

   model = GBMRanker(
       ranking_objective="rank:ndcg",
       n_estimators=2000,
       early_stopping_rounds=50,
   )
   model.fit(
       X_train, y_train,
       group=query_ids_train,
       eval_set=(X_valid, y_valid),
       eval_group=query_ids_valid,
   )

Current scope
-------------

- 5 ranking objectives implemented natively in Rust
- Single-label relevance only
- Group identifiers must be unsigned integers
