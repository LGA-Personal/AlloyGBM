Quickstart
==========

Regression
----------

.. code-block:: python

   from alloygbm import GBMRegressor, rmse

   X_train = [
       [0.0, 1.0],
       [1.0, 0.0],
       [2.0, 1.0],
       [3.0, 0.0],
   ]
   y_train = [0.2, 0.9, 1.8, 2.7]

   X_test = [
       [1.5, 1.0],
       [2.5, 0.0],
   ]
   y_test = [1.3, 2.3]

   model = GBMRegressor(
       learning_rate=0.05,
       max_depth=6,
       n_estimators=1200,
       training_policy="auto",
       deterministic=True,
       seed=7,
   )
   model.fit(X_train, y_train)

   predictions = model.predict(X_test)
   print(predictions)
   print("rmse:", rmse(y_test, predictions))

Binary classification
---------------------

.. code-block:: python

   from alloygbm import GBMClassifier, accuracy, log_loss

   model = GBMClassifier(
       learning_rate=0.05,
       max_depth=6,
       n_estimators=500,
       deterministic=True,
       seed=7,
   )
   model.fit(X_train, y_train)  # y must be {0, 1}

   labels = model.predict(X_test)            # [0, 1, 1, 0, ...]
   probas = model.predict_proba(X_test)      # shape (n_samples, 2)

   print("accuracy:", accuracy(y_test, labels))
   print("log_loss:", log_loss(y_test, probas[:, 1]))

``GBMClassifier`` uses binary cross-entropy loss internally and applies a
sigmoid transform to produce probabilities. It inherits sklearn's
``ClassifierMixin`` when sklearn is available.

Learning-to-rank
----------------

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

Supported ranking objectives: ``rank:pairwise``, ``rank:ndcg``,
``rank:xendcg``, ``queryrmse``, ``yetirank``.

Multi-output ranking
--------------------

.. code-block:: python

   from alloygbm import MultiLabelGBMRanker
   import numpy as np

   y_train = np.column_stack([clicks_train, conversions_train])
   model = MultiLabelGBMRanker(
       ranking_objective=["rank:ndcg", "rank:pairwise"],
       learning_rate=0.05,
       n_estimators=300,
       seed=7,
   )
   model.fit(X_train, y_train, group=query_ids_train)

   scores = model.predict(X_test)   # shape (n_rows, n_labels)

``MultiLabelGBMRanker`` trains one independent :class:`GBMRanker` per
label sharing ``group`` and optional ``factor_exposures``; see
:doc:`estimator` for the full reference. Joint shared-tree multi-label
boosting is queued for v0.7.3.

Interaction constraints
-----------------------

.. code-block:: python

   from alloygbm import GBMRegressor

   # Splits along any root-to-leaf path may only use features within ONE group.
   # Features outside all groups are unrestricted (LightGBM semantics).
   model = GBMRegressor(
       n_estimators=500,
       interaction_constraints=[[0, 1, 2], [3, 4]],
       seed=7,
   )
   model.fit(X_train, y_train)

Up to 64 groups per fit; enforced through both level-wise and leaf-wise
tree builders. Available on every estimator.

MorphBoost (optional adaptive mode)
-----------------------------------

Any of the three estimators supports an opt-in MorphBoost training mode
(see :doc:`morphboost`):

.. code-block:: python

   from alloygbm import GBMRegressor

   model = GBMRegressor(
       learning_rate=0.05,
       max_depth=6,
       n_estimators=1200,
       training_mode="morph",       # opt in
       seed=7,
   )
   model.fit(X_train, y_train)

A learning-rate schedule (``lr_schedule="warmup_cosine"``) can also be
applied independently of ``training_mode``, useful for low-LR
high-``n_estimators`` runs:

.. code-block:: python

   model = GBMRegressor(
       learning_rate=0.01,
       n_estimators=5000,
       training_mode="morph",
       lr_schedule="warmup_cosine",
       lr_warmup_frac=0.1,
   )

Validation and early stopping
-----------------------------

.. code-block:: python

   from alloygbm import GBMRegressor

   model = GBMRegressor(
       learning_rate=0.05,
       max_depth=6,
       n_estimators=1200,
       early_stopping_rounds=50,
       min_validation_improvement=1e-4,
       min_data_in_leaf=32,
       lambda_l2=1.0,
       deterministic=True,
       seed=7,
   )

   model.fit(
       X_train,
       y_train,
       eval_set=(X_valid, y_valid),
   )

   print("best_iteration_:", model.best_iteration_)
   print("best_score_:", model.best_score_)
   print("n_estimators_:", model.n_estimators_)
   print("evals_result_ keys:", model.evals_result_.keys())
   print("fit_timing_:", model.fit_timing_)

Use ``eval_set`` whenever you enable ``early_stopping_rounds``. Early stopping
monitors the objective-appropriate metric (RMSE for regression, log-loss for
classification, NDCG for ranking).

Model persistence
-----------------

.. code-block:: python

   import pickle

   # Pickle round-trip
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)

   # Native save/load
   model.save_model("model.agbm")
   loaded = GBMRegressor.load_model("model.agbm")

What happens during ``fit(...)``
--------------------------------

At a high level, AlloyGBM:

1. validates and normalizes the Python inputs
2. chooses a dense native fast path when possible
3. quantizes continuous features inside the native Rust training path
4. trains a native Rust artifact using the appropriate objective
5. stores the serialized artifact and a native predictor handle in the estimator

After fitting, the estimator supports:

- ``predict(...)``
- ``predict_proba(...)`` (classifier only)
- ``shap_values(...)``
- ``feature_importances(...)``
- artifact-backed prediction via ``predict_from_artifact(...)``
- fitted training summaries via ``best_iteration_``, ``best_score_``,
  ``n_estimators_``, ``evals_result_``, and ``fit_timing_``

NaN / missing values
--------------------

AlloyGBM handles NaN values natively. You do not need to impute missing values
before training or prediction. The engine learns the optimal split direction for
missing values at each node.

Dense array-like inputs
-----------------------

The Python bridge supports optimized paths for array-like objects exposing:

- ``to_numpy``
- ``to_list``
- ``tolist``

You do not need to eagerly convert everything to nested Python lists just to use
the native path.
