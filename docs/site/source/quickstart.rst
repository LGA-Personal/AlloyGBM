Quickstart
==========

Minimal regression example
--------------------------

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

Use ``eval_set`` whenever you enable ``early_stopping_rounds``.

What happens during ``fit(...)``
--------------------------------

At a high level, AlloyGBM:

1. validates and normalizes the Python inputs
2. chooses a dense native fast path when possible
3. quantizes continuous features inside the native Rust training path
4. trains a native Rust artifact
5. stores the serialized artifact and a native predictor handle in the estimator

After fitting, the estimator supports:

- ``predict(...)``
- ``shap_values(...)``
- ``feature_importances(...)``
- artifact-backed prediction via ``predict_from_artifact(...)``
- fitted training summaries via ``best_iteration_``, ``best_score_``,
  ``n_estimators_``, ``evals_result_``, and ``fit_timing_``

Continuous features
-------------------

AlloyGBM currently supports three continuous binning strategies:

- ``linear``
- ``rank``
- ``quantile``

``linear`` is the default. ``quantile`` is often a better choice when feature
distributions are strongly skewed.

Dense array-like inputs
-----------------------

The Python bridge supports optimized paths for array-like objects exposing:

- ``to_numpy``
- ``to_list``
- ``tolist``

You do not need to eagerly convert everything to nested Python lists just to use
the native path.
