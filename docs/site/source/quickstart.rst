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

What happens during ``fit(...)``
--------------------------------

At a high level, AlloyGBM:

1. validates and normalizes the Python inputs
2. chooses a dense native fast path when possible
3. quantizes continuous features for native training
4. trains a native Rust artifact
5. stores the serialized artifact and a native predictor handle in the estimator

After fitting, the estimator supports:

- ``predict(...)``
- ``shap_values(...)``
- ``feature_importances(...)``
- artifact-backed prediction via ``predict_from_artifact(...)``

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
