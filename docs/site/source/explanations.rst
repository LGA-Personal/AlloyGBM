Feature importances and SHAP
============================

AlloyGBM exposes SHAP-based explanation methods from the Python API.

Local explanations
------------------

Use ``shap_values(...)`` to get per-row, per-feature attributions:

.. code-block:: python

   from alloygbm import GBMRegressor

   model = GBMRegressor(deterministic=True, seed=7)
   model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 1.0, 2.0, 3.0])

   values = model.shap_values([[1.5], [2.5]])
   print(values)

To retrieve the expected value alongside the SHAP matrix:

.. code-block:: python

   expected_value, values = model.shap_values(
       [[1.5], [2.5]],
       include_expected_value=True,
   )

Global importance
-----------------

Use ``feature_importances(...)`` to aggregate SHAP importances across rows:

.. code-block:: python

   importance = model.feature_importances([[0.5], [1.5], [2.5]])
   print(importance)

Current expectations
--------------------

- ``shap_values(...)`` returns one attribution per feature for each input row
- ``feature_importances(...)`` returns ``(feature_name, importance)`` tuples
- feature names currently default to generated names such as ``f0`` and ``f1``

.. figure:: _static/shap_tree_path_example.png
   :alt: SHAP explanation example showing a highlighted decision-tree path and additive feature contributions to a prediction.
   :width: 90%
   :align: center

   Example of a prediction path and additive SHAP-style contribution breakdown.
