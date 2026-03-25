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

Suggested figure placeholder
----------------------------

.. note::

   Suggested diagram to add here:

   - filename: ``_static/shap_tree_path_example.png``
   - placement: directly below this note
   - concept: a small decision tree with one highlighted prediction path and
     a side panel showing the additive decomposition

   This would make the local-explanation story much easier to understand for
   first-time users.
