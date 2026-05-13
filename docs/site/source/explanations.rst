Feature importances and SHAP
============================

AlloyGBM exposes SHAP-based explanation methods from the Python API, backed by
a native Rust TreeSHAP implementation.

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

TreeSHAP implementation
-----------------------

AlloyGBM uses the polynomial-time TreeSHAP algorithm for computing exact
Shapley values:

- No practical limit on the number of features
- Computation scales with tree complexity, not exponentially with feature count
- Results are exact, not approximate

The previous brute-force method (limited to 25 features) has been replaced by
TreeSHAP in ``0.2.0``.

Current expectations
--------------------

- ``shap_values(...)`` returns one attribution per feature for each input row
- ``feature_importances(...)`` returns ``(feature_name, importance)`` tuples
- Feature names are captured from training data when available, or default to
  generated names such as ``f0`` and ``f1``

Supported estimators
--------------------

SHAP explanations work with all three estimators:

- ``GBMRegressor.shap_values(...)``
- ``GBMClassifier.shap_values(...)``
- ``GBMRanker.shap_values(...)``

Leaf model compatibility
------------------------

``leaf_model="constant"`` artifacts produce exact SHAP attributions
satisfying ``Σ shap_values + expected_value == predict(x)``.

As of v0.7.1, ``shap_values(...)`` and ``feature_importances(...)`` also
accept ``leaf_model="linear"`` (piecewise-linear) artifacts and return a
*best-effort interventional* decomposition: the path-based machinery
attributes each leaf's "constant part"
``intercept + Σ wj · μj_global``, and per-leaf row deviations
``wj · (xj − μj_global)`` are credited directly to the regressor features.
Global per-feature means ``μj_global`` are captured at fit time and stored
in the artifact, so SHAP is self-contained — the original training data is
not required at explain time.

Exact additivity holds when SHAP's internal path walker reaches the same
leaf as the predictor. Currently SHAP compares raw feature values against
stump ``threshold_bin`` indices cast to ``f32``, while the predictor crate
converts those indices to float thresholds at load time. For scalar leaves
the divergence is masked; for linear leaves on continuous-feature artifacts
it can cause measurable additivity drift. The strict additivity check is
therefore relaxed for linear-leaf models so explanations remain available;
tightening path-walk alignment is queued for a follow-up release. See
``docs/limitations.md`` for the full caveat.

.. figure:: _static/shap_tree_path_example.png
   :alt: SHAP explanation example showing a highlighted decision-tree path and additive feature contributions to a prediction.
   :width: 90%
   :align: center

   Example of a prediction path and additive SHAP-style contribution breakdown.
