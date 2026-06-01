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
accept ``leaf_model="linear"`` (piecewise-linear) artifacts and return an
*interventional* decomposition: the path-based machinery attributes each
leaf's "constant part" ``intercept + Σ wj · μj_global``, and per-leaf row
deviations ``wj · (xj − μj_global)`` are credited directly to the regressor
features along the row's path.  Global per-feature means ``μj_global`` are
captured at fit time and stored in the artifact, so SHAP is self-contained
— the original training data is not required at explain time.

As of v0.7.4, strict additivity
(``Σ shap_values + expected_value == predict(x)`` within
``atol + rtol·|predict(x)|``, default ``1e-5 + 1e-4·|predict(x)|``) also
holds for ``leaf_model="linear"`` artifacts on the default
predictor-aligned binning path.  v0.7.3's ``BinningContext`` aligns the
SHAP path walker to the predictor's float thresholds; v0.7.4 credits
``Σⱼ wⱼ·(xⱼ − μⱼ)`` at every visited node along the row's path (matching
how ``predict`` accumulates ``leaf.eval_row(row)`` at each visited node).
The legacy non-binning SHAP path retains a best-effort exemption for
linear leaves only.

.. figure:: _static/shap_tree_path_example.png
   :alt: SHAP explanation example showing a highlighted decision-tree path and additive feature contributions to a prediction.
   :width: 90%
   :align: center

   Example of a prediction path and additive SHAP-style contribution breakdown.

SHAP interaction values (v0.11.0+)
----------------------------------

``GBMRegressor.shap_interaction_values(X)`` returns pairwise SHAP
attributions as an ``(n_rows, n_features, n_features)`` tensor.
Implements Lundberg et al. (2020) "From local explanations to global
understanding with explainable AI for trees" Algorithm 2 in polynomial
time ``O(T · L · D² · M)`` where ``M`` is the feature count.

Invariants (within ``atol = 1e-5 + rtol = 1e-4 · |predict(x)|``):

- **Symmetric**: ``values[r][i][j] == values[r][j][i]``.
- **Row-marginal**: ``Σ_j values[r][i][j] == shap_values(X)[r][i]``.
- **Full additivity**: ``Σ_i Σ_j values[r][i][j] + expected_value
  == predict(x)``.

The diagonal ``values[r][i][i]`` is the "main effect" of feature ``i``
after subtracting all off-diagonal interactions.  Pass
``include_expected_value=True`` to receive a ``(expected_value,
interactions)`` tuple.

Scope limits:

- ``leaf_model="linear"`` artifacts are supported. The row-dependent linear
  deviation terms are attributed directly to the main effect (the diagonal of
  the interaction matrix) to preserve both row-marginal and full additivity
  invariants.
- ``GBMClassifier.shap_values(X)`` and
  ``GBMClassifier.shap_interaction_values(X)`` return a list of ``K`` arrays
  — one per class logit (v0.12.6+).
- ``MultiLabelGBMRanker.shap_values(X)`` and
  ``MultiLabelGBMRanker.shap_interaction_values(X)`` return a list of
  ``n_labels`` arrays — one per output (v0.12.6+).  Joint mode routes through
  per-output Rust entry points; independent mode fans out to per-label
  ``GBMRanker`` calls.
