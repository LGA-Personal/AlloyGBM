GBMRegressor
============

``GBMRegressor`` is the main Python estimator for regression in AlloyGBM.

Core parameters
---------------

- ``learning_rate: float = 0.1``
  - additive update step size
- ``max_depth: int = 6``
  - maximum tree depth
- ``n_estimators: int = 6``
  - requested boosting rounds
- ``row_subsample: float = 1.0``
  - per-round row sampling fraction
- ``col_subsample: float = 1.0``
  - per-round feature sampling fraction

Stopping and policy controls
----------------------------

- ``early_stopping_rounds: int | None = None``
- ``min_validation_improvement: float = 0.0``
- ``training_policy: str = "auto"``

``training_policy="auto"`` applies dataset-aware heuristics and is the
recommended default for practical use. ``manual`` is more appropriate for
controlled ablation work.

Early stopping is explicit-only. If ``early_stopping_rounds`` is set, call
``fit(..., eval_set=(X_valid, y_valid))``.

Leaf and split controls
-----------------------

- ``min_data_in_leaf: int = 1`` -- when ``training_policy="auto"``, the engine
  may increase this based on dataset size but will never reduce it below the
  value you set.
- ``lambda_l1: float = 0.0``
- ``lambda_l2: float = 0.0``
- ``min_child_hessian: float = 0.0``
- ``min_split_gain: float = 0.0`` -- minimum gain required for a split. The auto
  policy may set this adaptively.

These map directly to native training controls instead of relying on
environment-variable overrides.

Tree growth strategy
--------------------

- ``tree_growth: str = "level"`` -- ``level`` (depth-first) or ``leaf``
  (best-first, similar to LightGBM)
- ``max_leaves: int | None = None`` -- maximum leaves for leaf-wise growth

Constraints
-----------

- ``monotone_constraints: list[int] | dict[int, int] | None = None`` --
  constrain features to monotone increasing (+1), decreasing (-1), or
  unconstrained (0)
- ``feature_weights: list[float] | dict[int, float] | None = None`` --
  per-feature importance weights influencing split selection

Reproducibility
---------------

- ``seed: int = 0``
- ``deterministic: bool = True``

Continuous-feature controls
---------------------------

- ``continuous_binning_strategy: str = "linear"``
- ``continuous_binning_max_bins: int = 256``

Supports up to 65,535 bins per feature. Use ``quantile`` when you need more
robust handling of skewed continuous features.

Categorical support
-------------------

- ``categorical_feature_index: int | None = None`` -- single column (legacy)
- ``categorical_feature_indices: list[int] | None = None`` -- multiple columns
- ``categorical_smoothing: float = 20.0``
- ``categorical_min_samples_leaf: int = 1``
- ``categorical_time_aware: bool = False``
- ``max_cat_threshold: int = 0`` -- maximum category cardinality for native
  categorical splits. When a categorical feature has at most this many
  unique values, AlloyGBM uses the Fisher-sort algorithm for O(K log K)
  optimal binary partition with O(1) bitset prediction. Features exceeding
  the threshold fall back to target encoding. Default 0 disables native
  splits.

DRO leaf solver
---------------

- ``leaf_solver: str = "standard"`` -- ``"standard"`` keeps the usual scalar
  Newton leaf update; ``"dro"`` enables a fast robust scalar update that
  penalizes weak leaf signal by within-leaf gradient dispersion.
- ``dro_radius: float = 0.05`` -- non-negative penalty scale. ``0.0`` preserves
  standard-leaf predictions while recording DRO metadata.
- ``dro_metric: str = "wasserstein"`` -- the only accepted v0.7.0 value. It
  denotes a Wasserstein-inspired closed-form robust counterpart over leaf
  gradient uncertainty.

This is not a full Wasserstein optimizer over raw feature/target
distributions. Inference speed is unchanged because robust scalar leaf values
are stored directly in the artifact. ``leaf_solver="dro"`` works on all three
estimators, composes with ``training_mode="morph"``, and requires
``leaf_model="constant"`` in v0.7.0.

Factor-neutral boosting
-----------------------

- ``neutralization: str = "none"``

  - one of ``"none"``, ``"pre_target"``, ``"per_round_gradient"``, or
    ``"split_penalty"``

- ``factor_neutralization_lambda: float = 1e-6`` -- finite, non-negative ridge
  term added to ``F^T W F``.
- ``factor_penalty: float = 0.0`` -- finite, non-negative split exposure penalty
  scale. Only active for ``neutralization="split_penalty"``.

Pass factors as fit-time data:

.. code-block:: python

   model = GBMRegressor(neutralization="per_round_gradient", seed=7)
   model.fit(X_train, y_train, factor_exposures=F_train)

``factor_exposures`` must be dense, row-major, finite, and shaped
``(n_rows, n_factors)``. It is fit data, not constructor state, so sklearn
cloning remains clean and large matrices are not embedded in estimator params.

Mode semantics:

``neutralization="none"`` preserves current behavior and ignores
``factor_exposures`` unless a non-``None`` matrix is provided with an inactive
mode, in which case Python raises a clear validation error to prevent silent
user mistakes.

``neutralization="pre_target"`` residualizes the regression target once before
training:

.. code-block:: text

   y_perp = y - F (F^T W F + lambda I)^-1 F^T W y

This mode is supported for ``GBMRegressor`` only. It is rejected for
classification and ranking because target residualization is not well-defined
for class labels or ranking relevance. ``eval_set`` is also rejected for
``pre_target`` in this release because the public API does not yet accept
validation-set factor exposures to residualize validation targets consistently.

``neutralization="per_round_gradient"`` projects objective gradients before
each boosting round:

.. code-block:: text

   g_perp = g - F (F^T W F + lambda I)^-1 F^T W g

Hessians are unchanged. This mode is supported for regression, binary
classification, multiclass, and ranking. For multiclass, each class-gradient
column is projected independently against the same factor projector.

``neutralization="split_penalty"`` includes per-round gradient projection and
subtracts a factor-load penalty from split gain:

.. code-block:: text

   penalty = factor_penalty * || F_L^T update_L + F_R^T update_R ||^2 / max(row_count, 1)
   gain_final = gain_after_existing_modes - penalty

For scalar leaves, ``update_L`` and ``update_R`` are the candidate scalar leaf
values before any final MorphBoost depth/iteration leaf scaling. For DRO
leaves, the scalar values use the DRO effective gradients. For MorphBoost, the
order is: project gradients, compute standard/DRO gradient gain, blend
MorphBoost information score, subtract factor penalty, then apply MorphBoost
leaf scaling when storing leaves. ``split_penalty`` performs additional
factor-exposure work during split search and should be treated as the slowest
neutralization mode until production-scale benchmarks justify stronger claims.

Compatibility:

.. list-table::
   :header-rows: 1

   * - Feature
     - pre_target
     - per_round_gradient
     - split_penalty
   * - ``GBMRegressor``
     - supported
     - supported
     - supported
   * - ``GBMClassifier``
     - rejected
     - supported
     - supported
   * - ``GBMRanker``
     - rejected
     - supported
     - supported
   * - ``training_mode="morph"``
     - supported
     - supported
     - supported
   * - ``leaf_solver="dro"``
     - supported
     - supported
     - supported
   * - ``leaf_model="linear"``
     - supported
     - supported
     - rejected
   * - warm start
     - rejected in this release
     - rejected in this release
     - rejected in this release

This is a training-time regularization tool. It does not guarantee
prediction-time zero exposure unless predictions are neutralized against
evaluation-time factors outside the model.

Exposure matrices are not persisted in the estimator or artifact. Because the
artifact cannot prove factor compatibility yet, neutralized warm-start and
``init_model`` continuation are rejected in this release.

Piecewise-linear leaves
-----------------------

- ``leaf_model: str = "constant"``

  - ``"constant"`` (default) -- standard scalar leaf value, identical to all
    prior AlloyGBM behaviour.
  - ``"linear"`` -- each leaf stores a small linear model
    ``f_s(x) = b_s + Σ α_j x_j`` (up to 8 regressors per leaf, inherited from
    the split path's feature indices; the per-leaf cap is internal and not
    user-tunable in v0.7.0). Optimal weights are solved in closed form via the
    ridge regression ``α* = -(XᵀHX + λI)⁻¹ Xᵀg``, regularised by ``lambda_l2``.

Empirically, ``"linear"`` converges in fewer rounds on data with linear
within-node residual structure (~10× faster on linearly-structured datasets,
+3.5% RMSE on California Housing, +1.75pp accuracy on Breast Cancer), at a
2–8× per-round training overhead. Recommended ``lambda_l2 >= 0.01`` for weight
stability.

Limitations:

- Native-bitset categorical splits (``max_cat_threshold > 0``) fall back to
  constant leaves at the categorical split node; descendant leaves below the
  split use linear leaves on remaining numeric regressors.
- SHAP (``shap_values``, ``feature_importances``) supports ``leaf_model="linear"``
  as of v0.7.1, returning a best-effort interventional decomposition. Exact
  additivity holds for path-walk-aligned artifacts; on continuous-feature
  models the reconstruction can drift slightly because SHAP's internal path
  walker still compares against bin-index thresholds. See
  :doc:`explanations` and ``docs/limitations.md`` for details.
- ``leaf_model="linear"`` composes with ``training_mode="morph"``.

MorphBoost (Adaptive Split Criterion)
-------------------------------------

GBMRegressor (and the classifier / ranker subclasses) support an opt-in
MorphBoost training mode. See :doc:`morphboost` for the full guide.

- ``training_mode: str = "auto"`` -- one of ``"auto"`` (default), ``"manual"``,
  or ``"morph"``.
- ``morph_rate: float = 0.1`` -- per-iteration leaf shrinkage rate.
- ``evolution_pressure: float = 0.2`` -- EMA-driven gain shaping strength.
- ``morph_warmup_iters: int = 5`` -- rounds before the morph blend engages.
- ``info_score_weight: float = 0.3`` -- mixing weight for the
  information-theoretic gain term.
- ``depth_penalty_base: float = 0.9`` -- depth-based leaf penalty base.
- ``balance_penalty: bool = True`` -- whether to penalize imbalanced splits.
- ``lr_schedule: str = "constant"`` -- per-iteration LR schedule
  (``"constant"`` or ``"warmup_cosine"``); independent of ``training_mode``.
- ``lr_warmup_frac: float = 0.1`` -- linear-warmup fraction when
  ``lr_schedule="warmup_cosine"``.

Warm-starting
-------------

- ``warm_start: bool = False`` -- when ``True``, ``fit()`` continues from the
  previously fitted model

Diagnostics
-----------

- ``store_node_stats: bool = False``

This stores optional node-level debug statistics inside the artifact for later
analysis. It is not required for ordinary prediction.

Main methods
------------

- ``fit(X, y, *, sample_weight=None, eval_set=None, eval_sample_weight=None, group=None, eval_group=None, eval_time_index=None, categorical_feature_values=None, time_index=None, factor_exposures=None)``
- ``predict(X)``
- ``shap_values(X, *, include_expected_value=False)``
- ``feature_importances(X, *, method="shap")``
- ``predict_from_artifact(artifact_bytes, X)``
- ``save_model(path)``
- ``load_model(path)`` (classmethod)
- ``artifact_bytes`` -- property returning the raw artifact bytes
- ``score(X, y)``

Important ``fit(...)`` rules:

- ``early_stopping_rounds`` requires ``eval_set``
- ``eval_time_index`` requires ``eval_set``
- ``categorical_time_aware=True`` requires ``time_index`` during training and
  ``eval_time_index`` for validation when ``eval_set`` is used
- ``sample_weight`` applies per-sample weights to the training loss

Post-fit attributes
-------------------

After fitting, the estimator may expose:

- ``best_iteration_``
- ``best_score_``
- ``n_estimators_``
- ``rounds_completed_``
- ``stop_reason_``
- ``evals_result_`` -- shaped like ``{"train": {"rmse": [...]}, "validation": {"rmse": [...]}}``
- ``diagnostics_per_round_`` -- list of per-round dicts containing
  ``gradient_l2_norm``, ``gradient_variance``, ``hessian_l2_norm``,
  ``original_gradient_l2_norm``, ``projected_gradient_l2_norm``,
  ``neutralization_effectiveness``, ``n_active_rows``, ``n_active_features``.
  The three projection-related entries are ``None`` unless factor
  neutralization (``per_round_gradient`` or ``split_penalty``) is configured;
  ``pre_target`` mode never projects per round and therefore omits them.
- ``fit_timing_``
- ``feature_names_`` -- captured from training data or auto-generated

Recommended usage pattern
-------------------------

For most users:

- start with ``training_policy="auto"``
- keep ``deterministic=True`` during evaluation
- use time-aware validation for temporal or panel-like problems
- use the benchmark suite to compare profile shapes rather than trusting a
  single run
