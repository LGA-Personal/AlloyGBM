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
- ``dro_metric: str = "wasserstein"`` -- the only accepted v0.6.0 value. It
  denotes a Wasserstein-inspired closed-form robust counterpart over leaf
  gradient uncertainty.

This is not a full Wasserstein optimizer over raw feature/target
distributions. Inference speed is unchanged because robust scalar leaf values
are stored directly in the artifact. ``leaf_solver="dro"`` works on all three
estimators, composes with ``training_mode="morph"``, and requires
``leaf_model="constant"`` in v0.6.0.

Piecewise-linear leaves
-----------------------

- ``leaf_model: str = "constant"``

  - ``"constant"`` (default) -- standard scalar leaf value, identical to all
    prior AlloyGBM behaviour.
  - ``"linear"`` -- each leaf stores a small linear model
    ``f_s(x) = b_s + Σ α_j x_j`` (up to 8 regressors per leaf, inherited from
    the split path's feature indices; the per-leaf cap is internal and not
    user-tunable in v0.5.0). Optimal weights are solved in closed form via the
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
- SHAP (``shap_values``, ``feature_importances``) currently raises an error for
  ``leaf_model="linear"`` artifacts. Use ``"constant"`` if you need SHAP.
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

- ``fit(X, y, *, sample_weight=None, eval_set=None, eval_sample_weight=None, group=None, eval_group=None, eval_time_index=None, categorical_feature_values=None, time_index=None)``
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
- ``evals_result_`` -- shaped like ``{"train": {"rmse": [...]}, "validation": {"rmse": [...]}}``
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
