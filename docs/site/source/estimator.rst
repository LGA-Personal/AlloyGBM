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
  - per-round row sampling fraction; ignored when
    ``boosting_mode="goss"`` (GOSS uses gradient-based sampling
    instead).
- ``col_subsample: float = 1.0``
  - per-round feature sampling fraction
- ``quantile_alpha: float = 0.5``
  - Target quantile for ``"quantile"`` regression. Must be strictly in ``(0.0, 1.0)``.

Boosting mode
-------------

- ``boosting_mode: str = "standard"`` -- per-round sample-selection
  strategy.  Three values are accepted:

  - ``"standard"`` (default) -- uniform row subsampling under
    ``row_subsample``.  Byte-identical to v0.7.5 on every API
    surface.
  - ``"goss"`` -- LightGBM-style **G**\ radient-based **O**\ ne-**S**\ ide
    **S**\ ampling.  Each round, score rows by ``|gradient|``, keep
    the top ``goss_top_rate`` fraction, uniformly sample
    ``goss_other_rate`` fraction from the rest, and amplify the
    sampled-low rows' gradient and hessian by
    ``(n - top_n) / other_n`` (realized counts) so the histogram
    statistics remain an unbiased estimator of the full-data
    gradient sums.  Convergence is typically faster on data with a
    long-tailed gradient distribution (the canonical LightGBM
    advantage).
  - ``"dart"`` -- **D**\ ropouts meet **MART**.  Each round, drop a
    random subset of previously-trained trees, fit a new tree on the
    residuals of the dropped-out ensemble, then rescale the dropped
    trees + the new tree so the prediction sum stays unbiased.
    Reduces over-specialization of late trees; can improve
    generalization on noisy data.  Per-stump ``tree_weight: f32`` is
    persisted via a new ``DartTreeWeights`` artifact section.

- ``goss_top_rate: float = 0.2`` -- top-by-gradient kept fraction
  when ``boosting_mode="goss"``.  Must be in ``(0, 1)``.
- ``goss_other_rate: float = 0.1`` -- random-sample fraction from
  the remaining rows when ``boosting_mode="goss"``.  Must be in
  ``(0, 1)`` and ``goss_top_rate + goss_other_rate <= 1.0``.
- ``dart_drop_rate: float = 0.1`` -- per-tree drop probability per
  round when ``boosting_mode="dart"``.  Must be in ``(0, 1)``.
- ``dart_max_drop: int = 50`` -- cap on the number of trees dropped
  per round.  Must be ``>= 1``.
- ``dart_normalize_type: str = "tree"`` -- rescale policy after the
  new tree is fit.  ``"tree"`` mode sets new-tree weight to
  ``1 / (K + 1)`` and each dropped-tree weight to ``K / (K + 1)``;
  ``"forest"`` mode sets both to ``1 / (K + 1)`` (more aggressive
  rescale).
- ``dart_sample_type: str = "uniform"`` -- dropout sampling strategy.
  ``"uniform"`` picks each tree independently with probability
  ``dart_drop_rate``.  ``"weighted"`` biases dropout probability
  toward heavier-weight trees.

GOSS and DART are supported on the binary classifier / regression /
ranking single-output objective.  The multiclass softmax path
explicitly rejects non-``"standard"`` boosting modes pending
per-class gradient scoring (v0.10.x follow-up — applies to both
GOSS and DART).

As of v0.10.0, **DART + ``warm_start``** is supported on
``GBMRegressor``, binary ``GBMClassifier``, and ``GBMRanker``. The
continuation seeds ``dart_state.tree_weights`` from the prior model's
per-stump ``tree_weight`` snapshot and pre-populates the dropout
bookkeeping arrays so new-round dropouts can correctly subtract /
replay prior trees. Historical RNG-driven ``dropped_per_round`` is
intentionally not persisted; new rounds start fresh dropout
bookkeeping going forward.

.. code-block:: python

   base = GBMRegressor(n_estimators=10, boosting_mode="dart",
                       dart_drop_rate=0.1, seed=7)
   base.fit(X, y)

   cont = GBMRegressor(n_estimators=10, boosting_mode="dart",
                       dart_drop_rate=0.1, warm_start=True, seed=7)
   cont.fit(X, y, init_model=base)

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
- ``interaction_constraints: list[list[int]] | None = None`` --
  LightGBM-compatible interaction constraints.  Each inner list is a group
  of feature indices; any root-to-leaf path is restricted to splits on
  features from a single allowed group.  Features outside every group are
  unconstrained and may appear alongside any group.  Up to 64 groups per
  fit; enforced in both level-wise and leaf-wise growth.  ``None``
  disables the constraint (default).

Reproducibility
---------------

- ``seed: int = 0``
- ``deterministic: bool = True``

Continuous-feature controls
---------------------------

- ``continuous_binning_strategy: str = "quantile"``
- ``continuous_binning_max_bins: int = 256``

Supports up to 65,535 bins per feature. The default ``quantile`` strategy gives
more robust handling of skewed continuous features. Use ``linear`` when you want
equal-width bins for compatibility experiments or lower quantile-preprocessing cost.

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
- ``dro_metric: str = "wasserstein"`` -- the only accepted value today. It
  denotes a Wasserstein-inspired closed-form robust counterpart over leaf
  gradient uncertainty.

This is not a full Wasserstein optimizer over raw feature/target
distributions. Inference speed is unchanged because robust scalar leaf values
are stored directly in the artifact. ``leaf_solver="dro"`` works on all three
estimators, composes with ``training_mode="morph"``, and requires
``leaf_model="constant"``.

Factor-neutral boosting
-----------------------

- ``neutralization: str = "none"``

  - one of ``"none"``, ``"pre_target"``, ``"per_round_gradient"``, or
    ``"split_penalty"``

- ``factor_neutralization_lambda: float = 1e-6`` -- finite, non-negative ridge
  term added to ``F^T W F``.
- ``factor_penalty: float = 0.0`` -- finite, non-negative split exposure penalty
  scale. Only active for ``neutralization="split_penalty"``.
- ``factor_exposure_transform: str = "none"`` -- one of ``"none"``,
  ``"center"``, or ``"standardize"``. Applies column-wise preprocessing to
  fit-time ``factor_exposures`` before the projector and split-penalty
  calculations. When ``neutralization="split_penalty"`` and
  ``factor_penalty > 0``, the default effective transform is ``"standardize"``
  because the penalty scale depends on exposure units. Other neutralization
  modes still default to no transform.

Pass factors as fit-time data:

.. code-block:: python

   model = GBMRegressor(neutralization="per_round_gradient", seed=7)
   model.fit(X_train, y_train, factor_exposures=F_train)

``factor_exposures`` must be dense, row-major, finite, and shaped
``(n_rows, n_factors)``. It is fit data, not constructor state, so sklearn
cloning remains clean and large matrices are not embedded in estimator params.
With ``factor_exposure_transform="center"``, each factor column is
mean-centered. With ``"standardize"``, each column is centered and divided by
its population standard deviation; near-constant columns use a safe scale of
``1.0``. When active ``split_penalty`` uses the default constructor value
(``"none"``), the fitted diagnostics report ``transform="standardize"`` to
reflect the effective preprocessing actually applied. The fitted estimator records the training-column ``means`` and
``stds`` in ``factor_exposure_diagnostics_``. After fitting, the same
diagnostics dict also reports post-fit prediction exposure against the
transformed fit-time factors: ``prediction_exposure_dot`` (``F^T y_hat``, per
factor), ``prediction_exposure_abs``, and ``prediction_exposure_l2``.

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

Hessians are unchanged. For squared error, where the Hessian is constant, this
is an exact gradient-space residualization. For binary, GLM, ranking, and other
non-constant-Hessian objectives, the Newton numerator is projected while the
denominator remains the objective Hessian, so leaf values are not simply the
projection of an unneutralized model's leaf values. This mode is supported for
regression, binary classification, multiclass, and ranking. For multiclass,
each class-gradient column is projected independently against the same factor
projector.

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
     - supported
     - supported
     - supported

This is a training-time regularization tool. It does not guarantee
prediction-time zero exposure unless predictions are neutralized against
evaluation-time factors outside the model.

Exposure matrices are not persisted in the estimator or artifact (they
would balloon the model size and surface sensitive data). As of v0.7.1
neutralized warm-start and ``init_model`` continuation are supported: the
caller must supply the same ``factor_exposures`` matrix used for the
initial fit so the projection has the same column space. Omitting
``factor_exposures`` on a resumed fit raises a contract error.

``pre_target`` neutralization is idempotent under repeated residualization
against the same exposures, so warm-start continuation residualizes the
original targets again on the resumed fit and trains on the same
target stream as a fresh ``N + M``-round fit.

Piecewise-linear leaves
-----------------------

- ``leaf_model: str = "constant"``

  - ``"constant"`` (default) -- standard scalar leaf value, identical to all
    prior AlloyGBM behaviour.
  - ``"linear"`` -- each leaf stores a small linear model
    ``f_s(x) = b_s + Σ α_j x_j`` (up to 8 regressors per leaf, inherited from
    the split path's feature indices; the per-leaf cap is internal and not
    currently user-tunable). Optimal weights are solved in closed form via the
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
- SHAP (``shap_values``, ``feature_importances``) supports
  ``leaf_model="linear"`` with strict additivity as of v0.7.4: the
  reconstruction satisfies ``atol + rtol·|predict(x)|`` (default
  ``1e-5 + 1e-4·|predict(x)|``) on the default predictor-aligned binning
  path.  See :doc:`explanations` for the decomposition and
  ``docs/limitations.md`` for the legacy-non-binning exemption.
- ``leaf_model="linear"`` composes with ``training_mode="morph"``.

Multi-label ranking
-------------------

``MultiLabelGBMRanker`` is a unified multi-output ranking estimator: ``y``
has shape ``(n_rows, n_labels)`` and ``predict`` returns scores with the
same column layout.  As of v0.7.1 the wrapper trains one independent
:class:`GBMRanker` per label using a shared ``group`` (and optional shared
``factor_exposures``) so every per-label fit observes the same query
structure.  Each per-label ranker independently picks up every existing
:class:`GBMRanker` feature (warm-start, neutralization, MorphBoost, PL
leaves, DRO, interaction constraints, custom eval metrics).

``ranking_objective`` may be a single string (applied to every label) or
a list of length ``n_labels`` for heterogeneous objectives.  ``save_model``
serialises every per-label ranker into a single ``.mlrk`` bundle via
``pickle.HIGHEST_PROTOCOL``.

Joint shared-tree multi-label boosting is deferred to v0.10.0
(paired with the K-output shared-histogram primitive); see
``docs/limitations.md`` for the upgrade-path caveat.

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
- ``factor_exposure_diagnostics_`` -- ``None`` unless neutralization is active.
  Otherwise a dict with the selected ``transform`` plus per-factor training
  ``means`` and ``stds`` used by ``factor_exposure_transform``. After fit it
  also includes ``prediction_exposure_dot``, ``prediction_exposure_abs``, and
  ``prediction_exposure_l2``, computed from transformed fit-time exposures and
  fitted training predictions. Joint multi-label models report the dot/abs
  entries as factor-by-label matrices.
- ``fit_timing_``
- ``feature_names_`` -- captured from training data or auto-generated

Regression objectives (v0.11.1+)
--------------------------------

``GBMRegressor`` accepts the following values for the ``objective`` kwarg:

- ``"squared_error"`` (default) -- standard least-squares regression.
- ``"poisson"`` -- log-link Poisson regression for count targets.
  Targets must be ``>= 0``. ``predict()`` returns ``exp(raw)``.
- ``"gamma"`` -- log-link Gamma regression for strictly-positive
  continuous targets. Targets must be ``> 0``. ``predict()`` returns
  ``exp(raw)``.
- ``"tweedie"`` -- log-link compound Poisson-gamma regression for
  ``1 < variance_power < 2``. Useful for insurance/claims data with a
  mass at zero and a positive tail. Set
  ``tweedie_variance_power=1.5`` (or another value in ``(1, 2)``).
  Targets must be ``>= 0``. ``predict()`` returns ``exp(raw)``.
- ``"quantile"`` -- pinball loss regression with parameter ``quantile_alpha``.
  Uses a proxy Hessian ``h_i = w_i`` (sample weight) during split-finding,
  and performs an empirical quantile leaf refinement step at the end of
  each round acting on the full dataset.
- Custom callable -- any user-supplied
  ``(predictions, targets) → (gradients, hessians)`` function.

``tweedie_variance_power: float = 1.5`` -- only used when
``objective="tweedie"``. Must satisfy ``1 < p < 2``. For ``p = 1`` use
``objective="poisson"``; for ``p = 2`` use ``objective="gamma"``.

``quantile_alpha: float = 0.5`` -- quantile to estimate when ``objective="quantile"``.
Must be in ``(0, 1)``.

All three GLM objectives compose with ``boosting_mode="dart"``,
``boosting_mode="goss"``, warm-start, ``tree_growth="leaf"``,
``neutralization="per_round_gradient"`` /
``neutralization="split_penalty"``, and ``training_mode="morph"``.
``neutralization="pre_target"`` remains squared-error-only.

The ``"quantile"`` objective is supported in combination with DART, MorphBoost, and piecewise-linear leaves (``leaf_model="linear"``). It is explicitly rejected when combined with classification, ranking, or joint multi-output training.

Three deviance metrics in ``alloygbm.evaluation`` partner with the new
objectives: ``poisson_deviance``, ``gamma_deviance``, and
``tweedie_deviance``.

SHAP interaction values (v0.11.0+)
----------------------------------

``GBMRegressor.shap_interaction_values(X)`` returns pairwise SHAP
attributions as an ``(n_rows, n_features, n_features)`` tensor in
``O(T · L · D² · M)`` time via Lundberg et al. (2020) Algorithm 2.
See :doc:`explanations` for the full contract and scope limits.

Recommended usage pattern
-------------------------

For most users:

- start with ``training_policy="auto"``
- keep ``deterministic=True`` during evaluation
- use time-aware validation for temporal or panel-like problems
- use the benchmark suite to compare profile shapes rather than trusting a
  single run
