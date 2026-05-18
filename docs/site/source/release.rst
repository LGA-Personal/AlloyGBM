Release and platform policy
===========================

AlloyGBM ``0.8.0`` release notes and platform policy.

What's new in 0.8.0
-------------------

Feature release rolling up the three remaining v0.7.x carry-forward
items.  This section grows commit by commit on the ``v0.8.0-features``
branch.

**SHAP strict additivity on the mixed linear-rank binning path
(Limitation 4):**

- When ``continuous_binning_strategy="linear"`` triggered per-feature
  rank-based binning on at least one column (gated by the
  ``ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK`` experiment flag), the
  Python ``shap_values()`` flow used to fall back to the legacy
  quantize-then-walk SHAP path which exempts ``leaf_model="linear"``
  artifacts from strict additivity.
- v0.8.0 adds a new ``BinningContext::LinearRank`` variant to
  ``crates/shap/src/lib.rs``.  It carries per-feature sorted unique
  values, global ``feature_mins`` / ``feature_maxs``, and
  ``max_data_bin``.  At the ``explain_rows_from_model`` entry point
  SHAP internally quantizes the raw input rows to bin indices using
  exactly the same rules as
  ``predict_dense_quantized_linear_rank`` (linear quantize for
  unflagged features, rank quantize for flagged features, both with
  ``round_half_away_from_zero`` clamped to ``[0, max_data_bin]``) and
  dispatches the remainder of the path-walker with
  ``BinningContext::PreBinned`` semantics.  Both tree traversal and
  PL-leaf evaluation now operate in the same bin-index space the
  predictor uses, so strict additivity holds for
  ``leaf_model="linear"`` (and constant leaves stay correct).
- The Python ``_shap_binning_kwargs()`` helper returns
  ``binning_kind="linear_rank"`` whenever any per-feature rank flag is
  set; ``GBMClassifier`` and ``GBMRanker`` inherit the fix from
  ``GBMRegressor._shap_binning_kwargs``.
- Verified by
  ``bindings/python/tests/test_shap_linear_rank_strict_additivity.py``
  (architectural contract + strict additivity for both
  ``leaf_model="constant"`` and ``leaf_model="linear"``).  Closes
  Limitation 4.

What's new in 0.7.5
-------------------

Bug-fix release.  Closes Limitation 5 from v0.7.4 — the pre-existing
TreeSHAP polynomial-path additivity drift on trees with a feature
appearing more than once on a root-to-leaf path.  No user-visible API
breakage.

**TreeSHAP polynomial-path strict additivity:**

- The Rust port of TreeSHAP's polynomial-time algorithm in
  ``crates/shap/src/lib.rs::ts_unextend_path`` was shifting the entire
  ``PathElement`` struct (including ``pweight``) when removing a
  duplicate feature from the path.  This clobbered the pweights that
  the unwind loop had just carefully recomputed in place.  The
  reference implementation in ``slundberg/shap``
  (``shap/explainers/pytree.py``) stores the four path fields as four
  parallel arrays and only shifts the first three
  (``feature_index``, ``zero_fraction``, ``one_fraction``),
  preserving pweights.  Pre-existing in v0.7.3 and earlier; uncovered
  during v0.7.4 PR #27 review and pinned with an ``@xfail(strict=True)``
  test at that time pending this v0.7.x follow-up.
- The fix shifts the three fields explicitly and leaves ``pweight``
  alone.  Strict additivity now holds end-to-end on the polynomial
  path.
- Coverage: a synthetic full-tree sweep
  (``tree_shap_polynomial_path_matches_brute_force_on_full_trees``)
  covers depths 2-7 × n_features {2,3,5,8,12} including all
  configurations that force path-duplicate features, asserting
  polynomial matches brute-force per-feature within 1e-5.  The
  formerly ``@xfail(strict=True)`` regression
  ``test_strict_additivity_via_tree_shap_polynomial_path`` in
  ``bindings/python/tests/test_shap_pl_strict_additivity.py`` now
  passes as a regular test.

**Documentation:**

- ``docs/limitations.md``: Limitation 5 promoted to Resolved.
- Other documented v0.7.x follow-ups (mixed linear-rank SHAP path,
  GOSS+DART, joint multi-label ranking, shared-histogram engine)
  remain deferred to v0.8.0.

What's new in 0.7.4
-------------------

Bug-fix release.  Closes the remaining v0.7.x carryover documented in
``docs/limitations.md`` for SHAP strict additivity on
``leaf_model="linear"`` artifacts.  No user-visible API breakage.

**SHAP strict additivity for piecewise-linear leaves:**

- Pre-v0.7.4 ``distribute_linear_terms_for_row`` credited the per-feature
  deviation ``Σⱼ wⱼ·(xⱼ − μⱼ)`` only at each tree's terminal leaf.  The
  predictor accumulates ``leaf.eval_row(row)`` at **every visited node**
  along the row's path, so SHAP was uncrediting one
  ``Σⱼ wⱼ·(xⱼ − μⱼ)`` per internal node per tree per row — producing
  additivity gaps on the order of the predictions themselves
  (~3.85 on linear-data predictions of magnitude ~10 with
  ``n_estimators=100, max_depth=6``).
- v0.7.4 walks the full row path and credits the linear deviation at
  every visited leaf.  The brute-force Shapley and TreeSHAP polynomial
  paths share the helper so both get the fix.
- The ``model_has_linear_leaves`` exemption in ``verify_additivity`` is
  now gated on ``binning.is_none()``, so the predictor-aligned
  ``BinningContext`` callers — i.e. the default Python path for
  continuous features — get the strict
  ``atol + rtol·|predicted|`` tolerance check.
- Coverage: 44 new regression tests in
  ``bindings/python/tests/test_shap_pl_strict_additivity.py``
  exercising every binning strategy × max-bin width × ``lambda_l2`` ×
  ``max_depth`` × ``n_estimators`` combination, plus
  ``training_mode="manual"`` and ``"morph"``,
  ``interaction_constraints``, :class:`~alloygbm.GBMRanker`,
  :class:`~alloygbm.GBMClassifier` (via the internal Rust check, since
  the raw margin is not exposed in Python),
  ``feature_importances`` (brute-force exact path), and mixed
  scalar+linear-leaf artifacts.  Strict additivity holds on the default
  predictor-aligned binning path for any model that dispatches to the
  brute-force exact Shapley path
  (``distinct_split_feature_count <= MAX_EXACT_SPLIT_FEATURES = 25``).
  Larger models that trigger the polynomial-TreeSHAP path are subject
  to a pre-existing additivity drift documented as Limitation 5 (also
  present in v0.7.3 and earlier).

**Documentation:**

- Limitation 4 (new): SHAP on the mixed linear-rank binning path —
  ``continuous_binning_strategy="linear"`` with per-feature rank-based
  binning falls back to the legacy non-binning SHAP entry point,
  triggering the ``leaf_model="linear"`` exemption.  Narrow edge case;
  deferred to v0.8.0.
- Limitation 5 (new): pre-existing TreeSHAP polynomial-path additivity
  drift on large gradient-trained trees (>= 30 distinct split features,
  depth >= 6).  Uncovered during PR #27 review; investigated but not
  isolated in minimal Rust reproductions.  Coverage pinned by
  ``@xfail(strict=True)`` regression test
  (``test_strict_additivity_via_tree_shap_polynomial_path``) so the
  eventual fix flips the xfail to a regular pass.

**Documented for v0.7.x follow-ups (deferred to 0.8.0):**

- Joint shared-tree multi-label ranking.  The current
  :class:`~alloygbm.MultiLabelGBMRanker` trains K independent per-label
  rankers under a unified API and is numerically equivalent to training
  each label separately.  Joint shared-tree training lands alongside
  the v0.8.0 shared-histogram speedup where the architectural change
  has a real performance story.

What's new in 0.7.3
-------------------

Bug-fix release.  Closes the four limitations queued in v0.7.2 and
clears RUSTSEC-2025-0020.  No user-visible API breakage.

**SHAP additivity tolerance:**

- The internal additivity check now uses
  ``atol + rtol * |predicted|`` (atol=1e-5, rtol=1e-4) instead of a
  fixed ``1e-5`` absolute bound.  Larger explanation batches —
  ``feature_importances()`` over ~1000 rows of California Housing with
  ``n_estimators=200`` was the public-facing reproducer — no longer
  raise spurious ``RuntimeError`` on healthy ``leaf_model="constant"``
  artifacts.

**SHAP path-walker uses predictor-aligned float thresholds:**

- New ``shap::BinningContext`` (``Linear``, ``Quantile``, ``PreBinned``)
  plus four PyO3 entry points (``shap_explain_rows_with_binning``,
  ``shap_global_importance_with_binning``, plus dense variants).  When
  a binning context is provided, the path walker compares
  ``feature_value < float_threshold`` (matching the predictor's
  ``convert_bin_thresholds_to_float*``) instead of the legacy
  ``feature_value <= split.threshold_bin as f32``.  Eliminates the
  path-walk vs. predict-path divergence on continuous features for
  scalar-leaf artifacts.
- :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`,
  and :class:`~alloygbm.GBMRanker` now pass feature mins / maxs / cuts
  / binning kind into SHAP automatically.

**MorphBoost warm-start now persists EMA:**

- MorphMetadata artifact section bumped to v2 with appended
  ``Vec<GradientEmaStats>`` per class.  :class:`WarmStartState` and
  :class:`MultiClassWarmStartState` gain
  ``initial_ema_stats: Option<Vec<GradientEmaStats>>``.  Both
  single-class and multiclass training loops seed the fresh
  ``MorphState.ema_stats`` from this snapshot, so resuming a
  MorphBoost-trained model via ``init_model=`` no longer restarts the
  EMA cold.
- v1 artifacts decode with empty ``ema_stats``; the engine falls back
  to ``MorphState::new`` cold initialization, preserving prior
  behaviour for legacy artifacts.

**PyO3 0.23 → 0.24 (clears RUSTSEC-2025-0020):**

- Bumps ``pyo3 = "0.24"`` and ``numpy = "0.24"``.  The bindings were
  already on the ``Bound<>``-first API — zero source changes needed.
  ``deny.toml`` and ``.github/workflows/security-audit.yml`` no longer
  ignore RUSTSEC-2025-0020.

**Limitations documented for the next release:**

- SHAP additivity for piecewise-linear leaves on continuous features
  remains exempted from the strict check (linear weights and
  ``feature_baseline`` are still trained in bin space).
- Joint shared-tree multi-label boosting is still pending; the
  :class:`~alloygbm.MultiLabelGBMRanker` wrapper trains K independent
  per-label rankers.

What's new in 0.7.2
-------------------

Documentation, supply-chain, and repo-hygiene release.  No user-facing
Python API surface changes.

**Documentation:**

- Multiple docs still claimed warm-start was rejected, SHAP required
  ``leaf_model="constant"``, interaction constraints did not exist, or
  rankers were single-label only after v0.7.1 actually shipped those
  features.  README, ``docs/user/*.md``, the Sphinx mirror under
  ``docs/site/source/*.rst``, ``docs/roadmap/current.md``,
  ``CLAUDE.md``, ``AGENTS.md``, and ``benchmarks/README.md`` are now
  consistent with the v0.7.1 surface that actually shipped.
- ``docs/reference/release_checklist.md`` is now a top-to-bottom
  operating manual covering version bumps, doc updates, verification,
  tag/publish, and post-release bookkeeping.
- ``docs/site/source/api.rst`` now auto-documents
  :class:`~alloygbm.MultiLabelGBMRanker` (was missing in v0.7.1).
- New ``examples/`` directory with 8 self-contained end-to-end scripts.

**Repo hygiene & supply chain:**

- CI now runs the full pytest suite (455 tests) on every PR.  v0.7.1
  built the wheel and ran a handful of smoke snippets but never
  invoked ``pytest bindings/python/tests/`` — the Python test suite
  was not enforced on merge.
- ``Cargo.lock`` is tracked.
- ``maturin`` pinned in ``publish.yml`` to the same SemVer range
  declared in ``pyproject.toml``.
- ``cargo-audit`` + ``cargo-deny`` run weekly and on every PR that
  touches Cargo manifests, configured via the new ``deny.toml``.
- Coverage reporting via ``cargo-llvm-cov`` + ``pytest-cov`` →
  Codecov.
- ``publish = false`` on every workspace crate.
- New ``CONTRIBUTING.md``, ``SECURITY.md``, GitHub issue / PR /
  CODEOWNERS / Dependabot configs, ``.editorconfig``,
  ``requirements-dev.txt``, README badges.

**Limitations documented for the next release:**

- SHAP path-walker still compares against bin-index thresholds (carried
  over from v0.7.1).
- MorphBoost warm-start does not restore the EMA snapshot (carried
  over from v0.7.1).
- ``MultiLabelGBMRanker`` trains K independent per-label rankers;
  joint shared-tree multi-label boosting (carried over from v0.7.1).
- **NEW**: SHAP additivity check has a 1e-5 absolute tolerance that
  f32 round-off can exceed across larger evaluation samples; loosening
  to ``atol + rtol * |predict(x)|`` is queued.
- **NEW**: ``pyo3 = 0.23.5`` has RUSTSEC-2025-0020; not exploitable in
  AlloyGBM's code path.  Upgrading to ``pyo3 0.24+`` requires migrating
  the bindings to the ``Bound<>``-first API.

What's new in 0.7.1
-------------------

**SHAP for piecewise-linear leaves:**

- ``shap_values()`` now accepts ``leaf_model="linear"`` artifacts and
  returns an interventional decomposition: the path-based TreeSHAP /
  brute-force machinery attributes each leaf's "constant part"
  (``intercept + Σ wⱼ·μⱼ_global``) while per-leaf row deviations
  ``wⱼ · (xⱼ − μⱼ_global)`` are credited directly to each regressor.
  Global feature means are persisted in a new ``FeatureBaseline``
  artifact section so SHAP is self-contained at explain time.

**Per-round training diagnostics:**

- Every estimator exposes ``diagnostics_per_round_`` — a list of dicts
  containing ``gradient_l2_norm``, ``gradient_variance``,
  ``hessian_l2_norm``, sampling counts, and (when factor neutralization
  is active) ``neutralization_effectiveness`` ``= 1 − ‖projₘ‖ / ‖origₘ‖``.

**Neutralized warm-start:**

- ``init_model`` / ``warm_start=True`` with ``neutralization=*`` is
  supported across ``pre_target``, ``per_round_gradient``, and
  ``split_penalty`` provided the caller supplies the same
  ``factor_exposures`` matrix used for the initial fit. Mode,
  ``factor_neutralization_lambda``, and (for ``split_penalty``)
  ``factor_penalty`` must match; mismatches raise a clear "does not
  match" error.

**Interaction constraints:**

- LightGBM-compatible ``interaction_constraints=[[…]]`` on every
  estimator. Each group is a set of feature indices; any root-to-leaf
  path is restricted to splits on features from a single still-active
  group. Up to 64 groups per fit; enforced through both the level-wise
  and leaf-wise tree builders.

**Multi-label ranking:**

- New :class:`~alloygbm.MultiLabelGBMRanker` exposes a unified
  multi-output ranking API. ``y`` is shaped ``(n_rows, n_labels)`` and
  ``predict`` returns the same shape. Trains one independent
  :class:`~alloygbm.GBMRanker` per label sharing ``group`` /
  ``factor_exposures`` / kwargs, supports per-label
  ``ranking_objective`` lists, and slices ``eval_set`` y-columns per
  label so early stopping and custom eval metrics work end-to-end.

**Limitations documented for the next release:**

- SHAP path-walker still compares feature values against bin-index
  thresholds; strict additivity is relaxed for PL-leaf artifacts.
  Tightening this is queued for v0.7.2.
- MorphBoost warm-start does not restore the EMA snapshot from the
  artifact, so resumed training starts EMA cold.
- ``MultiLabelGBMRanker`` trains K independent per-label rankers.
  Joint shared-tree multi-label boosting is queued for v0.7.2.

What's new in 0.7.0
-------------------

**Factor-neutral boosting:**

- New ``neutralization`` parameter on :class:`~alloygbm.GBMRegressor`,
  :class:`~alloygbm.GBMClassifier`, and :class:`~alloygbm.GBMRanker`, with
  row-aligned fit-time ``factor_exposures``.
- ``neutralization="per_round_gradient"`` projects each boosting round's
  objective gradients away from user-supplied factors. Multiclass
  classification projects each class-gradient column independently.
- ``neutralization="pre_target"`` residualizes the target once before training
  for built-in squared-error regression. Classification, ranking, custom
  objectives, and validation sets are rejected for this mode in 0.7.0.
- ``neutralization="split_penalty"`` also subtracts a factor-load penalty from
  split gain via ``factor_penalty``. It supports constant leaves, composes with
  ``leaf_solver="dro"`` and ``training_mode="morph"``, and rejects
  ``leaf_model="linear"`` in 0.7.0.
- Neutralized ``warm_start`` and ``init_model`` continuation are rejected in
  0.7.0 — this restriction was lifted in v0.7.1 with the same-exposures
  contract documented above.

**Benchmarks:**

- ``alloygbm_factor_neutral`` and ``alloygbm_factor_neutral_dro`` arms added to
  ``benchmarks/run_model_comparison.py``.
- Benchmark datasets without explicit factors synthesize ``factor_exposures``
  from the first ``min(5, n_features)`` feature columns. These arms are smoke
  and stability checks, not standalone quality claims, because the synthesized
  factors are also present as model features.

What's new in 0.6.0
-------------------

**DRO-style scalar leaves:**

- New opt-in ``leaf_solver="dro"`` parameter on
  :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`, and
  :class:`~alloygbm.GBMRanker`. The solver is a fast, closed-form robust Newton
  update over within-leaf gradient uncertainty.
- ``dro_radius`` controls the gradient-uncertainty penalty and
  ``dro_metric="wasserstein"`` names the Wasserstein-inspired robust
  counterpart. This is not a full optimizer over raw feature/target
  distributions.
- ``leaf_solver="dro"`` requires ``leaf_model="constant"`` and composes with
  ``training_mode="morph"``.
- Inference speed is unchanged because robust scalar leaf values are stored
  directly in the artifact.

What's new in 0.5.0
-------------------

**Piecewise-linear (PL) tree leaves:**

- New opt-in ``leaf_model="linear"`` parameter on
  :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`, and
  :class:`~alloygbm.GBMRanker`. Each leaf stores a small linear model
  ``f_s(x) = b_s + Σ α_j x_j`` (up to 8 regressors per leaf, inherited from
  the split path's feature indices; the cap is internal and not user-tunable
  in v0.5.0). Optimal weights are solved in closed form via the ridge
  regression ``α* = -(XᵀHX + λI)⁻¹ Xᵀg``, regularised by ``lambda_l2``.
- Default ``leaf_model="constant"`` preserves all prior behaviour exactly.
- New artifact section ``ModelSectionKind::LinearLeafCoefficients`` stores
  per-stump linear leaf data; backward-compatible with v0.4.0 artifacts.
- Native-bitset categorical splits (``max_cat_threshold > 0``) fall back to
  constant leaves at the categorical split node; descendant numeric leaves
  use linear leaves normally.
- Multi-class softmax fits each per-class tree sequence with linear leaves
  independently.
- ``leaf_model="linear"`` composes with ``training_mode="morph"``.
- SHAP (``shap_values``, ``feature_importances``) currently raises an error
  for ``leaf_model="linear"`` artifacts; use ``leaf_model="constant"`` if you
  need SHAP.

**Performance:**

- ~10× faster convergence on linearly-structured datasets (fewer rounds to
  reach the same RMSE).
- +3.5% RMSE on California Housing and +1.75pp accuracy on Breast Cancer vs
  constant leaves.
- 2–8× per-round training overhead from the closed-form Cholesky solve.
  Recommended ``lambda_l2 >= 0.01`` for weight stability.

**Benchmarks:**

- ``alloygbm_linear`` and ``alloygbm_morph_linear`` arms added to
  ``benchmarks/run_model_comparison.py`` for all four task types.
- New ``benchmarks/pl_trees_benchmark.py`` script with convergence-curve and
  λ-sweep analysis.
- Benchmark report committed to ``docs/benchmarks/pl_trees_v1.md``.

What's new in 0.4.0
-------------------

**MorphBoost mode and SIMD acceleration:**

- New opt-in adaptive training mode via ``training_mode="morph"``,
  implementing the criterion from
  `Kriuk (2025) <https://arxiv.org/pdf/2511.13234>`_. Available on
  :class:`~alloygbm.GBMRegressor`, :class:`~alloygbm.GBMClassifier`, and
  :class:`~alloygbm.GBMRanker`. See :doc:`morphboost`.
- New per-iteration learning-rate schedule parameter ``lr_schedule``
  (``"constant"`` default, ``"warmup_cosine"`` available). Independent of
  ``training_mode`` — usable on its own.
- Schedule-aware auto early-stopping: when an LR schedule is active, the
  auto-tuned ``min_loss_improvement`` threshold is scaled by
  ``current_lr / max_lr``, and warmup-phase rounds are tolerated without
  termination.
- Backend SIMD acceleration via the ``wide`` crate (safe API; AVX2 / NEON
  intrinsics underneath, scalar fallback otherwise). Histogram bin-scan
  and EMA passes are now vectorized; histogram tile sizing is auto-tuned
  for high-feature workloads.
- New benchmark harnesses: ``benchmarks/morph_report.py``,
  ``benchmarks/morph_ablation.py``, and an enhanced
  ``benchmarks/numerai_benchmark.py`` with MorphBoost arms and a startup
  build-freshness check.
- ``benchmarks/run_model_comparison.py`` registers two new arms by default
  per task type: ``alloygbm_morph`` and ``alloygbm_morph_cosine``. New
  ``--models`` flag filters which arms run.

What's new in 0.3.2
--------------------

``0.3.2`` fixes silent zero-tree training in ``GBMRanker``, corrects signature
introspection, and adds a real-data ranking benchmark:

**GBMRanker training fixes:**

- The auto training policy's density-based ``min_split_gain`` and
  ``min_loss_improvement`` floors are no longer applied to ranking objectives.
  Ranking gradients are an order of magnitude smaller than
  regression/classification gradients; on datasets where
  ``row_count * feature_count >= 65 536`` these floors were causing training to
  exit after round 1 with zero trees committed.
- The main training loop's unconditional ``loss_improvement < 0`` early-exit no
  longer fires for ranking objectives, where round-to-round loss oscillation is
  expected behaviour.
- ``inspect.signature(GBMRanker.__init__)`` now returns the full parameter set
  (``ranking_objective`` plus all ``GBMRegressor`` parameters). Previously only
  three parameters were visible, causing tools that build kwargs via signature
  introspection to silently train with ``n_estimators=6``.

**Diagnostics:**

- ``stop_reason_`` and ``rounds_completed_`` attributes are now set on all
  estimators after ``fit()`` to surface the engine's early-stop reason and
  actual committed round count.

**Benchmarks:**

- Added ``california_ranking``: California Housing reframed as learning-to-rank
  with geographic grid cells as queries and ``median_house_value`` bucketed into
  5 graded relevance levels (~44 queries × 468 docs = ~20 595 rows).

What was new in 0.3.1
----------------------

``0.3.1`` fixed multiclass prediction and expanded the benchmark suite:

- Fixed ``class_trees`` threshold conversion so multiclass models predict
  correctly with continuous float features
- Fixed multiclass benchmark argmax label mapping with ``model.classes_``
- Added ``wine_multiclass``, ``digits_multiclass``, ``adult_income``,
  ``abalone_regression`` benchmark scenarios
- Activated ``synthetic_multiclass`` and ``synthetic_categorical`` scenarios
- Rewrote ``benchmarks/README.md``

What was new in 0.3.0
----------------------

``0.3.0`` adds native categorical splits, multi-class classification, and
custom objective/metric support:

**Native categorical splits:**

- Fisher-sort categorical split-finding with O(K log K) optimal binary
  partitions and O(1) bitset prediction
- ``max_cat_threshold`` parameter controls the maximum category cardinality
  for native splits (default 0 = disabled, opt-in)
- Category-to-ID mappings preserved through pickle, save/load, and params
- Full support across ``GBMRegressor``, ``GBMClassifier``, and ``GBMRanker``

**Multi-class classification:**

- ``GBMClassifier`` auto-detects K > 2 classes and uses softmax
  (multinomial cross-entropy) objective with K trees per round
- ``predict_proba`` returns (n_samples, K) probability matrix

**Custom objectives and metrics:**

- ``objective=callable`` for user-defined gradient/hessian computation
- ``eval_metric=callable`` for custom evaluation metrics with early stopping
- ``higher_is_better`` protocol for metric direction

What was new in 0.2.0
---------------------

``0.2.0`` was a major capability expansion from the regression-only ``0.1.x``
series:

**New estimators:**

- ``GBMClassifier`` -- binary classification with log-loss objective,
  ``predict_proba``, sklearn ``ClassifierMixin``
- ``GBMRanker`` -- learning-to-rank with 5 objectives (RankNet, LambdaMART,
  XE-NDCG, QueryRMSE, YetiRank)

**Core improvements:**

- NaN / missing value support across training and prediction
- Sample weight support via ``fit(..., sample_weight=...)``
- Group ID support via ``fit(..., group=...)``
- Model persistence: pickle, ``save_model``/``load_model``, artifact export
- Feature name capture from pandas DataFrames and other named inputs
- sklearn compatibility (``BaseEstimator``, ``RegressorMixin``,
  ``ClassifierMixin``, ``get_params``, ``set_params``, ``score``)
- ``min_split_gain`` exposed as a user parameter

**Training enhancements:**

- Leaf-wise (best-first) tree growth via ``tree_growth="leaf"``
- Monotone constraints via ``monotone_constraints``
- Feature importance weighting via ``feature_weights``
- ``max_leaves`` parameter for leaf-budget-oriented training
- Warm-starting / incremental training via ``warm_start=True``
- Up to 65,535 bins per feature (adaptive u8/u16 storage)
- Multiple categorical column support via ``categorical_feature_indices``
- Histogram buffer reuse to reduce allocation pressure
- Objective-aware training metric tracking (RMSE, log-loss, accuracy, NDCG)

**Explanations:**

- TreeSHAP (polynomial-time exact Shapley values, replaces the 25-feature
  brute-force method)
- SHAP limit raised from 20 to 25 features (for legacy brute-force path),
  then replaced entirely by TreeSHAP

**Metrics:**

- ``accuracy`` -- classification accuracy
- ``log_loss`` -- binary cross-entropy
- ``ndcg`` -- normalized discounted cumulative gain (with optional k)

**Benchmarks:**

- Classification scenarios: ``breast_cancer``, ``synthetic_classification``
- Ranking scenario: ``synthetic_ranking``
- Task-type-aware benchmark runner with per-type metrics and rendering

Validated release surface
-------------------------

For ``0.7.1``, the intended release surface is:

- macOS ``arm64`` wheel
- Linux ``x86_64`` manylinux wheel
- source distribution

Deferred targets
----------------

These are intentionally deferred:

- Windows wheels
- macOS Intel wheels

Release checklist summary
-------------------------

Before a public release:

- confirm package metadata and version
- confirm user docs are up to date
- confirm CI is green
- confirm the built wheel installs in a fresh environment
- confirm the publish workflow smoke-tests its wheel artifacts before upload
- confirm benchmark messaging stays narrow and defensible
