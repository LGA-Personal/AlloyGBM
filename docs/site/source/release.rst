Release and platform policy
===========================

AlloyGBM ``0.10.2`` release notes and platform policy.

What's new in 0.10.2
--------------------

Closes the leaf-wise multiclass DART limitation and the first slice of
joint-path feature parity (leaf-wise growth, native-categorical,
interaction constraints, row/col subsample, min_split_gain). The
remaining joint-path features land in v0.10.3 (GOSS, DART, warm-start
on joint) and v0.10.4 (MorphBoost, DRO, neutralization on joint).
Default behaviour for every existing user-facing API remains
byte-identical to v0.10.1 when the new features are not opted into.

**Joint trainer core feature parity:**
``engine::joint::fit_joint_multi_output`` now supports
``tree_growth="leaf"`` + ``max_leaves`` (via the new
``build_joint_round_leafwise`` priority-queue best-first growth),
``interaction_constraints`` (reusing the single-output
``InteractionConstraintIndex``), ``min_split_gain``, ``row_subsample``,
and ``col_subsample``. All five are exposed through
``MultiLabelGBMRanker(multi_label_mode="joint")`` Python surface;
``_JOINT_SUPPORTED_KWARGS`` grew to permit
``min_split_gain``, ``row_subsample``, ``col_subsample``,
``interaction_constraints``, ``tree_growth``, ``max_leaves``.

Native-categorical splits on the joint path are partially shipped:
the Rust-level
``find_best_multi_output_categorical_split`` Fisher-sort helper +
``fit_joint_multi_output_with_categorical`` entry point are in place
and sound when given bins where ``bin_index == category_id``. The
Python surface is intentionally *not* wired in v0.10.2 because the
current bridge bins all features with
``ContinuousBinningStrategy::Linear`` which doesn't preserve that
invariant for joint mode — ``categorical_feature_indices`` and
``max_cat_threshold`` are rejected in joint mode and tracked for
v0.10.3.

**Leaf-wise multiclass DART:**
``GBMClassifier(boosting_mode="dart")`` with K ≥ 3 classes now works
under ``tree_growth="leaf"`` + ``max_leaves``. The v0.10.1
``tree_growth='level'`` restriction in
``fit_multiclass_iterations_impl`` was lifted. Per-class
``dart_round_start_offsets[k]`` / ``dart_round_counts[k]`` bookkeeping
is growth-mode-agnostic because it snapshots ``class_stumps[k].len()``
around each ``build_tree_*`` call. Validation early-stopping DART
transition and DART warm-start tree-weight reconstruction work
without changes.

**Deferred to later v0.10.x point releases:**

- v0.10.3: GOSS, DART, and warm-start on the joint path.
- v0.10.4: MorphBoost, DRO, and neutralization on the joint path.

What's new in 0.10.1
--------------------

Closes the three v0.10.x-deferred limitations from v0.10.0:
``MultiLabelGBMRanker`` joint mode Python surface, multiclass softmax
+ GOSS, and multiclass softmax + DART (including warm-start). Default
behaviour for every existing user-facing API remains byte-identical
to v0.10.0 when the new features are not opted into.

**MultiLabelGBMRanker joint mode (Python surface):**

- ``MultiLabelGBMRanker(multi_label_mode="joint")`` now routes through
  a new PyO3 entry point (``train_joint_multi_label_ranker``) and
  ``JointPredictorHandle`` py-class to the v0.10.0 Rust joint trainer
  ``engine::joint::fit_joint_multi_output``. Default mode is still
  ``"independent"`` (the K-per-label ``GBMRanker`` fallback from
  v0.7.1) — joint is opt-in. Bundle format bumped to v2 with an
  explicit mode byte; v1 bundles still load as independent.

**Multiclass softmax + GOSS:**

- ``GBMClassifier(boosting_mode="goss")`` for K >= 3 classes. Per-row
  score :math:`s_i = \\sum_k |g_{i,k}|` (LightGBM convention) drives a
  shared sampling mask across all K class gradient buffers; the
  amplification factor is applied identically to every class's grad
  and hess. The multiclass round loop was refactored so the K gradient
  buffers are pre-computed before sampling.

**Multiclass softmax + DART (+ warm-start):**

- ``GBMClassifier(boosting_mode="dart")`` for K >= 3 classes. Per-class
  prediction vectors get per-round subtract/readd of dropped tree
  contributions scaled by ``dart_state.tree_weights``. Per-class
  ``dart_round_start_offsets`` / ``dart_round_counts`` arrays track the
  contiguous stump slice each (round, class) tree occupies in
  ``class_stumps[k]`` so dropout subtracts the WHOLE class tree, not
  just its root stump. After K new trees are built each round they are
  rescaled to ``new_w = 1/(n_dropped + 1)`` and the dropped trees are
  re-added at their rescaled weights. ``stump.tree_weight = new_w`` is
  stamped on every stump in the new round's per-class slice. Requires
  ``tree_growth="level"`` in v0.10.1.
- ``MultiClassWarmStartState.initial_dart_tree_weights`` carries the
  flat round-major × class-k per-tree weights from the prior fit, so
  continuation seeds ``dart_state.tree_weights`` correctly. The PyO3
  bridge reconstructs the per-tree weights by grouping
  ``class_stumps[k]`` by ``tree_id`` (decoded from
  ``node_id / TREE_NODE_STRIDE``) — taking the first stump's
  ``tree_weight`` per tree group, mirroring the predictor's
  ``apply_dart_tree_weights`` convention.

**Constraints:**

- Multiclass DART requires ``tree_growth="level"``; leaf-wise dropout
  indexing across K class trees is tracked as a follow-up.
- Joint mode supports level-wise growth, standard boosting, and the
  built-in ``squared_error`` / ``queryrmse`` / ``rank:pairwise`` /
  ``rank:ndcg`` / ``rank:xendcg`` objectives only. Joint-path feature
  parity (MorphBoost, neutralization, DRO, interaction constraints,
  leaf-wise, GOSS, DART, warm-start, ``row_subsample``,
  ``col_subsample``, ``min_split_gain``) is targeted for later v0.10.x
  releases — see ``docs/limitations.md``.

What's new in 0.10.0
--------------------

Infrastructure release: lays the Rust-level foundation for joint
multi-output learning and closes the v0.9.0 ``DART + warm_start``
follow-up. Default behaviour for every existing user-facing API
(``GBMRegressor``, ``GBMClassifier``, ``GBMRanker``,
``MultiLabelGBMRanker``) remains byte-identical to v0.9.0 — the new
``MultiOutputLeafValues`` artifact section is only emitted when the
(currently Rust-only) joint trainer produces a model.

**DART + warm_start continuation:**

- ``GBMRegressor``, ``GBMClassifier``, and ``GBMRanker`` now accept
  ``boosting_mode="dart"`` + ``warm_start=True`` (or
  ``fit(..., init_model=prior_model)``). The v0.9.0 rejection error
  is removed.
- ``WarmStartState`` gains an optional ``initial_dart_tree_weights``
  field that captures the per-stump ``tree_weight`` snapshot from the
  prior fit. The engine seeds ``dart_state.tree_weights`` from this
  snapshot and pre-populates the ``round_start_offsets`` /
  ``dart_round_counts`` arrays from the warm-start tree shapes.
- Historical RNG-driven ``dropped_per_round`` is intentionally not
  persisted; new rounds start fresh dropout bookkeeping going forward.

**Joint multi-output infrastructure (Rust):**

- ``MultiOutputHistogram`` (``crates/engine/src/shared_histogram.rs``)
  accumulates K (grad, hess) pairs per (feature, bin) in one sweep,
  with subtraction trick and multi-output split-gain helpers.
- ``MultiOutputLeafValues`` artifact section (kind index 13) stores
  per-stump K-output leaf values. ``TrainedStump`` gains optional
  ``multi_output_leaf_values: Option<(Vec<f32>, Vec<f32>)>``.
- Rust-level joint trainer (``crates/engine/src/joint.rs``):
  ``fit_joint_multi_output`` runs the full training loop with K
  per-output objectives (``squared_error``, ``queryrmse``,
  ``rank:pairwise``, ``rank:ndcg``, ``rank:xendcg``); ``JointPredictor``
  decodes the artifact and predicts K outputs per row.
- Scope intentionally minimal for v0.10.0: level-wise growth only,
  no MorphBoost / DRO / neutralization / leaf-wise / native-categorical
  / GOSS / DART / warm-start on the joint path.

**Deferred to v0.10.x:**

- Python ``MultiLabelGBMRanker(training_mode="joint")`` user-facing
  surface (Rust infrastructure complete; targeted for v0.10.1).
- Multiclass softmax + DART / GOSS (engine plumbing into the K-output
  histogram primitive is targeted for v0.10.1+).
- Leaf-wise / MorphBoost / DRO / neutralization on the joint path
  (feature parity with the single-output trainer is targeted for v0.10.x).

What's new in 0.9.0
-------------------

Minor feature release: closes the v0.8.0 DART placeholder
(Limitation 2) and resolves the linear-rank predict-path NaN routing
bug (Limitation 4).  Default behaviour is byte-identical to v0.8.0 on
every API surface — the new ``DartTreeWeights`` artifact section is
only emitted when at least one stump has a non-1.0 weight, which
never happens under ``boosting_mode="standard"`` (the default) or
``boosting_mode="goss"``.

**DART boosting mode (Dropouts meet MART):**

- New ``boosting_mode="dart"`` opt-in on ``GBMRegressor``, binary
  ``GBMClassifier``, and ``GBMRanker``, with four companion
  parameters: ``dart_drop_rate`` (default ``0.1``), ``dart_max_drop``
  (default ``50``), ``dart_normalize_type`` (``"tree"`` or
  ``"forest"``, default ``"tree"``), and ``dart_sample_type``
  (``"uniform"`` or ``"weighted"``, default ``"uniform"``).
- Per-round dropout + normalization cycle lives in a new module
  ``crates/engine/src/dart.rs``.  No new crate dependencies — uses
  the existing ``mixed_hash`` splitmix64 derivative so per-stump
  drop decisions are deterministic given ``seed`` + round index.
- Per-stump ``tree_weight: f32`` is plumbed through ``TrainedStump``
  and persisted via a new ``DartTreeWeights`` artifact section
  (``ModelSectionKind`` index 12).  Emitted only when at least one
  weight diverges from 1.0; pre-v0.9.0 artifacts continue to load
  with all weights defaulting to 1.0.
- The single-output training loop rejects ``boosting_mode="dart"``
  + ``warm_start`` with a clear error (tracked as a v0.10.x
  follow-up: would require persisting ``tree_weights`` and
  ``dropped_per_round`` in ``WarmStartState``).
- Multiclass softmax continues to reject ``boosting_mode != "standard"``
  with a clear error message; per-class gradient scoring during the
  dropout step is tracked as a v0.10.x follow-up.

**NaN routing on the linear-rank predict path (Limitation 4
resolved):**

- The predict-time quantize helpers in ``bindings/python/src/lib.rs``
  (``quantize_dense_values_linear_inplace_wide``,
  ``quantize_dense_values_linear_rank_inplace_wide``, and the inline
  loop in ``predict_dense_quantized_with_summary_bytes``) now preserve
  ``f32::NAN`` through the f32 cast instead of casting a finite bin
  index.  The predictor's existing
  ``feature_value.is_nan() -> default_left`` short-circuit at
  ``crates/predictor/src/lib.rs:148`` then fires automatically.
- ``LinearLeaf::eval`` (in ``alloygbm-core``) and
  ``LinearLeafCompact::eval`` (in ``alloygbm-predictor``) now skip
  NaN regressor features when accumulating the linear sum, so
  PL-leaf predictions don't NaN-poison on a ``w * NaN`` step.
- Pure-linear, pure-quantile, and rank-binning paths now share
  consistent NaN semantics: missing values always route through the
  learned ``default_left`` direction.

Known limitations carried forward to v0.10.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Multiclass softmax + DART is still rejected.
- DART + ``warm_start`` is rejected.
- Joint shared-tree multi-label ranking and the K-output
  shared-histogram engine primitive remain v0.10.0 targets.

What's new in 0.8.0
-------------------

Minor feature release: closes the mixed linear-rank SHAP carry-forward
from v0.7.4 (Limitation 4) and adds LightGBM-style GOSS sampling as a
new opt-in boosting mode.  Default behaviour is byte-identical to
v0.7.5 on every API surface.  The other two original v0.8.0 targets —
DART boosting mode and joint shared-tree multi-label ranking — were
scope-split out to v0.9.0 and v0.10.0 respectively so this release
could ship on a reviewable surface.  ``BoostingMode::Dart`` is reserved
in the API (Python ``boosting_mode="dart"`` raises
``NotImplementedError``; the Rust trainer rejects it with a clear error
message) so v0.9.0 can land DART training without further
``TrainParams`` churn.

**GOSS sampling (gradient-based one-side sampling):**

- New ``boosting_mode="goss"`` opt-in on ``GBMRegressor``,
  ``GBMClassifier`` (binary), and ``GBMRanker``, with companion
  ``goss_top_rate`` (default ``0.2``) and ``goss_other_rate``
  (default ``0.1``) parameters.  Default ``boosting_mode="standard"``
  is byte-identical to v0.7.5.
- Implements LightGBM's GOSS algorithm: at the start of each round
  rows are scored by ``|gradient|``, the top ``goss_top_rate``
  fraction is kept, ``goss_other_rate`` fraction is uniformly
  sampled from the rest, and the sampled-low rows' gradient +
  hessian are multiplied by ``(1 - goss_top_rate) / goss_other_rate``
  to preserve unbiased histogram statistics.
- Reorders the per-round training loop so gradient computation
  happens *before* row sampling — required because GOSS scores by
  gradient magnitude.  Standard and DART modes get the same
  pre-computed gradient buffer and fall back to uniform subsampling.
- Multiclass softmax explicitly rejects ``boosting_mode != "standard"``
  with a clear error message — per-class gradient scoring is tracked
  as a v0.8.1 follow-up.  DART is reserved for the next feature
  commit on ``v0.8.0-features`` and currently raises
  ``NotImplementedError`` in Python.

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
