Release and platform policy
===========================

AlloyGBM ``0.12.7`` release notes and platform policy.

What's new in 0.12.7
--------------------

**Feature and compatibility release on top of v0.12.6.** Closes limitation
#6 from ``docs/limitations.md``: Quantile regression now fully composes
with DART boosting, MorphBoost training, and piecewise-linear
(``leaf_model="linear"``) leaves.

- **Quantile objective compatibility extended.** ``GBMRegressor(objective="quantile")``
  now successfully composes with:
  
  - **DART boosting** (``boosting_mode="dart"``): leaf refinement operates
    correctly on dropped-out residuals.
  - **MorphBoost** (``training_mode="morph"``): leaf refinement scales
    intercept updates by MorphBoost per-round shrinkage and depth-based penalty.
  - **Piecewise-linear leaves** (``leaf_model="linear"``): leaf refinement
    calculates residual targets by subtracting the linear portion of predictions
    from training values, and only refines the flat leaf intercept (avoiding
    double-scaling of build-time solved linear slopes).

- **Linear leaves + quantile numeric test.** Added a new test function
  ``test_quantile_linear_leaves_numeric`` verifying that linear-leaf quantile
  regression fits linear relationships significantly better than standard
  constant-leaf models.

- **Fixed double-scaling blocker** on linear leaf weights during quantile
  leaf refinement. Solved linear weights already carry the appropriate
  learning rate scale from build time and are now left untouched during
  intercept refinement.

- **Fixed shrinkage sign-flip bug.** Clamped ``iter_shrinkage`` to ``[0.0, 1.0]``
  (using ``.max(0.0)``) in ``trainer/mod.rs`` to match the authoritative joint
  trainer formula and prevent negative scaling factors under large morph rates.

No artifact format change. Model artifacts written by v0.12.6 load and
predict identically under v0.12.7.

What's new in 0.12.6
--------------------

**Feature release on top of v0.12.5.** Closes limitation #3 from
``docs/limitations.md``: SHAP values and interaction values are now
supported on multiclass classifiers and multi-output (joint) rankers in
addition to single-output regressors.

- ``GBMClassifier.shap_values(X)`` and
  ``GBMClassifier.shap_interaction_values(X)`` return a list of ``K``
  arrays — one per class logit. Additivity per class:
  ``Σⱼ values[k][i][j] + expected_values[k] ≈ raw_logit_k(rows[i])``.
- ``MultiLabelGBMRanker.shap_values(X)`` and
  ``MultiLabelGBMRanker.shap_interaction_values(X)`` return a list of
  ``n_labels`` arrays — one per output. Joint mode
  (``multi_label_mode="joint"``) routes through new per-output Rust
  entry points with full binning-context support; independent mode
  fans out to per-label ``GBMRanker.shap_values``.
- ``global_importance_from_artifact_bytes`` now averages over outputs
  (divides by ``n_models``) so importance magnitudes remain comparable
  across single-output and multi-output models.
- The Rust crate gained four new public entry points:
  ``explain_rows_from_artifact_bytes_per_output``,
  ``explain_rows_from_artifact_bytes_with_binning_per_output``,
  ``explain_interactions_from_artifact_bytes_per_output``, and
  ``explain_interactions_from_artifact_bytes_with_binning_per_output``.
  The original single-output entry points keep their existing signature
  and now error on K>1 artifacts directing callers to the
  ``_per_output`` variants.

**Internal refactors.** ``load_artifact_context`` decomposed into
``unroll_multiclass``, ``parse_joint_baselines``, and
``unroll_multi_output`` helpers (orchestrator stays ~45 lines).
``bindings/python/src/predict.rs`` split into ``predict.rs`` (predictor
entry points) and ``shap_bridge.rs`` (all 16 SHAP PyO3 wrappers — 8
single-output + 8 ``_multi``). Continues the v0.12.2 / v0.12.3
decomposition pattern.

**No artifact format change.** Model artifacts written by v0.12.5 load
and predict identically under v0.12.6.

What's new in 0.12.5
--------------------

**Small feature release on top of v0.12.4.** Closes the
``leaf_model="linear"`` exception on SHAP interaction values that was
carved out when interactions originally shipped in v0.11.0.

- ``GBMRegressor.shap_interaction_values(X)`` now accepts artifacts
  trained with ``leaf_model="linear"``. The row-dependent linear
  deviation ``w_j · (x_j − μ_j)`` is credited to the diagonal of the
  interaction matrix (the regressor feature's main effect): standard
  TreeSHAP interactions run on the constant part of each leaf
  (``intercept + Σⱼ wⱼ·μⱼ``), then the per-row deviations are folded
  onto ``Φ[j][j]`` via the same helper that backs PL-leaf
  ``shap_values``. Full additivity (``Σᵢⱼ Φᵢⱼ + E = ŷ``) and row-marginal
  (``Σⱼ Φᵢⱼ = φᵢ``) hold by construction; the matrix is symmetric and
  ``expected_value`` is unchanged.
- Pragmatic caveat: this attribution does not split linear-deviation
  credit across path-feature × regressor-feature off-diagonals; a
  faithful PL-leaf interaction decomposition remains an open extension.
- Internal refactor: ``explain_interactions_from_model`` moved from
  ``crates/shap/src/lib.rs`` to ``crates/shap/src/tree_shap.rs`` next to
  its peer ``explain_rows_tree_shap``. Continues the v0.12.2 SHAP-crate
  decomposition pattern; no behavioral change.

No artifact format change. Model artifacts written by v0.12.4 load and
predict identically under v0.12.5. **644 pytest** (v0.12.4 baseline 643
plus the renamed-and-extended linear-leaf interactions test and the new
LinearRank × linear-leaves coverage) and **447 cargo** (v0.12.4 baseline
445 plus two new ``shap_interactions_linear_leaves_*_satisfies_additivity``
tests).

What's new in 0.12.4
--------------------

**Bugfix release on top of v0.12.3.** Two post-merge review findings
(issues #48, #49) from the v0.12.2 / v0.12.3 refactor PRs:

- ``GBMRegressor.__module__`` now reports its public ``alloygbm.regressor``
  shim path instead of the private ``alloygbm._regressor._core``
  implementation module. ``repr`` and newly-created pickle payloads no
  longer leak the internal package layout; old v0.12.3 pickles continue
  to load.
- The joint trainer's module-level documentation in
  ``crates/engine/src/joint/mod.rs`` is refreshed to reflect the v0.10.x
  feature parity (DART, GOSS, MorphBoost, DRO, factor neutralization,
  warm-start, leaf-wise growth, native categorical splits, interaction
  constraints) that had landed since the original v0.10.0 minimal scope.

No user-facing API changes, no behavioral changes, no new features.
Model artifacts written by v0.12.3 load and predict identically under
v0.12.4. **643 pytest** (the v0.12.3 baseline of 641 plus the two new
regression tests for the module-identity fix) and **445 cargo** tests
pass.

What's new in 0.12.3
--------------------

**Phases 6–8 of the structural refactor — completing the program.** No
user-facing API changes, no behavioral changes, no new features. The
6,619-line ``bindings/python/src/lib.rs`` (the PyO3 bridge) was decomposed
into nine focused submodules plus a slim ``lib.rs`` and an extracted
``tests/`` submodule; the 4,909-line ``bindings/python/alloygbm/regressor.py``
(the ``GBMRegressor`` estimator) was decomposed into a ``_regressor/`` mixin
package (``_base`` plus four mixins and a ``_core`` shell), with
``regressor.py`` reduced to a back-compat shim.

- **No new objectives, parameters, training modes, or estimator API.**
- **No artifact format changes.** Model artifacts written by v0.12.2 load
  and predict identically under v0.12.3.
- ``from alloygbm.regressor import GBMRegressor`` and the ``alloygbm.regressor``
  module name are unchanged; ``GBMClassifier`` / ``GBMRanker`` subclass
  ``GBMRegressor`` transparently.
- Closes the file-decomposition program (issue #44). The 445 cargo + 641
  pytest tests held at every refactor commit.

What's new in 0.12.2
--------------------

**Phase 4 + Phase 5 of the structural refactor.** No user-facing API
changes, no behavioral changes, no new features. The 3,925-line
``crates/shap/src/lib.rs`` was decomposed into eight focused
single-responsibility modules; the 5,088-line
``crates/engine/src/joint.rs`` was promoted to a ``crates/engine/src/joint/``
subdir with five sibling modules.

- **No new objectives, parameters, training modes, or estimator API.**
- **No artifact format changes.** Model artifacts written by v0.12.1 load
  and predict identically under v0.12.2; v0.12.2 produces byte-identical
  artifacts to v0.12.1 from the same training data.
- **No public Rust API changes.** Every ``pub`` symbol that resolved at
  ``alloygbm_shap::*`` or ``alloygbm_engine::joint::*`` in v0.12.1 still
  resolves at the same path in v0.12.2 via the ``pub use`` re-exports
  in the SHAP crate's ``lib.rs`` and in ``joint/mod.rs``.
- **Verified at every commit.** All 445 cargo workspace tests and all
  641 pytest tests pass unchanged on every one of the 15 refactor commits
  (9 for the SHAP crate, 6 for the engine joint trainer). Function bodies
  were moved byte-identically; visibility promotions on private items
  were limited to the minimum required for sibling-module access
  (private ``fn`` to ``pub(super)`` or ``pub(crate)``, never past
  ``pub(crate)``).

After this release, the remaining queued refactor work is the PyO3
binding (Phase 6), the Python regressor (Phase 7), and a cross-cutting
verification + ``CLAUDE.md`` refresh (Phase 8) — see tracking issue #44.
Each ships as its own patch release.

What's new in 0.12.1
--------------------

**Phase 2 + Phase 3 of the structural refactor.** No user-facing API
changes, no behavioral changes, no new features. The 4,822-line
``crates/core/src/lib.rs`` was decomposed into thirteen focused
single-responsibility modules; the 3,987-line
``crates/backend_cpu/src/lib.rs`` was decomposed into five sibling modules.

- **No new objectives, parameters, training modes, or estimator API.**
- **No artifact format changes.** Model artifacts written by v0.12.0 load
  and predict identically under v0.12.1; v0.12.1 produces byte-identical
  artifacts to v0.12.0 from the same training data.
- **No public Rust API changes.** Every ``pub`` symbol that resolved at
  ``alloygbm_core::*`` or ``alloygbm_backend_cpu::*`` in v0.12.0 still
  resolves at the same path in v0.12.1 via the ``pub use`` re-exports
  in each crate's ``lib.rs``.
- **Verified at every commit.** All 445 cargo workspace tests and all
  641 pytest tests pass unchanged on every one of the 18 refactor commits
  (13 for the core crate, 5 for backend_cpu). Function bodies were moved
  byte-identically; visibility promotions on private items in backend_cpu
  were limited to the minimum required for sibling-module access
  (private ``fn`` to ``pub(crate) fn``, never past ``pub(crate)``).

After this release, the remaining queued refactor work is the SHAP crate
(Phase 4), the engine joint trainer (Phase 5), the PyO3 binding
(Phase 6), the Python regressor (Phase 7), and a cross-cutting
verification + ``CLAUDE.md`` refresh (Phase 8) — see tracking issue #44.
Each ships as its own patch release.

What's new in 0.12.0
--------------------

**Engine crate refactor.** No user-facing API changes, no behavioral changes,
no new features. The 15,189-line ``crates/engine/src/lib.rs`` monolith was
decomposed into 24 focused single-responsibility modules across a new
``crates/engine/src/`` layout and a new ``crates/engine/src/trainer/``
submodule directory. The remaining ``lib.rs`` is 101 lines of module
declarations and ``pub use`` re-exports.

- **No new objectives, parameters, training modes, or estimator API.**
- **No artifact format changes.** Model artifacts written by v0.11.1 load and
  predict identically under v0.12.0; v0.12.0 produces byte-identical artifacts
  to v0.11.1 from the same training data.
- **No public Rust API changes.** Every ``pub`` symbol that resolved at
  ``alloygbm_engine::*`` in v0.11.1 still resolves at the same path in
  v0.12.0 via the ``pub use`` re-exports in ``lib.rs``.
- **Verified at every commit.** All 207 engine unit tests, all 445 workspace
  Rust tests, and all 641 pytest tests pass unchanged on every one of the 24
  refactor commits. Function bodies were moved byte-identically; visibility
  promotions were limited to the minimum required by the new module boundary
  (private ``fn`` to ``pub(crate) fn``, never past ``pub(crate)``).

Scope: only ``crates/engine/src/lib.rs``. The other large files
(``bindings/python/src/lib.rs``, ``crates/engine/src/joint.rs``,
``bindings/python/alloygbm/regressor.py``, ``crates/core/src/lib.rs``,
``crates/backend_cpu/src/lib.rs``, ``crates/shap/src/lib.rs``) are untouched
and queued for future releases.

What's new in 0.11.1
--------------------

**Quantile regression.** ``GBMRegressor`` accepts a new quantile regression
objective (``objective="quantile"``) with pinball loss semantics and parameter
``quantile_alpha`` (default ``0.5``, strictly in ``(0.0, 1.0)``).

- **Empirical Quantile Leaf Refinement**: At the end of each round, a custom
  post-growth leaf refinement step (``refine_quantile_leaf_values``) is run to
  replace Newton-Raphson leaf predictions with the actual empirical quantiles
  of residuals for all rows in each leaf.
- **Full-dataset refinement**: Under ``row_subsample < 1.0``, split-finding runs
  on the subsampled subset, but leaf refinement uses the entire training set to
  minimize the estimation variance of the empirical quantile.
- **Proxy Hessian**: Since the pinball loss has a zero second derivative everywhere,
  a proxy Hessian ``h_i = w_i`` (sample weight) is used during split-finding.
- **Quickselect optimization**: The unweighted refinement path uses a fast
  ``O(N)`` quickselect algorithm (``select_nth_unstable_by``) instead of sorting
  ``O(N log N)``, avoiding performance degradation.
- **Validation**: Gated validation ensures that invalid ``quantile_alpha`` settings
  are only rejected when ``objective="quantile"`` is active, leaving non-quantile
  models unaffected.

Scope limit: Single-output ``GBMRegressor`` only. Rejects combinations with DART
boosting, MorphBoost, linear leaves (``leaf_model="linear"``), classification,
ranking, and joint multi-output training.

What's new in 0.11.0
--------------------

Two small, independent wins in one release.

**SHAP interaction values.** ``GBMRegressor.shap_interaction_values(X)``
returns the ``(n_rows, n_features, n_features)`` pairwise SHAP-interaction
tensor in ``O(T · L · D² · M)`` time. Implements Lundberg et al. (2020)
Algorithm 2, ported verbatim from the canonical ``slundberg/shap`` C++
reference. Three invariants are pinned by tests: symmetric
(``Φ_ij == Φ_ji``), row-marginal recovers per-feature SHAP
(``Σ_j Φ_ij == φ_i``), and full additivity reconstructs the prediction
(``Σ_i Σ_j Φ_ij + expected_value == predict(x)`` within
``atol = 1e-5 + rtol = 1e-4 · |predict(x)|``). Constant-leaf artifacts only;
``leaf_model="linear"`` is rejected.

**Poisson / Gamma / Tweedie GLM objectives.** ``GBMRegressor`` accepts
three new log-link GLM objectives. All three use weighted-mean-in-log-space
initial predictions, Newton-Raphson leaves, and the standard ``ObjectiveOps``
machinery. ``predict()`` returns ``exp(raw)``. Tweedie supports
``1 < variance_power < 2`` (compound Poisson-gamma) via the new
``tweedie_variance_power: float = 1.5`` constructor kwarg. New deviance
metrics in ``alloygbm.evaluation``: ``poisson_deviance``, ``gamma_deviance``,
``tweedie_deviance(y_true, y_pred, variance_power=p)``. Target-domain
validation raises ``ValueError`` before training starts when targets violate
the domain (negative y for Poisson/Tweedie, non-positive y for Gamma).

Single-output ``GBMRegressor`` only; not on Ranker, Classifier,
multiclass, or the joint multi-output ranker.

What's new in 0.10.6
--------------------

Closes the last v0.10.4-deferred joint-path follow-up: all three factor
neutralization modes now work on the joint multi-output trainer.
``MultiLabelGBMRanker(multi_label_mode="joint", neutralization=…,
factor_exposures=…)`` supports ``"pre_target"``,
``"per_round_gradient"``, and ``"split_penalty"`` with the same surface
as the single-output ``GBMRegressor`` / ``GBMRanker``. The joint trainer
reaches full feature parity with the single-output path. Default
behaviour for every existing user-facing API remains byte-identical to
v0.10.5 when neutralization is not opted into.

**Three new modes**, all activated via the ``neutralization`` kwarg:

- ``pre_target`` — residualize each per-output target through the factor
  exposures once before training. Requires every per-output objective to
  be ``squared_error`` (the only objective where residualize-target
  equals residualize-gradient).
- ``per_round_gradient`` — project each of the K gradient buffers in
  place every round after computing them. Mirrors the single-output
  multiclass per-class projection pattern.
- ``split_penalty`` — subtract a K-output factor-load penalty from each
  candidate split's gain. Applies under both ``tree_growth="level"`` and
  ``tree_growth="leaf"``.

**Three new kwargs** admitted by ``_JOINT_SUPPORTED_KWARGS``:

- ``neutralization`` — ``"none"`` (default), ``"pre_target"``,
  ``"per_round_gradient"``, or ``"split_penalty"``
- ``factor_neutralization_lambda`` — ridge regularization on the projector
  Gram matrix (default ``1e-6``)
- ``factor_penalty`` — ``split_penalty`` mode's penalty multiplier
  (default ``0.0`` — ``0`` collapses to standard byte-for-byte)

Plus the ``factor_exposures=`` kwarg on ``fit()`` (already existed for the
independent-mode fallback; now honored on joint too). The PyO3 bridge
cross-validates the exposures-vs-config invariant: active config requires
exposures, exposures require an active config.

**Artifact:** new ``ModelSectionKind::NeutralizationMetadata`` (kind=14)
records the active config in the artifact so joint models are
self-describing. Metadata only; prediction never reads it (neutralization
is a training-time transformation; the trained leaf values already bake
in the projection).

**Byte-equivalence:** a fit with ``neutralization='none'`` (or
``kind=None``, or ``split_penalty=0``) produces byte-identical artifact
bytes to a pre-v0.10.6 fit. Pinned by
``joint_neutralization_inert_configs_match_v0_10_5_byte_for_byte``.
Composes with MorphBoost (``training_mode="morph"``), DRO leaves
(``leaf_solver="dro"``), DART boosting, and warm-start.

What's new in 0.10.5
--------------------

Closes the joint DRO leaves follow-up from v0.10.4.
``MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro",
dro_radius=…, dro_metric="wasserstein")`` now applies
Wasserstein-distributionally-robust leaf values on the joint
multi-output trainer, mirroring ``GBMRegressor`` / ``GBMRanker``'s
single-output leaf solver. Default behaviour for every existing
user-facing API remains byte-identical to v0.10.4 when DRO is not
opted into.

**Joint DRO leaves:**
routes the K-output Newton-Raphson leaf step through
``alloygbm_core::leaf_effective_gradient`` (the same helper used by
single-output ``GBMRegressor`` / ``GBMRanker`` since v0.6.x). Applied
in-build inside ``build_joint_round_inner``'s ``leaf_values`` closure
and ``build_joint_round_leafwise``'s per-output leaf computation — row
indices are already in scope at leaf-computation time. DRO is leaf-only:
split-gain dispatch still uses the standard K-output sum-of-XGBoost-gains
(multi-output histogram doesn't carry per-bin ``grad_sq``; adding it would
cost ~1.5× joint-round memory — split-time DRO is deferred pending
benchmark evidence).

Three new kwargs in ``_JOINT_SUPPORTED_KWARGS``:

- ``leaf_solver`` — ``"standard"`` (default) or ``"dro"``
- ``dro_radius`` — float ≥ 0; ``0.0`` collapses to standard byte-for-byte
- ``dro_metric`` — ``"wasserstein"`` (only supported value in v0.10.5)

Works under both ``tree_growth="level"`` and ``tree_growth="leaf"``, and
composes with MorphBoost (``training_mode="morph"``) and DART/GOSS
boosting modes. Byte-equivalent to v0.10.4 when ``lambda_l1 == 0`` AND
(``dro_config.is_none()`` OR ``dro_config.radius == 0.0``); pinned by
``joint_dro_radius_zero_matches_standard_byte_for_byte`` (cargo) and
``test_joint_dro_radius_zero_byte_equivalent_to_standard`` (pytest).

**Deferred to v0.10.6:**
joint factor neutralization (``neutralization`` + ``factor_exposures``).
Remains in ``docs/limitations.md`` Limitation 2 with explicit version
marker.

What's new in 0.10.4
--------------------

Adds MorphBoost (Kriuk 2025, arXiv:2511.13234) to the joint multi-output
trainer used by ``MultiLabelGBMRanker(multi_label_mode="joint")``. This
is the first of three deferred items from ``docs/limitations.md``
Limitation 2 to ship; DRO leaves landed in v0.10.5 and factor
neutralization on the joint trainer is tracked for v0.10.6. Default
behaviour for
every existing user-facing API remains byte-identical to v0.10.3 when
MorphBoost is not opted into.

**Joint MorphBoost surface:**
``MultiLabelGBMRanker(multi_label_mode="joint", training_mode="morph",
…)`` now activates MorphBoost on the shared-tree multi-output trainer.
Honors the full single-output MorphBoost surface — ``morph_rate``,
``evolution_pressure``, ``morph_warmup_iters``, ``info_score_weight``,
``depth_penalty_base``, ``balance_penalty``, ``lr_schedule``,
``lr_warmup_frac``. Per-iteration LR schedule (constant or
warmup-cosine), per-leaf depth penalty
(``depth_penalty_base ^ (depth/3)`` where
``depth = (local_node_id + 1).ilog2()``), and per-iteration leaf
shrinkage (``1 − morph_rate * round/total``) all apply uniformly across
the K-output leaf values.

**Multi-output morph gain:**
two new helpers in ``crates/engine/src/shared_histogram.rs`` —
``compute_multi_output_split_gain_morph`` and
``find_best_multi_output_categorical_split_morph`` — sum per-output
morph gain across the K outputs. Each output uses its own
``(grad_mean, grad_std)`` snapshot from ``MorphState::ema_stats[k]``.
Per-side row count for the info-gain term is approximated via
``hess.max(0.0) as u32`` (multi-output histogram doesn't carry exact
counts) — exact for objectives where hess ≡ 1 per row, monotone proxy
for ranking. Warmup byte-equivalence with the standard K-output gain
is guaranteed regardless.

**MorphBoost EMA warm-start (continuity, not byte-equivalence):**
``JointWarmStartState.initial_ema_stats: Option<Vec<GradientEmaStats>>``
re-seeds ``MorphState::ema_stats`` on warm-resume so the gradient-
statistics smoothing is continuous across the resume boundary — new
rounds see the same per-output ``(mean, std)`` they would have seen had
training never been interrupted. The PyO3 bridge auto-extracts the
snapshot from ``init_artifact_bytes`` via
``TrainedModel::from_artifact_bytes(…).morph_metadata``.

**MorphBoost warm-resume is intentionally NOT byte-equivalent to a fresh
longer fit.** Per-iteration leaf shrinkage and LR schedule are resolved
against the ``total_iterations`` horizon at training time; a prior fit
with ``n_estimators=6`` baked its first six trees against a 6-round
horizon and resuming with ``n_estimators=4`` cannot retroactively
re-scale them. The EMA continuity is the practical guarantee. This
mirrors the single-output MorphBoost warm-start behavior.

**Deferred to v0.10.5 / v0.10.6 (from v0.10.4):**
joint DRO leaves (``leaf_solver="dro"``) — shipped in v0.10.5 — and
joint factor neutralization (``neutralization`` + ``factor_exposures``)
— tracked for v0.10.6. See ``docs/limitations.md`` Limitation 2.

What's new in 0.10.3
--------------------

Closes the four "v0.10.3" follow-ups carved out of the v0.10.2
joint-trainer parity work: native-categorical Python wiring, joint
GOSS, joint DART, and joint warm-start. The
``MultiLabelGBMRanker(multi_label_mode="joint")`` wrapper now accepts
every kwarg the single-output trainer accepts (except MorphBoost / DRO
/ factor neutralization, which are tracked for v0.10.4). Default
behaviour for every existing user-facing API remains byte-identical to
v0.10.2 when the new knobs are not opted into.

**Joint native-categorical Python wiring:**
the Rust-level joint native-cat trainer
(``fit_joint_multi_output_with_categorical`` +
``find_best_multi_output_categorical_split``) was already in v0.10.2;
the PyO3 bridge ``train_joint_multi_label_ranker`` now re-bins
requested columns to ``bin_index == category_id`` before invoking the
trainer (mirrors the single-output
``apply_categorical_encoding_to_training_matrices_multi``). The
``_JOINT_SUPPORTED_KWARGS`` allow-list re-adds
``categorical_feature_indices`` and ``max_cat_threshold``.

**Joint GOSS:**
new ``select_joint_row_indices_for_round`` helper inside
``crates/engine/src/joint.rs`` mirrors
``select_row_indices_for_round_multiclass`` — per-row score is
:math:`s_i = \\sum_k |g_{i,k}|` across the K per-output gradient
buffers (LightGBM multiclass GOSS convention). A single row mask is
shared across all K buffers; the amplification factor mutates every
per-output gradient/hessian in lockstep so histograms remain unbiased.
``MultiLabelGBMRanker(multi_label_mode='joint', boosting_mode='goss',
goss_top_rate=..., goss_other_rate=...)``.

**Joint DART:**
dropout/normalize cycle added to ``fit_joint_inner``. One tree per
round on the joint trainer simplifies bookkeeping vs. multiclass DART:
``dart_state.tree_weights`` has length ``rounds_completed`` and
``dart_round_start_offsets[r]`` / ``dart_round_counts[r]`` collapse to
a flat per-round pair. Reuses ``engine::dart::{select_dropouts,
apply_normalization}`` unchanged. Per-stump ``tree_weight`` persists
via the existing ``DartTreeWeights`` artifact section (kind=11), and
``JointPredictor`` is extended with ``tree_weights: Vec<f32>`` so each
tree's leaf contribution is multiplied by ``tree_w`` at predict time.

**Joint warm-start:**
new ``JointWarmStartState { baselines, stumps,
initial_rounds_completed, initial_dart_tree_weights }`` + new
``fit_joint_multi_output_with_warm_start`` entry point.
``MultiLabelGBMRanker(multi_label_mode='joint', warm_start=True,
init_model=<prior_fit>)`` cracks open the prior fit's joint artifact,
replays prior stumps onto ``predictions`` via the shared
``walk_tree_into_predictions`` helper, re-encodes new-round
``node_id`` starting at ``initial_rounds_completed``, and (under DART)
reconstructs ``dart_state.tree_weights`` from per-stump
``tree_weight``. Per-round seeds mix
``global_round = round + initial_rounds`` so an N+M warm-resumed fit
produces identical RNG draws to a fresh N+M fit on rounds N..N+M.

**Deferred to later v0.10.x point releases:**

- v0.10.4: MorphBoost, DRO, and factor neutralization on the joint
  path.

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

**Deferred to later v0.10.x point releases (as documented in v0.10.2,
now closed):**

- v0.10.3 shipped: native-cat Python wiring, joint GOSS, joint DART,
  joint warm-start.
- v0.10.4: MorphBoost, DRO, and factor neutralization on the joint
  path.

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
