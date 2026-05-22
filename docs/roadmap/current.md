# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is aimed at strong practical performance on structured tabular workloads, with particular strength on financial and time-aware problems.

The `0.10.5` release closes the joint DRO leaves follow-up from
v0.10.4: `MultiLabelGBMRanker(multi_label_mode="joint",
leaf_solver="dro")` now routes per-output leaf values through
`alloygbm_core::leaf_effective_gradient` (the same helper the
single-output trainers have used since v0.6.x). Factor
neutralization on the joint trainer remains tracked for v0.10.6.
Default behaviour for every existing user-facing API remains
byte-identical to v0.10.4 when DRO is not opted into.

The `0.10.4` release adds MorphBoost to the joint multi-output
trainer used by `MultiLabelGBMRanker(multi_label_mode="joint")` —
the first of three deferred joint-path follow-ups from
`docs/limitations.md` Limitation 2. DRO leaves on the joint
trainer shipped in v0.10.5; factor neutralization on the
joint trainer is tracked for v0.10.6. Default behaviour for every
existing user-facing API remains byte-identical to v0.10.3 when
MorphBoost is not opted into.

## What Shipped In v0.10.4

- **Joint MorphBoost (closes the v0.10.4 MorphBoost follow-up):**
  `MultiLabelGBMRanker(multi_label_mode="joint",
  training_mode="morph", …)` now activates MorphBoost on the
  shared-tree multi-output trainer. The full single-output morph
  surface is honored — `morph_rate`, `evolution_pressure`,
  `morph_warmup_iters`, `info_score_weight`, `depth_penalty_base`,
  `balance_penalty`, `lr_schedule`, `lr_warmup_frac`. Two new
  helpers in `crates/engine/src/shared_histogram.rs`
  (`compute_multi_output_split_gain_morph`,
  `find_best_multi_output_categorical_split_morph`) sum per-output
  morph gain across K outputs, each using its own EMA snapshot
  from `MorphState::ema_stats[k]`. `JointMorphContext` (private to
  `joint.rs`) carries the per-round morph snapshot through
  `build_joint_round*`. Per-iteration LR schedule, per-leaf depth
  penalty (`depth_penalty_base ^ (depth/3)`, depth derived from
  `local_node_id.ilog2()`), and per-iteration leaf shrinkage
  (`1 − morph_rate * round/total`) all apply uniformly across the
  K-output leaf values.
- **Joint MorphBoost warm-start (EMA continuity, not byte-equivalence):**
  `JointWarmStartState.initial_ema_stats: Option<Vec<GradientEmaStats>>`
  re-seeds `MorphState::ema_stats` on warm-resume so gradient-statistics
  smoothing is continuous across the resume boundary. **Not byte-
  equivalent to a fresh longer fit** (PR #37 review C3): per-iteration
  leaf shrinkage and LR schedule are resolved against the
  `total_iterations` horizon at training time, so a `6+4` warm-resume
  does not match a fresh `n_estimators=10` MorphBoost fit — the prior
  six trees keep their original 6-round shrinkage. Mirrors the single-
  output MorphBoost warm-start behavior. The PyO3 bridge extracts the
  EMA snapshot automatically from `init_artifact_bytes` via
  `TrainedModel::from_artifact_bytes(…).morph_metadata`.

### v0.10.x follow-ups (deferred)

- **v0.10.6**: factor neutralization on the joint trainer
  (`neutralization="pre_target" | "per_round_gradient" |
  "split_penalty"` + `factor_exposures=`). The PyO3 bridge
  currently rejects `factor_exposures` unconditionally under
  `multi_label_mode="joint"`.

## What Shipped In v0.10.5

Closes the joint DRO leaves follow-up carved out of v0.10.4.
Default behaviour for every existing user-facing API remains
byte-identical to v0.10.4 when `leaf_solver="dro"` is not opted into
(byte-equivalence is pinned at `lambda_l1 == 0` and
`dro_config.radius == 0.0`).

- **Joint DRO leaves (closes the v0.10.5 DRO follow-up):**
  `MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro",
  dro_radius=…, dro_metric="wasserstein")` routes the K-output
  Newton-Raphson leaf step through `alloygbm_core::leaf_effective_gradient`
  (the same helper used by single-output `GBMRegressor` / `GBMRanker`
  since v0.6.x). Applied in-build inside `build_joint_round_inner`'s
  `leaf_values` closure and `build_joint_round_leafwise`'s per-output
  leaf computation — row indices are already in scope at
  leaf-computation time; no separate post-build pass is required.
  Accumulates `grad_sq_sum` alongside `grad_sum` per output.
  Split-gain dispatch is unchanged (multi-output histogram doesn't
  carry per-bin `grad_sq`; inflating it costs ~1.5× joint-round memory
  which isn't justified before benchmark evidence). Composes with
  MorphBoost (`training_mode="morph"`), DART, and GOSS boosting modes,
  and works under both `tree_growth="level"` and `tree_growth="leaf"`.
  Byte-equivalent to v0.10.4 when `lambda_l1 == 0` AND
  (`dro_config.is_none()` OR `dro_config.radius == 0.0`); pinned by
  `joint_dro_radius_zero_matches_standard_byte_for_byte` (cargo) and
  `test_joint_dro_radius_zero_byte_equivalent_to_standard` (pytest).
  `_JOINT_SUPPORTED_KWARGS` adds `leaf_solver`, `dro_radius`,
  `dro_metric`.
- **Joint DRO warm-start:** `JointWarmStartState` requires no new
  fields — the DRO config flows through `TrainParams` which is already
  threaded into `fit_joint_inner`, so warm-resumed fits with
  `leaf_solver="dro"` automatically apply DRO leaf shrinkage to new
  rounds with no additional bridge plumbing.

### v0.10.x follow-ups (deferred)

- **v0.10.6**: factor neutralization on the joint trainer
  (`neutralization="pre_target" | "per_round_gradient" |
  "split_penalty"` + `factor_exposures=`). The PyO3 bridge
  currently rejects `factor_exposures` unconditionally under
  `multi_label_mode="joint"`.

## What Shipped In v0.10.3

- **Joint native-categorical Python wiring (closes the v0.10.3
  native-cat-wiring follow-up):** the Rust-level joint native-cat
  trainer was already in v0.10.2; the PyO3 bridge
  `train_joint_multi_label_ranker` now re-bins requested columns to
  `bin_index == category_id` before invoking
  `fit_joint_multi_output_with_categorical`. Strategy mirrors the
  single-output path: collect unique non-NaN integer category values,
  sort them, assign category IDs in sort order, overwrite the binned
  column. The `_JOINT_SUPPORTED_KWARGS` allow-list re-adds
  `categorical_feature_indices` and `max_cat_threshold`.
- **Joint GOSS (closes the v0.10.3 joint-GOSS follow-up):** new
  `select_joint_row_indices_for_round` helper inside
  `crates/engine/src/joint.rs` mirrors
  `select_row_indices_for_round_multiclass`: per-row score is
  `s_i = Σₖ |g_{i,k}|` across the K per-output gradient buffers, a
  single row mask is shared across all buffers, and the amplification
  factor mutates every per-output gradient/hessian in lockstep so
  histograms remain unbiased.
- **Joint DART (closes the v0.10.3 joint-DART follow-up):** dropout /
  normalize cycle added to `fit_joint_inner`. One tree per round on
  the joint trainer simplifies bookkeeping vs. multiclass DART:
  `dart_state.tree_weights` has length `rounds_completed` and
  `dart_round_start_offsets[r]` / `dart_round_counts[r]` collapse to
  a flat per-round pair. Reuses `engine::dart::{select_dropouts,
  apply_normalization}` unchanged. Per-stump `tree_weight` persists
  via the existing `DartTreeWeights` artifact section.
  `JointPredictor` extended with `tree_weights: Vec<f32>` parallel to
  `rounds`, so `predict_row` multiplies each tree's leaf contribution
  by `tree_w` (collapsing to v0.10.2 behavior when every weight is
  1.0).
- **Joint warm-start (closes the v0.10.3 joint-warm-start follow-up):**
  new `JointWarmStartState { baselines, stumps,
  initial_rounds_completed, initial_dart_tree_weights }` + new
  `fit_joint_multi_output_with_warm_start` entry point.
  `MultiLabelGBMRanker(multi_label_mode='joint', warm_start=True,
  init_model=<prior_fit>)` cracks open the prior fit's joint
  artifact, replays prior stumps onto `predictions` via the shared
  `walk_tree_into_predictions` helper, re-encodes new-round `node_id`
  starting at `initial_rounds_completed`, and (under DART)
  reconstructs `dart_state.tree_weights` from per-stump `tree_weight`.
  Per-round seeds mix `global_round = round + initial_rounds`, so an
  N+M warm-resumed fit produces identical RNG draws to a fresh N+M
  fit on rounds N..N+M.
- **Refactor:** the v0.10.0 in-loop joint tree walk became a shared
  `walk_tree_into_predictions(tree_stumps, ..., sign, scale)` helper,
  used by round-end add, DART dropout subtract/re-add, and warm-start
  replay. `fit_joint_multi_output_with_categorical` now delegates to a
  private `fit_joint_inner`, matching the single-output engine's
  `fit_iterations*` → inner-impl pattern.

## What Shipped In v0.10.2

- **Joint trainer core feature parity (closes part of the v0.10.x
  joint-path-feature-parity follow-up):**
  `engine::joint::fit_joint_multi_output` now supports
  `tree_growth="leaf"` + `max_leaves` via the new
  `build_joint_round_leafwise` (priority-queue best-first growth
  keyed by K-output split gain), `interaction_constraints` (reusing
  the single-output `InteractionConstraintIndex`), `min_split_gain`,
  `row_subsample`, and `col_subsample`. All five are exposed through
  `MultiLabelGBMRanker(multi_label_mode="joint")` Python surface;
  `_JOINT_SUPPORTED_KWARGS` grew to permit `min_split_gain`,
  `row_subsample`, `col_subsample`, `interaction_constraints`,
  `tree_growth`, and `max_leaves`. Native-categorical splits land
  at the Rust level (`find_best_multi_output_categorical_split` +
  `fit_joint_multi_output_with_categorical`) but the Python surface
  is intentionally not wired in v0.10.2 — the current bridge bins
  with `ContinuousBinningStrategy::Linear` which doesn't preserve
  `bin_index == category_id` for joint mode, so
  `categorical_feature_indices` / `max_cat_threshold` are rejected
  in joint mode and tracked for v0.10.3.
- **Leaf-wise multiclass DART (closes the v0.10.x leaf-wise
  multiclass DART follow-up):** the v0.10.1 `tree_growth='level'`
  restriction in `fit_multiclass_iterations_impl` was lifted.
  Per-class `dart_round_start_offsets[k]` / `dart_round_counts[k]`
  bookkeeping snapshots `class_stumps[k].len()` around each
  `build_tree_*` call, which is growth-mode-agnostic — under
  leaf-wise growth each tree contributes a variable stump count
  (capped by `max_leaves`) but the round boundaries are still
  captured correctly. Validation early-stopping DART transition and
  DART warm-start tree-weight reconstruction both work without
  changes.

## What Shipped In v0.10.1

- **`MultiLabelGBMRanker(multi_label_mode="joint")`**: new PyO3 entry
  point `train_joint_multi_label_ranker` + `JointPredictorHandle`
  py-class wrap the v0.10.0 Rust joint trainer
  (`engine::joint::fit_joint_multi_output`). Default mode remains
  `"independent"`; joint is opt-in. The new kwarg is named
  `multi_label_mode` (not `training_mode`) to avoid colliding with
  `GBMRanker.training_mode` (MorphBoost selector). Bundle format
  bumped to v2 with an explicit mode byte; v1 bundles still load as
  independent. Strict allow-list (`n_estimators`, `learning_rate`,
  `seed`, `max_depth`, `min_data_in_leaf`, `lambda_l2`, `max_bin`)
  rejects every other kwarg until joint-path feature parity lands.
  `_normalize_group_for_joint` accepts both LightGBM group-sizes and
  per-row IDs, stable-sorts rows by group before fitting.
- **Multiclass softmax + GOSS** (`GBMClassifier(boosting_mode="goss")`
  with K ≥ 3 classes): per-row score `s_i = Σₖ |g_{i,k}|` (LightGBM
  convention) drives a shared sampling mask across all K class
  gradient buffers. The multiclass round loop was refactored to
  pre-compute K gradient buffers before sampling.
- **Multiclass softmax + DART** (`GBMClassifier(boosting_mode="dart")`
  with K ≥ 3 classes): per-class prediction vectors get per-round
  subtract/readd of dropped tree contributions scaled by
  `dart_state.tree_weights`. Per-class `dart_round_start_offsets[k]` +
  `dart_round_counts[k]` arrays index the full stump slice each
  (round, class) tree occupies in `class_stumps[k]`. After K new
  trees are built each round they are rescaled to
  `new_w = 1/(n_dropped + 1)` and dropped trees are re-added at
  their rescaled weights; `stump.tree_weight = new_w` is stamped on
  every stump in the new round's per-class slice. Requires
  `tree_growth="level"`.
- **Multiclass DART warm-start**:
  `MultiClassWarmStartState.initial_dart_tree_weights` carries flat
  round-major × class-k per-tree weights from the prior fit. The
  PyO3 bridge reconstructs the per-tree weights by grouping
  `class_stumps[k]` by `tree_id` decoded from
  `node_id / TREE_NODE_STRIDE` (matching the predictor's
  `apply_dart_tree_weights` convention).
- **DART round acceptance correctness**: `dart_state` mutations and
  `tree_weight` stamping are now deferred to the round-accept branch
  (previously committed before loss check, which desynced state on
  rejection). Rejection paths restore `class_predictions` from
  `dart_predictions_backup`.
- **Validation DART parity**: multiclass DART now mirrors the
  single-output validation transition (subtract dropped at w_old →
  add new K trees at new_w → re-add dropped at w_new) on
  `validation_class_predictions`, so `next_validation_loss` and
  early-stopping decisions are computed against the same full
  ensemble the model is training against.

## What Shipped In v0.10.0

Infrastructure release: laid the Rust-level foundation for joint
multi-output learning and closed the v0.9.0 `DART + warm_start`
follow-up. The new `MultiOutputHistogram` primitive
(`crates/engine/src/shared_histogram.rs`), `MultiOutputLeafValues`
artifact section (kind=13), and the joint trainer in
`crates/engine/src/joint.rs` (`fit_joint_multi_output` +
`JointPredictor`) all landed; v0.10.1 (above) wired the Python
surface.

The `0.9.0` release closes the v0.8.0 DART placeholder (Limitation 2)
and resolves the linear-rank predict-path NaN routing bug
(Limitation 4 from v0.8.0).  `boosting_mode="dart"` is fully wired
through the single-output trainer (`GBMRegressor`, binary
`GBMClassifier`, `GBMRanker`) with four LightGBM-style parameters:
`dart_drop_rate`, `dart_max_drop`, `dart_normalize_type`,
`dart_sample_type`.  Per-stump `tree_weight: f32` is persisted via a
new `DartTreeWeights` artifact section (kind=12), emitted only when
DART is active — Standard/GOSS artifacts stay byte-identical to
v0.8.0.  Multiclass softmax + DART and DART + `warm_start` are
rejected with clear errors; both are tracked as v0.10.x follow-ups.

The `0.8.0` release closes Limitation 4 (mixed linear-rank SHAP
strict additivity) and adds LightGBM-style GOSS (gradient-based
one-side sampling) as a new opt-in `boosting_mode="goss"` on all three
estimators (binary classifier path; multiclass softmax explicitly
rejects non-Standard modes pending per-class gradient scoring).
Default `boosting_mode="standard"` is byte-identical to v0.7.5.

The original v0.8.0 plan also targeted DART boosting mode and joint
shared-tree multi-label ranking, but both were scope-split out to
v0.9.0 and v0.10.0 respectively so this release could ship on a
reviewable surface.  v0.9.0 lands DART (above); v0.10.0 lands joint
multi-label.

The `0.7.5` release closes the last pre-existing v0.7.x SHAP
correctness gap — Limitation 5 from v0.7.4 (TreeSHAP polynomial-path
additivity drift on trees with a feature appearing more than once on a
root-to-leaf path).  Root cause: the Rust port of `ts_unextend_path`
shifted the entire `PathElement` struct (including `pweight`) when
removing a duplicate from the path; the reference implementation in
`slundberg/shap` uses four parallel arrays and only shifts the first
three, preserving the post-unwind pweights computed in place by the
unwind loop.  The fix shifts only `feature_index`, `zero_fraction`,
and `one_fraction` and leaves `pweight` alone.  No user-visible API
breakage.

The `0.7.4` release closes the remaining v0.7.x SHAP-additivity
carryover: strict additivity (`atol + rtol·|predict(x)|`) now holds for
`leaf_model="linear"` artifacts on the default predictor-aligned binning
path.  The fix walks the row's full path and credits
`Σⱼ wⱼ·(xⱼ − μⱼ)` at every visited node — matching how `predict`
accumulates `leaf.eval_row(row)` — across `GBMRegressor`,
`GBMClassifier`, `GBMRanker`, `training_mode="manual"` and `"morph"`,
both binning strategies, with or without `interaction_constraints`.
No user-visible API breakage.

The `0.7.3` release closes the four limitations queued in v0.7.2:
SHAP additivity tolerance (`atol + rtol * |p|`), SHAP path-walker
alignment with predictor float thresholds (new `BinningContext`),
MorphBoost warm-start EMA persistence (MorphMetadata artifact section
v2), and the pyo3 0.23 → 0.24 upgrade (clears RUSTSEC-2025-0020).
No user-visible API breakage.

The `0.7.2` release was documentation, supply-chain, and repo-hygiene
only — no user-facing Python API changes.  It aligned the docs with
the v0.7.1 surface that actually shipped, hardened CI (full pytest
suite gated on every PR, `cargo-audit` + `cargo-deny` weekly), added
an `examples/` library, and rewrote
`docs/reference/release_checklist.md` as a top-to-bottom operating
manual.

The `0.7.1` release built on the v0.7.0 factor-neutral boosting surface
with five additions: SHAP support for piecewise-linear leaves, per-round
training diagnostics on every estimator, neutralized warm-start (with a
matching-exposures contract), LightGBM-compatible feature interaction
constraints, and `MultiLabelGBMRanker` for multi-output ranking.

The `0.7.0` release introduced factor-neutral boosting, with fit-time factor
exposures, pre-target residualization, per-round gradient projection, and an
optional split exposure penalty. The `0.6.0` release introduced
`leaf_solver="dro"`, a conservative DRO-style scalar leaf solver that penalizes
within-leaf gradient uncertainty while preserving standard prediction-time
artifacts. The `0.5.0` release introduced piecewise-linear (PL) tree leaves via
`leaf_model="linear"` on all three estimators. The `0.4.0` release introduced
the opt-in MorphBoost adaptive split criterion, per-iteration learning-rate
schedules, and SIMD-accelerated histogram and EMA kernels.

## What Shipped In 0.8.0

Minor feature release.  Closes Limitation 4 (mixed linear-rank SHAP
strict additivity) and adds GOSS as a new opt-in boosting mode.

- **Mixed linear-rank SHAP strict additivity (Limitation 4).**  When
  `continuous_binning_strategy="linear"` triggers per-feature
  rank-based binning on at least one column, `shap_values()`
  previously fell back to the legacy quantize-then-walk path that
  exempts `leaf_model="linear"` artifacts from strict additivity.
  v0.8.0 adds a `BinningContext::LinearRank` variant carrying
  per-feature sorted unique values + global mins/maxs +
  `max_data_bin`.  At the `explain_rows_from_model` entry point SHAP
  internally quantizes raw rows to bin indices (matching exactly the
  predict-time helper `predict_dense_quantized_linear_rank`) and
  dispatches the remainder with `BinningContext::PreBinned` semantics
  so tree traversal and PL-leaf evaluation share the same bin-index
  space the predictor evaluates in.  Strict additivity now holds for
  `leaf_model="linear"` on this path; the architectural contract is
  pinned by
  `test_shap_linear_rank_strict_additivity.py::test_mixed_linear_rank_uses_predictor_aligned_binning`
  (binning_kind must be `linear_rank` when rank flags fire) and
  `test_mixed_linear_rank_strict_additivity` for both
  `leaf_model="constant"` and `leaf_model="linear"`.
- **GOSS sampling (gradient-based one-side sampling).**  New
  `boosting_mode="goss"` opt-in on `GBMRegressor`, `GBMClassifier`
  (binary), and `GBMRanker` with `goss_top_rate` (default `0.2`) and
  `goss_other_rate` (default `0.1`) controlling kept-top and
  sampled-low fractions.  Implements the LightGBM algorithm: score
  rows by `|gradient|`, keep the top fraction, sample from the rest,
  amplify sampled-low rows by `(n - top_n) / other_n` (using
  realized counts so `ceil()` rounding and the `other_n <= n - top_n`
  cap don't bias the gradient-sum estimator at small `n`).  Engine
  `crates/engine/src/sampling.rs`-style logic lives inline in the
  engine crate's lib.rs as `goss_sample_indices` +
  `select_row_indices_for_round`.  Multiclass softmax explicitly
  rejects non-Standard boosting modes pending per-class gradient
  scoring (v0.9.x follow-up).  DART (`boosting_mode="dart"`) is a
  placeholder in v0.8.0: parses + passes `validate_train_params` so
  parameter ranges remain testable, but the Python `__init__` raises
  `NotImplementedError` and the Rust single-output trainer rejects
  it at the entry point.  Full DART implementation targeted at
  v0.9.0.

## What Shipped In 0.7.5

Bug-fix release.  Closes Limitation 5 from v0.7.4 — the pre-existing
TreeSHAP polynomial-path additivity drift.

- **TreeSHAP polynomial-path strict additivity.**  `ts_unextend_path`
  in `crates/shap/src/lib.rs` previously shifted the entire
  `PathElement` struct (`feature_index`, `zero_fraction`,
  `one_fraction`, **`pweight`**) when removing a duplicate feature
  from the path.  The `pweight` shift clobbered the values the
  unwind loop had just recomputed in place.  The reference Python
  implementation in `slundberg/shap/explainers/pytree.py` stores the
  four fields as four parallel arrays and only shifts the first
  three, preserving pweights.  The Rust fix shifts those three
  fields explicitly and leaves `pweight` alone.  The formerly
  `@xfail(strict=True)` regression
  `test_strict_additivity_via_tree_shap_polynomial_path` in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` now
  passes as a regular test.  In-crate coverage:
  `tree_shap_polynomial_path_matches_brute_force_on_full_trees`
  sweeps depths 2-7 × n_features {2,3,5,8,12}, including all
  configurations that force path-duplicate features, and asserts
  polynomial matches brute-force per-feature within 1e-5.

## What Shipped In 0.7.4

Bug-fix release.  Closes the remaining v0.7.x carryover documented in
`docs/limitations.md` for SHAP strict additivity on `leaf_model="linear"`
artifacts.

- **SHAP strict additivity for PL leaves.**  Pre-v0.7.4,
  `distribute_linear_terms_for_row` credited the per-feature deviation
  `Σⱼ wⱼ·(xⱼ − μⱼ)` only at each tree's terminal leaf.  The predictor
  accumulates `leaf.eval_row(row)` at **every visited node** along the
  row's path, so SHAP was uncrediting one `Σⱼ wⱼ·(xⱼ − μⱼ)` per internal
  node per tree per row — producing additivity gaps on the order of the
  predictions themselves.  v0.7.4 walks the full path and credits the
  linear deviation at every visited leaf; the brute-force Shapley and
  TreeSHAP polynomial paths share the same helper so both get the fix.
  The `model_has_linear_leaves` exemption in `verify_additivity` is now
  gated on `binning.is_none()` — the predictor-aligned `BinningContext`
  callers (default Python path for continuous features) get the strict
  tolerance check.  Coverage: 44 new regression tests in
  `bindings/python/tests/test_shap_pl_strict_additivity.py`.

## What Shipped In 0.7.3

Bug-fix release.  Closes the four limitations queued in v0.7.2 and
clears RUSTSEC-2025-0020.

- **SHAP additivity tolerance.**  `atol + rtol * |predicted|` (numpy
  `allclose` semantics) replaces the fixed `1e-5` absolute bound.
  Larger explanation batches on healthy `leaf_model="constant"`
  artifacts no longer raise spurious additivity-check `RuntimeError`s
  due to accumulated f32 round-off.
- **SHAP path-walker uses predictor-aligned float thresholds.**  New
  `shap::BinningContext` enum (`Linear` / `Quantile` / `PreBinned`)
  plus four PyO3 entry points with binning kwargs.  Path walkers
  compare against the same float thresholds the predictor uses, with
  the predictor's strict-`<` semantics.  Eliminates the path-walk vs.
  predict-path divergence on continuous features for scalar-leaf
  artifacts.  The Python estimators automatically thread feature
  mins / maxs / cuts into SHAP, so `model.shap_values()` and
  `model.feature_importances()` Just Work on continuous data.
- **MorphBoost warm-start persists EMA.**  MorphMetadata artifact
  section bumped to v2 with appended `Vec<GradientEmaStats>` per
  class.  `WarmStartState` and `MultiClassWarmStartState` carry an
  optional EMA snapshot; both training loops seed the fresh
  `MorphState.ema_stats` from it.  Legacy v1 artifacts decode with
  empty `ema_stats` and fall back to the cold-EMA initialization.
- **PyO3 0.23 → 0.24 (clears RUSTSEC-2025-0020).**  Zero source
  changes — bindings were already on the `Bound<>`-first API.
  `deny.toml` and the cargo-audit CI step no longer ignore the
  advisory.

## What Shipped In 0.7.2

Documentation, supply-chain, and repo-hygiene release.  No user-facing
Python API surface changes.

- **Doc accuracy.**  Multiple docs that still claimed warm-start was
  rejected, SHAP required `leaf_model="constant"`, interaction
  constraints did not exist, or rankers were single-label only — even
  though v0.7.1 shipped all four — are now consistent with the
  actual API.  Touches README, `docs/user/*.md`, the Sphinx mirror
  under `docs/site/source/*.rst`, `docs/roadmap/current.md`,
  `CLAUDE.md`, `AGENTS.md`, and `benchmarks/README.md`.
- **Release operating manual.**
  `docs/reference/release_checklist.md` is now the authoritative
  inventory of version-pin files, content updates, audit `git grep`
  queries, verification matrix, tag/publish commands, and
  post-release bookkeeping.
- **Runnable examples.**  New `examples/` directory with 8 end-to-end
  scripts covering every public estimator and feature.
- **CI now runs the full pytest suite.**  v0.7.1 built the wheel and
  ran 7 smoke snippets but never invoked
  `pytest bindings/python/tests/` — meaning the 455-test Python suite
  was not enforced on merge.
- **Cargo.lock tracked**, `maturin` pinned in `publish.yml`,
  `cargo-audit` + `cargo-deny` weekly + on every Cargo-manifest PR,
  coverage reporting via Codecov, `publish = false` on every workspace
  crate.
- **Repo metadata.**  `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue /
  PR / CODEOWNERS / Dependabot configs, `.editorconfig`,
  `requirements-dev.txt`, README badges.

## What Shipped In 0.7.1

- **SHAP for piecewise-linear leaves** — `shap_values()` accepts
  `leaf_model="linear"` artifacts and returns an interventional
  decomposition (path-attributed leaf "constant part" plus per-leaf
  row deviations); global feature means are persisted in a new
  `FeatureBaseline` artifact section so SHAP is self-contained at
  explain time.
- **Per-round training diagnostics** — every estimator exposes
  `diagnostics_per_round_`: gradient L2 norm / variance, hessian L2
  norm, sampling counts, and (when factor neutralization is active)
  the `neutralization_effectiveness` score in `[0, 1]`.
- **Neutralized warm-start** — `init_model` / `warm_start=True` works
  across `pre_target`, `per_round_gradient`, and `split_penalty`
  provided the caller supplies the same `factor_exposures` matrix used
  for the initial fit; mode + lambda + (where applicable) penalty must
  match.
- **Feature interaction constraints** — LightGBM-compatible
  `interaction_constraints=[[…]]` on every estimator, up to 64 groups
  per fit, enforced in both level-wise and leaf-wise tree builders.
- **`MultiLabelGBMRanker`** — unified multi-output ranking estimator:
  `y` shaped `(n_rows, n_labels)`, `predict` returns the same shape.
  Trains one independent `GBMRanker` per label sharing `group` and
  `factor_exposures`; supports per-label `ranking_objective` lists.

## What Shipped In 0.7.0

- **Factor-neutral boosting** via `neutralization` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`, with row-aligned fit-time
  `factor_exposures`.
- **Per-round gradient projection** via `neutralization="per_round_gradient"`,
  projecting objective gradients away from user-supplied factors before each
  boosting round. Multiclass classification projects each class-gradient column
  independently.
- **Pre-target residualization** via `neutralization="pre_target"` for built-in
  squared-error `GBMRegressor` training. Classification, ranking, custom
  objectives, and validation sets are rejected for this mode in 0.7.0.
- **Split exposure penalty** via `neutralization="split_penalty"` and
  `factor_penalty`, compatible with constant leaves, DRO leaves, and
  MorphBoost. Piecewise-linear leaves are rejected for split-penalty mode in
  0.7.0.
- **Benchmark coverage**: `alloygbm_factor_neutral` and
  `alloygbm_factor_neutral_dro` arms were added to the comparative benchmark
  runner. Synthetic benchmark factors are smoke/stability checks unless callers
  provide domain factor exposures explicitly.

## What Shipped In 0.6.0

- **DRO-style scalar leaves** via `leaf_solver="dro"` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`. This is a fast closed-form robust Newton
  update over leaf gradient uncertainty, exposed with `dro_radius` and
  `dro_metric="wasserstein"`.
- **Conservative contract**: default `leaf_solver="standard"` preserves existing
  behavior; `dro_radius=0.0` preserves standard predictions while recording
  optional DRO metadata; the DRO solver does not claim full raw-distribution
  Wasserstein DRO guarantees.
- **Interactions**: `leaf_solver="dro"` composes with `training_mode="morph"`
  and requires `leaf_model="constant"` for this release.
- **Benchmark support**: `alloygbm_dro` was added to the comparative benchmark
  runner, with temporal/panel stability reporting focused on mean, worst, and
  standard deviation of task-normalized score.

## What Shipped In 0.5.0

- **Piecewise-linear (PL) tree leaves** via `leaf_model="linear"` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`. Each leaf stores a small linear model
  `f_s(x) = b_s + Σ α_j x_j` whose weights are solved in closed form via the
  ridge regression `α* = -(XᵀHX + λI)⁻¹ Xᵀg`, using the same L2 regularizer
  (`lambda_l2`) as the split criterion. Benchmarks show:
  - ~10× faster convergence on linearly-structured datasets (fewer rounds to reach
    the same RMSE)
  - +3.5% RMSE improvement on California Housing vs constant leaves
  - +1.75pp accuracy improvement on Breast Cancer classification
  - 2–8× training time overhead (Cholesky solve per node)
- **New artifact section** (`ModelSectionKind::LinearLeafCoefficients`) stores
  per-stump linear leaf data; backward-compatible with v0.4.0 artifacts
- **`alloygbm_linear` benchmark arm** in `run_model_comparison.py`; new
  `benchmarks/pl_trees_benchmark.py` script with convergence-curve and λ-sweep
  analysis; report at `docs/benchmarks/pl_trees_v1.md`
- Categorical-native splits continue to use constant leaves when
  `max_cat_threshold > 0`; descendant leaves below a categorical root node use
  linear leaves on all remaining numeric regressors

## What Shipped In 0.4.0

- **MorphBoost adaptive training mode** (`training_mode="morph"`) on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`. Implements the criterion from [Kriuk (2025)](https://arxiv.org/pdf/2511.13234) with EMA-driven gain shaping, depth/iteration leaf penalties, balance penalty, and an information-theoretic blend term ramped in via `tanh(iter/20)` warmup
- **Per-iteration learning-rate schedules** via the new `lr_schedule` parameter (`"constant"` default or `"warmup_cosine"`); schedule-aware auto early-stopping logic so warmup-phase rounds aren't classified as stalled
- **MorphBoost configuration persisted in artifacts** as an optional section so loaded models predict consistently
- **SIMD acceleration** via the `wide` crate (safe API, AVX2/NEON internally, scalar fallback): histogram bin-scan and EMA mean+variance pass are vectorized
- **Tile-size auto-tuning** for histogram parallelism on high-feature workloads (~2 tiles per thread, clamped to `[16, 64]`)
- **`alloygbm_morph` / `alloygbm_morph_cosine` benchmark arms** in `run_model_comparison.py`; new `--models` filter; new `morph_report.py`, `morph_ablation.py`, and updated `numerai_benchmark.py` harnesses (with build-freshness self-check at startup)
- **Dedicated MorphBoost user guide** at `docs/user/morphboost.md` (and Sphinx mirror) plus cross-references across all estimator docs and READMEs

## What Shipped In 0.3.2

- Fixed GBMRanker silent zero-tree training: the auto training policy's density-based `min_split_gain` floor and `min_loss_improvement` floor were being applied to ranking objectives, which have gradient magnitudes an order of magnitude smaller than regression/classification — no split cleared the floor and training exited on round 1. The auto policy is now objective-aware and skips those floors for all ranking objectives.
- Fixed training loop loss-regression early break firing on ranking objectives where round-to-round loss oscillation is expected and benign
- Fixed `inspect.signature(GBMRanker.__init__)` returning only 3 parameters (`self`, `ranking_objective`, `**kwargs`) — parameter-building tools (sklearn clone, benchmarks, IDEs) using signature introspection silently trained with `n_estimators=6` default; now exposes the full parameter set
- Added `stop_reason_` and `rounds_completed_` attributes on all estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) for training diagnostics
- Added `california_ranking` benchmark scenario: California Housing reframed as learning-to-rank with geographic grid cells as queries and median house value bucketed into 5 graded relevance levels (~44 queries × 468 docs)

## What Shipped In 0.3.1

- Fixed multiclass predictor threshold conversion: `class_trees` are now converted in all three threshold-conversion paths (linear, quantile, pre-binned); continuous-feature multiclass models now produce correct predictions
- Fixed multiclass benchmark argmax label mapping: `model.classes_` is now used so accuracy is correct for non-zero-indexed labels
- Added real-dataset benchmark scenarios: `wine_multiclass`, `digits_multiclass`, `adult_income`, `abalone_regression`
- Added `news_ranking` placeholder scenario with dataset selection instructions
- Activated `synthetic_multiclass` and `synthetic_categorical` benchmark scenarios
- Rewrote `benchmarks/README.md` with scenario table, feature coverage matrix, timing reference, and usage examples

## What Shipped In 0.3.0

- Native categorical splits with Fisher-sort algorithm and bitset-based O(1) prediction (`max_cat_threshold`)
- Multi-class classification (`GBMClassifier` with softmax/multinomial for K > 2 classes)
- Custom objective functions (`objective=callable`) with fast numpy I/O
- Custom evaluation metric callbacks (`eval_metric=callable`) with early stopping support
- Synthetic categorical and custom objective benchmark scenarios

## What Shipped In 0.2.0

- Binary classification (`GBMClassifier`) with log-loss objective
- Learning-to-rank (`GBMRanker`) with 5 objectives (RankNet, LambdaMART, XE-NDCG, QueryRMSE, YetiRank)
- NaN / missing value support across all crates
- Sample weight and group ID support from Python
- Model persistence (pickle, save/load, artifact export)
- Feature name capture and sklearn compatibility (`BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- TreeSHAP (polynomial-time, replaces the old 25-feature-capped brute-force method)
- Monotone constraints and feature importance weighting
- Leaf-wise (best-first) tree growth strategy
- Warm-starting / incremental training
- Up to 65,535 bins per feature (up from 256)
- Multiple categorical column support
- Histogram buffer reuse
- Objective-aware training metric tracking
- Expanded benchmark suite (regression + classification + ranking)

## Current Priorities

1. Close remaining performance gaps on broad tabular datasets.
2. Explore GPU/accelerator backend after the CPU baseline is solid enough to serve as reference.
3. Continue expanding the benchmark suite with real-world classification and ranking datasets.

## Longer-Term Themes

- Joint shared-tree multi-label ranking (one ensemble updating all label
  predictions simultaneously) — the v0.7.1 `MultiLabelGBMRanker` is a
  K-independent-rankers wrapper; a shared-tree engine is a v0.7.2+ follow-up.
- Path-walk alignment between SHAP and the predictor for piecewise-linear
  leaves (so strict additivity holds on continuous-feature artifacts).
- MorphBoost EMA snapshot persisted in the warm-start artifact so resumed
  training does not restart the EMA cold.
- Dart / GOSS boosting modes.
- GPU backend.

## Planning Style

The project no longer uses the old version-layer planning hierarchy as the active documentation model.

Going forward:

- current intent lives in `docs/roadmap/`
- research notes live in `docs/ideas/`
- benchmark framing lives in `docs/benchmarks/` and `benchmarks/`
- implementation plans from the 0.1.x cycle are archived in `docs/archive/v0.1_plans/`
