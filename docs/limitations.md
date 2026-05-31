# AlloyGBM Current Limitations

Last updated for v0.12.4.

## Remaining Limitations

### 1. CPU-Only Runtime

The `BackendOps` trait is designed for hardware abstraction, but only
`CpuBackend` exists. GPU/accelerator support is architecturally planned but
not implemented.



### 3. SHAP interactions on multi-output / multiclass models

`shap_interaction_values()` is `GBMRegressor`-only.  Multi-output
(joint multi-label ranker) and multiclass softmax models don't have an
interaction-values surface in v0.11.0.

### 4. GLM and Quantile regression objectives on Ranker / Classifier / multiclass

`objective="poisson"`, `"gamma"`, `"tweedie"`, and `"quantile"` are supported only on
single-output `GBMRegressor`. The Ranker, Classifier, and multiclass
softmax paths reject these objectives.

### 5. `neutralization="pre_target"` is squared-error-only

The pre-target factor-neutralization mode residualizes targets before
training. For non-squared-error objectives the
residualize-target == residualize-gradient identity that pre_target
relies on breaks down (the gradient under log-link is `μ − y`, not
`pred − y`). Use `"per_round_gradient"` or `"split_penalty"` with GLM
objectives.
### 6. Quantile regression is rejected for linear leaves, DART, and MorphBoost

The `"quantile"` objective uses empirical leaf refinement which assumes constant leaves, standard boosting (non-DART), and standard training mode (non-MorphBoost). These combinations are rejected at the Python and Rust layers.

## Resolved (Previously Limitations)

### v0.12.5

- **SHAP interactions on `leaf_model="linear"`:** `GBMRegressor.shap_interaction_values(X)` now supports linear leaves. The linear deviation terms `w_j · (x_j − μ_j)` are credited directly to the diagonal of the interaction matrix (the regressor feature's main effect), preserving both row-marginal and full additivity invariants according to standard TreeSHAP conventions for linear dependencies.
### v0.11.1

- **Quantile regression objective shipped.** `GBMRegressor(objective="quantile", quantile_alpha=…)` adds pinball loss regression with alpha ∈ (0.0, 1.0). Uses a proxy Hessian `h_i = w_i` during split-finding, an empirical quantile leaf refinement step at the end of each round acting on the full dataset, and a fast unweighted quickselect path. Explicitly rejected for linear leaves, DART, MorphBoost, classification, ranking, and joint training.

### v0.11.0

- **SHAP interaction values shipped.** `GBMRegressor.shap_interaction_values(X)`
  returns the `(n_rows, n_features, n_features)` pairwise SHAP tensor in
  `O(T · L · D² · M)` time via Lundberg et al. (2020) Algorithm 2 (ported
  verbatim from `slundberg/shap`'s canonical `tree_shap_recursive`).
  Row marginal recovers per-feature SHAP exactly; full sum reconstructs
  the prediction within `atol = 1e-5 + rtol = 1e-4 · |predict(x)|`.

- **Poisson / Gamma / Tweedie GLM objectives shipped.**
  `GBMRegressor(objective="poisson"|"gamma"|"tweedie",
  tweedie_variance_power=…)` adds log-link GLM regression to the
  single-output regressor. Standard Newton gradients/hessians;
  predictor-side `exp(raw)` post-transform; target-domain validation;
  new deviance metrics in `alloygbm.evaluation`. Composes with
  DART/GOSS/leaf-wise/warm-start/MorphBoost/per-round-gradient and
  split-penalty factor neutralization.

### v0.10.6

- **Joint-path advanced feature parity is COMPLETE.** The last v0.10.4-deferred
  follow-up — factor neutralization on the joint multi-output trainer — shipped
  in v0.10.6. `MultiLabelGBMRanker(multi_label_mode="joint",
  neutralization=…, factor_exposures=…)` now supports all three modes
  (`pre_target`, `per_round_gradient`, `split_penalty`) with the same surface
  as the single-output `GBMRegressor` / `GBMRanker`. New
  `effective_neutralization_config(params)` helper (in
  `crates/engine/src/joint.rs`) mirrors v0.10.5's `effective_dro_config`:
  returns `Some(cfg)` only when the config is non-inert. Both growth paths
  AND the artifact serializer consult this helper. New
  `ModelSectionKind::NeutralizationMetadata = 14` records the active config
  in the artifact (metadata only). `pre_target` requires every per-output
  objective to be `squared_error`; ranking objectives must use the other two
  modes. `split_penalty` generalizes the single-output K-output factor-load
  formula via the new `compute_multi_output_factor_split_penalty` helper in
  `shared_histogram.rs`. Byte-equivalent to v0.10.5 when neutralization is
  inert; pinned by `joint_neutralization_inert_configs_match_v0_10_5_byte_for_byte`.

### v0.10.5

- **Joint DRO leaves (was a v0.10.5 follow-up):**
  `MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro",
  dro_radius=…, dro_metric="wasserstein")` now applies
  Wasserstein-distributionally-robust leaf values on the joint
  multi-output trainer. Reuses `alloygbm_core::leaf_effective_gradient`
  (the same helper `GBMRegressor` / `GBMRanker` have used since v0.6.x).
  Applied in-build inside `build_joint_round_inner`'s `leaf_values`
  closure and `build_joint_round_leafwise`'s per-output leaf computation
  — row indices are already in scope at leaf-computation time. DRO is
  leaf-only: split-gain dispatch still uses the standard K-output
  sum-of-XGBoost-gains (multi-output histogram doesn't carry per-bin
  `grad_sq`; adding it would cost ~1.5× joint-round memory which isn't
  justified before benchmark evidence). Byte-equivalent to v0.10.4 when
  `lambda_l1 == 0` AND (`dro_config.is_none()` OR
  `dro_config.radius == 0.0`); pinned by
  `joint_dro_radius_zero_matches_standard_byte_for_byte` (cargo) and
  `test_joint_dro_radius_zero_byte_equivalent_to_standard` (pytest).
  Works under both `tree_growth="level"` and `tree_growth="leaf"`, and
  composes with MorphBoost (`training_mode="morph"`) and DART/GOSS
  boosting modes. `_JOINT_SUPPORTED_KWARGS` adds `leaf_solver`,
  `dro_radius`, `dro_metric`.

### v0.10.4

- **Joint MorphBoost (was a v0.10.4 follow-up):**
  `MultiLabelGBMRanker(multi_label_mode="joint",
  training_mode="morph", …)` now activates MorphBoost on the
  shared-tree multi-output trainer. Two new helpers in
  `crates/engine/src/shared_histogram.rs`
  (`compute_multi_output_split_gain_morph`,
  `find_best_multi_output_categorical_split_morph`) sum per-output
  morph gain across K outputs, each using its own EMA snapshot from
  `MorphState::ema_stats[k]`. `JointMorphContext` (private to
  `joint.rs`) carries the per-round morph snapshot through
  `build_joint_round*`. Per-iteration LR schedule, per-leaf depth
  penalty, and per-iteration leaf shrinkage all apply uniformly across
  the K-output leaf values. MorphBoost EMA persists through the
  artifact's `MorphMetadata` section;
  `JointWarmStartState.initial_ema_stats` re-seeds
  `MorphState::ema_stats` on warm-resume so gradient-statistics
  smoothing is continuous across the resume boundary. (Warm-resume is
  NOT byte-equivalent to a fresh longer fit — see the
  `joint_morph_warm_resume_preserves_ema_continuity_not_byte_equivalence`
  regression — because per-iteration leaf shrinkage and LR schedule are
  resolved against the original training horizon and not re-scaled on
  resume; mirrors single-output behavior.) The morph row-count proxy
  uses `morph_count_proxy` (`ceil(h).max(1)` when `h > 0`, `0`
  otherwise) so any positive-hessian bin contributes ≥ 1 count to the
  post-warmup `info_side` and balance-penalty terms — necessary for
  ranking objectives where per-row hessians are well below 1 (PR #37
  review C2). Warmup byte-equivalence with the standard K-output gain
  holds regardless.

### v0.10.3

- **Joint native-categorical Python wiring (was a v0.10.3 follow-up):**
  the PyO3 bridge `train_joint_multi_label_ranker` now re-bins requested
  columns to `bin_index == category_id` before calling
  `fit_joint_multi_output_with_categorical`, mirroring the single-output
  `apply_categorical_encoding_to_training_matrices_multi`. The
  `_JOINT_SUPPORTED_KWARGS` allow-list re-adds
  `categorical_feature_indices` and `max_cat_threshold`.
- **Joint GOSS (was a v0.10.3 follow-up):** new
  `select_joint_row_indices_for_round` helper in `joint.rs` scores rows
  by `s_i = Σₖ |g_{i,k}|` across the K per-output gradient buffers
  (LightGBM multiclass GOSS convention). A single row mask is shared
  across all K buffers; the amplification factor mutates every
  gradient/hessian in lockstep so histograms remain unbiased.
- **Joint DART (was a v0.10.3 follow-up):** dropout/normalize cycle
  added to `fit_joint_inner`. One tree per round simplifies bookkeeping
  vs. multiclass DART: `dart_state.tree_weights` has length
  `rounds_completed` and `dart_round_start_offsets[r]` /
  `dart_round_counts[r]` collapse to a flat per-round pair.
  `JointPredictor` extended with `tree_weights: Vec<f32>` so the
  per-tree weight is applied at predict time. Reuses
  `engine::dart::{select_dropouts, apply_normalization}` unchanged.
- **Joint warm-start (was a v0.10.3 follow-up):** new
  `JointWarmStartState` + `fit_joint_multi_output_with_warm_start`
  entry point. Prior stumps are replayed onto `predictions` (using the
  shared `walk_tree_into_predictions` helper); new-round `node_id` is
  re-encoded starting at `initial_rounds_completed`; per-round seeds
  mix `global_round = round + initial_rounds` so warm-resumed N+M
  matches fresh N+M. DART warm-resume reconstructs
  `dart_state.tree_weights` from per-stump `tree_weight`.

### v0.10.2

- **Joint trainer core feature parity (was the v0.10.x follow-up):** the
  joint multi-output trainer (`engine::joint::fit_joint_multi_output`) now
  supports `tree_growth="leaf"` + `max_leaves` (via the new
  `build_joint_round_leafwise` priority-queue best-first growth),
  `interaction_constraints` (reusing the single-output
  `InteractionConstraintIndex`), `min_split_gain`, `row_subsample`,
  and `col_subsample`. All five features are exposed through the
  `MultiLabelGBMRanker(multi_label_mode="joint")` Python surface; the
  `_JOINT_SUPPORTED_KWARGS` allow-list grew accordingly. Native-categorical
  was partially shipped in v0.10.2 (Rust-level
  `find_best_multi_output_categorical_split` +
  `fit_joint_multi_output_with_categorical` in place; Python surface
  closed in v0.10.3). GOSS, DART, and warm-start on the joint path
  shipped in v0.10.3. MorphBoost shipped in v0.10.4. DRO shipped in
  v0.10.5. Factor neutralization remains deferred to **v0.10.6**
  (see "Remaining Limitations" above).
- **Leaf-wise multiclass DART (was a v0.10.x follow-up):**
  `GBMClassifier(boosting_mode="dart")` with K ≥ 3 classes now works
  under `tree_growth="leaf"` + `max_leaves`. The v0.10.1 level-wise
  restriction in `fit_multiclass_iterations_impl` was lifted; the
  per-class `dart_round_start_offsets[k]` /
  `dart_round_counts[k]` bookkeeping is growth-mode-agnostic because
  it snapshots `class_stumps[k].len()` around each `build_tree_*`
  call — under leaf-wise growth each tree has a variable stump count
  (capped by `max_leaves`) but the round boundaries remain captured
  correctly. Validation early-stopping DART transition and DART
  warm-start tree-weight reconstruction both work without changes
  (verified by regression tests in
  `bindings/python/tests/test_multiclass_dart.py`).

### v0.10.1

- **Multi-Label Ranking joint mode — Python surface (was a v0.10.0
  follow-up):** `MultiLabelGBMRanker(multi_label_mode="joint")` now
  routes through the new PyO3 entry point
  `train_joint_multi_label_ranker` and `JointPredictorHandle` py-class
  to `engine::joint::fit_joint_multi_output`. The kwarg is named
  `multi_label_mode` (not the originally-planned `training_mode`) to
  avoid colliding with `GBMRanker.training_mode` (MorphBoost selector
  `"manual"` / `"morph"`), which would have broken v0.7.1 callers
  passing `training_mode="morph"` through the wrapper. Bundle format
  bumped to v2 with an explicit mode byte; v1 bundles still load as
  independent mode.
- **Multiclass softmax + GOSS (was a v0.10.x follow-up):** per-row
  score `s_i = Σₖ |g_{i,k}|` (LightGBM convention) drives a shared
  sampling mask across all K class gradient buffers. New helper
  `select_row_indices_for_round_multiclass` in
  `crates/engine/src/lib.rs`. The multiclass round loop was refactored
  to pre-compute K gradient buffers BEFORE row sampling so the
  multiclass scorer can see every class channel before deciding which
  rows to keep / amplify.
- **Multiclass softmax + DART (was a v0.10.x follow-up):** per-class
  prediction vectors get per-round subtract/readd of dropped tree
  contributions scaled by `dart_state.tree_weights`. Dropout flat
  index `prior_round * K + class_k` resolves to a single stump in
  `class_stumps[class_k][prior_round]`. After K new trees are built
  each round they are rescaled to `new_w = 1/(n_dropped + 1)` and
  `stamp_tree_weight` is committed onto each new stump.
  `MultiClassWarmStartState.initial_dart_tree_weights` (flat
  round-major × class-k) enables warm-start continuation. Requires
  `tree_growth="level"`.

### v0.10.0

- **DART + warm_start not yet supported (was a v0.9.0 follow-up):**
  v0.10.0 enables the combination. `WarmStartState` carries an
  optional `initial_dart_tree_weights` field; the engine seeds
  `dart_state.tree_weights` from this snapshot and pre-populates
  `round_start_offsets` / `dart_round_counts` so new-round dropouts
  correctly subtract/replay prior trees. The Python `init_model`
  warm-start path automatically detects when the prior fit used DART
  and forwards the per-stump weights through. Historical RNG-driven
  `dropped_per_round` is intentionally not persisted; new rounds
  start fresh dropout bookkeeping going forward.

### Previously resolved

The following were limitations in prior versions and have been addressed:

- No DART boosting mode (now: v0.9.0 — `boosting_mode="dart"` is
  fully supported for `GBMRegressor`, binary `GBMClassifier`, and
  `GBMRanker`. Four DART parameters expose the LightGBM-style API:
  `dart_drop_rate` (default 0.1), `dart_max_drop` (default 50),
  `dart_normalize_type` (`"tree"` or `"forest"`, default `"tree"`),
  and `dart_sample_type` (`"uniform"` or `"weighted"`, default
  `"uniform"`). The per-round dropout + normalization cycle lives in
  `crates/engine/src/dart.rs`; per-stump `tree_weight: f32` is
  persisted via a new `DartTreeWeights` artifact section (kind index
  12) that is only emitted when DART is active, keeping Standard /
  GOSS artifacts byte-identical to v0.8.0. At v0.9.0 ship time,
  multiclass DART and DART + warm_start were still rejected with
  clear errors; v0.10.0 resolves DART + warm_start, and multiclass
  DART remains a v0.10.x follow-up.)
- NaN routing on the linear-rank predict path (now: v0.9.0 — the
  predict-time quantize helpers
  (`quantize_dense_values_linear_inplace_wide`,
  `quantize_dense_values_linear_rank_inplace_wide`, and the inline
  loop in `predict_dense_quantized_with_summary_bytes`) now preserve
  `f32::NAN` through the f32 cast instead of falling through to bin
  0. The predictor's existing `feature_value.is_nan() → default_left`
  short-circuit at `crates/predictor/src/lib.rs:148` then fires
  automatically. `LinearLeaf::eval` and `LinearLeafCompact::eval`
  also skip NaN regressor features when accumulating the linear sum,
  preventing `w · NaN` poisoning of PL-leaf predictions. Pure-linear
  and pure-quantile paths now share consistent NaN semantics with
  the rank-binning path. Closes Limitation 4 from v0.8.0.)
- Regression-only (now: classification + ranking)
- Single categorical column only (now: multiple via `categorical_feature_indices`)
- Limited configurability (now: `min_split_gain`, monotone constraints, feature weights, `max_leaves`, leaf-wise growth)
- No NaN support (now: native NaN handling)
- No model persistence (now: pickle, save/load, artifact export)
- No sklearn compatibility (now: `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`)
- No sample weight / group ID from Python (now: fully supported)
- Feature names auto-generated only (now: captured from DataFrames)
- SHAP limited to 20 features (now: TreeSHAP with no practical limit)
- Only RMSE tracked during training (now: objective-aware metric tracking)
- No warm-starting (now: `warm_start=True` for all estimators including multiclass)
- Level-wise growth only (now: leaf-wise available)
- Bins capped at 256 (now: up to 65,535)
- No histogram reuse (now: buffer reuse across rounds)
- Binary classification only (now: multi-class softmax with K > 2 classes)
- No native categorical splits (now: `max_cat_threshold` enables Fisher-sort optimal binary partitions with O(1) bitset prediction)
- No custom objective/metric callbacks (now: `objective` callable and `eval_metric` callable)
- Multiclass warm-start unsupported (now: `warm_start=True` works for multiclass with round-offset continuity)
- Neutralized warm-start unsupported (now: v0.7.1 — `init_model` / `warm_start=True`
  with `neutralization=*` is supported as long as the caller supplies the
  same `factor_exposures` matrix used for the initial fit)
- No interaction constraints (now: v0.7.1 — `interaction_constraints=[[...]]`
  on every estimator, LightGBM-compatible semantics, up to 64 groups per
  fit; enforced through both the level-wise and leaf-wise tree builders)
- Single-label ranking only (partially resolved: v0.7.1 — `MultiLabelGBMRanker`
  exposes a unified `fit`/`predict` for K ranking labels per item, trained
  as K independent per-label `GBMRanker` instances sharing `group` and
  `factor_exposures`.  Joint shared-tree training is a follow-up; see
  Limitation 3)
- MorphBoost warm-start restarts EMA cold (now: v0.7.3 — MorphMetadata
  artifact section bumped to v2 with appended `Vec<GradientEmaStats>`;
  `WarmStartState` / `MultiClassWarmStartState` carry the snapshot and
  the engine seeds `MorphState.ema_stats` from it on resumed fits)
- SHAP additivity tolerance was f32-tight at `1e-5` absolute (now:
  v0.7.3 — `atol + rtol * |predicted|`, numpy `allclose` semantics)
- SHAP path-walker used bin-index thresholds instead of float
  thresholds (now: v0.7.3 — `shap::BinningContext` + new PyO3 entry
  points pass per-feature mins / maxs / cuts so the walker compares
  against the same float thresholds the predictor uses; resolved for
  scalar-leaf artifacts on continuous features.  The PL-leaf piece
  was finished in v0.7.4 — see the dedicated entry below)
- RUSTSEC-2025-0020 in `pyo3 < 0.24.1` (now: v0.7.3 — pyo3 0.23.5 →
  0.24, `deny.toml` and the cargo-audit CI step no longer ignore the
  advisory)
- SHAP on the mixed linear-rank binning path (now: v0.8.0 — new
  `BinningContext::LinearRank` variant carries per-feature sorted
  unique values + global mins/maxs + `max_data_bin`.  At the
  `explain_rows_from_model` entry point SHAP internally quantizes
  rows to bin indices (linear-quantize unflagged features,
  rank-quantize flagged features — exactly matching
  `predict_dense_quantized_linear_rank`) and dispatches the
  remainder with `BinningContext::PreBinned` semantics, so tree
  traversal and PL-leaf evaluation share the same bin-index space
  as the predictor.  Strict additivity now holds for
  `leaf_model="linear"` on the mixed linear-rank path; the
  predictor-aligned binning kwargs `_shap_binning_kwargs()`
  returns include `binning_kind="linear_rank"` whenever any
  per-feature rank flag is set.  Closes Limitation 4.)
- Strict SHAP additivity for `leaf_model="linear"` on continuous
  features (now: v0.7.4 — `distribute_linear_terms_for_row` credits
  `wⱼ · (xⱼ − μⱼ)` at every visited node along the row's path,
  matching how `predict` accumulates `leaf.eval_row(row)` at each
  visited node; the linear-leaf exemption in `verify_additivity` was
  removed when a `BinningContext` is supplied — i.e. for the default
  Python path on continuous features.  Strict additivity now holds
  for `GBMRegressor`, `GBMClassifier`, and `GBMRanker` under
  `leaf_model="linear"`, with `training_mode="manual"` or
  `"morph"`, on both `quantile` and `linear` binning, with or without
  `interaction_constraints`, across `lambda_l2`, `max_depth`,
  `n_estimators` and skewed-scale features.  The legacy non-binning
  path retains the exemption.)
- Multiclass prediction per-row allocation (now: zero-copy dense slice prediction)
- Single fixed split criterion (now: opt-in MorphBoost adaptive criterion via
  `training_mode="morph"`, including EMA-driven gain shaping, depth/iteration
  leaf penalties, and an info-theoretic blend term — see the
  [paper](https://arxiv.org/pdf/2511.13234))
- Constant-only learning rate (now: per-iteration `lr_schedule` parameter
  supports `"constant"` and `"warmup_cosine"`, with schedule-aware
  early-stopping logic)
- No SIMD-accelerated kernels (now: histogram bin-scan and EMA passes are
  vectorized via the `wide` crate; histogram tile sizing auto-tunes for
  high-feature workloads)
- Constant leaves only (now: `leaf_model="linear"` replaces scalar leaves with
  closed-form piecewise-linear leaves `f_s(x) = b_s + Σ α_j x_j`, available on
  all three estimators; `leaf_model="polynomial"` and `leaf_model="rff"` remain
  future work)
- No full raw-distribution Wasserstein DRO guarantee (now:
  `leaf_solver="dro"` provides a fast Wasserstein-inspired robust scalar leaf
  update over gradient uncertainty; exact distributional DRO is still research
  work)
- TreeSHAP polynomial-path additivity drift on trees with a feature
  appearing more than once on a root-to-leaf path (now: v0.7.5 —
  `ts_unextend_path` in `crates/shap/src/lib.rs` now shifts only
  the `feature_index`, `zero_fraction`, and `one_fraction` fields
  when removing a duplicate from the path, preserving the `pweight`
  values that the unwind loop has just computed in place.  The
  previous implementation shifted the entire `PathElement` struct,
  clobbering the post-unwind pweights with values from the elements
  being shifted down.  The reference implementation in
  `slundberg/shap` uses four parallel arrays and only shifts the
  first three.  Closes the pre-existing Limitation 5 from v0.7.3 /
  v0.7.4.  The Python regression
  `test_strict_additivity_via_tree_shap_polynomial_path` in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` is no
  longer `@xfail(strict=True)`.)
