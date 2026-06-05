# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, supporting regression, binary and multi-class classification, and learning-to-rank. It is aimed at strong practical performance on structured tabular workloads, with particular strength on financial and time-aware problems.

The `0.12.8` release narrows limitation #4 from `docs/limitations.md`: the GLM (`"poisson"`, `"gamma"`, `"tweedie"`) and `"quantile"` objectives now work on `GBMRanker` and `MultiLabelGBMRanker` (both `multi_label_mode="independent"` and `"joint"`) in addition to single-output `GBMRegressor`. Only the Classifier / multiclass softmax paths still reject these objectives.

**No artifact format change.** The joint `.alloy` bundle metadata grew a `ranking_objective` field (v3) but older bundles still load; binary model artifacts are unchanged. Test counts: 448 cargo + 653 pytest.

## What Shipped In v0.12.8

### GLM and Quantile objectives on rankers (#54)

Limitation #4 had restricted `"poisson"` / `"gamma"` / `"tweedie"` / `"quantile"` to single-output `GBMRegressor`. This release extends them to the ranker estimators:

- **`GBMRanker`**: accepts the GLM and quantile objectives via `ranking_objective=`. Because `GBMRanker` subclasses `GBMRegressor`, training and the artifact-recorded post-transform (predictor applies `exp` for GLM) come for free; the `objective="quantile"` rejection guards were removed.
- **`MultiLabelGBMRanker`**: both modes accept per-label GLM/quantile objectives (including mixed lists). Independent mode fans out to per-label `GBMRanker` instances. Joint mode extends `JointObjective` with `Poisson` / `Gamma` / `Tweedie { variance_power }` / `Quantile { alpha }` variants (delegating to the existing single-output `ObjectiveOps` impls), adds a joint empirical-quantile leaf-refinement pass (`refine_joint_quantile_leaves`), and applies the GLM `exp` post-transform on the Python predict surface (the joint predictor returns raw log-space scores).
- **Persistence fix**: the joint `.alloy` bundle now persists `ranking_objective` in its v3 metadata so the GLM post-transform survives a `save_model`/`load_model` roundtrip — previously a reloaded joint GLM model silently returned raw log-space predictions. Pinned by a new `test_multilabel_ranker_glm_save_load_roundtrip` regression test.

## What Shipped In v0.12.7

### Quantile regression compatibility extended (#53)

Limitation #6 had restricted the use of the `"quantile"` objective with DART, MorphBoost, and linear leaves. This release implements full mathematical and engine-level compatibility for these settings:
- **DART boosting** (`boosting_mode="dart"`): Leaf refinement operates correctly on dropped-out residuals by leveraging the DART prediction buffer evaluation before refinement.
- **MorphBoost** (`training_mode="morph"`): Leaf refinement scales intercept updates by MorphBoost per-round shrinkage (`iter_shrinkage` aligned to `tree_build.rs`) and depth-based penalty.
- **Piecewise-linear leaves** (`leaf_model="linear"`): Leaf refinement calculates residual targets by subtracting the linear portion of predictions from training values (correctly walking from root to terminal leaf to accumulate parent-relative delta weights for `max_depth >= 2`), and only refines the flat leaf intercept (avoiding double-scaling of build-time solved linear slopes).
- **Linear leaves + quantile numeric test**: Added a new robust, multi-feature, `max_depth >= 4` numeric test `test_quantile_linear_leaves_numeric` verifying that linear-leaf quantile regression fits linear relationships significantly better than standard constant-leaf models and that path-level weights accumulate correctly.

## What Shipped In v0.12.6

### Multiclass and multi-output SHAP (#52)

Limitation #3 had been the last "Remaining Limitations" entry in `docs/limitations.md` related to the explanation surface. Closing it required threading SHAP through two artifact shapes that don't fit the single-`TrainedModel` assumption built into TreeSHAP: multiclass softmax (K independent tree sequences via `MultiClassTrainedModel.class_stumps`) and joint multi-output (one shared tree per round with `MultiOutputLeafValues` storing absolute per-output K-vectors per stump).

The approach: unroll the artifact into K independent `TrainedModel` instances inside `load_artifact_context` and run standard TreeSHAP K times. For multiclass, each per-class `TrainedModel` gets its own `baseline_predictions[k]` plus the shared `FeatureBaseline` and `NativeCategoricalSplits` sections (per-stump `categorical_bitset` threaded back into each class's stumps via the global stump index). For joint multi-output, each per-stump multi-output K-vector is residualized against its parent's K-vector so each unrolled `TrainedModel`'s scalar-leaf representation is what standard TreeSHAP expects; per-output baselines are parsed from the joint trainer's objective metadata string (`joint_multi_output:obj1+obj2|baselines=v1,v2,...`) with explicit `ShapError::ContractViolation` on parse failure.

Four new Rust public entry points return `Vec<ShapExplanationBatch>` / `Vec<ShapInteractionBatch>`; the legacy single-output entry points stay unchanged in signature and now error on K>1 artifacts. Eight new PyO3 `_multi` wrappers expose them to Python. `_ShapMixin` detects multi-output (`n_classes_ > 2` or `multi_label_mode == "joint"`) and dispatches to the `_per_output` bridge automatically. `MultiLabelGBMRanker(multi_label_mode="joint")` inherits `_QuantizationMixin` + `_ShapMixin` so binning context is preserved through the SHAP path; independent mode fans out to per-label `GBMRanker.shap_values`.

Four new pytest cases in `test_shap_multiclass_multioutput.py` pin additivity + symmetry + row-marginal invariants for multiclass softmax (with and without `leaf_model="linear"`), joint multi-output `shap_values`, and joint multi-output `shap_interaction_values`.

Internal refactors (PR #52): `load_artifact_context` decomposed into `unroll_multiclass`, `parse_joint_baselines`, `unroll_multi_output` helpers; `bindings/python/src/predict.rs` split into a 396-line `predict.rs` (predictor entry points only) and a new 567-line `shap_bridge.rs` containing all 16 SHAP PyO3 wrappers. Continues the v0.12.2 / v0.12.3 decomposition pattern.

## What Shipped In v0.12.5

### SHAP interaction values on `leaf_model="linear"` (#51)

PL-leaf artifacts were the last remaining rejection on `GBMRegressor.shap_interaction_values(X)` after v0.11.0 shipped the interactions surface for scalar leaves. The math: each PL leaf value decomposes as `intercept + Σⱼ wⱼ·μⱼ` (constant, the same shape standard TreeSHAP wants) plus `Σⱼ wⱼ·(xⱼ − μⱼ)` (row-dependent). Standard TreeSHAP runs on the constant part; the per-row deviation is then folded onto the diagonal of the interaction matrix via `distribute_linear_terms_for_row` — the same helper that backs PL-leaf `shap_values`, so row-marginal `Σⱼ Φᵢⱼ = φᵢ` is automatic by construction, and adding to the diagonal preserves full additivity `Σᵢⱼ Φᵢⱼ + E = ŷ`. The matrix stays symmetric. The diagonal-only attribution is a pragmatic choice that loses path-feature × regressor-feature interaction credit — captured deliberately in `docs/limitations.md` as a future-research item, not claimed as the canonical algorithm.

Tests pin all three invariants on both sides: the renamed `test_shap_interaction_values_accepts_linear_leaf_model` (pytest) and the new `shap_interactions_linear_leaves_satisfies_additivity` + `shap_interactions_linear_leaves_mixed_with_scalar_leaves_satisfies_additivity` (cargo) all verify additivity + symmetry + row-marginal-matches-`shap_values`. A new `test_shap_interaction_values_linear_rank_with_linear_leaves` exercises the LinearRank × linear-leaves binning combo via `ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK=1` to confirm the quantize-and-recurse path also satisfies all three invariants.

Internal refactor (PR #51): `explain_interactions_from_model` moved from `crates/shap/src/lib.rs` to `crates/shap/src/tree_shap.rs` next to its peer `explain_rows_tree_shap`. Continues the v0.12.2 SHAP-crate decomposition pattern — `lib.rs` is back to thin entry-point glue (~165 lines).

## What Shipped In v0.12.4

### `GBMRegressor.__module__` public-shim identity (#48)

After the v0.12.3 `_regressor/` package decomposition, `GBMRegressor.__module__` was the private `alloygbm._regressor._core` (the module where the class is defined), which leaked the internal package layout through `repr` and made newly-created pickles depend on `_regressor._core` staying importable forever. v0.12.4 sets `GBMRegressor.__module__ = "alloygbm.regressor"` after the class definition so the public shim path is the canonical one. Old v0.12.3 pickles still load (the class object remains accessible at the private path); new pickles use the public path. Pinned by `bindings/python/tests/test_module_identity.py` (2 tests).

### Joint trainer module-doc refresh (#49)

`crates/engine/src/joint/mod.rs`'s module-level docstring claimed the v0.10.0 minimal scope (no DART/GOSS/MorphBoost/DRO/neutralization/leaf-wise/warm-start/native-categorical/interaction-constraints), but all of those features had been added to the joint path over v0.10.1–v0.10.6. Docstring rewritten to describe the current capability matrix accurately with per-feature release tags, plus an explicit note that this `mod.rs` is the scaffolding / re-export layer added in v0.12.2 (PR #46) — the actual implementation lives in the sibling modules (`helpers`, `types`, `build_round`, `fit`, `tests`).

## What Shipped In v0.12.3

### PyO3 bridge restructure (Phase 6)

`bindings/python/src/lib.rs` (6,619 lines) decomposed into 9 sibling modules (`errors`, `callbacks`, `pyclasses`, `quantization`, `params`, `categorical_bridge`, `predict`, `train`, `joint`) plus a slim `lib.rs` (mod decls, shared `pub(crate)` consts, the `#[pymodule]` registration, the shared `dense_rows_from_flat_values` helper) and an extracted `tests/` submodule. Eleven commits, one per logical extraction plus the test extraction. Every pyfunction/pyclass stays registered and importable unchanged; 445 cargo + 641 pytest green at every commit.

### GBMRegressor estimator restructure (Phase 7)

`regressor.py` (4,909 lines) decomposed into a `_regressor/` package via the mixin pattern (method bodies moved byte-identically; `GBMRegressor` assembled from `_ValidationMixin`, `_QuantizationMixin`, `_ShapMixin`, `_PersistenceMixin` over the `_GBMRegressorBase`). `_base` holds module-level helpers and the native loaders, which are now invoked as `_base._load_native_*()` so the white-box contract test's 119 monkeypatches retarget to a single stable module. `regressor.py` is a back-compat shim; `GBMClassifier`/`GBMRanker` subclassing is preserved. Nine commits, 641 pytest green throughout.

### Cross-cutting (Phase 8)

CLAUDE.md Project Structure and architectural-pointer refresh for the new layouts. Deferred (documented): the `crates/engine/src/trainer/mod.rs` `use crate::*;` glob tightening (explicit list would exceed ~50 entries) and the single `#[allow(dead_code)]` in `crates/engine/src/factor.rs` (predates this release).

The full inventory is in the v0.12.3 CHANGELOG entry. The post-refactor file layouts are reflected in `CLAUDE.md`'s Project Structure section.

## What Shipped In v0.12.2

### SHAP crate restructure (Phase 4)

`crates/shap/src/lib.rs` decomposed into 8 sibling modules (`error`, `types`, `binning`, `linear_leaf`, `importance`, `brute_force`, `tree_shap`, `tests/`). Nine small commits, one per logical extraction plus a final layout-cleanup pass. Every commit kept the 445 workspace cargo tests and 641 pytest tests passing.

### Engine joint trainer restructure (Phase 5)

`crates/engine/src/joint.rs` promoted to a `crates/engine/src/joint/` subdir with `mod.rs` reduced to 42 lines of scaffolding and 5 sibling modules (`helpers`, `types`, `build_round`, `fit`, `tests`). Six small commits — `git mv` to the subdir, then four content extractions, then a no-op verification pass. Same test-suite invariant maintained at every commit.

The full inventory is in the v0.12.2 CHANGELOG entry. The post-refactor file layouts are reflected in `CLAUDE.md`'s Project Structure section.

## What Shipped In v0.12.1

### Core crate restructure (Phase 2)

`crates/core/src/lib.rs` decomposed into 13 sibling modules (`error`, `dro`, `neutralization`, `training_mode`, `config`, `dataset`, `binned`, `histogram`, `linear_histogram`, `leaf`, `artifact_format`, `validation`, `tests/`). Thirteen small commits, one per logical extraction. Every commit kept the 445 workspace cargo tests and 641 pytest tests passing.

### Backend CPU crate restructure (Phase 3)

`crates/backend_cpu/src/lib.rs` decomposed into 5 sibling modules (`arena`, `split_helpers`, `factor_split`, `backend_ops`, `tests/`). The giant `impl CpuBackend { ... }` intrinsic-method block stays in lib.rs intact — splitting an inherent impl across files in Rust requires per-file `impl` blocks, which adds boilerplate without clear payoff at this scale. Five small commits. Same test-suite invariant maintained at every commit.

The full inventory is in the v0.12.1 CHANGELOG entry. The post-refactor file layouts are reflected in `CLAUDE.md`'s Project Structure section.

## What Shipped In v0.12.0

### Engine crate restructure

`crates/engine/src/lib.rs` decomposed into 28 sibling modules + 5 trainer submodules. Twenty-four small commits, one per logical extraction. Every commit kept the 207 engine tests, 445 workspace tests, and 641 pytest tests passing.

The full inventory and motivation is in the v0.12.0 CHANGELOG entry. The post-refactor file layout is reflected in `CLAUDE.md`'s Project Structure section. The follow-up plan covering the remaining 6 large files in the repo (Python bindings Rust glue, joint trainer, `regressor.py`, core, backend_cpu, shap) is at `docs/superpowers/plans/2026-05-23-refactor-large-files.md`.

## What Shipped In v0.11.1

### Quantile regression objective

`GBMRegressor` accepts a new quantile regression objective (`objective="quantile"`) with pinball loss semantics and parameter `quantile_alpha` (default `0.5`, strictly in `(0.0, 1.0)`):

- **Empirical Quantile Leaf Refinement**: At the end of each round, a custom post-growth leaf refinement step (`refine_quantile_leaf_values`) is run to replace Newton-Raphson leaf predictions with the actual empirical quantiles of residuals for all rows in each leaf.
- **Full-dataset refinement**: Under `row_subsample < 1.0`, split-finding runs on the subsampled subset, but leaf refinement uses the entire training set to minimize the estimation variance of the empirical quantile.
- **Proxy Hessian**: Since the pinball loss has a zero second derivative everywhere, a proxy Hessian `h_i = w_i` (sample weight) is used during split-finding.
- **Quickselect optimization**: The unweighted refinement path uses a fast `O(N)` quickselect algorithm (`select_nth_unstable_by`) instead of sorting `O(N log N)`.
- **Validation**: Gated validation ensures that invalid `quantile_alpha` settings are only rejected when `objective="quantile"` is active, leaving non-quantile models unaffected.

Scope limit: Single-output `GBMRegressor` only. Rejects combinations with DART boosting, MorphBoost, linear leaves (`leaf_model="linear"`), classification, ranking, and joint multi-output training.

## What Shipped In v0.11.0

### SHAP interaction values (Lundberg Algorithm 2)

`GBMRegressor.shap_interaction_values(X)` returns the
`(n_rows, n_features, n_features)` pairwise SHAP-interaction tensor in
`O(T · L · D² · M)` time. Implements Lundberg et al. (2020) "From local
explanations to global understanding with explainable AI for trees"
Algorithm 2, ported verbatim from `shap/cext/tree_shap.h`'s
`tree_shap_recursive`. Three invariants are pinned by tests:

- **Symmetric**: `values[r][i][j] == values[r][j][i]`.
- **Row-marginal**: `Σ_j values[r][i][j] == shap_values(X)[r][i]`.
- **Full additivity**: `Σ_i Σ_j values[r][i][j] + expected_value
  == predict(x)` within `atol = 1e-5 + rtol = 1e-4 · |predict(x)|`.

The diagonal is filled from the row-marginal invariant (the "main
effect" of each feature after subtracting off-diagonals). New Rust
crate-level surface: `alloygbm_shap::explain_interactions_from_artifact_bytes`,
`_with_binning`, and `ShapInteractionBatch`. Four new PyO3
pyfunctions wrap them.

Scope limit: constant-leaf artifacts only. `leaf_model="linear"` is
rejected by the entry point with a clear error. Multi-output and
multiclass interactions are deferred.

### Poisson / Gamma / Tweedie regression objectives

`GBMRegressor` accepts three new GLM regression objectives with
log-link semantics (`predict()` returns `exp(raw)`):

- `objective="poisson"` — count regression. Targets must be `>= 0`.
  Gradients: `(μ − y) · w`; hessians: `μ · w`; loss: Poisson deviance.
- `objective="gamma"` — strictly-positive continuous regression.
  Targets must be `> 0`. Gradients: `(1 − y/μ) · w`; hessians:
  `(y/μ) · w`; loss: Gamma deviance.
- `objective="tweedie"` — compound Poisson-gamma for
  `1 < variance_power < 2`. Set via new
  `tweedie_variance_power: float = 1.5` constructor kwarg. Targets must
  be `>= 0`. Gradients: `(μ^(2-p) − y·μ^(1-p)) · w`; hessians:
  `μ^(2-p) · w` (LightGBM/XGBoost simplified Newton form).

All three use weighted-mean-in-log-space initial predictions, reuse
the standard `ObjectiveOps` machinery (Newton-Raphson leaves), and
compose with DART/GOSS/leaf-wise/warm-start/MorphBoost,
`neutralization="per_round_gradient"`, and `"split_penalty"`. The
`"pre_target"` mode remains squared-error-only.

Three new deviance metrics in `alloygbm.evaluation`:
`poisson_deviance(y_true, y_pred)`, `gamma_deviance(y_true, y_pred)`,
`tweedie_deviance(y_true, y_pred, variance_power=p)`.

Scope limit: single-output `GBMRegressor` only. Not on `GBMRanker`,
`GBMClassifier`, multiclass softmax, or the joint multi-output ranker.

## What Shipped In v0.10.6

The `0.10.6` release closes the last v0.10.4-deferred joint-path
follow-up: all three factor-neutralization modes (`pre_target`,
`per_round_gradient`, `split_penalty`) now work on the joint
multi-output trainer via the same `neutralization=` /
`factor_exposures=` surface as the single-output `GBMRegressor` /
`GBMRanker`. The joint trainer reaches full feature parity with
the single-output path. A new
`ModelSectionKind::NeutralizationMetadata` artifact section
records the active config so joint models are self-describing.
Default behaviour for every existing user-facing API remains
byte-identical to v0.10.5 when neutralization is not opted into.

The `0.10.5` release closed the joint DRO leaves follow-up from
v0.10.4: `MultiLabelGBMRanker(multi_label_mode="joint",
leaf_solver="dro")` routes per-output leaf values through
`alloygbm_core::leaf_effective_gradient` (the same helper the
single-output trainers have used since v0.6.x). Default behaviour
for every existing user-facing API remained byte-identical to
v0.10.4 when DRO is not opted into.

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

## What Shipped In v0.10.6

Closes the last v0.10.4-deferred joint-path follow-up. The joint
multi-output trainer now reaches full feature parity with the
single-output path. Default behaviour for every existing user-facing
API remains byte-identical to v0.10.5 when neutralization is not
opted into (pinned by
`joint_neutralization_inert_configs_match_v0_10_5_byte_for_byte`).

- **Joint factor neutralization (closes the v0.10.6 follow-up):**
  `MultiLabelGBMRanker(multi_label_mode="joint", neutralization=…,
  factor_exposures=…)` supports all three modes:
  - `pre_target` — residualize each per-output target once before
    training via `FactorProjector::residualize_values_in_place`.
    Squared-error only (the only objective where residualize-target
    equals residualize-gradient).
  - `per_round_gradient` — build a `FactorProjector` once, project
    each of the K gradient buffers in place every round. Mirrors the
    single-output multiclass per-class projection pattern.
  - `split_penalty` — per-candidate K-output factor-load penalty via
    new `compute_multi_output_factor_split_penalty` helper in
    `shared_histogram.rs`. Threaded through both level-wise
    (`build_joint_round_inner`) and leaf-wise
    (`build_joint_round_leafwise`) growth paths; the leaf-wise heap
    ranks candidates by penalized gain.

  Wired through a new `effective_neutralization_config(params)` helper
  in `crates/engine/src/joint.rs` mirroring v0.10.5's
  `effective_dro_config` — returns `Some(cfg)` only when the config is
  non-inert (kind ≠ None, AND not SplitPenalty-with-zero-penalty).
  Both growth paths AND the artifact serializer consult this helper, so
  inert configs collapse to byte-equivalent v0.10.5 fits. New
  `ModelSectionKind::NeutralizationMetadata = 14` records the active
  config in the artifact (metadata only; prediction never reads it —
  neutralization is a training-time transformation). Composes with
  MorphBoost (`training_mode="morph"`), DRO leaves
  (`leaf_solver="dro"`), DART, and warm-start.

  `_JOINT_SUPPORTED_KWARGS` gains three entries: `neutralization`,
  `factor_neutralization_lambda`, `factor_penalty`. The
  `factor_exposures=` kwarg on `fit()` (already existed for the
  independent-mode fallback) is now honored on joint too.

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
