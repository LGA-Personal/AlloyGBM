# Changelog

## v0.12.5 (2026-05-31)

Small feature release on top of v0.12.4. Closes the `leaf_model="linear"` exception on SHAP interaction values that was carved out when interactions shipped in v0.11.0.

### Features

- **SHAP interaction values now accept `leaf_model="linear"` artifacts** (#51). `GBMRegressor.shap_interaction_values(X)` previously rejected piecewise-linear (PL) leaf artifacts because the standard TreeSHAP polynomial path operates on constant leaf values. The row-dependent linear deviation `w_j · (x_j − μ_j)` carried by PL leaves is now credited to the diagonal of the interaction matrix (the regressor feature's main effect): standard TreeSHAP interactions run on the constant part of each leaf (`intercept + Σⱼ wⱼ·μⱼ`), then `distribute_linear_terms_for_row` — the same helper that backs PL-leaf `shap_values` — folds the per-row deviations onto `Φ[j][j]`. Mathematical guarantees: full additivity (`Σᵢⱼ Φᵢⱼ + E = ŷ`) and row-marginal (`Σⱼ Φᵢⱼ = φᵢ`) hold by construction; matrix is symmetric and `expected_value` is unchanged (the deviations have zero expectation under the recorded baseline). Pragmatic caveat: this attribution does not split linear-deviation credit across path-feature × regressor-feature off-diagonals; a faithful PL-leaf interaction decomposition remains an open extension. Covered by 3 new tests pinning all three invariants (Rust + Python), plus a new Python test exercising the LinearRank × linear-leaves binning combination.

### Internal refactors

- **`explain_interactions_from_model` moved from `crates/shap/src/lib.rs` to `crates/shap/src/tree_shap.rs`** next to its peer `explain_rows_tree_shap`. Continues the v0.12.2 SHAP-crate decomposition pattern: `lib.rs` is back to thin entry-point glue (~165 lines) and the algorithmic body lives alongside the helpers it consumes (`tree_shap_interactions_row`, `distribute_linear_terms_for_row`, `model_has_linear_leaves`). No behavioral change.

### Documentation

- `crates/shap/src/lib.rs` rustdoc on `explain_interactions_from_artifact_bytes`, `bindings/python/alloygbm/_regressor/_shap.py` docstring on `shap_interaction_values`, `docs/user/explanations.md`, `docs/site/source/explanations.rst`, and the CLAUDE.md SHAP-interactions bullet all updated to describe the new linear-leaf treatment (replacing the v0.11.0 rejection language).

No artifact format change. Test counts: **644 pytest** (the v0.12.4 baseline of 643 plus the renamed-and-extended `test_shap_interaction_values_accepts_linear_leaf_model` and the new `test_shap_interaction_values_linear_rank_with_linear_leaves`) and **447 cargo** (the v0.12.4 baseline of 445 plus the two new `shap_interactions_linear_leaves_*_satisfies_additivity` tests).

## v0.12.4 (2026-05-29)

Bugfix release on top of v0.12.3. Two issues raised by post-merge LLM review of the v0.12.2 and v0.12.3 refactor PRs:

### Bug fixes

- **`GBMRegressor.__module__` now advertises the public shim path** (#48). After the v0.12.3 `_regressor/` package decomposition, `GBMRegressor` was defined inside `alloygbm._regressor._core`, so `__module__` exposed the private implementation path. This leaked the internal layout through `repr` (`<class 'alloygbm._regressor._core.GBMRegressor'>`) and, more importantly, newly-created pickles serialized with that private path — tying the pickle format to internals that the v0.12.3 refactor explicitly framed as private. `GBMRegressor.__module__` is now `"alloygbm.regressor"` (the stable back-compat shim). Old v0.12.3 pickles still load because `_regressor._core` continues to define the class; new pickles use the public path that survives future internal moves. Regression tests added in `bindings/python/tests/test_module_identity.py` pin both invariants (`__module__` equals the public path; pickle payload contains the public path and not the private one).
- **Joint trainer module docs refreshed** (#49). `crates/engine/src/joint/mod.rs`'s module-level documentation still described the original v0.10.0 minimal scope (no DART, no GOSS, no MorphBoost, no DRO, no neutralization, no leaf-wise growth, no warm-start, no native categorical splits, no interaction constraints), but the joint path reached full feature parity over the v0.10.x line. Docstring rewritten to describe the current capability matrix accurately with per-feature release tags (v0.10.2 leaf-wise + interaction constraints + native categorical + row/col subsample; v0.10.3 GOSS + DART + warm-start; v0.10.4 MorphBoost; v0.10.5 DRO leaves; v0.10.6 factor neutralization) plus a note that this `mod.rs` is the scaffolding / re-export layer added in v0.12.2 (PR #46).

No behavioral change, no API change, no artifact format change. Test counts: **643 pytest** (the v0.12.3 baseline of 641 plus the two new `test_module_identity.py` regression tests) and **445 cargo**.

## v0.12.3 (2026-05-29)

Completion of the structural refactor begun in v0.12.0. **No user-facing API changes, no behavioral changes, no new features.** This release decomposes the two remaining large files — the PyO3 bridge and the `GBMRegressor` estimator — into focused, single-responsibility modules. Patch release because every change is mechanical; the full test suite (445 cargo + 641 pytest) holds at every individual commit. Closes the file-decomposition program tracked in issue #44 (Phases 6–8).

### What changed structurally

- **`bindings/python/src/lib.rs`** (the PyO3 bridge crate `_alloygbm`) shrank from **6,619 lines to ~110 lines**. The remaining lines are module declarations, shared `pub(crate)` constants, the `#[pymodule] fn _alloygbm` registration, a shared `dense_rows_from_flat_values` helper, and `#[cfg(test)] mod tests;`. Nine new sibling modules under `bindings/python/src/` host the moved code:
  - `errors.rs` — `PredictorError`/`EngineError`/`ShapError` → `PyErr` converters
  - `callbacks.rs` — `CustomPythonObjective`, `CustomPythonMetricCallback`, numpy transfer helpers
  - `pyclasses.rs` — `NativeRuntimeInfo`, `NativeContinuousBinningMetadata`, `NativeTrainingSummary`, `NativeTrainingResult`, `NativeIterationDiagnostics`, `diagnostics_to_native`
  - `quantization.rs` — dense-value → binned-matrix preparation (`PreparedTrainingMatrices`, all `quantize_*`/`derive_*`/`prepare_*_matrices`, binning-strategy parse/validate)
  - `params.rs` — `build_train_params`, `build_binning_context`, and the `parse_*` config parsers
  - `categorical_bridge.rs` — categorical encoding + the Cholesky/residualize factor-neutralization bridge
  - `predict.rs` — `NativePredictorHandle`, the predict/shap `_impl` functions, and the 16 predictor/shap `#[pyfunction]` wrappers
  - `train.rs` — `train_regression_artifact_with_summary_dense_impl`, the summary builders, and the 5 `train_regression_artifact*` `#[pyfunction]`s
  - `joint.rs` — `JointPredictorHandle` + `train_joint_multi_label_ranker`
  - plus `tests/mod.rs` + `tests/main.rs` (the extracted unit tests)
  - Every previously-registered pyfunction/pyclass remains registered and importable from `alloygbm._alloygbm` unchanged.
- **`bindings/python/alloygbm/regressor.py`** (the `GBMRegressor` estimator, 4,909 lines) was decomposed into a `_regressor/` package, with `regressor.py` reduced to a back-compat shim:
  - `_base.py` — module-level constants, bin-size helpers, the native `_load_native_*` loaders, the sklearn `_GBMRegressorBase`, `_diagnostics_to_dicts`, `_validate_quantile_alpha`
  - `_validation.py` — `_ValidationMixin` (param resolution, objective/neutralization-contract, input validators, categorical inference — 27 methods)
  - `_quantization.py` — `_QuantizationMixin` (`predict_from_artifact`, native-matrix handling, dense/row quantization + binning derivation — 33 methods)
  - `_shap.py` — `_ShapMixin` (`shap_values`, `shap_interaction_values`, `feature_importances`, `_shap_binning_kwargs`)
  - `_persistence.py` — `_PersistenceMixin` (`__getstate__`/`__setstate__`, `save_model`/`load_model`, `save_artifact`, `artifact_bytes`, native predictor-handle, `_reset_fitted_state`)
  - `_core.py` — the `GBMRegressor` shell: `__init__`, `__repr__`, `get_params`/`set_params`, `fit`, `predict`, `score`, sklearn glue (the 11 headline methods) assembled over the four mixins
  - `from alloygbm.regressor import GBMRegressor` and the `alloygbm.regressor` module name are unchanged. `GBMClassifier`/`GBMRanker` continue to subclass `GBMRegressor` transparently via the MRO. Native loaders are now invoked as `_base._load_native_*()` so monkeypatch-based tests retarget cleanly to the `_base` module.

### Deferred (out of scope, documented)

- Tightening the `use crate::*;` glob in `crates/engine/src/trainer/mod.rs` to an explicit list — the list would exceed ~50 entries and hurt readability; engine-crate scope, not this release.
- The single `#[allow(dead_code)]` in `crates/engine/src/factor.rs` — predates this release.

## v0.12.2 (2026-05-27)

Continuation of the structural refactor begun in v0.12.0 and continued in v0.12.1. **No user-facing API changes, no behavioral changes, no new features.** This release decomposes the SHAP crate and the engine joint multi-output trainer into focused, single-responsibility modules. Patch release because every change is mechanical; the full test suite (445 cargo + 641 pytest) holds at every individual commit.

### What changed structurally

- **`crates/shap/src/lib.rs`** shrank from **3,925 lines to 246 lines** (93.7% reduction). The remaining 246 lines are mod declarations, `pub use` re-exports, the public `explain_*` entry points, and the `#[cfg(test)] mod tests;` line. Eight new sibling modules under `crates/shap/src/` host the moved code:
  - `error.rs` — `ShapError` enum, `ShapResult<T>` alias
  - `types.rs` — `ShapBatch`, `ShapInteractionBatch`, `ShapFeatureContribution`, plus shared artifact-loader helpers
  - `binning.rs` — `BinningContext` plus shared binning constants used by both SHAP entry points
  - `linear_leaf.rs` — Linear-leaf (PL) SHAP helpers for interventional decomposition through PL leaves
  - `importance.rs` — `feature_importance_from_artifact_bytes` global importance API
  - `brute_force.rs` — Legacy brute-force Shapley algorithm (`explain_from_artifact_bytes` legacy path)
  - `tree_shap.rs` — TreeSHAP polynomial-time algorithm (Lundberg et al. 2020) for both row-level Shapley values and pairwise interactions
  - `tests/` — extracted `tests/mod.rs` + `tests/main.rs` mirroring the established crate-test pattern

- **`crates/engine/src/joint.rs`** (5,088 lines) was promoted to a `crates/engine/src/joint/` subdir. `joint/mod.rs` is now 42 lines (scaffolding only — mod declarations + `pub use` re-exports). Five new sibling modules under `crates/engine/src/joint/` host the moved code:
  - `helpers.rs` — Private RNG / row-sampling / factor-sums / iteration helpers
  - `types.rs` — `JointObjective`, `JointPredictor`, `JointPredictorStump`, `JointWarmStartState`, `JointMorphContext`, and other joint-specific data types
  - `build_round.rs` — Per-round tree builders (`build_joint_round_inner`, `build_joint_round_leafwise`) for both level-wise and leaf-wise growth
  - `fit.rs` — Public training entry points (`fit_joint_multi_output`, `fit_joint_multi_output_with_categorical`, `fit_joint_multi_output_with_warm_start`) and the shared `fit_joint_inner` driver
  - `tests.rs` — Extracted joint-trainer unit tests

### What did NOT change

Behavioral surface, public API, on-disk artifact format, training output bytes, prediction output bytes. The full test suite — 445 cargo workspace tests + 641 pytest tests — passes at every individual commit (9 commits for the SHAP decomposition, 6 commits for the joint-trainer decomposition).

### Import compatibility

External consumers can keep their existing `use alloygbm_shap::*;` and `use alloygbm_engine::joint::*;` imports unchanged. Every previously-`pub` item is re-exported via `pub use` from the crate root (SHAP) or from `joint/mod.rs` (engine), so paths like `alloygbm_shap::explain_interactions_from_artifact_bytes` and `alloygbm_engine::joint::fit_joint_multi_output` continue to resolve. Items inside the joint subdir that were module-private are now `pub(super)` or `pub(crate)` for sibling-module access but remain inaccessible outside the engine crate.

### Remaining refactor follow-ups

This release closes Phases 4 and 5 of the original decomposition plan (tracking issue [#44](https://github.com/LGA-Personal/AlloyGBM/issues/44)). The remaining phases — the PyO3 binding (Phase 6), the Python regressor (Phase 7), and a cross-cutting verification + `CLAUDE.md` refresh (Phase 8) — ship as separate patch releases.

## v0.12.1 (2026-05-26)

Continuation of the structural refactor begun in v0.12.0. **No user-facing API changes, no behavioral changes, no new features.** This release decomposes two more large monolithic files — `crates/core/src/lib.rs` and `crates/backend_cpu/src/lib.rs` — into focused, single-responsibility modules. Patch release because every change is mechanical; the full test suite (445 cargo + 641 pytest) holds at every individual commit.

### What changed structurally

- **`crates/core/src/lib.rs`** shrank from **4,822 lines to 73 lines** (98.5% reduction). The 73 remaining lines are entirely `mod` declarations, `pub use` re-exports, and the `#[cfg(test)] mod tests;` line. Thirteen new sibling modules under `crates/core/src/` host the moved code:
  - `error.rs` — `CoreError` enum, `CoreResult<T>` alias
  - `dro.rs` — `DroMetric`, `DroConfig`
  - `neutralization.rs` — `NeutralizationKind`, `FactorNeutralizationConfig`, `FactorExposureMatrix`
  - `training_mode.rs` — `LrSchedule`, `MorphConfig`, `MorphPrecomputed`, `TrainingMode`, `GradientEmaStats`
  - `config.rs` — `TreeGrowth`, `LeafModelKind`, `LeafSolverKind`, `DartNormalize`, `DartSampleType`, `BoostingMode`, `Device`, `TrainParams`
  - `dataset.rs` — `DatasetSchema`, `DatasetMatrix`, `DenseMatrixView`, `ColumnarMatrixView`, `ColumnarMatrixColumnView`, `TrainingDataset`
  - `binned.rs` — `MISSING_BIN_U8`/`U16`, `BinStorage`, `BinnedMatrix`, transpose helpers
  - `histogram.rs` — `GradientPair`, `leaf_effective_gradient`, `leaf_gain_term`, `FeatureTile`, `NodeSlice`, `NodeStats`, `HistogramBin`, `FeatureHistogram`, `HistogramBundle`
  - `linear_histogram.rs` — `MAX_PL_REGRESSORS`, `MAX_PL_MATRIX_ENTRIES`, `LinearHistogramBin`, `pl_matrix_index`, `LinearFeatureHistogram`, `LinearHistogramBundle`, `subtract_linear_histogram_bundle`
  - `leaf.rs` — `LinearLeaf`, `LeafValue`, `SplitCandidate`, `PartitionResult`
  - `artifact_format.rs` — All artifact section types, payload encoders/decoders, JSON metadata serde, and the private JSON parsing helpers (1,710 lines — the largest leaf module)
  - `validation.rs` — All `validate_*` functions
  - `tests/` — extracted `tests/mod.rs` + `tests/main.rs` mirroring the Phase 1 engine-crate pattern

- **`crates/backend_cpu/src/lib.rs`** shrank from **3,987 lines to 1,507 lines** (62.2% reduction). The remaining 1,507 lines are predominantly the giant `impl CpuBackend { ... }` intrinsic-methods block (histogram building and split finding) which was intentionally kept intact — splitting an inherent `impl` across files in Rust requires fragmenting it into multiple `impl` blocks per file, which adds boilerplate without proportional payoff. Five new sibling modules under `crates/backend_cpu/src/` host the moved code:
  - `arena.rs` — `HistogramArena`, `HistogramKernelPath`, workload threshold constants
  - `split_helpers.rs` — `GainStrategy`, `ScalarSideStats`, `MissingDirectionCandidate`, `apply_feature_weight`, `l1_threshold_gradient`, `split_gain_term`, `categorical_bitset_for_prefix*`, `goes_left_for_split`
  - `factor_split.rs` — `FactorSplitScratch`, `FactorSplitCandidate`, `factor_split_penalty*`, `validate_factor_split_context`
  - `backend_ops.rs` — The `impl BackendOps for CpuBackend { ... }` trait implementation (449 lines)
  - `tests/` — extracted `tests/mod.rs` + `tests/main.rs`

### What did NOT change

Behavioral surface, public API, on-disk artifact format, training output bytes, prediction output bytes. The full test suite — 445 cargo workspace tests + 641 pytest tests — passes at every individual commit (13 commits for the core decomposition, 5 commits for the backend_cpu decomposition).

### Import compatibility

External consumers can keep their existing `use alloygbm_core::*;` and `use alloygbm_backend_cpu::*;` imports unchanged. Every previously-`pub` item is re-exported via `pub use` from each crate's `lib.rs`, so `alloygbm_core::TrainParams`, `alloygbm_core::ModelMetadata`, etc. all continue to resolve. Items inside `backend_cpu` that were module-private are now `pub(crate)` for sibling-module access but remain inaccessible outside the crate.

### Remaining refactor follow-ups

Tracking issue [#44](https://github.com/LGA-Personal/AlloyGBM/issues/44) lists the remaining four phases of the original decomposition plan: the SHAP crate (Phase 4), the engine joint trainer (Phase 5), the PyO3 binding (Phase 6), the Python regressor (Phase 7), and a cross-cutting verification + CLAUDE.md refresh (Phase 8). These ship as separate patch releases.

## v0.12.0 (2026-05-25)

Engine crate refactor. **No user-facing API changes, no behavioral changes, no new features.** This is a structural release: `crates/engine/src/lib.rs` was a 15,189-line monolith covering errors, training params, objectives, the trainer impl, artifact serde, sampling, leaf refinement, and more. This release decomposes that single file into 24 focused, single-responsibility modules.

### What changed structurally

`crates/engine/src/lib.rs` shrank from **15,189 lines to 101 lines** (99.3% reduction). The 101 remaining lines are entirely module declarations and `pub use` re-exports — no logic, no types, no impls. Twenty-eight new sibling modules and one new `trainer/` submodule directory were added:

| Module | Lines | Holds |
|---|---:|---|
| `crates/engine/src/error.rs` | 34 | `EngineError`, `EngineResult` |
| `crates/engine/src/env.rs` | 78 | `ALLOYGBM_EXPERIMENT_*` env-var consts + parsers |
| `crates/engine/src/tree_node.rs` | 116 | `TREE_NODE_STRIDE` + tree-node ID encode/decode |
| `crates/engine/src/morph_state.rs` | 161 | `MorphState` + `resolve_lr_schedule` + `MorphTreeContext` |
| `crates/engine/src/factor.rs` | 210 | `FactorProjector` + Cholesky + `apply_pre_target_neutralization` |
| `crates/engine/src/split_options.rs` | 84 | `SplitSelectionOptions`, `CategoricalFeatureInfo`, `FactorSplitContext`, `MorphContext`, `LinearContext` |
| `crates/engine/src/traits.rs` | 255 | `BackendOps`, `PerRoundMetricCallback`, `ObjectiveOps` traits |
| `crates/engine/src/objectives/squared.rs` | 144 | `SquaredErrorObjective` |
| `crates/engine/src/objectives/binary.rs` | 153 | `BinaryCrossEntropyObjective` + `sigmoid` |
| `crates/engine/src/objectives/glm.rs` | 356 | `PoissonObjective`, `GammaObjective`, `TweedieObjective` |
| `crates/engine/src/objectives/quantile.rs` | 278 | `QuantileObjective` + `weighted_quantile` + `resolve_boundaries_for_len` |
| `crates/engine/src/objectives/ranking.rs` | 839 | `QueryRMSEObjective`, `PairwiseRankingObjective`, `LambdaMARTObjective`, `XeNDCGObjective`, `YetiRankObjective`, `compute_group_boundaries` |
| `crates/engine/src/objectives/multiclass.rs` | 151 | `MultiClassSoftmaxObjective` |
| `crates/engine/src/multiclass_model.rs` | 378 | `MultiClassTrainedModel`, `MultiClassIterationRunSummary` |
| `crates/engine/src/types.rs` | 469 | `TrainedStump`, `IterationControls`, `IterationDiagnostics`, `IterationStopReason`, `TrainingPolicyMode`, `ArtifactCompatibilityReport`, etc. |
| `crates/engine/src/warm_start.rs` | 45 | `WarmStartState`, `MultiClassWarmStartState` |
| `crates/engine/src/trained_model.rs` | 533 | `TrainedModel` + impl |
| `crates/engine/src/artifact.rs` | 393 | Artifact section encode/decode (`encode_trained_model_payload`, etc.) |
| `crates/engine/src/loss.rs` | 115 | `squared_error_loss`, `binary_crossentropy_loss` |
| `crates/engine/src/sampling.rs` | 327 | `mixed_hash`, `goss_sample_indices`, `select_row_indices_for_round*` |
| `crates/engine/src/tiling.rs` | 70 | Feature-tile helpers |
| `crates/engine/src/round.rs` | 244 | Round application + tree-walk appliers |
| `crates/engine/src/leaf_refinement.rs` | 632 | Newton + empirical-quantile leaf refinement |
| `crates/engine/src/trainer/mod.rs` | 2,821 | `Trainer` struct + impl (the giant 40-method block) |
| `crates/engine/src/trainer/tree_build.rs` | 1,256 | `build_tree_level_wise`, `build_tree_leaf_wise`, `find_best_split_dispatch`, `PendingSplit` |
| `crates/engine/src/trainer/interaction.rs` | — | `InteractionConstraintIndex` + `filter_histogram_bundle_by_features` |
| `crates/engine/src/trainer/policy.rs` | — | `split_selection_options_for_training`, `should_apply_auto_split_l2` |
| `crates/engine/src/trainer/validate.rs` | 301 | `validate_*` fit-contract helpers |
| `crates/engine/src/tests/main.rs` | 4,431 | Engine unit tests (previously inline) |
| `crates/engine/src/tests/morph_state.rs` | 419 | MorphState unit tests (previously inline) |

The pre-existing `dart.rs`, `shared_histogram.rs`, and `joint.rs` modules are unchanged.

### Discipline rules followed (and verified)

This refactor was done as a sequence of 24 small commits, one per logical extraction. At every commit:

- **All 207 engine unit tests passed unchanged.**
- **Full workspace `cargo test --workspace` passed** (445 Rust tests across all crates: engine 207, backend_cpu 79, core 68, predictor 47, ranking-integration 20, shap 14, categorical 10).
- **Full pytest suite passed unchanged**: 641 passed, 16 subtests, identical to v0.11.1 baseline.
- **No public API changes**: every `pub` symbol from v0.11.1 still resolves at its old path. `alloygbm_engine::TrainedModel`, `alloygbm_engine::Trainer`, `alloygbm_engine::WarmStartState`, etc. all remain importable from the crate root via `pub use` re-exports.
- **No behavioral changes**: every moved function body is byte-identical to its v0.11.1 form. Visibility modifiers were promoted (e.g. private `fn` → `pub(crate) fn`) only when strictly required by the new module boundary; never promoted past `pub(crate)`.
- **No dependency changes**: lockfile diff is the version bumps and nothing else.

### Why the version bump

Even though there are zero user-facing changes, this is a v0.12.0 minor bump rather than a v0.11.2 patch because:

1. **Scope.** Twenty-four commits, ~5,000 lines of code physically relocated, twenty-eight new module files in the engine crate. A patch release implies a small targeted fix; this is structural surgery.
2. **Bug risk surface area.** Mechanical refactors at this scale have a non-zero risk of subtle defects — a visibility promotion that accidentally lets test code reach into private impls, a `use crate::*;` glob that masks an unused import, a tree-node ID re-encoding helper that resolves to the wrong constant. The test suite covers most paths but not all. A minor version bump signals to downstream consumers that the engine's internal layout has changed and they should re-run their own integration tests.
3. **CHANGELOG hygiene.** Future patch releases (v0.12.1, v0.12.2) for bugs discovered after merge should be obviously distinguishable from the refactor itself.

### What did NOT change

No new objectives. No new estimator parameters. No new training modes. No artifact format changes. No predictor post-transform changes. No Python API surface changes. No `GBMRegressor` / `GBMClassifier` / `GBMRanker` / `MultiLabelGBMRanker` method additions or removals. No `__getstate__`/`__setstate__` schema changes. Model artifacts written by v0.11.1 load and predict identically under v0.12.0. Model artifacts written by v0.12.0 are byte-identical to what v0.11.1 would have produced from the same training data.

### Known scope limits of this refactor

This release refactors **only** `crates/engine/src/lib.rs`. The remaining large files in the repository (`bindings/python/src/lib.rs`, `crates/engine/src/joint.rs`, `bindings/python/alloygbm/regressor.py`, `crates/core/src/lib.rs`, `crates/backend_cpu/src/lib.rs`, `crates/shap/src/lib.rs`) are untouched and will be tackled in follow-up releases per the plan at `docs/superpowers/plans/2026-05-23-refactor-large-files.md`.

## v0.11.1 (2026-05-23)

Quantile regression objective feature release.

### Quantile regression objective

`GBMRegressor` accepts a new quantile regression objective (`objective="quantile"`) with pinball loss semantics and parameter `quantile_alpha` (default `0.5`, strictly in `(0.0, 1.0)`):

- **Empirical Quantile Leaf Refinement**: At the end of each round, a custom post-growth leaf refinement step (`refine_quantile_leaf_values`) is run to replace Newton-Raphson leaf predictions with the actual empirical quantiles of residuals for all rows in each leaf.
- **Full-dataset refinement**: Under `row_subsample < 1.0`, split-finding runs on the subsampled subset, but leaf refinement uses the entire training set to minimize the estimation variance of the empirical quantile.
- **Proxy Hessian**: Since the pinball loss has a zero second derivative everywhere, a proxy Hessian `h_i = w_i` (sample weight) is used during split-finding.
- **Quickselect optimization**: The unweighted refinement path uses a fast `O(N)` quickselect algorithm (`select_nth_unstable_by`) instead of sorting `O(N log N)`, avoiding performance degradation.
- **Validation**: Gated validation ensures that invalid `quantile_alpha` settings are only rejected when `objective="quantile"` is active, leaving non-quantile models unaffected.

Scope limit: Single-output `GBMRegressor` only. Rejects combinations with DART boosting, MorphBoost, linear leaves (`leaf_model="linear"`), classification, ranking, and joint multi-output training.

## v0.11.0 (2026-05-22)

Two small, independent wins in one release.

### SHAP interaction values (Lundberg Algorithm 2)

`GBMRegressor.shap_interaction_values(X)` returns pairwise SHAP
attributions as a `(n_rows, n_features, n_features)` tensor.  Implements
Lundberg et al. (2020) "From local explanations to global understanding
with explainable AI for trees" Algorithm 2 in polynomial time
`O(T · L · D² · M)` where `M` is the feature count.  The implementation
is a verbatim port of the canonical `slundberg/shap` C++ reference
(`shap/cext/tree_shap.h::tree_shap_recursive`) — including the
`condition_fraction` accumulator approach, the "skip path-extend on
parent_feature_index == conditioning_feature" trick, and the unsigned
underflow that produces an empty leaf scan when conditioning fires at
the very first split of a tree.

Three invariants pinned by tests (within `atol = 1e-5 + rtol = 1e-4 · |predict(x)|`):

- **Symmetric**: `Φ_ij == Φ_ji`.
- **Row-marginal**: `Σ_j Φ_ij == φ_i` (per-feature SHAP).
- **Full additivity**: `Σ_i Σ_j Φ_ij + expected_value == predict(x)`.

The diagonal is filled from the row-marginal invariant; off-diagonals
match the brute-force exact-Shapley enumeration within 5e-3 on
synthetic depth-3 4-feature and depth-5 3-feature models (the latter
with forced feature duplicates on every path).

New Rust crate-level surface:

- `alloygbm_shap::ShapInteractionBatch`
- `alloygbm_shap::explain_interactions_from_artifact_bytes`
- `alloygbm_shap::explain_interactions_from_artifact_bytes_with_binning`

New PyO3 pyfunctions: `shap_explain_interactions`,
`shap_explain_interactions_dense`, and `_with_binning` variants.

Scope limit: constant-leaf artifacts only.  `leaf_model="linear"` is
rejected by the entry point with a clear error.  Multi-output and
multiclass softmax interactions are deferred.

### Poisson / Gamma / Tweedie regression objectives

`GBMRegressor` accepts three new GLM regression objectives with log-link
semantics (`predict()` returns `exp(raw)`):

- `objective="poisson"` — count regression. Targets must be `>= 0`.
  Gradients: `(μ − y) · w`; hessians: `μ · w`; loss: Poisson deviance.
- `objective="gamma"` — strictly-positive continuous regression. Targets
  must be `> 0`. Gradients: `(1 − y/μ) · w`; hessians: `(y/μ) · w`;
  loss: Gamma deviance.
- `objective="tweedie"` — compound Poisson-gamma for
  `1 < variance_power < 2`. Set via new
  `tweedie_variance_power: float = 1.5` constructor kwarg. Targets must
  be `>= 0`. Gradients: `(μ^(2-p) − y · μ^(1-p)) · w`; hessians:
  `μ^(2-p) · w` (LightGBM/XGBoost simplified Newton form).

All three use weighted-mean-in-log-space initial predictions and reuse
the standard `ObjectiveOps` machinery — Newton-Raphson leaves and all
training features (DART, GOSS, leaf-wise, warm-start, MorphBoost,
`neutralization="per_round_gradient"` and `"split_penalty"`) compose
without modification.

Pre-target factor neutralization remains squared-error-only (the
residualize-target == residualize-gradient identity doesn't hold for
log-link objectives).

New deviance metrics in `alloygbm.evaluation`:
`poisson_deviance(y_true, y_pred)`, `gamma_deviance(y_true, y_pred)`,
`tweedie_deviance(y_true, y_pred, variance_power=p)`.

Target-domain validation raises `ValueError` before training starts
when targets violate the objective's domain (negative y for
Poisson/Tweedie, non-positive y for Gamma).

`TrainParams` gains one new public field: `tweedie_variance_power: f32`
(default 1.5).  Only consulted for `objective="tweedie"`; ignored
otherwise.  Predictor post-transform table extended: `exp(raw.clamp(-50, 50))`
for `"poisson"`, `"gamma"`, `"tweedie"` artifacts.

Scope limit: single-output regression only.  Not on `GBMRanker`,
`GBMClassifier`, multiclass softmax, or the joint multi-output ranker.

### Internal helpers (small)

- New `glm_clamp_exp(eta)` and `glm_weighted_target_sum(targets, weights)`
  helpers in `crates/engine/src/lib.rs` shared across the three GLM
  objectives.

## v0.10.6 (2026-05-22)

### Joint trainer: factor neutralization (all three modes)

`MultiLabelGBMRanker(multi_label_mode="joint", neutralization=…,
factor_exposures=…)` now supports all three factor-neutralization modes
with the same surface as the single-output `GBMRegressor` /
`GBMRanker`. Closes the last v0.10.4-deferred follow-up; the joint
trainer reaches full feature parity with the single-output path.

Three modes, all activated via the `neutralization` kwarg:

- **`pre_target`** — residualize each per-output target through the factor
  exposures once before training. Requires every per-output objective to be
  `squared_error` (the only objective where residualize-target equals
  residualize-gradient, the identity pre_target relies on).
- **`per_round_gradient`** — project each of the K gradient buffers in
  place every round after computing them. Mirrors the single-output
  multiclass per-class projection pattern; works for any per-output
  objective.
- **`split_penalty`** — subtract a K-output factor-load penalty from each
  candidate split's gain. Applies under both `tree_growth="level"` and
  `tree_growth="leaf"`; the leaf-wise heap ranks candidates by penalized
  gain.

Three new kwargs admitted by `_JOINT_SUPPORTED_KWARGS`:

- `neutralization` — `"none"` (default), `"pre_target"`,
  `"per_round_gradient"`, or `"split_penalty"`
- `factor_neutralization_lambda` — ridge regularization on the projector
  Gram matrix (default `1e-6`)
- `factor_penalty` — `split_penalty` mode's penalty multiplier (default
  `0.0` — `0` collapses to standard byte-for-byte)

Plus the `factor_exposures=` kwarg on `fit()` (already existed for the
independent-mode fallback; now honored on joint too). The PyO3 bridge
cross-validates the exposures-vs-config invariant: active config requires
exposures, exposures require an active config.

### Artifact

New `ModelSectionKind::NeutralizationMetadata` (kind=14) records the
active config in the artifact so joint models are self-describing.
Metadata only; prediction never reads it (neutralization is a
training-time transformation on targets/gradients/split-gains; the
trained leaf values already bake in the projection).

### Byte-equivalence

A fit with `neutralization='none'` (or `kind=None`, or `split_penalty=0`)
produces byte-identical artifact bytes to a pre-v0.10.6 fit. Pinned by
`joint_neutralization_inert_configs_match_v0_10_5_byte_for_byte` in
`crates/engine/src/joint.rs`. Composes with MorphBoost
(`training_mode="morph"`), DRO leaves (`leaf_solver="dro"`),
DART boosting, and warm-start.

## v0.10.5 (2026-05-22)

### Joint trainer: DRO leaves

`MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro", dro_radius=…, dro_metric="wasserstein")`
now applies Wasserstein-distributionally-robust leaf values on the joint
multi-output trainer, mirroring `GBMRegressor` / `GBMRanker`'s single-output
leaf solver. Leaf-only: split-gain dispatch still uses the standard
K-output sum-of-XGBoost-gains (multi-output histogram doesn't carry per-bin
`grad_sq` and adding it would cost ~1.5× joint-round memory — split-time
DRO is deferred pending benchmark evidence).

Three new kwargs allowed in `_JOINT_SUPPORTED_KWARGS`:
- `leaf_solver` — `"standard"` (default) or `"dro"`
- `dro_radius` — float ≥ 0; `0.0` collapses to standard byte-for-byte
- `dro_metric` — `"wasserstein"` (only supported value in v0.10.5)

Works under both `tree_growth="level"` and `tree_growth="leaf"`, and
composes with MorphBoost (`training_mode="morph"`) and DART/GOSS
boosting modes. Factor neutralization on the joint trainer remains
deferred to **v0.10.6**.

## 0.10.4

Adds MorphBoost (Kriuk 2025, arXiv:2511.13234) to the joint multi-output
trainer used by `MultiLabelGBMRanker(multi_label_mode="joint")`. This is
the first of three deferred items from `docs/limitations.md` Limitation 2
to ship; DRO leaves and factor neutralization on the joint trainer land
in v0.10.5 and v0.10.6 respectively. Default behaviour for every existing
user-facing API remains byte-identical to v0.10.3 when MorphBoost is not
opted into (the engine skips morph plumbing entirely when
`params.morph_config.is_none()`).

### Added — MorphBoost on the joint trainer

- `MultiLabelGBMRanker(multi_label_mode="joint", training_mode="morph", …)`
  now activates MorphBoost on the shared-tree multi-output trainer.
  Honors the full single-output MorphBoost surface:
  `morph_rate`, `evolution_pressure`, `morph_warmup_iters`,
  `info_score_weight`, `depth_penalty_base`, `balance_penalty`,
  `lr_schedule`, `lr_warmup_frac`. Per-iteration LR schedule (constant
  or warmup-cosine), per-leaf depth penalty
  (`depth_penalty_base ^ (depth/3)` where
  `depth = (local_node_id + 1).ilog2()`), and per-iteration leaf
  shrinkage (`1 − morph_rate * round/total`) all apply uniformly across
  the K-output leaf values.

- Multi-output split-gain dispatch: two new helpers in
  `crates/engine/src/shared_histogram.rs` —
  `compute_multi_output_split_gain_morph` and
  `find_best_multi_output_categorical_split_morph` — sum per-output
  morph gain across K outputs. Each output uses its own
  `(grad_mean, grad_std)` snapshot from `MorphState::ema_stats[k]`. The
  morph formula (`crates/backend_cpu/src/morph.rs::compute_morph_gain`)
  is inlined per-output rather than depended on through the backend
  crate (engine cannot depend on backend-cpu).

- Per-side row count for the info-gain term is approximated via
  `hess.max(0.0) as u32` (multi-output histogram doesn't carry exact
  counts). Exact for objectives where hessian ≡ 1 per row
  (`squared_error`, `queryrmse`) and a monotone proxy for ranking. The
  dominant post-warmup signal is the gradient-gain term (weighted by
  `1 - info_score_weight`) which uses `(g, h)` directly. Threading
  exact per-bin counts would require a 1.5× expansion of
  `MultiOutputHistogram` and is deferred. Warmup byte-equivalence with
  the standard K-output gain is guaranteed regardless.

- MorphBoost EMA persists through the artifact's `MorphMetadata`
  section. `JointWarmStartState.initial_ema_stats: Option<Vec<GradientEmaStats>>`
  re-seeds `MorphState::ema_stats` on warm-resume so the gradient-
  statistics smoothing is continuous across the resume boundary — new
  rounds see the same per-output `(mean, std)` they would have seen
  had training never been interrupted. The PyO3 bridge auto-extracts
  the snapshot from `init_artifact_bytes` via
  `TrainedModel::from_artifact_bytes(...).morph_metadata` and threads
  it through.

  **MorphBoost warm-resume is NOT byte-equivalent to a fresh longer fit
  (PR #37 review C3).** Per-iteration leaf shrinkage
  (`1 − morph_rate * round/total`) and LR schedule are resolved against
  the `total_iterations` horizon at training time. A prior fit with
  `n_estimators=6` baked its first six trees against a 6-round horizon;
  resuming with `n_estimators=4` runs the new four rounds against a
  10-round horizon but the prior six trees keep their original
  shrinkage. So a `6+4` warm-resumed MorphBoost fit does not match a
  fresh `n_estimators=10` MorphBoost fit; the prior trees can't be
  retroactively re-scaled. The EMA continuity is the practical
  guarantee; byte-level reproducibility across a horizon change is
  intentionally out of scope. This mirrors the single-output MorphBoost
  warm-start behavior. The regression
  `joint_morph_warm_resume_preserves_ema_continuity_not_byte_equivalence`
  pins both invariants.

### Internal

- New `JointMorphContext` (private to `crates/engine/src/joint.rs`)
  carries the per-round morph snapshot needed by `build_joint_round*`:
  K per-output `(grad_mean, grad_std)` extracted from
  `MorphState::ema_stats`, the precomputed per-iteration constants,
  and the iteration / total horizon. Distinct from the `pub(crate)`
  `crate::MorphTreeContext` which is tied to single-output `MorphState`.

- `build_joint_round` factored into a public no-morph wrapper + a
  private `build_joint_round_inner` that takes
  `Option<&JointMorphContext>` and routes both the numeric threshold
  sweep and the Fisher-sort categorical scan through the morph
  variants when present. `build_joint_round_leafwise` gains the same
  `morph_ctx` parameter.

- `fit_joint_inner` constructs `MorphState::new(cfg, K,
  total_iterations, learning_rate)` when `params.morph_config.is_some()`.
  `total_iterations` covers warm-start prefix + new rounds so the LR
  schedule + leaf-shrink curve see the full horizon, mirroring the
  single-output multiclass path. EMA stats are updated per-round from
  freshly computed gradients BEFORE GOSS / row-subsample / tree-building
  so morph split selection sees the latest snapshot.

- `_JOINT_SUPPORTED_KWARGS` now contains 31 entries — 9 MorphBoost
  kwargs added. New `_build_joint_morph_config` helper in
  `bindings/python/alloygbm/multi_label_ranker.py` reuses the existing
  `alloygbm._morph.build_morph_config_dict` so defaults match
  `GBMRegressor` / `GBMRanker`.

### Deferred to v0.10.5 / v0.10.6

- **Joint DRO leaves** — `leaf_solver="dro"` on the joint trainer.
  Tracked for v0.10.5.

- **Joint factor neutralization** — `neutralization="pre_target" |
  "per_round_gradient" | "split_penalty"` + `factor_exposures=`. The
  PyO3 bridge currently rejects `factor_exposures` unconditionally
  under `multi_label_mode="joint"`. Tracked for v0.10.6.

Both deferred items remain in `docs/limitations.md` Limitation 2 with
explicit version markers.

## 0.10.3

Closes the four "v0.10.3" follow-ups carved out of the v0.10.2 joint-trainer
parity work in `docs/limitations.md` Limitation 2. The
`MultiLabelGBMRanker(multi_label_mode="joint")` wrapper now accepts
native-categorical splits from the Python surface, and the joint trainer
gains `boosting_mode="goss"`, `boosting_mode="dart"`, and `warm_start=True` +
`init_model=...`. Default behaviour for every existing user-facing API
remains byte-identical to v0.10.2 when the new knobs are not opted into.

### Added — joint native-categorical Python wiring

- The Rust-level joint native-categorical path
  (`fit_joint_multi_output_with_categorical` +
  `find_best_multi_output_categorical_split`) was already in v0.10.2; the
  PyO3 bridge now re-bins requested columns to `bin_index == category_id`
  before invoking the trainer, mirroring the single-output path's
  `apply_categorical_encoding_to_training_matrices_multi`. The
  `_JOINT_SUPPORTED_KWARGS` allow-list re-adds
  `categorical_feature_indices` and `max_cat_threshold`.

### Added — joint GOSS row sampling

- New `select_joint_row_indices_for_round` helper inside
  `crates/engine/src/joint.rs` mirrors
  `select_row_indices_for_round_multiclass`: per-row score is
  `s_i = Σₖ |g_{i,k}|` across the K per-output gradient buffers, a single
  row mask is shared across all buffers, and the amplification factor
  mutates every per-output gradient/hessian in lockstep so histograms
  remain unbiased.
  `MultiLabelGBMRanker(multi_label_mode='joint', boosting_mode='goss',
  goss_top_rate=..., goss_other_rate=...)`.

### Added — joint DART boosting

- Dropout / normalize cycle added to `fit_joint_inner` (the renamed
  inner of `fit_joint_multi_output_with_categorical`). One tree per
  round on the joint trainer simplifies bookkeeping vs. multiclass
  DART: `dart_state.tree_weights` has length `rounds_completed` and
  `dart_round_start_offsets[r]` / `dart_round_counts[r]` collapse to a
  flat per-round pair. Reuses `engine::dart::{select_dropouts,
  apply_normalization}` unchanged. The per-round flow:
  subtract dropped trees at -w_old → compute gradients on the
  dropped-out residual → build tree → pre-scale leaves by lr → walk
  new tree at scale 1.0 → rescale new tree from 1.0 → new_w; re-add
  dropped trees at w_old · drop_factor.
- After the round loop, per-tree `tree_weight` is stamped onto every
  stump in that tree so `TrainedModel::to_artifact_bytes` emits the
  existing `DartTreeWeights` section (kind=11) automatically.
- `JointPredictor` extended with `tree_weights: Vec<f32>` parallel to
  `rounds`. `predict_row` multiplies each tree's leaf contribution by
  `tree_w`, collapsing to v0.10.2 behavior when every weight is 1.0.
- Refactored the v0.10.0 in-loop joint tree walk into a shared
  `walk_tree_into_predictions(tree_stumps, ..., sign, scale)` helper,
  used by the new-tree update, DART dropout subtraction, DART
  re-add, and warm-start replay. The helper extracts local node IDs
  via `node_id % TREE_NODE_STRIDE` so it works both pre-encode
  (round-result stumps) and post-encode (stumps already in `all_stumps`).
  `MultiLabelGBMRanker(multi_label_mode='joint', boosting_mode='dart',
  dart_drop_rate=..., dart_max_drop=..., dart_normalize_type=...,
  dart_sample_type=...)`.

### Added — joint warm-start

- New `JointWarmStartState { baselines, stumps,
  initial_rounds_completed, initial_dart_tree_weights }` + new
  `fit_joint_multi_output_with_warm_start` entry point.
  `MultiLabelGBMRanker(multi_label_mode='joint', warm_start=True,
  init_model=<prior_fit>)` cracks open the prior fit's joint artifact
  + baselines + rounds_completed, replays prior stumps' contributions
  onto `predictions`, re-encodes new-round `node_id` starting at
  `initial_rounds_completed`, and (under DART) reconstructs
  `dart_state.tree_weights` from per-stump `tree_weight`.
- All per-round seeds (GOSS, row_subsample, col_subsample,
  `build_joint_round*`, DART `select_dropouts`) mix
  `global_round = round + initial_rounds` so an N+M warm-resumed fit
  produces identical RNG draws to a fresh N+M fit on rounds N..N+M.
- The `MultiLabelGBMRanker` bundle format v2 mode byte already encoded
  everything needed for warm-resume.
- `JointTrainingSummary.rounds_completed` now reports the TOTAL
  (prior + new) count.

### Deferred

- **Joint MorphBoost / DRO / factor neutralization** — tracked for
  v0.10.4. These touch the per-row gradient pipeline more invasively
  than GOSS/DART/warm-start and need their own design pass.

### Internal

- `fit_joint_multi_output_with_categorical` now delegates to a private
  `fit_joint_inner`, matching the single-output engine's
  `fit_iterations*` → inner-impl pattern. Both the cold-start and
  warm-start public entry points route through `fit_joint_inner`.
- Refactored the v0.10.0 in-loop tree walk into
  `walk_tree_into_predictions` (shared by round-end add, DART
  subtract/re-add, warm-start replay).
- PyO3 bridge `train_joint_multi_label_ranker` signature extended with
  six new keyword-only args (`boosting_mode`, `goss_*`, `dart_*`,
  `init_artifact_bytes`, `init_baselines`, `init_rounds_completed`)
  via PyO3's `#[pyo3(signature = ...)]` defaults.

### Fixed in v0.10.3 (PR #36 review)

- **C1 + C3 — joint native-categorical rebinner**: the v0.10.3 rebinner
  initially used `v.round() as i64` for category ID extraction and
  remapped raw values to dense `0..K-1` IDs in the binned matrix.
  `JointPredictor::predict_row` reads the raw feature value via `v as i64`
  (truncation) and treats it directly as the category ID, so two
  invariants must hold for train/predict agreement: (a) values must be
  exact integer-valued floats (round vs truncate disagree on `0.6`);
  (b) values must already be dense zero-based IDs (the dense remapping
  isn't reflected at predict time). The rebinner now uses truncation
  consistently AND validates both invariants up front, raising a clean
  `ValueError` pointing users at sklearn `LabelEncoder` when violated.
  Persisting and applying a per-feature `cat_to_id` mapping in the
  joint artifact path (so arbitrary numeric category IDs work without
  user pre-encoding) is tracked for v0.10.4 alongside
  `categorical_state` plumbing on the joint predictor.
- **C2 — warm-start stump feature_index validation**: the joint
  warm-start replay path (`walk_tree_into_predictions`) indexes
  `binned_matrix.bins[row * feature_count + feature]`. Without
  validation, a prior fit trained on a wider feature set would panic
  the Rust side via out-of-bounds indexing instead of surfacing a
  clean error. `fit_joint_inner` now validates every prior stump's
  `feature_index < feature_count` before replay and returns a
  user-actionable error.
- **C4 — `_fit_joint` init_model schema validation**: defense in depth
  on the Python boundary so we never trigger the Rust panic from C2.
  Validates `init_model._joint_feature_count == feature_count`,
  `init_model.n_labels_ == n_labels`, and rejects DART <-> non-DART
  warm-resume transitions (silently mishandles per-tree weights).

## 0.10.2

Closes the leaf-wise multiclass DART limitation and the first slice of
joint-path feature parity. The joint trainer
(`engine::joint::fit_joint_multi_output`) gains leaf-wise growth + native
categorical splits + interaction constraints + min_split_gain + row/col
subsample. `GBMClassifier(boosting_mode="dart")` with K ≥ 3 classes now
works under `tree_growth="leaf"`. Default behaviour for every existing
user-facing API remains byte-identical to v0.10.1 when the new features
are not opted into.

### Added — joint trainer core feature parity

- `engine::joint::fit_joint_multi_output` honors `tree_growth="leaf"` +
  `max_leaves` via new `build_joint_round_leafwise` (priority-queue
  best-first growth keyed by K-output split gain). Same constraints as
  level-wise (`min_data_in_leaf`, `min_split_gain`, etc.) apply.
- Native-categorical splits on the joint path (Rust-level only). The
  new `find_best_multi_output_categorical_split` in
  `crates/engine/src/shared_histogram.rs` (Fisher-sort over K outputs,
  ordering by output-0 Newton score, ≤ 64 categories per feature) is
  sound when fed bins where `bin_index == category_id`. The new entry
  point `fit_joint_multi_output_with_categorical` accepts a
  `&[CategoricalFeatureInfo]` slice; the original
  `fit_joint_multi_output` remains as a thin wrapper passing an empty
  slice (byte-identical to v0.10.1).
  **Python surface is NOT yet wired in v0.10.2** — the
  `MultiLabelGBMRanker(multi_label_mode="joint")` wrapper rejects
  `categorical_feature_indices` and `max_cat_threshold` because the
  current Python bridge bins all features with
  `ContinuousBinningStrategy::Linear`, which doesn't preserve the
  `bin_index == category_id` invariant the JointPredictor relies on.
  Wiring the proper categorical preparation path through the joint
  bridge is tracked for v0.10.3.
- `interaction_constraints` on the joint trainer — reuses the
  single-output `InteractionConstraintIndex` via `pub(crate)` visibility.
  `HashMap<u32, u64>` tracks per-node active group bitset; descent
  narrows on each split.
- `min_split_gain` honored on the joint trainer (rejects splits below
  the threshold).
- `row_subsample` on the joint trainer — seeded per-round Bernoulli row
  mask via xorshift64*; masked rows get zeroed gradients (LightGBM
  `bagging_fraction` semantics).
- `col_subsample` on the joint trainer — seeded per-round feature mask;
  all-masked edge case falls back to all-allowed (LightGBM
  `feature_fraction` behavior).
- Python `_JOINT_SUPPORTED_KWARGS` expanded to permit
  `min_split_gain`, `row_subsample`, `col_subsample`,
  `interaction_constraints`, `tree_growth`, `max_leaves`.
  `categorical_feature_indices` and `max_cat_threshold` are *not*
  permitted in joint mode in v0.10.2 (see native-categorical bullet
  above); they remain rejected with a clear `NotImplementedError`.

### Fixed in v0.10.2 (post-merge review)

- **Joint `col_subsample`**: the per-round feature mask was seeded by
  `params.seed` alone, so every tree sampled the identical feature
  subset (defeating column sampling's purpose). The seed now also
  mixes in the round index, matching LightGBM `feature_fraction` and
  the single-output trainer's per-round behavior.
- **Joint `row_subsample`**: previously zeroed gradients of unsampled
  rows but kept them in `node.row_indices`, which meant
  `min_data_in_leaf` could be satisfied by rows that didn't actually
  contribute to the histogram. The root node now contains only the
  sampled row subset, so `min_data_in_leaf` operates on the sampled
  set (matching the single-output trainer).
- **Test rename**: `test_multiclass_dart_leaf_wise_validation_early_stopping_works`
  was accidentally defined under the same name as the existing
  `test_multiclass_dart_with_validation_early_stopping`, so Python
  silently kept only the second and the v0.10.2 leaf-wise validation
  case wasn't running. Renamed to its intended name and re-verified
  it exercises the new code path.

### Added — leaf-wise multiclass DART

- `GBMClassifier(boosting_mode="dart")` with K ≥ 3 classes now supports
  `tree_growth="leaf"` + `max_leaves`. The v0.10.1 level-wise
  restriction in `fit_multiclass_iterations_impl` was lifted; the
  per-class `dart_round_start_offsets[k]` / `dart_round_counts[k]`
  bookkeeping was already growth-mode-agnostic (snapshots
  `class_stumps[k].len()` around each `build_tree_*` call). Validation
  early-stopping DART transition and DART warm-start tree-weight
  reconstruction both work without code changes; verified by new
  regression tests in `bindings/python/tests/test_multiclass_dart.py`.

### Deferred

- Joint trainer GOSS / DART / warm-start: v0.10.3
- Joint trainer MorphBoost / DRO / neutralization: v0.10.4

## 0.10.1

Closes the three v0.10.x deferred limitations from v0.10.0: joint
`MultiLabelGBMRanker` Python surface, multiclass softmax + GOSS, and
multiclass softmax + DART (including warm-start). Default behaviour
for every existing user-facing API remains byte-identical to v0.10.0
when the new features are not opted into.

### Added

- **`MultiLabelGBMRanker(multi_label_mode="joint")` Python surface.**
  New PyO3 entry point `train_joint_multi_label_ranker` wraps
  `engine::joint::fit_joint_multi_output` (v0.10.0 infra); new
  `JointPredictorHandle` py-class wraps `engine::joint::JointPredictor`
  for K-output prediction. Default mode is still `"independent"` (the
  K-per-label `GBMRanker` fallback from v0.7.1) — joint is opt-in.
  Bundle format (`.alloy` files written via `save_model`) bumped to
  v2 with an explicit mode byte; v1 bundles still load as independent
  mode. Joint mode in v0.10.1 supports level-wise growth, the
  standard boosting mode, and the built-in `squared_error` /
  `queryrmse` / `rank:pairwise` / `rank:ndcg` / `rank:xendcg`
  objectives. Joint-path feature parity (MorphBoost, neutralization,
  DRO, interaction constraints, leaf-wise, GOSS, DART, warm-start)
  is targeted for later v0.10.x releases.
- **`GBMClassifier(boosting_mode="goss")` for K ≥ 3 classes.** Per-row
  score `s_i = Σₖ |g_{i,k}|` (LightGBM convention) drives a shared
  sampling mask across all K class gradient buffers; the
  amplification factor is applied identically to every class's grad
  and hess. The multiclass round loop was refactored so the K
  gradient buffers are pre-computed before sampling (every class
  channel must be visible to the scorer before deciding which rows
  to keep / amplify).
- **`GBMClassifier(boosting_mode="dart")` for K ≥ 3 classes.** Per-
  class prediction vectors get per-round subtract/readd of dropped
  tree contributions scaled by `dart_state.tree_weights`. Dropout
  flat index `prior_round * K + class_k` resolves to a single stump
  in `class_stumps[class_k][prior_round]`. After K new trees are
  built each round they are rescaled to `new_w = 1/(n_dropped + 1)`,
  the dropped trees are re-added at their rescaled weights, and
  `stump.tree_weight = new_w` is stamped on each new stump. Requires
  `tree_growth="level"` in v0.10.1.
- **Multiclass DART warm-start.**
  `MultiClassWarmStartState.initial_dart_tree_weights` carries the
  flat round-major × class-k per-tree weights from the prior fit so
  continuation seeds `dart_state.tree_weights` correctly. Historical
  RNG-driven dropouts are intentionally not persisted; new rounds
  start fresh dropout bookkeeping (matching the v0.10.0 binary path).

### Changed

- **`MultiLabelGBMRanker.__init__` accepts `multi_label_mode`.**
  Defaults to `"independent"`; `"joint"` opts into the new shared
  tree training. The kwarg name avoids collision with
  `GBMRanker.training_mode` (MorphBoost selector `"manual"` /
  `"morph"`) that would have broken v0.7.1 callers passing
  `training_mode="morph"` through the multi-label wrapper.
- **`JointTrainingSummary`** gains a `rounds_completed: usize` field
  for parity with single-output training summaries.

## 0.10.0

Infrastructure release: lays the Rust-level foundation for joint
multi-output learning and closes the v0.9.0 `DART + warm_start`
follow-up. Default behaviour for every existing user-facing API
(`GBMRegressor`, `GBMClassifier`, `GBMRanker`, `MultiLabelGBMRanker`)
remains byte-identical to v0.9.0 — the new artifact section is only
emitted when the (currently Rust-only) joint trainer produces a model.

### Added

- **DART + `warm_start` continuation.** `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker` now accept the combination
  `boosting_mode="dart"` + `warm_start=True` (or `fit(..., init_model=prior_model)`).
  `WarmStartState` gains an optional `initial_dart_tree_weights`
  field that captures the per-stump `tree_weight` snapshot from the
  prior fit. The engine seeds `dart_state.tree_weights` from this
  snapshot and pre-populates the `round_start_offsets` /
  `dart_round_counts` arrays from the warm-start tree shapes so
  new-round dropouts can correctly subtract/replay prior trees.
  Historical RNG-driven `dropped_per_round` is intentionally not
  persisted; new rounds start fresh dropout bookkeeping going
  forward, matching the natural semantics of resuming an arbitrary
  DART continuation. The v0.9.0 rejection error is removed.
- **K-output shared-histogram primitive** (`crates/engine/src/shared_histogram.rs`).
  `MultiOutputHistogram` accumulates K (grad, hess) pairs per
  (feature, bin) in one sweep. Layout:
  `feature-major → bin-major → output-major → (grad, hess) interleaved`.
  Includes `build_multi_output_histogram_inplace`,
  `subtract_multi_output_histogram` (subtraction trick for K
  outputs), and `compute_multi_output_split_gain` (sums per-output
  Newton/XGBoost gain across K outputs). Foundation for joint
  multi-label boosting (v0.10.1) and multiclass DART/GOSS (v0.10.1+).
- **`MultiOutputLeafValues` artifact section** (kind index 13). New
  optional artifact section storing per-stump K-output leaf values
  (`Vec<f32>` of length `2 * n_outputs` per stump). `TrainedStump`
  gains an optional `multi_output_leaf_values: Option<(Vec<f32>, Vec<f32>)>`
  field carrying the K-vector for each child. Emitted only when a
  joint trainer is used; scalar / linear-leaf / multiclass-softmax
  artifacts remain byte-identical to v0.9.0.
- **Rust-level joint multi-output trainer.**
  `crates/engine/src/joint.rs` exposes:
  - `JointObjective` enum (`squared_error`, `queryrmse`, `rank:pairwise`,
    `rank:ndcg`, `rank:xendcg`) for per-output gradient dispatch.
  - `build_joint_round(params, binned_matrix, grads_per_output, n_outputs)` —
    one shared-tree level-wise round that uses the K-output histogram
    primitive and emits stumps carrying K-output leaf values.
  - `fit_joint_multi_output(params, feature_count, binned_matrix,
    targets_per_output, group_id, per_output_objective, n_estimators)` —
    full training loop returning a `JointTrainingSummary { baselines,
    model, per_output_objective_names }`. Each model serializes
    cleanly through the existing `TrainedModel::to_artifact_bytes`
    pipeline.
  - `JointPredictor` — compact joint-mode predictor that decodes a
    joint artifact and exposes `predict_row` / `predict_batch`
    returning shape `(n_outputs,)` / `(n_rows, n_outputs)`.
  Scope is intentionally minimal for v0.10.0: level-wise growth
  only, no MorphBoost / DRO / neutralization / leaf-wise /
  native-categorical / GOSS / DART / warm-start on the joint path.

### Deferred to v0.10.x

- **Python `MultiLabelGBMRanker(training_mode="joint")` user-facing surface** —
  the Rust infrastructure is complete; the Python wrapper is targeted
  for v0.10.1.
- **Multiclass softmax + DART / GOSS** — engine plumbing into the
  K-output histogram primitive is targeted for v0.10.1+.
- **Leaf-wise / MorphBoost / DRO / neutralization on the joint path** —
  feature parity with the single-output trainer is targeted for v0.10.x.

### Resolved limitations

- v0.9.0 "DART + warm_start not yet supported" → resolved.

## 0.9.0

Minor feature release: closes the v0.8.0 DART placeholder (Limitation 2)
and resolves the carry-forward NaN routing bug on the linear-rank
predict path (Limitation 4). Default behaviour is byte-identical to
v0.8.0 on every API surface — DART artifacts only emit the new
`DartTreeWeights` section when at least one stump has a non-1.0 weight,
which never happens under `boosting_mode="standard"` (the default) or
`boosting_mode="goss"`.

### Added

- **DART boosting mode (Dropouts meet MART).** New `boosting_mode="dart"`
  on `GBMRegressor`, binary `GBMClassifier`, and `GBMRanker`. Four
  parameters expose the LightGBM-style API:
  - `dart_drop_rate` (default `0.1`) — per-tree drop probability per round
  - `dart_max_drop` (default `50`) — cap on the number of trees dropped
    per round
  - `dart_normalize_type` (`"tree"` or `"forest"`, default `"tree"`) —
    rescale policy after the new tree is fit
  - `dart_sample_type` (`"uniform"` or `"weighted"`, default
    `"uniform"`) — uniform sampling or `|tree_weight|`-weighted sampling
  The per-round dropout + normalization cycle lives in a new module
  `crates/engine/src/dart.rs` (no new dependencies — uses the existing
  `mixed_hash` splitmix64 derivative for deterministic per-stump
  decisions). Per-stump `tree_weight: f32` is plumbed through
  `TrainedStump` and persisted via a new `DartTreeWeights` artifact
  section (kind index 12), emitted only when at least one weight
  diverges from 1.0. Multiclass DART and DART + `warm_start` are
  rejected with clear errors — tracked as v0.10.x follow-ups.

### Fixed

- **NaN routing on the linear-rank predict path (Limitation 4
  resolved).** The predict-time quantize helpers
  (`quantize_dense_values_linear_inplace_wide`,
  `quantize_dense_values_linear_rank_inplace_wide`, and the inline
  loop in `predict_dense_quantized_with_summary_bytes`) now preserve
  `f32::NAN` through the f32 cast instead of falling through to bin 0.
  The predictor's existing `feature_value.is_nan() → default_left`
  short-circuit at `crates/predictor/src/lib.rs:148` then fires
  automatically, restoring the learned-missing-direction routing on
  rank-binned columns. Additionally, `LinearLeaf::eval` and
  `LinearLeafCompact::eval` now skip NaN regressor features when
  accumulating the linear sum so PL-leaf predictions don't NaN-poison
  on a `w · NaN` step. Pure-linear, pure-quantile, and linear-rank
  paths now share consistent NaN semantics.

### Known limitations carried forward to v0.10.0

- Multiclass softmax + DART is still rejected — requires per-class
  gradient bookkeeping during the dropout step.
- DART + `warm_start` is rejected — requires persisting
  `tree_weights` and `dropped_per_round` in `WarmStartState`.
- Joint shared-tree multi-label ranking and the K-output
  shared-histogram engine primitive remain v0.10.0 targets.

## 0.8.0

Minor feature release: closes the mixed linear-rank SHAP carry-forward
from v0.7.4 (Limitation 4) and adds LightGBM-style GOSS sampling as a
new opt-in boosting mode.  Default behaviour is byte-identical to v0.7.5
on every API surface.

The other two original v0.8.0 targets — DART boosting mode and joint
shared-tree multi-label ranking — were scope-split out to v0.9.0 and
v0.10.0 respectively to keep this release reviewable.  `BoostingMode::Dart`
is reserved in the API (Python `boosting_mode="dart"` raises
`NotImplementedError`; the Rust trainer rejects it with a clear error
message) so v0.9.0 can land DART training without further `TrainParams`
changes.

### Added

- **GOSS sampling (gradient-based one-side sampling).**  New
  `boosting_mode="goss"` opt-in on `GBMRegressor`, `GBMClassifier`
  (binary), and `GBMRanker`, with `goss_top_rate` (default `0.2`) and
  `goss_other_rate` (default `0.1`) controlling the kept-top fraction
  and sampled-low fraction respectively.  Implements the LightGBM
  algorithm: score rows by `|gradient|`, keep the top `top_rate`,
  sample `other_rate` from the rest, and multiply the sampled-low
  rows' gradient + hessian by `(1 - top_rate) / other_rate` so the
  histogram statistics remain an unbiased estimator of the full-data
  gradient sums.  See `engine::goss_sample_indices` and
  `engine::select_row_indices_for_round`.  Multiclass softmax rejects
  non-Standard boosting modes with a clear error message
  ("...not yet supported for multiclass objectives...") — a v0.8.1
  follow-up will add per-class gradient scoring.
- `BoostingMode` enum and validation in `alloygbm-core` covering
  Standard / Goss / Dart.  DART is a placeholder in v0.8.0: the
  Python `__init__` raises `NotImplementedError("dart is reserved
  for a v0.8.0 follow-up commit")` and the Rust validation layer
  passes through the placeholder so DART implementation work can
  proceed incrementally without further core changes.

### Fixed

- **SHAP strict additivity on the mixed linear-rank binning path
  (Limitation 4).**  When `continuous_binning_strategy="linear"` is
  combined with per-feature *rank-based* linear binning on at least
  one column (`_continuous_feature_linear_rank_flags` has any `True`
  entry), `shap_values()` previously fell back to the legacy
  quantize-then-walk SHAP path, which exempts `leaf_model="linear"`
  artifacts from strict additivity via the
  `binning.is_none() && model_has_linear_leaves(model)` rule in
  `crates/shap/src/lib.rs::verify_additivity`.
  v0.8.0 adds `BinningContext::LinearRank`, a new variant carrying
  per-feature sorted unique values, global `feature_mins` /
  `feature_maxs`, and `max_data_bin`.  At the
  `explain_rows_from_model` entry point SHAP internally quantizes
  raw rows to bin indices using exactly the same rules as
  `predict_dense_quantized_linear_rank`
  (linear-quantize unflagged features, rank-quantize flagged
  features, both following `round_half_away_from_zero` clamped to
  `[0, max_data_bin]`) and dispatches the remainder of the
  path-walker with `BinningContext::PreBinned` semantics.  Both
  tree traversal and PL-leaf evaluation now share the bin-index
  space the predictor evaluates in, so strict additivity holds for
  `leaf_model="linear"` (and constant leaves stay correct).
  The Python `_shap_binning_kwargs()` helper returns
  `{"binning_kind": "linear_rank", "feature_mins": …,
  "feature_maxs": …, "max_data_bin": …,
  "linear_rank_per_feature": …}` whenever any per-feature rank
  flag fires; `GBMClassifier` and `GBMRanker` inherit the fix via
  the shared `GBMRegressor._shap_binning_kwargs` implementation.

## 0.7.5

Bug-fix release.  Closes Limitation 5 from v0.7.4 — the pre-existing
TreeSHAP polynomial-path additivity drift on trees with a feature
appearing more than once on a root-to-leaf path.  No user-visible API
breakage.

### Bug fixes

- **TreeSHAP polynomial-path strict additivity (Limitation 5).**  The
  Rust port of the TreeSHAP polynomial algorithm in
  `crates/shap/src/lib.rs::ts_unextend_path` was shifting the entire
  `PathElement` struct (including `pweight`) to fill the gap left by
  removing a duplicate feature from the path.  This clobbered the
  pweights that the unwind loop had just carefully recomputed in
  place — the reference implementation in `slundberg/shap`
  (`shap/explainers/pytree.py`) stores the four path fields as four
  parallel arrays and only shifts the first three
  (`feature_index`, `zero_fraction`, `one_fraction`), preserving
  pweights.  The fix shifts those three fields explicitly and leaves
  `pweight` alone.  Strict additivity now holds end-to-end on the
  polynomial path for the broad cases that were previously failing.
  Hands-on validation: a synthetic full-tree sweep
  (`tree_shap_polynomial_path_matches_brute_force_on_full_trees`)
  covers depths 2-7 × n_features {2,3,5,8,12} including all
  configurations that force path-duplicate features, asserting
  polynomial matches brute-force per-feature within 1e-5.  End-to-end
  Python coverage: the formerly `@xfail(strict=True)` regression
  `test_strict_additivity_via_tree_shap_polynomial_path` in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` now
  passes as a regular test.  Pre-existing in v0.7.3 and earlier;
  uncovered during v0.7.4 PR #27 review and pinned with an xfail at
  that time pending the v0.7.x follow-up that this release ships.

### Documentation

- `docs/limitations.md`: Limitation 5 promoted to Resolved.
- Other documented v0.7.x follow-ups (mixed linear-rank SHAP path,
  GOSS+DART, joint multi-label ranking, shared-histogram engine)
  remain deferred to v0.8.0.

## 0.7.4

Bug-fix release.  Closes the remaining v0.7.x carryover documented in
`docs/limitations.md` for SHAP strict additivity on
`leaf_model="linear"` artifacts.  No user-visible API breakage.

### Bug fixes

- **SHAP strict additivity for PL leaves.**  Pre-v0.7.4,
  `distribute_linear_terms_for_row` credited the per-feature deviation
  `Σⱼ wⱼ·(xⱼ − μⱼ)` only at the **terminal** leaf of each tree.  The
  predictor accumulates `leaf.eval_row(row)` at **every visited node**
  along the row's path, so SHAP was uncrediting one
  `Σⱼ wⱼ·(xⱼ − μⱼ)` per internal node per tree per row.  On a typical
  `n_estimators=100, max_depth=6` model this produced additivity gaps
  on the order of the predictions themselves (~4 units on
  predictions of magnitude ~10).  The fix walks the full path and
  credits the linear deviation for every visited leaf; the brute-force
  Shapley and TreeSHAP polynomial paths share the same helper so both
  get the fix.  The `model_has_linear_leaves` exemption in
  `verify_additivity` is now gated on `binning.is_none()` so the
  predictor-aligned `BinningContext` callers — i.e. the default Python
  path for continuous features — get the strict tolerance check
  (`atol + rtol·|predicted|`).
  Coverage: 44 new regression tests in
  `bindings/python/tests/test_shap_pl_strict_additivity.py` exercising
  every binning strategy × max-bin width × lambda × max-depth ×
  n-estimators combination plus `training_mode="manual"` and
  `"morph"`, `interaction_constraints`, `GBMRanker`, `GBMClassifier`
  (via the internal Rust check, since the raw margin isn't exposed in
  Python), `feature_importances()` (brute-force exact path), and
  mixed scalar+linear-leaf artifacts.  Strict additivity holds on the
  default predictor-aligned binning path for any model that dispatches
  to the brute-force exact Shapley path
  (`distinct_split_feature_count ≤ MAX_EXACT_SPLIT_FEATURES = 25`).
  Larger models that trigger the polynomial-TreeSHAP path are subject
  to a pre-existing additivity drift documented as Limitation 5 (also
  present in v0.7.3 and earlier).

### Documentation

- New Limitation 4 (`docs/limitations.md`): SHAP on the mixed
  linear-rank binning path — narrow edge case where
  `continuous_binning_strategy="linear"` combined with per-feature
  rank-based binning falls back to the legacy non-binning SHAP entry
  point, triggering the `leaf_model="linear"` exemption.  Deferred to
  v0.8.0.
- New Limitation 5 (`docs/limitations.md`): pre-existing TreeSHAP
  polynomial-path additivity drift on large gradient-trained trees
  (≥ 30 distinct split features, depth ≥ 6).  Uncovered during PR #27
  review; investigated but not isolated in minimal Rust reproductions.
  Coverage pinned by an `@xfail(strict=True)` regression test
  (`test_strict_additivity_via_tree_shap_polynomial_path`) so the
  eventual fix will flip the xfail to a regular pass.

### Documented for v0.7.x follow-ups (deferred to 0.8.0)

- Joint shared-tree multi-label ranking.  The current
  `MultiLabelGBMRanker` trains K independent per-label rankers under a
  unified API; this is numerically equivalent to training each label
  separately.  Joint shared-tree training (where a single ensemble
  updates all label predictions simultaneously) lands alongside the
  v0.8.0 shared-histogram speedup, where the architectural change has
  a real performance story.

## 0.7.3

Bug-fix release.  Closes the four limitations queued in v0.7.2 and
clears RUSTSEC-2025-0020.  No user-visible API breakage.

### Bug fixes

- **SHAP additivity tolerance.**  The check now uses
  `atol + rtol * |predicted|` (numpy `allclose` convention with
  `atol=1e-5`, `rtol=1e-4`) instead of a fixed `1e-5` absolute bound.
  Accumulated f32 round-off across larger explanation batches —
  `feature_importances()` over ~1000 rows of California Housing with
  `n_estimators=200` was the public-facing reproducer — no longer
  raises spurious "additivity check failed" `RuntimeError`s on healthy
  `leaf_model="constant"` artifacts.
- **SHAP path-walker uses predictor-aligned float thresholds.**
  Introduces `shap::BinningContext` (`Linear`, `Quantile`, `PreBinned`)
  and four new PyO3 entry points (`shap_explain_rows_with_binning`,
  `shap_global_importance_with_binning`, plus the `_dense` variants).
  When a binning context is provided, the path walkers compare
  `feature_value < float_threshold` with strict less-than (matching
  `convert_bin_thresholds_to_float*` in the predictor), eliminating
  the path-walk vs. predict-path divergence on continuous features.
  The Python regressor / classifier / ranker estimators now pass
  feature mins / maxs / cuts / binning kind into SHAP automatically,
  so `model.shap_values()` and `model.feature_importances()` Just Work
  for constant-leaf artifacts on continuous data.  The linear-leaf
  `model_has_linear_leaves` exemption stays in place for
  `binning=None` callers and is now only triggered for that legacy
  path.
- **MorphBoost warm-start now persists EMA.**  The MorphMetadata
  artifact section is bumped from v1 to v2 with an appended
  `Vec<GradientEmaStats>` (one entry per class).  `WarmStartState`
  and `MultiClassWarmStartState` gain an
  `initial_ema_stats: Option<Vec<GradientEmaStats>>` field; both
  single-class and multiclass training loops seed the fresh
  `MorphState.ema_stats` from this snapshot when warm-starting.
  Resuming a MorphBoost-trained model with `init_model=` now produces
  numerically meaningful continuations rather than starting the EMA
  cold.  v1 artifacts (pre-v0.7.3) decode with `ema_stats: Vec::new()`
  and the PyO3 bridge translates that into `initial_ema_stats: None`,
  preserving prior cold-EMA behaviour for legacy artifacts.
- **PyO3 0.23 → 0.24 (clears RUSTSEC-2025-0020).**  Bumps
  `pyo3 = "0.24"` and `numpy = "0.24"` in `bindings/python/Cargo.toml`.
  The bindings were already on the `Bound<>`-first API — zero
  `&PyAny`, zero `IntoPy`, no source changes needed.  The ignore
  entries in `deny.toml` and the cargo-audit CI step are removed; the
  advisory list is now intentionally empty.

### Documented for the next release (v0.7.4)

- **SHAP additivity for piecewise-linear leaves on continuous
  features.**  The bin-index vs. float-threshold mismatch in the path
  walker is fixed in v0.7.3, but linear-leaf weights and the
  `feature_baseline` are still trained in bin space, so SHAP's
  decomposition of `wⱼ · (xⱼ − μⱼ)` for `leaf_model="linear"`
  artifacts can still drift on continuous features.  Until a follow-up
  release switches PL weight training to raw feature space, the
  linear-leaf additivity check stays exempted.
- **Joint shared-tree multi-label boosting.**  `MultiLabelGBMRanker`
  still trains K independent per-label rankers.  A K-tree-per-round
  shared-ensemble objective for ranking is the remaining v0.7.x
  follow-up.

## 0.7.2

Documentation, supply-chain, and repo-hygiene release.  No user-facing
Python API surface changes.

### Documentation

- **Docs re-alignment with the v0.7.1 surface.**  Multiple docs still
  said "warm-start is rejected", "SHAP requires `leaf_model='constant'`",
  "no interaction constraints", or "single-label only" after v0.7.1
  shipped those features.  README, `docs/user/*.md`, the Sphinx mirror
  under `docs/site/source/*.rst`, `docs/roadmap/current.md`,
  `CLAUDE.md`, `AGENTS.md`, and `benchmarks/README.md` are now
  consistent with the actually-shipped v0.7.1 API.
- **Release guide & checklist.**  `docs/reference/release_checklist.md`
  is now a top-to-bottom operating manual: the authoritative inventory
  of every file that needs a version bump or content update, the stale-
  content `git grep` queries, the local + CI verification matrix, the
  tag/publish commands, and post-release bookkeeping.
- **API reference.**  `docs/site/source/api.rst` now auto-documents
  `MultiLabelGBMRanker` (missing in v0.7.1).
- **Runnable examples.**  New `examples/` directory with 8
  self-contained end-to-end scripts covering every public estimator,
  factor-neutral boosting, interaction constraints, warm-start
  continuation, and SHAP explanations.

### Repo hygiene & supply chain

- **CI now runs the full pytest suite.**  v0.7.1's `python-smoke` CI
  job built a wheel and ran 7 hand-written smoke snippets, but never
  invoked `pytest bindings/python/tests/` — meaning the 455-test
  Python suite was not enforced on merge.  It is now.
- **Cargo.lock is tracked.**  Reproducible builds between CI / local
  dev / release builds; standard guidance for workspaces that produce
  binaries (the Python extension cdylib).
- **`maturin` pinned in publish.yml** to the `>=1.7,<2.0` range
  declared in `pyproject.toml`'s build-system.requires, so a maturin
  major bump can't silently break a release.
- **`cargo-audit` + `cargo-deny` weekly + on every PR that touches
  Cargo manifests** via `.github/workflows/security-audit.yml`.
  Configured via the new `deny.toml` (narrow license allowlist,
  banned/duplicate dep checks, `crates.io`-only source restriction).
- **Coverage reporting** via `.github/workflows/coverage.yml`
  (`cargo-llvm-cov` + `pytest-cov`, both uploaded to Codecov).
- **`publish = false` on every workspace crate.**  None of them ship
  to crates.io; this prevents an accidental `cargo publish` from
  fragmenting the release surface and clears `cargo-deny`'s wildcard-
  dep warnings on workspace-internal `path` deps.
- **Repo metadata.**  New `CONTRIBUTING.md`, `SECURITY.md`,
  `.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.yml`,
  `.github/PULL_REQUEST_TEMPLATE.md`, `.github/CODEOWNERS`,
  `.github/dependabot.yml`, `.editorconfig`, `requirements-dev.txt`.
- **README badges:** CI status, PyPI version, Python versions, RTD
  docs, MIT license, Rust 1.92+.

### Documented for the next release (v0.7.3)

These are the limitations carried over from v0.7.1 plus two new ones
surfaced during this release's work:

- **SHAP path-walker still compares feature values against bin-index
  thresholds**; strict additivity is relaxed for PL-leaf artifacts.
  Path-walk alignment with the predictor's float thresholds.
- **MorphBoost warm-start does not restore the EMA snapshot** from the
  artifact, so resumed training starts EMA cold.
- **`MultiLabelGBMRanker` trains K independent per-label rankers.**
  Joint shared-tree multi-label boosting (one ensemble, K-tree-per-round
  ranking objective).
- **SHAP additivity tolerance is f32-tight (NEW).**  `shap_values()` and
  `feature_importances()` enforce
  `|predict(x) - (sum(shap) + expected_value)| <= 1e-5` per row.
  Accumulated f32 round-off across larger evaluation samples (e.g.
  `feature_importances()` over ~1000 rows of California Housing with
  `n_estimators=200`) exceeds it by a few ulps even on healthy
  `leaf_model="constant"` artifacts.  The arithmetic is correct; the
  tolerance is the issue.  Loosening to a relative-plus-absolute bound
  (`atol + rtol * |predict(x)|`) is queued.  Workaround: call
  `feature_importances()` on a representative subsample (≤500 rows).
- **PyO3 0.23 pinned — known advisory (NEW).**
  RUSTSEC-2025-0020 documents a buffer-overflow risk in
  `PyString::from_object` for `pyo3 < 0.24.1`.  AlloyGBM does not call
  `PyString::from_object`, so the advisory is not exploitable through
  the public Python API.  The upgrade to `pyo3 0.24+` (and matching
  `numpy` crate version) requires migrating the ~5,300-line bindings
  to the `Bound<>`-first API.  Ignored in `deny.toml` until the
  migration lands.

## 0.7.1

### New Features

- **SHAP for piecewise-linear leaves.**  `shap_values()` now accepts
  `leaf_model="linear"` artifacts and returns an interventional
  decomposition: the path-based TreeSHAP / brute-force machinery
  attributes each leaf's "constant part" (`intercept + Σ wⱼ·μⱼ_global`)
  while per-leaf row deviations `wⱼ·(xⱼ − μⱼ_global)` are credited
  directly to each regressor.  Global feature means are persisted in a
  new `FeatureBaseline` artifact section so SHAP is self-contained at
  explain time.
- **Per-round training diagnostics.**  Every estimator now exposes
  `diagnostics_per_round_` — a list of dicts containing
  `gradient_l2_norm`, `gradient_variance`, `hessian_l2_norm`, sampling
  counts, and (when factor neutralization is active) the
  `neutralization_effectiveness` score `1 − ‖projₘ‖ / ‖origₘ‖` clamped
  to `[0, 1]`.
- **Neutralized warm-start.**  `init_model` / `warm_start=True` with
  `neutralization=*` is now supported across `pre_target`,
  `per_round_gradient`, and `split_penalty` provided the caller supplies
  the same `factor_exposures` matrix used for the initial fit.  Mode,
  `factor_neutralization_lambda`, and (for `split_penalty`)
  `factor_penalty` must match the persisted contract; mismatches raise
  a clear "does not match" error.
- **Interaction constraints.**  LightGBM-compatible
  `interaction_constraints=[[…]]` on every estimator.  Each group is a
  set of feature indices; any root-to-leaf path is restricted to splits
  on features from a single still-active group.  Up to 64 groups per
  fit; enforced through both the level-wise and leaf-wise tree
  builders.
- **`MultiLabelGBMRanker`.**  Unified multi-output ranking estimator
  with `y` shaped `(n_rows, n_labels)` and `predict` returning the same
  shape.  Trains one independent `GBMRanker` per label sharing `group`
  / `factor_exposures` / kwargs, supports per-label
  `ranking_objective` lists, and slices `eval_set` y-columns per label
  so early stopping and custom eval metrics work end-to-end.

### Limitations Documented For The Next Release

- SHAP path-walker still compares feature values against bin-index
  thresholds; strict additivity is relaxed for PL-leaf artifacts.
  Path-walk alignment with the predictor's float thresholds is queued
  for v0.7.2.
- MorphBoost warm-start does not restore the EMA snapshot from the
  artifact, so resumed training starts EMA cold.  Persisting EMA is
  queued for v0.7.2.
- `MultiLabelGBMRanker` trains K independent per-label rankers.  Joint
  shared-tree multi-label boosting (one ensemble, K-tree-per-round
  ranking objective) is queued for v0.7.2.

## 0.7.0

### New Features

- **Factor-neutral boosting** via `neutralization` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`, with row-aligned fit-time
  `factor_exposures`. Supported modes are `none`, `pre_target`,
  `per_round_gradient`, and `split_penalty`.
- `neutralization="per_round_gradient"` projects each boosting round's
  objective gradients away from user-supplied factors. `split_penalty` also
  subtracts a factor-load penalty from split gain via `factor_penalty`.
- `factor_neutralization_lambda` controls the ridge term added to the factor
  Gram matrix used by target or gradient projection.

### Compatibility And Limitations

- `pre_target` is supported for `GBMRegressor` only and is rejected for
  classification and ranking.
- `per_round_gradient` supports `GBMRegressor`, `GBMClassifier`, and
  `GBMRanker`; multiclass classification projects each class-gradient column
  independently.
- `split_penalty` supports constant leaves and rejects
  `leaf_model="linear"`. It is compatible with `leaf_solver="dro"` and
  `training_mode="morph"`.
- `pre_target` rejects `eval_set` in this release because the public API does
  not yet accept validation-set factor exposures for consistent validation
  target residualization.
- `split_penalty` performs additional factor-exposure work during split search;
  benchmark it on production-sized data before assuming standard training
  throughput.
- Estimator `repr(...)` output now includes `neutralization`,
  `factor_neutralization_lambda`, and `factor_penalty`.
- This is training-time factor/gradient neutralization and split exposure
  regularization. It does not guarantee live-market or prediction-time zero
  exposure unless predictions are neutralized against evaluation-time factors
  outside the model.

### Benchmarks

- Added `alloygbm_factor_neutral` and `alloygbm_factor_neutral_dro` arms to
  `benchmarks/run_model_comparison.py`. For these arms, benchmark datasets
  without explicit factors synthesize `factor_exposures` from the first
  `min(5, n_features)` feature columns. These arms are smoke and stability
  checks, not standalone quality claims, because the synthesized factors are
  also present as model features.

## 0.6.0

### New Features

- **DRO-style scalar leaf solver** via `leaf_solver="dro"` on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`. The v0.6.0 implementation is a fast,
  closed-form robust Newton update over within-leaf gradient uncertainty:
  `dro_radius` scales a dispersion penalty before the usual L1 soft-threshold
  and L2 Hessian denominator. `dro_metric="wasserstein"` is the only accepted
  value and denotes this Wasserstein-inspired gradient-uncertainty counterpart,
  not a full optimizer over raw feature/target distributions.
- **Zero-radius parity**: `leaf_solver="dro", dro_radius=0.0` preserves standard
  constant-leaf predictions while writing optional DRO metadata to artifacts.
- **MorphBoost composition**: `leaf_solver="dro"` composes with
  `training_mode="morph"`; robust gradient gain is computed first, then MorphBoost
  blends in its information score and leaf scaling.

### Compatibility And Limitations

- Default `leaf_solver="standard"` preserves existing behavior.
- `leaf_solver="dro"` requires `leaf_model="constant"` in v0.6.0. PL trees
  (`leaf_model="linear"`) continue to use the standard linear-leaf solver.
- Inference speed is unchanged for DRO constant-leaf models because the robust
  leaf values are baked into the model artifact.

### Benchmarks

- Added an `alloygbm_dro` arm to `benchmarks/run_model_comparison.py`.
- Benchmark reports now include a temporal/panel stability section with mean,
  worst, and standard deviation of a task-normalized score across repeated runs.

## 0.5.1

### Performance

- **PL trees ~3× faster on Apple Silicon (NEON), expected ~5× on x86_64 with AVX2.** Inner-loop matrix histogram accumulation is now SIMD-vectorised via `wide::f32x8`. The per-row cost of a linear-leaf tree was ~30-44× slower than constant leaves in 0.5.0; 0.5.1 brings the ratio down to ~10-13×.
- Concrete numbers (regression, n_features=8, max_depth=6, manual policy, Apple M-series; before/after on the same hardware):

  | Scenario | 0.5.0 linear | 0.5.1 linear (SIMD) | Speedup |
  | --- | ---: | ---: | ---: |
  | n=20K, n_est=200 | 6.84 s | 2.31 s | **2.96×** |
  | n=50K, n_est=200 | 16.02 s | 4.99 s | **3.21×** |
  | n=100K, n_est=200 | 31.17 s | 9.49 s | **3.28×** |
  | n=50K, n_est=500 | 40.07 s | 12.40 s | **3.23×** |

### Internal

- `LinearHistogramBin.xt_hx` layout changed from a 36-entry compacted upper-triangle to a 64-entry stride-8 row-major (`xt_hx[j * 8 + k]`). The lower-triangle slots stay zero in the current scalar code paths and are populated by mirror values in the SIMD outer-product path; both representations are mathematically equivalent under the closed-form ridge solve, which only reads the upper triangle. `MAX_PL_MATRIX_ENTRIES` is now `MAX_PL_REGRESSORS * MAX_PL_REGRESSORS = 64`.
- `pl_matrix_index(j, k)` simplified to `j * MAX_PL_REGRESSORS + k`.
- New SIMD helpers in `crates/backend_cpu/src/pl.rs` (`add_xt_hx`, `sub_xt_hx`, `diff_xt_hx`, `copy_xt_hx`, `add_xtg`, `sub_xtg`, `diff_xtg`) replace the previous scalar versions throughout the bin scan and leaf solve. All are bit-exact with their scalar counterparts (lane-independent ops).
- `subtract_linear_histogram_bundle` operates on all 64 matrix entries instead of upper-triangle only — required so the histogram subtraction trick stays correct under the SIMD outer-product write pattern.
- 13 new property tests (7 SIMD-vs-scalar helpers, 1 layout-uniqueness invariant, 1 SIMD-vs-scalar-reference equivalence test on 1000 rows × 5 split features × d=6).

### Compatibility

- **Backward compatible.** v0.5.0 PL-trees model artifacts (`leaf_model="linear"`) load and predict identically in 0.5.1 — the layout change is internal to histogram construction at training time only; the `LinearLeafCoefficients` artifact section format is unchanged. Constant-leaf artifacts are unaffected.

## 0.5.0

### New Features

- **Piecewise-linear (PL) tree leaves** via `leaf_model="linear"` on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`. Each leaf stores a small linear model `f_s(x) = b_s + Σ α_j x_j` (up to 8 regressors per leaf, inherited from the split path's feature indices). Optimal weights are solved in closed form: `α* = -(XᵀHX + λI)⁻¹ Xᵀg`, regularised by `lambda_l2`. Default `leaf_model="constant"` preserves all prior behaviour exactly.
- **`LinearHistogramBundle`** -- parallel histogram structure accumulating `xtg` vector and `xtHx` matrix statistics alongside standard grad/hess bins. Standard SIMD histogram path is untouched.
- **`GainStrategy::Linear`** -- new dispatch arm in the split-gain criterion; closed-form PL gain computed via an ≤8×8 Cholesky solve in the new `crates/backend_cpu/src/pl.rs` module.
- **`LeafValue` enum** (`Scalar(f32)` | `Linear(LinearLeaf)`) replaces the plain `f32` leaf fields on `TrainedStump`. In-memory prediction during boosting evaluates the leaf's linear model at each row's feature values.
- **New artifact section** `ModelSectionKind::LinearLeafCoefficients` stores per-stump linear leaf data. Backward-compatible with v0.4.0 artifacts: a per-stump flag bit indicates linear leaves; older readers continue to work for constant-leaf models.
- **Categorical-native split interaction**: native-bitset categorical splits (`max_cat_threshold > 0`) continue to use constant leaves for that split node; descendant leaves below such a split use linear leaves on all remaining numeric regressors.

### Performance

- Benchmarks show **~10× faster convergence** on linearly-structured datasets (fewer rounds to reach the same RMSE) and **+3.5% RMSE improvement** on California Housing vs constant leaves.
- +1.75pp accuracy improvement on Breast Cancer classification with `leaf_model="linear"`.
- 2–8× training time overhead (Cholesky solve per node); recommended with `lambda_l2=0.01` for weight stability.

### Benchmarks

- **`alloygbm_linear` arm** added to `benchmarks/run_model_comparison.py` for all four task types.
- **`benchmarks/pl_trees_benchmark.py`** -- convergence-curve and λ-sweep analysis across regression, classification, and ranking scenarios.
- Benchmark report committed to `docs/benchmarks/pl_trees_v1.md`.

## 0.4.0

### New Features

- **MorphBoost adaptive training mode** -- opt-in via `training_mode="morph"` on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`. Augments the standard gradient gain with a normalized information-theoretic term (with `tanh(iter/20)` warmup ramp), per-class EMA-driven gain shaping, depth-based leaf penalty, per-iteration leaf shrinkage, and an optional balance penalty. Implementation follows the formulation in [Kriuk (2025), *MorphBoost*](https://arxiv.org/pdf/2511.13234) with deliberate corrections vs the paper's reference code.
- **Per-iteration learning-rate schedules** -- new `lr_schedule` parameter (`"constant"` default or `"warmup_cosine"`), independent of `training_mode`. The `warmup_cosine` schedule does linear warmup over `lr_warmup_frac * n_estimators` rounds then half-cosine decay to a `0.01 * learning_rate` floor.
- **Schedule-aware auto early-stopping** -- when an LR schedule is active, the auto-tuned `min_loss_improvement` threshold is scaled by `current_lr / max_lr`, and warmup-phase rounds (empty trees, slightly-negative loss improvements) do not terminate training. Outside warmup, behaviour is bit-identical to the previous policy.
- **MorphBoost configuration in artifacts** -- the `MorphConfig` and `final_iteration` are persisted as an optional artifact section so loaded models predict consistently.

### Performance

- **SIMD-accelerated kernels** via the `wide` crate (safe API; AVX2 / NEON intrinsics under the hood, scalar fallback otherwise). Standard-path histogram bin-scan and `GradientEmaStats` mean+variance pass are vectorized.
- **Tile-size auto-tuning** for histogram parallelism. The hard-coded `MAX_TILE_FEATURE_WIDTH=64` is replaced by a thread-count-aware helper that targets ~2 tiles per thread, clamped to `[16, 64]`. Improves utilization at high feature counts (Numerai 780-feature `medium` set, etc.).
- **Hoisted morph per-round constants** -- `tanh(iter/20)`, gradient/info-score blend coefficients, and warmup-branch flags are precomputed once per round into `MorphPrecomputed` rather than evaluated per bin candidate.

### Benchmarks

- **`alloygbm_morph` and `alloygbm_morph_cosine` arms** added to `benchmarks/run_model_comparison.py` for all four task types. New `--models` flag filters which arms run.
- **`benchmarks/morph_report.py`** -- focused MorphBoost-vs-peers comparison on a curated set of sklearn datasets (~2 minutes with `--quick`).
- **`benchmarks/morph_ablation.py`** -- toggles MorphBoost components individually on synthetic regression/binary/ranking datasets to attribute per-component impact.
- **`benchmarks/numerai_benchmark.py`** -- adds `alloygbm_morph` and `alloygbm_morph_cosine` arms, plus a startup build-freshness check that logs the loaded extension's mtime, the worktree's git HEAD, and a `STALE BUILD` warning if the extension predates HEAD.

### Documentation

- New dedicated MorphBoost guides at `docs/user/morphboost.md` and `docs/site/source/morphboost.rst`, with the formulation, full parameter reference, LR-schedule behaviour, and tuning notes.
- Cross-references and parameter notes added across the user guide, Sphinx site, benchmark READMEs, and the limitations doc.

## 0.3.2

### Bug Fixes

- **GBMRanker silent zero-tree training** -- the auto training policy's density-based `min_split_gain` and `min_loss_improvement` floors were being applied to ranking objectives, whose gradient magnitudes are an order of magnitude smaller than regression/classification gradients; on datasets with `row_count * feature_count >= 65_536` no split cleared the floor and training exited after round 1. The auto policy is now objective-aware and skips those floors for all ranking objectives (`rank:pairwise`, `rank:ndcg`, `rank:xendcg`, `queryrmse`, `yetirank`).
- **Training loop loss-regression break for ranking** -- the main training loop's unconditional `loss_improvement < 0` early-exit was firing on ranking objectives where round-to-round loss oscillation is expected; that guard is now skipped for objectives that require group IDs.
- **`GBMRanker` signature introspection** -- `inspect.signature(GBMRanker.__init__)` previously returned only `(self, ranking_objective, **kwargs)`, causing tools that build parameters via signature inspection (sklearn clone, benchmark runners, IDEs) to silently use `n_estimators=6` default; `__init__.__signature__` is now set to the combined `GBMRegressor + ranking_objective` parameter list.

### New Features

- **`stop_reason_` and `rounds_completed_` attributes** on all estimators (`GBMRegressor`, `GBMClassifier`, `GBMRanker`) -- set after `fit()` to surface the engine's early-stop reason and actual round count for diagnostics and debugging.

### Benchmarks

- **`california_ranking` scenario** -- California Housing dataset reframed as learning-to-rank: 1-degree lat/lon grid cells act as query groups (~44 queries × 468 docs = ~20,595 rows), and `median_house_value` is bucketed into 5 quantile-based relevance levels; provides a real-data complement to `synthetic_ranking`.

## 0.3.1

### Bug Fixes

- **Multiclass predictor threshold conversion** -- `convert_bin_thresholds_to_float*` functions in `crates/predictor` now correctly convert `class_trees` in addition to `trees`; previously, multiclass models with continuous float features produced near-random predictions because `class_trees` bin-ID thresholds were never converted to float values
- **Multiclass argmax label mapping** -- benchmark runner maps `np.argmax` column indices through `model.classes_` so accuracy is computed correctly when class labels are not exactly `0..K-1`

### Benchmarks

- **Real-dataset benchmark scenarios** -- added `wine_multiclass` (sklearn Wine, 3-class, 178 rows), `digits_multiclass` (sklearn Digits, 10-class, 1797 rows), `adult_income` (UCI Adult, binary classification, ~30K rows, mixed features), `abalone_regression` (UCI Abalone, regression, 4177 rows), and `news_ranking` (placeholder with setup instructions)
- **Multiclass classification support** in `run_model_comparison.py` -- stratified split, argmax predictions with label mapping, multiclass log-loss, separate factory functions with correct per-library multiclass objectives
- **Activated dormant scenarios** -- `synthetic_multiclass` and `synthetic_categorical` are now included in `AVAILABLE_SCENARIOS`
- **Rewritten `benchmarks/README.md`** -- comprehensive scenario table, task-type split strategies, feature coverage table, per-record timing reference, recently-shipped feature coverage matrix

## 0.3.0

### Native Categorical Splits

- **Fisher-sort categorical split-finding** -- optimal binary partition of categories in O(K log K) time via gradient-ordered category sorting with O(K) prefix-scan split evaluation
- **Bitset-based O(1) prediction** -- compact `Vec<u8>` bitset encoding where bit K=1 means category K goes left; prediction is a single bit-test per tree node
- **`max_cat_threshold` parameter** -- controls the maximum number of categories for native splits (default 0 = disabled, opt-in); features exceeding the threshold fall back to target encoding
- **Backward-compatible artifact format** -- new `NativeCategoricalSplits` section (ID=7) with `stump_flags` bit 1 encoding; old artifacts load without changes
- **Category-to-ID mapping** -- string categories are mapped to integer IDs at the Python layer; mappings are preserved through pickle, save/load, and get/set params
- **Full estimator support** -- works with `GBMRegressor`, `GBMClassifier`, and `GBMRanker` (via inheritance)

### Multi-Class Classification

- **`GBMClassifier` multi-class support** -- softmax (multinomial cross-entropy) objective for K > 2 classes, auto-detected from training labels
- **`predict_proba`** returns (n_samples, K) probability matrix with softmax normalization
- **Label encoding** -- arbitrary integer labels are mapped to 0..K-1 internally

### Custom Objectives and Metrics

- **Custom objective functions** via `objective=callable` -- user-defined gradient/hessian computation with fast numpy I/O
- **Custom evaluation metrics** via `eval_metric=callable` -- user-defined metric callbacks for early stopping and `evals_result_` tracking
- **`higher_is_better` protocol** -- custom metrics declare optimization direction

### Benchmarks

- **`synthetic_categorical`** benchmark scenario for evaluating native categorical split performance
- **`synthetic_custom_objective`** and **`synthetic_multiclass`** benchmark scenarios

## 0.2.0

Major capability expansion from the regression-only `0.1.x` series.

### New Estimators

- **`GBMClassifier`** -- binary classification with binary cross-entropy (log-loss) objective, `predict_proba`, `predict_log_proba`, sklearn `ClassifierMixin` integration
- **`GBMRanker`** -- learning-to-rank with 5 objectives:
  - `rank:pairwise` (RankNet)
  - `rank:ndcg` (LambdaMART)
  - `rank:xendcg` (cross-entropy NDCG approximation)
  - `queryrmse` (query-grouped RMSE)
  - `yetirank` (stochastic NDCG-weighted pairwise)

### Core Improvements

- **NaN / missing value support** across all crates -- training and prediction handle NaN natively with learned split directions
- **Sample weight support** via `fit(..., sample_weight=...)`
- **Group ID support** via `fit(..., group=...)` for ranking objectives
- **Model persistence** -- pickle round-trip, `save_model(path)` / `load_model(path)`, and `artifact_bytes` property for artifact export
- **Feature name capture** from pandas DataFrames and other named inputs
- **sklearn compatibility** -- `BaseEstimator`, `RegressorMixin`, `ClassifierMixin`, `get_params`, `set_params`, `score`, pipeline/cross-validation support
- **`min_split_gain` exposed** as a user-facing parameter

### Training Enhancements

- **Leaf-wise (best-first) tree growth** via `tree_growth="leaf"` -- similar to LightGBM's growth strategy
- **Monotone constraints** via `monotone_constraints` parameter
- **Feature importance weighting** via `feature_weights` parameter
- **`max_leaves` parameter** for leaf-budget-oriented training
- **Warm-starting / incremental training** via `warm_start=True`
- **Up to 65,535 bins per feature** (up from 256) with adaptive u8/u16 storage
- **Multiple categorical column support** via `categorical_feature_indices`
- **Histogram buffer reuse** to reduce allocation pressure
- **Objective-aware training metric tracking** -- `evals_result_` now tracks the appropriate metric per objective (RMSE, log-loss, accuracy, NDCG)

### Explanations

- **TreeSHAP** -- polynomial-time exact Shapley values (replaces the previous brute-force method limited to 20-25 features)
- SHAP explanations work with all three estimators

### New Metrics

- `accuracy` -- classification accuracy
- `log_loss` -- binary cross-entropy loss
- `ndcg` -- normalized discounted cumulative gain (with optional `k` parameter)

### Benchmarks

- **Classification scenarios**: `breast_cancer`, `synthetic_classification`
- **Ranking scenario**: `synthetic_ranking`
- Task-type-aware benchmark runner with per-type metrics, factories, and markdown rendering
- Library adapter classes for cross-library ranking comparison (LightGBM, XGBoost, CatBoost)

### Polish

- Codebase-wide hardening pass (Tier 6)
- Integration tests for warm-start, TreeSHAP, multi-categorical, wide bins, configurability, and native runtime

## 0.1.2

- Zero-copy numpy prediction (75-105x prediction speedup)
- Dense native preprocessing path
- Stage timing output in benchmarks

## 0.1.1

- Expanded benchmark suite (5 regression scenarios)
- Dataset-aware training policy improvements

## 0.1.0

- Initial public release
- `GBMRegressor` with squared-error objective
- Deterministic CPU training with Rayon parallelism
- SHAP explanations (brute-force, 20-feature limit)
- Purged time-series and panel cross-validation splits
- Native artifact prediction
- macOS arm64 and Linux x86_64 wheels
