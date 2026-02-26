# AlloyGBM v0.2.1 Implementation Notes

## Summary of What Was Built
- Added native training bridge entrypoint in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - `train_regression_artifact(...)` now trains via `Trainer` + `CpuBackend` + `SquaredErrorObjective` and returns serialized artifact bytes.
  - Added `train_regression_artifact_impl(...)` with deterministic pre-binned dataset construction and training flow.
- Updated [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - `GBMRegressor.fit(...)` now calls the native training bridge and stores fitted artifact bytes.
  - `GBMRegressor.predict(...)` now routes through native `predictor_predict_batch` using fitted artifact state.
  - Removed constant-baseline internal fit/predict behavior.
- Updated test evidence:
  - [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py): contract tests now verify bridge wiring through mocked native loaders.
  - [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py): added runtime wheel-installed deterministic native `fit/predict` integration coverage.
  - [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs): added Rust unit test `train_bridge_artifact_predictions_match_engine_predictions`.
- Updated binding dependencies in [bindings/python/Cargo.toml](/Users/lashby/Projects/AlloyGBM/bindings/python/Cargo.toml) to include `alloygbm-core`, `alloygbm-engine`, and `alloygbm-backend-cpu` as runtime dependencies.

## Non-Intuitive Decisions
- Decision: native training bridge currently requires pre-binned integer-valued, non-negative feature inputs.
- Reason: current engine/predictor threshold contract is bin-index-based; this layer stays scoped to native-backed fit/predict wiring without introducing full quantization metadata plumbing.
- Impact: `GBMRegressor.fit/predict` now execute natively but assume bin-like feature values until a later adapter/quantization slice.

- Decision: use fixed training rounds (`DEFAULT_TRAIN_ROUNDS = 6`) in the native bridge.
- Reason: `GBMRegressor` parameter surface in this layer does not yet include estimator-round control.
- Impact: deterministic and non-trivial native predictions are available now; estimator-round configurability is deferred.

- Decision: keep source-tree contract tests native-independent by mocking loader functions.
- Reason: `test_regressor_contract.py` is designed as fast contract/unit coverage without requiring built extension artifacts.
- Impact: real native execution evidence is captured in runtime integration tests instead.

## Plan Contradictions and Why
- Original Plan Statement: implementation and verification artifacts are both listed as deliverables for the layer.
- Implemented Decision: completed implementation and gate-command validation in this pass, and wrote `implementation_notes.md`; verification artifact is deferred.
- Reason: this execution pass follows implementation-skill boundary (`alloy-layer-implement`) while preserving explicit validation evidence for handoff to verify/closeout.
- Impact: layer status is advanced to implemented-with-notes, not verified.
- Rollback or Migration Consideration: none; adding `verification_report.md` later is additive and does not require code rollback.

## Boundary/Interface Changes vs Plan
- Added new Python-extension public function `train_regression_artifact(...)` in `_alloygbm`.
- Changed `GBMRegressor` runtime boundary from pure-Python baseline training to native artifact-backed training/prediction.
- No parent-plan/roadmap scope expansion was introduced.

## Known Gaps Deferred to Next Layer
- Full NumPy/pandas/Polars adapter normalization beyond sequence-of-sequences contract.
- Non-binned continuous-feature quantization metadata and transformation persistence for prediction-time mapping.
- Estimator round-count configurability (`n_estimators`-style control).
- Layer `verification_report.md` publication.

## Follow-Up Actions
- Add `docs/architecture/v1.0/v0.3/v0.2.1/verification_report.md` with criterion-mapped evidence from this pass.
- Keep `docs/architecture/state/layer_index.yaml` synchronized after verification closeout.
- Open `v0.2.2` plan for input-adapter/quantization contract expansion once `v0.2.1` is verified.
