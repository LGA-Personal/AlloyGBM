# v0.1.7 Implementation Notes

## Summary of What Was Built
- Added predictor-backed Python native binding entry point in `bindings/python/src/lib.rs`:
  - new exported function: `predictor_predict_batch(artifact_bytes, rows)`.
  - maps predictor errors to Python exceptions:
    - `PredictorError::InvalidInput` -> `ValueError`
    - `PredictorError::ContractViolation` / core decode failures -> `RuntimeError`
- Updated `bindings/python/Cargo.toml` to include `alloygbm-predictor` dependency for binding implementation.
- Extended Python regressor contract in `bindings/python/alloygbm/regressor.py`:
  - added `GBMRegressor.predict_from_artifact(...)` static method for artifact-backed inference through the native bridge.
  - added lazy native-loader helper with explicit unavailability error messaging.
- Added Python contract tests in `bindings/python/tests/test_regressor_contract.py`:
  - artifact payload type validation.
  - bridge loader invocation/argument forwarding behavior.
  - bridge loader error propagation behavior.

## Non-Intuitive Decisions
- Decision: do not keep Rust unit tests inside `bindings/python/src/lib.rs`.
- Reason: this crate is built as a PyO3 extension module (`extension-module`), and adding in-crate Rust tests causes `cargo test --workspace` linker failures on unresolved Python symbols for the test harness binary.
- Impact: verification coverage relies on:
  - existing predictor crate parity tests (`engine` vs `predictor` from shared artifact bytes),
  - bridge implementation being a thin delegation layer,
  - Python-level contract tests for wrapper behavior.

## Plan Contradictions and Why
- Partial contradiction to the initial `v0.1.7` plan wording about adding direct binding-layer parity tests inside the Python crate.
- Resolution: maintained workspace gate compatibility by using delegation + existing predictor parity evidence instead of in-crate PyO3 test harnesses.

## Boundary/Interface Changes vs Plan
- Added a new native module function `predictor_predict_batch`.
- Added a new Python API entry point `GBMRegressor.predict_from_artifact`.
- No changes to engine training semantics, predictor artifact format, or existing estimator `fit/predict` baseline behavior.

## Known Gaps Deferred to Next Layer
- Direct Python-native end-to-end parity execution (through an installed/imported extension in Python runtime) is not yet automated in `bindings/python/tests`.
- Parent rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/verification_report.md`
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Follow-Up Actions
- Define `v0.1.8` to add a stable Python-side native extension test harness (wheel/import path) for direct execution of `predictor_predict_batch` from Python test runtime.
