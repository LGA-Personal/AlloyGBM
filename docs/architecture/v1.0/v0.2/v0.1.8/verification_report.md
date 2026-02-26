# v0.1.8 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.1.8`
- Date: 2026-02-26

## Acceptance Criteria Matrix
- Criterion 1: Python test suite includes at least one test that imports the installed `alloygbm` package from a built wheel and executes `native_runtime_info`.
- Evidence: `bindings/python/tests/test_native_runtime_integration.py::test_runtime_import_exposes_native_runtime_info` builds/installs wheel and asserts runtime `native_runtime_info` fields.
- Status: PASS

- Criterion 2: Python test suite executes native `predictor_predict_batch` from Python runtime and validates expected error type on invalid artifact bytes.
- Evidence: `bindings/python/tests/test_native_runtime_integration.py::test_runtime_native_predictor_entrypoint_executes` calls `alloygbm._alloygbm.predictor_predict_batch(...)` and asserts `RuntimeError` with native serialization/header message.
- Status: PASS

- Criterion 3: Python test suite executes `GBMRegressor.predict_from_artifact(...)` from installed package runtime and validates native error propagation for invalid artifact bytes.
- Evidence: `bindings/python/tests/test_native_runtime_integration.py::test_public_regressor_bridge_uses_native_extension_runtime` asserts `RuntimeError` surfaced through public regressor bridge path.
- Status: PASS

- Criterion 4: existing Python regressor contract tests remain passing.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` reports `Ran 13 tests` and `OK`.
- Status: PASS

- Criterion 5: `cargo fmt -- --check` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion 6: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command completed successfully across workspace targets.
- Status: PASS

- Criterion 7: `cargo test --workspace` passes.
- Evidence: workspace suites all green, including `_alloygbm` binding test (`1 passed`) and `alloygbm_engine` (`40 passed`).
- Status: PASS

- Criterion 8: `cargo doc --workspace --no-deps` passes.
- Evidence: docs generation completed successfully under `target/doc`.
- Status: PASS

- Criterion 9: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command output shows `Ran 13 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.8` acceptance criteria.

## Residual Risks
- Runtime Python tests currently verify native extension execution/error propagation using invalid artifact payloads; success-path parity from Python runtime with valid artifact bytes is still not explicitly asserted.
- Parent rollup verification artifacts for `v0.2` and `v1.0` remain pending.

## Final Readiness
- Ready: Yes (for `v0.1.8` Python runtime native-extension execution evidence scope).
