# v0.1.9 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.1.9`
- Date: 2026-02-26

## Acceptance Criteria Matrix
- Criterion 1: runtime integration tests execute native predictor inference successfully on valid artifact bytes and assert deterministic expected values.
- Evidence: `bindings/python/tests/test_native_runtime_integration.py::test_runtime_native_predictor_entrypoint_returns_expected_values` executes `_alloygbm.predictor_predict_batch` on valid fixture artifact and asserts deterministic predictions to 5 decimal places.
- Status: PASS

- Criterion 2: runtime integration tests verify `GBMRegressor.predict_from_artifact(...)` matches direct native predictor predictions on valid artifact bytes.
- Evidence: `bindings/python/tests/test_native_runtime_integration.py::test_public_regressor_bridge_matches_native_success_path` compares bridge predictions and native predictions from the same fixture payload/rows.
- Status: PASS

- Criterion 3: existing runtime integration error-path checks continue passing.
- Evidence:
  - `test_runtime_native_predictor_entrypoint_executes`
  - `test_public_regressor_bridge_uses_native_extension_runtime`
  both pass and assert `RuntimeError` propagation on invalid artifact bytes.
- Status: PASS

- Criterion 4: existing Python regressor contract tests remain passing.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` reports `Ran 15 tests` and `OK`.
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
- Evidence: command output shows `Ran 15 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.9` acceptance criteria.

## Residual Risks
- Parent rollup verification artifacts for `v0.2` and `v1.0` remain pending.

## Final Readiness
- Ready: Yes (for `v0.1.9` Python runtime success-path parity scope).
