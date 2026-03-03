# v0.1.7 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.1.7`
- Date: 2026-02-26

## Acceptance Criteria Matrix
- Criterion 1: Python binding module exports a predictor-backed batch inference function that accepts artifact bytes and feature rows.
- Evidence: `bindings/python/src/lib.rs` exports `predictor_predict_batch` via `#[pyfunction]` and registers it in `_alloygbm` module.
- Status: PASS

- Criterion 2: binding function predictions match engine predictions from the same serialized model bytes on deterministic fixture rows.
- Evidence: binding-crate unit test `binding_bridge_predictions_match_engine_predictions` (in `bindings/python/src/lib.rs`) trains deterministic engine fixture data, serializes artifact bytes, invokes binding bridge implementation (`predictor_predict_batch_impl`), and asserts exact parity with engine `predict_batch` outputs.
- Status: PASS

- Criterion 3: binding function rejects invalid inputs (for example feature-count mismatch or empty rows) with clear Python errors.
- Evidence:
  - mapping in `bindings/python/src/lib.rs`: `PredictorError::InvalidInput` -> `PyValueError`.
  - predictor tests still passing for invalid input paths:
    - `predictor_row_rejects_feature_count_mismatch`
    - `batch_rejects_empty_rows`
  - Python contract tests verify artifact bridge wrapper behavior and error propagation.
- Status: PASS

- Criterion 4: existing Python `GBMRegressor` contract tests remain passing.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` reports `Ran 10 tests` and `OK`.
- Status: PASS

- Criterion 5: `cargo fmt -- --check` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion 6: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command completed successfully across workspace targets.
- Status: PASS

- Criterion 7: `cargo test --workspace` passes.
- Evidence: workspace suites all green, including `alloygbm_predictor` (`5 passed`) and `alloygbm_engine` (`40 passed`).
- Status: PASS

- Criterion 8: `cargo doc --workspace --no-deps` passes.
- Evidence: docs generation completed successfully under `target/doc`.
- Status: PASS

- Criterion 9: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command output shows `Ran 10 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.7` acceptance criteria.

## Residual Risks
- Python-runtime import execution of the extension function is still not exercised in `bindings/python/tests`; current evidence validates bridge behavior at Rust binding-layer plus Python wrapper contract level.
- Parent rollup verification artifacts for `v0.1` and `v1.0` remain pending.

## Final Readiness
- Ready: Yes (for `v0.1.7` Python binding predictor bridge scope).
