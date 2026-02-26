# AlloyGBM v0.2.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.3/v0.2.1`
- Date: 2026-02-26

## Acceptance Criteria Matrix
- Criterion: `GBMRegressor.fit(...)` stores native-backed fitted state and no longer relies solely on mean-target baseline behavior.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` now loads and calls `_alloygbm.train_regression_artifact` in `fit(...)`.
  - `bindings/python/tests/test_regressor_contract.py` includes `test_fit_and_predict_use_native_bridges`.
- Status: PASS

- Criterion: `GBMRegressor.predict(...)` executes against fitted native-backed state and preserves feature-count guardrails.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` now calls native `predictor_predict_batch` using stored artifact bytes.
  - `bindings/python/tests/test_regressor_contract.py` keeps `test_predict_rejects_feature_count_mismatch`.
- Status: PASS

- Criterion: Existing parameter contract tests (`get_params`/`set_params`) remain passing.
- Evidence:
  - `bindings/python/tests/test_regressor_contract.py` keeps `test_get_params_and_set_params_roundtrip`.
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS.
- Status: PASS

- Criterion: Python tests include deterministic evidence that fitted predictions are produced through native-backed behavior on fixture data.
- Evidence:
  - `bindings/python/tests/test_native_runtime_integration.py` adds `test_public_regressor_fit_predict_is_native_backed_and_deterministic`.
  - `bindings/python/src/lib.rs` adds `train_bridge_artifact_predictions_match_engine_predictions`.
  - `cargo test -p alloygbm-python` -> PASS.
- Status: PASS

- Criterion: `cargo fmt -- --check` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: command run completed successfully (workspace crates and doc-tests passing).
- Status: PASS

- Criterion: `cargo doc --workspace --no-deps` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command run completed successfully (`Ran 16 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- File: `bindings/python/src/lib.rs`
- Purpose: add Rust-side parity test for bridge-trained artifact prediction equivalence.

- File: `bindings/python/tests/test_regressor_contract.py`
- Purpose: verify `GBMRegressor.fit/predict` bridge wiring and preserve parameter/shape contract checks.

- File: `bindings/python/tests/test_native_runtime_integration.py`
- Purpose: verify wheel-installed runtime `GBMRegressor.fit/predict` deterministic native execution.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test -p alloygbm-python`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 16 tests`, `OK`)

## Residual Risks
- Native training bridge currently assumes pre-binned integer-valued non-negative feature inputs; generic dataframe/continuous-feature adapter behavior is deferred.
- Estimator round-count configurability is deferred beyond this layer.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: open next child layer (`v0.2.2`) for adapter/quantization and remaining wrapper-surface gaps.
