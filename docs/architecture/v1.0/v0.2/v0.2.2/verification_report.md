# AlloyGBM v0.2.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.2.2`
- Date: 2026-02-26

## Acceptance Criteria Matrix
- Criterion: `GBMRegressor.fit(...)` accepts sequence rows and NumPy-like/pandas-like/Polars-like tabular objects.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` routes `fit(...)` through `_coerce_sequence_like(...)`.
  - `bindings/python/tests/test_regressor_contract.py`: `test_fit_accepts_numpy_pandas_polars_like_inputs`.
- Status: PASS

- Criterion: `GBMRegressor.predict(...)` and `GBMRegressor.predict_from_artifact(...)` accept the same adapter-supported row inputs.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` routes prediction entrypoints through `_coerce_sequence_like(...)`.
  - `bindings/python/tests/test_regressor_contract.py`: `test_predict_accepts_pandas_like_rows`, `test_predict_from_artifact_accepts_polars_like_rows`.
  - `bindings/python/tests/test_native_runtime_integration.py`: `test_public_regressor_accepts_dataframe_like_adapters`.
- Status: PASS

- Criterion: Feature-count mismatch and malformed-input error semantics remain deterministic.
- Evidence:
  - Existing guards preserved in `regressor.py` (`X` row-shape checks and fitted feature-count comparison).
  - `bindings/python/tests/test_regressor_contract.py`: `test_predict_rejects_feature_count_mismatch`, `test_fit_rejects_non_convertible_adapter_inputs`, `test_predict_from_artifact_rejects_non_convertible_rows`.
- Status: PASS

- Criterion: Parameter contract behavior (`get_params`/`set_params`) remains passing.
- Evidence:
  - `bindings/python/tests/test_regressor_contract.py`: `test_get_params_and_set_params_roundtrip`.
  - Python unittest discovery run passes.
- Status: PASS

- Criterion: `cargo fmt -- --check` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: workspace tests and doc-tests pass.
- Status: PASS

- Criterion: `cargo doc --workspace --no-deps` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command run completed successfully (`Ran 20 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- File: `bindings/python/tests/test_regressor_contract.py`
- Purpose: validate duck-typed adapter support for fit/predict/predict_from_artifact and explicit non-convertible-input failure behavior.

- File: `bindings/python/tests/test_native_runtime_integration.py`
- Purpose: validate adapter inputs against wheel-installed native runtime execution.

## Commands Executed
- Command: `python3 -m unittest bindings/python/tests/test_regressor_contract.py`
- Result: PASS (`Ran 15 tests`)

- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 22 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Adapter normalization broadens accepted input containers, but native training bridge still assumes pre-binned integer-valued feature semantics.
- Estimator round-count configuration remains deferred.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: create and execute `v0.2.3` for remaining `v0.2` wrapper gaps.
