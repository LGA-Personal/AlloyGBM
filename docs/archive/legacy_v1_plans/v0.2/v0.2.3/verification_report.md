# AlloyGBM v0.2.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.2.3`
- Date: 2026-02-27

## Acceptance Criteria Matrix
- Criterion: `GBMRegressor(...)` accepts `n_estimators` with default value and rejects non-positive values.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` validates `n_estimators > 0` in constructor.
  - `bindings/python/tests/test_regressor_contract.py`: `test_constructor_rejects_invalid_values`.
- Status: PASS

- Criterion: `GBMRegressor.get_params()` and `set_params(...)` include `n_estimators` with stable sklearn-style behavior.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` includes `n_estimators` in `get_params` and `set_params`.
  - `bindings/python/tests/test_regressor_contract.py`: `test_get_params_and_set_params_roundtrip`, `test_set_params_rejects_invalid_n_estimators`.
- Status: PASS

- Criterion: `GBMRegressor.fit(...)` forwards configured `n_estimators` to native `train_regression_artifact(...)`.
- Evidence:
  - `bindings/python/alloygbm/regressor.py` forwards `rounds=self.n_estimators`.
  - `bindings/python/tests/test_regressor_contract.py`: `test_fit_and_predict_use_native_bridges` asserts forwarded round count in mock train call.
- Status: PASS

- Criterion: Native training bridge uses caller-provided rounds and rejects invalid round counts.
- Evidence:
  - `bindings/python/src/lib.rs`: `train_regression_artifact(...)` accepts `rounds` and checks `rounds == 0`.
  - `bindings/python/tests/test_native_runtime_integration.py`: `test_runtime_train_bridge_rejects_zero_rounds`.
- Status: PASS

- Criterion: Python contract/runtime tests provide evidence that configured rounds flow through successfully while existing predict/error contracts remain passing.
- Evidence:
  - `bindings/python/tests/test_native_runtime_integration.py`: `test_public_regressor_n_estimators_controls_training_rounds`.
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes (`Ran 25 tests`, `OK`).
- Status: PASS

- Criterion: `cargo fmt -- --check` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: command run completed successfully (all workspace crate/unit/doc tests passing).
- Status: PASS

- Criterion: `cargo doc --workspace --no-deps` passes.
- Evidence: command run completed successfully.
- Status: PASS

- Criterion: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command run completed successfully (`Ran 25 tests`, `OK`).
- Status: PASS

## Tests Added or Updated
- File: `bindings/python/tests/test_regressor_contract.py`
- Purpose: validate `n_estimators` constructor/set_params behavior and forwarding through mocked native train calls.

- File: `bindings/python/tests/test_native_runtime_integration.py`
- Purpose: verify runtime bridge rejects zero rounds and that `n_estimators` changes trained artifacts/predictions through the compiled extension.

## Commands Executed
- Command: `python3 -m unittest bindings/python/tests/test_regressor_contract.py`
- Result: PASS (`Ran 16 tests`, `OK`)

- Command: `cargo test -p alloygbm-python`
- Result: PASS (`2 passed`)

- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS

- Command: `cargo doc --workspace --no-deps`
- Result: PASS

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 25 tests`, `OK`)

## Residual Risks
- Native training bridge still expects pre-binned integer-valued non-negative features; generic continuous-feature quantization remains out of scope for this layer.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: close `v0.2` parent rollup artifacts and update layer index target to the next parent-level closure step.
