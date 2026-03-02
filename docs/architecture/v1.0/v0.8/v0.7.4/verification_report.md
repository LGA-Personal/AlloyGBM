# AlloyGBM v0.7.4 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.8/v0.7.4`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.8/v0.7.4/plan.md` exists and is decision-complete.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.4/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/plan.md) added with scope/interfaces/sequence/tests/criteria.
- Status: PASS

- Criterion: (2) Python extension exports SHAP explain/global-importance bridge functions backed by `alloygbm-shap`.
- Evidence: `shap_explain_rows` and `shap_global_importance` added and exported in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs); `alloygbm-shap` dependency added in [bindings/python/Cargo.toml](/Users/lashby/Projects/AlloyGBM/bindings/python/Cargo.toml).
- Status: PASS

- Criterion: (3) SHAP bridge errors map deterministically to Python exceptions.
- Evidence: `shap_error_to_pyerr(...)` in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs) maps invalid input to `PyValueError` and contract violations to `PyRuntimeError`.
- Status: PASS

- Criterion: (4) `GBMRegressor.shap_values` is available and returns additive SHAP outputs with deterministic shape.
- Evidence: `GBMRegressor.shap_values` added in [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py); runtime tests validate shape and additivity.
- Status: PASS

- Criterion: (5) `GBMRegressor.feature_importances(..., method="shap")` is available and returns SHAP global importance.
- Evidence: `GBMRegressor.feature_importances` added in [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py); contract/runtime tests validate bridge routing and parity with native output.
- Status: PASS

- Criterion: (6) Python contract/runtime tests cover SHAP shape, errors, and additivity consistency.
- Evidence:
  - contract coverage in [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py),
  - runtime coverage in [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py),
  - Rust binding-unit coverage in [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs).
- Status: PASS

- Criterion: (7) `implementation_notes.md` is created.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.4/implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/implementation_notes.md).
- Status: PASS

- Criterion: (8) `verification_report.md` is created.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md).
- Status: PASS

- Criterion: (9) `cargo fmt -- --check` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (10) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (11) `cargo test --workspace` passes.
- Evidence: command executed successfully; all workspace tests passed including updated Python-binding crate tests.
- Status: PASS

- Criterion: (12) Python unittest suite passes.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passed (`Ran 67 tests`, `OK`).
- Status: PASS

## Criterion-to-Test Mapping
- Criterion: (2), (3)
- Tests/Checks:
  - `tests::shap_bridge_explain_rows_matches_model_additivity`
  - `tests::shap_bridge_global_importance_is_sorted_descending`
  - module export registration in `_alloygbm` init path

- Criterion: (4)
- Tests/Checks:
  - `NativeRuntimeIntegrationTests::test_runtime_native_shap_explain_rows_is_additive`
  - `NativeRuntimeIntegrationTests::test_public_regressor_shap_values_and_feature_importances_match_native`
  - `GBMRegressorContractTests::test_shap_values_use_native_bridge_with_optional_expected_value`
  - `GBMRegressorContractTests::test_shap_values_reject_feature_count_mismatch`

- Criterion: (5)
- Tests/Checks:
  - `NativeRuntimeIntegrationTests::test_runtime_native_shap_global_importance_returns_expected_shape`
  - `NativeRuntimeIntegrationTests::test_public_regressor_shap_values_and_feature_importances_match_native`
  - `GBMRegressorContractTests::test_feature_importances_use_native_shap_global_bridge`
  - `GBMRegressorContractTests::test_feature_importances_reject_unsupported_method`

- Criterion: command gates
- Tests/Checks:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Tests Added or Updated
- File: [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs)
- Added tests:
  - `shap_bridge_explain_rows_matches_model_additivity`
  - `shap_bridge_global_importance_is_sorted_descending`

- File: [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
- Added tests:
  - `test_shap_values_requires_fit`
  - `test_feature_importances_requires_fit`
  - `test_shap_values_use_native_bridge_with_optional_expected_value`
  - `test_shap_values_reject_feature_count_mismatch`
  - `test_feature_importances_use_native_shap_global_bridge`
  - `test_feature_importances_reject_unsupported_method`

- File: [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
- Added tests:
  - `test_runtime_native_shap_explain_rows_is_additive`
  - `test_runtime_native_shap_global_importance_returns_expected_shape`
  - `test_public_regressor_shap_values_and_feature_importances_match_native`

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS
- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS
- Command: `cargo test --workspace`
- Result: PASS
- Command: `cargo doc --workspace --no-deps`
- Result: PASS
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 67 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- SHAP expected-value semantics depend on model/row distribution and are currently exposed through `include_expected_value`; callers must use the same row set when comparing additivity and global importance.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: mark layer state and proceed to parent `v0.8` closeout child (`v0.7.5`).
