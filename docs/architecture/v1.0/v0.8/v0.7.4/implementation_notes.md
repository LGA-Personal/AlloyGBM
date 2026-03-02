# AlloyGBM v0.7.4 Implementation Notes

## Summary of What Was Built
- Executed `v0.7.4` by exposing Rust SHAP APIs through the Python extension and wiring additive SHAP methods on `GBMRegressor`.
- Updated [bindings/python/Cargo.toml](/Users/lashby/Projects/AlloyGBM/bindings/python/Cargo.toml):
  - added dependency on `alloygbm-shap`.
- Updated [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - added SHAP bridge functions:
    - `shap_explain_rows(artifact_bytes, rows) -> (expected_value, values)`
    - `shap_global_importance(artifact_bytes, rows) -> [(feature_name, importance)]`
  - added deterministic error mapping for SHAP bridge:
    - `ShapError::InvalidInput` -> `PyValueError`
    - `ShapError::ContractViolation` -> `PyRuntimeError`
  - registered SHAP functions in `_alloygbm` module exports.
  - added Rust-side binding tests for SHAP additivity and global-importance ordering.
- Updated [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - added native loader helpers for SHAP bridge functions,
  - added `GBMRegressor.shap_values(X, include_expected_value=False)`,
  - added `GBMRegressor.feature_importances(X, method="shap")`.
- Updated Python tests:
  - [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)
  - [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
  - Added SHAP contract/runtime coverage for shape, method routing, errors, and additivity consistency.

## Non-Intuitive Decisions
- Decision: `GBMRegressor.shap_values` supports `include_expected_value=False` by default, with optional tuple return when `True`.
- Reason: preserve sklearn-like `shap_values` default while still enabling deterministic additivity checks without adding a separate API.
- Impact: default call returns matrix-only values; tests can request expected value explicitly for additivity assertions.

- Decision: `GBMRegressor.feature_importances` requires `X` and currently accepts only `method="shap"`.
- Reason: SHAP global importance is row-distribution dependent and should be explicit about the evaluation dataset.
- Impact: feature importance behavior is deterministic and traceable to the provided rows; unsupported methods fail fast with `ValueError`.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted in `v0.7.4/plan.md`.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- In-scope boundary changes only:
  - Python extension API surface expanded with SHAP bridge functions.
  - Python regressor API surface expanded with SHAP methods.
- Out-of-scope boundaries preserved:
  - no changes to Rust SHAP traversal/math internals,
  - no model format or artifact compatibility policy changes.

## Known Gaps Deferred to Next Layer
- Parent `v0.8` closeout artifacts remain pending:
  - `docs/architecture/v1.0/v0.8/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/verification_report.md`
- Additional SHAP modes (interaction/approximate) remain out-of-scope for this milestone.

## Follow-Up Actions
- Plan next child layer under `v0.8` (`v0.7.5`) for parent closeout and any remaining integration hardening.
- Preserve new SHAP bridge/additivity tests as non-regression gates during `v0.8` milestone closeout.
