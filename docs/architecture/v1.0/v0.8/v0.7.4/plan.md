# AlloyGBM v0.7.4 Plan (v0.8 Python SHAP Bridge Slice)

## Summary
- Goal: execute `v0.7.4` by exposing Rust SHAP runtime entrypoints through the Python extension and wiring additive SHAP methods on `GBMRegressor`.
- Success criteria:
  - Python extension exposes artifact-backed SHAP explain/global-importance functions with deterministic error mapping,
  - `GBMRegressor` exposes `shap_values` and a SHAP-based `feature_importances` path using fitted artifact state,
  - Python/runtime tests lock SHAP output shape, error behavior, and additivity consistency against predictions.
- Audience: engineers closing `v0.8` SHAP Python surface prior to parent milestone closeout.

## Scope
### In Scope
- `bindings/python/src/lib.rs`:
  - add native pyfunctions that bridge to `alloygbm-shap` artifact-backed APIs,
  - map `ShapError::InvalidInput` to `ValueError` and `ShapError::ContractViolation` to `RuntimeError`,
  - register new pyfunctions in `_alloygbm` module export surface.
- `bindings/python/alloygbm/regressor.py`:
  - add loader helpers for native SHAP bridge functions,
  - add `GBMRegressor.shap_values(X, include_expected_value=False)` using fitted `_artifact_bytes`,
  - add `GBMRegressor.feature_importances(X, method="shap")` path backed by SHAP global importance.
- Python tests:
  - contract tests for loader routing, argument validation, and regressor SHAP API behavior,
  - native runtime integration tests for SHAP bridge shape/additivity behavior and fitted-regressor SHAP consistency.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.8/v0.7.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md`

### Out of Scope
- Changes to Rust SHAP traversal/math internals in `crates/shap`.
- New model artifact sections or model format version changes.
- SHAP interaction values, approximate SHAP, or GPU/Metal SHAP.
- Parent-level `v0.8` closeout artifacts (handled after child completion set).

## Interfaces and Types
- `bindings/python/src/lib.rs`:
  - add `shap_explain_rows(artifact_bytes, rows) -> (expected_value, values)`,
  - add `shap_global_importance(artifact_bytes, rows) -> [(feature_name, importance)]`.
- `bindings/python/alloygbm/regressor.py`:
  - add `shap_values(...)` and `feature_importances(...)` methods,
  - preserve existing `fit/predict` signatures/behavior.
- `bindings/python/Cargo.toml`:
  - add dependency on `alloygbm-shap`.

Backward-compatibility expectations:
- Existing `fit`, `predict`, and predictor bridge APIs remain unchanged.
- Existing Python contract tests continue passing.
- Rust SHAP APIs remain source-of-truth for explanation semantics.

## Deliverables
1. Native bridge package:
  - Python-extension SHAP explain/global-importance functions registered and callable.
2. Regressor API package:
  - additive `shap_values` + SHAP feature-importance routing on `GBMRegressor`.
3. Test package:
  - Python contract/runtime tests proving shape, errors, and additivity consistency.
4. State package:
  - `implementation_notes.md`, `verification_report.md`, and layer index update to next target.

## Implementation Sequence
1. Author `v0.7.4` plan and lock scope to Python bridge work only.
2. Add `alloygbm-shap` dependency in `bindings/python/Cargo.toml`.
3. Implement SHAP bridge functions and error mapping in `bindings/python/src/lib.rs`.
4. Implement `GBMRegressor` SHAP methods and loader helpers in `regressor.py`.
5. Add/expand Python tests for SHAP API contract and runtime additivity checks.
6. Run verification gates and resolve failures.
7. Write `implementation_notes.md` and `verification_report.md`.
8. Update `docs/architecture/state/layer_index.yaml` to mark `v0.7.4` verified and advance next target.

## Test Cases and Scenarios
- Unit/contract cases:
  - `GBMRegressor.shap_values` requires fitted model and validates feature count,
  - `feature_importances(method=...)` rejects unsupported methods,
  - SHAP loaders route to native bridge functions with expected argument shapes.
- Integration/runtime cases:
  - native `shap_explain_rows` returns expected shape and satisfies additivity vs predictor output,
  - regressor `shap_values` output rows x feature_count and optional expected value path,
  - regressor `feature_importances(..., method="shap")` returns deterministic ordering.
- Failure/edge cases:
  - invalid artifacts propagate deterministic runtime errors,
  - malformed rows propagate validation errors through Python layer.
- Acceptance test mapping:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.8/v0.7.4/plan.md` exists and is decision-complete.
2. Python extension exports SHAP explain/global-importance bridge functions backed by `alloygbm-shap`.
3. SHAP bridge errors map deterministically to Python exceptions.
4. `GBMRegressor.shap_values` is available and returns additive SHAP outputs with deterministic shape.
5. `GBMRegressor.feature_importances(..., method="shap")` is available and returns SHAP global importance.
6. Python contract/runtime tests cover SHAP shape, errors, and additivity consistency.
7. `docs/architecture/v1.0/v0.8/v0.7.4/implementation_notes.md` is created.
8. `docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md` is created.
9. `cargo fmt -- --check` passes.
10. `cargo clippy --workspace --all-targets -- -D warnings` passes.
11. `cargo test --workspace` passes.
12. Python unittest suite passes.

## Risks and Mitigations
- Risk: Python SHAP surface drifts from Rust SHAP semantics.
  - Mitigation: bridge directly to artifact-backed Rust APIs and assert additivity in runtime tests.
- Risk: SHAP API additions regress existing regressor contract behavior.
  - Mitigation: preserve fit/predict code paths and run existing contract/integration suites unchanged.
- Risk: unclear shape/return contract for expected value handling.
  - Mitigation: define explicit `include_expected_value` parameter with deterministic return behavior.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Additivity tolerance remains inherited from Rust SHAP checks; Python tests use tolerance-based assertions.
- SHAP feature importance method default is `"shap"` for this layer and unsupported methods raise `ValueError`.
