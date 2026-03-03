# AlloyGBM v0.7.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.7/v0.7.1`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.7/v0.7.1/plan.md` is present and decision-complete for this slice.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.1/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/plan.md) added with objective/scope/interfaces/sequence/tests/criteria/risks/defaults.
- Status: PASS

- Criterion: (2) `crates/shap` provides artifact-backed SHAP batch explanation output with expected value + contribution matrix.
- Evidence: [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs) adds `ShapExplanationBatch` and `explain_rows_from_artifact_bytes(...)`.
- Status: PASS

- Criterion: (3) SHAP API validates rows and artifact compatibility deterministically with explicit errors.
- Evidence: `validate_rows(...)`, `load_artifact_context(...)`, and compatibility checks in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (4) SHAP output row width equals model feature count for all rows.
- Evidence: `explain_rows_from_artifact_has_deterministic_shape_and_additivity` asserts matrix dimensions in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (5) Fixture tests prove additivity (`expected_value + sum(phi_i)` matches model prediction within tolerance).
- Evidence: `explain_rows_from_artifact_has_deterministic_shape_and_additivity` validates per-row additivity using `ADDITIVITY_TOLERANCE` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (6) Global importance is computed from mean absolute SHAP contribution and returned in deterministic ordering.
- Evidence: `global_importance_from_shap_values(...)` and tests `global_importance_aggregates_mean_absolute_contribution` + `global_importance_from_artifact_uses_metadata_feature_names` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (7) `implementation_notes.md` is created.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.1/implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/implementation_notes.md).
- Status: PASS

- Criterion: (8) `verification_report.md` is created.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.1/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/verification_report.md).
- Status: PASS

- Criterion: (9) `cargo fmt -- --check` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (10) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (11) `cargo test --workspace` passes.
- Evidence: command executed successfully; all workspace tests passed including 8 `alloygbm-shap` tests.
- Status: PASS

- Criterion: (12) Python unittest suite passes.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passed (`Ran 58 tests`, `OK`).
- Status: PASS

## Criterion-to-Test Mapping
- Criterion: (1) Plan presence and completeness
- Tests/Checks: documentation artifact existence check for [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/plan.md)

- Criterion: (2) Artifact-backed SHAP explanation API
- Tests/Checks: `tests::explain_rows_from_artifact_has_deterministic_shape_and_additivity` and successful `cargo test -p alloygbm-shap`

- Criterion: (3) Deterministic row/artifact validation errors
- Tests/Checks: `tests::explain_rows_from_artifact_rejects_empty_rows`, `tests::explain_rows_from_artifact_rejects_feature_count_mismatch`, `tests::explain_rows_from_artifact_rejects_non_finite_features`, `tests::explain_rows_from_artifact_rejects_incompatible_required_sections`

- Criterion: (4) SHAP output width equals feature count
- Tests/Checks: `tests::explain_rows_from_artifact_has_deterministic_shape_and_additivity`

- Criterion: (5) Additivity identity within tolerance
- Tests/Checks: `tests::explain_rows_from_artifact_has_deterministic_shape_and_additivity`

- Criterion: (6) Global importance from mean absolute SHAP with deterministic ordering
- Tests/Checks: `tests::global_importance_aggregates_mean_absolute_contribution`, `tests::global_importance_from_artifact_uses_metadata_feature_names`

- Criterion: (7) implementation notes artifact exists
- Tests/Checks: documentation artifact existence check for [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/implementation_notes.md)

- Criterion: (8) verification report artifact exists
- Tests/Checks: documentation artifact existence check for [verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.1/verification_report.md)

- Criterion: (9) formatting gate
- Tests/Checks: `cargo fmt -- --check`

- Criterion: (10) lint gate
- Tests/Checks: `cargo clippy --workspace --all-targets -- -D warnings`

- Criterion: (11) workspace test gate
- Tests/Checks: `cargo test --workspace` (all crate/unit/doc tests pass)

- Criterion: (12) Python suite gate
- Tests/Checks: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` (`Ran 58 tests`, `OK`)

## Tests Added or Updated
- File: [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs)
- Purpose: add SHAP contract/additivity/global-importance tests and regression coverage for input and artifact validation.

## Commands Executed
- Command: `cargo test -p alloygbm-shap`
- Result: PASS (8 tests passed)
- Command: `cargo fmt -- --check`
- Result: PASS
- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS
- Command: `cargo test --workspace`
- Result: PASS
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 58 tests`, `OK`)

## Residual Uncovered Criteria
- None. All acceptance criteria have direct test/check evidence and passing command output.

## Residual Risks
- Current contribution assignment is a contract/additivity harness baseline; exact TreeSHAP path weighting remains deferred to `v0.7.2`.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: plan and execute `v0.7.2` exact TreeSHAP traversal layer while preserving `v0.7.1` contract tests.
