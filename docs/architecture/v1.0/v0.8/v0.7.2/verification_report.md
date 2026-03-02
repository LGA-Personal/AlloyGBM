# AlloyGBM v0.7.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.8/v0.7.2`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.8/v0.7.2/plan.md` exists and is decision-complete.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.2/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.2/plan.md) added with scope/interfaces/sequence/tests/criteria.
- Status: PASS

- Criterion: (2) `crates/shap` computes expected value via traversal expectation, not direct baseline passthrough.
- Evidence: `expected_prediction_for_subset(...)` + `expected_subtree(...)` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs) and test `explain_rows_from_artifact_computes_exact_expected_value_and_contributions`.
- Status: PASS

- Criterion: (3) `crates/shap` computes per-feature SHAP using exact Shapley subset weighting over split features.
- Evidence: `shapley_values_for_row(...)` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs) with factorial weighting across subset masks.
- Status: PASS

- Criterion: (4) Fixture tests verify exact expected value and contribution values for representative rows.
- Evidence: `explain_rows_from_artifact_computes_exact_expected_value_and_contributions` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs) asserts expected value `2.25` and row-level contribution matrix.
- Status: PASS

- Criterion: (5) Additivity check remains enforced and passing for fixture rows.
- Evidence: `verify_additivity(...)` path and fixture test reconstruction assertions in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (6) Global importance remains deterministic and derived from mean absolute SHAP contributions.
- Evidence: `global_importance_from_shap_values(...)` + tests `global_importance_aggregates_mean_absolute_contribution` and `global_importance_from_artifact_uses_metadata_feature_names` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (7) `implementation_notes.md` is created.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.2/implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.2/implementation_notes.md).
- Status: PASS

- Criterion: (8) `verification_report.md` is created.
- Evidence: [docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md).
- Status: PASS

- Criterion: (9) `cargo fmt -- --check` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (10) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (11) `cargo test --workspace` passes.
- Evidence: command executed successfully; all workspace tests passed including 9 `alloygbm-shap` tests.
- Status: PASS

- Criterion: (12) Python unittest suite passes.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passed (`Ran 58 tests`, `OK`).
- Status: PASS

## Criterion-to-Test Mapping
- Criterion: (2), (3), (4), (5)
- Tests/Checks:
  - `tests::explain_rows_from_artifact_computes_exact_expected_value_and_contributions`
  - internal runtime checks in `verify_additivity(...)`

- Criterion: (3) unused-feature behavior
- Tests/Checks:
  - `tests::explain_rows_from_artifact_assigns_zero_to_unused_features`

- Criterion: (6)
- Tests/Checks:
  - `tests::global_importance_aggregates_mean_absolute_contribution`
  - `tests::global_importance_from_artifact_uses_metadata_feature_names`

- Criterion: validation/compatibility regressions
- Tests/Checks:
  - `tests::explain_rows_from_artifact_rejects_empty_rows`
  - `tests::explain_rows_from_artifact_rejects_feature_count_mismatch`
  - `tests::explain_rows_from_artifact_rejects_non_finite_features`
  - `tests::explain_rows_from_artifact_rejects_incompatible_required_sections`

- Criterion: command gates
- Tests/Checks:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Tests Added or Updated
- File: [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs)
- Purpose: enforce exact expected-value/contribution assertions and unused-feature behavior while preserving existing validation/compatibility coverage.

## Commands Executed
- Command: `cargo test -p alloygbm-shap`
- Result: PASS (9 tests passed)
- Command: `cargo fmt -- --check`
- Result: PASS
- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS
- Command: `cargo test --workspace`
- Result: PASS
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 58 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Exact subset enumeration has exponential complexity in split-feature count and is guarded by a deterministic maximum.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: plan and execute `v0.7.3` next slice.
