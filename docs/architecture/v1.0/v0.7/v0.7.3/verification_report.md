# AlloyGBM v0.7.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.7/v0.7.3`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.7/v0.7.3/plan.md` exists and is decision-complete.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.3/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.3/plan.md) added with scope/interfaces/sequence/tests/criteria.
- Status: PASS

- Criterion: (2) SHAP additivity parity is validated against `alloygbm-predictor` outputs on artifact-backed fixtures.
- Evidence: `tests::explain_rows_from_artifact_matches_predictor_predictions` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs) reconstructs predictions from SHAP values and compares against `Predictor::predict_row(...)`.
- Status: PASS

- Criterion: (3) SHAP artifact compatibility coverage includes strict dual-section artifacts, legacy trees-only artifacts, and malformed required-section artifacts.
- Evidence:
  - strict path retained via existing fixture-based artifact tests,
  - `tests::explain_rows_from_artifact_accepts_legacy_trees_only_artifact`,
  - `tests::explain_rows_from_artifact_rejects_incompatible_required_sections`,
  - `tests::explain_rows_from_artifact_rejects_duplicate_trees_sections`,
  - `tests::explain_rows_from_artifact_rejects_metadata_feature_count_mismatch`
  in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (4) Global importance ordering is deterministic for equal-magnitude contributions.
- Evidence: `tests::global_importance_breaks_ties_by_feature_name` in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs).
- Status: PASS

- Criterion: (5) `implementation_notes.md` is created.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.3/implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.3/implementation_notes.md).
- Status: PASS

- Criterion: (6) `verification_report.md` is created.
- Evidence: [docs/architecture/v1.0/v0.7/v0.7.3/verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.7/v0.7.3/verification_report.md).
- Status: PASS

- Criterion: (7) `cargo fmt -- --check` passes.
- Evidence: command executed successfully after formatting update.
- Status: PASS

- Criterion: (8) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully.
- Status: PASS

- Criterion: (9) `cargo test --workspace` passes.
- Evidence: command executed successfully; all workspace tests passed including 14 `alloygbm-shap` tests.
- Status: PASS

- Criterion: (10) Python unittest suite passes.
- Evidence: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passed (`Ran 58 tests`, `OK`).
- Status: PASS

## Criterion-to-Test Mapping
- Criterion: (2)
- Tests/Checks:
  - `tests::explain_rows_from_artifact_matches_predictor_predictions`

- Criterion: (3)
- Tests/Checks:
  - `tests::explain_rows_from_artifact_accepts_legacy_trees_only_artifact`
  - `tests::explain_rows_from_artifact_rejects_incompatible_required_sections`
  - `tests::explain_rows_from_artifact_rejects_duplicate_trees_sections`
  - `tests::explain_rows_from_artifact_rejects_metadata_feature_count_mismatch`

- Criterion: (4)
- Tests/Checks:
  - `tests::global_importance_breaks_ties_by_feature_name`

- Criterion: regression baseline retained
- Tests/Checks:
  - `tests::explain_rows_from_artifact_computes_exact_expected_value_and_contributions`
  - `tests::explain_rows_from_artifact_assigns_zero_to_unused_features`
  - `tests::global_importance_aggregates_mean_absolute_contribution`
  - `tests::global_importance_from_artifact_uses_metadata_feature_names`

- Criterion: command gates
- Tests/Checks:
  - `cargo test -p alloygbm-shap`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Tests Added or Updated
- File: [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs)
- Added/updated tests:
  - `explain_rows_from_artifact_accepts_legacy_trees_only_artifact`
  - `explain_rows_from_artifact_rejects_duplicate_trees_sections`
  - `explain_rows_from_artifact_rejects_metadata_feature_count_mismatch`
  - `explain_rows_from_artifact_matches_predictor_predictions`
  - `global_importance_breaks_ties_by_feature_name`

## Commands Executed
- Command: `cargo test -p alloygbm-shap`
- Result: PASS (14 tests passed)
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
- Exact TreeSHAP currently relies on subset enumeration with `MAX_EXACT_SPLIT_FEATURES = 20`; very wide split-feature models return deterministic contract violations by design.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: plan and execute `v0.7.4` Python SHAP bridge layer.
