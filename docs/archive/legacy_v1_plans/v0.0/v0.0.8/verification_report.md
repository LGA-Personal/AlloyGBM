# v0.0.8 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.0/v0.0.8`
- Date: 2026-02-24

## Acceptance Criteria Matrix
- Criterion: `cargo fmt -- --check` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: workspace unit/doc test run completed with all green.
- Status: PASS

- Criterion: Engine tests verify iteration summary reports depth-budget stop when requested rounds exceed `TrainParams.max_depth`.
- Evidence:
  - `fit_iterations_summary_reports_depth_budget_stop_reason`
- Status: PASS

- Criterion: Engine tests verify run summary reports `effective_round_cap` for capped and uncapped runs.
- Evidence:
  - `fit_iterations_summary_reports_gain_threshold_stop_reason`
  - `fit_iterations_summary_reports_completed_requested_rounds`
  - `fit_iterations_summary_reports_depth_budget_stop_reason`
- Status: PASS

- Criterion: Engine tests verify compatibility report classification for strict dual-section, legacy trees-only, and malformed duplicate required sections.
- Evidence:
  - `artifact_compatibility_report_classifies_dual_section_payload`
  - `artifact_compatibility_report_classifies_legacy_trees_only_payload`
  - `artifact_compatibility_report_marks_malformed_required_sections_incompatible`
- Status: PASS

- Criterion: Engine tests verify `from_artifact_bytes_auto(...)` mode selection and malformed-layout rejection.
- Evidence:
  - `from_artifact_bytes_auto_selects_strict_for_dual_section_payload`
  - `from_artifact_bytes_auto_selects_legacy_for_trees_only_payload`
  - `from_artifact_bytes_auto_rejects_malformed_required_section_layouts`
- Status: PASS

- Criterion: Existing prediction-consistency artifact roundtrip test remains passing.
- Evidence:
  - `trained_model_artifact_roundtrip_preserves_predictions`
- Status: PASS

## Gap Analysis (Test-Gap-Closer Pass)
- Mapped each `v0.0.8` acceptance criterion in `plan.md` to direct command or test evidence.
- Added focused tests for the new depth-budget and compatibility-report/auto-mode surfaces.
- Re-ran the full verification command set after implementation; no uncovered criteria remained.
- Re-ran the verification command set again during this explicit `alloy-test-gap-closer` pass; results remained green with identical criterion coverage.

## Residual Uncovered Criteria
- None. All `v0.0.8` acceptance criteria have direct evidence.

## Tests Added or Updated
- File: `crates/engine/src/lib.rs`
- Purpose:
  - verify depth-budget stop reason and effective round cap behavior
  - verify compatibility report classification and auto-mode artifact import behavior.
- Additional tests required in this gap-closure rerun: none.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS
  - Notable counts:
    - `alloygbm-core`: 13 tests passed
    - `alloygbm-engine`: 27 tests passed

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

## Residual Risks
- Depth-budget behavior currently maps to stump rounds and does not represent full multi-node tree depth growth semantics.
- Default import mode remains legacy-compatible; strict-by-default migration policy is still undecided.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: None for `v0.0.8` acceptance criteria.
