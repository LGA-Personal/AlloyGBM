# v0.0.9 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.0.9`
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

- Criterion: Engine tests verify iterative summary reports `LossImprovementBelowThreshold` when configured minimum improvement is not met.
- Evidence:
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason`
- Status: PASS

- Criterion: Engine tests verify no stump round is committed when loss-threshold stop triggers before first round.
- Evidence:
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason` asserts `rounds_completed == 0` and empty stump list.
- Status: PASS

- Criterion: Engine tests verify summary loss trace starts from `initial_loss`, contains one value per completed round, and final loss matches trace tail.
- Evidence:
  - `fit_iterations_summary_tracks_loss_trace_for_completed_rounds`
  - `fit_iterations_summary_reports_completed_requested_rounds`
- Status: PASS

- Criterion: Existing depth-budget and artifact compatibility tests remain passing.
- Evidence:
  - `fit_iterations_summary_reports_depth_budget_stop_reason`
  - `artifact_compatibility_report_classifies_dual_section_payload`
  - `from_artifact_bytes_auto_selects_strict_for_dual_section_payload`
- Status: PASS

## Gap Analysis (Test-Gap-Closer Pass)
- Mapped each `v0.0.9` acceptance criterion in `plan.md` to direct command/test evidence.
- Added targeted tests for the new loss-threshold and loss-trace behavior.
- Re-ran full verification command set; no uncovered criteria remained.

## Residual Uncovered Criteria
- None. All `v0.0.9` acceptance criteria have direct evidence.

## Tests Added or Updated
- File: `crates/engine/src/lib.rs`
- Purpose:
  - add loss-threshold stop coverage
  - add loss-trace bookkeeping coverage
  - update existing iteration-control tests for new control field/signature.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS
  - Notable counts:
    - `alloygbm-core`: 13 tests passed
    - `alloygbm-engine`: 29 tests passed

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

## Residual Risks
- Loss-threshold policy currently uses training loss only; validation-driven early stopping remains deferred.
- Tree structure is still stump-level; multi-node depth behavior is not yet implemented.
- Artifact default import mode remains legacy-compatible.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: None for `v0.0.9` acceptance criteria.
