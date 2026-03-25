# v0.0.10 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.0/v0.0.10`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion: `cargo fmt -- --check` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: workspace unit/doc tests completed with all green.
- Status: PASS

- Criterion: Engine tests verify strict default still reports `LossImprovementBelowThreshold` with zero completed rounds when minimum improvement is not met.
- Evidence:
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason`
- Status: PASS

- Criterion: Engine tests verify non-zero weak-improvement allowance commits bounded weak rounds and then stops with `LossImprovementBelowThreshold`.
- Evidence:
  - `fit_iterations_summary_allows_bounded_weak_improvement_rounds`
- Status: PASS

- Criterion: Engine tests verify summary includes correct `weak_improvement_rounds_committed` count.
- Evidence:
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason`
  - `fit_iterations_summary_allows_bounded_weak_improvement_rounds`
  - `fit_iterations_summary_tracks_loss_trace_for_completed_rounds`
- Status: PASS

- Criterion: Existing depth-budget, loss-trace, and artifact compatibility tests remain passing.
- Evidence:
  - `fit_iterations_summary_reports_depth_budget_stop_reason`
  - `fit_iterations_summary_tracks_loss_trace_for_completed_rounds`
  - `artifact_compatibility_report_classifies_dual_section_payload`
  - `from_artifact_bytes_auto_selects_strict_for_dual_section_payload`
- Status: PASS

## Gap Analysis (Test-Gap-Closer Pass)
- Mapped each `v0.0.10` acceptance criterion in `plan.md` to direct command/test evidence.
- Added focused weak-improvement tolerance coverage and reran full verification command set.
- No uncovered criteria remained after rerun.

## Residual Uncovered Criteria
- None. All `v0.0.10` acceptance criteria have direct evidence.

## Tests Added or Updated
- File: `crates/engine/src/lib.rs`
- Purpose:
  - add bounded weak-improvement tolerance coverage
  - extend summary assertions to include weak-improvement commit counts
  - keep existing depth/loss/artifact behavior covered under regression tests.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS
  - Notable counts:
    - `alloygbm-core`: 13 tests passed
    - `alloygbm-engine`: 30 tests passed

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

## Residual Risks
- Weak-improvement tolerance can still accumulate marginal rounds under permissive settings; validation-set early stopping remains deferred.
- Tree growth remains stump-only.
- Artifact default import mode remains legacy-compatible.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: None for `v0.0.10` acceptance criteria.
