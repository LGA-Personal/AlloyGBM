# v0.0.7 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.0.7`
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

- Criterion: Engine tests verify iterative summary reports gain-threshold stop and completed-rounds stop.
- Evidence:
  - `fit_iterations_summary_reports_gain_threshold_stop_reason`
  - `fit_iterations_summary_reports_completed_requested_rounds`
- Status: PASS

- Criterion: Engine tests verify artifact compatibility modes for strict vs legacy behavior.
- Evidence:
  - `strict_mode_rejects_legacy_trees_only_payload`
  - `trained_model_artifact_accepts_legacy_trees_only_payload`
  - `strict_mode_accepts_dual_section_payload`
- Status: PASS

- Criterion: Existing prediction-consistency artifact roundtrip remains passing.
- Evidence:
  - `trained_model_artifact_roundtrip_preserves_predictions`
- Status: PASS

## Gap Analysis (Test-Gap-Closer Pass)
- Mapped each `v0.0.7` acceptance criterion in `plan.md` to direct command/test evidence.
- Added focused tests for the two new surfaces:
  - iteration stop-reason summary
  - explicit compatibility mode behavior.
- Re-ran the full verification command set in this gap-closure pass; no missing-test or missing-run gaps remained.

## Residual Uncovered Criteria
- None. All `v0.0.7` acceptance criteria have direct evidence.

## Tests Added or Updated
- File: `crates/engine/src/lib.rs`
- Purpose:
  - add summary stop-reason evidence
  - add strict/legacy compatibility mode evidence.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS
  - Notable counts:
    - `alloygbm-core`: 13 tests passed
    - `alloygbm-engine`: 20 tests passed

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

## Gap-Closure Outcome
- Additional tests required in this pass: none (existing focused tests already satisfied all criteria).
- Residual uncovered criteria after rerun: none.

## Residual Risks
- Training policy remains stump-level and does not yet capture multi-node depth logic.
- Compatibility default remains legacy-friendly; strict-only rollout policy is still deferred.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: None for `v0.0.7` acceptance criteria.
