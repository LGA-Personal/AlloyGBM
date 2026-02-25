# AlloyGBM v0.0.10 Plan (v0.1 Week 10 Weak-Improvement Tolerance Policy)

## Objective
Refine the `v0.0.9` loss-gated iterative policy to allow a bounded number of weak-but-non-worsening rounds before stopping, improving training-policy flexibility while preserving deterministic safeguards.

## Scope
- In scope:
  - Extend `IterationControls` with `max_consecutive_weak_improvements`.
  - Define weak improvement as: `0.0 <= loss_improvement < min_loss_improvement`.
  - Allow committing weak-improvement rounds up to configured consecutive bound.
  - Stop with `LossImprovementBelowThreshold` once weak-improvement streak exceeds allowed bound.
  - Add summary observability field for committed weak-improvement rounds.
  - Add focused tests for tolerance behavior and default strict behavior.
- Out of scope:
  - Multi-node depth expansion beyond stump/root partition approach.
  - Validation-set early stopping.
  - Predictor crate integration changes.
  - Artifact compatibility default-mode migration changes.

## Deliverables
1. Weak-improvement tolerance package:
  - `IterationControls.max_consecutive_weak_improvements` (default strict = `0`).
  - iterative loop tracks consecutive weak improvements and applies bounded tolerance.
2. Summary observability package:
  - `IterationRunSummary.weak_improvement_rounds_committed`.
3. Verification package:
  - tests proving weak-improvement tolerance allows bounded progress.
  - tests proving strict default still stops before first weak-improvement commit.

## Implementation Plan
1. Add `v0.0.10` plan artifact.
2. Extend iteration control and summary types in `crates/engine/src/lib.rs`.
3. Update iterative loop to track weak-improvement streak and committed weak-improvement count.
4. Add/adjust focused tests for:
  - strict default behavior (`max_consecutive_weak_improvements = 0`)
  - bounded weak-improvement allowance (`> 0`) and subsequent stop.
5. Run verification commands and capture evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify strict default still reports `LossImprovementBelowThreshold` with zero completed rounds when minimum improvement is not met.
5. Engine tests verify non-zero weak-improvement allowance commits bounded weak rounds and then stops with `LossImprovementBelowThreshold`.
6. Engine tests verify summary includes correct `weak_improvement_rounds_committed` count.
7. Existing depth-budget, loss-trace, and artifact compatibility tests remain passing.

## Risks and Mitigations
- Risk: tolerance could permit loss-worsening rounds.
  - Mitigation: only non-negative weak improvements are tolerated; loss-worsening remains a stop path.
- Risk: additional policy state introduces summary inconsistencies.
  - Mitigation: add targeted assertions for weak-round counts and stop behavior.
- Risk: defaults drift from current strict behavior.
  - Mitigation: set default tolerance to `0` in `fit_iterations(...)`.

## Exit Condition
`v0.0.10` is complete when weak-improvement tolerance behavior is test-backed, verification commands pass, and implementation/verification artifacts are recorded.
