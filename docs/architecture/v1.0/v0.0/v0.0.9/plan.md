# AlloyGBM v0.0.9 Plan (v0.0 Week 9 Loss-Gated Iteration Hardening)

## Objective
Harden the iterative stump-training policy for release-candidate readiness by adding loss-improvement gating and explicit run-loss trace evidence, while preserving existing artifact compatibility behavior.

## Scope
- In scope:
  - Extend `IterationControls` with minimum loss-improvement threshold.
  - Add stop reason for loss-threshold termination in iterative summary.
  - Track loss trace in `IterationRunSummary`:
    - initial loss before first round
    - per-completed-round loss values
    - final loss.
  - Apply loss-threshold gating so rounds are rejected when improvement is below configured threshold.
  - Add focused tests for loss-threshold stop behavior and loss-trace consistency.
- Out of scope:
  - Multi-node depth expansion beyond stump/root partition approach.
  - Predictor crate integration changes.
  - Non-CPU backend behavior changes.
  - Strict-by-default artifact import policy switch.

## Deliverables
1. Iteration hardening package:
  - `IterationControls` includes `min_loss_improvement`.
  - `IterationStopReason` includes `LossImprovementBelowThreshold`.
  - Iterative loop computes and enforces minimum loss improvement before committing a round.
2. Run-loss observability package:
  - `IterationRunSummary` exposes `initial_loss` and `loss_per_completed_round`.
  - `final_loss` remains present and aligned with trace tail when rounds complete.
3. Verification package:
  - focused tests for loss-threshold gating and loss-trace bookkeeping.

## Implementation Plan
1. Add `v0.0.9` plan artifact.
2. Extend iteration control/summary/stop-reason types in `crates/engine/src/lib.rs`.
3. Refactor iterative loop to evaluate candidate-round loss improvement before committing updates.
4. Add/adjust tests for:
  - threshold-triggered stop reason with zero completed rounds
  - summary loss trace length and final-loss consistency for completed rounds.
5. Run verification commands and capture evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify iterative summary reports `LossImprovementBelowThreshold` when configured minimum improvement is not met.
5. Engine tests verify no stump round is committed when loss-threshold stop triggers before first round.
6. Engine tests verify summary loss trace:
  - starts from `initial_loss`
  - contains one value per completed round
  - final loss matches the last recorded round loss when at least one round completes.
7. Existing depth-budget and artifact compatibility tests remain passing.

## Risks and Mitigations
- Risk: loss-threshold gating changes baseline iterative behavior.
  - Mitigation: default threshold remains `0.0` to preserve current permissive behavior.
- Risk: extra loss bookkeeping causes summary inconsistency.
  - Mitigation: derive `final_loss` from tracked loss state and assert via focused tests.
- Risk: threshold checks introduce numerical sensitivity near zero.
  - Mitigation: use finite/non-negative validation and deterministic fixtures for tests.

## Exit Condition
`v0.0.9` is complete when loss-gated iteration behavior is test-backed, verification commands pass, and implementation/verification artifacts are recorded.
