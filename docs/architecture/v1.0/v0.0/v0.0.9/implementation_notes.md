# v0.0.9 Implementation Notes

## Summary of What Was Built
- Hardened iterative policy controls in `crates/engine/src/lib.rs`:
  - extended `IterationControls` with `min_loss_improvement`
  - validated the new control in both constructor and runtime control validation.
- Extended run-stop observability:
  - added `IterationStopReason::LossImprovementBelowThreshold`.
- Added loss-trace observability to `IterationRunSummary`:
  - `initial_loss`
  - `loss_per_completed_round`
  - retained `final_loss` and aligned it with tracked loss state.
- Refactored `Trainer::fit_iterations_with_summary(...)` round commit logic:
  - computes candidate round loss before committing stump updates
  - rejects rounds whose loss improvement is below configured threshold
  - records per-round loss for committed rounds.
- Added focused engine tests:
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason`
  - `fit_iterations_summary_tracks_loss_trace_for_completed_rounds`
  - updated existing summary/control tests for new fields/signature.

## Non-Intuitive Decisions
- Decision: evaluate candidate-round loss before mutating live predictions/stump list.
- Reason: ensures rejected rounds leave no partial side effects and keeps summary state deterministic.
- Impact: threshold-triggered stops now preserve model/prediction state exactly as of last committed round.

- Decision: keep default `min_loss_improvement` at `0.0` in `fit_iterations(...)`.
- Reason: preserve permissive behavior for existing callers while allowing tighter gating via explicit controls.
- Impact: behavior remains backward compatible unless callers opt into stricter thresholding.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.0/v0.0.9/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate dependency-direction changes were made.
- Public `engine` API changes match plan:
  - `IterationControls` extended with `min_loss_improvement`
  - `IterationStopReason` extended with `LossImprovementBelowThreshold`
  - `IterationRunSummary` extended with loss-trace fields.

## Known Gaps Deferred to Next Layer
- Training remains stump/root-partition based; no multi-node depth growth yet.
- Artifact import defaults remain legacy-compatible (`AllowLegacyTreesOnly`); strict-by-default migration remains pending.
- Loss gating is objective-loss based only and does not yet include validation-set early stopping policy.

## Follow-Up Actions
- Plan `v0.0.10` for next tree-structure/depth-policy progression beyond stump-only round composition.
- Decide strict-by-default artifact import migration timeline once compatibility rollout constraints are finalized.
