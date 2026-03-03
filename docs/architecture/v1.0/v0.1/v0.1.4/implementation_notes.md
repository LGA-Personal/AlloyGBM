# v0.1.4 Implementation Notes

## Summary of What Was Built
- Implemented `v0.1.4` as a validation early-stopping semantics slice for `v0.1`.
- Updated `crates/engine/src/lib.rs` iterative training finalization to rollback model state to `best_validation_round` when stop reason is `ValidationLossPlateau`.
- Added rollback alignment for summary fields so returned outputs match the retained best checkpoint:
  - `rounds_completed`
  - `loss_per_completed_round`
  - `validation_loss_per_completed_round`
  - `sampled_rows_per_completed_round`
  - `sampled_features_per_completed_round`
  - `final_loss`
  - `final_validation_loss`
- Updated plateau-stop unit test expectations to assert best-checkpoint semantics instead of retaining plateau-triggering round state.

## Non-Intuitive Decisions
- Decision: perform rollback after loop completion rather than inside the round body.
- Reason: keeps the commit path simple and avoids duplicating per-round bookkeeping logic.
- Impact: plateau stop still captures stop reason from the triggering round, but returned model/summary now reflect best-validation checkpoint state.

- Decision: keep `best_validation_round` anchored to `0` when no committed round improves validation beyond threshold.
- Reason: aligns with explicit baseline-as-checkpoint semantics and allows deterministic zero-round rollback behavior.
- Impact: plateau-stop scenarios with strict improvement threshold can now return an empty stump list with baseline losses intact.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.1.4/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate-boundary or public API changes.
- Scope remained within engine behavior semantics and layer artifacts.

## Known Gaps Deferred to Next Layer
- Training is still stump-level iterative boosting and does not yet implement full depth-limited tree structure growth for broader `v0.1` completion.
- Parent rollup artifacts remain pending:
  - `docs/architecture/v1.0/v0.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.1/verification_report.md`

## Follow-Up Actions
- Define `v0.1.5` around remaining `v0.1` algorithm depth/behavior gaps while preserving deterministic and validation-checkpoint semantics.
