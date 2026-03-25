# v0.0.8 Implementation Notes

## Summary of What Was Built
- Extended iterative training observability/policy in `crates/engine/src/lib.rs`:
  - added `IterationStopReason::DepthBudgetReached`
  - added `IterationRunSummary::effective_round_cap`
  - `Trainer::fit_iterations_with_summary(...)` now enforces an effective round cap of `min(controls.rounds, TrainParams.max_depth as usize)`.
- Added artifact compatibility policy reporting in `crates/engine/src/lib.rs`:
  - new `ArtifactCompatibilityReport` struct
  - new `TrainedModel::artifact_compatibility_report(...)`
  - internal section-shape classifier (`artifact_compatibility_report_from_sections`).
- Added deterministic auto-mode artifact import:
  - new `TrainedModel::from_artifact_bytes_auto(...) -> (TrainedModel, ArtifactCompatibilityMode)`
  - mode is selected from compatibility report (`Strict` for dual-section, `AllowLegacyTreesOnly` for legacy trees-only).
- Hardened mode-aware import guards:
  - `TrainedModel::from_artifact_bytes_with_mode(...)` now validates section-shape compatibility against requested mode before payload decode.
- Added focused engine tests for:
  - depth-budget stop reason and `effective_round_cap` behavior
  - compatibility report classification for strict/legacy/malformed required sections
  - auto-mode import selection and malformed-layout rejection.

## Non-Intuitive Decisions
- Decision: Reused `TrainParams.max_depth` as the depth/round cap in this stump-based phase instead of introducing a separate depth knob in `IterationControls`.
- Reason: Keeps depth policy aligned with existing top-level training params and avoids parallel depth controls before multi-node growth exists.
- Impact: Existing default behavior remains permissive (`max_depth=6`), while capped behavior becomes explicit and test-backed.

- Decision: Kept `TrainedModel::from_artifact_bytes(...)` default mode as `AllowLegacyTreesOnly`.
- Reason: Preserve backward compatibility from `v0.0.7` while adding stronger policy observability and explicit auto-mode import.
- Impact: Callers can adopt stricter behavior incrementally via `from_artifact_bytes_with_mode(...)` or `from_artifact_bytes_auto(...)`.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.0/v0.0.8/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate dependency-direction changes were made.
- Public `engine` API additions match the plan:
  - `IterationRunSummary.effective_round_cap`
  - `IterationStopReason::DepthBudgetReached`
  - `ArtifactCompatibilityReport`
  - `TrainedModel::artifact_compatibility_report(...)`
  - `TrainedModel::from_artifact_bytes_auto(...)`.

## Known Gaps Deferred to Next Layer
- Training remains stump/root-partition based; no multi-node depth growth yet.
- Auto-mode compatibility policy is still internal to engine import behavior and not yet exposed through Python bindings.
- Artifact payload content is still limited to current trees/predictor-layout coverage; SHAP/categorical payload content remains deferred.

## Follow-Up Actions
- Plan `v0.0.9` for next tree-structure progression beyond stump-only updates.
- Decide migration policy/timeline for strict-by-default artifact import once compatibility rollout expectations are documented.
