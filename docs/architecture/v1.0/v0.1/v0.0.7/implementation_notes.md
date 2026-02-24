# v0.0.7 Implementation Notes

## Summary of What Was Built
- Added iterative run-observability types in `crates/engine/src/lib.rs`:
  - `IterationStopReason`
  - `IterationRunSummary`
- Added `Trainer::fit_iterations_with_summary(...)` and refactored `fit_iterations_with_controls(...)` to delegate to this summary-capable path.
- Added explicit artifact compatibility mode surface in `crates/engine/src/lib.rs`:
  - `ArtifactCompatibilityMode` enum (`Strict`, `AllowLegacyTreesOnly`)
  - `TrainedModel::from_artifact_bytes_with_mode(...)`
  - `TrainedModel::from_artifact_bytes(...)` now explicitly defaults to `AllowLegacyTreesOnly`.
- Added focused engine tests for `v0.0.7` behaviors:
  - summary stop-reason reporting for gain-threshold and completed-rounds paths
  - strict mode rejection of legacy trees-only artifacts
  - strict mode acceptance of dual-section artifacts

## Non-Intuitive Decisions
- Decision: Keep stop-reason reporting in a new summary method rather than changing existing return types.
- Reason: Preserves current call sites while adding observability for policy-driven stop behavior.
- Impact: Existing APIs remain source-compatible; richer observability is opt-in.

- Decision: Keep default import compatibility mode as `AllowLegacyTreesOnly`.
- Reason: Avoid behavioral break for existing callers relying on `from_artifact_bytes(...)`.
- Impact: Explicit strict mode is available for callers that want hard section requirements.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.0.7/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate dependency-direction changes were made.
- Public `engine` API additions align with plan:
  - iteration summary surface
  - explicit artifact compatibility mode import surface.

## Known Gaps Deferred to Next Layer
- Training remains stump/root-partition based; no multi-node depth expansion yet.
- Compatibility mode semantics are still internal-policy oriented; broader version policy remains deferred.
- Predictor/SHAP/categorical payload expansion remains out of this layer.

## Follow-Up Actions
- Plan `v0.0.8` for next tree-policy depth progression and compatibility policy hardening docs.
- Decide whether strict mode should become the default once compatibility migration path is documented.
