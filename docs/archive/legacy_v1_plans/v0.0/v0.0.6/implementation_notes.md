# v0.0.6 Implementation Notes

## Summary of What Was Built
- Extended `IterationControls` in `crates/engine/src/lib.rs` with leaf-update policy controls:
  - `min_abs_leaf_value`
  - `max_abs_leaf_value`
- Updated iterative training behavior in `Trainer::fit_iterations_with_controls(...)`:
  - leaf values are now clamped to `max_abs_leaf_value`
  - rounds are skipped when both leaf updates are below `min_abs_leaf_value`
  - existing split-gain and leaf-size gates remain in effect
- Added legacy artifact compatibility path in `TrainedModel::from_artifact_bytes(...)`:
  - accepts legacy shape with single `Trees` section by deriving predictor layout from metadata
  - still requires `Trees`, still rejects duplicate required sections
  - rejects non-legacy multi-section artifacts that omit `PredictorLayout`
- Added focused tests in `crates/engine/src/lib.rs` for:
  - leaf-policy guard behavior and clamping
  - invalid control values
  - legacy artifact acceptance
  - strict rejection for malformed section sets

## Non-Intuitive Decisions
- Decision: Keep legacy compatibility fallback narrow to exactly one-section (`Trees`) artifacts.
- Reason: This preserves backward compatibility for the known prior internal payload shape without relaxing validation for ambiguous payloads.
- Impact: Older `v0.0.4`-style artifacts can be loaded, but malformed multi-section payloads still fail fast.

- Decision: Default `fit_iterations(rounds)` now uses permissive leaf-policy defaults (`min_abs_leaf_value=0.0`, `max_abs_leaf_value=1_000_000.0`).
- Reason: Preserve near-equivalent behavior for existing callers while enabling stricter controls via `fit_iterations_with_controls`.
- Impact: Existing tests/callers continue to work, and policy tuning is opt-in.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.0/v0.0.6/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate dependency-direction changes were made.
- Public `engine` interface evolution:
  - `IterationControls::new(...)` now includes leaf policy parameters.
  - `TrainedModel::from_artifact_bytes(...)` now includes explicit legacy-compatibility behavior.

## Known Gaps Deferred to Next Layer
- Training remains stump-level and root-partition based; multi-node depth growth is still deferred.
- Artifact compatibility behavior is still internal and version-policy decisions for external guarantees are deferred.
- Predictor/SHAP/categorical payload content is still not implemented.

## Follow-Up Actions
- Plan `v0.0.7` for deeper tree-policy progression and explicit compatibility policy documentation.
- Decide whether to expose a first-class artifact compatibility mode toggle or keep implicit strict+legacy behavior.
