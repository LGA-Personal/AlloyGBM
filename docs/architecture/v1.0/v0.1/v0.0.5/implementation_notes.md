# v0.0.5 Implementation Notes

## Summary of What Was Built
- Added iterative training controls in `crates/engine/src/lib.rs`:
  - new `IterationControls` contract (`rounds`, `min_split_gain`, `min_rows_per_leaf`)
  - new `Trainer::fit_iterations_with_controls(...)`
  - existing `Trainer::fit_iterations(...)` now delegates to default controls for backward-compatible behavior.
- Extended engine artifact export/import coverage in `crates/engine/src/lib.rs`:
  - `TrainedModel::to_artifact_bytes(...)` now emits both `Trees` and `PredictorLayout` sections.
  - `TrainedModel::from_artifact_bytes(...)` now requires and validates both sections.
  - added duplicate required-section rejection and section-consistency checks.
  - added predictor-layout payload encode/decode helpers.
- Added focused engine tests for new behavior:
  - min-gain and min-leaf-row controls
  - invalid control rejection
  - missing/duplicate required artifact section rejection
  - dual-section artifact presence assertion in roundtrip.
- Applied the separate naming/path drift fix from `v0.0.4` drift report:
  - updated stale `v0.0.3` docstrings in Python module/test files.

## Non-Intuitive Decisions
- Decision: Keep `IterationControls` as a separate contract while preserving `fit_iterations(rounds)` compatibility.
- Reason: Existing tests/callers can keep the simpler API while `v0.0.5` introduces bounded controls without forced migration.
- Impact: Lower integration risk and cleaner staged adoption for future layers.

- Decision: Require both `Trees` and `PredictorLayout` sections on import.
- Reason: `v0.0.5` explicitly broadens artifact section coverage and adds stricter contract validation to avoid ambiguous partial payloads.
- Impact: Older single-section artifacts are rejected by engine import in this layer, but contract strictness is now test-backed and explicit.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.0.5/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No crate dependency boundary changes were made.
- Planned interface evolution occurred in-place:
  - `engine`: added `IterationControls`, `fit_iterations_with_controls`, stricter section validation.
  - Python bindings: docstring-only naming cleanup (no runtime behavior changes).

## Known Gaps Deferred to Next Layer
- Iterative training remains root-node stump updates; no multi-node depth growth policy yet.
- Artifact payload still does not include concrete SHAP/categorical state payloads.
- Python regressor remains baseline-only and is not yet connected to native engine artifact loading.

## Follow-Up Actions
- Plan `v0.0.6` for deeper tree-growth policy expansion and artifact compatibility strategy for evolving section sets.
- Decide whether to support compatibility fallback for legacy single-section internal artifacts or keep strict v0.0.5+ enforcement.
