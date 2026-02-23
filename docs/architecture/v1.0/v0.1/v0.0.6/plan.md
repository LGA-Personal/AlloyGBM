# AlloyGBM v0.0.6 Plan (v0.1 Week 6 Leaf Policy Controls + Artifact Compatibility)

## Objective
Extend the `v0.0.5` iterative stump loop with explicit leaf-update policy controls and introduce backward-compatible model artifact import behavior for legacy `Trees`-only payloads.

## Scope
- In scope:
  - Add iterative leaf policy controls in `engine`:
    - minimum absolute leaf-value threshold required to keep a stump round
    - maximum absolute leaf-value clamp for numeric stability
  - Preserve existing split-gain and min-rows-per-leaf controls from `v0.0.5`.
  - Update engine artifact import to support compatibility fallback:
    - accept legacy artifacts that contain only `Trees` section (derive predictor layout from metadata)
    - keep strict rejection for missing `Trees` and duplicate required sections
    - reject ambiguous multi-section payloads that omit `PredictorLayout`
  - Add focused tests for new leaf policy and artifact compatibility behavior.
- Out of scope:
  - Full multi-node depth-controlled tree growth.
  - Predictor crate integration or public compatibility guarantees outside internal `v0.1` artifacts.
  - SHAP/categorical artifact payload implementations.

## Deliverables
1. Iterative leaf-policy package:
   - `IterationControls` extended with leaf-value policy knobs.
   - Iterative loop applies clamp and minimum-update stop guard.
2. Artifact compatibility package:
   - `TrainedModel::from_artifact_bytes` accepts legacy `Trees`-only payloads.
   - Duplicate required sections still rejected.
   - Missing `PredictorLayout` in non-legacy multi-section payloads rejected.
3. Verification package:
   - tests covering new leaf policy behavior and compatibility fallback.
   - updated verification report with criterion-to-test mapping and command evidence.

## Implementation Plan
1. Add `v0.0.6` plan artifact.
2. Extend `IterationControls` and iterative loop behavior in `crates/engine/src/lib.rs`.
3. Add artifact import section-resolution helpers to support strict + legacy compatibility paths.
4. Add/adjust engine tests for:
   - leaf policy guards
   - legacy artifact acceptance
   - strict rejection cases for malformed section sets.
5. Run verification commands and record evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify leaf-policy controls can:
   - suppress stump addition when leaf updates are below configured minimum magnitude
   - clamp emitted leaf values to configured maximum magnitude.
5. Engine tests verify artifact import:
   - accepts legacy `Trees`-only payloads
   - rejects missing `Trees`
   - rejects duplicate required sections
   - rejects multi-section payloads without `PredictorLayout`.
6. Existing dual-section artifact roundtrip prediction-consistency test remains passing.

## Risks and Mitigations
- Risk: added control knobs break existing round behavior unexpectedly.
  - Mitigation: preserve default controls to approximate prior permissive behavior.
- Risk: compatibility fallback masks malformed artifacts.
  - Mitigation: allow fallback only for strict legacy shape (single `Trees` section); keep strict rejection otherwise.
- Risk: leaf clamping affects model output deterministically but unexpectedly.
  - Mitigation: add explicit tests and document this as configured policy behavior.

## Exit Condition
`v0.0.6` is complete when leaf-policy and artifact-compatibility behaviors are test-backed, verification commands pass, and implementation/verification artifacts are recorded.
