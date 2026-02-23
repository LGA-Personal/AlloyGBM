# AlloyGBM v0.0.4 Plan (v0.1 Week 4 Iterative Loop + Initial Model IO)

## Objective
Extend the `v0.0.3` executable slice into a minimal iterative training loop and add first end-to-end model artifact emission/parsing contracts.

## Scope
- In scope:
  - Iterative multi-round training in `engine` built on existing one-round primitives.
  - Minimal trained-model representation with row/batch prediction helpers in `engine`.
  - Initial model artifact binary writer/reader functions in `core`.
  - Engine-level export/import using the core model artifact contracts.
  - Focused tests for iterative behavior and artifact roundtrips.
- Out of scope:
  - Full production GBDT tree-growth policy (depth control, pruning, early stop).
  - Predictor crate production integration.
  - Cross-version model compatibility policy finalization.
  - Performance tuning.

## Deliverables
1. Iterative engine package:
   - `Trainer` supports configurable round count for repeated split updates.
   - trained model object captures baseline + learned stumps.
   - prediction methods for single-row and batch inputs.
2. Core model artifact package:
   - serialize model artifact bytes from metadata + binary sections.
   - parse artifact bytes back into metadata + section payloads.
3. End-to-end glue:
   - engine can export trained model to artifact bytes and reconstruct from bytes.
   - tests prove training -> export -> import -> prediction consistency.

## Implementation Plan
1. Add `v0.0.4` plan artifact.
2. Add core artifact serialization/deserialization utilities and tests.
3. Add iterative trainer/model structures in engine and prediction helpers.
4. Add engine export/import via core artifact utilities.
5. Add/extend tests for iterative rounds, contract guards, and artifact roundtrips.
6. Run verification suite and capture evidence in layer report.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Core tests verify model artifact serialize/deserialize roundtrip and malformed-input rejection.
5. Engine tests verify multi-round training produces at least one stump and non-trivial prediction updates on deterministic fixtures.
6. Engine tests verify exported artifact bytes can be reloaded with prediction consistency.

## Risks and Mitigations
- Risk: loop behavior drifts into scope meant for later milestones.
  - Mitigation: restrict to stump-level rounds and deterministic fixtures only.
- Risk: brittle binary parsing.
  - Mitigation: add strict length checks and malformed-input tests.
- Risk: over-coupling artifact layout to temporary model representation.
  - Mitigation: keep section format compact and versioned under v1 contract constants.

## Exit Condition
`v0.0.4` is complete when iterative training and model artifact roundtrips are test-backed, verification commands pass, and layer implementation/verification artifacts are recorded.
