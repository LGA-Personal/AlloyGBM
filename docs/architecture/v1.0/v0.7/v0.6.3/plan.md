# AlloyGBM v0.6.3 Plan (Engine/Artifact Categorical Integration Slice)

## Summary
- Goal: execute `v0.6.3` by integrating categorical encoding into engine training flow boundaries, persisting categorical state in model artifacts, and validating predictor replay compatibility.
- Success criteria:
  - engine exposes a categorical-aware training entrypoint that applies deterministic target encoding before iterative training,
  - `TrainedModel` artifacts persist optional categorical state using the `CategoricalState` section,
  - predictor artifact loading validates and replays models containing optional categorical state without regressing numeric behavior.
- Audience: engineers implementing `v0.7` integration work and reviewers gating readiness for Python bridge integration in `v0.6.4`.

## Scope
### In Scope
- Engine integration in `crates/engine`:
  - add categorical-aware training wrapper for single-feature target encoding in training path,
  - keep existing `fit_iterations` behavior unchanged for numeric-only paths.
- Artifact persistence:
  - include optional categorical-state section when present in `TrainedModel`,
  - decode optional categorical-state section from artifacts with feature-count validation.
- Predictor replay checks:
  - parse optional categorical-state section in predictor artifact loading,
  - verify replay parity remains intact for strict artifacts carrying categorical state.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.7/v0.6.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.7/v0.6.3/verification_report.md`

### Out of Scope
- End-to-end Python/API categorical controls (`v0.6.4` scope).
- Multi-feature categorical preprocessing orchestration and full categorical pipeline wiring across training/inference entrypoints.
- Categorical model-quality optimization work beyond deterministic integration correctness.
- Model format version changes.

## Interfaces and Types
- `crates/engine/src/lib.rs`:
  - `CategoricalTargetEncodingSpec`,
  - `Trainer::fit_iterations_with_single_target_encoded_feature`,
  - `TrainedModel` optional `categorical_state` persistence.
- `crates/predictor/src/lib.rs`:
  - predictor artifact decode path for optional categorical-state section.
- `crates/engine/Cargo.toml`:
  - dependency on `alloygbm-categorical`.

Backward-compatibility expectations:
- numeric-only engine/predictor behavior remains unchanged,
- strict/legacy required-section compatibility policy from `v0.6` remains intact,
- categorical-state section remains optional.

## Deliverables
1. Engine integration package:
  - categorical-aware training wrapper and deterministic preprocessing helpers.
2. Artifact package:
  - optional categorical-state serialization/deserialization through engine model artifacts.
3. Predictor package:
  - optional categorical-state decode support with replay parity tests.
4. Verification package:
  - criterion-mapped verification report with command evidence.
5. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.6.3` completion and `v0.6.4` next-target suggestion.

## Implementation Sequence
1. Add `v0.6.3` plan and lock integration boundaries.
2. Add engine dependency on `alloygbm-categorical`.
3. Implement categorical target-encoding training wrapper in engine with deterministic bin mapping.
4. Extend `TrainedModel` artifact serialization/deserialization for optional categorical state.
5. Extend predictor artifact decode path to parse optional categorical state.
6. Add/update tests for:
  - engine categorical wrapper behavior and state attachment,
  - artifact roundtrip with optional categorical state,
  - predictor replay parity with categorical-state artifacts.
7. Run targeted + full verification gates.
8. Write `implementation_notes.md` and `verification_report.md`.
9. Update `layer_index.yaml` to mark `v0.6.3` verified and set `v0.6.4` as next target.

## Test Cases and Scenarios
- Unit cases:
  - categorical wrapper validates feature index/value lengths,
  - deterministic encoded-bin mapping from encoded floats.
- Integration cases:
  - engine training with single categorical spec returns model with attached categorical state,
  - artifact roundtrip preserves optional categorical state.
- Replay/compatibility cases:
  - predictor parses artifacts containing optional `CategoricalState`,
  - engine vs predictor prediction parity remains unchanged for such artifacts.
- Acceptance test mapping:
  - `cargo test -p alloygbm-engine`,
  - `cargo test -p alloygbm-predictor`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. Engine provides categorical-aware training wrapper and keeps numeric-only path behavior unchanged.
2. `TrainedModel` artifact serialization includes optional categorical state when present.
3. Engine artifact decode restores optional categorical state with validation.
4. Predictor artifact decode accepts optional categorical state without replay regression.
5. `docs/architecture/v1.0/v0.7/v0.6.3/implementation_notes.md` is created.
6. `docs/architecture/v1.0/v0.7/v0.6.3/verification_report.md` is created.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. `cargo doc --workspace --no-deps` passes.
11. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: preprocessing wrapper semantics diverge from future multi-feature pipeline.
  - Mitigation: keep wrapper explicit/single-feature and document scope boundaries.
- Risk: optional categorical section introduces artifact compatibility regressions.
  - Mitigation: preserve required-section policy unchanged and add optional-section parity tests.
- Risk: encoded-bin mapping assumptions hurt future model-quality tuning.
  - Mitigation: treat current mapping as deterministic integration baseline and defer tuning to follow-on layers.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.6.3` focuses on Rust engine/predictor integration; Python categorical entrypoints remain deferred.
- Categorical-state artifacts are optional and additive.
