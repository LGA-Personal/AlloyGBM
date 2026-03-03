# AlloyGBM v0.0.7 Plan (v0.0 Week 7 Training Policy Observability + Explicit Artifact Compatibility Mode)

## Objective
Advance the iterative training policy surface with explicit stop-reason observability and formalize artifact compatibility handling behind an explicit mode API.

## Scope
- In scope:
  - Add iterative training run summary API in `engine`:
    - rounds requested vs executed
    - deterministic stop reason for why iteration stopped
    - final training loss snapshot for the executed run
  - Preserve existing `fit_iterations` and `fit_iterations_with_controls` call patterns while implementing summary-capable execution internally.
  - Add explicit artifact compatibility mode for import in `engine`:
    - strict mode requiring explicit `PredictorLayout`
    - legacy-compatible mode allowing known `Trees`-only legacy shape
  - Add focused tests for:
    - stop reason reporting
    - strict vs legacy compatibility behavior
- Out of scope:
  - Multi-node depth expansion beyond current stump/root partition approach.
  - Public predictor crate integration changes.
  - Non-CPU backend behavior changes.

## Deliverables
1. Training observability package:
   - `fit_iterations_with_summary(...)` (or equivalent) returning run summary + model.
   - stop-reason enum capturing configured policy termination paths.
2. Artifact compatibility mode package:
   - explicit compatibility mode enum and import entrypoint using it.
   - existing default import path remains stable.
3. Verification package:
   - tests covering summary stop reasons and compatibility mode behavior.

## Implementation Plan
1. Add `v0.0.7` plan artifact.
2. Add run-summary and stop-reason types in `crates/engine/src/lib.rs`.
3. Refactor iterative loop into summary-capable path; keep existing methods delegating for compatibility.
4. Add explicit artifact compatibility mode enum and mode-aware import function.
5. Add/adjust focused engine tests for new behavior.
6. Run verification commands and capture evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. Engine tests verify iterative summary reports:
   - gain-threshold stop reason when configured to block split addition
   - completed-rounds stop reason when requested rounds execute successfully.
5. Engine tests verify artifact compatibility modes:
   - strict mode rejects legacy `Trees`-only payload
   - legacy-compatible mode accepts legacy `Trees`-only payload
   - strict mode accepts dual-section payload.
6. Existing prediction-consistency artifact roundtrip test remains passing.

## Risks and Mitigations
- Risk: additional summary surface introduces behavior drift for existing call paths.
  - Mitigation: keep existing methods as wrappers over the new summary-capable core path.
- Risk: compatibility mode branching adds ambiguity in callers.
  - Mitigation: provide explicit enum and keep default behavior unchanged and documented.
- Risk: stop-reason logic becomes inconsistent with loop behavior.
  - Mitigation: add targeted tests for each asserted reason path.

## Exit Condition
`v0.0.7` is complete when summary/compatibility-mode behaviors are test-backed, verification commands pass, and layer implementation/verification artifacts are recorded.
