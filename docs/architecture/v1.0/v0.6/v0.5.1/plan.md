# AlloyGBM v0.5.1 Plan (v0.6 Compatibility Policy Baseline Slice)

## Summary
- Goal: execute the first `v0.6` child slice by locking model-format v1 compatibility policy and adding baseline IO contract tests for predictor ingestion behavior.
- Success criteria:
  - compatibility policy for strict dual-section artifacts and legacy trees-only artifacts is explicit and test-backed,
  - predictor artifact parsing rejects malformed required-section layouts deterministically,
  - workspace and Python verification gates remain green with no public API drift.
- Audience: engineers implementing `v0.6` compatibility hardening and reviewers validating readiness for deeper predictor-path canonicalization in `v0.5.2+`.

## Scope
### In Scope
- Define and lock compatibility-policy baseline for this parent milestone:
  - strict artifact shape: exactly one `Trees` section + exactly one `PredictorLayout` section,
  - legacy compatibility shape: trees-only payload accepted in compatibility path.
- Add baseline predictor contract tests for malformed artifact section layouts:
  - duplicate required sections,
  - missing required section combinations that are non-legacy.
- Preserve existing engine/predictor prediction parity checks and artifact roundtrip checks as non-regression gates.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.6/v0.5.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.6/v0.5.1/verification_report.md`

### Out of Scope
- Python API surface redesign or new public estimator methods.
- Model format version bump beyond v1.
- Predictor traversal/performance optimization work (`v0.5.2+` scope).
- SHAP/categorical/ranking/GPU expansion.

## Interfaces and Types
- `crates/predictor/src/lib.rs`:
  - `Predictor::from_artifact_bytes` required-section validation behavior and contract tests.
- `crates/engine/src/lib.rs`:
  - existing compatibility semantics remain the policy reference for strict and legacy modes (no API-breaking changes in this slice).
- `crates/core/src/lib.rs`:
  - `ModelSectionKind` and artifact descriptor semantics used by new malformed-artifact fixtures.

Backward-compatibility expectations:
- preserve existing acceptance of valid strict and legacy artifact payloads,
- preserve `GBMRegressor` Python behavior and native bridge contracts.

## Deliverables
1. Policy baseline package:
  - explicit `v0.5.1` documentation of compatibility policy in this plan and linked implementation/verification artifacts.
2. Test hardening package:
  - new predictor tests covering malformed section-layout rejection cases.
3. Verification package:
  - command evidence for workspace + Python gates,
  - layer `implementation_notes.md` and `verification_report.md`.
4. State package:
  - `docs/architecture/state/layer_index.yaml` update after verification.

## Implementation Sequence
1. Add `v0.5.1` plan and codify strict vs legacy artifact policy boundaries.
2. Add predictor-focused malformed artifact tests for required-section invariants.
3. Run targeted predictor test command, then full verification gates.
4. Write `implementation_notes.md` with decisions and deferred gaps.
5. Write `verification_report.md` with acceptance-criteria evidence mapping.
6. Update `docs/architecture/state/layer_index.yaml` for `v0.5.1` completion and next target suggestion.

## Test Cases and Scenarios
- Unit cases:
  - predictor rejects duplicate required sections,
  - predictor rejects non-legacy artifacts missing `PredictorLayout`,
  - predictor rejects artifacts missing `Trees`.
- Integration cases:
  - existing engine-vs-predictor prediction parity tests remain passing for strict and legacy payloads.
- Failure and edge cases:
  - malformed section-count layouts produce deterministic contract violations rather than silent fallback.
- Acceptance test mapping:
  - `cargo test -p alloygbm-predictor`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `v0.5.1` artifacts explicitly lock compatibility-policy baseline for strict and legacy model-format v1 payloads.
2. Predictor tests cover malformed required-section layouts (duplicate and missing required sections) and fail deterministically.
3. Existing strict/legacy artifact success-path tests remain passing without behavior regression.
4. No public API surface changes are introduced in this slice.
5. `docs/architecture/v1.0/v0.6/v0.5.1/implementation_notes.md` is created with implementation rationale and deferred gaps.
6. `docs/architecture/v1.0/v0.6/v0.5.1/verification_report.md` is created with criterion-to-evidence mapping.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. `cargo doc --workspace --no-deps` passes.
11. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: compatibility semantics diverge between `engine` and `predictor`.
  - Mitigation: keep strict/legacy parity tests and malformed-layout rejection tests in predictor aligned to engine artifact semantics.
- Risk: malformed payload handling regresses into permissive fallback.
  - Mitigation: lock rejection behavior with explicit tests for duplicate/missing required sections.
- Risk: scope creep into broader predictor integration work.
  - Mitigation: confine this slice to policy baseline and test hardening only.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.5.1` is baseline-policy and test-hardening only; deeper canonicalization work moves to `v0.5.2+`.
- Verification uses standard workspace gate commands plus Python unit suite.
