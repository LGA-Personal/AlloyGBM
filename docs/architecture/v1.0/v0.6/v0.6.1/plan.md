# AlloyGBM v0.6.1 Plan (v0.6 Categorical Contract Baseline Slice)

## Summary
- Goal: execute the first `v0.6` child slice by locking categorical-state artifact contract, schema validation invariants, and serialization expectations before broader categorical pipeline implementation.
- Success criteria:
  - categorical-state payload shape is explicit and test-backed,
  - categorical dataset/schema validation rules are deterministic and enforced in `core`,
  - strict/legacy required-section compatibility remains unchanged when optional categorical state is present.
- Audience: engineers implementing `v0.6` categorical work and reviewers gating readiness for encoder/runtime integration in `v0.6.2+`.

## Scope
### In Scope
- Contract baseline in `crates/core`:
  - define categorical-state payload format for `ModelSectionKind::CategoricalState`,
  - add deterministic encode/decode + validation helpers for the payload.
- Schema rule hardening:
  - enforce canonical categorical feature index invariants (in-bounds, strictly increasing, no duplicates).
- Serialization expectations:
  - ensure artifacts that include optional categorical-state section remain valid under v1 contract and required-section compatibility checks.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.6/v0.6.1/implementation_notes.md`
  - `docs/architecture/v1.0/v0.6/v0.6.1/verification_report.md`

### Out of Scope
- End-to-end target/frequency encoder implementation in `crates/categorical` (`v0.6.2+` scope).
- Engine training-path preprocessing integration for categorical features (`v0.6.3+` scope).
- Python API and bridge expansion for categorical runtime behavior (`v0.6.4+` scope).
- Model format version bump beyond v1.
- SHAP/ranking/GPU work.

## Interfaces and Types
- `crates/core/src/lib.rs`:
  - categorical-state payload type(s) and encode/decode helpers,
  - categorical schema validation logic.
- Existing compatibility interfaces remain policy anchors:
  - `required_section_compatibility_report`,
  - `serialize_model_artifact_v1`,
  - `deserialize_model_artifact_v1`.

Backward-compatibility expectations:
- strict required-section mode still means exactly one `Trees` and one `PredictorLayout` section (optional sections allowed),
- legacy trees-only compatibility behavior remains unchanged,
- no public Python API changes in this slice.

## Deliverables
1. Contract package:
  - explicit categorical-state payload format and helper APIs in `core`.
2. Validation package:
  - tightened dataset schema validation for categorical indices.
3. Test package:
  - unit tests for payload encode/decode/validation and schema invariants.
4. Verification package:
  - command evidence and criterion mapping in `verification_report.md`.
5. State package:
  - `docs/architecture/state/layer_index.yaml` updated for `v0.6.1` completion and `v0.6.2` next-target suggestion.

## Implementation Sequence
1. Add `v0.6.1` plan and lock categorical-state contract boundaries.
2. Implement categorical-state payload helpers in `crates/core/src/lib.rs`.
3. Tighten categorical schema validation invariants in `validate_dataset_schema`.
4. Add core tests that cover:
  - payload roundtrip and malformed payload failures,
  - schema duplicate/ordering violations,
  - required-section compatibility with optional categorical-state section.
5. Run targeted + full verification gates.
6. Write `implementation_notes.md` and `verification_report.md`.
7. Update `layer_index.yaml` to mark `v0.6.1` verified and set `v0.6.2` as next target.

## Test Cases and Scenarios
- Unit cases:
  - categorical-state payload encode/decode roundtrip,
  - decode rejects truncated/invalid payload versions and malformed feature-index lists,
  - schema validation rejects duplicate or non-ascending categorical feature indices.
- Integration cases:
  - model artifact serialize/deserialize remains valid when including `CategoricalState` section.
- Failure and edge cases:
  - validation rejects out-of-bounds categorical feature indices in payload vs feature count,
  - required-section compatibility remains strict-compatible for artifacts that include optional categorical section.
- Acceptance test mapping:
  - `cargo test -p alloygbm-core`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. `v0.6.1` artifacts explicitly define categorical-state section payload contract for v1 artifacts.
2. `core` provides deterministic categorical-state payload encode/decode + validation helpers.
3. Dataset schema validation enforces canonical categorical index invariants (strictly increasing, in-bounds).
4. Artifact compatibility report behavior for strict/legacy required sections remains unchanged with optional categorical section present.
5. `docs/architecture/v1.0/v0.6/v0.6.1/implementation_notes.md` is created.
6. `docs/architecture/v1.0/v0.6/v0.6.1/verification_report.md` is created.
7. `cargo fmt -- --check` passes.
8. `cargo clippy --workspace --all-targets -- -D warnings` passes.
9. `cargo test --workspace` passes.
10. `cargo doc --workspace --no-deps` passes.
11. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: contract defined in this slice drifts from future engine/predictor use.
  - Mitigation: centralize payload helpers in `core` and lock behavior with tests.
- Risk: stricter schema validation could break assumptions in future callers.
  - Mitigation: keep constraints minimal and explicit (bounds + ascending uniqueness), document errors clearly.
- Risk: optional categorical section accidentally changes strict/legacy required-section semantics.
  - Mitigation: add compatibility-report regression test with optional section present.

## Assumptions and Defaults
- Device scope remains CPU-only.
- `v0.6.1` is contract/validation baseline only; full categorical feature execution is deferred to later child slices.
- Categorical-state section is optional for non-categorical models.
