# AlloyGBM v0.5.3 Plan (Serialization Hardening + Failure-Mode Consistency Slice)

## Summary
- Goal: execute `v0.5.3` by hardening model-artifact serialization contract checks and making malformed-artifact failure behavior consistent across `core`, `engine`, and `predictor` ingestion paths.
- Success criteria:
  - artifact descriptor offsets are validated against payload-start and contiguous-layout invariants,
  - malformed required-section layouts yield deterministic compatibility-mode failures in both engine and predictor compatibility paths,
  - migration guidance is recorded for legacy trees-only and malformed artifacts.
- Audience: engineers completing `v0.5` model IO hardening before parent closeout.

## Scope
### In Scope
- Contract hardening in `crates/core/src/lib.rs`:
  - enforce payload-start bound for section offsets,
  - enforce contiguous ordered section offsets (no gaps/overlap) for v1 artifact contract.
- Failure-mode consistency:
  - add shared required-section compatibility report/error formatting helpers in `core`,
  - use shared helpers in `engine` and `predictor` compatibility gates so malformed required-section layouts report aligned deterministic errors.
- Test hardening:
  - add core tests for payload-start and non-contiguous descriptor rejection,
  - add engine/predictor tests that assert deterministic compatibility-mode errors for malformed required-section layouts.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.5/v0.5.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.5/v0.5.3/verification_report.md`

### Out of Scope
- Model format version bump beyond v1.
- Public Python API redesign.
- Predictor traversal/performance optimization.
- Parent `v0.5` rollup artifacts (`docs/architecture/v1.0/v0.5/implementation_notes.md`, `verification_report.md`).

## Interfaces and Types
- `crates/core/src/lib.rs`:
  - `validate_model_contract_v1` section-layout invariants,
  - shared compatibility reporting for required sections.
- `crates/engine/src/lib.rs`:
  - consume shared compatibility helpers for `from_artifact_bytes_with_mode` and `from_artifact_bytes_auto` error stability.
- `crates/predictor/src/lib.rs`:
  - add required-section compatibility gate prior to section extraction to align compatibility failures with engine path.

Backward-compatibility expectations:
- strict dual-section artifacts remain accepted,
- legacy trees-only artifacts remain accepted only on compatibility path,
- malformed required-section combinations remain rejected, now with consistent compatibility-mode diagnostics.

## Deliverables
1. Core contract-hardening package:
  - payload-start + contiguous-offset validation in `validate_model_contract_v1`.
2. Shared compatibility package:
  - core helper(s) used by engine and predictor for deterministic required-section compatibility diagnostics.
3. Test package:
  - new/updated unit tests in `core`, `engine`, and `predictor` for invariant and failure-mode behavior.
4. Verification package:
  - full gate evidence and layer artifacts.
5. State package:
  - `docs/architecture/state/layer_index.yaml` update after verification.

## Implementation Sequence
1. Add `v0.5.3` plan and lock invariants/acceptance criteria.
2. Implement core contract hardening and shared compatibility helper(s).
3. Wire helper usage into engine and predictor artifact ingest paths.
4. Add/update tests for invariants and failure-mode consistency.
5. Run targeted tests.
6. Run full verification gates and capture evidence.
7. Write `implementation_notes.md` and `verification_report.md`.
8. Update `layer_index.yaml` to mark `v0.5.3` complete and choose next target.

## Test Cases and Scenarios
- Unit cases:
  - reject section offset before payload start,
  - reject non-contiguous ordered sections,
  - compatibility gate reports deterministic section-count diagnostics for malformed required-section layouts.
- Integration cases:
  - strict and legacy success-path artifacts remain loadable,
  - predictor and engine both reject malformed required-section layouts with aligned diagnostics.
- Failure and edge cases:
  - duplicate/missing required sections continue to fail,
  - auto-mode selection failure reports deterministic section-count context.
- Acceptance test mapping:
  - `cargo test -p alloygbm-core`,
  - `cargo test -p alloygbm-engine`,
  - `cargo test -p alloygbm-predictor`,
  - `cargo test -p alloygbm-python`,
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Acceptance Criteria
1. Core artifact validation rejects section offsets that precede payload start.
2. Core artifact validation rejects non-contiguous section offsets in v1 artifact descriptor layout.
3. Engine compatibility-mode failures for malformed required-section layouts use deterministic section-count diagnostics.
4. Predictor compatibility-mode failures for malformed required-section layouts use deterministic section-count diagnostics aligned with engine behavior.
5. Existing strict and legacy success-path artifact ingestion remains green in engine and predictor tests.
6. `docs/architecture/v1.0/v0.5/v0.5.3/implementation_notes.md` is created.
7. `docs/architecture/v1.0/v0.5/v0.5.3/verification_report.md` is created.
8. `cargo fmt -- --check` passes.
9. `cargo clippy --workspace --all-targets -- -D warnings` passes.
10. `cargo test --workspace` passes.
11. `cargo doc --workspace --no-deps` passes.
12. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: strict offset contiguity rejects previously tolerated externally produced artifacts.
  - Mitigation: keep legacy support focused on required sections, and document contiguous-layout requirement as v1 contract behavior.
- Risk: compatibility failure messages drift again across crates.
  - Mitigation: centralize message formatting in `core` and consume from both engine and predictor.
- Risk: scope creep into parent closeout within this slice.
  - Mitigation: limit `v0.5.3` to hardening + consistency; defer rollup to parent `v0.5` artifacts.

## Assumptions and Defaults
- Device scope remains CPU-only.
- Model format remains v1; no schema identifier changes.
- Legacy compatibility remains trees-only in compatibility mode.
