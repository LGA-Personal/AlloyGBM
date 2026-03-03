# AlloyGBM v0.6.1 Implementation Notes

## Summary of What Was Built
- Executed the `v0.6.1` contract baseline slice for `v0.6` categorical support.
- Added categorical-state artifact contract helpers in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/core/src/lib.rs):
  - `CategoricalStatePayloadV1`,
  - `encode_categorical_state_payload_v1`,
  - `decode_categorical_state_payload_v1`,
  - `validate_categorical_state_payload_v1`,
  - `optional_single_section`,
  - `decode_optional_categorical_state_section_v1`.
- Tightened schema validation in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/core/src/lib.rs):
  - `validate_dataset_schema` now enforces strictly increasing categorical indices (no duplicates, no reordering).
- Added contract tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/crates/core/src/lib.rs):
  - categorical payload roundtrip and malformed-header rejection,
  - schema invariant rejection for duplicate/unsorted indices,
  - strict required-section compatibility remains true when optional `CategoricalState` section is present,
  - duplicate `CategoricalState` section rejection for optional section decode helper.

## Non-Intuitive Decisions
- Decision: enforce strictly increasing categorical feature indices in both dataset schema and categorical payload.
- Reason: canonical ordering removes ambiguity for artifact roundtrip and deterministic equality checks in later layers.
- Impact: callers must normalize categorical index lists before constructing schema/payload; behavior is deterministic with explicit validation errors.

- Decision: treat unknown categorical payload flags as hard serialization errors.
- Reason: this slice locks contract semantics and avoids silent acceptance of forward-incompatible payloads.
- Impact: future format expansion requires explicit version/flag handling updates rather than permissive fallback.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- Added new public `core` helpers/types for categorical-state payload contract; no public Python API changes.
- Required-section compatibility policy (`Trees` + `PredictorLayout` strict mode and legacy trees-only mode) was preserved.
- No `engine`, `predictor`, or `bindings/python` runtime behavior changes in this slice.

## Known Gaps Deferred to Next Layer
- `crates/categorical` still contains placeholder transform implementation; target/frequency encoder runtime work is deferred to `v0.6.2+`.
- Engine training preprocessing and predictor categorical replay integration are deferred to `v0.6.3+`.
- Python categorical runtime path changes are deferred to `v0.6.4+`.

## Follow-Up Actions
- Create `docs/architecture/v1.0/v0.6/v0.6.2/plan.md` for encoder implementation and deterministic fit/transform behavior.
- Integrate `core` categorical-state helpers into `engine`/`predictor` once `v0.6.2` contract details are finalized.
