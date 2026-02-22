# v0.0.2 Implementation Notes

## Summary of What Was Built
- Added concrete core contract types in `crates/core/src/lib.rs` for:
  - dataset/training carriers (`DatasetMatrix`, `TrainingDataset`, `BinnedMatrix`)
  - backend primitive carriers (`GradientPair`, `FeatureTile`, `NodeSlice`, histogram/split structs)
  - model-format v1 contracts (`ModelBinaryHeader`, `ModelSectionDescriptor`, `ModelIoContractV1`)
- Added core validation and serialization helpers:
  - dataset/schema validators
  - model-contract validator
  - metadata sidecar JSON serializer/deserializer
  - binary encode/decode helpers for header and section descriptors
- Refactored `crates/engine/src/lib.rs` to use typed contract signatures for `BackendOps` and `ObjectiveOps`.
- Added trainer-level contract validation (`validate_fit_contract`) while keeping training behavior stubbed.
- Updated `crates/backend_cpu/src/lib.rs` to conform to the new `BackendOps` method signatures.
- Extended Python estimator surface in `bindings/python/alloygbm/regressor.py` with `get_params`, `set_params`, and explicit `fit`/`predict` stubs.

## Non-Intuitive Decisions
- Implemented strict, deterministic JSON parsing for `ModelMetadata` without adding new dependencies. This keeps bootstrap complexity low and avoids introducing additional crates before algorithmic work starts.
- Kept `fit_stub` unimplemented after contract validation. The layer goal is interface stability, not tree growth behavior.
- `predict` enforces fitted-state checks first, then remains unimplemented by design.

## Plan Contradictions and Why
- No contradictions were introduced relative to `v0.0.2/plan.md`.

## Boundary/Interface Changes vs Plan
- No crate boundary changes were made.
- The expected boundary evolution occurred inside existing crates:
  - `core`: contract schema + model IO contracts
  - `engine`: backend/objective trait signatures now typed
  - `backend_cpu`: updated trait conformance only
  - Python bindings: richer sklearn-style stub interface

## Known Gaps Deferred to v0.0.3+
- No histogram tree-building logic is implemented.
- No objective-specific training loop beyond contract validation.
- No end-to-end model serialization files are emitted yet; only contract-level encode/decode primitives exist.
- Python `fit` and `predict` remain placeholders.

## Follow-Up Actions
- Create `v0.0.3` plan for minimal histogram training loop scaffolding against the stabilized contracts.
- Add first end-to-end model artifact writer/reader plumbing that uses `ModelIoContractV1`.
- Add Python integration smoke that exercises native extension plus estimator scaffolding together.
