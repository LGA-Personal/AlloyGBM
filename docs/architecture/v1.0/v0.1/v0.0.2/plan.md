# AlloyGBM v0.0.2 Plan (v0.1 Week 2 Contracts)

## Objective
Define concrete interface contracts for `core` and `engine`, plus model format v1 schema/serialization stubs, so `v0.0.3+` implementation can proceed without contract churn.

## Scope
- In scope:
  - Contract data carriers and validation helpers in `crates/core`.
  - Model format v1 binary header/section descriptors in `crates/core`.
  - Metadata sidecar JSON serializer/deserializer contracts in `crates/core`.
  - Backend/objective trait signature upgrades in `crates/engine`.
  - CPU backend conformance to updated trait signatures in `crates/backend_cpu`.
  - Python estimator API stubs for `fit`, `predict`, `get_params`, and `set_params`.
- Out of scope:
  - Histogram tree growth implementation.
  - CPU kernel optimization/SIMD work.
  - Full predictor, SHAP, and categorical algorithm behavior.
  - Ranking objective behavior and metrics.

## Deliverables
1. Core contract package:
   - Dataset/model schema structs with explicit invariants and validators.
   - Model format v1 constants and binary header/section descriptors.
   - Metadata JSON sidecar encode/decode helpers with roundtrip tests.
2. Engine contract package:
   - `BackendOps` method signatures upgraded to typed inputs/outputs.
   - `ObjectiveOps` method signatures upgraded to typed gradient contracts.
   - Trainer-level contract validation entrypoint that remains non-training.
3. Backend and Python alignment:
   - `backend_cpu` updated to compile against new engine contracts.
   - `GBMRegressor` exposes sklearn-like parameter utilities and explicit fit/predict stubs.

## Implementation Plan
1. Implement core contract types and validators first; add unit tests for invariants.
2. Implement model-format header/section encode/decode helpers and metadata JSON roundtrip tests.
3. Refactor engine/backend traits to consume/produce typed core contracts.
4. Update CPU backend stubs and engine tests to match new signatures.
5. Extend Python regressor stubs and add focused Python unit tests.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes with new contract tests.
4. Metadata JSON and model header/section roundtrip tests pass in `alloygbm-core`.
5. `Trainer` construction and contract-entrypoint tests pass in `alloygbm-engine`.
6. Python regressor supports constructor validation, `get_params`, `set_params`, and raises explicit `NotImplementedError` for `fit`/`predict`.

## Risks and Mitigations
- Risk: overfitting contracts to CPU-only assumptions.
  - Mitigation: keep interfaces backend-agnostic (`BinnedMatrix`, row-index views, histogram bundles).
- Risk: brittle metadata parser without full JSON dependency.
  - Mitigation: keep parser strict and deterministic; cover with roundtrip + malformed-input tests.
- Risk: introducing early API noise in Python.
  - Mitigation: expose only familiar sklearn-style stubs and defer behavior to later layers.

## Exit Condition
`v0.0.2` is complete when all contract artifacts compile, tests verify invariants/roundtrips, and implementation/verification docs are recorded for this layer.
