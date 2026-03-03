# AlloyGBM v0.0 Plan (Contracts-First Foundation)

## Objective
Establish the architecture, interfaces, and packaging foundation that de-risks implementation of `0.2+` and preserves backend-agnostic design for later CUDA/Metal phases.

## Deliverables
- Rust workspace and crate scaffolding:
  - `core`, `engine`, `backend_cpu`, `predictor`, `shap`, `categorical`, `bindings_python`
- Contract definitions:
  - dataset/config/model metadata types in `core`
  - backend primitive traits and trainer interfaces in `engine`
- Model IO contract v1 draft:
  - versioned binary model artifact
  - sidecar JSON metadata schema
- Python binding scaffold:
  - importable `alloygbm.GBMRegressor`
  - parameter schema validation
- CI foundation:
  - Linux + macOS
  - Python `3.10-3.13`
  - rustfmt/clippy/unit tests + Python smoke tests

## Implementation Sequence (4-week target)
1. Week 1: Workspace bootstrap, crate boundaries, lint/test/CI skeleton.
2. Week 2: Core type system, config validation, error taxonomy, metadata schema draft.
3. Week 3: Engine/backend contracts, predictor flat-tree contract, serialization stubs.
4. Week 4: Python wrapper skeleton, maturin wheel build, cross-platform smoke tests.

## Out of Scope for v0.0
- Full histogram trainer implementation.
- Performance optimization and SIMD tuning.
- Full SHAP algorithm.
- Full categorical transform execution pipeline.
- Ranking objective and ranking metrics.

## Acceptance Criteria
- `cargo test --workspace` passes.
- All crate docs/build checks pass in CI on Linux and macOS.
- Model metadata/version roundtrip tests pass.
- Python wheel builds and imports across supported Python versions.
- `GBMRegressor` constructor and parameter validation callable from Python.

## Risks and Mitigations
- Risk: interface churn before implementation starts.
  - Mitigation: ADRs for trait boundaries and model format before `0.1` coding.
- Risk: Python/Rust contract mismatch.
  - Mitigation: schema-driven parameter validation and smoke tests in CI.
- Risk: future GPU backends blocked by early CPU assumptions.
  - Mitigation: keep backend primitives abstract and avoid CPU-specific leakage into `engine`.
