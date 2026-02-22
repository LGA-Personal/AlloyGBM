# v0.0.1 Implementation Notes

## Summary of What Was Built
- Bootstrapped a Rust workspace with crate boundaries under `crates/`.
- Added a Python bindings crate and package scaffold under `bindings/python`.
- Added strict CI checks for Linux/macOS and Python `3.10-3.13` smoke tests.
- Added stub-only APIs across core, engine, backend, predictor, SHAP, and categorical modules.

## Non-Intuitive Decisions
- Chose Rust package names with `alloygbm-` prefixes (for example `alloygbm-core`) while keeping folder names short (`crates/core`) to avoid namespace confusion with standard library crate names.
- Exposed the native Python extension as `alloygbm._alloygbm` rather than top-level `_alloygbm` to keep the public import surface package-scoped.
- Added a root `README.md` because maturin reads root `pyproject.toml` metadata during `--manifest-path` builds and fails if the declared readme file is missing.

## Plan Contradictions and Why
- No contradictions were required during implementation.

## Boundary/Interface Changes vs Plan
- No boundary or interface changes were made versus the approved plan.

## Known Gaps Deferred to v0.0.2
- Trainer logic remains stubbed (`fit_stub`).
- Backend CPU kernels are stubbed.
- Predictor/SHAP/categorical routines are stubbed.
- Python estimator behavior is constructor validation only (no fit/predict).

## Follow-Up Actions
- Define concrete trait method signatures and data carriers for `v0.0.2` contract-definition work.
- Add model format section schema definitions and serializer/deserializer contracts in `core`.
- Add richer Python parameter schema and estimator method stubs (`fit`, `predict`) once `v0.0.2` starts.
