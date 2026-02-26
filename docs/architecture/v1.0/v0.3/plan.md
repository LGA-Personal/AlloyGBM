# AlloyGBM v0.3 Technical Plan

## Summary
- Goal: provide a high-level `0.3.0` parent plan that organizes sklearn-wrapper delivery through nested `v0.2.x` child slices over the existing Rust CPU training/prediction core.
- Success criteria:
  - `GBMRegressor` supports stable sklearn-style `fit`, `predict`, `get_params`, and `set_params` behavior,
  - wrapper supports common tabular inputs (`numpy.ndarray`, `pandas.DataFrame`, and Polars-like array exports),
  - packaging/runtime checks remain green for maturin-built extension and Python tests.
- Audience: engineers implementing `v0.3` child layers and reviewers gating progression beyond wrapper-contract maturity.

## Scope
### In Scope
- Python-facing estimator contract hardening for `GBMRegressor` and native bridge wiring needed to move beyond scaffold behavior.
- Input normalization and validation compatible with major array/dataframe interfaces used in quant workflows.
- Parameter surface compatibility and predictable error semantics for sklearn-style usage.
- Packaging and import/runtime verification of the Python extension module as part of layer acceptance gates.
- Child-layer decomposition and state-index updates for iterative `v0.3` delivery.

### Out of Scope
- Ranking objectives/metrics.
- SHAP algorithm implementation changes.
- Categorical pipeline execution expansion.
- SIMD/performance optimization campaigns (`0.5+` scope).
- CUDA/Metal/MLX backend work.

## Interfaces and Types
- `bindings/python/alloygbm/regressor.py`:
  - canonical Python estimator behavior and contract validation.
- `bindings/python/src/lib.rs`:
  - native extension entrypoints required by wrapper behavior.
- `bindings/python/tests/`:
  - contract and runtime integration evidence.
- `pyproject.toml` + `bindings/python/Cargo.toml`:
  - wheel/extension packaging contract and test/runtime dependencies.

Backward-compatibility expectations:
- keep parameter names and validation semantics stable once introduced in `v0.3` child layers;
- prefer additive bridge changes over breaking Python API shape changes during `0.3.x`.

## Implementation Sequence
1. Execute the first child slice at `docs/architecture/v1.0/v0.3/v0.2.1/plan.md` and complete its implementation/verification artifacts.
2. Re-evaluate remaining `0.3.0` acceptance gaps and open the next child slice (`v0.2.2`, then `v0.2.3`, etc.) one at a time.
3. Keep `docs/architecture/state/layer_index.yaml` aligned to the deepest active child target after each slice.
4. Close parent `v0.3` with rollup `implementation_notes.md` and `verification_report.md` once all child-slice acceptance criteria are satisfied.

## Test Cases and Scenarios
- Unit cases:
  - parameter validation and roundtrip behavior (`get_params`/`set_params`),
  - input-shape and type validation across supported Python containers.
- Integration cases:
  - wrapper `fit` then `predict` behavior via native-backed path,
  - wheel build/install/import and runtime invocation coverage.
- Failure and edge cases:
  - malformed inputs, feature-count mismatches, and missing native module conditions.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Risks and Mitigations
- Risk: wrapper scope drifts into non-`0.3` algorithmic work.
  - Mitigation: keep child-layer plans constrained to Python API/bridge and runtime evidence.
- Risk: input-adapter complexity causes inconsistent behavior across libraries.
  - Mitigation: define deterministic normalization rules and assert parity in tests per input type.
- Risk: packaging/runtime test steps are environment-sensitive.
  - Mitigation: keep runtime tests isolated and use explicit build/install setup in test fixtures.

## Assumptions and Defaults
- CPU-only device scope remains unchanged.
- `GBMRegressor` remains the primary public estimator class for `0.3.0`.
- Child layers under `v0.3` use `v0.2.x` numbering for organization.
- `v0.2.1` is the first execution step of `v0.3` and must be completed before opening later `v0.2.x` slices.
