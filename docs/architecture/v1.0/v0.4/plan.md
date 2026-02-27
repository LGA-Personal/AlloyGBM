# AlloyGBM v0.4 Technical Plan

## Summary
- Goal: deliver the `0.4.0` milestone by adding finance-grade evaluation metrics and leakage guardrails on top of verified `v0.1`/`v0.2`/`v0.3` foundations.
- Success criteria:
  - Python evaluation API covers baseline regression metrics and finance-oriented ranking diagnostics defined for `0.4.0`,
  - time-aware split helpers provide purge/embargo controls that prevent common leakage patterns in panel workflows,
  - all workspace and Python verification gates stay green while preserving `v0.3` wrapper compatibility.
- Audience: engineers implementing `v0.4` child slices and reviewers deciding readiness to progress beyond evaluation-tooling scope.

## Scope
### In Scope
- Additive evaluation API in `bindings/python/alloygbm/` for:
  - baseline regression metrics (`RMSE`, `MAE`, `R2`, correlation),
  - finance-oriented metrics required by parent scope (`rank-IC`, `hit-rate`, and `ICIR` support).
- Time-aware validation helpers for leakage-safe evaluation:
  - purge gap and embargo controls,
  - time/group-aware split generation for panel datasets.
- Deterministic input validation and explicit error semantics for evaluation/split helpers.
- Child-layer decomposition and state tracking under `v0.4` using `v0.3.x` slices.
- Parent closeout artifacts:
  - `docs/architecture/v1.0/v0.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/verification_report.md`

### Out of Scope
- Ranking objective training (`1.1.0` scope).
- CUDA/Metal backend expansion.
- SHAP algorithm expansion beyond already-verified behavior.
- SIMD/performance optimization campaigns.
- Continuous-feature quantization changes in native training bridge (known prior constraint).

## Interfaces and Types
- `bindings/python/alloygbm/evaluation.py` (new module expected in this layer):
  - metric helper functions with deterministic float outputs and strict input validation.
- `bindings/python/alloygbm/validation.py` (new module expected in this layer):
  - leakage-aware split helper APIs for time/group indexed data.
- `bindings/python/alloygbm/__init__.py`:
  - additive exports for evaluation/validation helpers without breaking existing public names.
- `bindings/python/tests/`:
  - deterministic unit/integration coverage for metrics and split behavior.
- Existing native bridge interfaces in `bindings/python/src/lib.rs`:
  - no breaking changes required for `v0.4`; Python-side evaluation tooling remains primary.

Backward-compatibility expectations:
- preserve existing `GBMRegressor` constructor, `fit`, `predict`, `get_params`, and `set_params` behavior from `v0.3`.
- introduce evaluation and validation helpers additively; no parameter renames or behavior flips in existing APIs.

## Implementation Sequence
1. Execute `docs/architecture/v1.0/v0.4/v0.3.1/plan.md` for baseline metric helpers and test harness.
2. Open next child slice (`v0.3.2`) for finance-oriented metrics (rank-IC, hit-rate, ICIR) and deterministic edge-case handling.
3. Open next child slice (`v0.3.3`) for time-aware split helpers with purge/embargo controls and panel validation rules.
4. Open final child slice (`v0.3.4`, if needed) for API polish, docs, and residual acceptance gaps.
5. Close parent `v0.4` with rollup notes, verification report, and `docs/architecture/state/layer_index.yaml` update.

## Test Cases and Scenarios
- Unit cases:
  - exact/near-exact fixture assertions for each metric function,
  - split-index validity checks (no overlap, purge enforcement, embargo enforcement),
  - deterministic behavior for repeated runs with identical inputs.
- Integration cases:
  - `GBMRegressor.fit`/`predict` output evaluated through new metric APIs in representative workflows,
  - panel-like time/group split generation consumed by evaluation pipeline tests.
- Failure and edge cases:
  - mismatched lengths, empty inputs, non-finite values, and malformed timestamps/groups,
  - constant-target/constant-prediction paths with explicit documented behavior.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Risks and Mitigations
- Risk: evaluation semantics drift from quant expectations.
  - Mitigation: encode metric definitions and edge-case defaults in tests and layer docs before implementation.
- Risk: leakage helper APIs become underspecified and inconsistent across child slices.
  - Mitigation: define split contracts and validation defaults in each child plan before code changes.
- Risk: scope creep into ranking training or backend work.
  - Mitigation: keep `v0.4` strictly limited to evaluation/validation tooling and additive Python API changes.
- Risk: changes regress previously verified `v0.3` wrapper contract.
  - Mitigation: keep existing wrapper tests as non-negotiable gates in every `v0.4` child verification run.

## Assumptions and Defaults
- Device scope remains CPU-only through `v0.4`.
- `v0.4` child layers continue `v0.3.x` numbering.
- Default implementation locus is Python package modules first; Rust/native changes are only allowed when required by an explicit child-layer plan.
- Parent completion requires both metric coverage and leakage guardrail coverage with passing gate commands.
