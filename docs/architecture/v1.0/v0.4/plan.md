# AlloyGBM v0.4 Technical Plan

## Summary
- Goal: deliver finance-grade evaluation and leakage guardrails (`0.4.0`) on top of the now-stable `v0.3` Python wrapper and native training path.
- Success criteria:
  - built-in evaluation metrics include regression metrics and finance-relevant metrics,
  - time-aware split tooling supports leakage-aware workflows (purged/embargo-aware behavior),
  - Python-facing evaluation helpers are test-covered and packaging/runtime gates remain green.
- Audience: engineers implementing `v0.4` child slices and reviewers validating quant-evaluation readiness.

## Scope
### In Scope
- Regression and finance-oriented metric surface for model evaluation (for example RMSE/MAE/R2/correlation/rank-oriented helper metrics).
- Time-aware validation helpers for panel/time-indexed datasets, including leakage-aware split controls.
- Python API contracts and test evidence for evaluation workflows using wrapper-produced predictions.
- Child-layer decomposition under `v0.4` via `v0.3.x` slices, with state-index updates.

### Out of Scope
- Ranking objective training (planned for `1.1.0`).
- CUDA/Metal backend expansion.
- SHAP algorithm expansion beyond existing scope.
- SIMD/performance optimization campaigns.

## Interfaces and Types
- `bindings/python/alloygbm/`:
  - public metric and evaluation helper interfaces exposed to Python workflows.
- `crates/core` and/or `crates/engine`:
  - metric primitives and evaluation contract types when shared with Rust-side logic.
- `bindings/python/tests/`:
  - contract and runtime evidence for evaluation and leakage guardrails.

Backward-compatibility expectations:
- preserve existing `GBMRegressor` fit/predict and parameter contracts established in `v0.3`;
- add evaluation APIs additively without breaking wrapper usage patterns.

## Implementation Sequence
1. Open first child slice at `docs/architecture/v1.0/v0.4/v0.3.1/plan.md`.
2. Implement baseline evaluation metric support and corresponding tests.
3. Implement leakage-aware split controls in subsequent `v0.3.x` child slices as needed.
4. Close parent `v0.4` with rollup implementation and verification artifacts once child acceptance criteria are met.

## Test Cases and Scenarios
- Unit cases:
  - metric computations on deterministic fixtures,
  - parameter and boundary validation for split/evaluation helpers.
- Integration cases:
  - wrapper training + prediction + evaluation flow on representative time-indexed datasets,
  - packaging/runtime invocation coverage for evaluation APIs.
- Failure and edge cases:
  - malformed indices/groups, leakage-prone split configs, and inconsistent input lengths.
- Acceptance test mapping:
  - `cargo fmt -- --check`,
  - `cargo clippy --workspace --all-targets -- -D warnings`,
  - `cargo test --workspace`,
  - `cargo doc --workspace --no-deps`,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`.

## Risks and Mitigations
- Risk: evaluation APIs drift from leakage-safe defaults.
  - Mitigation: enforce explicit split configuration validation and test failure modes.
- Risk: metric semantics mismatch between Python and Rust components.
  - Mitigation: use fixture-based parity tests and clear metric definition docs.
- Risk: scope creeps into ranking objective implementation.
  - Mitigation: keep `v0.4` limited to evaluation and validation tooling only.

## Assumptions and Defaults
- Device scope remains CPU-only for this phase.
- `v0.4` child layers use `v0.3.x` numbering for organization.
- `v0.3.1` is the first execution slice for `v0.4`.
