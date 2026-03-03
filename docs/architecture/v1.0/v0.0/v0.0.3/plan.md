# AlloyGBM v0.0.3 Plan (v0.0 Week 3 Executable Slice)

## Objective
Deliver the first executable training-path slice on top of `v0.0.2` contracts by implementing deterministic CPU primitive behavior, a minimal objective, and a one-round training orchestration path.

## Scope
- In scope:
  - Deterministic baseline implementations for `backend_cpu` primitive methods:
    - histogram build
    - best split selection
    - split application
    - gradient/hessian reductions
  - Minimal squared-error objective implementation in `engine`.
  - Trainer one-round flow that exercises contract-validated backend/objective interactions.
  - Python `GBMRegressor.fit/predict` minimal baseline behavior (constant predictor) with fitted-state handling.
  - Focused tests for backend behavior, trainer round flow, and Python fit/predict behavior.
- Out of scope:
  - Multi-level tree growth or full boosting loop.
  - Early stopping, subsampling, or regularization tuning.
  - Performance optimization/SIMD.
  - Predictor crate production inference path.

## Deliverables
1. CPU primitive baseline package:
   - `backend_cpu` methods return concrete outputs for valid inputs (no longer placeholder `NotImplemented`).
   - deterministic split candidate selection for a single node.
2. Engine executable slice:
   - `SquaredErrorObjective` implementation.
   - `Trainer::fit_one_round` (or equivalent) returning round summary metadata.
3. Python baseline estimator:
   - `fit` stores baseline mean target and marks fitted.
   - `predict` returns constant predictions with input-shape checks.

## Implementation Plan
1. Add `v0.0.3` plan artifact before code changes.
2. Implement `backend_cpu` concrete primitive logic with bounded validation.
3. Add minimal objective + one-round trainer flow in `engine`.
4. Implement Python baseline `fit/predict`.
5. Add/adjust Rust and Python tests to cover new behavior.
6. Run verification suite and record evidence in `verification_report.md`.

## Acceptance Criteria
1. `cargo fmt -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes, including new backend/engine tests for executable one-round flow.
4. `backend_cpu` tests verify histogram aggregation, split selection, and partition behavior on deterministic fixtures.
5. `engine` tests verify one-round fit flow returns coherent summary data and catches invalid row alignment contracts.
6. Python unit tests verify `GBMRegressor.fit` establishes fitted state and `predict` returns constant-length outputs with shape/fitted-state validation.

## Risks and Mitigations
- Risk: accidental drift toward full trainer complexity.
  - Mitigation: cap scope at a single round and deterministic fixtures.
- Risk: split-gain formula instability for tiny hessian sums.
  - Mitigation: use a small epsilon and add explicit test coverage.
- Risk: Python baseline behavior diverges from future native backend.
  - Mitigation: keep baseline semantics simple and documented as temporary.

## Exit Condition
`v0.0.3` is complete when the one-round training slice executes against concrete CPU primitives, Python fit/predict baseline is test-covered, and layer implementation/verification artifacts are recorded.
