# AlloyGBM v0.1.1 Plan (v0.2 Contract Lock: Subsampling + Validation Early Stopping)

## Objective
Establish the first executable `v0.2` child layer by locking training-control contracts for row/column subsampling and validation-set early-stopping, with deterministic baseline behavior and test-backed interfaces across Rust core/engine and Python estimator surface.

## Scope
- In scope:
  - Extend `core::TrainParams` to include `row_subsample`, `col_subsample`, `early_stopping_rounds`, and `min_validation_improvement` with validation.
  - Extend `engine::IterationControls` and run summary to represent subsampling and validation-loss tracking contracts.
  - Add an engine validation-aware iterative training entry point that can stop on validation-loss plateau.
  - Use deterministic baseline sampling behavior suitable for contract-lock phase (not tuned stochastic sampling).
  - Update Python `GBMRegressor` parameter contract (`__init__`, `get_params`, `set_params`) for the new controls.
  - Add/adjust tests proving config validation and validation-stop behavior.
- Out of scope:
  - Full-depth CART implementation beyond current stump-level iterative path.
  - Production-grade random sampling strategy/performance tuning.
  - Ranking, SHAP, categorical execution, CUDA/Metal, and broader sklearn UX expansion.

## Deliverables
1. Core contract package:
  - `crates/core/src/lib.rs` includes new train-control fields and validation rules.
2. Engine contract/execution package:
  - `crates/engine/src/lib.rs` includes:
    - subsampling + validation early-stopping fields on `IterationControls`,
    - validation-aware training summary fields,
    - `fit_iterations_with_validation_summary` flow with deterministic baseline behavior.
3. Python contract package:
  - `bindings/python/alloygbm/regressor.py` supports new control params and validation.
4. Verification package:
  - updated Rust/Python tests for new control contracts and stop-reason behavior.
  - layer artifacts: `implementation_notes.md`, `verification_report.md`.

## Implementation Sequence
1. Add `v0.1.1` plan artifact.
2. Update `core::TrainParams` and `validate_train_params` with `v0.2` control fields and tests.
3. Update engine controls/summary contracts and add validation-aware iteration entry point.
4. Implement deterministic contract-phase row/column subsampling plumbing in iterative training.
5. Update Python regressor contract for parameter parity.
6. Run verification commands and record evidence in layer artifacts.

## Acceptance Criteria
1. `TrainParams` includes `row_subsample`, `col_subsample`, `early_stopping_rounds`, and `min_validation_improvement`, and rejects invalid values.
2. `IterationControls` can represent subsampling and validation early-stopping policy with validation checks.
3. Engine provides a validation-aware iterative training path that reports validation loss trace and can stop with a validation plateau reason.
4. Python `GBMRegressor` exposes and validates the same new parameter set in constructor/get/set params.
5. `cargo fmt -- --check` passes.
6. `cargo clippy --workspace --all-targets -- -D warnings` passes.
7. `cargo test --workspace` passes.
8. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: contract additions create partial behavior that could be mistaken for final sampling semantics.
  - Mitigation: document deterministic baseline sampling in implementation notes and keep scope explicit.
- Risk: validation early-stopping API ambiguity without a validation dataset.
  - Mitigation: enforce explicit validation-dataset requirement when early-stopping policy is enabled.
- Risk: test brittleness from floating-point loss checks.
  - Mitigation: assert stop reasons and monotonic/relative behavior instead of brittle exact values.

## Exit Condition
`v0.1.1` is complete when control contracts are implemented and test-backed, validation-aware stopping is wired, verification commands are green, and layer artifacts are recorded.
