# v0.1.4 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.1.4`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion 1: when stop reason is `ValidationLossPlateau`, returned model state is rolled back to `best_validation_round` (including zero-round case).
- Evidence: `crates/engine/src/lib.rs` plateau finalization now truncates stumps/round state to `best_validation_round`; test `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason` asserts rollback to zero-round checkpoint (`rounds_completed == 0`, `model.stumps.is_empty()`).
- Status: PASS

- Criterion 2: summary fields align with rolled-back checkpoint (`rounds_completed`, loss/sample traces, `final_loss`, `final_validation_loss`).
- Evidence: same test asserts rollback-aligned summary semantics:
  - `validation_loss_per_completed_round.is_empty()`
  - `sampled_rows_per_completed_round.is_empty()`
  - `sampled_features_per_completed_round.is_empty()`
  - `final_loss == initial_loss`
  - `final_validation_loss == initial_validation_loss`
- Status: PASS

- Criterion 3: existing subsampling and validation-stop contract tests remain passing.
- Evidence: `cargo test --workspace` passed; `alloygbm_engine` still passes:
  - `sampled_row_indices_are_seeded_and_non_prefix`
  - `sampled_feature_tiles_are_seeded_and_non_prefix`
  - `sampled_indices_respect_ceil_minimum_and_upper_bound_rules`
  - `validation_early_stopping_requires_validation_dataset`
  - `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason`
- Status: PASS

- Criterion 4: `cargo fmt -- --check` passes.
- Evidence: command output (this pass) exit code `0`.
- Status: PASS

- Criterion 5: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command output (this pass) completed successfully with no warnings promoted to errors.
- Status: PASS

- Criterion 6: `cargo test --workspace` passes.
- Evidence: command output (this pass) reports all workspace unit/doc test suites passing.
- Status: PASS

- Criterion 7: `cargo doc --workspace --no-deps` passes.
- Evidence: command output (this pass) reports successful docs generation under `target/doc`.
- Status: PASS

- Criterion 8: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command output (this pass) reports `Ran 7 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.4` scope.

## Residual Risks
- Validation checkpoint semantics are now explicit, but broader `v0.1` tree-depth behavior remains incomplete.
- Parent `v0.1` rollup evidence remains pending additional child-layer completion.

## Final Readiness
- Ready: Yes (for `v0.1.4` validation best-checkpoint semantics scope).
