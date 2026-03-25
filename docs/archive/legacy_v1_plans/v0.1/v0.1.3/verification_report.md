# v0.1.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.1.3`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion 1: repository includes an end-to-end CPU backend training test using real `CpuBackend` that verifies trained-model loss beats naive constant baseline loss on a fixed fixture.
- Evidence: `crates/backend_cpu/src/lib.rs` test `cpu_backend_training_beats_naive_baseline_mse` trains with `Trainer + CpuBackend`, computes model MSE and baseline MSE, and asserts `model_mse < baseline_mse`.
- Status: PASS

- Criterion 2: repository includes deterministic reproducibility evidence for CPU backend training (same deterministic params/seed => identical artifact bytes).
- Evidence: `crates/backend_cpu/src/lib.rs` test `cpu_backend_deterministic_training_has_stable_artifact_bytes` performs two deterministic training runs and asserts serialized artifact byte equality.
- Status: PASS

- Criterion 3: existing `v0.1.1`/`v0.1.2` subsampling + validation-stop contracts remain passing.
- Evidence: `cargo test --workspace` passed; `alloygbm_engine` suite includes and passes:
  - `sampled_row_indices_are_seeded_and_non_prefix`
  - `sampled_feature_tiles_are_seeded_and_non_prefix`
  - `sampled_indices_respect_ceil_minimum_and_upper_bound_rules`
  - `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason`
  - `validation_early_stopping_requires_validation_dataset`
- Status: PASS

- Criterion 4: `cargo fmt -- --check` passes.
- Evidence: command output (this pass) exit code `0`.
- Status: PASS

- Criterion 5: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command output (this pass) completed successfully with no warnings promoted to errors.
- Status: PASS

- Criterion 6: `cargo test --workspace` passes.
- Evidence: command output (this pass) reports all workspace test suites passing, including new backend CPU integration tests.
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
- None for `v0.1.3` scope.

## Residual Risks
- This layer improves evidence quality but does not yet deliver full depth-limited tree-growth behavior expected by broader `0.1.0` scope.
- Parent `v0.1` rollup evidence remains pending additional child-layer completion.

## Final Readiness
- Ready: Yes (for `v0.1.3` CPU backend quality/reproducibility evidence scope).
