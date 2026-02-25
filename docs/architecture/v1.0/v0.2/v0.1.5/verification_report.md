# v0.1.5 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.1.5`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion 1: iterative training no longer caps rounds by `max_depth`; `effective_round_cap == controls.rounds`.
- Evidence: engine test `fit_iterations_summary_uses_round_count_as_round_cap` now asserts `rounds_requested == 3`, `effective_round_cap == 3`, and `rounds_completed == 3` with `max_depth: 1`.
- Status: PASS

- Criterion 2: within a round, engine can commit more than one split when `max_depth > 1` and child splits are valid.
- Evidence: engine test `fit_iterations_grows_multiple_nodes_per_round_when_depth_allows` asserts one completed round produces three stumps (`node_id` values include `0`, `1`, `2`).
- Status: PASS

- Criterion 3: non-root split contributions are path-conditioned for inference and validation application.
- Evidence:
  - `TrainedModel::predict_row` uses `row_satisfies_stump_path_features`.
  - validation candidate prediction path uses `apply_stump_to_binned_predictions_with_path` + `row_satisfies_stump_path_bins`.
  - direct unit test `predict_row_applies_non_root_nodes_only_when_path_matches` verifies child-node contribution is skipped when parent path does not match.
- Status: PASS

- Criterion 4: validation plateau rollback remains consistent when rounds can emit multiple stumps.
- Evidence: rollback now truncates by summed `stumps_per_completed_round` for `best_validation_round`; engine test `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason` remains passing.
- Status: PASS

- Criterion 5: existing control-contract behavior remains passing (gain/min-row/min-leaf/loss-improvement).
- Evidence: engine tests continue to pass:
  - `fit_iterations_controls_enforce_min_split_gain`
  - `fit_iterations_controls_enforce_min_rows_per_leaf`
  - `fit_iterations_controls_enforce_min_abs_leaf_value`
  - `fit_iterations_summary_reports_loss_improvement_threshold_stop_reason`
  - `fit_iterations_summary_allows_bounded_weak_improvement_rounds`
- Status: PASS

- Criterion 6: `cargo fmt -- --check` passes.
- Evidence: command exit status `0`.
- Status: PASS

- Criterion 7: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command completed successfully with no warnings promoted to errors.
- Status: PASS

- Criterion 8: `cargo test --workspace` passes.
- Evidence: workspace test suites all green, including `alloygbm_engine` (`39 passed`) and `alloygbm_backend_cpu` (`7 passed`).
- Status: PASS

- Criterion 9: `cargo doc --workspace --no-deps` passes.
- Evidence: docs generated successfully under `target/doc`.
- Status: PASS

- Criterion 10: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command output reports `Ran 7 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.5` scope.

## Residual Risks
- Tree structure is still represented through path-conditioned stumps rather than a dedicated explicit tree-node artifact schema.
- Parent rollup verification artifacts for `v0.2` and `v1.0` remain pending.

## Final Readiness
- Ready: Yes (for `v0.1.5` depth-limited multi-node round growth and path-aware application scope).
