# v0.1.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.1.2`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion 1: engine no longer uses prefix-only row/feature sampling behavior for subsampling rates `< 1.0`.
- Evidence: helper implementation in `crates/engine/src/lib.rs` uses seeded hash-ranked selectors (`sampled_indices`, `sampled_row_indices`, `sampled_feature_tiles`); tests `sampled_row_indices_are_seeded_and_non_prefix` and `sampled_feature_tiles_are_seeded_and_non_prefix` assert non-prefix selections.
- Status: PASS

- Criterion 2: row and feature sampled cardinalities match configured rates (ceil + minimum 1 rules).
- Evidence: tests `sampled_indices_respect_ceil_minimum_and_upper_bound_rules` and `sampled_feature_tiles_cover_expected_feature_count` validate cardinality behavior for low, partial, and full subsample rates.
- Status: PASS

- Criterion 3: deterministic mode yields reproducible sample selections for identical seed/inputs.
- Evidence: tests `sampled_row_indices_are_seeded_and_non_prefix` and `sampled_feature_tiles_are_seeded_and_non_prefix` assert same seed + same round returns identical selections.
- Status: PASS

- Criterion 4: iteration summary reports per-round sampled row and feature counts.
- Evidence: `IterationRunSummary` includes `sampled_rows_per_completed_round` and `sampled_features_per_completed_round`; test `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason` asserts expected vectors (`vec![4]`, `vec![2]`).
- Status: PASS

- Criterion 5: existing validation early-stopping behavior remains passing.
- Evidence: tests `validation_early_stopping_requires_validation_dataset` and `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason` pass and preserve the validation plateau stop contract.
- Status: PASS

- Criterion 6: `cargo fmt -- --check` passes.
- Evidence: command output (this pass) exit code `0`.
- Status: PASS

- Criterion 7: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command output (this pass) reports successful completion with no warnings promoted to errors.
- Status: PASS

- Criterion 8: `cargo test --workspace` passes.
- Evidence: command output (this pass) reports `alloygbm_engine` test suite passing with 37/37 tests, including new sampling tests.
- Status: PASS

- Criterion 9: `cargo doc --workspace --no-deps` passes.
- Evidence: command output (this pass) reports successful doc generation under `target/doc`.
- Status: PASS

- Criterion 10: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command output (this pass) reports `Ran 7 tests` and `OK`.
- Status: PASS

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS

## Residual Uncovered Criteria
- None for `v0.1.2` scope.

## Residual Risks
- Current model loop remains stump-level and does not yet satisfy full `0.2.0` behavior breadth.
- Parent `v0.2` rollup evidence is still pending additional child-layer completion.

## Final Readiness
- Ready: Yes (for `v0.1.2` seeded subsampling semantics scope).
