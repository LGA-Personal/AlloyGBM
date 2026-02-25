# v0.1.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.2/v0.1.2`
- Date: 2026-02-25

## Acceptance Criteria Matrix
- Criterion: engine no longer relies on prefix-only row/feature subsampling for rates `< 1.0`.
- Evidence: `crates/engine/src/lib.rs` sampling helpers replaced by seeded per-round hash-ranked selectors (`sampled_indices`, `sampled_row_indices`, `sampled_feature_tiles`).
- Status: PASS

- Criterion: sampled row/feature cardinalities match configured rate rules.
- Evidence: helper logic enforces ceil + minimum 1 behavior; test `sampled_feature_tiles_cover_expected_feature_count` validates exact feature coverage.
- Status: PASS

- Criterion: deterministic mode yields reproducible sample selections.
- Evidence: test `sampled_row_indices_are_seeded_and_non_prefix` validates repeatability for same seed/round and rejects prefix-pattern behavior.
- Status: PASS

- Criterion: iteration summary reports per-round sampled row/feature counts.
- Evidence: `IterationRunSummary` now includes sampled coverage vectors; validation-stop summary test asserts expected sampled counts.
- Status: PASS

- Criterion: prior validation early-stopping behavior remains intact.
- Evidence: `fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason` remains passing.
- Status: PASS

- Criterion: verification command gates pass.
- Evidence: all required commands completed successfully in this pass.
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
