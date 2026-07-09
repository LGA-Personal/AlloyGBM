# Task 1 Report

status: completed

files changed:
- `/Users/lashby/Projects/AlloyGBM/crates/engine/src/split_options.rs`
- `/Users/lashby/Projects/AlloyGBM/crates/engine/src/env.rs`
- `/Users/lashby/Projects/AlloyGBM/crates/engine/src/trainer/policy.rs`
- `/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/lib.rs`
- `/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/pl.rs`
- `/Users/lashby/Projects/AlloyGBM/crates/backend_cpu/src/tests/main.rs`

commits:
- `fix: enforce split row feasibility in backend scanners`

RED command/output summary:
- Added the two backend tests first in `crates/backend_cpu/src/tests/main.rs`.
- Ran the brief's literal focused command:
  - `cargo test -p alloygbm-backend-cpu numeric_split_scanner_skips_candidates_below_min_rows_per_leaf categorical_split_scanner_skips_candidates_below_min_rows_per_leaf`
  - Result: cargo CLI usage failure because `cargo test` accepts a single test filter before `--`.
- Ran a valid equivalent focused filter to verify RED:
  - `cargo test -p alloygbm-backend-cpu split_scanner_skips_candidates_below_min_rows_per_leaf`
  - Result: compile failure `E0560` because `SplitSelectionOptions` had no `min_rows_per_leaf` field in the new tests.

GREEN command/output summary:
- `cargo fmt --all --check`
  - Passed.
- `cargo test -p alloygbm-backend-cpu numeric_split_scanner_skips_candidates_below_min_rows_per_leaf`
  - Passed: `1 passed; 0 failed`.
- `cargo test -p alloygbm-backend-cpu categorical_split_scanner_skips_candidates_below_min_rows_per_leaf`
  - Passed: `1 passed; 0 failed`.
- `cargo test -p alloygbm-backend-cpu best_split_with_min_child_hessian_can_prune_all_splits`
  - Passed: `1 passed; 0 failed`.

self-review notes:
- Added `min_rows_per_leaf` to `SplitSelectionOptions` with a conservative default of `1`.
- Initialized the new field in env-derived options and in `trainer/policy.rs` for the pre-`IterationControls` path approved during execution.
- Enforced row feasibility in all requested backend scanner paths:
  - scalar numeric scanner
  - SIMD numeric scanner
  - categorical scanner
  - piecewise-linear scanner
- Updated explicit backend test option literals so the new field is always initialized and existing behavior remains unchanged where the threshold should stay permissive.

concerns:
- The brief's RED verification command is not a valid `cargo test` invocation; a single shared filter was needed to verify the intended failing tests.
