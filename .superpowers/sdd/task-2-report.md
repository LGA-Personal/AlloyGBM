# Task 2 Report: Scale the PL Solve and Use Path Features

## Scope

Implemented Task 2 only:

- Threaded `LinearFeatureScaler` through the backend PL trait/API surface.
- Standardized PL regressor values during histogram accumulation and direct partition solves.
- Emitted `LinearLeaf::scaled(...)` metadata from PL training so prediction still consumes raw rows.
- Switched linear regressor selection from `0..d` to split-path features with dedupe and `MAX_PL_REGRESSORS` capping.
- Changed parent/child PL delta subtraction to align by feature id instead of slot position.
- Added the requested Rust helper tests and Python raw-scale regression test.

No public Python constructor APIs were changed.
No docs or review-resolution docs were edited.

## TDD Evidence

### Red

Added the requested tests first in:

- `crates/engine/src/trainer/tree_build.rs`
- `bindings/python/tests/test_pl_trees.py`

Observed focused red results:

1. `cargo test -p alloygbm-engine linear_regressor_features_follow_split_path_not_first_columns -- --nocapture`
   - failed at compile time as expected:
     - `cannot find function linear_regressor_features_for_split in this scope`
2. `maturin develop --release`
   - succeeded
3. `.venv/bin/python -m pytest bindings/python/tests/test_pl_trees.py::TestPLRegressor::test_linear_leaves_handle_raw_scale_features -q`
   - passed immediately (`1 passed`)
   - this meant the new Python coverage did not expose a pre-existing failure in the current tree-building path, but it still served as the required regression guard for the standardized-solve work

### Green

After implementation, the exact Task 2 Step 8 commands succeeded:

1. `cargo test -p alloygbm-engine linear_regressor_features_follow_split_path_not_first_columns -- --nocapture`
   - result: `1 passed; 0 failed`
2. `cargo test -p alloygbm-engine linear_regressor_features_cap_at_max_pl_regressors -- --nocapture`
   - result: `1 passed; 0 failed`
3. `cargo test -p alloygbm-backend-cpu pl -- --nocapture`
   - result: `49 passed; 0 failed`
4. `maturin develop --release`
   - result: editable wheel rebuilt and installed successfully
5. `.venv/bin/python -m pytest bindings/python/tests/test_pl_trees.py::TestPLRegressor::test_linear_leaves_handle_raw_scale_features -q`
   - result: `1 passed`

## Files Changed

- `crates/backend_cpu/src/pl.rs`
- `crates/backend_cpu/src/pl_histogram.rs`
- `crates/backend_cpu/src/backend_ops.rs`
- `crates/engine/src/traits.rs`
- `crates/engine/src/trainer/tree_build.rs`
- `bindings/python/tests/test_pl_trees.py`

## Self-Review

- Both PL accumulation paths now solve on standardized coordinates:
  - histogram-backed accumulation in `pl_histogram.rs`
  - direct partition solve in `pl.rs`
- PL leaves now persist scaler-derived means and inverse standard deviations from training, so prediction still evaluates correctly from raw feature rows.
- Tree builders now propagate split-path feature order explicitly, dedupe repeated path features, and stop at `MAX_PL_REGRESSORS`.
- Parent-relative PL weight deltas are now subtracted by matching feature id, which avoids corrupt deltas when parent and child regressor order differs.
- Old v1 linear-leaf coefficient sections remain readable because Task 2 did not change artifact decoding or the backward-compatibility path added in Task 1.

## Task 2 Review Fix

### Findings addressed

- Critical: level-wise and leaf-wise PL path propagation now exclude native-categorical split features, so descendant linear leaves only regress on numeric path features.
- Important: added focused Rust tests for the helper contract, for repeated-feature/cap behavior, and for descendant linear leaves under both `tree_growth="level"` and `tree_growth="leaf"` with a native categorical ancestor.

### TDD RED/GREEN evidence for new tests

#### RED

1. `cargo test -p alloygbm-engine linear_regressor_path_skips_categorical_split_features -- --nocapture`
   - failed to compile as expected because `linear_regressor_path_features` did not exist yet.
2. `cargo test -p alloygbm-engine linear_leaf_leaf_wise_descendant_excludes_categorical_ancestor_features -- --nocapture`
   - failed in the same red state before behavioral assertions ran; the new helper was still missing, and the new test backend still needed its `LinearLeaf` type reference wired in.

#### GREEN

1. `cargo test -p alloygbm-engine linear_regressor_ -- --nocapture`
   - result: `3 passed; 0 failed`
   - coverage: path-order preservation, `MAX_PL_REGRESSORS` cap, categorical-exclusion helper contract
2. `cargo test -p alloygbm-engine categorical_ancestor_features -- --nocapture`
   - result: `2 passed; 0 failed`
   - coverage: both level-wise and leaf-wise descendant linear leaves stay on numeric regressor paths after a native categorical ancestor split

### Exact focused commands and result summaries

- `cargo test -p alloygbm-engine linear_regressor_path_skips_categorical_split_features -- --nocapture`
  - red: compile failure, missing `linear_regressor_path_features`
- `cargo test -p alloygbm-engine linear_leaf_leaf_wise_descendant_excludes_categorical_ancestor_features -- --nocapture`
  - red: compile failure in the same new-test state
- `cargo fmt --all`
  - result: formatting applied cleanly
- `cargo test -p alloygbm-engine linear_regressor_ -- --nocapture`
  - green: all three helper tests passed
- `cargo test -p alloygbm-engine categorical_ancestor_features -- --nocapture`
  - green: both trainer-level categorical-ancestor tests passed

### Files changed

- `crates/engine/src/trainer/tree_build.rs`
- `crates/engine/src/tests/main.rs`
- `.superpowers/sdd/task-2-report.md`
