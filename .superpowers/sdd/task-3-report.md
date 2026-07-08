# Task 3 Report

## Scope

Updated the owned user docs and review resolution notes to match the implemented
piecewise-linear leaf behavior from Tasks 1 and 2:

- `docs/user/gbmregressor.md`
- `docs/user/quickstart.md`
- `docs/site/source/estimator.rst`
- `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md`
- `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`

No Rust or Python runtime behavior was changed.

## Changes made

### User docs

- Updated the primary `leaf_model="linear"` description to state that each leaf
  uses standardized split-path regressor features, not raw `x_j` values.
- Documented that regressor selection comes from the distinct numeric features
  on the leaf's root-to-leaf split path.
- Documented the internal cap `MAX_PL_REGRESSORS = 8`.
- Updated the recommended `lambda_l2` guidance to reflect internal
  standardization and remaining per-leaf solve sensitivity.
- Added the NaN regressor-value semantics: NaNs contribute the mean-imputed
  standardized value (`z_j = 0`) to the linear term while split routing still
  uses AlloyGBM missing-value direction.
- Updated the quickstart PL section to mention path features, the cap of 8, and
  internal standardization.

### Review resolution notes

- Marked core review §1.5 as fixed in the pending v0.12.11 PR, with the
  path-based regressor-selection tests recorded.
- Marked special-modes review §1.3 as partially fixed in the pending v0.12.11
  PR, covering:
  - internal standardization during PL solves
  - v2 `LinearLeafCoefficients` scale metadata
  - v1 artifact compatibility via identity scaling
  - standardized evaluation for prediction / SHAP / artifact loading
  - path-based PL regressor selection
- Preserved the remaining open items called out in the task brief:
  f64 selected-child accumulation, explicit condition guards, PL histogram
  memory reductions, and raw-scale benchmark CI guardrails.

## Focused verification run

Ran the Task 3 Step 3 focused tests only, per the task instruction to avoid the
full final matrix in this subtask:

```text
cargo test -p alloygbm-core linear_leaf -- --nocapture
  -> 3 passed

cargo test -p alloygbm-predictor linear -- --nocapture
  -> 1 passed

cargo test -p alloygbm-shap linear -- --nocapture
  -> 12 passed

.venv/bin/python -m pytest bindings/python/tests/test_pl_trees.py -q
  -> 18 passed
```

Observed during verification:

- Existing compile warning in `crates/engine/src/trainer/tree_build.rs`:
  `linear_regressor_features_for_split` is currently unused.
- Initial cargo invocations briefly waited on existing package/build locks, then
  completed successfully.

## Self-review

- Scope stayed within the five owned documentation files plus this report.
- The docs now match the implemented behavior described in the task brief:
  standardized PL coordinates, path-based numeric regressor selection, the
  internal regressor cap, native categorical descendant behavior, and NaN
  semantics.
- No constructor/API surface changes were introduced.
- No artifact-format or compatibility code was touched.

## Not run

Did not run:

- `cargo test --workspace`
- `maturin develop --release`
- `.venv/bin/python -m pytest bindings/python/tests/ -q`

Reason: the user task explicitly said to run only the focused doc-adjacent
tests if feasible and to leave the full final matrix for the controller after
all reviews.
