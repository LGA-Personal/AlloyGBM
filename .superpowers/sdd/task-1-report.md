# Task 1 Report: Scaled Linear-Leaf Representation and Artifact Compatibility

## Scope

Implemented Task 1 only:

- Added scaled linear-leaf metadata and helpers in core.
- Added `LinearFeatureScaler` and exported it from `alloygbm_core`.
- Bumped linear-leaf coefficient artifact payloads to v2 while preserving v1 decode compatibility.
- Updated predictor compact linear-leaf evaluation to use per-slot means and inverse standard deviations.
- Updated SHAP linear-leaf decomposition to operate on standardized slot values.
- Updated existing tests and literals affected by the `LinearLeaf` shape change.

## TDD Evidence

### Red

Added the two requested core tests in `crates/core/src/tests/main.rs`.

Initial red run failed for the wrong reason because internal crate tests cannot import `alloygbm_core`; I corrected the test imports and reran.

Then the intended red failure was observed:

- `cargo test -p alloygbm-core linear_leaf_scaled_eval_uses_standardized_coordinates_and_mean_imputes_nan -- --nocapture`
- `cargo test -p alloygbm-core linear_leaf_coefficients_v1_decode_uses_identity_scaling -- --nocapture`

Observed compiler failures:

- no function or associated item named `scaled` found for `LinearLeaf`
- no function or associated item named `identity_scaled` found for `LinearLeaf`
- no field `feature_means` on `LinearLeaf`
- no field `feature_inv_stds` on `LinearLeaf`

### Green

After implementation, the exact Task 1 Step 8 commands succeeded:

1. `cargo test -p alloygbm-core linear_leaf_scaled_eval_uses_standardized_coordinates_and_mean_imputes_nan -- --nocapture`
   - result: `1 passed; 0 failed`
2. `cargo test -p alloygbm-core linear_leaf_coefficients_v1_decode_uses_identity_scaling -- --nocapture`
   - result: `1 passed; 0 failed`
3. `cargo test -p alloygbm-predictor linear -- --nocapture`
   - result: `0 passed; 0 failed; 21 filtered out`
   - note: this command still verified the crate compiled successfully after the leaf shape change
4. `cargo test -p alloygbm-shap linear -- --nocapture`
   - result: `11 passed; 0 failed`

## Files Changed

- `crates/core/src/leaf.rs`
- `crates/core/src/linear_histogram.rs`
- `crates/core/src/lib.rs`
- `crates/core/src/artifact_format.rs`
- `crates/core/src/tests/main.rs`
- `crates/predictor/src/lib.rs`
- `crates/shap/src/linear_leaf.rs`
- `crates/shap/src/tests/main.rs`
- `crates/backend_cpu/src/pl.rs`

## Scope Exception

`crates/backend_cpu/src/pl.rs` required a compile-only update from direct `LinearLeaf { ... }` construction to `LinearLeaf::identity_scaled(...)`. This did not thread scaler state into training and did not implement Task 2 behavior, but it was necessary for the predictor-focused verification command to compile after the `LinearLeaf` shape change.

## Self-Review

- `LinearLeaf::eval` and predictor compact evaluation now both standardize raw coordinates with the same NaN/non-finite behavior.
- Artifact v2 writes means and inverse standard deviations; artifact v1 decode defaults to identity scaling.
- SHAP constant/deviation decomposition now uses `slot_value(...)`, so additivity remains aligned with predictor evaluation for scaled leaves.
- Existing SHAP linear-leaf fixtures were updated to explicit identity scaling to preserve prior behavior.

## Task 1 Review Fix

### Findings addressed

- Added a v2 linear-leaf coefficient round-trip test covering non-identity `feature_means` and `feature_inv_stds`.
- Added a predictor-focused artifact round-trip test that proves predictor evaluation uses scaled linear-leaf metadata.
- Added a SHAP scaled-leaf additivity case with non-identity per-slot scaling and a same-path standardized-delta assertion.
- Restored `leaf_constant_part` accumulation to `f64`.
- Updated stale comments in core/SHAP docs to describe standardized slot values instead of raw `w * x` / `w * μ`.

### TDD RED/GREEN evidence for new tests

Red came from adding the tests first, before touching `crates/shap/src/linear_leaf.rs`.

- `cargo test -p alloygbm-core linear_leaf_coefficients_v2_roundtrip_preserves_scaled_metadata -- --nocapture`
  - red result: `1 passed; 0 failed` (the v2 write/read path was already correct; coverage was missing)
- `cargo test -p alloygbm-predictor pl_tree_predictor_uses_scaled_linear_leaf_metadata_after_artifact_roundtrip -- --nocapture`
  - red result: `1 passed; 0 failed` (predictor logic was already correct; coverage was missing)
- `cargo test -p alloygbm-shap shap_scaled_linear_leaves_remain_additive -- --nocapture`
  - initial red result: failed because the first draft used a `NaN` row and SHAP correctly rejects non-finite inputs; test was revised to finite rows while keeping non-identity scaling
- `cargo test -p alloygbm-shap leaf_constant_part_accumulates_scaled_terms_in_f64 -- --nocapture`
  - red result: failed with `constant part: 0`, exposing the f32 accumulation regression

Green after the SHAP fix and comment updates:

- `cargo test -p alloygbm-core linear_leaf_coefficients_v2_roundtrip_preserves_scaled_metadata -- --nocapture`
  - result: `1 passed; 0 failed`
- `cargo test -p alloygbm-predictor pl_tree_predictor_uses_scaled_linear_leaf_metadata_after_artifact_roundtrip -- --nocapture`
  - result: `1 passed; 0 failed`
- `cargo test -p alloygbm-shap shap_scaled_linear_leaves_remain_additive -- --nocapture`
  - result: `1 passed; 0 failed`
- `cargo test -p alloygbm-shap leaf_constant_part_accumulates_scaled_terms_in_f64 -- --nocapture`
  - result: `1 passed; 0 failed`
- `cargo test --workspace`
  - result: passed (`475 passed; 0 failed` across workspace unit/doc tests)

### Exact focused commands and result summaries

- `cargo test -p alloygbm-core linear_leaf_coefficients_v2_roundtrip_preserves_scaled_metadata -- --nocapture`
  - verifies v2 payload round-trip preserves scaled metadata and leaf evaluation
- `cargo test -p alloygbm-predictor pl_tree_predictor_uses_scaled_linear_leaf_metadata_after_artifact_roundtrip -- --nocapture`
  - verifies artifact decode + predictor PL evaluation use stored means/inverse stds
- `cargo test -p alloygbm-shap shap_scaled_linear_leaves_remain_additive -- --nocapture`
  - verifies standardized linear-leaf SHAP values reconstruct predictor output
- `cargo test -p alloygbm-shap leaf_constant_part_accumulates_scaled_terms_in_f64 -- --nocapture`
  - verifies the constant-part helper preserves cancellation-sensitive scaled terms

### Files changed

- `.superpowers/sdd/task-1-report.md`
- `crates/core/src/artifact_format.rs`
- `crates/core/src/leaf.rs`
- `crates/core/src/tests/main.rs`
- `crates/predictor/src/lib.rs`
- `crates/shap/src/brute_force.rs`
- `crates/shap/src/linear_leaf.rs`
- `crates/shap/src/tests/main.rs`
- `crates/shap/src/tree_shap.rs`
