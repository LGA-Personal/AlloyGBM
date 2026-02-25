# v0.0.5 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.1/v0.0.5`
- Date: 2026-02-23

## Acceptance Criteria Matrix
- Criterion: `cargo fmt -- --check` passes.
- Evidence: Command exit status `0`.
- Status: PASS

- Criterion: `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: Command exit status `0` after addressing one `needless_lifetimes` warning in `crates/engine/src/lib.rs`.
- Status: PASS

- Criterion: `cargo test --workspace` passes.
- Evidence: Workspace unit and doc tests complete with all green.
- Status: PASS

- Criterion: Engine tests verify iterative controls can prevent stump addition when gain threshold is not met and when leaves would be under minimum row count.
- Evidence:
  - `fit_iterations_controls_enforce_min_split_gain`
  - `fit_iterations_controls_enforce_min_rows_per_leaf`
- Status: PASS

- Criterion: Engine tests verify artifact import rejects missing required `PredictorLayout` or `Trees` sections and rejects duplicate required section kinds.
- Evidence:
  - `trained_model_artifact_rejects_missing_required_sections`
  - `trained_model_artifact_rejects_duplicate_required_sections`
- Status: PASS

- Criterion: Engine artifact roundtrip still preserves prediction consistency with dual-section export/import.
- Evidence:
  - `trained_model_artifact_roundtrip_preserves_predictions`
  - test asserts both `Trees` and `PredictorLayout` sections are present in serialized artifact.
- Status: PASS

- Criterion: Python docstrings in drift-reported files no longer reference `v0.0.3`.
- Evidence:
  - `bindings/python/alloygbm/__init__.py`
  - `bindings/python/alloygbm/regressor.py`
  - `bindings/python/tests/test_regressor_contract.py`
- Status: PASS

## Gap Analysis (Test-Gap-Closer Pass)
- Reviewed each acceptance criterion in `docs/architecture/v1.0/v0.1/v0.0.5/plan.md` and mapped it to direct command/test/file evidence.
- Coverage result:
  - command criteria (`fmt`, `clippy`, `workspace test`) are directly evidenced by command exits.
  - behavior criteria are directly evidenced by targeted engine tests:
    - control guards: `fit_iterations_controls_enforce_min_split_gain`, `fit_iterations_controls_enforce_min_rows_per_leaf`
    - artifact section validation: `trained_model_artifact_rejects_missing_required_sections`, `trained_model_artifact_rejects_duplicate_required_sections`
    - dual-section roundtrip: `trained_model_artifact_roundtrip_preserves_predictions`
  - naming drift criterion is evidenced by zero matches for `v0.0.3` in the listed Python files.
- Gaps found: none.

## Residual Uncovered Criteria
- None. All `v0.0.5` acceptance criteria have direct evidence.

## Tests Added or Updated
- File: `crates/engine/src/lib.rs`
- Purpose: Add coverage for control guards, invalid controls, and stricter required artifact section validation.

- File: `bindings/python/alloygbm/__init__.py`
- Purpose: Update stale versioned docstring to remove `v0.0.3` naming drift.

- File: `bindings/python/alloygbm/regressor.py`
- Purpose: Update stale versioned docstring to remove `v0.0.3` naming drift.

- File: `bindings/python/tests/test_regressor_contract.py`
- Purpose: Update stale versioned test-module docstring to remove `v0.0.3` naming drift.

## Commands Executed
- Command: `cargo fmt -- --check`
- Result: PASS

- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS

- Command: `cargo test --workspace`
- Result: PASS
  - Notable counts:
    - `alloygbm-core`: 13 tests passed
    - `alloygbm-engine`: 13 tests passed

- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 7 tests`, `OK`)

- Command: `rg -n "v0\\.0\\.3" bindings/python/alloygbm/__init__.py bindings/python/alloygbm/regressor.py bindings/python/tests/test_regressor_contract.py`
- Result: PASS (no matches)

## Residual Risks
- Strict required dual-section import means older single-section artifacts are now rejected by engine import.
- Iterative trainer remains stump-level; depth-controlled policy is still deferred.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release: None for `v0.0.5` acceptance criteria.
