# AlloyGBM v0.8 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.8`
- Date: 2026-03-02

## Parent Acceptance Criteria Matrix
- Criterion: (1) `crates/shap` no longer returns placeholder `NotImplemented` for supported CPU regression artifacts.
- Evidence:
  - artifact-backed SHAP runtime delivered in `v0.7.1`/`v0.7.2` with exact math in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs),
  - verified in [v0.7.2 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md).
- Status: PASS

- Criterion: (2) Per-row SHAP output dimensionality is `rows x feature_count` and validates input shapes deterministically.
- Evidence:
  - row-shape and validation tests in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs),
  - documented in [v0.7.1 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.1/verification_report.md) and maintained through `v0.7.4`.
- Status: PASS

- Criterion: (3) For verified fixtures, `expected_value + sum(phi_i)` matches predictor output within documented tolerance.
- Evidence:
  - Rust predictor parity checks in `v0.7.3` (`tests::explain_rows_from_artifact_matches_predictor_predictions`) documented in [v0.7.3 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.3/verification_report.md),
  - Python/native additivity checks in `v0.7.4` documented in [v0.7.4 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md).
- Status: PASS

- Criterion: (4) Global importance is computed from mean absolute SHAP contribution and exposed in deterministic ordering.
- Evidence:
  - Rust aggregation logic/tests in [crates/shap/src/lib.rs](/Users/lashby/Projects/AlloyGBM/crates/shap/src/lib.rs),
  - deterministic tie-ordering coverage in `v0.7.3`,
  - Python exposure and runtime checks in `v0.7.4`.
- Status: PASS

- Criterion: (5) SHAP artifact loading honors strict/legacy required-section compatibility checks without model-format changes.
- Evidence:
  - strict/legacy/malformed compatibility coverage in `v0.7.3`,
  - corresponding evidence in [v0.7.3 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.3/verification_report.md).
- Status: PASS

- Criterion: (6) Python exposes additive SHAP APIs without regressing existing fit/predict contract tests.
- Evidence:
  - Python bridge and regressor SHAP APIs delivered in `v0.7.4`:
    - [bindings/python/src/lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs)
    - [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py)
  - contract/runtime verification in [v0.7.4 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md).
- Status: PASS

- Criterion: (7) `v0.8` child slices and parent rollup artifacts are present and linked in verification.
- Evidence:
  - verified child slices:
    - [v0.7.1](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.1/verification_report.md)
    - [v0.7.2](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.2/verification_report.md)
    - [v0.7.3](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.3/verification_report.md)
    - [v0.7.4](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/v0.7.4/verification_report.md)
  - parent rollup artifacts:
    - [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/implementation_notes.md)
    - [verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.8/verification_report.md)
- Status: PASS

- Criterion: (8) `cargo fmt -- --check` passes at closeout.
- Evidence: command executed successfully on 2026-03-02 during `v0.7.5` closeout.
- Status: PASS

- Criterion: (9) `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
- Evidence: command executed successfully on 2026-03-02 during `v0.7.5` closeout.
- Status: PASS

- Criterion: (10) `cargo test --workspace` and Python unittest suite pass at closeout.
- Evidence:
  - `cargo test --workspace`: PASS on 2026-03-02,
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`: PASS (`Ran 67 tests`, `OK`) on 2026-03-02.
- Status: PASS

## Commands Executed for Parent Closeout
- `cargo fmt -- --check`: PASS
- `cargo clippy --workspace --all-targets -- -D warnings`: PASS
- `cargo test --workspace`: PASS
- `cargo doc --workspace --no-deps`: PASS
- `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`: PASS (`Ran 67 tests`, `OK`)

## Residual Risks
- Exact SHAP enumeration remains bounded by split-feature guardrail (`MAX_EXACT_SPLIT_FEATURES = 20`) and returns deterministic contract violations beyond that threshold.
- SHAP expected-value/global-importance behavior remains dataset-dependent by design.

## Final Readiness
- Ready: Yes
- Release recommendation: mark `v0.8` complete and proceed to planning `v0.9`.
