# AlloyGBM v0.9.4 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.4`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.4/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.4/plan.md).
- Status: PASS

- Criterion: (2) Benchmark runner validates Alloy runtime contract before benchmark execution.
- Evidence:
  - [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) includes `_verify_alloygbm_runtime_contract` and `_load_alloygbm_runtime`.
  - runtime check is executed in `main` before dataset loading and benchmark loops.
- Status: PASS

- Criterion: (3) Benchmark runner validates native training symbol presence before benchmark execution.
- Evidence:
  - `_verify_alloygbm_runtime_contract` checks for native `train_regression_artifact` and raises `RuntimeError` if absent.
  - contract behavior covered by [test_alloygbm_runtime_contract.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_alloygbm_runtime_contract.py).
- Status: PASS

- Criterion: (4) Incompatible runtime now fails fast with actionable error messaging.
- Evidence:
  - command: `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7 --scenarios dense_numeric`
  - result: exit code `2` with message:
    - `alloygbm runtime check failed: loaded alloygbm.GBMRegressor is not benchmark-compatible; missing required __init__ parameters: col_subsample, n_estimators, row_subsample`
- Status: PASS

- Criterion: (5) Benchmark output metadata includes Alloy runtime provenance details.
- Evidence:
  - [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) records `alloygbm_runtime` under emitted `params` payload (`module_path`, `native_module_path`, `init_parameters`) after successful runtime validation.
- Status: PASS

- Criterion: (6) Regression tests cover runtime contract pass/fail checks.
- Evidence:
  - [test_alloygbm_runtime_contract.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_alloygbm_runtime_contract.py) includes:
    - compatible runtime acceptance,
    - missing constructor-parameter rejection,
    - missing native symbol rejection.
  - `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 7 tests`).
- Status: PASS

- Criterion: (7) `benchmarks/README.md` documents runtime contract guard behavior.
- Evidence: [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) includes runtime contract/fail-fast section.
- Status: PASS

- Criterion: (8) `docs/architecture/v1.0/v0.9/v0.9.4/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.4/implementation_notes.md).
- Status: PASS

- Criterion: (9) `docs/architecture/v1.0/v0.9/v0.9.4/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

- Criterion: (10) `cargo fmt -- --check` and benchmark test suite command pass.
- Evidence:
  - `cargo fmt -- --check` -> PASS
  - `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS
- Status: PASS

- Criterion: (11) `docs/architecture/state/layer_index.yaml` marks `v0.9.4` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.5` (with `v0.9.6` queued).
- Evidence: [layer_index.yaml](/Users/lashby/Projects/AlloyGBM/docs/architecture/state/layer_index.yaml) updated in this slice.
- Status: PASS

## Tests Added or Updated
- Added:
  - [benchmarks/tests/test_alloygbm_runtime_contract.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_alloygbm_runtime_contract.py)
- Updated:
  - [benchmarks/run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py)
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md)

## Criterion-to-Test Mapping
- Criteria 2-3 and 6: runtime contract helper + dedicated benchmark tests.
- Criterion 4: stale-runtime fail-fast command execution.
- Criteria 7-9: documentation/layer artifact checks.
- Criteria 10-11: formatting gate + layer state progression evidence.

## Commands Executed
- `python3 -m py_compile benchmarks/run_model_comparison.py benchmarks/tests/test_temporal_leakage.py benchmarks/tests/test_alloygbm_runtime_contract.py` -> PASS
- `python3 -B benchmarks/run_model_comparison.py --help` -> PASS
- `cargo fmt -- --check` -> PASS
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 7 tests`, `OK`)
- `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7 --scenarios dense_numeric` -> EXPECTED FAIL (`exit=2`, runtime contract guard triggered)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Benchmark runs will remain blocked until benchmark environments install a compatible Alloy runtime package variant.
- Existing benchmark artifacts produced before this guard may reflect stale runtime behavior and should be treated as provisional.

## Final Readiness
- Ready: Yes
- Required follow-up before milestone closeout:
  - execute `docs/architecture/v1.0/v0.9/v0.9.5` for benchmark competitiveness improvements,
  - execute `docs/architecture/v1.0/v0.9/v0.9.6` for docs/tutorial and parent closeout readiness.
