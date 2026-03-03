# AlloyGBM v0.9.5 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.5`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.5/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.5/plan.md).
- Status: PASS

- Criterion: (2) Alloy native training accepts continuous float features for benchmark scenarios without `integer-valued bin` failures.
- Evidence:
  - Rust test `train_bridge_accepts_continuous_float_rows` in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs) passes.
  - Benchmark command:
    - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`
    - output artifact: `benchmarks/results/model_comparison_20260303T161814Z.json`
    - Alloy rows in this run are `PASS` (no `integer-valued bin` failures).
- Status: PASS

- Criterion: (3) Deterministic quantization/binning bridge exists for continuous features.
- Evidence:
  - Quantization bridge implemented in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs).
  - Rust test `train_bridge_quantization_is_deterministic_for_continuous_rows` passes.
  - Direct native call check confirmed deterministic artifact bytes for repeated continuous-input training.
- Status: PASS

- Criterion: (4) Integer-bin compatibility path remains functional.
- Evidence:
  - pre-binned strict path retained in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs), including overflow guard.
  - Rust test `train_bridge_pre_binned_path_rejects_u16_overflow` passes.
  - existing bridge parity tests remain PASS.
- Status: PASS

- Criterion: (5) Regression tests cover float acceptance and integer-bin backward compatibility.
- Evidence:
  - Rust tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
    - `train_bridge_accepts_continuous_float_rows`
    - `train_bridge_quantization_is_deterministic_for_continuous_rows`
    - `train_bridge_pre_binned_path_rejects_u16_overflow`
    - `train_bridge_large_round_counts_remain_supported_via_round_cap`
  - Python tests in [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py):
    - `test_fit_quantizes_continuous_rows_before_native_training`
    - `test_predict_quantizes_rows_when_model_fitted_on_continuous_inputs`
- Status: PASS

- Criterion: (6) `docs/architecture/v1.0/v0.9/v0.9.5/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.5/implementation_notes.md).
- Status: PASS

- Criterion: (7) `docs/architecture/v1.0/v0.9/v0.9.5/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

## Tests Added or Updated
- Updated:
  - [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs)
  - [regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py)
  - [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py)

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 73 tests`, `OK`)
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 7 tests`, `OK`)
- `python3 -m pip install -e .` -> PASS
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7 --scenarios dense_numeric dow_jones_financial` -> PASS (artifact set `20260303T161814Z`)

## Residual Risks
- Continuous-feature bridge uses coarse bounded quantization (`0..255`) and is correctness-oriented; competitiveness tuning is deferred.
- Round cap (`4096`) prevents overflow but truncates higher requested rounds until encoding architecture is expanded.

## Final Readiness
- Ready: Yes (for `v0.9.5` scope)
- Required follow-up before milestone closeout:
  - execute `docs/architecture/v1.0/v0.9/v0.9.6` for split/depth semantics and sensitivity validation,
  - execute `docs/architecture/v1.0/v0.9/v0.9.7` for competitiveness/policy hardening,
  - execute `docs/architecture/v1.0/v0.9/v0.9.8` for docs/tutorial and parent closeout.
