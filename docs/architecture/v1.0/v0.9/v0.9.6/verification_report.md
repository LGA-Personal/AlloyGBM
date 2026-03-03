# AlloyGBM v0.9.6 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.6`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.6/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.6/plan.md).
- Status: PASS

- Criterion: (2) Continuous-feature split/hist training path is validated on benchmark scenarios.
- Evidence:
  - benchmark diagnostics command:
    - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --scenarios dense_numeric dow_jones_financial`
  - artifact output:
    - `benchmarks/results/model_comparison_20260303T162728Z.json`
  - result summary:
    - total rows `54`, status `54 PASS`,
    - Alloy rows `18 PASS`, `0 FAIL`.
- Status: PASS

- Criterion: (3) Parameter-sensitivity diagnostics show meaningful depth/round effects.
- Evidence (from `model_comparison_20260303T162728Z.json`, Alloy rows):
  - `dense_numeric` profile means:
    - `shallow_high_lr`: `rmse_mean=0.752635`, `fit_mean=0.0448s`, `pred_mean=0.0157s`
    - `mid_balanced`: `rmse_mean=0.725325`, `fit_mean=0.5426s`, `pred_mean=0.5237s`
    - `deep_low_lr`: `rmse_mean=0.697629`, `fit_mean=4.0899s`, `pred_mean=8.4731s`
    - deep vs shallow deltas: `rmse=-0.055006`, `fit_ratio=91.34x`, `pred_ratio=538.72x`
  - `dow_jones_financial` profile means:
    - `shallow_high_lr`: `rmse_mean=3.887671`, `fit_mean=0.0302s`, `pred_mean=0.0071s`
    - `mid_balanced`: `rmse_mean=3.749126`, `fit_mean=0.4003s`, `pred_mean=0.1961s`
    - `deep_low_lr`: `rmse_mean=3.669329`, `fit_mean=1.7162s`, `pred_mean=1.2674s`
    - deep vs shallow deltas: `rmse=-0.218342`, `fit_ratio=56.88x`, `pred_ratio=178.0x`
- Status: PASS

- Criterion: (4) Regression tests cover depth/round behavioral sensitivity.
- Evidence:
  - Added tests in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py):
    - `test_continuous_dense_profile_depth_rounds_change_capacity`
    - `test_continuous_low_snr_financial_profiles_show_nontrivial_effects`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 75 tests`).
- Status: PASS

- Criterion: (5) `docs/architecture/v1.0/v0.9/v0.9.6/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.6/implementation_notes.md).
- Status: PASS

- Criterion: (6) `docs/architecture/v1.0/v0.9/v0.9.6/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

## Tests Added or Updated
- Updated:
  - [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py)
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md)

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 75 tests`, `OK`)
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 7 tests`, `OK`)
- `python3 -m pip install -e .` -> PASS
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --scenarios dense_numeric dow_jones_financial` -> PASS (artifacts `20260303T162728Z`)

## Residual Risks
- Low-SNR financial scenarios remain noisy by nature; profile conclusions should continue to use multi-seed medians.
- Quantization fidelity tradeoffs remain an optimization concern for `v0.9.7` rather than a `v0.9.6` correctness blocker.

## Final Readiness
- Ready: Yes
- Required follow-up before milestone closeout:
  - execute `docs/architecture/v1.0/v0.9/v0.9.7` for competitiveness and benchmark policy hardening,
  - execute `docs/architecture/v1.0/v0.9/v0.9.8` for docs/tutorial and parent closeout.
