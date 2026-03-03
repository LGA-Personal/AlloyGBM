# AlloyGBM v0.9.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.3`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.3/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.3/plan.md).
- Status: PASS

- Criterion: (2) `panel_time_series` prepared dataset target is future-horizon and not globally identical to same-row `co_gt`.
- Evidence:
  - [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/prepare.py) now sets `target_co_gt` from next strictly later timestamp.
  - `python3 -B benchmarks/panel_time_series/prepare.py --force-download --output-dir benchmarks/data/panel_time_series` -> PASS (`rows=7023`, `dropped_no_future_target=1`).
  - data check: `panel_exact_feature_target_match_ratio=0.0938345436423181` (no longer `1.0`).
- Status: PASS

- Criterion: (3) Timestamped benchmark scenarios split with zero timestamp overlap between train/test.
- Evidence:
  - [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) uses `_split_by_timestamp` with unique timestamp boundary split.
  - data checks:
    - `panel_overlap_timestamps=0`
    - `dow_overlap_timestamps=0`
- Status: PASS

- Criterion: (4) Benchmark runner rejects target-equivalent feature leakage with explicit error.
- Evidence:
  - [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) now raises `ValueError` for target-equivalent features.
  - regression test `test_split_dataset_rejects_target_equivalent_feature` passes in [test_temporal_leakage.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_temporal_leakage.py).
- Status: PASS

- Criterion: (5) Regression tests exist for future-target prep, timestamp split integrity, and leakage guard behavior.
- Evidence:
  - [test_temporal_leakage.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_temporal_leakage.py) includes 4 tests covering target horizon, timestamp split, leakage guard, and Dow Jones safeguard fields.
  - `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 4 tests`, `OK`).
- Status: PASS

- Criterion: (6) `benchmarks/README.md` and scenario notes document temporal leakage safeguards.
- Evidence:
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) includes temporal leakage safeguards section.
  - [benchmarks/panel_time_series/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/manifest.yaml) note updated for next-timestep target semantics.
- Status: PASS

- Criterion: (7) `docs/architecture/v1.0/v0.9/v0.9.3/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.3/implementation_notes.md).
- Status: PASS

- Criterion: (8) `docs/architecture/v1.0/v0.9/v0.9.3/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

- Criterion: (9) Standard Rust/Python non-regression commands pass.
- Evidence:
  - `cargo fmt -- --check` -> PASS
  - `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
  - `cargo test --workspace` -> PASS
  - `cargo doc --workspace --no-deps` -> PASS
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 71 tests`, `OK`)
- Status: PASS

- Criterion: (10) `docs/architecture/state/layer_index.yaml` marks `v0.9.3` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.4`.
- Evidence: [layer_index.yaml](/Users/lashby/Projects/AlloyGBM/docs/architecture/state/layer_index.yaml) updated in this slice.
- Status: PASS

## Tests Added or Updated
- Added:
  - [benchmarks/tests/test_temporal_leakage.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_temporal_leakage.py)
- Updated:
  - [benchmarks/panel_time_series/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/prepare.py)
  - [benchmarks/run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py)

## Criterion-to-Test Mapping
- Criteria 2-4: panel preparation execution + runner split/guard logic + regression tests.
- Criterion 5: benchmark test suite command pass.
- Criterion 6: documentation file updates.
- Criteria 9-10: repository gates + layer index transition evidence.

## Commands Executed
- `python3 -m py_compile benchmarks/panel_time_series/prepare.py benchmarks/dow_jones_financial/prepare.py benchmarks/run_model_comparison.py benchmarks/tests/test_temporal_leakage.py` -> PASS
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`Ran 4 tests`, `OK`)
- `python3 -B benchmarks/panel_time_series/prepare.py --force-download --output-dir benchmarks/data/panel_time_series` -> PASS (`rows=7023`, `dropped_no_future_target=1`)
- `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial` -> PASS (`kept_rows=720`, `dropped_rows=30`, `fallback_targets=0`)
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7 --scenarios panel_time_series dow_jones_financial` -> PASS (all scenario/profile/model rows PASS; outputs include `model_comparison_20260303T085134Z.*` and `model_comparison_profile_summary_20260303T085134Z.*`)
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 71 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Dow Jones target remains low-SNR and naturally noisy, so quality metrics remain volatile even without leakage.
- Timestamp-boundary split is stricter than previous behavior; historical benchmark values are not directly comparable without noting this change.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release:
  - execute `docs/architecture/v1.0/v0.9/v0.9.4` for benchmark runtime provenance hardening,
  - execute `docs/architecture/v1.0/v0.9/v0.9.5` and `docs/architecture/v1.0/v0.9/v0.9.6` for native continuous-feature training support,
  - execute `docs/architecture/v1.0/v0.9/v0.9.7` for competitiveness/policy improvements on the corrected harness,
  - execute `docs/architecture/v1.0/v0.9/v0.9.8` for documentation/tutorial and closeout readiness packaging.
