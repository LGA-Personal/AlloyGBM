# AlloyGBM v0.9.2 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.2`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.2/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.2/plan.md).
- Status: PASS

- Criterion: (2) `benchmarks/dow_jones_financial/manifest.yaml` and `prepare.py` exist and produce a deterministic prepared dataset.
- Evidence:
  - [manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/dow_jones_financial/manifest.yaml) and [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/dow_jones_financial/prepare.py) are present.
  - `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial` -> PASS (`kept_rows=720`, `dropped_rows=30`, `fallback_targets=0`) with stable ordered output at `benchmarks/data/dow_jones_financial/prepared/prepared.csv`.
- Status: PASS

- Criterion: (3) Dow Jones scenario preprocessing is leakage-aware with explicit target and excluded feature rules documented in code/comments and notes.
- Evidence:
  - [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/dow_jones_financial/prepare.py) emits `target_percent_change_next_weeks_price` and only current-week feature columns in `OUTPUT_FIELDS`; leakage-prone future columns are not emitted as model features.
  - fallback target behavior is explicit in `_row_target_percent_change`.
  - [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.2/implementation_notes.md) documents target/fallback behavior.
- Status: PASS

- Criterion: (4) `run_model_comparison.py` supports benchmark profile-matrix execution while preserving single-profile backward compatibility.
- Evidence:
  - [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) provides `--profile-grid`/`--profile` matrix options and keeps `--learning-rate`, `--max-depth`, `--rounds` single-profile path (`profile_grid=none`).
  - `python3 benchmarks/run_model_comparison.py --help` -> PASS.
- Status: PASS

- Criterion: (5) Benchmark outputs include profile metadata and per-profile records for all requested models.
- Evidence:
  - raw output schema includes `profile_name`, `profile_index`, `run_index`, `seed`, and profile hyperparameters (see [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py)).
  - run artifacts include profile-aware records for AlloyGBM/LightGBM/XGBoost: `benchmarks/results/model_comparison_20260303T083035Z.{csv,json,md}` and `benchmarks/results/model_comparison_20260303T083104Z.{csv,json,md}`.
- Status: PASS

- Criterion: (6) At least three default profiles (shallow/mid/deep) run successfully across AlloyGBM, LightGBM, and XGBoost.
- Evidence:
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7` -> PASS.
  - [model_comparison_20260303T083035Z.md](/Users/lashby/Projects/AlloyGBM/benchmarks/results/model_comparison_20260303T083035Z.md) contains all three profiles and all three models with `PASS` rows.
- Status: PASS

- Criterion: (7) Optional ultra profile (`10000` rounds, very low learning rate) is executable and documented with runtime caveats.
- Evidence:
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial` -> PASS.
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) documents the optional ultra command path on constrained scenarios.
- Status: PASS

- Criterion: (8) Updated benchmark summary artifact for `v0.9.2` reports profile-level best-quality and best-speed outcomes by scenario.
- Evidence: [benchmark_run_summary.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.2/benchmark_run_summary.md) added with scenario-level best RMSE and fastest-fit outcomes for default and ultra runs.
- Status: PASS

- Criterion: (9) `docs/architecture/v1.0/v0.9/v0.9.2/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.2/implementation_notes.md).
- Status: PASS

- Criterion: (10) `docs/architecture/v1.0/v0.9/v0.9.2/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

- Criterion: (11) Standard Rust/Python gate commands pass (`fmt`, `clippy`, `test`, `doc`, Python unittest discovery).
- Evidence:
  - `cargo fmt -- --check` -> PASS
  - `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
  - `cargo test --workspace` -> PASS
  - `cargo doc --workspace --no-deps` -> PASS
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 71 tests`, `OK`)
- Status: PASS

- Criterion: (12) `docs/architecture/state/layer_index.yaml` marks `v0.9.2` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.3`.
- Evidence: [layer_index.yaml](/Users/lashby/Projects/AlloyGBM/docs/architecture/state/layer_index.yaml) updated in this verification slice.
- Status: PASS

## Tests Added or Updated
- Test source additions: none.
- Verification additions:
  - profile-matrix benchmark executions (`default` and `default_ultra`) with persisted timestamped outputs.
  - finance scenario preparation command evidence and profile-summary artifact validation.

## Criterion-to-Test Mapping
- Criteria 1, 9, 10: layer artifact presence and content checks.
- Criteria 2-3: Dow Jones scenario code-path review + preparation command execution.
- Criteria 4-7: benchmark runner CLI compatibility checks + matrix/ultra command runs.
- Criterion 8: layer-local benchmark summary artifact linked to run outputs.
- Criterion 11: standard Rust/Python gate reruns.
- Criterion 12: layer index status/target updates.

## Commands Executed
- `python3 -m py_compile benchmarks/dow_jones_financial/prepare.py benchmarks/run_model_comparison.py` -> PASS
- `python3 benchmarks/dow_jones_financial/prepare.py --help` -> PASS
- `python3 benchmarks/run_model_comparison.py --help` -> PASS
- `python3 -B benchmarks/dow_jones_financial/prepare.py --force-download --output-dir benchmarks/data/dow_jones_financial` -> PASS
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7` -> PASS
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial` -> PASS
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 71 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Benchmark preparation depends on network availability for UCI data downloads.
- Ultra-profile runs are intentionally constrained because runtime grows quickly at `rounds=10000`.
- CI hard-fail benchmark thresholds remain deferred to later `v0.9` layers.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release:
  - plan and execute `docs/architecture/v1.0/v0.9/v0.9.3` temporal leakage hardening follow-up scope,
  - execute `docs/architecture/v1.0/v0.9/v0.9.4` runtime provenance hardening,
  - continue competitiveness and policy/doc closeout in `v0.9.5` and `v0.9.6`.
