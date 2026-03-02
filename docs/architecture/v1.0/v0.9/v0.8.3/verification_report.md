# AlloyGBM v0.8.3 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.8.3`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) Organized benchmark workspace is present with `dense_numeric`, `panel_time_series`, and `histogram_stress` scenario folders.
- Evidence: [benchmarks](/Users/lashby/Projects/AlloyGBM/benchmarks) includes all required scenario directories.
- Status: PASS

- Criterion: (2) Each scenario folder includes `manifest.yaml` and `prepare.py`.
- Evidence:
  - [benchmarks/dense_numeric/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/dense_numeric/manifest.yaml), [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/dense_numeric/prepare.py)
  - [benchmarks/panel_time_series/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/manifest.yaml), [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/prepare.py)
  - [benchmarks/histogram_stress/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/histogram_stress/manifest.yaml), [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/histogram_stress/prepare.py)
- Status: PASS

- Criterion: (3) `dense_numeric` and `panel_time_series` preparation scripts use UCI direct download URLs with no-auth flow.
- Evidence:
  - code-level: both scripts reference `https://archive.ics.uci.edu/ml/machine-learning-databases/...` URLs and implement no-auth download logic with urllib/curl/wget fallback.
  - run-level: both UCI-backed prepare commands executed successfully in this verification pass (see Commands Executed and `benchmark_run_summary.md`).
- Status: PASS

- Criterion: (4) `histogram_stress` preparation script produces deterministic synthetic data from seed-controlled generation.
- Evidence: [benchmarks/histogram_stress/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/histogram_stress/prepare.py) takes `--seed` and uses `random.Random(seed)` for deterministic generation.
- Status: PASS

- Criterion: (5) `docs/architecture/v1.0/v0.9/v0.8.3/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.8.3/implementation_notes.md).
- Status: PASS

- Criterion: (6) `docs/architecture/v1.0/v0.9/v0.8.3/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

- Criterion: (7) `cargo fmt -- --check` passes.
- Evidence: command executed successfully on 2026-03-02.
- Status: PASS

- Criterion: (8) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully on 2026-03-02.
- Status: PASS

- Criterion: (9) `cargo test --workspace` passes.
- Evidence: command executed successfully on 2026-03-02.
- Status: PASS

- Criterion: (10) `cargo doc --workspace --no-deps` passes.
- Evidence: command executed successfully on 2026-03-02.
- Status: PASS

- Criterion: (11) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command executed successfully on 2026-03-02 (`Ran 71 tests`, `OK`).
- Status: PASS

- Criterion: (12) `docs/architecture/state/layer_index.yaml` marks `v0.8.3` verified and advances next target to `docs/architecture/v1.0/v0.9/v0.8.4`.
- Evidence: [layer_index.yaml](/Users/lashby/Projects/AlloyGBM/docs/architecture/state/layer_index.yaml) updated in this slice.
- Status: PASS

## Tests Added or Updated
- File: N/A (no unit/integration test suite additions in this slice).
- Purpose: Benchmark reproducibility infrastructure and script-level preparation entrypoints were added; verification used script syntax/CLI smoke checks, cross-package comparison runs, and full repository gates.

## Criterion-to-Test Mapping
- Criterion 1-2: repository structure checks (`ls -R benchmarks`) + file presence evidence.
- Criterion 3: static script URL/download-flow inspection + attempted live UCI runs.
- Criterion 4: script logic inspection + executed synthetic generation command.
- Additional comparison evidence: cross-package run via `benchmarks/run_model_comparison.py` with persisted output artifacts.
- Criterion 5-6: artifact presence checks for implementation/verification docs.
- Criterion 7-11: command gate results from Rust and Python verification suite.
- Criterion 12: state index path/status verification via `layer_index.yaml`.

## Commands Executed
- Command: `python3 -m py_compile benchmarks/dense_numeric/prepare.py benchmarks/panel_time_series/prepare.py benchmarks/histogram_stress/prepare.py`
- Result: PASS
- Command: `python3 benchmarks/dense_numeric/prepare.py --help`
- Result: PASS
- Command: `python3 benchmarks/panel_time_series/prepare.py --help`
- Result: PASS
- Command: `python3 benchmarks/histogram_stress/prepare.py --help`
- Result: PASS
- Command: `python3 benchmarks/histogram_stress/prepare.py --rows 10 --features 4 --output-dir /tmp/alloy_hist_smoke`
- Result: PASS (wrote sample prepared csv)
- Command: `python3 -B benchmarks/dense_numeric/prepare.py --force-download --output-dir benchmarks/data/dense_numeric`
- Result: PASS (prepared dataset written to `benchmarks/data/dense_numeric/prepared/prepared.csv`)
- Command: `python3 -B benchmarks/panel_time_series/prepare.py --force-download --max-rows 50000 --output-dir benchmarks/data/panel_time_series`
- Result: PASS (prepared dataset written to `benchmarks/data/panel_time_series/prepared/prepared.csv`, rows=7024)
- Command: `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 80`
- Result: PASS (wrote `benchmarks/results/model_comparison_latest.{csv,json,md}` and timestamped outputs)
- Command: `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
- Result: PASS (benchmark timings produced; see `benchmark_run_summary.md`)
- Command: `bash scripts/benchmark_avx2_compare.sh --runs 1`
- Result: PASS (median comparison summary produced; see `benchmark_run_summary.md`)
- Command: `cargo fmt -- --check`
- Result: PASS
- Command: `cargo clippy --workspace --all-targets -- -D warnings`
- Result: PASS
- Command: `cargo test --workspace`
- Result: PASS
- Command: `cargo doc --workspace --no-deps`
- Result: PASS
- Command: `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- Result: PASS (`Ran 71 tests`, `OK`)

## Residual Uncovered Criteria
- None.

## Residual Risks
- Benchmark scripts rely on external network availability at execution time for UCI downloads.
- Benchmark result aggregation/baselines are not yet enforced by CI.

## Final Readiness
- Ready: Yes
- Required follow-up before merge/release:
  - execute `v0.8.4` migration/compatibility narrative slice and parent `v0.9` rollup.
