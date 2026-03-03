# AlloyGBM v0.9.1 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.9/v0.9.1`
- Date: 2026-03-03

## Acceptance Criteria Matrix
- Criterion: (1) `docs/architecture/v1.0/v0.9/v0.9.1/plan.md` is present and decision-complete.
- Evidence: [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/plan.md).
- Status: PASS

- Criterion: (2) `docs/architecture/v1.0/v0.9/v0.9.1/bug_triage.md` exists with prioritized issues, deterministic reproductions, and fix-to-test mapping.
- Evidence: [bug_triage.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/bug_triage.md).
- Status: PASS

- Criterion: (3) `scripts/benchmark_avx2_compare.sh` handles non-AVX2 environments without reporting misleading AVX2 delta percentages.
- Evidence:
  - script updated in [scripts/benchmark_avx2_compare.sh](/Users/lashby/Projects/AlloyGBM/scripts/benchmark_avx2_compare.sh),
  - command output shows both modes `runtime_avx2_enabled: false` and `medium_delta_vs_forced_scalar_median: n/a (runtime AVX2 unavailable)`.
- Status: PASS

- Criterion: (4) `docs/architecture/v1.0/v0.9/v0.9.1/implementation_notes.md` is present.
- Evidence: [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/implementation_notes.md).
- Status: PASS

- Criterion: (5) `docs/architecture/v1.0/v0.9/v0.9.1/verification_report.md` is present with criterion-to-evidence mapping.
- Evidence: this report.
- Status: PASS

- Criterion: (6) `cargo fmt -- --check` passes.
- Evidence: command executed successfully on 2026-03-03.
- Status: PASS

- Criterion: (7) `cargo clippy --workspace --all-targets -- -D warnings` passes.
- Evidence: command executed successfully on 2026-03-03.
- Status: PASS

- Criterion: (8) `cargo test --workspace` passes.
- Evidence: command executed successfully on 2026-03-03 (all workspace tests/doc-tests passed).
- Status: PASS

- Criterion: (9) `cargo doc --workspace --no-deps` passes.
- Evidence: command executed successfully on 2026-03-03.
- Status: PASS

- Criterion: (10) `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.
- Evidence: command executed successfully on 2026-03-03 (`Ran 71 tests`, `OK`).
- Status: PASS

- Criterion: (11) `docs/architecture/state/layer_index.yaml` marks `v0.9.1` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.2`.
- Evidence: [layer_index.yaml](/Users/lashby/Projects/AlloyGBM/docs/architecture/state/layer_index.yaml) updated in this slice.
- Status: PASS

## Tests Added or Updated
- Test source additions: none.
- Verification additions:
  - command-output assertion for AVX2 summary behavior in `scripts/benchmark_avx2_compare.sh`.
  - reproducibility evidence commands for deferred triage items (`BG-902`, `BG-903`).

## Criterion-to-Test Mapping
- Criteria 1-2 and 4-5: artifact presence/content verification.
- Criterion 3: script execution and summary-line assertions via `rg`.
- Criteria 6-10: standard Rust/Python gate reruns.
- Criterion 11: layer state index status and pointer checks.

## Commands Executed
- `cargo fmt -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo doc --workspace --no-deps` -> PASS
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`Ran 71 tests`, `OK`)
- `bash scripts/benchmark_avx2_compare.sh --runs 1 | tee /tmp/v091_avx2_after_fix.txt` -> PASS
- `rg -n \"runtime_avx2_enabled\\(default\\): false|runtime_avx2_enabled\\(forced_scalar\\): false|medium_delta_vs_forced_scalar_median: n/a \\(runtime AVX2 unavailable\\)\" /tmp/v091_avx2_after_fix.txt` -> PASS
- `rg -n -e \"--rounds 80\" docs/architecture/v1.0/v0.8/v0.8.3/benchmark_run_summary.md docs/architecture/v1.0/v0.8/v0.8.3/verification_report.md` -> PASS
- `rg -n -e \"CI-level regression thresholds\" -e \"Benchmark thresholds remain informational\" docs/architecture/v1.0/v0.8/implementation_notes.md docs/architecture/v1.0/v0.8/verification_report.md` -> PASS

## Residual Uncovered Criteria
- None.

## Residual Risks
- `BG-902`: shallow/deep benchmark matrix is not yet implemented (`v0.9.2` follow-up).
- `BG-903`: benchmark threshold policy is still not enforced as CI pass/fail (`v0.9.2`/`v0.9.7` follow-up).

## Final Readiness
- Ready: Yes
- Required follow-up before milestone closeout:
  - execute `docs/architecture/v1.0/v0.9/v0.9.2` for benchmark expansion and policy hardening progression.
