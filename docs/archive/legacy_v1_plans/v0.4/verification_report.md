# AlloyGBM v0.4 Verification Report

## Layer
- Path: `docs/architecture/v1.0/v0.4`
- Date: 2026-03-02

## Acceptance Criteria Matrix
- Criterion: (1) `v0.4` child slices collectively deliver benchmark harness coverage for backend CPU histogram/split hot paths with reproducible fixture definitions.
- Evidence:
  - Harness established in `v0.4.1` and expanded in `v0.4.2` with tiny/small/medium histogram plus split coverage.
  - Runtime-context benchmark output and comparison scripting added in `v0.4.4`.
  - Child reports:
    - [v0.4.1 verification](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.1/verification_report.md)
    - [v0.4.2 verification](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.2/verification_report.md)
    - [v0.4.3 verification](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.3/verification_report.md)
    - [v0.4.4 verification](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.4/verification_report.md)
- Status: PASS

- Criterion: (2) Kernel optimization changes are implemented in additive, reviewable slices and maintain deterministic prediction/correctness parity.
- Evidence:
  - Optimization progression is incremental from scalar row-first (`v0.4.1`) to hybrid routing (`v0.4.2`) to SIMD-dispatchable route (`v0.4.3`) and x86 portability hardening (`v0.4.4`).
  - Parity and invariance tests remained passing in each slice report.
- Status: PASS

- Criterion: (3) SIMD acceleration is introduced with explicit runtime feature detection and validated scalar fallback behavior.
- Evidence:
  - Runtime dispatch and AVX2 gating were introduced in `v0.4.3` and retained in `v0.4.4`.
  - Scalar fallback behavior and override control (`ALLOYGBM_DISABLE_AVX2`) validated in `v0.4.4`.
  - `x86_64` target compile/test/bench now pass after unsafe-code portability fix.
- Status: PASS (with caveat below)

- Criterion: (4) Parent verification report includes benchmark-evidence rollup summarizing baseline and post-change deltas for target workloads.
- Evidence:
  - Rollup summary:
    - `v0.4.1`: small `+43.09%`, medium `-24.27%`
    - `v0.4.2` median: small `+5.09%`, medium `-15.48%`
    - `v0.4.3` median: small `+5.62%`, medium `-17.56%`
    - `v0.4.4` (`x86_64` target compare script): medium delta (`default` vs forced scalar median) `-0.30%`
  - Bench and comparison commands documented in child reports.
- Status: PASS

- Criterion: (5) Existing engine/predictor/wrapper regression tests remain green throughout `v0.4` execution.
- Evidence:
  - Full workspace and Python tests were run and passed in `v0.4.2`, `v0.4.3`, and `v0.4.4` verification passes.
- Status: PASS

- Criterion: (6) `cargo fmt -- --check` passes at closeout.
- Evidence:
  - PASS in `v0.4.4` verification pass.
- Status: PASS

- Criterion: (7) `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
- Evidence:
  - PASS in `v0.4.4` verification pass.
- Status: PASS

- Criterion: (8) `cargo test --workspace` passes at closeout.
- Evidence:
  - PASS in `v0.4.4` verification pass.
- Status: PASS

- Criterion: (9) `cargo doc --workspace --no-deps` passes at closeout.
- Evidence:
  - PASS in `v0.4.4` verification pass.
- Status: PASS

- Criterion: (10) `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes at closeout.
- Evidence:
  - PASS in `v0.4.4` verification pass (`Ran 52 tests`, `OK`).
- Status: PASS

## Commands and Evidence Sources
- Primary command evidence source for closeout:
  - [v0.4.4 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.4/verification_report.md)
- Additional slice evidence:
  - [v0.4.1 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.1/verification_report.md)
  - [v0.4.2 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.2/verification_report.md)
  - [v0.4.3 verification report](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.4/v0.4.3/verification_report.md)

## Residual Uncovered Criteria
- None.

## Required Caveat (Apple Silicon / AVX2)
- This closeout was performed on Apple Silicon (`aarch64`).
- `x86_64` target benchmark binaries executed, but runtime output showed `runtime_avx2_enabled: false` on this machine.
- Interpretation:
  - portability, dispatch structure, and fallback behavior are verified,
  - AVX2-on-native-`x86_64` speedup characterization remains pending and should be collected for release-gate confidence.

## Final Readiness
- Ready: Yes (`v0.4` acceptance criteria satisfied with explicit evidence rollup and documented caveat).
- Required follow-up before higher-confidence performance gate:
  - run `scripts/benchmark_avx2_compare.sh` on a native AVX2-capable `x86_64` host and append results to release evidence.
