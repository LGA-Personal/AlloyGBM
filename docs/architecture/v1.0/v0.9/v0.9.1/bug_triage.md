# AlloyGBM v0.9.1 Bug Triage

## Scope
- Layer: `docs/architecture/v1.0/v0.9/v0.9.1`
- Date: 2026-03-03
- Objective: prioritize current defects/anomalies, capture deterministic reproduction commands, and map each item to concrete fix/verification actions.

## Prioritized Issues

### BG-901 - Misleading AVX2 delta output on non-AVX2 hosts
- Severity: P1
- Status: Fixed in `v0.9.1`
- Area: `scripts/benchmark_avx2_compare.sh`
- Expected behavior:
  - if AVX2 is unavailable for both default and forced-scalar runs, the delta should be reported as not applicable.
- Observed behavior before fix:
  - script printed numeric `medium_delta_vs_forced_scalar_median` (for example `-0.70%`) even when both runs reported `runtime_avx2_enabled: false`.
- Deterministic reproduction (pre-fix behavior):
  - `bash scripts/benchmark_avx2_compare.sh --runs 1`
  - observe summary contains:
    - `runtime_avx2_enabled(default): false`
    - `runtime_avx2_enabled(forced_scalar): false`
    - numeric `medium_delta_vs_forced_scalar_median: <percent>%`
- Implemented fix:
  - changed summary logic to emit `medium_delta_vs_forced_scalar_median: n/a (runtime AVX2 unavailable)` when both modes report AVX2 disabled.
- Verification command:
  - `bash scripts/benchmark_avx2_compare.sh --runs 1`
  - expected summary now includes `medium_delta_vs_forced_scalar_median: n/a (runtime AVX2 unavailable)`.
- Fix-to-test mapping:
  - source fix: `scripts/benchmark_avx2_compare.sh`
  - validation: command-output assertion in `v0.9.1` verification report.

### BG-902 - Benchmark protocol coverage gap (single rounds setting in prior baseline)
- Severity: P2
- Status: Open (target `v0.9.2`)
- Area: benchmark methodology/documentation
- Expected behavior:
  - benchmark evidence should include explicit shallow and deep run configurations for apples-to-apples comparisons.
- Observed evidence:
  - prior baseline references only `--rounds 80` in `v0.8.3` artifacts.
- Deterministic reproduction:
  - `rg -n -e \"--rounds 80\" docs/architecture/v1.0/v0.8/v0.8.3/benchmark_run_summary.md docs/architecture/v1.0/v0.8/v0.8.3/verification_report.md`
- Planned fix path:
  - implement run matrix for shallow/deep rounds and preserve both outputs in `v0.9.2`.
- Fix-to-test mapping:
  - planned validation commands:
    - `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 20`
    - `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 120`
  - evidence target: `v0.9.2` benchmark summary + verification report.

### BG-903 - Benchmark regressions not enforced as CI pass/fail policy
- Severity: P2
- Status: Open (target `v0.9.2`/`v0.9.4`)
- Area: CI and release-gate policy
- Expected behavior:
  - benchmark regression thresholds should have explicit go/no-go policy in CI/release gates.
- Observed evidence:
  - `v0.8` closeout documents this as a residual risk.
- Deterministic reproduction:
  - `rg -n -e \"CI-level regression thresholds\" -e \"Benchmark thresholds remain informational\" docs/architecture/v1.0/v0.8/implementation_notes.md docs/architecture/v1.0/v0.8/verification_report.md`
- Planned fix path:
  - define threshold policy and wire policy checks into CI/release evidence artifacts.
- Fix-to-test mapping:
  - planned validation in follow-up slices:
    - CI workflow/policy checks added and exercised in verification commands,
    - updated release-gate artifact shows threshold decision with pass/fail criteria.

## Exit Decision for v0.9.1
- Triaged issues count: 3
- Fixed in this slice: 1 (`BG-901`)
- Deferred with mapped owners/slices: 2 (`BG-902`, `BG-903`)
- Ready to proceed to: `docs/architecture/v1.0/v0.9/v0.9.2`
