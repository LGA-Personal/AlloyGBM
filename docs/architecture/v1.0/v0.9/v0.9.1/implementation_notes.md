# AlloyGBM v0.9.1 Implementation Notes

## Summary of What Was Built
- Completed the `v0.9.1` bug-triage slice under `docs/architecture/v1.0/v0.9/v0.9.1`.
- Added required layer artifacts:
  - [plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/plan.md)
  - [bug_triage.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/bug_triage.md)
  - [implementation_notes.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/implementation_notes.md)
  - [verification_report.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.1/verification_report.md)
- Implemented a low-risk triage fix in [scripts/benchmark_avx2_compare.sh](/Users/lashby/Projects/AlloyGBM/scripts/benchmark_avx2_compare.sh):
  - when both benchmark modes report `runtime_avx2_enabled: false`, AVX2 delta now reports `n/a (runtime AVX2 unavailable)` instead of a misleading percentage.

## Triage Outcomes
- BG-901 (P1): fixed in this slice.
  - Prior behavior could imply AVX2-related performance differences on non-AVX2 hosts.
  - Current behavior marks delta as not applicable when AVX2 is unavailable for both runs.
- BG-902 (P2): deferred to `v0.9.2`.
  - shallow/deep benchmark protocol coverage still needs explicit expansion.
- BG-903 (P2): deferred to `v0.9.2`/`v0.9.5`.
  - benchmark regression thresholds are still not codified as CI hard gates.

## Non-Intuitive Decisions
- Decision: apply one focused correctness fix in triage tooling rather than broad benchmark framework changes.
- Reason: parent `v0.9` sequence reserves benchmark expansion and deeper optimization for `v0.9.2` and `v0.9.3`.
- Impact: improves trustworthiness of immediate triage evidence while preserving scope boundaries.

## Plan Contradictions and Why
- No contradictions against `docs/architecture/v1.0/v0.9/plan.md` or `v0.9.1/plan.md`.
- Scope remained triage-first with one low-risk implementation fix and explicit deferred backlog.

## Boundary/Interface Changes vs Plan
- Changed shell summary behavior only in `scripts/benchmark_avx2_compare.sh`.
- No Rust public API changes.
- No Python public API changes.
- No model-format contract changes.

## Known Gaps Deferred to Next Layer
- `v0.9.2`: implement shallow/deep benchmark run matrix and publish paired evidence outputs.
- `v0.9.2`/`v0.9.5`: define and enforce benchmark-regression threshold policy in CI/release evidence.

## Follow-Up Actions
- Start `docs/architecture/v1.0/v0.9/v0.9.2` planning/implementation for benchmark protocol expansion.
- Preserve `BG-901` behavior as a guardrail while benchmarking improvements are implemented.
