# AlloyGBM v0.9.3 Implementation Notes

## Summary of What Was Built
- Hardened `panel_time_series` preparation in [prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/prepare.py):
  - rows are collected and time-ordered,
  - `target_co_gt` is now assigned from the next strictly later timestamp,
  - rows without a future target are dropped with explicit counter output.
- Hardened timestamp splitting and leakage detection in [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py):
  - added `_split_by_timestamp` helper to split by unique timestamp boundaries,
  - added guard that raises when any feature is exactly target-equivalent,
  - preserved CLI behavior and profile-matrix interfaces.
- Added benchmark regression tests in [test_temporal_leakage.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_temporal_leakage.py):
  - panel next-step target behavior,
  - no-overlap timestamp split behavior,
  - target-equivalent leakage rejection,
  - Dow Jones field-level safeguard assertion.
- Updated docs:
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md)
  - [benchmarks/panel_time_series/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/manifest.yaml)

## Non-Intuitive Decisions
- Decision: keep `target_co_gt` column name unchanged while changing semantics to next-timestep target.
- Reason: preserve manifest/runner compatibility and avoid unnecessary schema churn in existing benchmark pipelines.
- Impact: downstream code remains stable; documentation now clarifies horizon semantics.

- Decision: split by unique timestamp values, not row index cutoff.
- Reason: row-cutoff splitting can place the same timestamp in both train and test for grouped time-series data.
- Impact: benchmark quality metrics become temporally valid at timestamp granularity.

- Decision: add exact target-equivalence guard in runner.
- Reason: catches accidental direct leakage early and explicitly.
- Impact: invalid prepared datasets now fail fast instead of producing misleading metrics.

## Plan Contradictions and Why
- Original `v0.9` parent intent for `v0.9.3` was broad "targeted performance/quality improvements."
- Implemented focus in this slice: benchmark correctness hardening (temporal leakage control) before further performance tuning.
- Reason: benchmark trustworthiness is a prerequisite for meaningful optimization decisions.
- Impact: accuracy numbers became less optimistic, but are more decision-reliable.

## Boundary/Interface Changes vs Plan
- No Rust crate interface changes.
- No Python public estimator API changes.
- Benchmark data contract remains file-compatible; panel target semantics are now future-horizon.
- Runner CLI flags and output schemas remain backward-compatible.

## Known Gaps Deferred to Next Layer
- CI-level policy gating on benchmark quality/speed deltas is still deferred (`v0.9.4` or later).
- This slice validates temporal integrity; it does not yet optimize post-hardening benchmark quality.

## Follow-Up Actions
- Use leakage-hardened benchmark outputs as the baseline for `v0.9.4` documentation/tutorial and policy closeout.
- Consider adding group-aware rolling/purged split variants for future benchmark rigor upgrades.
