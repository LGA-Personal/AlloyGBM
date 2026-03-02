# AlloyGBM v0.8.3 Implementation Notes

## Summary of What Was Built
- Completed `v0.8.3` benchmark reproducibility slice by adding a dedicated benchmark workspace:
  - [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md)
  - [benchmarks/.gitignore](/Users/lashby/Projects/AlloyGBM/benchmarks/.gitignore)
  - [benchmarks/dense_numeric/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/dense_numeric/manifest.yaml)
  - [benchmarks/dense_numeric/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/dense_numeric/prepare.py)
  - [benchmarks/panel_time_series/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/manifest.yaml)
  - [benchmarks/panel_time_series/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/panel_time_series/prepare.py)
  - [benchmarks/histogram_stress/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/histogram_stress/manifest.yaml)
  - [benchmarks/histogram_stress/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/histogram_stress/prepare.py)
- Added `v0.8.3` planning and verification artifacts in `docs/architecture/v1.0/v0.9/v0.8.3/`.
- Added benchmark execution evidence artifact:
  - [benchmark_run_summary.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.8.3/benchmark_run_summary.md)
- Added cross-package model comparison runner:
  - [benchmarks/run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py)
  - outputs written to `benchmarks/results/model_comparison_latest.{csv,json,md}` with per-scenario speed and accuracy metrics for `alloygbm`, `lightgbm`, and `xgboost`.

## Non-Intuitive Decisions
- Decision: keep per-scenario `prepare.py` scripts self-contained instead of adding a shared benchmark utility library.
- Reason: reduces cross-script coupling in first benchmark workspace cut and keeps each scenario directly runnable/reviewable.
- Impact: some helper duplication exists, but scenario ownership and reproducibility are explicit.

- Decision: switch `panel_time_series` source to UCI Air Quality (`00360/AirQualityUCI.zip`).
- Reason: previously chosen URL returned 404 and did not provide stable direct-download behavior.
- Impact: panel benchmark remains UCI-backed and no-auth, with a reproducible downloadable source.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none.

## Boundary/Interface Changes vs Plan
- No runtime training/inference behavior changed.
- Changes are constrained to benchmark workspace artifacts and architecture documentation/state tracking.

## Known Gaps Deferred to Next Layer
- `v0.8.4` still needs migration/compatibility narrative finalization and parent `v0.9` rollup readiness.
- Benchmark result baselines and CI performance thresholds remain out of scope for this slice.

## Follow-Up Actions
- Open `v0.8.4` and finalize migration/checklist narrative required for `v0.9` closeout.
- Optionally expand benchmark workspace with result schema and run-history capture in later layers.
