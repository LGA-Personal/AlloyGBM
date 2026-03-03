# AlloyGBM v0.9.2 Implementation Notes

## Summary of What Was Built
- Implemented benchmark profile-matrix execution in [benchmarks/run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py) while preserving single-profile compatibility mode.
- Added new finance-oriented benchmark scenario:
  - [benchmarks/dow_jones_financial/manifest.yaml](/Users/lashby/Projects/AlloyGBM/benchmarks/dow_jones_financial/manifest.yaml)
  - [benchmarks/dow_jones_financial/prepare.py](/Users/lashby/Projects/AlloyGBM/benchmarks/dow_jones_financial/prepare.py)
- Extended benchmark output artifacts with profile-level summaries:
  - `model_comparison_profile_summary_latest.csv`
  - `model_comparison_profile_summary_latest.json`
- Updated benchmark usage docs in [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) to cover new scenario and matrix commands.
- Added the `v0.9.2` layer plan at [docs/architecture/v1.0/v0.9/v0.9.2/plan.md](/Users/lashby/Projects/AlloyGBM/docs/architecture/v1.0/v0.9/v0.9.2/plan.md).

## Non-Intuitive Decisions
- Decision: make profile-matrix mode opt-in (`--profile-grid`/`--profile`) and keep prior single-profile flags as default behavior.
- Reason: this preserves compatibility with existing command usage while enabling wider benchmark coverage for `v0.9.2`.
- Impact: old automation continues to work unchanged, and new matrix flows are additive.

- Decision: keep Dow Jones fallback target path in the same output target column when percent target is missing.
- Reason: keep prepared schema stable (`target_percent_change_next_weeks_price`) for the benchmark runner and manifest contract.
- Impact: fallback path remains deterministic and compatible, but interpretation should be revisited if fallback rows become non-trivial.

## Plan Contradictions and Why
- Original Plan Statement: update `layer_index.yaml` to mark `v0.9.2` verified and advance to `v0.9.3`.
- Implemented Decision: deferred `layer_index.yaml` advancement in this implementation step.
- Reason: this step is `alloy-layer-implement` scope; verification artifact and status transition are reserved for `alloy-layer-verify`.
- Impact: execution target remains `v0.9.2` until verification report is authored.
- Rollback or Migration Consideration: none; status update is documentation/state only and can be applied after verification.

## Boundary/Interface Changes vs Plan
- Added scenario enum support for `dow_jones_financial` in runner CLI.
- Added profile metadata to raw benchmark records (`profile_name`, `profile_index`, `run_index`, `seed`, profile hyperparameters).
- Added profile-summary outputs as additive artifacts without removing existing `model_comparison_latest.*` outputs.
- No Rust crate API changes and no Python estimator API changes.

## Known Gaps Deferred to Next Layer
- `docs/architecture/v1.0/v0.9/v0.9.2/verification_report.md` is not yet created (verification phase pending).
- `docs/architecture/state/layer_index.yaml` is not yet advanced to `v0.9.3` pending verification closeout.
- CI hard-fail threshold policy (`BG-903`) remains deferred to later `v0.9` slices.

## Follow-Up Actions
- Run `alloy-layer-verify` for `v0.9.2` and produce `verification_report.md`.
- After verification, update `layer_index.yaml` to set `v0.9.2` as verified and advance target to `v0.9.3`.
- Evaluate whether Dow Jones fallback target behavior should be tightened if fallback rows appear in future datasets.
