# AlloyGBM v0.9.4 Plan (Benchmark Runtime Provenance Hardening)

## Summary
- Goal: prevent invalid Alloy benchmark comparisons caused by silently importing stale/incompatible `alloygbm` Python runtimes.
- Success criteria:
  - benchmark runner fails fast when loaded `alloygbm` runtime lacks required training controls,
  - benchmark runner fails fast when loaded native extension lacks training entrypoint,
  - benchmark outputs include runtime provenance metadata (module/native paths),
  - regression tests cover runtime contract checks.
- Audience: engineers preparing `v0.9.5+` continuous-feature native-training enablement on top of trustworthy benchmark/runtime evidence.

## Scope
### In Scope
- Create `v0.9.4` layer artifacts:
  - `docs/architecture/v1.0/v0.9/v0.9.4/plan.md`
  - `docs/architecture/v1.0/v0.9/v0.9.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/v0.9.4/verification_report.md`
- Update `benchmarks/run_model_comparison.py` to:
  - validate `alloygbm.GBMRegressor` runtime contract before dataset prep/benchmark execution,
  - validate presence of native training symbol (`train_regression_artifact`),
  - expose runtime provenance in run metadata.
- Add benchmark regression tests for runtime contract checks.
- Update benchmark docs to describe the runtime contract guard behavior.
- Update layer sequencing docs to:
  - reserve `v0.9.5` and `v0.9.6` for continuous-feature native-training support,
  - reserve `v0.9.7` for competitiveness/policy hardening,
  - reserve `v0.9.8` for docs/tutorial closeout.

### Out of Scope
- Algorithmic benchmark competitiveness work (`v0.9.7` scope).
- Parent `v0.9` rollup closeout artifacts (`v0.9.8` scope).
- New model-family roadmap scope (ranking/GPU/new objectives).

## Interfaces and Types
- Benchmark runner CLI remains backward-compatible for profile arguments and output schema.
- Additional output metadata is additive: `params.alloygbm_runtime` in JSON artifacts.
- Runtime contract requirements for Alloy benchmark path:
  - `GBMRegressor.__init__` exposes: `learning_rate`, `max_depth`, `n_estimators`, `row_subsample`, `col_subsample`.
  - native module exposes `train_regression_artifact`.

## Implementation Sequence
1. Add runtime-contract validation helpers to benchmark runner.
2. Fail benchmark execution early with actionable errors when contract checks fail.
3. Add provenance metadata emission for loaded Alloy runtime.
4. Add/extend benchmark tests for contract acceptance/rejection paths.
5. Update benchmark README and `v0.9` sequencing docs (`v0.9.5` through `v0.9.8` shift).
6. Execute verification commands and update layer index target progression.

## Test Cases and Scenarios
- `python3 -m py_compile benchmarks/run_model_comparison.py benchmarks/tests/test_alloygbm_runtime_contract.py benchmarks/tests/test_temporal_leakage.py`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- `python3 -B benchmarks/run_model_comparison.py --help`
- `python3 -B benchmarks/run_model_comparison.py --profile-grid default --profile-seeds 7 --scenarios dense_numeric`
  - expected in stale runtime environments: fail-fast with explicit runtime contract message (no misleading benchmark output)
- Non-regression gate:
  - `cargo fmt -- --check`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.4/plan.md` is present and decision-complete.
2. Benchmark runner validates Alloy runtime contract before benchmark execution.
3. Benchmark runner validates native training symbol presence before benchmark execution.
4. Incompatible runtime now fails fast with actionable error messaging.
5. Benchmark output metadata includes Alloy runtime provenance details.
6. Regression tests cover runtime contract pass/fail checks.
7. `benchmarks/README.md` documents runtime contract guard behavior.
8. `docs/architecture/v1.0/v0.9/v0.9.4/implementation_notes.md` is present.
9. `docs/architecture/v1.0/v0.9/v0.9.4/verification_report.md` is present with criterion-to-evidence mapping.
10. `cargo fmt -- --check` and benchmark test suite command pass.
11. `docs/architecture/state/layer_index.yaml` marks `v0.9.4` verified and advances target to `docs/architecture/v1.0/v0.9/v0.9.5` (with `v0.9.6`/`v0.9.7`/`v0.9.8` queued).

## Risks and Mitigations
- Risk: users rely on older Alloy benchmark package variants.
  - Mitigation: fail-fast message explains incompatibility and required runtime contract.
- Risk: stricter contract checks block benchmark runs unexpectedly in existing environments.
  - Mitigation: ensure errors are explicit and emitted before expensive benchmark prep.
- Risk: provenance metadata drifts from runtime reality.
  - Mitigation: populate metadata from live imported module paths at run time.

## Assumptions and Defaults
- Benchmark evidence is only valid when Alloy runtime contract matches the current benchmarked training interface.
- `v0.9.5` and `v0.9.6` are reserved for native continuous-feature training support.
- `v0.9.7` is reserved for competitiveness and policy hardening after continuous-feature support lands.
- `v0.9.8` is reserved for docs/tutorial and parent closeout readiness.
