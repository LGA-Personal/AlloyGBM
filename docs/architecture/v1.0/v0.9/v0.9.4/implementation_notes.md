# AlloyGBM v0.9.4 Implementation Notes

## Summary of What Was Built
- Hardened benchmark runtime validation in [run_model_comparison.py](/Users/lashby/Projects/AlloyGBM/benchmarks/run_model_comparison.py):
  - added Alloy runtime contract checks for `GBMRegressor.__init__` parameters required by benchmark training controls,
  - added native extension contract check for `train_regression_artifact`,
  - moved runtime check to run-start so incompatible environments fail before expensive dataset prep,
  - added runtime provenance metadata (`module_path`, `native_module_path`, init parameter list) to run params.
- Added runtime contract regression tests in [test_alloygbm_runtime_contract.py](/Users/lashby/Projects/AlloyGBM/benchmarks/tests/test_alloygbm_runtime_contract.py).
- Updated benchmark operator docs in [benchmarks/README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) to document fail-fast runtime guard behavior.
- Re-sequenced `v0.9` milestone planning/docs:
  - `v0.9.4`: runtime provenance hardening (this slice),
  - `v0.9.5` and `v0.9.6`: native continuous-feature training support,
  - `v0.9.7`: benchmark improvement/competitiveness and policy hardening,
  - `v0.9.8`: docs/tutorial + closeout readiness.

## Non-Intuitive Decisions
- Decision: enforce runtime contract checks as a hard benchmark precondition.
- Reason: permissive fallback silently produced misleading Alloy benchmark comparisons when a stale site-packages runtime was imported.
- Impact: some environments now fail benchmark startup until a compatible runtime is installed, but invalid comparisons are prevented.

- Decision: validate contract based on required constructor/training capability, not file-path prefix checks.
- Reason: module path alone cannot guarantee compatibility (a valid local build may still be installed under site-packages).
- Impact: checks remain robust to install location while still catching stale baseline variants.

## Plan Contradictions and Why
- Parent/continuity docs previously positioned `v0.9.4` as docs/tutorial closeout.
- Updated sequence now places runtime provenance hardening first (`v0.9.4`) because benchmark evidence quality is prerequisite to meaningful continuous-feature support and later competitiveness/closeout slices.

## Boundary/Interface Changes vs Plan
- No Rust crate API changes.
- No model format changes.
- Benchmark runner CLI arguments are unchanged.
- JSON run metadata now includes additive `alloygbm_runtime` details.

## Known Gaps Deferred to Next Layer
- `v0.9.5`: execute continuous-feature support phase 1 (float ingestion + quantization bridge).
- `v0.9.6`: execute continuous-feature support phase 2 (split/depth semantics validation).
- `v0.9.7`: execute benchmark quality/speed improvement cycle and policy hardening.
- `v0.9.8`: documentation/tutorial and parent closeout rollup artifacts.

## Follow-Up Actions
1. Build/install a benchmark-compatible Alloy runtime in benchmark environments before rerunning comparison matrices.
2. Execute `v0.9.5` and `v0.9.6` continuous-feature native-training slices.
3. Execute `v0.9.7` competitiveness iteration and record deltas against the corrected harness.
4. Execute `v0.9.8` docs/tutorial closeout and parent rollup packaging.
