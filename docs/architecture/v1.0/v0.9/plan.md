# AlloyGBM v0.9 Technical Plan

## Summary
- Goal: deliver `0.9.0` by running a focused debugging, benchmark-expansion, temporal-integrity hardening, runtime-provenance hardening, and multi-slice native continuous-feature training enablement cycle before `v1.0` closeout.
- Success criteria:
  - benchmark coverage includes both shallow and deep training runs across existing scenarios with reproducible outputs,
  - benchmark quality metrics are produced under leakage-hardened temporal evaluation rules,
  - benchmark runner provenance checks prevent stale runtime comparisons,
  - Alloy native training accepts continuous float features without integer-bin runtime failures,
  - depth/round/profile controls materially affect Alloy quality/speed behavior on continuous-feature scenarios,
  - at least one measurable performance and/or accuracy competitiveness improvement versus `v0.8.3` baseline is documented after continuous-feature support lands,
  - user/operator documentation is upgraded with runnable tutorial guidance for core CPU workflows.
- Audience: engineers executing `v0.9.x` slices and reviewers deciding `v1.0` readiness.

## Scope
### In Scope
- Plan and execute `v0.9.x` child slices under `docs/architecture/v1.0/v0.9/`:
  - `v0.9.1`: bug triage + deterministic reproduction + fix plan with regression-test mapping.
  - `v0.9.2`: benchmark expansion to shallow/deep runs and refreshed comparison outputs.
  - `v0.9.3`: temporal leakage hardening for benchmark prep/splits plus benchmark integrity regression tests.
  - `v0.9.4`: benchmark runtime provenance and contract hardening so Alloy comparisons cannot silently run against stale package variants.
  - `v0.9.5`: native continuous-feature support phase 1 (input contract + quantization/binning ingestion path for float features).
  - `v0.9.6`: native continuous-feature support phase 2 (split/histogram training integration + depth/round behavior validation).
  - `v0.9.7`: benchmark competitiveness iteration and benchmark-threshold policy hardening on continuous-feature-capable training path.
  - `v0.9.8`: documentation/tutorial pass and `v0.9` parent closeout readiness.
- Maintain and extend benchmark evidence workflow:
  - `benchmarks/*/prepare.py`,
  - `benchmarks/run_model_comparison.py`,
  - `scripts/benchmark_avx2_compare.sh`,
  - result artifacts under `benchmarks/results/`.
- Implement bug fixes and optimizations in scope-limited areas (`crates/backend_cpu`, `crates/engine`, `crates/predictor`, `bindings/python`) where needed to satisfy `0.9.0` goals.
- Produce `v0.9` rollup artifacts at closeout:
  - `docs/architecture/v1.0/v0.9/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/verification_report.md`

### Out of Scope
- New roadmap features planned for `1.0+` or `1.1+` (for example ranking objectives, GPU backends, new objective families).
- Breaking API redesign for `alloygbm.GBMRegressor`.
- Model format major-version changes.
- Replacing the benchmark stack with a different framework.

## Interfaces and Types
- Benchmark artifacts/contracts:
  - `benchmarks/results/model_comparison_latest.{csv,json,md}` schema remains backward-compatible for existing columns.
  - Benchmark scenario manifests and prepare scripts remain deterministic and CLI-driven.
- Runtime/public interfaces to preserve:
  - Rust crate boundaries (`core`, `engine`, `backend_cpu`, `predictor`, `shap`, `categorical`) remain intact.
  - Python estimator contract (`fit`, `predict`, parameter semantics, model loading behavior) remains backward-compatible.
- Documentation interfaces:
  - `README.md` and benchmark/docs content should describe reproducible local benchmark and validation flow for CPU users.

Backward-compatibility expectations:
- no breaking change to Python public API signatures,
- no required migration beyond current compatibility policy established in `v0.8.4`,
- deterministic mode behavior remains stable.

## Implementation Sequence
1. `v0.9.1`: collect current defects/perf anomalies from test and benchmark evidence, reproduce deterministically, and lock prioritized fix list plus acceptance tests.
2. `v0.9.2`: expand benchmark protocol to shallow/deep runs and refresh comparison baselines with reproducible command set.
3. `v0.9.3`: implement and verify benchmark temporal integrity hardening:
   - enforce timestamp-boundary train/test splits for time-series scenarios,
   - reject target-equivalent feature leakage in benchmark runner,
   - add regression tests for these guarantees.
4. `v0.9.4`: enforce benchmark runtime provenance/contract requirements and add regression checks that fail fast on stale Alloy runtime bindings.
5. `v0.9.5`: implement continuous-feature ingestion in native training path:
   - accept float-valued features from Python without integer-bin contract errors,
   - add deterministic quantization/binning bridge compatible with existing histogram trainer path,
   - add regression tests for float-feature acceptance and legacy integer-bin compatibility.
6. `v0.9.6`: complete continuous-feature training semantics:
   - integrate quantized continuous features through split search/histogram accumulation,
   - validate that depth/round/profile settings materially affect Alloy outcomes on benchmark scenarios,
   - add focused diagnostics proving parameter sensitivity on dense and financial low-SNR tasks.
7. `v0.9.7`: run benchmark-improvement cycle and policy hardening on continuous-capable trainer:
   - tune performance/quality competitiveness against LightGBM/XGBoost,
   - define and enforce benchmark threshold policy in release/CI evidence.
8. `v0.9.8`: upgrade documentation/tutorials and produce operator-facing guidance for running training, benchmarking, and validation gates.
9. Parent closeout: publish `v0.9` implementation/verification rollups and advance state index target toward `v1.0` final closeout.

## Test Cases and Scenarios
- Unit cases:
  - bug-specific regression tests for each fixed defect,
  - deterministic behavior checks for fixed paths.
- Integration cases:
  - end-to-end train/predict parity checks across Rust/Python surfaces after continuous-feature support,
  - benchmark comparison runs covering shallow and deep configurations.
- Temporal integrity cases:
  - feature-target leakage guard checks in benchmark runner,
  - timestamp-boundary train/test split checks for panel and financial scenarios.
- Continuous-feature enablement cases:
  - float-feature training succeeds without `integer-valued bin` runtime failures,
  - parameter-sensitivity checks validate non-trivial depth/round effects in Alloy outputs.
- Failure and edge cases:
  - malformed input/model artifacts continue to surface stable error semantics,
  - benchmark and prep scripts fail with actionable diagnostics on missing dependencies/network outages.
- Acceptance test mapping:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
  - `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7`
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`
  - `bash scripts/benchmark_avx2_compare.sh --runs 3`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/plan.md` is present and decision-complete.
2. `v0.9.1` through `v0.9.8` child layers each have `plan.md`, `implementation_notes.md`, and `verification_report.md` by milestone closeout.
3. A reproducible bug log and fix-to-test mapping are documented in `v0.9` child artifacts.
4. Benchmark evidence includes both shallow and deep run outputs with preserved result artifacts in `benchmarks/results/`.
5. Benchmark runtime provenance/contract checks prevent silent stale-package comparisons and are covered by regression tests.
6. Alloy native training supports continuous float features across benchmark scenarios without integer-bin validation failures.
7. Depth/round/profile settings show meaningful behavior changes in Alloy metrics after continuous-feature support.
8. At least one measurable performance and/or accuracy improvement versus the `v0.8.3` baseline is documented with command-backed evidence.
9. Documentation/tutorial updates are published and validated for local execution.
10. `docs/architecture/v1.0/v0.9/implementation_notes.md` is present and summarizes child-slice outcomes.
11. `docs/architecture/v1.0/v0.9/verification_report.md` is present with criterion-to-evidence mapping.
12. `cargo fmt -- --check` passes at closeout.
13. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
14. `cargo test --workspace`, `cargo doc --workspace --no-deps`, and Python unittest discovery pass at closeout.
15. `docs/architecture/state/layer_index.yaml` marks `docs/architecture/v1.0/v0.9` as `planned-only`/`verified` according to progress and sets next execution target appropriately.

## Risks and Mitigations
- Risk: continuous-feature support expands into model-format or API-breaking redesign.
  - Mitigation: keep public API/model compatibility constraints explicit and stage internal trainer changes across `v0.9.5`/`v0.9.6`.
- Risk: benchmark improvements overfit to one scenario and regress others.
  - Mitigation: require scenario-by-scenario reporting for dense, panel, and histogram stress cases in shallow and deep settings.
- Risk: benchmark results are noisy across local environments.
  - Mitigation: run repeated measurements, report medians, and capture environment metadata with each benchmark summary.
- Risk: documentation drifts from actual commands.
  - Mitigation: validate documented command examples during verification.

## Assumptions and Defaults
- Device scope remains CPU-only for `v0.9`.
- Baseline comparator remains the `v0.8.3` benchmark evidence set.
- Immediate next child target after `v0.9.4` verification is `docs/architecture/v1.0/v0.9/v0.9.5`.
- `v0.9.5` through `v0.9.7` are now reserved for native continuous-feature support and resulting competitiveness/policy hardening.
- `v0.9.8` is reserved for docs/tutorial and parent closeout readiness.
- Any unresolved large design change is deferred to post-`1.0.0` planning unless it blocks correctness.
