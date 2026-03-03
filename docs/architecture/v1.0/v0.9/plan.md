# AlloyGBM v0.9 Technical Plan

## Summary
- Goal: deliver `0.9.0` by running a focused debugging, benchmark-expansion, performance-improvement, and documentation/tutorial cycle after `v0.8` hardening and before `v1.0` closeout.
- Success criteria:
  - benchmark coverage includes both shallow and deep training runs across existing scenarios with reproducible outputs,
  - benchmark quality metrics are produced under leakage-hardened temporal evaluation rules,
  - validated defects found during the cycle are fixed with regression tests,
  - benchmark competitiveness (speed and/or accuracy) improves from the current `v0.8.3` baseline with recorded evidence,
  - user/operator documentation is upgraded with runnable tutorial guidance for core CPU workflows.
- Audience: engineers executing `v0.9.x` slices and reviewers deciding `v1.0` readiness.

## Scope
### In Scope
- Plan and execute `v0.9.x` child slices under `docs/architecture/v1.0/v0.9/`:
  - `v0.9.1`: bug triage + deterministic reproduction + fix plan with regression-test mapping.
  - `v0.9.2`: benchmark expansion to shallow/deep runs and refreshed comparison outputs.
  - `v0.9.3`: temporal leakage hardening for benchmark prep/splits plus benchmark integrity regression tests.
  - `v0.9.4`: documentation/tutorial pass and `v0.9` parent closeout readiness.
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
4. `v0.9.4`: upgrade documentation/tutorials and produce operator-facing guidance for running training, benchmarking, and validation gates.
5. Parent closeout: publish `v0.9` implementation/verification rollups and advance state index target toward `v1.0` final closeout.

## Test Cases and Scenarios
- Unit cases:
  - bug-specific regression tests for each fixed defect,
  - deterministic behavior checks for fixed paths.
- Integration cases:
  - end-to-end train/predict parity checks across Rust/Python surfaces after optimizations/hardening,
  - benchmark comparison runs covering shallow and deep configurations.
- Temporal integrity cases:
  - feature-target leakage guard checks in benchmark runner,
  - timestamp-boundary train/test split checks for panel and financial scenarios.
- Failure and edge cases:
  - malformed input/model artifacts continue to surface stable error semantics,
  - benchmark and prep scripts fail with actionable diagnostics on missing dependencies/network outages.
- Acceptance test mapping:
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 20`
  - `python3 -B benchmarks/run_model_comparison.py --force-prepare --rounds 120`
  - `bash scripts/benchmark_avx2_compare.sh --runs 3`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/plan.md` is present and decision-complete.
2. `v0.9.1` through `v0.9.4` child layers each have `plan.md`, `implementation_notes.md`, and `verification_report.md`.
3. A reproducible bug log and fix-to-test mapping are documented in `v0.9` child artifacts.
4. Benchmark evidence includes both shallow and deep run outputs with preserved result artifacts in `benchmarks/results/`.
5. At least one measurable performance and/or accuracy improvement versus the `v0.8.3` baseline is documented with command-backed evidence.
6. Documentation/tutorial updates are published and validated for local execution.
7. `docs/architecture/v1.0/v0.9/implementation_notes.md` is present and summarizes child-slice outcomes.
8. `docs/architecture/v1.0/v0.9/verification_report.md` is present with criterion-to-evidence mapping.
9. `cargo fmt -- --check` passes at closeout.
10. `cargo clippy --workspace --all-targets -- -D warnings` passes at closeout.
11. `cargo test --workspace`, `cargo doc --workspace --no-deps`, and Python unittest discovery pass at closeout.
12. `docs/architecture/state/layer_index.yaml` marks `docs/architecture/v1.0/v0.9` as `planned-only`/`verified` according to progress and sets next execution target appropriately.

## Risks and Mitigations
- Risk: benchmark improvements overfit to one scenario and regress others.
  - Mitigation: require scenario-by-scenario reporting for dense, panel, and histogram stress cases in shallow and deep settings.
- Risk: bug-fix work expands scope into new feature development.
  - Mitigation: enforce defect/optimization-only scope and track feature requests as deferred items.
- Risk: benchmark results are noisy across local environments.
  - Mitigation: run repeated measurements, report medians, and capture environment metadata with each benchmark summary.
- Risk: documentation drifts from actual commands.
  - Mitigation: validate documented command examples during verification.

## Assumptions and Defaults
- Device scope remains CPU-only for `v0.9`.
- Baseline comparator remains the `v0.8.3` benchmark evidence set.
- Immediate next child target after `v0.9.3` verification is `docs/architecture/v1.0/v0.9/v0.9.4`.
- Any unresolved large design change is deferred to post-`1.0.0` planning unless it blocks correctness.
