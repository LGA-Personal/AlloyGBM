# AlloyGBM v0.9.7 Plan (Competitiveness and Benchmark Policy Hardening)

## Summary
- Goal: improve Alloy competitiveness on the now-valid continuous-feature training path and formalize benchmark threshold policy.
- Success criteria:
  - full benchmark matrix reruns complete with Alloy rows passing,
  - at least one measurable speed and/or quality improvement versus `v0.8.3` baseline is recorded,
  - benchmark threshold policy is codified in release/CI evidence.

## Scope
### In Scope
- Tune Alloy training path for speed/quality after continuous-feature support.
- Re-run benchmark suites (`default`, constrained `default_ultra`, AVX2 script where applicable).
- Produce benchmark regression report with scenario-by-scenario deltas.
- Define and document benchmark threshold policy and enforcement point.

### Out of Scope
- Final docs/tutorial closeout packaging (`v0.9.8`).

## Interfaces and Types
- Maintain Python API and model-format compatibility.
- Benchmark output schema remains additive/backward-compatible.

## Implementation Sequence
1. Identify highest-impact tuning opportunities from `v0.9.5/0.9.6` outputs.
2. Implement targeted optimization fixes.
3. Re-run full benchmark matrix and summarize deltas.
4. Finalize benchmark threshold policy artifact(s).
5. Publish layer implementation/verification artifacts.

## Test Cases and Scenarios
- `cargo fmt -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo doc --workspace --no-deps`
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29`
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default_ultra --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`
- `bash scripts/benchmark_avx2_compare.sh --runs 3`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.7/plan.md` is present and decision-complete.
2. Full benchmark reruns complete with Alloy rows passing on supported scenarios.
3. At least one measurable competitiveness improvement versus `v0.8.3` is documented.
4. Benchmark threshold policy is explicitly documented for CI/release evidence.
5. `docs/architecture/v1.0/v0.9/v0.9.7/implementation_notes.md` is present.
6. `docs/architecture/v1.0/v0.9/v0.9.7/verification_report.md` is present with criterion-to-evidence mapping.

## Risks and Mitigations
- Risk: improvements on one scenario regress others.
  - Mitigation: enforce scenario-by-scenario reporting and multi-seed medians.
- Risk: AVX2 evidence is unavailable on arm64 hosts.
  - Mitigation: treat AVX2 script as architecture-conditional and label results clearly.

## Assumptions and Defaults
- Continuous-feature correctness work in `v0.9.5/0.9.6` is complete.
- Immediate next layer after verification is `docs/architecture/v1.0/v0.9/v0.9.8`.
