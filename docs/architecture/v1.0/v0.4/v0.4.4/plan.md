# AlloyGBM v0.4.4 Plan (x86_64 AVX2 Benchmark/Tuning + Portability Fix)

## Summary
- Goal: execute the optional `v0.4` closeout slice by hardening the AVX2-capable histogram route for `x86_64` benchmarking, removing cross-target portability blockers, and adding reproducible AVX2-vs-scalar comparison workflow.
- Success criteria:
  - backend compiles and runs benchmark paths on both host (`aarch64`) and `x86_64` target builds under workspace lint policy,
  - AVX2 runtime-path behavior can be benchmarked against forced scalar fallback on `x86_64`,
  - correctness parity and existing regression gates remain green.
- Audience: engineers closing `v0.4` SIMD benchmark risk and reviewers validating `x86_64` measurement readiness before parent rollup.

## Scope
### In Scope
- `crates/backend_cpu/src/lib.rs` tuning and portability updates:
  - remove `unsafe` AVX2 implementation pattern that blocks `x86_64` target builds under `unsafe_code` lint policy,
  - keep runtime AVX2 dispatch semantics and scalar fallback behavior,
  - add deterministic runtime override for forced scalar fallback (`ALLOYGBM_DISABLE_AVX2`) to support A/B benchmark runs.
- `crates/backend_cpu/benches/histogram_kernels.rs` updates:
  - print runtime AVX2 enablement context for each benchmark run,
  - keep existing tiny/small/medium/split matrix unchanged for continuity.
- Add a repository script for reproducible AVX2-vs-scalar benchmark comparison on `x86_64`.
- Layer artifacts:
  - `docs/architecture/v1.0/v0.4/v0.4.4/implementation_notes.md`
  - `docs/architecture/v1.0/v0.4/v0.4.4/verification_report.md`

### Out of Scope
- AVX-512/NEON implementation.
- Multithreaded training-loop redesign.
- Parent `v0.4` rollup artifact authoring (handled after this slice).
- CUDA/Metal/MLX work.

## Interfaces and Types
- `crates/backend_cpu/src/lib.rs`:
  - retain `BackendOps` signatures and public behavior,
  - internal dispatch/fallback logic only.
- `crates/backend_cpu/benches/histogram_kernels.rs`:
  - benchmark output context additions; no benchmark matrix removals.
- `scripts/benchmark_avx2_compare.sh`:
  - standardized two-pass benchmark procedure (`default` vs forced scalar fallback) with delta summary.

Backward-compatibility expectations:
- no API contract changes for `core`, `engine`, `predictor`, or Python bindings.
- deterministic behavior and error semantics remain unchanged.

## Implementation Sequence
1. Replace unsafe AVX2-targeted histogram route with safe `x86_64` chunked row-first implementation compatible with workspace lint policy.
2. Add runtime AVX2 disable override (`ALLOYGBM_DISABLE_AVX2`) and keep dispatch behavior deterministic.
3. Extend benchmark output with runtime AVX2 context and add AVX2-vs-scalar comparison script.
4. Run host verification gates and `x86_64` target benchmark/compile checks.
5. Record evidence and update layer state index.

## Test Cases and Scenarios
- Unit cases:
  - dispatch selector still chooses per-feature for small workloads,
  - large workload uses AVX2-designated route only when runtime AVX2 is enabled,
  - large workload falls back to scalar when AVX2 is disabled/unavailable.
- Parity cases:
  - x86 chunked row-first route remains equivalent to scalar row-first outputs for fixture data.
- Benchmark cases:
  - baseline vs backend tiny/small/medium deltas remain within existing guardrails,
  - `x86_64` comparison script reports backend delta between default run and forced scalar fallback run.
- Acceptance test mapping:
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels`
  - `cargo bench -p alloygbm-backend-cpu --bench histogram_kernels --target x86_64-apple-darwin`
  - `bash scripts/benchmark_avx2_compare.sh --target x86_64-apple-darwin`
  - `cargo fmt -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace --no-deps`
  - `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`

## Acceptance Criteria
1. Backend no longer relies on `unsafe` AVX2 implementation constructs that break `x86_64` target builds under workspace lint settings.
2. Runtime AVX2 dispatch remains explicit and supports deterministic forced-scalar override via environment variable for benchmarking.
3. Benchmark output includes runtime AVX2 context signal for traceability.
4. Added AVX2-vs-scalar comparison script executes and reports deltas for `x86_64` target runs.
5. Existing histogram/split correctness tests remain passing with no API drift.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: runtime override introduces accidental production behavior change.
  - Mitigation: override is opt-in and default behavior remains unchanged.
- Risk: benchmark deltas vary significantly run-to-run.
  - Mitigation: compare same-command matrix and report explicit percentages from both runs.
- Risk: target-specific toolchain availability blocks `x86_64` verification.
  - Mitigation: include explicit target installation prerequisite and capture environment constraints in verification notes.

## Assumptions and Defaults
- `x86_64` benchmark target for this slice uses `x86_64-apple-darwin` in this repo environment.
- AVX2 capability is runtime-detected, not compile-time forced.
- Parent `v0.4` remains `planned-only` until rollup artifacts are produced.
