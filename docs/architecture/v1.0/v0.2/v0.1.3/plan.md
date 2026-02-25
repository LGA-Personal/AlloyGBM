# AlloyGBM v0.1.3 Plan (v0.2 CPU Backend Quality Gate)

## Objective
Close a remaining `v0.2` evidence gap by validating minimal histogram GBDT behavior through real `backend_cpu` + `engine` integration tests (not mock-only), including baseline-regression quality and deterministic reproducibility checks.

## Scope
- In scope:
  - Add end-to-end CPU-backend training tests that exercise `Trainer + CpuBackend + SquaredErrorObjective` together.
  - Add explicit baseline-regression quality evidence showing trained model loss beats a naive constant predictor on a deterministic fixture.
  - Add deterministic reproducibility evidence showing identical model artifacts for repeated deterministic training with identical params/seed.
  - Keep verification artifacts updated for this layer.
- Out of scope:
  - Full-depth CART tree structure redesign beyond current stump-level iterative loop.
  - Python API expansion (`v0.3` scope).
  - SIMD/performance optimization (`v0.5+` scope).
  - Parent `v0.2` rollup artifacts.

## Deliverables
1. CPU integration test package:
  - `crates/backend_cpu/src/lib.rs` includes end-to-end training quality/reproducibility tests.
2. Verification package:
  - `docs/architecture/v1.0/v0.2/v0.1.3/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/v0.1.3/verification_report.md`
3. State package:
  - `docs/architecture/state/layer_index.yaml` updated to reflect `v0.1.3` completion and next target.

## Implementation Sequence
1. Create `v0.1.3` plan artifact.
2. Add CPU backend integration tests for quality and deterministic reproducibility.
3. Run verification command gates and collect outputs.
4. Record implementation notes and criterion-mapped verification report.
5. Update layer state index to the next layer target.

## Acceptance Criteria
1. Repository includes an end-to-end CPU backend training test that uses real `CpuBackend` and verifies trained-model loss is lower than naive constant baseline loss on a fixed fixture.
2. Repository includes deterministic reproducibility evidence for CPU backend training (same deterministic params/seed => identical artifact bytes).
3. Existing `v0.1.1`/`v0.1.2` subsampling + validation-stop contracts remain passing.
4. `cargo fmt -- --check` passes.
5. `cargo clippy --workspace --all-targets -- -D warnings` passes.
6. `cargo test --workspace` passes.
7. `cargo doc --workspace --no-deps` passes.
8. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: fixture quality test may be unstable if data/setup is under-constrained.
  - Mitigation: use a deterministic fixture and strict deterministic training params.
- Risk: reproducibility test may fail if non-deterministic paths leak in.
  - Mitigation: assert deterministic mode explicitly and compare serialized artifact bytes directly.
- Risk: this layer could drift into broader algorithm redesign.
  - Mitigation: keep scope limited to verification-quality integration evidence only.

## Exit Condition
`v0.1.3` is complete when real CPU-backend integration tests provide quality/reproducibility evidence, all verification commands pass, and layer/state artifacts are updated.
