# AlloyGBM v0.1.2 Plan (v0.2 Seeded Per-Round Subsampling Semantics)

## Objective
Advance `v0.2` from deterministic prefix subsampling baseline to deterministic seeded per-round row/column subsampling semantics in the engine training loop, while preserving the `v0.1.1` validation-stop contract and reproducibility guarantees for deterministic mode.

## Scope
- In scope:
  - Replace prefix-based sampling helpers with seeded per-round sampling selectors for rows and features.
  - Ensure sampled cardinalities match configured `row_subsample` and `col_subsample` rates (ceil-based, minimum 1).
  - Support sparse/non-contiguous sampled features by generating one or more feature tiles.
  - Record per-round sampled row/feature counts in iteration summary for verification traceability.
  - Add/update engine unit tests proving seeded determinism and non-prefix behavior.
- Out of scope:
  - Deeper tree growth beyond current stump-level iterative loop.
  - Performance optimization/SIMD work.
  - Additional Python API parameters beyond current contract.
  - Parent `v0.2` rollup artifacts.

## Deliverables
1. Engine sampling package:
  - `crates/engine/src/lib.rs` uses seeded per-round row/feature selection instead of prefix selection.
2. Summary observability package:
  - iteration summary includes per-round sample coverage counts.
3. Verification package:
  - tests covering deterministic seed behavior and sampled-coverage expectations.
  - `implementation_notes.md` and `verification_report.md` for this layer.

## Implementation Sequence
1. Create `v0.1.2` plan artifact.
2. Replace sampling helper implementations with seeded per-round selectors.
3. Wire per-round sample count telemetry into iteration summary.
4. Add/update engine tests for deterministic/non-prefix sampling behavior and summary traces.
5. Run verification commands and capture evidence.

## Acceptance Criteria
1. Engine no longer uses prefix-only row/feature sampling behavior for subsampling rates `< 1.0`.
2. Row and feature sampled cardinalities match configured rates (with ceil + minimum 1 rules).
3. Deterministic mode yields reproducible sample selections for identical seed/inputs.
4. Iteration summary reports per-round sampled row and feature counts.
5. Existing validation early-stopping behavior remains passing.
6. `cargo fmt -- --check` passes.
7. `cargo clippy --workspace --all-targets -- -D warnings` passes.
8. `cargo test --workspace` passes.
9. `cargo doc --workspace --no-deps` passes.
10. `python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` passes.

## Risks and Mitigations
- Risk: seeded sampling can introduce flaky tests if randomness leaks into assertions.
  - Mitigation: assert deterministic invariants (same seed -> same selection; counts exact) instead of exact full-model metrics.
- Risk: non-contiguous feature selection may break backend expectations.
  - Mitigation: represent selected features as ordered tile list and keep backend contract unchanged.
- Risk: summary field expansion could break call-site assumptions.
  - Mitigation: update constructor paths and tests in one pass.

## Exit Condition
`v0.1.2` is complete when seeded per-round sampling semantics are implemented and verified, summary sample telemetry is present, and full verification commands pass with layer artifacts updated.
