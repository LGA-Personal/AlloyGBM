# AlloyGBM v0.9.6 Plan (Native Continuous-Feature Support Phase 2: Split/Depth Semantics)

## Summary
- Goal: ensure continuous-feature training path is not only accepted but behaviorally correct and sensitive to depth/round/profile controls.
- Success criteria:
  - split/histogram training over quantized continuous features is verified end-to-end,
  - Alloy quality/speed outputs vary materially with depth and rounds in expected directions,
  - diagnostics confirm non-trivial model-capacity effects on dense and financial low-SNR scenarios.

## Scope
### In Scope
- Integrate continuous-feature bins through split search and histogram accumulation paths.
- Add diagnostics/tests validating parameter sensitivity (depth, rounds, learning rate).
- Add regression coverage for low-SNR financial scenario behavior consistency.
- Update benchmark docs with interpretation caveats for continuous-feature behavior.

### Out of Scope
- Full benchmark competitiveness optimization and policy gates (`v0.9.7`).
- Docs/tutorial closeout (`v0.9.8`).

## Interfaces and Types
- Python API remains stable.
- Internal training logic may change but cannot require new mandatory user-facing parameters.
- Benchmark output schema remains backward-compatible.

## Implementation Sequence
1. Validate and harden split/hist integration for quantized continuous features.
2. Add tests proving depth/round sensitivity across representative datasets.
3. Run benchmark diagnostics focused on dense and financial scenarios.
4. Publish layer implementation/verification artifacts.

## Test Cases and Scenarios
- `cargo fmt -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7,17,29 --scenarios dense_numeric dow_jones_financial`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.6/plan.md` is present and decision-complete.
2. Continuous-feature split/hist training path is validated on benchmark scenarios.
3. Parameter-sensitivity diagnostics show meaningful depth/round effects.
4. Regression tests cover depth/round behavioral sensitivity.
5. `docs/architecture/v1.0/v0.9/v0.9.6/implementation_notes.md` is present.
6. `docs/architecture/v1.0/v0.9/v0.9.6/verification_report.md` is present with criterion-to-evidence mapping.

## Risks and Mitigations
- Risk: training behavior remains profile-invariant despite float support.
  - Mitigation: add explicit sensitivity tests as release blockers.
- Risk: low-SNR financial metrics are noisy and misinterpreted.
  - Mitigation: use multi-seed medians and scenario-specific commentary in verification report.

## Assumptions and Defaults
- `v0.9.5` float-ingestion blockers are closed before this slice starts.
- Immediate next layer after verification is `docs/architecture/v1.0/v0.9/v0.9.7`.
