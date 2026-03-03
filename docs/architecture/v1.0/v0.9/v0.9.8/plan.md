# AlloyGBM v0.9.8 Plan (Documentation, Tutorial, and Parent Closeout)

## Summary
- Goal: complete user/operator documentation and finalize `v0.9` parent closeout artifacts after continuous-feature and competitiveness slices.
- Success criteria:
  - docs/tutorial content reflects actual benchmark and training behavior,
  - `v0.9` parent rollup artifacts are complete with criterion-to-evidence mapping,
  - state index advances to the next parent milestone target after `v0.9` closeout.

## Scope
### In Scope
- Update benchmark and top-level usage docs for continuous-feature training behavior and constraints.
- Validate all documented commands end-to-end.
- Produce parent rollup artifacts:
  - `docs/architecture/v1.0/v0.9/implementation_notes.md`
  - `docs/architecture/v1.0/v0.9/verification_report.md`
- Update `docs/architecture/context/*` and `docs/architecture/state/layer_index.yaml` for post-`v0.9` continuity.

### Out of Scope
- New algorithmic features beyond `v0.9` scope.

## Interfaces and Types
- No public API or model-format breaking changes.
- Documentation must match verified command outputs.

## Implementation Sequence
1. Reconcile docs with verified behavior from `v0.9.1` through `v0.9.7`.
2. Validate runnable commands in docs/tutorial sections.
3. Author parent `v0.9` rollup implementation and verification reports.
4. Advance state index to next milestone target.

## Test Cases and Scenarios
- `cargo fmt -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo doc --workspace --no-deps`
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- Representative benchmark and usage commands documented in README/benchmarks docs.

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.8/plan.md` is present and decision-complete.
2. Documentation/tutorial updates for `v0.9` are published and command-validated.
3. `docs/architecture/v1.0/v0.9/implementation_notes.md` is present.
4. `docs/architecture/v1.0/v0.9/verification_report.md` is present with criterion-to-evidence mapping.
5. `docs/architecture/v1.0/v0.9/v0.9.8/implementation_notes.md` is present.
6. `docs/architecture/v1.0/v0.9/v0.9.8/verification_report.md` is present with criterion-to-evidence mapping.
7. `docs/architecture/state/layer_index.yaml` reflects `v0.9` closeout status and next target.

## Risks and Mitigations
- Risk: docs drift from final implementation details.
  - Mitigation: run every documented command during verification and record evidence links.
- Risk: parent rollup misses child-slice evidence.
  - Mitigation: map each parent acceptance criterion to child layer report references.

## Assumptions and Defaults
- All functional changes for `v0.9` are complete before this slice starts.
