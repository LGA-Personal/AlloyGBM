# AlloyGBM v0.9.5 Plan (Native Continuous-Feature Support Phase 1: Ingestion + Quantization Bridge)

## Summary
- Goal: remove the current native-training blocker where float-valued benchmark features fail with `must be an integer-valued bin` errors.
- Success criteria:
  - Alloy native training accepts continuous float features from Python inputs,
  - continuous features are deterministically mapped into internal bins compatible with existing histogram training path,
  - benchmark runner no longer fails Alloy rows due to integer-bin validation errors on `dense_numeric` and `dow_jones_financial` default profile checks.
- Audience: engineers implementing the first half of continuous-feature support before split-quality tuning work in `v0.9.6`.

## Scope
### In Scope
- Add/adjust native training input pipeline in `bindings/python/src/lib.rs` and dependent Rust crates so float features are accepted.
- Implement deterministic quantization/binning bridge for continuous features.
- Preserve compatibility for already-binned integer-valued inputs.
- Add regression tests for:
  - float-feature acceptance,
  - deterministic binning behavior,
  - backward-compatible integer-bin path.
- Run targeted benchmark validation command set and capture failures/successes.

### Out of Scope
- Deep split-search/tuning improvements intended for `v0.9.6`.
- Full competitiveness iteration and policy hardening (`v0.9.7`).
- Docs/tutorial closeout (`v0.9.8`).

## Interfaces and Types
- Public Python API remains backward-compatible (`GBMRegressor.fit/predict` signatures unchanged).
- Internal trainer may introduce new quantization metadata, but model-format major version must not change.
- Error contract changes:
  - integer-bin-only hard failure on valid float inputs should be removed,
  - invalid numeric inputs (NaN-only rows, non-finite handling) must retain actionable errors.

## Implementation Sequence
1. Identify and remove hard integer-bin validation guard for continuous float inputs.
2. Implement deterministic float-to-bin conversion path compatible with current trainer expectations.
3. Preserve and test existing integer-bin code path.
4. Add Python/Rust regression tests for float acceptance and deterministic behavior.
5. Run focused benchmark commands and publish `implementation_notes.md` + `verification_report.md`.

## Test Cases and Scenarios
- `cargo fmt -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `TESTING_WITH_LOCAL_MODULES=1 python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'`
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'`
- `python3 -B benchmarks/run_model_comparison.py --force-prepare --profile-grid default --profile-seeds 7 --scenarios dense_numeric dow_jones_financial`

## Acceptance Criteria
1. `docs/architecture/v1.0/v0.9/v0.9.5/plan.md` is present and decision-complete.
2. Alloy native training accepts continuous float features for benchmark scenarios without `integer-valued bin` failures.
3. Deterministic quantization/binning bridge exists for continuous features.
4. Integer-bin compatibility path remains functional.
5. Regression tests cover float acceptance and integer-bin backward compatibility.
6. `docs/architecture/v1.0/v0.9/v0.9.5/implementation_notes.md` is present.
7. `docs/architecture/v1.0/v0.9/v0.9.5/verification_report.md` is present with criterion-to-evidence mapping.

## Risks and Mitigations
- Risk: quantization bridge introduces metric instability across seeds.
  - Mitigation: deterministic binning and fixed-seed validation checks.
- Risk: fixing float ingestion degrades legacy pre-binned workflows.
  - Mitigation: explicit backward-compatibility tests for integer-valued inputs.

## Assumptions and Defaults
- This slice prioritizes correctness and support for continuous features over competitiveness tuning.
- Immediate next layer after verification is `docs/architecture/v1.0/v0.9/v0.9.6`.
