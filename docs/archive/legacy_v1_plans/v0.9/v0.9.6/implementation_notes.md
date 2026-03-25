# AlloyGBM v0.9.6 Implementation Notes

## Summary of What Was Built
- Validated and hardened `v0.9.5` continuous-feature bridge semantics through end-to-end sensitivity evidence rather than additional API changes.
- Added continuous-feature parameter-sensitivity regression coverage in [test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py):
  - dense continuous synthetic scenario verifies that higher-capacity profile settings (depth/rounds) reduce train RMSE and materially change predictions/artifact bytes,
  - low-SNR financial-style continuous synthetic scenario verifies non-trivial profile effects in train RMSE, test RMSE spread, and prediction deltas.
- Updated benchmark operator guidance in [README.md](/Users/lashby/Projects/AlloyGBM/benchmarks/README.md) with continuous-feature interpretation caveats:
  - deterministic quantization bridge behavior,
  - multi-seed profile diagnostics recommendation,
  - low-SNR profile-spread interpretation guidance.
- Ran targeted benchmark diagnostics (`dense_numeric`, `dow_jones_financial`, seeds `7,17,29`) and captured profile-sensitive behavior in `model_comparison_20260303T162728Z.*` artifacts.

## Non-Intuitive Decisions
- Decision: treat `v0.9.6` as semantic-validation and test-hardening slice, not a new algorithm rewrite.
- Reason: `v0.9.5` already introduced the continuous-feature bridge; `v0.9.6` plan required proving split/depth/round semantics and capacity sensitivity with robust evidence.
- Impact: no new mandatory estimator parameters or model-format changes were introduced.

- Decision: use both synthetic integration tests and real benchmark diagnostics for sensitivity evidence.
- Reason: synthetic tests provide deterministic regression guardrails, while benchmark runs provide scenario-grounded behavior evidence.
- Impact: sensitivity regressions can be caught in CI-scale tests without relying exclusively on long benchmark commands.

## Plan Contradictions and Why
- No contradictions against `v0.9.6/plan.md`.
- The layer scope was satisfied via semantic validation, diagnostics, and regression tests.

## Boundary/Interface Changes vs Plan
- Public Python API remains unchanged.
- No model format changes.
- No new mandatory runtime flags.

## Known Gaps Deferred to Next Layer
- `v0.9.7` remains responsible for competitiveness tuning and benchmark threshold policy hardening.
- Continuous quantization fidelity/performance tradeoffs remain tuning scope, not correctness scope.

## Follow-Up Actions
1. Use `v0.9.6` sensitivity diagnostics as baseline constraints for `v0.9.7` tuning.
2. Add competitiveness deltas and policy evidence in `v0.9.7` against `v0.8.3` baseline.
3. Preserve `v0.9.6` sensitivity tests as release blockers for future trainer changes.
