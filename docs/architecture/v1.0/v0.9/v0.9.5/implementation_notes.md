# AlloyGBM v0.9.5 Implementation Notes

## Summary of What Was Built
- Implemented native continuous-feature ingestion in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - removed strict fail-fast requirement that all features be pre-binned non-negative integers,
  - added deterministic float-to-bin quantization bridge for continuous inputs,
  - preserved strict pre-binned compatibility path (including overflow validation for explicit integer-bin inputs).
- Added a native round-cap safeguard in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
  - requested training rounds are capped at `4096` in Python bridge entrypoint,
  - prevents `encoded tree node id overflowed u32 range` on high-round benchmark profiles.
- Updated Python regressor bridge behavior in [regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - detect continuous vs pre-binned training inputs,
  - quantize continuous rows consistently before native training,
  - reuse the same quantization on `predict`, `shap_values`, and `feature_importances` when model was fit from continuous inputs.
- Added regression tests:
  - Rust bridge tests in [lib.rs](/Users/lashby/Projects/AlloyGBM/bindings/python/src/lib.rs):
    - continuous-float training acceptance,
    - deterministic quantization behavior,
    - pre-binned overflow compatibility guard,
    - large-round support via round cap.
  - Python contract tests in [test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py):
    - fit-time continuous quantization forwarding,
    - predict-time quantization when fitted on continuous inputs.

## Non-Intuitive Decisions
- Decision: keep quantization coarse and bounded (`0..255`) for continuous bridge inputs.
- Reason: this avoids histogram/memory blowups and keeps `v0.9.5` focused on compatibility/correctness rather than tuning quality.
- Impact: quality competitiveness is not the objective of this layer; finer quantization/tuning remains `v0.9.6+` work.

- Decision: cap bridge rounds at `4096`.
- Reason: current node-id encoding uses `u32` with fixed tree stride and overflows on larger tree counts.
- Impact: deep profile no longer fails from encoding overflow; exact requested rounds above cap are truncated pending broader encoding redesign.

## Plan Contradictions and Why
- `v0.9.5` plan targeted continuous-feature support and deterministic binning bridge.
- Implementation follows that plan and adds a minimal round-cap compatibility guard discovered during benchmark verification.

## Boundary/Interface Changes vs Plan
- No breaking changes to public Python estimator signatures.
- No model-format major version changes.
- Bridge semantics changed for continuous inputs (now quantized instead of rejected).

## Known Gaps Deferred to Next Layer
- `v0.9.6` will focus on split/depth semantics and parameter-sensitivity validation on continuous-feature path.
- `v0.9.7` remains competitiveness/policy hardening.

## Follow-Up Actions
1. Validate depth/round sensitivity and quantization behavior quality in `v0.9.6`.
2. Revisit node-id/round-cap architecture if higher-round support beyond `4096` is required.
3. Use `v0.9.7` to optimize competitiveness after `v0.9.6` semantic hardening.
