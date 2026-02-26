# AlloyGBM v0.2.2 Implementation Notes

## Summary of What Was Built
- Added input adapter normalization for wrapper methods in [bindings/python/alloygbm/regressor.py](/Users/lashby/Projects/AlloyGBM/bindings/python/alloygbm/regressor.py):
  - `fit(...)`, `predict(...)`, and `predict_from_artifact(...)` now accept sequence inputs and duck-typed adapters via `to_numpy`, `to_list`, or `tolist`.
  - Added `_coerce_sequence_like(...)` to normalize NumPy-like/pandas-like/Polars-like objects before existing shape/value validation.
- Preserved existing interface contracts:
  - parameter surface (`get_params`/`set_params`) unchanged,
  - existing feature-count mismatch and malformed-row validations preserved.
- Expanded test coverage:
  - [bindings/python/tests/test_regressor_contract.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_regressor_contract.py): added adapter-path tests for fit/predict/predict_from_artifact using fake NumPy/pandas/Polars-style objects.
  - [bindings/python/tests/test_native_runtime_integration.py](/Users/lashby/Projects/AlloyGBM/bindings/python/tests/test_native_runtime_integration.py): added runtime wheel-installed test proving dataframe-like adapter inputs work with native-backed `fit/predict` and artifact prediction.

## Non-Intuitive Decisions
- Decision: implement adapter support with dependency-agnostic duck typing rather than importing NumPy/pandas/Polars directly.
- Reason: keep layer scope focused on wrapper behavior and avoid adding hard dependency requirements to the package.
- Impact: adapter behavior works with real frameworks and framework-like objects exposing `to_numpy`/`to_list`/`tolist`.

- Decision: keep adapter coercion centralized in `_coerce_sequence_like(...)` and reuse existing validators.
- Reason: avoids duplicating shape/type checks across `fit`, `predict`, and `predict_from_artifact`.
- Impact: validation behavior remains consistent across all wrapper entrypoints.

## Plan Contradictions and Why
- Original Plan Statement: none contradicted during implementation.
- Implemented Decision: N/A.
- Reason: N/A.
- Impact: N/A.
- Rollback or Migration Consideration: none required.

## Boundary/Interface Changes vs Plan
- No public parameter additions/removals.
- Expanded accepted input boundary for `X`/`y` arguments through adapter normalization.
- No Rust engine/predictor contract changes.

## Known Gaps Deferred to Next Layer
- Continuous-feature quantization and mapping (currently native training bridge still expects bin-like integer-valued feature inputs).
- Explicit estimator round-count parameterization.
- Parent `v0.3` rollup artifacts (`implementation_notes.md`, `verification_report.md`).

## Follow-Up Actions
- Open `v0.2.3` plan to address remaining wrapper gaps (likely round-control and/or broader input semantics beyond adapter normalization).
- Keep `docs/architecture/state/layer_index.yaml` synchronized as each child slice closes.
