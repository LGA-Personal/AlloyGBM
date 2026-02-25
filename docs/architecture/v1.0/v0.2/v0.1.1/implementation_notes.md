# v0.1.1 Implementation Notes

## Summary of What Was Built
- Implemented the first executable `v0.2` child slice at `docs/architecture/v1.0/v0.2/v0.1.1` focused on control-contract lock-in.
- Extended core training params in `crates/core/src/lib.rs` with:
  - `row_subsample`
  - `col_subsample`
  - `early_stopping_rounds`
  - `min_validation_improvement`
  and added validation/test coverage.
- Extended engine iteration contracts in `crates/engine/src/lib.rs`:
  - `IterationControls` now carries subsampling and validation early-stopping fields.
  - `IterationRunSummary` now reports validation-loss trace metadata.
  - Added `ValidationDatasetRef` and `fit_iterations_with_validation_summary(...)`.
  - Added deterministic contract-phase sampling helpers and validation-loss plateau stop reason.
- Updated Python contract surface in `bindings/python/alloygbm/regressor.py` and tests to expose and validate the same control set.

## Non-Intuitive Decisions
- Decision: use deterministic prefix-based sampling for `row_subsample`/`col_subsample` in this layer.
- Reason: `v0.1.1` is a contract-lock slice, so deterministic behavior reduces moving parts while wiring interfaces.
- Impact: sampling semantics are intentionally conservative; this is not final stochastic production behavior.

- Decision: require an explicit validation dataset when validation early-stopping is enabled.
- Reason: avoids silent no-op policy behavior and keeps stop reasons interpretable.
- Impact: callers must route through the validation-aware engine entry point when `early_stopping_rounds` is set.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.2/v0.1.1/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- `core` contract surface expanded exactly for `v0.2` control fields.
- `engine` added a validation-aware training path without changing crate boundaries.
- Python interface remained scaffolding-level, but parameter contract parity is now aligned with Rust controls.

## Known Gaps Deferred to Next Layer
- Sampling strategy is deterministic baseline, not seeded stochastic sampling.
- No parent `v0.2` rollup artifacts yet (`implementation_notes.md` / `verification_report.md`).
- Full `0.2.0` scope items (deeper histogram growth breadth and broader quality/perf work) remain for subsequent child layers.

## Follow-Up Actions
- Implement `v0.1.2` to advance from deterministic sampling baseline toward stronger per-round sampling semantics and broader `v0.2` behavior completion.
- Keep validation-stop contracts stable while extending training behavior depth in subsequent `v0.2` slices.
