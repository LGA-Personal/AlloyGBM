# v0.1.5 Implementation Notes

## Summary of What Was Built
- Implemented `v0.1.5` as a depth-behavior slice for `v0.2` in `crates/engine/src/lib.rs`.
- Reworked iterative training to:
  - use `controls.rounds` as round cap,
  - grow node-conditioned splits breadth-first within each round up to `TrainParams.max_depth`,
  - accumulate multiple stumps per completed round when child nodes remain splittable.
- Added tree/path-aware stump semantics without changing artifact payload shape:
  - encoded per-round tree-local node identity using `split.node_id` with a fixed node-stride partition,
  - added path-matching helpers for feature rows and binned rows,
  - applied non-root stumps only when ancestor branch decisions match.
- Updated validation simulation and rollback logic:
  - validation candidate predictions now apply round stumps via path-aware application,
  - rollback on `ValidationLossPlateau` now truncates model stumps by per-round stump counts.
- Expanded/updated engine tests for new semantics:
  - `fit_iterations_summary_uses_round_count_as_round_cap`
  - `fit_iterations_grows_multiple_nodes_per_round_when_depth_allows`
  - `predict_row_applies_non_root_nodes_only_when_path_matches`
  - updated existing assertions that previously assumed one stump per completed round.

## Non-Intuitive Decisions
- Decision: keep artifact payload schema unchanged and encode tree partitioning through `split.node_id`.
- Reason: avoids format churn in a mid-`v0.2` slice while still enabling path-aware behavior.
- Impact: compatibility tests stay in place; path semantics are inferred from encoded node IDs and ancestor stumps.

- Decision: treat empty-branch child partitions as a non-viable split candidate (continue search) instead of hard-failing the round.
- Reason: deeper nodes can naturally become unsplittable under fixed-threshold candidates; this should not invalidate otherwise valid round behavior.
- Impact: round-level stop reasons remain control-driven (`GainBelowThreshold`, `LeafRowsBelowThreshold`, etc.) rather than contract-failing on expected degeneracy.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.2/v0.1.5/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- No public API additions were introduced.
- Scope remained within engine internals, test coverage, and layer/state artifacts.

## Known Gaps Deferred to Next Layer
- `v0.2` parent rollup artifacts are still pending:
  - `docs/architecture/v1.0/v0.2/implementation_notes.md`
  - `docs/architecture/v1.0/v0.2/verification_report.md`
- `v1.0` parent artifacts are still pending:
  - `docs/architecture/v1.0/implementation_notes.md`
  - `docs/architecture/v1.0/verification_report.md`

## Follow-Up Actions
- Define `v0.1.6` around remaining `v0.2` completion gaps (for example stronger quality/parity evidence for depth-grown behavior and any remaining predictor alignment items) while preserving seeded subsampling and validation rollback semantics.
