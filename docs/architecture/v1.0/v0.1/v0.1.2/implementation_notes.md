# v0.1.2 Implementation Notes

## Summary of What Was Built
- Implemented `v0.1.2` as the second `v0.1` child slice focused on replacing prefix-only subsampling semantics.
- Updated `crates/engine/src/lib.rs` sampling internals:
  - added seeded per-round index selection for rows/features,
  - replaced prefix selection with hash-ranked sampling selections,
  - converted sparse selected features into one-or-more `FeatureTile` ranges.
- Extended `IterationRunSummary` with per-round sampling coverage telemetry:
  - `sampled_rows_per_completed_round`
  - `sampled_features_per_completed_round`
- Added unit tests for seeded sampling determinism/non-prefix behavior and feature coverage accounting.

## Non-Intuitive Decisions
- Decision: sample by hash-ranking all candidate indices and taking the top-k by configured subsample rate.
- Reason: this guarantees exact sampled cardinality with deterministic reproducibility in deterministic mode.
- Impact: per-round sampled counts now match configuration exactly (ceil + minimum 1), independent of data ordering.

- Decision: represent sampled feature subsets as contiguous tile ranges rather than one index per tile.
- Reason: preserves backend contract shape while allowing non-contiguous feature subsets.
- Impact: backend histogram interface remains unchanged.

## Plan Contradictions and Why
- No contradictions to `docs/architecture/v1.0/v0.1/v0.1.2/plan.md` were introduced.

## Boundary/Interface Changes vs Plan
- Scope stayed in `engine` implementation + tests and layer artifacts.
- No crate-boundary changes and no Python API surface changes in this layer.

## Known Gaps Deferred to Next Layer
- Still stump-level iterative training (full-depth behavior remains future `v0.1` slices).
- Parent rollup artifacts for `v0.1` are not complete yet.
- Non-deterministic mode uses runtime seed perturbation and is intentionally not covered with strict reproducibility assertions.

## Follow-Up Actions
- Start `v0.1.3` to continue `v0.1` behavior completion (tree-growth/quality criteria beyond subsampling semantics).
- Keep seeded subsampling and validation-stop contracts stable while extending model behavior depth.
