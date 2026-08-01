# Sampled-Fit Prediction Delta Design

## Status

- Target: PR #129
- Review finding: July 2 core review section 2.7, fixed per-round O(N) overheads
- Base: `6076af71d1d21b503f359af7101f9345bd43e112`
- Scope: behavior-preserving internal training optimization and benchmark evidence

## Problem

Single-output and multiclass tree builders update `candidate_predictions` while committing splits.
Those updates already contain the accepted tree's exact leaf deltas for every sampled root row.
After construction, the trainer currently discards them, copies the committed prediction buffer a
second time, and routes every training row through the accepted tree. This full replay was added to
fix row-subsample and GOSS correctness, but it repeats work for sampled rows and retains two full
prediction copies per round.

The redundant work is most visible when row count is large and trees or feature sets are small.
The optimization must retain the correctness fix: every training row must receive every committed
tree contribution, including rows excluded from split finding.

## Goals

1. Preserve exact trained artifacts, predictions, loss histories, diagnostics, sampling decisions,
   and stop reasons for eligible fits.
2. Reuse the deltas already written by scalar and multiclass tree builders for sampled rows.
3. Route only rows excluded from tree construction through the accepted tree.
4. Remove the unconditional pre-build prediction copy for eligible fits by keeping committed and
   candidate buffers synchronized at round boundaries.
5. Cover full-row, uniform-subsample, and GOSS training under level-wise and leaf-wise growth.
6. Preserve conservative full-replay behavior where post-build processing changes tree
   contributions.
7. Add reproducible same-host baseline/candidate evidence across row and feature shapes.

## Non-Goals

- No Python estimator parameter or fitted attribute changes.
- No artifact schema or prediction semantics changes.
- No change to row-sampling probabilities, hashes, ordering, or GOSS amplification.
- No joint multi-output optimization. Joint builders do not update a candidate prediction buffer,
  so they do not contain this redundant sampled-row delta path.
- No DART prediction-state redesign. DART dropout and normalization remain a separate profiling
  and optimization item.
- No quantile-refinement redesign. Quantile refinement changes persisted leaf deltas after tree
  construction and therefore retains full replay.

## Considered Approaches

### Chosen: synchronized candidates plus excluded-row completion

Keep candidate and committed prediction buffers equal after initialization and every accepted
round. Tree construction mutates only the candidate's sampled rows. A row-restricted tree walker
then updates only excluded rows. Acceptance copies the completed candidate into the committed
buffer once, leaving both synchronized for the next round.

This removes one full copy and avoids rerouting sampled rows while retaining the trainer's current
transactional candidate/commit structure.

### Rejected: dense delta buffer with apply/rollback

A dedicated `Vec<f32>` could hold every row's round delta and apply it to committed predictions
after acceptance. It adds another O(N) buffer and zeroing pass, complicates multiclass storage, and
does not improve memory pressure.

### Rejected: excluded-row walk without candidate synchronization

Keeping the current pre-build copy and only replacing the full replay is a smaller patch, but it
leaves one avoidable full copy at every round and only partially addresses the review finding.

## Row Selection Contract

Introduce an internal row-selection result containing two sorted, disjoint vectors:

```text
RoundRowSelection {
    selected: Vec<u32>,
    excluded: Vec<u32>,
}
```

The vectors must form an exact partition of `0..row_count`. `selected` is byte-for-byte equivalent
to the current root-row result. `excluded` is derived during the same selection operation rather
than by cloning the selected vector or allocating a row-count mask.

Uniform sampling keeps the existing mixed-hash score, nth-selection boundary, and selected-row
sort. The unselected side of that same partition becomes `excluded` and is sorted independently.
Full-row selection returns all rows in `selected` and an empty `excluded` vector.

GOSS keeps the same top-gradient set, sampled-low set, realized-count amplification, and sorted
merged root rows. Unsampled low-gradient rows become `excluded`. Multiclass GOSS continues to share
one row partition and amplification across all class gradient buffers.

Existing feature-sampling helpers remain unchanged. Focused tests pin selected-row equivalence,
coverage, disjointness, ordering, edge counts, and GOSS amplification.

## Prediction Completion

Add an internal round helper that applies one tree only to explicit row indices. It uses the same
missing-value, categorical-bitset, scalar-leaf, and linear-leaf routing as the full walker.
Validation occurs before mutation: every row index must be in bounds, and raw-feature dimensions
must satisfy existing linear-leaf requirements.

For a completed tree:

```text
candidate before build = committed predictions
builder updates selected rows
restricted walker updates excluded rows
candidate after completion = committed predictions + accepted tree
```

For full-row fits, `excluded` is empty and the builder result is already complete. For sampled fits,
every row is updated exactly once: selected rows during split commits and excluded rows during
completion.

## Single-Output Lifecycle

Eligible fits are non-DART objectives without quantile leaf refinement.

1. Initialize `candidate_predictions` from committed `predictions` once.
2. Select `selected` and `excluded` rows.
3. Build the tree with `selected`, allowing the builder to update candidate deltas as today.
4. Apply the accepted tree only to `excluded` rows.
5. Evaluate candidate loss, validation, metrics, and stopping decisions unchanged.
6. On acceptance, copy candidate predictions into committed predictions. Both buffers are now
   synchronized; do not copy again at the next round start.
7. On a rejected path that continues training, restore candidate predictions from committed
   predictions before continuing. Break paths need no restoration because candidate state is not
   observed after the loop.

Standard, GOSS, MorphBoost, DRO, PL leaves, monotone bounds, missing routing, native categorical
splits, warm starts, validation, and custom metrics retain their existing behavior when their leaf
contributions are final at builder return.

Quantile refinement uses the existing fallback: reset candidate predictions from committed
predictions and replay the refined tree over all rows. DART retains its current per-round copy,
full replay, normalization, and backup logic.

## Multiclass Lifecycle

Non-DART multiclass fits use the same invariant independently for every class buffer.

Class-tree builders may remain serial or run in the fit-scoped Rayon pool. Each worker mutates only
its class candidate. After ordered build-result collection, each class tree is applied only to the
shared `excluded` row set. Existing acceptance copies candidate class buffers into committed class
buffers, so the buffers are synchronized without a new commit mechanism.

Warmup rejection paths that continue must resynchronize every modified class candidate. DART
multiclass retains the existing full-replay path. Class order, error selection, tree serialization,
and fit-thread policy remain unchanged.

## Correctness Coverage

Focused Rust tests must cover:

- uniform and GOSS row partitions exactly preserve the old selected rows;
- selected and excluded rows are sorted, disjoint, complete, and deterministic;
- restricted replay matches a full tree replay on excluded rows for numeric, missing, native
  categorical, and PL leaves;
- builder deltas plus excluded-row completion match the legacy copy-plus-full-replay oracle;
- level-wise and leaf-wise scalar fits retain exact artifacts and prediction bits under full,
  80%, 50%, and GOSS selection;
- multiclass serial and parallel class builds retain exact artifacts and probabilities;
- MorphBoost, DRO, monotone constraints, validation, custom metrics, and warm-start paths remain
  equivalent;
- quantile and DART take the fallback and retain exact behavior;
- rejected warmup rounds cannot leak candidate deltas into the next round;
- joint training remains unchanged.

Python regression tests retain the existing row-subsample/GOSS quality guards and add artifact and
prediction parity where the native test surface cannot cover estimator state.

## Benchmark Pack

Add `benchmarks/sampled_prediction_delta_benchmark.py` and focused harness tests. Full mode uses
separate release-built baseline and candidate worktrees with the native provenance manifests
introduced by PR #128. Each case runs in an isolated subprocess after an unmeasured warmup.

The matrix includes:

- tall/narrow and shallow/tall scalar regression at row subsample 1.0, 0.8, and 0.5;
- tall/narrow and medium/wide scalar GOSS;
- multiclass standard and GOSS on tall/narrow and medium/wide data;
- level-wise and leaf-wise growth where applicable;
- descriptive DART and quantile fallback sentinels;
- small-row/wide-feature cases as regression sentinels, not expected speedup cases.

Every measured pair must match artifact and prediction SHA-256 digests, completed rounds, stop
reason, and finite quality metrics. Five recorded repetitions are required for committed evidence.

Predeclared performance gates:

- aggregate geometric native-time ratio across delta-sensitive eligible cases must be at most
  `0.98`;
- aggregate native-time ratio across all eligible cases must be at most `1.03`;
- no eligible case median may exceed `1.08` unless its baseline median is below the harness noise
  floor;
- aggregate incremental-RSS ratio must be at most `1.05`, with every full-profile case measurable;
- DART and quantile fallback timings are descriptive but their equivalence gates are mandatory.

The quick CI profile runs a compact candidate self-consistency/equivalence contract. The full
baseline/candidate report is committed under `docs/benchmarks/` and is not run in CI.

## Documentation and Review Closure

Update:

- `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md` section 2.7;
- `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md` where sampled-fit correctness is
  discussed;
- benchmark indexes and Sphinx mirror;
- the unreleased changelog.

The resolution must distinguish the optimized scalar/multiclass delta path from the intentionally
unchanged joint, DART, and quantile fallback paths. It must report exact evidence and any shape that
does not improve.

## Acceptance Criteria

1. All existing Rust, Python, benchmark, formatting, clippy, and Sphinx checks pass.
2. Eligible baseline/candidate artifacts and predictions are exact in every benchmark repetition.
3. The candidate passes the predeclared timing and RSS gates.
4. Existing sampled-fit correctness tests remain green.
5. No Python API or artifact-format change is introduced.
6. Independent review finds no unresolved critical or important issues before draft PR creation.
