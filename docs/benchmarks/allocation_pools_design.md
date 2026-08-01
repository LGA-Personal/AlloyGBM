# Histogram and Partition Allocation Reuse

## Status

Approved design for PR #128, based on `main` at `e2b309b`.

This change closes core review section 2.4, which identifies three recurring allocation paths:

- histogram subtraction allocates a fresh complementary-child bundle;
- large parallel partitions allocate two row vectors per worker and then two final vectors;
- the standard SIMD split scanner allocates three cumulative arrays per feature and node.

## Goals

1. Reuse the consumed parent histogram allocation for the complementary child.
2. Reuse one consumed parent row-index allocation for one partition child.
3. Remove per-worker row-vector allocation from large parallel partitions.
4. Reuse numeric SIMD prefix arrays on each Rayon worker.
5. Preserve stable row order, reduction order, split choices, artifacts, predictions, and errors.
6. Cover level-wise, leaf-wise, serial, node-parallel, DRO, missing-value, and categorical paths.
7. Add deterministic tall, wide, shallow, and deep benchmark evidence.

## Non-Goals

- This PR does not replace owned node rows with a global `DataPartition` plus ranges.
- It does not change tree growth, objective mathematics, histogram precision, or split scoring.
- It does not add public tuning parameters, model fields, artifact sections, or dependencies.
- It does not pool buffers across concurrent fits or across completed trees.
- It does not optimize PL matrix histograms, categorical prefix scratch, or predictor allocation.
- It does not relax `unsafe_code = "forbid"`.

## Alternatives Considered

### One per-tree range-based DataPartition

Store every tree's rows in one backing array and represent nodes as ranges. Level-wise growth could
partition into a second array and ping-pong by depth with no row-vector allocation. This is the
review's ideal end state for a breadth-first-only trainer.

AlloyGBM also supports leaf-wise best-first growth, node-parallel level proposals, PL partition
solves, factor contexts, and APIs that currently accept `NodeSlice { row_indices: Vec<u32> }`.
Changing those contracts together would be a broad tree-builder rewrite with a much larger
correctness surface. It is not required to remove the concrete churn measured in the review.

### Shared mutex-protected pools

A tree-wide pool could lend histogram and row vectors to workers. Parallel node proposals would
then contend on pool access, buffers could not be returned while queued children own them, and
error paths would need explicit leases. The available parent allocations already have exactly the
right lifetime, so synchronization adds complexity without creating more reusable storage.

### Ownership reuse plus worker-local scratch

This is the selected design. A split consumes its node. The parent histogram and row-index vector
are therefore available for a child without a shared pool or lease. Prefix arrays are temporary
within one feature scan, so Rayon worker-local storage is the natural reusable owner.

The result keeps current data structures and scheduling while reducing each accepted split from
two new child row buffers to one, complementary histogram allocation from one to zero, and numeric
SIMD prefix allocations from three per scan to amortized worker-local growth.

## Histogram Ownership Reuse

Add an in-place operation to `HistogramBundle`:

```rust
pub fn subtract_child_in_place(
    &mut self,
    child: &HistogramBundle,
    node_id: u32,
) -> CoreResult<()>;
```

`self` initially contains the parent histogram. The operation validates that `self` and `child`
have identical feature indices, bin counts, and squared-gradient layouts. It then replaces each
parent column with `parent - child`, updates `node_id`, and rejects count underflow.

Validation happens before mutating numeric columns. Count subtraction is prevalidated before any
column is changed so an invalid child cannot leave the parent partially rewritten.

Both tree builders already own the parent bundle at the point where they build the smaller child:

1. Build the smaller child's histogram normally.
2. Move the parent bundle into a mutable local.
3. Subtract the smaller child in place.
4. Assign the reused bundle to the larger child.

This preserves the subtraction arithmetic and iteration order of the existing
`subtract_histogram_bundle_into` implementation. The allocating helper remains available for tests
and callers that do not own the parent.

Interaction-constraint filtering remains temporary split-search storage. It is never substituted
for the full parent bundle and therefore cannot corrupt descendant histogram layouts.

## Owned Partition API

Extend `BackendOps` with an ownership-aware method:

```rust
fn apply_split_owned_with_stats(
    &self,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: NodeSlice,
    split: &SplitCandidate,
) -> EngineResult<(PartitionResult, NodeStats, NodeStats)>;
```

The default implementation delegates to the existing borrowed method. Third-party or test
backends therefore remain source-compatible unless they choose to exploit ownership. `CpuBackend`
overrides it and reuses `node.row_indices` for one child.

The existing borrowed `apply_split_with_stats` contract remains intact for direct callers and
tests. Tree builders use the owned method after all consumers of the unsplit node rows have
finished. They capture the parent row count before transferring ownership for the partition
coverage check.

### Stable CPU partition

The CPU path scans the consumed parent vector in input order, writes right rows into earlier slots
of that same vector, and pushes left rows into one new vector. This safe forward compaction and the
left-row pushes both preserve input order, so both children match the existing partition order
exactly.

For nodes below the parallel threshold, the compaction pass also accumulates left and right
statistics in original row order. This matches the current sequential arithmetic.

For nodes at or above the parallel threshold:

1. Run the current ordered `par_chunks` statistics reduction without per-chunk row vectors.
2. Collect one fixed-size `ChunkStats` value per chunk.
3. Fold those values in chunk order, exactly as the current implementation does.
4. Reserve the exact left count derived from chunk statistics.
5. Perform stable forward compaction into the reused right vector and new left vector.

The second pass trades another feature-bin read for deterministic floating-point behavior and
removes `2 * worker_count` growable row buffers plus the final right-row allocation. The chunk size,
within-chunk accumulation order, and chunk fold order remain unchanged.

Validation of matrix bounds, gradient length, and feature index remains before any ownership
mutation. Classification uses the shared `goes_left_for_split` helper for numeric, missing-value,
and categorical splits.

## SIMD Split-Scan Scratch

Add a private `SplitScanScratch` alongside `HistogramArena`:

```rust
struct SplitScanScratch {
    cumulative_grad: Vec<f32>,
    cumulative_hess: Vec<f32>,
    cumulative_count: Vec<u32>,
}
```

A thread-local `RefCell<SplitScanScratch>` grows these vectors to the requested scan length, then
reuses their capacities on later scans. The active slices have exactly `scan_limit` elements and
are overwritten by the scalar prefix loop before SIMD reads.

Parallel feature scans naturally receive one scratch object per Rayon worker. The scratch borrow
does not cross a Rayon dispatch, callback, or recursive split search. Scalar, DRO, MorphBoost,
factor-neutral, PL, and categorical scanners continue using their existing paths.

The SIMD scanner's prefix accumulation, lane order, edge rejection, strict `gain > best_gain` tie
rule, missing-value direction order, and final candidate reconstruction remain unchanged.

## Concurrency and Memory Bounds

- Node-parallel proposals own disjoint parent rows and histograms; no synchronization is added.
- Multiclass class builds remain bounded by the fit-scoped Rayon pool introduced in PR #127.
- TLS split scratch is bounded by `workers * 3 * max_bins` scalar entries and grows only to the
  largest numeric feature scan observed by that worker.
- A partition keeps the parent row capacity plus one left-child allocation. It does not retain
  per-chunk row vectors after statistics reduction.
- Histogram reuse keeps one full parent allocation and allocates only the explicitly built smaller
  child bundle.

## Error and Panic Safety

- Layout and count-underflow validation precedes in-place histogram mutation.
- Backend validation precedes row-vector mutation.
- Indexed reads/writes, `Vec::truncate`, and `Vec::push` are safe operations; no uninitialized
  storage or raw pointers are used.
- If allocation fails, Rust's normal allocation failure behavior is unchanged.
- If a backend using the default owned method fails, the consumed node drops normally.
- Thread-local scratch borrows are scoped to one scanner invocation and cannot escape.

## Tests

### Core histogram tests

- In-place subtraction matches the allocating operation for scalar and DRO layouts.
- The parent bundle's column allocation pointers and capacities are retained.
- A layout mismatch or count underflow returns an error without partial mutation.

### CPU backend tests

- Owned sequential partition matches borrowed partition rows and statistics for numeric,
  categorical, and missing-value routing.
- Owned parallel partition matches the legacy chunked reduction and stable row order.
- The retained child keeps the parent row vector's allocation.
- Split-scan scratch grows once, reuses capacity, and handles a smaller subsequent scan.
- SIMD candidates remain identical to the scalar reference across seeded histogram fixtures.

### Engine tests

- Level-wise and leaf-wise training remain artifact-identical on accepted split trees.
- Node-parallel and serial level proposal paths remain artifact-identical.
- DRO histogram subtraction retains squared-gradient columns.
- Interaction constraints cannot pass a filtered histogram into descendant subtraction.

### Python regression tests

Focused estimator fits cover tall/narrow, wide/short, level-wise, leaf-wise, missing values, DRO,
and multiclass `n_jobs` execution. Existing full-suite tests remain the primary public-contract gate.

## Benchmark Contract

Add `benchmarks/allocation_reuse_benchmark.py` with deterministic synthetic cases:

| Case | Shape | Purpose |
| --- | --- | --- |
| tall/deep | many rows, few columns, depth 8 | parallel partition pressure |
| wide/deep | moderate rows, many columns, depth 8 | SIMD scratch and histogram pressure |
| short/wide | few rows, many columns | allocation-heavy control |
| shallow/tall | many rows, shallow trees | root partition control |

Each case runs level-wise and leaf-wise variants where practical, includes warmup plus repeated
measurements, and records native fit time, process RSS delta, artifact digest, prediction digest,
and RMSE. Quick CI coverage validates the harness and equivalence gates; full same-host baseline
versus candidate runs provide the performance evidence.

Acceptance requires:

- identical artifact and prediction digests for every paired case;
- identical RMSE within exact serialized prediction behavior;
- no material regression in median native fit time across the aggregate pack;
- no material increase in peak incremental RSS;
- improvement on at least one deep allocation-pressure case;
- all Rust and Python suites green.

Timing and RSS are descriptive, not universal performance claims. The resolution document records
the host, source commits, repetition count, and raw result artifact.

## Documentation

Update:

- `docs/reviews/2026-07-02-v0.12.10-core-resolutions.md` section 2.4;
- `docs/benchmarks/README.md`, `docs/user/benchmarks.md`, and the Sphinx mirror;
- `benchmarks/README.md` with quick and full commands;
- `CHANGELOG.md` under the pending release.

The resolution must distinguish this ownership-reuse implementation from a global range-based
`DataPartition`. The latter is not left as a required correctness item; it should be reconsidered
only if profiles show row ownership itself remains a dominant cost after this change.
