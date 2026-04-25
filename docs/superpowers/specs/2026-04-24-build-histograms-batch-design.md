# Build-Histograms Command-Buffer Batching (Approach A)

**Date:** 2026-04-24
**Stage:** Metal Backend Stage 3 — kill-criterion follow-up
**Scope:** Engine + Metal backend; level-wise tree growth only.

## Motivation

Stage 3 met its correctness gate at S3.11 but did not cross the
`metal_friendly >1.0× CPU` kill criterion (S3.12, see `DECISIONS.md:D-020`).
After the wide histogram scatter kernel landed (D-022), per-call GPU work
dropped from ~17 ms to 2–4 ms, but the `commit + waitUntilCompleted`
round-trip per node — fired thousands of times across a level-wise fit —
is now the dominant residual cost. Profiling shows `build_histograms`
still consuming 60–68% of total Metal-side wall time even though the
kernel itself is no longer the bottleneck.

Approach A batches the two highest-frequency call sites — `build_histograms`
on the smaller-child nodes within a level, and the `subtract_histogram_bundle`
finalisation for their larger siblings — into one command buffer per level.
This collapses N per-node CPU↔GPU stalls into one per phase per level
without touching the inner kernels (so determinism is preserved by
construction).

## Non-Goals

- Leaf-wise tree growth. The leaf-wise driver in `build_tree_leaf_wise`
  has no level boundary so its batching shape is different; it stays on
  the scalar `BackendOps` methods for now and may be revisited if Stage 3
  still misses the gate after Approach A lands.
- `apply_split`, `reduce_sums`, `apply_partition_leaf_updates`. These
  fire at lower frequency and are deferred. If the kill criterion is met
  with build-histograms batching alone we move to Stage 4; otherwise
  these are the obvious next candidates.
- Metal 4 ICB chaining (Stage 4). Out of scope here.

## Architecture

Add two batched methods to the `BackendOps` trait, both with default
implementations that delegate to the existing scalar methods so the
CPU backend works unchanged:

```rust
fn build_histograms_batch(&self, requests: &mut [HistogramBuildRequest<'_>]) -> Result<(), BackendError>;
fn subtract_histogram_bundle_batch(&self, requests: &mut [SubtractRequest<'_>]) -> Result<(), BackendError>;
```

`MetalBackend` overrides both with a single-command-buffer implementation
per call. Engine changes are confined to `build_tree_level_wise`; the
leaf-wise path is unchanged.

The wide histogram scatter kernel and the subtract kernel are unchanged
— this design only changes how their dispatches are aggregated.

## Data Flow (per level in `build_tree_level_wise`)

For each level the driver now runs three phases instead of one
serial loop:

1. **Per-node serial preparation.** For every active node decide
   whether to build-from-scratch or subtract-from-parent, allocate
   destination histogram bundles via the residency pool, and choose
   which sibling is the smaller-child (the one we build) and which is
   the larger (the one we subtract).
2. **Batched histogram build.** Collect every smaller-child request
   into `Vec<HistogramBuildRequest>`, pass once to
   `backend.build_histograms_batch`. The Metal implementation encodes
   N scatter dispatches into one command buffer, commits once, and
   waits once.
3. **Batched subtract.** Collect every larger-child request into
   `Vec<SubtractRequest>`, pass once to
   `backend.subtract_histogram_bundle_batch`. Same single-CB shape.

After phase 3 the driver does the remaining per-node work (best-split
search, partitioning) using the existing scalar trait methods. Those
later phases are out of scope for this change.

Root-of-tree special case: when there is exactly one node at the
"level" (level 0), both batched methods are called with single-element
slices. The Metal override still uses one command buffer; this keeps
the code path uniform.

## Components

### Request structs (engine crate)

```rust
pub struct HistogramBuildRequest<'a> {
    pub node_id: u32,
    pub row_indices: &'a RowIndexStorage,
    pub gradients: &'a [f32],
    pub hessians: &'a [f32],
    pub destination: &'a mut HistogramBundle,
}

pub struct SubtractRequest<'a> {
    pub parent: &'a HistogramBundle,
    pub sibling: &'a HistogramBundle,
    pub destination: &'a mut HistogramBundle,
}
```

Both structs borrow into engine-owned storage; the backend writes
through `destination` and never retains references past the call.
`row_indices` already lives in residency-pool-backed storage (S3.7),
so the GPU side can pull device buffers directly.

### Metal histogram batch (`crates/backend_metal/src/kernels/histogram.rs`)

Split the existing build path into:

- `encode_histograms_into(&self, encoder, request) -> Result<EncodedHandle, BackendError>` —
  binds buffers, sets pipeline (wide if eligible), encodes one dispatch.
  Reuses the existing scratch-buffer allocation and per-feature
  setup logic; just stops short of `commit()`.
- `finalize_histograms(&self, handles: &[EncodedHandle]) -> Result<(), BackendError>` —
  reads device-side count buffers back into the host-side `HistogramBundle`
  for each request. This is the count-accumulation step that already
  exists; it just runs over a slice instead of a single request.

The new `build_histograms_batch` method:

1. Open one command buffer + one compute encoder.
2. For each request, call `encode_histograms_into`, accumulating handles.
3. End encoding, commit, `waitUntilCompleted` once.
4. Call `finalize_histograms(&handles)`.
5. On commit error: return `BackendError` with the index of the first
   failed request (see error handling below).

### Metal subtract batch (`crates/backend_metal/src/kernels/subtract.rs`)

Same shape: split `dispatch_subtract` into `encode_subtract_into` +
`finalize_subtract` (the latter is currently a no-op since subtract
writes its full result to the destination buffer; keep the seam for
parity and future readback work). One command buffer per
`subtract_histogram_bundle_batch` call.

### Profile counters (`crates/backend_metal/src/profile.rs`)

Add two counters and wire them into `dump_if_enabled`:

```rust
pub(crate) static BUILD_HISTOGRAMS_BATCH: Counter = Counter::new();
pub(crate) static SUBTRACT_BATCH: Counter = Counter::new();
```

`BUILD_HISTOGRAMS_BATCH` wraps the whole batched call (including
commit/wait); the existing `BH_*` sub-phase counters keep recording
per-request work so we can still see the breakdown. `SUBTRACT_BATCH`
plays the same role for subtract.

### `MetalBackend` glue (`crates/backend_metal/src/lib.rs`)

Two new trait methods that:

1. Take a `ScopedProbe` on the new counter.
2. Call into the kernel-module batch entry points.
3. Translate errors into `BackendError`.

No state stored on `MetalBackend` itself — both calls are stateless
relative to the existing struct.

### Engine refactor (`crates/engine/src/lib.rs`)

`build_tree_level_wise` is restructured:

- Existing per-node loop is split into the three phases above.
- The `(u32, RowIndexStorage, HistogramBundle, f32)` tuple is built
  in two passes: phase-1 produces the destination bundles via residency
  pool; phase-2/3 fill them via the batched calls.
- The post-build best-split / apply-split / partition logic stays in
  its existing per-node loop, calling the unchanged scalar trait methods.

`build_tree_leaf_wise` is untouched. The new trait methods have default
impls that delegate to the scalar versions, so the leaf-wise path
silently gets the same behaviour it has today.

## Determinism

Bit-exactness is preserved by construction:

- The wide scatter kernel and subtract kernel are unchanged.
- Within one command buffer, dispatches against disjoint destination
  buffers are independent — Metal does not reorder writes that target
  the same buffer, and each request writes to a distinct
  `HistogramBundle.destination`.
- The post-batch host-side count-accumulation runs in request order,
  matching the single-call path.

S3.10 (residency round-trip) and S3.11 (Python parity) tests already
verify bit-exactness against CPU; they continue to be the gate.

## Error Handling

`BackendError` gains an optional request-index field for batched
calls. On Metal command-buffer failure:

1. Capture `commandBuffer.error` after `waitUntilCompleted`.
2. Return `BackendError::CommandBuffer { request_index: Option<usize>, source: ... }`.
3. The engine treats any `Err` from a batched call as a fatal training
   failure (same behaviour as the scalar path today).

We do not attempt partial-success recovery. If a batch fails, the
training run aborts. This matches existing behaviour and avoids the
complexity of mid-level state reconciliation.

## Testing

New unit tests in `crates/backend_metal/tests/`:

1. `histogram_batch_matches_scalar_per_call` — same N requests run via
   the batched API and via N scalar calls; assert byte-equal histogram
   bundles.
2. `subtract_batch_matches_scalar_per_call` — same shape for subtract.
3. `histogram_batch_single_request` — N=1 (root case) hits the same
   code path and produces identical results to the scalar call.
4. `histogram_batch_empty` — N=0 is a valid no-op (no command buffer
   submitted, success returned).

Existing tests that must continue to pass:

- `cargo test --workspace` (workspace-wide)
- `bindings/python/tests/` pytest suite (must remain at 365/365)
- `metal_residency_round_trip` and `metal_python_parity_*` (S3.10/S3.11)

## Acceptance Gate

After this lands:

1. `cargo test --workspace` green.
2. `bindings/python/tests/` 365/365 green.
3. `metal_friendly` benchmark crosses `>1.0× CPU` on **at least one
   config**. Without this Stage 3 cannot close.

If (3) fails, the next move is one of: extend batching to
`apply_split`/`reduce_sums`/`apply_partition_leaf_updates`, or move to
Stage 4 (Metal 4 ICB chaining) and treat Stage 3 as "best effort". That
decision belongs to a follow-up `DECISIONS.md` entry, not this spec.

## Risks

- **Borrow-checker friction in engine refactor.** Splitting the per-node
  loop into three phases changes the lifetime shape of `active_nodes`.
  Mitigation: build the destination bundles in phase 1 and hand the
  backend `&mut HistogramBundle` slices; do not try to keep the original
  tuple structure intact across phases.
- **Command buffer size limits.** Apple's hard cap on dispatches per CB
  is large (>10k) — well above the largest realistic level width
  (max\_leaves at most a few hundred). No mitigation needed; documented
  here for the record.
- **Profile-counter drift.** Sub-phase counters (`BH_GPU_DISPATCH`,
  `BH_COMMIT_WAIT`) need to be re-read in the batched implementation
  to mean "aggregate across the batch" rather than "per call".
  Mitigation: explicitly note this in the profile module's doc comment
  when wiring up `BUILD_HISTOGRAMS_BATCH`.
