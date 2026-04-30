# Metal Backend — Current Status

**Last updated:** 2026-04-30 (Stage 6 complete — kill criterion NOT MET, 0.33×)
**Active stage:** Stage 6 CLOSED — sorted row_indices improve batch histograms 41%, still memory-wall limited

---

## Stage 6 — Closed (2026-04-30)

### What shipped

Stage 6 sorts each node's row-index buffer **in-place** (ascending) inside
`encode_one_histogram_request` before binding it to the scatter kernel.

For GPU-resident nodes the pool buffer itself is sorted via unsafe
`from_raw_parts_mut` + `sort_unstable`. In-place sort is safe for all
downstream consumers (reduce_sums, apply_partition_leaf_updates, next-level
partition kernel) because they care only about *which* rows are present, not
their order. CPU-resident root node (always `[0..N]`, already sorted) is
uploaded unchanged.

**Bug fixed during implementation**: the original S6 draft used
`buffer_cache.write_row_indices` (a single shared reusable slot) for all
GPU nodes in a batch. Each node N+1 encoding overwrote the slot's underlying
Metal buffer in-place, corrupting node N's already-encoded dispatch — leading
to degenerate splits and "row_indices cannot be empty" errors. Fix: sort the
unique per-node pool buffer in-place and return it directly (as before S6),
avoiding the shared-slot overwrite hazard.

### Benchmark results — `metal_friendly_large_icb`

1M × 100 features, regression, depth=8, bins=255, 5 rounds, Apple M4 (3-run median):

| Stage | Metal | CPU | speedup |
|---|---|---|---|
| Stage 5 (tiled, ~90 ms/level) | ~4.5 s | ~1.5 s | 0.28–0.30× |
| Stage 6 (sorted, ~53 ms/level) | 3.04 s | 1.00 s | **0.33×** |

Batch histogram commit_wait: ~90 ms/level → ~53 ms/level (−41%).

**Kill criterion NOT MET (≥1.0× required). Best observed: 0.33×.**

### Why 0.33× not 0.65×

Stage 5's STATUS estimated sorted row_indices would bring batch histograms to
~30 ms/level (3× improvement) for an overall ~0.65×. Actual: ~53 ms/level.

The 3× estimate assumed every unsorted access was a cache miss.  For dense
(shallow) nodes it holds: depth-1 nodes have ~500 K rows, average gap ≈ 2 —
nearly sequential access, huge prefetch benefit.  For sparse (deep) nodes at
depth 8 (~3 906 rows, average gap ≈ 256 bytes) the gap exceeds the GPU cache
line size, so sorted order provides little benefit.  Since depth 8 dominates
the histogram work (most nodes), the aggregate gain is moderate.

### Root cause of performance ceiling (unchanged)

**The bottleneck remains random memory access to the 100 MB `binned_u8` buffer.**

At depth ≥ 6 the average gap between sorted row indices exceeds the GPU cache
line size, so ascending order gives no cache-line reuse. The GPU L2 (≈ 400 KB/CU)
still cannot cache the per-feature column (1 MB), while the CPU's 16 MB SLC
caches 10 MB of per-core bin data (Rayon × 10 P-cores).

---

## Next-up items

1. **Batch `apply_split` per level** — encode all level-N partitions in one
   `MTLCommandBuffer`. Current: 1 275 commits (5 rounds × 8 levels × 31.875 splits
   per level); target: 40 commits (5 rounds × 8 levels). Saves ~300 ms.
   Requires `apply_split_batch` in `BackendOps` + engine glue.

2. **Revisit ICB path (Stage 4b)** — the ICB histogrammer uses float atomics but
   avoids the scatter+reduce round-trip and processes all nodes in one chained GPU
   dispatch. Adding the subtraction trick to ICB could outperform the batch path.
   Currently 0.24× but untested with sorted row indices or batch apply_split.

3. **Device float atomics** — `atomic_fetch_add_explicit` on `device`-memory is
   available on Metal 3+ (A15+/M2+). Could replace the scatter-reduce two-pass
   approach with a single-pass atomic scatter directly into the histogram pool.
   No scratch allocation needed. Risk: atomic contention at popular bins.

---

## Stage 5 — Closed (2026-04-30, NOT MET: 0.28–0.30×)

Shipped `histogram_build_tiled` (private-register, 1 KB TG mem, no simd_shuffle),
`histogram_build_wide_dyn` (dynamic TG mem, 8 KB at 256 bins), gradient pre-gather
pass, GPU count accumulation kernel, and tiled-first dispatch priority.

Root cause identified: random access to 100 MB `binned_u8` overwhelms GPU L2.

---

## Stage 4b Checklist (reference)

- [x] **Task 1** `BackendOps::try_build_tree_level_wise` hook + engine free-function intercept.
- [x] **Task 2** Profile counters: `ICB_TREE`, `ICB_ENCODE`, `ICB_SUBMIT`, `ICB_READBACK`.
- [x] **Task 3** Cargo.toml features for `MTLIndirectCommandBuffer` + `MTLHeap`.
- [x] **Task 4** ICB Metal shaders (`icb_tree.metal`): three kernels — `icb_histogram`, `icb_split_find`, `icb_partition`.
- [x] **Task 5** `IcbPipelineCache` in `pipelines.rs`.
- [x] **Task 6** `IcbBufferPool` + `IcbSplitDecisionGpu` + `IcbConstantsGpu` in `icb_buffer_pool.rs`.
- [x] **Task 7** `IcbTreeEncoder::encode_and_run` in `kernels/icb_tree.rs`.
- [x] **Task 8** `MetalBackend::try_build_tree_level_wise` override in `lib.rs`.
- [x] **Task 9** Parity integration tests (`tests/icb_tree_parity.rs`): 4 tests — all pass on Metal 4.
- [x] **Task 10** `metal_friendly_large_icb` benchmark (0.24×, kill criterion NOT MET) + STATUS/SESSIONS update.

**Kill-criterion result: NOT MET (0.24×).**

---

## Stage 4a — Closed (2026-04-28, NOT MET: best 0.24×)

Shipped GPU split-finding batch kernel, `find_best_splits_batch`, `SplitDecisionPool`.

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- ~~**Stage 3** — GPU residency (row partitioning + histograms + subtract)~~ *(closed 2026-04-25 — NOT MET)*
- ~~**Stage 4a** — GPU split finding (batched find_best_splits_batch)~~ *(closed 2026-04-28 — NOT MET)*
- ~~**Stage 4b** — Metal 4 ICB chaining~~ *(closed 2026-04-29 — NOT MET: 0.24×)*
- ~~**Stage 5** — GPU histogram kernel optimization~~ *(closed 2026-04-30 — NOT MET: 0.30×)*
- ~~**Stage 6** — CPU row-sort before histogram dispatch~~ *(closed 2026-04-30 — NOT MET: 0.33×)*
- **Stage 7** — Batch `apply_split` per level (next)
