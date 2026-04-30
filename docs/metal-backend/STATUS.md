# Metal Backend — Current Status

**Last updated:** 2026-04-30 (Stage 5 complete — kill criterion NOT MET, 0.28–0.30×)
**Active stage:** Stage 5 CLOSED — root cause identified, next step is GPU row-sort

---

## Stage 5 — Closed (2026-04-30)

### What shipped

Stage 5 replaced the `build_histograms_batch` scatter kernel chain with a
priority-based selection of the best available kernel for the device:

1. **`histogram_build_tiled`** (tiled private-register, 1 KB TG mem) — first priority.
   No `simd_shuffle` serialisation; each thread accumulates into 32-bin-tile private
   registers, then `simd_sum` + threadgroup staging merges contributions.
   Up to 32× more concurrent threadgroups/CU vs the 32 KB wide kernel.

2. **`histogram_build_wide_dyn`** (runtime-sized TG mem, 8 KB at 256 bins) — second priority.
   Same simd-shuffle algorithm as `scatter_wide` but uses `[[threadgroup(0)]]` so the
   caller sets threadgroup length at dispatch; 4 concurrent TGs/CU vs 1 for static 32 KB.
   A gradient pre-gather pass (`histogram_gather_grads`) converts random gradient reads
   to sequential reads. **Use-gather is suppressed when the tiled kernel is selected.**

3. **`histogram_build_scatter_wide`** (static 32 KB) — third priority.
4. **`histogram_build_scatter`** (1-simdgroup, any bin count) — final fallback.

Also shipped: `histogram_count_accumulate` GPU count pass (eliminates CPU
`accumulate_counts` loop previously at 1137 ms/640 calls).

### Benchmark results — `metal_friendly_large_icb`

1M × 100 features, regression, depth=8, bins=255, 5 rounds, Apple M4:

| Kernel / optimization | commit_wait/level | speedup |
|---|---|---|
| `scatter_wide` (Stage 4a baseline) | ~115 ms | 0.24× |
| `scatter_wide_dyn` | ~90 ms | 0.28× |
| `scatter_wide_dyn` + gradient pre-gather | ~90 ms | 0.30× |
| `histogram_build_tiled` (first priority) | ~90 ms | 0.28–0.30× |

**Kill criterion NOT MET (≥1.0× required). Best observed: 0.30×.**

### Root cause of performance ceiling

**Profile summary (tiled kernel, 5 rounds, depth=8, 1M×100):**

| Site | calls | total_ms | % total |
|---|---|---|---|
| build_histograms (root, sequential) | 5 | 303 | 6.7% |
| build_histograms_batch | 40 | 3662 | 80.8% |
| ..commit_wait (GPU exec) | 40 | 3585 | — |
| apply_split | 1275 | 345 | 7.6% |
| reduce_sums | 2550 | 72 | 1.6% |
| apply_partition_leaf_updates | 1261 | 59 | 1.3% |
| **TOTAL** | — | **4530** | — |

**The bottleneck is random memory access to the 100 MB `binned_u8` buffer.**

Root-node histogram (`row_indices = [0..N]`, sequential scan):
- **60 ms for 1 M × 100 features** = 6.3 ns/feature-row

Batch-level histograms (`row_indices` = arbitrary node partition, random access):
- **90 ms for ~500 K × 100 features** = 18 ns/feature-row → **3× slower per row**

The 3× gap arises because:
- The GPU L2 cache is ≈ 400 KB/CU. The bin column for one feature is 1 MB;
  for 100 features, 100 MB total. Neither fits.
- The CPU benefits from the shared 16 MB SLC (which CAN cache 10 features
  per core with Rayon's 10-core parallelism — 10 MB fits in SLC).
- Sorting strategies (`scatter_wide_dyn`, tiled, gather) don't change the
  access pattern; all hit the same memory-latency wall.

**Experiments ruled out:**

| Experiment | Result |
|---|---|
| `ROWS_PER_CHUNK_DEFAULT` 8192 → 65536 | 0.19× (**WORSE** — long per-thread chains impede latency hiding) |
| Gradient pre-gather (wide_dyn) | +7% at most; SLC already caches 8 MB gradient buffer |
| `histogram_build_wide_dyn` vs `histogram_build_tiled` | Within noise (0.28–0.30×); same memory wall |
| Wave-scheduling overhead hypothesis | Falsified by chunk=65536 result (fewer waves → slower, not faster) |

### Path to ≥1.0× (next session)

The critical insight: **if `row_indices` were sorted before dispatch**, bin
lookups become ascending-stride reads in the feature column, enabling the
GPU hardware prefetcher.  Estimated improvement:

| Optimization | Estimated speedup | Complexity |
|---|---|---|
| GPU radix sort of node `row_indices` | batch histogram 90 ms → 30 ms/level → overall **~0.65×** | Medium (new sort kernel) |
| Batch `apply_split` per level (40 commits vs 1275) | +8% | Medium (engine API change) |
| Both combined | **~0.75×** | High |

Even with both, 1.0× requires further work (device float atomics or ICB).
The ICB path (Stage 4b) avoids random-access histogramming entirely by
processing all nodes in one chained GPU dispatch — it is the more promising
path to ≥1.0×.

---

## Next-up items

1. **GPU radix sort of `row_indices` per node** — implement a 4-pass
   GPU radix sort kernel (`partition_sort.metal`). Sort is applied per
   node after `apply_split`, before `build_histograms`. Estimated cost:
   ~15 ms/round (negligible vs 2400 ms savings). Requires new kernel +
   integration with `RowIndexResidencyPool`.

2. **Batch `apply_split` per level** — add `apply_split_batch` to
   `BackendOps` and engine; encode all level-N partitions in one
   `MTLCommandBuffer`. Saves ~300 ms (1275 commits → 40 commits).

3. **Revisit ICB path (Stage 4b)** — the ICB histogrammer uses float
   atomics which ARE sequential for a given node (no subtraction trick),
   but avoids the scatter+reduce round-trip overhead.  With the
   subtraction trick added to ICB, it could outperform the batch path.

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
- **Stage 6** — GPU radix sort of row_indices (next)
