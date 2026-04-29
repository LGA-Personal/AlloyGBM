# Metal Backend — Current Status

**Last updated:** 2026-04-29 (Stage 4b complete — kill criterion NOT MET, 0.24×)
**Active stage:** Stage 5 planning — GPU histogram throughput

---

## Stage 4b Checklist

Order matches the approved plan in
`docs/superpowers/plans/2026-04-29-stage-4b-icb-chaining.md`.

- [x] **Task 1** `BackendOps::try_build_tree_level_wise` hook + engine free-function intercept.
- [x] **Task 2** Profile counters: `ICB_TREE`, `ICB_ENCODE`, `ICB_SUBMIT`, `ICB_READBACK`.
- [x] **Task 3** Cargo.toml features for `MTLIndirectCommandBuffer` + `MTLHeap`.
- [x] **Task 4** ICB Metal shaders (`icb_tree.metal`): three kernels — `icb_histogram`, `icb_split_find`, `icb_partition`. Column-major bin access; per-level histogram regions; atomic float accumulation; NaN-left/NaN-right paths.
- [x] **Task 5** `IcbPipelineCache` in `pipelines.rs` — PSOs built with `MTLComputePipelineDescriptor` + `setSupportIndirectCommandBuffers(true)`.
- [x] **Task 6** `IcbBufferPool` + `IcbSplitDecisionGpu` + `IcbConstantsGpu` in `icb_buffer_pool.rs`.
- [x] **Task 7** `IcbTreeEncoder::encode_and_run` in `kernels/icb_tree.rs` — encodes depth×3 ICB commands, submits one `MTLCommandBuffer`, waits once, reads back decisions + leaf values + row_node_ids, reconstructs stumps + updates predictions.
- [x] **Task 8** `MetalBackend::try_build_tree_level_wise` override in `lib.rs` + eligibility gate (Metal 4, ≤14 depth, ≤1024 bins, no categoricals, dataset within pool dims).
- [x] **Task 9** Parity integration tests (`tests/icb_tree_parity.rs`): 4 tests vs `CpuBackend` — small/d=4, deep/d=8, prune/high-gain, multi-estimator/5-rounds. All pass on Metal 4 (macOS 26.4.1); silently skip on earlier hardware.
- [x] **Task 10** `metal_friendly_large_icb` benchmark (0.24×, kill criterion NOT MET) + STATUS/SESSIONS update.

**Verification (2026-04-29):**
- `cargo test --test icb_tree_parity -- --test-threads=1` — 4/4 pass on macOS 26.4.1 (Metal 4).
- `cargo test --workspace --exclude alloygbm-python -- --test-threads=1` — 244/244 pass.
- `metal_friendly_large_icb` benchmark: **0.24×** (CPU 1.53s, GPU 6.44s). Kill criterion NOT MET.

**Kill-criterion result: NOT MET.**

`metal_friendly_large_icb` benchmark (1M × 100, regression d=8, bins=255, 5 estimators, Apple M4):
- CPU: 1.53s — Metal (ICB): 6.44s — Ratio: **0.24×**

Post-Stage-4b bottleneck analysis:

The `commit_wait` time in Stage 4a (4626ms / 40 dispatches = ~115ms each) was dominated by actual GPU histogram compute (800M float atomic adds for 1M × 100 rows), not by CPU idle waiting. Eliminating per-level stalls via ICB removed the inter-level sync overhead, but that overhead was already ≪ GPU compute time. Additionally, the ICB path does not use the histogram subtraction trick (it builds full histograms for ALL nodes at each level), while Stage 4a used subtraction to skip the larger-child histogram. These effects roughly cancel.

**Root cause of remaining gap:** GPU histogram kernel throughput. On Apple M4 (unordered float atomics), 800M atomic adds take ~4.5s vs. the CPU Rayon loop at ~1.5s. Competitive throughput would require either: (a) a two-pass deterministic reduction (reduce to shared memory, then accumulate once), or (b) exploiting the histogram subtraction trick on the GPU to halve the atomic work at each level.

**Next action:** Stage 5 design — evaluate GPU histogram reduction approach to close the remaining 4× gap.

---

## Stage 4b — Bugs fixed in Task 9

Four blockers discovered and resolved during parity testing (see BUGS.md for details):

| ID | Symptom | Fix |
|----|---------|-----|
| B-003 (ICB PSO) | SIGSEGV in `setComputePipelineState` inside ICB | `MTLComputePipelineDescriptor` + `setSupportIndirectCommandBuffers(true)` |
| B-004 (histogram layout) | Level N accumulated on level N-1 data | Per-level histogram regions; single CPU zero before commit |
| B-005 (bin layout) | Row-major bin access in column-major buffer | Changed to `bin_data[f * row_count + gid]` |
| B-006 (last-level leaf values) | Rows at split nodes got wrong average leaf value | CPU left/right resolution from bin data in `update_candidate_predictions` |

---

## Stage 4a — Closed (2026-04-28)

**What shipped:**

- GPU split-finding kernel (`best_split.metal`): two-pass per-feature prefix scan + cross-feature weighted-gain reduction. Handles NaN-left/NaN-right paths; Fisher-sort stays on CPU for categoricals.
- Rust dispatch wrapper with buffer-offset pattern (one `MTLCommandBuffer` per batch, one encoder per pass, scratch buffer indexed by node slot).
- `SplitDecisionPool` RAII pool for device-side output buffers (24 bytes / node).
- `MetalBackend::find_best_splits_batch` override: eligibility gate (GPU-resident, bin_count ≤ 1024), pure-numeric fast path, mixed-mode merge for numeric+categorical models.
- `BackendOps::find_best_splits_batch` trait default + engine refactor: `build_tree_level_wise` now calls the batched method; one level → one GPU split-finding dispatch.
- `RuntimeBackend` forwarding (Python bridge).

**Kill-criterion result:**

NOT MET. Best ratio post-Stage-4a:
- `metal_friendly` best: **0.17×** (regression d=6, bins=1024, 200k×200).
- `metal_friendly_large` (1M×100, regression d=8): **0.24×**.

All configs remain well below 1.0× CPU parity.

**Post-Stage-4a bottleneck breakdown (regression d=8 bins=255, 1M×100, Apple M4):**

| Site | calls | total_ms | % total |
|---|---|---|---|
| build_histograms_batch  | 40   | 5192 | 74.8% |
| ..commit_wait           | 40   | 4626 | —     |
| ..count_accumulate      | 640  | 1137 | —     |
| find_best_splits_batch  | 40   |   45 |  0.7% |
| ..commit_wait           | 40   |   44 | —     |
| subtract_histogram_bundle_batch | 40 |  69 | 1.0% |
| apply_split             | 1275 |  580 |  8.4% |

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- ~~**Stage 3** — GPU residency (row partitioning + histograms + subtract)~~ *(closed 2026-04-25 — NOT MET)*
- ~~**Stage 4a** — GPU split finding (batched find_best_splits_batch)~~ *(closed 2026-04-28 — NOT MET)*
- ~~**Stage 4b** — Metal 4 ICB chaining~~ *(closed 2026-04-29 — NOT MET: 0.24×)*
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
