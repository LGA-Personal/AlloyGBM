# Metal Backend — Current Status

**Last updated:** 2026-04-25 (Stage 3 in-flight — batch infrastructure landed, RuntimeBackend forward gap found)
**Active stage:** Stage 3 — GPU residency (row partitioning + histograms + subtract)
**Active sub-task:** RuntimeBackend forwarding fix for `build_histograms_batch` / `subtract_histogram_bundle_batch`

---

## Stage 3 Checklist

Order matches the approved plan in
[/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md).

- [x] **S3.1** `HistogramStorage::{Cpu, Gpu}` and `RowIndexStorage::{Cpu, Gpu}` core enums.
- [x] **S3.2** `BackendOps` trait promotions — `subtract_histogram_bundle`, `apply_split`, `reduce_sums`.
- [x] **S3.4** CpuBackend wires all call sites on the `Cpu(..)` arm; CPU regression gate passed.
- [x] **S3.5** `shaders/partition.metal` — two-phase direction-flag + stream-compaction kernel.
- [x] **S3.6** `shaders/subtract.metal` — elementwise parent − child GPU kernel.
- [x] **S3.7a** `GpuHistogramHandle`/`GpuRowIndexHandle` newtypes + Gpu variants on storage enums.
- [x] **S3.7b** `HistogramResidencyPool` skeleton with `mint`/`get`/`release` surface.
- [x] **S3.7c** (bundled c.1/c.2/c.3) — GPU-resident histogram build/split/subtract end-to-end.
- [x] **S3.7d** `BackendOps::release_histograms` + `HistogramReleaseGuard` RAII helper.
- [x] **S3.7e** Row-index residency pool + `MetalBackend::apply_split` producing `Gpu(..)` + engine refactor. Commits: `07e0bd1` (S3.7e.2a), `f0eaf4f` (S3.7e.2b).
- [x] **S3.8** `MTLResidencySet` wrapper with PassThrough fallback.
- [x] **S3.9** `BudgetTracker` enforcing M2 free-on-consume policy.
- [x] **S3.3** Trainer-loop audit complete. See "S3.3 Audit Findings" in previous STATUS snapshot.
- [x] **S3.10** Rust unit tests — residency round-trip. Commit: `a1eceb3`.
- [x] **S3.11** Python `MetalStage3Tests`. Commit: `8de177b`.
- [ ] **S3.12** Benchmark re-run. Kill criterion `metal_friendly >1.0× CPU` NOT MET (D-023). See below.
- [x] **S3.13** `DECISIONS.md` entries D-015 through D-018. Commit: `883d9ae`.
- [ ] **S3.14** `docs/limitations.md` Section 1 rewrite — blocked on kill criterion.
- [ ] **S3.15** Full verification sweep + STATUS overwrite + SESSIONS Stage 3 close entry.

**Batch infrastructure sub-tasks (Approach A, Tasks 1–7):**
- [x] Task 1 — `BackendOps::build_histograms_batch` / `subtract_histogram_bundle_batch` scalar defaults in engine.
- [x] Task 2 — `build_tree_level_wise` refactored into three-phase shape (per-node prep → batched build → batched subtract).
- [x] Task 3 — `dispatch_subtract_batch_pool` kernel function (N subtracts in one MTLCommandBuffer).
- [x] Task 4 — `MetalBackend::subtract_histogram_bundle_batch` override.
- [x] Task 5 — Profile counters `BUILD_HISTOGRAMS_BATCH` / `SUBTRACT_BATCH` added.
- [x] Task 6 — `dispatch_histograms_batch` kernel function (N builds in one MTLCommandBuffer, extract helpers).
- [x] Task 7 — `MetalBackend::build_histograms_batch` override.
- [ ] **Task 8 / D-023 gap fix** — `RuntimeBackend` must forward `build_histograms_batch` and `subtract_histogram_bundle_batch` to `MetalBackend`. Without these two match arms the trait default fires instead, calling the scalar `build_histograms` per-request from within the default impl. The engine calls `build_histograms_batch` correctly; the gap is at the Python wrapper layer. Fix is two `match self { ... }` arms in `bindings/python/src/runtime_backend.rs`. After the fix: re-run `metal_friendly` with profiling to confirm batch counters fire and commit_wait drops from O(nodes) to O(levels).

---

## Stage 3 — In Flight (post-D-023 diagnosis)

What has shipped (Tasks 1–7 + all S3.* sub-tasks through S3.11):

- Full GPU-resident histogram pipeline: build → best_split → subtract, all pool-direct.
- Row-index residency pool + GPU `apply_split` + engine refactor (S3.7e).
- Histogram and row-index RAII release guards in both trainer loops.
- `BackendOps::build_histograms_batch` / `subtract_histogram_bundle_batch` with scalar defaults.
- `build_tree_level_wise` three-phase restructure (per-node prep → batched build → batched subtract).
- `MetalBackend::build_histograms_batch` and `subtract_histogram_bundle_batch` overrides routing N requests into one MTLCommandBuffer per phase per level.
- Profile counters `BUILD_HISTOGRAMS_BATCH` / `SUBTRACT_BATCH`.

What is NOT working yet:

- **`RuntimeBackend` in `bindings/python/src/runtime_backend.rs` does not forward `build_histograms_batch` or `subtract_histogram_bundle_batch`.** The Python-driven fit path (all benchmarks, all pytest fits) takes the scalar default instead of the Metal batch override. The batch infrastructure is correct; this is a two-line forward gap. Confirmed by D-023 profile: `BUILD_HISTOGRAMS_BATCH` and `SUBTRACT_BATCH` show 0 calls during the benchmark.

What is left for the next session:

1. **Fix `RuntimeBackend` forwarding** — add `build_histograms_batch` and `subtract_histogram_bundle_batch` arms to `RuntimeBackend`'s `BackendOps impl` in `bindings/python/src/runtime_backend.rs`. Same pattern as every other method in that file.
2. **Re-run `metal_friendly` with profiling** — confirm batch counters fire, `commit_wait` drops from ~528 calls → O(depth × estimators) calls, and measure the new speedup ratios.
3. **Evaluate kill criterion** with the corrected batch path active.
4. If MET: close Stage 3 (S3.12 tick, S3.14 docs, S3.15 sweep).
5. If NOT MET: diagnose the remaining bottleneck and scope the next fix.

---

## Verification at end of 2026-04-25 session

- `cargo test -p alloygbm-backend-metal` — 47/47 pass (per SESSIONS.md Task 7 entry).
- `pytest bindings/python/tests/ -q` — 365/365 pass (confirmed during Task 8).

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- **Stage 3** — GPU residency (row partitioning + histograms + subtract) **(in flight — batch infrastructure done, RuntimeBackend forward gap blocks kill criterion)**
- **Stage 4** — Metal 4 Indirect Command Buffer chaining (planned, not scoped)
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
