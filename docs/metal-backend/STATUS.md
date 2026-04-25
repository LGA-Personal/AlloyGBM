# Metal Backend — Current Status

**Last updated:** 2026-04-25 (Stage 3 complete — RuntimeBackend forwarding fixed, kill criterion NOT MET, Stage 3 closed)
**Active stage:** Stage 4 — Metal 4 Indirect Command Buffer chaining (next-up)

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
- [x] **S3.12** Benchmark re-run. Kill criterion `metal_friendly >1.0× CPU` NOT MET (D-023 + amendment). Best ratio: 0.16× (regression d=10 / d=6 bins=1024). Batch path confirmed active post-fix. Stage 3 closed on NOT MET — Approach A exhausted.
- [x] **S3.13** `DECISIONS.md` entries D-015 through D-018. Commit: `883d9ae`.
- [x] **S3.14** Deferred — Stage 3 did not meet kill criterion; `docs/limitations.md` Section 1 rewrite deferred to Stage 4 outcome.
- [x] **S3.15** STATUS overwrite + SESSIONS Stage 3 close entry (this file).

**Batch infrastructure sub-tasks (Approach A, Tasks 1–7 + forward fix):**
- [x] Task 1 — `BackendOps::build_histograms_batch` / `subtract_histogram_bundle_batch` scalar defaults in engine.
- [x] Task 2 — `build_tree_level_wise` refactored into three-phase shape (per-node prep → batched build → batched subtract).
- [x] Task 3 — `dispatch_subtract_batch_pool` kernel function (N subtracts in one MTLCommandBuffer).
- [x] Task 4 — `MetalBackend::subtract_histogram_bundle_batch` override.
- [x] Task 5 — Profile counters `BUILD_HISTOGRAMS_BATCH` / `SUBTRACT_BATCH` added.
- [x] Task 6 — `dispatch_histograms_batch` kernel function (N builds in one MTLCommandBuffer, extract helpers).
- [x] Task 7 — `MetalBackend::build_histograms_batch` override.
- [x] **Task 8 / D-023 gap fix** — `RuntimeBackend` forwarding arms for `build_histograms_batch` and `subtract_histogram_bundle_batch` added to `bindings/python/src/runtime_backend.rs`. Batch counters now fire (`build_histograms_batch`: 40 calls for depth=8 regression; `subtract_histogram_bundle_batch`: 40 calls). `commit_wait` dropped from 528 → 40 calls for that config.

---

## Stage 3 — Closed (2026-04-25)

**What shipped:**

- Full GPU-resident histogram pipeline: build → best_split → subtract, all pool-direct.
- Row-index residency pool + GPU `apply_split` + engine refactor (S3.7e).
- Histogram and row-index RAII release guards in both trainer loops.
- `BackendOps::build_histograms_batch` / `subtract_histogram_bundle_batch` with scalar defaults (engine) and Metal overrides (single MTLCommandBuffer per phase per level).
- `build_tree_level_wise` three-phase restructure (per-node prep → batched build → batched subtract).
- `RuntimeBackend` forwarding arms for both batch methods (Task 8 gap fix — this session).

**Kill-criterion result:**

NOT MET. Best ratio post-fix: 0.16× CPU parity (regression d=10 and d=6 bins=1024). All five
`metal_friendly` configs are below parity. Approach A's batching plan has been fully executed.

**Post-fix bottleneck breakdown (depth=8 regression, Apple M4):**

| Site | calls | total_ms | % total |
|---|---|---|---|
| build_histograms_batch | 40 | 2997 | 75.6% |
| &nbsp;&nbsp;.commit_wait | 40 | 2634 | — |
| &nbsp;&nbsp;.count_accumulate | 528 | 492 | — |
| best_split_with_options | 1051 | 268 | 6.8% |
| apply_split | 1051 | 281 | 7.1% |
| subtract_histogram_bundle_batch | 40 | 54 | 1.4% |

`commit_wait` (2634 ms, 66% of total) dominates even after batching
because each level still requires one synchronous CPU stall while the
GPU drains the entire level's histogram work. `count_accumulate` (492 ms)
is the next largest host-side cost.

**Residual gap analysis:**

The `waitUntilCompleted` stall is architectural: the CPU must see
histogram results to select splits and decide how to partition the next
level. Eliminating it requires either:
1. Keeping split-finding on the GPU (Metal 4 ICB — Stage 4), or
2. Async overlap of level N GPU work with level N−1 CPU work (requires
   two-level pipeline, not currently architected).

---

## Stage 4 — Next-Up: Metal 4 ICB Chaining

**Goal:** remove `waitUntilCompleted` between levels by encoding the
histogram build + split find + partition for all levels of a tree into
a single Indirect Command Buffer, committed once per estimator.

**Prerequisites not yet scoped:**
- GPU-side split finding (currently host-side in `best_split_with_options`).
- ICB residency for variable-depth trees (level width varies by split outcome — needs conditional dispatch or pre-allocated worst-case buffers).
- Metal 4 availability check (M4 on macOS 26+; fallback to Stage 3 path on older hardware).

**Next action:** write a Stage 4 plan document before starting implementation.

---

## Verification (end of 2026-04-25 session)

- `cargo test -p alloygbm-backend-metal` — 47/47 pass.
- `pytest bindings/python/tests/ -q` — 365/365 pass.
- `maturin develop --release` — clean build, no warnings.
- `build_histograms_batch` calls in profiling run: 40 (depth=8), 50 (depth=10), 30 (depth=6 bins=1024), 120 (multiclass_3), 400 (multiclass_10) — all non-zero.

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- ~~**Stage 3** — GPU residency (row partitioning + histograms + subtract)~~ *(closed 2026-04-25 — NOT MET, Approach A exhausted)*
- **Stage 4** — Metal 4 Indirect Command Buffer chaining **(next-up — not yet scoped)**
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
