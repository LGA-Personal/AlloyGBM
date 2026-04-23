# Metal Backend — Current Status

**Last updated:** 2026-04-23 (Stage 3 in-flight — S3.7c bundle + S3.7d lifecycle landed)
**Active stage:** Stage 3 — GPU residency (row partitioning + histograms + subtract)
**Active sub-task:** S3.3 (trainer-loop audit) → S3.7e (row-index residency pool + reduce_sums GPU arm + engine Gpu PartitionResult refactor)

---

## Stage 3 Checklist

Order matches the approved plan in
[/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md).

- [x] **S3.1** `HistogramStorage::{Cpu, Gpu}` and `RowIndexStorage::{Cpu, Gpu}` core enums. Gpu arm added under S3.7a (2026-04-21 commit `1b5e511`); Cpu arm had shipped earlier in the Stage 3 foundation commit.
- [x] **S3.2** `BackendOps` trait promotions — `subtract_histogram_bundle` is now a default trait method with CPU implementation in the free function; `apply_split` / `reduce_sums` signatures accept `&RowIndexStorage` / return storage-variant `PartitionResult`. Trainer call sites converted to `backend.subtract_histogram_bundle(...)` (three sites: level-wise smaller-first left/right, leaf-wise larger-derivation).
- [x] **S3.4** CpuBackend wires all call sites on the `Cpu(..)` arm; semantically unchanged. CPU regression gate passed (all existing Rust + Python tests).
- [x] **S3.5** `shaders/partition.metal` — two-phase direction-flag + stream-compaction kernel. `SPLIT_KIND` function constant (0 = continuous threshold, 1 = categorical bitset). Deterministic stable compaction preserves original row order. Cache-hit check + pipeline Arc sharing via `PartitionPipelineCache`.
- [x] **S3.6** `shaders/subtract.metal` — elementwise parent − child over flattened `[F × B × (grad, hess, counts)]`. Bit-exact with CPU (IEEE 754 §5.4 deterministic subtract of identical inputs). `SubtractPipelineCache` mirrors the other kernel caches.
- [x] **S3.7a** (decomposed from S3.7) — `GpuHistogramHandle(u64)` / `GpuRowIndexHandle(u64)` newtypes + `Gpu { handle, feature_count, bin_count }` / `Gpu { handle, row_count }` variants on the storage enums. Six new unit tests on core cover len / handle / gpu-accessor round-trip and legacy-shim panic on Gpu arm. Commit: `1b5e511`.
- [x] **S3.7b** (decomposed from S3.7) — `HistogramResidencyPool` skeleton in `backend_metal` with `mint` / `get` / `release` surface, overflow-checked shape math, integration with `ResidencyPool`. Commit: `ab3671f`.
- [x] **S3.7c** (bundled c.1/c.2/c.3) — `build_histograms` writes kernel output straight into pool-owned SoA buffers and returns `HistogramStorage::Gpu(..)`; `best_split` + `subtract` read pool buffers directly; Stage-2 ReusableSlots retired. Commits: `1199546` (c.1 SoA reduce), `c2886af` (c.2 pool API), `1f3c0b7` (c.3 end-to-end).
- [x] **S3.7d** (narrowed scope — histogram lifecycle only) — `BackendOps::release_histograms` trait method (default no-op) + `HistogramReleaseGuard` RAII helper. Level-wise + leaf-wise trainer loops wrap every per-node iteration in the guard so every continue/break/return path fires release deterministically. Leaf-wise queue-drain releases `PendingSplit`s that were never popped (e.g. early `break` on MaxLeavesReached). `MetalBackend::release_histograms` pattern-matches on storage variant → pool `release` on Gpu, no-op on Cpu. `RuntimeBackend` forwards the override. Closes the residency-pool leak hazard so the M2 budget projection in `backend_metal::budget` actually holds at runtime. Commit: `a81a863`.
- [x] **S3.8** `MTLResidencySet` wrapper in `residency.rs` with `PassThrough` fallback for macOS 13/14 + no-Metal builds. Attach-on-construct / detach-on-drop lifecycle; 3 unit tests. Wired into `MetalBackend` as of S3.7b.
- [x] **S3.9** `BudgetTracker` in `budget.rs` enforcing the M2 free-on-consume policy: refuses the fit when projected peak (`F × B × L × 12`) exceeds 80 % of `MTLDevice.recommendedMaxWorkingSetSize`. Returns `EngineError::BackendUnavailable` with `device="cpu"` fallback guidance in the error message. 6 unit tests on the tracker + 3 integration tests exercising `MetalBackend::check_histogram_budget` on a real device. Pathological-shape risk note lives at the top of `budget.rs`. Commit: `495eefa`.
- [ ] **S3.7e** (new — deferred from S3.7d) — Row-index residency pool + `reduce_sums` GPU arm + engine-side `Gpu PartitionResult` refactor. The engine's `apply_partition_leaf_updates`, `validate_partition_cover`, and `into_cpu_parts` all currently call `.left_row_indices()` / `.right_row_indices()` which panic on the Gpu arm — GPU row storage requires rewriting each of those helpers to dispatch on the variant. Substantially larger change than the histogram path; ships as its own sub-task. `reduce_sums` Gpu arm currently dead code (no producer) — wiring up a `RowIndexResidencyPool` + `MetalBackend::apply_split` producing `Gpu(..)` is what activates it.
- [ ] **S3.3** Trainer-loop audit — very thin as of today (engine already calls `BackendOps::subtract_histogram_bundle` through the trait, so the trainer hot loop is storage-variant-agnostic in Cpu mode). Concrete audit task: walk every `.feature_histograms()` / `.left_row_indices()` / `.right_row_indices()` call site in `crates/engine/src/lib.rs` and confirm each either (a) is Cpu-only by construction, or (b) pattern-matches on the variant. Blocks S3.7e pre-flight.
- [ ] **S3.10** Rust unit tests — residency round-trip (`build_histograms` → `best_split` → `subtract` → `best_split` on child, all values bit-exact after a single CPU read-back). Partition + subtract correctness tests already landed with S3.5 / S3.6. Histogram lifecycle tests landed with S3.7d.
- [ ] **S3.11** Python `MetalStage3Tests` — golden 50k × 100 × 255 × 20 fit pair with structural-plus-ulp gate, dedicated NaN-heavy / monotone-constraint / mixed-categorical (D-017 coverage) / memory-pressure (M2 budget guard fires cleanly) cases.
- [ ] **S3.12** Benchmark re-run on Apple M4 + new dated section in `BENCHMARKS.md`. Kill criterion: `metal_friendly` depth 8 + depth 10 + K=10 multiclass must all cross >1.0× CPU. Blocks stage close if it doesn't.
- [x] **S3.13** `DECISIONS.md` entries — D-015 (enum-variant storage API), D-016 (M2 residency budget), D-017 (categorical partition on GPU, split on CPU), D-018 (subtract / apply_split / reduce_sums promoted to `BackendOps`). Commit: `883d9ae`.
- [ ] **S3.14** `docs/limitations.md` Section 1 rewrite based on the S3.12 benchmark outcome.
- [ ] **S3.15** Full verification sweep + `STATUS.md` overwrite + `SESSIONS.md` Stage 3 close entry.

---

## Stage 3 — In Flight

What has shipped (counting everything on `claude/charming-carson-d08c9a` through `a81a863`):

- Core storage enums (`HistogramStorage` / `RowIndexStorage`) with both `Cpu` and `Gpu` variants + opaque `u64` handle newtypes.
- `BackendOps` trait has `subtract_histogram_bundle` as a default method; trainer call sites call through the backend rather than the free function.
- `BackendOps::release_histograms` trait method (default no-op) + `HistogramReleaseGuard` RAII helper wired into both level-wise and leaf-wise trainer loops. Queue-drain on leaf-wise early break covers `PendingSplit`s never popped.
- `shaders/partition.metal` + `kernels/partition.rs` — row partitioning on GPU for both continuous and categorical splits. Live behind `MetalBackend::apply_split`; falls back to CPU on scan-cap overflow. **Still produces `RowIndexStorage::Cpu(..)`** today — GPU row residency is S3.7e.
- `shaders/subtract.metal` + `kernels/subtract.rs` — elementwise subtract on GPU. Live behind `MetalBackend::subtract_histogram_bundle` when parent + child are both `Gpu(..)`.
- `residency.rs` — `MTLResidencySet` wrapper + `PassThrough` fallback. Wired into `MetalBackend::new`.
- `budget.rs` — M2 free-on-consume working-set tracker. Wired into `MetalBackend::new`. Now actually correct at runtime — S3.7d closed the residency-pool leak that previously broke the one-level-wide peak projection.
- `histogram_residency.rs` — GPU-resident histogram pool. Active end-to-end: build → split → subtract → release via engine-side RAII guard.
- `MetalBackend::build_histograms` returns `HistogramStorage::Gpu { handle, feature_count, bin_count }`; `best_split` / `subtract` consume handles directly; Stage-2 `split_grad`/`split_hess`/`split_counts` ReusableSlots retired.

What is left for the next session:

- **S3.3 audit (fast)** — walk `crates/engine/src/lib.rs` for remaining `.feature_histograms()` / `.left_row_indices()` / `.right_row_indices()` call sites; document which are Cpu-only by construction vs. which will need variant dispatch when S3.7e lands. Pre-flight for S3.7e; should take under an hour.
- **S3.7e** — row-index residency pool + `MetalBackend::apply_split` producing `RowIndexStorage::Gpu(..)` + `MetalBackend::reduce_sums` Gpu arm + engine-side refactor of `apply_partition_leaf_updates` / `validate_partition_cover` / `into_cpu_parts` to dispatch on the storage variant. The engine refactor is the bulk of the work; the Metal pool is straight-line copy of `HistogramResidencyPool`.
- **S3.10 / S3.11** — residency round-trip Rust tests + Python golden pair. Both land after S3.7e so the Gpu path is end-to-end resident.
- **S3.12** — benchmark (kill criterion). Stage 3 stands or falls by whether `metal_friendly` deep-tree configs cross >1.0× CPU; if they don't, dig in before declaring Stage 3 shipped.
- **S3.14 / S3.15** — docs + final verification sweep.

---

## Current Session Commits (2026-04-23)

- `1199546` — feat(backend_metal): histogram reduce pass emits SoA (S3.7c.1, D-019)
- `c2886af` — feat(backend_metal): residency pool API for GPU histograms (S3.7c.2)
- `1f3c0b7` — feat(backend_metal): pool-resident GPU histograms end-to-end (S3.7c.3, D-008, D-012, D-019)
- `a81a863` — feat(backend_metal): histogram handle lifecycle cleanup (S3.7d)

**Verification at end of session:**
- `cargo check --workspace` + `--no-default-features` — clean
- `cargo clippy --workspace --all-targets -- -D warnings` + `--no-default-features` — clean
- `cargo fmt --all --check` — clean
- `cargo test --workspace --exclude alloygbm-python` — 220 tests pass (38 core + 69 engine + 38 backend_metal (+2 release_histograms tests) + 23 backend_cpu + 19 categorical + 10 predictor + 23 shap)
- `maturin develop --release` with default features — `pytest bindings/python/tests -q` green (362 passed, 20 subtests passed)
- `maturin develop --release --no-default-features` — `pytest` green (334 passed, 28 Metal-gated skipped, 16 subtests passed)

---

## Blockers / Open Questions

- **S3.7e engine surface is the real work.** The histogram side was forgiving because it flowed through `BackendOps` methods — the trainer never touched `.feature_histograms()` directly on a Gpu arm. The row-index side is not — `apply_partition_leaf_updates` + `validate_partition_cover` + `into_cpu_parts` read row indices eagerly. Each has to be rewritten to dispatch on the variant or to take a `&dyn BackendOps` and delegate. S3.3 audit documents the call sites; S3.7e does the refactor. No API contract changes required on core (variants already exist as of S3.7a) — the work is all in engine + backend_metal.

- **Handle-lifecycle policy (carried forward, resolved for histograms; pending for row indices).** Histograms now use engine-side RAII drop guards (`HistogramReleaseGuard`) which call `BackendOps::release_histograms`. Row indices will need the same pattern — either a `BackendOps::release_row_indices(&PartitionResult)` + `PartitionReleaseGuard` following the same shape, or a lifetime-tracked `Arc<dyn ResidencyHandle>` on the Gpu variant. Recommend the former: it matches the already-shipped histogram pattern and doesn't cascade into core's `PartialEq` / `Debug` derives.

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- **Stage 3** — GPU residency (row partitioning + histograms + subtract) **(in flight — everything but S3.7e/S3.3/S3.10/S3.11/S3.12/S3.14/S3.15 done)**
- **Stage 4** — Metal 4 Indirect Command Buffer chaining (planned, not scoped)
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
