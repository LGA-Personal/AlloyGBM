# Metal Backend — Current Status

**Last updated:** 2026-04-21 (Stage 3 in-flight — S3.1/S3.2/S3.4–S3.9/S3.7a/S3.7b/S3.13 landed)
**Active stage:** Stage 3 — GPU residency (row partitioning + histograms + subtract)
**Active sub-task:** S3.3 (trainer-loop refactor threading `HistogramStorage` + `RowIndexStorage` through active-node tuples and PendingSplit) + S3.7c/d (bundled: `build_histograms` produces `Gpu(..)`, `best_split`/`subtract`/`apply_split` read pool buffers, Stage-2 ReusableSlots retired)

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
- [x] **S3.7b** (decomposed from S3.7) — `HistogramResidencyPool` skeleton in `backend_metal` with `mint` / `get` / `release` surface, overflow-checked shape math, integration with `ResidencyPool`. `MetalBackend` now carries a `residency: ResidencyPool` field + an `Arc<HistogramResidencyPool>` field; both are `#[allow(dead_code)]`-gated pending S3.7c. Four new unit tests cover round-trip, distinct handles, unknown-release no-op, and u32-squared-bytes overflow rejection. Commit: `ab3671f`.
- [ ] **S3.7c+d + S3.3 (bundled)** — `build_histograms` writes kernel output straight into pool-owned buffers and returns `HistogramStorage::Gpu(..)`; `best_split` reads pool entries directly (Stage-2 `split_grad`/`split_hess`/`split_counts` ReusableSlots are retired); `subtract_histogram_bundle` override dispatches the GPU subtract kernel when parent + child are both `Gpu(..)`; `apply_split` override returns `RowIndexStorage::Gpu(..)`. Trainer refactor flips every active-node tuple / PendingSplit field read to pattern-match on the storage variant; Cpu path remains semantically identical. These three sub-tasks are tightly coupled (handles must be live in the engine loop for the kernel reads to be meaningful) and ship as one atomic commit.
- [x] **S3.8** `MTLResidencySet` wrapper in `residency.rs` with `PassThrough` fallback for macOS 13/14 + no-Metal builds. Attach-on-construct / detach-on-drop lifecycle; 3 unit tests. Wired into `MetalBackend` as of S3.7b.
- [x] **S3.9** `BudgetTracker` in `budget.rs` enforcing the M2 free-on-consume policy: refuses the fit when projected peak (`F × B × L × 12`) exceeds 80 % of `MTLDevice.recommendedMaxWorkingSetSize`. Returns `EngineError::BackendUnavailable` with `device="cpu"` fallback guidance in the error message. 6 unit tests on the tracker + 3 integration tests exercising `MetalBackend::check_histogram_budget` on a real device. Pathological-shape risk note lives at the top of `budget.rs`. Commit: `495eefa`.
- [ ] **S3.10** Rust unit tests — residency round-trip (`build_histograms` → `best_split` → `subtract` → `best_split` on child, all values bit-exact after a single CPU read-back). Partition + subtract correctness tests already landed with S3.5 / S3.6.
- [ ] **S3.11** Python `MetalStage3Tests` — golden 50k × 100 × 255 × 20 fit pair with structural-plus-ulp gate, dedicated NaN-heavy / monotone-constraint / mixed-categorical (D-017 coverage) / memory-pressure (M2 budget guard fires cleanly) cases.
- [ ] **S3.12** Benchmark re-run on Apple M4 + new dated section in `BENCHMARKS.md`. Kill criterion: `metal_friendly` depth 8 + depth 10 + K=10 multiclass must all cross >1.0× CPU. Blocks stage close if it doesn't.
- [x] **S3.13** `DECISIONS.md` entries — D-015 (enum-variant storage API), D-016 (M2 residency budget), D-017 (categorical partition on GPU, split on CPU), D-018 (subtract / apply_split / reduce_sums promoted to `BackendOps`). Commit: `883d9ae`.
- [ ] **S3.14** `docs/limitations.md` Section 1 rewrite based on the S3.12 benchmark outcome.
- [ ] **S3.15** Full verification sweep + `STATUS.md` overwrite + `SESSIONS.md` Stage 3 close entry.

---

## Stage 3 — In Flight

What has shipped (counting everything on `claude/charming-carson-d08c9a` through `ab3671f`):

- Core storage enums (`HistogramStorage` / `RowIndexStorage`) with both `Cpu` and `Gpu` variants + opaque `u64` handle newtypes.
- `BackendOps` trait has `subtract_histogram_bundle` as a default method; trainer call sites call through the backend rather than the free function.
- `shaders/partition.metal` + `kernels/partition.rs` — row partitioning on GPU for both continuous and categorical splits. Live behind `MetalBackend::apply_split`; falls back to CPU on scan-cap overflow.
- `shaders/subtract.metal` + `kernels/subtract.rs` — elementwise subtract on GPU. `dispatch_subtract` is unit-test-reachable; the `BackendOps` override lands with S3.7c/d when parent + child are both `Gpu(..)`.
- `residency.rs` — `MTLResidencySet` wrapper + `PassThrough` fallback. Wired into `MetalBackend::new`.
- `budget.rs` — M2 free-on-consume working-set tracker. Wired into `MetalBackend::new`. `MetalBackend::check_histogram_budget(f, b, L)` is the public surface the trainer will call at fit start.
- `histogram_residency.rs` — GPU-resident histogram pool skeleton. Wired into `MetalBackend::new` behind `#[allow(dead_code)]`; S3.7c/d activates it.

What is left for the next session:

- **S3.7c+d + S3.3 (bundled)** — the next commit is the decisive one. It reroutes `build_histograms` output into pool-owned buffers, produces `HistogramStorage::Gpu(..)`, wires `best_split` / `subtract` / `apply_split` to consume handles, and threads the storage variants through the trainer loops. The audit shows that the engine is already largely storage-enum-aware in Cpu mode (≤10 `feature_histograms()` call sites; most are in subtract helpers + tests). The refactor is therefore mostly additive — add the Gpu arm handling next to the existing Cpu arm handling, with `CpuBackend` untouched.
- **S3.10 / S3.11** — kernel round-trip tests + Python golden pair. Both land after S3.7c/d so the Gpu path is actually exercised.
- **S3.12** — benchmark (kill criterion at this point). Stage 3 stands or falls by whether `metal_friendly` deep-tree configs cross >1.0× CPU; if they don't, dig in before declaring Stage 3 shipped.
- **S3.13 / S3.14 / S3.15** — docs + final verification sweep.

---

## Current Session Commits

- `1b5e511` — feat(core): Gpu variants on storage enums + handle newtypes (S3.7a)
- `ab3671f` — feat(backend_metal): HistogramResidencyPool skeleton + wiring (S3.7b)
- `0d23c66` — docs(metal-backend): STATUS + SESSIONS updated for S3.7a + S3.7b
- `883d9ae` — docs(metal-backend): DECISIONS D-015..D-018 for Stage 3 (S3.13)

Verification at end of session: workspace cargo check / clippy / fmt / test all green on both feature configs; 218 Rust tests pass (38 core + 69 engine + 36 backend_metal + 23 backend_cpu + 19 categorical + 10 predictor + 23 shap). Python tests are expected-green by construction (no CPU-path changes landed this session); full pytest sweep runs at S3.15.

---

## Blockers / Open Questions

- **S3.7c+d + S3.3 is a single-atomic-commit refactor** rather than three independent sub-tasks. The public-API surface flip (`build_histograms` returning `Gpu(..)`) is observable from the engine the moment it lands; there's no benign-no-op intermediate state unless we add a transitional `build_histograms_into_pool` dual method, which adds code to delete later. Decision for next session: ship it as one commit, with the commit message breaking out the three logical pieces for the archaeologist.

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- **Stage 3** — GPU residency (row partitioning + histograms + subtract) **(in flight — S3.1/S3.2/S3.4–S3.9 + S3.7a/b done)**
- **Stage 4** — Metal 4 Indirect Command Buffer chaining (planned, not scoped)
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
