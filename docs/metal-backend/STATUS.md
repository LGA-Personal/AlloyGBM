# Metal Backend — Current Status

**Last updated:** 2026-04-28 (Stage 4a complete — GPU split finding shipped, kill criterion NOT MET, Stage 4b required)
**Active stage:** Stage 4b — Metal 4 ICB chaining (design not yet started)

---

## Stage 4a Checklist

Order matches the approved plan in
`docs/superpowers/plans/2026-04-26-stage-4a-gpu-split-finding.md`.

- [x] **Task 1** `BackendOps::find_best_splits_batch` trait method + scalar default + test.
- [x] **Task 2** `RuntimeBackend` forwarding for `find_best_splits_batch` (Python bridge).
- [x] **Task 3** `build_tree_level_wise` refactored to use `find_best_splits_batch` (batched call per level).
- [x] **Task 4** Metal profile counters: `FIND_BEST_SPLITS_BATCH` + `BS_DISPATCH` / `BS_COMMIT_WAIT` / `BS_DECISION_READBACK` / `BS_CATEGORICAL_HOST_MERGE`.
- [x] **Task 5** `SplitDecisionPool` (sibling to `HistogramResidencyPool`): `mint` / `buffer_for` / `read_decisions` / `release` + `SplitDecisionReleaseGuard`.
- [x] **Task 6** `shaders/best_split.metal` — `best_split_per_feature` (threadgroup-per-(node,feature), simdgroup prefix scan, NaN-left/right, Newton gain) + `best_split_reduce_features` (per-node cross-feature weighted-gain reduce).
- [x] **Task 7** Rust dispatch wrapper (`kernels/best_split.rs`) + `BestSplitPipelineCache` + `MetalBackend::find_best_splits_batch` numeric-only override + parity tests (3/3 pass).
- [x] **Task 8** Mixed-mode merge: GPU handles numerics, host runs Fisher-sort for categoricals, per-node merge picks winner. Added mixed-mode parity test (4/4 pass).
- [x] **Task 9** `metal_friendly_large` benchmark (1M×100, regression d=8) + D-024 entry + STATUS + SESSIONS.

**Verification (2026-04-28):**
- `cargo test -p alloygbm-backend-metal` — 53/53 pass (49 unit + 4 parity).
- `pytest bindings/python/tests/ -q` — 365/365 pass.
- `maturin develop --release` — clean build, no warnings.

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

Stage 4a successfully eliminated `best_split_with_options` as a bottleneck
(was 6.8% in Stage 3; now 0.7% via GPU kernel). The dominant cost is still
`build_histograms_batch.commit_wait` at 66.7% of total — each level requires
one synchronous CPU stall. `count_accumulate` is 16.4% and also CPU-bound.

**Residual gap:**

Eliminating `waitUntilCompleted` requires Stage 4b's ICB chaining: encode the
entire tree's histogram build + split-find + partition into a single
pre-committed Indirect Command Buffer per estimator, submitted once without
per-level CPU sync. `count_accumulate` and `apply_split` also benefit from
this architecture (GPU-side node selection, no per-level readback).

---

## Stage 4b — Next-Up: Metal 4 ICB Chaining

**Goal:** Remove `waitUntilCompleted` between levels by encoding one ICB per
tree (histogram build → GPU split-find → partition for all levels), committed
once per estimator. Expected to eliminate `build_histograms_batch.commit_wait`
(66.7% of total time).

**Prerequisites:**
- GPU-side node selection (which nodes are active at each level? — needs GPU
  tree-state buffer updated by the partition kernel).
- Per-tree pre-allocated worst-case buffers (depth × 2^depth nodes).
- Metal 4 availability check (M4 on macOS 26+; fallback to Stage 4a path on
  older hardware).

**Next action:** fresh brainstorm + spec document before starting implementation.

---

## Cross-Stage Roadmap (reference only)

- ~~**Stage 1** — GPU histogram build~~ *(shipped 2026-04-20)*
- ~~**Stage 2** — GPU best-split finder~~ *(shipped 2026-04-20)*
- ~~**Stage 3** — GPU residency (row partitioning + histograms + subtract)~~ *(closed 2026-04-25 — NOT MET)*
- ~~**Stage 4a** — GPU split finding (batched find_best_splits_batch)~~ *(closed 2026-04-28 — NOT MET)*
- **Stage 4b** — Metal 4 ICB chaining **(next-up — design not started)**
- **Stage 5** — GPU inference tree traversal (planned, not scoped)
