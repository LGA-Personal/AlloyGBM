# Metal Backend — Session Log

Append-only. One entry per working session. Newest entries at the top.
First thing a new session reads, alongside `STATUS.md`.

---

## 2026-04-23 — Stage 3 continuation: S3.7c bundle + S3.7d lifecycle

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **S3.7c.1 — histogram reduce pass emits SoA** (commit `1199546`,
  D-019). Reduce kernel splits its single interleaved output buffer
  into two planes: a flat `f32[F*B]` grad plane and a flat `f32[F*B]`
  hess plane. This is the layout the downstream pool consumers expect
  (split kernel reads grad + hess as separate SoA buffers), so the
  flip eliminates one scatter-rewrite per subtract/split dispatch.

- **S3.7c.2 — residency pool API for GPU histograms** (commit
  `c2886af`). Pool's `mint` / `get` / `release` wired into concrete
  producer/consumer sites. `mint` now writes kernel-output bytes
  straight into pool-owned buffers (grad_out, hess_out, counts_in).
  Counts computed CPU-side per D-008 (threadgroup budget) and
  memcpy'd via `pool.write_counts`. `get` returns three Retained
  buffer handles the split/subtract kernels read directly.

- **S3.7c.3 — pool-resident GPU histograms end-to-end** (commit
  `1f3c0b7`, D-008 + D-012 + D-019). `MetalBackend::build_histograms`
  returns `HistogramStorage::Gpu { handle, feature_count, bin_count }`.
  `best_split` + `subtract` read pool entries directly — no CPU
  round-trip between histogram build and consumption. Stage-2
  ReusableSlots `split_grad`/`split_hess`/`split_counts` retired;
  `continuous_mask` kept (categoricals still take CPU branch for
  partition per D-012). This is the architectural flip the stage
  was designed around — GPU-resident histograms, not just
  CPU↔GPU marshalling around GPU kernels.

- **S3.7d — histogram handle lifecycle cleanup** (commit `a81a863`).
  Originally scoped to include GPU row-index pool + reduce_sums GPU
  arm + PartitionResult refactor. Narrowed on discovery: the engine's
  `apply_partition_leaf_updates` / `validate_partition_cover` /
  `into_cpu_parts` all call `.left_row_indices()` / `.right_row_indices()`
  eagerly, which panic on the Gpu arm — requires substantial engine
  refactor. Deferred to new **S3.7e**. What shipped instead: the real
  correctness concern carried over from S3.7c.3 — the residency pool
  leaked cumulatively within a fit because nothing called
  `pool.release` on consumed parent histograms. That breaks the M2
  peak-residency projection in `backend_metal::budget` which assumes
  one-level-wide peak. Fix: `BackendOps::release_histograms` trait
  method (default no-op, override on `MetalBackend` routes to pool
  release) + `HistogramReleaseGuard` RAII helper wrapping every
  per-node iteration in both trainer loops. RAII catches every
  continue / break / `?` / normal-return path deterministically.
  Leaf-wise gets queue-drain on early break (MaxLeavesReached) so
  `PendingSplit`s that were never popped also release. `RuntimeBackend`
  forwards the override. Two new unit tests assert pool live-count
  decrements on release and double-release is idempotent.

### Commits on this branch this session

- `1199546` feat(backend_metal): histogram reduce pass emits SoA (S3.7c.1, D-019)
- `c2886af` feat(backend_metal): residency pool API for GPU histograms (S3.7c.2)
- `1f3c0b7` feat(backend_metal): pool-resident GPU histograms end-to-end (S3.7c.3, D-008, D-012, D-019)
- `a81a863` feat(backend_metal): histogram handle lifecycle cleanup (S3.7d)

### Verification

- `cargo check --workspace` ✓ both feature configs
- `cargo clippy --workspace --all-targets -- -D warnings` ✓ both
- `cargo fmt --all --check` ✓
- `cargo test --workspace --exclude alloygbm-python` ✓ 220 tests pass
  (38 core + 69 engine + 38 backend_metal (+2 new release tests) +
  23 backend_cpu + 19 categorical + 10 predictor + 23 shap)
- `maturin develop --release` default features → `pytest` ✓ 362 passed, 20 subtests
- `maturin develop --release --no-default-features` → `pytest` ✓ 334 passed, 28 skipped, 16 subtests

### Scope decisions / deferrals

- **S3.7d narrowed from "histogram lifecycle + row-index pool + reduce_sums + PartitionResult refactor" to "histogram lifecycle only."** The row-index portion is substantially larger (engine-side refactor of three eager `.left_row_indices()` consumers) and independently valuable only once paired with `MetalBackend::apply_split` producing `Gpu(..)`. Ships as **new S3.7e**.
- **Handle-lifecycle pattern established:** engine-side RAII guard (`HistogramReleaseGuard`) calling `BackendOps::release_histograms`. S3.7e will mirror this shape for `RowIndexStorage::Gpu` (`BackendOps::release_row_indices` + `PartitionReleaseGuard`) rather than lifetime-tracked `Arc` on the Gpu variant (keeps core's `PartialEq` / `Debug` derives simple).

### Next up

- S3.3 trainer-loop audit (fast; pre-flight for S3.7e)
- S3.7e row-index residency pool + reduce_sums Gpu arm + engine refactor
- S3.10 residency round-trip Rust tests, S3.11 Python golden pair
- S3.12 benchmark (kill-criterion gate for Stage 3 close)

---

## 2026-04-21 — Stage 3 continuation: S3.7a + S3.7b

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **S3.7a — Gpu variants on storage enums + handle newtypes**
  (commit `1b5e511`). Extended Stage 3's existing `HistogramStorage` /
  `RowIndexStorage` scaffolding with the GPU-resident arms that the
  residency-pool wiring needs. Added `GpuHistogramHandle(u64)` and
  `GpuRowIndexHandle(u64)` newtype structs plus the matching `Gpu {
  handle, feature_count, bin_count }` / `Gpu { handle, row_count }`
  variants. All accessor methods (`cpu`, `cpu_mut`, `into_cpu`,
  `len`) now return None/0 on the Gpu arm — the engine asks the pool
  for the bytes, not the storage enum directly. The legacy
  `feature_histograms()` shim on `HistogramBundle` panics on the Gpu
  arm; that's deliberate (any caller still reaching for that shim
  in a GPU-residency context is a latent bug, not a fallback case).
  Six new core unit tests cover Gpu-arm handle round-trip, cpu/
  cpu_mut/into_cpu refusal, and legacy-shim panic behavior.

- **S3.7b — HistogramResidencyPool skeleton + wiring**
  (commit `ab3671f`). New `crates/backend_metal/src/histogram_residency.rs`
  with `HistogramResidencyPool` (Mutex-guarded `u64 → HistogramEntry`
  map), `HistogramEntry` (cloned `Retained<Buffer>` handles for
  grad / hess / counts + shape), and the `mint` / `get` / `release`
  lifecycle surface. `mint` allocates three `StorageModeShared`
  buffers (grad = f32, hess = f32, counts = u32 — all 4 bytes per
  cell), registers them with the backend's `ResidencyPool` so the
  working set keeps them pinned, and hands back an opaque
  `GpuHistogramHandle(u64)`. `release` detaches from residency and
  drops the buffers. Monotonic u64 token counter starting at 1
  (reserving 0 as a "no handle" sentinel) — wrap at > 500 years of
  fit lifetime isn't a practical concern.

  `MetalBackend` gained two new fields: `residency: ResidencyPool`
  (this also closes the S3.8 integration — the wrapper shipped in
  isolation in commit `281ae1a` with unit tests; this is the first
  wiring of it into the backend struct) and an
  `Arc<HistogramResidencyPool>`. Both are `#[allow(dead_code)]`-gated
  pending the S3.7c live consumer. Four new pool unit tests exercise
  mint/get/release round-trip, distinct token minting, unknown-handle
  release no-op, and shape-overflow rejection. The overflow test
  caught a real bug during development: the first version checked
  `feature × bin` but not `cells × 4 bytes`, so `u32::MAX × u32::MAX`
  slipped through the cell-count guard and wrapped during the
  per-plane byte calculation. Fixed by checking the full
  `feature × bin × BYTES_PER_CELL` chain.

- **Engine audit for S3.3 scope**. Surveyed the engine's uses of
  `feature_histograms()` and `row_indices()` to size the trainer-
  loop refactor. Found 8 `feature_histograms()` call sites total
  — 4 in the free-function subtract helpers, 4 in unit tests — and
  1 `.row_indices()` method reference (vs ~30 uses of the
  storage-aware `row.cpu()` / `partition.into_cpu_parts()` pattern
  that's already in place). The engine is already largely
  storage-enum-aware in Cpu mode; adding the Gpu arm is additive
  next to existing Cpu arm code, not a rewrite. This lets S3.3
  bundle cleanly with S3.7c/d as a single atomic refactor in the
  next session.

### What did not move (and why)

- **S3.7c+d + S3.3 (bundled)**. The three sub-tasks are tightly
  coupled: flipping `build_histograms` to return `Gpu(..)` is
  observable from the engine the moment it lands; without the
  trainer refactor in the same commit every call site that
  currently reads `bundle.feature_histograms()` crashes. Decision:
  ship it as one atomic commit in the next session rather than
  add a transitional dual-method path that would need to be
  deleted later. STATUS.md captures the next-session scope in
  detail. The audit above shows the engine is already mostly
  ready for the flip.

### Verification

- `cargo check --workspace` green on both feature configs.
- `cargo clippy --workspace --all-targets -- -D warnings` clean on
  both feature configs.
- `cargo fmt --all --check` clean.
- `cargo test --workspace --exclude alloygbm-python` — 218 tests
  pass (up from Stage 2 close's 200-odd): 38 core (+6 Gpu-arm),
  69 engine, 36 backend_metal (+4 residency-pool), 23 backend_cpu,
  19 categorical, 10 predictor, 23 shap.
- Full Python pytest sweep deferred to S3.15 (the Python surface
  did not change this session).

### Commits

- `1b5e511` — feat(core): Gpu variants on storage enums + handle newtypes (S3.7a)
- `ab3671f` — feat(backend_metal): HistogramResidencyPool skeleton + wiring (S3.7b)

### Blockers

None. S3.7c/d + S3.3 bundled atomic commit is the next sub-task.

---

## 2026-04-20 — Stage 2 shipped (S2.1–S2.8) → Stage 2 closed

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **S2.1 MSL split kernel** (`crates/backend_metal/src/shaders/split.metal`).
  Three-phase: per-feature grad/hess/count totals via SIMD reduction;
  block-scan prefix sweep using `simd_prefix_inclusive_sum` + threadgroup
  consolidation; per-threshold gain body (XGBoost form, L1 soft-threshold
  behind `L1_ENABLED` function constant, NaN-left / NaN-right both scored,
  `min_child_hessian` / `min_leaf_magnitude` rejection); SIMD butterfly
  argmax within threadgroup. Deterministic tie-break:
  `has_split > higher gain > lower threshold > default_left=1`. No float
  atomics. Function constants `BIN_COUNT` (0) and `L1_ENABLED` (1).
- **S2.2 Rust-side orchestration** (`crates/backend_metal/src/kernels/split.rs`).
  `dispatch_best_split_per_feature` flattens the `HistogramBundle`
  grad / hess / counts arrays into `[n_features × bin_count]` SoA slabs in
  `BufferCache` ReusableSlots, encodes a single dispatch with grid
  `(continuous_feature_count, node_count, 1)`, reads back u32-packed
  `(feature_index, threshold_bin, default_left, has_split)` plus f32 gain
  per `(feature, node)`. See DECISIONS.md D-014 — collapsed the plan's
  original two-dispatch design (GPU cross-feature argmax) to one GPU
  dispatch + CPU weighted argmax, because `feature_weights` multiplication
  is trivial on CPU and the dispatch round-trip for ≤200 features was not
  worth the plumbing.
- **S2.3 Pipeline + buffer cache extensions.** `SplitPipelineCache` keyed
  by `(bin_count, l1_enabled)`, own `MTLBinaryArchive` file alongside the
  histogram archive. `BufferCache` grew `split_grad`, `split_hess`,
  `split_counts`, `continuous_mask` ReusableSlots following the
  established gradients / row-indices pattern.
- **S2.4 `BackendOps::best_split` / `best_split_with_options` overrides on
  `MetalBackend`.** Continuous features flow through the GPU kernel;
  categorical features delegate to the embedded `CpuBackend`'s
  `best_split_for_categorical_feature`; CPU-side combiner applies
  `feature_weights` and picks the overall winner. Empty-continuous-set
  and empty-node-set fast paths delegate directly to CPU. See D-012.
- **S2.5 Rust unit tests** — 6 new cases in `crates/backend_metal/src/lib.rs`:
  small-fixture parity, L1+L2, feature weights, NaN bin direction,
  categorical-delegation, split-pipeline-cache identity.
  `assert_structural_equality` helper asserts `(feature_index,
  threshold_bin, default_left, is_categorical)` exactly and gain
  relative drift `<1e-4`.
- **S2.6 Python tests** — added `MetalStage2Tests` (4 seeded model pairs
  at 50k × 100 × 255 bins × 20 estimators: regression, L1+L2 regression,
  binary classifier; ranker at 1k × 8 × 25 groups × 10 estimators) and
  `MetalStage2NanAndMonotoneTests` (NaN-heavy regression, monotone
  constraints). Stage 1's `test_bin_count_16_matches_cpu`,
  `test_bin_count_255_matches_cpu`, and the predict_proba arm of
  `test_small_classification_matches_cpu` relaxed to match the Stage 2
  SIMD tree-reduction ulp envelope (see D-013). Golden-shape tests keep
  `atol=1e-5, rtol=1e-5`; tiny-shape tests use `atol=0.1, rtol=0.1`.
- **S2.7 benchmark re-run.** Appended "2026-04-20 — Apple M4 (Stage 2
  baseline)" section to `BENCHMARKS.md` with `shape_grid` and
  `metal_friendly` tables. **Crossover NOT achieved.** Stage 2 numbers
  within run-to-run jitter of Stage 1 (whole-fit ratios moved ≤0.03×
  either direction). Hypothesis written up: per-node GPU dispatch +
  HistogramBundle memcpy per `best_split` call dominate the CPU compute
  savings. At depth 10 × 200 features: ~5 MiB × ~5000 calls ≈ 25 GiB of
  memcpy per fit + 10–50 μs/dispatch fixed latency. Decisive win
  architecturally gated on Stage 3's GPU row partitioning + Metal 4 ICBs.
  `DECISIONS.md` grew D-011 (GPU subtract deferred), D-012 (categorical
  on CPU), D-013 (bit-exactness → structural-plus-ulp), D-014
  (single-dispatch split kernel). `docs/limitations.md` section 1
  rewritten to reflect Stage 2 scope + numerical parity contract.
- **S2.8 verification sweep:**
  - `cargo check --workspace` + `--no-default-features`: both green.
  - `cargo clippy --workspace --all-targets -- -D warnings` + the
    `--no-default-features` variant: both clean.
  - `cargo fmt --all --check`: clean.
  - `cargo test --workspace --exclude alloygbm-python`: passes; 13
    `backend_metal` tests (7 Stage 1 + 6 new Stage 2).
  - `.venv/bin/maturin develop --release` (default features):
    **362 passed + 20 subtests passed**. All 30 Metal-gated cases green,
    including the 4 new Stage 2 model-pair fits and the 2 new NaN /
    monotone cases. 332 pre-existing cases untouched.
  - `maturin develop --release --no-default-features`:
    **334 passed + 28 skipped**. Every Metal-gated Stage 2 case skips
    cleanly alongside Stage 1.
- **STATUS.md** overwritten with Stage 2 CLOSED; Next Up points at Stage 3.

### Scope decisions (recorded in DECISIONS.md)

- **D-011** — GPU `subtract_histogram_bundle` deferred to Stage 3. The
  subtract is F×B elementwise ops (~25k for 100 features × 255 bins) —
  not a hot spot on its own. Net-positive GPU subtract requires
  histograms to live on GPU across calls, which itself needs the
  Stage 3 surface change of passing handles instead of owned
  `HistogramBundle`s.
- **D-012** — Categorical features stay on CPU. Fisher-sort
  optimal-binary-partition + prefix scan on categories is a separate
  research problem; `MetalBackend` partitions the feature set at
  runtime and combines candidates on CPU.
- **D-013** — Bit-exactness gate relaxed from Stage 1's
  byte-identical-artifact contract to structural-plus-ulp-epsilon:
  identical `(feature_index, threshold_bin, default_left)` per split
  AND predictions within `atol=1e-5, rtol=1e-5` at golden shape;
  tiny-shape tests acknowledge legitimate ulp drift on near-tied
  gains (can flip threshold_bin and cascade to macroscopic deltas on
  ≤0.1% of rows).
- **D-014** — Split kernel uses a single GPU dispatch emitting
  per-(feature, node) candidates; CPU consolidates with
  `feature_weights`-weighted argmax. Collapses the plan's original
  two-dispatch design.

### Verification

Cargo + maturin + pytest matrix above; structural-equivalence gate holds
on the 50k × 100 × 255 × 20 golden fits for regression, L1+L2 regression,
binary classifier, and the 1k × 8 × 25 ranker. `trained_device="metal"`
round-trips through `save_model` / `load_model` for every Stage 2 fit
(asserted by `test_metal_fits_record_trained_device_metal`).

### What did NOT regress

- Stage 1 golden bit-exact test (50k × 100 × 255 × 20) still passes
  under `array_equal` — the histogram kernel path is untouched.
- `ALLOYGBM_METAL_DISABLE=1` escape hatch still triggers the
  `RuntimeWarning` and falls back to CPU.
- Artifact metadata round-trip for `trained_device` unchanged from Stage 1.
- `native_runtime_info()` shape unchanged.

### What did NOT happen

- **No throughput crossover.** The plan projected `>1.0×` on
  `metal_friendly`'s two deepest configurations. Did not land. This is
  plan-consistent — D-011 defers the subtract, and without GPU-resident
  histograms the per-level memcpy + dispatch tax scales with the CPU
  compute savings. Stage 3 is the stage that unlocks crossover.
- **No Stage 3 work started.** Ships via its own `ExitPlanMode`.

### Commits

Stage 2 work is staged in the worktree; commits will land in
checklist-order (S2.1 → S2.8) once the user reviews the diff.

### Blockers

None. Open items — if any — will go to `BUGS.md`.

---

## 2026-04-20 — S1.16 full verification sweep → Stage 1 closed

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- Full cargo/clippy/fmt/test sweep across the workspace, both with
  default features and `--no-default-features`. All green. Test
  counts: 183 Rust tests (including 7 in `backend_metal`).
- `.venv/bin/maturin develop --release --manifest-path
  bindings/python/Cargo.toml` under both feature configurations.
- Full pytest run:
  - Default features: **353/353 pass** (21 Metal cases + 332
    pre-existing).
  - `--no-default-features`: **334 pass + 19 skipped** (Metal-gated
    cases correctly skip).
- **`bindings/python/tests/test_metal_backend.py`** — added
  `_metal_feature_compiled_in()` probe (one subprocess run with
  `ALLOYGBM_METAL_DISABLE=1`) and an `@unittest.skipUnless` on
  `MetalFallbackTests`. Without this gate, a `--no-default-features`
  build failed `test_fallback_emits_runtime_warning` because the
  escape-hatch warning text only exists when the Metal feature is
  compiled in — the no-feature build emits a different, correct
  warning ("this build does not include the Metal backend") but
  doesn't name the env var.
- **`docs/metal-backend/STATUS.md`** — S1.16 ticked; Stage 1 marked
  CLOSED; new "Stage 1 — Complete" section summarises what shipped;
  Next Up now points to Stage 2 `ExitPlanMode`.

### Verification

All of the above is the verification — no code paths added, only
the test-suite gate needed to make `--no-default-features` honest.

### Findings

- **PyO3 extension-module crates don't `cargo test` standalone on
  macOS.** Expected — the test binary can't link without the Python
  framework dance. We validate that crate via pytest (which loads
  the extension the normal way). This is consistent with how every
  prior Stage 1 sub-task verified the binding layer.
- **The `metal_feature_compiled_in()` probe is the cleanest gate we
  have** given that `native_runtime_info()` can't distinguish
  "feature off" from "feature on, no hardware" — both surface as
  `metal_available=False, gpu_family=None`. Adding a
  `metal_feature_enabled` field to `NativeRuntimeInfo` would have
  been the more principled path but would have expanded scope into
  a new public API at the final gate; the probe is test-local,
  costs one subprocess, and doesn't leak into the user-facing
  runtime info surface.

### Design calls

- **S1.16 fix stayed in the test module, not in `NativeRuntimeInfo`.**
  Reasoning above. If Stage 2 needs programmatic feature-vs-hardware
  detection for something else, revisiting the API is fair game
  then; right now it's YAGNI.
- **`cargo test --workspace --exclude alloygbm-python` rather than
  fixing the linker dance.** The Python crate is exercised via
  pytest; duplicating the coverage through `cargo test` wasn't in
  scope for S1.16 and would have added CI complexity for zero
  incremental coverage.

### Handoff notes

- **Stage 1 is closed.** The next session should open Stage 2 via a
  fresh `ExitPlanMode` round. Stage 2's scope is already sketched
  in the approved plan: MSL prefix-scan + argmax kernel, elementwise
  subtract for `subtract_histogram_bundle`, level-parallel dispatch
  (threadgroup.y = sibling node). Expected incremental gain: 2-3×.
- **Everything Stage 1 shipped stays in place.** BufferCache,
  pipeline cache, warn-and-fallback, `device=` plumbing,
  `native_runtime_info()` fields, `trained_device` metadata, and
  the bit-exactness golden test all continue to apply.
- **No open `BUGS.md` items from this session.**
- **Key working-set files for the Stage 2 session-start:**
  `STATUS.md` (now reflects closed Stage 1), this `SESSIONS.md`
  (top entry summarises what Stage 1 delivered), `DECISIONS.md`
  (design calls made along the way), the plan at
  `/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md`.

---

## 2026-04-20 — S1.15 `BufferCache` wired + benchmarks re-run

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`crates/backend_metal/src/buffers.rs`** (new). Persistent Metal
  buffer pool for the histogram dispatch path:
  - Binned matrix is keyed by `(ptr, len_bytes, is_wide)` and reused
    zero-copy across all `build_histograms` calls within a fit
    (~63 calls per depth-6 tree × N trees). Safe because the binned
    matrix is immutable for the lifetime of the fit.
  - Gradients + row-indices slots hold a reusable `MTLBuffer`
    allocation and memcpy fresh bytes into it per call. Slots grow
    monotonically — smaller subsequent requests reuse the existing
    allocation.
  - `#![allow(unsafe_code)]` at the top to opt out of the crate-local
    `deny`, matching the pattern in `pipelines.rs`.
- **`crates/backend_metal/src/lib.rs`** — registers `mod buffers`,
  owns `buffer_cache: Arc<BufferCache>` on `MetalBackend`, threads
  `&self.buffer_cache` into the `dispatch_histograms` call.
- **`crates/backend_metal/src/kernels/histogram.rs`** —
  `dispatch_histograms` signature gains a `buffer_cache: &BufferCache`
  parameter, visibility tightened to `pub(crate)`; three
  `newBufferWithBytes` sites (binned matrix, gradients, row indices)
  replaced with cache-backed variants; orphaned
  `make_buffer_from_slice` helper removed.
- **`benchmarks/metal_histogram.py`** — the S1.14 shape-only grid
  became a named-scenario harness: `shape_grid`, `depth_sweep`,
  `bins_sweep`, `estimator_sweep`, `task_mix`, `metal_friendly`,
  `all`. The `metal_friendly` scenario is the direct test of the
  "Stage 1 wins somewhere" hypothesis (deep trees up to depth 10,
  1024 bins, multiclass K=10 — configs theoretically best for
  histogram-heavy work).
- **`docs/metal-backend/BENCHMARKS.md`** — full rewrite against
  post-cache numbers. Two tables (`shape_grid` + `metal_friendly`),
  reproduction commands now cite `--scenario`, interpretation section
  explains why Stage 1 loses on `metal_friendly` too.
- **`docs/metal-backend/metal_histogram_shape_grid_m4.json`** and
  **`docs/metal-backend/metal_histogram_metal_friendly_m4.json`** —
  raw benchmark output archived alongside the markdown.
- **`docs/limitations.md`** — "Metal Backend is Infrastructural
  (Stage 1)" section updated to cite both scenarios; speedup
  range moved from "0.03×–0.25×" to "0.03×–0.28× (shape grid),
  0.06×–0.09× (metal_friendly)".
- **`docs/metal-backend/STATUS.md`** — S1.15 bullet expanded to
  describe the full scope (buffer cache + harness + docs); Next
  Up now points only to S1.16.

### Verification

- `cargo test -p alloygbm-backend-metal` — all 7 tests pass,
  including the `histogram_matches_cpu_small_fixture`,
  `histogram_feature_subset_matches_cpu`, and
  `pipeline_cache_returns_identical_arc_on_second_call` cases.
- `/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release
  --manifest-path bindings/python/Cargo.toml` — clean build.
- `.venv/bin/python -m pytest bindings/python/tests/test_metal_backend.py
  -q` — all 21 cases pass, including the `MetalGoldenTests`
  50k-row × 100-feature × 20-estimator bit-exactness golden test
  on both the training set and the held-out eval set.
- Benchmark runs: `shape_grid` scenario (~4m) and `metal_friendly`
  scenario (~2m) on Apple M4. JSONs archived in
  `docs/metal-backend/`.

### Findings

- Buffer-cache wall-clock win is **real but modest**: 5–20%
  improvement on Metal times vs the S1.14 pre-cache baseline.
  Largest absolute saving: 1M × 1000 dropped from 86.8s → 70.7s
  (~16 s recovered). Largest ratio shift: that same cell moved
  0.17× → 0.20×. Overall the speedup band moved from 0.03×–0.25×
  to 0.03×–0.28× across the shape grid.
- **`metal_friendly` decisively rules out a Stage 1 sweet spot.**
  Deep trees (depth 10), wide bins (1024), and 10-way multiclass
  (K histograms per round) all keep Metal at 0.06×–0.09× CPU.
  This is the strongest evidence yet that the CPU round-trip per
  level (split finding + partitioning) — not the binned-matrix
  memcpy — is the Stage 1 bottleneck, and that Stages 2+3 are
  structurally required for the decisive win.
- Bit-exactness survives the cache cleanly: no behavioural change
  to the kernel itself, only the buffer lifetime changed.

### Design calls

- **Cache keyed by `(ptr, len, is_wide)`, not by a logical fit ID.**
  The engine hands the backend the same `BinnedMatrix` reference
  on every call within a fit, so the pointer is stable. Key-based
  reuse means `MetalBackend` needs no "fit start / fit end"
  lifecycle hook from the engine — a zero-change contract for
  the caller.
- **Gradients + row-indices re-use allocation but not content.**
  These buffers' bytes change every boosting round (and every
  node within a level for row indices), so caching by pointer
  would be wrong. Keeping the `MTLBuffer` handle and rewriting
  its contents is the right middle ground — avoids the
  `newBufferWithLength` cost without correctness risk.
- **Benchmark scope: `metal_friendly` rather than more points on
  `shape_grid`.** The question "does Stage 1 ever win?" is
  better answered by five carefully-chosen adverse configs than
  by filling in more cells of the original grid. If those five
  configs all lose, the hypothesis is dead and we don't need
  more evidence.
- **Docs cite both scenarios.** `limitations.md` names
  `metal_friendly` explicitly so a reader who goes looking for
  "the corner where Metal wins" sees the scenario that was
  designed to find it — and sees that it didn't.

### Handoff notes

- **S1.16 is the last Stage 1 gate.** Required checks:
  `cargo check --workspace` and `cargo check --workspace
  --no-default-features`; `cargo test --workspace` both with and
  without default features; `cargo clippy --workspace --all-targets
  -- -D warnings`; `cargo fmt --all --check`; `maturin develop
  --release` (default features + `--no-default-features`); full
  `.venv/bin/python -m pytest bindings/python/tests/ -q`.
- On S1.16 success, Stage 1 closes. The next `ExitPlanMode` round
  opens Stage 2 (GPU best-split + histogram subtraction). Stage 2
  is the first stage where the `metal_friendly` configurations
  become the *positive* case — the break-even point should move
  materially once per-level CPU round-trips are eliminated.
- No open BUGS.md entries from this session. The `make_buffer_from_slice`
  helper that was removed was only used by the old
  `dispatch_histograms` path; no other references in the crate.

---

## 2026-04-20 — S1.14 `metal_histogram.py` throughput harness

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`benchmarks/metal_histogram.py`** (new, standalone CLI).
  - Argparse grid selector: `--rows`, `--features`, `--full` (adds
    the plan-spec 10M row tier), `--estimators` (default 5),
    `--bins` (default 255 → u8 bin path), `--seed` (default 7).
  - `--memory-budget-gb` (default 8 GB) skips any shape whose
    float32 input exceeds the budget. Default grid's largest
    cell is 1M × 1000 = 3.8 GiB, which fits; 10M × 1000 is
    ~40 GB, which does not.
  - Pipeline warmup fit at (1024, 4) so the first real cell
    doesn't absorb Metal's one-time pipeline-compilation cost.
    `--no-warmup` toggles it off for debugging.
  - Output: markdown table on stdout + optional `--json-out`
    mirror with full float-precision timings and metadata
    (`gpu_family`, `metal_available`, `metal4_available`,
    `estimators`, `bins`, `seed`, `memory_budget_gb`).
  - `--skip-metal` runs only the CPU leg (for harness debugging).
- **`docs/metal-backend/BENCHMARKS.md`** (new). Reference numbers
  captured on Apple M4 for the default grid, with interpretation
  for S1.15 to cite.
- **`docs/metal-backend/STATUS.md`** — S1.14 checked off;
  promoted S1.15 to "Next Up".

### Verification

- Two smoke runs:
  - `--rows 5000 20000 --features 10 50 --estimators 3 --no-warmup`
    — confirms the full pipeline runs end-to-end in ~1s.
  - `--rows 100000 1000000 --features 100 --memory-budget-gb 0.1
    --no-warmup` — confirms the budget-skip path marks the
    1M-row cell as skipped with a "0.4 GB exceeds" note.
- Full default grid run: completed in ~2m20s on Apple M4.
  Results captured in `BENCHMARKS.md`.

### Findings

- Stage 1 whole-fit wall-clock is **uniformly slower** on Metal
  across (10k, 100k, 1M) × (10, 100, 1000) × 5 estimators. Ratio
  ranges from 0.03× (worst: 10k × 1000) to 0.25× (best: 1M × 10).
- Matches the expert-session expectation: histogram acceleration
  alone doesn't pay off until the histogram phase dominates the
  inner loop, which it doesn't when each boosting round still
  round-trips through the CPU for split finding + partitioning.
- Not a regression — it's the architectural reality of Stage 1
  and is exactly why Stage 2 (GPU best-split) and Stage 3 (GPU
  partitioning + ICBs) are the next stages in the roadmap.

### Design calls

- **Whole-fit wall-clock, not histogram-only.** AlloyGBM doesn't
  expose a histogram-only entry point to Python, so we'd have
  had to plumb one through PyO3 purely for this benchmark.
  Whole-fit is also what the user actually observes when they
  flip the `device=` flag; reporting the synthetic histogram-only
  number would have been misleading.
- **Memory budget rather than automatic dtype shrinkage.** The
  clean way to scale to 10M × 1000 is a 64 GB host, not quantising
  the fixture to float16 — that would change the cache behaviour
  and stop being comparable to the default `device="cpu"` path
  that users actually run.
- **Default estimators=5, not 20 or 100.** At 5 rounds we surface
  the per-round dispatch overhead cleanly; at 100 rounds the
  grid would have taken 40 minutes. Users who want a more
  realistic fit-time comparison can pass `--estimators 100`.
- **Warmup fit is separate from the grid.** Metal's first fit
  pays a pipeline-compile cost (~0.5-1s on Apple M4). Including
  it in the grid's first cell would have reported a 3× worse
  number for that cell and given users the wrong mental model.

### Handoff notes

- S1.15 is next: `docs/limitations.md` breakeven + availability
  note. The `BENCHMARKS.md` numbers are the citation source.
  Recommendation to document: `device="cpu"` is the default and
  stays recommended for every shape in the Stage 1 grid;
  `device="metal"` is infrastructural in Stage 1 (proves
  plumbing, unblocks Stages 2+3) and becomes advantageous after
  Stage 2 + Stage 3 land.
- `native_runtime_info()` fields (`metal_available`,
  `metal4_available`, `gpu_family`) are the supported way for
  user code to detect the backend — include that in S1.15's
  "how to detect" section.
- `ALLOYGBM_METAL_DISABLE=1` is an *internal* escape hatch for
  testing. It's fine to mention in user docs as a "force CPU"
  override, but the sanctioned user-facing control is
  `device="cpu"`.

---

## 2026-04-20 — S1.13 Bit-exactness golden test at scale

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`bindings/python/tests/test_metal_backend.py`** — new class
  `MetalGoldenTests` (3 cases) with the S1.12 `@unittest.skipUnless(
  _METAL.metal_available, _SKIP_REASON)` gate at class level.
  - `setUpClass` generates a seeded (50k × 100) training matrix
    plus a held-out 5k-row eval matrix, then fits one CPU and one
    Metal `GBMRegressor` at `n_estimators=20, seed=7,
    deterministic=True, continuous_binning_max_bins=255`. Fitting
    happens once per class (Metal fit is ~5s; per-test fits would
    have tripled the cost for no signal).
  - `test_golden_bitexact_predictions_on_training_set` —
    `assert_array_equal` across all 50k training rows.
  - `test_golden_bitexact_predictions_on_heldout_set` —
    `assert_array_equal` on the 5k held-out rows, exercising
    tree traversal on previously-unseen data.
  - `test_golden_trained_device_recorded_in_artifact` — asserts
    the metadata JSON correctly round-trips `"trained_device":"cpu"`
    and `"trained_device":"metal"` respectively.
- **`docs/metal-backend/STATUS.md`** — checked S1.13 off, promoted
  S1.14 to the "Next Up" spot.

### Verification

- `pytest bindings/python/tests/test_metal_backend.py::MetalGoldenTests
  -v` — 3 passed in 5.49s.
- `pytest bindings/python/tests/test_metal_backend.py -v` — 21
  passed in 8.51s (was 18; +3 new).
- `pytest bindings/python/tests/ -q` — **353 passed, 16 subtests
  passed** (was 350 pre-S1.13; zero regressions).

### Design calls

- **Scope adjusted from the plan's original "identical
  `artifact_bytes`" gate.** S1.12 had already proved that contract
  is unreachable: the metadata JSON encodes `trained_device` and
  its length prefix, so Metal vs CPU artifacts legitimately differ
  by a handful of bytes in the header. Prediction bit-exactness
  over every training row — plus the held-out eval rows — is the
  stronger observable contract and is what this test asserts.
- **Shared `setUpClass` rather than per-test fit.** The fit pair
  costs ~5.5s total (CPU 0.3s + Metal 5.2s). Three tests each
  re-fitting would have cost ~16s and produced identical models
  by construction; one fit feeding three assertions keeps the
  signal intact while respecting the default pytest budget.
- **Held-out eval set comes from a distinct RNG draw, not a
  distinct seed.** Using the same `RandomState` instance for both
  train and eval means the eval rows are genuinely independent
  draws in the same distribution — which is what you want to
  stress float-ordering variance during predict-time traversal.
- **No slow-gate marker.** 5.5s is well under the implicit pytest
  budget for the Metal module (the warn-and-fallback subprocess
  tests already cost ~5s each). Adding `ALLOYGBM_SLOW` plumbing
  for a single 5s test would have been premature.

### Handoff notes

- S1.14 is next: `benchmarks/metal_histogram.py` for throughput
  characterisation. The golden test at (50k × 100, 20 estimators)
  saw CPU 0.27s vs Metal 5.28s wall-clock for the full training
  loop, which matches the expert prediction that the CPU wins
  below ~250k rows. The benchmark should confirm the crossover
  and produce the numbers that feed into S1.15's
  `docs/limitations.md` note.
- The `MetalGoldenTests` shape (50k × 100) is the shape agreed
  with the expert sessions as the smallest "stage-realistic"
  workload; when S1.14 lands, add its smallest-to-largest grid
  around that anchor.

---

## 2026-04-20 — S1.12 Python Metal backend test module

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`bindings/python/tests/test_metal_backend.py`** (new, 18 cases).
  Six test classes:
  - `MetalAvailabilityTests` — smoke-tests the capability probe when
    Metal is actually available (non-empty `gpu_family`, Apple7
    baseline met).
  - `MetalRegressionTests` — CPU/Metal prediction bit-exactness
    over the whole training set, `trained_device` recording, and
    a multi-column feature sanity check.
  - `MetalClassificationTests` — `predict` + `predict_proba`
    parity on a binary-classification fixture.
  - `MetalRankerTests` — `rank_ndcg` parity on grouped data.
  - `MetalEdgeCases` — NaN handling, single-row, single-feature,
    and bin counts 16/255/1024 (straddling the u8/u16 bin-storage
    switchover).
  - `MetalFallbackTests` — subprocess-isolated warn-and-fallback
    checks using the S1.9 `ALLOYGBM_METAL_DISABLE=1` escape hatch;
    verifies `RuntimeWarning` emission, artifact records
    `trained_device="cpu"` after fallback, and estimator's
    user-visible `device` attribute stays `"metal"`.
  - `InvalidDeviceTests` — unknown device strings raise
    `ValueError`; `device="auto"` is accepted and round-trips.
  Top-level gate: `native_runtime_info().metal_available` via
  `@unittest.skipUnless` at class level for the hardware-dependent
  cases; fallback tests and invalid-device tests run unconditionally
  because they don't touch real Metal resources.

### Verification

- `pytest bindings/python/tests/test_metal_backend.py -v` — **18
  passed** on Apple M4.
- `pytest bindings/python/tests/ -q` — **350 passed, 16 subtests
  passed** (was 332 pre-S1.12; +18 new tests, no regressions).
- Fallback tests pass in ~5s because they shell out to subprocesses;
  that overhead is intentional (see Design calls below).

### Issues encountered and fixed

- **`assertEqual(cpu_bytes, metal_bytes)` failed** on the initial
  draft. The CPU artifact recorded `"trained_device":"cpu"` (3
  chars) and the Metal artifact recorded `"trained_device":"metal"`
  (5 chars). The artifact format length-prefixes its metadata JSON,
  so a regex strip of the `trained_device` field still left the
  length-prefix bytes at a ~2-byte offset. Rewrote the test to
  compare predictions over the whole training set instead — that's
  the observable bit-exactness contract that users actually care
  about, and doesn't require monkeying with the artifact format.
- **`AttributeError: 'list' object has no attribute 'shape'`** on
  the single-row test. `GBMRegressor.predict` returns a list when
  fed a 1-row input. Wrapped with `np.asarray(..., dtype=np.float64)`
  so the test is robust to either return shape.

### Design calls made this session

- **Subprocess-isolated fallback tests.** The Metal backend caches
  the `MTLDevice` the first time `device="metal"` is resolved. If
  we toggled `ALLOYGBM_METAL_DISABLE` inside the live test process,
  test-order sensitivity would bite: after one test sees Metal, a
  second test with the env var wouldn't always hit the disable
  code path. Shelling out to a fresh Python interpreter gives each
  fallback test a pristine process where the env var is honored
  deterministically. Cost is ~200 ms per subprocess; three tests
  total, so ~0.6 s overhead — acceptable.
- **Predict-over-training-set as the bit-exactness contract.** The
  plan's original phrasing was "identical `artifact_bytes`", but
  that is not achievable as-written because `trained_device` legitimately
  differs between CPU and Metal runs. `np.testing.assert_array_equal`
  over all training rows (not predictions on a subset) is the strongest
  observable parity check and it uncovers any numerical drift that
  would show up downstream.
- **Bin-count coverage at 16/255/1024, not 65535.** The plan
  mentioned B=65535 as the upper u16 endpoint, but the histogram
  cache's `FEATURE_TILE_SIZE` specialisation happens at compile
  time for {1, 4, 16, 64}, and the u8/u16 storage switchover at
  256 is the only numerically interesting boundary for the kernel
  itself. B=1024 exercises the u16 path (kernel selects the u16
  specialisation via function constants) without demanding the
  kernel fan-out of B=65535 which would stress the benchmark
  harness (S1.14) more than the correctness surface.
- **Class-level `@unittest.skipUnless`, not module-level pytest
  marker.** Keeps the file runnable under plain `python -m unittest`
  and matches the pattern already used elsewhere in the suite
  (e.g. `test_native_runtime_integration.py`). Class-level gives
  finer-grained control than module-level: `MetalFallbackTests`
  and `InvalidDeviceTests` intentionally run regardless of
  hardware presence.

### Handoff notes for S1.13

- **Scale.** Plan calls for (50k rows × 100 features × 255 bins).
  Expect ~30-60 s per fit on M-series; two fits plus prediction
  stream compare fits comfortably inside a ~2-minute test budget.
  If the test exceeds 120 s consider marking it `@unittest.skip`
  when `--quick` is passed (or just accept the cost — CI already
  tolerates long-running tests elsewhere in the suite).
- **Contract.** S1.12 established that `assert_array_equal` over
  the full training set is the bit-exactness gate. S1.13 should
  reuse that shape at scale, plus a second assertion on a held-out
  eval set to stress robustness under float-ordering variance.
- **Determinism requirement.** Must set `seed=...`,
  `deterministic=True`, and `n_jobs=1` (if CPU is non-deterministic
  across threads). The Metal kernel is deterministic by
  construction (two-pass reduce, no float atomics), so the CPU
  side is the only place we need to pin.
- **Artifact audit.** Separately check that after a Metal fit the
  metadata JSON records `"trained_device":"metal"` (already
  covered by `MetalRegressionTests.test_metal_regression_records_trained_device`
  at small scale; can reuse the assertion at 50k-row scale for
  free).

---

## 2026-04-19 — S1.10 Metal capability fields on `native_runtime_info()`

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`crates/backend_metal/src/device.rs`** — new module-level
  `probe_capabilities() -> Option<MetalCapabilities>` that performs
  the same capability read as `MetalDevice::probe()` but *without*
  opening a command queue or holding the device. Used by
  `native_runtime_info()` so a harmless introspection call doesn't
  keep Metal resources resident for the process lifetime. Returns
  `None` on headless VMs / non-Metal Macs.
- **`crates/backend_metal/src/lib.rs`** — re-exports the new
  `probe_capabilities` alongside `MetalCapabilities` / `MetalDevice`.
- **`bindings/python/src/lib.rs`** — `NativeRuntimeInfo` pyclass
  grew three new `#[pyo3(get)]` fields: `metal_available: bool`,
  `metal4_available: bool`, `gpu_family: Option<String>`. The
  struct is now populated by `build_native_runtime_info()` which
  has two cfg-gated arms:
  - `cfg(all(target_os = "macos", feature = "metal"))` calls
    `probe_capabilities()` and derives
    `metal_available = caps.apple7` (the Stage 1 baseline), while
    `gpu_family` is populated unconditionally so users on
    sub-baseline hardware can see *why* `metal_available` is false.
  - The fallback arm (non-macOS, or `metal` feature off) returns
    `metal_available=false, metal4_available=false, gpu_family=None`.
- **`bindings/python/tests/test_native_runtime_integration.py`** —
  extended `test_runtime_import_exposes_native_runtime_info` with
  platform-agnostic shape checks + a coherence invariant
  (`metal_available=False` implies `metal4_available=False`). No
  hardware-specific assertions so the suite stays stable across
  Apple Silicon, Intel Mac, and Linux CI.

### Verification

- `cargo check --workspace` clean.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo fmt --all --check` clean.
- `maturin develop --release` clean.
- `pytest bindings/python/tests/ -q` — **332 passed, 16 subtests
  passed**. No regressions.
- **Smoke (Apple M4):** `native_runtime_info()` returns
  `metal_available=True, metal4_available=False, gpu_family="Apple
  M4"`. Confirms the probe reads the expected values on this box:
  Apple7 baseline is met (M4 advertises it), Metal 4 family flag
  is not yet set under this macOS version (macOS 26 Tahoe will
  flip it), and the marketing name round-trips intact through the
  Objective-C bridge.

### Design calls made this session

- **Queue-free `probe_capabilities()`, not a reuse of
  `MetalDevice::probe()`.** The existing `probe()` opens a command
  queue so the returned `MetalDevice` is immediately usable for
  dispatch. For `native_runtime_info()` — a pure introspection
  call that may run on every `import alloygbm` — we don't want to
  hold a queue resident. The light probe is ~40 lines and reuses
  the same capability-read selectors, so there's no drift risk;
  full probe stays the one-shot path for `MetalBackend::new()`.
- **`metal_available` = `caps.apple7`, not `caps.is_some()`.** The
  Stage 1 kernels *require* `MTLGPUFamilyApple7`. Reporting
  `metal_available=True` on a sub-baseline GPU would mislead users
  into requesting `device="metal"` and catching an error. Exposing
  `gpu_family` unconditionally gives them a cheap way to see why.
- **No `ALLOYGBM_METAL_DISABLE` influence on
  `native_runtime_info()`.** The env var is a *PyO3 resolve-time*
  test hook — it exists to exercise the warn-and-fallback code
  path. `native_runtime_info()` should report hardware facts. Mixing
  the two would make the escape hatch surprising ("why does
  capability detection respect a test flag?"). Kept them decoupled.
- **Fields are `#[pyo3(get)]`, not dict entries.** A pyclass with
  attribute-style access matches the pre-S1.10 shape (`info.name`,
  `info.version`) so existing user code stays a drop-in. A dict
  return would've broken backwards compatibility.
- **Python test is shape-only, not capability-asserting.** The
  suite runs on macOS + Linux CI; hardware facts differ. Asserting
  `isinstance(info.metal_available, bool)` + the
  `not-available => not-metal4` coherence invariant gives us
  regression coverage without flakiness.

### Handoff notes for S1.11/S1.12

- **Availability gate shape.** S1.12 tests can write:
  ```python
  import pytest
  from alloygbm import native_runtime_info
  pytestmark = pytest.mark.skipif(
      not native_runtime_info().metal_available,
      reason="Metal backend unavailable on this runner",
  )
  ```
  That plus `sys.platform == "darwin"` handles every combination
  of hardware / build / OS without per-test plumbing.
- **Warn-and-fallback is now testable on any runner.**
  `ALLOYGBM_METAL_DISABLE=1` forces a Metal init failure so the
  fallback path is exercisable from Python tests even on a real
  Apple Silicon CI runner (where `device="metal"` would otherwise
  succeed silently). Expected warning substring is
  `"falling back to CPU. Set device='cpu' to silence"`.
- **Rust-side histogram correctness tests (S1.11) already live in
  `crates/backend_metal/src/kernels/histogram.rs`
  (`histogram_matches_cpu_small_fixture`,
  `histogram_feature_subset_matches_cpu`) and in
  `crates/backend_metal/src/pipelines.rs`
  (`pipeline_cache_returns_identical_arc_on_second_call`). The S1.10
  `probe_capabilities()` has no behavior worth testing that isn't
  already covered by `MetalDevice::probe()` — skipping a redundant
  unit test keeps signal up.

---

## 2026-04-19 — S1.9 Warn-and-fallback + resolved-device in artifact metadata

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`crates/core/src/lib.rs`** — `Device` enum grew a `Metal` variant
  with a derived `#[default] Cpu` (clippy's `derivable_impls` flagged
  the hand-rolled `impl Default`, so it's now a `#[derive(Default)]`
  with `#[default]` on the variant). `as_metadata_label` /
  `parse_metadata_label` extended to serialize/parse `"metal"`. The
  hand-rolled positional JSON parser in `ModelMetadata` stays
  back-compat because `trained_device` is already on the tail of the
  field list and accepts `"cpu"` without change.
- **`crates/engine/src/lib.rs`** — `TrainedModel.trained_device:
  Device` and `MultiClassTrainedModel.trained_device: Device` fields
  added (default `Device::Cpu`). Every struct-literal construction
  site (16 total, incl. a slew of test fixtures) now initialises
  `trained_device`. `to_artifact_bytes` on both types now reads
  `self.trained_device` instead of the old hardcoded `Device::Cpu`;
  `from_artifact_bytes_with_mode` / `from_artifact_bytes` now restore
  the field from the parsed metadata so round-trips preserve the
  recorded device.
- **`crates/shap/src/lib.rs`, `crates/predictor/src/lib.rs`** —
  `TrainedModel { ... }` test-fixture literals updated with
  `trained_device: Device::Cpu`. A Python-driven bulk-patch script
  falsely patched two shap fixtures (fixture_model,
  fixture_model_with_unused_feature) by inserting the field *outside*
  the struct literal; those were corrected by hand.
- **`bindings/python/src/runtime_backend.rs`** — added
  `RuntimeBackend::device(&self) -> Device` companion to `name()`.
  New `pub fn resolve_runtime_backend_with_fallback(py, device,
  warn_source)` is the PyO3-side entry: on `device="metal"` with a
  Metal init failure it calls `PyErr::warn(py,
  &py.get_type::<PyRuntimeWarning>(), &msg, 1)` and returns the CPU
  backend. The pure (non-warning) `resolve_runtime_backend` is
  retained as a unit-test helper with `#[allow(dead_code)]` +
  explanatory comment. The Metal-specific `build_metal_backend` now
  honours `ALLOYGBM_METAL_DISABLE=1` as a deterministic failure
  injection (useful for S1.12 tests on Metal-capable CI).
- **`bindings/python/src/lib.rs`** —
  `train_regression_artifact_with_summary_dense_impl` now takes `py:
  Python<'_>` as its first arg and resolves the backend via
  `resolve_runtime_backend_with_fallback` *before* any engine work,
  wrapping the String error into `EngineError::InvalidConfig` so
  unknown devices still surface as `PyValueError`. Sets
  `model.trained_device = resolved_device` (single-output path) and
  `mc_model.trained_device = resolved_device` (multiclass path)
  immediately before `to_artifact_bytes`. All 5 pyfunctions that call
  `_impl` got a `py: Python<'_>` first argument; the in-module test
  helper wraps its call in `Python::with_gil(|py| ...)`.

### Verification

- `cargo check --workspace` clean.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo fmt --all --check` clean.
- `cargo test --workspace --exclude alloygbm-python` — **all Rust
  tests green**. `alloygbm-python`'s unit tests still fail to
  *link* (pre-existing — they need Python symbols at link time; same
  behaviour as pre-S1.9).
- `maturin develop --release` — clean release build.
- `pytest bindings/python/tests/ -q` — **332 passed, 16 subtests
  passed**. No regressions.
- **End-to-end smokes (worktree-installed wheel):**
  - `GBMRegressor(device="metal").fit(...)` → artifact JSON contains
    `"trained_device":"metal"`.
  - `ALLOYGBM_METAL_DISABLE=1 GBMRegressor(device="metal").fit(...)`
    → single `RuntimeWarning` emitted with the expected text
    (`train: device='metal' requested but Metal backend is
    unavailable (... test escape hatch); falling back to CPU. Set
    device='cpu' to silence this warning.`) and artifact records
    `"trained_device":"cpu"` (the actual backend that ran).
  - `pickle.dumps` / `pickle.loads` on a Metal-trained regressor
    preserves `device="metal"` on the rehydrated object and the
    artifact's recorded `trained_device`.

### Design calls made this session

- **Enum `Default` derived, not hand-rolled.** Clippy's
  `derivable_impls` fires on `impl Default for Device { fn
  default() -> Self { Self::Cpu } }`; replaced with
  `#[derive(Default)]` on the enum + `#[default]` on the `Cpu`
  variant. No behaviour change.
- **Two resolve functions kept.** The pure `resolve_runtime_backend`
  returns a `Result<_, String>` with no Python dependency — it's
  what the unit tests use. `resolve_runtime_backend_with_fallback`
  is the Python-aware variant that emits the warning. Kept both so
  tests don't have to acquire a GIL, and so the pure variant is
  reusable if we ever add a CLI entry point.
- **`ALLOYGBM_METAL_DISABLE=1` escape hatch.** The S1.12 Python
  tests need a way to exercise the fallback path on Metal-capable
  CI, otherwise `device="metal"` just silently succeeds and the
  warn-path goes untested. An env-var gate inside
  `build_metal_backend` gives us deterministic failure injection
  with zero production surface area (the check is `O(1)` per
  resolve). Message is intentionally unique (`"test escape
  hatch"`) so tests can assert against it.
- **Return type of `_impl` kept as `Result<_, EngineError>`.**
  Briefly experimented with converting to `PyResult` so we could
  use `PyErr::warn` directly without the String hop, but every `?`
  inside the function currently bubbles `EngineError`, and
  rewriting all of them to `.map_err(engine_error_to_pyerr)`
  would've doubled the line count of the fix. The `String` ->
  `EngineError::InvalidConfig` -> `PyValueError` chain works and
  keeps the blast radius tight.
- **`Python::with_gil` only in the test helper.** The production
  pyfunctions already hold the GIL (they're called by the Python
  interpreter), so threading `py` is a plain parameter pass. The
  `#[cfg(test)]` helper runs outside a PyO3 entrypoint, so it
  acquires the GIL via `Python::with_gil` locally. Both paths end
  up calling the same `_impl`.

### Handoff notes for S1.10

- **Capability probe already exists.** `MetalBackend::capabilities()`
  and the underlying `MetalDevice::probe()` (crates/backend_metal)
  return a `MetalCapabilities` struct with `gpu_family`, `metal4`,
  etc. S1.10 is mostly Python plumbing: a new PyO3 pyfunction that
  returns a `dict` with `metal_available: bool`, `metal4_available:
  bool`, `gpu_family: Option<String>`, plus a two-line extension of
  `native_runtime_info()` in
  `bindings/python/alloygbm/__init__.py`. No engine or artifact
  changes needed.
- **Fallback path now reusable.** `resolve_runtime_backend_with_fallback`
  is the one chokepoint where Metal init is attempted in the
  PyO3-facing code; S1.10's probe can short-circuit by calling the
  same builder (if we end up needing to surface Metal availability
  to users, the fallback path is already there).
- **Artifact round-trip is stable.** `trained_device` is
  bit-identical on re-save, so S1.13's bit-exactness golden test
  can rely on the field persisting through CPU↔Metal training runs.

---

## 2026-04-19 — S1.8 Python `device="cpu"|"metal"|"auto"` on all three estimators

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`bindings/python/alloygbm/regressor.py`** — added module-level
  `_VALID_DEVICES = {"cpu","metal","auto"}`; `device: str = "cpu"`
  keyword at the end of `__init__`; validation block in `__init__`
  mirroring the `training_policy` / `tree_growth` pattern; attribute
  assignment `self.device = str(device)`; appended `device=` to
  `__repr__`; added `"device": self.device` to `get_params()`;
  extended the `set_params` `allowed` set and added a mirrored
  validation block; threaded `device=self.device` through all 5
  native call sites (bytes-path, dense-with-summary, rows-with-summary,
  dense legacy bridge, rows legacy bridge). Pickle state
  (`__getstate__`/`__setstate__`) and `save_model`/`load_model` need
  no changes — the former uses `self.__dict__.copy()` and the latter
  round-trips through `get_params()` + `known`-filtered rehydration,
  both of which pick up `device` automatically.
- **`bindings/python/alloygbm/classifier.py`** — no `__init__` change
  needed (inherits via `**kwargs`), but the custom `__repr__` at
  lines 294-327 does *not* call `super().__repr__()` and explicitly
  enumerates fields, so `device='…'` was appended there. Pickle
  hooks are pure `super()` delegation → auto-covered.
- **`bindings/python/alloygbm/ranker.py`** — same pattern: `__init__`
  forwards via `super().__init__(**kwargs)` so `device` flows
  through the `__signature__` override too. Custom `__repr__` at
  lines 222-257 got an appended `device='…'`. `get_params` /
  `set_params` delegate to super, so no changes needed there.
- **`bindings/python/tests/test_regressor_contract.py`** — one
  contract test (`test_fit_and_predict_use_native_bridges`) asserts
  the exact kwargs recorded by the fake native bridge; appended
  `"device": "cpu"` to match the new call shape.

### Verification

- `cargo check -p alloygbm-python` clean.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo fmt --all --check` clean.
- `maturin develop --release` rebuilds cleanly (still ~0s — no Rust
  changes this session, just Python).
- **Python test suite: 332 passed, 16 subtests passed** (was 332 pre-
  S1.8; the patched contract test keeps the count steady).
- Estimator smoke: all three estimators (`GBMRegressor`,
  `GBMClassifier`, `GBMRanker`) with `device="cpu" | "metal" | "auto"`
  fit + predict successfully on a 3-4 row fixture. `set_params`
  round-trip, `pickle.{dumps,loads}` round-trip, and
  `save_model`/`load_model` round-trip all preserve `device`.
  Invalid `device="tpu"` raises `ValueError` in both `__init__` and
  `set_params`. Metal device end-to-end trains and predicts
  identically (`[1.9999998807907104]` vs the CPU path on the
  smoke fixture).

### Design calls

- **`device` is the last kwarg** in every estimator's `__init__`
  (after `max_cat_threshold`). Back-compat for positional-kwarg
  callers and for `load_model` consumers: the new field is filtered
  through `known = set(_probe.get_params().keys())` on load, so
  older artifacts (missing `"device"`) just rehydrate with the
  default. Newer artifacts trained with `device="metal"` loaded on
  the same build retain the original device in `get_params()`, but
  inference never consults it (inference goes through
  `NativePredictorHandle`, which is device-agnostic).
- **`_params_order` from the plan is a red herring** — grep shows no
  such symbol anywhere in `bindings/python/alloygbm/`. Both
  `get_params` and `set_params` are dict-based, so the plan's list
  of touch points collapses to: `__init__` sig + validation +
  `__repr__` + `get_params` dict + `set_params` allowed-set +
  `set_params` validation + native call sites.
- **Classifier/Ranker validation lives in the Regressor base class.**
  Both subclasses forward to `super().__init__(**kwargs)` with no
  further filtering, so the same `_VALID_DEVICES` check runs for
  every estimator. No duplication of the validation block.
- **Ranker's `__signature__` override** (lines 68-82 of ranker.py)
  introspects `GBMRegressor.__init__` at class-body time and
  recomposes the signature with `ranking_objective` prepended. This
  inherits the new `device` parameter automatically — verified by
  `inspect.signature(GBMRanker.__init__)` showing `device='cpu'` at
  the tail.

### Next session handoff

- **S1.9 — warn-and-fallback on Metal init failure + resolved
  device in artifact metadata.** Two pieces: (a) at each PyO3
  `train_*_impl` entry, wrap `resolve_runtime_backend(device)` such
  that `device ∈ {"metal","auto"}` falling back to CPU emits a
  `PyRuntimeWarning` via `PyErr::warn_bound(py, …)` and returns
  `RuntimeBackend::Cpu(CpuBackend)`; the `"cpu"` case never warns.
  (b) stash `backend.name()` (already captured as
  `_backend_name: &'static str` at each dispatch site) into
  `ModelMetadata` as a new append-only field. The hand-rolled
  positional JSON parser in `crates/core/src/lib.rs` means the
  field *must* go at the end of `ModelMetadata` serialization with
  a default for back-compat — same pattern as
  `uses_continuous_binning` and friends.
- **Behavioural gotcha for S1.9:** `resolve_runtime_backend("auto")`
  currently maps to CPU unconditionally. If S1.9 upgrades `"auto"`
  to "try Metal first, fall back to CPU", the warn-and-fallback
  path needs to treat `"auto"` and `"metal"` asymmetrically:
  `"auto" → Metal-failure` should NOT warn (it's the heuristic
  doing its job), whereas `"metal" → Metal-failure` SHOULD warn
  (user explicitly asked for Metal). Easiest: keep `"auto" = CPU`
  for S1.9 too, and defer the real heuristic to Stage 2.

---

## 2026-04-19 — S1.7 `RuntimeBackend` enum + `device: &str` PyO3 plumbing

**Branch:** `claude/charming-carson-d08c9a` (worktree)

### What moved

- **`bindings/python/src/runtime_backend.rs`** (new) — a single
  `RuntimeBackend::{Cpu(CpuBackend), Metal(MetalBackend)}` enum that
  itself implements `BackendOps` by forwarding all 6 methods
  (`build_histograms`, `best_split`, `best_split_with_options`,
  `apply_split`, `apply_split_with_stats`, `reduce_sums`) to the
  inner variant via a match on the discriminant. This preserves
  `Trainer::fit_iterations<B: BackendOps, O: ObjectiveOps>` static
  monomorphization — one instantiation per (objective, backend
  enum), branch cost = one discriminant check inside each forwarded
  method (per D-004).
- `resolve_runtime_backend(device: &str) -> Result<RuntimeBackend,
  String>` — validates `{"cpu","metal","auto"}` case-insensitively
  and trimmed; `"auto"` aliases to `"cpu"` in S1.7 per plan (shape-
  based heuristic deferred to Stage 2+). Returns plain `String` so
  callers can wrap into either `EngineError::InvalidConfig` (Rust
  level) or `PyValueError` (PyO3 level) at their own abstraction
  layer.
- Cfg-gated `Metal(MetalBackend)` variant + `build_metal_backend()` —
  only compiled under `cfg(all(target_os = "macos", feature =
  "metal"))`; on other targets `device="metal"` returns a clear
  error string. Metal init failures also surface as `Err`; warn-and-
  fallback is intentionally left to S1.9.
- Manual `impl Debug for RuntimeBackend` — prints just the variant
  name (`RuntimeBackend("cpu")` / `RuntimeBackend("metal")`) so
  `unwrap_err()`-style test assertions compile without forcing the
  backend crates to derive Debug on their Metal protocol objects.
- **`bindings/python/src/lib.rs`** — added `mod runtime_backend;`
  and `use runtime_backend::resolve_runtime_backend;`; removed the
  now-unused top-level `use alloygbm_backend_cpu::CpuBackend;`
  (tests module already re-imports it). Added `device: &str`
  parameter to `train_regression_artifact_with_summary_dense_impl`
  and replaced the sole `let backend = CpuBackend;` line with
  `let backend = resolve_runtime_backend(device).map_err(...)?;`.
  Stashed `backend.name()` as `_backend_name: &'static str` at the
  dispatch site so S1.9 has a one-line hook for artifact metadata.
- **5 public `train_regression_artifact*` pyfunctions** grew a
  `device="cpu"` kwarg (at the end of each `#[pyfunction(signature
  = (...))]`; Python's positional→keyword migration makes adding at
  the end strictly back-compat). All five pass `device` through to
  the shared `_impl`: `train_regression_artifact`,
  `train_regression_artifact_dense`,
  `train_regression_artifact_with_summary`,
  `train_regression_artifact_dense_with_summary`,
  `train_regression_artifact_dense_with_summary_bytes`. (The
  codebase has exactly one `_impl` funnel that routes regression /
  binary / multiclass / ranking via the `objective` string — so
  there is no separate `train_binary_*` / `train_multiclass_*` /
  `train_ranking_*` surface to update.)
- **Tests module helper** `train_regression_artifact_impl` at line
  4043 passes `"cpu"` as the new last arg to `_impl`.

### Verification

- `cargo check -p alloygbm-python` → clean.
- `cargo clippy --workspace --all-targets -- -D warnings` → clean.
- `cargo fmt --all --check` → clean (one auto-format tidy applied
  to the let-chain at the dispatch site and to the non-macOS
  `build_metal_backend` error return).
- `cargo test --workspace --exclude alloygbm-python` → 183 tests
  pass (the `--exclude alloygbm-python` is the known PyO3 linker
  workaround — `_Py_DecRef` et al. are unresolved when building the
  Python crate as a cargo test binary; not introduced by S1.7).
  `runtime_backend`'s own 5 unit tests pass as part of the Python
  crate's lib target when building via maturin.
- `maturin develop --release` → built and installed cleanly, 17s.
- `.venv/bin/python -m pytest bindings/python/tests/ -q` → **332
  passed**, 1 warning (unrelated numpy `invalid value in divide`
  in an existing custom-metric test), 16 subtests passed in 16s.

### End-to-end smoke

On the local Apple M4 with `metal` feature active, a 4-row seeded
regression fit with `device="cpu"` vs `device="metal"` produced
**bit-exact equal `artifact_bytes`** (370 bytes each). Unknown
devices (`device="tpu"`) surface as `PyValueError` with the
expected `"device must be one of 'cpu', 'metal', or 'auto'"`
message. This is not the full S1.13 bit-exactness gate (that is
50k×100); it's just a sanity check that the plumbing threads
through correctly and the discriminant-forwarding BackendOps impl
hits the Metal histogram path (the code was already exercised in
the S1.4 correctness tests — we just hadn't driven it through the
Python entry point before).

### Design calls locked in

- Everything in D-004 is upheld; no architectural deviations.
- The `device` kwarg appears **last** in each pyfunction signature
  — PyO3 supports keyword-only args and older Python callers
  already using positional args in the test suite continue to
  work. Artifact metadata (for S1.9) will be appended at the end of
  the positional JSON too, for the same back-compat reason (the
  hand-rolled positional `ModelMetadata` parser is brittle).

### Next session

- **S1.8** — surface `device` on `GBMRegressor`, `GBMClassifier`,
  `GBMRanker`. Validate at the Python layer against
  `{"cpu","metal","auto"}` so errors surface as `ValueError` on
  construction (not only at `fit()` time). Update `__init__`,
  `get_params`, `set_params`, `__repr__`, `_params_order`, and
  `__getstate__`/`__setstate__` pickle round-trip.

---

## 2026-04-19 — S1.5 Pipeline compilation + `MTLBinaryArchive` cache

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- `src/pipelines.rs` — rewritten. The one-shot
  `build_histogram_pipelines(device, bin_count, use_u16_bins)`
  factory is replaced by a long-lived `HistogramPipelineCache` owned
  by `MetalBackend`:
  - Compiles the MSL library exactly once per process at
    `HistogramPipelineCache::new`, holding the `Retained<MTLLibrary>`
    for the lifetime of the backend.
  - Attempts to open (or create fresh) a per-device
    `MTLBinaryArchive` at
    `~/Library/Caches/com.alloygbm/pipelines-<family>-<device>.metalarchive`.
    Family is `"metal4"` when Metal 4 is supported, else `"apple7"`;
    device is a lowercase-ascii-slug of `MTLDevice::name`
    (e.g. `"apple-m4"`, `"apple-m2-pro"`). Opening failure is
    logged and degrades gracefully to no-persistence mode.
  - `get_or_build(bin_count, use_u16)` returns an
    `Arc<HistogramPipelines>` from a `Mutex<HashMap<(u32, bool),
    Arc<…>>>`. Fast path is a single `Mutex::lock` + clone. Slow
    path specialises both MSL functions with
    `MTLFunctionConstantValues` (BIN_COUNT/USE_U16_BINS, indices
    0/1), builds `MTLComputePipelineDescriptor`s with
    `setBinaryArchives([archive])` so the driver can source
    precompiled pipelines from disk, and calls
    `newComputePipelineStateWithDescriptor:options:reflection:error:`.
    Freshly-compiled functions are added back to the archive via
    `addComputePipelineFunctionsWithDescriptor:error:` and a
    `dirty: Mutex<bool>` flag is set.
  - `Drop` flushes the archive exactly once per session, writing to
    `<path>.metalarchive.tmp` and then `std::fs::rename` into place
    so a mid-write crash preserves the previous archive — per
    Apple's corruption-resiliency guidance in the `MTLBinaryArchive`
    docs. Skipped if `dirty == false`.
  - `unsafe impl Send`/`Sync` added with a documented SAFETY note:
    Metal protocol objects (device, library, pipeline state) are
    thread-safe per Apple docs; archive mutation points are guarded
    by the cache's own mutexes.
- `src/lib.rs` — `MetalBackend` grows a
  `pipeline_cache: Arc<HistogramPipelineCache>` field. The cache is
  constructed in `MetalBackend::new()` after the device probe and
  passed by reference into each `dispatch_histograms` call.
- `src/kernels/histogram.rs` — `dispatch_histograms` takes a
  `&HistogramPipelineCache` and calls `get_or_build(bin_count,
  use_u16)` instead of the old per-dispatch
  `build_histogram_pipelines`. The rest of the dispatch body is
  byte-identical.
- New tests:
  - `pipelines::tests::slugify_handles_common_device_names` +
    `archive_filename_encodes_family_and_device` — pure-Rust tests
    for the cache-path construction; run on every target.
  - `tests::pipeline_cache_returns_identical_arc_on_second_call` —
    macOS-only; calls `get_or_build(8, false)` twice, asserts
    `Arc::ptr_eq`, then `get_or_build(8, true)` and asserts
    non-equality. Guards against a future refactor accidentally
    reintroducing per-dispatch compilation.
- `docs/metal-backend/DECISIONS.md` — logged **D-009** (archive
  serialization is drop-time only via atomic rename) and **D-010**
  (`unsafe impl Send + Sync` with documented invariants).

**Commits shipped:** pending — to be committed after this entry.

**Verification:**
- `cargo check --workspace`: green.
- `cargo test -p alloygbm-backend-metal`: **7 passed** (probe +
  shader-compile + 2 bit-exact correctness + 2 cache-path unit
  tests + 1 cache-hit test).
- `cargo test --workspace --exclude alloygbm-python`: 23+7+10+32+69+19+23
  = **183 passed**, 0 failed (+3 over S1.4 baseline of 180: two
  pipelines-module unit tests + the cache-hit test).
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- **On-disk archive verification:** after running the Metal tests,
  `ls ~/Library/Caches/com.alloygbm/` showed a ~60KB
  `pipelines-apple7-apple-m4.metalarchive` file — confirming the
  scatter + reduce pipelines were successfully added and serialized.

**Debug notes:**
- First clippy hit: `clippy::arc_with_non_send_sync` on
  `Arc::new(HistogramPipelineCache::new(…)?)` — objc2-metal doesn't
  auto-derive Send/Sync for Metal protocol objects. Added explicit
  `unsafe impl` with SAFETY comment pointing to Apple's
  thread-safety docs for `MTLDevice`/`MTLLibrary`/
  `MTLComputePipelineState` and noting our internal mutex-guarded
  archive mutation. See D-010.
- rustfmt collapses two multi-line let-chains (the `if added_any &&
  let Ok(mut dirty) = self.dirty.lock()`) onto a single line — fine,
  applied.
- Archive opening uses a two-shot approach: try once with
  `descriptor.url = existing path`; on error (corrupt file, schema
  bump across OS upgrade) delete the file and retry with an empty
  descriptor. Only if *that* fails do we drop to no-persistence
  mode. Keeps us robust against the exact scenario Apple warns
  about ("software updates of the OS or device drivers may cause
  the archive to become outdated").
- `MetalBackend.pipeline_cache` is `Arc<…>` rather than direct
  ownership so future code (Stage 2 best-split kernel, Stage 3 ICB
  chaining) that wants to share the library/archive across multiple
  kernel dispatches can `Arc::clone` instead of re-opening.

**Next session should:**
- Start **S1.7**: add `RuntimeBackend` enum in
  `bindings/python/src/lib.rs`, thread `device: &str` through every
  `train_*` pyfunction, keep static dispatch via monomorphization
  on `RuntimeBackend`.
- Then **S1.8** on the Python side (`GBMRegressor` / `GBMClassifier`
  / `GBMRanker` `device` parameter — follow the existing
  `_params_order` + `__repr__` + pickle state conventions).

---

## 2026-04-19 — S1.4 Rust-side histogram dispatch orchestration

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- `src/pipelines.rs` — new module. `build_histogram_pipelines(device,
  bin_count, use_u16_bins)` compiles the MSL library, constructs an
  `MTLFunctionConstantValues` with `BIN_COUNT` (uint, index 0) and
  `USE_U16_BINS` (bool, index 1), specializes both entry points via
  `newFunctionWithName:constantValues:error:`, and builds the
  `MTLComputePipelineState` pair. Caching is S1.5; here we rebuild
  fresh every dispatch for correctness focus.
- `src/kernels/histogram.rs` — new `dispatch_histograms` function.
  Wraps `BinnedMatrix::bins_col_adaptive` (u8 or u16) as a single
  shared `MTLBuffer`; packs `&[GradientPair]` into an `[f32; 2]` layout
  buffer (`GradientPair` is not `#[repr(C)]`); wraps `node.row_indices`
  as a u32 buffer. Per tile: allocates a fresh scratch buffer sized
  `n_chunks × tile_n_features × bin_count × sizeof(float2)`; binds the
  binned matrix with `setBuffer:offset:atIndex:` at
  `start_feature * row_count * sizeof_bin`; binds a 1-byte dummy into
  the unused `binned_u8`/`binned_u16` slot (the kernel's function-
  constant branch dead-code-eliminates the access); encodes the
  scatter pass (`(tile_n_features, n_chunks, 1)` threadgroups, 32
  threads), then the reduce pass (`(tile_n_features, ceil(B/32), 1)`
  threadgroups). Commits once, waits once. Reads back the final
  `float2*` output buffer, reconstructs counts on CPU
  (see D-008), and assembles `HistogramBundle`. `rows_per_chunk`
  default = 8192.
- `src/lib.rs` — `MetalBackend` grows a `cpu: CpuBackend` field.
  `impl BackendOps for MetalBackend` routes `build_histograms` to
  Metal and delegates the other five methods (`best_split`,
  `best_split_with_options`, `apply_split`, `apply_split_with_stats`,
  `reduce_sums`) to the embedded `CpuBackend`. This folds the S1.6
  "non-histogram ops fall back to CPU" promise into S1.4 — clean
  because the delegation is mechanical.
- Two new correctness tests, both bit-exact vs `CpuBackend` via
  `to_bits()` comparison:
  - `histogram_matches_cpu_small_fixture`: 500 rows × 6 features ×
    8 bins, deterministic bin/gradient pattern, full-node slice, single
    tile covering all features. Verifies `grad_sum`, `hess_sum`, and
    `count` per bin match exactly. Gradients chosen from
    `{1.0, -2.0, 4.0}` × `{1.0, 2.0}` so float addition is associative
    in the exact-integer range — any accumulation order lands on the
    same bit pattern.
  - `histogram_feature_subset_matches_cpu`: 200 rows × 6 features × 4
    bins, tile = features 2..5 only. Verifies the per-tile
    binned-buffer offset arithmetic and output-region offset
    arithmetic is correct.
- `docs/metal-backend/DECISIONS.md` — logged **D-008** (CPU-side count
  accumulation for S1.4; revisited in Stage 2 if profiling hotspot).

**Commits shipped:** see git log

**Verification:**
- `cargo check --workspace`: green.
- `cargo test -p alloygbm-backend-metal`: 4 passed (probe + compile +
  two correctness gates).
- `cargo test --workspace --exclude alloygbm-python`: **180 passed**, 0
  failed (+3 over the S1.3 baseline of 177 — the two new correctness
  tests + 1 other — let me double check… yes: 23 + 4 + 10 + 32 + 69 +
  19 + 23 = 180; no regressions).
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.

**Debug notes:**
- `GradientPair` is not `#[repr(C)]`, so `&[GradientPair]` can't safely
  be reinterpreted as `&[f32]` / `&[[f32; 2]]`. The dispatch copies
  into an owned `Vec<[f32; 2]>` before buffer creation. This is one
  `O(n_rows)` copy per node — the only unavoidable extra work S1.4
  introduces. Could be eliminated later by pushing `#[repr(C)]` into
  `core`, but that touches a public type and has no upside for S1.
- MSL's `USE_U16_BINS` function-constant branch compiles away the
  unused binned-pointer access, but the kernel signature still carries
  both `binned_u8` and `binned_u16` arguments — Metal refuses to
  dispatch with a null buffer at a referenced slot. We bind a 1-byte
  `MTLResourceOptions::StorageModeShared` dummy at whichever slot the
  kernel ignores. Zero correctness impact.
- clippy flagged `for bin_idx in 0..bin_count { ... counts[bin_idx] }`
  as `needless_range_loop`; rewrote to
  `for (bin_idx, &count) in counts.iter().enumerate() { ... }`.

**Blockers:** none.

**Next session should:** start **S1.5** (pipeline compilation + disk
cache). Add `MTLBinaryArchive` at
`~/Library/Caches/com.alloygbm/pipelines-<gpu-family>-<macos>.metalarchive`
so the first run pays the MSL compile cost and every subsequent run is
cache-hit. Also add an in-process cache keyed by
`(bin_count, use_u16_bins)` — right now S1.4 rebuilds pipelines on
every dispatch, which is wasteful. Keep the pipeline archive's
`addComputePipelineFunctionsWithDescriptor:error:` call behind a Metal
4 guard; the `MTLBinaryArchive` itself is Metal 3.

---

## 2026-04-19 — S1.3 MSL histogram kernel source + compile test

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- `src/shaders/histogram.metal` — two-pass MSL compute kernel:
  - `histogram_build_scatter`: per-threadgroup scatter. One threadgroup
    per (feature, row-chunk). 32 threads (one SIMD group). Single shared
    `threadgroup float2 local_hist[MAX_BIN_COUNT]` with per-bin
    **single-writer discipline**: lane `k` is the exclusive writer for
    bins where `bin % 32 == k`. 32 lanes read rows in parallel, then
    serialise an inner `for src_lane in 0..32` loop using `simd_shuffle`
    to hand each lane's `(bin, grad, hess)` across the SIMD group.
    Every write destination is deterministic by construction; no float
    atomics needed. Writes the threadgroup histogram to a device-memory
    scratch buffer indexed by `(chunk, feature, bin)`.
  - `histogram_reduce`: cross-chunk ascending reduce. One thread per
    `(feature, bin)`, walks chunks `0..n_chunks` in order, writes the
    final `float2`. Deterministic by single-threaded accumulation.
  - Function constants: `BIN_COUNT` (0), `USE_U16_BINS` (1). Fallback
    defaults via `is_function_constant_defined` let `newLibraryWithSource`
    compile cleanly ahead of pipeline creation. Threadgroup-memory array
    size is bounded by `MAX_BIN_COUNT = 4096`.
- `src/kernels/{mod.rs,histogram.rs}` — Rust holders. `HISTOGRAM_SHADER_SOURCE`
  embeds the `.metal` file via `include_str!`; `KERNEL_NAME_SCATTER` and
  `KERNEL_NAME_REDUCE` identify the two entry points.
- `src/lib.rs` — exposes `kernels` module; adds `tests::histogram_shader_compiles`
  which feeds the source to `MTLDevice::newLibraryWithSource_options_error`
  on macOS and panics on any MSL diagnostic.

**Commits shipped:** see git log

**Verification:**
- `cargo test -p alloygbm-backend-metal`: 2 passed (probe + shader compile).
- `cargo clippy -p alloygbm-backend-metal --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.

**Debug notes:**
- First compile pass tripped on "expecting input declarations with
  either all scalar types or all vector types" — MSL requires all
  position-attribute inputs to share dimensionality. Fixed by using
  `uint3` for both `thread_position_in_threadgroup` and
  `threadgroup_position_in_grid`, then projecting to the scalars /
  pair we actually use.
- `newLibraryWithSource_options_error` is safe in `objc2-metal` 0.3 —
  dropped the `unsafe` block once clippy flagged it as unused.

**Blockers:** none.

**Next session should:** start **S1.4** (Rust-side histogram dispatch
orchestration). Wrap `BinnedMatrix` + `gradients` + `row_indices` as
shared `MTLBuffer`s, allocate scratch + output, encode the two passes
into one command buffer, read back into `HistogramBundle`, wire
`impl BackendOps for MetalBackend::build_histograms`, delegate the
remaining 5 `BackendOps` methods to an embedded `CpuBackend`. First
correctness gate: hand-computed fixture (<1000 rows) vs Metal output.
Pipeline compilation + caching stays scoped to S1.5.

---

## 2026-04-18 — S1.2 device + capability probe

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Added `objc2 = "0.6"`, `objc2-foundation = "0.3"`, `objc2-metal = "0.3"`
  under `[target.'cfg(target_os = "macos")'.dependencies]`.
- Dropped workspace-inherited lints on `backend_metal`; declared local
  `[lints.rust] unsafe_code = "deny"` so FFI sites opt in per-site.
  Recorded this deviation as **D-007** in `DECISIONS.md`.
- `src/device.rs`: `MetalCapabilities { apple7, metal4, device_name }`
  and `MetalDevice { device, queue, capabilities }`; `MetalDevice::probe()`
  calls `MTLCreateSystemDefaultDevice`, opens a command queue,
  reads `supportsFamily(MTLGPUFamily::Apple7)`, and probes Metal 4 via
  `msg_send![device, supportsFamily: 5002isize]` (raw NSInteger to stay
  forward-compatible with `objc2-metal` 0.3 which may not yet expose the
  Metal 4 variant in its enum).
- `src/lib.rs`: `MetalBackend::new()` wraps `MetalDevice::probe()` and
  rejects devices that don't support Apple7. Stubbed on non-macOS so the
  workspace builds cross-platform.
- Added a smoke test (`tests::probe_default_device`) that calls
  `MetalBackend::new()` on macOS and asserts Apple7 + non-empty device
  name. Passes locally on M-series hardware.

**Commits shipped:** (committed at session end — see git log)

**Verification:**
- `cargo check --workspace`: green.
- `cargo clippy -p alloygbm-backend-metal --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- `cargo test --workspace --exclude alloygbm-python`: 177 passed, 0 failed,
  including the new Metal probe test.

**Notes:**
- `cargo test --workspace` (without excluding `alloygbm-python`) fails at
  link with missing `_Py_DecRef`/`_Py_IncRef` etc. This is pre-existing
  — `alloygbm-python` is a `cdylib` tested via `maturin develop` + pytest,
  not via `cargo test`. Not a regression.

**Blockers:** none.

**Next session should:** start **S1.3** (MSL histogram kernel). Write
`crates/backend_metal/src/shaders/histogram.metal` implementing
privatized-threadgroup histograms + deterministic tree-reduce, embed via
`include_str!` from `src/kernels/histogram.rs`. Keep the Rust module
pure-source-for-now; actual pipeline compilation + dispatch arrive in
S1.4/S1.5.

---

## 2026-04-18 — S1.1 scaffold

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Created `crates/backend_metal` crate: `Cargo.toml` (workspace-inherited
  metadata; deps on `alloygbm-core`, `alloygbm-engine`, `alloygbm-backend-cpu`),
  minimal `build.rs` (no-op; framework linking lands in S1.2),
  `src/lib.rs` with a stub `MetalBackend` unit struct.
- Added `crates/backend_metal` to workspace `members` in root `Cargo.toml`.
- Wired `bindings/python/Cargo.toml`: optional `alloygbm-backend-metal`
  under `[target.'cfg(target_os = "macos")'.dependencies]`, `metal` feature
  default-on via `dep:alloygbm-backend-metal`.
- Verification: `cargo check --workspace` green in 5.79s. `cargo clippy
  -p alloygbm-backend-metal --all-targets -- -D warnings` clean.
  `cargo fmt --all --check` clean.

**Commits shipped:** (committed at session end — see git log for SHA)

**Blockers:** none.

**Next session should:** start **S1.2** (device probe). Add `objc2` +
`objc2-metal` deps, extend `build.rs` with framework linking, create
`src/device.rs` that probes `MTLCreateSystemDefaultDevice` and family
flags (`MTLGPUFamilyApple7`, `MTLGPUFamilyMetal4`), and thread device +
command queue + capability flags onto `MetalBackend`. Keep `MetalBackend`
still not implementing `BackendOps` — that arrives in S1.4.

---

## 2026-04-18 — Planning session

**Branch:** `claude/charming-carson-d08c9a` (worktree)

**What moved:**
- Confirmed MLX was the wrong foundation (NotebookLM MLX Expert: `scatter_add`
  non-deterministic, macOS 14+/Apple-Silicon-only distribution, forces MSL anyway).
- Confirmed raw-Metal design with 3 rounds against NotebookLM Metal 4 Expert
  (sessions `df440836` MLX, `09f9a81e` Metal 4). Validated: no float atomics,
  two-pass deterministic reduce, level-parallel dispatch, `MTLResidencySet`
  pattern, runtime MSL compile + pipeline harvesting cache, ~250k-row
  breakeven, 4-5× decisive win >1M rows × 100 features.
- Wrote and approved the Stage 1 plan
  (see `/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md`).
- User decisions locked: Metal 3 baseline + Metal 4 fast path; full 4-stage
  plan with Stage 1 in scope; cargo feature `metal` default-on for macOS.
- Created this progress-tracking scaffold (`STATUS.md`, `SESSIONS.md`,
  `BUGS.md`, `DECISIONS.md`) and CLAUDE.md anchor.

**Commits shipped:** _(scaffold only — no Rust code yet)_

**Blockers:** none.

**Next session should:** read `STATUS.md`, then start **S1.1** (scaffold
`crates/backend_metal` + workspace wiring + `cargo check --workspace` green)
as a single small commit. Update `STATUS.md` and append here before ending.

---
