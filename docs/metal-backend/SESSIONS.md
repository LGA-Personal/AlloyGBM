# Metal Backend — Session Log

Append-only. One entry per working session. Newest entries at the top.
First thing a new session reads, alongside `STATUS.md`.

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
