# Metal Backend — Current Status

**Last updated:** 2026-04-20 (S1.16 landed — Stage 1 complete)
**Active stage:** Stage 1 — Histogram build on Metal — **CLOSED**
**Active sub-task:** *(none — Stage 2 opens via next `ExitPlanMode`)*

---

## Stage 1 Checklist

Order matches the approved plan in
[/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md).

- [x] **S1.1** Scaffold `crates/backend_metal` (Cargo.toml, build.rs, empty lib, workspace member, feature flag wired)
- [x] **S1.2** Device + capability probe (`device.rs`) — `MTLCreateSystemDefaultDevice`, `MTLGPUFamilyApple7`, `MTLGPUFamilyMetal4` flag
- [x] **S1.3** MSL histogram kernel (`shaders/histogram.metal`) — privatized threadgroup histograms + two-pass deterministic reduce
- [x] **S1.4** Rust-side orchestration (`kernels/histogram.rs` + `pipelines.rs`) — buffer wrapping, encoding, submit, readback; `impl BackendOps for MetalBackend` (histogram on Metal, rest delegated to embedded `CpuBackend`)
- [x] **S1.5** Pipeline compilation + `MTLBinaryArchive` cache at `~/Library/Caches/com.alloygbm/` — `HistogramPipelineCache` with in-process `Mutex<HashMap<(bin_count, use_u16_bins), Arc<HistogramPipelines>>>` + on-disk archive persisted atomically at Drop
- [x] **S1.6** `MetalBackend` delegates non-histogram `BackendOps` methods to embedded `CpuBackend` *(delivered in S1.4)*
- [x] **S1.7** `RuntimeBackend` enum in `bindings/python/src/runtime_backend.rs`; `device: &str` on every `train_*` pyfunction (5 pyfunctions + `_impl` + test helper)
- [x] **S1.8** Python `device="cpu"|"metal"|"auto"` parameter across `GBMRegressor`, `GBMClassifier`, `GBMRanker` — `__init__` + validation + `get_params`/`set_params` + `__repr__` + 5 native call sites on Regressor; Classifier and Ranker `__repr__` extended (pickle/save-load round-trip via existing `__dict__` plumbing)
- [x] **S1.9** Warn-and-fallback on Metal init failure; store resolved device in artifact metadata JSON — `Device::Metal` variant + `TrainedModel::trained_device` / `MultiClassTrainedModel::trained_device` fields round-trip through `to_artifact_bytes`; `resolve_runtime_backend_with_fallback(py, device, "train")` emits a `RuntimeWarning` and falls back to CPU on Metal init failure; `ALLOYGBM_METAL_DISABLE=1` escape hatch added for exercising the fallback on Metal-capable hardware
- [x] **S1.10** Extend `native_runtime_info()` with `metal_available`, `metal4_available`, `gpu_family` — `probe_capabilities()` added to backend_metal for queue-free probing; `NativeRuntimeInfo` pyclass grew three new getters; graceful collapse to `False`/`None` on non-macOS and `--no-default-features` builds
- [ ] **S1.11** Rust unit tests for histogram kernel correctness (<1000 rows, hand-computed reference) *(delivered in S1.4: `histogram_matches_cpu_small_fixture` + `histogram_feature_subset_matches_cpu`; S1.5 adds `pipeline_cache_returns_identical_arc_on_second_call`)*
- [x] **S1.12** `bindings/python/tests/test_metal_backend.py` — macOS + `native_runtime_info().metal_available` gated; 18 cases covering availability probe, regression/classification/ranking bit-exactness vs CPU, NaN handling, single-row, single-feature, bin counts 16/255/1024, warn-and-fallback via `ALLOYGBM_METAL_DISABLE=1` (subprocess-isolated), and device-string validation (`auto` aliasing, unknown values raising `ValueError`)
- [x] **S1.13** Bit-exactness golden test at scale: `MetalGoldenTests` class in `test_metal_backend.py` — seeded (50k rows × 100 features, 255 bins, 20 estimators) fit pair, shared `setUpClass` so Metal fit runs once (~5s); three assertions: prediction bit-exactness over the full training set, prediction bit-exactness over a held-out 5k-row eval set, and `trained_device` correctly recorded in both artifacts. Scope adjusted from plan's original `artifact_bytes` contract to prediction equality — see "Next Up" note on why (metadata length prefix differs legitimately)
- [x] **S1.14** `benchmarks/metal_histogram.py` — standalone throughput harness: argparse grid selector (`--rows`, `--features`, `--full`, `--estimators`, `--bins`); `--memory-budget-gb` default 8 GB skips the 10M × 1000 (~40 GB) corner unless explicitly raised; Metal pipeline warmup fit amortises first-compile cost; markdown table on stdout + optional `--json-out`; reference numbers captured in `docs/metal-backend/BENCHMARKS.md` for Apple M4. Stage 1 whole-fit wall-clock is uniformly slower on Metal across the (10k/100k/1M) × (10/100/1000) grid — expected, and motivates Stage 2 as described in that doc
- [x] **S1.15** `docs/limitations.md` note on breakeven + availability + `BufferCache` — folded together because the limitations note citing the benchmarks would have been misleading without the buffer-cache optimisation landing first. Work:
  - **`crates/backend_metal/src/buffers.rs`** — persistent Metal buffer pool. Keys the binned matrix by `(ptr, len, is_wide)` for zero-copy reuse across the ~63 `build_histograms` calls per tree and across all trees in a fit; keeps reusable allocations for gradients + row-indices with fresh per-call memcpy. Wired into `MetalBackend` via `mod buffers; buffer_cache: Arc<BufferCache>`; `dispatch_histograms` replaces three `newBufferWithBytes` calls with cache-backed variants.
  - **`benchmarks/metal_histogram.py`** — expanded from the S1.14 shape-only grid into a named-scenario harness: `shape_grid`, `depth_sweep`, `bins_sweep`, `estimator_sweep`, `task_mix`, `metal_friendly`, `all`. The `metal_friendly` scenario explicitly tests the configurations theoretically most favourable to Stage 1 Metal (deep trees, 1024 bins, multiclass K=10) so we can disprove the "maybe Stage 1 wins somewhere" hypothesis directly.
  - **`docs/metal-backend/BENCHMARKS.md`** — rewritten against post-cache numbers. Shape-grid speedups moved from 0.03×–0.25× to 0.03×–0.28× (largest absolute win: 1M × 1000 dropped 86.8s → 70.7s). `metal_friendly` stays at 0.06×–0.09× across every config, proving Stage 1 cannot cross break-even on realistic shapes.
  - **`docs/limitations.md`** — replaced the "CPU-Only Runtime" section with "Metal Backend is Infrastructural (Stage 1)": cites both BENCHMARKS.md scenarios, states `device="cpu"` as the recommended default for every Stage 1 shape, explains the Stage 2+3 path to the decisive win, documents `native_runtime_info()` fields (`metal_available`, `metal4_available`, `gpu_family`) under "How to detect the backend", and the `ALLOYGBM_METAL_DISABLE=1` escape hatch.
  - Bit-exactness: all 21 `test_metal_backend.py` cases (including the 50k × 100 × 20-estimator golden test on train + held-out eval) pass with the cache wired in.
- [x] **S1.16** Full verification sweep:
  - `cargo check --workspace` green; `cargo check --workspace --no-default-features` green.
  - `cargo clippy --workspace --all-targets -- -D warnings` and `cargo clippy --workspace --all-targets --no-default-features -- -D warnings` both clean.
  - `cargo fmt --all --check` clean.
  - `cargo test --workspace --exclude alloygbm-python` (both with and without `--no-default-features`): **183 tests pass** (23 core + 7 backend_metal + 10 backend_cpu + 32 categorical + 69 engine + 19 predictor + 23 shap). The `alloygbm-python` crate is tested via pytest, not `cargo test`, because PyO3 extension modules can't produce a standalone test binary on macOS without the python framework linker dance — consistent with prior Stage 1 verification practice.
  - `maturin develop --release --manifest-path bindings/python/Cargo.toml` (default features) → **353/353 Python tests pass** (21 Metal-backend cases, including the 50k × 100 × 20-estimator golden bit-exactness test, plus the 332 pre-existing cases).
  - `maturin develop --release --manifest-path bindings/python/Cargo.toml --no-default-features` → **334 pass + 19 skipped** (Metal-gated cases correctly skip; the new `MetalFallbackTests` gate was added this sub-task — see below).
  - **Test fix landed during S1.16**: `MetalFallbackTests` in `test_metal_backend.py` now probes whether the `metal` crate feature is compiled in (via a one-shot `ALLOYGBM_METAL_DISABLE=1` subprocess warning probe) and `@unittest.skipUnless`-gates the class on that. Without this gate, no-default-features builds failed `test_fallback_emits_runtime_warning` because the escape-hatch warning text only exists when the feature is compiled in.

---

## Stage 1 — Complete

Stage 1 is closed as of 2026-04-20. Summary of what shipped:

- `crates/backend_metal` — full crate with device probe, MSL histogram kernel (privatised threadgroup + two-pass deterministic reduce, no float atomics), pipeline cache with `MTLBinaryArchive`, persistent `BufferCache` for binned-matrix / gradients / row-indices reuse.
- `bindings/python` — `device="cpu"|"metal"|"auto"` on all three estimators, `RuntimeBackend` enum, warn-and-fallback on Metal init failure, `ALLOYGBM_METAL_DISABLE=1` escape hatch, `trained_device` in artifact metadata, `native_runtime_info()` extended with `metal_available` / `metal4_available` / `gpu_family`.
- `benchmarks/metal_histogram.py` — named-scenario harness (`shape_grid`, `depth_sweep`, `bins_sweep`, `estimator_sweep`, `task_mix`, `metal_friendly`, `all`); M4 reference numbers archived as JSON.
- `docs/metal-backend/{BENCHMARKS,STATUS,SESSIONS,DECISIONS,BUGS}.md` — the working-set and rationale.
- `docs/limitations.md` — "Metal Backend is Infrastructural (Stage 1)" section with recommended-default `device="cpu"`.

**Throughput finding (expected):** Stage 1 Metal is uniformly slower than CPU across every shape in `shape_grid` (0.03×–0.28× CPU) and every config in `metal_friendly` (0.06×–0.09× CPU). This is architectural — the per-level CPU round-trip for split finding and row partitioning dominates latency; histogram acceleration alone cannot close the gap. Stages 2+3 eliminate that round-trip and are where the decisive 4-5× win lands.

**Bit-exactness contract held:** every Metal-trained model's predictions match its CPU-trained counterpart exactly, verified by the S1.13 50k-row × 100-feature × 20-estimator golden test on both the training set and a held-out 5k-row eval set.

---

## Next Up

1. Open Stage 2 via `ExitPlanMode`: GPU best-split + histogram subtraction on Metal (prefix-scan + argmax kernel; level-parallel dispatch; single compute pass per tree level). Keep Stage 1's infrastructure (BufferCache, pipeline cache, warn-and-fallback, device plumbing) as-is — Stage 2 rides on top of it.

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
