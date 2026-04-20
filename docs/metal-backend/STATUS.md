# Metal Backend — Current Status

**Last updated:** 2026-04-20 (S1.15 landed)
**Active stage:** Stage 1 — Histogram build on Metal
**Active sub-task:** S1.16 — full verification sweep (next)

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
- [ ] **S1.16** Full verification sweep (cargo check/test/clippy/fmt, maturin develop, pytest)

---

## Next Up

1. **S1.16** Full verification sweep before declaring Stage 1
   complete: `cargo check/test/clippy/fmt` across the workspace
   (with and without `--no-default-features`), `maturin develop
   --release`, and the full Python pytest run. This is the last
   Stage 1 item — on success, Stage 1 closes and the next
   `ExitPlanMode` round opens Stage 2 (GPU best-split).

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
