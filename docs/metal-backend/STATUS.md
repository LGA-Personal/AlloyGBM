# Metal Backend — Current Status

**Last updated:** 2026-04-20 (S1.12 landed)
**Active stage:** Stage 1 — Histogram build on Metal
**Active sub-task:** S1.13 — bit-exactness golden artifact test at scale (next)

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
- [ ] **S1.13** Bit-exactness golden test: seeded (50k rows × 100 features) CPU vs Metal → identical `artifact_bytes`
- [ ] **S1.14** `benchmarks/metal_histogram.py` — CPU vs Metal throughput at (10k, 100k, 1M, 10M) × (10, 100, 1000)
- [ ] **S1.15** `docs/limitations.md` note on breakeven + availability
- [ ] **S1.16** Full verification sweep (cargo check/test/clippy/fmt, maturin develop, pytest)

---

## Next Up

1. **S1.13** bit-exactness golden test at scale: seeded fit at
   (50k × 100, 255 bins) under CPU and Metal → identical
   prediction stream over the full training set. Uses the same
   `native_runtime_info().metal_available` skip guard as S1.12 so
   the test is a no-op on non-Metal runners. Scope note: the plan
   originally called for identical `artifact_bytes`, but S1.12
   proved that is not achievable as-written because the artifact
   metadata JSON encodes `trained_device` and its length prefix
   (Metal vs CPU legitimately differ by a few bytes there);
   prediction bit-exactness over the full training set is the
   stronger observable contract and is what S1.13 should assert.
2. **S1.14** `benchmarks/metal_histogram.py` throughput harness.
3. **S1.15** `docs/limitations.md` breakeven + availability note.
4. **S1.16** Full verification sweep before declaring Stage 1
   complete.

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
