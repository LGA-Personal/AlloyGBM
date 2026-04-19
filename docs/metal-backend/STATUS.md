# Metal Backend — Current Status

**Last updated:** 2026-04-19 (S1.5 landed)
**Active stage:** Stage 1 — Histogram build on Metal
**Active sub-task:** S1.7 — `RuntimeBackend` enum + `device: &str` PyO3 plumbing (next)

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
- [ ] **S1.7** `RuntimeBackend` enum in `bindings/python/src/lib.rs`; `device: &str` on every `train_*` pyfunction
- [ ] **S1.8** Python `device="cpu"|"metal"|"auto"` parameter across `GBMRegressor`, `GBMClassifier`, `GBMRanker`
- [ ] **S1.9** Warn-and-fallback on Metal init failure; store resolved device in artifact metadata JSON
- [ ] **S1.10** Extend `native_runtime_info()` with `metal_available`, `metal4_available`, `gpu_family`
- [ ] **S1.11** Rust unit tests for histogram kernel correctness (<1000 rows, hand-computed reference) *(delivered in S1.4: `histogram_matches_cpu_small_fixture` + `histogram_feature_subset_matches_cpu`; S1.5 adds `pipeline_cache_returns_identical_arc_on_second_call`)*
- [ ] **S1.12** `bindings/python/tests/test_metal_backend.py` — macOS + availability gated; covers regression, classification, ranking, NaN, B=16/255/65535
- [ ] **S1.13** Bit-exactness golden test: seeded (50k rows × 100 features) CPU vs Metal → identical `artifact_bytes`
- [ ] **S1.14** `benchmarks/metal_histogram.py` — CPU vs Metal throughput at (10k, 100k, 1M, 10M) × (10, 100, 1000)
- [ ] **S1.15** `docs/limitations.md` note on breakeven + availability
- [ ] **S1.16** Full verification sweep (cargo check/test/clippy/fmt, maturin develop, pytest)

---

## Next Up

1. **S1.7** — PyO3 plumbing. Today `bindings/python/src/lib.rs:3`
   hard-codes `CpuBackend` and every `train_*` pyfunction takes
   `backend: &CpuBackend` implicitly. The approved plan (and
   **D-004**) says:
   - Add a `RuntimeBackend::{Cpu(CpuBackend), Metal(MetalBackend)}`
     enum inside `bindings/python/src/lib.rs` that implements
     `BackendOps` by forwarding each method.
   - Every `train_*` pyfunction (regression, binary, multiclass,
     ranking; sparse + dense variants — count them all in
     `bindings/python/src/lib.rs`) grows a `device: &str` parameter
     resolved to one of `{"cpu","metal","auto"}`, producing the enum
     and passing it through to `trainer.fit_iterations`.
   - `device = "auto"` for S1.7 just means "cpu" (per plan — the
     shape-based heuristic is deferred to Stage 2+).
   - Preserve static monomorphization: the enum itself is the type
     the trainer sees, so `Trainer::fit_iterations<RuntimeBackend,
     ObjectiveOps>` monomorphizes once per objective, branch cost =
     one discriminant check inside each forwarded method.
2. **S1.8** then threads `device` through the Python estimators:
   `GBMRegressor`, `GBMClassifier`, `GBMRanker` — `__init__`,
   `get_params`, `set_params`, `__repr__`, `_params_order`, pickle
   state. Validate against `{"cpu","metal","auto"}`.
3. **S1.9** layers a `try { MetalBackend::new() } catch { warn; use
   CpuBackend }` fallback on the PyO3 side, and stores the
   `resolved_device` in artifact metadata (append-only field, so the
   hand-rolled positional parser stays back-compat).
4. **S1.10** is a cheap PyO3 + Python extension to
   `native_runtime_info()` exposing `metal_available: bool`,
   `metal4_available: bool`, `gpu_family: Optional[str]`. Relies on
   `MetalDevice::probe()` → `MetalCapabilities`.

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
