# Metal Backend — Current Status

**Last updated:** 2026-04-19 (S1.4 landed)
**Active stage:** Stage 1 — Histogram build on Metal
**Active sub-task:** S1.5 — Pipeline compilation + `MTLBinaryArchive` cache (next)

---

## Stage 1 Checklist

Order matches the approved plan in
[/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md).

- [x] **S1.1** Scaffold `crates/backend_metal` (Cargo.toml, build.rs, empty lib, workspace member, feature flag wired)
- [x] **S1.2** Device + capability probe (`device.rs`) — `MTLCreateSystemDefaultDevice`, `MTLGPUFamilyApple7`, `MTLGPUFamilyMetal4` flag
- [x] **S1.3** MSL histogram kernel (`shaders/histogram.metal`) — privatized threadgroup histograms + two-pass deterministic reduce
- [x] **S1.4** Rust-side orchestration (`kernels/histogram.rs` + `pipelines.rs`) — buffer wrapping, encoding, submit, readback; `impl BackendOps for MetalBackend` (histogram on Metal, rest delegated to embedded `CpuBackend`)
- [ ] **S1.5** Pipeline compilation + `MTLBinaryArchive` cache at `~/Library/Caches/com.alloygbm/`
- [ ] **S1.6** `MetalBackend` delegates non-histogram `BackendOps` methods to embedded `CpuBackend` *(delivered in S1.4)*
- [ ] **S1.7** `RuntimeBackend` enum in `bindings/python/src/lib.rs`; `device: &str` on every `train_*` pyfunction
- [ ] **S1.8** Python `device="cpu"|"metal"|"auto"` parameter across `GBMRegressor`, `GBMClassifier`, `GBMRanker`
- [ ] **S1.9** Warn-and-fallback on Metal init failure; store resolved device in artifact metadata JSON
- [ ] **S1.10** Extend `native_runtime_info()` with `metal_available`, `metal4_available`, `gpu_family`
- [ ] **S1.11** Rust unit tests for histogram kernel correctness (<1000 rows, hand-computed reference) *(delivered in S1.4: `histogram_matches_cpu_small_fixture` + `histogram_feature_subset_matches_cpu`)*
- [ ] **S1.12** `bindings/python/tests/test_metal_backend.py` — macOS + availability gated; covers regression, classification, ranking, NaN, B=16/255/65535
- [ ] **S1.13** Bit-exactness golden test: seeded (50k rows × 100 features) CPU vs Metal → identical `artifact_bytes`
- [ ] **S1.14** `benchmarks/metal_histogram.py` — CPU vs Metal throughput at (10k, 100k, 1M, 10M) × (10, 100, 1000)
- [ ] **S1.15** `docs/limitations.md` note on breakeven + availability
- [ ] **S1.16** Full verification sweep (cargo check/test/clippy/fmt, maturin develop, pytest)

---

## Next Up

1. **S1.5** — Pipeline compilation + caching. Today S1.4 calls
   `build_histogram_pipelines` afresh on every dispatch, which re-
   compiles the MSL library and re-builds both compute pipeline
   states for every histogram build. That's dozens of milliseconds
   per tree node at minimum. S1.5 introduces:
   - An `MTLBinaryArchive` on disk at
     `~/Library/Caches/com.alloygbm/pipelines-<gpu-family>-<macos>.metalarchive`
     that persists across runs so the first run compiles once and every
     subsequent run is cache-hit.
   - An in-process LRU keyed by `(bin_count, use_u16_bins)` so that
     training at mixed bin counts (rare) still amortises across a
     single session.
   - A best-effort `addComputePipelineFunctionsWithDescriptor:error:`
     call on the archive during pipeline build so Metal 4 pipeline
     harvesting can populate the archive opportunistically.
2. Then **S1.6** is trivially done (already shipped in S1.4), update
   the checklist mark and move to **S1.7** (Python plumbing).
3. **S1.11** is partially done (two small-fixture correctness tests are
   in place); extend with larger-seed + NaN-bin coverage when we reach
   that rung.

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
