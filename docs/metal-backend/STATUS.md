# Metal Backend — Current Status

**Last updated:** 2026-04-18 (S1.2 landed)
**Active stage:** Stage 1 — Histogram build on Metal
**Active sub-task:** S1.3 — MSL histogram kernel (next)

---

## Stage 1 Checklist

Order matches the approved plan in
[/Users/lashby/.claude/plans/okay-add-this-notebook-structured-star.md](../../.claude/plans/okay-add-this-notebook-structured-star.md).

- [x] **S1.1** Scaffold `crates/backend_metal` (Cargo.toml, build.rs, empty lib, workspace member, feature flag wired)
- [x] **S1.2** Device + capability probe (`device.rs`) — `MTLCreateSystemDefaultDevice`, `MTLGPUFamilyApple7`, `MTLGPUFamilyMetal4` flag
- [ ] **S1.3** MSL histogram kernel (`shaders/histogram.metal`) — privatized threadgroup histograms + two-pass deterministic reduce
- [ ] **S1.4** Rust-side orchestration (`kernels/histogram.rs`) — buffer wrapping, encoding, submit, readback
- [ ] **S1.5** Pipeline compilation + `MTLBinaryArchive` cache at `~/Library/Caches/com.alloygbm/`
- [ ] **S1.6** `MetalBackend` delegates non-histogram `BackendOps` methods to embedded `CpuBackend`
- [ ] **S1.7** `RuntimeBackend` enum in `bindings/python/src/lib.rs`; `device: &str` on every `train_*` pyfunction
- [ ] **S1.8** Python `device="cpu"|"metal"|"auto"` parameter across `GBMRegressor`, `GBMClassifier`, `GBMRanker`
- [ ] **S1.9** Warn-and-fallback on Metal init failure; store resolved device in artifact metadata JSON
- [ ] **S1.10** Extend `native_runtime_info()` with `metal_available`, `metal4_available`, `gpu_family`
- [ ] **S1.11** Rust unit tests for histogram kernel correctness (<1000 rows, hand-computed reference)
- [ ] **S1.12** `bindings/python/tests/test_metal_backend.py` — macOS + availability gated; covers regression, classification, ranking, NaN, B=16/255/65535
- [ ] **S1.13** Bit-exactness golden test: seeded (50k rows × 100 features) CPU vs Metal → identical `artifact_bytes`
- [ ] **S1.14** `benchmarks/metal_histogram.py` — CPU vs Metal throughput at (10k, 100k, 1M, 10M) × (10, 100, 1000)
- [ ] **S1.15** `docs/limitations.md` note on breakeven + availability
- [ ] **S1.16** Full verification sweep (cargo check/test/clippy/fmt, maturin develop, pytest)

---

## Next Up

1. **S1.3** — MSL histogram kernel (`shaders/histogram.metal`): write MSL for privatized per-threadgroup histograms in threadgroup memory (`F_TILE × B × 2` f32s), then a deterministic tree-reduce through a device-memory scratch buffer. Two compute passes, no float atomics. Embed via `include_str!` from `src/kernels/histogram.rs`. Compilation + dispatch wiring land in S1.4/S1.5 — S1.3 ships just the `.metal` source and a Rust module declaring `shader_source()`.
2. Then **S1.4** (Rust-side orchestration of the histogram dispatch).

---

## Blockers / Open Questions

_None yet._

---

## Cross-Stage Roadmap (reference only)

- **Stage 2** — GPU best-split + histogram subtraction (planned, not scoped)
- **Stage 3** — GPU row partitioning + Metal 4 ICBs (planned, not scoped)
- **Stage 4** — GPU inference tree traversal (planned, not scoped)

Each stage lands via its own `ExitPlanMode` round.
