# Metal Backend — Decision Log

Append-only. One short entry per architectural call made during implementation
that deviates from or refines the approved plan. Newest at the bottom so the
history reads chronologically top-to-bottom.

Entry format:
```
## D-NNN: Short title
Date: YYYY-MM-DD
Stage: <stage>
Decision: <one sentence>
Why: <one paragraph>
Alternatives considered: <one line each>
```

---

## D-001: Raw Metal over MLX

Date: 2026-04-18
Stage: planning
Decision: Build the GPU backend on raw Metal (objc2-metal + hand-written MSL),
not MLX.
Why: MLX's `scatter_add` is explicitly non-deterministic on f32, which
violates our reproducibility requirement. MLX restricts distribution to
macOS 14+ and Apple-Silicon-only wheels, blocking Intel-Mac/Linux fallback.
Custom MSL is required regardless for deterministic histograms — adding MLX
on top of that is pure dependency overhead.
Alternatives considered: MLX end-to-end (rejected — determinism + distribution);
hybrid MLX-for-tensor-ops + raw-Metal-for-scatter (rejected — MLX buys us
nothing for ops we don't have).

---

## D-002: Hybrid Metal 3 baseline + Metal 4 fast path

Date: 2026-04-18
Stage: planning
Decision: Target Metal 3 (macOS 13+) as the baseline; detect
`MTLGPUFamilyMetal4` at runtime and opt into Metal 4 enhancements
(pipeline harvesting, ICBs, `MTLResidencySet`) only where available.
Why: Metal 4 requires macOS 26 Tahoe which has a narrow install base. Metal 3
has every API needed for Stage 1 (compute encoders, argument buffers, atomics,
`MTLHeap`). Hybrid path widens reach ~dramatically for ~20% extra engineering
behind a single runtime capability flag.
Alternatives considered: Metal 4 only (rejected — excludes macOS 14/15);
Metal 3 only (rejected — forfeits ICB + pipeline harvesting gains in Stage 3).

---

## D-003: No float atomics anywhere

Date: 2026-04-18
Stage: planning
Decision: Histogram accumulation scatters into per-threadgroup private
histograms in threadgroup memory, then a deterministic two-pass tree reduce
through a device-memory scratch buffer. No `atomic_fetch_add_explicit` on
floats at any memory level.
Why: Float addition is non-associative; atomic order is non-deterministic.
Bit-exact reproducibility is a stated hard constraint. Two-pass reduction
guarantees deterministic reduction order at the cost of a small scratch
buffer (`num_threadgroups × F × B × 2 × sizeof(f32)`).
Alternatives considered: Native float atomics with epsilon-tolerance asserts
(rejected — would mask edge-case tree-split divergence); CAS-loop
deterministic atomics (rejected — more complex than a clean two-pass reduce).

---

## D-004: `RuntimeBackend` enum wrapper at the PyO3 boundary

Date: 2026-04-18
Stage: planning
Decision: Do not rewrite the engine to use `Box<dyn BackendOps>`. Keep
`Trainer::fit_iterations<B: BackendOps, O: ObjectiveOps>` generic. At the
PyO3 boundary add one `RuntimeBackend::{Cpu, Metal}` enum that implements
`BackendOps` by forwarding each method.
Why: Preserves static dispatch and monomorphization for both backends. Keeps
the engine ignorant of runtime device selection. Branch cost is one enum
discriminant at each method call — negligible vs. the compute inside.
Alternatives considered: `Box<dyn BackendOps>` end-to-end (rejected — loses
monomorphization, touches every engine generic bound);
generic-over-two-types (rejected — combinatorial explosion across objectives).

---

## D-005: `metal` feature default-on for macOS

Date: 2026-04-18
Stage: planning
Decision: Gate `backend_metal` behind a cargo feature `metal` on
`bindings/python`. Feature default-on via
`[target.'cfg(target_os = "macos")'.dependencies]`. Off on Linux/Windows/Intel.
Why: macOS users get GPU acceleration with no opt-in. Non-macOS and
Intel-Mac wheels build cleanly without Metal linkage. Source users can
`--no-default-features` to disable.
Alternatives considered: Always compiled on macOS (rejected — forces Metal
linkage even for source users who don't want it); opt-in only (rejected —
wheel users wouldn't benefit unless we ship separate wheels).

---

## D-006: Ship MSL source, compile at runtime, cache harvested pipelines

Date: 2026-04-18
Stage: planning
Decision: Embed `.metal` source via `include_str!`, compile at
`MetalBackend::new()` asynchronously, and cache the serialized pipeline
state in `~/Library/Caches/com.alloygbm/pipelines-<gpu>-<arch>.metallib`
(via `MTLBinaryArchive`).
Why: Avoids maturin-wheel-build-time dependency on `xcrun metal`. Forward-
compatible across macOS 14/15/16+/26 and M1–M4. First-run compile stutter
is amortized by the disk cache. Pipeline harvesting is a Metal 4 fast path
on top.
Alternatives considered: Precompiled `.metallib` in the wheel (rejected —
cross-OS/arch compile matrix pain in CI); AOT at build time via build.rs
(rejected — same CI pain plus brittle).

---

## D-007: Relax `unsafe_code = "forbid"` to `deny` in `backend_metal` only

Date: 2026-04-18
Stage: S1.2
Decision: `crates/backend_metal/Cargo.toml` does not inherit workspace
lints. It declares its own `[lints.rust]` with `unsafe_code = "deny"`,
so FFI call sites must opt in per-site via `#[allow(unsafe_code)]` +
`unsafe { ... }`. Every other workspace crate still inherits
`unsafe_code = "forbid"`.
Why: `objc2-metal` surfaces most Metal APIs (command encoding, buffer
creation, selector calls to cover Metal 4 gaps) as `unsafe fn`. The
workspace-wide `forbid` is unsatisfiable for any real Metal wrapper.
`deny`-at-crate + `allow`-at-site keeps unsafe visible, auditable, and
narrowly scoped — every `unsafe` block remains a review point — while
preserving the workspace invariant everywhere else.
Alternatives considered: relax the workspace lint globally (rejected —
punishes every other crate for one crate's FFI needs); pick a fully-safe
Metal wrapper (rejected — no mature option covers Metal 3 + Metal 4 +
compute + residency sets); hand-audited `unsafe_op_in_unsafe_fn` only
(rejected — still requires the outer lint to be relaxed).

---

## D-008: Histogram-bin counts computed on CPU post-readback

Date: 2026-04-19
Stage: S1.4
Decision: The MSL histogram kernel emits only `(grad_sum, hess_sum)`
float2 pairs. Per-bin row counts are reconstructed on the CPU after
readback by a single pass over `node.row_indices × selected_features`
reading `BinnedMatrix::col_bin`.
Why: The kernel's threadgroup-memory budget is already near the
Apple7 32 KB ceiling at `MAX_BIN_COUNT = 4096` (32 KB for the float2
histogram). Tracking counts in threadgroup memory would add a
`uint local_counts[MAX_BIN_COUNT]` of up to 16 KB, pushing beyond the
cliff. Count accumulation is inherently deterministic by integer
arithmetic — no order-dependence, no float-atomic concern — so
placing it on CPU never compromises the bit-exactness contract with
`CpuBackend`. Measured overhead on the 500-row fixture is a single-
digit millisecond scan; at 1M×100 it's roughly 100M u8 reads + 100M
increments, well inside the budget a Metal-accelerated training step
is already paying for bulk data movement. Revisited in Stage 2 if
profiling shows it as a hotspot (options: second integer-only
`histogram_count_build` kernel running in parallel with the float
pass; or widen the float scratch to `float3` once we shrink
`MAX_BIN_COUNT` to stay within tgmem).
Alternatives considered: extend `local_hist` to `float3(g,h,c)` or
parallel `uint counts` in tgmem (rejected — blows the Apple7 tgmem
cliff at MAX_BIN_COUNT=4096, would require shrinking the bin ceiling);
separate Metal count kernel (rejected for S1.4 — doubles kernel
surface area for no correctness gain; deferred to Stage 2).
