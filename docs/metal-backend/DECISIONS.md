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

---

## D-009: Archive serialization is drop-time only, atomic rename

Date: 2026-04-19
Stage: S1.5
Decision: `HistogramPipelineCache` writes its `MTLBinaryArchive` to
disk exactly once — in `Drop::drop` — by first writing to
`<path>.metalarchive.tmp` and then calling `std::fs::rename` into
place. An in-memory `dirty: Mutex<bool>` flag skips the write when
no new pipelines were added this session. Failure to add, serialize,
or rename is logged with `eprintln!` and swallowed; Metal simply
compiles on the next run.
Why: Apple's `MTLBinaryArchive` documentation warns that "updating a
MTLBinaryArchive at runtime in a shipping app configuration is not
recommended; such a scenario requires corruption resiliency".
Writing on every `get_or_build` would give both worst-case overhead
and worst-case corruption risk. Deferring to `Drop` means at most
one write per training session, and the temp-path + rename pattern
ensures a crash mid-write leaves the previous archive intact.
Skipping when `!dirty` avoids pointless rewrites on cache-hit-only
sessions (every subsequent run after the first).
Alternatives considered: serialize after each successful `add`
(rejected — corruption risk, per Apple); never serialize, treat as
in-process cache only (rejected — defeats the whole point of cross-
run persistence); background thread flushing periodically (rejected
— unnecessary complexity for a single-writer lifecycle).

---

## D-010: Mark `HistogramPipelineCache` `unsafe impl Send + Sync`

Date: 2026-04-19
Stage: S1.5
Decision: Add `unsafe impl Send for HistogramPipelineCache {}` and
`unsafe impl Sync for HistogramPipelineCache {}` inside the crate so
`MetalBackend` can store it behind `Arc` and participate in any
Rayon-driven engine calls.
Why: `objc2-metal` does not auto-derive `Send`/`Sync` for Metal
protocol objects, but Apple documents `MTLDevice`, `MTLLibrary`, and
`MTLComputePipelineState` as thread-safe for concurrent use: any
thread may query pipelines or submit commands without external
synchronization. The one mutation point we introduce —
`MTLBinaryArchive::addComputePipelineFunctions...` — is ordered by
our own `entries: Mutex<HashMap>`, which serializes slow-path
pipeline builds so no two threads touch the archive simultaneously.
The `dirty` flag is also behind a `Mutex`. With those invariants the
struct is safe to share across threads.
Alternatives considered: use `Rc` instead of `Arc` (rejected — breaks
any future `BackendOps: Send` bound and forces single-thread usage);
wrap the whole cache in a single `Mutex` (rejected — serialises
every cache-hit fast path that should be lock-free after `entries`
read-lock returns an `Arc`); require callers to clone pipelines on
demand (rejected — reintroduces the per-dispatch compile cost we
just eliminated).
