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

---

## D-011: Stage 2 defers GPU `subtract_histogram_bundle` to Stage 3

Date: 2026-04-20
Stage: S2 plan
Decision: Keep `subtract_histogram_bundle` on the CPU in Stage 2.
The Stage 2 plan had originally sketched GPU subtract as part of the
incremental win; this ADR records the explicit deferral.

Why: GPU subtract only becomes net-positive if the histograms stay
GPU-resident across calls — otherwise the CPU→GPU memcpy + GPU→CPU
readback of ~F×B floats per call dwarfs the F×B subtract it
replaces. Passing histogram *handles* (not owned `HistogramBundle`
values) across the `BackendOps` boundary is a surface change that
naturally belongs with Stage 3, where Metal 4 ICBs and GPU row
partitioning also require histograms to live on-device across
levels. Shipping it in Stage 2 would either (a) force the engine
refactor prematurely or (b) do the memcpy anyway and lose on every
call.

Alternatives considered: engine refactor to pass handles now
(rejected — rushes the Stage 3 surface change without its
correlated Stage 3 benefits); do GPU subtract with per-call
memcpy (rejected — pessimises the common path); leave Stage 2
incremental expectation at plan-original 2–3× (rejected — the
deferral genuinely moves the ceiling down; the plan was updated
to 1.5–2.5× to reflect reality).

Consequence: Stage 2 captures GPU speedup on `best_split` only.
See BENCHMARKS.md §"2026-04-20 — Apple M4 (Stage 2 baseline)" for
the measured outcome — the deferral was a non-trivial ingredient
in the Stage 2 crossover miss.

---

## D-012: Categorical features stay on CPU in Stage 2

Date: 2026-04-20
Stage: S2 plan
Decision: `MetalBackend::best_split_with_options` partitions the
feature set at call time — continuous features flow through the
GPU split kernel; categorical features are delegated to the
embedded `CpuBackend`'s Fisher-sort path
(`best_split_for_categorical_feature`). Per-feature candidates are
combined with `feature_weights` on the CPU at the end.

Why: the categorical split-finder sorts categories by per-class
score, then prefix-scans over sorted order to find the optimal
binary partition (Fisher 1958). Sorting on-GPU with stable
deterministic ordering, then doing a bin-conditional prefix scan,
is a genuinely harder GPU problem than the continuous-scan split
kernel — closer in shape to GPU radix-sort-then-scan primitives
than to the simple SIMD block-scan the continuous kernel uses.
The Fisher-sort variant ships in production at roughly the
frequency categorical splits appear in real fits — which is to
say, rarely enough that the CPU path handles it comfortably.

Alternatives considered: GPU categorical kernel now (rejected —
separate research problem, CPU path is adequate, Stage 2 is
already scope-heavy); disable categorical splits when
`device="metal"` (rejected — semantic break from the CPU path,
user-visible regression); require users to move categoricals
to `device="cpu"` manually (rejected — no-op of same cost
compared to the in-process delegation we already have).

Consequence: Stage 2 correctness test
`best_split_with_categorical_feature_delegates_to_cpu` exercises
the mixed-feature path. Pure-categorical models take the CPU
fast path entirely; the GPU dispatch is skipped.

---

## D-013: Stage 2 relaxes the bit-exactness gate to structural-plus-ulp-epsilon

Date: 2026-04-20
Stage: S2.6
Decision: Stage 2 replaces Stage 1's byte-identical-artifact gate
with a two-layer gate:
  (a) structural equivalence — same `(feature_index, threshold_bin,
      default_left)` per split, enforced in Rust unit tests on
      small well-conditioned fixtures (see `best_split_matches_cpu_*`
      in `crates/backend_metal/src/lib.rs`);
  (b) prediction equality within `atol=1e-5, rtol=1e-5` on the
      50k × 100 × 255 × 20 golden shape, and within
      `atol=0.1, rtol=0.1` on tiny shapes (≤1024 rows) where
      near-tied root splits can flip winner under SIMD block-scan
      reduction vs CPU serial sweep.

Why: the split kernel accumulates `left_grad`, `left_hess`,
`parent_gain_term` via SIMD `simd_prefix_inclusive_sum` + block-
scan, which is a tree reduction — not order-identical to the
CPU's strict left-to-right serial sum. At typical f32
magnitudes this drifts by a few ulps; at tiny shapes the drift is
enough to flip near-tied winners, which cascades into macroscopic
prediction deltas (~0.1) in ≤0.10f rows. The full 50k × 100
golden test still passes bit-exact `array_equal` predictions,
confirming the kernel is correct where gains are comfortably
separated — the structural-ulp gate catches the genuinely
different regimes.

Alternatives considered: lane-serial Phase 1 + Phase 2 in the
kernel for exact CPU-match (rejected — defeats the GPU
parallelism; single-lane walks of 256 bins × 200 features per
dispatch are ~5× slower than the current block-scan); loosen the
golden-shape tolerance to `1e-3` (rejected — masks genuine
regressions; the empirical result is tighter and should be held
to it); drop the Python parity tests entirely and rely on Rust
unit tests (rejected — the Stage 1 golden framing
(artifact-level + prediction-level) is user-observable and
should remain a regression gate).

---

## D-014: Stage 2 split kernel uses a single dispatch with CPU cross-feature argmax

Date: 2026-04-20
Stage: S2.1/S2.2
Decision: The approved plan sketched a two-dispatch design —
Dispatch 1 computed per-feature candidates, Dispatch 2 ran a
cross-feature argmax. The shipped implementation collapses this
into a single kernel + CPU-side argmax.

Why: `n_features` per call is at most a few thousand in
realistic fits; the per-feature `FeatureSplitCandidate` is 40
bytes; the readback is ~40 KiB per call, negligible next to the
much-bigger `HistogramBundle` memcpy. A second dispatch costs
~10–50 μs of fixed latency on M4 — the same order as the argmax
it saves. The CPU-side argmax also lets us apply
`feature_weights` (`weighted_gain = gain * weight`) in the place
where the weights are already owned; a GPU version would need a
third buffer slot.

Alternatives considered: follow the plan's two-dispatch design
(rejected — measurably slower at realistic `n_features`;
refactor surface kept for Stage 3 if residency justifies it);
do argmax on-CPU but keep a second dispatch for the
feature-weight multiply (rejected — pure overhead).

---

## D-015: Enum-variant storage API for GPU residency (`HistogramStorage`, `RowIndexStorage`)

Date: 2026-04-21
Stage: S3 plan / S3.1
Decision: `HistogramBundle` and row-index carriers gain a
`storage: HistogramStorage` / `rows: RowIndexStorage` field whose
variants are `Cpu(...)` (today's owned `Vec<...>` payload) and
`Gpu { handle, shape }` (opaque `u64` newtype handle plus the
shape metadata the engine needs for pattern-matching on the
variant). Every engine field-read pattern-matches on the variant;
the CPU backend only ever populates `Cpu(..)`; the Metal backend
returns `Gpu(..)` once residency is wired (S3.7c+d).

Why: the competing design (β) added a *parallel* `GpuHandle`
type alongside existing owned types with an `.as_cpu()` escape
hatch. That escape hatch silently re-introduces the GPU→CPU
memcpy that Stage 3 exists to eliminate — any trainer site that
reaches for `.as_cpu()` unknowingly defeats the whole
architectural change. Enum variants put the type system in
charge of that invariant: a `Gpu(..)` arm cannot be read as a
Cpu payload without explicit pattern matching, so every new
call site visibly declares whether it expects on-device data.
The CPU backend path remains semantically identical (it always
constructs `Cpu(..)`), preserving the full existing test suite
as a regression gate.

Alternatives considered: parallel handle-type-with-fallback
design β (rejected — `.as_cpu()` silently reintroduces the
memcpy; type system can't enforce "no accidental readback");
engine-level `dyn BackendStorage` trait objects (rejected —
loses monomorphization, touches every generic bound, and
Stage 3's code audit shows only ≤10 `feature_histograms()`
call sites so a full trait-object rewrite is overkill);
keep owned `Vec` payloads and add a side-channel `Option<GpuHandle>`
(rejected — invariant "exactly one of these is authoritative"
can't be encoded in the type system; code review burden
forever).

---

## D-016: M2 free-on-consume residency budget with pathological-shape risk note

Date: 2026-04-21
Stage: S3 plan / S3.9
Decision: GPU histogram residency follows the M2 free-on-consume
policy: histograms live for exactly one level (level-wise) or
until the corresponding `PendingSplit` pops (leaf-wise), then
drop immediately. At fit start, `MetalBackend::check_histogram_budget(F, B, L)`
refuses the fit with `EngineError::BackendUnavailable` (containing
a `device="cpu"` fallback hint) when the projected peak
`F × B × L × 12` bytes exceeds 80 % of
`MTLDevice.recommendedMaxWorkingSetSize`. No LRU spill layer
at this stage.

Why: the histogram working set grows as one level width ×
`feature_count × bin_count × 12 bytes` (grad f32 + hess f32 +
counts u32). Free-on-consume bounds peak residency to that one
level — strictly smaller than today's CPU path, which keeps
histograms alive across the full level-wise sweep anyway.
The 80 % ceiling leaves headroom for kernel scratch, pipeline
state, buffer-cache slots, and the OS graphics stack sharing the
unified memory pool. `recommendedMaxWorkingSetSize` (not raw
`MTLDevice.maxBufferLength`) is the right budget question because
Apple's driver penalises working sets above it with paging-like
behaviour that spikes latency by orders of magnitude.

Pathological risk carried forward to M3 as a documented followup:
leaf-wise + `max_leaves=1024` + 1000 features + 1024 bins
projects to roughly 12 GiB of histogram residency, which
exceeds the 80 % ceiling on M1/M2/M3 8-16 GiB machines (passes
on M3/M4 Max 36 GiB+). These configs will hit the budget
refusal cleanly and fall back to CPU. M3 (probe-detected LRU
spill with a `ALLOYGBM_METAL_HISTOGRAM_BUDGET_GIB` env-var knob)
is the planned follow-up if a user reports a real fit blocked
by this ceiling. The risk note lives at the top of
`backend_metal/src/budget.rs` so the next reader of the code
sees it before modifying the policy.

Alternatives considered: unconditional fit with OS-level paging
(rejected — Apple's unified-memory paging behavior is not a
latency you want inside a training loop); LRU spill to CPU as
part of M2 (rejected — Stage 3 ships without that complexity,
M3 owns it); 50 % ceiling for safety (rejected — too
conservative, refuses fits that would complete cleanly;
80 % matches what WebGPU's working-set heuristic settled on);
arithmetic budget of `F × B × L × 8` ignoring the counts u32
(rejected — under-counts and admits fits that then OOM).

---

## D-017: Categorical features — partition on GPU, split on CPU (extends D-012)

Date: 2026-04-21
Stage: S3 plan / S3.5
Decision: the Stage 3 GPU row-partition kernel handles *both*
continuous and categorical splits via a `SPLIT_KIND` function
constant (0 = threshold compare on `binned_matrix[row, feat]`;
1 = bitset membership test against the categorical split's
encoded bitset). Categorical *split finding* (Fisher-sort
optimal-binary-partition) stays on the CPU, extending D-012.

Why: row partitioning is a uniform per-row operation trivially
suited to GPU parallelism — each row independently reads its
own feature value and emits a left/right flag, then a
stream-compaction phase produces the contiguous left/right row
buffers. The bitset-membership variant costs one additional
`(bin >> 5) & 31`-style bit test per row and no new data-
movement pattern. By contrast, optimal-binary-partition over
categories requires sort-by-per-class-score + prefix-scan-over-
sorted-order, which is a genuinely hard GPU problem (closer in
shape to radix-sort-then-scan primitives than to the simple
bin-scan we already have). The shape of categorical vs.
continuous splits on the partition side is nearly identical
on-GPU; on the split-finding side it's starkly different.
Bundling them on the partition side costs ~10 extra lines in
the kernel; unbundling on the split side saves an unbounded
research detour.

Alternatives considered: CPU row partitioning for categorical
splits too (rejected — forces a CPU round-trip for any fit
using mixed feature types, breaking Stage 3's residency
invariant); full GPU categorical split-finder in Stage 3
(rejected — research problem, orthogonal to Stage 3's
residency thesis, belongs in its own stage); encode
categorical splits as dense one-hot bins so the continuous
kernel handles them (rejected — inflates bin counts
unboundedly, defeats the compact bitset representation
already shipped in D-012 for prediction).

---

## D-018: `subtract_histogram_bundle`, `apply_split`, `reduce_sums` promoted to `BackendOps`

Date: 2026-04-21
Stage: S3 plan / S3.2
Decision: three operations that were previously engine-owned
free functions (`subtract_histogram_bundle`) or were already
trait methods but took CPU-owned `Vec`s (`apply_split`,
`reduce_sums`) become first-class `BackendOps` methods whose
signatures consume / produce `HistogramStorage` / `RowIndexStorage`
variants. CPU backend implementations are the existing
elementwise logic on the `Cpu(..)` arm; Metal overrides
dispatch kernels when both operands are `Gpu(..)`.

Why: with D-015's enum-variant storage API landing, the engine
no longer knows whether it's holding CPU or GPU data at the
time it wants to subtract one histogram from another or
partition rows by a split. The natural dispatch point is the
backend — exactly where we already pattern-match on the rest
of per-device behaviour. Keeping these as engine free functions
would require the engine to sniff the storage variant and
branch manually at every call site, duplicating trait-dispatch
logic. `subtract_histogram_bundle` in particular is called at
three trainer sites (level-wise smaller-first left/right;
leaf-wise larger-derivation) so the duplication cost is real.

Alternatives considered: leave `subtract_histogram_bundle` as
a free function and inside it branch on storage variant
(rejected — hides a backend dispatch inside an engine utility
function, making the Metal override invisible to trait-aware
readers); add a separate `GpuHistogramOps` trait with the
GPU-only operations (rejected — splits behaviour across two
traits for no benefit; the engine always has a `BackendOps`
in hand already); keep `reduce_sums` taking `&[u32]` and
extract-CPU-rows in the trainer (rejected — exactly the
memcpy pattern Stage 3 exists to eliminate, even though
`reduce_sums` only runs once per tree).

---

## D-019: Histogram kernel emits SoA output (separate `grad_out` + `hess_out`)

Date: 2026-04-21
Stage: 3 (S3.7c)
Decision: Change `histogram.metal`'s reduce-pass output from a single
AoS `device float2* output` buffer to two separate SoA buffers
`device float* grad_out` and `device float* hess_out`, each sized
`[n_features × BIN_COUNT]`. The scatter pass still uses an internal
`float2 local_hist[...]` for threadgroup memory density; only the
reduce pass's final write splits into two planes.
Why: `HistogramResidencyPool` (D-015, S3.7b) stores each histogram
as three SoA buffers — `grad`, `hess`, `counts` — so the buffers
can be wired directly into `split.metal`, whose input contract
(D-014) is also three SoA buffers: `grad_sums`, `hess_sums`,
`counts`. The AoS output from the Stage 1 histogram kernel was
internally consistent but misaligned with both neighbours, forcing
a CPU re-plane copy between `build_histograms` and `best_split`.
That copy is exactly the round-trip Stage 3 exists to eliminate.
Flipping the reduce pass to SoA lets the kernel output land
directly in pool-owned buffers that the split kernel reads
without any reshape. Counts stay on CPU per D-008 — they're
accumulated post-dispatch and written directly into the pool's
counts buffer via `contents()`. The scatter pass's internal
`float2` threadgroup layout is unchanged because the per-bin
single-writer discipline benefits from keeping `(grad, hess)`
coresident in one cache line.
Alternatives considered: keep AoS output and store an
interleaved `grad_hess` buffer in the pool (rejected — forces a
reshape before split reads it, defeating the zero-copy goal);
keep AoS and copy to SoA on the Rust side before split
(rejected — that's the round-trip we're eliminating); add a
dedicated AoS→SoA GPU kernel (rejected — one extra dispatch
for zero end-user value when the reduce kernel can emit SoA
directly at the same cost).


---

## D-020: Stage 3 kill criterion not met — three readback paths remain

Date: 2026-04-24
Stage: 3 (S3.12)
Decision: Stage 3 as currently shipped (S3.1–S3.11) does NOT cross
the approved `metal_friendly` >1.0× CPU bar. All deep-tree configs
land at 0.05×–0.06× CPU, within jitter of Stage 1 and Stage 2
baselines. Do not advance to Stage 4 / mark Stage 3 COMPLETE; the
residency infrastructure shipped correctly but the crossover
thesis is unmet because three of five overridden `BackendOps`
methods still do full CPU readbacks per call.
Why: `build_histograms` CPU count accumulation (D-008) reads back
the full row-index buffer via `slice::from_raw_parts(..).to_vec()`
on every call; `reduce_sums` reads back the full row-index buffer;
`apply_partition_leaf_updates` reads back both sides. The
per-level round-trip moved from "HistogramBundle flat-copy" to
"row-index full-copy × 3 sites" — roughly the same bandwidth at
`metal_friendly` shape and strictly more at 1M-row shapes.
Consistent with the measured numbers. See the Stage 3 section in
`BENCHMARKS.md` for the full diagnosis and the three candidate
follow-ups (GPU count accumulation in the histogram kernel; GPU
reduce_sums requiring a gradient pool; GPU apply_partition_leaf_
updates requiring a prediction pool).
Alternatives considered: ship Stage 3 as infrastructural and
advance to Stage 4 (rejected — the approved plan's kill criterion
is explicit: "we stop to debug rather than ship a second
infrastructure-only stage"; ICBs are marginal on top of a loop
that's still round-tripping); revert Stage 3 pool work (rejected —
the pools and variants are correct and the subtract path is
net-positive; the right next step is to close the three readback
paths, not revert).

---

## D-021: Stage 3 bottleneck is `build_histograms.gpu_dispatch`, not readback

Date: 2026-04-24
Stage: 3 (post-S3.12 diagnosis)
Decision: D-020's readback-closure hypothesis is wrong. Rust-level
profiling with `ALLOYGBM_METAL_PROFILE=1` (see `crates/backend_metal/
src/profile.rs`) shows that on the `metal_friendly` regression /
depth=10 config, `build_histograms.gpu_dispatch` is 6.58s of an
8.39s total (78.5% of whole-fit time, ~88% of `build_histograms`),
while every combined readback path — `build_histograms.row_readback`
+ `reduce_sums.readback` + `apply_partition_leaf_updates.readback` —
sums to ~12ms (0.15% of total). `.gpu_dispatch` averages 45ms per
`build_histograms` call across 145 calls. The bottleneck is the
GPU work itself (allocation + encode + commit + wait), not the
host round-trip.
Why: The profile module wraps each overridden `BackendOps` method
with a scoped probe, with sub-phase probes inside
`build_histograms`, `reduce_sums`, and `apply_partition_leaf_
updates`. `reduce_sums` and `apply_partition_leaf_updates`
combined are 0.7% of top-level time — optimising them cannot
move the kill-criterion needle by any meaningful fraction. The
candidate follow-ups listed in D-020 (GPU counts kernel, GPU
reduce_sums, GPU leaf_update) would have bought ~0.3% speedup
each. We need a different fix.
Three suspected sub-causes inside `.gpu_dispatch`, to be
confirmed by a second profiling pass with finer probes:
(a) `threads_per_tg = 32` ([histogram.rs:283-287]) — 1 simdgroup
per threadgroup, severely under-occupying M4 GPU cores that want
128–256 threads/TG; (b) scratch buffer allocated via
`newBufferWithLength_options` per-tile per-call inside the
dispatch loop ([histogram.rs:238-242]), bypassing `BufferCache`;
(c) synchronous per-call `commit`+`waitUntilCompleted` with ~100+
calls per fit preventing GPU pipelining.
Alternatives considered: trust the original diagnosis and
implement a GPU counts kernel (rejected — measurement shows it
fixes 0.3% of the problem); skip profiling and try threadgroup-
size bump blind (rejected — fast but cheaper to measure first
given we've already been burned once by an unmeasured
hypothesis); spawn finer probes via Instruments.app GPU trace
(deferred — the existing in-process probes are sufficient to
localise the cause inside `.gpu_dispatch` without adding an
external-tool dependency; can be added if finer probes don't
converge).

Second pass (finer sub-probes inside `.gpu_dispatch`, depth=8
`metal_friendly` regression):

  build_histograms                            10.47s   90.2% of total
    .gpu_dispatch                              9.23s   88% of build_histograms
      ..scratch_alloc (528 calls)              0.12s    1.3%
      ..encode        (528 calls)              0.06s    0.7%
      ..commit_wait   (528 calls)              9.04s   97.9%

`.commit_wait` (the `command_buffer.commit()` +
`waitUntilCompleted()` block) averages **17ms per call**.
Scratch allocation and encoding are microscopic. The GPU work
itself dominates — the bottleneck is the kernel, not the
dispatch overhead.

Root cause in `shaders/histogram.metal`: `SIMD_WIDTH = 32`
threads per threadgroup (1 simdgroup), inner loop at line 148
serialises over 32 source lanes via `simd_shuffle` with a
per-bin ownership check (line 154: `(src_bin % SIMD_WIDTH) ==
thread_in_tg`). For uniformly-distributed bins, ~1 in 32 lanes
commits a write per shuffle step; the rest idle. Effective
arithmetic throughput is ~1/32 of theoretical peak. This is a
determinism workaround for the "no float atomics at any memory
level" constraint (D-003), and it pays a ~30× compute cost.
Under-occupancy compounds the problem: M4 cores can host many
more simdgroups per core but the kernel only ships 1.
