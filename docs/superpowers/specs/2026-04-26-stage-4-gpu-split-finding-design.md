# Stage 4 — GPU Split Finding + ICB Chaining

**Date:** 2026-04-26
**Stage:** Metal Backend Stage 4 — close the kill-criterion gap
**Scope:** Two coupled architectural changes shipped in two phases, both within
the existing `charming-carson-d08c9a` worktree, single merge at the end.

## Motivation

Stage 3's Approach A batch path landed correctly and is fully exercised
(D-023 amendment). Best `metal_friendly` ratio post-fix: 0.16× CPU
(regression d=10, regression d=6 bins=1024). The kill criterion
`metal_friendly >1.0× CPU on at least one config` was NOT MET.

The post-fix profile dump (depth=8 regression, Apple M4) attributes
2997 ms of 3966 ms total Metal time (75.6%) to `build_histograms_batch`,
inside which `commit_wait` is 2634 ms. Even though Approach A
collapsed 528 → 40 commits per fit, every level still requires a
synchronous `waitUntilCompleted` because the host needs the histogram
data to run `best_split_with_options`.

Eliminating that stall requires moving split finding onto the GPU so
the host no longer reads histograms back per level. ICB chaining on
top then collapses per-level commit overhead into per-tree commit
overhead. Stage 4 is both changes.

## Non-Goals

- **Categorical Fisher-sort on GPU.** No major library does this on GPU
  (LightGBM falls back to CPU; XGBoost uses one-hot; CatBoost uses
  ordered target stats). Adds determinism risk for marginal gain. We
  use the per-feature mixed-mode pattern instead — GPU handles numeric,
  host handles categorical, merge per node.
- **Monotonicity / interaction constraints / custom objectives on GPU.**
  Models with these constraints fall through to the Stage 3 path via
  the eligibility predicate.
- **Leaf-wise tree growth on GPU split finding.** `build_tree_leaf_wise`
  uses the scalar default impl of the new trait method (i.e. host-side
  split finding) — same handling as Stage 3's batched methods.
- **Stage 4b ICB chaining detail.** This spec defines 4b at a high
  level only. We design 4b properly only if 4a's benchmark gate fails.

## Architecture

Stage 4 ships in two phases inside one worktree, with a benchmark gate
between them:

- **Stage 4a — GPU split finding.** New Metal compute kernel +
  `BackendOps::find_best_splits_batch` trait method. Mixed-mode
  numeric/categorical handling.
- **Stage 4b — Metal 4 ICB chaining.** *Conditional.* Layered on 4a
  if the post-4a benchmark still misses the kill criterion.

Both phases commit incrementally. The single merge to `main` happens
after Stage 4 closes (whichever phase ends it).

## Stage 4a — GPU Split Finding

### Why this is the architecturally important change

Stage 4a removes two costs at once:

1. **Histogram readback.** Today's `count_accumulate` (492 ms, ~12% of
   Metal time at depth=8) materialises the device-side histogram counts
   onto the host so split finding can read them. With GPU split
   finding the kernel reads counts directly from the residency pool —
   the host never sees histograms.
2. **Most of `commit_wait`.** The per-level wait still happens (until
   4b ships), but the wait now drains a small kernel pipeline (~5–15 ms
   per level expected) instead of waiting for histogram readback to
   complete. Expected per-level latency drops from ~65 ms to single
   digits.

The data crossing the bus per level shrinks from
`O(nodes × features × bins × 8 bytes)` (full histograms) to
`O(nodes × 24 bytes)` (split decisions only).

### Data flow per level (revised `build_tree_level_wise`)

The level loop changes from a 5-step shape to a 7-step shape:

1. **Per-node prep** — unchanged from Stage 3. Decide build-vs-subtract,
   allocate destination bundles, choose smaller/larger child.
2. **`build_histograms_batch`** — unchanged, Stage 3. One CB.
3. **`subtract_histogram_bundle_batch`** — unchanged, Stage 3. One CB.
4. **`find_best_splits_batch` (NEW).** One CB encoding (per-feature
   kernel + cross-feature reduce) for every active node's histogram
   bundle. Numeric features only. Single commit + wait.
5. **Decision readback (NEW).** Read N×24-byte `SplitDecision` array
   from device-side output buffer into a host `Vec<SplitDecision>`.
   Tiny — at most a few KB per level.
6. **Categorical host merge (NEW, conditional).** If the model has
   categorical features, run `best_split_with_options` per active node
   restricted to the categorical feature subset. Per-node merge: pick
   the better of GPU-numeric-best vs host-categorical-best. Skipped
   entirely when no categorical features exist.
7. **Apply-split + leaf decisions** — mostly unchanged from Stage 3.
   For each node, check `gain >= min_split_gain`. If yes, call existing
   `apply_split_with_stats` (already GPU-side via Stage 3). If no, leaf
   with Newton-step value. Driven from the new `SplitDecision` array
   instead of the host-computed one.

Steps 2 + 3 + 4 are three separate command buffers per level. Stage 4b
collapses 2 + 3 + 4 + 7 into one per-tree ICB. For 4a we keep them
separate.

### Components

**New files:**

- `crates/backend_metal/src/shaders/best_split.metal` — MSL kernel.
  Two entry points: `best_split_per_feature` (per-(node,feature)
  threadgroup) and `best_split_reduce_features` (per-node threadgroup).
- `crates/backend_metal/src/kernels/best_split.rs` — Rust dispatch
  wrapper, mirrors the shape of `kernels/histogram.rs`.

**Modified files:**

- `crates/engine/src/lib.rs` — new `BackendOps::find_best_splits_batch`
  trait method with scalar default that delegates to per-node
  `best_split_with_options`. `build_tree_level_wise` switches to call
  it after `build_histograms_batch` + `subtract_histogram_bundle_batch`.
- `crates/backend_metal/src/lib.rs` —
  `MetalBackend::find_best_splits_batch` override, gated on
  `is_eligible_for_gpu_split_finding(params, schema)`.
- `crates/backend_metal/src/profile.rs` — new counters:
  `FIND_BEST_SPLITS_BATCH`, `BS_DISPATCH`, `BS_COMMIT_WAIT`,
  `BS_DECISION_READBACK`, `BS_CATEGORICAL_HOST_MERGE`.
- `bindings/python/src/runtime_backend.rs` — forwarding arm for the
  new trait method (avoid the Task-8 forwarding-gap mistake).
- `crates/backend_metal/src/pipelines.rs` — pipeline cache entry for
  the two best-split kernel entry points.

### Kernel design

**Per-feature kernel (`best_split_per_feature`):**

- One threadgroup per (active_node, numeric_feature) pair. Threadgroup
  size = bin count, padded to a multiple of 32 (SIMD width).
- Each thread owns one bin's `(grad, hess, count)`.
- Per-feature work inside one threadgroup:
  1. **Load.** Each thread reads its bin's `(grad, hess, count)` from
     the histogram buffer in the residency pool.
  2. **Prefix sum.** Two-pass deterministic prefix scan (same pattern
     as the histogram kernel — `simd_prefix_inclusive_sum` +
     block-scan). Computes left-side cumulative
     `(grad_left, hess_left, count_left)` at each bin boundary.
  3. **Total subtraction.** Right-side
     `(grad_right, hess_right, count_right) = (totals - left)` —
     totals come from the bundle's per-feature totals slot.
  4. **Per-bin gain.** Newton-step formula:
     `gain = grad_left²/(hess_left + λ) + grad_right²/(hess_right + λ) - grad_total²/(hess_total + λ)`.
  5. **Constraint mask.** Each thread sets `gain = -∞` if its split
     violates `min_rows_per_leaf` or `min_child_hessian`.
     `min_split_gain` is checked host-side after readback so the
     kernel doesn't need it.
  6. **Argmax reduction.** Threadgroup reduces to (best_bin,
     best_gain) — deterministic two-pass: SIMD max → threadgroup
     memory → thread 0 final reduce. Tie-break: lower bin index wins.
  7. **Missing-direction decision.** Run gain math twice (missing-left
     vs missing-right) — keep the better. Encoded into `flags`.
  8. **Write.** Thread 0 writes the per-(node, feature) result to a
     scratch buffer.

**Cross-feature reduction kernel (`best_split_reduce_features`):**

- One threadgroup per active node. Reads `numeric_feature_count`
  per-feature results from the scratch buffer, emits the final
  `SplitDecision`. Tie-break: lower feature index wins on equal gain.
- Tiny — well under 1% of split-finding work — but kept as a separate
  pass to keep the inner kernel simpler.

**Determinism:**

- All reductions are two-pass with fixed thread-to-bin mapping. No
  atomics on floats (matches D-003).
- Tie-breaking rules are explicit in code: lower bin index inside a
  feature, lower feature index across features. Both rules match the
  CPU path's behaviour — verify in `crates/backend_cpu/src/lib.rs`
  before kernel implementation.
- Multiclass: one tree per class is fit serially — kernel is invoked
  once per class tree per level. No multiclass-aware kernel logic.

### Output buffer

One `SplitDecision` per active node (~24 bytes packed):

```rust
#[repr(C)]
struct SplitDecision {
    feature_idx: u32,        // 0xFFFFFFFF if no valid split found
    bin_threshold: u32,      // bin index (not value)
    gain: f32,
    grad_left: f32,
    hess_left: f32,
    flags: u32,              // bit 0: missing-goes-right
                             // bit 1: invalid (no split met constraints)
}
```

Output buffer is minted from a new sibling `SplitDecisionPool` (parallel
to `HistogramResidencyPool`). Sibling rather than shared because the
lifecycle differs — histograms live one level for build but multi-level
for parent-of-subtract; split decisions live exactly one level. A
dedicated pool keeps the lifecycle invariant clear and avoids forcing
the histogram pool to mint variable-sized entries. Freed via a
`SplitDecisionReleaseGuard` mirroring the existing
`HistogramReleaseGuard` RAII pattern.

### Eligibility predicate

```rust
fn is_eligible_for_gpu_split_finding(
    params: &TrainParams,
    schema: &FeatureSchema,
) -> bool {
    // Categorical-only models have nothing for the GPU kernel to do.
    // Mixed numeric/categorical models go through the per-node merge
    // path; categoricals don't disqualify.
    if schema.numeric_feature_indices().is_empty() { return false; }

    // Constraints not yet supported kernel-side.
    if params.monotone_constraints.is_some() { return false; }
    if params.interaction_constraints.is_some() { return false; }
    if params.has_custom_objective() { return false; }

    // Metal 3 path is unchanged in Stage 4a — best-split is a normal
    // compute kernel and works on Metal 3 too. capabilities.metal4
    // only gates Stage 4b.
    true
}
```

If predicate returns false, `MetalBackend::find_best_splits_batch`
calls the trait's scalar default which delegates to per-node
`best_split_with_options`. The whole model takes the Stage 3 path.

### Error handling

- **Kernel pipeline compile failure at backend init.** Log warning,
  set `gpu_split_finding_disabled = true` on the backend, route
  through scalar default for the rest of the process. Same fallback
  pattern Stage 3's eligibility checks use.
- **Per-call command-buffer failure (device lost, OOM).** Return
  `BackendError::CommandBuffer`. Engine treats as fatal — matches
  current Stage 3 behaviour for batch failures.
- **Per-node "no valid split found"** (all bin gains failed
  `min_rows_per_leaf`). Kernel writes
  `feature_idx = u32::MAX, flags |= INVALID`. Host reads this and
  emits a leaf node — same outcome as the CPU path returning
  `Ok(None)` from `best_split_with_options`.

## Determinism

Bit-exactness is preserved by construction:

- Per-feature reductions are two-pass deterministic — no float atomics,
  fixed thread-to-bin mapping, no race conditions.
- Cross-feature reduction is deterministic — fixed tie-break rule that
  matches the CPU path.
- Categorical merge is host-side using the unchanged
  `best_split_with_options` path; no determinism risk introduced.
- The Newton-step gain formula uses the same regularisation parameters
  the CPU path uses, computed in the same order.

Existing parity tests (S3.10 residency round-trip, S3.11 Python
parity) continue to gate Stage 4a.

## Testing

**New parity tests** in `crates/backend_metal/tests/best_split_parity.rs`:

1. `find_best_splits_batch_matches_scalar_all_numeric` — small fixture,
   3 nodes, all-numeric features, byte-equal `SplitDecision` per node.
2. `find_best_splits_batch_matches_scalar_mixed_numeric_categorical` —
   verifies the per-node merge step produces identical decisions to
   the all-host path.
3. `find_best_splits_batch_falls_back_on_monotonicity` — model with
   monotonicity constraint takes Stage 3 path (verifies eligibility
   predicate).
4. `find_best_splits_batch_invalid_split_emits_leaf` — fixture
   constructed so no split satisfies `min_rows_per_leaf`; assert
   resulting tree has only the root as a leaf.

**Existing tests that must continue to pass:**

- `cargo test -p alloygbm-backend-metal -- --test-threads=1` —
  expected 51+ tests passing (47 today + 4 new).
- `cargo test --workspace --exclude alloygbm-python` — green.
- `pytest bindings/python/tests/ -q` — 365/365.
- `metal_python_parity_*` (S3.11) — bit-exact match against CPU.

**Determinism test** — same seed, same data, run 5× and assert
byte-identical model artifact. Already part of the existing suite;
must keep passing.

**Benchmark runs after Stage 4a lands:**

- `metal_friendly` re-run (5 configs, same as Stage 3).
- New `metal_friendly_large` config (1M rows × 100 features × bins=255)
  matching the original plan's "decisive 4-5× win above 1M×100"
  prediction. Add to `benchmarks/metal_histogram.py`.

## Stage 4a Acceptance Gate

After 4a lands:

1. `cargo test --workspace --exclude alloygbm-python` green.
2. `pytest bindings/python/tests/ -q` — 365/365 green.
3. `metal_friendly` ratios captured + documented in D-024.
4. `metal_friendly_large` (1M×100) ratio captured + documented in
   D-024.
5. **If `metal_friendly_large` crosses >1.0× CPU on at least one
   config:** Stage 4 is done. Skip 4b. Update `docs/limitations.md`
   Section 1 with the documented Metal-vs-CPU threshold. Stage 4
   merges to `main`.
6. **If neither config crosses >1.0× CPU:** D-024 records the
   residual cost breakdown and Stage 4b kicks off (see below).

## Stage 4b — High-Level Sketch (Conditional)

**Goal:** collapse the per-level commit/wait into per-tree commit/wait
via Metal 4 Indirect Command Buffer chaining.

**Approach:** `MTL4CommandAllocator` encodes
(build → subtract → find_split → apply_split) for all levels of a tree
into one ICB. Indirect dispatch arguments pull dispatch sizes from
device-side buffers written by the previous kernel — the host never
reads back between levels, only at end-of-tree. Submit one ICB per
class tree per round.

**Open architectural questions** (deferred until 4a benchmarks come in):

- **Variable-depth tree handling.** Trees terminate early when no
  active node beats `min_split_gain`. Two patterns to handle this in
  an ICB: zero-dispatch on inactive nodes (kernel does nothing for
  zero threadgroups), or worst-case allocation with a "skip" flag
  buffer the kernels respect. Picked during the 4b spec.
- **Buffer reuse across levels.** Level N+1 input is level N output.
  Need pool entries that persist for the full ICB lifetime, freed at
  end-of-tree. Likely requires a per-tree pool scope on top of the
  existing `HistogramResidencyPool`.
- **Indirect-dispatch encoding overhead.** Metal 4 ICBs are cheap to
  record but the dispatch-args buffer needs writes from the split
  kernel. The split kernel may need a small "epilogue" pass to write
  the next level's dispatch args.
- **Capability gating.** ICB path requires Metal 4 (macOS 26+); falls
  back to per-level submission (the 4a path) on Metal 3.

**4b acceptance gate** — same as 4a, plus the new `metal_friendly_large`
config crosses >1.0× CPU on at least one config. If even 4b doesn't
cross, document the residual gap in `D-025` and accept Stage 4 as
"best effort" (matches the original plan's contingency).

## Risks

- **Determinism regression in the per-feature reduction.** The Newton
  gain formula is sensitive to floating-point reduction order. The
  two-pass prefix scan + two-pass argmax reduction must match the
  CPU's serial scan exactly. Mitigation: parity tests run at every
  step; small-fixture byte-equal assertions catch divergence early.
- **Tie-breaking rule drift between GPU kernel and host categorical
  merge.** If the kernel uses one tie-break rule and `merge_with_host`
  uses another, mixed-feature models will diverge from the all-host
  baseline. Mitigation: the merge step calls `best_of(gpu, host)`
  using the SAME comparator function the CPU path uses today —
  factor it out before kernel implementation.
- **Eligibility-predicate gaps.** A new training feature lands that
  the kernel doesn't support but the predicate doesn't gate, and
  models silently take the GPU path with wrong results. Mitigation:
  the predicate returns false for any param the kernel doesn't model;
  add a regression test for each new training feature going forward.
- **`metal_friendly_large` doesn't cross even after 4b.** The
  contingency from the original plan applies — Stage 4 ships as
  "best effort" and `docs/limitations.md` documents the threshold
  where Metal wins (which may simply be "none of the tested
  configurations").

## Out-of-Scope (Deferred)

- GPU-side `apply_partition_leaf_updates` (current cost: 0.2% of
  total — not worth touching).
- GPU-side `reduce_sums` (0.3% of total — same reason).
- Metal 4 pipeline harvesting (originally part of Stage 3; deferred —
  not on the critical path for the kill criterion).
- Stage 5 (GPU inference / tree traversal) — separate roadmap entry.
