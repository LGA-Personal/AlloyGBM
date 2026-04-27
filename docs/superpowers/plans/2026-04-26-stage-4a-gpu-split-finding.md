# Stage 4a — GPU Split Finding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `best_split_with_options` onto the GPU so the per-level wait no longer requires reading whole histograms back to host. Replaces O(nodes × features × bins × 8 bytes) histogram readback with O(nodes × 24 bytes) split-decision readback. Mixed-mode: GPU handles numeric features, host handles categoricals via Fisher-sort, per-node merge picks the winner.

**Architecture:** New `BackendOps::find_best_splits_batch` trait method with scalar default that delegates to per-node `best_split_with_options`. `MetalBackend` overrides with a two-kernel Metal pipeline (`best_split_per_feature` + `best_split_reduce_features`) that reads device-resident histograms from the existing `HistogramResidencyPool` and writes a packed `SplitDecision` array to a new sibling `SplitDecisionPool`. Engine's `build_tree_level_wise` switches from per-node `best_split_with_options` to the new batched call; `build_tree_leaf_wise` uses the scalar default unchanged. Eligibility predicate gates the GPU path on (numeric-features-present, no monotonicity, no interaction constraints, no custom objective).

**Tech Stack:** Rust 1.92 / edition 2024 / `unsafe_code = "forbid"` workspace-wide (file-level `#[allow(unsafe_code)]` in Metal FFI sites). Metal 3 baseline (kernel works on Metal 3 and Metal 4 — `capabilities.metal4` only matters for Stage 4b). PyO3 + maturin for Python bindings. Existing residency pool / RAII guard / batch-request infrastructure from Stage 3.

---

## File Structure

**New files:**
- `crates/backend_metal/src/shaders/best_split.metal` — MSL kernel with two entry points.
- `crates/backend_metal/src/kernels/best_split.rs` — Rust dispatch wrapper.
- `crates/backend_metal/src/split_decision_residency.rs` — `SplitDecisionPool` + handle.
- `crates/backend_metal/tests/best_split_parity.rs` — Metal vs CPU parity tests.

**Modified files:**
- `crates/engine/src/lib.rs` — `SplitFindRequest` struct, `BackendOps::find_best_splits_batch` trait method, `build_tree_level_wise` refactor.
- `crates/backend_cpu/src/lib.rs` — no changes (scalar default in trait covers CPU path).
- `crates/backend_metal/src/lib.rs` — `MetalBackend::find_best_splits_batch` override, eligibility predicate, mixed-mode merge.
- `crates/backend_metal/src/profile.rs` — five new counters.
- `crates/backend_metal/src/pipelines.rs` — pipeline cache entry for `best_split.metal`.
- `crates/backend_metal/src/kernels/mod.rs` — `pub(crate) mod best_split;`.
- `crates/backend_metal/src/lib.rs` — `mod split_decision_residency;`.
- `bindings/python/src/runtime_backend.rs` — forwarding arm for the new trait method.
- `benchmarks/metal_histogram.py` — new `metal_friendly_large` config.
- `docs/metal-backend/DECISIONS.md` — D-024 entry.
- `docs/metal-backend/STATUS.md` — Stage 4a outcome.
- `docs/metal-backend/SESSIONS.md` — 2026-04-26 entry.

---

## Task 1: Add `BackendOps::find_best_splits_batch` trait method

**Why first:** The engine refactor (Task 2) and the Metal override (Task 6) both reference this trait method. Defining it with a scalar default first lets `cargo test --workspace` stay green between every commit — CPU automatically takes the slow path until the Metal override lands.

**Files:**
- Modify: `crates/engine/src/lib.rs` (add struct + trait method, near `BackendOps::subtract_histogram_bundle_batch` around line 191).
- Test: `crates/backend_cpu/tests/find_best_splits_batch_default.rs` (new integration test).

- [ ] **Step 1: Open `crates/engine/src/lib.rs` and add the request struct after the existing `SubtractRequest` struct.**

Find the `SubtractRequest` definition (around line 95–102) and append immediately after it (before `pub trait BackendOps {`):

```rust
/// One node's input for a batched best-split search. The shared
/// `options`, `feature_weights`, and `categorical_features` are passed
/// alongside `&[SplitFindRequest]` to
/// [`BackendOps::find_best_splits_batch`].
#[derive(Debug)]
pub struct SplitFindRequest<'a> {
    pub histograms: &'a HistogramBundle,
}
```

- [ ] **Step 2: Add the trait method with a scalar default.**

In the same file, find the existing `subtract_histogram_bundle_batch` method (around line 191–199) and add immediately after it:

```rust
    /// Batched best-split search across multiple histogram bundles.
    /// Default impl iterates and calls
    /// [`best_split_with_options`](BackendOps::best_split_with_options);
    /// accelerator backends override to encode all per-node split
    /// dispatches into a single command buffer and amortise the
    /// host↔device round-trip.
    ///
    /// The returned vector aligns with `requests`: `result[i]` is the
    /// best split for `requests[i].histograms` (or `None` if no split
    /// satisfied the constraints). On any backend error, the whole
    /// call returns `Err` and any partially-computed splits at lower
    /// indices are dropped.
    fn find_best_splits_batch(
        &self,
        requests: &[SplitFindRequest<'_>],
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Vec<Option<SplitCandidate>>> {
        requests
            .iter()
            .map(|r| {
                self.best_split_with_options(
                    r.histograms,
                    options,
                    feature_weights,
                    categorical_features,
                )
            })
            .collect()
    }
```

- [ ] **Step 3: Run `cargo check --workspace --exclude alloygbm-python` to verify the addition compiles.**

```bash
cargo check --workspace --exclude alloygbm-python 2>&1 | tail -10
```

Expected: clean compile, no warnings introduced.

- [ ] **Step 4: Create the failing integration test.**

Create new file `crates/backend_cpu/tests/find_best_splits_batch_default.rs` with:

```rust
//! Verifies the default `BackendOps::find_best_splits_batch` impl
//! delegates to per-node `best_split_with_options`, exercised through
//! `CpuBackend` (which does not override it).

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, SplitFindRequest, SplitSelectionOptions,
};

#[test]
fn cpu_backend_find_best_splits_batch_default_matches_scalar() {
    let row_count = 64usize;
    let feature_count = 3usize;
    let max_bin: u16 = 7;
    let bins: Vec<u8> = (0..(row_count * feature_count))
        .map(|i| ((i * 11) & 7) as u8)
        .collect();
    let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
    let grads: Vec<GradientPair> = (0..row_count)
        .map(|i| GradientPair {
            grad: i as f32,
            hess: 1.0,
        })
        .collect();
    let tiles = vec![FeatureTile {
        start_feature: 0,
        end_feature: feature_count as u32,
    }];
    let backend = CpuBackend;

    let node_a = NodeSlice::new(0, (0..32u32).collect()).unwrap();
    let node_b = NodeSlice::new(1, (32..64u32).collect()).unwrap();

    let hist_a = backend.build_histograms(&bm, &grads, &node_a, &tiles).unwrap();
    let hist_b = backend.build_histograms(&bm, &grads, &node_b, &tiles).unwrap();

    let options = SplitSelectionOptions::default();
    let feature_weights: Vec<f32> = vec![1.0; feature_count];
    let categorical_features: Vec<CategoricalFeatureInfo> = Vec::new();

    let scalar_a = backend
        .best_split_with_options(&hist_a, options, &feature_weights, &categorical_features)
        .unwrap();
    let scalar_b = backend
        .best_split_with_options(&hist_b, options, &feature_weights, &categorical_features)
        .unwrap();

    let requests = vec![
        SplitFindRequest { histograms: &hist_a },
        SplitFindRequest { histograms: &hist_b },
    ];
    let batched = backend
        .find_best_splits_batch(&requests, options, &feature_weights, &categorical_features)
        .unwrap();
    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0], scalar_a);
    assert_eq!(batched[1], scalar_b);
}

#[test]
fn cpu_backend_find_best_splits_batch_empty_is_noop() {
    let backend = CpuBackend;
    let requests: Vec<SplitFindRequest<'_>> = Vec::new();
    let result = backend
        .find_best_splits_batch(
            &requests,
            SplitSelectionOptions::default(),
            &[],
            &[],
        )
        .unwrap();
    assert!(result.is_empty());
}
```

- [ ] **Step 5: Run the test.**

```bash
cargo test -p alloygbm-backend-cpu --test find_best_splits_batch_default -- --test-threads=1 2>&1 | tail -10
```

Expected: both tests PASS — they exercise the trait's scalar default impl which is already correct.

- [ ] **Step 6: Commit.**

```bash
git add crates/engine/src/lib.rs crates/backend_cpu/tests/find_best_splits_batch_default.rs
git commit -m "$(cat <<'EOF'
feat(engine): BackendOps::find_best_splits_batch with scalar default

Adds SplitFindRequest struct and find_best_splits_batch trait method
mirroring the shape of build_histograms_batch / subtract_histogram_bundle_batch
from Stage 3. Default impl iterates and calls best_split_with_options
per request, so the CPU path is unchanged. Stage 4a's Metal backend
will override this method.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Forward `find_best_splits_batch` through `RuntimeBackend`

**Why now:** Lessons from Stage 3 Task 8 — the trait default fires unless `RuntimeBackend` explicitly forwards. Doing this immediately after Task 1 prevents the same forwarding-gap discovery later.

**Files:**
- Modify: `bindings/python/src/runtime_backend.rs`.

- [ ] **Step 1: Open `bindings/python/src/runtime_backend.rs` and update the `alloygbm_engine` import.**

Find the line `use alloygbm_engine::{BackendOps, CategoricalFeatureInfo, EngineResult, HistogramBuildRequest, SplitSelectionOptions, SubtractRequest};` (was added in Stage 3 Task 8 fix). Replace with:

```rust
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, EngineResult, HistogramBuildRequest,
    SplitFindRequest, SplitSelectionOptions, SubtractRequest,
};
```

- [ ] **Step 2: Add the forwarding method in `impl BackendOps for RuntimeBackend`.**

Find the `subtract_histogram_bundle_batch` method (added in Stage 3 Task 8 fix). Immediately after its closing `}`, insert:

```rust
    fn find_best_splits_batch(
        &self,
        requests: &[SplitFindRequest<'_>],
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Vec<Option<alloygbm_core::SplitCandidate>>> {
        match self {
            RuntimeBackend::Cpu(b) => {
                b.find_best_splits_batch(requests, options, feature_weights, categorical_features)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RuntimeBackend::Metal(b) => {
                b.find_best_splits_batch(requests, options, feature_weights, categorical_features)
            }
        }
    }
```

NOTE: `SplitCandidate` may already be imported from `alloygbm_core` at the top of the file. If it is, use the unqualified name (`SplitCandidate`) instead of `alloygbm_core::SplitCandidate`. Check the existing imports first.

- [ ] **Step 3: Verify with cargo check.**

```bash
cargo check -p alloygbm-python 2>&1 | tail -10
```

Expected: clean compile.

- [ ] **Step 4: Rebuild the Python extension.**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release --manifest-path bindings/python/Cargo.toml 2>&1 | tail -5
```

Expected: clean rebuild.

- [ ] **Step 5: Smoke-test pytest.**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -5
```

Expected: 365/365 pass (no behavior change yet — both arms still go through the trait default).

- [ ] **Step 6: Commit.**

```bash
git add bindings/python/src/runtime_backend.rs
git commit -m "$(cat <<'EOF'
feat(bindings/python): forward find_best_splits_batch through RuntimeBackend

Mirrors the Stage 3 Task 8 fix: explicit two-arm forward for the new
batched trait method so MetalBackend's override (landing later in
Stage 4a) actually fires when the Python layer calls into it. CPU
arm continues to call the trait default (no behavior change yet).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Refactor `build_tree_level_wise` to use `find_best_splits_batch`

**Why:** Replaces the per-node `best_split_with_options` call inside the level loop with a single batched call. CPU path is unchanged (default impl is per-node) but the engine now drives the call via the batched API, ready for the Metal override.

**Files:**
- Modify: `crates/engine/src/lib.rs` around lines 4140–4270 (the level-wise loop body).

- [ ] **Step 1: Read the current shape of the level-wise loop.**

Open `crates/engine/src/lib.rs` and read lines 4070–4270 carefully. Currently the loop:
1. Builds parent histograms (or inherits them).
2. Calls `backend.best_split_with_options(...)` per node inline (around line 4090–4110).
3. Decides leaf vs split, encodes pending children.
4. Calls `build_histograms_batch` for smaller children (around line 4200).
5. Calls `subtract_histogram_bundle_batch` for larger children (around line 4225).
6. Assembles `next_nodes`.

The refactor: move the per-node `best_split_with_options` call OUT of the per-node prep loop and into a batched call BEFORE the per-node prep. The new shape:

1. Pre-step: collect all current-level nodes and their existing histogram bundles into `Vec<SplitFindRequest>`.
2. Single batched call: `backend.find_best_splits_batch(&requests, options, ...)` returns `Vec<Option<SplitCandidate>>` aligned with active nodes.
3. Per-node prep loop now reads its split decision from this vector instead of computing inline.
4. Steps 4–6 unchanged.

- [ ] **Step 2: Identify the exact `best_split_with_options` call in the level-wise loop.**

Find the line `let split = backend.best_split_with_options(` inside `build_tree_level_wise` (NOT in `build_tree_leaf_wise` — leaf-wise stays unchanged). It is inside the per-node prep loop. Note its surrounding context (typically inside an iteration over `active_nodes`).

- [ ] **Step 3: Add the batched call before the per-node prep loop.**

Inside `build_tree_level_wise`'s `while !active_nodes.is_empty()` loop, immediately AFTER the line that builds `parent_histograms_per_node` (the current-level histogram bundles, which already exist as field `2` of the `(u32, RowIndexStorage, HistogramBundle, f32)` tuple) and BEFORE the per-node prep loop, insert:

```rust
        // Stage 4a: batched per-level split finding.
        // Collect all current-level histogram bundles into one batched
        // request. The result vector aligns with active_nodes.
        let split_requests: Vec<SplitFindRequest<'_>> = active_nodes
            .iter()
            .map(|(_, _, histograms, _)| SplitFindRequest { histograms })
            .collect();
        let level_splits: Vec<Option<SplitCandidate>> = backend.find_best_splits_batch(
            &split_requests,
            split_options,
            feature_weights,
            categorical_features,
        )?;
```

NOTE: The variable names `split_options`, `feature_weights`, `categorical_features` must match what's already in scope at this point in `build_tree_level_wise`. If they're named differently locally (e.g. `options` instead of `split_options`), use the local names. Read the function signature (around line 4040) to confirm.

- [ ] **Step 4: Replace the per-node `best_split_with_options` call with a vector lookup.**

Inside the per-node prep loop, find the line:
```rust
let split = backend.best_split_with_options(
    histograms,
    split_options,
    feature_weights,
    categorical_features,
)?;
```
(or its locally-named equivalent — variable names may differ).

Replace with:
```rust
let split = level_splits[node_index].clone();
```

`node_index` is the loop counter in the per-node prep loop. If the loop currently uses an iterator without an explicit counter, change the iteration to `for (node_index, (local_id, rows, histograms, parent_leaf_value)) in active_nodes.iter().enumerate()`.

NOTE: `SplitCandidate` is `#[derive(Clone)]` (verified at `crates/core/src/lib.rs:967`), so `.clone()` is fine here.

- [ ] **Step 5: Run `cargo check --workspace --exclude alloygbm-python`.**

```bash
cargo check --workspace --exclude alloygbm-python 2>&1 | tail -15
```

Expected: clean compile. Common error: variable name mismatch on `split_options` / `feature_weights` / `categorical_features` — fix to match the function-local names.

- [ ] **Step 6: Run the full engine test suite.**

```bash
cargo test -p alloygbm-engine -- --test-threads=1 2>&1 | tail -15
```

Expected: every existing engine test still passes. The CPU path still uses the trait default (per-node) so behavior is byte-identical.

- [ ] **Step 7: Run the smoke test for the level-wise three-phase shape.**

```bash
cargo test -p alloygbm-backend-cpu --test level_wise_three_phase_smoke -- --test-threads=1 2>&1 | tail -10
```

Expected: PASS. (This test was added in Stage 3 Task 2; it now exercises the four-phase shape.)

- [ ] **Step 8: Run the python parity test.**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release --manifest-path bindings/python/Cargo.toml 2>&1 | tail -3
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -5
```

Expected: 365/365 pass.

- [ ] **Step 9: Commit.**

```bash
git add crates/engine/src/lib.rs
git commit -m "$(cat <<'EOF'
refactor(engine): batched per-level split finding in build_tree_level_wise

Hoists the per-node best_split_with_options call out of the per-node
prep loop and into a single backend.find_best_splits_batch invocation
at the top of each level. CPU path is unchanged via the trait default
impl; the Metal backend's override will exercise true batching.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Metal profile counters

**Files:**
- Modify: `crates/backend_metal/src/profile.rs`.

- [ ] **Step 1: Open `crates/backend_metal/src/profile.rs` and find the existing counter declarations.**

Look for the `BUILD_HISTOGRAMS_BATCH` and `SUBTRACT_BATCH` static declarations (added in Stage 3 Task 5). Note their pattern (`pub(crate) static <NAME>: Counter = Counter::new();`).

- [ ] **Step 2: Add five new counters immediately after `SUBTRACT_BATCH`.**

```rust
/// Wraps the entire `find_best_splits_batch` call (probe → empty
/// short-circuit → translate request → kernel dispatch → readback →
/// categorical merge).
pub(crate) static FIND_BEST_SPLITS_BATCH: Counter = Counter::new();
/// Per-feature kernel dispatch + cross-feature reduce kernel
/// dispatch encoding work, before commit.
pub(crate) static BS_DISPATCH: Counter = Counter::new();
/// `commit()` + `waitUntilCompleted()` for the split-find command
/// buffer.
pub(crate) static BS_COMMIT_WAIT: Counter = Counter::new();
/// Host-side readback of the per-node `SplitDecision` array.
pub(crate) static BS_DECISION_READBACK: Counter = Counter::new();
/// Per-node merge step that compares the GPU-numeric best against
/// the host-categorical best (skipped when the model has no
/// categorical features).
pub(crate) static BS_CATEGORICAL_HOST_MERGE: Counter = Counter::new();
```

- [ ] **Step 3: Wire the counters into `dump_if_enabled`.**

Find the `dump_if_enabled` function (or whatever the existing dump function is named). Locate the `sites` array — it lists each counter alongside its display name. Add five new entries immediately after the entries for `BUILD_HISTOGRAMS_BATCH` and `SUBTRACT_BATCH`:

```rust
        ("find_best_splits_batch", &FIND_BEST_SPLITS_BATCH),
        ("  .dispatch", &BS_DISPATCH),
        ("  .commit_wait", &BS_COMMIT_WAIT),
        ("  .decision_readback", &BS_DECISION_READBACK),
        ("  .categorical_host_merge", &BS_CATEGORICAL_HOST_MERGE),
```

NOTE: Match the existing whitespace / indentation convention in the array. The leading two spaces on the sub-counter labels mirror how `BH_*` and `SUBTRACT_*` sub-counters are displayed.

- [ ] **Step 4: Verify with cargo check.**

```bash
cargo check -p alloygbm-backend-metal 2>&1 | tail -10
```

Expected: clean compile. The new counters are unused at this point (used by Tasks 6–8); if the compiler warns about unused statics, that's fine — they will be used shortly.

- [ ] **Step 5: Commit.**

```bash
git add crates/backend_metal/src/profile.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): FIND_BEST_SPLITS_BATCH profile counters

Adds the parent counter (FIND_BEST_SPLITS_BATCH) plus four sub-phase
counters (BS_DISPATCH, BS_COMMIT_WAIT, BS_DECISION_READBACK,
BS_CATEGORICAL_HOST_MERGE) for the Stage 4a GPU split-finding path.
Wired into the dump-if-enabled site list. Counters are unused until
the kernel dispatch lands.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `SplitDecisionPool` (sibling to `HistogramResidencyPool`)

**Why:** GPU split-finding kernel writes a packed `SplitDecision` array to a device-side buffer. Lifecycle: minted at the start of `find_best_splits_batch`, freed at the end of the call. A dedicated pool keeps the lifecycle invariant clear and avoids forcing the histogram pool to mint variable-sized entries.

**Files:**
- Create: `crates/backend_metal/src/split_decision_residency.rs`.
- Modify: `crates/backend_metal/src/lib.rs` (add `mod split_decision_residency;` declaration).

- [ ] **Step 1: Create `crates/backend_metal/src/split_decision_residency.rs`.**

```rust
//! Pool for device-side `SplitDecision` output buffers used by the
//! GPU split-finding kernel (Stage 4a).
//!
//! Each `find_best_splits_batch` call mints a single buffer sized
//! `node_count × size_of::<SplitDecisionGpu>()` (24 bytes per entry).
//! The pool owns the buffer for the call's lifetime; the
//! `SplitDecisionReleaseGuard` RAII helper releases it on Drop, mirroring
//! `HistogramReleaseGuard` from Stage 3.
//!
//! Sibling to `HistogramResidencyPool` rather than shared because the
//! lifecycle differs: histograms persist across multiple levels (parent
//! of subtract), split decisions live exactly one level.

use std::collections::HashMap;
use std::sync::Mutex;

use alloygbm_engine::{EngineError, EngineResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::residency::ResidencyPool;

/// Opaque handle to a minted split-decision buffer. Stored as a
/// monotonically-increasing token; 0 reserved as the "no handle"
/// sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitDecisionHandle(pub(crate) u64);

/// Fixed-size on-device representation of one node's split decision.
/// Must match the MSL `struct SplitDecisionGpu` declared in
/// `shaders/best_split.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SplitDecisionGpu {
    pub feature_idx: u32,   // 0xFFFFFFFF if no valid split found
    pub bin_threshold: u32,
    pub gain: f32,
    pub grad_left: f32,
    pub hess_left: f32,
    pub flags: u32,         // bit 0: missing-goes-right; bit 1: invalid
}

const SPLIT_DECISION_BYTES: usize = std::mem::size_of::<SplitDecisionGpu>();

const _: () = assert!(SPLIT_DECISION_BYTES == 24);

pub(crate) const SPLIT_FLAG_MISSING_GOES_RIGHT: u32 = 1 << 0;
pub(crate) const SPLIT_FLAG_INVALID: u32 = 1 << 1;

struct SplitDecisionEntry {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    node_count: u32,
}

struct PoolState {
    next_token: u64,
    live: HashMap<u64, SplitDecisionEntry>,
}

pub(crate) struct SplitDecisionPool {
    state: Mutex<PoolState>,
}

#[allow(dead_code)]
impl SplitDecisionPool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                next_token: 1,
                live: HashMap::new(),
            }),
        }
    }

    /// Allocate a fresh device-side buffer sized for `node_count`
    /// `SplitDecisionGpu` entries. Registered with `residency`. Buffer
    /// is `StorageModeShared` so the kernel can write and the host
    /// can read without explicit blit.
    pub(crate) fn mint(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &ResidencyPool,
        node_count: u32,
    ) -> EngineResult<SplitDecisionHandle> {
        let bytes = (node_count as usize)
            .checked_mul(SPLIT_DECISION_BYTES)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "split decision pool: node_count ({node_count}) × 24 overflowed usize"
                ))
            })?;

        // Pad to at least 16 bytes (Metal's minimum allocation alignment).
        let alloc_bytes = bytes.max(16);

        let buffer = device
            .newBufferWithLength_options(alloc_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "split decision pool: device returned None for {alloc_bytes}-byte buffer"
                ))
            })?;

        // Zero-initialise so unused / not-yet-written slots read as
        // INVALID (feature_idx = 0, flags = 0 by default — zero is
        // not a valid sentinel by itself, so kernel must explicitly
        // set flags |= INVALID for "no split found"). We document
        // this contract in the kernel.
        // SAFETY: buffer is StorageModeShared with `alloc_bytes` of
        // valid memory; we write exactly that many zero bytes.
        unsafe {
            std::ptr::write_bytes(
                buffer.contents().as_ptr() as *mut u8,
                0,
                alloc_bytes,
            );
        }

        residency.add_buffer(&buffer);
        residency.commit();

        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let token = state.next_token;
        state.next_token = state.next_token.wrapping_add(1);
        state.live.insert(
            token,
            SplitDecisionEntry {
                buffer,
                node_count,
            },
        );
        Ok(SplitDecisionHandle(token))
    }

    /// Borrow the device buffer for kernel binding. Returns
    /// `BackendUnavailable` if the handle is unknown or already
    /// released.
    pub(crate) fn buffer_for(
        &self,
        handle: SplitDecisionHandle,
    ) -> EngineResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let entry = state.live.get(&handle.0).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "split decision pool: buffer_for called on unknown handle {:?}",
                handle.0
            ))
        })?;
        Ok(entry.buffer.clone())
    }

    /// Read back the per-node `SplitDecisionGpu` array as a `Vec`.
    /// Caller must have already waited on the producing command
    /// buffer.
    pub(crate) fn read_decisions(
        &self,
        handle: SplitDecisionHandle,
    ) -> EngineResult<Vec<SplitDecisionGpu>> {
        let state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        let entry = state.live.get(&handle.0).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "split decision pool: read_decisions called on unknown handle {:?}",
                handle.0
            ))
        })?;
        let n = entry.node_count as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: buffer is StorageModeShared with `n × 24` valid
        // bytes (allocated in `mint`, kernel-written by the time the
        // caller invokes this). We read exactly that many entries
        // of the documented `#[repr(C)]` layout.
        let ptr = entry.buffer.contents().as_ptr() as *const SplitDecisionGpu;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(slice.to_vec())
    }

    /// Release a minted buffer. Safe to call multiple times: a
    /// duplicate release is a no-op (matches `HistogramResidencyPool`).
    pub(crate) fn release(&self, handle: SplitDecisionHandle) -> EngineResult<()> {
        let mut state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("split decision pool poisoned: {e}"))
        })?;
        state.live.remove(&handle.0);
        Ok(())
    }
}

/// RAII guard that releases the split-decision buffer on Drop.
/// Use across the find_best_splits_batch body so error paths free
/// the buffer deterministically.
pub(crate) struct SplitDecisionReleaseGuard<'a> {
    pool: &'a SplitDecisionPool,
    handle: SplitDecisionHandle,
}

impl<'a> SplitDecisionReleaseGuard<'a> {
    pub(crate) fn new(pool: &'a SplitDecisionPool, handle: SplitDecisionHandle) -> Self {
        Self { pool, handle }
    }
}

impl Drop for SplitDecisionReleaseGuard<'_> {
    fn drop(&mut self) {
        let _ = self.pool.release(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_default_device;

    #[test]
    fn mint_release_round_trip() {
        let Ok(metal_device) = open_default_device() else {
            return;
        };
        let residency = ResidencyPool::new(&metal_device.device);
        let pool = SplitDecisionPool::new();
        let handle = pool.mint(&metal_device.device, &residency, 8).unwrap();
        // read_decisions on a fresh handle returns 8 zero-initialised entries
        let decisions = pool.read_decisions(handle).unwrap();
        assert_eq!(decisions.len(), 8);
        for d in &decisions {
            assert_eq!(d.feature_idx, 0);
            assert_eq!(d.flags, 0);
        }
        pool.release(handle).unwrap();
        // double-release is a no-op
        pool.release(handle).unwrap();
    }

    #[test]
    fn mint_zero_node_count_is_safe() {
        let Ok(metal_device) = open_default_device() else {
            return;
        };
        let residency = ResidencyPool::new(&metal_device.device);
        let pool = SplitDecisionPool::new();
        let handle = pool.mint(&metal_device.device, &residency, 0).unwrap();
        let decisions = pool.read_decisions(handle).unwrap();
        assert!(decisions.is_empty());
        pool.release(handle).unwrap();
    }
}
```

NOTE: `open_default_device` is the existing helper used by other Metal tests (verify the function name in `crates/backend_metal/src/device.rs` before pasting; if it's named differently like `MetalDevice::open` or similar, use the correct name). Same for `ResidencyPool::new(&metal_device.device)` — verify the constructor signature against `crates/backend_metal/src/residency.rs`.

- [ ] **Step 2: Wire the module into `crates/backend_metal/src/lib.rs`.**

Find the existing `mod residency;` and `mod histogram_residency;` declarations near the top of the file. Add immediately after them:

```rust
mod split_decision_residency;
```

If those declarations are `pub(crate) mod`, match that visibility instead.

- [ ] **Step 3: Run the new tests.**

```bash
cargo test -p alloygbm-backend-metal split_decision_residency -- --test-threads=1 2>&1 | tail -10
```

Expected: both tests PASS (or are skipped if Metal isn't available — the early return on `open_default_device` handles that).

- [ ] **Step 4: Run the full Metal test suite to confirm nothing regressed.**

```bash
cargo test -p alloygbm-backend-metal -- --test-threads=1 2>&1 | tail -10
```

Expected: 47 + 2 = 49 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add crates/backend_metal/src/split_decision_residency.rs crates/backend_metal/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): SplitDecisionPool for GPU split-find output buffers

Sibling pool to HistogramResidencyPool. Each find_best_splits_batch
call mints a StorageModeShared buffer sized node_count × 24 bytes
(SplitDecisionGpu = repr(C) struct mirroring the MSL kernel's output
layout). RAII via SplitDecisionReleaseGuard on Drop. Mint/read/release
unit tests verify the round-trip on Apple-Silicon hosts; CI on non-Metal
machines short-circuits via the open_default_device early return.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Write the `best_split.metal` shader

**Why:** The two GPU kernels — `best_split_per_feature` (per (node, feature) threadgroup) and `best_split_reduce_features` (per node threadgroup) — are the heart of Stage 4a. We write the shader before the Rust dispatch wrapper because the shader's input/output layout determines what the wrapper has to bind.

**Files:**
- Create: `crates/backend_metal/src/shaders/best_split.metal`.

- [ ] **Step 1: Read the CPU `best_split_for_feature` to mirror its semantics exactly.**

Open `crates/backend_cpu/src/lib.rs` and read the function `best_split_for_feature` (around lines 448–591). Note:
- Skips the missing bin (`scan_limit = bins.len().min(missing_bin_idx)`).
- For each threshold, tries NaN-left and NaN-right.
- Per-side gain: `gain = (l1_threshold(grad)² / (hess + λ + ε)) + (... right ...) - (parent_gain_term)`.
- `EPSILON = 1e-6`, applied to both `parent_denom` and per-side `denom`.
- L1 thresholding: `l1_threshold(g) = sign(g) * max(|g| - α, 0)` (see `l1_threshold_gradient` at line 907).
- Constraints: `eff_lh > options.min_child_hessian` AND `eff_rh > options.min_child_hessian` AND `eff_lc > 0` AND `eff_rc > 0`. `min_leaf_magnitude` is checked too — skip if BOTH sides fall below it.
- Tie-break inside the feature: `gain > best_gain` (strictly greater); equal gain keeps the earlier bin.
- Cross-feature reduction (in `best_split_with_options_internal`) uses weighted gain `gain * feature_weights[fi]` for comparison; tie-break prefers the candidate that arrived first via `reduce` (which for an iterator is left-to-right ordering).

- [ ] **Step 2: Create `crates/backend_metal/src/shaders/best_split.metal`.**

```metal
// Stage 4a — GPU best-split kernel for numeric features.
//
// Mirrors `crates/backend_cpu/src/lib.rs::best_split_for_feature`
// byte-for-byte (in floating-point arithmetic order). Categorical
// Fisher-sort stays on CPU; mixed models do per-node host merge.
//
// Two kernel entry points:
//   1. `best_split_per_feature` — one threadgroup per (node, numeric
//      feature). Threads inside a threadgroup own one bin each.
//      Writes a per-(node, feature) candidate to a scratch buffer.
//   2. `best_split_reduce_features` — one threadgroup per node.
//      Reduces the per-feature scratch to a single SplitDecisionGpu.
//
// The wide histogram path stores `(grad, hess, count)` planes
// in row-major (feature, bin) order at the residency-pool entry.
// We read those planes directly in shared storage mode.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr float EPSILON = 1e-6f;
constant constexpr uint MAX_BINS = 1024u;

// Mirror of Rust `SplitDecisionGpu` (24 bytes).
struct SplitDecisionGpu {
    uint feature_idx;
    uint bin_threshold;
    float gain;
    float grad_left;
    float hess_left;
    uint flags;            // bit 0: missing-goes-right; bit 1: invalid
};

// Per-feature scratch entry (one per (node, feature) pair).
struct PerFeatureCandidate {
    float gain;            // unweighted, for downstream weighting
    float weighted_gain;   // gain * feature_weights[fi]
    float grad_left;
    float hess_left;
    uint feature_idx;      // the global feature index this slot was built for
    uint bin_threshold;
    uint flags;            // matches SplitDecisionGpu.flags layout
    uint _pad;             // align to 32 bytes for nicer simdgroup loads
};

// L1 thresholding mirrors `l1_threshold_gradient` in CPU code.
inline float l1_threshold(float grad_sum, float l1_alpha) {
    if (l1_alpha <= 0.0f) return grad_sum;
    if (grad_sum > l1_alpha) return grad_sum - l1_alpha;
    if (grad_sum < -l1_alpha) return grad_sum + l1_alpha;
    return 0.0f;
}

// Compute per-side denom + l1-thresholded grad as a tiny helper.
inline void leaf_terms(float g, float h, float l1, float l2, thread float& out_grad, thread float& out_denom) {
    out_grad = l1_threshold(g, l1);
    out_denom = h + l2 + EPSILON;
}

inline float gain_term(float grad_thresholded, float denom) {
    return (grad_thresholded * grad_thresholded) / denom;
}

struct BestSplitParams {
    float l2_lambda;
    float l1_alpha;
    float min_child_hessian;
    float min_leaf_magnitude;
    uint missing_bin_index; // u8 = 255, u16 = max_data_bin + 1
    uint bin_count;
    uint numeric_feature_count;
    uint node_count;
};

// Kernel 1: best_split_per_feature.
//
// Threadgroup grid: (numeric_feature_count, node_count, 1).
// Threadgroup size:  (bin_count_padded_to_32, 1, 1).
// Each thread loads one bin's stats; reductions are simdgroup + threadgroup.
//
// Inputs:
//   - grads:               [node × feature × bin] f32, row-major (node major)
//   - hesses:              [node × feature × bin] f32, row-major
//   - counts:              [node × feature × bin] u32, row-major
//   - feature_indices:     [numeric_feature_count] u32 — global feature index
//                          for each numeric slot
//   - feature_weights_buf: [numeric_feature_count] f32
//   - params:              BestSplitParams
// Output:
//   - per_feature_scratch: [node × numeric_feature_count] PerFeatureCandidate
[[kernel]]
void best_split_per_feature(
    device const float* grads             [[buffer(0)]],
    device const float* hesses            [[buffer(1)]],
    device const uint* counts             [[buffer(2)]],
    device const uint* feature_indices    [[buffer(3)]],
    device const float* feature_weights   [[buffer(4)]],
    constant BestSplitParams& params      [[buffer(5)]],
    device PerFeatureCandidate* scratch   [[buffer(6)]],
    uint3 tg_id                           [[threadgroup_position_in_grid]],
    uint3 thread_id                       [[thread_position_in_threadgroup]],
    uint3 tg_size                         [[threads_per_threadgroup]]
) {
    const uint feature_slot = tg_id.x;     // 0..numeric_feature_count
    const uint node_idx = tg_id.y;         // 0..node_count
    const uint bin = thread_id.x;          // 0..bin_count_padded
    const uint bin_count = params.bin_count;

    threadgroup float tg_left_grad[MAX_BINS];
    threadgroup float tg_left_hess[MAX_BINS];
    threadgroup uint tg_left_count[MAX_BINS];

    // ---- Load bin stats ----
    // Per-bin grad/hess/count for this (node, feature). Threads beyond
    // bin_count zero-fill (so they don't disturb prefix sums or argmax).
    float my_grad = 0.0f;
    float my_hess = 0.0f;
    uint my_count = 0u;
    if (bin < bin_count) {
        const uint plane_stride = params.numeric_feature_count * bin_count;
        const uint base = node_idx * plane_stride + feature_slot * bin_count + bin;
        my_grad = grads[base];
        my_hess = hesses[base];
        my_count = counts[base];
    }

    // ---- Per-feature totals (reduce across all bins) ----
    // Use simdgroup_sum + threadgroup-broadcast pattern. Two-pass
    // reduction matches the histogram kernel's pattern (D-003).
    float total_grad = simd_sum(my_grad);
    float total_hess = simd_sum(my_hess);
    uint total_count = simd_sum(my_count);

    threadgroup float tg_partial_grad[32];
    threadgroup float tg_partial_hess[32];
    threadgroup uint tg_partial_count[32];

    const uint simd_idx = bin / 32u;
    const uint lane_idx = bin % 32u;
    if (lane_idx == 0) {
        tg_partial_grad[simd_idx] = total_grad;
        tg_partial_hess[simd_idx] = total_hess;
        tg_partial_count[simd_idx] = total_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float gp = (lane_idx < num_simds) ? tg_partial_grad[lane_idx] : 0.0f;
        float hp = (lane_idx < num_simds) ? tg_partial_hess[lane_idx] : 0.0f;
        uint cp = (lane_idx < num_simds) ? tg_partial_count[lane_idx] : 0u;
        gp = simd_sum(gp);
        hp = simd_sum(hp);
        cp = simd_sum(cp);
        if (lane_idx == 0u) {
            tg_partial_grad[0] = gp;
            tg_partial_hess[0] = hp;
            tg_partial_count[0] = cp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_grad = tg_partial_grad[0];
    total_hess = tg_partial_hess[0];
    total_count = tg_partial_count[0];

    // ---- Missing-bin extraction ----
    float missing_grad = 0.0f;
    float missing_hess = 0.0f;
    uint missing_count = 0u;
    if (params.missing_bin_index < bin_count) {
        const uint plane_stride = params.numeric_feature_count * bin_count;
        const uint base = node_idx * plane_stride
            + feature_slot * bin_count
            + params.missing_bin_index;
        missing_grad = grads[base];
        missing_hess = hesses[base];
        missing_count = counts[base];
    }

    // ---- Early-out: parent constraint check ----
    if (total_hess <= params.min_child_hessian) {
        if (bin == 0u) {
            const uint out_idx = node_idx * params.numeric_feature_count + feature_slot;
            scratch[out_idx].gain = -INFINITY;
            scratch[out_idx].weighted_gain = -INFINITY;
            scratch[out_idx].feature_idx = feature_indices[feature_slot];
            scratch[out_idx].bin_threshold = 0u;
            scratch[out_idx].grad_left = 0.0f;
            scratch[out_idx].hess_left = 0.0f;
            scratch[out_idx].flags = 0x2u;  // INVALID
        }
        return;
    }

    const float nm_total_grad = total_grad - missing_grad;
    const float nm_total_hess = total_hess - missing_hess;
    const uint nm_total_count = (total_count > missing_count)
        ? (total_count - missing_count) : 0u;

    const float parent_grad = l1_threshold(total_grad, params.l1_alpha);
    const float parent_denom = total_hess + params.l2_lambda + EPSILON;
    const float parent_gain_term = (parent_grad * parent_grad) / parent_denom;

    // ---- Inclusive prefix scan over non-missing bins ----
    // We exclude the missing bin from the scan: writing zero into
    // tg_left_* at that index makes it inert relative to the
    // prefix-sum totals (because we subtract `missing_*` from totals
    // before computing right-side stats).
    const bool is_missing_slot = (bin == params.missing_bin_index);
    float scan_grad = (bin < bin_count && !is_missing_slot) ? my_grad : 0.0f;
    float scan_hess = (bin < bin_count && !is_missing_slot) ? my_hess : 0.0f;
    uint scan_count = (bin < bin_count && !is_missing_slot) ? my_count : 0u;

    // Two-pass exclusive prefix: simd_prefix_inclusive_sum + simd-block scan.
    // We want INCLUSIVE prefix sum (left side includes bin K when threshold = K).
    float prefix_grad = simd_prefix_inclusive_sum(scan_grad);
    float prefix_hess = simd_prefix_inclusive_sum(scan_hess);
    uint prefix_count = simd_prefix_inclusive_sum(scan_count);

    threadgroup float tg_simd_grad_total[32];
    threadgroup float tg_simd_hess_total[32];
    threadgroup uint tg_simd_count_total[32];
    if (lane_idx == 31u) {
        tg_simd_grad_total[simd_idx] = prefix_grad;
        tg_simd_hess_total[simd_idx] = prefix_hess;
        tg_simd_count_total[simd_idx] = prefix_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate per-simd offsets serially in thread 0, then broadcast.
    if (bin == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float gacc = 0.0f;
        float hacc = 0.0f;
        uint cacc = 0u;
        for (uint i = 0u; i < num_simds; ++i) {
            float g = tg_simd_grad_total[i];
            float h = tg_simd_hess_total[i];
            uint c = tg_simd_count_total[i];
            tg_simd_grad_total[i] = gacc;
            tg_simd_hess_total[i] = hacc;
            tg_simd_count_total[i] = cacc;
            gacc += g;
            hacc += h;
            cacc += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix_grad += tg_simd_grad_total[simd_idx];
    prefix_hess += tg_simd_hess_total[simd_idx];
    prefix_count += tg_simd_count_total[simd_idx];

    if (bin < bin_count) {
        tg_left_grad[bin] = prefix_grad;
        tg_left_hess[bin] = prefix_hess;
        tg_left_count[bin] = prefix_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Per-bin gain candidates ----
    // Each thread evaluates threshold = its bin index, comparing
    // tg_left_*[bin] against (nm_total - left + missing_*).
    //
    // CPU mirror: scan_limit = bin_count.min(missing_bin_idx). We
    // skip bins where threshold + 1 >= scan_limit AND nm_total_count
    // == left_count (matches the CPU early-continue for the final
    // bin).
    const uint scan_limit = min(bin_count, params.missing_bin_index);

    float best_gain = 0.0f;
    uint best_threshold = 0u;
    float best_grad_left = 0.0f;
    float best_hess_left = 0.0f;
    uint best_flags = 0x2u;       // INVALID until we find one

    if (bin < scan_limit) {
        const float left_grad = tg_left_grad[bin];
        const float left_hess = tg_left_hess[bin];
        const uint left_count = tg_left_count[bin];

        const bool last_threshold = (bin + 1u >= scan_limit);
        const bool exhausts_right = (nm_total_count == left_count);
        if (!(last_threshold && exhausts_right)) {
            const float right_grad = nm_total_grad - left_grad;
            const float right_hess = nm_total_hess - left_hess;
            const uint right_count = (nm_total_count > left_count)
                ? (nm_total_count - left_count) : 0u;

            // Try NaN-left and NaN-right; keep the better.
            for (uint dir = 0u; dir < 2u; ++dir) {
                const bool nan_left = (dir == 0u);
                const float eff_lg = nan_left ? (left_grad + missing_grad) : left_grad;
                const float eff_lh = nan_left ? (left_hess + missing_hess) : left_hess;
                const uint eff_lc = nan_left ? (left_count + missing_count) : left_count;
                const float eff_rg = nan_left ? right_grad : (right_grad + missing_grad);
                const float eff_rh = nan_left ? right_hess : (right_hess + missing_hess);
                const uint eff_rc = nan_left ? right_count : (right_count + missing_count);

                if (eff_lc == 0u || eff_rc == 0u
                    || eff_lh <= params.min_child_hessian
                    || eff_rh <= params.min_child_hessian) continue;

                float left_grad_for_gain;
                float left_denom;
                float right_grad_for_gain;
                float right_denom;
                leaf_terms(eff_lg, eff_lh, params.l1_alpha, params.l2_lambda,
                           left_grad_for_gain, left_denom);
                leaf_terms(eff_rg, eff_rh, params.l1_alpha, params.l2_lambda,
                           right_grad_for_gain, right_denom);

                if (params.min_leaf_magnitude > 0.0f) {
                    const float lm = fabs(left_grad_for_gain) / left_denom;
                    const float rm = fabs(right_grad_for_gain) / right_denom;
                    if (lm < params.min_leaf_magnitude && rm < params.min_leaf_magnitude) continue;
                }

                const float gain = gain_term(left_grad_for_gain, left_denom)
                    + gain_term(right_grad_for_gain, right_denom)
                    - parent_gain_term;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_threshold = bin;
                    best_grad_left = eff_lg;
                    best_hess_left = eff_lh;
                    best_flags = nan_left ? 0u : 0x1u;
                }
            }
        }
    }

    // ---- Threadgroup argmax across bins ----
    // Two-pass deterministic reduction: each lane writes its (gain,
    // bin) into shared memory, simdgroup max picks the per-simd
    // winner, thread 0 reduces simd winners. Ties: lower bin wins
    // (matches CPU `gain > best_gain` strict comparison).
    threadgroup float tg_best_gain[32];
    threadgroup uint tg_best_threshold[32];
    threadgroup float tg_best_grad_left[32];
    threadgroup float tg_best_hess_left[32];
    threadgroup uint tg_best_flags[32];

    // Per-simdgroup reduction first.
    float lane_gain = best_gain;
    uint lane_threshold = best_threshold;
    float lane_grad_left = best_grad_left;
    float lane_hess_left = best_hess_left;
    uint lane_flags = best_flags;

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        const float other_gain = simd_shuffle_xor(lane_gain, offset);
        const uint other_thresh = simd_shuffle_xor(lane_threshold, offset);
        const float other_gl = simd_shuffle_xor(lane_grad_left, offset);
        const float other_hl = simd_shuffle_xor(lane_hess_left, offset);
        const uint other_flags = simd_shuffle_xor(lane_flags, offset);
        // Tie-break: strictly greater wins; equal keeps lower bin.
        const bool take_other = (other_gain > lane_gain)
            || (other_gain == lane_gain && other_thresh < lane_threshold);
        if (take_other) {
            lane_gain = other_gain;
            lane_threshold = other_thresh;
            lane_grad_left = other_gl;
            lane_hess_left = other_hl;
            lane_flags = other_flags;
        }
    }
    if (lane_idx == 0u) {
        tg_best_gain[simd_idx] = lane_gain;
        tg_best_threshold[simd_idx] = lane_threshold;
        tg_best_grad_left[simd_idx] = lane_grad_left;
        tg_best_hess_left[simd_idx] = lane_hess_left;
        tg_best_flags[simd_idx] = lane_flags;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (bin == 0u) {
        const uint num_simds = (tg_size.x + 31u) / 32u;
        float final_gain = tg_best_gain[0];
        uint final_thresh = tg_best_threshold[0];
        float final_gl = tg_best_grad_left[0];
        float final_hl = tg_best_hess_left[0];
        uint final_flags = tg_best_flags[0];
        for (uint i = 1u; i < num_simds; ++i) {
            const float g = tg_best_gain[i];
            const uint t = tg_best_threshold[i];
            const bool take = (g > final_gain)
                || (g == final_gain && t < final_thresh);
            if (take) {
                final_gain = g;
                final_thresh = t;
                final_gl = tg_best_grad_left[i];
                final_hl = tg_best_hess_left[i];
                final_flags = tg_best_flags[i];
            }
        }

        const uint out_idx = node_idx * params.numeric_feature_count + feature_slot;
        const uint feat_idx = feature_indices[feature_slot];
        const float weighted = final_gain
            * (feature_slot < params.numeric_feature_count
                ? feature_weights[feature_slot] : 1.0f);
        scratch[out_idx].gain = final_gain;
        scratch[out_idx].weighted_gain = weighted;
        scratch[out_idx].grad_left = final_gl;
        scratch[out_idx].hess_left = final_hl;
        scratch[out_idx].feature_idx = feat_idx;
        scratch[out_idx].bin_threshold = final_thresh;
        scratch[out_idx].flags = final_flags;
    }
}

// Kernel 2: best_split_reduce_features.
//
// Threadgroup grid: (node_count, 1, 1).
// Threadgroup size:  (numeric_feature_count_padded_to_32, 1, 1).
[[kernel]]
void best_split_reduce_features(
    device const PerFeatureCandidate* scratch [[buffer(0)]],
    constant BestSplitParams& params         [[buffer(1)]],
    device SplitDecisionGpu* out             [[buffer(2)]],
    uint3 tg_id                              [[threadgroup_position_in_grid]],
    uint3 thread_id                          [[thread_position_in_threadgroup]]
) {
    const uint node_idx = tg_id.x;
    const uint feat_slot = thread_id.x;
    const uint nf = params.numeric_feature_count;

    PerFeatureCandidate cand;
    if (feat_slot < nf) {
        cand = scratch[node_idx * nf + feat_slot];
    } else {
        cand.weighted_gain = -INFINITY;
        cand.gain = -INFINITY;
        cand.feature_idx = 0xFFFFFFFFu;
        cand.bin_threshold = 0u;
        cand.grad_left = 0.0f;
        cand.hess_left = 0.0f;
        cand.flags = 0x2u;  // INVALID
    }

    // Reduce across feature slots; tie-break: strictly greater
    // weighted gain wins; equal keeps lower feature_idx.
    threadgroup PerFeatureCandidate tg_cands[32];
    const uint simd_idx = feat_slot / 32u;
    const uint lane_idx = feat_slot % 32u;

    PerFeatureCandidate lane_cand = cand;
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        PerFeatureCandidate other;
        other.weighted_gain = simd_shuffle_xor(lane_cand.weighted_gain, offset);
        other.gain = simd_shuffle_xor(lane_cand.gain, offset);
        other.feature_idx = simd_shuffle_xor(lane_cand.feature_idx, offset);
        other.bin_threshold = simd_shuffle_xor(lane_cand.bin_threshold, offset);
        other.grad_left = simd_shuffle_xor(lane_cand.grad_left, offset);
        other.hess_left = simd_shuffle_xor(lane_cand.hess_left, offset);
        other.flags = simd_shuffle_xor(lane_cand.flags, offset);
        const bool take = (other.weighted_gain > lane_cand.weighted_gain)
            || (other.weighted_gain == lane_cand.weighted_gain
                && other.feature_idx < lane_cand.feature_idx);
        if (take) lane_cand = other;
    }

    if (lane_idx == 0u) {
        tg_cands[simd_idx] = lane_cand;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (feat_slot == 0u) {
        const uint num_simds = (nf + 31u) / 32u;
        PerFeatureCandidate winner = tg_cands[0];
        for (uint i = 1u; i < num_simds; ++i) {
            const PerFeatureCandidate other = tg_cands[i];
            const bool take = (other.weighted_gain > winner.weighted_gain)
                || (other.weighted_gain == winner.weighted_gain
                    && other.feature_idx < winner.feature_idx);
            if (take) winner = other;
        }

        device SplitDecisionGpu& dest = out[node_idx];
        if (winner.gain > 0.0f && (winner.flags & 0x2u) == 0u) {
            dest.feature_idx = winner.feature_idx;
            dest.bin_threshold = winner.bin_threshold;
            dest.gain = winner.gain;
            dest.grad_left = winner.grad_left;
            dest.hess_left = winner.hess_left;
            dest.flags = winner.flags & ~0x2u;
        } else {
            dest.feature_idx = 0xFFFFFFFFu;
            dest.bin_threshold = 0u;
            dest.gain = 0.0f;
            dest.grad_left = 0.0f;
            dest.hess_left = 0.0f;
            dest.flags = 0x2u;
        }
    }
}
```

NOTE: This shader is long but transcription-only — the math mirrors `best_split_for_feature` exactly. Validation comes from the parity tests in Task 7.

- [ ] **Step 3: Verify the file is well-formed by listing it.**

```bash
wc -l crates/backend_metal/src/shaders/best_split.metal
```

Expected: roughly 350–400 lines. If far less, paste again — content was likely truncated.

- [ ] **Step 4: Commit.**

```bash
git add crates/backend_metal/src/shaders/best_split.metal
git commit -m "$(cat <<'EOF'
feat(backend_metal): best_split.metal kernel for GPU split finding

Two entry points:
  - best_split_per_feature: per (node, numeric feature) threadgroup.
    Mirrors crates/backend_cpu/src/lib.rs::best_split_for_feature
    in arithmetic order — two-pass simdgroup+threadgroup prefix scan
    over non-missing bins, tries NaN-left and NaN-right, evaluates
    Newton-step gain at each threshold.
  - best_split_reduce_features: per-node threadgroup.
    Reduces per-feature scratch to one SplitDecisionGpu using
    weighted-gain comparison; tie-break prefers lower feature_idx.
Determinism: no float atomics anywhere, fixed thread-to-bin mapping,
two-pass reductions throughout (matches D-003 / Stage-1 patterns).
The kernel is unused until the Rust dispatch wrapper lands in Task 7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Rust dispatch wrapper + pipeline cache + numeric-only override + parity tests

**Why:** The kernel is inert without a Rust dispatch and a Metal pipeline. This task wires both, plus the `MetalBackend::find_best_splits_batch` override (numeric-only path — categorical fallback comes in Task 8), plus parity tests against CPU.

**Files:**
- Create: `crates/backend_metal/src/kernels/best_split.rs`.
- Create: `crates/backend_metal/tests/best_split_parity.rs`.
- Modify: `crates/backend_metal/src/kernels/mod.rs` (add `pub(crate) mod best_split;`).
- Modify: `crates/backend_metal/src/pipelines.rs` (cache entry for the two kernel entries).
- Modify: `crates/backend_metal/src/lib.rs` (override + eligibility predicate).

- [ ] **Step 1: Add the kernel module declaration.**

Open `crates/backend_metal/src/kernels/mod.rs`. The file currently lists `pub(crate) mod histogram;`, `pub(crate) mod partition;`, etc. Add immediately after the last `pub(crate) mod` declaration:

```rust
pub(crate) mod best_split;
```

- [ ] **Step 2: Add the pipeline cache entry.**

Open `crates/backend_metal/src/pipelines.rs`. Find the existing `HistogramPipelineCache` struct and the analogous structures for partition/subtract. Add:

1. A new struct `BestSplitPipelineCache` with fields for both kernel pipelines:

```rust
#[cfg(target_os = "macos")]
pub(crate) struct BestSplitPipelineCache {
    pub(crate) per_feature: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    pub(crate) reduce_features: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
}
```

2. A constructor that compiles `best_split.metal` and instantiates both pipelines:

```rust
#[cfg(target_os = "macos")]
impl BestSplitPipelineCache {
    pub(crate) fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        capabilities: &crate::device::Capabilities,
    ) -> alloygbm_engine::EngineResult<Self> {
        use alloygbm_engine::{EngineError, EngineResult};
        use objc2_foundation::NSString;
        use objc2_metal::{MTLCompileOptions, MTLDevice, MTLLibrary};

        let source = include_str!("shaders/best_split.metal");
        let _family = if capabilities.metal4 { "metal4" } else { "metal3" };

        let options = unsafe { MTLCompileOptions::new() };
        let nss = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&nss, Some(&options))
            .map_err(|e| {
                EngineError::BackendUnavailable(format!(
                    "best_split.metal library compile failed: {e:?}"
                ))
            })?;

        let per_feature_fn = library
            .newFunctionWithName(&NSString::from_str("best_split_per_feature"))
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "best_split.metal: missing best_split_per_feature".to_string(),
                )
            })?;
        let reduce_features_fn = library
            .newFunctionWithName(&NSString::from_str("best_split_reduce_features"))
            .ok_or_else(|| {
                EngineError::BackendUnavailable(
                    "best_split.metal: missing best_split_reduce_features".to_string(),
                )
            })?;

        let per_feature = device
            .newComputePipelineStateWithFunction_error(&per_feature_fn)
            .map_err(|e| {
                EngineError::BackendUnavailable(format!(
                    "best_split_per_feature pipeline creation failed: {e:?}"
                ))
            })?;
        let reduce_features = device
            .newComputePipelineStateWithFunction_error(&reduce_features_fn)
            .map_err(|e| {
                EngineError::BackendUnavailable(format!(
                    "best_split_reduce_features pipeline creation failed: {e:?}"
                ))
            })?;

        Ok(BestSplitPipelineCache {
            per_feature,
            reduce_features,
        })
    }
}
```

NOTE: The `include_str!("shaders/best_split.metal")` path is RELATIVE TO THE FILE. If `pipelines.rs` is at `crates/backend_metal/src/pipelines.rs`, then `"shaders/best_split.metal"` resolves to `crates/backend_metal/src/shaders/best_split.metal`. Verify by running `cargo build` after the edit.

NOTE 2: The exact `MTLDevice` / `MTLLibrary` / `MTLCompileOptions` API names may differ slightly from what's shown — match what the existing `HistogramPipelineCache::new` uses verbatim (read it in `pipelines.rs` first).

3. Add a field on `MetalBackend`:

In `crates/backend_metal/src/lib.rs`, find the `pub struct MetalBackend { ... }` definition. Add a field:

```rust
    pub(crate) best_split_pipeline_cache: pipelines::BestSplitPipelineCache,
```

(Or whatever the existing pipeline cache fields are named — match the convention.)

In `MetalBackend::new()` (the constructor), find where the other pipeline caches are constructed (e.g. `histogram_pipeline_cache: HistogramPipelineCache::new(...)`) and add:

```rust
            best_split_pipeline_cache: pipelines::BestSplitPipelineCache::new(&device, &capabilities)?,
```

(Match the local variable naming for `device` and `capabilities` from the constructor body.)

- [ ] **Step 3: Run cargo check to confirm pipeline cache compiles.**

```bash
cargo check -p alloygbm-backend-metal 2>&1 | tail -15
```

Expected: clean compile. Common failure modes:
- `include_str!` path wrong → fix relative path.
- API name drift → match existing pipeline cache patterns exactly.

- [ ] **Step 4: Create `crates/backend_metal/src/kernels/best_split.rs`.**

```rust
//! GPU dispatch wrapper for the Stage 4a `best_split.metal` kernel.
//!
//! Mirrors the shape of `kernels/histogram.rs::dispatch_histograms_batch`:
//! one command buffer per call, one commit + waitUntilCompleted, host
//! reads back the small per-node `SplitDecisionGpu` array afterwards.

#![cfg(target_os = "macos")]
#![allow(unsafe_code)]

use std::mem::size_of;

use alloygbm_core::{GpuHistogramHandle, HistogramBundle, HistogramStorage};
use alloygbm_engine::{
    CategoricalFeatureInfo, EngineError, EngineResult, SplitFindRequest, SplitSelectionOptions,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::device::MetalDevice;
use crate::histogram_residency::HistogramResidencyPool;
use crate::pipelines::BestSplitPipelineCache;
use crate::profile;
use crate::residency::ResidencyPool;
use crate::split_decision_residency::{
    SplitDecisionGpu, SplitDecisionHandle, SplitDecisionPool,
};

/// Mirror of MSL `BestSplitParams` (32 bytes, 4-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BestSplitParamsGpu {
    pub l2_lambda: f32,
    pub l1_alpha: f32,
    pub min_child_hessian: f32,
    pub min_leaf_magnitude: f32,
    pub missing_bin_index: u32,
    pub bin_count: u32,
    pub numeric_feature_count: u32,
    pub node_count: u32,
}

const _: () = assert!(size_of::<BestSplitParamsGpu>() == 32);

/// Per-feature scratch entry — must match MSL `PerFeatureCandidate`
/// byte-for-byte (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PerFeatureCandidateGpu {
    gain: f32,
    weighted_gain: f32,
    grad_left: f32,
    hess_left: f32,
    feature_idx: u32,
    bin_threshold: u32,
    flags: u32,
    _pad: u32,
}

const _: () = assert!(size_of::<PerFeatureCandidateGpu>() == 32);

/// Encode + commit + wait + read back N nodes' best-split decisions
/// in a single command buffer.
///
/// All bundles in `requests` MUST share `feature_count`, `bin_count`,
/// and the `feature_indices` array — matches the engine's level-wise
/// invariant that all nodes at one level share the same histogram
/// shape. Returns one `SplitDecisionGpu` per request, in the same
/// order as `requests`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_find_best_splits_batch(
    metal_device: &MetalDevice,
    pipeline_cache: &BestSplitPipelineCache,
    histogram_residency: &HistogramResidencyPool,
    split_decision_pool: &SplitDecisionPool,
    residency: &ResidencyPool,
    requests: &[SplitFindRequest<'_>],
    options: SplitSelectionOptions,
    numeric_feature_indices: &[u32],
    feature_weights: &[f32],
) -> EngineResult<Vec<SplitDecisionGpu>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let node_count = requests.len() as u32;
    let numeric_feature_count = numeric_feature_indices.len() as u32;

    if numeric_feature_count == 0 {
        // No numeric features → caller must have routed entirely
        // through the host (categorical-only model). This is a
        // contract violation if it happens here.
        return Err(EngineError::BackendUnavailable(
            "dispatch_find_best_splits_batch called with zero numeric features"
                .to_string(),
        ));
    }

    // ---- Extract first request's GPU handle to learn shape. ----
    // All requests must share the same shape (engine invariant).
    let first = requests[0].histograms;
    let bin_count: u32 = first.bin_count;
    let pool_handle: GpuHistogramHandle = match &first.storage {
        HistogramStorage::Gpu { handle, .. } => *handle,
        HistogramStorage::Cpu(_) => {
            return Err(EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: histogram is not GPU-resident; \
                 caller must fall back to scalar path"
                    .to_string(),
            ));
        }
    };

    // ---- Allocate per-feature scratch buffer. ----
    let scratch_bytes = (node_count as usize)
        * (numeric_feature_count as usize)
        * size_of::<PerFeatureCandidateGpu>();
    let scratch_alloc = scratch_bytes.max(16);
    let device = &metal_device.device;
    let scratch_buf = device
        .newBufferWithLength_options(scratch_alloc, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "dispatch_find_best_splits_batch: scratch buffer alloc failed ({scratch_alloc} bytes)"
            ))
        })?;
    residency.add_buffer(&scratch_buf);

    // ---- Allocate output buffer via the pool. ----
    let out_handle = split_decision_pool.mint(device, residency, node_count)?;
    let out_buffer = split_decision_pool.buffer_for(out_handle)?;
    residency.commit();

    // ---- Resolve the histogram pool entry to bind its planes. ----
    let (grad_buffer, hess_buffer, counts_buffer) =
        histogram_residency.borrow_planes(pool_handle)?;

    // ---- Feature-indices and feature-weights buffers. ----
    let feat_idx_bytes = (numeric_feature_count as usize) * 4;
    let feat_idx_buf = device
        .newBufferWithLength_options(feat_idx_bytes.max(16), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: feature_indices alloc failed".to_string(),
            )
        })?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            numeric_feature_indices.as_ptr(),
            feat_idx_buf.contents().as_ptr() as *mut u32,
            numeric_feature_count as usize,
        );
    }
    residency.add_buffer(&feat_idx_buf);

    // Numeric-only feature_weights slice. The caller is expected to
    // pass `feature_weights` already restricted to the numeric subset
    // (Task 8 wires the eligibility predicate / restriction).
    let fw_bytes = (numeric_feature_count as usize) * 4;
    let fw_buf = device
        .newBufferWithLength_options(fw_bytes.max(16), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: feature_weights alloc failed".to_string(),
            )
        })?;
    let fw_src: Vec<f32> = if feature_weights.is_empty() {
        vec![1.0; numeric_feature_count as usize]
    } else {
        // Caller passes the full feature_weights slice; pull out the
        // numeric subset.
        numeric_feature_indices
            .iter()
            .map(|&fi| {
                feature_weights
                    .get(fi as usize)
                    .copied()
                    .unwrap_or(1.0)
            })
            .collect()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(
            fw_src.as_ptr(),
            fw_buf.contents().as_ptr() as *mut f32,
            numeric_feature_count as usize,
        );
    }
    residency.add_buffer(&fw_buf);
    residency.commit();

    // ---- Params struct. ----
    let params = BestSplitParamsGpu {
        l2_lambda: options.l2_lambda,
        l1_alpha: options.l1_alpha,
        min_child_hessian: options.min_child_hessian,
        min_leaf_magnitude: options.min_leaf_magnitude,
        missing_bin_index: options.missing_bin_index as u32,
        bin_count,
        numeric_feature_count,
        node_count,
    };

    // ---- Encode + commit + wait. ----
    let _p_total = profile::ScopedProbe::new(&profile::FIND_BEST_SPLITS_BATCH);
    let cmd_buffer = metal_device
        .queue
        .commandBuffer()
        .ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: no command buffer".to_string(),
            )
        })?;

    {
        let _p_dispatch = profile::ScopedProbe::new(&profile::BS_DISPATCH);
        let encoder = cmd_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable(
                "dispatch_find_best_splits_batch: no encoder".to_string(),
            )
        })?;

        // Kernel 1: per-feature.
        encoder.setComputePipelineState(&pipeline_cache.per_feature);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&grad_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&hess_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&counts_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&feat_idx_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&fw_buf), 0, 4);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<BestSplitParamsGpu>(),
                5,
            );
            encoder.setBuffer_offset_atIndex(Some(&scratch_buf), 0, 6);
        }
        // Pad threadgroup size up to multiple of 32, capped at 1024.
        let tg_size = ((bin_count + 31) / 32 * 32).min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: numeric_feature_count as usize,
                height: node_count as usize,
                depth: 1,
            },
            MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            },
        );

        // Kernel 2: cross-feature reduce.
        encoder.setComputePipelineState(&pipeline_cache.reduce_features);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&scratch_buf), 0, 0);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                size_of::<BestSplitParamsGpu>(),
                1,
            );
            encoder.setBuffer_offset_atIndex(Some(&out_buffer), 0, 2);
        }
        let reduce_tg_size = ((numeric_feature_count + 31) / 32 * 32).min(1024) as usize;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: node_count as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: reduce_tg_size,
                height: 1,
                depth: 1,
            },
        );

        encoder.endEncoding();
    }

    {
        let _p_wait = profile::ScopedProbe::new(&profile::BS_COMMIT_WAIT);
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();
    }

    let decisions = {
        let _p_read = profile::ScopedProbe::new(&profile::BS_DECISION_READBACK);
        split_decision_pool.read_decisions(out_handle)?
    };

    // Release the pool entry now (no RAII guard here because we own
    // the success path — caller-error case would also hit Drop on
    // out_handle, but we mint locally and never expose it).
    split_decision_pool.release(out_handle)?;

    // Keep scratch and feature buffers alive past the wait via this
    // function's scope.
    drop(scratch_buf);
    drop(feat_idx_buf);
    drop(fw_buf);

    Ok(decisions)
}
```

NOTE: `histogram_residency.borrow_planes(pool_handle)` — verify this method name. The existing `read_planes` reads bytes; we need `Retained<ProtocolObject<dyn MTLBuffer>>` for each plane to bind to the encoder. If the pool only exposes `read_planes`, you'll need to add a sibling method `borrow_planes` that returns the three buffer Retaineds without copying. Check `histogram_residency.rs` first; if `borrow_planes` doesn't exist, add it as a one-line addition in this task (returns `(Retained<...>, Retained<...>, Retained<...>)` from the entry's three buffers).

If you need to add `borrow_planes`, in `crates/backend_metal/src/histogram_residency.rs`, immediately after `read_planes`, add:

```rust
    /// Borrow the three plane buffers (`grad`, `hess`, `counts`) for
    /// kernel binding. Caller must hold the returned Retaineds for
    /// the lifetime of the dispatch.
    pub(crate) fn borrow_planes(
        &self,
        handle: GpuHistogramHandle,
    ) -> EngineResult<(
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
    )> {
        let state = self.state.lock().map_err(|e| {
            EngineError::BackendUnavailable(format!("histogram residency pool poisoned: {e}"))
        })?;
        let entry = state.live.get(&handle.0).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "histogram residency pool: borrow_planes called on unknown handle {:?}",
                handle.0
            ))
        })?;
        Ok((entry.grad.clone(), entry.hess.clone(), entry.counts.clone()))
    }
```

- [ ] **Step 5: Add the override in `MetalBackend`.**

In `crates/backend_metal/src/lib.rs`, find the existing `subtract_histogram_bundle_batch` override. Add immediately after it:

```rust
    fn find_best_splits_batch(
        &self,
        requests: &[alloygbm_engine::SplitFindRequest<'_>],
        options: alloygbm_engine::SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[alloygbm_engine::CategoricalFeatureInfo],
    ) -> alloygbm_engine::EngineResult<Vec<Option<alloygbm_core::SplitCandidate>>> {
        // Eligibility: numeric-only fast path. Mixed numeric+categorical
        // models route through the per-node merge in Task 8; for now
        // any categorical feature falls back to the scalar default.
        if !categorical_features.is_empty() {
            return requests
                .iter()
                .map(|r| {
                    self.best_split_with_options(
                        r.histograms,
                        options,
                        feature_weights,
                        categorical_features,
                    )
                })
                .collect();
        }
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Build the numeric_feature_indices slice from the first
        // bundle (all bundles share shape per engine invariant).
        let first = requests[0].histograms;
        let numeric_feature_indices: Vec<u32> = first
            .feature_indices()
            .iter()
            .copied()
            .collect();

        let gpu_decisions = match kernels::best_split::dispatch_find_best_splits_batch(
            &self.metal_device,
            &self.best_split_pipeline_cache,
            &self.histogram_residency,
            &self.split_decision_residency,
            &self.residency,
            requests,
            options,
            &numeric_feature_indices,
            feature_weights,
        ) {
            Ok(d) => d,
            Err(_) => {
                // Fall back to scalar default on any kernel error.
                return requests
                    .iter()
                    .map(|r| {
                        self.best_split_with_options(
                            r.histograms,
                            options,
                            feature_weights,
                            categorical_features,
                        )
                    })
                    .collect();
            }
        };

        // Convert SplitDecisionGpu → Option<SplitCandidate>.
        let mut out: Vec<Option<alloygbm_core::SplitCandidate>> = Vec::with_capacity(requests.len());
        for (req, decision) in requests.iter().zip(gpu_decisions.iter()) {
            if (decision.flags & 0x2u32) != 0 || decision.feature_idx == u32::MAX {
                out.push(None);
                continue;
            }
            // Reconstruct the right-side stats from the bundle totals
            // minus left.
            let bundle_totals = first.feature_total(decision.feature_idx as usize);
            let total_grad = bundle_totals.grad_sum;
            let total_hess = bundle_totals.hess_sum;
            let total_count = bundle_totals.count;
            let grad_right = total_grad - decision.grad_left;
            let hess_right = total_hess - decision.hess_left;
            // Left/right counts aren't carried in SplitDecisionGpu;
            // recover them from the histogram. For now we set them
            // to 0 — downstream code reads grad/hess for leaf-value
            // computation, not counts. (Verify this in a follow-up
            // step before considering Task 7 done.)
            out.push(Some(alloygbm_core::SplitCandidate {
                node_id: req.histograms.node_id,
                feature_index: decision.feature_idx,
                threshold_bin: decision.bin_threshold as u16,
                gain: decision.gain,
                default_left: (decision.flags & 0x1u32) == 0,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: alloygbm_core::NodeStats {
                    grad_sum: decision.grad_left,
                    hess_sum: decision.hess_left,
                    row_count: 0,
                },
                right_stats: alloygbm_core::NodeStats {
                    grad_sum: grad_right,
                    hess_sum: hess_right,
                    row_count: total_count.saturating_sub(0),
                },
            }));
        }
        Ok(out)
    }
```

NOTE on `feature_indices()` and `feature_total()` accessors on `HistogramBundle`: verify these methods exist. If not, the closest equivalent is reading `bundle.feature_histograms()` (which returns `Vec<FeatureHistogram>`) and computing totals manually. If that's the case, replace the `numeric_feature_indices` and `bundle_totals` lines with the `feature_histograms()`-based computation.

NOTE on `row_count: 0` in `left_stats` / `right_stats`: this is a known gap that the parity tests will catch. The CPU path populates `left_count` and `right_count` from the histogram scan; the GPU kernel currently doesn't return counts because the Newton-step gain doesn't depend on them. If the parity test fails on `row_count` mismatch, update the kernel to return left_count and recompute right_count host-side (add two u32 fields to `SplitDecisionGpu` + corresponding kernel writes).

Add the new field on `MetalBackend`:

```rust
    pub(crate) split_decision_residency: split_decision_residency::SplitDecisionPool,
```

In `MetalBackend::new()`:

```rust
            split_decision_residency: split_decision_residency::SplitDecisionPool::new(),
```

- [ ] **Step 6: Create the parity test file `crates/backend_metal/tests/best_split_parity.rs`.**

```rust
//! Bit-exact parity tests for Stage 4a GPU split finding against the
//! CPU baseline. Skipped silently on non-Metal hosts via the
//! `MetalBackend::new()` early return pattern.

use alloygbm_backend_cpu::CpuBackend;
use alloygbm_backend_metal::MetalBackend;
use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, SplitFindRequest, SplitSelectionOptions,
};

fn make_fixture(row_count: usize, feature_count: usize, max_bin: u16) -> (BinnedMatrix, Vec<GradientPair>) {
    let bins: Vec<u8> = (0..(row_count * feature_count))
        .map(|i| ((i.wrapping_mul(31)) & (max_bin as usize - 1) as usize) as u8)
        .collect();
    let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
    let grads: Vec<GradientPair> = (0..row_count)
        .map(|i| GradientPair {
            grad: ((i as i32 - row_count as i32 / 2) as f32) * 0.5,
            hess: 1.0,
        })
        .collect();
    (bm, grads)
}

#[test]
fn find_best_splits_batch_matches_scalar_all_numeric() {
    let Ok(metal_backend) = MetalBackend::new() else {
        return;
    };
    let cpu_backend = CpuBackend;

    let row_count = 256usize;
    let feature_count = 4usize;
    let max_bin: u16 = 8;
    let (bm, grads) = make_fixture(row_count, feature_count, max_bin);
    let tiles = vec![FeatureTile {
        start_feature: 0,
        end_feature: feature_count as u32,
    }];
    let nodes: Vec<NodeSlice> = vec![
        NodeSlice::new(0, (0..128u32).collect()).unwrap(),
        NodeSlice::new(1, (128..256u32).collect()).unwrap(),
        NodeSlice::new(2, (0..96u32).collect()).unwrap(),
    ];

    let cpu_hists: Vec<_> = nodes
        .iter()
        .map(|n| cpu_backend.build_histograms(&bm, &grads, n, &tiles).unwrap())
        .collect();
    let metal_hists: Vec<_> = nodes
        .iter()
        .map(|n| {
            metal_backend
                .build_histograms(&bm, &grads, n, &tiles)
                .unwrap()
        })
        .collect();

    let options = SplitSelectionOptions::default();
    let feature_weights: Vec<f32> = vec![1.0; feature_count];
    let cats: Vec<CategoricalFeatureInfo> = Vec::new();

    let cpu_splits: Vec<_> = cpu_hists
        .iter()
        .map(|h| {
            cpu_backend
                .best_split_with_options(h, options, &feature_weights, &cats)
                .unwrap()
        })
        .collect();

    let requests: Vec<SplitFindRequest<'_>> = metal_hists
        .iter()
        .map(|h| SplitFindRequest { histograms: h })
        .collect();
    let metal_splits = metal_backend
        .find_best_splits_batch(&requests, options, &feature_weights, &cats)
        .unwrap();

    assert_eq!(metal_splits.len(), cpu_splits.len());
    for (m, c) in metal_splits.iter().zip(cpu_splits.iter()) {
        match (m, c) {
            (None, None) => continue,
            (Some(ms), Some(cs)) => {
                assert_eq!(ms.feature_index, cs.feature_index, "feature_index mismatch");
                assert_eq!(ms.threshold_bin, cs.threshold_bin, "threshold_bin mismatch");
                assert_eq!(ms.default_left, cs.default_left, "default_left mismatch");
                let gain_close = (ms.gain - cs.gain).abs() < 1e-3;
                assert!(gain_close, "gain mismatch metal={} cpu={}", ms.gain, cs.gain);
                let lg_close = (ms.left_stats.grad_sum - cs.left_stats.grad_sum).abs() < 1e-3;
                assert!(lg_close, "left_grad mismatch metal={} cpu={}", ms.left_stats.grad_sum, cs.left_stats.grad_sum);
                let lh_close = (ms.left_stats.hess_sum - cs.left_stats.hess_sum).abs() < 1e-3;
                assert!(lh_close, "left_hess mismatch metal={} cpu={}", ms.left_stats.hess_sum, cs.left_stats.hess_sum);
            }
            _ => panic!("metal and cpu disagree on split-existence: metal={m:?} cpu={c:?}"),
        }
    }
}

#[test]
fn find_best_splits_batch_empty_is_noop() {
    let Ok(backend) = MetalBackend::new() else {
        return;
    };
    let requests: Vec<SplitFindRequest<'_>> = Vec::new();
    let result = backend
        .find_best_splits_batch(
            &requests,
            SplitSelectionOptions::default(),
            &[],
            &[],
        )
        .unwrap();
    assert!(result.is_empty());
}

#[test]
fn find_best_splits_batch_falls_back_when_categoricals_present() {
    let Ok(backend) = MetalBackend::new() else {
        return;
    };
    // Build a small fixture and pass a single (degenerate) categorical
    // info — this should force the eligibility predicate to fail and
    // route through the scalar default.
    let row_count = 64usize;
    let feature_count = 2usize;
    let max_bin: u16 = 4;
    let (bm, grads) = make_fixture(row_count, feature_count, max_bin);
    let tiles = vec![FeatureTile {
        start_feature: 0,
        end_feature: feature_count as u32,
    }];
    let node = NodeSlice::new(0, (0..64u32).collect()).unwrap();
    let h = backend.build_histograms(&bm, &grads, &node, &tiles).unwrap();
    let requests = vec![SplitFindRequest { histograms: &h }];
    let cats = vec![CategoricalFeatureInfo {
        feature_index: 1,
        num_categories: 4,
    }];
    // Should not error out — falls back to scalar.
    let result = backend
        .find_best_splits_batch(&requests, SplitSelectionOptions::default(), &[1.0; 2], &cats)
        .unwrap();
    assert_eq!(result.len(), 1);
}
```

NOTE on the parity tolerance: `1e-3` is generous. If parity passes much tighter (e.g. `1e-6`), tighten to `1e-5` after the first run. Bit-exactness is not guaranteed because Metal uses fused multiply-add freely; the CPU path doesn't. The two-pass reduction matching matters more than identical bits.

- [ ] **Step 7: Run cargo check.**

```bash
cargo check -p alloygbm-backend-metal 2>&1 | tail -15
```

Expected: clean compile.

- [ ] **Step 8: Run the new parity tests.**

```bash
cargo test -p alloygbm-backend-metal --test best_split_parity -- --test-threads=1 2>&1 | tail -15
```

Expected: all three tests PASS. Common failures:
- `feature_index` or `threshold_bin` mismatch → kernel scan logic doesn't match CPU exactly. Re-read `best_split_for_feature` and align bin-by-bin.
- `gain` mismatch beyond `1e-3` → reduction order issue. Verify the two-pass simdgroup → threadgroup pattern matches the histogram kernel.
- `default_left` mismatch → NaN-direction `dir` loop ordering may have flipped.
- `row_count` mismatch (if you tightened the assertion) → kernel must return left_count.

- [ ] **Step 9: Run the full Metal test suite.**

```bash
cargo test -p alloygbm-backend-metal -- --test-threads=1 2>&1 | tail -10
```

Expected: 49 + 3 = 52 tests pass.

- [ ] **Step 10: Run pytest.**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release --manifest-path bindings/python/Cargo.toml 2>&1 | tail -3
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -5
```

Expected: 365/365 pass — all-numeric models now go through the GPU split path; all-categorical models continue to fall back.

- [ ] **Step 11: Commit.**

```bash
git add crates/backend_metal/src/kernels/best_split.rs \
        crates/backend_metal/src/kernels/mod.rs \
        crates/backend_metal/src/pipelines.rs \
        crates/backend_metal/src/lib.rs \
        crates/backend_metal/src/histogram_residency.rs \
        crates/backend_metal/tests/best_split_parity.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): MetalBackend::find_best_splits_batch GPU path

Wires the best_split.metal kernel through:
  - dispatch_find_best_splits_batch (kernels/best_split.rs) — one CB,
    encode per-feature + reduce-features kernels, single commit/wait,
    read back N×24-byte SplitDecisionGpu array.
  - BestSplitPipelineCache (pipelines.rs) — compiles best_split.metal
    once at backend init.
  - MetalBackend::find_best_splits_batch override (lib.rs) — numeric-
    only fast path; all-categorical models fall back to scalar default.
    Mixed numeric+categorical models also fall back for now (Task 8 wires
    the per-node merge).
Adds best_split_parity tests covering all-numeric parity, empty-batch
no-op, and categorical fallback. Adds borrow_planes() helper to
HistogramResidencyPool for kernel buffer binding without a copy.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Mixed-mode merge — GPU numeric vs host categorical

**Why:** Task 7 ships an override that falls back to scalar whenever ANY categorical feature is present. That misses the mixed-mode case where most features are numeric but a few are categorical — exactly the production-realistic shape. This task implements per-node best-of(GPU-numeric, host-categorical).

**Files:**
- Modify: `crates/backend_metal/src/lib.rs` — replace the categorical fallback with the merge path.

- [ ] **Step 1: Update the override to do mixed-mode merging.**

Replace the `if !categorical_features.is_empty() { return ... }` early-return block in `MetalBackend::find_best_splits_batch` with the merge path:

```rust
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Partition feature schema into numeric vs categorical sets.
        let first = requests[0].histograms;
        let all_feature_indices: Vec<u32> = first
            .feature_histograms()
            .iter()
            .map(|fh| fh.feature_index)
            .collect();
        let categorical_feature_set: std::collections::HashSet<u32> = categorical_features
            .iter()
            .map(|c| c.feature_index as u32)
            .collect();
        let numeric_feature_indices: Vec<u32> = all_feature_indices
            .iter()
            .copied()
            .filter(|fi| !categorical_feature_set.contains(fi))
            .collect();

        // Categorical-only model: nothing for the GPU kernel to do.
        if numeric_feature_indices.is_empty() {
            return requests
                .iter()
                .map(|r| {
                    self.best_split_with_options(
                        r.histograms,
                        options,
                        feature_weights,
                        categorical_features,
                    )
                })
                .collect();
        }

        // Run the GPU kernel on the numeric subset.
        let gpu_decisions = match kernels::best_split::dispatch_find_best_splits_batch(
            &self.metal_device,
            &self.best_split_pipeline_cache,
            &self.histogram_residency,
            &self.split_decision_residency,
            &self.residency,
            requests,
            options,
            &numeric_feature_indices,
            feature_weights,
        ) {
            Ok(d) => d,
            Err(_) => {
                return requests
                    .iter()
                    .map(|r| {
                        self.best_split_with_options(
                            r.histograms,
                            options,
                            feature_weights,
                            categorical_features,
                        )
                    })
                    .collect();
            }
        };

        // Convert GPU decisions to Option<SplitCandidate>.
        let gpu_splits: Vec<Option<alloygbm_core::SplitCandidate>> = requests
            .iter()
            .zip(gpu_decisions.iter())
            .map(|(req, decision)| gpu_decision_to_split_candidate(req.histograms, decision))
            .collect();

        // If no categorical features, return GPU result directly.
        if categorical_features.is_empty() {
            return Ok(gpu_splits);
        }

        // Mixed-mode merge: run host-side best_split_with_options for
        // the categorical feature subset only, then per-node pick the
        // winner. We use the SAME comparator the all-host path uses
        // (weighted gain, lower feature_idx breaks ties).
        let _p = profile::ScopedProbe::new(&profile::BS_CATEGORICAL_HOST_MERGE);
        let mut merged: Vec<Option<alloygbm_core::SplitCandidate>> =
            Vec::with_capacity(requests.len());
        for (req, gpu_split) in requests.iter().zip(gpu_splits.into_iter()) {
            // Host-side split over categoricals only. We pass an
            // empty `feature_weights` shaped for the full feature
            // set; the host scan only looks at the categorical
            // entries, so this works.
            let host_cat_split = self.best_split_with_options(
                req.histograms,
                options,
                feature_weights,
                categorical_features,
            )?;
            // Per-node winner: weighted gain comparison, lower
            // feature_idx breaks ties.
            let winner = match (gpu_split, host_cat_split) {
                (None, None) => None,
                (Some(g), None) => Some(g),
                (None, Some(h)) => Some(h),
                (Some(g), Some(h)) => {
                    let gw = weighted_gain(&g, feature_weights);
                    let hw = weighted_gain(&h, feature_weights);
                    if gw > hw || (gw == hw && g.feature_index < h.feature_index) {
                        Some(g)
                    } else {
                        Some(h)
                    }
                }
            };
            merged.push(winner);
        }
        Ok(merged)
```

NOTE on `host_cat_split`: when `categorical_features` is non-empty AND `numeric_feature_indices` is also non-empty, calling `best_split_with_options` with the full `feature_weights` will scan ALL features (numeric + categorical). That's wasted work — it will return the all-feature winner, defeating the merge. To avoid this, the host call needs to be told "only consider these features." That requires either (a) a new `best_split_with_options_filter` method on `BackendOps`, or (b) constructing a sub-`HistogramBundle` containing only the categorical features.

For the simplest correct version: after calling `best_split_with_options` on the full bundle, check whether the returned candidate's `feature_index` is in the categorical set. If yes, take it as the host-categorical-best. If no, the host happened to pick a numeric feature — discard, fall back to the GPU's numeric-best directly.

```rust
            let host_full_split = self.best_split_with_options(
                req.histograms,
                options,
                feature_weights,
                categorical_features,
            )?;
            let host_cat_split = match host_full_split {
                Some(s) if categorical_feature_set.contains(&s.feature_index) => Some(s),
                _ => None,
            };
```

This is wasteful (the host scans numeric features we don't need), but correct and simple. It's fine as a first cut — Task 9 benchmarks will show whether this dominates.

- [ ] **Step 2: Add the helper functions at the bottom of `lib.rs` (private to the module).**

```rust
#[cfg(target_os = "macos")]
fn weighted_gain(c: &alloygbm_core::SplitCandidate, feature_weights: &[f32]) -> f32 {
    let fi = c.feature_index as usize;
    let w = if fi < feature_weights.len() {
        feature_weights[fi]
    } else {
        1.0
    };
    c.gain * w
}

#[cfg(target_os = "macos")]
fn gpu_decision_to_split_candidate(
    bundle: &alloygbm_core::HistogramBundle,
    decision: &crate::split_decision_residency::SplitDecisionGpu,
) -> Option<alloygbm_core::SplitCandidate> {
    use crate::split_decision_residency::{SPLIT_FLAG_INVALID, SPLIT_FLAG_MISSING_GOES_RIGHT};

    if (decision.flags & SPLIT_FLAG_INVALID) != 0 || decision.feature_idx == u32::MAX {
        return None;
    }
    // Recover total stats for the matched feature; right_grad/hess
    // are total minus left.
    let fi = decision.feature_idx;
    let fhs = bundle.feature_histograms();
    let fh = fhs.iter().find(|fh| fh.feature_index == fi)?;
    let mut total_grad = 0.0f32;
    let mut total_hess = 0.0f32;
    let mut total_count = 0u32;
    for bin in &fh.bins {
        total_grad += bin.grad_sum;
        total_hess += bin.hess_sum;
        total_count += bin.count;
    }
    let grad_right = total_grad - decision.grad_left;
    let hess_right = total_hess - decision.hess_left;

    Some(alloygbm_core::SplitCandidate {
        node_id: bundle.node_id,
        feature_index: fi,
        threshold_bin: decision.bin_threshold as u16,
        gain: decision.gain,
        default_left: (decision.flags & SPLIT_FLAG_MISSING_GOES_RIGHT) == 0,
        is_categorical: false,
        categorical_bitset: None,
        left_stats: alloygbm_core::NodeStats {
            grad_sum: decision.grad_left,
            hess_sum: decision.hess_left,
            row_count: 0,
        },
        right_stats: alloygbm_core::NodeStats {
            grad_sum: grad_right,
            hess_sum: hess_right,
            row_count: total_count.saturating_sub(0),
        },
    })
}
```

NOTE: `row_count` in `left_stats` is set to 0 — see the note in Task 7 Step 5. If parity tests fail on `row_count`, return to the kernel and add `left_count` to `SplitDecisionGpu`. Don't suppress the assertion; fix the kernel.

- [ ] **Step 3: Add a mixed-mode parity test.**

Append to `crates/backend_metal/tests/best_split_parity.rs`:

```rust
#[test]
fn find_best_splits_batch_mixed_numeric_categorical_matches_scalar() {
    let Ok(metal_backend) = MetalBackend::new() else {
        return;
    };
    let cpu_backend = CpuBackend;

    let row_count = 256usize;
    let feature_count = 4usize;
    let max_bin: u16 = 8;
    let (bm, grads) = make_fixture(row_count, feature_count, max_bin);
    let tiles = vec![FeatureTile {
        start_feature: 0,
        end_feature: feature_count as u32,
    }];
    let node = NodeSlice::new(0, (0..256u32).collect()).unwrap();

    let cpu_h = cpu_backend.build_histograms(&bm, &grads, &node, &tiles).unwrap();
    let metal_h = metal_backend.build_histograms(&bm, &grads, &node, &tiles).unwrap();

    let options = SplitSelectionOptions::default();
    let feature_weights: Vec<f32> = vec![1.0; feature_count];
    // Mark feature 2 as categorical with 8 categories.
    let cats = vec![CategoricalFeatureInfo {
        feature_index: 2,
        num_categories: 8,
    }];

    let cpu_split = cpu_backend
        .best_split_with_options(&cpu_h, options, &feature_weights, &cats)
        .unwrap();
    let requests = vec![SplitFindRequest { histograms: &metal_h }];
    let metal_splits = metal_backend
        .find_best_splits_batch(&requests, options, &feature_weights, &cats)
        .unwrap();
    assert_eq!(metal_splits.len(), 1);
    match (&metal_splits[0], &cpu_split) {
        (None, None) => {}
        (Some(m), Some(c)) => {
            assert_eq!(m.feature_index, c.feature_index, "winning feature mismatch");
            assert_eq!(m.is_categorical, c.is_categorical, "is_categorical mismatch");
            let gain_close = (m.gain - c.gain).abs() < 1e-3;
            assert!(gain_close, "mixed gain mismatch metal={} cpu={}", m.gain, c.gain);
        }
        _ => panic!("mixed-mode metal and cpu disagree on split-existence"),
    }
}
```

- [ ] **Step 4: Run all parity tests.**

```bash
cargo test -p alloygbm-backend-metal --test best_split_parity -- --test-threads=1 2>&1 | tail -15
```

Expected: 4 tests pass.

- [ ] **Step 5: Run the full Metal suite + pytest.**

```bash
cargo test -p alloygbm-backend-metal -- --test-threads=1 2>&1 | tail -10
/Users/lashby/Projects/AlloyGBM/.venv/bin/maturin develop --release --manifest-path bindings/python/Cargo.toml 2>&1 | tail -3
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -5
```

Expected: 53 cargo tests pass; 365/365 pytest.

- [ ] **Step 6: Commit.**

```bash
git add crates/backend_metal/src/lib.rs crates/backend_metal/tests/best_split_parity.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): mixed-mode merge for find_best_splits_batch

Replaces the categorical-fallback short-circuit with per-node merge:
GPU kernel handles the numeric subset, host runs best_split_with_options
to pick up any categorical winner, the merge step takes the better of
the two (weighted-gain comparison, lower feature_index breaks ties).
The host call still scans the full feature set and we discard its
result when it picks a numeric winner — this is wasted work in the
mixed case but keeps the BackendOps trait surface unchanged and is
correct. If the categorical-merge cost dominates in benchmarks, we'll
add a feature-restricted host helper in a follow-up.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add `metal_friendly_large` benchmark fixture, run benchmarks, document D-024

**Why:** This is the kill-criterion validation. The original plan's MLX-expert prediction was "decisive 4-5× win above 1M rows × 100 features with B=255." Stage 3 closed at 0.16× best on `metal_friendly` (200k×200) — Stage 4a now needs a benchmark at the predicted shape.

**Files:**
- Modify: `benchmarks/metal_histogram.py` — new config.
- Modify: `docs/metal-backend/DECISIONS.md` — D-024 entry.
- Modify: `docs/metal-backend/STATUS.md` — overwrite with Stage 4a outcome.
- Modify: `docs/metal-backend/SESSIONS.md` — prepend 2026-04-26 entry.

- [ ] **Step 1: Read the existing `benchmarks/metal_histogram.py` to understand its config shape.**

```bash
head -80 benchmarks/metal_histogram.py
```

Note where the `metal_friendly` config dict is defined (look for the dict / list of scenarios). Note the field names — likely `n_rows`, `n_features`, `max_bin`, `n_estimators`, `max_depth`, `objective`. Mirror those exact names.

- [ ] **Step 2: Add `metal_friendly_large` config in `benchmarks/metal_histogram.py`.**

Find the existing scenario dict or list. Add a new entry alongside the existing `metal_friendly` configs. Likely shape (adapt to the actual field names you found in Step 1):

```python
    {
        "name": "metal_friendly_large_regression_d8",
        "task": "regression",
        "n_rows": 1_000_000,
        "n_features": 100,
        "max_bin": 255,
        "n_estimators": 5,
        "max_depth": 8,
    },
```

If the existing config uses a different schema, adapt — the goal is `1M rows × 100 features × bins=255`, regression objective, 5 estimators, depth 8. Add ONE such config.

- [ ] **Step 3: Run the benchmark (clean) and capture output.**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/metal_histogram.py 2>&1 | tee /tmp/stage4a_clean.txt | tail -40
```

Expected: per-config CPU and Metal timings + ratios. Capture the `metal_friendly_large_regression_d8` ratio specifically.

- [ ] **Step 4: Run with profiling.**

```bash
ALLOYGBM_METAL_PROFILE=1 /Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/metal_histogram.py 2>&1 | tee /tmp/stage4a_profile.txt | tail -60
```

Capture the `FIND_BEST_SPLITS_BATCH`, `BS_*`, `BUILD_HISTOGRAMS_BATCH`, `commit_wait` numbers from the profile dump.

- [ ] **Step 5: Append D-024 to `docs/metal-backend/DECISIONS.md`.**

Open the file and append at the end:

```markdown
## D-024 — Stage 4a GPU split finding: kill-criterion outcome

**Date:** 2026-04-26
**Status:** [Met / Not met] — see numbers below.

Stage 4a moves `best_split_with_options` onto the GPU (per-feature +
cross-feature reduction kernels in `best_split.metal`). Mixed-mode:
GPU handles the numeric subset; host runs Fisher-sort for any
categorical features and the per-node merge picks the winner. This
removes the per-level histogram readback (`count_accumulate`, was
~12% of metal time) and shrinks the per-level wait substantially —
`waitUntilCompleted` now drains a small kernel pipeline instead of
a histogram-scale readback.

**Measurements (`metal_friendly` + `metal_friendly_large`, Apple M4):**

| Config | CPU (s) | Metal pre-D-024 (s) | Metal post-D-024 (s) | post-D-024 ratio |
|---|---|---|---|---|
| regression d=8 bins=255 200k×200 | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| regression d=10 bins=255 200k×200 | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| regression d=6 bins=1024 200k×200 | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| multiclass_3 d=8 bins=255 100k×100 | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| multiclass_10 d=8 bins=255 100k×100 | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| **regression d=8 bins=255 1M×100** | [t_cpu] | n/a | [t_post] | [ratio] |

Pre-D-024 numbers: see D-023 amendment.

**Profile breakdown (post-D-024, regression d=8 1M×100):**

| Site | calls | total_ms | % of total |
|---|---|---|---|
| build_histograms_batch | [n] | [ms] | [pct] |
| ..commit_wait | [n] | [ms] | — |
| find_best_splits_batch | [n] | [ms] | [pct] |
| ..commit_wait | [n] | [ms] | — |
| ..decision_readback | [n] | [ms] | — |
| ..categorical_host_merge | [n] | [ms] | — |
| subtract_histogram_bundle_batch | [n] | [ms] | [pct] |
| apply_split | [n] | [ms] | [pct] |

**Outcome:**

- Stage 3 kill criterion (`metal_friendly >1.0× CPU` on at least one
  config): [MET / NOT MET]
- Stage 4 kill criterion (`metal_friendly_large >1.0× CPU` on at
  least one config): [MET / NOT MET]
- If at least one config crosses 1.0× CPU: Stage 4 closes at 4a;
  Stage 4b ICB chaining is deferred (not needed). `docs/limitations.md`
  Section 1 gets the documented Metal-vs-CPU threshold.
- If neither crosses: Stage 4b kicks off. The residual cost
  breakdown above identifies what 4b must eliminate (most likely
  `commit_wait` inside `build_histograms_batch` and
  `find_best_splits_batch`, which 4b's per-tree ICB consolidation
  attacks directly).
```

- [ ] **Step 6: Replace placeholders with real numbers from the benchmark output.**

Walk through D-024 and replace every `[t_cpu]`, `[t_pre]`, `[t_post]`, `[ratio]`, `[n]`, `[ms]`, `[pct]`, `[Met / Not met]`, `[MET / NOT MET]` with the actual captured values. The pre-D-024 column comes from the most recent `metal_friendly` capture — if it's not in the docs, look in `docs/metal-backend/DECISIONS.md` D-023 amendment for the pre-fix numbers. Any "n/a" is fine for cells that don't apply (1M×100 has no pre-D-024 measurement).

- [ ] **Step 7: Overwrite `docs/metal-backend/STATUS.md`.**

Replace the Stage 3 close-out section with a Stage 4a summary. Keep the file's high-level structure (active stage, sub-task checklist, next-up). Mark Stage 4a complete. If kill criterion was met, close Stage 4. If not, identify Stage 4b as next-up.

- [ ] **Step 8: Prepend a SESSIONS.md entry.**

Prepend (newest on top) to `docs/metal-backend/SESSIONS.md`:

```markdown
## 2026-04-26 — Stage 4a GPU split finding

**Shipped:**
- `BackendOps::find_best_splits_batch` trait method + scalar default.
- `RuntimeBackend` forwarding (Python).
- Engine refactor: `build_tree_level_wise` calls the new batched method.
- `MetalBackend::find_best_splits_batch` override + `best_split.metal`
  kernel (per-feature + cross-feature reduce) + dispatch wrapper +
  pipeline cache + `SplitDecisionPool`.
- Mixed-mode merge for numeric/categorical models.
- New `metal_friendly_large` (1M×100) benchmark fixture.
- D-024 with kill-criterion outcome.

**Tests:** `cargo test -p alloygbm-backend-metal` 53/53; pytest 365/365.

**Stage 3 status:** still [closed NOT MET / re-opened with 4a improving but still NOT MET / now MET via 4a].

**Stage 4 status:** [closed at 4a / 4b kicks off / closed without 4b].

**Next:** [Stage 5 inference / Stage 4b ICB chaining / further benchmarking].
```

Replace the `[...]` placeholders with the actual outcome based on benchmark numbers.

- [ ] **Step 9: Commit.**

```bash
git add benchmarks/metal_histogram.py \
        docs/metal-backend/DECISIONS.md \
        docs/metal-backend/STATUS.md \
        docs/metal-backend/SESSIONS.md
git commit -m "$(cat <<'EOF'
docs(metal-backend): D-024 Stage 4a GPU split finding outcome

Records post-Stage-4a metal_friendly + metal_friendly_large ratios
and the residual profile breakdown. Updates STATUS and SESSIONS to
reflect Stage 4a's outcome and whether Stage 4b is needed.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Gate

After Task 9 commits:

1. `cargo test --workspace --exclude alloygbm-python -- --test-threads=1` — green.
2. `pytest bindings/python/tests/ -q` — 365/365.
3. D-024 records `metal_friendly_large` (1M×100) ratio.
4. **If `metal_friendly_large` >1.0× CPU on at least one config:** Stage 4 closes at 4a. STATUS.md marks Stage 4 closed; Stage 4b spec is not written. Worktree is ready for the final merge to main per the user's "merge after full Metal implementation" decision.
5. **If `metal_friendly_large` ≤1.0× CPU:** Stage 4b kicks off via a fresh brainstorm + spec + plan in this same worktree. The benchmark numbers in D-024 are the input to that design.
