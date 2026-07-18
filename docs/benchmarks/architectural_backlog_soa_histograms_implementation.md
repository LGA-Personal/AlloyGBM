# SoA Histogram Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep scalar histogram statistics in structure-of-arrays storage through split selection and avoid computing `grad_sq_sum` when DRO is inactive.

**Architecture:** Replace the materialized `Vec<FeatureHistogram>` handoff with borrowed per-feature views over `HistogramArena`. Make squared-gradient storage optional and retain the existing PL histogram bundle as a separate path. Preserve split iteration order and emitted artifacts exactly.

**Tech Stack:** Rust 1.92, Rayon, existing `alloygbm-core`, `alloygbm-backend-cpu`, and workspace tests.

## Global Constraints

- Keep `unsafe_code = "forbid"`.
- Do not change the artifact format or public Python parameters.
- Preserve exact prediction digests for scalar, DRO, and linear-leaf benchmark cases.
- Run the full same-host `soa_histograms` benchmark before claiming a speedup.

---

### Task 1: Define The Borrowed Histogram Contract

**Files:**
- Modify: `crates/core/src/histogram.rs`
- Modify: `crates/backend_cpu/src/arena.rs`
- Test: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Produces: `HistogramFeatureView<'a>` with `grad_sums`, `hess_sums`, optional `grad_sq_sums`, and `counts` slices.
- Produces: `HistogramBundleView<'a>` that resolves a feature index without allocation.

- [ ] **Step 1: Write failing layout/view tests**

```rust
#[test]
fn histogram_feature_view_reads_aligned_soa_bins() {
    let arena = fixture_histogram_arena();
    let view = arena.bundle_view(2, false).feature(1).expect("feature");
    assert_eq!(view.grad_sums(), &[3.0, 4.0]);
    assert_eq!(view.hess_sums(), &[5.0, 6.0]);
    assert_eq!(view.counts(), &[7, 8]);
    assert!(view.grad_sq_sums().is_none());
}
```

- [ ] **Step 2: Run the focused test and confirm it fails because the view API is absent**

Run: `cargo test -p alloygbm-backend-cpu histogram_feature_view_reads_aligned_soa_bins`

- [ ] **Step 3: Add the view types and checked slice construction**

Use `feature_index * bin_count..(feature_index + 1) * bin_count`; return a contract error for inconsistent slice lengths. Represent squared gradients as `Option<&[f32]>`.

- [ ] **Step 4: Run backend tests and commit**

Run: `cargo test -p alloygbm-backend-cpu`

Commit: `git commit -am "Add borrowed SoA histogram views"`

### Task 2: Gate Squared-Gradient Accumulation

**Files:**
- Modify: `crates/backend_cpu/src/arena.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/backend_cpu/src/backend_ops.rs`
- Test: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Consumes: `SplitSelectionOptions::dro_config`.
- Produces: `HistogramArena::prepare(feature_count, bin_count, include_grad_sq)`.

- [ ] **Step 1: Add a failing standard-path test** proving the arena has no squared-gradient buffer when `dro_config` is absent and a DRO test proving the values match the current reducer.
- [ ] **Step 2: Run both tests and confirm the standard case fails on the unconditional allocation.**
- [ ] **Step 3: Change accumulation kernels to branch once per histogram build, not once per row/bin.** Keep separate with-DRO and without-DRO loops so the standard inner loop performs no square or extra write.
- [ ] **Step 4: Run `cargo test -p alloygbm-backend-cpu` and commit.**

Commit: `git commit -am "Skip squared gradients outside DRO"`

### Task 3: Scan SoA Views Directly

**Files:**
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/backend_cpu/src/split_helpers.rs`
- Modify: `crates/engine/src/traits.rs`
- Test: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**
- Changes `BackendOps::build_histograms` to return an owned SoA bundle whose feature views borrow from that bundle.
- Keeps PL statistics and native-categorical bitsets in explicit side structures.

- [ ] **Step 1: Add an equivalence test** that runs every numeric threshold and both missing directions through the old materialized reference helper and new SoA scanner.
- [ ] **Step 2: Verify RED by calling the not-yet-existing SoA scanner.**
- [ ] **Step 3: Port prefix scans and histogram subtraction to aligned slices.** Preserve feature order, threshold order, and tie-breaking.
- [ ] **Step 4: Delete `materialize_tile_histograms` only after all call sites use views.**
- [ ] **Step 5: Run `cargo test --workspace` and commit.**

Commit: `git commit -am "Scan SoA histograms without materialization"`

### Task 4: Verify Special Modes And Performance

**Files:**
- Test: `bindings/python/tests/test_dro.py`
- Test: `bindings/python/tests/test_linear_leaves.py`
- Test: `crates/backend_cpu/src/tests/main.rs`

- [ ] **Step 1: Add deterministic artifact/prediction parity tests** for standard, DRO, PL, native-categorical, MorphBoost, and factor-neutral gain dispatch.
- [ ] **Step 2: Run `cargo fmt --all --check`, clippy, workspace tests, and the Python suite.**
- [ ] **Step 3: Run the full benchmark comparison.**

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario soa_histograms \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [ ] **Step 4: Commit benchmark evidence and resolution update separately from production code.**
