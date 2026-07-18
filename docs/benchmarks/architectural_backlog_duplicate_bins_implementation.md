# Duplicate Bin Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant binned-matrix copies while retaining the column-major hot path and exact bin semantics.

**Architecture:** Migrate callers to `row_bin`/`col_bin`, remove legacy mirrored `Vec<u8>` fields, then make row-major adaptive storage optional. Default training uses column-major-only storage; a measured heuristic may retain dual layout only if a real row-first workload justifies it.

**Tech Stack:** Rust 1.92, adaptive u8/u16 `BinStorage`, existing categorical and missing-bin tests.

## Global Constraints

- Preserve u8/u16, NaN sentinel, and native-categorical remapping behavior.
- Do not change artifacts or Python estimator parameters.
- Keep all indexing safe under `unsafe_code = "forbid"`.
- Require at least 20% end-to-end RSS-delta reduction in both full shapes and
  both u8/u16 storage modes.

---

### Task 1: Eliminate Direct Legacy-Vector Reads

**Files:**
- Modify: `crates/core/src/binned.rs`
- Modify: all `rg '\.(bins|bins_col)\b' crates` call sites
- Test: `crates/core/src/tests/main.rs`

- [ ] **Step 1: Add u8/u16 accessor parity tests** covering ordinary, maximum, and missing bins.
- [ ] **Step 2: Replace direct reads with `row_bin`, `col_bin`, `has_col_major`, or typed slices exposed by `BinStorage`.**
- [ ] **Step 3: Run `cargo test --workspace` and confirm no direct legacy-vector reads remain.**
- [ ] **Step 4: Commit.**

Commit: `git commit -am "Route binned matrix reads through adaptive storage"`

### Task 2: Remove Mirrored u8 Storage

**Files:**
- Modify: `crates/core/src/binned.rs`
- Modify: `crates/core/src/validation.rs`
- Test: `crates/core/src/tests/main.rs`

**Interfaces:**
- `BinnedMatrix` retains `bins_adaptive` and `bins_col_adaptive`; delete `bins` and `bins_col`.
- `set_bin(row, feature, value)` updates every storage layout that is present.

- [ ] **Step 1: Add a failing allocation-size test** asserting u8 construction stores no more than two `row_count * feature_count` payloads and u16 construction stores no u8 mirror.
- [ ] **Step 2: Delete legacy vectors and update validation.**
- [ ] **Step 3: Run core/backend/engine tests and commit.**

Commit: `git commit -am "Remove legacy binned matrix mirrors"`

### Task 3: Make Row-Major Storage Optional

**Files:**
- Modify: `crates/core/src/binned.rs`
- Modify: `bindings/python/src/quantization.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Test: `crates/core/src/tests/main.rs`
- Test: `bindings/python/src/tests.rs`

**Interfaces:**

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinnedLayout {
    ColumnMajor,
    Dual,
}
```

- [ ] **Step 1: Add failing column-only construction and categorical `set_bin` tests.**
- [ ] **Step 2: Build column-major storage directly from dense values where possible; avoid constructing then transposing a retained row payload.**
- [ ] **Step 3: Make row-only kernels request `Dual` explicitly; default Python training to `ColumnMajor`.**
- [ ] **Step 4: Run the full suite and commit.**

Commit: `git commit -am "Default training to column-major bins"`

### Task 4: Verify Memory And Throughput

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario duplicate_bins \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [ ] **Step 1: Verify exact prediction digests, native training no worse than 3%, and bridge preparation at most 95% of baseline.**
- [ ] **Step 2: Verify RSS delta is at most 80% of baseline for both wide and tall cases.**
- [ ] **Step 3: Commit the benchmark evidence separately.**
