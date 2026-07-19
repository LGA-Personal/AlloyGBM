# Compact Predictor Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sparse heap-indexed runtime predictor slots with compact contiguous nodes and side tables while preserving the artifact format.

**Architecture:** Translate heap-local artifact ids into compact indices during `Predictor::from_artifact_bytes`. Store scalar routing fields inline and categorical bitsets/linear leaves in side tables. Traversal follows explicit child indices; SHAP continues consuming unchanged artifacts.

**Tech Stack:** Rust 1.92, existing artifact decoder, predictor crate, PyO3 native predictor handle.

**Status:** Completed by the compact predictor follow-up. The implementation uses private
`u32::MAX` sentinel indices behind typed accessors instead of `Option<u32>` fields because the
literal interface sketched below would exceed its own 48-byte inline-node budget. The resulting
runtime node is 40 bytes.

## Global Constraints

- Artifact bytes and public persistence remain unchanged.
- Preserve load-time contribution collapse and DART weight folding order.
- Keep `unsafe_code = "forbid"`.
- Resolve the pre-existing multiclass-DART persistence contract separately before using that combination as a parity gate.

---

### Task 1: Add Compact Runtime Types

**Files:**
- Modify: `crates/predictor/src/lib.rs`
- Test: `crates/predictor/src/lib.rs`

**Interfaces:**

```rust
struct CompactPredictorNode {
    feature_index: u32,
    threshold_bin: f32,
    left_child: Option<u32>,
    right_child: Option<u32>,
    left_leaf_value: f32,
    right_leaf_value: f32,
    categorical_index: Option<u32>,
    left_linear_index: Option<u32>,
    right_linear_index: Option<u32>,
    flags: u8,
}
```

- [x] **Step 1: Add failing size and side-table tests.** Require the inline node to remain at or below 48 bytes on 64-bit targets.
- [x] **Step 2: Add `CompactPredictorTree` with node, categorical, and linear side tables.**
- [x] **Step 3: Commit without switching production traversal.** The type and traversal changes landed together so the intermediate commit did not retain dead runtime types.

Commit: `git commit -am "Add compact predictor runtime types"`

### Task 2: Translate Heap Ids At Load Time

**Files:**
- Modify: `crates/predictor/src/lib.rs`
- Test: `crates/predictor/src/lib.rs`

- [x] **Step 1: Add a failing right-spine test** with local ids ending at 65,534 and assert compact node count equals populated stump count.
- [x] **Step 2: Build a temporary `BTreeMap<local_id, compact_index>`, validate duplicates/bounds, and resolve child indices.**
- [x] **Step 3: Populate side tables in deterministic stump order.**
- [x] **Step 4: Preserve feature validation and collapse contributions before discarding transient stumps.**
- [x] **Step 5: Run predictor tests and commit.**

Commit: `git commit -am "Load artifacts into compact predictor trees"`

### Task 3: Switch Traversal And Pin Parity

**Files:**
- Modify: `crates/predictor/src/lib.rs`
- Test: `crates/predictor/src/lib.rs`
- Test: `bindings/python/tests/test_persistence.py`
- Test: `bindings/python/tests/test_shap_additivity_tolerance.py`

- [x] **Step 1: Add exact scalar tests** for dense/batch APIs, NaN/default routing, categorical unknowns, objective transforms, and DART.
- [x] **Step 2: Add tolerance-preserving linear, multiclass, and SHAP-additivity tests.**
- [x] **Step 3: Replace heap arithmetic traversal with explicit compact child indices and side-table lookup.**
- [x] **Step 4: Delete `nodes_by_local_id` only after all predictor paths compile.**
- [x] **Step 5: Run full Rust/Python suites and commit.**

Commit: `git commit -am "Traverse compact predictor nodes"`

### Task 4: Verify The Sparse-Spine Benchmark

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario compact_nodes \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [x] **Step 1: Confirm exact artifact and prediction digests.**
- [x] **Step 2: Require at least 75% loaded-RSS reduction, 15% deep-spine throughput improvement, and no shallow-control regression above 5%.**
- [x] **Step 3: Commit evidence separately.**
