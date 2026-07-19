# Node-Level Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize expensive work across active nodes at a level while preserving deterministic tree structure and sibling histogram subtraction.

**Architecture:** Compute immutable per-node proposals in parallel, then commit accepted splits in ascending local-node order. Disable nested histogram Rayon work inside the node-parallel region; keep the root and single-node levels on the existing per-node parallel path.

**Tech Stack:** Rust 1.92, Rayon, existing engine/backend traits and deterministic artifact tests.

## Global Constraints

- Start only after histogram and partition ownership can be moved into independent node work items.
- Keep `unsafe_code = "forbid"` and do not mutate shared prediction buffers from Rayon workers.
- Preserve artifact bytes across repeated runs at the same thread count; compare
  cross-thread predictions with the existing tight numerical tolerance.
- Do not change leaf-wise growth in this project.

---

### Task 1: Separate Node Proposal From Ordered Commit

**Files:**
- Modify: `crates/engine/src/trainer/tree_build.rs`
- Test: `crates/engine/src/tests/main.rs`

**Interfaces:**
- Produces: private `LevelNodeProposal` containing local id, split, partition, child statistics, child histograms, and deferred leaf updates.
- Produces: `propose_level_node(...) -> EngineResult<Option<LevelNodeProposal>>` with no shared mutation.

- [x] **Step 1: Add an ordering guard** that reverses node outcomes and verifies commit order by local id.
- [x] **Step 2: Extract proposal computation without changing the sequential loop.**
- [x] **Step 3: Sort proposals by `local_node_id` and apply candidate-prediction/stump updates sequentially.**
- [x] **Step 4: Run engine and workspace tests; commit.**

Commit: `git commit -am "Separate level node proposals from commit"`

### Task 2: Add Explicit Histogram Execution Policy

**Files:**
- Modify: `crates/engine/src/traits.rs`
- Modify: `crates/backend_cpu/src/backend_ops.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Test: `crates/backend_cpu/src/tests/main.rs`

**Interfaces:**

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistogramExecution {
    Sequential,
    Parallel,
}
```

- [x] **Step 1: Add a failing equivalence test** for sequential and parallel histogram execution.
- [x] **Step 2: Thread the policy through `BackendOps::build_histograms`.**
- [x] **Step 3: Ensure the sequential policy never enters a Rayon iterator; keep current thresholds under `Parallel`.**
- [x] **Step 4: Run backend tests and commit.**

Commit: `git commit -am "Make histogram execution policy explicit"`

### Task 3: Parallelize Multi-Node Levels

**Files:**
- Modify: `crates/engine/src/trainer/tree_build.rs`
- Test: `crates/engine/src/tests/main.rs`

- [x] **Step 1: Add repeated-run determinism tests at 1 and 8 threads** with depth 8 and tied candidate gains.
- [x] **Step 2: Use `active_nodes.into_par_iter()` only when at least two nodes have sufficient aggregate work.** Each worker uses `HistogramExecution::Sequential`.
- [x] **Step 3: Keep sibling subtraction local to the proposal that owns the parent histogram.**
- [x] **Step 4: Commit proposals in local-id order and run `cargo test --workspace`.**
- [x] **Step 5: Commit.**

Commit: `git commit -am "Parallelize level-wise node proposals"`

### Task 4: Verify Scaling

- [x] **Step 1: Run fmt, clippy, Rust, and Python suites.**
- [x] **Step 2: Run the full same-host node benchmark with 1 and 8 Rayon threads.**

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario node_parallelism \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [x] **Step 3: Reject the redesign if eight-thread native training does not improve by 15%, candidate 8-thread scaling is below 1.25x, or one-thread time regresses by more than 5%.**

## Outcome

Implemented at `29241f1` against baseline `519a9d4`. The full same-host candidate run improved
median eight-thread native training by 40.5%, reached 1.64x one-to-eight-thread scaling, and changed
single-thread native time by -3.0%. Predictions and RMSE matched exactly, repeated same-thread
artifacts were byte-stable, and all acceptance gates passed. Eight-thread incremental fit RSS rose
by 15.05 MiB because independent proposals own child histograms concurrently; this is recorded as
the accepted throughput tradeoff.
