# Stage 4b — Metal 4 ICB Chaining Design

**Date:** 2026-04-28
**Status:** Approved, ready for implementation planning
**Predecessor:** Stage 4a (GPU split finding) — closed 2026-04-28, kill criterion NOT MET (0.24×)

---

## Problem Statement

After Stage 4a, the dominant bottleneck is `build_histograms_batch.commit_wait` at **66.7% of total training time** (4626ms for 40 calls at 1M×100, regression d=8). The root cause is one synchronous `waitUntilCompleted` per tree level: the CPU encodes a histogram-build command buffer, commits it, blocks until the GPU finishes, reads back split decisions, then encodes the next level. For depth=8 that is 8 CPU stalls per tree.

Stage 4b eliminates all per-level stalls by pre-encoding the entire tree — all levels, all three GPU phases — into a single Metal `MTLIndirectCommandBuffer` submitted once per tree with exactly one `waitUntilCompleted`.

**Kill criterion for Stage 4b:** ≥ 1.0× CPU parity on the `metal_friendly_large` benchmark (1M×100, regression d=8).

---

## Hardware Targeting

- **Metal 4 path** (this spec): requires `MTLDevice.supportsFamily(.apple9)` — M4 chip, macOS 26+. Full ICB-per-tree chaining with Metal 4 inherited residency sets (`MTLHeap` passed via `use_heap`).
- **Metal 3 and below**: falls back to the Stage 4a dispatch path transparently. No API change, no performance regression.
- **Future Option A** (deferred, not in scope): bring ICB chaining to Metal 3 using argument-buffer resource binding instead of inherited residency sets. Noted for a future stage.

---

## Architecture Overview

One `MTLIndirectCommandBuffer` (ICB) is created per `MetalBackend` instance and re-encoded at the start of each tree. The ICB stores `depth × 3` compute commands — one histogram, one split-find, one partition command per level. An outer `MTLCommandBuffer` (CPU-encoded once per tree) sequences ICB execution ranges with `memoryBarrier` calls between phases and is submitted as a single unit.

**Per-tree execution flow:**
```
CPU: reset pool buffers (zero node_active, set row_node_id=0, node_active[0]=1)
CPU: encode ICB (depth × 3 commands, ~microseconds)
CPU: encode outer MTLCommandBuffer (depth × 3 executeCommandsInBuffer + barriers)
GPU: level 0 histogram → barrier → level 0 split-find → barrier → level 0 partition → barrier
GPU: level 1 histogram → barrier → level 1 split-find → barrier → level 1 partition → barrier
     ...
GPU: level depth-1 histogram → barrier → split-find → barrier → partition
CPU: waitUntilCompleted  ← ONE sync per tree (was depth syncs in Stage 4a)
CPU: read back split_decisions + leaf_values → build Vec<TreeNode>
```

**ICB path eligibility** (same gate as Stage 4a):
- Metal 4 device
- All features in batch are numeric (categoricals stay on Stage 4a path)
- `bin_count ≤ 1024`

The engine (`build_tree_level_wise` in `crates/engine/src/lib.rs`) requires **no changes**. Routing is entirely internal to `MetalBackend`.

---

## GPU Kernels

Three new shaders in `shaders/icb_tree.metal`. All share a push-constant struct:

```metal
struct IcbConstants {
    uint32_t row_count;
    uint32_t feature_count;
    uint32_t bin_count;
    uint32_t level_node_offset;  // global id of first node at this level (= 2^L - 1)
    uint32_t level_node_end;     // level_node_offset + 2^L
    uint32_t level_node_count;   // 2^L
    uint32_t min_rows_per_leaf;
    float    min_split_gain;
    float    lambda;             // L2 regularisation
    float    learning_rate;      // used for leaf value: -lr * grad / (hess + lambda)
    uint32_t _pad;               // align to 48 bytes
};
```

Node numbering: root = 0; node n's left child = 2n+1, right child = 2n+2. At level L, nodes are `(2^L - 1)` through `(2^(L+1) - 2)` inclusive (= `level_node_offset` through `level_node_end - 1`).

### `icb_histogram`

Accumulates grad/hess into the shared histogram buffer. One thread per row.

```metal
kernel void icb_histogram(
    device const uint16_t*  row_node_id  [[ buffer(0) ]],
    device const uint8_t*   node_active  [[ buffer(1) ]],
    device const float*     gradients    [[ buffer(2) ]],
    device const float*     hessians     [[ buffer(3) ]],
    device const uint8_t*   bin_data     [[ buffer(4) ]],  // column-major N×F
    device atomic_float*    histograms   [[ buffer(5) ]],  // [nodes × F × B × 2] f32
    constant IcbConstants&  c            [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= c.row_count) return;
    uint16_t node = row_node_id[gid];
    if (node < c.level_node_offset || node >= c.level_node_end) return;
    if (!node_active[node]) return;
    uint local_node = node - c.level_node_offset;
    for (uint f = 0; f < c.feature_count; f++) {
        uint bin = bin_data[gid * c.feature_count + f];
        uint base = (local_node * c.feature_count + f) * c.bin_count * 2;
        atomic_fetch_add_explicit(&histograms[base + bin * 2],     gradients[gid], memory_order_relaxed);
        atomic_fetch_add_explicit(&histograms[base + bin * 2 + 1], hessians[gid],  memory_order_relaxed);
    }
}
```

### `icb_split_find`

One thread per node at level L. Prefix-scans the histogram (same logic as `best_split_per_feature` in `best_split.metal`), finds best split, writes decision and activates/deactivates children.

```metal
kernel void icb_split_find(
    device const atomic_float*  histograms   [[ buffer(0) ]],
    device SplitDecision*       decisions    [[ buffer(1) ]],
    device uint8_t*             node_active  [[ buffer(2) ]],
    device float*               leaf_values  [[ buffer(3) ]],
    constant IcbConstants&      c            [[ buffer(4) ]],
    uint node [[ thread_position_in_grid ]]
) {
    if (node >= c.level_node_count) return;
    uint global_node = c.level_node_offset + node;
    if (!node_active[global_node]) return;
    // prefix-scan histogram to find best (feature, bin) by Newton gain
    // ...
    if (best_gain > c.min_split_gain && left_count >= c.min_rows_per_leaf
                                     && right_count >= c.min_rows_per_leaf) {
        decisions[global_node] = best_decision;
        node_active[2 * global_node + 1] = 1;  // left child
        node_active[2 * global_node + 2] = 1;  // right child
    } else {
        leaf_values[global_node] = -c.learning_rate * grad_sum / (hess_sum + c.lambda);
        // children remain inactive
    }
}
```

### `icb_partition`

One thread per row. Updates `row_node_id` based on the split decision for each row's current node.

```metal
kernel void icb_partition(
    device uint16_t*              row_node_id  [[ buffer(0) ]],
    device const uint8_t*         node_active  [[ buffer(1) ]],
    device const SplitDecision*   decisions    [[ buffer(2) ]],
    device const uint8_t*         bin_data     [[ buffer(3) ]],
    constant IcbConstants&        c            [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= c.row_count) return;
    uint16_t node = row_node_id[gid];
    if (node < c.level_node_offset || node >= c.level_node_end) return;
    if (!node_active[node]) return;
    SplitDecision d = decisions[node];
    uint bin = bin_data[gid * c.feature_count + d.feature_index];
    bool goes_left = (bin <= d.threshold_bin) ^ d.nan_goes_right;
    row_node_id[gid] = goes_left ? (2 * node + 1) : (2 * node + 2);
}
```

---

## Buffer Pre-allocation

All buffers are allocated in `IcbBufferPool::new()` at backend creation. They are reused across all trees and all estimators.

| Buffer | Element type | Count | Purpose |
|---|---|---|---|
| `row_node_id` | u16 | N | Current node for each training row. Reset to 0 each tree. |
| `node_active` | u8 | 2^depth_max | Whether node n is active. Node 0 = true at tree start. |
| `split_decisions` | `SplitDecision` (24 B) | 2^depth_max | Best split per node, written by `icb_split_find`. Read back at end of tree. |
| `leaf_values` | f32 | 2^depth_max | Leaf value for nodes that do not split. Written by `icb_split_find`. |
| `histograms` | f32 | 2^(depth_max-1) × F × B × 2 | **Reused per level** — level L overwrites level L-1's slots (safe due to barriers in outer encoder). |
| `gradients` | f32 | N | Uploaded from CPU before each tree. |
| `hessians` | f32 | N | Uploaded from CPU before each tree. |

All buffers are sub-allocated from a single `MTLHeap` so that `encoder.use_heap(pool.heap)` in the outer command encoder covers all of them (Metal 4 inherited residency pattern — eliminates per-buffer `use_resource` calls inside ICB commands).

**Memory budget at 1M×100, d=8, bins=255:**
- `row_node_id`: 2MB
- `histograms` (reused): 128 × 100 × 255 × 8 B = 26.1MB
- `split_decisions` + `leaf_values` + `node_active`: < 1MB
- `gradients` + `hessians`: 8MB
- **Total: ~37MB** — well within M4 unified memory headroom

**Histogram subtraction trick dropped for ICB path.** The existing `subtract_histogram_bundle_batch` (which saves histogram work by building only the smaller child and subtracting from parent) is not used in the ICB path — pre-encoding requires knowing which child is smaller before execution. Both children are built from scratch. Re-introducing it via a GPU-side "small-child selection" kernel is a noted future optimisation.

---

## ICB Encoding (Rust)

`IcbTreeEncoder` in `crates/backend_metal/src/kernels/icb_tree.rs`.

```rust
pub struct IcbBufferPool {
    pub row_node_id:     MetalBuffer,
    pub node_active:     MetalBuffer,
    pub split_decisions: MetalBuffer,
    pub leaf_values:     MetalBuffer,
    pub histograms:      MetalBuffer,
    pub gradients:       MetalBuffer,
    pub hessians:        MetalBuffer,
    pub heap:            MTLHeap,
    row_count:           usize,
    depth_max:           u8,
}

impl IcbBufferPool {
    pub fn reset_for_tree(&self) {
        // zero node_active, set row_node_id[0..N] = 0, set node_active[0] = 1
    }
    pub fn upload_gradients(&self, grads: &[f32], hess: &[f32]) { ... }
    pub fn read_split_decisions(&self, depth: u8) -> Vec<SplitDecisionCpu> { ... }
}

pub struct IcbTreeEncoder {
    icb:              MTLIndirectCommandBuffer,  // depth_max * 3 commands
    histogram_pso:    MTLComputePipelineState,
    split_find_pso:   MTLComputePipelineState,
    partition_pso:    MTLComputePipelineState,
    command_queue:    MTLCommandQueue,
    depth_max:        u8,
}

impl IcbTreeEncoder {
    pub fn encode_tree(&self, pool: &IcbBufferPool, params: &IcbTreeParams)
        -> MTLCommandBuffer
    {
        // 1. Build ICB: depth × 3 commands with per-level IcbConstants baked in
        for level in 0..params.depth {
            let node_offset = (1u32 << level) - 1;
            let node_count  =  1u32 << level;
            let constants = IcbConstants { ..., level_node_offset: node_offset,
                                                level_node_count:  node_count, ... };
            // histogram command: N threads
            let hist = self.icb.indirect_compute_command(level * 3 + 0);
            hist.set_compute_pipeline_state(&self.histogram_pso);
            // bind pool buffers + constants ...
            hist.concurrent_dispatch_threads(MTLSize(pool.row_count, 1, 1), ...);

            // split_find command: node_count threads
            let sf = self.icb.indirect_compute_command(level * 3 + 1);
            // ...
            sf.concurrent_dispatch_threads(MTLSize(node_count as u64, 1, 1), ...);

            // partition command: N threads
            let part = self.icb.indirect_compute_command(level * 3 + 2);
            // ...
            part.concurrent_dispatch_threads(MTLSize(pool.row_count, 1, 1), ...);
        }

        // 2. Outer MTLCommandBuffer: sequence ICB ranges with barriers
        let cmd_buf = self.command_queue.command_buffer();
        let encoder = cmd_buf.compute_command_encoder();
        encoder.use_heap(&pool.heap);  // covers all pool buffers

        for level in 0..params.depth {
            let base = (level * 3) as u64;
            encoder.execute_commands_in_buffer(&self.icb, NSRange { location: base,     length: 1 });
            encoder.memory_barrier_with_resources(&[&pool.histograms]);
            encoder.execute_commands_in_buffer(&self.icb, NSRange { location: base + 1, length: 1 });
            encoder.memory_barrier_with_resources(&[&pool.split_decisions, &pool.node_active]);
            encoder.execute_commands_in_buffer(&self.icb, NSRange { location: base + 2, length: 1 });
            encoder.memory_barrier_with_resources(&[&pool.row_node_id]);
        }
        encoder.end_encoding();
        cmd_buf
    }
}
```

**`IcbTreeParams`** — a plain struct constructed from `TrainParams` at the call site:

```rust
pub struct IcbTreeParams {
    pub depth:             u8,
    pub feature_count:     u32,
    pub bin_count:         u32,
    pub min_split_gain:    f32,
    pub lambda:            f32,
    pub learning_rate:     f32,
    pub min_rows_per_leaf: u32,
}
```

**Dispatch site in `MetalBackend`:**

```rust
fn build_tree_icb(&self, params: &TrainParams, grads: &[f32], hess: &[f32]) -> Result<Tree> {
    let encoder = self.icb_encoder.as_ref().unwrap();
    let pool    = self.icb_pool.as_ref().unwrap();
    pool.reset_for_tree();
    pool.upload_gradients(grads, hess);
    let cmd_buf = encoder.encode_tree(pool, &params.into());
    cmd_buf.commit();
    cmd_buf.wait_until_completed();  // ONE sync per tree
    let decisions = pool.read_split_decisions(params.max_depth);
    reconstruct_tree_from_decisions(&decisions, params)
}
```

**Metal 4 gate in `MetalBackend::new()`:**

```rust
let metal4 = device.supports_family(MTLGPUFamily::Apple9);
let (icb_encoder, icb_pool) = if metal4 {
    (Some(IcbTreeEncoder::new(&device, max_depth, ...)),
     Some(IcbBufferPool::new(&device, row_count, feature_count, max_depth, ...)))
} else {
    (None, None)
};
```

**Routing in `MetalBackend`'s `BackendOps` implementation:**

```rust
if self.icb_encoder.is_some() && all_numeric && bin_count <= 1024 {
    self.build_tree_icb(params, grads, hess)
} else {
    self.build_tree_stage4a(params, grads, hess)  // existing path unchanged
}
```

---

## Tree Reconstruction

After `wait_until_completed`, `pool.read_split_decisions(depth)` reads back the `split_decisions` and `leaf_values` buffers. `reconstruct_tree_from_decisions` walks the node numbering (root = 0, children = 2n+1 / 2n+2) and builds a `Vec<TreeNode>` in the same format as the CPU and Stage 4a paths. Nodes where `node_active[n] == 0` at their level become leaves using `leaf_values[n]`.

---

## Profile Counters

New counters (replacing `BS_DISPATCH` / `BS_COMMIT_WAIT` for the Metal 4 path):

| Counter | What it measures |
|---|---|
| `ICB_ENCODE` | CPU time to build the ICB + outer command buffer per tree |
| `ICB_SUBMIT` | Wall time from `commit()` to `wait_until_completed()` returning (GPU time) |
| `ICB_READBACK` | CPU time to read split decisions and reconstruct `TreeNode` structs |

---

## Testing

All tests in `crates/backend_metal/tests/icb_tree_parity.rs`.

1. **`icb_tree_matches_stage4a_small`** — regression, 50k×20, d=4, bins=64. Build one tree via ICB and one via Stage 4a; assert all `TreeNode` split fields match exactly, gain within 0.1, leaf values within 0.05.

2. **`icb_tree_matches_stage4a_deep`** — regression, 200k×50, d=8, bins=255. Same assertions. Exercises fully-populated level-wise tree.

3. **`icb_tree_prunes_correctly`** — regression with `min_split_gain = 1e4` forcing most nodes to become leaves at level 2. Asserts leaf count matches Stage 4a path (validates `node_active` masking).

4. **`full_estimator_icb_matches_stage4a`** — trains 5 estimators on the same data via each path, asserts final predictions agree within atol=0.05. Catches accumulated rounding drift across trees.

**Benchmark addition:** `metal_friendly_large_icb` scenario in `benchmarks/metal_histogram.py` — same 1M×100 d=8 config, forces ICB path, reports Stage 4a vs Stage 4b wall time and ratio vs CPU baseline.

---

## Future Work (out of scope)

- **Option A — Metal 3 ICB**: bring ICB chaining to M1/M2/M3 using argument-buffer resource binding instead of Metal 4 inherited residency sets. Requires a second resource-binding strategy but re-uses the same three kernels.
- **Histogram subtraction trick in ICB**: add a GPU-side "small-child selection" kernel that writes a flag, then use it to conditionally build only one child and subtract for the other. Requires one additional command per level.
- **GPU gradient computation**: move `compute_gradients` (currently CPU) onto the GPU to eliminate the gradient upload cost between trees. Would eliminate the last CPU↔GPU transfer per estimator.
