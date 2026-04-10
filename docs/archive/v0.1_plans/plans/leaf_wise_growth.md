# Plan: Leaf-Wise (Best-First) Tree Growth

## Status: Not Started

## Summary

AlloyGBM currently grows trees level-by-level (depth-first BFS through `active_nodes` in engine/src/lib.rs:1227). All nodes at depth `d` are split before moving to depth `d+1`. This is XGBoost's approach.

LightGBM's key innovation is **leaf-wise growth**: instead of splitting all nodes at a level, pick the single leaf with the highest gain across the entire tree and split it. This often produces more accurate trees with fewer leaves, because the tree "spends" its splits where they matter most.

This plan covers adding leaf-wise growth as an alternative tree building strategy.

---

## Questions to Resolve Before Starting

1. **Default behavior**: Should the default change to leaf-wise, or remain level-wise? Recommendation: remain level-wise by default for backward compatibility. Add `tree_growth: str = "level"` parameter with options `"level"` and `"leaf"`.

2. **`max_leaves` requirement**: Leaf-wise growth needs `max_leaves` as the primary stopping criterion (instead of `max_depth`). Should `max_depth` still apply as a secondary limit? Recommendation: yes -- `max_leaves` is primary, `max_depth` is a safety cap (LightGBM does this).

3. **Histogram reuse**: Leaf-wise growth benefits heavily from histogram caching (keep computed histograms for nodes that aren't split yet). Without caching, leaf-wise is slower than level-wise because histograms must be rebuilt each time a node is reconsidered. Recommendation: implement histogram caching alongside leaf-wise growth, or at minimum, implement the subtraction trick for deferred nodes.

---

## Architecture Overview

### Current Level-Wise Loop

```
for depth in 0..max_depth:
    for each active_node at this depth:
        build histograms
        find best split
        partition rows
    active_nodes = child nodes from all splits
```

### Target Leaf-Wise Loop

```
priority_queue = [root_node]
leaves_used = 1

while leaves_used < max_leaves and priority_queue is not empty:
    node = priority_queue.pop_max_gain()
    split node
    leaves_used += 1  # split one leaf into two = net +1
    for each child:
        build histograms (or use subtraction trick)
        find best split (stores gain for priority ordering)
        push child into priority_queue
```

### Key Differences

| Aspect | Level-Wise | Leaf-Wise |
|--------|-----------|-----------|
| Splitting order | All nodes at depth d | Best gain across all depths |
| Primary stop criterion | `max_depth` | `max_leaves` |
| Tree shape | Balanced | Potentially unbalanced |
| Histogram computation | All nodes at a level, then discard | Must cache or recompute for deferred nodes |
| Typical accuracy | Good | Often better for same leaf count |
| Overfitting risk | Lower | Higher (needs regularization) |

---

## Implementation

### Phase 1: Priority Queue Infrastructure

**`crates/engine/src/lib.rs`**

#### Step 1.1: Define a `PendingSplit` struct

```rust
struct PendingSplit {
    node_id: u32,
    row_indices: Vec<u32>,
    split_candidate: SplitCandidate,
    histograms: HistogramBundle,
    parent_leaf_value: f32,
    depth: usize,
}

impl Ord for PendingSplit {
    // Order by gain (max-heap)
}
```

#### Step 1.2: Implement leaf-wise training loop

New method `fit_iterations_leaf_wise`:

```rust
fn fit_iterations_leaf_wise<B: BackendOps, O: ObjectiveOps>(
    &self,
    dataset: &TrainingDataset,
    binned_matrix: &BinnedMatrix,
    backend: &B,
    objective: &O,
    controls: IterationControls,
) -> EngineResult<IterationRunSummary>
```

The loop:
1. For each boosting round:
   a. Compute gradients (same as level-wise)
   b. Build root histograms
   c. Find root's best split, push to priority queue
   d. While `leaves_used < max_leaves` and queue non-empty:
      - Pop highest-gain pending split
      - Check `max_depth` constraint
      - Apply split (partition rows, compute leaf values)
      - For each child: build histograms, find best split, push to queue
   e. Commit tree, update predictions

### Phase 2: Histogram Caching (Important for Performance)

Without histogram caching, leaf-wise growth is **slower** than level-wise because:
- Level-wise: build histograms for all nodes at depth d, use subtraction trick for one child
- Leaf-wise: when revisiting a deferred node, must rebuild its histograms from scratch

#### Step 2.1: Cache histograms in priority queue

Each `PendingSplit` already stores `histograms: HistogramBundle`. When a node is deferred (not the max-gain node), its histograms stay in memory. When it's eventually popped, histograms are ready.

Memory cost: `O(max_leaves * feature_count * max_bin * sizeof(HistogramBin))`. For 256 features, 256 bins, 12 bytes per bin: ~192 KB per cached node. With 256 max_leaves: ~48 MB. Acceptable.

#### Step 2.2: Subtraction trick for children

When splitting node N into children L and R:
- Build histograms for the smaller child (fewer rows = faster)
- Compute the larger child's histograms by subtracting smaller from parent: `H_larger = H_parent - H_smaller`

This already exists in level-wise mode (`subtract_histogram_bundle`). Reuse it.

### Phase 3: Python API

**`bindings/python/alloygbm/regressor.py`**

```python
GBMRegressor.__init__(
    # ...
    tree_growth: str = "level",    # "level" or "leaf"
    max_leaves: int | None = None,  # required for leaf-wise, optional for level-wise
)
```

Validation:
- If `tree_growth="leaf"` and `max_leaves` is None, raise error (leaf-wise needs a leaf budget)
- If `tree_growth="level"` and `max_leaves` is set, use it as an additional constraint (covered in Configurability plan Feature D)

### Phase 4: Bridge

Pass `tree_growth` mode and `max_leaves` through bridge. The engine selects the appropriate training loop.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Priority queue + loop | 1 (`engine/src/lib.rs`) | ~150-200 | High |
| Phase 2: Histogram caching | 1 (`engine/src/lib.rs`) | ~50-80 | Medium |
| Phase 3: Python API | 1 (`regressor.py`) | ~20-30 | Low |
| Phase 4: Bridge | 1 (`bindings/python/src/lib.rs`) | ~20-30 | Low |

Total: ~240-340 lines. This is a significant feature -- it adds a second tree building algorithm.

---

## Risk Areas

### Correctness

The leaf-wise loop must produce valid trees that the predictor can traverse. The predictor uses node_id-based lookup, so the node_id encoding must be correct for unbalanced trees. Currently, `encode_tree_node_id` combines round index and local node ID. Verify that unbalanced trees don't violate any ID assumptions.

### Auto Policy Interaction

The auto training policy tunes parameters assuming level-wise growth. Leaf-wise growth may need different heuristics (e.g., leaf-wise typically needs stronger regularization to prevent overfitting). Consider adding leaf-wise-specific auto policy adjustments.

### Memory Pressure

Caching histograms for all pending nodes increases memory usage proportional to `max_leaves`. For very large `max_leaves` (1000+) with wide datasets, this could be significant. Consider a memory-budget mode that evicts least-promising cached histograms.

### SHAP Compatibility

The SHAP implementation traverses tree structures. Verify it handles unbalanced trees correctly. Since SHAP operates on the artifact's split/leaf node representation (not the tree building process), it should work unchanged.

---

## Testing Strategy

1. **Equivalence**: For a fully balanced tree (max_leaves = 2^max_depth), leaf-wise and level-wise should produce identical trees (both explore the same splits in the same order)
2. **Unbalanced trees**: With max_leaves < 2^max_depth, verify leaf-wise produces unbalanced but valid trees
3. **Prediction correctness**: Predictions from leaf-wise model match manual tree traversal
4. **Early stopping**: Verify early stopping works correctly with leaf-wise
5. **Artifact roundtrip**: Leaf-wise model artifacts serialize/deserialize correctly

---

## Non-Goals

- **Histogram-based gradient-based one-side sampling (GOSS)**: LightGBM's other innovation. Separate initiative.
- **Exclusive Feature Bundling (EFB)**: LightGBM's sparse feature optimization. Separate initiative.
- **Voting parallel training**: LightGBM's distributed training approach. Out of scope.
