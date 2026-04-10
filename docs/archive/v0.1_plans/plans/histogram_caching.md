# Plan: Global Histogram Caching Between Rounds

## Status: Not Started

## Summary

The engine implements the histogram subtraction trick for child nodes (`subtract_histogram_bundle` in backend_cpu/src/lib.rs), which is good -- when splitting a node, only the smaller child's histogram is built from scratch, and the larger child's is computed by subtraction. However, there is no histogram caching between boosting rounds -- histograms are rebuilt from scratch each iteration.

In theory, caching root histograms across rounds could save computation. In practice, the benefit is limited because gradients change every round (new residuals), so all histograms must be recomputed. The real optimization is ensuring the subtraction trick is maximally utilized *within* each round.

---

## Questions to Resolve Before Starting

1. **Is this actually a bottleneck?**: Profile first. The root histogram must be built fresh each round (gradients change). The subtraction trick is already used for children. The remaining opportunity is caching histograms for nodes at depth > 0 when their rows don't change between rounds -- but rows DO change with row subsampling. For `row_subsample=1.0` and `col_subsample=1.0`, rows are stable but gradients still change, so histograms still need rebuilding.

2. **Gradient histogram vs. structure histogram**: XGBoost caches the *structure* of the previous tree to guide the current tree's split order. This is different from caching gradient histograms. Is structural caching in scope? Recommendation: no, this is a fundamentally different optimization.

---

## Analysis

### What Changes Between Rounds

| Component | Changes? | Implication |
|-----------|----------|-------------|
| Row indices per node | Yes (new tree structure) | Can't reuse node-level histograms |
| Gradient values | Yes (new residuals) | Can't reuse gradient sums in histograms |
| Bin assignments | No (fixed after binning) | Could pre-sort rows by bin for faster access |
| Feature set | Maybe (col_subsample) | Must rebuild if features change |

### Where Time Is Spent

The histogram building kernel is the most expensive operation in tree training. Potential optimizations:

1. **Pre-sorted row indices by bin** (a.k.a. "data pre-sorting"): Sort row indices by bin value for each feature once before training. Then histogram building becomes a sequential scan through gradient pairs, improving cache locality. This is a structural optimization, not a caching one.

2. **Root histogram reuse across siblings**: Already done via subtraction trick.

3. **Histogram allocation reuse**: Instead of allocating new `Vec<HistogramBin>` each round, reuse pre-allocated buffers. This reduces allocation pressure.

### Recommendation

The most impactful optimization is **histogram buffer reuse** (avoid re-allocation) rather than histogram *value* caching (values must be recomputed). Additionally, ensuring the subtraction trick is applied at all depths (not just one level) provides the most benefit.

---

## Implementation

### Phase 1: Histogram Buffer Pool

**`crates/engine/src/lib.rs`** and **`crates/backend_cpu/src/lib.rs`**

#### Step 1.1: Pre-allocate histogram buffers

Before the boosting loop, allocate a pool of `HistogramBundle` buffers sized for the maximum possible tree:

```rust
let max_nodes_per_level = 2_usize.pow(max_depth as u32);
let mut histogram_pool: Vec<HistogramBundle> = (0..max_nodes_per_level)
    .map(|_| HistogramBundle::new_zeroed(feature_count, bin_count))
    .collect();
```

Each round, zero out and reuse these buffers instead of allocating new ones.

#### Step 1.2: Zero-fill instead of re-allocate

Add `HistogramBundle::reset(&mut self)` method that zeros all gradient sums and counts without deallocating.

### Phase 2: Maximize Subtraction Trick Coverage

**`crates/engine/src/lib.rs`**

Currently the subtraction trick is used for the second child at each level. Verify it's applied at **all** depths, not just depth 1. The current code in the level-wise loop does use it for each split's larger child. Audit to confirm.

### Phase 3: Data Pre-Sorting (Optional, Higher Impact)

Pre-sort row indices by bin value for each feature. This transforms histogram building from random access into sequential access through gradient pairs, dramatically improving cache performance for large datasets.

#### Step 3.1: Build sorted index

After binning, create `sorted_indices: Vec<Vec<u32>>` where `sorted_indices[feature_id]` is the row indices sorted by their bin value for that feature.

#### Step 3.2: Use sorted index in histogram kernel

The histogram kernel iterates over rows and accumulates gradient sums per bin. With pre-sorted indices, all rows with bin=0 come first, then bin=1, etc. This means the histogram bin being accumulated doesn't jump around -- much better cache behavior.

Trade-off: pre-sorting costs O(N * F) time and O(N * F) memory. For large datasets this is significant. Only beneficial when N is large enough that cache effects dominate.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Buffer pool | 2 (engine, backend_cpu) | ~40-60 | Low |
| Phase 2: Subtraction audit | 1 (engine) | ~10-20 (verification) | Very Low |
| Phase 3: Data pre-sorting | 2 (engine, backend_cpu) | ~80-120 | Medium |

---

## Testing Strategy

1. **Correctness**: Histogram values with buffer reuse match fresh allocation
2. **Performance**: Benchmark histogram building with/without buffer pool on California Housing dataset
3. **Memory**: Verify peak memory doesn't increase significantly with buffer pool
4. **Pre-sorting**: If implemented, verify histogram values match unsorted path

---

## Non-Goals

- **GPU histogram building**: Covered by the GPU/accelerator limitation (excluded from planning)
- **Distributed histogram aggregation**: Out of scope
- **Cross-round gradient caching**: Not feasible since gradients change every round
