# Plan: SHAP Exact Method -- Address 20-Feature Limit

## Status: Not Started

## Summary

`MAX_EXACT_SPLIT_FEATURES` is 20 in `crates/shap/src/lib.rs`. The SHAP implementation uses exact Shapley values via brute-force subset enumeration (2^N subsets for N split features). If a model uses more than 20 distinct features in its splits, SHAP computation errors out. Real models on wide datasets will commonly exceed this. The polynomial-time TreeSHAP algorithm (Lundberg et al., 2020) is not implemented.

---

## Questions to Resolve Before Starting

1. **Approach**: Which path to pursue?
   - **Option A**: Implement TreeSHAP (polynomial-time, O(TLD²) where T=trees, L=leaves, D=depth). This is the correct long-term solution.
   - **Option B**: Raise the limit and accept exponential time for small-ish models (e.g., 25 or 30 features). Quick fix.
   - **Option C**: Implement approximate SHAP (sampling-based). Faster than exact, simpler than TreeSHAP, but approximate.
   Recommendation: **Option A** (TreeSHAP) as the long-term goal. **Option B** as an immediate quick fix. They're not mutually exclusive.

2. **TreeSHAP variant**: Lundberg's paper describes two variants:
   - **Tree Path-Dependent** ("interventional"): Follows the tree's split logic, accounts for feature dependencies in the tree structure. Faster, used by default in the `shap` Python library.
   - **Tree Path-Independent** ("marginal"): Uses the full conditional expectation. Theoretically cleaner but slower.
   Recommendation: **Path-Dependent** (interventional). It's what users expect from TreeSHAP and what LightGBM/XGBoost implement.

3. **SHAP for multi-output models**: If classification (Limitation #1) is implemented, multi-class models produce K outputs. SHAP values would be per-class. Should this be in scope? Recommendation: defer, document as known limitation. Single-output SHAP first.

---

## Option B: Quick Fix -- Raise the Limit

### Files to Modify

**`crates/shap/src/lib.rs`**

Change:
```rust
const MAX_EXACT_SPLIT_FEATURES: usize = 20;
```

To:
```rust
const MAX_EXACT_SPLIT_FEATURES: usize = 25;  // or 30
```

**Trade-offs**:
- 20 features: 2^20 = ~1M subsets per row -- fast
- 25 features: 2^25 = ~33M subsets per row -- slow but feasible
- 30 features: 2^30 = ~1B subsets per row -- very slow, may take minutes per row
- 35 features: impractical

This is a one-line change but doesn't solve the fundamental problem. Models with 50+ split features remain unsupported.

### Complexity: Trivial (~1 line)

---

## Option A: Implement TreeSHAP (Polynomial Time)

### Algorithm Overview

TreeSHAP computes exact Shapley values in O(TLD²) time, where:
- T = number of trees
- L = number of leaves per tree
- D = maximum depth per tree

The key insight: instead of enumerating all 2^N feature subsets, recursively walk through each tree and track the "proportion" of each feature's contribution at each decision node.

### The Lundberg Algorithm (Path-Dependent / Interventional)

For each tree and each input row:
1. Start at the root with an empty path
2. At each internal node:
   - If the feature at this node is in the path: follow the true branch
   - If not: follow both branches, weighted by the fraction of training data going each way
3. At each leaf: compute the contribution of each feature on the path using the SHAP weighting formula
4. Sum contributions across all trees

The algorithm maintains a "path" structure that tracks:
- Which features have been encountered
- The fraction of data reaching this node through each path
- The "one fraction" and "zero fraction" for the combinatorial weighting

### Implementation Steps

**`crates/shap/src/lib.rs`**

#### Step A.1: Define the path tracking structure

```rust
struct PathElement {
    feature_index: usize,
    zero_fraction: f64,  // fraction of data if this feature is "off"
    one_fraction: f64,   // fraction of data if this feature is "on"
    pweight: f64,        // path weight for SHAP combinatorics
}

struct Path {
    elements: Vec<PathElement>,
}
```

#### Step A.2: Implement the recursive tree walk

```rust
fn tree_shap_recursive(
    node: &TreeNode,
    row: &[f32],
    path: &mut Path,
    shap_values: &mut [f64],
    // tree structure: nodes, children, thresholds, leaf_values, cover (training data counts)
)
```

At each internal node:
1. Push a new `PathElement` for this node's feature
2. Compute `zero_fraction` = (child_cover / parent_cover) for the branch NOT taken by the input
3. Compute `one_fraction` = 1.0 (the input follows a definite path)
4. Recurse into the child that the input goes to (with one_fraction=1, zero_fraction=child_cover/parent_cover for the other child)
5. At leaves: unwind the path and compute contributions using the SHAP weighting formula

#### Step A.3: The SHAP weighting formula

The contribution of feature `i` at a leaf is:

```
for each element j in the path:
    contribj = sum over subsets S not containing j of:
        [w(|S|, M) * (f(S ∪ {j}) - f(S))]
```

where M is the number of features in the path, and w is the Shapley kernel weight.

In practice, this is computed using the `extend_path` and `unwind_path` operations described in Lundberg's supplementary material.

#### Step A.4: Cover (training data counts) per node

TreeSHAP needs to know how many training samples pass through each node. This is available during training (`NodeStats.row_count`) but is NOT currently stored in the artifact.

**Critical requirement**: Store per-node training sample counts in the artifact, or recompute from the tree structure. Options:
- Add `cover` to the split node serialization (increases artifact size)
- Recompute from leaf counts using the tree structure (possible since row counts of children sum to parent)
- Use hessian sums as a proxy for cover (hessian = weight for MSE, sum of hessians ≈ cover if weights are uniform)

Recommendation: Use hessian sums already stored in the split nodes. `NodeStats.hess_sum` is serialized in the artifact and directly proportional to row count (for MSE with uniform weights, hess_sum = row_count).

#### Step A.5: Public API

```rust
pub fn tree_shap_values(
    predictor: &Predictor,
    row: &[f32],
) -> Result<Vec<f64>, ShapError>
```

Keep the existing exact brute-force method as a fallback (for verification) and add TreeSHAP as the default:

```rust
pub fn shap_values(
    predictor: &Predictor,
    row: &[f32],
    method: ShapMethod,  // Exact, TreeShap
) -> Result<Vec<f64>, ShapError>
```

### Phase 2: Verification

The exact brute-force SHAP and TreeSHAP should produce **identical** results (both are exact). Use the existing brute-force implementation to verify TreeSHAP correctness on models with <= 20 features.

### Phase 3: Python API

**`bindings/python/alloygbm/regressor.py`**

Update `shap_values()` to use TreeSHAP by default:
```python
def shap_values(self, X, method='tree'):
    """Compute SHAP values.

    method: 'tree' (polynomial-time TreeSHAP) or 'exact' (brute-force, limited to 20 features)
    """
```

---

## Estimated Complexity

| Component | Lines | Risk |
|-----------|-------|------|
| Quick fix (raise limit) | 1 | None |
| TreeSHAP path tracking | ~40-50 | Medium |
| TreeSHAP recursive walk | ~80-120 | High |
| SHAP weighting (extend/unwind) | ~50-70 | High |
| Node cover extraction | ~20-30 | Low |
| Public API + method selection | ~20-30 | Low |
| Verification tests | ~40-60 | Medium |
| Python API | ~10-15 | Low |

Total for TreeSHAP: ~260-375 lines in `crates/shap/src/lib.rs` + ~10-15 in Python. This is the most algorithmically complex plan.

---

## Risk Areas

### Algorithmic Correctness

TreeSHAP is subtle. The `extend_path`, `unwind_path`, and contribution accumulation steps must exactly implement the Shapley weighting formula. Off-by-one errors or incorrect fraction calculations will produce wrong SHAP values that satisfy additivity but are individually incorrect.

Mitigation: Verify against brute-force exact SHAP on every test case with <= 20 features. The two methods must agree to within floating-point tolerance.

### Node Cover Accuracy

If using hessian sums as a proxy for cover, the proxy is only exact when all sample weights are equal. With weighted training (Limitation #10), hessian sums ≠ row counts. Need to store actual row counts or use weights-aware cover.

### Numerical Stability

The SHAP weighting involves division by combinatorial terms that can be very small (for deep trees with many features). Use f64 throughout the computation (the current exact method already uses f64).

### Performance

TreeSHAP is O(TLD²) per row, where D is tree depth. For a model with 100 trees, 64 leaves each, depth 6: ~100 * 64 * 36 = 230K operations per row. This is fast. For deeper trees or more leaves, it scales quadratically in depth.

For batch SHAP over many rows, this can be parallelized across rows (each row is independent).

---

## Testing Strategy

1. **Cross-validation**: For models with <= 20 split features, verify TreeSHAP matches brute-force exact SHAP
2. **Additivity**: `sum(shap_values) + expected_value ≈ model_prediction` for every row
3. **Symmetry**: Features not used in any split should have SHAP value = 0
4. **Scale**: Run TreeSHAP on a model with 100+ split features, verify it completes in reasonable time
5. **Edge cases**: Single-tree model, single-split tree, all rows identical
6. **Consistency with reference**: Compare against the `shap` Python library's TreeExplainer on the same model (requires converting AlloyGBM's tree format)

---

## References

- Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
- Lundberg, S. M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence.
- Reference implementation: https://github.com/shap/shap (TreeExplainer)

---

## Non-Goals

- **Approximate SHAP** (sampling-based): Could be added later as a third method option
- **Multi-output SHAP**: For multi-class classification (deferred until classification is implemented)
- **SHAP interaction values**: Second-order Shapley interactions. Algorithmically more complex.
- **GPU-accelerated SHAP**: Out of scope (no GPU backend)
