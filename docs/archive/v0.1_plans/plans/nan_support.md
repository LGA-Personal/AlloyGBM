# Plan: Missing Value (NaN) Support

## Status: Not Started

## Summary

AlloyGBM currently rejects NaN/Infinity values in all features. Both training and prediction explicitly validate that all values are finite:
- `validate_dense_values_finite()` in `bindings/python/src/lib.rs` checks every cell
- `_validate_rows` in `regressor.py` casts values to `float()` with no NaN handling
- The `BinnedMatrix` has no reserved "missing" bin
- The predictor's threshold comparisons assume all values are valid floats

The `ColumnarMatrixColumnView` has a `validity: Option<&[bool]>` bitmap, suggesting missing-value support was *considered* at the data layer, but it's never wired into training or prediction.

This is a significant practical limitation -- real-world tabular data almost always has missing values.

---

## Questions to Resolve Before Starting

1. **Missing value routing strategy**: When a tree node splits on feature `f` and a row has `NaN` for feature `f`, which child does it go to?
   - **Option A (XGBoost approach)**: Learn the optimal direction during training. For each split, try routing NaN rows left and right, pick whichever gives better gain. Store the "default direction" in the split.
   - **Option B (LightGBM approach)**: Route NaN to the child with larger gradient sum (effectively learned, but determined by a simpler heuristic).
   - **Option C (Simple)**: Always route NaN left (or right). Simple but suboptimal.
   Recommendation: **Option A** -- learn optimal direction. It's the most principled and provides the best accuracy.

2. **Missing bin in BinnedMatrix**: Should NaN get a special bin value (e.g., bin 255 reserved for missing), or should missing be tracked separately via a validity bitmap?
   - **Option A (Special bin)**: Reserve one bin value (e.g., `u8::MAX = 255`) as the missing bin. Reduces usable bins from 256 to 255. Simple to implement -- histogram includes a "missing" bin. Reduces `continuous_binning_max_bins` ceiling by 1.
   - **Option B (Validity bitmap)**: Separate `Vec<bool>` or bitset per column indicating which values are present. More memory, more complex indexing, but preserves full 256-bin range.
   Recommendation: **Option A** -- special bin value. Simpler, well-understood (LightGBM uses this), and losing 1 bin out of 256 is negligible.

3. **Prediction-time NaN handling**: The predictor currently compares feature values against float thresholds. For NaN, it needs to follow the "default direction" stored in the split. How is this stored in the artifact format?
   Recommendation: Add a `default_left: bool` flag to each split node in the artifact.

4. **Dense vs. sparse path**: Should the zero-copy numpy prediction path handle NaN? NaN in float arrays propagates through comparisons (`NaN <= threshold` is `false` in IEEE 754), which would always route right. This must be explicitly handled.

---

## Architecture Overview

### Affected Components

1. **Data ingestion**: Remove NaN rejection, map NaN to missing bin
2. **BinnedMatrix**: Reserve a "missing" bin value
3. **Histogram building**: Accumulate missing-value gradients separately
4. **Split finding**: Try both NaN-left and NaN-right, pick better
5. **Tree building**: Store default direction in split nodes
6. **Artifact format**: Serialize default direction per split
7. **Predictor**: Route NaN according to default direction
8. **Python API**: Remove NaN validation, document behavior

---

## Phase 1: Missing Bin in BinnedMatrix

### Files to Modify

**`crates/core/src/lib.rs`**

#### Step 1.1: Define missing bin constant

```rust
pub const MISSING_BIN: u8 = 255;
pub const MAX_USABLE_BINS: u16 = 255;  // 0..254 for real values, 255 for missing
```

#### Step 1.2: Update `BinnedMatrix` documentation

Document that bin value 255 is reserved for missing values. Update `continuous_binning_max_bins` validation to cap at 255 instead of 256.

### Files to Modify

**`bindings/python/src/lib.rs`**

#### Step 1.3: Update binning logic

In `prepare_training_matrices_from_dense_values` and related functions, when quantizing continuous features:
- If a value is NaN, assign bin `MISSING_BIN` (255)
- If a value is finite, quantize as usual (bins 0..254)
- Remove `validate_dense_values_finite()` checks (or make them warn instead of error)

---

## Phase 2: Histogram Building with Missing Values

### Files to Modify

**`crates/backend_cpu/src/lib.rs`**

#### Step 2.1: Separate missing-value gradient accumulation

During histogram building, the missing bin (255) accumulates gradients separately. The histogram for each feature needs a "missing" bucket alongside the regular bins:

Option A: Include missing bin as part of the regular histogram array (bin index 255 is the missing bucket). This works if the histogram array is always 256 elements. Currently, histogram size is `max_bin + 1`, which could be < 256. Need to ensure missing bin is always allocated.

Option B: Store missing gradients as a separate `GradientPair` alongside each feature histogram. Cleaner separation.

Recommendation: **Option A** if histograms are always 256 elements (they use `u8` bins so max is 256). Verify that `max_bin + 1 <= 256` is always true and that histograms are allocated at the full size.

The histogram kernel functions (`TinyNodeScalar`, `BinHeavyPerFeatureScalar`, `ArenaRowFirstUnrolled`) iterate over rows and accumulate into `bins[bin_value]`. NaN rows will accumulate into `bins[255]` naturally -- no kernel changes needed if the histogram arrays are 256 elements wide.

**Key verification**: Check that `HistogramBin` arrays are allocated as `vec![HistogramBin::default(); 256]` or `vec![HistogramBin::default(); max_bin as usize + 1]`. If the latter, and `max_bin < 255`, then NaN rows would write out of bounds. Must ensure allocation covers bin 255.

---

## Phase 3: Split Finding with Missing Value Direction

### Files to Modify

**`crates/backend_cpu/src/lib.rs`**

#### Step 3.1: Modify `best_split_for_feature`

Current logic (line ~416): scans bins left-to-right, accumulating left_grad/left_hess. The best split maximizes gain.

New logic: For each candidate split threshold, try two configurations:
1. NaN goes left: `left_grad += missing_grad`, `left_hess += missing_hess`
2. NaN goes right: `right_grad += missing_grad`, `right_hess += missing_hess`

Pick the configuration with better gain. Store the direction.

Implementation approach:
```rust
// Extract missing value stats
let missing_bin_stats = &feature_histogram.bins[MISSING_BIN as usize];
let missing_grad = missing_bin_stats.grad_sum;
let missing_hess = missing_bin_stats.hess_sum;
let missing_count = missing_bin_stats.count;

// Compute total (excluding missing)
let non_missing_total_grad = total_grad - missing_grad;
// ... etc

// For each threshold, try NaN-left and NaN-right
for threshold_bin in 0..num_bins-1 {
    // ... accumulate left/right for non-missing rows ...

    // Try NaN left
    let gain_nan_left = compute_gain(left_grad + missing_grad, left_hess + missing_hess,
                                      right_grad, right_hess);
    // Try NaN right
    let gain_nan_right = compute_gain(left_grad, left_hess,
                                       right_grad + missing_grad, right_hess + missing_hess);

    let (gain, default_left) = if gain_nan_left >= gain_nan_right {
        (gain_nan_left, true)
    } else {
        (gain_nan_right, false)
    };
}
```

#### Step 3.2: Extend `SplitCandidate` with default direction

```rust
pub struct SplitCandidate {
    // ... existing fields ...
    pub default_left: bool,  // NEW: should NaN go left?
}
```

---

## Phase 4: Tree Building and Row Partitioning

### Files to Modify

**`crates/engine/src/lib.rs`** and **`crates/backend_cpu/src/lib.rs`**

#### Step 4.1: Update `apply_split_with_stats`

The partitioning function routes rows to left or right child based on `bin <= threshold_bin`. For NaN rows (`bin == MISSING_BIN`), route according to `split.default_left`:

```rust
for &row_index in node.row_indices {
    let bin = binned_matrix.get_bin(row_index, split.feature_index);
    if bin == MISSING_BIN {
        if split.default_left { left_indices.push(row_index); }
        else { right_indices.push(row_index); }
    } else if bin <= split.threshold_bin {
        left_indices.push(row_index);
    } else {
        right_indices.push(row_index);
    }
}
```

---

## Phase 5: Artifact Format

### Files to Modify

**`crates/core/src/lib.rs`** and **`crates/engine/src/lib.rs`**

#### Step 5.1: Add `default_left` to split serialization

The tree nodes are serialized in the artifact's Trees section and PredictorLayout section. Each split node needs a `default_left: bool` flag.

Check the current split node serialization format. If it uses a fixed-size binary layout, adding a bool requires either:
- A new bit in an existing flags field (if one exists)
- An extra byte per split node
- A format version bump

Recommendation: Use a bit in an existing field if possible, otherwise add a flags byte. Bump the format version and maintain backward compatibility (old artifacts default to `default_left = false`).

#### Step 5.2: Backward compatibility

Old artifacts (without `default_left`) should load with `default_left = false` for all splits (NaN always goes right). This is safe because old models were trained on data without NaN, so the direction doesn't matter.

---

## Phase 6: Predictor

### Files to Modify

**`crates/predictor/src/lib.rs`**

#### Step 6.1: Handle NaN in prediction

Currently, `predict_row` compares `value <= threshold` to decide left/right. For NaN:
```rust
if value.is_nan() {
    if node.default_left { go_left(); }
    else { go_right(); }
} else if value <= threshold {
    go_left();
} else {
    go_right();
}
```

**Performance consideration**: Adding a NaN check per node per feature is a branch prediction concern. For the hot path (no NaN in data), the branch should be perfectly predicted (always not-NaN). But verify with benchmarks.

Alternative: Use IEEE 754 behavior. `NaN <= threshold` is `false`, so without explicit handling, NaN always goes right. To support `default_left = true`, we need the explicit check.

#### Step 6.2: Zero-copy numpy prediction path

The fast numpy prediction path in Python (using float thresholds directly) must also handle NaN. numpy's `np.isnan()` or equivalent must be used.

---

## Phase 7: Python API

### Files to Modify

**`bindings/python/alloygbm/regressor.py`** and **`bindings/python/src/lib.rs`**

#### Step 7.1: Remove NaN rejection

Remove or modify `_validate_rows` to allow NaN values. Remove `validate_dense_values_finite()` from the bridge.

#### Step 7.2: Document NaN behavior

- NaN values are treated as missing
- During training, the model learns the optimal direction for missing values at each split
- During prediction, NaN values follow the learned direction
- Infinity values are still rejected (or could be allowed and treated as extreme values)

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Missing bin | 2 (core, bridge) | ~30-40 | Low |
| Phase 2: Histograms | 1 (backend_cpu) | ~20-40 | Medium |
| Phase 3: Split finding | 1 (backend_cpu) | ~50-80 | Medium-High |
| Phase 4: Row partitioning | 2 (engine, backend_cpu) | ~20-30 | Low |
| Phase 5: Artifact format | 2 (core, engine) | ~40-60 | Medium |
| Phase 6: Predictor | 1 (predictor) | ~30-50 | Medium |
| Phase 7: Python API | 2 (regressor.py, bridge) | ~20-30 | Low |

Total: ~210-330 lines across 5-6 files. This is a cross-cutting feature touching every layer.

---

## Risk Areas

### Histogram Sizing

The biggest risk is the histogram array sizing. If histograms are allocated as `vec![HistogramBin; max_bin + 1]` and `max_bin` is < 255, then writing to index 255 (MISSING_BIN) would panic. Must ensure histograms are always 256 elements, or allocate as `max(max_bin + 1, 256)` when NaN values are present.

### Prediction Performance

Adding `is_nan()` checks in the hot prediction path could impact performance. Benchmark carefully. If the overhead is measurable, consider:
- Only checking NaN when the model was trained with NaN data (flag in metadata)
- Using SIMD-friendly NaN detection
- Separate code paths for "definitely no NaN" vs. "might have NaN"

### Artifact Format Migration

Changing the split node format is a breaking change if not handled carefully. Old code reading new artifacts with `default_left` fields would fail unless backward compatibility is maintained. Use format versioning.

### Column Subsampling

When columns are subsampled, a row might have NaN in a feature that isn't selected for a given tree. This is fine -- the NaN handling only matters for features that appear in the tree's splits.

---

## Testing Strategy

1. **Basic NaN handling**: Train on data with NaN values, verify model trains without error
2. **Learned direction**: Train on data where NaN rows have high target values. Verify the model learns to route NaN to the child with higher predictions.
3. **Prediction with NaN**: Predict on data with NaN, verify results use learned directions
4. **No NaN baseline**: Train/predict without NaN, verify identical results to current code
5. **All NaN feature**: A feature that is entirely NaN should produce no useful splits
6. **Artifact roundtrip**: Model with NaN-aware splits serializes/deserializes correctly
7. **Performance**: Benchmark prediction speed with and without NaN data to measure overhead

---

## Non-Goals

- **NaN imputation**: Filling in missing values before training (e.g., mean imputation). Users can do this themselves.
- **Sparse matrix support**: Handling structurally missing values (as opposed to explicitly NaN). Different representation.
- **Per-feature NaN strategy**: Different NaN handling for different features. All features use the same learned-direction approach.
