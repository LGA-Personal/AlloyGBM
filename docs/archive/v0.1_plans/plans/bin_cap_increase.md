# Plan: Increase Bin Cap Beyond 256

## Status: Not Started

## Summary

The `BinnedMatrix` uses `Vec<u8>` for bin storage, capping at 256 bins per feature. While adequate for many use cases, LightGBM defaults to 255 bins but supports up to 65535 for high-cardinality continuous features. The `threshold_bin` field in `SplitCandidate` is already `u16`, but the actual bin storage is `u8`, so features with more than 256 distinct values lose resolution.

Increasing the bin cap requires changing the bin storage type and updating all code that processes bins.

---

## Questions to Resolve Before Starting

1. **Target bin count**: What should the new maximum be?
   - 512 (u16, modest increase)
   - 65536 (u16, matches LightGBM's max)
   Recommendation: Support up to 65536 (u16). The `threshold_bin` in `SplitCandidate` is already u16.

2. **Memory trade-off**: Doubling bin storage from u8 to u16 doubles the BinnedMatrix memory. For a dataset with 100K rows and 100 features:
   - u8: 100K * 100 = 10 MB
   - u16: 100K * 100 * 2 = 20 MB
   Recommendation: Default to 256 bins (u8 mode) for backward compat. Only use u16 when user requests > 256 bins.

3. **Dual-mode vs. single-mode**: Should we support both u8 (fast, small) and u16 (flexible) bin modes? Or always use u16?
   - **Option A (Dual-mode)**: BinnedMatrix has a `BinStorage` enum (`U8(Vec<u8>)` / `U16(Vec<u16>)`). Histogram kernels are generic over bin type.
   - **Option B (Single u16)**: Always use u16. Simpler code, 2x memory for bins.
   - **Option C (Adaptive)**: u8 if max_bins <= 256, u16 otherwise. Best of both worlds but requires runtime dispatch.
   Recommendation: **Option C** -- adaptive. Most users will stay at <= 256 bins and benefit from the u8 fast path. Users who need more bins pay the u16 cost.

---

## Architecture Overview

### Current Bin Type Usage

| Location | Current Type | Notes |
|----------|-------------|-------|
| `BinnedMatrix.bins` | `Vec<u8>` | Column-major bin storage |
| `BinnedMatrix.max_bin` | `u16` | Already u16 |
| `SplitCandidate.threshold_bin` | `u16` | Already u16 |
| Histogram kernels | `bins[i] as usize` | Cast u8 -> usize for indexing |
| `encode_bins_from_encoded_values` | Returns `Vec<u8>` | Bin assignment |
| Predictor threshold comparison | Compares against u16 threshold | Already u16 |

The good news: `max_bin` and `threshold_bin` are already u16, so the tree structure, artifact format, and predictor already support > 256 bins. The bottleneck is purely the `Vec<u8>` storage in `BinnedMatrix`.

---

## Phase 1: Adaptive Bin Storage

### Files to Modify

**`crates/core/src/lib.rs`**

#### Step 1.1: Define bin storage enum

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum BinStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl BinStorage {
    pub fn get(&self, index: usize) -> u16 {
        match self {
            Self::U8(bins) => bins[index] as u16,
            Self::U16(bins) => bins[index],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(bins) => bins.len(),
            Self::U16(bins) => bins.len(),
        }
    }
}
```

#### Step 1.2: Update `BinnedMatrix`

```rust
pub struct BinnedMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub max_bin: u16,
    pub bins: BinStorage,  // was: Vec<u8>
}
```

### Files to Modify

**`crates/backend_cpu/src/lib.rs`**

#### Step 1.3: Update histogram kernels

Each histogram kernel accesses `bins[offset]` directly. With the enum, they need to go through `bins.get(offset)` which returns u16. Or, for performance, match on the enum once at the top of the kernel and dispatch to type-specialized inner loops:

```rust
match &binned_matrix.bins {
    BinStorage::U8(bins) => build_histograms_u8(bins, ...),
    BinStorage::U16(bins) => build_histograms_u16(bins, ...),
}
```

This avoids per-element match overhead while keeping both paths.

#### Step 1.4: Update histogram allocation

Histograms must be sized to `max_bin + 1`. For u16 bins with `max_bin = 1000`, histograms would be 1001 entries. Currently, histogram sizes are often 256. This is the main memory impact on the histogram side.

**Important**: Each `HistogramBin` is `{ grad_sum: f32, hess_sum: f32, count: u32 }` = 12 bytes. For 1001 bins * 100 features = ~1.2 MB per node's histograms. Manageable, but grows with max_bins.

### Files to Modify

**`crates/engine/src/lib.rs`**

#### Step 1.5: Update all `BinnedMatrix` construction

Everywhere a `BinnedMatrix` is built (binning logic, categorical encoding), use `BinStorage::U8` when max_bin <= 255, `BinStorage::U16` otherwise.

---

## Phase 2: Python API

### Files to Modify

**`bindings/python/alloygbm/regressor.py`**

#### Step 2.1: Increase `continuous_binning_max_bins` ceiling

Currently capped at 256 (`_MAX_CONTINUOUS_QUANTIZED_BIN + 1`). Raise to 65536.

```python
_MAX_CONTINUOUS_QUANTIZED_BIN = 65535
```

Default remains 256 for backward compatibility.

### Files to Modify

**`bindings/python/src/lib.rs`**

#### Step 2.2: Update binning bridge

The bridge constructs `BinnedMatrix` from continuous values. Update to produce `BinStorage::U16` when `max_bins > 256`.

---

## Phase 3: Predictor Updates

### Files to Modify

**`crates/predictor/src/lib.rs`**

The predictor compares against `threshold_bin: u16`, which already supports > 256 bins. However, the predictor might need to read bin data from the artifact. Verify the artifact format stores bins correctly for u16.

Actually, the predictor operates on **float thresholds** (converted from bin thresholds), not raw bins. So the predictor path is likely **unchanged** -- it doesn't use `BinnedMatrix` at all. Verify this.

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Core + backend | 3 (core, engine, backend_cpu) | ~150-200 | Medium-High |
| Phase 2: Python API | 2 (regressor.py, bridge) | ~20-30 | Low |
| Phase 3: Predictor | 1 (predictor) | ~10-20 (verify) | Very Low |

Total: ~180-250 lines across 4 files. The main complexity is updating the histogram kernels -- they're performance-critical hot loops.

---

## Risk Areas

### Performance Regression

Switching from `Vec<u8>` to `BinStorage` adds indirection. Even with type-specialized kernel dispatch, the u8 path might be slightly slower due to the enum wrapper. Benchmark the u8 path before and after to ensure zero regression.

Mitigation: Keep the u8 inner loops identical to today's code. Only the outer dispatch changes.

### Histogram Memory

With 65536 bins, a single feature's histogram is 65536 * 12 bytes = 768 KB. For 100 features, that's 75 MB per node. This is impractical for large bin counts.

Mitigation: Document that very high bin counts (> 1024) should only be used selectively. Consider per-feature bin counts (covered in Configurability plan Feature F) to allow high bins only where needed.

### Column-Major Duplicate

The `BinnedMatrix` has a column-major duplicate for cache-friendly histogram access. This must also support the new bin storage type. Memory overhead doubles again for u16.

---

## Testing Strategy

1. **u8 path unchanged**: Training with max_bins=256, verify identical results to current code
2. **u16 path correctness**: Training with max_bins=512, verify trees produce valid predictions
3. **Adaptive dispatch**: Verify u8 used for <= 256, u16 for > 256
4. **Performance**: Benchmark histogram building for both paths, verify u8 has no regression
5. **Artifact roundtrip**: Model trained with > 256 bins serializes/deserializes correctly
6. **Edge cases**: max_bins=1 (minimum), max_bins=65536 (maximum)

---

## Non-Goals

- **u32 bins**: Supporting > 65536 bins. Extremely niche and impractical memory-wise.
- **Sparse bin storage**: Only storing non-zero bins. Would help for very sparse data but is a fundamental redesign.
- **Per-feature bin count from Python**: Covered in Configurability plan Feature F.
