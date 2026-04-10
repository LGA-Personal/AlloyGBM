# Plan: Group ID Support from Python

## Status: Not Started

## Summary

The Rust engine's `TrainingDataset` already has `group_id: Option<Vec<u32>>`, but the Python bridge always passes `group_id: None`. Group IDs are a prerequisite for learning-to-rank objectives (LambdaMART, LambdaRank) and also useful for grouped cross-validation and grouped evaluation metrics.

This plan covers exposing the existing `group_id` field from Python. The actual ranking objective implementation is covered in the Classification & Ranking plan (#1).

---

## Questions to Resolve Before Starting

1. **Group ID format**: Should the Python API accept:
   - `group: array-like of int` -- one group label per row (like LightGBM's `group` parameter)
   - `group: array-like of int` -- group sizes in order (like XGBoost's `qid` format)
   Recommendation: group label per row (more intuitive, same as LightGBM). Convert to sequential u32 IDs internally.

2. **Validation set groups**: Should `eval_set` also accept group IDs? Recommendation: yes, for consistent ranking evaluation.

3. **Timing**: Should this be implemented before or alongside the ranking objective? Recommendation: implement the plumbing now (it's minimal) so the ranking objective plan has the infrastructure ready.

---

## Implementation

### Phase 1: Python API

**`bindings/python/alloygbm/regressor.py`**

Add `group` parameter to `fit()`:
```python
def fit(self, X, y, *, group: object | None = None, eval_group: object | None = None, ...):
```

Validation:
- Must be array-like of integers, length matching `n_rows`
- Values must be non-negative
- Rows with the same group ID must be contiguous (or sort them internally)

### Phase 2: Bridge

**`bindings/python/src/lib.rs`**

Add `group_id: Option<Vec<u32>>` to all 5 training pyfunctions and pass through to `TrainingDataset` construction. Currently `group_id: None` is hardcoded -- change to pass the user's value.

### Phase 3: Engine Validation

**`crates/engine/src/lib.rs`**

Add validation that if `group_id` is provided:
- Length matches `row_count`
- Group IDs form contiguous blocks (all rows in a group are adjacent)
- This is important for ranking objectives that operate on groups

---

## Estimated Complexity

| Phase | Files Changed | Lines Changed | Risk |
|-------|--------------|--------------|------|
| Phase 1: Python API | 1 (`regressor.py`) | ~20-30 | Very Low |
| Phase 2: Bridge | 1 (`bindings/python/src/lib.rs`) | ~30-40 | Very Low |
| Phase 3: Validation | 1 (`engine/src/lib.rs`) | ~15-20 | Very Low |

Total: ~65-90 lines. Pure plumbing -- the Rust field already exists.

---

## Testing Strategy

1. **Passthrough**: Group IDs reach the engine's `TrainingDataset` correctly
2. **Validation**: Non-contiguous groups rejected, negative IDs rejected, length mismatch rejected
3. **No-op for regression**: With `SquaredErrorObjective`, group IDs are stored but don't affect training (they'll matter once ranking objectives are added)

---

## Non-Goals

- **Ranking objective implementation**: Covered in Classification & Ranking plan (#1)
- **Grouped cross-validation**: sklearn's `GroupKFold` will work automatically once groups are exposed
- **Group-aware evaluation metrics** (NDCG, MAP): Covered in Classification & Ranking plan (#1)
