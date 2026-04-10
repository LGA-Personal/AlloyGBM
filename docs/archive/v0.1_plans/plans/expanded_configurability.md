# Plan: Expanded Configurability

## Status: Not Started

## Summary

AlloyGBM exposes a reasonable set of training parameters (`learning_rate`, `max_depth`, `n_estimators`, `row_subsample`, `col_subsample`, `early_stopping_rounds`, `min_validation_improvement`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `min_child_hessian`, `seed`, `deterministic`, binning strategy, training policy). However, compared to XGBoost/LightGBM, several important knobs are missing.

This plan covers exposing existing internal controls and adding new constraint mechanisms. It is organized as independent, additive features -- each can be implemented separately.

---

## Questions to Resolve Before Starting

1. **Priority ordering**: Which missing parameters matter most to the user? Recommendation: expose `min_split_gain` first (it already exists internally), then monotone constraints (high practical value), then the rest.

2. **Custom objective / custom metric**: Should the `ObjectiveOps` trait be exposed to Python? This is a significant design question -- it requires either a callback protocol (Python function called per gradient computation, slow) or a fixed menu of named objectives (fast, limited). Recommendation: defer to the Classification & Ranking plan for named objectives; custom Python callbacks are a separate, lower-priority item.

3. **API naming conventions**: Should new parameters follow XGBoost naming (`gamma` for min_split_gain, `reg_alpha`/`reg_lambda`) or LightGBM naming (`min_gain_to_split`, `lambda_l1`/`lambda_l2`) or AlloyGBM's own convention? Current AlloyGBM already uses `lambda_l1`/`lambda_l2` (LightGBM-style) and `min_data_in_leaf` (LightGBM-style). Recommendation: continue with LightGBM-style naming for consistency.

4. **Leaf-wise growth**: Should `max_leaves` / leaf-wise growth be part of this plan or a separate plan? It's listed as Limitation #15. Recommendation: keep it in the separate Level-Wise Tree Growth plan (#15), since it's a fundamental tree building algorithm change.

---

## Feature A: Expose `min_split_gain` as a User-Facing Parameter

### Current State

`min_split_gain` already exists in `IterationControls` (engine/src/lib.rs:303) and is used in the tree building loop (line 1241):
```rust
if !split.gain.is_finite() || split.gain <= controls.min_split_gain {
    round_rejection_reason = IterationStopReason::GainBelowThreshold;
    continue;
}
```

In auto policy mode, `min_split_gain` is set heuristically based on binned density (line 1074):
- Dense data: `0.0001`
- Sparse data: `0.001`
- Small data: `0.0`

In manual policy mode, it defaults to `0.0`.

The user has **no way to set this directly** -- it's not in `TrainParams` and not in the Python API.

### Implementation

#### Step A.1: Add `min_split_gain` to `TrainParams` (`core/src/lib.rs`)

```rust
pub struct TrainParams {
    // ... existing fields ...
    pub min_split_gain: f32,  // NEW, default 0.0
}
```

Add validation: `min_split_gain` must be finite and >= 0.

#### Step A.2: Wire into `IterationControls` construction (`engine/src/lib.rs`)

In `default_iteration_controls` (line ~1014), set `controls.min_split_gain = self.params.min_split_gain`.

In `auto_iteration_controls` (line ~1039), apply the auto heuristic but respect user override: if user set `min_split_gain > 0`, use `max(user_value, auto_value)`.

#### Step A.3: Add to Python API (`regressor.py`)

Add `min_split_gain: float = 0.0` to `GBMRegressor.__init__`, `get_params()`, `set_params()`.

#### Step A.4: Pass through bridge (`bindings/python/src/lib.rs`)

Add `min_split_gain` to `TrainParams` construction in the bridge functions.

### Complexity: Very Low (~20 lines across 3 files)

---

## Feature B: Monotone Constraints

### Overview

Monotone constraints force the model to respect a monotonic relationship between a feature and the prediction. For feature `i`:
- `+1`: prediction must be non-decreasing as feature `i` increases
- `-1`: prediction must be non-increasing as feature `i` increases
- `0`: no constraint (default)

This is critical for domain knowledge injection (e.g., "house price should increase with square footage").

### Implementation Approach

There are two approaches used by existing libraries:

**Approach 1 (XGBoost-style)**: At split evaluation time, check if the proposed split respects monotonicity. For a feature with constraint `+1`, the left child's leaf value must be <= right child's leaf value. If violated, skip the split.

**Approach 2 (LightGBM-style)**: At split evaluation time, clamp the leaf values to respect monotonicity, then re-evaluate the gain with clamped values. More flexible but more complex.

Recommendation: **Approach 1** -- simpler, well-understood, and sufficient for most use cases.

#### Step B.1: Add constraint storage to `TrainParams` (`core/src/lib.rs`)

```rust
pub struct TrainParams {
    // ... existing fields ...
    pub monotone_constraints: Vec<i8>,  // +1, -1, 0 per feature; empty = no constraints
}
```

Validation: length must be 0 (no constraints) or equal to `feature_count`. Values must be in {-1, 0, +1}.

#### Step B.2: Pass constraints into split selection (`engine/src/lib.rs`)

Add `monotone_constraints: &[i8]` to `SplitSelectionOptions` or pass separately to `best_split_with_options`.

In the tree building loop (line ~1236), after finding the best split, check monotonicity:

```rust
// After computing left_leaf_absolute and right_leaf_absolute (line ~1279-1282)
if !monotone_constraints.is_empty() {
    let constraint = monotone_constraints[split.feature_index as usize];
    if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
        continue; // violates non-decreasing
    }
    if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
        continue; // violates non-increasing
    }
}
```

**Important subtlety**: The above checks leaf *absolute* values, which account for the parent's contribution. This is correct because the final prediction for rows going left vs. right must respect monotonicity.

**Alternative (more aggressive)**: Check during histogram scan in `best_split_for_feature` (`backend_cpu/src/lib.rs:416`). This would prune invalid splits before they're even considered, but requires passing constraints down into the backend. More efficient for constrained features but more invasive.

Recommendation: Start with the simpler check in the tree building loop. If performance is a concern (many constrained features, most splits rejected), move the check into the backend.

#### Step B.3: Add to Python API

```python
GBMRegressor.__init__(
    # ...
    monotone_constraints: list[int] | dict[int, int] | None = None,
)
```

Accept either:
- `list[int]`: one constraint per feature, length must match feature count
- `dict[int, int]`: sparse format, only constrained features specified
- `None`: no constraints

Convert to dense `Vec<i8>` before passing to Rust.

#### Step B.4: Bridge wiring

Add `monotone_constraints: Vec<i8>` to `TrainParams` construction in bridge functions.

### Testing

- Train with constraint `+1` on a feature, verify that for every split on that feature, left leaf <= right leaf
- Train with constraint `-1`, verify reverse
- Verify unconstrained features are unaffected
- Edge case: all features constrained, verify training still produces a valid model (may produce shallower trees)

### Complexity: Medium (~100-150 lines across 4 files)

---

## Feature C: Interaction Constraints

### Overview

Interaction constraints limit which features can appear together in the same tree branch. For example, if features {A, B} and {C, D} are in separate interaction groups, A and C cannot both be split on in the same path from root to leaf.

### Implementation Approach

XGBoost's approach: maintain a set of "allowed features" per node. When a node is split on feature `i`, the child nodes can only split on features that are in the same interaction group as `i`, plus features in groups that haven't been used yet on this path.

#### Step C.1: Add constraint storage

```rust
pub struct TrainParams {
    // ... existing fields ...
    pub interaction_constraints: Vec<Vec<usize>>,  // groups of feature indices
}
```

Empty = no constraints. Each inner Vec is a group of features that may interact.

#### Step C.2: Track allowed features per node

In the tree building loop, each node needs a set of allowed feature indices. When splitting node `n` on feature `f`:
- Find which group(s) contain `f`
- Children of `n` can split on features in those groups, plus features in groups not yet used on this path

This requires tracking the split history per path, which is a moderate change to the tree building loop. The `active_nodes` vec would need to carry an additional "allowed features" mask.

#### Step C.3: Filter features in histogram scan

When calling `backend.best_split_with_options`, pass an allowed-features mask. The backend should skip histograms for disallowed features.

Alternatively, build histograms for all features but only evaluate splits for allowed ones. Less efficient but simpler.

#### Step C.4: Python API

```python
interaction_constraints: list[list[int]] | None = None
```

### Testing

- Train with `interaction_constraints=[[0, 1], [2, 3]]`, inspect trees to verify no path has both a feature from group 1 and group 2
- Verify unconstrained training is unaffected (empty list = all features can interact)

### Complexity: Medium-High (~150-200 lines, requires carrying per-node state through tree building)

---

## Feature D: `max_leaves` (Leaf Count Limit for Level-Wise Growth)

### Overview

Even without switching to leaf-wise growth (Limitation #15), a `max_leaves` parameter can limit the total number of leaves per tree in level-wise mode. This is simpler than leaf-wise growth and provides a useful control knob.

### Implementation

#### Step D.1: Add to `IterationControls`

```rust
pub struct IterationControls {
    // ... existing fields ...
    pub max_leaves_per_tree: Option<usize>,  // None = unlimited (depth-limited only)
}
```

#### Step D.2: Enforce in tree building loop

Track leaf count during level-by-level expansion. When `max_leaves_per_tree` is reached, stop expanding even if `max_depth` hasn't been hit.

In the tree building loop (line ~1227), add a `leaf_count` counter. Each split adds one leaf (splits a leaf into two, net +1). When `leaf_count >= max_leaves`, break.

Implementation subtlety: with level-wise growth, all nodes at a level are split before moving to the next. If adding all splits at level `d` would exceed `max_leaves`, need a strategy:
- Option A: Split all nodes at level `d` anyway (may slightly exceed `max_leaves`)
- Option B: At level `d`, only split the top-N nodes by gain that fit within `max_leaves`

Recommendation: Option B is more precise but requires sorting splits by gain within a level. Option A is simpler and acceptable as a first implementation.

#### Step D.3: Add to Python API

```python
max_leaves: int | None = None  # None means depth-limited only
```

### Complexity: Low-Medium (~50-80 lines)

---

## Feature E: Custom Evaluation Metric Callback

### Overview

Currently, only RMSE is tracked during training (via `squared_error_loss`). Users can't track MAE, R², or custom metrics alongside. This is related to Limitation #13 (Only RMSE Tracked During Training).

### Implementation

This is better covered in the Training Metric Tracking plan (#13). Mentioning here for completeness.

---

## Feature F: `max_bin` Per Feature

### Overview

Currently all features share the same bin count (up to 256, set by `continuous_binning_max_bins`). Some features may benefit from more or fewer bins. However, the `BinnedMatrix` uses `Vec<u8>` (max 256), so per-feature bin counts can only go *down* from 256, not up. Going above 256 requires changing the bin storage type (Limitation #7).

### Implementation

#### Step F.1: Accept per-feature max bins

```python
per_feature_max_bins: dict[int, int] | None = None  # feature_index -> max_bins
```

#### Step F.2: Apply during binning

In the binning step (`prepare_training_matrices_from_dense_values`), when computing bin thresholds for each feature, use the feature-specific max_bins if provided.

This requires changes in the binning logic in `bindings/python/src/lib.rs` where continuous features are quantized.

### Complexity: Medium (~80-120 lines, touches binning internals)

### Recommendation: Lower priority -- the global 256-bin cap is adequate for most use cases. Address alongside Limitation #7 if bin cap is raised.

---

## Feature G: Feature Importance Weighting at Split Time

### Overview

Allow users to specify per-feature weights that influence split selection. A feature with weight 0.5 would need 2x the gain to be selected over a feature with weight 1.0. This is different from `col_subsample` (which randomly excludes features) -- it's a deterministic bias.

### Implementation

#### Step G.1: Add to `TrainParams`

```rust
pub feature_weights: Vec<f32>,  // per-feature weights, empty = uniform
```

#### Step G.2: Apply in split selection

In `best_split_for_feature` (`backend_cpu/src/lib.rs:416`), multiply the computed gain by the feature's weight:

```rust
let weighted_gain = gain * feature_weight;
if weighted_gain > best_gain { ... }
```

Or alternatively, compare `gain / feature_weight > best_gain / best_weight` to avoid biasing absolute gain values.

#### Step G.3: Python API

```python
feature_weights: list[float] | dict[int, float] | None = None
```

### Complexity: Low (~40-60 lines)

---

## Implementation Priority

Recommended implementation order based on user value and complexity:

| Priority | Feature | Complexity | User Value |
|----------|---------|-----------|------------|
| 1 | A: `min_split_gain` | Very Low | High -- already exists internally |
| 2 | B: Monotone constraints | Medium | Very High -- domain knowledge |
| 3 | D: `max_leaves` | Low-Medium | Medium -- useful control knob |
| 4 | C: Interaction constraints | Medium-High | Medium -- advanced use case |
| 5 | G: Feature importance weighting | Low | Low-Medium |
| 6 | F: Per-feature `max_bin` | Medium | Low |

Features A-D should be the main focus. Features E, F, G are nice-to-haves.

---

## Files Changed Summary

| File | Features Affected |
|------|------------------|
| `crates/core/src/lib.rs` | A, B, C, D, G (TrainParams fields + validation) |
| `crates/engine/src/lib.rs` | A, B, C, D (IterationControls, tree building loop) |
| `crates/backend_cpu/src/lib.rs` | B, C, G (split selection, feature filtering) |
| `bindings/python/src/lib.rs` | A, B, C, D, F, G (bridge param passing) |
| `bindings/python/alloygbm/regressor.py` | A, B, C, D, F, G (constructor, get/set_params) |

---

## Risk Areas and Troubleshooting

### Monotone Constraint + Auto Policy Interaction

The auto policy adjusts `min_split_gain` and regularization heuristically. Monotone constraints could cause many splits to be rejected, leading to very shallow trees. The auto policy doesn't account for this. Consider: if monotone constraints are active, relax `min_split_gain` slightly or warn the user.

### Interaction Constraint Complexity

Interaction constraints require per-node state tracking through the tree building loop. The current loop doesn't carry per-node metadata beyond `(node_id, row_indices, histograms, parent_leaf_value)`. Adding an allowed-features mask increases memory per active node. For deep trees with many active nodes, this could be significant.

Mitigation: Use a compact bitset (e.g., `Vec<u64>` where each bit represents a feature) instead of `Vec<usize>`.

### `max_leaves` + Level-Wise Growth

The current level-wise growth processes all nodes at a depth before moving to the next. With `max_leaves`, some nodes at a level may need to be skipped. This changes the tree shape from "balanced" to "unbalanced" -- the predictor and SHAP code should handle this correctly since they already support arbitrary tree shapes (via node_id -> split/leaf mapping).

### Backward Compatibility

All new parameters have sensible defaults that reproduce current behavior:
- `min_split_gain = 0.0` (auto policy still applies its heuristic)
- `monotone_constraints = []` (no constraints)
- `interaction_constraints = []` (no constraints)
- `max_leaves = None` (depth-limited only)
- `feature_weights = []` (uniform weighting)

No existing behavior changes unless the user explicitly sets new parameters.

---

## Testing Strategy

### Per-Feature Tests

Each feature (A-G) should have:
1. **Parameter validation**: Invalid inputs rejected (negative values, wrong lengths, etc.)
2. **Default behavior**: With default params, behavior is identical to current code
3. **Functional correctness**: The constraint/limit actually works as specified
4. **Edge cases**: All features constrained, zero leaves allowed, empty constraint lists

### Integration Tests

1. **Monotone + regularization**: Verify monotone constraints work correctly with L1/L2 regularization
2. **Multiple new params**: Set `min_split_gain`, `monotone_constraints`, and `max_leaves` together
3. **Auto policy interaction**: Verify auto policy doesn't override user-set `min_split_gain`

### Python Tests

1. `get_params()` / `set_params()` roundtrip with all new params
2. `__repr__` includes new params
3. Constructor validation for all new params

---

## Non-Goals (Out of Scope)

- **Leaf-wise (best-first) growth**: Covered in Limitation #15 plan
- **DART / GOSS boosting modes**: Major algorithm additions beyond configurability
- **Custom objective from Python**: Requires callback protocol design, separate initiative
- **Custom evaluation metric during training**: Covered in Limitation #13 plan
- **`scale_pos_weight`**: Moot without classification support (Limitation #1)
- **Raising bin cap above 256**: Covered in Limitation #7 plan
