# Cross-Plan Conflict Resolution Guide

## Purpose

This document identifies every place where two or more limitation plans touch the same code region, and prescribes how to resolve conflicts as plans are implemented 1-2 at a time. It also recommends an implementation order that minimizes merge pain.

---

## Table of Contents

1. [Recommended Implementation Order](#recommended-implementation-order)
2. [Hot Spot Map: Shared Code Regions](#hot-spot-map)
3. [Conflict Catalog](#conflict-catalog)
4. [Design Decisions That Must Be Made Once](#design-decisions)
5. [Per-Plan Dependency Graph](#dependency-graph)
6. [Migration Patterns](#migration-patterns)

---

## Recommended Implementation Order

Plans are grouped into tiers. Within a tier, order is flexible. Across tiers, earlier tiers should land first.

### Tier 0: Standalone / No Conflicts
These can be implemented at any time with no dependencies:

| Plan | Rationale |
|------|-----------|
| #5 Model Persistence | Pure Python addition, no Rust changes |
| #7 SHAP Feature Limit (quick fix only) | 1-line constant change, fully isolated |
| #11 Feature Names | Uses existing metadata field, additive |
| #8 sklearn Compatibility | Pure Python class inheritance, additive |

### Tier 1: Foundation Layer
These establish patterns that later plans depend on:

| Plan | Rationale |
|------|-----------|
| #10 Sample Weight Support | Pure plumbing, unlocks weighted loss for later plans |
| #11 Group ID Support | Pure plumbing, prerequisite for ranking in Plan #1 |
| #4 NaN Support | Cross-cutting but establishes MISSING_BIN convention before Plan #7 changes bin storage |
| #3 Expanded Configurability (Feature A only: min_split_gain) | Trivial, establishes pattern for adding TrainParams fields |

### Tier 2: Engine Generalization
These make fundamental engine changes:

| Plan | Rationale |
|------|-----------|
| #1 Classification & Ranking | Replaces 6 hardcoded `squared_error_loss()` calls with trait dispatch. Every plan that touches loss monitoring depends on this being done first. |
| #13 Training Metric Tracking | Extends the generalized loss from Plan #1 to support multiple metrics. Must come after Plan #1. |

### Tier 3: Structural Changes
These modify tree building or data storage:

| Plan | Rationale |
|------|-----------|
| #3 Expanded Configurability (Features B-G) | Monotone constraints, interaction constraints, max_leaves, feature weights |
| #7 Bin Cap Increase | Changes BinnedMatrix storage type. Must come after Plan #4 (NaN) to coordinate bin sentinel values. |
| #15 Leaf-Wise Tree Growth | Adds alternative tree building loop. Coordinate with Plan #3 on `max_leaves` semantics. |
| #16 Histogram Caching | Performance optimization, should come after Plans #4, #7, #15 settle histogram kernel code. |

### Tier 4: Multi-Feature and Warm-Start
These build on the stable foundation:

| Plan | Rationale |
|------|-----------|
| #2 Multiple Categorical Columns | Generalizes encoding. Low conflict but touches bridge heavily. |
| #14 Warm-Starting | Needs stable artifact format (after Plans #1, #4 extend it). |
| #8 SHAP Feature Limit (TreeSHAP) | Full TreeSHAP algorithm, algorithmically complex but isolated to `crates/shap/`. |

---

## Hot Spot Map

These are the code regions that the most plans touch. **When implementing any plan, check this map to see what other plans also modify this region.**

### 🔴 Critical Hot Spots (4+ plans touch)

#### Hot Spot A: `TrainParams` struct (`crates/core/src/lib.rs:35-67`)
**Plans that add fields: #1, #3, #14**

| Plan | Fields Added |
|------|-------------|
| #1 | (none directly -- objective is passed separately, not stored in TrainParams) |
| #3 | `min_split_gain: f32`, `monotone_constraints: Vec<i8>`, `interaction_constraints: Vec<Vec<usize>>`, `feature_weights: Vec<f32>` |
| #14 | `tree_growth: TreeGrowthMode` (enum), reuses #3's `max_leaves` |

**Conflict**: Plans #3 and #14 both want `max_leaves`. See [Decision D1](#d1-max_leaves-semantics).

**Pattern for adding fields**: Always add new fields at the end of the struct. Always add a default value in `impl Default`. Always add validation in `validate_train_params()`. This way each PR adds fields without conflicting with other PRs' field additions.

---

#### Hot Spot B: `IterationControls` struct (`crates/engine/src/lib.rs:301-313`)
**Plans that modify: #3, #12, #14**

| Plan | Change |
|------|--------|
| #3 | `min_split_gain` already exists (wire user value), add `max_leaves_per_tree: Option<usize>` |
| #12 | Add `eval_metrics: Vec<EvalMetric>` or similar |
| #14 | Add `tree_growth_mode: TreeGrowthMode`, reuse `max_leaves_per_tree` |

**Pattern**: Same as TrainParams -- append fields, provide defaults, add validation. The `auto_iteration_controls()` function (~line 1039) must be updated for each new field.

---

#### Hot Spot C: Training Loop (`crates/engine/src/lib.rs:1110-1380`)
**Plans that modify: #1, #3, #4, #9, #12, #14**

This is the most dangerous hot spot. The training loop is ~270 lines of dense logic.

| Sub-region | Plans | What changes |
|-----------|-------|-------------|
| Lines ~1155-1170: initial/per-round loss computation | #1, #9, #12 | Replace `squared_error_loss()` with trait dispatch (#1), add weighted loss (#9), add multi-metric (#12) |
| Lines ~1227-1232: level-wise depth loop start | #14 | Add conditional branch for leaf-wise vs level-wise |
| Lines ~1237-1244: best split + gain threshold | #3 | Add monotone constraint check, user-controlled min_split_gain |
| Lines ~1246-1262: partition + row count check | #4 | Add NaN routing (default_left) in partitioning |
| Lines ~1270-1290: leaf value computation | #3 | Add monotone constraint clamping |
| Lines ~1379-1421: validation loss tracking | #1, #9, #12 | Same generalization as training loss |
| Lines ~1505-1521: early stopping comparison | #1, #12 | Use primary eval_metric for early stopping |

**Recommended approach**: Each plan should modify *its specific sub-region* and leave the rest untouched. The sub-regions are fairly well-separated. The risk is when two plans touch the same sub-region:

- **Loss computation** (lines ~1155-1170, ~1379-1421, ~1505-1521): Plan #1 must go first (generalize from hardcoded MSE to trait dispatch). Then #12 extends (multi-metric). Then #9 adds weight awareness to the already-generalized loss.
- **Split finding** (lines ~1237-1290): Plan #3 (constraints) and Plan #4 (NaN routing) touch adjacent but non-overlapping code. Can be done in either order.
- **Tree building algorithm** (line ~1227): Plan #14 adds a branch, not a modification. The existing level-wise code stays intact.

---

#### Hot Spot D: Bridge Training Functions (`bindings/python/src/lib.rs`)
**Plans that add parameters: #1, #2, #3, #9, #10, #11, #12, #13, #14**

The 5 training pyfunctions each have ~15 parameters. Nearly every plan adds 1-3 more.

| Plan | Parameters Added |
|------|-----------------|
| #1 | `objective: String` |
| #2 | `categorical_feature_indices: Option<Vec<usize>>`, `categorical_feature_values: Option<HashMap<usize, Vec<String>>>` (replacing singular versions) |
| #3 | `min_split_gain: f32`, `monotone_constraints: Vec<i8>`, `interaction_constraints: Vec<Vec<usize>>`, `max_leaves: Option<usize>`, `feature_weights: Vec<f32>` |
| #9 | `sample_weights: Option<Vec<f32>>`, `validation_sample_weights: Option<Vec<f32>>` |
| #10 | `group_id: Option<Vec<u32>>`, `validation_group_id: Option<Vec<u32>>` |
| #11 | `feature_names: Option<Vec<String>>` |
| #12 | `eval_metric: Option<Vec<String>>` |
| #13 | `init_artifact_bytes: Option<Vec<u8>>` |
| #14 | `tree_growth: String` |

**This will reach 30+ parameters per function.** See [Decision D3](#d3-bridge-parameter-grouping).

---

#### Hot Spot E: `GBMRegressor.__init__()` (`bindings/python/alloygbm/regressor.py:182-290`)
**Plans that add constructor params: #2, #3, #8, #13, #14**

| Plan | Parameters Added |
|------|-----------------|
| #2 | `categorical_feature_indices: list[int] | None` (replacing singular `categorical_feature_index`) |
| #3 | `min_split_gain: float`, `monotone_constraints`, `interaction_constraints`, `max_leaves`, `feature_weights` |
| #8 | (no new params, but adds `warm_start: bool` attribute) |
| #13 | `warm_start: bool` |
| #14 | `tree_growth: str`, `max_leaves: int | None` |

**Pattern**: Each plan appends new params. The `get_params()`, `set_params()`, `_params_order` list, and `__repr__` must all be updated in sync. When implementing a plan, always update all four together.

---

#### Hot Spot F: `GBMRegressor.fit()` signature (`bindings/python/alloygbm/regressor.py:~520-600`)
**Plans that add fit() kwargs: #2, #9, #10, #11, #12, #13**

| Plan | Keyword Args Added to fit() |
|------|----------------------------|
| #2 | Changes `categorical_feature_values` type from `list[str]` to `dict[int, list[str]]` |
| #9 | `sample_weight`, `eval_sample_weight` |
| #10 | `group`, `eval_group` |
| #11 | `feature_name: list[str] | None` |
| #12 | `eval_metric: str | list[str]` (or this might be a constructor param instead) |
| #13 | `init_model: GBMRegressor | None` |

**Pattern**: All are keyword-only args with `None` defaults, so they're non-breaking. But the fit() body grows with each plan's validation and forwarding logic. Keep each plan's validation in a dedicated helper method (e.g., `_validate_sample_weight()`, `_validate_group()`) to keep fit() readable.

---

#### Hot Spot G: Artifact Format (`crates/core/src/lib.rs` + `crates/engine/src/lib.rs`)
**Plans that extend the artifact: #1, #4, #11**

| Plan | Format Change |
|------|--------------|
| #1 | Add `objective` and `num_classes` to ModelMetadata JSON |
| #4 | Add `default_left: bool` per split node in binary section |
| #11 | Use existing `feature_names` field in ModelMetadata JSON (no format change) |

**Backward compatibility rule**: Old artifacts must load in new code. New artifacts should degrade gracefully in old code (or fail with a clear version error).

- **Plan #1**: Add `"objective":"squared_error"` to metadata JSON. Old artifacts without this field default to `"squared_error"`.
- **Plan #4**: Add `default_left` bit to split nodes. Old artifacts without this bit default to `false` (NaN goes right). Requires bumping the section format version.

**Conflict**: If Plans #1 and #4 are implemented close together, the metadata JSON parser (`deserialize_metadata_json` in core/src/lib.rs:1125) and the split node binary format both change. The hand-rolled JSON parser uses positional `consume_literal` calls -- adding fields requires careful ordering. See [Decision D4](#d4-metadata-parser-strategy).

---

#### Hot Spot H: `best_split_for_feature()` (`crates/backend_cpu/src/lib.rs:416-512`)
**Plans that modify split finding: #3, #4, #6**

| Plan | Change |
|------|--------|
| #3 (B: Monotone) | After finding best split, check leaf value ordering |
| #3 (G: Feature weights) | Multiply gain by feature weight |
| #4 | For each threshold, try NaN-left and NaN-right, pick better |
| #6 | Histogram index type changes (u8 -> u16 for > 256 bins) |

These changes are mostly additive to the inner loop. The execution order matters:

1. **Plan #4** first: doubles the gain computation (two NaN directions). This restructures the inner loop.
2. **Plan #3 (G)** second: applies weight to the already-computed gain. Non-conflicting.
3. **Plan #3 (B)** third: post-split validation, outside the inner loop.
4. **Plan #6** changes the bin index type but not the gain logic. Can be done independently.

---

### 🟡 Moderate Hot Spots (2-3 plans touch)

#### `squared_error_loss()` (engine/src/lib.rs:~2694) and its 6 call sites
**Plans: #1, #9, #12**

Plan #1 replaces this with trait-dispatched `objective.loss()`. Plan #9 adds weight awareness. Plan #12 adds multi-metric computation. **Do Plan #1 first** -- it eliminates the 6 hardcoded call sites. Then #12 and #9 build on the generalized interface.

#### `SplitSelectionOptions` (engine/src/lib.rs:48-54)
**Plans: #3, #4**

Plan #3 adds `monotone_constraints: &[i8]` and `feature_weights: &[f32]`. Plan #4 doesn't modify this struct (NaN direction is computed inside `best_split_for_feature`). Low conflict.

#### `CategoricalStatePayloadV1` and encoding functions (core + engine)
**Plans: #2**

Only Plan #2 touches this. The existing format already supports Vec<u32> indices. Low conflict.

#### Predictor (`crates/predictor/src/lib.rs`)
**Plans: #1, #4**

Plan #1 adds post-prediction transforms (sigmoid/softmax). Plan #4 adds NaN routing. Both are additive and non-conflicting -- NaN routing happens at node traversal, transforms happen after final prediction.

#### SHAP (`crates/shap/src/lib.rs`)
**Plans: #8**

Only Plan #8 touches this. Fully isolated. No conflicts.

---

## Conflict Catalog

Each entry describes a specific conflict, which plans are involved, and the resolution.

### C1: `max_leaves` Field Ownership
**Plans: #3 (Feature D), #14 (Leaf-Wise Growth)**

Both plans add `max_leaves` but with different semantics:
- Plan #3: Optional secondary constraint in level-wise growth
- Plan #14: Primary stopping criterion in leaf-wise growth

**Resolution**: Use a single `max_leaves: Option<usize>` field in both `TrainParams` and `IterationControls`. Interpretation depends on `tree_growth` mode:
- If `tree_growth = "level"`: `max_leaves` is a secondary limit (stop expanding when reached, even if max_depth not hit)
- If `tree_growth = "leaf"`: `max_leaves` is the primary limit (required, error if None)

**Which plan implements it**: Whichever lands first defines the field. The second plan adjusts semantics if needed. Recommendation: Plan #3 adds the field first (it's simpler), Plan #14 adds the `tree_growth` mode and adjusts the field's interpretation.

---

### C2: Loss Computation Generalization Order
**Plans: #1, #9, #12**

All three plans touch the loss computation in the training loop:
- Plan #1: Replace `squared_error_loss()` with `objective.loss()`
- Plan #9: Make loss computation weight-aware
- Plan #12: Compute multiple metrics per round

**Resolution**: Strict ordering: **#1 → #12 → #9**.
1. Plan #1 generalizes loss to trait dispatch (single metric)
2. Plan #12 extends to multiple metrics using the trait
3. Plan #9 adds weights to the already-generalized, multi-metric framework

If #9 is implemented before #1, the weighted loss would be added to `squared_error_loss()` directly. Then when #1 lands, it must also generalize the weighted version. This works but is more refactoring.

If #12 is implemented before #1, it would add multi-metric around the still-hardcoded `squared_error_loss()`. Then #1 must generalize each metric call. Again workable but messier.

---

### C3: Missing Bin vs. Bin Cap
**Plans: #4, #7**

Plan #4 reserves bin 255 as MISSING_BIN (NaN sentinel). Plan #7 introduces u16 bin storage for > 256 bins.

**Resolution**:
- **u8 mode** (max_bins ≤ 255): MISSING_BIN = 255. Max usable bins = 255 (0..254 for values, 255 for NaN). This is the common case.
- **u16 mode** (max_bins > 255): MISSING_BIN = 65535 (u16::MAX). Max usable bins = 65535.
- Define `MISSING_BIN` as a function of the storage type, not a compile-time constant:
  ```rust
  impl BinStorage {
      pub fn missing_bin(&self) -> u16 {
          match self {
              Self::U8(_) => 255,
              Self::U16(_) => 65535,
          }
      }
  }
  ```

**Which plan goes first**: Plan #4 first (establish the sentinel convention). Plan #7 then generalizes the sentinel to match the storage type.

---

### C4: Histogram Kernel Specialization
**Plans: #4, #6, #3 (B, G), #16**

Multiple plans modify histogram building or split finding:
- Plan #4: Separate NaN-bin accumulation, try two directions
- Plan #6: Dual u8/u16 dispatch in kernels
- Plan #3: Feature filtering (interaction constraints), gain weighting (feature weights)
- Plan #16: Buffer reuse, data pre-sorting

**Resolution**: Layer the changes:
1. **Outer dispatch** (Plan #6): Match on `BinStorage::U8` vs `BinStorage::U16`, call type-specialized inner kernels
2. **Inner kernel** (Plans #4, #3): Within each type-specialized kernel, handle NaN accumulation and feature filtering
3. **Buffer reuse** (Plan #16): Orthogonal -- affects allocation, not kernel logic
4. **Data pre-sorting** (Plan #16): Orthogonal -- affects row traversal order, not accumulation logic

The key insight: Plan #6's type dispatch is the outermost layer. Plans #4 and #3 modify the logic inside each specialized kernel. Plan #16 affects how buffers and row indices are managed, not the kernel math.

---

### C5: Bridge Parameter Explosion
**Plans: All**

See [Decision D3](#d3-bridge-parameter-grouping) below.

---

### C6: fit() Body Growth
**Plans: #2, #9, #10, #11, #12, #13**

Each plan adds validation logic and forwarding code to `fit()`.

**Resolution**: Use the **helper method pattern**. Each plan's validation goes in a dedicated private method:
```python
def fit(self, X, y, *, sample_weight=None, group=None, feature_name=None, ...):
    # ...existing validation...
    sample_weight = self._validate_sample_weight(sample_weight, n_rows)  # Plan #9
    group = self._validate_group(group, n_rows)                          # Plan #10
    feature_names = self._resolve_feature_names(X, feature_name)         # Plan #11
    # ...
```

The `fit()` body stays linear and readable. Each helper method is self-contained and doesn't conflict with others.

---

### C7: Artifact Metadata JSON Parser
**Plans: #1, #4**

The hand-rolled JSON parser in `core/src/lib.rs:1125` uses positional `consume_literal` calls. It's brittle -- adding fields requires careful ordering.

**Resolution**: See [Decision D4](#d4-metadata-parser-strategy). In short: switch to serde_json or make the parser order-independent before Plans #1 and #4 add new fields.

---

### C8: Warm-Start + Artifact Stability
**Plan #13 depends on: #1, #4**

Warm-starting loads a previous model's artifact and continues training. If the artifact format changes (Plans #1, #4), warm-starting must handle both old and new artifact versions.

**Resolution**: Implement Plan #13 after Plans #1 and #4 have settled the artifact format. Or, implement #13 first against the current format and let #1/#4 update the warm-start path when they land.

---

### C9: sklearn Pickle + Persistence
**Plans: #5, #8**

Plan #5 implements `__getstate__`/`__setstate__` for pickle. Plan #8's `check_estimator` also tests pickling.

**Resolution**: No real conflict. If Plan #5 lands first, Plan #8 gets pickle support for free. If Plan #8 lands first without pickle, `check_estimator` will flag the pickle test as failing -- Plan #5 then fixes it.

---

### C10: Categorical `categorical_feature_index` → `categorical_feature_indices`
**Plan #2**

Plan #2 renames the singular parameter to plural. This is a breaking API change in Python.

**Resolution**: Keep both parameters for backward compat. The old singular `categorical_feature_index` converts to `[index]` internally. Emit a deprecation warning. The Rust bridge accepts the new plural form; the old singular form is handled entirely in Python.

This is a **one-time migration** that doesn't conflict with other plans.

---

### C11: Training Summary Format
**Plans: #1, #12, #13**

`IterationRunSummary` currently has `loss_per_completed_round: Vec<f32>` (single metric). Plans #1 and #12 extend this. Plan #13 (warm-start) needs to concatenate summaries across warm-start sessions.

**Resolution**:
- Plan #1 renames `loss_per_completed_round` to be objective-agnostic (or adds `objective_loss_per_round`)
- Plan #12 adds `additional_metrics_per_round: HashMap<String, Vec<f32>>`
- Plan #13 concatenates the existing vectors when warm-starting

No direct conflict if fields are additive. Plan #13 must handle whichever fields exist at the time it's implemented.

---

## Design Decisions That Must Be Made Once

These decisions affect multiple plans. Make the decision before implementing the first plan that touches the area.

<a id="d1-max_leaves-semantics"></a>
### D1: `max_leaves` Semantics

**Affected plans**: #3, #14

**Decision**: Single `max_leaves: Option<usize>` field in `TrainParams`. Add `tree_growth: TreeGrowthMode` enum (`LevelWise`, `LeafWise`). Semantics:
- `LevelWise` + `max_leaves = Some(n)`: Secondary constraint (stop expanding levels when n leaves reached)
- `LevelWise` + `max_leaves = None`: Depth-limited only (current behavior)
- `LeafWise` + `max_leaves = Some(n)`: Primary constraint (required)
- `LeafWise` + `max_leaves = None`: Error at validation time

**When to decide**: Before implementing either Plan #3 Feature D or Plan #14.

---

<a id="d2-missing-bin-sentinel"></a>
### D2: Missing Bin Sentinel Value

**Affected plans**: #4, #7

**Decision**: Define `missing_bin()` as a method on the bin storage type, not a global constant:
- `BinStorage::U8`: missing = 255
- `BinStorage::U16`: missing = 65535

If Plan #4 lands before Plan #7, use `const MISSING_BIN: u8 = 255`. When Plan #7 lands, refactor to the method-based approach.

**When to decide**: Before implementing Plan #4.

---

<a id="d3-bridge-parameter-grouping"></a>
### D3: Bridge Parameter Grouping Strategy

**Affected plans**: All plans that add bridge parameters (#1, #2, #3, #9, #10, #11, #12, #13, #14)

**Decision**: Introduce parameter structs to group related arguments. Options:

**Option A: Rust-side parameter structs**
```rust
struct CategoricalOptions {
    feature_indices: Option<Vec<usize>>,
    feature_values: Option<HashMap<usize, Vec<String>>>,
    smoothing: f64,
    min_samples_leaf: u32,
    time_aware: bool,
}

struct DatasetOptions {
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    feature_names: Option<Vec<String>>,
}
```

Pro: Clean Rust API. Con: PyO3 struct mapping adds complexity.

**Option B: Keep flat parameters but consolidate into fewer bridge functions**
Instead of 5 training pyfunctions, have 1 universal function that accepts all parameters.

Pro: Single function to maintain. Con: One massive function signature.

**Option C: Accept a Python dict of options**
```rust
#[pyo3(signature = (values, targets, params, **options))]
fn train_model(values: &[f32], targets: &[f32], params: TrainParams, options: Option<&PyDict>) -> ...
```

Pro: Infinitely extensible without signature changes. Con: Loses type safety, harder to document.

**Recommendation**: Option A for the Rust side (cleaner, type-safe), with PyO3's `#[pyo3(from_py_with = "...")]` for conversion. Start by grouping into 3-4 structs as plans land. Don't try to design the final grouping upfront -- let it emerge.

**When to decide**: Before the 3rd plan adds bridge parameters. The first 2 plans can add flat params without grouping. Once the pattern becomes painful, refactor into structs.

---

<a id="d4-metadata-parser-strategy"></a>
### D4: Metadata JSON Parser Strategy

**Affected plans**: #1, #4, #11

**Current state**: `serialize_metadata_json` and `deserialize_metadata_json` in `core/src/lib.rs` are hand-rolled with positional `consume_literal` calls. Adding fields requires appending to a fixed-order format string and parser.

**Decision**: Options:
1. **Keep hand-rolled, extend carefully**: Add new fields at the end of the JSON object. Parser uses `consume_literal` for known fields, ignores unknown trailing content. Simple but fragile.
2. **Switch to serde_json**: Add `serde` and `serde_json` dependencies. Use `#[derive(Serialize, Deserialize)]` with `#[serde(default)]` for backward compat. Robust, standard, allows field reordering.
3. **Hybrid**: Keep hand-rolled serialization (no new deps) but make parser more flexible (scan for field names rather than assuming position).

**Recommendation**: Option 2 (serde_json). The workspace already forbids unsafe code, and serde is a standard Rust dependency. The investment pays off immediately since Plans #1, #4, and #11 all need new metadata fields. With `#[serde(default)]`, backward compatibility is automatic.

**When to decide**: Before implementing Plan #1 (which adds `objective` and `num_classes` to metadata).

**If serde is unacceptable**: Use Option 3. Make the parser scan for `"field_name":` rather than assuming field order. Still hand-rolled but robust against field reordering.

---

### D5: `GBMRegressor` vs. `GBMClassifier` Shared Infrastructure

**Affected plans**: #1, #2, #3, #5, #8, #9, #10, #11, #12, #13, #14

**Decision**: When Plan #1 adds `GBMClassifier`, how much code should be shared with `GBMRegressor`?

Options:
1. **Copy and adapt**: Duplicate `GBMRegressor` into `GBMClassifier`, modify as needed. Fast to implement, maintenance burden.
2. **Base class extraction**: Extract common logic into `_GBMBase` class, have both inherit from it. Clean, but requires refactoring `GBMRegressor` first.
3. **Composition**: Both classes delegate to a shared `_GBMModel` internal object. Maximum code reuse but more indirection.

**Recommendation**: Option 2 (base class extraction). Extract `_GBMBase` with all the shared logic (fit flow, prediction infrastructure, parameter management, persistence, sklearn compat). `GBMRegressor` and `GBMClassifier` only override objective selection, post-transforms, `score()`, and validation.

**When to decide**: Before implementing Plan #1's Python API phase (Phase 5 in that plan).

**Impact on other plans**: If other plans (e.g., #5, #8) land before #1, they modify `GBMRegressor` directly. When #1 later extracts `_GBMBase`, those changes need to be moved into the base class. This is a refactoring concern, not a conflict -- but it's easier if #1 goes early.

---

### D6: Evaluation Metric Location (Constructor vs. fit())

**Affected plans**: #12, #1

**Decision**: Should `eval_metric` be a constructor parameter or a `fit()` parameter?

- **Constructor param**: Set once, applies to all `fit()` calls. Consistent with LightGBM.
- **fit() param**: Can change between calls. More flexible.

**Recommendation**: Constructor parameter (`GBMRegressor(eval_metric="rmse")`) with an optional `fit()` override. This matches LightGBM's convention and works well with sklearn's `get_params()`/`set_params()`.

---

## Per-Plan Dependency Graph

```
Plan #10 (Group ID) ──────────────────────────────────────┐
Plan #9  (Sample Weight) ────────────────────────────┐    │
Plan #11 (Feature Names) ─────────────────────┐      │    │
                                               │      │    │
Plan #4  (NaN Support) ──┐                    │      │    │
                         │                    │      │    │
Plan #7  (Bin Cap) ──────┤ (needs #4 first)  │      │    │
                         │                    │      │    │
Plan #1  (Classification) ◄───────────────────┤──────┤────┤
         │                                    │      │    │
         ▼                                    │      │    │
Plan #12 (Metrics) ◄──────────────────────────┘──────┘    │
         │                                                 │
         ▼                                                 │
Plan #3  (Configurability) ───────────────────────────┐   │
         │                                             │   │
         ▼                                             │   │
Plan #15 (Leaf-Wise) ◄────────────────────────────────┘   │
         │                                                 │
         ▼                                                 │
Plan #16 (Histogram Cache) ──────────────────────────────  │
                                                           │
Plan #5  (Persistence) ───── independent ──────────────    │
Plan #8  (sklearn) ────────── independent ──────────────   │
Plan #14 (Warm-Start) ◄───────────────────────────────────┘
         │ (needs stable artifact format from #1, #4)
         ▼
Plan #2  (Multi-Categorical) ── independent ──────────────
Plan #8  (SHAP TreeSHAP) ────── independent ──────────────
```

Arrows indicate "should land before". Plans without arrows to each other can be done in any order.

---

## Migration Patterns

These patterns help when implementing plan X after plan Y has already landed.

### Pattern M1: Adding a TrainParams Field After Other Fields Were Added

When adding a field to `TrainParams`:
1. Add field at end of struct definition
2. Add default in `impl Default`
3. Add validation in `validate_train_params()`
4. Add to `IterationControls::new()` if applicable
5. Add to bridge `TrainParams` construction in all 5 pyfunctions
6. Add to `GBMRegressor.__init__()`, `get_params()`, `set_params()`, `__repr__`, `_params_order`
7. Add to `NativeTrainingSummary` if it affects training output

Each step is a different file/location, so merge conflicts are per-line, not per-block.

### Pattern M2: Adding a fit() Keyword Argument After Others Were Added

1. Add kwarg with `None` default to `fit()` signature
2. Add validation helper method `_validate_<name>()`
3. Call validation in fit() body (insert at appropriate point)
4. Forward to native training call
5. Update bridge function signatures

Each kwarg is independent -- they don't interact in the function signature. The only conflict risk is the bridge function, where parameter lists get very long.

### Pattern M3: Extending the Artifact Format After Previous Extensions

1. Check current `format_version` in metadata
2. Add new field with `#[serde(default)]` (or hand-roll with optional parsing)
3. Old artifacts without the field get the default value
4. Increment format_version if binary section layout changes (but NOT for JSON-only additions)

### Pattern M4: Modifying the Training Loop After Previous Modifications

1. Identify which sub-region your change touches (see Hot Spot C table)
2. Read the current code at that sub-region (it may have changed since the plan was written)
3. Make your change in the smallest possible scope
4. Run the existing test suite to verify you didn't break other plans' changes

---

## Quick Reference: "I'm Implementing Plan X, What Else Do I Need to Know?"

| Plan | Read These Conflicts | Coordinate With | Must Land After |
|------|---------------------|-----------------|-----------------|
| #1 (Classification) | C2, C7, C11 | D4, D5, D6 | (none -- foundation) |
| #2 (Multi-Categorical) | C10 | (none) | (independent) |
| #3 (Configurability) | C1, C4 | D1 | (Feature A: independent; Features B-G: after #4 for clean split-finding) |
| #4 (NaN Support) | C3, C4, C7 | D2, D4 | (independent, but ideally before #7) |
| #5 (Persistence) | C9 | (none) | (independent) |
| #7 (Bin Cap) | C3, C4 | D2 | After #4 (coordinate MISSING_BIN) |
| #8 (SHAP) | (none) | (none) | (independent) |
| #9 (sklearn) | C9 | (none) | (independent) |
| #10 (Sample Weight) | C2 | (none) | Before or after #1 (either order works) |
| #11 (Group ID) | (none) | (none) | (independent) |
| #12 (Feature Names) | (none) | (none) | (independent) |
| #13 (Metrics) | C2, C11 | D6 | After #1 (needs generalized loss) |
| #14 (Warm-Start) | C8, C11 | (none) | After #1 and #4 (stable artifact format) |
| #15 (Leaf-Wise) | C1, C4 | D1 | After #3 Feature D (coordinate max_leaves) |
| #16 (Histogram Cache) | C4 | (none) | After #4, #7, #15 (settle histogram kernel code) |
