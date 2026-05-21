# AlloyGBM - Claude Code Project Guide

## What This Is

AlloyGBM is a Rust-first Gradient Boosted Decision Tree (GBDT) library with Python bindings via PyO3. It supports regression, binary/multi-class classification, and learning-to-rank. Published on PyPI as `alloygbm`.

## Project Structure

```
AlloyGBM/
  Cargo.toml              # Workspace root (6 crates, edition 2024, Rust 1.92.0)
  crates/
    core/src/lib.rs        # Data structures: TrainParams, BinnedMatrix, ModelMetadata, artifact serde, NaN handling, FeatureBaseline section
    engine/src/lib.rs      # Training loop, ObjectiveOps trait (8 objectives), Trainer, IterationControls, IterationDiagnostics, interaction constraints, WarmStartState (with optional DART tree_weights snapshot)
    engine/src/dart.rs     # DART dropout + normalize helpers (v0.9.0)
    engine/src/shared_histogram.rs  # K-output MultiOutputHistogram primitive (v0.10.0)
    engine/src/joint.rs    # Joint multi-output trainer + JointPredictor (v0.10.0)
    backend_cpu/src/lib.rs # Histogram kernels, split finding, NaN-aware partitioning (Rayon parallelism)
    predictor/src/lib.rs   # Prediction from trained artifacts (post-transforms: identity, sigmoid)
    shap/src/lib.rs        # TreeSHAP (polynomial-time) + legacy brute-force Shapley; PL-leaf interventional decomposition
    categorical/src/lib.rs # Target encoding, frequency encoding (multi-column support)
  bindings/python/
    src/lib.rs             # PyO3 bridge: training pyfunctions for all objectives, NativePredictorHandle
    alloygbm/
      __init__.py             # Public API: GBMRegressor, GBMClassifier, GBMRanker, MultiLabelGBMRanker, metrics, validation
      regressor.py            # GBMRegressor (sklearn-compatible, ~3400 lines)
      classifier.py           # GBMClassifier (binary cross-entropy, predict_proba, ClassifierMixin)
      ranker.py               # GBMRanker (5 ranking objectives, group-sorted training)
      multi_label_ranker.py   # MultiLabelGBMRanker (multi_label_mode="independent": K per-label GBMRankers; "joint" v0.10.1+: shared trees via fit_joint_multi_output)
      evaluation.py           # Metrics: rmse, mae, r2_score, accuracy, log_loss, ndcg, etc.
      validation.py           # Purged time-series and panel cross-validation splits
  docs/
    limitations.md         # Current limitation analysis (v0.10.2, with v0.10.3 / v0.10.4 follow-ups)
    roadmap/current.md     # Active roadmap and per-release history
    user/                  # User-facing Markdown docs (mirrored by docs/site/source/*.rst)
    site/                  # Sphinx site (Read the Docs)
  benchmarks/              # Cross-library comparison (regression, classification, ranking)
```

## Build & Test Commands

```bash
# Rust
cargo check --workspace
cargo test --workspace
cargo clippy --workspace

# Python (requires maturin + virtual env)
maturin develop --release      # Build and install Python extension
.venv/bin/python -m pytest bindings/python/tests/ -q   # Run Python tests

# Quick smoke tests
.venv/bin/python -c "from alloygbm import GBMRegressor; m = GBMRegressor(n_estimators=3); m.fit([[1],[2],[3]], [1,2,3]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMClassifier; m = GBMClassifier(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,0,1,1]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import GBMRanker; m = GBMRanker(n_estimators=3); m.fit([[1],[2],[3],[4]], [0,1,0,1], group=[0,0,1,1]); print(m.predict([[2]]))"
.venv/bin/python -c "from alloygbm import MultiLabelGBMRanker; import numpy as np; m = MultiLabelGBMRanker(n_estimators=3); m.fit([[1],[2],[3],[4]], np.array([[0,1],[1,0],[0,1],[1,0]]), group=[0,0,1,1]); print(m.predict([[2]]))"
```

## Critical Conventions

- **`unsafe_code = "forbid"`** -- no unsafe Rust anywhere in the workspace
- **Edition 2024** with Rust 1.92.0 minimum
- **Newton-Raphson leaf values**: `leaf = -lr * grad_sum / (hess_sum + lambda + eps)` -- general-purpose for any objective
- **Hand-rolled JSON serde** for `ModelMetadata` in `core/src/lib.rs` -- positional parser, very brittle. Adding fields requires careful ordering.
- **`BinnedMatrix`** uses adaptive `Vec<u8>` or `Vec<u16>` -- up to 65,535 bins, column-major duplicate for cache-friendly histograms
- **Artifact format**: Binary with magic bytes `AGBM`, versioned sections (Trees, PredictorLayout, CategoricalState, NativeCategoricalSplits, LinearLeafCoefficients, FeatureBaseline, DartTreeWeights, MultiOutputLeafValues), JSON metadata header. Includes objective type for post-transform dispatch.

## Key Architectural Patterns

- **ObjectiveOps trait** (`engine/src/lib.rs`): Generic trait with `initial_prediction`, `compute_gradients`, `compute_gradients_into`. Implementations: SquaredError, BinaryCrossEntropy, MulticlassSoftmax, RankPairwise, RankNdcg, RankXendcg, QueryRmse, YetiRank.
- **BackendOps trait** (`engine/src/lib.rs`): Abstraction over hardware. Only `CpuBackend` exists.
- **Training policy**: Auto mode with dataset-aware heuristics for `min_split_gain`, `min_rows_per_leaf`, regularization. Manual mode uses raw user params.
- **Tree growth**: Level-wise (default) or leaf-wise (best-first) via `tree_growth` parameter.
- **Histogram subtraction trick**: Used for child nodes within a level (smaller child built from scratch, larger = parent - smaller). Histogram buffers are reused across rounds.
- **NaN handling**: Missing values get a dedicated bin; split finding learns the optimal direction for NaN.
- **Model persistence**: Pickle support via `__getstate__`/`__setstate__`, `save_model`/`load_model`, and raw `artifact_bytes` property.
- **Native categorical splits**: Fisher-sort algorithm finds optimal binary partition in O(K log K); compact bitset encoding for O(1) prediction. Controlled by `max_cat_threshold` (default 0 = disabled). Category-to-ID mappings stored in Python model state.
- **Custom objectives/metrics**: User-provided callable for gradient/hessian (`objective=`) and evaluation metric (`eval_metric=`) with fast numpy I/O.
- **K-output shared histograms** (`crates/engine/src/shared_histogram.rs`, v0.10.0): `MultiOutputHistogram` accumulates K (grad, hess) pairs per (feature, bin) in one sweep. Layout: `feature-major → bin-major → output-major → (grad, hess) interleaved`, accessed via `idx(feature, bin, output, HistComponent::Grad|Hess)`. `subtract_multi_output_histogram` implements the parent-minus-left subtraction trick across all K slots. `compute_multi_output_split_gain` sums per-output Newton/XGBoost gain `Σₖ (G_L_k² / (H_L_k + λ) + G_R_k² / (H_R_k + λ) − G_k² / (H_k + λ))`. Foundation primitive consumed by the joint multi-output trainer (`crates/engine/src/joint.rs`); future consumers will include multiclass DART/GOSS.
- **Joint multi-output trainer** (`crates/engine/src/joint.rs`, v0.10.0): `fit_joint_multi_output` runs a level-wise training loop with K per-output objectives (`JointObjective` enum: `squared_error`, `queryrmse`, `rank:pairwise`, `rank:ndcg`, `rank:xendcg`). Splits are chosen using `compute_multi_output_split_gain` over K outputs; leaves store K Newton-Raphson values via `TrainedStump.multi_output_leaf_values: Option<(Vec<f32>, Vec<f32>)>` (left K-vector + right K-vector). Persists via the `MultiOutputLeafValues` artifact section (kind=13). `JointPredictor` decodes the artifact and predicts K outputs per row. Scope intentionally minimal for v0.10.0: level-wise growth only, no MorphBoost/DRO/neutralization/leaf-wise/native-categorical/GOSS/DART/warm-start on the joint path — these land incrementally across v0.10.x.
- **DART + warm_start** (v0.10.0): `WarmStartState.initial_dart_tree_weights: Option<Vec<f32>>` carries the per-stump `tree_weight` snapshot from the prior fit. The engine seeds `dart_state.tree_weights` from this snapshot (one weight per tree, derived from the first stump of each tree) and pre-populates `round_start_offsets` / `dart_round_counts` from the warm-start tree shapes so new-round dropouts correctly subtract/replay prior trees. Historical RNG-driven `dropped_per_round` is intentionally not persisted; new rounds start fresh dropout bookkeeping going forward. The Python bridge extracts saved weights from `init_model` automatically whenever any stump has a non-default weight.
- **`MultiLabelGBMRanker(multi_label_mode="joint")` Python surface** (v0.10.1): new PyO3 entry point `train_joint_multi_label_ranker` + `JointPredictorHandle` py-class in `bindings/python/src/lib.rs` wrap `engine::joint::fit_joint_multi_output` / `engine::joint::JointPredictor` (v0.10.0 infra). Default `multi_label_mode="independent"` preserves the K-per-label `GBMRanker` fallback from v0.7.1. The kwarg is named `multi_label_mode` (not `training_mode`) to avoid colliding with `GBMRanker.training_mode` (MorphBoost selector). Bundle format `.alloy` bumped to v2 with an explicit mode byte; v1 bundles still load as independent. `_fit_joint` enforces a strict `_JOINT_SUPPORTED_KWARGS` allow-list — anything outside it raises `NotImplementedError`. The allow-list grew across v0.10.x as joint-path feature parity landed; see the v0.10.2 "Joint trainer core feature parity" entry below for the current set. `_normalize_group_for_joint` accepts both LightGBM group-sizes and per-row IDs and stable-sorts rows by group before fitting.
- **Multiclass softmax + GOSS** (v0.10.1): `GBMClassifier(boosting_mode="goss")` works for K ≥ 3 classes via a new `select_row_indices_for_round_multiclass` helper in `crates/engine/src/lib.rs`. Per-row score `s_i = Σₖ |g_{i,k}|` (LightGBM convention) drives a shared sampling mask across all K class gradient buffers. The multiclass round loop in `fit_multiclass_iterations_impl` was refactored so the K gradient buffers are pre-computed BEFORE row sampling.
- **Multiclass softmax + DART** (v0.10.1, expanded v0.10.2): `GBMClassifier(boosting_mode="dart")` works for K ≥ 3 classes via per-class `dart_round_start_offsets[k]` + `dart_round_counts[k]` arrays in `fit_multiclass_iterations_impl` (mirroring the single-output path's flat `round_start_offsets` / `dart_round_counts`). Dropout flat index `flat_idx = r * K + class_k` resolves to `class_stumps[class_k][start..start+count]` — the WHOLE class tree slice, not just one stump. Each round's `dart_round_finalize: Option<(new_w, new_dropped_weights)>` defers `dart_state` mutation + per-stump `tree_weight` stamping to the round-accept branch. Validation predictions get the full DART transition (subtract dropped → scale new at new_w → re-add dropped at w_new) for early-stopping correctness. **v0.10.2**: the `tree_growth="level"` restriction is lifted; leaf-wise multiclass DART works because the per-class `dart_round_*` bookkeeping snapshots `class_stumps[k].len()` around each `build_tree_*` call, which is growth-mode-agnostic — under leaf-wise each tree contributes a variable stump count (capped by `max_leaves`) but the round boundaries are still captured correctly. Warm-start via `MultiClassWarmStartState.initial_dart_tree_weights` (flat round-major × class-k): the PyO3 bridge reconstructs per-tree weights by grouping `class_stumps[k]` by `tree_id = node_id / TREE_NODE_STRIDE` and taking the first stump's `tree_weight` per tree (mirrors `apply_dart_tree_weights` in `crates/predictor/src/lib.rs`).
- **Joint trainer core feature parity** (v0.10.2): the joint multi-output trainer (`crates/engine/src/joint.rs`) gained six features matching the single-output path:
  1. `min_split_gain` — reject splits whose K-output sum-of-gains falls below the threshold.
  2. `row_subsample` — seeded Bernoulli row mask per round via xorshift64*; masked rows get zeroed gradients (equivalent to histogram exclusion). LightGBM `bagging_fraction` semantics.
  3. `col_subsample` — seeded per-round feature mask; if RNG masks every feature, falls back to all-allowed (LightGBM `feature_fraction` behavior).
  4. `interaction_constraints` — reuses `InteractionConstraintIndex` from the single-output trainer via `pub(crate)` visibility. `HashMap<u32, u64>` tracks per-node active group bitset; `descend` narrows it on each split.
  5. `tree_growth="leaf"` + `max_leaves` — new `build_joint_round_leafwise` builds a `BinaryHeap<JointLeafCandidate>` keyed by gain. Each pop commits one stump and evaluates its two children's best splits, pushing them onto the heap. Stops when heap empty, leaf_count ≥ max_leaves, or candidate depth ≥ max_depth. Honors all the constraints above.
  6. Native-categorical splits — new `find_best_multi_output_categorical_split` in `shared_histogram.rs` runs Fisher-sort over K outputs (sorts categories by output-0 score, prefix-scans the sum of XGBoost gains across K outputs, returns the best partition as a `u64` left-bitset). New `fit_joint_multi_output_with_categorical` entry point accepts `&[CategoricalFeatureInfo]`. `JointPredictorStump` gains `is_categorical: bool` + `cat_bitset: u64`; `predict_row` branches on `is_categorical` to route by bitset bit (raw value as category ID) instead of threshold compare. `u64_to_bitset_bytes` / `bitset_bytes_to_u64` helpers convert between the joint trainer's compact u64 form and the single-output `Vec<u8>` byte-per-bit-K-of-byte-K/8 convention. Python kwargs `categorical_feature_indices` + `max_cat_threshold` are derived from the binned column (max non-missing bin + 1 = num_categories); the helper silently falls back to numeric for features outside [2, max_cat_threshold] or > 64 categories (LightGBM semantics).
  
  Still deferred to v0.10.3 (joint GOSS / DART / warm-start) and v0.10.4 (joint MorphBoost / DRO / neutralization). The `_JOINT_SUPPORTED_KWARGS` allow-list in `multi_label_ranker.py` is the source of truth for what's permitted in joint mode at any given release.
- **Piecewise-linear (PL) leaves** (`leaf_model="linear"`): `LeafValue` enum (`Scalar(f32)` | `Linear(LinearLeaf)`) replaces the plain `f32` leaf fields on `TrainedStump`. `LinearLeaf { intercept, weights, regressor_features }` stores closed-form ridge weights `α* = -(XᵀHX + λI)⁻¹ Xᵀg`. A parallel `LinearHistogramBundle` (module `crates/backend_cpu/src/pl_histogram.rs`) accumulates `xtg` and `xtHx` matrix statistics alongside the standard grad/hess bins; the standard SIMD path is untouched. Split gain for PL candidates is computed in `crates/backend_cpu/src/pl.rs` via an 8×8 Cholesky solve. The `GainStrategy::Linear(&LinearContext)` dispatch variant mirrors the MorphBoost precedent. Coefficients are persisted in a new `ModelSectionKind::LinearLeafCoefficients` artifact section; the predictor branches on a per-stump flag bit when evaluating leaves. Native-bitset categorical splits fall back to constant leaves; descendant numeric leaves use linear models normally.

## When Implementing Changes

1. **Run `cargo test --workspace` and `.venv/bin/python -m pytest bindings/python/tests/ -q` before and after** -- the existing test suite must not regress
2. **Commit granularly** -- one commit per logical change, not one giant commit
3. **When adding fields to structs** (TrainParams, IterationControls, etc.) -- add at the end, add a default, add validation
4. **When adding Python parameters** -- update `__init__`, `get_params()`, `set_params()`, `__repr__`, and `_params_order` together
5. **When adding a new objective** -- implement `ObjectiveOps`, add a variant to the objective dispatch in `engine`, update the predictor post-transform table, and add Python-side estimator support

## Cutting A Release

Follow [`docs/reference/release_checklist.md`](docs/reference/release_checklist.md) top-to-bottom. It's the authoritative inventory of every file that needs a version bump or content update (3 version-pin files + CHANGELOG + 14+ doc files), the stale-content `git grep` queries, the local + CI verification matrix, the tag/publish commands, and post-release bookkeeping. Skipping it is what made v0.7.1 docs drift; don't.
