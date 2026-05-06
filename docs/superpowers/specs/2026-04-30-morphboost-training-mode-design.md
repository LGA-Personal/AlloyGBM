# MorphBoost-Inspired Training Mode for AlloyGBM

**Date:** 2026-04-30
**Status:** Design approved — pending implementation plan
**Source paper:** Kriuk, B. (2025). *MorphBoost: Self-Organizing Universal Gradient Boosting with Adaptive Tree Morphing.* arXiv:2511.13234.
**Reference implementation:** `/Users/lashby/Projects/_gbdt_refs/morphboost/morphboost.py`

## 1. Motivation

AlloyGBM is competitive with LightGBM, XGBoost, and CatBoost but does not yet exceed them on accuracy metrics across broad tabular benchmarks. The MorphBoost paper proposes an adaptive training procedure whose split criterion evolves during training — early rounds use pure gradient-based scoring, later rounds blend in normalized information-theoretic terms — combined with leaf-value depth penalties, iteration shrinkage, and an adaptive learning-rate schedule.

The reference Python implementation is unoptimized and contains a notable bug: `_morph_split_function` is defined but never invoked from `_find_best_split_morphing`, so the headline morphing criterion is not actually applied to splits in the published benchmark code. Our implementation will faithfully realize the equations from the paper *and* port the additional mechanisms the reference code does ship (depth penalties, balance penalty, adaptive LR schedule, problem fingerprinting).

The intended outcome is a togglable training profile that, on AlloyGBM's existing benchmark suite, can be promoted to default if it meets the criteria in §6.

## 2. Goals & non-goals

### Goals

- Add a new training profile `training_mode="morph"` available on `GBMRegressor`, `GBMClassifier`, and `GBMRanker`.
- Implement the morphing split criterion, leaf-value modifications, and adaptive learning-rate schedule in optimized Rust, called via the existing PyO3 bridge.
- Preserve backward compatibility: existing `"auto"` and `"manual"` modes are unchanged. Non-morph code paths execute identical instructions to today (no overhead).
- Ensure morph-trained models round-trip through artifact save/load with byte-identical predictions.
- Provide an ablation harness so each morph component can be evaluated independently.

### Non-goals (v1)

- "Quantum splits" and "neural embeddings" from the reference code (unused / toy).
- Vectorized BFS prediction (AlloyGBM's Rust predictor already exceeds the reference's optimization target).
- Leaf-value clipping at ±10 (already covered by `min_leaf_magnitude`).
- Promoting `morph` to default (deferred until benchmark phase shows clear wins).

## 3. Scope: what we port

| Mechanism | Source | Implementation surface |
|---|---|---|
| Morphing split criterion (paper eqs. 2–4) | Paper | `crates/backend_cpu` |
| Leaf-value depth penalty `0.9^(d/3)` | Reference code | `crates/engine` |
| Iteration shrinkage `1 - morph_rate · t/T` | Reference code | `crates/engine` |
| Split-gain balance penalty (unbalanced splits) | Reference code | `crates/backend_cpu` |
| Warm-up + cosine LR schedule | Reference code | `crates/engine` |
| Problem fingerprinting → depth/regularization defaults | Reference code | Python (extends auto-policy) |
| Interaction-aware feature importance bonus | Reference code | Python (post-training) |

## 4. Architecture

### 4.1 Rust types (`crates/core/src/lib.rs`)

```rust
pub enum TrainingMode {
    Auto,
    Manual,
    Morph,
}

pub struct MorphConfig {
    pub morph_rate: f32,            // default 0.1
    pub evolution_pressure: f32,    // default 0.2 (ρ in info-score smoothing)
    pub morph_warmup_iters: u32,    // default 5
    pub info_score_weight: f32,     // default 0.3 (blend weight on info component)
    pub depth_penalty_base: f32,    // default 0.9
    pub balance_penalty: bool,      // default true
    pub lr_schedule: LrSchedule,
}

pub enum LrSchedule {
    Constant,
    WarmupCosine { warmup_frac: f32 }, // default warmup_frac = 0.1
}

pub struct GradientEmaStats {
    pub mean: f32,
    pub std: f32,
    pub alpha: f32, // 0.05 per the paper
}
```

### 4.2 Backend trait extension (`crates/engine/src/lib.rs`)

A new method is added to `BackendOps` rather than extending `SplitSelectionOptions`. This keeps non-morph paths fully untouched and avoids carrying a possibly-empty `MorphContext` through the existing hot path.

```rust
pub struct MorphContext {
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub config: MorphConfig,
}

pub trait BackendOps {
    // ... existing methods unchanged ...

    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>>;
}
```

The default impl delegates to `best_split_with_options` when iteration < warmup, allowing implementations to share warm-up code.

### 4.3 Morphing gain computation (`crates/backend_cpu/src/lib.rs`)

For a candidate split with left/right histogram bins yielding `(g_L, h_L, n_L)` and `(g_R, h_R, n_R)`:

```text
gradient_score = (g_L² / (h_L + λ)) + (g_R² / (h_R + λ))
               - ((g_L+g_R)² / (h_L+h_R + λ))

if iteration < morph_warmup_iters:
    gain = gradient_score
else:
    g_mean_L = g_L / max(n_L, 1)
    g_mean_R = g_R / max(n_R, 1)
    g_norm_L = (g_mean_L - grad_mean) / (grad_std + ε)
    g_norm_R = (g_mean_R - grad_mean) / (grad_std + ε)
    smoothing = 1.0 + ρ · (t / T)
    info_L = |g_norm_L| · log1p(|g_mean_L|) / smoothing
    info_R = |g_norm_R| · log1p(|g_mean_R|) / smoothing
    info_parent = same formula on (g_L+g_R, n_L+n_R)
    info_score = info_L + info_R - info_parent
    morph_weight = tanh(t / 20)
    gain = (1 - info_score_weight) · gradient_score
         + info_score_weight · info_score · morph_weight

if balance_penalty and min(n_L,n_R)/(n_L+n_R) < 0.1:
    balance_ratio = min(n_L,n_R)/(n_L+n_R)
    gain -= 0.5 · (1 - exp(-10 · balance_ratio))
```

**Histogram-level adaptation:** AlloyGBM operates on histogram bins, not raw samples. The info score is computed on bin-aggregated statistics (`g_bin / count_bin`) rather than per-sample. This is faithful to the paper's intent — the bin-level gradient mean is exactly what split finding sees.

### 4.4 Engine changes (`crates/engine/src/lib.rs`)

- `Trainer` gains optional `morph_state: Option<MorphState>`:
  - Per-class `Vec<GradientEmaStats>` (length 1 for single-output, K for multiclass softmax)
  - Resolved `LrSchedule` precomputed per-iteration into `Vec<f32>`
- After each round's gradient computation, EMA stats update with α = 0.05.
- Leaf-value computation gains a multiplicative post-pass:
  ```rust
  if let Some(morph) = &morph_state {
      let depth_penalty = morph.config.depth_penalty_base.powf(depth as f32 / 3.0);
      let iter_shrinkage = 1.0
          - morph.config.morph_rate * (t as f32 / total as f32).min(1.0);
      leaf_value *= depth_penalty * iter_shrinkage;
  }
  ```
- Per-iteration learning rate sourced from the resolved schedule.

### 4.5 Artifact format

A new optional section `MorphMetadataPayload` is added to the binary artifact format:

```rust
pub struct MorphMetadataPayload {
    pub config: MorphConfig,
    pub final_iteration: u32,
    pub final_total: u32,
}
```

- Section is **optional**: artifacts trained in non-morph mode omit it; loaders without the section default to non-morph behavior.
- Predictions are deterministic from artifact alone (leaf values are baked in already), so this section is metadata only — preserved for introspection and reproducibility, not required for prediction.

### 4.6 PyO3 bridge (`bindings/python/src/lib.rs`)

`MorphConfig` is added to the existing `TrainParams` struct as `Option<MorphConfig>`. The existing `train_*_artifact*` entry points pick it up automatically — no new entry-point fanout. PyO3 conversion: morph parameters are read from a Python dict on the params side.

### 4.7 Python API (`bindings/python/alloygbm/`)

New parameters on `GBMRegressor`, `GBMClassifier`, `GBMRanker`:

| Parameter | Type | Default | Active when |
|---|---|---|---|
| `training_mode` | `str` | `"auto"` | always |
| `morph_rate` | `float` | `0.1` | `training_mode="morph"` |
| `evolution_pressure` | `float` | `0.2` | `training_mode="morph"` |
| `morph_warmup_iters` | `int` | `5` | `training_mode="morph"` |
| `lr_schedule` | `str` | `"constant"` | always |
| `lr_warmup_frac` | `float` | `0.1` | `lr_schedule="warmup_cosine"` |

Per CLAUDE.md, each parameter is wired into `__init__`, `get_params`, `set_params`, `__repr__`, and `_params_order`.

**Problem fingerprinting** (`_compute_morph_fingerprint(X, y)`) is a morph-mode-specific routine that lives alongside (not replacing) the existing auto-policy. `training_mode` is mutually exclusive — `auto` uses today's auto-policy heuristics, `morph` uses morph fingerprinting, `manual` uses raw user params. Fingerprinting runs once at fit time when `training_mode="morph"` and only overrides parameters not explicitly set by the user. Fast path uses the paper's fixed constants (`complexity=0.2`, `interaction=0.15`, `noise=0.1`); full path computes variance ratios + linear-vs-quadratic correlation on ≤5 features.

**Interaction-aware feature importance** is computed post-training in Python by traversing the artifact's tree structures: shallow-depth nodes (depth < 3) get a 0.3-weighted bonus on multiplicatively correlated features, identified via paper's correlation heuristic on training data.

## 5. Backward compatibility

- All non-morph paths execute identical instructions to today. The new `best_split_morph` is only called when `training_mode="morph"`.
- All existing `cargo test --workspace` and `pytest bindings/python/tests/` tests pass unchanged.
- Loading a pre-morph artifact yields identical predictions.
- Saving + loading a morph artifact yields byte-identical predictions.
- Pickle support for morph-trained models works via the existing `__getstate__`/`__setstate__` path (which serializes the artifact bytes).

## 6. Validation plan

### 6.1 Correctness tests (during implementation)

- **Numerical equivalence at warmup:** `best_split_morph` with `iteration < morph_warmup_iters` produces identical splits to `best_split_with_options` (bit-equal gain values).
- **Determinism:** same seed + same data + same params → byte-identical artifacts across runs.
- **Round-trip:** morph artifact → save → load → predict produces identical outputs.
- **Backward compat:** full existing test suite passes.

### 6.2 Benchmark sweep

Run `benchmarks/run_model_comparison.py` over all 17 scenarios in three configurations:
1. AlloyGBM `training_mode="auto"` (current default)
2. AlloyGBM `training_mode="morph", lr_schedule="constant"` (isolates split criterion + leaf modifications)
3. AlloyGBM `training_mode="morph", lr_schedule="warmup_cosine"` (full stack)

Plus existing LightGBM/XGBoost/CatBoost baselines. Metrics: RMSE/MAE/R² (regression), accuracy/log-loss (classification), NDCG (ranking), training time, peak memory.

### 6.3 Promote-to-default criteria

`morph` becomes the default when **all** of:

- Wins or ties on ≥ 60% of regression scenarios (RMSE)
- Wins or ties on ≥ 60% of classification scenarios (log-loss)
- NDCG within 1% of `auto` on ranking scenarios
- Training-time overhead ≤ 15% vs `auto` mode
- All existing tests still pass

Otherwise ship as opt-in and iterate.

### 6.4 Ablation harness

`benchmarks/morph_ablation.py` toggles each morph component independently (split criterion, leaf modifications, LR schedule) on 3 representative datasets (1 regression, 1 classification, 1 ranking). Outputs a markdown table identifying which components contribute. Used both for implementation validation and user-facing documentation.

## 7. Open questions / risks

- **Histogram-level info score vs per-sample:** the adaptation is faithful in spirit but not bit-equal to the paper. Empirical evidence from §6.2 will confirm the equivalence is benign.
- **Per-class EMA for multiclass:** doubles the EMA-stat memory footprint to `O(K)`. Negligible for practical K but called out for completeness.
- **Native categorical splits + morphing:** the morphing gain formula composes cleanly with Fisher-sort (which maximizes whatever scalar gain is supplied). No expected interaction issues, but explicitly tested in §6.1.
- **Learning rate schedule + warm-start:** if a user warm-starts a morph-trained model, the schedule needs to resume mid-curve. v1 will document this as resuming the schedule from `iteration = trees_added_so_far`; if this proves unintuitive in practice we'll revisit.

## 8. Out of scope (explicitly deferred)

- Promoting morph to default (gated on §6.3 results)
- GPU implementation of morph gain (CPU only in v1)
- Morphing for custom user-supplied objectives (v1 supports it transparently if the custom objective produces standard gradient/hessian arrays — the morph machinery is objective-agnostic)
- Auto-tuning of `morph_rate` and `evolution_pressure` from training dynamics
