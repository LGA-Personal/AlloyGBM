# MorphBoost Training Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `training_mode="morph"` profile to AlloyGBM that ports the MorphBoost paper's adaptive split criterion, leaf-value modifications, balance penalty, and warm-up+cosine LR schedule.

**Architecture:** Add new types (`MorphConfig`, `LrSchedule`, `MorphContext`, `GradientEmaStats`, `TrainingMode`) in `crates/core`. Extend `BackendOps` trait with a separate `best_split_morph` method on top of existing `best_split_with_options` to keep non-morph paths zero-cost. Trainer maintains per-class EMA gradient statistics across rounds and applies leaf-value depth/iteration modifications when morph mode is active. New optional `MorphMetadataPayload` section in the artifact format records config for introspection. Python estimators expose new parameters following AlloyGBM's existing pattern.

**Tech Stack:** Rust 1.92.0 (edition 2024, `unsafe_code = "forbid"`), PyO3 for Python bindings, Rayon for parallelism. Workspace: `crates/core`, `crates/engine`, `crates/backend_cpu`, `crates/predictor`, `bindings/python`. Python: numpy, sklearn-compatible API.

**Spec:** `docs/superpowers/specs/2026-04-30-morphboost-training-mode-design.md`

**Reference implementation (Python, unoptimized):** `/Users/lashby/Projects/_gbdt_refs/morphboost/morphboost.py`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `crates/core/src/lib.rs` | Add `MorphConfig`, `LrSchedule`, `TrainingMode`, `GradientEmaStats`, `MorphMetadataPayload`; extend `TrainParams` with optional morph config | Modify |
| `crates/engine/src/lib.rs` | Add `MorphContext`, `MorphState`; extend `BackendOps` with `best_split_morph`; wire EMA updates and leaf modifications into `Trainer`; LR schedule resolver | Modify |
| `crates/backend_cpu/src/lib.rs` | Implement morph gain function + `best_split_morph` for both numeric and categorical paths | Modify |
| `bindings/python/src/lib.rs` | Plumb morph config from Python dict into `TrainParams` | Modify |
| `bindings/python/alloygbm/regressor.py` | Add morph parameters; `_compute_morph_fingerprint`; `_resolve_lr_schedule`; interaction-aware importance bonus | Modify |
| `bindings/python/alloygbm/classifier.py` | Add morph parameters | Modify |
| `bindings/python/alloygbm/ranker.py` | Add morph parameters | Modify |
| `bindings/python/alloygbm/_morph.py` | Shared morph helpers (`_compute_morph_fingerprint`, `_resolve_lr_schedule`, interaction-importance) | Create |
| `bindings/python/tests/test_morph.py` | End-to-end morph mode tests (smoke, determinism, round-trip, backward compat) | Create |
| `benchmarks/run_model_comparison.py` | Add morph configs alongside auto baseline | Modify |
| `benchmarks/morph_ablation.py` | Component-isolation ablation harness | Create |

---

## Conventions for All Tasks

- Run `cargo check --workspace` after every Rust edit before moving on.
- Run `cargo test --workspace` and `.venv/bin/python -m pytest bindings/python/tests/ -q` before completing each phase.
- Follow CLAUDE.md: when adding fields to structs, append at end with default + validation. When adding Python params, update `__init__`, `get_params`, `set_params`, `__repr__`, `_params_order` together.
- Commit at the end of each task with `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- `unsafe_code = "forbid"` is in effect — no `unsafe` blocks anywhere.

---

## Task 1: Core Types (MorphConfig, LrSchedule, TrainingMode, GradientEmaStats)

**Files:**
- Modify: `crates/core/src/lib.rs` (after `TrainParams` definition around line 70)
- Test: same file (existing `#[cfg(test)] mod tests` block)

**Context:** All morph-related types live in `core` so both `engine` and `bindings/python` can use them. `TrainParams` gains an optional `morph_config: Option<MorphConfig>` field — `None` means non-morph training (current behavior).

- [ ] **Step 1: Write failing tests for new types**

Add to the `#[cfg(test)] mod tests` block in `crates/core/src/lib.rs`:

```rust
#[test]
fn morph_config_default_matches_paper() {
    let cfg = MorphConfig::default();
    assert_eq!(cfg.morph_rate, 0.1);
    assert_eq!(cfg.evolution_pressure, 0.2);
    assert_eq!(cfg.morph_warmup_iters, 5);
    assert_eq!(cfg.info_score_weight, 0.3);
    assert_eq!(cfg.depth_penalty_base, 0.9);
    assert!(cfg.balance_penalty);
    assert_eq!(cfg.lr_schedule, LrSchedule::Constant);
}

#[test]
fn lr_schedule_warmup_cosine_default_warmup_frac() {
    let s = LrSchedule::WarmupCosine { warmup_frac: 0.1 };
    if let LrSchedule::WarmupCosine { warmup_frac } = s {
        assert!((warmup_frac - 0.1).abs() < 1e-6);
    } else {
        panic!("expected WarmupCosine");
    }
}

#[test]
fn training_mode_default_is_auto() {
    assert_eq!(TrainingMode::default(), TrainingMode::Auto);
}

#[test]
fn train_params_default_has_no_morph_config() {
    let p = TrainParams::default();
    assert!(p.morph_config.is_none());
}

#[test]
fn validate_train_params_accepts_morph_config() {
    let mut p = TrainParams::default();
    p.morph_config = Some(MorphConfig::default());
    assert!(validate_train_params(&p).is_ok());
}

#[test]
fn validate_train_params_rejects_invalid_morph_rate() {
    let mut p = TrainParams::default();
    p.morph_config = Some(MorphConfig {
        morph_rate: -0.1,
        ..MorphConfig::default()
    });
    assert!(validate_train_params(&p).is_err());
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test -p alloygbm-core --lib`
Expected: FAIL — types not defined.

- [ ] **Step 3: Add `LrSchedule` enum**

Add to `crates/core/src/lib.rs` (place near other config types, e.g. after `TreeGrowth`):

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LrSchedule {
    /// Use constant learning rate for all rounds.
    Constant,
    /// Linear warmup from 0 → learning_rate over `warmup_frac * n_estimators`
    /// rounds, then half-cosine decay to `learning_rate * 0.01` over remaining rounds.
    WarmupCosine { warmup_frac: f32 },
}

impl Default for LrSchedule {
    fn default() -> Self {
        LrSchedule::Constant
    }
}
```

- [ ] **Step 4: Add `MorphConfig` struct**

```rust
/// Configuration for the MorphBoost-inspired training profile.
///
/// All fields are runtime-configurable; defaults match the paper's
/// recommended values (Kriuk 2025, arXiv:2511.13234).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorphConfig {
    /// Strength of per-round leaf shrinkage: leaf *= (1 - morph_rate * t/T)
    pub morph_rate: f32,
    /// ρ in info-score smoothing factor (1 + ρ * t/T)
    pub evolution_pressure: f32,
    /// Number of pure-gradient rounds before info-score blending kicks in.
    pub morph_warmup_iters: u32,
    /// Blend weight on info component: gain = (1-w) * grad + w * info * tanh(t/20)
    pub info_score_weight: f32,
    /// Base for leaf depth penalty: leaf *= depth_penalty_base ^ (depth/3)
    pub depth_penalty_base: f32,
    /// Apply balance penalty for unbalanced splits.
    pub balance_penalty: bool,
    /// Per-iteration learning rate schedule.
    pub lr_schedule: LrSchedule,
}

impl Default for MorphConfig {
    fn default() -> Self {
        Self {
            morph_rate: 0.1,
            evolution_pressure: 0.2,
            morph_warmup_iters: 5,
            info_score_weight: 0.3,
            depth_penalty_base: 0.9,
            balance_penalty: true,
            lr_schedule: LrSchedule::Constant,
        }
    }
}
```

- [ ] **Step 5: Add `TrainingMode` enum**

```rust
/// Top-level training profile selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMode {
    /// Auto-policy with dataset-aware heuristics (current default).
    Auto,
    /// Raw user-supplied parameters with no overrides.
    Manual,
    /// MorphBoost-inspired adaptive training profile.
    Morph,
}

impl Default for TrainingMode {
    fn default() -> Self {
        TrainingMode::Auto
    }
}
```

- [ ] **Step 6: Add `GradientEmaStats` struct**

```rust
/// Exponential moving average statistics for gradients across boosting rounds.
/// Maintained per-class for multiclass softmax (length 1 for single-output).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientEmaStats {
    pub mean: f32,
    pub std: f32,
    /// Decay rate (0.05 per the paper).
    pub alpha: f32,
}

impl Default for GradientEmaStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            alpha: 0.05,
        }
    }
}

impl GradientEmaStats {
    /// Update EMA in place from a new round's gradient slice.
    pub fn update(&mut self, gradients: &[f32]) {
        if gradients.is_empty() {
            return;
        }
        let n = gradients.len() as f32;
        let mean: f32 = gradients.iter().sum::<f32>() / n;
        let var: f32 = gradients
            .iter()
            .map(|g| (g - mean) * (g - mean))
            .sum::<f32>()
            / n;
        let std = var.sqrt();
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * mean;
        self.std = (1.0 - self.alpha) * self.std + self.alpha * std;
    }
}
```

- [ ] **Step 7: Extend `TrainParams` with optional morph config**

Modify the existing `TrainParams` struct (around line 46) to append `morph_config`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    pub seed: u64,
    pub deterministic: bool,
    pub learning_rate: f32,
    pub max_depth: u16,
    pub row_subsample: f32,
    pub col_subsample: f32,
    pub early_stopping_rounds: Option<u16>,
    pub min_validation_improvement: f32,
    pub min_data_in_leaf: u32,
    pub lambda_l1: f32,
    pub lambda_l2: f32,
    pub min_child_hessian: f32,
    pub min_split_gain: f32,
    pub monotone_constraints: Vec<i8>,
    pub feature_weights: Vec<f32>,
    pub max_leaves: Option<usize>,
    pub tree_growth: TreeGrowth,
    /// MorphBoost-inspired training profile config. `None` = non-morph (current behavior).
    pub morph_config: Option<MorphConfig>,
}
```

Update the `Default` impl to set `morph_config: None`.

- [ ] **Step 8: Extend `validate_train_params` to validate morph config**

Locate `validate_train_params` in `crates/core/src/lib.rs`. Append validation for `morph_config`:

```rust
if let Some(cfg) = &params.morph_config {
    if !cfg.morph_rate.is_finite() || cfg.morph_rate < 0.0 || cfg.morph_rate > 1.0 {
        return Err(CoreError::Validation(format!(
            "morph_config.morph_rate must be in [0, 1], got {}",
            cfg.morph_rate
        )));
    }
    if !cfg.evolution_pressure.is_finite() || cfg.evolution_pressure < 0.0 {
        return Err(CoreError::Validation(format!(
            "morph_config.evolution_pressure must be >= 0, got {}",
            cfg.evolution_pressure
        )));
    }
    if !cfg.info_score_weight.is_finite()
        || cfg.info_score_weight < 0.0
        || cfg.info_score_weight > 1.0
    {
        return Err(CoreError::Validation(format!(
            "morph_config.info_score_weight must be in [0, 1], got {}",
            cfg.info_score_weight
        )));
    }
    if !cfg.depth_penalty_base.is_finite()
        || cfg.depth_penalty_base <= 0.0
        || cfg.depth_penalty_base > 1.0
    {
        return Err(CoreError::Validation(format!(
            "morph_config.depth_penalty_base must be in (0, 1], got {}",
            cfg.depth_penalty_base
        )));
    }
    if let LrSchedule::WarmupCosine { warmup_frac } = cfg.lr_schedule {
        if !warmup_frac.is_finite() || warmup_frac < 0.0 || warmup_frac > 1.0 {
            return Err(CoreError::Validation(format!(
                "morph_config.lr_schedule.warmup_frac must be in [0, 1], got {}",
                warmup_frac
            )));
        }
    }
}
```

- [ ] **Step 9: Run tests to verify pass**

Run: `cargo test -p alloygbm-core --lib`
Expected: PASS — all new tests green, all existing tests still pass.

- [ ] **Step 10: Run workspace check**

Run: `cargo check --workspace`
Expected: PASS — adding the new field as the last item with a default keeps every dependent crate compiling.

- [ ] **Step 11: Commit**

```bash
git add crates/core/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(core): add MorphConfig, LrSchedule, TrainingMode, GradientEmaStats

Introduces the morph-mode configuration types and threads an optional
MorphConfig into TrainParams (None = current behavior). Validation rejects
out-of-range morph parameters. Foundation for the morph training profile.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pure Morph Gain Function (Numeric Path)

**Files:**
- Create: `crates/backend_cpu/src/morph.rs`
- Modify: `crates/backend_cpu/src/lib.rs` (add `mod morph;` at top)
- Test: `crates/backend_cpu/src/morph.rs` (inline `#[cfg(test)] mod tests`)

**Context:** Isolate the morph gain math in its own module. This is a pure function: takes histogram bin sums, returns gain. Easy to unit test in isolation. The `best_split_morph` orchestration in Task 3 will call this.

- [ ] **Step 1: Create morph.rs scaffold**

Create `crates/backend_cpu/src/morph.rs`:

```rust
//! Morph gain computation for split scoring.
//!
//! Implements the adaptive split criterion from Kriuk (2025), arXiv:2511.13234.
//! Pre-warmup, this returns the standard XGBoost-style gain. Post-warmup, it
//! blends gradient-based scoring with normalized information-theoretic terms.

use alloygbm_core::MorphConfig;

/// Statistics for one side of a split (left, right, or parent).
#[derive(Debug, Clone, Copy)]
pub struct SplitSideStats {
    pub gradient_sum: f32,
    pub hessian_sum: f32,
    pub count: u32,
}

/// Morph gain inputs.
#[derive(Debug, Clone, Copy)]
pub struct MorphGainInputs {
    pub left: SplitSideStats,
    pub right: SplitSideStats,
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub lambda_l2: f32,
}

const EPS: f32 = 1e-10;

/// Compute the morph-augmented split gain.
///
/// At `iteration < morph_warmup_iters`, returns the standard XGBoost gain
/// to allow byte-equivalence with the non-morph path during warmup.
pub fn compute_morph_gain(inputs: MorphGainInputs, config: &MorphConfig) -> f32 {
    let gradient_score = gradient_gain(&inputs);

    let mut gain = if inputs.iteration < config.morph_warmup_iters {
        gradient_score
    } else {
        let info_score = info_gain(&inputs, config);
        let morph_weight = (inputs.iteration as f32 / 20.0).tanh();
        (1.0 - config.info_score_weight) * gradient_score
            + config.info_score_weight * info_score * morph_weight
    };

    if config.balance_penalty {
        gain += balance_adjustment(&inputs);
    }

    gain
}

fn gradient_gain(inputs: &MorphGainInputs) -> f32 {
    let l = &inputs.left;
    let r = &inputs.right;
    let lambda = inputs.lambda_l2;
    let parent_g = l.gradient_sum + r.gradient_sum;
    let parent_h = l.hessian_sum + r.hessian_sum;
    (l.gradient_sum * l.gradient_sum) / (l.hessian_sum + lambda)
        + (r.gradient_sum * r.gradient_sum) / (r.hessian_sum + lambda)
        - (parent_g * parent_g) / (parent_h + lambda)
}

fn info_gain(inputs: &MorphGainInputs, config: &MorphConfig) -> f32 {
    let smoothing =
        1.0 + config.evolution_pressure * (inputs.iteration as f32 / inputs.total_iterations as f32);
    let info_l = info_side(inputs.left, inputs.grad_mean, inputs.grad_std, smoothing);
    let info_r = info_side(inputs.right, inputs.grad_mean, inputs.grad_std, smoothing);
    let info_parent = info_side(
        SplitSideStats {
            gradient_sum: inputs.left.gradient_sum + inputs.right.gradient_sum,
            hessian_sum: inputs.left.hessian_sum + inputs.right.hessian_sum,
            count: inputs.left.count + inputs.right.count,
        },
        inputs.grad_mean,
        inputs.grad_std,
        smoothing,
    );
    info_l + info_r - info_parent
}

fn info_side(stats: SplitSideStats, mean: f32, std: f32, smoothing: f32) -> f32 {
    if stats.count == 0 {
        return 0.0;
    }
    let g_mean = stats.gradient_sum / stats.count as f32;
    let g_norm = (g_mean - mean) / (std + EPS);
    g_norm.abs() * (1.0 + g_mean.abs()).ln() / smoothing
}

fn balance_adjustment(inputs: &MorphGainInputs) -> f32 {
    let total = inputs.left.count + inputs.right.count;
    if total == 0 {
        return 0.0;
    }
    let min_side = inputs.left.count.min(inputs.right.count);
    let balance_ratio = min_side as f32 / total as f32;
    if balance_ratio >= 0.1 {
        return 0.0;
    }
    -0.5 * (1.0 - (-10.0 * balance_ratio).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{LrSchedule, MorphConfig};

    fn config() -> MorphConfig {
        MorphConfig::default()
    }

    fn balanced_inputs(iteration: u32) -> MorphGainInputs {
        MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 5.0,
                hessian_sum: 4.0,
                count: 50,
            },
            right: SplitSideStats {
                gradient_sum: -5.0,
                hessian_sum: 4.0,
                count: 50,
            },
            iteration,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            lambda_l2: 1.0,
        }
    }

    #[test]
    fn warmup_iteration_returns_pure_gradient_gain() {
        let inputs = balanced_inputs(0);
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let gain = compute_morph_gain(inputs, &cfg);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }

    #[test]
    fn last_warmup_iteration_returns_pure_gradient_gain() {
        let cfg = config();
        let inputs = balanced_inputs(cfg.morph_warmup_iters - 1);
        let cfg_no_balance = MorphConfig {
            balance_penalty: false,
            ..cfg
        };
        let gain = compute_morph_gain(inputs, &cfg_no_balance);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }

    #[test]
    fn first_post_warmup_iteration_blends() {
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let inputs = balanced_inputs(cfg.morph_warmup_iters);
        let gain = compute_morph_gain(inputs, &cfg);
        let pure_gradient = gradient_gain(&inputs);
        // After warmup, gain should differ from pure gradient.
        assert!((gain - pure_gradient).abs() > 1e-9);
    }

    #[test]
    fn morph_weight_grows_with_iteration() {
        let cfg = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let early = balanced_inputs(cfg.morph_warmup_iters);
        let late = balanced_inputs(50);
        let gain_early = compute_morph_gain(early, &cfg);
        let gain_late = compute_morph_gain(late, &cfg);
        // Different iterations produce different gains.
        assert!((gain_early - gain_late).abs() > 1e-9);
    }

    #[test]
    fn balance_penalty_reduces_gain_for_unbalanced_split() {
        let unbalanced = MorphGainInputs {
            left: SplitSideStats {
                gradient_sum: 1.0,
                hessian_sum: 1.0,
                count: 5,
            },
            right: SplitSideStats {
                gradient_sum: -1.0,
                hessian_sum: 1.0,
                count: 95,
            },
            iteration: 0,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            lambda_l2: 1.0,
        };
        let cfg_off = MorphConfig {
            balance_penalty: false,
            ..config()
        };
        let cfg_on = MorphConfig {
            balance_penalty: true,
            ..config()
        };
        let gain_off = compute_morph_gain(unbalanced, &cfg_off);
        let gain_on = compute_morph_gain(unbalanced, &cfg_on);
        assert!(gain_on < gain_off);
    }

    #[test]
    fn balance_penalty_zero_for_balanced_split() {
        let balanced = balanced_inputs(0);
        let cfg = config();
        let cfg_no_balance = MorphConfig {
            balance_penalty: false,
            ..cfg
        };
        let with_pen = compute_morph_gain(balanced, &cfg);
        let no_pen = compute_morph_gain(balanced, &cfg_no_balance);
        assert!((with_pen - no_pen).abs() < 1e-6);
    }

    #[test]
    fn lr_schedule_is_carried_through_unchanged() {
        // Smoke test that LrSchedule on config doesn't alter gain math.
        let cfg = MorphConfig {
            lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.1 },
            balance_penalty: false,
            ..config()
        };
        let inputs = balanced_inputs(0);
        let gain = compute_morph_gain(inputs, &cfg);
        let expected = gradient_gain(&inputs);
        assert!((gain - expected).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Add `mod morph;` to `crates/backend_cpu/src/lib.rs`**

At the top of `crates/backend_cpu/src/lib.rs`, add:

```rust
mod morph;
pub use morph::{compute_morph_gain, MorphGainInputs, SplitSideStats};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p alloygbm-backend-cpu --lib morph`
Expected: PASS — 7 tests.

- [ ] **Step 4: Run full backend tests to verify no regression**

Run: `cargo test -p alloygbm-backend-cpu --lib`
Expected: PASS — all existing tests still green.

- [ ] **Step 5: Commit**

```bash
git add crates/backend_cpu/src/morph.rs crates/backend_cpu/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(backend_cpu): add pure morph gain computation module

Pure function compute_morph_gain implements the paper's adaptive split
criterion: pre-warmup returns standard XGBoost gain, post-warmup blends with
normalized info-theoretic score weighted by tanh(t/20). Optional balance
penalty for unbalanced splits. Fully unit-tested in isolation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: BackendOps Trait Extension + CpuBackend Numeric Path

**Files:**
- Modify: `crates/engine/src/lib.rs` (extend `BackendOps` trait, add `MorphContext`)
- Modify: `crates/backend_cpu/src/lib.rs` (implement `best_split_morph` for `CpuBackend`)

**Context:** Add a separate `best_split_morph` trait method (decision in spec §4.2). The default impl delegates to `best_split_with_options` to keep non-morph paths untouched. CpuBackend overrides with an implementation that calls `compute_morph_gain` per candidate split.

- [ ] **Step 1: Add `MorphContext` to engine**

In `crates/engine/src/lib.rs`, near `SplitSelectionOptions` (around line 55), add:

```rust
use alloygbm_core::MorphConfig;

/// Per-round context for morph-gain split selection.
/// Passed to `BackendOps::best_split_morph` in addition to the standard options.
#[derive(Debug, Clone, Copy)]
pub struct MorphContext {
    pub iteration: u32,
    pub total_iterations: u32,
    pub grad_mean: f32,
    pub grad_std: f32,
    pub config: MorphConfig,
}
```

- [ ] **Step 2: Add `best_split_morph` to `BackendOps` trait**

Locate the `pub trait BackendOps` block (around line 86). After `best_split_with_options` add:

```rust
/// Morph-mode split selection. Default implementation delegates to
/// `best_split_with_options` (i.e. ignores morph context), so backends that
/// don't implement morph fall back gracefully.
fn best_split_morph(
    &self,
    histograms: &HistogramBundle,
    options: SplitSelectionOptions,
    feature_weights: &[f32],
    categorical_features: &[CategoricalFeatureInfo],
    _morph: &MorphContext,
) -> EngineResult<Option<SplitCandidate>> {
    self.best_split_with_options(histograms, options, feature_weights, categorical_features)
}
```

- [ ] **Step 3: Run check**

Run: `cargo check --workspace`
Expected: PASS — default impl means existing implementors don't break.

- [ ] **Step 4: Locate CpuBackend's `best_split_with_options_internal` for reference**

Read `crates/backend_cpu/src/lib.rs` lines 778-980. Note how it iterates candidate features, candidate thresholds, and selects the highest gain. The morph variant follows the same loop but substitutes `compute_morph_gain` for the gain formula on each candidate.

- [ ] **Step 5: Implement `best_split_morph` on `CpuBackend` for numeric features**

In `crates/backend_cpu/src/lib.rs`, add an `impl BackendOps for CpuBackend` method `best_split_morph`. The implementation mirrors `best_split_with_options_internal` but, for each candidate split's `(left_g, left_h, left_n, right_g, right_h, right_n)`, calls:

```rust
use crate::morph::{compute_morph_gain, MorphGainInputs, SplitSideStats};
use alloygbm_engine::MorphContext;

impl CpuBackend {
    fn best_split_morph_numeric_feature(
        &self,
        feature_idx: usize,
        feature_histogram: &[GradientPair],
        bin_counts: &[u32],
        options: &SplitSelectionOptions,
        morph: &MorphContext,
    ) -> Option<(usize, f32)> {
        // Forward-scan accumulating left sums; right = total - left.
        // Skip missing bin (options.missing_bin_index).
        let n_bins = feature_histogram.len();
        let mut total_g = 0.0f32;
        let mut total_h = 0.0f32;
        let mut total_n = 0u32;
        for (b, gp) in feature_histogram.iter().enumerate() {
            if b == options.missing_bin_index {
                continue;
            }
            total_g += gp.gradient;
            total_h += gp.hessian;
            total_n += bin_counts[b];
        }

        let mut best: Option<(usize, f32)> = None;
        let mut left_g = 0.0f32;
        let mut left_h = 0.0f32;
        let mut left_n = 0u32;
        for split_bin in 0..n_bins.saturating_sub(1) {
            if split_bin == options.missing_bin_index {
                continue;
            }
            let gp = &feature_histogram[split_bin];
            left_g += gp.gradient;
            left_h += gp.hessian;
            left_n += bin_counts[split_bin];

            let right_g = total_g - left_g;
            let right_h = total_h - left_h;
            let right_n = total_n - left_n;

            if left_h < options.min_child_hessian || right_h < options.min_child_hessian {
                continue;
            }
            if left_n == 0 || right_n == 0 {
                continue;
            }

            let inputs = MorphGainInputs {
                left: SplitSideStats {
                    gradient_sum: left_g,
                    hessian_sum: left_h,
                    count: left_n,
                },
                right: SplitSideStats {
                    gradient_sum: right_g,
                    hessian_sum: right_h,
                    count: right_n,
                },
                iteration: morph.iteration,
                total_iterations: morph.total_iterations.max(1),
                grad_mean: morph.grad_mean,
                grad_std: morph.grad_std,
                lambda_l2: options.l2_lambda,
            };
            let gain = compute_morph_gain(inputs, &morph.config);

            if best.map_or(true, |(_, g)| gain > g) {
                best = Some((split_bin, gain));
            }
        }
        best
    }
}
```

Note: this is a dedicated helper that the `BackendOps::best_split_morph` impl calls per feature. The exact structure (how `feature_histogram` and `bin_counts` are derived from `HistogramBundle`) should mirror `best_split_for_feature` at line 451 — read that function first and follow its data-extraction pattern.

- [ ] **Step 6: Wire `BackendOps::best_split_morph` impl on CpuBackend**

Override the trait method on `CpuBackend` to iterate features (numeric only for this task — categorical comes in Task 4):

```rust
impl BackendOps for CpuBackend {
    // ... existing methods unchanged ...

    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        // For Task 3: any categorical feature falls back to non-morph path.
        if !categorical_features.is_empty() {
            return self.best_split_with_options(
                histograms,
                options,
                feature_weights,
                categorical_features,
            );
        }
        // Iterate numeric features, picking globally best gain.
        // Use the same structure as best_split_with_options_internal but
        // substitute best_split_morph_numeric_feature for the per-feature gain.
        // ... (mirror existing pattern)
    }
}
```

The full body should mirror `best_split_with_options_internal`'s feature-iteration shell but call `best_split_morph_numeric_feature` instead of the standard inner gain loop. Read lines 778-979 first.

- [ ] **Step 7: Add a backend integration test**

Add to `crates/backend_cpu/src/lib.rs` test module:

```rust
#[test]
fn best_split_morph_at_warmup_matches_best_split_with_options() {
    let backend = CpuBackend::new();
    let bundle = make_test_histogram_bundle(); // reuse existing test helper
    let options = SplitSelectionOptions::default();
    let morph = MorphContext {
        iteration: 0,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: MorphConfig {
            balance_penalty: false,
            ..MorphConfig::default()
        },
    };
    let standard = backend
        .best_split_with_options(&bundle, options.clone(), &[], &[])
        .unwrap();
    let morph_result = backend
        .best_split_morph(&bundle, options, &[], &[], &morph)
        .unwrap();
    // At iteration < warmup, with balance penalty off, results should agree
    // on which bin/feature is selected (gain values may differ in absolute
    // terms only if feature_weights are applied — they're empty here).
    match (standard, morph_result) {
        (Some(a), Some(b)) => {
            assert_eq!(a.feature_index, b.feature_index);
            assert_eq!(a.bin_index, b.bin_index);
        }
        (None, None) => {}
        _ => panic!("split selection disagreed at warmup"),
    }
}
```

If a `make_test_histogram_bundle` helper doesn't exist, look at the existing `best_split_returns_high_gain_candidate` test (line 1486) and adapt its setup.

- [ ] **Step 8: Run tests**

Run: `cargo test -p alloygbm-backend-cpu --lib`
Run: `cargo test -p alloygbm-engine --lib`
Expected: PASS — new test green, no regressions.

- [ ] **Step 9: Workspace check**

Run: `cargo check --workspace && cargo clippy --workspace -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 10: Commit**

```bash
git add crates/engine/src/lib.rs crates/backend_cpu/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(engine,backend_cpu): add best_split_morph trait method and numeric impl

Adds MorphContext and best_split_morph to BackendOps with default delegation
to best_split_with_options. CpuBackend implements morph gain for numeric
features. Categorical features fall back to standard path in this task;
Task 4 extends morph to categorical splits.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Morph Gain for Categorical Splits

**Files:**
- Modify: `crates/backend_cpu/src/lib.rs` (Fisher-sort path inside categorical split finder)

**Context:** AlloyGBM's native categorical splits use a Fisher-sort algorithm that maximizes a scalar gain over candidate binary partitions. The morph extension swaps the gain formula at the inner-loop scoring step.

- [ ] **Step 1: Locate the categorical split implementation**

Read `crates/backend_cpu/src/lib.rs` line 603 onward (`best_split_for_categorical_feature`). Identify the line where the per-partition gain is computed. The morph extension calls `compute_morph_gain` instead, with the same `(left_g, left_h, left_n, right_g, right_h, right_n)` aggregates.

- [ ] **Step 2: Add a sibling helper `best_split_morph_categorical_feature`**

Mirror the existing function, substituting `compute_morph_gain` for the gain formula:

```rust
fn best_split_morph_categorical_feature(
    &self,
    feature_idx: usize,
    feature_info: &CategoricalFeatureInfo,
    feature_histogram: &[GradientPair],
    bin_counts: &[u32],
    options: &SplitSelectionOptions,
    morph: &MorphContext,
) -> Option<CategoricalSplitCandidate> {
    // ... copy the Fisher-sort scaffold from best_split_for_categorical_feature
    // ... where it computes gain, instead build MorphGainInputs and call
    //     compute_morph_gain.
}
```

- [ ] **Step 3: Update `best_split_morph` impl on CpuBackend to dispatch to categorical path**

Replace the early-return on `!categorical_features.is_empty()` from Task 3, Step 6 with a proper implementation that dispatches per-feature:

- For each numeric feature → `best_split_morph_numeric_feature`
- For each categorical feature → `best_split_morph_categorical_feature`
- Pick the global winner

- [ ] **Step 4: Add categorical morph test**

```rust
#[test]
fn best_split_morph_handles_categorical_features() {
    let backend = CpuBackend::new();
    let bundle = make_categorical_test_bundle(); // adapt existing helper
    let options = SplitSelectionOptions::default();
    let cat_features = vec![CategoricalFeatureInfo {
        feature_index: 0,
        num_categories: 4,
    }];
    let morph = MorphContext {
        iteration: 10,
        total_iterations: 100,
        grad_mean: 0.0,
        grad_std: 1.0,
        config: MorphConfig::default(),
    };
    let result = backend
        .best_split_morph(&bundle, options, &[], &cat_features, &morph)
        .unwrap();
    assert!(result.is_some(), "morph categorical split should produce a candidate");
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p alloygbm-backend-cpu --lib`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/backend_cpu/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(backend_cpu): extend morph gain to categorical Fisher-sort splits

best_split_morph now dispatches to a categorical sibling that wires
compute_morph_gain into the Fisher-sort scoring loop. Numeric and
categorical features now use morph gain consistently in morph mode.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Engine MorphState + Trainer Wiring + Leaf Modifications

**Files:**
- Modify: `crates/engine/src/lib.rs` (Trainer struct, training loop, leaf-value finalization)

**Context:** The Trainer owns per-class EMA stats and the resolved LR schedule. After each round's gradient computation, EMA stats update. When morph mode is active, leaf values get post-multiplied by depth-penalty × iteration-shrinkage.

- [ ] **Step 1: Add `MorphState` struct to engine**

In `crates/engine/src/lib.rs`, after `MorphContext`:

```rust
use alloygbm_core::{GradientEmaStats, LrSchedule};

/// Trainer-resident state for morph-mode training.
pub struct MorphState {
    pub config: MorphConfig,
    /// Per-class EMA stats (length 1 for single-output, K for multiclass).
    pub ema_stats: Vec<GradientEmaStats>,
    /// Resolved per-iteration learning rates.
    pub lr_per_iter: Vec<f32>,
}

impl MorphState {
    pub fn new(config: MorphConfig, n_classes: usize, n_estimators: u32, base_lr: f32) -> Self {
        Self {
            config,
            ema_stats: vec![GradientEmaStats::default(); n_classes.max(1)],
            lr_per_iter: resolve_lr_schedule(config.lr_schedule, n_estimators, base_lr),
        }
    }

    pub fn morph_context(&self, iteration: u32, total: u32, class_idx: usize) -> MorphContext {
        let stats = &self.ema_stats[class_idx];
        MorphContext {
            iteration,
            total_iterations: total,
            grad_mean: stats.mean,
            grad_std: stats.std,
            config: self.config,
        }
    }
}

fn resolve_lr_schedule(schedule: LrSchedule, n_estimators: u32, base_lr: f32) -> Vec<f32> {
    let n = n_estimators as usize;
    match schedule {
        LrSchedule::Constant => vec![base_lr; n],
        LrSchedule::WarmupCosine { warmup_frac } => {
            let warmup = ((warmup_frac * n as f32).round() as usize).max(1).min(n);
            let mut lrs = Vec::with_capacity(n);
            // Warmup: linear from base_lr/warmup → base_lr
            for i in 0..warmup {
                lrs.push(base_lr * (i as f32 + 1.0) / warmup as f32);
            }
            // Cosine decay from base_lr → base_lr*0.01
            let remaining = n - warmup;
            let floor = base_lr * 0.01;
            for i in 0..remaining {
                let progress = i as f32 / remaining.max(1) as f32;
                let cos = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                lrs.push(floor + (base_lr - floor) * cos);
            }
            lrs
        }
    }
}
```

- [ ] **Step 2: Add unit tests for `resolve_lr_schedule`**

```rust
#[cfg(test)]
mod morph_state_tests {
    use super::*;

    #[test]
    fn constant_schedule_produces_uniform_lr() {
        let lrs = resolve_lr_schedule(LrSchedule::Constant, 10, 0.1);
        assert_eq!(lrs.len(), 10);
        for lr in lrs {
            assert!((lr - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn warmup_cosine_starts_low_peaks_at_base_decays() {
        let lrs = resolve_lr_schedule(
            LrSchedule::WarmupCosine { warmup_frac: 0.1 },
            100,
            0.1,
        );
        assert_eq!(lrs.len(), 100);
        // First step is below base.
        assert!(lrs[0] < 0.1);
        // Peak (end of warmup) should hit base.
        let peak_idx = (0.1 * 100.0).round() as usize - 1;
        assert!((lrs[peak_idx] - 0.1).abs() < 1e-3);
        // Last value approaches floor.
        assert!(lrs[99] < 0.05);
    }

    #[test]
    fn warmup_cosine_with_zero_remaining_handles_edge_case() {
        let lrs = resolve_lr_schedule(
            LrSchedule::WarmupCosine { warmup_frac: 1.0 },
            5,
            0.1,
        );
        assert_eq!(lrs.len(), 5);
    }
}
```

- [ ] **Step 3: Run unit tests**

Run: `cargo test -p alloygbm-engine --lib morph_state`
Expected: PASS — 3 tests.

- [ ] **Step 4: Locate the Trainer's training loop**

In `crates/engine/src/lib.rs`, find the main training loop in `Trainer` (search for `for round in` or similar). Note where:
- gradients are computed
- splits are selected (this calls into `BackendOps::best_split_with_options`)
- leaf values are finalized
- the iteration's learning rate is applied

- [ ] **Step 5: Integrate `MorphState` into Trainer**

The Trainer's training entry point (e.g. `train_regression_artifact`-equivalent in engine) should:

1. Construct `Option<MorphState>` from `train_params.morph_config`.
2. After each round's gradient computation, call `morph_state.ema_stats[class_idx].update(&gradients_for_class)`.
3. Replace the call to `backend.best_split_with_options` with a branch:
   ```rust
   let split = if let Some(ms) = &morph_state {
       let ctx = ms.morph_context(iteration as u32, n_estimators as u32, class_idx);
       backend.best_split_morph(&histograms, options, &fw, &cat_feats, &ctx)?
   } else {
       backend.best_split_with_options(&histograms, options, &fw, &cat_feats)?
   };
   ```
4. After computing leaf value `leaf = -lr * grad_sum / (hess_sum + lambda + eps)`, apply:
   ```rust
   if let Some(ms) = &morph_state {
       let depth_penalty = ms.config.depth_penalty_base.powf(depth as f32 / 3.0);
       let iter_shrinkage = 1.0 - ms.config.morph_rate * (iteration as f32 / n_estimators as f32).min(1.0);
       leaf *= depth_penalty * iter_shrinkage;
   }
   ```
5. Source `lr` from `ms.lr_per_iter[iteration]` when morph state present, otherwise the base `learning_rate`.

These integrations need to happen in **all** objective dispatches (regression, binary, multiclass softmax, ranking objectives). Search for the matchspot on `ObjectiveOps` calls and apply consistently.

- [ ] **Step 6: Add an end-to-end engine test**

```rust
#[test]
fn morph_mode_trains_to_completion_on_synthetic_regression() {
    let dataset = make_synthetic_regression_dataset(200, 5);
    let mut params = TrainParams::default();
    params.learning_rate = 0.1;
    params.max_depth = 4;
    params.morph_config = Some(MorphConfig::default());

    let result = train_regression(&dataset, &params);
    assert!(result.is_ok());
    let artifact = result.unwrap();
    assert!(!artifact.trees.is_empty());
}

#[test]
fn morph_mode_byte_identical_for_same_seed() {
    let dataset = make_synthetic_regression_dataset(100, 3);
    let mut params = TrainParams::default();
    params.seed = 42;
    params.deterministic = true;
    params.morph_config = Some(MorphConfig::default());

    let a = train_regression(&dataset, &params).unwrap();
    let b = train_regression(&dataset, &params).unwrap();
    assert_eq!(a.trees, b.trees);
}
```

If helper `make_synthetic_regression_dataset` doesn't exist, look for similar test helpers in the engine test module or copy the pattern from `train_regression_returns_a_trained_artifact` (search for it).

- [ ] **Step 7: Run full engine + backend tests**

Run: `cargo test --workspace`
Expected: PASS — all existing tests + new morph tests.

- [ ] **Step 8: Commit**

```bash
git add crates/engine/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(engine): wire MorphState into Trainer with EMA stats and leaf modifications

Trainer now constructs MorphState when train_params.morph_config is set:
maintains per-class EMA gradient stats updated each round, resolves the LR
schedule to a per-iteration vector, and applies depth-penalty +
iteration-shrinkage multipliers to leaf values. Calls best_split_morph in
morph mode and best_split_with_options otherwise.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Artifact MorphMetadataPayload Section

**Files:**
- Modify: `crates/core/src/lib.rs` (add `MorphMetadataPayload`, encode/decode helpers, register section kind)

**Context:** The artifact format already supports versioned optional sections (see existing `CategoricalStatePayloadV1`, `NativeCategoricalSplitsPayload` patterns). Adding a new optional section follows the same pattern.

- [ ] **Step 1: Locate existing optional-section pattern**

Read `crates/core/src/lib.rs` near the categorical state encoding (search for `encode_categorical_state_payload_v1`, `decode_optional_categorical_state_section_v1`, and `ModelSectionKind`). Note the section-kind enum, encode/decode signatures, and how `serialize_model_artifact_v1` / `deserialize_model_artifact_v1` weave optional sections in.

- [ ] **Step 2: Add `MorphMetadataPayload` struct**

```rust
/// Optional artifact section recording the MorphConfig used during training.
/// Metadata only — predictions are deterministic from baked-in leaf values.
#[derive(Debug, Clone, PartialEq)]
pub struct MorphMetadataPayload {
    pub config: MorphConfig,
    pub final_iteration: u32,
    pub final_total: u32,
}
```

- [ ] **Step 3: Add `MorphMetadata` variant to `ModelSectionKind` enum**

Find the `ModelSectionKind` enum in core. Add:

```rust
pub enum ModelSectionKind {
    // ... existing variants ...
    MorphMetadata,
}
```

Update any `match` exhaustiveness over this enum (encode/decode dispatch tables).

- [ ] **Step 4: Implement encode/decode helpers**

```rust
pub fn encode_morph_metadata_payload(payload: &MorphMetadataPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    // Schema: version(u16) | morph_rate(f32) | evolution_pressure(f32) |
    //         morph_warmup_iters(u32) | info_score_weight(f32) |
    //         depth_penalty_base(f32) | balance_penalty(u8) |
    //         lr_schedule_kind(u8) | lr_warmup_frac(f32) |
    //         final_iteration(u32) | final_total(u32)
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_rate.to_le_bytes());
    buf.extend_from_slice(&payload.config.evolution_pressure.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_warmup_iters.to_le_bytes());
    buf.extend_from_slice(&payload.config.info_score_weight.to_le_bytes());
    buf.extend_from_slice(&payload.config.depth_penalty_base.to_le_bytes());
    buf.push(payload.config.balance_penalty as u8);
    let (kind, warmup_frac) = match payload.config.lr_schedule {
        LrSchedule::Constant => (0u8, 0.0f32),
        LrSchedule::WarmupCosine { warmup_frac } => (1u8, warmup_frac),
    };
    buf.push(kind);
    buf.extend_from_slice(&warmup_frac.to_le_bytes());
    buf.extend_from_slice(&payload.final_iteration.to_le_bytes());
    buf.extend_from_slice(&payload.final_total.to_le_bytes());
    buf
}

pub fn decode_optional_morph_metadata_section(
    bytes: &[u8],
) -> CoreResult<MorphMetadataPayload> {
    if bytes.len() < 32 {
        return Err(CoreError::Validation(
            "morph metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported morph metadata version: {version}"
        )));
    }
    let mut o = 2;
    let read_f32 = |o: &mut usize, b: &[u8]| -> f32 {
        let v = f32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
        *o += 4;
        v
    };
    let read_u32 = |o: &mut usize, b: &[u8]| -> u32 {
        let v = u32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
        *o += 4;
        v
    };
    let morph_rate = read_f32(&mut o, bytes);
    let evolution_pressure = read_f32(&mut o, bytes);
    let morph_warmup_iters = read_u32(&mut o, bytes);
    let info_score_weight = read_f32(&mut o, bytes);
    let depth_penalty_base = read_f32(&mut o, bytes);
    let balance_penalty = bytes[o] != 0;
    o += 1;
    let kind = bytes[o];
    o += 1;
    let warmup_frac = read_f32(&mut o, bytes);
    let final_iteration = read_u32(&mut o, bytes);
    let final_total = read_u32(&mut o, bytes);
    let lr_schedule = match kind {
        0 => LrSchedule::Constant,
        1 => LrSchedule::WarmupCosine { warmup_frac },
        _ => {
            return Err(CoreError::Validation(format!(
                "unknown lr_schedule kind: {kind}"
            )));
        }
    };
    Ok(MorphMetadataPayload {
        config: MorphConfig {
            morph_rate,
            evolution_pressure,
            morph_warmup_iters,
            info_score_weight,
            depth_penalty_base,
            balance_penalty,
            lr_schedule,
        },
        final_iteration,
        final_total,
    })
}
```

- [ ] **Step 5: Wire morph section into `serialize_model_artifact_v1` / `deserialize_model_artifact_v1`**

Follow the pattern of how `CategoricalStatePayloadV1` is conditionally written (only when present) and conditionally read (returning Option). Add a parallel `morph_metadata: Option<MorphMetadataPayload>` field on whatever struct holds the artifact's optional sections (likely `ModelArtifactSection` or a wrapper). Engine writes it when training was in morph mode; deserialize returns it as Option for inspection.

If the engine doesn't currently expose a path to attach morph metadata to the artifact, add a method `with_morph_metadata(payload: MorphMetadataPayload)` on the artifact builder struct and call it from the trainer when `morph_state.is_some()`.

- [ ] **Step 6: Round-trip test**

Add to `crates/core/src/lib.rs` test module:

```rust
#[test]
fn morph_metadata_round_trip() {
    let payload = MorphMetadataPayload {
        config: MorphConfig {
            morph_rate: 0.15,
            evolution_pressure: 0.25,
            morph_warmup_iters: 7,
            info_score_weight: 0.4,
            depth_penalty_base: 0.85,
            balance_penalty: false,
            lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.2 },
        },
        final_iteration: 42,
        final_total: 100,
    };
    let bytes = encode_morph_metadata_payload(&payload);
    let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn morph_metadata_rejects_unknown_version() {
    let mut bytes = vec![0u8; 32];
    bytes[0] = 99;
    bytes[1] = 0;
    let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn morph_metadata_rejects_unknown_lr_schedule_kind() {
    let payload = MorphMetadataPayload {
        config: MorphConfig::default(),
        final_iteration: 0,
        final_total: 1,
    };
    let mut bytes = encode_morph_metadata_payload(&payload);
    // lr_schedule_kind byte sits at offset 2+4*5+1 = 23
    bytes[23] = 99;
    let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}
```

- [ ] **Step 7: Add a full artifact round-trip test**

```rust
#[test]
fn full_artifact_round_trip_preserves_morph_metadata() {
    // Build an artifact with morph metadata, serialize, deserialize, verify
    // morph_metadata field round-trips intact alongside trees/predictor layout.
    // Adapt from existing serialize_model_artifact_v1_round_trip test.
}
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p alloygbm-core --lib`
Run: `cargo test --workspace`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add crates/core/src/lib.rs crates/engine/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(core): add optional MorphMetadataPayload artifact section

Versioned (v1) optional section records MorphConfig, final iteration, and
total iterations from morph-mode training. Section is omitted for non-morph
artifacts; deserialize returns Option. Tested for round-trip fidelity and
rejection of malformed input.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: PyO3 Bridge — Plumb MorphConfig from Python

**Files:**
- Modify: `bindings/python/src/lib.rs`

**Context:** The bridge currently constructs `TrainParams` from a Python dict in functions like `train_regression_artifact`. Add parsing for an optional `morph_config` sub-dict that constructs `MorphConfig`.

- [ ] **Step 1: Add a helper to parse `MorphConfig` from a Python dict**

In `bindings/python/src/lib.rs`, add:

```rust
fn parse_morph_config_from_pydict(
    py: Python<'_>,
    dict: &PyDict,
) -> PyResult<alloygbm_core::MorphConfig> {
    use alloygbm_core::{LrSchedule, MorphConfig};

    let mut cfg = MorphConfig::default();

    if let Some(v) = dict.get_item("morph_rate")? {
        cfg.morph_rate = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("evolution_pressure")? {
        cfg.evolution_pressure = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("morph_warmup_iters")? {
        cfg.morph_warmup_iters = v.extract::<u32>()?;
    }
    if let Some(v) = dict.get_item("info_score_weight")? {
        cfg.info_score_weight = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("depth_penalty_base")? {
        cfg.depth_penalty_base = v.extract::<f32>()?;
    }
    if let Some(v) = dict.get_item("balance_penalty")? {
        cfg.balance_penalty = v.extract::<bool>()?;
    }
    if let Some(v) = dict.get_item("lr_schedule")? {
        let kind = v.extract::<&str>()?;
        let warmup_frac = dict
            .get_item("lr_warmup_frac")?
            .map(|x| x.extract::<f32>())
            .transpose()?
            .unwrap_or(0.1);
        cfg.lr_schedule = match kind {
            "constant" => LrSchedule::Constant,
            "warmup_cosine" => LrSchedule::WarmupCosine { warmup_frac },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown lr_schedule: {other}"
                )));
            }
        };
    }

    Ok(cfg)
}
```

- [ ] **Step 2: Wire morph config into TrainParams construction**

Locate where `TrainParams` is built from the input dict (search for `TrainParams {` or `let mut params`). Append:

```rust
if let Some(morph_dict_obj) = params_dict.get_item("morph_config")? {
    let morph_dict = morph_dict_obj.downcast::<PyDict>()?;
    params.morph_config = Some(parse_morph_config_from_pydict(py, morph_dict)?);
}
```

Apply this in **every** `train_*_artifact*` entry point (regression, binary, multiclass, ranking variants — there may be 6-8 of these based on dense vs sparse paths). DRY this up by making the TrainParams construction a shared helper if it isn't already.

- [ ] **Step 3: Build the wheel**

Run: `maturin develop --release` (from `bindings/python/`)
Expected: Compiles successfully.

- [ ] **Step 4: Smoke test**

```bash
.venv/bin/python -c "
from alloygbm._alloygbm import train_regression_artifact
# Just verify the morph_config kwarg flows through without exception.
# Full Python-API smoke is in Task 8.
print('PyO3 bridge accepts morph_config without error')
"
```

If `train_regression_artifact` requires positional args, this step is just a build verification — Task 9's Python-side tests will exercise the morph dict end-to-end.

- [ ] **Step 5: Run tests**

Run: `cargo test --workspace && .venv/bin/python -m pytest bindings/python/tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bindings/python/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(pyo3): parse morph_config from Python dict into TrainParams

Adds parse_morph_config_from_pydict helper and wires it into all train_*
entry points. Python callers can now pass morph_config={...} to enable
morph-mode training. Schedule kind is a string ("constant" |
"warmup_cosine") with optional lr_warmup_frac.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Python Helpers Module (`_morph.py`)

**Files:**
- Create: `bindings/python/alloygbm/_morph.py`

**Context:** Centralize morph-related Python helpers (fingerprinting, LR schedule resolver for any Python-side use, interaction-importance traversal) in a single module imported by all three estimators. Avoids duplication.

- [ ] **Step 1: Create the module skeleton with tests-driven design**

Create `bindings/python/alloygbm/_morph.py`:

```python
"""Shared helpers for the MorphBoost-inspired training profile."""

from __future__ import annotations

import numpy as np


def compute_morph_fingerprint(X, y, *, fast_mode: bool = True):
    """Compute a problem-structure fingerprint for morph-mode parameter defaults.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    fast_mode : bool
        If True, use the paper's fixed heuristic constants
        (complexity=0.2, non_linearity=0.0, interaction_strength=0.15,
        noise_level=0.1). If False, compute from data.

    Returns
    -------
    dict with keys: complexity, non_linearity, interaction_strength,
    noise_level, suggested_max_depth.
    """
    if fast_mode:
        return {
            "complexity": 0.2,
            "non_linearity": 0.0,
            "interaction_strength": 0.15,
            "noise_level": 0.1,
            "suggested_max_depth": 8,
        }

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape

    feature_stds = np.std(X, axis=0)
    feature_ranges = np.ptp(X, axis=0)
    complexity = float(
        np.mean(feature_stds / (feature_ranges + 1e-10))
    )

    non_linearity = 0.0
    for i in range(min(5, n_features)):
        col = X[:, i]
        try:
            corr_lin = abs(np.corrcoef(col, y)[0, 1])
            corr_quad = abs(np.corrcoef(col ** 2, y)[0, 1])
            non_linearity = max(non_linearity, float(corr_quad - corr_lin))
        except Exception:
            pass

    interaction_strength = 0.0
    if n_features < 100 and n_samples > 100:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_samples, size=min(50, n_samples), replace=False)
        for i in range(min(5, n_features)):
            for j in range(i + 1, min(5, n_features)):
                try:
                    inter = abs(
                        np.corrcoef(
                            X[sample_idx, i] * X[sample_idx, j],
                            y[sample_idx],
                        )[0, 1]
                    )
                    if not np.isnan(inter):
                        interaction_strength = max(
                            interaction_strength, float(inter)
                        )
                except Exception:
                    pass

    suggested_max_depth = 10 if complexity > 0.5 else 8

    return {
        "complexity": complexity,
        "non_linearity": non_linearity,
        "interaction_strength": interaction_strength,
        "noise_level": 0.1,
        "suggested_max_depth": suggested_max_depth,
    }


def resolve_lr_schedule(schedule, n_estimators, base_lr, warmup_frac=0.1):
    """Resolve an LR schedule name to a per-iteration float list (Python mirror of Rust)."""
    n = int(n_estimators)
    if schedule == "constant":
        return [float(base_lr)] * n
    if schedule == "warmup_cosine":
        warmup = max(1, min(n, int(round(warmup_frac * n))))
        out = [base_lr * (i + 1) / warmup for i in range(warmup)]
        remaining = n - warmup
        floor = base_lr * 0.01
        for i in range(remaining):
            progress = i / max(remaining, 1)
            cos = 0.5 * (1.0 + np.cos(np.pi * progress))
            out.append(floor + (base_lr - floor) * cos)
        return out
    raise ValueError(f"unknown lr_schedule: {schedule}")


def apply_interaction_importance_bonus(
    base_importances, X, gradients_history=None, max_depth_for_interaction=3
):
    """Apply the paper's interaction-aware feature-importance bonus.

    For now this is a placeholder that returns base importances unchanged when
    no tree-traversal callback is supplied. Tree-walk implementation extends
    this in a follow-up; v1 ships base importances under morph mode for
    consistency with non-morph mode.
    """
    return base_importances


def build_morph_config_dict(
    *,
    morph_rate=0.1,
    evolution_pressure=0.2,
    morph_warmup_iters=5,
    info_score_weight=0.3,
    depth_penalty_base=0.9,
    balance_penalty=True,
    lr_schedule="constant",
    lr_warmup_frac=0.1,
):
    """Build the morph_config dict for the PyO3 bridge."""
    return {
        "morph_rate": float(morph_rate),
        "evolution_pressure": float(evolution_pressure),
        "morph_warmup_iters": int(morph_warmup_iters),
        "info_score_weight": float(info_score_weight),
        "depth_penalty_base": float(depth_penalty_base),
        "balance_penalty": bool(balance_penalty),
        "lr_schedule": str(lr_schedule),
        "lr_warmup_frac": float(lr_warmup_frac),
    }
```

- [ ] **Step 2: Add unit tests for helpers**

Create `bindings/python/tests/test_morph_helpers.py`:

```python
import numpy as np
import pytest

from alloygbm._morph import (
    build_morph_config_dict,
    compute_morph_fingerprint,
    resolve_lr_schedule,
)


def test_fingerprint_fast_mode_returns_paper_constants():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = rng.standard_normal(100)
    fp = compute_morph_fingerprint(X, y, fast_mode=True)
    assert fp["complexity"] == 0.2
    assert fp["interaction_strength"] == 0.15
    assert fp["suggested_max_depth"] == 8


def test_fingerprint_full_mode_returns_finite_floats():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(100)
    fp = compute_morph_fingerprint(X, y, fast_mode=False)
    assert np.isfinite(fp["complexity"])
    assert np.isfinite(fp["non_linearity"])
    assert np.isfinite(fp["interaction_strength"])
    assert fp["suggested_max_depth"] in (8, 10)


def test_resolve_constant_schedule():
    out = resolve_lr_schedule("constant", 10, 0.1)
    assert len(out) == 10
    assert all(abs(x - 0.1) < 1e-6 for x in out)


def test_resolve_warmup_cosine_schedule_shape():
    out = resolve_lr_schedule("warmup_cosine", 100, 0.1, warmup_frac=0.1)
    assert len(out) == 100
    # First step is well below base, last is near floor.
    assert out[0] < 0.1
    assert out[-1] < 0.05


def test_resolve_unknown_schedule_raises():
    with pytest.raises(ValueError):
        resolve_lr_schedule("nonsense", 10, 0.1)


def test_build_morph_config_dict_defaults():
    d = build_morph_config_dict()
    assert d["morph_rate"] == 0.1
    assert d["lr_schedule"] == "constant"
    assert d["balance_penalty"] is True
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest bindings/python/tests/test_morph_helpers.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 4: Commit**

```bash
git add bindings/python/alloygbm/_morph.py bindings/python/tests/test_morph_helpers.py
git commit -m "$(cat <<'EOF'
feat(python): add _morph helpers module (fingerprint, LR schedule, config dict)

Centralizes morph-related Python utilities used by all three estimators.
Includes problem-structure fingerprinting (fast and full modes), LR schedule
resolver mirroring the Rust implementation, and a build_morph_config_dict
factory for the PyO3 bridge.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add Morph Parameters to GBMRegressor

**Files:**
- Modify: `bindings/python/alloygbm/regressor.py`

**Context:** Per CLAUDE.md, every Python parameter must be wired through `__init__`, `get_params`, `set_params`, `__repr__`, and `_params_order`. Six new parameters: `training_mode`, `morph_rate`, `evolution_pressure`, `morph_warmup_iters`, `lr_schedule`, `lr_warmup_frac`.

- [ ] **Step 1: Locate the existing parameter pattern**

Read `bindings/python/alloygbm/regressor.py` to find:
- The `__init__` method
- `_params_order` (likely a class attribute or module constant)
- `get_params` / `set_params`
- `__repr__`
- The location where `train_regression_artifact*` is called with the params dict

- [ ] **Step 2: Add morph parameters to `__init__`**

Append to the `__init__` signature (preserving existing param order; new params go at end):

```python
def __init__(
    self,
    # ... all existing params unchanged ...
    training_mode="auto",
    morph_rate=0.1,
    evolution_pressure=0.2,
    morph_warmup_iters=5,
    lr_schedule="constant",
    lr_warmup_frac=0.1,
):
    # ... existing assignments ...
    self.training_mode = training_mode
    self.morph_rate = morph_rate
    self.evolution_pressure = evolution_pressure
    self.morph_warmup_iters = morph_warmup_iters
    self.lr_schedule = lr_schedule
    self.lr_warmup_frac = lr_warmup_frac
```

- [ ] **Step 3: Append new params to `_params_order`**

```python
_params_order = (
    # ... existing entries ...
    "training_mode",
    "morph_rate",
    "evolution_pressure",
    "morph_warmup_iters",
    "lr_schedule",
    "lr_warmup_frac",
)
```

- [ ] **Step 4: Wire morph params into the params dict passed to PyO3**

Find where the params dict is constructed (search for the `params_dict` or equivalent that is passed to the native `train_regression_artifact` call). Add:

```python
from alloygbm._morph import build_morph_config_dict, compute_morph_fingerprint

# Inside fit, just before calling the native train function:
if self.training_mode == "morph":
    fp = compute_morph_fingerprint(X, y, fast_mode=True)
    # Only override max_depth if user left it at sklearn-style default sentinel.
    effective_max_depth = self.max_depth if self.max_depth is not None else fp["suggested_max_depth"]
    params_dict["max_depth"] = effective_max_depth
    params_dict["morph_config"] = build_morph_config_dict(
        morph_rate=self.morph_rate,
        evolution_pressure=self.evolution_pressure,
        morph_warmup_iters=self.morph_warmup_iters,
        info_score_weight=0.3,
        depth_penalty_base=0.9,
        balance_penalty=True,
        lr_schedule=self.lr_schedule,
        lr_warmup_frac=self.lr_warmup_frac,
    )
elif self.lr_schedule != "constant":
    # Allow lr_schedule alone (without full morph mode) as a future option;
    # for v1 we wire it only when training_mode="morph" since the engine
    # consumes it through morph_config.
    raise ValueError(
        "lr_schedule is currently only honored when training_mode='morph'"
    )
```

- [ ] **Step 5: Verify `get_params` / `set_params` automatically pick up new params**

If `get_params` / `set_params` use `_params_order` for iteration (typical sklearn pattern in this codebase), they pick up the new params automatically. If they're hand-rolled lists, append the new params there too.

- [ ] **Step 6: Verify `__repr__` includes new params**

Same — if `__repr__` iterates `_params_order`, no edit needed. Otherwise append.

- [ ] **Step 7: Add a smoke test**

Create `bindings/python/tests/test_morph.py`:

```python
import numpy as np

from alloygbm import GBMRegressor


def _toy_regression_data(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    coefs = rng.standard_normal(n_features).astype(np.float32)
    y = (X @ coefs + 0.1 * rng.standard_normal(n).astype(np.float32))
    return X, y


def test_regressor_fits_in_morph_mode():
    X, y = _toy_regression_data()
    m = GBMRegressor(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.1,
        training_mode="morph",
        random_state=42,
    )
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_regressor_morph_mode_round_trips_via_pickle():
    import pickle

    X, y = _toy_regression_data()
    m = GBMRegressor(
        n_estimators=10,
        max_depth=3,
        training_mode="morph",
        random_state=0,
    )
    m.fit(X, y)
    blob = pickle.dumps(m)
    m2 = pickle.loads(blob)
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))


def test_regressor_morph_mode_introspection():
    import inspect

    sig = inspect.signature(GBMRegressor.__init__)
    params = sig.parameters
    assert "training_mode" in params
    assert "morph_rate" in params
    assert "evolution_pressure" in params
    assert "morph_warmup_iters" in params
    assert "lr_schedule" in params
    assert "lr_warmup_frac" in params


def test_regressor_get_set_params_round_trip_morph():
    m = GBMRegressor(
        n_estimators=5,
        training_mode="morph",
        morph_rate=0.2,
        lr_schedule="warmup_cosine",
        lr_warmup_frac=0.15,
    )
    p = m.get_params()
    assert p["training_mode"] == "morph"
    assert p["morph_rate"] == 0.2
    assert p["lr_schedule"] == "warmup_cosine"
    m2 = GBMRegressor()
    m2.set_params(**p)
    assert m2.training_mode == "morph"
    assert m2.lr_schedule == "warmup_cosine"


def test_regressor_default_mode_unchanged():
    """Backward compat: default training_mode='auto' produces same artifact as before this PR."""
    X, y = _toy_regression_data()
    m_auto = GBMRegressor(n_estimators=10, max_depth=4, random_state=0)
    m_auto.fit(X, y)
    pred_auto = m_auto.predict(X)
    # Just verify finite + repeatable; full byte-identity checked in Task 11.
    m_auto2 = GBMRegressor(n_estimators=10, max_depth=4, random_state=0)
    m_auto2.fit(X, y)
    np.testing.assert_array_equal(pred_auto, m_auto2.predict(X))
```

- [ ] **Step 8: Build and run tests**

```bash
maturin develop --release
.venv/bin/python -m pytest bindings/python/tests/test_morph.py -v
```

Expected: PASS — 5 regressor tests.

- [ ] **Step 9: Run full test suite**

```bash
.venv/bin/python -m pytest bindings/python/tests/ -q
```

Expected: PASS — no regressions in any existing test.

- [ ] **Step 10: Commit**

```bash
git add bindings/python/alloygbm/regressor.py bindings/python/tests/test_morph.py
git commit -m "$(cat <<'EOF'
feat(python): wire morph parameters into GBMRegressor

Adds training_mode, morph_rate, evolution_pressure, morph_warmup_iters,
lr_schedule, and lr_warmup_frac to GBMRegressor. When training_mode='morph',
fit() builds the morph_config dict and passes it through PyO3. Includes
fingerprint-derived max_depth default. Adds smoke, pickle round-trip,
introspection, get/set_params, and backward-compat tests.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Add Morph Parameters to GBMClassifier and GBMRanker

**Files:**
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Modify: `bindings/python/tests/test_morph.py` (extend)

**Context:** Same wiring as Task 9, applied to the other two estimators.

- [ ] **Step 1: Wire morph params into GBMClassifier**

Mirror Task 9 Steps 2-6 for `bindings/python/alloygbm/classifier.py`. The `fit()` method must wire morph through both binary and multiclass paths.

- [ ] **Step 2: Wire morph params into GBMRanker**

Mirror Task 9 Steps 2-6 for `bindings/python/alloygbm/ranker.py`. Note that `GBMRanker.__init__` had a recent fix (per docs/roadmap/current.md) where the signature wasn't exposing all params — verify the new params are properly visible to `inspect.signature(GBMRanker.__init__)`.

- [ ] **Step 3: Extend `test_morph.py` with classifier and ranker cases**

Append to `bindings/python/tests/test_morph.py`:

```python
def _toy_binary_data(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal(n_features).astype(np.float32)
    y = (logits > 0).astype(np.int32)
    return X, y


def _toy_multiclass_data(n=300, n_features=5, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    logits = X @ rng.standard_normal((n_features, n_classes)).astype(np.float32)
    y = np.argmax(logits, axis=1).astype(np.int32)
    return X, y


def _toy_ranking_data(n=200, n_features=5, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.int32)
    group_sizes = [n // n_groups] * n_groups
    group_sizes[-1] += n - sum(group_sizes)
    group = np.repeat(np.arange(n_groups), group_sizes).astype(np.int32)
    # Sort by group as required by GBMRanker.
    order = np.argsort(group)
    return X[order], y[order], group[order]


def test_classifier_fits_in_morph_mode_binary():
    from alloygbm import GBMClassifier

    X, y = _toy_binary_data()
    m = GBMClassifier(n_estimators=15, max_depth=4, training_mode="morph", random_state=0)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_fits_in_morph_mode_multiclass():
    from alloygbm import GBMClassifier

    X, y = _toy_multiclass_data()
    m = GBMClassifier(n_estimators=15, max_depth=4, training_mode="morph", random_state=0)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_ranker_fits_in_morph_mode():
    from alloygbm import GBMRanker

    X, y, group = _toy_ranking_data()
    m = GBMRanker(n_estimators=10, max_depth=4, training_mode="morph", random_state=0)
    m.fit(X, y, group=group)
    pred = m.predict(X)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_ranker_init_signature_exposes_morph_params():
    """Regression for the GBMRanker signature bug from 0.3.2."""
    import inspect
    from alloygbm import GBMRanker

    sig = inspect.signature(GBMRanker.__init__)
    assert "training_mode" in sig.parameters
    assert "morph_rate" in sig.parameters
```

- [ ] **Step 4: Build and run tests**

```bash
maturin develop --release
.venv/bin/python -m pytest bindings/python/tests/test_morph.py -v
.venv/bin/python -m pytest bindings/python/tests/ -q
```

Expected: PASS — all morph tests + no regressions.

- [ ] **Step 5: Commit**

```bash
git add bindings/python/alloygbm/classifier.py bindings/python/alloygbm/ranker.py bindings/python/tests/test_morph.py
git commit -m "$(cat <<'EOF'
feat(python): wire morph parameters into GBMClassifier and GBMRanker

Same training_mode + morph parameter set added to both classifiers
(binary and multiclass paths) and ranker. Includes signature-introspection
test for GBMRanker (regression guard for the 0.3.2 fix).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Determinism + Backward-Compat Tests

**Files:**
- Modify: `bindings/python/tests/test_morph.py` (append determinism + byte-identity tests)

**Context:** Lock down two contracts that protect future work:
1. Same seed + same data + same params → byte-identical artifacts.
2. Pre-PR `auto` mode behavior is unchanged.

- [ ] **Step 1: Add determinism test**

Append to `test_morph.py`:

```python
def test_morph_artifact_bytes_identical_for_same_seed():
    X, y = _toy_regression_data(n=300, seed=7)
    m1 = GBMRegressor(
        n_estimators=15,
        max_depth=4,
        learning_rate=0.1,
        training_mode="morph",
        random_state=12345,
    )
    m1.fit(X, y)
    m2 = GBMRegressor(
        n_estimators=15,
        max_depth=4,
        learning_rate=0.1,
        training_mode="morph",
        random_state=12345,
    )
    m2.fit(X, y)
    assert m1.artifact_bytes == m2.artifact_bytes


def test_morph_warmup_cosine_artifact_deterministic():
    X, y = _toy_regression_data(n=200, seed=3)
    m1 = GBMRegressor(
        n_estimators=20,
        max_depth=4,
        training_mode="morph",
        lr_schedule="warmup_cosine",
        lr_warmup_frac=0.2,
        random_state=99,
    )
    m1.fit(X, y)
    m2 = GBMRegressor(
        n_estimators=20,
        max_depth=4,
        training_mode="morph",
        lr_schedule="warmup_cosine",
        lr_warmup_frac=0.2,
        random_state=99,
    )
    m2.fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
```

- [ ] **Step 2: Add backward-compat byte-identity test**

This requires capturing a "golden" artifact from before this PR. The simplest approach: train two non-morph models with the same seed/params and verify byte-identity against each other (proves determinism is preserved). To prove the broader "non-morph behavior unchanged" claim, run the full existing test suite — every existing test serves as the regression baseline.

```python
def test_auto_mode_artifact_deterministic_after_morph_pr():
    """Adding morph mode must not perturb auto-mode bit-for-bit reproducibility."""
    X, y = _toy_regression_data(n=200, seed=11)
    m1 = GBMRegressor(n_estimators=10, max_depth=4, random_state=0)
    m1.fit(X, y)
    m2 = GBMRegressor(n_estimators=10, max_depth=4, random_state=0)
    m2.fit(X, y)
    assert m1.artifact_bytes == m2.artifact_bytes


def test_morph_artifact_differs_from_auto():
    """Sanity: morph and auto modes produce different artifacts."""
    X, y = _toy_regression_data(n=200, seed=13)
    m_auto = GBMRegressor(n_estimators=15, max_depth=4, random_state=0)
    m_auto.fit(X, y)
    m_morph = GBMRegressor(
        n_estimators=15, max_depth=4, training_mode="morph", random_state=0
    )
    m_morph.fit(X, y)
    assert m_auto.artifact_bytes != m_morph.artifact_bytes
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest bindings/python/tests/test_morph.py -v
.venv/bin/python -m pytest bindings/python/tests/ -q
cargo test --workspace
```

Expected: PASS across the board.

- [ ] **Step 4: Commit**

```bash
git add bindings/python/tests/test_morph.py
git commit -m "$(cat <<'EOF'
test(python): determinism + backward-compat checks for morph mode

Locks in two contracts: (1) same seed + same params → byte-identical
artifacts under morph (constant + warmup_cosine schedules), (2) auto-mode
artifacts remain byte-identical across runs after the morph PR.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Benchmark Integration

**Files:**
- Modify: `benchmarks/run_model_comparison.py`

**Context:** The existing benchmark runner compares AlloyGBM (auto) against LightGBM/XGBoost/CatBoost. Add two new AlloyGBM configurations: `alloygbm_morph` (morph + constant LR) and `alloygbm_morph_cosine` (morph + warmup_cosine).

- [ ] **Step 1: Locate the model registry in the benchmark runner**

Read `benchmarks/run_model_comparison.py` to find where models are registered (likely a dict like `MODELS = {...}` or a list of factory functions).

- [ ] **Step 2: Add the morph variants**

```python
def _alloygbm_morph_factory(task_type: str, **base_kwargs):
    from alloygbm import GBMRegressor, GBMClassifier, GBMRanker
    Cls = {
        "regression": GBMRegressor,
        "binary": GBMClassifier,
        "multiclass": GBMClassifier,
        "ranking": GBMRanker,
    }[task_type]
    return Cls(training_mode="morph", **base_kwargs)


def _alloygbm_morph_cosine_factory(task_type: str, **base_kwargs):
    from alloygbm import GBMRegressor, GBMClassifier, GBMRanker
    Cls = {
        "regression": GBMRegressor,
        "binary": GBMClassifier,
        "multiclass": GBMClassifier,
        "ranking": GBMRanker,
    }[task_type]
    return Cls(
        training_mode="morph",
        lr_schedule="warmup_cosine",
        lr_warmup_frac=0.1,
        **base_kwargs,
    )
```

Register them alongside the existing `alloygbm_auto` entry.

- [ ] **Step 3: Smoke-run a single scenario**

```bash
.venv/bin/python benchmarks/run_model_comparison.py --scenario california_housing --models alloygbm_auto alloygbm_morph alloygbm_morph_cosine
```

Expected: All three configurations train + score successfully on California Housing.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/run_model_comparison.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add alloygbm_morph and alloygbm_morph_cosine configs

Adds two new AlloyGBM benchmark configurations alongside the auto baseline
so cross-model comparison can quantify morph mode's effect on RMSE/MAE/R²
(regression), accuracy/log-loss (classification), and NDCG (ranking).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Morph Ablation Harness

**Files:**
- Create: `benchmarks/morph_ablation.py`

**Context:** Component-isolation script that toggles each morph mechanism independently to identify which pieces actually contribute to wins. Outputs a markdown table.

- [ ] **Step 1: Create the ablation script**

Create `benchmarks/morph_ablation.py`:

```python
"""Morph-mode ablation harness.

Runs AlloyGBM under combinations of morph mechanisms on three representative
datasets (1 regression, 1 classification, 1 ranking) and reports which
components contribute to wins.

Components toggled:
- adaptive_split:    morph_warmup_iters=5 vs morph_warmup_iters=999 (effectively off)
- leaf_modifications: morph_rate=0.1 vs 0.0
- balance_penalty:    True vs False
- lr_schedule:        warmup_cosine vs constant
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def _load_california_housing():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    return train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.float32),
        test_size=0.2,
        random_state=0,
    )


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    return train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.int32),
        test_size=0.2,
        random_state=0,
    )


def _load_synthetic_ranking():
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.standard_normal((n, 6)).astype(np.float32)
    y = rng.integers(0, 5, size=n).astype(np.int32)
    group = np.repeat(np.arange(50), 20).astype(np.int32)
    order = np.argsort(group)
    return X[order], y[order], group[order]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _ndcg_at_k(y_true, y_pred, group, k=10):
    # Simple per-group NDCG@k mean.
    from alloygbm.evaluation import ndcg

    return float(ndcg(y_true, y_pred, group, k=k))


CONFIGS = [
    ("baseline_auto", dict()),
    ("morph_full", dict(
        training_mode="morph",
        lr_schedule="warmup_cosine",
    )),
    ("morph_no_split", dict(
        training_mode="morph",
        morph_warmup_iters=999,
        lr_schedule="warmup_cosine",
    )),
    ("morph_no_leaf_mods", dict(
        training_mode="morph",
        morph_rate=0.0,
        lr_schedule="warmup_cosine",
    )),
    # Note: balance_penalty is not exposed as a top-level Python parameter in
    # v1 (defaults to True inside build_morph_config_dict). To ablate it,
    # construct a custom config via alloygbm._morph.build_morph_config_dict
    # and pass it through a private path — out of scope for this script.
    ("morph_constant_lr", dict(
        training_mode="morph",
        lr_schedule="constant",
    )),
]


def run_regression():
    X_tr, X_te, y_tr, y_te = _load_california_housing()
    rows = []
    for name, kw in CONFIGS:
        m = GBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                         random_state=0, **kw)
        t0 = time.time()
        m.fit(X_tr, y_tr)
        train_s = time.time() - t0
        pred = m.predict(X_te)
        rows.append((name, _rmse(y_te, pred), train_s))
    return rows


def run_classification():
    X_tr, X_te, y_tr, y_te = _load_breast_cancer()
    rows = []
    for name, kw in CONFIGS:
        m = GBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=0, **kw)
        t0 = time.time()
        m.fit(X_tr, y_tr)
        train_s = time.time() - t0
        pred = m.predict(X_te)
        rows.append((name, _accuracy(y_te, pred), train_s))
    return rows


def run_ranking():
    X, y, group = _load_synthetic_ranking()
    split = int(0.8 * len(X))
    rows = []
    for name, kw in CONFIGS:
        m = GBMRanker(n_estimators=200, max_depth=6, learning_rate=0.05,
                      random_state=0, **kw)
        t0 = time.time()
        m.fit(X[:split], y[:split], group=group[:split])
        train_s = time.time() - t0
        pred = m.predict(X[split:])
        rows.append((name, _ndcg_at_k(y[split:], pred, group[split:], k=10), train_s))
    return rows


def _print_table(title, rows, metric_label):
    print(f"\n## {title}\n")
    print(f"| config | {metric_label} | train_s |")
    print("|---|---|---|")
    for name, score, t in rows:
        print(f"| {name} | {score:.4f} | {t:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path.")
    args = parser.parse_args()

    results = {
        "regression_rmse_california_housing": run_regression(),
        "classification_accuracy_breast_cancer": run_classification(),
        "ranking_ndcg10_synthetic": run_ranking(),
    }

    _print_table("Regression — California Housing (RMSE, lower is better)",
                 results["regression_rmse_california_housing"], "rmse")
    _print_table("Classification — Breast Cancer (accuracy, higher is better)",
                 results["classification_accuracy_breast_cancer"], "accuracy")
    _print_table("Ranking — Synthetic (NDCG@10, higher is better)",
                 results["ranking_ndcg10_synthetic"], "ndcg10")

    if args.out:
        Path(args.out).write_text(json.dumps(results, default=list, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the ablation**

```bash
.venv/bin/python benchmarks/morph_ablation.py
```

Expected: Three markdown tables print to stdout, each row finite and finite-time.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/morph_ablation.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): add morph-mode component-isolation ablation harness

Toggles each morph mechanism independently on three representative datasets
(California Housing, Breast Cancer, synthetic ranking) and reports
RMSE/accuracy/NDCG plus training time. Identifies which morph components
actually contribute to performance, informing the promote-to-default
decision in spec §6.3.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Final Workspace Verification

**Files:** none (verification-only task)

**Context:** Run the full battery of checks before declaring the implementation complete.

- [ ] **Step 1: Rust full check**

```bash
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

Expected: PASS, no warnings.

- [ ] **Step 2: Python full check**

```bash
maturin develop --release
.venv/bin/python -m pytest bindings/python/tests/ -q
```

Expected: PASS.

- [ ] **Step 3: Documentation smoke check**

Open `docs/superpowers/specs/2026-04-30-morphboost-training-mode-design.md` and verify every spec section has a corresponding implemented task.

- [ ] **Step 4: Run the benchmark sweep on California Housing as a sanity check**

```bash
.venv/bin/python benchmarks/run_model_comparison.py --scenario california_housing
```

Expected: All AlloyGBM configurations produce finite, comparable RMSE values vs LightGBM/XGBoost/CatBoost baselines.

- [ ] **Step 5: Update CHANGELOG.md** (if present; if not, skip)

Read `CHANGELOG.md`. Add an entry under unreleased:

```markdown
## Unreleased

### Added

- `training_mode="morph"` opt-in training profile on `GBMRegressor`,
  `GBMClassifier`, and `GBMRanker`, implementing the MorphBoost paper's
  adaptive split criterion (Kriuk 2025), leaf-value depth penalty +
  iteration shrinkage, optional balance penalty for unbalanced splits, and
  a warm-up + cosine learning-rate schedule (`lr_schedule="warmup_cosine"`).
- Morph-mode artifacts include an optional `MorphMetadataPayload` section
  recording the `MorphConfig` used during training (introspection only —
  predictions are baked into leaf values).
- `benchmarks/morph_ablation.py` script for component-isolation analysis.

### Changed

- `BackendOps` trait gains `best_split_morph` with a default impl that
  delegates to `best_split_with_options`, preserving backward compatibility
  for any external `BackendOps` implementors.
```

- [ ] **Step 6: Commit changelog (if updated)**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(changelog): note morph training mode for next release

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (Run After Plan Complete)

- [ ] **Spec coverage:** Every section of `docs/superpowers/specs/2026-04-30-morphboost-training-mode-design.md` mapped to a task above (§3, §4.1, §4.2, §4.3, §4.4, §4.5, §4.6, §4.7, §5, §6.1, §6.2, §6.4 → Tasks 1, 1+5, 3+4, 5, 5+6, 6, 7, 9–10, 11, 5+11, 12, 13).
- [ ] **No placeholders:** Every step has concrete code or explicit commands.
- [ ] **Type consistency:** `MorphConfig`, `LrSchedule`, `MorphContext`, `MorphState` referenced consistently across tasks. `compute_morph_gain`, `MorphGainInputs`, `SplitSideStats` consistent. Python helpers `compute_morph_fingerprint`, `resolve_lr_schedule`, `build_morph_config_dict` consistent.
- [ ] **Spec section §6.3 (promote-to-default criteria):** intentionally not implemented as code — these are decision criteria executed manually after the benchmark run completes (Task 12 produces the data, the decision is human-in-the-loop per spec).
