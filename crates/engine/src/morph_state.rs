use alloygbm_core::{
    GradientEmaStats, GradientPair, LrSchedule, MorphConfig, MorphPrecomputed,
};

use crate::MorphContext;

/// Trainer-resident state for morph-mode training.
///
/// Holds the per-class EMA gradient statistics (length 1 for single-output
/// objectives, K for multiclass softmax) and the resolved per-iteration
/// learning-rate schedule.
#[derive(Debug, Clone)]
pub struct MorphState {
    pub config: MorphConfig,
    /// Per-class EMA stats (length 1 for single-output, K for multiclass).
    pub ema_stats: Vec<GradientEmaStats>,
    /// Resolved per-iteration learning rates (length = `total_iterations`).
    pub lr_per_iter: Vec<f32>,
    /// Reused per round to avoid allocations when extracting grad values for EMA.
    grad_scratch: Vec<f32>,
}

impl MorphState {
    pub fn new(config: MorphConfig, n_classes: usize, total_iterations: u32, base_lr: f32) -> Self {
        Self {
            config,
            ema_stats: vec![GradientEmaStats::default(); n_classes.max(1)],
            lr_per_iter: resolve_lr_schedule(config.lr_schedule, total_iterations, base_lr),
            grad_scratch: Vec::new(),
        }
    }

    pub fn morph_context(&self, iteration: u32, total: u32, class_idx: usize) -> MorphContext {
        let stats = &self.ema_stats[class_idx.min(self.ema_stats.len().saturating_sub(1))];
        MorphContext {
            iteration,
            total_iterations: total,
            grad_mean: stats.mean,
            grad_std: stats.std,
            config: self.config,
            precomputed: MorphPrecomputed::for_iteration(iteration, &self.config),
        }
    }

    pub fn lr_for_iter(&self, iteration: usize) -> f32 {
        self.lr_per_iter
            .get(iteration)
            .copied()
            .unwrap_or_else(|| self.lr_per_iter.last().copied().unwrap_or(0.1))
    }

    /// Returns a scale factor for `min_loss_improvement` at the given iteration.
    ///
    /// Loss reduction in a GBM round is approximately linear in the learning rate
    /// (leaf values are `-lr × grad / hess`, so per-leaf contribution to loss
    /// reduction is `lr × grad² / hess`). When an LR schedule is active, early
    /// (warmup) and late (decay) rounds intentionally use a smaller LR than the
    /// schedule's peak, so the absolute loss improvement is correspondingly
    /// smaller. Without this normalization, the auto-tuned weak-improvement
    /// early-stopping threshold would terminate training during warmup.
    ///
    /// Returns `current_lr / max_lr_in_schedule`, clamped to `[0.01, 1.0]`. The
    /// lower clamp prevents pathological zeroing if a schedule ever uses lr=0;
    /// the upper clamp is mathematical (current_lr cannot exceed the max).
    ///
    /// For a `Constant` schedule, this is always 1.0 (no-op).
    pub fn lr_loss_threshold_scale(&self, iteration: usize) -> f32 {
        let current_lr = self.lr_for_iter(iteration);
        let max_lr = self
            .lr_per_iter
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_lr.is_finite() || max_lr <= 0.0 {
            return 1.0;
        }
        (current_lr / max_lr).clamp(0.01, 1.0)
    }

    /// Returns `true` when the given iteration falls within the explicit
    /// warmup phase of a non-constant LR schedule.
    ///
    /// During warmup, the learning rate ramps up from a small fraction of the
    /// peak. Early-stopping signals — empty trees (`NoSplitCandidate`) and weak
    /// loss improvement — are expected consequences of the tiny LR and must NOT
    /// trigger termination. Once warmup completes and LR reaches its peak,
    /// these signals regain their normal "training has truly stalled" meaning.
    ///
    /// Boundary semantics mirror `resolve_lr_schedule` exactly: warmup spans
    /// `[0, round((warmup_frac * n_estimators)).max(1).min(n))`.
    ///
    /// For `LrSchedule::Constant`, returns `false` at all iterations.
    pub fn is_in_warmup_phase(&self, iteration: usize) -> bool {
        match self.config.lr_schedule {
            LrSchedule::Constant => false,
            LrSchedule::WarmupCosine { warmup_frac } => {
                let n = self.lr_per_iter.len();
                if n == 0 {
                    return false;
                }
                let warmup = ((warmup_frac * n as f32).round() as usize).max(1).min(n);
                iteration < warmup
            }
        }
    }

    /// Update EMA stats for `class_idx` from a slice of [`GradientPair`]s,
    /// reusing an internal scratch buffer to avoid a per-round heap allocation.
    pub fn update_ema_from_gradient_pairs(&mut self, pairs: &[GradientPair], class_idx: usize) {
        self.grad_scratch.clear();
        self.grad_scratch.extend(pairs.iter().map(|g| g.grad));
        self.ema_stats[class_idx].update(&self.grad_scratch);
    }
}

/// Resolve a [`LrSchedule`] into a per-iteration learning-rate vector.
///
/// `Constant`: returns `vec![base_lr; n]`.
/// `WarmupCosine { warmup_frac }`: linear warmup over `round(warmup_frac * n)`
/// rounds (clamped to `[1, n]`), then half-cosine decay from `base_lr` toward a
/// floor of `base_lr * 0.01` over the remaining rounds.
pub fn resolve_lr_schedule(schedule: LrSchedule, n_estimators: u32, base_lr: f32) -> Vec<f32> {
    let n = n_estimators as usize;
    if n == 0 {
        return Vec::new();
    }
    match schedule {
        LrSchedule::Constant => vec![base_lr; n],
        LrSchedule::WarmupCosine { warmup_frac } => {
            let warmup = ((warmup_frac * n as f32).round() as usize).max(1).min(n);
            let mut lrs = Vec::with_capacity(n);
            for i in 0..warmup {
                lrs.push(base_lr * (i as f32 + 1.0) / warmup as f32);
            }
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

/// Per-round morph-mode context handed to `build_tree_*` so they can:
/// 1. dispatch split finding via `best_split_morph`,
/// 2. source the per-iteration learning rate from the LR schedule,
/// 3. apply depth-penalty + iteration-shrinkage multipliers to leaf values.
///
/// `None` selects the legacy (byte-identical) non-morph training path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MorphTreeContext<'a> {
    pub(crate) state: &'a MorphState,
    pub(crate) iteration: u32,
    pub(crate) total_iterations: u32,
    pub(crate) class_idx: usize,
    /// Resolved learning rate for this iteration (replaces `params.learning_rate`).
    pub(crate) lr: f32,
}
