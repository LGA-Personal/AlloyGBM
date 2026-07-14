/// Per-iteration learning rate schedule for MorphBoost training.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LrSchedule {
    /// Use constant learning rate for all rounds.
    #[default]
    Constant,
    /// Linear warmup from 0 → learning_rate over `warmup_frac * n_estimators`
    /// rounds, then half-cosine decay to `learning_rate * 0.01` over remaining rounds.
    WarmupCosine { warmup_frac: f32 },
}

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
    /// Maximum blend weight on the normalized information component.
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
            info_score_weight: 0.1,
            depth_penalty_base: 0.9,
            balance_penalty: true,
            lr_schedule: LrSchedule::Constant,
        }
    }
}

/// Per-round constants for morph gain computation. Compute once per round (not per bin).
/// Eliminates redundant ramp, blend-coefficient, and warmup-branch
/// computation in the inner per-bin gain loop.
#[derive(Debug, Clone, Copy)]
pub struct MorphPrecomputed {
    pub in_warmup: bool,
    /// Horizon-scaled information ramp — only meaningful post-warmup.
    pub morph_weight: f32,
    /// Complement of `info_score_coeff` (post-warmup; 1.0 in warmup)
    pub gradient_score_coeff: f32,
    /// `info_score_weight * morph_weight` (post-warmup; 0.0 in warmup)
    pub info_score_coeff: f32,
    /// Mirrors `cfg.balance_penalty` for fast access without dereferencing config
    pub balance_penalty: bool,
    /// True if `info_score_coeff` is below an epsilon — skip `info_gain` entirely
    pub info_score_negligible: bool,
}

impl MorphPrecomputed {
    /// Builds constants for one training round across the complete fit horizon.
    ///
    /// The ramp reaches `tanh(1)` one third into the fit and approaches one by
    /// the final round. This keeps its shape proportional to `n_estimators`
    /// instead of saturating after a fixed number of rounds.
    pub fn for_iteration(iteration: u32, total_iterations: u32, cfg: &MorphConfig) -> Self {
        let in_warmup = iteration < cfg.morph_warmup_iters;
        if in_warmup {
            return Self {
                in_warmup: true,
                morph_weight: 0.0,
                gradient_score_coeff: 1.0,
                info_score_coeff: 0.0,
                balance_penalty: cfg.balance_penalty,
                info_score_negligible: true,
            };
        }
        let progress = iteration as f32 / total_iterations.max(1) as f32;
        let morph_weight = (3.0 * progress).tanh();
        let info_score_coeff = cfg.info_score_weight * morph_weight;
        Self {
            in_warmup: false,
            morph_weight,
            gradient_score_coeff: 1.0 - info_score_coeff,
            info_score_coeff,
            balance_penalty: cfg.balance_penalty,
            info_score_negligible: info_score_coeff.abs() < 1e-6,
        }
    }
}

/// Top-level training profile selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingMode {
    /// Auto-policy with dataset-aware heuristics (current default).
    #[default]
    Auto,
    /// Raw user-supplied parameters with no overrides.
    Manual,
    /// MorphBoost-inspired adaptive training profile.
    Morph,
}

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
    ///
    /// Non-finite inputs (NaN, Inf) are silently skipped so the running stats
    /// don't get permanently poisoned by transient numerical issues in
    /// upstream gradient computation.
    ///
    /// Note: variance is computed with population divisor (n), not sample
    /// divisor (n-1). This is intentional for EMA smoothing where n is large.
    pub fn update(&mut self, gradients: &[f32]) {
        if gradients.is_empty() {
            return;
        }
        let n = gradients.len() as f32;
        // SIMD-vectorized single-pass computation: sum + sum-of-squares.
        // var = E[X²] - E[X]² (algebraically equivalent to the 2-pass form,
        // numerically slightly less stable but fine for gradient stats).
        let sum = crate::simd::sum_f32(gradients);
        let sumsq = crate::simd::sum_squares_f32(gradients);
        let mean = sum / n;
        if !mean.is_finite() {
            return;
        }
        // Clamp to 0 to guard against tiny FP negatives from cancellation.
        let var = (sumsq / n - mean * mean).max(0.0);
        if !var.is_finite() {
            return;
        }
        let std = var.sqrt();
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * mean;
        self.std = (1.0 - self.alpha) * self.std + self.alpha * std;
    }

    #[cfg(test)]
    pub(crate) fn update_two_pass_legacy(&mut self, gradients: &[f32]) {
        // Preserved only for parity testing against the new single-pass form.
        if gradients.is_empty() {
            return;
        }
        let n = gradients.len() as f32;
        let mean: f32 = gradients.iter().sum::<f32>() / n;
        if !mean.is_finite() {
            return;
        }
        let var: f32 = gradients
            .iter()
            .map(|g| (g - mean) * (g - mean))
            .sum::<f32>()
            / n;
        if !var.is_finite() {
            return;
        }
        let std = var.sqrt();
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * mean;
        self.std = (1.0 - self.alpha) * self.std + self.alpha * std;
    }
}
