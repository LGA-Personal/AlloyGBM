use alloygbm_core::GradientEmaStats;

use crate::TrainedStump;

/// Initial model state for warm-starting (continuing training from a previous model).
#[derive(Debug, Clone)]
pub struct WarmStartState {
    /// Baseline prediction (initial bias) from the original model.
    pub baseline_prediction: f32,
    /// Previously trained tree stumps.
    pub stumps: Vec<TrainedStump>,
    /// Number of rounds already completed in the initial model.
    pub initial_rounds_completed: usize,
    /// MorphBoost EMA snapshot from the previous fit (v0.7.3+).  When
    /// `Some` and the current fit also uses `training_mode="morph"`,
    /// the engine seeds `MorphState::ema_stats` from this snapshot so a
    /// resumed `N + M`-round model matches a fresh `N + M`-round fit.
    /// Empty / missing → EMA starts cold (legacy v0.7.1/v0.7.2 behaviour).
    pub initial_ema_stats: Option<Vec<GradientEmaStats>>,
    /// v0.10.0+: When the prior fit used DART, the per-stump `tree_weight`
    /// array (length = `stumps.len()`). `None` for non-DART warm-starts.
    /// On a DART warm-start continuation, the engine seeds
    /// `dart_state.tree_weights` from this snapshot so prior-tree dropouts
    /// during new rounds use the correct accumulated weights. Historical
    /// `dropped_per_round` arrays do *not* round-trip — new rounds start
    /// fresh dropout bookkeeping going forward.
    pub initial_dart_tree_weights: Option<Vec<f32>>,
}

/// State needed to continue multiclass training from a prior model.
pub struct MultiClassWarmStartState {
    pub baseline_predictions: Vec<f32>,
    pub class_stumps: Vec<Vec<TrainedStump>>,
    pub initial_rounds_completed: usize,
    /// MorphBoost EMA snapshot from the previous fit (v0.7.3+).  See
    /// `WarmStartState::initial_ema_stats`.
    pub initial_ema_stats: Option<Vec<GradientEmaStats>>,
    /// v0.10.1+: per-tree weights for multiclass DART warm-start.  Flat
    /// layout `[round 0 class 0, round 0 class 1, ..., round 0 class
    /// K-1, round 1 class 0, ...]` — round-major × class-k.  Length
    /// must equal `initial_rounds_completed * K`.  `None` means the
    /// prior fit was not multiclass DART; the engine falls back to a
    /// fresh DART state in that case.
    pub initial_dart_tree_weights: Option<Vec<f32>>,
}
