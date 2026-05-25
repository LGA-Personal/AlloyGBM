//! Runtime experiment toggles driven by `ALLOYGBM_EXPERIMENT_*` environment
//! variables.
//!
//! These helpers gate experimental knobs (split regularization, leaf
//! refinement, manual policy override, etc.) without surfacing them as user
//! facing kwargs. They are intentionally kept in a small, self-contained
//! module so the rest of the engine can call into them without pulling in
//! lib.rs internals.

use alloygbm_core::MISSING_BIN_U8;

use crate::SplitSelectionOptions;
use crate::error::{EngineError, EngineResult};

pub(crate) const SPLIT_L2_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_SPLIT_L2";
pub(crate) const SPLIT_L1_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_SPLIT_L1";
pub(crate) const MIN_CHILD_HESS_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS";
pub(crate) const SPLIT_MIN_LEAF_MAGNITUDE_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_SPLIT_MIN_LEAF_MAGNITUDE";
pub(crate) const FORCE_MANUAL_POLICY_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_FORCE_MANUAL_POLICY";
pub(crate) const ENABLE_LEAF_REFINEMENT_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_ENABLE_LEAF_REFINEMENT";

pub(crate) fn split_selection_options_from_env() -> EngineResult<SplitSelectionOptions> {
    Ok(SplitSelectionOptions {
        l2_lambda: parse_nonnegative_env_f32(SPLIT_L2_ENV_VAR)?,
        l1_alpha: parse_nonnegative_env_f32(SPLIT_L1_ENV_VAR)?,
        min_child_hessian: parse_nonnegative_env_f32(MIN_CHILD_HESS_ENV_VAR)?,
        min_leaf_magnitude: parse_nonnegative_env_f32(SPLIT_MIN_LEAF_MAGNITUDE_ENV_VAR)?,
        dro_config: None,
        missing_bin_index: MISSING_BIN_U8 as usize,
    })
}

pub(crate) fn experiment_force_manual_policy_enabled() -> bool {
    env_toggle_enabled(FORCE_MANUAL_POLICY_ENV_VAR)
}

pub(crate) fn experiment_leaf_refinement_enabled() -> bool {
    env_toggle_enabled(ENABLE_LEAF_REFINEMENT_ENV_VAR)
}

fn env_toggle_enabled(env_name: &str) -> bool {
    match std::env::var(env_name) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn parse_nonnegative_env_f32(env_name: &str) -> EngineResult<f32> {
    match std::env::var(env_name) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Ok(0.0);
            }
            let parsed = trimmed.parse::<f32>().map_err(|_| {
                EngineError::InvalidConfig(format!(
                    "{env_name} must be a finite, non-negative f32 value"
                ))
            })?;
            if !parsed.is_finite() || parsed < 0.0 {
                return Err(EngineError::InvalidConfig(format!(
                    "{env_name} must be finite and >= 0"
                )));
            }
            Ok(parsed)
        }
        Err(_) => Ok(0.0),
    }
}

pub(crate) fn split_l2_env_is_configured() -> bool {
    std::env::var_os(SPLIT_L2_ENV_VAR).is_some()
}
