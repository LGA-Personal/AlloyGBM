//! Trainer module — gradient-boosting iteration controller.
//!
//! This module collects the trainer-internal helpers; the `Trainer` struct
//! itself still lives in `crate::lib` and will move in a subsequent task.

mod interaction;
mod policy;
mod tree_build;
mod validate;

pub(crate) use interaction::InteractionConstraintIndex;
pub(crate) use policy::split_selection_options_for_training;
#[cfg(test)]
pub(crate) use policy::should_apply_auto_split_l2;
pub(crate) use tree_build::{
    LEAF_EPSILON, apply_single_categorical_target_encoding, build_tree_leaf_wise,
    build_tree_level_wise, validate_iteration_controls,
};
#[cfg(test)]
pub(crate) use tree_build::{subtract_histogram_bundle, subtract_histogram_bundle_into};
pub(crate) use validate::{
    binned_feature_density, compute_feature_means_from_matrix, factor_split_context_for_node,
    gradient_neutralization_config, prepare_pre_target_training_dataset, target_variance,
    validate_gradient_pair_length, validate_gradient_pairs,
    validate_neutralization_fit_contract, validate_neutralization_fit_contract_for_support,
    validate_partition_cover, validate_training_alignment,
    validate_warm_start_neutralization_contract,
};
