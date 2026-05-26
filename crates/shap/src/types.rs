use alloygbm_core::{
    deserialize_model_artifact_v1, format_required_section_mode_error,
    required_section_compatibility_report,
};
use alloygbm_engine::{ArtifactCompatibilityMode, TrainedModel, TrainedStump};
use std::collections::HashMap;

use crate::binning::MAX_EXACT_SPLIT_FEATURES;
use crate::brute_force::{decode_tree_node_id, tree_local_key};
use crate::error::{ShapError, ShapResult};

#[derive(Debug, Clone, PartialEq)]
pub struct ShapExplanationBatch {
    pub expected_value: f32,
    pub values: Vec<Vec<f32>>,
}

/// Pairwise SHAP interaction values returned by `explain_interactions_from_artifact_bytes`.
///
/// `values[row][i][j]` is the SHAP interaction contribution of feature pair `(i, j)`
/// to the prediction of row `row`.  The matrix is symmetric: `values[r][i][j] == values[r][j][i]`.
/// The diagonal `values[r][i][i]` is the "main effect" of feature `i` after removing all
/// interactions; the row-marginal `Σ_j values[r][i][j]` recovers the per-feature SHAP
/// value from `ShapExplanationBatch`, and the full sum `Σ_i Σ_j values[r][i][j] + expected_value`
/// recovers the prediction (additivity).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapInteractionBatch {
    pub expected_value: f32,
    pub values: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactShapContext {
    pub(crate) feature_names: Vec<String>,
    pub(crate) model: TrainedModel,
}

#[derive(Debug)]
pub(crate) struct ModelStructure<'a> {
    pub(crate) tree_root_ids: Vec<u32>,
    pub(crate) nodes_by_tree_local_id: HashMap<u64, &'a TrainedStump>,
    pub(crate) split_features: Vec<usize>,
    pub(crate) split_feature_bit_positions: Vec<Option<u8>>,
}

pub(crate) fn load_artifact_context(artifact_bytes: &[u8]) -> ShapResult<ArtifactShapContext> {
    let parsed = deserialize_model_artifact_v1(artifact_bytes)
        .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

    if parsed.contract.metadata.num_classes.is_some() {
        return Err(ShapError::ContractViolation(
            "SHAP values are not yet supported for multi-class models".to_string(),
        ));
    }

    let compatibility_report = required_section_compatibility_report(&parsed.sections);
    if !compatibility_report.legacy_compatible {
        return Err(ShapError::ContractViolation(
            format_required_section_mode_error(compatibility_report, true),
        ));
    }

    let model = TrainedModel::from_artifact_bytes_with_mode(
        artifact_bytes,
        ArtifactCompatibilityMode::AllowLegacyTreesOnly,
    )
    .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

    if parsed.contract.metadata.feature_names.len() != model.feature_count {
        return Err(ShapError::ContractViolation(format!(
            "metadata feature count {} does not match model feature count {}",
            parsed.contract.metadata.feature_names.len(),
            model.feature_count
        )));
    }

    Ok(ArtifactShapContext {
        feature_names: parsed.contract.metadata.feature_names,
        model,
    })
}

pub(crate) fn build_model_structure(model: &TrainedModel) -> ShapResult<ModelStructure<'_>> {
    let mut nodes_by_tree_local_id = HashMap::new();
    let mut tree_root_ids = Vec::new();
    let mut split_features = Vec::new();

    for stump in &model.stumps {
        let (tree_id, local_node_id) = decode_tree_node_id(stump.split.node_id);
        let node_key = tree_local_key(tree_id, local_node_id);
        nodes_by_tree_local_id.insert(node_key, stump);
        if local_node_id == 0 {
            tree_root_ids.push(tree_id);
        }

        let feature_index = stump.split.feature_index as usize;
        if feature_index >= model.feature_count {
            return Err(ShapError::ContractViolation(format!(
                "stump feature_index {} exceeds model feature_count {}",
                stump.split.feature_index, model.feature_count
            )));
        }
        split_features.push(feature_index);
    }

    tree_root_ids.sort_unstable();
    tree_root_ids.dedup();
    split_features.sort_unstable();
    split_features.dedup();

    if split_features.len() > MAX_EXACT_SPLIT_FEATURES {
        return Err(ShapError::ContractViolation(format!(
            "exact SHAP supports at most {MAX_EXACT_SPLIT_FEATURES} distinct split features per model (found {})",
            split_features.len()
        )));
    }

    let mut split_feature_bit_positions = vec![None; model.feature_count];
    for (bit_position, feature_index) in split_features.iter().enumerate() {
        split_feature_bit_positions[*feature_index] = Some(bit_position as u8);
    }

    Ok(ModelStructure {
        tree_root_ids,
        nodes_by_tree_local_id,
        split_features,
        split_feature_bit_positions,
    })
}
