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
    pub(crate) models: Vec<TrainedModel>,
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

    let compatibility_report = required_section_compatibility_report(&parsed.sections);
    let has_multi_output = parsed
        .sections
        .iter()
        .any(|s| s.descriptor.kind == alloygbm_core::ModelSectionKind::MultiOutputLeafValues);
    if parsed.contract.metadata.num_classes.is_none()
        && !compatibility_report.legacy_compatible
        && !has_multi_output
    {
        return Err(ShapError::ContractViolation(
            format_required_section_mode_error(compatibility_report, true),
        ));
    }

    let mut models = Vec::new();

    if parsed.contract.metadata.num_classes.is_some() {
        let mc_model = alloygbm_engine::MultiClassTrainedModel::from_artifact_bytes(artifact_bytes)
            .map_err(|e| ShapError::ContractViolation(e.to_string()))?;

        if parsed.contract.metadata.feature_names.len() != mc_model.feature_count {
            return Err(ShapError::ContractViolation(format!(
                "metadata feature count {} does not match model feature count {}",
                parsed.contract.metadata.feature_names.len(),
                mc_model.feature_count
            )));
        }

        // Decode optional FeatureBaseline section.
        let feature_baseline =
            alloygbm_core::decode_optional_feature_baseline_section(&parsed.sections)
                .map_err(|error| ShapError::ContractViolation(error.to_string()))?
                .map(|payload| payload.feature_means)
                .filter(|means| means.len() == mc_model.feature_count);

        // Decode optional NativeCategoricalSplits section.
        let mut native_categorical_feature_indices = Vec::new();
        let mut class_stumps = mc_model.class_stumps;

        if let Some(cat_payload) =
            alloygbm_core::decode_optional_native_categorical_splits_section(&parsed.sections)
                .map_err(|error| ShapError::ContractViolation(error.to_string()))?
        {
            native_categorical_feature_indices = cat_payload.native_categorical_feature_indices;
            let stump_bitsets: std::collections::HashMap<u32, Vec<u8>> =
                cat_payload.stump_bitsets.into_iter().collect();
            let mut global_idx = 0usize;
            for stumps in &mut class_stumps {
                for stump in stumps.iter_mut() {
                    if let Some(bitset) = stump_bitsets.get(&(global_idx as u32)) {
                        stump.split.categorical_bitset = Some(bitset.clone());
                    }
                    global_idx += 1;
                }
            }
        }

        for (k, stumps) in class_stumps.into_iter().enumerate() {
            models.push(TrainedModel {
                baseline_prediction: mc_model.baseline_predictions[k],
                feature_count: mc_model.feature_count,
                stumps,
                categorical_state: mc_model.categorical_state.clone(),
                node_debug_stats: None,
                objective: mc_model.objective.clone(),
                native_categorical_feature_indices: native_categorical_feature_indices.clone(),
                morph_metadata: mc_model.morph_metadata.clone(),
                dro_metadata: mc_model.dro_metadata.clone(),
                feature_baseline: feature_baseline.clone(),
                neutralization_metadata: None,
            });
        }
    } else {
        let base_model = TrainedModel::from_artifact_bytes_with_mode(
            artifact_bytes,
            ArtifactCompatibilityMode::AllowLegacyTreesOnly,
        )
        .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

        if parsed.contract.metadata.feature_names.len() != base_model.feature_count {
            return Err(ShapError::ContractViolation(format!(
                "metadata feature count {} does not match model feature count {}",
                parsed.contract.metadata.feature_names.len(),
                base_model.feature_count
            )));
        }

        let mut baselines = None;
        if parsed
            .contract
            .metadata
            .objective
            .starts_with("joint_multi_output:")
            && parsed.contract.metadata.objective.contains("|baselines=")
        {
            let pos = parsed
                .contract
                .metadata
                .objective
                .find("|baselines=")
                .unwrap();
            let baselines_str = &parsed.contract.metadata.objective[pos + "|baselines=".len()..];
            let parsed_baselines: Result<Vec<f32>, _> =
                baselines_str.split(',').map(|s| s.parse::<f32>()).collect();
            match parsed_baselines {
                Ok(b) => baselines = Some(b),
                Err(e) => {
                    return Err(ShapError::ContractViolation(format!(
                        "Failed to parse |baselines= field in objective metadata string: {e}"
                    )));
                }
            }
        }

        if let Some((left_vec, _)) = base_model
            .stumps
            .first()
            .and_then(|s| s.multi_output_leaf_values.as_ref())
        {
            let n_outputs = left_vec.len();
            for k in 0..n_outputs {
                let mut k_model = base_model.clone();
                // Map tree_id -> Map local_node_id -> (left_val, right_val)
                let mut stump_vals: HashMap<u32, HashMap<u32, (f32, f32)>> = HashMap::new();
                for stump in &k_model.stumps {
                    if let Some((left_mo, right_mo)) = &stump.multi_output_leaf_values {
                        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
                        stump_vals
                            .entry(tree_id)
                            .or_default()
                            .insert(local_id, (left_mo[k], right_mo[k]));
                    }
                }

                for stump in &mut k_model.stumps {
                    if let Some((left_mo, right_mo)) = &stump.multi_output_leaf_values {
                        let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
                        let parent_val = if local_id == 0 {
                            0.0
                        } else {
                            let parent_id = (local_id - 1) / 2;
                            if let Some(tree_map) = stump_vals.get(&tree_id) {
                                if let Some(&(p_left, p_right)) = tree_map.get(&parent_id) {
                                    if local_id % 2 == 1 { p_left } else { p_right }
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            }
                        };
                        stump.left_leaf_value =
                            alloygbm_core::LeafValue::Scalar(left_mo[k] - parent_val);
                        stump.right_leaf_value =
                            alloygbm_core::LeafValue::Scalar(right_mo[k] - parent_val);
                    }
                    stump.multi_output_leaf_values = None;
                }
                if let Some(&b) = baselines.as_ref().and_then(|b| b.get(k)) {
                    k_model.baseline_prediction = b;
                }
                models.push(k_model);
            }
        } else {
            models.push(base_model);
        }
    }

    Ok(ArtifactShapContext {
        feature_names: parsed.contract.metadata.feature_names,
        models,
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
