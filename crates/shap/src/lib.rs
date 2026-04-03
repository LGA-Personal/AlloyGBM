use alloygbm_core::{
    ModelMetadata, deserialize_model_artifact_v1, format_required_section_mode_error,
    required_section_compatibility_report,
};
use alloygbm_engine::{ArtifactCompatibilityMode, TrainedModel, TrainedStump};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

const TREE_NODE_STRIDE: u32 = 1 << 20;
const ADDITIVITY_TOLERANCE: f32 = 1e-5;
const MAX_EXACT_SPLIT_FEATURES: usize = 25;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapError {
    InvalidInput(String),
    ContractViolation(String),
}

impl Display for ShapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::ContractViolation(message) => write!(f, "contract violation: {message}"),
        }
    }
}

impl Error for ShapError {}

pub type ShapResult<T> = Result<T, ShapError>;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapExplanationBatch {
    pub expected_value: f32,
    pub values: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct ArtifactShapContext {
    feature_names: Vec<String>,
    model: TrainedModel,
}

#[derive(Debug)]
struct ModelStructure<'a> {
    tree_root_ids: Vec<u32>,
    nodes_by_tree_local_id: HashMap<u64, &'a TrainedStump>,
    split_features: Vec<usize>,
    split_feature_bit_positions: Vec<Option<u8>>,
}

pub fn explain_rows_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<ShapExplanationBatch> {
    let context = load_artifact_context(artifact_bytes)?;
    explain_rows_from_model(&context.model, rows)
}

pub fn global_importance_from_shap_values(
    feature_names: &[String],
    shap_values: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if shap_values.is_empty() {
        return Err(ShapError::InvalidInput(
            "shap_values cannot be empty".to_string(),
        ));
    }

    let feature_count = feature_names.len();
    let mut contribution_sums = vec![0.0_f32; feature_count];
    for (row_index, row_values) in shap_values.iter().enumerate() {
        if row_values.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row_values.len()
            )));
        }
        for (feature_index, value) in row_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} contribution must be finite"
                )));
            }
            contribution_sums[feature_index] += value.abs();
        }
    }

    let row_count = shap_values.len() as f32;
    let mut global_importance = feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            (
                feature_name.clone(),
                contribution_sums[feature_index] / row_count,
            )
        })
        .collect::<Vec<_>>();

    global_importance.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    Ok(global_importance)
}

pub fn global_importance_from_artifact_bytes(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> ShapResult<Vec<(String, f32)>> {
    let context = load_artifact_context(artifact_bytes)?;
    let explanation = explain_rows_from_model(&context.model, rows)?;
    global_importance_from_shap_values(&context.feature_names, &explanation.values)
}

// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `explain_rows_from_artifact_bytes` for artifact-backed explanations.
pub fn shap_values_stub(metadata: &ModelMetadata, rows: &[Vec<f32>]) -> ShapResult<Vec<Vec<f32>>> {
    let feature_count = metadata.feature_names.len();
    validate_rows(rows, feature_count)?;
    Ok(vec![vec![0.0; feature_count]; rows.len()])
}

// Legacy compatibility shim for the v0.0.1 placeholder API. Prefer
// `global_importance_from_shap_values`.
pub fn global_importance_stub(
    metadata: &ModelMetadata,
    feature_names: &[String],
) -> ShapResult<Vec<(String, f32)>> {
    if feature_names.is_empty() {
        return Err(ShapError::InvalidInput(
            "feature_names cannot be empty".to_string(),
        ));
    }
    if feature_names.len() != metadata.feature_names.len() {
        return Err(ShapError::InvalidInput(format!(
            "feature_names length {} does not match metadata feature count {}",
            feature_names.len(),
            metadata.feature_names.len()
        )));
    }

    Ok(feature_names
        .iter()
        .map(|name| (name.clone(), 0.0_f32))
        .collect())
}

fn load_artifact_context(artifact_bytes: &[u8]) -> ShapResult<ArtifactShapContext> {
    let parsed = deserialize_model_artifact_v1(artifact_bytes)
        .map_err(|error| ShapError::ContractViolation(error.to_string()))?;

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

fn explain_rows_from_model(
    model: &TrainedModel,
    rows: &[Vec<f32>],
) -> ShapResult<ShapExplanationBatch> {
    validate_rows(rows, model.feature_count)?;

    let model_structure = build_model_structure(model)?;
    let expected_value =
        expected_prediction_for_subset(model, rows[0].as_slice(), 0, &model_structure)?;

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let values_by_subset = compute_subset_expectations(model, row, &model_structure)?;
        let row_expected_value = values_by_subset[0];

        if (row_expected_value - expected_value).abs() > ADDITIVITY_TOLERANCE {
            return Err(ShapError::ContractViolation(format!(
                "row {row_index} expected value drift: {row_expected_value} vs baseline {expected_value}"
            )));
        }

        let contributions = shapley_values_for_row(
            model,
            row,
            &values_by_subset,
            &model_structure,
            row_index,
            expected_value,
        )?;
        row_contributions.push(contributions);
    }

    Ok(ShapExplanationBatch {
        expected_value,
        values: row_contributions,
    })
}

fn build_model_structure(model: &TrainedModel) -> ShapResult<ModelStructure<'_>> {
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

fn compute_subset_expectations(
    model: &TrainedModel,
    row: &[f32],
    model_structure: &ModelStructure<'_>,
) -> ShapResult<Vec<f32>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = 1_usize
        .checked_shl(split_feature_count as u32)
        .ok_or_else(|| ShapError::ContractViolation("subset count overflow".to_string()))?;

    let mut values_by_subset = Vec::with_capacity(subset_count);
    for subset_mask in 0..subset_count {
        let value =
            expected_prediction_for_subset(model, row, subset_mask as u64, model_structure)?;
        values_by_subset.push(value);
    }
    Ok(values_by_subset)
}

fn expected_prediction_for_subset(
    model: &TrainedModel,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
) -> ShapResult<f32> {
    let mut prediction = model.baseline_prediction;
    for tree_id in &model_structure.tree_root_ids {
        prediction += expected_subtree(*tree_id, 0, row, subset_mask, model_structure)?;
    }
    Ok(prediction)
}

fn expected_subtree(
    tree_id: u32,
    local_node_id: u32,
    row: &[f32],
    subset_mask: u64,
    model_structure: &ModelStructure<'_>,
) -> ShapResult<f32> {
    let node_key = tree_local_key(tree_id, local_node_id);
    let Some(stump) = model_structure.nodes_by_tree_local_id.get(&node_key) else {
        return Ok(0.0);
    };

    let split_feature_index = stump.split.feature_index as usize;
    if split_feature_index >= row.len() {
        return Err(ShapError::ContractViolation(format!(
            "split feature_index {} exceeds row feature length {}",
            stump.split.feature_index,
            row.len()
        )));
    }

    let threshold = stump.split.threshold_bin as f32;
    let left_child_local = left_child_local_id(local_node_id)?;
    let right_child_local = right_child_local_id(local_node_id)?;

    if let Some(bit_position) = model_structure.split_feature_bit_positions[split_feature_index] {
        let is_known = (subset_mask & (1_u64 << bit_position)) != 0;
        if is_known {
            let goes_left = row[split_feature_index] <= threshold;
            if goes_left {
                return Ok(stump.left_leaf_value
                    + expected_subtree(
                        tree_id,
                        left_child_local,
                        row,
                        subset_mask,
                        model_structure,
                    )?);
            }
            return Ok(stump.right_leaf_value
                + expected_subtree(
                    tree_id,
                    right_child_local,
                    row,
                    subset_mask,
                    model_structure,
                )?);
        }
    }

    let left_count = stump.split.left_stats.row_count as f32;
    let right_count = stump.split.right_stats.row_count as f32;
    let total_count = left_count + right_count;
    let left_probability = if total_count > 0.0 {
        left_count / total_count
    } else {
        0.5
    };
    let right_probability = 1.0 - left_probability;

    let left_expected = stump.left_leaf_value
        + expected_subtree(tree_id, left_child_local, row, subset_mask, model_structure)?;
    let right_expected = stump.right_leaf_value
        + expected_subtree(
            tree_id,
            right_child_local,
            row,
            subset_mask,
            model_structure,
        )?;

    Ok(left_probability * left_expected + right_probability * right_expected)
}

fn shapley_values_for_row(
    model: &TrainedModel,
    row: &[f32],
    values_by_subset: &[f32],
    model_structure: &ModelStructure<'_>,
    row_index: usize,
    expected_value: f32,
) -> ShapResult<Vec<f32>> {
    let split_feature_count = model_structure.split_features.len();
    let subset_count = values_by_subset.len();

    let mut contributions = vec![0.0_f32; model.feature_count];
    if split_feature_count == 0 {
        verify_additivity(model, row, &contributions, row_index, expected_value)?;
        return Ok(contributions);
    }

    let factorials = factorial_table(split_feature_count);
    let total_factorial = factorials[split_feature_count];

    for (feature_bit_position, &feature_index) in model_structure.split_features.iter().enumerate()
    {
        let feature_bit = 1_u64 << feature_bit_position;
        let mut phi = 0.0_f64;

        for subset_mask in 0..subset_count {
            let subset_mask_u64 = subset_mask as u64;
            if (subset_mask_u64 & feature_bit) != 0 {
                continue;
            }

            let with_feature_mask = subset_mask_u64 | feature_bit;
            let subset_size = subset_mask_u64.count_ones() as usize;
            let weight = factorials[subset_size]
                * factorials[split_feature_count - subset_size - 1]
                / total_factorial;

            let marginal =
                values_by_subset[with_feature_mask as usize] - values_by_subset[subset_mask];
            phi += weight * marginal as f64;
        }

        contributions[feature_index] = phi as f32;
    }

    verify_additivity(model, row, &contributions, row_index, expected_value)?;
    Ok(contributions)
}

fn verify_additivity(
    model: &TrainedModel,
    row: &[f32],
    contributions: &[f32],
    row_index: usize,
    expected_value: f32,
) -> ShapResult<()> {
    let predicted = model
        .predict_row(row)
        .map_err(|error| ShapError::ContractViolation(error.to_string()))?;
    let reconstructed = expected_value + contributions.iter().sum::<f32>();
    if (predicted - reconstructed).abs() > ADDITIVITY_TOLERANCE {
        return Err(ShapError::ContractViolation(format!(
            "row {row_index} additivity check failed: predicted={predicted}, reconstructed={reconstructed}, tolerance={ADDITIVITY_TOLERANCE}"
        )));
    }
    Ok(())
}

fn factorial_table(max_value: usize) -> Vec<f64> {
    let mut factorials = vec![1.0_f64; max_value + 1];
    for value in 1..=max_value {
        factorials[value] = factorials[value - 1] * value as f64;
    }
    factorials
}

fn tree_local_key(tree_id: u32, local_node_id: u32) -> u64 {
    ((tree_id as u64) << 32) | local_node_id as u64
}

fn left_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ShapError::ContractViolation("left child node id overflow".to_string()))
}

fn right_child_local_id(local_node_id: u32) -> ShapResult<u32> {
    local_node_id
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| ShapError::ContractViolation("right child node id overflow".to_string()))
}

fn validate_rows(rows: &[Vec<f32>], feature_count: usize) -> ShapResult<()> {
    if feature_count == 0 {
        return Err(ShapError::InvalidInput(
            "model feature_count must be greater than 0".to_string(),
        ));
    }
    if rows.is_empty() {
        return Err(ShapError::InvalidInput("rows cannot be empty".to_string()));
    }

    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != feature_count {
            return Err(ShapError::InvalidInput(format!(
                "row {row_index} feature count {} does not match expected {feature_count}",
                row.len()
            )));
        }
        for (feature_index, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(ShapError::InvalidInput(format!(
                    "row {row_index} feature {feature_index} must be finite"
                )));
            }
        }
    }

    Ok(())
}

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{
        Device, ModelMetadata, ModelSectionKind, NodeStats, SplitCandidate,
        serialize_model_artifact_v1,
    };
    use alloygbm_engine::TrainedModel;
    use alloygbm_predictor::Predictor;

    fn sample_metadata(feature_names: &[&str]) -> ModelMetadata {
        ModelMetadata {
            format_version: 1,
            feature_names: feature_names
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            trained_device: Device::Cpu,
        }
    }

    fn split(node_id: u32, feature_index: u32, threshold_bin: u16) -> SplitCandidate {
        SplitCandidate {
            node_id,
            feature_index,
            threshold_bin,
            gain: 1.0,
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 1.0,
                row_count: 1,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 1.0,
                row_count: 1,
            },
        }
    }

    fn fixture_model() -> TrainedModel {
        TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: split(0, 0, 1),
                    left_leaf_value: 1.0,
                    right_leaf_value: 2.0,
                },
                TrainedStump {
                    split: split(1, 1, 0),
                    left_leaf_value: 0.1,
                    right_leaf_value: 0.2,
                },
                TrainedStump {
                    split: split(2, 1, 1),
                    left_leaf_value: 0.3,
                    right_leaf_value: 0.4,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
        }
    }

    fn fixture_model_with_unused_feature() -> TrainedModel {
        TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 3,
            stumps: fixture_model().stumps,
            categorical_state: None,
            node_debug_stats: None,
        }
    }

    fn fixture_rows() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.0],
            vec![0.0, 2.0],
            vec![3.0, 0.0],
            vec![3.0, 2.0],
        ]
    }

    fn fixture_trees_payload() -> Vec<u8> {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&artifact).expect("artifact parses");
        parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists")
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= ADDITIVITY_TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn explain_rows_from_artifact_rejects_empty_rows() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_feature_count_mismatch() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[vec![0.0]]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_non_finite_features() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let result = explain_rows_from_artifact_bytes(&artifact, &[vec![f32::NAN, 0.0]]);
        assert!(matches!(result, Err(ShapError::InvalidInput(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_incompatible_required_sections() {
        let layout_payload = {
            let strict_artifact = fixture_model()
                .to_artifact_bytes()
                .expect("artifact serializes");
            let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
            parsed
                .sections
                .iter()
                .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
                .map(|section| section.payload.clone())
                .expect("predictor layout payload exists")
        };

        let incompatible_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[(ModelSectionKind::PredictorLayout, layout_payload)],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&incompatible_artifact, &[vec![0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_accepts_legacy_trees_only_artifact() {
        let legacy_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[(ModelSectionKind::Trees, fixture_trees_payload())],
        )
        .expect("artifact serializes");

        let explanation = explain_rows_from_artifact_bytes(&legacy_artifact, &fixture_rows())
            .expect("legacy artifact explains");
        assert_close(explanation.expected_value, 2.25);
        assert_eq!(explanation.values.len(), 4);
        assert_eq!(explanation.values[0].len(), 2);
    }

    #[test]
    fn explain_rows_from_artifact_rejects_duplicate_trees_sections() {
        let trees_payload = fixture_trees_payload();
        let duplicate_trees_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1"]),
            &[
                (ModelSectionKind::Trees, trees_payload.clone()),
                (ModelSectionKind::Trees, trees_payload),
            ],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&duplicate_trees_artifact, &[vec![0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_rejects_metadata_feature_count_mismatch() {
        let mismatched_artifact = serialize_model_artifact_v1(
            &sample_metadata(&["f0", "f1", "f2"]),
            &[(ModelSectionKind::Trees, fixture_trees_payload())],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&mismatched_artifact, &[vec![0.0, 0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_computes_exact_expected_value_and_contributions() {
        let model = fixture_model();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = fixture_rows();

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        assert_close(explanation.expected_value, 2.25);
        assert_eq!(explanation.values.len(), rows.len());
        for row_values in &explanation.values {
            assert_eq!(row_values.len(), model.feature_count);
        }

        let expected_values = [
            vec![-0.6, -0.05],
            vec![-0.6, 0.05],
            vec![0.6, -0.05],
            vec![0.6, 0.05],
        ];

        for (actual_row, expected_row) in explanation.values.iter().zip(expected_values.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_close(*actual, *expected);
            }
        }

        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn explain_rows_from_artifact_matches_predictor_predictions() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor loads");
        let rows = fixture_rows();

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        for (row_index, row) in rows.iter().enumerate() {
            let predicted = predictor.predict_row(row).expect("predicts");
            let reconstructed =
                explanation.expected_value + explanation.values[row_index].iter().sum::<f32>();
            assert_close(reconstructed, predicted);
        }
    }

    #[test]
    fn explain_rows_from_artifact_assigns_zero_to_unused_features() {
        let model = fixture_model_with_unused_feature();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = vec![vec![0.0, 0.0, 5.0], vec![3.0, 2.0, 9.0]];

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        assert_eq!(explanation.values[0].len(), 3);
        assert_close(explanation.values[0][2], 0.0);
        assert_close(explanation.values[1][2], 0.0);
    }

    #[test]
    fn global_importance_aggregates_mean_absolute_contribution() {
        let feature_names = vec!["f0".to_string(), "f1".to_string()];
        let shap_values = vec![
            vec![-0.6, -0.05],
            vec![-0.6, 0.05],
            vec![0.6, -0.05],
            vec![0.6, 0.05],
        ];

        let global = global_importance_from_shap_values(&feature_names, &shap_values)
            .expect("global importance computes");
        assert_close(global[0].1, 0.6);
        assert_close(global[1].1, 0.05);
        assert_eq!(global[0].0, "f0");
        assert_eq!(global[1].0, "f1");
    }

    #[test]
    fn global_importance_from_artifact_uses_metadata_feature_names() {
        let artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let global = global_importance_from_artifact_bytes(&artifact, &fixture_rows())
            .expect("global computes");

        assert_eq!(global.len(), 2);
        assert_eq!(global[0].0, "f0");
        assert_eq!(global[1].0, "f1");
    }

    #[test]
    fn global_importance_breaks_ties_by_feature_name() {
        let feature_names = vec!["zeta".to_string(), "alpha".to_string(), "beta".to_string()];
        let shap_values = vec![vec![1.0, -1.0, 0.0], vec![-1.0, 1.0, 0.0]];

        let global = global_importance_from_shap_values(&feature_names, &shap_values)
            .expect("global importance computes");

        assert_eq!(global.len(), 3);
        assert_eq!(global[0].0, "alpha");
        assert_eq!(global[1].0, "zeta");
        assert_eq!(global[2].0, "beta");
        assert_close(global[0].1, 1.0);
        assert_close(global[1].1, 1.0);
        assert_close(global[2].1, 0.0);
    }

    #[test]
    fn legacy_stub_helpers_return_deterministic_outputs() {
        let metadata = sample_metadata(&["f0", "f1"]);
        let rows = fixture_rows();
        let shap_values = shap_values_stub(&metadata, &rows).expect("stub values compute");
        assert_eq!(shap_values.len(), rows.len());
        assert_eq!(shap_values[0], vec![0.0, 0.0]);

        let global = global_importance_stub(&metadata, &metadata.feature_names)
            .expect("stub global computes");
        assert_eq!(
            global,
            vec![("f0".to_string(), 0.0), ("f1".to_string(), 0.0)]
        );
    }
}
