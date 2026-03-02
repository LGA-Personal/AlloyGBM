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

    let stumps_by_node = model
        .stumps
        .iter()
        .map(|stump| (stump.split.node_id, stump))
        .collect::<HashMap<_, _>>();

    let mut row_contributions = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let mut contributions = vec![0.0_f32; model.feature_count];
        for stump in &model.stumps {
            if !row_satisfies_stump_path_features(row, stump, &stumps_by_node)? {
                continue;
            }

            let feature_index = stump.split.feature_index as usize;
            if feature_index >= model.feature_count {
                return Err(ShapError::ContractViolation(format!(
                    "stump feature_index {} exceeds model feature_count {}",
                    stump.split.feature_index, model.feature_count
                )));
            }
            let threshold = stump.split.threshold_bin as f32;
            let contribution = if row[feature_index] <= threshold {
                stump.left_leaf_value
            } else {
                stump.right_leaf_value
            };
            contributions[feature_index] += contribution;
        }

        let predicted = model
            .predict_row(row)
            .map_err(|error| ShapError::ContractViolation(error.to_string()))?;
        let reconstructed = model.baseline_prediction + contributions.iter().sum::<f32>();
        if (predicted - reconstructed).abs() > ADDITIVITY_TOLERANCE {
            return Err(ShapError::ContractViolation(format!(
                "row {row_index} additivity check failed: predicted={predicted}, reconstructed={reconstructed}, tolerance={ADDITIVITY_TOLERANCE}"
            )));
        }

        row_contributions.push(contributions);
    }

    Ok(ShapExplanationBatch {
        expected_value: model.baseline_prediction,
        values: row_contributions,
    })
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

fn encode_tree_node_id(tree_index: u32, local_node_id: u32) -> ShapResult<u32> {
    if local_node_id >= TREE_NODE_STRIDE {
        return Err(ShapError::ContractViolation(format!(
            "local node_id {local_node_id} exceeds supported tree-node stride {TREE_NODE_STRIDE}"
        )));
    }

    tree_index
        .checked_mul(TREE_NODE_STRIDE)
        .and_then(|base| base.checked_add(local_node_id))
        .ok_or_else(|| {
            ShapError::ContractViolation("encoded tree node id overflowed u32 range".to_string())
        })
}

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

fn row_satisfies_stump_path_features(
    features: &[f32],
    stump: &TrainedStump,
    stumps_by_node: &HashMap<u32, &TrainedStump>,
) -> ShapResult<bool> {
    let (tree_id, mut local_node_id) = decode_tree_node_id(stump.split.node_id);
    while local_node_id > 0 {
        let parent_local = (local_node_id - 1) / 2;
        let parent_node_id = encode_tree_node_id(tree_id, parent_local)?;
        let Some(parent_stump) = stumps_by_node.get(&parent_node_id) else {
            return Ok(false);
        };

        let feature_index = parent_stump.split.feature_index as usize;
        if feature_index >= features.len() {
            return Err(ShapError::ContractViolation(format!(
                "split feature_index {} exceeds feature length {}",
                parent_stump.split.feature_index,
                features.len()
            )));
        }

        let went_left = features[feature_index] <= parent_stump.split.threshold_bin as f32;
        let expected_left = local_node_id == parent_local * 2 + 1;
        if went_left != expected_left {
            return Ok(false);
        }

        local_node_id = parent_local;
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{
        Device, ModelMetadata, ModelSectionKind, NodeStats, SplitCandidate,
        serialize_model_artifact_v1,
    };
    use alloygbm_engine::TrainedModel;

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            format_version: 1,
            feature_names: vec!["f0".to_string(), "f1".to_string()],
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
        let model = fixture_model();
        let strict_artifact = model.to_artifact_bytes().expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
        let layout_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
            .map(|section| section.payload.clone())
            .expect("predictor layout payload exists");

        let incompatible_artifact = serialize_model_artifact_v1(
            &sample_metadata(),
            &[(ModelSectionKind::PredictorLayout, layout_payload)],
        )
        .expect("artifact serializes");

        let result = explain_rows_from_artifact_bytes(&incompatible_artifact, &[vec![0.0, 0.0]]);
        assert!(matches!(result, Err(ShapError::ContractViolation(_))));
    }

    #[test]
    fn explain_rows_from_artifact_has_deterministic_shape_and_additivity() {
        let model = fixture_model();
        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let rows = fixture_rows();

        let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
        assert_eq!(explanation.expected_value, model.baseline_prediction);
        assert_eq!(explanation.values.len(), rows.len());
        for row_values in &explanation.values {
            assert_eq!(row_values.len(), model.feature_count);
        }

        let expected_contributions = vec![
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![2.0, 0.3],
            vec![2.0, 0.4],
        ];
        assert_eq!(explanation.values, expected_contributions);

        for (row, values) in rows.iter().zip(explanation.values.iter()) {
            let predicted = model.predict_row(row).expect("predicts");
            let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
            assert!((predicted - reconstructed).abs() <= ADDITIVITY_TOLERANCE);
        }
    }

    #[test]
    fn global_importance_aggregates_mean_absolute_contribution() {
        let feature_names = vec!["f0".to_string(), "f1".to_string()];
        let shap_values = vec![
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![2.0, 0.3],
            vec![2.0, 0.4],
        ];

        let global = global_importance_from_shap_values(&feature_names, &shap_values)
            .expect("global importance computes");
        assert_eq!(global[0], ("f0".to_string(), 1.5));
        assert_eq!(global[1], ("f1".to_string(), 0.25));
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
    fn legacy_stub_helpers_return_deterministic_outputs() {
        let metadata = sample_metadata();
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
