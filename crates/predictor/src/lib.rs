use alloygbm_core::{
    CoreError, MODEL_FORMAT_V1, ModelArtifactSection, ModelMetadata, ModelSectionKind,
    deserialize_model_artifact_v1,
};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictorError {
    InvalidInput(String),
    ContractViolation(String),
    Core(CoreError),
}

impl Display for PredictorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::ContractViolation(msg) => write!(f, "contract violation: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for PredictorError {}

impl From<CoreError> for PredictorError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

pub type PredictorResult<T> = Result<T, PredictorError>;

#[derive(Debug, Clone, PartialEq)]
struct PredictorStump {
    node_id: u32,
    feature_index: u32,
    threshold_bin: u16,
    left_leaf_value: f32,
    right_leaf_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PredictorLayoutPayload {
    feature_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    pub metadata: ModelMetadata,
    baseline_prediction: f32,
    stumps: Vec<PredictorStump>,
}

impl Predictor {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            baseline_prediction: 0.0,
            stumps: Vec::new(),
        }
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> PredictorResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(PredictorError::from)?;
        let metadata = parsed.contract.metadata;
        let metadata_feature_count = metadata.feature_names.len();
        let trees_section = required_single_section(&parsed.sections, ModelSectionKind::Trees)?;
        let predictor_layout = resolve_predictor_layout(&parsed.sections, metadata_feature_count)?;
        let (payload_feature_count, baseline_prediction, stumps) =
            decode_trained_model_payload(&trees_section.payload)?;

        if predictor_layout.feature_count != metadata_feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "predictor layout feature_count {} does not match metadata feature count {}",
                predictor_layout.feature_count, metadata_feature_count
            )));
        }
        if payload_feature_count != predictor_layout.feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "trees payload feature_count {} does not match predictor layout feature_count {}",
                payload_feature_count, predictor_layout.feature_count
            )));
        }
        if payload_feature_count != metadata_feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "trees payload feature_count {} does not match metadata feature count {}",
                payload_feature_count, metadata_feature_count
            )));
        }

        Ok(Self {
            metadata,
            baseline_prediction,
            stumps,
        })
    }

    pub fn predict_row(&self, features: &[f32]) -> PredictorResult<f32> {
        let feature_count = self.metadata.feature_names.len();
        if features.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                feature_count
            )));
        }
        let stumps_by_node = self
            .stumps
            .iter()
            .map(|stump| (stump.node_id, stump))
            .collect::<HashMap<_, _>>();

        let mut prediction = self.baseline_prediction;
        for stump in &self.stumps {
            if !row_satisfies_stump_path_features(features, stump, &stumps_by_node)? {
                continue;
            }
            let feature_index = stump.feature_index as usize;
            let feature_value = features[feature_index];
            let threshold = stump.threshold_bin as f32;
            prediction += if feature_value <= threshold {
                stump.left_leaf_value
            } else {
                stump.right_leaf_value
            };
        }

        Ok(prediction)
    }

    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }
        rows.iter().map(|row| self.predict_row(row)).collect()
    }

    pub fn predict_row_stub(&self, features: &[f32]) -> PredictorResult<f32> {
        self.predict_row(features)
    }

    pub fn predict_batch_stub(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        self.predict_batch(rows)
    }
}

fn required_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> PredictorResult<&ModelArtifactSection> {
    optional_single_section(sections, kind)?.ok_or_else(|| {
        PredictorError::ContractViolation(format!(
            "model artifact missing required {:?} section",
            kind
        ))
    })
}

fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> PredictorResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(PredictorError::ContractViolation(format!(
                "model artifact contains duplicate required {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

fn resolve_predictor_layout(
    sections: &[ModelArtifactSection],
    metadata_feature_count: usize,
) -> PredictorResult<PredictorLayoutPayload> {
    if let Some(section) = optional_single_section(sections, ModelSectionKind::PredictorLayout)? {
        return decode_predictor_layout_payload(&section.payload);
    }

    if sections.len() == 1 && sections[0].descriptor.kind == ModelSectionKind::Trees {
        return Ok(PredictorLayoutPayload {
            feature_count: metadata_feature_count,
        });
    }

    Err(PredictorError::ContractViolation(
        "model artifact missing required PredictorLayout section".to_string(),
    ))
}

fn decode_predictor_layout_payload(bytes: &[u8]) -> PredictorResult<PredictorLayoutPayload> {
    const LAYOUT_LEN: usize = 12;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
    if bytes.len() != LAYOUT_LEN {
        return Err(PredictorError::ContractViolation(format!(
            "predictor layout payload length {} does not match expected {LAYOUT_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported predictor layout format version {format_version}"
        )));
    }

    let feature_count = read_u32_le(bytes, 4)? as usize;
    let threshold_mode = read_u32_le(bytes, 8)?;
    if threshold_mode != THRESHOLD_MODE_BIN_INDEX {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported predictor layout threshold mode {threshold_mode}"
        )));
    }

    Ok(PredictorLayoutPayload { feature_count })
}

fn decode_trained_model_payload(
    bytes: &[u8],
) -> PredictorResult<(usize, f32, Vec<PredictorStump>)> {
    const HEADER_SIZE: usize = 16;
    const STUMP_SIZE: usize = 32;
    if bytes.len() < HEADER_SIZE {
        return Err(PredictorError::ContractViolation(
            "model payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported model payload format version {format_version}"
        )));
    }
    let feature_count = read_u32_le(bytes, 4)? as usize;
    let stump_count = read_u32_le(bytes, 8)? as usize;
    let baseline_prediction = read_f32_le(bytes, 12)?;

    let expected_len = HEADER_SIZE
        .checked_add(stump_count.checked_mul(STUMP_SIZE).ok_or_else(|| {
            PredictorError::ContractViolation("stump payload length overflow".to_string())
        })?)
        .ok_or_else(|| PredictorError::ContractViolation("payload length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(PredictorError::ContractViolation(format!(
            "model payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut stumps = Vec::with_capacity(stump_count);
    for stump_index in 0..stump_count {
        let base = HEADER_SIZE + stump_index * STUMP_SIZE;
        let node_id = read_u32_le(bytes, base)?;
        let feature_index = read_u32_le(bytes, base + 4)?;
        let threshold_bin = read_u16_le(bytes, base + 8)?;
        let _gain = read_f32_le(bytes, base + 12)?;
        let left_leaf_value = read_f32_le(bytes, base + 16)?;
        let right_leaf_value = read_f32_le(bytes, base + 20)?;
        stumps.push(PredictorStump {
            node_id,
            feature_index,
            threshold_bin,
            left_leaf_value,
            right_leaf_value,
        });
    }

    Ok((feature_count, baseline_prediction, stumps))
}

const TREE_NODE_STRIDE: u32 = 1 << 20;

fn encode_tree_node_id(tree_index: u32, local_node_id: u32) -> PredictorResult<u32> {
    if local_node_id >= TREE_NODE_STRIDE {
        return Err(PredictorError::ContractViolation(format!(
            "local node_id {local_node_id} exceeds supported tree-node stride {TREE_NODE_STRIDE}"
        )));
    }
    tree_index
        .checked_mul(TREE_NODE_STRIDE)
        .and_then(|base| base.checked_add(local_node_id))
        .ok_or_else(|| {
            PredictorError::ContractViolation(
                "encoded tree node id overflowed u32 range".to_string(),
            )
        })
}

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

fn row_satisfies_stump_path_features(
    features: &[f32],
    stump: &PredictorStump,
    stumps_by_node: &HashMap<u32, &PredictorStump>,
) -> PredictorResult<bool> {
    let (tree_id, mut local_node_id) = decode_tree_node_id(stump.node_id);
    while local_node_id > 0 {
        let parent_local = (local_node_id - 1) / 2;
        let parent_node_id = encode_tree_node_id(tree_id, parent_local)?;
        let Some(parent_stump) = stumps_by_node.get(&parent_node_id) else {
            return Ok(false);
        };
        let feature_index = parent_stump.feature_index as usize;
        if feature_index >= features.len() {
            return Err(PredictorError::ContractViolation(format!(
                "split feature_index {} exceeds feature length {}",
                parent_stump.feature_index,
                features.len()
            )));
        }
        let went_left = features[feature_index] <= parent_stump.threshold_bin as f32;
        let expected_left = local_node_id == parent_local * 2 + 1;
        if went_left != expected_left {
            return Ok(false);
        }
        local_node_id = parent_local;
    }
    Ok(true)
}

fn read_u32_le(bytes: &[u8], start: usize) -> PredictorResult<u32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading u32".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}

fn read_u16_le(bytes: &[u8], start: usize) -> PredictorResult<u16> {
    let end = start + 2;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

fn read_f32_le(bytes: &[u8], start: usize) -> PredictorResult<f32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading f32".to_string(),
        ));
    }
    Ok(f32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_core::{
        BinnedMatrix, DatasetMatrix, Device, ModelSectionKind, TrainParams, TrainingDataset,
        serialize_model_artifact_v1,
    };
    use alloygbm_engine::{SquaredErrorObjective, Trainer};

    fn predictor_stub() -> Predictor {
        let metadata = ModelMetadata {
            format_version: 1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
        };
        Predictor::new(metadata)
    }

    fn quality_fixture_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    4.0, 0.0, //
                    5.0, 0.0, //
                    6.0, 0.0, //
                    7.0, 0.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        }
    }

    fn quality_fixture_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            8,
            2,
            7,
            vec![
                0, 0, //
                1, 0, //
                2, 0, //
                3, 0, //
                4, 0, //
                5, 0, //
                6, 0, //
                7, 0, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn fixture_rows(dataset: &TrainingDataset) -> Vec<Vec<f32>> {
        dataset
            .matrix
            .values
            .chunks(dataset.matrix.feature_count)
            .map(|row| row.to_vec())
            .collect()
    }

    fn fixture_params() -> TrainParams {
        TrainParams {
            seed: 7,
            deterministic: true,
            learning_rate: 0.3,
            max_depth: 2,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
        }
    }

    fn train_engine_model() -> (alloygbm_engine::TrainedModel, TrainingDataset) {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 2)
            .expect("training succeeds");
        (model, dataset)
    }

    fn strict_artifact_payloads() -> (ModelMetadata, Vec<u8>, Vec<u8>) {
        let (engine_model, _) = train_engine_model();
        let strict_artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&strict_artifact)
            .expect("strict artifact parses");
        let trees_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists");
        let layout_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
            .map(|section| section.payload.clone())
            .expect("predictor layout payload exists");
        (parsed.contract.metadata, trees_payload, layout_payload)
    }

    #[test]
    fn predictor_from_artifact_matches_engine_predictions() {
        let (engine_model, dataset) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let predictor_predictions = predictor.predict_batch(&rows).expect("predictor predicts");
        assert_eq!(engine_predictions, predictor_predictions);
    }

    #[test]
    fn predictor_row_matches_engine_prediction() {
        let (engine_model, dataset) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);
        let row = &rows[0];

        let engine_prediction = engine_model.predict_row(row).expect("engine predicts");
        let predictor_prediction = predictor.predict_row(row).expect("predictor predicts");
        assert_eq!(engine_prediction, predictor_prediction);
    }

    #[test]
    fn predictor_accepts_legacy_trees_only_artifact() {
        let (engine_model, dataset) = train_engine_model();
        let strict_artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&strict_artifact)
            .expect("strict artifact parses");
        let trees_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists");
        let legacy_artifact = serialize_model_artifact_v1(
            &parsed.contract.metadata,
            &[(ModelSectionKind::Trees, trees_payload)],
        )
        .expect("legacy artifact serializes");

        let predictor = Predictor::from_artifact_bytes(&legacy_artifact).expect("legacy parses");
        let rows = fixture_rows(&dataset);
        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let predictor_predictions = predictor.predict_batch(&rows).expect("predictor predicts");
        assert_eq!(engine_predictions, predictor_predictions);
    }

    #[test]
    fn predictor_rejects_duplicate_required_sections() {
        let (metadata, trees_payload, layout_payload) = strict_artifact_payloads();
        let duplicate_layout_artifact = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("duplicate artifact serializes");

        assert!(matches!(
            Predictor::from_artifact_bytes(&duplicate_layout_artifact),
            Err(PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn predictor_rejects_non_legacy_missing_predictor_layout_section() {
        let (metadata, trees_payload, _) = strict_artifact_payloads();
        let non_legacy_missing_layout = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::ShapAux, vec![9_u8]),
            ],
        )
        .expect("artifact serializes");

        assert!(matches!(
            Predictor::from_artifact_bytes(&non_legacy_missing_layout),
            Err(PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn predictor_rejects_missing_trees_section() {
        let (metadata, _, layout_payload) = strict_artifact_payloads();
        let missing_trees = serialize_model_artifact_v1(
            &metadata,
            &[(ModelSectionKind::PredictorLayout, layout_payload)],
        )
        .expect("artifact serializes");

        assert!(matches!(
            Predictor::from_artifact_bytes(&missing_trees),
            Err(PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn predictor_row_rejects_feature_count_mismatch() {
        let (engine_model, _) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let result = predictor.predict_row(&[1.0]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }

    #[test]
    fn batch_rejects_empty_rows() {
        let pred = predictor_stub();
        let result = pred.predict_batch(&[]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }

    #[test]
    fn row_stub_kept_as_alias() {
        let pred = predictor_stub();
        let result = pred.predict_row_stub(&[1.0, 2.0]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }
}
