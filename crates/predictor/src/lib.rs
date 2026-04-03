use alloygbm_core::{
    CategoricalStatePayloadV1, CoreError, MODEL_FORMAT_V1, ModelArtifactSection, ModelMetadata,
    ModelSectionKind, decode_optional_categorical_state_section_v1, deserialize_model_artifact_v1,
    format_required_section_mode_error, required_section_compatibility_report,
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

const PARALLEL_PREDICT_MIN_ROWS: usize = 256;
const PARALLEL_PREDICT_MIN_WORK_ITEMS: usize = 16_384;

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
struct PredictorTreeNode {
    feature_index: usize,
    threshold_bin: f32,
    left_leaf_value: f32,
    right_leaf_value: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct PredictorTree {
    nodes_by_local_id: Vec<Option<PredictorTreeNode>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    pub metadata: ModelMetadata,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    baseline_prediction: f32,
    trees: Vec<PredictorTree>,
    /// When true, `threshold_bin` fields contain float thresholds (not bin indices)
    /// and prediction uses `<` comparison instead of `<=`.
    use_float_thresholds: bool,
}

impl Predictor {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            categorical_state: None,
            baseline_prediction: 0.0,
            trees: Vec::new(),
            use_float_thresholds: false,
        }
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> PredictorResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(PredictorError::from)?;
        let compatibility_report = required_section_compatibility_report(&parsed.sections);
        if !compatibility_report.legacy_compatible {
            return Err(PredictorError::ContractViolation(
                format_required_section_mode_error(compatibility_report, true),
            ));
        }
        let metadata = parsed.contract.metadata;
        let metadata_feature_count = metadata.feature_names.len();
        let trees_section = required_single_section(&parsed.sections, ModelSectionKind::Trees)?;
        let predictor_layout = resolve_predictor_layout(&parsed.sections, metadata_feature_count)?;
        let categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, metadata_feature_count)
                .map_err(PredictorError::from)?;
        let (payload_feature_count, baseline_prediction, stumps) =
            decode_trained_model_payload(&trees_section.payload)?;
        let trees = build_predictor_trees(&stumps)?;

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
            categorical_state,
            baseline_prediction,
            trees,
            use_float_thresholds: false,
        })
    }

    /// Convert bin-index thresholds to float thresholds using per-feature min/max.
    /// After calling this, prediction compares raw float features directly — no quantization needed.
    /// Uses the midpoint between adjacent bin boundaries as the float threshold, with `<` comparison.
    pub fn convert_bin_thresholds_to_float(
        &mut self,
        feature_mins: &[f32],
        feature_maxs: &[f32],
    ) -> PredictorResult<()> {
        let feature_count = self.metadata.feature_names.len();
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_mins/maxs length ({}/{}) must match feature_count {}",
                feature_mins.len(),
                feature_maxs.len(),
                feature_count
            )));
        }
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                let fi = node.feature_index;
                let min_val = feature_mins[fi];
                let max_val = feature_maxs[fi];
                let span = max_val - min_val;
                if span <= f32::EPSILON {
                    // Constant feature — any threshold works since all values are equal.
                    node.threshold_bin = min_val + f32::EPSILON;
                } else {
                    // bin = round(((value - min) / span) * 255)
                    // Split: bin <= threshold_bin  ↔  value < min + ((threshold_bin + 0.5) / 255) * span
                    let bin = node.threshold_bin;
                    node.threshold_bin = min_val + ((bin + 0.5) / 255.0) * span;
                }
            }
        }
        self.use_float_thresholds = true;
        Ok(())
    }

    /// Convert bin-index thresholds to float thresholds using per-feature quantile cuts.
    /// For quantile binning: `bin = bisect_right(cuts, value)`, split is `bin <= threshold_bin`.
    /// The float equivalent: `value < cuts[threshold_bin]`.
    pub fn convert_bin_thresholds_to_float_quantile(
        &mut self,
        feature_cuts: &[Vec<f32>],
    ) -> PredictorResult<()> {
        let feature_count = self.metadata.feature_names.len();
        if feature_cuts.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_cuts length {} must match feature_count {}",
                feature_cuts.len(),
                feature_count
            )));
        }
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                let fi = node.feature_index;
                let cuts = &feature_cuts[fi];
                let bin = node.threshold_bin as usize;
                if bin < cuts.len() {
                    node.threshold_bin = cuts[bin];
                } else {
                    // All values go left — set threshold beyond any possible value
                    node.threshold_bin = f32::MAX;
                }
            }
        }
        self.use_float_thresholds = true;
        Ok(())
    }

    /// Convert bin-index thresholds to float thresholds for pre-binned integer data.
    /// For pre-binned data, values are integers (0..max_bin) and split is `bin <= threshold_bin`.
    /// The float equivalent: `value < threshold_bin + 0.5`.
    pub fn convert_bin_thresholds_to_float_prebinned(&mut self) -> PredictorResult<()> {
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                node.threshold_bin += 0.5;
            }
        }
        self.use_float_thresholds = true;
        Ok(())
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
        self.predict_row_with_feature_count(features, feature_count)
    }

    fn predict_row_with_feature_count(
        &self,
        features: &[f32],
        feature_count: usize,
    ) -> PredictorResult<f32> {
        let use_float = self.use_float_thresholds;
        let mut prediction = self.baseline_prediction;
        for tree in &self.trees {
            let mut local_node_id: usize = 0;
            while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                if node.feature_index >= feature_count {
                    return Err(PredictorError::ContractViolation(format!(
                        "split feature_index {} exceeds feature length {}",
                        node.feature_index, feature_count
                    )));
                }
                let feature_value = features[node.feature_index];
                let went_left = if use_float {
                    feature_value < node.threshold_bin
                } else {
                    feature_value <= node.threshold_bin
                };
                prediction += if went_left {
                    node.left_leaf_value
                } else {
                    node.right_leaf_value
                };
                local_node_id = if went_left {
                    local_node_id.saturating_mul(2).saturating_add(1)
                } else {
                    local_node_id.saturating_mul(2).saturating_add(2)
                };
            }
        }

        Ok(prediction)
    }

    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }
        let feature_count = self.metadata.feature_names.len();
        for row in rows {
            if row.len() != feature_count {
                return Err(PredictorError::InvalidInput(format!(
                    "feature length {} does not match model feature_count {}",
                    row.len(),
                    feature_count
                )));
            }
        }

        if should_parallel_predict_batch(rows.len(), self.trees.len()) {
            rows.par_iter()
                .map(|row| self.predict_row_with_feature_count(row, feature_count))
                .collect()
        } else {
            rows.iter()
                .map(|row| self.predict_row_with_feature_count(row, feature_count))
                .collect()
        }
    }

    /// Predict from a flat row-major dense slice — zero per-row allocation.
    pub fn predict_batch_dense(
        &self,
        values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> PredictorResult<Vec<f32>> {
        let model_feature_count = self.metadata.feature_names.len();
        if feature_count != model_feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_count {} does not match model feature_count {}",
                feature_count, model_feature_count
            )));
        }
        if values.len() != row_count * feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "values length {} does not match row_count * feature_count {}",
                values.len(),
                row_count * feature_count
            )));
        }
        if row_count == 0 {
            return Err(PredictorError::InvalidInput(
                "row_count must be greater than 0".to_string(),
            ));
        }

        if should_parallel_predict_batch(row_count, self.trees.len()) {
            (0..row_count)
                .into_par_iter()
                .map(|row_index| {
                    let row = &values[row_index * feature_count..(row_index + 1) * feature_count];
                    self.predict_row_dense_unchecked(row, feature_count)
                })
                .collect()
        } else {
            (0..row_count)
                .map(|row_index| {
                    let row = &values[row_index * feature_count..(row_index + 1) * feature_count];
                    self.predict_row_dense_unchecked(row, feature_count)
                })
                .collect()
        }
    }

    /// Inner prediction on a row slice — no length validation (caller ensures correctness).
    fn predict_row_dense_unchecked(
        &self,
        features: &[f32],
        feature_count: usize,
    ) -> PredictorResult<f32> {
        let use_float = self.use_float_thresholds;
        let mut prediction = self.baseline_prediction;
        for tree in &self.trees {
            let mut local_node_id: usize = 0;
            while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                if node.feature_index >= feature_count {
                    return Err(PredictorError::ContractViolation(format!(
                        "split feature_index {} exceeds feature length {}",
                        node.feature_index, feature_count
                    )));
                }
                let feature_value = features[node.feature_index];
                let went_left = if use_float {
                    feature_value < node.threshold_bin
                } else {
                    feature_value <= node.threshold_bin
                };
                prediction += if went_left {
                    node.left_leaf_value
                } else {
                    node.right_leaf_value
                };
                local_node_id = if went_left {
                    local_node_id * 2 + 1
                } else {
                    local_node_id * 2 + 2
                };
            }
        }
        Ok(prediction)
    }

    /// Predict from raw native-endian f32 bytes — avoids Python list→Vec<f32> overhead.
    /// Each parallel chunk converts bytes→f32 and predicts on the fly using a thread-local
    /// row buffer, avoiding any large intermediate allocation.
    pub fn predict_batch_dense_bytes(
        &self,
        bytes: &[u8],
        row_count: usize,
        feature_count: usize,
    ) -> PredictorResult<Vec<f32>> {
        let model_feature_count = self.metadata.feature_names.len();
        if feature_count != model_feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_count {} does not match model feature_count {}",
                feature_count, model_feature_count
            )));
        }
        let expected_bytes = row_count * feature_count * 4;
        if bytes.len() != expected_bytes {
            return Err(PredictorError::InvalidInput(format!(
                "bytes length {} does not match expected {} (row_count={} * feature_count={} * 4)",
                bytes.len(),
                expected_bytes,
                row_count,
                feature_count
            )));
        }
        if row_count == 0 {
            return Err(PredictorError::InvalidInput(
                "row_count must be greater than 0".to_string(),
            ));
        }

        let row_bytes = feature_count * 4;
        let chunk_size = 4096.max(row_count / (rayon::current_num_threads().max(1) * 4));
        let mut predictions = vec![0.0_f32; row_count];

        // Each parallel chunk gets one reusable row buffer (feature_count × 4 bytes).
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let mut row_buf = vec![0.0_f32; feature_count];
                for (local_idx, pred) in out_chunk.iter_mut().enumerate() {
                    let row_index = row_start + local_idx;
                    let byte_start = row_index * row_bytes;
                    for (fi, item) in row_buf.iter_mut().enumerate().take(feature_count) {
                        let bi = byte_start + fi * 4;
                        *item = f32::from_ne_bytes([
                            bytes[bi],
                            bytes[bi + 1],
                            bytes[bi + 2],
                            bytes[bi + 3],
                        ]);
                    }
                    *pred = self.predict_row_dense_unchecked(&row_buf, feature_count)?;
                }
                Ok::<(), PredictorError>(())
            })?;

        Ok(predictions)
    }

    pub fn predict_row_stub(&self, features: &[f32]) -> PredictorResult<f32> {
        self.predict_row(features)
    }

    pub fn predict_batch_stub(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        self.predict_batch(rows)
    }
}

fn should_parallel_predict_batch(row_count: usize, tree_count: usize) -> bool {
    row_count >= PARALLEL_PREDICT_MIN_ROWS
        && row_count.saturating_mul(tree_count.max(1)) >= PARALLEL_PREDICT_MIN_WORK_ITEMS
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

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

fn build_predictor_trees(stumps: &[PredictorStump]) -> PredictorResult<Vec<PredictorTree>> {
    let mut grouped_by_tree: BTreeMap<u32, Vec<(u32, PredictorTreeNode)>> = BTreeMap::new();
    for stump in stumps {
        let (tree_id, local_node_id) = decode_tree_node_id(stump.node_id);
        grouped_by_tree.entry(tree_id).or_default().push((
            local_node_id,
            PredictorTreeNode {
                feature_index: stump.feature_index as usize,
                threshold_bin: stump.threshold_bin as f32,
                left_leaf_value: stump.left_leaf_value,
                right_leaf_value: stump.right_leaf_value,
            },
        ));
    }

    let mut trees = Vec::with_capacity(grouped_by_tree.len());
    for (tree_id, nodes) in grouped_by_tree {
        let max_local_node_id = nodes
            .iter()
            .map(|(local_node_id, _)| *local_node_id as usize)
            .max()
            .unwrap_or(0);
        let mut nodes_by_local_id = vec![None; max_local_node_id + 1];
        for (local_node_id, node) in nodes {
            let local_node_id = local_node_id as usize;
            if nodes_by_local_id[local_node_id].is_some() {
                return Err(PredictorError::ContractViolation(format!(
                    "tree {tree_id} contains duplicate local node_id {local_node_id}"
                )));
            }
            nodes_by_local_id[local_node_id] = Some(node);
        }
        trees.push(PredictorTree { nodes_by_local_id });
    }

    Ok(trees)
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
        BinnedMatrix, CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix,
        Device, ModelSectionKind, TrainParams, TrainingDataset, serialize_model_artifact_v1,
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
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
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
    fn predictor_replays_artifact_with_optional_categorical_state() {
        let (engine_model, dataset) = train_engine_model();
        let engine_model = engine_model
            .with_categorical_state(Some(CategoricalStatePayloadV1 {
                format_version: CATEGORICAL_STATE_FORMAT_V1,
                leakage_safe_target_encoding: true,
                categorical_feature_indices: vec![1],
            }))
            .expect("categorical state is valid");
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);

        assert!(predictor.categorical_state.is_some());
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

        let result = Predictor::from_artifact_bytes(&duplicate_layout_artifact);
        match result {
            Err(PredictorError::ContractViolation(message)) => {
                let parsed =
                    alloygbm_core::deserialize_model_artifact_v1(&duplicate_layout_artifact)
                        .expect("artifact parses");
                let report = alloygbm_core::required_section_compatibility_report(&parsed.sections);
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(report, true)
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
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

        let result = Predictor::from_artifact_bytes(&missing_trees);
        match result {
            Err(PredictorError::ContractViolation(message)) => {
                let parsed =
                    alloygbm_core::deserialize_model_artifact_v1(&missing_trees).expect("parses");
                let report = alloygbm_core::required_section_compatibility_report(&parsed.sections);
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(report, true)
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
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

    #[test]
    fn batch_parallelization_policy_requires_sufficient_workload() {
        assert!(!should_parallel_predict_batch(
            PARALLEL_PREDICT_MIN_ROWS - 1,
            100
        ));
        assert!(!should_parallel_predict_batch(PARALLEL_PREDICT_MIN_ROWS, 1));
        assert!(should_parallel_predict_batch(PARALLEL_PREDICT_MIN_ROWS, 64));
        assert!(should_parallel_predict_batch(4_096, 4));
    }
}
