use std::error::Error;
use std::fmt::{Display, Formatter};

pub const MODEL_FORMAT_V1: u32 = 1;
pub const MODEL_BINARY_MAGIC: [u8; 4] = *b"AGBM";
pub const MODEL_BINARY_HEADER_LEN: usize = 16;
pub const MODEL_SECTION_DESCRIPTOR_LEN: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn as_metadata_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
        }
    }

    pub fn parse_metadata_label(value: &str) -> CoreResult<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            other => Err(CoreError::Validation(format!(
                "unsupported trained_device '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    pub seed: u64,
    pub deterministic: bool,
    pub learning_rate: f32,
    pub max_depth: u16,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            deterministic: true,
            learning_rate: 0.1,
            max_depth: 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSchema {
    pub feature_count: usize,
    pub has_time_index: bool,
    pub has_group_id: bool,
    pub categorical_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub values: Vec<f32>,
}

impl DatasetMatrix {
    pub fn new(row_count: usize, feature_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            values,
        };
        validate_dataset_matrix(&matrix)?;
        Ok(matrix)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataset {
    pub matrix: DatasetMatrix,
    pub targets: Vec<f32>,
    pub sample_weights: Option<Vec<f32>>,
    pub time_index: Option<Vec<i64>>,
    pub group_id: Option<Vec<u32>>,
}

impl TrainingDataset {
    pub fn row_count(&self) -> usize {
        self.matrix.row_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinnedMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub max_bin: u16,
    pub bins: Vec<u16>,
}

impl BinnedMatrix {
    pub fn new(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins: Vec<u16>,
    ) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            bins,
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

impl GradientPair {
    pub fn new(grad: f32, hess: f32) -> CoreResult<Self> {
        if !grad.is_finite() || !hess.is_finite() {
            return Err(CoreError::Validation(
                "gradient and hessian must be finite".to_string(),
            ));
        }
        if hess <= 0.0 {
            return Err(CoreError::Validation(
                "hessian must be greater than 0".to_string(),
            ));
        }
        Ok(Self { grad, hess })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureTile {
    pub start_feature: u32,
    pub end_feature: u32,
}

impl FeatureTile {
    pub fn new(start_feature: u32, end_feature: u32) -> CoreResult<Self> {
        if start_feature >= end_feature {
            return Err(CoreError::Validation(
                "feature tile must satisfy start_feature < end_feature".to_string(),
            ));
        }
        Ok(Self {
            start_feature,
            end_feature,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSlice {
    pub node_id: u32,
    pub row_indices: Vec<u32>,
}

impl NodeSlice {
    pub fn new(node_id: u32, row_indices: Vec<u32>) -> CoreResult<Self> {
        if row_indices.is_empty() {
            return Err(CoreError::Validation(
                "node row_indices cannot be empty".to_string(),
            ));
        }
        Ok(Self {
            node_id,
            row_indices,
        })
    }

    pub fn validate_bounds(&self, row_count: usize) -> CoreResult<()> {
        for &row_index in &self.row_indices {
            let row_index = row_index as usize;
            if row_index >= row_count {
                return Err(CoreError::Validation(format!(
                    "row index {row_index} is out of bounds for row_count {row_count}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub row_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBin {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureHistogram {
    pub feature_index: u32,
    pub bins: Vec<HistogramBin>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBundle {
    pub node_id: u32,
    pub feature_histograms: Vec<FeatureHistogram>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitCandidate {
    pub node_id: u32,
    pub feature_index: u32,
    pub threshold_bin: u16,
    pub gain: f32,
    pub left_stats: NodeStats,
    pub right_stats: NodeStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionResult {
    pub left_row_indices: Vec<u32>,
    pub right_row_indices: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub trained_device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelBinaryHeader {
    pub magic: [u8; 4],
    pub format_version: u32,
    pub section_count: u32,
    pub metadata_json_len: u32,
}

impl ModelBinaryHeader {
    pub fn new(section_count: u32, metadata_json_len: u32) -> Self {
        Self {
            magic: MODEL_BINARY_MAGIC,
            format_version: MODEL_FORMAT_V1,
            section_count,
            metadata_json_len,
        }
    }

    pub fn encode(self) -> [u8; MODEL_BINARY_HEADER_LEN] {
        let mut bytes = [0_u8; MODEL_BINARY_HEADER_LEN];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.format_version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.section_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.metadata_json_len.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.len() != MODEL_BINARY_HEADER_LEN {
            return Err(CoreError::Serialization(format!(
                "model header must be {MODEL_BINARY_HEADER_LEN} bytes"
            )));
        }

        let mut magic = [0_u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != MODEL_BINARY_MAGIC {
            return Err(CoreError::Serialization(
                "model header magic mismatch".to_string(),
            ));
        }

        let format_version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let section_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let metadata_json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        Ok(Self {
            magic,
            format_version,
            section_count,
            metadata_json_len,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSectionKind {
    Trees,
    PredictorLayout,
    ShapAux,
    CategoricalState,
    Unknown(u32),
}

impl ModelSectionKind {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Trees => 1,
            Self::PredictorLayout => 2,
            Self::ShapAux => 3,
            Self::CategoricalState => 4,
            Self::Unknown(value) => value,
        }
    }

    pub fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Trees,
            2 => Self::PredictorLayout,
            3 => Self::ShapAux,
            4 => Self::CategoricalState,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSectionDescriptor {
    pub kind: ModelSectionKind,
    pub offset: u64,
    pub length: u64,
}

impl ModelSectionDescriptor {
    pub fn encode(self) -> [u8; MODEL_SECTION_DESCRIPTOR_LEN] {
        let mut bytes = [0_u8; MODEL_SECTION_DESCRIPTOR_LEN];
        bytes[0..4].copy_from_slice(&self.kind.to_u32().to_le_bytes());
        bytes[4..12].copy_from_slice(&self.offset.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.length.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.len() != MODEL_SECTION_DESCRIPTOR_LEN {
            return Err(CoreError::Serialization(format!(
                "model section descriptor must be {MODEL_SECTION_DESCRIPTOR_LEN} bytes"
            )));
        }

        let kind = ModelSectionKind::from_u32(u32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]));
        let offset = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        let length = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);

        Ok(Self {
            kind,
            offset,
            length,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIoContractV1 {
    pub header: ModelBinaryHeader,
    pub sections: Vec<ModelSectionDescriptor>,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactSection {
    pub descriptor: ModelSectionDescriptor,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedModelArtifactV1 {
    pub contract: ModelIoContractV1,
    pub metadata_json: String,
    pub sections: Vec<ModelArtifactSection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidConfig(String),
    Validation(String),
    Io(String),
    Serialization(String),
    NotImplemented(String),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl Error for CoreError {}

pub type CoreResult<T> = Result<T, CoreError>;

pub fn validate_train_params(params: &TrainParams) -> CoreResult<()> {
    if !(0.0..=1.0).contains(&params.learning_rate) || params.learning_rate == 0.0 {
        return Err(CoreError::InvalidConfig(
            "learning_rate must be in (0.0, 1.0]".to_string(),
        ));
    }

    if params.max_depth == 0 {
        return Err(CoreError::InvalidConfig(
            "max_depth must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

pub fn validate_dataset_schema(schema: &DatasetSchema) -> CoreResult<()> {
    if schema.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }

    for &feature_index in &schema.categorical_feature_indices {
        if feature_index >= schema.feature_count {
            return Err(CoreError::Validation(format!(
                "categorical feature index {feature_index} is out of bounds for feature_count {}",
                schema.feature_count
            )));
        }
    }

    Ok(())
}

pub fn validate_dataset_matrix(matrix: &DatasetMatrix) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    if matrix.values.len() != matrix.row_count * matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "matrix values length {} does not match row_count * feature_count {}",
            matrix.values.len(),
            matrix.row_count * matrix.feature_count
        )));
    }
    Ok(())
}

pub fn validate_training_dataset(dataset: &TrainingDataset) -> CoreResult<()> {
    validate_dataset_matrix(&dataset.matrix)?;
    if dataset.targets.len() != dataset.matrix.row_count {
        return Err(CoreError::Validation(format!(
            "targets length {} does not match row_count {}",
            dataset.targets.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(weights) = &dataset.sample_weights
        && weights.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "sample_weights length {} does not match row_count {}",
            weights.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(time_index) = &dataset.time_index
        && time_index.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "time_index length {} does not match row_count {}",
            time_index.len(),
            dataset.matrix.row_count
        )));
    }

    if let Some(group_id) = &dataset.group_id
        && group_id.len() != dataset.matrix.row_count
    {
        return Err(CoreError::Validation(format!(
            "group_id length {} does not match row_count {}",
            group_id.len(),
            dataset.matrix.row_count
        )));
    }

    Ok(())
}

pub fn validate_binned_matrix(matrix: &BinnedMatrix) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    if matrix.max_bin == 0 {
        return Err(CoreError::Validation(
            "max_bin must be greater than 0".to_string(),
        ));
    }
    if matrix.bins.len() != matrix.row_count * matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "bins length {} does not match row_count * feature_count {}",
            matrix.bins.len(),
            matrix.row_count * matrix.feature_count
        )));
    }
    for &bin in &matrix.bins {
        if bin > matrix.max_bin {
            return Err(CoreError::Validation(format!(
                "bin value {bin} exceeds max_bin {}",
                matrix.max_bin
            )));
        }
    }
    Ok(())
}

pub fn validate_model_contract_v1(contract: &ModelIoContractV1) -> CoreResult<()> {
    if contract.header.magic != MODEL_BINARY_MAGIC {
        return Err(CoreError::Serialization(
            "model contract magic mismatch".to_string(),
        ));
    }
    if contract.header.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "unsupported format_version {}, expected {MODEL_FORMAT_V1}",
            contract.header.format_version
        )));
    }
    if contract.metadata.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "metadata format_version {}, expected {MODEL_FORMAT_V1}",
            contract.metadata.format_version
        )));
    }
    if contract.sections.len() != contract.header.section_count as usize {
        return Err(CoreError::Serialization(format!(
            "section table length {} does not match header section_count {}",
            contract.sections.len(),
            contract.header.section_count
        )));
    }

    let mut previous_end = 0_u64;
    for section in &contract.sections {
        if section.length == 0 {
            return Err(CoreError::Serialization(
                "section length must be greater than 0".to_string(),
            ));
        }
        if section.offset < previous_end {
            return Err(CoreError::Serialization(
                "section offsets must be non-overlapping and ordered".to_string(),
            ));
        }
        previous_end = section
            .offset
            .checked_add(section.length)
            .ok_or_else(|| CoreError::Serialization("section offset overflow".to_string()))?;
    }

    Ok(())
}

pub fn serialize_metadata_json(metadata: &ModelMetadata) -> String {
    let feature_names = metadata
        .feature_names
        .iter()
        .map(|name| format!("\"{}\"", escape_json_string(name)))
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "{{\"format_version\":{},\"feature_names\":[{}],\"trained_device\":\"{}\"}}",
        metadata.format_version,
        feature_names,
        metadata.trained_device.as_metadata_label()
    )
}

pub fn deserialize_metadata_json(input: &str) -> CoreResult<ModelMetadata> {
    let compact = compact_json(input)?;
    let mut index = 0_usize;

    index = consume_literal(&compact, index, "{\"format_version\":")?;
    let (format_version, next_index) = parse_u32(&compact, index)?;
    index = next_index;

    index = consume_literal(&compact, index, ",\"feature_names\":")?;
    let (feature_names, next_index) = parse_string_array(&compact, index)?;
    index = next_index;

    index = consume_literal(&compact, index, ",\"trained_device\":")?;
    let (trained_device_raw, next_index) = parse_quoted_string(&compact, index)?;
    index = next_index;

    index = consume_literal(&compact, index, "}")?;
    if index != compact.len() {
        return Err(CoreError::Serialization(
            "unexpected trailing content in metadata json".to_string(),
        ));
    }

    Ok(ModelMetadata {
        format_version,
        feature_names,
        trained_device: Device::parse_metadata_label(&trained_device_raw)?,
    })
}

pub fn serialize_model_artifact_v1(
    metadata: &ModelMetadata,
    sections: &[(ModelSectionKind, Vec<u8>)],
) -> CoreResult<Vec<u8>> {
    if sections.is_empty() {
        return Err(CoreError::Serialization(
            "model artifact requires at least one section".to_string(),
        ));
    }

    let metadata_json = serialize_metadata_json(metadata);
    let metadata_json_bytes = metadata_json.as_bytes();
    let metadata_json_len = u32::try_from(metadata_json_bytes.len()).map_err(|_| {
        CoreError::Serialization("metadata json length exceeds u32::MAX".to_string())
    })?;

    let section_count = u32::try_from(sections.len())
        .map_err(|_| CoreError::Serialization("section count exceeds u32::MAX".to_string()))?;
    let descriptor_table_len = sections
        .len()
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let data_start = MODEL_BINARY_HEADER_LEN
        .checked_add(descriptor_table_len)
        .and_then(|value| value.checked_add(metadata_json_bytes.len()))
        .ok_or_else(|| CoreError::Serialization("artifact header length overflow".to_string()))?;

    let mut descriptors = Vec::with_capacity(sections.len());
    let mut offset = data_start as u64;
    for (kind, payload) in sections {
        if payload.is_empty() {
            return Err(CoreError::Serialization(
                "section payload cannot be empty".to_string(),
            ));
        }
        let length = u64::try_from(payload.len())
            .map_err(|_| CoreError::Serialization("section length overflow".to_string()))?;
        descriptors.push(ModelSectionDescriptor {
            kind: *kind,
            offset,
            length,
        });
        offset = offset
            .checked_add(length)
            .ok_or_else(|| CoreError::Serialization("section offset overflow".to_string()))?;
    }

    let contract = ModelIoContractV1 {
        header: ModelBinaryHeader::new(section_count, metadata_json_len),
        sections: descriptors.clone(),
        metadata: metadata.clone(),
    };
    validate_model_contract_v1(&contract)?;

    let final_len = usize::try_from(offset)
        .map_err(|_| CoreError::Serialization("artifact length exceeds usize".to_string()))?;
    let mut bytes = Vec::with_capacity(final_len);
    bytes.extend_from_slice(&contract.header.encode());
    for descriptor in &descriptors {
        bytes.extend_from_slice(&descriptor.encode());
    }
    bytes.extend_from_slice(metadata_json_bytes);
    for (_, payload) in sections {
        bytes.extend_from_slice(payload);
    }

    Ok(bytes)
}

pub fn deserialize_model_artifact_v1(bytes: &[u8]) -> CoreResult<ParsedModelArtifactV1> {
    if bytes.len() < MODEL_BINARY_HEADER_LEN {
        return Err(CoreError::Serialization(
            "artifact too small to contain model header".to_string(),
        ));
    }

    let header = ModelBinaryHeader::decode(&bytes[0..MODEL_BINARY_HEADER_LEN])?;
    if header.format_version != MODEL_FORMAT_V1 {
        return Err(CoreError::Serialization(format!(
            "unsupported format_version {}, expected {MODEL_FORMAT_V1}",
            header.format_version
        )));
    }

    let section_count = header.section_count as usize;
    let descriptor_table_len = section_count
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let descriptor_start = MODEL_BINARY_HEADER_LEN;
    let descriptor_end = descriptor_start
        .checked_add(descriptor_table_len)
        .ok_or_else(|| CoreError::Serialization("descriptor range overflow".to_string()))?;
    if bytes.len() < descriptor_end {
        return Err(CoreError::Serialization(
            "artifact truncated in section descriptor table".to_string(),
        ));
    }

    let mut descriptors = Vec::with_capacity(section_count);
    for section_index in 0..section_count {
        let start = descriptor_start + section_index * MODEL_SECTION_DESCRIPTOR_LEN;
        let end = start + MODEL_SECTION_DESCRIPTOR_LEN;
        descriptors.push(ModelSectionDescriptor::decode(&bytes[start..end])?);
    }

    let metadata_json_len = header.metadata_json_len as usize;
    let metadata_start = descriptor_end;
    let metadata_end = metadata_start
        .checked_add(metadata_json_len)
        .ok_or_else(|| CoreError::Serialization("metadata range overflow".to_string()))?;
    if bytes.len() < metadata_end {
        return Err(CoreError::Serialization(
            "artifact truncated in metadata payload".to_string(),
        ));
    }
    let metadata_json = std::str::from_utf8(&bytes[metadata_start..metadata_end])
        .map_err(|err| {
            CoreError::Serialization(format!("metadata json is not valid UTF-8: {err}"))
        })?
        .to_string();
    let metadata = deserialize_metadata_json(&metadata_json)?;

    let contract = ModelIoContractV1 {
        header,
        sections: descriptors.clone(),
        metadata,
    };
    validate_model_contract_v1(&contract)?;

    let mut parsed_sections = Vec::with_capacity(descriptors.len());
    for descriptor in &descriptors {
        let start = usize::try_from(descriptor.offset)
            .map_err(|_| CoreError::Serialization("section offset exceeds usize".to_string()))?;
        let length = usize::try_from(descriptor.length)
            .map_err(|_| CoreError::Serialization("section length exceeds usize".to_string()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| CoreError::Serialization("section range overflow".to_string()))?;

        if end > bytes.len() {
            return Err(CoreError::Serialization(
                "artifact truncated in section payload".to_string(),
            ));
        }

        parsed_sections.push(ModelArtifactSection {
            descriptor: *descriptor,
            payload: bytes[start..end].to_vec(),
        });
    }

    Ok(ParsedModelArtifactV1 {
        contract,
        metadata_json,
        sections: parsed_sections,
    })
}

fn escape_json_string(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn compact_json(input: &str) -> CoreResult<String> {
    let mut compact = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;

    for ch in input.chars() {
        if in_string {
            compact.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        if ch.is_whitespace() {
            continue;
        }

        compact.push(ch);
        if ch == '"' {
            in_string = true;
        }
    }

    if in_string || escaped {
        return Err(CoreError::Serialization(
            "metadata json has unterminated string".to_string(),
        ));
    }

    Ok(compact)
}

fn consume_literal(input: &str, index: usize, literal: &str) -> CoreResult<usize> {
    if index > input.len() || !input[index..].starts_with(literal) {
        return Err(CoreError::Serialization(format!(
            "expected literal '{literal}' at index {index}"
        )));
    }
    Ok(index + literal.len())
}

fn parse_u32(input: &str, mut index: usize) -> CoreResult<(u32, usize)> {
    let start = index;
    while let Some(byte) = input.as_bytes().get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        index += 1;
    }
    if start == index {
        return Err(CoreError::Serialization(format!(
            "expected unsigned integer at index {start}"
        )));
    }
    let value = input[start..index]
        .parse::<u32>()
        .map_err(|err| CoreError::Serialization(format!("invalid integer: {err}")))?;
    Ok((value, index))
}

fn parse_string_array(input: &str, mut index: usize) -> CoreResult<(Vec<String>, usize)> {
    index = consume_literal(input, index, "[")?;
    let mut values = Vec::new();

    if input[index..].starts_with(']') {
        return Ok((values, index + 1));
    }

    loop {
        let (value, next_index) = parse_quoted_string(input, index)?;
        values.push(value);
        index = next_index;

        if input[index..].starts_with(',') {
            index += 1;
            continue;
        }
        if input[index..].starts_with(']') {
            index += 1;
            break;
        }
        return Err(CoreError::Serialization(format!(
            "expected ',' or ']' at index {index}"
        )));
    }

    Ok((values, index))
}

fn parse_quoted_string(input: &str, index: usize) -> CoreResult<(String, usize)> {
    if !input[index..].starts_with('"') {
        return Err(CoreError::Serialization(format!(
            "expected quoted string at index {index}"
        )));
    }

    let mut output = String::new();
    let mut escaped = false;
    let body_start = index + 1;
    for (relative_offset, ch) in input[body_start..].char_indices() {
        if escaped {
            let decoded = match ch {
                '\\' => '\\',
                '"' => '"',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => {
                    return Err(CoreError::Serialization(format!(
                        "unsupported escape sequence '\\{other}'"
                    )));
                }
            };
            output.push(decoded);
            escaped = false;
            continue;
        }

        match ch {
            '\\' => escaped = true,
            '"' => {
                let end_index = body_start + relative_offset + ch.len_utf8();
                return Ok((output, end_index));
            }
            _ => output.push(ch),
        }
    }

    Err(CoreError::Serialization(
        "unterminated quoted string".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["feature_0".to_string(), "ticker\"id".to_string()],
            trained_device: Device::Cpu,
        }
    }

    #[test]
    fn validates_default_train_params() {
        let params = TrainParams::default();
        assert!(validate_train_params(&params).is_ok());
    }

    #[test]
    fn rejects_invalid_learning_rate() {
        let params = TrainParams {
            learning_rate: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            validate_train_params(&params),
            Err(CoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validates_dataset_schema() {
        let schema = DatasetSchema {
            feature_count: 4,
            has_time_index: true,
            has_group_id: true,
            categorical_feature_indices: vec![1, 3],
        };
        assert!(validate_dataset_schema(&schema).is_ok());
    }

    #[test]
    fn rejects_training_dataset_with_mismatched_targets() {
        let dataset = TrainingDataset {
            matrix: DatasetMatrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]).expect("valid matrix"),
            targets: vec![1.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
        };
        assert!(matches!(
            validate_training_dataset(&dataset),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn binned_matrix_rejects_bin_above_max() {
        let matrix = BinnedMatrix {
            row_count: 1,
            feature_count: 2,
            max_bin: 7,
            bins: vec![3, 8],
        };
        assert!(matches!(
            validate_binned_matrix(&matrix),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn gradient_pair_rejects_non_positive_hessian() {
        assert!(matches!(
            GradientPair::new(0.1, 0.0),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn model_header_roundtrip() {
        let header = ModelBinaryHeader::new(2, 128);
        let bytes = header.encode();
        let decoded = ModelBinaryHeader::decode(&bytes).expect("header should decode");
        assert_eq!(decoded, header);
    }

    #[test]
    fn section_descriptor_roundtrip() {
        let descriptor = ModelSectionDescriptor {
            kind: ModelSectionKind::Trees,
            offset: 16,
            length: 64,
        };
        let bytes = descriptor.encode();
        let decoded = ModelSectionDescriptor::decode(&bytes).expect("descriptor should decode");
        assert_eq!(decoded, descriptor);
    }

    #[test]
    fn metadata_json_roundtrip() {
        let metadata = sample_metadata();
        let json = serialize_metadata_json(&metadata);
        let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
        assert_eq!(decoded, metadata);
    }

    #[test]
    fn metadata_json_rejects_unknown_device() {
        let json = "{\"format_version\":1,\"feature_names\":[\"f0\"],\"trained_device\":\"cuda\"}";
        assert!(matches!(
            deserialize_metadata_json(json),
            Err(CoreError::Validation(_))
        ));
    }

    #[test]
    fn model_contract_rejects_overlapping_sections() {
        let contract = ModelIoContractV1 {
            header: ModelBinaryHeader::new(2, 64),
            sections: vec![
                ModelSectionDescriptor {
                    kind: ModelSectionKind::Trees,
                    offset: 16,
                    length: 64,
                },
                ModelSectionDescriptor {
                    kind: ModelSectionKind::PredictorLayout,
                    offset: 40,
                    length: 10,
                },
            ],
            metadata: ModelMetadata {
                format_version: MODEL_FORMAT_V1,
                feature_names: vec!["f0".to_string()],
                trained_device: Device::Cpu,
            },
        };
        assert!(matches!(
            validate_model_contract_v1(&contract),
            Err(CoreError::Serialization(_))
        ));
    }

    #[test]
    fn model_artifact_roundtrip() {
        let metadata = sample_metadata();
        let sections = vec![
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
        ];

        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(parsed.contract.metadata, metadata);
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].descriptor.kind, ModelSectionKind::Trees);
        assert_eq!(parsed.sections[0].payload, vec![1_u8, 2, 3, 4]);
        assert_eq!(
            parsed.sections[1].descriptor.kind,
            ModelSectionKind::PredictorLayout
        );
        assert_eq!(parsed.sections[1].payload, vec![9_u8, 8, 7]);
    }

    #[test]
    fn model_artifact_deserialize_rejects_truncated_payload() {
        let metadata = sample_metadata();
        let sections = vec![(ModelSectionKind::Trees, vec![1_u8, 2, 3, 4])];
        let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            deserialize_model_artifact_v1(truncated),
            Err(CoreError::Serialization(_))
        ));
    }
}
