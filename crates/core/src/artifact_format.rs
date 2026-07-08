//! Binary artifact format and JSON metadata serde.
//!
//! Defines the AlloyGBM model artifact wire format: header, section descriptors,
//! per-section payload types (categorical state, native categorical splits, morph /
//! dro / neutralization metadata, linear-leaf coefficients, dart tree weights,
//! multi-output leaf values, feature baseline) and their encode/decode helpers, the
//! `ModelMetadata` JSON serde, and the top-level `serialize_model_artifact_v1` /
//! `deserialize_model_artifact_v1` entry points.

use crate::config::Device;
use crate::dro::{DroConfig, DroMetric};
use crate::error::{CoreError, CoreResult};
use crate::leaf::LinearLeaf;
use crate::linear_histogram::MAX_PL_REGRESSORS;
use crate::neutralization::{FactorNeutralizationConfig, NeutralizationKind};
use crate::training_mode::{GradientEmaStats, LrSchedule, MorphConfig};
use crate::validation::validate_model_contract_v1;

pub const MODEL_FORMAT_V1: u32 = 1;
pub const MODEL_BINARY_MAGIC: [u8; 4] = *b"AGBM";
pub const MODEL_BINARY_HEADER_LEN: usize = 16;
pub const MODEL_SECTION_DESCRIPTOR_LEN: usize = 20;
pub const MAX_MODEL_ARTIFACT_SECTIONS: usize = 64;
pub const MAX_MODEL_SECTION_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;
pub const CATEGORICAL_STATE_FORMAT_V1: u32 = 1;
const CATEGORICAL_STATE_HEADER_LEN: usize = 16;
const CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub format_version: u32,
    pub feature_names: Vec<String>,
    pub trained_device: Device,
    /// Objective used to train this model (e.g. "squared_error", "binary_crossentropy").
    /// Defaults to "squared_error" for backward compatibility with older artifacts.
    pub objective: String,
    /// Number of classes for multi-class classification models.
    /// `None` for single-output models (regression, binary classification, ranking).
    pub num_classes: Option<u32>,
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
    NodeDebugStats,
    MultiClassTrees,
    NativeCategoricalSplits,
    MorphMetadata,
    /// Per-stump linear leaf coefficients (intercept + weights + regressor features).
    /// Written alongside the Trees section for `leaf_model="linear"` models.
    LinearLeafCoefficients,
    /// Metadata for DRO-style scalar leaf solving.
    DroMetadata,
    /// Global per-feature training-set means.  Optional section written by
    /// piecewise-linear (`leaf_model="linear"`) artifacts so that SHAP can
    /// compute interventional attributions for linear leaves without needing
    /// the original training data.  Length matches `metadata.feature_names`.
    FeatureBaseline,
    /// v0.9.0+: Per-stump `tree_weight: f32` (one entry per `TrainedStump`,
    /// in stump order). Emitted only when the model was trained with
    /// `BoostingMode::Dart { .. }`. Absent for standard / GOSS artifacts —
    /// readers must default to 1.0 for back-compat.
    DartTreeWeights,
    /// v0.10.0+: Per-stump multi-output leaf values for joint multi-label
    /// trainers. Layout per stump: `Vec<f32>` of length `n_leaves * n_outputs`
    /// (row-major, leaf-major with output as inner axis). The payload also
    /// stores `n_outputs` once at the start. Emitted only when the model was
    /// trained with the joint multi-output entry point; absent for scalar /
    /// linear-leaf / multiclass-softmax artifacts.
    MultiOutputLeafValues,
    /// v0.10.6+: Optional artifact section recording the factor neutralization
    /// configuration that was active during training. Metadata only — prediction
    /// never reads it (neutralization is a training-time transformation on
    /// gradients/targets/split-gains; the trained leaf values already bake in
    /// the projection). Absent when `neutralization_config` is None / inert.
    NeutralizationMetadata,
    Unknown(u32),
}

impl ModelSectionKind {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Trees => 1,
            Self::PredictorLayout => 2,
            Self::ShapAux => 3,
            Self::CategoricalState => 4,
            Self::NodeDebugStats => 5,
            Self::MultiClassTrees => 6,
            Self::NativeCategoricalSplits => 7,
            Self::MorphMetadata => 8,
            Self::LinearLeafCoefficients => 9,
            Self::DroMetadata => 10,
            Self::FeatureBaseline => 11,
            Self::DartTreeWeights => 12,
            Self::MultiOutputLeafValues => 13,
            Self::NeutralizationMetadata => 14,
            Self::Unknown(value) => value,
        }
    }

    pub fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Trees,
            2 => Self::PredictorLayout,
            3 => Self::ShapAux,
            4 => Self::CategoricalState,
            5 => Self::NodeDebugStats,
            6 => Self::MultiClassTrees,
            7 => Self::NativeCategoricalSplits,
            8 => Self::MorphMetadata,
            9 => Self::LinearLeafCoefficients,
            10 => Self::DroMetadata,
            11 => Self::FeatureBaseline,
            12 => Self::DartTreeWeights,
            13 => Self::MultiOutputLeafValues,
            14 => Self::NeutralizationMetadata,
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
pub struct CategoricalStatePayloadV1 {
    pub format_version: u32,
    pub leakage_safe_target_encoding: bool,
    pub categorical_feature_indices: Vec<u32>,
}

/// Payload for the NativeCategoricalSplits artifact section.
/// Stores which features use native categorical splits and the per-stump bitsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCategoricalSplitsPayload {
    /// Feature indices that use native categorical splits (sorted ascending).
    pub native_categorical_feature_indices: Vec<u32>,
    /// Per-stump bitsets: (stump_index, bitset). Only categorical stumps appear here.
    pub stump_bitsets: Vec<(u32, Vec<u8>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredSectionCompatibilityReport {
    pub trees_section_count: usize,
    pub predictor_layout_section_count: usize,
    pub strict_compatible: bool,
    pub legacy_trees_only_compatible: bool,
    pub legacy_compatible: bool,
}

pub fn required_section_compatibility_report(
    sections: &[ModelArtifactSection],
) -> RequiredSectionCompatibilityReport {
    let trees_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::Trees)
        .count();
    let predictor_layout_section_count = sections
        .iter()
        .filter(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
        .count();

    let strict_compatible = trees_section_count == 1 && predictor_layout_section_count == 1;
    let legacy_trees_only_compatible = trees_section_count == 1
        && predictor_layout_section_count == 0
        && sections.len() == 1
        && sections[0].descriptor.kind == ModelSectionKind::Trees;
    let legacy_compatible = strict_compatible || legacy_trees_only_compatible;

    RequiredSectionCompatibilityReport {
        trees_section_count,
        predictor_layout_section_count,
        strict_compatible,
        legacy_trees_only_compatible,
        legacy_compatible,
    }
}

pub fn format_required_section_mode_error(
    report: RequiredSectionCompatibilityReport,
    allow_legacy_trees_only: bool,
) -> String {
    if allow_legacy_trees_only {
        return format!(
            "legacy-compatible mode only supports strict dual-section artifacts or legacy Trees-only artifacts (found Trees={}, PredictorLayout={})",
            report.trees_section_count, report.predictor_layout_section_count
        );
    }
    format!(
        "strict compatibility mode requires exactly one Trees and one PredictorLayout section (found Trees={}, PredictorLayout={})",
        report.trees_section_count, report.predictor_layout_section_count
    )
}

pub fn format_required_section_auto_mode_error(
    report: RequiredSectionCompatibilityReport,
) -> String {
    format!(
        "unable to determine artifact compatibility mode (Trees sections: {}, PredictorLayout sections: {})",
        report.trees_section_count, report.predictor_layout_section_count
    )
}

pub fn encode_categorical_state_payload_v1(
    payload: &CategoricalStatePayloadV1,
) -> CoreResult<Vec<u8>> {
    validate_categorical_state_payload_v1(payload, None)?;

    let feature_count = u32::try_from(payload.categorical_feature_indices.len()).map_err(|_| {
        CoreError::Serialization("categorical feature count exceeds u32::MAX".to_string())
    })?;
    let mut bytes = Vec::with_capacity(
        CATEGORICAL_STATE_HEADER_LEN + payload.categorical_feature_indices.len() * 4,
    );
    bytes.extend_from_slice(&payload.format_version.to_le_bytes());
    let mut flags = 0_u32;
    if payload.leakage_safe_target_encoding {
        flags |= CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING;
    }
    bytes.extend_from_slice(&flags.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    for &feature_index in &payload.categorical_feature_indices {
        bytes.extend_from_slice(&feature_index.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode_categorical_state_payload_v1(bytes: &[u8]) -> CoreResult<CategoricalStatePayloadV1> {
    if bytes.len() < CATEGORICAL_STATE_HEADER_LEN {
        return Err(CoreError::Serialization(format!(
            "categorical state payload length {} is smaller than header length {CATEGORICAL_STATE_HEADER_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    let flags = read_u32_le(bytes, 4)?;
    let feature_count = read_u32_le(bytes, 8)? as usize;
    let _reserved = read_u32_le(bytes, 12)?;

    if flags & !CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING != 0 {
        return Err(CoreError::Serialization(format!(
            "categorical state payload contains unknown flags: {}",
            flags & !CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING
        )));
    }

    let expected_len = CATEGORICAL_STATE_HEADER_LEN
        .checked_add(feature_count.checked_mul(4).ok_or_else(|| {
            CoreError::Serialization("categorical state length overflow".to_string())
        })?)
        .ok_or_else(|| CoreError::Serialization("categorical state length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(CoreError::Serialization(format!(
            "categorical state payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut categorical_feature_indices = Vec::with_capacity(feature_count);
    let mut cursor = CATEGORICAL_STATE_HEADER_LEN;
    for _ in 0..feature_count {
        categorical_feature_indices.push(read_u32_le(bytes, cursor)?);
        cursor += 4;
    }

    let payload = CategoricalStatePayloadV1 {
        format_version,
        leakage_safe_target_encoding: (flags & CATEGORICAL_STATE_FLAG_LEAKAGE_SAFE_TARGET_ENCODING)
            != 0,
        categorical_feature_indices,
    };
    validate_categorical_state_payload_v1(&payload, None)?;
    Ok(payload)
}

pub fn validate_categorical_state_payload_v1(
    payload: &CategoricalStatePayloadV1,
    model_feature_count: Option<usize>,
) -> CoreResult<()> {
    if payload.format_version != CATEGORICAL_STATE_FORMAT_V1 {
        return Err(CoreError::Validation(format!(
            "unsupported categorical state format_version {}, expected {CATEGORICAL_STATE_FORMAT_V1}",
            payload.format_version
        )));
    }
    if payload.categorical_feature_indices.is_empty() {
        return Err(CoreError::Validation(
            "categorical state must include at least one categorical feature index".to_string(),
        ));
    }

    let mut previous = None;
    for &feature_index in &payload.categorical_feature_indices {
        if let Some(previous) = previous
            && feature_index <= previous
        {
            return Err(CoreError::Validation(format!(
                "categorical state feature indices must be strictly increasing (found {feature_index} after {previous})"
            )));
        }
        previous = Some(feature_index);
    }

    if let Some(model_feature_count) = model_feature_count {
        for &feature_index in &payload.categorical_feature_indices {
            if feature_index as usize >= model_feature_count {
                return Err(CoreError::Validation(format!(
                    "categorical state feature index {} is out of bounds for feature_count {}",
                    feature_index, model_feature_count
                )));
            }
        }
    }

    Ok(())
}

pub fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> CoreResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(CoreError::Serialization(format!(
                "model artifact contains duplicate {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

pub fn decode_optional_categorical_state_section_v1(
    sections: &[ModelArtifactSection],
    model_feature_count: usize,
) -> CoreResult<Option<CategoricalStatePayloadV1>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::CategoricalState)?
    else {
        return Ok(None);
    };

    let payload = decode_categorical_state_payload_v1(&section.payload)?;
    validate_categorical_state_payload_v1(&payload, Some(model_feature_count))?;
    Ok(Some(payload))
}

/// Encode native categorical splits payload for artifact serialization.
///
/// Format:
/// - [4 bytes] num_native_categorical_features (u32 LE)
/// - [4 bytes] stump_bitset_count (u32 LE)
/// - [num_native_categorical_features * 4 bytes] feature indices (u32 LE each)
/// - For each stump bitset:
///   - [4 bytes] stump_index (u32 LE)
///   - [2 bytes] bitset_len (u16 LE)
///   - [bitset_len bytes] bitset data
pub fn encode_native_categorical_splits_payload(
    payload: &NativeCategoricalSplitsPayload,
) -> CoreResult<Vec<u8>> {
    let feature_count =
        u32::try_from(payload.native_categorical_feature_indices.len()).map_err(|_| {
            CoreError::Serialization("native cat feature count exceeds u32::MAX".to_string())
        })?;
    let stump_count = u32::try_from(payload.stump_bitsets.len()).map_err(|_| {
        CoreError::Serialization("native cat stump count exceeds u32::MAX".to_string())
    })?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&stump_count.to_le_bytes());
    for &fi in &payload.native_categorical_feature_indices {
        bytes.extend_from_slice(&fi.to_le_bytes());
    }
    for (stump_index, bitset) in &payload.stump_bitsets {
        let bitset_len = u16::try_from(bitset.len())
            .map_err(|_| CoreError::Serialization("bitset length exceeds u16::MAX".to_string()))?;
        bytes.extend_from_slice(&stump_index.to_le_bytes());
        bytes.extend_from_slice(&bitset_len.to_le_bytes());
        bytes.extend_from_slice(bitset);
    }
    Ok(bytes)
}

/// Decode native categorical splits payload from artifact bytes.
pub fn decode_native_categorical_splits_payload(
    bytes: &[u8],
) -> CoreResult<NativeCategoricalSplitsPayload> {
    const HEADER_SIZE: usize = 8; // feature_count(4) + stump_count(4)
    if bytes.len() < HEADER_SIZE {
        return Err(CoreError::Serialization(
            "native categorical splits payload too small for header".to_string(),
        ));
    }

    let feature_count = read_u32_le(bytes, 0)? as usize;
    let stump_count = read_u32_le(bytes, 4)? as usize;

    let feature_section_len = feature_count.checked_mul(4).ok_or_else(|| {
        CoreError::Serialization("native cat feature section length overflow".to_string())
    })?;
    if bytes.len() < HEADER_SIZE + feature_section_len {
        return Err(CoreError::Serialization(
            "native categorical splits payload too small for feature indices".to_string(),
        ));
    }

    let mut native_categorical_feature_indices = Vec::with_capacity(feature_count);
    let mut cursor = HEADER_SIZE;
    for _ in 0..feature_count {
        native_categorical_feature_indices.push(read_u32_le(bytes, cursor)?);
        cursor += 4;
    }

    let mut stump_bitsets = Vec::with_capacity(stump_count);
    for _ in 0..stump_count {
        if cursor + 6 > bytes.len() {
            return Err(CoreError::Serialization(
                "native categorical splits payload truncated in stump bitset header".to_string(),
            ));
        }
        let stump_index = read_u32_le(bytes, cursor)?;
        cursor += 4;
        let bitset_len = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + bitset_len > bytes.len() {
            return Err(CoreError::Serialization(
                "native categorical splits payload truncated in bitset data".to_string(),
            ));
        }
        let bitset = bytes[cursor..cursor + bitset_len].to_vec();
        cursor += bitset_len;
        stump_bitsets.push((stump_index, bitset));
    }

    Ok(NativeCategoricalSplitsPayload {
        native_categorical_feature_indices,
        stump_bitsets,
    })
}

/// Decode optional NativeCategoricalSplits section from model artifact.
pub fn decode_optional_native_categorical_splits_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<NativeCategoricalSplitsPayload>> {
    let Some(section) =
        optional_single_section(sections, ModelSectionKind::NativeCategoricalSplits)?
    else {
        return Ok(None);
    };
    let payload = decode_native_categorical_splits_payload(&section.payload)?;
    Ok(Some(payload))
}

/// Optional artifact section recording the MorphConfig used during training.
/// Metadata only — predictions are deterministic from baked-in leaf values.
/// Section is omitted entirely for non-morph artifacts.
///
/// **Version history.**
///
/// * v1 (v0.4.0+): `config` + `final_iteration` + `final_total`.  Fixed
///   36-byte payload.
/// * v2 (v0.7.3+): appends a length-prefixed `ema_stats: Vec<GradientEmaStats>`
///   so MorphBoost warm-starts can resume with the EMA state from the
///   previous fit rather than restarting it cold.  Legacy v1 artifacts
///   decode with `ema_stats = Vec::new()` and the warm-start path
///   falls back to a cold EMA (legacy v0.7.1/v0.7.2 behavior).
#[derive(Debug, Clone, PartialEq)]
pub struct MorphMetadataPayload {
    pub config: MorphConfig,
    pub final_iteration: u32,
    pub final_total: u32,
    /// EMA snapshot captured at training-finalize time.  Empty when the
    /// payload was decoded from a pre-v0.7.3 (version 1) artifact, in
    /// which case warm-start initializes the EMA cold.  Indexed by
    /// class for multiclass models (length 1 for single-output).
    pub ema_stats: Vec<GradientEmaStats>,
}

pub fn encode_morph_metadata_payload(payload: &MorphMetadataPayload) -> Vec<u8> {
    // v2 layout: 36 bytes header (same as v1) + 4 bytes count +
    // 12 bytes per GradientEmaStats (mean, std, alpha as little-endian f32).
    let ema_section_len = 4 + payload.ema_stats.len() * 12;
    let mut buf = Vec::with_capacity(36 + ema_section_len);
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_rate.to_le_bytes());
    buf.extend_from_slice(&payload.config.evolution_pressure.to_le_bytes());
    buf.extend_from_slice(&payload.config.morph_warmup_iters.to_le_bytes());
    buf.extend_from_slice(&payload.config.info_score_weight.to_le_bytes());
    buf.extend_from_slice(&payload.config.depth_penalty_base.to_le_bytes());
    buf.push(payload.config.balance_penalty as u8);
    let (kind, warmup_frac) = match payload.config.lr_schedule {
        LrSchedule::Constant => (0u8, 0.0f32),
        LrSchedule::WarmupCosine { warmup_frac } => (1u8, warmup_frac),
    };
    buf.push(kind);
    buf.extend_from_slice(&warmup_frac.to_le_bytes());
    buf.extend_from_slice(&payload.final_iteration.to_le_bytes());
    buf.extend_from_slice(&payload.final_total.to_le_bytes());
    // v2 EMA tail.
    let ema_count = payload.ema_stats.len() as u32;
    buf.extend_from_slice(&ema_count.to_le_bytes());
    for stats in &payload.ema_stats {
        buf.extend_from_slice(&stats.mean.to_le_bytes());
        buf.extend_from_slice(&stats.std.to_le_bytes());
        buf.extend_from_slice(&stats.alpha.to_le_bytes());
    }
    buf
}

pub fn decode_optional_morph_metadata_section(bytes: &[u8]) -> CoreResult<MorphMetadataPayload> {
    if bytes.len() < 36 {
        return Err(CoreError::Validation(
            "morph metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 && version != 2 {
        return Err(CoreError::Validation(format!(
            "unsupported morph metadata version: {version}"
        )));
    }
    let mut o = 2usize;
    macro_rules! read_f32 {
        () => {{
            let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            o += 4;
            v
        }};
    }
    macro_rules! read_u32 {
        () => {{
            let v = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            o += 4;
            v
        }};
    }
    let morph_rate = read_f32!();
    let evolution_pressure = read_f32!();
    let morph_warmup_iters = read_u32!();
    let info_score_weight = read_f32!();
    let depth_penalty_base = read_f32!();
    let balance_penalty = bytes[o] != 0;
    o += 1;
    let lr_kind = bytes[o];
    o += 1;
    let warmup_frac = read_f32!();
    let final_iteration = read_u32!();
    let final_total = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    o += 4;
    let lr_schedule = match lr_kind {
        0 => LrSchedule::Constant,
        1 => LrSchedule::WarmupCosine { warmup_frac },
        _ => {
            return Err(CoreError::Validation(format!(
                "unknown lr_schedule kind: {lr_kind}"
            )));
        }
    };
    // v2 tail: optional EMA stats.  v1 artifacts have no tail and
    // decode with `ema_stats = Vec::new()`.
    let ema_stats = if version >= 2 && o + 4 <= bytes.len() {
        let count =
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        o += 4;
        let expected_tail = count.checked_mul(12).ok_or_else(|| {
            CoreError::Validation("morph metadata ema count overflow".to_string())
        })?;
        if o + expected_tail > bytes.len() {
            return Err(CoreError::Validation(format!(
                "morph metadata ema tail truncated: expected {} bytes after header, got {}",
                expected_tail,
                bytes.len() - o
            )));
        }
        let mut stats = Vec::with_capacity(count);
        for _ in 0..count {
            let mean = read_f32!();
            let std = read_f32!();
            let alpha = read_f32!();
            stats.push(GradientEmaStats { mean, std, alpha });
        }
        stats
    } else {
        Vec::new()
    };
    Ok(MorphMetadataPayload {
        config: MorphConfig {
            morph_rate,
            evolution_pressure,
            morph_warmup_iters,
            info_score_weight,
            depth_penalty_base,
            balance_penalty,
            lr_schedule,
        },
        final_iteration,
        final_total,
        ema_stats,
    })
}

/// Decode an optional MorphMetadata section from a parsed model artifact.
/// Returns `None` if no such section exists (non-morph artifact).
pub fn decode_optional_morph_metadata_artifact_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<MorphMetadataPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::MorphMetadata)? else {
        return Ok(None);
    };
    let payload = decode_optional_morph_metadata_section(&section.payload)?;
    Ok(Some(payload))
}

// ── DRO leaf-solver metadata section ────────────────────────────────────────

/// Optional artifact section recording the DRO leaf solver configuration.
/// Metadata only — prediction uses baked scalar leaf values.
#[derive(Debug, Clone, PartialEq)]
pub struct DroMetadataPayload {
    pub config: DroConfig,
}

pub fn encode_dro_metadata_payload(payload: &DroMetadataPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(7);
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&payload.config.radius.to_le_bytes());
    buf.push(match payload.config.metric {
        DroMetric::Wasserstein => 0,
    });
    buf
}

pub fn decode_dro_metadata_payload(bytes: &[u8]) -> CoreResult<DroMetadataPayload> {
    if bytes.len() < 7 {
        return Err(CoreError::Validation(
            "dro metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported dro metadata version: {version}"
        )));
    }
    let radius = f32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
    if !radius.is_finite() || radius < 0.0 {
        return Err(CoreError::Validation(
            "dro metadata radius must be finite and >= 0".to_string(),
        ));
    }
    let metric = match bytes[6] {
        0 => DroMetric::Wasserstein,
        other => {
            return Err(CoreError::Validation(format!(
                "unsupported dro metadata metric kind: {other}"
            )));
        }
    };
    Ok(DroMetadataPayload {
        config: DroConfig { radius, metric },
    })
}

pub fn decode_optional_dro_metadata_artifact_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<DroMetadataPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::DroMetadata)? else {
        return Ok(None);
    };
    Ok(Some(decode_dro_metadata_payload(&section.payload)?))
}

// ── Factor-neutralization metadata section ────────────────────────────────

/// Optional artifact section recording the factor neutralization configuration.
/// Metadata only — prediction never reads it. Mirrors `DroMetadataPayload`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeutralizationMetadataPayload {
    pub config: FactorNeutralizationConfig,
}

pub fn encode_neutralization_metadata_payload(payload: &NeutralizationMetadataPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11);
    buf.extend_from_slice(&1u16.to_le_bytes()); // version
    buf.push(match payload.config.kind {
        NeutralizationKind::None => 0,
        NeutralizationKind::PreTarget => 1,
        NeutralizationKind::PerRoundGradient => 2,
        NeutralizationKind::SplitPenalty => 3,
    });
    buf.extend_from_slice(&payload.config.ridge_lambda.to_le_bytes());
    buf.extend_from_slice(&payload.config.split_penalty.to_le_bytes());
    buf
}

pub fn decode_neutralization_metadata_payload(
    bytes: &[u8],
) -> CoreResult<NeutralizationMetadataPayload> {
    if bytes.len() < 11 {
        return Err(CoreError::Validation(
            "neutralization metadata section too short".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported neutralization metadata version: {version}"
        )));
    }
    let kind = match bytes[2] {
        0 => NeutralizationKind::None,
        1 => NeutralizationKind::PreTarget,
        2 => NeutralizationKind::PerRoundGradient,
        3 => NeutralizationKind::SplitPenalty,
        other => {
            return Err(CoreError::Validation(format!(
                "unsupported neutralization metadata kind: {other}"
            )));
        }
    };
    let ridge_lambda = f32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(CoreError::Validation(
            "neutralization metadata ridge_lambda must be finite and >= 0".to_string(),
        ));
    }
    let split_penalty = f32::from_le_bytes([bytes[7], bytes[8], bytes[9], bytes[10]]);
    if !split_penalty.is_finite() || split_penalty < 0.0 {
        return Err(CoreError::Validation(
            "neutralization metadata split_penalty must be finite and >= 0".to_string(),
        ));
    }
    Ok(NeutralizationMetadataPayload {
        config: FactorNeutralizationConfig {
            kind,
            ridge_lambda,
            split_penalty,
        },
    })
}

pub fn decode_optional_neutralization_metadata_artifact_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<NeutralizationMetadataPayload>> {
    let Some(section) =
        optional_single_section(sections, ModelSectionKind::NeutralizationMetadata)?
    else {
        return Ok(None);
    };
    Ok(Some(decode_neutralization_metadata_payload(
        &section.payload,
    )?))
}

// ── Linear-leaf coefficients section ─────────────────────────────────────────

/// One stump's linear-leaf entries inside the `LinearLeafCoefficients` section.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeafEntry {
    pub stump_idx: u32,
    pub left_leaf: Option<LinearLeaf>,
    pub right_leaf: Option<LinearLeaf>,
}

/// Payload for `ModelSectionKind::LinearLeafCoefficients`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeafCoefficientsPayload {
    pub entries: Vec<LinearLeafEntry>,
}

/// Encode a `LinearLeafCoefficientsPayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=2] [u32 entry_count]
/// For each entry:
///   [u32 stump_idx] [u8 flags]
///   if flags & 1: [u8 d] [f32 intercept] [d × f32 weights] [d × u32 regressor_features]
///               [d × f32 feature_means] [d × f32 feature_inv_stds]
///   if flags & 2: same for right leaf
/// ```
pub fn encode_linear_leaf_coefficients_payload(payload: &LinearLeafCoefficientsPayload) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&2u32.to_le_bytes()); // version
    buf.extend_from_slice(&(payload.entries.len() as u32).to_le_bytes());
    for entry in &payload.entries {
        buf.extend_from_slice(&entry.stump_idx.to_le_bytes());
        let flags: u8 =
            (entry.left_leaf.is_some() as u8) | ((entry.right_leaf.is_some() as u8) << 1);
        buf.push(flags);

        let write_leaf = |buf: &mut Vec<u8>, leaf: &LinearLeaf| {
            let d = leaf.weights.len().min(MAX_PL_REGRESSORS);
            buf.push(d as u8);
            buf.extend_from_slice(&leaf.intercept.to_le_bytes());
            for i in 0..d {
                buf.extend_from_slice(&leaf.weights[i].to_le_bytes());
            }
            for i in 0..d {
                let feat = *leaf.regressor_features.get(i).unwrap_or(&0);
                buf.extend_from_slice(&feat.to_le_bytes());
            }
            for i in 0..d {
                let mean = *leaf.feature_means.get(i).unwrap_or(&0.0);
                buf.extend_from_slice(&mean.to_le_bytes());
            }
            for i in 0..d {
                let inv_std = *leaf.feature_inv_stds.get(i).unwrap_or(&1.0);
                buf.extend_from_slice(&inv_std.to_le_bytes());
            }
        };
        if let Some(ref ll) = entry.left_leaf {
            write_leaf(&mut buf, ll);
        }
        if let Some(ref rl) = entry.right_leaf {
            write_leaf(&mut buf, rl);
        }
    }
    buf
}

/// Decode a `LinearLeafCoefficientsPayload` from raw section bytes.
pub fn decode_linear_leaf_coefficients_payload(
    bytes: &[u8],
) -> CoreResult<LinearLeafCoefficientsPayload> {
    if bytes.len() < 8 {
        return Err(CoreError::Validation(
            "linear leaf coefficients section too short".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 && version != 2 {
        return Err(CoreError::Validation(format!(
            "unsupported linear leaf coefficients version: {version}"
        )));
    }
    let entry_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut o = 8usize;
    let mut entries = Vec::with_capacity(entry_count);

    let read_u32 = |bytes: &[u8], o: &mut usize| -> CoreResult<u32> {
        if *o + 4 > bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let v = u32::from_le_bytes([bytes[*o], bytes[*o + 1], bytes[*o + 2], bytes[*o + 3]]);
        *o += 4;
        Ok(v)
    };
    let read_f32 = |bytes: &[u8], o: &mut usize| -> CoreResult<f32> {
        if *o + 4 > bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let v = f32::from_le_bytes([bytes[*o], bytes[*o + 1], bytes[*o + 2], bytes[*o + 3]]);
        *o += 4;
        Ok(v)
    };

    for _ in 0..entry_count {
        let stump_idx = read_u32(bytes, &mut o)?;
        if o >= bytes.len() {
            return Err(CoreError::Validation(
                "unexpected end of linear leaf coefficients data".to_string(),
            ));
        }
        let flags = bytes[o];
        o += 1;

        let read_leaf = |bytes: &[u8], o: &mut usize| -> CoreResult<LinearLeaf> {
            if *o >= bytes.len() {
                return Err(CoreError::Validation(
                    "unexpected end reading linear leaf".to_string(),
                ));
            }
            let d = bytes[*o] as usize;
            *o += 1;
            let intercept = read_f32(bytes, o)?;
            let mut weights = Vec::with_capacity(d);
            for _ in 0..d {
                weights.push(read_f32(bytes, o)?);
            }
            let mut regressor_features = Vec::with_capacity(d);
            for _ in 0..d {
                regressor_features.push(read_u32(bytes, o)?);
            }
            let (feature_means, feature_inv_stds) = if version >= 2 {
                let mut means = Vec::with_capacity(d);
                for _ in 0..d {
                    means.push(read_f32(bytes, o)?);
                }
                let mut inv_stds = Vec::with_capacity(d);
                for _ in 0..d {
                    inv_stds.push(read_f32(bytes, o)?);
                }
                (means, inv_stds)
            } else {
                (vec![0.0; d], vec![1.0; d])
            };
            Ok(LinearLeaf::scaled(
                intercept,
                weights,
                regressor_features,
                feature_means,
                feature_inv_stds,
            ))
        };

        let left_leaf = if flags & 1 != 0 {
            Some(read_leaf(bytes, &mut o)?)
        } else {
            None
        };
        let right_leaf = if flags & 2 != 0 {
            Some(read_leaf(bytes, &mut o)?)
        } else {
            None
        };
        entries.push(LinearLeafEntry {
            stump_idx,
            left_leaf,
            right_leaf,
        });
    }

    Ok(LinearLeafCoefficientsPayload { entries })
}

/// Decode an optional `LinearLeafCoefficients` section from parsed artifact sections.
/// Returns `None` if no such section exists (constant-leaf artifact).
pub fn decode_optional_linear_leaf_coefficients_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<LinearLeafCoefficientsPayload>> {
    let Some(section) =
        optional_single_section(sections, ModelSectionKind::LinearLeafCoefficients)?
    else {
        return Ok(None);
    };
    let payload = decode_linear_leaf_coefficients_payload(&section.payload)?;
    Ok(Some(payload))
}

// ── DART tree weights section ────────────────────────────────────────────────

/// Payload for `ModelSectionKind::DartTreeWeights`.
///
/// One entry per `TrainedStump`, in stump order. Emitted only by
/// DART-trained models; readers must default to `tree_weight = 1.0`
/// for stumps not covered by this section (back-compat with v0.8.0).
#[derive(Debug, Clone, PartialEq)]
pub struct DartTreeWeightsPayload {
    /// One entry per stump, in stump order. Length must equal
    /// `model.stumps.len()` at load time.
    pub weights: Vec<f32>,
}

/// Encode a `DartTreeWeightsPayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=1] [u32 weight_count] [weight_count × f32 LE weights]
/// ```
pub fn encode_dart_tree_weights_payload(payload: &DartTreeWeightsPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + 4 * payload.weights.len());
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(payload.weights.len() as u32).to_le_bytes());
    for w in &payload.weights {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    buf
}

/// Decode a `DartTreeWeightsPayload` from raw section bytes.
pub fn decode_dart_tree_weights_payload(bytes: &[u8]) -> CoreResult<DartTreeWeightsPayload> {
    if bytes.len() < 8 {
        return Err(CoreError::Validation(
            "DartTreeWeights section too short for header".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported DartTreeWeights version: {version}"
        )));
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + 4 * count;
    if bytes.len() != expected {
        return Err(CoreError::Validation(format!(
            "DartTreeWeights payload length {} != expected {} ({} weights)",
            bytes.len(),
            expected,
            count
        )));
    }
    let mut weights = Vec::with_capacity(count);
    for i in 0..count {
        let off = 8 + 4 * i;
        weights.push(f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]));
    }
    Ok(DartTreeWeightsPayload { weights })
}

/// Decode an optional `DartTreeWeights` section. Returns `Ok(None)` for
/// pre-v0.9.0 artifacts (no section present) — the caller must default
/// to `tree_weight = 1.0` for every stump in that case.
pub fn decode_optional_dart_tree_weights_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<DartTreeWeightsPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::DartTreeWeights)?
    else {
        return Ok(None);
    };
    let payload = decode_dart_tree_weights_payload(&section.payload)?;
    Ok(Some(payload))
}

// ── Multi-output leaf values section ─────────────────────────────────────────

/// Payload for `ModelSectionKind::MultiOutputLeafValues`.
///
/// Stores K leaf values per leaf for joint multi-output trainers. One outer
/// `Vec<f32>` per stump (in stump order); inner length must equal
/// `n_leaves(stump) * n_outputs`, row-major with leaf-major outer axis and
/// output-major inner axis (so `inner[leaf_idx * n_outputs + k]` is leaf
/// `leaf_idx`'s value for output `k`).
#[derive(Debug, Clone, PartialEq)]
pub struct MultiOutputLeafValuesPayload {
    pub n_outputs: u32,
    /// One entry per stump in tree order. Each inner `Vec<f32>` has length
    /// `n_leaves * n_outputs`.
    pub per_stump_leaf_values: Vec<Vec<f32>>,
}

/// Encode a `MultiOutputLeafValuesPayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=1] [u32 n_outputs] [u32 n_stumps]
/// For each stump:
///   [u32 len] [len × f32 LE values]
/// ```
pub fn encode_multi_output_leaf_values_payload(payload: &MultiOutputLeafValuesPayload) -> Vec<u8> {
    let total_values: usize = payload.per_stump_leaf_values.iter().map(|v| v.len()).sum();
    let mut buf =
        Vec::with_capacity(12 + 4 * payload.per_stump_leaf_values.len() + 4 * total_values);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&payload.n_outputs.to_le_bytes());
    buf.extend_from_slice(&(payload.per_stump_leaf_values.len() as u32).to_le_bytes());
    for stump in &payload.per_stump_leaf_values {
        buf.extend_from_slice(&(stump.len() as u32).to_le_bytes());
        for &v in stump {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Decode a `MultiOutputLeafValuesPayload` from raw section bytes.
pub fn decode_multi_output_leaf_values_payload(
    bytes: &[u8],
) -> CoreResult<MultiOutputLeafValuesPayload> {
    if bytes.len() < 12 {
        return Err(CoreError::Validation(
            "MultiOutputLeafValues section too short for header".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported MultiOutputLeafValues version: {version}"
        )));
    }
    let n_outputs = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let n_stumps = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let mut cursor = 12usize;
    let mut per_stump_leaf_values = Vec::with_capacity(n_stumps);
    for _ in 0..n_stumps {
        if cursor + 4 > bytes.len() {
            return Err(CoreError::Validation(
                "MultiOutputLeafValues: truncated stump length".to_string(),
            ));
        }
        let len = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]) as usize;
        cursor += 4;
        if cursor + len * 4 > bytes.len() {
            return Err(CoreError::Validation(
                "MultiOutputLeafValues: truncated leaf values".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let v = f32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            cursor += 4;
            values.push(v);
        }
        per_stump_leaf_values.push(values);
    }
    Ok(MultiOutputLeafValuesPayload {
        n_outputs,
        per_stump_leaf_values,
    })
}

/// Decode an optional `MultiOutputLeafValues` section. Returns `Ok(None)` for
/// pre-v0.10.0 artifacts (no section present) — only joint multi-output
/// trainers emit this section.
pub fn decode_optional_multi_output_leaf_values_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<MultiOutputLeafValuesPayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::MultiOutputLeafValues)?
    else {
        return Ok(None);
    };
    let payload = decode_multi_output_leaf_values_payload(&section.payload)?;
    Ok(Some(payload))
}

// ── Feature baseline section ─────────────────────────────────────────────────

/// Payload for `ModelSectionKind::FeatureBaseline`.
///
/// Stores the global (training-set marginal) mean for each feature.  Length
/// matches `ModelMetadata::feature_names`.  Used by SHAP for piecewise-linear
/// leaves so that linear-leaf contributions can be decomposed into a
/// path-attributed expected value plus per-feature deviations
/// `wj * (z_j(row_raw_j) - z_j(feature_means_raw_j))`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBaselinePayload {
    pub feature_means: Vec<f32>,
}

/// Encode a `FeatureBaselinePayload` to bytes.
///
/// Layout:
/// ```text
/// [u32 version=1] [u32 feature_count] [feature_count × f32 means]
/// ```
pub fn encode_feature_baseline_payload(payload: &FeatureBaselinePayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + payload.feature_means.len() * 4);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&(payload.feature_means.len() as u32).to_le_bytes());
    for m in &payload.feature_means {
        buf.extend_from_slice(&m.to_le_bytes());
    }
    buf
}

/// Decode a `FeatureBaselinePayload` from raw section bytes.
pub fn decode_feature_baseline_payload(bytes: &[u8]) -> CoreResult<FeatureBaselinePayload> {
    if bytes.len() < 8 {
        return Err(CoreError::Validation(
            "feature baseline section too short".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if version != 1 {
        return Err(CoreError::Validation(format!(
            "unsupported feature baseline version: {version}"
        )));
    }
    let feature_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + feature_count * 4;
    if bytes.len() < expected {
        return Err(CoreError::Validation(format!(
            "feature baseline section too short: need {expected} bytes, got {}",
            bytes.len()
        )));
    }
    let mut feature_means = Vec::with_capacity(feature_count);
    let mut o = 8usize;
    for _ in 0..feature_count {
        let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        feature_means.push(v);
        o += 4;
    }
    Ok(FeatureBaselinePayload { feature_means })
}

/// Decode an optional `FeatureBaseline` section from parsed artifact sections.
/// Returns `None` if no such section exists (legacy artifact without baseline).
pub fn decode_optional_feature_baseline_section(
    sections: &[ModelArtifactSection],
) -> CoreResult<Option<FeatureBaselinePayload>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::FeatureBaseline)?
    else {
        return Ok(None);
    };
    let payload = decode_feature_baseline_payload(&section.payload)?;
    Ok(Some(payload))
}
pub fn serialize_metadata_json(metadata: &ModelMetadata) -> String {
    let feature_names = metadata
        .feature_names
        .iter()
        .map(|name| format!("\"{}\"", escape_json_string(name)))
        .collect::<Vec<_>>()
        .join(",");

    let num_classes_fragment = match metadata.num_classes {
        Some(k) => format!(",\"num_classes\":{k}"),
        None => String::new(),
    };

    format!(
        "{{\"format_version\":{},\"feature_names\":[{}],\"trained_device\":\"{}\",\"objective\":\"{}\"{}}}",
        metadata.format_version,
        feature_names,
        metadata.trained_device.as_metadata_label(),
        escape_json_string(&metadata.objective),
        num_classes_fragment
    )
}

pub fn deserialize_metadata_json(input: &str) -> CoreResult<ModelMetadata> {
    let compact = compact_json(input)?;
    let mut index = 0_usize;
    let mut format_version = None;
    let mut feature_names = None;
    let mut trained_device_raw = None;
    let mut objective = None;
    let mut num_classes = None;

    index = consume_literal(&compact, index, "{")?;
    if compact[index..].starts_with('}') {
        index += 1;
    } else {
        loop {
            let (key, next_index) = parse_quoted_string(&compact, index)?;
            index = consume_literal(&compact, next_index, ":")?;
            match key.as_str() {
                "format_version" => {
                    if format_version.is_some() {
                        return Err(CoreError::Serialization(
                            "duplicate metadata field 'format_version'".to_string(),
                        ));
                    }
                    let (value, next_index) = parse_u32(&compact, index)?;
                    format_version = Some(value);
                    index = next_index;
                }
                "feature_names" => {
                    if feature_names.is_some() {
                        return Err(CoreError::Serialization(
                            "duplicate metadata field 'feature_names'".to_string(),
                        ));
                    }
                    let (value, next_index) = parse_string_array(&compact, index)?;
                    feature_names = Some(value);
                    index = next_index;
                }
                "trained_device" => {
                    if trained_device_raw.is_some() {
                        return Err(CoreError::Serialization(
                            "duplicate metadata field 'trained_device'".to_string(),
                        ));
                    }
                    let (value, next_index) = parse_quoted_string(&compact, index)?;
                    trained_device_raw = Some(value);
                    index = next_index;
                }
                "objective" => {
                    if objective.is_some() {
                        return Err(CoreError::Serialization(
                            "duplicate metadata field 'objective'".to_string(),
                        ));
                    }
                    let (value, next_index) = parse_quoted_string(&compact, index)?;
                    objective = Some(value);
                    index = next_index;
                }
                "num_classes" => {
                    if num_classes.is_some() {
                        return Err(CoreError::Serialization(
                            "duplicate metadata field 'num_classes'".to_string(),
                        ));
                    }
                    let (value, next_index) = parse_u32(&compact, index)?;
                    num_classes = Some(value);
                    index = next_index;
                }
                _ => {
                    index = skip_json_value(&compact, index, 0)?;
                }
            }

            if compact[index..].starts_with(',') {
                index += 1;
                continue;
            }
            if compact[index..].starts_with('}') {
                index += 1;
                break;
            }
            return Err(CoreError::Serialization(format!(
                "expected ',' or '}}' at index {index}"
            )));
        }
    }

    if index != compact.len() {
        return Err(CoreError::Serialization(
            "unexpected trailing content in metadata json".to_string(),
        ));
    }

    let format_version = format_version.ok_or_else(|| {
        CoreError::Serialization("metadata missing required field 'format_version'".to_string())
    })?;
    let feature_names = feature_names.ok_or_else(|| {
        CoreError::Serialization("metadata missing required field 'feature_names'".to_string())
    })?;
    let trained_device_raw = trained_device_raw.ok_or_else(|| {
        CoreError::Serialization("metadata missing required field 'trained_device'".to_string())
    })?;

    Ok(ModelMetadata {
        format_version,
        feature_names,
        trained_device: Device::parse_metadata_label(&trained_device_raw)?,
        objective: objective.unwrap_or_else(|| "squared_error".to_string()),
        num_classes,
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

    if sections.len() > MAX_MODEL_ARTIFACT_SECTIONS {
        return Err(CoreError::Serialization(format!(
            "section_count {} exceeds maximum {MAX_MODEL_ARTIFACT_SECTIONS}",
            sections.len()
        )));
    }
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
        if payload.len() as u64 > MAX_MODEL_SECTION_PAYLOAD_BYTES {
            return Err(CoreError::Serialization(format!(
                "section length {} exceeds maximum {MAX_MODEL_SECTION_PAYLOAD_BYTES}",
                payload.len()
            )));
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
    if section_count > MAX_MODEL_ARTIFACT_SECTIONS {
        return Err(CoreError::Serialization(format!(
            "section_count {section_count} exceeds maximum {MAX_MODEL_ARTIFACT_SECTIONS}"
        )));
    }
    let metadata_json_len = header.metadata_json_len as usize;
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

const MAX_METADATA_UNKNOWN_FIELD_DEPTH: usize = 64;

fn skip_json_value(input: &str, index: usize, depth: usize) -> CoreResult<usize> {
    if depth > MAX_METADATA_UNKNOWN_FIELD_DEPTH {
        return Err(CoreError::Serialization(format!(
            "metadata json nesting exceeds maximum depth {MAX_METADATA_UNKNOWN_FIELD_DEPTH}"
        )));
    }
    if index >= input.len() {
        return Err(CoreError::Serialization(format!(
            "expected json value at index {index}"
        )));
    }

    match input.as_bytes()[index] {
        b'"' => {
            let (_, next_index) = parse_quoted_string(input, index)?;
            Ok(next_index)
        }
        b'{' => skip_json_object(input, index, depth + 1),
        b'[' => skip_json_array(input, index, depth + 1),
        b't' => consume_literal(input, index, "true"),
        b'f' => consume_literal(input, index, "false"),
        b'n' => consume_literal(input, index, "null"),
        b'-' | b'0'..=b'9' => skip_json_number(input, index),
        _ => Err(CoreError::Serialization(format!(
            "expected json value at index {index}"
        ))),
    }
}

fn skip_json_object(input: &str, mut index: usize, depth: usize) -> CoreResult<usize> {
    index = consume_literal(input, index, "{")?;
    if input[index..].starts_with('}') {
        return Ok(index + 1);
    }

    loop {
        let (_, next_index) = parse_quoted_string(input, index)?;
        index = consume_literal(input, next_index, ":")?;
        index = skip_json_value(input, index, depth)?;

        if input[index..].starts_with(',') {
            index += 1;
            continue;
        }
        if input[index..].starts_with('}') {
            return Ok(index + 1);
        }
        return Err(CoreError::Serialization(format!(
            "expected ',' or '}}' at index {index}"
        )));
    }
}

fn skip_json_array(input: &str, mut index: usize, depth: usize) -> CoreResult<usize> {
    index = consume_literal(input, index, "[")?;
    if input[index..].starts_with(']') {
        return Ok(index + 1);
    }

    loop {
        index = skip_json_value(input, index, depth)?;

        if input[index..].starts_with(',') {
            index += 1;
            continue;
        }
        if input[index..].starts_with(']') {
            return Ok(index + 1);
        }
        return Err(CoreError::Serialization(format!(
            "expected ',' or ']' at index {index}"
        )));
    }
}

fn skip_json_number(input: &str, mut index: usize) -> CoreResult<usize> {
    if input[index..].starts_with('-') {
        index += 1;
    }

    let integer_start = index;
    while let Some(byte) = input.as_bytes().get(index) {
        if !byte.is_ascii_digit() {
            break;
        }
        index += 1;
    }
    if integer_start == index {
        return Err(CoreError::Serialization(format!(
            "expected json number at index {integer_start}"
        )));
    }

    if input[index..].starts_with('.') {
        index += 1;
        let fraction_start = index;
        while let Some(byte) = input.as_bytes().get(index) {
            if !byte.is_ascii_digit() {
                break;
            }
            index += 1;
        }
        if fraction_start == index {
            return Err(CoreError::Serialization(format!(
                "expected json number fraction at index {fraction_start}"
            )));
        }
    }

    if input[index..].starts_with('e') || input[index..].starts_with('E') {
        index += 1;
        if input[index..].starts_with('+') || input[index..].starts_with('-') {
            index += 1;
        }
        let exponent_start = index;
        while let Some(byte) = input.as_bytes().get(index) {
            if !byte.is_ascii_digit() {
                break;
            }
            index += 1;
        }
        if exponent_start == index {
            return Err(CoreError::Serialization(format!(
                "expected json number exponent at index {exponent_start}"
            )));
        }
    }

    Ok(index)
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

fn read_u32_le(bytes: &[u8], offset: usize) -> CoreResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| CoreError::Serialization("u32 read overflow".to_string()))?;
    if end > bytes.len() {
        return Err(CoreError::Serialization(format!(
            "u32 read out of bounds at offset {offset}"
        )));
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}
