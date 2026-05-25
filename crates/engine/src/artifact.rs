//! Artifact serialization/deserialization helpers for `TrainedModel`.
//!
//! Extracted from `lib.rs` in refactor Task 1.16.

use alloygbm_core::{
    LeafValue, MODEL_FORMAT_V1, ModelArtifactSection, ModelSectionKind, NodeStats, SplitCandidate,
    required_section_compatibility_report,
};

use crate::error::{EngineError, EngineResult};
use crate::trained_model::TrainedModel;
use crate::types::{
    ArtifactCompatibilityMode, ArtifactCompatibilityReport, NodeDebugStats, TrainedStump,
};

pub(crate) fn required_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> EngineResult<&ModelArtifactSection> {
    optional_single_section(sections, kind)?.ok_or_else(|| {
        EngineError::ContractViolation(format!(
            "model artifact missing required {:?} section",
            kind
        ))
    })
}

fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> EngineResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(EngineError::ContractViolation(format!(
                "model artifact contains duplicate required {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

pub(crate) fn artifact_compatibility_report_from_sections(
    sections: &[ModelArtifactSection],
) -> ArtifactCompatibilityReport {
    let report = required_section_compatibility_report(sections);
    let recommended_mode = if report.strict_compatible {
        Some(ArtifactCompatibilityMode::Strict)
    } else if report.legacy_trees_only_compatible {
        Some(ArtifactCompatibilityMode::AllowLegacyTreesOnly)
    } else {
        None
    };

    ArtifactCompatibilityReport {
        trees_section_count: report.trees_section_count,
        predictor_layout_section_count: report.predictor_layout_section_count,
        strict_compatible: report.strict_compatible,
        legacy_trees_only_compatible: report.legacy_trees_only_compatible,
        legacy_compatible: report.legacy_compatible,
        recommended_mode,
    }
}

pub(crate) fn resolve_predictor_layout(
    sections: &[ModelArtifactSection],
    metadata_feature_count: usize,
    compatibility_mode: ArtifactCompatibilityMode,
) -> EngineResult<PredictorLayoutPayload> {
    if let Some(section) = optional_single_section(sections, ModelSectionKind::PredictorLayout)? {
        return decode_predictor_layout_payload(&section.payload);
    }

    if compatibility_mode == ArtifactCompatibilityMode::AllowLegacyTreesOnly
        && sections.len() == 1
        && sections[0].descriptor.kind == ModelSectionKind::Trees
    {
        // Compatibility path for v0.0.4 legacy payloads that only carried Trees.
        return Ok(PredictorLayoutPayload {
            feature_count: metadata_feature_count,
        });
    }

    Err(EngineError::ContractViolation(
        "model artifact missing required PredictorLayout section".to_string(),
    ))
}

pub(crate) fn encode_predictor_layout_payload(model: &TrainedModel) -> EngineResult<Vec<u8>> {
    let feature_count = u32::try_from(model.feature_count).map_err(|_| {
        EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
    })?;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&THRESHOLD_MODE_BIN_INDEX.to_le_bytes());
    Ok(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PredictorLayoutPayload {
    pub(crate) feature_count: usize,
}

pub(crate) fn decode_predictor_layout_payload(
    bytes: &[u8],
) -> EngineResult<PredictorLayoutPayload> {
    const LAYOUT_LEN: usize = 12;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
    if bytes.len() != LAYOUT_LEN {
        return Err(EngineError::ContractViolation(format!(
            "predictor layout payload length {} does not match expected {LAYOUT_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported predictor layout format version {format_version}"
        )));
    }

    let feature_count = read_u32_le(bytes, 4)? as usize;
    let threshold_mode = read_u32_le(bytes, 8)?;
    if threshold_mode != THRESHOLD_MODE_BIN_INDEX {
        return Err(EngineError::ContractViolation(format!(
            "unsupported predictor layout threshold mode {threshold_mode}"
        )));
    }

    Ok(PredictorLayoutPayload { feature_count })
}

pub(crate) fn encode_node_debug_stats_payload(
    node_debug_stats: &[NodeDebugStats],
) -> EngineResult<Vec<u8>> {
    let record_count = u32::try_from(node_debug_stats.len()).map_err(|_| {
        EngineError::ContractViolation("node debug stats count exceeds u32::MAX".to_string())
    })?;

    let mut bytes = Vec::with_capacity(8 + node_debug_stats.len() * 40);
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    for record in node_debug_stats {
        bytes.extend_from_slice(&record.node_id.to_le_bytes());
        bytes.extend_from_slice(&record.feature_index.to_le_bytes());
        bytes.extend_from_slice(&record.threshold_bin.to_le_bytes());
        let flags: u16 = if record.default_left { 1 } else { 0 };
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&record.gain.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.grad_sum.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.hess_sum.to_le_bytes());
        bytes.extend_from_slice(&record.left_stats.row_count.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.grad_sum.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.hess_sum.to_le_bytes());
        bytes.extend_from_slice(&record.right_stats.row_count.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) fn decode_node_debug_stats_payload(bytes: &[u8]) -> EngineResult<Vec<NodeDebugStats>> {
    const HEADER_SIZE: usize = 8;
    const RECORD_SIZE: usize = 40;
    if bytes.len() < HEADER_SIZE {
        return Err(EngineError::ContractViolation(
            "node debug stats payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported node debug stats format version {format_version}"
        )));
    }
    let record_count = read_u32_le(bytes, 4)? as usize;
    let expected_len = HEADER_SIZE
        .checked_add(record_count.checked_mul(RECORD_SIZE).ok_or_else(|| {
            EngineError::ContractViolation("node debug stats payload length overflow".to_string())
        })?)
        .ok_or_else(|| {
            EngineError::ContractViolation("node debug stats payload length overflow".to_string())
        })?;
    if bytes.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
            "node debug stats payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut records = Vec::with_capacity(record_count);
    for record_index in 0..record_count {
        let base = HEADER_SIZE + record_index * RECORD_SIZE;
        let nds_flags = read_u16_le(bytes, base + 10)?;
        records.push(NodeDebugStats {
            node_id: read_u32_le(bytes, base)?,
            feature_index: read_u32_le(bytes, base + 4)?,
            threshold_bin: read_u16_le(bytes, base + 8)?,
            gain: read_f32_le(bytes, base + 12)?,
            default_left: (nds_flags & 1) != 0,
            left_stats: NodeStats {
                grad_sum: read_f32_le(bytes, base + 16)?,
                hess_sum: read_f32_le(bytes, base + 20)?,
                grad_sq_sum: 0.0,
                row_count: read_u32_le(bytes, base + 24)?,
            },
            right_stats: NodeStats {
                grad_sum: read_f32_le(bytes, base + 28)?,
                hess_sum: read_f32_le(bytes, base + 32)?,
                grad_sq_sum: 0.0,
                row_count: read_u32_le(bytes, base + 36)?,
            },
        });
    }
    Ok(records)
}

pub(crate) fn decode_optional_node_debug_stats_section(
    sections: &[ModelArtifactSection],
) -> EngineResult<Option<Vec<NodeDebugStats>>> {
    let Some(section) = optional_single_section(sections, ModelSectionKind::NodeDebugStats)? else {
        return Ok(None);
    };
    Ok(Some(decode_node_debug_stats_payload(&section.payload)?))
}

pub(crate) fn encode_trained_model_payload(model: &TrainedModel) -> EngineResult<Vec<u8>> {
    let feature_count = u32::try_from(model.feature_count).map_err(|_| {
        EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
    })?;
    let stump_count = u32::try_from(model.stumps.len())
        .map_err(|_| EngineError::ContractViolation("stump count exceeds u32::MAX".to_string()))?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&feature_count.to_le_bytes());
    bytes.extend_from_slice(&stump_count.to_le_bytes());
    bytes.extend_from_slice(&model.baseline_prediction.to_le_bytes());

    for stump in &model.stumps {
        bytes.extend_from_slice(&stump.split.node_id.to_le_bytes());
        bytes.extend_from_slice(&stump.split.feature_index.to_le_bytes());
        bytes.extend_from_slice(&stump.split.threshold_bin.to_le_bytes());
        let mut stump_flags: u16 = if stump.split.default_left { 1 } else { 0 };
        if stump.split.is_categorical {
            stump_flags |= 2; // bit 1 = is_categorical
        }
        bytes.extend_from_slice(&stump_flags.to_le_bytes());
        bytes.extend_from_slice(&stump.split.gain.to_le_bytes());
        bytes.extend_from_slice(&stump.left_leaf_value.as_scalar().to_le_bytes());
        bytes.extend_from_slice(&stump.right_leaf_value.as_scalar().to_le_bytes());
        bytes.extend_from_slice(&stump.split.left_stats.row_count.to_le_bytes());
        bytes.extend_from_slice(&stump.split.right_stats.row_count.to_le_bytes());
    }

    Ok(bytes)
}

pub(crate) fn decode_trained_model_payload(bytes: &[u8]) -> EngineResult<TrainedModel> {
    const HEADER_SIZE: usize = 16;
    const STUMP_SIZE: usize = 32;
    if bytes.len() < HEADER_SIZE {
        return Err(EngineError::ContractViolation(
            "model payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(EngineError::ContractViolation(format!(
            "unsupported model payload format version {format_version}"
        )));
    }
    let feature_count = read_u32_le(bytes, 4)? as usize;
    let stump_count = read_u32_le(bytes, 8)? as usize;
    let baseline_prediction = read_f32_le(bytes, 12)?;

    let expected_len = HEADER_SIZE
        .checked_add(stump_count.checked_mul(STUMP_SIZE).ok_or_else(|| {
            EngineError::ContractViolation("stump payload length overflow".to_string())
        })?)
        .ok_or_else(|| EngineError::ContractViolation("payload length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
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
        let flags = read_u16_le(bytes, base + 10)?;
        let default_left = (flags & 1) != 0;
        let is_categorical = (flags & 2) != 0;
        let gain = read_f32_le(bytes, base + 12)?;
        let left_leaf_value = read_f32_le(bytes, base + 16)?;
        let right_leaf_value = read_f32_le(bytes, base + 20)?;
        let left_count = read_u32_le(bytes, base + 24)?;
        let right_count = read_u32_le(bytes, base + 28)?;

        stumps.push(TrainedStump {
            split: SplitCandidate {
                node_id,
                feature_index,
                threshold_bin,
                gain,
                default_left,
                is_categorical,
                categorical_bitset: None, // populated from NativeCategoricalSplits section
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: left_count as f32,
                    grad_sq_sum: 0.0,
                    row_count: left_count,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: right_count as f32,
                    grad_sq_sum: 0.0,
                    row_count: right_count,
                },
            },
            left_leaf_value: LeafValue::Scalar(left_leaf_value),
            right_leaf_value: LeafValue::Scalar(right_leaf_value),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        });
    }

    Ok(TrainedModel {
        baseline_prediction,
        feature_count,
        stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    })
}

pub(crate) fn read_u32_le(bytes: &[u8], start: usize) -> EngineResult<u32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
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

pub(crate) fn read_u16_le(bytes: &[u8], start: usize) -> EngineResult<u16> {
    let end = start + 2;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
            "unexpected end of payload when reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

pub(crate) fn read_f32_le(bytes: &[u8], start: usize) -> EngineResult<f32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(EngineError::ContractViolation(
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
