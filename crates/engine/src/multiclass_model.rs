use alloygbm_core::{
    CategoricalStatePayloadV1, Device, DroMetadataPayload, LeafValue,
    LinearLeafCoefficientsPayload, LinearLeafEntry, MODEL_FORMAT_V1, ModelMetadata,
    ModelSectionKind, MorphMetadataPayload, NodeStats, SplitCandidate,
    decode_optional_categorical_state_section_v1, decode_optional_dro_metadata_artifact_section,
    decode_optional_linear_leaf_coefficients_section,
    decode_optional_morph_metadata_artifact_section, deserialize_model_artifact_v1,
    encode_categorical_state_payload_v1, encode_dro_metadata_payload,
    encode_linear_leaf_coefficients_payload, encode_morph_metadata_payload,
    serialize_model_artifact_v1, validate_categorical_state_payload_v1,
};

use crate::artifact::{read_f32_le, read_u16_le, read_u32_le, required_single_section};
use crate::error::{EngineError, EngineResult};
use crate::tree_node::TREE_NODE_STRIDE;
use crate::{IterationDiagnostics, IterationStopReason, TrainedStump};

/// Trained multi-class model: K tree sequences (one per class).
#[derive(Debug, Clone, PartialEq)]
pub struct MultiClassTrainedModel {
    pub num_classes: usize,
    pub baseline_predictions: Vec<f32>,
    pub feature_count: usize,
    pub class_stumps: Vec<Vec<TrainedStump>>,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    pub objective: String,
    /// Morph training metadata (None for non-morph artifacts).
    pub morph_metadata: Option<MorphMetadataPayload>,
    /// DRO leaf-solver metadata (None for standard leaf solving).
    pub dro_metadata: Option<DroMetadataPayload>,
}

impl MultiClassTrainedModel {
    pub fn rounds_completed(&self) -> usize {
        self.class_stumps
            .iter()
            .flat_map(|stumps| stumps.iter())
            .map(|stump| stump.split.node_id / TREE_NODE_STRIDE)
            .max()
            .map(|max_tree_id| max_tree_id as usize + 1)
            .unwrap_or(0)
    }

    pub fn with_categorical_state(
        mut self,
        state: Option<CategoricalStatePayloadV1>,
    ) -> EngineResult<Self> {
        if let Some(ref state) = state {
            validate_categorical_state_payload_v1(state, Some(self.feature_count))?;
        }
        self.categorical_state = state;
        Ok(self)
    }

    pub fn to_artifact_bytes(&self) -> EngineResult<Vec<u8>> {
        let feature_count_u32 = u32::try_from(self.feature_count).map_err(|_| {
            EngineError::ContractViolation("feature_count exceeds u32::MAX".to_string())
        })?;
        let num_classes_u32 = u32::try_from(self.num_classes).map_err(|_| {
            EngineError::ContractViolation("num_classes exceeds u32::MAX".to_string())
        })?;

        // Build MultiClassTrees section payload
        let mut mc_payload = Vec::new();
        mc_payload.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
        mc_payload.extend_from_slice(&num_classes_u32.to_le_bytes());
        mc_payload.extend_from_slice(&feature_count_u32.to_le_bytes());

        for baseline in &self.baseline_predictions {
            mc_payload.extend_from_slice(&baseline.to_le_bytes());
        }

        for class_stumps in &self.class_stumps {
            let count = u32::try_from(class_stumps.len()).map_err(|_| {
                EngineError::ContractViolation("stump count exceeds u32::MAX".to_string())
            })?;
            mc_payload.extend_from_slice(&count.to_le_bytes());
        }

        for class_stumps in &self.class_stumps {
            for stump in class_stumps {
                mc_payload.extend_from_slice(&stump.split.node_id.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.feature_index.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.threshold_bin.to_le_bytes());
                let mut flags: u16 = if stump.split.default_left { 1 } else { 0 };
                if stump.split.is_categorical {
                    flags |= 2;
                }
                mc_payload.extend_from_slice(&flags.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.gain.to_le_bytes());
                mc_payload.extend_from_slice(&stump.left_leaf_value.as_scalar().to_le_bytes());
                mc_payload.extend_from_slice(&stump.right_leaf_value.as_scalar().to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.left_stats.row_count.to_le_bytes());
                mc_payload.extend_from_slice(&stump.split.right_stats.row_count.to_le_bytes());
            }
        }

        // Build PredictorLayout payload
        let mut layout_payload = Vec::new();
        const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
        layout_payload.extend_from_slice(&MODEL_FORMAT_V1.to_le_bytes());
        layout_payload.extend_from_slice(&feature_count_u32.to_le_bytes());
        layout_payload.extend_from_slice(&THRESHOLD_MODE_BIN_INDEX.to_le_bytes());

        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..self.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: self.objective.clone(),
            num_classes: Some(num_classes_u32),
        };

        let mut sections = vec![
            (ModelSectionKind::MultiClassTrees, mc_payload),
            (ModelSectionKind::PredictorLayout, layout_payload),
        ];
        if let Some(categorical_state) = self.categorical_state.as_ref() {
            let categorical_payload = encode_categorical_state_payload_v1(categorical_state)?;
            sections.push((ModelSectionKind::CategoricalState, categorical_payload));
        }
        // Morph metadata section (optional — only for morph-trained artifacts)
        if let Some(morph) = self.morph_metadata.as_ref() {
            sections.push((
                ModelSectionKind::MorphMetadata,
                encode_morph_metadata_payload(morph),
            ));
        }
        // DRO metadata section (optional — only for DRO leaf-solver artifacts)
        if let Some(dro) = self.dro_metadata.as_ref() {
            sections.push((
                ModelSectionKind::DroMetadata,
                encode_dro_metadata_payload(dro),
            ));
        }
        // Linear leaf coefficients section (optional — only for pl-tree artifacts)
        // Multi-class linear leaf serialization: use prefix-sum offsets so each class
        // can have a different number of stumps.  global_idx = prefix[class_idx] + stump_within_class.
        {
            // Build prefix sums: prefix[k] = total stumps in classes 0..k
            let mut prefix = vec![0usize; self.class_stumps.len() + 1];
            for (k, cs) in self.class_stumps.iter().enumerate() {
                prefix[k + 1] = prefix[k] + cs.len();
            }
            let linear_entries: Vec<LinearLeafEntry> = self
                .class_stumps
                .iter()
                .enumerate()
                .flat_map(|(class_idx, class_stumps)| {
                    let class_offset = prefix[class_idx];
                    class_stumps
                        .iter()
                        .enumerate()
                        .filter_map(move |(stump_idx, stump)| {
                            let left = match &stump.left_leaf_value {
                                LeafValue::Linear(ll) => Some(ll.clone()),
                                _ => None,
                            };
                            let right = match &stump.right_leaf_value {
                                LeafValue::Linear(rl) => Some(rl.clone()),
                                _ => None,
                            };
                            if left.is_some() || right.is_some() {
                                let global_idx = (class_offset + stump_idx) as u32;
                                Some(LinearLeafEntry {
                                    stump_idx: global_idx,
                                    left_leaf: left,
                                    right_leaf: right,
                                })
                            } else {
                                None
                            }
                        })
                })
                .collect();
            if !linear_entries.is_empty() {
                sections.push((
                    ModelSectionKind::LinearLeafCoefficients,
                    encode_linear_leaf_coefficients_payload(&LinearLeafCoefficientsPayload {
                        entries: linear_entries,
                    }),
                ));
            }
        }

        serialize_model_artifact_v1(&metadata, &sections).map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;

        let mc_section =
            required_single_section(&parsed.sections, ModelSectionKind::MultiClassTrees)?;

        let payload = &mc_section.payload;
        const MC_HEADER_SIZE: usize = 12; // format_version + num_classes + feature_count
        if payload.len() < MC_HEADER_SIZE {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small".to_string(),
            ));
        }

        let format_version = read_u32_le(payload, 0)?;
        if format_version != MODEL_FORMAT_V1 {
            return Err(EngineError::ContractViolation(format!(
                "unsupported multiclass trees format version {format_version}"
            )));
        }
        let num_classes = read_u32_le(payload, 4)? as usize;
        let feature_count = read_u32_le(payload, 8)? as usize;

        let baselines_start = MC_HEADER_SIZE;
        let baselines_end = baselines_start + num_classes * 4;
        if payload.len() < baselines_end {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small for baselines".to_string(),
            ));
        }
        let mut baseline_predictions = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            baseline_predictions.push(read_f32_le(payload, baselines_start + k * 4)?);
        }

        let counts_start = baselines_end;
        let counts_end = counts_start + num_classes * 4;
        if payload.len() < counts_end {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small for stump counts".to_string(),
            ));
        }
        let mut stump_counts = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            stump_counts.push(read_u32_le(payload, counts_start + k * 4)? as usize);
        }

        const STUMP_SIZE: usize = 32;
        let total_stumps: usize = stump_counts.iter().sum();
        let stumps_start = counts_end;
        let expected_len = stumps_start + total_stumps * STUMP_SIZE;
        if payload.len() != expected_len {
            return Err(EngineError::ContractViolation(format!(
                "multiclass trees payload length {} does not match expected {expected_len}",
                payload.len()
            )));
        }

        let mut class_stumps = Vec::with_capacity(num_classes);
        let mut offset = stumps_start;
        for &count in stump_counts.iter().take(num_classes) {
            let mut stumps = Vec::with_capacity(count);
            for _ in 0..count {
                let node_id = read_u32_le(payload, offset)?;
                let feature_index = read_u32_le(payload, offset + 4)?;
                let threshold_bin = read_u16_le(payload, offset + 8)?;
                let flags = read_u16_le(payload, offset + 10)?;
                let default_left = (flags & 1) != 0;
                let is_categorical = (flags & 2) != 0;
                let gain = read_f32_le(payload, offset + 12)?;
                let left_leaf_value = read_f32_le(payload, offset + 16)?;
                let right_leaf_value = read_f32_le(payload, offset + 20)?;
                let left_count = read_u32_le(payload, offset + 24)?;
                let right_count = read_u32_le(payload, offset + 28)?;

                stumps.push(TrainedStump {
                    split: SplitCandidate {
                        node_id,
                        feature_index,
                        threshold_bin,
                        gain,
                        default_left,
                        is_categorical,
                        categorical_bitset: None,
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
                offset += STUMP_SIZE;
            }
            class_stumps.push(stumps);
        }

        let categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, feature_count)?;

        let morph_metadata = decode_optional_morph_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;
        let dro_metadata = decode_optional_dro_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;

        // Decode optional linear leaf coefficients and backfill class_stumps.
        // Global stump index uses prefix-sum offsets (same encoding as serialization).
        if let Some(ll_payload) = decode_optional_linear_leaf_coefficients_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            let mut prefix = vec![0usize; class_stumps.len() + 1];
            for (k, cs) in class_stumps.iter().enumerate() {
                prefix[k + 1] = prefix[k] + cs.len();
            }
            for entry in ll_payload.entries {
                let global_idx = entry.stump_idx as usize;
                let class_idx = prefix[1..].partition_point(|&p| p <= global_idx);
                if class_idx < class_stumps.len() {
                    let stump_idx = global_idx - prefix[class_idx];
                    if stump_idx < class_stumps[class_idx].len() {
                        if let Some(ll) = entry.left_leaf {
                            class_stumps[class_idx][stump_idx].left_leaf_value =
                                LeafValue::Linear(ll);
                        }
                        if let Some(rl) = entry.right_leaf {
                            class_stumps[class_idx][stump_idx].right_leaf_value =
                                LeafValue::Linear(rl);
                        }
                    }
                }
            }
        }

        Ok(Self {
            num_classes,
            baseline_predictions,
            feature_count,
            class_stumps,
            categorical_state,
            objective: parsed.contract.metadata.objective.clone(),
            morph_metadata,
            dro_metadata,
        })
    }
}

/// Summary from a multi-class training run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiClassIterationRunSummary {
    pub model: MultiClassTrainedModel,
    pub rounds_requested: usize,
    pub effective_round_cap: usize,
    pub rounds_completed: usize,
    pub stop_reason: IterationStopReason,
    pub initial_loss: f32,
    pub initial_validation_loss: Option<f32>,
    pub loss_per_completed_round: Vec<f32>,
    pub validation_loss_per_completed_round: Vec<f32>,
    pub sampled_rows_per_completed_round: Vec<usize>,
    pub sampled_features_per_completed_round: Vec<usize>,
    pub best_validation_loss: Option<f32>,
    pub best_validation_round: Option<usize>,
    pub weak_improvement_rounds_committed: usize,
    pub final_loss: f32,
    pub final_validation_loss: Option<f32>,
    /// Per-round custom metric values (empty when no custom metric callback is used).
    pub custom_metric_per_round: Vec<f32>,
    /// Name of the custom metric (None when no custom metric callback is used).
    pub custom_metric_name: Option<String>,
    /// Per-round diagnostic snapshot aggregated across the K class buffers
    /// (mean-of-class for norms / variance, max-of-class for
    /// `neutralization_effectiveness`).  See [`IterationDiagnostics`].
    pub diagnostics_per_round: Vec<IterationDiagnostics>,
}
