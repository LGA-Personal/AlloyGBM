use alloygbm_core::{
    CategoricalStatePayloadV1, DartTreeWeightsPayload, Device, DroMetadataPayload, LeafValue,
    LinearLeafCoefficientsPayload, LinearLeafEntry, MAX_MODEL_CLASSES, MAX_MODEL_FEATURES,
    MAX_MODEL_STUMPS, MODEL_FORMAT_V1, ModelMetadata, ModelSectionKind, MorphMetadataPayload,
    NodeStats, SplitCandidate, decode_optional_categorical_state_section_v1,
    decode_optional_dart_tree_weights_section, decode_optional_dro_metadata_artifact_section,
    decode_optional_linear_leaf_coefficients_section,
    decode_optional_morph_metadata_artifact_section, deserialize_model_artifact_v1,
    encode_categorical_state_payload_v1, encode_dart_tree_weights_payload,
    encode_dro_metadata_payload, encode_linear_leaf_coefficients_payload,
    encode_morph_metadata_payload, serialize_model_artifact_v1,
    validate_categorical_state_payload_v1,
};

use crate::artifact::{
    decode_predictor_layout_payload, read_f32_le, read_u16_le, read_u32_le, required_single_section,
};
use crate::error::{EngineError, EngineResult};
use crate::tree_node::TREE_NODE_STRIDE;
use crate::{IterationDiagnostics, IterationStopReason, ResolvedTrainingPolicy, TrainedStump};

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
        validate_multiclass_model_for_serialization(self)?;
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
        // Reuse the optional per-stump DART overlay. Multiclass stump payloads
        // are class-major, so the weights use the same class-major flattening.
        if self
            .class_stumps
            .iter()
            .flatten()
            .any(|stump| (stump.tree_weight - 1.0).abs() > f32::EPSILON)
        {
            sections.push((
                ModelSectionKind::DartTreeWeights,
                encode_dart_tree_weights_payload(&DartTreeWeightsPayload {
                    weights: self
                        .class_stumps
                        .iter()
                        .flatten()
                        .map(|stump| stump.tree_weight)
                        .collect(),
                }),
            ));
        }

        serialize_model_artifact_v1(&metadata, &sections).map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;

        if parsed.contract.metadata.objective != "multiclass_softmax" {
            return Err(EngineError::ContractViolation(format!(
                "MultiClassTrees requires objective multiclass_softmax, found {:?}",
                parsed.contract.metadata.objective
            )));
        }

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

        if !(2..=MAX_MODEL_CLASSES).contains(&num_classes) {
            return Err(EngineError::ContractViolation(format!(
                "multiclass payload num_classes {num_classes} must be in [2, {MAX_MODEL_CLASSES}]"
            )));
        }
        if feature_count == 0 || feature_count > MAX_MODEL_FEATURES {
            return Err(EngineError::ContractViolation(format!(
                "multiclass payload feature_count {feature_count} must be in [1, {MAX_MODEL_FEATURES}]"
            )));
        }
        if parsed.contract.metadata.num_classes != Some(num_classes as u32) {
            return Err(EngineError::ContractViolation(format!(
                "metadata num_classes {:?} does not match multiclass payload num_classes {num_classes}",
                parsed.contract.metadata.num_classes
            )));
        }
        let metadata_feature_count = parsed.contract.metadata.feature_names.len();
        if feature_count != metadata_feature_count {
            return Err(EngineError::ContractViolation(format!(
                "multiclass payload feature_count {feature_count} does not match metadata feature count {metadata_feature_count}"
            )));
        }
        let layout_section =
            required_single_section(&parsed.sections, ModelSectionKind::PredictorLayout)?;
        let predictor_layout = decode_predictor_layout_payload(&layout_section.payload)?;
        if predictor_layout.feature_count != feature_count {
            return Err(EngineError::ContractViolation(format!(
                "predictor layout feature_count {} does not match multiclass payload feature_count {feature_count}",
                predictor_layout.feature_count
            )));
        }

        let baselines_start = MC_HEADER_SIZE;
        let class_bytes = num_classes.checked_mul(4).ok_or_else(|| {
            EngineError::ContractViolation("multiclass class table length overflow".to_string())
        })?;
        let baselines_end = baselines_start.checked_add(class_bytes).ok_or_else(|| {
            EngineError::ContractViolation("multiclass baseline offset overflow".to_string())
        })?;
        if payload.len() < baselines_end {
            return Err(EngineError::ContractViolation(
                "multiclass trees payload too small for baselines".to_string(),
            ));
        }
        let mut baseline_predictions = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            let baseline = read_f32_le(payload, baselines_start + k * 4)?;
            if !baseline.is_finite() {
                return Err(EngineError::ContractViolation(format!(
                    "multiclass baseline {k} must be finite"
                )));
            }
            baseline_predictions.push(baseline);
        }

        let counts_start = baselines_end;
        let counts_end = counts_start.checked_add(class_bytes).ok_or_else(|| {
            EngineError::ContractViolation("multiclass stump count offset overflow".to_string())
        })?;
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
        let total_stumps = stump_counts.iter().try_fold(0usize, |total, &count| {
            total.checked_add(count).ok_or_else(|| {
                EngineError::ContractViolation("multiclass stump count overflow".to_string())
            })
        })?;
        if total_stumps > MAX_MODEL_STUMPS {
            return Err(EngineError::ContractViolation(format!(
                "multiclass stump count {total_stumps} exceeds maximum {MAX_MODEL_STUMPS}"
            )));
        }
        let stumps_start = counts_end;
        let stump_bytes = total_stumps.checked_mul(STUMP_SIZE).ok_or_else(|| {
            EngineError::ContractViolation("multiclass stump payload length overflow".to_string())
        })?;
        let expected_len = stumps_start.checked_add(stump_bytes).ok_or_else(|| {
            EngineError::ContractViolation("multiclass payload length overflow".to_string())
        })?;
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
                if flags & !3 != 0 {
                    return Err(EngineError::ContractViolation(format!(
                        "multiclass stump contains unsupported flags {flags:#x}"
                    )));
                }
                let default_left = (flags & 1) != 0;
                let is_categorical = (flags & 2) != 0;
                let gain = read_f32_le(payload, offset + 12)?;
                let left_leaf_value = read_f32_le(payload, offset + 16)?;
                let right_leaf_value = read_f32_le(payload, offset + 20)?;
                let left_count = read_u32_le(payload, offset + 24)?;
                let right_count = read_u32_le(payload, offset + 28)?;

                if feature_index as usize >= feature_count {
                    return Err(EngineError::ContractViolation(format!(
                        "multiclass stump split feature_index {feature_index} exceeds model feature_count {feature_count}"
                    )));
                }
                if !gain.is_finite()
                    || !left_leaf_value.is_finite()
                    || !right_leaf_value.is_finite()
                {
                    return Err(EngineError::ContractViolation(
                        "multiclass stump contains non-finite gain or leaf value".to_string(),
                    ));
                }

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

        // Pre-DART multiclass artifacts have no overlay and retain the 1.0
        // defaults assigned while decoding MultiClassTrees.
        if let Some(dart_payload) = decode_optional_dart_tree_weights_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            if dart_payload.weights.len() != total_stumps {
                return Err(EngineError::ContractViolation(format!(
                    "multiclass DartTreeWeights length {} != flattened stump count {total_stumps}",
                    dart_payload.weights.len(),
                )));
            }
            for (stump, &weight) in class_stumps.iter_mut().flatten().zip(&dart_payload.weights) {
                stump.tree_weight = weight;
            }
        }

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
                if global_idx >= total_stumps {
                    return Err(EngineError::ContractViolation(format!(
                        "linear leaf stump index {} exceeds multiclass stump count {total_stumps}",
                        entry.stump_idx
                    )));
                }
                let class_idx = prefix[1..].partition_point(|&p| p <= global_idx);
                let stump_idx = global_idx - prefix[class_idx];
                if let Some(ll) = entry.left_leaf {
                    validate_multiclass_linear_leaf_features(
                        &ll.regressor_features,
                        feature_count,
                        entry.stump_idx,
                    )?;
                    class_stumps[class_idx][stump_idx].left_leaf_value = LeafValue::Linear(ll);
                }
                if let Some(rl) = entry.right_leaf {
                    validate_multiclass_linear_leaf_features(
                        &rl.regressor_features,
                        feature_count,
                        entry.stump_idx,
                    )?;
                    class_stumps[class_idx][stump_idx].right_leaf_value = LeafValue::Linear(rl);
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

fn validate_multiclass_linear_leaf_features(
    regressor_features: &[u32],
    feature_count: usize,
    stump_index: u32,
) -> EngineResult<()> {
    for &feature_index in regressor_features {
        if feature_index as usize >= feature_count {
            return Err(EngineError::ContractViolation(format!(
                "linear leaf stump index {stump_index} regressor feature_index {feature_index} exceeds model feature_count {feature_count}"
            )));
        }
    }
    Ok(())
}

fn validate_multiclass_model_for_serialization(model: &MultiClassTrainedModel) -> EngineResult<()> {
    if model.objective != "multiclass_softmax" {
        return Err(EngineError::ContractViolation(format!(
            "MultiClassTrees requires objective multiclass_softmax, found {:?}",
            model.objective
        )));
    }
    if !(2..=MAX_MODEL_CLASSES).contains(&model.num_classes) {
        return Err(EngineError::ContractViolation(format!(
            "num_classes {} must be in [2, {MAX_MODEL_CLASSES}]",
            model.num_classes
        )));
    }
    if model.feature_count == 0 || model.feature_count > MAX_MODEL_FEATURES {
        return Err(EngineError::ContractViolation(format!(
            "feature_count {} must be in [1, {MAX_MODEL_FEATURES}]",
            model.feature_count
        )));
    }
    if model.baseline_predictions.len() != model.num_classes
        || model.class_stumps.len() != model.num_classes
    {
        return Err(EngineError::ContractViolation(format!(
            "num_classes {} must match baseline count {} and class stump table count {}",
            model.num_classes,
            model.baseline_predictions.len(),
            model.class_stumps.len()
        )));
    }
    if model
        .baseline_predictions
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(EngineError::ContractViolation(
            "multiclass baselines must be finite".to_string(),
        ));
    }
    let total_stumps = model
        .class_stumps
        .iter()
        .try_fold(0usize, |total, stumps| total.checked_add(stumps.len()))
        .ok_or_else(|| {
            EngineError::ContractViolation("multiclass stump count overflow".to_string())
        })?;
    if total_stumps > MAX_MODEL_STUMPS {
        return Err(EngineError::ContractViolation(format!(
            "multiclass stump count {total_stumps} exceeds maximum {MAX_MODEL_STUMPS}"
        )));
    }
    for (stump_index, stump) in model.class_stumps.iter().flatten().enumerate() {
        if stump.split.feature_index as usize >= model.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "stump {stump_index} split feature_index {} exceeds model feature_count {}",
                stump.split.feature_index, model.feature_count
            )));
        }
        if !stump.split.gain.is_finite()
            || !stump.left_leaf_value.as_scalar().is_finite()
            || !stump.right_leaf_value.as_scalar().is_finite()
            || !stump.tree_weight.is_finite()
            || stump.tree_weight < 0.0
        {
            return Err(EngineError::ContractViolation(format!(
                "stump {stump_index} contains invalid gain, leaf value, or tree weight"
            )));
        }
        if let LeafValue::Linear(leaf) = &stump.left_leaf_value {
            validate_multiclass_linear_leaf_features(
                &leaf.regressor_features,
                model.feature_count,
                stump_index as u32,
            )?;
        }
        if let LeafValue::Linear(leaf) = &stump.right_leaf_value {
            validate_multiclass_linear_leaf_features(
                &leaf.regressor_features,
                model.feature_count,
                stump_index as u32,
            )?;
        }
    }
    Ok(())
}

/// Summary from a multi-class training run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiClassIterationRunSummary {
    pub model: MultiClassTrainedModel,
    pub rounds_requested: usize,
    pub effective_round_cap: usize,
    pub resolved_training_policy: ResolvedTrainingPolicy,
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
