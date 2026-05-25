use std::collections::HashMap;

use alloygbm_core::{
    CategoricalStatePayloadV1, DartTreeWeightsPayload, Device, DroMetadataPayload,
    FeatureBaselinePayload, LeafValue, LinearLeafCoefficientsPayload, LinearLeafEntry,
    MODEL_FORMAT_V1, ModelMetadata, ModelSectionKind, MorphMetadataPayload,
    NativeCategoricalSplitsPayload, NeutralizationMetadataPayload,
    decode_optional_categorical_state_section_v1, decode_optional_dart_tree_weights_section,
    decode_optional_dro_metadata_artifact_section, decode_optional_feature_baseline_section,
    decode_optional_linear_leaf_coefficients_section,
    decode_optional_morph_metadata_artifact_section,
    decode_optional_native_categorical_splits_section,
    decode_optional_neutralization_metadata_artifact_section, deserialize_model_artifact_v1,
    encode_categorical_state_payload_v1, encode_dart_tree_weights_payload,
    encode_dro_metadata_payload, encode_feature_baseline_payload,
    encode_linear_leaf_coefficients_payload, encode_morph_metadata_payload,
    encode_native_categorical_splits_payload, encode_neutralization_metadata_payload,
    format_required_section_auto_mode_error, format_required_section_mode_error,
    serialize_model_artifact_v1, validate_categorical_state_payload_v1,
};

use crate::artifact::{
    artifact_compatibility_report_from_sections, decode_optional_node_debug_stats_section,
    decode_trained_model_payload, encode_node_debug_stats_payload, encode_predictor_layout_payload,
    encode_trained_model_payload, required_single_section, resolve_predictor_layout,
};
use crate::error::{EngineError, EngineResult};
use crate::tree_node::{decode_tree_node_id, row_satisfies_stump_path_features, split_went_left};
use crate::types::{
    ArtifactCompatibilityMode, ArtifactCompatibilityReport, NodeDebugStats, TrainedStump,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TrainedModel {
    pub baseline_prediction: f32,
    pub feature_count: usize,
    pub stumps: Vec<TrainedStump>,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    pub node_debug_stats: Option<Vec<NodeDebugStats>>,
    /// Objective name recorded in the model artifact metadata.
    pub objective: String,
    /// Feature indices that use native categorical splits (empty if none).
    pub native_categorical_feature_indices: Vec<u32>,
    /// Morph training metadata (None for non-morph artifacts).
    pub morph_metadata: Option<MorphMetadataPayload>,
    /// DRO leaf-solver metadata (None for standard leaf solving).
    pub dro_metadata: Option<DroMetadataPayload>,
    /// Global per-feature training-set means.  `Some(_)` only when the model
    /// uses piecewise-linear leaves and the feature baseline was recorded at
    /// fit time.  Length equals `feature_count`.  Consumed by SHAP for
    /// interventional decomposition of linear-leaf contributions.
    pub feature_baseline: Option<Vec<f32>>,
    /// v0.10.6+: Optional factor-neutralization configuration that was active
    /// during training. `Some(...)` only when the joint trainer's
    /// `effective_neutralization_config` returned a non-inert config. Mirrors
    /// `dro_metadata` — metadata only, prediction never reads it.
    pub neutralization_metadata: Option<NeutralizationMetadataPayload>,
}

impl TrainedModel {
    /// Count the number of distinct tree rounds in this model.
    pub fn rounds_completed(&self) -> usize {
        if self.stumps.is_empty() {
            return 0;
        }
        let max_tree_id = self
            .stumps
            .iter()
            .map(|s| decode_tree_node_id(s.split.node_id).0 as usize)
            .max()
            .unwrap_or(0);
        max_tree_id + 1
    }

    pub fn with_categorical_state(
        mut self,
        categorical_state: Option<CategoricalStatePayloadV1>,
    ) -> EngineResult<Self> {
        if let Some(state) = categorical_state.as_ref() {
            validate_categorical_state_payload_v1(state, Some(self.feature_count))?;
        }
        self.categorical_state = categorical_state;
        Ok(self)
    }

    pub fn with_node_debug_stats(
        mut self,
        node_debug_stats: Option<Vec<NodeDebugStats>>,
    ) -> EngineResult<Self> {
        if let Some(stats) = node_debug_stats.as_ref() {
            for stat in stats {
                if stat.feature_index as usize >= self.feature_count {
                    return Err(EngineError::ContractViolation(format!(
                        "node debug stats feature_index {} exceeds feature_count {}",
                        stat.feature_index, self.feature_count
                    )));
                }
            }
        }
        self.node_debug_stats = node_debug_stats;
        Ok(self)
    }

    pub fn with_node_debug_stats_from_stumps(self) -> EngineResult<Self> {
        let stats = self
            .stumps
            .iter()
            .map(|stump| NodeDebugStats {
                node_id: stump.split.node_id,
                feature_index: stump.split.feature_index,
                threshold_bin: stump.split.threshold_bin,
                gain: stump.split.gain,
                default_left: stump.split.default_left,
                left_stats: stump.split.left_stats.clone(),
                right_stats: stump.split.right_stats.clone(),
            })
            .collect();
        self.with_node_debug_stats(Some(stats))
    }

    pub fn predict_row(&self, features: &[f32]) -> EngineResult<f32> {
        if features.len() != self.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                self.feature_count
            )));
        }

        let stumps_by_node = self
            .stumps
            .iter()
            .map(|stump| (stump.split.node_id, stump))
            .collect::<HashMap<_, _>>();
        let mut prediction = self.baseline_prediction;
        for stump in &self.stumps {
            if !row_satisfies_stump_path_features(features, stump, &stumps_by_node)? {
                continue;
            }
            let feature_index = stump.split.feature_index as usize;
            let feature_value = features[feature_index];
            let leaf = if split_went_left(&stump.split, feature_value) {
                stump.left_leaf_value.eval_row(features)
            } else {
                stump.right_leaf_value.eval_row(features)
            };
            // v0.9.0: DART artifacts carry a per-stump `tree_weight` that
            // scales the leaf contribution at predict time. Non-DART
            // models have `tree_weight = 1.0` and this multiplication is
            // a no-op (bit-identical to v0.8.0).
            prediction += stump.tree_weight * leaf;
        }

        Ok(prediction)
    }

    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> EngineResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(EngineError::ContractViolation(
                "rows cannot be empty".to_string(),
            ));
        }
        rows.iter().map(|row| self.predict_row(row)).collect()
    }

    pub fn to_artifact_bytes(&self) -> EngineResult<Vec<u8>> {
        let trees_payload = encode_trained_model_payload(self)?;
        let predictor_layout_payload = encode_predictor_layout_payload(self)?;
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..self.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: self.objective.clone(),
            num_classes: None,
        };

        let mut sections = vec![
            (ModelSectionKind::Trees, trees_payload),
            (ModelSectionKind::PredictorLayout, predictor_layout_payload),
        ];
        if let Some(categorical_state) = self.categorical_state.as_ref() {
            let categorical_payload = encode_categorical_state_payload_v1(categorical_state)?;
            sections.push((ModelSectionKind::CategoricalState, categorical_payload));
        }
        if let Some(node_debug_stats) = self.node_debug_stats.as_ref() {
            let node_stats_payload = encode_node_debug_stats_payload(node_debug_stats)?;
            sections.push((ModelSectionKind::NodeDebugStats, node_stats_payload));
        }
        // Serialize native categorical splits if any stumps are categorical.
        if self.stumps.iter().any(|s| s.split.is_categorical) {
            let stump_bitsets: Vec<(u32, Vec<u8>)> = self
                .stumps
                .iter()
                .enumerate()
                .filter(|(_, s)| s.split.is_categorical)
                .map(|(i, s)| {
                    (
                        i as u32,
                        s.split.categorical_bitset.clone().unwrap_or_default(),
                    )
                })
                .collect();
            let payload = NativeCategoricalSplitsPayload {
                native_categorical_feature_indices: self.native_categorical_feature_indices.clone(),
                stump_bitsets,
            };
            let cat_bytes = encode_native_categorical_splits_payload(&payload)?;
            sections.push((ModelSectionKind::NativeCategoricalSplits, cat_bytes));
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
        // Neutralization metadata section (optional — only for joint artifacts with
        // factor neutralization active at training time).
        if let Some(neut) = self.neutralization_metadata.as_ref() {
            sections.push((
                ModelSectionKind::NeutralizationMetadata,
                encode_neutralization_metadata_payload(neut),
            ));
        }
        // Linear leaf coefficients section (optional — only for pl-tree artifacts)
        {
            let linear_entries: Vec<LinearLeafEntry> = self
                .stumps
                .iter()
                .enumerate()
                .filter_map(|(idx, stump)| {
                    let left = match &stump.left_leaf_value {
                        LeafValue::Linear(ll) => Some(ll.clone()),
                        _ => None,
                    };
                    let right = match &stump.right_leaf_value {
                        LeafValue::Linear(rl) => Some(rl.clone()),
                        _ => None,
                    };
                    if left.is_some() || right.is_some() {
                        Some(LinearLeafEntry {
                            stump_idx: idx as u32,
                            left_leaf: left,
                            right_leaf: right,
                        })
                    } else {
                        None
                    }
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
        // FeatureBaseline section (optional — written only when linear leaves
        // are present and the baseline was captured at fit time).  Provides
        // global per-feature means so SHAP can decompose linear leaves
        // interventionally without needing the original training data.
        if let Some(baseline) = self.feature_baseline.as_ref()
            && baseline.len() == self.feature_count
            && self.stumps.iter().any(|s| {
                matches!(s.left_leaf_value, LeafValue::Linear(_))
                    || matches!(s.right_leaf_value, LeafValue::Linear(_))
            })
        {
            sections.push((
                ModelSectionKind::FeatureBaseline,
                encode_feature_baseline_payload(&FeatureBaselinePayload {
                    feature_means: baseline.clone(),
                }),
            ));
        }

        // DART per-stump tree weights (optional). Emitted only when at least
        // one stump has a non-default weight, which keeps Standard/GOSS
        // artifacts byte-identical to v0.8.0.
        if self
            .stumps
            .iter()
            .any(|s| (s.tree_weight - 1.0).abs() > f32::EPSILON)
        {
            sections.push((
                ModelSectionKind::DartTreeWeights,
                encode_dart_tree_weights_payload(&DartTreeWeightsPayload {
                    weights: self.stumps.iter().map(|s| s.tree_weight).collect(),
                }),
            ));
        }

        // Multi-output leaf values section (v0.10.0+). Emitted only when at
        // least one stump carries K-output leaves (joint multi-label
        // trainer). One Vec<f32> per stump: [left_K_values..., right_K_values...].
        if self
            .stumps
            .iter()
            .any(|s| s.multi_output_leaf_values.is_some())
        {
            let n_outputs = self
                .stumps
                .iter()
                .find_map(|s| s.multi_output_leaf_values.as_ref().map(|v| v.0.len()))
                .unwrap_or(0) as u32;
            let per_stump_leaf_values: Vec<Vec<f32>> = self
                .stumps
                .iter()
                .map(|s| match s.multi_output_leaf_values.as_ref() {
                    Some((left, right)) => {
                        let mut packed = Vec::with_capacity(left.len() + right.len());
                        packed.extend_from_slice(left);
                        packed.extend_from_slice(right);
                        packed
                    }
                    None => Vec::new(),
                })
                .collect();
            sections.push((
                ModelSectionKind::MultiOutputLeafValues,
                alloygbm_core::encode_multi_output_leaf_values_payload(
                    &alloygbm_core::MultiOutputLeafValuesPayload {
                        n_outputs,
                        per_stump_leaf_values,
                    },
                ),
            ));
        }

        serialize_model_artifact_v1(&metadata, &sections).map_err(EngineError::from)
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> EngineResult<Self> {
        Self::from_artifact_bytes_with_mode(bytes, ArtifactCompatibilityMode::AllowLegacyTreesOnly)
    }

    pub fn artifact_compatibility_report(
        bytes: &[u8],
    ) -> EngineResult<ArtifactCompatibilityReport> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;
        Ok(artifact_compatibility_report_from_sections(
            &parsed.sections,
        ))
    }

    pub fn from_artifact_bytes_auto(
        bytes: &[u8],
    ) -> EngineResult<(Self, ArtifactCompatibilityMode)> {
        let report = Self::artifact_compatibility_report(bytes)?;
        let mode = report.recommended_mode.ok_or_else(|| {
            EngineError::ContractViolation(format_required_section_auto_mode_error(
                report.required_section_report(),
            ))
        })?;
        let model = Self::from_artifact_bytes_with_mode(bytes, mode)?;
        Ok((model, mode))
    }

    pub fn from_artifact_bytes_with_mode(
        bytes: &[u8],
        compatibility_mode: ArtifactCompatibilityMode,
    ) -> EngineResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(EngineError::from)?;
        let compatibility_report = artifact_compatibility_report_from_sections(&parsed.sections);

        match compatibility_mode {
            ArtifactCompatibilityMode::Strict if !compatibility_report.strict_compatible => {
                return Err(EngineError::ContractViolation(
                    format_required_section_mode_error(
                        compatibility_report.required_section_report(),
                        false,
                    ),
                ));
            }
            ArtifactCompatibilityMode::AllowLegacyTreesOnly
                if !compatibility_report.legacy_compatible =>
            {
                return Err(EngineError::ContractViolation(
                    format_required_section_mode_error(
                        compatibility_report.required_section_report(),
                        true,
                    ),
                ));
            }
            _ => {}
        }

        let trees_section = required_single_section(&parsed.sections, ModelSectionKind::Trees)?;
        let metadata_feature_count = parsed.contract.metadata.feature_names.len();
        let predictor_layout =
            resolve_predictor_layout(&parsed.sections, metadata_feature_count, compatibility_mode)?;

        let mut model = decode_trained_model_payload(&trees_section.payload)?;

        if predictor_layout.feature_count != metadata_feature_count {
            return Err(EngineError::ContractViolation(format!(
                "predictor layout feature_count {} does not match metadata feature count {}",
                predictor_layout.feature_count, metadata_feature_count
            )));
        }
        if model.feature_count != predictor_layout.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "decoded trees feature_count {} does not match predictor layout feature_count {}",
                model.feature_count, predictor_layout.feature_count
            )));
        }
        if model.feature_count != metadata_feature_count {
            return Err(EngineError::ContractViolation(format!(
                "decoded trees feature_count {} does not match metadata feature count {}",
                model.feature_count, metadata_feature_count
            )));
        }

        model.categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, metadata_feature_count)?;
        model.node_debug_stats = decode_optional_node_debug_stats_section(&parsed.sections)?;

        // Decode optional native categorical splits section and populate stump bitsets.
        if let Some(cat_payload) =
            decode_optional_native_categorical_splits_section(&parsed.sections)?
        {
            model.native_categorical_feature_indices =
                cat_payload.native_categorical_feature_indices;
            for (stump_index, bitset) in cat_payload.stump_bitsets {
                let idx = stump_index as usize;
                if idx < model.stumps.len() {
                    model.stumps[idx].split.categorical_bitset = Some(bitset);
                }
            }
        }

        // Decode optional morph metadata section.
        model.morph_metadata = decode_optional_morph_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;
        model.dro_metadata = decode_optional_dro_metadata_artifact_section(&parsed.sections)
            .map_err(EngineError::from)?;
        model.neutralization_metadata =
            decode_optional_neutralization_metadata_artifact_section(&parsed.sections)
                .map_err(EngineError::from)?;

        // Decode optional linear leaf coefficients section and backfill LeafValue::Linear on stumps.
        if let Some(ll_payload) = decode_optional_linear_leaf_coefficients_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            for entry in ll_payload.entries {
                let idx = entry.stump_idx as usize;
                if idx < model.stumps.len() {
                    if let Some(ll) = entry.left_leaf {
                        model.stumps[idx].left_leaf_value = LeafValue::Linear(ll);
                    }
                    if let Some(rl) = entry.right_leaf {
                        model.stumps[idx].right_leaf_value = LeafValue::Linear(rl);
                    }
                }
            }
        }

        // Decode optional FeatureBaseline section.  Only retain when the
        // length matches feature_count to defend against artifact corruption
        // or schema drift; mismatches silently fall back to `None`, which
        // SHAP treats as "no linear-leaf support recorded for this artifact".
        model.feature_baseline = decode_optional_feature_baseline_section(&parsed.sections)
            .map_err(EngineError::from)?
            .map(|payload| payload.feature_means)
            .filter(|means| means.len() == metadata_feature_count);

        // Decode optional DartTreeWeights section and apply per-stump weights.
        // Pre-v0.9.0 artifacts have no section; stumps keep their default 1.0.
        if let Some(dart_payload) = decode_optional_dart_tree_weights_section(&parsed.sections)
            .map_err(EngineError::from)?
        {
            if dart_payload.weights.len() != model.stumps.len() {
                return Err(EngineError::ContractViolation(format!(
                    "DartTreeWeights length {} != stump count {}",
                    dart_payload.weights.len(),
                    model.stumps.len()
                )));
            }
            for (stump, w) in model.stumps.iter_mut().zip(dart_payload.weights.iter()) {
                stump.tree_weight = *w;
            }
        }

        // Decode optional MultiOutputLeafValues section (v0.10.0+) and attach
        // K-output leaf values to stumps. Pre-v0.10.0 artifacts have no section.
        if let Some(mo_payload) =
            alloygbm_core::decode_optional_multi_output_leaf_values_section(&parsed.sections)
                .map_err(EngineError::from)?
        {
            if mo_payload.per_stump_leaf_values.len() != model.stumps.len() {
                return Err(EngineError::ContractViolation(format!(
                    "MultiOutputLeafValues length {} != stump count {}",
                    mo_payload.per_stump_leaf_values.len(),
                    model.stumps.len()
                )));
            }
            let k = mo_payload.n_outputs as usize;
            for (stump, packed) in model
                .stumps
                .iter_mut()
                .zip(mo_payload.per_stump_leaf_values.into_iter())
            {
                if packed.is_empty() {
                    continue;
                }
                if packed.len() != 2 * k {
                    return Err(EngineError::ContractViolation(format!(
                        "MultiOutputLeafValues stump entry has {} values, expected 2 × n_outputs = {}",
                        packed.len(),
                        2 * k
                    )));
                }
                let (left, right) = packed.split_at(k);
                stump.multi_output_leaf_values = Some((left.to_vec(), right.to_vec()));
            }
        }

        model.feature_count = metadata_feature_count;
        model.objective = parsed.contract.metadata.objective.clone();
        Ok(model)
    }
}
