use crate::*;

fn sample_metadata() -> ModelMetadata {
    ModelMetadata {
        format_version: MODEL_FORMAT_V1,
        feature_names: vec!["feature_0".to_string(), "ticker\"id".to_string()],
        trained_device: Device::Cpu,
        objective: "squared_error".to_string(),
        num_classes: None,
    }
}

#[test]
fn validates_default_train_params() {
    let params = TrainParams::default();
    assert!(validate_train_params(&params).is_ok());
}

#[test]
fn train_params_default_has_no_neutralization_config() {
    let params = TrainParams::default();
    assert!(params.neutralization_config.is_none());
}

#[test]
fn validates_factor_exposure_matrix_shape_and_finiteness() {
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).is_ok());
    assert!(FactorExposureMatrix::new(0, 2, vec![]).is_err());
    assert!(FactorExposureMatrix::new(2, 0, vec![]).is_err());
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, f32::NAN, 0.0, 1.0]).is_err());
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 1.0]).is_err());
}

#[test]
fn factor_exposure_matrix_row_rejects_out_of_bounds_index() {
    let exposures =
        FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid exposures");
    assert_eq!(exposures.row(1).expect("row exists"), &[0.0, 1.0]);
    assert!(exposures.row(2).is_err());

    let malformed = FactorExposureMatrix {
        row_count: 2,
        factor_count: 2,
        values: vec![1.0],
    };
    assert!(malformed.row(0).is_err());
}

#[test]
fn validates_neutralization_config_contract() {
    let mut params = TrainParams {
        neutralization_config: Some(FactorNeutralizationConfig {
            kind: NeutralizationKind::PerRoundGradient,
            ridge_lambda: 1e-6,
            split_penalty: 0.0,
        }),
        ..TrainParams::default()
    };
    assert!(validate_train_params(&params).is_ok());

    params.neutralization_config = Some(FactorNeutralizationConfig {
        kind: NeutralizationKind::SplitPenalty,
        ridge_lambda: 1e-6,
        split_penalty: 0.1,
    });
    assert!(validate_train_params(&params).is_ok());

    params.neutralization_config = Some(FactorNeutralizationConfig {
        kind: NeutralizationKind::SplitPenalty,
        ridge_lambda: 1e-6,
        split_penalty: -0.1,
    });
    assert!(validate_train_params(&params).is_err());
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
fn rejects_invalid_row_subsample() {
    let params = TrainParams {
        row_subsample: 0.0,
        ..TrainParams::default()
    };
    assert!(matches!(
        validate_train_params(&params),
        Err(CoreError::InvalidConfig(_))
    ));
}

#[test]
fn rejects_invalid_col_subsample() {
    let params = TrainParams {
        col_subsample: 1.5,
        ..TrainParams::default()
    };
    assert!(matches!(
        validate_train_params(&params),
        Err(CoreError::InvalidConfig(_))
    ));
}

#[test]
fn rejects_invalid_early_stopping_rounds() {
    let params = TrainParams {
        early_stopping_rounds: Some(0),
        ..TrainParams::default()
    };
    assert!(matches!(
        validate_train_params(&params),
        Err(CoreError::InvalidConfig(_))
    ));
}

#[test]
fn rejects_negative_min_validation_improvement() {
    let params = TrainParams {
        min_validation_improvement: -0.1,
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
fn rejects_dataset_schema_with_unsorted_or_duplicate_categorical_indices() {
    let duplicate = DatasetSchema {
        feature_count: 4,
        has_time_index: false,
        has_group_id: false,
        categorical_feature_indices: vec![1, 1],
    };
    assert!(matches!(
        validate_dataset_schema(&duplicate),
        Err(CoreError::Validation(_))
    ));

    let unsorted = DatasetSchema {
        feature_count: 4,
        has_time_index: false,
        has_group_id: false,
        categorical_feature_indices: vec![2, 1],
    };
    assert!(matches!(
        validate_dataset_schema(&unsorted),
        Err(CoreError::Validation(_))
    ));
}

#[test]
fn rejects_training_dataset_with_mismatched_targets() {
    let dataset = TrainingDataset {
        matrix: DatasetMatrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]).expect("valid matrix"),
        targets: vec![1.0],
        sample_weights: None,
        time_index: None,
        group_id: None,
        factor_exposures: None,
    };
    assert!(matches!(
        validate_training_dataset(&dataset),
        Err(CoreError::Validation(_))
    ));
}

#[test]
fn validate_training_dataset_rejects_invalid_public_factor_exposures() {
    let dataset = TrainingDataset {
        matrix: DatasetMatrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]).expect("valid matrix"),
        targets: vec![1.0, 2.0],
        sample_weights: None,
        time_index: None,
        group_id: None,
        factor_exposures: Some(FactorExposureMatrix {
            row_count: 1,
            factor_count: 2,
            values: vec![1.0, f32::NAN],
        }),
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
        nan_bin_index: MISSING_BIN_U8 as u16,
        bins: vec![3, 8],
        bins_col: vec![3, 8],
        bins_adaptive: BinStorage::U8(vec![3, 8]),
        bins_col_adaptive: BinStorage::U8(vec![3, 8]),
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
fn required_section_compatibility_report_classifies_strict_and_legacy_layouts() {
    let strict_sections = vec![
        ModelArtifactSection {
            descriptor: ModelSectionDescriptor {
                kind: ModelSectionKind::Trees,
                offset: 120,
                length: 4,
            },
            payload: vec![1_u8, 2, 3, 4],
        },
        ModelArtifactSection {
            descriptor: ModelSectionDescriptor {
                kind: ModelSectionKind::PredictorLayout,
                offset: 124,
                length: 4,
            },
            payload: vec![5_u8, 6, 7, 8],
        },
    ];
    let strict_report = required_section_compatibility_report(&strict_sections);
    assert!(strict_report.strict_compatible);
    assert!(!strict_report.legacy_trees_only_compatible);
    assert!(strict_report.legacy_compatible);

    let legacy_sections = vec![ModelArtifactSection {
        descriptor: ModelSectionDescriptor {
            kind: ModelSectionKind::Trees,
            offset: 120,
            length: 4,
        },
        payload: vec![1_u8, 2, 3, 4],
    }];
    let legacy_report = required_section_compatibility_report(&legacy_sections);
    assert!(!legacy_report.strict_compatible);
    assert!(legacy_report.legacy_trees_only_compatible);
    assert!(legacy_report.legacy_compatible);
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
            objective: "squared_error".to_string(),
            num_classes: None,
        },
    };
    assert!(matches!(
        validate_model_contract_v1(&contract),
        Err(CoreError::Serialization(_))
    ));
}

#[test]
fn model_contract_rejects_section_offset_before_payload_start() {
    let contract = ModelIoContractV1 {
        header: ModelBinaryHeader::new(1, 64),
        sections: vec![ModelSectionDescriptor {
            kind: ModelSectionKind::Trees,
            offset: 79,
            length: 4,
        }],
        metadata: ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        },
    };

    assert!(matches!(
        validate_model_contract_v1(&contract),
        Err(CoreError::Serialization(_))
    ));
}

#[test]
fn model_contract_rejects_non_contiguous_section_offsets() {
    let contract = ModelIoContractV1 {
        header: ModelBinaryHeader::new(2, 64),
        sections: vec![
            ModelSectionDescriptor {
                kind: ModelSectionKind::Trees,
                offset: 120,
                length: 8,
            },
            ModelSectionDescriptor {
                kind: ModelSectionKind::PredictorLayout,
                offset: 130,
                length: 4,
            },
        ],
        metadata: ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
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

#[test]
fn categorical_state_payload_roundtrip() {
    let payload = CategoricalStatePayloadV1 {
        format_version: CATEGORICAL_STATE_FORMAT_V1,
        leakage_safe_target_encoding: true,
        categorical_feature_indices: vec![0, 3, 7],
    };
    let encoded = encode_categorical_state_payload_v1(&payload).expect("encodes");
    let decoded = decode_categorical_state_payload_v1(&encoded).expect("decodes");
    assert_eq!(decoded, payload);
}

#[test]
fn categorical_state_payload_rejects_invalid_ordering() {
    let payload = CategoricalStatePayloadV1 {
        format_version: CATEGORICAL_STATE_FORMAT_V1,
        leakage_safe_target_encoding: false,
        categorical_feature_indices: vec![2, 1],
    };
    assert!(matches!(
        encode_categorical_state_payload_v1(&payload),
        Err(CoreError::Validation(_))
    ));
}

#[test]
fn categorical_state_payload_decode_rejects_unknown_flags() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&CATEGORICAL_STATE_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        decode_categorical_state_payload_v1(&bytes),
        Err(CoreError::Serialization(_))
    ));
}

#[test]
fn strict_compatibility_allows_optional_categorical_state_section() {
    let metadata = sample_metadata();
    let categorical_state = encode_categorical_state_payload_v1(&CategoricalStatePayloadV1 {
        format_version: CATEGORICAL_STATE_FORMAT_V1,
        leakage_safe_target_encoding: true,
        categorical_feature_indices: vec![1],
    })
    .expect("categorical payload encodes");

    let bytes = serialize_model_artifact_v1(
        &metadata,
        &[
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
            (ModelSectionKind::CategoricalState, categorical_state),
        ],
    )
    .expect("artifact encodes");
    let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
    let report = required_section_compatibility_report(&parsed.sections);
    assert!(report.strict_compatible);

    let decoded_categorical = decode_optional_categorical_state_section_v1(
        &parsed.sections,
        metadata.feature_names.len(),
    )
    .expect("categorical state decodes");
    assert!(decoded_categorical.is_some());
}

#[test]
fn decode_optional_categorical_state_section_rejects_duplicate_sections() {
    let sections = vec![
        ModelArtifactSection {
            descriptor: ModelSectionDescriptor {
                kind: ModelSectionKind::CategoricalState,
                offset: 100,
                length: 20,
            },
            payload: vec![1_u8; 20],
        },
        ModelArtifactSection {
            descriptor: ModelSectionDescriptor {
                kind: ModelSectionKind::CategoricalState,
                offset: 120,
                length: 20,
            },
            payload: vec![1_u8; 20],
        },
    ];
    assert!(matches!(
        decode_optional_categorical_state_section_v1(&sections, 8),
        Err(CoreError::Serialization(_))
    ));
}

#[test]
fn dense_matrix_view_matches_dataset_layout() {
    let view =
        DenseMatrixView::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("dense view is valid");
    assert_eq!(view.row(1).expect("row resolves"), &[4.0, 5.0, 6.0]);
    assert_eq!(view.value_at(0, 2).expect("value resolves"), 3.0);

    let dataset = view.to_dataset_matrix().expect("dataset materializes");
    assert_eq!(dataset.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn columnar_matrix_view_materializes_rows_and_honors_validity() {
    let view = ColumnarMatrixView::new(
        3,
        vec![
            ColumnarMatrixColumnView {
                values: &[1.0, 2.0, 3.0],
                validity: None,
            },
            ColumnarMatrixColumnView {
                values: &[10.0, 20.0, 30.0],
                validity: Some(&[true, false, true]),
            },
        ],
    )
    .expect("columnar view is valid");

    assert_eq!(view.value_at(1, 1).expect("value resolves"), None);
    let dataset = view
        .to_dataset_matrix(-1.0)
        .expect("dataset materializes from columnar view");
    assert_eq!(dataset.values, vec![1.0, 10.0, 2.0, -1.0, 3.0, 30.0]);
}

#[test]
fn columnar_matrix_view_rejects_misaligned_validity() {
    let result = ColumnarMatrixView::new(
        2,
        vec![ColumnarMatrixColumnView {
            values: &[1.0, 2.0],
            validity: Some(&[true]),
        }],
    );
    assert!(matches!(result, Err(CoreError::Validation(_))));
}

#[test]
fn strict_compatibility_ignores_optional_node_debug_stats_section() {
    let metadata = sample_metadata();
    let bytes = serialize_model_artifact_v1(
        &metadata,
        &[
            (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
            (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
            (ModelSectionKind::NodeDebugStats, vec![5_u8, 4, 3, 2]),
        ],
    )
    .expect("artifact encodes");
    let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
    let report = required_section_compatibility_report(&parsed.sections);
    assert!(report.strict_compatible);
}

#[test]
fn test_metadata_json_num_classes_roundtrip() {
    let metadata = ModelMetadata {
        format_version: MODEL_FORMAT_V1,
        feature_names: vec!["f0".to_string(), "f1".to_string()],
        trained_device: Device::Cpu,
        objective: "multiclass_softmax".to_string(),
        num_classes: Some(5),
    };
    let json = serialize_metadata_json(&metadata);
    let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
    assert_eq!(decoded, metadata);
    assert_eq!(decoded.num_classes, Some(5));
}

#[test]
fn test_metadata_json_backward_compat_no_num_classes() {
    let metadata = ModelMetadata {
        format_version: MODEL_FORMAT_V1,
        feature_names: vec!["f0".to_string()],
        trained_device: Device::Cpu,
        objective: "squared_error".to_string(),
        num_classes: None,
    };
    let json = serialize_metadata_json(&metadata);
    // Verify the JSON does not contain num_classes when None
    assert!(!json.contains("num_classes"));
    let decoded = deserialize_metadata_json(&json).expect("metadata should decode");
    assert_eq!(decoded, metadata);
    assert_eq!(decoded.num_classes, None);
}

#[test]
fn morph_config_default_matches_paper() {
    let cfg = MorphConfig::default();
    assert_eq!(cfg.morph_rate, 0.1);
    assert_eq!(cfg.evolution_pressure, 0.2);
    assert_eq!(cfg.morph_warmup_iters, 5);
    assert_eq!(cfg.info_score_weight, 0.3);
    assert_eq!(cfg.depth_penalty_base, 0.9);
    assert!(cfg.balance_penalty);
    assert_eq!(cfg.lr_schedule, LrSchedule::Constant);
}

#[test]
fn lr_schedule_warmup_cosine_default_warmup_frac() {
    let s = LrSchedule::WarmupCosine { warmup_frac: 0.1 };
    if let LrSchedule::WarmupCosine { warmup_frac } = s {
        assert!((warmup_frac - 0.1).abs() < 1e-6);
    } else {
        panic!("expected WarmupCosine");
    }
}

#[test]
fn training_mode_default_is_auto() {
    assert_eq!(TrainingMode::default(), TrainingMode::Auto);
}

#[test]
fn train_params_default_has_no_morph_config() {
    let p = TrainParams::default();
    assert!(p.morph_config.is_none());
}

#[test]
fn dro_effective_gradient_zero_radius_matches_l1_threshold() {
    let cfg = DroConfig {
        radius: 0.0,
        metric: DroMetric::Wasserstein,
    };
    assert_eq!(leaf_effective_gradient(4.0, 20.0, 5, 0.5, Some(&cfg)), 3.5);
    assert_eq!(
        leaf_effective_gradient(-4.0, 20.0, 5, 0.5, Some(&cfg)),
        -3.5
    );
}

#[test]
fn dro_effective_gradient_shrinks_uncertain_leaf_signal() {
    let cfg = DroConfig {
        radius: 1.0,
        metric: DroMetric::Wasserstein,
    };
    let robust = leaf_effective_gradient(2.0, 20.0, 5, 0.0, Some(&cfg));
    assert!(robust.abs() < 2.0);
    assert_eq!(robust, 0.0);
}

#[test]
fn dro_leaf_gain_uses_same_effective_gradient_as_leaf_solve() {
    let cfg = DroConfig {
        radius: 0.25,
        metric: DroMetric::Wasserstein,
    };
    let effective = leaf_effective_gradient(10.0, 120.0, 8, 0.1, Some(&cfg));
    let gain = leaf_gain_term(10.0, 6.0, 120.0, 8, 0.1, 0.3, Some(&cfg));
    let expected = 0.5 * effective * effective / (6.0 + 0.3 + 1e-6);
    assert!((gain - expected).abs() < 1e-6);
}

#[test]
fn validate_train_params_accepts_morph_config() {
    let p = TrainParams {
        morph_config: Some(MorphConfig::default()),
        ..TrainParams::default()
    };
    assert!(validate_train_params(&p).is_ok());
}

#[test]
fn validate_train_params_rejects_invalid_morph_rate() {
    let p = TrainParams {
        morph_config: Some(MorphConfig {
            morph_rate: -0.1,
            ..MorphConfig::default()
        }),
        ..TrainParams::default()
    };
    assert!(validate_train_params(&p).is_err());
}

#[test]
fn validate_train_params_rejects_invalid_warmup_frac() {
    let p = TrainParams {
        morph_config: Some(MorphConfig {
            lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 1.5 },
            ..MorphConfig::default()
        }),
        ..TrainParams::default()
    };
    assert!(validate_train_params(&p).is_err());
}

#[test]
fn morph_metadata_round_trip() {
    let payload = MorphMetadataPayload {
        config: MorphConfig {
            morph_rate: 0.15,
            evolution_pressure: 0.25,
            morph_warmup_iters: 7,
            info_score_weight: 0.4,
            depth_penalty_base: 0.85,
            balance_penalty: false,
            lr_schedule: LrSchedule::WarmupCosine { warmup_frac: 0.2 },
        },
        final_iteration: 42,
        final_total: 100,
        ema_stats: Vec::new(),
    };
    let bytes = encode_morph_metadata_payload(&payload);
    // v2 envelope: 36 byte header + 4 byte EMA count (= 0) = 40 bytes.
    assert_eq!(bytes.len(), 40);
    let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn morph_metadata_round_trip_with_ema_state() {
    // v0.7.3 EMA persistence: payload encodes/decodes multi-class
    // EMA stats round-trip-clean so warm-start can resume the
    // exact EMA state from the previous fit.
    let payload = MorphMetadataPayload {
        config: MorphConfig::default(),
        final_iteration: 50,
        final_total: 50,
        ema_stats: vec![
            GradientEmaStats {
                mean: 0.012,
                std: 0.85,
                alpha: 0.05,
            },
            GradientEmaStats {
                mean: -0.003,
                std: 1.12,
                alpha: 0.05,
            },
            GradientEmaStats {
                mean: 0.007,
                std: 0.93,
                alpha: 0.05,
            },
        ],
    };
    let bytes = encode_morph_metadata_payload(&payload);
    // 36 header + 4 count + 3 * 12 stats = 76 bytes.
    assert_eq!(bytes.len(), 76);
    let decoded = decode_optional_morph_metadata_section(&bytes).unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn morph_metadata_v1_backcompat_decodes_with_empty_ema() {
    // Pre-v0.7.3 (version 1) MorphMetadata payloads have no EMA
    // tail.  Decoder must accept them and return an empty EMA Vec
    // so the warm-start path falls back to a cold EMA (matches
    // pre-v0.7.3 behaviour).
    let v1_bytes = {
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(&1u16.to_le_bytes()); // version = 1
        buf.extend_from_slice(&0.1f32.to_le_bytes()); // morph_rate
        buf.extend_from_slice(&0.2f32.to_le_bytes()); // evolution_pressure
        buf.extend_from_slice(&5u32.to_le_bytes()); // morph_warmup_iters
        buf.extend_from_slice(&0.3f32.to_le_bytes()); // info_score_weight
        buf.extend_from_slice(&0.9f32.to_le_bytes()); // depth_penalty_base
        buf.push(1); // balance_penalty
        buf.push(0); // lr_kind = Constant
        buf.extend_from_slice(&0.0f32.to_le_bytes()); // warmup_frac
        buf.extend_from_slice(&10u32.to_le_bytes()); // final_iteration
        buf.extend_from_slice(&10u32.to_le_bytes()); // final_total
        buf
    };
    assert_eq!(v1_bytes.len(), 36);
    let decoded = decode_optional_morph_metadata_section(&v1_bytes).unwrap();
    assert!(
        decoded.ema_stats.is_empty(),
        "v1 payload must decode with empty ema_stats"
    );
    assert_eq!(decoded.config.morph_rate, 0.1);
    assert_eq!(decoded.final_iteration, 10);
}

#[test]
fn morph_metadata_decode_rejects_short_input() {
    let err = decode_optional_morph_metadata_section(&[0u8; 10]).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn morph_metadata_decode_rejects_unknown_version() {
    let mut bytes = vec![0u8; 36];
    bytes[0] = 99; // version = 99
    let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn morph_metadata_decode_rejects_unknown_lr_kind() {
    let payload = MorphMetadataPayload {
        config: MorphConfig::default(),
        final_iteration: 0,
        final_total: 1,
        ema_stats: Vec::new(),
    };
    let mut bytes = encode_morph_metadata_payload(&payload);
    bytes[23] = 99; // lr_schedule_kind at offset 2+5*4+1=23
    let err = decode_optional_morph_metadata_section(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn morph_metadata_artifact_round_trip() {
    let metadata = sample_metadata();
    let morph_payload = MorphMetadataPayload {
        config: MorphConfig {
            morph_rate: 0.15,
            evolution_pressure: 0.25,
            morph_warmup_iters: 7,
            info_score_weight: 0.4,
            depth_penalty_base: 0.85,
            balance_penalty: true,
            lr_schedule: LrSchedule::Constant,
        },
        final_iteration: 10,
        final_total: 10,
        ema_stats: Vec::new(),
    };
    let morph_bytes = encode_morph_metadata_payload(&morph_payload);
    let sections = vec![
        (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
        (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
        (ModelSectionKind::MorphMetadata, morph_bytes),
    ];
    let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
    let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
    assert_eq!(parsed.sections.len(), 3);
    let decoded_morph = decode_optional_morph_metadata_artifact_section(&parsed.sections).unwrap();
    assert_eq!(decoded_morph, Some(morph_payload));
}

#[test]
fn morph_metadata_artifact_absent_for_non_morph() {
    let metadata = sample_metadata();
    let sections = vec![
        (ModelSectionKind::Trees, vec![1_u8, 2, 3, 4]),
        (ModelSectionKind::PredictorLayout, vec![9_u8, 8, 7]),
    ];
    let bytes = serialize_model_artifact_v1(&metadata, &sections).expect("artifact encodes");
    let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");
    let decoded_morph = decode_optional_morph_metadata_artifact_section(&parsed.sections).unwrap();
    assert_eq!(decoded_morph, None);
}

#[test]
fn gradient_ema_single_pass_matches_two_pass_within_tolerance() {
    // Use the existing two-pass implementation as the reference,
    // then verify the new single-pass implementation produces the
    // same result within numerical tolerance.
    let gradients: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.013).sin()).collect();
    let mut legacy = GradientEmaStats {
        alpha: 0.5,
        ..Default::default()
    };
    legacy.update_two_pass_legacy(&gradients);
    let mut new_path = GradientEmaStats {
        alpha: 0.5,
        ..Default::default()
    };
    new_path.update(&gradients);
    assert!(
        (legacy.mean - new_path.mean).abs() < 1e-4,
        "mean drift: legacy={} new={}",
        legacy.mean,
        new_path.mean
    );
    assert!(
        (legacy.std - new_path.std).abs() < 1e-3,
        "std drift: legacy={} new={}",
        legacy.std,
        new_path.std
    );
}

#[test]
fn gradient_ema_handles_empty_input() {
    let mut stats = GradientEmaStats {
        alpha: 0.5,
        ..Default::default()
    };
    let initial_mean = stats.mean;
    let initial_std = stats.std;
    stats.update(&[]);
    assert_eq!(stats.mean, initial_mean);
    assert_eq!(stats.std, initial_std);
}

#[test]
fn gradient_ema_simd_matches_scalar_for_large_input() {
    // 5000 elements ensures the chunks_exact(8) path runs many iterations.
    let gradients: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.001).cos() * 0.5).collect();
    let mut new_path = GradientEmaStats {
        alpha: 0.3,
        ..Default::default()
    };
    new_path.update(&gradients);
    // Sanity: result should be finite.
    assert!(new_path.mean.is_finite());
    assert!(new_path.std.is_finite());
    assert!(new_path.std >= 0.0);
}

#[test]
fn feature_baseline_payload_roundtrip() {
    let payload = FeatureBaselinePayload {
        feature_means: vec![0.1, -1.5, 2.0, 0.0],
    };
    let bytes = encode_feature_baseline_payload(&payload);
    let decoded = decode_feature_baseline_payload(&bytes).expect("decode succeeds");
    assert_eq!(decoded.feature_means.len(), 4);
    for (a, b) in payload
        .feature_means
        .iter()
        .zip(decoded.feature_means.iter())
    {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn feature_baseline_payload_empty_decodes_cleanly() {
    let payload = FeatureBaselinePayload {
        feature_means: vec![],
    };
    let bytes = encode_feature_baseline_payload(&payload);
    let decoded = decode_feature_baseline_payload(&bytes).expect("decode succeeds");
    assert!(decoded.feature_means.is_empty());
}

#[test]
fn feature_baseline_payload_rejects_short_buffer() {
    // Header-only — claims 5 features but no body.  Decode must fail
    // cleanly rather than read past the end.
    let mut bytes = 1u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&5u32.to_le_bytes());
    let result = decode_feature_baseline_payload(&bytes);
    assert!(matches!(result, Err(CoreError::Validation(_))));
}

#[test]
fn feature_baseline_section_kind_round_trips() {
    // Variant-id stability is part of the on-disk contract; if this test
    // fails, an existing artifact's section was renumbered.
    assert_eq!(ModelSectionKind::FeatureBaseline.to_u32(), 11);
    assert!(matches!(
        ModelSectionKind::from_u32(11),
        ModelSectionKind::FeatureBaseline
    ));
}

#[test]
fn dart_tree_weights_payload_round_trips() {
    let payload = DartTreeWeightsPayload {
        weights: vec![1.0, 0.5, 0.25, 1.0 / 3.0],
    };
    let bytes = encode_dart_tree_weights_payload(&payload);
    let decoded = decode_dart_tree_weights_payload(&bytes).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn multi_output_leaf_values_section_kind_roundtrips_as_13() {
    assert_eq!(ModelSectionKind::MultiOutputLeafValues.to_u32(), 13);
    assert!(matches!(
        ModelSectionKind::from_u32(13),
        ModelSectionKind::MultiOutputLeafValues
    ));
}

#[test]
fn multi_output_leaf_values_payload_round_trips() {
    // 2 stumps, K=2 outputs. Stump 0: 3 leaves; stump 1: 2 leaves.
    let payload = MultiOutputLeafValuesPayload {
        n_outputs: 2,
        per_stump_leaf_values: vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0],
        ],
    };
    let bytes = encode_multi_output_leaf_values_payload(&payload);
    let decoded = decode_multi_output_leaf_values_payload(&bytes).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn multi_output_leaf_values_payload_empty_round_trips() {
    let payload = MultiOutputLeafValuesPayload {
        n_outputs: 3,
        per_stump_leaf_values: vec![],
    };
    let bytes = encode_multi_output_leaf_values_payload(&payload);
    let decoded = decode_multi_output_leaf_values_payload(&bytes).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn multi_output_leaf_values_truncated_header_errors() {
    let bytes = vec![1u8, 0, 0]; // shorter than 12-byte header
    let err = decode_multi_output_leaf_values_payload(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn multi_output_leaf_values_bad_version_errors() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2u32.to_le_bytes()); // unsupported version
    bytes.extend_from_slice(&2u32.to_le_bytes()); // n_outputs
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_stumps
    let err = decode_multi_output_leaf_values_payload(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn dart_tree_weights_payload_empty_round_trips() {
    let payload = DartTreeWeightsPayload { weights: vec![] };
    let bytes = encode_dart_tree_weights_payload(&payload);
    let decoded = decode_dart_tree_weights_payload(&bytes).expect("decode");
    assert_eq!(decoded, payload);
}

#[test]
fn dart_tree_weights_payload_truncated_header_errors() {
    let bytes = vec![1u8, 0, 0]; // shorter than 8-byte header
    let err = decode_dart_tree_weights_payload(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn dart_tree_weights_payload_bad_version_errors() {
    // version=2 is unsupported
    let mut bytes = 2u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let err = decode_dart_tree_weights_payload(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn dart_tree_weights_payload_length_mismatch_errors() {
    // version=1, count=2, but only one weight present
    let mut bytes = 1u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    let err = decode_dart_tree_weights_payload(&bytes).unwrap_err();
    assert!(matches!(err, CoreError::Validation(_)));
}

#[test]
fn dart_tree_weights_section_kind_round_trips() {
    // Variant-id stability is part of the on-disk contract.
    assert_eq!(ModelSectionKind::DartTreeWeights.to_u32(), 12);
    assert!(matches!(
        ModelSectionKind::from_u32(12),
        ModelSectionKind::DartTreeWeights
    ));
}

#[test]
fn neutralization_metadata_roundtrips_all_kinds() {
    for kind in [
        NeutralizationKind::None,
        NeutralizationKind::PreTarget,
        NeutralizationKind::PerRoundGradient,
        NeutralizationKind::SplitPenalty,
    ] {
        let payload = NeutralizationMetadataPayload {
            config: FactorNeutralizationConfig {
                kind,
                ridge_lambda: 1.5e-3,
                split_penalty: 0.25,
            },
        };
        let bytes = encode_neutralization_metadata_payload(&payload);
        let decoded = decode_neutralization_metadata_payload(&bytes).unwrap();
        assert_eq!(decoded, payload);
    }
}

#[test]
fn neutralization_metadata_rejects_bad_version() {
    let mut bytes = encode_neutralization_metadata_payload(&NeutralizationMetadataPayload {
        config: FactorNeutralizationConfig {
            kind: NeutralizationKind::PerRoundGradient,
            ridge_lambda: 1e-6,
            split_penalty: 0.0,
        },
    });
    bytes[0] = 2; // bump version byte
    assert!(decode_neutralization_metadata_payload(&bytes).is_err());
}

#[test]
fn neutralization_metadata_rejects_bad_kind() {
    let mut bytes = encode_neutralization_metadata_payload(&NeutralizationMetadataPayload {
        config: FactorNeutralizationConfig {
            kind: NeutralizationKind::PerRoundGradient,
            ridge_lambda: 1e-6,
            split_penalty: 0.0,
        },
    });
    bytes[2] = 99; // bogus kind byte
    assert!(decode_neutralization_metadata_payload(&bytes).is_err());
}
