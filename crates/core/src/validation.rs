use crate::artifact_format::{
    MAX_MODEL_ARTIFACT_SECTIONS, MAX_MODEL_SECTION_PAYLOAD_BYTES, MODEL_BINARY_HEADER_LEN,
    MODEL_BINARY_MAGIC, MODEL_FORMAT_V1, MODEL_SECTION_DESCRIPTOR_LEN, ModelIoContractV1,
};
use crate::binned::{BinStorage, BinnedMatrix};
use crate::config::{BoostingMode, LeafModelKind, LeafSolverKind, TrainParams, TreeGrowth};
use crate::dataset::{
    ColumnarMatrixView, DatasetMatrix, DatasetSchema, DenseMatrixView, TrainingDataset,
};
use crate::error::{CoreError, CoreResult};
use crate::neutralization::NeutralizationKind;
use crate::training_mode::LrSchedule;

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

    if !(0.0..=1.0).contains(&params.row_subsample) || params.row_subsample == 0.0 {
        return Err(CoreError::InvalidConfig(
            "row_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&params.col_subsample) || params.col_subsample == 0.0 {
        return Err(CoreError::InvalidConfig(
            "col_subsample must be in (0.0, 1.0]".to_string(),
        ));
    }

    if let Some(rounds) = params.early_stopping_rounds
        && rounds == 0
    {
        return Err(CoreError::InvalidConfig(
            "early_stopping_rounds must be greater than 0 when set".to_string(),
        ));
    }

    if !params.min_validation_improvement.is_finite() || params.min_validation_improvement < 0.0 {
        return Err(CoreError::InvalidConfig(
            "min_validation_improvement must be finite and >= 0".to_string(),
        ));
    }

    if params.min_data_in_leaf == 0 {
        return Err(CoreError::InvalidConfig(
            "min_data_in_leaf must be greater than 0".to_string(),
        ));
    }

    if !params.lambda_l1.is_finite() || params.lambda_l1 < 0.0 {
        return Err(CoreError::InvalidConfig(
            "lambda_l1 must be finite and >= 0".to_string(),
        ));
    }

    if !params.lambda_l2.is_finite() || params.lambda_l2 < 0.0 {
        return Err(CoreError::InvalidConfig(
            "lambda_l2 must be finite and >= 0".to_string(),
        ));
    }

    if !params.min_child_hessian.is_finite() || params.min_child_hessian < 0.0 {
        return Err(CoreError::InvalidConfig(
            "min_child_hessian must be finite and >= 0".to_string(),
        ));
    }

    if !params.poisson_max_delta_step.is_finite() || params.poisson_max_delta_step < 0.0 {
        return Err(CoreError::InvalidConfig(
            "poisson_max_delta_step must be finite and >= 0".to_string(),
        ));
    }

    for &c in &params.monotone_constraints {
        if c != -1 && c != 0 && c != 1 {
            return Err(CoreError::InvalidConfig(
                "monotone_constraints values must be -1, 0, or +1".to_string(),
            ));
        }
    }

    for &w in &params.feature_weights {
        if !w.is_finite() || w < 0.0 {
            return Err(CoreError::InvalidConfig(
                "feature_weights values must be finite and >= 0".to_string(),
            ));
        }
    }

    if params.interaction_constraints.len() > 64 {
        return Err(CoreError::InvalidConfig(format!(
            "interaction_constraints supports at most 64 groups (got {})",
            params.interaction_constraints.len()
        )));
    }
    for (gi, group) in params.interaction_constraints.iter().enumerate() {
        if group.is_empty() {
            return Err(CoreError::InvalidConfig(format!(
                "interaction_constraints group {gi} is empty; groups must contain at least one feature index"
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for &f in group {
            if !seen.insert(f) {
                return Err(CoreError::InvalidConfig(format!(
                    "interaction_constraints group {gi} contains duplicate feature index {f}"
                )));
            }
        }
    }

    if let Some(max_leaves) = params.max_leaves
        && max_leaves < 2
    {
        return Err(CoreError::InvalidConfig(
            "max_leaves must be >= 2 when set (a tree needs at least 2 leaves)".to_string(),
        ));
    }

    if params.tree_growth == TreeGrowth::Leaf && params.max_leaves.is_none() {
        return Err(CoreError::InvalidConfig(
            "tree_growth='leaf' requires max_leaves to be set".to_string(),
        ));
    }

    if let Some(config) = params.neutralization_config {
        if config.kind == NeutralizationKind::None {
            return Err(CoreError::InvalidConfig(
                "neutralization_config must be None when neutralization kind is None".to_string(),
            ));
        }
        if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_neutralization_lambda must be finite and >= 0".to_string(),
            ));
        }
        if !config.split_penalty.is_finite() || config.split_penalty < 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_penalty must be finite and >= 0".to_string(),
            ));
        }
        if config.kind != NeutralizationKind::SplitPenalty && config.split_penalty != 0.0 {
            return Err(CoreError::InvalidConfig(
                "factor_penalty is only valid with neutralization='split_penalty'".to_string(),
            ));
        }
        if config.kind == NeutralizationKind::SplitPenalty
            && params.leaf_model == LeafModelKind::Linear
        {
            return Err(CoreError::InvalidConfig(
                "neutralization='split_penalty' requires leaf_model='constant'".to_string(),
            ));
        }
    }

    if params.leaf_solver == LeafSolverKind::Dro {
        if params.leaf_model != LeafModelKind::Constant {
            return Err(CoreError::InvalidConfig(
                "leaf_solver='dro' requires leaf_model='constant'".to_string(),
            ));
        }
        let Some(cfg) = params.dro_config else {
            return Err(CoreError::InvalidConfig(
                "leaf_solver='dro' requires dro_config".to_string(),
            ));
        };
        if !cfg.radius.is_finite() || cfg.radius < 0.0 {
            return Err(CoreError::InvalidConfig(
                "dro_config.radius must be finite and >= 0".to_string(),
            ));
        }
    } else if params.dro_config.is_some() {
        return Err(CoreError::InvalidConfig(
            "dro_config is only valid with leaf_solver='dro'".to_string(),
        ));
    }

    if let Some(cfg) = &params.morph_config {
        if !cfg.morph_rate.is_finite() || !(0.0..=1.0).contains(&cfg.morph_rate) {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.morph_rate must be in [0, 1], got {}",
                cfg.morph_rate
            )));
        }
        if !cfg.evolution_pressure.is_finite() || cfg.evolution_pressure < 0.0 {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.evolution_pressure must be >= 0, got {}",
                cfg.evolution_pressure
            )));
        }
        if !cfg.info_score_weight.is_finite() || !(0.0..=1.0).contains(&cfg.info_score_weight) {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.info_score_weight must be in [0, 1], got {}",
                cfg.info_score_weight
            )));
        }
        if !cfg.depth_penalty_base.is_finite()
            || cfg.depth_penalty_base <= 0.0
            || cfg.depth_penalty_base > 1.0
        {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.depth_penalty_base must be in (0, 1], got {}",
                cfg.depth_penalty_base
            )));
        }
        if let LrSchedule::WarmupCosine { warmup_frac } = cfg.lr_schedule
            && (!warmup_frac.is_finite() || !(0.0..=1.0).contains(&warmup_frac))
        {
            return Err(CoreError::InvalidConfig(format!(
                "morph_config.lr_schedule.warmup_frac must be in [0, 1], got {}",
                warmup_frac
            )));
        }
    }

    match params.boosting_mode {
        BoostingMode::Standard => {}
        BoostingMode::Goss {
            top_rate,
            other_rate,
        } => {
            if !top_rate.is_finite() || !(0.0..1.0).contains(&top_rate) || top_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_top_rate in (0, 1), got {top_rate}"
                )));
            }
            if !other_rate.is_finite() || !(0.0..1.0).contains(&other_rate) || other_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_other_rate in (0, 1), got {other_rate}"
                )));
            }
            if top_rate + other_rate > 1.0 + f32::EPSILON {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='goss' requires goss_top_rate + goss_other_rate <= 1.0 (got {} + {} = {})",
                    top_rate,
                    other_rate,
                    top_rate + other_rate
                )));
            }
        }
        BoostingMode::Dart {
            drop_rate,
            max_drop,
            ..
        } => {
            if !drop_rate.is_finite() || !(0.0..1.0).contains(&drop_rate) || drop_rate == 0.0 {
                return Err(CoreError::InvalidConfig(format!(
                    "boosting_mode='dart' requires dart_drop_rate in (0, 1), got {drop_rate}"
                )));
            }
            if max_drop == 0 {
                return Err(CoreError::InvalidConfig(
                    "boosting_mode='dart' requires dart_max_drop >= 1".to_string(),
                ));
            }
        }
    }

    Ok(())
}

pub fn validate_dataset_schema(schema: &DatasetSchema) -> CoreResult<()> {
    if schema.feature_count == 0 {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }

    let mut previous = None;
    for &feature_index in &schema.categorical_feature_indices {
        if feature_index >= schema.feature_count {
            return Err(CoreError::Validation(format!(
                "categorical feature index {feature_index} is out of bounds for feature_count {}",
                schema.feature_count
            )));
        }
        if let Some(previous) = previous
            && feature_index <= previous
        {
            return Err(CoreError::Validation(format!(
                "categorical feature indices must be strictly increasing (found {feature_index} after {previous})"
            )));
        }
        previous = Some(feature_index);
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
    // Allow empty values for metadata-only matrices (no categorical encoding).
    if !matrix.values.is_empty() && matrix.values.len() != matrix.row_count * matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "matrix values length {} does not match row_count * feature_count {}",
            matrix.values.len(),
            matrix.row_count * matrix.feature_count
        )));
    }
    Ok(())
}

pub fn validate_dense_matrix_view(matrix: &DenseMatrixView<'_>) -> CoreResult<()> {
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

pub fn validate_columnar_matrix_view(matrix: &ColumnarMatrixView<'_>) -> CoreResult<()> {
    if matrix.row_count == 0 {
        return Err(CoreError::Validation(
            "row_count must be greater than 0".to_string(),
        ));
    }
    if matrix.columns.is_empty() {
        return Err(CoreError::Validation(
            "feature_count must be greater than 0".to_string(),
        ));
    }
    for (feature_index, column) in matrix.columns.iter().enumerate() {
        if column.values.len() != matrix.row_count {
            return Err(CoreError::Validation(format!(
                "column {feature_index} length {} does not match row_count {}",
                column.values.len(),
                matrix.row_count
            )));
        }
        if let Some(validity) = column.validity
            && validity.len() != matrix.row_count
        {
            return Err(CoreError::Validation(format!(
                "column {feature_index} validity length {} does not match row_count {}",
                validity.len(),
                matrix.row_count
            )));
        }
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

    if let Some(factor_exposures) = &dataset.factor_exposures {
        if factor_exposures.row_count != dataset.matrix.row_count {
            return Err(CoreError::Validation(format!(
                "factor_exposures row_count {} does not match row_count {}",
                factor_exposures.row_count, dataset.matrix.row_count
            )));
        }
        if factor_exposures.factor_count == 0 {
            return Err(CoreError::Validation(
                "factor_exposures factor_count must be greater than 0".to_string(),
            ));
        }
        let expected_len = factor_exposures
            .row_count
            .checked_mul(factor_exposures.factor_count)
            .ok_or_else(|| {
                CoreError::Validation(
                    "factor_exposures row_count * factor_count overflow".to_string(),
                )
            })?;
        if factor_exposures.values.len() != expected_len {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                factor_exposures.values.len(),
                expected_len
            )));
        }
        if factor_exposures.values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::Validation(
                "factor_exposures must contain only finite values".to_string(),
            ));
        }
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
    if matrix.storage_feature_count == 0 {
        return Err(CoreError::Validation(
            "storage_feature_count must be greater than 0".to_string(),
        ));
    }
    if matrix.storage_max_bin == 0 {
        return Err(CoreError::Validation(
            "storage_max_bin must be greater than 0".to_string(),
        ));
    }
    if let Some(map) = &matrix.feature_bundle_map
        && (map.original_feature_count() != matrix.feature_count
            || map.effective_feature_count() != matrix.storage_feature_count)
    {
        return Err(CoreError::Validation(
            "feature bundle map dimensions do not match binned matrix".to_string(),
        ));
    }
    let expected_len = matrix.row_count * matrix.storage_feature_count;
    if matrix.bins_col_adaptive.len() != expected_len {
        return Err(CoreError::Validation(format!(
            "column-major bins length {} does not match row_count * feature_count {}",
            matrix.bins_col_adaptive.len(),
            expected_len
        )));
    }
    if matrix.has_row_major() && matrix.bins_adaptive.len() != expected_len {
        return Err(CoreError::Validation(format!(
            "row-major bins length {} does not match row_count * feature_count {}",
            matrix.bins_adaptive.len(),
            expected_len
        )));
    }
    // Validate that no stored bin exceeds the physical storage range.
    // The NaN sentinel bin is also allowed (it may exceed max_bin).
    let nan_bin = matrix.nan_bin_index;
    match &matrix.bins_col_adaptive {
        BinStorage::U8(bins) => {
            for &bin in bins {
                let b = u16::from(bin);
                if b > matrix.storage_max_bin && b != nan_bin {
                    return Err(CoreError::Validation(format!(
                        "bin value {bin} exceeds storage_max_bin {}",
                        matrix.storage_max_bin
                    )));
                }
            }
        }
        BinStorage::U16(bins) => {
            for &bin in bins {
                if bin > matrix.storage_max_bin && bin != nan_bin {
                    return Err(CoreError::Validation(format!(
                        "bin value {bin} exceeds storage_max_bin {}",
                        matrix.storage_max_bin
                    )));
                }
            }
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
    if contract.sections.len() > MAX_MODEL_ARTIFACT_SECTIONS {
        return Err(CoreError::Serialization(format!(
            "section_count {} exceeds maximum {MAX_MODEL_ARTIFACT_SECTIONS}",
            contract.sections.len()
        )));
    }

    let descriptor_table_len = contract
        .sections
        .len()
        .checked_mul(MODEL_SECTION_DESCRIPTOR_LEN)
        .ok_or_else(|| CoreError::Serialization("section table length overflow".to_string()))?;
    let payload_start = MODEL_BINARY_HEADER_LEN
        .checked_add(descriptor_table_len)
        .and_then(|value| value.checked_add(contract.header.metadata_json_len as usize))
        .ok_or_else(|| CoreError::Serialization("artifact header length overflow".to_string()))?
        as u64;

    let mut expected_offset = payload_start;
    for section in &contract.sections {
        if section.length == 0 {
            return Err(CoreError::Serialization(
                "section length must be greater than 0".to_string(),
            ));
        }
        if section.length > MAX_MODEL_SECTION_PAYLOAD_BYTES {
            return Err(CoreError::Serialization(format!(
                "section length {} exceeds maximum {MAX_MODEL_SECTION_PAYLOAD_BYTES}",
                section.length
            )));
        }
        if section.offset < payload_start {
            return Err(CoreError::Serialization(format!(
                "section offset {} precedes payload start {payload_start}",
                section.offset
            )));
        }
        if section.offset != expected_offset {
            return Err(CoreError::Serialization(format!(
                "section offsets must be contiguous and ordered (expected {}, found {})",
                expected_offset, section.offset
            )));
        }
        expected_offset = section
            .offset
            .checked_add(section.length)
            .ok_or_else(|| CoreError::Serialization("section offset overflow".to_string()))?;
    }

    Ok(())
}
