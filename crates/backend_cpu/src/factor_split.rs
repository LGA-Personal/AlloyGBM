use alloygbm_engine::{EngineError, EngineResult, FactorSplitContext};

pub(crate) struct FactorSplitScratch {
    left_factor_sums: Vec<f32>,
    right_factor_sums: Vec<f32>,
    pub(crate) categorical_bitset: Vec<u8>,
}

impl FactorSplitScratch {
    pub(crate) fn new(factor_count: usize) -> Self {
        Self {
            left_factor_sums: vec![0.0; factor_count],
            right_factor_sums: vec![0.0; factor_count],
            categorical_bitset: Vec::new(),
        }
    }

    fn clear_factor_sums(&mut self) {
        self.left_factor_sums.fill(0.0);
        self.right_factor_sums.fill(0.0);
    }
}

pub(crate) struct FactorSplitCandidate<'a> {
    pub(crate) feature_index: u32,
    pub(crate) threshold_bin: u16,
    pub(crate) default_left: bool,
    pub(crate) categorical_bitset: Option<&'a [u8]>,
    pub(crate) left_leaf_value: f32,
    pub(crate) right_leaf_value: f32,
}

pub(crate) fn factor_split_penalty_for_candidate(
    context: &FactorSplitContext<'_>,
    scratch: &mut FactorSplitScratch,
    candidate: FactorSplitCandidate<'_>,
) -> f32 {
    if context.factor_penalty == 0.0 {
        return 0.0;
    }

    let factor_count = context.exposures.factor_count;
    scratch.clear_factor_sums();
    let feature_index = candidate.feature_index as usize;
    let feature_count = context.binned_matrix.feature_count;
    let missing = context.binned_matrix.missing_bin();

    for &row_index in context.row_indices {
        let row_index = row_index as usize;
        let bin = context
            .binned_matrix
            .row_bin(row_index * feature_count + feature_index);
        let goes_left = if bin == missing {
            candidate.default_left
        } else if let Some(bitset) = candidate.categorical_bitset {
            let byte_idx = (bin / 8) as usize;
            let bit_idx = (bin % 8) as usize;
            byte_idx < bitset.len() && (bitset[byte_idx] & (1 << bit_idx)) != 0
        } else {
            bin <= candidate.threshold_bin
        };
        let exposure_start = row_index * factor_count;
        let exposure_row = &context.exposures.values[exposure_start..exposure_start + factor_count];
        let target_sums = if goes_left {
            &mut scratch.left_factor_sums
        } else {
            &mut scratch.right_factor_sums
        };
        for (sum, exposure) in target_sums.iter_mut().zip(exposure_row) {
            *sum += *exposure;
        }
    }

    factor_split_penalty(
        &scratch.left_factor_sums,
        &scratch.right_factor_sums,
        candidate.left_leaf_value,
        candidate.right_leaf_value,
        context.factor_penalty,
        context.row_indices.len(),
    )
}

pub(crate) fn factor_split_penalty(
    left_factor_sums: &[f32],
    right_factor_sums: &[f32],
    left_leaf_value: f32,
    right_leaf_value: f32,
    factor_penalty: f32,
    row_count: usize,
) -> f32 {
    if factor_penalty == 0.0 {
        return 0.0;
    }
    let mut norm_sq = 0.0_f32;
    for i in 0..left_factor_sums.len() {
        let load = left_factor_sums[i] * left_leaf_value + right_factor_sums[i] * right_leaf_value;
        norm_sq += load * load;
    }
    factor_penalty * norm_sq / row_count.max(1) as f32
}

pub(crate) fn validate_factor_split_context(context: &FactorSplitContext<'_>) -> EngineResult<()> {
    if !context.factor_penalty.is_finite() || context.factor_penalty < 0.0 {
        return Err(EngineError::ContractViolation(
            "factor split penalty must be finite and >= 0".to_string(),
        ));
    }
    if context.exposures.factor_count == 0 {
        return Err(EngineError::ContractViolation(
            "factor_exposures factor_count must be greater than 0".to_string(),
        ));
    }
    let expected_len = context
        .exposures
        .row_count
        .checked_mul(context.exposures.factor_count)
        .ok_or_else(|| {
            EngineError::ContractViolation(
                "factor_exposures row_count * factor_count overflow".to_string(),
            )
        })?;
    if context.exposures.values.len() != expected_len {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures values length {} does not match row_count * factor_count {}",
            context.exposures.values.len(),
            expected_len
        )));
    }
    if context
        .exposures
        .values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(EngineError::ContractViolation(
            "factor_exposures must contain only finite values".to_string(),
        ));
    }
    if context.exposures.row_count != context.binned_matrix.row_count {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures row_count {} does not match binned matrix row_count {}",
            context.exposures.row_count, context.binned_matrix.row_count
        )));
    }
    for &row_index in context.row_indices {
        let row_index = row_index as usize;
        if row_index >= context.exposures.row_count {
            return Err(EngineError::ContractViolation(format!(
                "factor split context row index {row_index} is out of bounds for row_count {}",
                context.exposures.row_count
            )));
        }
    }
    Ok(())
}
