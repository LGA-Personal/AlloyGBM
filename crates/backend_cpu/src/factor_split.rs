use alloygbm_engine::{EngineError, EngineResult, FactorSplitContext};

pub(crate) struct FactorSplitScratch {
    left_factor_sums: Vec<f32>,
    right_factor_sums: Vec<f32>,
    missing_factor_sums: Vec<f32>,
    non_missing_factor_sums: Vec<f32>,
    bin_factor_sums: Vec<f32>,
    categorical_bin_order: Vec<usize>,
    numeric_scan_limit: usize,
    categorical_scan_limit: usize,
    pub(crate) categorical_bitset: Vec<u8>,
}

impl FactorSplitScratch {
    pub(crate) fn new(factor_count: usize) -> Self {
        Self {
            left_factor_sums: vec![0.0; factor_count],
            right_factor_sums: vec![0.0; factor_count],
            missing_factor_sums: vec![0.0; factor_count],
            non_missing_factor_sums: vec![0.0; factor_count],
            bin_factor_sums: Vec::new(),
            categorical_bin_order: Vec::new(),
            numeric_scan_limit: 0,
            categorical_scan_limit: 0,
            categorical_bitset: Vec::new(),
        }
    }

    #[cfg(test)]
    fn clear_factor_sums(&mut self) {
        self.left_factor_sums.fill(0.0);
        self.right_factor_sums.fill(0.0);
    }

    pub(crate) fn prepare_numeric_prefix(
        &mut self,
        context: &FactorSplitContext<'_>,
        feature_index: usize,
        scan_limit: usize,
        missing_bin: usize,
    ) {
        let factor_count = context.exposures.factor_count;
        self.numeric_scan_limit = scan_limit;
        self.left_factor_sums.resize(factor_count, 0.0);
        self.right_factor_sums.resize(factor_count, 0.0);
        self.missing_factor_sums.resize(factor_count, 0.0);
        self.non_missing_factor_sums.resize(factor_count, 0.0);
        self.bin_factor_sums
            .resize(scan_limit.saturating_mul(factor_count), 0.0);

        self.left_factor_sums.fill(0.0);
        self.right_factor_sums.fill(0.0);
        self.missing_factor_sums.fill(0.0);
        self.non_missing_factor_sums.fill(0.0);
        self.bin_factor_sums.fill(0.0);

        if context.factor_penalty == 0.0 || scan_limit == 0 {
            return;
        }

        let feature_count = context.binned_matrix.feature_count;
        for &row_index in context.row_indices {
            let row_index = row_index as usize;
            let bin = context
                .binned_matrix
                .row_bin(row_index * feature_count + feature_index) as usize;
            let exposure_start = row_index * factor_count;
            let exposure_row =
                &context.exposures.values[exposure_start..exposure_start + factor_count];
            if bin == missing_bin {
                for (sum, exposure) in self.missing_factor_sums.iter_mut().zip(exposure_row) {
                    *sum += *exposure;
                }
            } else if bin < scan_limit {
                let bin_base = bin * factor_count;
                for (factor_index, exposure) in exposure_row.iter().enumerate().take(factor_count) {
                    self.bin_factor_sums[bin_base + factor_index] += *exposure;
                    self.non_missing_factor_sums[factor_index] += *exposure;
                }
            }
        }
    }

    pub(crate) fn add_numeric_threshold_bin_to_left(&mut self, threshold_bin: usize) {
        if threshold_bin >= self.numeric_scan_limit {
            return;
        }
        let factor_count = self.left_factor_sums.len();
        let bin_base = threshold_bin * factor_count;
        for factor_index in 0..factor_count {
            self.left_factor_sums[factor_index] += self.bin_factor_sums[bin_base + factor_index];
        }
    }

    pub(crate) fn prepare_categorical_prefix(
        &mut self,
        context: &FactorSplitContext<'_>,
        feature_index: usize,
        sorted_category_bins: &[u16],
        missing_bin: usize,
    ) {
        let factor_count = context.exposures.factor_count;
        self.categorical_scan_limit = sorted_category_bins.len();
        self.left_factor_sums.resize(factor_count, 0.0);
        self.right_factor_sums.resize(factor_count, 0.0);
        self.missing_factor_sums.resize(factor_count, 0.0);
        self.non_missing_factor_sums.resize(factor_count, 0.0);
        self.bin_factor_sums
            .resize(sorted_category_bins.len().saturating_mul(factor_count), 0.0);

        self.left_factor_sums.fill(0.0);
        self.right_factor_sums.fill(0.0);
        self.missing_factor_sums.fill(0.0);
        self.non_missing_factor_sums.fill(0.0);
        self.bin_factor_sums.fill(0.0);

        let order_map_len = sorted_category_bins
            .iter()
            .copied()
            .max()
            .map(|bin| bin as usize + 1)
            .unwrap_or(0);
        self.categorical_bin_order.resize(order_map_len, usize::MAX);
        self.categorical_bin_order.fill(usize::MAX);
        for (order_index, &bin) in sorted_category_bins.iter().enumerate() {
            let bin = bin as usize;
            if bin < self.categorical_bin_order.len() {
                self.categorical_bin_order[bin] = order_index;
            }
        }

        if context.factor_penalty == 0.0 || sorted_category_bins.is_empty() {
            return;
        }

        let feature_count = context.binned_matrix.feature_count;
        for &row_index in context.row_indices {
            let row_index = row_index as usize;
            let bin = context
                .binned_matrix
                .row_bin(row_index * feature_count + feature_index) as usize;
            let exposure_start = row_index * factor_count;
            let exposure_row =
                &context.exposures.values[exposure_start..exposure_start + factor_count];
            if bin == missing_bin {
                for (sum, exposure) in self.missing_factor_sums.iter_mut().zip(exposure_row) {
                    *sum += *exposure;
                }
                continue;
            }

            for (sum, exposure) in self.non_missing_factor_sums.iter_mut().zip(exposure_row) {
                *sum += *exposure;
            }
            if bin < self.categorical_bin_order.len() {
                let order_index = self.categorical_bin_order[bin];
                if order_index != usize::MAX {
                    let bin_base = order_index * factor_count;
                    for (factor_index, exposure) in
                        exposure_row.iter().enumerate().take(factor_count)
                    {
                        self.bin_factor_sums[bin_base + factor_index] += *exposure;
                    }
                }
            }
        }
    }

    pub(crate) fn add_categorical_prefix_bin_to_left(&mut self, category_order_index: usize) {
        if category_order_index >= self.categorical_scan_limit {
            return;
        }
        let factor_count = self.left_factor_sums.len();
        let bin_base = category_order_index * factor_count;
        for factor_index in 0..factor_count {
            self.left_factor_sums[factor_index] += self.bin_factor_sums[bin_base + factor_index];
        }
    }

    fn prefix_penalty(
        &self,
        default_left: bool,
        left_leaf_value: f32,
        right_leaf_value: f32,
        factor_penalty: f32,
        row_count: usize,
    ) -> f32 {
        if factor_penalty == 0.0 {
            return 0.0;
        }
        let mut norm_sq = 0.0_f32;
        for factor_index in 0..self.left_factor_sums.len() {
            let prefix_left = self.left_factor_sums[factor_index];
            let missing = self.missing_factor_sums[factor_index];
            let non_missing_right = self.non_missing_factor_sums[factor_index] - prefix_left;
            let left_sum = if default_left {
                prefix_left + missing
            } else {
                prefix_left
            };
            let right_sum = if default_left {
                non_missing_right
            } else {
                non_missing_right + missing
            };
            let load = left_sum * left_leaf_value + right_sum * right_leaf_value;
            norm_sq += load * load;
        }
        factor_penalty * norm_sq / row_count.max(1) as f32
    }

    pub(crate) fn numeric_prefix_penalty(
        &self,
        default_left: bool,
        left_leaf_value: f32,
        right_leaf_value: f32,
        factor_penalty: f32,
        row_count: usize,
    ) -> f32 {
        self.prefix_penalty(
            default_left,
            left_leaf_value,
            right_leaf_value,
            factor_penalty,
            row_count,
        )
    }

    pub(crate) fn categorical_prefix_penalty(
        &self,
        default_left: bool,
        left_leaf_value: f32,
        right_leaf_value: f32,
        factor_penalty: f32,
        row_count: usize,
    ) -> f32 {
        self.prefix_penalty(
            default_left,
            left_leaf_value,
            right_leaf_value,
            factor_penalty,
            row_count,
        )
    }
}

#[cfg(test)]
pub(crate) struct FactorSplitCandidate<'a> {
    pub(crate) feature_index: u32,
    pub(crate) threshold_bin: u16,
    pub(crate) default_left: bool,
    pub(crate) categorical_bitset: Option<&'a [u8]>,
    pub(crate) left_leaf_value: f32,
    pub(crate) right_leaf_value: f32,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{BinnedMatrix, FactorExposureMatrix};

    fn factor_fixture() -> (BinnedMatrix, FactorExposureMatrix, Vec<u32>) {
        let binned = BinnedMatrix::new(5, 1, 3, vec![0_u8, 1, 2, 3, alloygbm_core::MISSING_BIN_U8])
            .expect("binned matrix");
        let exposures = FactorExposureMatrix {
            row_count: 5,
            factor_count: 2,
            values: vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0],
        };
        let rows = vec![0_u32, 1, 2, 3, 4];
        (binned, exposures, rows)
    }

    #[test]
    fn numeric_prefix_penalty_matches_row_scan_for_each_threshold_and_missing_direction() {
        let (binned, exposures, rows) = factor_fixture();
        let context = FactorSplitContext {
            binned_matrix: &binned,
            exposures: &exposures,
            row_indices: &rows,
            factor_penalty: 0.75,
        };
        let scan_limit = 4;
        let mut prefix_scratch = FactorSplitScratch::new(exposures.factor_count);
        prefix_scratch.prepare_numeric_prefix(
            &context,
            0,
            scan_limit,
            binned.missing_bin() as usize,
        );
        let mut row_scan_scratch = FactorSplitScratch::new(exposures.factor_count);

        for threshold_bin in 0..scan_limit {
            prefix_scratch.add_numeric_threshold_bin_to_left(threshold_bin);
            for default_left in [true, false] {
                let slow = factor_split_penalty_for_candidate(
                    &context,
                    &mut row_scan_scratch,
                    FactorSplitCandidate {
                        feature_index: 0,
                        threshold_bin: threshold_bin as u16,
                        default_left,
                        categorical_bitset: None,
                        left_leaf_value: 0.25,
                        right_leaf_value: -0.5,
                    },
                );
                let fast = prefix_scratch.numeric_prefix_penalty(
                    default_left,
                    0.25,
                    -0.5,
                    context.factor_penalty,
                    context.row_indices.len(),
                );
                assert!(
                    (slow - fast).abs() < 1e-6,
                    "threshold={threshold_bin} default_left={default_left} slow={slow} fast={fast}"
                );
            }
        }
    }

    #[test]
    fn numeric_prefix_penalty_is_zero_when_factor_penalty_is_zero() {
        let (binned, exposures, rows) = factor_fixture();
        let context = FactorSplitContext {
            binned_matrix: &binned,
            exposures: &exposures,
            row_indices: &rows,
            factor_penalty: 0.0,
        };
        let mut scratch = FactorSplitScratch::new(exposures.factor_count);
        scratch.prepare_numeric_prefix(&context, 0, 4, binned.missing_bin() as usize);
        scratch.add_numeric_threshold_bin_to_left(0);
        assert_eq!(
            scratch.numeric_prefix_penalty(true, 0.25, -0.5, 0.0, rows.len()),
            0.0
        );
    }

    #[test]
    fn categorical_prefix_penalty_matches_row_scan_for_each_prefix_and_missing_direction() {
        let (binned, exposures, rows) = factor_fixture();
        let context = FactorSplitContext {
            binned_matrix: &binned,
            exposures: &exposures,
            row_indices: &rows,
            factor_penalty: 0.75,
        };
        let sorted_categories = [2_u16, 0, 3, 1];
        let mut prefix_scratch = FactorSplitScratch::new(exposures.factor_count);
        prefix_scratch.prepare_categorical_prefix(
            &context,
            0,
            &sorted_categories,
            binned.missing_bin() as usize,
        );
        let mut row_scan_scratch = FactorSplitScratch::new(exposures.factor_count);
        let mut bitset = vec![0_u8; 1];

        for prefix_index in 0..sorted_categories.len() - 1 {
            prefix_scratch.add_categorical_prefix_bin_to_left(prefix_index);
            let bin = sorted_categories[prefix_index];
            bitset[(bin / 8) as usize] |= 1 << (bin % 8);

            for default_left in [true, false] {
                let slow = factor_split_penalty_for_candidate(
                    &context,
                    &mut row_scan_scratch,
                    FactorSplitCandidate {
                        feature_index: 0,
                        threshold_bin: 0,
                        default_left,
                        categorical_bitset: Some(&bitset),
                        left_leaf_value: 0.25,
                        right_leaf_value: -0.5,
                    },
                );
                let fast = prefix_scratch.categorical_prefix_penalty(
                    default_left,
                    0.25,
                    -0.5,
                    context.factor_penalty,
                    context.row_indices.len(),
                );
                assert!(
                    (slow - fast).abs() < 1e-6,
                    "prefix_index={prefix_index} default_left={default_left} slow={slow} fast={fast}"
                );
            }
        }
    }
}
