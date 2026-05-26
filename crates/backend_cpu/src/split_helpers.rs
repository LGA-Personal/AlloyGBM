use alloygbm_core::{SplitCandidate, leaf_gain_term};
use alloygbm_engine::{MorphContext, SplitSelectionOptions};

/// Controls which gain formula is used inside `best_split_for_feature_inner`.
///
/// `Standard` uses the XGBoost gain formula.
/// `Morph` delegates to `compute_morph_gain` from the morph module.
pub(crate) enum GainStrategy<'a> {
    Standard,
    Morph(&'a MorphContext),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarSideStats {
    pub(crate) grad: f32,
    pub(crate) hess: f32,
    pub(crate) grad_sq: f32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MissingDirectionCandidate {
    pub(crate) left: ScalarSideStats,
    pub(crate) right: ScalarSideStats,
    pub(crate) default_left: bool,
}

/// Apply a per-feature weight to a split candidate's gain for cross-feature comparison.
///
/// The gain stored in `SplitCandidate` remains unweighted (the true gain);
/// the weighted gain is only used when comparing splits across features.
pub(crate) fn apply_feature_weight(candidate: &SplitCandidate, feature_weights: &[f32]) -> f32 {
    let fi = candidate.feature_index as usize;
    if fi < feature_weights.len() {
        candidate.gain * feature_weights[fi]
    } else {
        candidate.gain
    }
}

pub(crate) fn l1_threshold_gradient(grad_sum: f32, l1_alpha: f32) -> f32 {
    if l1_alpha <= 0.0 {
        return grad_sum;
    }
    if grad_sum > l1_alpha {
        grad_sum - l1_alpha
    } else if grad_sum < -l1_alpha {
        grad_sum + l1_alpha
    } else {
        0.0
    }
}

pub(crate) fn split_gain_term(
    grad_sum: f32,
    hess_sum: f32,
    grad_sq_sum: f32,
    row_count: u32,
    options: &SplitSelectionOptions,
) -> f32 {
    2.0 * leaf_gain_term(
        grad_sum,
        hess_sum,
        grad_sq_sum,
        row_count,
        options.l1_alpha,
        options.l2_lambda,
        options.dro_config.as_ref(),
    )
}

pub(crate) fn categorical_bitset_for_prefix(
    num_categories: usize,
    categories: &[(u16, f32, f32, f32, u32)],
    prefix_end: usize,
) -> Vec<u8> {
    let bitset_len = num_categories.div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    categorical_bitset_for_prefix_into(num_categories, categories, prefix_end, &mut bitset);
    bitset
}

pub(crate) fn categorical_bitset_for_prefix_into(
    num_categories: usize,
    categories: &[(u16, f32, f32, f32, u32)],
    prefix_end: usize,
    bitset: &mut Vec<u8>,
) {
    let bitset_len = num_categories.div_ceil(8);
    bitset.clear();
    bitset.resize(bitset_len, 0);
    for &(bin_id, _, _, _, _) in &categories[..=prefix_end] {
        let byte_idx = (bin_id / 8) as usize;
        let bit_idx = (bin_id % 8) as usize;
        if byte_idx < bitset.len() {
            bitset[byte_idx] |= 1 << bit_idx;
        }
    }
}

/// Determine if a row goes to the left child for a given split.
/// Handles both continuous (threshold comparison) and categorical (bitset membership) splits.
#[inline]
pub(crate) fn goes_left_for_split(bin_val: u16, missing: u16, split: &SplitCandidate) -> bool {
    if bin_val == missing {
        split.default_left
    } else if split.is_categorical {
        split
            .categorical_bitset
            .as_ref()
            .map_or(split.default_left, |bs| {
                let byte_idx = (bin_val / 8) as usize;
                let bit_idx = (bin_val % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            })
    } else {
        bin_val <= split.threshold_bin
    }
}
