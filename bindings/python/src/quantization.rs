use crate::pyclasses::ContinuousBinningMetadataInternal;
use alloygbm_core::{
    BinnedLayout, BinnedMatrix, DatasetMatrix, DenseMatrixView, MISSING_BIN_U8, TrainingDataset,
};
use alloygbm_engine::EngineError;
use rayon::prelude::*;

fn is_pre_binned_integer_value(value: f32) -> bool {
    if value < 0.0 {
        return false;
    }
    let rounded = value.round();
    (value - rounded).abs() <= crate::PRE_BINNED_INTEGER_TOLERANCE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContinuousBinningStrategy {
    Linear,
    Rank,
    Quantile,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedTrainingMatrices {
    pub(crate) dataset: TrainingDataset,
    pub(crate) binned_matrix: BinnedMatrix,
    pub(crate) metadata: ContinuousBinningMetadataInternal,
}

pub(crate) fn parse_continuous_binning_strategy(
    value: &str,
) -> Result<ContinuousBinningStrategy, EngineError> {
    match value {
        "linear" => Ok(ContinuousBinningStrategy::Linear),
        "rank" => Ok(ContinuousBinningStrategy::Rank),
        "quantile" => Ok(ContinuousBinningStrategy::Quantile),
        other => Err(EngineError::InvalidConfig(format!(
            "continuous_binning_strategy must be one of: linear, quantile, rank; received '{other}'"
        ))),
    }
}

fn validate_continuous_binning_max_bins(max_bins: usize) -> Result<(), EngineError> {
    if !(crate::MIN_CONTINUOUS_QUANTIZED_BINS
        ..=(crate::MAX_CONTINUOUS_QUANTIZED_BIN_U16 as usize + 1))
        .contains(&max_bins)
    {
        return Err(EngineError::InvalidConfig(format!(
            "continuous_binning_max_bins must be in [{}, {}]",
            crate::MIN_CONTINUOUS_QUANTIZED_BINS,
            crate::MAX_CONTINUOUS_QUANTIZED_BIN_U16 as usize + 1
        )));
    }
    Ok(())
}

/// Whether the given max_bins requires u16 bin storage.
fn needs_wide_bins(max_bins: usize) -> bool {
    max_bins > (crate::MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1)
}

fn env_toggle_enabled(env_name: &str) -> bool {
    match std::env::var(env_name) {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn linear_tail_rank_enabled_from_env() -> bool {
    env_toggle_enabled(crate::LINEAR_TAIL_RANK_ENV_VAR)
}

fn linear_tail_core_span_ratio_threshold_from_env() -> f32 {
    match std::env::var(crate::LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR) {
        Ok(value) => value
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|parsed| parsed.is_finite())
            .map(|parsed| parsed.clamp(0.0, 1.0))
            .unwrap_or(crate::DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD),
        Err(_) => crate::DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD,
    }
}

fn round_half_away_from_zero(value: f32) -> i32 {
    if value >= 0.0 {
        (value + 0.5).floor() as i32
    } else {
        (value - 0.5).ceil() as i32
    }
}

pub(crate) fn quantize_linear_value(value: f32, min_value: f32, max_value: f32) -> u8 {
    quantize_linear_value_wide(
        value,
        min_value,
        max_value,
        crate::MAX_CONTINUOUS_QUANTIZED_BIN_U8,
    ) as u8
}

fn quantize_rank_value(value: f32, sorted_values: &[f32]) -> u8 {
    quantize_rank_value_wide(value, sorted_values, crate::MAX_CONTINUOUS_QUANTIZED_BIN_U8) as u8
}

/// Parameterized linear quantization that supports arbitrary max_data_bin (u16).
///
/// **NaN handling:** callers that feed the result directly to the
/// predictor (as a quantized f32 bin index) must check `value.is_nan()`
/// themselves and emit `f32::NAN` to the output buffer instead of
/// casting this function's return. The predictor's `predictor_went_left`
/// short-circuits to `default_left` on NaN feature values, matching the
/// pure-linear / pure-quantile paths. See
/// `quantize_dense_values_linear_inplace_wide` and
/// `quantize_dense_values_linear_rank_inplace_wide` for the v0.9.0 NaN
/// preservation pattern — this addresses Limitation 4 in
/// `docs/limitations.md` (v0.8.0).
fn quantize_linear_value_wide(
    value: f32,
    min_value: f32,
    max_value: f32,
    max_data_bin: u16,
) -> u16 {
    if value <= min_value {
        return 0;
    }
    if value >= max_value {
        return max_data_bin;
    }
    let span = max_value - min_value;
    if span <= crate::PRE_BINNED_INTEGER_TOLERANCE {
        return 0;
    }
    let scaled = ((value - min_value) / span) * max_data_bin as f32;
    round_half_away_from_zero(scaled).clamp(0, max_data_bin as i32) as u16
}

/// Parameterized rank quantization that supports arbitrary max_data_bin (u16).
///
/// **NaN handling:** see `quantize_linear_value_wide` — callers feeding
/// the result to the predictor must preserve NaN through the f32 cast.
fn quantize_rank_value_wide(value: f32, sorted_values: &[f32], max_data_bin: u16) -> u16 {
    if sorted_values.len() <= 1 {
        return 0;
    }
    let insertion = sorted_values.partition_point(|probe| *probe <= value);
    let rank = insertion.saturating_sub(1).min(sorted_values.len() - 1);
    let scaled =
        (rank as f32 * max_data_bin as f32) / (sorted_values.len().saturating_sub(1) as f32);
    round_half_away_from_zero(scaled).clamp(0, max_data_bin as i32) as u16
}

#[inline]
fn quantize_quantile_value(value: f32, cuts: &[f32], max_data_bin: u16) -> u16 {
    cuts.partition_point(|probe| *probe <= value)
        .min(usize::from(max_data_bin)) as u16
}

/// Parameterized linear quantize for predict-time: supports arbitrary max_data_bin.
///
/// v0.9.0: NaN inputs are passed through as `f32::NAN` instead of
/// being cast to a finite bin. The predictor's `predictor_went_left`
/// detects NaN and routes through `default_left`. See Limitation 4 in
/// `docs/limitations.md` (resolved in v0.9.0).
pub(crate) fn quantize_dense_values_linear_inplace_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    feature_mins: &[f32],
    feature_maxs: &[f32],
    max_data_bin: u16,
) -> Vec<f32> {
    let total = row_count * feature_count;
    let mut quantized = vec![0.0_f32; total];
    let chunk_size = 4096.max(row_count / rayon::current_num_threads().max(1));
    quantized
        .par_chunks_mut(chunk_size * feature_count)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * chunk_size;
            let rows_in_chunk = out_chunk.len() / feature_count;
            for local_row in 0..rows_in_chunk {
                let row_index = row_start + local_row;
                let base = row_index * feature_count;
                let out_base = local_row * feature_count;
                for fi in 0..feature_count {
                    let value = values[base + fi];
                    out_chunk[out_base + fi] = if value.is_nan() {
                        f32::NAN
                    } else {
                        quantize_linear_value_wide(
                            value,
                            feature_mins[fi],
                            feature_maxs[fi],
                            max_data_bin,
                        ) as f32
                    };
                }
            }
        });
    quantized
}

/// Parameterized linear+rank quantize for predict-time: supports arbitrary max_data_bin.
pub(crate) fn quantize_dense_values_linear_rank_inplace_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    feature_mins: &[f32],
    feature_maxs: &[f32],
    rank_flags: &[bool],
    feature_sorted_values: &[Vec<f32>],
    max_data_bin: u16,
) -> Vec<f32> {
    let total = row_count * feature_count;
    let mut quantized = vec![0.0_f32; total];
    let chunk_size = 4096.max(row_count / rayon::current_num_threads().max(1));
    quantized
        .par_chunks_mut(chunk_size * feature_count)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let row_start = chunk_idx * chunk_size;
            let rows_in_chunk = out_chunk.len() / feature_count;
            for local_row in 0..rows_in_chunk {
                let row_index = row_start + local_row;
                let base = row_index * feature_count;
                let out_base = local_row * feature_count;
                for fi in 0..feature_count {
                    let value = values[base + fi];
                    // v0.9.0 Limitation 4 fix: preserve NaN through the
                    // f32 cast so the predictor's `is_nan` check fires
                    // and routes through `default_left` even on
                    // rank-binned features (which previously silently
                    // fell through to bin 0).
                    out_chunk[out_base + fi] = if value.is_nan() {
                        f32::NAN
                    } else {
                        let bin = if rank_flags[fi] {
                            quantize_rank_value_wide(
                                value,
                                &feature_sorted_values[fi],
                                max_data_bin,
                            )
                        } else {
                            quantize_linear_value_wide(
                                value,
                                feature_mins[fi],
                                feature_maxs[fi],
                                max_data_bin,
                            )
                        };
                        bin as f32
                    };
                }
            }
        });
    quantized
}

fn derive_dense_feature_bounds(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> (Vec<f32>, Vec<f32>) {
    let results: Vec<(f32, f32)> = (0..feature_count)
        .into_par_iter()
        .map(|feature_index| {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for row_index in 0..row_count {
                let value = values[row_index * feature_count + feature_index];
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
            (min_val, max_val)
        })
        .collect();
    let mut mins = Vec::with_capacity(feature_count);
    let mut maxs = Vec::with_capacity(feature_count);
    for (min_val, max_val) in results {
        mins.push(min_val);
        maxs.push(max_val);
    }
    (mins, maxs)
}

fn derive_dense_sorted_feature_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Vec<Vec<f32>> {
    (0..feature_count)
        .into_par_iter()
        .map(|feature_index| {
            let mut column = Vec::with_capacity(row_count);
            for row_index in 0..row_count {
                let value = values[row_index * feature_count + feature_index];
                if !value.is_nan() {
                    column.push(value);
                }
            }
            column.sort_by(f32::total_cmp);
            column
        })
        .collect()
}

fn quantile_cuts_from_sorted_values(sorted_values: &[f32], max_bins: usize) -> Vec<f32> {
    if sorted_values.len() <= 1 {
        return Vec::new();
    }
    let bin_count = max_bins.min(sorted_values.len());
    let mut cuts = Vec::with_capacity(bin_count.saturating_sub(1));
    for quantile_index in 1..bin_count {
        let rank = ((quantile_index as u128 * sorted_values.len() as u128) / bin_count as u128)
            .min((sorted_values.len() - 1) as u128) as usize;
        let cut_value = sorted_values[rank];
        if cuts.last().copied().is_some_and(|last| cut_value <= last) {
            continue;
        }
        cuts.push(cut_value);
    }
    cuts
}

fn evenly_spaced_row_index(sample_index: usize, sample_count: usize, row_count: usize) -> usize {
    if sample_count <= 1 {
        return row_count / 2;
    }
    ((sample_index as u128 * row_count.saturating_sub(1) as u128)
        / sample_count.saturating_sub(1) as u128) as usize
}

fn derive_dense_feature_quantile_cuts(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    max_bins: usize,
    sketch_max_rows: Option<usize>,
) -> (Vec<Vec<f32>>, Vec<String>) {
    let sampled_row_count = sketch_max_rows.filter(|max_rows| row_count > *max_rows);
    let selected_row_count = sampled_row_count.unwrap_or(row_count);
    let derive_feature_cuts = |feature_index: usize| {
        let mut column = Vec::with_capacity(selected_row_count);
        for selected_index in 0..selected_row_count {
            let row_index = if sampled_row_count.is_some() {
                evenly_spaced_row_index(selected_index, selected_row_count, row_count)
            } else {
                selected_index
            };
            let value = values[row_index * feature_count + feature_index];
            if !value.is_nan() {
                column.push(value);
            }
        }
        column.sort_unstable_by(f32::total_cmp);
        quantile_cuts_from_sorted_values(&column, max_bins)
    };

    const SORT_SCRATCH_BUDGET_BYTES: usize = 20 * 1024 * 1024;
    let bytes_per_column = selected_row_count.saturating_mul(size_of::<f32>()).max(1);
    let columns_per_batch = (SORT_SCRATCH_BUDGET_BYTES / bytes_per_column).max(1);
    let cuts = if columns_per_batch >= rayon::current_num_threads()
        || columns_per_batch >= feature_count
    {
        (0..feature_count)
            .into_par_iter()
            .map(derive_feature_cuts)
            .collect()
    } else {
        let mut cuts = Vec::with_capacity(feature_count);
        for start in (0..feature_count).step_by(columns_per_batch) {
            let end = (start + columns_per_batch).min(feature_count);
            cuts.extend(
                (start..end)
                    .into_par_iter()
                    .map(&derive_feature_cuts)
                    .collect::<Vec<_>>(),
            );
        }
        cuts
    };
    let method = if sampled_row_count.is_some() {
        "sketch"
    } else {
        "exact"
    };
    (cuts, vec![method.to_string(); feature_count])
}

fn derive_linear_tail_rank_plan(
    sorted_values: &[Vec<f32>],
    core_span_ratio_threshold: f32,
) -> Vec<bool> {
    sorted_values
        .par_iter()
        .map(|values| {
            let value_count = values.len();
            if value_count < 5 {
                return false;
            }
            let full_span = values[value_count - 1] - values[0];
            if full_span <= crate::PRE_BINNED_INTEGER_TOLERANCE {
                return false;
            }
            let trim_count = ((value_count as f32 * 0.1).floor() as usize).max(1);
            if trim_count * 2 >= value_count {
                return false;
            }
            let core_low = values[trim_count];
            let core_high = values[value_count - 1 - trim_count];
            let core_span = core_high - core_low;
            let ratio = core_span / full_span;
            ratio.is_finite() && ratio <= core_span_ratio_threshold
        })
        .collect()
}

fn validate_dense_values_allow_nan(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Result<(), EngineError> {
    let dense_view = DenseMatrixView::new(row_count, feature_count, values)?;
    for row_index in 0..dense_view.row_count {
        let row = dense_view.row(row_index)?;
        for (feature_index, &value) in row.iter().enumerate() {
            if value.is_infinite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must not be infinite"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn prepare_validation_matrices_from_dense_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    time_index: Option<Vec<i64>>,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    strategy: ContinuousBinningStrategy,
    training_metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
    max_bins: usize,
) -> Result<PreparedTrainingMatrices, EngineError> {
    if targets.len() != row_count {
        return Err(EngineError::ContractViolation(format!(
            "rows length {} does not match targets length {}",
            row_count,
            targets.len()
        )));
    }
    validate_dense_values_allow_nan(values, row_count, feature_count)?;

    // When training took the pre-binned path (all-integer features, feature_mins = None),
    // validation must also use the pre-binned path.  Non-integer values (e.g. 1.5, 2.5)
    // are rounded to the nearest integer bin; NaN maps to the missing-bin sentinel.
    if !training_metadata.uses_continuous_binning {
        let use_wide = needs_wide_bins(max_bins);
        if use_wide {
            let max_data_bin = (max_bins - 2) as u16;
            let nan_bin = max_data_bin + 1;
            let mut bins_u16 = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin_seen = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let bin = if value.is_nan() {
                    nan_bin
                } else {
                    let rounded = value.round();
                    if rounded > 65535.0 {
                        return Err(EngineError::ContractViolation(format!(
                            "validation value at index {index} exceeds max supported bin 65535"
                        )));
                    }
                    rounded as u16
                };
                max_bin_seen = max_bin_seen.max(bin);
                bins_u16.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            let dataset = TrainingDataset {
                matrix: if need_dense_values {
                    DatasetMatrix::new(row_count, feature_count, dense_values_out)?
                } else {
                    DatasetMatrix::new_metadata_only(row_count, feature_count)?
                },
                targets: targets.to_vec(),
                sample_weights,
                time_index,
                group_id,
                factor_exposures: None,
            };
            let binned_matrix = BinnedMatrix::new_u16_with_layout(
                row_count,
                feature_count,
                if max_bin_seen == 0 { 1 } else { max_bin_seen },
                nan_bin,
                bins_u16,
                BinnedLayout::ColumnMajor,
            )?;
            return Ok(PreparedTrainingMatrices {
                dataset,
                binned_matrix,
                metadata: training_metadata.clone(),
            });
        } else {
            let mut bins = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin_seen = 0_u16;
            for (index, &value) in values.iter().enumerate() {
                let bin = if value.is_nan() {
                    MISSING_BIN_U8
                } else {
                    let rounded = value.round();
                    if rounded > 255.0 {
                        return Err(EngineError::ContractViolation(format!(
                            "validation value at index {index} exceeds max supported bin 255"
                        )));
                    }
                    rounded as u8
                };
                max_bin_seen = max_bin_seen.max(u16::from(bin));
                bins.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            let dataset = TrainingDataset {
                matrix: if need_dense_values {
                    DatasetMatrix::new(row_count, feature_count, dense_values_out)?
                } else {
                    DatasetMatrix::new_metadata_only(row_count, feature_count)?
                },
                targets: targets.to_vec(),
                sample_weights,
                time_index,
                group_id,
                factor_exposures: None,
            };
            let binned_matrix = BinnedMatrix::new_with_layout(
                row_count,
                feature_count,
                if max_bin_seen == 0 { 1 } else { max_bin_seen },
                bins,
                BinnedLayout::ColumnMajor,
            )?;
            return Ok(PreparedTrainingMatrices {
                dataset,
                binned_matrix,
                metadata: training_metadata.clone(),
            });
        }
    }

    if needs_wide_bins(max_bins) {
        let max_data_bin = (max_bins - 2) as u16;
        let nan_bin = max_data_bin + 1;
        let (dense_values, bins_u16, max_bin) = quantize_dense_values_with_metadata_wide(
            values,
            row_count,
            feature_count,
            strategy,
            training_metadata,
            need_dense_values,
            max_data_bin,
        )?;
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
            factor_exposures: None,
        };
        let binned_matrix = BinnedMatrix::new_u16_with_layout(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            nan_bin,
            bins_u16,
            BinnedLayout::ColumnMajor,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata: training_metadata.clone(),
        })
    } else {
        let (dense_values, bins, max_bin) = quantize_dense_values_with_metadata(
            values,
            row_count,
            feature_count,
            strategy,
            training_metadata,
            need_dense_values,
            (max_bins - 2) as u8,
        )?;
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
            factor_exposures: None,
        };
        let binned_matrix = BinnedMatrix::new_with_layout(
            row_count,
            feature_count,
            if max_bin == 0 { 1 } else { max_bin },
            bins,
            BinnedLayout::ColumnMajor,
        )?;
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata: training_metadata.clone(),
        })
    }
}

fn quantize_dense_values_with_metadata(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
    max_data_bin: u8,
) -> Result<(Vec<f32>, Vec<u8>, u16), EngineError> {
    // Validate metadata upfront so parallel closures don't need to return Result.
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();

    let total_cells = row_count * feature_count;
    let mut dense_values = if need_dense_values {
        vec![0.0_f32; total_cells]
    } else {
        Vec::new()
    };
    let mut bins = vec![0_u8; total_cells];

    let chunk_size = (row_count / rayon::current_num_threads().max(1)).max(256);

    let max_bin = if need_dense_values {
        dense_values
            .par_chunks_mut(chunk_size * feature_count)
            .zip(bins.par_chunks_mut(chunk_size * feature_count))
            .enumerate()
            .map(|(chunk_idx, (dense_chunk, bin_chunk))| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = dense_chunk.len() / feature_count;
                let mut local_max_bin = 0_u8;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            MISSING_BIN_U8
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value(value, &sv[feature_index])
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value(value, &sv[feature_index])
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    quantize_quantile_value(
                                        value,
                                        &cuts[feature_index],
                                        u16::from(max_data_bin),
                                    ) as u8
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        dense_chunk[dst_base + feature_index] = bin as f32;
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                u16::from(local_max_bin)
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    } else {
        bins.par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .map(|(chunk_idx, bin_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = bin_chunk.len() / feature_count;
                let mut local_max_bin = 0_u8;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            MISSING_BIN_U8
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value(value, &sv[feature_index])
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value(value, &sv[feature_index])
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    quantize_quantile_value(
                                        value,
                                        &cuts[feature_index],
                                        u16::from(max_data_bin),
                                    ) as u8
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                u16::from(local_max_bin)
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    };

    Ok((dense_values, bins, max_bin))
}

/// u16 variant of `quantize_dense_values_with_metadata` for max_bins > 256.
/// Data bins scale to 0..max_data_bin; NaN gets max_data_bin + 1.
fn quantize_dense_values_with_metadata_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    need_dense_values: bool,
    max_data_bin: u16,
) -> Result<(Vec<f32>, Vec<u16>, u16), EngineError> {
    let nan_bin = max_data_bin + 1;
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();

    let total_cells = row_count * feature_count;
    let mut dense_values = if need_dense_values {
        vec![0.0_f32; total_cells]
    } else {
        Vec::new()
    };
    let mut bins = vec![0_u16; total_cells];

    let chunk_size = (row_count / rayon::current_num_threads().max(1)).max(256);

    let max_bin = if need_dense_values {
        dense_values
            .par_chunks_mut(chunk_size * feature_count)
            .zip(bins.par_chunks_mut(chunk_size * feature_count))
            .enumerate()
            .map(|(chunk_idx, (dense_chunk, bin_chunk))| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = dense_chunk.len() / feature_count;
                let mut local_max_bin = 0_u16;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            nan_bin
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value_wide(
                                            value,
                                            &sv[feature_index],
                                            max_data_bin,
                                        )
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value_wide(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                            max_data_bin,
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value_wide(
                                        value,
                                        &sv[feature_index],
                                        max_data_bin,
                                    )
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    quantize_quantile_value(
                                        value,
                                        &cuts[feature_index],
                                        max_data_bin,
                                    )
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        dense_chunk[dst_base + feature_index] = bin as f32;
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                local_max_bin
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    } else {
        bins.par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .map(|(chunk_idx, bin_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let chunk_rows = bin_chunk.len() / feature_count;
                let mut local_max_bin = 0_u16;
                for local_row in 0..chunk_rows {
                    let row_index = row_start + local_row;
                    let src_base = row_index * feature_count;
                    let dst_base = local_row * feature_count;
                    for feature_index in 0..feature_count {
                        let value = values[src_base + feature_index];
                        let bin = if value.is_nan() {
                            nan_bin
                        } else {
                            match strategy {
                                ContinuousBinningStrategy::Linear => {
                                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                                        let sv = sorted_ref.expect("sorted values validated");
                                        quantize_rank_value_wide(
                                            value,
                                            &sv[feature_index],
                                            max_data_bin,
                                        )
                                    } else {
                                        let mins = mins_ref.expect("mins validated");
                                        let maxs = maxs_ref.expect("maxs validated");
                                        quantize_linear_value_wide(
                                            value,
                                            mins[feature_index],
                                            maxs[feature_index],
                                            max_data_bin,
                                        )
                                    }
                                }
                                ContinuousBinningStrategy::Rank => {
                                    let sv = sorted_ref.expect("sorted values validated");
                                    quantize_rank_value_wide(
                                        value,
                                        &sv[feature_index],
                                        max_data_bin,
                                    )
                                }
                                ContinuousBinningStrategy::Quantile => {
                                    let cuts = cuts_ref.expect("cuts validated");
                                    quantize_quantile_value(
                                        value,
                                        &cuts[feature_index],
                                        max_data_bin,
                                    )
                                }
                            }
                        };
                        local_max_bin = local_max_bin.max(bin);
                        bin_chunk[dst_base + feature_index] = bin;
                    }
                }
                local_max_bin
            })
            .reduce(|| 0_u16, |a, b| a.max(b))
    };

    Ok((dense_values, bins, max_bin))
}

fn quantize_dense_values_to_column_major(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    max_data_bin: u8,
) -> Result<(Vec<f32>, Vec<u8>, u16), EngineError> {
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();
    let quantize = |feature_index: usize, value: f32| {
        if value.is_nan() {
            MISSING_BIN_U8
        } else {
            match strategy {
                ContinuousBinningStrategy::Linear => {
                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                        let sorted = sorted_ref.expect("sorted values validated");
                        quantize_rank_value(value, &sorted[feature_index])
                    } else {
                        let mins = mins_ref.expect("mins validated");
                        let maxs = maxs_ref.expect("maxs validated");
                        quantize_linear_value(value, mins[feature_index], maxs[feature_index])
                    }
                }
                ContinuousBinningStrategy::Rank => {
                    let sorted = sorted_ref.expect("sorted values validated");
                    quantize_rank_value(value, &sorted[feature_index])
                }
                ContinuousBinningStrategy::Quantile => {
                    let cuts = cuts_ref.expect("cuts validated");
                    quantize_quantile_value(value, &cuts[feature_index], u16::from(max_data_bin))
                        as u8
                }
            }
        }
    };

    let mut bins = vec![0_u8; row_count * feature_count];
    let max_bin = if strategy == ContinuousBinningStrategy::Quantile {
        let cuts = cuts_ref.expect("cuts validated");
        bins.par_chunks_mut(row_count.max(1))
            .enumerate()
            .map(|(feature_index, column)| {
                let feature_cuts = &cuts[feature_index];
                let mut local_max = 0_u8;
                for (row_index, destination) in column.iter_mut().enumerate() {
                    let value = values[row_index * feature_count + feature_index];
                    let bin = if value.is_nan() {
                        MISSING_BIN_U8
                    } else {
                        quantize_quantile_value(value, feature_cuts, u16::from(max_data_bin)) as u8
                    };
                    local_max = local_max.max(bin);
                    *destination = bin;
                }
                u16::from(local_max)
            })
            .reduce(|| 0_u16, u16::max)
    } else {
        bins.par_chunks_mut(row_count.max(1))
            .enumerate()
            .map(|(feature_index, column)| {
                let mut local_max = 0_u8;
                for (row_index, destination) in column.iter_mut().enumerate() {
                    let value = values[row_index * feature_count + feature_index];
                    let bin = quantize(feature_index, value);
                    local_max = local_max.max(bin);
                    *destination = bin;
                }
                u16::from(local_max)
            })
            .reduce(|| 0_u16, u16::max)
    };

    Ok((Vec::new(), bins, max_bin))
}

fn quantize_dense_values_to_column_major_wide(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    strategy: ContinuousBinningStrategy,
    metadata: &ContinuousBinningMetadataInternal,
    max_data_bin: u16,
) -> Result<(Vec<f32>, Vec<u16>, u16), EngineError> {
    let nan_bin = max_data_bin + 1;
    let mins_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_mins.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear minima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let maxs_ref = match strategy {
        ContinuousBinningStrategy::Linear => {
            Some(metadata.feature_maxs.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous linear maxima are missing".to_string())
            })?)
        }
        _ => None,
    };
    let sorted_ref = match strategy {
        ContinuousBinningStrategy::Rank => {
            Some(metadata.feature_sorted_values.as_ref().ok_or_else(|| {
                EngineError::ContractViolation(
                    "continuous rank sorted values are missing".to_string(),
                )
            })?)
        }
        ContinuousBinningStrategy::Linear => metadata.feature_sorted_values.as_ref(),
        _ => None,
    };
    let cuts_ref = match strategy {
        ContinuousBinningStrategy::Quantile => {
            Some(metadata.feature_quantile_cuts.as_ref().ok_or_else(|| {
                EngineError::ContractViolation("continuous quantile cuts are missing".to_string())
            })?)
        }
        _ => None,
    };
    let rank_flags = metadata.feature_linear_rank_flags.as_ref();
    let quantize = |feature_index: usize, value: f32| {
        if value.is_nan() {
            nan_bin
        } else {
            match strategy {
                ContinuousBinningStrategy::Linear => {
                    if rank_flags.is_some_and(|flags| flags[feature_index]) {
                        let sorted = sorted_ref.expect("sorted values validated");
                        quantize_rank_value_wide(value, &sorted[feature_index], max_data_bin)
                    } else {
                        let mins = mins_ref.expect("mins validated");
                        let maxs = maxs_ref.expect("maxs validated");
                        quantize_linear_value_wide(
                            value,
                            mins[feature_index],
                            maxs[feature_index],
                            max_data_bin,
                        )
                    }
                }
                ContinuousBinningStrategy::Rank => {
                    let sorted = sorted_ref.expect("sorted values validated");
                    quantize_rank_value_wide(value, &sorted[feature_index], max_data_bin)
                }
                ContinuousBinningStrategy::Quantile => {
                    let cuts = cuts_ref.expect("cuts validated");
                    quantize_quantile_value(value, &cuts[feature_index], max_data_bin)
                }
            }
        }
    };

    let mut bins = vec![0_u16; row_count * feature_count];
    let max_bin = if strategy == ContinuousBinningStrategy::Quantile {
        let cuts = cuts_ref.expect("cuts validated");
        bins.par_chunks_mut(row_count.max(1))
            .enumerate()
            .map(|(feature_index, column)| {
                let feature_cuts = &cuts[feature_index];
                let mut local_max = 0_u16;
                for (row_index, destination) in column.iter_mut().enumerate() {
                    let value = values[row_index * feature_count + feature_index];
                    let bin = if value.is_nan() {
                        nan_bin
                    } else {
                        quantize_quantile_value(value, feature_cuts, max_data_bin)
                    };
                    local_max = local_max.max(bin);
                    *destination = bin;
                }
                local_max
            })
            .reduce(|| 0_u16, u16::max)
    } else {
        bins.par_chunks_mut(row_count.max(1))
            .enumerate()
            .map(|(feature_index, column)| {
                let mut local_max = 0_u16;
                for (row_index, destination) in column.iter_mut().enumerate() {
                    let value = values[row_index * feature_count + feature_index];
                    let bin = quantize(feature_index, value);
                    local_max = local_max.max(bin);
                    *destination = bin;
                }
                local_max
            })
            .reduce(|| 0_u16, u16::max)
    };

    Ok((Vec::new(), bins, max_bin))
}

pub(crate) fn prepare_training_matrices_from_dense_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
    targets: &[f32],
    time_index: Option<Vec<i64>>,
    sample_weights: Option<Vec<f32>>,
    group_id: Option<Vec<u32>>,
    strategy: ContinuousBinningStrategy,
    max_bins: usize,
    quantile_sketch_max_rows: Option<usize>,
    need_dense_values: bool,
    binned_layout: BinnedLayout,
) -> Result<PreparedTrainingMatrices, EngineError> {
    validate_continuous_binning_max_bins(max_bins)?;
    if quantile_sketch_max_rows == Some(0) {
        return Err(EngineError::ContractViolation(
            "quantile_sketch_max_rows must be greater than 0 when set".to_string(),
        ));
    }
    let dense_view = DenseMatrixView::new(row_count, feature_count, values)?;
    if targets.len() != dense_view.row_count {
        return Err(EngineError::ContractViolation(format!(
            "rows length {} does not match targets length {}",
            dense_view.row_count,
            targets.len()
        )));
    }

    let mut use_pre_binned_path = true;
    for row_index in 0..dense_view.row_count {
        let row = dense_view.row(row_index)?;
        for (feature_index, &value) in row.iter().enumerate() {
            if value.is_infinite() {
                return Err(EngineError::ContractViolation(format!(
                    "row {row_index} feature {feature_index} must not be infinite"
                )));
            }
            if use_pre_binned_path && (value.is_nan() || !is_pre_binned_integer_value(value)) {
                use_pre_binned_path = false;
            }
        }
    }

    // Build binning metadata (shared by u8 and u16 paths).
    let wide_bins = needs_wide_bins(max_bins);
    let (metadata, use_wide) = if use_pre_binned_path {
        (ContinuousBinningMetadataInternal::pre_binned(), wide_bins)
    } else {
        let meta = match strategy {
            ContinuousBinningStrategy::Linear => {
                let (feature_mins, feature_maxs) =
                    derive_dense_feature_bounds(values, row_count, feature_count);
                if linear_tail_rank_enabled_from_env() {
                    let sorted_values =
                        derive_dense_sorted_feature_values(values, row_count, feature_count);
                    let rank_flags = derive_linear_tail_rank_plan(
                        &sorted_values,
                        linear_tail_core_span_ratio_threshold_from_env(),
                    );
                    ContinuousBinningMetadataInternal {
                        uses_continuous_binning: true,
                        feature_mins: Some(feature_mins),
                        feature_maxs: Some(feature_maxs),
                        feature_sorted_values: if rank_flags.iter().any(|flag| *flag) {
                            Some(sorted_values)
                        } else {
                            None
                        },
                        feature_quantile_cuts: None,
                        feature_quantile_cut_methods: None,
                        feature_linear_rank_flags: Some(rank_flags),
                    }
                } else {
                    ContinuousBinningMetadataInternal {
                        uses_continuous_binning: true,
                        feature_mins: Some(feature_mins),
                        feature_maxs: Some(feature_maxs),
                        feature_sorted_values: None,
                        feature_quantile_cuts: None,
                        feature_quantile_cut_methods: None,
                        feature_linear_rank_flags: None,
                    }
                }
            }
            ContinuousBinningStrategy::Rank => ContinuousBinningMetadataInternal {
                uses_continuous_binning: true,
                feature_mins: None,
                feature_maxs: None,
                feature_sorted_values: Some(derive_dense_sorted_feature_values(
                    values,
                    row_count,
                    feature_count,
                )),
                feature_quantile_cuts: None,
                feature_quantile_cut_methods: None,
                feature_linear_rank_flags: None,
            },
            ContinuousBinningStrategy::Quantile => {
                let (cuts, methods) = derive_dense_feature_quantile_cuts(
                    values,
                    row_count,
                    feature_count,
                    max_bins,
                    quantile_sketch_max_rows,
                );
                ContinuousBinningMetadataInternal {
                    uses_continuous_binning: true,
                    feature_mins: None,
                    feature_maxs: None,
                    feature_sorted_values: None,
                    feature_quantile_cuts: Some(cuts),
                    feature_quantile_cut_methods: Some(methods),
                    feature_linear_rank_flags: None,
                }
            }
        };
        (meta, wide_bins)
    };

    // Encode bins and build BinnedMatrix — u8 fast path or u16 wide path.
    let direct_column_major = binned_layout == BinnedLayout::ColumnMajor && !need_dense_values;
    if use_wide {
        let max_data_bin = (max_bins - 2) as u16;
        let nan_bin = max_data_bin + 1;
        let (dense_values, bins_u16, max_bin) = if use_pre_binned_path {
            // Pre-binned u16 path: Python already quantized values as f32 integers.
            let mut bins_u16 = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin = 0_u16;
            let source_indices: Box<dyn Iterator<Item = usize>> = if direct_column_major {
                Box::new((0..feature_count).flat_map(|feature| {
                    (0..row_count).map(move |row| row * feature_count + feature)
                }))
            } else {
                Box::new(0..values.len())
            };
            for index in source_indices {
                let value = values[index];
                let rounded = value.round();
                if rounded > 65535.0 {
                    return Err(EngineError::ContractViolation(format!(
                        "value at index {index} exceeds max supported bin 65535"
                    )));
                }
                let bin = rounded as u16;
                max_bin = max_bin.max(bin);
                bins_u16.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            (dense_values_out, bins_u16, max_bin)
        } else if direct_column_major {
            quantize_dense_values_to_column_major_wide(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                max_data_bin,
            )?
        } else {
            quantize_dense_values_with_metadata_wide(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                need_dense_values,
                max_data_bin,
            )?
        };
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
            factor_exposures: None,
        };
        let max_bin = if max_bin == 0 { 1 } else { max_bin };
        let binned_matrix = if direct_column_major {
            BinnedMatrix::new_u16_from_column_major(
                row_count,
                feature_count,
                max_bin,
                nan_bin,
                bins_u16,
            )?
        } else {
            BinnedMatrix::new_u16_with_layout(
                row_count,
                feature_count,
                max_bin,
                nan_bin,
                bins_u16,
                binned_layout,
            )?
        };
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata,
        })
    } else {
        // u8 path: pre-binned or continuous with max_bins <= 256.
        let (dense_values, bins, max_bin) = if use_pre_binned_path {
            let mut bins = Vec::with_capacity(values.len());
            let mut dense_values_out = if need_dense_values {
                Vec::with_capacity(values.len())
            } else {
                Vec::new()
            };
            let mut max_bin = 0_u16;
            let source_indices: Box<dyn Iterator<Item = usize>> = if direct_column_major {
                Box::new((0..feature_count).flat_map(|feature| {
                    (0..row_count).map(move |row| row * feature_count + feature)
                }))
            } else {
                Box::new(0..values.len())
            };
            for index in source_indices {
                let value = values[index];
                let rounded = value.round();
                if rounded > 255.0 {
                    return Err(EngineError::ContractViolation(format!(
                        "value at index {index} exceeds max supported bin 255"
                    )));
                }
                let bin = rounded as u8;
                max_bin = max_bin.max(u16::from(bin));
                bins.push(bin);
                if need_dense_values {
                    dense_values_out.push(bin as f32);
                }
            }
            (dense_values_out, bins, max_bin)
        } else if direct_column_major {
            quantize_dense_values_to_column_major(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                (max_bins - 2) as u8,
            )?
        } else {
            quantize_dense_values_with_metadata(
                values,
                row_count,
                feature_count,
                strategy,
                &metadata,
                need_dense_values,
                (max_bins - 2) as u8,
            )?
        };
        let dataset = TrainingDataset {
            matrix: if need_dense_values {
                DatasetMatrix::new(row_count, feature_count, dense_values)?
            } else {
                DatasetMatrix::new_metadata_only(row_count, feature_count)?
            },
            targets: targets.to_vec(),
            sample_weights,
            time_index,
            group_id,
            factor_exposures: None,
        };
        let max_bin = if max_bin == 0 { 1 } else { max_bin };
        let binned_matrix = if direct_column_major {
            BinnedMatrix::new_from_column_major(row_count, feature_count, max_bin, bins)?
        } else {
            BinnedMatrix::new_with_layout(row_count, feature_count, max_bin, bins, binned_layout)?
        };
        Ok(PreparedTrainingMatrices {
            dataset,
            binned_matrix,
            metadata,
        })
    }
}
