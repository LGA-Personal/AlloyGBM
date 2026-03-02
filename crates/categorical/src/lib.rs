use alloygbm_core::CoreError;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct TargetEncoderConfig {
    pub smoothing: f64,
    pub min_samples_leaf: u32,
    pub time_aware: bool,
}

impl Default for TargetEncoderConfig {
    fn default() -> Self {
        Self {
            smoothing: 20.0,
            min_samples_leaf: 1,
            time_aware: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryTargetStats {
    pub category: String,
    pub count: u32,
    pub mean: f32,
    pub encoded: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetEncoderState {
    pub config: TargetEncoderConfig,
    pub global_mean: f32,
    pub category_stats: Vec<CategoryTargetStats>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryFrequency {
    pub category: String,
    pub count: u32,
    pub frequency: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyEncoderState {
    pub total_count: u32,
    pub category_frequencies: Vec<CategoryFrequency>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoricalError {
    InvalidInput(String),
    Core(CoreError),
}

impl Display for CategoricalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for CategoricalError {}

impl From<CoreError> for CategoricalError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

pub type CategoricalResult<T> = Result<T, CategoricalError>;
type CategoryTargetAggregates = BTreeMap<String, (f64, u32)>;

pub fn fit_target_encoder(
    config: &TargetEncoderConfig,
    values: &[String],
    targets: &[f32],
    time_index: Option<&[i64]>,
) -> CategoricalResult<TargetEncoderState> {
    validate_target_encoder_config(config)?;
    validate_values_and_targets(values, targets)?;
    validate_time_index(config, time_index, values.len())?;

    let (global_mean, category_sums) = aggregate_target_statistics(values, targets)?;
    let category_stats = category_sums
        .into_iter()
        .map(|(category, (sum, count))| {
            let mean = (sum / count as f64) as f32;
            CategoryTargetStats {
                category,
                count,
                mean,
                encoded: smoothed_target_encoding(global_mean, sum, count, config),
            }
        })
        .collect::<Vec<_>>();

    Ok(TargetEncoderState {
        config: config.clone(),
        global_mean: global_mean as f32,
        category_stats,
    })
}

pub fn transform_target_encoder(
    state: &TargetEncoderState,
    values: &[String],
) -> CategoricalResult<Vec<f32>> {
    validate_values(values)?;

    let lookup = state
        .category_stats
        .iter()
        .map(|stats| (stats.category.as_str(), stats.encoded))
        .collect::<BTreeMap<_, _>>();

    Ok(values
        .iter()
        .map(|value| {
            lookup
                .get(value.as_str())
                .copied()
                .unwrap_or(state.global_mean)
        })
        .collect())
}

pub fn fit_transform_target_encoder(
    config: &TargetEncoderConfig,
    values: &[String],
    targets: &[f32],
    time_index: Option<&[i64]>,
) -> CategoricalResult<(TargetEncoderState, Vec<f32>)> {
    let state = fit_target_encoder(config, values, targets, time_index)?;
    if !config.time_aware {
        let encoded = transform_target_encoder(&state, values)?;
        return Ok((state, encoded));
    }

    let time_index = time_index.ok_or_else(|| {
        CategoricalError::InvalidInput("time_aware target encoding requires time_index".to_string())
    })?;
    let encoded = fit_transform_target_encoder_time_aware(config, values, targets, time_index)?;
    Ok((state, encoded))
}

pub fn fit_frequency_encoder(values: &[String]) -> CategoricalResult<FrequencyEncoderState> {
    validate_values(values)?;
    let total_count = u32::try_from(values.len()).map_err(|_| {
        CategoricalError::InvalidInput("values length exceeds u32::MAX".to_string())
    })?;

    let mut counts = BTreeMap::<String, u32>::new();
    for value in values {
        *counts.entry(value.clone()).or_insert(0) += 1;
    }

    let category_frequencies = counts
        .into_iter()
        .map(|(category, count)| CategoryFrequency {
            category,
            count,
            frequency: count as f32 / total_count as f32,
        })
        .collect::<Vec<_>>();

    Ok(FrequencyEncoderState {
        total_count,
        category_frequencies,
    })
}

pub fn transform_frequency_encoder(
    state: &FrequencyEncoderState,
    values: &[String],
) -> CategoricalResult<Vec<f32>> {
    validate_values(values)?;

    let lookup = state
        .category_frequencies
        .iter()
        .map(|frequency| (frequency.category.as_str(), frequency.frequency))
        .collect::<BTreeMap<_, _>>();

    Ok(values
        .iter()
        .map(|value| lookup.get(value.as_str()).copied().unwrap_or(0.0))
        .collect())
}

pub fn fit_transform_frequency_encoder(
    values: &[String],
) -> CategoricalResult<(FrequencyEncoderState, Vec<f32>)> {
    let state = fit_frequency_encoder(values)?;
    let transformed = transform_frequency_encoder(&state, values)?;
    Ok((state, transformed))
}

fn fit_transform_target_encoder_time_aware(
    config: &TargetEncoderConfig,
    values: &[String],
    targets: &[f32],
    time_index: &[i64],
) -> CategoricalResult<Vec<f32>> {
    let mut row_indices = (0..values.len()).collect::<Vec<_>>();
    row_indices.sort_by_key(|&index| (time_index[index], index));

    let mut encoded = vec![0.0_f32; values.len()];
    let mut category_sums = BTreeMap::<String, (f64, u32)>::new();
    let mut global_sum = 0.0_f64;
    let mut global_count = 0_u32;
    let mut cursor = 0_usize;

    while cursor < row_indices.len() {
        let group_time = time_index[row_indices[cursor]];
        let mut group_end = cursor + 1;
        while group_end < row_indices.len() && time_index[row_indices[group_end]] == group_time {
            group_end += 1;
        }

        for &row_index in &row_indices[cursor..group_end] {
            let prior_global_mean = if global_count == 0 {
                0.0_f64
            } else {
                global_sum / global_count as f64
            };

            let (category_sum, category_count) = category_sums
                .get(values[row_index].as_str())
                .copied()
                .unwrap_or((0.0, 0));
            encoded[row_index] =
                smoothed_target_encoding(prior_global_mean, category_sum, category_count, config);
        }

        for &row_index in &row_indices[cursor..group_end] {
            let target = targets[row_index] as f64;
            let entry = category_sums
                .entry(values[row_index].clone())
                .or_insert((0.0, 0));
            entry.0 += target;
            entry.1 += 1;
            global_sum += target;
            global_count += 1;
        }

        cursor = group_end;
    }

    Ok(encoded)
}

fn validate_target_encoder_config(config: &TargetEncoderConfig) -> CategoricalResult<()> {
    if !config.smoothing.is_finite() || config.smoothing < 0.0 {
        return Err(CategoricalError::InvalidInput(
            "smoothing must be finite and >= 0".to_string(),
        ));
    }
    if config.min_samples_leaf == 0 {
        return Err(CategoricalError::InvalidInput(
            "min_samples_leaf must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

fn validate_values(values: &[String]) -> CategoricalResult<()> {
    if values.is_empty() {
        return Err(CategoricalError::InvalidInput(
            "values cannot be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_values_and_targets(values: &[String], targets: &[f32]) -> CategoricalResult<()> {
    validate_values(values)?;
    if values.len() != targets.len() {
        return Err(CategoricalError::InvalidInput(
            "values and targets must have matching lengths".to_string(),
        ));
    }
    for (index, target) in targets.iter().enumerate() {
        if !target.is_finite() {
            return Err(CategoricalError::InvalidInput(format!(
                "target at index {index} must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_time_index(
    config: &TargetEncoderConfig,
    time_index: Option<&[i64]>,
    row_count: usize,
) -> CategoricalResult<()> {
    if !config.time_aware {
        return Ok(());
    }
    let time_index = time_index.ok_or_else(|| {
        CategoricalError::InvalidInput("time_aware target encoding requires time_index".to_string())
    })?;
    if time_index.len() != row_count {
        return Err(CategoricalError::InvalidInput(format!(
            "time_index length {} does not match row count {}",
            time_index.len(),
            row_count
        )));
    }
    Ok(())
}

fn aggregate_target_statistics(
    values: &[String],
    targets: &[f32],
) -> CategoricalResult<(f64, CategoryTargetAggregates)> {
    let mut global_sum = 0.0_f64;
    let mut category_sums = CategoryTargetAggregates::new();
    for (value, &target) in values.iter().zip(targets) {
        let target = target as f64;
        global_sum += target;
        let entry = category_sums.entry(value.clone()).or_insert((0.0, 0));
        entry.0 += target;
        entry.1 += 1;
    }
    let global_mean = global_sum / targets.len() as f64;
    Ok((global_mean, category_sums))
}

fn smoothed_target_encoding(
    global_mean: f64,
    category_sum: f64,
    category_count: u32,
    config: &TargetEncoderConfig,
) -> f32 {
    if category_count < config.min_samples_leaf {
        return global_mean as f32;
    }
    let category_mean = category_sum / category_count as f64;
    if config.smoothing == 0.0 {
        return category_mean as f32;
    }
    let weight = category_count as f64 / (category_count as f64 + config.smoothing);
    (weight * category_mean + (1.0 - weight) * global_mean) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f32, right: f32) -> bool {
        (left - right).abs() <= 1e-6
    }

    #[test]
    fn fit_target_encoder_rejects_mismatched_lengths() {
        let cfg = TargetEncoderConfig::default();
        let values = vec!["A".to_string(), "B".to_string()];
        let targets = vec![1.0];
        let result = fit_target_encoder(&cfg, &values, &targets, Some(&[1, 2]));
        assert!(matches!(result, Err(CategoricalError::InvalidInput(_))));
    }

    #[test]
    fn fit_target_encoder_rejects_non_finite_targets() {
        let cfg = TargetEncoderConfig::default();
        let values = vec!["A".to_string()];
        let targets = vec![f32::NAN];
        let result = fit_target_encoder(&cfg, &values, &targets, Some(&[1]));
        assert!(matches!(result, Err(CategoricalError::InvalidInput(_))));
    }

    #[test]
    fn fit_target_encoder_requires_time_index_when_time_aware() {
        let cfg = TargetEncoderConfig::default();
        let values = vec!["A".to_string()];
        let targets = vec![1.0];
        let result = fit_target_encoder(&cfg, &values, &targets, None);
        assert!(matches!(result, Err(CategoricalError::InvalidInput(_))));
    }

    #[test]
    fn fit_target_encoder_builds_deterministic_sorted_state() {
        let cfg = TargetEncoderConfig {
            smoothing: 0.0,
            min_samples_leaf: 1,
            time_aware: false,
        };
        let values = vec![
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
        ];
        let targets = vec![1.0, 5.0, 3.0, 7.0];
        let state = fit_target_encoder(&cfg, &values, &targets, None).expect("state is valid");
        assert_eq!(state.category_stats.len(), 2);
        assert_eq!(state.category_stats[0].category, "a");
        assert_eq!(state.category_stats[1].category, "b");
        assert!(approx_eq(state.global_mean, 4.0));
        assert!(approx_eq(state.category_stats[0].encoded, 6.0));
        assert!(approx_eq(state.category_stats[1].encoded, 2.0));
    }

    #[test]
    fn fit_transform_target_encoder_time_aware_prevents_same_timestamp_leakage() {
        let cfg = TargetEncoderConfig {
            smoothing: 0.0,
            min_samples_leaf: 1,
            time_aware: true,
        };
        let values = vec!["A".to_string(), "A".to_string(), "A".to_string()];
        let targets = vec![0.0, 10.0, 10.0];
        let times = vec![1, 2, 2];
        let (_, encoded) =
            fit_transform_target_encoder(&cfg, &values, &targets, Some(&times)).expect("encodes");
        assert!(approx_eq(encoded[0], 0.0));
        assert!(approx_eq(encoded[1], 0.0));
        assert!(approx_eq(encoded[2], 0.0));
    }

    #[test]
    fn fit_transform_target_encoder_non_time_aware_maps_full_state() {
        let cfg = TargetEncoderConfig {
            smoothing: 0.0,
            min_samples_leaf: 1,
            time_aware: false,
        };
        let values = vec!["A".to_string(), "A".to_string(), "B".to_string()];
        let targets = vec![1.0, 3.0, 5.0];
        let (_, encoded) =
            fit_transform_target_encoder(&cfg, &values, &targets, None).expect("encodes");
        assert!(approx_eq(encoded[0], 2.0));
        assert!(approx_eq(encoded[1], 2.0));
        assert!(approx_eq(encoded[2], 5.0));
    }

    #[test]
    fn transform_target_encoder_uses_global_mean_for_unknown_categories() {
        let cfg = TargetEncoderConfig {
            smoothing: 0.0,
            min_samples_leaf: 1,
            time_aware: false,
        };
        let values = vec!["A".to_string(), "B".to_string()];
        let targets = vec![2.0, 4.0];
        let state = fit_target_encoder(&cfg, &values, &targets, None).expect("state is valid");
        let transformed = transform_target_encoder(&state, &["C".to_string()]).expect("transforms");
        assert!(approx_eq(transformed[0], 3.0));
    }

    #[test]
    fn fit_frequency_encoder_returns_sorted_frequencies() {
        let values = vec![
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        let state = fit_frequency_encoder(&values).expect("state is valid");
        assert_eq!(state.total_count, 4);
        assert_eq!(state.category_frequencies.len(), 3);
        assert_eq!(state.category_frequencies[0].category, "a");
        assert_eq!(state.category_frequencies[1].category, "b");
        assert_eq!(state.category_frequencies[2].category, "c");
        assert!(approx_eq(state.category_frequencies[1].frequency, 0.5));
    }

    #[test]
    fn transform_frequency_encoder_unknown_category_returns_zero() {
        let values = vec!["A".to_string(), "A".to_string(), "B".to_string()];
        let state = fit_frequency_encoder(&values).expect("state is valid");
        let transformed =
            transform_frequency_encoder(&state, &["C".to_string()]).expect("transforms");
        assert!(approx_eq(transformed[0], 0.0));
    }

    #[test]
    fn fit_transform_frequency_encoder_roundtrip_shape_matches() {
        let values = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
        ];
        let (_, transformed) = fit_transform_frequency_encoder(&values).expect("encodes");
        assert_eq!(transformed.len(), values.len());
        assert!(approx_eq(transformed[0], 0.75));
        assert!(approx_eq(transformed[2], 0.25));
    }
}
