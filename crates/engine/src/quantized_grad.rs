//! Quantized gradients for histogram building (Shi et al., NeurIPS 2022;
//! LightGBM's `use_quantized_grad`).
//!
//! Each round's gradients and Hessians are stochastically rounded to small
//! integers. Histograms then sum integers instead of floats, and a whole
//! `(grad, hess, count)` triple fits in one `i64` add per row. Split gains read
//! the integer sums scaled back to floats, so split selection is unchanged.
//! Leaf values never see the rounding: they come from the float gradients the
//! row partition sums, which is the paper's leaf refit.

use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, LinearFeatureScaler,
    LinearHistogramBundle, LinearLeaf, NodeSlice, NodeStats, PartitionResult, SplitCandidate,
    TrainParams,
};
use rayon::prelude::*;

use crate::error::{EngineError, EngineResult};
use crate::sampling::mixed_hash;
use crate::split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, MorphContext, PreparedLinearSplit,
    SplitNoise, SplitSelectionOptions, SplitShortlist,
};
use crate::traits::{BackendOps, HistogramExecution};

/// Smallest and largest supported `quantized_grad_bits`.
pub const MIN_QUANTIZED_GRAD_BITS: u8 = 2;
pub const MAX_QUANTIZED_GRAD_BITS: u8 = 8;

/// Rows per parallel quantization chunk.
const QUANTIZE_CHUNK_ROWS: usize = 16_384;

/// Uniform draws shared by every quantization, as in LightGBM: reading a
/// fixed table is several times cheaper than drawing per row, and it keeps
/// the rounding loop free of a serial RNG dependency.
const UNIFORM_TABLE_LEN: usize = 1 << 16;
const UNIFORM_TABLE_MASK: usize = UNIFORM_TABLE_LEN - 1;

/// The uniform table and the same table reversed.
fn uniform_tables() -> (&'static [f32], &'static [f32]) {
    static TABLES: std::sync::OnceLock<(Vec<f32>, Vec<f32>)> = std::sync::OnceLock::new();
    let (forward, reversed) = TABLES.get_or_init(|| {
        let forward = (0..UNIFORM_TABLE_LEN as u64)
            .map(|index| (mixed_hash(index) >> 40) as f32 / (1u64 << 24) as f32)
            .collect::<Vec<_>>();
        let reversed = forward.iter().rev().copied().collect();
        (forward, reversed)
    });
    (forward, reversed)
}

/// `floor(value)` for `value` in about `[-2^14, 2^14]`, by truncating a
/// shifted positive value (a plain `floor` is a libm call on baseline
/// x86-64). Out-of-range values saturate and are clamped by the caller.
#[inline(always)]
fn stochastic_floor(value: f32) -> i32 {
    const OFFSET: f32 = 16_384.0;
    (value + OFFSET) as i32 - OFFSET as i32
}

/// One round's gradients as integers: `grad ≈ grad_level * grad_scale`,
/// `hess ≈ hess_level * hess_scale`, with stochastic rounding so each level
/// is an unbiased estimate of the float value.
#[derive(Debug, Clone)]
pub struct QuantizedGradients {
    /// Per row, `(grad_level << 16) | hess_level`.
    packed: Vec<i32>,
    grad_scale: f32,
    hess_scale: f32,
    bits: u8,
    /// Every row has the same (top) Hessian level, as for unweighted
    /// squared error; histogram kernels can then derive Hessian sums from
    /// row counts.
    constant_hess: bool,
}

impl QuantizedGradients {
    /// Quantize `gradients` to `bits`-bit levels: gradients to signed levels
    /// in `±(2^(bits-1) - 1)`, Hessians to `0..=2^bits - 2`. `seed` fixes the
    /// rounding draws, so a fit stays reproducible.
    pub fn quantize(gradients: &[GradientPair], bits: u8, seed: u64) -> EngineResult<Self> {
        let mut quantized = Self {
            packed: Vec::new(),
            grad_scale: 1.0,
            hess_scale: 1.0,
            bits,
            constant_hess: false,
        };
        quantized.requantize(gradients, bits, seed)?;
        Ok(quantized)
    }

    /// [`Self::quantize`] into this value, reusing its buffer (a fresh
    /// row-sized allocation per tree costs page faults on large data).
    pub fn requantize(
        &mut self,
        gradients: &[GradientPair],
        bits: u8,
        seed: u64,
    ) -> EngineResult<()> {
        if !(MIN_QUANTIZED_GRAD_BITS..=MAX_QUANTIZED_GRAD_BITS).contains(&bits) {
            return Err(EngineError::InvalidConfig(format!(
                "quantized_grad_bits must be in {MIN_QUANTIZED_GRAD_BITS}..={MAX_QUANTIZED_GRAD_BITS}, got {bits}"
            )));
        }
        let (max_abs_grad, min_hess, max_hess) = gradients
            .par_chunks(QUANTIZE_CHUNK_ROWS)
            .map(|chunk| {
                chunk
                    .iter()
                    .fold((0.0_f32, f32::INFINITY, 0.0_f32), |(g, lo, hi), pair| {
                        (g.max(pair.grad.abs()), lo.min(pair.hess), hi.max(pair.hess))
                    })
            })
            .reduce(
                || (0.0, f32::INFINITY, 0.0),
                |a, b| (a.0.max(b.0), a.1.min(b.1), a.2.max(b.2)),
            );
        let grad_levels = Self::max_grad_level_for(bits);
        let hess_levels = Self::max_hess_level_for(bits);
        let grad_scale = if max_abs_grad > 0.0 {
            max_abs_grad / grad_levels as f32
        } else {
            1.0
        };
        let hess_scale = if max_hess > 0.0 {
            max_hess / hess_levels as f32
        } else {
            1.0
        };
        let inv_grad = 1.0 / grad_scale;
        let inv_hess = 1.0 / hess_scale;
        let constant_hess = min_hess == max_hess;
        let (uniforms, uniforms_reversed) = uniform_tables();
        // Each round reads the shared table from its own random offset, so
        // the rounding draws change from round to round.
        let offset = mixed_hash(seed) as usize;
        let packed = &mut self.packed;
        packed.resize(gradients.len(), 0);
        packed
            .par_chunks_mut(QUANTIZE_CHUNK_ROWS)
            .zip(gradients.par_chunks(QUANTIZE_CHUNK_ROWS))
            .enumerate()
            .for_each(|(chunk_index, (out, chunk))| {
                let mut draw = offset.wrapping_add(chunk_index * QUANTIZE_CHUNK_ROWS);
                let mut done = 0;
                // Runs that read the table contiguously, so the loop below
                // vectorizes.
                while done < chunk.len() {
                    let start = draw & UNIFORM_TABLE_MASK;
                    let run = (chunk.len() - done).min(UNIFORM_TABLE_LEN - start);
                    let pairs = &chunk[done..done + run];
                    let out = &mut out[done..done + run];
                    let u_grad = &uniforms[start..start + run];
                    if constant_hess {
                        for ((slot, pair), &u) in out.iter_mut().zip(pairs).zip(u_grad) {
                            let grad_level = stochastic_floor(pair.grad * inv_grad + u)
                                .clamp(-grad_levels, grad_levels);
                            // A constant Hessian maps to the top level
                            // exactly: no rounding noise in Hessians.
                            *slot = (grad_level << 16) | hess_levels;
                        }
                    } else {
                        // Hessian draws read the table in reverse, so they
                        // are not the gradient draws.
                        let u_hess = &uniforms_reversed[start..start + run];
                        for (((slot, pair), &u), &v) in
                            out.iter_mut().zip(pairs).zip(u_grad).zip(u_hess)
                        {
                            let grad_level = stochastic_floor(pair.grad * inv_grad + u)
                                .clamp(-grad_levels, grad_levels);
                            let hess_level = stochastic_floor(pair.hess.max(0.0) * inv_hess + v)
                                .clamp(0, hess_levels);
                            *slot = (grad_level << 16) | hess_level;
                        }
                    }
                    done += run;
                    draw = draw.wrapping_add(run);
                }
            });
        self.grad_scale = grad_scale;
        self.hess_scale = hess_scale;
        self.bits = bits;
        self.constant_hess = constant_hess;
        Ok(())
    }

    pub fn max_grad_level_for(bits: u8) -> i32 {
        (1_i32 << (bits - 1)) - 1
    }

    pub fn max_hess_level_for(bits: u8) -> i32 {
        (1_i32 << bits) - 2
    }

    pub fn len(&self) -> usize {
        self.packed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    pub fn packed(&self) -> &[i32] {
        &self.packed
    }

    pub fn grad_scale(&self) -> f32 {
        self.grad_scale
    }

    pub fn hess_scale(&self) -> f32 {
        self.hess_scale
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Whether every row carries the same Hessian level.
    pub fn constant_hess(&self) -> bool {
        self.constant_hess
    }

    /// The signed gradient level of a packed value.
    #[inline(always)]
    pub fn grad_level(packed: i32) -> i32 {
        packed >> 16
    }

    /// The Hessian level of a packed value.
    #[inline(always)]
    pub fn hess_level(packed: i32) -> i32 {
        packed & 0xFFFF
    }

    /// The float pair a packed value stands for.
    #[inline]
    pub fn dequantize(&self, packed: i32) -> GradientPair {
        GradientPair {
            grad: Self::grad_level(packed) as f32 * self.grad_scale,
            hess: Self::hess_level(packed) as f32 * self.hess_scale,
        }
    }

    /// Every row's float pair, in row order.
    pub fn dequantize_all(&self) -> Vec<GradientPair> {
        self.packed
            .iter()
            .map(|&packed| self.dequantize(packed))
            .collect()
    }
}

/// Bit layout that sums whole `(grad, hess, count)` triples of up to
/// `max_rows` rows in one `i64`: count in the low bits, the Hessian level sum
/// above it, the signed gradient level sum on top. All three sums are exact
/// as long as no field overflows, which the widths below guarantee. With a
/// constant Hessian level the Hessian field is left out and its sum is the
/// count times that level, which leaves room for much larger nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedPacking {
    pub hess_shift: u32,
    pub grad_shift: u32,
    /// The shared Hessian level when the Hessian field is left out, else 0.
    pub constant_hess_level: i64,
}

impl QuantizedPacking {
    /// The packing for a node of `rows` rows at `bits` bits, or `None` when
    /// the sums do not fit in 64 bits.
    pub fn for_rows(rows: usize, bits: u8, constant_hess: bool) -> Option<Self> {
        let rows = rows as u64;
        let bit_len = |value: u64| u64::BITS - value.leading_zeros();
        let count_bits = bit_len(rows);
        let hess_bits = if constant_hess {
            0
        } else {
            bit_len(rows * QuantizedGradients::max_hess_level_for(bits) as u64)
        };
        // One extra bit for the sign.
        let grad_bits = bit_len(rows * QuantizedGradients::max_grad_level_for(bits) as u64) + 1;
        (count_bits + hess_bits + grad_bits <= i64::BITS).then_some(Self {
            hess_shift: count_bits,
            grad_shift: count_bits + hess_bits,
            constant_hess_level: if constant_hess {
                i64::from(QuantizedGradients::max_hess_level_for(bits))
            } else {
                0
            },
        })
    }

    /// One row's packed `(grad, hess, 1)` triple.
    #[inline(always)]
    pub fn expand(self, packed: i32) -> i64 {
        let hess = if self.constant_hess_level == 0 {
            i64::from(QuantizedGradients::hess_level(packed)) << self.hess_shift
        } else {
            0
        };
        (i64::from(QuantizedGradients::grad_level(packed)) << self.grad_shift) + hess + 1
    }

    /// `(grad_shift, hess_shift, hess_mask, count_mask, hess_level_per_row)`
    /// for decoding many sums in a loop: grad levels are `sum >> grad_shift`,
    /// rows `sum & count_mask`, and hess levels
    /// `((sum >> hess_shift) & hess_mask) + rows * hess_level_per_row` (the
    /// mask is zero under a constant Hessian, the per-row level zero otherwise).
    #[inline(always)]
    pub fn fields(self) -> (u32, u32, i64, i64, i64) {
        let hess_mask = (1_i64 << (self.grad_shift - self.hess_shift)) - 1;
        (
            self.grad_shift,
            self.hess_shift,
            hess_mask,
            (1_i64 << self.hess_shift) - 1,
            self.constant_hess_level,
        )
    }

    /// Split a summed triple back into `(grad_level_sum, hess_level_sum, count)`.
    #[inline(always)]
    pub fn decode(self, sum: i64) -> (i64, i64, u32) {
        let count_mask = (1_i64 << self.hess_shift) - 1;
        let count = sum & count_mask;
        let hess = if self.constant_hess_level == 0 {
            let hess_mask = (1_i64 << (self.grad_shift - self.hess_shift)) - 1;
            (sum >> self.hess_shift) & hess_mask
        } else {
            count * self.constant_hess_level
        };
        (sum >> self.grad_shift, hess, count as u32)
    }
}

/// A backend whose histograms come from one round's quantized gradients.
///
/// The tree builders pass the float gradients everywhere; this wrapper swaps
/// in [`BackendOps::build_histograms_quantized`] whenever a histogram is asked
/// for over exactly those gradients, and forwards everything else (split
/// scans, partitions and their float leaf statistics) to the inner backend.
///
/// With `quantized: None` it forwards every call unchanged, so a fit without
/// quantization trains exactly as it would on the inner backend.
pub(crate) struct QuantizedHistogramBackend<'a, B: BackendOps> {
    inner: &'a B,
    source: &'a [GradientPair],
    quantized: Option<&'a QuantizedGradients>,
}

impl<'a, B: BackendOps> QuantizedHistogramBackend<'a, B> {
    pub(crate) fn new(
        inner: &'a B,
        source: &'a [GradientPair],
        quantized: Option<&'a QuantizedGradients>,
    ) -> Self {
        debug_assert!(quantized.is_none_or(|quantized| quantized.len() == source.len()));
        Self {
            inner,
            source,
            quantized,
        }
    }

    /// The quantized gradients to build from, when the request is for
    /// histograms over exactly the round's float gradients.
    fn quantized_for(
        &self,
        gradients: &[GradientPair],
        include_grad_sq: bool,
    ) -> Option<&'a QuantizedGradients> {
        self.quantized.filter(|_| {
            !include_grad_sq
                && gradients.len() == self.source.len()
                && std::ptr::eq(gradients.as_ptr(), self.source.as_ptr())
        })
    }
}

/// Quantize one tree's gradients when the fit asked for it. `stream`
/// separates trees grown in the same round (multiclass classes).
pub(crate) fn quantize_tree_gradients(
    params: &TrainParams,
    gradients: &[GradientPair],
    seed_base: u64,
    round_index: usize,
    stream: u64,
) -> EngineResult<Option<QuantizedGradients>> {
    params
        .quantized_grad_bits
        .map(|bits| {
            let seed = mixed_hash(
                seed_base
                    ^ 0x5157_4E54_4752_4144
                    ^ mixed_hash(round_index as u64)
                    ^ mixed_hash(stream.wrapping_add(0xA5A5_A5A5)),
            );
            QuantizedGradients::quantize(gradients, bits, seed)
        })
        .transpose()
}

impl<B: BackendOps> BackendOps for QuantizedHistogramBackend<'_, B> {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        self.build_histograms_with_execution(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            false,
            HistogramExecution::Parallel,
        )
    }

    fn build_histograms_with_grad_sq(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        include_grad_sq: bool,
    ) -> EngineResult<HistogramBundle> {
        self.build_histograms_with_execution(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            include_grad_sq,
            HistogramExecution::Parallel,
        )
    }

    fn build_histograms_with_execution(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        include_grad_sq: bool,
        execution: HistogramExecution,
    ) -> EngineResult<HistogramBundle> {
        if let Some(quantized) = self.quantized_for(gradients, include_grad_sq) {
            self.inner.build_histograms_quantized(
                binned_matrix,
                quantized,
                node,
                feature_tiles,
                execution,
            )
        } else {
            self.inner.build_histograms_with_execution(
                binned_matrix,
                gradients,
                node,
                feature_tiles,
                include_grad_sq,
                execution,
            )
        }
    }

    fn build_histograms_quantized(
        &self,
        binned_matrix: &BinnedMatrix,
        quantized: &QuantizedGradients,
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        execution: HistogramExecution,
    ) -> EngineResult<HistogramBundle> {
        self.inner.build_histograms_quantized(
            binned_matrix,
            quantized,
            node,
            feature_tiles,
            execution,
        )
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split(histograms)
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_with_options(
            histograms,
            options,
            feature_weights,
            categorical_features,
        )
    }

    fn shortlist_standard_splits(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        max_numeric_features: usize,
    ) -> EngineResult<SplitShortlist> {
        self.inner.shortlist_standard_splits(
            histograms,
            options,
            feature_weights,
            categorical_features,
            max_numeric_features,
        )
    }

    fn best_split_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
        )
    }

    fn best_split_with_noise(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
        noise: Option<SplitNoise>,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_with_noise(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
            noise,
        )
    }

    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_morph(
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
        )
    }

    fn best_split_morph_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_morph_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
            factor_context,
        )
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        self.inner.apply_split(binned_matrix, node, split)
    }

    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        self.inner
            .apply_split_with_stats(binned_matrix, gradients, node, split)
    }

    fn apply_split_owned_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        self.inner
            .apply_split_owned_with_stats(binned_matrix, gradients, node, split)
    }

    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats> {
        self.inner.reduce_sums(gradients, row_indices)
    }

    fn build_linear_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        regressor_features: &[u32],
        feature_scaler: &LinearFeatureScaler,
        raw_feature_values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> EngineResult<LinearHistogramBundle> {
        self.inner.build_linear_histograms(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            regressor_features,
            feature_scaler,
            raw_feature_values,
            row_count,
            feature_count,
        )
    }

    fn best_split_linear(
        &self,
        linear_histograms: &LinearHistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        ctx: &LinearContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.inner.best_split_linear(
            linear_histograms,
            options,
            feature_weights,
            categorical_features,
            ctx,
        )
    }

    fn evaluate_shortlisted_linear_feature(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split_feature_index: u32,
        linear_context: &LinearContext,
        feature_scaler: &LinearFeatureScaler,
        raw_feature_values: &[f32],
        row_count: usize,
        feature_count: usize,
        options: SplitSelectionOptions,
        learning_rate: f32,
        parent_leaf_value: f32,
        parent_linear_leaf: Option<&LinearLeaf>,
    ) -> EngineResult<Option<PreparedLinearSplit>> {
        self.inner.evaluate_shortlisted_linear_feature(
            binned_matrix,
            gradients,
            node,
            split_feature_index,
            linear_context,
            feature_scaler,
            raw_feature_values,
            row_count,
            feature_count,
            options,
            learning_rate,
            parent_leaf_value,
            parent_linear_leaf,
        )
    }

    fn compute_linear_leaf_pair(
        &self,
        linear_histograms: &LinearHistogramBundle,
        feature_index: u32,
        threshold_bin: usize,
        default_left: bool,
        missing_bin_index: usize,
        learning_rate: f32,
        l2_lambda: f32,
        feature_scaler: &LinearFeatureScaler,
        max_abs_leaf_value: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        self.inner.compute_linear_leaf_pair(
            linear_histograms,
            feature_index,
            threshold_bin,
            default_left,
            missing_bin_index,
            learning_rate,
            l2_lambda,
            feature_scaler,
            max_abs_leaf_value,
        )
    }

    fn compute_linear_leaf_pair_from_partitions(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        raw_feature_values: &[f32],
        feature_count: usize,
        split_feature_index: u32,
        threshold_bin: u16,
        default_left: bool,
        regressor_features: &[u32],
        feature_scaler: &LinearFeatureScaler,
        left_rows: &[u32],
        right_rows: &[u32],
        learning_rate: f32,
        l2_lambda: f32,
        max_abs_leaf_value: f32,
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        self.inner.compute_linear_leaf_pair_from_partitions(
            binned_matrix,
            gradients,
            raw_feature_values,
            feature_count,
            split_feature_index,
            threshold_bin,
            default_left,
            regressor_features,
            feature_scaler,
            left_rows,
            right_rows,
            learning_rate,
            l2_lambda,
            max_abs_leaf_value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(values: &[(f32, f32)]) -> Vec<GradientPair> {
        values
            .iter()
            .map(|&(grad, hess)| GradientPair { grad, hess })
            .collect()
    }

    #[test]
    fn levels_stay_in_range_and_constant_hessians_are_exact() {
        let gradients = pairs(&[(-2.0, 1.0), (0.3, 1.0), (2.0, 1.0), (0.0, 1.0)]);
        let quantized = QuantizedGradients::quantize(&gradients, 4, 7).unwrap();
        assert!(quantized.constant_hess());
        for (&packed, pair) in quantized.packed().iter().zip(&gradients) {
            let grad = QuantizedGradients::grad_level(packed);
            assert!((-7..=7).contains(&grad));
            assert_eq!(QuantizedGradients::hess_level(packed), 14);
            assert_eq!(quantized.dequantize(packed).hess, pair.hess);
        }
        assert_eq!(QuantizedGradients::grad_level(quantized.packed()[0]), -7);
        assert_eq!(QuantizedGradients::grad_level(quantized.packed()[2]), 7);
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        let gradients = vec![
            GradientPair {
                grad: 0.37,
                hess: 0.61
            };
            200_000
        ];
        let mut with_extremes = gradients.clone();
        with_extremes.push(GradientPair {
            grad: 1.0,
            hess: 1.0,
        });
        let quantized = QuantizedGradients::quantize(&with_extremes, 3, 11).unwrap();
        let rows = gradients.len();
        let (grad_sum, hess_sum) = quantized.packed()[..rows]
            .iter()
            .map(|&packed| quantized.dequantize(packed))
            .fold((0.0_f64, 0.0_f64), |(g, h), pair| {
                (g + f64::from(pair.grad), h + f64::from(pair.hess))
            });
        assert!((grad_sum / rows as f64 - 0.37).abs() < 2e-3);
        assert!((hess_sum / rows as f64 - 0.61).abs() < 2e-3);
    }

    #[test]
    fn quantization_is_reproducible_per_seed() {
        let gradients = pairs(&[(0.1, 0.2), (-0.4, 0.9), (0.25, 0.5), (0.9, 0.1)]);
        let a = QuantizedGradients::quantize(&gradients, 4, 3).unwrap();
        let b = QuantizedGradients::quantize(&gradients, 4, 3).unwrap();
        assert_eq!(a.packed(), b.packed());
    }

    #[test]
    fn packing_round_trips_signed_sums() {
        let packing = QuantizedPacking::for_rows(1000, 4, false).unwrap();
        let values = [(-7, 14), (3, 0), (-1, 5), (7, 14)];
        let sum: i64 = values
            .iter()
            .map(|&(g, h)| packing.expand((g << 16) | h))
            .sum();
        assert_eq!(packing.decode(sum), (2, 33, 4));
        let all_negative: i64 = (0..1000).map(|_| packing.expand((-7 << 16) | 14)).sum();
        assert_eq!(packing.decode(all_negative), (-7000, 14_000, 1000));
    }

    #[test]
    fn packing_refuses_nodes_that_overflow() {
        assert!(QuantizedPacking::for_rows(200_000, 4, false).is_some());
        assert!(QuantizedPacking::for_rows(10_000_000, 4, false).is_none());
        assert!(QuantizedPacking::for_rows(60_000, 8, false).is_some());
        assert!(QuantizedPacking::for_rows(1_000_000, 8, false).is_none());
        // Without a Hessian field, far larger nodes fit.
        assert!(QuantizedPacking::for_rows(100_000_000, 8, true).is_some());
    }

    #[test]
    fn constant_hessian_packing_derives_hessian_sums_from_counts() {
        let packing = QuantizedPacking::for_rows(5_000_000, 4, true).unwrap();
        let sum: i64 = [(-7, 14), (3, 14), (7, 14)]
            .iter()
            .map(|&(g, h)| packing.expand((g << 16) | h))
            .sum();
        assert_eq!(packing.decode(sum), (3, 42, 3));
    }

    #[test]
    fn rejects_unsupported_bit_widths() {
        let gradients = pairs(&[(0.1, 0.2)]);
        assert!(QuantizedGradients::quantize(&gradients, 1, 0).is_err());
        assert!(QuantizedGradients::quantize(&gradients, 9, 0).is_err());
    }
}
