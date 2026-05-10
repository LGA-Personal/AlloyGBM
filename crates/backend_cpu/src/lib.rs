use alloygbm_core::{
    BinnedMatrix, Device, FeatureHistogram, FeatureTile, GradientPair, HistogramBin,
    HistogramBundle, LinearFeatureHistogram, LinearHistogramBundle, LinearLeaf, NodeSlice,
    NodeStats, PartitionResult, SplitCandidate, leaf_effective_gradient, leaf_gain_term,
};
use alloygbm_engine::{
    BackendOps, CategoricalFeatureInfo, EngineError, EngineResult, FactorSplitContext,
    LinearContext, MorphContext, SplitSelectionOptions,
};
use rayon::prelude::*;
use std::cell::RefCell;

mod morph;
pub use morph::{MorphGainInputs, SplitSideStats, compute_morph_gain};

mod pl_histogram;
pub use pl_histogram::build_linear_histograms_cpu;

mod pl;

pub use alloygbm_core::simd;

thread_local! {
    /// Per-thread reusable histogram arena to avoid repeated allocation.
    static THREAD_ARENA: RefCell<HistogramArena> = RefCell::new(HistogramArena::new(0, 0));
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackend;

const SMALL_TILE_WORKLOAD_THRESHOLD: usize = 16_384;
const PARALLEL_TILE_WORKLOAD_THRESHOLD: usize = 131_072;
const TINY_NODE_ROW_THRESHOLD: usize = 32;
const BIN_HEAVY_THRESHOLD: usize = 512;

/// Controls which gain formula is used inside `best_split_for_feature_inner`.
///
/// `Standard` uses the XGBoost gain formula.
/// `Morph` delegates to `compute_morph_gain` from the morph module.
enum GainStrategy<'a> {
    Standard,
    Morph(&'a MorphContext),
}

#[derive(Debug, Clone, Copy)]
struct ScalarSideStats {
    grad: f32,
    hess: f32,
    grad_sq: f32,
    count: u32,
}

#[derive(Debug, Clone, Copy)]
struct MissingDirectionCandidate {
    left: ScalarSideStats,
    right: ScalarSideStats,
    default_left: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistogramKernelPath {
    TinyNodeScalar,
    BinHeavyPerFeatureScalar,
    ArenaRowFirstUnrolled,
}

#[derive(Debug, Clone)]
struct HistogramArena {
    bin_count: usize,
    grad_sums: Vec<f32>,
    hess_sums: Vec<f32>,
    grad_sq_sums: Vec<f32>,
    counts: Vec<u32>,
}

impl HistogramArena {
    fn new(tile_feature_count: usize, bin_count: usize) -> Self {
        let flat_len = tile_feature_count * bin_count;
        Self {
            bin_count,
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            grad_sq_sums: vec![0.0; flat_len],
            counts: vec![0; flat_len],
        }
    }

    /// Zero all accumulators without deallocating, allowing the arena to be reused.
    fn reset(&mut self) {
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
        self.grad_sq_sums.fill(0.0);
        self.counts.fill(0);
    }

    /// Resize the arena to handle a new tile size without unnecessary re-allocation.
    /// Only reallocates if the new tile requires more capacity.
    fn resize_for_tile(&mut self, tile_feature_count: usize, bin_count: usize) {
        let flat_len = tile_feature_count * bin_count;
        self.bin_count = bin_count;
        if self.grad_sums.len() == flat_len {
            self.reset();
        } else {
            self.grad_sums.resize(flat_len, 0.0);
            self.hess_sums.resize(flat_len, 0.0);
            self.grad_sq_sums.resize(flat_len, 0.0);
            self.counts.resize(flat_len, 0);
            self.grad_sums.fill(0.0);
            self.hess_sums.fill(0.0);
            self.grad_sq_sums.fill(0.0);
            self.counts.fill(0);
        }
    }

    fn materialize(&self, start_feature: usize, feature_histograms: &mut Vec<FeatureHistogram>) {
        CpuBackend::materialize_tile_histograms(
            start_feature,
            self.bin_count,
            &self.grad_sums,
            &self.hess_sums,
            &self.grad_sq_sums,
            &self.counts,
            feature_histograms,
        );
    }
}

/// Apply a per-feature weight to a split candidate's gain for cross-feature comparison.
///
/// The gain stored in `SplitCandidate` remains unweighted (the true gain);
/// the weighted gain is only used when comparing splits across features.
fn apply_feature_weight(candidate: &SplitCandidate, feature_weights: &[f32]) -> f32 {
    let fi = candidate.feature_index as usize;
    if fi < feature_weights.len() {
        candidate.gain * feature_weights[fi]
    } else {
        candidate.gain
    }
}

impl CpuBackend {
    pub fn device(&self) -> Device {
        Device::Cpu
    }

    fn select_histogram_kernel_path(
        row_count: usize,
        tile_workload: usize,
        bin_count: usize,
    ) -> HistogramKernelPath {
        if row_count <= TINY_NODE_ROW_THRESHOLD || tile_workload <= SMALL_TILE_WORKLOAD_THRESHOLD {
            HistogramKernelPath::TinyNodeScalar
        } else if bin_count >= BIN_HEAVY_THRESHOLD {
            HistogramKernelPath::BinHeavyPerFeatureScalar
        } else {
            HistogramKernelPath::ArenaRowFirstUnrolled
        }
    }

    fn materialize_tile_histograms(
        start_feature: usize,
        bin_count: usize,
        grad_sums: &[f32],
        hess_sums: &[f32],
        grad_sq_sums: &[f32],
        counts: &[u32],
        feature_histograms: &mut Vec<FeatureHistogram>,
    ) {
        let tile_feature_count = grad_sums.len() / bin_count;
        for local_feature_index in 0..tile_feature_count {
            let base = local_feature_index * bin_count;
            let mut bins = Vec::with_capacity(bin_count);
            for bin_index in 0..bin_count {
                let flat_index = base + bin_index;
                bins.push(HistogramBin {
                    grad_sum: grad_sums[flat_index],
                    hess_sum: hess_sums[flat_index],
                    grad_sq_sum: grad_sq_sums[flat_index],
                    count: counts[flat_index],
                });
            }

            feature_histograms.push(FeatureHistogram {
                feature_index: (start_feature + local_feature_index) as u32,
                bins,
            });
        }
    }

    fn build_tile_histograms_per_feature(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        start_feature: usize,
        end_feature: usize,
        bin_count: usize,
        feature_histograms: &mut Vec<FeatureHistogram>,
    ) {
        let row_count = binned_matrix.row_count;
        let use_col_major = binned_matrix.has_col_major();
        for feature_index in start_feature..end_feature {
            let mut bins = vec![
                HistogramBin {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    count: 0,
                };
                bin_count
            ];

            if use_col_major {
                // Column-major: sequential bin reads — cache-friendly
                let col_base = feature_index * row_count;
                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let bin_index = binned_matrix.col_bin(col_base + row_index) as usize;
                    let gradient = gradients[row_index];
                    let target_bin = &mut bins[bin_index];
                    target_bin.grad_sum += gradient.grad;
                    target_bin.hess_sum += gradient.hess;
                    target_bin.grad_sq_sum += gradient.grad * gradient.grad;
                    target_bin.count += 1;
                }
            } else {
                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let cell_index = row_index * binned_matrix.feature_count + feature_index;
                    let bin_index = binned_matrix.row_bin(cell_index) as usize;
                    let gradient = gradients[row_index];
                    let target_bin = &mut bins[bin_index];
                    target_bin.grad_sum += gradient.grad;
                    target_bin.hess_sum += gradient.hess;
                    target_bin.grad_sq_sum += gradient.grad * gradient.grad;
                    target_bin.count += 1;
                }
            }

            feature_histograms.push(FeatureHistogram {
                feature_index: feature_index as u32,
                bins,
            });
        }
    }

    #[cfg(test)]
    fn build_tile_histograms_row_first(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        start_feature: usize,
        end_feature: usize,
        arena: &mut HistogramArena,
    ) {
        let tile_feature_count = end_feature - start_feature;

        for &row_index in &node.row_indices {
            let row_index = row_index as usize;
            let row_base = row_index * binned_matrix.feature_count + start_feature;
            let gradient = gradients[row_index];
            for local_feature_index in 0..tile_feature_count {
                let bin_index = binned_matrix.row_bin(row_base + local_feature_index) as usize;
                let flat_index = local_feature_index * arena.bin_count + bin_index;
                arena.grad_sums[flat_index] += gradient.grad;
                arena.hess_sums[flat_index] += gradient.hess;
                arena.grad_sq_sums[flat_index] += gradient.grad * gradient.grad;
                arena.counts[flat_index] += 1;
            }
        }
    }

    fn should_parallelize_tiles(
        feature_tile_count: usize,
        row_count: usize,
        selected_feature_count: usize,
    ) -> bool {
        feature_tile_count > 1
            && row_count.saturating_mul(selected_feature_count) >= PARALLEL_TILE_WORKLOAD_THRESHOLD
            && rayon::current_num_threads() > 1
    }

    fn build_feature_histograms_for_tile(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        tile: &FeatureTile,
        bin_count: usize,
    ) -> EngineResult<Vec<FeatureHistogram>> {
        let feature_count = binned_matrix.feature_count;
        if tile.end_feature as usize > feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature tile end {} exceeds feature_count {}",
                tile.end_feature, feature_count
            )));
        }

        let start_feature = tile.start_feature as usize;
        let end_feature = tile.end_feature as usize;
        let tile_feature_count = end_feature - start_feature;
        let tile_workload = node.row_indices.len().saturating_mul(tile_feature_count);
        let mut feature_histograms = Vec::with_capacity(tile_feature_count);

        match Self::select_histogram_kernel_path(node.row_indices.len(), tile_workload, bin_count) {
            HistogramKernelPath::TinyNodeScalar | HistogramKernelPath::BinHeavyPerFeatureScalar => {
                Self::build_tile_histograms_per_feature(
                    binned_matrix,
                    gradients,
                    node,
                    start_feature,
                    end_feature,
                    bin_count,
                    &mut feature_histograms,
                );
            }
            HistogramKernelPath::ArenaRowFirstUnrolled => {
                if binned_matrix.has_col_major() {
                    // Feature-first with column-major bins: 3KB working set per feature (fits L1).
                    Self::build_tile_histograms_per_feature(
                        binned_matrix,
                        gradients,
                        node,
                        start_feature,
                        end_feature,
                        bin_count,
                        &mut feature_histograms,
                    );
                } else {
                    THREAD_ARENA.with(|cell| {
                        let mut arena = cell.borrow_mut();
                        arena.resize_for_tile(tile_feature_count, bin_count);
                        Self::build_tile_histograms_row_first_unrolled(
                            binned_matrix,
                            gradients,
                            node,
                            start_feature,
                            end_feature,
                            &mut arena,
                        );
                        arena.materialize(start_feature, &mut feature_histograms);
                    });
                }
            }
        }

        Ok(feature_histograms)
    }

    fn build_histograms_internal(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        parallel_tiles: bool,
    ) -> EngineResult<HistogramBundle> {
        if gradients.len() != binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "gradients length {} does not match row_count {}",
                gradients.len(),
                binned_matrix.row_count
            )));
        }
        if feature_tiles.is_empty() {
            return Err(EngineError::ContractViolation(
                "feature_tiles cannot be empty".to_string(),
            ));
        }
        node.validate_bounds(binned_matrix.row_count)?;

        let bin_count = binned_matrix.max_bin as usize + 1;
        let selected_feature_count = feature_tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum();
        let mut feature_histograms = Vec::with_capacity(selected_feature_count);

        if parallel_tiles {
            let per_tile_histograms = feature_tiles
                .par_iter()
                .map(|tile| {
                    Self::build_feature_histograms_for_tile(
                        binned_matrix,
                        gradients,
                        node,
                        tile,
                        bin_count,
                    )
                })
                .collect::<Vec<_>>();

            for tile_histograms in per_tile_histograms {
                feature_histograms.extend(tile_histograms?);
            }
        } else {
            for tile in feature_tiles {
                feature_histograms.extend(Self::build_feature_histograms_for_tile(
                    binned_matrix,
                    gradients,
                    node,
                    tile,
                    bin_count,
                )?);
            }
        }

        Ok(HistogramBundle {
            node_id: node.node_id,
            feature_histograms,
        })
    }

    fn build_tile_histograms_row_first_unrolled(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        start_feature: usize,
        end_feature: usize,
        arena: &mut HistogramArena,
    ) {
        let tile_feature_count = end_feature - start_feature;
        let feature_count = binned_matrix.feature_count;

        // Process rows in 8-wide chunks to improve instruction-level parallelism.
        let mut row_chunks = node.row_indices.chunks_exact(8);
        for row_chunk in &mut row_chunks {
            let row0 = row_chunk[0] as usize;
            let row1 = row_chunk[1] as usize;
            let row2 = row_chunk[2] as usize;
            let row3 = row_chunk[3] as usize;
            let row4 = row_chunk[4] as usize;
            let row5 = row_chunk[5] as usize;
            let row6 = row_chunk[6] as usize;
            let row7 = row_chunk[7] as usize;

            let gradient0 = gradients[row0];
            let gradient1 = gradients[row1];
            let gradient2 = gradients[row2];
            let gradient3 = gradients[row3];
            let gradient4 = gradients[row4];
            let gradient5 = gradients[row5];
            let gradient6 = gradients[row6];
            let gradient7 = gradients[row7];

            let row_base0 = row0 * feature_count + start_feature;
            let row_base1 = row1 * feature_count + start_feature;
            let row_base2 = row2 * feature_count + start_feature;
            let row_base3 = row3 * feature_count + start_feature;
            let row_base4 = row4 * feature_count + start_feature;
            let row_base5 = row5 * feature_count + start_feature;
            let row_base6 = row6 * feature_count + start_feature;
            let row_base7 = row7 * feature_count + start_feature;

            for local_feature_index in 0..tile_feature_count {
                let base = local_feature_index * arena.bin_count;

                let idx0 = base + binned_matrix.row_bin(row_base0 + local_feature_index) as usize;
                let idx1 = base + binned_matrix.row_bin(row_base1 + local_feature_index) as usize;
                let idx2 = base + binned_matrix.row_bin(row_base2 + local_feature_index) as usize;
                let idx3 = base + binned_matrix.row_bin(row_base3 + local_feature_index) as usize;
                let idx4 = base + binned_matrix.row_bin(row_base4 + local_feature_index) as usize;
                let idx5 = base + binned_matrix.row_bin(row_base5 + local_feature_index) as usize;
                let idx6 = base + binned_matrix.row_bin(row_base6 + local_feature_index) as usize;
                let idx7 = base + binned_matrix.row_bin(row_base7 + local_feature_index) as usize;

                arena.grad_sums[idx0] += gradient0.grad;
                arena.hess_sums[idx0] += gradient0.hess;
                arena.grad_sq_sums[idx0] += gradient0.grad * gradient0.grad;
                arena.counts[idx0] += 1;

                arena.grad_sums[idx1] += gradient1.grad;
                arena.hess_sums[idx1] += gradient1.hess;
                arena.grad_sq_sums[idx1] += gradient1.grad * gradient1.grad;
                arena.counts[idx1] += 1;

                arena.grad_sums[idx2] += gradient2.grad;
                arena.hess_sums[idx2] += gradient2.hess;
                arena.grad_sq_sums[idx2] += gradient2.grad * gradient2.grad;
                arena.counts[idx2] += 1;

                arena.grad_sums[idx3] += gradient3.grad;
                arena.hess_sums[idx3] += gradient3.hess;
                arena.grad_sq_sums[idx3] += gradient3.grad * gradient3.grad;
                arena.counts[idx3] += 1;

                arena.grad_sums[idx4] += gradient4.grad;
                arena.hess_sums[idx4] += gradient4.hess;
                arena.grad_sq_sums[idx4] += gradient4.grad * gradient4.grad;
                arena.counts[idx4] += 1;

                arena.grad_sums[idx5] += gradient5.grad;
                arena.hess_sums[idx5] += gradient5.hess;
                arena.grad_sq_sums[idx5] += gradient5.grad * gradient5.grad;
                arena.counts[idx5] += 1;

                arena.grad_sums[idx6] += gradient6.grad;
                arena.hess_sums[idx6] += gradient6.hess;
                arena.grad_sq_sums[idx6] += gradient6.grad * gradient6.grad;
                arena.counts[idx6] += 1;

                arena.grad_sums[idx7] += gradient7.grad;
                arena.hess_sums[idx7] += gradient7.hess;
                arena.grad_sq_sums[idx7] += gradient7.grad * gradient7.grad;
                arena.counts[idx7] += 1;
            }
        }

        for &row_index in row_chunks.remainder() {
            let row_index = row_index as usize;
            let row_base = row_index * feature_count + start_feature;
            let gradient = gradients[row_index];
            for local_feature_index in 0..tile_feature_count {
                let bin_index = binned_matrix.row_bin(row_base + local_feature_index) as usize;
                let flat_index = local_feature_index * arena.bin_count + bin_index;
                arena.grad_sums[flat_index] += gradient.grad;
                arena.hess_sums[flat_index] += gradient.hess;
                arena.grad_sq_sums[flat_index] += gradient.grad * gradient.grad;
                arena.counts[flat_index] += 1;
            }
        }
    }

    fn best_split_for_feature(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        if options.dro_config.is_some() || factor_context.is_some() {
            return Self::best_split_for_feature_inner(
                feature_histogram,
                node_id,
                options,
                GainStrategy::Standard,
                factor_context,
            );
        }
        // Standard-path bin-scan goes through the SIMD-vectorized fast path.
        // The morph path retains the scalar implementation because its gain
        // formula calls `tanh`/`ln`/`exp`, which are not safely vectorizable
        // through the `wide` crate.
        Self::best_split_for_feature_standard_simd(feature_histogram, node_id, options)
    }

    /// SIMD-vectorized standard-gain bin-scan for a single numeric feature.
    ///
    /// This is the fast path used by `best_split_for_feature` when the gain
    /// strategy is `GainStrategy::Standard`. Cumulative left-side stats are
    /// computed scalar-sequentially (the prefix scan is inherently serial),
    /// then per-bin gain candidates are evaluated 8-wide with `f32x8`.
    ///
    /// Output is byte-identical to `best_split_for_feature_inner(_, _, _,
    /// GainStrategy::Standard)` within float-rounding tolerance (verified by
    /// the parity tests in this module).
    fn best_split_for_feature_standard_simd(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
    ) -> Option<SplitCandidate> {
        use crate::simd::{f32x8, l1_threshold_f32x8};
        use wide::{CmpGe, CmpGt};

        const EPSILON: f32 = 1e-6;

        if feature_histogram.bins.len() < 2 {
            return None;
        }

        // Extract missing-value stats if the histogram covers the NaN bin.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
            if missing_bin_idx < feature_histogram.bins.len() {
                let mb = &feature_histogram.bins[missing_bin_idx];
                (mb.grad_sum, mb.hess_sum, mb.grad_sq_sum, mb.count)
            } else {
                (0.0_f32, 0.0_f32, 0.0_f32, 0_u32)
            };

        let mut total_grad = 0.0_f32;
        let mut total_hess = 0.0_f32;
        let mut total_grad_sq = 0.0_f32;
        let mut total_count = 0_u32;
        for bin in &feature_histogram.bins {
            total_grad += bin.grad_sum;
            total_hess += bin.hess_sum;
            total_grad_sq += bin.grad_sq_sum;
            total_count += bin.count;
        }

        if total_hess <= options.min_child_hessian {
            return None;
        }

        let nm_total_grad = total_grad - missing_grad;
        let nm_total_hess = total_hess - missing_hess;
        let nm_total_grad_sq = total_grad_sq - missing_grad_sq;
        let nm_total_count = total_count.saturating_sub(missing_count);

        let parent_denom = total_hess + options.l2_lambda + EPSILON;
        let parent_grad = l1_threshold_gradient(total_grad, options.l1_alpha);
        let parent_gain_term = (parent_grad * parent_grad) / parent_denom;

        let scan_limit = feature_histogram.bins.len().min(missing_bin_idx);
        if scan_limit == 0 {
            return None;
        }

        // Pre-compute scalar cumulative left-side stats. The prefix scan is
        // inherently sequential, so we keep it in scalar code.
        let mut cum_left_grad = vec![0.0_f32; scan_limit];
        let mut cum_left_hess = vec![0.0_f32; scan_limit];
        let mut cum_left_grad_sq = vec![0.0_f32; scan_limit];
        let mut cum_left_count = vec![0_u32; scan_limit];
        {
            let mut g = 0.0_f32;
            let mut h = 0.0_f32;
            let mut q = 0.0_f32;
            let mut c = 0_u32;
            for (i, bin) in feature_histogram.bins.iter().enumerate().take(scan_limit) {
                g += bin.grad_sum;
                h += bin.hess_sum;
                q += bin.grad_sq_sum;
                c += bin.count;
                cum_left_grad[i] = g;
                cum_left_hess[i] = h;
                cum_left_grad_sq[i] = q;
                cum_left_count[i] = c;
            }
        }

        // Per-NaN-direction broadcast values.
        let l1_alpha = options.l1_alpha;
        let l2_lambda = options.l2_lambda;
        let min_child_hessian = options.min_child_hessian;
        let min_leaf_mag = options.min_leaf_magnitude;
        let nm_total_grad_v = f32x8::splat(nm_total_grad);
        let nm_total_hess_v = f32x8::splat(nm_total_hess);
        let nm_total_count_f = nm_total_count as f32;
        let nm_total_count_v = f32x8::splat(nm_total_count_f);
        let missing_grad_v = f32x8::splat(missing_grad);
        let missing_hess_v = f32x8::splat(missing_hess);
        let missing_count_f = missing_count as f32;
        let missing_count_v = f32x8::splat(missing_count_f);
        let l2_lambda_v = f32x8::splat(l2_lambda);
        let eps_v = f32x8::splat(EPSILON);
        let min_child_hess_v = f32x8::splat(min_child_hessian);
        let min_leaf_mag_v = f32x8::splat(min_leaf_mag);
        let parent_gain_term_v = f32x8::splat(parent_gain_term);
        let neg_inf_v = f32x8::splat(f32::NEG_INFINITY);

        // Best result tracking — store as (gain, threshold_bin, default_left).
        // Final SplitCandidate is built once at the end from the cumulative arrays.
        let mut best_gain = 0.0_f32;
        let mut best_threshold: usize = usize::MAX;
        let mut best_default_left = false;

        // For each NaN direction, evaluate gain across all bins in 8-wide chunks.
        for &default_left in &[true, false] {
            let nan_left_mask = default_left;
            // For each chunk-of-8 starting at `chunk_start`:
            let mut chunk_start = 0usize;
            while chunk_start < scan_limit {
                let chunk_end = (chunk_start + 8).min(scan_limit);
                let chunk_len = chunk_end - chunk_start;

                // Load 8 lanes of cumulative left stats (zero-pad the tail).
                let mut lg_arr = [0.0_f32; 8];
                let mut lh_arr = [0.0_f32; 8];
                let mut lc_arr = [0.0_f32; 8];
                for j in 0..chunk_len {
                    lg_arr[j] = cum_left_grad[chunk_start + j];
                    lh_arr[j] = cum_left_hess[chunk_start + j];
                    lc_arr[j] = cum_left_count[chunk_start + j] as f32;
                }
                let lg_v = f32x8::from(lg_arr);
                let lh_v = f32x8::from(lh_arr);
                let lc_v = f32x8::from(lc_arr);

                // Right-side stats (before NaN routing).
                let rg_v = nm_total_grad_v - lg_v;
                let rh_v = nm_total_hess_v - lh_v;
                let rc_v = nm_total_count_v - lc_v;

                // Apply NaN-direction routing.
                let (eff_lg, eff_lh, eff_lc, eff_rg, eff_rh, eff_rc) = if nan_left_mask {
                    (
                        lg_v + missing_grad_v,
                        lh_v + missing_hess_v,
                        lc_v + missing_count_v,
                        rg_v,
                        rh_v,
                        rc_v,
                    )
                } else {
                    (
                        lg_v,
                        lh_v,
                        lc_v,
                        rg_v + missing_grad_v,
                        rh_v + missing_hess_v,
                        rc_v + missing_count_v,
                    )
                };

                // L1-thresholded gradient sums.
                let lg_l1 = l1_threshold_f32x8(eff_lg, l1_alpha);
                let rg_l1 = l1_threshold_f32x8(eff_rg, l1_alpha);

                // Denominators.
                let l_denom = eff_lh + l2_lambda_v + eps_v;
                let r_denom = eff_rh + l2_lambda_v + eps_v;

                // Gain.
                let gain_v =
                    (lg_l1 * lg_l1) / l_denom + (rg_l1 * rg_l1) / r_denom - parent_gain_term_v;

                // Validity mask:
                //   eff_lc != 0 (lc > 0)
                //   eff_rc != 0 (rc > 0)
                //   eff_lh > min_child_hessian
                //   eff_rh > min_child_hessian
                let zero_v = f32x8::ZERO;
                let lc_pos = eff_lc.cmp_gt(zero_v);
                let rc_pos = eff_rc.cmp_gt(zero_v);
                let lh_ok = eff_lh.cmp_gt(min_child_hess_v);
                let rh_ok = eff_rh.cmp_gt(min_child_hess_v);
                // Combine via bitwise AND on the float-mask representation.
                let valid_mask = lc_pos & rc_pos & lh_ok & rh_ok;

                // min_leaf_magnitude filter: candidate is rejected when BOTH
                // sides' leaf magnitudes are below the threshold.
                let final_gain = if min_leaf_mag > 0.0 {
                    let l_leaf_mag = lg_l1.abs() / l_denom;
                    let r_leaf_mag = rg_l1.abs() / r_denom;
                    // pass if either side >= min_leaf_mag (i.e. NOT both below).
                    let l_passes = l_leaf_mag.cmp_ge(min_leaf_mag_v);
                    let r_passes = r_leaf_mag.cmp_ge(min_leaf_mag_v);
                    let leaf_mag_ok = l_passes | r_passes;
                    let combined = valid_mask & leaf_mag_ok;
                    // If combined-mask is all-ones (lane valid), keep gain;
                    // else replace with -inf.
                    combined.blend(gain_v, neg_inf_v)
                } else {
                    valid_mask.blend(gain_v, neg_inf_v)
                };

                // Extract gain to scalar for tail-masking, edge-threshold
                // rejection, and horizontal argmax. The lane-extract overhead
                // is small (8 floats) compared to the vectorized gain math.
                let mut g_arr = final_gain.to_array();
                // Mask out tail-padded lanes.
                for slot in g_arr.iter_mut().skip(chunk_len) {
                    *slot = f32::NEG_INFINITY;
                }
                // Skip "edge" thresholds where left covers all non-missing bins.
                // Replicates the scalar check:
                //   if threshold_bin + 1 >= scan_limit && nm_total_count == left_count { skip }
                for (j, slot) in g_arr.iter_mut().take(chunk_len).enumerate() {
                    let threshold_bin = chunk_start + j;
                    let left_count = cum_left_count[threshold_bin];
                    if threshold_bin + 1 >= scan_limit && nm_total_count == left_count {
                        *slot = f32::NEG_INFINITY;
                    }
                }
                // Horizontal argmax for this chunk against the running best.
                for (j, &g) in g_arr.iter().take(chunk_len).enumerate() {
                    if g > best_gain {
                        best_gain = g;
                        best_threshold = chunk_start + j;
                        best_default_left = default_left;
                    }
                }

                chunk_start = chunk_end;
            }
        }

        if best_threshold == usize::MAX {
            return None;
        }

        // Reconstruct the chosen candidate's stats from the cumulative arrays.
        let threshold_bin = best_threshold;
        let left_grad = cum_left_grad[threshold_bin];
        let left_hess = cum_left_hess[threshold_bin];
        let left_grad_sq = cum_left_grad_sq[threshold_bin];
        let left_count = cum_left_count[threshold_bin];
        let right_grad = nm_total_grad - left_grad;
        let right_hess = nm_total_hess - left_hess;
        let right_grad_sq = nm_total_grad_sq - left_grad_sq;
        let right_count = nm_total_count.saturating_sub(left_count);

        let (eff_lg, eff_lh, eff_lq, eff_lc, eff_rg, eff_rh, eff_rq, eff_rc) = if best_default_left
        {
            (
                left_grad + missing_grad,
                left_hess + missing_hess,
                left_grad_sq + missing_grad_sq,
                left_count + missing_count,
                right_grad,
                right_hess,
                right_grad_sq,
                right_count,
            )
        } else {
            (
                left_grad,
                left_hess,
                left_grad_sq,
                left_count,
                right_grad + missing_grad,
                right_hess + missing_hess,
                right_grad_sq + missing_grad_sq,
                right_count + missing_count,
            )
        };

        Some(SplitCandidate {
            node_id,
            feature_index: feature_histogram.feature_index,
            threshold_bin: threshold_bin as u16,
            gain: best_gain,
            default_left: best_default_left,
            is_categorical: false,
            categorical_bitset: None,
            left_stats: NodeStats {
                grad_sum: eff_lg,
                hess_sum: eff_lh,
                grad_sq_sum: eff_lq,
                row_count: eff_lc,
            },
            right_stats: NodeStats {
                grad_sum: eff_rg,
                hess_sum: eff_rh,
                grad_sq_sum: eff_rq,
                row_count: eff_rc,
            },
        })
    }

    /// Shared scaffold for numeric split finding, parameterised by gain strategy.
    ///
    /// This is the single source of truth for:
    /// - Missing-bin extraction
    /// - Total-hessian guard
    /// - Forward-scan accumulation
    /// - NaN-direction candidate generation
    /// - L1 thresholding (applied to both left and right gradient sums)
    /// - EPSILON denominators (1e-6)
    /// - `min_leaf_magnitude` filtering
    /// - `SplitCandidate` construction
    ///
    /// The ONLY divergence point is the gain formula, controlled by `GainStrategy`.
    fn best_split_for_feature_inner(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if feature_histogram.bins.len() < 2 {
            return None;
        }

        // Extract missing-value stats if the histogram covers the NaN bin.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
            if missing_bin_idx < feature_histogram.bins.len() {
                let mb = &feature_histogram.bins[missing_bin_idx];
                (mb.grad_sum, mb.hess_sum, mb.grad_sq_sum, mb.count)
            } else {
                (0.0, 0.0, 0.0, 0)
            };

        let mut total_grad = 0.0_f32;
        let mut total_hess = 0.0_f32;
        let mut total_grad_sq = 0.0_f32;
        let mut total_count = 0_u32;
        for bin in &feature_histogram.bins {
            total_grad += bin.grad_sum;
            total_hess += bin.hess_sum;
            total_grad_sq += bin.grad_sq_sum;
            total_count += bin.count;
        }

        if total_hess <= options.min_child_hessian {
            return None;
        }

        // Non-missing totals for the scan loop.
        let nm_total_grad = total_grad - missing_grad;
        let nm_total_hess = total_hess - missing_hess;
        let nm_total_grad_sq = total_grad_sq - missing_grad_sq;
        let nm_total_count = total_count.saturating_sub(missing_count);

        let parent_gain_term =
            split_gain_term(total_grad, total_hess, total_grad_sq, total_count, &options);

        let mut factor_scratch =
            factor_context.map(|ctx| FactorSplitScratch::new(ctx.exposures.factor_count));
        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        let mut left_grad = 0.0_f32;
        let mut left_hess = 0.0_f32;
        let mut left_grad_sq = 0.0_f32;
        let mut left_count = 0_u32;

        // Scan only non-missing bins (0..min(num_bins-1, MISSING_BIN-1)).
        let scan_limit = feature_histogram.bins.len().min(missing_bin_idx);
        for (threshold_bin, bin) in feature_histogram.bins.iter().enumerate().take(scan_limit) {
            left_grad += bin.grad_sum;
            left_hess += bin.hess_sum;
            left_grad_sq += bin.grad_sq_sum;
            left_count += bin.count;

            // Skip if this isn't a valid split point (need at least the
            // next non-missing bin on the right).
            if threshold_bin + 1 >= scan_limit && nm_total_count == left_count {
                continue;
            }

            let right_grad = nm_total_grad - left_grad;
            let right_hess = nm_total_hess - left_hess;
            let right_grad_sq = nm_total_grad_sq - left_grad_sq;
            let right_count = nm_total_count.saturating_sub(left_count);

            let candidates = [
                MissingDirectionCandidate {
                    left: ScalarSideStats {
                        grad: left_grad + missing_grad,
                        hess: left_hess + missing_hess,
                        grad_sq: left_grad_sq + missing_grad_sq,
                        count: left_count + missing_count,
                    },
                    right: ScalarSideStats {
                        grad: right_grad,
                        hess: right_hess,
                        grad_sq: right_grad_sq,
                        count: right_count,
                    },
                    default_left: true,
                },
                MissingDirectionCandidate {
                    left: ScalarSideStats {
                        grad: left_grad,
                        hess: left_hess,
                        grad_sq: left_grad_sq,
                        count: left_count,
                    },
                    right: ScalarSideStats {
                        grad: right_grad + missing_grad,
                        hess: right_hess + missing_hess,
                        grad_sq: right_grad_sq + missing_grad_sq,
                        count: right_count + missing_count,
                    },
                    default_left: false,
                },
            ];

            for candidate in candidates {
                let left = candidate.left;
                let right = candidate.right;
                if left.count == 0
                    || right.count == 0
                    || left.hess <= options.min_child_hessian
                    || right.hess <= options.min_child_hessian
                {
                    continue;
                }

                // Apply L1 thresholding uniformly before gain computation.
                let left_grad_for_gain = leaf_effective_gradient(
                    left.grad,
                    left.grad_sq,
                    left.count,
                    options.l1_alpha,
                    options.dro_config.as_ref(),
                );
                let right_grad_for_gain = leaf_effective_gradient(
                    right.grad,
                    right.grad_sq,
                    right.count,
                    options.l1_alpha,
                    options.dro_config.as_ref(),
                );
                let left_denom = left.hess + options.l2_lambda + EPSILON;
                let right_denom = right.hess + options.l2_lambda + EPSILON;

                // Apply min_leaf_magnitude filter uniformly.
                if options.min_leaf_magnitude > 0.0 {
                    let left_leaf_magnitude = left_grad_for_gain.abs() / left_denom;
                    let right_leaf_magnitude = right_grad_for_gain.abs() / right_denom;
                    if left_leaf_magnitude < options.min_leaf_magnitude
                        && right_leaf_magnitude < options.min_leaf_magnitude
                    {
                        continue;
                    }
                }

                let mut gain = match &strategy {
                    GainStrategy::Standard => {
                        split_gain_term(left.grad, left.hess, left.grad_sq, left.count, &options)
                            + split_gain_term(
                                right.grad,
                                right.hess,
                                right.grad_sq,
                                right.count,
                                &options,
                            )
                            - parent_gain_term
                    }
                    GainStrategy::Morph(morph) => {
                        use crate::morph::{MorphGainInputs, SplitSideStats, compute_morph_gain};
                        let inputs = MorphGainInputs {
                            left: SplitSideStats {
                                gradient_sum: left_grad_for_gain,
                                hessian_sum: left.hess,
                                count: left.count,
                            },
                            right: SplitSideStats {
                                gradient_sum: right_grad_for_gain,
                                hessian_sum: right.hess,
                                count: right.count,
                            },
                            iteration: morph.iteration,
                            total_iterations: morph.total_iterations.max(1),
                            grad_mean: morph.grad_mean,
                            grad_std: morph.grad_std,
                            lambda_l2: options.l2_lambda,
                        };
                        compute_morph_gain(inputs, &morph.config, &morph.precomputed)
                    }
                };

                if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_mut()) {
                    let left_leaf_value = -left_grad_for_gain / left_denom;
                    let right_leaf_value = -right_grad_for_gain / right_denom;
                    gain -= factor_split_penalty_for_candidate(
                        ctx,
                        scratch,
                        feature_histogram.feature_index,
                        threshold_bin as u16,
                        candidate.default_left,
                        None,
                        left_leaf_value,
                        right_leaf_value,
                    );
                }

                if !gain.is_finite() {
                    continue;
                }

                if gain > best_gain {
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index,
                        threshold_bin: threshold_bin as u16,
                        gain,
                        default_left: candidate.default_left,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: left.grad,
                            hess_sum: left.hess,
                            grad_sq_sum: left.grad_sq,
                            row_count: left.count,
                        },
                        right_stats: NodeStats {
                            grad_sum: right.grad,
                            hess_sum: right.hess,
                            grad_sq_sum: right.grad_sq,
                            row_count: right.count,
                        },
                    });
                }
            }
        }

        best_candidate
    }

    /// Find the best categorical split for a feature using the Fisher-sort algorithm.
    ///
    /// Algorithm:
    /// 1. Extract per-category stats from histogram bins (0..num_categories)
    /// 2. Sort categories by `grad_sum / (hess_sum + l2_lambda + eps)` ascending
    /// 3. Prefix scan over sorted order to find best binary partition
    /// 4. Build bitset from the best partition
    fn best_split_for_categorical_feature(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
        num_categories: usize,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        Self::best_split_for_categorical_feature_inner(
            feature_histogram,
            node_id,
            options,
            num_categories,
            GainStrategy::Standard,
            factor_context,
        )
    }

    /// Morph-mode best split for a single categorical feature.
    ///
    /// Thin wrapper around `best_split_for_categorical_feature_inner` with
    /// `GainStrategy::Morph`.
    fn best_split_morph_categorical_feature(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: &SplitSelectionOptions,
        num_categories: usize,
        morph: &MorphContext,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        Self::best_split_for_categorical_feature_inner(
            feature_histogram,
            node_id,
            *options,
            num_categories,
            GainStrategy::Morph(morph),
            factor_context,
        )
    }

    /// Shared scaffold for categorical split finding, parameterised by gain strategy.
    ///
    /// This is the single source of truth for:
    /// - Missing-bin extraction
    /// - Total-hessian guard
    /// - Fisher-sort ordering (by raw leaf score for standard mode, or by
    ///   DRO effective gradient score when robust scalar leaves are active)
    /// - Prefix-scan candidate generation
    /// - NaN-direction candidate generation
    /// - L1 thresholding (applied to both left and right gradient sums)
    /// - EPSILON denominators (1e-6)
    /// - `min_leaf_magnitude` filtering
    /// - Bitset construction
    /// - `SplitCandidate` construction
    ///
    /// The ONLY divergence point is the gain formula, controlled by `GainStrategy`.
    fn best_split_for_categorical_feature_inner(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
        num_categories: usize,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if num_categories < 2 {
            return None;
        }

        // Extract missing-value stats.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
            if missing_bin_idx < feature_histogram.bins.len() {
                let mb = &feature_histogram.bins[missing_bin_idx];
                (mb.grad_sum, mb.hess_sum, mb.grad_sq_sum, mb.count)
            } else {
                (0.0, 0.0, 0.0, 0)
            };

        // Collect populated categories (bins 0..num_categories).
        let mut categories: Vec<(u16, f32, f32, f32, u32)> = Vec::new(); // (bin_id, grad, hess, grad_sq, count)
        let mut nm_total_grad = 0.0_f32;
        let mut nm_total_hess = 0.0_f32;
        let mut nm_total_grad_sq = 0.0_f32;
        let mut nm_total_count = 0_u32;

        let scan_limit = num_categories
            .min(feature_histogram.bins.len())
            .min(missing_bin_idx);
        for bin_id in 0..scan_limit {
            let bin = &feature_histogram.bins[bin_id];
            if bin.count > 0 {
                categories.push((
                    bin_id as u16,
                    bin.grad_sum,
                    bin.hess_sum,
                    bin.grad_sq_sum,
                    bin.count,
                ));
            }
            nm_total_grad += bin.grad_sum;
            nm_total_hess += bin.hess_sum;
            nm_total_grad_sq += bin.grad_sq_sum;
            nm_total_count += bin.count;
        }

        if categories.len() < 2 {
            return None;
        }

        let total_grad = nm_total_grad + missing_grad;
        let total_hess = nm_total_hess + missing_hess;
        let total_grad_sq = nm_total_grad_sq + missing_grad_sq;

        if total_hess <= options.min_child_hessian {
            return None;
        }

        let total_count = nm_total_count + missing_count;
        let parent_gain_term =
            split_gain_term(total_grad, total_hess, total_grad_sq, total_count, &options);

        // Sort categories by the same gradient signal used for leaf values.
        // For standard mode this preserves the historical raw-gradient Fisher
        // ordering. For DRO mode, the robust effective gradient can reorder
        // categories when gradient dispersion changes the leaf signal.
        let dro_sort_config = options.dro_config.filter(|config| config.radius > 0.0);
        categories.sort_by(|a, b| {
            let grad_a = if dro_sort_config.is_some() {
                leaf_effective_gradient(a.1, a.3, a.4, options.l1_alpha, dro_sort_config.as_ref())
            } else {
                a.1
            };
            let grad_b = if dro_sort_config.is_some() {
                leaf_effective_gradient(b.1, b.3, b.4, options.l1_alpha, dro_sort_config.as_ref())
            } else {
                b.1
            };
            let score_a = grad_a / (a.2 + options.l2_lambda + EPSILON);
            let score_b = grad_b / (b.2 + options.l2_lambda + EPSILON);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Prefix scan over sorted categories to find best partition.
        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        let mut left_grad = 0.0_f32;
        let mut left_hess = 0.0_f32;
        let mut left_grad_sq = 0.0_f32;
        let mut left_count = 0_u32;
        let mut factor_scratch =
            factor_context.map(|ctx| FactorSplitScratch::new(ctx.exposures.factor_count));

        // Try splits: first k categories go left, rest go right (k = 1..len-1).
        for k in 0..categories.len() - 1 {
            let (_, g, h, q, c) = categories[k];
            left_grad += g;
            left_hess += h;
            left_grad_sq += q;
            left_count += c;

            let right_grad = nm_total_grad - left_grad;
            let right_hess = nm_total_hess - left_hess;
            let right_grad_sq = nm_total_grad_sq - left_grad_sq;
            let right_count = nm_total_count.saturating_sub(left_count);

            let candidates = [
                MissingDirectionCandidate {
                    left: ScalarSideStats {
                        grad: left_grad + missing_grad,
                        hess: left_hess + missing_hess,
                        grad_sq: left_grad_sq + missing_grad_sq,
                        count: left_count + missing_count,
                    },
                    right: ScalarSideStats {
                        grad: right_grad,
                        hess: right_hess,
                        grad_sq: right_grad_sq,
                        count: right_count,
                    },
                    default_left: true,
                },
                MissingDirectionCandidate {
                    left: ScalarSideStats {
                        grad: left_grad,
                        hess: left_hess,
                        grad_sq: left_grad_sq,
                        count: left_count,
                    },
                    right: ScalarSideStats {
                        grad: right_grad + missing_grad,
                        hess: right_hess + missing_hess,
                        grad_sq: right_grad_sq + missing_grad_sq,
                        count: right_count + missing_count,
                    },
                    default_left: false,
                },
            ];

            for candidate in candidates {
                let left = candidate.left;
                let right = candidate.right;
                if left.count == 0
                    || right.count == 0
                    || left.hess <= options.min_child_hessian
                    || right.hess <= options.min_child_hessian
                {
                    continue;
                }

                // Apply L1 thresholding uniformly before gain computation.
                let left_grad_for_gain = leaf_effective_gradient(
                    left.grad,
                    left.grad_sq,
                    left.count,
                    options.l1_alpha,
                    options.dro_config.as_ref(),
                );
                let right_grad_for_gain = leaf_effective_gradient(
                    right.grad,
                    right.grad_sq,
                    right.count,
                    options.l1_alpha,
                    options.dro_config.as_ref(),
                );
                let left_denom = left.hess + options.l2_lambda + EPSILON;
                let right_denom = right.hess + options.l2_lambda + EPSILON;

                // Apply min_leaf_magnitude filter uniformly.
                if options.min_leaf_magnitude > 0.0 {
                    let left_leaf_magnitude = left_grad_for_gain.abs() / left_denom;
                    let right_leaf_magnitude = right_grad_for_gain.abs() / right_denom;
                    if left_leaf_magnitude < options.min_leaf_magnitude
                        && right_leaf_magnitude < options.min_leaf_magnitude
                    {
                        continue;
                    }
                }

                let mut gain = match &strategy {
                    GainStrategy::Standard => {
                        split_gain_term(left.grad, left.hess, left.grad_sq, left.count, &options)
                            + split_gain_term(
                                right.grad,
                                right.hess,
                                right.grad_sq,
                                right.count,
                                &options,
                            )
                            - parent_gain_term
                    }
                    GainStrategy::Morph(morph) => {
                        use crate::morph::{MorphGainInputs, SplitSideStats, compute_morph_gain};
                        let inputs = MorphGainInputs {
                            left: SplitSideStats {
                                gradient_sum: left_grad_for_gain,
                                hessian_sum: left.hess,
                                count: left.count,
                            },
                            right: SplitSideStats {
                                gradient_sum: right_grad_for_gain,
                                hessian_sum: right.hess,
                                count: right.count,
                            },
                            iteration: morph.iteration,
                            total_iterations: morph.total_iterations.max(1),
                            grad_mean: morph.grad_mean,
                            grad_std: morph.grad_std,
                            lambda_l2: options.l2_lambda,
                        };
                        compute_morph_gain(inputs, &morph.config, &morph.precomputed)
                    }
                };

                if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_mut()) {
                    categorical_bitset_for_prefix_into(
                        num_categories,
                        &categories,
                        k,
                        &mut scratch.categorical_bitset,
                    );
                    let bitset = std::mem::take(&mut scratch.categorical_bitset);
                    let left_leaf_value = -left_grad_for_gain / left_denom;
                    let right_leaf_value = -right_grad_for_gain / right_denom;
                    gain -= factor_split_penalty_for_candidate(
                        ctx,
                        scratch,
                        feature_histogram.feature_index,
                        0,
                        candidate.default_left,
                        Some(&bitset),
                        left_leaf_value,
                        right_leaf_value,
                    );
                    scratch.categorical_bitset = bitset;
                }

                if !gain.is_finite() {
                    continue;
                }

                if gain > best_gain {
                    let bitset = if let Some(scratch) = factor_scratch.as_ref() {
                        scratch.categorical_bitset.clone()
                    } else {
                        categorical_bitset_for_prefix(num_categories, &categories, k)
                    };
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index,
                        threshold_bin: 0, // unused for categorical
                        gain,
                        default_left: candidate.default_left,
                        is_categorical: true,
                        categorical_bitset: Some(bitset),
                        left_stats: NodeStats {
                            grad_sum: left.grad,
                            hess_sum: left.hess,
                            grad_sq_sum: left.grad_sq,
                            row_count: left.count,
                        },
                        right_stats: NodeStats {
                            grad_sum: right.grad,
                            hess_sum: right.hess,
                            grad_sq_sum: right.grad_sq,
                            row_count: right.count,
                        },
                    });
                }
            }
        }

        best_candidate
    }

    const PARALLEL_SPLIT_FEATURE_THRESHOLD: usize = 16;

    fn best_split_with_options_internal(
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        let find_best = |fh: &FeatureHistogram| -> Option<SplitCandidate> {
            let fi = fh.feature_index as usize;
            if let Some(cat_info) = categorical_features.iter().find(|c| c.feature_index == fi) {
                Self::best_split_for_categorical_feature(
                    fh,
                    histograms.node_id,
                    options,
                    cat_info.num_categories,
                    factor_context,
                )
            } else {
                Self::best_split_for_feature(fh, histograms.node_id, options, factor_context)
            }
        };

        if histograms.feature_histograms.len() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
            histograms
                .feature_histograms
                .par_iter()
                .filter_map(&find_best)
                .reduce_with(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        } else {
            histograms
                .feature_histograms
                .iter()
                .filter_map(&find_best)
                .reduce(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        }
    }

    /// Morph-mode best split for a single numeric feature.
    ///
    /// Thin wrapper around `best_split_for_feature_inner` with `GainStrategy::Morph`.
    fn best_split_morph_numeric_feature(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: &SplitSelectionOptions,
        morph: &MorphContext,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        Self::best_split_for_feature_inner(
            feature_histogram,
            node_id,
            *options,
            GainStrategy::Morph(morph),
            factor_context,
        )
    }

    fn apply_split_with_stats_parallel(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        type ChunkResult = (Vec<u32>, Vec<u32>, f32, f32, f32, f32, f32, f32);
        let chunk_size = (node.row_indices.len() / rayon::current_num_threads().max(1)).max(4096);
        let chunk_results: Vec<ChunkResult> = node
            .row_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut left = Vec::new();
                let mut right = Vec::new();
                let mut lg = 0.0_f32;
                let mut lh = 0.0_f32;
                let mut lq = 0.0_f32;
                let mut rg = 0.0_f32;
                let mut rh = 0.0_f32;
                let mut rq = 0.0_f32;
                let feature_index = split.feature_index as usize;
                let use_col_major = binned_matrix.has_col_major();
                let col_base = feature_index * binned_matrix.row_count;
                let missing = binned_matrix.missing_bin();
                for &row_index_u32 in chunk {
                    let row_index = row_index_u32 as usize;
                    let bin_val = if use_col_major {
                        binned_matrix.col_bin(col_base + row_index)
                    } else {
                        let cell_index = row_index * binned_matrix.feature_count + feature_index;
                        binned_matrix.row_bin(cell_index)
                    };
                    let gradient = gradients[row_index];
                    if goes_left_for_split(bin_val, missing, split) {
                        left.push(row_index_u32);
                        lg += gradient.grad;
                        lh += gradient.hess;
                        lq += gradient.grad * gradient.grad;
                    } else {
                        right.push(row_index_u32);
                        rg += gradient.grad;
                        rh += gradient.hess;
                        rq += gradient.grad * gradient.grad;
                    }
                }
                (left, right, lg, lh, lq, rg, rh, rq)
            })
            .collect();

        let total_rows = node.row_indices.len();
        let mut left_row_indices = Vec::with_capacity(total_rows / 2);
        let mut right_row_indices = Vec::with_capacity(total_rows / 2);
        let mut left_grad_sum = 0.0_f32;
        let mut left_hess_sum = 0.0_f32;
        let mut left_grad_sq_sum = 0.0_f32;
        let mut right_grad_sum = 0.0_f32;
        let mut right_hess_sum = 0.0_f32;
        let mut right_grad_sq_sum = 0.0_f32;

        for (left, right, lg, lh, lq, rg, rh, rq) in chunk_results {
            left_row_indices.extend(left);
            right_row_indices.extend(right);
            left_grad_sum += lg;
            left_hess_sum += lh;
            left_grad_sq_sum += lq;
            right_grad_sum += rg;
            right_hess_sum += rh;
            right_grad_sq_sum += rq;
        }

        let left_count = left_row_indices.len() as u32;
        let right_count = right_row_indices.len() as u32;
        Ok((
            PartitionResult {
                left_row_indices,
                right_row_indices,
            },
            NodeStats {
                grad_sum: left_grad_sum,
                hess_sum: left_hess_sum,
                grad_sq_sum: left_grad_sq_sum,
                row_count: left_count,
            },
            NodeStats {
                grad_sum: right_grad_sum,
                hess_sum: right_hess_sum,
                grad_sq_sum: right_grad_sq_sum,
                row_count: right_count,
            },
        ))
    }
}

fn l1_threshold_gradient(grad_sum: f32, l1_alpha: f32) -> f32 {
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

fn split_gain_term(
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

fn categorical_bitset_for_prefix(
    num_categories: usize,
    categories: &[(u16, f32, f32, f32, u32)],
    prefix_end: usize,
) -> Vec<u8> {
    let bitset_len = num_categories.div_ceil(8);
    let mut bitset = vec![0u8; bitset_len];
    categorical_bitset_for_prefix_into(num_categories, categories, prefix_end, &mut bitset);
    bitset
}

fn categorical_bitset_for_prefix_into(
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

struct FactorSplitScratch {
    left_factor_sums: Vec<f32>,
    right_factor_sums: Vec<f32>,
    categorical_bitset: Vec<u8>,
}

impl FactorSplitScratch {
    fn new(factor_count: usize) -> Self {
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

fn factor_split_penalty_for_candidate(
    context: &FactorSplitContext<'_>,
    scratch: &mut FactorSplitScratch,
    feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
    categorical_bitset: Option<&[u8]>,
    left_leaf_value: f32,
    right_leaf_value: f32,
) -> f32 {
    if context.factor_penalty == 0.0 {
        return 0.0;
    }

    let factor_count = context.exposures.factor_count;
    scratch.clear_factor_sums();
    let feature_index = feature_index as usize;
    let feature_count = context.binned_matrix.feature_count;
    let missing = context.binned_matrix.missing_bin();

    for &row_index in context.row_indices {
        let row_index = row_index as usize;
        let bin = context
            .binned_matrix
            .row_bin(row_index * feature_count + feature_index);
        let goes_left = if bin == missing {
            default_left
        } else if let Some(bitset) = categorical_bitset {
            let byte_idx = (bin / 8) as usize;
            let bit_idx = (bin % 8) as usize;
            byte_idx < bitset.len() && (bitset[byte_idx] & (1 << bit_idx)) != 0
        } else {
            bin <= threshold_bin
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
        left_leaf_value,
        right_leaf_value,
        context.factor_penalty,
        context.row_indices.len(),
    )
}

fn factor_split_penalty(
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

fn validate_factor_split_context(context: &FactorSplitContext<'_>) -> EngineResult<()> {
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

/// Determine if a row goes to the left child for a given split.
/// Handles both continuous (threshold comparison) and categorical (bitset membership) splits.
#[inline]
fn goes_left_for_split(bin_val: u16, missing: u16, split: &SplitCandidate) -> bool {
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

impl BackendOps for CpuBackend {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        let selected_feature_count = feature_tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum();
        let parallel_tiles = Self::should_parallelize_tiles(
            feature_tiles.len(),
            node.row_indices.len(),
            selected_feature_count,
        );
        Self::build_histograms_internal(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            parallel_tiles,
        )
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        Ok(Self::best_split_with_options_internal(
            histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
            None,
        ))
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        Ok(Self::best_split_with_options_internal(
            histograms,
            options,
            feature_weights,
            categorical_features,
            None,
        ))
    }

    fn best_split_with_factor_context(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> EngineResult<Option<SplitCandidate>> {
        if let Some(ctx) = factor_context {
            validate_factor_split_context(ctx)?;
        }
        Ok(Self::best_split_with_options_internal(
            histograms,
            options,
            feature_weights,
            categorical_features,
            factor_context,
        ))
    }

    fn best_split_morph(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        morph: &MorphContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        self.best_split_morph_with_factor_context(
            histograms,
            options,
            feature_weights,
            categorical_features,
            morph,
            None,
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
        if let Some(ctx) = factor_context {
            validate_factor_split_context(ctx)?;
        }
        let find_best = |fh: &FeatureHistogram| -> Option<SplitCandidate> {
            let fi = fh.feature_index as usize;
            if let Some(cat_info) = categorical_features.iter().find(|c| c.feature_index == fi) {
                Self::best_split_morph_categorical_feature(
                    fh,
                    histograms.node_id,
                    &options,
                    cat_info.num_categories,
                    morph,
                    factor_context,
                )
            } else {
                Self::best_split_morph_numeric_feature(
                    fh,
                    histograms.node_id,
                    &options,
                    morph,
                    factor_context,
                )
            }
        };

        let result =
            if histograms.feature_histograms.len() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
                histograms
                    .feature_histograms
                    .par_iter()
                    .filter_map(find_best)
                    .reduce_with(|a, b| {
                        if apply_feature_weight(&b, feature_weights)
                            > apply_feature_weight(&a, feature_weights)
                        {
                            b
                        } else {
                            a
                        }
                    })
            } else {
                histograms
                    .feature_histograms
                    .iter()
                    .filter_map(find_best)
                    .reduce(|a, b| {
                        if apply_feature_weight(&b, feature_weights)
                            > apply_feature_weight(&a, feature_weights)
                        {
                            b
                        } else {
                            a
                        }
                    })
            };

        Ok(result)
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        node.validate_bounds(binned_matrix.row_count)?;
        if split.feature_index as usize >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                split.feature_index, binned_matrix.feature_count
            )));
        }

        let mut left_row_indices = Vec::new();
        let mut right_row_indices = Vec::new();
        let feature_index = split.feature_index as usize;
        let use_col_major = binned_matrix.has_col_major();
        let col_base = feature_index * binned_matrix.row_count;
        let missing = binned_matrix.missing_bin();
        for &row_index in &node.row_indices {
            let row_index = row_index as usize;
            let bin_val = if use_col_major {
                binned_matrix.col_bin(col_base + row_index)
            } else {
                let cell_index = row_index * binned_matrix.feature_count + feature_index;
                binned_matrix.row_bin(cell_index)
            };
            if goes_left_for_split(bin_val, missing, split) {
                left_row_indices.push(row_index as u32);
            } else {
                right_row_indices.push(row_index as u32);
            }
        }

        Ok(PartitionResult {
            left_row_indices,
            right_row_indices,
        })
    }

    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        node.validate_bounds(binned_matrix.row_count)?;
        if gradients.len() != binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "gradients length {} does not match row_count {}",
                gradients.len(),
                binned_matrix.row_count
            )));
        }
        if split.feature_index as usize >= binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "split feature_index {} exceeds feature_count {}",
                split.feature_index, binned_matrix.feature_count
            )));
        }

        const PARALLEL_PARTITION_THRESHOLD: usize = 50_000;

        if node.row_indices.len() >= PARALLEL_PARTITION_THRESHOLD {
            return Self::apply_split_with_stats_parallel(binned_matrix, gradients, node, split);
        }

        let mut left_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut right_row_indices = Vec::with_capacity(node.row_indices.len() / 2);
        let mut left_grad_sum = 0.0_f32;
        let mut left_hess_sum = 0.0_f32;
        let mut left_grad_sq_sum = 0.0_f32;
        let mut right_grad_sum = 0.0_f32;
        let mut right_hess_sum = 0.0_f32;
        let mut right_grad_sq_sum = 0.0_f32;

        let feature_index = split.feature_index as usize;
        let use_col_major = binned_matrix.has_col_major();
        let col_base = feature_index * binned_matrix.row_count;
        let missing = binned_matrix.missing_bin();
        for &row_index_u32 in &node.row_indices {
            let row_index = row_index_u32 as usize;
            let bin_val = if use_col_major {
                binned_matrix.col_bin(col_base + row_index)
            } else {
                let cell_index = row_index * binned_matrix.feature_count + feature_index;
                binned_matrix.row_bin(cell_index)
            };
            let gradient = gradients[row_index];
            if goes_left_for_split(bin_val, missing, split) {
                left_row_indices.push(row_index_u32);
                left_grad_sum += gradient.grad;
                left_hess_sum += gradient.hess;
                left_grad_sq_sum += gradient.grad * gradient.grad;
            } else {
                right_row_indices.push(row_index_u32);
                right_grad_sum += gradient.grad;
                right_hess_sum += gradient.hess;
                right_grad_sq_sum += gradient.grad * gradient.grad;
            }
        }

        let partition = PartitionResult {
            left_row_indices,
            right_row_indices,
        };
        let left_stats = NodeStats {
            grad_sum: left_grad_sum,
            hess_sum: left_hess_sum,
            grad_sq_sum: left_grad_sq_sum,
            row_count: partition.left_row_indices.len() as u32,
        };
        let right_stats = NodeStats {
            grad_sum: right_grad_sum,
            hess_sum: right_hess_sum,
            grad_sq_sum: right_grad_sq_sum,
            row_count: partition.right_row_indices.len() as u32,
        };

        Ok((partition, left_stats, right_stats))
    }

    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats> {
        if row_indices.is_empty() {
            return Err(EngineError::ContractViolation(
                "row_indices cannot be empty".to_string(),
            ));
        }

        let mut grad_sum = 0.0_f32;
        let mut hess_sum = 0.0_f32;
        let mut grad_sq_sum = 0.0_f32;
        for &row_index in row_indices {
            let gradient = gradients.get(row_index as usize).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "row index {row_index} is out of bounds for gradients length {}",
                    gradients.len()
                ))
            })?;
            grad_sum += gradient.grad;
            hess_sum += gradient.hess;
            grad_sq_sum += gradient.grad * gradient.grad;
        }

        Ok(NodeStats {
            grad_sum,
            hess_sum,
            grad_sq_sum,
            row_count: row_indices.len() as u32,
        })
    }

    fn build_linear_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        regressor_features: &[u32],
        raw_feature_values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> EngineResult<LinearHistogramBundle> {
        pl_histogram::build_linear_histograms_cpu(
            binned_matrix,
            gradients,
            node,
            feature_tiles,
            regressor_features,
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
        _categorical_features: &[CategoricalFeatureInfo],
        ctx: &LinearContext,
    ) -> EngineResult<Option<SplitCandidate>> {
        let node_id = linear_histograms.node_id;
        let find_best = |linear_fh: &LinearFeatureHistogram| -> Option<SplitCandidate> {
            pl::best_split_linear_for_feature(linear_fh, node_id, options, ctx)
        };

        let result = if linear_histograms.feature_histograms.len()
            >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD
        {
            linear_histograms
                .feature_histograms
                .par_iter()
                .filter_map(find_best)
                .reduce_with(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        } else {
            linear_histograms
                .feature_histograms
                .iter()
                .filter_map(find_best)
                .reduce(|a, b| {
                    if apply_feature_weight(&b, feature_weights)
                        > apply_feature_weight(&a, feature_weights)
                    {
                        b
                    } else {
                        a
                    }
                })
        };

        Ok(result)
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
    ) -> Option<(LinearLeaf, LinearLeaf)> {
        let d = linear_histograms.num_regressors;
        if d == 0 {
            return None;
        }
        let lin_fh = linear_histograms
            .feature_histograms
            .iter()
            .find(|fh| fh.feature_index == feature_index)?;

        let (l_xtg, l_xthx, l_gs, l_hs, r_xtg, r_xthx, r_gs, r_hs) =
            pl::leaf_linear_stats_for_split(lin_fh, threshold_bin, missing_bin_index, default_left);

        let regressor_features = &linear_histograms.regressor_features;
        let left_leaf = pl::solve_pl_leaf(
            &l_xtg,
            &l_xthx,
            l_gs,
            l_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        );
        let right_leaf = pl::solve_pl_leaf(
            &r_xtg,
            &r_xthx,
            r_gs,
            r_hs,
            learning_rate,
            l2_lambda,
            regressor_features,
        );

        Some((left_leaf, right_leaf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{
        DatasetMatrix, FactorExposureMatrix, FeatureTile, LeafModelKind, TrainParams,
        TrainingDataset, TreeGrowth,
    };
    use alloygbm_engine::{FactorSplitContext, SquaredErrorObjective, Trainer};

    fn sample_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            4,
            2,
            3,
            vec![
                0, 0, //
                1, 0, //
                2, 1, //
                3, 1, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn quality_fixture_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    4.0, 0.0, //
                    5.0, 0.0, //
                    6.0, 0.0, //
                    7.0, 0.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn quality_fixture_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            8,
            2,
            7,
            vec![
                0, 0, //
                1, 0, //
                2, 0, //
                3, 0, //
                4, 0, //
                5, 0, //
                6, 0, //
                7, 0, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn fixture_rows(dataset: &TrainingDataset) -> Vec<Vec<f32>> {
        dataset
            .matrix
            .values
            .chunks(dataset.matrix.feature_count)
            .map(|row| row.to_vec())
            .collect()
    }

    fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
        let error_sum = predictions
            .iter()
            .zip(targets)
            .map(|(prediction, target)| {
                let error = prediction - target;
                error * error
            })
            .sum::<f32>();
        error_sum / predictions.len() as f32
    }

    fn fixture_params() -> TrainParams {
        TrainParams {
            seed: 7,
            deterministic: true,
            learning_rate: 0.3,
            max_depth: 6,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
            monotone_constraints: Vec::new(),
            feature_weights: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
            leaf_model: LeafModelKind::Constant,
            leaf_solver: alloygbm_core::LeafSolverKind::Standard,
            dro_config: None,
            neutralization_config: None,
        }
    }

    fn sample_gradients() -> Vec<GradientPair> {
        vec![
            GradientPair {
                grad: 2.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -2.0,
                hess: 1.0,
            },
        ]
    }

    fn sample_node() -> NodeSlice {
        NodeSlice::new(0, vec![0, 1, 2, 3]).expect("node is valid")
    }

    #[test]
    fn build_histograms_aggregates_bins() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        assert_eq!(histograms.feature_histograms.len(), 2);
        let feature0 = &histograms.feature_histograms[0];
        assert_eq!(feature0.feature_index, 0);
        assert_eq!(feature0.bins.len(), 4);
        assert_eq!(feature0.bins[0].count, 1);
        assert_eq!(feature0.bins[1].count, 1);
        assert_eq!(feature0.bins[2].count, 1);
        assert_eq!(feature0.bins[3].count, 1);
        assert!((feature0.bins[0].grad_sum - 2.0).abs() < 1e-6);
        assert!((feature0.bins[3].grad_sum + 2.0).abs() < 1e-6);
    }

    #[test]
    fn build_histograms_is_tile_partition_invariant() {
        let backend = CpuBackend;
        let matrix = sample_binned_matrix();
        let gradients = sample_gradients();
        let node = sample_node();

        let single_tile = backend
            .build_histograms(
                &matrix,
                &gradients,
                &node,
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("single-tile histograms should build");
        let split_tiles = backend
            .build_histograms(
                &matrix,
                &gradients,
                &node,
                &[
                    FeatureTile::new(0, 1).expect("feature tile is valid"),
                    FeatureTile::new(1, 2).expect("feature tile is valid"),
                ],
            )
            .expect("split-tile histograms should build");

        assert_eq!(single_tile, split_tiles);
        assert_eq!(
            backend
                .best_split(&single_tile)
                .expect("single-tile split should succeed"),
            backend
                .best_split(&split_tiles)
                .expect("split-tile split should succeed")
        );
    }

    #[test]
    fn histogram_tile_strategies_are_equivalent() {
        let matrix = sample_binned_matrix();
        let gradients = sample_gradients();
        let node = sample_node();
        let bin_count = matrix.max_bin as usize + 1;

        let mut per_feature = Vec::new();
        CpuBackend::build_tile_histograms_per_feature(
            &matrix,
            &gradients,
            &node,
            0,
            2,
            bin_count,
            &mut per_feature,
        );

        let mut row_first = Vec::new();
        let mut arena = HistogramArena::new(2, bin_count);
        CpuBackend::build_tile_histograms_row_first(&matrix, &gradients, &node, 0, 2, &mut arena);
        arena.materialize(0, &mut row_first);

        assert_eq!(per_feature, row_first);
    }

    #[test]
    fn histogram_kernel_path_prefers_tiny_node_scalar_for_small_nodes() {
        let path = CpuBackend::select_histogram_kernel_path(8, SMALL_TILE_WORKLOAD_THRESHOLD, 16);
        assert_eq!(path, HistogramKernelPath::TinyNodeScalar);
    }

    #[test]
    fn histogram_kernel_path_prefers_unrolled_for_large_tiles() {
        let path =
            CpuBackend::select_histogram_kernel_path(256, SMALL_TILE_WORKLOAD_THRESHOLD + 1, 64);
        assert_eq!(path, HistogramKernelPath::ArenaRowFirstUnrolled);
    }

    #[test]
    fn histogram_kernel_path_prefers_bin_heavy_fallback_for_wide_bins() {
        let path = CpuBackend::select_histogram_kernel_path(
            512,
            SMALL_TILE_WORKLOAD_THRESHOLD + 1,
            BIN_HEAVY_THRESHOLD,
        );
        assert_eq!(path, HistogramKernelPath::BinHeavyPerFeatureScalar);
    }

    #[test]
    fn tile_parallelization_policy_requires_sufficient_workload() {
        assert!(!CpuBackend::should_parallelize_tiles(1, 4096, 128));
        assert!(!CpuBackend::should_parallelize_tiles(4, 128, 8));

        let expected = rayon::current_num_threads() > 1;
        assert_eq!(CpuBackend::should_parallelize_tiles(4, 4096, 128), expected);
    }

    #[test]
    fn build_histograms_parallel_tiles_match_sequential() {
        let backend = CpuBackend;
        let matrix = quality_fixture_binned_matrix();
        let gradients = (0..matrix.row_count)
            .map(|row_index| {
                let grad = (row_index as f32 % 23.0) - 11.0;
                let hess = 1.0 + (row_index as f32 % 5.0) * 0.1;
                GradientPair::new(grad, hess).expect("gradient pair should be valid")
            })
            .collect::<Vec<_>>();
        let node = NodeSlice::new(0, (0..matrix.row_count as u32).collect())
            .expect("node should be valid");
        let feature_tiles = vec![
            FeatureTile::new(0, 1).expect("feature tile should be valid"),
            FeatureTile::new(1, 2).expect("feature tile should be valid"),
        ];

        let sequential = CpuBackend::build_histograms_internal(
            &matrix,
            &gradients,
            &node,
            &feature_tiles,
            false,
        )
        .expect("sequential histograms should build");
        let parallel =
            CpuBackend::build_histograms_internal(&matrix, &gradients, &node, &feature_tiles, true)
                .expect("parallel histograms should build");

        assert_eq!(sequential, parallel);
        assert_eq!(
            backend
                .best_split(&sequential)
                .expect("sequential split should succeed"),
            backend
                .best_split(&parallel)
                .expect("parallel split should succeed")
        );
    }

    #[test]
    fn unrolled_row_first_histograms_match_per_feature() {
        let matrix = quality_fixture_binned_matrix();
        let gradients = (0..matrix.row_count)
            .map(|row_index| {
                GradientPair::new((row_index as f32 - 3.5) * 0.5, 1.0 + row_index as f32 * 0.1)
                    .expect("gradient pair is finite")
            })
            .collect::<Vec<_>>();
        let node = NodeSlice::new(0, (0..matrix.row_count as u32).collect())
            .expect("node indices are valid");
        let bin_count = matrix.max_bin as usize + 1;

        let mut per_feature = Vec::new();
        CpuBackend::build_tile_histograms_per_feature(
            &matrix,
            &gradients,
            &node,
            0,
            matrix.feature_count,
            bin_count,
            &mut per_feature,
        );

        let mut unrolled = Vec::new();
        let mut unrolled_arena = HistogramArena::new(matrix.feature_count, bin_count);
        CpuBackend::build_tile_histograms_row_first_unrolled(
            &matrix,
            &gradients,
            &node,
            0,
            matrix.feature_count,
            &mut unrolled_arena,
        );
        unrolled_arena.materialize(0, &mut unrolled);

        assert_eq!(per_feature, unrolled);
    }

    #[test]
    fn best_split_returns_high_gain_candidate() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");
        let split = backend
            .best_split(&histograms)
            .expect("split search should succeed")
            .expect("split should exist");

        assert_eq!(split.feature_index, 0);
        assert_eq!(split.threshold_bin, 1);
        assert!(split.gain > 0.0);
        assert_eq!(split.left_stats.row_count, 2);
        assert_eq!(split.right_stats.row_count, 2);
    }

    #[test]
    fn best_split_with_l2_regularization_reduces_gain_magnitude() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let unregularized = backend
            .best_split(&histograms)
            .expect("unregularized split search should succeed")
            .expect("unregularized split should exist");
        let regularized = backend
            .best_split_with_options(
                &histograms,
                SplitSelectionOptions {
                    l2_lambda: 1.0,
                    l1_alpha: 0.0,
                    min_child_hessian: 0.0,
                    min_leaf_magnitude: 0.0,
                    dro_config: None,
                    missing_bin_index: 255,
                },
                &[],
                &[],
            )
            .expect("regularized split search should succeed")
            .expect("regularized split should exist");

        assert_eq!(unregularized.feature_index, regularized.feature_index);
        assert_eq!(unregularized.threshold_bin, regularized.threshold_bin);
        assert!(regularized.gain < unregularized.gain);
    }

    #[test]
    fn best_split_with_l1_regularization_reduces_gain_magnitude() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let unregularized = backend
            .best_split(&histograms)
            .expect("unregularized split search should succeed")
            .expect("unregularized split should exist");
        let regularized = backend
            .best_split_with_options(
                &histograms,
                SplitSelectionOptions {
                    l2_lambda: 0.0,
                    l1_alpha: 0.5,
                    min_child_hessian: 0.0,
                    min_leaf_magnitude: 0.0,
                    dro_config: None,
                    missing_bin_index: 255,
                },
                &[],
                &[],
            )
            .expect("regularized split search should succeed")
            .expect("regularized split should exist");

        assert_eq!(unregularized.feature_index, regularized.feature_index);
        assert_eq!(unregularized.threshold_bin, regularized.threshold_bin);
        assert!(regularized.gain < unregularized.gain);
    }

    #[test]
    fn factor_split_penalty_reduces_factor_loaded_gain() {
        let backend = CpuBackend;
        let matrix = sample_binned_matrix();
        let node = sample_node();
        let histograms = backend
            .build_histograms(
                &matrix,
                &sample_gradients(),
                &node,
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");
        let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0])
            .expect("factor exposures are valid");
        let no_penalty = backend
            .best_split_with_options(&histograms, SplitSelectionOptions::default(), &[], &[])
            .expect("split search should succeed")
            .expect("split should exist");
        let factor_context = FactorSplitContext {
            binned_matrix: &matrix,
            exposures: &exposures,
            row_indices: &node.row_indices,
            factor_penalty: 0.1,
        };
        let penalized = backend
            .best_split_with_factor_context(
                &histograms,
                SplitSelectionOptions::default(),
                &[],
                &[],
                Some(&factor_context),
            )
            .expect("split search should succeed")
            .expect("split should exist");
        assert!(penalized.gain <= no_penalty.gain);
    }

    #[test]
    fn factor_split_penalty_formula_matches_expected() {
        let left_factor_sums = [3.0_f32, -1.0];
        let right_factor_sums = [-2.0_f32, 4.0];
        let penalty =
            factor_split_penalty(&left_factor_sums, &right_factor_sums, 0.5, -0.25, 2.0, 5);

        let load0 = 3.0 * 0.5 + -2.0 * -0.25;
        let load1 = -1.0 * 0.5 + 4.0 * -0.25;
        let expected = 2.0 * (load0 * load0 + load1 * load1) / 5.0;
        assert!((penalty - expected).abs() < 1e-6);
    }

    #[test]
    fn factor_split_penalty_rejects_malformed_factor_context() {
        let backend = CpuBackend;
        let matrix = sample_binned_matrix();
        let node = sample_node();
        let histograms = backend
            .build_histograms(
                &matrix,
                &sample_gradients(),
                &node,
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");
        let cases = [
            (
                FactorExposureMatrix {
                    row_count: 4,
                    factor_count: 0,
                    values: Vec::new(),
                },
                "factor_exposures factor_count must be greater than 0",
            ),
            (
                FactorExposureMatrix {
                    row_count: 4,
                    factor_count: 1,
                    values: vec![1.0, 1.0, -1.0],
                },
                "factor_exposures values length 3 does not match row_count * factor_count 4",
            ),
            (
                FactorExposureMatrix {
                    row_count: 4,
                    factor_count: 1,
                    values: vec![1.0, f32::NAN, -1.0, -1.0],
                },
                "factor_exposures must contain only finite values",
            ),
        ];

        for (malformed, expected_message) in cases {
            let factor_context = FactorSplitContext {
                binned_matrix: &matrix,
                exposures: &malformed,
                row_indices: &node.row_indices,
                factor_penalty: 0.1,
            };

            let err = backend
                .best_split_with_factor_context(
                    &histograms,
                    SplitSelectionOptions::default(),
                    &[],
                    &[],
                    Some(&factor_context),
                )
                .expect_err("malformed factor context should be rejected");
            assert!(matches!(err, EngineError::ContractViolation(_)));
            assert!(
                err.to_string().contains(expected_message),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn best_split_with_min_child_hessian_can_prune_all_splits() {
        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let split = backend
            .best_split_with_options(
                &histograms,
                SplitSelectionOptions {
                    l2_lambda: 0.0,
                    l1_alpha: 0.0,
                    min_child_hessian: 10.0,
                    min_leaf_magnitude: 0.0,
                    dro_config: None,
                    missing_bin_index: 255,
                },
                &[],
                &[],
            )
            .expect("split search should succeed");

        assert!(split.is_none());
    }

    #[test]
    fn best_split_with_min_leaf_magnitude_skips_weak_leaf_updates() {
        let backend = CpuBackend;
        let histograms = HistogramBundle {
            node_id: 0,
            feature_histograms: vec![
                FeatureHistogram {
                    feature_index: 0,
                    bins: vec![
                        HistogramBin {
                            grad_sum: 1.0,
                            hess_sum: 20.0,
                            grad_sq_sum: 0.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: -1.0,
                            hess_sum: 20.0,
                            grad_sq_sum: 0.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
                            grad_sq_sum: 0.0,
                            count: 0,
                        },
                    ],
                },
                FeatureHistogram {
                    feature_index: 1,
                    bins: vec![
                        HistogramBin {
                            grad_sum: 0.5,
                            hess_sum: 5.0,
                            grad_sq_sum: 0.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: -0.5,
                            hess_sum: 5.0,
                            grad_sq_sum: 0.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
                            grad_sq_sum: 0.0,
                            count: 0,
                        },
                    ],
                },
            ],
        };

        let unfiltered = backend
            .best_split(&histograms)
            .expect("default split search should succeed")
            .expect("default split should exist");
        let filtered = backend
            .best_split_with_options(
                &histograms,
                SplitSelectionOptions {
                    l2_lambda: 0.0,
                    l1_alpha: 0.0,
                    min_child_hessian: 0.0,
                    min_leaf_magnitude: 0.06,
                    dro_config: None,
                    missing_bin_index: 255,
                },
                &[],
                &[],
            )
            .expect("magnitude-filtered split search should succeed")
            .expect("magnitude-filtered split should exist");

        assert_eq!(unfiltered.feature_index, 0);
        assert_eq!(filtered.feature_index, 1);
        assert!(filtered.gain > 0.0);
    }

    #[test]
    fn apply_split_partitions_rows() {
        let backend = CpuBackend;
        let split = SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin: 1,
            gain: 1.0,
            default_left: false,
            is_categorical: false,
            categorical_bitset: None,
            left_stats: NodeStats {
                grad_sum: 3.0,
                hess_sum: 2.0,
                grad_sq_sum: 0.0,
                row_count: 2,
            },
            right_stats: NodeStats {
                grad_sum: -3.0,
                hess_sum: 2.0,
                grad_sq_sum: 0.0,
                row_count: 2,
            },
        };
        let partition = backend
            .apply_split(&sample_binned_matrix(), &sample_node(), &split)
            .expect("partition should succeed");

        assert_eq!(partition.left_row_indices, vec![0, 1]);
        assert_eq!(partition.right_row_indices, vec![2, 3]);
    }

    #[test]
    fn apply_split_with_stats_matches_partition_and_reduction_reference() {
        let backend = CpuBackend;
        let matrix = sample_binned_matrix();
        let node = sample_node();
        let gradients = sample_gradients();
        let split = SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin: 1,
            gain: 1.0,
            default_left: false,
            is_categorical: false,
            categorical_bitset: None,
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                row_count: 0,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                row_count: 0,
            },
        };

        let (partition, left_stats, right_stats) = backend
            .apply_split_with_stats(&matrix, &gradients, &node, &split)
            .expect("fused split should succeed");
        let reference_partition = backend
            .apply_split(&matrix, &node, &split)
            .expect("reference split should succeed");
        let reference_left = backend
            .reduce_sums(&gradients, &reference_partition.left_row_indices)
            .expect("reference left reduction should succeed");
        let reference_right = backend
            .reduce_sums(&gradients, &reference_partition.right_row_indices)
            .expect("reference right reduction should succeed");

        assert_eq!(partition, reference_partition);
        assert_eq!(left_stats, reference_left);
        assert_eq!(right_stats, reference_right);
    }

    #[test]
    fn reduce_sums_aggregates_requested_rows() {
        let backend = CpuBackend;
        let stats = backend
            .reduce_sums(&sample_gradients(), &[0, 3])
            .expect("reductions should succeed");
        assert_eq!(stats.row_count, 2);
        assert!(stats.grad_sum.abs() < 1e-6);
        assert!((stats.hess_sum - 2.0).abs() < 1e-6);
    }

    #[test]
    fn backend_reports_cpu_device() {
        assert_eq!(CpuBackend.device(), Device::Cpu);
    }

    #[test]
    fn cpu_backend_training_beats_naive_baseline_mse() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
            .expect("training succeeds");

        assert!(!model.stumps.is_empty());

        let rows = fixture_rows(&dataset);
        let model_predictions = model.predict_batch(&rows).expect("predictions succeed");
        let baseline_prediction =
            dataset.targets.iter().sum::<f32>() / dataset.targets.len() as f32;
        let baseline_predictions = vec![baseline_prediction; dataset.targets.len()];

        let model_mse = mean_squared_error(&model_predictions, &dataset.targets);
        let baseline_mse = mean_squared_error(&baseline_predictions, &dataset.targets);
        assert!(model_mse < baseline_mse);
    }

    #[test]
    fn cpu_backend_deterministic_training_has_stable_artifact_bytes() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model_a = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
            .expect("first training succeeds");
        let model_b = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 6)
            .expect("second training succeeds");

        let bytes_a = model_a.to_artifact_bytes().expect("artifact serializes");
        let bytes_b = model_b.to_artifact_bytes().expect("artifact serializes");
        assert_eq!(bytes_a, bytes_b);
    }

    // ── Native categorical split tests ──────────────────────────────────

    #[test]
    fn test_best_split_categorical_basic() {
        // 3-category feature (bins 0,1,2) + NaN bin (bin 255)
        // Category 0: positive grad, category 1: positive grad, category 2: negative grad
        // Optimal split: categories 0,1 go left, category 2 goes right (or vice versa)
        let num_cats = 3;
        let nan_bin = 255usize;
        let num_bins = nan_bin + 1;
        let mut bins = vec![
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            };
            num_bins
        ];
        // Category 0: grad=-2.0, hess=2.0 (score = -2/2 = -1.0)
        bins[0] = HistogramBin {
            grad_sum: -2.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 10,
        };
        // Category 1: grad=-1.5, hess=2.0 (score = -1.5/2 = -0.75)
        bins[1] = HistogramBin {
            grad_sum: -1.5,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 10,
        };
        // Category 2: grad=3.5, hess=2.0 (score = 3.5/2 = 1.75)
        bins[2] = HistogramBin {
            grad_sum: 3.5,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 10,
        };
        // NaN bin: no data
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };

        let options = SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: nan_bin,
        };

        let result =
            CpuBackend::best_split_for_categorical_feature(&fh, 0, options, num_cats, None);
        assert!(result.is_some(), "should find a split");
        let split = result.unwrap();
        assert!(split.is_categorical);
        assert!(split.categorical_bitset.is_some());
        assert!(split.gain > 0.0, "gain should be positive");

        // Verify bitset: categories 0 and 1 should be on one side, category 2 on the other
        let bitset = split.categorical_bitset.as_ref().unwrap();
        let cat0_left = bitset[0] & (1 << 0) != 0;
        let cat1_left = bitset[0] & (1 << 1) != 0;
        let cat2_left = bitset[0] & (1 << 2) != 0;
        // Categories 0,1 have similar scores and should be grouped together
        assert_eq!(cat0_left, cat1_left, "cats 0 and 1 should be on same side");
        assert_ne!(cat0_left, cat2_left, "cat 2 should be on opposite side");
    }

    #[test]
    fn dro_categorical_split_stats_match_direct_scan() {
        let num_cats = 4usize;
        let nan_bin = 15usize;
        let mut bins = vec![
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            };
            nan_bin + 1
        ];
        bins[0] = HistogramBin {
            grad_sum: -4.0,
            hess_sum: 3.0,
            grad_sq_sum: 7.0,
            count: 6,
        };
        bins[1] = HistogramBin {
            grad_sum: -2.0,
            hess_sum: 2.0,
            grad_sq_sum: 3.0,
            count: 4,
        };
        bins[2] = HistogramBin {
            grad_sum: 3.0,
            hess_sum: 2.5,
            grad_sq_sum: 5.0,
            count: 5,
        };
        bins[3] = HistogramBin {
            grad_sum: 4.0,
            hess_sum: 3.5,
            grad_sq_sum: 8.0,
            count: 7,
        };
        bins[nan_bin] = HistogramBin {
            grad_sum: 0.75,
            hess_sum: 1.0,
            grad_sq_sum: 0.75,
            count: 2,
        };
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = SplitSelectionOptions {
            dro_config: Some(alloygbm_core::DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            missing_bin_index: nan_bin,
            ..SplitSelectionOptions::default()
        };

        let split = CpuBackend::best_split_for_categorical_feature(&fh, 0, options, num_cats, None)
            .expect("dro categorical split should exist");
        let bitset = split
            .categorical_bitset
            .as_ref()
            .expect("categorical split has bitset");
        let mut expected_left = HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        let mut expected_right = HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        for (bin_id, bin) in fh.bins.iter().enumerate() {
            if bin.count == 0 {
                continue;
            }
            let goes_left = if bin_id == nan_bin {
                split.default_left
            } else if bin_id < num_cats {
                bitset[bin_id / 8] & (1 << (bin_id % 8)) != 0
            } else {
                continue;
            };
            let target = if goes_left {
                &mut expected_left
            } else {
                &mut expected_right
            };
            target.grad_sum += bin.grad_sum;
            target.hess_sum += bin.hess_sum;
            target.grad_sq_sum += bin.grad_sq_sum;
            target.count += bin.count;
        }

        assert!((split.left_stats.grad_sum - expected_left.grad_sum).abs() < 1e-6);
        assert!((split.left_stats.hess_sum - expected_left.hess_sum).abs() < 1e-6);
        assert!((split.left_stats.grad_sq_sum - expected_left.grad_sq_sum).abs() < 1e-6);
        assert_eq!(split.left_stats.row_count, expected_left.count);
        assert!((split.right_stats.grad_sum - expected_right.grad_sum).abs() < 1e-6);
        assert!((split.right_stats.hess_sum - expected_right.hess_sum).abs() < 1e-6);
        assert!((split.right_stats.grad_sq_sum - expected_right.grad_sq_sum).abs() < 1e-6);
        assert_eq!(split.right_stats.row_count, expected_right.count);
    }

    #[test]
    fn test_best_split_categorical_single_populated() {
        // Only 1 category has data -> no valid split possible
        let num_cats = 3;
        let nan_bin = 255usize;
        let num_bins = nan_bin + 1;
        let mut bins = vec![
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            };
            num_bins
        ];
        bins[1] = HistogramBin {
            grad_sum: 2.0,
            hess_sum: 5.0,
            grad_sq_sum: 0.0,
            count: 20,
        };
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };

        let options = SplitSelectionOptions::default();
        let result =
            CpuBackend::best_split_for_categorical_feature(&fh, 0, options, num_cats, None);
        assert!(
            result.is_none(),
            "single populated category should not split"
        );
    }

    #[test]
    fn test_apply_split_categorical_bitset() {
        // Create a BinnedMatrix with 6 rows, 1 feature.
        // Category bin values: [0, 1, 2, 0, 1, 2]
        // Bitset: category 0 and 1 go left (bits 0,1 set = 0b0000_0011 = 3)
        let binned = BinnedMatrix::new(
            6,
            1,
            2, // max_bin = 2
            vec![0, 1, 2, 0, 1, 2],
        )
        .expect("valid matrix");

        let split = SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin: 0, // unused for categorical
            gain: 1.0,
            default_left: true,
            is_categorical: true,
            categorical_bitset: Some(vec![0b0000_0011]), // cats 0,1 go left
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                row_count: 0,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                row_count: 0,
            },
        };

        let node_slice = NodeSlice {
            node_id: 0,
            row_indices: (0..6).collect(),
        };

        let backend = CpuBackend;
        let partition = backend
            .apply_split(&binned, &node_slice, &split)
            .expect("partition should succeed");
        let left = &partition.left_row_indices;
        let right = &partition.right_row_indices;
        // Rows with bin 0 or 1 go left, rows with bin 2 go right
        assert_eq!(left.len(), 4, "categories 0,1 should go left");
        assert_eq!(right.len(), 2, "category 2 should go right");
        // Verify specific rows
        assert!(left.contains(&0)); // bin 0
        assert!(left.contains(&1)); // bin 1
        assert!(left.contains(&3)); // bin 0
        assert!(left.contains(&4)); // bin 1
        assert!(right.contains(&2)); // bin 2
        assert!(right.contains(&5)); // bin 2
    }

    #[test]
    fn best_split_morph_at_warmup_matches_best_split_with_options() {
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        use alloygbm_engine::MorphContext;

        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let options = SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: 255,
        };

        let cfg = MorphConfig {
            balance_penalty: false,
            ..MorphConfig::default()
        };
        let morph = MorphContext {
            iteration: 0,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            config: cfg,
            precomputed: MorphPrecomputed::for_iteration(0, &cfg),
        };

        let standard = backend
            .best_split_with_options(&histograms, options, &[], &[])
            .expect("standard split search should succeed");
        let morph_result = backend
            .best_split_morph(&histograms, options, &[], &[], &morph)
            .expect("morph split search should succeed");

        // At iteration < warmup with balance penalty off, compute_morph_gain returns
        // exactly the standard XGBoost gain, so both paths must select the same split.
        assert!(
            standard.is_some(),
            "test fixture must produce a non-trivial split (standard path returned None)"
        );
        match (standard, morph_result) {
            (Some(a), Some(b)) => {
                assert_eq!(a.feature_index, b.feature_index, "feature_index disagreed");
                assert_eq!(a.threshold_bin, b.threshold_bin, "threshold_bin disagreed");
            }
            (None, None) => {}
            (a, b) => panic!(
                "split selection presence disagreed: standard={:?}, morph={:?}",
                a, b
            ),
        }
    }

    /// Regression test: warmup byte-equivalence must hold even with non-zero L1
    /// and L2 regularisation. This specifically guards against the bugs where:
    /// - EPSILON was missing from `gradient_gain` denominators (Issue 1)
    /// - L1 thresholding was not applied in the morph path (Issue 2)
    /// - `min_leaf_magnitude` was not checked in the morph path (Issue 3)
    #[test]
    fn best_split_morph_at_warmup_matches_with_l1_l2_regularization() {
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        use alloygbm_engine::MorphContext;

        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let options = SplitSelectionOptions {
            l2_lambda: 1.0,
            l1_alpha: 0.5,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: 255,
        };

        let cfg = MorphConfig {
            balance_penalty: false,
            ..MorphConfig::default()
        };
        let morph = MorphContext {
            iteration: 0,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            config: cfg,
            precomputed: MorphPrecomputed::for_iteration(0, &cfg),
        };

        let standard = backend
            .best_split_with_options(&histograms, options, &[], &[])
            .expect("standard split search should succeed");
        let morph_result = backend
            .best_split_morph(&histograms, options, &[], &[], &morph)
            .expect("morph split search should succeed");

        assert!(
            standard.is_some(),
            "test fixture must produce a non-trivial split (standard path returned None)"
        );
        let a = standard.unwrap();
        let b = morph_result.unwrap();
        assert_eq!(
            a.feature_index, b.feature_index,
            "feature_index disagreed under L1/L2 regularization"
        );
        assert_eq!(
            a.threshold_bin, b.threshold_bin,
            "threshold_bin disagreed under L1/L2 regularization"
        );
    }

    #[test]
    fn best_split_morph_with_dro_uses_robust_gradient_gain_signal() {
        use alloygbm_core::{DroConfig, DroMetric, MorphConfig, MorphPrecomputed};
        use alloygbm_engine::MorphContext;

        let backend = CpuBackend;
        let histograms = backend
            .build_histograms(
                &sample_binned_matrix(),
                &sample_gradients(),
                &sample_node(),
                &[FeatureTile::new(0, 2).expect("feature tile is valid")],
            )
            .expect("histograms should build");

        let options = SplitSelectionOptions {
            l2_lambda: 0.1,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            dro_config: Some(DroConfig {
                radius: 0.05,
                metric: DroMetric::Wasserstein,
            }),
            missing_bin_index: 255,
        };

        let cfg = MorphConfig {
            morph_warmup_iters: 0,
            balance_penalty: false,
            ..MorphConfig::default()
        };
        let morph = MorphContext {
            iteration: 10,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            config: cfg,
            precomputed: MorphPrecomputed::for_iteration(10, &cfg),
        };

        let split = backend
            .best_split_morph(&histograms, options, &[], &[], &morph)
            .expect("morph split search should succeed")
            .expect("test fixture should produce a split");

        let left_gradient_sum = leaf_effective_gradient(
            split.left_stats.grad_sum,
            split.left_stats.grad_sq_sum,
            split.left_stats.row_count,
            options.l1_alpha,
            options.dro_config.as_ref(),
        );
        let right_gradient_sum = leaf_effective_gradient(
            split.right_stats.grad_sum,
            split.right_stats.grad_sq_sum,
            split.right_stats.row_count,
            options.l1_alpha,
            options.dro_config.as_ref(),
        );
        let expected = compute_morph_gain(
            MorphGainInputs {
                left: SplitSideStats {
                    gradient_sum: left_gradient_sum,
                    hessian_sum: split.left_stats.hess_sum,
                    count: split.left_stats.row_count,
                },
                right: SplitSideStats {
                    gradient_sum: right_gradient_sum,
                    hessian_sum: split.right_stats.hess_sum,
                    count: split.right_stats.row_count,
                },
                iteration: morph.iteration,
                total_iterations: morph.total_iterations,
                grad_mean: morph.grad_mean,
                grad_std: morph.grad_std,
                lambda_l2: options.l2_lambda,
            },
            &morph.config,
            &morph.precomputed,
        );

        assert!((split.gain - expected).abs() < 1e-6);
    }

    /// Regression test: at `iteration < morph_warmup_iters` with `balance_penalty=false`,
    /// the morph categorical path must select the same partition as the standard path.
    ///
    /// Uses a 4-category bundle where categories 0,1 have strongly negative gradients
    /// and categories 2,3 have strongly positive gradients, making the best split
    /// unambiguous regardless of the gain formula used.
    #[test]
    fn best_split_morph_at_warmup_matches_categorical_split() {
        use alloygbm_core::{MorphConfig, MorphPrecomputed};
        use alloygbm_engine::{CategoricalFeatureInfo, MorphContext};

        // Build a HistogramBundle with one categorical feature (4 categories).
        // Categories 0,1: negative gradient (score < 0)
        // Categories 2,3: positive gradient (score > 0)
        // Fisher-sort will place cats 0,1 on the left side, cats 2,3 on the right.
        let num_cats = 4usize;
        let nan_bin = 255usize;
        let num_bins = nan_bin + 1;
        let mut bins = vec![
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                grad_sq_sum: 0.0,
                count: 0,
            };
            num_bins
        ];
        // Category 0: strongly negative gradient
        bins[0] = HistogramBin {
            grad_sum: -4.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 20,
        };
        // Category 1: negative gradient
        bins[1] = HistogramBin {
            grad_sum: -3.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 20,
        };
        // Category 2: positive gradient
        bins[2] = HistogramBin {
            grad_sum: 3.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 20,
        };
        // Category 3: strongly positive gradient
        bins[3] = HistogramBin {
            grad_sum: 4.0,
            hess_sum: 2.0,
            grad_sq_sum: 0.0,
            count: 20,
        };

        let feature_histogram = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let histograms = HistogramBundle {
            node_id: 0,
            feature_histograms: vec![feature_histogram],
        };

        let options = SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            dro_config: None,
            missing_bin_index: nan_bin,
        };

        let cat_features = vec![CategoricalFeatureInfo {
            feature_index: 0,
            num_categories: num_cats,
        }];

        let cfg = MorphConfig {
            balance_penalty: false,
            ..MorphConfig::default()
        };
        let morph = MorphContext {
            iteration: 0,
            total_iterations: 100,
            grad_mean: 0.0,
            grad_std: 1.0,
            config: cfg,
            precomputed: MorphPrecomputed::for_iteration(0, &cfg),
        };

        let backend = CpuBackend;
        let standard = backend
            .best_split_with_options(&histograms, options, &[], &cat_features)
            .expect("standard split search should succeed");
        let morph_result = backend
            .best_split_morph(&histograms, options, &[], &cat_features, &morph)
            .expect("morph split search should succeed");

        assert!(
            standard.is_some(),
            "test fixture must produce a non-trivial split (standard path returned None)"
        );
        let a = standard.unwrap();
        let b = morph_result.unwrap();

        assert!(a.is_categorical, "standard split should be categorical");
        assert!(b.is_categorical, "morph split should be categorical");
        assert_eq!(
            a.feature_index, b.feature_index,
            "feature_index disagreed for categorical morph at warmup"
        );
        // Both paths must select the same bitset partition.
        assert_eq!(
            a.categorical_bitset, b.categorical_bitset,
            "categorical_bitset disagreed for morph at warmup"
        );
        assert_eq!(
            a.default_left, b.default_left,
            "default_left (NaN direction) disagreed for morph at warmup"
        );
        assert!(
            (a.gain - b.gain).abs() < 1e-5,
            "gain diverged at warmup: standard={}, morph={}",
            a.gain,
            b.gain
        );
    }

    fn make_options(
        l1_alpha: f32,
        l2_lambda: f32,
        min_child_hessian: f32,
        min_leaf_magnitude: f32,
        missing_bin_index: usize,
    ) -> SplitSelectionOptions {
        SplitSelectionOptions {
            l1_alpha,
            l2_lambda,
            min_child_hessian,
            min_leaf_magnitude,
            dro_config: None,
            missing_bin_index,
        }
    }

    #[test]
    fn simd_standard_bin_scan_matches_scalar() {
        let bins: Vec<HistogramBin> = (0..32)
            .map(|i| HistogramBin {
                grad_sum: ((i as f32 - 15.5) * 0.1).sin(),
                hess_sum: 0.5 + (i as f32 * 0.05).cos().abs(),
                grad_sq_sum: 0.0,
                count: 10 + (i as u32 % 7),
            })
            .collect();
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = make_options(0.05, 0.1, 1.0, 0.0, 31);
        let scalar =
            CpuBackend::best_split_for_feature_inner(&fh, 0, options, GainStrategy::Standard, None);
        let simd = CpuBackend::best_split_for_feature_standard_simd(&fh, 0, options);
        match (scalar, simd) {
            (Some(s), Some(v)) => {
                assert_eq!(s.threshold_bin, v.threshold_bin, "threshold_bin mismatch");
                assert!(
                    (s.gain - v.gain).abs() < 1e-4,
                    "gain drift: scalar={} simd={}",
                    s.gain,
                    v.gain
                );
                assert_eq!(s.default_left, v.default_left);
            }
            (None, None) => {}
            (a, b) => panic!(
                "scalar/simd disagree on Some-ness: scalar={}, simd={}",
                a.is_some(),
                b.is_some()
            ),
        }
    }

    #[test]
    fn simd_standard_bin_scan_matches_scalar_with_l1() {
        let bins: Vec<HistogramBin> = (0..16)
            .map(|i| HistogramBin {
                grad_sum: (i as f32 - 7.5) * 0.02,
                hess_sum: 1.0,
                grad_sq_sum: 0.0,
                count: 20,
            })
            .collect();
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = make_options(0.10, 0.1, 0.5, 0.0, 15);
        let scalar =
            CpuBackend::best_split_for_feature_inner(&fh, 0, options, GainStrategy::Standard, None);
        let simd = CpuBackend::best_split_for_feature_standard_simd(&fh, 0, options);
        match (scalar, simd) {
            (Some(s), Some(v)) => {
                assert_eq!(s.threshold_bin, v.threshold_bin);
                assert!((s.gain - v.gain).abs() < 1e-4);
            }
            (None, None) => {}
            _ => panic!("scalar/simd disagreement"),
        }
    }

    #[test]
    fn simd_standard_bin_scan_matches_scalar_with_min_leaf_magnitude() {
        // Exercise the min_leaf_magnitude rejection branch.
        let bins: Vec<HistogramBin> = (0..16)
            .map(|i| HistogramBin {
                grad_sum: ((i as f32 - 7.5) * 0.05).sin(),
                hess_sum: 1.0 + (i as f32 * 0.1).cos().abs(),
                grad_sq_sum: 0.0,
                count: 12 + (i as u32 % 5),
            })
            .collect();
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = make_options(0.0, 0.1, 0.0, 0.05, 15);
        let scalar =
            CpuBackend::best_split_for_feature_inner(&fh, 0, options, GainStrategy::Standard, None);
        let simd = CpuBackend::best_split_for_feature_standard_simd(&fh, 0, options);
        match (scalar, simd) {
            (Some(s), Some(v)) => {
                assert_eq!(s.threshold_bin, v.threshold_bin);
                assert!((s.gain - v.gain).abs() < 1e-4);
                assert_eq!(s.default_left, v.default_left);
            }
            (None, None) => {}
            _ => panic!("scalar/simd disagreement on min_leaf_magnitude path"),
        }
    }

    #[test]
    fn simd_standard_bin_scan_matches_scalar_with_missing_bin() {
        // Real missing-bin contribution exercises the NaN-direction routing.
        let mut bins: Vec<HistogramBin> = (0..16)
            .map(|i| HistogramBin {
                grad_sum: ((i as f32 - 7.5) * 0.1).sin(),
                hess_sum: 1.0 + (i as f32 * 0.05).cos().abs(),
                grad_sq_sum: 0.0,
                count: 8 + (i as u32 % 4),
            })
            .collect();
        // Simulate non-trivial missing bin at index 15.
        bins[15] = HistogramBin {
            grad_sum: 0.4,
            hess_sum: 1.5,
            grad_sq_sum: 0.0,
            count: 7,
        };
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = make_options(0.0, 0.1, 0.5, 0.0, 15);
        let scalar =
            CpuBackend::best_split_for_feature_inner(&fh, 0, options, GainStrategy::Standard, None);
        let simd = CpuBackend::best_split_for_feature_standard_simd(&fh, 0, options);
        match (scalar, simd) {
            (Some(s), Some(v)) => {
                assert_eq!(s.threshold_bin, v.threshold_bin);
                assert!((s.gain - v.gain).abs() < 1e-4);
                assert_eq!(s.default_left, v.default_left);
                assert_eq!(s.left_stats.row_count, v.left_stats.row_count);
                assert_eq!(s.right_stats.row_count, v.right_stats.row_count);
            }
            (None, None) => {}
            _ => panic!("scalar/simd disagreement on missing-bin path"),
        }
    }

    #[test]
    fn dro_missing_bin_split_stats_match_direct_scan() {
        let missing_bin = 7usize;
        let mut bins: Vec<HistogramBin> = (0..=missing_bin)
            .map(|i| HistogramBin {
                grad_sum: (i as f32 - 3.0) * 0.7,
                hess_sum: 1.0 + i as f32 * 0.2,
                grad_sq_sum: 0.5 + i as f32 * 0.4,
                count: 3 + i as u32,
            })
            .collect();
        bins[missing_bin] = HistogramBin {
            grad_sum: -0.8,
            hess_sum: 1.4,
            grad_sq_sum: 1.2,
            count: 5,
        };
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };
        let options = SplitSelectionOptions {
            dro_config: Some(alloygbm_core::DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            missing_bin_index: missing_bin,
            ..SplitSelectionOptions::default()
        };

        let split =
            CpuBackend::best_split_for_feature_inner(&fh, 0, options, GainStrategy::Standard, None)
                .expect("dro split with missing bin should exist");
        let mut expected_left = HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        let mut expected_right = HistogramBin {
            grad_sum: 0.0,
            hess_sum: 0.0,
            grad_sq_sum: 0.0,
            count: 0,
        };
        for (bin_id, bin) in fh.bins.iter().enumerate() {
            let goes_left = if bin_id == missing_bin {
                split.default_left
            } else {
                bin_id <= split.threshold_bin as usize
            };
            let target = if goes_left {
                &mut expected_left
            } else {
                &mut expected_right
            };
            target.grad_sum += bin.grad_sum;
            target.hess_sum += bin.hess_sum;
            target.grad_sq_sum += bin.grad_sq_sum;
            target.count += bin.count;
        }

        assert!((split.left_stats.grad_sum - expected_left.grad_sum).abs() < 1e-6);
        assert!((split.left_stats.hess_sum - expected_left.hess_sum).abs() < 1e-6);
        assert!((split.left_stats.grad_sq_sum - expected_left.grad_sq_sum).abs() < 1e-6);
        assert_eq!(split.left_stats.row_count, expected_left.count);
        assert!((split.right_stats.grad_sum - expected_right.grad_sum).abs() < 1e-6);
        assert!((split.right_stats.hess_sum - expected_right.hess_sum).abs() < 1e-6);
        assert!((split.right_stats.grad_sq_sum - expected_right.grad_sq_sum).abs() < 1e-6);
        assert_eq!(split.right_stats.row_count, expected_right.count);
    }
}
