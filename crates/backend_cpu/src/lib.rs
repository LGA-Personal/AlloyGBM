use alloygbm_core::{
    BinnedMatrix, Device, FeatureTile, GradientPair, HistogramBundle, HistogramFeatureView,
    NodeSlice, NodeStats, PartitionResult, SplitCandidate, leaf_effective_gradient,
};
use alloygbm_engine::{
    CategoricalFeatureInfo, EngineError, EngineResult, FactorSplitContext, MorphContext,
    SplitSelectionOptions,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeMap;

mod arena;
mod backend_ops;
mod factor_split;
mod morph;
pub use morph::{MorphGainInputs, SplitSideStats, compute_morph_gain};

mod pl_histogram;
pub use pl_histogram::build_linear_histograms_cpu;

mod pl;

mod split_helpers;

use arena::{
    BIN_HEAVY_THRESHOLD, HistogramArena, HistogramKernelPath, PARALLEL_TILE_WORKLOAD_THRESHOLD,
    SMALL_TILE_WORKLOAD_THRESHOLD, TINY_NODE_ROW_THRESHOLD,
};
use factor_split::{FactorSplitScratch, with_factor_split_scratch};
use split_helpers::{
    GainStrategy, MissingDirectionCandidate, ScalarSideStats, apply_feature_weight,
    categorical_bitset_for_prefix, categorical_bitset_for_prefix_into, goes_left_for_split,
    l1_threshold_gradient, split_gain_term,
};

pub use alloygbm_core::simd;

thread_local! {
    /// Per-thread reusable histogram arena to avoid repeated allocation.
    static THREAD_ARENA: RefCell<HistogramArena> = RefCell::new(HistogramArena::new(0, 0, false));
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuBackend;

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

    fn build_tile_histograms_per_feature<const INCLUDE_GRAD_SQ: bool>(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        start_feature: usize,
        end_feature: usize,
        arena: &mut HistogramArena,
    ) {
        let row_count = binned_matrix.row_count;
        let use_col_major = binned_matrix.has_col_major();
        for feature_index in start_feature..end_feature {
            let base = (feature_index - start_feature) * arena.bin_count;

            if use_col_major {
                // Column-major: sequential bin reads — cache-friendly
                let col_base = feature_index * row_count;
                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let bin_index = binned_matrix.col_bin(col_base + row_index) as usize;
                    let gradient = gradients[row_index];
                    let target = base + bin_index;
                    arena.grad_sums[target] += gradient.grad;
                    arena.hess_sums[target] += gradient.hess;
                    if INCLUDE_GRAD_SQ {
                        arena.grad_sq_sums.as_mut().expect("DRO arena")[target] +=
                            gradient.grad * gradient.grad;
                    }
                    arena.counts[target] += 1;
                }
            } else {
                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let cell_index = row_index * binned_matrix.feature_count + feature_index;
                    let bin_index = binned_matrix.row_bin(cell_index) as usize;
                    let gradient = gradients[row_index];
                    let target = base + bin_index;
                    arena.grad_sums[target] += gradient.grad;
                    arena.hess_sums[target] += gradient.hess;
                    if INCLUDE_GRAD_SQ {
                        arena.grad_sq_sums.as_mut().expect("DRO arena")[target] +=
                            gradient.grad * gradient.grad;
                    }
                    arena.counts[target] += 1;
                }
            }
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
                arena.grad_sq_sums.as_mut().expect("DRO arena")[flat_index] +=
                    gradient.grad * gradient.grad;
                arena.counts[flat_index] += 1;
            }
        }
    }

    pub(crate) fn should_parallelize_tiles(
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
        include_grad_sq: bool,
    ) -> EngineResult<HistogramBundle> {
        if binned_matrix.feature_bundle_map().is_some() {
            return Self::build_feature_histograms_for_bundled_tile(
                binned_matrix,
                gradients,
                node,
                tile,
                bin_count,
                include_grad_sq,
            );
        }
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
        THREAD_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            arena.resize_for_tile(tile_feature_count, bin_count, include_grad_sq);
            let use_per_feature = matches!(
                Self::select_histogram_kernel_path(
                    node.row_indices.len(),
                    tile_workload,
                    bin_count
                ),
                HistogramKernelPath::TinyNodeScalar | HistogramKernelPath::BinHeavyPerFeatureScalar
            ) || binned_matrix.has_col_major();
            match (use_per_feature, include_grad_sq) {
                (true, false) => Self::build_tile_histograms_per_feature::<false>(
                    binned_matrix,
                    gradients,
                    node,
                    start_feature,
                    end_feature,
                    &mut arena,
                ),
                (true, true) => Self::build_tile_histograms_per_feature::<true>(
                    binned_matrix,
                    gradients,
                    node,
                    start_feature,
                    end_feature,
                    &mut arena,
                ),
                (false, false) => Self::build_tile_histograms_row_first_unrolled::<false>(
                    binned_matrix,
                    gradients,
                    node,
                    start_feature,
                    end_feature,
                    &mut arena,
                ),
                (false, true) => Self::build_tile_histograms_row_first_unrolled::<true>(
                    binned_matrix,
                    gradients,
                    node,
                    start_feature,
                    end_feature,
                    &mut arena,
                ),
            }
            arena
                .to_bundle(node.node_id, start_feature)
                .map_err(EngineError::from)
        })
    }

    pub(crate) fn build_feature_histograms_for_bundled_tile(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        tile: &FeatureTile,
        bin_count: usize,
        include_grad_sq: bool,
    ) -> EngineResult<HistogramBundle> {
        let map = binned_matrix.feature_bundle_map().ok_or_else(|| {
            EngineError::ContractViolation(
                "bundled histogram kernel requires a feature bundle map".to_string(),
            )
        })?;
        let start_feature = tile.start_feature as usize;
        let end_feature = tile.end_feature as usize;
        if end_feature > binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature tile end {} exceeds feature_count {}",
                tile.end_feature, binned_matrix.feature_count
            )));
        }
        let mut storage_groups = BTreeMap::new();
        for feature in start_feature..end_feature {
            let assignment = map.assignment(feature).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "feature bundle map is missing feature {feature}"
                ))
            })?;
            storage_groups
                .entry(assignment.storage_feature)
                .or_insert_with(Vec::new)
                .push((feature - start_feature, assignment));
        }

        THREAD_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            arena.resize_for_tile(end_feature - start_feature, bin_count, include_grad_sq);
            for (storage_feature, assignments) in storage_groups {
                for &row_index in &node.row_indices {
                    let row_index = row_index as usize;
                    let storage_bin = binned_matrix.storage_col_bin(storage_feature, row_index);
                    let gradient = gradients[row_index];
                    for &(local_feature, assignment) in &assignments {
                        let bin = assignment.decode(storage_bin, binned_matrix.missing_bin());
                        let target = local_feature * arena.bin_count + usize::from(bin);
                        arena.grad_sums[target] += gradient.grad;
                        arena.hess_sums[target] += gradient.hess;
                        if include_grad_sq {
                            arena.grad_sq_sums.as_mut().expect("DRO arena")[target] +=
                                gradient.grad * gradient.grad;
                        }
                        arena.counts[target] += 1;
                    }
                }
            }
            arena
                .to_bundle(node.node_id, start_feature)
                .map_err(EngineError::from)
        })
    }

    pub(crate) fn build_histograms_internal(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        parallel_tiles: bool,
        include_grad_sq: bool,
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
        let mut per_tile_histograms = if parallel_tiles {
            feature_tiles
                .par_iter()
                .map(|tile| {
                    Self::build_feature_histograms_for_tile(
                        binned_matrix,
                        gradients,
                        node,
                        tile,
                        bin_count,
                        include_grad_sq,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            feature_tiles
                .iter()
                .map(|tile| {
                    Self::build_feature_histograms_for_tile(
                        binned_matrix,
                        gradients,
                        node,
                        tile,
                        bin_count,
                        include_grad_sq,
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut histograms = per_tile_histograms
            .drain(..1)
            .next()
            .expect("feature tiles are non-empty")?;
        for tile_histograms in per_tile_histograms {
            histograms
                .append(tile_histograms?)
                .map_err(EngineError::from)?;
        }
        Ok(histograms)
    }

    fn build_tile_histograms_row_first_unrolled<const INCLUDE_GRAD_SQ: bool>(
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
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx0] +=
                        gradient0.grad * gradient0.grad;
                }
                arena.counts[idx0] += 1;

                arena.grad_sums[idx1] += gradient1.grad;
                arena.hess_sums[idx1] += gradient1.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx1] +=
                        gradient1.grad * gradient1.grad;
                }
                arena.counts[idx1] += 1;

                arena.grad_sums[idx2] += gradient2.grad;
                arena.hess_sums[idx2] += gradient2.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx2] +=
                        gradient2.grad * gradient2.grad;
                }
                arena.counts[idx2] += 1;

                arena.grad_sums[idx3] += gradient3.grad;
                arena.hess_sums[idx3] += gradient3.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx3] +=
                        gradient3.grad * gradient3.grad;
                }
                arena.counts[idx3] += 1;

                arena.grad_sums[idx4] += gradient4.grad;
                arena.hess_sums[idx4] += gradient4.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx4] +=
                        gradient4.grad * gradient4.grad;
                }
                arena.counts[idx4] += 1;

                arena.grad_sums[idx5] += gradient5.grad;
                arena.hess_sums[idx5] += gradient5.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx5] +=
                        gradient5.grad * gradient5.grad;
                }
                arena.counts[idx5] += 1;

                arena.grad_sums[idx6] += gradient6.grad;
                arena.hess_sums[idx6] += gradient6.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx6] +=
                        gradient6.grad * gradient6.grad;
                }
                arena.counts[idx6] += 1;

                arena.grad_sums[idx7] += gradient7.grad;
                arena.hess_sums[idx7] += gradient7.hess;
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[idx7] +=
                        gradient7.grad * gradient7.grad;
                }
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
                if INCLUDE_GRAD_SQ {
                    arena.grad_sq_sums.as_mut().expect("DRO arena")[flat_index] +=
                        gradient.grad * gradient.grad;
                }
                arena.counts[flat_index] += 1;
            }
        }
    }

    fn best_split_for_feature(
        feature_histogram: HistogramFeatureView<'_>,
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
        feature_histogram: HistogramFeatureView<'_>,
        node_id: u32,
        options: SplitSelectionOptions,
    ) -> Option<SplitCandidate> {
        use crate::simd::{f32x8, l1_threshold_f32x8};
        use wide::{CmpGe, CmpGt};

        const EPSILON: f32 = 1e-6;

        if feature_histogram.len() < 2 {
            return None;
        }

        let grad_sums = feature_histogram.grad_sums();
        let hess_sums = feature_histogram.hess_sums();
        let counts = feature_histogram.counts();

        // Extract missing-value stats if the histogram covers the NaN bin.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_count) =
            if missing_bin_idx < feature_histogram.len() {
                (
                    grad_sums[missing_bin_idx],
                    hess_sums[missing_bin_idx],
                    counts[missing_bin_idx],
                )
            } else {
                (0.0_f32, 0.0_f32, 0_u32)
            };

        let mut total_grad = 0.0_f32;
        let mut total_hess = 0.0_f32;
        let mut total_count = 0_u32;
        for index in 0..feature_histogram.len() {
            total_grad += grad_sums[index];
            total_hess += hess_sums[index];
            total_count += counts[index];
        }

        if total_hess <= options.min_child_hessian {
            return None;
        }

        let nm_total_grad = total_grad - missing_grad;
        let nm_total_hess = total_hess - missing_hess;
        let nm_total_count = total_count.saturating_sub(missing_count);

        let parent_denom = total_hess + options.l2_lambda + EPSILON;
        let parent_grad = l1_threshold_gradient(total_grad, options.l1_alpha);
        let parent_gain_term = (parent_grad * parent_grad) / parent_denom;

        let scan_limit = feature_histogram.len().min(missing_bin_idx);
        if scan_limit == 0 {
            return None;
        }

        // Pre-compute scalar cumulative left-side stats. The prefix scan is
        // inherently sequential, so we keep it in scalar code.
        let mut cum_left_grad = vec![0.0_f32; scan_limit];
        let mut cum_left_hess = vec![0.0_f32; scan_limit];
        let mut cum_left_count = vec![0_u32; scan_limit];
        {
            let mut g = 0.0_f32;
            let mut h = 0.0_f32;
            let mut c = 0_u32;
            for i in 0..scan_limit {
                g += grad_sums[i];
                h += hess_sums[i];
                c += counts[i];
                cum_left_grad[i] = g;
                cum_left_hess[i] = h;
                cum_left_count[i] = c;
            }
        }

        // Per-NaN-direction broadcast values.
        let l1_alpha = options.l1_alpha;
        let l2_lambda = options.l2_lambda;
        let min_child_hessian = options.min_child_hessian;
        let min_rows = options.min_rows_per_leaf as f32;
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
        let min_rows_v = f32x8::splat(min_rows);
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
                //   eff_lc >= min_rows_per_leaf
                //   eff_rc >= min_rows_per_leaf
                //   eff_lh > min_child_hessian
                //   eff_rh > min_child_hessian
                let lc_ok = eff_lc.cmp_ge(min_rows_v);
                let rc_ok = eff_rc.cmp_ge(min_rows_v);
                let lh_ok = eff_lh.cmp_gt(min_child_hess_v);
                let rh_ok = eff_rh.cmp_gt(min_child_hess_v);
                // Combine via bitwise AND on the float-mask representation.
                let valid_mask = lc_ok & rc_ok & lh_ok & rh_ok;

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
        let left_count = cum_left_count[threshold_bin];
        let right_grad = nm_total_grad - left_grad;
        let right_hess = nm_total_hess - left_hess;
        let right_count = nm_total_count.saturating_sub(left_count);

        let (eff_lg, eff_lh, eff_lq, eff_lc, eff_rg, eff_rh, eff_rq, eff_rc) = if best_default_left
        {
            (
                left_grad + missing_grad,
                left_hess + missing_hess,
                0.0,
                left_count + missing_count,
                right_grad,
                right_hess,
                0.0,
                right_count,
            )
        } else {
            (
                left_grad,
                left_hess,
                0.0,
                left_count,
                right_grad + missing_grad,
                right_hess + missing_hess,
                0.0,
                right_count + missing_count,
            )
        };

        Some(SplitCandidate {
            node_id,
            feature_index: feature_histogram.feature_index(),
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
        feature_histogram: HistogramFeatureView<'_>,
        node_id: u32,
        options: SplitSelectionOptions,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        if let Some(ctx) = factor_context {
            return with_factor_split_scratch(ctx.exposures.factor_count, |scratch| {
                Self::best_split_for_feature_inner_with_scratch(
                    feature_histogram,
                    node_id,
                    options,
                    strategy,
                    factor_context,
                    Some(scratch),
                )
            });
        }
        Self::best_split_for_feature_inner_with_scratch(
            feature_histogram,
            node_id,
            options,
            strategy,
            factor_context,
            None,
        )
    }

    fn best_split_for_feature_inner_with_scratch(
        feature_histogram: HistogramFeatureView<'_>,
        node_id: u32,
        options: SplitSelectionOptions,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
        mut factor_scratch: Option<&mut FactorSplitScratch>,
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if feature_histogram.len() < 2 {
            return None;
        }

        // Extract missing-value stats if the histogram covers the NaN bin.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
            if missing_bin_idx < feature_histogram.len() {
                let mb = feature_histogram.bin(missing_bin_idx).expect("bounded bin");
                (mb.grad_sum, mb.hess_sum, mb.grad_sq_sum, mb.count)
            } else {
                (0.0, 0.0, 0.0, 0)
            };

        let mut total_grad = 0.0_f32;
        let mut total_hess = 0.0_f32;
        let mut total_grad_sq = 0.0_f32;
        let mut total_count = 0_u32;
        for bin in feature_histogram.bins() {
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

        if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_deref_mut()) {
            scratch.prepare_numeric_prefix(
                ctx,
                feature_histogram.feature_index() as usize,
                feature_histogram.len().min(missing_bin_idx),
                missing_bin_idx,
            );
        }
        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        let mut left_grad = 0.0_f32;
        let mut left_hess = 0.0_f32;
        let mut left_grad_sq = 0.0_f32;
        let mut left_count = 0_u32;

        // Scan only non-missing bins (0..min(num_bins-1, MISSING_BIN-1)).
        let scan_limit = feature_histogram.len().min(missing_bin_idx);
        for (threshold_bin, bin) in feature_histogram.bins().enumerate().take(scan_limit) {
            left_grad += bin.grad_sum;
            left_hess += bin.hess_sum;
            left_grad_sq += bin.grad_sq_sum;
            left_count += bin.count;
            if let Some(scratch) = factor_scratch.as_deref_mut() {
                scratch.add_numeric_threshold_bin_to_left(threshold_bin);
            }

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
                let min_rows = options.min_rows_per_leaf as u32;
                if left.count < min_rows
                    || right.count < min_rows
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

                if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_deref_mut())
                {
                    let left_leaf_value = -left_grad_for_gain / left_denom;
                    let right_leaf_value = -right_grad_for_gain / right_denom;
                    gain -= scratch.numeric_prefix_penalty(
                        candidate.default_left,
                        left_leaf_value,
                        right_leaf_value,
                        ctx.factor_penalty,
                        ctx.row_indices.len(),
                    );
                }

                if !gain.is_finite() {
                    continue;
                }

                if gain > best_gain {
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index(),
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
        feature_histogram: HistogramFeatureView<'_>,
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
    pub(crate) fn best_split_morph_categorical_feature(
        feature_histogram: HistogramFeatureView<'_>,
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
        feature_histogram: HistogramFeatureView<'_>,
        node_id: u32,
        options: SplitSelectionOptions,
        num_categories: usize,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        if let Some(ctx) = factor_context {
            return with_factor_split_scratch(ctx.exposures.factor_count, |scratch| {
                Self::best_split_for_categorical_feature_inner_with_scratch(
                    feature_histogram,
                    node_id,
                    options,
                    num_categories,
                    strategy,
                    factor_context,
                    Some(scratch),
                )
            });
        }
        Self::best_split_for_categorical_feature_inner_with_scratch(
            feature_histogram,
            node_id,
            options,
            num_categories,
            strategy,
            factor_context,
            None,
        )
    }

    fn best_split_for_categorical_feature_inner_with_scratch(
        feature_histogram: HistogramFeatureView<'_>,
        node_id: u32,
        options: SplitSelectionOptions,
        num_categories: usize,
        strategy: GainStrategy<'_>,
        factor_context: Option<&FactorSplitContext<'_>>,
        mut factor_scratch: Option<&mut FactorSplitScratch>,
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if num_categories < 2 {
            return None;
        }

        // Extract missing-value stats.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_grad_sq, missing_count) =
            if missing_bin_idx < feature_histogram.len() {
                let mb = feature_histogram.bin(missing_bin_idx).expect("bounded bin");
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
            .min(feature_histogram.len())
            .min(missing_bin_idx);
        for bin_id in 0..scan_limit {
            let bin = feature_histogram.bin(bin_id).expect("bounded bin");
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
        if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_deref_mut()) {
            let sorted_category_bins: Vec<u16> =
                categories.iter().map(|category| category.0).collect();
            scratch.prepare_categorical_prefix(
                ctx,
                feature_histogram.feature_index() as usize,
                &sorted_category_bins,
                missing_bin_idx,
            );
        }

        // Try splits: first k categories go left, rest go right (k = 1..len-1).
        for k in 0..categories.len() - 1 {
            let (_, g, h, q, c) = categories[k];
            left_grad += g;
            left_hess += h;
            left_grad_sq += q;
            left_count += c;
            if let Some(scratch) = factor_scratch.as_deref_mut() {
                scratch.add_categorical_prefix_bin_to_left(k);
                categorical_bitset_for_prefix_into(
                    num_categories,
                    &categories,
                    k,
                    &mut scratch.categorical_bitset,
                );
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
                let min_rows = options.min_rows_per_leaf as u32;
                if left.count < min_rows
                    || right.count < min_rows
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

                if let (Some(ctx), Some(scratch)) = (factor_context, factor_scratch.as_deref_mut())
                {
                    let left_leaf_value = -left_grad_for_gain / left_denom;
                    let right_leaf_value = -right_grad_for_gain / right_denom;
                    gain -= scratch.categorical_prefix_penalty(
                        candidate.default_left,
                        left_leaf_value,
                        right_leaf_value,
                        ctx.factor_penalty,
                        ctx.row_indices.len(),
                    );
                }

                if !gain.is_finite() {
                    continue;
                }

                if gain > best_gain {
                    let bitset = if let Some(scratch) = factor_scratch.as_deref() {
                        scratch.categorical_bitset.clone()
                    } else {
                        categorical_bitset_for_prefix(num_categories, &categories, k)
                    };
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index(),
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

    pub(crate) const PARALLEL_SPLIT_FEATURE_THRESHOLD: usize = 16;

    pub(crate) fn best_split_with_options_internal(
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
        factor_context: Option<&FactorSplitContext<'_>>,
    ) -> Option<SplitCandidate> {
        let find_best = |fh: HistogramFeatureView<'_>| -> Option<SplitCandidate> {
            let fi = fh.feature_index() as usize;
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

        if histograms.feature_count() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
            (0..histograms.feature_count())
                .into_par_iter()
                .filter_map(|index| find_best(histograms.feature(index).expect("bounded feature")))
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
            histograms.features().filter_map(find_best).reduce(|a, b| {
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
    pub(crate) fn best_split_morph_numeric_feature(
        feature_histogram: HistogramFeatureView<'_>,
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

    pub(crate) fn apply_split_with_stats_parallel(
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

#[cfg(test)]
mod tests;
