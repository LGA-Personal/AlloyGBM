use alloygbm_core::{
    BinnedMatrix, Device, FeatureHistogram, FeatureTile, GradientPair, HistogramBin,
    HistogramBundle, NodeSlice, NodeStats, PartitionResult, SplitCandidate,
};
use alloygbm_engine::{BackendOps, CategoricalFeatureInfo, EngineError, EngineResult, SplitSelectionOptions};
use rayon::prelude::*;
use std::cell::RefCell;

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
    counts: Vec<u32>,
}

impl HistogramArena {
    fn new(tile_feature_count: usize, bin_count: usize) -> Self {
        let flat_len = tile_feature_count * bin_count;
        Self {
            bin_count,
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            counts: vec![0; flat_len],
        }
    }

    /// Zero all accumulators without deallocating, allowing the arena to be reused.
    fn reset(&mut self) {
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
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
            self.counts.resize(flat_len, 0);
            self.grad_sums.fill(0.0);
            self.hess_sums.fill(0.0);
            self.counts.fill(0);
        }
    }

    fn materialize(&self, start_feature: usize, feature_histograms: &mut Vec<FeatureHistogram>) {
        CpuBackend::materialize_tile_histograms(
            start_feature,
            self.bin_count,
            &self.grad_sums,
            &self.hess_sums,
            &self.counts,
            feature_histograms,
        );
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
                arena.counts[idx0] += 1;

                arena.grad_sums[idx1] += gradient1.grad;
                arena.hess_sums[idx1] += gradient1.hess;
                arena.counts[idx1] += 1;

                arena.grad_sums[idx2] += gradient2.grad;
                arena.hess_sums[idx2] += gradient2.hess;
                arena.counts[idx2] += 1;

                arena.grad_sums[idx3] += gradient3.grad;
                arena.hess_sums[idx3] += gradient3.hess;
                arena.counts[idx3] += 1;

                arena.grad_sums[idx4] += gradient4.grad;
                arena.hess_sums[idx4] += gradient4.hess;
                arena.counts[idx4] += 1;

                arena.grad_sums[idx5] += gradient5.grad;
                arena.hess_sums[idx5] += gradient5.hess;
                arena.counts[idx5] += 1;

                arena.grad_sums[idx6] += gradient6.grad;
                arena.hess_sums[idx6] += gradient6.hess;
                arena.counts[idx6] += 1;

                arena.grad_sums[idx7] += gradient7.grad;
                arena.hess_sums[idx7] += gradient7.hess;
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
                arena.counts[flat_index] += 1;
            }
        }
    }

    fn best_split_for_feature(
        feature_histogram: &FeatureHistogram,
        node_id: u32,
        options: SplitSelectionOptions,
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if feature_histogram.bins.len() < 2 {
            return None;
        }

        // Extract missing-value stats if the histogram covers the NaN bin.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_count) =
            if missing_bin_idx < feature_histogram.bins.len() {
                let mb = &feature_histogram.bins[missing_bin_idx];
                (mb.grad_sum, mb.hess_sum, mb.count)
            } else {
                (0.0, 0.0, 0)
            };

        let mut total_grad = 0.0_f32;
        let mut total_hess = 0.0_f32;
        let mut total_count = 0_u32;
        for bin in &feature_histogram.bins {
            total_grad += bin.grad_sum;
            total_hess += bin.hess_sum;
            total_count += bin.count;
        }

        if total_hess <= options.min_child_hessian {
            return None;
        }

        // Non-missing totals for the scan loop.
        let nm_total_grad = total_grad - missing_grad;
        let nm_total_hess = total_hess - missing_hess;
        let nm_total_count = total_count.saturating_sub(missing_count);

        let parent_denom = total_hess + options.l2_lambda + EPSILON;
        let parent_grad = l1_threshold_gradient(total_grad, options.l1_alpha);
        let parent_gain_term = (parent_grad * parent_grad) / parent_denom;

        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        let mut left_grad = 0.0_f32;
        let mut left_hess = 0.0_f32;
        let mut left_count = 0_u32;

        // Scan only non-missing bins (0..min(num_bins-1, MISSING_BIN-1)).
        let scan_limit = feature_histogram.bins.len().min(missing_bin_idx);
        for (threshold_bin, bin) in feature_histogram.bins.iter().enumerate().take(scan_limit) {
            left_grad += bin.grad_sum;
            left_hess += bin.hess_sum;
            left_count += bin.count;

            // Skip if this isn't a valid split point (need at least the
            // next non-missing bin on the right).
            if threshold_bin + 1 >= scan_limit && nm_total_count == left_count {
                continue;
            }

            let right_grad = nm_total_grad - left_grad;
            let right_hess = nm_total_hess - left_hess;
            let right_count = nm_total_count.saturating_sub(left_count);

            // Try both NaN directions and pick the better one.
            let candidates: [(f32, f32, u32, f32, f32, u32, bool); 2] = [
                // NaN goes left
                (
                    left_grad + missing_grad,
                    left_hess + missing_hess,
                    left_count + missing_count,
                    right_grad,
                    right_hess,
                    right_count,
                    true,
                ),
                // NaN goes right
                (
                    left_grad,
                    left_hess,
                    left_count,
                    right_grad + missing_grad,
                    right_hess + missing_hess,
                    right_count + missing_count,
                    false,
                ),
            ];

            for &(eff_lg, eff_lh, eff_lc, eff_rg, eff_rh, eff_rc, default_left) in &candidates {
                if eff_lc == 0
                    || eff_rc == 0
                    || eff_lh <= options.min_child_hessian
                    || eff_rh <= options.min_child_hessian
                {
                    continue;
                }

                let left_grad_for_gain = l1_threshold_gradient(eff_lg, options.l1_alpha);
                let right_grad_for_gain = l1_threshold_gradient(eff_rg, options.l1_alpha);
                let left_denom = eff_lh + options.l2_lambda + EPSILON;
                let right_denom = eff_rh + options.l2_lambda + EPSILON;
                if options.min_leaf_magnitude > 0.0 {
                    let left_leaf_magnitude = left_grad_for_gain.abs() / left_denom;
                    let right_leaf_magnitude = right_grad_for_gain.abs() / right_denom;
                    if left_leaf_magnitude < options.min_leaf_magnitude
                        && right_leaf_magnitude < options.min_leaf_magnitude
                    {
                        continue;
                    }
                }

                let gain = (left_grad_for_gain * left_grad_for_gain) / left_denom
                    + (right_grad_for_gain * right_grad_for_gain) / right_denom
                    - parent_gain_term;

                if gain > best_gain {
                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index,
                        threshold_bin: threshold_bin as u16,
                        gain,
                        default_left,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: eff_lg,
                            hess_sum: eff_lh,
                            row_count: eff_lc,
                        },
                        right_stats: NodeStats {
                            grad_sum: eff_rg,
                            hess_sum: eff_rh,
                            row_count: eff_rc,
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
    ) -> Option<SplitCandidate> {
        const EPSILON: f32 = 1e-6;

        if num_categories < 2 {
            return None;
        }

        // Extract missing-value stats.
        let missing_bin_idx = options.missing_bin_index;
        let (missing_grad, missing_hess, missing_count) =
            if missing_bin_idx < feature_histogram.bins.len() {
                let mb = &feature_histogram.bins[missing_bin_idx];
                (mb.grad_sum, mb.hess_sum, mb.count)
            } else {
                (0.0, 0.0, 0)
            };

        // Collect populated categories (bins 0..num_categories).
        let mut categories: Vec<(u16, f32, f32, u32)> = Vec::new(); // (bin_id, grad, hess, count)
        let mut nm_total_grad = 0.0_f32;
        let mut nm_total_hess = 0.0_f32;
        let mut nm_total_count = 0_u32;

        let scan_limit = num_categories.min(feature_histogram.bins.len()).min(missing_bin_idx);
        for bin_id in 0..scan_limit {
            let bin = &feature_histogram.bins[bin_id];
            if bin.count > 0 {
                categories.push((bin_id as u16, bin.grad_sum, bin.hess_sum, bin.count));
            }
            nm_total_grad += bin.grad_sum;
            nm_total_hess += bin.hess_sum;
            nm_total_count += bin.count;
        }

        if categories.len() < 2 {
            return None;
        }

        let total_grad = nm_total_grad + missing_grad;
        let total_hess = nm_total_hess + missing_hess;

        if total_hess <= options.min_child_hessian {
            return None;
        }

        let parent_denom = total_hess + options.l2_lambda + EPSILON;
        let parent_grad_l1 = l1_threshold_gradient(total_grad, options.l1_alpha);
        let parent_gain_term = (parent_grad_l1 * parent_grad_l1) / parent_denom;

        // Sort categories by score: grad_sum / (hess_sum + l2_lambda + eps) ascending.
        categories.sort_by(|a, b| {
            let score_a = a.1 / (a.2 + options.l2_lambda + EPSILON);
            let score_b = b.1 / (b.2 + options.l2_lambda + EPSILON);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Prefix scan over sorted categories to find best partition.
        let mut best_candidate: Option<SplitCandidate> = None;
        let mut best_gain = 0.0_f32;
        let mut left_grad = 0.0_f32;
        let mut left_hess = 0.0_f32;
        let mut left_count = 0_u32;

        // Try splits: first k categories go left, rest go right (k = 1..len-1).
        for k in 0..categories.len() - 1 {
            let (_, g, h, c) = categories[k];
            left_grad += g;
            left_hess += h;
            left_count += c;

            let right_grad = nm_total_grad - left_grad;
            let right_hess = nm_total_hess - left_hess;
            let right_count = nm_total_count.saturating_sub(left_count);

            // Try both NaN directions.
            let candidates: [(f32, f32, u32, f32, f32, u32, bool); 2] = [
                // NaN goes left
                (
                    left_grad + missing_grad,
                    left_hess + missing_hess,
                    left_count + missing_count,
                    right_grad,
                    right_hess,
                    right_count,
                    true,
                ),
                // NaN goes right
                (
                    left_grad,
                    left_hess,
                    left_count,
                    right_grad + missing_grad,
                    right_hess + missing_hess,
                    right_count + missing_count,
                    false,
                ),
            ];

            for &(eff_lg, eff_lh, eff_lc, eff_rg, eff_rh, eff_rc, default_left) in &candidates {
                if eff_lc == 0
                    || eff_rc == 0
                    || eff_lh <= options.min_child_hessian
                    || eff_rh <= options.min_child_hessian
                {
                    continue;
                }

                let left_grad_for_gain = l1_threshold_gradient(eff_lg, options.l1_alpha);
                let right_grad_for_gain = l1_threshold_gradient(eff_rg, options.l1_alpha);
                let left_denom = eff_lh + options.l2_lambda + EPSILON;
                let right_denom = eff_rh + options.l2_lambda + EPSILON;
                if options.min_leaf_magnitude > 0.0 {
                    let left_leaf_magnitude = left_grad_for_gain.abs() / left_denom;
                    let right_leaf_magnitude = right_grad_for_gain.abs() / right_denom;
                    if left_leaf_magnitude < options.min_leaf_magnitude
                        && right_leaf_magnitude < options.min_leaf_magnitude
                    {
                        continue;
                    }
                }

                let gain = (left_grad_for_gain * left_grad_for_gain) / left_denom
                    + (right_grad_for_gain * right_grad_for_gain) / right_denom
                    - parent_gain_term;

                if gain > best_gain {
                    // Build bitset: categories 0..=k in sorted order go left.
                    let bitset_len = num_categories.div_ceil(8);
                    let mut bitset = vec![0u8; bitset_len];
                    for &(bin_id, _, _, _) in &categories[..=k] {
                        let byte_idx = (bin_id / 8) as usize;
                        let bit_idx = (bin_id % 8) as usize;
                        if byte_idx < bitset.len() {
                            bitset[byte_idx] |= 1 << bit_idx;
                        }
                    }

                    best_gain = gain;
                    best_candidate = Some(SplitCandidate {
                        node_id,
                        feature_index: feature_histogram.feature_index,
                        threshold_bin: 0, // unused for categorical
                        gain,
                        default_left,
                        is_categorical: true,
                        categorical_bitset: Some(bitset),
                        left_stats: NodeStats {
                            grad_sum: eff_lg,
                            hess_sum: eff_lh,
                            row_count: eff_lc,
                        },
                        right_stats: NodeStats {
                            grad_sum: eff_rg,
                            hess_sum: eff_rh,
                            row_count: eff_rc,
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
    ) -> Option<SplitCandidate> {
        // Apply per-feature weights when comparing splits across features.
        // The gain stored in SplitCandidate remains unweighted (the true gain);
        // the weighted gain is only used for the cross-feature comparison.
        let weighted_gain = |candidate: &SplitCandidate| -> f32 {
            let fi = candidate.feature_index as usize;
            if fi < feature_weights.len() {
                candidate.gain * feature_weights[fi]
            } else {
                candidate.gain
            }
        };

        let find_best = |fh: &FeatureHistogram| -> Option<SplitCandidate> {
            let fi = fh.feature_index as usize;
            if let Some(cat_info) = categorical_features.iter().find(|c| c.feature_index == fi) {
                Self::best_split_for_categorical_feature(
                    fh,
                    histograms.node_id,
                    options,
                    cat_info.num_categories,
                )
            } else {
                Self::best_split_for_feature(fh, histograms.node_id, options)
            }
        };

        if histograms.feature_histograms.len() >= Self::PARALLEL_SPLIT_FEATURE_THRESHOLD {
            histograms
                .feature_histograms
                .par_iter()
                .filter_map(&find_best)
                .reduce_with(|a, b| {
                    if weighted_gain(&b) > weighted_gain(&a) {
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
                    if weighted_gain(&b) > weighted_gain(&a) {
                        b
                    } else {
                        a
                    }
                })
        }
    }

    fn apply_split_with_stats_parallel(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        type ChunkResult = (Vec<u32>, Vec<u32>, f32, f32, f32, f32);
        let chunk_size = (node.row_indices.len() / rayon::current_num_threads().max(1)).max(4096);
        let chunk_results: Vec<ChunkResult> = node
            .row_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut left = Vec::new();
                let mut right = Vec::new();
                let mut lg = 0.0_f32;
                let mut lh = 0.0_f32;
                let mut rg = 0.0_f32;
                let mut rh = 0.0_f32;
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
                    } else {
                        right.push(row_index_u32);
                        rg += gradient.grad;
                        rh += gradient.hess;
                    }
                }
                (left, right, lg, lh, rg, rh)
            })
            .collect();

        let total_rows = node.row_indices.len();
        let mut left_row_indices = Vec::with_capacity(total_rows / 2);
        let mut right_row_indices = Vec::with_capacity(total_rows / 2);
        let mut left_grad_sum = 0.0_f32;
        let mut left_hess_sum = 0.0_f32;
        let mut right_grad_sum = 0.0_f32;
        let mut right_hess_sum = 0.0_f32;

        for (left, right, lg, lh, rg, rh) in chunk_results {
            left_row_indices.extend(left);
            right_row_indices.extend(right);
            left_grad_sum += lg;
            left_hess_sum += lh;
            right_grad_sum += rg;
            right_hess_sum += rh;
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
                row_count: left_count,
            },
            NodeStats {
                grad_sum: right_grad_sum,
                hess_sum: right_hess_sum,
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
        ))
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
        let mut right_grad_sum = 0.0_f32;
        let mut right_hess_sum = 0.0_f32;

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
            } else {
                right_row_indices.push(row_index_u32);
                right_grad_sum += gradient.grad;
                right_hess_sum += gradient.hess;
            }
        }

        let partition = PartitionResult {
            left_row_indices,
            right_row_indices,
        };
        let left_stats = NodeStats {
            grad_sum: left_grad_sum,
            hess_sum: left_hess_sum,
            row_count: partition.left_row_indices.len() as u32,
        };
        let right_stats = NodeStats {
            grad_sum: right_grad_sum,
            hess_sum: right_hess_sum,
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
        for &row_index in row_indices {
            let gradient = gradients.get(row_index as usize).ok_or_else(|| {
                EngineError::ContractViolation(format!(
                    "row index {row_index} is out of bounds for gradients length {}",
                    gradients.len()
                ))
            })?;
            grad_sum += gradient.grad;
            hess_sum += gradient.hess;
        }

        Ok(NodeStats {
            grad_sum,
            hess_sum,
            row_count: row_indices.len() as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::{DatasetMatrix, FeatureTile, TrainParams, TrainingDataset, TreeGrowth};
    use alloygbm_engine::{SquaredErrorObjective, Trainer};

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
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: -1.0,
                            hess_sum: 20.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
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
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: -0.5,
                            hess_sum: 5.0,
                            count: 5,
                        },
                        HistogramBin {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
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
                row_count: 2,
            },
            right_stats: NodeStats {
                grad_sum: -3.0,
                hess_sum: 2.0,
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
                row_count: 0,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
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
                count: 0,
            };
            num_bins
        ];
        // Category 0: grad=-2.0, hess=2.0 (score = -2/2 = -1.0)
        bins[0] = HistogramBin {
            grad_sum: -2.0,
            hess_sum: 2.0,
            count: 10,
        };
        // Category 1: grad=-1.5, hess=2.0 (score = -1.5/2 = -0.75)
        bins[1] = HistogramBin {
            grad_sum: -1.5,
            hess_sum: 2.0,
            count: 10,
        };
        // Category 2: grad=3.5, hess=2.0 (score = 3.5/2 = 1.75)
        bins[2] = HistogramBin {
            grad_sum: 3.5,
            hess_sum: 2.0,
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
            missing_bin_index: nan_bin,
        };

        let result =
            CpuBackend::best_split_for_categorical_feature(&fh, 0, options, num_cats);
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
    fn test_best_split_categorical_single_populated() {
        // Only 1 category has data -> no valid split possible
        let num_cats = 3;
        let nan_bin = 255usize;
        let num_bins = nan_bin + 1;
        let mut bins = vec![
            HistogramBin {
                grad_sum: 0.0,
                hess_sum: 0.0,
                count: 0,
            };
            num_bins
        ];
        bins[1] = HistogramBin {
            grad_sum: 2.0,
            hess_sum: 5.0,
            count: 20,
        };
        let fh = FeatureHistogram {
            feature_index: 0,
            bins,
        };

        let options = SplitSelectionOptions::default();
        let result =
            CpuBackend::best_split_for_categorical_feature(&fh, 0, options, num_cats);
        assert!(result.is_none(), "single populated category should not split");
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
                row_count: 0,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
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
}
