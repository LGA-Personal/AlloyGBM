//! Integer histogram kernels for quantized gradients.
//!
//! Each node row's quantized `(grad, hess)` levels and a count of one are
//! packed into a single `i64` (see [`QuantizedPacking`]), so accumulating a
//! row into a bin is one integer add where the float kernels do three. The
//! sums are exact, which also makes a sparse feature's dominant bin (node
//! total minus every other bin) exact. Each feature's integer bins are scaled
//! back to floats when they are folded into the arena, so the bundle the
//! split scanners read has the usual layout.

use std::cell::RefCell;
use std::sync::atomic::Ordering;

use alloygbm_core::{
    BinColumn, BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, NodeSlice,
};
use alloygbm_engine::{EngineError, EngineResult, QuantizedGradients, QuantizedPacking};
use rayon::prelude::*;

use crate::arena::HistogramArena;
use crate::{
    CpuBackend, NODE_GENERATION, RowPositions, SPARSE_WALK_COST, THREAD_ARENA, THREAD_ROW_POSITIONS,
};

/// Bin copies the interleaved kernel accumulates into.
const INTERLEAVED_COPIES: usize = 4;

/// Nodes with at least this many rows per bin use the interleaved kernel;
/// below it, zeroing and adding up the extra copies costs more than it saves.
const INTERLEAVE_MIN_ROWS_PER_BIN: usize = 4;

thread_local! {
    /// Per-thread integer bins for the feature being accumulated.
    static THREAD_QUANTIZED_BINS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

/// One node's rows in packed form, shared by every tile.
struct QuantizedNode<'a> {
    /// `values[i]` is the packed triple for `node.row_indices[i]`.
    values: &'a [i64],
    packing: QuantizedPacking,
    grad_scale: f64,
    hess_scale: f64,
    /// Node total and build generation, when some feature has a dominant bin.
    sparse: Option<(i64, u64)>,
}

impl CpuBackend {
    pub(crate) fn build_histograms_quantized_internal(
        binned_matrix: &BinnedMatrix,
        quantized: &QuantizedGradients,
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        parallel_tiles: bool,
    ) -> EngineResult<HistogramBundle> {
        if quantized.len() != binned_matrix.row_count {
            return Err(EngineError::ContractViolation(format!(
                "quantized gradients length {} does not match row_count {}",
                quantized.len(),
                binned_matrix.row_count
            )));
        }
        if feature_tiles.is_empty() {
            return Err(EngineError::ContractViolation(
                "feature_tiles cannot be empty".to_string(),
            ));
        }
        node.validate_bounds(binned_matrix.row_count)?;

        let packed = quantized.packed();
        let integer_kernels =
            binned_matrix.feature_bundle_map().is_none() && binned_matrix.has_col_major();
        let packing = QuantizedPacking::for_rows(
            node.row_indices.len(),
            quantized.bits(),
            quantized.constant_hess(),
        );
        let (true, Some(packing)) = (integer_kernels, packing) else {
            // Bundled or row-major matrices, and nodes too large for one i64
            // per bin, sum the same levels as floats.
            let node_gradients = node
                .row_indices
                .iter()
                .map(|&row| quantized.dequantize(packed[row as usize]))
                .collect::<Vec<GradientPair>>();
            return Self::build_histograms_from_node_gradients(
                binned_matrix,
                &node_gradients,
                node,
                feature_tiles,
                parallel_tiles,
                false,
            );
        };

        let values = node
            .row_indices
            .iter()
            .map(|&row| packing.expand(packed[row as usize]))
            .collect::<Vec<i64>>();
        let sparse = binned_matrix
            .dominant_bins()
            .iter()
            .any(Option::is_some)
            .then(|| {
                (
                    values
                        .iter()
                        .fold(0_i64, |total, &value| total.wrapping_add(value)),
                    NODE_GENERATION.fetch_add(1, Ordering::Relaxed),
                )
            });
        let quantized_node = QuantizedNode {
            values: &values,
            packing,
            grad_scale: f64::from(quantized.grad_scale()),
            hess_scale: f64::from(quantized.hess_scale()),
            sparse,
        };
        let bin_count = binned_matrix.max_bin as usize + 1;
        let total_bins = Self::node_bundle_bins(binned_matrix, feature_tiles, bin_count);
        if parallel_tiles {
            let mut per_tile_histograms = feature_tiles
                .par_iter()
                .map(|tile| {
                    Self::build_quantized_tile_then(
                        binned_matrix,
                        &quantized_node,
                        node,
                        tile,
                        bin_count,
                        |arena, start_feature| {
                            arena
                                .to_bundle(node.node_id, start_feature)
                                .map_err(EngineError::from)
                        },
                    )
                })
                .collect::<Vec<_>>();
            let mut histograms = per_tile_histograms
                .drain(..1)
                .next()
                .expect("feature tiles are non-empty")?;
            histograms.reserve_bins(total_bins.saturating_sub(histograms.total_bin_count()));
            for tile_histograms in per_tile_histograms {
                histograms
                    .append(tile_histograms?)
                    .map_err(EngineError::from)?;
            }
            return Ok(histograms);
        }
        let mut histograms: Option<HistogramBundle> = None;
        for tile in feature_tiles {
            Self::build_quantized_tile_then(
                binned_matrix,
                &quantized_node,
                node,
                tile,
                bin_count,
                |arena, start_feature| {
                    match &mut histograms {
                        Some(bundle) => arena.append_onto(bundle, start_feature)?,
                        None => {
                            let mut bundle = arena.to_bundle(node.node_id, start_feature)?;
                            bundle
                                .reserve_bins(total_bins.saturating_sub(bundle.total_bin_count()));
                            histograms = Some(bundle);
                        }
                    }
                    Ok(())
                },
            )?;
        }
        Ok(histograms.expect("feature tiles are non-empty"))
    }

    fn build_quantized_tile_then<R>(
        binned_matrix: &BinnedMatrix,
        quantized_node: &QuantizedNode<'_>,
        node: &NodeSlice,
        tile: &FeatureTile,
        bin_count: usize,
        finish: impl FnOnce(&HistogramArena, usize) -> EngineResult<R>,
    ) -> EngineResult<R> {
        if tile.end_feature as usize > binned_matrix.feature_count {
            return Err(EngineError::ContractViolation(format!(
                "feature tile end {} exceeds feature_count {}",
                tile.end_feature, binned_matrix.feature_count
            )));
        }
        let start_feature = tile.start_feature as usize;
        let end_feature = tile.end_feature as usize;
        THREAD_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            match binned_matrix.compact_histogram_layout() {
                Some(layout) => {
                    arena.resize_for_compact_tile(layout, start_feature, end_feature, false)
                }
                None => arena.resize_for_tile(end_feature - start_feature, bin_count, false),
            }
            THREAD_QUANTIZED_BINS.with(|bins| {
                let mut bins = bins.borrow_mut();
                for feature in start_feature..end_feature {
                    Self::accumulate_quantized_feature(
                        binned_matrix,
                        quantized_node,
                        node,
                        feature,
                        feature - start_feature,
                        &mut arena,
                        &mut bins,
                    );
                }
            });
            finish(&arena, start_feature)
        })
    }

    fn accumulate_quantized_feature(
        binned_matrix: &BinnedMatrix,
        quantized_node: &QuantizedNode<'_>,
        node: &NodeSlice,
        feature: usize,
        local_feature: usize,
        arena: &mut HistogramArena,
        bins: &mut Vec<i64>,
    ) {
        let base = arena.offsets[local_feature];
        let feature_bins = arena.offsets[local_feature + 1] - base;
        let slot_cap = arena.slot_caps[local_feature];
        bins.clear();
        bins.resize(feature_bins, 0);
        let column = binned_matrix
            .col_bins(feature)
            .expect("integer kernels require unbundled column-major bins");
        let values = quantized_node.values;
        let dominant = quantized_node.sparse.and_then(|(total, generation)| {
            let dominant = binned_matrix
                .dominant_bins()
                .get(feature)
                .copied()
                .flatten()?;
            Some((usize::from(dominant).min(slot_cap), total, generation))
        });
        match dominant {
            Some((skip_slot, total, generation)) => {
                let other_rows = binned_matrix.non_dominant_rows(feature);
                if other_rows.len().saturating_mul(SPARSE_WALK_COST) < node.row_indices.len() {
                    THREAD_ROW_POSITIONS.with(|cell| {
                        let mut positions = cell.borrow_mut();
                        positions.stamp(generation, binned_matrix.row_count, &node.row_indices);
                        match column {
                            BinColumn::U8(column) => accumulate_sparse_rows(
                                column, other_rows, &positions, values, bins, slot_cap,
                            ),
                            BinColumn::U16(column) => accumulate_sparse_rows(
                                column, other_rows, &positions, values, bins, slot_cap,
                            ),
                        }
                    });
                } else {
                    match column {
                        BinColumn::U8(column) => accumulate_column_skipping(
                            column,
                            values,
                            &node.row_indices,
                            bins,
                            slot_cap,
                            skip_slot,
                        ),
                        BinColumn::U16(column) => accumulate_column_skipping(
                            column,
                            values,
                            &node.row_indices,
                            bins,
                            slot_cap,
                            skip_slot,
                        ),
                    }
                }
                // Exact: the skipped slot is the node total less the rest.
                let others = bins
                    .iter()
                    .enumerate()
                    .filter(|&(slot, _)| slot != skip_slot)
                    .fold(0_i64, |sum, (_, &value)| sum.wrapping_add(value));
                bins[skip_slot] = total.wrapping_sub(others);
            }
            None if node.row_indices.len() >= INTERLEAVE_MIN_ROWS_PER_BIN * feature_bins => {
                bins.resize(feature_bins * INTERLEAVED_COPIES, 0);
                match column {
                    BinColumn::U8(column) => accumulate_column_interleaved(
                        column,
                        values,
                        &node.row_indices,
                        bins,
                        feature_bins,
                        slot_cap,
                    ),
                    BinColumn::U16(column) => accumulate_column_interleaved(
                        column,
                        values,
                        &node.row_indices,
                        bins,
                        feature_bins,
                        slot_cap,
                    ),
                }
                let (first, rest) = bins.split_at_mut(feature_bins);
                for copy in rest.chunks_exact(feature_bins) {
                    for (sum, &value) in first.iter_mut().zip(copy) {
                        *sum = sum.wrapping_add(value);
                    }
                }
                bins.truncate(feature_bins);
            }
            None => match column {
                BinColumn::U8(column) => {
                    accumulate_column(column, values, &node.row_indices, bins, slot_cap)
                }
                BinColumn::U16(column) => {
                    accumulate_column(column, values, &node.row_indices, bins, slot_cap)
                }
            },
        }
        // Branch-free so the fold vectorizes: on a node of a few hundred rows
        // it touches as many bins as the row loop touches rows. Empty bins
        // decode to zeros, which is what the arena already holds.
        let (grad_shift, hess_shift, hess_mask, count_mask, hess_per_row) =
            quantized_node.packing.fields();
        let grad_scale = quantized_node.grad_scale as f32;
        let hess_scale = quantized_node.hess_scale as f32;
        let grad_sums = &mut arena.grad_sums[base..base + feature_bins];
        let hess_sums = &mut arena.hess_sums[base..base + feature_bins];
        let counts = &mut arena.counts[base..base + feature_bins];
        for ((&sum, (grad, hess)), count) in bins
            .iter()
            .zip(grad_sums.iter_mut().zip(hess_sums.iter_mut()))
            .zip(counts.iter_mut())
        {
            let rows = sum & count_mask;
            let hess_levels = ((sum >> hess_shift) & hess_mask) + rows * hess_per_row;
            *grad = (sum >> grad_shift) as f32 * grad_scale;
            *hess = hess_levels as f32 * hess_scale;
            *count = rows as u32;
        }
    }
}

#[inline(always)]
fn accumulate_column<T: Copy + Into<usize>>(
    column: &[T],
    values: &[i64],
    row_indices: &[u32],
    bins: &mut [i64],
    slot_cap: usize,
) {
    for (&row, &value) in row_indices.iter().zip(values) {
        let bin: usize = column[row as usize].into();
        let slot = &mut bins[bin.min(slot_cap)];
        *slot = slot.wrapping_add(value);
    }
}

/// [`accumulate_column`] into [`INTERLEAVED_COPIES`] copies of the bins,
/// row `i` going to copy `i % INTERLEAVED_COPIES`. Rows that land in the same
/// bin back to back (common with few bins) otherwise wait on each other's
/// read-modify-write; spread over copies they do not. `bins` holds the copies
/// one after another, `feature_bins` apart, and the caller adds them up.
#[inline(always)]
fn accumulate_column_interleaved<T: Copy + Into<usize>>(
    column: &[T],
    values: &[i64],
    row_indices: &[u32],
    bins: &mut [i64],
    feature_bins: usize,
    slot_cap: usize,
) {
    let (copy0, rest) = bins.split_at_mut(feature_bins);
    let (copy1, rest) = rest.split_at_mut(feature_bins);
    let (copy2, copy3) = rest.split_at_mut(feature_bins);
    let mut rows = row_indices.chunks_exact(INTERLEAVED_COPIES);
    let mut row_values = values.chunks_exact(INTERLEAVED_COPIES);
    for (rows, row_values) in (&mut rows).zip(&mut row_values) {
        let bin = |at: usize| -> usize { column[rows[at] as usize].into().min(slot_cap) };
        let (b0, b1, b2, b3) = (bin(0), bin(1), bin(2), bin(3));
        copy0[b0] = copy0[b0].wrapping_add(row_values[0]);
        copy1[b1] = copy1[b1].wrapping_add(row_values[1]);
        copy2[b2] = copy2[b2].wrapping_add(row_values[2]);
        copy3[b3] = copy3[b3].wrapping_add(row_values[3]);
    }
    accumulate_column(
        column,
        row_values.remainder(),
        rows.remainder(),
        copy0,
        slot_cap,
    );
}

#[inline(always)]
fn accumulate_column_skipping<T: Copy + Into<usize>>(
    column: &[T],
    values: &[i64],
    row_indices: &[u32],
    bins: &mut [i64],
    slot_cap: usize,
    skip_slot: usize,
) {
    for (&row, &value) in row_indices.iter().zip(values) {
        let bin: usize = column[row as usize].into();
        let slot = bin.min(slot_cap);
        if slot != skip_slot {
            bins[slot] = bins[slot].wrapping_add(value);
        }
    }
}

/// Accumulate the node's rows among `other_rows` (one feature's rows outside
/// its dominant bin).
#[inline(always)]
fn accumulate_sparse_rows<T: Copy + Into<usize>>(
    column: &[T],
    other_rows: &[u32],
    positions: &RowPositions,
    values: &[i64],
    bins: &mut [i64],
    slot_cap: usize,
) {
    for &row in other_rows {
        if let Some(position) = positions.position(row) {
            let bin: usize = column[row as usize].into();
            let slot = &mut bins[bin.min(slot_cap)];
            *slot = slot.wrapping_add(values[position]);
        }
    }
}
