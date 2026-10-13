use alloygbm_core::{CompactHistogramLayout, CoreResult, HistogramBundle};

pub(crate) const SMALL_TILE_WORKLOAD_THRESHOLD: usize = 16_384;
pub(crate) const PARALLEL_TILE_WORKLOAD_THRESHOLD: usize = 131_072;
pub(crate) const TINY_NODE_ROW_THRESHOLD: usize = 32;
pub(crate) const BIN_HEAVY_THRESHOLD: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistogramKernelPath {
    TinyNodeScalar,
    BinHeavyPerFeatureScalar,
    ArenaRowFirstUnrolled,
}

/// One bin's running totals, held interleaved.
///
/// The arena keeps its histograms as separate grad / hess / grad_sq / count
/// arrays, which is the layout the split scanners read. Accumulating rows
/// straight into those means every row update writes three or four *distinct*
/// cache lines. Rows go into this interleaved scratch instead -- one line per
/// update -- and are folded into the arena once per feature, which costs
/// O(bins) rather than O(rows).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BinAccumulator {
    pub(crate) grad: f32,
    pub(crate) hess: f32,
    pub(crate) grad_sq: f32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct HistogramArena {
    /// Widest feature's bin count in the current tile (the scratch length).
    pub(crate) bin_count: usize,
    /// `tile_feature_count + 1` start offsets of each feature's bins.
    pub(crate) offsets: Vec<usize>,
    /// Per tile feature, the bin slot a row's bin is clamped to: a compact
    /// feature's missing slot (`data_bins`), so the matrix-wide missing bin
    /// lands there; `bin_count - 1` for a uniform tile, where it is a no-op.
    pub(crate) slot_caps: Vec<usize>,
    /// `Some(edge_limit)` while the tile uses the compact layout.
    pub(crate) compact_edge_limit: Option<usize>,
    pub(crate) grad_sums: Vec<f32>,
    pub(crate) hess_sums: Vec<f32>,
    pub(crate) grad_sq_sums: Option<Vec<f32>>,
    pub(crate) counts: Vec<u32>,
    /// Per-feature interleaved accumulation scratch, `bin_count` long.
    ///
    /// Lives on the arena so it is reused for the life of the thread rather
    /// than reallocated for every tile.
    pub(crate) scratch: Vec<BinAccumulator>,
}

impl HistogramArena {
    pub(crate) fn new(tile_feature_count: usize, bin_count: usize, include_grad_sq: bool) -> Self {
        let mut arena = Self {
            bin_count: 0,
            offsets: Vec::new(),
            slot_caps: Vec::new(),
            compact_edge_limit: None,
            grad_sums: Vec::new(),
            hess_sums: Vec::new(),
            grad_sq_sums: None,
            counts: Vec::new(),
            scratch: Vec::new(),
        };
        arena.resize_for_tile(tile_feature_count, bin_count, include_grad_sq);
        arena
    }

    /// Zero all accumulators without deallocating, allowing the arena to be reused.
    fn reset(&mut self) {
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
        if let Some(values) = &mut self.grad_sq_sums {
            values.fill(0.0);
        }
        self.counts.fill(0);
    }

    /// Resize the arena for a uniform tile (`bin_count` bins per feature)
    /// without unnecessary re-allocation.
    pub(crate) fn resize_for_tile(
        &mut self,
        tile_feature_count: usize,
        bin_count: usize,
        include_grad_sq: bool,
    ) {
        self.offsets.clear();
        self.offsets
            .extend((0..=tile_feature_count).map(|index| index * bin_count));
        self.slot_caps.clear();
        self.slot_caps
            .resize(tile_feature_count, bin_count.saturating_sub(1));
        self.compact_edge_limit = None;
        self.resize_columns(bin_count, include_grad_sq);
    }

    /// Resize the arena for features `start_feature..end_feature` of a matrix
    /// whose histograms use the compact layout.
    pub(crate) fn resize_for_compact_tile(
        &mut self,
        layout: &CompactHistogramLayout,
        start_feature: usize,
        end_feature: usize,
        include_grad_sq: bool,
    ) {
        self.offsets.clear();
        self.slot_caps.clear();
        self.offsets.push(0);
        let mut widest = 0;
        for &data_bins in &layout.data_bins[start_feature..end_feature] {
            let data_bins = usize::from(data_bins);
            let len = data_bins + 1;
            widest = widest.max(len);
            self.slot_caps.push(data_bins);
            self.offsets
                .push(self.offsets.last().expect("non-empty") + len);
        }
        self.compact_edge_limit = Some(layout.edge_limit);
        self.resize_columns(widest, include_grad_sq);
    }

    fn resize_columns(&mut self, bin_count: usize, include_grad_sq: bool) {
        let flat_len = *self.offsets.last().unwrap_or(&0);
        self.bin_count = bin_count;
        if self.grad_sums.len() != flat_len {
            self.grad_sums.resize(flat_len, 0.0);
            self.hess_sums.resize(flat_len, 0.0);
            self.counts.resize(flat_len, 0);
        }
        match (&mut self.grad_sq_sums, include_grad_sq) {
            (Some(values), true) => values.resize(flat_len, 0.0),
            (None, true) => self.grad_sq_sums = Some(vec![0.0; flat_len]),
            (Some(_), false) => self.grad_sq_sums = None,
            (None, false) => {}
        }
        if self.scratch.len() < bin_count {
            self.scratch.resize(bin_count, BinAccumulator::default());
        }
        self.reset();
    }

    /// Flat index of `bin` for tile feature `local_feature`.
    #[inline(always)]
    pub(crate) fn slot(&self, local_feature: usize, bin: usize) -> usize {
        self.offsets[local_feature] + bin.min(self.slot_caps[local_feature])
    }

    /// Append this tile's histograms to `bundle` without an intermediate
    /// bundle; the same result as `bundle.append(self.to_bundle(..))`.
    pub(crate) fn append_onto(
        &self,
        bundle: &mut HistogramBundle,
        start_feature: usize,
    ) -> CoreResult<()> {
        let feature_count = self.offsets.len() - 1;
        bundle.append_parts(
            (start_feature..start_feature + feature_count).map(|feature| feature as u32),
            &self.offsets,
            self.bin_count,
            self.compact_edge_limit,
            &self.grad_sums,
            &self.hess_sums,
            self.grad_sq_sums.as_deref(),
            &self.counts,
        )
    }

    pub(crate) fn to_bundle(
        &self,
        node_id: u32,
        start_feature: usize,
    ) -> CoreResult<HistogramBundle> {
        let feature_count = self.offsets.len() - 1;
        let feature_indices = (start_feature..start_feature + feature_count)
            .map(|feature| feature as u32)
            .collect();
        match self.compact_edge_limit {
            Some(edge_limit) => HistogramBundle::from_soa_compact(
                node_id,
                feature_indices,
                self.offsets.clone(),
                edge_limit,
                self.grad_sums.clone(),
                self.hess_sums.clone(),
                self.grad_sq_sums.clone(),
                self.counts.clone(),
            ),
            None => HistogramBundle::from_soa(
                node_id,
                feature_indices,
                self.bin_count,
                self.grad_sums.clone(),
                self.hess_sums.clone(),
                self.grad_sq_sums.clone(),
                self.counts.clone(),
            ),
        }
    }
}
