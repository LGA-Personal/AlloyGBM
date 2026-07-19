use alloygbm_core::{CoreResult, HistogramBundle};

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

#[derive(Debug, Clone)]
pub(crate) struct HistogramArena {
    pub(crate) bin_count: usize,
    pub(crate) grad_sums: Vec<f32>,
    pub(crate) hess_sums: Vec<f32>,
    pub(crate) grad_sq_sums: Option<Vec<f32>>,
    pub(crate) counts: Vec<u32>,
}

impl HistogramArena {
    pub(crate) fn new(tile_feature_count: usize, bin_count: usize, include_grad_sq: bool) -> Self {
        let flat_len = tile_feature_count * bin_count;
        Self {
            bin_count,
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            grad_sq_sums: include_grad_sq.then(|| vec![0.0; flat_len]),
            counts: vec![0; flat_len],
        }
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

    /// Resize the arena to handle a new tile size without unnecessary re-allocation.
    /// Only reallocates if the new tile requires more capacity.
    pub(crate) fn resize_for_tile(
        &mut self,
        tile_feature_count: usize,
        bin_count: usize,
        include_grad_sq: bool,
    ) {
        let flat_len = tile_feature_count * bin_count;
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
        self.reset();
    }

    pub(crate) fn to_bundle(
        &self,
        node_id: u32,
        start_feature: usize,
    ) -> CoreResult<HistogramBundle> {
        let feature_count = self.grad_sums.len() / self.bin_count;
        HistogramBundle::from_soa(
            node_id,
            (start_feature..start_feature + feature_count)
                .map(|feature| feature as u32)
                .collect(),
            self.bin_count,
            self.grad_sums.clone(),
            self.hess_sums.clone(),
            self.grad_sq_sums.clone(),
            self.counts.clone(),
        )
    }
}
