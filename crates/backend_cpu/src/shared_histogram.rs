//! K-output shared histogram used by joint multi-label and multiclass DART/GOSS.

#[derive(Debug, Clone, Copy)]
pub enum HistComponent {
    Grad = 0,
    Hess = 1,
}

#[derive(Debug, Clone)]
pub struct MultiOutputHistogram {
    pub n_features: usize,
    pub n_bins: usize,
    pub n_outputs: usize,
    /// Flat storage. Layout: feature-major → bin-major → output-major →
    /// (grad, hess) interleaved. Index helper: `idx(f, b, k, comp)`.
    data: Vec<f32>,
}

impl MultiOutputHistogram {
    pub fn new(n_features: usize, n_bins: usize, n_outputs: usize) -> Self {
        let n = n_features * n_bins * n_outputs * 2;
        Self {
            n_features,
            n_bins,
            n_outputs,
            data: vec![0.0_f32; n],
        }
    }

    #[inline]
    pub fn idx(&self, feature: usize, bin: usize, output: usize, comp: HistComponent) -> usize {
        debug_assert!(feature < self.n_features);
        debug_assert!(bin < self.n_bins);
        debug_assert!(output < self.n_outputs);
        ((feature * self.n_bins + bin) * self.n_outputs + output) * 2 + comp as usize
    }

    pub fn len_flat(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
}

/// Build a multi-output histogram for a single feature column in one sweep.
///
/// `grads` and `hess` are row-major with output as the inner axis:
/// `grads[row * n_outputs + k]` is the gradient for row `row`, output `k`.
/// Length must equal `bins.len() * n_outputs`.
pub fn build_multi_output_histogram_inplace(
    histogram: &mut MultiOutputHistogram,
    feature: usize,
    bins: &[u8],
    grads: &[f32],
    hess: &[f32],
    n_outputs: usize,
) {
    debug_assert_eq!(n_outputs, histogram.n_outputs);
    debug_assert_eq!(grads.len(), bins.len() * n_outputs);
    debug_assert_eq!(hess.len(), bins.len() * n_outputs);

    let n_bins = histogram.n_bins;
    let stride = histogram.n_outputs * 2;
    // Slab for this feature; outputs are the inner-most dimension.
    let feature_offset = feature * n_bins * stride;

    for (row, &bin) in bins.iter().enumerate() {
        let bin = bin as usize;
        debug_assert!(bin < n_bins);
        let bin_offset = feature_offset + bin * stride;
        for k in 0..n_outputs {
            let g = grads[row * n_outputs + k];
            let h = hess[row * n_outputs + k];
            let pair_offset = bin_offset + k * 2;
            histogram.data[pair_offset] += g;
            histogram.data[pair_offset + 1] += h;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_output_histogram_layout_is_feature_bin_output_major() {
        // (n_features=2, n_bins=4, n_outputs=3)
        let h = MultiOutputHistogram::new(2, 4, 3);
        assert_eq!(h.len_flat(), 2 * 4 * 3 * 2); // *2 for (grad, hess)
        // Index for (feature=1, bin=2, output=0, GRAD) should be unique
        let idx_g = h.idx(1, 2, 0, HistComponent::Grad);
        let idx_h = h.idx(1, 2, 0, HistComponent::Hess);
        assert_ne!(idx_g, idx_h);
        assert!(idx_g < h.len_flat() && idx_h < h.len_flat());
    }

    #[test]
    fn build_kernel_accumulates_per_output_grad_hess() {
        // 3 rows, 1 feature, 4 bins (incl. missing=3), 2 outputs.
        let bins: Vec<u8> = vec![0, 1, 0]; // row → bin
        // grads/hess interleaved per output: [g0_r0, g1_r0, g0_r1, g1_r1, g0_r2, g1_r2]
        let grads = [1.0_f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let hess = [0.1_f32, 1.0, 0.2, 2.0, 0.3, 3.0];

        let mut h = MultiOutputHistogram::new(1, 4, 2);
        build_multi_output_histogram_inplace(
            &mut h, /*feature=*/ 0, &bins, &grads, &hess, /*n_outputs=*/ 2,
        );

        // Output 0, bin 0 should aggregate rows 0+2 → g=4.0, h=0.4
        let i = h.idx(0, 0, 0, HistComponent::Grad);
        assert!((h.data()[i] - 4.0).abs() < 1e-6);
        let i = h.idx(0, 0, 0, HistComponent::Hess);
        assert!((h.data()[i] - 0.4).abs() < 1e-6);

        // Output 1, bin 1 should aggregate row 1 only → g=20.0, h=2.0
        let i = h.idx(0, 1, 1, HistComponent::Grad);
        assert!((h.data()[i] - 20.0).abs() < 1e-6);
    }
}
