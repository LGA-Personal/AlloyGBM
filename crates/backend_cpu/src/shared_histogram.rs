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
}
