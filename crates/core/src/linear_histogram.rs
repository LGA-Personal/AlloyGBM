/// Maximum number of regressor features per leaf for piecewise-linear trees.
/// Matches `min(8, max_depth)` at run time, but we always pad to this constant
/// so that bin layouts are fixed-size and cache-friendly.
pub const MAX_PL_REGRESSORS: usize = 8;

/// Number of `XᵀHX` matrix entries stored per histogram bin: a padded 8×8 = 64
/// stride-8 layout (changed from a 36-entry compacted upper-triangle in v0.5.0
/// so that all matrix operations map cleanly to `wide::f32x8` lanes with no
/// scalar tail). The lower-triangle entries stay zero (`XᵀHX` is symmetric and
/// only the upper triangle is populated by `pl_histogram`); SIMD operations on
/// those zero slots are harmless.
pub const MAX_PL_MATRIX_ENTRIES: usize = MAX_PL_REGRESSORS * MAX_PL_REGRESSORS;

/// A single histogram bin for a piecewise-linear (PL) leaf model.
///
/// Stores the `(Xᵀg, XᵀHX)` statistics needed for the closed-form ridge-
/// regression leaf-weight solve `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
///
/// Only the first `d` entries of `xtg` and only the upper-triangle of the
/// `d × d` block of `xt_hx` are written by the histogram builder; the rest of
/// each array is zero padding (so SIMD operations covering the full storage
/// produce mathematically correct results).  `d` is recorded in the parent
/// [`LinearHistogramBundle`].
///
/// `xt_hx` uses a stride-8 row-major layout: `xt_hx[j * MAX_PL_REGRESSORS + k]`
/// holds `Σ h_i x_{i,j} x_{i,k}` for `j ≤ k < d`.  See [`pl_matrix_index`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearHistogramBin {
    /// Sum of gradients for samples in this bin.
    pub grad_sum: f32,
    /// Sum of hessians for samples in this bin.
    pub hess_sum: f32,
    /// Number of samples in this bin.
    pub count: u32,
    /// `Xᵀg` vector: `xtg[j] = Σ_{i in bin} g_i * x_{i, regressor_j}` for j = 0..d.
    pub xtg: [f32; MAX_PL_REGRESSORS],
    /// `XᵀHX` stride-8 row-major: `xt_hx[j * MAX_PL_REGRESSORS + k]` for j ≤ k < d.
    pub xt_hx: [f32; MAX_PL_MATRIX_ENTRIES],
}

impl Default for LinearHistogramBin {
    fn default() -> Self {
        Self {
            grad_sum: 0.0,
            hess_sum: 0.0,
            count: 0,
            xtg: [0.0; MAX_PL_REGRESSORS],
            xt_hx: [0.0; MAX_PL_MATRIX_ENTRIES],
        }
    }
}

/// Return the flat index into the stride-8 row-major `xt_hx` array for element
/// `(j, k)`.
///
/// The histogram builder writes only the upper triangle (`j ≤ k < d`); lower-
/// triangle entries stay zero.  Callers may also reference `(j, k)` pairs in
/// the lower triangle for symmetric reads — those slots are zero and a SIMD
/// reduction over the full row produces the correct result.
///
/// Panics in debug builds if either index is out of range.
#[inline]
pub fn pl_matrix_index(j: usize, k: usize) -> usize {
    debug_assert!(j < MAX_PL_REGRESSORS, "j ({j}) must be < MAX_PL_REGRESSORS");
    debug_assert!(k < MAX_PL_REGRESSORS, "k ({k}) must be < MAX_PL_REGRESSORS");
    j * MAX_PL_REGRESSORS + k
}

/// Per-feature histogram for piecewise-linear leaf statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFeatureHistogram {
    pub feature_index: u32,
    pub bins: Vec<LinearHistogramBin>,
}

/// Bundle of per-feature PL histograms for a single tree node.
///
/// `num_regressors` tells how many entries in each bin's `xtg` / `xt_hx`
/// fields are valid; the rest are zero padding.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearHistogramBundle {
    pub node_id: u32,
    /// Number of active regressors for this node (1 ≤ d ≤ MAX_PL_REGRESSORS).
    pub num_regressors: usize,
    /// Which feature indices are the regressors for this node, length = num_regressors.
    pub regressor_features: Vec<u32>,
    pub feature_histograms: Vec<LinearFeatureHistogram>,
}

impl LinearHistogramBundle {
    /// Zero all bin values in-place without deallocating.
    pub fn reset(&mut self, node_id: u32) {
        self.node_id = node_id;
        for fh in &mut self.feature_histograms {
            for bin in &mut fh.bins {
                *bin = LinearHistogramBin::default();
            }
        }
    }

    /// Create a pre-allocated, zeroed linear histogram bundle.
    pub fn new_zeroed(
        node_id: u32,
        feature_indices: &[u32],
        bin_count: usize,
        num_regressors: usize,
        regressor_features: Vec<u32>,
    ) -> Self {
        debug_assert!(
            num_regressors <= MAX_PL_REGRESSORS,
            "num_regressors ({num_regressors}) exceeds MAX_PL_REGRESSORS"
        );
        let feature_histograms = feature_indices
            .iter()
            .map(|&fi| LinearFeatureHistogram {
                feature_index: fi,
                bins: vec![LinearHistogramBin::default(); bin_count],
            })
            .collect();
        Self {
            node_id,
            num_regressors,
            regressor_features,
            feature_histograms,
        }
    }
}

/// Compute the larger-child linear histogram bundle using the subtraction trick.
///
/// Given `parent` and the `smaller` child's bundle (both with the same node
/// layout), returns the `larger` child bundle via element-wise subtraction:
/// `larger[f][b] = parent[f][b] - smaller[f][b]` for all fields.
///
/// Both bundles must have the same number of features (in the same order) and
/// the same number of bins per feature.
pub fn subtract_linear_histogram_bundle(
    parent: &LinearHistogramBundle,
    smaller: &LinearHistogramBundle,
) -> LinearHistogramBundle {
    debug_assert_eq!(
        parent.feature_histograms.len(),
        smaller.feature_histograms.len(),
        "feature count mismatch in linear histogram subtraction"
    );
    debug_assert_eq!(
        parent.num_regressors, smaller.num_regressors,
        "num_regressors mismatch in linear histogram subtraction"
    );
    let feature_histograms = parent
        .feature_histograms
        .iter()
        .zip(smaller.feature_histograms.iter())
        .map(|(pfh, sfh)| {
            debug_assert_eq!(pfh.bins.len(), sfh.bins.len());
            let bins = pfh
                .bins
                .iter()
                .zip(sfh.bins.iter())
                .map(|(pb, sb)| {
                    // Operate on all `MAX_PL_REGRESSORS` / `MAX_PL_MATRIX_ENTRIES`
                    // slots — the histogram builder may populate both triangles
                    // of `xt_hx` (full 8×8 outer product) and unused entries
                    // are zero in both `pb` and `sb`, so subtracting all
                    // entries is correct and matches the SIMD-friendly
                    // stride-8 layout.
                    let mut xtg = [0.0f32; MAX_PL_REGRESSORS];
                    for (j, xtg_j) in xtg.iter_mut().enumerate() {
                        *xtg_j = pb.xtg[j] - sb.xtg[j];
                    }
                    let mut xt_hx = [0.0f32; MAX_PL_MATRIX_ENTRIES];
                    for (i, slot) in xt_hx.iter_mut().enumerate() {
                        *slot = pb.xt_hx[i] - sb.xt_hx[i];
                    }
                    LinearHistogramBin {
                        grad_sum: pb.grad_sum - sb.grad_sum,
                        hess_sum: pb.hess_sum - sb.hess_sum,
                        count: pb.count - sb.count,
                        xtg,
                        xt_hx,
                    }
                })
                .collect();
            LinearFeatureHistogram {
                feature_index: pfh.feature_index,
                bins,
            }
        })
        .collect();
    LinearHistogramBundle {
        node_id: smaller.node_id, // larger child gets a different node_id assigned by caller
        num_regressors: parent.num_regressors,
        regressor_features: parent.regressor_features.clone(),
        feature_histograms,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearFeatureScaler {
    means: Vec<f32>,
    inv_stds: Vec<f32>,
}

impl LinearFeatureScaler {
    pub fn identity(feature_count: usize) -> Self {
        Self {
            means: vec![0.0; feature_count],
            inv_stds: vec![1.0; feature_count],
        }
    }

    pub fn from_raw_matrix(raw: &[f32], row_count: usize, feature_count: usize) -> Self {
        if row_count == 0 || feature_count == 0 || raw.len() < row_count * feature_count {
            return Self::identity(feature_count);
        }

        let mut sums = vec![0.0_f64; feature_count];
        let mut counts = vec![0_u64; feature_count];
        for row in 0..row_count {
            let base = row * feature_count;
            for feature in 0..feature_count {
                let value = raw[base + feature];
                if value.is_finite() {
                    sums[feature] += value as f64;
                    counts[feature] += 1;
                }
            }
        }

        let mut means = vec![0.0_f32; feature_count];
        for feature in 0..feature_count {
            if counts[feature] > 0 {
                means[feature] = (sums[feature] / counts[feature] as f64) as f32;
            }
        }

        let mut sumsq = vec![0.0_f64; feature_count];
        for row in 0..row_count {
            let base = row * feature_count;
            for feature in 0..feature_count {
                let value = raw[base + feature];
                if value.is_finite() {
                    let centered = value as f64 - means[feature] as f64;
                    sumsq[feature] += centered * centered;
                }
            }
        }

        let mut inv_stds = vec![1.0_f32; feature_count];
        for feature in 0..feature_count {
            if counts[feature] > 1 {
                let variance = sumsq[feature] / counts[feature] as f64;
                let std = variance.sqrt();
                if std.is_finite() && std > 1e-12 {
                    inv_stds[feature] = (1.0 / std) as f32;
                }
            }
        }

        Self { means, inv_stds }
    }

    #[inline]
    pub fn mean(&self, feature_index: u32) -> f32 {
        self.means
            .get(feature_index as usize)
            .copied()
            .unwrap_or(0.0)
    }

    #[inline]
    pub fn inv_std(&self, feature_index: u32) -> f32 {
        self.inv_stds
            .get(feature_index as usize)
            .copied()
            .unwrap_or(1.0)
    }

    #[inline]
    pub fn scaled_value(&self, feature_index: u32, raw_value: f32) -> f32 {
        if !raw_value.is_finite() {
            return 0.0;
        }
        (raw_value - self.mean(feature_index)) * self.inv_std(feature_index)
    }

    pub fn leaf_means_and_inv_stds(&self, regressor_features: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let means = regressor_features
            .iter()
            .map(|&feature| self.mean(feature))
            .collect();
        let inv_stds = regressor_features
            .iter()
            .map(|&feature| self.inv_std(feature))
            .collect();
        (means, inv_stds)
    }
}
