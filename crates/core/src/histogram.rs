use crate::dro::DroConfig;
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientPair {
    pub grad: f32,
    pub hess: f32,
}

impl GradientPair {
    pub fn new(grad: f32, hess: f32) -> CoreResult<Self> {
        if !grad.is_finite() || !hess.is_finite() {
            return Err(CoreError::Validation(
                "gradient and hessian must be finite".to_string(),
            ));
        }
        if hess <= 0.0 {
            return Err(CoreError::Validation(
                "hessian must be greater than 0".to_string(),
            ));
        }
        Ok(Self { grad, hess })
    }
}

pub fn leaf_effective_gradient(
    grad_sum: f32,
    grad_sq_sum: f32,
    row_count: u32,
    l1_alpha: f32,
    dro_config: Option<&DroConfig>,
) -> f32 {
    let mut threshold = l1_alpha.max(0.0);
    if let Some(cfg) = dro_config
        && cfg.radius > 0.0
    {
        let n = row_count.max(1) as f64;
        let mean = f64::from(grad_sum) / n;
        let variance = (f64::from(grad_sq_sum) / n - mean * mean).max(0.0);
        threshold += (f64::from(cfg.radius) * n.sqrt() * variance.sqrt()) as f32;
    }
    if grad_sum > threshold {
        grad_sum - threshold
    } else if grad_sum < -threshold {
        grad_sum + threshold
    } else {
        0.0
    }
}

pub fn leaf_gain_term(
    grad_sum: f32,
    hess_sum: f32,
    grad_sq_sum: f32,
    row_count: u32,
    l1_alpha: f32,
    l2_lambda: f32,
    dro_config: Option<&DroConfig>,
) -> f32 {
    const EPSILON: f32 = 1e-6;
    let effective = leaf_effective_gradient(grad_sum, grad_sq_sum, row_count, l1_alpha, dro_config);
    0.5 * effective * effective / (hess_sum + l2_lambda + EPSILON)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureTile {
    pub start_feature: u32,
    pub end_feature: u32,
}

impl FeatureTile {
    pub fn new(start_feature: u32, end_feature: u32) -> CoreResult<Self> {
        if start_feature >= end_feature {
            return Err(CoreError::Validation(
                "feature tile must satisfy start_feature < end_feature".to_string(),
            ));
        }
        Ok(Self {
            start_feature,
            end_feature,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSlice {
    pub node_id: u32,
    pub row_indices: Vec<u32>,
}

impl NodeSlice {
    pub fn new(node_id: u32, row_indices: Vec<u32>) -> CoreResult<Self> {
        if row_indices.is_empty() {
            return Err(CoreError::Validation(
                "node row_indices cannot be empty".to_string(),
            ));
        }
        Ok(Self {
            node_id,
            row_indices,
        })
    }

    pub fn validate_bounds(&self, row_count: usize) -> CoreResult<()> {
        for &row_index in &self.row_indices {
            let row_index = row_index as usize;
            if row_index >= row_count {
                return Err(CoreError::Validation(format!(
                    "row index {row_index} is out of bounds for row_count {row_count}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub grad_sq_sum: f32,
    pub row_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBin {
    pub grad_sum: f32,
    pub hess_sum: f32,
    pub grad_sq_sum: f32,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureHistogram {
    pub feature_index: u32,
    pub bins: Vec<HistogramBin>,
}

#[derive(Debug, Clone, Copy)]
pub struct HistogramFeatureView<'a> {
    feature_index: u32,
    grad_sums: &'a [f32],
    hess_sums: &'a [f32],
    grad_sq_sums: Option<&'a [f32]>,
    counts: &'a [u32],
    /// `Some(edge_limit)` for a compact view (see [`HistogramBundle`]); its
    /// last bin is the missing bin.
    compact_edge_limit: Option<usize>,
}

impl<'a> HistogramFeatureView<'a> {
    pub fn feature_index(self) -> u32 {
        self.feature_index
    }

    pub fn len(self) -> usize {
        self.grad_sums.len()
    }

    pub fn is_empty(self) -> bool {
        self.grad_sums.is_empty()
    }

    pub fn grad_sums(self) -> &'a [f32] {
        self.grad_sums
    }

    pub fn hess_sums(self) -> &'a [f32] {
        self.hess_sums
    }

    pub fn grad_sq_sums(self) -> Option<&'a [f32]> {
        self.grad_sq_sums
    }

    pub fn counts(self) -> &'a [u32] {
        self.counts
    }

    /// Index of this feature's missing-value bin within the view.
    ///
    /// A compact view stores the missing bin last. A uniform view keeps it at
    /// the matrix-wide index the caller passes (`options.missing_bin_index`),
    /// which may lie past the end when the histogram has no missing bin.
    #[inline]
    pub fn missing_index(self, uniform_missing_bin: usize) -> usize {
        match self.compact_edge_limit {
            Some(_) => self.len().saturating_sub(1),
            None => uniform_missing_bin,
        }
    }

    /// The scan limit the "last threshold" rule compares against.
    ///
    /// Split scans skip a threshold whose left side already holds every
    /// non-missing row when `threshold + 1 >= limit`. For a uniform view the
    /// limit is the view's own scan limit; a compact view drops the empty bins
    /// between its last data bin and the missing bin, so it carries the limit
    /// its uniform layout would have had, which keeps the rule -- and the
    /// chosen split -- unchanged.
    #[inline]
    pub fn edge_scan_limit(self, scan_limit: usize) -> usize {
        self.compact_edge_limit.unwrap_or(scan_limit)
    }

    pub fn is_compact(self) -> bool {
        self.compact_edge_limit.is_some()
    }

    pub fn bin(self, index: usize) -> Option<HistogramBin> {
        Some(HistogramBin {
            grad_sum: *self.grad_sums.get(index)?,
            hess_sum: self.hess_sums[index],
            grad_sq_sum: self.grad_sq_sums.map_or(0.0, |values| values[index]),
            count: self.counts[index],
        })
    }

    pub fn bins(self) -> impl ExactSizeIterator<Item = HistogramBin> + 'a {
        (0..self.len()).map(move |index| HistogramBin {
            grad_sum: self.grad_sums[index],
            hess_sum: self.hess_sums[index],
            grad_sq_sum: self.grad_sq_sums.map_or(0.0, |values| values[index]),
            count: self.counts[index],
        })
    }
}

/// Per-node gradient histograms for a set of features, stored as SoA columns.
///
/// Two layouts share the flat columns:
///
/// - **Uniform**: every feature owns `bin_count` consecutive bins and the
///   missing bin sits at the matrix-wide missing index.
/// - **Compact**: feature `i` owns `offsets[i]..offsets[i + 1]`, sized to the
///   bins its data actually uses plus one trailing missing bin. A feature with
///   two distinct values costs three bins instead of 256 (or 65,536 for wide
///   bins), which is what makes wide, low-cardinality data cheap to histogram,
///   subtract and scan. Data bin `b` keeps index `b`, so split thresholds mean
///   the same thing in both layouts.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBundle {
    pub node_id: u32,
    feature_indices: Vec<u32>,
    bin_count: usize,
    /// `feature_count + 1` start offsets into the flat columns.
    offsets: Vec<usize>,
    /// `Some(limit)` for the compact layout; see
    /// [`HistogramFeatureView::edge_scan_limit`].
    compact_edge_limit: Option<usize>,
    grad_sums: Vec<f32>,
    hess_sums: Vec<f32>,
    grad_sq_sums: Option<Vec<f32>>,
    counts: Vec<u32>,
}

impl HistogramBundle {
    #[allow(clippy::too_many_arguments)]
    pub fn from_soa(
        node_id: u32,
        feature_indices: Vec<u32>,
        bin_count: usize,
        grad_sums: Vec<f32>,
        hess_sums: Vec<f32>,
        grad_sq_sums: Option<Vec<f32>>,
        counts: Vec<u32>,
    ) -> CoreResult<Self> {
        let expected_len = feature_indices
            .len()
            .checked_mul(bin_count)
            .ok_or_else(|| {
                CoreError::Validation("histogram dimensions overflow usize".to_string())
            })?;
        if grad_sums.len() != expected_len
            || hess_sums.len() != expected_len
            || counts.len() != expected_len
            || grad_sq_sums
                .as_ref()
                .is_some_and(|values| values.len() != expected_len)
        {
            return Err(CoreError::Validation(format!(
                "histogram SoA columns must all have length {expected_len}"
            )));
        }
        let offsets = uniform_offsets(feature_indices.len(), bin_count);
        Ok(Self {
            node_id,
            feature_indices,
            bin_count,
            offsets,
            compact_edge_limit: None,
            grad_sums,
            hess_sums,
            grad_sq_sums,
            counts,
        })
    }

    /// Build a compact-layout bundle (see the type docs). `offsets` has one
    /// entry per feature plus a final end offset; each feature needs at least
    /// one bin (its missing bin).
    #[allow(clippy::too_many_arguments)]
    pub fn from_soa_compact(
        node_id: u32,
        feature_indices: Vec<u32>,
        offsets: Vec<usize>,
        edge_limit: usize,
        grad_sums: Vec<f32>,
        hess_sums: Vec<f32>,
        grad_sq_sums: Option<Vec<f32>>,
        counts: Vec<u32>,
    ) -> CoreResult<Self> {
        if offsets.len() != feature_indices.len() + 1
            || offsets.first() != Some(&0)
            || offsets.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(CoreError::Validation(
                "compact histogram offsets must start at 0 and give every feature a bin"
                    .to_string(),
            ));
        }
        let expected_len = *offsets.last().expect("offsets are non-empty");
        if grad_sums.len() != expected_len
            || hess_sums.len() != expected_len
            || counts.len() != expected_len
            || grad_sq_sums
                .as_ref()
                .is_some_and(|values| values.len() != expected_len)
        {
            return Err(CoreError::Validation(format!(
                "histogram SoA columns must all have length {expected_len}"
            )));
        }
        let bin_count = offsets
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .unwrap_or(0);
        Ok(Self {
            node_id,
            feature_indices,
            bin_count,
            offsets,
            compact_edge_limit: Some(edge_limit),
            grad_sums,
            hess_sums,
            grad_sq_sums,
            counts,
        })
    }

    pub fn from_feature_histograms(
        node_id: u32,
        feature_histograms: Vec<FeatureHistogram>,
        include_grad_sq: bool,
    ) -> CoreResult<Self> {
        let bin_count = feature_histograms
            .first()
            .map_or(0, |feature| feature.bins.len());
        if feature_histograms
            .iter()
            .any(|feature| feature.bins.len() != bin_count)
        {
            return Err(CoreError::Validation(
                "all feature histograms must have the same bin count".to_string(),
            ));
        }
        let flat_len = feature_histograms.len() * bin_count;
        let mut feature_indices = Vec::with_capacity(feature_histograms.len());
        let mut grad_sums = Vec::with_capacity(flat_len);
        let mut hess_sums = Vec::with_capacity(flat_len);
        let mut grad_sq_sums = include_grad_sq.then(|| Vec::with_capacity(flat_len));
        let mut counts = Vec::with_capacity(flat_len);
        for feature in feature_histograms {
            feature_indices.push(feature.feature_index);
            for bin in feature.bins {
                grad_sums.push(bin.grad_sum);
                hess_sums.push(bin.hess_sum);
                if let Some(values) = &mut grad_sq_sums {
                    values.push(bin.grad_sq_sum);
                }
                counts.push(bin.count);
            }
        }
        Self::from_soa(
            node_id,
            feature_indices,
            bin_count,
            grad_sums,
            hess_sums,
            grad_sq_sums,
            counts,
        )
    }

    pub fn feature_count(&self) -> usize {
        self.feature_indices.len()
    }

    /// Bins per feature for a uniform bundle; the widest feature's bin count
    /// for a compact one.
    pub fn bin_count(&self) -> usize {
        self.bin_count
    }

    pub fn is_compact(&self) -> bool {
        self.compact_edge_limit.is_some()
    }

    /// Total bins across every feature (the length of each SoA column).
    pub fn total_bin_count(&self) -> usize {
        self.grad_sums.len()
    }

    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    pub fn has_grad_sq_sums(&self) -> bool {
        self.grad_sq_sums.is_some()
    }

    pub fn feature(&self, index: usize) -> Option<HistogramFeatureView<'_>> {
        let feature_index = *self.feature_indices.get(index)?;
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        Some(HistogramFeatureView {
            feature_index,
            grad_sums: &self.grad_sums[start..end],
            hess_sums: &self.hess_sums[start..end],
            grad_sq_sums: self.grad_sq_sums.as_ref().map(|values| &values[start..end]),
            counts: &self.counts[start..end],
            compact_edge_limit: self.compact_edge_limit,
        })
    }

    pub fn features(&self) -> impl ExactSizeIterator<Item = HistogramFeatureView<'_>> {
        (0..self.feature_count()).map(|index| {
            self.feature(index)
                .expect("feature index is bounded by feature_count")
        })
    }

    pub fn set_bin(
        &mut self,
        feature_index: usize,
        bin_index: usize,
        bin: HistogramBin,
    ) -> CoreResult<()> {
        if feature_index >= self.feature_count()
            || bin_index >= self.offsets[feature_index + 1] - self.offsets[feature_index]
        {
            return Err(CoreError::Validation(
                "histogram feature or bin index is out of bounds".to_string(),
            ));
        }
        let index = self.offsets[feature_index] + bin_index;
        self.grad_sums[index] = bin.grad_sum;
        self.hess_sums[index] = bin.hess_sum;
        if let Some(values) = &mut self.grad_sq_sums {
            values[index] = bin.grad_sq_sum;
        }
        self.counts[index] = bin.count;
        Ok(())
    }

    pub fn append(&mut self, other: Self) -> CoreResult<()> {
        if self.compact_edge_limit != other.compact_edge_limit
            || (self.compact_edge_limit.is_none() && self.bin_count != other.bin_count)
        {
            return Err(CoreError::Validation(
                "cannot append histograms with different bin layouts".to_string(),
            ));
        }
        if self.has_grad_sq_sums() != other.has_grad_sq_sums() {
            return Err(CoreError::Validation(
                "cannot append histograms with different squared-gradient layouts".to_string(),
            ));
        }
        let base = self.grad_sums.len();
        self.offsets
            .extend(other.offsets[1..].iter().map(|&offset| base + offset));
        self.bin_count = self.bin_count.max(other.bin_count);
        self.feature_indices.extend(other.feature_indices);
        self.grad_sums.extend(other.grad_sums);
        self.hess_sums.extend(other.hess_sums);
        if let (Some(dest), Some(source)) = (&mut self.grad_sq_sums, other.grad_sq_sums) {
            dest.extend(source);
        }
        self.counts.extend(other.counts);
        Ok(())
    }

    pub fn filtered(&self, is_allowed: impl Fn(u32) -> bool) -> Self {
        let selected: Vec<usize> = self
            .feature_indices
            .iter()
            .enumerate()
            .filter_map(|(index, &feature)| is_allowed(feature).then_some(index))
            .collect();
        let mut offsets = Vec::with_capacity(selected.len() + 1);
        offsets.push(0);
        for &source_feature in &selected {
            let len = self.offsets[source_feature + 1] - self.offsets[source_feature];
            offsets.push(offsets.last().expect("non-empty") + len);
        }
        let flat_len = *offsets.last().expect("non-empty");
        let mut result = Self {
            node_id: self.node_id,
            feature_indices: selected
                .iter()
                .map(|&index| self.feature_indices[index])
                .collect(),
            bin_count: self.bin_count,
            offsets,
            compact_edge_limit: self.compact_edge_limit,
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            grad_sq_sums: self.grad_sq_sums.as_ref().map(|_| vec![0.0; flat_len]),
            counts: vec![0; flat_len],
        };
        for (dest_feature, &source_feature) in selected.iter().enumerate() {
            let source_start = self.offsets[source_feature];
            let source_end = self.offsets[source_feature + 1];
            let dest_start = result.offsets[dest_feature];
            let dest_end = result.offsets[dest_feature + 1];
            result.grad_sums[dest_start..dest_end]
                .copy_from_slice(&self.grad_sums[source_start..source_end]);
            result.hess_sums[dest_start..dest_end]
                .copy_from_slice(&self.hess_sums[source_start..source_end]);
            result.counts[dest_start..dest_end]
                .copy_from_slice(&self.counts[source_start..source_end]);
            if let (Some(dest), Some(source)) = (&mut result.grad_sq_sums, &self.grad_sq_sums) {
                dest[dest_start..dest_end].copy_from_slice(&source[source_start..source_end]);
            }
        }
        result
    }

    fn validate_subtraction_layout(
        parent: &Self,
        child: &Self,
        destination: &Self,
    ) -> CoreResult<()> {
        if parent.feature_indices != child.feature_indices
            || destination.feature_indices != parent.feature_indices
            || parent.offsets != child.offsets
            || destination.offsets != parent.offsets
            || parent.compact_edge_limit != child.compact_edge_limit
            || destination.compact_edge_limit != parent.compact_edge_limit
            || parent.has_grad_sq_sums() != child.has_grad_sq_sums()
            || destination.has_grad_sq_sums() != parent.has_grad_sq_sums()
        {
            return Err(CoreError::Validation(
                "histogram subtraction requires identical layouts".to_string(),
            ));
        }
        Ok(())
    }

    pub fn subtract_into(&mut self, parent: &Self, child: &Self, node_id: u32) -> CoreResult<()> {
        Self::validate_subtraction_layout(parent, child, self)?;
        if parent
            .counts
            .iter()
            .zip(&child.counts)
            .any(|(&parent, &child)| child > parent)
        {
            return Err(CoreError::Validation(
                "child histogram count exceeds parent".to_string(),
            ));
        }
        self.node_id = node_id;
        for ((dest, &parent), &child) in self
            .grad_sums
            .iter_mut()
            .zip(&parent.grad_sums)
            .zip(&child.grad_sums)
        {
            *dest = parent - child;
        }
        for ((dest, &parent), &child) in self
            .hess_sums
            .iter_mut()
            .zip(&parent.hess_sums)
            .zip(&child.hess_sums)
        {
            *dest = parent - child;
        }
        if let (Some(dest), Some(parent), Some(child)) = (
            &mut self.grad_sq_sums,
            &parent.grad_sq_sums,
            &child.grad_sq_sums,
        ) {
            for ((dest, &parent), &child) in dest.iter_mut().zip(parent).zip(child) {
                *dest = parent - child;
            }
        }
        for ((dest, &parent), &child) in self
            .counts
            .iter_mut()
            .zip(&parent.counts)
            .zip(&child.counts)
        {
            *dest = parent - child;
        }
        Ok(())
    }

    pub fn subtract_child_in_place(&mut self, child: &Self, node_id: u32) -> CoreResult<()> {
        Self::validate_subtraction_layout(self, child, self)?;
        if self
            .counts
            .iter()
            .zip(&child.counts)
            .any(|(&parent, &child)| child > parent)
        {
            return Err(CoreError::Validation(
                "child histogram count exceeds parent".to_string(),
            ));
        }
        self.node_id = node_id;
        for (parent, &child) in self.grad_sums.iter_mut().zip(&child.grad_sums) {
            *parent -= child;
        }
        for (parent, &child) in self.hess_sums.iter_mut().zip(&child.hess_sums) {
            *parent -= child;
        }
        if let (Some(parent), Some(child)) = (&mut self.grad_sq_sums, &child.grad_sq_sums) {
            for (parent, &child) in parent.iter_mut().zip(child) {
                *parent -= child;
            }
        }
        for (parent, &child) in self.counts.iter_mut().zip(&child.counts) {
            *parent -= child;
        }
        Ok(())
    }

    /// Zero all bin values in-place without deallocating.
    ///
    /// This resets gradient sums, hessian sums, and counts to zero for every
    /// bin in every feature histogram, allowing the bundle to be reused for a
    /// new node without re-allocating.
    pub fn reset(&mut self, node_id: u32) {
        self.node_id = node_id;
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
        if let Some(values) = &mut self.grad_sq_sums {
            values.fill(0.0);
        }
        self.counts.fill(0);
    }

    /// Create a pre-allocated, zeroed histogram bundle for the given features and bin count.
    pub fn new_zeroed(feature_indices: &[u32], bin_count: usize) -> Self {
        Self::new_zeroed_with_grad_sq(feature_indices, bin_count, true)
    }

    pub fn new_zeroed_with_grad_sq(
        feature_indices: &[u32],
        bin_count: usize,
        include_grad_sq: bool,
    ) -> Self {
        let flat_len = feature_indices.len() * bin_count;
        Self {
            node_id: 0,
            feature_indices: feature_indices.to_vec(),
            bin_count,
            offsets: uniform_offsets(feature_indices.len(), bin_count),
            compact_edge_limit: None,
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            grad_sq_sums: include_grad_sq.then(|| vec![0.0; flat_len]),
            counts: vec![0; flat_len],
        }
    }
}

fn uniform_offsets(feature_count: usize, bin_count: usize) -> Vec<usize> {
    (0..=feature_count).map(|index| index * bin_count).collect()
}

#[cfg(test)]
mod tests {
    use super::HistogramBundle;

    #[test]
    fn feature_view_reads_aligned_soa_columns() {
        let bundle = HistogramBundle::from_soa(
            7,
            vec![2, 5],
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            Some(vec![100.0, 200.0, 300.0, 400.0]),
            vec![1, 2, 3, 4],
        )
        .expect("valid SoA histogram");

        let feature = bundle.feature(1).expect("second feature");
        assert_eq!(feature.feature_index(), 5);
        assert_eq!(feature.grad_sums(), &[3.0, 4.0]);
        assert_eq!(feature.hess_sums(), &[30.0, 40.0]);
        assert_eq!(feature.grad_sq_sums(), Some(&[300.0, 400.0][..]));
        assert_eq!(feature.counts(), &[3, 4]);
    }

    #[test]
    fn standard_bundle_omits_squared_gradient_storage() {
        let bundle = HistogramBundle::from_soa(
            3,
            vec![1],
            2,
            vec![1.0, 2.0],
            vec![4.0, 5.0],
            None,
            vec![6, 7],
        )
        .expect("valid SoA histogram");

        assert!(!bundle.has_grad_sq_sums());
        let feature = bundle.feature(0).expect("feature");
        assert_eq!(feature.grad_sq_sums(), None);
        assert_eq!(feature.bin(1).expect("bin").grad_sq_sum, 0.0);
    }
}
