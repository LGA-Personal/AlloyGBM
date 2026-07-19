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

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBundle {
    pub node_id: u32,
    feature_indices: Vec<u32>,
    bin_count: usize,
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
        Ok(Self {
            node_id,
            feature_indices,
            bin_count,
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

    pub fn bin_count(&self) -> usize {
        self.bin_count
    }

    pub fn feature_indices(&self) -> &[u32] {
        &self.feature_indices
    }

    pub fn has_grad_sq_sums(&self) -> bool {
        self.grad_sq_sums.is_some()
    }

    pub fn feature(&self, index: usize) -> Option<HistogramFeatureView<'_>> {
        let feature_index = *self.feature_indices.get(index)?;
        let start = index * self.bin_count;
        let end = start + self.bin_count;
        Some(HistogramFeatureView {
            feature_index,
            grad_sums: &self.grad_sums[start..end],
            hess_sums: &self.hess_sums[start..end],
            grad_sq_sums: self.grad_sq_sums.as_ref().map(|values| &values[start..end]),
            counts: &self.counts[start..end],
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
        if feature_index >= self.feature_count() || bin_index >= self.bin_count {
            return Err(CoreError::Validation(
                "histogram feature or bin index is out of bounds".to_string(),
            ));
        }
        let index = feature_index * self.bin_count + bin_index;
        self.grad_sums[index] = bin.grad_sum;
        self.hess_sums[index] = bin.hess_sum;
        if let Some(values) = &mut self.grad_sq_sums {
            values[index] = bin.grad_sq_sum;
        }
        self.counts[index] = bin.count;
        Ok(())
    }

    pub fn append(&mut self, other: Self) -> CoreResult<()> {
        if self.bin_count != other.bin_count {
            return Err(CoreError::Validation(
                "cannot append histograms with different bin counts".to_string(),
            ));
        }
        if self.has_grad_sq_sums() != other.has_grad_sq_sums() {
            return Err(CoreError::Validation(
                "cannot append histograms with different squared-gradient layouts".to_string(),
            ));
        }
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
        let mut result = Self::new_zeroed_with_grad_sq(
            &selected
                .iter()
                .map(|&index| self.feature_indices[index])
                .collect::<Vec<_>>(),
            self.bin_count,
            self.has_grad_sq_sums(),
        );
        result.node_id = self.node_id;
        for (dest_feature, &source_feature) in selected.iter().enumerate() {
            let source_start = source_feature * self.bin_count;
            let source_end = source_start + self.bin_count;
            let dest_start = dest_feature * self.bin_count;
            let dest_end = dest_start + self.bin_count;
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

    pub fn subtract_into(&mut self, parent: &Self, child: &Self, node_id: u32) -> CoreResult<()> {
        if parent.feature_indices != child.feature_indices
            || self.feature_indices != parent.feature_indices
            || parent.bin_count != child.bin_count
            || self.bin_count != parent.bin_count
            || parent.has_grad_sq_sums() != child.has_grad_sq_sums()
            || self.has_grad_sq_sums() != parent.has_grad_sq_sums()
        {
            return Err(CoreError::Validation(
                "histogram subtraction requires identical layouts".to_string(),
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
            *dest = parent.checked_sub(child).ok_or_else(|| {
                CoreError::Validation("child histogram count exceeds parent".to_string())
            })?;
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
            grad_sums: vec![0.0; flat_len],
            hess_sums: vec![0.0; flat_len],
            grad_sq_sums: include_grad_sq.then(|| vec![0.0; flat_len]),
            counts: vec![0; flat_len],
        }
    }
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
