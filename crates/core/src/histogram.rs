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

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBundle {
    pub node_id: u32,
    pub feature_histograms: Vec<FeatureHistogram>,
}

impl HistogramBundle {
    /// Zero all bin values in-place without deallocating.
    ///
    /// This resets gradient sums, hessian sums, and counts to zero for every
    /// bin in every feature histogram, allowing the bundle to be reused for a
    /// new node without re-allocating.
    pub fn reset(&mut self, node_id: u32) {
        self.node_id = node_id;
        for fh in &mut self.feature_histograms {
            for bin in &mut fh.bins {
                bin.grad_sum = 0.0;
                bin.hess_sum = 0.0;
                bin.grad_sq_sum = 0.0;
                bin.count = 0;
            }
        }
    }

    /// Create a pre-allocated, zeroed histogram bundle for the given features and bin count.
    pub fn new_zeroed(feature_indices: &[u32], bin_count: usize) -> Self {
        let feature_histograms = feature_indices
            .iter()
            .map(|&fi| FeatureHistogram {
                feature_index: fi,
                bins: vec![
                    HistogramBin {
                        grad_sum: 0.0,
                        hess_sum: 0.0,
                        grad_sq_sum: 0.0,
                        count: 0,
                    };
                    bin_count
                ],
            })
            .collect();
        Self {
            node_id: 0,
            feature_histograms,
        }
    }
}
