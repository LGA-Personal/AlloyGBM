/// Uncertainty metric used by the DRO leaf solver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DroMetric {
    /// Wasserstein-inspired uncertainty radius over leaf gradient dispersion.
    #[default]
    Wasserstein,
}

/// Configuration for the fast DRO-style scalar leaf solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroConfig {
    /// Non-negative robustness radius. `0.0` is exactly standard leaf behavior.
    pub radius: f32,
    /// Uncertainty metric for interpreting the radius.
    pub metric: DroMetric,
}

impl Default for DroConfig {
    fn default() -> Self {
        Self {
            radius: 0.05,
            metric: DroMetric::Wasserstein,
        }
    }
}
