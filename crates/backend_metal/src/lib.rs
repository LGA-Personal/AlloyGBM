//! Metal GPU backend for AlloyGBM on Apple Silicon.
//!
//! The crate compiles as a stub on non-macOS targets so `cargo check
//! --workspace` stays green cross-platform; the real implementation is
//! gated by `cfg(target_os = "macos")`.
//!
//! Stage 1 scope is tracked in `docs/metal-backend/STATUS.md`.

#[cfg(target_os = "macos")]
mod device;

#[cfg(target_os = "macos")]
pub use device::{MetalCapabilities, MetalDevice, probe_capabilities};

pub mod kernels;

#[cfg(target_os = "macos")]
pub mod pipelines;

#[cfg(target_os = "macos")]
use alloygbm_backend_cpu::CpuBackend;
#[cfg(target_os = "macos")]
use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, NodeSlice, NodeStats,
    PartitionResult, SplitCandidate,
};
#[cfg(target_os = "macos")]
use alloygbm_engine::{BackendOps, CategoricalFeatureInfo, EngineResult, SplitSelectionOptions};

#[cfg(target_os = "macos")]
pub struct MetalBackend {
    pub metal_device: MetalDevice,
    /// Compiled-once + pipeline-cached histogram kernels (S1.5). The
    /// cache is wrapped in `Arc` so it is cheap to clone a handle into
    /// `dispatch_histograms` without holding a borrow on `self` across
    /// the call.
    pipeline_cache: std::sync::Arc<pipelines::HistogramPipelineCache>,
    /// CPU backend embedded as the fallback for every `BackendOps`
    /// method except `build_histograms` (S1.6 promise realised in S1.4).
    cpu: CpuBackend,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    /// Probe the system Metal device and build a backend handle. Returns
    /// an error when Metal is unavailable — callers (the PyO3 layer) are
    /// expected to warn-and-fall-back to `CpuBackend`.
    pub fn new() -> Result<Self, String> {
        let metal_device = MetalDevice::probe()?;
        if !metal_device.capabilities.apple7 {
            return Err(format!(
                "Metal backend requires GPU family Apple7 or later; \
                 device '{}' does not support it",
                metal_device.capabilities.device_name
            ));
        }
        let pipeline_cache = std::sync::Arc::new(pipelines::HistogramPipelineCache::new(
            metal_device.device.clone(),
            &metal_device.capabilities,
        )?);
        Ok(Self {
            metal_device,
            pipeline_cache,
            cpu: CpuBackend,
        })
    }

    /// Read-only capability snapshot.
    pub fn capabilities(&self) -> &MetalCapabilities {
        &self.metal_device.capabilities
    }
}

#[cfg(target_os = "macos")]
impl BackendOps for MetalBackend {
    fn build_histograms(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
    ) -> EngineResult<HistogramBundle> {
        kernels::histogram::dispatch_histograms(
            &self.metal_device,
            &self.pipeline_cache,
            binned_matrix,
            gradients,
            node,
            feature_tiles,
        )
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        self.cpu.best_split(histograms)
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        self.cpu
            .best_split_with_options(histograms, options, feature_weights, categorical_features)
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        self.cpu.apply_split(binned_matrix, node, split)
    }

    fn apply_split_with_stats(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<(PartitionResult, NodeStats, NodeStats)> {
        self.cpu
            .apply_split_with_stats(binned_matrix, gradients, node, split)
    }

    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        row_indices: &[u32],
    ) -> EngineResult<NodeStats> {
        self.cpu.reduce_sums(gradients, row_indices)
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        Err("Metal backend is only available on macOS".to_string())
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #![allow(unsafe_code)]

    use super::*;
    use alloygbm_core::{FeatureTile, GradientPair, NodeSlice};
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice;

    #[test]
    fn probe_default_device() {
        match MetalBackend::new() {
            Ok(backend) => {
                let caps = backend.capabilities();
                assert!(caps.apple7, "expected Apple7+ on the CI/dev machine");
                assert!(!caps.device_name.is_empty());
            }
            Err(_) => {
                // Headless runner without a Metal device — not a failure.
            }
        }
    }

    #[test]
    fn histogram_shader_compiles() {
        let Ok(backend) = MetalBackend::new() else {
            return; // no Metal device available — skip.
        };

        let source = NSString::from_str(kernels::histogram::HISTOGRAM_SHADER_SOURCE);
        let result = backend
            .metal_device
            .device
            .newLibraryWithSource_options_error(&source, None);
        match result {
            Ok(_library) => {}
            Err(err) => panic!(
                "histogram.metal failed to compile: {}",
                err.localizedDescription()
            ),
        }
    }

    /// Tiny fixture (<1000 rows, small bin count) where the histogram is
    /// hand-computable; verifies bit-exact equality between `MetalBackend`
    /// and `CpuBackend` on (grad_sum, hess_sum, count) per bin.
    ///
    /// Gradients are chosen from a small set of exact f32 integer values
    /// (1.0, 2.0, 4.0) so floating-point addition is associative in the
    /// integer range — hence any ordering of accumulation lands on the
    /// same bit pattern, independent of chunk boundaries / SIMD lane
    /// serialisation.
    #[test]
    fn histogram_matches_cpu_small_fixture() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return; // no Metal device — skip.
        };

        let row_count = 500usize;
        let feature_count = 6usize;
        let max_bin: u16 = 7; // 8 bins (0..=7) including the implicit NaN sentinel at bin 7.

        // Deterministic PRNG-free bin assignment: bin = (row * 31 + feat * 17) & 7.
        let mut bins_row_major = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                let bin = ((row.wrapping_mul(31) ^ feat.wrapping_mul(17)) & 7) as u8;
                bins_row_major.push(bin);
            }
        }
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();

        // Gradients with exact f32 integer representations.
        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|i| {
                let grad = match i % 3 {
                    0 => 1.0,
                    1 => -2.0,
                    _ => 4.0,
                };
                let hess = match i % 2 {
                    0 => 1.0,
                    _ => 2.0,
                };
                GradientPair::new(grad, hess).unwrap()
            })
            .collect();

        // Full-node slice over every row.
        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(0, feature_count as u32).unwrap()];

        let cpu = CpuBackend;
        let cpu_hist = cpu
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("cpu histogram");
        let metal_hist = backend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("metal histogram");

        assert_eq!(cpu_hist.node_id, metal_hist.node_id);
        assert_eq!(
            cpu_hist.feature_histograms.len(),
            metal_hist.feature_histograms.len()
        );
        for (cpu_fh, metal_fh) in cpu_hist
            .feature_histograms
            .iter()
            .zip(metal_hist.feature_histograms.iter())
        {
            assert_eq!(cpu_fh.feature_index, metal_fh.feature_index);
            assert_eq!(
                cpu_fh.bins.len(),
                metal_fh.bins.len(),
                "feature {} bin count",
                cpu_fh.feature_index
            );
            for (bin_idx, (cpu_bin, metal_bin)) in
                cpu_fh.bins.iter().zip(metal_fh.bins.iter()).enumerate()
            {
                assert_eq!(
                    cpu_bin.count, metal_bin.count,
                    "feature {} bin {} count",
                    cpu_fh.feature_index, bin_idx
                );
                assert_eq!(
                    cpu_bin.grad_sum.to_bits(),
                    metal_bin.grad_sum.to_bits(),
                    "feature {} bin {} grad_sum: cpu={} metal={}",
                    cpu_fh.feature_index,
                    bin_idx,
                    cpu_bin.grad_sum,
                    metal_bin.grad_sum
                );
                assert_eq!(
                    cpu_bin.hess_sum.to_bits(),
                    metal_bin.hess_sum.to_bits(),
                    "feature {} bin {} hess_sum: cpu={} metal={}",
                    cpu_fh.feature_index,
                    bin_idx,
                    cpu_bin.hess_sum,
                    metal_bin.hess_sum
                );
            }
        }
    }

    /// Feature-subset tile: request histograms for only features 2..=4 and
    /// verify the Metal result still matches CPU on that subset.
    #[test]
    fn histogram_feature_subset_matches_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 200usize;
        let feature_count = 6usize;
        let max_bin: u16 = 3; // 4 bins

        let mut bins_row_major = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                bins_row_major.push(((row + feat) & 3) as u8);
            }
        }
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();

        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair::new(if i % 2 == 0 { 1.0 } else { -1.0 }, 2.0).unwrap())
            .collect();

        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(2, 5).unwrap()];

        let cpu_hist = CpuBackend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("cpu histogram");
        let metal_hist = backend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("metal histogram");

        assert_eq!(cpu_hist.feature_histograms.len(), 3);
        assert_eq!(metal_hist.feature_histograms.len(), 3);
        for (cpu_fh, metal_fh) in cpu_hist
            .feature_histograms
            .iter()
            .zip(metal_hist.feature_histograms.iter())
        {
            assert_eq!(cpu_fh.feature_index, metal_fh.feature_index);
            for (cpu_bin, metal_bin) in cpu_fh.bins.iter().zip(metal_fh.bins.iter()) {
                assert_eq!(cpu_bin.count, metal_bin.count);
                assert_eq!(cpu_bin.grad_sum.to_bits(), metal_bin.grad_sum.to_bits());
                assert_eq!(cpu_bin.hess_sum.to_bits(), metal_bin.hess_sum.to_bits());
            }
        }
    }

    /// S1.5: two successive `get_or_build` calls with the same key
    /// must return the exact same `Arc`. This guards against the
    /// pipeline cache being bypassed (e.g. if a refactor accidentally
    /// reintroduces per-dispatch compilation).
    #[test]
    fn pipeline_cache_returns_identical_arc_on_second_call() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let first = backend
            .pipeline_cache
            .get_or_build(8, false)
            .expect("first build");
        let second = backend
            .pipeline_cache
            .get_or_build(8, false)
            .expect("second build");

        // Same allocation ⇒ same pipelines ⇒ no recompilation.
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "pipeline cache must return the same Arc on hit"
        );

        // Different key ⇒ distinct Arc.
        let wide = backend
            .pipeline_cache
            .get_or_build(8, true)
            .expect("u16 variant build");
        assert!(!std::sync::Arc::ptr_eq(&first, &wide));
    }
}
