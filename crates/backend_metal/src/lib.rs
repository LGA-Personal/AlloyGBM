//! Metal GPU backend for AlloyGBM on Apple Silicon.
//!
//! The crate compiles as a stub on non-macOS targets so `cargo check
//! --workspace` stays green cross-platform; the real implementation is
//! gated by `cfg(target_os = "macos")`.
//!
//! Stage 1 scope is tracked in `docs/metal-backend/STATUS.md`.

#[cfg(target_os = "macos")]
mod budget;
#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod histogram_residency;
#[cfg(target_os = "macos")]
mod residency;

#[cfg(target_os = "macos")]
pub use device::{MetalCapabilities, MetalDevice, probe_capabilities};

pub mod kernels;

#[cfg(target_os = "macos")]
pub mod pipelines;

#[cfg(target_os = "macos")]
use alloygbm_backend_cpu::CpuBackend;
#[cfg(target_os = "macos")]
use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, HistogramBundle, HistogramStorage, NodeSlice,
    NodeStats, PartitionResult, SplitCandidate,
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
    /// S2.3 — compiled-once + pipeline-cached best-split kernel.
    /// Specialized on `(bin_count, l1_enabled)`; typical fits hit a
    /// single key and every call past the first is an `Arc::clone`.
    split_pipeline_cache: std::sync::Arc<pipelines::SplitPipelineCache>,
    /// S3.5 — compiled-once + pipeline-cached row-partition kernel.
    /// Specialized on `(block_size, split_kind, bin_is_u16)`; the
    /// hot path through a training run hits two keys (continuous +
    /// categorical if any) so every call past compilation is an
    /// `Arc::clone`.
    partition_pipeline_cache: std::sync::Arc<pipelines::PartitionPipelineCache>,
    /// S3.6 — compiled-once + pipeline-cached elementwise subtract
    /// kernel. Single function constant (`BLOCK_SIZE`); effectively a
    /// one-entry cache under the current architecture. Retained as a
    /// cache for parity with the other kernels and for forward
    /// compatibility once S3.7 introduces GPU-resident histograms
    /// that will call this kernel once per non-smaller-sibling node.
    ///
    /// `dead_code` allow: the consumer is S3.7 (BackendOps trait
    /// promotion of `subtract_histogram_bundle` + MetalBackend GPU
    /// implementation). Until then the field is exercised by unit
    /// tests but not by the release-library surface.
    #[allow(dead_code)]
    subtract_pipeline_cache: std::sync::Arc<pipelines::SubtractPipelineCache>,
    /// Persistent Metal buffer pool for the histogram dispatch path.
    /// Caches the binned matrix (immutable per fit) and reuses the
    /// allocations for gradients + row indices across the ~63
    /// `build_histograms` calls a depth-6 tree makes. Without this,
    /// each call was doing a fresh `newBufferWithBytes` for the
    /// whole column-major binned matrix — tens of GiB of memcpy per
    /// fit at realistic scales.
    ///
    /// S2.3 extended this pool with four additional reusable slots
    /// for the split kernel inputs (grad/hess/counts/mask).
    buffer_cache: std::sync::Arc<buffers::BufferCache>,
    /// CPU backend embedded as the fallback for every `BackendOps`
    /// method except `build_histograms` (S1.6 promise realised in S1.4)
    /// and the categorical half of `best_split` (Stage 2 — see
    /// DECISIONS: Fisher-sort is a separate research problem on GPU).
    cpu: CpuBackend,
    /// S3.9 — GPU working-set budget snapshot. Captured once at
    /// construction; used by `check_histogram_budget` at fit start
    /// to refuse pathological shapes (leaf-wise + huge leaf count +
    /// wide feature set + wide bin grid) before they cause
    /// GPU-side residency thrash. Consumer is S3.3 (trainer-loop
    /// refactor) and S3.7 (residency wiring); the tracker ships
    /// in isolation so it can be unit-tested without those
    /// refactors in place.
    #[allow(dead_code)]
    budget: budget::BudgetTracker,
    /// S3.8 — `MTLResidencySet` wrapper (macOS 15+) with
    /// `PassThrough` fallback. Pins GPU-resident histogram + row
    /// index buffers to the working set for the lifetime of the
    /// backend; attached to `metal_device.queue` at construction.
    /// `dead_code` allow: S3.7c is the first live consumer
    /// (feeds `HistogramResidencyPool` allocations through
    /// `add_buffer`).
    #[allow(dead_code)]
    residency: residency::ResidencyPool,
    /// S3.7b — skeleton GPU-resident histogram pool. Handed into
    /// `build_histograms` once S3.7c wires the kernel output to
    /// land in pool-owned buffers instead of the Stage-1
    /// read-back-to-CPU path. Stored under an `Arc` so kernel
    /// dispatchers can clone a handle without holding a borrow
    /// across the call.
    #[allow(dead_code)]
    histogram_residency: std::sync::Arc<histogram_residency::HistogramResidencyPool>,
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
        let split_pipeline_cache = std::sync::Arc::new(pipelines::SplitPipelineCache::new(
            metal_device.device.clone(),
            &metal_device.capabilities,
        )?);
        let partition_pipeline_cache = std::sync::Arc::new(pipelines::PartitionPipelineCache::new(
            metal_device.device.clone(),
            &metal_device.capabilities,
        )?);
        let subtract_pipeline_cache = std::sync::Arc::new(pipelines::SubtractPipelineCache::new(
            metal_device.device.clone(),
            &metal_device.capabilities,
        )?);
        let buffer_cache = std::sync::Arc::new(buffers::BufferCache::new());
        let budget = budget::BudgetTracker::new(&metal_device.device);
        // S3.8: attach a residency set to the command queue. Creation
        // failures degrade to a pass-through no-op (documented in
        // `residency.rs`), so `new` is infallible from the user's
        // perspective.
        let residency = residency::ResidencyPool::new(
            &metal_device.device,
            metal_device.queue.clone(),
            "alloygbm::metal-backend::residency",
        );
        let histogram_residency =
            std::sync::Arc::new(histogram_residency::HistogramResidencyPool::new());
        Ok(Self {
            metal_device,
            pipeline_cache,
            split_pipeline_cache,
            partition_pipeline_cache,
            subtract_pipeline_cache,
            buffer_cache,
            cpu: CpuBackend,
            budget,
            residency,
            histogram_residency,
        })
    }

    /// Read-only capability snapshot.
    pub fn capabilities(&self) -> &MetalCapabilities {
        &self.metal_device.capabilities
    }

    /// Pre-fit GPU working-set budget check. Returns `Ok(())` when the
    /// projected peak histogram residency (F × B × L × 12 bytes) is
    /// below 80 % of `MTLDevice.recommendedMaxWorkingSetSize`;
    /// otherwise returns `EngineError::BackendUnavailable` with an
    /// actionable diagnostic (device="cpu" fallback + M3 roadmap
    /// pointer).
    ///
    /// Call once at fit start. The trainer threads `(n_features,
    /// bin_count, max_level_width)` into this guard before building
    /// the root histograms so the refusal fires before any GPU
    /// allocation happens. `max_level_width` is the widest count
    /// of live sibling nodes the trainer will reach — for level-wise
    /// growth this is `2.pow(max_depth - 1)`; for leaf-wise growth
    /// it is bounded by `num_leaves`.
    ///
    /// See `budget.rs` for the rationale, including the M2
    /// pathological-case risk note.
    pub fn check_histogram_budget(
        &self,
        n_features: u32,
        bin_count: u32,
        max_level_width: u32,
    ) -> EngineResult<()> {
        self.budget
            .check_projected_peak(n_features, bin_count, max_level_width)
    }

    /// Diagnostic accessor: Apple's `recommendedMaxWorkingSetSize`
    /// for this device. Used by tests + potential future
    /// `native_runtime_info` extension.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.budget.recommended_max_working_set_size()
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
            &self.buffer_cache,
            &self.histogram_residency,
            &self.residency,
            binned_matrix,
            gradients,
            node,
            feature_tiles,
        )
    }

    fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
        // Default options + no feature weights + no categoricals: this
        // is the hot path for continuous-only, unregularised splits.
        kernels::split::dispatch_best_split(
            &self.metal_device,
            &self.split_pipeline_cache,
            &self.buffer_cache,
            &self.histogram_residency,
            &self.cpu,
            histograms,
            SplitSelectionOptions::default(),
            &[],
            &[],
        )
    }

    fn best_split_with_options(
        &self,
        histograms: &HistogramBundle,
        options: SplitSelectionOptions,
        feature_weights: &[f32],
        categorical_features: &[CategoricalFeatureInfo],
    ) -> EngineResult<Option<SplitCandidate>> {
        kernels::split::dispatch_best_split(
            &self.metal_device,
            &self.split_pipeline_cache,
            &self.buffer_cache,
            &self.histogram_residency,
            &self.cpu,
            histograms,
            options,
            feature_weights,
            categorical_features,
        )
    }

    /// S3.7c.3 — override the engine's default CPU subtract with a
    /// pool-direct GPU dispatch when both inputs are already
    /// GPU-resident. This is the Stage-3 hot path: the CPU round-trip
    /// (flatten → upload → dispatch → readback → repack) that the
    /// standalone `dispatch_subtract` incurs collapses to a single
    /// kernel launch against pool-owned buffers.
    ///
    /// Dispatch matrix:
    /// * **Gpu + Gpu** → pool-direct kernel dispatch; returns a
    ///   freshly-minted `HistogramStorage::Gpu` bundle. Zero memcpy.
    /// * **Cpu + Cpu** → delegate to the embedded `CpuBackend`, which
    ///   uses the engine's default free-function (elementwise CPU
    ///   loop). Bit-exact with the pre-Stage-3 behaviour.
    /// * **Mixed** → contract violation. The trainer produces sibling
    ///   histograms from a single `build_histograms` call, so both
    ///   bundles at a given level are always the same storage
    ///   variant. A mixed pair here means a bug upstream; fail loudly
    ///   rather than silently re-materialising.
    fn subtract_histogram_bundle(
        &self,
        parent: &HistogramBundle,
        child: &HistogramBundle,
        node_id: u32,
    ) -> EngineResult<HistogramBundle> {
        match (&parent.storage, &child.storage) {
            (
                HistogramStorage::Gpu { handle: ph, .. },
                HistogramStorage::Gpu { handle: ch, .. },
            ) => kernels::subtract::dispatch_subtract_pool(
                &self.metal_device,
                &self.subtract_pipeline_cache,
                &self.histogram_residency,
                &self.residency,
                *ph,
                *ch,
                node_id,
            ),
            (HistogramStorage::Cpu(_), HistogramStorage::Cpu(_)) => {
                self.cpu.subtract_histogram_bundle(parent, child, node_id)
            }
            _ => Err(alloygbm_engine::EngineError::ContractViolation(
                "MetalBackend::subtract_histogram_bundle: mixed Cpu/Gpu storage variants \
                 — sibling histograms must share the same storage variant"
                    .to_string(),
            )),
        }
    }

    fn apply_split(
        &self,
        binned_matrix: &BinnedMatrix,
        node: &NodeSlice,
        split: &SplitCandidate,
    ) -> EngineResult<PartitionResult> {
        // S3.5 — try the GPU partition kernel first. It falls back to
        // CPU on two conditions:
        //   * Node exceeds the single-threadgroup scan cap
        //     (`MAX_BLOCKS_SINGLE_SCAN`). A future hierarchical scan
        //     would lift this.
        //   * Any other Metal-side failure (buffer alloc, pipeline
        //     build, command-buffer encode). The CPU path is always
        //     correct, so the backend never fails a fit because of a
        //     Metal hiccup.
        match kernels::partition::dispatch_partition(
            &self.metal_device,
            &self.partition_pipeline_cache,
            &self.buffer_cache,
            binned_matrix,
            node,
            split,
        ) {
            Ok(result) => Ok(result),
            Err(alloygbm_engine::EngineError::BackendUnavailable(_)) => {
                // Scan-cap overflow or transient Metal failure. CPU
                // path is always correct.
                self.cpu.apply_split(binned_matrix, node, split)
            }
            Err(err) => Err(err),
        }
    }
    // `apply_split_with_stats` uses the default trait impl — it calls
    // our GPU-accelerated `apply_split` above and then runs
    // `reduce_sums` on both halves. `reduce_sums` is still CPU-side
    // (Stage 3's scope doesn't include a GPU reduce); the round-trip
    // is the cost of producing stats without GPU residency.

    fn reduce_sums(
        &self,
        gradients: &[GradientPair],
        rows: &alloygbm_core::RowIndexStorage,
    ) -> EngineResult<NodeStats> {
        // S3.2 — trait surface now accepts `&RowIndexStorage`. Until
        // S3.7e ships a GPU row-index pool (deferred from S3.7d —
        // engine-side `PartitionResult` is still CPU-resident), every
        // call we see is `Cpu(..)` — delegate to the embedded
        // CpuBackend. When MetalBackend starts producing `Gpu(..)`
        // row-index variants, this override will read directly from
        // the GPU-resident buffer without a CPU round-trip.
        self.cpu.reduce_sums(gradients, rows)
    }

    /// Release the GPU-resident histogram pool entry backing this
    /// bundle, if any.
    ///
    /// Called by the trainer hot loops (via `HistogramReleaseGuard`)
    /// at the moment each parent histogram is no longer needed —
    /// level-wise that's end-of-inner-loop-iteration; leaf-wise
    /// that's end-of-pop-iteration plus queue-drain on early break.
    /// Without this, pool entries would accumulate across the full
    /// fit and `budget::BudgetTracker`'s one-level-wide peak
    /// projection would under-count by a factor of ~tree-size,
    /// silently breaking the M2 free-on-consume guarantee (D-016).
    ///
    /// Pattern-matches on storage:
    /// * `Gpu { handle, .. }` — dispatches `histogram_residency
    ///   .release(handle)`, which also detaches the three backing
    ///   buffers from the `MTLResidencySet`.
    /// * `Cpu(..)` — no-op (no residency to detach).
    ///
    /// Errors from the pool mutex are **swallowed** via the trait's
    /// `EngineResult<()>` surface returning `Ok(())`: a use-after-
    /// release or poisoned-mutex failure here is not worth aborting
    /// the fit for, and the trainer's `HistogramReleaseGuard`
    /// ignores the return value via `Drop` anyway. The worst
    /// observable consequence is one leaked GPU buffer for this
    /// fit's lifetime.
    fn release_histograms(&self, bundle: &alloygbm_core::HistogramBundle) -> EngineResult<()> {
        match &bundle.storage {
            HistogramStorage::Gpu { handle, .. } => {
                self.histogram_residency.release(*handle, &self.residency);
            }
            HistogramStorage::Cpu(_) => {
                // CPU-resident histograms are plain `Vec<_>`s —
                // nothing to detach, the caller's `Drop` on the
                // bundle will free the memory.
            }
        }
        Ok(())
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
    use alloygbm_core::{FeatureHistogram, FeatureTile, GradientPair, HistogramBin, NodeSlice};
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice;

    /// Materialise a `HistogramBundle` into a CPU-side
    /// `Vec<FeatureHistogram>` for test assertions. Post-S3.7c.2,
    /// `MetalBackend::build_histograms` returns `HistogramStorage::Gpu`;
    /// tests that want to compare bin-by-bin against a `CpuBackend`
    /// result go through this helper rather than calling
    /// `.feature_histograms()` directly (which panics on Gpu).
    ///
    /// Test-only — the production hot paths (`best_split`,
    /// `subtract_histogram_bundle`) read pool buffers directly without
    /// this round-trip.
    fn materialize_bundle_for_test(
        backend: &MetalBackend,
        bundle: &HistogramBundle,
    ) -> Vec<FeatureHistogram> {
        match &bundle.storage {
            alloygbm_core::HistogramStorage::Cpu(fhs) => fhs.clone(),
            alloygbm_core::HistogramStorage::Gpu { handle, .. } => {
                let planes = backend
                    .histogram_residency
                    .read_planes(*handle)
                    .expect("pool readback");
                let bc = planes.bin_count as usize;
                planes
                    .feature_indices
                    .iter()
                    .enumerate()
                    .map(|(local_f, &feature_index)| {
                        let base = local_f * bc;
                        let bins = (0..bc)
                            .map(|b| HistogramBin {
                                grad_sum: planes.grad[base + b],
                                hess_sum: planes.hess[base + b],
                                count: planes.counts[base + b],
                            })
                            .collect();
                        FeatureHistogram {
                            feature_index,
                            bins,
                        }
                    })
                    .collect()
            }
        }
    }

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
        let cpu_fhs = cpu_hist.feature_histograms();
        let metal_fhs = materialize_bundle_for_test(&backend, &metal_hist);
        assert_eq!(cpu_fhs.len(), metal_fhs.len());
        for (cpu_fh, metal_fh) in cpu_fhs.iter().zip(metal_fhs.iter()) {
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

        let cpu_fhs = cpu_hist.feature_histograms();
        let metal_fhs = materialize_bundle_for_test(&backend, &metal_hist);
        assert_eq!(cpu_fhs.len(), 3);
        assert_eq!(metal_fhs.len(), 3);
        for (cpu_fh, metal_fh) in cpu_fhs.iter().zip(metal_fhs.iter()) {
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

    // ---------------------------------------------------------------
    // S2.5 — Best-split kernel correctness
    // ---------------------------------------------------------------

    /// Build a seeded small fixture (row-major binned matrix +
    /// gradients) and the matching `HistogramBundle` via
    /// `CpuBackend::build_histograms`. Returned bundle is usable as
    /// direct input to both `cpu.best_split_with_options` and
    /// `metal.best_split_with_options`.
    fn build_fixture(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        gradient_pattern: impl Fn(usize) -> (f32, f32),
        missing_bin_index: u16,
    ) -> (
        alloygbm_core::BinnedMatrix,
        Vec<GradientPair>,
        alloygbm_core::HistogramBundle,
    ) {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let mut bins = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                let b =
                    (row.wrapping_mul(31) ^ feat.wrapping_mul(17)) % (missing_bin_index as usize);
                bins.push(b as u8);
            }
        }
        let binned =
            alloygbm_core::BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();

        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|i| {
                let (g, h) = gradient_pattern(i);
                GradientPair::new(g, h).unwrap()
            })
            .collect();

        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(0, feature_count as u32).unwrap()];

        let hist = CpuBackend
            .build_histograms(&binned, &gradients, &node, &tiles)
            .expect("cpu histogram fixture");

        (binned, gradients, hist)
    }

    fn assert_structural_equality(
        cpu: Option<&SplitCandidate>,
        metal: Option<&SplitCandidate>,
        context: &str,
    ) {
        match (cpu, metal) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(
                    a.feature_index, b.feature_index,
                    "{context}: feature_index mismatch (cpu={}, metal={})",
                    a.feature_index, b.feature_index
                );
                assert_eq!(
                    a.threshold_bin, b.threshold_bin,
                    "{context}: threshold_bin mismatch on feature {} (cpu={}, metal={})",
                    a.feature_index, a.threshold_bin, b.threshold_bin
                );
                assert_eq!(
                    a.default_left, b.default_left,
                    "{context}: default_left mismatch on feature {}",
                    a.feature_index
                );
                assert_eq!(
                    a.is_categorical, b.is_categorical,
                    "{context}: is_categorical mismatch"
                );
                // Allow tiny ulp drift on gain (block-scan vs serial sweep).
                let gain_rel = (a.gain - b.gain).abs() / a.gain.abs().max(1e-6);
                assert!(
                    gain_rel < 1e-4,
                    "{context}: gain drifted too far (cpu={}, metal={}, rel={})",
                    a.gain,
                    b.gain,
                    gain_rel
                );
            }
            (cpu_res, metal_res) => {
                panic!(
                    "{context}: one side returned a candidate and the other did not: \
                     cpu={:?} metal={:?}",
                    cpu_res.is_some(),
                    metal_res.is_some()
                );
            }
        }
    }

    /// 200 rows × 4 features × 16 bins: well-conditioned fixture where
    /// Metal and CPU must pick the same split.
    #[test]
    fn best_split_matches_cpu_small_fixture() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let max_bin: u16 = 15; // 16 bins (0..=14 data + 15 NaN sentinel)
        let missing_bin_index = max_bin;
        let (_, _, hist) = build_fixture(
            200,
            4,
            max_bin,
            |i| match i % 5 {
                0 => (1.0, 1.0),
                1 => (-2.0, 2.0),
                2 => (3.0, 1.0),
                3 => (-1.0, 2.0),
                _ => (2.0, 1.0),
            },
            missing_bin_index,
        );

        let options = alloygbm_engine::SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: missing_bin_index as usize,
        };

        let cpu_result = CpuBackend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("cpu best_split");
        let metal_result = backend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("metal best_split");

        assert_structural_equality(cpu_result.as_ref(), metal_result.as_ref(), "small_fixture");
    }

    /// L1 + L2 regularisation on: exercises both kernel specialisations
    /// (`L1_ENABLED=true`, non-zero lambda in the denominator).
    #[test]
    fn best_split_with_l1_l2_matches_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let max_bin: u16 = 31;
        let missing_bin_index = max_bin;
        let (_, _, hist) = build_fixture(
            400,
            6,
            max_bin,
            |i| (if i % 2 == 0 { 1.5 } else { -2.0 }, 1.0),
            missing_bin_index,
        );

        let options = alloygbm_engine::SplitSelectionOptions {
            l2_lambda: 2.0,
            l1_alpha: 0.5,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: missing_bin_index as usize,
        };

        let cpu_result = CpuBackend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("cpu best_split");
        let metal_result = backend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("metal best_split");

        assert_structural_equality(cpu_result.as_ref(), metal_result.as_ref(), "l1_l2");
    }

    /// Non-uniform feature_weights must pick the same winner.
    #[test]
    fn best_split_with_feature_weights_matches_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let max_bin: u16 = 15;
        let missing_bin_index = max_bin;
        let (_, _, hist) = build_fixture(
            300,
            5,
            max_bin,
            |i| (if i % 3 == 0 { 2.0 } else { -1.0 }, 1.0),
            missing_bin_index,
        );

        let options = alloygbm_engine::SplitSelectionOptions {
            l2_lambda: 1.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: missing_bin_index as usize,
        };

        // Strongly penalise feature 0 and favour feature 3 so the
        // weighted cross-feature argmax has to agree across backends.
        let weights = vec![0.1_f32, 1.0, 1.0, 3.0, 1.0];

        let cpu_result = CpuBackend
            .best_split_with_options(&hist, options, &weights, &[])
            .expect("cpu best_split");
        let metal_result = backend
            .best_split_with_options(&hist, options, &weights, &[])
            .expect("metal best_split");

        assert_structural_equality(
            cpu_result.as_ref(),
            metal_result.as_ref(),
            "feature_weights",
        );
    }

    /// Missing-bin direction: a feature that's heavy on NaNs should
    /// produce identical default_left on both backends.
    #[test]
    fn best_split_with_missing_bin_matches_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_core::MISSING_BIN_U8;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 500usize;
        let feature_count = 4usize;
        let max_bin: u16 = MISSING_BIN_U8 as u16; // 255

        // Mix of real bin values and the missing sentinel across all
        // features — every 7th row on every feature is a "missing"
        // bin, biasing the NaN-direction choice.
        let mut bins = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                let b = if row % 7 == (feat + 1) {
                    max_bin as u8 // sentinel
                } else {
                    ((row + feat * 3) % 64) as u8
                };
                bins.push(b);
            }
        }
        let binned =
            alloygbm_core::BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|i| {
                let g = if i % 3 == 0 { 2.0 } else { -1.0 };
                GradientPair::new(g, 1.0).unwrap()
            })
            .collect();
        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(0, feature_count as u32).unwrap()];

        let hist = CpuBackend
            .build_histograms(&binned, &gradients, &node, &tiles)
            .expect("cpu histogram");

        let options = alloygbm_engine::SplitSelectionOptions {
            l2_lambda: 0.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: max_bin as usize,
        };

        let cpu_result = CpuBackend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("cpu best_split");
        let metal_result = backend
            .best_split_with_options(&hist, options, &[], &[])
            .expect("metal best_split");

        assert_structural_equality(cpu_result.as_ref(), metal_result.as_ref(), "missing_bin");
    }

    /// Mixed continuous + categorical features. The kernel should skip
    /// categoricals; the CPU path handles them; the final winner
    /// matches CPU end-to-end.
    #[test]
    fn best_split_with_categorical_feature_delegates_to_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::{BackendOps, CategoricalFeatureInfo};

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let max_bin: u16 = 15;
        let missing_bin_index = max_bin;
        let (_, _, hist) = build_fixture(
            300,
            5,
            max_bin,
            |i| match i % 4 {
                0 => (1.0, 1.0),
                1 => (-2.0, 2.0),
                2 => (1.5, 1.0),
                _ => (-0.5, 1.0),
            },
            missing_bin_index,
        );

        let options = alloygbm_engine::SplitSelectionOptions {
            l2_lambda: 1.0,
            l1_alpha: 0.0,
            min_child_hessian: 0.0,
            min_leaf_magnitude: 0.0,
            missing_bin_index: missing_bin_index as usize,
        };

        // Feature 1 is categorical; the remaining four are continuous.
        let categoricals = vec![CategoricalFeatureInfo {
            feature_index: 1,
            num_categories: 8,
        }];

        let cpu_result = CpuBackend
            .best_split_with_options(&hist, options, &[], &categoricals)
            .expect("cpu best_split");
        let metal_result = backend
            .best_split_with_options(&hist, options, &[], &categoricals)
            .expect("metal best_split");

        assert_structural_equality(cpu_result.as_ref(), metal_result.as_ref(), "categorical");
    }

    /// S2.3 — mirror the histogram-pipeline-cache Arc-reuse assertion
    /// for the split pipeline cache.
    #[test]
    fn split_pipeline_cache_returns_identical_arc_on_second_call() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let first = backend
            .split_pipeline_cache
            .get_or_build(16, false)
            .expect("first build");
        let second = backend
            .split_pipeline_cache
            .get_or_build(16, false)
            .expect("second build");
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "split pipeline cache must return the same Arc on hit"
        );

        let with_l1 = backend
            .split_pipeline_cache
            .get_or_build(16, true)
            .expect("l1 variant build");
        assert!(!std::sync::Arc::ptr_eq(&first, &with_l1));
    }

    /// Smoke test: compile the partition MSL library without
    /// specialization. Any MSL syntax error surfaces here before the
    /// more expensive pipeline-build path runs.
    #[test]
    fn partition_shader_compiles() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let source = NSString::from_str(kernels::partition::PARTITION_SHADER_SOURCE);
        let result = backend
            .metal_device
            .device
            .newLibraryWithSource_options_error(&source, None);
        if let Err(err) = result {
            panic!(
                "partition.metal failed to compile: {}",
                err.localizedDescription()
            );
        }
    }

    /// S3.5 — GPU partition must produce bit-identical output (in
    /// identical order) to `CpuBackend::apply_split` on a small,
    /// deterministic fixture. Covers continuous threshold, NaN
    /// default-left/right, and both u8 / u16 bin storage.
    #[test]
    fn partition_matches_cpu_small_fixture() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 2_000usize;
        let feature_count = 3usize;
        let max_bin: u16 = 7;

        let mut bins_row_major = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                let bin = ((row.wrapping_mul(31) ^ feat.wrapping_mul(17)) & 7) as u8;
                bins_row_major.push(bin);
            }
        }
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();

        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();

        // Try multiple splits — default_left true/false, several
        // threshold bins, each combined with each feature — and
        // assert equality with CPU for all of them.
        for feature_index in 0..feature_count as u32 {
            for threshold_bin in [0u16, 2, 4, 6] {
                for default_left in [false, true] {
                    let split = SplitCandidate {
                        node_id: 0,
                        feature_index,
                        threshold_bin,
                        gain: 1.0,
                        default_left,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
                            row_count: 0,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 0.0,
                            row_count: 0,
                        },
                    };

                    let cpu = CpuBackend;
                    let cpu_result = cpu.apply_split(&binned_matrix, &node, &split).unwrap();
                    let metal_result = backend.apply_split(&binned_matrix, &node, &split).unwrap();

                    assert_eq!(
                        cpu_result.left_row_indices(),
                        metal_result.left_row_indices(),
                        "left-partition mismatch for feature={feature_index} threshold={threshold_bin} default_left={default_left}"
                    );
                    assert_eq!(
                        cpu_result.right_row_indices(),
                        metal_result.right_row_indices(),
                        "right-partition mismatch for feature={feature_index} threshold={threshold_bin} default_left={default_left}"
                    );
                }
            }
        }
    }

    /// Categorical bitset split must produce bit-identical rows to
    /// the CPU path. Exercises `SPLIT_KIND = 1`.
    #[test]
    fn partition_categorical_matches_cpu() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 1_500usize;
        let feature_count = 2usize;
        let max_bin: u16 = 7;

        let mut bins_row_major = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                let bin = ((row.wrapping_mul(11) ^ feat.wrapping_mul(13)) & 7) as u8;
                bins_row_major.push(bin);
            }
        }
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();

        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();

        // Bitset: bins 1, 3, 5 go left (bit set). Needs 1 byte for
        // max_bin=7 since all bins fit in 8 bits.
        let bitset: Vec<u8> = vec![0b0010_1010];

        let split = SplitCandidate {
            node_id: 0,
            feature_index: 0,
            threshold_bin: 0,
            gain: 1.0,
            default_left: true,
            is_categorical: true,
            categorical_bitset: Some(bitset),
            left_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                row_count: 0,
            },
            right_stats: NodeStats {
                grad_sum: 0.0,
                hess_sum: 0.0,
                row_count: 0,
            },
        };

        let cpu = CpuBackend;
        let cpu_result = cpu.apply_split(&binned_matrix, &node, &split).unwrap();
        let metal_result = backend.apply_split(&binned_matrix, &node, &split).unwrap();

        assert_eq!(
            cpu_result.left_row_indices(),
            metal_result.left_row_indices()
        );
        assert_eq!(
            cpu_result.right_row_indices(),
            metal_result.right_row_indices()
        );
    }

    /// Pipeline cache must return the same Arc on repeated lookups
    /// of the same key, and distinct Arcs across different keys
    /// (matches the pattern used by the histogram + split caches).
    #[test]
    fn partition_pipeline_cache_returns_identical_arc_on_second_call() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let spec_a = kernels::partition::PartitionSpecKey {
            block_size: kernels::partition::BLOCK_SIZE,
            split_kind: 0,
            bin_is_u16: false,
        };
        let first = backend
            .partition_pipeline_cache
            .get_or_build(spec_a)
            .expect("first build");
        let second = backend
            .partition_pipeline_cache
            .get_or_build(spec_a)
            .expect("second build");
        assert!(std::sync::Arc::ptr_eq(&first, &second));

        let spec_b = kernels::partition::PartitionSpecKey {
            block_size: kernels::partition::BLOCK_SIZE,
            split_kind: 1, // categorical variant
            bin_is_u16: false,
        };
        let categorical = backend
            .partition_pipeline_cache
            .get_or_build(spec_b)
            .expect("categorical variant build");
        assert!(!std::sync::Arc::ptr_eq(&first, &categorical));
    }

    // -------- S3.6 — subtract kernel ---------------------------

    /// Bare-library compile smoke test for `subtract.metal`. Mirrors
    /// `partition_shader_compiles` — on some objc2-metal versions a
    /// kernel source with an unused function constant can slip past
    /// the cache build if that specialization is never requested, so
    /// we compile the library directly from source to catch syntax or
    /// type errors up front.
    #[test]
    fn subtract_shader_compiles() {
        let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
            return;
        };
        let source = NSString::from_str(kernels::subtract::SUBTRACT_SHADER_SOURCE);
        device
            .newLibraryWithSource_options_error(&source, None)
            .expect("subtract.metal should compile without errors");
    }

    /// Elementwise subtract must match `subtract_histogram_bundle` on
    /// the CPU bit-for-bit. Float32 / float32 subtraction of exactly-
    /// the-same inputs is deterministic on every known GPU, so we
    /// demand byte equality rather than an epsilon.
    #[test]
    fn subtract_matches_cpu_small_fixture() {
        use alloygbm_core::{FeatureHistogram, HistogramBin, HistogramBundle};

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        // 5 features × 12 bins fixture. Deterministic content that
        // leaves a non-trivial delta in every (feature, bin).
        let feature_count = 5;
        let bin_count = 12;
        let make_bundle = |seed: u32, node_id: u32| -> HistogramBundle {
            let mut fhs = Vec::with_capacity(feature_count);
            for f in 0..feature_count {
                let mut bins = Vec::with_capacity(bin_count);
                for b in 0..bin_count {
                    let n = (seed + f as u32 * 97 + b as u32 * 13) as f32;
                    bins.push(HistogramBin {
                        grad_sum: n * 0.5 + seed as f32 * 0.125,
                        hess_sum: (n + 1.0) * 0.25,
                        count: (seed + f as u32 * 3 + b as u32) * 4,
                    });
                }
                fhs.push(FeatureHistogram {
                    feature_index: f as u32,
                    bins,
                });
            }
            HistogramBundle::from_cpu(node_id, fhs)
        };

        // Construct a proper parent ⊇ child relationship: we build
        // the child first, then derive the parent by adding known
        // deltas. This guarantees `parent_counts >= child_counts`
        // pointwise (the MSL kernel relies on this invariant for the
        // u32 subtraction).
        let child = make_bundle(7, 2);
        let mut parent_fhs = Vec::with_capacity(feature_count);
        for (f, cfh) in child.feature_histograms().iter().enumerate() {
            let mut bins = Vec::with_capacity(bin_count);
            for (b, cb) in cfh.bins.iter().enumerate() {
                bins.push(HistogramBin {
                    grad_sum: cb.grad_sum + 0.75 + (b as f32) * 0.01,
                    hess_sum: cb.hess_sum + 0.5 + (f as f32) * 0.02,
                    count: cb.count + 5 + (b as u32) + (f as u32) * 3,
                });
            }
            parent_fhs.push(FeatureHistogram {
                feature_index: f as u32,
                bins,
            });
        }
        let parent = HistogramBundle::from_cpu(1, parent_fhs);

        let out_node_id = 99;
        let metal_result = kernels::subtract::dispatch_subtract(
            &backend.metal_device,
            &backend.subtract_pipeline_cache,
            &parent,
            &child,
            out_node_id,
        )
        .expect("metal subtract should succeed");

        // CPU reference via the same elementwise loop the engine
        // uses. Hand-inlined to avoid a cross-crate dependency on
        // the engine's private `subtract_histogram_bundle_into`.
        let mut cpu_fhs = Vec::with_capacity(feature_count);
        for (pfh, cfh) in parent
            .feature_histograms()
            .iter()
            .zip(child.feature_histograms())
        {
            let mut bins = Vec::with_capacity(bin_count);
            for (pb, cb) in pfh.bins.iter().zip(&cfh.bins) {
                bins.push(HistogramBin {
                    grad_sum: pb.grad_sum - cb.grad_sum,
                    hess_sum: pb.hess_sum - cb.hess_sum,
                    count: pb.count - cb.count,
                });
            }
            cpu_fhs.push(FeatureHistogram {
                feature_index: pfh.feature_index,
                bins,
            });
        }
        let cpu_result = HistogramBundle::from_cpu(out_node_id, cpu_fhs);

        // Structural equality.
        assert_eq!(metal_result.node_id, out_node_id);
        let metal_fhs = metal_result.feature_histograms();
        let cpu_fhs = cpu_result.feature_histograms();
        assert_eq!(metal_fhs.len(), cpu_fhs.len());
        for (mfh, cfh) in metal_fhs.iter().zip(cpu_fhs) {
            assert_eq!(mfh.feature_index, cfh.feature_index);
            assert_eq!(mfh.bins.len(), cfh.bins.len());
            for (mb, cb) in mfh.bins.iter().zip(&cfh.bins) {
                // Bit-exact: f32 subtract of identical inputs is
                // deterministic across CPU and GPU (IEEE 754 §5.4).
                assert_eq!(mb.grad_sum.to_bits(), cb.grad_sum.to_bits());
                assert_eq!(mb.hess_sum.to_bits(), cb.hess_sum.to_bits());
                assert_eq!(mb.count, cb.count);
            }
        }
    }

    // ---------------------------------------------------------------
    // S3.9 — GPU working-set budget checks.
    // ---------------------------------------------------------------

    /// A realistic fit shape should fit comfortably under the
    /// budget ceiling on every modern Apple-Silicon Mac.
    #[test]
    fn check_histogram_budget_accepts_realistic_shape() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        // 100 features × 256 bins × 64 live nodes × 12 bytes = ~19 MiB.
        // Sub-1-GiB fit shapes should always pass.
        assert!(
            backend.check_histogram_budget(100, 256, 64).is_ok(),
            "realistic shape should fit under the working-set budget"
        );
    }

    /// The plan's M2 pathological shape: leaf-wise + max_leaves=1024 +
    /// 1000 features + 1024 bins. Projects to ~12 GiB, which exceeds
    /// the 80 %-of-recommended ceiling on every sub-M3-Ultra chip.
    ///
    /// On a hypothetical 256+ GiB UMA machine this would actually
    /// fit, so the assertion is shaped as "either reject with a
    /// clear diagnostic, OR accept on a huge-UMA machine". Both are
    /// correct; the test lives to catch silent wrapping / misconfig.
    #[test]
    fn check_histogram_budget_rejects_pathological_shape() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let recommended = backend.recommended_max_working_set_size();
        let ceiling_bytes = recommended * 8 / 10;
        // 12 GiB projection per the M2 risk note.
        let projected_bytes: u64 = 1000 * 1024 * 1024 * 12;
        match backend.check_histogram_budget(1000, 1024, 1024) {
            Ok(()) => {
                // Must only be possible on machines whose ceiling
                // exceeds 12 GiB — sanity-check that assumption.
                assert!(
                    projected_bytes <= ceiling_bytes,
                    "unexpected OK with ceiling {ceiling_bytes} < projected {projected_bytes}"
                );
            }
            Err(alloygbm_engine::EngineError::BackendUnavailable(msg)) => {
                assert!(
                    msg.contains("exceeds the working-set budget"),
                    "diagnostic missing budget phrase: {msg}"
                );
                assert!(
                    msg.contains("device=\"cpu\""),
                    "diagnostic missing cpu-fallback guidance: {msg}"
                );
            }
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// Every Apple-Silicon device reports a non-zero recommended
    /// working-set size; a zero would be a Metal-binding regression.
    #[test]
    fn recommended_working_set_size_is_positive() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        assert!(backend.recommended_max_working_set_size() > 0);
    }

    /// Pipeline cache must return the same Arc on repeated lookups,
    /// mirroring the histogram / split / partition cache contracts.
    #[test]
    fn subtract_pipeline_cache_returns_identical_arc_on_second_call() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let spec = kernels::subtract::SubtractSpecKey {
            block_size: kernels::subtract::BLOCK_SIZE,
        };
        let first = backend
            .subtract_pipeline_cache
            .get_or_build(spec)
            .expect("first build");
        let second = backend
            .subtract_pipeline_cache
            .get_or_build(spec)
            .expect("second build");
        assert!(std::sync::Arc::ptr_eq(&first, &second));
    }

    /// S3.7d — `release_histograms` must decrement the residency
    /// pool's live-entry count on a Gpu bundle and no-op on a Cpu
    /// bundle. Without this, trainer hot loops would leak pool
    /// entries for the full fit duration — silently breaking
    /// `budget::BudgetTracker`'s one-level-wide peak projection.
    #[test]
    fn release_histograms_frees_gpu_pool_entry() {
        use alloygbm_backend_cpu::CpuBackend;
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return; // no Metal device — skip.
        };

        // Tiny fixture: enough to make `build_histograms` actually
        // mint a pool entry. Shape doesn't matter beyond being legal.
        let row_count = 32usize;
        let feature_count = 2usize;
        let max_bin: u16 = 3;

        let mut bins_row_major = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for feat in 0..feature_count {
                bins_row_major.push(((row + feat) & 3) as u8);
            }
        }
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();
        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|_| GradientPair::new(1.0, 1.0).unwrap())
            .collect();
        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(0, feature_count as u32).unwrap()];

        // Baseline: pool empty before any build.
        assert_eq!(backend.histogram_residency.live_count(), 0);

        // Build two bundles; pool should grow to 2.
        let bundle_a = backend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("metal histogram A");
        let bundle_b = backend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("metal histogram B");
        assert_eq!(
            backend.histogram_residency.live_count(),
            2,
            "two build_histograms calls must mint two pool entries"
        );

        // Release A; pool should drop to 1. B still live.
        backend
            .release_histograms(&bundle_a)
            .expect("release bundle A");
        assert_eq!(
            backend.histogram_residency.live_count(),
            1,
            "release_histograms on Gpu bundle must decrement live_count"
        );

        // Release B; pool should drop to 0.
        backend
            .release_histograms(&bundle_b)
            .expect("release bundle B");
        assert_eq!(
            backend.histogram_residency.live_count(),
            0,
            "second release must decrement live_count to 0"
        );

        // A Cpu-variant bundle must be a no-op: doesn't try to
        // look up a pool handle, doesn't error. This matters when
        // the trainer processes a Cpu fallback path within a
        // Metal fit (doesn't happen today but the contract needs
        // to hold).
        let cpu = CpuBackend;
        let cpu_bundle = cpu
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("cpu bundle");
        backend
            .release_histograms(&cpu_bundle)
            .expect("cpu bundle release must be a no-op, not an error");
        assert_eq!(
            backend.histogram_residency.live_count(),
            0,
            "releasing a Cpu bundle must not touch the Gpu pool"
        );

        // Keep bundles alive to here so Drop order doesn't surprise
        // the test — the release calls above are the semantic gate.
        drop(bundle_a);
        drop(bundle_b);
        drop(cpu_bundle);
    }

    /// S3.7d — double-release of the same handle must be a no-op
    /// (matches `HistogramResidencyPool::release`'s HashMap::remove
    /// semantics). The `HistogramReleaseGuard` in the trainer may
    /// hand the backend the same `&HistogramBundle` twice in
    /// pathological cases (e.g. a future refactor that adds a
    /// redundant guard); the contract is: idempotent release,
    /// never panic, never corrupt pool state.
    #[test]
    fn release_histograms_is_idempotent_on_gpu_bundle() {
        use alloygbm_engine::BackendOps;

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 16usize;
        let feature_count = 2usize;
        let max_bin: u16 = 3;
        let bins_row_major: Vec<u8> = (0..row_count * feature_count)
            .map(|i| (i & 3) as u8)
            .collect();
        let binned_matrix =
            BinnedMatrix::new(row_count, feature_count, max_bin, bins_row_major).unwrap();
        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|_| GradientPair::new(1.0, 1.0).unwrap())
            .collect();
        let row_indices: Vec<u32> = (0..row_count as u32).collect();
        let node = NodeSlice::new(0, row_indices).unwrap();
        let tiles = vec![FeatureTile::new(0, feature_count as u32).unwrap()];

        let bundle = backend
            .build_histograms(&binned_matrix, &gradients, &node, &tiles)
            .expect("metal histogram");
        assert_eq!(backend.histogram_residency.live_count(), 1);

        backend.release_histograms(&bundle).expect("first release");
        assert_eq!(backend.histogram_residency.live_count(), 0);

        // Second release on the same (now-dead) handle must not
        // panic or error; pool treats unknown handles as no-ops.
        backend
            .release_histograms(&bundle)
            .expect("idempotent release");
        assert_eq!(backend.histogram_residency.live_count(), 0);
    }
}
