//! GPU working-set budget enforcement (Stage 3 M2 policy).
//!
//! # What this enforces
//!
//! Stage 3's memory policy (D-016) is **free-on-consume**: GPU
//! histograms live for exactly one level (level-wise) or until the
//! `PendingSplit` pops (leaf-wise). Peak residency is therefore
//! bounded analytically by
//!
//! ```text
//! peak_bytes = n_features × bin_count × level_width × 12
//! ```
//!
//! where 12 bytes = `grad (f32) + hess (f32) + counts (u32)` per
//! histogram cell. `level_width` is the number of *live* nodes whose
//! histograms must be simultaneously resident; with the subtraction
//! trick in play this equals the count of smaller-sibling nodes at
//! the widest level the trainer reaches.
//!
//! `BudgetTracker::check_projected_peak` refuses the fit when
//! projected peak exceeds **80 %** of
//! `MTLDevice.recommendedMaxWorkingSetSize`. We leave a 20 %
//! headroom for (a) `split` / `partition` kernel scratch buffers,
//! (b) the binned matrix + gradient staging that survive across
//! fits, and (c) OS-side working-set churn.
//!
//! # M2 pathological case (risk note)
//!
//! The policy's worst-case shape is leaf-wise with
//! `max_leaves = 1024`, 1000 features, and 1024 bins, which projects
//! to ~12 GiB of histogram residency. That exceeds
//! `recommendedMaxWorkingSetSize` on every M-series chip below
//! M3 Max 36+ GiB, so the budget guard **refuses** the fit on
//! those chips rather than proceed into swap.
//!
//! **This is strictly better than the status quo:** CPU training
//! pages to swap in this shape (slow but survives);
//! `BudgetTracker` emits a clear, actionable error pointing at
//! `device="cpu"` as the immediate workaround and at M3 (probe-
//! detected LRU spill via `ALLOYGBM_METAL_HISTOGRAM_BUDGET_GIB`)
//! as the planned architectural follow-up. The explicit-refusal
//! vs. silent-swap trade-off is a deliberate call — see
//! `docs/metal-backend/DECISIONS.md` D-016.
//!
//! # Why 80 %, not 100 %
//!
//! `recommendedMaxWorkingSetSize` is Apple's *recommended* ceiling
//! — allocating up to that threshold is safe but leaves zero room
//! for other GPU consumers (window compositor, other Metal apps,
//! ML Neural Engine sharing unified memory). 80 % keeps the system
//! responsive while still using most of the budget for training.

#![allow(unsafe_code)]

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use alloygbm_engine::{EngineError, EngineResult};

/// 12 bytes per histogram cell: f32 grad + f32 hess + u32 counts.
const HISTOGRAM_CELL_BYTES: u64 = 12;

/// Headroom factor. Peak residency must stay below
/// `ceiling × recommendedMaxWorkingSetSize`.
const CEILING_FRACTION_BPS: u64 = 8_000; // 80.00 % in basis points

/// Immutable budget snapshot captured once per `MetalBackend`.
///
/// `dead_code` allow: consumer is S3.7 + S3.3 — the trainer plumbs
/// `(n_features, bin_count, max_level_width)` into a fit-start
/// precondition check. The tracker ships in isolation so it can be
/// unit-tested without the trainer refactor.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BudgetTracker {
    /// Apple's recommended cap on total working-set bytes for this
    /// device. Stable across the device's lifetime.
    recommended_max_working_set_size: u64,
}

#[allow(dead_code)]
impl BudgetTracker {
    /// Capture the device's `recommendedMaxWorkingSetSize`. Called
    /// once per `MetalBackend::new()` — the value does not change
    /// at runtime.
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            recommended_max_working_set_size: device.recommendedMaxWorkingSetSize(),
        }
    }

    /// Budget ceiling in bytes (80 % of `recommendedMaxWorkingSetSize`).
    pub(crate) fn ceiling_bytes(&self) -> u64 {
        self.recommended_max_working_set_size * CEILING_FRACTION_BPS / 10_000
    }

    /// Raw device cap. Surfaced for diagnostic messages.
    pub(crate) fn recommended_max_working_set_size(&self) -> u64 {
        self.recommended_max_working_set_size
    }

    /// Project peak histogram residency bytes for the given fit
    /// shape. Saturating arithmetic so a pathological product
    /// doesn't silently wrap — `u64::MAX` is a perfectly valid
    /// "clearly too large" sentinel that propagates to the
    /// ceiling check.
    pub(crate) fn project_peak_bytes(n_features: u32, bin_count: u32, level_width: u32) -> u64 {
        (n_features as u64)
            .saturating_mul(bin_count as u64)
            .saturating_mul(level_width as u64)
            .saturating_mul(HISTOGRAM_CELL_BYTES)
    }

    /// Pre-fit budget check. Returns `Ok(())` when projected peak is
    /// under the ceiling; returns `EngineError::BackendUnavailable`
    /// with an actionable diagnostic otherwise.
    ///
    /// Using `BackendUnavailable` (rather than a dedicated
    /// `InsufficientMemory` variant) keeps the error-machinery
    /// footprint at zero crates-touched for S3.9 and naturally
    /// composes with the existing Python-layer fallback-to-CPU
    /// path: when this error bubbles out of `MetalBackend::new()`
    /// or a fit-start guard, the Python estimator emits
    /// `RuntimeWarning` and re-runs under `device="cpu"`.
    pub(crate) fn check_projected_peak(
        &self,
        n_features: u32,
        bin_count: u32,
        level_width: u32,
    ) -> EngineResult<()> {
        let peak = Self::project_peak_bytes(n_features, bin_count, level_width);
        let ceiling = self.ceiling_bytes();
        if peak <= ceiling {
            return Ok(());
        }
        // Format the numbers in GiB for human readability and keep
        // the device-side cap in the message so the user can see
        // *why* the refusal happened, not just *that* it did.
        let peak_gib = peak as f64 / (1024.0 * 1024.0 * 1024.0);
        let ceiling_gib = ceiling as f64 / (1024.0 * 1024.0 * 1024.0);
        let recommended_gib =
            self.recommended_max_working_set_size as f64 / (1024.0 * 1024.0 * 1024.0);
        Err(EngineError::BackendUnavailable(format!(
            "Metal backend: projected histogram residency ({peak_gib:.2} GiB for \
             {n_features} features × {bin_count} bins × {level_width} live nodes) \
             exceeds the working-set budget ({ceiling_gib:.2} GiB, 80 % of the \
             device-recommended {recommended_gib:.2} GiB). \
             Retry with device=\"cpu\", with a smaller max_depth/num_leaves, or with \
             a smaller feature subset. Probe-detected LRU spill is the planned \
             follow-up (see docs/metal-backend/DECISIONS.md M3 roadmap)."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_peak_matches_closed_form() {
        // 10 features × 256 bins × 4 live nodes × 12 bytes = 122_880
        assert_eq!(BudgetTracker::project_peak_bytes(10, 256, 4), 122_880);
    }

    #[test]
    fn project_peak_saturates_on_overflow() {
        // u32::MAX × u32::MAX × u32::MAX would overflow u64; saturate
        // instead of wrap.
        let peak = BudgetTracker::project_peak_bytes(u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(peak, u64::MAX);
    }

    #[test]
    fn check_accepts_small_fit_within_budget() {
        let tracker = BudgetTracker {
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024, // 8 GiB
        };
        // 100 × 256 × 64 × 12 = ~200 MiB — well under 6.4 GiB ceiling
        assert!(tracker.check_projected_peak(100, 256, 64).is_ok());
    }

    #[test]
    fn check_rejects_pathological_leaf_wise_shape() {
        let tracker = BudgetTracker {
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024, // 8 GiB
        };
        // Plan's M2 pathological case:
        //   1000 features × 1024 bins × 1024 leaves × 12 = ~12 GiB
        let err = tracker
            .check_projected_peak(1000, 1024, 1024)
            .expect_err("12 GiB should exceed 6.4 GiB ceiling");
        match err {
            EngineError::BackendUnavailable(msg) => {
                assert!(msg.contains("exceeds the working-set budget"));
                assert!(msg.contains("device=\"cpu\""));
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn ceiling_is_eighty_percent_of_recommended() {
        let tracker = BudgetTracker {
            recommended_max_working_set_size: 10_000,
        };
        assert_eq!(tracker.ceiling_bytes(), 8_000);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn tracker_reads_device_working_set_size() {
        use crate::device::MetalDevice;
        let Ok(metal_device) = MetalDevice::probe() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let tracker = BudgetTracker::new(&metal_device.device);
        // Every real Apple-Silicon GPU reports a non-zero recommended
        // working-set size; a zero here would be a Metal-binding
        // regression we want to catch.
        assert!(tracker.recommended_max_working_set_size() > 0);
        assert!(tracker.ceiling_bytes() > 0);
        assert!(tracker.ceiling_bytes() < tracker.recommended_max_working_set_size());
    }
}
