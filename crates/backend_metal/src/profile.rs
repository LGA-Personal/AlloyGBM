//! Lightweight per-call-site timing for Stage 3 Metal hot-path diagnosis.
//!
//! # Why this exists (temporary)
//!
//! Stage 3 as shipped through S3.11 met its correctness gate but
//! did NOT cross the `metal_friendly` >1.0× CPU kill criterion
//! (S3.12). The diagnosis in `DECISIONS.md:D-020` identifies three
//! residual readback paths — `build_histograms` count
//! accumulation, `reduce_sums`, `apply_partition_leaf_updates` —
//! as the suspected dominant cost. This module instruments those
//! sites (plus the GPU-native sites as a control) with
//! `Instant::now()` probes and atomic accumulators so we can
//! **measure** the breakdown before writing more kernels.
//!
//! Gating: probes always record; `dump_if_enabled()` only prints
//! if `ALLOYGBM_METAL_PROFILE=1`. The dump fires from
//! `Drop for MetalBackend`, which is the last thing that runs at
//! the end of a fit (the Python estimator constructs a new
//! backend per fit).
//!
//! Overhead: `Instant::now()` on macOS aarch64 is ~5–10 ns per
//! call via `mach_absolute_time`. Our hottest sites fire at most
//! a few thousand times per fit and take milliseconds each, so
//! instrumentation noise is well below 1%.
//!
//! This module should be retired once the readback-closure work
//! is complete and Stage 3 meets its kill criterion.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Atomic (calls, total-nanos) counter for one call site. Safe
/// to increment from any thread; the Metal backend is
/// single-producer today but that may change.
pub(crate) struct Counter {
    calls: AtomicU64,
    total_ns: AtomicU64,
}

impl Counter {
    pub(crate) const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
        }
    }

    pub(crate) fn record_ns(&self, ns: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, u64) {
        (
            self.calls.load(Ordering::Relaxed),
            self.total_ns.load(Ordering::Relaxed),
        )
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.total_ns.store(0, Ordering::Relaxed);
    }
}

/// RAII probe: start on construction, record on drop.
pub(crate) struct ScopedProbe<'a> {
    counter: &'a Counter,
    start: Instant,
}

impl<'a> ScopedProbe<'a> {
    pub(crate) fn new(counter: &'a Counter) -> Self {
        Self {
            counter,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedProbe<'_> {
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.counter.record_ns(ns);
    }
}

// ---- Top-level BackendOps method probes ---------------------------

pub(crate) static BUILD_HISTOGRAMS: Counter = Counter::new();
pub(crate) static BEST_SPLIT: Counter = Counter::new();
pub(crate) static BEST_SPLIT_WITH_OPTIONS: Counter = Counter::new();
pub(crate) static SUBTRACT: Counter = Counter::new();
pub(crate) static APPLY_SPLIT: Counter = Counter::new();
pub(crate) static APPLY_SPLIT_WITH_STATS: Counter = Counter::new();
pub(crate) static REDUCE_SUMS: Counter = Counter::new();
pub(crate) static APPLY_PARTITION_LEAF_UPDATES: Counter = Counter::new();
pub(crate) static RELEASE_HISTOGRAMS: Counter = Counter::new();
pub(crate) static RELEASE_ROW_INDICES: Counter = Counter::new();

// ---- Batched call-site probes (D-023) -----------------------------
//
// `BUILD_HISTOGRAMS_BATCH` wraps the whole batched call (encoding +
// commit + waitUntilCompleted + count finalisation). Per-request
// sub-phase work continues to record into the existing `BH_*`
// counters; in the batched path those represent aggregated work
// across the whole batch rather than one per call.
pub(crate) static BUILD_HISTOGRAMS_BATCH: Counter = Counter::new();
pub(crate) static SUBTRACT_BATCH: Counter = Counter::new();

// ---- build_histograms sub-phases (the suspected cost) -------------

pub(crate) static BH_GPU_DISPATCH: Counter = Counter::new();
pub(crate) static BH_ROW_READBACK: Counter = Counter::new();
pub(crate) static BH_COUNT_ACCUMULATE: Counter = Counter::new();
pub(crate) static BH_BUFFER_SETUP: Counter = Counter::new();

// Finer sub-probes inside BH_GPU_DISPATCH (D-021 second pass).
// These overlap with BH_GPU_DISPATCH; reported as a second-level
// indent in dump_if_enabled.
pub(crate) static BH_SCRATCH_ALLOC: Counter = Counter::new();
pub(crate) static BH_ENCODE: Counter = Counter::new();
pub(crate) static BH_COMMIT_WAIT: Counter = Counter::new();

// ---- apply_split sub-phases ---------------------------------------

pub(crate) static AS_GPU_DISPATCH: Counter = Counter::new();

// ---- reduce_sums / leaf_updates sub-phases ------------------------

pub(crate) static RS_READBACK: Counter = Counter::new();
pub(crate) static RS_CPU_REDUCE: Counter = Counter::new();
pub(crate) static PLU_READBACK: Counter = Counter::new();
pub(crate) static PLU_CPU_UPDATE: Counter = Counter::new();

/// Environment gate. When `ALLOYGBM_METAL_PROFILE=1` is set at
/// the time of `MetalBackend::drop`, this module emits a
/// per-call-site breakdown on stderr.
pub(crate) fn profiling_enabled() -> bool {
    std::env::var("ALLOYGBM_METAL_PROFILE").ok().as_deref() == Some("1")
}

/// Print the per-call-site breakdown to stderr in a human-readable
/// table, then reset all counters so the next fit starts fresh.
///
/// Called from `Drop for MetalBackend`. No-op unless
/// `profiling_enabled()`.
pub(crate) fn dump_if_enabled() {
    if !profiling_enabled() {
        return;
    }
    struct Site {
        name: &'static str,
        counter: &'static Counter,
        indented: bool,
    }
    let sites = [
        Site {
            name: "build_histograms",
            counter: &BUILD_HISTOGRAMS,
            indented: false,
        },
        Site {
            name: "build_histograms_batch",
            counter: &BUILD_HISTOGRAMS_BATCH,
            indented: false,
        },
        Site {
            name: "  .buffer_setup",
            counter: &BH_BUFFER_SETUP,
            indented: true,
        },
        Site {
            name: "  .gpu_dispatch",
            counter: &BH_GPU_DISPATCH,
            indented: true,
        },
        Site {
            name: "    ..scratch_alloc",
            counter: &BH_SCRATCH_ALLOC,
            indented: true,
        },
        Site {
            name: "    ..encode",
            counter: &BH_ENCODE,
            indented: true,
        },
        Site {
            name: "    ..commit_wait",
            counter: &BH_COMMIT_WAIT,
            indented: true,
        },
        Site {
            name: "  .row_readback",
            counter: &BH_ROW_READBACK,
            indented: true,
        },
        Site {
            name: "  .count_accumulate",
            counter: &BH_COUNT_ACCUMULATE,
            indented: true,
        },
        Site {
            name: "best_split",
            counter: &BEST_SPLIT,
            indented: false,
        },
        Site {
            name: "best_split_with_options",
            counter: &BEST_SPLIT_WITH_OPTIONS,
            indented: false,
        },
        Site {
            name: "subtract_histogram_bundle",
            counter: &SUBTRACT,
            indented: false,
        },
        Site {
            name: "subtract_histogram_bundle_batch",
            counter: &SUBTRACT_BATCH,
            indented: false,
        },
        Site {
            name: "apply_split",
            counter: &APPLY_SPLIT,
            indented: false,
        },
        Site {
            name: "  .gpu_dispatch",
            counter: &AS_GPU_DISPATCH,
            indented: true,
        },
        Site {
            name: "apply_split_with_stats",
            counter: &APPLY_SPLIT_WITH_STATS,
            indented: false,
        },
        Site {
            name: "reduce_sums",
            counter: &REDUCE_SUMS,
            indented: false,
        },
        Site {
            name: "  .readback",
            counter: &RS_READBACK,
            indented: true,
        },
        Site {
            name: "  .cpu_reduce",
            counter: &RS_CPU_REDUCE,
            indented: true,
        },
        Site {
            name: "apply_partition_leaf_updates",
            counter: &APPLY_PARTITION_LEAF_UPDATES,
            indented: false,
        },
        Site {
            name: "  .readback",
            counter: &PLU_READBACK,
            indented: true,
        },
        Site {
            name: "  .cpu_update",
            counter: &PLU_CPU_UPDATE,
            indented: true,
        },
        Site {
            name: "release_histograms",
            counter: &RELEASE_HISTOGRAMS,
            indented: false,
        },
        Site {
            name: "release_row_indices",
            counter: &RELEASE_ROW_INDICES,
            indented: false,
        },
    ];

    // Compute non-indented total to get percentages.
    let total_top_ns: u64 = sites
        .iter()
        .filter(|s| !s.indented)
        .map(|s| s.counter.snapshot().1)
        .sum();

    eprintln!();
    eprintln!("==== AlloyGBM Metal profile (ALLOYGBM_METAL_PROFILE=1) ====");
    eprintln!(
        "{:<34} {:>10} {:>14} {:>12} {:>8}",
        "call site", "calls", "total_ms", "avg_us", "% total"
    );
    eprintln!("{}", "-".repeat(80));
    for site in sites.iter() {
        let (calls, total_ns) = site.counter.snapshot();
        if calls == 0 {
            continue;
        }
        let total_ms = total_ns as f64 / 1e6;
        let avg_us = (total_ns as f64 / calls as f64) / 1e3;
        let pct = if site.indented || total_top_ns == 0 {
            String::from("   -")
        } else {
            format!("{:>6.1}%", 100.0 * total_ns as f64 / total_top_ns as f64)
        };
        eprintln!(
            "{:<34} {:>10} {:>14.2} {:>12.2} {:>8}",
            site.name, calls, total_ms, avg_us, pct
        );
    }
    eprintln!("{}", "-".repeat(80));
    eprintln!(
        "top-level total = {:.2} ms (sub-phases are indented and not double-counted)",
        total_top_ns as f64 / 1e6
    );
    eprintln!();

    // Reset so the next fit starts clean.
    for site in sites.iter() {
        site.counter.reset();
    }
}
