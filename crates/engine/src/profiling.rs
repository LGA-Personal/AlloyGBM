//! Opt-in per-stage timing for the training loop.
//!
//! Scaling work is only as good as its measurements: guessing which stage
//! dominates produced a "speedup" that was slower in absolute terms. Set
//! `ALLOYGBM_PROFILE=1` to get a per-stage breakdown of a fit on stderr, so
//! optimization targets the stage that actually costs.
//!
//! Disabled by default and gated behind a single atomic load, so the
//! instrumentation is free when it is off.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ALLOYGBM_PROFILE")
            .map(|value| value != "0" && !value.is_empty())
            .unwrap_or(false)
    })
}

/// Stages of a single boosting round, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    Gradients,
    RowSampling,
    FeatureTiles,
    PredictionCopy,
    TreeBuild,
    PredictionUpdate,
    Loss,
    Validation,
}

impl Stage {
    const ALL: [Stage; 8] = [
        Stage::Gradients,
        Stage::RowSampling,
        Stage::FeatureTiles,
        Stage::PredictionCopy,
        Stage::TreeBuild,
        Stage::PredictionUpdate,
        Stage::Loss,
        Stage::Validation,
    ];

    fn label(self) -> &'static str {
        match self {
            Stage::Gradients => "gradients",
            Stage::RowSampling => "row_sampling",
            Stage::FeatureTiles => "feature_tiles",
            Stage::PredictionCopy => "prediction_copy",
            Stage::TreeBuild => "tree_build",
            Stage::PredictionUpdate => "prediction_update",
            Stage::Loss => "loss",
            Stage::Validation => "validation",
        }
    }
}

/// Accumulates per-stage durations across the rounds of one fit.
#[derive(Debug, Default)]
pub(crate) struct RoundProfile {
    totals: [Duration; 8],
    rounds: usize,
}

impl RoundProfile {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Time `f`, attributing the elapsed duration to `stage`.
    pub(crate) fn time<T>(&mut self, stage: Stage, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let started = Instant::now();
        let value = f();
        self.totals[stage as usize] += started.elapsed();
        value
    }

    pub(crate) fn note_round(&mut self) {
        if enabled() {
            self.rounds += 1;
        }
    }

    /// Print the breakdown to stderr, largest stage first.
    pub(crate) fn report(&self, rows: usize, features: usize) {
        if !enabled() || self.rounds == 0 {
            return;
        }
        let total: Duration = self.totals.iter().sum();
        let total_secs = total.as_secs_f64().max(f64::MIN_POSITIVE);
        let mut stages: Vec<(Stage, Duration)> = Stage::ALL
            .iter()
            .map(|stage| (*stage, self.totals[*stage as usize]))
            .filter(|(_, elapsed)| !elapsed.is_zero())
            .collect();
        stages.sort_by(|a, b| b.1.cmp(&a.1));

        eprintln!(
            "\n[alloygbm profile] rows={rows} features={features} rounds={} threads={} measured={:.3}s",
            self.rounds,
            rayon::current_num_threads(),
            total_secs,
        );
        for (stage, elapsed) in stages {
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "  {:<18} {:8.3}s  {:5.1}%  {:8.3} ms/round",
                stage.label(),
                secs,
                100.0 * secs / total_secs,
                1000.0 * secs / self.rounds as f64,
            );
        }
        report_tree_stages();
    }
}

// ── Sub-stage counters ───────────────────────────────────────────────────
//
// Tree building runs across Rayon workers, so its internal stages cannot use
// the single-threaded `RoundProfile`. These global nanosecond counters are
// cheap (one relaxed fetch_add per timed region, and nothing at all when
// profiling is off) and are reported alongside the round breakdown.

static HISTOGRAM_NS: AtomicU64 = AtomicU64::new(0);
static SPLIT_FIND_NS: AtomicU64 = AtomicU64::new(0);
static PARTITION_NS: AtomicU64 = AtomicU64::new(0);

/// Internal stages of tree construction.
#[derive(Debug, Clone, Copy)]
pub enum TreeStage {
    Histogram,
    SplitFind,
    Partition,
}

impl TreeStage {
    fn counter(self) -> &'static AtomicU64 {
        match self {
            TreeStage::Histogram => &HISTOGRAM_NS,
            TreeStage::SplitFind => &SPLIT_FIND_NS,
            TreeStage::Partition => &PARTITION_NS,
        }
    }
}

/// Time `f` and attribute it to a tree-building sub-stage.
///
/// Note the totals sum wall-clock across workers, so on N threads they can
/// exceed the elapsed round time; read them as shares of tree-building work,
/// not as elapsed time.
pub fn time_tree_stage<T>(stage: TreeStage, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let value = f();
    stage
        .counter()
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    value
}

pub(crate) fn report_tree_stages() {
    if !enabled() {
        return;
    }
    let stages = [
        ("histogram_build", HISTOGRAM_NS.swap(0, Ordering::Relaxed)),
        ("split_find", SPLIT_FIND_NS.swap(0, Ordering::Relaxed)),
        ("partition", PARTITION_NS.swap(0, Ordering::Relaxed)),
    ];
    let total: u64 = stages.iter().map(|(_, ns)| *ns).sum();
    if total == 0 {
        return;
    }
    eprintln!("  -- tree_build internals (summed across workers) --");
    let mut sorted = stages;
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (label, ns) in sorted {
        eprintln!(
            "  {:<18} {:8.3}s  {:5.1}%",
            label,
            ns as f64 / 1e9,
            100.0 * ns as f64 / total as f64
        );
    }
}
