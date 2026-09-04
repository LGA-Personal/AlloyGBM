//! Opt-in per-stage timing for the training loop.
//!
//! Profiling is disabled unless `ALLOYGBM_PROFILE` is `human`, `json`, or the
//! backwards-compatible value `1`. The disabled path does not create timing
//! objects or touch the tree-worker counters.

use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const JSON_PREFIX: &str = "[alloygbm profile json] ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileMode {
    Disabled,
    Human,
    Json,
}

pub(crate) fn parse_profile_mode(value: Option<&str>) -> ProfileMode {
    match value {
        Some("1" | "human") => ProfileMode::Human,
        Some("json") => ProfileMode::Json,
        _ => ProfileMode::Disabled,
    }
}

fn profile_mode() -> ProfileMode {
    static MODE: OnceLock<ProfileMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_profile_mode(std::env::var("ALLOYGBM_PROFILE").ok().as_deref()))
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

/// A frozen, machine-readable profile for one training loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainingProfileSnapshot {
    pub rows: usize,
    pub features: usize,
    pub rounds: usize,
    pub threads: usize,
    pub loop_wall_ns: u64,
    pub untimed_ns: u64,
    pub stage_ns: BTreeMap<String, u64>,
    pub tree_stage_ns: BTreeMap<String, u64>,
}

/// Accumulates per-stage durations across the rounds of one fit.
#[derive(Debug)]
pub(crate) struct RoundProfile {
    mode: ProfileMode,
    totals: [Duration; 8],
    rounds: usize,
    started: Option<Instant>,
}

impl Default for RoundProfile {
    fn default() -> Self {
        Self {
            mode: profile_mode(),
            totals: [Duration::ZERO; 8],
            rounds: 0,
            started: None,
        }
    }
}

impl RoundProfile {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_mode(mode: ProfileMode) -> Self {
        Self {
            mode,
            totals: [Duration::ZERO; 8],
            rounds: 0,
            started: None,
        }
    }

    /// Time `f`, attributing the elapsed duration to `stage`.
    pub(crate) fn time<T>(&mut self, stage: Stage, f: impl FnOnce() -> T) -> T {
        if self.mode == ProfileMode::Disabled {
            return f();
        }
        let started = Instant::now();
        let value = f();
        self.totals[stage as usize] += started.elapsed();
        value
    }

    pub(crate) fn note_round(&mut self) {
        if self.mode != ProfileMode::Disabled {
            self.started.get_or_insert_with(Instant::now);
            self.rounds += 1;
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.mode != ProfileMode::Disabled
    }

    pub(crate) fn snapshot_if_enabled(
        &self,
        rows: usize,
        features: usize,
        threads: usize,
    ) -> Option<TrainingProfileSnapshot> {
        self.is_enabled()
            .then(|| self.snapshot(rows, features, threads))
    }

    /// Freeze loop timing and drain the worker counters exactly once.
    pub(crate) fn snapshot(
        &self,
        rows: usize,
        features: usize,
        threads: usize,
    ) -> TrainingProfileSnapshot {
        let enabled = self.mode != ProfileMode::Disabled;
        let stage_ns: BTreeMap<String, u64> = Stage::ALL
            .iter()
            .map(|stage| {
                (
                    stage.label().to_string(),
                    duration_ns(self.totals[*stage as usize]),
                )
            })
            .collect();
        let staged_ns = stage_ns.values().copied().fold(0_u64, u64::saturating_add);
        let loop_wall_ns = if enabled {
            self.started
                .map(|started| duration_ns(started.elapsed()))
                .unwrap_or(staged_ns)
        } else {
            0
        };
        let tree_stage_ns = if enabled {
            drain_tree_counters()
        } else {
            tree_stage_map([0, 0, 0])
        };
        TrainingProfileSnapshot {
            rows,
            features,
            rounds: self.rounds,
            threads,
            loop_wall_ns,
            untimed_ns: loop_wall_ns.saturating_sub(staged_ns),
            stage_ns,
            tree_stage_ns,
        }
    }

    /// Print a frozen breakdown to stderr.
    pub(crate) fn report(&self, snapshot: Option<&TrainingProfileSnapshot>) {
        let Some(snapshot) = snapshot else {
            return;
        };
        if self.mode == ProfileMode::Disabled || snapshot.rounds == 0 {
            return;
        }
        if self.mode == ProfileMode::Json {
            let json =
                serde_json::to_string(snapshot).expect("training profile snapshot must serialize");
            eprintln!("{JSON_PREFIX}{json}");
            return;
        }
        let total_secs = (snapshot.loop_wall_ns as f64 / 1e9).max(f64::MIN_POSITIVE);
        let total = snapshot.loop_wall_ns;
        let mut stages: Vec<(Option<&str>, u64)> = snapshot
            .stage_ns
            .iter()
            .map(|(label, elapsed)| (Some(label.as_str()), *elapsed))
            .filter(|(_, elapsed)| *elapsed != 0)
            .collect();
        stages.push((None, snapshot.untimed_ns));
        stages.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!(
            "\n[alloygbm profile] rows={} features={} rounds={} threads={} loop_wall={:.3}s",
            snapshot.rows, snapshot.features, snapshot.rounds, snapshot.threads, total_secs,
        );
        for (stage, elapsed) in stages {
            let secs = elapsed as f64 / 1e9;
            eprintln!(
                "  {:<18} {:8.3}s  {:5.1}%  {:8.3} ms/round",
                stage.unwrap_or("other (untimed)"),
                secs,
                if total == 0 {
                    0.0
                } else {
                    100.0 * elapsed as f64 / total as f64
                },
                1000.0 * secs / snapshot.rounds as f64,
            );
        }
        report_tree_stages(&snapshot.tree_stage_ns);
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn tree_stage_map(values: [u64; 3]) -> BTreeMap<String, u64> {
    ["histogram_build", "split_find", "partition"]
        .into_iter()
        .zip(values)
        .map(|(label, value)| (label.to_string(), value))
        .collect()
}

// Tree building runs across Rayon workers, so its internal stages use global
// relaxed counters and are drained once when a fit snapshot is frozen.
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
pub fn time_tree_stage<T>(stage: TreeStage, f: impl FnOnce() -> T) -> T {
    if profile_mode() == ProfileMode::Disabled {
        return f();
    }
    let started = Instant::now();
    let value = f();
    stage
        .counter()
        .fetch_add(duration_ns(started.elapsed()), Ordering::Relaxed);
    value
}

fn drain_tree_counters() -> BTreeMap<String, u64> {
    tree_stage_map([
        HISTOGRAM_NS.swap(0, Ordering::Relaxed),
        SPLIT_FIND_NS.swap(0, Ordering::Relaxed),
        PARTITION_NS.swap(0, Ordering::Relaxed),
    ])
}

fn report_tree_stages(values: &BTreeMap<String, u64>) {
    let total: u64 = values.values().copied().fold(0, u64::saturating_add);
    if total == 0 {
        return;
    }
    eprintln!("  -- tree_build internals (summed across workers) --");
    let mut stages: Vec<_> = values.iter().collect();
    stages.sort_by(|a, b| b.1.cmp(a.1));
    for (label, ns) in stages {
        eprintln!(
            "  {:<18} {:8.3}s  {:5.1}%",
            label,
            *ns as f64 / 1e9,
            100.0 * *ns as f64 / total as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::thread;

    static PROFILE_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn snapshot_contains_exact_stage_labels_and_zero_values() {
        let _guard = PROFILE_TEST_LOCK.lock().unwrap();
        let profile = RoundProfile::with_mode(ProfileMode::Json);
        let snapshot = profile.snapshot(12, 3, 2);
        assert_eq!(snapshot.rows, 12);
        assert_eq!(snapshot.features, 3);
        assert_eq!(snapshot.rounds, 0);
        assert_eq!(snapshot.threads, 2);
        assert_eq!(snapshot.stage_ns.len(), 8);
        assert!(snapshot.stage_ns.values().all(|value| *value == 0));
        assert_eq!(snapshot.tree_stage_ns.len(), 3);
        assert!(snapshot.tree_stage_ns.values().all(|value| *value == 0));
    }

    #[test]
    fn disabled_profile_does_not_construct_a_snapshot() {
        let profile = RoundProfile::with_mode(ProfileMode::Disabled);
        assert!(profile.snapshot_if_enabled(12, 3, 2).is_none());
    }

    #[test]
    fn snapshot_saturates_untimed_residual() {
        let _guard = PROFILE_TEST_LOCK.lock().unwrap();
        let mut profile = RoundProfile::with_mode(ProfileMode::Json);
        profile.rounds = 1;
        profile.started = Some(Instant::now());
        profile.totals[Stage::Gradients as usize] = Duration::from_secs(1);
        let snapshot = profile.snapshot(1, 1, 1);
        assert_eq!(snapshot.untimed_ns, 0);
    }

    #[test]
    fn snapshot_wall_time_is_frozen_before_later_work() {
        let _guard = PROFILE_TEST_LOCK.lock().unwrap();
        let mut profile = RoundProfile::with_mode(ProfileMode::Json);
        profile.note_round();
        let snapshot = profile.snapshot(1, 1, 1);
        thread::sleep(Duration::from_millis(2));
        profile.totals[Stage::Gradients as usize] = Duration::from_secs(1);
        assert!(snapshot.loop_wall_ns < 1_000_000_000);
        assert_eq!(snapshot.stage_ns["gradients"], 0);
        let later_snapshot = profile.snapshot(1, 1, 1);
        assert_eq!(later_snapshot.stage_ns["gradients"], 1_000_000_000);
    }

    #[test]
    fn profile_mode_parsing_is_explicit() {
        assert_eq!(parse_profile_mode(None), ProfileMode::Disabled);
        assert_eq!(parse_profile_mode(Some("")), ProfileMode::Disabled);
        assert_eq!(parse_profile_mode(Some("0")), ProfileMode::Disabled);
        assert_eq!(parse_profile_mode(Some("1")), ProfileMode::Human);
        assert_eq!(parse_profile_mode(Some("human")), ProfileMode::Human);
        assert_eq!(parse_profile_mode(Some("json")), ProfileMode::Json);
        assert_eq!(parse_profile_mode(Some("anything")), ProfileMode::Disabled);
    }

    #[test]
    fn snapshot_drains_tree_counters_between_fits() {
        let _guard = PROFILE_TEST_LOCK.lock().unwrap();
        HISTOGRAM_NS.store(7, Ordering::Relaxed);
        SPLIT_FIND_NS.store(11, Ordering::Relaxed);
        PARTITION_NS.store(13, Ordering::Relaxed);
        let profile = RoundProfile::with_mode(ProfileMode::Json);
        let first = profile.snapshot(1, 1, 1);
        assert_eq!(first.tree_stage_ns["histogram_build"], 7);
        assert_eq!(first.tree_stage_ns["split_find"], 11);
        assert_eq!(first.tree_stage_ns["partition"], 13);
        let second = profile.snapshot(1, 1, 1);
        assert_eq!(second.tree_stage_ns["histogram_build"], 0);
        assert_eq!(second.tree_stage_ns["split_find"], 0);
        assert_eq!(second.tree_stage_ns["partition"], 0);
    }
}
