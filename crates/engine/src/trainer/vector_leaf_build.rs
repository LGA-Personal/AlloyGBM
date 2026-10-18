//! One tree per round with K-vector leaves, for multiclass softmax.
//!
//! The per-class path grows K trees per round, each scored on one class's
//! gradients. Here one tree is grown per round: a split is scored on the sum
//! of the per-class Newton gains, and each leaf holds one Newton step per
//! class. The tree is returned as K per-class trees that share its structure
//! (same node ids, features and thresholds; delta-encoded scalar leaves), so
//! the rest of the multiclass trainer, the artifact, the predictor and SHAP
//! treat it exactly like a per-class round.
//!
//! Split scoring and leaf solving read separate gradient columns
//! ([`VectorLeafGradients`]). Today both are the K class gradients; a
//! gradient sketch (SketchBoost) can later pass fewer, derived columns for
//! scoring while the leaves still solve on all K classes.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use alloygbm_core::{
    BinColumn, BinnedMatrix, GradientPair, LeafValue, NodeStats, SplitCandidate, TrainParams,
    TreeGrowth,
};
use rayon::prelude::*;

use crate::error::{EngineError, EngineResult};
use crate::tree_node::{encode_tree_node_id, left_child_node_id, right_child_node_id};
use crate::{IterationControls, SplitNoise, SplitSelectionOptions, TrainedStump};

use super::LEAF_EPSILON;

/// Below this many `rows x features x columns` cells a node's histogram is
/// built on one thread; above it, features are built in parallel.
const MIN_PARALLEL_HISTOGRAM_CELLS: usize = 1 << 16;

/// Per-feature histogram sizes. Feature `f` has data bins `0..data_bins[f]`
/// and one missing slot at index `data_bins[f]`.
#[derive(Debug, Clone)]
pub(crate) struct VectorLeafBinLayout {
    data_bins: Vec<usize>,
    /// Mean of `data_bins`, at least 1.
    mean_data_bins: usize,
    missing_bin: u16,
}

impl VectorLeafBinLayout {
    pub(crate) fn new(binned_matrix: &BinnedMatrix) -> Self {
        let missing_bin = binned_matrix.missing_bin();
        let data_bins = match binned_matrix.compact_histogram_layout() {
            Some(layout) => layout
                .data_bins
                .iter()
                .map(|&bins| usize::from(bins))
                .collect(),
            None => {
                let uniform =
                    (usize::from(binned_matrix.max_bin) + 1).min(usize::from(missing_bin));
                vec![uniform; binned_matrix.feature_count]
            }
        };
        let mean_data_bins = (data_bins.iter().sum::<usize>() / data_bins.len().max(1)).max(1);
        Self {
            data_bins,
            mean_data_bins,
            missing_bin,
        }
    }

    /// Histogram slot for a stored bin. Data bins past the layout (only
    /// possible for decoded bundled features) share the last data slot; no
    /// threshold reaches it, so their routing is unchanged.
    #[inline]
    fn slot(&self, feature: usize, bin: u16) -> usize {
        let data_bins = self.data_bins[feature];
        if bin == self.missing_bin {
            data_bins
        } else {
            usize::from(bin).min(data_bins.saturating_sub(1))
        }
    }
}

/// Gradient columns for one vector-leaf round.
pub(crate) struct VectorLeafGradients<'a> {
    /// One buffer per class; leaf values solve on these.
    pub(crate) leaf: &'a [Vec<GradientPair>],
    /// Columns that score splits. `None` scores on `leaf`. A gradient sketch
    /// passes its own (usually fewer) columns here and leaves `leaf` intact.
    pub(crate) split: Option<&'a [Vec<GradientPair>]>,
}

pub(crate) struct VectorLeafTreeInputs<'a> {
    pub(crate) binned_matrix: &'a BinnedMatrix,
    pub(crate) layout: &'a VectorLeafBinLayout,
    pub(crate) gradients: VectorLeafGradients<'a>,
    pub(crate) root_rows: Vec<u32>,
    /// Features this round may split on, ascending.
    pub(crate) features: Vec<usize>,
    pub(crate) round_index: usize,
    pub(crate) split_options: SplitSelectionOptions,
    pub(crate) params: &'a TrainParams,
    pub(crate) controls: &'a IterationControls,
    pub(crate) feature_weights: &'a [f32],
    /// Set when leaves solve with the full softmax Hessian.
    pub(crate) full_hessian: Option<FullHessianInputs<'a>>,
}

/// What a full-Hessian leaf solve needs beyond the gradients.
pub(crate) struct FullHessianInputs<'a> {
    /// Softmax probabilities, row-major: `probabilities[row * K + k]`.
    pub(crate) probabilities: Vec<f32>,
    /// Class index per row.
    pub(crate) targets: &'a [f32],
}

/// Row-major `(grad, hess)` pairs for `cols` columns: entry
/// `(row * cols + c) * 2` is column `c`'s gradient for `row`.
struct PackedColumns {
    cols: usize,
    values: Vec<f32>,
}

impl PackedColumns {
    fn pack(columns: &[Vec<GradientPair>], row_count: usize) -> EngineResult<Self> {
        let cols = columns.len();
        if cols == 0 {
            return Err(EngineError::ContractViolation(
                "vector-leaf trees need at least one gradient column".to_string(),
            ));
        }
        if let Some(bad) = columns.iter().position(|column| column.len() != row_count) {
            return Err(EngineError::ContractViolation(format!(
                "gradient column {bad} has {} rows, expected {row_count}",
                columns[bad].len()
            )));
        }
        let mut values = vec![0.0_f32; row_count * cols * 2];
        for (c, column) in columns.iter().enumerate() {
            for (row, pair) in column.iter().enumerate() {
                let at = (row * cols + c) * 2;
                values[at] = pair.grad;
                values[at + 1] = pair.hess;
            }
        }
        Ok(Self { cols, values })
    }

    #[inline]
    fn row(&self, row: u32) -> &[f32] {
        let start = row as usize * self.cols * 2;
        &self.values[start..start + self.cols * 2]
    }

    /// Per-column `(grad, hess)` sums over `rows`, interleaved.
    fn sums(&self, rows: &[u32]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.cols * 2];
        for &row in rows {
            for (acc, &value) in out.iter_mut().zip(self.row(row)) {
                *acc += f64::from(value);
            }
        }
        out
    }
}

/// One feature's histogram. Entry `i` holds `(grad, hess)` per column at
/// `sums[i * cols * 2..]` and its row count at `counts[i]`; the last entry is
/// the missing slot.
#[derive(Clone)]
struct FeatureHistogram {
    /// `f32`, like the per-class histograms; split scans widen to `f64`.
    sums: Vec<f32>,
    counts: Vec<u32>,
    /// `None`: dense, entry `i` is data slot `i` (`data_bins + 1` entries).
    /// `Some`: sparse, entry `i` is data slot `slots[i]`, ascending, listing
    /// only non-empty slots. Small nodes use it, since a dense histogram of
    /// a node with fewer rows than bins is mostly zeros.
    slots: Option<Vec<u32>>,
}

impl FeatureHistogram {
    fn entries(&self) -> usize {
        self.counts.len()
    }
}

/// One node's histograms over the round's features (same order as
/// `VectorLeafTreeInputs::features`).
#[derive(Clone)]
struct NodeHistogram {
    features: Vec<FeatureHistogram>,
}

impl NodeHistogram {
    fn is_dense(&self) -> bool {
        self.features.iter().all(|feature| feature.slots.is_none())
    }

    /// Parent minus child; both must be dense.
    fn subtract_in_place(&mut self, child: &NodeHistogram) {
        debug_assert!(self.is_dense() && child.is_dense());
        for (parent, child) in self.features.iter_mut().zip(&child.features) {
            for (p, c) in parent.sums.iter_mut().zip(&child.sums) {
                *p -= c;
            }
            for (p, c) in parent.counts.iter_mut().zip(&child.counts) {
                *p -= c;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitChoice {
    feature: usize,
    threshold_bin: usize,
    default_left: bool,
    gain: f64,
}

/// A split whose children and leaves have been worked out but not yet
/// admitted to the tree.
struct NodeProposal {
    choice: SplitChoice,
    left_rows: Vec<u32>,
    right_rows: Vec<u32>,
    /// Per class `(grad_sum, hess_sum)` of each child, on the leaf columns.
    left_sums: Vec<f64>,
    right_sums: Vec<f64>,
    left_values: Vec<f64>,
    right_values: Vec<f64>,
}

struct Builder<'a> {
    inputs: &'a VectorLeafTreeInputs<'a>,
    leaf_columns: PackedColumns,
    /// Split-scoring columns; `None` means the leaf columns score splits.
    split_columns: Option<PackedColumns>,
    noise: Option<SplitNoise>,
    class_count: usize,
    /// Whether small nodes may use sparse histograms (tests turn it off to
    /// check both layouts grow the same tree).
    sparse_histograms: bool,
}

impl<'a> Builder<'a> {
    fn scoring(&self) -> &PackedColumns {
        self.split_columns.as_ref().unwrap_or(&self.leaf_columns)
    }

    /// Whether a node with `rows` rows gets sparse histograms: sorting its
    /// rows by bin costs less than zeroing, subtracting and scanning every
    /// bin of every column.
    fn prefers_sparse(&self, rows: usize) -> bool {
        if !self.sparse_histograms {
            return false;
        }
        let log_rows = (usize::BITS - rows.leading_zeros()).max(1) as usize;
        rows.saturating_mul(log_rows) < self.inputs.layout.mean_data_bins * self.scoring().cols
    }

    fn build_histogram(&self, rows: &[u32]) -> NodeHistogram {
        if self.prefers_sparse(rows.len()) {
            self.build_histogram_with(rows, |feature| self.build_sparse_feature(rows, feature))
        } else {
            self.build_histogram_with(rows, |feature| self.build_dense_feature(rows, feature))
        }
    }

    fn build_histogram_with(
        &self,
        rows: &[u32],
        build_one: impl Fn(usize) -> FeatureHistogram + Sync,
    ) -> NodeHistogram {
        let features = &self.inputs.features;
        let cells = rows
            .len()
            .saturating_mul(features.len())
            .saturating_mul(self.scoring().cols);
        let features = if cells >= MIN_PARALLEL_HISTOGRAM_CELLS && features.len() > 1 {
            features
                .par_iter()
                .map(|&feature| build_one(feature))
                .collect()
        } else {
            features.iter().map(|&feature| build_one(feature)).collect()
        };
        NodeHistogram { features }
    }

    /// Calls `visit(slot, row)` for every row, reading the feature's column
    /// directly when it is stored unbundled.
    #[inline]
    fn for_each_slot(&self, rows: &[u32], feature: usize, mut visit: impl FnMut(usize, u32)) {
        let binned = self.inputs.binned_matrix;
        let layout = self.inputs.layout;
        match binned.col_bins(feature) {
            Some(BinColumn::U8(column)) => {
                for &row in rows {
                    visit(layout.slot(feature, u16::from(column[row as usize])), row);
                }
            }
            Some(BinColumn::U16(column)) => {
                for &row in rows {
                    visit(layout.slot(feature, column[row as usize]), row);
                }
            }
            None => {
                for &row in rows {
                    visit(
                        layout.slot(feature, binned.bin_at(row as usize, feature)),
                        row,
                    );
                }
            }
        }
    }

    fn build_dense_feature(&self, rows: &[u32], feature: usize) -> FeatureHistogram {
        let packed = self.scoring();
        let width = packed.cols * 2;
        let entries = self.inputs.layout.data_bins[feature] + 1;
        let mut sums = vec![0.0_f32; entries * width];
        let mut counts = vec![0_u32; entries];
        self.for_each_slot(rows, feature, |slot, row| {
            counts[slot] += 1;
            let base = slot * width;
            for (acc, &value) in sums[base..base + width].iter_mut().zip(packed.row(row)) {
                *acc += value;
            }
        });
        FeatureHistogram {
            sums,
            counts,
            slots: None,
        }
    }

    fn build_sparse_feature(&self, rows: &[u32], feature: usize) -> FeatureHistogram {
        let packed = self.scoring();
        let width = packed.cols * 2;
        let missing_slot = self.inputs.layout.data_bins[feature];
        let mut keyed = Vec::with_capacity(rows.len());
        self.for_each_slot(rows, feature, |slot, row| keyed.push((slot as u32, row)));
        // Rows stay in node order within a slot, so sums add in the same
        // order on every run.
        keyed.sort_by_key(|&(slot, _)| slot);
        let mut slots = Vec::new();
        let mut sums = Vec::new();
        let mut counts = Vec::new();
        let mut missing_sums = vec![0.0_f32; width];
        let mut missing_count = 0_u32;
        for &(slot, row) in &keyed {
            let acc = if slot as usize == missing_slot {
                missing_count += 1;
                &mut missing_sums[..]
            } else {
                if slots.last() != Some(&slot) {
                    slots.push(slot);
                    counts.push(0);
                    sums.resize(sums.len() + width, 0.0);
                }
                *counts.last_mut().expect("entry pushed above") += 1;
                let base = sums.len() - width;
                &mut sums[base..]
            };
            for (acc, &value) in acc.iter_mut().zip(packed.row(row)) {
                *acc += value;
            }
        }
        sums.extend_from_slice(&missing_sums);
        counts.push(missing_count);
        FeatureHistogram {
            sums,
            counts,
            slots: Some(slots),
        }
    }

    /// `effective(g)^2 / (h + lambda)`, the per-column Newton gain term.
    #[inline]
    fn gain_term(&self, grad: f64, hess: f64) -> f64 {
        let options = &self.inputs.split_options;
        let effective = l1_threshold(grad, f64::from(options.l1_alpha));
        effective * effective / (hess + f64::from(options.l2_lambda) + f64::from(LEAF_EPSILON))
    }

    /// Unscaled Newton leaf magnitude, as the per-class split scan's
    /// `min_leaf_magnitude` filter measures it.
    #[inline]
    fn leaf_magnitude(&self, grad: f64, hess: f64) -> f64 {
        let options = &self.inputs.split_options;
        l1_threshold(grad, f64::from(options.l1_alpha)).abs()
            / (hess + f64::from(options.l2_lambda) + f64::from(LEAF_EPSILON))
    }

    fn best_split(
        &self,
        hist: &NodeHistogram,
        node_rows: &[u32],
        node_id: u32,
    ) -> Option<SplitChoice> {
        let packed = self.scoring();
        let cols = packed.cols;
        let width = cols * 2;
        let totals = packed.sums(node_rows);
        let total_count = node_rows.len() as u64;
        let options = &self.inputs.split_options;
        let min_rows = options.min_rows_per_leaf.max(1) as u64;
        let min_child_hessian = f64::from(options.min_child_hessian);
        let min_leaf_magnitude = f64::from(options.min_leaf_magnitude);
        let parent_term: f64 = (0..cols)
            .map(|c| self.gain_term(totals[2 * c], totals[2 * c + 1]))
            .sum();

        let mut left = vec![0.0_f64; width];
        let mut best: Option<(f64, SplitChoice)> = None;
        for (position, &feature) in self.inputs.features.iter().enumerate() {
            let weight = self
                .inputs
                .feature_weights
                .get(feature)
                .copied()
                .unwrap_or(1.0);
            if weight <= 0.0 {
                continue;
            }
            let feature_hist = &hist.features[position];
            let data_bins = self.inputs.layout.data_bins[feature];
            if data_bins < 2 {
                continue;
            }
            let missing_entry = feature_hist.entries() - 1;
            let missing_base = missing_entry * width;
            let missing = &feature_hist.sums[missing_base..missing_base + width];
            let missing_count = u64::from(feature_hist.counts[missing_entry]);
            let directions: &[bool] = if missing_count > 0 {
                &[true, false]
            } else {
                &[true]
            };
            let mut feature_best: Option<SplitChoice> = None;
            for &default_left in directions {
                left.iter_mut().for_each(|value| *value = 0.0);
                let mut left_count = 0_u64;
                for entry in 0..missing_entry {
                    // An empty bin leaves the partition unchanged, so its
                    // threshold ties the previous one and never wins; sparse
                    // histograms leave empty bins out altogether.
                    if feature_hist.counts[entry] == 0 {
                        continue;
                    }
                    let threshold_bin = feature_hist
                        .slots
                        .as_ref()
                        .map_or(entry, |slots| slots[entry] as usize);
                    // A threshold on the last data bin sends every data row
                    // left; the per-class scan does not offer it either.
                    if threshold_bin + 1 >= data_bins {
                        break;
                    }
                    let base = entry * width;
                    for (acc, &value) in left.iter_mut().zip(&feature_hist.sums[base..base + width])
                    {
                        *acc += f64::from(value);
                    }
                    left_count += u64::from(feature_hist.counts[entry]);
                    let (l_count, r_count) = if default_left {
                        (
                            left_count + missing_count,
                            total_count - left_count - missing_count,
                        )
                    } else {
                        (left_count, total_count - left_count)
                    };
                    if l_count < min_rows || r_count < min_rows {
                        continue;
                    }
                    let mut gain = -parent_term;
                    let mut l_hess = 0.0;
                    let mut r_hess = 0.0;
                    let mut l_magnitude = 0.0_f64;
                    let mut r_magnitude = 0.0_f64;
                    for c in 0..cols {
                        let (mut lg, mut lh) = (left[2 * c], left[2 * c + 1]);
                        if default_left {
                            lg += f64::from(missing[2 * c]);
                            lh += f64::from(missing[2 * c + 1]);
                        }
                        let rg = totals[2 * c] - lg;
                        // Subtracted f32 histograms can leave a hessian a
                        // rounding error below zero.
                        let lh = lh.max(0.0);
                        let rh = (totals[2 * c + 1] - lh).max(0.0);
                        gain += self.gain_term(lg, lh) + self.gain_term(rg, rh);
                        l_hess += lh;
                        r_hess += rh;
                        if min_leaf_magnitude > 0.0 {
                            l_magnitude = l_magnitude.max(self.leaf_magnitude(lg, lh));
                            r_magnitude = r_magnitude.max(self.leaf_magnitude(rg, rh));
                        }
                    }
                    // The per-class scan checks each class tree's own hessian;
                    // a shared tree checks the mean over its columns.
                    if l_hess / (cols as f64) < min_child_hessian
                        || r_hess / (cols as f64) < min_child_hessian
                    {
                        continue;
                    }
                    if min_leaf_magnitude > 0.0
                        && l_magnitude < min_leaf_magnitude
                        && r_magnitude < min_leaf_magnitude
                    {
                        continue;
                    }
                    if !gain.is_finite() || gain <= 0.0 {
                        continue;
                    }
                    if feature_best.is_none_or(|current| gain > current.gain) {
                        feature_best = Some(SplitChoice {
                            feature,
                            threshold_bin,
                            default_left,
                            gain,
                        });
                    }
                }
            }
            let Some(candidate) = feature_best else {
                continue;
            };
            let weighted = candidate.gain * f64::from(weight);
            let replace = match (self.noise, best.as_ref()) {
                (_, None) => true,
                (Some(noise), Some((current, _))) => {
                    weighted * f64::from(noise.gain_multiplier(node_id, feature as u32)) > *current
                }
                (None, Some((current, _))) => gain_materially_exceeds(weighted, *current),
            };
            if replace {
                let ranking = match self.noise {
                    Some(noise) => {
                        weighted * f64::from(noise.gain_multiplier(node_id, feature as u32))
                    }
                    None => weighted,
                };
                best = Some((ranking, candidate));
            }
        }
        best.map(|(_, choice)| choice)
    }

    /// Partition the node, solve both leaves and apply the per-class leaf
    /// checks. `None` when the split leaves a child too small or every class
    /// moves by less than `min_abs_leaf_value` on both sides.
    fn propose(
        &self,
        choice: SplitChoice,
        rows: &[u32],
        parent_values: &[f64],
    ) -> Option<NodeProposal> {
        let binned = self.inputs.binned_matrix;
        let layout = self.inputs.layout;
        let missing_slot = layout.data_bins[choice.feature];
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for &row in rows {
            let slot = layout.slot(choice.feature, binned.bin_at(row as usize, choice.feature));
            let goes_left = if slot == missing_slot {
                choice.default_left
            } else {
                slot <= choice.threshold_bin
            };
            if goes_left {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        let min_rows = self.inputs.controls.min_rows_per_leaf.max(1);
        if left_rows.len() < min_rows || right_rows.len() < min_rows {
            return None;
        }
        let left_sums = self.leaf_columns.sums(&left_rows);
        let right_sums = self.leaf_columns.sums(&right_rows);
        let left_values = self.solve_leaf(&left_rows, &left_sums);
        let right_values = self.solve_leaf(&right_rows, &right_sums);
        let min_abs = f64::from(self.inputs.controls.min_abs_leaf_value);
        let moves = (0..self.class_count).any(|k| {
            (left_values[k] - parent_values[k]).abs() >= min_abs
                || (right_values[k] - parent_values[k]).abs() >= min_abs
        });
        if !moves {
            return None;
        }
        Some(NodeProposal {
            choice,
            left_rows,
            right_rows,
            left_sums,
            right_sums,
            left_values,
            right_values,
        })
    }

    fn solve_leaf(&self, rows: &[u32], sums: &[f64]) -> Vec<f64> {
        match &self.inputs.full_hessian {
            Some(full) => self
                .full_hessian_leaf_values(full, rows, sums)
                .unwrap_or_else(|| self.leaf_values(sums)),
            None => self.leaf_values(sums),
        }
    }

    /// Newton step with the full softmax Hessian, `-(H + lambda I)^-1 g`,
    /// where `H` sums `A_i (diag(p_i) - p_i p_i^T)` over the leaf's rows and
    /// `A_i` is the row's gradient scale (sample weight times any GOSS
    /// amplification), read off as `sum_k |g_ik| / sum_k |p_ik - y_ik|`.
    /// The step is centred: softmax ignores a shift shared by every class,
    /// and `H` is singular along that shift. `None` when the solve fails,
    /// and the caller falls back to the diagonal step.
    fn full_hessian_leaf_values(
        &self,
        full: &FullHessianInputs<'_>,
        rows: &[u32],
        sums: &[f64],
    ) -> Option<Vec<f64>> {
        let k = self.class_count;
        let mut hessian = vec![0.0_f64; k * k];
        let mut scaled = vec![0.0_f64; k];
        for &row in rows {
            let r = row as usize;
            let probabilities = &full.probabilities[r * k..(r + 1) * k];
            let target = full.targets[r] as usize;
            let pairs = self.leaf_columns.row(row);
            let mut gradient_mass = 0.0_f64;
            let mut residual_mass = 0.0_f64;
            for (c, &p) in probabilities.iter().enumerate() {
                gradient_mass += f64::from(pairs[2 * c]).abs();
                let indicator = if c == target { 1.0 } else { 0.0 };
                residual_mass += (f64::from(p) - indicator).abs();
            }
            if gradient_mass == 0.0 || residual_mass <= 1e-12 {
                continue;
            }
            let row_scale = gradient_mass / residual_mass;
            for (value, &p) in scaled.iter_mut().zip(probabilities) {
                *value = row_scale * f64::from(p);
            }
            for j in 0..k {
                hessian[j * k + j] += scaled[j];
                let p_j = f64::from(probabilities[j]);
                for c in 0..=j {
                    hessian[j * k + c] -= scaled[c] * p_j;
                }
            }
        }
        let options = &self.inputs.split_options;
        let ridge = f64::from(options.l2_lambda) + f64::from(LEAF_EPSILON);
        for j in 0..k {
            hessian[j * k + j] += ridge;
        }
        let mut step: Vec<f64> = (0..k)
            .map(|c| l1_threshold(sums[2 * c], f64::from(options.l1_alpha)))
            .collect();
        cholesky_solve_in_place(&mut hessian, &mut step, k)?;
        let mean = step.iter().sum::<f64>() / k as f64;
        let scale = f64::from(self.inputs.params.learning_rate);
        let cap = f64::from(self.inputs.controls.max_abs_leaf_value);
        let values: Vec<f64> = step
            .iter()
            .map(|&value| (-scale * (value - mean)).clamp(-cap, cap))
            .collect();
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(values)
    }

    /// Absolute leaf outputs, one Newton step per class, scaled by the
    /// learning rate and clamped like the per-class builder's.
    fn leaf_values(&self, sums: &[f64]) -> Vec<f64> {
        let options = &self.inputs.split_options;
        let scale = f64::from(self.inputs.params.learning_rate);
        let cap = f64::from(self.inputs.controls.max_abs_leaf_value);
        (0..self.class_count)
            .map(|k| {
                let effective = l1_threshold(sums[2 * k], f64::from(options.l1_alpha));
                let value = -scale * effective
                    / (sums[2 * k + 1] + f64::from(options.l2_lambda) + f64::from(LEAF_EPSILON));
                value.clamp(-cap, cap)
            })
            .collect()
    }

    /// Emit one stump per class for an admitted split.
    fn emit(
        &self,
        local_node_id: u32,
        proposal: &NodeProposal,
        parent_values: &[f64],
        class_stumps: &mut [Vec<TrainedStump>],
    ) -> EngineResult<()> {
        let node_id = encode_tree_node_id(self.inputs.round_index, local_node_id)?;
        let left_count = proposal.left_rows.len() as u32;
        let right_count = proposal.right_rows.len() as u32;
        for (k, stumps) in class_stumps.iter_mut().enumerate() {
            let (lg, lh) = (proposal.left_sums[2 * k], proposal.left_sums[2 * k + 1]);
            let (rg, rh) = (proposal.right_sums[2 * k], proposal.right_sums[2 * k + 1]);
            // This class's share of the shared split's gain, on the leaf
            // columns, so gain importance stays per class.
            let class_gain =
                self.gain_term(lg, lh) + self.gain_term(rg, rh) - self.gain_term(lg + rg, lh + rh);
            stumps.push(TrainedStump {
                split: SplitCandidate {
                    node_id,
                    feature_index: proposal.choice.feature as u32,
                    threshold_bin: proposal.choice.threshold_bin as u16,
                    gain: class_gain as f32,
                    default_left: proposal.choice.default_left,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: lg as f32,
                        hess_sum: lh as f32,
                        grad_sq_sum: 0.0,
                        row_count: left_count,
                    },
                    right_stats: NodeStats {
                        grad_sum: rg as f32,
                        hess_sum: rh as f32,
                        grad_sq_sum: 0.0,
                        row_count: right_count,
                    },
                },
                left_leaf_value: LeafValue::Scalar(
                    (proposal.left_values[k] - parent_values[k]) as f32,
                ),
                right_leaf_value: LeafValue::Scalar(
                    (proposal.right_values[k] - parent_values[k]) as f32,
                ),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            });
        }
        Ok(())
    }

    /// Histograms for both children. A child small enough for sparse
    /// histograms is built from its rows. Otherwise the smaller child is
    /// built and the larger is the parent minus it, when the parent is
    /// dense.
    fn child_histograms(
        &self,
        parent: NodeHistogram,
        left_rows: &[u32],
        right_rows: &[u32],
    ) -> (NodeHistogram, NodeHistogram) {
        let left_is_smaller = left_rows.len() <= right_rows.len();
        let (smaller_rows, larger_rows) = if left_is_smaller {
            (left_rows, right_rows)
        } else {
            (right_rows, left_rows)
        };
        let (smaller, larger) = if self.prefers_sparse(larger_rows.len()) || !parent.is_dense() {
            (
                self.build_histogram(smaller_rows),
                self.build_histogram(larger_rows),
            )
        } else {
            let smaller = self.build_histogram_with(smaller_rows, |feature| {
                self.build_dense_feature(smaller_rows, feature)
            });
            let mut larger = parent;
            larger.subtract_in_place(&smaller);
            (smaller, larger)
        };
        if left_is_smaller {
            (smaller, larger)
        } else {
            (larger, smaller)
        }
    }
}

struct LevelNode {
    local_node_id: u32,
    rows: Vec<u32>,
    histogram: Option<NodeHistogram>,
    values: Vec<f64>,
}

/// A level node with its proposal and, when the tree goes deeper, its
/// children's histograms.
type LevelOutcome = (
    LevelNode,
    Option<(NodeProposal, Option<(NodeHistogram, NodeHistogram)>)>,
);

struct PendingLeaf {
    choice: SplitChoice,
    local_node_id: u32,
    rows: Vec<u32>,
    histogram: Option<NodeHistogram>,
    values: Vec<f64>,
    depth: usize,
}

impl PartialEq for PendingLeaf {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PendingLeaf {}

impl PartialOrd for PendingLeaf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingLeaf {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest gain first; on ties the lower node id, so the order never
        // depends on heap internals.
        self.choice
            .gain
            .partial_cmp(&other.choice.gain)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.local_node_id.cmp(&self.local_node_id))
    }
}

/// Grow one vector-leaf tree.
///
/// Returns one stump list per class (empty lists when the root does not
/// split) and adds each final leaf's per-class output to
/// `candidate_predictions` for the rows it holds, as the per-class builders
/// do for their selected rows.
pub(crate) fn build_vector_leaf_tree(
    inputs: &VectorLeafTreeInputs<'_>,
    candidate_predictions: &mut [Vec<f32>],
) -> EngineResult<Vec<Vec<TrainedStump>>> {
    build_vector_leaf_tree_with(inputs, candidate_predictions, true)
}

fn build_vector_leaf_tree_with(
    inputs: &VectorLeafTreeInputs<'_>,
    candidate_predictions: &mut [Vec<f32>],
    sparse_histograms: bool,
) -> EngineResult<Vec<Vec<TrainedStump>>> {
    let row_count = inputs.binned_matrix.row_count;
    let class_count = inputs.gradients.leaf.len();
    if candidate_predictions.len() != class_count {
        return Err(EngineError::ContractViolation(format!(
            "vector-leaf tree has {class_count} classes but {} prediction buffers",
            candidate_predictions.len()
        )));
    }
    let builder = Builder {
        inputs,
        leaf_columns: PackedColumns::pack(inputs.gradients.leaf, row_count)?,
        split_columns: inputs
            .gradients
            .split
            .map(|columns| PackedColumns::pack(columns, row_count))
            .transpose()?,
        noise: SplitNoise::new(
            inputs.controls.split_noise_strength,
            inputs.controls.split_noise_seed,
        ),
        class_count,
        sparse_histograms,
    };
    let mut class_stumps: Vec<Vec<TrainedStump>> = vec![Vec::new(); class_count];
    let mut final_leaves: Vec<(Vec<u32>, Vec<f64>)> = Vec::new();
    if inputs.params.tree_growth == TreeGrowth::Leaf {
        grow_leaf_wise(&builder, &mut class_stumps, &mut final_leaves)?;
    } else {
        grow_level_wise(&builder, &mut class_stumps, &mut final_leaves)?;
    }
    if class_stumps.iter().all(Vec::is_empty) {
        return Ok(class_stumps);
    }
    for (rows, values) in &final_leaves {
        for (predictions, &value) in candidate_predictions.iter_mut().zip(values) {
            let value = value as f32;
            for &row in rows {
                predictions[row as usize] += value;
            }
        }
    }
    Ok(class_stumps)
}

fn can_split(builder: &Builder<'_>, rows: &[u32]) -> bool {
    rows.len() >= builder.inputs.controls.min_rows_per_leaf.max(1) * 2
}

fn grow_level_wise(
    builder: &Builder<'_>,
    class_stumps: &mut [Vec<TrainedStump>],
    final_leaves: &mut Vec<(Vec<u32>, Vec<f64>)>,
) -> EngineResult<()> {
    let inputs = builder.inputs;
    let max_depth = inputs.params.max_depth as usize;
    let min_split_gain = f64::from(inputs.controls.min_split_gain);
    let root_rows = inputs.root_rows.clone();
    let root_histogram = builder.build_histogram(&root_rows);
    let mut level = vec![LevelNode {
        local_node_id: 0,
        rows: root_rows,
        histogram: Some(root_histogram),
        values: vec![0.0; builder.class_count],
    }];
    let mut splits = 0_usize;
    for depth in 0..max_depth {
        if level.is_empty() {
            break;
        }
        let needs_child_histograms = depth + 1 < max_depth;
        // Proposals, including the children's histograms, are worked out
        // for every node of the level in parallel; admission below runs in
        // node order so the tree never depends on scheduling.
        let propose = |mut node: LevelNode| -> EngineResult<LevelOutcome> {
            if !can_split(builder, &node.rows) {
                return Ok((node, None));
            }
            let node_id = encode_tree_node_id(inputs.round_index, node.local_node_id)?;
            let histogram = node
                .histogram
                .take()
                .ok_or_else(|| EngineError::ContractViolation("missing node histogram".into()))?;
            let Some(proposal) = builder
                .best_split(&histogram, &node.rows, node_id)
                .filter(|choice| choice.gain > min_split_gain)
                .and_then(|choice| builder.propose(choice, &node.rows, &node.values))
            else {
                return Ok((node, None));
            };
            let children = needs_child_histograms.then(|| {
                builder.child_histograms(histogram, &proposal.left_rows, &proposal.right_rows)
            });
            Ok((node, Some((proposal, children))))
        };
        let outcomes: Vec<EngineResult<LevelOutcome>> = if level.len() > 1 {
            level.into_par_iter().map(propose).collect()
        } else {
            level.into_iter().map(propose).collect()
        };
        let mut next_level = Vec::new();
        for outcome in outcomes {
            let (node, proposal) = outcome?;
            let Some((proposal, children)) = proposal else {
                final_leaves.push((node.rows, node.values));
                continue;
            };
            if let Some(max_leaves) = inputs.controls.max_leaves
                && splits + 2 > max_leaves
            {
                final_leaves.push((node.rows, node.values));
                continue;
            }
            builder.emit(node.local_node_id, &proposal, &node.values, class_stumps)?;
            splits += 1;
            let (left_histogram, right_histogram) = match children {
                Some((left, right)) => (Some(left), Some(right)),
                None => (None, None),
            };
            next_level.push(LevelNode {
                local_node_id: left_child_node_id(node.local_node_id)?,
                rows: proposal.left_rows,
                histogram: left_histogram,
                values: proposal.left_values,
            });
            next_level.push(LevelNode {
                local_node_id: right_child_node_id(node.local_node_id)?,
                rows: proposal.right_rows,
                histogram: right_histogram,
                values: proposal.right_values,
            });
        }
        level = next_level;
    }
    final_leaves.extend(level.into_iter().map(|node| (node.rows, node.values)));
    Ok(())
}

fn grow_leaf_wise(
    builder: &Builder<'_>,
    class_stumps: &mut [Vec<TrainedStump>],
    final_leaves: &mut Vec<(Vec<u32>, Vec<f64>)>,
) -> EngineResult<()> {
    let inputs = builder.inputs;
    let max_depth = inputs.params.max_depth as usize;
    let max_leaves = inputs.controls.max_leaves.unwrap_or(usize::MAX);
    let min_split_gain = f64::from(inputs.controls.min_split_gain);
    let class_count = builder.class_count;

    // A node that can be split goes on the heap with its best split; any
    // other node is a final leaf straight away.
    let consider = |local_node_id: u32,
                    rows: Vec<u32>,
                    histogram: Option<NodeHistogram>,
                    values: Vec<f64>,
                    depth: usize,
                    heap: &mut BinaryHeap<PendingLeaf>,
                    final_leaves: &mut Vec<(Vec<u32>, Vec<f64>)>|
     -> EngineResult<()> {
        if depth < max_depth
            && can_split(builder, &rows)
            && let Some(histogram) = histogram
        {
            let node_id = encode_tree_node_id(inputs.round_index, local_node_id)?;
            if let Some(choice) = builder
                .best_split(&histogram, &rows, node_id)
                .filter(|choice| choice.gain > min_split_gain)
            {
                heap.push(PendingLeaf {
                    choice,
                    local_node_id,
                    rows,
                    histogram: Some(histogram),
                    values,
                    depth,
                });
                return Ok(());
            }
        }
        final_leaves.push((rows, values));
        Ok(())
    };

    let mut heap = BinaryHeap::new();
    let root_rows = inputs.root_rows.clone();
    let root_histogram = builder.build_histogram(&root_rows);
    consider(
        0,
        root_rows,
        Some(root_histogram),
        vec![0.0; class_count],
        0,
        &mut heap,
        final_leaves,
    )?;
    let mut leaves = 1_usize;
    while let Some(pending) = heap.pop() {
        if leaves + 1 > max_leaves {
            final_leaves.push((pending.rows, pending.values));
            break;
        }
        let Some(proposal) = builder.propose(pending.choice, &pending.rows, &pending.values) else {
            final_leaves.push((pending.rows, pending.values));
            continue;
        };
        builder.emit(
            pending.local_node_id,
            &proposal,
            &pending.values,
            class_stumps,
        )?;
        leaves += 1;
        let child_depth = pending.depth + 1;
        let (left_histogram, right_histogram) = match pending.histogram {
            Some(parent) if child_depth < max_depth => {
                let (left, right) =
                    builder.child_histograms(parent, &proposal.left_rows, &proposal.right_rows);
                (Some(left), Some(right))
            }
            _ => (None, None),
        };
        consider(
            left_child_node_id(pending.local_node_id)?,
            proposal.left_rows,
            left_histogram,
            proposal.left_values,
            child_depth,
            &mut heap,
            final_leaves,
        )?;
        consider(
            right_child_node_id(pending.local_node_id)?,
            proposal.right_rows,
            right_histogram,
            proposal.right_values,
            child_depth,
            &mut heap,
            final_leaves,
        )?;
    }
    final_leaves.extend(
        heap.into_iter()
            .map(|pending| (pending.rows, pending.values)),
    );
    Ok(())
}

/// Solves `A x = b` for symmetric positive-definite `A` (`n x n`, row-major;
/// only the lower triangle is read), leaving `x` in `rhs`. `A` is
/// overwritten with its Cholesky factor. `None` when `A` is not
/// numerically positive definite.
fn cholesky_solve_in_place(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> Option<()> {
    for j in 0..n {
        let mut diagonal = matrix[j * n + j];
        for c in 0..j {
            diagonal -= matrix[j * n + c] * matrix[j * n + c];
        }
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return None;
        }
        let pivot = diagonal.sqrt();
        matrix[j * n + j] = pivot;
        for i in j + 1..n {
            let mut value = matrix[i * n + j];
            for c in 0..j {
                value -= matrix[i * n + c] * matrix[j * n + c];
            }
            matrix[i * n + j] = value / pivot;
        }
    }
    // Forward substitution: L y = b.
    for i in 0..n {
        let mut value = rhs[i];
        for c in 0..i {
            value -= matrix[i * n + c] * rhs[c];
        }
        rhs[i] = value / matrix[i * n + i];
    }
    // Back substitution: L^T x = y.
    for i in (0..n).rev() {
        let mut value = rhs[i];
        for c in i + 1..n {
            value -= matrix[c * n + i] * rhs[c];
        }
        rhs[i] = value / matrix[i * n + i];
    }
    Some(())
}

/// Softmax probabilities of every row, row-major (`[row * K + k]`), computed
/// the way the multiclass objective computes them.
pub(crate) fn softmax_rows(class_predictions: &[Vec<f32>]) -> Vec<f32> {
    let k = class_predictions.len();
    let n = class_predictions.first().map_or(0, Vec::len);
    let mut probabilities = vec![0.0_f32; n * k];
    probabilities
        .par_chunks_mut(k.max(1))
        .enumerate()
        .for_each(|(row, out)| {
            let mut max_logit = f32::NEG_INFINITY;
            for class_preds in class_predictions {
                max_logit = max_logit.max(class_preds[row]);
            }
            let mut sum_exp = 0.0_f32;
            for (value, class_preds) in out.iter_mut().zip(class_predictions) {
                *value = (class_preds[row] - max_logit).exp();
                sum_exp += *value;
            }
            for value in out.iter_mut() {
                *value /= sum_exp;
            }
        });
    probabilities
}

/// Replaces each class Hessian with `|g|`, i.e. `w |p_k - y_k|`. Rows with a
/// zero gradient keep their (tiny) softmax Hessian.
pub(crate) fn apply_abs_residual_hessian(gradients: &mut [GradientPair]) {
    for pair in gradients.iter_mut() {
        if pair.grad != 0.0 {
            pair.hess = pair.grad.abs();
        }
    }
}

/// Largest class count the full-Hessian leaf solve accepts: each leaf sums a
/// `K x K` matrix per row.
pub(crate) const MAX_FULL_HESSIAN_CLASSES: usize = 16;

pub(crate) fn validate_multiclass_hessian(
    params: &TrainParams,
    class_count: usize,
) -> EngineResult<()> {
    if params.multiclass_hessian != alloygbm_core::MulticlassHessian::Full {
        return Ok(());
    }
    if params.multiclass_strategy != alloygbm_core::MulticlassStrategy::VectorLeaves {
        return Err(EngineError::InvalidConfig(
            "multiclass_hessian='full' requires multiclass_strategy='vector_leaves'".to_string(),
        ));
    }
    if class_count > MAX_FULL_HESSIAN_CLASSES {
        return Err(EngineError::InvalidConfig(format!(
            "multiclass_hessian='full' supports at most {MAX_FULL_HESSIAN_CLASSES} classes, \
             got {class_count}"
        )));
    }
    if params
        .neutralization_config
        .is_some_and(|config| config.kind != alloygbm_core::NeutralizationKind::None)
    {
        return Err(EngineError::InvalidConfig(
            "multiclass_hessian='full' does not support factor neutralization".to_string(),
        ));
    }
    // The full-Hessian solve rebuilds each leaf's Hessian from the softmax
    // probabilities, so a per-row GRN term added to the row Hessians would be
    // dropped silently.
    if params.grn_strength > 0.0 && params.grn_form == alloygbm_core::GrnForm::PerRow {
        return Err(EngineError::InvalidConfig(
            "multiclass_hessian='full' does not support grn_form='per_row'".to_string(),
        ));
    }
    Ok(())
}

/// A split-scoring sketch only has a vector-leaf tree to score.
pub(crate) fn validate_multiclass_sketch(params: &TrainParams) -> EngineResult<()> {
    if params.multiclass_sketch != alloygbm_core::MulticlassSketch::None
        && params.multiclass_strategy != alloygbm_core::MulticlassStrategy::VectorLeaves
    {
        return Err(EngineError::InvalidConfig(
            "multiclass_sketch requires multiclass_strategy='vector_leaves'".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn l1_threshold(grad: f64, l1_alpha: f64) -> f64 {
    if l1_alpha <= 0.0 {
        grad
    } else if grad > l1_alpha {
        grad - l1_alpha
    } else if grad < -l1_alpha {
        grad + l1_alpha
    } else {
        0.0
    }
}

/// Same rule the per-class split selection uses to compare features when no
/// split noise is active.
#[inline]
fn gain_materially_exceeds(candidate: f64, current: f64) -> bool {
    let tolerance = 1e-6 * candidate.abs().max(current.abs()).max(1.0);
    candidate > current + tolerance
}

/// Reject options the vector-leaf builder does not implement yet, rather
/// than ignore them.
pub(crate) fn validate_vector_leaf_support(
    params: &TrainParams,
    categorical_features: &[crate::CategoricalFeatureInfo],
) -> EngineResult<()> {
    let unsupported = |what: &str| {
        Err(EngineError::InvalidConfig(format!(
            "multiclass_strategy='vector_leaves' does not support {what} yet; \
             use multiclass_strategy='per_class_trees'"
        )))
    };
    if params.morph_config.is_some() {
        return unsupported("training_mode='morph'");
    }
    if params.leaf_model == alloygbm_core::LeafModelKind::Linear {
        return unsupported("leaf_model='linear'");
    }
    if params.leaf_solver == alloygbm_core::LeafSolverKind::Dro {
        return unsupported("leaf_solver='dro'");
    }
    if params
        .neutralization_config
        .is_some_and(|config| config.kind == alloygbm_core::NeutralizationKind::SplitPenalty)
    {
        return unsupported("neutralization='split_penalty'");
    }
    if !params.interaction_constraints.is_empty() {
        return unsupported("interaction_constraints");
    }
    if !categorical_features.is_empty() {
        return unsupported("native categorical splits (max_cat_threshold > 0)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MultiClassSoftmaxObjective;

    const CLASSES: usize = 5;

    /// Few rows over many bins, so most nodes take the sparse path.
    fn wide_fixture(row_count: usize, feature_count: usize) -> (BinnedMatrix, Vec<f32>) {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut bins = Vec::with_capacity(row_count * feature_count);
        let mut targets = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let mut score = 0_u64;
            for feature in 0..feature_count {
                let bin = if (row + feature) % 13 == 0 {
                    alloygbm_core::MISSING_BIN_U8
                } else {
                    (next() % 200) as u8
                };
                if feature < 2 && bin != alloygbm_core::MISSING_BIN_U8 {
                    score += u64::from(bin);
                }
                bins.push(bin);
            }
            targets.push(((score / 80 + next() % 2) % CLASSES as u64) as f32);
        }
        let binned = BinnedMatrix::new(row_count, feature_count, 199, bins).expect("bins");
        (binned, targets)
    }

    fn gradients(targets: &[f32]) -> Vec<Vec<GradientPair>> {
        let objective = MultiClassSoftmaxObjective::new(CLASSES).expect("objective");
        let predictions: Vec<Vec<f32>> = (0..CLASSES)
            .map(|k| {
                (0..targets.len())
                    .map(|row| ((row * (k + 3)) % 7) as f32 * 0.1)
                    .collect()
            })
            .collect();
        objective
            .compute_gradients_all_classes(&predictions, targets, None)
            .expect("gradients")
    }

    fn grow(
        binned: &BinnedMatrix,
        leaf: &[Vec<GradientPair>],
        split: Option<&[Vec<GradientPair>]>,
        tree_growth: TreeGrowth,
        sparse_histograms: bool,
    ) -> (Vec<Vec<TrainedStump>>, Vec<Vec<f32>>) {
        let params = TrainParams {
            max_depth: 5,
            lambda_l2: 1.0,
            learning_rate: 0.2,
            tree_growth,
            ..TrainParams::default()
        };
        let controls = IterationControls::new(1, 0.0, 2, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls")
            .with_max_leaves((tree_growth == TreeGrowth::Leaf).then_some(12))
            .expect("max leaves");
        let layout = VectorLeafBinLayout::new(binned);
        let inputs = VectorLeafTreeInputs {
            binned_matrix: binned,
            layout: &layout,
            gradients: VectorLeafGradients { leaf, split },
            root_rows: (0..binned.row_count as u32).collect(),
            features: (0..binned.feature_count).collect(),
            round_index: 0,
            split_options: SplitSelectionOptions {
                l2_lambda: 1.0,
                min_rows_per_leaf: 2,
                ..SplitSelectionOptions::default()
            },
            params: &params,
            controls: &controls,
            feature_weights: &[],
            full_hessian: None,
        };
        let mut predictions = vec![vec![0.0_f32; binned.row_count]; leaf.len()];
        let trees = build_vector_leaf_tree_with(&inputs, &mut predictions, sparse_histograms)
            .expect("tree");
        (trees, predictions)
    }

    fn structure(trees: &[Vec<TrainedStump>]) -> Vec<(u32, u32, u16, bool)> {
        trees[0]
            .iter()
            .map(|stump| {
                (
                    stump.split.node_id,
                    stump.split.feature_index,
                    stump.split.threshold_bin,
                    stump.split.default_left,
                )
            })
            .collect()
    }

    #[test]
    fn sparse_and_dense_histograms_grow_the_same_tree() {
        let (binned, targets) = wide_fixture(150, 6);
        let leaf = gradients(&targets);
        for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
            let (sparse, sparse_predictions) = grow(&binned, &leaf, None, tree_growth, true);
            let (dense, dense_predictions) = grow(&binned, &leaf, None, tree_growth, false);
            assert!(
                structure(&sparse).len() > 3,
                "fixture grew too small a tree"
            );
            assert_eq!(structure(&sparse), structure(&dense));
            for (s, d) in sparse.iter().flatten().zip(dense.iter().flatten()) {
                assert_eq!(s.left_leaf_value, d.left_leaf_value);
                assert_eq!(s.right_leaf_value, d.right_leaf_value);
            }
            assert_eq!(sparse_predictions, dense_predictions);
        }
    }

    #[test]
    fn explicit_split_columns_equal_to_leaf_columns_change_nothing() {
        let (binned, targets) = wide_fixture(400, 4);
        let leaf = gradients(&targets);
        for tree_growth in [TreeGrowth::Level, TreeGrowth::Leaf] {
            let implicit = grow(&binned, &leaf, None, tree_growth, true);
            let explicit = grow(&binned, &leaf, Some(&leaf), tree_growth, true);
            assert_eq!(implicit, explicit);
        }
    }

    #[test]
    fn split_columns_choose_splits_and_leaf_columns_solve_leaves() {
        let (binned, targets) = wide_fixture(400, 4);
        let leaf = gradients(&targets);
        // Score splits on the first class alone: the tree follows that
        // class, while every class still gets its own Newton leaves.
        let first_class = leaf[..1].to_vec();
        let (trees, _) = grow(&binned, &leaf, Some(&first_class), TreeGrowth::Level, true);
        let (per_class, _) = grow(&binned, &first_class, None, TreeGrowth::Level, true);
        assert_eq!(structure(&trees), structure(&per_class));
        assert_eq!(trees.len(), CLASSES);
        assert!(
            trees[1..]
                .iter()
                .any(|stumps| stumps[0].left_leaf_value != trees[0][0].left_leaf_value)
        );
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4, 2], [2, 3]], b = [2, 1] -> x = [0.5, 0].
        let mut matrix = vec![4.0, 0.0, 2.0, 3.0];
        let mut rhs = vec![2.0, 1.0];
        cholesky_solve_in_place(&mut matrix, &mut rhs, 2).expect("positive definite");
        assert!(
            (rhs[0] - 0.5).abs() < 1e-12 && rhs[1].abs() < 1e-12,
            "{rhs:?}"
        );
        let mut singular = vec![1.0, 0.0, 1.0, 1.0];
        assert!(cholesky_solve_in_place(&mut singular, &mut [1.0, 1.0], 2).is_none());
    }

    #[test]
    fn abs_residual_hessian_is_the_gradient_magnitude() {
        let mut pairs = vec![
            GradientPair {
                grad: -0.75,
                hess: 0.1875,
            },
            GradientPair {
                grad: 0.25,
                hess: 0.1875,
            },
            GradientPair {
                grad: 0.0,
                hess: 1e-7,
            },
        ];
        apply_abs_residual_hessian(&mut pairs);
        let hessians: Vec<f32> = pairs.iter().map(|pair| pair.hess).collect();
        assert_eq!(hessians, vec![0.75, 0.25, 1e-7]);
    }
}
