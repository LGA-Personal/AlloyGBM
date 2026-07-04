use alloygbm_core::{
    CategoricalStatePayloadV1, CoreError, MODEL_FORMAT_V1, ModelArtifactSection, ModelMetadata,
    ModelSectionKind, decode_optional_categorical_state_section_v1,
    decode_optional_dart_tree_weights_section, decode_optional_linear_leaf_coefficients_section,
    decode_optional_native_categorical_splits_section, deserialize_model_artifact_v1,
    format_required_section_mode_error, required_section_compatibility_report,
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

const PARALLEL_PREDICT_MIN_ROWS: usize = 256;
const PARALLEL_PREDICT_MIN_WORK_ITEMS: usize = 16_384;
// Loading artifacts is a trust boundary: sparse heap-style node IDs can
// otherwise force large per-tree slot arrays before prediction ever runs. The
// same limit is enforced at train time (`encode_tree_node_id`), so every model
// the trainer emits loads and every artifact accepted here could have been
// trained.
use alloygbm_core::MAX_TREE_NODE_SLOTS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictorError {
    InvalidInput(String),
    ContractViolation(String),
    Core(CoreError),
}

impl Display for PredictorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::ContractViolation(msg) => write!(f, "contract violation: {msg}"),
            Self::Core(err) => write!(f, "core error: {err}"),
        }
    }
}

impl Error for PredictorError {}

impl From<CoreError> for PredictorError {
    fn from(value: CoreError) -> Self {
        Self::Core(value)
    }
}

pub type PredictorResult<T> = Result<T, PredictorError>;

/// Return type for `decode_multiclass_trees_payload`:
/// `(num_classes, feature_count, baselines, per_class_stumps)`.
type MultiClassTreesPayload = (usize, usize, Vec<f32>, Vec<Vec<PredictorStump>>);

/// Compact linear leaf for fast predictor evaluation.
/// Stores regressor feature indices as `usize` for zero-cost indexing.
#[derive(Debug, Clone, PartialEq)]
struct LinearLeafCompact {
    intercept: f32,
    weights: Vec<f32>,
    feature_indices: Vec<usize>,
}

impl LinearLeafCompact {
    /// PL-leaf evaluation with v0.9.0 NaN policy: NaN feature values
    /// contribute 0.0 to the linear sum. Matches `LinearLeaf::eval` in
    /// `alloygbm_core`. See Limitation 4 in `docs/limitations.md`.
    #[inline]
    fn eval(&self, features: &[f32]) -> f32 {
        let mut v = self.intercept;
        for (w, &fi) in self.weights.iter().zip(self.feature_indices.iter()) {
            if fi < features.len() {
                let x = features[fi];
                if !x.is_nan() {
                    v += w * x;
                }
            }
        }
        v
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PredictorStump {
    node_id: u32,
    feature_index: u32,
    threshold_bin: u16,
    default_left: bool,
    is_categorical: bool,
    categorical_bitset: Option<Vec<u8>>,
    left_leaf_value: f32,
    right_leaf_value: f32,
    left_linear: Option<LinearLeafCompact>,
    right_linear: Option<LinearLeafCompact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PredictorLayoutPayload {
    feature_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct PredictorTreeNode {
    feature_index: usize,
    threshold_bin: f32,
    default_left: bool,
    is_categorical: bool,
    categorical_bitset: Option<Vec<u8>>,
    left_leaf_value: f32,
    right_leaf_value: f32,
    left_linear: Option<LinearLeafCompact>,
    right_linear: Option<LinearLeafCompact>,
}

impl PredictorTreeNode {
    #[inline]
    fn eval_left_leaf(&self, features: &[f32]) -> f32 {
        match &self.left_linear {
            Some(ll) => ll.eval(features),
            None => self.left_leaf_value,
        }
    }

    #[inline]
    fn eval_right_leaf(&self, features: &[f32]) -> f32 {
        match &self.right_linear {
            Some(rl) => rl.eval(features),
            None => self.right_leaf_value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PredictorTree {
    nodes_by_local_id: Vec<Option<PredictorTreeNode>>,
    /// DART per-tree multiplicative weight. `1.0` for non-DART models
    /// and for any tree where the loader didn't find a `DartTreeWeights`
    /// entry. Applied multiplicatively to every leaf accumulation when
    /// traversing this tree.
    tree_weight: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    pub metadata: ModelMetadata,
    pub categorical_state: Option<CategoricalStatePayloadV1>,
    baseline_prediction: f32,
    trees: Vec<PredictorTree>,
    /// When true, `threshold_bin` fields contain float thresholds (not bin indices)
    /// and prediction uses `<` comparison instead of `<=`.
    use_float_thresholds: bool,
    // Multi-class fields (None for single-output models)
    num_classes: Option<usize>,
    baseline_predictions: Option<Vec<f32>>,
    class_trees: Option<Vec<Vec<PredictorTree>>>,
}

/// Determine if a prediction row goes left at a tree node.
/// Handles continuous (threshold comparison) and categorical (bitset membership) splits.
#[inline]
fn predictor_went_left(node: &PredictorTreeNode, feature_value: f32, use_float: bool) -> bool {
    if feature_value.is_nan() {
        node.default_left
    } else if node.is_categorical {
        let cat_id = feature_value as u16;
        node.categorical_bitset
            .as_ref()
            .map_or(node.default_left, |bs| {
                let byte_idx = (cat_id / 8) as usize;
                let bit_idx = (cat_id % 8) as usize;
                byte_idx < bs.len() && (bs[byte_idx] & (1 << bit_idx)) != 0
            })
    } else if use_float {
        feature_value < node.threshold_bin
    } else {
        feature_value <= node.threshold_bin
    }
}

impl Predictor {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            categorical_state: None,
            baseline_prediction: 0.0,
            trees: Vec::new(),
            use_float_thresholds: false,
            num_classes: None,
            baseline_predictions: None,
            class_trees: None,
        }
    }

    pub fn from_artifact_bytes(bytes: &[u8]) -> PredictorResult<Self> {
        let parsed = deserialize_model_artifact_v1(bytes).map_err(PredictorError::from)?;
        let metadata = parsed.contract.metadata;
        let metadata_feature_count = metadata.feature_names.len();
        let categorical_state =
            decode_optional_categorical_state_section_v1(&parsed.sections, metadata_feature_count)
                .map_err(PredictorError::from)?;

        // Check for multi-class trees section first
        if let Some(mc_section) =
            optional_single_section(&parsed.sections, ModelSectionKind::MultiClassTrees)?
        {
            let predictor_layout =
                resolve_predictor_layout(&parsed.sections, metadata_feature_count)?;
            let (num_classes, feature_count, baselines, mut per_class_stumps) =
                decode_multiclass_trees_payload(&mc_section.payload)?;

            // Decode optional NativeCategoricalSplits section for multiclass.
            if let Some(cat_payload) =
                decode_optional_native_categorical_splits_section(&parsed.sections)
                    .map_err(PredictorError::from)?
            {
                // Stump indices in the bitset section are global (flat) across all classes.
                let mut global_idx = 0usize;
                let stump_bitsets: std::collections::HashMap<u32, Vec<u8>> =
                    cat_payload.stump_bitsets.into_iter().collect();
                for class_stumps in &mut per_class_stumps {
                    for stump in class_stumps.iter_mut() {
                        if let Some(bitset) = stump_bitsets.get(&(global_idx as u32)) {
                            stump.categorical_bitset = Some(bitset.clone());
                        }
                        global_idx += 1;
                    }
                }
            }

            if predictor_layout.feature_count != metadata_feature_count {
                return Err(PredictorError::ContractViolation(format!(
                    "predictor layout feature_count {} does not match metadata feature count {}",
                    predictor_layout.feature_count, metadata_feature_count
                )));
            }
            if feature_count != metadata_feature_count {
                return Err(PredictorError::ContractViolation(format!(
                    "multiclass trees feature_count {} does not match metadata feature count {}",
                    feature_count, metadata_feature_count
                )));
            }

            // Decode optional linear leaf coefficients for multiclass.
            // Global stump index uses prefix-sum offsets:
            // global_idx = prefix[class_idx] + stump_within_class, where
            // prefix[k] = sum of stump counts for classes 0..k.
            if let Some(ll_payload) =
                decode_optional_linear_leaf_coefficients_section(&parsed.sections)
                    .map_err(PredictorError::from)?
            {
                // Build prefix sums from per-class stump counts.
                let mut prefix = vec![0usize; per_class_stumps.len() + 1];
                for (k, cs) in per_class_stumps.iter().enumerate() {
                    prefix[k + 1] = prefix[k] + cs.len();
                }
                for entry in ll_payload.entries {
                    let global_idx = entry.stump_idx as usize;
                    // Binary search to find which class this index belongs to.
                    // partition_point on prefix[1..] finds first k where prefix[k+1] > global_idx.
                    let class_idx = prefix[1..].partition_point(|&p| p <= global_idx);
                    if class_idx < per_class_stumps.len() {
                        let stump_idx = global_idx - prefix[class_idx];
                        if stump_idx < per_class_stumps[class_idx].len() {
                            if let Some(ll) = entry.left_leaf {
                                per_class_stumps[class_idx][stump_idx].left_linear =
                                    Some(linear_leaf_to_compact(&ll));
                            }
                            if let Some(rl) = entry.right_leaf {
                                per_class_stumps[class_idx][stump_idx].right_linear =
                                    Some(linear_leaf_to_compact(&rl));
                            }
                        }
                    }
                }
            }

            // Multiclass does not yet support DART (see v0.9.0 rejection in
            // engine fit_multiclass_iterations_impl), so even if a future
            // artifact carries DartTreeWeights here we keep the per-tree
            // weight at 1.0 across classes. This is forward-compatible:
            // when multiclass DART lands we can apply the overlay below.
            let mut class_trees = Vec::with_capacity(num_classes);
            for stumps in &per_class_stumps {
                let (trees, _tree_ids) = build_predictor_trees(stumps)?;
                class_trees.push(trees);
            }

            return Ok(Self {
                metadata,
                categorical_state,
                baseline_prediction: 0.0,
                trees: Vec::new(),
                use_float_thresholds: false,
                num_classes: Some(num_classes),
                baseline_predictions: Some(baselines),
                class_trees: Some(class_trees),
            });
        }

        // Single-output path (existing behavior)
        let compatibility_report = required_section_compatibility_report(&parsed.sections);
        if !compatibility_report.legacy_compatible {
            return Err(PredictorError::ContractViolation(
                format_required_section_mode_error(compatibility_report, true),
            ));
        }
        let trees_section = required_single_section(&parsed.sections, ModelSectionKind::Trees)?;
        let predictor_layout = resolve_predictor_layout(&parsed.sections, metadata_feature_count)?;
        let (payload_feature_count, baseline_prediction, mut stumps) =
            decode_trained_model_payload(&trees_section.payload)?;

        // Decode optional NativeCategoricalSplits section and populate stump bitsets.
        if let Some(cat_payload) =
            decode_optional_native_categorical_splits_section(&parsed.sections)
                .map_err(PredictorError::from)?
        {
            for (stump_index, bitset) in cat_payload.stump_bitsets {
                let idx = stump_index as usize;
                if idx < stumps.len() {
                    stumps[idx].categorical_bitset = Some(bitset);
                }
            }
        }

        // Decode optional linear leaf coefficients for single-output.
        if let Some(ll_payload) = decode_optional_linear_leaf_coefficients_section(&parsed.sections)
            .map_err(PredictorError::from)?
        {
            for entry in ll_payload.entries {
                let idx = entry.stump_idx as usize;
                if idx < stumps.len() {
                    if let Some(ll) = entry.left_leaf {
                        stumps[idx].left_linear = Some(linear_leaf_to_compact(&ll));
                    }
                    if let Some(rl) = entry.right_leaf {
                        stumps[idx].right_linear = Some(linear_leaf_to_compact(&rl));
                    }
                }
            }
        }

        let (mut trees, tree_ids) = build_predictor_trees(&stumps)?;

        // Decode optional DartTreeWeights and apply per-tree weights.
        if let Some(dart_payload) = decode_optional_dart_tree_weights_section(&parsed.sections)
            .map_err(PredictorError::from)?
        {
            apply_dart_tree_weights(&mut trees, &tree_ids, &stumps, &dart_payload.weights)?;
        }

        if predictor_layout.feature_count != metadata_feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "predictor layout feature_count {} does not match metadata feature count {}",
                predictor_layout.feature_count, metadata_feature_count
            )));
        }
        if payload_feature_count != predictor_layout.feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "trees payload feature_count {} does not match predictor layout feature_count {}",
                payload_feature_count, predictor_layout.feature_count
            )));
        }
        if payload_feature_count != metadata_feature_count {
            return Err(PredictorError::ContractViolation(format!(
                "trees payload feature_count {} does not match metadata feature count {}",
                payload_feature_count, metadata_feature_count
            )));
        }

        Ok(Self {
            metadata,
            categorical_state,
            baseline_prediction,
            trees,
            use_float_thresholds: false,
            num_classes: None,
            baseline_predictions: None,
            class_trees: None,
        })
    }

    /// Convert bin-index thresholds to float thresholds using per-feature min/max.
    /// After calling this, prediction compares raw float features directly — no quantization needed.
    /// Uses the midpoint between adjacent bin boundaries as the float threshold, with `<` comparison.
    ///
    /// `max_data_bin` is the maximum bin index used for data (excluding the NaN sentinel).
    /// For the default 256-bin configuration this is 254; for wider bins (e.g. 512) it is
    /// `max_bins - 2`.
    pub fn convert_bin_thresholds_to_float(
        &mut self,
        feature_mins: &[f32],
        feature_maxs: &[f32],
        max_data_bin: u16,
    ) -> PredictorResult<()> {
        let feature_count = self.metadata.feature_names.len();
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_mins/maxs length ({}/{}) must match feature_count {}",
                feature_mins.len(),
                feature_maxs.len(),
                feature_count
            )));
        }
        let divisor = max_data_bin as f32;
        let convert = |node: &mut PredictorTreeNode, fi: usize| {
            let min_val = feature_mins[fi];
            let max_val = feature_maxs[fi];
            let span = max_val - min_val;
            if span <= f32::EPSILON {
                node.threshold_bin = min_val + f32::EPSILON;
            } else {
                let bin = node.threshold_bin;
                node.threshold_bin = min_val + ((bin + 0.5) / divisor) * span;
            }
        };
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                let fi = node.feature_index;
                convert(node, fi);
            }
        }
        if let Some(class_trees) = &mut self.class_trees {
            for trees in class_trees.iter_mut() {
                for tree in trees.iter_mut() {
                    for node in tree.nodes_by_local_id.iter_mut().flatten() {
                        let fi = node.feature_index;
                        convert(node, fi);
                    }
                }
            }
        }
        self.use_float_thresholds = true;
        Ok(())
    }

    /// Convert bin-index thresholds to float thresholds using per-feature quantile cuts.
    /// For quantile binning: `bin = bisect_right(cuts, value)`, split is `bin <= threshold_bin`.
    /// The float equivalent: `value < cuts[threshold_bin]`.
    pub fn convert_bin_thresholds_to_float_quantile(
        &mut self,
        feature_cuts: &[Vec<f32>],
    ) -> PredictorResult<()> {
        let feature_count = self.metadata.feature_names.len();
        if feature_cuts.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_cuts length {} must match feature_count {}",
                feature_cuts.len(),
                feature_count
            )));
        }
        let convert = |node: &mut PredictorTreeNode, fi: usize| {
            let cuts = &feature_cuts[fi];
            let bin = node.threshold_bin as usize;
            node.threshold_bin = if bin < cuts.len() {
                cuts[bin]
            } else {
                f32::MAX
            };
        };
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                let fi = node.feature_index;
                convert(node, fi);
            }
        }
        if let Some(class_trees) = &mut self.class_trees {
            for trees in class_trees.iter_mut() {
                for tree in trees.iter_mut() {
                    for node in tree.nodes_by_local_id.iter_mut().flatten() {
                        let fi = node.feature_index;
                        convert(node, fi);
                    }
                }
            }
        }
        self.use_float_thresholds = true;
        Ok(())
    }

    /// Convert bin-index thresholds to float thresholds for pre-binned integer data.
    /// For pre-binned data, values are integers (0..max_bin) and split is `bin <= threshold_bin`.
    /// The float equivalent: `value < threshold_bin + 0.5`.
    pub fn convert_bin_thresholds_to_float_prebinned(&mut self) -> PredictorResult<()> {
        for tree in &mut self.trees {
            for node in tree.nodes_by_local_id.iter_mut().flatten() {
                node.threshold_bin += 0.5;
            }
        }
        if let Some(class_trees) = &mut self.class_trees {
            for trees in class_trees.iter_mut() {
                for tree in trees.iter_mut() {
                    for node in tree.nodes_by_local_id.iter_mut().flatten() {
                        node.threshold_bin += 0.5;
                    }
                }
            }
        }
        self.use_float_thresholds = true;
        Ok(())
    }

    pub fn is_multiclass(&self) -> bool {
        self.num_classes.is_some()
    }

    pub fn num_classes(&self) -> Option<usize> {
        self.num_classes
    }

    pub fn predict_row(&self, features: &[f32]) -> PredictorResult<f32> {
        if self.is_multiclass() {
            return Err(PredictorError::ContractViolation(
                "use predict_batch_multiclass for multi-class models".to_string(),
            ));
        }
        let feature_count = self.metadata.feature_names.len();
        if features.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                feature_count
            )));
        }
        let raw = self.predict_row_with_feature_count(features, feature_count)?;
        Ok(self.post_transform(raw))
    }

    fn predict_row_with_feature_count(
        &self,
        features: &[f32],
        feature_count: usize,
    ) -> PredictorResult<f32> {
        let use_float = self.use_float_thresholds;
        let mut prediction = self.baseline_prediction;
        for tree in &self.trees {
            let mut local_node_id: usize = 0;
            while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                if node.feature_index >= feature_count {
                    return Err(PredictorError::ContractViolation(format!(
                        "split feature_index {} exceeds feature length {}",
                        node.feature_index, feature_count
                    )));
                }
                let feature_value = features[node.feature_index];
                let went_left = predictor_went_left(node, feature_value, use_float);
                let leaf = if went_left {
                    node.eval_left_leaf(features)
                } else {
                    node.eval_right_leaf(features)
                };
                prediction += tree.tree_weight * leaf;
                local_node_id = if went_left {
                    local_node_id.saturating_mul(2).saturating_add(1)
                } else {
                    local_node_id.saturating_mul(2).saturating_add(2)
                };
            }
        }

        Ok(prediction)
    }

    pub fn predict_batch(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        if self.is_multiclass() {
            return Err(PredictorError::ContractViolation(
                "use predict_batch_multiclass for multi-class models".to_string(),
            ));
        }
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }
        let feature_count = self.metadata.feature_names.len();
        for row in rows {
            if row.len() != feature_count {
                return Err(PredictorError::InvalidInput(format!(
                    "feature length {} does not match model feature_count {}",
                    row.len(),
                    feature_count
                )));
            }
        }

        if should_parallel_predict_batch(rows.len(), self.trees.len()) {
            rows.par_iter()
                .map(|row| {
                    let raw = self.predict_row_with_feature_count(row, feature_count)?;
                    Ok(self.post_transform(raw))
                })
                .collect()
        } else {
            rows.iter()
                .map(|row| {
                    let raw = self.predict_row_with_feature_count(row, feature_count)?;
                    Ok(self.post_transform(raw))
                })
                .collect()
        }
    }

    /// Predict from a flat row-major dense slice — zero per-row allocation.
    pub fn predict_batch_dense(
        &self,
        values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> PredictorResult<Vec<f32>> {
        if self.is_multiclass() {
            return Err(PredictorError::ContractViolation(
                "use predict_batch_dense_multiclass for multi-class models".to_string(),
            ));
        }
        let model_feature_count = self.metadata.feature_names.len();
        if feature_count != model_feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_count {} does not match model feature_count {}",
                feature_count, model_feature_count
            )));
        }
        if values.len() != row_count * feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "values length {} does not match row_count * feature_count {}",
                values.len(),
                row_count * feature_count
            )));
        }
        if row_count == 0 {
            return Err(PredictorError::InvalidInput(
                "row_count must be greater than 0".to_string(),
            ));
        }

        if should_parallel_predict_batch(row_count, self.trees.len()) {
            (0..row_count)
                .into_par_iter()
                .map(|row_index| {
                    let row = &values[row_index * feature_count..(row_index + 1) * feature_count];
                    let raw = self.predict_row_dense_unchecked(row, feature_count)?;
                    Ok(self.post_transform(raw))
                })
                .collect()
        } else {
            (0..row_count)
                .map(|row_index| {
                    let row = &values[row_index * feature_count..(row_index + 1) * feature_count];
                    let raw = self.predict_row_dense_unchecked(row, feature_count)?;
                    Ok(self.post_transform(raw))
                })
                .collect()
        }
    }

    /// Inner prediction on a row slice — no length validation (caller ensures correctness).
    fn predict_row_dense_unchecked(
        &self,
        features: &[f32],
        feature_count: usize,
    ) -> PredictorResult<f32> {
        let use_float = self.use_float_thresholds;
        let mut prediction = self.baseline_prediction;
        for tree in &self.trees {
            let mut local_node_id: usize = 0;
            while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                if node.feature_index >= feature_count {
                    return Err(PredictorError::ContractViolation(format!(
                        "split feature_index {} exceeds feature length {}",
                        node.feature_index, feature_count
                    )));
                }
                let feature_value = features[node.feature_index];
                let went_left = predictor_went_left(node, feature_value, use_float);
                let leaf = if went_left {
                    node.eval_left_leaf(features)
                } else {
                    node.eval_right_leaf(features)
                };
                prediction += tree.tree_weight * leaf;
                local_node_id = if went_left {
                    local_node_id * 2 + 1
                } else {
                    local_node_id * 2 + 2
                };
            }
        }
        Ok(prediction)
    }

    /// Predict from raw native-endian f32 bytes — avoids Python list→Vec<f32> overhead.
    /// Each parallel chunk converts bytes→f32 and predicts on the fly using a thread-local
    /// row buffer, avoiding any large intermediate allocation.
    pub fn predict_batch_dense_bytes(
        &self,
        bytes: &[u8],
        row_count: usize,
        feature_count: usize,
    ) -> PredictorResult<Vec<f32>> {
        if self.is_multiclass() {
            return Err(PredictorError::ContractViolation(
                "use predict_batch_multiclass for multi-class models".to_string(),
            ));
        }
        let model_feature_count = self.metadata.feature_names.len();
        if feature_count != model_feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_count {} does not match model feature_count {}",
                feature_count, model_feature_count
            )));
        }
        let expected_bytes = row_count * feature_count * 4;
        if bytes.len() != expected_bytes {
            return Err(PredictorError::InvalidInput(format!(
                "bytes length {} does not match expected {} (row_count={} * feature_count={} * 4)",
                bytes.len(),
                expected_bytes,
                row_count,
                feature_count
            )));
        }
        if row_count == 0 {
            return Err(PredictorError::InvalidInput(
                "row_count must be greater than 0".to_string(),
            ));
        }

        let row_bytes = feature_count * 4;
        let chunk_size = 4096.max(row_count / (rayon::current_num_threads().max(1) * 4));
        let mut predictions = vec![0.0_f32; row_count];

        // Each parallel chunk gets one reusable row buffer (feature_count × 4 bytes).
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let mut row_buf = vec![0.0_f32; feature_count];
                for (local_idx, pred) in out_chunk.iter_mut().enumerate() {
                    let row_index = row_start + local_idx;
                    let byte_start = row_index * row_bytes;
                    for (fi, item) in row_buf.iter_mut().enumerate().take(feature_count) {
                        let bi = byte_start + fi * 4;
                        *item = f32::from_ne_bytes([
                            bytes[bi],
                            bytes[bi + 1],
                            bytes[bi + 2],
                            bytes[bi + 3],
                        ]);
                    }
                    let raw = self.predict_row_dense_unchecked(&row_buf, feature_count)?;
                    *pred = self.post_transform(raw);
                }
                Ok::<(), PredictorError>(())
            })?;

        Ok(predictions)
    }

    pub fn predict_row_stub(&self, features: &[f32]) -> PredictorResult<f32> {
        self.predict_row(features)
    }

    pub fn predict_batch_stub(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        self.predict_batch(rows)
    }

    /// Apply objective-specific post-transform to a raw prediction.
    /// - `"squared_error"`, `"queryrmse"`, ranking objectives: identity.
    /// - `"binary_crossentropy"`: sigmoid (logit → probability).
    /// - `"poisson"` / `"gamma"` / `"tweedie"`: `exp` (log-link → mean μ).
    ///   η is clamped to [-50, 50] to keep μ in finite f32 range, mirroring
    ///   the training-side clamp in `glm_clamp_exp`.
    #[inline]
    fn post_transform(&self, raw: f32) -> f32 {
        match self.metadata.objective.as_str() {
            "binary_crossentropy" => sigmoid(raw),
            "poisson" | "gamma" | "tweedie" => raw.clamp(-50.0, 50.0).exp(),
            "quantile" => raw,
            _ => raw,
        }
    }

    /// Predict raw logits (before post-transform).
    pub fn predict_row_raw(&self, features: &[f32]) -> PredictorResult<f32> {
        let feature_count = self.metadata.feature_names.len();
        if features.len() != feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature length {} does not match model feature_count {}",
                features.len(),
                feature_count
            )));
        }
        self.predict_row_with_feature_count(features, feature_count)
    }

    // -- Multi-class prediction ------------------------------------------------

    /// Predict K probabilities per row (softmax-normalized).
    /// Returns a flat `Vec<f32>` of length `n_rows * K` in row-major order:
    /// `[row0_class0, row0_class1, ..., row0_classK-1, row1_class0, ...]`
    pub fn predict_batch_multiclass(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        let k = self.num_classes.ok_or_else(|| {
            PredictorError::ContractViolation(
                "predict_batch_multiclass called on single-output model".to_string(),
            )
        })?;
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }
        let feature_count = self.metadata.feature_names.len();
        for row in rows {
            if row.len() != feature_count {
                return Err(PredictorError::InvalidInput(format!(
                    "feature length {} does not match model feature_count {}",
                    row.len(),
                    feature_count
                )));
            }
        }
        let baselines = self.baseline_predictions.as_ref().unwrap();
        let class_trees = self.class_trees.as_ref().unwrap();

        let predict_row = |features: &[f32]| -> PredictorResult<Vec<f32>> {
            let mut logits = baselines.clone();
            for (class_k, trees) in class_trees.iter().enumerate() {
                for tree in trees {
                    let mut local_node_id: usize = 0;
                    while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                        if node.feature_index >= feature_count {
                            return Err(PredictorError::ContractViolation(format!(
                                "split feature_index {} exceeds feature length {}",
                                node.feature_index, feature_count
                            )));
                        }
                        let feature_value = features[node.feature_index];
                        let went_left =
                            predictor_went_left(node, feature_value, self.use_float_thresholds);
                        let leaf = if went_left {
                            node.eval_left_leaf(features)
                        } else {
                            node.eval_right_leaf(features)
                        };
                        logits[class_k] += tree.tree_weight * leaf;
                        local_node_id = if went_left {
                            local_node_id * 2 + 1
                        } else {
                            local_node_id * 2 + 2
                        };
                    }
                }
            }
            softmax_in_place(&mut logits);
            Ok(logits)
        };

        let mut output = Vec::with_capacity(rows.len() * k);
        if should_parallel_predict_batch(rows.len(), class_trees.iter().map(|t| t.len()).sum()) {
            let results: PredictorResult<Vec<Vec<f32>>> =
                rows.par_iter().map(|row| predict_row(row)).collect();
            for probs in results? {
                output.extend_from_slice(&probs);
            }
        } else {
            for row in rows {
                let probs = predict_row(row)?;
                output.extend_from_slice(&probs);
            }
        }
        Ok(output)
    }

    /// Multi-class prediction from a flat row-major dense slice.
    /// Returns a flat `Vec<f32>` of length `row_count * K`.
    pub fn predict_batch_dense_multiclass(
        &self,
        values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> PredictorResult<Vec<f32>> {
        if !self.is_multiclass() {
            return Err(PredictorError::ContractViolation(
                "predict_batch_dense_multiclass called on single-output model".to_string(),
            ));
        }
        let model_feature_count = self.metadata.feature_names.len();
        if feature_count != model_feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "feature_count {} does not match model feature_count {}",
                feature_count, model_feature_count
            )));
        }
        if values.len() != row_count * feature_count {
            return Err(PredictorError::InvalidInput(format!(
                "values length {} does not match row_count * feature_count {}",
                values.len(),
                row_count * feature_count
            )));
        }
        if row_count == 0 {
            return Err(PredictorError::InvalidInput(
                "row_count must be greater than 0".to_string(),
            ));
        }

        let k = self.num_classes.unwrap(); // already validated is_multiclass above
        let baselines = self.baseline_predictions.as_ref().unwrap();
        let class_trees = self.class_trees.as_ref().unwrap();
        let use_float = self.use_float_thresholds;

        let predict_row_from_slice = |features: &[f32]| -> PredictorResult<Vec<f32>> {
            let mut logits = baselines.clone();
            for (class_k, trees) in class_trees.iter().enumerate() {
                for tree in trees {
                    let mut local_node_id: usize = 0;
                    while let Some(Some(node)) = tree.nodes_by_local_id.get(local_node_id) {
                        if node.feature_index >= feature_count {
                            return Err(PredictorError::ContractViolation(format!(
                                "split feature_index {} exceeds feature length {}",
                                node.feature_index, feature_count
                            )));
                        }
                        let feature_value = features[node.feature_index];
                        let went_left = predictor_went_left(node, feature_value, use_float);
                        let leaf = if went_left {
                            node.eval_left_leaf(features)
                        } else {
                            node.eval_right_leaf(features)
                        };
                        logits[class_k] += tree.tree_weight * leaf;
                        local_node_id = if went_left {
                            local_node_id * 2 + 1
                        } else {
                            local_node_id * 2 + 2
                        };
                    }
                }
            }
            softmax_in_place(&mut logits);
            Ok(logits)
        };

        let total_trees: usize = class_trees.iter().map(|t| t.len()).sum();
        let mut output = Vec::with_capacity(row_count * k);
        if should_parallel_predict_batch(row_count, total_trees) {
            let results: PredictorResult<Vec<Vec<f32>>> = (0..row_count)
                .into_par_iter()
                .map(|i| {
                    let row = &values[i * feature_count..(i + 1) * feature_count];
                    predict_row_from_slice(row)
                })
                .collect();
            for probs in results? {
                output.extend_from_slice(&probs);
            }
        } else {
            for i in 0..row_count {
                let row = &values[i * feature_count..(i + 1) * feature_count];
                let probs = predict_row_from_slice(row)?;
                output.extend_from_slice(&probs);
            }
        }
        Ok(output)
    }

    /// Predict raw logits in batch (before post-transform).
    pub fn predict_batch_raw(&self, rows: &[Vec<f32>]) -> PredictorResult<Vec<f32>> {
        if rows.is_empty() {
            return Err(PredictorError::InvalidInput(
                "rows cannot be empty".to_string(),
            ));
        }
        let feature_count = self.metadata.feature_names.len();
        for row in rows {
            if row.len() != feature_count {
                return Err(PredictorError::InvalidInput(format!(
                    "feature length {} does not match model feature_count {}",
                    row.len(),
                    feature_count
                )));
            }
        }
        if should_parallel_predict_batch(rows.len(), self.trees.len()) {
            rows.par_iter()
                .map(|row| self.predict_row_with_feature_count(row, feature_count))
                .collect()
        } else {
            rows.iter()
                .map(|row| self.predict_row_with_feature_count(row, feature_count))
                .collect()
        }
    }
}

/// Numerically stable sigmoid: avoids overflow for large negative inputs.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn should_parallel_predict_batch(row_count: usize, tree_count: usize) -> bool {
    row_count >= PARALLEL_PREDICT_MIN_ROWS
        && row_count.saturating_mul(tree_count.max(1)) >= PARALLEL_PREDICT_MIN_WORK_ITEMS
}

fn required_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> PredictorResult<&ModelArtifactSection> {
    optional_single_section(sections, kind)?.ok_or_else(|| {
        PredictorError::ContractViolation(format!(
            "model artifact missing required {:?} section",
            kind
        ))
    })
}

fn optional_single_section(
    sections: &[ModelArtifactSection],
    kind: ModelSectionKind,
) -> PredictorResult<Option<&ModelArtifactSection>> {
    let mut found = None;
    for section in sections {
        if section.descriptor.kind != kind {
            continue;
        }
        if found.is_some() {
            return Err(PredictorError::ContractViolation(format!(
                "model artifact contains duplicate required {:?} sections",
                kind
            )));
        }
        found = Some(section);
    }
    Ok(found)
}

fn resolve_predictor_layout(
    sections: &[ModelArtifactSection],
    metadata_feature_count: usize,
) -> PredictorResult<PredictorLayoutPayload> {
    if let Some(section) = optional_single_section(sections, ModelSectionKind::PredictorLayout)? {
        return decode_predictor_layout_payload(&section.payload);
    }

    if sections.len() == 1 && sections[0].descriptor.kind == ModelSectionKind::Trees {
        return Ok(PredictorLayoutPayload {
            feature_count: metadata_feature_count,
        });
    }

    Err(PredictorError::ContractViolation(
        "model artifact missing required PredictorLayout section".to_string(),
    ))
}

fn decode_predictor_layout_payload(bytes: &[u8]) -> PredictorResult<PredictorLayoutPayload> {
    const LAYOUT_LEN: usize = 12;
    const THRESHOLD_MODE_BIN_INDEX: u32 = 1;
    if bytes.len() != LAYOUT_LEN {
        return Err(PredictorError::ContractViolation(format!(
            "predictor layout payload length {} does not match expected {LAYOUT_LEN}",
            bytes.len()
        )));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported predictor layout format version {format_version}"
        )));
    }

    let feature_count = read_u32_le(bytes, 4)? as usize;
    let threshold_mode = read_u32_le(bytes, 8)?;
    if threshold_mode != THRESHOLD_MODE_BIN_INDEX {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported predictor layout threshold mode {threshold_mode}"
        )));
    }

    Ok(PredictorLayoutPayload { feature_count })
}

fn decode_trained_model_payload(
    bytes: &[u8],
) -> PredictorResult<(usize, f32, Vec<PredictorStump>)> {
    const HEADER_SIZE: usize = 16;
    const STUMP_SIZE: usize = 32;
    if bytes.len() < HEADER_SIZE {
        return Err(PredictorError::ContractViolation(
            "model payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported model payload format version {format_version}"
        )));
    }
    let feature_count = read_u32_le(bytes, 4)? as usize;
    let stump_count = read_u32_le(bytes, 8)? as usize;
    let baseline_prediction = read_f32_le(bytes, 12)?;

    let expected_len = HEADER_SIZE
        .checked_add(stump_count.checked_mul(STUMP_SIZE).ok_or_else(|| {
            PredictorError::ContractViolation("stump payload length overflow".to_string())
        })?)
        .ok_or_else(|| PredictorError::ContractViolation("payload length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(PredictorError::ContractViolation(format!(
            "model payload length {} does not match expected {}",
            bytes.len(),
            expected_len
        )));
    }

    let mut stumps = Vec::with_capacity(stump_count);
    for stump_index in 0..stump_count {
        let base = HEADER_SIZE + stump_index * STUMP_SIZE;
        let node_id = read_u32_le(bytes, base)?;
        let feature_index = read_u32_le(bytes, base + 4)?;
        let threshold_bin = read_u16_le(bytes, base + 8)?;
        let stump_flags = read_u16_le(bytes, base + 10)?;
        let default_left = (stump_flags & 1) != 0;
        let is_categorical = (stump_flags & 2) != 0;
        let _gain = read_f32_le(bytes, base + 12)?;
        let left_leaf_value = read_f32_le(bytes, base + 16)?;
        let right_leaf_value = read_f32_le(bytes, base + 20)?;
        stumps.push(PredictorStump {
            node_id,
            feature_index,
            threshold_bin,
            default_left,
            is_categorical,
            categorical_bitset: None, // populated from NativeCategoricalSplits section
            left_leaf_value,
            right_leaf_value,
            left_linear: None,  // populated from LinearLeafCoefficients section
            right_linear: None, // populated from LinearLeafCoefficients section
        });
    }

    Ok((feature_count, baseline_prediction, stumps))
}

const TREE_NODE_STRIDE: u32 = 1 << 20;

fn decode_tree_node_id(node_id: u32) -> (u32, u32) {
    (node_id / TREE_NODE_STRIDE, node_id % TREE_NODE_STRIDE)
}

fn linear_leaf_to_compact(ll: &alloygbm_core::LinearLeaf) -> LinearLeafCompact {
    LinearLeafCompact {
        intercept: ll.intercept,
        weights: ll.weights.to_vec(),
        feature_indices: ll.regressor_features.iter().map(|&f| f as usize).collect(),
    }
}

/// Build predictor trees and a parallel `Vec<u32>` of source tree_ids
/// (one per built tree, in the same order). The tree_ids let callers
/// overlay per-tree-id state — most notably DART tree weights —
/// without re-grouping stumps.
fn build_predictor_trees(
    stumps: &[PredictorStump],
) -> PredictorResult<(Vec<PredictorTree>, Vec<u32>)> {
    let mut grouped_by_tree: BTreeMap<u32, Vec<(u32, PredictorTreeNode)>> = BTreeMap::new();
    let mut first_stump_idx_per_tree: BTreeMap<u32, usize> = BTreeMap::new();
    for (stump_idx, stump) in stumps.iter().enumerate() {
        let (tree_id, local_node_id) = decode_tree_node_id(stump.node_id);
        first_stump_idx_per_tree.entry(tree_id).or_insert(stump_idx);
        grouped_by_tree.entry(tree_id).or_default().push((
            local_node_id,
            PredictorTreeNode {
                feature_index: stump.feature_index as usize,
                threshold_bin: stump.threshold_bin as f32,
                default_left: stump.default_left,
                is_categorical: stump.is_categorical,
                categorical_bitset: stump.categorical_bitset.clone(),
                left_leaf_value: stump.left_leaf_value,
                right_leaf_value: stump.right_leaf_value,
                left_linear: stump.left_linear.clone(),
                right_linear: stump.right_linear.clone(),
            },
        ));
    }

    let mut trees = Vec::with_capacity(grouped_by_tree.len());
    let mut tree_ids = Vec::with_capacity(grouped_by_tree.len());
    for (tree_id, nodes) in grouped_by_tree {
        let max_local_node_id = nodes
            .iter()
            .map(|(local_node_id, _)| *local_node_id as usize)
            .max()
            .unwrap_or(0);
        let required_node_slots = max_local_node_id.checked_add(1).ok_or_else(|| {
            PredictorError::ContractViolation(format!(
                "tree {tree_id} local node_id {max_local_node_id} overflowed predictor node slot count"
            ))
        })?;
        if required_node_slots > MAX_TREE_NODE_SLOTS {
            return Err(PredictorError::ContractViolation(format!(
                "tree {tree_id} local node_id {max_local_node_id} requires {required_node_slots} predictor node slots, exceeds predictor tree node slot limit {MAX_TREE_NODE_SLOTS}"
            )));
        }
        let mut nodes_by_local_id = vec![None; max_local_node_id + 1];
        for (local_node_id, node) in nodes {
            let local_node_id = local_node_id as usize;
            if nodes_by_local_id[local_node_id].is_some() {
                return Err(PredictorError::ContractViolation(format!(
                    "tree {tree_id} contains duplicate local node_id {local_node_id}"
                )));
            }
            nodes_by_local_id[local_node_id] = Some(node);
        }
        trees.push(PredictorTree {
            nodes_by_local_id,
            tree_weight: 1.0,
        });
        tree_ids.push(tree_id);
    }

    Ok((trees, tree_ids))
}

/// Apply a `DartTreeWeights` payload (length parallel to `stumps`) onto
/// the per-tree weights of `trees`. Each tree gets the weight of the
/// first stump that maps to it. Returns an error if the payload length
/// doesn't match `stumps`.
fn apply_dart_tree_weights(
    trees: &mut [PredictorTree],
    tree_ids: &[u32],
    stumps: &[PredictorStump],
    weights: &[f32],
) -> PredictorResult<()> {
    if weights.len() != stumps.len() {
        return Err(PredictorError::ContractViolation(format!(
            "DartTreeWeights length {} != stump count {}",
            weights.len(),
            stumps.len()
        )));
    }
    // tree_id -> position in `trees`/`tree_ids`
    let mut tree_index: BTreeMap<u32, usize> = BTreeMap::new();
    for (i, &tid) in tree_ids.iter().enumerate() {
        tree_index.insert(tid, i);
    }
    // First stump for each tree_id — its weight is the per-tree weight.
    let mut seen: BTreeMap<u32, ()> = BTreeMap::new();
    for (stump_idx, stump) in stumps.iter().enumerate() {
        let (tid, _) = decode_tree_node_id(stump.node_id);
        if seen.insert(tid, ()).is_some() {
            continue;
        }
        if let Some(&ti) = tree_index.get(&tid) {
            trees[ti].tree_weight = weights[stump_idx];
        }
    }
    Ok(())
}

fn read_u32_le(bytes: &[u8], start: usize) -> PredictorResult<u32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading u32".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}

fn read_u16_le(bytes: &[u8], start: usize) -> PredictorResult<u16> {
    let end = start + 2;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading u16".to_string(),
        ));
    }
    Ok(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

fn read_f32_le(bytes: &[u8], start: usize) -> PredictorResult<f32> {
    let end = start + 4;
    if end > bytes.len() {
        return Err(PredictorError::ContractViolation(
            "unexpected end of payload when reading f32".to_string(),
        ));
    }
    Ok(f32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ]))
}

fn softmax_in_place(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        sum += *logit;
    }
    if sum > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= sum;
        }
    }
}

/// Decode a MultiClassTrees section payload.
/// Returns `(num_classes, feature_count, baselines, per_class_stumps)`.
fn decode_multiclass_trees_payload(bytes: &[u8]) -> PredictorResult<MultiClassTreesPayload> {
    const MC_HEADER_SIZE: usize = 12;
    const STUMP_SIZE: usize = 32;

    if bytes.len() < MC_HEADER_SIZE {
        return Err(PredictorError::ContractViolation(
            "multiclass trees payload too small".to_string(),
        ));
    }

    let format_version = read_u32_le(bytes, 0)?;
    if format_version != MODEL_FORMAT_V1 {
        return Err(PredictorError::ContractViolation(format!(
            "unsupported multiclass trees format version {format_version}"
        )));
    }
    let num_classes = read_u32_le(bytes, 4)? as usize;
    let feature_count = read_u32_le(bytes, 8)? as usize;

    let baselines_start = MC_HEADER_SIZE;
    let baselines_end = baselines_start + num_classes * 4;
    if bytes.len() < baselines_end {
        return Err(PredictorError::ContractViolation(
            "multiclass trees payload too small for baselines".to_string(),
        ));
    }
    let mut baselines = Vec::with_capacity(num_classes);
    for k in 0..num_classes {
        baselines.push(read_f32_le(bytes, baselines_start + k * 4)?);
    }

    let counts_start = baselines_end;
    let counts_end = counts_start + num_classes * 4;
    if bytes.len() < counts_end {
        return Err(PredictorError::ContractViolation(
            "multiclass trees payload too small for stump counts".to_string(),
        ));
    }
    let mut stump_counts = Vec::with_capacity(num_classes);
    for k in 0..num_classes {
        stump_counts.push(read_u32_le(bytes, counts_start + k * 4)? as usize);
    }

    let total_stumps: usize = stump_counts.iter().sum();
    let stumps_start = counts_end;
    let expected_len = stumps_start + total_stumps * STUMP_SIZE;
    if bytes.len() != expected_len {
        return Err(PredictorError::ContractViolation(format!(
            "multiclass trees payload length {} does not match expected {expected_len}",
            bytes.len()
        )));
    }

    let mut per_class_stumps = Vec::with_capacity(num_classes);
    let mut offset = stumps_start;
    for &count in stump_counts.iter().take(num_classes) {
        let mut stumps = Vec::with_capacity(count);
        for _ in 0..count {
            let node_id = read_u32_le(bytes, offset)?;
            let feature_index = read_u32_le(bytes, offset + 4)?;
            let threshold_bin = read_u16_le(bytes, offset + 8)?;
            let flags = read_u16_le(bytes, offset + 10)?;
            let default_left = (flags & 1) != 0;
            let is_categorical = (flags & 2) != 0;
            let _gain = read_f32_le(bytes, offset + 12)?;
            let left_leaf_value = read_f32_le(bytes, offset + 16)?;
            let right_leaf_value = read_f32_le(bytes, offset + 20)?;

            stumps.push(PredictorStump {
                node_id,
                feature_index,
                threshold_bin,
                default_left,
                is_categorical,
                categorical_bitset: None,
                left_leaf_value,
                right_leaf_value,
                left_linear: None,  // populated from LinearLeafCoefficients section
                right_linear: None, // populated from LinearLeafCoefficients section
            });
            offset += STUMP_SIZE;
        }
        per_class_stumps.push(stumps);
    }

    Ok((num_classes, feature_count, baselines, per_class_stumps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_core::{
        BinnedMatrix, CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DatasetMatrix,
        Device, LeafModelKind, ModelSectionKind, TrainParams, TrainingDataset, TreeGrowth,
        serialize_model_artifact_v1,
    };
    use alloygbm_core::{LeafValue, NodeStats, SplitCandidate};
    use alloygbm_engine::{SquaredErrorObjective, TrainedModel, TrainedStump, Trainer};

    fn predictor_stub() -> Predictor {
        let metadata = ModelMetadata {
            format_version: 1,
            feature_names: vec!["f0".to_string()],
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        Predictor::new(metadata)
    }

    fn quality_fixture_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    4.0, 0.0, //
                    5.0, 0.0, //
                    6.0, 0.0, //
                    7.0, 0.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn quality_fixture_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            8,
            2,
            7,
            vec![
                0, 0, //
                1, 0, //
                2, 0, //
                3, 0, //
                4, 0, //
                5, 0, //
                6, 0, //
                7, 0, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn fixture_rows(dataset: &TrainingDataset) -> Vec<Vec<f32>> {
        dataset
            .matrix
            .values
            .chunks(dataset.matrix.feature_count)
            .map(|row| row.to_vec())
            .collect()
    }

    fn fixture_params() -> TrainParams {
        TrainParams {
            seed: 7,
            deterministic: true,
            learning_rate: 0.3,
            max_depth: 2,
            row_subsample: 1.0,
            col_subsample: 1.0,
            early_stopping_rounds: None,
            min_validation_improvement: 0.0,
            min_data_in_leaf: 1,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_child_hessian: 0.0,
            min_split_gain: 0.0,
            monotone_constraints: Vec::new(),
            feature_weights: Vec::new(),
            interaction_constraints: Vec::new(),
            max_leaves: None,
            tree_growth: TreeGrowth::Level,
            morph_config: None,
            leaf_solver: alloygbm_core::LeafSolverKind::Standard,
            dro_config: None,
            leaf_model: LeafModelKind::Constant,
            neutralization_config: None,
            boosting_mode: alloygbm_core::BoostingMode::Standard,
            tweedie_variance_power: 1.5,
            poisson_max_delta_step: 0.7,
            quantile_alpha: 0.5,
        }
    }

    fn train_engine_model() -> (alloygbm_engine::TrainedModel, TrainingDataset) {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 2)
            .expect("training succeeds");
        (model, dataset)
    }

    fn strict_artifact_payloads() -> (ModelMetadata, Vec<u8>, Vec<u8>) {
        let (engine_model, _) = train_engine_model();
        let strict_artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&strict_artifact)
            .expect("strict artifact parses");
        let trees_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists");
        let layout_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
            .map(|section| section.payload.clone())
            .expect("predictor layout payload exists");
        (parsed.contract.metadata, trees_payload, layout_payload)
    }

    #[test]
    fn predictor_from_artifact_matches_engine_predictions() {
        let (engine_model, dataset) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let predictor_predictions = predictor.predict_batch(&rows).expect("predictor predicts");
        assert_eq!(engine_predictions, predictor_predictions);
    }

    #[test]
    fn predictor_replays_artifact_with_optional_categorical_state() {
        let (engine_model, dataset) = train_engine_model();
        let engine_model = engine_model
            .with_categorical_state(Some(CategoricalStatePayloadV1 {
                format_version: CATEGORICAL_STATE_FORMAT_V1,
                leakage_safe_target_encoding: true,
                categorical_feature_indices: vec![1],
            }))
            .expect("categorical state is valid");
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);

        assert!(predictor.categorical_state.is_some());
        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let predictor_predictions = predictor.predict_batch(&rows).expect("predictor predicts");
        assert_eq!(engine_predictions, predictor_predictions);
    }

    #[test]
    fn predictor_row_matches_engine_prediction() {
        let (engine_model, dataset) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let rows = fixture_rows(&dataset);
        let row = &rows[0];

        let engine_prediction = engine_model.predict_row(row).expect("engine predicts");
        let predictor_prediction = predictor.predict_row(row).expect("predictor predicts");
        assert_eq!(engine_prediction, predictor_prediction);
    }

    #[test]
    fn predictor_accepts_legacy_trees_only_artifact() {
        let (engine_model, dataset) = train_engine_model();
        let strict_artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&strict_artifact)
            .expect("strict artifact parses");
        let trees_payload = parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
            .map(|section| section.payload.clone())
            .expect("trees payload exists");
        let legacy_artifact = serialize_model_artifact_v1(
            &parsed.contract.metadata,
            &[(ModelSectionKind::Trees, trees_payload)],
        )
        .expect("legacy artifact serializes");

        let predictor = Predictor::from_artifact_bytes(&legacy_artifact).expect("legacy parses");
        let rows = fixture_rows(&dataset);
        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let predictor_predictions = predictor.predict_batch(&rows).expect("predictor predicts");
        assert_eq!(engine_predictions, predictor_predictions);
    }

    #[test]
    fn predictor_rejects_duplicate_required_sections() {
        let (metadata, trees_payload, layout_payload) = strict_artifact_payloads();
        let duplicate_layout_artifact = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("duplicate artifact serializes");

        let result = Predictor::from_artifact_bytes(&duplicate_layout_artifact);
        match result {
            Err(PredictorError::ContractViolation(message)) => {
                let parsed =
                    alloygbm_core::deserialize_model_artifact_v1(&duplicate_layout_artifact)
                        .expect("artifact parses");
                let report = alloygbm_core::required_section_compatibility_report(&parsed.sections);
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(report, true)
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    #[test]
    fn predictor_rejects_non_legacy_missing_predictor_layout_section() {
        let (metadata, trees_payload, _) = strict_artifact_payloads();
        let non_legacy_missing_layout = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::ShapAux, vec![9_u8]),
            ],
        )
        .expect("artifact serializes");

        assert!(matches!(
            Predictor::from_artifact_bytes(&non_legacy_missing_layout),
            Err(PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn predictor_rejects_missing_trees_section() {
        let (metadata, _, layout_payload) = strict_artifact_payloads();
        let missing_trees = serialize_model_artifact_v1(
            &metadata,
            &[(ModelSectionKind::PredictorLayout, layout_payload)],
        )
        .expect("artifact serializes");

        let result = Predictor::from_artifact_bytes(&missing_trees);
        match result {
            Err(PredictorError::ContractViolation(message)) => {
                let parsed =
                    alloygbm_core::deserialize_model_artifact_v1(&missing_trees).expect("parses");
                let report = alloygbm_core::required_section_compatibility_report(&parsed.sections);
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(report, true)
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    #[test]
    fn predictor_rejects_excessive_local_node_id() {
        let excessive_local_node_id = 65_536;
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 1,
            stumps: vec![TrainedStump::new_unweighted(
                SplitCandidate {
                    node_id: excessive_local_node_id,
                    feature_index: 0,
                    threshold_bin: 0,
                    gain: 1.0,
                    default_left: true,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: -1.0,
                        hess_sum: 1.0,
                        grad_sq_sum: 1.0,
                        row_count: 1,
                    },
                    right_stats: NodeStats {
                        grad_sum: 1.0,
                        hess_sum: 1.0,
                        grad_sq_sum: 1.0,
                        row_count: 1,
                    },
                },
                LeafValue::Scalar(-0.1),
                LeafValue::Scalar(0.1),
            )],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
            neutralization_metadata: None,
        };
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        let result = Predictor::from_artifact_bytes(&artifact);

        match result {
            Err(PredictorError::ContractViolation(message)) => {
                assert!(
                    message.contains("local node_id 65536"),
                    "unexpected error message: {message}"
                );
                assert!(
                    message.contains("exceeds predictor tree node slot limit"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    #[test]
    fn predictor_row_rejects_feature_count_mismatch() {
        let (engine_model, _) = train_engine_model();
        let artifact = engine_model
            .to_artifact_bytes()
            .expect("artifact serializes");
        let predictor = Predictor::from_artifact_bytes(&artifact).expect("artifact parses");
        let result = predictor.predict_row(&[1.0]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }

    #[test]
    fn batch_rejects_empty_rows() {
        let pred = predictor_stub();
        let result = pred.predict_batch(&[]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }

    #[test]
    fn row_stub_kept_as_alias() {
        let pred = predictor_stub();
        let result = pred.predict_row_stub(&[1.0, 2.0]);
        assert!(matches!(result, Err(PredictorError::InvalidInput(_))));
    }

    #[test]
    fn batch_parallelization_policy_requires_sufficient_workload() {
        assert!(!should_parallel_predict_batch(
            PARALLEL_PREDICT_MIN_ROWS - 1,
            100
        ));
        assert!(!should_parallel_predict_batch(PARALLEL_PREDICT_MIN_ROWS, 1));
        assert!(should_parallel_predict_batch(PARALLEL_PREDICT_MIN_ROWS, 64));
        assert!(should_parallel_predict_batch(4_096, 4));
    }

    // -- Multi-class tests ---------------------------------------------------

    fn multiclass_fixture_dataset() -> TrainingDataset {
        // 9 samples, 2 features, 3 classes (0, 1, 2)
        // Class 0: feature[0] < 1
        // Class 1: feature[0] in [1, 2)
        // Class 2: feature[0] >= 2
        TrainingDataset {
            matrix: DatasetMatrix::new(
                9,
                2,
                vec![
                    0.0, 0.5, //
                    0.3, 0.8, //
                    0.5, 0.2, //
                    1.0, 0.3, //
                    1.3, 0.7, //
                    1.7, 0.4, //
                    2.0, 0.6, //
                    2.5, 0.1, //
                    2.8, 0.9, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn multiclass_fixture_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            9,
            2,
            9,
            vec![
                0, 2, //
                1, 4, //
                2, 1, //
                3, 1, //
                4, 3, //
                5, 2, //
                6, 3, //
                7, 0, //
                8, 4, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn train_multiclass_artifact() -> Vec<u8> {
        use alloygbm_engine::{MultiClassSoftmaxObjective, Trainer};

        let dataset = multiclass_fixture_dataset();
        let binned_matrix = multiclass_fixture_binned_matrix();
        let params = fixture_params();
        let trainer = Trainer::new(params).unwrap();
        let backend = CpuBackend;
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();

        let controls = trainer
            .iteration_controls_for_policy(
                &dataset,
                &binned_matrix,
                5,
                alloygbm_engine::TrainingPolicyMode::Manual,
            )
            .unwrap();

        let summary = trainer
            .fit_multiclass_iterations_with_summary(
                &dataset,
                &binned_matrix,
                &backend,
                &obj,
                controls,
            )
            .unwrap();

        summary.model.to_artifact_bytes().unwrap()
    }

    #[test]
    fn test_multiclass_predictor_from_artifact() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        assert!(predictor.is_multiclass());
        assert_eq!(predictor.num_classes(), Some(3));
    }

    #[test]
    fn test_multiclass_predict_batch_shape() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        let rows = vec![vec![0.2, 0.5], vec![1.5, 0.3], vec![2.7, 0.8]];
        let result = predictor.predict_batch_multiclass(&rows).unwrap();
        // 3 rows * 3 classes = 9 values
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_multiclass_predict_softmax_sums_to_one() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        let rows = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![0.5, 0.3],
            vec![1.5, 0.7],
        ];
        let result = predictor.predict_batch_multiclass(&rows).unwrap();
        assert_eq!(result.len(), 15); // 5 * 3

        for row_idx in 0..5 {
            let start = row_idx * 3;
            let row_sum: f32 = result[start..start + 3].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {row_idx} probabilities sum to {row_sum}, expected 1.0"
            );
            // All probabilities should be in [0, 1]
            for &p in &result[start..start + 3] {
                assert!(
                    (0.0..=1.0).contains(&p),
                    "probability {p} out of [0,1] range"
                );
            }
        }
    }

    #[test]
    fn test_multiclass_predict_dense_matches_batch() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        let rows = vec![vec![0.2, 0.5], vec![1.5, 0.3], vec![2.7, 0.8]];
        let batch_result = predictor.predict_batch_multiclass(&rows).unwrap();
        let dense_values: Vec<f32> = rows.iter().flatten().copied().collect();
        let dense_result = predictor
            .predict_batch_dense_multiclass(&dense_values, 3, 2)
            .unwrap();
        assert_eq!(batch_result.len(), dense_result.len());
        for (a, b) in batch_result.iter().zip(dense_result.iter()) {
            assert!((a - b).abs() < 1e-6, "batch {a} != dense {b}");
        }
    }

    #[test]
    fn test_multiclass_predict_row_errors_on_multiclass() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        let result = predictor.predict_row(&[0.5, 0.3]);
        assert!(result.is_err());
        if let Err(PredictorError::ContractViolation(msg)) = result {
            assert!(
                msg.contains("multiclass") || msg.contains("multi-class"),
                "unexpected error: {msg}"
            );
        } else {
            panic!(
                "expected ContractViolation error for predict_row on multiclass, got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_multiclass_predict_batch_errors_on_multiclass() {
        let artifact = train_multiclass_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();
        let result = predictor.predict_batch(&[vec![0.5, 0.3]]);
        assert!(result.is_err());
    }

    /// Build an artifact with one categorical split (feature 0: cats 0,1 left → leaf -0.1,
    /// cats 2,3 right → leaf 0.1) and baseline 0.5.
    fn build_categorical_artifact() -> Vec<u8> {
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![TrainedStump {
                split: SplitCandidate {
                    node_id: 0,
                    feature_index: 0,
                    threshold_bin: 0,
                    gain: 2.0,
                    default_left: true,
                    is_categorical: true,
                    categorical_bitset: Some(vec![0b0000_0011]), // cats 0,1 go left
                    left_stats: NodeStats {
                        grad_sum: -1.0,
                        hess_sum: 2.0,
                        grad_sq_sum: 0.0,
                        row_count: 10,
                    },
                    right_stats: NodeStats {
                        grad_sum: 1.0,
                        hess_sum: 2.0,
                        grad_sq_sum: 0.0,
                        row_count: 10,
                    },
                },
                left_leaf_value: LeafValue::Scalar(-0.1),
                right_leaf_value: LeafValue::Scalar(0.1),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            }],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
            neutralization_metadata: None,
        };
        model.to_artifact_bytes().expect("serialize should succeed")
    }

    #[test]
    fn test_predict_categorical_split() {
        let artifact = build_categorical_artifact();
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();

        // Category 0 → left → baseline + left_leaf_value = 0.5 + (-0.1) = 0.4
        let pred0 = predictor.predict_row(&[0.0, 99.0]).unwrap();
        assert!(
            (pred0 - 0.4).abs() < 1e-6,
            "cat 0 should go left: got {pred0}"
        );

        // Category 1 → left → 0.4
        let pred1 = predictor.predict_row(&[1.0, 99.0]).unwrap();
        assert!(
            (pred1 - 0.4).abs() < 1e-6,
            "cat 1 should go left: got {pred1}"
        );

        // Category 2 → right → baseline + right_leaf_value = 0.5 + 0.1 = 0.6
        let pred2 = predictor.predict_row(&[2.0, 99.0]).unwrap();
        assert!(
            (pred2 - 0.6).abs() < 1e-6,
            "cat 2 should go right: got {pred2}"
        );

        // Category 3 → right → 0.6
        let pred3 = predictor.predict_row(&[3.0, 99.0]).unwrap();
        assert!(
            (pred3 - 0.6).abs() < 1e-6,
            "cat 3 should go right: got {pred3}"
        );

        // NaN → default_left (true) → 0.4
        let pred_nan = predictor.predict_row(&[f32::NAN, 99.0]).unwrap();
        assert!(
            (pred_nan - 0.4).abs() < 1e-6,
            "NaN should go default_left: got {pred_nan}"
        );
    }

    #[test]
    fn test_predict_categorical_and_continuous_mixed() {
        // Build an artifact with one categorical stump (feature 0) and one continuous stump (feature 1).
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 2,
            stumps: vec![
                // Categorical split on feature 0: cats 0,1 left (-0.2), cats 2+ right (+0.2)
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 0,
                        gain: 2.0,
                        default_left: true,
                        is_categorical: true,
                        categorical_bitset: Some(vec![0b0000_0011]),
                        left_stats: NodeStats {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            grad_sq_sum: 0.0,
                            row_count: 10,
                        },
                        right_stats: NodeStats {
                            grad_sum: 1.0,
                            hess_sum: 2.0,
                            grad_sq_sum: 0.0,
                            row_count: 10,
                        },
                    },
                    left_leaf_value: LeafValue::Scalar(-0.2),
                    right_leaf_value: LeafValue::Scalar(0.2),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                },
                // Continuous split on feature 1: threshold_bin 3 (i.e. <=3 left, >3 right)
                // node_id in tree 1 (tree_id=1, local=0 → 1 * 1048576 + 0)
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 1_048_576,
                        feature_index: 1,
                        threshold_bin: 3,
                        gain: 1.5,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.5,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 5,
                        },
                        right_stats: NodeStats {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 5,
                        },
                    },
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(-0.1),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                },
            ],
            categorical_state: None,
            node_debug_stats: None,
            objective: "squared_error".to_string(),
            native_categorical_feature_indices: vec![0],
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
            neutralization_metadata: None,
        };
        let artifact = model.to_artifact_bytes().expect("serialize should succeed");
        let predictor = Predictor::from_artifact_bytes(&artifact).unwrap();

        // Cat 0 (left, -0.2) + continuous 2.0 (<=3, left, +0.1) = -0.1
        let p1 = predictor.predict_row(&[0.0, 2.0]).unwrap();
        assert!((p1 - (-0.1)).abs() < 1e-6, "cat0+cont_left: got {p1}");

        // Cat 0 (left, -0.2) + continuous 5.0 (>3, right, -0.1) = -0.3
        let p2 = predictor.predict_row(&[0.0, 5.0]).unwrap();
        assert!((p2 - (-0.3)).abs() < 1e-6, "cat0+cont_right: got {p2}");

        // Cat 2 (right, +0.2) + continuous 2.0 (left, +0.1) = 0.3
        let p3 = predictor.predict_row(&[2.0, 2.0]).unwrap();
        assert!((p3 - 0.3).abs() < 1e-6, "cat2+cont_left: got {p3}");

        // Cat 2 (right, +0.2) + continuous 5.0 (right, -0.1) = 0.1
        let p4 = predictor.predict_row(&[2.0, 5.0]).unwrap();
        assert!((p4 - 0.1).abs() < 1e-6, "cat2+cont_right: got {p4}");

        // Verify batch prediction matches individual predictions
        let batch = predictor
            .predict_batch(&[
                vec![0.0, 2.0],
                vec![0.0, 5.0],
                vec![2.0, 2.0],
                vec![2.0, 5.0],
            ])
            .unwrap();
        assert!((batch[0] - p1).abs() < 1e-6);
        assert!((batch[1] - p2).abs() < 1e-6);
        assert!((batch[2] - p3).abs() < 1e-6);
        assert!((batch[3] - p4).abs() < 1e-6);
    }

    #[test]
    fn pl_tree_artifact_roundtrip_train_predict_via_predictor() {
        // Train a 2-feature model with leaf_model=Linear, save, reload via Predictor,
        // and verify predictions are bit-for-bit identical.
        let n = 16_usize;
        let fc = 2_usize;
        let float_values: Vec<f32> = (0..n)
            .flat_map(|i| {
                let x0 = i as f32 / (n as f32 - 1.0);
                let x1 = 1.0 - x0;
                [x0, x1]
            })
            .collect();
        let targets: Vec<f32> = (0..n)
            .map(|i| float_values[i * fc] * 1.5 - float_values[i * fc + 1] * 0.8)
            .collect();

        let dataset = TrainingDataset {
            matrix: DatasetMatrix::new(n, fc, float_values.clone()).expect("matrix ok"),
            targets: targets.clone(),
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        };

        // Build a 4-bin binned matrix.
        let bins: Vec<u8> = (0..n)
            .flat_map(|i| {
                let b0 = (i * 4 / n) as u8;
                let b1 = 3_u8.saturating_sub(b0);
                [b0, b1]
            })
            .collect();
        let binned = BinnedMatrix::new(n, fc, 4, bins).expect("binned ok");

        let params = TrainParams {
            leaf_model: LeafModelKind::Linear,
            ..Default::default()
        };

        let trainer = Trainer::new(params).expect("params valid");
        let engine_model = trainer
            .fit_iterations(&dataset, &binned, &CpuBackend, &SquaredErrorObjective, 4)
            .expect("training succeeds");

        // Save via engine, reload via Predictor.
        let bytes = engine_model.to_artifact_bytes().expect("serializes");
        let predictor = Predictor::from_artifact_bytes(&bytes).expect("Predictor loads");

        // Verify LinearLeafCoefficients section is present.
        let parsed = deserialize_model_artifact_v1(&bytes).expect("parses");
        assert!(
            parsed
                .sections
                .iter()
                .any(|s| s.descriptor.kind == ModelSectionKind::LinearLeafCoefficients),
            "LinearLeafCoefficients section missing"
        );

        // Predictions must match between engine model and Predictor.
        let test_rows = [[0.0f32, 1.0], [0.333, 0.667], [0.667, 0.333], [1.0, 0.0]];
        for row in &test_rows {
            let engine_pred = engine_model
                .predict_batch(&[row.to_vec()])
                .expect("engine predicts")[0];
            let pred_pred = predictor.predict_row(row).expect("predictor predicts");
            assert!(
                (engine_pred - pred_pred).abs() < 1e-4,
                "mismatch for row {:?}: engine={engine_pred} predictor={pred_pred}",
                row
            );
        }
    }
}
