#![allow(clippy::needless_range_loop)]

use crate::binning::{ADDITIVITY_ATOL, ADDITIVITY_RTOL, additivity_tolerance};
use crate::brute_force::{
    compute_subset_expectations, factorial_table, local_path_predict, shapley_values_for_row_f64,
};
use crate::types::build_model_structure;
use crate::*;
use alloygbm_core::{
    Device, LeafValue, LinearLeaf, ModelMetadata, ModelSectionKind, NodeStats, SplitCandidate,
    deserialize_model_artifact_v1, serialize_model_artifact_v1,
};
use alloygbm_engine::{TrainedModel, TrainedStump};
use alloygbm_predictor::Predictor;

fn sample_metadata(feature_names: &[&str]) -> ModelMetadata {
    ModelMetadata {
        format_version: 1,
        feature_names: feature_names
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        trained_device: Device::Cpu,
        objective: "squared_error".to_string(),
        num_classes: None,
    }
}

fn split(node_id: u32, feature_index: u32, threshold_bin: u16) -> SplitCandidate {
    SplitCandidate {
        node_id,
        feature_index,
        threshold_bin,
        gain: 1.0,
        default_left: false,
        is_categorical: false,
        categorical_bitset: None,
        left_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 1.0,
            grad_sq_sum: 0.0,
            row_count: 1,
        },
        right_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: 1.0,
            grad_sq_sum: 0.0,
            row_count: 1,
        },
    }
}

fn split_with_counts(
    node_id: u32,
    feature_index: u32,
    threshold_bin: u16,
    left_count: u32,
    right_count: u32,
) -> SplitCandidate {
    SplitCandidate {
        node_id,
        feature_index,
        threshold_bin,
        gain: 1.0,
        default_left: false,
        is_categorical: false,
        categorical_bitset: None,
        left_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: left_count as f32,
            grad_sq_sum: 0.0,
            row_count: left_count,
        },
        right_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: right_count as f32,
            grad_sq_sum: 0.0,
            row_count: right_count,
        },
    }
}

fn fixture_model() -> TrainedModel {
    TrainedModel {
        baseline_prediction: 0.5,
        feature_count: 2,
        stumps: vec![
            TrainedStump {
                split: split(0, 0, 1),
                left_leaf_value: LeafValue::Scalar(1.0),
                right_leaf_value: LeafValue::Scalar(2.0),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split(1, 1, 0),
                left_leaf_value: LeafValue::Scalar(0.1),
                right_leaf_value: LeafValue::Scalar(0.2),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split(2, 1, 1),
                left_leaf_value: LeafValue::Scalar(0.3),
                right_leaf_value: LeafValue::Scalar(0.4),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    }
}

fn fixture_model_with_unused_feature() -> TrainedModel {
    TrainedModel {
        baseline_prediction: 0.5,
        feature_count: 3,
        stumps: fixture_model().stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    }
}

fn fixture_rows() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![0.0, 2.0],
        vec![3.0, 0.0],
        vec![3.0, 2.0],
    ]
}

fn fixture_trees_payload() -> Vec<u8> {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let parsed = deserialize_model_artifact_v1(&artifact).expect("artifact parses");
    parsed
        .sections
        .iter()
        .find(|section| section.descriptor.kind == ModelSectionKind::Trees)
        .map(|section| section.payload.clone())
        .expect("trees payload exists")
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= ADDITIVITY_ATOL,
        "expected {expected}, got {actual}"
    );
}

fn fixture_model_with_dart_weights() -> TrainedModel {
    // Mirror `fixture_model` but stamp a non-unit tree_weight so the
    // SHAP pre-scaling path in `explain_rows_from_model` is
    // exercised. In real DART artifacts every stump in a tree
    // shares the same tree_weight; `fixture_model`'s three stumps
    // all encode tree_id=0 (raw node_ids 0/1/2 with the default
    // TREE_NODE_STRIDE), so a uniform weight here matches that
    // invariant.
    let mut model = fixture_model();
    for stump in model.stumps.iter_mut() {
        stump.tree_weight = 0.25;
    }
    model
}

#[test]
fn shap_additivity_holds_on_dart_artifact() {
    // Regression test for the v0.9.0 PR review (#5): SHAP must
    // apply per-stump tree_weight so the sum of contributions plus
    // expected_value reconstructs the predictor's output. Pre-fix,
    // SHAP summed unweighted leaf values and additivity broke for
    // any DART artifact with non-1.0 weights.
    let model = fixture_model_with_dart_weights();
    let rows = fixture_rows();
    let explanation = explain_rows_from_model(&model, &rows, None).expect("explain succeeds");
    for (row_idx, row) in rows.iter().enumerate() {
        let predicted = model.predict_row(row).expect("predict_row");
        let reconstructed: f32 =
            explanation.expected_value + explanation.values[row_idx].iter().sum::<f32>();
        let tol = ADDITIVITY_ATOL + ADDITIVITY_RTOL * predicted.abs();
        assert!(
            (predicted - reconstructed).abs() <= tol,
            "row {row_idx}: predict={predicted}, expected+sum(shap)={reconstructed}, tol={tol}"
        );
    }
}

#[test]
fn explain_rows_from_artifact_rejects_empty_rows() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let result = explain_rows_from_artifact_bytes(&artifact, &[]);
    assert!(matches!(result, Err(ShapError::InvalidInput(_))));
}

#[test]
fn explain_rows_from_artifact_rejects_feature_count_mismatch() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let result = explain_rows_from_artifact_bytes(&artifact, &[vec![0.0]]);
    assert!(matches!(result, Err(ShapError::InvalidInput(_))));
}

#[test]
fn explain_rows_from_artifact_rejects_non_finite_features() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let result = explain_rows_from_artifact_bytes(&artifact, &[vec![f32::NAN, 0.0]]);
    assert!(matches!(result, Err(ShapError::InvalidInput(_))));
}

#[test]
fn explain_rows_from_artifact_rejects_incompatible_required_sections() {
    let layout_payload = {
        let strict_artifact = fixture_model()
            .to_artifact_bytes()
            .expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
        parsed
            .sections
            .iter()
            .find(|section| section.descriptor.kind == ModelSectionKind::PredictorLayout)
            .map(|section| section.payload.clone())
            .expect("predictor layout payload exists")
    };

    let incompatible_artifact = serialize_model_artifact_v1(
        &sample_metadata(&["f0", "f1"]),
        &[(ModelSectionKind::PredictorLayout, layout_payload)],
    )
    .expect("artifact serializes");

    let result = explain_rows_from_artifact_bytes(&incompatible_artifact, &[vec![0.0, 0.0]]);
    assert!(matches!(result, Err(ShapError::ContractViolation(_))));
}

#[test]
fn explain_rows_from_artifact_accepts_legacy_trees_only_artifact() {
    let legacy_artifact = serialize_model_artifact_v1(
        &sample_metadata(&["f0", "f1"]),
        &[(ModelSectionKind::Trees, fixture_trees_payload())],
    )
    .expect("artifact serializes");

    let explanation = explain_rows_from_artifact_bytes(&legacy_artifact, &fixture_rows())
        .expect("legacy artifact explains");
    assert_close(explanation.expected_value, 2.25);
    assert_eq!(explanation.values.len(), 4);
    assert_eq!(explanation.values[0].len(), 2);
}

#[test]
fn explain_rows_from_artifact_rejects_duplicate_trees_sections() {
    let trees_payload = fixture_trees_payload();
    let duplicate_trees_artifact = serialize_model_artifact_v1(
        &sample_metadata(&["f0", "f1"]),
        &[
            (ModelSectionKind::Trees, trees_payload.clone()),
            (ModelSectionKind::Trees, trees_payload),
        ],
    )
    .expect("artifact serializes");

    let result = explain_rows_from_artifact_bytes(&duplicate_trees_artifact, &[vec![0.0, 0.0]]);
    assert!(matches!(result, Err(ShapError::ContractViolation(_))));
}

#[test]
fn explain_rows_from_artifact_rejects_metadata_feature_count_mismatch() {
    let mismatched_artifact = serialize_model_artifact_v1(
        &sample_metadata(&["f0", "f1", "f2"]),
        &[(ModelSectionKind::Trees, fixture_trees_payload())],
    )
    .expect("artifact serializes");

    let result = explain_rows_from_artifact_bytes(&mismatched_artifact, &[vec![0.0, 0.0, 0.0]]);
    assert!(matches!(result, Err(ShapError::ContractViolation(_))));
}

#[test]
fn explain_rows_from_artifact_computes_exact_expected_value_and_contributions() {
    let model = fixture_model();
    let artifact = model.to_artifact_bytes().expect("artifact serializes");
    let rows = fixture_rows();

    let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
    assert_close(explanation.expected_value, 2.25);
    assert_eq!(explanation.values.len(), rows.len());
    for row_values in &explanation.values {
        assert_eq!(row_values.len(), model.feature_count);
    }

    let expected_values = [
        vec![-0.6, -0.05],
        vec![-0.6, 0.05],
        vec![0.6, -0.05],
        vec![0.6, 0.05],
    ];

    for (actual_row, expected_row) in explanation.values.iter().zip(expected_values.iter()) {
        for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
            assert_close(*actual, *expected);
        }
    }

    for (row, values) in rows.iter().zip(explanation.values.iter()) {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn explain_rows_from_artifact_matches_predictor_predictions() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor loads");
    let rows = fixture_rows();

    let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
    for (row_index, row) in rows.iter().enumerate() {
        let predicted = predictor.predict_row(row).expect("predicts");
        let reconstructed =
            explanation.expected_value + explanation.values[row_index].iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn binning_context_linear_matches_predictor_conversion() {
    // The float threshold SHAP computes must equal the float
    // threshold the predictor would compute via
    // `convert_bin_thresholds_to_float`.  Spot-check a few bins.
    let ctx = BinningContext::Linear {
        feature_mins: vec![-2.0, 0.0],
        feature_maxs: vec![3.0, 10.0],
        max_data_bin: 254,
    };
    // Predictor formula: min + ((bin + 0.5) / 254) * (max - min).
    for &bin in &[0u16, 1, 64, 127, 254] {
        let shap_thr_f0 = ctx.float_threshold(0, bin);
        let expected_f0 = -2.0 + ((bin as f32 + 0.5) / 254.0) * 5.0;
        assert!((shap_thr_f0 - expected_f0).abs() < 1e-6);
        let shap_thr_f1 = ctx.float_threshold(1, bin);
        let expected_f1 = 0.0 + ((bin as f32 + 0.5) / 254.0) * 10.0;
        assert!((shap_thr_f1 - expected_f1).abs() < 1e-6);
    }
}

#[test]
fn binning_context_prebinned_matches_predictor_conversion() {
    let ctx = BinningContext::PreBinned;
    for &bin in &[0u16, 1, 64, 127, 254] {
        // Predictor pre-binned: float threshold = bin + 0.5.
        assert!((ctx.float_threshold(0, bin) - (bin as f32 + 0.5)).abs() < 1e-6);
    }
}

#[test]
fn binning_context_quantile_matches_predictor_conversion() {
    let ctx = BinningContext::Quantile {
        feature_cuts: vec![vec![0.1, 0.5, 0.9], vec![1.0, 2.0, 3.0, 4.0]],
    };
    assert!((ctx.float_threshold(0, 0) - 0.1).abs() < 1e-6);
    assert!((ctx.float_threshold(0, 2) - 0.9).abs() < 1e-6);
    // Past the last cut → f32::MAX.
    assert_eq!(ctx.float_threshold(0, 3), f32::MAX);
    assert_eq!(ctx.float_threshold(1, 4), f32::MAX);
}

#[test]
fn binning_context_linear_rank_inverts_rank_mapping() {
    // For a rank-flagged feature with 5 unique sorted values and
    // max_data_bin = 4, the rank-to-bin formula is bin = round(rank).
    // Threshold conversion: float_threshold(bin) = sorted[r*] where
    // r* = ceil((bin + 0.5) * (N-1) / max_data_bin).
    let sorted = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let ctx = BinningContext::LinearRank {
        per_feature: vec![Some(sorted.clone())],
        feature_mins: vec![1.0],
        feature_maxs: vec![5.0],
        max_data_bin: 4,
    };
    // r* = ceil(0.5 * 4 / 4) = ceil(0.5) = 1 → sorted[1] = 2.0
    assert!((ctx.float_threshold(0, 0) - 2.0).abs() < 1e-6);
    // r* = ceil(1.5 * 4 / 4) = ceil(1.5) = 2 → sorted[2] = 3.0
    assert!((ctx.float_threshold(0, 1) - 3.0).abs() < 1e-6);
    // r* = ceil(2.5 * 4 / 4) = ceil(2.5) = 3 → sorted[3] = 4.0
    assert!((ctx.float_threshold(0, 2) - 4.0).abs() < 1e-6);
    // r* = ceil(3.5 * 4 / 4) = ceil(3.5) = 4 → sorted[4] = 5.0
    assert!((ctx.float_threshold(0, 3) - 5.0).abs() < 1e-6);
    // Bin past the data range clamps to the last sorted value.
    assert!((ctx.float_threshold(0, 4) - 5.0).abs() < 1e-6);
}

#[test]
fn binning_context_linear_rank_falls_back_to_linear_for_non_rank_features() {
    // Feature 0 uses rank binning, feature 1 falls back to standard
    // linear (per_feature[1] is None).
    let sorted = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
    let ctx = BinningContext::LinearRank {
        per_feature: vec![Some(sorted), None],
        feature_mins: vec![0.0, -2.0],
        feature_maxs: vec![4.0, 3.0],
        max_data_bin: 254,
    };
    // Feature 1 (None) must match the existing Linear formula
    // exactly.
    for &bin in &[0u16, 1, 64, 127, 254] {
        let got = ctx.float_threshold(1, bin);
        let expected = -2.0 + ((bin as f32 + 0.5) / 254.0) * 5.0;
        assert!(
            (got - expected).abs() < 1e-6,
            "bin {bin}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn binning_context_linear_rank_matches_predictor_on_round_trip() {
    // Generate a small sorted-values fixture; for each unique value
    // compute the bin via the predictor's rank-quantize formula,
    // then convert the bin back to a float via float_threshold,
    // and assert the predictor's `value < float_threshold` decision
    // matches the integer-bin comparison `quantized_bin <= bin - 1`.
    let sorted: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();
    let max_data_bin: u16 = 8;
    let ctx = BinningContext::LinearRank {
        per_feature: vec![Some(sorted.clone())],
        feature_mins: vec![sorted[0]],
        feature_maxs: vec![*sorted.last().unwrap()],
        max_data_bin,
    };
    let n = sorted.len();
    // Mimic quantize_rank_value_wide(value, sorted, max_data_bin).
    let quantize = |value: f32| -> u16 {
        let insertion = sorted.partition_point(|probe| *probe <= value);
        let rank = insertion.saturating_sub(1).min(n - 1);
        let scaled = (rank as f32 * max_data_bin as f32) / (n - 1) as f32;
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5).floor() as i32
        } else {
            (scaled - 0.5).ceil() as i32
        };
        rounded.clamp(0, max_data_bin as i32) as u16
    };
    for &threshold_bin in &[0u16, 1, 3, 4, 6, 7] {
        let float_threshold = ctx.float_threshold(0, threshold_bin);
        // Every sorted value should agree on side: the predictor's
        // bin comparison `quantize(v) <= threshold_bin` must equal
        // SHAP's `v < float_threshold`.
        for &value in &sorted {
            let predictor_left = quantize(value) <= threshold_bin;
            let shap_left = value < float_threshold;
            assert_eq!(
                predictor_left,
                shap_left,
                "threshold_bin={threshold_bin}, value={value}, float_threshold={float_threshold}, quantize={}",
                quantize(value),
            );
        }
    }
}

#[test]
fn binning_context_explanation_matches_predictor_on_constant_leaves() {
    // Build a simple two-tree constant-leaf model with bin-index
    // thresholds.  SHAP without binning would compare raw values
    // against bin indices and reach a different leaf than the
    // predictor (which uses float thresholds).  With binning, SHAP
    // must reach the same leaf and produce strict additivity.
    let model = fixture_model();
    let artifact = model.to_artifact_bytes().expect("serializes");

    // Feature mins/maxs that put raw input values within the
    // float-threshold-converted decision region.  fixture_model
    // splits on feature 0 at bin 2 and feature 1 at bin 1.
    let binning = BinningContext::Linear {
        feature_mins: vec![0.0, 0.0],
        feature_maxs: vec![10.0, 10.0],
        max_data_bin: 254,
    };
    let rows = vec![
        vec![0.05_f32, 0.05_f32], // both below thresholds
        vec![5.0_f32, 5.0_f32],   // both above
    ];

    let explanation = explain_rows_from_artifact_bytes_with_binning(&artifact, &rows, &binning)
        .expect("with-binning explains");
    // Additivity check inside explain enforces strict tolerance
    // when binning is provided and leaves are scalar — if this
    // returns Ok the path walker matched the predictor's path.
    for (row_index, row) in rows.iter().enumerate() {
        let reconstructed =
            explanation.expected_value + explanation.values[row_index].iter().sum::<f32>();
        // Validate against a hand walk via local_path_predict —
        // same code path the verify_additivity call uses internally.
        let predicted = local_path_predict(&model, row, Some(&binning));
        assert!((reconstructed - predicted).abs() < 1e-4);
    }
}

#[test]
fn explain_rows_from_artifact_assigns_zero_to_unused_features() {
    let model = fixture_model_with_unused_feature();
    let artifact = model.to_artifact_bytes().expect("artifact serializes");
    let rows = vec![vec![0.0, 0.0, 5.0], vec![3.0, 2.0, 9.0]];

    let explanation = explain_rows_from_artifact_bytes(&artifact, &rows).expect("explains");
    assert_eq!(explanation.values[0].len(), 3);
    assert_close(explanation.values[0][2], 0.0);
    assert_close(explanation.values[1][2], 0.0);
}

#[test]
fn global_importance_aggregates_mean_absolute_contribution() {
    let feature_names = vec!["f0".to_string(), "f1".to_string()];
    let shap_values = vec![
        vec![-0.6, -0.05],
        vec![-0.6, 0.05],
        vec![0.6, -0.05],
        vec![0.6, 0.05],
    ];

    let global = global_importance_from_shap_values(&feature_names, &shap_values)
        .expect("global importance computes");
    assert_close(global[0].1, 0.6);
    assert_close(global[1].1, 0.05);
    assert_eq!(global[0].0, "f0");
    assert_eq!(global[1].0, "f1");
}

#[test]
fn global_importance_from_artifact_uses_metadata_feature_names() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let global =
        global_importance_from_artifact_bytes(&artifact, &fixture_rows()).expect("global computes");

    assert_eq!(global.len(), 2);
    assert_eq!(global[0].0, "f0");
    assert_eq!(global[1].0, "f1");
}

#[test]
fn global_importance_breaks_ties_by_feature_name() {
    let feature_names = vec!["zeta".to_string(), "alpha".to_string(), "beta".to_string()];
    let shap_values = vec![vec![1.0, -1.0, 0.0], vec![-1.0, 1.0, 0.0]];

    let global = global_importance_from_shap_values(&feature_names, &shap_values)
        .expect("global importance computes");

    assert_eq!(global.len(), 3);
    assert_eq!(global[0].0, "alpha");
    assert_eq!(global[1].0, "zeta");
    assert_eq!(global[2].0, "beta");
    assert_close(global[0].1, 1.0);
    assert_close(global[1].1, 1.0);
    assert_close(global[2].1, 0.0);
}

#[test]
fn legacy_stub_helpers_return_deterministic_outputs() {
    let metadata = sample_metadata(&["f0", "f1"]);
    let rows = fixture_rows();
    let shap_values = shap_values_stub(&metadata, &rows).expect("stub values compute");
    assert_eq!(shap_values.len(), rows.len());
    assert_eq!(shap_values[0], vec![0.0, 0.0]);

    let global =
        global_importance_stub(&metadata, &metadata.feature_names).expect("stub global computes");
    assert_eq!(
        global,
        vec![("f0".to_string(), 0.0), ("f1".to_string(), 0.0)]
    );
}

// -------------------------------------------------------------------
// TreeSHAP tests
// -------------------------------------------------------------------

#[test]
fn tree_shap_matches_brute_force_on_fixture_model() {
    let model = fixture_model();
    let rows = fixture_rows();

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    assert_close(brute_force.expected_value, tree_shap.expected_value);
    assert_eq!(brute_force.values.len(), tree_shap.values.len());

    for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
        assert_eq!(bf_row.len(), ts_row.len());
        for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }
}

#[test]
fn tree_shap_matches_brute_force_on_unused_feature_model() {
    let model = fixture_model_with_unused_feature();
    let rows = vec![vec![0.0, 0.0, 5.0], vec![3.0, 2.0, 9.0]];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
        for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }
    // Feature 2 should be zero in both.
    assert_close(tree_shap.values[0][2], 0.0);
    assert_close(tree_shap.values[1][2], 0.0);
}

#[test]
fn tree_shap_additivity_holds_for_all_rows() {
    let model = fixture_model();
    let rows = fixture_rows();
    let explanation = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (row, values) in rows.iter().zip(explanation.values.iter()) {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn tree_shap_single_stump_model() {
    // A single-tree, single-node (depth-1) model splitting on feature 0.
    let model = TrainedModel {
        baseline_prediction: 1.0,
        feature_count: 2,
        stumps: vec![TrainedStump {
            split: split_with_counts(0, 0, 5, 3, 7),
            left_leaf_value: LeafValue::Scalar(-0.5),
            right_leaf_value: LeafValue::Scalar(0.3),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };

    let rows = vec![vec![3.0, 0.0], vec![8.0, 0.0]];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (bf_row, ts_row) in brute_force.values.iter().zip(tree_shap.values.iter()) {
        for (bf_val, ts_val) in bf_row.iter().zip(ts_row.iter()) {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }
}

#[test]
fn tree_shap_multi_tree_model() {
    // Two trees, each with depth 1, splitting on different features.
    let stride = 1u32 << 20;
    let model = TrainedModel {
        baseline_prediction: 0.0,
        feature_count: 3,
        stumps: vec![
            TrainedStump {
                split: split_with_counts(0, 0, 5, 4, 6),
                left_leaf_value: LeafValue::Scalar(1.0),
                right_leaf_value: LeafValue::Scalar(-1.0),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(stride, 1, 3, 5, 5),
                left_leaf_value: LeafValue::Scalar(0.5),
                right_leaf_value: LeafValue::Scalar(-0.5),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };

    let rows = vec![
        vec![3.0, 1.0, 0.0],
        vec![8.0, 5.0, 0.0],
        vec![3.0, 5.0, 0.0],
    ];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (row_idx, (bf_row, ts_row)) in brute_force
        .values
        .iter()
        .zip(tree_shap.values.iter())
        .enumerate()
    {
        for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }
}

/// Regression test for asymmetric-depth TreeSHAP attribution
/// (added while reviewing PR #27).
///
/// `explain_rows_tree_shap` must match the brute-force exact Shapley
/// path even when leaves are at varying depths (the common case for
/// any real model with `min_data_in_leaf`, `min_split_gain`, or
/// early-stop).  This test holds today on the minimal asymmetric
/// topology but the polynomial path has a separate, pre-existing
/// additivity drift on much larger / deeper variable-depth trees
/// (see Limitation 5 in `docs/limitations.md`).
#[test]
fn tree_shap_asymmetric_depth_tree_matches_brute_force_and_predict() {
    // Stumps:
    //   id 0 (root):         feat 0, threshold 1, leaves {1.0, 2.0}, counts l=80 r=20
    //   id 1 (left child):   feat 1, threshold 2, leaves {3.0, 4.0}, counts l=50 r=30 (sum=80)
    //   id 2 (right child):  DOES NOT EXIST — depth-1 early-stop on the right
    let model = TrainedModel {
        baseline_prediction: 0.5,
        feature_count: 2,
        stumps: vec![
            TrainedStump {
                split: split_with_counts(0, 0, 1, 80, 20),
                left_leaf_value: LeafValue::Scalar(1.0),
                right_leaf_value: LeafValue::Scalar(2.0),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(1, 1, 2, 50, 30),
                left_leaf_value: LeafValue::Scalar(3.0),
                right_leaf_value: LeafValue::Scalar(4.0),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };

    let rows = vec![
        vec![0.0, 0.0], // L at root → L at stump 1 → adds 1.0 + 3.0 → predict = 4.5
        vec![0.0, 5.0], // L at root → R at stump 1 → adds 1.0 + 4.0 → predict = 5.5
        vec![5.0, 0.0], // R at root → stump 2 missing → adds 2.0 → predict = 2.5
        vec![5.0, 5.0], // R at root → stump 2 missing → adds 2.0 → predict = 2.5
    ];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    // Brute-force gives the reference (exact 2^N Shapley).
    for (row_idx, (bf_row, ts_row)) in brute_force
        .values
        .iter()
        .zip(tree_shap.values.iter())
        .enumerate()
    {
        for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
            assert!(
                (bf_val - ts_val).abs() <= 1e-5,
                "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val} \
                 (this asymmetric-depth tree is what TreeSHAP gets wrong without the v0.7.4 fix)"
            );
        }
    }

    // Independent additivity check against TrainedModel::predict_row.
    for (row_idx, (row, ts_values)) in rows.iter().zip(tree_shap.values.iter()).enumerate() {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = tree_shap.expected_value + ts_values.iter().sum::<f32>();
        assert!(
            (predicted - reconstructed).abs() <= 1e-5,
            "row {row_idx}: predict_row={predicted}, expected_value+Σphi={reconstructed}, \
             gap={}",
            (predicted - reconstructed).abs()
        );
    }
}

/// Spine-tree reproducer: every level only goes deeper on the left
/// (stumps at 0, 1, 3, 7), missing all right-side and inner descendant
/// stumps.  Rows reach leaves at depths 1, 2, 3, 4 depending on where
/// they branch off the spine.  This is the topology most real models
/// produce when one branch is dominant and others early-stop.
#[test]
fn tree_shap_spine_tree_matches_brute_force() {
    let model = TrainedModel {
        baseline_prediction: 0.0,
        feature_count: 4,
        stumps: vec![
            TrainedStump {
                split: split_with_counts(0, 0, 1, 70, 30),
                left_leaf_value: LeafValue::Scalar(0.1),
                right_leaf_value: LeafValue::Scalar(0.2),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(1, 1, 1, 50, 20),
                left_leaf_value: LeafValue::Scalar(0.3),
                right_leaf_value: LeafValue::Scalar(0.4),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(3, 2, 1, 30, 20),
                left_leaf_value: LeafValue::Scalar(0.5),
                right_leaf_value: LeafValue::Scalar(0.6),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(7, 3, 1, 20, 10),
                left_leaf_value: LeafValue::Scalar(0.7),
                right_leaf_value: LeafValue::Scalar(0.8),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };

    let rows = vec![
        vec![0.0, 0.0, 0.0, 0.0], // LLLL → visits 0,1,3,7 → pred=baseline+0.1+0.3+0.5+0.7
        vec![0.0, 0.0, 0.0, 5.0], // LLLR → visits 0,1,3,7 → pred=baseline+0.1+0.3+0.5+0.8
        vec![0.0, 0.0, 5.0, 0.0], // LLR_ → visits 0,1,3 → pred=baseline+0.1+0.3+0.6
        vec![0.0, 5.0, 0.0, 0.0], // LR__ → visits 0,1 → pred=baseline+0.1+0.4
        vec![5.0, 0.0, 0.0, 0.0], // R___ → visits 0 → pred=baseline+0.2
    ];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (row_idx, (bf_row, ts_row)) in brute_force
        .values
        .iter()
        .zip(tree_shap.values.iter())
        .enumerate()
    {
        for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
            assert!(
                (bf_val - ts_val).abs() <= 1e-5,
                "row {row_idx} feature {feat_idx}: bf={bf_val:.6} ts={ts_val:.6} \
                 gap={:.3e} — TreeSHAP must match brute force on asymmetric spine trees",
                (bf_val - ts_val).abs()
            );
        }
    }

    // Additivity vs predict_row for each row.
    for (row_idx, (row, ts_values)) in rows.iter().zip(tree_shap.values.iter()).enumerate() {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = tree_shap.expected_value + ts_values.iter().sum::<f32>();
        assert!(
            (predicted - reconstructed).abs() <= 1e-5,
            "row {row_idx}: predict_row={predicted:.6}, expected_value+Σphi={reconstructed:.6}, \
             gap={:.3e}",
            (predicted - reconstructed).abs()
        );
    }
}

/// Build a SplitCandidate with a categorical bitset.
fn categorical_split_with_counts(
    node_id: u32,
    feature_index: u32,
    bitset: Vec<u8>,
    left_count: u32,
    right_count: u32,
) -> SplitCandidate {
    SplitCandidate {
        node_id,
        feature_index,
        threshold_bin: 0,
        gain: 1.0,
        default_left: true,
        is_categorical: true,
        categorical_bitset: Some(bitset),
        left_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: left_count as f32,
            grad_sq_sum: 0.0,
            row_count: left_count,
        },
        right_stats: NodeStats {
            grad_sum: 0.0,
            hess_sum: right_count as f32,
            grad_sq_sum: 0.0,
            row_count: right_count,
        },
    }
}

#[test]
fn brute_force_categorical_split_additivity() {
    // Single tree with one categorical split on feature 0.
    // Bitset 0b0000_0101 = categories {0, 2} go left; {1, 3} go right.
    let model = TrainedModel {
        baseline_prediction: 1.0,
        feature_count: 2,
        stumps: vec![TrainedStump {
            split: categorical_split_with_counts(0, 0, vec![0b0000_0101], 4, 6),
            left_leaf_value: LeafValue::Scalar(-0.3),
            right_leaf_value: LeafValue::Scalar(0.2),
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

    // Feature 0 values: 0.0 (cat 0, left), 1.0 (cat 1, right),
    //                    2.0 (cat 2, left), 3.0 (cat 3, right)
    let rows = vec![
        vec![0.0, 5.0], // cat 0 -> left
        vec![1.0, 5.0], // cat 1 -> right
        vec![2.0, 5.0], // cat 2 -> left
        vec![3.0, 5.0], // cat 3 -> right
    ];

    let explanation = explain_rows_brute_force(&model, &rows, None).expect("brute force works");

    // Verify additivity: sum of SHAP values + expected_value == prediction
    for (row, values) in rows.iter().zip(explanation.values.iter()) {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn tree_shap_categorical_split_additivity() {
    // Same model as brute_force_categorical_split_additivity.
    let model = TrainedModel {
        baseline_prediction: 1.0,
        feature_count: 2,
        stumps: vec![TrainedStump {
            split: categorical_split_with_counts(0, 0, vec![0b0000_0101], 4, 6),
            left_leaf_value: LeafValue::Scalar(-0.3),
            right_leaf_value: LeafValue::Scalar(0.2),
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

    let rows = vec![
        vec![0.0, 5.0],
        vec![1.0, 5.0],
        vec![2.0, 5.0],
        vec![3.0, 5.0],
    ];

    let explanation = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (row, values) in rows.iter().zip(explanation.values.iter()) {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = explanation.expected_value + values.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn tree_shap_matches_brute_force_on_categorical_model() {
    // Two trees: first uses a categorical split on feature 0, second
    // uses a numeric split on feature 1. This exercises both split types
    // in the same model and verifies the algorithms agree.
    let stride = 1u32 << 20;
    let model = TrainedModel {
        baseline_prediction: 0.5,
        feature_count: 2,
        stumps: vec![
            TrainedStump {
                // Tree 0: categorical split on feature 0
                // Bitset 0b0000_0011 = categories {0, 1} go left
                split: categorical_split_with_counts(0, 0, vec![0b0000_0011], 5, 5),
                left_leaf_value: LeafValue::Scalar(-0.2),
                right_leaf_value: LeafValue::Scalar(0.3),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                // Tree 1: numeric split on feature 1 at threshold 3
                split: split_with_counts(stride, 1, 3, 4, 6),
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

    let rows = vec![
        vec![0.0, 1.0], // cat 0 left, numeric left
        vec![1.0, 5.0], // cat 1 left, numeric right
        vec![2.0, 1.0], // cat 2 right, numeric left
        vec![3.0, 5.0], // cat 3 right, numeric right
    ];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    assert_close(brute_force.expected_value, tree_shap.expected_value);

    for (row_idx, (bf_row, ts_row)) in brute_force
        .values
        .iter()
        .zip(tree_shap.values.iter())
        .enumerate()
    {
        for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }

    // Also verify additivity for both algorithms.
    for (row, values) in rows.iter().zip(brute_force.values.iter()) {
        let predicted = model.predict_row(row).expect("predicts");
        let reconstructed = brute_force.expected_value + values.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn tree_shap_deep_tree_with_repeated_feature() {
    // A single tree of depth 2 that splits on feature 0 at both levels.
    // Root (node 0): split on f0 at 5
    //   Left (node 1): split on f0 at 2
    //   Right (node 2): split on f1 at 3
    let model = TrainedModel {
        baseline_prediction: 0.5,
        feature_count: 2,
        stumps: vec![
            TrainedStump {
                split: split_with_counts(0, 0, 5, 6, 4),
                left_leaf_value: LeafValue::Scalar(0.2),
                right_leaf_value: LeafValue::Scalar(-0.3),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(1, 0, 2, 3, 3),
                left_leaf_value: LeafValue::Scalar(0.1),
                right_leaf_value: LeafValue::Scalar(-0.1),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(2, 1, 3, 2, 2),
                left_leaf_value: LeafValue::Scalar(0.15),
                right_leaf_value: LeafValue::Scalar(-0.15),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };

    let rows = vec![
        vec![1.0, 1.0],
        vec![1.0, 5.0],
        vec![4.0, 1.0],
        vec![4.0, 5.0],
        vec![8.0, 1.0],
        vec![8.0, 5.0],
    ];

    let brute_force = explain_rows_brute_force(&model, &rows, None).expect("brute force works");
    let tree_shap = explain_rows_tree_shap(&model, &rows, None).expect("tree shap works");

    for (row_idx, (bf_row, ts_row)) in brute_force
        .values
        .iter()
        .zip(tree_shap.values.iter())
        .enumerate()
    {
        for (feat_idx, (bf_val, ts_val)) in bf_row.iter().zip(ts_row.iter()).enumerate() {
            assert!(
                (bf_val - ts_val).abs() <= ADDITIVITY_ATOL,
                "row {row_idx} feature {feat_idx}: brute force {bf_val} vs tree shap {ts_val}"
            );
        }
    }
}

// ── Linear-leaf (piecewise-linear / leaf_model='linear') SHAP tests ─────

/// Build a 2-feature, 1-stump model whose left/right leaves are linear in
/// feature 1 with regressor mean 0.5.  Row layout: feature 0 is the split
/// feature, feature 1 is the linear regressor.
fn linear_fixture_model(feature_baseline: Option<Vec<f32>>) -> TrainedModel {
    // Tree:  split on feature 0 at bin 1
    //   left  leaf:  intercept=0.4, w=0.7 on feature 1
    //   right leaf:  intercept=-0.2, w=-0.3 on feature 1
    TrainedModel {
        baseline_prediction: 0.1,
        feature_count: 2,
        stumps: vec![TrainedStump {
            split: split_with_counts(0, 0, 1, 6, 4),
            left_leaf_value: LeafValue::Linear(LinearLeaf::identity_scaled(
                0.4,
                vec![0.7],
                vec![1],
            )),
            right_leaf_value: LeafValue::Linear(LinearLeaf::identity_scaled(
                -0.2,
                vec![-0.3],
                vec![1],
            )),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline,
        neutralization_metadata: None,
    }
}

fn scaled_linear_fixture_model(feature_baseline: Option<Vec<f32>>) -> TrainedModel {
    TrainedModel {
        baseline_prediction: -0.2,
        feature_count: 2,
        stumps: vec![TrainedStump {
            split: split_with_counts(0, 0, 1, 5, 5),
            left_leaf_value: LeafValue::Linear(LinearLeaf::scaled(
                0.4,
                vec![0.7],
                vec![1],
                vec![10.0],
                vec![0.5],
            )),
            right_leaf_value: LeafValue::Linear(LinearLeaf::scaled(
                -0.3,
                vec![-0.5],
                vec![1],
                vec![10.0],
                vec![0.5],
            )),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline,
        neutralization_metadata: None,
    }
}

#[test]
fn shap_linear_leaves_does_not_reject_artifact() {
    // Regression guard: TreeSHAP used to error with `NotSupported` when any
    // leaf was linear; v0.7.1 lifts that and decomposes the leaf instead.
    let model = linear_fixture_model(Some(vec![0.0, 0.5]));
    let artifact = model.to_artifact_bytes().expect("artifact serializes");
    let rows = vec![vec![0.0, 0.5], vec![3.0, 0.5]];
    let result = explain_rows_from_artifact_bytes(&artifact, &rows);
    assert!(result.is_ok(), "expected ok, got {result:?}");
}

#[test]
fn shap_linear_leaves_additivity_with_baseline_brute_force() {
    // Brute-force exact path (≤ 25 split features).  With a baseline
    // recorded for feature 1, `Σ phi[i] + expected_value == predict(x)`
    // and the path-attribution-vs-linear-deviation split is well-defined.
    let baseline = vec![0.0_f32, 0.5_f32];
    let model = linear_fixture_model(Some(baseline));
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let rows = vec![
        vec![0.0_f32, 1.0_f32], // goes left
        vec![3.0_f32, 1.0_f32], // goes right
        vec![0.0_f32, -1.0_f32],
        vec![3.0_f32, -1.0_f32],
    ];
    let explanation =
        explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");
    for (row, phi) in rows.iter().zip(explanation.values.iter()) {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn shap_scaled_linear_leaves_remain_additive() {
    let baseline = vec![0.0_f32, 12.0_f32];
    let model = scaled_linear_fixture_model(Some(baseline));
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let rows = vec![
        vec![0.0_f32, 12.0_f32],
        vec![0.0_f32, 14.0_f32],
        vec![3.0_f32, 8.0_f32],
    ];
    let explanation =
        explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");
    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");

    for (row_idx, (row, phi)) in rows.iter().zip(explanation.values.iter()).enumerate() {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
        let tol = additivity_tolerance(predicted);
        assert!(
            (reconstructed - predicted).abs() <= tol,
            "row {row_idx}: reconstructed {reconstructed} vs predicted {predicted}"
        );
    }

    let delta_phi_feature_1 = explanation.values[1][1] - explanation.values[0][1];
    assert!(
        (delta_phi_feature_1 - 0.7).abs() <= ADDITIVITY_ATOL,
        "expected standardized delta 0.7, got {delta_phi_feature_1}"
    );
}

#[test]
fn shap_linear_leaves_additivity_without_baseline_brute_force() {
    // Back-compat: artifact produced before v0.7.1 will not carry a
    // FeatureBaseline section.  SHAP must still satisfy additivity in
    // that case (treating the global baseline as 0 — degraded
    // interventional decomposition but still exact in aggregate).
    let model = linear_fixture_model(None);
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let rows = vec![vec![0.0_f32, 1.0_f32], vec![3.0_f32, -0.5_f32]];
    let explanation =
        explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");
    for (row, phi) in rows.iter().zip(explanation.values.iter()) {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
        assert_close(reconstructed, predicted);
    }
}

#[test]
fn leaf_constant_part_accumulates_scaled_terms_in_f64() {
    let leaf = LeafValue::Linear(LinearLeaf::scaled(
        16_777_216.0,
        vec![1.0, -1.0],
        vec![0, 1],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
    ));
    let baseline = vec![1.0_f32, 16_777_216.0_f32];

    let constant = crate::linear_leaf::leaf_constant_part(&leaf, Some(&baseline));
    assert!((constant - 1.0).abs() < 1e-12, "constant part: {constant}");
}

#[test]
fn shap_linear_leaves_attribute_deviation_to_regressor_feature() {
    // For a row sitting exactly at the baseline of the regressor, all
    // linear-deviation terms vanish and SHAP[regressor] == 0 (any
    // attribution must come purely from path effects).  Conversely, when
    // the regressor sits off-baseline, that feature picks up the
    // standardized deviation w_j * (z_j(x) - z_j(baseline)) on top of any
    // path contribution.
    let baseline = vec![0.0_f32, 0.5_f32];
    let model = linear_fixture_model(Some(baseline.clone()));
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    // Row 0: feature 0 = 0 → goes left, feature 1 = 0.5 (= μ_1).
    // Linear deviation w_left * (0.5 - 0.5) = 0.
    let on_baseline_row = vec![0.0_f32, 0.5_f32];
    let off_baseline_row = vec![0.0_f32, 1.5_f32];
    let explanation = explain_rows_from_artifact_bytes(
        &artifact,
        &[on_baseline_row.clone(), off_baseline_row.clone()],
    )
    .expect("explanation succeeds");

    // The two rows take the same path (feature 0 = 0 → left); they only
    // differ in feature 1.  Therefore SHAP[feature 1] must differ by
    // exactly w_left * (1.5 - 0.5) = 0.7 * 1.0 = 0.7.
    let delta_phi_feat1 = explanation.values[1][1] - explanation.values[0][1];
    assert!(
        (delta_phi_feat1 - 0.7).abs() <= ADDITIVITY_ATOL,
        "expected ΔSHAP[feature 1] = 0.7, got {delta_phi_feat1}"
    );
}

/// Build a 3-feature 2-stump model that mixes a scalar leaf with a linear
/// leaf, so we exercise the codepath that has to handle both leaf flavours
/// within a single tree.
fn mixed_leaf_fixture_model() -> TrainedModel {
    TrainedModel {
        baseline_prediction: 0.0,
        feature_count: 3,
        stumps: vec![
            TrainedStump {
                split: split_with_counts(0, 0, 1, 5, 5),
                // Left leaf: scalar
                left_leaf_value: LeafValue::Scalar(0.3),
                // Right child has another split, so the right leaf value
                // here is the partial contribution along that branch.
                right_leaf_value: LeafValue::Scalar(-0.1),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
            TrainedStump {
                split: split_with_counts(2, 2, 0, 3, 2),
                left_leaf_value: LeafValue::Linear(LinearLeaf::identity_scaled(
                    0.1,
                    vec![0.4],
                    vec![1],
                )),
                right_leaf_value: LeafValue::Linear(LinearLeaf::identity_scaled(
                    -0.2,
                    vec![0.6],
                    vec![1],
                )),
                tree_weight: 1.0,
                multi_output_leaf_values: None,
            },
        ],
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: Some(vec![0.0, 0.5, 0.0]),
        neutralization_metadata: None,
    }
}

#[test]
fn shap_linear_leaves_mixed_with_scalar_leaves_satisfies_additivity() {
    let model = mixed_leaf_fixture_model();
    let artifact = model.to_artifact_bytes().expect("artifact serializes");
    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");

    let rows = vec![
        vec![0.0_f32, 0.5_f32, 0.0_f32],  // left scalar leaf
        vec![3.0_f32, 1.0_f32, 0.0_f32],  // right→left linear leaf
        vec![3.0_f32, -0.5_f32, 2.0_f32], // right→right linear leaf
    ];
    let explanation =
        explain_rows_from_artifact_bytes(&artifact, &rows).expect("explanation succeeds");

    for (row_idx, (row, phi)) in rows.iter().zip(explanation.values.iter()).enumerate() {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = explanation.expected_value + phi.iter().sum::<f32>();
        assert!(
            (reconstructed - predicted).abs() <= ADDITIVITY_ATOL,
            "row {row_idx}: reconstructed {reconstructed} vs predicted {predicted}"
        );
    }
}

#[test]
fn shap_interactions_linear_leaves_satisfies_additivity() {
    let baseline = vec![0.0_f32, 0.5_f32];
    let model = linear_fixture_model(Some(baseline));
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let rows = vec![
        vec![0.0_f32, 1.0_f32], // goes left
        vec![3.0_f32, 1.0_f32], // goes right
        vec![0.0_f32, -1.0_f32],
        vec![3.0_f32, -1.0_f32],
    ];
    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows)
        .expect("interaction explanation succeeds");
    let per_feature = explain_rows_from_artifact_bytes(&artifact, &rows).expect("per-feature");

    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");
    let feature_count = model.feature_count;
    for (row_idx, (row, matrix)) in rows.iter().zip(batch.values.iter()).enumerate() {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = batch.expected_value
            + matrix
                .iter()
                .map(|m_row| m_row.iter().sum::<f32>())
                .sum::<f32>();
        assert!(
            (reconstructed - predicted).abs() <= ADDITIVITY_ATOL,
            "row {row_idx}: reconstructed {reconstructed} vs predicted {predicted}"
        );

        // Symmetry: matrix[i][j] == matrix[j][i]
        for i in 0..feature_count {
            for j in 0..feature_count {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-5,
                    "symmetry: matrix[{i}][{j}]={} matrix[{j}][{i}]={}",
                    matrix[i][j],
                    matrix[j][i]
                );
            }
        }

        // Row marginal: Σ_j Φ_ij == φ_i
        for i in 0..feature_count {
            let marginal: f32 = matrix[i].iter().sum();
            let phi = per_feature.values[row_idx][i];
            assert!(
                (marginal - phi).abs() < 1e-4,
                "row {row_idx} feature {i}: marginal={marginal} phi={phi}"
            );
        }
    }
}

#[test]
fn shap_interactions_linear_leaves_mixed_with_scalar_leaves_satisfies_additivity() {
    let model = mixed_leaf_fixture_model();
    let artifact = model.to_artifact_bytes().expect("artifact serializes");
    let predictor = Predictor::from_artifact_bytes(&artifact).expect("predictor builds");

    let rows = vec![
        vec![0.0_f32, 0.5_f32, 0.0_f32],  // left scalar leaf
        vec![3.0_f32, 1.0_f32, 0.0_f32],  // right→left linear leaf
        vec![3.0_f32, -0.5_f32, 2.0_f32], // right→right linear leaf
    ];
    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows)
        .expect("interaction explanation succeeds");
    let per_feature = explain_rows_from_artifact_bytes(&artifact, &rows).expect("per-feature");

    let feature_count = model.feature_count;
    for (row_idx, (row, matrix)) in rows.iter().zip(batch.values.iter()).enumerate() {
        let predicted = predictor.predict_row(row).expect("predict succeeds");
        let reconstructed = batch.expected_value
            + matrix
                .iter()
                .map(|m_row| m_row.iter().sum::<f32>())
                .sum::<f32>();
        assert!(
            (reconstructed - predicted).abs() <= ADDITIVITY_ATOL,
            "row {row_idx}: reconstructed {reconstructed} vs predicted {predicted}"
        );

        // Symmetry: matrix[i][j] == matrix[j][i]
        for i in 0..feature_count {
            for j in 0..feature_count {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-5,
                    "symmetry: matrix[{i}][{j}]={} matrix[{j}][{i}]={}",
                    matrix[i][j],
                    matrix[j][i]
                );
            }
        }

        // Row marginal: Σ_j Φ_ij == φ_i
        for i in 0..feature_count {
            let marginal: f32 = matrix[i].iter().sum();
            let phi = per_feature.values[row_idx][i];
            assert!(
                (marginal - phi).abs() < 1e-4,
                "row {row_idx} feature {i}: marginal={marginal} phi={phi}"
            );
        }
    }
}

// ── TreeSHAP polynomial-path diagnostic: synthetic deep trees ────────────
//
// Used to localize Limitation 5 (TreeSHAP polynomial-path additivity
// drift on deep trees with many distinct splits).  The strategy:
// build a synthetic depth-D tree using only F (≤25) distinct features
// so the brute-force exact path remains tractable, then call BOTH
// `explain_rows_brute_force` and `explain_rows_tree_shap` directly
// and require they agree per-feature.  Brute-force is the ground
// truth (it enumerates all 2^F subsets).
//
// We sweep over depth and feature-pattern strategies to find the
// minimal topology that triggers the polynomial-path bug.

/// Build a full binary tree of depth `depth` with stumps at every
/// internal node.  Each stump's feature is chosen via `feature_for`.
/// Each stump's leaves are scalar values: `leaf_value_for(node_id,
/// goes_left)`.  Per-stump cover is `cover_for(node_id)`.  Threshold
/// is `node_id as u16 % 4` (arbitrary; rows below choose splits that
/// always go a deterministic direction).
fn build_full_tree(
    feature_count: usize,
    depth: usize,
    feature_for: impl Fn(u32) -> u32,
    leaf_value_for: impl Fn(u32, bool) -> f32,
    cover_for: impl Fn(u32) -> u32,
) -> Vec<TrainedStump> {
    // Pre-compute per-leaf covers, then propagate up so each parent's
    // left_stats/right_stats == sum of its descendant leaf covers.
    // Without this consistency, `node.cover()` recursion and the
    // per-stump left_stats.row_count would disagree.
    // node_id convention: children of n are 2n+1 and 2n+2.  For a
    // full tree of depth D: internal nodes have ids [0, 2^D - 1),
    // leaves have ids [2^D - 1, 2^(D+1) - 1).
    let n_leaves = 1u32 << depth; // count
    let leaf_id_start = n_leaves - 1; // first leaf node_id
    let total_nodes = (1u32 << (depth + 1)) - 1; // = internal + leaves
    let mut subtree_cover = vec![0u32; total_nodes as usize];
    for leaf_node_id in leaf_id_start..total_nodes {
        subtree_cover[leaf_node_id as usize] = cover_for(leaf_node_id).max(1);
    }
    // Bottom-up propagation: internal node count = leaf_id_start.
    for node_id in (0..leaf_id_start).rev() {
        let l = subtree_cover[(2 * node_id + 1) as usize];
        let r = subtree_cover[(2 * node_id + 2) as usize];
        subtree_cover[node_id as usize] = l + r;
    }

    let mut stumps = Vec::new();
    for node_id in 0..leaf_id_start {
        let feat = feature_for(node_id);
        let left_count = subtree_cover[(2 * node_id + 1) as usize];
        let right_count = subtree_cover[(2 * node_id + 2) as usize];
        let _ = feature_count; // sanity argument; not used directly
        stumps.push(TrainedStump {
            split: split_with_counts(
                node_id,
                feat,
                (node_id as u16) & 0x3,
                left_count,
                right_count,
            ),
            left_leaf_value: LeafValue::Scalar(leaf_value_for(node_id, true)),
            right_leaf_value: LeafValue::Scalar(leaf_value_for(node_id, false)),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        });
    }
    stumps
}

/// Compute the depth of a node_id in a full binary tree
/// (root = 0, children of n = 2n+1, 2n+2).
fn node_depth(mut node_id: u32) -> u32 {
    let mut depth = 0;
    while node_id > 0 {
        node_id = (node_id - 1) / 2;
        depth += 1;
    }
    depth
}

fn synthetic_deep_model(depth: usize, n_features: usize, _seed: u64) -> TrainedModel {
    let stumps = build_full_tree(
        n_features,
        depth,
        // Feature pattern: feature index = node's depth in the tree.
        // Guarantees every root-to-leaf path uses DISTINCT features
        // (no duplicates), as long as n_features > depth.  This
        // isolates the duplicate-handling code path from other
        // potential bugs.  When n_features <= depth, paths cycle
        // through features (forces duplicates).
        |node_id| node_depth(node_id) % n_features as u32,
        // Leaf value: deterministic pseudo-random in [-1, 1].
        |node_id, goes_left| {
            let h = ((node_id as u64).wrapping_mul(0xD6E8_FD50_89A4_7A4D)
                ^ if goes_left { 0xAAAA } else { 0x5555 }) as u32;
            ((h as f32) / (u32::MAX as f32) - 0.5) * 2.0
        },
        // Cover: weight by node depth so deep nodes have small cover.
        |node_id| {
            // Approximate row count: total / (2^subtree_depth)
            // for node at depth d in a full tree of depth `depth`.
            // For diagnosis we don't need realism, just non-zero.
            let h = ((node_id as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) as u32;
            (h % 100) + 1
        },
    );
    TrainedModel {
        baseline_prediction: 0.0,
        feature_count: n_features,
        stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    }
}

fn deterministic_rows(feature_count: usize, n_rows: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_rows);
    let mut state = seed;
    for _ in 0..n_rows {
        let row = (0..feature_count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Choose values in [0, 5) so any threshold <= 3 splits.
                (((state >> 32) as f32) / (u32::MAX as f32 + 1.0)) * 5.0
            })
            .collect::<Vec<_>>();
        out.push(row);
    }
    out
}

/// Regression test for the TreeSHAP polynomial-path additivity drift
/// closed in v0.7.5 (formerly Limitation 5).
///
/// The bug was in `ts_unextend_path`: when removing a duplicate feature
/// entry from the path, the function shifted the entire `PathElement`
/// struct (including `pweight`), clobbering the pweights that the
/// unwind loop had just carefully computed in place.  The reference
/// implementation in slundberg/shap stores the four path fields as
/// four parallel arrays and only shifts the first three (feature_index,
/// zero_fraction, one_fraction), preserving pweights.
///
/// This sweep builds synthetic full binary trees of varying depth and
/// distinct-feature count, then asserts that the polynomial TreeSHAP
/// path agrees with the brute-force exact path per-feature within
/// floating-point tolerance.  Both `n_features < depth` (forced
/// path-duplicates) and `n_features >= depth` (no duplicates) are
/// covered, so the unwind path is exercised across the full matrix.
///
/// Brute-force is the ground truth (it enumerates 2^N subsets).
/// Capped at depth 7 to keep brute-force tractable.
#[test]
fn tree_shap_polynomial_path_matches_brute_force_on_full_trees() {
    for &depth in &[2_usize, 3, 4, 5, 6, 7] {
        for &n_features in &[2_usize, 3, 5, 8, 12] {
            let model = synthetic_deep_model(depth, n_features, 0xABCD_EF01);
            let rows = deterministic_rows(n_features, 4, 0x1234_5678);
            let bf = explain_rows_brute_force(&model, &rows, None)
                .expect("brute-force exact path succeeds");
            let poly = explain_rows_tree_shap(&model, &rows, None)
                .expect("polynomial path succeeds (no additivity drift)");

            for (row_idx, (bf_row, poly_row)) in
                bf.values.iter().zip(poly.values.iter()).enumerate()
            {
                for (feat_idx, (a, b)) in bf_row.iter().zip(poly_row.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-5,
                        "depth={depth} n_features={n_features} row={row_idx} \
                         feat={feat_idx}: brute_force={a}, polynomial={b}, \
                         |diff|={}",
                        (a - b).abs(),
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// v0.11.0: SHAP interaction values (Lundberg Algorithm 2)
// ---------------------------------------------------------------------

/// Brute-force pairwise SHAP interaction oracle.  O(2^k) in the number of
/// distinct split features k; only valid for fixtures with k ≤ ~10.
///
/// `Φ_ij = (1/2) · Σ_{S ⊆ N\{i,j}} (|S|! · (k-|S|-2)! / (k-1)!) ·
///         [f(S ∪ {i,j}) − f(S ∪ {i}) − f(S ∪ {j}) + f(S)]`
/// where the sums are over subsets of the split-feature set excluding `i, j`.
/// Off-diagonal entries between non-split-features are zero (the model can't
/// depend on them).  The diagonal is filled from
/// `Φ_ii = φ_i − Σ_{j ≠ i} Φ_ij` to enforce the row-marginal invariant.
fn brute_force_interactions_for_row(model: &TrainedModel, row: &[f32]) -> (f32, Vec<Vec<f32>>) {
    let n = model.feature_count;
    let structure = build_model_structure(model).expect("model structure");
    let subset_expectations = compute_subset_expectations(model, row, &structure, None, None)
        .expect("subset expectations");
    let split_features = &structure.split_features;
    let k = split_features.len();
    let factorials = factorial_table(k.max(2));
    let mut phi = shapley_values_for_row_f64(model, row, &subset_expectations, &structure, 0)
        .expect("per-feature shap");

    if crate::linear_leaf::model_has_linear_leaves(model) {
        crate::linear_leaf::distribute_linear_terms_for_row(
            model,
            row,
            model.feature_baseline.as_deref(),
            None,
            &mut phi,
        );
    }

    let mut matrix = vec![vec![0.0_f64; n]; n];

    // Off-diagonal: only nonzero for pairs of split features.
    if k >= 2 {
        for a in 0..k {
            for b in (a + 1)..k {
                let feat_i = split_features[a];
                let feat_j = split_features[b];
                let bit_i = 1_u64 << a;
                let bit_j = 1_u64 << b;
                let mut accum = 0.0_f64;
                let others_count = k - 2;
                // Build the list of OTHER split-feature bit positions.
                let others: Vec<u32> = (0..k as u32)
                    .filter(|&p| p as usize != a && p as usize != b)
                    .collect();
                for sub_mask in 0..(1u32 << others_count) {
                    let mut s_bits: u64 = 0;
                    let mut s_size = 0_usize;
                    for (idx, &pos) in others.iter().enumerate() {
                        if (sub_mask >> idx) & 1 == 1 {
                            s_bits |= 1_u64 << pos;
                            s_size += 1;
                        }
                    }
                    let f_s = subset_expectations[s_bits as usize] as f64;
                    let f_si = subset_expectations[(s_bits | bit_i) as usize] as f64;
                    let f_sj = subset_expectations[(s_bits | bit_j) as usize] as f64;
                    let f_sij = subset_expectations[(s_bits | bit_i | bit_j) as usize] as f64;
                    // Weight: |S|! · (k - |S| - 2)! / (k - 1)!
                    let weight =
                        factorials[s_size] * factorials[k - s_size - 2] / factorials[k - 1];
                    accum += weight * (f_sij - f_si - f_sj + f_s);
                }
                let half = 0.5 * accum;
                matrix[feat_i][feat_j] = half;
                matrix[feat_j][feat_i] = half;
            }
        }
    }

    // Diagonal: Φ_ii = φ_i − Σ_{j ≠ i} Φ_ij
    for i in 0..n {
        let off: f64 = (0..n).filter(|&j| j != i).map(|j| matrix[i][j]).sum();
        matrix[i][i] = phi[i] - off;
    }

    let expected_value = subset_expectations[0];
    let matrix_f32: Vec<Vec<f32>> = matrix
        .into_iter()
        .map(|row_i| row_i.into_iter().map(|v| v as f32).collect())
        .collect();
    (expected_value, matrix_f32)
}

#[test]
fn brute_force_interactions_satisfy_additivity_on_fixture() {
    let model = fixture_model();
    for row in fixture_rows() {
        let (expected, interactions) = brute_force_interactions_for_row(&model, &row);
        let reconstructed: f32 = interactions
            .iter()
            .map(|r| r.iter().sum::<f32>())
            .sum::<f32>()
            + expected;
        let predicted = local_path_predict(&model, &row, None);
        let tol = additivity_tolerance(predicted);
        assert!(
            (reconstructed - predicted).abs() < tol,
            "row={row:?} predicted={predicted} reconstructed={reconstructed} tol={tol}"
        );
    }
}

#[test]
fn explain_interactions_from_artifact_returns_pairwise_matrix() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let rows = fixture_rows();

    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows)
        .expect("interaction explanation succeeds");

    assert_eq!(batch.values.len(), rows.len());
    let feature_count = fixture_model().feature_count;
    for row_interactions in &batch.values {
        assert_eq!(row_interactions.len(), feature_count);
        for column in row_interactions {
            assert_eq!(column.len(), feature_count);
        }
    }
}

#[test]
fn tree_shap_interactions_additivity_holds() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let rows = fixture_rows();
    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("batch");
    let model = fixture_model();
    for (row_idx, row) in rows.iter().enumerate() {
        let matrix = &batch.values[row_idx];
        let reconstructed: f32 =
            matrix.iter().map(|r| r.iter().sum::<f32>()).sum::<f32>() + batch.expected_value;
        let predicted = local_path_predict(&model, row, None);
        let tol = additivity_tolerance(predicted);
        assert!(
            (reconstructed - predicted).abs() < tol,
            "row={row_idx} reconstructed={reconstructed} predicted={predicted}"
        );
    }
}

#[test]
fn tree_shap_interactions_row_marginal_equals_per_feature_shap() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let rows = fixture_rows();
    let pairwise = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("pairwise");
    let per_feature = explain_rows_from_artifact_bytes(&artifact, &rows).expect("per-feature");

    let feature_count = fixture_model().feature_count;
    for row_idx in 0..rows.len() {
        for i in 0..feature_count {
            let marginal: f32 = pairwise.values[row_idx][i].iter().sum();
            let phi = per_feature.values[row_idx][i];
            assert!(
                (marginal - phi).abs() < 1e-4,
                "row={row_idx} i={i} marginal={marginal} phi={phi}"
            );
        }
    }
}

#[test]
fn tree_shap_interactions_matrix_is_symmetric() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let rows = fixture_rows();
    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("batch");
    for row_matrix in &batch.values {
        for i in 0..row_matrix.len() {
            for j in 0..row_matrix.len() {
                let a = row_matrix[i][j];
                let b = row_matrix[j][i];
                assert!(
                    (a - b).abs() < 1e-5,
                    "symmetry: a[{i}][{j}]={a} a[{j}][{i}]={b}"
                );
            }
        }
    }
}

#[test]
fn tree_shap_interactions_match_brute_force_on_fixture() {
    let artifact = fixture_model()
        .to_artifact_bytes()
        .expect("artifact serializes");
    let model = fixture_model();
    let rows = fixture_rows();

    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("interactions");
    for (row_idx, row) in rows.iter().enumerate() {
        let (_, expected_matrix) = brute_force_interactions_for_row(&model, row);
        for i in 0..model.feature_count {
            for j in 0..model.feature_count {
                let got = batch.values[row_idx][i][j];
                let want = expected_matrix[i][j];
                assert!(
                    (got - want).abs() < 1e-4,
                    "row={row_idx} i={i} j={j} got={got} want={want}"
                );
            }
        }
    }
}

#[test]
fn tree_shap_interactions_synthetic_depth_3_four_features_matches_brute_force() {
    let model = synthetic_deep_model(3, 4, 1729);
    let rows = deterministic_rows(4, 5, 1729);
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("batch");
    for (row_idx, row) in rows.iter().enumerate() {
        let (_, expected_matrix) = brute_force_interactions_for_row(&model, row);
        for i in 0..4 {
            for j in 0..4 {
                let got = batch.values[row_idx][i][j];
                let want = expected_matrix[i][j];
                assert!(
                    (got - want).abs() < 5e-3,
                    "row={row_idx} i={i} j={j} got={got} want={want}"
                );
            }
        }
    }
}

#[test]
fn tree_shap_interactions_depth_5_three_features_satisfies_additivity() {
    // 3 features at depth 5 — forces feature duplicates on every path
    // (since depth > n_features), exercising the duplicate-handling
    // branch in `ts_recurse_conditioning`.
    let model = synthetic_deep_model(5, 3, 4242);
    let rows = deterministic_rows(3, 8, 4242);
    let artifact = model.to_artifact_bytes().expect("artifact serializes");

    let batch = explain_interactions_from_artifact_bytes(&artifact, &rows).expect("batch");
    for (row_idx, row) in rows.iter().enumerate() {
        let matrix = &batch.values[row_idx];
        let reconstructed: f32 =
            matrix.iter().map(|r| r.iter().sum::<f32>()).sum::<f32>() + batch.expected_value;
        let predicted = local_path_predict(&model, row, None);
        let tol = additivity_tolerance(predicted) * 4.0;
        assert!(
            (reconstructed - predicted).abs() < tol,
            "row={row_idx} reconstructed={reconstructed} predicted={predicted} tol={tol}"
        );
    }
}
