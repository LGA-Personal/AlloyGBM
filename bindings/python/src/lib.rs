#![allow(clippy::too_many_arguments)]

mod callbacks;
mod categorical_bridge;
mod errors;
mod joint;
mod params;
mod predict;
mod pyclasses;
mod quantization;
mod train;
use crate::joint::{JointPredictorHandle, train_joint_multi_label_ranker};
use crate::predict::{
    NativePredictorHandle, predictor_predict_batch, predictor_predict_batch_canonical,
    predictor_predict_batch_canonical_dense, predictor_predict_batch_dense,
    shap_explain_interactions, shap_explain_interactions_dense,
    shap_explain_interactions_dense_with_binning, shap_explain_interactions_with_binning,
    shap_explain_rows, shap_explain_rows_dense, shap_explain_rows_dense_with_binning,
    shap_explain_rows_with_binning, shap_global_importance, shap_global_importance_dense,
    shap_global_importance_dense_with_binning, shap_global_importance_with_binning,
};
use crate::pyclasses::{
    NativeContinuousBinningMetadata, NativeIterationDiagnostics, NativeRuntimeInfo,
    NativeTrainingResult, NativeTrainingSummary, native_runtime_info,
};
use crate::train::{
    train_regression_artifact, train_regression_artifact_dense,
    train_regression_artifact_dense_with_summary,
    train_regression_artifact_dense_with_summary_bytes, train_regression_artifact_with_summary,
};

use alloygbm_core::DenseMatrixView;
use pyo3::prelude::*;

pub(crate) const DEFAULT_TRAIN_ROUNDS: usize = 6;
pub(crate) const MAX_SUPPORTED_TRAIN_ROUNDS: usize = 4096;
pub(crate) const PRE_BINNED_INTEGER_TOLERANCE: f32 = 1e-6;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U8: u16 = 254;
pub(crate) const MAX_CONTINUOUS_QUANTIZED_BIN_U16: u16 = 65534;
pub(crate) const MIN_CONTINUOUS_QUANTIZED_BINS: usize = 2;
pub(crate) const LINEAR_TAIL_RANK_ENV_VAR: &str = "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK";
pub(crate) const LINEAR_TAIL_CORE_SPAN_RATIO_ENV_VAR: &str =
    "ALLOYGBM_EXPERIMENT_LINEAR_TAIL_CORE_SPAN_RATIO";
pub(crate) const DEFAULT_LINEAR_TAIL_CORE_SPAN_RATIO_THRESHOLD: f32 = 0.10;

pub(crate) fn dense_rows_from_flat_values(
    values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> Result<Vec<Vec<f32>>, String> {
    DenseMatrixView::new(row_count, feature_count, values).map_err(|error| error.to_string())?;
    Ok(values
        .chunks(feature_count)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>())
}

#[pymodule]
fn _alloygbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeRuntimeInfo>()?;
    m.add_class::<NativePredictorHandle>()?;
    m.add_class::<JointPredictorHandle>()?;
    m.add_class::<NativeContinuousBinningMetadata>()?;
    m.add_class::<NativeTrainingSummary>()?;
    m.add_class::<NativeTrainingResult>()?;
    m.add_class::<NativeIterationDiagnostics>()?;
    m.add_function(wrap_pyfunction!(native_runtime_info, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch_dense, m)?)?;
    m.add_function(wrap_pyfunction!(predictor_predict_batch_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(
        predictor_predict_batch_canonical_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_interactions_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(
        shap_explain_interactions_dense_with_binning,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance, m)?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance_dense, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(shap_explain_rows_dense_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(shap_global_importance_with_binning, m)?)?;
    m.add_function(wrap_pyfunction!(
        shap_global_importance_dense_with_binning,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact_dense, m)?)?;
    m.add_function(wrap_pyfunction!(train_regression_artifact_with_summary, m)?)?;
    m.add_function(wrap_pyfunction!(
        train_regression_artifact_dense_with_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        train_regression_artifact_dense_with_summary_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(train_joint_multi_label_ranker, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_TRAIN_ROUNDS, MAX_CONTINUOUS_QUANTIZED_BIN_U8};
    use crate::categorical_bridge::{flatten_rows, resolve_categorical_spec};
    use crate::predict::{
        predictor_predict_batch_canonical_impl, predictor_predict_batch_impl,
        shap_explain_rows_impl, shap_global_importance_impl,
    };
    use crate::quantization::ContinuousBinningStrategy;
    use crate::train::train_regression_artifact_with_summary_dense_impl;
    use alloygbm_backend_cpu::CpuBackend;
    use alloygbm_categorical::TargetEncoderConfig;
    use alloygbm_core::{
        BinnedMatrix, DatasetMatrix, FactorExposureMatrix, FactorNeutralizationConfig,
        LeafModelKind, ModelSectionKind, NeutralizationKind, TrainParams, TrainingDataset,
        TreeGrowth, deserialize_model_artifact_v1, serialize_model_artifact_v1,
    };
    use alloygbm_engine::{
        CategoricalTargetEncodingSpec, EngineError, SquaredErrorObjective, Trainer,
        TrainingPolicyMode,
    };

    fn train_regression_artifact_impl(
        rows: &[Vec<f32>],
        targets: &[f32],
        params: TrainParams,
        rounds: usize,
        time_index: Option<Vec<i64>>,
        categorical_spec: Option<CategoricalTargetEncodingSpec>,
        training_policy: TrainingPolicyMode,
        store_node_debug_stats: bool,
    ) -> Result<Vec<u8>, EngineError> {
        let (dense_values, row_count, feature_count) = flatten_rows(rows)?;
        train_regression_artifact_with_summary_dense_impl(
            &dense_values,
            row_count,
            feature_count,
            targets,
            None, // sample_weights
            None, // group_id
            None, // factor_exposures
            None,
            None,
            None,
            None, // validation_sample_weights
            None, // validation_group_id
            params,
            rounds,
            time_index,
            None,
            categorical_spec.into_iter().collect(),
            Vec::new(),
            training_policy,
            store_node_debug_stats,
            ContinuousBinningStrategy::Linear,
            MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1,
            "squared_error",
            None, // init_artifact_bytes
            None, // num_classes
            None, // custom_objective_fn
            None, // custom_loss_fn
            None, // custom_metric_fn
            0,    // max_cat_threshold
        )
        .map(|result| result.artifact_bytes)
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

    fn binned_matrix_from_fixture_dataset(dataset: &TrainingDataset) -> BinnedMatrix {
        let mut bins = Vec::with_capacity(dataset.row_count() * dataset.matrix.feature_count);
        for row in 0..dataset.row_count() {
            for feature in 0..dataset.matrix.feature_count {
                let value = dataset.matrix.values[row * dataset.matrix.feature_count + feature];
                bins.push(value as u8);
            }
        }
        BinnedMatrix::new(dataset.row_count(), dataset.matrix.feature_count, 3, bins)
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
            leaf_model: LeafModelKind::Constant,
            leaf_solver: alloygbm_core::LeafSolverKind::Standard,
            dro_config: None,
            neutralization_config: None,
            boosting_mode: alloygbm_core::BoostingMode::Standard,
            tweedie_variance_power: 1.5,
            quantile_alpha: 0.5,
        }
    }

    fn fixture_categorical_values() -> Vec<String> {
        vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ]
    }

    fn train_fixture_model() -> (alloygbm_engine::TrainedModel, TrainingDataset) {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(
                &dataset,
                &binned,
                &backend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("training succeeds");
        (model, dataset)
    }

    fn legacy_trees_only_artifact_bytes() -> (Vec<u8>, Vec<Vec<f32>>) {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let strict_artifact = model.to_artifact_bytes().expect("artifact serializes");
        let parsed = deserialize_model_artifact_v1(&strict_artifact).expect("artifact parses");
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
        (legacy_artifact, rows)
    }

    #[test]
    fn binding_bridge_predictions_match_engine_predictions() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let rows = fixture_rows(&dataset);
        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 2)
            .expect("training succeeds");

        let artifact = model.to_artifact_bytes().expect("artifact serializes");
        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("bridge predicts");

        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn train_bridge_artifact_predictions_match_engine_predictions() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);

        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("bridge training succeeds");
        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("bridge predicts");

        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn canonical_bridge_predictions_match_engine_for_strict_artifacts() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let strict_artifact = model.to_artifact_bytes().expect("artifact serializes");

        let engine_predictions = model.predict_batch(&rows).expect("engine predicts");
        let canonical_predictions = predictor_predict_batch_canonical_impl(&strict_artifact, &rows)
            .expect("canonical bridge predicts");

        assert_eq!(canonical_predictions, engine_predictions);
    }

    #[test]
    fn canonical_bridge_rejects_legacy_trees_only_artifacts() {
        let (legacy_artifact, rows) = legacy_trees_only_artifact_bytes();
        let result = predictor_predict_batch_canonical_impl(&legacy_artifact, &rows);
        assert!(matches!(
            result,
            Err(alloygbm_predictor::PredictorError::ContractViolation(_))
        ));
    }

    #[test]
    fn train_bridge_categorical_path_matches_engine_predictions() {
        let dataset = quality_fixture_dataset();
        let binned = quality_fixture_binned_matrix();
        let rows = fixture_rows(&dataset);
        let categorical_spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: fixture_categorical_values(),
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };

        let trainer = Trainer::new(fixture_params()).expect("params are valid");
        let backend = CpuBackend;
        let engine_model = trainer
            .fit_iterations_with_single_target_encoded_feature(
                &dataset,
                &binned,
                &categorical_spec,
                &backend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("categorical engine training succeeds");
        let bridge_artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            Some(categorical_spec.clone()),
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("categorical bridge training succeeds");

        let bridge_model =
            alloygbm_engine::TrainedModel::from_artifact_bytes(&bridge_artifact).expect("parses");
        assert!(bridge_model.categorical_state.is_some());

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&bridge_artifact, &rows).expect("bridge predicts");
        assert_eq!(bridge_predictions, engine_predictions);
    }

    fn target_encoding_factor_loaded_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: DatasetMatrix::new(
                8,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    3.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    2.0, 1.0, //
                    3.0, 1.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: Some(
                FactorExposureMatrix::new(8, 1, vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
                    .expect("factor exposures are valid"),
            ),
        }
    }

    #[test]
    fn train_bridge_pre_target_categorical_encoding_matches_engine_residualized_targets() {
        let dataset = target_encoding_factor_loaded_dataset();
        let binned = binned_matrix_from_fixture_dataset(&dataset);
        let rows = fixture_rows(&dataset);
        let categorical_spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: vec![
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
            ],
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };
        let params = TrainParams {
            neutralization_config: Some(FactorNeutralizationConfig {
                kind: NeutralizationKind::PreTarget,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..fixture_params()
        };

        let engine_model = Trainer::new(params.clone())
            .expect("params are valid")
            .fit_iterations_with_single_target_encoded_feature(
                &dataset,
                &binned,
                &categorical_spec,
                &CpuBackend,
                &SquaredErrorObjective,
                DEFAULT_TRAIN_ROUNDS,
            )
            .expect("engine training succeeds");
        let bridge_artifact = train_regression_artifact_with_summary_dense_impl(
            &dataset.matrix.values,
            dataset.row_count(),
            dataset.matrix.feature_count,
            &dataset.targets,
            None,
            None,
            dataset.factor_exposures.clone(),
            None,
            None,
            None,
            None,
            None,
            params,
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            vec![categorical_spec],
            Vec::new(),
            TrainingPolicyMode::Manual,
            false,
            ContinuousBinningStrategy::Linear,
            MAX_CONTINUOUS_QUANTIZED_BIN_U8 as usize + 1,
            "squared_error",
            None,
            None,
            None,
            None,
            None,
            0,
        )
        .expect("bridge training succeeds")
        .artifact_bytes;

        let engine_predictions = engine_model.predict_batch(&rows).expect("engine predicts");
        let bridge_predictions =
            predictor_predict_batch_impl(&bridge_artifact, &rows).expect("bridge predicts");
        assert_eq!(bridge_predictions, engine_predictions);
    }

    #[test]
    fn train_bridge_rejects_partial_categorical_arguments() {
        let rows = fixture_rows(&quality_fixture_dataset());

        let missing_values = resolve_categorical_spec(Some(1), None, 20.0, 1, false, 8);
        assert!(matches!(
            missing_values,
            Err(alloygbm_engine::EngineError::ContractViolation(_))
        ));

        let missing_index = resolve_categorical_spec(
            None,
            Some(fixture_categorical_values()),
            20.0,
            1,
            false,
            rows.len(),
        );
        assert!(matches!(
            missing_index,
            Err(alloygbm_engine::EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn shap_bridge_explain_rows_matches_model_additivity() {
        let (model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let artifact = model.to_artifact_bytes().expect("artifact serializes");

        let (expected_value, values) =
            shap_explain_rows_impl(&artifact, &rows).expect("shap bridge explains");
        assert_eq!(values.len(), rows.len());
        assert_eq!(values[0].len(), dataset.matrix.feature_count);

        let predictions = predictor_predict_batch_impl(&artifact, &rows).expect("predicts");
        for (row_values, prediction) in values.iter().zip(predictions.iter()) {
            let reconstructed = expected_value + row_values.iter().sum::<f32>();
            assert!((reconstructed - prediction).abs() <= 1e-5);
        }
    }

    #[test]
    fn shap_bridge_global_importance_is_sorted_descending() {
        let (_model, dataset) = train_fixture_model();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("bridge training succeeds");

        let global =
            shap_global_importance_impl(&artifact, &rows).expect("global importance computes");
        assert_eq!(global.len(), dataset.matrix.feature_count);
        for (name, value) in &global {
            assert!(name.starts_with('f'));
            assert!(*value >= 0.0);
        }
        for pair in global.windows(2) {
            assert!(pair[0].1 >= pair[1].1);
        }
    }

    #[test]
    fn train_bridge_accepts_continuous_float_rows() {
        let rows = vec![
            vec![-2.7, 0.10],
            vec![0.20, 1.90],
            vec![3.60, 2.20],
            vec![8.40, 5.50],
            vec![15.25, 9.10],
            vec![30.75, 12.80],
        ];
        let targets = vec![-2.0, -0.5, 0.5, 1.5, 3.0, 6.0];

        let artifact = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("continuous rows should train");

        let predictions =
            predictor_predict_batch_impl(&artifact, &rows).expect("continuous rows should predict");
        assert_eq!(predictions.len(), rows.len());
    }

    #[test]
    fn train_bridge_quantization_is_deterministic_for_continuous_rows() {
        let rows = vec![
            vec![-1.5, 0.25],
            vec![-0.6, 0.75],
            vec![0.4, 1.20],
            vec![1.4, 1.80],
            vec![2.6, 3.40],
            vec![5.9, 8.10],
        ];
        let targets = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

        let artifact_a = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("first deterministic training succeeds");
        let artifact_b = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("second deterministic training succeeds");

        assert_eq!(artifact_a, artifact_b);
    }

    #[test]
    fn train_bridge_pre_binned_path_rejects_u16_overflow() {
        let rows = vec![vec![70000.0, 0.0], vec![1.0, 0.0]];
        let targets = vec![0.0, 1.0];
        let result = train_regression_artifact_impl(
            &rows,
            &targets,
            fixture_params(),
            1,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        );
        assert!(matches!(
            result,
            Err(alloygbm_engine::EngineError::ContractViolation(message))
            if message.contains("exceeds max supported bin")
        ));
    }

    #[test]
    fn train_bridge_large_round_counts_remain_supported_via_round_cap() {
        let dataset = quality_fixture_dataset();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            4096,
            None,
            None,
            TrainingPolicyMode::Manual,
            false,
        )
        .expect("max supported round count should train");
        assert!(!artifact.is_empty());
    }

    #[test]
    fn train_bridge_can_store_node_debug_stats_section() {
        let dataset = quality_fixture_dataset();
        let rows = fixture_rows(&dataset);
        let artifact = train_regression_artifact_impl(
            &rows,
            &dataset.targets,
            fixture_params(),
            DEFAULT_TRAIN_ROUNDS,
            None,
            None,
            TrainingPolicyMode::Manual,
            true,
        )
        .expect("bridge training succeeds");
        let parsed = deserialize_model_artifact_v1(&artifact).expect("artifact parses");
        assert!(
            parsed
                .sections
                .iter()
                .any(|section| section.descriptor.kind == ModelSectionKind::NodeDebugStats)
        );
    }
}
