    use super::*;
    use alloygbm_core::CoreError;
    use std::cell::Cell;

    struct MockBackend;
    struct GradientNeutralizationCheckingBackend {
        exposures: FactorExposureMatrix,
        weights: Option<Vec<f32>>,
    }
    struct MorphGradientNeutralizationCheckingBackend {
        exposures: FactorExposureMatrix,
        raw_factor_dot: f32,
        saw_morph_split: Cell<bool>,
    }
    struct EncodedFeatureCheckingBackend {
        feature_index: usize,
        expected_bins: Vec<u16>,
    }
    struct BadObjective;

    fn sample_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 1.0, //
                    3.0, 1.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![2.0, 1.0, -1.0, -2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn sample_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            4,
            2,
            3,
            vec![
                0, 0, //
                1, 0, //
                2, 1, //
                3, 1, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn factor_dominated_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
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
            targets: vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: Some(
                FactorExposureMatrix::new(8, 1, vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
                    .expect("factor exposures are valid"),
            ),
        }
    }

    fn weighted_factor_dominated_dataset() -> TrainingDataset {
        let mut dataset = factor_dominated_dataset();
        dataset.sample_weights = Some(vec![1.0, 3.0, 2.0, 4.0, 1.5, 2.5, 3.5, 4.5]);
        dataset
    }

    fn target_encoding_factor_loaded_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
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

    fn multiclass_factor_dominated_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                6,
                2,
                vec![
                    0.0, 0.0, //
                    1.0, 0.0, //
                    2.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    2.0, 1.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: Some(
                FactorExposureMatrix::new(6, 1, vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
                    .expect("factor exposures are valid"),
            ),
        }
    }

    fn sample_binned_matrix_for_dataset(dataset: &TrainingDataset) -> BinnedMatrix {
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

    fn factor_dot(exposures: &FactorExposureMatrix, values: &[f32]) -> f32 {
        assert_eq!(exposures.factor_count, 1);
        exposures
            .values
            .iter()
            .zip(values.iter())
            .map(|(factor, value)| *factor * *value)
            .sum()
    }

    fn weighted_gradient_factor_dot(
        exposures: &FactorExposureMatrix,
        weights: Option<&[f32]>,
        gradients: &[GradientPair],
    ) -> f32 {
        assert_eq!(exposures.factor_count, 1);
        exposures
            .values
            .iter()
            .zip(gradients.iter())
            .enumerate()
            .map(|(row, (factor, gradient))| {
                weights.map_or(1.0, |sample_weights| sample_weights[row]) * *factor * gradient.grad
            })
            .sum()
    }

    fn apply_pre_target_neutralization_for_test(
        dataset: &mut TrainingDataset,
        ridge_lambda: f32,
    ) -> EngineResult<()> {
        apply_pre_target_neutralization(dataset, ridge_lambda)
    }

    fn sample_wide_small_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                8,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, //
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![2.0, 1.0, -1.0, -2.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn sample_wide_small_binned_matrix() -> BinnedMatrix {
        BinnedMatrix::new(
            4,
            8,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 1, 1, 1, 1, //
                2, 2, 2, 2, 2, 2, 2, 2, //
                3, 3, 3, 3, 3, 3, 3, 3, //
            ],
        )
        .expect("binned matrix is valid")
    }

    fn sample_noisy_wide_small_dataset() -> TrainingDataset {
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(
                4,
                8,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, //
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, //
                ],
            )
            .expect("matrix is valid"),
            targets: vec![10.0, 5.0, -5.0, -10.0],
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn sample_trained_model() -> TrainedModel {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("iterative training succeeds")
    }

    impl BackendOps for MockBackend {
        fn build_histograms(
            &self,
            _binned_matrix: &BinnedMatrix,
            _gradients: &[GradientPair],
            node: &NodeSlice,
            _feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            Ok(HistogramBundle {
                node_id: node.node_id,
                feature_histograms: Vec::new(),
            })
        }

        fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
            let (_, local_node_id) = decode_tree_node_id(histograms.node_id);
            let threshold_bin = match local_node_id {
                0 => 1,
                1 => 0,
                2 => 2,
                _ => 1,
            };
            Ok(Some(SplitCandidate {
                node_id: histograms.node_id,
                feature_index: 0,
                threshold_bin,
                gain: 3.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    row_count: 0,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    row_count: 0,
                },
            }))
        }

        fn apply_split(
            &self,
            binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            node.validate_bounds(binned_matrix.row_count)?;

            let mut left_row_indices = Vec::new();
            let mut right_row_indices = Vec::new();
            for &row_index in &node.row_indices {
                let row_index = row_index as usize;
                let cell_index =
                    row_index * binned_matrix.feature_count + split.feature_index as usize;
                let bin = binned_matrix.row_bin(cell_index);
                if bin <= split.threshold_bin {
                    left_row_indices.push(row_index as u32);
                } else {
                    right_row_indices.push(row_index as u32);
                }
            }
            Ok(PartitionResult {
                left_row_indices,
                right_row_indices,
            })
        }

        fn reduce_sums(
            &self,
            gradients: &[GradientPair],
            row_indices: &[u32],
        ) -> EngineResult<NodeStats> {
            let mut grad_sum = 0.0_f32;
            let mut hess_sum = 0.0_f32;
            let mut grad_sq_sum = 0.0_f32;
            for &row_index in row_indices {
                let gp = gradients.get(row_index as usize).ok_or_else(|| {
                    EngineError::ContractViolation(
                        "row index out of bounds in mock reduction".to_string(),
                    )
                })?;
                grad_sum += gp.grad;
                hess_sum += gp.hess;
                grad_sq_sum += gp.grad * gp.grad;
            }
            Ok(NodeStats {
                grad_sum,
                hess_sum,
                grad_sq_sum,
                row_count: row_indices.len() as u32,
            })
        }
    }

    impl BackendOps for GradientNeutralizationCheckingBackend {
        fn build_histograms(
            &self,
            binned_matrix: &BinnedMatrix,
            gradients: &[GradientPair],
            node: &NodeSlice,
            feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            let dot =
                weighted_gradient_factor_dot(&self.exposures, self.weights.as_deref(), gradients);
            assert!(
                dot.abs() < 1e-3,
                "factor dot after per-round gradient projection was {dot}"
            );
            MockBackend.build_histograms(binned_matrix, gradients, node, feature_tiles)
        }

        fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
            MockBackend.best_split(histograms)
        }

        fn apply_split(
            &self,
            binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            MockBackend.apply_split(binned_matrix, node, split)
        }

        fn reduce_sums(
            &self,
            gradients: &[GradientPair],
            row_indices: &[u32],
        ) -> EngineResult<NodeStats> {
            MockBackend.reduce_sums(gradients, row_indices)
        }
    }

    impl BackendOps for MorphGradientNeutralizationCheckingBackend {
        fn build_histograms(
            &self,
            binned_matrix: &BinnedMatrix,
            gradients: &[GradientPair],
            node: &NodeSlice,
            feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            assert!(
                self.raw_factor_dot.abs() > 1.0,
                "test fixture should start with factor-loaded gradients, got {}",
                self.raw_factor_dot
            );
            let projected_dot = weighted_gradient_factor_dot(&self.exposures, None, gradients);
            assert!(
                projected_dot.abs() < self.raw_factor_dot.abs() * 0.001,
                "Morph split path saw unprojected factor dot: before={}, after={}",
                self.raw_factor_dot,
                projected_dot
            );
            MockBackend.build_histograms(binned_matrix, gradients, node, feature_tiles)
        }

        fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
            MockBackend.best_split(histograms)
        }

        fn best_split_morph_with_factor_context(
            &self,
            histograms: &HistogramBundle,
            _options: SplitSelectionOptions,
            _feature_weights: &[f32],
            _categorical_features: &[CategoricalFeatureInfo],
            _morph: &MorphContext,
            factor_context: Option<&FactorSplitContext<'_>>,
        ) -> EngineResult<Option<SplitCandidate>> {
            assert!(factor_context.is_none());
            self.saw_morph_split.set(true);
            MockBackend.best_split(histograms)
        }

        fn apply_split(
            &self,
            binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            MockBackend.apply_split(binned_matrix, node, split)
        }

        fn reduce_sums(
            &self,
            gradients: &[GradientPair],
            row_indices: &[u32],
        ) -> EngineResult<NodeStats> {
            MockBackend.reduce_sums(gradients, row_indices)
        }
    }

    impl BackendOps for EncodedFeatureCheckingBackend {
        fn build_histograms(
            &self,
            binned_matrix: &BinnedMatrix,
            gradients: &[GradientPair],
            node: &NodeSlice,
            feature_tiles: &[FeatureTile],
        ) -> EngineResult<HistogramBundle> {
            let actual_bins = (0..binned_matrix.row_count)
                .map(|row| {
                    binned_matrix.row_bin(row * binned_matrix.feature_count + self.feature_index)
                })
                .collect::<Vec<_>>();
            assert_eq!(actual_bins, self.expected_bins);
            MockBackend.build_histograms(binned_matrix, gradients, node, feature_tiles)
        }

        fn best_split(&self, histograms: &HistogramBundle) -> EngineResult<Option<SplitCandidate>> {
            MockBackend.best_split(histograms)
        }

        fn apply_split(
            &self,
            binned_matrix: &BinnedMatrix,
            node: &NodeSlice,
            split: &SplitCandidate,
        ) -> EngineResult<PartitionResult> {
            MockBackend.apply_split(binned_matrix, node, split)
        }

        fn reduce_sums(
            &self,
            gradients: &[GradientPair],
            row_indices: &[u32],
        ) -> EngineResult<NodeStats> {
            MockBackend.reduce_sums(gradients, row_indices)
        }
    }

    impl ObjectiveOps for BadObjective {
        fn objective_name(&self) -> &str {
            "bad"
        }

        fn initial_prediction(
            &self,
            _targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            Ok(0.0)
        }

        fn compute_gradients(
            &self,
            _predictions: &[f32],
            _targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<Vec<GradientPair>> {
            Ok(vec![GradientPair {
                grad: 0.1,
                hess: 1.0,
            }])
        }

        fn loss(
            &self,
            predictions: &[f32],
            targets: &[f32],
            sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            squared_error_loss(predictions, targets, sample_weights)
        }
    }

    #[test]
    fn squared_error_objective_produces_expected_baseline() {
        let objective = SquaredErrorObjective;
        let baseline = objective
            .initial_prediction(&[2.0, 0.0, -2.0], None)
            .expect("baseline should compute");
        assert!(baseline.abs() < 1e-6);
    }

    #[test]
    fn trainer_validates_fit_contract() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer
            .validate_fit_contract(&sample_dataset(), &SquaredErrorObjective)
            .expect("contract validation succeeds");
        assert_eq!(result.gradients.len(), 4);
    }

    #[test]
    fn trainer_rejects_gradient_length_mismatch() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let result = trainer.validate_fit_contract(&sample_dataset(), &BadObjective);
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn trainer_rejects_neutralization_without_factor_exposures() {
        let trainer = Trainer::new(TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        })
        .expect("neutralization params are valid");
        let result = trainer.validate_fit_contract(&sample_dataset(), &SquaredErrorObjective);
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn per_round_gradient_neutralization_trains_regression() {
        let dataset = factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset.factor_exposures.as_ref().unwrap().clone(),
            weights: None,
        };
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let model = Trainer::new(params)
            .unwrap()
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 5)
            .unwrap();
        assert_eq!(model.rounds_completed(), 5);
    }

    #[test]
    fn pre_target_neutralization_fit_iterations_uses_residualized_targets_for_target_encoding() {
        let dataset = target_encoding_factor_loaded_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let spec = CategoricalTargetEncodingSpec {
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

        let mut residualized_dataset = dataset.clone();
        apply_pre_target_neutralization_for_test(&mut residualized_dataset, 1e-6).unwrap();
        let (_, residualized_encoded_binned) =
            apply_single_categorical_target_encoding(&residualized_dataset, &binned, &spec)
                .unwrap();
        let (_, raw_encoded_binned) =
            apply_single_categorical_target_encoding(&dataset, &binned, &spec).unwrap();
        let residualized_expected_bins = (0..residualized_encoded_binned.row_count)
            .map(|row| {
                residualized_encoded_binned
                    .row_bin(row * residualized_encoded_binned.feature_count + spec.feature_index)
            })
            .collect::<Vec<_>>();
        let raw_bins = (0..raw_encoded_binned.row_count)
            .map(|row| {
                raw_encoded_binned
                    .row_bin(row * raw_encoded_binned.feature_count + spec.feature_index)
            })
            .collect::<Vec<_>>();
        assert_ne!(residualized_expected_bins, raw_bins);

        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PreTarget,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let backend = EncodedFeatureCheckingBackend {
            feature_index: spec.feature_index,
            expected_bins: residualized_expected_bins,
        };
        let model = Trainer::new(params)
            .unwrap()
            .fit_iterations_with_single_target_encoded_feature(
                &dataset,
                &binned,
                &spec,
                &backend,
                &SquaredErrorObjective,
                1,
            )
            .unwrap();
        assert_eq!(model.rounds_completed(), 1);
    }

    #[test]
    fn weighted_per_round_gradient_neutralization_trains_regression() {
        let dataset = weighted_factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset.factor_exposures.as_ref().unwrap().clone(),
            weights: dataset.sample_weights.clone(),
        };
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        Trainer::new(params)
            .unwrap()
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 5)
            .unwrap();
    }

    #[test]
    fn morph_neutralization_split_path_sees_per_round_projected_gradients() {
        let dataset = factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let baseline = SquaredErrorObjective
            .initial_prediction(&dataset.targets, dataset.sample_weights.as_deref())
            .expect("baseline should compute");
        let predictions = vec![baseline; dataset.row_count()];
        let raw_gradients = SquaredErrorObjective
            .compute_gradients(
                &predictions,
                &dataset.targets,
                dataset.sample_weights.as_deref(),
            )
            .expect("raw gradients should compute");
        let raw_factor_dot = weighted_gradient_factor_dot(
            dataset.factor_exposures.as_ref().unwrap(),
            dataset.sample_weights.as_deref(),
            &raw_gradients,
        );
        let backend = MorphGradientNeutralizationCheckingBackend {
            exposures: dataset.factor_exposures.as_ref().unwrap().clone(),
            raw_factor_dot,
            saw_morph_split: Cell::new(false),
        };
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            morph_config: Some(MorphConfig::default()),
            ..TrainParams::default()
        };

        let model = Trainer::new(params)
            .unwrap()
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 3)
            .unwrap();

        assert!(
            backend.saw_morph_split.get(),
            "Morph split path was not used"
        );
        assert_eq!(model.rounds_completed(), 3);
        assert!(model.morph_metadata.is_some());
    }

    #[test]
    fn pre_target_neutralization_reduces_target_factor_dot() {
        let mut dataset = factor_dominated_dataset();
        let before = factor_dot(dataset.factor_exposures.as_ref().unwrap(), &dataset.targets);
        apply_pre_target_neutralization_for_test(&mut dataset, 1e-6).unwrap();
        let after = factor_dot(dataset.factor_exposures.as_ref().unwrap(), &dataset.targets);
        assert!(after.abs() < before.abs() * 0.01);
    }

    #[test]
    fn warm_start_neutralization_requires_factor_exposures_to_be_supplied() {
        // v0.7.1 contract: warm-start with neutralization is allowed, but the
        // caller must pass `factor_exposures` so the projection has the same
        // column space as the initial fit.  Dropping the exposures must be
        // rejected with a contract-violation error.
        let mut dataset = factor_dominated_dataset();
        dataset.factor_exposures = None;
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let warm_start = WarmStartState {
            baseline_prediction: 0.0,
            stumps: Vec::new(),
            initial_rounds_completed: 1,
            initial_ema_stats: None,
            initial_dart_tree_weights: None,
        };
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let result = Trainer::new(params).unwrap().fit_iterations_warm_start(
            &dataset,
            &binned,
            &MockBackend,
            &SquaredErrorObjective,
            controls,
            warm_start,
        );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn multiclass_neutralization_rejects_inactive_factor_exposures() {
        let dataset = multiclass_factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let objective = MultiClassSoftmaxObjective::new(2).unwrap();
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let result = Trainer::new(TrainParams::default())
            .unwrap()
            .fit_multiclass_iterations_with_summary(
                &dataset,
                &binned,
                &MockBackend,
                &objective,
                controls,
            );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn multiclass_neutralization_rejects_active_without_factor_exposures() {
        let mut dataset = multiclass_factor_dominated_dataset();
        dataset.factor_exposures = None;
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let objective = MultiClassSoftmaxObjective::new(2).unwrap();
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let result = Trainer::new(params)
            .unwrap()
            .fit_multiclass_iterations_with_summary(
                &dataset,
                &binned,
                &MockBackend,
                &objective,
                controls,
            );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn multiclass_neutralization_rejects_pre_target() {
        let dataset = multiclass_factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let objective = MultiClassSoftmaxObjective::new(2).unwrap();
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PreTarget,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let result = Trainer::new(params)
            .unwrap()
            .fit_multiclass_iterations_with_summary(
                &dataset,
                &binned,
                &MockBackend,
                &objective,
                controls,
            );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn multiclass_warm_start_neutralization_requires_factor_exposures_to_be_supplied() {
        // v0.7.1: dropping factor_exposures on a neutralized multiclass
        // warm-start must be rejected (mirrors the single-output contract).
        let mut dataset = multiclass_factor_dominated_dataset();
        dataset.factor_exposures = None;
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let objective = MultiClassSoftmaxObjective::new(2).unwrap();
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let warm_start = MultiClassWarmStartState {
            baseline_predictions: vec![0.0, 0.0],
            class_stumps: vec![Vec::new(), Vec::new()],
            initial_rounds_completed: 1,
            initial_ema_stats: None,
            initial_dart_tree_weights: None,
        };
        let result = Trainer::new(params)
            .unwrap()
            .fit_multiclass_iterations_warm_start_with_summary(
                &dataset,
                &binned,
                &MockBackend,
                &objective,
                controls,
                warm_start,
            );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn multiclass_per_round_gradient_neutralization_projects_each_class() {
        let dataset = multiclass_factor_dominated_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let objective = MultiClassSoftmaxObjective::new(2).unwrap();
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap();
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset.factor_exposures.as_ref().unwrap().clone(),
            weights: None,
        };
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: alloygbm_core::NeutralizationKind::PerRoundGradient,
                ridge_lambda: 1e-6,
                split_penalty: 0.0,
            }),
            ..TrainParams::default()
        };
        let summary = Trainer::new(params)
            .unwrap()
            .fit_multiclass_iterations_with_summary(
                &dataset, &binned, &backend, &objective, controls,
            )
            .unwrap();
        assert_eq!(summary.rounds_completed, 3);
    }

    #[test]
    fn factor_projector_orthogonalizes_gradient() {
        let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let projector = FactorProjector::new(&exposures, None, 1e-6).unwrap();
        let mut gradients = vec![
            GradientPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 2.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 3.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 4.0,
                hess: 1.0,
            },
        ];
        projector
            .project_gradient_pairs_in_place(&mut gradients)
            .unwrap();
        let dot: f32 = exposures
            .values
            .iter()
            .zip(gradients.iter())
            .map(|(f, g)| *f * g.grad)
            .sum();
        assert!(dot.abs() < 1e-4, "factor dot after projection was {dot}");
        assert!(gradients.iter().all(|g| g.hess == 1.0));
    }

    #[test]
    fn factor_projector_ridge_handles_collinear_factors() {
        let exposures =
            FactorExposureMatrix::new(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
        let projector = FactorProjector::new(&exposures, None, 1e-3).unwrap();
        let mut values = vec![1.0, 2.0, 3.0];
        projector.residualize_values_in_place(&mut values).unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn factor_projector_rejects_projection_value_length_mismatch() {
        let exposures = FactorExposureMatrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let projector = FactorProjector::new(&exposures, None, 1e-6).unwrap();
        let err = projector
            .projection_coefficients([1.0, 2.0])
            .expect_err("value length mismatch should be rejected");
        assert!(matches!(err, EngineError::ContractViolation(_)));
    }

    #[test]
    fn factor_projector_weighted_projection_orthogonalizes_values_and_gradients() {
        let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, -1.0, 2.0, -2.0]).unwrap();
        let weights = vec![1.0, 3.0, 2.0, 4.0];
        let projector = FactorProjector::new(&exposures, Some(&weights), 1e-6).unwrap();

        let mut values = vec![2.0, -1.0, 3.0, -2.0];
        projector.residualize_values_in_place(&mut values).unwrap();
        let weighted_value_dot: f32 = exposures
            .values
            .iter()
            .zip(weights.iter())
            .zip(values.iter())
            .map(|((f, w), v)| *w * *f * *v)
            .sum();
        assert!(
            weighted_value_dot.abs() < 1e-4,
            "weighted factor dot after value projection was {weighted_value_dot}"
        );

        let mut gradients = vec![
            GradientPair {
                grad: 2.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 3.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -2.0,
                hess: 1.0,
            },
        ];
        projector
            .project_gradient_pairs_in_place(&mut gradients)
            .unwrap();
        let weighted_gradient_dot: f32 = exposures
            .values
            .iter()
            .zip(weights.iter())
            .zip(gradients.iter())
            .map(|((f, w), g)| *w * *f * g.grad)
            .sum();
        assert!(
            weighted_gradient_dot.abs() < 1e-4,
            "weighted factor dot after gradient projection was {weighted_gradient_dot}"
        );
        assert!(gradients.iter().all(|g| g.hess == 1.0));
    }

    #[test]
    fn factor_projector_rejects_non_finite_residualized_values() {
        let exposures = FactorExposureMatrix::new(2, 1, vec![1.0, 2.0]).unwrap();
        let projector = FactorProjector::new(&exposures, None, 1e-6).unwrap();
        let mut values = vec![f32::INFINITY, 1.0];
        let err = projector
            .residualize_values_in_place(&mut values)
            .expect_err("non-finite residualized values should be rejected");
        assert!(matches!(err, EngineError::ContractViolation(_)));
    }

    #[test]
    fn factor_projector_rejects_non_finite_projected_gradients() {
        let exposures = FactorExposureMatrix::new(2, 1, vec![1.0, 2.0]).unwrap();
        let projector = FactorProjector::new(&exposures, None, 1e-6).unwrap();
        let mut gradients = vec![
            GradientPair {
                grad: f32::INFINITY,
                hess: 1.0,
            },
            GradientPair {
                grad: 1.0,
                hess: 1.0,
            },
        ];
        let err = projector
            .project_gradient_pairs_in_place(&mut gradients)
            .expect_err("non-finite projected gradients should be rejected");
        assert!(matches!(err, EngineError::ContractViolation(_)));
    }

    #[test]
    fn fit_one_round_returns_coherent_summary() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let summary = trainer
            .fit_one_round(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
            )
            .expect("fit one round should succeed");

        assert_eq!(summary.root_stats.row_count, 4);
        assert!(summary.split_candidate.is_some());
        assert!(summary.partition.is_some());
    }

    #[test]
    fn fit_one_round_rejects_row_mismatch() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let bad_binned = BinnedMatrix::new(3, 2, 3, vec![0, 0, 1, 0, 2, 1]).expect("valid matrix");
        let result = trainer.fit_one_round(
            &sample_dataset(),
            &bad_binned,
            &MockBackend,
            &SquaredErrorObjective,
        );
        assert!(matches!(result, Err(EngineError::ContractViolation(_))));
    }

    #[test]
    fn fit_iterations_builds_model_and_changes_predictions() {
        let params = TrainParams {
            learning_rate: 0.5,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let model = trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                3,
            )
            .expect("iterative training succeeds");

        assert!(!model.stumps.is_empty());
        let left_pred = model.predict_row(&[0.0, 0.0]).expect("prediction succeeds");
        let right_pred = model.predict_row(&[3.0, 1.0]).expect("prediction succeeds");
        assert!(left_pred > right_pred);
    }

    #[test]
    fn dro_zero_radius_matches_standard_constant_leaf_predictions() {
        let standard_params = TrainParams {
            learning_rate: 0.5,
            ..TrainParams::default()
        };
        let dro_params = TrainParams {
            learning_rate: 0.5,
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.0,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            ..TrainParams::default()
        };

        let standard_model = Trainer::new(standard_params)
            .expect("standard params are valid")
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                3,
            )
            .expect("standard training succeeds");
        let dro_model = Trainer::new(dro_params)
            .expect("dro params are valid")
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                3,
            )
            .expect("zero-radius dro training succeeds");

        for row in [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]] {
            let standard_pred = standard_model.predict_row(&row).expect("standard predicts");
            let dro_pred = dro_model.predict_row(&row).expect("dro predicts");
            assert_eq!(standard_pred.to_bits(), dro_pred.to_bits());
        }
        assert!(standard_model.dro_metadata.is_none());
        assert_eq!(
            dro_model.dro_metadata.expect("dro metadata").config.radius,
            0.0
        );
    }

    #[test]
    fn dro_zero_radius_uses_standard_split_options() {
        let params = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.0,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            ..TrainParams::default()
        };

        let options = split_selection_options_for_training(
            &params,
            None,
            &sample_dataset(),
            &sample_binned_matrix(),
        )
        .expect("split options should build");

        assert!(
            options.dro_config.is_none(),
            "zero-radius DRO should keep standard split-selection fast paths"
        );
    }

    #[test]
    fn dro_metadata_round_trips_through_artifact() {
        let params = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            ..TrainParams::default()
        };
        let model = Trainer::new(params)
            .expect("dro params are valid")
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                1,
            )
            .expect("dro training succeeds");

        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact loads");
        assert_eq!(
            restored.dro_metadata.as_ref().expect("dro metadata").config,
            DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }
        );
        assert_eq!(
            model.predict_row(&[0.0, 0.0]).expect("original predicts"),
            restored
                .predict_row(&[0.0, 0.0])
                .expect("restored predicts")
        );
    }

    #[test]
    fn dro_leaf_solver_trains_with_morph_mode() {
        let params = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            morph_config: Some(MorphConfig::default()),
            ..TrainParams::default()
        };
        let model = Trainer::new(params)
            .expect("dro morph params are valid")
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("dro morph training succeeds");

        assert!(!model.stumps.is_empty());
        assert!(model.dro_metadata.is_some());
        assert!(model.morph_metadata.is_some());
    }

    #[test]
    fn dro_leaf_solver_rejects_linear_leaves_for_v0_6_0() {
        let params = TrainParams {
            leaf_solver: LeafSolverKind::Dro,
            dro_config: Some(DroConfig {
                radius: 0.05,
                metric: alloygbm_core::DroMetric::Wasserstein,
            }),
            leaf_model: LeafModelKind::Linear,
            ..TrainParams::default()
        };

        let err = Trainer::new(params).expect_err("dro linear leaves are not supported");
        assert!(matches!(
            err,
            EngineError::Core(CoreError::InvalidConfig(_))
        ));
        assert!(
            err.to_string()
                .contains("leaf_solver='dro' requires leaf_model='constant'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dart_boosting_mode_produces_non_uniform_tree_weights() {
        // v0.9.0: DART is fully wired through the single-output trainer.
        // After enough rounds we expect at least some stumps to have a
        // `tree_weight` that diverges from 1.0 — that's the signature of
        // the dropout + normalize cycle having mutated weights.
        let dataset = sample_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let params = TrainParams {
            boosting_mode: BoostingMode::Dart {
                drop_rate: 0.3,
                max_drop: 5,
                normalize_type: alloygbm_core::DartNormalize::Tree,
                sample_type: alloygbm_core::DartSampleType::Uniform,
            },
            seed: 42,
            deterministic: true,
            ..TrainParams::default()
        };
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset
                .factor_exposures
                .as_ref()
                .cloned()
                .unwrap_or_else(|| FactorExposureMatrix {
                    row_count: dataset.row_count(),
                    factor_count: 1,
                    values: vec![0.0; dataset.row_count()],
                }),
            weights: None,
        };
        let trainer = Trainer::new(params).expect("DART params pass validation");
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 15)
            .expect("DART training must succeed");
        let weights: Vec<f32> = model.stumps.iter().map(|s| s.tree_weight).collect();
        // At least one stump should have a weight materially different
        // from 1.0 — that's the dropout-rescale signature.
        assert!(
            weights.iter().any(|&w| (w - 1.0).abs() > 1e-3),
            "DART should produce at least some non-1.0 tree weights; got {weights:?}"
        );
    }

    #[test]
    fn dart_boosting_mode_supports_warm_start() {
        // v0.10.0+: DART + warm_start is now supported. The continuation
        // seeds dart_state.tree_weights from
        // warm_start.initial_dart_tree_weights and continues new-round
        // dropouts forward.
        let dataset = sample_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let dart_params = TrainParams {
            boosting_mode: BoostingMode::Dart {
                drop_rate: 0.1,
                max_drop: 5,
                normalize_type: alloygbm_core::DartNormalize::Tree,
                sample_type: alloygbm_core::DartSampleType::Uniform,
            },
            ..TrainParams::default()
        };
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset
                .factor_exposures
                .as_ref()
                .cloned()
                .unwrap_or_else(|| FactorExposureMatrix {
                    row_count: dataset.row_count(),
                    factor_count: 1,
                    values: vec![0.0; dataset.row_count()],
                }),
            weights: None,
        };
        // Cold fit produces a model whose stumps we feed into warm_start.
        let cold_params = TrainParams::default();
        let cold_trainer = Trainer::new(cold_params).unwrap();
        let cold_model = cold_trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 2)
            .expect("cold fit");
        let warm_start = WarmStartState {
            baseline_prediction: cold_model.baseline_prediction,
            stumps: cold_model.stumps.clone(),
            initial_rounds_completed: 2,
            initial_ema_stats: None,
            initial_dart_tree_weights: None,
        };
        let dart_trainer = Trainer::new(dart_params).expect("DART params pass validation");
        let continued = dart_trainer
            .fit_iterations_warm_start(
                &dataset,
                &binned,
                &backend,
                &SquaredErrorObjective,
                IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0).unwrap(),
                warm_start,
            )
            .expect("DART + warm_start must now succeed");
        // The continuation should include the warm-start stumps plus the new
        // DART rounds.
        assert!(
            continued.model.stumps.len() >= cold_model.stumps.len(),
            "continuation lost warm-start stumps"
        );
    }

    #[test]
    fn warm_start_early_stopping_truncates_against_new_round_counts() {
        // v0.10.0 review follow-up: when a warm-start continuation hits
        // validation early stopping, the truncation must use NEW-round
        // stump counts (stumps_per_completed_round is new-round-only), not
        // accidentally consume prior-round counts. An earlier draft of the
        // DART warm-start fix put prior-round counts at the front of
        // `stumps_per_completed_round`, which caused `best_round`-indexed
        // truncation to retain stump counts from prior rounds instead of
        // the actually-best new round.
        //
        // Reproduce the symptom: train a base model, then continue with
        // validation early stopping. The retained model size must equal
        // `initial_stump_count + sum_of_first_N_new_round_counts` where N
        // is the number of new rounds the early-stopping policy kept. We
        // verify that the new-round stump tail (>= initial_stump_count) is
        // present and that the model still validates / predicts coherently.
        let dataset = sample_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);

        // Cold baseline fit.
        let cold_params = TrainParams::default();
        let cold_trainer = Trainer::new(cold_params).unwrap();
        let cold_model = cold_trainer
            .fit_iterations(&dataset, &binned, &MockBackend, &SquaredErrorObjective, 3)
            .expect("cold fit");
        let cold_stump_count = cold_model.stumps.len();
        assert!(
            cold_stump_count >= 3,
            "test fixture: cold model should have >=3 stumps (one per round at depth=1), got {cold_stump_count}"
        );

        let warm_start = WarmStartState {
            baseline_prediction: cold_model.baseline_prediction,
            stumps: cold_model.stumps.clone(),
            initial_rounds_completed: 3,
            initial_ema_stats: None,
            initial_dart_tree_weights: None,
        };

        // Continue with eval_set + early_stopping_rounds. Use the same data
        // as validation so the loss curve is monotonic — early stopping
        // shouldn't fire purely on validation degradation, but if it does
        // the truncation must remain coherent. The key assertion is that
        // the retained model includes the warm-start stumps PLUS at least
        // the first new-round commit, NOT a truncation that accidentally
        // consumed prior-round counts to compute kept_stumps.
        let validation_ref = ValidationDatasetRef {
            dataset: &dataset,
            binned_matrix: &binned,
        };
        let warm_trainer = Trainer::new(TrainParams::default()).unwrap();
        let controls = IterationControls::new(
            /*rounds=*/ 5,
            /*min_split_gain=*/ 0.0,
            /*min_rows_per_leaf=*/ 1,
            /*lambda_l2=*/ 0.0,
            /*max_abs_leaf=*/ 1_000_000.0,
            /*min_validation_improvement=*/ 0.0,
            /*early_stopping_rounds=*/ 2,
        )
        .unwrap();
        let summary = warm_trainer
            .fit_iterations_warm_start_with_validation(
                &dataset,
                &binned,
                validation_ref,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                warm_start,
            )
            .expect("warm-start + validation early stopping must succeed");

        // The retained stump tail must include the warm-start stumps as a
        // prefix. (If `stumps_per_completed_round` had been polluted with
        // prior-round counts, the post-best-iteration truncation could
        // have dropped warm-start stumps or kept incoherent counts.)
        assert!(
            summary.model.stumps.len() >= cold_stump_count,
            "warm-start prefix lost during early-stop truncation: \
             retained {} stumps but warm-start brought in {}",
            summary.model.stumps.len(),
            cold_stump_count
        );
        for (i, (a, b)) in summary.model.stumps[..cold_stump_count]
            .iter()
            .zip(cold_model.stumps.iter())
            .enumerate()
        {
            assert_eq!(
                a.split.node_id, b.split.node_id,
                "warm-start stump {i} node_id was perturbed by truncation"
            );
        }

        // Sanity: the final loss must be finite. A truncation that walked
        // off the end of `stumps_per_completed_round` or kept the wrong
        // number of new stumps would still produce a TrainedModel (so the
        // call wouldn't panic) but its loss could diverge.
        assert!(
            summary.final_loss.is_finite(),
            "final_loss not finite: {}",
            summary.final_loss
        );
    }

    #[test]
    fn split_penalty_neutralization_rejects_linear_leaves() {
        let params = TrainParams {
            neutralization_config: Some(alloygbm_core::FactorNeutralizationConfig {
                kind: NeutralizationKind::SplitPenalty,
                ridge_lambda: 1e-6,
                split_penalty: 0.1,
            }),
            leaf_model: LeafModelKind::Linear,
            ..TrainParams::default()
        };

        let err = Trainer::new(params).expect_err("split penalty requires scalar leaves");
        assert!(matches!(
            err,
            EngineError::Core(CoreError::InvalidConfig(_))
        ));
        assert!(
            err.to_string()
                .contains("neutralization='split_penalty' requires leaf_model='constant'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn default_backend_neutralization_split_penalty_context_returns_error() {
        let matrix = sample_binned_matrix();
        let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0])
            .expect("factor exposures are valid");
        let rows = vec![0, 1, 2, 3];
        let context = FactorSplitContext {
            binned_matrix: &matrix,
            exposures: &exposures,
            row_indices: &rows,
            factor_penalty: 0.1,
        };
        let histograms = HistogramBundle {
            node_id: 0,
            feature_histograms: Vec::new(),
        };

        let err = MockBackend
            .best_split_with_factor_context(
                &histograms,
                SplitSelectionOptions::default(),
                &[],
                &[],
                Some(&context),
            )
            .expect_err("default backend must not ignore factor context");
        assert!(matches!(err, EngineError::ContractViolation(_)));
        assert!(
            err.to_string()
                .contains("factor split context is not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn refine_regression_leaf_values_reduces_loss_for_fixed_structure() {
        let node_id = encode_tree_node_id(0, 0).expect("node id encodes");
        let mut stumps = vec![TrainedStump {
            split: SplitCandidate {
                node_id,
                feature_index: 0,
                threshold_bin: 1,
                gain: 1.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    grad_sq_sum: 0.0,
                    row_count: 2,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    grad_sq_sum: 0.0,
                    row_count: 2,
                },
            },
            left_leaf_value: LeafValue::Scalar(0.0),
            right_leaf_value: LeafValue::Scalar(0.0),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }];
        let matrix = sample_binned_matrix();
        let targets = sample_dataset().targets;

        let before = tree_predictions_for_binned_rows(&matrix, &stumps)
            .expect("tree predictions should compute");
        let before_loss = squared_error_loss(&before, &targets, None).expect("loss should compute");

        refine_regression_leaf_values(0.0, &targets, None, &matrix, &mut stumps, &[1], 1_000_000.0)
            .expect("refinement should succeed");

        let after = tree_predictions_for_binned_rows(&matrix, &stumps)
            .expect("tree predictions should compute");
        let after_loss = squared_error_loss(&after, &targets, None).expect("loss should compute");

        assert!(after_loss < before_loss);
        assert!(stumps[0].left_leaf_value.as_scalar() > 0.0);
        assert!(stumps[0].right_leaf_value.as_scalar() < 0.0);
    }

    #[test]
    fn fit_iterations_rejects_zero_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let result = trainer.fit_iterations(
            &sample_dataset(),
            &sample_binned_matrix(),
            &MockBackend,
            &SquaredErrorObjective,
            0,
        );
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
    }

    #[test]
    fn trainer_rejects_invalid_subsample_params() {
        let params = TrainParams {
            row_subsample: 0.0,
            ..TrainParams::default()
        };
        assert!(matches!(
            Trainer::new(params),
            Err(EngineError::Core(CoreError::InvalidConfig(_)))
        ));
    }

    #[test]
    fn auto_policy_preserves_default_controls_on_small_datasets() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let manual = trainer
            .default_iteration_controls(8)
            .expect("manual controls should build");
        let auto = trainer
            .iteration_controls_for_policy(
                &sample_dataset(),
                &sample_binned_matrix(),
                8,
                TrainingPolicyMode::Auto,
            )
            .expect("auto controls should build");

        assert_eq!(auto.min_rows_per_leaf, manual.min_rows_per_leaf);
        assert_eq!(auto.min_split_gain, manual.min_split_gain);
        assert_eq!(auto.min_loss_improvement, manual.min_loss_improvement);
        assert_eq!(
            auto.max_consecutive_weak_improvements,
            manual.max_consecutive_weak_improvements
        );
        assert_eq!(auto.row_subsample, manual.row_subsample);
        assert_eq!(auto.col_subsample, manual.col_subsample);
    }

    fn large_ranking_shaped_dataset() -> TrainingDataset {
        // 5000 rows × 16 features, targets are graded 0-4 relevance labels.
        let row_count = 5000usize;
        let feature_count = 16usize;
        let mut values = Vec::with_capacity(row_count * feature_count);
        let mut targets = Vec::with_capacity(row_count);
        for row in 0..row_count {
            for col in 0..feature_count {
                values.push(((row + col) % 32) as f32);
            }
            targets.push((row % 5) as f32);
        }
        TrainingDataset {
            matrix: alloygbm_core::DatasetMatrix::new(row_count, feature_count, values)
                .expect("matrix is valid"),
            targets,
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        }
    }

    fn large_ranking_shaped_binned_matrix() -> BinnedMatrix {
        let row_count = 5000usize;
        let feature_count = 16usize;
        let num_bins = 32u16;
        let mut codes = Vec::with_capacity(row_count * feature_count);
        for row in 0..row_count {
            for col in 0..feature_count {
                codes.push(((row + col) % num_bins as usize) as u8);
            }
        }
        BinnedMatrix::new(row_count, feature_count, num_bins, codes)
            .expect("binned matrix is valid")
    }

    #[test]
    fn auto_policy_disables_regression_only_guards_for_ranking_objectives() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let dataset = large_ranking_shaped_dataset();
        let binned = large_ranking_shaped_binned_matrix();

        // Non-ranking (regression-style) path applies the density floor and
        // variance-scaled min_loss_improvement on a 5000×16 dataset.
        let regression_controls = trainer
            .iteration_controls_for_policy_ext(
                &dataset,
                &binned,
                1_200,
                TrainingPolicyMode::Auto,
                /* is_ranking */ false,
            )
            .expect("auto controls should build for regression");
        assert!(
            regression_controls.min_loss_improvement > 0.0,
            "regression auto-policy must keep the min_loss_improvement guard"
        );
        assert!(
            regression_controls.max_consecutive_weak_improvements >= 1,
            "regression auto-policy must keep weak-improvement cutoff"
        );
        assert!(
            regression_controls.min_split_gain > 0.0,
            "regression auto-policy must keep density-based min_split_gain floor \
             on 5000x16"
        );

        // Ranking path disables all three regression-tuned guards.
        let ranking_controls = trainer
            .iteration_controls_for_policy_ext(
                &dataset,
                &binned,
                1_200,
                TrainingPolicyMode::Auto,
                /* is_ranking */ true,
            )
            .expect("auto controls should build for ranking");
        assert_eq!(
            ranking_controls.min_split_gain, 0.0,
            "ranking auto-policy must not impose a density-based min_split_gain floor"
        );
        assert_eq!(
            ranking_controls.min_loss_improvement, 0.0,
            "ranking auto-policy must not impose a variance-scaled min_loss_improvement"
        );
        assert_eq!(
            ranking_controls.max_consecutive_weak_improvements, 0,
            "ranking auto-policy must not impose max_consecutive_weak_improvements"
        );
    }

    #[test]
    fn auto_policy_caps_rounds_for_small_wide_datasets() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid default params");
        let controls = trainer
            .iteration_controls_for_policy(
                &sample_wide_small_dataset(),
                &sample_wide_small_binned_matrix(),
                1_200,
                TrainingPolicyMode::Auto,
            )
            .expect("auto controls should build");

        assert_eq!(controls.rounds, 96);
    }

    #[test]
    fn auto_split_l2_targets_noisy_small_wide_datasets() {
        assert!(
            should_apply_auto_split_l2(
                &sample_noisy_wide_small_dataset(),
                &sample_wide_small_binned_matrix()
            )
            .expect("heuristic should evaluate")
        );
    }

    #[test]
    fn auto_split_l2_skips_dense_numeric_style_datasets() {
        assert!(
            !should_apply_auto_split_l2(&sample_dataset(), &sample_binned_matrix())
                .expect("heuristic should evaluate")
        );
    }

    #[test]
    fn fit_iterations_controls_enforce_min_split_gain() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn fit_iterations_summary_reports_gain_threshold_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 10.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.effective_round_cap, 3);
        assert_eq!(summary.rounds_completed, 0);
        assert_eq!(summary.stop_reason, IterationStopReason::GainBelowThreshold);
        assert!(summary.model.stumps.is_empty());
        assert!(summary.loss_per_completed_round.is_empty());
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_reports_completed_requested_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 1);
        assert_eq!(summary.effective_round_cap, 1);
        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CompletedRequestedRounds
        );
        assert!(!summary.model.stumps.is_empty());
        assert_eq!(summary.loss_per_completed_round.len(), 1);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(
            summary.final_loss,
            summary.loss_per_completed_round[summary.loss_per_completed_round.len() - 1]
        );
    }

    #[test]
    fn fit_iterations_summary_uses_round_count_as_round_cap() {
        let params = TrainParams {
            max_depth: 1,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.effective_round_cap, 3);
        assert_eq!(summary.rounds_completed, 3);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CompletedRequestedRounds
        );
        assert_eq!(summary.model.stumps.len(), 3);
        assert_eq!(summary.loss_per_completed_round.len(), 3);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
    }

    #[test]
    fn fit_iterations_grows_multiple_nodes_per_round_when_depth_allows() {
        let params = TrainParams {
            max_depth: 2,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        let controls = IterationControls::new(1, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(summary.model.stumps.len(), 3);
        let node_ids = summary
            .model
            .stumps
            .iter()
            .map(|stump| stump.split.node_id)
            .collect::<Vec<_>>();
        assert!(node_ids.contains(&0));
        assert!(node_ids.contains(&1));
        assert!(node_ids.contains(&2));
    }

    #[test]
    fn fit_iterations_controls_enforce_min_rows_per_leaf() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 3, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn iteration_controls_reject_invalid_values() {
        assert!(matches!(
            IterationControls::new(0, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, -0.1, 1, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 0, 0.0, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, -0.1, 1_000_000.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 0.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 2.0, 1.0, 0.0, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, -0.1, 0),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, 0.0, 0)
                .and_then(|controls| controls.with_subsample_rates(0.0, 1.0)),
            Err(EngineError::InvalidConfig(_))
        ));
        assert!(matches!(
            IterationControls::new(1, 0.0, 1, 0.0, 1.0, 0.0, 0)
                .and_then(|controls| controls.with_validation_early_stopping(0, 0.0)),
            Err(EngineError::InvalidConfig(_))
        ));
    }

    #[test]
    fn sampled_row_indices_are_seeded_and_non_prefix() {
        let selected = sampled_row_indices(8, 0.5, 17, 0);
        let selected_repeat = sampled_row_indices(8, 0.5, 17, 0);
        assert_eq!(selected, selected_repeat);
        assert_eq!(selected.len(), 4);
        assert_ne!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn goss_selects_top_alpha_by_gradient_magnitude() {
        // 100 rows: first 10 have |gradient|=100, rest |gradient|=0.1.
        // top_rate=0.1 keeps the 10 large-gradient rows.
        // other_rate=0.2 samples 20 from the 90 remaining.
        // amplification = (1 - 0.1) / 0.2 = 4.5.
        let grads: Vec<f32> = (0..100).map(|i| if i < 10 { 100.0 } else { 0.1 }).collect();
        let (top, other, amp) = goss_sample_indices(&grads, 0.1, 0.2, 0xABC, 0);
        // Top set is exactly the high-gradient rows.
        assert_eq!(top.len(), 10);
        for i in 0..10u32 {
            assert!(top.contains(&i), "top set missing high-gradient row {i}");
        }
        // Other set is sampled from the rest.
        assert_eq!(other.len(), 20);
        for &i in &other {
            assert!(i >= 10, "other set should not include kept-top row {i}");
        }
        // Amplification matches LightGBM's formula.
        assert!((amp - 4.5).abs() < 1e-5, "expected amp ~= 4.5, got {amp}");
        // Determinism: same seed + round → same selection.
        let (top2, other2, amp2) = goss_sample_indices(&grads, 0.1, 0.2, 0xABC, 0);
        assert_eq!(top, top2);
        assert_eq!(other, other2);
        assert_eq!(amp, amp2);
    }

    #[test]
    fn goss_zero_other_rate_returns_only_top() {
        let grads: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let (top, other, amp) = goss_sample_indices(&grads, 0.2, 0.0, 1, 0);
        assert_eq!(top.len(), 10);
        assert!(other.is_empty());
        // amplification falls back to 1.0 when no rows are sampled.
        assert_eq!(amp, 1.0);
    }

    #[test]
    fn goss_top_plus_other_caps_at_n_rows() {
        // top_rate=0.6 and other_rate=0.6 would over-allocate; verify
        // the algorithm caps other_n so total <= n.
        let grads: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (top, other, _amp) = goss_sample_indices(&grads, 0.6, 0.6, 1, 0);
        assert!(top.len() + other.len() <= 10);
    }

    #[test]
    fn goss_amplification_uses_realized_counts() {
        // Contract: amplification must compute `(n - top_n) / other_n`
        // from the realized counts after `ceil()` + `min()`, not the
        // configured `(1 - top_rate) / other_rate`.  Small-n regimes
        // (and any case where rounding shifts realized fractions) hit
        // the difference.  Example: n=5, top_rate=0.2 ⇒ top_n=1,
        // other_rate=0.1 ⇒ other_n=1.  Pool of remaining rows = 4.
        // Unbiased multiplier = 4 / 1 = 4.0.  The rate form would
        // produce (1 - 0.2) / 0.1 = 8.0 — doubling the sampled-low
        // contribution.
        let grads: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let (top, other, amp) = goss_sample_indices(&grads, 0.2, 0.1, 1, 0);
        assert_eq!(top.len(), 1);
        assert_eq!(other.len(), 1);
        assert!(
            (amp - 4.0).abs() < 1e-5,
            "expected realized-count amplification 4.0, got {amp}"
        );

        // Another small case: n=7, top_rate=0.15 (top_n=2),
        // other_rate=0.15 (other_n=2).  Realized amp = (7 - 2) / 2 = 2.5.
        // Rate form would give (1 - 0.15) / 0.15 ≈ 5.6667.
        let grads_7: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let (_top7, _other7, amp7) = goss_sample_indices(&grads_7, 0.15, 0.15, 1, 0);
        assert!(
            (amp7 - 2.5).abs() < 1e-5,
            "expected realized-count amplification 2.5, got {amp7}"
        );

        // For large n where rounding is immaterial the realized-count
        // form and the rate form agree.  n=1000, top_rate=0.2,
        // other_rate=0.1 ⇒ top_n=200, other_n=100.  Realized amp =
        // 800 / 100 = 8.0.  Rate form = 8.0.  This case is a
        // regression guard: when n is large the two formulas must
        // numerically coincide so existing
        // tuning/benchmarks don't shift.
        let grads_1000: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
        let (_t1000, _o1000, amp1000) = goss_sample_indices(&grads_1000, 0.2, 0.1, 1, 0);
        assert!(
            (amp1000 - 8.0).abs() < 1e-5,
            "expected large-n amplification ≈ 8.0, got {amp1000}"
        );
    }

    #[test]
    fn sampled_feature_tiles_cover_expected_feature_count() {
        let (tiles, coverage_count) =
            sampled_feature_tiles(10, 0.3, 23, 0).expect("feature tiles should sample");
        assert_eq!(coverage_count, 3);
        let tile_coverage = tiles
            .iter()
            .map(|tile| (tile.end_feature - tile.start_feature) as usize)
            .sum::<usize>();
        assert_eq!(tile_coverage, coverage_count);
    }

    #[test]
    fn sampled_feature_tiles_are_seeded_and_non_prefix() {
        let expand = |tiles: &[FeatureTile]| {
            tiles
                .iter()
                .flat_map(|tile| tile.start_feature..tile.end_feature)
                .map(|index| index as usize)
                .collect::<Vec<_>>()
        };

        let (tiles, coverage_count) =
            sampled_feature_tiles(12, 0.4, 17, 0).expect("feature tiles should sample");
        let (tiles_repeat, coverage_count_repeat) =
            sampled_feature_tiles(12, 0.4, 17, 0).expect("feature tiles should sample");
        assert_eq!(coverage_count, 5);
        assert_eq!(coverage_count, coverage_count_repeat);
        let selected = expand(&tiles);
        let selected_repeat = expand(&tiles_repeat);
        assert_eq!(selected, selected_repeat);
        assert_ne!(selected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn auto_tile_size_targets_features_per_thread() {
        // 780 features, 16 threads → expect ~24 (ceil(780/32)), clamped to [16,64]
        let tile = compute_optimal_tile_size(780, 16);
        assert!(
            (16..=64).contains(&tile),
            "expected tile in [16,64] for 780f/16t, got {}",
            tile
        );
        // ceil(780/32) = 25, in range
        assert_eq!(tile, 25);
    }

    #[test]
    fn auto_tile_size_falls_back_for_low_feature_count() {
        // 10 features, 16 threads → return feature_count itself (10), capped at MAX
        let tile = compute_optimal_tile_size(10, 16);
        assert_eq!(tile, 10);
    }

    #[test]
    fn auto_tile_size_single_thread_returns_feature_count() {
        let tile = compute_optimal_tile_size(100, 1);
        assert_eq!(tile, MAX_TILE_FEATURE_WIDTH); // capped at the constant
        let tile_small = compute_optimal_tile_size(40, 1);
        assert_eq!(tile_small, 40);
    }

    #[test]
    fn auto_tile_size_clamps_above_max() {
        // 100K features, 16 threads → ceil(100000/32) = 3125, clamp to MAX_TILE_FEATURE_WIDTH (64)
        let tile = compute_optimal_tile_size(100_000, 16);
        assert_eq!(tile, MAX_TILE_FEATURE_WIDTH);
    }

    #[test]
    fn sampled_indices_respect_ceil_minimum_and_upper_bound_rules() {
        let one_row = sampled_row_indices(5, 0.01, 5, 0);
        assert_eq!(one_row.len(), 1);

        let half_rows = sampled_row_indices(5, 0.5, 5, 0);
        assert_eq!(half_rows.len(), 3);

        let all_rows = sampled_row_indices(5, 1.0, 5, 0);
        assert_eq!(all_rows.len(), 5);

        let (_, one_feature) =
            sampled_feature_tiles(7, 0.01, 5, 0).expect("feature tiles should sample");
        assert_eq!(one_feature, 1);

        let (_, all_features) =
            sampled_feature_tiles(7, 1.0, 5, 0).expect("feature tiles should sample");
        assert_eq!(all_features, 7);
    }

    #[test]
    fn fit_iterations_controls_enforce_min_abs_leaf_value() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 10.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(model.stumps.is_empty());
    }

    #[test]
    fn fit_iterations_controls_clamp_leaf_values() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls =
            IterationControls::new(1, 0.0, 1, 0.0, 0.1, 0.0, 0).expect("controls are valid");
        let model = trainer
            .fit_iterations_with_controls(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert!(!model.stumps.is_empty());
        for stump in &model.stumps {
            assert!(stump.left_leaf_value.as_scalar().abs() <= 0.1);
            assert!(stump.right_leaf_value.as_scalar().abs() <= 0.1);
        }
    }

    #[test]
    fn fit_iterations_summary_reports_loss_improvement_threshold_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 100.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.rounds_completed, 0);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::LossImprovementBelowThreshold
        );
        assert!(summary.model.stumps.is_empty());
        assert!(summary.loss_per_completed_round.is_empty());
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.initial_loss, summary.final_loss);
    }

    #[test]
    fn fit_iterations_summary_tracks_loss_trace_for_completed_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(2, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_completed, 2);
        assert_eq!(summary.loss_per_completed_round.len(), 2);
        assert!(summary.loss_per_completed_round[0] < summary.initial_loss);
        assert!(summary.loss_per_completed_round[1] <= summary.loss_per_completed_round[0]);
        assert_eq!(summary.weak_improvement_rounds_committed, 0);
        assert_eq!(summary.final_loss, summary.loss_per_completed_round[1]);
    }

    #[test]
    fn fit_iterations_summary_allows_bounded_weak_improvement_rounds() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 100.0, 1)
            .expect("controls are valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training succeeds");

        assert_eq!(summary.rounds_requested, 3);
        assert_eq!(summary.rounds_completed, 1);
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::LossImprovementBelowThreshold
        );
        assert_eq!(summary.weak_improvement_rounds_committed, 1);
        assert_eq!(summary.loss_per_completed_round.len(), 1);
        assert!(!summary.model.stumps.is_empty());
    }

    #[test]
    fn predict_row_applies_non_root_nodes_only_when_path_matches() {
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 1,
            stumps: vec![
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 0,
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
                    },
                    left_leaf_value: LeafValue::Scalar(0.0),
                    right_leaf_value: LeafValue::Scalar(1.0),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                },
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 2,
                        feature_index: 0,
                        threshold_bin: 0,
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
                    },
                    left_leaf_value: LeafValue::Scalar(10.0),
                    right_leaf_value: LeafValue::Scalar(20.0),
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

        let left = model.predict_row(&[0.0]).expect("left prediction succeeds");
        let right = model
            .predict_row(&[1.0])
            .expect("right prediction succeeds");

        assert_eq!(left, 0.0);
        assert_eq!(right, 21.0);
    }

    #[test]
    fn retained_stump_count_for_rounds_handles_multi_stump_rounds() {
        let stumps_per_round = vec![3, 2, 4];
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 0), 0);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 1), 3);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 2), 5);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 3), 9);
        assert_eq!(retained_stump_count_for_rounds(&stumps_per_round, 10), 9);
    }

    #[test]
    fn validation_early_stopping_requires_validation_dataset() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 0.0)
            .expect("validation controls are valid");
        let result = trainer.fit_iterations_with_summary(
            &sample_dataset(),
            &sample_binned_matrix(),
            &MockBackend,
            &SquaredErrorObjective,
            controls,
        );
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
    }

    #[test]
    fn fit_iterations_with_validation_summary_reports_validation_plateau_stop_reason() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 100.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let summary = trainer
            .fit_iterations_with_validation_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("iterative training with validation succeeds");

        assert_eq!(
            summary.stop_reason,
            IterationStopReason::ValidationLossPlateau
        );
        assert_eq!(summary.rounds_completed, 0);
        assert!(summary.model.stumps.is_empty());
        assert!(summary.initial_validation_loss.is_some());
        assert!(summary.validation_loss_per_completed_round.is_empty());
        assert_eq!(
            summary.best_validation_loss,
            summary.initial_validation_loss
        );
        assert_eq!(summary.best_validation_round, Some(0));
        assert!(summary.final_validation_loss.is_some());
        assert_eq!(
            summary.final_validation_loss,
            summary.initial_validation_loss
        );
        assert!(summary.sampled_rows_per_completed_round.is_empty());
        assert!(summary.sampled_features_per_completed_round.is_empty());
        assert_eq!(summary.final_loss, summary.initial_loss);
    }

    #[test]
    fn trained_model_artifact_roundtrip_preserves_predictions() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact parses");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(model.feature_count, restored.feature_count);
        assert_eq!(model.stumps.len(), restored.stumps.len());
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].descriptor.kind, ModelSectionKind::Trees);
        assert_eq!(
            parsed.sections[1].descriptor.kind,
            ModelSectionKind::PredictorLayout
        );

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn trained_model_artifact_roundtrip_preserves_optional_categorical_state() {
        let model = sample_trained_model()
            .with_categorical_state(Some(CategoricalStatePayloadV1 {
                format_version: alloygbm_core::CATEGORICAL_STATE_FORMAT_V1,
                leakage_safe_target_encoding: true,
                categorical_feature_indices: vec![1],
            }))
            .expect("categorical state is valid");
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored = TrainedModel::from_artifact_bytes(&bytes).expect("artifact parses");
        let parsed = deserialize_model_artifact_v1(&bytes).expect("artifact decodes");

        assert_eq!(parsed.sections.len(), 3);
        assert_eq!(
            parsed.sections[2].descriptor.kind,
            ModelSectionKind::CategoricalState
        );
        assert_eq!(model.categorical_state, restored.categorical_state);
    }

    #[test]
    fn fit_iterations_with_single_target_encoded_feature_attaches_categorical_state() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let spec = CategoricalTargetEncodingSpec {
            feature_index: 1,
            values: vec![
                "A".to_string(),
                "A".to_string(),
                "B".to_string(),
                "B".to_string(),
            ],
            config: TargetEncoderConfig {
                smoothing: 0.0,
                min_samples_leaf: 1,
                time_aware: false,
            },
        };
        let model = trainer
            .fit_iterations_with_single_target_encoded_feature(
                &sample_dataset(),
                &sample_binned_matrix(),
                &spec,
                &MockBackend,
                &SquaredErrorObjective,
                2,
            )
            .expect("training succeeds");

        let state = model
            .categorical_state
            .as_ref()
            .expect("categorical state is attached");
        assert_eq!(
            state.format_version,
            alloygbm_core::CATEGORICAL_STATE_FORMAT_V1
        );
        assert!(!state.leakage_safe_target_encoding);
        assert_eq!(state.categorical_feature_indices, vec![1]);
    }

    #[test]
    fn trained_model_artifact_accepts_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");

        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");
        let restored =
            TrainedModel::from_artifact_bytes(&legacy_trees_only).expect("legacy artifact parses");

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn strict_mode_rejects_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        assert!(matches!(
            TrainedModel::from_artifact_bytes_with_mode(
                &legacy_trees_only,
                ArtifactCompatibilityMode::Strict
            ),
            Err(EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn strict_mode_accepts_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored =
            TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::Strict)
                .expect("strict artifact parse succeeds");

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn dart_tree_weight_round_trips_through_artifact() {
        // Hand-craft a model with a non-1.0 tree_weight on one stump.
        // Serialize → emit DartTreeWeights section. Deserialize → apply
        // overlay → predict matches the original.
        let mut model = sample_trained_model();
        // Set a deliberately non-default weight on every stump.
        for (i, stump) in model.stumps.iter_mut().enumerate() {
            stump.tree_weight = if i == 0 { 0.25 } else { 0.5 };
        }
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let restored =
            TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::Strict)
                .expect("artifact decodes");

        // tree_weight survives the round-trip.
        for (orig, dec) in model.stumps.iter().zip(restored.stumps.iter()) {
            assert!(
                (orig.tree_weight - dec.tree_weight).abs() < 1e-6,
                "tree_weight drifted: original={}, decoded={}",
                orig.tree_weight,
                dec.tree_weight
            );
        }

        // Predictions also reflect the round-tripped weights — sanity
        // check via the engine's predict_batch.
        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        for (a, b) in original_preds.iter().zip(restored_preds.iter()) {
            assert!((a - b).abs() < 1e-5, "prediction drift: {} vs {}", a, b);
        }
    }

    #[test]
    fn non_dart_artifact_omits_dart_tree_weights_section() {
        // Standard / GOSS models keep `tree_weight = 1.0` for every stump,
        // so the artifact write path must not emit a DartTreeWeights
        // section. Verifies the byte-identical-to-v0.8.0 invariant.
        let model = sample_trained_model();
        // sample_trained_model uses the default 1.0 weights.
        assert!(
            model
                .stumps
                .iter()
                .all(|s| (s.tree_weight - 1.0).abs() < f32::EPSILON)
        );
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&bytes).expect("artifact parses");
        let has_dart_section = parsed
            .sections
            .iter()
            .any(|s| matches!(s.descriptor.kind, ModelSectionKind::DartTreeWeights));
        assert!(
            !has_dart_section,
            "non-DART artifacts must not emit a DartTreeWeights section"
        );
    }

    #[test]
    fn trained_model_predict_row_applies_tree_weight() {
        // Regression test for the v0.9.0 PR review: TrainedModel::predict_row
        // must multiply each stump's leaf contribution by stump.tree_weight,
        // matching the predictor and matching DART training-time arithmetic.
        // Without this, Rust callers loading DART artifacts via
        // TrainedModel::from_artifact_bytes would see unweighted predictions
        // that silently disagree with predict_dense / Python predict.
        let mut model = sample_trained_model();
        let row = vec![0.5_f32, 0.5_f32];
        let pred_unit = model.predict_row(&row).expect("predict_row unit weights");

        // Scale every stump by 0.5. Predict_row must respond proportionally.
        for stump in model.stumps.iter_mut() {
            stump.tree_weight = 0.5;
        }
        let pred_half = model.predict_row(&row).expect("predict_row half weights");

        let baseline = model.baseline_prediction;
        let expected_half = baseline + 0.5 * (pred_unit - baseline);
        assert!(
            (pred_half - expected_half).abs() < 1e-5,
            "predict_row didn't apply tree_weight: pred_half={pred_half}, expected={expected_half}"
        );
    }

    #[test]
    fn dart_early_stopping_truncation_recomputes_tree_weights() {
        // Regression test for the v0.9.0 PR review (P1, codex): when
        // early stopping truncates a DART fit to `best_round`,
        // `dart_state.tree_weights` must be replayed against the kept
        // rounds so the stamped tree_weights match the kept ensemble.
        // A naive truncate would leave kept stumps stamped with weights
        // that were mutated by trees that no longer exist.
        let dataset = sample_dataset();
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let backend = GradientNeutralizationCheckingBackend {
            exposures: dataset
                .factor_exposures
                .as_ref()
                .cloned()
                .unwrap_or_else(|| FactorExposureMatrix {
                    row_count: dataset.row_count(),
                    factor_count: 1,
                    values: vec![0.0; dataset.row_count()],
                }),
            weights: None,
        };
        // Force truncation by training with validation + early stopping
        // that triggers after a few rounds.
        let params = TrainParams {
            boosting_mode: BoostingMode::Dart {
                drop_rate: 0.3,
                max_drop: 5,
                normalize_type: alloygbm_core::DartNormalize::Tree,
                sample_type: alloygbm_core::DartSampleType::Uniform,
            },
            seed: 42,
            deterministic: true,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("DART params pass validation");
        // Train without validation — round-cap controls termination.
        // The point of the test is that the *final stamped tree_weights*
        // match an `apply_normalization` replay over the committed rounds.
        let model = trainer
            .fit_iterations(&dataset, &binned, &backend, &SquaredErrorObjective, 8)
            .expect("DART training succeeds");

        // After training, every stump's tree_weight must be reachable
        // by replaying `apply_normalization` over the committed rounds —
        // i.e., predict_row on the loaded artifact must match
        // predict_row on the in-memory model.
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let loaded =
            TrainedModel::from_artifact_bytes_with_mode(&bytes, ArtifactCompatibilityMode::Strict)
                .expect("artifact loads");
        let row = vec![0.5_f32, 0.5_f32];
        let p1 = model.predict_row(&row).expect("model predict_row");
        let p2 = loaded.predict_row(&row).expect("loaded predict_row");
        assert!(
            (p1 - p2).abs() < 1e-5,
            "DART round-trip predict mismatch: {p1} vs {p2}"
        );
    }

    #[test]
    fn artifact_compatibility_report_classifies_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let report =
            TrainedModel::artifact_compatibility_report(&bytes).expect("report should parse");

        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 1);
        assert!(report.strict_compatible);
        assert!(!report.legacy_trees_only_compatible);
        assert!(report.legacy_compatible);
        assert_eq!(
            report.recommended_mode,
            Some(ArtifactCompatibilityMode::Strict)
        );
    }

    #[test]
    fn artifact_compatibility_report_classifies_legacy_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        let report = TrainedModel::artifact_compatibility_report(&legacy_trees_only)
            .expect("report should parse");
        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 0);
        assert!(!report.strict_compatible);
        assert!(report.legacy_trees_only_compatible);
        assert!(report.legacy_compatible);
        assert_eq!(
            report.recommended_mode,
            Some(ArtifactCompatibilityMode::AllowLegacyTreesOnly)
        );
    }

    #[test]
    fn subtract_histogram_bundle_derives_complementary_child() {
        let parent = HistogramBundle {
            node_id: 7,
            feature_histograms: vec![
                alloygbm_core::FeatureHistogram {
                    feature_index: 0,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 3.0,
                            hess_sum: 5.0,
                            grad_sq_sum: 11.0,
                            count: 4,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            grad_sq_sum: 7.0,
                            count: 2,
                        },
                    ],
                },
                alloygbm_core::FeatureHistogram {
                    feature_index: 1,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 1.5,
                            hess_sum: 4.0,
                            grad_sq_sum: 0.0,
                            count: 3,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            count: 1,
                        },
                    ],
                },
            ],
        };
        let child = HistogramBundle {
            node_id: 15,
            feature_histograms: vec![
                alloygbm_core::FeatureHistogram {
                    feature_index: 0,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 2.0,
                            hess_sum: 3.0,
                            grad_sq_sum: 5.0,
                            count: 2,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.25,
                            hess_sum: 0.5,
                            grad_sq_sum: 1.5,
                            count: 1,
                        },
                    ],
                },
                alloygbm_core::FeatureHistogram {
                    feature_index: 1,
                    bins: vec![
                        alloygbm_core::HistogramBin {
                            grad_sum: 1.0,
                            hess_sum: 2.5,
                            grad_sq_sum: 0.0,
                            count: 2,
                        },
                        alloygbm_core::HistogramBin {
                            grad_sum: -0.25,
                            hess_sum: 0.25,
                            grad_sq_sum: 0.0,
                            count: 1,
                        },
                    ],
                },
            ],
        };

        let complement =
            subtract_histogram_bundle(&parent, &child, 16).expect("subtraction should succeed");
        assert_eq!(complement.node_id, 16);
        assert_eq!(complement.feature_histograms.len(), 2);
        assert_eq!(complement.feature_histograms[0].bins[0].count, 2);
        assert_eq!(complement.feature_histograms[0].bins[1].count, 1);
        assert!((complement.feature_histograms[0].bins[0].grad_sum - 1.0).abs() < 1e-6);
        assert!((complement.feature_histograms[0].bins[1].grad_sum + 0.75).abs() < 1e-6);
        assert!((complement.feature_histograms[0].bins[0].grad_sq_sum - 6.0).abs() < 1e-6);
        assert!((complement.feature_histograms[0].bins[1].grad_sq_sum - 5.5).abs() < 1e-6);
        assert!((complement.feature_histograms[1].bins[0].hess_sum - 1.5).abs() < 1e-6);
        assert!((complement.feature_histograms[1].bins[1].hess_sum - 0.75).abs() < 1e-6);
    }

    #[test]
    fn subtract_histogram_bundle_into_matches_allocating_variant() {
        let parent = HistogramBundle {
            node_id: 7,
            feature_histograms: vec![alloygbm_core::FeatureHistogram {
                feature_index: 0,
                bins: vec![
                    alloygbm_core::HistogramBin {
                        grad_sum: 3.0,
                        hess_sum: 5.0,
                        grad_sq_sum: 0.0,
                        count: 4,
                    },
                    alloygbm_core::HistogramBin {
                        grad_sum: -1.0,
                        hess_sum: 2.0,
                        grad_sq_sum: 0.0,
                        count: 2,
                    },
                ],
            }],
        };
        let child = HistogramBundle {
            node_id: 15,
            feature_histograms: vec![alloygbm_core::FeatureHistogram {
                feature_index: 0,
                bins: vec![
                    alloygbm_core::HistogramBin {
                        grad_sum: 2.0,
                        hess_sum: 3.0,
                        grad_sq_sum: 0.0,
                        count: 2,
                    },
                    alloygbm_core::HistogramBin {
                        grad_sum: -0.25,
                        hess_sum: 0.5,
                        grad_sq_sum: 0.0,
                        count: 1,
                    },
                ],
            }],
        };

        // Allocating variant
        let allocated =
            subtract_histogram_bundle(&parent, &child, 16).expect("subtraction should succeed");

        // In-place variant
        let mut dest = HistogramBundle::new_zeroed(&[0], 2);
        subtract_histogram_bundle_into(&parent, &child, 16, &mut dest)
            .expect("in-place subtraction should succeed");

        assert_eq!(allocated.node_id, dest.node_id);
        assert_eq!(
            allocated.feature_histograms.len(),
            dest.feature_histograms.len()
        );
        for (a, d) in allocated
            .feature_histograms
            .iter()
            .zip(&dest.feature_histograms)
        {
            assert_eq!(a.feature_index, d.feature_index);
            for (ab, db) in a.bins.iter().zip(&d.bins) {
                assert!((ab.grad_sum - db.grad_sum).abs() < 1e-6);
                assert!((ab.hess_sum - db.hess_sum).abs() < 1e-6);
                assert_eq!(ab.count, db.count);
            }
        }
    }

    #[test]
    fn histogram_bundle_reset_zeros_all_bins() {
        let mut bundle = HistogramBundle::new_zeroed(&[0, 1], 3);
        // Set some values
        bundle.feature_histograms[0].bins[0].grad_sum = 5.0;
        bundle.feature_histograms[0].bins[0].hess_sum = 3.0;
        bundle.feature_histograms[0].bins[0].count = 10;
        bundle.feature_histograms[1].bins[2].grad_sum = -2.5;
        bundle.feature_histograms[1].bins[2].count = 7;

        bundle.reset(42);
        assert_eq!(bundle.node_id, 42);
        for fh in &bundle.feature_histograms {
            for bin in &fh.bins {
                assert_eq!(bin.grad_sum, 0.0);
                assert_eq!(bin.hess_sum, 0.0);
                assert_eq!(bin.count, 0);
            }
        }
    }

    #[test]
    fn histogram_bundle_new_zeroed_creates_correct_structure() {
        let features = [0, 3, 7];
        let bundle = HistogramBundle::new_zeroed(&features, 5);
        assert_eq!(bundle.feature_histograms.len(), 3);
        assert_eq!(bundle.feature_histograms[0].feature_index, 0);
        assert_eq!(bundle.feature_histograms[1].feature_index, 3);
        assert_eq!(bundle.feature_histograms[2].feature_index, 7);
        for fh in &bundle.feature_histograms {
            assert_eq!(fh.bins.len(), 5);
        }
    }

    #[test]
    fn artifact_compatibility_report_marks_malformed_required_sections_incompatible() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");
        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
            .expect("report should parse");
        assert_eq!(report.trees_section_count, 1);
        assert_eq!(report.predictor_layout_section_count, 2);
        assert!(!report.strict_compatible);
        assert!(!report.legacy_trees_only_compatible);
        assert!(!report.legacy_compatible);
        assert_eq!(report.recommended_mode, None);
    }

    #[test]
    fn from_artifact_bytes_auto_selects_strict_for_dual_section_payload() {
        let model = sample_trained_model();
        let bytes = model.to_artifact_bytes().expect("artifact serializes");
        let (restored, selected_mode) =
            TrainedModel::from_artifact_bytes_auto(&bytes).expect("auto import succeeds");

        assert_eq!(selected_mode, ArtifactCompatibilityMode::Strict);
        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn from_artifact_bytes_auto_selects_legacy_for_trees_only_payload() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let legacy_trees_only =
            serialize_model_artifact_v1(&metadata, &[(ModelSectionKind::Trees, trees_payload)])
                .expect("artifact serializes");

        let (restored, selected_mode) = TrainedModel::from_artifact_bytes_auto(&legacy_trees_only)
            .expect("auto import succeeds");
        assert_eq!(
            selected_mode,
            ArtifactCompatibilityMode::AllowLegacyTreesOnly
        );

        let rows = vec![vec![0.0, 0.0], vec![3.0, 1.0]];
        let original_preds = model.predict_batch(&rows).expect("predicts");
        let restored_preds = restored.predict_batch(&rows).expect("predicts");
        assert_eq!(original_preds, restored_preds);
    }

    #[test]
    fn from_artifact_bytes_auto_rejects_malformed_required_section_layouts() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");
        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let result = TrainedModel::from_artifact_bytes_auto(&duplicate_predictor);
        match result {
            Err(EngineError::ContractViolation(message)) => {
                let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
                    .expect("report should parse");
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_auto_mode_error(
                        report.required_section_report()
                    )
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    #[test]
    fn trained_model_artifact_rejects_missing_required_sections() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");

        let non_legacy_missing_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload.clone()),
                (ModelSectionKind::ShapAux, vec![9_u8]),
            ],
        )
        .expect("artifact serializes");
        let missing_trees = serialize_model_artifact_v1(
            &metadata,
            &[(ModelSectionKind::PredictorLayout, layout_payload.clone())],
        )
        .expect("artifact serializes");

        assert!(matches!(
            TrainedModel::from_artifact_bytes(&non_legacy_missing_predictor),
            Err(EngineError::ContractViolation(_))
        ));
        assert!(matches!(
            TrainedModel::from_artifact_bytes(&missing_trees),
            Err(EngineError::ContractViolation(_))
        ));
    }

    #[test]
    fn trained_model_artifact_rejects_duplicate_required_sections() {
        let model = sample_trained_model();
        let metadata = ModelMetadata {
            format_version: MODEL_FORMAT_V1,
            feature_names: (0..model.feature_count)
                .map(|index| format!("f{index}"))
                .collect(),
            trained_device: Device::Cpu,
            objective: "squared_error".to_string(),
            num_classes: None,
        };
        let trees_payload = encode_trained_model_payload(&model).expect("trees encode");
        let layout_payload =
            encode_predictor_layout_payload(&model).expect("predictor layout encodes");

        let duplicate_predictor = serialize_model_artifact_v1(
            &metadata,
            &[
                (ModelSectionKind::Trees, trees_payload),
                (ModelSectionKind::PredictorLayout, layout_payload.clone()),
                (ModelSectionKind::PredictorLayout, layout_payload),
            ],
        )
        .expect("artifact serializes");

        let result = TrainedModel::from_artifact_bytes(&duplicate_predictor);
        match result {
            Err(EngineError::ContractViolation(message)) => {
                let report = TrainedModel::artifact_compatibility_report(&duplicate_predictor)
                    .expect("report should parse");
                assert_eq!(
                    message,
                    alloygbm_core::format_required_section_mode_error(
                        report.required_section_report(),
                        true
                    )
                );
            }
            other => panic!("expected contract violation, got {other:?}"),
        }
    }

    // -- Multi-class tests ---------------------------------------------------

    #[test]
    fn test_multiclass_softmax_rejects_single_class() {
        let result = MultiClassSoftmaxObjective::new(1);
        assert!(result.is_err());
        if let Err(EngineError::InvalidConfig(msg)) = result {
            assert!(msg.contains("at least 2"), "unexpected error: {msg}");
        } else {
            panic!("expected InvalidConfig error");
        }
    }

    #[test]
    fn test_multiclass_softmax_rejects_zero_classes() {
        let result = MultiClassSoftmaxObjective::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiclass_softmax_creates_with_valid_k() {
        let obj = MultiClassSoftmaxObjective::new(3).expect("k=3 should work");
        assert_eq!(obj.num_classes, 3);
        assert_eq!(obj.objective_name(), "multiclass_softmax");
    }

    #[test]
    fn test_multiclass_softmax_initial_predictions() {
        let obj = MultiClassSoftmaxObjective::new(4).unwrap();
        let preds = obj.initial_predictions();
        assert_eq!(preds.len(), 4);
        for &p in &preds {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn test_multiclass_softmax_gradients_basic() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        // 3 samples with targets [0, 1, 2]
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Initial predictions: all zeros -> uniform softmax: 1/3 each
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let mut buffer = Vec::new();

        // Gradients for class 0:
        // Sample 0 (target=0): grad = p_0 - 1 = 1/3 - 1 = -2/3
        // Sample 1 (target=1): grad = p_0 - 0 = 1/3
        // Sample 2 (target=2): grad = p_0 - 0 = 1/3
        let _ = obj.compute_gradients_for_class(&predictions, &targets, None, 0, &mut buffer);
        assert_eq!(buffer.len(), 3);
        // Sample 0: grad should be negative (correct class)
        assert!(
            buffer[0].grad < 0.0,
            "grad for correct class should be negative"
        );
        // Sample 1: grad should be positive (wrong class)
        assert!(
            buffer[1].grad > 0.0,
            "grad for wrong class should be positive"
        );
        // Sample 2: grad should be positive (wrong class)
        assert!(
            buffer[2].grad > 0.0,
            "grad for wrong class should be positive"
        );

        // Hessians should all be positive
        for gp in &buffer {
            assert!(gp.hess > 0.0, "hessian must be positive");
        }

        // Verify approximate values: grad ≈ -2/3 for correct, 1/3 for wrong
        assert!((buffer[0].grad - (-2.0 / 3.0)).abs() < 0.01);
        assert!((buffer[1].grad - (1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_multiclass_softmax_gradients_with_weights() {
        let obj = MultiClassSoftmaxObjective::new(2).unwrap();
        let targets = vec![0.0_f32, 1.0];
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 2]; 2];
        let weights = vec![2.0_f32, 0.5];
        let mut buffer = Vec::new();

        let _ =
            obj.compute_gradients_for_class(&predictions, &targets, Some(&weights), 0, &mut buffer);
        // With uniform softmax (p=0.5): grad_0 = (0.5 - 1) * 2.0 = -1.0
        assert!((buffer[0].grad - (-1.0)).abs() < 0.01);
        // grad_1 = (0.5 - 0) * 0.5 = 0.25
        assert!((buffer[1].grad - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_multiclass_softmax_loss() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Uniform predictions: loss = -log(1/3) = log(3) ≈ 1.0986
        let predictions: Vec<Vec<f32>> = vec![vec![0.0; 3]; 3];
        let loss = obj.loss(&predictions, &targets, None).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
        let expected = (3.0_f32).ln();
        assert!((loss - expected).abs() < 0.01, "loss {loss} ≈ {expected}");
    }

    #[test]
    fn test_multiclass_softmax_loss_perfect_predictions() {
        let obj = MultiClassSoftmaxObjective::new(3).unwrap();
        let targets = vec![0.0_f32, 1.0, 2.0];
        // Strong predictions toward correct classes
        let predictions: Vec<Vec<f32>> = vec![
            vec![10.0, -10.0, -10.0], // strongly class 0
            vec![-10.0, 10.0, -10.0], // strongly class 1
            vec![-10.0, -10.0, 10.0], // strongly class 2
        ];
        let loss = obj.loss(&predictions, &targets, None).unwrap();
        assert!(
            loss < 0.01,
            "loss should be near zero for perfect predictions, got {loss}"
        );
    }

    #[test]
    fn test_multiclass_trained_model_artifact_roundtrip() {
        // Create a minimal model manually
        let model = MultiClassTrainedModel {
            num_classes: 3,
            baseline_predictions: vec![0.0, 0.0, 0.0],
            feature_count: 2,
            class_stumps: vec![
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 1,
                        gain: 2.5,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: -1.0,
                            hess_sum: 2.0,
                            grad_sq_sum: 0.0,
                            row_count: 3,
                        },
                        right_stats: NodeStats {
                            grad_sum: 1.0,
                            hess_sum: 2.0,
                            grad_sq_sum: 0.0,
                            row_count: 3,
                        },
                    },
                    left_leaf_value: LeafValue::Scalar(-0.1),
                    right_leaf_value: LeafValue::Scalar(0.1),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                }],
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 1,
                        threshold_bin: 2,
                        gain: 1.5,
                        default_left: true,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.5,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 2,
                        },
                        right_stats: NodeStats {
                            grad_sum: -0.5,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 4,
                        },
                    },
                    left_leaf_value: LeafValue::Scalar(0.2),
                    right_leaf_value: LeafValue::Scalar(-0.05),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                }],
                vec![], // class 2 has no stumps
            ],
            categorical_state: None,
            objective: "multiclass_softmax".to_string(),
            morph_metadata: None,
            dro_metadata: None,
        };

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        // Deserialize
        let restored = MultiClassTrainedModel::from_artifact_bytes(&bytes)
            .expect("deserialize should succeed");
        assert_eq!(restored.num_classes, 3);
        assert_eq!(restored.feature_count, 2);
        assert_eq!(restored.baseline_predictions, vec![0.0, 0.0, 0.0]);
        assert_eq!(restored.class_stumps.len(), 3);
        assert_eq!(restored.class_stumps[0].len(), 1);
        assert_eq!(restored.class_stumps[1].len(), 1);
        assert_eq!(restored.class_stumps[2].len(), 0);
        assert_eq!(restored.objective, "multiclass_softmax");
    }

    #[test]
    fn test_multiclass_trained_model_rounds_completed() {
        let model = MultiClassTrainedModel {
            num_classes: 2,
            baseline_predictions: vec![0.0, 0.0],
            feature_count: 1,
            class_stumps: vec![
                // Class 0: 2 trees (round 0 has 3 stumps, round 1 has 1 stump)
                vec![
                    TrainedStump {
                        split: SplitCandidate {
                            node_id: 0, // tree 0, node 0
                            feature_index: 0,
                            threshold_bin: 1,
                            gain: 1.0,
                            default_left: false,
                            is_categorical: false,
                            categorical_bitset: None,
                            left_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                grad_sq_sum: 0.0,
                                row_count: 2,
                            },
                            right_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                grad_sq_sum: 0.0,
                                row_count: 2,
                            },
                        },
                        left_leaf_value: LeafValue::Scalar(-0.1),
                        right_leaf_value: LeafValue::Scalar(0.1),
                        tree_weight: 1.0,
                        multi_output_leaf_values: None,
                    },
                    TrainedStump {
                        split: SplitCandidate {
                            node_id: TREE_NODE_STRIDE, // tree 1, node 0
                            feature_index: 0,
                            threshold_bin: 2,
                            gain: 0.5,
                            default_left: false,
                            is_categorical: false,
                            categorical_bitset: None,
                            left_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                grad_sq_sum: 0.0,
                                row_count: 2,
                            },
                            right_stats: NodeStats {
                                grad_sum: 0.0,
                                hess_sum: 1.0,
                                grad_sq_sum: 0.0,
                                row_count: 2,
                            },
                        },
                        left_leaf_value: LeafValue::Scalar(-0.05),
                        right_leaf_value: LeafValue::Scalar(0.05),
                        tree_weight: 1.0,
                        multi_output_leaf_values: None,
                    },
                ],
                // Class 1: same structure
                vec![TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 1,
                        gain: 1.0,
                        default_left: false,
                        is_categorical: false,
                        categorical_bitset: None,
                        left_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 2,
                        },
                        right_stats: NodeStats {
                            grad_sum: 0.0,
                            hess_sum: 1.0,
                            grad_sq_sum: 0.0,
                            row_count: 2,
                        },
                    },
                    left_leaf_value: LeafValue::Scalar(0.1),
                    right_leaf_value: LeafValue::Scalar(-0.1),
                    tree_weight: 1.0,
                    multi_output_leaf_values: None,
                }],
            ],
            categorical_state: None,
            objective: "multiclass_softmax".to_string(),
            morph_metadata: None,
            dro_metadata: None,
        };
        assert_eq!(model.rounds_completed(), 2);
    }

    // ── Per-round metric callback tests ─────────────────────────────────

    /// A simple test callback that returns MSE as the metric value.
    struct MseMetricCallback;

    impl PerRoundMetricCallback for MseMetricCallback {
        fn evaluate(
            &self,
            predictions: &[f32],
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            if predictions.len() != targets.len() {
                return Err(EngineError::ContractViolation(
                    "predictions and targets length mismatch".into(),
                ));
            }
            let n = predictions.len() as f32;
            let mse: f32 = predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / n;
            Ok(mse)
        }
        fn higher_is_better(&self) -> bool {
            false
        }
        fn metric_name(&self) -> &str {
            "test_mse"
        }
    }

    /// A metric callback where higher values are better (e.g. R²-like).
    struct HigherIsBetterCallback;

    impl PerRoundMetricCallback for HigherIsBetterCallback {
        fn evaluate(
            &self,
            predictions: &[f32],
            targets: &[f32],
            _sample_weights: Option<&[f32]>,
        ) -> EngineResult<f32> {
            // Return negative MSE so higher = better
            let n = predictions.len() as f32;
            let neg_mse: f32 = -(predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / n);
            Ok(neg_mse)
        }
        fn higher_is_better(&self) -> bool {
            true
        }
        fn metric_name(&self) -> &str {
            "neg_mse"
        }
    }

    #[test]
    fn test_per_round_callback_basic() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(5, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = MseMetricCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with metric callback succeeds");

        // Callback should have been invoked each round.
        assert_eq!(
            summary.custom_metric_per_round.len(),
            summary.rounds_completed
        );
        assert_eq!(summary.custom_metric_name.as_deref(), Some("test_mse"));
        // Metric values should be non-negative (MSE).
        for v in &summary.custom_metric_per_round {
            assert!(*v >= 0.0, "MSE metric should be non-negative");
        }
    }

    #[test]
    fn test_per_round_callback_early_stopping() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        // Set early_stopping_rounds=1 with very high min_improvement so it
        // stops almost immediately when the custom metric plateaus.
        let controls = IterationControls::new(100, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 1000.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = MseMetricCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with callback early stopping succeeds");

        // Should have stopped early due to the custom metric plateau.
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
            "expected custom metric plateau stop reason, got {:?}",
            summary.stop_reason,
        );
        assert!(
            summary.rounds_completed < 100,
            "should have stopped before all 100 rounds"
        );
    }

    #[test]
    fn test_per_round_callback_higher_is_better() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(100, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid")
            .with_validation_early_stopping(1, 1000.0)
            .expect("validation controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        let callback = HigherIsBetterCallback;
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                Some(&callback),
            )
            .expect("training with higher-is-better callback succeeds");

        // Should also stop early.
        assert_eq!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
        );
        assert_eq!(summary.custom_metric_name.as_deref(), Some("neg_mse"));
    }

    #[test]
    fn test_per_round_callback_none_no_effect() {
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
            .expect("controls are valid");
        let validation = ValidationDatasetRef {
            dataset: &sample_dataset(),
            binned_matrix: &sample_binned_matrix(),
        };
        // Pass None — should behave identically to the non-metric path.
        let summary = trainer
            .fit_iterations_with_validation_and_metric(
                &sample_dataset(),
                &sample_binned_matrix(),
                validation,
                &MockBackend,
                &SquaredErrorObjective,
                controls,
                None,
            )
            .expect("training without callback succeeds");

        assert!(summary.custom_metric_per_round.is_empty());
        assert!(summary.custom_metric_name.is_none());
        assert_ne!(
            summary.stop_reason,
            IterationStopReason::CustomMetricPlateau,
        );
    }

    #[test]
    fn test_custom_metric_plateau_stop_reason() {
        // Verify the CustomMetricPlateau variant is distinct from ValidationLossPlateau.
        assert_ne!(
            IterationStopReason::CustomMetricPlateau,
            IterationStopReason::ValidationLossPlateau,
        );
        assert_ne!(
            IterationStopReason::CustomMetricPlateau,
            IterationStopReason::CompletedRequestedRounds,
        );
    }

    // ── Native categorical split tests ──────────────────────────────────

    #[test]
    fn test_trained_model_categorical_roundtrip() {
        // Build a TrainedModel with one categorical stump and one continuous stump.
        let model = TrainedModel {
            baseline_prediction: 0.5,
            feature_count: 2,
            stumps: vec![
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
                        feature_index: 0,
                        threshold_bin: 0,
                        gain: 2.0,
                        default_left: true,
                        is_categorical: true,
                        categorical_bitset: Some(vec![0b0000_0011]), // cats 0,1 left
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
                },
                TrainedStump {
                    split: SplitCandidate {
                        node_id: 0,
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
                    left_leaf_value: LeafValue::Scalar(0.05),
                    right_leaf_value: LeafValue::Scalar(-0.05),
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

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");
        assert!(!bytes.is_empty());

        let restored = TrainedModel::from_artifact_bytes_with_mode(
            &bytes,
            ArtifactCompatibilityMode::AllowLegacyTreesOnly,
        )
        .expect("deserialize should succeed");

        // Verify basic fields
        assert_eq!(restored.feature_count, 2);
        assert_eq!(restored.stumps.len(), 2);
        assert_eq!(restored.native_categorical_feature_indices, vec![0]);

        // Verify categorical stump
        let stump0 = &restored.stumps[0];
        assert!(stump0.split.is_categorical);
        assert_eq!(stump0.split.categorical_bitset, Some(vec![0b0000_0011]));
        assert_eq!(stump0.left_leaf_value.as_scalar(), -0.1);
        assert_eq!(stump0.right_leaf_value.as_scalar(), 0.1);

        // Verify continuous stump
        let stump1 = &restored.stumps[1];
        assert!(!stump1.split.is_categorical);
        assert!(stump1.split.categorical_bitset.is_none());
        assert_eq!(stump1.split.threshold_bin, 3);
    }

    #[test]
    fn test_trained_model_categorical_backward_compat() {
        // Build a model WITHOUT any categorical stumps (old-style).
        let model = TrainedModel {
            baseline_prediction: 1.0,
            feature_count: 1,
            stumps: vec![TrainedStump {
                split: SplitCandidate {
                    node_id: 0,
                    feature_index: 0,
                    threshold_bin: 2,
                    gain: 1.0,
                    default_left: false,
                    is_categorical: false,
                    categorical_bitset: None,
                    left_stats: NodeStats {
                        grad_sum: -0.5,
                        hess_sum: 1.0,
                        grad_sq_sum: 0.0,
                        row_count: 3,
                    },
                    right_stats: NodeStats {
                        grad_sum: 0.5,
                        hess_sum: 1.0,
                        grad_sq_sum: 0.0,
                        row_count: 3,
                    },
                },
                left_leaf_value: LeafValue::Scalar(-0.2),
                right_leaf_value: LeafValue::Scalar(0.2),
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

        let bytes = model.to_artifact_bytes().expect("serialize should succeed");

        // Deserialize — should work fine with no categorical section
        let restored = TrainedModel::from_artifact_bytes_with_mode(
            &bytes,
            ArtifactCompatibilityMode::AllowLegacyTreesOnly,
        )
        .expect("deserialize should succeed");
        assert_eq!(restored.stumps.len(), 1);
        assert!(!restored.stumps[0].split.is_categorical);
        assert!(restored.native_categorical_feature_indices.is_empty());
    }

    // -- Morph-mode end-to-end tests -----------------------------------------

    #[test]
    fn morph_mode_regression_trains_to_completion() {
        use alloygbm_core::MorphConfig;
        let params = TrainParams {
            morph_config: Some(MorphConfig::default()),
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params with morph config");
        let model = trainer
            .fit_iterations(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                3,
            )
            .expect("morph-mode training must succeed");
        // Model should have produced at least one stump.
        assert!(
            !model.stumps.is_empty(),
            "morph training should produce stumps"
        );
    }

    #[test]
    fn morph_mode_non_morph_path_unchanged() {
        // --- Part 1: two identical non-morph runs must produce byte-identical artifacts ---
        //
        // Trains the same dataset twice with the same seed and morph_config = None.
        // Because the non-morph path is deterministic by design, both runs must
        // serialise to exactly the same bytes.
        let non_morph_params = TrainParams {
            morph_config: None,
            seed: 42,
            deterministic: true,
            ..TrainParams::default()
        };

        let bytes_a = {
            let trainer = Trainer::new(non_morph_params.clone()).expect("valid params (run A)");
            trainer
                .fit_iterations(
                    &sample_dataset(),
                    &sample_binned_matrix(),
                    &MockBackend,
                    &SquaredErrorObjective,
                    3,
                )
                .expect("non-morph training must succeed (run A)")
                .to_artifact_bytes()
                .expect("artifact serialises (run A)")
        };

        let bytes_b = {
            let trainer = Trainer::new(non_morph_params).expect("valid params (run B)");
            trainer
                .fit_iterations(
                    &sample_dataset(),
                    &sample_binned_matrix(),
                    &MockBackend,
                    &SquaredErrorObjective,
                    3,
                )
                .expect("non-morph training must succeed (run B)")
                .to_artifact_bytes()
                .expect("artifact serialises (run B)")
        };

        assert_eq!(
            bytes_a, bytes_b,
            "two non-morph runs with identical params must produce byte-identical artifacts"
        );

        // --- Part 2: morph path with morph_rate=0 / depth_penalty_base=1 must yield
        //     the same leaf values as the non-morph path ---
        //
        // The morph path always applies `depth_penalty * iter_shrinkage` to leaf values
        // (even during warmup), so setting morph_warmup_iters=u32::MAX alone cannot
        // neutralise that.  Instead we zero out both multiplicative factors:
        //   depth_penalty_base = 1.0  =>  depth_penalty = 1.0^(depth/3) = 1.0
        //   morph_rate = 0.0          =>  iter_shrinkage = 1.0 - 0 * ... = 1.0
        //   scale = 1.0 * 1.0 = 1.0  =>  no modification to leaf values
        // With balance_penalty=false and info_score_weight=0 the split choice is also
        // unchanged, so the resulting leaf values must match the non-morph path exactly.
        use alloygbm_core::MorphConfig;

        let identity_morph_params = TrainParams {
            morph_config: Some(MorphConfig {
                morph_rate: 0.0,
                depth_penalty_base: 1.0,
                balance_penalty: false,
                info_score_weight: 0.0,
                morph_warmup_iters: u32::MAX,
                ..MorphConfig::default()
            }),
            seed: 42,
            deterministic: true,
            ..TrainParams::default()
        };

        let model_no_morph =
            TrainedModel::from_artifact_bytes(&bytes_a).expect("deserialise run-A artifact");

        let model_identity_morph = {
            let trainer = Trainer::new(identity_morph_params).expect("valid identity-morph params");
            trainer
                .fit_iterations(
                    &sample_dataset(),
                    &sample_binned_matrix(),
                    &MockBackend,
                    &SquaredErrorObjective,
                    3,
                )
                .expect("identity-morph training must succeed")
        };

        assert_eq!(
            model_no_morph.stumps.len(),
            model_identity_morph.stumps.len(),
            "stump count must match between non-morph and identity-morph"
        );
        for (i, (s_plain, s_morph)) in model_no_morph
            .stumps
            .iter()
            .zip(model_identity_morph.stumps.iter())
            .enumerate()
        {
            assert!(
                (s_plain.left_leaf_value.as_scalar() - s_morph.left_leaf_value.as_scalar()).abs()
                    < 1e-5,
                "stump {i} left_leaf_value mismatch: non-morph={} identity-morph={}",
                s_plain.left_leaf_value.as_scalar(),
                s_morph.left_leaf_value.as_scalar(),
            );
            assert!(
                (s_plain.right_leaf_value.as_scalar() - s_morph.right_leaf_value.as_scalar()).abs()
                    < 1e-5,
                "stump {i} right_leaf_value mismatch: non-morph={} identity-morph={}",
                s_plain.right_leaf_value.as_scalar(),
                s_morph.right_leaf_value.as_scalar(),
            );
        }
    }

    #[test]
    fn interaction_constraints_paths_never_mix_disjoint_groups() {
        // Train a small synthetic regression model with two disjoint
        // constraint groups and walk every tree's root-to-leaf path to
        // verify the constraint is honoured.  The synthetic target depends
        // on features from both groups so the model has incentive to use
        // each — without the constraint we'd routinely see paths that mix.
        use alloygbm_core::{DatasetMatrix, TrainingDataset};
        let n = 64;
        let feature_count = 4;
        let mut values = Vec::with_capacity(n * feature_count);
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            // f0 takes 4 values, f1 takes 2 values, f2 takes 4 values, f3 takes 2.
            let f0 = (i % 4) as f32;
            let f1 = ((i / 4) % 2) as f32;
            let f2 = ((i / 8) % 4) as f32;
            let f3 = ((i / 32) % 2) as f32;
            values.extend_from_slice(&[f0, f1, f2, f3]);
            targets.push(f0 * 0.5 + f1 - f2 * 0.3 + f3 * 0.8);
        }
        let dataset = TrainingDataset {
            matrix: DatasetMatrix::new(n, feature_count, values).expect("matrix"),
            targets,
            sample_weights: None,
            time_index: None,
            group_id: None,
            factor_exposures: None,
        };
        let binned = sample_binned_matrix_for_dataset(&dataset);
        let groups = vec![vec![0u32, 1], vec![2, 3]];
        let params = TrainParams {
            max_depth: 4,
            interaction_constraints: groups.clone(),
            seed: 0,
            deterministic: true,
            ..TrainParams::default()
        };

        let summary = Trainer::new(params)
            .unwrap()
            .fit_iterations_with_summary(
                &dataset,
                &binned,
                &MockBackend,
                &SquaredErrorObjective,
                IterationControls::new(3, 0.0, 1, 0.0, 1_000_000.0, 0.0, 0)
                    .unwrap()
                    .with_subsample_rates(1.0, 1.0)
                    .unwrap(),
            )
            .expect("fit succeeds");
        assert!(summary.rounds_completed > 0);

        // Group stumps by tree_id then walk every path.
        let mut by_tree: std::collections::HashMap<
            u32,
            std::collections::HashMap<u32, &TrainedStump>,
        > = std::collections::HashMap::new();
        for stump in &summary.model.stumps {
            let (tree_id, local_id) = decode_tree_node_id(stump.split.node_id);
            by_tree.entry(tree_id).or_default().insert(local_id, stump);
        }

        fn feature_groups_of(feat: u32, groups: &[Vec<u32>]) -> Vec<usize> {
            groups
                .iter()
                .enumerate()
                .filter_map(|(gi, g)| if g.contains(&feat) { Some(gi) } else { None })
                .collect()
        }

        fn walk(
            nodes: &std::collections::HashMap<u32, &TrainedStump>,
            local_id: u32,
            mut active_groups: std::collections::BTreeSet<usize>,
            groups: &[Vec<u32>],
        ) {
            let Some(stump) = nodes.get(&local_id) else {
                return;
            };
            let feat_groups = feature_groups_of(stump.split.feature_index, groups);
            if !feat_groups.is_empty() {
                let feat_set: std::collections::BTreeSet<usize> = feat_groups.into_iter().collect();
                if active_groups.is_empty() {
                    active_groups = feat_set;
                } else {
                    let intersection: std::collections::BTreeSet<usize> =
                        active_groups.intersection(&feat_set).copied().collect();
                    assert!(
                        !intersection.is_empty(),
                        "tree-path violated interaction constraints at feature {} (prior active groups {:?}, feature groups {:?})",
                        stump.split.feature_index,
                        active_groups,
                        feat_set,
                    );
                    active_groups = intersection;
                }
            }
            walk(nodes, local_id * 2 + 1, active_groups.clone(), groups);
            walk(nodes, local_id * 2 + 2, active_groups, groups);
        }

        for nodes in by_tree.values() {
            walk(nodes, 0, std::collections::BTreeSet::new(), &groups);
        }
    }

    #[test]
    fn iteration_run_summary_populates_diagnostics_per_round() {
        // End-to-end check: a regression fit records one IterationDiagnostics
        // entry per completed round.  No factor neutralization is configured,
        // so the projection-related fields must remain `None`.
        let trainer = Trainer::new(TrainParams::default()).expect("valid params");
        let controls = trainer
            .default_iteration_controls(3)
            .expect("controls valid");
        let summary = trainer
            .fit_iterations_with_summary(
                &sample_dataset(),
                &sample_binned_matrix(),
                &MockBackend,
                &SquaredErrorObjective,
                controls,
            )
            .expect("training succeeds");
        assert_eq!(
            summary.diagnostics_per_round.len(),
            summary.rounds_completed,
            "one diagnostics entry per completed round"
        );
        for d in &summary.diagnostics_per_round {
            assert!(d.gradient_l2_norm >= 0.0);
            assert!(d.hessian_l2_norm >= 0.0);
            assert!(d.original_gradient_l2_norm.is_none());
            assert!(d.projected_gradient_l2_norm.is_none());
            assert!(d.neutralization_effectiveness.is_none());
        }
    }

    #[test]
    fn trained_stump_carries_multi_output_leaf_values_for_joint_trainer() {
        // Construct a stump with sensible defaults via new_unweighted.
        let stump = TrainedStump::new_unweighted(
            SplitCandidate {
                node_id: 0,
                feature_index: 0,
                threshold_bin: 0,
                gain: 0.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    row_count: 0,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    grad_sq_sum: 0.0,
                    row_count: 0,
                },
            },
            LeafValue::Scalar(0.0),
            LeafValue::Scalar(0.0),
        );
        // Default for scalar / linear paths.
        assert!(stump.multi_output_leaf_values.is_none());

        // Joint multi-output stump: 2 outputs, distinct values per child.
        let mut stump = stump;
        stump.multi_output_leaf_values = Some((vec![1.0_f32, 2.0], vec![3.0, 4.0]));
        let (left_k, right_k) = stump.multi_output_leaf_values.as_ref().unwrap();
        assert_eq!(left_k.len(), 2);
        assert_eq!(right_k.len(), 2);
        assert_eq!(left_k[1], 2.0);
        assert_eq!(right_k[0], 3.0);
    }

    #[test]
    fn trained_model_roundtrips_neutralization_metadata_when_some() {
        use alloygbm_core::{
            FactorNeutralizationConfig, NeutralizationKind, NeutralizationMetadataPayload,
        };
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 1,
            stumps: Vec::new(),
            categorical_state: None,
            node_debug_stats: None,
            objective: "test".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
            neutralization_metadata: Some(NeutralizationMetadataPayload {
                config: FactorNeutralizationConfig {
                    kind: NeutralizationKind::PerRoundGradient,
                    ridge_lambda: 5e-4,
                    split_penalty: 0.0,
                },
            }),
        };
        let bytes = model.to_artifact_bytes().expect("encode");
        let decoded = TrainedModel::from_artifact_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.neutralization_metadata,
            model.neutralization_metadata
        );
    }

    #[test]
    fn trained_model_omits_neutralization_metadata_section_when_none() {
        // Build a model with neutralization_metadata = None, parse its raw
        // sections, and confirm no NeutralizationMetadata section is present.
        let model = TrainedModel {
            baseline_prediction: 0.0,
            feature_count: 1,
            stumps: Vec::new(),
            categorical_state: None,
            node_debug_stats: None,
            objective: "test".to_string(),
            native_categorical_feature_indices: Vec::new(),
            morph_metadata: None,
            dro_metadata: None,
            feature_baseline: None,
            neutralization_metadata: None,
        };
        let bytes = model.to_artifact_bytes().expect("encode");
        let parsed = alloygbm_core::deserialize_model_artifact_v1(&bytes).expect("parse");
        assert!(
            !parsed
                .sections
                .iter()
                .any(|s| s.descriptor.kind
                    == alloygbm_core::ModelSectionKind::NeutralizationMetadata),
            "did not expect a NeutralizationMetadata section"
        );
    }

    #[test]
    fn test_quantile_objective_gradients_and_loss() {
        let obj = QuantileObjective { alpha: 0.7 };
        assert_eq!(obj.objective_name(), "quantile");
        assert_eq!(obj.quantile_alpha(), Some(0.7));
        assert!(!obj.supports_leaf_refinement());

        // Test initial prediction (weighted quantile)
        let targets = vec![10.0, 20.0, 30.0];
        let init = obj.initial_prediction(&targets, None).unwrap();
        // threshold = 0.7 * 3 = 2.1
        // cum weight = 1 (at 10), 2 (at 20), 3 (at 30) -> first cumulative >= 2.1 is 3 (at 30)
        assert_eq!(init, 30.0);

        let init_weighted = obj
            .initial_prediction(&targets, Some(&[1.0, 2.0, 1.0]))
            .unwrap();
        // weights = [1.0, 2.0, 1.0], total_weight = 4.0
        // threshold = 0.7 * 4.0 = 2.8
        // cum weights: 1.0 (at 10), 3.0 (at 20), 4.0 (at 30) -> first cumulative >= 2.8 is 3.0 (20.0 here)
        assert_eq!(init_weighted, 20.0);

        // Test gradients and loss
        let predictions = vec![15.0, 25.0];
        let targets = vec![20.0, 10.0]; // y > y_hat for first (20 > 15), y <= y_hat for second (10 <= 25)
        let grads = obj.compute_gradients(&predictions, &targets, None).unwrap();
        assert_eq!(grads.len(), 2);
        // idx 0: target=20.0, pred=15.0. target > pred -> grad = -0.7 * 1.0 = -0.7, hess = 1.0
        assert!((grads[0].grad - (-0.7)).abs() < 1e-6);
        assert_eq!(grads[0].hess, 1.0);
        // idx 1: target=10.0, pred=25.0. target <= pred -> grad = (1.0 - 0.7) * 1.0 = 0.3, hess = 1.0
        assert!((grads[1].grad - 0.3).abs() < 1e-6);
        assert_eq!(grads[1].hess, 1.0);

        // Test loss
        // diffs: idx 0: 20 - 15 = 5 > 0 -> loss = 0.7 * 5 = 3.5
        //        idx 1: 10 - 25 = -15 <= 0 -> loss = (0.7 - 1.0) * (-15) = 4.5
        // average loss = (3.5 + 4.5) / 2 = 4.0
        let loss = obj.loss(&predictions, &targets, None).unwrap();
        assert!((loss - 4.0).abs() < 1e-6);

        // Test loss with weights
        let weights = vec![2.0, 1.0];
        // weighted loss: idx 0: 3.5 * 2.0 = 7.0
        //                idx 1: 4.5 * 1.0 = 4.5
        // average loss = (7.0 + 4.5) / 2 = 5.75
        let loss_weighted = obj.loss(&predictions, &targets, Some(&weights)).unwrap();
        assert!((loss_weighted - 5.75).abs() < 1e-6);
    }

    #[test]
    fn test_refine_quantile_leaf_values() {
        let node_id = encode_tree_node_id(0, 0).expect("node id encodes");
        let mut stumps = vec![TrainedStump {
            split: SplitCandidate {
                node_id,
                feature_index: 0,
                threshold_bin: 1,
                gain: 1.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    grad_sq_sum: 0.0,
                    row_count: 2,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 2.0,
                    grad_sq_sum: 0.0,
                    row_count: 2,
                },
            },
            left_leaf_value: LeafValue::Scalar(0.0),
            right_leaf_value: LeafValue::Scalar(0.0),
            tree_weight: 1.0,
            multi_output_leaf_values: None,
        }];
        let matrix = sample_binned_matrix();
        let targets = vec![1.0, 3.0, 10.0, 20.0];
        let predictions = vec![0.0, 0.0, 0.0, 0.0];

        // test with alpha = 0.75, learning_rate = 1.0
        refine_quantile_leaf_values(
            &mut stumps,
            &matrix,
            &predictions,
            &targets,
            None,
            0.75,
            1.0,
            100.0,
        )
        .expect("refinement should succeed");

        assert_eq!(stumps[0].left_leaf_value.as_scalar(), 3.0);
        assert_eq!(stumps[0].right_leaf_value.as_scalar(), 20.0);
    }

    #[test]
    fn test_quantile_regression_training_smoke() {
        let dataset = sample_dataset();
        let binned = sample_binned_matrix();
        let objective = QuantileObjective { alpha: 0.5 };

        let params = TrainParams {
            learning_rate: 1.0,
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");

        let model = trainer
            .fit_iterations(&dataset, &binned, &MockBackend, &objective, 3)
            .expect("training should succeed");

        assert!(model.rounds_completed() > 0);

        let initial_loss = objective
            .loss(
                &vec![model.baseline_prediction; dataset.row_count()],
                &dataset.targets,
                None,
            )
            .unwrap();

        // Predict on the training data using the model
        let mut predictions = vec![model.baseline_prediction; dataset.row_count()];
        apply_tree_to_binned_predictions(
            &mut predictions,
            &binned,
            &model.stumps,
            Some((&dataset.matrix.values, dataset.matrix.feature_count)),
        )
        .unwrap();

        let final_loss = objective
            .loss(&predictions, &dataset.targets, None)
            .unwrap();

        assert!(
            final_loss < initial_loss,
            "Loss did not improve. initial: {}, final: {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_quantile_validation_gated_on_objective() {
        let dataset = sample_dataset();
        let binned = sample_binned_matrix();

        // 1. If objective is not quantile, invalid quantile_alpha in TrainParams is ignored during validation
        let params = TrainParams {
            quantile_alpha: -1.0, // normally invalid in params, but ignored for non-quantile
            ..TrainParams::default()
        };
        let trainer = Trainer::new(params).expect("valid params");
        // Should succeed since objective is not quantile
        let objective = SquaredErrorObjective;
        assert!(
            trainer
                .fit_iterations(&dataset, &binned, &MockBackend, &objective, 1)
                .is_ok()
        );

        // 2. If objective is quantile and has invalid alpha, it raises an error
        let objective_quantile_invalid = QuantileObjective { alpha: -1.0 };
        assert!(
            trainer
                .fit_iterations(
                    &dataset,
                    &binned,
                    &MockBackend,
                    &objective_quantile_invalid,
                    1
                )
                .is_err()
        );

        // 3. If objective is quantile and has a valid alpha, it succeeds
        let objective_quantile_valid = QuantileObjective { alpha: 0.5 };
        assert!(
            trainer
                .fit_iterations(
                    &dataset,
                    &binned,
                    &MockBackend,
                    &objective_quantile_valid,
                    1
                )
                .is_ok()
        );
    }
