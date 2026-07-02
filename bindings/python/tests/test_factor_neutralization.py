import pickle
import tempfile
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def factor_data():
    f = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(-1, 1)
    x = np.column_stack([f[:, 0], np.sin(np.arange(24, dtype=np.float32))]).astype(
        np.float32
    )
    y = (2.0 * f[:, 0] + 0.1 * x[:, 1]).astype(np.float32)
    return x, y, f


def bridge_train_kwargs(**overrides):
    kwargs = {
        "values": [0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 1.0],
        "row_count": 4,
        "feature_count": 2,
        "targets": [0.0, 1.0, 2.0, 3.0],
        "learning_rate": 0.1,
        "max_depth": 2,
        "row_subsample": 1.0,
        "col_subsample": 1.0,
        "min_validation_improvement": 0.0,
        "seed": 1,
        "deterministic": True,
        "rounds": 1,
    }
    kwargs.update(overrides)
    return kwargs


class FactorNeutralizationTests(unittest.TestCase):
    def test_params_roundtrip(self):
        model = GBMRegressor(
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
            factor_exposure_transform="center",
        )
        params = model.get_params()
        self.assertEqual(params["neutralization"], "per_round_gradient")
        self.assertEqual(params["factor_neutralization_lambda"], 1e-4)
        self.assertEqual(params["factor_exposure_transform"], "center")
        model.set_params(neutralization="pre_target", factor_penalty=0.0)
        self.assertEqual(model.neutralization, "pre_target")

    def test_invalid_constructor_values(self):
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="bad")
        with self.assertRaises(ValueError):
            GBMRegressor(factor_neutralization_lambda=-1.0)
        with self.assertRaises(ValueError):
            GBMRegressor(factor_penalty=-1.0)
        with self.assertRaises(ValueError):
            GBMRegressor(factor_exposure_transform="bad")

    def test_requires_factor_exposures_when_active(self):
        x, y, _ = factor_data()
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="per_round_gradient").fit(x, y)

    def test_regressor_trains_with_per_round_gradient(self):
        x, y, f = factor_data()
        model = GBMRegressor(neutralization="per_round_gradient", n_estimators=5, seed=1)
        model.fit(x, y, factor_exposures=f)
        self.assertEqual(np.asarray(model.predict(x)).shape, (len(x),))

    def test_factor_exposures_are_centered_when_requested(self):
        x, y, f = factor_data()
        shifted = f + np.float32(10.0)
        model = GBMRegressor(
            neutralization="per_round_gradient",
            factor_exposure_transform="center",
            n_estimators=2,
            seed=5,
        )
        model.fit(x, y, factor_exposures=shifted)
        diagnostics = model.factor_exposure_diagnostics_
        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics["transform"], "center")
        self.assertAlmostEqual(diagnostics["means"][0], float(np.mean(shifted[:, 0])))
        self.assertAlmostEqual(diagnostics["stds"][0], float(np.std(shifted[:, 0])))

    def test_factor_exposures_are_standardized_when_requested(self):
        x, y, f = factor_data()
        scaled = np.column_stack([f[:, 0] * 3.0 + 5.0, np.ones(len(f), dtype=np.float32)])
        model = GBMRegressor(
            neutralization="split_penalty",
            factor_penalty=0.1,
            factor_exposure_transform="standardize",
            n_estimators=2,
            seed=6,
        )
        model.fit(x, y, factor_exposures=scaled)
        diagnostics = model.factor_exposure_diagnostics_
        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics["transform"], "standardize")
        self.assertAlmostEqual(
            diagnostics["means"][0], float(np.mean(scaled[:, 0])), places=6
        )
        self.assertAlmostEqual(diagnostics["stds"][1], 0.0)

    def test_factor_neutralization_with_dro_and_morph(self):
        x, y, f = factor_data()
        model = GBMRegressor(
            neutralization="per_round_gradient",
            leaf_solver="dro",
            dro_radius=0.01,
            training_mode="morph",
            n_estimators=5,
            seed=2,
        )
        model.fit(x, y, factor_exposures=f)
        predictions = np.asarray(model.predict(x))
        self.assertEqual(predictions.shape, (len(x),))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertGreater(float(np.ptp(predictions)), 1e-6)
        self.assertEqual(model.neutralization, "per_round_gradient")
        self.assertEqual(model.leaf_solver, "dro")
        self.assertEqual(model.training_mode, "morph")

    def test_regressor_trains_with_split_penalty(self):
        x, y, f = factor_data()
        model = GBMRegressor(
            neutralization="split_penalty",
            factor_penalty=0.1,
            n_estimators=5,
            seed=1,
        )
        model.fit(x, y, factor_exposures=f)
        self.assertEqual(np.asarray(model.predict(x)).shape, (len(x),))

    def test_split_penalty_rejects_linear_leaves(self):
        x, y, f = factor_data()
        with self.assertRaises(ValueError):
            GBMRegressor(
                neutralization="split_penalty",
                factor_penalty=0.1,
                leaf_model="linear",
            ).fit(x, y, factor_exposures=f)

    def test_set_params_invalid_neutralization_combo_is_atomic(self):
        model = GBMRegressor(neutralization="split_penalty", factor_penalty=0.1)
        with self.assertRaises(ValueError):
            model.set_params(neutralization="none")
        self.assertEqual(model.neutralization, "split_penalty")
        self.assertEqual(model.factor_penalty, 0.1)

    def test_bridge_rejects_factor_penalty_without_split_penalty(self):
        from alloygbm._alloygbm import train_regression_artifact_dense_with_summary

        with self.assertRaises(ValueError):
            train_regression_artifact_dense_with_summary(
                **bridge_train_kwargs(neutralization="none", factor_penalty=0.1)
            )

    def test_bridge_rejects_split_penalty_with_linear_leaf_model(self):
        from alloygbm._alloygbm import train_regression_artifact_dense_with_summary

        with self.assertRaises(ValueError):
            train_regression_artifact_dense_with_summary(
                **bridge_train_kwargs(
                    neutralization="split_penalty",
                    factor_penalty=0.1,
                    leaf_model="linear",
                    factor_exposure_values=[1.0, 2.0, 3.0, 4.0],
                    factor_exposure_row_count=4,
                    factor_exposure_factor_count=1,
                )
            )

    def test_ranker_sorts_factor_exposures_with_unsorted_groups(self):
        x, y, f = factor_data()
        group = np.repeat([2, 0, 1], 8)
        model = GBMRanker(
            neutralization="per_round_gradient",
            n_estimators=5,
            seed=1,
        )
        model.fit(x, y, group=group, factor_exposures=f)
        self.assertEqual(len(model.predict(x)), len(x))

    def test_pre_target_rejected_for_classifier_and_ranker(self):
        x, y, f = factor_data()
        with self.assertRaises(ValueError):
            GBMClassifier(neutralization="pre_target").fit(
                x, (y > 0).astype(np.int32), factor_exposures=f
            )
        with self.assertRaises(ValueError):
            GBMRanker(neutralization="pre_target").fit(
                x, y, group=np.repeat([0, 1, 2], 8), factor_exposures=f
            )

    def test_pre_target_rejected_for_binary_crossentropy_regressor(self):
        x, y, f = factor_data()
        with self.assertRaisesRegex(
            ValueError, "pre_target.*GBMRegressor squared-error"
        ):
            GBMRegressor(
                objective="binary_crossentropy",
                neutralization="pre_target",
                n_estimators=2,
            ).fit(x, (y > 0).astype(np.float32), factor_exposures=f)

    def test_pre_target_rejected_for_custom_objective_regressor(self):
        x, y, f = factor_data()

        def custom_mse(y_true, y_pred):
            return y_pred - y_true, np.ones_like(y_true)

        with self.assertRaisesRegex(
            ValueError, "pre_target.*GBMRegressor squared-error"
        ):
            GBMRegressor(
                objective=custom_mse,
                neutralization="pre_target",
                n_estimators=2,
            ).fit(x, y, factor_exposures=f)

    def test_pre_target_rejects_eval_set_without_validation_exposures(self):
        x, y, f = factor_data()
        with self.assertRaisesRegex(
            ValueError, "pre_target.*eval_set.*validation factor_exposures"
        ):
            GBMRegressor(
                neutralization="pre_target",
                n_estimators=2,
                early_stopping_rounds=1,
            ).fit(x, y, eval_set=(x, y), factor_exposures=f)

    def test_bridge_rejects_pre_target_before_non_squared_error_dispatch(self):
        from alloygbm._alloygbm import train_regression_artifact_dense_with_summary

        with self.assertRaisesRegex(
            ValueError, "pre_target.*GBMRegressor squared-error"
        ):
            train_regression_artifact_dense_with_summary(
                **bridge_train_kwargs(
                    objective="binary_crossentropy",
                    neutralization="pre_target",
                    targets=[0.0, 1.0, 1.0, 0.0],
                    factor_exposure_values=[1.0, 2.0, 3.0, 4.0],
                    factor_exposure_row_count=4,
                    factor_exposure_factor_count=1,
                )
            )

    def test_bridge_rejects_pre_target_with_validation_targets(self):
        from alloygbm._alloygbm import train_regression_artifact_dense_with_summary

        with self.assertRaisesRegex(
            ValueError, "pre_target.*validation targets.*validation factor_exposures"
        ):
            train_regression_artifact_dense_with_summary(
                **bridge_train_kwargs(
                    neutralization="pre_target",
                    factor_exposure_values=[1.0, 2.0, 3.0, 4.0],
                    factor_exposure_row_count=4,
                    factor_exposure_factor_count=1,
                    validation_values=[0.0, 0.0, 1.0, 1.0],
                    validation_row_count=2,
                    validation_targets=[0.0, 1.0],
                    early_stopping_rounds=1,
                )
            )

    def test_classifier_repr_includes_neutralization_params(self):
        model = GBMClassifier(
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
            factor_exposure_transform="center",
        )
        text = repr(model)
        self.assertIn("neutralization='per_round_gradient'", text)
        self.assertIn("factor_neutralization_lambda=0.0001", text)
        self.assertIn("factor_penalty=0.0", text)
        self.assertIn("factor_exposure_transform='center'", text)
        self.assertIn(
            "factor_exposure_transform='standardize'",
            repr(
                GBMRanker(
                    neutralization="per_round_gradient",
                    factor_exposure_transform="standardize",
                )
            ),
        )

    def test_pre_target_rejected_by_classifier_and_ranker_constructors(self):
        with self.assertRaises(ValueError):
            GBMClassifier(neutralization="pre_target")
        with self.assertRaises(ValueError):
            GBMRanker(neutralization="pre_target")

    def test_pre_target_rejected_by_classifier_and_ranker_set_params(self):
        with self.assertRaises(ValueError):
            GBMClassifier().set_params(neutralization="pre_target")
        with self.assertRaises(ValueError):
            GBMRanker().set_params(neutralization="pre_target")

    def test_pickle_preserves_params_and_predictions(self):
        x, y, f = factor_data()
        model = GBMRegressor(
            neutralization="per_round_gradient", n_estimators=5, seed=1
        ).fit(x, y, factor_exposures=f)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict(x), model.predict(x), atol=1e-6)
        self.assertEqual(restored.neutralization, "per_round_gradient")

    def test_neutralized_warm_start_succeeds_when_exposures_supplied(self):
        # v0.7.1: warm-starting a neutralized model is supported as long as
        # the caller supplies the same `factor_exposures` matrix.  Without it
        # we fail loudly so the caller doesn't accidentally project against
        # a different column space mid-training.
        x, y, f = factor_data()
        model = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            warm_start=True,
            seed=1,
        )
        model.fit(x, y, factor_exposures=f)
        model.n_estimators = 3
        # Re-fitting with exposures should succeed and add the requested rounds.
        model.fit(x, y, factor_exposures=f)
        # Missing exposures must raise per the v0.7.1 contract.
        with self.assertRaisesRegex(
            ValueError,
            "warm-start training of a neutralized model requires factor_exposures",
        ):
            model.fit(x, y)

    def test_neutralized_init_model_warm_start_succeeds_when_exposures_supplied(self):
        # An init_model carrying neutralization metadata can be resumed by a
        # new estimator instance, again contingent on factor_exposures being
        # supplied.
        x, y, f = factor_data()
        base = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            seed=1,
        ).fit(x, y, factor_exposures=f)
        model = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            seed=2,
        )
        # Success path.
        model.fit(x, y, init_model=base, factor_exposures=f)
        # Missing exposures is rejected even when init_model has neutralization
        # metadata, because we can't reconstruct the original exposures from
        # the artifact alone (they're not persisted).
        model2 = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            seed=2,
        )
        with self.assertRaisesRegex(
            ValueError,
            "warm-start training of a neutralized model requires factor_exposures",
        ):
            model2.fit(x, y, init_model=base)

    def test_neutralized_init_model_rejected_after_params_mutated_to_none(self):
        # Even if the caller mutates `neutralization` to "none" on the init
        # model, the persisted fit contract still says the original training
        # used neutralization — so the contract check is driven from the
        # *fitted* neutralization, not the current params.  The warm-start
        # mode-mismatch guard rejects the resume with a clear error.
        x, y, f = factor_data()
        base = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            seed=1,
        ).fit(x, y, factor_exposures=f)
        base.set_params(neutralization="none")

        with self.assertRaisesRegex(
            ValueError,
            r"init_model neutralization 'per_round_gradient' does not match",
        ):
            GBMRegressor(n_estimators=2, seed=2).fit(x, y, init_model=base)

    def test_save_load_preserves_fitted_neutralization_after_params_mutation(self):
        # Same contract holds across save/load: the persisted fit
        # neutralization mode is what drives the warm-start contract check,
        # regardless of subsequent param mutations.
        x, y, f = factor_data()
        base = GBMRegressor(
            neutralization="per_round_gradient",
            n_estimators=2,
            seed=1,
        ).fit(x, y, factor_exposures=f)
        base.set_params(neutralization="none")

        with tempfile.NamedTemporaryFile(suffix=".agbm") as tmp:
            base.save_model(tmp.name)
            restored = GBMRegressor.load_model(tmp.name)

        with self.assertRaisesRegex(
            ValueError,
            r"init_model neutralization 'per_round_gradient' does not match",
        ):
            GBMRegressor(n_estimators=2, seed=2).fit(x, y, init_model=restored)
