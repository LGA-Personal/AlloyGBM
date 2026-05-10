import pickle
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
            neutralization="per_round_gradient", factor_neutralization_lambda=1e-4
        )
        params = model.get_params()
        self.assertEqual(params["neutralization"], "per_round_gradient")
        self.assertEqual(params["factor_neutralization_lambda"], 1e-4)
        model.set_params(neutralization="pre_target", factor_penalty=0.0)
        self.assertEqual(model.neutralization, "pre_target")

    def test_invalid_constructor_values(self):
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="bad")
        with self.assertRaises(ValueError):
            GBMRegressor(factor_neutralization_lambda=-1.0)
        with self.assertRaises(ValueError):
            GBMRegressor(factor_penalty=-1.0)

    def test_requires_factor_exposures_when_active(self):
        x, y, _ = factor_data()
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="per_round_gradient").fit(x, y)

    def test_regressor_trains_with_per_round_gradient(self):
        x, y, f = factor_data()
        model = GBMRegressor(neutralization="per_round_gradient", n_estimators=5, seed=1)
        model.fit(x, y, factor_exposures=f)
        self.assertEqual(np.asarray(model.predict(x)).shape, (len(x),))

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
        self.assertEqual(np.asarray(model.predict(x)).shape, (len(x),))

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

    def test_classifier_repr_includes_neutralization_params(self):
        model = GBMClassifier(
            neutralization="per_round_gradient",
            factor_neutralization_lambda=1e-4,
        )
        text = repr(model)
        self.assertIn("neutralization='per_round_gradient'", text)
        self.assertIn("factor_neutralization_lambda=0.0001", text)
        self.assertIn("factor_penalty=0.0", text)

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
