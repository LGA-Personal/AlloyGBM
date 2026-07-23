from __future__ import annotations

import pickle
import tempfile
import unittest

import numpy as np

from alloygbm import GBMRegressor, MultiLabelGBMRanker


def _exclusive_fixture(rows: int = 256, width: int = 8) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((rows, width), dtype=np.float32)
    active = np.arange(rows) % width
    X[np.arange(rows), active] = 1.0
    y = (active.astype(np.float32) - 2.0) ** 2
    return X, y


class ExactFeatureBundlingTests(unittest.TestCase):
    def test_exact_bundling_preserves_artifact_and_predictions(self) -> None:
        X, y = _exclusive_fixture()
        parameters = {
            "n_estimators": 8,
            "max_depth": 4,
            "training_policy": "manual",
            "seed": 17,
        }

        baseline = GBMRegressor(**parameters).fit(X, y)
        bundled = GBMRegressor(**parameters, feature_bundling="exact").fit(X, y)

        self.assertEqual(bundled.artifact_bytes, baseline.artifact_bytes)
        np.testing.assert_array_equal(bundled.predict(X), baseline.predict(X))
        self.assertEqual(
            bundled.feature_bundling_diagnostics_,
            {
                "active": True,
                "original_feature_count": 8,
                "effective_feature_count": 1,
                "bundle_count": 1,
                "bundled_feature_count": 8,
                "skipped_feature_count": 0,
                "observed_conflict_count": 0,
            },
        )

    def test_exact_bundling_skips_constrained_features(self) -> None:
        X, y = _exclusive_fixture(width=4)

        model = GBMRegressor(
            n_estimators=3,
            training_policy="manual",
            feature_bundling="exact",
            monotone_constraints=[1, -1, 1, -1],
        ).fit(X[:, :4], y)

        self.assertFalse(model.feature_bundling_diagnostics_["active"])
        self.assertEqual(model.feature_bundling_diagnostics_["skipped_feature_count"], 4)

    def test_feature_bundling_diagnostics_survive_pickle(self) -> None:
        X, y = _exclusive_fixture()
        model = GBMRegressor(
            n_estimators=3,
            training_policy="manual",
            feature_bundling="exact",
        ).fit(X, y)

        restored = pickle.loads(pickle.dumps(model))

        self.assertEqual(
            restored.feature_bundling_diagnostics_,
            model.feature_bundling_diagnostics_,
        )
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    def test_feature_bundling_diagnostics_survive_model_file(self) -> None:
        X, y = _exclusive_fixture()
        model = GBMRegressor(
            n_estimators=3,
            training_policy="manual",
            feature_bundling="exact",
        ).fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".alloygbm") as handle:
            model.save_model(handle.name)
            restored = GBMRegressor.load_model(handle.name)

        self.assertEqual(
            restored.feature_bundling_diagnostics_,
            model.feature_bundling_diagnostics_,
        )
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    def test_wide_bundle_storage_preserves_skipped_nan_feature(self) -> None:
        X, y = _exclusive_fixture(rows=128, width=4)
        X = np.column_stack([X, np.zeros(len(X), dtype=np.float32)])
        X[0, -1] = np.nan
        parameters = {
            "n_estimators": 4,
            "max_depth": 3,
            "training_policy": "manual",
            "continuous_binning_strategy": "linear",
            "continuous_binning_max_bins": 256,
            "seed": 29,
        }

        baseline = GBMRegressor(**parameters).fit(X, y)
        bundled = GBMRegressor(**parameters, feature_bundling="exact").fit(X, y)

        self.assertTrue(bundled.feature_bundling_diagnostics_["active"])
        self.assertEqual(bundled.artifact_bytes, baseline.artifact_bytes)
        np.testing.assert_array_equal(bundled.predict(X), baseline.predict(X))

    def test_validation_bin_outside_training_span_forces_fallback(self) -> None:
        X = np.zeros((8, 2), dtype=np.float32)
        X[0, 0] = 1.0
        X[1, 1] = 1.0
        y = np.arange(8, dtype=np.float32)
        X_validation = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_validation = np.array([0.0, 1.0], dtype=np.float32)
        parameters = {
            "n_estimators": 4,
            "max_depth": 2,
            "training_policy": "manual",
            "early_stopping_rounds": 1,
            "seed": 31,
        }

        baseline = GBMRegressor(**parameters).fit(
            X, y, eval_set=(X_validation, y_validation)
        )
        bundled = GBMRegressor(
            **parameters, feature_bundling="exact"
        ).fit(X, y, eval_set=(X_validation, y_validation))

        self.assertFalse(bundled.feature_bundling_diagnostics_["active"])
        self.assertGreater(
            bundled.feature_bundling_diagnostics_["observed_conflict_count"], 0
        )
        self.assertEqual(bundled.artifact_bytes, baseline.artifact_bytes)

    def test_exhausted_u16_storage_forces_fallback(self) -> None:
        X = np.zeros((8, 3), dtype=np.float32)
        X[0, 0] = 1.0
        X[1, 1] = 1.0
        X[0, 2] = 65535.0
        y = np.arange(8, dtype=np.float32)
        parameters = {
            "n_estimators": 3,
            "max_depth": 2,
            "training_policy": "manual",
            "continuous_binning_max_bins": 65535,
            "seed": 37,
        }

        baseline = GBMRegressor(**parameters).fit(X, y)
        bundled = GBMRegressor(**parameters, feature_bundling="exact").fit(X, y)

        self.assertFalse(bundled.feature_bundling_diagnostics_["active"])
        self.assertEqual(bundled.artifact_bytes, baseline.artifact_bytes)
        np.testing.assert_array_equal(bundled.predict(X), baseline.predict(X))

    def test_shap_and_importances_keep_original_feature_shape(self) -> None:
        X, y = _exclusive_fixture()
        model = GBMRegressor(
            n_estimators=5,
            training_policy="manual",
            feature_bundling="exact",
        ).fit(X, y)

        expected_value, shap_values = model.shap_values(
            X[:16], include_expected_value=True
        )
        importances = model.feature_importances(X[:64])

        self.assertEqual(np.asarray(shap_values).shape, (16, X.shape[1]))
        self.assertEqual(len(importances), X.shape[1])
        np.testing.assert_allclose(
            np.asarray(shap_values).sum(axis=1) + expected_value,
            model.predict(X[:16]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_interaction_constrained_features_are_skipped_individually(self) -> None:
        X, y = _exclusive_fixture(width=4)

        model = GBMRegressor(
            n_estimators=3,
            training_policy="manual",
            feature_bundling="exact",
            interaction_constraints=[[0, 1]],
        ).fit(X[:, :4], y)

        self.assertTrue(model.feature_bundling_diagnostics_["active"])
        self.assertEqual(model.feature_bundling_diagnostics_["skipped_feature_count"], 2)
        self.assertEqual(model.feature_bundling_diagnostics_["effective_feature_count"], 3)

    def test_categorical_features_are_skipped_individually(self) -> None:
        X, y = _exclusive_fixture(width=4)
        categories = [["a" if value else "b" for value in X[:, 0]]]

        model = GBMRegressor(
            n_estimators=3,
            training_policy="manual",
            feature_bundling="exact",
            categorical_feature_indices=[0],
        ).fit(X[:, :4], y, categorical_feature_values_list=categories)

        self.assertTrue(model.feature_bundling_diagnostics_["active"])
        self.assertEqual(model.feature_bundling_diagnostics_["skipped_feature_count"], 1)
        self.assertEqual(model.feature_bundling_diagnostics_["effective_feature_count"], 2)

    def test_independent_multi_label_ranker_reports_bundling(self) -> None:
        X, y = _exclusive_fixture(rows=128, width=4)
        labels = np.column_stack([y, y[::-1]])
        group = np.repeat(np.arange(16), 8)

        model = MultiLabelGBMRanker(
            n_estimators=2,
            max_depth=2,
            training_policy="manual",
            feature_bundling="exact",
        ).fit(X, labels, group=group)

        self.assertTrue(model.feature_bundling_diagnostics_["active"])
        self.assertEqual(model.feature_bundling_diagnostics_["effective_feature_count"], 1)

        with tempfile.NamedTemporaryFile(suffix=".alloygbm") as handle:
            model.save_model(handle.name)
            restored = MultiLabelGBMRanker.load_model(handle.name)

        self.assertEqual(
            restored.feature_bundling_diagnostics_,
            model.feature_bundling_diagnostics_,
        )
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    def test_joint_multi_label_ranker_rejects_bundling_explicitly(self) -> None:
        X, y = _exclusive_fixture(rows=32, width=4)
        labels = np.column_stack([y, y[::-1]])
        group = np.repeat(np.arange(4), 8)
        model = MultiLabelGBMRanker(
            multi_label_mode="joint",
            n_estimators=2,
            feature_bundling="exact",
        )

        with self.assertRaisesRegex(NotImplementedError, "independent"):
            model.fit(X, labels, group=group)


if __name__ == "__main__":
    unittest.main()
