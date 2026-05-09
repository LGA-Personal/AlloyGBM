"""DRO leaf-solver integration tests."""

from __future__ import annotations

import pickle
import tempfile
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, 1.1, 1.8, 2.0, 2.7, 2.9, 3.6, 3.7], dtype=np.float32)
    return X, y


class DroLeafSolverTests(unittest.TestCase):
    def test_zero_radius_predictions_match_standard(self) -> None:
        X, y = _regression_data()
        common = dict(
            n_estimators=4,
            max_depth=2,
            learning_rate=0.2,
            seed=17,
            deterministic=True,
        )
        standard = GBMRegressor(**common).fit(X, y)
        dro = GBMRegressor(leaf_solver="dro", dro_radius=0.0, **common).fit(X, y)

        self.assertTrue(np.allclose(standard.predict(X), dro.predict(X), atol=0.0))

    def test_dro_regressor_pickle_and_save_load_preserve_predictions(self) -> None:
        X, y = _regression_data()
        model = GBMRegressor(
            leaf_solver="dro",
            dro_radius=0.05,
            n_estimators=5,
            max_depth=2,
            seed=19,
        ).fit(X, y)
        original = np.asarray(model.predict(X), dtype=np.float64)

        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.get_params()["leaf_solver"], "dro")
        self.assertEqual(restored.get_params()["dro_metric"], "wasserstein")
        self.assertTrue(np.allclose(original, restored.predict(X), atol=1e-6))

        with tempfile.NamedTemporaryFile(suffix=".agbm") as f:
            model.save_model(f.name)
            loaded = GBMRegressor.load_model(f.name)
        self.assertEqual(loaded.get_params()["leaf_solver"], "dro")
        self.assertTrue(np.allclose(original, loaded.predict(X), atol=1e-6))

    def test_dro_classifier_binary_and_multiclass_train(self) -> None:
        X, _ = _regression_data()
        y_binary = (X[:, 0] > 1.0).astype(np.int64)
        binary = GBMClassifier(
            leaf_solver="dro",
            n_estimators=4,
            max_depth=2,
            seed=23,
        ).fit(X, y_binary)
        self.assertEqual(binary.predict_proba(X).shape, (len(X), 2))

        y_multi = np.asarray([0, 0, 1, 1, 2, 2, 2, 1], dtype=np.int64)
        multiclass = GBMClassifier(
            leaf_solver="dro",
            n_estimators=3,
            max_depth=2,
            seed=29,
        ).fit(X, y_multi)
        self.assertEqual(multiclass.predict_proba(X).shape, (len(X), 3))

    def test_dro_ranker_trains(self) -> None:
        X, _ = _regression_data()
        y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.float32)
        group = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        ranker = GBMRanker(
            leaf_solver="dro",
            n_estimators=3,
            max_depth=2,
            seed=31,
            training_policy="manual",
        ).fit(X, y, group=group)
        self.assertEqual(len(ranker.predict(X)), len(X))

    def test_dro_with_morph_trains(self) -> None:
        X, y = _regression_data()
        model = GBMRegressor(
            leaf_solver="dro",
            training_mode="morph",
            n_estimators=3,
            max_depth=2,
            seed=37,
        ).fit(X, y)
        self.assertEqual(len(model.predict(X)), len(X))


if __name__ == "__main__":
    unittest.main()
