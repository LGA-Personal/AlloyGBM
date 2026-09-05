"""Tests for GBMClassifier and objective-aware training metric tracking."""

from __future__ import annotations

import math
import pickle
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRegressor, accuracy, log_loss


class GBMClassifierTests(unittest.TestCase):
    """Tests for binary classification with GBMClassifier."""

    def _make_binary_dataset(
        self, n_train: int = 100, n_val: int = 30, n_features: int = 3, seed: int = 42
    ) -> tuple:
        rng = np.random.RandomState(seed)
        X_train = rng.randn(n_train, n_features).astype(np.float32)
        y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(np.float32)
        X_val = rng.randn(n_val, n_features).astype(np.float32)
        y_val = (X_val[:, 0] + 0.5 * X_val[:, 1] > 0).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_fit_and_predict_returns_class_labels(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=10, seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_train)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(y_train))
        self.assertTrue(all(p in (0, 1) for p in preds))

    def test_predict_proba_returns_probabilities(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=10, seed=42)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_train)
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape, (len(y_train), 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_predict_log_proba(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=10, seed=42)
        clf.fit(X_train, y_train)
        log_probs = clf.predict_log_proba(X_train)
        probs = clf.predict_proba(X_train)
        self.assertEqual(log_probs.shape, probs.shape)
        expected = np.log(np.clip(probs, 1e-15, None))
        self.assertTrue(np.allclose(log_probs, expected, atol=1e-10))

    def test_fitted_attributes(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=5, seed=42)
        clf.fit(X_train, y_train)
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        self.assertEqual(clf.n_classes_, 2)
        self.assertIsNotNone(clf.n_estimators_)

    def test_validates_non_integer_targets(self) -> None:
        X = np.array([[1], [2], [3]], dtype=np.float32)
        clf = GBMClassifier(n_estimators=3, seed=42)
        with self.assertRaisesRegex(ValueError, "Unknown label type: continuous"):
            clf.fit(X, [0.0, 0.5, 1.0])

    def test_validates_at_least_two_classes_present(self) -> None:
        X = np.array([[1], [2], [3]], dtype=np.float32)
        clf = GBMClassifier(n_estimators=3, seed=42)
        with self.assertRaisesRegex(ValueError, "at least 2 classes"):
            clf.fit(X, [0, 0, 0])

    def test_early_stopping_with_eval_set(self) -> None:
        X_train, y_train, X_val, y_val = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=100, early_stopping_rounds=5, seed=42)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        self.assertIsNotNone(clf.best_iteration_)
        self.assertLess(clf.n_estimators_, 100)

    def test_pickle_roundtrip(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        clf = GBMClassifier(n_estimators=5, seed=42)
        clf.fit(X_train, y_train)
        original_preds = clf.predict_proba(X_train)
        restored = pickle.loads(pickle.dumps(clf))
        restored_preds = restored.predict_proba(X_train)
        self.assertTrue(np.allclose(original_preds, restored_preds, atol=1e-5))

    def test_objective_name_is_binary_crossentropy(self) -> None:
        clf = GBMClassifier()
        self.assertEqual(clf._objective_name(), "binary_crossentropy")

    def test_rejects_explicit_string_objectives_with_alternative_estimators(self) -> None:
        X_train, y_train, _, _ = self._make_binary_dataset()
        for objective in ("quantile", "poisson"):
            with self.subTest(objective=objective):
                clf = GBMClassifier(objective=objective)
                with self.assertRaisesRegex(ValueError, "objective is auto-detected") as cm:
                    clf.fit(X_train, y_train)
                message = str(cm.exception)
                self.assertIn(f"objective={objective!r}", message)
                self.assertIn(f"GBMRegressor(objective={objective!r})", message)
                self.assertIn(f"GBMRanker(ranking_objective={objective!r})", message)

                clf = GBMClassifier()
                clf.set_params(objective=objective)
                with self.assertRaisesRegex(ValueError, "objective is auto-detected") as cm:
                    clf.fit(X_train, y_train)
                self.assertIn(f"GBMRanker(ranking_objective={objective!r})", str(cm.exception))

    def test_accuracy_metric(self) -> None:
        self.assertEqual(accuracy([0, 1, 1, 0], [0, 1, 1, 0]), 1.0)
        self.assertEqual(accuracy([0, 1, 1, 0], [1, 0, 0, 1]), 0.0)
        self.assertEqual(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)

    def test_log_loss_metric(self) -> None:
        ll = log_loss([0, 1], [0.1, 0.9])
        self.assertGreater(ll, 0.0)
        self.assertLess(ll, 0.2)
        ll_worse = log_loss([0, 1], [0.5, 0.5])
        self.assertGreater(ll_worse, ll)


class TrainingMetricTrackingTests(unittest.TestCase):
    """Tests for objective-aware evals_result_ tracking."""

    def test_regressor_evals_result_has_mse_and_rmse(self) -> None:
        X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        reg = GBMRegressor(n_estimators=5, seed=42)
        reg.fit(X, y)
        er = reg.evals_result_
        self.assertIn("train", er)
        self.assertIn("rmse", er["train"])
        self.assertIn("mse", er["train"])
        self.assertEqual(len(er["train"]["rmse"]), 5)
        self.assertEqual(len(er["train"]["mse"]), 5)
        # rmse^2 should equal mse
        for rmse_val, mse_val in zip(er["train"]["rmse"], er["train"]["mse"]):
            self.assertAlmostEqual(rmse_val ** 2, mse_val, places=3)

    def test_regressor_evals_result_with_eval_set(self) -> None:
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2).astype(np.float32)
        y_train = X_train[:, 0].astype(np.float32)
        X_val = rng.randn(20, 2).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)
        reg = GBMRegressor(n_estimators=5, seed=42)
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))
        er = reg.evals_result_
        self.assertIn("validation", er)
        self.assertIn("rmse", er["validation"])
        self.assertIn("mse", er["validation"])
        self.assertEqual(len(er["validation"]["rmse"]), 5)
        self.assertEqual(len(er["validation"]["mse"]), 5)

    def test_classifier_evals_result_has_logloss_and_rmse(self) -> None:
        X = np.array([[0], [0], [1], [1], [2], [2]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        clf = GBMClassifier(n_estimators=5, seed=42)
        clf.fit(X, y)
        er = clf.evals_result_
        self.assertIn("train", er)
        self.assertIn("logloss", er["train"])
        self.assertIn("rmse", er["train"])
        self.assertEqual(len(er["train"]["logloss"]), 5)
        self.assertEqual(len(er["train"]["rmse"]), 5)
        # logloss values should be positive
        self.assertTrue(all(v > 0 for v in er["train"]["logloss"]))

    def test_classifier_evals_result_with_eval_set(self) -> None:
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.float32)
        X_val = rng.randn(20, 2).astype(np.float32)
        y_val = (X_val[:, 0] > 0).astype(np.float32)
        clf = GBMClassifier(n_estimators=10, early_stopping_rounds=5, seed=42)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        er = clf.evals_result_
        self.assertIn("validation", er)
        self.assertIn("logloss", er["validation"])
        self.assertIn("rmse", er["validation"])

    def test_regressor_no_logloss_key(self) -> None:
        X = np.array([[1], [2], [3]], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)
        reg = GBMRegressor(n_estimators=3, seed=42)
        reg.fit(X, y)
        self.assertNotIn("logloss", reg.evals_result_["train"])

    def test_classifier_no_mse_key(self) -> None:
        X = np.array([[0], [0], [1], [1], [2], [2]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        clf = GBMClassifier(n_estimators=3, seed=42)
        clf.fit(X, y)
        self.assertNotIn("mse", clf.evals_result_["train"])


if __name__ == "__main__":
    unittest.main()


class ColsampleBynodeMulticlassContractTests(unittest.TestCase):
    """`colsample_bynode` must not be silently ignored on multiclass.

    The parameter is enforced only by the single-output tree builders; the
    multiclass round loop passes no colsample context. Before this contract
    the parameter was accepted on a multiclass fit and produced a
    byte-identical (unsampled) model, which is the same silent-no-op failure
    mode as the exclusive-feature-bundling case. Joint `MultiLabelGBMRanker`
    already rejects kwargs it cannot honour; multiclass now matches.
    """

    @staticmethod
    def _data():
        rng = np.random.default_rng(4)
        X = rng.standard_normal((300, 6)).astype(np.float32)
        signal = 2.0 * X[:, 0] + X[:, 1]
        multiclass = np.digitize(signal, np.quantile(signal, [0.33, 0.66]))
        binary = (signal > 0).astype(int)
        return X, binary, multiclass

    def test_multiclass_rejects_colsample_bynode(self) -> None:
        X, _, multiclass = self._data()
        with self.assertRaises(ValueError) as caught:
            GBMClassifier(n_estimators=3, colsample_bynode=0.5).fit(X, multiclass)
        message = str(caught.exception)
        self.assertIn("colsample_bynode", message)
        self.assertIn("multiclass", message)

    def test_multiclass_accepts_the_inert_default(self) -> None:
        X, _, multiclass = self._data()
        model = GBMClassifier(n_estimators=3, colsample_bynode=1.0).fit(X, multiclass)
        self.assertEqual(len(np.asarray(model.predict(X))), len(multiclass))

    def test_binary_still_applies_colsample_bynode(self) -> None:
        # Guards the other direction: the rejection must not disable the
        # parameter where it genuinely works.
        X, binary, _ = self._data()
        sampled = GBMClassifier(n_estimators=15, seed=3, colsample_bynode=0.3).fit(
            X, binary
        )
        full = GBMClassifier(n_estimators=15, seed=3).fit(X, binary)
        self.assertNotEqual(sampled.artifact_bytes, full.artifact_bytes)
