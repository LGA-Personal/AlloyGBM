"""Tests for multi-class classification support in GBMClassifier."""

from __future__ import annotations

import math
import pickle
import tempfile
import unittest

import numpy as np

from alloygbm import GBMClassifier, multiclass_log_loss


def _make_multiclass_dataset(
    n_train: int = 200,
    n_val: int = 60,
    n_features: int = 4,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple:
    """Create a synthetic multi-class dataset with linearly separable classes."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)

    # Assign class labels based on quantiles of first feature
    boundaries = np.linspace(-2, 2, n_classes + 1)[1:-1]
    y_train = np.digitize(X_train[:, 0] + 0.3 * X_train[:, 1], boundaries).astype(
        np.int64
    )
    y_val = np.digitize(X_val[:, 0] + 0.3 * X_val[:, 1], boundaries).astype(
        np.int64
    )
    return X_train, y_train, X_val, y_val


class TestFitAndPredictThreeClasses(unittest.TestCase):
    """Synthetic data with 3 classes, verify predictions are valid class labels."""

    def test_fit_and_predict_three_classes(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=20, seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_train)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), len(y_train))
        valid_labels = set(int(c) for c in np.unique(y_train))
        for p in preds:
            self.assertIn(p, valid_labels)


class TestPredictProbaShapeAndSum(unittest.TestCase):
    """predict_proba returns shape (n, K) with rows summing to 1.0."""

    def test_predict_proba_shape_and_sum(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=20, seed=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_train)
        self.assertIsInstance(proba, np.ndarray)
        n_classes = len(np.unique(y_train))
        self.assertEqual(proba.shape, (len(y_train), n_classes))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestPredictProbaValuesInRange(unittest.TestCase):
    """All predicted probability values are in [0, 1]."""

    def test_predict_proba_values_in_range(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=20, seed=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_train)
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))


class TestClassesAttributeSetCorrectly(unittest.TestCase):
    """classes_ and n_classes_ match training labels."""

    def test_classes_attribute_set_correctly(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=4)
        clf = GBMClassifier(n_estimators=10, seed=42)
        clf.fit(X_train, y_train)
        expected_classes = sorted(set(int(c) for c in y_train))
        self.assertEqual(clf.classes_, expected_classes)
        self.assertEqual(clf.n_classes_, len(expected_classes))


class TestLabelEncodingNonContiguous(unittest.TestCase):
    """Labels {2, 5, 9} are mapped correctly; predictions use original labels."""

    def test_label_encoding_non_contiguous(self) -> None:
        rng = np.random.RandomState(42)
        n = 150
        X = rng.randn(n, 3).astype(np.float32)
        # Assign non-contiguous labels based on feature values
        raw = np.digitize(X[:, 0], [-0.5, 0.5])
        label_map_forward = {0: 2, 1: 5, 2: 9}
        y = np.array([label_map_forward[int(v)] for v in raw], dtype=np.int64)

        clf = GBMClassifier(n_estimators=20, seed=42)
        clf.fit(X, y)

        self.assertEqual(clf.classes_, [2, 5, 9])
        self.assertEqual(clf.n_classes_, 3)

        preds = clf.predict(X)
        valid_labels = {2, 5, 9}
        for p in preds:
            self.assertIn(p, valid_labels)

        # Verify internal label encoder/decoder exist
        self.assertIsNotNone(clf._label_encoder)
        self.assertIsNotNone(clf._label_decoder)
        self.assertEqual(set(clf._label_encoder.keys()), {2, 5, 9})
        self.assertEqual(set(clf._label_decoder.values()), {2, 5, 9})


class TestEarlyStoppingMulticlass(unittest.TestCase):
    """Early stopping with eval_set sets best_iteration_."""

    def test_early_stopping_multiclass(self) -> None:
        X_train, y_train, X_val, y_val = _make_multiclass_dataset(
            n_train=200, n_val=60, n_classes=3
        )
        clf = GBMClassifier(
            n_estimators=200, early_stopping_rounds=5, seed=42
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))
        self.assertIsNotNone(clf.best_iteration_)
        self.assertLessEqual(clf.n_estimators_, 200)


class TestMulticlassLogLossMetric(unittest.TestCase):
    """multiclass_log_loss metric correctness against manual computation."""

    def test_multiclass_log_loss_metric(self) -> None:
        y_true = [0, 1, 2]
        y_prob = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
        result = multiclass_log_loss(y_true, y_prob)

        # Manual computation: -mean(log(p[y_i]))
        eps = 1e-15
        probs = [0.7, 0.8, 0.6]
        expected = -sum(math.log(max(p, eps)) for p in probs) / len(probs)
        self.assertAlmostEqual(result, expected, places=10)


class TestPickleRoundtripMulticlass(unittest.TestCase):
    """Pickle/unpickle preserves multi-class predictions."""

    def test_pickle_roundtrip_multiclass(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=15, seed=42)
        clf.fit(X_train, y_train)

        original_preds = clf.predict(X_train)
        original_proba = clf.predict_proba(X_train)

        restored = pickle.loads(pickle.dumps(clf))

        restored_preds = restored.predict(X_train)
        restored_proba = restored.predict_proba(X_train)

        self.assertEqual(original_preds, restored_preds)
        np.testing.assert_allclose(original_proba, restored_proba, atol=1e-5)
        self.assertEqual(restored.classes_, clf.classes_)
        self.assertEqual(restored.n_classes_, clf.n_classes_)
        self.assertEqual(restored._label_encoder, clf._label_encoder)
        self.assertEqual(restored._label_decoder, clf._label_decoder)


class TestSaveLoadModelMulticlass(unittest.TestCase):
    """save_model/load_model roundtrip preserves multi-class state."""

    def test_save_load_model_multiclass(self) -> None:
        rng = np.random.RandomState(42)
        n = 150
        X = rng.randn(n, 3).astype(np.float32)
        raw = np.digitize(X[:, 0], [-0.5, 0.5])
        label_map_forward = {0: 2, 1: 5, 2: 9}
        y = np.array([label_map_forward[int(v)] for v in raw], dtype=np.int64)

        clf = GBMClassifier(n_estimators=15, seed=42)
        clf.fit(X, y)
        original_preds = clf.predict(X)
        original_proba = clf.predict_proba(X)

        with tempfile.NamedTemporaryFile(suffix=".agbm", delete=False) as f:
            path = f.name
        clf.save_model(path)
        restored = GBMClassifier.load_model(path)

        restored_preds = restored.predict(X)
        restored_proba = restored.predict_proba(X)

        self.assertEqual(original_preds, restored_preds)
        np.testing.assert_allclose(original_proba, restored_proba, atol=1e-5)
        self.assertEqual(restored.classes_, clf.classes_)
        self.assertEqual(restored.n_classes_, clf.n_classes_)


class TestAutoDetectBinaryVsMulticlass(unittest.TestCase):
    """Auto-detection: {0,1} -> binary, {0,1,2} -> multiclass."""

    def test_auto_detect_binary_vs_multiclass(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3).astype(np.float32)

        # Binary: labels {0, 1}
        y_bin = (X[:, 0] > 0).astype(np.int64)
        clf_bin = GBMClassifier(n_estimators=5, seed=42)
        clf_bin.fit(X, y_bin)
        self.assertEqual(clf_bin.n_classes_, 2)
        self.assertEqual(clf_bin.classes_, [0, 1])

        # Multiclass: labels {0, 1, 2}
        y_mc = np.digitize(X[:, 0], [-0.5, 0.5]).astype(np.int64)
        clf_mc = GBMClassifier(n_estimators=5, seed=42)
        clf_mc.fit(X, y_mc)
        self.assertEqual(clf_mc.n_classes_, 3)
        self.assertEqual(clf_mc.classes_, [0, 1, 2])


class TestTwoClassesUsesBinaryPath(unittest.TestCase):
    """Labels {0,1} should use binary_crossentropy; predict_proba shape (n, 2)."""

    def test_two_classes_uses_binary_path(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(80, 3).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        clf = GBMClassifier(n_estimators=10, seed=42)
        clf.fit(X, y)

        self.assertEqual(clf._objective_name(), "binary_crossentropy")
        self.assertFalse(clf._is_multiclass)

        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestScoreReturnsAccuracyMulticlass(unittest.TestCase):
    """score() returns accuracy in [0, 1] for multi-class problems."""

    def test_score_returns_accuracy_multiclass(self) -> None:
        X_train, y_train, X_val, y_val = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=30, seed=42)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        self.assertGreaterEqual(train_score, 0.0)
        self.assertLessEqual(train_score, 1.0)
        # Verify score matches manual accuracy calculation
        preds = clf.predict(X_train)
        manual_acc = sum(
            1 for p, t in zip(preds, y_train) if p == int(t)
        ) / len(y_train)
        self.assertAlmostEqual(train_score, manual_acc, places=10)


class TestFiveClassProblem(unittest.TestCase):
    """Validate multi-class with K=5 classes."""

    def test_five_class_problem(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(
            n_train=300, n_features=5, n_classes=5, seed=42
        )
        clf = GBMClassifier(n_estimators=30, seed=42)
        clf.fit(X_train, y_train)

        self.assertEqual(clf.n_classes_, 5)
        preds = clf.predict(X_train)
        valid_labels = set(int(c) for c in np.unique(y_train))
        for p in preds:
            self.assertIn(p, valid_labels)

        proba = clf.predict_proba(X_train)
        self.assertEqual(proba.shape, (len(y_train), 5))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))


class TestPredictLogProbaMulticlass(unittest.TestCase):
    """predict_log_proba shape matches predict_proba; values are negative."""

    def test_predict_log_proba_multiclass(self) -> None:
        X_train, y_train, _, _ = _make_multiclass_dataset(n_classes=3)
        clf = GBMClassifier(n_estimators=20, seed=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_train)
        log_proba = clf.predict_log_proba(X_train)

        self.assertEqual(log_proba.shape, proba.shape)
        # Log of values in (0, 1] should be <= 0
        self.assertTrue(np.all(log_proba <= 0.0))
        # Verify consistency with predict_proba
        expected = np.log(np.clip(proba, 1e-15, None))
        np.testing.assert_allclose(log_proba, expected, atol=1e-10)


class TestMulticlassLogLossPerfectPredictions(unittest.TestCase):
    """Log loss near 0 for perfect one-hot predictions."""

    def test_multiclass_log_loss_perfect_predictions(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2]
        y_prob = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = multiclass_log_loss(y_true, y_prob)
        # With clipping at 1e-15, perfect predictions yield loss very close to 0
        self.assertAlmostEqual(result, 0.0, places=10)


class TestMulticlassLogLossErrorHandling(unittest.TestCase):
    """Wrong shapes raise ValueError."""

    def test_1d_y_prob_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_log_loss([0, 1, 2], [0.3, 0.4, 0.3])

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_log_loss(
                [0, 1],
                [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
            )

    def test_single_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_log_loss([0, 0], [[1.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
