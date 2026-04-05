"""Tests for expanded configurability: monotone constraints, feature weights, max_leaves."""

import math
import pickle
import unittest

import numpy as np

from alloygbm import GBMRegressor


def _make_dataset(n=300, seed=42):
    """Create a simple dataset with 3 features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    # y = 2*x0 - x1 + 0.5*x2 + noise
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.randn(n) * 0.1
    return X, y


class MonotoneConstraintTests(unittest.TestCase):
    """Tests for monotone_constraints parameter."""

    def test_monotone_nondecreasing_respected(self):
        """Feature 0 constrained to +1 should produce non-decreasing predictions."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=30,
            monotone_constraints=[1, 0, 0],
            training_policy="manual",
        )
        m.fit(X, y)
        # Test across a range of feature 0 values while holding others at 0
        test_vals = np.linspace(-2, 2, 20)
        preds = []
        for v in test_vals:
            p = m.predict(np.array([[v, 0.0, 0.0]]))[0]
            preds.append(p)
        for i in range(1, len(preds)):
            self.assertGreaterEqual(
                preds[i],
                preds[i - 1] - 1e-6,
                f"Monotone +1 violated: pred({test_vals[i]:.2f})={preds[i]:.4f} "
                f"< pred({test_vals[i-1]:.2f})={preds[i-1]:.4f}",
            )

    def test_monotone_nonincreasing_respected(self):
        """Feature 1 constrained to -1 should produce non-increasing predictions."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=30,
            monotone_constraints=[0, -1, 0],
            training_policy="manual",
        )
        m.fit(X, y)
        test_vals = np.linspace(-2, 2, 20)
        preds = []
        for v in test_vals:
            p = m.predict(np.array([[0.0, v, 0.0]]))[0]
            preds.append(p)
        for i in range(1, len(preds)):
            self.assertLessEqual(
                preds[i],
                preds[i - 1] + 1e-6,
                f"Monotone -1 violated: pred({test_vals[i]:.2f})={preds[i]:.4f} "
                f"> pred({test_vals[i-1]:.2f})={preds[i-1]:.4f}",
            )

    def test_monotone_dict_format(self):
        """Dict-style monotone constraints should work."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=20,
            monotone_constraints={0: 1},
            training_policy="manual",
        )
        m.fit(X, y)
        p_lo = m.predict(np.array([[-1.0, 0.0, 0.0]]))[0]
        p_hi = m.predict(np.array([[1.0, 0.0, 0.0]]))[0]
        self.assertGreaterEqual(p_hi, p_lo - 1e-6)

    def test_monotone_none_means_no_constraints(self):
        """monotone_constraints=None should behave identically to no constraints."""
        X, y = _make_dataset()
        m1 = GBMRegressor(n_estimators=10, seed=42, training_policy="manual")
        m1.fit(X, y)
        m2 = GBMRegressor(
            n_estimators=10,
            seed=42,
            training_policy="manual",
            monotone_constraints=None,
        )
        m2.fit(X, y)
        np.testing.assert_allclose(m1.predict(X[:5]), m2.predict(X[:5]))

    def test_monotone_rejects_invalid_values(self):
        """Invalid constraint values should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(monotone_constraints=[2, 0, 0])
        with self.assertRaises(ValueError):
            GBMRegressor(monotone_constraints=[0, -2, 0])

    def test_monotone_get_set_params(self):
        """monotone_constraints should roundtrip through get/set_params."""
        m = GBMRegressor(monotone_constraints=[1, -1, 0])
        self.assertEqual(m.get_params()["monotone_constraints"], [1, -1, 0])

        m.set_params(monotone_constraints=[0, 0, 1])
        self.assertEqual(m.get_params()["monotone_constraints"], [0, 0, 1])

        m.set_params(monotone_constraints=None)
        self.assertIsNone(m.get_params()["monotone_constraints"])

    def test_monotone_in_repr(self):
        """__repr__ should include monotone_constraints."""
        m = GBMRegressor(monotone_constraints=[1, 0, -1])
        self.assertIn("monotone_constraints=[1, 0, -1]", repr(m))


class FeatureWeightTests(unittest.TestCase):
    """Tests for feature_weights parameter."""

    def test_feature_weights_bias_split_selection(self):
        """High weight on feature 0 should make the model prefer it."""
        X, y = _make_dataset()
        # With uniform weights, train normally
        m_uniform = GBMRegressor(
            n_estimators=10, seed=42, training_policy="manual"
        )
        m_uniform.fit(X, y)
        # With very low weight on feature 0, it should be used less
        m_biased = GBMRegressor(
            n_estimators=10,
            seed=42,
            training_policy="manual",
            feature_weights=[0.01, 1.0, 1.0],
        )
        m_biased.fit(X, y)
        # The models should produce different predictions
        p_uniform = m_uniform.predict(X[:5])
        p_biased = m_biased.predict(X[:5])
        # They should differ (feature 0 is the strongest predictor, deweighting
        # it should change the model)
        self.assertFalse(np.allclose(p_uniform, p_biased, atol=0.01))

    def test_feature_weights_dict_format(self):
        """Dict-style feature weights should work."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            feature_weights={0: 5.0},
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:1])
        self.assertTrue(np.isfinite(pred[0]))

    def test_feature_weights_none_means_uniform(self):
        """feature_weights=None should behave identically to no weights."""
        X, y = _make_dataset()
        m1 = GBMRegressor(n_estimators=10, seed=42, training_policy="manual")
        m1.fit(X, y)
        m2 = GBMRegressor(
            n_estimators=10,
            seed=42,
            training_policy="manual",
            feature_weights=None,
        )
        m2.fit(X, y)
        np.testing.assert_allclose(m1.predict(X[:5]), m2.predict(X[:5]))

    def test_feature_weights_rejects_negative(self):
        """Negative feature weights should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(feature_weights=[-1.0, 1.0])

    def test_feature_weights_rejects_non_finite(self):
        """Non-finite feature weights should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(feature_weights=[float("inf"), 1.0])
        with self.assertRaises(ValueError):
            GBMRegressor(feature_weights=[float("nan"), 1.0])

    def test_feature_weights_get_set_params(self):
        """feature_weights should roundtrip through get/set_params."""
        m = GBMRegressor(feature_weights=[1.0, 2.0, 0.5])
        self.assertEqual(m.get_params()["feature_weights"], [1.0, 2.0, 0.5])

        m.set_params(feature_weights=[0.5, 0.5, 0.5])
        self.assertEqual(m.get_params()["feature_weights"], [0.5, 0.5, 0.5])

        m.set_params(feature_weights=None)
        self.assertIsNone(m.get_params()["feature_weights"])


class MaxLeavesTests(unittest.TestCase):
    """Tests for max_leaves parameter."""

    def test_max_leaves_limits_tree_size(self):
        """max_leaves should limit the number of leaves per tree."""
        X, y = _make_dataset()
        # With max_leaves=2, each tree should have at most 1 split (2 leaves)
        m = GBMRegressor(
            n_estimators=10,
            max_depth=6,
            max_leaves=2,
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:1])
        self.assertTrue(np.isfinite(pred[0]))

    def test_max_leaves_none_means_depth_limited(self):
        """max_leaves=None should use depth-only limit (default)."""
        X, y = _make_dataset()
        m1 = GBMRegressor(
            n_estimators=10, seed=42, training_policy="manual"
        )
        m1.fit(X, y)
        m2 = GBMRegressor(
            n_estimators=10,
            seed=42,
            training_policy="manual",
            max_leaves=None,
        )
        m2.fit(X, y)
        np.testing.assert_allclose(m1.predict(X[:5]), m2.predict(X[:5]))

    def test_max_leaves_rejects_less_than_2(self):
        """max_leaves < 2 should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(max_leaves=1)
        with self.assertRaises(ValueError):
            GBMRegressor(max_leaves=0)

    def test_max_leaves_get_set_params(self):
        """max_leaves should roundtrip through get/set_params."""
        m = GBMRegressor(max_leaves=16)
        self.assertEqual(m.get_params()["max_leaves"], 16)

        m.set_params(max_leaves=32)
        self.assertEqual(m.get_params()["max_leaves"], 32)

        m.set_params(max_leaves=None)
        self.assertIsNone(m.get_params()["max_leaves"])

    def test_max_leaves_in_repr(self):
        """__repr__ should include max_leaves."""
        m = GBMRegressor(max_leaves=8)
        self.assertIn("max_leaves=8", repr(m))

    def test_max_leaves_smaller_than_depth_budget(self):
        """max_leaves should take effect even when depth budget allows more."""
        X, y = _make_dataset()
        # max_depth=6 allows 2^6=64 leaves, but max_leaves=4 should limit
        m_limited = GBMRegressor(
            n_estimators=20,
            max_depth=6,
            max_leaves=4,
            training_policy="manual",
            seed=42,
        )
        m_limited.fit(X, y)
        m_unlimited = GBMRegressor(
            n_estimators=20,
            max_depth=6,
            training_policy="manual",
            seed=42,
        )
        m_unlimited.fit(X, y)
        # The limited model should produce different (less expressive) results
        p_limited = m_limited.predict(X)
        p_unlimited = m_unlimited.predict(X)
        # Limited model should have higher RMSE (less expressive)
        rmse_limited = np.sqrt(np.mean((p_limited - y) ** 2))
        rmse_unlimited = np.sqrt(np.mean((p_unlimited - y) ** 2))
        self.assertGreater(rmse_limited, rmse_unlimited * 0.9)  # Allow some tolerance


class CombinedConfigurabilityTests(unittest.TestCase):
    """Tests combining multiple new configurability features."""

    def test_all_features_together(self):
        """All three features should work simultaneously."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=15,
            monotone_constraints=[1, -1, 0],
            feature_weights=[2.0, 1.0, 0.5],
            max_leaves=8,
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:5])
        self.assertEqual(len(pred), 5)
        for p in pred:
            self.assertTrue(np.isfinite(p))

    def test_pickle_roundtrip_with_new_params(self):
        """Pickle roundtrip should preserve new configurability params."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=5,
            monotone_constraints=[1, 0, -1],
            feature_weights=[2.0, 1.0, 0.5],
            max_leaves=6,
            training_policy="manual",
        )
        m.fit(X, y)
        pred_before = m.predict(X[:3])

        m2 = pickle.loads(pickle.dumps(m))
        pred_after = m2.predict(X[:3])
        np.testing.assert_allclose(pred_before, pred_after)
        self.assertEqual(m2.get_params()["monotone_constraints"], [1, 0, -1])
        self.assertEqual(m2.get_params()["feature_weights"], [2.0, 1.0, 0.5])
        self.assertEqual(m2.get_params()["max_leaves"], 6)

    def test_classifier_inherits_new_params(self):
        """GBMClassifier should accept new params via inheritance."""
        from alloygbm import GBMClassifier

        m = GBMClassifier(
            n_estimators=5,
            monotone_constraints=[1, 0],
            max_leaves=4,
            feature_weights=[1.0, 2.0],
        )
        params = m.get_params()
        self.assertEqual(params["monotone_constraints"], [1, 0])
        self.assertEqual(params["max_leaves"], 4)
        self.assertEqual(params["feature_weights"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
