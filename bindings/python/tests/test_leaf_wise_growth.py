"""Tests for leaf-wise (best-first) tree growth strategy."""

import unittest

import numpy as np

from alloygbm import GBMRegressor


def _make_balanced_dataset(n=200, seed=42):
    """Produce a simple dataset where features map linearly to target."""
    import random

    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)]
    y = [row[0] * 2.0 + row[1] * 0.5 + rng.gauss(0, 0.1) for row in X]
    return X, y


def _make_unbalanced_dataset(n=300, seed=99):
    """Dataset with one dominant feature and a few weak interaction features.

    Leaf-wise should focus most splits on the dominant feature early,
    while level-wise would distribute splits evenly across depth.
    """
    import random

    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(n)]
    y = [
        10.0 * row[0] + 0.01 * row[1] + 0.01 * row[2] + rng.gauss(0, 0.5)
        for row in X
    ]
    return X, y


class TestLeafWiseGrowthBasic(unittest.TestCase):
    """Basic functionality tests for tree_growth='leaf'."""

    def test_leaf_wise_trains_and_predicts(self):
        """Leaf-wise growth should produce a fitted model that can predict."""
        X, y = _make_balanced_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=6,
            max_leaves=8,
            tree_growth="leaf",
            training_policy="manual",
        )
        m.fit(X, y)
        preds = m.predict(X[:5])
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_leaf_wise_default_is_level(self):
        """Default tree_growth should be 'level'."""
        m = GBMRegressor(n_estimators=3)
        self.assertEqual(m.tree_growth, "level")

    def test_invalid_tree_growth_raises(self):
        """Invalid tree_growth string should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(n_estimators=3, tree_growth="invalid")

    def test_leaf_without_max_leaves_raises(self):
        """tree_growth='leaf' without max_leaves should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(n_estimators=3, tree_growth="leaf")

    def test_level_without_max_leaves_ok(self):
        """tree_growth='level' without max_leaves should work fine."""
        m = GBMRegressor(n_estimators=3, tree_growth="level")
        self.assertIsNone(m.max_leaves)


class TestLeafWiseGrowthQuality(unittest.TestCase):
    """Quality and equivalence tests for leaf-wise growth."""

    def test_leaf_wise_produces_reasonable_rmse(self):
        """Leaf-wise model should achieve reasonable predictive quality."""
        X, y = _make_balanced_dataset(n=300)
        m = GBMRegressor(
            n_estimators=50,
            max_depth=6,
            max_leaves=16,
            tree_growth="leaf",
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y)
        preds = m.predict(X)
        mse = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)
        rmse = mse**0.5
        # With 50 rounds and a simple linear target, RMSE should be well under 1.0
        self.assertLess(rmse, 1.0, f"Leaf-wise RMSE {rmse:.4f} too high")

    def test_leaf_wise_vs_level_wise_comparable(self):
        """On balanced data, both strategies should achieve similar quality."""
        X, y = _make_balanced_dataset(n=200, seed=77)
        common_params = dict(
            n_estimators=30,
            max_depth=4,
            learning_rate=0.1,
            training_policy="manual",
            seed=42,
        )

        m_level = GBMRegressor(**common_params, tree_growth="level")
        m_level.fit(X, y)
        preds_level = m_level.predict(X)
        mse_level = sum((p - t) ** 2 for p, t in zip(preds_level, y)) / len(y)

        m_leaf = GBMRegressor(
            **common_params, tree_growth="leaf", max_leaves=16
        )
        m_leaf.fit(X, y)
        preds_leaf = m_leaf.predict(X)
        mse_leaf = sum((p - t) ** 2 for p, t in zip(preds_leaf, y)) / len(y)

        # Both should be reasonable (neither should be catastrophically worse)
        self.assertLess(mse_level**0.5, 2.0)
        self.assertLess(mse_leaf**0.5, 2.0)

    def test_max_leaves_constrains_tree_size(self):
        """With small max_leaves, leaf-wise should still train without error."""
        X, y = _make_balanced_dataset()
        m = GBMRegressor(
            n_estimators=5,
            max_depth=10,
            max_leaves=3,
            tree_growth="leaf",
            training_policy="manual",
        )
        m.fit(X, y)
        preds = m.predict(X[:3])
        self.assertEqual(len(preds), 3)

    def test_leaf_wise_with_unbalanced_data(self):
        """Leaf-wise should handle datasets with one dominant feature."""
        X, y = _make_unbalanced_dataset()
        m = GBMRegressor(
            n_estimators=30,
            max_depth=8,
            max_leaves=16,
            tree_growth="leaf",
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y)
        preds = m.predict(X)
        mse = sum((p - t) ** 2 for p, t in zip(preds, y)) / len(y)
        rmse = mse**0.5
        self.assertLess(rmse, 3.0, f"Leaf-wise RMSE on unbalanced data {rmse:.4f} too high")


class TestLeafWiseGrowthParams(unittest.TestCase):
    """Parameter handling tests for tree_growth."""

    def test_get_params_includes_tree_growth(self):
        """get_params() should include tree_growth."""
        m = GBMRegressor(n_estimators=3, tree_growth="leaf", max_leaves=4)
        params = m.get_params()
        self.assertEqual(params["tree_growth"], "leaf")

    def test_set_params_tree_growth(self):
        """set_params() should update tree_growth."""
        m = GBMRegressor(n_estimators=3, max_leaves=4)
        m.set_params(tree_growth="leaf")
        self.assertEqual(m.tree_growth, "leaf")

    def test_set_params_invalid_tree_growth_raises(self):
        """set_params() with invalid tree_growth should raise."""
        m = GBMRegressor(n_estimators=3)
        with self.assertRaises(ValueError):
            m.set_params(tree_growth="bestfirst")

    def test_set_params_leaf_without_max_leaves_raises(self):
        """set_params(tree_growth='leaf') without max_leaves should raise."""
        m = GBMRegressor(n_estimators=3)
        with self.assertRaises(ValueError):
            m.set_params(tree_growth="leaf")

    def test_repr_includes_tree_growth(self):
        """__repr__ should include tree_growth."""
        m = GBMRegressor(n_estimators=3, tree_growth="leaf", max_leaves=8)
        r = repr(m)
        self.assertIn("tree_growth='leaf'", r)

    def test_clone_preserves_tree_growth(self):
        """Cloning via get_params/set_params roundtrip should preserve tree_growth."""
        m1 = GBMRegressor(
            n_estimators=3, tree_growth="leaf", max_leaves=8
        )
        m2 = GBMRegressor(**m1.get_params())
        self.assertEqual(m2.tree_growth, "leaf")
        self.assertEqual(m2.max_leaves, 8)


class TestLeafWiseGrowthWithConstraints(unittest.TestCase):
    """Test leaf-wise growth with other features like monotone constraints."""

    def test_leaf_wise_with_monotone_constraints(self):
        """Leaf-wise growth should work with monotone constraints."""
        X, y = _make_balanced_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=6,
            max_leaves=8,
            tree_growth="leaf",
            training_policy="manual",
            monotone_constraints={0: 1},  # Feature 0 monotonically increasing
        )
        m.fit(X, y)
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_leaf_wise_with_regularization(self):
        """Leaf-wise growth should work with L1/L2 regularization."""
        X, y = _make_balanced_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=6,
            max_leaves=8,
            tree_growth="leaf",
            training_policy="manual",
            lambda_l1=0.1,
            lambda_l2=0.1,
        )
        m.fit(X, y)
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_leaf_wise_with_subsampling(self):
        """Leaf-wise growth should work with row/col subsampling."""
        X, y = _make_balanced_dataset(n=200)
        m = GBMRegressor(
            n_estimators=10,
            max_depth=6,
            max_leaves=8,
            tree_growth="leaf",
            training_policy="manual",
            row_subsample=0.8,
            col_subsample=0.8,
            seed=42,
        )
        m.fit(X, y)
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_leaf_wise_with_early_stopping(self):
        """Leaf-wise growth should work with early stopping."""
        X, y = _make_balanced_dataset(n=200)
        split = 150
        m = GBMRegressor(
            n_estimators=100,
            max_depth=6,
            max_leaves=8,
            tree_growth="leaf",
            training_policy="manual",
            early_stopping_rounds=5,
        )
        m.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
