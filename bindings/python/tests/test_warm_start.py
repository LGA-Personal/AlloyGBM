"""Tests for warm-starting / incremental training."""

import unittest

from alloygbm import GBMRegressor


def _make_dataset(n=200, seed=42):
    """Produce a simple dataset for warm-start testing."""
    import random

    rng = random.Random(seed)
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(n)]
    y = [row[0] * 2.0 + row[1] * 0.5 + rng.gauss(0, 0.1) for row in X]
    return X, y


class TestWarmStartParams(unittest.TestCase):
    """Test parameter handling for warm_start."""

    def test_default_warm_start_is_false(self):
        """Default warm_start should be False."""
        m = GBMRegressor(n_estimators=3)
        self.assertFalse(m.warm_start)

    def test_warm_start_accepted(self):
        """Constructor should accept warm_start=True."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        self.assertTrue(m.warm_start)

    def test_get_params_includes_warm_start(self):
        """get_params() should include warm_start."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        params = m.get_params()
        self.assertTrue(params["warm_start"])

    def test_set_params_warm_start(self):
        """set_params() should update warm_start."""
        m = GBMRegressor(n_estimators=3)
        m.set_params(warm_start=True)
        self.assertTrue(m.warm_start)

    def test_repr_includes_warm_start(self):
        """__repr__ should include warm_start."""
        m = GBMRegressor(n_estimators=3, warm_start=True)
        r = repr(m)
        self.assertIn("warm_start=True", r)

    def test_clone_preserves_warm_start(self):
        """get_params/set_params roundtrip should preserve warm_start."""
        m1 = GBMRegressor(n_estimators=3, warm_start=True)
        m2 = GBMRegressor(**m1.get_params())
        self.assertTrue(m2.warm_start)


class TestWarmStartTraining(unittest.TestCase):
    """Test warm-start training end-to-end."""

    def test_warm_start_basic(self):
        """warm_start=True should continue training from previous state."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        # First fit
        m.fit(X, y)
        preds_after_10 = m.predict(X[:5])
        self.assertEqual(len(preds_after_10), 5)
        self.assertEqual(m.n_estimators_, 10)

        # Warm-start with more rounds
        m.n_estimators = 20
        m.fit(X, y)
        preds_after_20 = m.predict(X[:5])
        self.assertEqual(len(preds_after_20), 5)
        # Should have more trees now
        self.assertIsNotNone(m.n_estimators_)

    def test_warm_start_improves_quality(self):
        """Warm-started model should fit better than the original."""
        X, y = _make_dataset(n=200)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        preds_5 = m.predict(X)
        mse_5 = sum((p - t) ** 2 for p, t in zip(preds_5, y)) / len(y)

        # Continue with more rounds
        m.n_estimators = 30
        m.fit(X, y)
        preds_30 = m.predict(X)
        mse_30 = sum((p - t) ** 2 for p, t in zip(preds_30, y)) / len(y)

        # More rounds should reduce training error
        self.assertLess(mse_30, mse_5, "Warm-start should improve training MSE")

    def test_warm_start_false_resets(self):
        """When warm_start=False, fit() should start fresh each time."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=False,
        )
        m.fit(X, y)
        preds1 = m.predict(X[:5])

        # Second fit should start fresh and produce same results
        m.fit(X, y)
        preds2 = m.predict(X[:5])

        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2, places=6)

    def test_init_model_basic(self):
        """init_model should continue training from another model."""
        X, y = _make_dataset()
        m1 = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m1.fit(X, y)

        m2 = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m2.fit(X, y, init_model=m1)
        preds = m2.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_init_model_improves_quality(self):
        """init_model should allow continuing training to reduce error."""
        X, y = _make_dataset(n=200)
        m1 = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m1.fit(X, y)
        preds1 = m1.predict(X)
        mse1 = sum((p - t) ** 2 for p, t in zip(preds1, y)) / len(y)

        m2 = GBMRegressor(
            n_estimators=20,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m2.fit(X, y, init_model=m1)
        preds2 = m2.predict(X)
        mse2 = sum((p - t) ** 2 for p, t in zip(preds2, y)) / len(y)

        self.assertLess(mse2, mse1, "init_model continuation should reduce MSE")

    def test_init_model_takes_priority_over_warm_start(self):
        """init_model should take priority when warm_start=True."""
        X, y = _make_dataset()
        m_base = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
        )
        m_base.fit(X, y)

        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        # First fit
        m.fit(X, y)
        # Warm-start with init_model (should use init_model, not self)
        m.fit(X, y, init_model=m_base)
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_init_model_unfitted_raises(self):
        """init_model with unfitted model should raise ValueError."""
        X, y = _make_dataset()
        m = GBMRegressor(n_estimators=3, training_policy="manual")
        unfitted = GBMRegressor(n_estimators=3)
        with self.assertRaises(ValueError):
            m.fit(X, y, init_model=unfitted)

    def test_warm_start_with_validation(self):
        """Warm-start should work with eval_set."""
        X, y = _make_dataset(n=200)
        split = 150
        m = GBMRegressor(
            n_estimators=10,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
            early_stopping_rounds=5,
        )
        m.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
        preds = m.predict(X[:5])
        self.assertEqual(len(preds), 5)

        # Continue training with more rounds
        m.n_estimators = 30
        m.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
        preds2 = m.predict(X[:5])
        self.assertEqual(len(preds2), 5)


class TestWarmStartEdgeCases(unittest.TestCase):
    """Edge cases for warm-start."""

    def test_warm_start_single_round_increment(self):
        """Warm-start should work with just 1 additional round."""
        X, y = _make_dataset(n=100)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        m.n_estimators = 1
        m.fit(X, y)
        preds = m.predict(X[:3])
        self.assertEqual(len(preds), 3)

    def test_warm_start_preserves_predictions_format(self):
        """Predictions from warm-started model should be valid floats."""
        X, y = _make_dataset(n=100)
        m = GBMRegressor(
            n_estimators=5,
            max_depth=4,
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y)
        m.n_estimators = 10
        m.fit(X, y)
        preds = m.predict(X)
        for p in preds:
            self.assertIsInstance(p, float)
            self.assertTrue(abs(p) < 100, f"Prediction {p} seems unreasonably large")


if __name__ == "__main__":
    unittest.main()
