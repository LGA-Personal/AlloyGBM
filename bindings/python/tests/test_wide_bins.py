"""Tests for u16 adaptive bin storage (bin cap increase beyond 256)."""

import pickle
import unittest

import numpy as np

from alloygbm import GBMRegressor


def _make_dataset(n=300, seed=42, n_features=3):
    """Create a simple dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = 2 * X[:, 0] - X[:, 1] + rng.randn(n) * 0.1
    return X, y


class WideBinBasicTests(unittest.TestCase):
    """Tests for max_bins > 256 (u16 bin path)."""

    def test_wide_bins_train_and_predict(self):
        """Training with max_bins=512 should produce valid predictions."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            continuous_binning_max_bins=512,
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:5])
        self.assertEqual(len(pred), 5)
        for p in pred:
            self.assertTrue(np.isfinite(p))

    def test_wide_bins_1024(self):
        """Training with max_bins=1024 should work."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=5,
            continuous_binning_max_bins=1024,
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:3])
        self.assertEqual(len(pred), 3)
        for p in pred:
            self.assertTrue(np.isfinite(p))

    def test_wide_bins_reduces_rmse_vs_few_bins(self):
        """More bins should give equal or lower RMSE than fewer bins."""
        X, y = _make_dataset(n=500)
        m_few = GBMRegressor(
            n_estimators=20,
            continuous_binning_max_bins=8,
            training_policy="manual",
            seed=42,
        )
        m_few.fit(X, y)
        m_wide = GBMRegressor(
            n_estimators=20,
            continuous_binning_max_bins=512,
            training_policy="manual",
            seed=42,
        )
        m_wide.fit(X, y)
        rmse_few = np.sqrt(np.mean((m_few.predict(X) - y) ** 2))
        rmse_wide = np.sqrt(np.mean((m_wide.predict(X) - y) ** 2))
        # Wide bins should give equal or better fit
        self.assertLessEqual(rmse_wide, rmse_few * 1.05)

    def test_default_256_unchanged(self):
        """Default max_bins=256 should use u8 path and produce same results."""
        X, y = _make_dataset()
        m1 = GBMRegressor(n_estimators=10, seed=42, training_policy="manual")
        m1.fit(X, y)
        m2 = GBMRegressor(
            n_estimators=10,
            seed=42,
            training_policy="manual",
            continuous_binning_max_bins=256,
        )
        m2.fit(X, y)
        np.testing.assert_allclose(m1.predict(X[:5]), m2.predict(X[:5]))


class WideBinNaNTests(unittest.TestCase):
    """Tests for NaN handling with wide bins."""

    def test_wide_bins_with_nan_values(self):
        """NaN values should be handled correctly in u16 mode."""
        X, y = _make_dataset()
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        X_nan[5, 1] = np.nan
        m = GBMRegressor(
            n_estimators=10,
            continuous_binning_max_bins=512,
            training_policy="manual",
        )
        m.fit(X_nan, y)
        pred = m.predict(X[:5])
        self.assertEqual(len(pred), 5)
        for p in pred:
            self.assertTrue(np.isfinite(p))

    def test_wide_bins_predict_with_nan(self):
        """Prediction with NaN inputs should work in u16 mode."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=10,
            continuous_binning_max_bins=512,
            training_policy="manual",
        )
        m.fit(X, y)
        X_test = np.array([[0.0, np.nan, 0.0]])
        pred = m.predict(X_test)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))


class WideBinPickleTests(unittest.TestCase):
    """Tests for pickle roundtrip with wide bins."""

    def test_pickle_preserves_wide_bin_model(self):
        """Pickle roundtrip should preserve model trained with wide bins."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=5,
            continuous_binning_max_bins=512,
            training_policy="manual",
        )
        m.fit(X, y)
        pred_before = m.predict(X[:3])

        m2 = pickle.loads(pickle.dumps(m))
        pred_after = m2.predict(X[:3])
        np.testing.assert_allclose(pred_before, pred_after)
        self.assertEqual(m2.get_params()["continuous_binning_max_bins"], 512)


class WideBinValidationTests(unittest.TestCase):
    """Tests for parameter validation of max_bins."""

    def test_rejects_above_65535(self):
        """max_bins > 65535 should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(continuous_binning_max_bins=65536)

    def test_rejects_below_2(self):
        """max_bins < 2 should raise ValueError."""
        with self.assertRaises(ValueError):
            GBMRegressor(continuous_binning_max_bins=1)

    def test_accepts_boundary_values(self):
        """Boundary values should be accepted."""
        m = GBMRegressor(continuous_binning_max_bins=2)
        self.assertEqual(m.get_params()["continuous_binning_max_bins"], 2)
        m = GBMRegressor(continuous_binning_max_bins=65535)
        self.assertEqual(m.get_params()["continuous_binning_max_bins"], 65535)

    def test_wide_bins_in_repr(self):
        """Non-default max_bins should appear in repr."""
        m = GBMRegressor(continuous_binning_max_bins=512)
        self.assertIn("continuous_binning_max_bins=512", repr(m))

    def test_get_set_params(self):
        """continuous_binning_max_bins should roundtrip through get/set_params."""
        m = GBMRegressor(continuous_binning_max_bins=512)
        self.assertEqual(m.get_params()["continuous_binning_max_bins"], 512)
        m.set_params(continuous_binning_max_bins=1024)
        self.assertEqual(m.get_params()["continuous_binning_max_bins"], 1024)


class WideBinStrategyTests(unittest.TestCase):
    """Tests for wide bins with different binning strategies."""

    def test_wide_bins_rank_strategy(self):
        """Wide bins with rank strategy should work."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=5,
            continuous_binning_max_bins=512,
            continuous_binning_strategy="rank",
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:3])
        self.assertEqual(len(pred), 3)
        for p in pred:
            self.assertTrue(np.isfinite(p))

    def test_wide_bins_quantile_strategy(self):
        """Wide bins with quantile strategy should work."""
        X, y = _make_dataset()
        m = GBMRegressor(
            n_estimators=5,
            continuous_binning_max_bins=512,
            continuous_binning_strategy="quantile",
            training_policy="manual",
        )
        m.fit(X, y)
        pred = m.predict(X[:3])
        self.assertEqual(len(pred), 3)
        for p in pred:
            self.assertTrue(np.isfinite(p))


if __name__ == "__main__":
    unittest.main()
