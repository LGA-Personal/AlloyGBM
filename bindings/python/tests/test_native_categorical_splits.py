"""Tests for native categorical splits in AlloyGBM.

Native categorical splits allow the GBM to partition categories optimally at
each tree node, instead of using target encoding.  Key parameter:
``max_cat_threshold`` (int, default 0) controls the maximum cardinality for
native splits.  0 = disabled (all target encoding).
"""

from __future__ import annotations

import math
import pickle
import tempfile
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

_CATEGORIES_4 = ["alpha", "bravo", "charlie", "delta"]


def _make_cat_regression_data(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Dataset where a 4-category feature strongly determines y.

    Returns (X_numeric, y, cat_values) where X_numeric[:, 0] contains the
    integer-coded category (0-3) and cat_values is the list of string labels.
    """
    rng = np.random.RandomState(seed)
    cat_ids = np.arange(n) % 4
    cat_strings = [_CATEGORIES_4[i] for i in cat_ids]
    numeric = rng.randn(n).astype(np.float32)
    X = np.column_stack([cat_ids.astype(np.float32), numeric])
    # Target is strongly determined by category
    cat_effects = np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float32)
    y = cat_effects[cat_ids] + 0.1 * numeric + rng.randn(n).astype(np.float32) * 0.05
    return X, y, cat_strings


def _make_cat_classification_data(
    n: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Binary classification where category strongly determines label."""
    rng = np.random.RandomState(seed)
    cat_ids = np.arange(n) % 4
    cat_strings = [_CATEGORIES_4[i] for i in cat_ids]
    numeric = rng.randn(n).astype(np.float32)
    X = np.column_stack([cat_ids.astype(np.float32), numeric])
    # Categories 0,1 -> class 0; categories 2,3 -> class 1
    y = (cat_ids >= 2).astype(np.float32)
    return X, y, cat_strings


def _make_cat_ranking_data(
    n_queries: int = 10,
    docs_per_query: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Ranking data with a categorical feature."""
    rng = np.random.RandomState(seed)
    n = n_queries * docs_per_query
    group = np.repeat(np.arange(n_queries, dtype=np.uint32), docs_per_query)
    cat_ids = np.arange(n) % 4
    cat_strings = [_CATEGORIES_4[i] for i in cat_ids]
    numeric = rng.randn(n).astype(np.float32)
    X = np.column_stack([cat_ids.astype(np.float32), numeric])
    # Relevance depends on category
    base_rel = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = np.clip(base_rel[cat_ids] + rng.randn(n).astype(np.float32) * 0.3, 0, 4)
    return X, y.astype(np.float32), group, cat_strings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNativeCategoricalSplits(unittest.TestCase):
    """Test native categorical split support across estimators."""

    # 1. Basic regression with native categorical splits
    def test_native_cat_regression_basic(self) -> None:
        """GBMRegressor with max_cat_threshold=64 should fit and predict."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds = m.predict(X)
        self.assertEqual(len(preds), len(y))

        # Predictions should cluster by category:
        # categories 0,1 ~ 1.0 and categories 2,3 ~ 3.0
        preds_arr = np.array(preds, dtype=np.float32)
        cat_ids = (X[:, 0]).astype(int)
        mean_low = preds_arr[cat_ids < 2].mean()
        mean_high = preds_arr[cat_ids >= 2].mean()
        self.assertLess(mean_low, 2.0)
        self.assertGreater(mean_high, 2.0)
        self.assertGreater(mean_high - mean_low, 1.0)

    # 2. Native vs target encoding comparison
    def test_native_cat_regression_vs_target_encoding(self) -> None:
        """Native splits should be at least as good as target encoding."""
        X, y, cats = _make_cat_regression_data(n=400, seed=99)

        common_params = dict(
            n_estimators=15,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
            learning_rate=0.1,
        )

        # Native categorical splits
        m_native = GBMRegressor(max_cat_threshold=64, **common_params)
        m_native.fit(X, y, categorical_feature_values_list=[cats])
        preds_native = np.array(m_native.predict(X), dtype=np.float32)
        rmse_native = float(np.sqrt(np.mean((preds_native - y) ** 2)))

        # Target encoding (max_cat_threshold=0)
        m_target = GBMRegressor(max_cat_threshold=0, **common_params)
        m_target.fit(X, y, categorical_feature_values_list=[cats])
        preds_target = np.array(m_target.predict(X), dtype=np.float32)
        rmse_target = float(np.sqrt(np.mean((preds_target - y) ** 2)))

        # Native should be at least as good or within a small tolerance
        self.assertLessEqual(rmse_native, rmse_target + 0.5)

    # 3. Binary classification
    def test_native_cat_binary_classification(self) -> None:
        """GBMClassifier with native categorical splits should work."""
        X, y, cats = _make_cat_classification_data()
        clf = GBMClassifier(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        clf.fit(X, y, categorical_feature_values_list=[cats])

        preds = clf.predict(X)
        self.assertEqual(len(preds), len(y))
        # All predictions should be 0 or 1
        for p in preds:
            self.assertIn(p, (0, 1, 0.0, 1.0))

        proba = clf.predict_proba(X)
        self.assertEqual(len(proba), len(y))

    # 4. Multiclass (max_cat_threshold effectively not applied yet)
    def test_native_cat_multiclass(self) -> None:
        """Multiclass + categorical features should work without errors."""
        rng = np.random.RandomState(42)
        n = 150
        cat_ids = np.arange(n) % 3
        cat_strings = [["x", "y", "z"][i] for i in cat_ids]
        numeric = rng.randn(n).astype(np.float32)
        X = np.column_stack([cat_ids.astype(np.float32), numeric])
        y = cat_ids.astype(np.int64)  # 3 classes: 0, 1, 2

        clf = GBMClassifier(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        clf.fit(X, y, categorical_feature_values_list=[cat_strings])
        preds = clf.predict(X)
        self.assertEqual(len(preds), n)
        valid_labels = {0, 1, 2}
        for p in preds:
            self.assertIn(p, valid_labels)

    # 5. Ranking
    def test_native_cat_ranking(self) -> None:
        """GBMRanker with native categorical splits should work."""
        X, y, group, cats = _make_cat_ranking_data()
        ranker = GBMRanker(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        ranker.fit(
            X, y,
            group=group,
            categorical_feature_values_list=[cats],
        )
        preds = ranker.predict(X)
        self.assertEqual(len(preds), len(y))

    # 6. Early stopping with categorical features
    def test_native_cat_early_stopping(self) -> None:
        """Early stopping with categorical features and max_cat_threshold."""
        X, y, cats = _make_cat_regression_data(n=300, seed=77)
        # Split into train and validation
        X_train, X_val = X[:200], X[200:]
        y_train, y_val = y[:200], y[200:]
        cats_train, cats_val = cats[:200], cats[200:]

        m = GBMRegressor(
            n_estimators=100,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
            early_stopping_rounds=5,
        )
        m.fit(
            X_train, y_train,
            categorical_feature_values_list=[cats_train],
            eval_set=(X_val, y_val),
        )
        preds = m.predict(X_val)
        self.assertEqual(len(preds), len(y_val))
        # Early stopping should have triggered before 100 rounds
        self.assertLess(m.n_estimators_, 100)

    # 7. Warm start with categorical features
    def test_native_cat_warm_start(self) -> None:
        """warm_start=True with categorical features should work."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=5,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
            warm_start=True,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds_5 = m.predict(X[:5])
        self.assertEqual(m.n_estimators_, 5)

        # Continue training
        m.n_estimators = 10
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds_10 = m.predict(X[:5])
        self.assertEqual(m.n_estimators_, 10)
        # Predictions should change (more trees)
        self.assertFalse(
            all(abs(a - b) < 1e-9 for a, b in zip(preds_5, preds_10)),
            "Predictions should differ after warm-starting with more trees",
        )

    # 8. Multiple categorical features
    def test_native_cat_multiple_categorical_features(self) -> None:
        """Two categorical columns with different cardinalities."""
        rng = np.random.RandomState(42)
        n = 200
        cat1_ids = np.arange(n) % 3  # 3 categories
        cat2_ids = np.arange(n) % 5  # 5 categories
        cat1_strings = [["low", "mid", "high"][i] for i in cat1_ids]
        cat2_strings = [["a", "b", "c", "d", "e"][i] for i in cat2_ids]
        numeric = rng.randn(n).astype(np.float32)
        X = np.column_stack([
            cat1_ids.astype(np.float32),
            cat2_ids.astype(np.float32),
            numeric,
        ])
        effects1 = np.array([1.0, 2.0, 3.0])[cat1_ids]
        effects2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])[cat2_ids]
        y = (effects1 + effects2 + 0.1 * numeric).astype(np.float32)

        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0, 1],
            training_policy="manual",
            seed=42,
        )
        m.fit(
            X, y,
            categorical_feature_values_list=[cat1_strings, cat2_strings],
        )
        preds = m.predict(X)
        self.assertEqual(len(preds), n)
        rmse = float(np.sqrt(np.mean((np.array(preds) - y) ** 2)))
        # Should fit reasonably well
        self.assertLess(rmse, 2.0)

    # 9. Mixed continuous and categorical features
    def test_native_cat_mixed_continuous_and_categorical(self) -> None:
        """Mix of continuous and categorical features."""
        rng = np.random.RandomState(42)
        n = 200
        cat_ids = np.arange(n) % 4
        cat_strings = [_CATEGORIES_4[i] for i in cat_ids]
        x_cont1 = rng.randn(n).astype(np.float32)
        x_cont2 = rng.randn(n).astype(np.float32)
        X = np.column_stack([
            x_cont1,
            cat_ids.astype(np.float32),
            x_cont2,
        ])
        # Target depends on both continuous and categorical features
        cat_effect = np.array([0.0, 1.0, 2.0, 3.0])[cat_ids]
        y = (x_cont1 * 2.0 + cat_effect + x_cont2 * 0.5).astype(np.float32)

        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[1],  # column 1 is categorical
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cat_strings])
        preds = m.predict(X)
        self.assertEqual(len(preds), n)
        rmse = float(np.sqrt(np.mean((np.array(preds) - y) ** 2)))
        self.assertLess(rmse, 3.0)

    # 10. High cardinality fallback to target encoding
    def test_native_cat_high_cardinality_fallback(self) -> None:
        """Cardinality > max_cat_threshold falls back to target encoding."""
        rng = np.random.RandomState(42)
        n = 200
        # 20 unique categories, but max_cat_threshold=4
        cat_ids = np.arange(n) % 20
        cat_strings = [f"cat_{i}" for i in cat_ids]
        numeric = rng.randn(n).astype(np.float32)
        X = np.column_stack([cat_ids.astype(np.float32), numeric])
        y = (cat_ids.astype(np.float32) * 0.5 + numeric * 0.1).astype(np.float32)

        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=4,  # lower than cardinality of 20
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cat_strings])
        preds = m.predict(X)
        self.assertEqual(len(preds), n)

        # Feature 0 should NOT be in native_cat_mappings because its
        # cardinality (20) exceeds max_cat_threshold (4)
        mappings = m._native_cat_mappings_
        if mappings is not None:
            self.assertNotIn(0, mappings)

    # 11. max_cat_threshold=0 disables native categorical splits
    def test_native_cat_max_cat_threshold_zero_disables(self) -> None:
        """max_cat_threshold=0 should use target encoding only."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=0,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds = m.predict(X)
        self.assertEqual(len(preds), len(y))
        # No native cat mappings when threshold is 0
        self.assertIsNone(m._native_cat_mappings_)

    # 12. Unseen category at predict time
    def test_native_cat_unseen_category_at_predict(self) -> None:
        """Unknown category at predict time should not crash."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])

        # Predict with numeric X (as would happen with numpy) -- unseen
        # integer values that weren't in training should still work since
        # the underlying bins handle OOV gracefully.
        X_test = np.array([[99.0, 0.5]], dtype=np.float32)
        preds = m.predict(X_test)
        self.assertEqual(len(preds), 1)
        # Should produce a finite number
        self.assertTrue(math.isfinite(preds[0]))

    # 13. Single category (no possible split)
    def test_native_cat_single_category_no_split(self) -> None:
        """Feature with 1 unique category should train without errors."""
        rng = np.random.RandomState(42)
        n = 100
        cat_strings = ["only_one"] * n
        numeric = rng.randn(n).astype(np.float32)
        X = np.column_stack([np.zeros(n, dtype=np.float32), numeric])
        y = (numeric * 2.0).astype(np.float32)

        m = GBMRegressor(
            n_estimators=5,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cat_strings])
        preds = m.predict(X)
        self.assertEqual(len(preds), n)

    # 14. Pickle roundtrip
    def test_native_cat_pickle_roundtrip(self) -> None:
        """Pickle should preserve category mappings and predictions."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds_before = m.predict(X[:10])

        data = pickle.dumps(m)
        m2 = pickle.loads(data)

        preds_after = m2.predict(X[:10])
        np.testing.assert_allclose(preds_before, preds_after, rtol=0.0, atol=0.0)

        # Verify mappings are preserved
        self.assertEqual(m._native_cat_mappings_, m2._native_cat_mappings_)
        self.assertEqual(m2.max_cat_threshold, 64)

    # 15. save_model / load_model roundtrip
    def test_native_cat_save_load_model(self) -> None:
        """save_model/load_model should preserve everything."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])
        preds_before = m.predict(X[:10])

        with tempfile.NamedTemporaryFile(suffix=".agbm", delete=False) as f:
            path = f.name
        m.save_model(path)
        m2 = GBMRegressor.load_model(path)

        preds_after = m2.predict(X[:10])
        np.testing.assert_allclose(preds_before, preds_after, rtol=0.0, atol=0.0)

        # Verify mappings and params are preserved
        self.assertEqual(m._native_cat_mappings_, m2._native_cat_mappings_)
        self.assertEqual(m2.max_cat_threshold, 64)

    # 16. get_params / set_params with max_cat_threshold
    def test_native_cat_get_params_set_params(self) -> None:
        """sklearn API should handle max_cat_threshold correctly."""
        m = GBMRegressor(n_estimators=5, max_cat_threshold=32)
        params = m.get_params()
        self.assertEqual(params["max_cat_threshold"], 32)

        m.set_params(max_cat_threshold=128)
        self.assertEqual(m.max_cat_threshold, 128)
        self.assertEqual(m.get_params()["max_cat_threshold"], 128)

        # Clone via get_params
        m2 = GBMRegressor(**m.get_params())
        self.assertEqual(m2.max_cat_threshold, 128)

        # repr should include max_cat_threshold
        r = repr(m)
        self.assertIn("max_cat_threshold=128", r)

    # 17. DataFrame predict with string categorical columns
    def test_native_cat_dataframe_predict(self) -> None:
        """DataFrame with string categorical columns should auto-convert."""
        X, y, cats = _make_cat_regression_data()
        m = GBMRegressor(
            n_estimators=10,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )
        m.fit(X, y, categorical_feature_values_list=[cats])

        # Only test if pandas is available
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        # Build a DataFrame with string values in the categorical column
        df = pd.DataFrame({
            "cat_col": ["alpha", "bravo", "charlie", "delta"],
            "num": [0.0, 0.0, 0.0, 0.0],
        })
        preds = m.predict(df)
        self.assertEqual(len(preds), 4)

        # Predictions should cluster: alpha/bravo ~ low, charlie/delta ~ high
        self.assertLess(preds[0], 2.5)  # alpha
        self.assertLess(preds[1], 2.5)  # bravo
        self.assertGreater(preds[2], 1.5)  # charlie
        self.assertGreater(preds[3], 1.5)  # delta

    # 18. Both tree_growth modes work
    def test_native_cat_level_wise_and_leaf_wise(self) -> None:
        """Both tree_growth modes should work with native categorical splits."""
        X, y, cats = _make_cat_regression_data()
        common = dict(
            n_estimators=8,
            max_cat_threshold=64,
            categorical_feature_indices=[0],
            training_policy="manual",
            seed=42,
        )

        for growth_mode, max_leaves in [("level", None), ("leaf", 8)]:
            with self.subTest(tree_growth=growth_mode):
                kwargs = dict(**common, tree_growth=growth_mode)
                if max_leaves is not None:
                    kwargs["max_leaves"] = max_leaves
                m = GBMRegressor(**kwargs)
                m.fit(X, y, categorical_feature_values_list=[cats])
                preds = m.predict(X[:10])
                self.assertEqual(len(preds), 10)
                # Should produce finite predictions
                for p in preds:
                    self.assertTrue(math.isfinite(p), f"Non-finite prediction: {p}")


if __name__ == "__main__":
    unittest.main()
